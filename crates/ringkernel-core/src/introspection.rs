//! Per-Actor Introspection API — FR-015
//!
//! Runtime actor inspection for debugging and monitoring:
//! - List all actors with state, queue depth, message rate
//! - Inspect per-actor metrics (read-only)
//! - Peek at queued messages
//! - Trace recent message processing with timing
//!
//! # Streaming Introspection (v1.1)
//!
//! Live introspection streaming for sub-millisecond metric freshness:
//! - [`IntrospectionStream`] — subscribe to periodic metric emissions per-actor
//! - [`MetricAggregator`] — EWMA rate calculation and latency histogram
//! - [`LiveMetrics`] — the observation record emitted to subscribers
//! - Wire formats ([`SubscribeMetricsRequest`], [`LiveMetricsEvent`]) for GPU/CPU bridge

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use tokio::sync::mpsc;

use crate::actor::{ActorId, ActorState};
use crate::hlc::HlcTimestamp;

/// Snapshot of a single actor's state for introspection.
#[derive(Debug, Clone)]
pub struct ActorSnapshot {
    /// Actor identifier.
    pub id: ActorId,
    /// Human-readable name (from registry, if registered).
    pub name: Option<String>,
    /// Current state.
    pub state: ActorState,
    /// Parent actor (if supervised).
    pub parent: Option<ActorId>,
    /// Children (if any).
    pub children: Vec<ActorId>,
    /// Queue metrics.
    pub queue: QueueSnapshot,
    /// Performance metrics.
    pub performance: PerformanceSnapshot,
    /// When this snapshot was taken.
    pub snapshot_at: Instant,
}

/// Snapshot of an actor's queue state.
#[derive(Debug, Clone, Default)]
pub struct QueueSnapshot {
    /// Current input queue depth.
    pub input_depth: u32,
    /// Current output queue depth.
    pub output_depth: u32,
    /// Input queue capacity.
    pub input_capacity: u32,
    /// Output queue capacity.
    pub output_capacity: u32,
    /// Queue pressure (0-255).
    pub pressure: u8,
}

impl QueueSnapshot {
    /// Input queue utilization (0.0 - 1.0).
    pub fn input_utilization(&self) -> f64 {
        if self.input_capacity == 0 {
            return 0.0;
        }
        self.input_depth as f64 / self.input_capacity as f64
    }
}

/// Snapshot of an actor's performance metrics.
#[derive(Debug, Clone, Default)]
pub struct PerformanceSnapshot {
    /// Total messages processed (lifetime).
    pub messages_processed: u64,
    /// Messages processed per second (recent window).
    pub messages_per_second: f64,
    /// Average processing latency per message.
    pub avg_latency: Duration,
    /// Maximum processing latency observed.
    pub max_latency: Duration,
    /// Number of restarts.
    pub restart_count: u32,
    /// Uptime since last restart.
    pub uptime: Duration,
}

/// A recent message processing trace entry.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    /// Message sequence number.
    pub sequence: u64,
    /// When the message was received.
    pub received_at: Instant,
    /// Processing duration.
    pub duration: Duration,
    /// Source actor (if K2K).
    pub source: Option<ActorId>,
    /// Outcome.
    pub outcome: TraceOutcome,
}

/// Outcome of a traced message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceOutcome {
    /// Successfully processed.
    Success,
    /// Processing failed.
    Failed(String),
    /// Message was forwarded to another actor.
    Forwarded(ActorId),
    /// Message was dropped (e.g., filtered).
    Dropped,
}

/// Per-actor trace buffer (ring buffer of recent processing traces).
pub struct TraceBuffer {
    entries: Vec<TraceEntry>,
    capacity: usize,
    write_pos: usize,
    total: u64,
}

impl TraceBuffer {
    /// Create a new trace buffer.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            write_pos: 0,
            total: 0,
        }
    }

    /// Record a trace entry.
    pub fn record(&mut self, entry: TraceEntry) {
        if self.entries.len() < self.capacity {
            self.entries.push(entry);
        } else {
            self.entries[self.write_pos] = entry;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.total += 1;
    }

    /// Get recent trace entries (most recent first).
    pub fn recent(&self, limit: usize) -> Vec<&TraceEntry> {
        let mut result: Vec<&TraceEntry> = self.entries.iter().collect();
        result.sort_by_key(|e| std::cmp::Reverse(e.received_at));
        result.truncate(limit);
        result
    }

    /// Total entries recorded (lifetime).
    pub fn total_recorded(&self) -> u64 {
        self.total
    }

    /// Current buffer size.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Introspection service that aggregates data from all actors.
pub struct IntrospectionService {
    /// Per-actor trace buffers.
    traces: HashMap<ActorId, TraceBuffer>,
    /// Default trace buffer capacity.
    trace_capacity: usize,
}

impl IntrospectionService {
    /// Create a new introspection service.
    pub fn new(trace_capacity: usize) -> Self {
        Self {
            traces: HashMap::new(),
            trace_capacity,
        }
    }

    /// Register an actor for tracing.
    pub fn register_actor(&mut self, id: ActorId) {
        self.traces
            .entry(id)
            .or_insert_with(|| TraceBuffer::new(self.trace_capacity));
    }

    /// Record a trace entry for an actor.
    pub fn record_trace(&mut self, actor: ActorId, entry: TraceEntry) {
        self.traces
            .entry(actor)
            .or_insert_with(|| TraceBuffer::new(self.trace_capacity))
            .record(entry);
    }

    /// Get recent traces for an actor.
    pub fn get_traces(&self, actor: ActorId, limit: usize) -> Vec<&TraceEntry> {
        self.traces
            .get(&actor)
            .map(|buf| buf.recent(limit))
            .unwrap_or_default()
    }

    /// Deregister an actor.
    pub fn deregister_actor(&mut self, id: ActorId) {
        self.traces.remove(&id);
    }

    /// Number of traced actors.
    pub fn actor_count(&self) -> usize {
        self.traces.len()
    }
}

impl Default for IntrospectionService {
    fn default() -> Self {
        Self::new(100)
    }
}

// =============================================================================
// Streaming Introspection (v1.1 spec §3.2)
// =============================================================================

/// Default EWMA smoothing factor for rate calculations.
///
/// Alpha = 0.2 gives moderate smoothing (newest sample weighted 20%, history 80%).
pub const DEFAULT_EWMA_ALPHA: f64 = 0.2;

/// Capacity of the [`LatencyHistogram`] ring buffer.
pub const LATENCY_HISTOGRAM_CAPACITY: usize = 1024;

/// Live per-actor metrics delivered to subscribers.
///
/// A `LiveMetrics` observation represents a point-in-time snapshot of an
/// actor's operational state, emitted periodically (per the subscriber's
/// configured interval) by a backend or aggregator.
#[derive(Debug, Clone)]
pub struct LiveMetrics {
    /// Actor whose metrics are being reported.
    pub actor_id: ActorId,
    /// HLC timestamp for this observation (causal ordering across nodes).
    pub timestamp: HlcTimestamp,
    /// Current inbound queue depth.
    pub queue_depth: usize,
    /// Inbound message rate (messages/sec), EWMA-smoothed.
    pub inbound_rate: f64,
    /// Outbound message rate (messages/sec), EWMA-smoothed.
    pub outbound_rate: f64,
    /// Observed p50 processing latency.
    pub latency_p50: Duration,
    /// Observed p99 processing latency.
    pub latency_p99: Duration,
    /// Resident actor state size in bytes.
    pub state_size_bytes: u64,
    /// GPU utilization (0.0–1.0). Always 0.0 on the CPU backend.
    pub gpu_utilization: f32,
    /// Tenant this actor belongs to (0 = unspecified).
    pub tenant_id: u64,
}

/// A handle to a single live-metrics subscription.
///
/// Wraps the sender half of a tokio unbounded MPSC channel along with the
/// subscriber's desired emission interval. The interval is stored so the
/// [`IntrospectionStream`] can honor per-subscriber cadence when dispatching.
pub struct SubscriberHandle {
    /// Desired emission interval for this subscriber.
    pub interval: Duration,
    /// Last time we forwarded a metric to this subscriber.
    last_sent_at: parking_lot::Mutex<Option<Instant>>,
    /// Unique subscription identifier (for wire-protocol correlation and teardown).
    pub subscription_id: u64,
    sender: mpsc::UnboundedSender<LiveMetrics>,
}

impl SubscriberHandle {
    /// Subscription identifier.
    pub fn subscription_id(&self) -> u64 {
        self.subscription_id
    }

    /// Return true if the receiver has been dropped.
    pub fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }

    /// Try to deliver a metric, respecting the subscriber's interval gate.
    ///
    /// Returns `Ok(true)` if delivered, `Ok(false)` if suppressed by the
    /// interval gate, and `Err(())` if the receiver has dropped (the caller
    /// should discard the handle).
    fn try_send(&self, metrics: LiveMetrics) -> std::result::Result<bool, ()> {
        let now = Instant::now();
        {
            let mut last = self.last_sent_at.lock();
            if let Some(prev) = *last {
                if now.duration_since(prev) < self.interval {
                    return Ok(false);
                }
            }
            *last = Some(now);
        }
        self.sender.send(metrics).map(|_| true).map_err(|_| ())
    }
}

/// A ring-buffer latency histogram for p50/p99 percentile computation.
///
/// Stores up to [`LATENCY_HISTOGRAM_CAPACITY`] recent samples. Percentiles
/// are computed by copying and sorting the live samples on demand (this is
/// intentional — HDR histogram is overkill for v1.1).
#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    samples: [Duration; LATENCY_HISTOGRAM_CAPACITY],
    idx: usize,
    count: u64,
}

impl LatencyHistogram {
    /// Create an empty histogram.
    pub fn new() -> Self {
        Self {
            samples: [Duration::ZERO; LATENCY_HISTOGRAM_CAPACITY],
            idx: 0,
            count: 0,
        }
    }

    /// Record a latency sample.
    pub fn record(&mut self, d: Duration) {
        self.samples[self.idx] = d;
        self.idx = (self.idx + 1) % LATENCY_HISTOGRAM_CAPACITY;
        self.count = self.count.saturating_add(1);
    }

    /// Number of samples recorded (lifetime).
    pub fn total_recorded(&self) -> u64 {
        self.count
    }

    /// Number of live samples currently in the ring buffer.
    pub fn live_samples(&self) -> usize {
        if (self.count as usize) < LATENCY_HISTOGRAM_CAPACITY {
            self.count as usize
        } else {
            LATENCY_HISTOGRAM_CAPACITY
        }
    }

    /// Compute the p50 (median) latency. Returns `Duration::ZERO` if empty.
    pub fn p50(&self) -> Duration {
        self.percentile(0.50)
    }

    /// Compute the p99 latency. Returns `Duration::ZERO` if empty.
    pub fn p99(&self) -> Duration {
        self.percentile(0.99)
    }

    /// Compute an arbitrary percentile in `[0.0, 1.0]`.
    pub fn percentile(&self, p: f64) -> Duration {
        let n = self.live_samples();
        if n == 0 {
            return Duration::ZERO;
        }
        let mut buf: Vec<Duration> = self.samples[..n].to_vec();
        buf.sort_unstable();
        let p = p.clamp(0.0, 1.0);
        // Nearest-rank: ceil(p * n) - 1, clamped to [0, n-1].
        let rank = ((p * n as f64).ceil() as usize)
            .saturating_sub(1)
            .min(n - 1);
        buf[rank]
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-actor mutable state tracked by [`MetricAggregator`].
#[derive(Debug)]
struct ActorMetricState {
    last_sample_at: Instant,
    inbound_ewma: f64,
    outbound_ewma: f64,
    inbound_count: u64,
    outbound_count: u64,
    // Delta counters since the last EWMA update.
    inbound_delta: u64,
    outbound_delta: u64,
    latency_histogram: LatencyHistogram,
    state_size_bytes: u64,
    queue_depth: usize,
    gpu_utilization: f32,
    tenant_id: u64,
    hlc_node_id: u64,
    initialized: bool,
}

impl ActorMetricState {
    fn new(hlc_node_id: u64) -> Self {
        Self {
            last_sample_at: Instant::now(),
            inbound_ewma: 0.0,
            outbound_ewma: 0.0,
            inbound_count: 0,
            outbound_count: 0,
            inbound_delta: 0,
            outbound_delta: 0,
            latency_histogram: LatencyHistogram::new(),
            state_size_bytes: 0,
            queue_depth: 0,
            gpu_utilization: 0.0,
            tenant_id: 0,
            hlc_node_id,
            initialized: false,
        }
    }
}

/// Aggregates per-actor metric counters and produces smoothed snapshots.
///
/// The aggregator is the CPU-side single source of truth for streaming
/// metrics. Backends (CPU dispatcher, CUDA K2H processor) feed it raw
/// counters via [`record_inbound`], [`record_outbound`], and
/// [`record_latency`]; subscribers consume smoothed snapshots via
/// [`snapshot`] / [`snapshot_all`].
///
/// The aggregator is `Send + Sync`. Internal state is guarded by
/// `parking_lot::RwLock`.
///
/// [`record_inbound`]: MetricAggregator::record_inbound
/// [`record_outbound`]: MetricAggregator::record_outbound
/// [`record_latency`]: MetricAggregator::record_latency
/// [`snapshot`]: MetricAggregator::snapshot
/// [`snapshot_all`]: MetricAggregator::snapshot_all
pub struct MetricAggregator {
    per_actor: RwLock<HashMap<ActorId, ActorMetricState>>,
    ewma_alpha: f64,
    hlc_node_id: u64,
}

impl MetricAggregator {
    /// Create an aggregator with the default EWMA alpha ([`DEFAULT_EWMA_ALPHA`]).
    pub fn new() -> Self {
        Self::with_alpha(DEFAULT_EWMA_ALPHA)
    }

    /// Create an aggregator with a custom EWMA smoothing factor.
    ///
    /// `alpha` is clamped to `(0.0, 1.0]`.
    pub fn with_alpha(alpha: f64) -> Self {
        let ewma_alpha = if alpha.is_finite() && alpha > 0.0 && alpha <= 1.0 {
            alpha
        } else {
            DEFAULT_EWMA_ALPHA
        };
        Self {
            per_actor: RwLock::new(HashMap::new()),
            ewma_alpha,
            hlc_node_id: 0,
        }
    }

    /// Configure the HLC node identifier used when stamping snapshots.
    pub fn with_hlc_node_id(mut self, node_id: u64) -> Self {
        self.hlc_node_id = node_id;
        self
    }

    /// Current EWMA smoothing factor.
    pub fn ewma_alpha(&self) -> f64 {
        self.ewma_alpha
    }

    /// Record `count` inbound messages for `actor_id`.
    pub fn record_inbound(&self, actor_id: ActorId, count: u64) {
        let mut guard = self.per_actor.write();
        let state = guard
            .entry(actor_id)
            .or_insert_with(|| ActorMetricState::new(self.hlc_node_id));
        state.inbound_count = state.inbound_count.saturating_add(count);
        state.inbound_delta = state.inbound_delta.saturating_add(count);
    }

    /// Record `count` outbound messages for `actor_id`.
    pub fn record_outbound(&self, actor_id: ActorId, count: u64) {
        let mut guard = self.per_actor.write();
        let state = guard
            .entry(actor_id)
            .or_insert_with(|| ActorMetricState::new(self.hlc_node_id));
        state.outbound_count = state.outbound_count.saturating_add(count);
        state.outbound_delta = state.outbound_delta.saturating_add(count);
    }

    /// Record an observed processing latency sample for `actor_id`.
    pub fn record_latency(&self, actor_id: ActorId, d: Duration) {
        let mut guard = self.per_actor.write();
        let state = guard
            .entry(actor_id)
            .or_insert_with(|| ActorMetricState::new(self.hlc_node_id));
        state.latency_histogram.record(d);
    }

    /// Set the current queue depth for `actor_id`.
    pub fn set_queue_depth(&self, actor_id: ActorId, depth: usize) {
        let mut guard = self.per_actor.write();
        let state = guard
            .entry(actor_id)
            .or_insert_with(|| ActorMetricState::new(self.hlc_node_id));
        state.queue_depth = depth;
    }

    /// Set the current resident state size for `actor_id`.
    pub fn set_state_size(&self, actor_id: ActorId, bytes: u64) {
        let mut guard = self.per_actor.write();
        let state = guard
            .entry(actor_id)
            .or_insert_with(|| ActorMetricState::new(self.hlc_node_id));
        state.state_size_bytes = bytes;
    }

    /// Set the current GPU utilization for `actor_id`. Values outside `[0.0, 1.0]` are clamped.
    pub fn set_gpu_utilization(&self, actor_id: ActorId, util: f32) {
        let util = util.clamp(0.0, 1.0);
        let mut guard = self.per_actor.write();
        let state = guard
            .entry(actor_id)
            .or_insert_with(|| ActorMetricState::new(self.hlc_node_id));
        state.gpu_utilization = util;
    }

    /// Associate a tenant with `actor_id`.
    pub fn set_tenant(&self, actor_id: ActorId, tenant_id: u64) {
        let mut guard = self.per_actor.write();
        let state = guard
            .entry(actor_id)
            .or_insert_with(|| ActorMetricState::new(self.hlc_node_id));
        state.tenant_id = tenant_id;
    }

    /// Remove an actor's state. Returns true if the actor was tracked.
    pub fn remove_actor(&self, actor_id: &ActorId) -> bool {
        self.per_actor.write().remove(actor_id).is_some()
    }

    /// Number of actors currently tracked.
    pub fn tracked_actors(&self) -> usize {
        self.per_actor.read().len()
    }

    /// Produce a smoothed snapshot for `actor_id`, updating EWMA rates.
    ///
    /// Returns `None` if the actor is unknown.
    pub fn snapshot(&self, actor_id: &ActorId) -> Option<LiveMetrics> {
        let mut guard = self.per_actor.write();
        let state = guard.get_mut(actor_id)?;
        let (p50, p99) = (state.latency_histogram.p50(), state.latency_histogram.p99());
        let metrics = Self::fold_snapshot(*actor_id, state, self.ewma_alpha, p50, p99);
        Some(metrics)
    }

    /// Produce smoothed snapshots for all tracked actors.
    pub fn snapshot_all(&self) -> Vec<LiveMetrics> {
        let mut guard = self.per_actor.write();
        let alpha = self.ewma_alpha;
        let mut out = Vec::with_capacity(guard.len());
        for (id, state) in guard.iter_mut() {
            let (p50, p99) = (state.latency_histogram.p50(), state.latency_histogram.p99());
            out.push(Self::fold_snapshot(*id, state, alpha, p50, p99));
        }
        out
    }

    fn fold_snapshot(
        actor_id: ActorId,
        state: &mut ActorMetricState,
        alpha: f64,
        p50: Duration,
        p99: Duration,
    ) -> LiveMetrics {
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_sample_at).as_secs_f64();

        // Compute per-sample rate from the counter deltas since the last
        // snapshot. Guard against zero elapsed time (back-to-back snapshots).
        let inbound_rate_sample = if elapsed > 0.0 {
            state.inbound_delta as f64 / elapsed
        } else {
            0.0
        };
        let outbound_rate_sample = if elapsed > 0.0 {
            state.outbound_delta as f64 / elapsed
        } else {
            0.0
        };

        if !state.initialized {
            // Seed the EWMA with the first real observation so we don't bias
            // toward zero for many intervals.
            state.inbound_ewma = inbound_rate_sample;
            state.outbound_ewma = outbound_rate_sample;
            state.initialized = true;
        } else {
            state.inbound_ewma = alpha * inbound_rate_sample + (1.0 - alpha) * state.inbound_ewma;
            state.outbound_ewma =
                alpha * outbound_rate_sample + (1.0 - alpha) * state.outbound_ewma;
        }

        state.inbound_delta = 0;
        state.outbound_delta = 0;
        state.last_sample_at = now;

        let physical = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        let timestamp = HlcTimestamp::new(physical, 0, state.hlc_node_id);

        LiveMetrics {
            actor_id,
            timestamp,
            queue_depth: state.queue_depth,
            inbound_rate: state.inbound_ewma,
            outbound_rate: state.outbound_ewma,
            latency_p50: p50,
            latency_p99: p99,
            state_size_bytes: state.state_size_bytes,
            gpu_utilization: state.gpu_utilization,
            tenant_id: state.tenant_id,
        }
    }
}

impl Default for MetricAggregator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MetricAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricAggregator")
            .field("ewma_alpha", &self.ewma_alpha)
            .field("hlc_node_id", &self.hlc_node_id)
            .field("tracked_actors", &self.tracked_actors())
            .finish()
    }
}

/// Streaming live-metric dispatcher with per-actor subscriptions.
///
/// Backends call [`emit`] periodically with fresh [`LiveMetrics`]; the
/// stream fans out to all subscribers registered for the metric's actor
/// (plus any "all-actors" subscribers). Subscribers receive metrics via
/// tokio unbounded MPSC channels, so dispatch is non-blocking.
///
/// Broken receivers (dropped by the subscriber) are detected on the next
/// dispatch attempt and auto-pruned.
///
/// [`emit`]: IntrospectionStream::emit
pub struct IntrospectionStream {
    /// Per-actor subscribers. `ActorId::MAX` sentinel below is *not* used;
    /// global subscribers live in a dedicated bucket.
    subscriptions: RwLock<HashMap<ActorId, Vec<Arc<SubscriberHandle>>>>,
    /// Subscribers that want every actor's metrics.
    global_subscriptions: RwLock<Vec<Arc<SubscriberHandle>>>,
    /// Aggregator that computes snapshots (optional — callers may emit
    /// directly for GPU-sourced metrics).
    aggregator: Arc<MetricAggregator>,
    /// Monotonic counter for subscription IDs.
    next_subscription_id: AtomicU64,
}

impl IntrospectionStream {
    /// Create a new introspection stream with a fresh aggregator.
    pub fn new() -> Self {
        Self::with_aggregator(Arc::new(MetricAggregator::new()))
    }

    /// Create a stream bound to an existing aggregator.
    pub fn with_aggregator(aggregator: Arc<MetricAggregator>) -> Self {
        Self {
            subscriptions: RwLock::new(HashMap::new()),
            global_subscriptions: RwLock::new(Vec::new()),
            aggregator,
            next_subscription_id: AtomicU64::new(1),
        }
    }

    /// Access the underlying aggregator (shared via `Arc`).
    pub fn aggregator(&self) -> Arc<MetricAggregator> {
        self.aggregator.clone()
    }

    /// Subscribe to metrics for one actor at the given interval.
    ///
    /// Returns the receiver half of a tokio unbounded MPSC channel. The
    /// sender lives inside a [`SubscriberHandle`] held by the stream; when
    /// the caller drops the returned receiver, the handle becomes closed
    /// and is pruned on the next dispatch.
    ///
    /// An interval of `Duration::ZERO` is equivalent to unsubscribing — in
    /// that case this function returns a receiver whose sender has already
    /// been dropped.
    pub fn subscribe(
        &self,
        actor_id: ActorId,
        interval: Duration,
    ) -> mpsc::UnboundedReceiver<LiveMetrics> {
        let (tx, rx) = mpsc::unbounded_channel();
        if interval.is_zero() {
            // "interval = 0" means unsubscribe per spec. Drop tx so the
            // receiver reports closed immediately, and proactively remove
            // any existing subscriptions for this actor.
            drop(tx);
            self.unsubscribe(actor_id);
            return rx;
        }
        let handle = Arc::new(SubscriberHandle {
            interval,
            last_sent_at: parking_lot::Mutex::new(None),
            subscription_id: self.next_subscription_id.fetch_add(1, Ordering::Relaxed),
            sender: tx,
        });
        self.subscriptions
            .write()
            .entry(actor_id)
            .or_default()
            .push(handle);
        rx
    }

    /// Subscribe to metrics for every actor.
    ///
    /// `interval` of `Duration::ZERO` is equivalent to an immediate no-op
    /// (returns a closed receiver).
    pub fn subscribe_all(&self, interval: Duration) -> mpsc::UnboundedReceiver<LiveMetrics> {
        let (tx, rx) = mpsc::unbounded_channel();
        if interval.is_zero() {
            drop(tx);
            return rx;
        }
        let handle = Arc::new(SubscriberHandle {
            interval,
            last_sent_at: parking_lot::Mutex::new(None),
            subscription_id: self.next_subscription_id.fetch_add(1, Ordering::Relaxed),
            sender: tx,
        });
        self.global_subscriptions.write().push(handle);
        rx
    }

    /// Drop every subscription for `actor_id`.
    pub fn unsubscribe(&self, actor_id: ActorId) {
        self.subscriptions.write().remove(&actor_id);
    }

    /// Drop every global "all-actors" subscription.
    pub fn unsubscribe_all(&self) {
        self.global_subscriptions.write().clear();
    }

    /// Emit a metric to all applicable subscribers.
    ///
    /// Called by backends (CPU dispatcher) or K2H processor (CUDA). Any
    /// subscribers whose receiver has been dropped are auto-removed during
    /// dispatch. Interval gating is applied per-subscriber, so a fast
    /// producer with slow subscribers will not spam them.
    pub fn emit(&self, metrics: LiveMetrics) {
        // Dispatch to per-actor subscribers.
        let actor_id = metrics.actor_id;
        {
            let mut guard = self.subscriptions.write();
            if let Some(subs) = guard.get_mut(&actor_id) {
                Self::dispatch_and_prune(subs, &metrics);
                if subs.is_empty() {
                    guard.remove(&actor_id);
                }
            }
        }
        // Dispatch to global subscribers.
        {
            let mut guard = self.global_subscriptions.write();
            Self::dispatch_and_prune(&mut guard, &metrics);
        }
    }

    fn dispatch_and_prune(subs: &mut Vec<Arc<SubscriberHandle>>, metrics: &LiveMetrics) {
        subs.retain(|handle| {
            if handle.is_closed() {
                return false;
            }
            match handle.try_send(metrics.clone()) {
                Ok(_) => true,
                Err(()) => false,
            }
        });
    }

    /// Number of active subscribers for one actor.
    pub fn subscriber_count(&self, actor_id: &ActorId) -> usize {
        self.subscriptions
            .read()
            .get(actor_id)
            .map(|v| v.iter().filter(|h| !h.is_closed()).count())
            .unwrap_or(0)
    }

    /// Number of active "all-actors" subscribers.
    pub fn global_subscriber_count(&self) -> usize {
        self.global_subscriptions
            .read()
            .iter()
            .filter(|h| !h.is_closed())
            .count()
    }

    /// Total number of active subscribers across all buckets.
    pub fn total_subscribers(&self) -> usize {
        let per_actor: usize = self
            .subscriptions
            .read()
            .values()
            .map(|v| v.iter().filter(|h| !h.is_closed()).count())
            .sum();
        per_actor + self.global_subscriber_count()
    }

    /// Run a full aggregation cycle: snapshot every tracked actor and emit
    /// the results. Useful for drivers that want a single "tick" call.
    pub fn tick(&self) -> usize {
        let snapshots = self.aggregator.snapshot_all();
        let n = snapshots.len();
        for metrics in snapshots {
            self.emit(metrics);
        }
        n
    }
}

impl Default for IntrospectionStream {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for IntrospectionStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntrospectionStream")
            .field("total_subscribers", &self.total_subscribers())
            .field("aggregator", &self.aggregator)
            .finish()
    }
}

// =============================================================================
// H2K / K2H Wire Protocol (GPU-mockable, CPU-side only for v1.1)
// =============================================================================

/// H2K command: subscribe to metrics from a GPU actor.
///
/// Encoded into the H2K message payload on the CUDA backend. `interval_us`
/// of 0 is interpreted as an unsubscribe request.
///
/// # Wire Layout
///
/// The type is `#[repr(C)]` and `Pod`-like (no padding-sensitive fields) so
/// it maps 1:1 to the GPU-side C struct of the same name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct SubscribeMetricsRequest {
    /// Target actor (thread-block index on GPU).
    pub actor_id: u64,
    /// Emission interval in microseconds; 0 = unsubscribe.
    pub interval_us: u64,
    /// Caller-generated subscription ID (echoed back in events).
    pub subscription_id: u64,
}

impl SubscribeMetricsRequest {
    /// Size of the wire representation in bytes.
    pub const WIRE_SIZE: usize = std::mem::size_of::<Self>();

    /// Construct a new subscribe request.
    pub const fn new(actor_id: u64, interval_us: u64, subscription_id: u64) -> Self {
        Self {
            actor_id,
            interval_us,
            subscription_id,
        }
    }

    /// Construct an unsubscribe request (interval = 0).
    pub const fn unsubscribe(actor_id: u64, subscription_id: u64) -> Self {
        Self::new(actor_id, 0, subscription_id)
    }

    /// Return true if this message is an unsubscribe request.
    pub const fn is_unsubscribe(&self) -> bool {
        self.interval_us == 0
    }

    /// Serialize to a fixed-size byte array.
    pub fn to_bytes(&self) -> [u8; Self::WIRE_SIZE] {
        // SAFETY: `Self` is `#[repr(C)]` with only `u64` fields — no padding,
        // no invalid bit patterns.
        unsafe { std::mem::transmute::<Self, [u8; Self::WIRE_SIZE]>(*self) }
    }

    /// Deserialize from a byte slice. Returns `None` if the slice is too short.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::WIRE_SIZE {
            return None;
        }
        let mut buf = [0u8; Self::WIRE_SIZE];
        buf.copy_from_slice(&bytes[..Self::WIRE_SIZE]);
        // SAFETY: All-u64 layout; every bit pattern is a valid `Self`.
        Some(unsafe { std::mem::transmute::<[u8; Self::WIRE_SIZE], Self>(buf) })
    }
}

/// K2H response: periodic metric emission from a GPU actor.
///
/// This is the on-wire form of [`LiveMetrics`] — fixed-size, `#[repr(C)]`,
/// and compact so it can be emitted at high frequency from GPU-side. The
/// CPU-side K2H processor lifts it into a [`LiveMetrics`] for subscribers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct LiveMetricsEvent {
    /// Subscription this event is emitted for.
    pub subscription_id: u64,
    /// Source actor (thread-block index on GPU).
    pub actor_id: u64,
    /// Microsecond-precision timestamp (wall clock, GPU-side origin).
    pub timestamp_us: u64,
    /// Total inbound messages observed (monotonic, cumulative).
    pub inbound_total: u64,
    /// Total outbound messages observed (monotonic, cumulative).
    pub outbound_total: u64,
    /// Tenant identifier (0 = unspecified).
    pub tenant_id: u64,
    /// Current inbound queue depth.
    pub queue_depth: u32,
    /// Observed p50 latency in microseconds.
    pub latency_p50_us: u32,
    /// Observed p99 latency in microseconds.
    pub latency_p99_us: u32,
    /// Resident actor state size in bytes (truncated to u32).
    pub state_size_bytes: u32,
    /// GPU utilization percentage 0–100.
    pub gpu_utilization_pct: u8,
    /// Explicit padding for 16-byte alignment (C ABI stability).
    pub _pad: [u8; 7],
}

impl LiveMetricsEvent {
    /// Size of the wire representation in bytes.
    pub const WIRE_SIZE: usize = std::mem::size_of::<Self>();

    /// Construct a new event. The padding field is zeroed.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        subscription_id: u64,
        actor_id: u64,
        timestamp_us: u64,
        inbound_total: u64,
        outbound_total: u64,
        tenant_id: u64,
        queue_depth: u32,
        latency_p50_us: u32,
        latency_p99_us: u32,
        state_size_bytes: u32,
        gpu_utilization_pct: u8,
    ) -> Self {
        Self {
            subscription_id,
            actor_id,
            timestamp_us,
            inbound_total,
            outbound_total,
            tenant_id,
            queue_depth,
            latency_p50_us,
            latency_p99_us,
            state_size_bytes,
            gpu_utilization_pct,
            _pad: [0; 7],
        }
    }

    /// Serialize to a fixed-size byte array.
    pub fn to_bytes(&self) -> [u8; Self::WIRE_SIZE] {
        // SAFETY: `Self` is `#[repr(C)]` with `u64`/`u32`/`u8` fields and an
        // explicit padding array — no implicit padding holes, no invalid bit
        // patterns.
        unsafe { std::mem::transmute::<Self, [u8; Self::WIRE_SIZE]>(*self) }
    }

    /// Deserialize from a byte slice. Returns `None` if the slice is too short.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::WIRE_SIZE {
            return None;
        }
        let mut buf = [0u8; Self::WIRE_SIZE];
        buf.copy_from_slice(&bytes[..Self::WIRE_SIZE]);
        // SAFETY: Plain-old-data layout; all bit patterns are valid.
        Some(unsafe { std::mem::transmute::<[u8; Self::WIRE_SIZE], Self>(buf) })
    }

    /// Lift into a [`LiveMetrics`] observation.
    pub fn into_live_metrics(self, hlc_node_id: u64) -> LiveMetrics {
        LiveMetrics {
            actor_id: ActorId(self.actor_id as u32),
            timestamp: HlcTimestamp::new(self.timestamp_us, 0, hlc_node_id),
            queue_depth: self.queue_depth as usize,
            // Wire form carries cumulative totals; rate smoothing is a
            // CPU-side responsibility. We expose totals-as-rates = 0 to
            // indicate "not yet smoothed".
            inbound_rate: 0.0,
            outbound_rate: 0.0,
            latency_p50: Duration::from_micros(self.latency_p50_us as u64),
            latency_p99: Duration::from_micros(self.latency_p99_us as u64),
            state_size_bytes: self.state_size_bytes as u64,
            gpu_utilization: (self.gpu_utilization_pct as f32 / 100.0).clamp(0.0, 1.0),
            tenant_id: self.tenant_id,
        }
    }
}

// Compile-time size assertions — changing wire sizes is a breaking change.
const _: () = assert!(SubscribeMetricsRequest::WIRE_SIZE == 24);
const _: () = assert!(LiveMetricsEvent::WIRE_SIZE == 72);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_buffer_basic() {
        let mut buf = TraceBuffer::new(3);

        for i in 0..5 {
            buf.record(TraceEntry {
                sequence: i,
                received_at: Instant::now(),
                duration: Duration::from_micros(100),
                source: None,
                outcome: TraceOutcome::Success,
            });
        }

        assert_eq!(buf.len(), 3); // Capacity limited
        assert_eq!(buf.total_recorded(), 5);
    }

    #[test]
    fn test_trace_buffer_recent() {
        let mut buf = TraceBuffer::new(10);

        for i in 0..5 {
            buf.record(TraceEntry {
                sequence: i,
                received_at: Instant::now(),
                duration: Duration::from_micros(100),
                source: None,
                outcome: TraceOutcome::Success,
            });
            std::thread::sleep(Duration::from_millis(1));
        }

        let recent = buf.recent(3);
        assert_eq!(recent.len(), 3);
        // Most recent should have highest sequence
        assert!(recent[0].sequence > recent[2].sequence);
    }

    #[test]
    fn test_introspection_service() {
        let mut svc = IntrospectionService::new(10);

        let actor = ActorId(1);
        svc.register_actor(actor);

        svc.record_trace(
            actor,
            TraceEntry {
                sequence: 1,
                received_at: Instant::now(),
                duration: Duration::from_micros(50),
                source: Some(ActorId(2)),
                outcome: TraceOutcome::Forwarded(ActorId(3)),
            },
        );

        let traces = svc.get_traces(actor, 10);
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].sequence, 1);
    }

    #[test]
    fn test_queue_snapshot_utilization() {
        let snap = QueueSnapshot {
            input_depth: 75,
            input_capacity: 100,
            ..Default::default()
        };
        assert!((snap.input_utilization() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_trace_outcome_variants() {
        assert_eq!(TraceOutcome::Success, TraceOutcome::Success);
        assert_ne!(TraceOutcome::Success, TraceOutcome::Dropped);
    }

    // =========================================================================
    // Streaming introspection tests (v1.1 §3.2)
    // =========================================================================

    fn mk_metrics(id: u32) -> LiveMetrics {
        LiveMetrics {
            actor_id: ActorId(id),
            timestamp: HlcTimestamp::new(0, 0, 0),
            queue_depth: 0,
            inbound_rate: 0.0,
            outbound_rate: 0.0,
            latency_p50: Duration::ZERO,
            latency_p99: Duration::ZERO,
            state_size_bytes: 0,
            gpu_utilization: 0.0,
            tenant_id: 0,
        }
    }

    #[tokio::test]
    async fn test_subscribe_receives_metric() {
        let stream = IntrospectionStream::new();
        let actor = ActorId(42);
        let mut rx = stream.subscribe(actor, Duration::from_nanos(1));
        assert_eq!(stream.subscriber_count(&actor), 1);

        stream.emit(mk_metrics(42));
        let got = rx.recv().await.expect("metric delivered");
        assert_eq!(got.actor_id, actor);
    }

    #[tokio::test]
    async fn test_subscribe_filters_by_actor() {
        let stream = IntrospectionStream::new();
        let mut rx = stream.subscribe(ActorId(1), Duration::from_nanos(1));

        // Emit for a different actor — should not be delivered.
        stream.emit(mk_metrics(2));
        // Emit for the subscribed actor — should be delivered.
        stream.emit(mk_metrics(1));

        let got = rx.recv().await.expect("metric delivered");
        assert_eq!(got.actor_id, ActorId(1));
        assert!(rx.try_recv().is_err(), "no other metric should be queued");
    }

    #[tokio::test]
    async fn test_unsubscribe_stops_delivery() {
        let stream = IntrospectionStream::new();
        let actor = ActorId(7);
        let mut rx = stream.subscribe(actor, Duration::from_nanos(1));
        stream.unsubscribe(actor);
        assert_eq!(stream.subscriber_count(&actor), 0);
        stream.emit(mk_metrics(7));

        // The sender was dropped inside the stream; the receiver should
        // either return None (closed) or have nothing queued.
        match rx.recv().await {
            None => {}
            Some(_) => panic!("no metric should be delivered after unsubscribe"),
        }
    }

    #[tokio::test]
    async fn test_subscribe_interval_zero_is_unsubscribe() {
        let stream = IntrospectionStream::new();
        let actor = ActorId(3);
        // First subscribe normally, then call with interval zero.
        let _keep = stream.subscribe(actor, Duration::from_micros(1));
        assert_eq!(stream.subscriber_count(&actor), 1);

        let mut rx_zero = stream.subscribe(actor, Duration::ZERO);
        assert_eq!(stream.subscriber_count(&actor), 0);
        stream.emit(mk_metrics(3));
        assert!(rx_zero.recv().await.is_none(), "zero interval = closed");
    }

    #[tokio::test]
    async fn test_multi_subscriber_fanout() {
        let stream = IntrospectionStream::new();
        let actor = ActorId(10);
        let mut rx1 = stream.subscribe(actor, Duration::from_nanos(1));
        let mut rx2 = stream.subscribe(actor, Duration::from_nanos(1));
        let mut rx3 = stream.subscribe(actor, Duration::from_nanos(1));
        assert_eq!(stream.subscriber_count(&actor), 3);

        stream.emit(mk_metrics(10));

        assert_eq!(rx1.recv().await.unwrap().actor_id, actor);
        assert_eq!(rx2.recv().await.unwrap().actor_id, actor);
        assert_eq!(rx3.recv().await.unwrap().actor_id, actor);
    }

    #[tokio::test]
    async fn test_broken_receiver_auto_cleanup() {
        let stream = IntrospectionStream::new();
        let actor = ActorId(9);
        let rx1 = stream.subscribe(actor, Duration::from_nanos(1));
        let mut rx2 = stream.subscribe(actor, Duration::from_nanos(1));
        assert_eq!(stream.subscriber_count(&actor), 2);

        // Drop rx1 — next emit should prune its handle.
        drop(rx1);
        stream.emit(mk_metrics(9));
        assert_eq!(stream.subscriber_count(&actor), 1);

        // rx2 is still alive and should have received the metric.
        assert_eq!(rx2.recv().await.unwrap().actor_id, actor);
    }

    #[tokio::test]
    async fn test_broken_receiver_closes_empty_bucket() {
        let stream = IntrospectionStream::new();
        let actor = ActorId(15);
        let rx = stream.subscribe(actor, Duration::from_nanos(1));
        drop(rx);
        stream.emit(mk_metrics(15));
        assert_eq!(stream.subscriber_count(&actor), 0);
        // Emitting again after pruning must not panic.
        stream.emit(mk_metrics(15));
    }

    #[tokio::test]
    async fn test_subscribe_all_receives_all_actors() {
        let stream = IntrospectionStream::new();
        let mut rx = stream.subscribe_all(Duration::from_nanos(1));
        stream.emit(mk_metrics(1));
        stream.emit(mk_metrics(2));
        stream.emit(mk_metrics(3));

        let mut seen = Vec::new();
        for _ in 0..3 {
            seen.push(rx.recv().await.unwrap().actor_id);
        }
        seen.sort_by_key(|a| a.0);
        assert_eq!(seen, vec![ActorId(1), ActorId(2), ActorId(3)]);
    }

    #[tokio::test]
    async fn test_subscribe_all_plus_specific_both_fire() {
        let stream = IntrospectionStream::new();
        let mut rx_all = stream.subscribe_all(Duration::from_nanos(1));
        let mut rx_one = stream.subscribe(ActorId(5), Duration::from_nanos(1));

        stream.emit(mk_metrics(5));

        assert_eq!(rx_all.recv().await.unwrap().actor_id, ActorId(5));
        assert_eq!(rx_one.recv().await.unwrap().actor_id, ActorId(5));
    }

    #[tokio::test]
    async fn test_subscribe_all_interval_zero_is_no_op() {
        let stream = IntrospectionStream::new();
        let mut rx = stream.subscribe_all(Duration::ZERO);
        stream.emit(mk_metrics(1));
        assert!(rx.recv().await.is_none());
        assert_eq!(stream.global_subscriber_count(), 0);
    }

    #[tokio::test]
    async fn test_unsubscribe_all_clears_global() {
        let stream = IntrospectionStream::new();
        let _rx1 = stream.subscribe_all(Duration::from_nanos(1));
        let _rx2 = stream.subscribe_all(Duration::from_nanos(1));
        assert_eq!(stream.global_subscriber_count(), 2);
        stream.unsubscribe_all();
        assert_eq!(stream.global_subscriber_count(), 0);
    }

    #[tokio::test]
    async fn test_interval_gating_suppresses_faster_emits() {
        let stream = IntrospectionStream::new();
        let actor = ActorId(11);
        let mut rx = stream.subscribe(actor, Duration::from_millis(500));

        // First emit — delivered (no prior sample).
        stream.emit(mk_metrics(11));
        // Second emit immediately after — suppressed by interval gate.
        stream.emit(mk_metrics(11));

        assert_eq!(rx.recv().await.unwrap().actor_id, actor);
        assert!(
            rx.try_recv().is_err(),
            "second emit should be gated by interval"
        );
    }

    #[tokio::test]
    async fn test_total_subscribers_sums_buckets() {
        let stream = IntrospectionStream::new();
        let _a = stream.subscribe(ActorId(1), Duration::from_nanos(1));
        let _b = stream.subscribe(ActorId(1), Duration::from_nanos(1));
        let _c = stream.subscribe(ActorId(2), Duration::from_nanos(1));
        let _d = stream.subscribe_all(Duration::from_nanos(1));
        assert_eq!(stream.total_subscribers(), 4);
    }

    #[test]
    fn test_subscription_ids_are_unique_and_monotonic() {
        let stream = IntrospectionStream::new();
        let _a = stream.subscribe(ActorId(1), Duration::from_nanos(1));
        let _b = stream.subscribe(ActorId(1), Duration::from_nanos(1));
        let _c = stream.subscribe_all(Duration::from_nanos(1));

        let subs = stream.subscriptions.read();
        let handles = subs.get(&ActorId(1)).expect("bucket exists");
        assert_eq!(handles.len(), 2);
        assert!(handles[0].subscription_id < handles[1].subscription_id);

        let globals = stream.global_subscriptions.read();
        assert_eq!(globals.len(), 1);
        assert!(globals[0].subscription_id > handles[1].subscription_id);
    }

    // ---- MetricAggregator ---------------------------------------------------

    #[test]
    fn test_aggregator_record_and_snapshot() {
        let agg = MetricAggregator::new();
        let a = ActorId(1);
        agg.record_inbound(a, 10);
        agg.record_outbound(a, 5);
        agg.set_queue_depth(a, 3);
        agg.set_state_size(a, 4096);
        agg.set_gpu_utilization(a, 0.75);
        agg.set_tenant(a, 42);

        let snap = agg.snapshot(&a).expect("snapshot exists");
        assert_eq!(snap.queue_depth, 3);
        assert_eq!(snap.state_size_bytes, 4096);
        assert!((snap.gpu_utilization - 0.75).abs() < 1e-6);
        assert_eq!(snap.tenant_id, 42);
        // The first snapshot seeds EWMA with the raw rate — must be positive.
        assert!(snap.inbound_rate > 0.0);
        assert!(snap.outbound_rate > 0.0);
    }

    #[test]
    fn test_aggregator_snapshot_unknown_actor_returns_none() {
        let agg = MetricAggregator::new();
        assert!(agg.snapshot(&ActorId(999)).is_none());
    }

    #[test]
    fn test_aggregator_snapshot_all_covers_every_actor() {
        let agg = MetricAggregator::new();
        for i in 0..5 {
            agg.record_inbound(ActorId(i), 1);
        }
        let all = agg.snapshot_all();
        assert_eq!(all.len(), 5);
        let mut ids: Vec<u32> = all.iter().map(|m| m.actor_id.0).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_aggregator_remove_actor() {
        let agg = MetricAggregator::new();
        agg.record_inbound(ActorId(1), 1);
        assert_eq!(agg.tracked_actors(), 1);
        assert!(agg.remove_actor(&ActorId(1)));
        assert_eq!(agg.tracked_actors(), 0);
        assert!(!agg.remove_actor(&ActorId(1)));
    }

    #[test]
    fn test_aggregator_gpu_utilization_clamped() {
        let agg = MetricAggregator::new();
        let a = ActorId(1);
        agg.record_inbound(a, 1);
        agg.set_gpu_utilization(a, 5.0);
        assert!((agg.snapshot(&a).unwrap().gpu_utilization - 1.0).abs() < 1e-6);
        agg.set_gpu_utilization(a, -1.0);
        assert_eq!(agg.snapshot(&a).unwrap().gpu_utilization, 0.0);
    }

    #[test]
    fn test_aggregator_with_custom_alpha() {
        let agg = MetricAggregator::with_alpha(0.5);
        assert!((agg.ewma_alpha() - 0.5).abs() < 1e-9);
        // Invalid alpha falls back to default.
        let agg = MetricAggregator::with_alpha(0.0);
        assert!((agg.ewma_alpha() - DEFAULT_EWMA_ALPHA).abs() < 1e-9);
        let agg = MetricAggregator::with_alpha(2.0);
        assert!((agg.ewma_alpha() - DEFAULT_EWMA_ALPHA).abs() < 1e-9);
        let agg = MetricAggregator::with_alpha(f64::NAN);
        assert!((agg.ewma_alpha() - DEFAULT_EWMA_ALPHA).abs() < 1e-9);
    }

    #[test]
    fn test_aggregator_ewma_smooths_spikes() {
        // Use a small alpha so a single spike cannot dominate.
        let agg = MetricAggregator::with_alpha(0.1);
        let a = ActorId(1);

        // Seed the EWMA with a steady base rate.
        for _ in 0..5 {
            agg.record_inbound(a, 100);
            std::thread::sleep(Duration::from_millis(20));
            let _ = agg.snapshot(&a);
        }
        let baseline = agg.snapshot(&a).unwrap().inbound_rate;

        // Send a giant spike and take one snapshot — EWMA should pull
        // the reported rate toward, but not all the way to, the spike.
        agg.record_inbound(a, 1_000_000);
        std::thread::sleep(Duration::from_millis(20));
        let after = agg.snapshot(&a).unwrap().inbound_rate;

        assert!(after > baseline, "spike should raise the smoothed rate");
        // With alpha=0.1 the rate cannot jump by more than ~10% of the
        // raw spike delta in a single sample.
        let raw_spike_rate = 1_000_000.0 / 0.020;
        assert!(
            after < raw_spike_rate,
            "EWMA must not fully adopt a single spike (after={after}, raw={raw_spike_rate})"
        );
    }

    #[test]
    fn test_aggregator_ewma_known_sequence() {
        // With alpha=1.0, EWMA degenerates to the most recent sample.
        // This lets us exercise the seeding + update paths deterministically.
        let agg = MetricAggregator::with_alpha(1.0);
        let a = ActorId(1);

        agg.record_inbound(a, 10);
        std::thread::sleep(Duration::from_millis(10));
        let first = agg.snapshot(&a).unwrap().inbound_rate;
        assert!(first > 0.0);

        // With alpha=1, the next snapshot should equal the new raw rate.
        agg.record_inbound(a, 0); // no new messages
        std::thread::sleep(Duration::from_millis(10));
        let second = agg.snapshot(&a).unwrap().inbound_rate;
        assert_eq!(second, 0.0, "alpha=1 adopts the raw rate verbatim");
    }

    // ---- LatencyHistogram ----------------------------------------------------

    #[test]
    fn test_latency_histogram_empty() {
        let h = LatencyHistogram::new();
        assert_eq!(h.p50(), Duration::ZERO);
        assert_eq!(h.p99(), Duration::ZERO);
        assert_eq!(h.live_samples(), 0);
        assert_eq!(h.total_recorded(), 0);
    }

    #[test]
    fn test_latency_histogram_percentiles() {
        let mut h = LatencyHistogram::new();
        for i in 1..=100u64 {
            h.record(Duration::from_millis(i));
        }
        assert_eq!(h.live_samples(), 100);
        // p50 with nearest-rank on 100 samples = rank 49 (1-indexed 50th).
        assert_eq!(h.p50(), Duration::from_millis(50));
        // p99 = rank 98 (1-indexed 99th).
        assert_eq!(h.p99(), Duration::from_millis(99));
    }

    #[test]
    fn test_latency_histogram_ring_wraparound() {
        let mut h = LatencyHistogram::new();
        // Fill past capacity — the oldest samples are overwritten.
        for i in 0..(LATENCY_HISTOGRAM_CAPACITY + 100) {
            h.record(Duration::from_micros(i as u64));
        }
        assert_eq!(h.live_samples(), LATENCY_HISTOGRAM_CAPACITY);
        assert_eq!(
            h.total_recorded(),
            (LATENCY_HISTOGRAM_CAPACITY + 100) as u64
        );
        // p99 should reflect only the retained (most recent) samples.
        assert!(h.p99() > Duration::from_micros(100));
    }

    #[test]
    fn test_latency_histogram_percentile_clamps() {
        let mut h = LatencyHistogram::new();
        h.record(Duration::from_millis(5));
        h.record(Duration::from_millis(10));
        assert_eq!(h.percentile(-1.0), h.percentile(0.0));
        assert_eq!(h.percentile(2.0), h.percentile(1.0));
    }

    #[test]
    fn test_aggregator_latency_snapshot_reflects_histogram() {
        let agg = MetricAggregator::new();
        let a = ActorId(1);
        for i in 1..=10u64 {
            agg.record_latency(a, Duration::from_millis(i));
        }
        agg.record_inbound(a, 1); // ensure rate fields are populated
        let snap = agg.snapshot(&a).unwrap();
        assert!(snap.latency_p50 >= Duration::from_millis(5));
        assert!(snap.latency_p99 >= Duration::from_millis(9));
    }

    // ---- IntrospectionStream <-> Aggregator integration ---------------------

    #[tokio::test]
    async fn test_stream_tick_emits_all_aggregated() {
        let stream = IntrospectionStream::new();
        let agg = stream.aggregator();
        agg.record_inbound(ActorId(1), 10);
        agg.record_inbound(ActorId(2), 20);
        let mut rx = stream.subscribe_all(Duration::from_nanos(1));

        let emitted = stream.tick();
        assert_eq!(emitted, 2);

        let mut seen = Vec::new();
        for _ in 0..2 {
            seen.push(rx.recv().await.unwrap().actor_id.0);
        }
        seen.sort();
        assert_eq!(seen, vec![1, 2]);
    }

    #[tokio::test]
    async fn test_stream_with_shared_aggregator() {
        let agg = Arc::new(MetricAggregator::new());
        let stream = IntrospectionStream::with_aggregator(agg.clone());
        assert!(Arc::ptr_eq(&agg, &stream.aggregator()));
        agg.record_inbound(ActorId(1), 1);
        let mut rx = stream.subscribe(ActorId(1), Duration::from_nanos(1));
        assert_eq!(stream.tick(), 1);
        assert_eq!(rx.recv().await.unwrap().actor_id, ActorId(1));
    }

    // ---- Concurrency --------------------------------------------------------

    #[tokio::test]
    async fn test_concurrent_subscribe_unsubscribe_race() {
        // Manual loom-style stress: many threads subscribing and unsubscribing
        // while another thread emits metrics. The invariant is safety (no
        // panic, no poisoned lock), not delivery guarantees.
        use std::sync::Barrier;
        use std::thread;

        let stream = Arc::new(IntrospectionStream::new());
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = Vec::new();

        // Two subscriber threads.
        for worker in 0..2 {
            let s = stream.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                for i in 0..200 {
                    let actor = ActorId((i + worker) % 8);
                    let _rx = s.subscribe(actor, Duration::from_nanos(1));
                    if i % 3 == 0 {
                        s.unsubscribe(actor);
                    }
                }
            }));
        }

        // Emitter thread.
        {
            let s = stream.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                for i in 0..400 {
                    s.emit(mk_metrics((i % 8) as u32));
                }
            }));
        }

        // Reader thread probes subscriber counts to exercise read locks.
        {
            let s = stream.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                for i in 0..400 {
                    let _ = s.subscriber_count(&ActorId((i % 8) as u32));
                    let _ = s.total_subscribers();
                }
            }));
        }

        for h in handles {
            h.join().expect("worker thread did not panic");
        }
        // After the storm, the stream is still functional.
        let mut rx = stream.subscribe(ActorId(123), Duration::from_nanos(1));
        stream.emit(mk_metrics(123));
        assert_eq!(rx.recv().await.unwrap().actor_id, ActorId(123));
    }

    #[tokio::test]
    async fn test_concurrent_aggregator_record_and_snapshot() {
        use std::sync::Barrier;
        use std::thread;

        let agg = Arc::new(MetricAggregator::new());
        let barrier = Arc::new(Barrier::new(3));
        let mut handles = Vec::new();

        for _ in 0..2 {
            let a = agg.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                for i in 0..500 {
                    a.record_inbound(ActorId((i % 4) as u32), 1);
                    a.record_latency(
                        ActorId((i % 4) as u32),
                        Duration::from_micros((i % 100) as u64),
                    );
                }
            }));
        }
        {
            let a = agg.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                for _ in 0..500 {
                    let _ = a.snapshot_all();
                }
            }));
        }

        for h in handles {
            h.join().expect("no panic");
        }
        assert_eq!(agg.tracked_actors(), 4);
    }

    // ---- Wire protocol ------------------------------------------------------

    #[test]
    fn test_subscribe_metrics_request_roundtrip() {
        let req = SubscribeMetricsRequest::new(42, 1000, 7);
        assert!(!req.is_unsubscribe());
        let bytes = req.to_bytes();
        assert_eq!(bytes.len(), SubscribeMetricsRequest::WIRE_SIZE);
        let decoded = SubscribeMetricsRequest::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, req);
    }

    #[test]
    fn test_subscribe_metrics_request_unsubscribe() {
        let req = SubscribeMetricsRequest::unsubscribe(42, 7);
        assert!(req.is_unsubscribe());
        assert_eq!(req.interval_us, 0);
        let decoded =
            SubscribeMetricsRequest::from_bytes(&req.to_bytes()).expect("roundtrip decode");
        assert!(decoded.is_unsubscribe());
    }

    #[test]
    fn test_subscribe_metrics_request_short_buffer() {
        let short = [0u8; SubscribeMetricsRequest::WIRE_SIZE - 1];
        assert!(SubscribeMetricsRequest::from_bytes(&short).is_none());
    }

    #[test]
    fn test_live_metrics_event_roundtrip() {
        let evt = LiveMetricsEvent::new(
            /* subscription_id */ 9,
            /* actor_id */ 42,
            /* timestamp_us */ 1_700_000_000_000_000,
            /* inbound_total */ 123,
            /* outbound_total */ 45,
            /* tenant_id */ 7,
            /* queue_depth */ 16,
            /* latency_p50_us */ 500,
            /* latency_p99_us */ 2_500,
            /* state_size_bytes */ 8_192,
            /* gpu_utilization_pct */ 73,
        );
        let bytes = evt.to_bytes();
        assert_eq!(bytes.len(), LiveMetricsEvent::WIRE_SIZE);
        let decoded = LiveMetricsEvent::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, evt);
    }

    #[test]
    fn test_live_metrics_event_short_buffer() {
        let short = [0u8; LiveMetricsEvent::WIRE_SIZE - 1];
        assert!(LiveMetricsEvent::from_bytes(&short).is_none());
    }

    #[test]
    fn test_live_metrics_event_into_live_metrics() {
        let evt = LiveMetricsEvent::new(1, 42, 1_234_567, 0, 0, 9, 16, 500, 2_500, 4096, 80);
        let metrics = evt.into_live_metrics(3);
        assert_eq!(metrics.actor_id, ActorId(42));
        assert_eq!(metrics.timestamp.physical, 1_234_567);
        assert_eq!(metrics.timestamp.node_id, 3);
        assert_eq!(metrics.queue_depth, 16);
        assert_eq!(metrics.latency_p50, Duration::from_micros(500));
        assert_eq!(metrics.latency_p99, Duration::from_micros(2_500));
        assert_eq!(metrics.state_size_bytes, 4096);
        assert!((metrics.gpu_utilization - 0.80).abs() < 1e-6);
        assert_eq!(metrics.tenant_id, 9);
    }

    #[test]
    fn test_wire_sizes_are_stable() {
        // Compile-time assertions guard size changes; these runtime checks
        // give a friendlier error message if the layout drifts.
        assert_eq!(SubscribeMetricsRequest::WIRE_SIZE, 24);
        assert_eq!(LiveMetricsEvent::WIRE_SIZE, 72);
    }

    #[test]
    fn test_live_metrics_event_roundtrip_preserves_gpu_pct_clamp() {
        // GPU util pct is a u8 0–100; conversion to f32 should clamp on read.
        let evt = LiveMetricsEvent::new(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 200);
        let m = evt.into_live_metrics(0);
        assert_eq!(m.gpu_utilization, 1.0);
    }

    #[test]
    fn test_subscribe_metrics_request_size_stable() {
        // Size must match the GPU-side C struct exactly.
        assert_eq!(std::mem::size_of::<SubscribeMetricsRequest>(), 24);
    }

    #[test]
    fn test_live_metrics_event_size_stable() {
        assert_eq!(std::mem::size_of::<LiveMetricsEvent>(), 72);
    }

    // ---- Existing pull API remains working (regression guard) ---------------

    #[test]
    fn test_pull_api_still_works_alongside_streaming() {
        let mut svc = IntrospectionService::new(4);
        let a = ActorId(1);
        svc.register_actor(a);
        svc.record_trace(
            a,
            TraceEntry {
                sequence: 0,
                received_at: Instant::now(),
                duration: Duration::from_micros(10),
                source: None,
                outcome: TraceOutcome::Success,
            },
        );
        assert_eq!(svc.get_traces(a, 10).len(), 1);
        // Streaming stack coexists, no interference.
        let stream = IntrospectionStream::new();
        assert_eq!(stream.total_subscribers(), 0);
    }
}
