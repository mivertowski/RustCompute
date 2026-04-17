//! Per-Actor Introspection API — FR-015
//!
//! Runtime actor inspection for debugging and monitoring:
//! - List all actors with state, queue depth, message rate
//! - Inspect per-actor metrics (read-only)
//! - Peek at queued messages
//! - Trace recent message processing with timing

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::actor::{ActorId, ActorState};

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
        result.sort_by(|a, b| b.received_at.cmp(&a.received_at));
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
}
