//! Ring context providing GPU intrinsics facade for kernel handlers.
//!
//! The RingContext provides a unified interface for GPU operations that
//! abstracts over different backends (CUDA, Metal, WebGPU, CPU).
//!
//! # Domain Support (FR-2)
//!
//! RingContext supports domain-aware operations including:
//! - Domain metadata access via `domain()`
//! - Metrics collection via `record_latency()` and `record_throughput()`
//! - Alert emission via `emit_alert()`
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::prelude::*;
//!
//! fn process(ctx: &mut RingContext, msg: MyMessage) {
//!     let start = std::time::Instant::now();
//!
//!     // Process message...
//!
//!     // Record metrics
//!     ctx.record_latency("process_message", start.elapsed().as_micros() as u64);
//!     ctx.record_throughput("messages", 1);
//!
//!     // Emit alert if needed
//!     if start.elapsed().as_millis() > 100 {
//!         ctx.emit_alert(KernelAlert::high_latency("Slow message processing", 100));
//!     }
//! }
//! ```

use crate::domain::Domain;
use crate::hlc::{HlcClock, HlcTimestamp};
use crate::message::MessageEnvelope;
use crate::types::{BlockId, Dim3, FenceScope, GlobalThreadId, MemoryOrder, ThreadId, WarpId};
use tokio::sync::mpsc;

// ============================================================================
// Metrics Types (FR-2)
// ============================================================================

/// Type of metric entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Latency measurement in microseconds.
    Latency,
    /// Throughput measurement (count per time period).
    Throughput,
    /// Counter increment.
    Counter,
    /// Gauge value (absolute measurement).
    Gauge,
}

/// A single metrics entry recorded by the kernel.
#[derive(Debug, Clone)]
pub struct MetricsEntry {
    /// Operation name (e.g., "process_order", "validate_tx").
    pub operation: String,
    /// Type of metric.
    pub metric_type: MetricType,
    /// Metric value.
    pub value: u64,
    /// When this metric was recorded.
    pub timestamp: HlcTimestamp,
    /// Kernel ID that recorded this metric.
    pub kernel_id: u64,
    /// Domain of the kernel (if any).
    pub domain: Option<Domain>,
}

/// Buffer for collecting metrics within a kernel context.
#[derive(Debug)]
pub struct ContextMetricsBuffer {
    entries: Vec<MetricsEntry>,
    capacity: usize,
}

impl ContextMetricsBuffer {
    /// Create a new metrics buffer with specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity.min(1024)), // Cap allocation
            capacity,
        }
    }

    /// Record a new metrics entry.
    pub fn record(&mut self, entry: MetricsEntry) {
        if self.entries.len() < self.capacity {
            self.entries.push(entry);
        }
        // If full, oldest entries are silently dropped
        // (ring buffer behavior could be added for production)
    }

    /// Drain all buffered entries.
    pub fn drain(&mut self) -> Vec<MetricsEntry> {
        std::mem::take(&mut self.entries)
    }

    /// Check if buffer is full.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    /// Get current entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for ContextMetricsBuffer {
    fn default() -> Self {
        Self::new(256)
    }
}

// ============================================================================
// Alert Types (FR-2)
// ============================================================================

/// Severity level for kernel alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational alert.
    Info = 0,
    /// Warning - potential issue.
    Warning = 1,
    /// Error - operation failed.
    Error = 2,
    /// Critical - system-level issue.
    Critical = 3,
}

/// Type of kernel alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelAlertType {
    /// High latency detected.
    HighLatency,
    /// Queue approaching capacity.
    QueuePressure,
    /// Memory pressure detected.
    MemoryPressure,
    /// Processing error occurred.
    ProcessingError,
    /// Domain-specific alert (custom type code).
    DomainAlert(u32),
    /// Custom application alert (custom type code).
    Custom(u32),
}

/// Routing destination for alerts.
#[derive(Debug, Clone, Copy, Default)]
pub enum AlertRouting {
    /// Route to host only (default).
    #[default]
    Host,
    /// Route to specific kernel via K2K.
    Kernel(u64),
    /// Broadcast to all kernels in same domain.
    Domain,
    /// Route to external monitoring system (via host bridge).
    External,
}

/// An alert emitted from a kernel.
#[derive(Debug, Clone)]
pub struct KernelAlert {
    /// Severity level.
    pub severity: AlertSeverity,
    /// Alert type.
    pub alert_type: KernelAlertType,
    /// Human-readable message.
    pub message: String,
    /// Source kernel ID.
    pub source_kernel: u64,
    /// Source domain (if applicable).
    pub source_domain: Option<Domain>,
    /// When this alert was created.
    pub timestamp: HlcTimestamp,
    /// Routing destination.
    pub routing: AlertRouting,
}

impl KernelAlert {
    /// Create a new alert.
    pub fn new(
        severity: AlertSeverity,
        alert_type: KernelAlertType,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            alert_type,
            message: message.into(),
            source_kernel: 0,
            source_domain: None,
            timestamp: HlcTimestamp::zero(),
            routing: AlertRouting::default(),
        }
    }

    /// Create a high latency alert.
    pub fn high_latency(message: impl Into<String>, latency_us: u64) -> Self {
        Self::new(
            AlertSeverity::Warning,
            KernelAlertType::HighLatency,
            format!("{} ({}µs)", message.into(), latency_us),
        )
    }

    /// Create a processing error alert.
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(
            AlertSeverity::Error,
            KernelAlertType::ProcessingError,
            message,
        )
    }

    /// Create a queue pressure warning.
    pub fn queue_pressure(message: impl Into<String>, utilization_pct: u32) -> Self {
        Self::new(
            AlertSeverity::Warning,
            KernelAlertType::QueuePressure,
            format!("{} ({}% full)", message.into(), utilization_pct),
        )
    }

    /// Set routing destination.
    pub fn with_routing(mut self, routing: AlertRouting) -> Self {
        self.routing = routing;
        self
    }
}

// ============================================================================
// RingContext
// ============================================================================

/// GPU intrinsics facade for kernel handlers.
///
/// This struct provides access to GPU-specific operations like thread
/// identification, synchronization, and atomic operations. The actual
/// implementation varies by backend.
///
/// # Lifetime
///
/// The context is borrowed for the duration of the kernel handler execution.
pub struct RingContext<'a> {
    /// Thread identity within block.
    pub thread_id: ThreadId,
    /// Block identity within grid.
    pub block_id: BlockId,
    /// Block dimensions.
    pub block_dim: Dim3,
    /// Grid dimensions.
    pub grid_dim: Dim3,
    /// HLC clock instance.
    clock: &'a HlcClock,
    /// Kernel ID.
    kernel_id: u64,
    /// Backend implementation.
    backend: ContextBackend,
    /// Domain this kernel operates in (FR-2).
    domain: Option<Domain>,
    /// Metrics buffer for this context (FR-2).
    metrics_buffer: ContextMetricsBuffer,
    /// Alert sender channel (FR-2).
    alert_sender: Option<mpsc::UnboundedSender<KernelAlert>>,
}

/// Backend-specific context implementation.
#[derive(Debug, Clone)]
pub enum ContextBackend {
    /// CPU backend (for testing).
    Cpu,
    /// CUDA backend.
    Cuda,
    /// Metal backend.
    Metal,
    /// WebGPU backend.
    Wgpu,
}

impl<'a> RingContext<'a> {
    /// Create a new context with basic configuration.
    ///
    /// For domain-aware contexts, use `new_with_options` instead.
    pub fn new(
        thread_id: ThreadId,
        block_id: BlockId,
        block_dim: Dim3,
        grid_dim: Dim3,
        clock: &'a HlcClock,
        kernel_id: u64,
        backend: ContextBackend,
    ) -> Self {
        Self {
            thread_id,
            block_id,
            block_dim,
            grid_dim,
            clock,
            kernel_id,
            backend,
            domain: None,
            metrics_buffer: ContextMetricsBuffer::default(),
            alert_sender: None,
        }
    }

    /// Create a new context with full configuration (FR-2).
    ///
    /// # Arguments
    ///
    /// * `thread_id` - Thread identity within block
    /// * `block_id` - Block identity within grid
    /// * `block_dim` - Block dimensions
    /// * `grid_dim` - Grid dimensions
    /// * `clock` - HLC clock instance
    /// * `kernel_id` - Unique kernel identifier
    /// * `backend` - Backend implementation
    /// * `domain` - Optional domain for this kernel
    /// * `metrics_capacity` - Metrics buffer capacity
    /// * `alert_sender` - Optional channel for emitting alerts
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_options(
        thread_id: ThreadId,
        block_id: BlockId,
        block_dim: Dim3,
        grid_dim: Dim3,
        clock: &'a HlcClock,
        kernel_id: u64,
        backend: ContextBackend,
        domain: Option<Domain>,
        metrics_capacity: usize,
        alert_sender: Option<mpsc::UnboundedSender<KernelAlert>>,
    ) -> Self {
        Self {
            thread_id,
            block_id,
            block_dim,
            grid_dim,
            clock,
            kernel_id,
            backend,
            domain,
            metrics_buffer: ContextMetricsBuffer::new(metrics_capacity),
            alert_sender,
        }
    }

    // === Thread Identity ===

    /// Get thread ID within block.
    #[inline]
    pub fn thread_id(&self) -> ThreadId {
        self.thread_id
    }

    /// Get block ID within grid.
    #[inline]
    pub fn block_id(&self) -> BlockId {
        self.block_id
    }

    /// Get global thread ID across all blocks.
    #[inline]
    pub fn global_thread_id(&self) -> GlobalThreadId {
        GlobalThreadId::from_block_thread(self.block_id, self.thread_id, self.block_dim)
    }

    /// Get warp ID within block.
    #[inline]
    pub fn warp_id(&self) -> WarpId {
        let linear = self
            .thread_id
            .linear_for_dim(self.block_dim.x, self.block_dim.y);
        WarpId::from_thread_linear(linear)
    }

    /// Get lane ID within warp (0-31).
    #[inline]
    pub fn lane_id(&self) -> u32 {
        let linear = self
            .thread_id
            .linear_for_dim(self.block_dim.x, self.block_dim.y);
        WarpId::lane_id(linear)
    }

    /// Get block dimensions.
    #[inline]
    pub fn block_dim(&self) -> Dim3 {
        self.block_dim
    }

    /// Get grid dimensions.
    #[inline]
    pub fn grid_dim(&self) -> Dim3 {
        self.grid_dim
    }

    /// Get kernel ID.
    #[inline]
    pub fn kernel_id(&self) -> u64 {
        self.kernel_id
    }

    // === Synchronization ===

    /// Synchronize all threads in the block.
    ///
    /// All threads in the block must reach this barrier before any
    /// thread can proceed past it.
    #[inline]
    pub fn sync_threads(&self) {
        match self.backend {
            ContextBackend::Cpu => {
                // CPU: no-op (single-threaded simulation)
            }
            _ => {
                // GPU backends would call __syncthreads() or equivalent
                // Placeholder for actual implementation
            }
        }
    }

    /// Synchronize all threads in the grid (cooperative groups).
    ///
    /// Requires cooperative kernel launch support.
    #[inline]
    pub fn sync_grid(&self) {
        match self.backend {
            ContextBackend::Cpu => {
                // CPU: no-op
            }
            _ => {
                // GPU backends would call cooperative grid sync
            }
        }
    }

    /// Synchronize threads within a warp.
    #[inline]
    pub fn sync_warp(&self) {
        match self.backend {
            ContextBackend::Cpu => {
                // CPU: no-op
            }
            _ => {
                // GPU backends would call __syncwarp()
            }
        }
    }

    // === Memory Fencing ===

    /// Memory fence at the specified scope.
    #[inline]
    pub fn thread_fence(&self, scope: FenceScope) {
        match (self.backend.clone(), scope) {
            (ContextBackend::Cpu, _) => {
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            }
            _ => {
                // GPU backends would call appropriate fence intrinsic
            }
        }
    }

    /// Thread-scope fence (compiler barrier).
    #[inline]
    pub fn fence_thread(&self) {
        self.thread_fence(FenceScope::Thread);
    }

    /// Block-scope fence.
    #[inline]
    pub fn fence_block(&self) {
        self.thread_fence(FenceScope::Block);
    }

    /// Device-scope fence.
    #[inline]
    pub fn fence_device(&self) {
        self.thread_fence(FenceScope::Device);
    }

    /// System-scope fence (CPU+GPU visible).
    #[inline]
    pub fn fence_system(&self) {
        self.thread_fence(FenceScope::System);
    }

    // === HLC Operations ===

    /// Get current HLC timestamp.
    #[inline]
    pub fn now(&self) -> HlcTimestamp {
        self.clock.now()
    }

    /// Generate a new HLC timestamp (advances clock).
    #[inline]
    pub fn tick(&self) -> HlcTimestamp {
        self.clock.tick()
    }

    /// Update clock with received timestamp.
    #[inline]
    pub fn update_clock(&self, received: &HlcTimestamp) -> crate::error::Result<HlcTimestamp> {
        self.clock.update(received)
    }

    // === Atomic Operations ===

    /// Atomic add and return old value.
    #[inline]
    pub fn atomic_add(
        &self,
        ptr: &std::sync::atomic::AtomicU64,
        val: u64,
        order: MemoryOrder,
    ) -> u64 {
        let ordering = match order {
            MemoryOrder::Relaxed => std::sync::atomic::Ordering::Relaxed,
            MemoryOrder::Acquire => std::sync::atomic::Ordering::Acquire,
            MemoryOrder::Release => std::sync::atomic::Ordering::Release,
            MemoryOrder::AcquireRelease => std::sync::atomic::Ordering::AcqRel,
            MemoryOrder::SeqCst => std::sync::atomic::Ordering::SeqCst,
        };
        ptr.fetch_add(val, ordering)
    }

    /// Atomic compare-and-swap.
    #[inline]
    pub fn atomic_cas(
        &self,
        ptr: &std::sync::atomic::AtomicU64,
        expected: u64,
        desired: u64,
        success: MemoryOrder,
        failure: MemoryOrder,
    ) -> Result<u64, u64> {
        let success_ord = match success {
            MemoryOrder::Relaxed => std::sync::atomic::Ordering::Relaxed,
            MemoryOrder::Acquire => std::sync::atomic::Ordering::Acquire,
            MemoryOrder::Release => std::sync::atomic::Ordering::Release,
            MemoryOrder::AcquireRelease => std::sync::atomic::Ordering::AcqRel,
            MemoryOrder::SeqCst => std::sync::atomic::Ordering::SeqCst,
        };
        let failure_ord = match failure {
            MemoryOrder::Relaxed => std::sync::atomic::Ordering::Relaxed,
            MemoryOrder::Acquire => std::sync::atomic::Ordering::Acquire,
            MemoryOrder::Release => std::sync::atomic::Ordering::Release,
            MemoryOrder::AcquireRelease => std::sync::atomic::Ordering::AcqRel,
            MemoryOrder::SeqCst => std::sync::atomic::Ordering::SeqCst,
        };
        ptr.compare_exchange(expected, desired, success_ord, failure_ord)
    }

    /// Atomic exchange.
    #[inline]
    pub fn atomic_exchange(
        &self,
        ptr: &std::sync::atomic::AtomicU64,
        val: u64,
        order: MemoryOrder,
    ) -> u64 {
        let ordering = match order {
            MemoryOrder::Relaxed => std::sync::atomic::Ordering::Relaxed,
            MemoryOrder::Acquire => std::sync::atomic::Ordering::Acquire,
            MemoryOrder::Release => std::sync::atomic::Ordering::Release,
            MemoryOrder::AcquireRelease => std::sync::atomic::Ordering::AcqRel,
            MemoryOrder::SeqCst => std::sync::atomic::Ordering::SeqCst,
        };
        ptr.swap(val, ordering)
    }

    // === Warp Primitives ===

    /// Warp shuffle - get value from another lane.
    ///
    /// Returns the value from the specified source lane.
    #[inline]
    pub fn warp_shuffle<T: Copy>(&self, value: T, src_lane: u32) -> T {
        match self.backend {
            ContextBackend::Cpu => {
                // CPU: just return own value (no other lanes)
                let _ = src_lane;
                value
            }
            _ => {
                // GPU would use __shfl_sync()
                let _ = src_lane;
                value
            }
        }
    }

    /// Warp shuffle down - get value from lane + delta.
    #[inline]
    pub fn warp_shuffle_down<T: Copy>(&self, value: T, delta: u32) -> T {
        self.warp_shuffle(value, self.lane_id().saturating_add(delta))
    }

    /// Warp shuffle up - get value from lane - delta.
    #[inline]
    pub fn warp_shuffle_up<T: Copy>(&self, value: T, delta: u32) -> T {
        self.warp_shuffle(value, self.lane_id().saturating_sub(delta))
    }

    /// Warp shuffle XOR - get value from lane XOR mask.
    #[inline]
    pub fn warp_shuffle_xor<T: Copy>(&self, value: T, mask: u32) -> T {
        self.warp_shuffle(value, self.lane_id() ^ mask)
    }

    /// Warp ballot - get bitmask of lanes where predicate is true.
    #[inline]
    pub fn warp_ballot(&self, predicate: bool) -> u32 {
        match self.backend {
            ContextBackend::Cpu => {
                // CPU: single thread, return 1 or 0
                if predicate {
                    1
                } else {
                    0
                }
            }
            _ => {
                // GPU would use __ballot_sync()
                if predicate {
                    1 << self.lane_id()
                } else {
                    0
                }
            }
        }
    }

    /// Warp all - check if predicate is true for all lanes.
    #[inline]
    pub fn warp_all(&self, predicate: bool) -> bool {
        match self.backend {
            ContextBackend::Cpu => predicate,
            _ => {
                // GPU would use __all_sync()
                predicate
            }
        }
    }

    /// Warp any - check if predicate is true for any lane.
    #[inline]
    pub fn warp_any(&self, predicate: bool) -> bool {
        match self.backend {
            ContextBackend::Cpu => predicate,
            _ => {
                // GPU would use __any_sync()
                predicate
            }
        }
    }

    // === K2K Messaging ===

    /// Send message to another kernel (K2K).
    ///
    /// This is a placeholder; actual implementation requires runtime support.
    #[inline]
    pub fn k2k_send(
        &self,
        _target_kernel: u64,
        _envelope: &MessageEnvelope,
    ) -> crate::error::Result<()> {
        // K2K messaging requires runtime bridge support
        Err(crate::error::RingKernelError::NotSupported(
            "K2K messaging requires runtime".to_string(),
        ))
    }

    /// Try to receive message from K2K queue.
    #[inline]
    pub fn k2k_try_recv(&self) -> crate::error::Result<MessageEnvelope> {
        // K2K messaging requires runtime bridge support
        Err(crate::error::RingKernelError::NotSupported(
            "K2K messaging requires runtime".to_string(),
        ))
    }

    // === Domain Operations (FR-2) ===

    /// Get the domain this kernel operates in.
    ///
    /// Returns `None` if no domain was configured.
    #[inline]
    pub fn domain(&self) -> Option<&Domain> {
        self.domain.as_ref()
    }

    /// Set the domain for this kernel context.
    ///
    /// This allows runtime domain assignment for kernels that process
    /// messages from multiple domains.
    #[inline]
    pub fn set_domain(&mut self, domain: Domain) {
        self.domain = Some(domain);
    }

    /// Clear the domain setting.
    #[inline]
    pub fn clear_domain(&mut self) {
        self.domain = None;
    }

    // === Metrics Operations (FR-2) ===

    /// Record a latency metric in microseconds.
    ///
    /// # Arguments
    ///
    /// * `operation` - Name of the operation (e.g., "process_order")
    /// * `latency_us` - Latency in microseconds
    ///
    /// # Example
    ///
    /// ```ignore
    /// let start = std::time::Instant::now();
    /// // ... process message ...
    /// ctx.record_latency("process", start.elapsed().as_micros() as u64);
    /// ```
    pub fn record_latency(&mut self, operation: &str, latency_us: u64) {
        let entry = MetricsEntry {
            operation: operation.to_string(),
            metric_type: MetricType::Latency,
            value: latency_us,
            timestamp: self.clock.now(),
            kernel_id: self.kernel_id,
            domain: self.domain,
        };
        self.metrics_buffer.record(entry);
    }

    /// Record a throughput metric (count per time period).
    ///
    /// # Arguments
    ///
    /// * `operation` - Name of the operation (e.g., "messages_processed")
    /// * `count` - Number of items processed
    pub fn record_throughput(&mut self, operation: &str, count: u64) {
        let entry = MetricsEntry {
            operation: operation.to_string(),
            metric_type: MetricType::Throughput,
            value: count,
            timestamp: self.clock.now(),
            kernel_id: self.kernel_id,
            domain: self.domain,
        };
        self.metrics_buffer.record(entry);
    }

    /// Record a counter increment.
    ///
    /// Counters are monotonically increasing values (e.g., total orders received).
    pub fn record_counter(&mut self, operation: &str, increment: u64) {
        let entry = MetricsEntry {
            operation: operation.to_string(),
            metric_type: MetricType::Counter,
            value: increment,
            timestamp: self.clock.now(),
            kernel_id: self.kernel_id,
            domain: self.domain,
        };
        self.metrics_buffer.record(entry);
    }

    /// Record a gauge value (absolute measurement).
    ///
    /// Gauges represent point-in-time values that can go up or down
    /// (e.g., queue depth, memory usage).
    pub fn record_gauge(&mut self, operation: &str, value: u64) {
        let entry = MetricsEntry {
            operation: operation.to_string(),
            metric_type: MetricType::Gauge,
            value,
            timestamp: self.clock.now(),
            kernel_id: self.kernel_id,
            domain: self.domain,
        };
        self.metrics_buffer.record(entry);
    }

    /// Flush and return all buffered metrics.
    ///
    /// After calling this method, the metrics buffer will be empty.
    /// This is typically called by the runtime when transferring metrics
    /// to the host telemetry pipeline.
    pub fn flush_metrics(&mut self) -> Vec<MetricsEntry> {
        self.metrics_buffer.drain()
    }

    /// Get the number of buffered metrics entries.
    pub fn metrics_count(&self) -> usize {
        self.metrics_buffer.len()
    }

    /// Check if the metrics buffer is full.
    pub fn metrics_buffer_full(&self) -> bool {
        self.metrics_buffer.is_full()
    }

    // === Alert Operations (FR-2) ===

    /// Emit an alert from this kernel.
    ///
    /// Alerts are sent to the configured alert channel (if any).
    /// The alert is enriched with kernel ID, domain, and timestamp.
    ///
    /// # Arguments
    ///
    /// * `alert` - The alert to emit (can be created via `KernelAlert::new()` helpers)
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.emit_alert(KernelAlert::high_latency("Slow processing", 500));
    /// ctx.emit_alert(KernelAlert::error("Processing failed"));
    /// ```
    pub fn emit_alert(&self, alert: impl Into<KernelAlert>) {
        if let Some(ref sender) = self.alert_sender {
            let mut alert = alert.into();
            // Enrich alert with context info
            alert.source_kernel = self.kernel_id;
            alert.source_domain = self.domain;
            alert.timestamp = self.clock.now();
            // Send (ignore errors - fire and forget)
            let _ = sender.send(alert);
        }
    }

    /// Check if alert sending is configured.
    #[inline]
    pub fn has_alert_channel(&self) -> bool {
        self.alert_sender.is_some()
    }

    /// Emit a high-latency alert if latency exceeds threshold.
    ///
    /// Convenience method that only emits an alert when latency exceeds
    /// the specified threshold.
    ///
    /// # Arguments
    ///
    /// * `operation` - Name of the operation
    /// * `latency_us` - Actual latency in microseconds
    /// * `threshold_us` - Threshold above which to emit alert
    pub fn alert_if_slow(&self, operation: &str, latency_us: u64, threshold_us: u64) {
        if latency_us > threshold_us {
            self.emit_alert(KernelAlert::high_latency(
                format!("{} exceeded threshold", operation),
                latency_us,
            ));
        }
    }
}

impl<'a> std::fmt::Debug for RingContext<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingContext")
            .field("thread_id", &self.thread_id)
            .field("block_id", &self.block_id)
            .field("block_dim", &self.block_dim)
            .field("grid_dim", &self.grid_dim)
            .field("kernel_id", &self.kernel_id)
            .field("backend", &self.backend)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_context(clock: &HlcClock) -> RingContext<'_> {
        RingContext::new(
            ThreadId::new_1d(0),
            BlockId::new_1d(0),
            Dim3::new_1d(256),
            Dim3::new_1d(1),
            clock,
            1,
            ContextBackend::Cpu,
        )
    }

    #[test]
    fn test_thread_identity() {
        let clock = HlcClock::new(1);
        let ctx = make_test_context(&clock);

        assert_eq!(ctx.thread_id().x, 0);
        assert_eq!(ctx.block_id().x, 0);
        assert_eq!(ctx.global_thread_id().x, 0);
    }

    #[test]
    fn test_warp_id() {
        let clock = HlcClock::new(1);
        let ctx = RingContext::new(
            ThreadId::new_1d(35), // Thread 35 is in warp 1, lane 3
            BlockId::new_1d(0),
            Dim3::new_1d(256),
            Dim3::new_1d(1),
            &clock,
            1,
            ContextBackend::Cpu,
        );

        assert_eq!(ctx.warp_id().0, 1);
        assert_eq!(ctx.lane_id(), 3);
    }

    #[test]
    fn test_hlc_operations() {
        let clock = HlcClock::new(1);
        let ctx = make_test_context(&clock);

        let ts1 = ctx.now();
        let ts2 = ctx.tick();
        assert!(ts2 >= ts1);
    }

    #[test]
    fn test_warp_ballot_cpu() {
        let clock = HlcClock::new(1);
        let ctx = make_test_context(&clock);

        assert_eq!(ctx.warp_ballot(true), 1);
        assert_eq!(ctx.warp_ballot(false), 0);
    }

    // === FR-2 Tests ===

    #[test]
    fn test_domain_operations() {
        let clock = HlcClock::new(1);
        let mut ctx = make_test_context(&clock);

        // Initially no domain
        assert!(ctx.domain().is_none());

        // Set domain
        ctx.set_domain(Domain::OrderMatching);
        assert_eq!(ctx.domain(), Some(&Domain::OrderMatching));

        // Clear domain
        ctx.clear_domain();
        assert!(ctx.domain().is_none());
    }

    #[test]
    fn test_context_with_domain() {
        let clock = HlcClock::new(1);
        let ctx = RingContext::new_with_options(
            ThreadId::new_1d(0),
            BlockId::new_1d(0),
            Dim3::new_1d(256),
            Dim3::new_1d(1),
            &clock,
            42,
            ContextBackend::Cpu,
            Some(Domain::RiskManagement),
            128,
            None,
        );

        assert_eq!(ctx.domain(), Some(&Domain::RiskManagement));
        assert_eq!(ctx.kernel_id(), 42);
    }

    #[test]
    fn test_metrics_buffer() {
        let mut buffer = ContextMetricsBuffer::new(3);

        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        assert_eq!(buffer.len(), 0);

        let entry = MetricsEntry {
            operation: "test".to_string(),
            metric_type: MetricType::Latency,
            value: 100,
            timestamp: HlcTimestamp::zero(),
            kernel_id: 1,
            domain: None,
        };

        buffer.record(entry.clone());
        assert_eq!(buffer.len(), 1);

        buffer.record(entry.clone());
        buffer.record(entry.clone());
        assert!(buffer.is_full());

        // Drain returns all entries
        let entries = buffer.drain();
        assert_eq!(entries.len(), 3);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_record_metrics() {
        let clock = HlcClock::new(1);
        let mut ctx = RingContext::new_with_options(
            ThreadId::new_1d(0),
            BlockId::new_1d(0),
            Dim3::new_1d(256),
            Dim3::new_1d(1),
            &clock,
            100,
            ContextBackend::Cpu,
            Some(Domain::Compliance),
            256,
            None,
        );

        ctx.record_latency("process_order", 500);
        ctx.record_throughput("orders_per_sec", 1000);
        ctx.record_counter("total_orders", 1);
        ctx.record_gauge("queue_depth", 42);

        assert_eq!(ctx.metrics_count(), 4);

        let metrics = ctx.flush_metrics();
        assert_eq!(metrics.len(), 4);

        // Verify entries
        assert_eq!(metrics[0].operation, "process_order");
        assert_eq!(metrics[0].metric_type, MetricType::Latency);
        assert_eq!(metrics[0].value, 500);
        assert_eq!(metrics[0].kernel_id, 100);
        assert_eq!(metrics[0].domain, Some(Domain::Compliance));

        assert_eq!(metrics[1].metric_type, MetricType::Throughput);
        assert_eq!(metrics[2].metric_type, MetricType::Counter);
        assert_eq!(metrics[3].metric_type, MetricType::Gauge);
        assert_eq!(metrics[3].value, 42);

        // After flush, buffer is empty
        assert_eq!(ctx.metrics_count(), 0);
    }

    #[test]
    fn test_kernel_alert_constructors() {
        let alert = KernelAlert::high_latency("Slow", 500);
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.alert_type, KernelAlertType::HighLatency);
        assert!(alert.message.contains("500µs"));

        let alert = KernelAlert::error("Failed");
        assert_eq!(alert.severity, AlertSeverity::Error);
        assert_eq!(alert.alert_type, KernelAlertType::ProcessingError);

        let alert = KernelAlert::queue_pressure("Input queue", 85);
        assert_eq!(alert.alert_type, KernelAlertType::QueuePressure);
        assert!(alert.message.contains("85%"));

        let alert = KernelAlert::new(
            AlertSeverity::Critical,
            KernelAlertType::Custom(42),
            "Custom alert",
        )
        .with_routing(AlertRouting::External);
        assert_eq!(alert.severity, AlertSeverity::Critical);
        assert!(matches!(alert.routing, AlertRouting::External));
    }

    #[test]
    fn test_emit_alert_with_channel() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let clock = HlcClock::new(1);
        let ctx = RingContext::new_with_options(
            ThreadId::new_1d(0),
            BlockId::new_1d(0),
            Dim3::new_1d(256),
            Dim3::new_1d(1),
            &clock,
            42,
            ContextBackend::Cpu,
            Some(Domain::OrderMatching),
            256,
            Some(tx),
        );

        assert!(ctx.has_alert_channel());
        ctx.emit_alert(KernelAlert::error("Test error"));

        // Receive the alert
        let alert = rx.try_recv().expect("Should receive alert");
        assert_eq!(alert.source_kernel, 42);
        assert_eq!(alert.source_domain, Some(Domain::OrderMatching));
        assert_eq!(alert.alert_type, KernelAlertType::ProcessingError);
    }

    #[test]
    fn test_emit_alert_without_channel() {
        let clock = HlcClock::new(1);
        let ctx = make_test_context(&clock);

        assert!(!ctx.has_alert_channel());
        // Should not panic when no channel configured
        ctx.emit_alert(KernelAlert::error("No-op"));
    }

    #[test]
    fn test_alert_if_slow() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let clock = HlcClock::new(1);
        let ctx = RingContext::new_with_options(
            ThreadId::new_1d(0),
            BlockId::new_1d(0),
            Dim3::new_1d(256),
            Dim3::new_1d(1),
            &clock,
            1,
            ContextBackend::Cpu,
            None,
            256,
            Some(tx),
        );

        // Below threshold - no alert
        ctx.alert_if_slow("fast_op", 50, 100);
        assert!(rx.try_recv().is_err());

        // Above threshold - alert sent
        ctx.alert_if_slow("slow_op", 150, 100);
        let alert = rx.try_recv().expect("Should receive alert");
        assert!(alert.message.contains("slow_op"));
        assert!(alert.message.contains("150µs"));
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Error);
        assert!(AlertSeverity::Error < AlertSeverity::Critical);
    }
}
