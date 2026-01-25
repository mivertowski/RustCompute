//! Lock-free message queue implementation.
//!
//! This module provides the core message queue abstraction used for
//! communication between host and GPU kernels. The queue uses a ring
//! buffer design with atomic operations for lock-free access.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Result, RingKernelError};
use crate::message::MessageEnvelope;

/// Statistics for a message queue.
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    /// Total messages enqueued.
    pub enqueued: u64,
    /// Total messages dequeued.
    pub dequeued: u64,
    /// Messages dropped due to full queue.
    pub dropped: u64,
    /// Current queue depth.
    pub depth: u64,
    /// Maximum queue depth observed.
    pub max_depth: u64,
}

/// Trait for message queue implementations.
///
/// Message queues provide lock-free FIFO communication between
/// producers (host or other kernels) and consumers (GPU kernels).
pub trait MessageQueue: Send + Sync {
    /// Get the queue capacity.
    fn capacity(&self) -> usize;

    /// Get current queue size.
    fn len(&self) -> usize;

    /// Check if queue is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if queue is full.
    fn is_full(&self) -> bool {
        self.len() >= self.capacity()
    }

    /// Try to enqueue a message envelope.
    fn try_enqueue(&self, envelope: MessageEnvelope) -> Result<()>;

    /// Try to dequeue a message envelope.
    fn try_dequeue(&self) -> Result<MessageEnvelope>;

    /// Get queue statistics.
    fn stats(&self) -> QueueStats;

    /// Reset queue statistics.
    fn reset_stats(&self);
}

/// Single-producer single-consumer lock-free ring buffer.
///
/// This implementation is optimized for the common case of one
/// producer (host) and one consumer (GPU kernel).
pub struct SpscQueue {
    /// Ring buffer storage.
    buffer: Vec<parking_lot::Mutex<Option<MessageEnvelope>>>,
    /// Capacity (power of 2).
    capacity: usize,
    /// Mask for index wrapping.
    mask: usize,
    /// Head pointer (producer writes here).
    head: AtomicU64,
    /// Tail pointer (consumer reads from here).
    tail: AtomicU64,
    /// Statistics.
    stats: QueueStatsInner,
}

/// Internal statistics with atomics.
struct QueueStatsInner {
    enqueued: AtomicU64,
    dequeued: AtomicU64,
    dropped: AtomicU64,
    max_depth: AtomicU64,
}

impl SpscQueue {
    /// Create a new queue with the given capacity.
    ///
    /// Capacity will be rounded up to the next power of 2.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let mask = capacity - 1;

        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(parking_lot::Mutex::new(None));
        }

        Self {
            buffer,
            capacity,
            mask,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            stats: QueueStatsInner {
                enqueued: AtomicU64::new(0),
                dequeued: AtomicU64::new(0),
                dropped: AtomicU64::new(0),
                max_depth: AtomicU64::new(0),
            },
        }
    }

    /// Get current depth.
    fn depth(&self) -> u64 {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    /// Update max depth statistic.
    fn update_max_depth(&self) {
        let depth = self.depth();
        let mut max = self.stats.max_depth.load(Ordering::Relaxed);
        while depth > max {
            match self.stats.max_depth.compare_exchange_weak(
                max,
                depth,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => max = current,
            }
        }
    }
}

impl MessageQueue for SpscQueue {
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.depth() as usize
    }

    fn try_enqueue(&self, envelope: MessageEnvelope) -> Result<()> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);

        // Check if full
        if head.wrapping_sub(tail) >= self.capacity as u64 {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
            return Err(RingKernelError::QueueFull {
                capacity: self.capacity,
            });
        }

        // Get slot
        let index = (head as usize) & self.mask;
        let mut slot = self.buffer[index].lock();
        *slot = Some(envelope);
        drop(slot);

        // Advance head
        self.head.store(head.wrapping_add(1), Ordering::Release);

        // Update stats
        self.stats.enqueued.fetch_add(1, Ordering::Relaxed);
        self.update_max_depth();

        Ok(())
    }

    fn try_dequeue(&self) -> Result<MessageEnvelope> {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);

        // Check if empty
        if head == tail {
            return Err(RingKernelError::QueueEmpty);
        }

        // Get slot
        let index = (tail as usize) & self.mask;
        let mut slot = self.buffer[index].lock();
        let envelope = slot.take().ok_or(RingKernelError::QueueEmpty)?;
        drop(slot);

        // Advance tail
        self.tail.store(tail.wrapping_add(1), Ordering::Release);

        // Update stats
        self.stats.dequeued.fetch_add(1, Ordering::Relaxed);

        Ok(envelope)
    }

    fn stats(&self) -> QueueStats {
        QueueStats {
            enqueued: self.stats.enqueued.load(Ordering::Relaxed),
            dequeued: self.stats.dequeued.load(Ordering::Relaxed),
            dropped: self.stats.dropped.load(Ordering::Relaxed),
            depth: self.depth(),
            max_depth: self.stats.max_depth.load(Ordering::Relaxed),
        }
    }

    fn reset_stats(&self) {
        self.stats.enqueued.store(0, Ordering::Relaxed);
        self.stats.dequeued.store(0, Ordering::Relaxed);
        self.stats.dropped.store(0, Ordering::Relaxed);
        self.stats.max_depth.store(0, Ordering::Relaxed);
    }
}

/// Multi-producer single-consumer lock-free queue.
///
/// This variant allows multiple producers (e.g., multiple host threads
/// or kernel-to-kernel messaging) to enqueue messages concurrently.
pub struct MpscQueue {
    /// Inner SPSC queue.
    inner: SpscQueue,
    /// Lock for producers.
    producer_lock: parking_lot::Mutex<()>,
}

impl MpscQueue {
    /// Create a new MPSC queue.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: SpscQueue::new(capacity),
            producer_lock: parking_lot::Mutex::new(()),
        }
    }
}

impl MessageQueue for MpscQueue {
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn try_enqueue(&self, envelope: MessageEnvelope) -> Result<()> {
        let _guard = self.producer_lock.lock();
        self.inner.try_enqueue(envelope)
    }

    fn try_dequeue(&self) -> Result<MessageEnvelope> {
        self.inner.try_dequeue()
    }

    fn stats(&self) -> QueueStats {
        self.inner.stats()
    }

    fn reset_stats(&self) {
        self.inner.reset_stats()
    }
}

/// Bounded queue with blocking operations.
pub struct BoundedQueue {
    /// Inner MPSC queue.
    inner: MpscQueue,
    /// Condvar for waiting on space.
    not_full: parking_lot::Condvar,
    /// Condvar for waiting on data.
    not_empty: parking_lot::Condvar,
    /// Mutex for condvar coordination.
    mutex: parking_lot::Mutex<()>,
}

impl BoundedQueue {
    /// Create a new bounded queue.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: MpscQueue::new(capacity),
            not_full: parking_lot::Condvar::new(),
            not_empty: parking_lot::Condvar::new(),
            mutex: parking_lot::Mutex::new(()),
        }
    }

    /// Blocking enqueue with timeout.
    pub fn enqueue_timeout(
        &self,
        envelope: MessageEnvelope,
        timeout: std::time::Duration,
    ) -> Result<()> {
        let deadline = std::time::Instant::now() + timeout;

        loop {
            match self.inner.try_enqueue(envelope.clone()) {
                Ok(()) => {
                    self.not_empty.notify_one();
                    return Ok(());
                }
                Err(RingKernelError::QueueFull { .. }) => {
                    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                    if remaining.is_zero() {
                        return Err(RingKernelError::Timeout(timeout));
                    }
                    let mut guard = self.mutex.lock();
                    let _ = self.not_full.wait_for(&mut guard, remaining);
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Blocking dequeue with timeout.
    pub fn dequeue_timeout(&self, timeout: std::time::Duration) -> Result<MessageEnvelope> {
        let deadline = std::time::Instant::now() + timeout;

        loop {
            match self.inner.try_dequeue() {
                Ok(envelope) => {
                    self.not_full.notify_one();
                    return Ok(envelope);
                }
                Err(RingKernelError::QueueEmpty) => {
                    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                    if remaining.is_zero() {
                        return Err(RingKernelError::Timeout(timeout));
                    }
                    let mut guard = self.mutex.lock();
                    let _ = self.not_empty.wait_for(&mut guard, remaining);
                }
                Err(e) => return Err(e),
            }
        }
    }
}

impl MessageQueue for BoundedQueue {
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn try_enqueue(&self, envelope: MessageEnvelope) -> Result<()> {
        let result = self.inner.try_enqueue(envelope);
        if result.is_ok() {
            self.not_empty.notify_one();
        }
        result
    }

    fn try_dequeue(&self) -> Result<MessageEnvelope> {
        let result = self.inner.try_dequeue();
        if result.is_ok() {
            self.not_full.notify_one();
        }
        result
    }

    fn stats(&self) -> QueueStats {
        self.inner.stats()
    }

    fn reset_stats(&self) {
        self.inner.reset_stats()
    }
}

// ============================================================================
// Queue Tiering Support
// ============================================================================

/// Queue capacity tiers for dynamic queue allocation.
///
/// Instead of dynamic resizing (which is complex for GPU memory),
/// we provide predefined tiers that can be selected based on expected load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum QueueTier {
    /// Small queues (256 messages) - low traffic, minimal memory.
    Small,
    /// Medium queues (1024 messages) - moderate traffic.
    #[default]
    Medium,
    /// Large queues (4096 messages) - high traffic.
    Large,
    /// Extra large queues (16384 messages) - very high traffic.
    ExtraLarge,
}

impl QueueTier {
    /// Get the capacity for this tier.
    pub fn capacity(&self) -> usize {
        match self {
            Self::Small => 256,
            Self::Medium => 1024,
            Self::Large => 4096,
            Self::ExtraLarge => 16384,
        }
    }

    /// Suggest a tier based on expected message rate.
    ///
    /// # Arguments
    ///
    /// * `messages_per_second` - Expected message throughput
    /// * `target_headroom_ms` - Desired buffer time in milliseconds
    ///
    /// # Returns
    ///
    /// Recommended tier based on traffic patterns.
    pub fn for_throughput(messages_per_second: u64, target_headroom_ms: u64) -> Self {
        let needed_capacity = (messages_per_second * target_headroom_ms) / 1000;

        if needed_capacity <= 256 {
            Self::Small
        } else if needed_capacity <= 1024 {
            Self::Medium
        } else if needed_capacity <= 4096 {
            Self::Large
        } else {
            Self::ExtraLarge
        }
    }

    /// Get the next tier up (for capacity planning).
    pub fn upgrade(&self) -> Self {
        match self {
            Self::Small => Self::Medium,
            Self::Medium => Self::Large,
            Self::Large => Self::ExtraLarge,
            Self::ExtraLarge => Self::ExtraLarge, // Already at max
        }
    }

    /// Get the tier below (for memory optimization).
    pub fn downgrade(&self) -> Self {
        match self {
            Self::Small => Self::Small, // Already at min
            Self::Medium => Self::Small,
            Self::Large => Self::Medium,
            Self::ExtraLarge => Self::Large,
        }
    }
}

/// Factory for creating appropriately-sized message queues.
///
/// # Example
///
/// ```ignore
/// use ringkernel_core::queue::{QueueFactory, QueueTier};
///
/// // Create a medium-sized MPSC queue
/// let queue = QueueFactory::create_mpsc(QueueTier::Medium);
///
/// // Create based on expected throughput
/// let queue = QueueFactory::create_for_throughput(10000, 100); // 10k msg/s, 100ms buffer
/// ```
pub struct QueueFactory;

impl QueueFactory {
    /// Create an SPSC queue with the specified tier.
    pub fn create_spsc(tier: QueueTier) -> SpscQueue {
        SpscQueue::new(tier.capacity())
    }

    /// Create an MPSC queue with the specified tier.
    pub fn create_mpsc(tier: QueueTier) -> MpscQueue {
        MpscQueue::new(tier.capacity())
    }

    /// Create a bounded queue with the specified tier.
    pub fn create_bounded(tier: QueueTier) -> BoundedQueue {
        BoundedQueue::new(tier.capacity())
    }

    /// Create a queue based on expected throughput.
    ///
    /// # Arguments
    ///
    /// * `messages_per_second` - Expected message throughput
    /// * `headroom_ms` - Desired buffer time in milliseconds
    pub fn create_for_throughput(
        messages_per_second: u64,
        headroom_ms: u64,
    ) -> Box<dyn MessageQueue> {
        let tier = QueueTier::for_throughput(messages_per_second, headroom_ms);
        Box::new(Self::create_mpsc(tier))
    }
}

/// Queue health status from monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueHealth {
    /// Queue utilization is healthy (< warning threshold).
    Healthy,
    /// Queue is approaching capacity (>= warning, < critical threshold).
    Warning,
    /// Queue is near capacity (>= critical threshold).
    Critical,
}

/// Monitor for queue health and utilization.
///
/// Provides real-time health checking for message queues without
/// dynamic resizing, allowing applications to take action when
/// queues approach capacity.
///
/// # Example
///
/// ```ignore
/// use ringkernel_core::queue::{QueueMonitor, SpscQueue, QueueHealth};
///
/// let queue = SpscQueue::new(1024);
/// let monitor = QueueMonitor::default();
///
/// // Check health periodically
/// match monitor.check(&queue) {
///     QueueHealth::Healthy => { /* normal operation */ }
///     QueueHealth::Warning => { /* consider throttling producers */ }
///     QueueHealth::Critical => { /* alert! take immediate action */ }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct QueueMonitor {
    /// Utilization threshold for warning (0.0 - 1.0).
    pub warning_threshold: f64,
    /// Utilization threshold for critical (0.0 - 1.0).
    pub critical_threshold: f64,
}

impl Default for QueueMonitor {
    fn default() -> Self {
        Self {
            warning_threshold: 0.75,  // 75%
            critical_threshold: 0.90, // 90%
        }
    }
}

impl QueueMonitor {
    /// Create a new queue monitor with custom thresholds.
    pub fn new(warning_threshold: f64, critical_threshold: f64) -> Self {
        Self {
            warning_threshold,
            critical_threshold,
        }
    }

    /// Check the health of a queue.
    pub fn check(&self, queue: &dyn MessageQueue) -> QueueHealth {
        let utilization = self.utilization(queue);

        if utilization >= self.critical_threshold {
            QueueHealth::Critical
        } else if utilization >= self.warning_threshold {
            QueueHealth::Warning
        } else {
            QueueHealth::Healthy
        }
    }

    /// Get the current utilization (0.0 - 1.0).
    pub fn utilization(&self, queue: &dyn MessageQueue) -> f64 {
        let capacity = queue.capacity();
        if capacity == 0 {
            return 0.0;
        }
        queue.len() as f64 / capacity as f64
    }

    /// Get current utilization percentage.
    pub fn utilization_percent(&self, queue: &dyn MessageQueue) -> f64 {
        self.utilization(queue) * 100.0
    }

    /// Suggest whether to upgrade the queue tier based on observed utilization.
    ///
    /// Returns `Some(QueueTier)` if upgrade is recommended, `None` otherwise.
    pub fn suggest_upgrade(
        &self,
        queue: &dyn MessageQueue,
        current_tier: QueueTier,
    ) -> Option<QueueTier> {
        let stats = queue.stats();
        let utilization = self.utilization(queue);

        // Upgrade if:
        // - Current utilization is at warning level
        // - Max observed depth is above critical threshold
        let max_util = if queue.capacity() > 0 {
            stats.max_depth as f64 / queue.capacity() as f64
        } else {
            0.0
        };

        if utilization >= self.warning_threshold || max_util >= self.critical_threshold {
            let upgraded = current_tier.upgrade();
            if upgraded != current_tier {
                return Some(upgraded);
            }
        }

        None
    }

    /// Check if queue has experienced drops.
    pub fn has_drops(&self, queue: &dyn MessageQueue) -> bool {
        queue.stats().dropped > 0
    }

    /// Get the drop rate (drops / total attempted enqueues).
    pub fn drop_rate(&self, queue: &dyn MessageQueue) -> f64 {
        let stats = queue.stats();
        let total_attempted = stats.enqueued + stats.dropped;
        if total_attempted == 0 {
            return 0.0;
        }
        stats.dropped as f64 / total_attempted as f64
    }
}

/// Comprehensive queue metrics snapshot.
#[derive(Debug, Clone)]
pub struct QueueMetrics {
    /// Queue health status.
    pub health: QueueHealth,
    /// Current utilization (0.0 - 1.0).
    pub utilization: f64,
    /// Queue statistics.
    pub stats: QueueStats,
    /// Current tier (if known).
    pub tier: Option<QueueTier>,
    /// Suggested tier upgrade (if recommended).
    pub suggested_upgrade: Option<QueueTier>,
}

impl QueueMetrics {
    /// Capture metrics from a queue.
    pub fn capture(
        queue: &dyn MessageQueue,
        monitor: &QueueMonitor,
        current_tier: Option<QueueTier>,
    ) -> Self {
        let health = monitor.check(queue);
        let utilization = monitor.utilization(queue);
        let stats = queue.stats();
        let suggested_upgrade = current_tier.and_then(|tier| monitor.suggest_upgrade(queue, tier));

        Self {
            health,
            utilization,
            stats,
            tier: current_tier,
            suggested_upgrade,
        }
    }
}

// ============================================================================
// Partitioned Queue
// ============================================================================

/// A partitioned queue for reduced contention with multiple producers.
///
/// Instead of a single queue with a lock, this uses multiple independent
/// partitions (SPSC queues) to reduce contention when many producers
/// are sending messages concurrently.
///
/// Producers are routed to partitions based on their source ID, ensuring
/// messages from the same source go to the same partition (preserving order).
///
/// # Example
///
/// ```ignore
/// use ringkernel_core::queue::{PartitionedQueue, QueueTier};
///
/// // Create 4 partitions with Medium tier capacity each
/// let queue = PartitionedQueue::new(4, QueueTier::Medium.capacity());
///
/// // Enqueue with source-based routing
/// queue.try_enqueue_from(source_id, envelope)?;
///
/// // Dequeue from any partition that has messages
/// if let Some(envelope) = queue.try_dequeue_any() {
///     // process message
/// }
/// ```
pub struct PartitionedQueue {
    /// Individual partition queues.
    partitions: Vec<SpscQueue>,
    /// Number of partitions.
    partition_count: usize,
    /// Round-robin dequeue index.
    dequeue_index: AtomicU64,
}

impl PartitionedQueue {
    /// Creates a new partitioned queue.
    ///
    /// # Arguments
    ///
    /// * `partition_count` - Number of partitions (should be power of 2 for efficiency)
    /// * `capacity_per_partition` - Capacity of each partition
    pub fn new(partition_count: usize, capacity_per_partition: usize) -> Self {
        let partition_count = partition_count.max(1).next_power_of_two();
        let partitions = (0..partition_count)
            .map(|_| SpscQueue::new(capacity_per_partition))
            .collect();

        Self {
            partitions,
            partition_count,
            dequeue_index: AtomicU64::new(0),
        }
    }

    /// Creates a partitioned queue with default settings.
    ///
    /// Uses 4 partitions with Medium tier capacity.
    pub fn with_defaults() -> Self {
        Self::new(4, QueueTier::Medium.capacity())
    }

    /// Creates a partitioned queue sized for high contention.
    ///
    /// Uses 8 partitions with Large tier capacity.
    pub fn for_high_contention() -> Self {
        Self::new(8, QueueTier::Large.capacity())
    }

    /// Returns the partition index for a given source ID.
    #[inline]
    pub fn partition_for(&self, source_id: u64) -> usize {
        (source_id as usize) & (self.partition_count - 1)
    }

    /// Returns the number of partitions.
    pub fn partition_count(&self) -> usize {
        self.partition_count
    }

    /// Returns the capacity per partition.
    pub fn capacity_per_partition(&self) -> usize {
        self.partitions.first().map_or(0, |p| p.capacity())
    }

    /// Total capacity across all partitions.
    pub fn total_capacity(&self) -> usize {
        self.capacity_per_partition() * self.partition_count
    }

    /// Total messages across all partitions.
    pub fn total_messages(&self) -> usize {
        self.partitions.iter().map(|p| p.len()).sum()
    }

    /// Enqueues a message to a partition based on source ID.
    ///
    /// Messages from the same source always go to the same partition,
    /// preserving ordering for that source.
    pub fn try_enqueue_from(&self, source_id: u64, envelope: MessageEnvelope) -> Result<()> {
        let partition = self.partition_for(source_id);
        self.partitions[partition].try_enqueue(envelope)
    }

    /// Enqueues a message using the envelope's source kernel ID.
    pub fn try_enqueue(&self, envelope: MessageEnvelope) -> Result<()> {
        let source_id = envelope.header.source_kernel;
        self.try_enqueue_from(source_id, envelope)
    }

    /// Tries to dequeue from a specific partition.
    pub fn try_dequeue_partition(&self, partition: usize) -> Result<MessageEnvelope> {
        if partition >= self.partition_count {
            return Err(RingKernelError::InvalidConfig(format!(
                "Invalid partition index: {} (max: {})",
                partition,
                self.partition_count - 1
            )));
        }
        self.partitions[partition].try_dequeue()
    }

    /// Tries to dequeue from any partition that has messages.
    ///
    /// Uses round-robin to fairly distribute dequeues across partitions.
    pub fn try_dequeue_any(&self) -> Option<MessageEnvelope> {
        let start_index = self.dequeue_index.fetch_add(1, Ordering::Relaxed) as usize;

        for i in 0..self.partition_count {
            let partition = (start_index + i) & (self.partition_count - 1);
            if let Ok(envelope) = self.partitions[partition].try_dequeue() {
                return Some(envelope);
            }
        }

        None
    }

    /// Returns statistics for a specific partition.
    pub fn partition_stats(&self, partition: usize) -> Option<QueueStats> {
        self.partitions.get(partition).map(|p| p.stats())
    }

    /// Returns aggregated statistics across all partitions.
    pub fn stats(&self) -> PartitionedQueueStats {
        let mut total = QueueStats::default();
        let mut partition_stats = Vec::with_capacity(self.partition_count);

        for partition in &self.partitions {
            let stats = partition.stats();
            total.enqueued += stats.enqueued;
            total.dequeued += stats.dequeued;
            total.dropped += stats.dropped;
            total.depth += stats.depth;
            if stats.max_depth > total.max_depth {
                total.max_depth = stats.max_depth;
            }
            partition_stats.push(stats);
        }

        PartitionedQueueStats {
            total,
            partition_stats,
            partition_count: self.partition_count,
        }
    }

    /// Resets statistics for all partitions.
    pub fn reset_stats(&self) {
        for partition in &self.partitions {
            partition.reset_stats();
        }
    }
}

/// Statistics for a partitioned queue.
#[derive(Debug, Clone)]
pub struct PartitionedQueueStats {
    /// Aggregated statistics.
    pub total: QueueStats,
    /// Per-partition statistics.
    pub partition_stats: Vec<QueueStats>,
    /// Number of partitions.
    pub partition_count: usize,
}

impl PartitionedQueueStats {
    /// Returns the load imbalance factor (max/avg).
    ///
    /// A value of 1.0 indicates perfect balance.
    /// Higher values indicate imbalance (some partitions have more messages).
    pub fn load_imbalance(&self) -> f64 {
        if self.partition_count == 0 {
            return 1.0;
        }

        let avg = self.total.depth as f64 / self.partition_count as f64;
        if avg == 0.0 {
            return 1.0;
        }

        let max = self
            .partition_stats
            .iter()
            .map(|s| s.depth)
            .max()
            .unwrap_or(0);
        max as f64 / avg
    }

    /// Returns the utilization of the most loaded partition.
    pub fn max_partition_utilization(&self, capacity_per_partition: usize) -> f64 {
        if capacity_per_partition == 0 {
            return 0.0;
        }

        let max = self
            .partition_stats
            .iter()
            .map(|s| s.depth)
            .max()
            .unwrap_or(0);
        max as f64 / capacity_per_partition as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hlc::HlcTimestamp;
    use crate::message::MessageHeader;

    fn make_envelope() -> MessageEnvelope {
        MessageEnvelope {
            header: MessageHeader::new(1, 0, 1, 8, HlcTimestamp::now(1)),
            payload: vec![1, 2, 3, 4, 5, 6, 7, 8],
        }
    }

    #[test]
    fn test_spsc_basic() {
        let queue = SpscQueue::new(16);

        assert!(queue.is_empty());
        assert!(!queue.is_full());

        let env = make_envelope();
        queue.try_enqueue(env).unwrap();

        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());

        let _ = queue.try_dequeue().unwrap();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_spsc_full() {
        let queue = SpscQueue::new(4);

        for _ in 0..4 {
            queue.try_enqueue(make_envelope()).unwrap();
        }

        assert!(queue.is_full());
        assert!(matches!(
            queue.try_enqueue(make_envelope()),
            Err(RingKernelError::QueueFull { .. })
        ));
    }

    #[test]
    fn test_spsc_stats() {
        let queue = SpscQueue::new(16);

        for _ in 0..10 {
            queue.try_enqueue(make_envelope()).unwrap();
        }

        for _ in 0..5 {
            let _ = queue.try_dequeue().unwrap();
        }

        let stats = queue.stats();
        assert_eq!(stats.enqueued, 10);
        assert_eq!(stats.dequeued, 5);
        assert_eq!(stats.depth, 5);
    }

    #[test]
    fn test_mpsc_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let queue = Arc::new(MpscQueue::new(1024));
        let mut handles = vec![];

        // Spawn multiple producers
        for _ in 0..4 {
            let q = Arc::clone(&queue);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    q.try_enqueue(make_envelope()).unwrap();
                }
            }));
        }

        // Wait for producers
        for h in handles {
            h.join().unwrap();
        }

        let stats = queue.stats();
        assert_eq!(stats.enqueued, 400);
    }

    #[test]
    fn test_bounded_timeout() {
        let queue = BoundedQueue::new(2);

        // Fill queue
        queue.try_enqueue(make_envelope()).unwrap();
        queue.try_enqueue(make_envelope()).unwrap();

        // Should timeout
        let result = queue.enqueue_timeout(make_envelope(), std::time::Duration::from_millis(10));
        assert!(matches!(result, Err(RingKernelError::Timeout(_))));
    }

    // ========================================================================
    // Queue Tiering Tests
    // ========================================================================

    #[test]
    fn test_queue_tier_capacities() {
        assert_eq!(QueueTier::Small.capacity(), 256);
        assert_eq!(QueueTier::Medium.capacity(), 1024);
        assert_eq!(QueueTier::Large.capacity(), 4096);
        assert_eq!(QueueTier::ExtraLarge.capacity(), 16384);
    }

    #[test]
    fn test_queue_tier_for_throughput() {
        // Low traffic - 1000 msg/s with 100ms buffer = 100 msgs needed
        assert_eq!(QueueTier::for_throughput(1000, 100), QueueTier::Small);

        // Medium traffic - 5000 msg/s with 100ms buffer = 500 msgs needed
        assert_eq!(QueueTier::for_throughput(5000, 100), QueueTier::Medium);

        // High traffic - 20000 msg/s with 100ms buffer = 2000 msgs needed
        assert_eq!(QueueTier::for_throughput(20000, 100), QueueTier::Large);

        // Very high traffic - 100000 msg/s with 100ms buffer = 10000 msgs needed
        assert_eq!(
            QueueTier::for_throughput(100000, 100),
            QueueTier::ExtraLarge
        );
    }

    #[test]
    fn test_queue_tier_upgrade_downgrade() {
        assert_eq!(QueueTier::Small.upgrade(), QueueTier::Medium);
        assert_eq!(QueueTier::Medium.upgrade(), QueueTier::Large);
        assert_eq!(QueueTier::Large.upgrade(), QueueTier::ExtraLarge);
        assert_eq!(QueueTier::ExtraLarge.upgrade(), QueueTier::ExtraLarge); // Max

        assert_eq!(QueueTier::Small.downgrade(), QueueTier::Small); // Min
        assert_eq!(QueueTier::Medium.downgrade(), QueueTier::Small);
        assert_eq!(QueueTier::Large.downgrade(), QueueTier::Medium);
        assert_eq!(QueueTier::ExtraLarge.downgrade(), QueueTier::Large);
    }

    #[test]
    fn test_queue_factory_creates_correct_capacity() {
        let spsc = QueueFactory::create_spsc(QueueTier::Medium);
        assert_eq!(spsc.capacity(), 1024);

        let mpsc = QueueFactory::create_mpsc(QueueTier::Large);
        assert_eq!(mpsc.capacity(), 4096);

        let bounded = QueueFactory::create_bounded(QueueTier::Small);
        assert_eq!(bounded.capacity(), 256);
    }

    #[test]
    fn test_queue_factory_throughput_based() {
        let queue = QueueFactory::create_for_throughput(10000, 100);
        // 10000 msg/s * 100ms = 1000 msgs, needs Medium tier = 1024
        assert_eq!(queue.capacity(), 1024);
    }

    #[test]
    fn test_queue_monitor_health_levels() {
        let monitor = QueueMonitor::default();
        let queue = SpscQueue::new(100); // Will round to 128

        // Empty - healthy
        assert_eq!(monitor.check(&queue), QueueHealth::Healthy);

        // Fill to 60% - still healthy
        for _ in 0..76 {
            queue.try_enqueue(make_envelope()).unwrap();
        }
        assert_eq!(monitor.check(&queue), QueueHealth::Healthy);

        // Fill to 80% - warning
        for _ in 0..26 {
            queue.try_enqueue(make_envelope()).unwrap();
        }
        assert_eq!(monitor.check(&queue), QueueHealth::Warning);

        // Fill to 95% - critical
        for _ in 0..18 {
            queue.try_enqueue(make_envelope()).unwrap();
        }
        assert_eq!(monitor.check(&queue), QueueHealth::Critical);
    }

    #[test]
    fn test_queue_monitor_utilization() {
        let monitor = QueueMonitor::default();
        let queue = SpscQueue::new(100); // Will round to 128

        assert!((monitor.utilization(&queue) - 0.0).abs() < 0.001);

        for _ in 0..64 {
            queue.try_enqueue(make_envelope()).unwrap();
        }
        assert!((monitor.utilization(&queue) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_queue_monitor_drop_detection() {
        let monitor = QueueMonitor::default();
        let queue = SpscQueue::new(4);

        // Fill queue completely
        for _ in 0..4 {
            queue.try_enqueue(make_envelope()).unwrap();
        }
        assert!(!monitor.has_drops(&queue));

        // Attempt another enqueue (should fail and record drop)
        let _ = queue.try_enqueue(make_envelope());
        assert!(monitor.has_drops(&queue));
        assert!(monitor.drop_rate(&queue) > 0.0);
    }

    #[test]
    fn test_queue_monitor_upgrade_suggestion() {
        let monitor = QueueMonitor::default();
        let queue = SpscQueue::new(QueueTier::Small.capacity());

        // Empty queue - no upgrade needed
        assert!(monitor.suggest_upgrade(&queue, QueueTier::Small).is_none());

        // Fill to warning level (>= 75%)
        for _ in 0..200 {
            queue.try_enqueue(make_envelope()).unwrap();
        }

        // Should suggest upgrade
        let suggestion = monitor.suggest_upgrade(&queue, QueueTier::Small);
        assert_eq!(suggestion, Some(QueueTier::Medium));

        // Already at max tier - no upgrade possible
        let large_queue = SpscQueue::new(QueueTier::ExtraLarge.capacity());
        for _ in 0..(QueueTier::ExtraLarge.capacity() * 3 / 4) {
            large_queue.try_enqueue(make_envelope()).unwrap();
        }
        assert!(monitor
            .suggest_upgrade(&large_queue, QueueTier::ExtraLarge)
            .is_none());
    }

    #[test]
    fn test_queue_metrics_capture() {
        let queue = SpscQueue::new(QueueTier::Medium.capacity());
        let monitor = QueueMonitor::default();

        // Add some messages
        for _ in 0..100 {
            queue.try_enqueue(make_envelope()).unwrap();
        }

        let metrics = QueueMetrics::capture(&queue, &monitor, Some(QueueTier::Medium));

        assert_eq!(metrics.health, QueueHealth::Healthy);
        assert!(metrics.utilization < 0.15);
        assert_eq!(metrics.stats.enqueued, 100);
        assert_eq!(metrics.tier, Some(QueueTier::Medium));
        assert!(metrics.suggested_upgrade.is_none());
    }

    // ========================================================================
    // Partitioned Queue Tests
    // ========================================================================

    #[test]
    fn test_partitioned_queue_creation() {
        let queue = PartitionedQueue::new(4, 256);
        assert_eq!(queue.partition_count(), 4);
        assert_eq!(queue.capacity_per_partition(), 256);
        assert_eq!(queue.total_capacity(), 1024);
    }

    #[test]
    fn test_partitioned_queue_rounds_to_power_of_two() {
        let queue = PartitionedQueue::new(3, 256);
        assert_eq!(queue.partition_count(), 4); // Rounded up to 4
    }

    #[test]
    fn test_partitioned_queue_routing() {
        let queue = PartitionedQueue::with_defaults();

        // Same source ID should always go to same partition
        let partition1 = queue.partition_for(12345);
        let partition2 = queue.partition_for(12345);
        assert_eq!(partition1, partition2);

        // Different source IDs may go to different partitions
        let partition_a = queue.partition_for(0);
        let partition_b = queue.partition_for(1);
        assert!(partition_a != partition_b || queue.partition_count() == 1);
    }

    #[test]
    fn test_partitioned_queue_enqueue_dequeue() {
        let queue = PartitionedQueue::new(4, 64);

        // Enqueue from different sources
        for source in 0..16u64 {
            let mut env = make_envelope();
            env.header.source_kernel = source;
            queue.try_enqueue(env).unwrap();
        }

        assert_eq!(queue.total_messages(), 16);

        // Dequeue all
        for _ in 0..16 {
            let env = queue.try_dequeue_any();
            assert!(env.is_some());
        }

        assert_eq!(queue.total_messages(), 0);
        assert!(queue.try_dequeue_any().is_none());
    }

    #[test]
    fn test_partitioned_queue_stats() {
        let queue = PartitionedQueue::new(4, 64);

        // Enqueue to different partitions
        for source in 0..20u64 {
            let mut env = make_envelope();
            env.header.source_kernel = source;
            queue.try_enqueue(env).unwrap();
        }

        let stats = queue.stats();
        assert_eq!(stats.total.enqueued, 20);
        assert_eq!(stats.partition_count, 4);
        assert_eq!(stats.partition_stats.len(), 4);
    }

    #[test]
    fn test_partitioned_queue_load_imbalance() {
        let queue = PartitionedQueue::new(4, 64);

        // All messages go to same partition (source 0 maps to partition 0)
        for _ in 0..10 {
            let mut env = make_envelope();
            env.header.source_kernel = 0;
            queue.try_enqueue(env).unwrap();
        }

        let stats = queue.stats();
        // All 10 messages in one partition, avg = 2.5, max = 10
        // Imbalance = 10 / 2.5 = 4.0
        assert!((stats.load_imbalance() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_partitioned_queue_dequeue_partition() {
        let queue = PartitionedQueue::new(4, 64);

        // Enqueue to a specific partition (source 0)
        let mut env = make_envelope();
        env.header.source_kernel = 0;
        queue.try_enqueue(env).unwrap();

        let partition = queue.partition_for(0);

        // Dequeue from that specific partition
        let result = queue.try_dequeue_partition(partition);
        assert!(result.is_ok());

        // Invalid partition should error
        let result = queue.try_dequeue_partition(100);
        assert!(result.is_err());
    }
}
