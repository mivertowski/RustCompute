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
}
