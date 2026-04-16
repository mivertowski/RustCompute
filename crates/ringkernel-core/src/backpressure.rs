//! Credit-Based Backpressure & Flow Control — FR-003
//!
//! Implements credit-based flow control for K2K messaging:
//! - Producer requests credits from consumer before sending
//! - Consumer grants credits based on available queue capacity
//! - When credits exhausted: producer blocks, buffers, or drops (configurable)
//! - Queue watermark signals: LOW_WATER / HIGH_WATER
//! - Per-channel flow control
//!
//! # Protocol
//!
//! ```text
//! Producer                          Consumer
//!    │                                  │
//!    │── RequestCredits(n) ──────────►  │
//!    │                                  │── GrantCredits(n) ─►
//!    │◄── GrantCredits(n) ──────────── │
//!    │                                  │
//!    │── Message (credit-1) ─────────► │
//!    │── Message (credit-1) ─────────► │
//!    │── Message (credit-1) ─────────► │
//!    │   (credits exhausted)            │
//!    │── RequestCredits(n) ──────────►  │
//!    │   ...                            │
//! ```

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Instant;

/// Flow control configuration for a K2K channel.
#[derive(Debug, Clone)]
pub struct FlowControlConfig {
    /// Initial credits granted to producer.
    pub initial_credits: u32,
    /// Credits to grant on each refill.
    pub refill_amount: u32,
    /// Queue depth at which HIGH_WATER signal fires.
    pub high_water_mark: u32,
    /// Queue depth at which LOW_WATER signal fires (after HIGH_WATER).
    pub low_water_mark: u32,
    /// What to do when credits are exhausted.
    pub overflow_policy: OverflowPolicy,
}

impl Default for FlowControlConfig {
    fn default() -> Self {
        Self {
            initial_credits: 64,
            refill_amount: 32,
            high_water_mark: 192, // 75% of 256
            low_water_mark: 64,   // 25% of 256
            overflow_policy: OverflowPolicy::Block,
        }
    }
}

/// Policy when producer has no credits left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowPolicy {
    /// Block the producer until credits are available.
    Block,
    /// Buffer messages locally (up to buffer_limit).
    Buffer { limit: u32 },
    /// Drop the message and route to dead letter queue.
    DropToDlq,
    /// Drop the message silently (with metric increment).
    DropSilent,
}

/// Watermark signal emitted when queue depth crosses thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatermarkSignal {
    /// Queue depth rose above high_water_mark — producer should slow down.
    HighWater,
    /// Queue depth fell below low_water_mark — producer can resume.
    LowWater,
}

/// Per-channel flow controller.
///
/// Tracks credits, watermark state, and flow control metrics.
pub struct FlowController {
    /// Available credits for the producer.
    credits: AtomicI64,
    /// Configuration.
    config: FlowControlConfig,
    /// Current watermark state.
    is_high_water: std::sync::atomic::AtomicBool,
    /// Metrics.
    metrics: FlowMetrics,
}

/// Flow control metrics.
pub struct FlowMetrics {
    /// Total credits granted.
    pub credits_granted: AtomicU64,
    /// Total credits consumed (messages sent).
    pub credits_consumed: AtomicU64,
    /// Times producer was blocked (credits exhausted).
    pub backpressure_events: AtomicU64,
    /// Messages routed to DLQ due to overflow.
    pub dlq_routed: AtomicU64,
    /// Messages dropped silently.
    pub dropped: AtomicU64,
    /// High water signals emitted.
    pub high_water_signals: AtomicU64,
    /// Low water signals emitted.
    pub low_water_signals: AtomicU64,
}

impl FlowController {
    /// Create a new flow controller with the given config.
    pub fn new(config: FlowControlConfig) -> Self {
        let initial = config.initial_credits as i64;
        Self {
            credits: AtomicI64::new(initial),
            config,
            is_high_water: std::sync::atomic::AtomicBool::new(false),
            metrics: FlowMetrics {
                credits_granted: AtomicU64::new(0),
                credits_consumed: AtomicU64::new(0),
                backpressure_events: AtomicU64::new(0),
                dlq_routed: AtomicU64::new(0),
                dropped: AtomicU64::new(0),
                high_water_signals: AtomicU64::new(0),
                low_water_signals: AtomicU64::new(0),
            },
        }
    }

    /// Try to acquire a credit for sending. Returns true if credit available.
    pub fn try_acquire(&self) -> bool {
        let prev = self.credits.fetch_sub(1, Ordering::AcqRel);
        if prev > 0 {
            self.metrics.credits_consumed.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            // Restore the credit (we didn't actually send)
            self.credits.fetch_add(1, Ordering::Release);
            self.metrics.backpressure_events.fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    /// Grant credits to the producer (called by consumer when queue has space).
    pub fn grant(&self, amount: u32) {
        self.credits.fetch_add(amount as i64, Ordering::Release);
        self.metrics.credits_granted.fetch_add(amount as u64, Ordering::Relaxed);
    }

    /// Refill credits by the configured refill amount.
    pub fn refill(&self) {
        self.grant(self.config.refill_amount);
    }

    /// Available credits.
    pub fn available_credits(&self) -> i64 {
        self.credits.load(Ordering::Acquire)
    }

    /// Check watermark state based on current queue depth.
    ///
    /// Returns a watermark signal if a threshold was just crossed.
    pub fn check_watermark(&self, queue_depth: u32) -> Option<WatermarkSignal> {
        let was_high = self.is_high_water.load(Ordering::Acquire);

        if !was_high && queue_depth >= self.config.high_water_mark {
            self.is_high_water.store(true, Ordering::Release);
            self.metrics.high_water_signals.fetch_add(1, Ordering::Relaxed);
            Some(WatermarkSignal::HighWater)
        } else if was_high && queue_depth <= self.config.low_water_mark {
            self.is_high_water.store(false, Ordering::Release);
            self.metrics.low_water_signals.fetch_add(1, Ordering::Relaxed);
            Some(WatermarkSignal::LowWater)
        } else {
            None
        }
    }

    /// Get the overflow policy.
    pub fn overflow_policy(&self) -> OverflowPolicy {
        self.config.overflow_policy
    }

    /// Record a message dropped (for metrics).
    pub fn record_drop(&self) {
        self.metrics.dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a message routed to DLQ (for metrics).
    pub fn record_dlq(&self) {
        self.metrics.dlq_routed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot of flow control metrics.
    pub fn metrics_snapshot(&self) -> FlowMetricsSnapshot {
        FlowMetricsSnapshot {
            credits_available: self.credits.load(Ordering::Relaxed),
            credits_granted: self.metrics.credits_granted.load(Ordering::Relaxed),
            credits_consumed: self.metrics.credits_consumed.load(Ordering::Relaxed),
            backpressure_events: self.metrics.backpressure_events.load(Ordering::Relaxed),
            dlq_routed: self.metrics.dlq_routed.load(Ordering::Relaxed),
            dropped: self.metrics.dropped.load(Ordering::Relaxed),
            high_water_signals: self.metrics.high_water_signals.load(Ordering::Relaxed),
            low_water_signals: self.metrics.low_water_signals.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of flow control metrics (non-atomic, for reporting).
#[derive(Debug, Clone)]
pub struct FlowMetricsSnapshot {
    pub credits_available: i64,
    pub credits_granted: u64,
    pub credits_consumed: u64,
    pub backpressure_events: u64,
    pub dlq_routed: u64,
    pub dropped: u64,
    pub high_water_signals: u64,
    pub low_water_signals: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_controller_basic() {
        let fc = FlowController::new(FlowControlConfig {
            initial_credits: 3,
            ..Default::default()
        });

        assert!(fc.try_acquire());
        assert!(fc.try_acquire());
        assert!(fc.try_acquire());
        assert!(!fc.try_acquire()); // No credits left

        fc.grant(2);
        assert!(fc.try_acquire());
        assert!(fc.try_acquire());
        assert!(!fc.try_acquire());
    }

    #[test]
    fn test_watermark_signals() {
        let fc = FlowController::new(FlowControlConfig {
            high_water_mark: 80,
            low_water_mark: 20,
            ..Default::default()
        });

        assert_eq!(fc.check_watermark(50), None);
        assert_eq!(fc.check_watermark(80), Some(WatermarkSignal::HighWater));
        assert_eq!(fc.check_watermark(90), None); // Already high
        assert_eq!(fc.check_watermark(20), Some(WatermarkSignal::LowWater));
        assert_eq!(fc.check_watermark(10), None); // Already low
    }

    #[test]
    fn test_metrics_tracking() {
        let fc = FlowController::new(FlowControlConfig {
            initial_credits: 2,
            ..Default::default()
        });

        fc.try_acquire();
        fc.try_acquire();
        fc.try_acquire(); // Should fail

        let m = fc.metrics_snapshot();
        assert_eq!(m.credits_consumed, 2);
        assert_eq!(m.backpressure_events, 1);
    }

    #[test]
    fn test_refill() {
        let fc = FlowController::new(FlowControlConfig {
            initial_credits: 0,
            refill_amount: 10,
            ..Default::default()
        });

        assert!(!fc.try_acquire());
        fc.refill();
        assert_eq!(fc.available_credits(), 10);
        for _ in 0..10 {
            assert!(fc.try_acquire());
        }
        assert!(!fc.try_acquire());
    }
}
