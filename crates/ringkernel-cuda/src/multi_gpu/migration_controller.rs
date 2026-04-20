//! Migration rate limiting and buffer-budget accounting.
//!
//! The migration orchestrator asks the controller to approve every
//! migration before it starts. The controller enforces three independent
//! invariants:
//!
//! 1. **Concurrency cap** — at most `max_concurrent` migrations run in
//!    parallel. Spec §3.1 default is 1000.
//! 2. **Buffer budget** — total staging memory across all live
//!    migrations is bounded by `global_buffer_budget`. Spec §3.1 default
//!    is 8 GB.
//! 3. **Per-second rate limit** — a simple token bucket caps
//!    migrations-per-second to `rate_limit_per_sec`.
//!
//! Granted requests return a [`MigrationPermit`] that the caller must
//! release on completion. The permit tracks the buffer-budget charge so
//! the controller can refund it precisely (rather than over-
//! approximating).
//!
//! The implementation is deliberately lock-light: concurrency and
//! buffer counters are atomic and the rate limiter only grabs a mutex
//! on refill.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use ringkernel_core::error::Result;

use super::migration::MultiGpuError;

/// Configuration for [`MigrationController`].
#[derive(Debug, Clone)]
pub struct MigrationControllerConfig {
    /// Maximum migrations that may run in parallel.
    pub max_concurrent: u32,
    /// Global in-flight buffer budget, in bytes.
    pub global_buffer_budget: u64,
    /// Optional path used for disk-backed staging during bursts. When
    /// set, staging buffers allocated through the runtime will spill
    /// here rather than returning
    /// [`MultiGpuError::StagingOverflow`].
    pub disk_staging: Option<PathBuf>,
    /// Maximum migrations per second (token bucket).
    pub rate_limit_per_sec: u32,
}

impl Default for MigrationControllerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 1000,
            global_buffer_budget: 8 * 1024 * 1024 * 1024, // 8 GB
            disk_staging: None,
            rate_limit_per_sec: 2000,
        }
    }
}

/// Controller that rate-limits and budgets migrations.
pub struct MigrationController {
    config: MigrationControllerConfig,
    in_flight: AtomicU32,
    buffer_used: AtomicU64,
    rate_limiter: TokenBucket,
}

impl MigrationController {
    /// Create a new controller.
    pub fn new(config: MigrationControllerConfig) -> Self {
        let rate_limit = config.rate_limit_per_sec.max(1);
        Self {
            rate_limiter: TokenBucket::new(rate_limit as u64, rate_limit as u64),
            config,
            in_flight: AtomicU32::new(0),
            buffer_used: AtomicU64::new(0),
        }
    }

    /// Config accessor (used by the staging-buffer allocator on the
    /// runtime side).
    pub fn config(&self) -> &MigrationControllerConfig {
        &self.config
    }

    /// Ask permission to start a migration that will consume
    /// `estimated_bytes` of staging memory. On success the caller
    /// receives a [`MigrationPermit`]; if any invariant is violated an
    /// appropriate [`MultiGpuError`] is returned and all counters are
    /// left untouched (atomic compare-and-swap semantics).
    pub fn try_acquire(&self, estimated_bytes: u64) -> Result<MigrationPermit> {
        // 1) Concurrency cap.
        loop {
            let current = self.in_flight.load(Ordering::Acquire);
            if current >= self.config.max_concurrent {
                return Err(MultiGpuError::ConcurrencyLimitReached {
                    current,
                    limit: self.config.max_concurrent,
                }
                .into());
            }
            if self
                .in_flight
                .compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }

        // 2) Buffer budget.
        let prev = self
            .buffer_used
            .fetch_add(estimated_bytes, Ordering::AcqRel);
        if prev + estimated_bytes > self.config.global_buffer_budget {
            // Roll back both counters.
            self.buffer_used
                .fetch_sub(estimated_bytes, Ordering::AcqRel);
            self.in_flight.fetch_sub(1, Ordering::AcqRel);
            return Err(MultiGpuError::BudgetExhausted {
                in_flight_bytes: prev + estimated_bytes,
                budget_bytes: self.config.global_buffer_budget,
            }
            .into());
        }

        // 3) Rate limit.
        if !self.rate_limiter.try_acquire() {
            self.buffer_used
                .fetch_sub(estimated_bytes, Ordering::AcqRel);
            self.in_flight.fetch_sub(1, Ordering::AcqRel);
            return Err(MultiGpuError::RateLimited {
                limit_per_sec: self.config.rate_limit_per_sec,
            }
            .into());
        }

        Ok(MigrationPermit { estimated_bytes })
    }

    /// Release a previously-acquired permit. Safe to call at most once
    /// per permit (we consume `permit` by value).
    pub fn release(&self, permit: MigrationPermit) {
        self.buffer_used
            .fetch_sub(permit.estimated_bytes, Ordering::AcqRel);
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
    }

    /// Current concurrent migrations.
    pub fn current_concurrent(&self) -> u32 {
        self.in_flight.load(Ordering::Acquire)
    }

    /// Current buffer bytes in use.
    pub fn current_buffer_used(&self) -> u64 {
        self.buffer_used.load(Ordering::Acquire)
    }

    /// Whether disk staging is configured.
    pub fn disk_staging(&self) -> Option<&PathBuf> {
        self.config.disk_staging.as_ref()
    }
}

/// Permit returned by [`MigrationController::try_acquire`]. Must be
/// returned via [`MigrationController::release`].
#[derive(Debug)]
#[must_use = "A permit must be returned via MigrationController::release"]
pub struct MigrationPermit {
    estimated_bytes: u64,
}

impl MigrationPermit {
    /// Bytes this permit reserved against the buffer budget.
    pub fn estimated_bytes(&self) -> u64 {
        self.estimated_bytes
    }
}

/// Private token bucket.
struct TokenBucket {
    tokens: AtomicU64,
    capacity: u64,
    refill_per_sec: u64,
    last_refill: Mutex<Instant>,
}

impl TokenBucket {
    fn new(capacity: u64, refill_per_sec: u64) -> Self {
        Self {
            tokens: AtomicU64::new(capacity),
            capacity,
            refill_per_sec,
            last_refill: Mutex::new(Instant::now()),
        }
    }

    fn refill(&self) {
        let mut last = self.last_refill.lock();
        let now = Instant::now();
        let elapsed = now.duration_since(*last);
        // Avoid excessive lock traffic on every call.
        if elapsed < Duration::from_micros(100) {
            return;
        }
        let to_add = (elapsed.as_secs_f64() * self.refill_per_sec as f64) as u64;
        if to_add > 0 {
            let cur = self.tokens.load(Ordering::Acquire);
            let new = (cur + to_add).min(self.capacity);
            self.tokens.store(new, Ordering::Release);
            *last = now;
        }
    }

    fn try_acquire(&self) -> bool {
        self.refill();
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current == 0 {
                return false;
            }
            if self
                .tokens
                .compare_exchange(current, current - 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ringkernel_core::error::RingKernelError;

    fn small_config() -> MigrationControllerConfig {
        MigrationControllerConfig {
            max_concurrent: 2,
            global_buffer_budget: 1024,
            disk_staging: None,
            rate_limit_per_sec: 100,
        }
    }

    #[test]
    fn try_acquire_succeeds_within_limits() {
        let ctrl = MigrationController::new(small_config());
        let p = ctrl.try_acquire(100).expect("within limits");
        assert_eq!(ctrl.current_concurrent(), 1);
        assert_eq!(ctrl.current_buffer_used(), 100);
        ctrl.release(p);
        assert_eq!(ctrl.current_concurrent(), 0);
        assert_eq!(ctrl.current_buffer_used(), 0);
    }

    #[test]
    fn concurrency_limit_enforced() {
        let ctrl = MigrationController::new(small_config());
        let p1 = ctrl.try_acquire(100).unwrap();
        let p2 = ctrl.try_acquire(100).unwrap();
        let err = ctrl.try_acquire(100).expect_err("should be rejected");
        match err {
            RingKernelError::MigrationFailed(msg) => {
                assert!(msg.contains("concurrency"), "{msg}");
            }
            other => panic!("wrong error: {other:?}"),
        }
        assert_eq!(ctrl.current_concurrent(), 2);
        ctrl.release(p1);
        ctrl.release(p2);
    }

    #[test]
    fn buffer_budget_enforced() {
        let ctrl = MigrationController::new(small_config());
        let _p = ctrl.try_acquire(900).unwrap();
        // 900 + 200 > 1024 -> rejected
        let err = ctrl.try_acquire(200).expect_err("budget exhausted");
        match err {
            RingKernelError::MigrationFailed(msg) => {
                assert!(msg.contains("buffer budget"), "{msg}");
            }
            other => panic!("wrong error: {other:?}"),
        }
        // Counters must be unchanged after the failed acquisition.
        assert_eq!(ctrl.current_concurrent(), 1);
        assert_eq!(ctrl.current_buffer_used(), 900);
    }

    #[test]
    fn rate_limit_enforced() {
        let cfg = MigrationControllerConfig {
            max_concurrent: 100,
            global_buffer_budget: u64::MAX,
            disk_staging: None,
            rate_limit_per_sec: 2, // bucket capacity 2
        };
        let ctrl = MigrationController::new(cfg);
        let _p1 = ctrl.try_acquire(1).unwrap();
        let _p2 = ctrl.try_acquire(1).unwrap();
        // Third immediate call should be rate-limited.
        let err = ctrl.try_acquire(1).expect_err("rate limited");
        match err {
            RingKernelError::MigrationFailed(msg) => {
                assert!(msg.contains("rate limit"), "{msg}");
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn release_refunds_budget() {
        let ctrl = MigrationController::new(small_config());
        let p = ctrl.try_acquire(500).unwrap();
        ctrl.release(p);
        assert_eq!(ctrl.current_buffer_used(), 0);
        // Full budget is now reusable again.
        let p = ctrl.try_acquire(500).unwrap();
        ctrl.release(p);
    }

    #[test]
    fn default_config_has_spec_defaults() {
        let cfg = MigrationControllerConfig::default();
        assert_eq!(cfg.max_concurrent, 1000);
        assert_eq!(cfg.global_buffer_budget, 8 * 1024 * 1024 * 1024);
        assert!(cfg.rate_limit_per_sec > 0);
    }

    #[test]
    fn disk_staging_accessor() {
        let mut cfg = small_config();
        cfg.disk_staging = Some(PathBuf::from("/tmp/spill"));
        let ctrl = MigrationController::new(cfg);
        assert_eq!(
            ctrl.disk_staging().map(|p| p.to_string_lossy().to_string()),
            Some("/tmp/spill".to_string())
        );
    }
}
