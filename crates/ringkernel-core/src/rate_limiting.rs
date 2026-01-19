//! Rate limiting for enterprise workloads.
//!
//! This module provides rate limiting capabilities for controlling request rates
//! to GPU kernels and system resources, supporting both global and per-tenant limits.
//!
//! # Features
//!
//! - Token bucket and sliding window algorithms
//! - Per-tenant rate limiting
//! - Global rate limiting
//! - Configurable burst capacity
//! - Real-time statistics
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::rate_limiting::{RateLimiter, RateLimitConfig};
//!
//! let config = RateLimitConfig::default()
//!     .with_requests_per_second(100)
//!     .with_burst_size(50);
//!
//! let limiter = RateLimiter::new(config);
//!
//! if limiter.check("tenant_1").is_ok() {
//!     // Request allowed
//! }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

// ============================================================================
// RATE LIMIT ERRORS
// ============================================================================

/// Errors that can occur during rate limiting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitError {
    /// Request rate exceeded the configured limit.
    RateLimitExceeded {
        /// Time until the rate limit resets.
        retry_after: Duration,
        /// Current request count in the window.
        current_count: u64,
        /// Maximum allowed requests.
        limit: u64,
    },
    /// The specified tenant was not found.
    TenantNotFound(String),
    /// The rate limiter is disabled.
    Disabled,
}

impl std::fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RateLimitExceeded {
                retry_after,
                current_count,
                limit,
            } => {
                write!(
                    f,
                    "Rate limit exceeded: {}/{} requests, retry after {:?}",
                    current_count, limit, retry_after
                )
            }
            Self::TenantNotFound(id) => write!(f, "Tenant not found: {}", id),
            Self::Disabled => write!(f, "Rate limiter is disabled"),
        }
    }
}

impl std::error::Error for RateLimitError {}

/// Result type for rate limiting operations.
pub type RateLimitResult<T> = Result<T, RateLimitError>;

// ============================================================================
// RATE LIMIT ALGORITHMS
// ============================================================================

/// Rate limiting algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RateLimitAlgorithm {
    /// Token bucket algorithm.
    ///
    /// Tokens are added at a fixed rate. Each request consumes one token.
    /// Allows bursting up to the bucket capacity.
    #[default]
    TokenBucket,
    /// Sliding window algorithm.
    ///
    /// Counts requests in a sliding time window. Provides smoother
    /// rate limiting than fixed windows.
    SlidingWindow,
    /// Fixed window algorithm.
    ///
    /// Counts requests in fixed time windows. Simple but can allow
    /// bursting at window boundaries.
    FixedWindow,
    /// Leaky bucket algorithm.
    ///
    /// Requests are processed at a constant rate. Excess requests
    /// are queued or rejected.
    LeakyBucket,
}

// ============================================================================
// RATE LIMIT CONFIGURATION
// ============================================================================

/// Configuration for rate limiting.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per second.
    pub requests_per_second: u64,
    /// Burst capacity (for token bucket).
    pub burst_size: u64,
    /// Window size for sliding/fixed window algorithms.
    pub window_size: Duration,
    /// Algorithm to use.
    pub algorithm: RateLimitAlgorithm,
    /// Whether rate limiting is enabled.
    pub enabled: bool,
    /// Whether to track per-tenant limits.
    pub per_tenant: bool,
    /// Default quota for new tenants.
    pub default_tenant_quota: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 1000,
            burst_size: 100,
            window_size: Duration::from_secs(1),
            algorithm: RateLimitAlgorithm::TokenBucket,
            enabled: true,
            per_tenant: true,
            default_tenant_quota: 100,
        }
    }
}

impl RateLimitConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the requests per second limit.
    pub fn with_requests_per_second(mut self, rps: u64) -> Self {
        self.requests_per_second = rps;
        self
    }

    /// Set the burst size.
    pub fn with_burst_size(mut self, size: u64) -> Self {
        self.burst_size = size;
        self
    }

    /// Set the window size.
    pub fn with_window_size(mut self, size: Duration) -> Self {
        self.window_size = size;
        self
    }

    /// Set the algorithm.
    pub fn with_algorithm(mut self, algorithm: RateLimitAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Enable or disable rate limiting.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Enable or disable per-tenant limiting.
    pub fn with_per_tenant(mut self, per_tenant: bool) -> Self {
        self.per_tenant = per_tenant;
        self
    }

    /// Set the default tenant quota.
    pub fn with_default_tenant_quota(mut self, quota: u64) -> Self {
        self.default_tenant_quota = quota;
        self
    }

    /// Create a strict rate limit configuration.
    pub fn strict(rps: u64) -> Self {
        Self {
            requests_per_second: rps,
            burst_size: rps / 10, // 10% burst
            window_size: Duration::from_secs(1),
            algorithm: RateLimitAlgorithm::SlidingWindow,
            enabled: true,
            per_tenant: true,
            default_tenant_quota: rps / 10,
        }
    }

    /// Create a permissive rate limit configuration.
    pub fn permissive(rps: u64) -> Self {
        Self {
            requests_per_second: rps,
            burst_size: rps * 2, // 200% burst
            window_size: Duration::from_secs(1),
            algorithm: RateLimitAlgorithm::TokenBucket,
            enabled: true,
            per_tenant: false,
            default_tenant_quota: rps,
        }
    }
}

// ============================================================================
// TOKEN BUCKET
// ============================================================================

/// Token bucket rate limiter state.
#[derive(Debug)]
struct TokenBucket {
    /// Current number of tokens.
    tokens: AtomicU64,
    /// Maximum tokens (burst capacity).
    capacity: u64,
    /// Tokens added per second.
    refill_rate: u64,
    /// Last refill time.
    last_refill: RwLock<Instant>,
}

impl TokenBucket {
    fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            tokens: AtomicU64::new(capacity),
            capacity,
            refill_rate,
            last_refill: RwLock::new(Instant::now()),
        }
    }

    fn try_acquire(&self) -> RateLimitResult<()> {
        self.refill();

        let current = self.tokens.load(Ordering::Acquire);
        if current == 0 {
            // Calculate retry after based on refill rate
            let retry_after = if self.refill_rate > 0 {
                Duration::from_secs_f64(1.0 / self.refill_rate as f64)
            } else {
                Duration::from_secs(1)
            };

            return Err(RateLimitError::RateLimitExceeded {
                retry_after,
                current_count: self.capacity - current,
                limit: self.capacity,
            });
        }

        // Try to decrement
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current == 0 {
                let retry_after = if self.refill_rate > 0 {
                    Duration::from_secs_f64(1.0 / self.refill_rate as f64)
                } else {
                    Duration::from_secs(1)
                };
                return Err(RateLimitError::RateLimitExceeded {
                    retry_after,
                    current_count: self.capacity,
                    limit: self.capacity,
                });
            }

            if self
                .tokens
                .compare_exchange(current, current - 1, Ordering::Release, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    fn refill(&self) {
        let now = Instant::now();
        let mut last = self.last_refill.write();
        let elapsed = now.duration_since(*last);

        // Calculate tokens to add
        let tokens_to_add = (elapsed.as_secs_f64() * self.refill_rate as f64) as u64;

        if tokens_to_add > 0 {
            let current = self.tokens.load(Ordering::Acquire);
            let new_tokens = (current + tokens_to_add).min(self.capacity);
            self.tokens.store(new_tokens, Ordering::Release);
            *last = now;
        }
    }

    #[allow(dead_code)]
    fn available_tokens(&self) -> u64 {
        self.refill();
        self.tokens.load(Ordering::Acquire)
    }
}

// ============================================================================
// SLIDING WINDOW
// ============================================================================

/// Sliding window rate limiter state.
#[derive(Debug)]
struct SlidingWindow {
    /// Request timestamps in the current window.
    requests: RwLock<Vec<Instant>>,
    /// Window size.
    window_size: Duration,
    /// Maximum requests per window.
    limit: u64,
}

impl SlidingWindow {
    fn new(window_size: Duration, limit: u64) -> Self {
        Self {
            requests: RwLock::new(Vec::with_capacity(limit as usize)),
            window_size,
            limit,
        }
    }

    fn try_acquire(&self) -> RateLimitResult<()> {
        let now = Instant::now();
        let window_start = now - self.window_size;

        let mut requests = self.requests.write();

        // Remove old requests
        requests.retain(|&t| t > window_start);

        if requests.len() as u64 >= self.limit {
            // Find oldest request in window to calculate retry time
            let oldest = requests.iter().min().copied().unwrap_or(now);
            let retry_after = oldest + self.window_size - now;

            return Err(RateLimitError::RateLimitExceeded {
                retry_after,
                current_count: requests.len() as u64,
                limit: self.limit,
            });
        }

        requests.push(now);
        Ok(())
    }

    #[allow(dead_code)]
    fn current_count(&self) -> u64 {
        let now = Instant::now();
        let window_start = now - self.window_size;

        let requests = self.requests.read();
        requests.iter().filter(|&&t| t > window_start).count() as u64
    }
}

// ============================================================================
// FIXED WINDOW
// ============================================================================

/// Fixed window rate limiter state.
#[derive(Debug)]
struct FixedWindow {
    /// Request count in current window.
    count: AtomicU64,
    /// Window start time.
    window_start: RwLock<Instant>,
    /// Window size.
    window_size: Duration,
    /// Maximum requests per window.
    limit: u64,
}

impl FixedWindow {
    fn new(window_size: Duration, limit: u64) -> Self {
        Self {
            count: AtomicU64::new(0),
            window_start: RwLock::new(Instant::now()),
            window_size,
            limit,
        }
    }

    fn try_acquire(&self) -> RateLimitResult<()> {
        let now = Instant::now();

        // Check if we need to reset the window
        {
            let start = *self.window_start.read();
            if now.duration_since(start) >= self.window_size {
                let mut start_write = self.window_start.write();
                // Double-check after acquiring write lock
                if now.duration_since(*start_write) >= self.window_size {
                    *start_write = now;
                    self.count.store(0, Ordering::Release);
                }
            }
        }

        // Try to increment count
        loop {
            let current = self.count.load(Ordering::Acquire);
            if current >= self.limit {
                let start = *self.window_start.read();
                let retry_after = (start + self.window_size).saturating_duration_since(now);

                return Err(RateLimitError::RateLimitExceeded {
                    retry_after,
                    current_count: current,
                    limit: self.limit,
                });
            }

            if self
                .count
                .compare_exchange(current, current + 1, Ordering::Release, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    #[allow(dead_code)]
    fn current_count(&self) -> u64 {
        self.count.load(Ordering::Acquire)
    }
}

// ============================================================================
// LEAKY BUCKET
// ============================================================================

/// Leaky bucket rate limiter state.
#[derive(Debug)]
struct LeakyBucket {
    /// Current water level (pending requests).
    level: AtomicU64,
    /// Maximum capacity.
    capacity: u64,
    /// Leak rate (requests per second).
    leak_rate: u64,
    /// Last leak time.
    last_leak: RwLock<Instant>,
}

impl LeakyBucket {
    fn new(capacity: u64, leak_rate: u64) -> Self {
        Self {
            level: AtomicU64::new(0),
            capacity,
            leak_rate,
            last_leak: RwLock::new(Instant::now()),
        }
    }

    fn try_acquire(&self) -> RateLimitResult<()> {
        self.leak();

        loop {
            let current = self.level.load(Ordering::Acquire);
            if current >= self.capacity {
                let retry_after = if self.leak_rate > 0 {
                    Duration::from_secs_f64(1.0 / self.leak_rate as f64)
                } else {
                    Duration::from_secs(1)
                };

                return Err(RateLimitError::RateLimitExceeded {
                    retry_after,
                    current_count: current,
                    limit: self.capacity,
                });
            }

            if self
                .level
                .compare_exchange(current, current + 1, Ordering::Release, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    fn leak(&self) {
        let now = Instant::now();
        let mut last = self.last_leak.write();
        let elapsed = now.duration_since(*last);

        let leaked = (elapsed.as_secs_f64() * self.leak_rate as f64) as u64;

        if leaked > 0 {
            let current = self.level.load(Ordering::Acquire);
            let new_level = current.saturating_sub(leaked);
            self.level.store(new_level, Ordering::Release);
            *last = now;
        }
    }

    #[allow(dead_code)]
    fn current_level(&self) -> u64 {
        self.leak();
        self.level.load(Ordering::Acquire)
    }
}

// ============================================================================
// RATE LIMITER
// ============================================================================

/// Internal limiter state.
enum LimiterState {
    TokenBucket(TokenBucket),
    SlidingWindow(SlidingWindow),
    FixedWindow(FixedWindow),
    LeakyBucket(LeakyBucket),
}

impl LimiterState {
    fn try_acquire(&self) -> RateLimitResult<()> {
        match self {
            Self::TokenBucket(b) => b.try_acquire(),
            Self::SlidingWindow(w) => w.try_acquire(),
            Self::FixedWindow(w) => w.try_acquire(),
            Self::LeakyBucket(b) => b.try_acquire(),
        }
    }
}

/// Per-tenant rate limiter entry.
struct TenantLimiter {
    state: LimiterState,
    quota: u64,
}

/// Global rate limiter.
///
/// Provides rate limiting for GPU kernel requests with support for
/// multiple algorithms and per-tenant limits.
pub struct RateLimiter {
    config: RateLimitConfig,
    /// Global limiter state.
    global: LimiterState,
    /// Per-tenant limiters.
    tenants: RwLock<HashMap<String, TenantLimiter>>,
    /// Statistics.
    stats: RateLimiterStats,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration.
    pub fn new(config: RateLimitConfig) -> Self {
        let global = Self::create_global_limiter(&config);

        Self {
            config,
            global,
            tenants: RwLock::new(HashMap::new()),
            stats: RateLimiterStats::default(),
        }
    }

    fn create_global_limiter(config: &RateLimitConfig) -> LimiterState {
        let limit = config.requests_per_second;
        match config.algorithm {
            RateLimitAlgorithm::TokenBucket => {
                LimiterState::TokenBucket(TokenBucket::new(config.burst_size, limit))
            }
            RateLimitAlgorithm::SlidingWindow => {
                LimiterState::SlidingWindow(SlidingWindow::new(config.window_size, limit))
            }
            RateLimitAlgorithm::FixedWindow => {
                LimiterState::FixedWindow(FixedWindow::new(config.window_size, limit))
            }
            RateLimitAlgorithm::LeakyBucket => {
                LimiterState::LeakyBucket(LeakyBucket::new(config.burst_size, limit))
            }
        }
    }

    fn create_tenant_limiter(config: &RateLimitConfig, quota: u64) -> LimiterState {
        // For tenant limiters, the quota is both the capacity and the refill rate
        match config.algorithm {
            RateLimitAlgorithm::TokenBucket => {
                // Tenant bucket: capacity = quota (allows up to quota requests)
                // refill_rate = quota (refills at quota per second)
                LimiterState::TokenBucket(TokenBucket::new(quota, quota))
            }
            RateLimitAlgorithm::SlidingWindow => {
                LimiterState::SlidingWindow(SlidingWindow::new(config.window_size, quota))
            }
            RateLimitAlgorithm::FixedWindow => {
                LimiterState::FixedWindow(FixedWindow::new(config.window_size, quota))
            }
            RateLimitAlgorithm::LeakyBucket => {
                LimiterState::LeakyBucket(LeakyBucket::new(quota, quota))
            }
        }
    }

    /// Check if a request should be allowed (global limit).
    pub fn check(&self) -> RateLimitResult<()> {
        if !self.config.enabled {
            return Err(RateLimitError::Disabled);
        }

        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);

        match self.global.try_acquire() {
            Ok(()) => {
                self.stats.allowed_requests.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                self.stats.rejected_requests.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Check if a request should be allowed for a specific tenant.
    pub fn check_tenant(&self, tenant_id: &str) -> RateLimitResult<()> {
        if !self.config.enabled {
            return Err(RateLimitError::Disabled);
        }

        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);

        // First check global limit
        if let Err(e) = self.global.try_acquire() {
            self.stats.rejected_requests.fetch_add(1, Ordering::Relaxed);
            return Err(e);
        }

        // Then check tenant limit if per-tenant is enabled
        if self.config.per_tenant {
            let tenants = self.tenants.read();
            if let Some(limiter) = tenants.get(tenant_id) {
                match limiter.state.try_acquire() {
                    Ok(()) => {
                        self.stats.allowed_requests.fetch_add(1, Ordering::Relaxed);
                        Ok(())
                    }
                    Err(e) => {
                        self.stats.rejected_requests.fetch_add(1, Ordering::Relaxed);
                        Err(e)
                    }
                }
            } else {
                // Tenant not registered, use default quota
                drop(tenants);
                self.register_tenant(tenant_id, self.config.default_tenant_quota);
                self.check_tenant(tenant_id)
            }
        } else {
            self.stats.allowed_requests.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Register a tenant with a custom quota.
    pub fn register_tenant(&self, tenant_id: &str, quota: u64) {
        let limiter = TenantLimiter {
            state: Self::create_tenant_limiter(&self.config, quota),
            quota,
        };

        let mut tenants = self.tenants.write();
        tenants.insert(tenant_id.to_string(), limiter);
    }

    /// Update a tenant's quota.
    pub fn update_tenant_quota(&self, tenant_id: &str, quota: u64) -> RateLimitResult<()> {
        let mut tenants = self.tenants.write();
        if let Some(limiter) = tenants.get_mut(tenant_id) {
            limiter.quota = quota;
            limiter.state = Self::create_tenant_limiter(&self.config, quota);
            Ok(())
        } else {
            Err(RateLimitError::TenantNotFound(tenant_id.to_string()))
        }
    }

    /// Remove a tenant.
    pub fn remove_tenant(&self, tenant_id: &str) -> bool {
        let mut tenants = self.tenants.write();
        tenants.remove(tenant_id).is_some()
    }

    /// Get the number of registered tenants.
    pub fn tenant_count(&self) -> usize {
        self.tenants.read().len()
    }

    /// Get the current statistics.
    pub fn stats(&self) -> RateLimiterStatsSnapshot {
        RateLimiterStatsSnapshot {
            total_requests: self.stats.total_requests.load(Ordering::Relaxed),
            allowed_requests: self.stats.allowed_requests.load(Ordering::Relaxed),
            rejected_requests: self.stats.rejected_requests.load(Ordering::Relaxed),
            tenant_count: self.tenant_count(),
        }
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        self.stats.total_requests.store(0, Ordering::Relaxed);
        self.stats.allowed_requests.store(0, Ordering::Relaxed);
        self.stats.rejected_requests.store(0, Ordering::Relaxed);
    }

    /// Get the configuration.
    pub fn config(&self) -> &RateLimitConfig {
        &self.config
    }

    /// Check if rate limiting is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

// ============================================================================
// STATISTICS
// ============================================================================

/// Internal statistics counters.
#[derive(Debug, Default)]
struct RateLimiterStats {
    total_requests: AtomicU64,
    allowed_requests: AtomicU64,
    rejected_requests: AtomicU64,
}

/// Snapshot of rate limiter statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RateLimiterStatsSnapshot {
    /// Total number of requests checked.
    pub total_requests: u64,
    /// Number of requests allowed.
    pub allowed_requests: u64,
    /// Number of requests rejected.
    pub rejected_requests: u64,
    /// Number of registered tenants.
    pub tenant_count: usize,
}

impl RateLimiterStatsSnapshot {
    /// Calculate the rejection rate.
    pub fn rejection_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.rejected_requests as f64 / self.total_requests as f64
        }
    }

    /// Calculate the acceptance rate.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_requests == 0 {
            1.0
        } else {
            self.allowed_requests as f64 / self.total_requests as f64
        }
    }
}

// ============================================================================
// RATE LIMIT GUARD (RAII)
// ============================================================================

/// RAII guard for rate-limited operations.
///
/// Automatically tracks the completion of rate-limited operations.
pub struct RateLimitGuard<'a> {
    limiter: &'a RateLimiter,
    tenant_id: Option<String>,
    _started: Instant,
}

impl<'a> RateLimitGuard<'a> {
    /// Create a new guard after successfully acquiring a rate limit token.
    fn new(limiter: &'a RateLimiter, tenant_id: Option<String>) -> Self {
        Self {
            limiter,
            tenant_id,
            _started: Instant::now(),
        }
    }

    /// Get the tenant ID if this is a tenant-scoped guard.
    pub fn tenant_id(&self) -> Option<&str> {
        self.tenant_id.as_deref()
    }
}

impl<'a> Drop for RateLimitGuard<'a> {
    fn drop(&mut self) {
        // Could track completion time here for metrics
        let _ = self.limiter;
    }
}

/// Extension trait for acquiring guards.
pub trait RateLimiterExt {
    /// Try to acquire a rate limit guard.
    fn try_acquire(&self) -> RateLimitResult<RateLimitGuard<'_>>;

    /// Try to acquire a tenant-scoped rate limit guard.
    fn try_acquire_tenant(&self, tenant_id: &str) -> RateLimitResult<RateLimitGuard<'_>>;
}

impl RateLimiterExt for RateLimiter {
    fn try_acquire(&self) -> RateLimitResult<RateLimitGuard<'_>> {
        self.check()?;
        Ok(RateLimitGuard::new(self, None))
    }

    fn try_acquire_tenant(&self, tenant_id: &str) -> RateLimitResult<RateLimitGuard<'_>> {
        self.check_tenant(tenant_id)?;
        Ok(RateLimitGuard::new(self, Some(tenant_id.to_string())))
    }
}

// ============================================================================
// RATE LIMITER BUILDER
// ============================================================================

/// Builder for creating rate limiters.
pub struct RateLimiterBuilder {
    config: RateLimitConfig,
    tenants: Vec<(String, u64)>,
}

impl RateLimiterBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: RateLimitConfig::default(),
            tenants: Vec::new(),
        }
    }

    /// Set the requests per second limit.
    pub fn with_requests_per_second(mut self, rps: u64) -> Self {
        self.config.requests_per_second = rps;
        self
    }

    /// Set the burst size.
    pub fn with_burst_size(mut self, size: u64) -> Self {
        self.config.burst_size = size;
        self
    }

    /// Set the algorithm.
    pub fn with_algorithm(mut self, algorithm: RateLimitAlgorithm) -> Self {
        self.config.algorithm = algorithm;
        self
    }

    /// Set the window size.
    pub fn with_window_size(mut self, size: Duration) -> Self {
        self.config.window_size = size;
        self
    }

    /// Enable or disable rate limiting.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Enable or disable per-tenant limiting.
    pub fn with_per_tenant(mut self, per_tenant: bool) -> Self {
        self.config.per_tenant = per_tenant;
        self
    }

    /// Add a tenant with a specific quota.
    pub fn with_tenant(mut self, tenant_id: impl Into<String>, quota: u64) -> Self {
        self.tenants.push((tenant_id.into(), quota));
        self
    }

    /// Build the rate limiter.
    pub fn build(self) -> RateLimiter {
        let limiter = RateLimiter::new(self.config);

        for (tenant_id, quota) in self.tenants {
            limiter.register_tenant(&tenant_id, quota);
        }

        limiter
    }
}

impl Default for RateLimiterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SHARED RATE LIMITER
// ============================================================================

/// Thread-safe, shareable rate limiter.
pub type SharedRateLimiter = Arc<RateLimiter>;

/// Create a shared rate limiter.
pub fn shared_rate_limiter(config: RateLimitConfig) -> SharedRateLimiter {
    Arc::new(RateLimiter::new(config))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.requests_per_second, 1000);
        assert_eq!(config.burst_size, 100);
        assert!(config.enabled);
        assert!(config.per_tenant);
    }

    #[test]
    fn test_rate_limit_config_builder() {
        let config = RateLimitConfig::new()
            .with_requests_per_second(500)
            .with_burst_size(50)
            .with_algorithm(RateLimitAlgorithm::SlidingWindow);

        assert_eq!(config.requests_per_second, 500);
        assert_eq!(config.burst_size, 50);
        assert_eq!(config.algorithm, RateLimitAlgorithm::SlidingWindow);
    }

    #[test]
    fn test_rate_limit_config_strict() {
        let config = RateLimitConfig::strict(100);
        assert_eq!(config.requests_per_second, 100);
        assert_eq!(config.burst_size, 10);
        assert_eq!(config.algorithm, RateLimitAlgorithm::SlidingWindow);
    }

    #[test]
    fn test_rate_limiter_allows_within_limit() {
        let config = RateLimitConfig::new()
            .with_requests_per_second(100)
            .with_burst_size(10);

        let limiter = RateLimiter::new(config);

        // Should allow up to burst_size requests
        for _ in 0..5 {
            assert!(limiter.check().is_ok());
        }
    }

    #[test]
    fn test_rate_limiter_rejects_over_limit() {
        let config = RateLimitConfig::new()
            .with_requests_per_second(100)
            .with_burst_size(5)
            .with_algorithm(RateLimitAlgorithm::TokenBucket);

        let limiter = RateLimiter::new(config);

        // Exhaust the bucket
        for _ in 0..5 {
            assert!(limiter.check().is_ok());
        }

        // Should reject
        let result = limiter.check();
        assert!(matches!(
            result,
            Err(RateLimitError::RateLimitExceeded { .. })
        ));
    }

    #[test]
    fn test_rate_limiter_disabled() {
        let config = RateLimitConfig::new().with_enabled(false);
        let limiter = RateLimiter::new(config);

        let result = limiter.check();
        assert!(matches!(result, Err(RateLimitError::Disabled)));
    }

    #[test]
    fn test_rate_limiter_tenant() {
        let config = RateLimitConfig::new()
            .with_requests_per_second(1000)
            .with_burst_size(100)
            .with_per_tenant(true)
            .with_default_tenant_quota(5);

        let limiter = RateLimiter::new(config);

        // Auto-register tenant on first request
        for _ in 0..5 {
            assert!(limiter.check_tenant("tenant_1").is_ok());
        }

        // Tenant limit should be hit
        let result = limiter.check_tenant("tenant_1");
        assert!(matches!(
            result,
            Err(RateLimitError::RateLimitExceeded { .. })
        ));

        // Different tenant should still work
        assert!(limiter.check_tenant("tenant_2").is_ok());
    }

    #[test]
    fn test_rate_limiter_register_tenant() {
        let config = RateLimitConfig::new()
            .with_requests_per_second(1000)
            .with_burst_size(100);

        let limiter = RateLimiter::new(config);
        limiter.register_tenant("tenant_1", 10);

        assert_eq!(limiter.tenant_count(), 1);

        for _ in 0..10 {
            assert!(limiter.check_tenant("tenant_1").is_ok());
        }
    }

    #[test]
    fn test_rate_limiter_stats() {
        let config = RateLimitConfig::new()
            .with_requests_per_second(100)
            .with_burst_size(5);

        let limiter = RateLimiter::new(config);

        for _ in 0..5 {
            let _ = limiter.check();
        }
        // This should be rejected
        let _ = limiter.check();

        let stats = limiter.stats();
        assert_eq!(stats.total_requests, 6);
        assert_eq!(stats.allowed_requests, 5);
        assert_eq!(stats.rejected_requests, 1);
    }

    #[test]
    fn test_rate_limiter_stats_snapshot() {
        let stats = RateLimiterStatsSnapshot {
            total_requests: 100,
            allowed_requests: 80,
            rejected_requests: 20,
            tenant_count: 5,
        };

        assert!((stats.rejection_rate() - 0.2).abs() < f64::EPSILON);
        assert!((stats.acceptance_rate() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sliding_window() {
        let config = RateLimitConfig::new()
            .with_algorithm(RateLimitAlgorithm::SlidingWindow)
            .with_requests_per_second(5)
            .with_window_size(Duration::from_secs(1));

        let limiter = RateLimiter::new(config);

        for _ in 0..5 {
            assert!(limiter.check().is_ok());
        }

        let result = limiter.check();
        assert!(matches!(
            result,
            Err(RateLimitError::RateLimitExceeded { .. })
        ));
    }

    #[test]
    fn test_fixed_window() {
        let config = RateLimitConfig::new()
            .with_algorithm(RateLimitAlgorithm::FixedWindow)
            .with_requests_per_second(5)
            .with_window_size(Duration::from_secs(1));

        let limiter = RateLimiter::new(config);

        for _ in 0..5 {
            assert!(limiter.check().is_ok());
        }

        let result = limiter.check();
        assert!(matches!(
            result,
            Err(RateLimitError::RateLimitExceeded { .. })
        ));
    }

    #[test]
    fn test_leaky_bucket() {
        let config = RateLimitConfig::new()
            .with_algorithm(RateLimitAlgorithm::LeakyBucket)
            .with_requests_per_second(100)
            .with_burst_size(5);

        let limiter = RateLimiter::new(config);

        for _ in 0..5 {
            assert!(limiter.check().is_ok());
        }

        let result = limiter.check();
        assert!(matches!(
            result,
            Err(RateLimitError::RateLimitExceeded { .. })
        ));
    }

    #[test]
    fn test_rate_limiter_builder() {
        let limiter = RateLimiterBuilder::new()
            .with_requests_per_second(500)
            .with_burst_size(50)
            .with_tenant("tenant_1", 100)
            .with_tenant("tenant_2", 200)
            .build();

        assert_eq!(limiter.tenant_count(), 2);
        assert_eq!(limiter.config().requests_per_second, 500);
    }

    #[test]
    fn test_update_tenant_quota() {
        let limiter = RateLimiterBuilder::new().with_tenant("tenant_1", 5).build();

        // Use up initial quota
        for _ in 0..5 {
            assert!(limiter.check_tenant("tenant_1").is_ok());
        }

        // Should fail
        assert!(limiter.check_tenant("tenant_1").is_err());

        // Update quota
        assert!(limiter.update_tenant_quota("tenant_1", 10).is_ok());

        // Should work again
        for _ in 0..10 {
            assert!(limiter.check_tenant("tenant_1").is_ok());
        }
    }

    #[test]
    fn test_remove_tenant() {
        let limiter = RateLimiterBuilder::new()
            .with_tenant("tenant_1", 100)
            .build();

        assert_eq!(limiter.tenant_count(), 1);
        assert!(limiter.remove_tenant("tenant_1"));
        assert_eq!(limiter.tenant_count(), 0);
        assert!(!limiter.remove_tenant("tenant_1")); // Already removed
    }

    #[test]
    fn test_rate_limit_guard() {
        let config = RateLimitConfig::new().with_burst_size(10);
        let limiter = RateLimiter::new(config);

        {
            let guard = limiter.try_acquire().unwrap();
            assert!(guard.tenant_id().is_none());
        }

        limiter.register_tenant("tenant_1", 10);
        {
            let guard = limiter.try_acquire_tenant("tenant_1").unwrap();
            assert_eq!(guard.tenant_id(), Some("tenant_1"));
        }
    }
}
