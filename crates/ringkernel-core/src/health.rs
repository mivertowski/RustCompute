//! Health monitoring and resilience infrastructure for RingKernel.
//!
//! This module provides production-ready health and resilience features:
//!
//! - **Health Checks** - Kernel liveness and readiness probes
//! - **Circuit Breakers** - Fault isolation and recovery
//! - **Retry Policies** - Configurable retry with backoff
//! - **Graceful Degradation** - Load shedding and fallback modes
//! - **Watchdog** - Automatic kernel health monitoring
//!
//! ## Usage
//!
//! ```ignore
//! use ringkernel_core::health::{HealthChecker, CircuitBreaker, RetryPolicy};
//!
//! // Create health checker
//! let checker = HealthChecker::new()
//!     .liveness_check("kernel_alive", || async { true })
//!     .readiness_check("queue_ready", || async { queue_depth < 1000 });
//!
//! // Create circuit breaker
//! let breaker = CircuitBreaker::new()
//!     .failure_threshold(5)
//!     .recovery_timeout(Duration::from_secs(30));
//!
//! // Execute with circuit breaker
//! let result = breaker.execute(|| async { risky_operation() }).await;
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{Result, RingKernelError};
use crate::runtime::KernelId;

// ============================================================================
// Health Check Types
// ============================================================================

/// Health status of a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Component is healthy and operating normally.
    Healthy,
    /// Component is degraded but still functional.
    Degraded,
    /// Component is unhealthy and not functional.
    Unhealthy,
    /// Health status is unknown (check not yet run).
    Unknown,
}

impl HealthStatus {
    /// Check if status represents a healthy state.
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy | HealthStatus::Degraded)
    }

    /// Check if status represents an unhealthy state.
    pub fn is_unhealthy(&self) -> bool {
        matches!(self, HealthStatus::Unhealthy)
    }
}

/// Result of a health check.
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Check name.
    pub name: String,
    /// Health status.
    pub status: HealthStatus,
    /// Human-readable message.
    pub message: Option<String>,
    /// Duration of the check.
    pub duration: Duration,
    /// Timestamp when check was performed.
    pub checked_at: Instant,
}

/// Type alias for async health check functions.
pub type HealthCheckFn =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = HealthStatus> + Send>> + Send + Sync>;

/// A health check definition.
pub struct HealthCheck {
    /// Check name.
    pub name: String,
    /// Check function.
    check_fn: HealthCheckFn,
    /// Whether this is a liveness check.
    pub is_liveness: bool,
    /// Whether this is a readiness check.
    pub is_readiness: bool,
    /// Timeout for check execution.
    pub timeout: Duration,
    /// Last result.
    last_result: RwLock<Option<HealthCheckResult>>,
}

impl HealthCheck {
    /// Create a new health check.
    pub fn new(name: impl Into<String>, check_fn: HealthCheckFn) -> Self {
        Self {
            name: name.into(),
            check_fn,
            is_liveness: false,
            is_readiness: false,
            timeout: Duration::from_secs(5),
            last_result: RwLock::new(None),
        }
    }

    /// Mark as liveness check.
    pub fn liveness(mut self) -> Self {
        self.is_liveness = true;
        self
    }

    /// Mark as readiness check.
    pub fn readiness(mut self) -> Self {
        self.is_readiness = true;
        self
    }

    /// Set timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Execute the health check.
    pub async fn check(&self) -> HealthCheckResult {
        let start = Instant::now();
        let status = (self.check_fn)().await;
        let duration = start.elapsed();

        let result = HealthCheckResult {
            name: self.name.clone(),
            status,
            message: None,
            duration,
            checked_at: Instant::now(),
        };

        *self.last_result.write() = Some(result.clone());
        result
    }

    /// Get last check result.
    pub fn last_result(&self) -> Option<HealthCheckResult> {
        self.last_result.read().clone()
    }
}

/// Health checker that manages multiple health checks.
pub struct HealthChecker {
    /// Registered health checks.
    checks: RwLock<Vec<Arc<HealthCheck>>>,
    /// Check interval (used by async runtime loop).
    #[allow(dead_code)]
    check_interval: Duration,
    /// Running state (used by async runtime loop).
    #[allow(dead_code)]
    running: std::sync::atomic::AtomicBool,
}

impl HealthChecker {
    /// Create a new health checker.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            checks: RwLock::new(Vec::new()),
            check_interval: Duration::from_secs(10),
            running: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Set check interval.
    pub fn with_interval(self: Arc<Self>, interval: Duration) -> Arc<Self> {
        // Note: This would require interior mutability or builder pattern
        // For now, we just use the default
        let _ = interval;
        self
    }

    /// Register a health check.
    pub fn register(&self, check: HealthCheck) {
        self.checks.write().push(Arc::new(check));
    }

    /// Register a simple liveness check.
    pub fn register_liveness<F, Fut>(&self, name: impl Into<String>, check_fn: F)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        let name = name.into();
        let check = HealthCheck::new(
            name,
            Arc::new(move || {
                let fut = check_fn();
                Box::pin(async move {
                    if fut.await {
                        HealthStatus::Healthy
                    } else {
                        HealthStatus::Unhealthy
                    }
                }) as Pin<Box<dyn Future<Output = HealthStatus> + Send>>
            }),
        )
        .liveness();
        self.register(check);
    }

    /// Register a simple readiness check.
    pub fn register_readiness<F, Fut>(&self, name: impl Into<String>, check_fn: F)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        let name = name.into();
        let check = HealthCheck::new(
            name,
            Arc::new(move || {
                let fut = check_fn();
                Box::pin(async move {
                    if fut.await {
                        HealthStatus::Healthy
                    } else {
                        HealthStatus::Unhealthy
                    }
                }) as Pin<Box<dyn Future<Output = HealthStatus> + Send>>
            }),
        )
        .readiness();
        self.register(check);
    }

    /// Run all health checks.
    pub async fn check_all(&self) -> Vec<HealthCheckResult> {
        let checks = self.checks.read().clone();
        let mut results = Vec::with_capacity(checks.len());

        for check in checks {
            results.push(check.check().await);
        }

        results
    }

    /// Run liveness checks only.
    pub async fn check_liveness(&self) -> Vec<HealthCheckResult> {
        let checks = self.checks.read().clone();
        let mut results = Vec::new();

        for check in checks.iter().filter(|c| c.is_liveness) {
            results.push(check.check().await);
        }

        results
    }

    /// Run readiness checks only.
    pub async fn check_readiness(&self) -> Vec<HealthCheckResult> {
        let checks = self.checks.read().clone();
        let mut results = Vec::new();

        for check in checks.iter().filter(|c| c.is_readiness) {
            results.push(check.check().await);
        }

        results
    }

    /// Get overall liveness status.
    pub async fn is_alive(&self) -> bool {
        let results = self.check_liveness().await;
        results.iter().all(|r| r.status.is_healthy())
    }

    /// Get overall readiness status.
    pub async fn is_ready(&self) -> bool {
        let results = self.check_readiness().await;
        results.iter().all(|r| r.status.is_healthy())
    }

    /// Get aggregate health status.
    pub async fn aggregate_status(&self) -> HealthStatus {
        let results = self.check_all().await;

        if results.is_empty() {
            return HealthStatus::Unknown;
        }

        let all_healthy = results.iter().all(|r| r.status == HealthStatus::Healthy);
        let any_unhealthy = results.iter().any(|r| r.status == HealthStatus::Unhealthy);

        if all_healthy {
            HealthStatus::Healthy
        } else if any_unhealthy {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        }
    }

    /// Get check count.
    pub fn check_count(&self) -> usize {
        self.checks.read().len()
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self {
            checks: RwLock::new(Vec::new()),
            check_interval: Duration::from_secs(10),
            running: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

// ============================================================================
// Circuit Breaker
// ============================================================================

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed (allowing requests).
    Closed,
    /// Circuit is open (rejecting requests).
    Open,
    /// Circuit is half-open (allowing test requests).
    HalfOpen,
}

/// Circuit breaker configuration.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit.
    pub failure_threshold: u32,
    /// Number of successes to close circuit from half-open.
    pub success_threshold: u32,
    /// Duration to wait before transitioning from open to half-open.
    pub recovery_timeout: Duration,
    /// Duration of sliding window for counting failures.
    pub window_duration: Duration,
    /// Maximum concurrent requests in half-open state.
    pub half_open_max_requests: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            recovery_timeout: Duration::from_secs(30),
            window_duration: Duration::from_secs(60),
            half_open_max_requests: 3,
        }
    }
}

/// Circuit breaker for fault isolation.
pub struct CircuitBreaker {
    /// Configuration.
    config: CircuitBreakerConfig,
    /// Current state.
    state: RwLock<CircuitState>,
    /// Failure count in current window.
    failure_count: AtomicU32,
    /// Success count in half-open state.
    success_count: AtomicU32,
    /// Time when circuit opened.
    opened_at: RwLock<Option<Instant>>,
    /// Current requests in half-open state.
    half_open_requests: AtomicU32,
    /// Total requests.
    total_requests: AtomicU64,
    /// Total failures.
    total_failures: AtomicU64,
    /// Total rejections (due to open circuit).
    total_rejections: AtomicU64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with default config.
    pub fn new() -> Arc<Self> {
        Self::with_config(CircuitBreakerConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(config: CircuitBreakerConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            opened_at: RwLock::new(None),
            half_open_requests: AtomicU32::new(0),
            total_requests: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            total_rejections: AtomicU64::new(0),
        })
    }

    /// Get current state.
    pub fn state(&self) -> CircuitState {
        // Check if we should transition from open to half-open
        let current_state = *self.state.read();
        if current_state == CircuitState::Open {
            if let Some(opened_at) = *self.opened_at.read() {
                if opened_at.elapsed() >= self.config.recovery_timeout {
                    *self.state.write() = CircuitState::HalfOpen;
                    self.half_open_requests.store(0, Ordering::SeqCst);
                    self.success_count.store(0, Ordering::SeqCst);
                    return CircuitState::HalfOpen;
                }
            }
        }
        current_state
    }

    /// Check if circuit allows requests.
    pub fn is_allowed(&self) -> bool {
        match self.state() {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen => {
                self.half_open_requests.load(Ordering::SeqCst) < self.config.half_open_max_requests
            }
        }
    }

    /// Record a successful operation.
    pub fn record_success(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let state = self.state();
        if state == CircuitState::HalfOpen {
            let success_count = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
            self.half_open_requests.fetch_sub(1, Ordering::SeqCst);

            if success_count >= self.config.success_threshold {
                self.close();
            }
        }
    }

    /// Record a failed operation.
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_failures.fetch_add(1, Ordering::Relaxed);

        let state = self.state();
        match state {
            CircuitState::Closed => {
                let failure_count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
                if failure_count >= self.config.failure_threshold {
                    self.open();
                }
            }
            CircuitState::HalfOpen => {
                self.half_open_requests.fetch_sub(1, Ordering::SeqCst);
                self.open();
            }
            CircuitState::Open => {}
        }
    }

    /// Record a rejection (request not attempted due to open circuit).
    pub fn record_rejection(&self) {
        self.total_rejections.fetch_add(1, Ordering::Relaxed);
    }

    /// Open the circuit.
    fn open(&self) {
        *self.state.write() = CircuitState::Open;
        *self.opened_at.write() = Some(Instant::now());
    }

    /// Close the circuit.
    fn close(&self) {
        *self.state.write() = CircuitState::Closed;
        *self.opened_at.write() = None;
        self.failure_count.store(0, Ordering::SeqCst);
        self.success_count.store(0, Ordering::SeqCst);
    }

    /// Force reset the circuit to closed state.
    pub fn reset(&self) {
        self.close();
    }

    /// Acquire permission to execute (for half-open state).
    fn acquire_half_open(&self) -> bool {
        if self.state() != CircuitState::HalfOpen {
            return true;
        }

        let current = self.half_open_requests.load(Ordering::SeqCst);
        if current >= self.config.half_open_max_requests {
            return false;
        }

        self.half_open_requests.fetch_add(1, Ordering::SeqCst);
        true
    }

    /// Execute an operation with circuit breaker protection.
    pub async fn execute<F, Fut, T, E>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = std::result::Result<T, E>>,
        E: std::fmt::Display,
    {
        if !self.is_allowed() {
            self.record_rejection();
            return Err(RingKernelError::BackendError(
                "Circuit breaker is open".to_string(),
            ));
        }

        if !self.acquire_half_open() {
            self.record_rejection();
            return Err(RingKernelError::BackendError(
                "Circuit breaker half-open limit reached".to_string(),
            ));
        }

        match operation().await {
            Ok(result) => {
                self.record_success();
                Ok(result)
            }
            Err(e) => {
                self.record_failure();
                Err(RingKernelError::BackendError(format!(
                    "Operation failed: {}",
                    e
                )))
            }
        }
    }

    /// Get circuit breaker statistics.
    pub fn stats(&self) -> CircuitBreakerStats {
        CircuitBreakerStats {
            state: self.state(),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_failures: self.total_failures.load(Ordering::Relaxed),
            total_rejections: self.total_rejections.load(Ordering::Relaxed),
            failure_count: self.failure_count.load(Ordering::Relaxed),
            success_count: self.success_count.load(Ordering::Relaxed),
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            config: CircuitBreakerConfig::default(),
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            opened_at: RwLock::new(None),
            half_open_requests: AtomicU32::new(0),
            total_requests: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            total_rejections: AtomicU64::new(0),
        }
    }
}

/// Circuit breaker statistics.
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    /// Current state.
    pub state: CircuitState,
    /// Total requests attempted.
    pub total_requests: u64,
    /// Total failures.
    pub total_failures: u64,
    /// Total rejections.
    pub total_rejections: u64,
    /// Current failure count.
    pub failure_count: u32,
    /// Current success count (in half-open).
    pub success_count: u32,
}

// ============================================================================
// Retry Policy
// ============================================================================

/// Backoff strategy for retries.
#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    /// Fixed delay between retries.
    Fixed(Duration),
    /// Linear backoff (delay * attempt).
    Linear {
        /// Initial delay.
        initial: Duration,
        /// Maximum delay.
        max: Duration,
    },
    /// Exponential backoff (delay * 2^attempt).
    Exponential {
        /// Initial delay.
        initial: Duration,
        /// Maximum delay.
        max: Duration,
        /// Multiplier (default 2.0).
        multiplier: f64,
    },
    /// No delay between retries.
    None,
}

impl BackoffStrategy {
    /// Calculate delay for given attempt number (0-indexed).
    pub fn delay(&self, attempt: u32) -> Duration {
        match self {
            BackoffStrategy::Fixed(d) => *d,
            BackoffStrategy::Linear { initial, max } => {
                let delay = initial.mul_f64((attempt + 1) as f64);
                delay.min(*max)
            }
            BackoffStrategy::Exponential {
                initial,
                max,
                multiplier,
            } => {
                let factor = multiplier.powi(attempt as i32);
                let delay = initial.mul_f64(factor);
                delay.min(*max)
            }
            BackoffStrategy::None => Duration::ZERO,
        }
    }
}

/// Retry policy configuration.
#[derive(Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts.
    pub max_attempts: u32,
    /// Backoff strategy.
    pub backoff: BackoffStrategy,
    /// Whether to add jitter to delays.
    pub jitter: bool,
    /// Retryable error predicate.
    #[allow(clippy::type_complexity)]
    retryable: Option<Arc<dyn Fn(&str) -> bool + Send + Sync>>,
}

impl std::fmt::Debug for RetryPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryPolicy")
            .field("max_attempts", &self.max_attempts)
            .field("backoff", &self.backoff)
            .field("jitter", &self.jitter)
            .field("retryable", &self.retryable.is_some())
            .finish()
    }
}

impl RetryPolicy {
    /// Create a new retry policy.
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            backoff: BackoffStrategy::Exponential {
                initial: Duration::from_millis(100),
                max: Duration::from_secs(30),
                multiplier: 2.0,
            },
            jitter: true,
            retryable: None,
        }
    }

    /// Set backoff strategy.
    pub fn with_backoff(mut self, backoff: BackoffStrategy) -> Self {
        self.backoff = backoff;
        self
    }

    /// Disable jitter.
    pub fn without_jitter(mut self) -> Self {
        self.jitter = false;
        self
    }

    /// Set retryable error predicate.
    pub fn with_retryable<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.retryable = Some(Arc::new(predicate));
        self
    }

    /// Check if an error is retryable.
    pub fn is_retryable(&self, error: &str) -> bool {
        self.retryable.as_ref().map(|p| p(error)).unwrap_or(true)
    }

    /// Get delay for an attempt with optional jitter.
    pub fn get_delay(&self, attempt: u32) -> Duration {
        let base_delay = self.backoff.delay(attempt);

        if self.jitter && base_delay > Duration::ZERO {
            // Add up to 25% jitter
            let jitter_factor = 0.75 + (rand_u64() % 50) as f64 / 200.0;
            base_delay.mul_f64(jitter_factor)
        } else {
            base_delay
        }
    }

    /// Execute an operation with retry.
    pub async fn execute<F, Fut, T, E>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = std::result::Result<T, E>>,
        E: std::fmt::Display,
    {
        let mut last_error = String::new();

        for attempt in 0..self.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = format!("{}", e);

                    // Check if retryable
                    if !self.is_retryable(&last_error) {
                        return Err(RingKernelError::BackendError(format!(
                            "Non-retryable error: {}",
                            last_error
                        )));
                    }

                    // Last attempt, don't wait
                    if attempt + 1 >= self.max_attempts {
                        break;
                    }

                    // Wait before retry
                    let delay = self.get_delay(attempt);
                    tokio::time::sleep(delay).await;
                }
            }
        }

        Err(RingKernelError::BackendError(format!(
            "Operation failed after {} attempts: {}",
            self.max_attempts, last_error
        )))
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new(3)
    }
}

/// Simple pseudo-random number generator for jitter.
fn rand_u64() -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::time::SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    hasher.finish()
}

// ============================================================================
// Graceful Degradation
// ============================================================================

/// Degradation level for system operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DegradationLevel {
    /// Full functionality.
    Normal = 0,
    /// Minor degradation (e.g., increased latency acceptable).
    Light = 1,
    /// Moderate degradation (e.g., some features disabled).
    Moderate = 2,
    /// Severe degradation (e.g., read-only mode).
    Severe = 3,
    /// Critical (e.g., emergency mode only).
    Critical = 4,
}

impl DegradationLevel {
    /// Get the next worse degradation level.
    ///
    /// Returns Critical if already at Critical.
    pub fn next_worse(self) -> Self {
        match self {
            DegradationLevel::Normal => DegradationLevel::Light,
            DegradationLevel::Light => DegradationLevel::Moderate,
            DegradationLevel::Moderate => DegradationLevel::Severe,
            DegradationLevel::Severe => DegradationLevel::Critical,
            DegradationLevel::Critical => DegradationLevel::Critical,
        }
    }

    /// Get the next better degradation level.
    ///
    /// Returns Normal if already at Normal.
    pub fn next_better(self) -> Self {
        match self {
            DegradationLevel::Normal => DegradationLevel::Normal,
            DegradationLevel::Light => DegradationLevel::Normal,
            DegradationLevel::Moderate => DegradationLevel::Light,
            DegradationLevel::Severe => DegradationLevel::Moderate,
            DegradationLevel::Critical => DegradationLevel::Severe,
        }
    }
}

/// Load shedding policy.
#[derive(Debug, Clone)]
pub struct LoadSheddingPolicy {
    /// Queue depth threshold for shedding.
    pub queue_threshold: usize,
    /// CPU utilization threshold (0.0-1.0).
    pub cpu_threshold: f64,
    /// Memory utilization threshold (0.0-1.0).
    pub memory_threshold: f64,
    /// Percentage of requests to shed (0.0-1.0).
    pub shed_ratio: f64,
}

impl Default for LoadSheddingPolicy {
    fn default() -> Self {
        Self {
            queue_threshold: 10000,
            cpu_threshold: 0.9,
            memory_threshold: 0.85,
            shed_ratio: 0.1,
        }
    }
}

/// Graceful degradation manager.
pub struct DegradationManager {
    /// Current degradation level.
    level: RwLock<DegradationLevel>,
    /// Load shedding policy.
    policy: LoadSheddingPolicy,
    /// Level change callbacks.
    #[allow(clippy::type_complexity)]
    callbacks: RwLock<Vec<Arc<dyn Fn(DegradationLevel, DegradationLevel) + Send + Sync>>>,
    /// Shed counter for probabilistic shedding.
    shed_counter: AtomicU64,
    /// Total requests.
    total_requests: AtomicU64,
    /// Shed requests.
    shed_requests: AtomicU64,
}

impl DegradationManager {
    /// Create a new degradation manager.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            level: RwLock::new(DegradationLevel::Normal),
            policy: LoadSheddingPolicy::default(),
            callbacks: RwLock::new(Vec::new()),
            shed_counter: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            shed_requests: AtomicU64::new(0),
        })
    }

    /// Create with custom policy.
    pub fn with_policy(policy: LoadSheddingPolicy) -> Arc<Self> {
        Arc::new(Self {
            level: RwLock::new(DegradationLevel::Normal),
            policy,
            callbacks: RwLock::new(Vec::new()),
            shed_counter: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            shed_requests: AtomicU64::new(0),
        })
    }

    /// Get current degradation level.
    pub fn level(&self) -> DegradationLevel {
        *self.level.read()
    }

    /// Set degradation level.
    pub fn set_level(&self, new_level: DegradationLevel) {
        let old_level = *self.level.read();
        if old_level != new_level {
            *self.level.write() = new_level;

            // Notify callbacks
            let callbacks = self.callbacks.read().clone();
            for callback in callbacks {
                callback(old_level, new_level);
            }
        }
    }

    /// Register level change callback.
    pub fn on_level_change<F>(&self, callback: F)
    where
        F: Fn(DegradationLevel, DegradationLevel) + Send + Sync + 'static,
    {
        self.callbacks.write().push(Arc::new(callback));
    }

    /// Check if request should be shed.
    pub fn should_shed(&self) -> bool {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let level = self.level();
        if level == DegradationLevel::Normal {
            return false;
        }

        // Increase shed probability based on degradation level
        let base_ratio = self.policy.shed_ratio;
        let level_factor = match level {
            DegradationLevel::Normal => 0.0,
            DegradationLevel::Light => 1.0,
            DegradationLevel::Moderate => 2.0,
            DegradationLevel::Severe => 3.0,
            DegradationLevel::Critical => 4.0,
        };

        let shed_probability = (base_ratio * level_factor).min(0.9);

        // Probabilistic shedding
        let counter = self.shed_counter.fetch_add(1, Ordering::Relaxed);
        let should_shed = (counter % 100) < (shed_probability * 100.0) as u64;

        if should_shed {
            self.shed_requests.fetch_add(1, Ordering::Relaxed);
        }

        should_shed
    }

    /// Check if a feature should be disabled at current level.
    pub fn is_feature_disabled(&self, required_level: DegradationLevel) -> bool {
        self.level() > required_level
    }

    /// Get shedding statistics.
    pub fn stats(&self) -> DegradationStats {
        let total = self.total_requests.load(Ordering::Relaxed);
        let shed = self.shed_requests.load(Ordering::Relaxed);

        DegradationStats {
            level: self.level(),
            total_requests: total,
            shed_requests: shed,
            shed_ratio: if total > 0 {
                shed as f64 / total as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for DegradationManager {
    fn default() -> Self {
        Self {
            level: RwLock::new(DegradationLevel::Normal),
            policy: LoadSheddingPolicy::default(),
            callbacks: RwLock::new(Vec::new()),
            shed_counter: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            shed_requests: AtomicU64::new(0),
        }
    }
}

/// Degradation statistics.
#[derive(Debug, Clone)]
pub struct DegradationStats {
    /// Current level.
    pub level: DegradationLevel,
    /// Total requests.
    pub total_requests: u64,
    /// Shed requests.
    pub shed_requests: u64,
    /// Actual shed ratio.
    pub shed_ratio: f64,
}

// ============================================================================
// Kernel Health Watchdog
// ============================================================================

/// Kernel health status for watchdog.
#[derive(Debug, Clone)]
pub struct KernelHealth {
    /// Kernel ID.
    pub kernel_id: KernelId,
    /// Last heartbeat time.
    pub last_heartbeat: Instant,
    /// Health status.
    pub status: HealthStatus,
    /// Consecutive failure count.
    pub failure_count: u32,
    /// Message processing rate.
    pub messages_per_sec: f64,
    /// Current queue depth.
    pub queue_depth: usize,
}

/// Watchdog for monitoring kernel health.
pub struct KernelWatchdog {
    /// Watched kernels.
    kernels: RwLock<HashMap<KernelId, KernelHealth>>,
    /// Heartbeat timeout.
    heartbeat_timeout: Duration,
    /// Check interval (used by async runtime loop).
    #[allow(dead_code)]
    check_interval: Duration,
    /// Failure threshold before marking unhealthy.
    failure_threshold: u32,
    /// Running state (used by async runtime loop).
    #[allow(dead_code)]
    running: std::sync::atomic::AtomicBool,
    /// Unhealthy kernel callbacks.
    #[allow(clippy::type_complexity)]
    callbacks: RwLock<Vec<Arc<dyn Fn(&KernelHealth) + Send + Sync>>>,
}

impl KernelWatchdog {
    /// Create a new kernel watchdog.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            kernels: RwLock::new(HashMap::new()),
            heartbeat_timeout: Duration::from_secs(30),
            check_interval: Duration::from_secs(5),
            failure_threshold: 3,
            running: std::sync::atomic::AtomicBool::new(false),
            callbacks: RwLock::new(Vec::new()),
        })
    }

    /// Set heartbeat timeout.
    pub fn with_heartbeat_timeout(self: Arc<Self>, timeout: Duration) -> Arc<Self> {
        let _ = timeout; // Would need interior mutability
        self
    }

    /// Register a kernel to watch.
    pub fn watch(&self, kernel_id: KernelId) {
        let health = KernelHealth {
            kernel_id: kernel_id.clone(),
            last_heartbeat: Instant::now(),
            status: HealthStatus::Healthy,
            failure_count: 0,
            messages_per_sec: 0.0,
            queue_depth: 0,
        };
        self.kernels.write().insert(kernel_id, health);
    }

    /// Unregister a kernel.
    pub fn unwatch(&self, kernel_id: &KernelId) {
        self.kernels.write().remove(kernel_id);
    }

    /// Record heartbeat from kernel.
    pub fn heartbeat(&self, kernel_id: &KernelId) {
        if let Some(health) = self.kernels.write().get_mut(kernel_id) {
            health.last_heartbeat = Instant::now();
            health.failure_count = 0;
            if health.status == HealthStatus::Unhealthy {
                health.status = HealthStatus::Healthy;
            }
        }
    }

    /// Update kernel metrics.
    pub fn update_metrics(&self, kernel_id: &KernelId, messages_per_sec: f64, queue_depth: usize) {
        if let Some(health) = self.kernels.write().get_mut(kernel_id) {
            health.messages_per_sec = messages_per_sec;
            health.queue_depth = queue_depth;
        }
    }

    /// Check all kernel health.
    pub fn check_all(&self) -> Vec<KernelHealth> {
        let now = Instant::now();
        let mut kernels = self.kernels.write();
        let mut results = Vec::with_capacity(kernels.len());

        for health in kernels.values_mut() {
            // Check heartbeat timeout
            if now.duration_since(health.last_heartbeat) > self.heartbeat_timeout {
                health.failure_count += 1;
                if health.failure_count >= self.failure_threshold {
                    health.status = HealthStatus::Unhealthy;
                } else {
                    health.status = HealthStatus::Degraded;
                }
            }

            results.push(health.clone());
        }

        // Notify callbacks for unhealthy kernels
        drop(kernels);
        let callbacks = self.callbacks.read().clone();
        for health in results
            .iter()
            .filter(|h| h.status == HealthStatus::Unhealthy)
        {
            for callback in &callbacks {
                callback(health);
            }
        }

        results
    }

    /// Register unhealthy kernel callback.
    pub fn on_unhealthy<F>(&self, callback: F)
    where
        F: Fn(&KernelHealth) + Send + Sync + 'static,
    {
        self.callbacks.write().push(Arc::new(callback));
    }

    /// Get health for specific kernel.
    pub fn get_health(&self, kernel_id: &KernelId) -> Option<KernelHealth> {
        self.kernels.read().get(kernel_id).cloned()
    }

    /// Get all unhealthy kernels.
    pub fn unhealthy_kernels(&self) -> Vec<KernelHealth> {
        self.kernels
            .read()
            .values()
            .filter(|h| h.status == HealthStatus::Unhealthy)
            .cloned()
            .collect()
    }

    /// Get watched kernel count.
    pub fn watched_count(&self) -> usize {
        self.kernels.read().len()
    }
}

impl Default for KernelWatchdog {
    fn default() -> Self {
        Self {
            kernels: RwLock::new(HashMap::new()),
            heartbeat_timeout: Duration::from_secs(30),
            check_interval: Duration::from_secs(5),
            failure_threshold: 3,
            running: std::sync::atomic::AtomicBool::new(false),
            callbacks: RwLock::new(Vec::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(HealthStatus::Degraded.is_healthy());
        assert!(!HealthStatus::Unhealthy.is_healthy());
        assert!(HealthStatus::Unhealthy.is_unhealthy());
    }

    #[tokio::test]
    async fn test_health_checker() {
        let checker = HealthChecker::new();

        checker.register_liveness("test_alive", || async { true });
        checker.register_readiness("test_ready", || async { true });

        assert_eq!(checker.check_count(), 2);
        assert!(checker.is_alive().await);
        assert!(checker.is_ready().await);
    }

    #[tokio::test]
    async fn test_health_checker_unhealthy() {
        let checker = HealthChecker::new();

        checker.register_liveness("failing_check", || async { false });

        assert!(!checker.is_alive().await);

        let status = checker.aggregate_status().await;
        assert_eq!(status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_circuit_breaker_initial_state() {
        let breaker = CircuitBreaker::new();
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert!(breaker.is_allowed());
    }

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let breaker = CircuitBreaker::with_config(config);

        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(!breaker.is_allowed());
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };
        let breaker = CircuitBreaker::with_config(config);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);

        breaker.reset();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_backoff_strategy_fixed() {
        let backoff = BackoffStrategy::Fixed(Duration::from_secs(1));
        assert_eq!(backoff.delay(0), Duration::from_secs(1));
        assert_eq!(backoff.delay(5), Duration::from_secs(1));
    }

    #[test]
    fn test_backoff_strategy_exponential() {
        let backoff = BackoffStrategy::Exponential {
            initial: Duration::from_millis(100),
            max: Duration::from_secs(10),
            multiplier: 2.0,
        };

        assert_eq!(backoff.delay(0), Duration::from_millis(100));
        assert_eq!(backoff.delay(1), Duration::from_millis(200));
        assert_eq!(backoff.delay(2), Duration::from_millis(400));
    }

    #[test]
    fn test_backoff_strategy_linear() {
        let backoff = BackoffStrategy::Linear {
            initial: Duration::from_millis(100),
            max: Duration::from_secs(1),
        };

        assert_eq!(backoff.delay(0), Duration::from_millis(100));
        assert_eq!(backoff.delay(1), Duration::from_millis(200));
        assert_eq!(backoff.delay(9), Duration::from_secs(1)); // Capped
    }

    #[tokio::test]
    async fn test_retry_policy_success() {
        let policy = RetryPolicy::new(3);

        let result: Result<i32> = policy.execute(|| async { Ok::<_, &str>(42) }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_degradation_manager_levels() {
        let manager = DegradationManager::new();

        assert_eq!(manager.level(), DegradationLevel::Normal);

        manager.set_level(DegradationLevel::Moderate);
        assert_eq!(manager.level(), DegradationLevel::Moderate);
    }

    #[test]
    fn test_degradation_feature_disabled() {
        let manager = DegradationManager::new();

        manager.set_level(DegradationLevel::Severe);

        assert!(!manager.is_feature_disabled(DegradationLevel::Critical));
        assert!(manager.is_feature_disabled(DegradationLevel::Moderate));
        assert!(manager.is_feature_disabled(DegradationLevel::Normal));
    }

    #[test]
    fn test_kernel_watchdog() {
        let watchdog = KernelWatchdog::new();

        let kernel_id = KernelId::new("test_kernel");
        watchdog.watch(kernel_id.clone());

        assert_eq!(watchdog.watched_count(), 1);

        watchdog.heartbeat(&kernel_id);
        let health = watchdog.get_health(&kernel_id).unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_kernel_watchdog_metrics() {
        let watchdog = KernelWatchdog::new();

        let kernel_id = KernelId::new("test_kernel");
        watchdog.watch(kernel_id.clone());

        watchdog.update_metrics(&kernel_id, 1000.0, 50);

        let health = watchdog.get_health(&kernel_id).unwrap();
        assert_eq!(health.messages_per_sec, 1000.0);
        assert_eq!(health.queue_depth, 50);
    }
}
