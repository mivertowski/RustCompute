//! Enterprise feature integration for RingKernel ecosystem.
//!
//! This module provides integration between RingKernel's enterprise features
//! (health monitoring, circuit breakers, degradation, metrics) and the ecosystem
//! integrations (Axum, Tower, Actix).
//!
//! # Features
//!
//! - `EnterpriseState` - Shared state wrapping `RingKernelContext`
//! - `health_check_route` - Axum route for health status
//! - `metrics_route` - Prometheus metrics endpoint
//! - `CircuitBreakerMiddleware` - Tower middleware for circuit breaker protection
//! - `DegradationMiddleware` - Tower middleware for graceful degradation
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ecosystem::enterprise::{EnterpriseState, enterprise_routes};
//! use ringkernel_core::prelude::*;
//!
//! let runtime = RuntimeBuilder::new()
//!     .production()
//!     .build()?;
//!
//! runtime.start()?;
//!
//! let state = EnterpriseState::new(runtime);
//!
//! let app = Router::new()
//!     .merge(enterprise_routes(state.clone()))
//!     .with_state(state);
//! ```

use std::sync::Arc;

use ringkernel_core::runtime_context::{
    CircuitGuard, DegradationGuard, HealthCycleResult, LifecycleState, OperationPriority,
    RingKernelContext, RuntimeStatsSnapshot, WatchdogResult,
};

/// Shared enterprise state for web framework integrations.
#[derive(Clone)]
pub struct EnterpriseState {
    /// The underlying RingKernel context.
    context: Arc<RingKernelContext>,
}

impl EnterpriseState {
    /// Create new enterprise state from a RingKernelContext.
    pub fn new(context: Arc<RingKernelContext>) -> Self {
        Self { context }
    }

    /// Get the underlying context.
    pub fn context(&self) -> &Arc<RingKernelContext> {
        &self.context
    }

    /// Get current lifecycle state.
    pub fn lifecycle_state(&self) -> LifecycleState {
        self.context.lifecycle_state()
    }

    /// Check if the runtime is accepting work.
    pub fn is_accepting_work(&self) -> bool {
        self.context.is_accepting_work()
    }

    /// Run a health check cycle and return result.
    pub fn run_health_check(&self) -> HealthCycleResult {
        self.context.run_health_check_cycle()
    }

    /// Run a watchdog scan and return result.
    pub fn run_watchdog_scan(&self) -> WatchdogResult {
        self.context.run_watchdog_cycle()
    }

    /// Get current runtime statistics.
    pub fn stats(&self) -> RuntimeStatsSnapshot {
        self.context.stats()
    }

    /// Get Prometheus metrics string.
    pub fn prometheus_metrics(&self) -> String {
        self.context.flush_metrics()
    }

    /// Create a circuit guard for an operation.
    pub fn circuit_guard(&self, operation_name: &str) -> CircuitGuard<'_> {
        CircuitGuard::new(&self.context, operation_name)
    }

    /// Create a degradation guard for priority-based load shedding.
    pub fn degradation_guard(&self) -> DegradationGuard<'_> {
        DegradationGuard::new(&self.context)
    }

    /// Check if an operation should be allowed at the given priority.
    pub fn allow_operation(&self, priority: OperationPriority) -> bool {
        self.degradation_guard().allow_operation(priority)
    }
}

// ============================================================================
// Response Types
// ============================================================================

use serde::{Deserialize, Serialize};

/// Health check response for REST APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnterpriseHealthResponse {
    /// Overall health status.
    pub status: String,
    /// Lifecycle state.
    pub lifecycle: String,
    /// Circuit breaker state.
    pub circuit_state: String,
    /// Degradation level.
    pub degradation_level: String,
    /// Whether the runtime is accepting work.
    pub accepting_work: bool,
    /// Number of stale kernels detected.
    pub stale_kernels: usize,
}

/// Statistics response for REST APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnterpriseStatsResponse {
    /// Runtime uptime in seconds.
    pub uptime_seconds: f64,
    /// Total kernels launched.
    pub kernels_launched: u64,
    /// Total messages processed.
    pub messages_processed: u64,
    /// Total migrations completed.
    pub migrations_completed: u64,
    /// Total checkpoints created.
    pub checkpoints_created: u64,
    /// Total health checks run.
    pub health_checks_run: u64,
    /// Total circuit breaker trips.
    pub circuit_breaker_trips: u64,
}

/// Liveness probe response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessResponse {
    /// Whether the runtime is alive.
    pub alive: bool,
    /// Current lifecycle state.
    pub state: String,
}

/// Readiness probe response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessResponse {
    /// Whether the runtime is ready to accept traffic.
    pub ready: bool,
    /// Current lifecycle state.
    pub state: String,
    /// Reason if not ready.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl From<HealthCycleResult> for EnterpriseHealthResponse {
    fn from(result: HealthCycleResult) -> Self {
        Self {
            status: format!("{:?}", result.status),
            lifecycle: String::new(), // Will be filled by handler
            circuit_state: format!("{:?}", result.circuit_state),
            degradation_level: format!("{:?}", result.degradation_level),
            accepting_work: false, // Will be filled by handler
            stale_kernels: 0,      // Will be filled by handler
        }
    }
}

impl From<RuntimeStatsSnapshot> for EnterpriseStatsResponse {
    fn from(stats: RuntimeStatsSnapshot) -> Self {
        Self {
            uptime_seconds: stats.uptime.as_secs_f64(),
            kernels_launched: stats.kernels_launched,
            messages_processed: stats.messages_processed,
            migrations_completed: stats.migrations_completed,
            checkpoints_created: stats.checkpoints_created,
            health_checks_run: stats.health_checks_run,
            circuit_breaker_trips: stats.circuit_breaker_trips,
        }
    }
}

// ============================================================================
// Axum Integration
// ============================================================================

#[cfg(feature = "axum")]
/// Axum integration for enterprise features.
///
/// Provides ready-to-use routes for health checks, metrics, and liveness/readiness probes.
pub mod axum_integration {
    use super::*;
    use ::axum::{
        extract::State,
        http::StatusCode,
        response::IntoResponse,
        routing::get,
        Json, Router,
    };

    /// Create enterprise routes for Axum.
    ///
    /// Returns a router with these endpoints:
    /// - `GET /health` - Full health check
    /// - `GET /health/live` - Liveness probe (for Kubernetes)
    /// - `GET /health/ready` - Readiness probe (for Kubernetes)
    /// - `GET /stats` - Runtime statistics
    /// - `GET /metrics` - Prometheus metrics
    pub fn enterprise_routes(state: EnterpriseState) -> Router<EnterpriseState> {
        Router::new()
            .route("/health", get(health_handler))
            .route("/health/live", get(liveness_handler))
            .route("/health/ready", get(readiness_handler))
            .route("/stats", get(stats_handler))
            .route("/metrics", get(metrics_handler))
            .with_state(state)
    }

    /// Health check handler.
    pub async fn health_handler(
        State(state): State<EnterpriseState>,
    ) -> impl IntoResponse {
        let health_result = state.run_health_check();
        let watchdog_result = state.run_watchdog_scan();

        let response = EnterpriseHealthResponse {
            status: format!("{:?}", health_result.status),
            lifecycle: format!("{:?}", state.lifecycle_state()),
            circuit_state: format!("{:?}", health_result.circuit_state),
            degradation_level: format!("{:?}", health_result.degradation_level),
            accepting_work: state.is_accepting_work(),
            stale_kernels: watchdog_result.stale_kernels,
        };

        let status = if state.is_accepting_work() {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        };

        (status, Json(response))
    }

    /// Liveness probe handler.
    pub async fn liveness_handler(
        State(state): State<EnterpriseState>,
    ) -> impl IntoResponse {
        let lifecycle = state.lifecycle_state();
        let alive = lifecycle.is_active();

        let response = LivenessResponse {
            alive,
            state: format!("{:?}", lifecycle),
        };

        let status = if alive {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        };

        (status, Json(response))
    }

    /// Readiness probe handler.
    pub async fn readiness_handler(
        State(state): State<EnterpriseState>,
    ) -> impl IntoResponse {
        let lifecycle = state.lifecycle_state();
        let ready = state.is_accepting_work();

        let reason = if !ready {
            Some(format!("Lifecycle state: {:?}", lifecycle))
        } else {
            None
        };

        let response = ReadinessResponse {
            ready,
            state: format!("{:?}", lifecycle),
            reason,
        };

        let status = if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        };

        (status, Json(response))
    }

    /// Statistics handler.
    pub async fn stats_handler(
        State(state): State<EnterpriseState>,
    ) -> impl IntoResponse {
        let stats = state.stats();
        let response: EnterpriseStatsResponse = stats.into();
        Json(response)
    }

    /// Prometheus metrics handler.
    pub async fn metrics_handler(
        State(state): State<EnterpriseState>,
    ) -> impl IntoResponse {
        let metrics = state.prometheus_metrics();
        (
            StatusCode::OK,
            [("content-type", "text/plain; charset=utf-8")],
            metrics,
        )
    }
}

#[cfg(feature = "axum")]
pub use axum_integration::*;

// ============================================================================
// Tower Middleware
// ============================================================================

#[cfg(feature = "tower")]
/// Tower middleware for enterprise features.
///
/// Provides circuit breaker and degradation-aware middleware layers.
pub mod tower_integration {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use ::tower::{Layer, Service};

    /// Layer that adds circuit breaker protection to a service.
    #[derive(Clone)]
    pub struct CircuitBreakerLayer {
        state: EnterpriseState,
        operation_name: String,
    }

    impl CircuitBreakerLayer {
        /// Create a new circuit breaker layer.
        pub fn new(state: EnterpriseState, operation_name: impl Into<String>) -> Self {
            Self {
                state,
                operation_name: operation_name.into(),
            }
        }
    }

    impl<S> Layer<S> for CircuitBreakerLayer {
        type Service = CircuitBreakerService<S>;

        fn layer(&self, inner: S) -> Self::Service {
            CircuitBreakerService {
                inner,
                state: self.state.clone(),
                operation_name: self.operation_name.clone(),
            }
        }
    }

    /// Service wrapper that applies circuit breaker protection.
    #[derive(Clone)]
    pub struct CircuitBreakerService<S> {
        inner: S,
        state: EnterpriseState,
        operation_name: String,
    }

    impl<S, Request> Service<Request> for CircuitBreakerService<S>
    where
        S: Service<Request> + Clone,
        S::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        type Response = S::Response;
        type Error = Box<dyn std::error::Error + Send + Sync>;
        type Future = CircuitBreakerFuture<S::Future>;

        fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            // Check if circuit is open
            let cb = self.state.context().circuit_breaker();
            if cb.state() == ringkernel_core::health::CircuitState::Open {
                return Poll::Ready(Err(format!(
                    "Circuit breaker open for operation: {}",
                    self.operation_name
                ).into()));
            }

            self.inner.poll_ready(cx).map_err(Into::into)
        }

        fn call(&mut self, request: Request) -> Self::Future {
            let cb = self.state.context().circuit_breaker();
            CircuitBreakerFuture {
                inner: self.inner.call(request),
                circuit_breaker: cb,
            }
        }
    }

    pin_project_lite::pin_project! {
        /// Future that records success/failure to the circuit breaker.
        pub struct CircuitBreakerFuture<F> {
            #[pin]
            inner: F,
            circuit_breaker: std::sync::Arc<ringkernel_core::health::CircuitBreaker>,
        }
    }

    impl<F, T, E> Future for CircuitBreakerFuture<F>
    where
        F: Future<Output = Result<T, E>>,
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        type Output = Result<T, Box<dyn std::error::Error + Send + Sync>>;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            let this = self.project();
            match this.inner.poll(cx) {
                Poll::Ready(Ok(response)) => {
                    this.circuit_breaker.record_success();
                    Poll::Ready(Ok(response))
                }
                Poll::Ready(Err(e)) => {
                    this.circuit_breaker.record_failure();
                    let boxed_error: Box<dyn std::error::Error + Send + Sync> = e.into();
                    Poll::Ready(Err(boxed_error))
                }
                Poll::Pending => Poll::Pending,
            }
        }
    }

    /// Layer that applies degradation-based load shedding.
    #[derive(Clone)]
    pub struct DegradationLayer {
        state: EnterpriseState,
        priority: OperationPriority,
    }

    impl DegradationLayer {
        /// Create a new degradation layer with the given operation priority.
        pub fn new(state: EnterpriseState, priority: OperationPriority) -> Self {
            Self { state, priority }
        }

        /// Create a layer for low priority operations (shed first).
        pub fn low_priority(state: EnterpriseState) -> Self {
            Self::new(state, OperationPriority::Low)
        }

        /// Create a layer for normal priority operations.
        pub fn normal_priority(state: EnterpriseState) -> Self {
            Self::new(state, OperationPriority::Normal)
        }

        /// Create a layer for high priority operations (shed last).
        pub fn high_priority(state: EnterpriseState) -> Self {
            Self::new(state, OperationPriority::High)
        }

        /// Create a layer for critical operations (never shed).
        pub fn critical(state: EnterpriseState) -> Self {
            Self::new(state, OperationPriority::Critical)
        }
    }

    impl<S> Layer<S> for DegradationLayer {
        type Service = DegradationService<S>;

        fn layer(&self, inner: S) -> Self::Service {
            DegradationService {
                inner,
                state: self.state.clone(),
                priority: self.priority,
            }
        }
    }

    /// Service wrapper that applies degradation-based load shedding.
    #[derive(Clone)]
    pub struct DegradationService<S> {
        inner: S,
        state: EnterpriseState,
        priority: OperationPriority,
    }

    impl<S, Request> Service<Request> for DegradationService<S>
    where
        S: Service<Request>,
        S::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        type Response = S::Response;
        type Error = Box<dyn std::error::Error + Send + Sync>;
        type Future = DegradationFuture<S::Future>;

        fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            // Check if operation is allowed at current degradation level
            if !self.state.allow_operation(self.priority) {
                return Poll::Ready(Err(format!(
                    "Load shedding: operation priority {:?} rejected at current degradation level",
                    self.priority
                ).into()));
            }

            self.inner.poll_ready(cx).map_err(Into::into)
        }

        fn call(&mut self, request: Request) -> Self::Future {
            DegradationFuture {
                inner: self.inner.call(request),
            }
        }
    }

    pin_project_lite::pin_project! {
        /// Future wrapper for degradation service.
        pub struct DegradationFuture<F> {
            #[pin]
            inner: F,
        }
    }

    impl<F, T, E> Future for DegradationFuture<F>
    where
        F: Future<Output = Result<T, E>>,
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        type Output = Result<T, Box<dyn std::error::Error + Send + Sync>>;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            let this = self.project();
            match this.inner.poll(cx) {
                Poll::Ready(Ok(v)) => Poll::Ready(Ok(v)),
                Poll::Ready(Err(e)) => {
                    let boxed_error: Box<dyn std::error::Error + Send + Sync> = e.into();
                    Poll::Ready(Err(boxed_error))
                }
                Poll::Pending => Poll::Pending,
            }
        }
    }
}

#[cfg(feature = "tower")]
pub use tower_integration::*;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ringkernel_core::runtime_context::RuntimeBuilder;

    #[test]
    fn test_enterprise_state_creation() {
        let context = RuntimeBuilder::new().development().build().unwrap();
        let state = EnterpriseState::new(context);

        assert_eq!(state.lifecycle_state(), LifecycleState::Initializing);
    }

    #[test]
    fn test_enterprise_state_health_check() {
        let context = RuntimeBuilder::new().development().build().unwrap();
        context.start().unwrap();

        let state = EnterpriseState::new(context);
        let result = state.run_health_check();

        assert!(state.is_accepting_work());
        assert!(matches!(
            result.status,
            ringkernel_core::health::HealthStatus::Healthy
        ));
    }

    #[test]
    fn test_enterprise_state_stats() {
        let context = RuntimeBuilder::new().development().build().unwrap();
        context.record_kernel_launch();
        context.record_messages(100);

        let state = EnterpriseState::new(context);
        let stats = state.stats();

        assert_eq!(stats.kernels_launched, 1);
        assert_eq!(stats.messages_processed, 100);
    }

    #[test]
    fn test_health_response_serialization() {
        let response = EnterpriseHealthResponse {
            status: "Healthy".to_string(),
            lifecycle: "Running".to_string(),
            circuit_state: "Closed".to_string(),
            degradation_level: "Normal".to_string(),
            accepting_work: true,
            stale_kernels: 0,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"Healthy\""));
        assert!(json.contains("\"accepting_work\":true"));
    }

    #[test]
    fn test_stats_response_from_snapshot() {
        let stats = RuntimeStatsSnapshot {
            uptime: std::time::Duration::from_secs(60),
            kernels_launched: 10,
            messages_processed: 1000,
            migrations_completed: 2,
            checkpoints_created: 5,
            health_checks_run: 12,
            circuit_breaker_trips: 0,
        };

        let response: EnterpriseStatsResponse = stats.into();

        assert_eq!(response.uptime_seconds, 60.0);
        assert_eq!(response.kernels_launched, 10);
        assert_eq!(response.messages_processed, 1000);
    }

    #[test]
    fn test_allow_operation_priority() {
        let context = RuntimeBuilder::new().development().build().unwrap();
        context.start().unwrap();

        let state = EnterpriseState::new(context);

        // At normal degradation level, all priorities should be allowed
        assert!(state.allow_operation(OperationPriority::Low));
        assert!(state.allow_operation(OperationPriority::Normal));
        assert!(state.allow_operation(OperationPriority::High));
        assert!(state.allow_operation(OperationPriority::Critical));
    }
}
