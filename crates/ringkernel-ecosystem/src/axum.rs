//! Axum web framework integration for RingKernel.
//!
//! This module provides utilities for building web APIs that delegate
//! compute-intensive operations to GPU Ring Kernels.
//!
//! # Example
//!
//! ```ignore
//! use axum::{Router, routing::post};
//! use ringkernel_ecosystem::axum::{AppState, gpu_handler};
//!
//! let app = Router::new()
//!     .route("/api/compute", post(gpu_handler::<VectorAddRequest, VectorAddResponse>))
//!     .with_state(state);
//! ```

use axum::extract::{FromRef, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use parking_lot::RwLock;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use ringkernel_core::message::RingMessage;
use ringkernel_core::runtime::KernelId;

use crate::error::{AppError, EcosystemError, Result};

/// Application state for Axum handlers.
#[derive(Clone)]
pub struct AppState<R> {
    /// Runtime handle.
    runtime: Arc<R>,
    /// Configuration.
    config: AxumConfig,
    /// Request statistics.
    stats: Arc<RequestStats>,
}

/// Configuration for Axum integration.
#[derive(Debug, Clone)]
pub struct AxumConfig {
    /// Default timeout for GPU operations.
    pub default_timeout: Duration,
    /// Maximum request body size.
    pub max_body_size: usize,
    /// Enable request logging.
    pub enable_logging: bool,
    /// Enable CORS.
    pub enable_cors: bool,
}

impl Default for AxumConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            max_body_size: 10 * 1024 * 1024, // 10MB
            enable_logging: true,
            enable_cors: true,
        }
    }
}

/// Request statistics.
#[derive(Debug, Default)]
pub struct RequestStats {
    /// Total requests received.
    pub total_requests: AtomicU64,
    /// Successful requests.
    pub successful_requests: AtomicU64,
    /// Failed requests.
    pub failed_requests: AtomicU64,
    /// Total latency in microseconds.
    pub total_latency_us: AtomicU64,
}

impl RequestStats {
    /// Create new request stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful request.
    pub fn record_success(&self, latency_us: u64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us.fetch_add(latency_us, Ordering::Relaxed);
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Get average latency in microseconds.
    pub fn avg_latency_us(&self) -> f64 {
        let total = self.total_latency_us.load(Ordering::Relaxed);
        let count = self.successful_requests.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            total as f64 / count as f64
        }
    }

    /// Get success rate (0.0 to 1.0).
    pub fn success_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        let success = self.successful_requests.load(Ordering::Relaxed);
        if total == 0 {
            1.0
        } else {
            success as f64 / total as f64
        }
    }
}

/// Runtime handle trait for Axum integration.
#[async_trait::async_trait]
pub trait RuntimeHandle: Send + Sync + 'static {
    /// Send a message to a kernel.
    async fn send_message(&self, kernel_id: &str, data: Vec<u8>) -> Result<()>;

    /// Receive a message from a kernel with timeout.
    async fn receive_message(&self, kernel_id: &str, timeout: Duration) -> Result<Vec<u8>>;

    /// List available kernels.
    fn available_kernels(&self) -> Vec<String>;

    /// Check if a kernel is healthy.
    fn is_kernel_healthy(&self, kernel_id: &str) -> bool;
}

impl<R: RuntimeHandle> AppState<R> {
    /// Create new application state.
    pub fn new(runtime: Arc<R>) -> Self {
        Self::with_config(runtime, AxumConfig::default())
    }

    /// Create new application state with custom configuration.
    pub fn with_config(runtime: Arc<R>, config: AxumConfig) -> Self {
        Self {
            runtime,
            config,
            stats: Arc::new(RequestStats::new()),
        }
    }

    /// Get the runtime handle.
    pub fn runtime(&self) -> &Arc<R> {
        &self.runtime
    }

    /// Get the configuration.
    pub fn config(&self) -> &AxumConfig {
        &self.config
    }

    /// Get request statistics.
    pub fn stats(&self) -> &RequestStats {
        &self.stats
    }
}

impl<R: RuntimeHandle> FromRef<AppState<R>> for Arc<R> {
    fn from_ref(state: &AppState<R>) -> Self {
        state.runtime.clone()
    }
}

/// API request wrapper.
#[derive(Debug, Clone, Serialize)]
pub struct ApiRequest<T> {
    /// The request payload.
    pub data: T,
    /// Target kernel ID.
    pub kernel_id: String,
    /// Optional request ID for tracing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// API response wrapper.
#[derive(Debug, Clone, Serialize)]
pub struct ApiResponse<T> {
    /// The response payload.
    pub data: T,
    /// Processing latency in microseconds.
    pub latency_us: u64,
    /// Request ID if provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

/// Generic GPU handler for processing requests through a kernel.
///
/// This handler can be used with any request/response types that implement
/// the required traits.
pub async fn gpu_handler<R, Req, Resp>(
    State(state): State<AppState<R>>,
    Json(request): Json<ApiRequest<Req>>,
) -> std::result::Result<Json<ApiResponse<Resp>>, AppError>
where
    R: RuntimeHandle,
    Req: RingMessage + DeserializeOwned + Send,
    Resp: RingMessage + Serialize + Send,
{
    let start = std::time::Instant::now();

    // Serialize the request
    let data = request.data.serialize();

    // Send to kernel
    if let Err(e) = state.runtime.send_message(&request.kernel_id, data).await {
        state.stats.record_failure();
        return Err(AppError(e));
    }

    // Wait for response
    let response_data = match state
        .runtime
        .receive_message(&request.kernel_id, state.config.default_timeout)
        .await
    {
        Ok(data) => data,
        Err(e) => {
            state.stats.record_failure();
            return Err(AppError(e));
        }
    };

    // Deserialize response
    let response = Resp::deserialize(&response_data).map_err(|e| {
        state.stats.record_failure();
        AppError(EcosystemError::Deserialization(format!(
            "Failed to deserialize response: {}",
            e
        )))
    })?;

    let latency_us = start.elapsed().as_micros() as u64;
    state.stats.record_success(latency_us);

    Ok(Json(ApiResponse {
        data: response,
        latency_us,
        request_id: request.request_id,
    }))
}

/// Health check response.
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    /// Overall health status.
    pub healthy: bool,
    /// Available kernels.
    pub kernels: Vec<KernelHealth>,
    /// Request statistics.
    pub stats: StatsSnapshot,
}

/// Individual kernel health.
#[derive(Debug, Clone, Serialize)]
pub struct KernelHealth {
    /// Kernel ID.
    pub id: String,
    /// Whether kernel is healthy.
    pub healthy: bool,
}

/// Snapshot of request statistics.
#[derive(Debug, Clone, Serialize)]
pub struct StatsSnapshot {
    /// Total requests.
    pub total_requests: u64,
    /// Successful requests.
    pub successful_requests: u64,
    /// Failed requests.
    pub failed_requests: u64,
    /// Average latency in microseconds.
    pub avg_latency_us: f64,
    /// Success rate.
    pub success_rate: f64,
}

/// Health check handler.
pub async fn health_handler<R: RuntimeHandle>(
    State(state): State<AppState<R>>,
) -> Json<HealthResponse> {
    let kernels: Vec<KernelHealth> = state
        .runtime
        .available_kernels()
        .into_iter()
        .map(|id| KernelHealth {
            healthy: state.runtime.is_kernel_healthy(&id),
            id,
        })
        .collect();

    let all_healthy = kernels.iter().all(|k| k.healthy);

    let stats = StatsSnapshot {
        total_requests: state.stats.total_requests.load(Ordering::Relaxed),
        successful_requests: state.stats.successful_requests.load(Ordering::Relaxed),
        failed_requests: state.stats.failed_requests.load(Ordering::Relaxed),
        avg_latency_us: state.stats.avg_latency_us(),
        success_rate: state.stats.success_rate(),
    };

    Json(HealthResponse {
        healthy: all_healthy,
        kernels,
        stats,
    })
}

/// Router builder for RingKernel endpoints.
pub struct RouterBuilder<R> {
    state: AppState<R>,
    routes: Vec<(String, String, String)>, // (path, method, kernel_id)
}

impl<R: RuntimeHandle + Clone> RouterBuilder<R> {
    /// Create a new router builder.
    pub fn new(runtime: Arc<R>) -> Self {
        Self {
            state: AppState::new(runtime),
            routes: Vec::new(),
        }
    }

    /// Create a new router builder with custom configuration.
    pub fn with_config(runtime: Arc<R>, config: AxumConfig) -> Self {
        Self {
            state: AppState::with_config(runtime, config),
            routes: Vec::new(),
        }
    }

    /// Add a route for a kernel.
    pub fn route(mut self, path: &str, kernel_id: &str) -> Self {
        self.routes
            .push((path.to_string(), "POST".to_string(), kernel_id.to_string()));
        self
    }

    /// Get the application state.
    pub fn state(&self) -> AppState<R> {
        self.state.clone()
    }
}

/// Middleware for logging GPU requests.
pub mod middleware {
    use axum::extract::Request;
    use axum::middleware::Next;
    use axum::response::Response;

    /// Logging middleware for GPU requests.
    pub async fn logging_middleware(request: Request, next: Next) -> Response {
        let method = request.method().clone();
        let uri = request.uri().clone();
        let start = std::time::Instant::now();

        tracing::info!(
            method = %method,
            uri = %uri,
            "GPU request received"
        );

        let response = next.run(request).await;
        let latency = start.elapsed();

        tracing::info!(
            method = %method,
            uri = %uri,
            status = %response.status(),
            latency_ms = %latency.as_millis(),
            "GPU request completed"
        );

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRuntime;

    #[async_trait::async_trait]
    impl RuntimeHandle for MockRuntime {
        async fn send_message(&self, _kernel_id: &str, _data: Vec<u8>) -> Result<()> {
            Ok(())
        }

        async fn receive_message(&self, _kernel_id: &str, _timeout: Duration) -> Result<Vec<u8>> {
            Ok(vec![1, 2, 3, 4])
        }

        fn available_kernels(&self) -> Vec<String> {
            vec!["kernel1".to_string(), "kernel2".to_string()]
        }

        fn is_kernel_healthy(&self, _kernel_id: &str) -> bool {
            true
        }
    }

    #[test]
    fn test_axum_config_default() {
        let config = AxumConfig::default();
        assert_eq!(config.default_timeout, Duration::from_secs(30));
        assert!(config.enable_logging);
        assert!(config.enable_cors);
    }

    #[test]
    fn test_request_stats() {
        let stats = RequestStats::new();
        stats.record_success(1000);
        stats.record_success(2000);
        stats.record_failure();

        assert_eq!(stats.total_requests.load(Ordering::Relaxed), 3);
        assert_eq!(stats.successful_requests.load(Ordering::Relaxed), 2);
        assert_eq!(stats.failed_requests.load(Ordering::Relaxed), 1);
        assert_eq!(stats.avg_latency_us(), 1500.0);
    }

    #[test]
    fn test_app_state_creation() {
        let runtime = Arc::new(MockRuntime);
        let state = AppState::new(runtime);
        assert_eq!(state.config().default_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_router_builder() {
        let runtime = Arc::new(MockRuntime);
        let builder = RouterBuilder::new(runtime)
            .route("/api/add", "vector_add")
            .route("/api/multiply", "vector_multiply");

        assert_eq!(builder.routes.len(), 2);
    }
}
