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
use axum::Json;
use serde::{de::DeserializeOwned, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use ringkernel_core::message::RingMessage;

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
        self.total_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);
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

// ============================================================================
// PERSISTENT GPU STATE INTEGRATION
// ============================================================================

#[cfg(feature = "persistent")]
pub use persistent_state::*;

#[cfg(feature = "persistent")]
mod persistent_state {
    use super::*;
    use crate::persistent::{
        CommandId, PersistentCommand, PersistentConfig, PersistentHandle, PersistentResponse,
        PersistentStats,
    };
    use axum::routing::{get, post};
    use axum::Router;
    use serde::Deserialize;
    use tokio::sync::broadcast;

    /// Shared state wrapping a persistent GPU simulation.
    ///
    /// This state provides low-latency command injection (~0.03µs) compared
    /// to traditional launch-per-request patterns (~317µs).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use axum::Router;
    /// use ringkernel_ecosystem::axum::PersistentGpuState;
    ///
    /// let handle = create_persistent_handle(); // Your PersistentHandle impl
    /// let state = PersistentGpuState::new(handle, PersistentAxumConfig::default());
    ///
    /// let app = Router::new()
    ///     .merge(state.routes())
    ///     .with_state(state);
    /// ```
    #[derive(Clone)]
    pub struct PersistentGpuState<H: PersistentHandle + Clone> {
        /// Handle to the persistent kernel.
        handle: Arc<H>,
        /// Configuration.
        config: PersistentAxumConfig,
        /// Broadcast channel for streaming responses.
        response_tx: broadcast::Sender<PersistentResponse>,
        /// Request statistics.
        stats: Arc<RequestStats>,
    }

    /// Configuration for persistent Axum integration.
    #[derive(Debug, Clone)]
    pub struct PersistentAxumConfig {
        /// Base Axum config.
        pub base: AxumConfig,
        /// Persistent kernel config.
        pub persistent: PersistentConfig,
        /// Broadcast channel capacity.
        pub broadcast_capacity: usize,
        /// API prefix (e.g., "/api/v1").
        pub api_prefix: String,
    }

    impl Default for PersistentAxumConfig {
        fn default() -> Self {
            Self {
                base: AxumConfig::default(),
                persistent: PersistentConfig::default(),
                broadcast_capacity: 1024,
                api_prefix: "/api".to_string(),
            }
        }
    }

    impl PersistentAxumConfig {
        /// Create a new configuration with default values.
        pub fn new() -> Self {
            Self::default()
        }

        /// Set the API prefix.
        pub fn with_api_prefix(mut self, prefix: impl Into<String>) -> Self {
            self.api_prefix = prefix.into();
            self
        }

        /// Set the broadcast channel capacity.
        pub fn with_broadcast_capacity(mut self, capacity: usize) -> Self {
            self.broadcast_capacity = capacity;
            self
        }
    }

    impl<H: PersistentHandle + Clone + 'static> PersistentGpuState<H> {
        /// Create new persistent GPU state.
        pub fn new(handle: Arc<H>, config: PersistentAxumConfig) -> Self {
            let (response_tx, _) = broadcast::channel(config.broadcast_capacity);
            Self {
                handle,
                config,
                response_tx,
                stats: Arc::new(RequestStats::new()),
            }
        }

        /// Get the kernel ID.
        pub fn kernel_id(&self) -> &str {
            self.handle.kernel_id()
        }

        /// Check if the kernel is running.
        pub fn is_running(&self) -> bool {
            self.handle.is_running()
        }

        /// Get current statistics.
        pub fn kernel_stats(&self) -> PersistentStats {
            self.handle.stats()
        }

        /// Get request statistics.
        pub fn request_stats(&self) -> &RequestStats {
            &self.stats
        }

        /// Get a broadcast receiver for streaming responses.
        pub fn subscribe(&self) -> broadcast::Receiver<PersistentResponse> {
            self.response_tx.subscribe()
        }

        /// Broadcast a response to all subscribers.
        pub fn broadcast(&self, response: PersistentResponse) {
            let _ = self.response_tx.send(response);
        }

        /// Poll responses and broadcast them.
        pub fn poll_and_broadcast(&self) {
            for response in self.handle.poll_responses() {
                self.broadcast(response);
            }
        }

        /// Create routes for persistent GPU endpoints.
        ///
        /// Routes:
        /// - `POST /step` - Run simulation steps
        /// - `POST /impulse` - Inject impulse at position
        /// - `POST /pause` - Pause simulation
        /// - `POST /resume` - Resume simulation
        /// - `GET /stats` - Get current statistics
        /// - `GET /health` - Health check
        pub fn routes(self) -> Router {
            Router::new()
                .route(
                    &format!("{}/step", self.config.api_prefix),
                    post(step_handler::<H>),
                )
                .route(
                    &format!("{}/impulse", self.config.api_prefix),
                    post(impulse_handler::<H>),
                )
                .route(
                    &format!("{}/pause", self.config.api_prefix),
                    post(pause_handler::<H>),
                )
                .route(
                    &format!("{}/resume", self.config.api_prefix),
                    post(resume_handler::<H>),
                )
                .route(
                    &format!("{}/stats", self.config.api_prefix),
                    get(persistent_stats_handler::<H>),
                )
                .route(
                    &format!("{}/health", self.config.api_prefix),
                    get(persistent_health_handler::<H>),
                )
                .with_state(self)
        }
    }

    // ========================================================================
    // REQUEST/RESPONSE TYPES
    // ========================================================================

    /// Request to run simulation steps.
    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct StepRequest {
        /// Number of steps to run.
        pub count: u64,
    }

    /// Request to inject an impulse.
    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct ImpulseRequest {
        /// X position.
        pub x: u32,
        /// Y position.
        pub y: u32,
        /// Z position (default 0 for 2D).
        #[serde(default)]
        pub z: u32,
        /// Impulse value.
        pub value: f32,
    }

    /// Command response.
    #[derive(Debug, Clone, Serialize)]
    pub struct CommandResponse {
        /// Whether the command was accepted.
        pub success: bool,
        /// Command ID for tracking.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub command_id: Option<u64>,
        /// Error message if failed.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
    }

    impl CommandResponse {
        fn success(cmd_id: CommandId) -> Self {
            Self {
                success: true,
                command_id: Some(cmd_id.0),
                error: None,
            }
        }

        fn error(msg: impl Into<String>) -> Self {
            Self {
                success: false,
                command_id: None,
                error: Some(msg.into()),
            }
        }
    }

    /// Statistics response.
    #[derive(Debug, Clone, Serialize)]
    pub struct PersistentStatsResponse {
        /// Kernel ID.
        pub kernel_id: String,
        /// Whether kernel is running.
        pub running: bool,
        /// Current simulation step.
        pub current_step: u64,
        /// Total commands processed.
        pub commands_processed: u64,
        /// Commands pending.
        pub commands_pending: u64,
        /// Request statistics.
        pub request_stats: StatsSnapshot,
    }

    /// Health response for persistent kernel.
    #[derive(Debug, Clone, Serialize)]
    pub struct PersistentHealthResponse {
        /// Overall health status.
        pub healthy: bool,
        /// Kernel ID.
        pub kernel_id: String,
        /// Whether kernel is running.
        pub running: bool,
        /// Current step.
        pub current_step: u64,
    }

    // ========================================================================
    // HANDLER IMPLEMENTATIONS
    // ========================================================================

    /// Handler for running simulation steps.
    pub async fn step_handler<H: PersistentHandle + Clone + 'static>(
        State(state): State<PersistentGpuState<H>>,
        Json(request): Json<StepRequest>,
    ) -> Json<CommandResponse> {
        let start = std::time::Instant::now();

        if !state.is_running() {
            state.stats.record_failure();
            return Json(CommandResponse::error("Kernel not running"));
        }

        let cmd = PersistentCommand::RunSteps {
            count: request.count,
        };

        match state.handle.send_command(cmd) {
            Ok(cmd_id) => {
                state
                    .stats
                    .record_success(start.elapsed().as_micros() as u64);
                Json(CommandResponse::success(cmd_id))
            }
            Err(e) => {
                state.stats.record_failure();
                Json(CommandResponse::error(e.to_string()))
            }
        }
    }

    /// Handler for injecting impulses.
    pub async fn impulse_handler<H: PersistentHandle + Clone + 'static>(
        State(state): State<PersistentGpuState<H>>,
        Json(request): Json<ImpulseRequest>,
    ) -> Json<CommandResponse> {
        let start = std::time::Instant::now();

        if !state.is_running() {
            state.stats.record_failure();
            return Json(CommandResponse::error("Kernel not running"));
        }

        let cmd = PersistentCommand::Inject {
            position: (request.x, request.y, request.z),
            value: request.value,
        };

        match state.handle.send_command(cmd) {
            Ok(cmd_id) => {
                state
                    .stats
                    .record_success(start.elapsed().as_micros() as u64);
                Json(CommandResponse::success(cmd_id))
            }
            Err(e) => {
                state.stats.record_failure();
                Json(CommandResponse::error(e.to_string()))
            }
        }
    }

    /// Handler for pausing the simulation.
    pub async fn pause_handler<H: PersistentHandle + Clone + 'static>(
        State(state): State<PersistentGpuState<H>>,
    ) -> Json<CommandResponse> {
        let start = std::time::Instant::now();

        if !state.is_running() {
            state.stats.record_failure();
            return Json(CommandResponse::error("Kernel not running"));
        }

        match state.handle.send_command(PersistentCommand::Pause) {
            Ok(cmd_id) => {
                state
                    .stats
                    .record_success(start.elapsed().as_micros() as u64);
                Json(CommandResponse::success(cmd_id))
            }
            Err(e) => {
                state.stats.record_failure();
                Json(CommandResponse::error(e.to_string()))
            }
        }
    }

    /// Handler for resuming the simulation.
    pub async fn resume_handler<H: PersistentHandle + Clone + 'static>(
        State(state): State<PersistentGpuState<H>>,
    ) -> Json<CommandResponse> {
        let start = std::time::Instant::now();

        if !state.is_running() {
            state.stats.record_failure();
            return Json(CommandResponse::error("Kernel not running"));
        }

        match state.handle.send_command(PersistentCommand::Resume) {
            Ok(cmd_id) => {
                state
                    .stats
                    .record_success(start.elapsed().as_micros() as u64);
                Json(CommandResponse::success(cmd_id))
            }
            Err(e) => {
                state.stats.record_failure();
                Json(CommandResponse::error(e.to_string()))
            }
        }
    }

    /// Handler for getting statistics.
    pub async fn persistent_stats_handler<H: PersistentHandle + Clone + 'static>(
        State(state): State<PersistentGpuState<H>>,
    ) -> Json<PersistentStatsResponse> {
        let kernel_stats = state.kernel_stats();
        let request_stats = state.request_stats();

        Json(PersistentStatsResponse {
            kernel_id: state.kernel_id().to_string(),
            running: state.is_running(),
            current_step: kernel_stats.current_step,
            commands_processed: kernel_stats.messages_processed,
            commands_pending: kernel_stats.pending_commands as u64,
            request_stats: StatsSnapshot {
                total_requests: request_stats.total_requests.load(Ordering::Relaxed),
                successful_requests: request_stats.successful_requests.load(Ordering::Relaxed),
                failed_requests: request_stats.failed_requests.load(Ordering::Relaxed),
                avg_latency_us: request_stats.avg_latency_us(),
                success_rate: request_stats.success_rate(),
            },
        })
    }

    /// Handler for health checks.
    pub async fn persistent_health_handler<H: PersistentHandle + Clone + 'static>(
        State(state): State<PersistentGpuState<H>>,
    ) -> Json<PersistentHealthResponse> {
        let stats = state.kernel_stats();

        Json(PersistentHealthResponse {
            healthy: state.is_running(),
            kernel_id: state.kernel_id().to_string(),
            running: state.is_running(),
            current_step: stats.current_step,
        })
    }
}

// ============================================================================
// SSE (SERVER-SENT EVENTS) INTEGRATION
// ============================================================================

#[cfg(feature = "axum-sse")]
pub use sse::*;

#[cfg(feature = "axum-sse")]
mod sse {
    use super::*;
    use crate::persistent::{PersistentHandle, PersistentResponse};
    use axum::response::sse::{Event, Sse};
    use axum::Router;
    use futures::stream::Stream;
    use std::convert::Infallible;
    use tokio::sync::broadcast;

    // Re-use PersistentGpuState from parent module
    #[cfg(feature = "persistent")]
    use super::persistent_state::PersistentGpuState;

    /// SSE event stream for persistent kernel responses.
    ///
    /// Clients can connect to receive real-time updates from the GPU kernel.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use axum::Router;
    /// use ringkernel_ecosystem::axum::{PersistentGpuState, sse_handler};
    ///
    /// let app = Router::new()
    ///     .route("/events", get(sse_handler::<MyPersistentHandle>))
    ///     .with_state(state);
    /// ```
    pub async fn sse_handler<H: PersistentHandle + Clone + 'static>(
        State(state): State<PersistentGpuState<H>>,
    ) -> Sse<impl Stream<Item = std::result::Result<Event, Infallible>>> {
        let rx = state.subscribe();

        let stream = async_stream::stream! {
            let mut rx = rx;
            loop {
                match rx.recv().await {
                    Ok(response) => {
                        let event_type = match &response {
                            PersistentResponse::Ack { .. } => "ack",
                            PersistentResponse::Progress { .. } => "progress",
                            PersistentResponse::Stats { .. } => "stats",
                            PersistentResponse::Error { .. } => "error",
                            PersistentResponse::Terminated { .. } => "terminated",
                            PersistentResponse::Custom { .. } => "custom",
                        };

                        if let Ok(data) = serde_json::to_string(&SseResponse::from(response)) {
                            yield Ok(Event::default()
                                .event(event_type)
                                .data(data));
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // Skip lagged messages
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
        };

        Sse::new(stream).keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
    }

    /// SSE response wrapper for JSON serialization.
    #[derive(Debug, Clone, Serialize)]
    pub struct SseResponse {
        /// Event type.
        pub event_type: String,
        /// Command ID if applicable.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub command_id: Option<u64>,
        /// Current step for progress events.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub current_step: Option<u64>,
        /// Remaining steps for progress events.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub remaining: Option<u64>,
        /// Error code if applicable.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error_code: Option<u32>,
        /// Error message if applicable.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error_message: Option<String>,
    }

    impl From<PersistentResponse> for SseResponse {
        fn from(response: PersistentResponse) -> Self {
            match response {
                PersistentResponse::Ack { cmd_id } => Self {
                    event_type: "ack".to_string(),
                    command_id: Some(cmd_id.0),
                    current_step: None,
                    remaining: None,
                    error_code: None,
                    error_message: None,
                },
                PersistentResponse::Progress {
                    cmd_id,
                    current_step,
                    remaining,
                } => Self {
                    event_type: "progress".to_string(),
                    command_id: Some(cmd_id.0),
                    current_step: Some(current_step),
                    remaining: Some(remaining),
                    error_code: None,
                    error_message: None,
                },
                PersistentResponse::Stats { cmd_id, stats: _ } => Self {
                    event_type: "stats".to_string(),
                    command_id: Some(cmd_id.0),
                    current_step: None,
                    remaining: None,
                    error_code: None,
                    error_message: None,
                },
                PersistentResponse::Error {
                    cmd_id,
                    code,
                    message,
                } => Self {
                    event_type: "error".to_string(),
                    command_id: Some(cmd_id.0),
                    current_step: None,
                    remaining: None,
                    error_code: Some(code),
                    error_message: Some(message),
                },
                PersistentResponse::Terminated { final_step } => Self {
                    event_type: "terminated".to_string(),
                    command_id: None,
                    current_step: Some(final_step),
                    remaining: None,
                    error_code: None,
                    error_message: None,
                },
                PersistentResponse::Custom {
                    cmd_id,
                    type_id: _,
                    payload: _,
                } => Self {
                    event_type: "custom".to_string(),
                    command_id: Some(cmd_id.0),
                    current_step: None,
                    remaining: None,
                    error_code: None,
                    error_message: None,
                },
            }
        }
    }

    /// Add SSE routes to a persistent GPU state.
    pub fn with_sse_routes<H: PersistentHandle + Clone + 'static>(
        router: Router<PersistentGpuState<H>>,
        prefix: &str,
    ) -> Router<PersistentGpuState<H>> {
        use axum::routing::get;
        router.route(&format!("{}/events", prefix), get(sse_handler::<H>))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
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
