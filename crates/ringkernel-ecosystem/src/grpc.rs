//! gRPC integration for RingKernel using Tonic.
//!
//! This module provides a gRPC server that exposes Ring Kernel functionality
//! over the network, enabling distributed GPU computing.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ecosystem::grpc::{RingKernelGrpcServer, GrpcConfig};
//!
//! let server = RingKernelGrpcServer::new(runtime, GrpcConfig::default());
//! server.serve("[::1]:50051").await?;
//! ```

use prost::Message;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tonic::Status;

use crate::error::Result;

/// Configuration for gRPC server.
#[derive(Debug, Clone)]
pub struct GrpcConfig {
    /// Default timeout for operations.
    pub timeout: Duration,
    /// Maximum message size in bytes.
    pub max_message_size: usize,
    /// Enable reflection for debugging.
    pub enable_reflection: bool,
    /// Enable health checking.
    pub enable_health_check: bool,
    /// Maximum concurrent streams.
    pub max_concurrent_streams: u32,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_message_size: 16 * 1024 * 1024, // 16MB
            enable_reflection: true,
            enable_health_check: true,
            max_concurrent_streams: 200,
        }
    }
}

/// Runtime handle trait for gRPC integration.
#[async_trait::async_trait]
pub trait RuntimeHandle: Send + Sync + 'static {
    /// Process a request through a kernel.
    async fn process(&self, kernel_id: &str, data: Vec<u8>, timeout: Duration) -> Result<Vec<u8>>;

    /// List available kernels.
    fn available_kernels(&self) -> Vec<String>;

    /// Check if a kernel is healthy.
    fn is_kernel_healthy(&self, kernel_id: &str) -> bool;

    /// Get kernel metrics.
    fn get_kernel_metrics(&self, kernel_id: &str) -> Option<KernelMetrics>;
}

/// Kernel metrics exposed via gRPC.
#[derive(Debug, Clone, Default)]
pub struct KernelMetrics {
    /// Messages processed.
    pub messages_processed: u64,
    /// Messages dropped.
    pub messages_dropped: u64,
    /// Average latency in microseconds.
    pub avg_latency_us: u64,
    /// Current queue depth.
    pub queue_depth: u32,
}

/// gRPC server statistics.
#[derive(Debug, Default)]
pub struct ServerStats {
    /// Total requests received.
    pub total_requests: AtomicU64,
    /// Successful requests.
    pub successful_requests: AtomicU64,
    /// Failed requests.
    pub failed_requests: AtomicU64,
    /// Total bytes received.
    pub bytes_received: AtomicU64,
    /// Total bytes sent.
    pub bytes_sent: AtomicU64,
}

impl ServerStats {
    /// Create new server stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful request.
    pub fn record_success(&self, bytes_in: usize, bytes_out: usize) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        self.bytes_received
            .fetch_add(bytes_in as u64, Ordering::Relaxed);
        self.bytes_sent
            .fetch_add(bytes_out as u64, Ordering::Relaxed);
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }
}

/// Process request message.
#[derive(Clone, PartialEq, Message)]
pub struct ProcessRequest {
    /// Target kernel ID.
    #[prost(string, tag = "1")]
    pub kernel_id: String,
    /// Request payload.
    #[prost(bytes = "vec", tag = "2")]
    pub payload: Vec<u8>,
    /// Optional timeout in milliseconds.
    #[prost(uint64, optional, tag = "3")]
    pub timeout_ms: Option<u64>,
    /// Optional request ID for tracing.
    #[prost(string, optional, tag = "4")]
    pub request_id: Option<String>,
}

/// Process response message.
#[derive(Clone, PartialEq, Message)]
pub struct ProcessResponse {
    /// Response payload.
    #[prost(bytes = "vec", tag = "1")]
    pub payload: Vec<u8>,
    /// Processing latency in microseconds.
    #[prost(uint64, tag = "2")]
    pub latency_us: u64,
    /// Request ID if provided.
    #[prost(string, optional, tag = "3")]
    pub request_id: Option<String>,
}

/// Kernel info request.
#[derive(Clone, PartialEq, Message)]
pub struct KernelInfoRequest {
    /// Kernel ID to query (empty for all).
    #[prost(string, tag = "1")]
    pub kernel_id: String,
}

/// Kernel info response.
#[derive(Clone, PartialEq, Message)]
pub struct KernelInfoResponse {
    /// Available kernels.
    #[prost(message, repeated, tag = "1")]
    pub kernels: Vec<KernelInfo>,
}

/// Information about a kernel.
#[derive(Clone, PartialEq, Message)]
pub struct KernelInfo {
    /// Kernel ID.
    #[prost(string, tag = "1")]
    pub id: String,
    /// Whether kernel is healthy.
    #[prost(bool, tag = "2")]
    pub healthy: bool,
    /// Messages processed.
    #[prost(uint64, tag = "3")]
    pub messages_processed: u64,
    /// Average latency in microseconds.
    #[prost(uint64, tag = "4")]
    pub avg_latency_us: u64,
}

/// gRPC server for Ring Kernel.
pub struct RingKernelGrpcServer<R> {
    runtime: Arc<R>,
    config: GrpcConfig,
    stats: Arc<ServerStats>,
}

impl<R: RuntimeHandle> RingKernelGrpcServer<R> {
    /// Create a new gRPC server.
    pub fn new(runtime: Arc<R>, config: GrpcConfig) -> Self {
        Self {
            runtime,
            config,
            stats: Arc::new(ServerStats::new()),
        }
    }

    /// Get server statistics.
    pub fn stats(&self) -> &ServerStats {
        &self.stats
    }

    /// Process a request.
    pub async fn process_request(
        &self,
        request: ProcessRequest,
    ) -> std::result::Result<ProcessResponse, Status> {
        let start = std::time::Instant::now();

        let timeout = request
            .timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(self.config.timeout);

        let bytes_in = request.payload.len();

        match self
            .runtime
            .process(&request.kernel_id, request.payload, timeout)
            .await
        {
            Ok(result) => {
                let latency_us = start.elapsed().as_micros() as u64;
                let bytes_out = result.len();
                self.stats.record_success(bytes_in, bytes_out);

                Ok(ProcessResponse {
                    payload: result,
                    latency_us,
                    request_id: request.request_id,
                })
            }
            Err(e) => {
                self.stats.record_failure();
                Err(e.into())
            }
        }
    }

    /// Get kernel information.
    pub fn get_kernel_info(&self, kernel_id: &str) -> KernelInfoResponse {
        let kernels = if kernel_id.is_empty() {
            self.runtime
                .available_kernels()
                .into_iter()
                .map(|id| {
                    let metrics = self.runtime.get_kernel_metrics(&id);
                    KernelInfo {
                        healthy: self.runtime.is_kernel_healthy(&id),
                        id,
                        messages_processed: metrics
                            .as_ref()
                            .map(|m| m.messages_processed)
                            .unwrap_or(0),
                        avg_latency_us: metrics.as_ref().map(|m| m.avg_latency_us).unwrap_or(0),
                    }
                })
                .collect()
        } else {
            let metrics = self.runtime.get_kernel_metrics(kernel_id);
            vec![KernelInfo {
                id: kernel_id.to_string(),
                healthy: self.runtime.is_kernel_healthy(kernel_id),
                messages_processed: metrics.as_ref().map(|m| m.messages_processed).unwrap_or(0),
                avg_latency_us: metrics.as_ref().map(|m| m.avg_latency_us).unwrap_or(0),
            }]
        };

        KernelInfoResponse { kernels }
    }
}

/// Builder for gRPC server.
pub struct GrpcServerBuilder<R> {
    runtime: Arc<R>,
    config: GrpcConfig,
}

impl<R: RuntimeHandle> GrpcServerBuilder<R> {
    /// Create a new builder.
    pub fn new(runtime: Arc<R>) -> Self {
        Self {
            runtime,
            config: GrpcConfig::default(),
        }
    }

    /// Set the configuration.
    pub fn config(mut self, config: GrpcConfig) -> Self {
        self.config = config;
        self
    }

    /// Set timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set maximum message size.
    pub fn max_message_size(mut self, size: usize) -> Self {
        self.config.max_message_size = size;
        self
    }

    /// Enable reflection.
    pub fn enable_reflection(mut self, enable: bool) -> Self {
        self.config.enable_reflection = enable;
        self
    }

    /// Build the server.
    pub fn build(self) -> RingKernelGrpcServer<R> {
        RingKernelGrpcServer::new(self.runtime, self.config)
    }
}

/// Client for connecting to remote Ring Kernel gRPC servers.
pub struct RingKernelGrpcClient {
    endpoint: String,
    config: GrpcConfig,
}

impl RingKernelGrpcClient {
    /// Create a new client.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            config: GrpcConfig::default(),
        }
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: GrpcConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the endpoint.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

// ============================================================================
// PERSISTENT KERNEL GRPC INTEGRATION
// ============================================================================

#[cfg(feature = "persistent")]
pub use persistent_grpc::*;

#[cfg(feature = "persistent")]
mod persistent_grpc {
    use super::*;
    use crate::persistent::{
        PersistentCommand, PersistentConfig, PersistentHandle, PersistentResponse, PersistentStats,
    };
    use futures::stream::Stream;
    use std::pin::Pin;
    use tokio::sync::broadcast;

    /// Configuration for persistent gRPC service.
    #[derive(Debug, Clone)]
    pub struct PersistentGrpcConfig {
        /// Base gRPC config.
        pub base: GrpcConfig,
        /// Persistent kernel config.
        pub persistent: PersistentConfig,
        /// Broadcast channel capacity for streaming.
        pub broadcast_capacity: usize,
    }

    impl Default for PersistentGrpcConfig {
        fn default() -> Self {
            Self {
                base: GrpcConfig::default(),
                persistent: PersistentConfig::default(),
                broadcast_capacity: 1024,
            }
        }
    }

    // ========================================================================
    // PROTOBUF MESSAGES
    // ========================================================================

    /// Step request for running simulation steps.
    #[derive(Clone, PartialEq, Message)]
    pub struct PersistentStepRequest {
        /// Number of steps to run.
        #[prost(uint64, tag = "1")]
        pub count: u64,
        /// Optional request ID for tracing.
        #[prost(string, optional, tag = "2")]
        pub request_id: Option<String>,
    }

    /// Impulse request for injecting values.
    #[derive(Clone, PartialEq, Message)]
    pub struct PersistentImpulseRequest {
        /// X position.
        #[prost(uint32, tag = "1")]
        pub x: u32,
        /// Y position.
        #[prost(uint32, tag = "2")]
        pub y: u32,
        /// Z position.
        #[prost(uint32, tag = "3")]
        pub z: u32,
        /// Value to inject.
        #[prost(float, tag = "4")]
        pub value: f32,
        /// Optional request ID for tracing.
        #[prost(string, optional, tag = "5")]
        pub request_id: Option<String>,
    }

    /// Command response.
    #[derive(Clone, PartialEq, Message)]
    pub struct PersistentCommandResponse {
        /// Whether the command was accepted.
        #[prost(bool, tag = "1")]
        pub success: bool,
        /// Command ID for tracking.
        #[prost(uint64, tag = "2")]
        pub command_id: u64,
        /// Latency in nanoseconds.
        #[prost(uint64, tag = "3")]
        pub latency_ns: u64,
        /// Error message if failed.
        #[prost(string, optional, tag = "4")]
        pub error: Option<String>,
        /// Request ID if provided.
        #[prost(string, optional, tag = "5")]
        pub request_id: Option<String>,
    }

    /// Stats request.
    #[derive(Clone, PartialEq, Message)]
    pub struct PersistentStatsRequest {}

    /// Stats response.
    #[derive(Clone, PartialEq, Message)]
    pub struct PersistentStatsResponse {
        /// Kernel ID.
        #[prost(string, tag = "1")]
        pub kernel_id: String,
        /// Whether kernel is running.
        #[prost(bool, tag = "2")]
        pub running: bool,
        /// Current simulation step.
        #[prost(uint64, tag = "3")]
        pub current_step: u64,
        /// Total messages processed.
        #[prost(uint64, tag = "4")]
        pub messages_processed: u64,
        /// Pending commands.
        #[prost(uint32, tag = "5")]
        pub pending_commands: u32,
    }

    /// Streaming response message.
    #[derive(Clone, PartialEq, Message)]
    pub struct PersistentStreamResponse {
        /// Response type.
        #[prost(enumeration = "PersistentResponseType", tag = "1")]
        pub response_type: i32,
        /// Command ID if applicable.
        #[prost(uint64, optional, tag = "2")]
        pub command_id: Option<u64>,
        /// Current step for progress events.
        #[prost(uint64, optional, tag = "3")]
        pub current_step: Option<u64>,
        /// Remaining steps for progress events.
        #[prost(uint64, optional, tag = "4")]
        pub remaining: Option<u64>,
        /// Error code if applicable.
        #[prost(uint32, optional, tag = "5")]
        pub error_code: Option<u32>,
        /// Error message if applicable.
        #[prost(string, optional, tag = "6")]
        pub error_message: Option<String>,
    }

    /// Response type enum.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
    #[repr(i32)]
    pub enum PersistentResponseType {
        /// Acknowledgment.
        #[default]
        Ack = 0,
        /// Progress update.
        Progress = 1,
        /// Statistics.
        Stats = 2,
        /// Error.
        Error = 3,
        /// Terminated.
        Terminated = 4,
        /// Custom response.
        Custom = 5,
    }

    impl PersistentResponseType {
        /// Check if value is valid.
        pub fn is_valid(value: i32) -> bool {
            matches!(value, 0..=5)
        }
    }

    impl TryFrom<i32> for PersistentResponseType {
        type Error = ();

        fn try_from(value: i32) -> std::result::Result<Self, <Self as TryFrom<i32>>::Error> {
            match value {
                0 => Ok(PersistentResponseType::Ack),
                1 => Ok(PersistentResponseType::Progress),
                2 => Ok(PersistentResponseType::Stats),
                3 => Ok(PersistentResponseType::Error),
                4 => Ok(PersistentResponseType::Terminated),
                5 => Ok(PersistentResponseType::Custom),
                _ => Err(()),
            }
        }
    }

    impl From<PersistentResponse> for PersistentStreamResponse {
        fn from(response: PersistentResponse) -> Self {
            match response {
                PersistentResponse::Ack { cmd_id } => Self {
                    response_type: PersistentResponseType::Ack as i32,
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
                    response_type: PersistentResponseType::Progress as i32,
                    command_id: Some(cmd_id.0),
                    current_step: Some(current_step),
                    remaining: Some(remaining),
                    error_code: None,
                    error_message: None,
                },
                PersistentResponse::Stats { cmd_id, stats: _ } => Self {
                    response_type: PersistentResponseType::Stats as i32,
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
                    response_type: PersistentResponseType::Error as i32,
                    command_id: Some(cmd_id.0),
                    current_step: None,
                    remaining: None,
                    error_code: Some(code),
                    error_message: Some(message),
                },
                PersistentResponse::Terminated { final_step } => Self {
                    response_type: PersistentResponseType::Terminated as i32,
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
                    response_type: PersistentResponseType::Custom as i32,
                    command_id: Some(cmd_id.0),
                    current_step: None,
                    remaining: None,
                    error_code: None,
                    error_message: None,
                },
            }
        }
    }

    // ========================================================================
    // GRPC SERVICE
    // ========================================================================

    /// gRPC service for persistent GPU kernels.
    ///
    /// Provides unary RPCs for commands and server-streaming for responses.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use ringkernel_ecosystem::grpc::PersistentGrpcService;
    ///
    /// let service = PersistentGrpcService::new(handle, PersistentGrpcConfig::default());
    ///
    /// // Unary RPC
    /// let response = service.run_steps(PersistentStepRequest { count: 100, request_id: None }).await?;
    ///
    /// // Server streaming RPC
    /// let stream = service.stream_responses().await?;
    /// while let Some(response) = stream.next().await {
    ///     // Handle response
    /// }
    /// ```
    pub struct PersistentGrpcService<H: PersistentHandle> {
        /// Handle to the persistent kernel.
        handle: Arc<H>,
        /// Configuration.
        config: PersistentGrpcConfig,
        /// Broadcast channel for streaming responses.
        response_tx: broadcast::Sender<PersistentResponse>,
        /// Statistics.
        stats: Arc<ServerStats>,
    }

    impl<H: PersistentHandle> PersistentGrpcService<H> {
        /// Create a new persistent gRPC service.
        pub fn new(handle: Arc<H>, config: PersistentGrpcConfig) -> Self {
            let (response_tx, _) = broadcast::channel(config.broadcast_capacity);
            Self {
                handle,
                config,
                response_tx,
                stats: Arc::new(ServerStats::new()),
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

        /// Get server statistics.
        pub fn server_stats(&self) -> &ServerStats {
            &self.stats
        }

        /// Subscribe to response stream.
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

        /// Run simulation steps (unary RPC).
        pub async fn run_steps(
            &self,
            request: PersistentStepRequest,
        ) -> std::result::Result<PersistentCommandResponse, Status> {
            let start = std::time::Instant::now();

            if !self.handle.is_running() {
                self.stats.record_failure();
                return Err(Status::unavailable("Kernel not running"));
            }

            let cmd = PersistentCommand::RunSteps {
                count: request.count,
            };

            match self.handle.send_command(cmd) {
                Ok(cmd_id) => {
                    let latency_ns = start.elapsed().as_nanos() as u64;
                    self.stats.record_success(8, 24);
                    Ok(PersistentCommandResponse {
                        success: true,
                        command_id: cmd_id.0,
                        latency_ns,
                        error: None,
                        request_id: request.request_id,
                    })
                }
                Err(e) => {
                    self.stats.record_failure();
                    Ok(PersistentCommandResponse {
                        success: false,
                        command_id: 0,
                        latency_ns: start.elapsed().as_nanos() as u64,
                        error: Some(e.to_string()),
                        request_id: request.request_id,
                    })
                }
            }
        }

        /// Inject impulse (unary RPC).
        pub async fn inject_impulse(
            &self,
            request: PersistentImpulseRequest,
        ) -> std::result::Result<PersistentCommandResponse, Status> {
            let start = std::time::Instant::now();

            if !self.handle.is_running() {
                self.stats.record_failure();
                return Err(Status::unavailable("Kernel not running"));
            }

            let cmd = PersistentCommand::Inject {
                position: (request.x, request.y, request.z),
                value: request.value,
            };

            match self.handle.send_command(cmd) {
                Ok(cmd_id) => {
                    let latency_ns = start.elapsed().as_nanos() as u64;
                    self.stats.record_success(16, 24);
                    Ok(PersistentCommandResponse {
                        success: true,
                        command_id: cmd_id.0,
                        latency_ns,
                        error: None,
                        request_id: request.request_id,
                    })
                }
                Err(e) => {
                    self.stats.record_failure();
                    Ok(PersistentCommandResponse {
                        success: false,
                        command_id: 0,
                        latency_ns: start.elapsed().as_nanos() as u64,
                        error: Some(e.to_string()),
                        request_id: request.request_id,
                    })
                }
            }
        }

        /// Get current statistics (unary RPC).
        pub async fn get_stats(
            &self,
            _request: PersistentStatsRequest,
        ) -> std::result::Result<PersistentStatsResponse, Status> {
            let stats = self.handle.stats();

            Ok(PersistentStatsResponse {
                kernel_id: self.handle.kernel_id().to_string(),
                running: self.handle.is_running(),
                current_step: stats.current_step,
                messages_processed: stats.messages_processed,
                pending_commands: stats.pending_commands,
            })
        }

        /// Stream responses (server-streaming RPC).
        ///
        /// Returns a stream of responses from the persistent kernel.
        pub fn stream_responses(
            &self,
        ) -> Pin<Box<dyn Stream<Item = std::result::Result<PersistentStreamResponse, Status>> + Send>>
        {
            let rx = self.subscribe();

            let stream = async_stream::try_stream! {
                let mut rx = rx;
                loop {
                    match rx.recv().await {
                        Ok(response) => {
                            yield PersistentStreamResponse::from(response);
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!("Lagged {} messages in gRPC stream", n);
                            continue;
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            break;
                        }
                    }
                }
            };

            Box::pin(stream)
        }
    }

    impl<H: PersistentHandle> Clone for PersistentGrpcService<H> {
        fn clone(&self) -> Self {
            Self {
                handle: self.handle.clone(),
                config: self.config.clone(),
                response_tx: self.response_tx.clone(),
                stats: self.stats.clone(),
            }
        }
    }

    /// Builder for persistent gRPC service.
    pub struct PersistentGrpcServiceBuilder<H: PersistentHandle> {
        handle: Arc<H>,
        config: PersistentGrpcConfig,
    }

    impl<H: PersistentHandle> PersistentGrpcServiceBuilder<H> {
        /// Create a new builder.
        pub fn new(handle: Arc<H>) -> Self {
            Self {
                handle,
                config: PersistentGrpcConfig::default(),
            }
        }

        /// Set the configuration.
        pub fn config(mut self, config: PersistentGrpcConfig) -> Self {
            self.config = config;
            self
        }

        /// Set timeout.
        pub fn timeout(mut self, timeout: Duration) -> Self {
            self.config.base.timeout = timeout;
            self
        }

        /// Set broadcast capacity.
        pub fn broadcast_capacity(mut self, capacity: usize) -> Self {
            self.config.broadcast_capacity = capacity;
            self
        }

        /// Build the service.
        pub fn build(self) -> PersistentGrpcService<H> {
            PersistentGrpcService::new(self.handle, self.config)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRuntime;

    #[async_trait::async_trait]
    impl RuntimeHandle for MockRuntime {
        async fn process(
            &self,
            _kernel_id: &str,
            _data: Vec<u8>,
            _timeout: Duration,
        ) -> Result<Vec<u8>> {
            Ok(vec![1, 2, 3, 4])
        }

        fn available_kernels(&self) -> Vec<String> {
            vec!["kernel1".to_string(), "kernel2".to_string()]
        }

        fn is_kernel_healthy(&self, _kernel_id: &str) -> bool {
            true
        }

        fn get_kernel_metrics(&self, _kernel_id: &str) -> Option<KernelMetrics> {
            Some(KernelMetrics {
                messages_processed: 100,
                messages_dropped: 1,
                avg_latency_us: 500,
                queue_depth: 10,
            })
        }
    }

    #[test]
    fn test_grpc_config_default() {
        let config = GrpcConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.enable_reflection);
        assert!(config.enable_health_check);
    }

    #[test]
    fn test_server_stats() {
        let stats = ServerStats::new();
        stats.record_success(100, 200);
        stats.record_failure();

        assert_eq!(stats.total_requests.load(Ordering::Relaxed), 2);
        assert_eq!(stats.successful_requests.load(Ordering::Relaxed), 1);
        assert_eq!(stats.failed_requests.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_server_builder() {
        let runtime = Arc::new(MockRuntime);
        let server = GrpcServerBuilder::new(runtime)
            .timeout(Duration::from_secs(60))
            .max_message_size(32 * 1024 * 1024)
            .build();

        assert_eq!(server.config.timeout, Duration::from_secs(60));
        assert_eq!(server.config.max_message_size, 32 * 1024 * 1024);
    }

    #[test]
    fn test_get_kernel_info() {
        let runtime = Arc::new(MockRuntime);
        let server = RingKernelGrpcServer::new(runtime, GrpcConfig::default());

        let info = server.get_kernel_info("");
        assert_eq!(info.kernels.len(), 2);
        assert!(info.kernels.iter().all(|k| k.healthy));
    }

    #[tokio::test]
    async fn test_process_request() {
        let runtime = Arc::new(MockRuntime);
        let server = RingKernelGrpcServer::new(runtime, GrpcConfig::default());

        let request = ProcessRequest {
            kernel_id: "test".to_string(),
            payload: vec![1, 2, 3],
            timeout_ms: None,
            request_id: Some("req-123".to_string()),
        };

        let response = server.process_request(request).await.unwrap();
        assert_eq!(response.payload, vec![1, 2, 3, 4]);
        assert!(response.latency_us > 0);
        assert_eq!(response.request_id, Some("req-123".to_string()));
    }
}
