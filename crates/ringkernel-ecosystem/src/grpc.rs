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

use parking_lot::RwLock;
use prost::Message;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tonic::{Request, Response, Status};

use crate::error::{EcosystemError, Result};

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
    async fn process(&self, kernel_id: &str, data: Vec<u8>, timeout: Duration)
        -> Result<Vec<u8>>;

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
        self.bytes_received.fetch_add(bytes_in as u64, Ordering::Relaxed);
        self.bytes_sent.fetch_add(bytes_out as u64, Ordering::Relaxed);
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
                messages_processed: metrics
                    .as_ref()
                    .map(|m| m.messages_processed)
                    .unwrap_or(0),
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
