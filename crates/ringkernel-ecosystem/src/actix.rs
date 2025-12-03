//! Actix actor framework integration for RingKernel.
//!
//! This module provides a bridge between Actix actors and GPU Ring Kernels,
//! allowing seamless message passing between the CPU actor system and
//! GPU-resident persistent actors.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ecosystem::actix::{GpuActorBridge, GpuActorConfig};
//!
//! let bridge = GpuActorBridge::new(runtime, "vector_add", GpuActorConfig::default()).start();
//! let result: VectorAddResponse = bridge.send(request).await??;
//! ```

use actix::prelude::*;
use parking_lot::RwLock;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use ringkernel_core::message::RingMessage;
use ringkernel_core::runtime::KernelId;

use crate::error::{EcosystemError, Result};

/// Configuration for GPU actor bridge.
#[derive(Debug, Clone)]
pub struct GpuActorConfig {
    /// Timeout for GPU operations.
    pub timeout: Duration,
    /// Maximum in-flight messages.
    pub max_in_flight: usize,
    /// Enable response caching.
    pub enable_caching: bool,
}

impl Default for GpuActorConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_in_flight: 1000,
            enable_caching: false,
        }
    }
}

/// Runtime handle trait for abstracting the RingKernel runtime.
///
/// This trait allows the bridge to work with any compatible runtime implementation.
#[async_trait::async_trait]
pub trait RuntimeHandle: Send + Sync + 'static {
    /// Send a message to a kernel.
    async fn send_message(&self, kernel_id: &str, data: Vec<u8>) -> Result<()>;

    /// Receive a message from a kernel with timeout.
    async fn receive_message(&self, kernel_id: &str, timeout: Duration) -> Result<Vec<u8>>;

    /// Check if a kernel is available.
    fn is_kernel_available(&self, kernel_id: &str) -> bool;
}

/// Bridge between Actix actors and GPU Ring Kernels.
///
/// This actor forwards messages to GPU kernels and returns responses
/// to the calling Actix actor.
pub struct GpuActorBridge<R: RuntimeHandle> {
    /// Runtime handle.
    runtime: Arc<R>,
    /// Target kernel ID.
    kernel_id: KernelId,
    /// Configuration.
    config: GpuActorConfig,
    /// In-flight message counter.
    in_flight: Arc<RwLock<usize>>,
}

impl<R: RuntimeHandle> GpuActorBridge<R> {
    /// Create a new GPU actor bridge.
    pub fn new(runtime: Arc<R>, kernel_id: impl Into<String>, config: GpuActorConfig) -> Self {
        Self {
            runtime,
            kernel_id: KernelId::new(kernel_id),
            config,
            in_flight: Arc::new(RwLock::new(0)),
        }
    }

    /// Get the kernel ID this bridge connects to.
    pub fn kernel_id(&self) -> &KernelId {
        &self.kernel_id
    }

    /// Get current number of in-flight messages.
    pub fn in_flight_count(&self) -> usize {
        *self.in_flight.read()
    }

    /// Check if the bridge can accept more messages.
    pub fn can_accept(&self) -> bool {
        *self.in_flight.read() < self.config.max_in_flight
    }
}

impl<R: RuntimeHandle> Actor for GpuActorBridge<R> {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        tracing::info!(
            kernel_id = %self.kernel_id,
            "GPU actor bridge started"
        );
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        tracing::info!(
            kernel_id = %self.kernel_id,
            in_flight = self.in_flight_count(),
            "GPU actor bridge stopped"
        );
    }
}

/// Message wrapper for sending to GPU kernels.
#[derive(Message)]
#[rtype(result = "Result<GpuResponse>")]
pub struct GpuRequest {
    /// Serialized request data.
    pub data: Vec<u8>,
    /// Optional correlation ID.
    pub correlation_id: Option<String>,
}

impl GpuRequest {
    /// Create a new GPU request from a RingMessage.
    pub fn from_message<M: RingMessage>(message: &M) -> Self {
        Self {
            data: message.serialize(),
            correlation_id: message.correlation_id().map(|id| id.to_string()),
        }
    }
}

/// Response from GPU kernel.
#[derive(Debug, Clone)]
pub struct GpuResponse {
    /// Serialized response data.
    pub data: Vec<u8>,
    /// Processing latency in microseconds.
    pub latency_us: u64,
}

impl<R: RuntimeHandle> Handler<GpuRequest> for GpuActorBridge<R> {
    type Result = ResponseActFuture<Self, Result<GpuResponse>>;

    fn handle(&mut self, msg: GpuRequest, _ctx: &mut Self::Context) -> Self::Result {
        // Check capacity
        if !self.can_accept() {
            return Box::pin(
                async {
                    Err(EcosystemError::ServiceUnavailable(
                        "Too many in-flight messages".to_string(),
                    ))
                }
                .into_actor(self),
            );
        }

        // Increment in-flight counter
        {
            let mut count = self.in_flight.write();
            *count += 1;
        }

        let runtime = self.runtime.clone();
        let kernel_id = self.kernel_id.as_str().to_string();
        let timeout = self.config.timeout;
        let in_flight = self.in_flight.clone();
        let start = std::time::Instant::now();

        Box::pin(
            async move {
                // Send to kernel
                runtime.send_message(&kernel_id, msg.data).await?;

                // Wait for response
                let response_data = runtime.receive_message(&kernel_id, timeout).await?;

                let latency_us = start.elapsed().as_micros() as u64;

                Ok(GpuResponse {
                    data: response_data,
                    latency_us,
                })
            }
            .into_actor(self)
            .map(move |result, _act, _ctx| {
                // Decrement in-flight counter
                let mut count = in_flight.write();
                *count = count.saturating_sub(1);
                result
            }),
        )
    }
}

/// Typed GPU request message.
///
/// This provides type-safe message passing between Actix and GPU kernels.
#[derive(Message)]
#[rtype(result = "Result<Resp>")]
pub struct TypedGpuRequest<Req, Resp> {
    /// The request message.
    pub request: Req,
    /// Phantom data for response type.
    _response: PhantomData<Resp>,
}

impl<Req, Resp> TypedGpuRequest<Req, Resp> {
    /// Create a new typed request.
    pub fn new(request: Req) -> Self {
        Self {
            request,
            _response: PhantomData,
        }
    }
}

/// Extension trait for creating GPU actor bridges from Actix system.
pub trait GpuActorExt {
    /// Create and start a GPU actor bridge.
    fn start_gpu_bridge<R: RuntimeHandle>(
        runtime: Arc<R>,
        kernel_id: impl Into<String>,
        config: GpuActorConfig,
    ) -> Addr<GpuActorBridge<R>>;
}

impl GpuActorExt for System {
    fn start_gpu_bridge<R: RuntimeHandle>(
        runtime: Arc<R>,
        kernel_id: impl Into<String>,
        config: GpuActorConfig,
    ) -> Addr<GpuActorBridge<R>> {
        GpuActorBridge::new(runtime, kernel_id, config).start()
    }
}

/// Health check message for the bridge.
#[derive(Message)]
#[rtype(result = "BridgeHealth")]
pub struct HealthCheck;

/// Bridge health status.
#[derive(Debug, Clone)]
pub struct BridgeHealth {
    /// Whether the bridge is healthy.
    pub healthy: bool,
    /// Number of in-flight messages.
    pub in_flight: usize,
    /// Maximum allowed in-flight messages.
    pub max_in_flight: usize,
    /// Whether the target kernel is available.
    pub kernel_available: bool,
}

impl<R: RuntimeHandle> Handler<HealthCheck> for GpuActorBridge<R> {
    type Result = BridgeHealth;

    fn handle(&mut self, _msg: HealthCheck, _ctx: &mut Self::Context) -> Self::Result {
        let in_flight = self.in_flight_count();
        let kernel_available = self.runtime.is_kernel_available(self.kernel_id.as_str());

        BridgeHealth {
            healthy: kernel_available && in_flight < self.config.max_in_flight,
            in_flight,
            max_in_flight: self.config.max_in_flight,
            kernel_available,
        }
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

        fn is_kernel_available(&self, _kernel_id: &str) -> bool {
            true
        }
    }

    #[test]
    fn test_gpu_actor_config_default() {
        let config = GpuActorConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_in_flight, 1000);
        assert!(!config.enable_caching);
    }

    #[test]
    fn test_bridge_creation() {
        let runtime = Arc::new(MockRuntime);
        let bridge = GpuActorBridge::new(runtime, "test_kernel", GpuActorConfig::default());
        assert_eq!(bridge.kernel_id().as_str(), "test_kernel");
        assert!(bridge.can_accept());
        assert_eq!(bridge.in_flight_count(), 0);
    }
}
