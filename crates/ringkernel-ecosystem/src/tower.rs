//! Tower service integration for RingKernel.
//!
//! This module provides Tower service implementations that wrap GPU Ring Kernels,
//! enabling use of Tower middleware (timeouts, rate limiting, load shedding, etc.)
//! with GPU-accelerated backends.
//!
//! # Example
//!
//! ```ignore
//! use tower::ServiceBuilder;
//! use ringkernel_ecosystem::tower::RingKernelService;
//!
//! let service = ServiceBuilder::new()
//!     .timeout(Duration::from_secs(5))
//!     .rate_limit(1000, Duration::from_secs(1))
//!     .service(RingKernelService::new(runtime, "vector_add"));
//! ```

use futures::future::BoxFuture;
use parking_lot::RwLock;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use ringkernel_core::message::RingMessage;
use ringkernel_core::runtime::KernelId;
use tower_service::Service;

use crate::error::{EcosystemError, Result};

/// Configuration for RingKernel Tower service.
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// Default timeout for operations.
    pub timeout: Duration,
    /// Maximum concurrent requests.
    pub max_concurrency: usize,
    /// Enable request buffering.
    pub buffer_requests: bool,
    /// Buffer size if buffering is enabled.
    pub buffer_size: usize,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_concurrency: 1000,
            buffer_requests: false,
            buffer_size: 1024,
        }
    }
}

/// Runtime handle trait for abstracting the RingKernel runtime.
#[async_trait::async_trait]
pub trait RuntimeHandle: Send + Sync + Clone + 'static {
    /// Send a message to a kernel.
    async fn send_message(&self, kernel_id: &str, data: Vec<u8>) -> Result<()>;

    /// Receive a message from a kernel with timeout.
    async fn receive_message(&self, kernel_id: &str, timeout: Duration) -> Result<Vec<u8>>;

    /// Check if service is ready.
    fn is_ready(&self, kernel_id: &str) -> bool;
}

/// Ring Kernel as a Tower service.
///
/// This service wraps a GPU kernel and implements the Tower `Service` trait,
/// enabling use of Tower middleware and the broader Tower ecosystem.
pub struct RingKernelService<R, Req, Resp> {
    /// Runtime handle.
    runtime: R,
    /// Target kernel ID.
    kernel_id: KernelId,
    /// Configuration.
    config: ServiceConfig,
    /// Active request count.
    active_requests: Arc<AtomicU64>,
    /// Service state.
    state: Arc<RwLock<ServiceState>>,
    /// Type markers.
    _marker: PhantomData<(Req, Resp)>,
}

/// Internal service state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServiceState {
    /// Service is ready to accept requests.
    Ready,
    /// Service is temporarily overloaded.
    Overloaded,
    /// Service is shutting down.
    ShuttingDown,
}

impl<R, Req, Resp> RingKernelService<R, Req, Resp>
where
    R: RuntimeHandle,
{
    /// Create a new RingKernel service.
    pub fn new(runtime: R, kernel_id: impl Into<String>) -> Self {
        Self::with_config(runtime, kernel_id, ServiceConfig::default())
    }

    /// Create a new RingKernel service with custom configuration.
    pub fn with_config(runtime: R, kernel_id: impl Into<String>, config: ServiceConfig) -> Self {
        Self {
            runtime,
            kernel_id: KernelId::new(kernel_id),
            config,
            active_requests: Arc::new(AtomicU64::new(0)),
            state: Arc::new(RwLock::new(ServiceState::Ready)),
            _marker: PhantomData,
        }
    }

    /// Get the kernel ID.
    pub fn kernel_id(&self) -> &KernelId {
        &self.kernel_id
    }

    /// Get current active request count.
    pub fn active_requests(&self) -> u64 {
        self.active_requests.load(Ordering::Relaxed)
    }

    /// Check if service is accepting requests.
    pub fn is_accepting(&self) -> bool {
        *self.state.read() == ServiceState::Ready
            && (self.active_requests.load(Ordering::Relaxed) as usize) < self.config.max_concurrency
    }

    /// Gracefully shut down the service.
    pub fn shutdown(&self) {
        *self.state.write() = ServiceState::ShuttingDown;
    }
}

impl<R, Req, Resp> Clone for RingKernelService<R, Req, Resp>
where
    R: Clone,
{
    fn clone(&self) -> Self {
        Self {
            runtime: self.runtime.clone(),
            kernel_id: self.kernel_id.clone(),
            config: self.config.clone(),
            active_requests: self.active_requests.clone(),
            state: self.state.clone(),
            _marker: PhantomData,
        }
    }
}

/// Request wrapper for Tower service.
pub struct ServiceRequest<T> {
    /// The inner request.
    pub inner: T,
    /// Optional request ID for tracing.
    pub request_id: Option<String>,
}

impl<T> ServiceRequest<T> {
    /// Create a new service request.
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            request_id: None,
        }
    }

    /// Create a service request with a request ID.
    pub fn with_id(inner: T, request_id: impl Into<String>) -> Self {
        Self {
            inner,
            request_id: Some(request_id.into()),
        }
    }
}

/// Response wrapper from Tower service.
#[derive(Debug, Clone)]
pub struct ServiceResponse<T> {
    /// The inner response.
    pub inner: T,
    /// Processing latency in microseconds.
    pub latency_us: u64,
    /// Request ID if provided.
    pub request_id: Option<String>,
}

impl<R, Req, Resp> Service<ServiceRequest<Req>> for RingKernelService<R, Req, Resp>
where
    R: RuntimeHandle,
    Req: RingMessage + Send + 'static,
    Resp: RingMessage + Send + 'static,
{
    type Response = ServiceResponse<Resp>;
    type Error = EcosystemError;
    type Future = BoxFuture<'static, Result<Self::Response>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        let state = *self.state.read();
        match state {
            ServiceState::ShuttingDown => Poll::Ready(Err(EcosystemError::ServiceUnavailable(
                "Service is shutting down".to_string(),
            ))),
            ServiceState::Overloaded => Poll::Ready(Err(EcosystemError::ServiceUnavailable(
                "Service is overloaded".to_string(),
            ))),
            ServiceState::Ready => {
                if self.runtime.is_ready(self.kernel_id.as_str()) {
                    Poll::Ready(Ok(()))
                } else {
                    Poll::Pending
                }
            }
        }
    }

    fn call(&mut self, req: ServiceRequest<Req>) -> Self::Future {
        let runtime = self.runtime.clone();
        let kernel_id = self.kernel_id.as_str().to_string();
        let timeout = self.config.timeout;
        let active_requests = self.active_requests.clone();
        let request_id = req.request_id.clone();

        // Increment active requests
        active_requests.fetch_add(1, Ordering::Relaxed);

        Box::pin(async move {
            let start = std::time::Instant::now();

            // Serialize request
            let data = req.inner.serialize();

            // Send to kernel
            let send_result = runtime.send_message(&kernel_id, data).await;
            if let Err(e) = send_result {
                active_requests.fetch_sub(1, Ordering::Relaxed);
                return Err(e);
            }

            // Wait for response
            let response_data = match runtime.receive_message(&kernel_id, timeout).await {
                Ok(data) => data,
                Err(e) => {
                    active_requests.fetch_sub(1, Ordering::Relaxed);
                    return Err(e);
                }
            };

            // Decrement active requests
            active_requests.fetch_sub(1, Ordering::Relaxed);

            // Deserialize response
            let response = Resp::deserialize(&response_data).map_err(|e| {
                EcosystemError::Deserialization(format!("Failed to deserialize response: {}", e))
            })?;

            let latency_us = start.elapsed().as_micros() as u64;

            Ok(ServiceResponse {
                inner: response,
                latency_us,
                request_id,
            })
        })
    }
}

/// Builder for constructing RingKernel services with middleware.
pub struct ServiceBuilder<R> {
    runtime: R,
    config: ServiceConfig,
}

impl<R: RuntimeHandle> ServiceBuilder<R> {
    /// Create a new service builder.
    pub fn new(runtime: R) -> Self {
        Self {
            runtime,
            config: ServiceConfig::default(),
        }
    }

    /// Set the timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set maximum concurrency.
    pub fn max_concurrency(mut self, max: usize) -> Self {
        self.config.max_concurrency = max;
        self
    }

    /// Enable request buffering.
    pub fn buffer(mut self, size: usize) -> Self {
        self.config.buffer_requests = true;
        self.config.buffer_size = size;
        self
    }

    /// Build a service for the specified kernel.
    pub fn build<Req, Resp>(self, kernel_id: impl Into<String>) -> RingKernelService<R, Req, Resp> {
        RingKernelService::with_config(self.runtime, kernel_id, self.config)
    }
}

/// Layer for creating RingKernel services.
#[derive(Clone)]
pub struct RingKernelLayer<R> {
    runtime: R,
    kernel_id: KernelId,
    config: ServiceConfig,
}

impl<R: RuntimeHandle> RingKernelLayer<R> {
    /// Create a new layer.
    pub fn new(runtime: R, kernel_id: impl Into<String>) -> Self {
        Self {
            runtime,
            kernel_id: KernelId::new(kernel_id),
            config: ServiceConfig::default(),
        }
    }

    /// Create a new layer with custom configuration.
    pub fn with_config(runtime: R, kernel_id: impl Into<String>, config: ServiceConfig) -> Self {
        Self {
            runtime,
            kernel_id: KernelId::new(kernel_id),
            config,
        }
    }
}

impl<R, S, Req, Resp> tower::Layer<S> for RingKernelLayer<R>
where
    R: RuntimeHandle,
{
    type Service = RingKernelService<R, Req, Resp>;

    fn layer(&self, _inner: S) -> Self::Service {
        RingKernelService::with_config(
            self.runtime.clone(),
            self.kernel_id.as_str(),
            self.config.clone(),
        )
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

        fn is_ready(&self, _kernel_id: &str) -> bool {
            true
        }
    }

    #[test]
    fn test_service_config_default() {
        let config = ServiceConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_concurrency, 1000);
        assert!(!config.buffer_requests);
    }

    #[test]
    fn test_service_builder() {
        let runtime = MockRuntime;
        let service: RingKernelService<_, Vec<u8>, Vec<u8>> = ServiceBuilder::new(runtime)
            .timeout(Duration::from_secs(5))
            .max_concurrency(100)
            .build("test_kernel");

        assert_eq!(service.kernel_id().as_str(), "test_kernel");
        assert_eq!(service.active_requests(), 0);
        assert!(service.is_accepting());
    }

    #[test]
    fn test_service_shutdown() {
        let runtime = MockRuntime;
        let service: RingKernelService<_, Vec<u8>, Vec<u8>> =
            RingKernelService::new(runtime, "test_kernel");

        assert!(service.is_accepting());
        service.shutdown();
        assert!(!service.is_accepting());
    }
}
