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
    #[allow(dead_code)]
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
///
/// Type parameters:
/// - `R`: The runtime handle type
/// - `Req`: The request message type
/// - `Resp`: The response message type
#[derive(Clone)]
pub struct RingKernelLayer<R, Req, Resp> {
    runtime: R,
    kernel_id: KernelId,
    config: ServiceConfig,
    _marker: PhantomData<(Req, Resp)>,
}

impl<R: RuntimeHandle, Req, Resp> RingKernelLayer<R, Req, Resp> {
    /// Create a new layer.
    pub fn new(runtime: R, kernel_id: impl Into<String>) -> Self {
        Self {
            runtime,
            kernel_id: KernelId::new(kernel_id),
            config: ServiceConfig::default(),
            _marker: PhantomData,
        }
    }

    /// Create a new layer with custom configuration.
    pub fn with_config(runtime: R, kernel_id: impl Into<String>, config: ServiceConfig) -> Self {
        Self {
            runtime,
            kernel_id: KernelId::new(kernel_id),
            config,
            _marker: PhantomData,
        }
    }
}

impl<R, S, Req, Resp> tower::Layer<S> for RingKernelLayer<R, Req, Resp>
where
    R: RuntimeHandle,
    Req: Send + 'static,
    Resp: Send + 'static,
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

// ============================================================================
// PERSISTENT KERNEL SERVICE INTEGRATION
// ============================================================================

#[cfg(feature = "persistent")]
pub use persistent_service::*;

#[cfg(feature = "persistent")]
mod persistent_service {
    use super::*;
    use crate::persistent::{
        CommandId, PersistentCommand, PersistentConfig, PersistentHandle, PersistentStats,
    };

    /// Configuration for persistent Tower service.
    #[derive(Debug, Clone, Default)]
    pub struct PersistentServiceConfig {
        /// Base service config.
        pub base: ServiceConfig,
        /// Persistent kernel config.
        pub persistent: PersistentConfig,
        /// Whether to wait for command acknowledgment.
        pub wait_for_ack: bool,
    }

    impl PersistentServiceConfig {
        /// Create a new configuration with default values.
        pub fn new() -> Self {
            Self::default()
        }

        /// Set whether to wait for command acknowledgment.
        pub fn with_wait_for_ack(mut self, wait: bool) -> Self {
            self.wait_for_ack = wait;
            self
        }
    }

    /// Tower service wrapping a persistent GPU kernel.
    ///
    /// Unlike `RingKernelService` which uses launch-per-request, this service
    /// maintains a single long-running GPU kernel and injects commands
    /// with ~0.03µs latency (vs ~317µs for traditional kernel launches).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use tower::ServiceBuilder;
    /// use ringkernel_ecosystem::tower::PersistentKernelService;
    ///
    /// let service = ServiceBuilder::new()
    ///     .timeout(Duration::from_secs(5))
    ///     .rate_limit(10000, Duration::from_secs(1))
    ///     .service(PersistentKernelService::new(handle));
    /// ```
    pub struct PersistentKernelService<H: PersistentHandle> {
        /// Handle to the persistent kernel.
        handle: Arc<H>,
        /// Configuration.
        config: PersistentServiceConfig,
        /// Active request count.
        active_requests: Arc<AtomicU64>,
        /// Service state.
        state: Arc<RwLock<ServiceState>>,
    }

    impl<H: PersistentHandle> PersistentKernelService<H> {
        /// Create a new persistent kernel service.
        pub fn new(handle: Arc<H>) -> Self {
            Self::with_config(handle, PersistentServiceConfig::default())
        }

        /// Create a new persistent kernel service with custom configuration.
        pub fn with_config(handle: Arc<H>, config: PersistentServiceConfig) -> Self {
            Self {
                handle,
                config,
                active_requests: Arc::new(AtomicU64::new(0)),
                state: Arc::new(RwLock::new(ServiceState::Ready)),
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
        pub fn stats(&self) -> PersistentStats {
            self.handle.stats()
        }

        /// Get current active request count.
        pub fn active_requests(&self) -> u64 {
            self.active_requests.load(Ordering::Relaxed)
        }

        /// Check if service is accepting requests.
        pub fn is_accepting(&self) -> bool {
            *self.state.read() == ServiceState::Ready
                && self.handle.is_running()
                && (self.active_requests.load(Ordering::Relaxed) as usize)
                    < self.config.base.max_concurrency
        }

        /// Gracefully shut down the service.
        pub fn shutdown(&self) {
            *self.state.write() = ServiceState::ShuttingDown;
        }
    }

    impl<H: PersistentHandle> Clone for PersistentKernelService<H> {
        fn clone(&self) -> Self {
            Self {
                handle: self.handle.clone(),
                config: self.config.clone(),
                active_requests: self.active_requests.clone(),
                state: self.state.clone(),
            }
        }
    }

    /// Request types for persistent kernel service.
    #[derive(Debug, Clone)]
    pub enum PersistentServiceRequest {
        /// Run simulation steps.
        RunSteps {
            /// Number of steps to run.
            count: u64,
        },
        /// Pause simulation.
        Pause,
        /// Resume simulation.
        Resume,
        /// Inject impulse at position.
        Inject {
            /// Position (x, y, z) to inject at.
            position: (u32, u32, u32),
            /// Value to inject.
            value: f32,
        },
        /// Get current statistics.
        GetStats,
        /// Custom command.
        Custom {
            /// Custom command type ID.
            type_id: u32,
            /// Custom command payload.
            payload: Vec<u8>,
        },
    }

    impl From<PersistentServiceRequest> for PersistentCommand {
        fn from(req: PersistentServiceRequest) -> Self {
            match req {
                PersistentServiceRequest::RunSteps { count } => {
                    PersistentCommand::RunSteps { count }
                }
                PersistentServiceRequest::Pause => PersistentCommand::Pause,
                PersistentServiceRequest::Resume => PersistentCommand::Resume,
                PersistentServiceRequest::Inject { position, value } => {
                    PersistentCommand::Inject { position, value }
                }
                PersistentServiceRequest::GetStats => PersistentCommand::GetStats,
                PersistentServiceRequest::Custom { type_id, payload } => {
                    PersistentCommand::Custom { type_id, payload }
                }
            }
        }
    }

    /// Response types for persistent kernel service.
    #[derive(Debug, Clone)]
    pub struct PersistentServiceResponse {
        /// Command ID for tracking.
        pub command_id: CommandId,
        /// Processing latency in nanoseconds.
        pub latency_ns: u64,
    }

    impl<H: PersistentHandle + 'static> Service<PersistentServiceRequest>
        for PersistentKernelService<H>
    {
        type Response = PersistentServiceResponse;
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
                    if self.handle.is_running() {
                        Poll::Ready(Ok(()))
                    } else {
                        Poll::Ready(Err(EcosystemError::KernelNotRunning(
                            self.handle.kernel_id().to_string(),
                        )))
                    }
                }
            }
        }

        fn call(&mut self, req: PersistentServiceRequest) -> Self::Future {
            let handle = self.handle.clone();
            let active_requests = self.active_requests.clone();
            let wait_for_ack = self.config.wait_for_ack;
            let timeout = self.config.base.timeout;

            // Increment active requests
            active_requests.fetch_add(1, Ordering::Relaxed);

            Box::pin(async move {
                let start = std::time::Instant::now();

                // Convert to persistent command and send
                let cmd: PersistentCommand = req.into();
                let cmd_id = match handle.send_command(cmd) {
                    Ok(id) => id,
                    Err(e) => {
                        active_requests.fetch_sub(1, Ordering::Relaxed);
                        return Err(e);
                    }
                };

                // Optionally wait for acknowledgment
                if wait_for_ack {
                    if let Err(e) = handle.wait_for_command(cmd_id, timeout).await {
                        active_requests.fetch_sub(1, Ordering::Relaxed);
                        return Err(e);
                    }
                }

                // Decrement active requests
                active_requests.fetch_sub(1, Ordering::Relaxed);

                let latency_ns = start.elapsed().as_nanos() as u64;

                Ok(PersistentServiceResponse {
                    command_id: cmd_id,
                    latency_ns,
                })
            })
        }
    }

    /// Builder for constructing persistent kernel services.
    pub struct PersistentServiceBuilder<H: PersistentHandle> {
        handle: Arc<H>,
        config: PersistentServiceConfig,
    }

    impl<H: PersistentHandle> PersistentServiceBuilder<H> {
        /// Create a new service builder.
        pub fn new(handle: Arc<H>) -> Self {
            Self {
                handle,
                config: PersistentServiceConfig::default(),
            }
        }

        /// Set the timeout.
        pub fn timeout(mut self, timeout: Duration) -> Self {
            self.config.base.timeout = timeout;
            self
        }

        /// Set maximum concurrency.
        pub fn max_concurrency(mut self, max: usize) -> Self {
            self.config.base.max_concurrency = max;
            self
        }

        /// Set whether to wait for acknowledgment.
        pub fn wait_for_ack(mut self, wait: bool) -> Self {
            self.config.wait_for_ack = wait;
            self
        }

        /// Build the service.
        pub fn build(self) -> PersistentKernelService<H> {
            PersistentKernelService::with_config(self.handle, self.config)
        }
    }

    /// Layer for creating persistent kernel services.
    #[derive(Clone)]
    pub struct PersistentKernelLayer<H: PersistentHandle> {
        handle: Arc<H>,
        config: PersistentServiceConfig,
    }

    impl<H: PersistentHandle> PersistentKernelLayer<H> {
        /// Create a new layer.
        pub fn new(handle: Arc<H>) -> Self {
            Self {
                handle,
                config: PersistentServiceConfig::default(),
            }
        }

        /// Create a new layer with custom configuration.
        pub fn with_config(handle: Arc<H>, config: PersistentServiceConfig) -> Self {
            Self { handle, config }
        }
    }

    impl<H: PersistentHandle + 'static, S> tower::Layer<S> for PersistentKernelLayer<H> {
        type Service = PersistentKernelService<H>;

        fn layer(&self, _inner: S) -> Self::Service {
            PersistentKernelService::with_config(self.handle.clone(), self.config.clone())
        }
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
