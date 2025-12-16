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
            correlation_id: Some(message.correlation_id().0.to_string()),
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
pub struct TypedGpuRequest<Req, Resp: 'static> {
    /// The request message.
    pub request: Req,
    /// Phantom data for response type.
    _response: PhantomData<Resp>,
}

impl<Req, Resp: 'static> TypedGpuRequest<Req, Resp> {
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
    type Result = MessageResult<HealthCheck>;

    fn handle(&mut self, _msg: HealthCheck, _ctx: &mut Self::Context) -> Self::Result {
        let in_flight = self.in_flight_count();
        let kernel_available = self.runtime.is_kernel_available(self.kernel_id.as_str());

        MessageResult(BridgeHealth {
            healthy: kernel_available && in_flight < self.config.max_in_flight,
            in_flight,
            max_in_flight: self.config.max_in_flight,
            kernel_available,
        })
    }
}

// ============================================================================
// PERSISTENT GPU ACTOR INTEGRATION
// ============================================================================

#[cfg(feature = "persistent")]
pub use persistent_actor::*;

#[cfg(feature = "persistent")]
mod persistent_actor {
    use super::*;
    use crate::persistent::{
        CommandId, CommandTracker, PersistentCommand, PersistentConfig, PersistentHandle,
        PersistentResponse, PersistentStats,
    };
    use std::collections::HashMap;

    /// Configuration for persistent GPU actor.
    #[derive(Debug, Clone)]
    pub struct PersistentActorConfig {
        /// Base actor config.
        pub base: GpuActorConfig,
        /// Persistent kernel config.
        pub persistent: PersistentConfig,
        /// Buffer size for response broadcasting.
        pub response_buffer_size: usize,
    }

    impl Default for PersistentActorConfig {
        fn default() -> Self {
            Self {
                base: GpuActorConfig::default(),
                persistent: PersistentConfig::default(),
                response_buffer_size: 1024,
            }
        }
    }

    impl PersistentActorConfig {
        /// Create a new configuration with default values.
        pub fn new() -> Self {
            Self::default()
        }

        /// Set the base actor configuration.
        pub fn with_base(mut self, base: GpuActorConfig) -> Self {
            self.base = base;
            self
        }

        /// Set the persistent kernel configuration.
        pub fn with_persistent(mut self, persistent: PersistentConfig) -> Self {
            self.persistent = persistent;
            self
        }

        /// Set the response buffer size.
        pub fn with_response_buffer_size(mut self, size: usize) -> Self {
            self.response_buffer_size = size;
            self
        }
    }

    /// Actix actor backed by a persistent GPU kernel.
    ///
    /// Unlike `GpuActorBridge` which uses launch-per-message, this actor
    /// maintains a single long-running GPU kernel and injects commands
    /// with ~0.03µs latency (vs ~317µs for traditional kernel launches).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use ringkernel_ecosystem::actix::{GpuPersistentActor, PersistentActorConfig};
    /// use ringkernel_ecosystem::persistent::PersistentHandle;
    ///
    /// // Create handle to persistent kernel (e.g., CudaPersistentHandle)
    /// let handle: Arc<dyn PersistentHandle> = create_persistent_handle();
    ///
    /// // Start the actor
    /// let actor = GpuPersistentActor::new(handle, PersistentActorConfig::default()).start();
    ///
    /// // Send commands with ultra-low latency
    /// actor.send(RunStepsCmd { count: 1000 }).await?;
    /// actor.send(InjectCmd { position: (10, 10, 10), value: 1.0 }).await?;
    /// ```
    pub struct GpuPersistentActor<H: PersistentHandle> {
        /// Handle to the persistent kernel.
        handle: Arc<H>,
        /// Configuration.
        config: PersistentActorConfig,
        /// Command tracker for correlating responses.
        tracker: CommandTracker,
        /// Pending command waiters.
        pending: HashMap<CommandId, tokio::sync::oneshot::Sender<PersistentResponse>>,
        /// Subscribers for response streaming.
        subscribers: Vec<Recipient<PersistentResponseMsg>>,
        /// Handle for the response polling interval.
        poll_handle: Option<SpawnHandle>,
    }

    impl<H: PersistentHandle> GpuPersistentActor<H> {
        /// Create a new persistent GPU actor.
        pub fn new(handle: Arc<H>, config: PersistentActorConfig) -> Self {
            Self {
                handle,
                config,
                tracker: CommandTracker::new(),
                pending: HashMap::new(),
                subscribers: Vec::new(),
                poll_handle: None,
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

        /// Poll for responses and dispatch them.
        fn poll_responses(&mut self, _ctx: &mut Context<Self>) {
            for response in self.handle.poll_responses() {
                // Check for pending command waiters
                if let Some(cmd_id) = response.command_id() {
                    if let Some(sender) = self.pending.remove(&cmd_id) {
                        let _ = sender.send(response.clone());
                    }
                }

                // Notify tracker
                self.tracker.notify(&response);

                // Broadcast to subscribers
                let msg = PersistentResponseMsg(response);
                self.subscribers
                    .retain(|subscriber| subscriber.try_send(msg.clone()).is_ok());
            }
        }
    }

    impl<H: PersistentHandle + 'static> Actor for GpuPersistentActor<H> {
        type Context = Context<Self>;

        fn started(&mut self, ctx: &mut Self::Context) {
            tracing::info!(
                kernel_id = %self.kernel_id(),
                "Persistent GPU actor started"
            );

            // Start response polling loop
            self.poll_handle = Some(
                ctx.run_interval(self.config.persistent.poll_interval, |act, ctx| {
                    act.poll_responses(ctx)
                }),
            );
        }

        fn stopped(&mut self, _ctx: &mut Self::Context) {
            tracing::info!(
                kernel_id = %self.kernel_id(),
                "Persistent GPU actor stopped"
            );

            // Initiate graceful shutdown if kernel is still running
            if self.handle.is_running() {
                let handle = self.handle.clone();
                actix_rt::spawn(async move {
                    if let Err(e) = handle.shutdown().await {
                        tracing::warn!("Failed to shut down kernel: {}", e);
                    }
                });
            }
        }
    }

    // ========================================================================
    // COMMAND MESSAGES
    // ========================================================================

    /// Run simulation steps command.
    #[derive(Message, Debug, Clone)]
    #[rtype(result = "Result<CommandId>")]
    pub struct RunStepsCmd {
        /// Number of steps to run.
        pub count: u64,
    }

    impl RunStepsCmd {
        /// Create a new run steps command.
        pub fn new(count: u64) -> Self {
            Self { count }
        }
    }

    /// Pause kernel command.
    #[derive(Message, Debug, Clone, Copy)]
    #[rtype(result = "Result<()>")]
    pub struct PauseCmd;

    /// Resume kernel command.
    #[derive(Message, Debug, Clone, Copy)]
    #[rtype(result = "Result<()>")]
    pub struct ResumeCmd;

    /// Inject impulse command.
    #[derive(Message, Debug, Clone)]
    #[rtype(result = "Result<CommandId>")]
    pub struct InjectCmd {
        /// Position to inject at (x, y, z).
        pub position: (u32, u32, u32),
        /// Value to inject.
        pub value: f32,
    }

    impl InjectCmd {
        /// Create a new inject command.
        pub fn new(x: u32, y: u32, z: u32, value: f32) -> Self {
            Self {
                position: (x, y, z),
                value,
            }
        }
    }

    /// Get current stats command.
    #[derive(Message, Debug, Clone, Copy)]
    #[rtype(result = "PersistentStats")]
    pub struct GetStatsCmd;

    /// Subscribe to response stream.
    #[derive(Message)]
    #[rtype(result = "()")]
    pub struct Subscribe(pub Recipient<PersistentResponseMsg>);

    /// Unsubscribe from response stream.
    #[derive(Message)]
    #[rtype(result = "()")]
    pub struct Unsubscribe(pub Recipient<PersistentResponseMsg>);

    /// Shutdown the kernel gracefully.
    #[derive(Message, Debug, Clone, Copy)]
    #[rtype(result = "Result<()>")]
    pub struct ShutdownCmd;

    /// Persistent response message for subscribers.
    #[derive(Message, Debug, Clone)]
    #[rtype(result = "()")]
    pub struct PersistentResponseMsg(pub PersistentResponse);

    // ========================================================================
    // HANDLER IMPLEMENTATIONS
    // ========================================================================

    impl<H: PersistentHandle + 'static> Handler<RunStepsCmd> for GpuPersistentActor<H> {
        type Result = Result<CommandId>;

        fn handle(&mut self, msg: RunStepsCmd, _ctx: &mut Self::Context) -> Self::Result {
            let cmd = PersistentCommand::RunSteps { count: msg.count };
            self.handle.send_command(cmd)
        }
    }

    impl<H: PersistentHandle + 'static> Handler<PauseCmd> for GpuPersistentActor<H> {
        type Result = Result<()>;

        fn handle(&mut self, _msg: PauseCmd, _ctx: &mut Self::Context) -> Self::Result {
            self.handle.send_command(PersistentCommand::Pause)?;
            Ok(())
        }
    }

    impl<H: PersistentHandle + 'static> Handler<ResumeCmd> for GpuPersistentActor<H> {
        type Result = Result<()>;

        fn handle(&mut self, _msg: ResumeCmd, _ctx: &mut Self::Context) -> Self::Result {
            self.handle.send_command(PersistentCommand::Resume)?;
            Ok(())
        }
    }

    impl<H: PersistentHandle + 'static> Handler<InjectCmd> for GpuPersistentActor<H> {
        type Result = Result<CommandId>;

        fn handle(&mut self, msg: InjectCmd, _ctx: &mut Self::Context) -> Self::Result {
            let cmd = PersistentCommand::Inject {
                position: msg.position,
                value: msg.value,
            };
            self.handle.send_command(cmd)
        }
    }

    impl<H: PersistentHandle + 'static> Handler<GetStatsCmd> for GpuPersistentActor<H> {
        type Result = MessageResult<GetStatsCmd>;

        fn handle(&mut self, _msg: GetStatsCmd, _ctx: &mut Self::Context) -> Self::Result {
            MessageResult(self.handle.stats())
        }
    }

    impl<H: PersistentHandle + 'static> Handler<Subscribe> for GpuPersistentActor<H> {
        type Result = ();

        fn handle(&mut self, msg: Subscribe, _ctx: &mut Self::Context) -> Self::Result {
            if !self.subscribers.contains(&msg.0) {
                self.subscribers.push(msg.0);
            }
        }
    }

    impl<H: PersistentHandle + 'static> Handler<Unsubscribe> for GpuPersistentActor<H> {
        type Result = ();

        fn handle(&mut self, msg: Unsubscribe, _ctx: &mut Self::Context) -> Self::Result {
            self.subscribers.retain(|s| s != &msg.0);
        }
    }

    impl<H: PersistentHandle + 'static> Handler<ShutdownCmd> for GpuPersistentActor<H> {
        type Result = ResponseActFuture<Self, Result<()>>;

        fn handle(&mut self, _msg: ShutdownCmd, _ctx: &mut Self::Context) -> Self::Result {
            let handle = self.handle.clone();
            Box::pin(async move { handle.shutdown().await }.into_actor(self))
        }
    }

    // ========================================================================
    // EXTENSION TRAIT
    // ========================================================================

    /// Extension trait for creating persistent GPU actor bridges from Actix system.
    pub trait GpuPersistentActorExt {
        /// Create and start a persistent GPU actor.
        fn start_persistent_gpu<H: PersistentHandle + 'static>(
            handle: Arc<H>,
            config: PersistentActorConfig,
        ) -> Addr<GpuPersistentActor<H>>;
    }

    impl GpuPersistentActorExt for System {
        fn start_persistent_gpu<H: PersistentHandle + 'static>(
            handle: Arc<H>,
            config: PersistentActorConfig,
        ) -> Addr<GpuPersistentActor<H>> {
            GpuPersistentActor::new(handle, config).start()
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
