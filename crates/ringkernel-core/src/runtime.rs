//! Runtime traits and types for kernel management.
//!
//! This module defines the core runtime abstraction that backends implement
//! to provide kernel lifecycle management, message passing, and monitoring.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::error::Result;
use crate::message::{MessageEnvelope, RingMessage};
use crate::telemetry::KernelMetrics;
use crate::types::KernelMode;

/// Unique kernel identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelId(pub String);

impl KernelId {
    /// Create a new kernel ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for KernelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for KernelId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for KernelId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Kernel lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelState {
    /// Kernel is created but not launched.
    Created,
    /// Kernel is launched and initializing.
    Launched,
    /// Kernel is active and processing messages.
    Active,
    /// Kernel is deactivated (paused).
    Deactivated,
    /// Kernel is terminating.
    Terminating,
    /// Kernel has terminated.
    Terminated,
}

impl KernelState {
    /// Check if kernel can be activated.
    pub fn can_activate(&self) -> bool {
        matches!(self, Self::Launched | Self::Deactivated)
    }

    /// Check if kernel can be deactivated.
    pub fn can_deactivate(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Check if kernel can be terminated.
    pub fn can_terminate(&self) -> bool {
        matches!(self, Self::Active | Self::Deactivated | Self::Launched)
    }

    /// Check if kernel is running (can process messages).
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Check if kernel is finished.
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Terminated)
    }
}

/// Kernel status including state and metrics.
#[derive(Debug, Clone)]
pub struct KernelStatus {
    /// Kernel identifier.
    pub id: KernelId,
    /// Current state.
    pub state: KernelState,
    /// Execution mode.
    pub mode: KernelMode,
    /// Messages in input queue.
    pub input_queue_depth: usize,
    /// Messages in output queue.
    pub output_queue_depth: usize,
    /// Total messages processed.
    pub messages_processed: u64,
    /// Uptime since launch.
    pub uptime: Duration,
}

/// GPU backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Backend {
    /// CPU backend (for testing).
    Cpu,
    /// NVIDIA CUDA backend.
    Cuda,
    /// Apple Metal backend.
    Metal,
    /// WebGPU cross-platform backend.
    Wgpu,
    /// Automatically select best available backend.
    #[default]
    Auto,
}

impl Backend {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Cpu => "CPU",
            Backend::Cuda => "CUDA",
            Backend::Metal => "Metal",
            Backend::Wgpu => "WebGPU",
            Backend::Auto => "Auto",
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Options for launching a kernel.
#[derive(Debug, Clone)]
pub struct LaunchOptions {
    /// Execution mode (persistent or event-driven).
    pub mode: KernelMode,
    /// Grid size (number of blocks).
    pub grid_size: u32,
    /// Block size (threads per block).
    pub block_size: u32,
    /// Input queue capacity.
    pub input_queue_capacity: usize,
    /// Output queue capacity.
    pub output_queue_capacity: usize,
    /// Shared memory size in bytes.
    pub shared_memory_size: usize,
    /// Whether to activate immediately after launch.
    pub auto_activate: bool,
    /// Enable cooperative groups for grid-wide synchronization.
    /// Requires GPU support for cooperative kernel launch.
    pub cooperative: bool,
}

impl Default for LaunchOptions {
    fn default() -> Self {
        Self {
            mode: KernelMode::Persistent,
            grid_size: 1,
            block_size: 256,
            input_queue_capacity: 1024,
            output_queue_capacity: 1024,
            shared_memory_size: 0,
            auto_activate: true,
            cooperative: false,
        }
    }
}

impl LaunchOptions {
    /// Create options for a single-block kernel.
    pub fn single_block(block_size: u32) -> Self {
        Self {
            block_size,
            ..Default::default()
        }
    }

    /// Create options for a multi-block kernel.
    pub fn multi_block(grid_size: u32, block_size: u32) -> Self {
        Self {
            grid_size,
            block_size,
            ..Default::default()
        }
    }

    /// Set execution mode.
    pub fn with_mode(mut self, mode: KernelMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set queue capacities.
    pub fn with_queue_capacity(mut self, capacity: usize) -> Self {
        self.input_queue_capacity = capacity;
        self.output_queue_capacity = capacity;
        self
    }

    /// Set shared memory size.
    pub fn with_shared_memory(mut self, size: usize) -> Self {
        self.shared_memory_size = size;
        self
    }

    /// Disable auto-activation.
    pub fn without_auto_activate(mut self) -> Self {
        self.auto_activate = false;
        self
    }

    /// Set the grid size (number of blocks).
    pub fn with_grid_size(mut self, grid_size: u32) -> Self {
        self.grid_size = grid_size;
        self
    }

    /// Set the block size (threads per block).
    pub fn with_block_size(mut self, block_size: u32) -> Self {
        self.block_size = block_size;
        self
    }

    /// Enable cooperative groups for grid-wide synchronization.
    ///
    /// When enabled, the kernel will be launched cooperatively, allowing
    /// all blocks to synchronize via `grid.sync()`. Requires GPU support
    /// and nvcc at build time.
    pub fn with_cooperative(mut self, cooperative: bool) -> Self {
        self.cooperative = cooperative;
        self
    }

    /// Set priority hint for kernel scheduling.
    ///
    /// Note: This is a hint for future use - currently ignored by backends.
    pub fn with_priority(self, _priority: u8) -> Self {
        // Priority hint stored for future scheduling use
        self
    }

    /// Set input queue capacity only.
    pub fn with_input_queue_capacity(mut self, capacity: usize) -> Self {
        self.input_queue_capacity = capacity;
        self
    }

    /// Set output queue capacity only.
    pub fn with_output_queue_capacity(mut self, capacity: usize) -> Self {
        self.output_queue_capacity = capacity;
        self
    }
}

/// Type-erased future for async operations.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Backend-agnostic runtime trait for kernel management.
///
/// This trait is implemented by each backend (CPU, CUDA, Metal, WebGPU)
/// to provide kernel lifecycle management and message passing.
#[async_trait]
pub trait RingKernelRuntime: Send + Sync {
    /// Get the backend type.
    fn backend(&self) -> Backend;

    /// Check if a specific backend is available.
    fn is_backend_available(&self, backend: Backend) -> bool;

    /// Launch a kernel.
    async fn launch(&self, kernel_id: &str, options: LaunchOptions) -> Result<KernelHandle>;

    /// Get a handle to an existing kernel.
    fn get_kernel(&self, kernel_id: &KernelId) -> Option<KernelHandle>;

    /// List all kernel IDs.
    fn list_kernels(&self) -> Vec<KernelId>;

    /// Get runtime metrics.
    fn metrics(&self) -> RuntimeMetrics;

    /// Shutdown the runtime and terminate all kernels.
    async fn shutdown(&self) -> Result<()>;
}

/// Handle to a launched kernel.
///
/// Provides an ergonomic API for interacting with a kernel.
#[derive(Clone)]
pub struct KernelHandle {
    /// Kernel identifier.
    id: KernelId,
    /// Inner implementation.
    inner: Arc<dyn KernelHandleInner>,
}

impl KernelHandle {
    /// Create a new kernel handle.
    pub fn new(id: KernelId, inner: Arc<dyn KernelHandleInner>) -> Self {
        Self { id, inner }
    }

    /// Get the kernel ID.
    pub fn id(&self) -> &KernelId {
        &self.id
    }

    /// Activate the kernel.
    pub async fn activate(&self) -> Result<()> {
        self.inner.activate().await
    }

    /// Deactivate the kernel.
    pub async fn deactivate(&self) -> Result<()> {
        self.inner.deactivate().await
    }

    /// Terminate the kernel.
    pub async fn terminate(&self) -> Result<()> {
        self.inner.terminate().await
    }

    /// Send a message to the kernel.
    pub async fn send<M: RingMessage>(&self, message: M) -> Result<()> {
        let envelope = MessageEnvelope::new(
            &message,
            0, // Host source
            self.inner.kernel_id_num(),
            self.inner.current_timestamp(),
        );
        self.inner.send_envelope(envelope).await
    }

    /// Send a raw envelope.
    pub async fn send_envelope(&self, envelope: MessageEnvelope) -> Result<()> {
        self.inner.send_envelope(envelope).await
    }

    /// Receive a message from the kernel.
    pub async fn receive(&self) -> Result<MessageEnvelope> {
        self.inner.receive().await
    }

    /// Receive a message with timeout.
    pub async fn receive_timeout(&self, timeout: Duration) -> Result<MessageEnvelope> {
        self.inner.receive_timeout(timeout).await
    }

    /// Try to receive a message (non-blocking).
    pub fn try_receive(&self) -> Result<MessageEnvelope> {
        self.inner.try_receive()
    }

    /// Send request and wait for response (call pattern).
    pub async fn call<M: RingMessage>(
        &self,
        message: M,
        timeout: Duration,
    ) -> Result<MessageEnvelope> {
        // Generate correlation ID
        let correlation = crate::message::CorrelationId::generate();

        // Create envelope with correlation
        let mut envelope = MessageEnvelope::new(
            &message,
            0,
            self.inner.kernel_id_num(),
            self.inner.current_timestamp(),
        );
        envelope.header.correlation_id = correlation;

        // Send and wait for correlated response
        self.inner.send_envelope(envelope).await?;
        self.inner.receive_correlated(correlation, timeout).await
    }

    /// Get kernel status.
    pub fn status(&self) -> KernelStatus {
        self.inner.status()
    }

    /// Get kernel metrics.
    pub fn metrics(&self) -> KernelMetrics {
        self.inner.metrics()
    }

    /// Wait for kernel to terminate.
    pub async fn wait(&self) -> Result<()> {
        self.inner.wait().await
    }

    /// Get the current kernel state.
    ///
    /// This is a convenience method that returns just the state from status().
    pub fn state(&self) -> KernelState {
        self.status().state
    }

    /// Suspend (deactivate) the kernel.
    ///
    /// This is an alias for `deactivate()` for more intuitive API usage.
    pub async fn suspend(&self) -> Result<()> {
        self.deactivate().await
    }

    /// Resume (activate) the kernel.
    ///
    /// This is an alias for `activate()` for more intuitive API usage.
    pub async fn resume(&self) -> Result<()> {
        self.activate().await
    }

    /// Check if the kernel is currently active.
    pub fn is_active(&self) -> bool {
        self.state() == KernelState::Active
    }

    /// Check if the kernel has terminated.
    pub fn is_terminated(&self) -> bool {
        self.state() == KernelState::Terminated
    }
}

impl std::fmt::Debug for KernelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelHandle")
            .field("id", &self.id)
            .finish()
    }
}

/// Inner trait for kernel handle implementation.
///
/// This is implemented by each backend to provide the actual functionality.
#[async_trait]
pub trait KernelHandleInner: Send + Sync {
    /// Get numeric kernel ID.
    fn kernel_id_num(&self) -> u64;

    /// Get current timestamp.
    fn current_timestamp(&self) -> crate::hlc::HlcTimestamp;

    /// Activate kernel.
    async fn activate(&self) -> Result<()>;

    /// Deactivate kernel.
    async fn deactivate(&self) -> Result<()>;

    /// Terminate kernel.
    async fn terminate(&self) -> Result<()>;

    /// Send message envelope.
    async fn send_envelope(&self, envelope: MessageEnvelope) -> Result<()>;

    /// Receive message.
    async fn receive(&self) -> Result<MessageEnvelope>;

    /// Receive with timeout.
    async fn receive_timeout(&self, timeout: Duration) -> Result<MessageEnvelope>;

    /// Try receive (non-blocking).
    fn try_receive(&self) -> Result<MessageEnvelope>;

    /// Receive correlated response.
    async fn receive_correlated(
        &self,
        correlation: crate::message::CorrelationId,
        timeout: Duration,
    ) -> Result<MessageEnvelope>;

    /// Get status.
    fn status(&self) -> KernelStatus;

    /// Get metrics.
    fn metrics(&self) -> KernelMetrics;

    /// Wait for termination.
    async fn wait(&self) -> Result<()>;
}

/// Runtime-level metrics.
#[derive(Debug, Clone, Default)]
pub struct RuntimeMetrics {
    /// Number of active kernels.
    pub active_kernels: usize,
    /// Total kernels launched.
    pub total_launched: u64,
    /// Total messages sent.
    pub messages_sent: u64,
    /// Total messages received.
    pub messages_received: u64,
    /// GPU memory used (bytes).
    pub gpu_memory_used: u64,
    /// Host memory used (bytes).
    pub host_memory_used: u64,
}

/// Builder for creating a runtime instance.
#[derive(Debug, Clone)]
pub struct RuntimeBuilder {
    /// Selected backend.
    pub backend: Backend,
    /// Device index (for multi-GPU).
    pub device_index: usize,
    /// Enable debug mode.
    pub debug: bool,
    /// Enable profiling.
    pub profiling: bool,
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            device_index: 0,
            debug: false,
            profiling: false,
        }
    }
}

impl RuntimeBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Set device index.
    pub fn device(mut self, index: usize) -> Self {
        self.device_index = index;
        self
    }

    /// Enable debug mode.
    pub fn debug(mut self, enable: bool) -> Self {
        self.debug = enable;
        self
    }

    /// Enable profiling.
    pub fn profiling(mut self, enable: bool) -> Self {
        self.profiling = enable;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_state_transitions() {
        assert!(KernelState::Launched.can_activate());
        assert!(KernelState::Deactivated.can_activate());
        assert!(!KernelState::Active.can_activate());
        assert!(!KernelState::Terminated.can_activate());

        assert!(KernelState::Active.can_deactivate());
        assert!(!KernelState::Launched.can_deactivate());

        assert!(KernelState::Active.can_terminate());
        assert!(KernelState::Deactivated.can_terminate());
        assert!(!KernelState::Terminated.can_terminate());
    }

    #[test]
    fn test_launch_options_builder() {
        let opts = LaunchOptions::multi_block(4, 128)
            .with_mode(KernelMode::EventDriven)
            .with_queue_capacity(2048)
            .with_shared_memory(4096)
            .without_auto_activate();

        assert_eq!(opts.grid_size, 4);
        assert_eq!(opts.block_size, 128);
        assert_eq!(opts.mode, KernelMode::EventDriven);
        assert_eq!(opts.input_queue_capacity, 2048);
        assert_eq!(opts.shared_memory_size, 4096);
        assert!(!opts.auto_activate);
    }

    #[test]
    fn test_kernel_id() {
        let id1 = KernelId::new("test_kernel");
        let id2: KernelId = "test_kernel".into();
        assert_eq!(id1, id2);
        assert_eq!(id1.as_str(), "test_kernel");
    }

    #[test]
    fn test_backend_name() {
        assert_eq!(Backend::Cpu.name(), "CPU");
        assert_eq!(Backend::Cuda.name(), "CUDA");
        assert_eq!(Backend::Metal.name(), "Metal");
        assert_eq!(Backend::Wgpu.name(), "WebGPU");
    }
}
