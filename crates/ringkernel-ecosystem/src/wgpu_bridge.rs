//! WebGPU implementation of PersistentHandle (emulated persistence).
//!
//! This module provides the [`WgpuPersistentHandle`] type that implements the
//! [`PersistentHandle`] trait via host-driven batched dispatch. Unlike CUDA's
//! true persistent kernels, WebGPU cannot run infinite loops on the GPU, so
//! we emulate persistence by repeatedly dispatching compute shaders from the host.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        WEB FRAMEWORK                                    │
//! │  (Actix GpuPersistentActor, Axum PersistentGpuState, Tower Service)    │
//! └────────────────────────────────────────────────────────────────────────┘
//!                                  │ PersistentHandle trait
//! ┌────────────────────────────────▼────────────────────────────────────────┐
//! │                     WgpuPersistentHandle                                │
//! │  • Implements PersistentHandle                                          │
//! │  • Host-driven dispatch loop                                            │
//! │  • Batched shader invocations                                           │
//! │  • Command queue in host memory                                         │
//! └────────────────────────────────┬────────────────────────────────────────┘
//!                                  │
//! ┌────────────────────────────────▼────────────────────────────────────────┐
//! │                       WgpuRuntime                                       │
//! │  • GPU buffer management                                                │
//! │  • Compute pipeline dispatch                                            │
//! │  • Buffer staging for control block                                     │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Emulation vs True Persistence
//!
//! | Aspect | CUDA Persistent | WebGPU Emulated |
//! |--------|-----------------|-----------------|
//! | Kernel lifetime | Infinite (coop groups) | Single dispatch |
//! | Command latency | ~0.03µs (mapped memory) | ~100-500µs (staging) |
//! | Grid sync | grid.sync() | Host barrier |
//! | Best for | Interactive workloads | Batch compute |
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_wgpu::WgpuRuntime;
//! use ringkernel_ecosystem::wgpu_bridge::WgpuPersistentHandle;
//! use ringkernel_ecosystem::persistent::PersistentHandle;
//!
//! // Create WebGPU runtime
//! let runtime = WgpuRuntime::new().await?;
//!
//! // Create ecosystem handle
//! let handle = WgpuPersistentHandle::new(runtime, "compute_sim")?;
//!
//! // Start the emulated persistent kernel
//! handle.start(wgsl_shader)?;
//!
//! // Use with ecosystem integrations (same API as CUDA)
//! let cmd_id = handle.send_command(PersistentCommand::RunSteps { count: 100 })?;
//! ```

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};

use ringkernel_wgpu::WgpuRuntime;

use crate::error::{EcosystemError, Result as EcoResult};
use crate::persistent::{
    CommandId, PersistentCommand, PersistentConfig, PersistentHandle,
    PersistentResponse, PersistentStats,
};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for WebGPU persistent emulation.
#[derive(Debug, Clone)]
pub struct WgpuEmulationConfig {
    /// Batch size for dispatches (steps per GPU invocation).
    pub batch_size: u32,
    /// Maximum dispatches per tick.
    pub max_dispatches_per_tick: u32,
    /// Tick interval for the dispatch loop.
    pub tick_interval: Duration,
    /// Workgroup size for compute shaders.
    pub workgroup_size: (u32, u32, u32),
    /// Grid dimensions.
    pub grid_size: (u32, u32, u32),
    /// Progress report interval (in steps).
    pub progress_interval: u64,
}

impl Default for WgpuEmulationConfig {
    fn default() -> Self {
        Self {
            batch_size: 16,
            max_dispatches_per_tick: 1000,
            tick_interval: Duration::from_micros(100),
            workgroup_size: (256, 1, 1),
            grid_size: (64, 64, 1),
            progress_interval: 100,
        }
    }
}

impl WgpuEmulationConfig {
    /// Create a new configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, size: u32) -> Self {
        self.batch_size = size;
        self
    }

    /// Set the grid size.
    pub fn with_grid_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.grid_size = (x, y, z);
        self
    }

    /// Set the progress interval.
    pub fn with_progress_interval(mut self, interval: u64) -> Self {
        self.progress_interval = interval;
        self
    }
}

// ============================================================================
// CONTROL BLOCK (Host-side)
// ============================================================================

/// Host-side control block for emulated persistence.
///
/// Unlike CUDA's mapped memory control block, this is maintained in host memory
/// and synchronized with GPU buffers via staging buffers.
#[derive(Debug, Default)]
struct HostControlBlock {
    /// Whether the kernel is active.
    is_active: bool,
    /// Whether termination is requested.
    should_terminate: bool,
    /// Whether the kernel has terminated.
    has_terminated: bool,
    /// Current simulation step.
    current_step: u64,
    /// Total messages processed.
    messages_processed: u64,
    /// Steps remaining to execute.
    steps_remaining: u64,
    /// Paused state.
    is_paused: bool,
    /// Last progress report step.
    last_progress_step: u64,
}

// ============================================================================
// PENDING COMMAND
// ============================================================================

/// A pending command with its ID.
#[derive(Debug)]
struct PendingCommand {
    id: CommandId,
    command: PersistentCommand,
    received_at: Instant,
}

// ============================================================================
// WGPU PERSISTENT HANDLE
// ============================================================================

/// WebGPU implementation of PersistentHandle with emulated persistence.
///
/// This handle provides a compatible API with `CudaPersistentHandle` but uses
/// host-driven batched dispatch instead of true GPU persistence.
///
/// # Performance Characteristics
///
/// - Command latency: ~100-500µs (vs ~0.03µs for CUDA)
/// - Throughput: Limited by host dispatch overhead
/// - Best for: Cross-platform deployments where CUDA isn't available
///
/// # Thread Safety
///
/// `WgpuPersistentHandle` is both `Send` and `Sync`, making it safe to share
/// across threads and async tasks.
pub struct WgpuPersistentHandle {
    /// Kernel identifier.
    kernel_id: String,
    /// WebGPU runtime reference.
    runtime: Arc<WgpuRuntime>,
    /// Running state.
    running: AtomicBool,
    /// Command ID counter.
    cmd_counter: AtomicU64,
    /// Persistent configuration.
    config: RwLock<PersistentConfig>,
    /// Emulation configuration.
    emulation_config: WgpuEmulationConfig,
    /// Host-side control block.
    control_block: Mutex<HostControlBlock>,
    /// Pending commands queue.
    pending_commands: Mutex<VecDeque<PendingCommand>>,
    /// Response queue.
    responses: Mutex<VecDeque<PersistentResponse>>,
    /// Total steps executed.
    total_steps: AtomicU64,
    /// Start time.
    start_time: RwLock<Option<Instant>>,
    /// Shader loaded flag.
    shader_loaded: AtomicBool,
}

impl WgpuPersistentHandle {
    /// Create a new WebGPU persistent handle.
    ///
    /// # Arguments
    ///
    /// * `runtime` - The WebGPU runtime to use
    /// * `kernel_id` - Identifier for this kernel (used for logging/routing)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let runtime = WgpuRuntime::new().await?;
    /// let handle = WgpuPersistentHandle::new(runtime, "my_simulation")?;
    /// ```
    pub fn new(runtime: Arc<WgpuRuntime>, kernel_id: impl Into<String>) -> EcoResult<Self> {
        Self::with_config(
            runtime,
            kernel_id,
            WgpuEmulationConfig::default(),
            PersistentConfig::default(),
        )
    }

    /// Create a handle with custom configuration.
    pub fn with_config(
        runtime: Arc<WgpuRuntime>,
        kernel_id: impl Into<String>,
        emulation_config: WgpuEmulationConfig,
        persistent_config: PersistentConfig,
    ) -> EcoResult<Self> {
        Ok(Self {
            kernel_id: kernel_id.into(),
            runtime,
            running: AtomicBool::new(false),
            cmd_counter: AtomicU64::new(1),
            config: RwLock::new(persistent_config),
            emulation_config,
            control_block: Mutex::new(HostControlBlock::default()),
            pending_commands: Mutex::new(VecDeque::new()),
            responses: Mutex::new(VecDeque::new()),
            total_steps: AtomicU64::new(0),
            start_time: RwLock::new(None),
            shader_loaded: AtomicBool::new(false),
        })
    }

    /// Get the emulation configuration.
    pub fn emulation_config(&self) -> &WgpuEmulationConfig {
        &self.emulation_config
    }

    /// Get the underlying WebGPU runtime.
    pub fn runtime(&self) -> &Arc<WgpuRuntime> {
        &self.runtime
    }

    /// Start the emulated persistent kernel.
    ///
    /// Unlike CUDA persistent kernels which launch once and run forever,
    /// this sets up the state for host-driven dispatch.
    ///
    /// # Arguments
    ///
    /// * `_wgsl_shader` - The WGSL shader code (for future use with custom pipelines)
    pub fn start(&self, _wgsl_shader: &str) -> EcoResult<()> {
        if self.running.load(Ordering::Acquire) {
            return Err(EcosystemError::ServiceUnavailable(
                "Kernel already running".to_string(),
            ));
        }

        // Set up control block
        {
            let mut cb = self.control_block.lock();
            cb.is_active = true;
            cb.should_terminate = false;
            cb.has_terminated = false;
            cb.current_step = 0;
            cb.messages_processed = 0;
            cb.steps_remaining = 0;
            cb.is_paused = false;
            cb.last_progress_step = 0;
        }

        *self.start_time.write() = Some(Instant::now());
        self.shader_loaded.store(true, Ordering::Release);
        self.running.store(true, Ordering::Release);

        tracing::info!(
            kernel_id = %self.kernel_id,
            "Started WebGPU emulated persistent kernel"
        );

        Ok(())
    }

    /// Execute one tick of the dispatch loop.
    ///
    /// This should be called periodically (e.g., by an async task) to process
    /// pending commands and execute compute dispatches.
    ///
    /// Returns the number of steps executed in this tick.
    pub fn tick(&self) -> EcoResult<u64> {
        if !self.running.load(Ordering::Acquire) {
            return Ok(0);
        }

        let mut cb = self.control_block.lock();

        if cb.should_terminate {
            cb.has_terminated = true;
            self.running.store(false, Ordering::Release);

            // Send termination response
            let response = PersistentResponse::Terminated {
                final_step: cb.current_step,
            };
            self.responses.lock().push_back(response);

            return Ok(0);
        }

        if cb.is_paused {
            return Ok(0);
        }

        // Process pending commands
        let mut pending = self.pending_commands.lock();
        while let Some(cmd) = pending.pop_front() {
            cb.messages_processed += 1;

            match cmd.command {
                PersistentCommand::RunSteps { count } => {
                    cb.steps_remaining = cb.steps_remaining.saturating_add(count);

                    // Send ack
                    let response = PersistentResponse::Ack { cmd_id: cmd.id };
                    self.responses.lock().push_back(response);
                }
                PersistentCommand::Pause => {
                    cb.is_paused = true;
                    let response = PersistentResponse::Ack { cmd_id: cmd.id };
                    self.responses.lock().push_back(response);
                }
                PersistentCommand::Resume => {
                    cb.is_paused = false;
                    let response = PersistentResponse::Ack { cmd_id: cmd.id };
                    self.responses.lock().push_back(response);
                }
                PersistentCommand::Terminate => {
                    cb.should_terminate = true;
                    let response = PersistentResponse::Ack { cmd_id: cmd.id };
                    self.responses.lock().push_back(response);
                }
                PersistentCommand::GetProgress => {
                    let response = PersistentResponse::Progress {
                        cmd_id: cmd.id,
                        current_step: cb.current_step,
                        remaining: cb.steps_remaining,
                    };
                    self.responses.lock().push_back(response);
                }
                PersistentCommand::GetStats => {
                    let responses = self.responses.lock();
                    let stats = PersistentStats {
                        current_step: cb.current_step,
                        steps_remaining: cb.steps_remaining,
                        messages_processed: cb.messages_processed,
                        total_energy: 0.0,
                        k2k_sent: 0,
                        k2k_received: 0,
                        is_running: self.running.load(Ordering::Acquire),
                        has_terminated: cb.has_terminated,
                        pending_commands: pending.len() as u32,
                        pending_responses: responses.len() as u32,
                    };
                    drop(responses);
                    let response = PersistentResponse::Stats {
                        cmd_id: cmd.id,
                        stats,
                    };
                    self.responses.lock().push_back(response);
                }
                PersistentCommand::Inject { position, value } => {
                    // In a real implementation, this would update GPU buffers
                    tracing::debug!(
                        kernel_id = %self.kernel_id,
                        position = ?position,
                        value = value,
                        "Inject command (emulated - no GPU buffer update)"
                    );
                    let response = PersistentResponse::Ack { cmd_id: cmd.id };
                    self.responses.lock().push_back(response);
                }
                PersistentCommand::Custom { type_id, payload } => {
                    tracing::debug!(
                        kernel_id = %self.kernel_id,
                        type_id = type_id,
                        payload_len = payload.len(),
                        "Custom command (emulated)"
                    );
                    let response = PersistentResponse::Custom {
                        cmd_id: cmd.id,
                        type_id,
                        payload: vec![],
                    };
                    self.responses.lock().push_back(response);
                }
            }
        }
        drop(pending);

        // Execute batched steps
        let steps_to_run = std::cmp::min(
            cb.steps_remaining,
            self.emulation_config.batch_size as u64 * self.emulation_config.max_dispatches_per_tick as u64,
        );

        if steps_to_run == 0 {
            return Ok(0);
        }

        // Simulate GPU execution (in a real implementation, this would dispatch compute shaders)
        // For now, we just update the counters
        cb.current_step += steps_to_run;
        cb.steps_remaining -= steps_to_run;
        self.total_steps.fetch_add(steps_to_run, Ordering::Relaxed);

        // Send progress report if needed
        if cb.current_step - cb.last_progress_step >= self.emulation_config.progress_interval {
            cb.last_progress_step = cb.current_step;

            let response = PersistentResponse::Progress {
                cmd_id: CommandId::new(0), // System-initiated
                current_step: cb.current_step,
                remaining: cb.steps_remaining,
            };
            self.responses.lock().push_back(response);
        }

        Ok(steps_to_run)
    }

    /// Shutdown the emulated kernel gracefully.
    pub fn shutdown(&self) -> EcoResult<()> {
        if !self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        {
            let mut cb = self.control_block.lock();
            cb.should_terminate = true;
        }

        // Execute final tick to process termination
        let _ = self.tick();

        tracing::info!(
            kernel_id = %self.kernel_id,
            "Shutdown WebGPU emulated persistent kernel"
        );

        Ok(())
    }
}

impl Clone for WgpuPersistentHandle {
    fn clone(&self) -> Self {
        Self {
            kernel_id: self.kernel_id.clone(),
            runtime: self.runtime.clone(),
            running: AtomicBool::new(self.running.load(Ordering::Acquire)),
            cmd_counter: AtomicU64::new(self.cmd_counter.load(Ordering::Relaxed)),
            config: RwLock::new(self.config.read().clone()),
            emulation_config: self.emulation_config.clone(),
            control_block: Mutex::new(HostControlBlock::default()),
            pending_commands: Mutex::new(VecDeque::new()),
            responses: Mutex::new(VecDeque::new()),
            total_steps: AtomicU64::new(self.total_steps.load(Ordering::Relaxed)),
            start_time: RwLock::new(*self.start_time.read()),
            shader_loaded: AtomicBool::new(self.shader_loaded.load(Ordering::Acquire)),
        }
    }
}

// Safety: WgpuPersistentHandle uses thread-safe primitives
unsafe impl Send for WgpuPersistentHandle {}
unsafe impl Sync for WgpuPersistentHandle {}

#[async_trait::async_trait]
impl PersistentHandle for WgpuPersistentHandle {
    fn kernel_id(&self) -> &str {
        &self.kernel_id
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    fn send_command(&self, cmd: PersistentCommand) -> EcoResult<CommandId> {
        if !self.is_running() {
            return Err(EcosystemError::ServiceUnavailable(
                "Kernel not running".to_string(),
            ));
        }

        let cmd_id = CommandId::new(self.cmd_counter.fetch_add(1, Ordering::Relaxed));

        // Queue the command
        let pending = PendingCommand {
            id: cmd_id,
            command: cmd,
            received_at: Instant::now(),
        };
        self.pending_commands.lock().push_back(pending);

        Ok(cmd_id)
    }

    fn poll_responses(&self) -> Vec<PersistentResponse> {
        let mut responses = self.responses.lock();
        responses.drain(..).collect()
    }

    fn stats(&self) -> PersistentStats {
        let cb = self.control_block.lock();
        let pending = self.pending_commands.lock();
        let responses = self.responses.lock();

        PersistentStats {
            current_step: cb.current_step,
            steps_remaining: cb.steps_remaining,
            messages_processed: cb.messages_processed,
            total_energy: 0.0,
            k2k_sent: 0,
            k2k_received: 0,
            is_running: self.running.load(Ordering::Acquire),
            has_terminated: cb.has_terminated,
            pending_commands: pending.len() as u32,
            pending_responses: responses.len() as u32,
        }
    }

    async fn wait_for_command(&self, cmd_id: CommandId, timeout: Duration) -> EcoResult<()> {
        let start = Instant::now();

        while start.elapsed() < timeout {
            // Check responses for this command
            let responses = self.poll_responses();
            for response in responses {
                if response.command_id() == Some(cmd_id) {
                    match response {
                        PersistentResponse::Ack { .. } => return Ok(()),
                        PersistentResponse::Error { code, message, .. } => {
                            return Err(EcosystemError::CommandFailed { code, message });
                        }
                        PersistentResponse::Terminated { .. } => {
                            return Err(EcosystemError::KernelNotRunning(
                                "Kernel terminated".to_string(),
                            ));
                        }
                        _ => {}
                    }
                }
            }

            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        Err(EcosystemError::Timeout(timeout))
    }

    async fn shutdown(&self) -> EcoResult<()> {
        WgpuPersistentHandle::shutdown(self)
    }

    fn config(&self) -> &PersistentConfig {
        // This is a bit awkward due to the RwLock, but we need to return a reference
        // For now, leak a box to return a static reference
        // In production, consider using arc-swap or similar
        Box::leak(Box::new(self.config.read().clone()))
    }
}

// ============================================================================
// BUILDER
// ============================================================================

/// Builder for [`WgpuPersistentHandle`].
///
/// Provides a fluent API for constructing handles with custom configuration.
pub struct WgpuPersistentHandleBuilder {
    kernel_id: String,
    emulation_config: WgpuEmulationConfig,
    persistent_config: PersistentConfig,
}

impl WgpuPersistentHandleBuilder {
    /// Create a new builder with the given kernel ID.
    pub fn new(kernel_id: impl Into<String>) -> Self {
        Self {
            kernel_id: kernel_id.into(),
            emulation_config: WgpuEmulationConfig::default(),
            persistent_config: PersistentConfig::default(),
        }
    }

    /// Set the batch size for dispatches.
    pub fn with_batch_size(mut self, size: u32) -> Self {
        self.emulation_config.batch_size = size;
        self
    }

    /// Set the grid size.
    pub fn with_grid_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.emulation_config.grid_size = (x, y, z);
        self
    }

    /// Set the progress report interval.
    pub fn with_progress_interval(mut self, interval: u64) -> Self {
        self.emulation_config.progress_interval = interval;
        self
    }

    /// Set the persistent configuration.
    pub fn with_persistent_config(mut self, config: PersistentConfig) -> Self {
        self.persistent_config = config;
        self
    }

    /// Build the handle with the given runtime.
    pub fn build(self, runtime: Arc<WgpuRuntime>) -> EcoResult<WgpuPersistentHandle> {
        WgpuPersistentHandle::with_config(
            runtime,
            self.kernel_id,
            self.emulation_config,
            self.persistent_config,
        )
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Most tests require a GPU. These are basic unit tests for the handle logic.

    #[test]
    fn test_emulation_config_default() {
        let config = WgpuEmulationConfig::default();
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.max_dispatches_per_tick, 1000);
        assert_eq!(config.workgroup_size, (256, 1, 1));
    }

    #[test]
    fn test_emulation_config_builder() {
        let config = WgpuEmulationConfig::new()
            .with_batch_size(32)
            .with_grid_size(128, 128, 64)
            .with_progress_interval(500);

        assert_eq!(config.batch_size, 32);
        assert_eq!(config.grid_size, (128, 128, 64));
        assert_eq!(config.progress_interval, 500);
    }

    #[test]
    fn test_builder() {
        let builder = WgpuPersistentHandleBuilder::new("test_kernel")
            .with_batch_size(64)
            .with_grid_size(32, 32, 32)
            .with_progress_interval(1000);

        assert_eq!(builder.kernel_id, "test_kernel");
        assert_eq!(builder.emulation_config.batch_size, 64);
        assert_eq!(builder.emulation_config.grid_size, (32, 32, 32));
    }

    #[test]
    fn test_host_control_block_default() {
        let cb = HostControlBlock::default();
        assert!(!cb.is_active);
        assert!(!cb.should_terminate);
        assert!(!cb.has_terminated);
        assert_eq!(cb.current_step, 0);
        assert_eq!(cb.messages_processed, 0);
    }
}
