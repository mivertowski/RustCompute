//! Persistent GPU kernel integration for RingKernel.
//!
//! This module provides abstract traits and types for persistent kernel operations,
//! enabling ecosystem integrations (Actix, Axum, Tower, gRPC) to leverage the
//! 11,327x faster command injection latency of persistent GPU actors.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
//! │ Web Framework   │────>│ PersistentHandle │────>│ GPU Kernel      │
//! │ (Actix/Axum)    │     │ (trait)          │     │ (CUDA/Metal)    │
//! └─────────────────┘     └──────────────────┘     └─────────────────┘
//!        │                       │                        │
//!   HTTP/WS Request        send_command()          Mapped Memory
//!        │                  (~0.03µs)              Command Queue
//!        ▼                       │                        │
//!   JSON Response          poll_responses()         K2H Response
//! ```
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ecosystem::persistent::{PersistentHandle, PersistentCommand};
//!
//! async fn run_simulation<H: PersistentHandle>(handle: &H) {
//!     // Send command with ~0.03µs latency (vs ~317µs traditional)
//!     let cmd_id = handle.send_command(PersistentCommand::RunSteps { count: 100 })?;
//!
//!     // Poll for responses
//!     for response in handle.poll_responses() {
//!         println!("Got response: {:?}", response);
//!     }
//! }
//! ```

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::Result;

#[cfg(test)]
use crate::error::EcosystemError;

// ============================================================================
// COMMAND IDENTIFIERS
// ============================================================================

/// Unique identifier for tracking commands through the system.
///
/// Command IDs are monotonically increasing and can be used to correlate
/// responses with their originating commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CommandId(pub u64);

impl CommandId {
    /// Create a new command ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for CommandId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cmd-{}", self.0)
    }
}

impl From<u64> for CommandId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

// ============================================================================
// COMMANDS
// ============================================================================

/// Commands that can be sent to a persistent GPU kernel.
///
/// These commands are injected into the kernel's H2K (host-to-kernel) queue
/// with minimal latency (~0.03µs), allowing for real-time interactive control.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersistentCommand {
    /// Run N processing/simulation steps.
    ///
    /// The kernel will execute the specified number of steps and send
    /// progress updates according to its configured interval.
    RunSteps {
        /// Number of steps to execute.
        count: u64,
    },

    /// Pause kernel execution.
    ///
    /// The kernel remains resident on the GPU but stops processing.
    /// Use `Resume` to continue.
    Pause,

    /// Resume kernel execution after a pause.
    Resume,

    /// Inject a value at a specific position.
    ///
    /// Commonly used for injecting impulses in simulations.
    Inject {
        /// 3D position (x, y, z).
        position: (u32, u32, u32),
        /// Value to inject (e.g., amplitude).
        value: f32,
    },

    /// Request a progress update.
    ///
    /// The kernel will respond with current step count and remaining steps.
    GetProgress,

    /// Request current statistics.
    GetStats,

    /// Terminate the kernel gracefully.
    ///
    /// The kernel will complete any in-progress work and shut down.
    Terminate,

    /// Custom command with application-specific payload.
    ///
    /// Use this for domain-specific commands not covered by standard types.
    Custom {
        /// Command type identifier.
        type_id: u32,
        /// Serialized payload data.
        payload: Vec<u8>,
    },
}

impl PersistentCommand {
    /// Create a RunSteps command.
    pub fn run_steps(count: u64) -> Self {
        Self::RunSteps { count }
    }

    /// Create an Inject command.
    pub fn inject(x: u32, y: u32, z: u32, value: f32) -> Self {
        Self::Inject {
            position: (x, y, z),
            value,
        }
    }

    /// Create a custom command.
    pub fn custom(type_id: u32, payload: Vec<u8>) -> Self {
        Self::Custom { type_id, payload }
    }
}

// ============================================================================
// RESPONSES
// ============================================================================

/// Responses from a persistent GPU kernel.
///
/// These are polled from the kernel's K2H (kernel-to-host) queue and
/// provide feedback on command execution, progress, and errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersistentResponse {
    /// Command was acknowledged and queued for execution.
    Ack {
        /// ID of the acknowledged command.
        cmd_id: CommandId,
    },

    /// Progress update during step execution.
    Progress {
        /// ID of the RunSteps command.
        cmd_id: CommandId,
        /// Current step number.
        current_step: u64,
        /// Steps remaining.
        remaining: u64,
    },

    /// Statistics snapshot.
    Stats {
        /// ID of the GetStats command.
        cmd_id: CommandId,
        /// Current statistics.
        stats: PersistentStats,
    },

    /// Error occurred during command execution.
    Error {
        /// ID of the failed command.
        cmd_id: CommandId,
        /// Error code.
        code: u32,
        /// Human-readable error message.
        message: String,
    },

    /// Kernel has terminated.
    Terminated {
        /// Final step count before termination.
        final_step: u64,
    },

    /// Custom response with application-specific payload.
    Custom {
        /// ID of the originating command.
        cmd_id: CommandId,
        /// Response type identifier.
        type_id: u32,
        /// Serialized payload data.
        payload: Vec<u8>,
    },
}

impl PersistentResponse {
    /// Get the command ID associated with this response, if any.
    pub fn command_id(&self) -> Option<CommandId> {
        match self {
            Self::Ack { cmd_id }
            | Self::Progress { cmd_id, .. }
            | Self::Stats { cmd_id, .. }
            | Self::Error { cmd_id, .. }
            | Self::Custom { cmd_id, .. } => Some(*cmd_id),
            Self::Terminated { .. } => None,
        }
    }

    /// Check if this is an error response.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Check if this indicates termination.
    pub fn is_terminated(&self) -> bool {
        matches!(self, Self::Terminated { .. })
    }
}

// ============================================================================
// STATISTICS
// ============================================================================

/// Statistics from a persistent GPU kernel.
///
/// These statistics provide visibility into kernel performance and state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistentStats {
    /// Current step/iteration number.
    pub current_step: u64,

    /// Steps remaining in current batch (0 if idle).
    pub steps_remaining: u64,

    /// Total messages processed by the kernel.
    pub messages_processed: u64,

    /// Total energy/value accumulated (simulation-specific).
    pub total_energy: f32,

    /// K2K messages sent to other kernels.
    pub k2k_sent: u64,

    /// K2K messages received from other kernels.
    pub k2k_received: u64,

    /// Whether the kernel is currently running.
    pub is_running: bool,

    /// Whether the kernel has terminated.
    pub has_terminated: bool,

    /// Commands pending in the H2K queue.
    pub pending_commands: u32,

    /// Responses pending in the K2H queue.
    pub pending_responses: u32,
}

impl PersistentStats {
    /// Check if the kernel is idle (no steps remaining).
    pub fn is_idle(&self) -> bool {
        self.is_running && self.steps_remaining == 0
    }

    /// Check if the kernel is actively processing.
    pub fn is_processing(&self) -> bool {
        self.is_running && self.steps_remaining > 0
    }
}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for persistent kernel integration.
#[derive(Debug, Clone)]
pub struct PersistentConfig {
    /// H2K (host-to-kernel) queue capacity. Must be power of 2.
    pub h2k_capacity: usize,

    /// K2H (kernel-to-host) queue capacity. Must be power of 2.
    pub k2h_capacity: usize,

    /// Progress report interval in steps. 0 = no automatic progress.
    pub progress_interval: u64,

    /// Polling interval for responses when waiting.
    pub poll_interval: Duration,

    /// Default timeout for command completion.
    pub command_timeout: Duration,

    /// Maximum number of in-flight commands.
    pub max_in_flight: usize,
}

impl Default for PersistentConfig {
    fn default() -> Self {
        Self {
            h2k_capacity: 256,
            k2h_capacity: 256,
            progress_interval: 100,
            poll_interval: Duration::from_micros(100),
            command_timeout: Duration::from_secs(30),
            max_in_flight: 1000,
        }
    }
}

impl PersistentConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set H2K queue capacity.
    pub fn with_h2k_capacity(mut self, capacity: usize) -> Self {
        self.h2k_capacity = capacity.next_power_of_two();
        self
    }

    /// Set K2H queue capacity.
    pub fn with_k2h_capacity(mut self, capacity: usize) -> Self {
        self.k2h_capacity = capacity.next_power_of_two();
        self
    }

    /// Set progress reporting interval.
    pub fn with_progress_interval(mut self, interval: u64) -> Self {
        self.progress_interval = interval;
        self
    }

    /// Set polling interval.
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Set command timeout.
    pub fn with_command_timeout(mut self, timeout: Duration) -> Self {
        self.command_timeout = timeout;
        self
    }

    /// Set maximum in-flight commands.
    pub fn with_max_in_flight(mut self, max: usize) -> Self {
        self.max_in_flight = max;
        self
    }
}

// ============================================================================
// PERSISTENT HANDLE TRAIT
// ============================================================================

/// Abstract handle for persistent kernel operations.
///
/// This trait abstracts the persistent kernel infrastructure, allowing
/// ecosystem integrations to work with any backend that supports persistent
/// execution (CUDA now, potentially Metal/WebGPU in future).
///
/// # Performance
///
/// The key advantage of persistent kernels is command injection latency:
/// - Traditional kernel launch: ~317µs per command
/// - Persistent command injection: ~0.03µs per command (11,327x faster)
///
/// This makes persistent kernels ideal for:
/// - Real-time interactive applications (60+ FPS)
/// - High-frequency command streams
/// - Dynamic parameter adjustment
///
/// # Example
///
/// ```ignore
/// use ringkernel_ecosystem::persistent::{PersistentHandle, PersistentCommand};
///
/// async fn interactive_loop<H: PersistentHandle>(handle: &H) -> Result<()> {
///     // Start simulation
///     handle.send_command(PersistentCommand::RunSteps { count: 1000 })?;
///
///     loop {
///         // Poll for updates (non-blocking)
///         for response in handle.poll_responses() {
///             match response {
///                 PersistentResponse::Progress { current_step, .. } => {
///                     println!("Step: {}", current_step);
///                 }
///                 PersistentResponse::Terminated { .. } => break,
///                 _ => {}
///             }
///         }
///
///         // Inject impulse on user input
///         if user_clicked() {
///             handle.send_command(PersistentCommand::inject(10, 10, 10, 1.0))?;
///         }
///
///         tokio::time::sleep(Duration::from_millis(16)).await; // 60 FPS
///     }
///     Ok(())
/// }
/// ```
#[async_trait::async_trait]
pub trait PersistentHandle: Send + Sync + 'static {
    /// Get the kernel identifier.
    fn kernel_id(&self) -> &str;

    /// Check if the kernel is currently running.
    fn is_running(&self) -> bool;

    /// Send a command to the kernel.
    ///
    /// This is non-blocking and returns immediately after queuing the command.
    /// The command will be processed asynchronously by the GPU kernel.
    ///
    /// # Returns
    ///
    /// Returns a `CommandId` that can be used to track the command's execution
    /// via `poll_responses()` or `wait_for_command()`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The kernel is not running
    /// - The H2K queue is full
    /// - The command is invalid
    fn send_command(&self, cmd: PersistentCommand) -> Result<CommandId>;

    /// Poll for responses from the kernel.
    ///
    /// This is non-blocking and returns all available responses from the
    /// K2H queue. Call this periodically to receive progress updates,
    /// acknowledgments, and errors.
    ///
    /// # Returns
    ///
    /// A vector of responses (may be empty if no responses are available).
    fn poll_responses(&self) -> Vec<PersistentResponse>;

    /// Get current kernel statistics.
    ///
    /// This reads from the control block and does not require a round-trip
    /// to the kernel's message queue.
    fn stats(&self) -> PersistentStats;

    /// Wait for a specific command to complete.
    ///
    /// This polls for responses until an acknowledgment or error for the
    /// specified command ID is received, or the timeout expires.
    ///
    /// # Arguments
    ///
    /// * `cmd_id` - The command ID to wait for
    /// * `timeout` - Maximum time to wait
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The timeout expires
    /// - The command fails with an error
    /// - The kernel terminates
    async fn wait_for_command(&self, cmd_id: CommandId, timeout: Duration) -> Result<()>;

    /// Gracefully shut down the kernel.
    ///
    /// This sends a Terminate command and waits for the kernel to acknowledge
    /// shutdown. Any in-progress work will be completed before termination.
    async fn shutdown(&self) -> Result<()>;

    /// Get the configuration for this handle.
    fn config(&self) -> &PersistentConfig;
}

// ============================================================================
// COMMAND TRACKER
// ============================================================================

/// Tracks in-flight commands for correlation with responses.
///
/// This is a utility struct that can be used by `PersistentHandle` implementations
/// to track pending commands and notify waiters when responses arrive.
#[derive(Debug)]
pub struct CommandTracker {
    next_id: std::sync::atomic::AtomicU64,
    pending: parking_lot::RwLock<
        std::collections::HashMap<CommandId, tokio::sync::oneshot::Sender<PersistentResponse>>,
    >,
}

impl CommandTracker {
    /// Create a new command tracker.
    pub fn new() -> Self {
        Self {
            next_id: std::sync::atomic::AtomicU64::new(1),
            pending: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Generate a new command ID.
    pub fn next_id(&self) -> CommandId {
        CommandId(
            self.next_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Register a waiter for a command.
    ///
    /// Returns a receiver that will receive the response when it arrives.
    pub fn register(
        &self,
        cmd_id: CommandId,
    ) -> tokio::sync::oneshot::Receiver<PersistentResponse> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending.write().insert(cmd_id, tx);
        rx
    }

    /// Notify a waiter that a response has arrived.
    ///
    /// Returns true if a waiter was notified, false if no waiter was registered.
    pub fn notify(&self, response: &PersistentResponse) -> bool {
        if let Some(cmd_id) = response.command_id() {
            if let Some(tx) = self.pending.write().remove(&cmd_id) {
                let _ = tx.send(response.clone());
                return true;
            }
        }
        false
    }

    /// Get the number of pending commands.
    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
    }
}

impl Default for CommandTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// MOCK IMPLEMENTATION (for testing)
// ============================================================================

/// Mock implementations for testing.
#[cfg(test)]
pub mod mock {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;

    /// Mock implementation of `PersistentHandle` for testing.
    pub struct MockPersistentHandle {
        kernel_id: String,
        running: AtomicBool,
        cmd_counter: AtomicU64,
        step_counter: AtomicU64,
        config: PersistentConfig,
        responses: parking_lot::Mutex<Vec<PersistentResponse>>,
    }

    impl MockPersistentHandle {
        /// Create a new mock handle.
        pub fn new(kernel_id: impl Into<String>) -> Self {
            Self {
                kernel_id: kernel_id.into(),
                running: AtomicBool::new(true),
                cmd_counter: AtomicU64::new(1),
                step_counter: AtomicU64::new(0),
                config: PersistentConfig::default(),
                responses: parking_lot::Mutex::new(Vec::new()),
            }
        }

        /// Queue a response for the next poll.
        pub fn queue_response(&self, response: PersistentResponse) {
            self.responses.lock().push(response);
        }

        /// Set the running state.
        pub fn set_running(&self, running: bool) {
            self.running.store(running, Ordering::Release);
        }
    }

    #[async_trait::async_trait]
    impl PersistentHandle for MockPersistentHandle {
        fn kernel_id(&self) -> &str {
            &self.kernel_id
        }

        fn is_running(&self) -> bool {
            self.running.load(Ordering::Acquire)
        }

        fn send_command(&self, cmd: PersistentCommand) -> Result<CommandId> {
            if !self.is_running() {
                return Err(EcosystemError::ServiceUnavailable(
                    "Kernel not running".to_string(),
                ));
            }

            let cmd_id = CommandId(self.cmd_counter.fetch_add(1, Ordering::Relaxed));

            // Auto-generate ack response
            self.responses
                .lock()
                .push(PersistentResponse::Ack { cmd_id });

            // Handle specific commands
            match cmd {
                PersistentCommand::RunSteps { count } => {
                    self.step_counter.fetch_add(count, Ordering::Relaxed);
                }
                PersistentCommand::Terminate => {
                    self.running.store(false, Ordering::Release);
                    self.responses.lock().push(PersistentResponse::Terminated {
                        final_step: self.step_counter.load(Ordering::Relaxed),
                    });
                }
                _ => {}
            }

            Ok(cmd_id)
        }

        fn poll_responses(&self) -> Vec<PersistentResponse> {
            std::mem::take(&mut *self.responses.lock())
        }

        fn stats(&self) -> PersistentStats {
            PersistentStats {
                current_step: self.step_counter.load(Ordering::Relaxed),
                is_running: self.is_running(),
                has_terminated: !self.is_running(),
                ..Default::default()
            }
        }

        async fn wait_for_command(&self, cmd_id: CommandId, timeout: Duration) -> Result<()> {
            let start = std::time::Instant::now();
            loop {
                for response in self.poll_responses() {
                    if response.command_id() == Some(cmd_id) {
                        if response.is_error() {
                            if let PersistentResponse::Error { message, .. } = response {
                                return Err(EcosystemError::Internal(message));
                            }
                        }
                        return Ok(());
                    }
                }

                if start.elapsed() > timeout {
                    return Err(EcosystemError::Timeout(timeout));
                }

                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }

        async fn shutdown(&self) -> Result<()> {
            let cmd_id = self.send_command(PersistentCommand::Terminate)?;
            self.wait_for_command(cmd_id, Duration::from_secs(5)).await
        }

        fn config(&self) -> &PersistentConfig {
            &self.config
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_id() {
        let id = CommandId::new(42);
        assert_eq!(id.as_u64(), 42);
        assert_eq!(format!("{}", id), "cmd-42");
    }

    #[test]
    fn test_persistent_command() {
        let cmd = PersistentCommand::run_steps(100);
        assert!(matches!(cmd, PersistentCommand::RunSteps { count: 100 }));

        let cmd = PersistentCommand::inject(1, 2, 3, 4.5);
        assert!(matches!(
            cmd,
            PersistentCommand::Inject {
                position: (1, 2, 3),
                value
            } if (value - 4.5).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn test_persistent_response_command_id() {
        let response = PersistentResponse::Ack {
            cmd_id: CommandId(1),
        };
        assert_eq!(response.command_id(), Some(CommandId(1)));

        let response = PersistentResponse::Terminated { final_step: 100 };
        assert_eq!(response.command_id(), None);
    }

    #[test]
    fn test_persistent_stats() {
        let mut stats = PersistentStats::default();
        stats.is_running = true;
        stats.steps_remaining = 0;
        assert!(stats.is_idle());

        stats.steps_remaining = 100;
        assert!(stats.is_processing());
    }

    #[test]
    fn test_persistent_config_builder() {
        let config = PersistentConfig::new()
            .with_h2k_capacity(100)
            .with_k2h_capacity(200)
            .with_progress_interval(50)
            .with_command_timeout(Duration::from_secs(60));

        assert_eq!(config.h2k_capacity, 128); // Rounded up to power of 2
        assert_eq!(config.k2h_capacity, 256); // Rounded up to power of 2
        assert_eq!(config.progress_interval, 50);
        assert_eq!(config.command_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_command_tracker() {
        let tracker = CommandTracker::new();

        let id1 = tracker.next_id();
        let id2 = tracker.next_id();
        assert_ne!(id1, id2);

        assert_eq!(tracker.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_mock_persistent_handle() {
        let handle = mock::MockPersistentHandle::new("test-kernel");

        assert!(handle.is_running());
        assert_eq!(handle.kernel_id(), "test-kernel");

        // Send command
        let cmd_id = handle
            .send_command(PersistentCommand::RunSteps { count: 100 })
            .unwrap();

        // Poll responses
        let responses = handle.poll_responses();
        assert!(!responses.is_empty());
        assert!(matches!(
            responses[0],
            PersistentResponse::Ack { cmd_id: id } if id == cmd_id
        ));

        // Check stats
        let stats = handle.stats();
        assert_eq!(stats.current_step, 100);
        assert!(stats.is_running);
    }

    #[tokio::test]
    async fn test_mock_shutdown() {
        let handle = mock::MockPersistentHandle::new("test-kernel");

        handle.shutdown().await.unwrap();

        assert!(!handle.is_running());
        assert!(handle.stats().has_terminated);
    }
}
