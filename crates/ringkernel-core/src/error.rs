//! Error types for RingKernel operations.

use thiserror::Error;

/// Result type alias for RingKernel operations.
pub type Result<T> = std::result::Result<T, RingKernelError>;

/// Comprehensive error type for RingKernel operations.
#[derive(Error, Debug)]
pub enum RingKernelError {
    // ===== Kernel Lifecycle Errors =====
    /// Kernel not found with the given ID.
    #[error("kernel not found: {0}")]
    KernelNotFound(String),

    /// Kernel is already active.
    #[error("kernel already active: {0}")]
    KernelAlreadyActive(String),

    /// Kernel is not active.
    #[error("kernel not active: {0}")]
    KernelNotActive(String),

    /// Kernel has already terminated.
    #[error("kernel already terminated: {0}")]
    KernelTerminated(String),

    /// Invalid kernel state transition.
    #[error("invalid state transition from {from:?} to {to:?}")]
    InvalidStateTransition {
        /// Current state
        from: String,
        /// Attempted target state
        to: String,
    },

    /// Invalid kernel state.
    #[error("invalid state: expected {expected}, got {actual}")]
    InvalidState {
        /// Expected state
        expected: String,
        /// Actual state
        actual: String,
    },

    /// Kernel launch failed.
    #[error("kernel launch failed: {0}")]
    LaunchFailed(String),

    /// Kernel compilation failed (NVRTC, shader compilation, etc.).
    #[error("kernel compilation failed: {0}")]
    CompilationError(String),

    // ===== Message Errors =====
    /// Queue is full, message cannot be enqueued.
    #[error("queue full: capacity {capacity}, attempted to enqueue message")]
    QueueFull {
        /// Queue capacity
        capacity: usize,
    },

    /// Queue is empty, no message to dequeue.
    #[error("queue empty")]
    QueueEmpty,

    /// Message serialization failed.
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// Message deserialization failed.
    #[error("deserialization error: {0}")]
    DeserializationError(String),

    /// Message validation failed.
    #[error("message validation failed: {0}")]
    ValidationError(String),

    /// Message too large.
    #[error("message too large: {size} bytes (max: {max} bytes)")]
    MessageTooLarge {
        /// Actual message size
        size: usize,
        /// Maximum allowed size
        max: usize,
    },

    /// Message timeout.
    #[error("message timeout after {0:?}")]
    Timeout(std::time::Duration),

    // ===== Memory Errors =====
    /// GPU memory allocation failed.
    #[error("GPU memory allocation failed: {size} bytes - {reason}")]
    AllocationFailed {
        /// Requested size
        size: usize,
        /// Failure reason
        reason: String,
    },

    /// Host memory allocation failed.
    #[error("host memory allocation failed: {size} bytes")]
    HostAllocationFailed {
        /// Requested size
        size: usize,
    },

    /// Memory transfer failed.
    #[error("memory transfer failed: {0}")]
    TransferFailed(String),

    /// Invalid memory alignment.
    #[error("invalid alignment: expected {expected}, got {actual}")]
    InvalidAlignment {
        /// Expected alignment
        expected: usize,
        /// Actual alignment
        actual: usize,
    },

    /// Out of GPU memory.
    #[error("out of GPU memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory {
        /// Requested size
        requested: usize,
        /// Available memory
        available: usize,
    },

    /// Memory pool exhausted.
    #[error("memory pool exhausted")]
    PoolExhausted,

    /// Generic memory error.
    #[error("memory error: {0}")]
    MemoryError(String),

    // ===== Backend Errors =====
    /// Backend not available.
    #[error("backend not available: {0}")]
    BackendUnavailable(String),

    /// Backend initialization failed.
    #[error("backend initialization failed: {0}")]
    BackendInitFailed(String),

    /// No suitable GPU device found.
    #[error("no GPU device found")]
    NoDeviceFound,

    /// Device selection failed.
    #[error("device selection failed: {0}")]
    DeviceSelectionFailed(String),

    /// Backend operation failed.
    #[error("backend error: {0}")]
    BackendError(String),

    // ===== Synchronization Errors =====
    /// Deadlock detected.
    #[error("deadlock detected")]
    DeadlockDetected,

    /// Lock poisoned.
    #[error("lock poisoned")]
    LockPoisoned,

    /// Channel closed.
    #[error("channel closed")]
    ChannelClosed,

    // ===== HLC Errors =====
    /// Clock skew too large.
    #[error("clock skew too large: {skew_ms}ms (max: {max_ms}ms)")]
    ClockSkew {
        /// Detected skew in milliseconds
        skew_ms: u64,
        /// Maximum allowed skew
        max_ms: u64,
    },

    /// Invalid timestamp.
    #[error("invalid timestamp")]
    InvalidTimestamp,

    // ===== K2K Messaging Errors =====
    /// K2K messaging error.
    #[error("K2K error: {0}")]
    K2KError(String),

    /// K2K destination not found.
    #[error("K2K destination not found: {0}")]
    K2KDestinationNotFound(String),

    /// K2K delivery failed.
    #[error("K2K delivery failed: {0}")]
    K2KDeliveryFailed(String),

    // ===== Pub/Sub Errors =====
    /// Pub/sub error.
    #[error("pub/sub error: {0}")]
    PubSubError(String),

    /// Topic not found.
    #[error("topic not found: {0}")]
    TopicNotFound(String),

    /// Subscription error.
    #[error("subscription error: {0}")]
    SubscriptionError(String),

    // ===== Multi-GPU Errors =====
    /// Multi-GPU coordination error.
    #[error("multi-GPU error: {0}")]
    MultiGpuError(String),

    /// Device not available.
    #[error("device not available: {0}")]
    DeviceNotAvailable(String),

    /// Cross-device transfer failed.
    #[error("cross-device transfer failed: {0}")]
    CrossDeviceTransferFailed(String),

    // ===== Telemetry Errors =====
    /// Telemetry error.
    #[error("telemetry error: {0}")]
    TelemetryError(String),

    /// Metrics collection failed.
    #[error("metrics collection failed: {0}")]
    MetricsCollectionFailed(String),

    // ===== Configuration Errors =====
    /// Invalid configuration.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Missing required configuration.
    #[error("missing configuration: {0}")]
    MissingConfig(String),

    // ===== I/O Errors =====
    /// I/O error wrapper.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    // ===== Generic Errors =====
    /// Internal error.
    #[error("internal error: {0}")]
    Internal(String),

    /// Feature not supported.
    #[error("feature not supported: {0}")]
    NotSupported(String),

    /// Operation cancelled.
    #[error("operation cancelled")]
    Cancelled,
}

impl RingKernelError {
    /// Returns true if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            RingKernelError::QueueFull { .. }
                | RingKernelError::QueueEmpty
                | RingKernelError::Timeout(_)
                | RingKernelError::PoolExhausted
        )
    }

    /// Returns true if this error indicates a resource issue.
    pub fn is_resource_error(&self) -> bool {
        matches!(
            self,
            RingKernelError::AllocationFailed { .. }
                | RingKernelError::HostAllocationFailed { .. }
                | RingKernelError::OutOfMemory { .. }
                | RingKernelError::PoolExhausted
        )
    }

    /// Returns true if this is a fatal error requiring restart.
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            RingKernelError::BackendInitFailed(_)
                | RingKernelError::NoDeviceFound
                | RingKernelError::LockPoisoned
                | RingKernelError::Internal(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RingKernelError::KernelNotFound("test_kernel".to_string());
        assert_eq!(format!("{}", err), "kernel not found: test_kernel");

        let err = RingKernelError::QueueFull { capacity: 1024 };
        assert!(format!("{}", err).contains("1024"));
    }

    #[test]
    fn test_error_classification() {
        assert!(RingKernelError::QueueFull { capacity: 1024 }.is_recoverable());
        assert!(RingKernelError::OutOfMemory {
            requested: 1000,
            available: 100
        }
        .is_resource_error());
        assert!(RingKernelError::LockPoisoned.is_fatal());
    }
}
