//! Error types for hybrid processing.

use std::fmt;

/// Error type for hybrid processing operations.
#[derive(Debug, Clone)]
pub enum HybridError {
    /// GPU is not available.
    GpuNotAvailable,
    /// GPU execution failed.
    GpuExecutionFailed(String),
    /// Workload size exceeds limits.
    WorkloadTooLarge {
        /// Requested size.
        requested: usize,
        /// Maximum allowed.
        maximum: usize,
    },
    /// Configuration error.
    ConfigError(String),
    /// Resource allocation failed.
    ResourceAllocationFailed(String),
}

impl fmt::Display for HybridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HybridError::GpuNotAvailable => write!(f, "GPU is not available"),
            HybridError::GpuExecutionFailed(msg) => write!(f, "GPU execution failed: {}", msg),
            HybridError::WorkloadTooLarge { requested, maximum } => {
                write!(f, "Workload size {} exceeds maximum {}", requested, maximum)
            }
            HybridError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            HybridError::ResourceAllocationFailed(msg) => {
                write!(f, "Resource allocation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for HybridError {}

/// Result type for hybrid processing operations.
pub type HybridResult<T> = Result<T, HybridError>;
