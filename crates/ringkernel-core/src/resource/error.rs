//! Error types for resource management.

use std::fmt;

/// Resource-related errors.
#[derive(Debug, Clone)]
pub enum ResourceError {
    /// Memory limit exceeded.
    MemoryLimitExceeded {
        /// Requested bytes.
        requested: u64,
        /// Current usage.
        current: u64,
        /// Maximum limit.
        max: u64,
    },
    /// Insufficient system memory.
    InsufficientSystemMemory {
        /// Requested bytes.
        requested: u64,
        /// Available bytes.
        available: u64,
        /// Safety margin.
        margin: u64,
    },
    /// Resource allocation failed.
    AllocationFailed(String),
    /// Resource already reserved.
    AlreadyReserved(String),
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceError::MemoryLimitExceeded {
                requested,
                current,
                max,
            } => {
                write!(
                    f,
                    "Memory limit exceeded: requested {} bytes, current usage {} bytes, limit {} bytes",
                    requested, current, max
                )
            }
            ResourceError::InsufficientSystemMemory {
                requested,
                available,
                margin,
            } => {
                write!(
                    f,
                    "Insufficient system memory: requested {} bytes, available {} bytes (margin: {} bytes)",
                    requested, available, margin
                )
            }
            ResourceError::AllocationFailed(msg) => {
                write!(f, "Resource allocation failed: {}", msg)
            }
            ResourceError::AlreadyReserved(msg) => {
                write!(f, "Resource already reserved: {}", msg)
            }
        }
    }
}

impl std::error::Error for ResourceError {}

/// Result type for resource operations.
pub type ResourceResult<T> = Result<T, ResourceError>;
