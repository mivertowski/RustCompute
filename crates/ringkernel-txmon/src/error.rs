//! Typed error handling for the ringkernel-txmon crate.

use std::fmt;

/// Errors produced by the txmon crate.
#[derive(Debug, thiserror::Error)]
pub enum TxMonError {
    /// Kernel lifecycle state violation (e.g., activating an already-active kernel).
    #[error("kernel state error: {0}")]
    KernelState(KernelStateError),

    /// CUDA codegen transpilation failure.
    #[error("codegen transpile failed for `{kernel}`: {reason}")]
    Transpile {
        kernel: &'static str,
        reason: String,
    },

    /// CUDA codegen feature is not enabled.
    #[error("CUDA codegen feature not enabled (build with --features cuda-codegen)")]
    CodegenNotEnabled,

    /// CUDA runtime error (device init, PTX load, kernel launch, memcpy, etc.).
    #[error("CUDA error: {context}: {reason}")]
    Cuda {
        context: &'static str,
        reason: String,
    },

    /// NVRTC compilation error.
    #[error("NVRTC compilation error: {0}")]
    NvrtcCompile(String),
}

/// Kernel lifecycle state errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelStateError {
    /// Attempted to activate a kernel that is already active.
    AlreadyActive,
    /// Attempted to activate a terminated or terminating kernel.
    Terminated,
    /// Attempted to pause a kernel that is not active.
    NotActive,
}

impl fmt::Display for KernelStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyActive => write!(f, "kernel already active"),
            Self::Terminated => write!(f, "cannot activate terminated kernel"),
            Self::NotActive => write!(f, "kernel not active"),
        }
    }
}

impl std::error::Error for KernelStateError {}

impl From<KernelStateError> for TxMonError {
    fn from(e: KernelStateError) -> Self {
        TxMonError::KernelState(e)
    }
}
