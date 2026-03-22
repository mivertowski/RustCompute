//! Error types for accounting network operations.

use thiserror::Error;

/// Result type alias for accounting network operations.
pub type Result<T> = std::result::Result<T, AccNetError>;

/// Comprehensive error type for accounting network operations.
#[derive(Error, Debug)]
pub enum AccNetError {
    // ===== CUDA Device Errors =====
    /// CUDA is not available on this system.
    #[error("CUDA is not available on this system")]
    CudaUnavailable,

    /// Failed to create or initialize a CUDA device.
    #[error("failed to create CUDA device: {0}")]
    DeviceCreation(String),

    /// GPU synchronization failed.
    #[error("GPU synchronize failed: {0}")]
    GpuSync(String),

    // ===== Kernel Compilation Errors =====
    /// NVRTC compilation of CUDA source failed.
    #[error("NVRTC compilation failed for '{kernel}': {reason}")]
    NvrtcCompilation {
        /// Kernel name that failed to compile.
        kernel: String,
        /// Compilation error details.
        reason: String,
    },

    /// Failed to load a compiled PTX module.
    #[error("failed to load PTX for '{kernel}': {reason}")]
    PtxLoad {
        /// Kernel name.
        kernel: String,
        /// Load error details.
        reason: String,
    },

    // ===== Kernel Execution Errors =====
    /// Kernel function was not found in the loaded module.
    #[error("kernel function '{0}' not found")]
    KernelNotFound(String),

    /// Kernel launch failed.
    #[error("kernel launch failed: {0}")]
    KernelLaunch(String),

    // ===== GPU Memory Errors =====
    /// Failed to copy data from host to device.
    #[error("failed to copy {field} to device: {reason}")]
    HostToDevice {
        /// Name of the buffer being copied.
        field: String,
        /// Error details.
        reason: String,
    },

    /// Failed to copy data from device to host.
    #[error("failed to copy results from device: {0}")]
    DeviceToHost(String),

    /// Failed to allocate GPU memory.
    #[error("failed to allocate GPU memory for '{field}': {reason}")]
    GpuAlloc {
        /// Name of the buffer being allocated.
        field: String,
        /// Error details.
        reason: String,
    },

    // ===== Code Generation Errors =====
    /// Kernel code generation (transpilation) failed.
    #[error("failed to generate {kernel} kernel: {reason}")]
    CodeGen {
        /// Name of the kernel being generated.
        kernel: String,
        /// Generation error details.
        reason: String,
    },

    // ===== Validation Errors =====
    /// Configuration validation failed.
    #[error("validation error: {0}")]
    Validation(String),
}

impl From<String> for AccNetError {
    fn from(s: String) -> Self {
        AccNetError::Validation(s)
    }
}

impl From<&str> for AccNetError {
    fn from(s: &str) -> Self {
        AccNetError::Validation(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AccNetError::CudaUnavailable;
        assert_eq!(format!("{err}"), "CUDA is not available on this system");

        let err = AccNetError::DeviceCreation("no driver".to_string());
        assert_eq!(format!("{err}"), "failed to create CUDA device: no driver");

        let err = AccNetError::NvrtcCompilation {
            kernel: "suspense_detection".to_string(),
            reason: "syntax error".to_string(),
        };
        assert!(format!("{err}").contains("suspense_detection"));
        assert!(format!("{err}").contains("syntax error"));

        let err = AccNetError::KernelNotFound("pagerank".to_string());
        assert!(format!("{err}").contains("pagerank"));

        let err = AccNetError::HostToDevice {
            field: "balance_debit".to_string(),
            reason: "out of memory".to_string(),
        };
        assert!(format!("{err}").contains("balance_debit"));

        let err = AccNetError::CodeGen {
            kernel: "fraud_detector".to_string(),
            reason: "unsupported op".to_string(),
        };
        assert!(format!("{err}").contains("fraud_detector"));
    }

    #[test]
    fn test_from_string() {
        let err: AccNetError = "bad config".to_string().into();
        assert!(matches!(err, AccNetError::Validation(_)));
        assert!(format!("{err}").contains("bad config"));
    }

    #[test]
    fn test_from_str() {
        let err: AccNetError = "bad config".into();
        assert!(matches!(err, AccNetError::Validation(_)));
    }

    #[test]
    fn test_result_alias() {
        let ok: Result<i32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: Result<i32> = Err(AccNetError::CudaUnavailable);
        assert!(err.is_err());
    }
}
