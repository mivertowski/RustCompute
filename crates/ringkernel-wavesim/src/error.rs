//! Error types for wave simulation operations.

use thiserror::Error;

/// Result type alias for wave simulation operations.
pub type Result<T> = std::result::Result<T, WaveSimError>;

/// Comprehensive error type for wave simulation operations.
#[derive(Error, Debug, Clone)]
pub enum WaveSimError {
    // ===== Simulation Errors =====
    /// Simulation grid initialization failed.
    #[error("simulation initialization failed: {0}")]
    SimulationInit(String),

    /// Simulation step failed.
    #[error("simulation step failed: {0}")]
    SimulationStep(String),

    // ===== Backend Errors =====
    /// Backend switch failed.
    #[error("backend switch failed: {0}")]
    BackendSwitch(String),

    /// GPU backend creation failed.
    #[error("GPU backend creation failed: {0}")]
    GpuBackendCreation(String),

    // ===== Kernel Errors =====
    /// Kernel launch or execution failed.
    #[error("kernel error: {0}")]
    Kernel(String),

    // ===== RingKernel Errors =====
    /// Underlying RingKernel error (stored as string since RingKernelError is not Clone).
    #[error("ringkernel error: {0}")]
    RingKernel(String),
}

impl From<ringkernel_core::error::RingKernelError> for WaveSimError {
    fn from(e: ringkernel_core::error::RingKernelError) -> Self {
        WaveSimError::RingKernel(e.to_string())
    }
}

impl From<String> for WaveSimError {
    fn from(s: String) -> Self {
        WaveSimError::Kernel(s)
    }
}

impl From<&str> for WaveSimError {
    fn from(s: &str) -> Self {
        WaveSimError::Kernel(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = WaveSimError::SimulationInit("bad params".to_string());
        assert_eq!(
            format!("{err}"),
            "simulation initialization failed: bad params"
        );

        let err = WaveSimError::BackendSwitch("no GPU".to_string());
        assert_eq!(format!("{err}"), "backend switch failed: no GPU");

        let err = WaveSimError::GpuBackendCreation("CUDA unavailable".to_string());
        assert!(format!("{err}").contains("CUDA unavailable"));

        let err = WaveSimError::Kernel("launch failed".to_string());
        assert!(format!("{err}").contains("launch failed"));

        let err = WaveSimError::SimulationStep("divergence".to_string());
        assert!(format!("{err}").contains("divergence"));
    }

    #[test]
    fn test_from_string() {
        let err: WaveSimError = "bad config".to_string().into();
        assert!(matches!(err, WaveSimError::Kernel(_)));
        assert!(format!("{err}").contains("bad config"));
    }

    #[test]
    fn test_from_str() {
        let err: WaveSimError = "bad config".into();
        assert!(matches!(err, WaveSimError::Kernel(_)));
    }

    #[test]
    fn test_from_ringkernel_error() {
        let rk_err = ringkernel_core::error::RingKernelError::BackendError("test".to_string());
        let err: WaveSimError = rk_err.into();
        assert!(matches!(err, WaveSimError::RingKernel(_)));
        assert!(format!("{err}").contains("test"));
    }

    #[test]
    fn test_result_alias() {
        let ok: Result<i32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: Result<i32> = Err(WaveSimError::SimulationInit("fail".to_string()));
        assert!(err.is_err());
    }

    #[test]
    fn test_clone() {
        let err = WaveSimError::BackendSwitch("gpu failed".to_string());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }
}
