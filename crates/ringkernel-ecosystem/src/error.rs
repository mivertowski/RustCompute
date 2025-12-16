//! Error types for ecosystem integrations.

use thiserror::Error;

/// Ecosystem integration error type.
#[derive(Error, Debug)]
pub enum EcosystemError {
    /// RingKernel core error.
    #[error("RingKernel error: {0}")]
    RingKernel(#[from] ringkernel_core::error::RingKernelError),

    /// Timeout waiting for response.
    #[error("Operation timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// Kernel not found.
    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    /// Invalid configuration.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Deserialization error.
    #[error("Deserialization error: {0}")]
    Deserialization(String),

    /// Data conversion error.
    #[error("Data conversion error: {0}")]
    DataConversion(String),

    /// Service unavailable.
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// Actor error (for actix integration).
    #[cfg(feature = "actix")]
    #[error("Actor error: {0}")]
    Actor(String),

    /// gRPC error (for tonic integration).
    #[cfg(feature = "grpc")]
    #[error("gRPC error: {0}")]
    Grpc(String),

    /// Arrow error (for arrow integration).
    #[cfg(feature = "arrow")]
    #[error("Arrow error: {0}")]
    Arrow(String),

    /// Polars error (for polars integration).
    #[cfg(feature = "polars")]
    #[error("Polars error: {0}")]
    Polars(String),

    /// Candle error (for ML integration).
    #[cfg(feature = "candle")]
    #[error("Candle error: {0}")]
    Candle(String),

    /// Persistent kernel error (for persistent integration).
    #[cfg(feature = "persistent")]
    #[error("Persistent kernel error: {0}")]
    Persistent(String),

    /// Command queue full.
    #[cfg(feature = "persistent")]
    #[error("Command queue full")]
    QueueFull,

    /// Kernel not running.
    #[cfg(feature = "persistent")]
    #[error("Kernel not running: {0}")]
    KernelNotRunning(String),

    /// Command failed with error code.
    #[cfg(feature = "persistent")]
    #[error("Command failed (code {code}): {message}")]
    CommandFailed {
        /// Error code from kernel.
        code: u32,
        /// Error message.
        message: String,
    },
}

/// Result type for ecosystem operations.
pub type Result<T> = std::result::Result<T, EcosystemError>;

/// Error that can be returned from Axum handlers.
#[cfg(feature = "axum")]
#[derive(Debug)]
pub struct AppError(pub EcosystemError);

#[cfg(feature = "axum")]
impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(feature = "axum")]
impl std::error::Error for AppError {}

#[cfg(feature = "axum")]
impl axum::response::IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;

        let status = match &self.0 {
            EcosystemError::KernelNotFound(_) => StatusCode::NOT_FOUND,
            EcosystemError::Timeout(_) => StatusCode::GATEWAY_TIMEOUT,
            EcosystemError::Config(_) => StatusCode::BAD_REQUEST,
            EcosystemError::ServiceUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (status, self.0.to_string()).into_response()
    }
}

#[cfg(feature = "axum")]
impl From<EcosystemError> for AppError {
    fn from(err: EcosystemError) -> Self {
        AppError(err)
    }
}

#[cfg(feature = "grpc")]
impl From<EcosystemError> for tonic::Status {
    fn from(err: EcosystemError) -> Self {
        use tonic::Code;

        let code = match &err {
            EcosystemError::KernelNotFound(_) => Code::NotFound,
            EcosystemError::Timeout(_) => Code::DeadlineExceeded,
            EcosystemError::Config(_) => Code::InvalidArgument,
            EcosystemError::ServiceUnavailable(_) => Code::Unavailable,
            _ => Code::Internal,
        };

        tonic::Status::new(code, err.to_string())
    }
}
