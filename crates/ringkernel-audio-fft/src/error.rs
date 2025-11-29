//! Error types for audio FFT processing.

use thiserror::Error;

/// Result type for audio FFT operations.
pub type Result<T> = std::result::Result<T, AudioFftError>;

/// Errors that can occur during audio FFT processing.
#[derive(Error, Debug)]
pub enum AudioFftError {
    /// Error reading audio file.
    #[error("Failed to read audio file: {0}")]
    FileReadError(String),

    /// Error writing audio file.
    #[error("Failed to write audio file: {0}")]
    FileWriteError(String),

    /// Invalid audio format.
    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),

    /// FFT processing error.
    #[error("FFT processing error: {0}")]
    FftError(String),

    /// Audio device error.
    #[error("Audio device error: {0}")]
    DeviceError(String),

    /// K2K messaging error.
    #[error("K2K messaging error: {0}")]
    K2KError(String),

    /// Kernel error.
    #[error("Kernel error: {0}")]
    KernelError(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Buffer underrun/overrun.
    #[error("Buffer {0}")]
    BufferError(String),

    /// Processing timeout.
    #[error("Processing timeout: {0}")]
    Timeout(String),

    /// RingKernel core error.
    #[error("RingKernel error: {0}")]
    RingKernelError(#[from] ringkernel_core::error::RingKernelError),

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl AudioFftError {
    /// Create a file read error.
    pub fn file_read(msg: impl Into<String>) -> Self {
        Self::FileReadError(msg.into())
    }

    /// Create a file write error.
    pub fn file_write(msg: impl Into<String>) -> Self {
        Self::FileWriteError(msg.into())
    }

    /// Create an invalid format error.
    pub fn invalid_format(msg: impl Into<String>) -> Self {
        Self::InvalidFormat(msg.into())
    }

    /// Create an FFT error.
    pub fn fft(msg: impl Into<String>) -> Self {
        Self::FftError(msg.into())
    }

    /// Create a device error.
    pub fn device(msg: impl Into<String>) -> Self {
        Self::DeviceError(msg.into())
    }

    /// Create a K2K error.
    pub fn k2k(msg: impl Into<String>) -> Self {
        Self::K2KError(msg.into())
    }

    /// Create a kernel error.
    pub fn kernel(msg: impl Into<String>) -> Self {
        Self::KernelError(msg.into())
    }

    /// Create a config error.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::ConfigError(msg.into())
    }

    /// Create a buffer error.
    pub fn buffer(msg: impl Into<String>) -> Self {
        Self::BufferError(msg.into())
    }

    /// Create a timeout error.
    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::Timeout(msg.into())
    }
}
