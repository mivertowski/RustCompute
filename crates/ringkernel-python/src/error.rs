//! Exception hierarchy for RingKernel Python bindings.
//!
//! Maps Rust errors to Python exceptions with actionable context.

use pyo3::prelude::*;
use pyo3::exceptions::{PyException, PyTimeoutError, PyValueError};
use std::fmt;

// Base exception for all RingKernel errors
pyo3::create_exception!(ringkernel, RingKernelError, PyException,
    "Base exception for all RingKernel errors.");

// Memory-related errors
pyo3::create_exception!(ringkernel, MemoryLimitError, RingKernelError,
    "Memory limit exceeded during allocation.");

// Kernel-related errors
pyo3::create_exception!(ringkernel, KernelError, RingKernelError,
    "Kernel not found or in invalid state.");

pyo3::create_exception!(ringkernel, KernelStateError, KernelError,
    "Invalid kernel state transition.");

// CUDA-specific errors
pyo3::create_exception!(ringkernel, CudaError, RingKernelError,
    "CUDA operation failed.");

pyo3::create_exception!(ringkernel, CudaDeviceError, CudaError,
    "CUDA device not available or invalid.");

pyo3::create_exception!(ringkernel, CudaMemoryError, CudaError,
    "CUDA memory operation failed.");

// Queue-related errors
pyo3::create_exception!(ringkernel, QueueError, RingKernelError,
    "Queue operation failed.");

pyo3::create_exception!(ringkernel, QueueFullError, QueueError,
    "Queue is full, message dropped.");

pyo3::create_exception!(ringkernel, QueueEmptyError, QueueError,
    "Queue is empty, no messages available.");

// K2K messaging errors
pyo3::create_exception!(ringkernel, K2KError, RingKernelError,
    "Kernel-to-kernel messaging error.");

pyo3::create_exception!(ringkernel, K2KDeliveryError, K2KError,
    "Message delivery failed.");

// Benchmark errors
pyo3::create_exception!(ringkernel, BenchmarkError, RingKernelError,
    "Benchmark operation failed.");

// Hybrid dispatch errors
pyo3::create_exception!(ringkernel, HybridError, RingKernelError,
    "Hybrid CPU/GPU dispatch error.");

pyo3::create_exception!(ringkernel, GpuNotAvailableError, HybridError,
    "GPU not available for execution.");

// Resource errors
pyo3::create_exception!(ringkernel, ResourceError, RingKernelError,
    "Resource management error.");

pyo3::create_exception!(ringkernel, ReservationError, ResourceError,
    "Memory reservation failed.");

/// Internal error type for the Python bindings.
#[derive(Debug)]
pub struct PyRingKernelError {
    pub kind: ErrorKind,
    pub message: String,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    // Core errors
    Runtime,
    Timeout,
    InvalidArgument,

    // Memory errors
    MemoryLimit,
    OutOfMemory,

    // Kernel errors
    KernelNotFound,
    InvalidKernelState,

    // CUDA errors
    CudaNotAvailable,
    CudaDeviceError,
    CudaMemoryError,
    CudaOperationError,

    // Queue errors
    QueueFull,
    QueueEmpty,

    // K2K errors
    K2KDeliveryFailed,
    K2KEndpointNotFound,

    // Hybrid errors
    GpuNotAvailable,
    WorkloadTooLarge,

    // Resource errors
    ReservationFailed,

    // Benchmark errors
    BenchmarkFailed,
}

impl PyRingKernelError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            context: None,
        }
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Convert to a Python exception.
    pub fn into_py_err(self) -> PyErr {
        let full_message = match self.context {
            Some(ctx) => format!("{}: {}", self.message, ctx),
            None => self.message,
        };

        match self.kind {
            ErrorKind::Runtime => RingKernelError::new_err(full_message),
            ErrorKind::Timeout => PyTimeoutError::new_err(full_message),
            ErrorKind::InvalidArgument => PyValueError::new_err(full_message),

            ErrorKind::MemoryLimit | ErrorKind::OutOfMemory => {
                MemoryLimitError::new_err(full_message)
            }

            ErrorKind::KernelNotFound => KernelError::new_err(full_message),
            ErrorKind::InvalidKernelState => KernelStateError::new_err(full_message),

            ErrorKind::CudaNotAvailable => CudaDeviceError::new_err(full_message),
            ErrorKind::CudaDeviceError => CudaDeviceError::new_err(full_message),
            ErrorKind::CudaMemoryError => CudaMemoryError::new_err(full_message),
            ErrorKind::CudaOperationError => CudaError::new_err(full_message),

            ErrorKind::QueueFull => QueueFullError::new_err(full_message),
            ErrorKind::QueueEmpty => QueueEmptyError::new_err(full_message),

            ErrorKind::K2KDeliveryFailed => K2KDeliveryError::new_err(full_message),
            ErrorKind::K2KEndpointNotFound => K2KError::new_err(full_message),

            ErrorKind::GpuNotAvailable => GpuNotAvailableError::new_err(full_message),
            ErrorKind::WorkloadTooLarge => HybridError::new_err(full_message),

            ErrorKind::ReservationFailed => ReservationError::new_err(full_message),

            ErrorKind::BenchmarkFailed => BenchmarkError::new_err(full_message),
        }
    }
}

impl fmt::Display for PyRingKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.context {
            Some(ctx) => write!(f, "{}: {}", self.message, ctx),
            None => write!(f, "{}", self.message),
        }
    }
}

impl std::error::Error for PyRingKernelError {}

impl From<PyRingKernelError> for PyErr {
    fn from(err: PyRingKernelError) -> PyErr {
        err.into_py_err()
    }
}

// Conversion from ringkernel-core errors
impl From<ringkernel_core::error::RingKernelError> for PyRingKernelError {
    fn from(err: ringkernel_core::error::RingKernelError) -> Self {
        use ringkernel_core::error::RingKernelError as RKError;

        match &err {
            // Kernel lifecycle errors
            RKError::KernelNotFound(id) => {
                PyRingKernelError::new(ErrorKind::KernelNotFound, format!("Kernel not found: {}", id))
            }
            RKError::KernelAlreadyActive(id) => {
                PyRingKernelError::new(ErrorKind::InvalidKernelState, format!("Kernel already active: {}", id))
            }
            RKError::KernelNotActive(id) => {
                PyRingKernelError::new(ErrorKind::InvalidKernelState, format!("Kernel not active: {}", id))
            }
            RKError::KernelTerminated(id) => {
                PyRingKernelError::new(ErrorKind::InvalidKernelState, format!("Kernel terminated: {}", id))
            }
            RKError::InvalidStateTransition { from, to } => {
                PyRingKernelError::new(
                    ErrorKind::InvalidKernelState,
                    format!("Invalid state transition: {} -> {}", from, to),
                )
            }
            RKError::InvalidState { expected, actual } => {
                PyRingKernelError::new(
                    ErrorKind::InvalidKernelState,
                    format!("Expected state {}, got {}", expected, actual),
                )
            }
            RKError::LaunchFailed(reason) => {
                PyRingKernelError::new(ErrorKind::CudaOperationError, format!("Launch failed: {}", reason))
            }
            RKError::CompilationError(msg) => {
                PyRingKernelError::new(ErrorKind::CudaOperationError, format!("Compilation failed: {}", msg))
            }

            // Queue errors
            RKError::QueueFull { capacity } => {
                PyRingKernelError::new(ErrorKind::QueueFull, format!("Queue full (capacity: {})", capacity))
            }
            RKError::QueueEmpty => {
                PyRingKernelError::new(ErrorKind::QueueEmpty, "Queue is empty")
            }

            // Message errors
            RKError::SerializationError(msg) => {
                PyRingKernelError::new(ErrorKind::Runtime, format!("Serialization failed: {}", msg))
            }
            RKError::DeserializationError(msg) => {
                PyRingKernelError::new(ErrorKind::Runtime, format!("Deserialization failed: {}", msg))
            }
            RKError::ValidationError(msg) => {
                PyRingKernelError::new(ErrorKind::InvalidArgument, format!("Validation failed: {}", msg))
            }
            RKError::MessageTooLarge { size, max } => {
                PyRingKernelError::new(
                    ErrorKind::InvalidArgument,
                    format!("Message too large: {} bytes (max: {} bytes)", size, max),
                )
            }
            RKError::Timeout(duration) => {
                PyRingKernelError::new(ErrorKind::Timeout, format!("Timeout after {:?}", duration))
            }

            // Memory errors
            RKError::AllocationFailed { size, reason } => {
                PyRingKernelError::new(
                    ErrorKind::OutOfMemory,
                    format!("Allocation of {} bytes failed: {}", size, reason),
                )
            }
            RKError::HostAllocationFailed { size } => {
                PyRingKernelError::new(
                    ErrorKind::OutOfMemory,
                    format!("Host allocation of {} bytes failed", size),
                )
            }
            RKError::OutOfMemory { requested, available } => {
                PyRingKernelError::new(
                    ErrorKind::OutOfMemory,
                    format!("Out of memory: requested {} bytes, available {} bytes", requested, available),
                )
            }
            RKError::PoolExhausted => {
                PyRingKernelError::new(ErrorKind::OutOfMemory, "Memory pool exhausted")
            }
            RKError::TransferFailed(msg) => {
                PyRingKernelError::new(ErrorKind::CudaMemoryError, format!("Transfer failed: {}", msg))
            }
            RKError::MemoryError(msg) => {
                PyRingKernelError::new(ErrorKind::CudaMemoryError, msg.clone())
            }

            // Backend errors
            RKError::BackendUnavailable(backend) => {
                PyRingKernelError::new(ErrorKind::CudaNotAvailable, format!("Backend not available: {}", backend))
            }
            RKError::BackendInitFailed(msg) => {
                PyRingKernelError::new(ErrorKind::CudaNotAvailable, format!("Backend init failed: {}", msg))
            }
            RKError::NoDeviceFound => {
                PyRingKernelError::new(ErrorKind::CudaDeviceError, "No GPU device found")
            }
            RKError::DeviceNotAvailable(msg) => {
                PyRingKernelError::new(ErrorKind::CudaDeviceError, format!("Device not available: {}", msg))
            }
            RKError::BackendError(msg) => {
                PyRingKernelError::new(ErrorKind::CudaOperationError, msg.clone())
            }

            // K2K errors
            RKError::K2KError(msg) => {
                PyRingKernelError::new(ErrorKind::K2KDeliveryFailed, msg.clone())
            }
            RKError::K2KDestinationNotFound(dest) => {
                PyRingKernelError::new(ErrorKind::K2KEndpointNotFound, format!("Destination not found: {}", dest))
            }
            RKError::K2KDeliveryFailed(msg) => {
                PyRingKernelError::new(ErrorKind::K2KDeliveryFailed, msg.clone())
            }

            // HLC errors
            RKError::ClockSkew { skew_ms, max_ms } => {
                PyRingKernelError::new(
                    ErrorKind::Runtime,
                    format!("Clock skew too large: {}ms (max: {}ms)", skew_ms, max_ms),
                )
            }

            // Sync errors
            RKError::ChannelClosed => {
                PyRingKernelError::new(ErrorKind::Runtime, "Channel closed")
            }
            RKError::LockPoisoned => {
                PyRingKernelError::new(ErrorKind::Runtime, "Lock poisoned")
            }
            RKError::DeadlockDetected => {
                PyRingKernelError::new(ErrorKind::Runtime, "Deadlock detected")
            }

            // Health errors
            RKError::CircuitBreakerOpen { name } => {
                PyRingKernelError::new(ErrorKind::Runtime, format!("Circuit breaker open: {}", name))
            }
            RKError::RetryExhausted { attempts, reason } => {
                PyRingKernelError::new(
                    ErrorKind::Runtime,
                    format!("Retry exhausted after {} attempts: {}", attempts, reason),
                )
            }

            // Internal/generic errors
            RKError::Internal(msg) => {
                PyRingKernelError::new(ErrorKind::Runtime, format!("Internal error: {}", msg))
            }
            RKError::NotSupported(feature) => {
                PyRingKernelError::new(ErrorKind::Runtime, format!("Not supported: {}", feature))
            }
            RKError::Cancelled => {
                PyRingKernelError::new(ErrorKind::Runtime, "Operation cancelled")
            }

            // Catch-all for any other errors
            _ => {
                PyRingKernelError::new(ErrorKind::Runtime, err.to_string())
            }
        }
    }
}

// Convenience trait for converting Results
#[allow(dead_code)]
pub trait IntoPyResult<T> {
    fn into_py_result(self) -> PyResult<T>;
}

impl<T, E: Into<PyRingKernelError>> IntoPyResult<T> for Result<T, E> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(|e| e.into().into_py_err())
    }
}

/// Register exception types with the Python module.
pub fn register_exceptions(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create exceptions module
    let exceptions = PyModule::new_bound(py, "exceptions")?;

    // Add base exception
    exceptions.add("RingKernelError", py.get_type_bound::<RingKernelError>())?;

    // Memory errors
    exceptions.add("MemoryLimitError", py.get_type_bound::<MemoryLimitError>())?;

    // Kernel errors
    exceptions.add("KernelError", py.get_type_bound::<KernelError>())?;
    exceptions.add("KernelStateError", py.get_type_bound::<KernelStateError>())?;

    // CUDA errors
    exceptions.add("CudaError", py.get_type_bound::<CudaError>())?;
    exceptions.add("CudaDeviceError", py.get_type_bound::<CudaDeviceError>())?;
    exceptions.add("CudaMemoryError", py.get_type_bound::<CudaMemoryError>())?;

    // Queue errors
    exceptions.add("QueueError", py.get_type_bound::<QueueError>())?;
    exceptions.add("QueueFullError", py.get_type_bound::<QueueFullError>())?;
    exceptions.add("QueueEmptyError", py.get_type_bound::<QueueEmptyError>())?;

    // K2K errors
    exceptions.add("K2KError", py.get_type_bound::<K2KError>())?;
    exceptions.add("K2KDeliveryError", py.get_type_bound::<K2KDeliveryError>())?;

    // Benchmark errors
    exceptions.add("BenchmarkError", py.get_type_bound::<BenchmarkError>())?;

    // Hybrid errors
    exceptions.add("HybridError", py.get_type_bound::<HybridError>())?;
    exceptions.add("GpuNotAvailableError", py.get_type_bound::<GpuNotAvailableError>())?;

    // Resource errors
    exceptions.add("ResourceError", py.get_type_bound::<ResourceError>())?;
    exceptions.add("ReservationError", py.get_type_bound::<ReservationError>())?;

    // Add exceptions module to main module
    m.add_submodule(&exceptions)?;

    // Also add commonly-used exceptions at top level
    m.add("RingKernelError", py.get_type_bound::<RingKernelError>())?;
    m.add("CudaError", py.get_type_bound::<CudaError>())?;
    m.add("KernelError", py.get_type_bound::<KernelError>())?;
    m.add("MemoryLimitError", py.get_type_bound::<MemoryLimitError>())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_with_context() {
        let err = PyRingKernelError::new(ErrorKind::MemoryLimit, "Allocation failed")
            .with_context("requested 1GB");
        assert!(err.to_string().contains("Allocation failed"));
        assert!(err.to_string().contains("requested 1GB"));
    }

    #[test]
    fn test_error_kind_mapping() {
        let err = PyRingKernelError::new(ErrorKind::QueueFull, "Queue is full");
        assert_eq!(err.kind, ErrorKind::QueueFull);
    }
}
