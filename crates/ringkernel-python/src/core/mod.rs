//! Core RingKernel Python bindings.
//!
//! This module provides the fundamental types for the RingKernel runtime:
//!
//! - `RingKernel` - Main runtime for managing GPU kernels
//! - `KernelHandle` - Handle for interacting with launched kernels
//! - `HlcTimestamp`, `HlcClock` - Hybrid logical clock types
//! - `MessageId`, `MessageEnvelope` - Message types
//! - `K2KBroker`, `K2KConfig` - Kernel-to-kernel messaging
//! - `QueueStats`, `QueueTier` - Queue monitoring

pub mod hlc;
pub mod k2k;
pub mod message;
pub mod queue;
pub mod runtime;

// Re-export main types (public API)
#[allow(unused_imports)]
pub use hlc::{PyHlcClock, PyHlcTimestamp};
#[allow(unused_imports)]
pub use k2k::{PyDeliveryReceipt, PyDeliveryStatus, PyK2KBroker, PyK2KConfig, PyK2KStats};
#[allow(unused_imports)]
pub use message::{PyCorrelationId, PyMessageEnvelope, PyMessageHeader, PyMessageId, PyPriority};
#[allow(unused_imports)]
pub use queue::{PyQueueHealth, PyQueueMetrics, PyQueueMonitor, PyQueueStats, PyQueueTier};
#[allow(unused_imports)]
pub use runtime::{PyKernelHandle, PyKernelState, PyLaunchOptions, PyRingKernel};

use pyo3::prelude::*;

/// Register all core types with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register submodules
    hlc::register(m)?;
    message::register(m)?;
    queue::register(m)?;
    k2k::register(m)?;
    runtime::register(m)?;

    Ok(())
}
