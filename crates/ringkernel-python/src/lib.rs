//! RingKernel Python Bindings
//!
//! Python bindings for the RingKernel GPU-native persistent actor framework.
//!
//! # Overview
//!
//! This crate provides PyO3-based Python bindings for RingKernel, exposing:
//!
//! - Core runtime and kernel management
//! - Hybrid Logical Clock (HLC) for causal ordering
//! - Kernel-to-Kernel (K2K) messaging
//! - Queue monitoring and statistics
//! - Resource management
//!
//! # Example
//!
//! ```python
//! import asyncio
//! import ringkernel
//!
//! async def main():
//!     # Create runtime
//!     runtime = await ringkernel.RingKernel.create(backend="cpu")
//!
//!     # Launch kernel
//!     options = ringkernel.LaunchOptions().with_queue_capacity(2048)
//!     kernel = await runtime.launch("processor", options)
//!
//!     # Send/receive messages
//!     await kernel.send(envelope)
//!     response = await kernel.receive_timeout(1.0)
//!
//!     # Cleanup
//!     await kernel.terminate()
//!     await runtime.shutdown()
//!
//! asyncio.run(main())
//! ```

#![warn(missing_docs)]
// Allow clippy false positives from PyO3 macro expansion
#![allow(clippy::useless_conversion)]
// Suppress PyO3 internal cfg warnings
#![allow(unexpected_cfgs)]

mod core;
mod error;

// Feature-gated modules
#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "benchmark")]
mod benchmark;

mod hybrid;
mod resource;

use pyo3::prelude::*;

/// RingKernel Python module.
///
/// This is the main entry point for the Python bindings.
#[pymodule]
fn _ringkernel(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "GPU-native persistent actor model framework")?;

    // Register exceptions
    error::register_exceptions(py, m)?;

    // Register core module
    core::register(m)?;

    // Register hybrid dispatcher
    hybrid::register(m)?;

    // Register resource management
    resource::register(m)?;

    // Feature-gated registrations
    #[cfg(feature = "cuda")]
    cuda::register(m)?;

    #[cfg(feature = "benchmark")]
    benchmark::register(m)?;

    Ok(())
}
