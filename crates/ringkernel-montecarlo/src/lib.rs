//! GPU-accelerated Monte Carlo primitives for variance reduction.
//!
//! This crate provides variance reduction techniques for Monte Carlo simulation
//! that can be accelerated on GPU via RingKernel.
//!
//! # Features
//!
//! - **Counter-based PRNGs**: Philox and other stateless generators suitable for GPU
//! - **Variance Reduction**: Antithetic variates, control variates, importance sampling
//! - **GPU Compatibility**: All types are designed for zero-copy GPU transfer
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_montecarlo::prelude::*;
//!
//! // Create a Philox-based RNG
//! let mut rng = PhiloxRng::new(0, 42);
//!
//! // Generate uniform random numbers
//! let u: f32 = rng.next_uniform();
//!
//! // Generate normal variates
//! let z: f32 = rng.next_normal();
//!
//! // Antithetic variates for variance reduction
//! let (u1, u2) = antithetic_pair(&mut rng);
//! // u1 and u2 are negatively correlated
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod rng;
pub mod variance;

/// GPU-accelerated implementations (requires `cuda` feature).
#[cfg(feature = "cuda")]
pub mod gpu;

/// Error types for Monte Carlo operations.
#[derive(Debug, thiserror::Error)]
pub enum MonteCarloError {
    /// Invalid parameter error.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    /// Numerical error (e.g., division by zero, overflow).
    #[error("Numerical error: {0}")]
    NumericalError(String),
    /// RNG state error.
    #[error("RNG error: {0}")]
    RngError(String),
}

/// Result type for Monte Carlo operations.
pub type Result<T> = std::result::Result<T, MonteCarloError>;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::rng::{GpuRng, PhiloxRng, PhiloxState};
    pub use crate::variance::{
        antithetic_pair, control_variate_estimate, importance_sample, AntitheticVariates,
        ControlVariates, ImportanceSampling,
    };
    pub use crate::{MonteCarloError, Result};

    // GPU types (when cuda feature is enabled)
    #[cfg(feature = "cuda")]
    pub use crate::gpu::{
        GpuAntitheticVariates, GpuImportanceSampling, GpuMonteCarloError, GpuPhiloxRng,
    };
}

// Re-exports
pub use rng::{GpuRng, PhiloxRng, PhiloxState};
pub use variance::{AntitheticVariates, ControlVariates, ImportanceSampling};
