//! Variance reduction techniques for Monte Carlo simulation.
//!
//! This module provides several variance reduction methods:
//!
//! - **Antithetic Variates**: Use negatively correlated pairs to reduce variance
//! - **Control Variates**: Use correlated variables with known mean
//! - **Importance Sampling**: Reweight samples from a different distribution

mod antithetic;
mod control;
mod importance;

pub use antithetic::{antithetic_pair, AntitheticVariates};
pub use control::{control_variate_estimate, ControlVariates};
pub use importance::{importance_sample, ImportanceSampling};
