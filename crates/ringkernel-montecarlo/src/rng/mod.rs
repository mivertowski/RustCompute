//! Random number generators for GPU Monte Carlo.
//!
//! This module provides counter-based PRNGs that are suitable for GPU execution:
//! - Stateless (state is the counter value)
//! - Parallel-friendly (each thread can have its own counter)
//! - Reproducible (same counter = same output)

mod philox;
mod traits;

pub use philox::{PhiloxRng, PhiloxState};
pub use traits::GpuRng;
