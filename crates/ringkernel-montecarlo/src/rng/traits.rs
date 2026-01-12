//! Traits for GPU-compatible random number generators.

use bytemuck::{Pod, Zeroable};

/// Trait for GPU-compatible random number generators.
///
/// Types implementing this trait can be used for Monte Carlo simulation
/// on both CPU and GPU. The state type must be POD for GPU transfer.
///
/// # Design
///
/// GPU RNGs typically use counter-based designs where:
/// - State is just a counter and key (no large state arrays)
/// - Each thread can have independent state by using thread ID in key
/// - Same counter value always produces same output (reproducible)
pub trait GpuRng {
    /// State type (must be POD for GPU transfer).
    type State: Pod + Zeroable + Copy + Default;

    /// Generate next uniform random number in [0, 1).
    fn next_uniform(state: &mut Self::State) -> f32;

    /// Generate next uniform random number in [0, 1) as f64.
    fn next_uniform_f64(state: &mut Self::State) -> f64 {
        Self::next_uniform(state) as f64
    }

    /// Generate next standard normal (Gaussian) variate using Box-Muller transform.
    fn next_normal(state: &mut Self::State) -> f32 {
        // Box-Muller transform
        let u1 = Self::next_uniform(state);
        let u2 = Self::next_uniform(state);

        // Avoid log(0)
        let u1 = if u1 < 1e-10 { 1e-10 } else { u1 };

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;

        r * theta.cos()
    }

    /// Generate a pair of standard normal variates (more efficient than two calls to next_normal).
    fn next_normal_pair(state: &mut Self::State) -> (f32, f32) {
        let u1 = Self::next_uniform(state);
        let u2 = Self::next_uniform(state);

        let u1 = if u1 < 1e-10 { 1e-10 } else { u1 };

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;

        (r * theta.cos(), r * theta.sin())
    }

    /// Generate uniform integer in [0, max).
    fn next_u32(state: &mut Self::State, max: u32) -> u32 {
        let u = Self::next_uniform(state);
        (u * max as f32) as u32
    }

    /// Skip n values (advance counter without generating values).
    fn skip(state: &mut Self::State, n: u64);

    /// Create state from seed and stream ID (for parallel streams).
    fn seed(seed: u64, stream: u64) -> Self::State;
}

/// Extension trait for convenience methods on RNG instances.
#[allow(dead_code)]
pub trait GpuRngExt: GpuRng {
    /// Generate uniform in range [low, high).
    fn uniform_range(state: &mut Self::State, low: f32, high: f32) -> f32 {
        low + Self::next_uniform(state) * (high - low)
    }

    /// Generate normal with mean and standard deviation.
    fn normal(state: &mut Self::State, mean: f32, stddev: f32) -> f32 {
        mean + Self::next_normal(state) * stddev
    }

    /// Generate exponential variate with rate lambda.
    fn exponential(state: &mut Self::State, lambda: f32) -> f32 {
        let u = Self::next_uniform(state);
        let u = if u < 1e-10 { 1e-10 } else { u };
        -(1.0 / lambda) * u.ln()
    }
}

// Blanket implementation
impl<T: GpuRng> GpuRngExt for T {}

#[cfg(test)]
mod tests {
    // Tests are in philox.rs
}
