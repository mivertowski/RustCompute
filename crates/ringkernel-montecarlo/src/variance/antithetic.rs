//! Antithetic variates for variance reduction.
//!
//! Antithetic variates use negatively correlated pairs of samples to reduce
//! variance. If U is uniform on [0,1], then 1-U is also uniform on [0,1]
//! but negatively correlated with U.

use crate::rng::GpuRng;

/// Generate an antithetic pair of uniform variates.
///
/// Returns (u, 1-u) where u is uniform in [0, 1).
/// The pair is negatively correlated, which can reduce variance
/// when the estimator is monotonic.
///
/// # Example
///
/// ```ignore
/// let (u1, u2) = antithetic_pair(&mut state);
/// // Estimate E[f(U)] using both samples
/// let estimate = 0.5 * (f(u1) + f(u2));
/// ```
pub fn antithetic_pair<R: GpuRng>(state: &mut R::State) -> (f32, f32) {
    let u = R::next_uniform(state);
    (u, 1.0 - u)
}

/// Generate an antithetic pair of normal variates.
///
/// Returns (z, -z) where z is standard normal.
pub fn antithetic_normal_pair<R: GpuRng>(state: &mut R::State) -> (f32, f32) {
    let z = R::next_normal(state);
    (z, -z)
}

/// Configuration for antithetic variates estimator.
#[derive(Debug, Clone)]
pub struct AntitheticVariates {
    /// Number of antithetic pairs to generate.
    pub n_pairs: usize,
}

impl AntitheticVariates {
    /// Create new antithetic variates configuration.
    pub fn new(n_pairs: usize) -> Self {
        Self { n_pairs }
    }

    /// Estimate E[f(U)] using antithetic variates.
    ///
    /// The estimator is:
    /// `(1/n) * sum_i (f(U_i) + f(1-U_i)) / 2`
    ///
    /// This has lower variance than naive estimation when f is monotonic.
    pub fn estimate<R: GpuRng, F>(&self, state: &mut R::State, f: F) -> f32
    where
        F: Fn(f32) -> f32,
    {
        let mut sum = 0.0;
        for _ in 0..self.n_pairs {
            let (u1, u2) = antithetic_pair::<R>(state);
            sum += 0.5 * (f(u1) + f(u2));
        }
        sum / self.n_pairs as f32
    }

    /// Estimate E[f(Z)] where Z ~ N(0,1) using antithetic variates.
    pub fn estimate_normal<R: GpuRng, F>(&self, state: &mut R::State, f: F) -> f32
    where
        F: Fn(f32) -> f32,
    {
        let mut sum = 0.0;
        for _ in 0..self.n_pairs {
            let (z1, z2) = antithetic_normal_pair::<R>(state);
            sum += 0.5 * (f(z1) + f(z2));
        }
        sum / self.n_pairs as f32
    }
}

impl Default for AntitheticVariates {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::PhiloxRng;

    #[test]
    fn test_antithetic_pair_range() {
        let mut state = PhiloxRng::seed(42, 0);

        for _ in 0..100 {
            let (u1, u2) = antithetic_pair::<PhiloxRng>(&mut state);
            assert!((0.0..1.0).contains(&u1));
            assert!((0.0..1.0).contains(&u2));
            assert!((u1 + u2 - 1.0).abs() < 1e-6, "u1 + u2 should equal 1");
        }
    }

    #[test]
    fn test_antithetic_normal_pair() {
        let mut state = PhiloxRng::seed(42, 0);

        for _ in 0..100 {
            let (z1, z2) = antithetic_normal_pair::<PhiloxRng>(&mut state);
            assert!((z1 + z2).abs() < 1e-6, "z1 + z2 should equal 0");
        }
    }

    #[test]
    fn test_antithetic_variance_reduction() {
        // Test on f(x) = x^2, estimating E[X^2] where X ~ U[0,1]
        // True value is 1/3
        let n = 10000;

        // Naive estimation
        let mut state = PhiloxRng::seed(123, 0);
        let naive_sum: f32 = (0..n)
            .map(|_| {
                let u = PhiloxRng::next_uniform(&mut state);
                u * u
            })
            .sum();
        let naive_estimate = naive_sum / n as f32;

        // Antithetic estimation
        let mut state = PhiloxRng::seed(123, 0);
        let av = AntitheticVariates::new(n / 2);
        let av_estimate = av.estimate::<PhiloxRng, _>(&mut state, |u| u * u);

        let true_value = 1.0 / 3.0;

        // Both should be close to true value
        assert!(
            (naive_estimate - true_value).abs() < 0.05,
            "Naive estimate {} far from {}",
            naive_estimate,
            true_value
        );
        assert!(
            (av_estimate - true_value).abs() < 0.05,
            "AV estimate {} far from {}",
            av_estimate,
            true_value
        );
    }
}
