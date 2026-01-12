//! Control variates for variance reduction.
//!
//! Control variates reduce variance by using a correlated variable with
//! known expectation. If Y is correlated with X and E[Y] is known, then:
//!
//! X* = X - c(Y - E[Y])
//!
//! is an unbiased estimator of E[X] with potentially lower variance.

use crate::rng::GpuRng;

/// Estimate using control variates.
///
/// Given samples of X and Y where E[Y] = mu_y is known,
/// computes the control variate estimator:
///
/// `X* = mean(X) - c * (mean(Y) - mu_y)`
///
/// where c is the optimal coefficient `cov(X,Y) / var(Y)`.
///
/// # Arguments
///
/// * `x_samples` - Samples of the quantity to estimate
/// * `y_samples` - Samples of the control variate (same length as x_samples)
/// * `mu_y` - Known mean of Y
///
/// # Returns
///
/// Tuple of (estimate, optimal_c, variance_reduction_factor)
pub fn control_variate_estimate(
    x_samples: &[f32],
    y_samples: &[f32],
    mu_y: f32,
) -> (f32, f32, f32) {
    let n = x_samples.len();
    assert_eq!(n, y_samples.len(), "Sample arrays must have same length");
    assert!(n > 1, "Need at least 2 samples");

    let n_f = n as f32;

    // Compute means
    let mean_x: f32 = x_samples.iter().sum::<f32>() / n_f;
    let mean_y: f32 = y_samples.iter().sum::<f32>() / n_f;

    // Compute variance of Y and covariance of X, Y
    let mut var_y = 0.0;
    let mut cov_xy = 0.0;
    let mut var_x = 0.0;

    for i in 0..n {
        let dx = x_samples[i] - mean_x;
        let dy = y_samples[i] - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
        cov_xy += dx * dy;
    }

    var_x /= n_f - 1.0;
    var_y /= n_f - 1.0;
    cov_xy /= n_f - 1.0;

    // Optimal coefficient
    let c = if var_y > 1e-10 { cov_xy / var_y } else { 0.0 };

    // Control variate estimate
    let estimate = mean_x - c * (mean_y - mu_y);

    // Variance reduction factor (R² of regression)
    let r_squared = if var_x > 1e-10 && var_y > 1e-10 {
        (cov_xy * cov_xy) / (var_x * var_y)
    } else {
        0.0
    };

    // Variance reduction: Var(X*) = Var(X) * (1 - R²)
    let variance_reduction = 1.0 - r_squared;

    (estimate, c, variance_reduction)
}

/// Configuration for control variates estimator.
#[derive(Debug, Clone)]
pub struct ControlVariates {
    /// Number of samples to generate.
    pub n_samples: usize,
}

impl ControlVariates {
    /// Create new control variates configuration.
    pub fn new(n_samples: usize) -> Self {
        Self { n_samples }
    }

    /// Estimate E[f(U)] using control variate g(U) with known mean mu_g.
    ///
    /// # Arguments
    ///
    /// * `state` - RNG state
    /// * `f` - Function to estimate expectation of
    /// * `g` - Control variate function
    /// * `mu_g` - Known mean E[g(U)]
    ///
    /// # Returns
    ///
    /// Tuple of (estimate, optimal_c, variance_reduction)
    pub fn estimate<R: GpuRng, F, G>(
        &self,
        state: &mut R::State,
        f: F,
        g: G,
        mu_g: f32,
    ) -> (f32, f32, f32)
    where
        F: Fn(f32) -> f32,
        G: Fn(f32) -> f32,
    {
        let mut x_samples = Vec::with_capacity(self.n_samples);
        let mut y_samples = Vec::with_capacity(self.n_samples);

        for _ in 0..self.n_samples {
            let u = R::next_uniform(state);
            x_samples.push(f(u));
            y_samples.push(g(u));
        }

        control_variate_estimate(&x_samples, &y_samples, mu_g)
    }
}

impl Default for ControlVariates {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::PhiloxRng;

    #[test]
    fn test_control_variate_perfect_correlation() {
        // If Y = X, then c = 1 and estimate should equal mu_y
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = x.clone();
        let mu_y = 3.0; // E[Y] = 3

        let (estimate, c, _) = control_variate_estimate(&x, &y, mu_y);

        assert!(
            (c - 1.0).abs() < 1e-6,
            "c should be 1 for perfect correlation"
        );
        assert!(
            (estimate - mu_y).abs() < 1e-6,
            "Estimate should equal mu_y when Y = X"
        );
    }

    #[test]
    fn test_control_variate_reduces_variance() {
        // Estimate E[e^U] using U as control variate with E[U] = 0.5
        let n = 5000;

        // Without control variate
        let mut state = PhiloxRng::seed(42, 0);
        let naive_samples: Vec<f32> = (0..n)
            .map(|_| PhiloxRng::next_uniform(&mut state).exp())
            .collect();
        let naive_mean: f32 = naive_samples.iter().sum::<f32>() / n as f32;

        // With control variate
        let mut state = PhiloxRng::seed(42, 0);
        let cv = ControlVariates::new(n);
        let (cv_estimate, _c, var_reduction) =
            cv.estimate::<PhiloxRng, _, _>(&mut state, |u| u.exp(), |u| u, 0.5);

        // True value: E[e^U] = e - 1 ≈ 1.718
        let true_value = std::f32::consts::E - 1.0;

        // Both should be reasonably close
        assert!(
            (naive_mean - true_value).abs() < 0.1,
            "Naive {} far from {}",
            naive_mean,
            true_value
        );
        assert!(
            (cv_estimate - true_value).abs() < 0.1,
            "CV {} far from {}",
            cv_estimate,
            true_value
        );

        // Variance should be reduced (var_reduction < 1)
        assert!(
            var_reduction < 1.0,
            "Control variate should reduce variance"
        );
    }
}
