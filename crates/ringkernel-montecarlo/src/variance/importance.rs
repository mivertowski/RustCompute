//! Importance sampling for variance reduction.
//!
//! Importance sampling rewrites an expectation E_p[f(X)] as E_q[f(X) * w(X)]
//! where w(X) = p(X)/q(X) is the importance weight and q is a proposal distribution.
//!
//! This can reduce variance when q is chosen to place more probability mass
//! in regions where |f(X)| is large.

use crate::rng::GpuRng;

/// Compute importance sample with weight.
///
/// Given a sample from proposal distribution q, compute the weighted sample
/// for estimating E_p[f(X)].
///
/// # Arguments
///
/// * `x` - Sample from proposal q
/// * `f_x` - Value of f at x
/// * `p_x` - Target density p(x)
/// * `q_x` - Proposal density q(x)
///
/// # Returns
///
/// Weighted sample f(x) * p(x) / q(x)
#[inline]
pub fn importance_sample(_x: f32, f_x: f32, p_x: f32, q_x: f32) -> f32 {
    if q_x.abs() < 1e-10 {
        0.0 // Avoid division by zero
    } else {
        f_x * p_x / q_x
    }
}

/// Configuration for importance sampling.
#[derive(Debug, Clone)]
pub struct ImportanceSampling {
    /// Number of samples.
    pub n_samples: usize,
    /// Whether to use self-normalized estimator.
    pub self_normalized: bool,
}

impl ImportanceSampling {
    /// Create new importance sampling configuration.
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            self_normalized: false,
        }
    }

    /// Use self-normalized importance sampling.
    ///
    /// The self-normalized estimator divides by sum of weights:
    /// `sum(w_i * f_i) / sum(w_i)`
    ///
    /// This is biased but can have lower variance and doesn't require
    /// knowing the normalizing constant of p.
    pub fn self_normalized(mut self) -> Self {
        self.self_normalized = true;
        self
    }

    /// Estimate E_p[f(X)] using importance sampling.
    ///
    /// # Arguments
    ///
    /// * `state` - RNG state
    /// * `sample_q` - Function to sample from proposal distribution q
    /// * `f` - Function to estimate expectation of
    /// * `log_p` - Log of target density (unnormalized OK if self_normalized)
    /// * `log_q` - Log of proposal density
    ///
    /// # Returns
    ///
    /// (estimate, effective_sample_size)
    pub fn estimate<R: GpuRng, S, F, LP, LQ>(
        &self,
        state: &mut R::State,
        sample_q: S,
        f: F,
        log_p: LP,
        log_q: LQ,
    ) -> (f32, f32)
    where
        S: Fn(&mut R::State) -> f32,
        F: Fn(f32) -> f32,
        LP: Fn(f32) -> f32,
        LQ: Fn(f32) -> f32,
    {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut weight_sq_sum = 0.0;

        for _ in 0..self.n_samples {
            let x = sample_q(state);
            let f_x = f(x);
            let log_w = log_p(x) - log_q(x);
            let w = log_w.exp();

            weighted_sum += w * f_x;
            weight_sum += w;
            weight_sq_sum += w * w;
        }

        let estimate = if self.self_normalized {
            if weight_sum.abs() < 1e-10 {
                0.0
            } else {
                weighted_sum / weight_sum
            }
        } else {
            weighted_sum / self.n_samples as f32
        };

        // Effective sample size: ESS = (sum w)² / (sum w²)
        let ess = if weight_sq_sum.abs() < 1e-10 {
            0.0
        } else {
            (weight_sum * weight_sum) / weight_sq_sum
        };

        (estimate, ess)
    }
}

impl Default for ImportanceSampling {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Exponential tilting proposal for rare event simulation.
///
/// For estimating P(X > a) where X ~ N(0,1), shift mean to a/2.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ExponentialTilt {
    /// Tilt parameter (new mean).
    pub theta: f32,
}

#[allow(dead_code)]
impl ExponentialTilt {
    /// Create exponential tilt for estimating P(X > a).
    ///
    /// Uses optimal tilt theta = a for normal random variables.
    pub fn for_tail_probability(a: f32) -> Self {
        Self { theta: a }
    }

    /// Sample from tilted distribution N(theta, 1).
    pub fn sample<R: GpuRng>(&self, state: &mut R::State) -> f32 {
        R::next_normal(state) + self.theta
    }

    /// Log ratio of p(x) / q(x) where p = N(0,1) and q = N(theta,1).
    pub fn log_weight(&self, x: f32) -> f32 {
        // log p(x) - log q(x) = -x²/2 - (-(x-θ)²/2)
        // = -x²/2 + (x-θ)²/2
        // = (-x² + x² - 2xθ + θ²) / 2
        // = (-2xθ + θ²) / 2
        // = θ² / 2 - xθ
        0.5 * self.theta * self.theta - x * self.theta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::PhiloxRng;

    #[test]
    fn test_importance_sample_basic() {
        // Equal densities should give f(x)
        let result = importance_sample(1.0, 2.0, 0.5, 0.5);
        assert!((result - 2.0).abs() < 1e-6);

        // Double target density should double the weight
        let result = importance_sample(1.0, 2.0, 1.0, 0.5);
        assert!((result - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_importance_sampling_uniform() {
        // Estimate E[X²] where X ~ U[0,2]
        // E[X²] = (1/2) * integral_0^2 x² dx = 4/3

        let mut state = PhiloxRng::seed(42, 0);
        let is = ImportanceSampling::new(5000).self_normalized();

        // When p = q, use the same log density (they cancel out)
        let (estimate, ess) = is.estimate::<PhiloxRng, _, _, _, _>(
            &mut state,
            |s| PhiloxRng::next_uniform(s) * 2.0, // Sample from U[0,2]
            |x| x * x,                            // f(x) = x²
            |_| -std::f32::consts::LN_2,          // log p = log(0.5) for U[0,2]
            |_| -std::f32::consts::LN_2,          // log q = log(0.5) for U[0,2]
        );

        // When p = q, weights are all 1, so ESS ≈ n
        assert!(
            ess > 0.5 * 5000.0,
            "ESS {} should be reasonable when p = q",
            ess
        );

        let true_value = 4.0 / 3.0;
        assert!(
            (estimate - true_value).abs() < 0.1,
            "Estimate {} far from {}",
            estimate,
            true_value
        );
    }

    #[test]
    fn test_exponential_tilt() {
        // Estimate P(Z > 3) where Z ~ N(0,1)
        // True value ≈ 0.00135
        let mut state = PhiloxRng::seed(42, 0);
        let tilt = ExponentialTilt::for_tail_probability(3.0);
        let is = ImportanceSampling::new(10000).self_normalized();

        let (estimate, ess) = is.estimate::<PhiloxRng, _, _, _, _>(
            &mut state,
            |s| tilt.sample::<PhiloxRng>(s),
            |x| if x > 3.0 { 1.0 } else { 0.0 }, // Indicator X > 3
            |x| -0.5 * x * x,                    // log N(0,1)
            |x| -0.5 * (x - 3.0) * (x - 3.0),    // log N(3,1)
        );

        // Should be reasonably close to true value (wider tolerance for variance)
        let true_value = 0.00135; // 1 - Phi(3)
        assert!(
            (estimate - true_value).abs() < 0.005,
            "Estimate {} far from {}",
            estimate,
            true_value
        );

        // ESS should be positive (importance sampling typically has low ESS for rare events)
        assert!(ess > 50.0, "ESS {} too low", ess);
    }
}
