//! Benchmark configuration.

use std::time::Duration;

/// Benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations.
    pub warmup_iterations: usize,
    /// Number of measurement iterations.
    pub measurement_iterations: usize,
    /// Workload sizes to test.
    pub workload_sizes: Vec<usize>,
    /// Convergence threshold for iterative algorithms.
    pub convergence_threshold: f64,
    /// Maximum iterations for convergence.
    pub max_iterations: usize,
    /// Regression threshold (e.g., 0.10 = 10% slowdown triggers warning).
    pub regression_threshold: f64,
    /// Timeout for individual benchmark runs.
    pub timeout: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            measurement_iterations: 5,
            workload_sizes: vec![10_000, 50_000, 100_000],
            convergence_threshold: 1e-6,
            max_iterations: 100,
            regression_threshold: 0.10,
            timeout: Duration::from_secs(60),
        }
    }
}

impl BenchmarkConfig {
    /// Creates a quick benchmark configuration for fast feedback.
    #[must_use]
    pub fn quick() -> Self {
        Self {
            warmup_iterations: 1,
            measurement_iterations: 3,
            workload_sizes: vec![10_000, 50_000],
            convergence_threshold: 1e-5,
            max_iterations: 50,
            regression_threshold: 0.15,
            timeout: Duration::from_secs(30),
        }
    }

    /// Creates a comprehensive benchmark configuration for thorough analysis.
    #[must_use]
    pub fn comprehensive() -> Self {
        Self {
            warmup_iterations: 5,
            measurement_iterations: 10,
            workload_sizes: vec![10_000, 25_000, 50_000, 75_000, 100_000, 150_000],
            convergence_threshold: 1e-6,
            max_iterations: 200,
            regression_threshold: 0.05,
            timeout: Duration::from_secs(300),
        }
    }

    /// Creates a CI-optimized configuration.
    #[must_use]
    pub fn ci() -> Self {
        Self {
            warmup_iterations: 2,
            measurement_iterations: 5,
            workload_sizes: vec![10_000, 50_000],
            convergence_threshold: 1e-5,
            max_iterations: 100,
            regression_threshold: 0.10,
            timeout: Duration::from_secs(120),
        }
    }

    /// Builder method to set warmup iterations.
    #[must_use]
    pub fn with_warmup(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    /// Builder method to set measurement iterations.
    #[must_use]
    pub fn with_measurements(mut self, iterations: usize) -> Self {
        self.measurement_iterations = iterations;
        self
    }

    /// Builder method to set workload sizes.
    #[must_use]
    pub fn with_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.workload_sizes = sizes;
        self
    }

    /// Builder method to set regression threshold.
    #[must_use]
    pub fn with_regression_threshold(mut self, threshold: f64) -> Self {
        self.regression_threshold = threshold;
        self
    }

    /// Builder method to set timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = BenchmarkConfig::default();
        assert!(!config.workload_sizes.is_empty());
        assert!(config.warmup_iterations > 0);
        assert!(config.measurement_iterations > 0);
    }

    #[test]
    fn test_config_quick() {
        let config = BenchmarkConfig::quick();
        assert!(config.measurement_iterations < BenchmarkConfig::default().measurement_iterations);
    }

    #[test]
    fn test_config_comprehensive() {
        let config = BenchmarkConfig::comprehensive();
        assert!(config.measurement_iterations > BenchmarkConfig::default().measurement_iterations);
    }

    #[test]
    fn test_config_builder() {
        let config = BenchmarkConfig::default()
            .with_warmup(10)
            .with_measurements(20)
            .with_sizes(vec![1000, 2000])
            .with_regression_threshold(0.05);

        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.measurement_iterations, 20);
        assert_eq!(config.workload_sizes, vec![1000, 2000]);
        assert!((config.regression_threshold - 0.05).abs() < f64::EPSILON);
    }
}
