//! Benchmarkable trait and workload configuration.

use super::result::{BenchmarkResult, WorkloadSize};

/// Configuration for a single workload execution.
#[derive(Debug, Clone)]
pub struct WorkloadConfig {
    /// Size of the workload (element count).
    pub size: usize,
    /// Convergence threshold for iterative algorithms.
    pub convergence_threshold: f64,
    /// Maximum iterations for convergence.
    pub max_iterations: usize,
    /// Custom parameters (workload-specific).
    pub custom_params: std::collections::HashMap<String, String>,
}

impl Default for WorkloadConfig {
    fn default() -> Self {
        Self {
            size: 10_000,
            convergence_threshold: 1e-6,
            max_iterations: 100,
            custom_params: std::collections::HashMap::new(),
        }
    }
}

impl WorkloadConfig {
    /// Creates a new workload configuration.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            size,
            ..Default::default()
        }
    }

    /// Builder method to set convergence threshold.
    #[must_use]
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Builder method to set max iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Builder method to add a custom parameter.
    #[must_use]
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_params.insert(key.into(), value.into());
        self
    }
}

/// Trait for workloads that can be benchmarked.
///
/// Implement this trait for any workload you want to benchmark.
/// The benchmark suite will call `execute` multiple times with
/// different configurations.
///
/// # Example
///
/// ```ignore
/// use ringkernel_core::benchmark::{Benchmarkable, WorkloadConfig, BenchmarkResult, WorkloadSize};
/// use std::time::Duration;
///
/// struct PageRankWorkload {
///     num_nodes: usize,
///     num_edges: usize,
/// }
///
/// impl Benchmarkable for PageRankWorkload {
///     fn name(&self) -> &str {
///         "PageRank"
///     }
///
///     fn code(&self) -> &str {
///         "PR"
///     }
///
///     fn execute(&self, config: &WorkloadConfig) -> BenchmarkResult {
///         let start = std::time::Instant::now();
///         // ... run algorithm ...
///         let elapsed = start.elapsed();
///
///         BenchmarkResult {
///             workload_id: self.name().to_string(),
///             size: config.size,
///             throughput_ops: (self.num_edges as f64) / elapsed.as_secs_f64(),
///             total_time: elapsed,
///             iterations: Some(42),
///             converged: Some(true),
///             measurement_times: vec![elapsed],
///             custom_metrics: Default::default(),
///         }
///     }
///
///     fn workload_size(&self) -> WorkloadSize {
///         WorkloadSize {
///             elements: self.num_nodes,
///             edges: Some(self.num_edges),
///             bytes: None,
///         }
///     }
/// }
/// ```
pub trait Benchmarkable: Send + Sync {
    /// Returns the human-readable name of the workload.
    fn name(&self) -> &str;

    /// Returns a short code for tables (e.g., "PR" for PageRank).
    fn code(&self) -> &str;

    /// Executes the workload and returns benchmark results.
    fn execute(&self, config: &WorkloadConfig) -> BenchmarkResult;

    /// Returns the workload size information.
    ///
    /// Override this if your workload has additional size metrics.
    fn workload_size(&self) -> WorkloadSize {
        WorkloadSize::default()
    }

    /// Returns whether this workload requires GPU.
    fn requires_gpu(&self) -> bool {
        true
    }

    /// Returns an optional description of the workload.
    fn description(&self) -> Option<&str> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct TestWorkload;

    impl Benchmarkable for TestWorkload {
        fn name(&self) -> &str {
            "Test"
        }

        fn code(&self) -> &str {
            "TST"
        }

        fn execute(&self, config: &WorkloadConfig) -> BenchmarkResult {
            BenchmarkResult {
                workload_id: self.name().to_string(),
                size: config.size,
                throughput_ops: 1000.0,
                total_time: Duration::from_millis(10),
                iterations: Some(10),
                converged: Some(true),
                measurement_times: vec![Duration::from_millis(10)],
                custom_metrics: Default::default(),
            }
        }
    }

    #[test]
    fn test_workload_config_default() {
        let config = WorkloadConfig::default();
        assert!(config.size > 0);
    }

    #[test]
    fn test_workload_config_builder() {
        let config = WorkloadConfig::new(1000)
            .with_convergence_threshold(1e-5)
            .with_max_iterations(50)
            .with_param("damping", "0.85");

        assert_eq!(config.size, 1000);
        assert!((config.convergence_threshold - 1e-5).abs() < f64::EPSILON);
        assert_eq!(config.max_iterations, 50);
        assert_eq!(
            config.custom_params.get("damping"),
            Some(&"0.85".to_string())
        );
    }

    #[test]
    fn test_benchmarkable_trait() {
        let workload = TestWorkload;
        assert_eq!(workload.name(), "Test");
        assert_eq!(workload.code(), "TST");
        assert!(workload.requires_gpu());

        let result = workload.execute(&WorkloadConfig::default());
        assert_eq!(result.workload_id, "Test");
        assert!(result.throughput_ops > 0.0);
    }
}
