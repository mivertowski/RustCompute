//! Benchmark result types.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Workload size information.
#[derive(Debug, Clone, Default)]
pub struct WorkloadSize {
    /// Primary element count (nodes, items, etc.).
    pub elements: usize,
    /// Secondary element count (edges, connections, etc.).
    pub edges: Option<usize>,
    /// Memory footprint in bytes.
    pub bytes: Option<usize>,
}

impl WorkloadSize {
    /// Creates a new workload size.
    #[must_use]
    pub fn new(elements: usize) -> Self {
        Self {
            elements,
            edges: None,
            bytes: None,
        }
    }

    /// Builder method to set edge count.
    #[must_use]
    pub fn with_edges(mut self, edges: usize) -> Self {
        self.edges = Some(edges);
        self
    }

    /// Builder method to set byte count.
    #[must_use]
    pub fn with_bytes(mut self, bytes: usize) -> Self {
        self.bytes = Some(bytes);
        self
    }
}

/// Single benchmark result.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Workload identifier.
    pub workload_id: String,
    /// Size of the workload (element count).
    pub size: usize,
    /// Throughput in operations per second.
    pub throughput_ops: f64,
    /// Total execution time.
    pub total_time: Duration,
    /// Number of iterations (for iterative algorithms).
    pub iterations: Option<usize>,
    /// Whether convergence was achieved.
    pub converged: Option<bool>,
    /// Individual measurement times.
    pub measurement_times: Vec<Duration>,
    /// Custom metrics (workload-specific).
    pub custom_metrics: HashMap<String, f64>,
}

impl BenchmarkResult {
    /// Creates a new benchmark result.
    #[must_use]
    pub fn new(workload_id: impl Into<String>, size: usize, total_time: Duration) -> Self {
        let throughput_ops = if total_time.as_secs_f64() > 0.0 {
            size as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        Self {
            workload_id: workload_id.into(),
            size,
            throughput_ops,
            total_time,
            iterations: None,
            converged: None,
            measurement_times: vec![total_time],
            custom_metrics: HashMap::new(),
        }
    }

    /// Computes result from multiple measurements.
    pub fn from_measurements(
        workload_id: impl Into<String>,
        size: usize,
        iterations: Option<usize>,
        converged: Option<bool>,
        measurements: &[Duration],
    ) -> Self {
        let total_time = if measurements.is_empty() {
            Duration::ZERO
        } else {
            let sum: Duration = measurements.iter().sum();
            sum / measurements.len() as u32
        };

        let throughput_ops = if total_time.as_secs_f64() > 0.0 {
            let ops = if let Some(iters) = iterations {
                size * iters
            } else {
                size
            };
            ops as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        Self {
            workload_id: workload_id.into(),
            size,
            throughput_ops,
            total_time,
            iterations,
            converged,
            measurement_times: measurements.to_vec(),
            custom_metrics: HashMap::new(),
        }
    }

    /// Builder method to add a custom metric.
    #[must_use]
    pub fn with_metric(mut self, key: impl Into<String>, value: f64) -> Self {
        self.custom_metrics.insert(key.into(), value);
        self
    }

    /// Returns throughput in millions of operations per second.
    #[must_use]
    pub fn throughput_mops(&self) -> f64 {
        self.throughput_ops / 1_000_000.0
    }

    /// Returns total time in milliseconds.
    #[must_use]
    pub fn total_time_ms(&self) -> f64 {
        self.total_time.as_secs_f64() * 1000.0
    }

    /// Calculates standard deviation of throughput from measurements.
    #[must_use]
    pub fn throughput_stddev(&self) -> f64 {
        if self.measurement_times.len() < 2 {
            return 0.0;
        }

        let throughputs: Vec<f64> = self
            .measurement_times
            .iter()
            .filter_map(|t| {
                let secs = t.as_secs_f64();
                if secs > 0.0 {
                    Some(self.size as f64 / secs)
                } else {
                    None
                }
            })
            .collect();

        if throughputs.is_empty() {
            return 0.0;
        }

        let mean = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let variance =
            throughputs.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / throughputs.len() as f64;

        variance.sqrt()
    }
}

/// Baseline for regression comparison.
#[derive(Debug, Clone)]
pub struct BenchmarkBaseline {
    /// Results by (workload_id, size).
    pub results: HashMap<(String, usize), BenchmarkResult>,
    /// Baseline version/commit.
    pub version: String,
    /// When baseline was recorded.
    pub timestamp: SystemTime,
}

impl BenchmarkBaseline {
    /// Creates a new baseline from results.
    #[must_use]
    pub fn from_results(results: &[BenchmarkResult], version: &str) -> Self {
        let mut map = HashMap::new();
        for result in results {
            map.insert((result.workload_id.clone(), result.size), result.clone());
        }
        Self {
            results: map,
            version: version.to_string(),
            timestamp: SystemTime::now(),
        }
    }

    /// Gets baseline result for a workload and size.
    #[must_use]
    pub fn get(&self, workload_id: &str, size: usize) -> Option<&BenchmarkResult> {
        self.results.get(&(workload_id.to_string(), size))
    }

    /// Returns the number of baseline results.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Returns whether the baseline is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_size() {
        let size = WorkloadSize::new(1000)
            .with_edges(5000)
            .with_bytes(1024 * 1024);

        assert_eq!(size.elements, 1000);
        assert_eq!(size.edges, Some(5000));
        assert_eq!(size.bytes, Some(1024 * 1024));
    }

    #[test]
    fn test_benchmark_result_new() {
        let result = BenchmarkResult::new("test", 1000, Duration::from_millis(100));

        assert_eq!(result.workload_id, "test");
        assert_eq!(result.size, 1000);
        assert!(result.throughput_ops > 0.0);
        assert_eq!(result.total_time, Duration::from_millis(100));
    }

    #[test]
    fn test_benchmark_result_from_measurements() {
        let measurements = vec![
            Duration::from_millis(100),
            Duration::from_millis(110),
            Duration::from_millis(90),
        ];

        let result =
            BenchmarkResult::from_measurements("test", 1000, Some(50), Some(true), &measurements);

        assert_eq!(result.workload_id, "test");
        assert_eq!(result.iterations, Some(50));
        assert_eq!(result.converged, Some(true));
        assert!(result.throughput_ops > 0.0);
    }

    #[test]
    fn test_benchmark_result_throughput_mops() {
        let result = BenchmarkResult::new("test", 1_000_000, Duration::from_secs(1));
        assert!((result.throughput_mops() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_benchmark_baseline() {
        let results = vec![
            BenchmarkResult::new("workload_a", 1000, Duration::from_millis(100)),
            BenchmarkResult::new("workload_b", 2000, Duration::from_millis(200)),
        ];

        let baseline = BenchmarkBaseline::from_results(&results, "v1.0");

        assert_eq!(baseline.len(), 2);
        assert!(!baseline.is_empty());
        assert!(baseline.get("workload_a", 1000).is_some());
        assert!(baseline.get("workload_a", 2000).is_none());
    }
}
