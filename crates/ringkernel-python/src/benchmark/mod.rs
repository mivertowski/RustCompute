//! Benchmark Python bindings for RingKernel.
//!
//! Provides benchmarking infrastructure with regression detection and statistical analysis.

use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;
use std::time::Duration;

use ringkernel_core::benchmark::{
    BenchmarkBaseline, BenchmarkConfig as RustBenchmarkConfig,
    BenchmarkResult as RustBenchmarkResult, BenchmarkSuite as RustBenchmarkSuite,
    ConfidenceInterval, DetailedStatistics, RegressionEntry,
    RegressionReport as RustRegressionReport, RegressionStatus,
    ScalingMetrics as RustScalingMetrics, WorkloadConfig as RustWorkloadConfig,
};

#[allow(unused_imports)]
use crate::error::PyRingKernelError;

// =============================================================================
// Configuration Types
// =============================================================================

/// Configuration for benchmark execution.
#[pyclass]
#[derive(Clone)]
pub struct PyBenchmarkConfig {
    inner: RustBenchmarkConfig,
}

#[pymethods]
impl PyBenchmarkConfig {
    /// Create a new benchmark configuration.
    ///
    /// Args:
    ///     warmup_iterations: Number of warmup runs.
    ///     measurement_iterations: Number of measurement runs.
    ///     regression_threshold: Threshold for regression detection (0.10 = 10%).
    #[new]
    #[pyo3(signature = (warmup_iterations=5, measurement_iterations=10, regression_threshold=0.10))]
    fn new(
        warmup_iterations: usize,
        measurement_iterations: usize,
        regression_threshold: f64,
    ) -> Self {
        Self {
            inner: RustBenchmarkConfig {
                warmup_iterations,
                measurement_iterations,
                regression_threshold,
                ..Default::default()
            },
        }
    }

    /// Quick configuration for fast feedback (1 warmup, 3 measurements).
    #[classmethod]
    fn quick(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustBenchmarkConfig::quick(),
        }
    }

    /// Comprehensive configuration for thorough analysis (5 warmup, 10 measurements).
    #[classmethod]
    fn comprehensive(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustBenchmarkConfig::comprehensive(),
        }
    }

    /// CI-optimized configuration (2 warmup, 5 measurements).
    #[classmethod]
    fn ci(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustBenchmarkConfig::ci(),
        }
    }

    /// Set warmup iterations.
    fn with_warmup(&self, iterations: usize) -> Self {
        Self {
            inner: self.inner.clone().with_warmup(iterations),
        }
    }

    /// Set measurement iterations.
    fn with_measurements(&self, iterations: usize) -> Self {
        Self {
            inner: self.inner.clone().with_measurements(iterations),
        }
    }

    /// Set workload sizes.
    fn with_sizes(&self, sizes: Vec<usize>) -> Self {
        Self {
            inner: self.inner.clone().with_sizes(sizes),
        }
    }

    /// Set regression threshold (e.g., 0.10 = 10%).
    fn with_regression_threshold(&self, threshold: f64) -> Self {
        Self {
            inner: self.inner.clone().with_regression_threshold(threshold),
        }
    }

    /// Set timeout in seconds.
    fn with_timeout_secs(&self, seconds: f64) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_timeout(Duration::from_secs_f64(seconds)),
        }
    }

    /// Number of warmup iterations.
    #[getter]
    fn warmup_iterations(&self) -> usize {
        self.inner.warmup_iterations
    }

    /// Number of measurement iterations.
    #[getter]
    fn measurement_iterations(&self) -> usize {
        self.inner.measurement_iterations
    }

    /// Regression detection threshold.
    #[getter]
    fn regression_threshold(&self) -> f64 {
        self.inner.regression_threshold
    }

    /// Workload sizes.
    #[getter]
    fn workload_sizes(&self) -> Vec<usize> {
        self.inner.workload_sizes.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "BenchmarkConfig(warmup={}, measurements={}, threshold={:.0}%)",
            self.inner.warmup_iterations,
            self.inner.measurement_iterations,
            self.inner.regression_threshold * 100.0
        )
    }
}

/// Configuration for individual workload execution.
#[pyclass]
#[derive(Clone)]
pub struct PyWorkloadConfig {
    inner: RustWorkloadConfig,
}

#[pymethods]
impl PyWorkloadConfig {
    /// Create a new workload configuration.
    ///
    /// Args:
    ///     size: Element count for this workload.
    #[new]
    fn new(size: usize) -> Self {
        Self {
            inner: RustWorkloadConfig::new(size),
        }
    }

    /// Set convergence threshold.
    fn with_convergence_threshold(&self, threshold: f64) -> Self {
        Self {
            inner: self.inner.clone().with_convergence_threshold(threshold),
        }
    }

    /// Set maximum iterations.
    fn with_max_iterations(&self, max: usize) -> Self {
        Self {
            inner: self.inner.clone().with_max_iterations(max),
        }
    }

    /// Add custom parameter.
    fn with_param(&self, key: &str, value: &str) -> Self {
        Self {
            inner: self.inner.clone().with_param(key, value),
        }
    }

    /// Workload size.
    #[getter]
    fn size(&self) -> usize {
        self.inner.size
    }

    /// Convergence threshold.
    #[getter]
    fn convergence_threshold(&self) -> f64 {
        self.inner.convergence_threshold
    }

    /// Maximum iterations.
    #[getter]
    fn max_iterations(&self) -> usize {
        self.inner.max_iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "WorkloadConfig(size={}, max_iter={})",
            self.inner.size, self.inner.max_iterations
        )
    }
}

// =============================================================================
// Result Types
// =============================================================================

/// Single benchmark execution result.
#[pyclass]
#[derive(Clone)]
pub struct PyBenchmarkResult {
    inner: RustBenchmarkResult,
}

#[pymethods]
impl PyBenchmarkResult {
    /// Create a new benchmark result.
    ///
    /// Args:
    ///     workload_id: Workload identifier.
    ///     size: Workload size.
    ///     total_time_secs: Total execution time in seconds.
    #[new]
    fn new(workload_id: &str, size: usize, total_time_secs: f64) -> Self {
        Self {
            inner: RustBenchmarkResult::new(
                workload_id,
                size,
                Duration::from_secs_f64(total_time_secs),
            ),
        }
    }

    /// Create from measurement times.
    ///
    /// Args:
    ///     workload_id: Workload identifier.
    ///     size: Workload size.
    ///     measurement_times_secs: List of measurement times in seconds.
    ///     iterations: Number of iterations (if iterative algorithm).
    ///     converged: Whether convergence was achieved.
    #[classmethod]
    #[pyo3(signature = (workload_id, size, measurement_times_secs, iterations=None, converged=None))]
    fn from_measurements(
        _cls: &Bound<'_, PyType>,
        workload_id: &str,
        size: usize,
        measurement_times_secs: Vec<f64>,
        iterations: Option<usize>,
        converged: Option<bool>,
    ) -> Self {
        let durations: Vec<Duration> = measurement_times_secs
            .into_iter()
            .map(Duration::from_secs_f64)
            .collect();
        Self {
            inner: RustBenchmarkResult::from_measurements(
                workload_id,
                size,
                iterations,
                converged,
                &durations,
            ),
        }
    }

    /// Add custom metric.
    fn with_metric(&self, key: &str, value: f64) -> Self {
        Self {
            inner: self.inner.clone().with_metric(key, value),
        }
    }

    /// Workload identifier.
    #[getter]
    fn workload_id(&self) -> &str {
        &self.inner.workload_id
    }

    /// Workload size.
    #[getter]
    fn size(&self) -> usize {
        self.inner.size
    }

    /// Operations per second.
    #[getter]
    fn throughput_ops(&self) -> f64 {
        self.inner.throughput_ops
    }

    /// Total time in seconds.
    #[getter]
    fn total_time_secs(&self) -> f64 {
        self.inner.total_time.as_secs_f64()
    }

    /// Number of iterations (if applicable).
    #[getter]
    fn iterations(&self) -> Option<usize> {
        self.inner.iterations
    }

    /// Whether convergence was achieved (if applicable).
    #[getter]
    fn converged(&self) -> Option<bool> {
        self.inner.converged
    }

    /// Throughput in millions of ops/sec.
    fn throughput_mops(&self) -> f64 {
        self.inner.throughput_mops()
    }

    /// Total time in milliseconds.
    fn total_time_ms(&self) -> f64 {
        self.inner.total_time_ms()
    }

    /// Standard deviation of throughput.
    fn throughput_stddev(&self) -> f64 {
        self.inner.throughput_stddev()
    }

    /// Custom metrics dictionary.
    fn custom_metrics(&self) -> HashMap<String, f64> {
        self.inner.custom_metrics.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "BenchmarkResult(workload='{}', size={}, throughput={:.2}Mops/s, time={:.2}ms)",
            self.inner.workload_id,
            self.inner.size,
            self.throughput_mops(),
            self.total_time_ms()
        )
    }
}

impl From<RustBenchmarkResult> for PyBenchmarkResult {
    fn from(inner: RustBenchmarkResult) -> Self {
        Self { inner }
    }
}

impl From<PyBenchmarkResult> for RustBenchmarkResult {
    fn from(py: PyBenchmarkResult) -> Self {
        py.inner
    }
}

// =============================================================================
// Statistics Types
// =============================================================================

/// Statistical confidence interval.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyConfidenceInterval {
    /// Lower bound.
    #[pyo3(get)]
    pub lower: f64,
    /// Upper bound.
    #[pyo3(get)]
    pub upper: f64,
    /// Confidence level (e.g., 0.95 = 95%).
    #[pyo3(get)]
    pub confidence_level: f64,
}

#[pymethods]
impl PyConfidenceInterval {
    /// Compute 95% confidence interval from values.
    #[classmethod]
    fn from_values(_cls: &Bound<'_, PyType>, values: Vec<f64>) -> Self {
        ConfidenceInterval::from_values(&values).into()
    }

    /// Compute confidence interval with custom confidence level.
    #[classmethod]
    fn from_values_with_confidence(
        _cls: &Bound<'_, PyType>,
        values: Vec<f64>,
        confidence_level: f64,
    ) -> Self {
        ConfidenceInterval::from_values_with_confidence(&values, confidence_level).into()
    }

    /// Width of confidence interval.
    fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Midpoint of confidence interval.
    fn midpoint(&self) -> f64 {
        (self.lower + self.upper) / 2.0
    }

    fn __repr__(&self) -> String {
        format!(
            "ConfidenceInterval([{:.4}, {:.4}], {:.0}%)",
            self.lower,
            self.upper,
            self.confidence_level * 100.0
        )
    }
}

impl From<ConfidenceInterval> for PyConfidenceInterval {
    fn from(ci: ConfidenceInterval) -> Self {
        Self {
            lower: ci.lower,
            upper: ci.upper,
            confidence_level: ci.confidence_level,
        }
    }
}

/// Detailed statistics with percentiles.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyDetailedStatistics {
    /// Number of samples.
    #[pyo3(get)]
    pub count: usize,
    /// Mean value.
    #[pyo3(get)]
    pub mean: f64,
    /// Standard deviation.
    #[pyo3(get)]
    pub std_dev: f64,
    /// Minimum value.
    #[pyo3(get)]
    pub min: f64,
    /// Maximum value.
    #[pyo3(get)]
    pub max: f64,
    /// 50th percentile (median).
    #[pyo3(get)]
    pub median: f64,
    /// 5th percentile.
    #[pyo3(get)]
    pub p5: f64,
    /// 25th percentile.
    #[pyo3(get)]
    pub p25: f64,
    /// 75th percentile.
    #[pyo3(get)]
    pub p75: f64,
    /// 95th percentile.
    #[pyo3(get)]
    pub p95: f64,
    /// 99th percentile.
    #[pyo3(get)]
    pub p99: f64,
}

#[pymethods]
impl PyDetailedStatistics {
    /// Compute statistics from values.
    #[classmethod]
    fn from_values(_cls: &Bound<'_, PyType>, values: Vec<f64>) -> Self {
        DetailedStatistics::from_values(&values).into()
    }

    /// Coefficient of variation (stddev / mean).
    fn coefficient_of_variation(&self) -> f64 {
        if self.mean == 0.0 {
            0.0
        } else {
            self.std_dev / self.mean
        }
    }

    /// Interquartile range (p75 - p25).
    fn iqr(&self) -> f64 {
        self.p75 - self.p25
    }

    fn __repr__(&self) -> String {
        format!(
            "DetailedStatistics(mean={:.4}, stddev={:.4}, median={:.4}, n={})",
            self.mean, self.std_dev, self.median, self.count
        )
    }
}

impl From<DetailedStatistics> for PyDetailedStatistics {
    fn from(s: DetailedStatistics) -> Self {
        Self {
            count: s.count,
            mean: s.mean,
            std_dev: s.std_dev,
            min: s.min,
            max: s.max,
            median: s.median,
            p5: s.p5,
            p25: s.p25,
            p75: s.p75,
            p95: s.p95,
            p99: s.p99,
        }
    }
}

/// Scaling behavior analysis.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyScalingMetrics {
    /// Scaling exponent (log-log slope).
    #[pyo3(get)]
    pub exponent: f64,
    /// Coefficient of determination (R^2).
    #[pyo3(get)]
    pub r_squared: f64,
    /// Number of data points.
    #[pyo3(get)]
    pub data_points: usize,
}

#[pymethods]
impl PyScalingMetrics {
    /// Compute scaling metrics from sizes and throughputs.
    #[classmethod]
    fn from_sizes_and_throughputs(
        _cls: &Bound<'_, PyType>,
        sizes: Vec<usize>,
        throughputs: Vec<f64>,
    ) -> Self {
        RustScalingMetrics::from_sizes_and_throughputs(&sizes, &throughputs).into()
    }

    /// Qualitative assessment of scaling quality.
    fn scaling_quality(&self) -> &'static str {
        if self.r_squared < 0.5 {
            "Poor"
        } else if self.exponent > 0.9 {
            "Excellent"
        } else if self.exponent > 0.7 {
            "Good"
        } else if self.exponent > 0.5 {
            "Fair"
        } else {
            "Sub-linear"
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ScalingMetrics(exponent={:.3}, R²={:.3}, quality='{}')",
            self.exponent,
            self.r_squared,
            self.scaling_quality()
        )
    }
}

impl From<RustScalingMetrics> for PyScalingMetrics {
    fn from(s: RustScalingMetrics) -> Self {
        Self {
            exponent: s.exponent,
            r_squared: s.r_squared,
            data_points: s.data_points,
        }
    }
}

// =============================================================================
// Regression Detection Types
// =============================================================================

/// Regression status classification.
#[pyclass(frozen, eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyRegressionStatus {
    /// Performance improved significantly.
    Improved = 0,
    /// Performance unchanged (within threshold).
    Unchanged = 1,
    /// Performance regressed significantly.
    Regressed = 2,
}

#[pymethods]
impl PyRegressionStatus {
    /// Get symbol ("+", "=", "-").
    fn symbol(&self) -> &'static str {
        match self {
            Self::Improved => "+",
            Self::Unchanged => "=",
            Self::Regressed => "-",
        }
    }

    /// Get text ("IMPROVED", "UNCHANGED", "REGRESSED").
    fn text(&self) -> &'static str {
        match self {
            Self::Improved => "IMPROVED",
            Self::Unchanged => "UNCHANGED",
            Self::Regressed => "REGRESSED",
        }
    }

    fn __repr__(&self) -> String {
        format!("RegressionStatus.{}", self.text())
    }
}

impl From<RegressionStatus> for PyRegressionStatus {
    fn from(s: RegressionStatus) -> Self {
        match s {
            RegressionStatus::Improved => Self::Improved,
            RegressionStatus::Unchanged => Self::Unchanged,
            RegressionStatus::Regressed => Self::Regressed,
        }
    }
}

/// Single regression comparison entry.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyRegressionEntry {
    /// Workload identifier.
    #[pyo3(get)]
    pub workload_id: String,
    /// Workload size.
    #[pyo3(get)]
    pub size: usize,
    /// Current throughput (ops/sec).
    #[pyo3(get)]
    pub current_throughput: f64,
    /// Baseline throughput (ops/sec).
    #[pyo3(get)]
    pub baseline_throughput: f64,
    /// Percent change (positive = improvement).
    #[pyo3(get)]
    pub percent_change: f64,
    /// Regression status.
    #[pyo3(get)]
    pub status: PyRegressionStatus,
}

#[pymethods]
impl PyRegressionEntry {
    fn __repr__(&self) -> String {
        format!(
            "RegressionEntry(workload='{}', size={}, change={:+.1}%, status={})",
            self.workload_id,
            self.size,
            self.percent_change * 100.0,
            self.status.text()
        )
    }
}

impl From<&RegressionEntry> for PyRegressionEntry {
    fn from(e: &RegressionEntry) -> Self {
        Self {
            workload_id: e.workload_id.clone(),
            size: e.size,
            current_throughput: e.current_throughput,
            baseline_throughput: e.baseline_throughput,
            percent_change: e.percent_change,
            status: e.status.into(),
        }
    }
}

/// Complete regression analysis report.
#[pyclass]
#[derive(Clone)]
pub struct PyRegressionReport {
    inner: RustRegressionReport,
}

#[pymethods]
impl PyRegressionReport {
    /// Check if any regressions were detected.
    fn has_regressions(&self) -> bool {
        self.inner.has_regressions()
    }

    /// Total number of comparisons.
    fn total_comparisons(&self) -> usize {
        self.inner.total_comparisons()
    }

    /// Number of regressions.
    #[getter]
    fn regression_count(&self) -> usize {
        self.inner.regression_count
    }

    /// Number of improvements.
    #[getter]
    fn improvement_count(&self) -> usize {
        self.inner.improvement_count
    }

    /// Number unchanged.
    #[getter]
    fn unchanged_count(&self) -> usize {
        self.inner.unchanged_count
    }

    /// Threshold used for detection.
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    /// Overall status.
    #[getter]
    fn overall_status(&self) -> PyRegressionStatus {
        self.inner.overall_status.into()
    }

    /// Get all comparison entries.
    fn entries(&self) -> Vec<PyRegressionEntry> {
        self.inner.entries.iter().map(Into::into).collect()
    }

    /// Get the worst regression.
    fn worst_regression(&self) -> Option<PyRegressionEntry> {
        self.inner.worst_regression().map(Into::into)
    }

    /// Get the best improvement.
    fn best_improvement(&self) -> Option<PyRegressionEntry> {
        self.inner.best_improvement().map(Into::into)
    }

    /// Generate summary text.
    fn summary(&self) -> String {
        self.inner.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "RegressionReport(regressions={}, improvements={}, unchanged={}, status={})",
            self.inner.regression_count,
            self.inner.improvement_count,
            self.inner.unchanged_count,
            self.inner.overall_status.text()
        )
    }
}

impl From<RustRegressionReport> for PyRegressionReport {
    fn from(inner: RustRegressionReport) -> Self {
        Self { inner }
    }
}

// =============================================================================
// Suite Type
// =============================================================================

/// Main benchmark suite for running and tracking benchmarks.
#[pyclass]
pub struct PyBenchmarkSuite {
    inner: RustBenchmarkSuite,
}

#[pymethods]
impl PyBenchmarkSuite {
    /// Create a new benchmark suite.
    ///
    /// Args:
    ///     config: Benchmark configuration.
    #[new]
    fn new(config: &PyBenchmarkConfig) -> Self {
        Self {
            inner: RustBenchmarkSuite::new(config.inner.clone()),
        }
    }

    /// Create with default configuration.
    #[classmethod]
    fn with_defaults(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustBenchmarkSuite::with_defaults(),
        }
    }

    /// Get configuration.
    fn config(&self) -> PyBenchmarkConfig {
        PyBenchmarkConfig {
            inner: self.inner.config().clone(),
        }
    }

    /// Get all collected results.
    fn results(&self) -> Vec<PyBenchmarkResult> {
        self.inner
            .results()
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Add a result to the suite.
    fn add_result(&mut self, result: &PyBenchmarkResult) {
        self.inner.add_result(result.inner.clone());
    }

    /// Create baseline from current results.
    ///
    /// Args:
    ///     version: Version string for the baseline.
    fn create_baseline(&self, version: &str) -> PyBenchmarkBaseline {
        PyBenchmarkBaseline {
            inner: self.inner.create_baseline(version),
        }
    }

    /// Set baseline for regression comparison.
    fn set_baseline(&mut self, baseline: &PyBenchmarkBaseline) {
        self.inner.set_baseline(baseline.inner.clone());
    }

    /// Compare results to baseline.
    fn compare_to_baseline(&self) -> Option<PyRegressionReport> {
        self.inner.compare_to_baseline().map(Into::into)
    }

    /// Compute scaling metrics for a workload.
    fn scaling_metrics_for(&self, workload_id: &str) -> PyScalingMetrics {
        self.inner.scaling_metrics_for(workload_id).into()
    }

    /// Generate Markdown report.
    fn generate_markdown_report(&self) -> String {
        self.inner.generate_markdown_report()
    }

    /// Generate JSON export.
    fn generate_json_export(&self) -> String {
        self.inner.generate_json_export()
    }

    /// Generate LaTeX table.
    fn generate_latex_table(&self) -> String {
        self.inner.generate_latex_table()
    }

    /// Clear all results.
    fn clear_results(&mut self) {
        self.inner.clear_results();
    }

    /// Number of results.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!("BenchmarkSuite(results={})", self.inner.len())
    }
}

/// Baseline for regression detection.
#[pyclass]
#[derive(Clone)]
pub struct PyBenchmarkBaseline {
    inner: BenchmarkBaseline,
}

#[pymethods]
impl PyBenchmarkBaseline {
    /// Create baseline from results.
    #[classmethod]
    fn from_results(
        _cls: &Bound<'_, PyType>,
        results: Vec<PyBenchmarkResult>,
        version: &str,
    ) -> Self {
        let rust_results: Vec<RustBenchmarkResult> = results.into_iter().map(Into::into).collect();
        Self {
            inner: BenchmarkBaseline::from_results(&rust_results, version),
        }
    }

    /// Version/commit hash.
    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }

    /// Number of baseline results.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get a baseline result.
    fn get(&self, workload_id: &str, size: usize) -> Option<PyBenchmarkResult> {
        self.inner.get(workload_id, size).cloned().map(Into::into)
    }

    fn __repr__(&self) -> String {
        format!(
            "BenchmarkBaseline(version='{}', results={})",
            self.inner.version,
            self.inner.len()
        )
    }
}

// =============================================================================
// Module Registration
// =============================================================================

/// Register benchmark types with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create benchmark submodule
    let benchmark = PyModule::new_bound(m.py(), "benchmark")?;

    // Configuration types
    benchmark.add_class::<PyBenchmarkConfig>()?;
    benchmark.add_class::<PyWorkloadConfig>()?;

    // Result types
    benchmark.add_class::<PyBenchmarkResult>()?;
    benchmark.add_class::<PyBenchmarkBaseline>()?;

    // Statistics types
    benchmark.add_class::<PyConfidenceInterval>()?;
    benchmark.add_class::<PyDetailedStatistics>()?;
    benchmark.add_class::<PyScalingMetrics>()?;

    // Regression types
    benchmark.add_class::<PyRegressionStatus>()?;
    benchmark.add_class::<PyRegressionEntry>()?;
    benchmark.add_class::<PyRegressionReport>()?;

    // Suite
    benchmark.add_class::<PyBenchmarkSuite>()?;

    // Add submodule
    m.add_submodule(&benchmark)?;

    // Add commonly-used types at top level
    m.add_class::<PyBenchmarkSuite>()?;
    m.add_class::<PyBenchmarkConfig>()?;

    Ok(())
}
