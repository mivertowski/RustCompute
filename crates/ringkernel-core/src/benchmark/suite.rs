//! Benchmark suite for running and tracking benchmarks.

use std::time::{SystemTime, UNIX_EPOCH};

use super::config::BenchmarkConfig;
use super::regression::RegressionReport;
use super::result::{BenchmarkBaseline, BenchmarkResult};
use super::statistics::{ConfidenceInterval, DetailedStatistics, ScalingMetrics};
use super::traits::{Benchmarkable, WorkloadConfig};

/// Benchmark suite for running and tracking benchmarks.
pub struct BenchmarkSuite {
    /// Configuration.
    config: BenchmarkConfig,
    /// Historical results.
    results: Vec<BenchmarkResult>,
    /// Current baseline.
    baseline: Option<BenchmarkBaseline>,
}

impl BenchmarkSuite {
    /// Creates a new benchmark suite.
    #[must_use]
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            baseline: None,
        }
    }

    /// Creates a suite with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(BenchmarkConfig::default())
    }

    /// Sets the baseline for regression comparison.
    pub fn set_baseline(&mut self, baseline: BenchmarkBaseline) {
        self.baseline = Some(baseline);
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &BenchmarkConfig {
        &self.config
    }

    /// Returns collected results.
    #[must_use]
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Adds a benchmark result.
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Runs a benchmarkable workload with the given configuration.
    pub fn run<B: Benchmarkable>(&mut self, workload: &B, workload_config: &WorkloadConfig) {
        let result = workload.execute(workload_config);
        self.results.push(result);
    }

    /// Runs a benchmarkable workload across all configured sizes.
    pub fn run_all_sizes<B: Benchmarkable>(&mut self, workload: &B) {
        for &size in &self.config.workload_sizes.clone() {
            let workload_config = WorkloadConfig::new(size)
                .with_convergence_threshold(self.config.convergence_threshold)
                .with_max_iterations(self.config.max_iterations);

            let result = workload.execute(&workload_config);
            self.results.push(result);
        }
    }

    /// Compares current results to baseline.
    #[must_use]
    pub fn compare_to_baseline(&self) -> Option<RegressionReport> {
        self.baseline.as_ref().map(|baseline| {
            RegressionReport::compare(&self.results, baseline, self.config.regression_threshold)
        })
    }

    /// Creates a baseline from current results.
    #[must_use]
    pub fn create_baseline(&self, version: &str) -> BenchmarkBaseline {
        BenchmarkBaseline::from_results(&self.results, version)
    }

    /// Generates a Markdown report.
    #[must_use]
    pub fn generate_markdown_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Benchmark Report\n\n");
        report.push_str(&format!("Generated: {}\n\n", format_timestamp()));

        // Summary table
        report.push_str("## Results Summary\n\n");
        report.push_str(
            "| Workload | Size | Throughput (ops/s) | Time (ms) | Iterations | Converged |\n",
        );
        report.push_str(
            "|----------|------|-------------------|-----------|------------|-----------|\n",
        );

        for result in &self.results {
            report.push_str(&format!(
                "| {} | {} | {:.2} | {:.2} | {} | {} |\n",
                result.workload_id,
                format_number(result.size),
                result.throughput_ops,
                result.total_time_ms(),
                result.iterations.map_or("-".to_string(), |i| i.to_string()),
                result
                    .converged
                    .map_or("-".to_string(), |c| if c { "Yes" } else { "No" }
                        .to_string())
            ));
        }

        // Regression comparison
        if let Some(regression) = self.compare_to_baseline() {
            report.push_str("\n## Regression Analysis\n\n");

            let status_text = regression.overall_status.text();

            report.push_str(&format!(
                "Overall Status: **{}** ({} regressions, {} improvements, {} unchanged)\n\n",
                status_text,
                regression.regression_count,
                regression.improvement_count,
                regression.unchanged_count
            ));

            if !regression.entries.is_empty() {
                report.push_str(
                    "| Workload | Size | Current (ops/s) | Baseline (ops/s) | Change | Status |\n",
                );
                report.push_str(
                    "|----------|------|-----------------|------------------|--------|--------|\n",
                );

                for entry in &regression.entries {
                    report.push_str(&format!(
                        "| {} | {} | {:.2} | {:.2} | {:+.1}% | {} |\n",
                        entry.workload_id,
                        format_number(entry.size),
                        entry.current_throughput,
                        entry.baseline_throughput,
                        entry.percent_change,
                        entry.status.text()
                    ));
                }
            }
        }

        report
    }

    /// Generates a JSON export of all results.
    #[cfg(feature = "benchmark")]
    #[must_use]
    pub fn generate_json_export(&self) -> String {
        let mut data = Vec::new();

        for result in &self.results {
            let times_ms: Vec<f64> = result
                .measurement_times
                .iter()
                .map(|d| d.as_secs_f64() * 1000.0)
                .collect();
            let ci = ConfidenceInterval::from_values(&times_ms);
            let stats = DetailedStatistics::from_values(&times_ms);

            let entry = serde_json::json!({
                "workload_id": result.workload_id,
                "size": result.size,
                "throughput_ops": result.throughput_ops,
                "total_time_ms": result.total_time_ms(),
                "iterations": result.iterations,
                "converged": result.converged,
                "statistics": {
                    "mean": stats.mean,
                    "std_dev": stats.std_dev,
                    "min": stats.min,
                    "max": stats.max,
                    "median": stats.median,
                    "p5": stats.p5,
                    "p25": stats.p25,
                    "p75": stats.p75,
                    "p95": stats.p95,
                    "p99": stats.p99,
                },
                "confidence_interval": {
                    "lower": ci.lower,
                    "upper": ci.upper,
                    "confidence_level": ci.confidence_level,
                },
                "custom_metrics": result.custom_metrics,
            });

            data.push(entry);
        }

        serde_json::to_string_pretty(&data).unwrap_or_default()
    }

    /// Generates a LaTeX throughput table.
    #[must_use]
    pub fn generate_latex_table(&self) -> String {
        let mut latex = String::new();

        latex.push_str("% Throughput Table - Generated by RingKernel Benchmark Suite\n");
        latex.push_str("\\begin{table}[htbp]\n");
        latex.push_str("\\centering\n");
        latex.push_str("\\caption{Benchmark Throughput (ops/s)}\n");
        latex.push_str("\\label{tab:throughput}\n");

        // Collect unique workloads and sizes
        let mut workloads: Vec<&str> = self
            .results
            .iter()
            .map(|r| r.workload_id.as_str())
            .collect();
        workloads.sort();
        workloads.dedup();

        let mut sizes: Vec<usize> = self.results.iter().map(|r| r.size).collect();
        sizes.sort();
        sizes.dedup();

        // Build table header
        latex.push_str("\\begin{tabular}{l");
        for _ in &sizes {
            latex.push('r');
        }
        latex.push_str("}\n");
        latex.push_str("\\toprule\n");

        latex.push_str("Workload");
        for size in &sizes {
            latex.push_str(&format!(" & {}K", size / 1000));
        }
        latex.push_str(" \\\\\n");
        latex.push_str("\\midrule\n");

        // Build table body
        for workload in &workloads {
            latex.push_str(workload);

            for size in &sizes {
                let result = self
                    .results
                    .iter()
                    .find(|r| r.workload_id == *workload && r.size == *size);

                if let Some(r) = result {
                    latex.push_str(&format!(" & {:.1}", r.throughput_ops / 1000.0));
                } else {
                    latex.push_str(" & -");
                }
            }
            latex.push_str(" \\\\\n");
        }

        latex.push_str("\\bottomrule\n");
        latex.push_str("\\end{tabular}\n");
        latex.push_str("\\end{table}\n");

        latex
    }

    /// Computes scaling metrics for a specific workload.
    #[must_use]
    pub fn scaling_metrics_for(&self, workload_id: &str) -> ScalingMetrics {
        let results: Vec<&BenchmarkResult> = self
            .results
            .iter()
            .filter(|r| r.workload_id == workload_id)
            .collect();

        ScalingMetrics::from_results(&results)
    }

    /// Clears collected results.
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Returns the number of results.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Returns whether the suite has no results.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

/// Formats a number with thousand separators.
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Formats current timestamp.
fn format_timestamp() -> String {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    let mut year = 1970;
    let mut remaining_days = days_since_epoch;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let month_days = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for &days in &month_days {
        if remaining_days < days as u64 {
            break;
        }
        remaining_days -= days as u64;
        month += 1;
    }

    let day = remaining_days + 1;

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
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
                throughput_ops: config.size as f64 * 100.0,
                total_time: Duration::from_millis(10),
                iterations: Some(10),
                converged: Some(true),
                measurement_times: vec![Duration::from_millis(10)],
                custom_metrics: Default::default(),
            }
        }
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let suite = BenchmarkSuite::new(BenchmarkConfig::default());
        assert!(suite.is_empty());
        assert!(suite.results().is_empty());
    }

    #[test]
    fn test_benchmark_suite_run() {
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::default());
        let workload = TestWorkload;

        suite.run(&workload, &WorkloadConfig::new(1000));

        assert_eq!(suite.len(), 1);
        assert_eq!(suite.results()[0].workload_id, "Test");
    }

    #[test]
    fn test_benchmark_suite_run_all_sizes() {
        let config = BenchmarkConfig::default().with_sizes(vec![100, 200, 300]);
        let mut suite = BenchmarkSuite::new(config);
        let workload = TestWorkload;

        suite.run_all_sizes(&workload);

        assert_eq!(suite.len(), 3);
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1000000), "1,000,000");
        assert_eq!(format_number(100), "100");
    }

    #[test]
    fn test_generate_markdown_report() {
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::default());

        suite.add_result(BenchmarkResult {
            workload_id: "test".to_string(),
            size: 1000,
            throughput_ops: 100000.0,
            total_time: Duration::from_millis(50),
            iterations: Some(50),
            converged: Some(true),
            measurement_times: vec![Duration::from_millis(50)],
            custom_metrics: Default::default(),
        });

        let report = suite.generate_markdown_report();
        assert!(report.contains("# Benchmark Report"));
        assert!(report.contains("test"));
        assert!(report.contains("1,000"));
    }

    #[test]
    fn test_scaling_metrics() {
        let mut suite = BenchmarkSuite::new(BenchmarkConfig::default());

        // Add results with linear scaling
        for size in [100, 200, 400, 800] {
            suite.add_result(BenchmarkResult::new(
                "linear_workload",
                size,
                Duration::from_micros(size as u64),
            ));
        }

        let metrics = suite.scaling_metrics_for("linear_workload");
        assert!(metrics.data_points >= 4);
    }
}
