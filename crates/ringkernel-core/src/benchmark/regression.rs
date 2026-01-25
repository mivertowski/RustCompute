//! Regression detection and reporting.

use super::result::{BenchmarkBaseline, BenchmarkResult};

/// Regression status for a benchmark comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionStatus {
    /// Performance improved significantly.
    Improved,
    /// Performance unchanged (within threshold).
    Unchanged,
    /// Performance regressed significantly.
    Regressed,
}

impl RegressionStatus {
    /// Returns a symbol representation.
    #[must_use]
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Improved => "+",
            Self::Unchanged => "=",
            Self::Regressed => "-",
        }
    }

    /// Returns a text representation.
    #[must_use]
    pub fn text(&self) -> &'static str {
        match self {
            Self::Improved => "IMPROVED",
            Self::Unchanged => "UNCHANGED",
            Self::Regressed => "REGRESSED",
        }
    }
}

/// Single regression comparison entry.
#[derive(Debug, Clone)]
pub struct RegressionEntry {
    /// Workload identifier.
    pub workload_id: String,
    /// Workload size.
    pub size: usize,
    /// Current throughput.
    pub current_throughput: f64,
    /// Baseline throughput.
    pub baseline_throughput: f64,
    /// Percent change (positive = improvement, negative = regression).
    pub percent_change: f64,
    /// Regression status.
    pub status: RegressionStatus,
}

impl RegressionEntry {
    /// Creates a new regression entry.
    #[must_use]
    pub fn new(
        workload_id: impl Into<String>,
        size: usize,
        current_throughput: f64,
        baseline_throughput: f64,
        threshold: f64,
    ) -> Self {
        let percent_change = if baseline_throughput > 0.0 {
            ((current_throughput - baseline_throughput) / baseline_throughput) * 100.0
        } else {
            0.0
        };

        let status = if percent_change < -threshold * 100.0 {
            RegressionStatus::Regressed
        } else if percent_change > threshold * 100.0 {
            RegressionStatus::Improved
        } else {
            RegressionStatus::Unchanged
        };

        Self {
            workload_id: workload_id.into(),
            size,
            current_throughput,
            baseline_throughput,
            percent_change,
            status,
        }
    }
}

/// Regression report comparing current results to baseline.
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// All comparisons.
    pub entries: Vec<RegressionEntry>,
    /// Number of regressions.
    pub regression_count: usize,
    /// Number of improvements.
    pub improvement_count: usize,
    /// Number unchanged.
    pub unchanged_count: usize,
    /// Overall status.
    pub overall_status: RegressionStatus,
    /// Regression threshold used.
    pub threshold: f64,
}

impl RegressionReport {
    /// Creates a report comparing current results to baseline.
    #[must_use]
    pub fn compare(
        current: &[BenchmarkResult],
        baseline: &BenchmarkBaseline,
        threshold: f64,
    ) -> Self {
        let mut entries = Vec::new();
        let mut regression_count = 0;
        let mut improvement_count = 0;
        let mut unchanged_count = 0;

        for result in current {
            if let Some(base) = baseline.get(&result.workload_id, result.size) {
                let entry = RegressionEntry::new(
                    &result.workload_id,
                    result.size,
                    result.throughput_ops,
                    base.throughput_ops,
                    threshold,
                );

                match entry.status {
                    RegressionStatus::Regressed => regression_count += 1,
                    RegressionStatus::Improved => improvement_count += 1,
                    RegressionStatus::Unchanged => unchanged_count += 1,
                }

                entries.push(entry);
            }
        }

        let overall_status = if regression_count > 0 {
            RegressionStatus::Regressed
        } else if improvement_count > 0 {
            RegressionStatus::Improved
        } else {
            RegressionStatus::Unchanged
        };

        Self {
            entries,
            regression_count,
            improvement_count,
            unchanged_count,
            overall_status,
            threshold,
        }
    }

    /// Returns whether any regressions were detected.
    #[must_use]
    pub fn has_regressions(&self) -> bool {
        self.regression_count > 0
    }

    /// Returns the total number of comparisons.
    #[must_use]
    pub fn total_comparisons(&self) -> usize {
        self.entries.len()
    }

    /// Returns the worst regression (largest negative change).
    #[must_use]
    pub fn worst_regression(&self) -> Option<&RegressionEntry> {
        self.entries
            .iter()
            .filter(|e| e.status == RegressionStatus::Regressed)
            .min_by(|a, b| a.percent_change.partial_cmp(&b.percent_change).unwrap())
    }

    /// Returns the best improvement (largest positive change).
    #[must_use]
    pub fn best_improvement(&self) -> Option<&RegressionEntry> {
        self.entries
            .iter()
            .filter(|e| e.status == RegressionStatus::Improved)
            .max_by(|a, b| a.percent_change.partial_cmp(&b.percent_change).unwrap())
    }

    /// Generates a summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Overall: {} | {} regressions, {} improvements, {} unchanged | Threshold: {:.0}%",
            self.overall_status.text(),
            self.regression_count,
            self.improvement_count,
            self.unchanged_count,
            self.threshold * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_regression_status() {
        assert_eq!(RegressionStatus::Improved.symbol(), "+");
        assert_eq!(RegressionStatus::Unchanged.symbol(), "=");
        assert_eq!(RegressionStatus::Regressed.symbol(), "-");
    }

    #[test]
    fn test_regression_entry() {
        // Regression case
        let entry = RegressionEntry::new("test", 1000, 80.0, 100.0, 0.10);
        assert_eq!(entry.status, RegressionStatus::Regressed);
        assert!(entry.percent_change < 0.0);

        // Improvement case
        let entry = RegressionEntry::new("test", 1000, 120.0, 100.0, 0.10);
        assert_eq!(entry.status, RegressionStatus::Improved);
        assert!(entry.percent_change > 0.0);

        // Unchanged case
        let entry = RegressionEntry::new("test", 1000, 105.0, 100.0, 0.10);
        assert_eq!(entry.status, RegressionStatus::Unchanged);
    }

    #[test]
    fn test_regression_report() {
        let baseline_results = vec![BenchmarkResult::new(
            "workload_a",
            1000,
            Duration::from_millis(100),
        )];
        let baseline = BenchmarkBaseline::from_results(&baseline_results, "v1.0");

        // Test regression
        let current_regressed = vec![BenchmarkResult {
            workload_id: "workload_a".to_string(),
            size: 1000,
            throughput_ops: 8000.0, // 20% slower than baseline (10000)
            total_time: Duration::from_millis(125),
            iterations: None,
            converged: None,
            measurement_times: vec![Duration::from_millis(125)],
            custom_metrics: Default::default(),
        }];

        let report = RegressionReport::compare(&current_regressed, &baseline, 0.10);
        assert!(report.has_regressions());
        assert_eq!(report.regression_count, 1);

        // Test improvement
        let current_improved = vec![BenchmarkResult {
            workload_id: "workload_a".to_string(),
            size: 1000,
            throughput_ops: 12000.0, // 20% faster
            total_time: Duration::from_millis(83),
            iterations: None,
            converged: None,
            measurement_times: vec![Duration::from_millis(83)],
            custom_metrics: Default::default(),
        }];

        let report = RegressionReport::compare(&current_improved, &baseline, 0.10);
        assert!(!report.has_regressions());
        assert_eq!(report.improvement_count, 1);
    }
}
