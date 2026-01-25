//! Statistical analysis utilities for benchmarks.

use super::result::BenchmarkResult;
use std::time::Duration;

/// Confidence interval for a measurement.
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceInterval {
    /// Lower bound of CI.
    pub lower: f64,
    /// Upper bound of CI.
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95%).
    pub confidence_level: f64,
}

impl ConfidenceInterval {
    /// Computes 95% confidence interval from throughput values.
    #[must_use]
    pub fn from_values(values: &[f64]) -> Self {
        Self::from_values_with_confidence(values, 0.95)
    }

    /// Computes confidence interval with custom confidence level.
    #[must_use]
    pub fn from_values_with_confidence(values: &[f64], confidence_level: f64) -> Self {
        if values.is_empty() {
            return Self {
                lower: 0.0,
                upper: 0.0,
                confidence_level,
            };
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        let variance = if values.len() > 1 {
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        // Standard error
        let std_error = std_dev / n.sqrt();

        // t-value approximation for 95% CI
        let t_value = if values.len() >= 30 {
            1.96
        } else {
            // Approximation for smaller samples
            2.0 + 4.0 / values.len() as f64
        };

        Self {
            lower: mean - t_value * std_error,
            upper: mean + t_value * std_error,
            confidence_level,
        }
    }

    /// Computes 95% confidence interval from Duration measurements.
    #[must_use]
    pub fn from_durations(durations: &[Duration]) -> Self {
        let values: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        Self::from_values(&values)
    }

    /// Returns the width of the confidence interval.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Returns the midpoint of the confidence interval.
    #[must_use]
    pub fn midpoint(&self) -> f64 {
        (self.lower + self.upper) / 2.0
    }
}

/// Detailed statistics with percentiles.
#[derive(Debug, Clone)]
pub struct DetailedStatistics {
    /// Number of samples.
    pub count: usize,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// 5th percentile.
    pub p5: f64,
    /// 25th percentile.
    pub p25: f64,
    /// 75th percentile.
    pub p75: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
}

impl DetailedStatistics {
    /// Computes detailed statistics from values.
    #[must_use]
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                count: 0,
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                p5: 0.0,
                p25: 0.0,
                p75: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let mean = sorted.iter().sum::<f64>() / n as f64;

        let variance = if n > 1 {
            sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };

        Self {
            count: n,
            mean,
            std_dev: variance.sqrt(),
            min: sorted[0],
            max: sorted[n - 1],
            median: compute_percentile(&sorted, 50.0),
            p5: compute_percentile(&sorted, 5.0),
            p25: compute_percentile(&sorted, 25.0),
            p75: compute_percentile(&sorted, 75.0),
            p95: compute_percentile(&sorted, 95.0),
            p99: compute_percentile(&sorted, 99.0),
        }
    }

    /// Computes detailed statistics from Duration measurements.
    #[must_use]
    pub fn from_durations(durations: &[Duration]) -> Self {
        let values: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        Self::from_values(&values)
    }

    /// Returns the coefficient of variation (stddev / mean).
    #[must_use]
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < f64::EPSILON {
            0.0
        } else {
            self.std_dev / self.mean
        }
    }

    /// Returns the interquartile range (p75 - p25).
    #[must_use]
    pub fn iqr(&self) -> f64 {
        self.p75 - self.p25
    }
}

/// Scaling metrics for performance analysis.
#[derive(Debug, Clone)]
pub struct ScalingMetrics {
    /// Scaling exponent (log-log slope).
    pub exponent: f64,
    /// Coefficient of determination (R²).
    pub r_squared: f64,
    /// Number of data points used.
    pub data_points: usize,
}

impl ScalingMetrics {
    /// Computes scaling metrics from benchmark results.
    #[must_use]
    pub fn from_results(results: &[&BenchmarkResult]) -> Self {
        if results.len() < 2 {
            return Self {
                exponent: 0.0,
                r_squared: 0.0,
                data_points: results.len(),
            };
        }

        let n = results.len() as f64;

        // Log-log regression: log(throughput) = exponent * log(size) + intercept
        let log_x: Vec<f64> = results.iter().map(|r| (r.size as f64).ln()).collect();
        let log_y: Vec<f64> = results
            .iter()
            .map(|r| r.throughput_ops.max(f64::EPSILON).ln())
            .collect();

        let sum_x: f64 = log_x.iter().sum();
        let sum_y: f64 = log_y.iter().sum();
        let sum_xy: f64 = log_x.iter().zip(log_y.iter()).map(|(x, y)| x * y).sum();
        let sum_xx: f64 = log_x.iter().map(|x| x * x).sum();
        let sum_yy: f64 = log_y.iter().map(|y| y * y).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        let exponent = if denom.abs() > f64::EPSILON {
            (n * sum_xy - sum_x * sum_y) / denom
        } else {
            0.0
        };

        // R-squared calculation
        let ss_tot = sum_yy - (sum_y * sum_y) / n;
        let intercept = (sum_y - exponent * sum_x) / n;
        let ss_res: f64 = log_x
            .iter()
            .zip(log_y.iter())
            .map(|(x, y)| {
                let predicted = exponent * x + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot.abs() > f64::EPSILON {
            (1.0 - ss_res / ss_tot).max(0.0)
        } else {
            0.0
        };

        Self {
            exponent,
            r_squared,
            data_points: results.len(),
        }
    }

    /// Computes scaling metrics from sizes and throughputs.
    #[must_use]
    pub fn from_sizes_and_throughputs(sizes: &[usize], throughputs: &[f64]) -> Self {
        if sizes.len() != throughputs.len() || sizes.len() < 2 {
            return Self {
                exponent: 0.0,
                r_squared: 0.0,
                data_points: sizes.len().min(throughputs.len()),
            };
        }

        let n = sizes.len() as f64;

        let log_x: Vec<f64> = sizes.iter().map(|&s| (s as f64).ln()).collect();
        let log_y: Vec<f64> = throughputs
            .iter()
            .map(|&t| t.max(f64::EPSILON).ln())
            .collect();

        let sum_x: f64 = log_x.iter().sum();
        let sum_y: f64 = log_y.iter().sum();
        let sum_xy: f64 = log_x.iter().zip(log_y.iter()).map(|(x, y)| x * y).sum();
        let sum_xx: f64 = log_x.iter().map(|x| x * x).sum();
        let sum_yy: f64 = log_y.iter().map(|y| y * y).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        let exponent = if denom.abs() > f64::EPSILON {
            (n * sum_xy - sum_x * sum_y) / denom
        } else {
            0.0
        };

        let ss_tot = sum_yy - (sum_y * sum_y) / n;
        let intercept = (sum_y - exponent * sum_x) / n;
        let ss_res: f64 = log_x
            .iter()
            .zip(log_y.iter())
            .map(|(x, y)| {
                let predicted = exponent * x + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot.abs() > f64::EPSILON {
            (1.0 - ss_res / ss_tot).max(0.0)
        } else {
            0.0
        };

        Self {
            exponent,
            r_squared,
            data_points: sizes.len(),
        }
    }

    /// Returns a qualitative assessment of scaling.
    #[must_use]
    pub fn scaling_quality(&self) -> &'static str {
        if self.exponent > 0.9 {
            "Excellent (near-linear)"
        } else if self.exponent > 0.7 {
            "Good"
        } else if self.exponent > 0.5 {
            "Fair"
        } else if self.exponent > 0.0 {
            "Sub-linear"
        } else {
            "Poor (negative scaling)"
        }
    }
}

/// Computes percentile from sorted values.
fn compute_percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Computes public percentile from unsorted values.
#[must_use]
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    compute_percentile(&sorted, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_interval() {
        let values = vec![100.0, 102.0, 98.0, 101.0, 99.0];
        let ci = ConfidenceInterval::from_values(&values);

        assert!((ci.confidence_level - 0.95).abs() < f64::EPSILON);
        assert!(ci.lower < 100.0);
        assert!(ci.upper > 100.0);
        assert!(ci.width() > 0.0);
    }

    #[test]
    fn test_confidence_interval_empty() {
        let ci = ConfidenceInterval::from_values(&[]);
        assert_eq!(ci.lower, 0.0);
        assert_eq!(ci.upper, 0.0);
    }

    #[test]
    fn test_detailed_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = DetailedStatistics::from_values(&values);

        assert_eq!(stats.count, 10);
        assert!((stats.mean - 5.5).abs() < 0.001);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert!((stats.median - 5.5).abs() < 1.0); // Approximate due to discrete data
    }

    #[test]
    fn test_detailed_statistics_empty() {
        let stats = DetailedStatistics::from_values(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_scaling_metrics() {
        // Perfect linear scaling: throughput = size
        let sizes = vec![100, 200, 400, 800];
        let throughputs = vec![100.0, 200.0, 400.0, 800.0];
        let metrics = ScalingMetrics::from_sizes_and_throughputs(&sizes, &throughputs);

        assert!((metrics.exponent - 1.0).abs() < 0.1); // Should be close to 1.0
        assert!(metrics.r_squared > 0.99); // High R² for perfect linear
    }

    #[test]
    fn test_scaling_metrics_insufficient_data() {
        let sizes = vec![100];
        let throughputs = vec![100.0];
        let metrics = ScalingMetrics::from_sizes_and_throughputs(&sizes, &throughputs);

        assert_eq!(metrics.exponent, 0.0);
        assert_eq!(metrics.data_points, 1);
    }

    #[test]
    fn test_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 100.0), 5.0);
        assert_eq!(percentile(&values, 50.0), 3.0);
    }
}
