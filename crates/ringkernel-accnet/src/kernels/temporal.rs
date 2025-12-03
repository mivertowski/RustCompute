//! Temporal analysis kernels for time-series anomaly detection.
//!
//! These kernels analyze transaction patterns over time to detect:
//! - Seasonal variations
//! - Behavioral anomalies
//! - Trend changes

use crate::models::{
    AccountingNetwork, BehavioralBaseline, SeasonalPattern,
    TemporalAlert, TemporalAlertType, TimeGranularity,
    HybridTimestamp,
};
use std::collections::HashMap;

/// Configuration for temporal analysis kernels.
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Block size for GPU dispatch.
    pub block_size: u32,
    /// Z-score threshold for anomalies.
    pub z_score_threshold: f64,
    /// Minimum data points for baseline.
    pub min_baseline_points: usize,
    /// Time granularity for analysis.
    pub granularity: TimeGranularity,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            z_score_threshold: 3.0,
            min_baseline_points: 30,
            granularity: TimeGranularity::Daily,
        }
    }
}

/// Result of temporal analysis.
#[derive(Debug, Clone, Default)]
pub struct TemporalResult {
    /// Account baselines.
    pub baselines: HashMap<u16, BehavioralBaseline>,
    /// Detected seasonal patterns.
    pub seasonal_patterns: Vec<SeasonalPattern>,
    /// Generated alerts.
    pub alerts: Vec<TemporalAlert>,
    /// Analysis statistics.
    pub stats: TemporalStats,
}

/// Statistics from temporal analysis.
#[derive(Debug, Clone, Default)]
pub struct TemporalStats {
    /// Time periods analyzed.
    pub periods_analyzed: usize,
    /// Baselines computed.
    pub baselines_computed: usize,
    /// Anomalies detected.
    pub anomalies_detected: usize,
}

/// Temporal analysis kernel dispatcher.
pub struct TemporalKernel {
    config: TemporalConfig,
}

impl TemporalKernel {
    /// Create a new temporal kernel.
    pub fn new(config: TemporalConfig) -> Self {
        Self { config }
    }

    /// Analyze temporal patterns (CPU fallback).
    pub fn analyze(&self, network: &AccountingNetwork) -> TemporalResult {
        let mut result = TemporalResult::default();

        // Build time series per account
        let account_series = self.build_time_series(network);

        // Compute baselines
        for (account_idx, series) in &account_series {
            if series.len() >= self.config.min_baseline_points {
                let baseline = self.compute_baseline(*account_idx, series);
                result.baselines.insert(*account_idx, baseline);
                result.stats.baselines_computed += 1;
            }
        }

        // Detect anomalies using baselines
        for (account_idx, series) in &account_series {
            if let Some(baseline) = result.baselines.get(account_idx) {
                for (timestamp, value) in series {
                    let (is_anomaly, score) = baseline.is_anomaly(*value);
                    if is_anomaly {
                        result.alerts.push(TemporalAlert {
                            id: uuid::Uuid::new_v4(),
                            account_id: *account_idx,
                            alert_type: TemporalAlertType::Anomaly,
                            severity: score,
                            trigger_value: *value,
                            expected_value: baseline.mean,
                            deviation: (*value - baseline.mean).abs() / baseline.std_dev.max(0.001),
                            timestamp: *timestamp,
                            message: format!(
                                "Anomalous activity: {:.2} (expected {:.2} ± {:.2})",
                                value, baseline.mean, baseline.std_dev * 2.0
                            ),
                        });
                        result.stats.anomalies_detected += 1;
                    }
                }
            }
        }

        result.stats.periods_analyzed = account_series.values().map(|s| s.len()).sum();

        result
    }

    /// Build time series from flows.
    fn build_time_series(&self, network: &AccountingNetwork) -> HashMap<u16, Vec<(HybridTimestamp, f64)>> {
        let mut series: HashMap<u16, Vec<(HybridTimestamp, f64)>> = HashMap::new();

        for flow in &network.flows {
            let amount = flow.amount.to_f64();

            // Add to source account (outflow)
            series.entry(flow.source_account_index)
                .or_default()
                .push((flow.timestamp, -amount));

            // Add to target account (inflow)
            series.entry(flow.target_account_index)
                .or_default()
                .push((flow.timestamp, amount));
        }

        // Sort by timestamp
        for series_vec in series.values_mut() {
            series_vec.sort_by_key(|(ts, _)| ts.physical);
        }

        series
    }

    /// Compute baseline for a time series.
    fn compute_baseline(&self, account_id: u16, series: &[(HybridTimestamp, f64)]) -> BehavioralBaseline {
        let values: Vec<f64> = series.iter().map(|(_, v)| *v).collect();
        let n = values.len() as f64;

        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Compute percentiles
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p5_idx = ((n * 0.05) as usize).min(sorted.len().saturating_sub(1));
        let p25_idx = ((n * 0.25) as usize).min(sorted.len().saturating_sub(1));
        let p50_idx = ((n * 0.50) as usize).min(sorted.len().saturating_sub(1));
        let p75_idx = ((n * 0.75) as usize).min(sorted.len().saturating_sub(1));
        let p95_idx = ((n * 0.95) as usize).min(sorted.len().saturating_sub(1));

        let median = sorted.get(p50_idx).copied().unwrap_or(mean);
        let q1 = sorted.get(p25_idx).copied().unwrap_or(min);
        let q3 = sorted.get(p75_idx).copied().unwrap_or(max);
        let iqr = q3 - q1;

        // Compute MAD (Median Absolute Deviation)
        let mut abs_deviations: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
        abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad_idx = ((abs_deviations.len() as f64 * 0.5) as usize).min(abs_deviations.len().saturating_sub(1));
        let mad = abs_deviations.get(mad_idx).copied().unwrap_or(0.0);

        let mut baseline = BehavioralBaseline::new(account_id);
        baseline.mean = mean;
        baseline.std_dev = std_dev;
        baseline.min_value = min;
        baseline.max_value = max;
        baseline.median = median;
        baseline.mad = mad;
        baseline.q1 = q1;
        baseline.q3 = q3;
        baseline.iqr = iqr;
        baseline.p5 = sorted.get(p5_idx).copied().unwrap_or(min);
        baseline.p95 = sorted.get(p95_idx).copied().unwrap_or(max);
        baseline.period_count = series.len() as u16;

        baseline
    }
}

impl Default for TemporalKernel {
    fn default() -> Self {
        Self::new(TemporalConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_kernel_creation() {
        let kernel = TemporalKernel::default();
        assert_eq!(kernel.config.z_score_threshold, 3.0);
    }

    #[test]
    fn test_baseline_computation() {
        let kernel = TemporalKernel::default();
        let series: Vec<(HybridTimestamp, f64)> = (0..100)
            .map(|i| (HybridTimestamp::new(i * 1000, 0), (i % 10) as f64))
            .collect();

        let baseline = kernel.compute_baseline(0, &series);
        assert!(baseline.mean >= 0.0);
        assert!(baseline.std_dev >= 0.0);
        assert_eq!(baseline.period_count, 100);
    }
}
