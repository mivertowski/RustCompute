//! Academic benchmark harness for RingKernel GPU actor paradigm validation.
//!
//! Provides statistical rigor for paper-quality results:
//! - Configurable warmup and measurement iterations
//! - Percentile computation (p50, p95, p99, p99.9)
//! - Confidence intervals (95% CI)
//! - Cohen's d effect size for comparisons
//! - CSV and JSON export for reproducibility
//! - Outlier detection via Modified Z-score (MAD-based)

use std::fmt;
use std::fs;
use std::io::Write;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Configuration for an academic benchmark run.
pub struct BenchmarkConfig {
    /// Number of warmup iterations (discarded).
    pub warmup_iterations: usize,
    /// Number of measurement iterations per trial.
    pub measurement_iterations: usize,
    /// Number of independent trials.
    pub trials: usize,
    /// Output directory for CSV/JSON results.
    pub output_dir: String,
    /// Experiment name (used in filenames).
    pub experiment_name: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 100,
            measurement_iterations: 1000,
            trials: 10,
            output_dir: "benchmark_results".to_string(),
            experiment_name: "unnamed".to_string(),
        }
    }
}

/// Collected measurements from a benchmark run.
#[derive(Clone)]
pub struct Measurements {
    /// Raw values in measurement order.
    pub values: Vec<f64>,
    /// Unit of measurement.
    pub unit: String,
    /// Configuration label.
    pub configuration: String,
    /// Metric name.
    pub metric: String,
}

/// Computed statistics for a set of measurements.
pub struct Statistics {
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub ci_95_lower: f64,
    pub ci_95_upper: f64,
    pub cv: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
    pub outlier_count: usize,
}

impl Statistics {
    /// Compute statistics from raw measurements.
    pub fn compute(measurements: &Measurements) -> Self {
        let values = &measurements.values;
        let n = values.len();
        assert!(n > 0, "cannot compute statistics on empty measurements");

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
        let std_dev = variance.sqrt();
        let ci_margin = 1.96 * std_dev / (n as f64).sqrt();

        // Percentiles (linear interpolation)
        let percentile = |p: f64| -> f64 {
            let idx = p * (n - 1) as f64;
            let lower = idx.floor() as usize;
            let upper = idx.ceil().min((n - 1) as f64) as usize;
            let frac = idx - lower as f64;
            sorted[lower] * (1.0 - frac) + sorted[upper] * frac
        };

        // Outlier detection: Modified Z-score using MAD
        let median_val = percentile(0.5);
        let mad = {
            let mut abs_devs: Vec<f64> = values.iter().map(|v| (v - median_val).abs()).collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = abs_devs.len() / 2;
            if abs_devs.len() % 2 == 0 {
                (abs_devs[mid - 1] + abs_devs[mid]) / 2.0
            } else {
                abs_devs[mid]
            }
        };
        let outlier_count = if mad > 0.0 {
            values
                .iter()
                .filter(|v| (0.6745 * (*v - median_val).abs() / mad) > 3.5)
                .count()
        } else {
            0
        };

        Self {
            n,
            mean,
            median: median_val,
            std_dev,
            ci_95_lower: mean - ci_margin,
            ci_95_upper: mean + ci_margin,
            cv: if mean != 0.0 { std_dev / mean } else { 0.0 },
            min: sorted[0],
            max: sorted[n - 1],
            p50: percentile(0.5),
            p95: percentile(0.95),
            p99: percentile(0.99),
            p999: percentile(0.999),
            outlier_count,
        }
    }
}

impl fmt::Display for Statistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  n:          {}", self.n)?;
        writeln!(f, "  mean:       {:.4}", self.mean)?;
        writeln!(f, "  median:     {:.4}", self.median)?;
        writeln!(f, "  std_dev:    {:.4}", self.std_dev)?;
        writeln!(f, "  95% CI:     [{:.4}, {:.4}]", self.ci_95_lower, self.ci_95_upper)?;
        writeln!(f, "  CV:         {:.2}%", self.cv * 100.0)?;
        writeln!(f, "  min/max:    {:.4} / {:.4}", self.min, self.max)?;
        writeln!(f, "  p50:        {:.4}", self.p50)?;
        writeln!(f, "  p95:        {:.4}", self.p95)?;
        writeln!(f, "  p99:        {:.4}", self.p99)?;
        writeln!(f, "  p99.9:      {:.4}", self.p999)?;
        write!(f, "  outliers:   {}", self.outlier_count)
    }
}

/// Comparison between two experimental conditions.
pub struct Comparison {
    pub baseline_label: String,
    pub treatment_label: String,
    pub baseline: Statistics,
    pub treatment: Statistics,
    pub speedup: f64,
    pub cohens_d: f64,
    pub p_value: f64,
    pub significant: bool,
}

impl Comparison {
    /// Compare two measurement sets (baseline vs treatment).
    pub fn compare(baseline: &Measurements, treatment: &Measurements) -> Self {
        let b_stats = Statistics::compute(baseline);
        let t_stats = Statistics::compute(treatment);

        let speedup = if t_stats.mean != 0.0 {
            b_stats.mean / t_stats.mean
        } else {
            f64::INFINITY
        };

        // Cohen's d (pooled standard deviation)
        let n1 = b_stats.n as f64;
        let n2 = t_stats.n as f64;
        let s_pooled = ((((n1 - 1.0) * b_stats.std_dev.powi(2)) + ((n2 - 1.0) * t_stats.std_dev.powi(2)))
            / (n1 + n2 - 2.0))
            .sqrt();
        let cohens_d = if s_pooled != 0.0 {
            (b_stats.mean - t_stats.mean).abs() / s_pooled
        } else {
            0.0
        };

        // Welch's t-test
        let se = ((b_stats.std_dev.powi(2) / n1) + (t_stats.std_dev.powi(2) / n2)).sqrt();
        let t_stat = if se != 0.0 {
            (b_stats.mean - t_stats.mean) / se
        } else {
            0.0
        };
        // Approximate p-value using normal distribution for large n
        let p_value = 2.0 * normal_cdf(-t_stat.abs());
        let significant = p_value < 0.05;

        Self {
            baseline_label: baseline.configuration.clone(),
            treatment_label: treatment.configuration.clone(),
            baseline: b_stats,
            treatment: t_stats,
            speedup,
            cohens_d,
            p_value,
            significant,
        }
    }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== {} vs {} ===", self.baseline_label, self.treatment_label)?;
        writeln!(f, "  Baseline mean:  {:.4}", self.baseline.mean)?;
        writeln!(f, "  Treatment mean: {:.4}", self.treatment.mean)?;
        writeln!(f, "  Speedup:        {:.2}x", self.speedup)?;
        writeln!(f, "  Cohen's d:      {:.2} ({})", self.cohens_d, effect_size_label(self.cohens_d))?;
        writeln!(f, "  p-value:        {:.2e}", self.p_value)?;
        write!(f, "  Significant:    {}", if self.significant { "YES (p < 0.05)" } else { "NO" })
    }
}

/// Export raw measurements to CSV.
pub fn export_csv(
    measurements: &[Measurements],
    experiment: &str,
    output_dir: &str,
) -> std::io::Result<String> {
    fs::create_dir_all(output_dir)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after UNIX epoch")
        .as_secs();
    let filename = format!("{}/{}_{}.csv", output_dir, experiment, timestamp);

    let mut file = fs::File::create(&filename)?;
    writeln!(file, "experiment,configuration,trial,iteration,metric,value,unit")?;

    for m in measurements {
        for (i, v) in m.values.iter().enumerate() {
            let trial = i / 1000 + 1; // assuming 1000 per trial
            let iteration = i % 1000 + 1;
            writeln!(
                file,
                "{},{},{},{},{},{},{}",
                experiment, m.configuration, trial, iteration, m.metric, v, m.unit
            )?;
        }
    }

    Ok(filename)
}

/// Export summary statistics as JSON.
pub fn export_json(
    stats: &Statistics,
    measurements: &Measurements,
    experiment: &str,
    output_dir: &str,
) -> std::io::Result<String> {
    fs::create_dir_all(output_dir)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after UNIX epoch")
        .as_secs();
    let filename = format!(
        "{}/{}_{}_{}.json",
        output_dir, experiment, measurements.configuration, timestamp
    );

    let json = format!(
        r#"{{
  "experiment": "{}",
  "configuration": "{}",
  "metric": "{}",
  "unit": "{}",
  "n": {},
  "mean": {},
  "median": {},
  "std_dev": {},
  "ci_95_lower": {},
  "ci_95_upper": {},
  "cv": {},
  "min": {},
  "max": {},
  "p50": {},
  "p95": {},
  "p99": {},
  "p999": {},
  "outliers": {},
  "timestamp": "{}"
}}"#,
        experiment,
        measurements.configuration,
        measurements.metric,
        measurements.unit,
        stats.n,
        stats.mean,
        stats.median,
        stats.std_dev,
        stats.ci_95_lower,
        stats.ci_95_upper,
        stats.cv,
        stats.min,
        stats.max,
        stats.p50,
        stats.p95,
        stats.p99,
        stats.p999,
        stats.outlier_count,
        timestamp
    );

    fs::write(&filename, json)?;
    Ok(filename)
}

/// Export a comparison as JSON.
pub fn export_comparison_json(
    comparison: &Comparison,
    experiment: &str,
    output_dir: &str,
) -> std::io::Result<String> {
    fs::create_dir_all(output_dir)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after UNIX epoch")
        .as_secs();
    let filename = format!("{}/{}_comparison_{}.json", output_dir, experiment, timestamp);

    let json = format!(
        r#"{{
  "experiment": "{}",
  "comparison": "{}_vs_{}",
  "baseline": {{
    "label": "{}",
    "mean": {},
    "ci_95": [{}, {}],
    "p99": {}
  }},
  "treatment": {{
    "label": "{}",
    "mean": {},
    "ci_95": [{}, {}],
    "p99": {}
  }},
  "speedup": {},
  "cohens_d": {},
  "effect_size": "{}",
  "p_value": {},
  "significant": {},
  "timestamp": "{}"
}}"#,
        experiment,
        comparison.baseline_label,
        comparison.treatment_label,
        comparison.baseline_label,
        comparison.baseline.mean,
        comparison.baseline.ci_95_lower,
        comparison.baseline.ci_95_upper,
        comparison.baseline.p99,
        comparison.treatment_label,
        comparison.treatment.mean,
        comparison.treatment.ci_95_lower,
        comparison.treatment.ci_95_upper,
        comparison.treatment.p99,
        comparison.speedup,
        comparison.cohens_d,
        effect_size_label(comparison.cohens_d),
        comparison.p_value,
        comparison.significant,
        timestamp
    );

    fs::write(&filename, json)?;
    Ok(filename)
}

/// Run a timed benchmark collecting measurements.
pub fn measure<F: FnMut()>(
    config: &BenchmarkConfig,
    configuration: &str,
    metric: &str,
    unit: &str,
    mut f: F,
) -> Measurements {
    // Warmup
    for _ in 0..config.warmup_iterations {
        f();
    }

    // Measurement
    let total = config.trials * config.measurement_iterations;
    let mut values = Vec::with_capacity(total);

    for _trial in 0..config.trials {
        for _iter in 0..config.measurement_iterations {
            let start = Instant::now();
            f();
            let elapsed = start.elapsed();
            values.push(duration_to_unit(elapsed, unit));
        }
    }

    Measurements {
        values,
        unit: unit.to_string(),
        configuration: configuration.to_string(),
        metric: metric.to_string(),
    }
}

/// Convert Duration to the specified unit.
fn duration_to_unit(d: Duration, unit: &str) -> f64 {
    match unit {
        "ns" | "nanoseconds" => d.as_nanos() as f64,
        "us" | "microseconds" => d.as_nanos() as f64 / 1_000.0,
        "ms" | "milliseconds" => d.as_nanos() as f64 / 1_000_000.0,
        "s" | "seconds" => d.as_secs_f64(),
        _ => d.as_nanos() as f64,
    }
}

/// Approximate normal CDF using Abramowitz & Stegun.
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    0.5 * (1.0 + sign * y)
}

/// Label for Cohen's d effect size.
fn effect_size_label(d: f64) -> &'static str {
    let d = d.abs();
    if d < 0.2 {
        "negligible"
    } else if d < 0.5 {
        "small"
    } else if d < 0.8 {
        "medium"
    } else if d < 1.2 {
        "large"
    } else {
        "very large"
    }
}

/// Capture system information for reproducibility.
pub fn capture_system_info() -> String {
    let mut info = String::new();
    info.push_str("{\n");

    // Try to get GPU info
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version,compute_cap,memory.total,clocks.current.graphics,power.draw", "--format=csv,noheader"])
        .output()
    {
        if output.status.success() {
            let gpu = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info.push_str(&format!("  \"gpu\": \"{}\",\n", gpu.replace('"', "\\\"")));
        }
    }

    // CUDA version
    if let Ok(output) = std::process::Command::new("nvcc").arg("--version").output() {
        if output.status.success() {
            let nvcc = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = nvcc.lines().find(|l| l.contains("release")) {
                info.push_str(&format!("  \"cuda\": \"{}\",\n", line.trim().replace('"', "\\\"")));
            }
        }
    }

    info.push_str(&format!("  \"rust\": \"{}\",\n", env!("CARGO_PKG_VERSION")));
    info.push_str(&format!("  \"ringkernel_cuda_arch\": \"{}\",\n",
        std::env::var("RINGKERNEL_CUDA_ARCH").unwrap_or_else(|_| "auto".to_string())));

    info.push_str("}\n");
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_basic() {
        let m = Measurements {
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            unit: "us".to_string(),
            configuration: "test".to_string(),
            metric: "latency".to_string(),
        };
        let s = Statistics::compute(&m);
        assert_eq!(s.n, 10);
        assert!((s.mean - 5.5).abs() < 0.01);
        assert!((s.median - 5.5).abs() < 0.01);
        assert!(s.p99 > s.p95);
        assert!(s.p95 > s.p50);
    }

    #[test]
    fn test_statistics_outlier_detection() {
        let mut values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        values.push(10000.0); // obvious outlier
        let m = Measurements {
            values,
            unit: "us".to_string(),
            configuration: "test".to_string(),
            metric: "latency".to_string(),
        };
        let s = Statistics::compute(&m);
        assert!(s.outlier_count > 0, "should detect the outlier");
    }

    #[test]
    fn test_comparison_significance() {
        let baseline = Measurements {
            values: vec![100.0; 100],
            unit: "us".to_string(),
            configuration: "traditional".to_string(),
            metric: "latency".to_string(),
        };
        let treatment = Measurements {
            values: vec![1.0; 100],
            unit: "us".to_string(),
            configuration: "persistent".to_string(),
            metric: "latency".to_string(),
        };
        let c = Comparison::compare(&baseline, &treatment);
        assert!((c.speedup - 100.0).abs() < 0.1);
        assert!(c.cohens_d > 1.0, "should be large effect size");
    }

    #[test]
    fn test_effect_size_labels() {
        assert_eq!(effect_size_label(0.1), "negligible");
        assert_eq!(effect_size_label(0.3), "small");
        assert_eq!(effect_size_label(0.6), "medium");
        assert_eq!(effect_size_label(1.0), "large");
        assert_eq!(effect_size_label(2.0), "very large");
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.001);
        assert!(normal_cdf(3.0) > 0.998);
        assert!(normal_cdf(-3.0) < 0.002);
    }
}
