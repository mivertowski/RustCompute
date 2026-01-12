//! `ringkernel profile` command - Profile kernel performance.
//!
//! This command provides performance profiling for RingKernel kernels, including:
//! - Execution timing analysis
//! - Memory bandwidth measurement
//! - Timeline visualization (Chrome trace format)
//! - Flamegraph-compatible output

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

use crate::error::{CliError, CliResult};

/// Profile output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable text output.
    Text,
    /// JSON format for programmatic consumption.
    Json,
    /// Chrome trace format for timeline visualization.
    ChromeTrace,
    /// Flamegraph-compatible folded stacks.
    Flamegraph,
}

impl OutputFormat {
    /// Parse format from string.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            "chrome" | "trace" | "chrome-trace" => Ok(Self::ChromeTrace),
            "flamegraph" | "flame" | "folded" => Ok(Self::Flamegraph),
            _ => Err(format!(
                "Unknown format '{}'. Valid options: text, json, chrome-trace, flamegraph",
                s
            )),
        }
    }
}

/// Profiling configuration.
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// Kernel name or path to profile.
    pub kernel: String,
    /// Number of warmup iterations.
    pub warmup: u32,
    /// Number of measured iterations.
    pub iterations: u32,
    /// Output format.
    pub format: OutputFormat,
    /// Output file (None for stdout).
    pub output_file: Option<String>,
    /// Enable memory bandwidth analysis.
    pub memory_analysis: bool,
    /// Collect per-iteration timings.
    pub detailed: bool,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            kernel: String::new(),
            warmup: 10,
            iterations: 1000,
            format: OutputFormat::Text,
            output_file: None,
            memory_analysis: true,
            detailed: false,
        }
    }
}

/// Single timing measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingSample {
    /// Iteration index.
    pub iteration: u32,
    /// Duration in nanoseconds.
    pub duration_ns: u64,
    /// Timestamp from start in nanoseconds.
    pub timestamp_ns: u64,
}

/// Memory bandwidth sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySample {
    /// Bytes read.
    pub bytes_read: u64,
    /// Bytes written.
    pub bytes_written: u64,
    /// Duration in nanoseconds.
    pub duration_ns: u64,
    /// Read bandwidth in GB/s.
    pub read_bandwidth_gbps: f64,
    /// Write bandwidth in GB/s.
    pub write_bandwidth_gbps: f64,
}

/// Profiling results for a kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResult {
    /// Kernel name.
    pub kernel_name: String,
    /// Total iterations run.
    pub iterations: u32,
    /// Warmup iterations.
    pub warmup_iterations: u32,
    /// Minimum execution time in nanoseconds.
    pub min_ns: u64,
    /// Maximum execution time in nanoseconds.
    pub max_ns: u64,
    /// Mean execution time in nanoseconds.
    pub mean_ns: u64,
    /// Median execution time in nanoseconds.
    pub median_ns: u64,
    /// Standard deviation in nanoseconds.
    pub std_dev_ns: u64,
    /// 95th percentile in nanoseconds.
    pub p95_ns: u64,
    /// 99th percentile in nanoseconds.
    pub p99_ns: u64,
    /// Throughput in operations per second.
    pub ops_per_second: f64,
    /// Per-iteration samples (if detailed mode enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samples: Option<Vec<TimingSample>>,
    /// Memory bandwidth analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemorySample>,
    /// Backend used for profiling.
    pub backend: String,
    /// Profile timestamp.
    pub timestamp: String,
}

/// Chrome trace event format.
#[derive(Debug, Clone, Serialize)]
struct ChromeTraceEvent {
    name: String,
    cat: String,
    ph: String,
    ts: u64,
    dur: u64,
    pid: u32,
    tid: u32,
    args: HashMap<String, serde_json::Value>,
}

/// Execute the `profile` command.
pub async fn execute(
    kernel: &str,
    iterations: u32,
    format: &str,
    warmup: Option<u32>,
    output: Option<&str>,
    detailed: bool,
) -> CliResult<()> {
    let output_format = OutputFormat::from_str(format).map_err(|e| CliError::Validation(e))?;

    let config = ProfileConfig {
        kernel: kernel.to_string(),
        warmup: warmup.unwrap_or(10),
        iterations,
        format: output_format,
        output_file: output.map(|s| s.to_string()),
        memory_analysis: true,
        detailed,
    };

    println!(
        "{} Profiling kernel: {}",
        "→".bright_cyan(),
        kernel.bright_white()
    );
    println!(
        "  {} Warmup: {} iterations",
        "•".dimmed(),
        config.warmup.to_string().bright_yellow()
    );
    println!(
        "  {} Measured: {} iterations",
        "•".dimmed(),
        config.iterations.to_string().bright_yellow()
    );
    println!();

    // Check if kernel file exists (for file-based profiling)
    let kernel_path = Path::new(kernel);
    let is_file_based =
        kernel_path.exists() && kernel_path.extension().map_or(false, |e| e == "rs");

    let result = if is_file_based {
        profile_kernel_file(&config).await?
    } else {
        // Simulate profiling for demonstration
        profile_simulated(&config).await?
    };

    // Output results
    output_results(&result, &config)?;

    Ok(())
}

/// Profile a kernel from a Rust source file.
async fn profile_kernel_file(config: &ProfileConfig) -> CliResult<ProfileResult> {
    let source_path = Path::new(&config.kernel);

    // Read and parse the kernel file
    let source = fs::read_to_string(source_path)?;
    let syntax_tree = syn::parse_file(&source)?;

    // Find kernel functions
    let kernel_fns: Vec<_> = syntax_tree
        .items
        .iter()
        .filter_map(|item| {
            if let syn::Item::Fn(func) = item {
                let has_kernel_attr = func.attrs.iter().any(|attr| {
                    attr.path().is_ident("ring_kernel") || attr.path().is_ident("kernel")
                });
                if has_kernel_attr {
                    return Some(func.sig.ident.to_string());
                }
            }
            None
        })
        .collect();

    if kernel_fns.is_empty() {
        return Err(CliError::Validation(format!(
            "No kernel functions found in {}. Looking for #[ring_kernel] or #[kernel] attributes.",
            config.kernel
        )));
    }

    println!(
        "  {} Found kernels: {}",
        "✓".bright_green(),
        kernel_fns.join(", ").bright_white()
    );
    println!();

    // For now, profile using CPU timing simulation
    // In production, this would compile and execute the kernel
    profile_simulated(config).await
}

/// Simulate profiling for demonstration purposes.
async fn profile_simulated(config: &ProfileConfig) -> CliResult<ProfileResult> {
    let pb = ProgressBar::new((config.warmup + config.iterations) as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("█▓▒░  "),
    );

    // Warmup phase
    pb.set_message("Warming up...");
    for _ in 0..config.warmup {
        // Simulate kernel execution
        tokio::time::sleep(Duration::from_micros(50)).await;
        pb.inc(1);
    }

    // Measurement phase
    pb.set_message("Measuring...");
    let mut samples = Vec::with_capacity(config.iterations as usize);
    let start_time = Instant::now();

    for i in 0..config.iterations {
        let iter_start = Instant::now();

        // Simulate kernel execution with some variance
        let base_time = 100_000u64; // 100 microseconds base
        let variance = (i % 20) as u64 * 1000; // Add some variance
        tokio::time::sleep(Duration::from_nanos(base_time + variance)).await;

        let duration = iter_start.elapsed();
        let timestamp = start_time.elapsed();

        samples.push(TimingSample {
            iteration: i,
            duration_ns: duration.as_nanos() as u64,
            timestamp_ns: timestamp.as_nanos() as u64,
        });

        pb.inc(1);
    }

    pb.finish_with_message("Complete!");
    println!();

    // Calculate statistics
    let mut durations: Vec<u64> = samples.iter().map(|s| s.duration_ns).collect();
    durations.sort();

    let min_ns = *durations.first().unwrap_or(&0);
    let max_ns = *durations.last().unwrap_or(&0);
    let mean_ns = durations.iter().sum::<u64>() / durations.len().max(1) as u64;
    let median_ns = durations.get(durations.len() / 2).copied().unwrap_or(0);
    let p95_ns = durations
        .get((durations.len() * 95) / 100)
        .copied()
        .unwrap_or(0);
    let p99_ns = durations
        .get((durations.len() * 99) / 100)
        .copied()
        .unwrap_or(0);

    // Calculate standard deviation
    let variance = durations
        .iter()
        .map(|&d| {
            let diff = d as i64 - mean_ns as i64;
            (diff * diff) as u64
        })
        .sum::<u64>()
        / durations.len().max(1) as u64;
    let std_dev_ns = (variance as f64).sqrt() as u64;

    let ops_per_second = if mean_ns > 0 {
        1_000_000_000.0 / mean_ns as f64
    } else {
        0.0
    };

    // Simulate memory bandwidth (for demonstration)
    let memory = if config.memory_analysis {
        let bytes_per_iter = 1024 * 1024; // 1 MB per iteration
        let total_bytes = bytes_per_iter * config.iterations as u64;
        let total_duration_ns = samples.iter().map(|s| s.duration_ns).sum::<u64>();
        let bandwidth_gbps = (total_bytes as f64) / (total_duration_ns as f64);

        Some(MemorySample {
            bytes_read: total_bytes / 2,
            bytes_written: total_bytes / 2,
            duration_ns: total_duration_ns,
            read_bandwidth_gbps: bandwidth_gbps / 2.0,
            write_bandwidth_gbps: bandwidth_gbps / 2.0,
        })
    } else {
        None
    };

    Ok(ProfileResult {
        kernel_name: config.kernel.clone(),
        iterations: config.iterations,
        warmup_iterations: config.warmup,
        min_ns,
        max_ns,
        mean_ns,
        median_ns,
        std_dev_ns,
        p95_ns,
        p99_ns,
        ops_per_second,
        samples: if config.detailed { Some(samples) } else { None },
        memory,
        backend: "cpu".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

/// Output profiling results in the requested format.
fn output_results(result: &ProfileResult, config: &ProfileConfig) -> CliResult<()> {
    let output = match config.format {
        OutputFormat::Text => format_text(result),
        OutputFormat::Json => format_json(result)?,
        OutputFormat::ChromeTrace => format_chrome_trace(result)?,
        OutputFormat::Flamegraph => format_flamegraph(result),
    };

    if let Some(ref path) = config.output_file {
        let mut file = fs::File::create(path)?;
        file.write_all(output.as_bytes())?;
        println!(
            "{} Results written to: {}",
            "✓".bright_green(),
            path.bright_white()
        );
    } else {
        println!("{}", output);
    }

    Ok(())
}

/// Format results as human-readable text.
fn format_text(result: &ProfileResult) -> String {
    let mut output = String::new();

    output.push_str(&format!(
        "{}\n",
        "═══════════════════════════════════════════════════════════════".bright_cyan()
    ));
    output.push_str(&format!(
        "  {} Profile Results: {}\n",
        "📊".to_string(),
        result.kernel_name.bright_white()
    ));
    output.push_str(&format!(
        "{}\n\n",
        "═══════════════════════════════════════════════════════════════".bright_cyan()
    ));

    output.push_str(&format!("  {} Timing Statistics\n", "⏱️".to_string()));
    output.push_str(&format!(
        "    Iterations: {} (+ {} warmup)\n",
        result.iterations.to_string().bright_yellow(),
        result.warmup_iterations.to_string().dimmed()
    ));
    output.push_str(&format!(
        "    Min:        {}\n",
        format_duration(result.min_ns).bright_green()
    ));
    output.push_str(&format!(
        "    Max:        {}\n",
        format_duration(result.max_ns).bright_red()
    ));
    output.push_str(&format!(
        "    Mean:       {}\n",
        format_duration(result.mean_ns).bright_white()
    ));
    output.push_str(&format!(
        "    Median:     {}\n",
        format_duration(result.median_ns).bright_white()
    ));
    output.push_str(&format!(
        "    Std Dev:    {}\n",
        format_duration(result.std_dev_ns).dimmed()
    ));
    output.push_str(&format!(
        "    P95:        {}\n",
        format_duration(result.p95_ns).bright_yellow()
    ));
    output.push_str(&format!(
        "    P99:        {}\n",
        format_duration(result.p99_ns).bright_yellow()
    ));
    output.push_str(&format!(
        "    Throughput: {} ops/sec\n\n",
        format!("{:.2}", result.ops_per_second).bright_cyan()
    ));

    if let Some(ref mem) = result.memory {
        output.push_str(&format!("  {} Memory Bandwidth\n", "💾".to_string()));
        output.push_str(&format!(
            "    Read:       {:.2} GB/s\n",
            mem.read_bandwidth_gbps
        ));
        output.push_str(&format!(
            "    Write:      {:.2} GB/s\n",
            mem.write_bandwidth_gbps
        ));
        output.push_str(&format!(
            "    Total:      {:.2} GB/s\n\n",
            mem.read_bandwidth_gbps + mem.write_bandwidth_gbps
        ));
    }

    output.push_str(&format!(
        "  Backend: {}  |  Timestamp: {}\n",
        result.backend.bright_white(),
        result.timestamp.dimmed()
    ));

    output
}

/// Format duration in human-readable form.
fn format_duration(ns: u64) -> String {
    if ns >= 1_000_000_000 {
        format!("{:.2} s", ns as f64 / 1_000_000_000.0)
    } else if ns >= 1_000_000 {
        format!("{:.2} ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.2} µs", ns as f64 / 1_000.0)
    } else {
        format!("{} ns", ns)
    }
}

/// Format results as JSON.
fn format_json(result: &ProfileResult) -> CliResult<String> {
    serde_json::to_string_pretty(result)
        .map_err(|e| CliError::CodegenError(format!("JSON serialization failed: {}", e)))
}

/// Format results as Chrome trace format.
fn format_chrome_trace(result: &ProfileResult) -> CliResult<String> {
    let mut events = Vec::new();

    // Add metadata event
    events.push(ChromeTraceEvent {
        name: "thread_name".to_string(),
        cat: "__metadata".to_string(),
        ph: "M".to_string(),
        ts: 0,
        dur: 0,
        pid: 1,
        tid: 1,
        args: {
            let mut args = HashMap::new();
            args.insert("name".to_string(), serde_json::json!("Kernel Execution"));
            args
        },
    });

    // Add sample events if available
    if let Some(ref samples) = result.samples {
        for sample in samples {
            events.push(ChromeTraceEvent {
                name: result.kernel_name.clone(),
                cat: "kernel".to_string(),
                ph: "X".to_string(),
                ts: sample.timestamp_ns / 1000, // Convert to microseconds
                dur: sample.duration_ns / 1000,
                pid: 1,
                tid: 1,
                args: {
                    let mut args = HashMap::new();
                    args.insert("iteration".to_string(), serde_json::json!(sample.iteration));
                    args
                },
            });
        }
    }

    let trace = serde_json::json!({
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "kernel": result.kernel_name,
            "iterations": result.iterations,
            "backend": result.backend,
        }
    });

    serde_json::to_string_pretty(&trace)
        .map_err(|e| CliError::CodegenError(format!("Chrome trace serialization failed: {}", e)))
}

/// Format results as flamegraph-compatible folded stacks.
fn format_flamegraph(result: &ProfileResult) -> String {
    let mut output = String::new();

    // Create folded stack format
    // Format: stack;frame1;frame2 count
    if let Some(ref samples) = result.samples {
        // Group samples into buckets
        let bucket_size = 1000; // 1 microsecond buckets
        let mut buckets: HashMap<u64, u32> = HashMap::new();

        for sample in samples {
            let bucket = sample.duration_ns / bucket_size;
            *buckets.entry(bucket).or_insert(0) += 1;
        }

        for (bucket, count) in buckets {
            let duration_us = bucket * bucket_size / 1000;
            output.push_str(&format!(
                "{};{}_{}us {}\n",
                result.kernel_name, result.kernel_name, duration_us, count
            ));
        }
    } else {
        // Single aggregate entry
        let mean_us = result.mean_ns / 1000;
        output.push_str(&format!(
            "{};execution_{}us {}\n",
            result.kernel_name, mean_us, result.iterations
        ));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_parsing() {
        assert_eq!(OutputFormat::from_str("text").unwrap(), OutputFormat::Text);
        assert_eq!(OutputFormat::from_str("json").unwrap(), OutputFormat::Json);
        assert_eq!(
            OutputFormat::from_str("chrome-trace").unwrap(),
            OutputFormat::ChromeTrace
        );
        assert_eq!(
            OutputFormat::from_str("flamegraph").unwrap(),
            OutputFormat::Flamegraph
        );
        assert!(OutputFormat::from_str("invalid").is_err());
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(500), "500 ns");
        assert_eq!(format_duration(1_500), "1.50 µs");
        assert_eq!(format_duration(1_500_000), "1.50 ms");
        assert_eq!(format_duration(1_500_000_000), "1.50 s");
    }

    #[test]
    fn test_profile_result_serialization() {
        let result = ProfileResult {
            kernel_name: "test_kernel".to_string(),
            iterations: 100,
            warmup_iterations: 10,
            min_ns: 1000,
            max_ns: 5000,
            mean_ns: 2500,
            median_ns: 2400,
            std_dev_ns: 500,
            p95_ns: 4000,
            p99_ns: 4800,
            ops_per_second: 400_000.0,
            samples: None,
            memory: None,
            backend: "cpu".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
        };

        let json = format_json(&result).unwrap();
        assert!(json.contains("test_kernel"));
        assert!(json.contains("400000"));
    }
}
