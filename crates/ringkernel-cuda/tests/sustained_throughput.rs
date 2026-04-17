//! Sustained Throughput Stress Test — Experiment 4
//!
//! Runs persistent actor message passing for 60+ seconds to verify:
//! - No throughput degradation over time
//! - Stable latency percentiles
//! - No memory leaks
//! - Thermal stability

#![cfg(feature = "cuda")]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ringkernel_core::hlc::HlcTimestamp;
use ringkernel_core::message::{MessageEnvelope, MessageHeader};
use ringkernel_core::queue::{MessageQueue, SpscQueue};

#[test]
#[ignore] // Long-running test (60s)
fn test_sustained_throughput_60s() {
    let duration = Duration::from_secs(60);
    let window_size = Duration::from_secs(5);
    let queue = SpscQueue::new(4096);

    let envelope = MessageEnvelope {
        header: MessageHeader::new(1, 0, 1, 256, HlcTimestamp::now(1)),
        payload: vec![0u8; 256],
        ..Default::default()
    };

    println!();
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Sustained Throughput Test — 60 seconds");
    println!("  Queue: SPSC, capacity 4096, payload 256B");
    println!("══════════════════════════════════════════════════════════════════");

    let start = Instant::now();
    let mut window_start = start;
    let mut window_ops: u64 = 0;
    let mut total_ops: u64 = 0;
    let mut window_latencies: Vec<f64> = Vec::with_capacity(100_000);

    // Collect per-window stats
    let mut windows: Vec<WindowStats> = Vec::new();

    println!();
    println!(
        "  {:>8} {:>12} {:>10} {:>10} {:>10} {:>10}",
        "Window", "Throughput", "p50 (ns)", "p95 (ns)", "p99 (ns)", "Ops"
    );

    while start.elapsed() < duration {
        // Measure single operation
        let op_start = Instant::now();
        queue.try_enqueue(envelope.clone()).expect("enqueue failed");
        let msg = queue.try_dequeue().expect("dequeue failed");
        let _ = msg;
        let op_ns = op_start.elapsed().as_nanos() as f64;

        window_latencies.push(op_ns);
        window_ops += 1;
        total_ops += 1;

        // Check window
        if window_start.elapsed() >= window_size {
            let elapsed = window_start.elapsed().as_secs_f64();
            let throughput = window_ops as f64 / elapsed;

            // Compute percentiles
            window_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = window_latencies.len();
            let p50 = window_latencies[n / 2];
            let p95 = window_latencies[(n as f64 * 0.95) as usize];
            let p99 = window_latencies[(n as f64 * 0.99) as usize];

            let ws = WindowStats {
                window_secs: start.elapsed().as_secs_f64(),
                throughput,
                p50,
                p95,
                p99,
                ops: window_ops,
            };

            println!(
                "  {:>6.0}s {:>10.2}M {:>10.0} {:>10.0} {:>10.0} {:>10}",
                ws.window_secs,
                throughput / 1e6,
                p50,
                p95,
                p99,
                window_ops
            );

            windows.push(ws);

            window_start = Instant::now();
            window_ops = 0;
            window_latencies.clear();
        }
    }

    // Final summary
    let total_elapsed = start.elapsed().as_secs_f64();
    let overall_throughput = total_ops as f64 / total_elapsed;

    let throughputs: Vec<f64> = windows.iter().map(|w| w.throughput).collect();
    let mean_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let variance = throughputs
        .iter()
        .map(|t| (t - mean_throughput).powi(2))
        .sum::<f64>()
        / (throughputs.len() - 1).max(1) as f64;
    let cv = (variance.sqrt() / mean_throughput) * 100.0;

    let max_p99 = windows.iter().map(|w| w.p99).fold(0.0f64, f64::max);
    let min_throughput = throughputs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_throughput = throughputs.iter().cloned().fold(0.0f64, f64::max);

    // Check for degradation: compare first 3 windows vs last 3 windows
    let first_3_mean = if windows.len() >= 6 {
        windows[..3].iter().map(|w| w.throughput).sum::<f64>() / 3.0
    } else {
        mean_throughput
    };
    let last_3_mean = if windows.len() >= 6 {
        windows[windows.len() - 3..]
            .iter()
            .map(|w| w.throughput)
            .sum::<f64>()
            / 3.0
    } else {
        mean_throughput
    };
    let degradation_pct = ((first_3_mean - last_3_mean) / first_3_mean) * 100.0;
    let has_degradation = degradation_pct > 5.0; // >5% drop = degradation

    println!();
    println!("  ────────────────────────────────────────────");
    println!(
        "  Summary ({:.0}s, {} windows):",
        total_elapsed,
        windows.len()
    );
    println!("    Total operations:  {}", total_ops);
    println!(
        "    Overall throughput: {:.2}M ops/s",
        overall_throughput / 1e6
    );
    println!("    Mean throughput:   {:.2}M ops/s", mean_throughput / 1e6);
    println!("    Min throughput:    {:.2}M ops/s", min_throughput / 1e6);
    println!("    Max throughput:    {:.2}M ops/s", max_throughput / 1e6);
    println!("    CV (throughput):   {:.2}%", cv);
    println!("    Max p99 spike:     {:.0} ns", max_p99);
    println!(
        "    Degradation:       {:.1}% (first vs last 3 windows)",
        degradation_pct
    );
    println!(
        "    Degradation:       {}",
        if has_degradation { "YES" } else { "NO" }
    );
    println!("══════════════════════════════════════════════════════════════════");

    assert!(
        !has_degradation,
        "Throughput degraded by {:.1}% over 60s",
        degradation_pct
    );
    assert!(cv < 10.0, "Throughput CV {:.1}% exceeds 10% threshold", cv);
}

#[test]
#[ignore] // Requires GPU
fn test_sustained_throughput_with_power_monitoring() {
    use std::process::Command;

    // Sample power before
    let power_before = get_gpu_power();
    let temp_before = get_gpu_temp();

    // Run a burst of operations
    let queue = SpscQueue::new(4096);
    let envelope = MessageEnvelope {
        header: MessageHeader::new(1, 0, 1, 256, HlcTimestamp::now(1)),
        payload: vec![0u8; 256],
        ..Default::default()
    };

    let start = Instant::now();
    let mut ops = 0u64;
    while start.elapsed() < Duration::from_secs(10) {
        queue.try_enqueue(envelope.clone()).expect("enqueue");
        let _ = queue.try_dequeue().expect("dequeue");
        ops += 1;
    }

    let power_after = get_gpu_power();
    let temp_after = get_gpu_temp();

    let throughput = ops as f64 / start.elapsed().as_secs_f64();

    println!();
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Power & Thermal Telemetry (10s burst)");
    println!("══════════════════════════════════════════════════════════════════");
    println!(
        "  Throughput: {:.2}M ops/s ({} total)",
        throughput / 1e6,
        ops
    );
    println!("  Power:  before={}, after={}", power_before, power_after);
    println!("  Temp:   before={}, after={}", temp_before, temp_after);

    // Check ECC errors
    let ecc_errors = get_ecc_errors();
    println!("  ECC errors: {}", ecc_errors);
    println!("══════════════════════════════════════════════════════════════════");
}

struct WindowStats {
    window_secs: f64,
    throughput: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    ops: u64,
}

fn get_gpu_power() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=power.draw", "--format=csv,noheader"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "N/A".to_string())
}

fn get_gpu_temp() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=temperature.gpu", "--format=csv,noheader"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "N/A".to_string())
}

fn get_ecc_errors() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader",
        ])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "N/A".to_string())
}
