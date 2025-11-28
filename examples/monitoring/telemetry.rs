//! # Real-Time Telemetry and Monitoring
//!
//! This example demonstrates RingKernel's telemetry and observability
//! features for production monitoring.
//!
//! ## Telemetry Components
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     Telemetry Pipeline                       │
//! │                                                              │
//! │  ┌─────────┐    ┌─────────────┐    ┌───────────────────┐   │
//! │  │ Kernels │───▶│  Collector  │───▶│    Subscribers    │   │
//! │  │ (GPU)   │    │             │    │ - Dashboard       │   │
//! │  └─────────┘    │ - Metrics   │    │ - Alerting        │   │
//! │                 │ - Latency   │    │ - Storage         │   │
//! │                 │ - Errors    │    └───────────────────┘   │
//! │                 └─────────────┘                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Metrics Collected
//!
//! - **Throughput**: Messages processed per second
//! - **Latency**: P50, P95, P99 processing times
//! - **Queue depths**: Input/output queue utilization
//! - **Error rates**: Drop rates and error counts
//! - **Resource usage**: GPU memory, kernel count
//!
//! ## Run this example:
//! ```bash
//! cargo run --example telemetry
//! ```

use ringkernel::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::interval;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Real-Time Telemetry Example ===\n");

    // Create telemetry configuration
    let telemetry_config = TelemetryConfig {
        collection_interval_ms: 100,
        max_history_samples: 1000,
        channel_buffer_size: 256,
        enable_histograms: true,
        drop_rate_alert_threshold: 0.01,
        latency_alert_threshold_us: 10_000,
    };

    println!("Telemetry configuration:");
    println!("  Collection interval: {} ms", telemetry_config.collection_interval_ms);
    println!("  History samples: {}", telemetry_config.max_history_samples);
    println!("  Alert threshold (drop rate): {:.1}%", telemetry_config.drop_rate_alert_threshold * 100.0);
    println!("  Alert threshold (latency): {} µs\n", telemetry_config.latency_alert_threshold_us);

    // Create telemetry pipeline
    let pipeline = TelemetryPipeline::new(telemetry_config);

    // Create metrics collector
    let collector = MetricsCollector::new();

    // ====== Basic Telemetry Buffer ======
    println!("=== Telemetry Buffer Demo ===\n");
    demonstrate_telemetry_buffer();

    // ====== Metrics Collection ======
    println!("\n=== Metrics Collection Demo ===\n");
    demonstrate_metrics_collection(&collector);

    // ====== Latency Histograms ======
    println!("\n=== Latency Histogram Demo ===\n");
    demonstrate_histograms(&collector);

    // ====== Alert System ======
    println!("\n=== Alert System Demo ===\n");
    demonstrate_alerts(&pipeline).await;

    // ====== Real-time Monitoring Loop ======
    println!("\n=== Real-time Monitoring Demo ===\n");
    demonstrate_monitoring_loop(&collector).await;

    println!("\n=== Example completed! ===");
    Ok(())
}

fn demonstrate_telemetry_buffer() {
    // Create a telemetry buffer (64 bytes, GPU-friendly)
    let mut buffer = TelemetryBuffer::new();

    println!("TelemetryBuffer (64 bytes, cache-aligned):");
    println!("  Size: {} bytes", std::mem::size_of::<TelemetryBuffer>());
    println!("  Alignment: {} bytes\n", std::mem::align_of::<TelemetryBuffer>());

    // Simulate kernel telemetry updates
    for i in 1..=100 {
        buffer.messages_processed += 1;
        buffer.total_latency_us += (100 + i * 5) as u64;
        buffer.min_latency_us = buffer.min_latency_us.min(100);
        buffer.max_latency_us = buffer.max_latency_us.max((100 + i * 5) as u64);
    }

    // Add some dropped messages
    buffer.messages_dropped = 3;

    println!("After processing 100 messages:");
    println!("  Processed: {}", buffer.messages_processed);
    println!("  Dropped: {}", buffer.messages_dropped);
    println!("  Avg latency: {:.2} µs", buffer.avg_latency_us());
    println!("  Min latency: {} µs", buffer.min_latency_us);
    println!("  Max latency: {} µs", buffer.max_latency_us);
    println!("  Drop rate: {:.2}%", buffer.drop_rate() * 100.0);
}

fn demonstrate_metrics_collection(collector: &MetricsCollector) {
    let kernel_id = KernelId::new("compute_kernel");

    println!("Recording metrics for kernel '{}'...\n", kernel_id);

    // Record various metrics
    for i in 0..50 {
        // Messages with varying latencies
        let latency = 100 + (i % 10) * 50; // 100-550 µs
        collector.record_message_processed(&kernel_id, latency);

        // Occasional drops
        if i % 20 == 0 {
            collector.record_message_dropped(&kernel_id);
        }

        // Occasional errors
        if i % 30 == 0 {
            collector.record_error(&kernel_id, 1001); // Example error code
        }
    }

    // Retrieve telemetry
    if let Some(telemetry) = collector.get_telemetry(&kernel_id) {
        println!("Collected telemetry:");
        println!("  Messages processed: {}", telemetry.messages_processed);
        println!("  Messages dropped: {}", telemetry.messages_dropped);
        println!("  Avg latency: {:.2} µs", telemetry.avg_latency_us());
        println!("  Min latency: {} µs", telemetry.min_latency_us);
        println!("  Max latency: {} µs", telemetry.max_latency_us);
        println!("  Drop rate: {:.2}%", telemetry.drop_rate() * 100.0);
        println!("  Last error: {}", telemetry.last_error);
    }
}

fn demonstrate_histograms(collector: &MetricsCollector) {
    let kernel_id = KernelId::new("histogram_demo");

    println!("Recording latency distribution...\n");

    // Generate realistic latency distribution (log-normal like)
    let latencies = [
        // Most messages fast (100-200µs)
        100, 120, 110, 150, 130, 140, 180, 160, 190, 200,
        105, 115, 125, 135, 145, 155, 165, 175, 185, 195,
        // Some medium (200-500µs)
        250, 300, 350, 400, 450, 280, 320, 380,
        // Few slow (500-1000µs)
        600, 750, 900,
        // Rare outliers
        1500, 2000,
    ];

    for &latency in &latencies {
        collector.record_message_processed(&kernel_id, latency);
    }

    // Get histogram percentiles
    if let Some(histogram) = collector.get_histogram(&kernel_id) {
        println!("Latency percentiles:");
        println!("  P50: {} µs", histogram.percentile(50.0));
        println!("  P90: {} µs", histogram.percentile(90.0));
        println!("  P95: {} µs", histogram.percentile(95.0));
        println!("  P99: {} µs", histogram.percentile(99.0));
        println!("  Max: {} µs", histogram.percentile(100.0));
    }
}

async fn demonstrate_alerts(pipeline: &Arc<TelemetryPipeline>) {
    println!("Subscribing to telemetry alerts...\n");

    // Subscribe to alerts
    let mut alert_receiver = pipeline.subscribe();

    // Spawn alert handler
    let alert_handler = tokio::spawn(async move {
        println!("Alert handler started. Waiting for alerts...\n");

        // Wait for a few alerts
        let mut count = 0;
        while count < 3 {
            match tokio::time::timeout(Duration::from_millis(500), alert_receiver.recv()).await {
                Ok(Ok(event)) => {
                    match event {
                        TelemetryEvent::Snapshot(snapshot) => {
                            println!("Received snapshot:");
                            println!("  Active kernels: {}", snapshot.aggregate.active_kernels);
                            println!("  Throughput: {:.2}/s", snapshot.aggregate.throughput);
                        }
                        TelemetryEvent::Alert(alert) => {
                            println!("ALERT: {:?}", alert.alert_type);
                            println!("  Severity: {:?}", alert.severity);
                            println!("  Message: {}", alert.message);
                            if let Some(kernel) = &alert.kernel_id {
                                println!("  Kernel: {}", kernel);
                            }
                            count += 1;
                        }
                    }
                }
                Ok(Err(_)) => break,
                Err(_) => {
                    // Timeout - simulate generating an alert
                    println!("  (No alerts received - in production, alerts come from kernels)");
                    count += 1;
                }
            }
        }
    });

    // Wait for handler to complete
    let _ = tokio::time::timeout(Duration::from_secs(2), alert_handler).await;

    println!("\nAlert types available:");
    println!("  - HighDropRate: When drop rate exceeds threshold");
    println!("  - HighLatency: When avg latency exceeds threshold");
    println!("  - KernelError: When kernel reports an error");
    println!("  - QueueFull: When input queue is at capacity");
}

async fn demonstrate_monitoring_loop(collector: &MetricsCollector) {
    println!("Running monitoring loop for 3 seconds...\n");

    let kernel_ids = vec![
        KernelId::new("kernel_a"),
        KernelId::new("kernel_b"),
        KernelId::new("kernel_c"),
    ];

    // Simulate activity
    let collector = Arc::new(collector.clone());
    let collector_sim = collector.clone();
    let kernel_ids_sim = kernel_ids.clone();

    let simulator = tokio::spawn(async move {
        let mut count = 0;
        loop {
            for kernel_id in &kernel_ids_sim {
                collector_sim.record_message_processed(
                    kernel_id,
                    100 + (count % 50) * 10,
                );
            }
            count += 1;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Monitoring loop
    let mut monitor_interval = interval(Duration::from_secs(1));
    let start = Instant::now();

    while start.elapsed() < Duration::from_secs(3) {
        monitor_interval.tick().await;

        println!("--- Monitoring snapshot @ {:?} ---", start.elapsed());

        for kernel_id in &kernel_ids {
            if let Some(telemetry) = collector.get_telemetry(kernel_id) {
                print_kernel_status(kernel_id, &telemetry);
            }
        }
        println!();
    }

    simulator.abort();
}

fn print_kernel_status(kernel_id: &KernelId, telemetry: &TelemetryBuffer) {
    let status = if telemetry.drop_rate() > 0.01 {
        "WARNING"
    } else {
        "OK"
    };

    println!(
        "  {} [{}]: {} msgs, {:.1}µs avg, {:.2}% drop",
        kernel_id,
        status,
        telemetry.messages_processed,
        telemetry.avg_latency_us(),
        telemetry.drop_rate() * 100.0
    );
}
