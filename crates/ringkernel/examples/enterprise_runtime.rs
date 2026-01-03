//! Enterprise Runtime Features Example
//!
//! This example demonstrates the enterprise features of RingKernel:
//! - Unified configuration with presets
//! - Lifecycle management (start, drain, shutdown)
//! - Health monitoring cycles
//! - Circuit breaker protection
//! - Graceful degradation
//! - Metrics export
//!
//! Run with: `cargo run -p ringkernel --example enterprise_runtime`

use ringkernel_core::prelude::*;
use ringkernel_core::config::ConfigBuilder;
use ringkernel_core::health::DegradationLevel;
use std::time::Duration;
use std::thread;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== RingKernel Enterprise Runtime Demo ===\n");

    // =========================================================================
    // Part 1: Configuration Presets
    // =========================================================================
    println!("--- Part 1: Configuration Presets ---\n");

    // Development preset (verbose logging, relaxed limits)
    let dev_runtime = RuntimeBuilder::new()
        .development()
        .build()?;
    println!(
        "Development config: env={:?}, tracing={}",
        dev_runtime.config().general.environment,
        dev_runtime.config().observability.tracing_enabled
    );

    // Production preset (optimized for reliability)
    let prod_runtime = RuntimeBuilder::new()
        .production()
        .build()?;
    println!(
        "Production config: env={:?}, tracing={}",
        prod_runtime.config().general.environment,
        prod_runtime.config().observability.tracing_enabled
    );

    // High-performance preset (minimal overhead)
    let perf_runtime = RuntimeBuilder::new()
        .high_performance()
        .build()?;
    println!(
        "High-perf config: env={:?}, tracing={}",
        perf_runtime.config().general.environment,
        perf_runtime.config().observability.tracing_enabled
    );

    // Custom configuration
    let custom_config = ConfigBuilder::new()
        .with_general(|g| {
            g.app_name("enterprise-demo")
                .app_version("1.0.0")
                .environment(ringkernel_core::config::Environment::Staging)
        })
        .with_health(|h| {
            h.check_interval(Duration::from_secs(5))
                .heartbeat_timeout(Duration::from_secs(30))
        })
        .with_multi_gpu(|m| {
            m.auto_select_device(true)
                .max_kernels_per_device(100)
                .enable_p2p(true)
        })
        .build()?;

    let custom_runtime = RuntimeBuilder::new()
        .with_config(custom_config)
        .build()?;
    println!(
        "Custom config: app={}, version={}",
        custom_runtime.config().general.app_name,
        custom_runtime.config().general.app_version
    );

    println!();

    // =========================================================================
    // Part 2: Lifecycle Management
    // =========================================================================
    println!("--- Part 2: Lifecycle Management ---\n");

    let runtime = RuntimeBuilder::new()
        .development()
        .build()?;

    // Check initial state
    println!("Initial state: {:?}", runtime.lifecycle_state());
    println!("Is accepting work: {}", runtime.is_accepting_work());

    // Start the runtime
    runtime.start()?;
    println!("After start: {:?}", runtime.lifecycle_state());
    println!("Is accepting work: {}", runtime.is_accepting_work());

    // Simulate some work
    runtime.record_kernel_launch();
    runtime.record_kernel_launch();
    runtime.record_messages(1000);
    runtime.record_checkpoint();

    // Check uptime
    thread::sleep(Duration::from_millis(50));
    println!("Uptime: {:?}", runtime.uptime());

    // Get application info
    let app_info = runtime.app_info();
    println!(
        "App: {} v{} ({})",
        app_info.name, app_info.version, app_info.environment
    );

    println!();

    // =========================================================================
    // Part 3: Health Monitoring
    // =========================================================================
    println!("--- Part 3: Health Monitoring ---\n");

    // Run health check cycle
    let health_result = runtime.run_health_check_cycle();
    println!("Health status: {:?}", health_result.status);
    println!("Circuit state: {:?}", health_result.circuit_state);
    println!("Degradation level: {:?}", health_result.degradation_level);

    // Run watchdog cycle
    let watchdog_result = runtime.run_watchdog_cycle();
    println!("Stale kernels: {}", watchdog_result.stale_kernels);

    // Check background task status
    let task_status = runtime.background_task_status();
    println!(
        "Last health check: {:?} ago",
        task_status.health_check_age.unwrap_or_default()
    );
    println!(
        "Last watchdog scan: {:?} ago",
        task_status.watchdog_scan_age.unwrap_or_default()
    );

    println!();

    // =========================================================================
    // Part 4: Circuit Breaker Protection
    // =========================================================================
    println!("--- Part 4: Circuit Breaker Protection ---\n");

    let circuit_guard = CircuitGuard::new(&runtime, "demo-operation");

    // Execute with circuit breaker protection
    let result: Result<i32> = circuit_guard.execute(|| {
        println!("  Executing protected operation...");
        Ok(42)
    });
    println!("Protected execution result: {:?}", result);

    // Check circuit breaker state
    let cb = runtime.circuit_breaker();
    println!("Circuit breaker state: {:?}", cb.state());

    // Simulate failures to see circuit breaker in action
    println!("\nSimulating failures...");
    for i in 0..5 {
        cb.record_failure();
        println!("  After failure {}: state={:?}", i + 1, cb.state());
    }

    println!();

    // =========================================================================
    // Part 5: Graceful Degradation
    // =========================================================================
    println!("--- Part 5: Graceful Degradation ---\n");

    let degradation_guard = DegradationGuard::new(&runtime);

    // Check what operations are allowed at each priority level
    println!("At Normal degradation level:");
    println!("  Low priority allowed: {}", degradation_guard.allow_operation(OperationPriority::Low));
    println!("  Normal priority allowed: {}", degradation_guard.allow_operation(OperationPriority::Normal));
    println!("  High priority allowed: {}", degradation_guard.allow_operation(OperationPriority::High));
    println!("  Critical priority allowed: {}", degradation_guard.allow_operation(OperationPriority::Critical));

    // Demonstrate level progression
    println!("\nDegradation level progression:");
    let mut level = DegradationLevel::Normal;
    for _ in 0..5 {
        let next = level.next_worse();
        println!("  {:?} -> {:?}", level, next);
        level = next;
    }

    println!();

    // =========================================================================
    // Part 6: Metrics Export
    // =========================================================================
    println!("--- Part 6: Metrics Export ---\n");

    // Get runtime metrics snapshot
    let metrics = runtime.metrics_snapshot();
    println!("Runtime Metrics:");
    println!("  Kernels launched: {}", metrics.kernels_launched);
    println!("  Messages processed: {}", metrics.messages_processed);
    println!("  Health checks: {}", metrics.health_checks_run);
    println!("  Uptime: {:.2}s", metrics.uptime_seconds);

    // Get statistics snapshot
    let stats = runtime.stats();
    println!("\nStatistics Snapshot:");
    println!("  Uptime: {:?}", stats.uptime);
    println!("  Checkpoints: {}", stats.checkpoints_created);
    println!("  Circuit trips: {}", stats.circuit_breaker_trips);

    // Export Prometheus metrics
    let prometheus_metrics = runtime.flush_metrics();
    println!("\nPrometheus metrics exported ({} bytes)", prometheus_metrics.len());

    println!();

    // =========================================================================
    // Part 7: Graceful Shutdown
    // =========================================================================
    println!("--- Part 7: Graceful Shutdown ---\n");

    // Request shutdown (transitions to Draining)
    runtime.request_shutdown()?;
    println!("After request_shutdown: {:?}", runtime.lifecycle_state());
    println!("Shutdown requested: {}", runtime.is_shutdown_requested());

    // Complete shutdown (transitions to Stopped)
    let shutdown_report = runtime.complete_shutdown()?;
    println!("\nShutdown Report:");
    println!("  State: {:?}", runtime.lifecycle_state());
    println!("  Shutdown duration: {:?}", shutdown_report.duration);
    println!("  Total uptime: {:?}", shutdown_report.total_uptime);
    println!("  Final kernels launched: {}", shutdown_report.final_stats.kernels_launched);
    println!("  Final messages: {}", shutdown_report.final_stats.messages_processed);

    println!("\n=== Enterprise Runtime Demo Complete ===");

    Ok(())
}
