//! Benchmark utility for accounting network kernels.
//!
//! Compares CPU vs GPU performance for analysis kernels,
//! including the Ring Kernel Actor System.

use ringkernel_accnet::actors::{CoordinatorConfig, GpuActorRuntime};
use ringkernel_accnet::cuda::{AnalysisRuntime, Backend};
use ringkernel_accnet::kernels::TransformationKernel;
use ringkernel_accnet::prelude::*;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Accounting Network Kernel Benchmarks                   ║");
    println!("║                  CPU vs GPU Analysis                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Initialize runtimes
    let cpu_runtime = AnalysisRuntime::with_backend(Backend::Cpu);
    let gpu_runtime = AnalysisRuntime::with_backend(Backend::Auto);
    let mut actor_runtime = GpuActorRuntime::new(CoordinatorConfig::default());

    println!("Runtime Configuration:");
    println!("  CPU Runtime: Active");

    let gpu_available = gpu_runtime.is_cuda_active();
    if gpu_available {
        let status = gpu_runtime.status();
        println!(
            "  GPU Runtime: {} (CC {}.{})",
            status.cuda_device_name.as_deref().unwrap_or("Unknown"),
            status.cuda_compute_capability.map(|c| c.0).unwrap_or(0),
            status.cuda_compute_capability.map(|c| c.1).unwrap_or(0)
        );
    } else {
        println!("  GPU Runtime: Not available (CUDA feature or hardware missing)");
    }

    let actor_status = actor_runtime.status();
    println!(
        "  Ring Kernel Actors: {} ({} kernels)",
        if actor_status.cuda_active {
            "GPU"
        } else {
            "CPU"
        },
        actor_status.kernels_launched
    );
    println!();

    // Setup
    let archetype = CompanyArchetype::retail_standard();
    let template = ChartOfAccountsTemplate::retail_standard();

    // Benchmark different network sizes
    let sizes = vec![
        (100, 10),     // Small: 100 entries, ~10 flows each
        (500, 50),     // Medium: 500 entries
        (1000, 100),   // Large: 1000 entries
        (5000, 500),   // Very Large: 5000 entries
        (10000, 1000), // Huge: 10000 entries
    ];

    println!("═══════════════════════════════════════════════════════════════");
    println!("                    TRANSFORMATION BENCHMARKS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Journal Entry → Transaction Flow Transformation:");
    println!("{:<12} {:>12} {:>15}", "Entries", "Time (ms)", "Throughput");
    println!("{:-<12} {:-<12} {:-<15}", "", "", "");

    for &(size, _) in &sizes {
        let mut generator =
            TransactionGenerator::new(archetype.clone(), GeneratorConfig::default());
        let entries = generator.generate_batch(size);
        let kernel = TransformationKernel::default();

        let start = Instant::now();
        let iterations = 50;
        for _ in 0..iterations {
            let _ = kernel.transform(&entries);
        }
        let elapsed = start.elapsed();
        let per_iter_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        let throughput = size as f64 / (elapsed.as_secs_f64() / iterations as f64);

        println!(
            "{:<12} {:>12.3} {:>12.0} e/s",
            size, per_iter_ms, throughput
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                     ANALYSIS BENCHMARKS");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create networks of various sizes
    let mut benchmark_networks = Vec::new();

    for &(entry_count, _) in &sizes {
        let mut generator =
            TransactionGenerator::new(archetype.clone(), GeneratorConfig::default());
        let entries = generator.generate_batch(entry_count);
        let transform_kernel = TransformationKernel::default();
        let result = transform_kernel.transform(&entries);

        let entity_id = uuid::Uuid::new_v4();
        let mut network = AccountingNetwork::new(entity_id, 2024, 1);

        for (idx, def) in template.accounts.iter().enumerate() {
            let node = AccountNode::new(uuid::Uuid::new_v4(), def.account_type, idx as u16);
            let metadata = AccountMetadata::new(&def.code, &def.name);
            network.add_account(node, metadata);
        }

        for flow in result.flows {
            network.add_flow(flow);
        }

        benchmark_networks.push((entry_count, network));
    }

    // CPU Benchmarks
    println!("CPU Analysis (baseline):");
    println!(
        "{:<12} {:>10} {:>12} {:>15}",
        "Accounts", "Flows", "Time (ms)", "Throughput"
    );
    println!("{:-<12} {:-<10} {:-<12} {:-<15}", "", "", "", "");

    let mut cpu_times = Vec::new();

    for (_, network) in &benchmark_networks {
        let start = Instant::now();
        let iterations = 50;
        for _ in 0..iterations {
            let _ = cpu_runtime.analyze(network);
        }
        let elapsed = start.elapsed();
        let per_iter_us = elapsed.as_micros() as f64 / iterations as f64;
        let per_iter_ms = per_iter_us / 1000.0;
        let throughput = network.flows.len() as f64 / (per_iter_us / 1_000_000.0);

        cpu_times.push(per_iter_us);

        println!(
            "{:<12} {:>10} {:>12.3} {:>12.0} f/s",
            network.accounts.len(),
            network.flows.len(),
            per_iter_ms,
            throughput
        );
    }

    // GPU Benchmarks
    if gpu_available {
        println!("\nGPU Analysis (CUDA):");
        println!(
            "{:<12} {:>10} {:>12} {:>15} {:>10}",
            "Accounts", "Flows", "Time (ms)", "Throughput", "Speedup"
        );
        println!(
            "{:-<12} {:-<10} {:-<12} {:-<15} {:-<10}",
            "", "", "", "", ""
        );

        for (i, (_, network)) in benchmark_networks.iter().enumerate() {
            let start = Instant::now();
            let iterations = 50;
            for _ in 0..iterations {
                let _ = gpu_runtime.analyze(network);
            }
            let elapsed = start.elapsed();
            let per_iter_us = elapsed.as_micros() as f64 / iterations as f64;
            let per_iter_ms = per_iter_us / 1000.0;
            let throughput = network.flows.len() as f64 / (per_iter_us / 1_000_000.0);
            let speedup = cpu_times[i] / per_iter_us;

            println!(
                "{:<12} {:>10} {:>12.3} {:>12.0} f/s {:>9.1}x",
                network.accounts.len(),
                network.flows.len(),
                per_iter_ms,
                throughput,
                speedup
            );
        }

        // Detailed kernel benchmarks
        #[cfg(feature = "cuda")]
        {
            println!("\n═══════════════════════════════════════════════════════════════");
            println!("                   DETAILED GPU BENCHMARKS");
            println!("═══════════════════════════════════════════════════════════════\n");

            // Use the largest network for detailed benchmarks
            if let Some((_, network)) = benchmark_networks.last() {
                if let Some(results) = gpu_runtime.run_benchmarks(network) {
                    println!(
                        "Device: {} (CC {}.{})",
                        results.device_name,
                        results.compute_capability.0,
                        results.compute_capability.1
                    );
                    println!();

                    println!("Individual Kernel Performance:");
                    println!(
                        "{:<25} {:>12} {:>12} {:>15}",
                        "Kernel", "Elements", "Time (µs)", "Throughput"
                    );
                    println!("{:-<25} {:-<12} {:-<12} {:-<15}", "", "", "", "");

                    for kernel in &results.kernels {
                        println!(
                            "{:<25} {:>12} {:>12} {:>12.2} ME/s",
                            kernel.name, kernel.elements, kernel.time_us, kernel.throughput_meps
                        );
                    }

                    println!();
                    println!("Summary:");
                    println!("  Total GPU time: {} µs", results.total_gpu_time_us);
                    println!("  Total CPU time: {} µs", results.total_cpu_time_us);
                    println!("  Overall speedup: {:.2}x", results.speedup);
                }
            }
        }
    }

    // Ring Kernel Actor System Benchmarks
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                 RING KERNEL ACTOR BENCHMARKS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Ring Kernel Actor-Based Analytics:");
    println!(
        "{:<12} {:>10} {:>12} {:>15} {:>10}",
        "Accounts", "Flows", "Time (ms)", "Throughput", "vs CPU"
    );
    println!(
        "{:-<12} {:-<10} {:-<12} {:-<15} {:-<10}",
        "", "", "", "", ""
    );

    let mut actor_times = Vec::new();

    for (i, (_, network)) in benchmark_networks.iter().enumerate() {
        let start = Instant::now();
        let iterations = 50;
        for _ in 0..iterations {
            let _ = actor_runtime.analyze(network);
        }
        let elapsed = start.elapsed();
        let per_iter_us = elapsed.as_micros() as f64 / iterations as f64;
        let per_iter_ms = per_iter_us / 1000.0;
        let throughput = network.flows.len() as f64 / (per_iter_us / 1_000_000.0);
        let speedup = cpu_times[i] / per_iter_us;

        actor_times.push(per_iter_us);

        println!(
            "{:<12} {:>10} {:>12.3} {:>12.0} f/s {:>9.1}x",
            network.accounts.len(),
            network.flows.len(),
            per_iter_ms,
            throughput,
            speedup
        );
    }

    // Ring Kernel Actor Statistics
    let final_status = actor_runtime.status();
    println!("\nRing Kernel Actor Statistics:");
    println!(
        "  Total Snapshots Processed: {}",
        final_status.coordinator_stats.snapshots_processed
    );
    println!(
        "  Total Processing Time: {} µs",
        final_status.coordinator_stats.total_processing_time_us
    );
    println!(
        "  Average Processing Time: {:.2} µs",
        final_status.coordinator_stats.avg_processing_time_us
    );
    println!(
        "  Fraud Patterns Detected: {}",
        final_status.coordinator_stats.total_fraud_patterns
    );
    println!(
        "  GAAP Violations Detected: {}",
        final_status.coordinator_stats.total_gaap_violations
    );
    println!(
        "  Suspense Accounts Flagged: {}",
        final_status.coordinator_stats.total_suspense_accounts
    );

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        BENCHMARK COMPLETE");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Summary
    println!("Notes:");
    println!("  - Throughput is measured in flows/sec (f/s) or elements/sec");
    println!("  - GPU speedup is relative to CPU baseline");
    println!("  - Times include host-device memory transfers");
    println!("  - Ring Kernel Actors use persistent GPU kernels with K2K messaging");
    if !gpu_available {
        println!("\n  To enable GPU benchmarks, build with: cargo build --features cuda");
    }
}
