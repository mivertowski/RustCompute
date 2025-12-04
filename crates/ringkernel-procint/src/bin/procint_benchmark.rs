//! Process Intelligence Benchmark
//!
//! Measures throughput of GPU kernels for process mining.

use ringkernel_procint::fabric::{GeneratorConfig, ProcessEventGenerator, SectorTemplate};
use ringkernel_procint::kernels::{
    DfgConstructionKernel, PartialOrderKernel, PatternConfig, PatternDetectionKernel,
};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        RingKernel Process Intelligence Benchmark             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let batch_size = 10_000;
    let num_batches = 10;

    // Initialize generator
    let config = GeneratorConfig {
        events_per_second: 100_000,
        batch_size,
        concurrent_cases: 100,
        ..Default::default()
    };
    let mut generator = ProcessEventGenerator::new(
        SectorTemplate::Healthcare(ringkernel_procint::fabric::HealthcareConfig::default()),
        config,
    );

    // Initialize kernels
    let mut dfg_kernel = DfgConstructionKernel::new(64).with_cpu_only();
    let mut pattern_kernel = PatternDetectionKernel::new(PatternConfig::default()).with_cpu_only();
    let mut partial_order_kernel = PartialOrderKernel::new().with_cpu_only();

    println!("Configuration:");
    println!("  Batch size: {}", batch_size);
    println!("  Batches: {}", num_batches);
    println!("  Total events: {}", batch_size * num_batches);
    println!();

    // Warm-up
    println!("Warming up...");
    let events = generator.generate_batch(batch_size);
    let _ = dfg_kernel.process(&events);
    println!();

    // DFG Construction Benchmark
    println!("DFG Construction Kernel:");
    let mut total_time = 0u64;
    let mut total_events = 0usize;

    for i in 0..num_batches {
        let events = generator.generate_batch(batch_size);
        let start = Instant::now();
        let result = dfg_kernel.process(&events);
        let elapsed = start.elapsed().as_micros() as u64;
        total_time += elapsed;
        total_events += events.len();

        if i == 0 || i == num_batches - 1 {
            println!(
                "  Batch {}: {} events, {}μs, {:.0} eps, {} edges",
                i + 1,
                events.len(),
                elapsed,
                events.len() as f64 * 1_000_000.0 / elapsed as f64,
                result.dfg.edge_count()
            );
        }
    }

    let avg_eps = total_events as f64 * 1_000_000.0 / total_time as f64;
    println!(
        "  Average: {:.0} events/sec ({:.2}M eps)",
        avg_eps,
        avg_eps / 1_000_000.0
    );
    println!();

    // Pattern Detection Benchmark
    println!("Pattern Detection Kernel:");
    total_time = 0;
    let mut total_patterns = 0usize;

    for i in 0..num_batches {
        let events = generator.generate_batch(batch_size);
        let dfg_result = dfg_kernel.process(&events);

        let start = Instant::now();
        let result = pattern_kernel.detect(dfg_result.dfg.nodes());
        let elapsed = start.elapsed().as_micros() as u64;
        total_time += elapsed;
        total_patterns += result.patterns.len();

        if i == 0 || i == num_batches - 1 {
            println!(
                "  Batch {}: {} nodes, {}μs, {} patterns",
                i + 1,
                dfg_result.dfg.nodes().len(),
                elapsed,
                result.patterns.len()
            );
        }
    }

    println!("  Total patterns detected: {}", total_patterns);
    println!(
        "  Avg time per batch: {:.0}μs",
        total_time as f64 / num_batches as f64
    );
    println!();

    // Partial Order Derivation Benchmark
    println!("Partial Order Derivation Kernel:");
    total_time = 0;
    let mut total_traces = 0usize;

    for i in 0..num_batches {
        let events = generator.generate_batch(batch_size);

        let start = Instant::now();
        let result = partial_order_kernel.derive(&events);
        let elapsed = start.elapsed().as_micros() as u64;
        total_time += elapsed;
        total_traces += result.traces.len();

        if i == 0 || i == num_batches - 1 {
            println!(
                "  Batch {}: {} traces, {}μs, avg width {:.2}",
                i + 1,
                result.traces.len(),
                elapsed,
                result.avg_width()
            );
        }
    }

    println!("  Total traces: {}", total_traces);
    let traces_per_sec = total_traces as f64 * 1_000_000.0 / total_time as f64;
    println!("  Throughput: {:.0} traces/sec", traces_per_sec);
    println!();

    // Summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("Summary:");
    println!(
        "  DFG Construction: {:.2}M events/sec",
        avg_eps / 1_000_000.0
    );
    println!(
        "  Pattern Detection: {} patterns in {}ms",
        total_patterns,
        total_time / 1000
    );
    println!(
        "  Partial Order: {:.0}K traces/sec",
        traces_per_sec / 1000.0
    );
    println!();
    println!(
        "Backend: {}",
        if cfg!(feature = "cuda") {
            "CUDA"
        } else {
            "CPU"
        }
    );
}
