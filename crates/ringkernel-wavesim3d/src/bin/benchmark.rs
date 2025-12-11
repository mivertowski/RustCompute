//! GPU benchmark for 3D wave simulation.
//!
//! Compares performance between:
//! - CPU (parallel Rayon)
//! - GPU Stencil method
//! - GPU Actor method

use ringkernel_wavesim3d::simulation::{
    AcousticParams3D, ComputationMethod, Environment, SimulationConfig, SimulationEngine,
};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        RingKernel WaveSim3D - GPU Benchmark                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let width = 64;
    let height = 64;
    let depth = 64;
    let num_steps = 100;

    println!("Configuration:");
    println!(
        "  Grid: {}×{}×{} = {} cells",
        width,
        height,
        depth,
        width * height * depth
    );
    println!("  Steps: {}", num_steps);
    println!();

    // CPU benchmark
    println!("Running CPU benchmark...");
    let cpu_time = {
        let mut engine = SimulationEngine::new_cpu(
            width,
            height,
            depth,
            AcousticParams3D::new(Environment::default(), 0.1),
        );
        engine.inject_impulse(width / 2, height / 2, depth / 2, 1.0);

        let start = Instant::now();
        engine.step_n(num_steps);
        start.elapsed()
    };
    println!("  CPU time: {:?}", cpu_time);
    println!(
        "  CPU throughput: {:.2} Mcells/s",
        (width * height * depth * num_steps) as f64 / cpu_time.as_secs_f64() / 1e6
    );
    println!();

    // GPU Stencil benchmark
    #[cfg(feature = "cuda")]
    {
        println!("Running GPU Stencil benchmark...");
        match SimulationEngine::new_gpu(
            width,
            height,
            depth,
            AcousticParams3D::new(Environment::default(), 0.1),
        ) {
            Ok(mut engine) => {
                engine.inject_impulse(width / 2, height / 2, depth / 2, 1.0);

                // Warm up
                engine.step_n(10);

                let start = Instant::now();
                engine.step_n(num_steps);
                let gpu_time = start.elapsed();

                println!("  GPU Stencil time: {:?}", gpu_time);
                println!(
                    "  GPU Stencil throughput: {:.2} Mcells/s",
                    (width * height * depth * num_steps) as f64 / gpu_time.as_secs_f64() / 1e6
                );
                println!(
                    "  Speedup vs CPU: {:.1}x",
                    cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
                );
            }
            Err(e) => {
                println!("  GPU Stencil not available: {}", e);
            }
        }
        println!();

        // GPU Actor benchmark
        println!("Running GPU Actor benchmark...");
        let config = SimulationConfig {
            width,
            height,
            depth,
            cell_size: 0.1,
            prefer_gpu: true,
            computation_method: ComputationMethod::Actor,
            ..Default::default()
        };

        let mut engine = config.build();
        if engine.is_using_gpu() && engine.computation_method() == ComputationMethod::Actor {
            engine.inject_impulse(width / 2, height / 2, depth / 2, 1.0);

            // Warm up
            engine.step_n(10);

            let start = Instant::now();
            engine.step_n(num_steps);
            let actor_time = start.elapsed();

            println!("  GPU Actor time: {:?}", actor_time);
            println!(
                "  GPU Actor throughput: {:.2} Mcells/s",
                (width * height * depth * num_steps) as f64 / actor_time.as_secs_f64() / 1e6
            );
            println!(
                "  Speedup vs CPU: {:.1}x",
                cpu_time.as_secs_f64() / actor_time.as_secs_f64()
            );

            // Print actor stats
            if let Some(stats) = engine.actor_stats() {
                println!();
                println!("Actor Statistics:");
                println!("  Total cells: {}", stats.total_cells);
                println!("  Inbox capacity: {}", stats.inbox_capacity);
                println!(
                    "  Memory usage: {:.2} MB",
                    stats.memory_usage_bytes as f64 / 1e6
                );
            }
        } else {
            println!("  GPU Actor not available (falling back to CPU)");
        }
    }

    println!("\nBenchmark complete.");
}
