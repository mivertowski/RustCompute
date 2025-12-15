//! GPU benchmark for 3D wave simulation.
//!
//! Compares performance between:
//! - CPU (parallel Rayon)
//! - GPU Stencil method
//! - GPU Actor method (per-cell actors)
//! - GPU Block Actor method (hybrid block-based actors)
//! - GPU Persistent method (true persistent kernel actor)

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
    let cpu_throughput = (width * height * depth * num_steps) as f64 / cpu_time.as_secs_f64() / 1e6;
    println!("  CPU time: {:?}", cpu_time);
    println!("  CPU throughput: {:.2} Mcells/s", cpu_throughput);
    println!();

    #[cfg(feature = "cuda")]
    {
        let mut results: Vec<(&str, f64, f64)> =
            vec![("CPU", cpu_time.as_secs_f64() * 1000.0, cpu_throughput)];

        // GPU Stencil benchmark
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
                let gpu_throughput =
                    (width * height * depth * num_steps) as f64 / gpu_time.as_secs_f64() / 1e6;

                println!("  GPU Stencil time: {:?}", gpu_time);
                println!("  GPU Stencil throughput: {:.2} Mcells/s", gpu_throughput);
                println!(
                    "  Speedup vs CPU: {:.1}x",
                    cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
                );
                results.push((
                    "GPU Stencil",
                    gpu_time.as_secs_f64() * 1000.0,
                    gpu_throughput,
                ));
            }
            Err(e) => {
                println!("  GPU Stencil not available: {}", e);
            }
        }
        println!();

        // GPU Block Actor benchmark
        println!("Running GPU Block Actor benchmark...");
        let block_config = SimulationConfig {
            width,
            height,
            depth,
            cell_size: 0.1,
            prefer_gpu: true,
            computation_method: ComputationMethod::BlockActor,
            ..Default::default()
        };

        let mut engine = block_config.build();
        if engine.is_using_gpu() && engine.computation_method() == ComputationMethod::BlockActor {
            engine.inject_impulse(width / 2, height / 2, depth / 2, 1.0);

            // Warm up
            engine.step_n(10);

            let start = Instant::now();
            engine.step_n(num_steps);
            let block_actor_time = start.elapsed();
            let block_throughput =
                (width * height * depth * num_steps) as f64 / block_actor_time.as_secs_f64() / 1e6;

            println!("  GPU Block Actor time: {:?}", block_actor_time);
            println!(
                "  GPU Block Actor throughput: {:.2} Mcells/s",
                block_throughput
            );
            println!(
                "  Speedup vs CPU: {:.1}x",
                cpu_time.as_secs_f64() / block_actor_time.as_secs_f64()
            );

            // Print block actor stats
            if let Some(stats) = engine.block_actor_stats() {
                println!();
                println!("Block Actor Statistics:");
                println!(
                    "  Blocks: {} ({} × {} × {})",
                    stats.num_blocks, stats.block_dims.0, stats.block_dims.1, stats.block_dims.2
                );
                println!("  Cells per block: {}", stats.cells_per_block);
                println!("  Total cells: {}", stats.total_cells);
                println!(
                    "  Memory usage: {:.2} MB",
                    stats.memory_usage_bytes as f64 / 1e6
                );
            }
            results.push((
                "GPU Block Actor",
                block_actor_time.as_secs_f64() * 1000.0,
                block_throughput,
            ));
        } else {
            println!("  GPU Block Actor not available (falling back to CPU)");
        }
        println!();

        // GPU Persistent benchmark (true persistent kernel actor)
        #[cfg(feature = "cuda-codegen")]
        {
            println!("Running GPU Persistent benchmark...");
            println!("  (Single persistent kernel - true GPU native actor paradigm)");

            // Use smaller grid for persistent - must fit cooperative limits (48 blocks max)
            // 24×24×24 with 8×8×8 tiles = 3×3×3 = 27 blocks
            let persistent_width = 24;
            let persistent_height = 24;
            let persistent_depth = 24;

            let persistent_config = SimulationConfig {
                width: persistent_width,
                height: persistent_height,
                depth: persistent_depth,
                cell_size: 0.1,
                prefer_gpu: true,
                computation_method: ComputationMethod::Persistent,
                ..Default::default()
            };

            let mut engine = persistent_config.build();
            if engine.is_using_gpu() && engine.computation_method() == ComputationMethod::Persistent {
                engine.inject_impulse(persistent_width / 2, persistent_height / 2, persistent_depth / 2, 1.0);

                // Warm up
                engine.step_n(10);

                let start = Instant::now();
                engine.step_n(num_steps);
                let persistent_time = start.elapsed();
                let persistent_throughput =
                    (persistent_width * persistent_height * persistent_depth * num_steps) as f64 / persistent_time.as_secs_f64() / 1e6;

                println!("  Grid: {}×{}×{} (fits cooperative limits)", persistent_width, persistent_height, persistent_depth);
                println!("  GPU Persistent time: {:?}", persistent_time);
                println!(
                    "  GPU Persistent throughput: {:.2} Mcells/s",
                    persistent_throughput
                );

                // Print persistent stats
                if let Some(stats) = engine.persistent_stats() {
                    println!();
                    println!("Persistent Kernel Statistics:");
                    println!("  Current step: {}", stats.current_step);
                    println!("  Total energy: {:.6}", stats.total_energy);
                    println!("  Messages processed: {}", stats.messages_processed);
                    println!("  K2K sent: {}", stats.k2k_sent);
                    println!("  K2K received: {}", stats.k2k_received);
                    println!("  Kernel running: {}", stats.is_running);
                }

                // For comparison, show normalized throughput vs same-size CPU
                let cpu_small_time = {
                    let mut cpu_engine = SimulationEngine::new_cpu(
                        persistent_width,
                        persistent_height,
                        persistent_depth,
                        AcousticParams3D::new(Environment::default(), 0.1),
                    );
                    cpu_engine.inject_impulse(persistent_width / 2, persistent_height / 2, persistent_depth / 2, 1.0);
                    let start = Instant::now();
                    cpu_engine.step_n(num_steps);
                    start.elapsed()
                };
                println!(
                    "  Speedup vs CPU (same grid): {:.1}x",
                    cpu_small_time.as_secs_f64() / persistent_time.as_secs_f64()
                );

                results.push((
                    "GPU Persistent",
                    persistent_time.as_secs_f64() * 1000.0,
                    persistent_throughput,
                ));
            } else {
                println!("  GPU Persistent not available (falling back to CPU)");
                println!("  Requires cuda-codegen feature and nvcc compiler");
            }
            println!();
        }

        // GPU Actor benchmark (per-cell)
        println!("Running GPU Actor (per-cell) benchmark...");
        let actor_config = SimulationConfig {
            width,
            height,
            depth,
            cell_size: 0.1,
            prefer_gpu: true,
            computation_method: ComputationMethod::Actor,
            ..Default::default()
        };

        let mut engine = actor_config.build();
        if engine.is_using_gpu() && engine.computation_method() == ComputationMethod::Actor {
            engine.inject_impulse(width / 2, height / 2, depth / 2, 1.0);

            // Warm up
            engine.step_n(10);

            let start = Instant::now();
            engine.step_n(num_steps);
            let actor_time = start.elapsed();
            let actor_throughput =
                (width * height * depth * num_steps) as f64 / actor_time.as_secs_f64() / 1e6;

            println!("  GPU Actor time: {:?}", actor_time);
            println!("  GPU Actor throughput: {:.2} Mcells/s", actor_throughput);
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
            results.push((
                "GPU Actor (per-cell)",
                actor_time.as_secs_f64() * 1000.0,
                actor_throughput,
            ));
        } else {
            println!("  GPU Actor not available (falling back to CPU)");
        }

        // Print summary table
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║                     Performance Summary                          ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║ Method              │ Time (ms)  │ Throughput   │ vs CPU        ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        for (name, time_ms, throughput) in &results {
            let speedup = cpu_time.as_secs_f64() * 1000.0 / time_ms;
            println!(
                "║ {:18} │ {:>10.3} │ {:>8.2} Mc/s │ {:>8.1}x     ║",
                name, time_ms, throughput, speedup
            );
        }
        println!("╚══════════════════════════════════════════════════════════════════╝");

        // Calculate Block Actor vs Actor improvement
        if results.len() >= 3 {
            let block_throughput = results
                .iter()
                .find(|(n, _, _)| *n == "GPU Block Actor")
                .map(|(_, _, t)| *t);
            let actor_throughput = results
                .iter()
                .find(|(n, _, _)| *n == "GPU Actor (per-cell)")
                .map(|(_, _, t)| *t);
            if let (Some(block_t), Some(actor_t)) = (block_throughput, actor_throughput) {
                println!(
                    "\nBlock Actor vs Per-Cell Actor: {:.1}x faster",
                    block_t / actor_t
                );
            }
        }
    }

    // Additional benchmark with smaller grid that fits cooperative limits
    #[cfg(feature = "cooperative")]
    {
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║     Cooperative Launch Test (Small Grid)                         ║");
        println!("╚══════════════════════════════════════════════════════════════════╝\n");

        // Use a grid size that fits within 144 concurrent blocks
        // 40×40×40 = 64000 cells, 5×5×5 = 125 blocks
        let coop_width = 40;
        let coop_height = 40;
        let coop_depth = 40;
        let coop_steps = 100;

        println!("Configuration for cooperative test:");
        println!(
            "  Grid: {}×{}×{} = {} cells",
            coop_width,
            coop_height,
            coop_depth,
            coop_width * coop_height * coop_depth
        );
        println!("  Blocks: 5×5×5 = 125 (within 144 limit)");
        println!("  Steps: {}", coop_steps);
        println!();

        let coop_config = SimulationConfig {
            width: coop_width,
            height: coop_height,
            depth: coop_depth,
            cell_size: 0.1,
            prefer_gpu: true,
            computation_method: ComputationMethod::BlockActor,
            ..Default::default()
        };

        let mut engine = coop_config.build();
        if engine.is_using_gpu() && engine.computation_method() == ComputationMethod::BlockActor {
            engine.inject_impulse(coop_width / 2, coop_height / 2, coop_depth / 2, 1.0);

            // Check if cooperative mode is available
            if let Some(stats) = engine.block_actor_stats() {
                println!("Block Actor initialized:");
                println!("  Blocks: {}", stats.num_blocks);
                if let Some(max_coop) = stats.max_cooperative_blocks {
                    println!("  Max cooperative blocks: {}", max_coop);
                    if stats.num_blocks as u32 <= max_coop {
                        println!("  ✓ Grid fits within cooperative limits!");
                    } else {
                        println!("  ✗ Grid too large for cooperative launch");
                    }
                }
            }
            println!();

            // Warm up
            engine.step_n(10);

            let start = Instant::now();
            engine.step_n(coop_steps);
            let coop_time = start.elapsed();
            let coop_throughput = (coop_width * coop_height * coop_depth * coop_steps) as f64
                / coop_time.as_secs_f64()
                / 1e6;

            println!("Results:");
            println!("  Time: {:?}", coop_time);
            println!("  Throughput: {:.2} Mcells/s", coop_throughput);
        }
    }

    println!("\nBenchmark complete.");
}
