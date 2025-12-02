//! Performance benchmark for WaveSim.
//!
//! Run with: cargo run -p ringkernel-wavesim --bin benchmark --release

use ringkernel_wavesim::simulation::{AcousticParams, SimulationGrid, TileKernelGrid};
use ringkernel::prelude::Backend;
use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         RingKernel WaveSim Performance Evaluation                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Part 1: Maximum Grid Size Tests
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PART 1: Maximum Stable Grid Size (CPU SimulationGrid)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let sizes = [8, 16, 32, 64, 128, 256, 512, 1024];
    let mut max_working = 8u32;

    println!("{:<12} {:>10} {:>12} {:>15}", "Grid Size", "Cells", "Init (ms)", "10 steps (ms)");
    println!("{}", "-".repeat(52));

    for &size in &sizes {
        let params = AcousticParams::new(343.0, 1.0);

        let init_start = Instant::now();
        let result = std::panic::catch_unwind(|| {
            SimulationGrid::new(size, size, params)
        });
        let init_time = init_start.elapsed();

        match result {
            Ok(mut grid) => {
                grid.inject_impulse(size / 2, size / 2, 1.0);

                let step_start = Instant::now();
                for _ in 0..10 {
                    grid.step();
                }
                let step_time = step_start.elapsed();

                max_working = size;
                println!(
                    "{:<12} {:>10} {:>12.2} {:>15.2}",
                    format!("{}x{}", size, size),
                    size * size,
                    init_time.as_secs_f64() * 1000.0,
                    step_time.as_secs_f64() * 1000.0
                );
            }
            Err(_) => {
                println!(
                    "{:<12} {:>10} {:>12} {:>15}",
                    format!("{}x{}", size, size),
                    size * size,
                    "PANIC",
                    "-"
                );
                break;
            }
        }
    }
    println!();
    println!("Maximum stable grid: {}x{} ({} cells)", max_working, max_working, max_working * max_working);
    println!();

    // =========================================================================
    // Part 2: Performance Benchmarks
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PART 2: Simulation Performance (1000 steps)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let bench_sizes = [32, 64, 128, 256];

    println!("{:<12} {:>10} {:>12} {:>12} {:>15}", "Grid Size", "Cells", "Total (ms)", "Steps/sec", "Cells*Steps/sec");
    println!("{}", "-".repeat(65));

    for &size in &bench_sizes {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = SimulationGrid::new(size, size, params);
        grid.inject_impulse(size / 2, size / 2, 1.0);

        let steps = 1000u32;
        let start = Instant::now();
        for _ in 0..steps {
            grid.step();
        }
        let elapsed = start.elapsed();

        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
        let cells_steps_per_sec = (size * size * steps) as f64 / elapsed.as_secs_f64();

        println!(
            "{:<12} {:>10} {:>12.1} {:>12.0} {:>15.0}",
            format!("{}x{}", size, size),
            size * size,
            total_ms,
            steps_per_sec,
            cells_steps_per_sec
        );
    }
    println!();

    // =========================================================================
    // Part 3: Cell Size Impact Analysis
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PART 3: Cell Size Impact on Real-Time Performance (64x64 grid)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    println!("The GUI targets 10ms of simulation time per frame at 60 FPS.");
    println!("Smaller cell sizes require more steps per frame to maintain real-time.");
    println!();

    let cell_sizes = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005];
    let grid_size = 64u32;
    let target_sim_time = 0.01; // 10ms of simulation per frame

    println!("{:<12} {:>12} {:>15} {:>12} {:>10}", "Cell Size", "Time Step", "Steps/Frame", "Est. FPS", "Status");
    println!("{}", "-".repeat(65));

    for &cell_size in &cell_sizes {
        let params = AcousticParams::new(343.0, cell_size);
        let mut grid = SimulationGrid::new(grid_size, grid_size, params.clone());
        grid.inject_impulse(grid_size / 2, grid_size / 2, 1.0);

        let dt = params.time_step;
        let steps_per_frame = ((target_sim_time / dt) as u32).clamp(1, 10000);

        // Benchmark actual performance
        let bench_frames = 100;
        let start = Instant::now();
        for _ in 0..bench_frames {
            for _ in 0..steps_per_frame {
                grid.step();
            }
        }
        let elapsed = start.elapsed();

        let actual_fps = bench_frames as f64 / elapsed.as_secs_f64();

        let status = if actual_fps >= 60.0 {
            "Excellent"
        } else if actual_fps >= 30.0 {
            "Good"
        } else if actual_fps >= 10.0 {
            "Acceptable"
        } else if actual_fps >= 2.0 {
            "Slow"
        } else {
            "Too slow"
        };

        println!(
            "{:<12.4} {:>12.6} {:>15} {:>12.1} {:>10}",
            cell_size,
            dt,
            steps_per_frame,
            actual_fps,
            status
        );
    }
    println!();

    // =========================================================================
    // Part 4: Memory Usage Estimate
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PART 4: Memory Usage Estimates");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // CellState: ~52 bytes (pressure, velocity, neighbors, flags)
    let cell_size_bytes = 52usize;

    println!("{:<12} {:>12} {:>15}", "Grid Size", "Cells", "Est. Memory");
    println!("{}", "-".repeat(42));

    for &size in &[32u32, 64, 128, 256, 512, 1024] {
        let cells = (size * size) as usize;
        let mem_bytes = cells * cell_size_bytes;
        let mem_str = if mem_bytes < 1024 {
            format!("{} B", mem_bytes)
        } else if mem_bytes < 1024 * 1024 {
            format!("{:.1} KB", mem_bytes as f64 / 1024.0)
        } else {
            format!("{:.1} MB", mem_bytes as f64 / (1024.0 * 1024.0))
        };

        println!(
            "{:<12} {:>12} {:>15}",
            format!("{}x{}", size, size),
            cells,
            mem_str
        );
    }
    println!();

    // =========================================================================
    // Part 5: Recommendations
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PART 5: Recommendations");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("Grid Size Recommendations:");
    println!("  - Interactive (60 FPS): up to 128x128 with cell_size >= 0.1m");
    println!("  - Smooth (30 FPS): up to 256x256 with cell_size >= 0.5m");
    println!("  - Acceptable (10 FPS): up to 512x512 with cell_size >= 1.0m");
    println!();
    println!("Cell Size Recommendations:");
    println!("  - For real-time: cell_size >= 0.05m (50mm)");
    println!("  - For high detail: cell_size >= 0.01m (10mm) at smaller grids");
    println!("  - Minimum practical: ~0.005m (5mm) causes significant slowdown");
    println!();
    println!("Note: The TileKernelGrid uses 16x16 tile actors with K2K messaging");
    println!("for halo exchange, showcasing the RingKernel actor model.");
    println!();

    // =========================================================================
    // Part 6: Tile Actor Grid Performance
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("PART 6: Tile Actor Grid Performance (K2K Messaging)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("TileKernelGrid: 16x16 tiles with K2K halo exchange.");
    println!("This showcases RingKernel's actor model with inter-tile messaging.");
    println!();

    let tile_sizes = [32, 64, 128];
    let tile_steps = 100;

    println!("{:<12} {:>10} {:>12} {:>12} {:>15} {:>15}",
             "Grid Size", "Tiles", "Total (ms)", "Steps/sec", "K2K Messages", "Msg/step");
    println!("{}", "-".repeat(85));

    for &size in &tile_sizes {
        let params = AcousticParams::new(343.0, 1.0);
        match TileKernelGrid::new(size, size, params, Backend::Cpu).await {
            Ok(mut grid) => {
                grid.inject_impulse(size / 2, size / 2, 1.0);

                let start = Instant::now();
                for _ in 0..tile_steps {
                    let _ = grid.step().await;
                }
                let elapsed = start.elapsed();

                let stats = grid.k2k_stats();
                let total_ms = elapsed.as_secs_f64() * 1000.0;
                let steps_per_sec = tile_steps as f64 / elapsed.as_secs_f64();
                let msg_per_step = stats.messages_delivered as f64 / tile_steps as f64;

                println!(
                    "{:<12} {:>10} {:>12.1} {:>12.0} {:>15} {:>15.1}",
                    format!("{}x{}", size, size),
                    grid.tile_count(),
                    total_ms,
                    steps_per_sec,
                    stats.messages_delivered,
                    msg_per_step
                );
            }
            Err(e) => {
                println!("{:<12} - Error: {}", format!("{}x{}", size, size), e);
            }
        }
    }
    println!();

    println!("Benchmark complete.");
}
