//! Comprehensive benchmark for WaveSim across all backends.
//!
//! Run with: cargo run -p ringkernel-wavesim --bin full_benchmark --release --features "cuda,wgpu"

use ringkernel_wavesim::simulation::{AcousticParams, SimulationGrid, TileKernelGrid};
use ringkernel::prelude::Backend;
use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║   RingKernel WaveSim Full Benchmark (CPU, CUDA, WGPU, GPU-Persistent)   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // Section 1: CPU SimulationGrid (SoA + SIMD + Rayon)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("SECTION 1: CPU SimulationGrid (SoA + SIMD + Rayon)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    benchmark_simulation_grid().await;

    // =========================================================================
    // Section 2: TileKernelGrid CPU (Actor Model + K2K Messaging)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("SECTION 2: TileKernelGrid CPU (Actor Model + K2K Messaging)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    benchmark_tile_grid_cpu().await;

    // =========================================================================
    // Section 3: TileKernelGrid with WGPU GPU Compute (Legacy - upload/download per step)
    // =========================================================================
    #[cfg(feature = "wgpu")]
    {
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("SECTION 3: TileKernelGrid WGPU Legacy (upload/download per step)");
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!();

        benchmark_tile_grid_gpu().await;
    }

    #[cfg(not(feature = "wgpu"))]
    {
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("SECTION 3: WGPU GPU Compute (SKIPPED - feature not enabled)");
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!();
        println!("Run with --features wgpu to enable GPU benchmarks.");
        println!();
    }

    // =========================================================================
    // Section 4: GPU-Persistent CUDA (state stays on GPU, halo-only transfers)
    // =========================================================================
    #[cfg(feature = "cuda")]
    {
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("SECTION 4: TileKernelGrid CUDA GPU-Persistent (halo-only transfers)");
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!();

        benchmark_cuda_persistent().await;
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("SECTION 4: CUDA GPU-Persistent (SKIPPED - feature not enabled)");
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!();
        println!("Run with --features cuda to enable CUDA benchmarks.");
        println!();
    }

    // =========================================================================
    // Section 5: GPU-Persistent WGPU (state stays on GPU, halo-only transfers)
    // =========================================================================
    #[cfg(feature = "wgpu")]
    {
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("SECTION 5: TileKernelGrid WGPU GPU-Persistent (halo-only transfers)");
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!();

        benchmark_wgpu_persistent().await;
    }

    // =========================================================================
    // Section 6: Comparison Summary
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("SECTION 6: Full Comparison (64x64 grid, 1000 steps)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    benchmark_full_comparison().await;

    println!();
    println!("Benchmark complete!");
    println!();
}

async fn benchmark_simulation_grid() {
    let sizes = [32u32, 64, 128, 256, 512];
    let steps = 1000u32;

    println!("Running {} steps per grid size...", steps);
    println!();
    println!("{:<12} {:>10} {:>12} {:>15} {:>20}",
             "Grid Size", "Cells", "Time (ms)", "Steps/sec", "Throughput (M cells/s)");
    println!("{}", "-".repeat(75));

    for &size in &sizes {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = SimulationGrid::new(size, size, params);
        grid.inject_impulse(size / 2, size / 2, 1.0);

        // Warmup
        for _ in 0..10 {
            grid.step();
        }

        let start = Instant::now();
        for _ in 0..steps {
            grid.step();
        }
        let elapsed = start.elapsed();

        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
        let throughput = (size as f64 * size as f64 * steps as f64) / elapsed.as_secs_f64() / 1_000_000.0;

        println!(
            "{:<12} {:>10} {:>12.2} {:>15.0} {:>20.1}",
            format!("{}x{}", size, size),
            size * size,
            total_ms,
            steps_per_sec,
            throughput
        );
    }
    println!();
}

async fn benchmark_tile_grid_cpu() {
    // Conservative sizes for tile grid to avoid crashes
    let sizes = [32u32, 64, 128, 256];
    let steps = 100u32;

    println!("Running {} steps per grid size (16x16 tiles)...", steps);
    println!();
    println!("{:<12} {:>8} {:>12} {:>12} {:>15} {:>12}",
             "Grid Size", "Tiles", "Time (ms)", "Steps/sec", "K2K Msgs", "Msgs/step");
    println!("{}", "-".repeat(80));

    for &size in &sizes {
        let params = AcousticParams::new(343.0, 1.0);
        match TileKernelGrid::new(size, size, params, Backend::Cpu).await {
            Ok(mut grid) => {
                grid.inject_impulse(size / 2, size / 2, 1.0);

                // Warmup
                for _ in 0..5 {
                    let _ = grid.step().await;
                }

                // Reset K2K stats after warmup
                let stats_before = grid.k2k_stats();

                let start = Instant::now();
                for _ in 0..steps {
                    let _ = grid.step().await;
                }
                let elapsed = start.elapsed();

                let stats_after = grid.k2k_stats();
                let k2k_msgs = stats_after.messages_delivered - stats_before.messages_delivered;

                let total_ms = elapsed.as_secs_f64() * 1000.0;
                let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
                let msgs_per_step = k2k_msgs as f64 / steps as f64;

                println!(
                    "{:<12} {:>8} {:>12.2} {:>12.0} {:>15} {:>12.1}",
                    format!("{}x{}", size, size),
                    grid.tile_count(),
                    total_ms,
                    steps_per_sec,
                    k2k_msgs,
                    msgs_per_step
                );
            }
            Err(e) => {
                println!("{:<12} - Error: {}", format!("{}x{}", size, size), e);
            }
        }
    }
    println!();
}

#[cfg(feature = "wgpu")]
async fn benchmark_tile_grid_gpu() {
    // Conservative sizes for GPU to avoid memory issues
    let sizes = [32u32, 64, 128];
    let steps = 100u32;

    println!("Running {} steps per grid size (16x16 tiles, GPU FDTD)...", steps);
    println!();

    // First check if WGPU is available
    println!("Initializing WGPU...");

    println!();
    println!("{:<12} {:>8} {:>12} {:>12} {:>15} {:>12}",
             "Grid Size", "Tiles", "Time (ms)", "Steps/sec", "K2K Msgs", "Msgs/step");
    println!("{}", "-".repeat(80));

    for &size in &sizes {
        let params = AcousticParams::new(343.0, 1.0);
        match TileKernelGrid::new(size, size, params, Backend::Cpu).await {
            Ok(mut grid) => {
                // Enable GPU compute
                match grid.enable_gpu_compute().await {
                    Ok(()) => {
                        grid.inject_impulse(size / 2, size / 2, 1.0);

                        // Warmup
                        for _ in 0..5 {
                            let _ = grid.step_gpu().await;
                        }

                        // Reset K2K stats after warmup
                        let stats_before = grid.k2k_stats();

                        let start = Instant::now();
                        for _ in 0..steps {
                            let _ = grid.step_gpu().await;
                        }
                        let elapsed = start.elapsed();

                        let stats_after = grid.k2k_stats();
                        let k2k_msgs = stats_after.messages_delivered - stats_before.messages_delivered;

                        let total_ms = elapsed.as_secs_f64() * 1000.0;
                        let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
                        let msgs_per_step = k2k_msgs as f64 / steps as f64;

                        println!(
                            "{:<12} {:>8} {:>12.2} {:>12.0} {:>15} {:>12.1}",
                            format!("{}x{}", size, size),
                            grid.tile_count(),
                            total_ms,
                            steps_per_sec,
                            k2k_msgs,
                            msgs_per_step
                        );
                    }
                    Err(e) => {
                        println!("{:<12} - GPU init error: {}", format!("{}x{}", size, size), e);
                    }
                }
            }
            Err(e) => {
                println!("{:<12} - Grid error: {}", format!("{}x{}", size, size), e);
            }
        }
    }
    println!();

    // Also compare CPU vs GPU for same tile grid
    println!("CPU vs GPU Comparison (64x64 grid, 100 steps):");
    println!();

    let size = 64u32;
    let params = AcousticParams::new(343.0, 1.0);

    // CPU tile grid
    if let Ok(mut cpu_grid) = TileKernelGrid::new(size, size, params.clone(), Backend::Cpu).await {
        cpu_grid.inject_impulse(size / 2, size / 2, 1.0);

        let start = Instant::now();
        for _ in 0..steps {
            let _ = cpu_grid.step().await;
        }
        let cpu_time = start.elapsed();
        println!("  CPU TileGrid: {:.2} ms ({:.0} steps/sec)",
                 cpu_time.as_secs_f64() * 1000.0,
                 steps as f64 / cpu_time.as_secs_f64());
    }

    // GPU tile grid
    if let Ok(mut gpu_grid) = TileKernelGrid::new(size, size, params, Backend::Cpu).await {
        if gpu_grid.enable_gpu_compute().await.is_ok() {
            gpu_grid.inject_impulse(size / 2, size / 2, 1.0);

            let start = Instant::now();
            for _ in 0..steps {
                let _ = gpu_grid.step_gpu().await;
            }
            let gpu_time = start.elapsed();
            println!("  GPU TileGrid: {:.2} ms ({:.0} steps/sec)",
                     gpu_time.as_secs_f64() * 1000.0,
                     steps as f64 / gpu_time.as_secs_f64());
        }
    }
    println!();
}

#[cfg(feature = "cuda")]
async fn benchmark_cuda_persistent() {
    let sizes = [32u32, 64, 128, 256];
    let steps = 1000u32;

    println!("Running {} steps per grid size (16x16 tiles, CUDA GPU-Persistent)...", steps);
    println!("State stays GPU-resident, only halos transferred (64 bytes/edge/step)");
    println!();
    println!("{:<12} {:>8} {:>12} {:>12} {:>18}",
             "Grid Size", "Tiles", "Time (ms)", "Steps/sec", "Halo Transfer (KB)");
    println!("{}", "-".repeat(70));

    for &size in &sizes {
        let params = AcousticParams::new(343.0, 1.0);
        match TileKernelGrid::new(size, size, params, Backend::Cpu).await {
            Ok(mut grid) => {
                // Enable CUDA GPU-persistent compute
                match grid.enable_cuda_persistent() {
                    Ok(()) => {
                        // Inject impulse using CUDA method
                        let _ = grid.inject_impulse_cuda(size / 2, size / 2, 1.0);

                        // Warmup
                        for _ in 0..10 {
                            let _ = grid.step_cuda_persistent();
                        }

                        let start = Instant::now();
                        for _ in 0..steps {
                            let _ = grid.step_cuda_persistent();
                        }
                        let elapsed = start.elapsed();

                        // Calculate halo transfer: 4 edges × 16 floats × 4 bytes × 2 (send+recv) × tiles × steps
                        let tiles = grid.tile_count();
                        let halo_bytes_per_step = tiles * 4 * 16 * 4 * 2;
                        let total_halo_kb = (halo_bytes_per_step * steps as usize) as f64 / 1024.0;

                        let total_ms = elapsed.as_secs_f64() * 1000.0;
                        let steps_per_sec = steps as f64 / elapsed.as_secs_f64();

                        println!(
                            "{:<12} {:>8} {:>12.2} {:>12.0} {:>18.1}",
                            format!("{}x{}", size, size),
                            tiles,
                            total_ms,
                            steps_per_sec,
                            total_halo_kb
                        );
                    }
                    Err(e) => {
                        println!("{:<12} - CUDA init error: {}", format!("{}x{}", size, size), e);
                    }
                }
            }
            Err(e) => {
                println!("{:<12} - Grid error: {}", format!("{}x{}", size, size), e);
            }
        }
    }
    println!();
}

#[cfg(feature = "wgpu")]
async fn benchmark_wgpu_persistent() {
    let sizes = [32u32, 64, 128, 256];
    let steps = 1000u32;

    println!("Running {} steps per grid size (16x16 tiles, WGPU GPU-Persistent)...", steps);
    println!("State stays GPU-resident, only halos transferred (64 bytes/edge/step)");
    println!();
    println!("{:<12} {:>8} {:>12} {:>12} {:>18}",
             "Grid Size", "Tiles", "Time (ms)", "Steps/sec", "Halo Transfer (KB)");
    println!("{}", "-".repeat(70));

    for &size in &sizes {
        let params = AcousticParams::new(343.0, 1.0);
        match TileKernelGrid::new(size, size, params, Backend::Cpu).await {
            Ok(mut grid) => {
                // Enable WGPU GPU-persistent compute
                match grid.enable_wgpu_persistent().await {
                    Ok(()) => {
                        // Inject impulse using WGPU method
                        let _ = grid.inject_impulse_wgpu(size / 2, size / 2, 1.0);

                        // Warmup
                        for _ in 0..10 {
                            let _ = grid.step_wgpu_persistent();
                        }

                        let start = Instant::now();
                        for _ in 0..steps {
                            let _ = grid.step_wgpu_persistent();
                        }
                        let elapsed = start.elapsed();

                        // Calculate halo transfer
                        let tiles = grid.tile_count();
                        let halo_bytes_per_step = tiles * 4 * 16 * 4 * 2;
                        let total_halo_kb = (halo_bytes_per_step * steps as usize) as f64 / 1024.0;

                        let total_ms = elapsed.as_secs_f64() * 1000.0;
                        let steps_per_sec = steps as f64 / elapsed.as_secs_f64();

                        println!(
                            "{:<12} {:>8} {:>12.2} {:>12.0} {:>18.1}",
                            format!("{}x{}", size, size),
                            tiles,
                            total_ms,
                            steps_per_sec,
                            total_halo_kb
                        );
                    }
                    Err(e) => {
                        println!("{:<12} - WGPU init error: {}", format!("{}x{}", size, size), e);
                    }
                }
            }
            Err(e) => {
                println!("{:<12} - Grid error: {}", format!("{}x{}", size, size), e);
            }
        }
    }
    println!();
}

async fn benchmark_full_comparison() {
    let size = 64u32;
    let steps = 1000u32;

    println!("Grid: {}x{}, Steps: {}", size, size, steps);
    println!();
    println!("{:<35} {:>12} {:>15} {:>12}",
             "Backend", "Time (ms)", "Steps/sec", "Speedup");
    println!("{}", "-".repeat(78));

    let mut baseline_steps_per_sec: Option<f64> = None;

    // 1. CPU SimulationGrid (baseline)
    {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = SimulationGrid::new(size, size, params);
        grid.inject_impulse(size / 2, size / 2, 1.0);

        // Warmup
        for _ in 0..10 {
            grid.step();
        }

        let start = Instant::now();
        for _ in 0..steps {
            grid.step();
        }
        let elapsed = start.elapsed();

        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
        baseline_steps_per_sec = Some(steps_per_sec);

        println!(
            "{:<35} {:>12.2} {:>15.0} {:>12}",
            "CPU SimulationGrid (SoA+SIMD)",
            total_ms,
            steps_per_sec,
            "1.00x (baseline)"
        );
    }

    // 2. TileKernelGrid CPU
    if let Ok(mut grid) = TileKernelGrid::new(size, size, AcousticParams::new(343.0, 1.0), Backend::Cpu).await {
        grid.inject_impulse(size / 2, size / 2, 1.0);

        for _ in 0..10 {
            let _ = grid.step().await;
        }

        let start = Instant::now();
        for _ in 0..steps {
            let _ = grid.step().await;
        }
        let elapsed = start.elapsed();

        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
        let speedup = steps_per_sec / baseline_steps_per_sec.unwrap();

        println!(
            "{:<35} {:>12.2} {:>15.0} {:>12.2}x",
            "TileKernelGrid CPU (K2K)",
            total_ms,
            steps_per_sec,
            speedup
        );
    }

    // 3. TileKernelGrid WGPU Legacy
    #[cfg(feature = "wgpu")]
    if let Ok(mut grid) = TileKernelGrid::new(size, size, AcousticParams::new(343.0, 1.0), Backend::Cpu).await {
        if grid.enable_gpu_compute().await.is_ok() {
            grid.inject_impulse(size / 2, size / 2, 1.0);

            for _ in 0..10 {
                let _ = grid.step_gpu().await;
            }

            let start = Instant::now();
            for _ in 0..steps {
                let _ = grid.step_gpu().await;
            }
            let elapsed = start.elapsed();

            let total_ms = elapsed.as_secs_f64() * 1000.0;
            let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
            let speedup = steps_per_sec / baseline_steps_per_sec.unwrap();

            println!(
                "{:<35} {:>12.2} {:>15.0} {:>12.2}x",
                "TileKernelGrid WGPU Legacy",
                total_ms,
                steps_per_sec,
                speedup
            );
        }
    }

    // 4. TileKernelGrid CUDA GPU-Persistent
    #[cfg(feature = "cuda")]
    if let Ok(mut grid) = TileKernelGrid::new(size, size, AcousticParams::new(343.0, 1.0), Backend::Cpu).await {
        if grid.enable_cuda_persistent().is_ok() {
            let _ = grid.inject_impulse_cuda(size / 2, size / 2, 1.0);

            for _ in 0..10 {
                let _ = grid.step_cuda_persistent();
            }

            let start = Instant::now();
            for _ in 0..steps {
                let _ = grid.step_cuda_persistent();
            }
            let elapsed = start.elapsed();

            let total_ms = elapsed.as_secs_f64() * 1000.0;
            let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
            let speedup = steps_per_sec / baseline_steps_per_sec.unwrap();

            println!(
                "{:<35} {:>12.2} {:>15.0} {:>12.2}x",
                "TileKernelGrid CUDA GPU-Persistent",
                total_ms,
                steps_per_sec,
                speedup
            );
        }
    }

    // 5. TileKernelGrid WGPU GPU-Persistent
    #[cfg(feature = "wgpu")]
    if let Ok(mut grid) = TileKernelGrid::new(size, size, AcousticParams::new(343.0, 1.0), Backend::Cpu).await {
        if grid.enable_wgpu_persistent().await.is_ok() {
            let _ = grid.inject_impulse_wgpu(size / 2, size / 2, 1.0);

            for _ in 0..10 {
                let _ = grid.step_wgpu_persistent();
            }

            let start = Instant::now();
            for _ in 0..steps {
                let _ = grid.step_wgpu_persistent();
            }
            let elapsed = start.elapsed();

            let total_ms = elapsed.as_secs_f64() * 1000.0;
            let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
            let speedup = steps_per_sec / baseline_steps_per_sec.unwrap();

            println!(
                "{:<35} {:>12.2} {:>15.0} {:>12.2}x",
                "TileKernelGrid WGPU GPU-Persistent",
                total_ms,
                steps_per_sec,
                speedup
            );
        }
    }

    println!();
}
