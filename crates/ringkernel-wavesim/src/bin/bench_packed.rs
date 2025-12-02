//! Benchmark for the packed CUDA backend with GPU-only halo exchange.
//!
//! Run with: cargo run -p ringkernel-wavesim --bin bench_packed --release --features cuda

use ringkernel_wavesim::simulation::{AcousticParams, CudaPackedBackend, SimulationGrid};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║   CUDA Packed Backend Benchmark (GPU-Only Halo Exchange)                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Test parameters
    let sizes = [32u32, 64, 128, 256, 512];
    let steps = 10000u32;
    let tile_size = 16u32;

    // Physics params
    let params = AcousticParams::new(343.0, 1.0);
    let c2 = params.courant_number().powi(2);
    let damping = 1.0 - params.damping;

    println!("Running {} steps per grid size (tile size: {}x{})", steps, tile_size, tile_size);
    println!("GPU-only halo exchange: ZERO host transfers during simulation");
    println!();

    // Section 1: CUDA Packed Backend
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("CUDA Packed Backend (GPU-Only Halo Exchange)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("{:<12} {:>8} {:>12} {:>12} {:>15} {:>12}",
             "Grid Size", "Tiles", "Halo Copies", "Time (ms)", "Steps/sec", "M cells/s");
    println!("{}", "-".repeat(80));

    for &size in &sizes {
        match CudaPackedBackend::new(size, size, tile_size) {
            Ok(mut backend) => {
                backend.set_params(c2, damping);

                // Inject impulse at center
                backend.inject_impulse(size / 2, size / 2, 1.0).unwrap();

                // Warmup (10 steps)
                for _ in 0..10 {
                    backend.step().unwrap();
                }
                backend.synchronize().unwrap();

                // Benchmark
                let start = Instant::now();
                backend.step_batch(steps).unwrap();
                let elapsed = start.elapsed();

                let total_ms = elapsed.as_secs_f64() * 1000.0;
                let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
                let cells = size as f64 * size as f64;
                let throughput = (cells * steps as f64) / elapsed.as_secs_f64() / 1_000_000.0;

                println!(
                    "{:<12} {:>8} {:>12} {:>12.2} {:>15.0} {:>12.1}",
                    format!("{}x{}", size, size),
                    backend.tile_count(),
                    backend.halo_copy_count(),
                    total_ms,
                    steps_per_sec,
                    throughput
                );
            }
            Err(e) => {
                println!("{:<12} - Error: {}", format!("{}x{}", size, size), e);
            }
        }
    }
    println!();

    // Section 2: CPU SimulationGrid (baseline comparison)
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("CPU SimulationGrid (Baseline - SoA + SIMD + Rayon)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("{:<12} {:>10} {:>12} {:>15} {:>12}",
             "Grid Size", "Cells", "Time (ms)", "Steps/sec", "M cells/s");
    println!("{}", "-".repeat(65));

    for &size in &sizes {
        let mut grid = SimulationGrid::new(size, size, params.clone());
        grid.inject_impulse(size / 2, size / 2, 1.0);

        // Warmup
        for _ in 0..10 {
            grid.step();
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..steps {
            grid.step();
        }
        let elapsed = start.elapsed();

        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
        let cells = size as f64 * size as f64;
        let throughput = (cells * steps as f64) / elapsed.as_secs_f64() / 1_000_000.0;

        println!(
            "{:<12} {:>10} {:>12.2} {:>15.0} {:>12.1}",
            format!("{}x{}", size, size),
            size * size,
            total_ms,
            steps_per_sec,
            throughput
        );
    }
    println!();

    // Section 3: Direct Comparison
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Direct Comparison: CPU vs CUDA Packed (256x256 grid, {} steps)", steps);
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let size = 256u32;

    // CPU
    let mut cpu_grid = SimulationGrid::new(size, size, params.clone());
    cpu_grid.inject_impulse(size / 2, size / 2, 1.0);
    for _ in 0..10 { cpu_grid.step(); }

    let cpu_start = Instant::now();
    for _ in 0..steps {
        cpu_grid.step();
    }
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_steps_per_sec = steps as f64 / cpu_elapsed.as_secs_f64();

    println!("CPU SimulationGrid:    {:>12.2} ms  ({:>10.0} steps/sec)",
             cpu_elapsed.as_secs_f64() * 1000.0, cpu_steps_per_sec);

    // CUDA Packed
    if let Ok(mut cuda_backend) = CudaPackedBackend::new(size, size, tile_size) {
        cuda_backend.set_params(c2, damping);
        cuda_backend.inject_impulse(size / 2, size / 2, 1.0).unwrap();
        for _ in 0..10 { cuda_backend.step().unwrap(); }
        cuda_backend.synchronize().unwrap();

        let cuda_start = Instant::now();
        cuda_backend.step_batch(steps).unwrap();
        let cuda_elapsed = cuda_start.elapsed();
        let cuda_steps_per_sec = steps as f64 / cuda_elapsed.as_secs_f64();

        println!("CUDA Packed Backend:   {:>12.2} ms  ({:>10.0} steps/sec)",
                 cuda_elapsed.as_secs_f64() * 1000.0, cuda_steps_per_sec);

        let speedup = cuda_steps_per_sec / cpu_steps_per_sec;
        if speedup >= 1.0 {
            println!("\nCUDA Packed is {:.2}x FASTER than CPU", speedup);
        } else {
            println!("\nCUDA Packed is {:.2}x slower than CPU", 1.0 / speedup);
        }
    }
    println!();

    // Section 4: Correctness Verification
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Correctness Verification (32x32 grid, 50 steps)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let size = 32u32;
    let verify_steps = 50u32;

    let mut cpu_grid = SimulationGrid::new(size, size, params.clone());
    cpu_grid.inject_impulse(size / 2, size / 2, 1.0);

    if let Ok(mut cuda_backend) = CudaPackedBackend::new(size, size, tile_size) {
        cuda_backend.set_params(c2, damping);
        cuda_backend.inject_impulse(size / 2, size / 2, 1.0).unwrap();

        // Run steps
        for _ in 0..verify_steps {
            cpu_grid.step();
            cuda_backend.step().unwrap();
        }
        cuda_backend.synchronize().unwrap();

        // Compare
        let cpu_pressure = cpu_grid.pressure_slice();
        let cuda_pressure = cuda_backend.read_pressure_grid().unwrap();

        let mut max_diff = 0.0f32;
        let mut total_diff = 0.0f32;

        for y in 0..size as usize {
            for x in 0..size as usize {
                let cpu_val = cpu_pressure[y * size as usize + x];
                let cuda_val = cuda_pressure[y * size as usize + x];
                let diff = (cpu_val - cuda_val).abs();
                max_diff = max_diff.max(diff);
                total_diff += diff;
            }
        }

        let avg_diff = total_diff / (size * size) as f32;
        let cpu_center = cpu_pressure[(size / 2) as usize * size as usize + (size / 2) as usize];
        let cuda_center = cuda_pressure[(size / 2) as usize * size as usize + (size / 2) as usize];

        println!("CPU center pressure:  {:.6}", cpu_center);
        println!("CUDA center pressure: {:.6}", cuda_center);
        println!("Max difference:       {:.6}", max_diff);
        println!("Avg difference:       {:.6}", avg_diff);

        if max_diff < 0.1 {
            println!("\n✓ CUDA Packed results match CPU within tolerance!");
        } else {
            println!("\n✗ Results differ - investigating...");
        }
    }
    println!();
}
