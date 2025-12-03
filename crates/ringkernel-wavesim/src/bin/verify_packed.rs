//! Verify CUDA Packed backend correctness against TileKernelGrid CPU.
//!
//! Run with: cargo run -p ringkernel-wavesim --bin verify_packed --release --features cuda

use ringkernel::prelude::Backend;
use ringkernel_wavesim::simulation::{AcousticParams, CudaPackedBackend, TileKernelGrid};

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Verifying CUDA Packed Backend vs TileKernelGrid CPU");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let size = 64u32;
    let tile_size = 16u32;
    let steps = 100u32;

    let params = AcousticParams::new(343.0, 1.0);
    let c2 = params.courant_number().powi(2);
    let damping = 1.0 - params.damping;

    println!(
        "Grid: {}x{}, Tile size: {}x{}, Steps: {}",
        size, size, tile_size, tile_size, steps
    );
    println!("c2 = {:.6}, damping = {:.6}", c2, damping);
    println!();

    // Create TileKernelGrid (CPU)
    let mut cpu_grid = TileKernelGrid::new(size, size, params.clone(), Backend::Cpu)
        .await
        .expect("Failed to create CPU grid");

    // Create CUDA Packed backend
    let mut cuda_backend =
        CudaPackedBackend::new(size, size, tile_size).expect("Failed to create CUDA backend");
    cuda_backend.set_params(c2, damping);

    // Inject same impulse
    cpu_grid.inject_impulse(size / 2, size / 2, 1.0);
    cuda_backend
        .inject_impulse(size / 2, size / 2, 1.0)
        .unwrap();

    println!("Running {} steps and comparing...", steps);
    println!();

    // Run steps and compare periodically
    for step in 1..=steps {
        cpu_grid.step().await.unwrap();
        cuda_backend.step().unwrap();

        if step % 20 == 0 || step == 1 || step == steps {
            cuda_backend.synchronize().unwrap();

            let cpu_pressure = cpu_grid.get_pressure_grid();
            let cuda_pressure = cuda_backend.read_pressure_grid().unwrap();

            let mut max_diff = 0.0f32;
            let mut total_diff = 0.0f32;
            let mut max_diff_pos = (0, 0);

            for y in 0..size as usize {
                for x in 0..size as usize {
                    let cpu_val = cpu_pressure[y][x];
                    let cuda_val = cuda_pressure[y * size as usize + x];
                    let diff = (cpu_val - cuda_val).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        max_diff_pos = (x, y);
                    }
                    total_diff += diff;
                }
            }

            let _avg_diff = total_diff / (size * size) as f32;
            let cpu_center = cpu_pressure[(size / 2) as usize][(size / 2) as usize];
            let cuda_center =
                cuda_pressure[(size / 2) as usize * size as usize + (size / 2) as usize];

            println!(
                "Step {:>3}: CPU center={:>10.6}, CUDA center={:>10.6}, max_diff={:.6} at ({},{})",
                step, cpu_center, cuda_center, max_diff, max_diff_pos.0, max_diff_pos.1
            );
        }
    }

    println!();

    // Final detailed comparison
    cuda_backend.synchronize().unwrap();
    let cpu_pressure = cpu_grid.get_pressure_grid();
    let cuda_pressure = cuda_backend.read_pressure_grid().unwrap();

    let mut max_diff = 0.0f32;
    let mut total_diff = 0.0f32;

    for y in 0..size as usize {
        for x in 0..size as usize {
            let cpu_val = cpu_pressure[y][x];
            let cuda_val = cuda_pressure[y * size as usize + x];
            let diff = (cpu_val - cuda_val).abs();
            max_diff = max_diff.max(diff);
            total_diff += diff;
        }
    }

    let avg_diff = total_diff / (size * size) as f32;

    // Energy comparison
    let cpu_energy: f32 = cpu_pressure
        .iter()
        .flat_map(|row| row.iter())
        .map(|&p| p * p)
        .sum();
    let cuda_energy: f32 = cuda_pressure.iter().map(|&p| p * p).sum();

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Final Results");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("Max difference:    {:.6}", max_diff);
    println!("Avg difference:    {:.6}", avg_diff);
    println!("CPU total energy:  {:.6}", cpu_energy);
    println!("CUDA total energy: {:.6}", cuda_energy);
    println!(
        "Energy diff:       {:.6} ({:.2}%)",
        (cpu_energy - cuda_energy).abs(),
        ((cpu_energy - cuda_energy).abs() / cpu_energy.max(0.0001) * 100.0)
    );
    println!();

    if max_diff < 0.01 {
        println!("✓ CUDA Packed results match TileKernelGrid CPU within tight tolerance!");
    } else if max_diff < 0.1 {
        println!("~ CUDA Packed results are similar to TileKernelGrid CPU (acceptable)");
    } else {
        println!("✗ Significant differences detected - needs investigation");

        // Print sample area
        println!();
        println!("CPU pressure around center:");
        let c = (size / 2) as usize;
        for dy in -2i32..=2 {
            for dx in -2i32..=2 {
                let y = (c as i32 + dy) as usize;
                let x = (c as i32 + dx) as usize;
                print!("{:>9.5} ", cpu_pressure[y][x]);
            }
            println!();
        }

        println!();
        println!("CUDA pressure around center:");
        for dy in -2i32..=2 {
            for dx in -2i32..=2 {
                let y = (c as i32 + dy) as usize;
                let x = (c as i32 + dx) as usize;
                print!("{:>9.5} ", cuda_pressure[y * size as usize + x]);
            }
            println!();
        }
    }
    println!();
}
