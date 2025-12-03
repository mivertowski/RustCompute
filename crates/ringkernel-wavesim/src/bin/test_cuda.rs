//! Quick correctness test for CUDA GPU-Persistent backend.
//!
//! Run with: cargo run -p ringkernel-wavesim --bin test_cuda --release --features cuda

use ringkernel::prelude::Backend;
use ringkernel_wavesim::simulation::{AcousticParams, TileKernelGrid};

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Testing CUDA GPU-Persistent Backend Correctness");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let size = 32u32;
    let params = AcousticParams::new(343.0, 1.0);

    println!("Grid: {}x{}, Tile size: 16x16", size, size);
    println!();

    // Create two grids - CPU and CUDA
    let mut cpu_grid = TileKernelGrid::new(size, size, params.clone(), Backend::Cpu)
        .await
        .expect("CPU grid creation failed");
    let mut cuda_grid = TileKernelGrid::new(size, size, params, Backend::Cpu)
        .await
        .expect("CUDA grid creation failed");

    // Enable CUDA for one grid
    println!("Initializing CUDA backend...");
    cuda_grid
        .enable_cuda_persistent()
        .expect("CUDA init failed");
    println!("CUDA backend initialized successfully!");
    println!();

    // Inject same impulse
    cpu_grid.inject_impulse(size / 2, size / 2, 1.0);
    cuda_grid
        .inject_impulse_cuda(size / 2, size / 2, 1.0)
        .expect("CUDA impulse failed");

    println!("Impulse injected at ({}, {})", size / 2, size / 2);
    println!();

    // Run same number of steps
    let steps = 20;
    println!("Running {} simulation steps...", steps);

    for i in 0..steps {
        cpu_grid.step().await.unwrap();
        cuda_grid.step_cuda_persistent().expect("CUDA step failed");

        if (i + 1) % 5 == 0 {
            // Compare after every 5 steps
            let cpu_pressure = cpu_grid.get_pressure_grid();
            let cuda_pressure = cuda_grid
                .read_pressure_from_cuda()
                .expect("CUDA read failed");

            let cpu_center = cpu_pressure[size as usize / 2][size as usize / 2];
            let cuda_center = cuda_pressure[size as usize / 2][size as usize / 2];
            let diff = (cpu_center - cuda_center).abs();

            println!(
                "  Step {}: CPU={:.6}, CUDA={:.6}, diff={:.6}",
                i + 1,
                cpu_center,
                cuda_center,
                diff
            );
        }
    }
    println!();

    // Final comparison
    let cpu_pressure = cpu_grid.get_pressure_grid();
    let cuda_pressure = cuda_grid
        .read_pressure_from_cuda()
        .expect("CUDA read failed");

    // Compare full grid
    let mut max_diff = 0.0f32;
    let mut total_diff = 0.0f32;
    let mut count = 0;
    let mut max_diff_pos = (0, 0);

    for y in 0..size as usize {
        for x in 0..size as usize {
            let cpu_val = cpu_pressure[y][x];
            let cuda_val = cuda_pressure[y][x];
            let diff = (cpu_val - cuda_val).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_pos = (x, y);
            }
            total_diff += diff;
            count += 1;
        }
    }

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("Final Comparison Results");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!(
        "CPU center pressure:  {:.6}",
        cpu_pressure[size as usize / 2][size as usize / 2]
    );
    println!(
        "CUDA center pressure: {:.6}",
        cuda_pressure[size as usize / 2][size as usize / 2]
    );
    println!();
    println!(
        "Max difference:       {:.6} at ({}, {})",
        max_diff, max_diff_pos.0, max_diff_pos.1
    );
    println!("Avg difference:       {:.6}", total_diff / count as f32);
    println!();

    // Energy comparison
    let cpu_energy: f32 = cpu_pressure
        .iter()
        .flat_map(|row| row.iter())
        .map(|&p| p * p)
        .sum();
    let cuda_energy: f32 = cuda_pressure
        .iter()
        .flat_map(|row| row.iter())
        .map(|&p| p * p)
        .sum();
    println!("CPU total energy:     {:.6}", cpu_energy);
    println!("CUDA total energy:    {:.6}", cuda_energy);
    println!(
        "Energy difference:    {:.6} ({:.2}%)",
        (cpu_energy - cuda_energy).abs(),
        ((cpu_energy - cuda_energy).abs() / cpu_energy * 100.0)
    );
    println!();

    if max_diff < 0.01 {
        println!("✓ CUDA results match CPU within tolerance!");
    } else {
        println!("✗ CUDA results differ significantly from CPU");

        // Print some sample values for debugging
        println!();
        println!("Sample values around center:");
        let c = size as usize / 2;
        for dy in -2i32..=2 {
            for dx in -2i32..=2 {
                let y = (c as i32 + dy) as usize;
                let x = (c as i32 + dx) as usize;
                if y < size as usize && x < size as usize {
                    print!("{:8.4} ", cpu_pressure[y][x]);
                }
            }
            println!();
        }
        println!();
        println!("CUDA values around center:");
        for dy in -2i32..=2 {
            for dx in -2i32..=2 {
                let y = (c as i32 + dy) as usize;
                let x = (c as i32 + dx) as usize;
                if y < size as usize && x < size as usize {
                    print!("{:8.4} ", cuda_pressure[y][x]);
                }
            }
            println!();
        }
    }
    println!();
}
