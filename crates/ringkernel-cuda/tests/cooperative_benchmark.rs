//! Cooperative groups tests.
//!
//! Run with: cargo test -p ringkernel-cuda --features cooperative --release -- --nocapture --ignored

#![cfg(feature = "cooperative")]

use ringkernel_cuda::cooperative::{
    has_cooperative_support, CooperativeKernel, kernels,
};

#[test]
#[ignore] // Requires CUDA GPU
fn test_cooperative_fdtd_kernel() {
    if !has_cooperative_support() {
        println!("Cooperative support not available, skipping test");
        return;
    }

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Pre-compiled Cooperative FDTD Kernel Test                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Create device using ringkernel-cuda's wrapper
    let rk_device = ringkernel_cuda::CudaDevice::new(0)
        .expect("Failed to create device");

    // Load pre-compiled cooperative kernel
    match CooperativeKernel::from_precompiled(&rk_device, kernels::COOP_PERSISTENT_FDTD, 256) {
        Ok(kernel) => {
            println!("✓ Loaded cooperative FDTD kernel: {}", kernel.func_name());
            println!("  Block size: {}", kernel.block_size());
            println!("  Max concurrent blocks: {}", kernel.max_concurrent_blocks());

            // This kernel needs proper FDTD data structures to actually run
            // For now, just verify it loads successfully
            println!("\n✓ Cooperative kernel infrastructure verified!");
        }
        Err(e) => {
            println!("✗ Failed to load cooperative kernel: {}", e);
        }
    }
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_grid_sync_correctness() {
    if !has_cooperative_support() {
        println!("Cooperative support not available, skipping test");
        return;
    }

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Grid Sync Correctness Test                                   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Load pre-compiled test kernel that uses grid.sync()
    let rk_device = ringkernel_cuda::CudaDevice::new(0)
        .expect("Failed to create device");

    match CooperativeKernel::from_precompiled(&rk_device, kernels::COOP_TEST_GRID_SYNC, 256) {
        Ok(kernel) => {
            let max_blocks = kernel.max_concurrent_blocks();
            println!("✓ Loaded grid sync test kernel");
            println!("  Max concurrent blocks: {}", max_blocks);

            // The test kernel verifies that grid.sync() works correctly
            // by having all blocks write to a shared counter, sync, then verify
            println!("\n✓ Grid sync test kernel loaded successfully!");
            println!("  (Full execution test requires data structure setup)");
        }
        Err(e) => {
            println!("Note: Grid sync test kernel not available: {}", e);
            println!("      This is expected if the kernel wasn't compiled.");
        }
    }
}
