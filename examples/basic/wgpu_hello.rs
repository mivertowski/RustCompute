//! WebGPU Hello World Example
//!
//! This example demonstrates basic WebGPU backend usage with RingKernel.
//! It creates a kernel, activates it, and shows the kernel lifecycle.
//!
//! # Requirements
//!
//! - WebGPU-compatible GPU (Vulkan, Metal, DX12, or browser)
//! - The `wgpu` feature enabled
//!
//! # Run
//!
//! ```bash
//! cargo run -p ringkernel --example wgpu_hello --features wgpu
//! ```

use std::time::Duration;

use ringkernel::prelude::*;
use ringkernel_wgpu::WgpuRuntime;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    println!("=== WebGPU RingKernel Example ===\n");

    // Check if WebGPU is available
    if !ringkernel_wgpu::is_wgpu_available() {
        eprintln!("WebGPU is not available on this system.");
        eprintln!("Make sure you have a compatible GPU and drivers.");
        return Ok(());
    }

    println!("WebGPU is available!\n");

    // Create the WebGPU runtime
    println!("Creating WebGPU runtime...");
    let runtime = WgpuRuntime::new().await?;
    println!("Backend: {:?}", runtime.backend());
    println!("K2K enabled: {}", runtime.is_k2k_enabled());

    // Launch a kernel with custom options
    println!("\nLaunching kernel 'compute_worker'...");
    let options = LaunchOptions::default()
        .with_block_size(256)
        .with_input_queue_capacity(64)
        .with_output_queue_capacity(64)
        .with_mode(KernelMode::EventDriven);

    let kernel = runtime.launch("compute_worker", options).await?;
    println!("Kernel launched: {}", kernel.id());
    println!("Initial state: {:?}", kernel.status().state);

    // Show kernel status
    let status = kernel.status();
    println!("\nKernel Status:");
    println!("  ID: {}", status.id);
    println!("  State: {:?}", status.state);
    println!("  Mode: {:?}", status.mode);
    println!("  Input queue depth: {}", status.input_queue_depth);
    println!("  Output queue depth: {}", status.output_queue_depth);
    println!("  Messages processed: {}", status.messages_processed);
    println!("  Uptime: {:?}", status.uptime);

    // Activate the kernel
    println!("\nActivating kernel...");
    kernel.activate().await?;
    println!("State after activate: {:?}", kernel.status().state);

    // Let it run briefly
    println!("\nKernel is now active and ready to process messages.");
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Show metrics
    let metrics = kernel.metrics();
    println!("\nKernel Metrics:");
    println!("  Messages processed: {}", metrics.telemetry.messages_processed);
    println!("  Messages dropped: {}", metrics.telemetry.messages_dropped);
    println!("  Uptime: {:?}", metrics.uptime);

    // Deactivate the kernel
    println!("\nDeactivating kernel...");
    kernel.deactivate().await?;
    println!("State after deactivate: {:?}", kernel.status().state);

    // Reactivate and terminate
    println!("\nReactivating kernel...");
    kernel.activate().await?;
    println!("State: {:?}", kernel.status().state);

    // Terminate
    println!("\nTerminating kernel...");
    kernel.terminate().await?;
    println!("State after terminate: {:?}", kernel.status().state);

    // Show runtime metrics
    let runtime_metrics = runtime.metrics();
    println!("\nRuntime Metrics:");
    println!("  Active kernels: {}", runtime_metrics.active_kernels);
    println!("  Total launched: {}", runtime_metrics.total_launched);

    // Shutdown the runtime
    println!("\nShutting down runtime...");
    runtime.shutdown().await?;
    println!("Runtime shutdown complete.");

    println!("\n=== WebGPU Example Complete ===");

    Ok(())
}
