//! # Hello Kernel Example
//!
//! This example demonstrates the fundamentals of RingKernel:
//! - Creating a runtime with backend selection
//! - Launching persistent GPU kernels
//! - Kernel lifecycle management (activate, suspend, terminate)
//! - Basic message passing
//!
//! ## Run this example:
//! ```bash
//! cargo run -p ringkernel --example basic_hello_kernel
//! ```
//!
//! ## What this demonstrates:
//!
//! Unlike traditional GPU programming where kernels execute once and terminate,
//! RingKernel kernels persist on the GPU. They maintain state between messages
//! and continue processing until explicitly terminated.
//!
//! ```text
//! Traditional GPU:                    RingKernel:
//!
//! Launch → Execute → Terminate        Launch → Activate → Process messages...
//! Launch → Execute → Terminate                            Process messages...
//! Launch → Execute → Terminate                            Process messages...
//!                                                         Terminate
//! ```

use ringkernel::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt::init();

    println!("=== RingKernel Hello World Example ===\n");

    // Step 1: Create a runtime
    // -------------------------
    // The runtime manages GPU resources and kernel lifecycle.
    // Backend::Auto will select the best available GPU backend,
    // falling back to CPU if no GPU is available.

    println!("1. Creating RingKernel runtime...");

    let runtime = RingKernel::builder()
        .backend(Backend::Auto) // Auto-select best backend
        .debug(true) // Enable debug output
        .build()
        .await?;

    println!("   Backend selected: {:?}", runtime.backend());
    println!(
        "   Available backends: {:?}\n",
        ringkernel::availability::available_backends()
    );

    // Step 2: Launch a kernel
    // -----------------------
    // Kernels are launched with a unique ID and launch options.
    // The kernel is created but not yet active.

    println!("2. Launching kernel 'hello_processor'...");

    // By default, kernels auto-activate. Use without_auto_activate() to launch in Launched state.
    let options = LaunchOptions::default()
        .with_queue_capacity(1024) // Configure input queue size
        .without_auto_activate(); // Start in Launched state for manual activation

    let kernel = runtime.launch("hello_processor", options).await?;

    println!("   Kernel ID: {}", kernel.id());
    println!("   Initial state: {:?}", kernel.state());

    // Step 3: Activate the kernel
    // ---------------------------
    // Activating transitions the kernel from Launched to Active state.
    // The kernel begins its main processing loop on the GPU.

    println!("\n3. Activating kernel...");
    kernel.activate().await?;
    println!("   State after activation: {:?}", kernel.state());

    // Step 4: Check kernel status
    // ---------------------------
    // We can query the kernel's status including telemetry data.

    println!("\n4. Checking kernel status...");
    let status = kernel.status();
    println!("   State: {:?}", status.state);
    println!("   Messages in queue: {}", status.input_queue_depth);
    println!("   Messages processed: {}", status.messages_processed);

    // Step 5: Suspend and resume (demonstration)
    // ------------------------------------------
    // Kernels can be suspended to pause processing while preserving state.

    println!("\n5. Demonstrating suspend/resume...");
    kernel.suspend().await?;
    println!("   State after suspend: {:?}", kernel.state());

    // Kernel state is preserved, can be resumed later
    kernel.resume().await?;
    println!("   State after resume: {:?}", kernel.state());

    // Step 6: Get runtime metrics
    // ---------------------------
    // The runtime tracks overall performance metrics.

    println!("\n6. Runtime metrics:");
    let metrics = runtime.metrics();
    println!("   Active kernels: {}", metrics.active_kernels);
    println!("   Total launched: {}", metrics.total_launched);

    // Step 7: List all kernels
    // ------------------------
    println!("\n7. Listing all kernels:");
    for kernel_id in runtime.list_kernels() {
        println!("   - {}", kernel_id);
    }

    // Step 8: Clean shutdown
    // ----------------------
    // Always terminate kernels before shutting down the runtime.

    println!("\n8. Shutting down...");
    kernel.terminate().await?;
    println!("   Kernel terminated");

    runtime.shutdown().await?;
    println!("   Runtime shut down\n");

    println!("=== Example completed successfully! ===");

    Ok(())
}

// Additional examples of launch options configurations:

/// Example: High-throughput configuration for data processing
#[allow(dead_code)]
fn high_throughput_options() -> LaunchOptions {
    LaunchOptions::default()
        .with_queue_capacity(8192) // Large queue for batching
        .with_shared_memory(65536) // 64KB shared memory
        .with_priority(priority::HIGH) // Higher priority
}

/// Example: Low-latency configuration for real-time processing
#[allow(dead_code)]
fn low_latency_options() -> LaunchOptions {
    LaunchOptions::default()
        .with_queue_capacity(256) // Small queue for fast turnaround
        .with_shared_memory(16384) // 16KB shared memory
        .with_priority(priority::CRITICAL) // Highest priority
}

/// Example: Multi-block configuration for parallel processing
#[allow(dead_code)]
fn parallel_options() -> LaunchOptions {
    LaunchOptions::default()
        .with_grid_size(4) // 4 blocks
        .with_block_size(256) // 256 threads per block
        .with_queue_capacity(4096)
}
