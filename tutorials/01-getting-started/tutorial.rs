//! # Tutorial 01: Getting Started with RingKernel
//!
//! Welcome to RingKernel! This tutorial will walk you through the basic concepts
//! of GPU-native persistent actors.
//!
//! ## What You'll Learn
//!
//! 1. Creating a simple runtime
//! 2. Launching your first kernel
//! 3. Sending and receiving messages
//! 4. Understanding kernel lifecycles
//!
//! ## Running This Tutorial
//!
//! ```bash
//! cargo run -p ringkernel-tutorials --bin tutorial-01
//! ```
//!
//! ## Prerequisites
//!
//! - Rust 1.75+
//! - Basic understanding of async Rust
//! - No GPU required (we'll use the CPU backend)

// Tutorial code uses doc comments for educational explanations between sections
#![allow(
    clippy::empty_line_after_doc_comments,
    clippy::empty_line_after_outer_attr
)]

use ringkernel_core::prelude::*;
use ringkernel_cpu::CpuRuntime;

// ============================================================================
// STEP 1: Understanding the Runtime
// ============================================================================

/// The runtime is your gateway to GPU kernels. It manages:
/// - Kernel lifecycle (launch, activate, terminate)
/// - Message queues for communication
/// - Backend selection (CPU, CUDA, WebGPU)
///
/// For this tutorial, we use `CpuRuntime` which simulates GPU execution
/// on the CPU - perfect for learning without needing actual GPU hardware.

// ============================================================================
// STEP 2: Launching a Kernel
// ============================================================================

/// Kernels are persistent actors that run on the GPU.
/// Unlike traditional GPU kernels that run once and exit,
/// RingKernel actors stay alive and process messages continuously.
///
/// Key concepts:
/// - **Launch**: Creates the kernel with configuration
/// - **Activate**: Starts the kernel's message processing loop
/// - **Terminate**: Gracefully shuts down the kernel

// ============================================================================
// STEP 3: Message Communication
// ============================================================================

/// Messages are how you communicate with GPU kernels.
/// They're automatically serialized for GPU-safe transfer.
///
/// The communication pattern is:
/// 1. Host sends message to kernel (H2K - Host to Kernel)
/// 2. Kernel processes and responds (K2H - Kernel to Host)
///
/// This creates a request-response pattern similar to actors.

// ============================================================================
// MAIN TUTORIAL CODE
// ============================================================================

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize logging for better debugging
    tracing_subscriber::fmt().with_env_filter("info").init();

    println!("===========================================");
    println!("   Tutorial 01: Getting Started");
    println!("===========================================\n");

    // STEP 1: Create a runtime
    println!("Step 1: Creating CPU Runtime...");
    let runtime = CpuRuntime::new().await?;
    println!("   Runtime created with backend: CPU");
    println!(
        "   K2K messaging: {}",
        if runtime.is_k2k_enabled() {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!();

    // STEP 2: Launch a kernel
    println!("Step 2: Launching a kernel...");

    // Configure the kernel launch options
    let options = LaunchOptions::default().with_queue_capacity(128); // Messages the queue can hold

    // Launch the kernel
    let kernel = runtime.launch("tutorial_kernel", options).await?;
    println!("   Kernel launched: {}", kernel.id());
    println!("   State: {:?}", kernel.state());
    println!();

    // STEP 3: Work with kernel state
    println!("Step 3: Understanding kernel states...");
    println!("   Kernels go through these states:");
    println!("   - Created    -> Just launched, not yet processing");
    println!("   - Active     -> Processing messages");
    println!("   - Terminated -> Shut down");
    println!();

    // The kernel should be active (auto-activated by default)
    println!("   Current kernel state: {:?}", kernel.state());
    println!();

    // STEP 4: Simulate message processing
    println!("Step 4: Message queue basics...");
    println!("   Queue capacity: {}", 128);
    println!("   Queues use lock-free ring buffers for:");
    println!("   - Zero-copy message transfer");
    println!("   - High-throughput communication");
    println!("   - Low latency (< 1 microsecond)");
    println!();

    // STEP 5: Graceful shutdown
    println!("Step 5: Graceful shutdown...");
    kernel.terminate().await?;
    println!("   Kernel terminated gracefully");
    println!("   Final state: {:?}", kernel.state());
    println!();

    // Summary
    println!("===========================================");
    println!("   Tutorial Complete!");
    println!("===========================================");
    println!();
    println!("What you learned:");
    println!("  - Create a runtime for kernel management");
    println!("  - Launch kernels with custom options");
    println!("  - Understand kernel state transitions");
    println!("  - Gracefully terminate kernels");
    println!();
    println!("Next: Tutorial 02 - Message Passing");
    println!("      Learn to send/receive messages to kernels");

    Ok(())
}

// ============================================================================
// EXERCISES
// ============================================================================

// Exercise 1: Try changing the queue capacity and observe the behavior
//
// Exercise 2: Launch multiple kernels and list them all
//
// Exercise 3: Try activating and terminating the same kernel multiple times
//             (Hint: you'll get an error!)

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tutorial_completes() {
        let runtime = CpuRuntime::new().await.unwrap();
        let kernel = runtime
            .launch("test", LaunchOptions::default())
            .await
            .unwrap();
        kernel.terminate().await.unwrap();
    }
}
