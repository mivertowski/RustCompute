//! # Kernel States and Lifecycle Example
//!
//! This example demonstrates the full kernel lifecycle and state machine:
//!
//! ```text
//!                 ┌──────────┐
//!                 │  Idle    │ (Initial state after launch)
//!                 └────┬─────┘
//!                      │ activate()
//!                      ▼
//!    suspend() ┌──────────┐ error
//!         ┌────│ Running  │────────┐
//!         │    └────┬─────┘        │
//!         │         │              ▼
//!         ▼         │         ┌────────┐
//!    ┌──────────┐   │         │ Error  │
//!    │Suspended │───┘         └────────┘
//!    └────┬─────┘ activate()
//!         │
//!         │ terminate()
//!         ▼
//!    ┌──────────┐
//!    │Terminated│ (Final state)
//!    └──────────┘
//! ```
//!
//! ## Run this example:
//! ```bash
//! cargo run --example kernel_states
//! ```

use ringkernel::prelude::*;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Kernel Lifecycle Example ===\n");

    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await?;

    // Create multiple kernels to demonstrate different states
    println!("Creating kernels...\n");

    let kernel_a = runtime.launch("kernel_a", LaunchOptions::default()).await?;
    let kernel_b = runtime.launch("kernel_b", LaunchOptions::default()).await?;
    let kernel_c = runtime.launch("kernel_c", LaunchOptions::default()).await?;

    // ====== State: Idle ======
    println!("=== State: IDLE ===");
    println!("Kernels start in Idle state after launch.");
    print_states(&[&kernel_a, &kernel_b, &kernel_c]);

    // ====== State: Running ======
    println!("\n=== State: RUNNING ===");
    println!("After activate(), kernels begin processing.");

    kernel_a.activate().await?;
    kernel_b.activate().await?;
    // kernel_c stays Idle

    print_states(&[&kernel_a, &kernel_b, &kernel_c]);

    // ====== State: Suspended ======
    println!("\n=== State: SUSPENDED ===");
    println!("Suspended kernels pause but preserve state.");

    kernel_b.suspend().await?;

    print_states(&[&kernel_a, &kernel_b, &kernel_c]);

    // Resume suspended kernel
    println!("\nResuming kernel_b...");
    kernel_b.activate().await?;
    print_states(&[&kernel_a, &kernel_b, &kernel_c]);

    // ====== Checking Status Details ======
    println!("\n=== Kernel Status Details ===");
    for (name, kernel) in [("A", &kernel_a), ("B", &kernel_b), ("C", &kernel_c)] {
        let status = kernel.status();
        println!("\nKernel {}:", name);
        println!("  State: {:?}", status.state);
        println!("  Input queue depth: {}", status.input_queue_depth);
        println!("  Output queue depth: {}", status.output_queue_depth);
        println!("  Messages processed: {}", status.messages_processed);
        println!("  Uptime: {:?}", status.uptime);
    }

    // ====== State: Terminated ======
    println!("\n=== State: TERMINATED ===");
    println!("Terminated kernels release all resources.");

    kernel_a.terminate().await?;
    kernel_b.terminate().await?;
    kernel_c.terminate().await?;

    // After termination, kernels are removed from the runtime
    println!("\nKernels remaining: {:?}", runtime.list_kernels());

    // ====== Demonstrating Multiple Kernels ======
    println!("\n=== Multi-Kernel Orchestration ===");
    demonstrate_orchestration(&runtime).await?;

    runtime.shutdown().await?;
    println!("\n=== Example completed! ===");

    Ok(())
}

fn print_states(kernels: &[&KernelHandle]) {
    for kernel in kernels {
        println!("  {} -> {:?}", kernel.id(), kernel.state());
    }
}

/// Demonstrates orchestrating multiple kernels
async fn demonstrate_orchestration(runtime: &RingKernel) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nLaunching pipeline: ingress -> processor -> egress");

    // Create a processing pipeline
    let ingress = runtime.launch("ingress", LaunchOptions::default()).await?;
    let processor = runtime.launch("processor", LaunchOptions::default()).await?;
    let egress = runtime.launch("egress", LaunchOptions::default()).await?;

    // Activate in order (downstream first for back-pressure handling)
    println!("Activating pipeline (downstream first)...");
    egress.activate().await?;
    processor.activate().await?;
    ingress.activate().await?;

    println!("Pipeline active. Processing would happen here...");

    // Graceful shutdown (upstream first)
    println!("Shutting down pipeline (upstream first)...");
    ingress.suspend().await?;
    processor.suspend().await?;
    egress.suspend().await?;

    // Final termination
    ingress.terminate().await?;
    processor.terminate().await?;
    egress.terminate().await?;

    println!("Pipeline shut down cleanly.");

    Ok(())
}
