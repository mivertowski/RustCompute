//! Hello Kernel Example
//!
//! Demonstrates basic RingKernel usage: launching a kernel,
//! sending messages, and receiving responses.

use ringkernel::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for logs
    tracing_subscriber::fmt::init();

    println!("🚀 RingKernel Hello World Example\n");

    // Create runtime with auto-detected backend
    println!("Creating RingKernel runtime...");
    let runtime = RingKernel::builder()
        .backend(Backend::Auto)
        .debug(true)
        .build()
        .await?;

    println!("✓ Runtime created with backend: {:?}\n", runtime.backend());

    // Launch a simple kernel
    println!("Launching 'hello' kernel...");
    let kernel = runtime
        .launch(
            "hello",
            LaunchOptions::single_block(256).with_queue_capacity(64),
        )
        .await?;

    println!("✓ Kernel launched: {}\n", kernel.id());

    // Check status
    let status = kernel.status();
    println!("Kernel status:");
    println!("  State: {:?}", status.state);
    println!("  Mode: {:?}", status.mode);
    println!("  Uptime: {:?}", status.uptime);
    println!();

    // Get metrics
    let metrics = kernel.metrics();
    println!("Kernel metrics:");
    println!("  Messages processed: {}", metrics.telemetry.messages_processed);
    println!("  Avg latency: {:.2}µs", metrics.telemetry.avg_latency_us());
    println!();

    // Terminate the kernel
    println!("Terminating kernel...");
    kernel.terminate().await?;
    println!("✓ Kernel terminated\n");

    // Shutdown runtime
    println!("Shutting down runtime...");
    runtime.shutdown().await?;
    println!("✓ Runtime shutdown complete\n");

    println!("👋 Hello Kernel example complete!");

    Ok(())
}
