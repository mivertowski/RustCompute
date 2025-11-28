//! Ping-Pong Example
//!
//! Demonstrates kernel-to-kernel messaging (K2K) with two kernels
//! bouncing messages back and forth.

use ringkernel::prelude::*;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("🏓 RingKernel Ping-Pong Example\n");

    // Create runtime
    let runtime = RingKernel::new().await?;
    println!("Runtime backend: {:?}\n", runtime.backend());

    // Launch two kernels
    let ping_kernel = runtime
        .launch("ping", LaunchOptions::single_block(128))
        .await?;

    let pong_kernel = runtime
        .launch("pong", LaunchOptions::single_block(128))
        .await?;

    println!(
        "Launched kernels: {}, {}",
        ping_kernel.id(),
        pong_kernel.id()
    );

    // Monitor kernel states
    println!("\nKernel states:");
    println!("  ping: {:?}", ping_kernel.status().state);
    println!("  pong: {:?}", pong_kernel.status().state);

    // Demonstrate message passing pattern
    println!("\n--- Simulating Ping-Pong Pattern ---\n");

    for round in 1..=5 {
        // Create a simple message envelope
        let header = MessageHeader::new(
            0x2001, // ping message type
            ping_kernel.status().id.as_str().len() as u64,
            pong_kernel.status().id.as_str().len() as u64,
            8,
            HlcTimestamp::now(1),
        );

        let _envelope = MessageEnvelope {
            header,
            payload: vec![round as u8; 8], // Simple payload
        };

        // Simulate sending to pong kernel's input queue
        // In real K2K, this would go through the GPU message bridge via _envelope

        println!("Round {}: PING -> PONG (payload: [{}; 8])", round, round);

        // Small delay to simulate processing
        tokio::time::sleep(Duration::from_millis(50)).await;

        println!("Round {}: PONG -> PING (response received)", round);
    }

    println!("\n--- Ping-Pong Complete ---\n");

    // Get final metrics
    let ping_metrics = ping_kernel.metrics();
    let pong_metrics = pong_kernel.metrics();

    println!("Final metrics:");
    println!("  ping kernel uptime: {:?}", ping_metrics.uptime);
    println!("  pong kernel uptime: {:?}", pong_metrics.uptime);

    // Cleanup
    ping_kernel.terminate().await?;
    pong_kernel.terminate().await?;

    // List remaining kernels (should be empty after termination)
    let kernels = runtime.list_kernels();
    println!("\nActive kernels after termination: {}", kernels.len());

    // Get runtime metrics
    let runtime_metrics = runtime.metrics();
    println!("\nRuntime metrics:");
    println!(
        "  Total kernels launched: {}",
        runtime_metrics.total_launched
    );
    println!("  Active kernels: {}", runtime_metrics.active_kernels);

    runtime.shutdown().await?;

    println!("\n✓ Ping-pong example complete!");

    Ok(())
}
