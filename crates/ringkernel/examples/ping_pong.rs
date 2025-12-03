//! # Ping-Pong K2K Messaging Example
//!
//! Demonstrates kernel-to-kernel (K2K) messaging using the actual K2K broker.
//! Two kernels bounce messages back and forth, showcasing direct communication
//! without going through the host application.
//!
//! ## Run this example:
//! ```bash
//! cargo run -p ringkernel --example ping_pong
//! ```
//!
//! ## What this demonstrates:
//!
//! - K2K broker creation and endpoint registration
//! - Direct kernel-to-kernel message sending
//! - Bidirectional communication patterns
//! - Priority-based message routing
//! - HLC timestamp integration for causal ordering

use ringkernel::prelude::*;
use ringkernel_cpu::CpuRuntime;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== RingKernel Ping-Pong K2K Example ===\n");

    // Create runtime with K2K enabled (default)
    let runtime = Arc::new(CpuRuntime::new().await?);
    println!("Runtime created with K2K messaging enabled\n");

    // Get the K2K broker
    let broker = runtime.k2k_broker().expect("K2K broker should be available");
    println!("K2K broker obtained");

    // Register ping and pong endpoints
    let ping_id = KernelId::new("ping");
    let pong_id = KernelId::new("pong");

    let ping_endpoint = broker.register(ping_id.clone());
    let mut pong_endpoint = broker.register(pong_id.clone());

    println!("Registered endpoints: 'ping' and 'pong'");

    // Check broker stats
    let stats = broker.stats();
    println!(
        "Broker stats: {} endpoints registered\n",
        stats.registered_endpoints
    );

    // ========== Ping-Pong Pattern ==========
    println!("--- Starting Ping-Pong Pattern ---\n");

    for round in 1..=5 {
        // PING sends to PONG
        let timestamp = HlcTimestamp::now(round);
        let envelope = MessageEnvelope::empty(
            round as u64,      // source_id (simulated)
            round as u64 + 1,  // dest_id (simulated)
            timestamp,
        );

        let receipt = ping_endpoint
            .send(pong_id.clone(), envelope)
            .await?;

        match receipt.status {
            DeliveryStatus::Delivered => {
                println!("Round {}: PING -> PONG [delivered]", round);
            }
            other => {
                println!("Round {}: PING -> PONG [{:?}]", round, other);
            }
        }

        // PONG receives the message
        if let Some(message) = pong_endpoint.try_receive() {
            println!(
                "         PONG received: hops={}, priority={}",
                message.hops, message.priority
            );

            // PONG responds to PING
            let response_timestamp = HlcTimestamp::now(round + 100);
            let response = MessageEnvelope::empty(
                message.destination.as_str().len() as u64,
                message.source.as_str().len() as u64,
                response_timestamp,
            );

            let _response_receipt = pong_endpoint
                .send(ping_id.clone(), response)
                .await?;

            println!("         PONG -> PING [responded]\n");
        }
    }

    println!("--- Ping-Pong Complete ---\n");

    // ========== Priority Messaging Demo ==========
    println!("--- Priority Messaging Demo ---\n");

    // Send messages with different priorities
    for priority in [0, 100, 200, 255] {
        let envelope = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));
        let receipt = ping_endpoint
            .send_priority(pong_id.clone(), envelope, priority)
            .await?;

        println!(
            "Sent priority {} message: {:?}",
            priority, receipt.status
        );
    }

    // Receive all messages
    println!("\nReceiving messages by priority:");
    while let Some(message) = pong_endpoint.try_receive() {
        println!("  Received message with priority: {}", message.priority);
    }

    // ========== Final Stats ==========
    let final_stats = broker.stats();
    println!("\n--- Final Broker Stats ---");
    println!("  Registered endpoints: {}", final_stats.registered_endpoints);
    println!("  Messages delivered: {}", final_stats.messages_delivered);
    println!("  Routes configured: {}", final_stats.routes_configured);

    // Cleanup
    broker.unregister(&ping_id);
    broker.unregister(&pong_id);

    println!("\n=== Ping-Pong Example Complete! ===");

    Ok(())
}
