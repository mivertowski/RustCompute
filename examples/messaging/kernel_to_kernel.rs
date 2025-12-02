//! # Kernel-to-Kernel (K2K) Messaging Example
//!
//! This example demonstrates direct messaging between kernels using the K2K
//! infrastructure. K2K messaging allows kernels to communicate without going
//! through the host application.
//!
//! ## Run this example:
//! ```bash
//! cargo run -p ringkernel --example kernel_to_kernel
//! ```

use ringkernel::prelude::*;
use ringkernel_cpu::CpuRuntime;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Kernel-to-Kernel (K2K) Messaging Example ===\n");

    // Create runtime with K2K enabled (default)
    let runtime = Arc::new(CpuRuntime::new().await?);
    println!("Created CPU runtime with K2K messaging enabled");

    // Get the K2K broker
    let broker = runtime.k2k_broker().expect("K2K broker should be available");
    println!("K2K broker created with default configuration\n");

    // ========== Direct Broker Usage ==========
    println!("=== Direct K2K Broker Usage ===\n");

    // Register two kernel endpoints directly with the broker
    let producer_id = KernelId::new("producer");
    let consumer_id = KernelId::new("consumer");

    let producer_endpoint = broker.register(producer_id.clone());
    let mut consumer_endpoint = broker.register(consumer_id.clone());

    println!("Registered 'producer' and 'consumer' endpoints");

    // Check broker stats
    let stats = broker.stats();
    println!(
        "Broker stats: {} registered endpoints, {} messages delivered\n",
        stats.registered_endpoints, stats.messages_delivered
    );

    // Create and send a message from producer to consumer
    let timestamp = HlcTimestamp::now(1);
    let envelope = MessageEnvelope::empty(1, 2, timestamp);

    println!("Producer sending message to consumer...");
    let receipt = producer_endpoint
        .send(consumer_id.clone(), envelope)
        .await?;

    match receipt.status {
        DeliveryStatus::Delivered => {
            println!("  Message delivered successfully!");
            println!("  Message ID: {}", receipt.message_id);
        }
        other => {
            println!("  Delivery status: {:?}", other);
        }
    }

    // Consumer receives the message
    if let Some(message) = consumer_endpoint.try_receive() {
        println!("\nConsumer received message:");
        println!("  From: {}", message.source);
        println!("  To: {}", message.destination);
        println!("  Priority: {}", message.priority);
        println!("  Hops: {}", message.hops);
    }

    println!();

    // ========== Priority Messaging ==========
    println!("=== Priority Messaging ===\n");

    // Send messages with different priorities
    for priority in [0, 50, 100, 200] {
        let envelope = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));
        let receipt = producer_endpoint
            .send_priority(consumer_id.clone(), envelope, priority)
            .await?;

        println!(
            "Sent priority {} message: {:?}",
            priority, receipt.status
        );
    }

    // Receive all messages
    println!("\nReceiving priority messages:");
    while let Some(message) = consumer_endpoint.try_receive() {
        println!("  Received message with priority: {}", message.priority);
    }

    println!();

    // ========== Bidirectional Communication ==========
    println!("=== Bidirectional Communication ===\n");

    // Create ping-pong pattern
    let alice_id = KernelId::new("alice");
    let bob_id = KernelId::new("bob");

    let alice_endpoint = broker.register(alice_id.clone());
    let bob_endpoint = broker.register(bob_id.clone());

    // Alice sends to Bob
    let envelope = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));
    alice_endpoint.send(bob_id.clone(), envelope).await?;
    println!("Alice -> Bob: Sent message");

    // Bob sends to Alice
    let envelope = MessageEnvelope::empty(2, 1, HlcTimestamp::now(2));
    bob_endpoint.send(alice_id.clone(), envelope).await?;
    println!("Bob -> Alice: Sent message");

    // Check final stats
    let stats = broker.stats();
    println!(
        "\nFinal broker stats: {} endpoints, {} messages delivered",
        stats.registered_endpoints, stats.messages_delivered
    );

    println!();

    // ========== Using Runtime Kernels ==========
    println!("=== K2K with Runtime Kernels ===\n");

    // Launch actual kernels through the runtime
    let kernel_a = runtime.launch("kernel_a", LaunchOptions::default()).await?;
    let kernel_b = runtime.launch("kernel_b", LaunchOptions::default()).await?;

    println!("Launched kernels: {} and {}", kernel_a.id(), kernel_b.id());

    // These kernels are automatically registered with the K2K broker
    assert!(broker.is_registered(&KernelId::new("kernel_a")));
    assert!(broker.is_registered(&KernelId::new("kernel_b")));
    println!("Both kernels are registered with K2K broker");

    let stats = broker.stats();
    println!(
        "Broker now has {} registered endpoints\n",
        stats.registered_endpoints
    );

    // Cleanup
    kernel_a.terminate().await?;
    kernel_b.terminate().await?;

    // Unregister the manually registered endpoints
    broker.unregister(&producer_id);
    broker.unregister(&consumer_id);
    broker.unregister(&alice_id);
    broker.unregister(&bob_id);

    println!("=== Example completed! ===");

    Ok(())
}
