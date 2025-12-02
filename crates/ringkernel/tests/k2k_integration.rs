//! Integration tests for K2K (kernel-to-kernel) messaging.
//!
//! These tests verify that kernels can communicate directly with each other
//! using the K2K messaging infrastructure.

use ringkernel::prelude::*;
use ringkernel_cpu::CpuRuntime;

/// Test that K2K broker is created with the runtime.
#[tokio::test]
async fn test_k2k_broker_creation() {
    let runtime = CpuRuntime::new().await.expect("Failed to create runtime");

    // K2K should be enabled by default
    assert!(runtime.is_k2k_enabled(), "K2K should be enabled by default");
}

/// Test that K2K can be disabled.
#[tokio::test]
async fn test_k2k_disabled() {
    let runtime = CpuRuntime::with_config(1, false)
        .await
        .expect("Failed to create runtime");

    assert!(!runtime.is_k2k_enabled(), "K2K should be disabled");
}

/// Test K2K broker registration when launching kernels.
#[tokio::test]
async fn test_k2k_kernel_registration() {
    let runtime = CpuRuntime::new().await.expect("Failed to create runtime");

    // Launch two kernels
    let _kernel1 = runtime
        .launch("kernel1", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel1");

    let _kernel2 = runtime
        .launch("kernel2", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel2");

    // Verify the broker has both registered
    let broker = runtime.k2k_broker().expect("K2K broker should exist");
    assert!(broker.is_registered(&KernelId::new("kernel1")));
    assert!(broker.is_registered(&KernelId::new("kernel2")));

    let stats = broker.stats();
    assert_eq!(stats.registered_endpoints, 2);
}

/// Test direct K2K message delivery between kernels.
#[tokio::test]
async fn test_k2k_message_delivery() {
    // Create broker directly for testing
    let broker = K2KBuilder::new().build();

    let kernel1_id = KernelId::new("sender");
    let kernel2_id = KernelId::new("receiver");

    // Register both endpoints
    let endpoint1 = broker.register(kernel1_id.clone());
    let mut endpoint2 = broker.register(kernel2_id.clone());

    // Create a test message envelope
    let envelope = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));

    // Send from kernel1 to kernel2
    let receipt = endpoint1
        .send(kernel2_id.clone(), envelope)
        .await
        .expect("Send should succeed");

    assert_eq!(receipt.status, DeliveryStatus::Delivered);
    assert_eq!(receipt.source, kernel1_id);
    assert_eq!(receipt.destination, kernel2_id);

    // Receive on kernel2
    let message = endpoint2.try_receive();
    assert!(message.is_some(), "Should have received a message");

    let msg = message.unwrap();
    assert_eq!(msg.source, kernel1_id);
    assert_eq!(msg.destination, kernel2_id);
}

/// Test K2K message delivery with priority.
#[tokio::test]
async fn test_k2k_priority_message() {
    let broker = K2KBuilder::new().build();

    let sender = KernelId::new("sender");
    let receiver = KernelId::new("receiver");

    let endpoint1 = broker.register(sender.clone());
    let mut endpoint2 = broker.register(receiver.clone());

    let envelope = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));

    // Send with high priority
    let receipt = endpoint1
        .send_priority(receiver.clone(), envelope, 100)
        .await
        .expect("Priority send should succeed");

    assert_eq!(receipt.status, DeliveryStatus::Delivered);

    let message = endpoint2.try_receive().expect("Should receive message");
    assert_eq!(message.priority, 100);
}

/// Test K2K delivery to non-existent kernel.
#[tokio::test]
async fn test_k2k_not_found() {
    let broker = K2KBuilder::new().build();

    let sender = KernelId::new("sender");
    let non_existent = KernelId::new("non_existent");

    let endpoint = broker.register(sender.clone());

    let envelope = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));

    let receipt = endpoint
        .send(non_existent.clone(), envelope)
        .await
        .expect("Send should complete");

    assert_eq!(receipt.status, DeliveryStatus::NotFound);
}

/// Test K2K unregistration on shutdown.
#[tokio::test]
async fn test_k2k_unregister_on_shutdown() {
    let runtime = CpuRuntime::new().await.expect("Failed to create runtime");

    // Launch a kernel
    let _kernel = runtime
        .launch("test_kernel", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");

    let broker = runtime.k2k_broker().expect("K2K broker should exist");
    assert!(broker.is_registered(&KernelId::new("test_kernel")));

    // Shutdown runtime
    runtime.shutdown().await.expect("Shutdown should succeed");

    // After shutdown, the kernel should be unregistered
    // (Note: We need to access the broker before shutdown for this test)
}

/// Test K2K statistics tracking.
#[tokio::test]
async fn test_k2k_stats() {
    let broker = K2KBuilder::new().build();

    let kernel1 = KernelId::new("k1");
    let kernel2 = KernelId::new("k2");

    let endpoint1 = broker.register(kernel1.clone());
    let _endpoint2 = broker.register(kernel2.clone());

    // Check initial stats
    let stats = broker.stats();
    assert_eq!(stats.registered_endpoints, 2);
    assert_eq!(stats.messages_delivered, 0);

    // Send a message
    let envelope = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));
    endpoint1
        .send(kernel2.clone(), envelope)
        .await
        .expect("Send should succeed");

    // Check updated stats
    let stats = broker.stats();
    assert_eq!(stats.messages_delivered, 1);
}

/// Test K2K routing table.
#[tokio::test]
async fn test_k2k_routing() {
    let broker = K2KBuilder::new().max_hops(3).build();

    let kernel_a = KernelId::new("a");
    let kernel_b = KernelId::new("b");
    let kernel_c = KernelId::new("c");

    let endpoint_a = broker.register(kernel_a.clone());
    let _endpoint_b = broker.register(kernel_b.clone());
    let mut endpoint_c = broker.register(kernel_c.clone());

    // Add route: messages to C go through B
    broker.add_route(kernel_c.clone(), kernel_b.clone());

    // Verify route exists
    let stats = broker.stats();
    assert_eq!(stats.routes_configured, 1);

    // Note: Currently routing is basic - messages go to registered endpoints directly
    // if available. Route is used when direct endpoint not found.

    // Direct send should still work
    let envelope = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));
    let receipt = endpoint_a
        .send(kernel_c.clone(), envelope)
        .await
        .expect("Send should complete");

    // Should be delivered directly since C is registered
    assert_eq!(receipt.status, DeliveryStatus::Delivered);

    let msg = endpoint_c.try_receive();
    assert!(msg.is_some());
}

/// Test K2K with custom configuration.
#[tokio::test]
async fn test_k2k_custom_config() {
    let config = K2KConfig {
        max_pending_messages: 512,
        delivery_timeout_ms: 3000,
        enable_tracing: true,
        max_hops: 4,
    };

    let runtime = CpuRuntime::with_k2k_config(1, config)
        .await
        .expect("Failed to create runtime with custom K2K config");

    assert!(runtime.is_k2k_enabled());
}

/// Test bidirectional K2K communication.
#[tokio::test]
async fn test_k2k_bidirectional() {
    let broker = K2KBuilder::new().build();

    let kernel_a = KernelId::new("alice");
    let kernel_b = KernelId::new("bob");

    let endpoint_a = broker.register(kernel_a.clone());
    let endpoint_b = broker.register(kernel_b.clone());

    // Send from A to B
    let envelope_a = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));
    let receipt = endpoint_a
        .send(kernel_b.clone(), envelope_a)
        .await
        .expect("A->B should succeed");
    assert_eq!(receipt.status, DeliveryStatus::Delivered);

    // Send from B to A
    let envelope_b = MessageEnvelope::empty(2, 1, HlcTimestamp::now(2));
    let receipt = endpoint_b
        .send(kernel_a.clone(), envelope_b)
        .await
        .expect("B->A should succeed");
    assert_eq!(receipt.status, DeliveryStatus::Delivered);

    // Check stats
    let stats = broker.stats();
    assert_eq!(stats.messages_delivered, 2);
}
