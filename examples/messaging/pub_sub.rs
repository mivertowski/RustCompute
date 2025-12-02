//! # Topic-based Pub/Sub Messaging
//!
//! This example demonstrates the publish-subscribe pattern using
//! RingKernel's PubSubBroker for decoupled kernel communication.
//!
//! ## Pattern Overview
//!
//! ```text
//! Publishers                    Broker                    Subscribers
//!
//! ┌─────────┐                 ┌────────┐                ┌─────────────┐
//! │Kernel A │──publish───────▶│        │───────────────▶│ Kernel X    │
//! └─────────┘   "sensors/*"   │        │  "sensors/+"   └─────────────┘
//!                             │ PubSub │
//! ┌─────────┐                 │ Broker │                ┌─────────────┐
//! │Kernel B │──publish───────▶│        │───────────────▶│ Kernel Y    │
//! └─────────┘   "alerts/high" │        │  "alerts/#"    └─────────────┘
//!                             └────────┘
//! ```
//!
//! ## Topic Patterns
//!
//! - Exact match: `"sensors/temperature"` matches only that topic
//! - Single-level wildcard: `"sensors/*"` matches `sensors/temp`, `sensors/humidity`
//! - Multi-level wildcard: `"sensors/#"` matches `sensors/a`, `sensors/a/b/c`
//!
//! ## Run this example:
//! ```bash
//! cargo run -p ringkernel --example pub_sub
//! ```

use ringkernel::prelude::*;
use ringkernel::pubsub::PubSubConfig;
use std::sync::Arc;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Pub/Sub Messaging Example ===\n");

    // Create the PubSub broker with custom configuration
    let config = PubSubConfig {
        channel_buffer_size: 1024,
        max_retained_messages: 100,
        ..Default::default()
    };

    let broker = PubSubBroker::new(config);

    println!("PubSub broker created.\n");

    // ====== Basic Pub/Sub ======
    println!("=== Basic Pub/Sub ===");
    demonstrate_basic_pubsub(&broker).await;

    // ====== Wildcard Topics ======
    println!("\n=== Wildcard Topics ===");
    demonstrate_wildcards(&broker).await;

    // ====== QoS Levels ======
    println!("\n=== Quality of Service ===");
    demonstrate_qos();

    // ====== Real-World Scenario ======
    println!("\n=== Real-World Scenario: IoT Sensor Network ===");
    demonstrate_iot_scenario(&broker).await;

    println!("\n=== Example completed! ===");
    Ok(())
}

async fn demonstrate_basic_pubsub(broker: &Arc<PubSubBroker>) {
    println!("Creating subscriptions...\n");

    // Create subscribers (simulated kernel IDs)
    let kernel1 = KernelId::new("processor_1");
    let kernel2 = KernelId::new("processor_2");

    // Subscribe to specific topics
    let mut sub1 = broker.subscribe(kernel1.clone(), Topic::new("events/user/login"));
    let mut sub2 = broker.subscribe(kernel2.clone(), Topic::new("events/user/login"));

    println!("  Subscription 1 ID: {}", sub1.id);
    println!("  Subscription 2 ID: {}", sub2.id);

    // Publish a message
    let publisher = KernelId::new("auth_service");
    let envelope = MessageEnvelope::empty(1, 0, HlcTimestamp::now(1));

    broker
        .publish(
            Topic::new("events/user/login"),
            publisher,
            envelope,
            HlcTimestamp::now(1),
        )
        .expect("publish failed");

    println!("\n  Published to 'events/user/login'");
    println!("  Both subscribers will receive the message");

    // Try to receive (non-blocking for demo)
    if let Some(pub_msg) = sub1.try_receive() {
        println!("  Subscriber 1 received: topic={}", pub_msg.topic);
    }

    // Unsubscribe
    broker.unsubscribe(sub1.id);
    println!("\n  Unsubscribed processor_1");
}

async fn demonstrate_wildcards(broker: &Arc<PubSubBroker>) {
    let subscriber = KernelId::new("wildcard_subscriber");

    // Single-level wildcard (*)
    println!("Single-level wildcard '*':");
    let pattern = Topic::new("sensors/*/temperature");
    println!("  Pattern: {}", pattern.name());
    println!("  Matches: sensors/room1/temperature, sensors/room2/temperature");
    println!("  No match: sensors/building/room1/temperature\n");

    let _sub1 = broker.subscribe(subscriber.clone(), pattern);

    // Multi-level wildcard (#)
    println!("Multi-level wildcard '#':");
    let pattern = Topic::new("alerts/#");
    println!("  Pattern: {}", pattern.name());
    println!("  Matches: alerts/high, alerts/low, alerts/system/cpu/high");

    let _sub2 = broker.subscribe(subscriber.clone(), pattern);

    // Test matching
    println!("\nTesting topic matching:");
    let test_cases = [
        ("sensors/room1/temperature", "sensors/*/temperature", true),
        ("sensors/room2/temperature", "sensors/*/temperature", true),
        ("sensors/a/b/temperature", "sensors/*/temperature", false),
        ("alerts/high", "alerts/#", true),
        ("alerts/system/cpu", "alerts/#", true),
    ];

    for (topic, pattern, expected) in test_cases {
        let topic_obj = Topic::new(topic);
        let pattern_obj = Topic::new(pattern);
        let matches = pattern_obj.matches(&topic_obj);
        let status = if matches == expected { "OK" } else { "FAIL" };
        println!("  {} ~ {} => {} [{}]", topic, pattern, matches, status);
    }
}

fn demonstrate_qos() {
    println!("Quality of Service levels:\n");

    let qos_levels = [
        (QoS::AtMostOnce, "At Most Once", "Fire and forget - fastest, no guarantee"),
        (
            QoS::AtLeastOnce,
            "At Least Once",
            "Retries until ACK - may duplicate",
        ),
        (
            QoS::ExactlyOnce,
            "Exactly Once",
            "Deduplication - slowest, reliable",
        ),
    ];

    for (qos, name, description) in qos_levels {
        println!("  {:?} - {}", qos, name);
        println!("    {}\n", description);
    }

    println!("Choosing QoS:");
    println!("  - Metrics/telemetry: AtMostOnce (loss acceptable)");
    println!("  - Commands/events: AtLeastOnce (idempotent handlers)");
    println!("  - Financial/critical: ExactlyOnce (must process once)");
}

async fn demonstrate_iot_scenario(broker: &Arc<PubSubBroker>) {
    println!("Simulating IoT sensor network:\n");

    // Subscribers with different interests
    let dashboard = KernelId::new("dashboard");
    let alerting = KernelId::new("alerting");
    let storage = KernelId::new("storage");

    // Dashboard wants all sensor data
    let _sub_dashboard = broker.subscribe(dashboard.clone(), Topic::new("sensors/#"));
    println!("  Dashboard subscribed to: sensors/#");

    // Alerting only wants temperature anomalies
    let _sub_alerting = broker.subscribe(alerting.clone(), Topic::new("sensors/*/temperature/alert"));
    println!("  Alerting subscribed to: sensors/*/temperature/alert");

    // Storage wants everything for archival
    let _sub_storage = broker.subscribe(storage.clone(), Topic::new("#"));
    println!("  Storage subscribed to: # (everything)\n");

    // Simulate sensor publications
    let sensors = ["room1", "room2", "server_room"];
    let sensor_kernel = KernelId::new("sensor_gateway");

    println!("Publishing sensor data:");
    for sensor in sensors {
        // Regular temperature reading
        let topic = format!("sensors/{}/temperature", sensor);
        let envelope = MessageEnvelope::empty(1, 0, HlcTimestamp::now(1));
        broker
            .publish(
                Topic::new(&topic),
                sensor_kernel.clone(),
                envelope,
                HlcTimestamp::now(1),
            )
            .expect("publish failed");
        println!("  Published: {}", topic);
    }

    // Temperature alert
    let alert_topic = "sensors/server_room/temperature/alert";
    let envelope = MessageEnvelope::empty(1, 0, HlcTimestamp::now(1));
    broker
        .publish(
            Topic::new(alert_topic),
            sensor_kernel.clone(),
            envelope,
            HlcTimestamp::now(1),
        )
        .expect("publish failed");
    println!("  Published: {} (ALERT)", alert_topic);

    println!("\nMessage routing:");
    println!("  Dashboard receives: all 4 messages (sensors/#)");
    println!("  Alerting receives: 1 message (the alert)");
    println!("  Storage receives: all 4 messages (#)");

    // Note: In a real scenario, you'd use async receive or spawn tasks
    println!("\nNote: Use subscription.receive().await in production for async message handling");
}
