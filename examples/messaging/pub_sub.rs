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
//! - Single-level wildcard: `"sensors/+"` matches `sensors/temp`, `sensors/humidity`
//! - Multi-level wildcard: `"sensors/#"` matches `sensors/a`, `sensors/a/b/c`
//!
//! ## Run this example:
//! ```bash
//! cargo run --example pub_sub
//! ```

use ringkernel::prelude::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Pub/Sub Messaging Example ===\n");

    // Create the PubSub broker
    let config = PubSubConfig {
        channel_buffer_size: 1024,
        max_retained_messages: 100,
        ..Default::default()
    };

    let broker = PubSubBroker::new(config);

    println!("PubSub broker created.\n");

    // ====== Basic Pub/Sub ======
    println!("=== Basic Pub/Sub ===");
    demonstrate_basic_pubsub(&broker).await?;

    // ====== Wildcard Topics ======
    println!("\n=== Wildcard Topics ===");
    demonstrate_wildcards(&broker).await?;

    // ====== QoS Levels ======
    println!("\n=== Quality of Service ===");
    demonstrate_qos(&broker).await?;

    // ====== Message Retention ======
    println!("\n=== Message Retention ===");
    demonstrate_retention(&broker).await?;

    // ====== Real-World Scenario ======
    println!("\n=== Real-World Scenario: IoT Sensor Network ===");
    demonstrate_iot_scenario(&broker).await?;

    println!("\n=== Example completed! ===");
    Ok(())
}

async fn demonstrate_basic_pubsub(broker: &Arc<PubSubBroker>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating subscriptions...\n");

    // Create subscribers
    let kernel1 = KernelId::new("processor_1");
    let kernel2 = KernelId::new("processor_2");

    // Subscribe to specific topics
    let sub1 = broker.subscribe(
        &kernel1,
        Topic::new("events/user/login"),
        QoS::AtMostOnce,
    )?;

    let sub2 = broker.subscribe(
        &kernel2,
        Topic::new("events/user/login"),
        QoS::AtMostOnce,
    )?;

    println!("  Subscription 1 ID: {}", sub1.id);
    println!("  Subscription 2 ID: {}", sub2.id);

    // Publish a message
    let publisher = KernelId::new("auth_service");
    let envelope = MessageEnvelope::empty(1, 0, HlcTimestamp::now(1));

    broker.publish(
        &publisher,
        Topic::new("events/user/login"),
        envelope,
        QoS::AtMostOnce,
        false,
    ).await?;

    println!("\n  Published to 'events/user/login'");
    println!("  Both subscribers will receive the message");

    // Unsubscribe
    broker.unsubscribe(sub1.id)?;
    println!("\n  Unsubscribed processor_1");

    Ok(())
}

async fn demonstrate_wildcards(broker: &Arc<PubSubBroker>) -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = KernelId::new("wildcard_subscriber");

    // Single-level wildcard (+)
    println!("Single-level wildcard '+':");
    let pattern = Topic::new("sensors/+/temperature");
    println!("  Pattern: {}", pattern.name());
    println!("  Matches: sensors/room1/temperature, sensors/room2/temperature");
    println!("  No match: sensors/building/room1/temperature\n");

    broker.subscribe(&subscriber, pattern, QoS::AtMostOnce)?;

    // Multi-level wildcard (#)
    println!("Multi-level wildcard '#':");
    let pattern = Topic::new("alerts/#");
    println!("  Pattern: {}", pattern.name());
    println!("  Matches: alerts/high, alerts/low, alerts/system/cpu/high");

    broker.subscribe(&subscriber, pattern, QoS::AtMostOnce)?;

    // Test matching
    println!("\nTesting topic matching:");
    let test_cases = [
        ("sensors/room1/temperature", "sensors/+/temperature", true),
        ("sensors/room2/temperature", "sensors/+/temperature", true),
        ("sensors/a/b/temperature", "sensors/+/temperature", false),
        ("alerts/high", "alerts/#", true),
        ("alerts/system/cpu", "alerts/#", true),
    ];

    for (topic, pattern, expected) in test_cases {
        let matches = Topic::new(topic).matches_pattern(&Topic::new(pattern));
        let status = if matches == expected { "OK" } else { "FAIL" };
        println!("  {} ~ {} => {} [{}]", topic, pattern, matches, status);
    }

    Ok(())
}

async fn demonstrate_qos(broker: &Arc<PubSubBroker>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Quality of Service levels:\n");

    let qos_levels = [
        (QoS::AtMostOnce, "At Most Once", "Fire and forget - fastest, no guarantee"),
        (QoS::AtLeastOnce, "At Least Once", "Retries until ACK - may duplicate"),
        (QoS::ExactlyOnce, "Exactly Once", "Deduplication - slowest, reliable"),
    ];

    for (qos, name, description) in qos_levels {
        println!("  {:?} - {}", qos, name);
        println!("    {}\n", description);
    }

    println!("Choosing QoS:");
    println!("  - Metrics/telemetry: AtMostOnce (loss acceptable)");
    println!("  - Commands/events: AtLeastOnce (idempotent handlers)");
    println!("  - Financial/critical: ExactlyOnce (must process once)");

    Ok(())
}

async fn demonstrate_retention(broker: &Arc<PubSubBroker>) -> Result<(), Box<dyn std::error::Error>> {
    let publisher = KernelId::new("config_service");

    // Publish with retention
    println!("Publishing retained message...");
    let envelope = MessageEnvelope::empty(1, 0, HlcTimestamp::now(1));

    broker.publish(
        &publisher,
        Topic::new("config/database/connection"),
        envelope,
        QoS::AtLeastOnce,
        true, // retain = true
    ).await?;

    println!("  Topic: config/database/connection");
    println!("  Retained: true\n");

    // New subscriber gets retained message immediately
    println!("New subscriber joining...");
    let new_subscriber = KernelId::new("new_service");
    let _sub = broker.subscribe(
        &new_subscriber,
        Topic::new("config/database/connection"),
        QoS::AtLeastOnce,
    )?;

    println!("  New subscriber immediately receives last retained message");
    println!("  Useful for: configuration, status, last-known-good values");

    // Get retained messages
    let retained = broker.get_retained(&Topic::new("config/database/connection"));
    println!("\n  Retained messages on topic: {}", retained.len());

    Ok(())
}

async fn demonstrate_iot_scenario(broker: &Arc<PubSubBroker>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Simulating IoT sensor network:\n");

    // Subscribers with different interests
    let dashboard = KernelId::new("dashboard");
    let alerting = KernelId::new("alerting");
    let storage = KernelId::new("storage");

    // Dashboard wants all sensor data
    broker.subscribe(&dashboard, Topic::new("sensors/#"), QoS::AtMostOnce)?;
    println!("  Dashboard subscribed to: sensors/#");

    // Alerting only wants temperature anomalies
    broker.subscribe(&alerting, Topic::new("sensors/+/temperature/alert"), QoS::AtLeastOnce)?;
    println!("  Alerting subscribed to: sensors/+/temperature/alert");

    // Storage wants everything for archival
    broker.subscribe(&storage, Topic::new("#"), QoS::ExactlyOnce)?;
    println!("  Storage subscribed to: # (everything)\n");

    // Simulate sensor publications
    let sensors = ["room1", "room2", "server_room"];
    let sensor_kernel = KernelId::new("sensor_gateway");

    println!("Publishing sensor data:");
    for sensor in sensors {
        // Regular temperature reading
        let topic = format!("sensors/{}/temperature", sensor);
        let envelope = MessageEnvelope::empty(1, 0, HlcTimestamp::now(1));
        broker.publish(
            &sensor_kernel,
            Topic::new(&topic),
            envelope,
            QoS::AtMostOnce,
            false,
        ).await?;
        println!("  Published: {}", topic);
    }

    // Temperature alert
    let alert_topic = "sensors/server_room/temperature/alert";
    let envelope = MessageEnvelope::empty(1, 0, HlcTimestamp::now(1));
    broker.publish(
        &sensor_kernel,
        Topic::new(alert_topic),
        envelope,
        QoS::AtLeastOnce,
        false,
    ).await?;
    println!("  Published: {} (ALERT)", alert_topic);

    println!("\nMessage routing:");
    println!("  Dashboard receives: all 4 messages");
    println!("  Alerting receives: 1 message (the alert)");
    println!("  Storage receives: all 4 messages");

    // Show stats
    let stats = broker.stats();
    println!("\nBroker statistics:");
    println!("  Total subscriptions: {}", stats.total_subscriptions);
    println!("  Messages published: {}", stats.messages_published);

    Ok(())
}
