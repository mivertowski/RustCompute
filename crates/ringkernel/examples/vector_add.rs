//! # Vector Add Example
//!
//! Demonstrates message passing with custom message types for vector addition.
//! Shows two approaches:
//! 1. Manual `RingMessage` trait implementation (educational)
//! 2. Using `#[derive(RingMessage)]` macro (recommended)
//!
//! ## Run this example:
//! ```bash
//! cargo run -p ringkernel --example vector_add
//! ```
//!
//! ## What this demonstrates:
//!
//! - Custom message types with RingMessage trait
//! - Zero-copy serialization via rkyv
//! - Correlation ID for request-response matching
//! - Both manual and derive-macro approaches

use ringkernel::prelude::*;
use ringkernel_derive::RingMessage;
use rkyv::{Archive, Deserialize, Serialize};

// ============================================================================
// APPROACH 1: Derive Macro (Recommended)
// ============================================================================

/// Request message using the derive macro - simple and concise.
#[derive(Debug, Clone, RingMessage, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
#[message(type_id = 0x2001)]
pub struct DerivedVectorAddRequest {
    /// Message ID field
    #[message(id)]
    pub id: MessageId,
    /// Correlation ID for request-response matching
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// First vector
    pub a: Vec<f32>,
    /// Second vector
    pub b: Vec<f32>,
}

/// Response message using derive macro
#[derive(Debug, Clone, RingMessage, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
#[message(type_id = 0x2002)]
pub struct DerivedVectorAddResponse {
    #[message(id)]
    pub id: MessageId,
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Result vector
    pub result: Vec<f32>,
    /// Processing time in microseconds
    pub processing_time_us: u64,
}

// ============================================================================
// APPROACH 2: Manual Implementation (Educational)
// ============================================================================

/// Request message with manual RingMessage implementation.
/// This shows exactly what the derive macro generates.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct ManualVectorAddRequest {
    pub id: u64,
    pub correlation_id: u64,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl RingMessage for ManualVectorAddRequest {
    fn message_type() -> u64 {
        0x1001 // Explicit type ID
    }

    fn message_id(&self) -> MessageId {
        MessageId::new(self.id)
    }

    fn correlation_id(&self) -> CorrelationId {
        CorrelationId::new(self.correlation_id)
    }

    fn priority(&self) -> Priority {
        Priority::Normal
    }

    fn serialize(&self) -> Vec<u8> {
        rkyv::to_bytes::<_, 256>(self)
            .map(|v| v.to_vec())
            .unwrap_or_default()
    }

    fn deserialize(bytes: &[u8]) -> Result<Self> {
        let archived = unsafe { rkyv::archived_root::<Self>(bytes) };
        archived
            .deserialize(&mut rkyv::Infallible)
            .map_err(|_| RingKernelError::DeserializationError("Failed to deserialize".into()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== RingKernel Vector Add Example ===\n");

    // Create runtime
    let runtime = RingKernel::new().await?;
    println!("Runtime backend: {:?}\n", runtime.backend());

    // Launch kernel
    let kernel = runtime
        .launch("vector_add", LaunchOptions::default())
        .await?;

    println!("Launched kernel: {}\n", kernel.id());

    // Test vectors
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    println!("Input vectors:");
    println!("  a = {:?}", a);
    println!("  b = {:?}\n", b);

    // ========== Derived Message Demo ==========
    println!("--- Using #[derive(RingMessage)] ---\n");

    let derived_request = DerivedVectorAddRequest {
        id: MessageId::generate(),
        correlation_id: CorrelationId::generate(),
        a: a.clone(),
        b: b.clone(),
    };

    println!(
        "Message type ID: 0x{:04X}",
        DerivedVectorAddRequest::message_type()
    );
    println!("Message ID: {}", derived_request.message_id());
    println!("Correlation ID: {:?}", derived_request.correlation_id());

    // Serialize and deserialize (use fully qualified syntax to disambiguate)
    let bytes = RingMessage::serialize(&derived_request);
    println!("Serialized size: {} bytes", bytes.len());

    let restored = <DerivedVectorAddRequest as RingMessage>::deserialize(&bytes)?;
    println!("Restored a = {:?}", restored.a);
    println!("Restored b = {:?}\n", restored.b);

    // ========== Manual Implementation Demo ==========
    println!("--- Using Manual Implementation ---\n");

    let manual_request = ManualVectorAddRequest {
        id: MessageId::generate().inner(),
        correlation_id: CorrelationId::generate().0,
        a: a.clone(),
        b: b.clone(),
    };

    println!(
        "Message type ID: 0x{:04X}",
        ManualVectorAddRequest::message_type()
    );

    let bytes = RingMessage::serialize(&manual_request);
    println!("Serialized size: {} bytes", bytes.len());

    let restored = <ManualVectorAddRequest as RingMessage>::deserialize(&bytes)?;
    println!("Restored a = {:?}", restored.a);
    println!("Restored b = {:?}\n", restored.b);

    // ========== Simulate Processing ==========
    println!("--- Simulated GPU Processing ---\n");

    // In a real GPU kernel, processing would happen on device
    let result: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let response = DerivedVectorAddResponse {
        id: MessageId::generate(),
        correlation_id: derived_request.correlation_id,
        result,
        processing_time_us: 42,
    };

    println!("Result: {:?}", response.result);
    println!("Processing time: {}us", response.processing_time_us);

    // Cleanup
    kernel.terminate().await?;
    runtime.shutdown().await?;

    println!("\n=== Vector Add Example Complete! ===");
    println!("\nRecommendation: Use #[derive(RingMessage)] for cleaner code.");
    println!("The manual implementation shows what the macro generates.");

    Ok(())
}
