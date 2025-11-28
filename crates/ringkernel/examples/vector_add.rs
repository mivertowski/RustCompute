//! Vector Add Example
//!
//! Demonstrates message passing pattern with custom message types
//! for a vector addition operation.

use ringkernel::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

/// Request message for vector addition.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct VectorAddRequest {
    /// Message ID.
    pub id: u64,
    /// Correlation ID for request-response matching.
    pub correlation_id: u64,
    /// First vector.
    pub a: Vec<f32>,
    /// Second vector.
    pub b: Vec<f32>,
}

impl RingMessage for VectorAddRequest {
    fn message_type() -> u64 {
        0x1001 // Vector add request type
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
            .map_err(|_| RingKernelError::DeserializationError("Failed to deserialize".to_string()))
    }
}

/// Response message for vector addition.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct VectorAddResponse {
    /// Message ID.
    pub id: u64,
    /// Correlation ID matching the request.
    pub correlation_id: u64,
    /// Result vector.
    pub result: Vec<f32>,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
}

impl RingMessage for VectorAddResponse {
    fn message_type() -> u64 {
        0x1002 // Vector add response type
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
            .map_err(|_| RingKernelError::DeserializationError("Failed to deserialize".to_string()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("🧮 RingKernel Vector Add Example\n");

    // Create runtime
    let runtime = RingKernel::new().await?;
    println!("Runtime backend: {:?}\n", runtime.backend());

    // Launch vector add kernel
    let kernel = runtime
        .launch("vector_add", LaunchOptions::default())
        .await?;

    println!("Launched kernel: {}", kernel.id());

    // Create test vectors
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    println!("\nInput vectors:");
    println!("  a = {:?}", a);
    println!("  b = {:?}", b);

    // Create request
    let request = VectorAddRequest {
        id: MessageId::generate().inner(),
        correlation_id: CorrelationId::generate().0,
        a: a.clone(),
        b: b.clone(),
    };

    println!(
        "\nSending request (correlation_id: {})...",
        request.correlation_id
    );

    // Send message
    kernel.send(request.clone()).await?;

    // In a real GPU kernel, the processing would happen on the GPU
    // For this example with CPU backend, we simulate the response
    println!("(Simulating GPU processing...)");

    // Simulate response (in real usage, kernel handler would create this)
    let response = VectorAddResponse {
        id: MessageId::generate().inner(),
        correlation_id: request.correlation_id,
        result: a.iter().zip(&b).map(|(x, y)| x + y).collect(),
        processing_time_us: 42,
    };

    println!("\nResult:");
    println!("  result = {:?}", response.result);
    println!("  processing time: {}µs", response.processing_time_us);

    // Cleanup
    kernel.terminate().await?;
    runtime.shutdown().await?;

    println!("\n✓ Vector add example complete!");

    Ok(())
}
