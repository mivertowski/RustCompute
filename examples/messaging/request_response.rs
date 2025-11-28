//! # Request-Response Messaging Pattern
//!
//! This example demonstrates the request-response messaging pattern
//! using RingKernel's type-safe message passing.
//!
//! ## Pattern Overview
//!
//! ```text
//! ┌─────────┐   Request    ┌─────────────┐
//! │  Host   │─────────────▶│   Kernel    │
//! │ (CPU)   │              │   (GPU)     │
//! │         │◀─────────────│             │
//! └─────────┘   Response   └─────────────┘
//! ```
//!
//! ## Key Concepts
//!
//! - **Messages are type-safe**: Define message types with the `RingMessage` trait
//! - **Correlation IDs**: Track request-response pairs
//! - **Priorities**: Control processing order
//! - **Timestamps**: HLC for causal ordering
//!
//! ## Run this example:
//! ```bash
//! cargo run --example request_response
//! ```

use ringkernel::prelude::*;
use std::time::Duration;

// Define request/response message types
// In a real application, derive RingMessage with the proc macro

/// Vector addition request
#[derive(Debug, Clone)]
struct VectorAddRequest {
    /// Unique message ID
    id: MessageId,
    /// First vector
    a: Vec<f32>,
    /// Second vector
    b: Vec<f32>,
    /// Correlation ID for tracking
    correlation_id: Option<CorrelationId>,
}

impl VectorAddRequest {
    fn new(a: Vec<f32>, b: Vec<f32>) -> Self {
        Self {
            id: MessageId::generate(),
            a,
            b,
            correlation_id: Some(CorrelationId::generate()),
        }
    }
}

/// Vector addition response
#[derive(Debug, Clone)]
struct VectorAddResponse {
    /// Response message ID
    id: MessageId,
    /// Result vector
    result: Vec<f32>,
    /// Original correlation ID
    correlation_id: Option<CorrelationId>,
    /// Processing timestamp
    timestamp: HlcTimestamp,
}

/// Matrix multiply request
#[derive(Debug, Clone)]
struct MatrixMultiplyRequest {
    id: MessageId,
    /// Matrix A (row-major)
    a: Vec<f32>,
    /// Matrix B (row-major)
    b: Vec<f32>,
    /// Matrix dimensions (m, k, n)
    dims: (usize, usize, usize),
    priority: u8,
}

impl MatrixMultiplyRequest {
    fn new(a: Vec<f32>, b: Vec<f32>, dims: (usize, usize, usize)) -> Self {
        Self {
            id: MessageId::generate(),
            a,
            b,
            dims,
            priority: priority::NORMAL,
        }
    }

    fn with_high_priority(mut self) -> Self {
        self.priority = priority::HIGH;
        self
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Request-Response Pattern Example ===\n");

    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await?;

    // Launch compute kernels
    let vector_kernel = runtime
        .launch("vector_compute", LaunchOptions::default())
        .await?;

    let matrix_kernel = runtime
        .launch("matrix_compute", LaunchOptions::default().with_priority(priority::HIGH))
        .await?;

    vector_kernel.activate().await?;
    matrix_kernel.activate().await?;

    println!("Kernels activated.\n");

    // ====== Basic Request-Response ======
    println!("=== Basic Request-Response ===");
    demonstrate_basic_request_response().await;

    // ====== Correlated Requests ======
    println!("\n=== Correlated Requests ===");
    demonstrate_correlation().await;

    // ====== Priority-based Processing ======
    println!("\n=== Priority-based Processing ===");
    demonstrate_priorities().await;

    // ====== Batch Requests ======
    println!("\n=== Batch Requests ===");
    demonstrate_batch_requests().await;

    // ====== Timeout Handling ======
    println!("\n=== Timeout Handling ===");
    demonstrate_timeout_handling().await;

    // Cleanup
    vector_kernel.terminate().await?;
    matrix_kernel.terminate().await?;
    runtime.shutdown().await?;

    println!("\n=== Example completed! ===");
    Ok(())
}

async fn demonstrate_basic_request_response() {
    println!("Sending a vector addition request...");

    let request = VectorAddRequest::new(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
    );

    println!("  Request ID: {}", request.id);
    println!("  Vector A: {:?}", request.a);
    println!("  Vector B: {:?}", request.b);

    // In a real implementation, you would:
    // kernel.send(request).await?;
    // let response: VectorAddResponse = kernel.receive().await?;

    // Simulated response
    let response = VectorAddResponse {
        id: MessageId::generate(),
        result: vec![6.0, 8.0, 10.0, 12.0],
        correlation_id: request.correlation_id,
        timestamp: HlcTimestamp::now(1),
    };

    println!("\n  Response received:");
    println!("    Result: {:?}", response.result);
    println!("    Timestamp: {:?}", response.timestamp);
}

async fn demonstrate_correlation() {
    println!("Sending multiple correlated requests...\n");

    // Create requests with correlation IDs
    let requests: Vec<_> = (0..3)
        .map(|i| {
            let req = VectorAddRequest::new(
                vec![i as f32; 4],
                vec![(i + 1) as f32; 4],
            );
            println!(
                "  Request {}: correlation_id = {:?}",
                i, req.correlation_id
            );
            req
        })
        .collect();

    println!("\n  Correlation IDs allow matching responses to requests");
    println!("  even when responses arrive out of order.");

    // In production:
    // let pending: HashMap<CorrelationId, oneshot::Sender<Response>> = ...;
    // for req in requests {
    //     kernel.send(req).await?;
    // }
    // while let Some(resp) = kernel.receive().await? {
    //     if let Some(sender) = pending.remove(&resp.correlation_id) {
    //         sender.send(resp);
    //     }
    // }
}

async fn demonstrate_priorities() {
    println!("Messages can have different priorities:\n");

    let priorities = [
        (priority::LOW, "LOW (0-63)"),
        (priority::NORMAL, "NORMAL (64-127)"),
        (priority::HIGH, "HIGH (128-191)"),
        (priority::CRITICAL, "CRITICAL (192-255)"),
    ];

    for (prio, name) in priorities {
        println!("  Priority {}: {}", prio, name);
    }

    println!("\n  Higher priority messages are processed first.");
    println!("  Use CRITICAL sparingly - for time-sensitive operations.");

    // Example: High-priority matrix multiplication
    let urgent_request = MatrixMultiplyRequest::new(
        vec![1.0; 16],
        vec![1.0; 16],
        (4, 4, 4),
    )
    .with_high_priority();

    println!("\n  Created high-priority request: priority = {}", urgent_request.priority);
}

async fn demonstrate_batch_requests() {
    println!("Batching improves throughput for many small operations:\n");

    // Create a batch of requests
    let batch_size = 100;
    let requests: Vec<_> = (0..batch_size)
        .map(|i| VectorAddRequest::new(vec![i as f32], vec![1.0]))
        .collect();

    println!("  Created batch of {} requests", batch_size);

    // In production, send batch efficiently:
    // let futures: Vec<_> = requests.iter().map(|r| kernel.send(r)).collect();
    // join_all(futures).await;

    println!("  Batching reduces per-message overhead");
    println!("  Optimal batch size depends on message size and queue capacity");
}

async fn demonstrate_timeout_handling() {
    println!("Always use timeouts for receive operations:\n");

    let timeout = Duration::from_secs(5);
    println!("  Timeout: {:?}", timeout);

    // In production:
    // match tokio::time::timeout(timeout, kernel.receive()).await {
    //     Ok(Ok(response)) => println!("Got response: {:?}", response),
    //     Ok(Err(e)) => println!("Kernel error: {}", e),
    //     Err(_) => println!("Timeout - kernel did not respond"),
    // }

    println!("  Timeouts prevent deadlocks when kernels are slow or stuck");
    println!("  Consider retry logic for transient failures");
}
