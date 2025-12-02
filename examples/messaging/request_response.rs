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
//! cargo run -p ringkernel --example request_response
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
#[allow(dead_code)]
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
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct MatrixMultiplyRequest {
    id: MessageId,
    /// Matrix A (row-major)
    a: Vec<f32>,
    /// Matrix B (row-major)
    b: Vec<f32>,
    /// Matrix dimensions (m, k, n)
    dims: (usize, usize, usize),
    priority_level: u8,
}

impl MatrixMultiplyRequest {
    fn new(a: Vec<f32>, b: Vec<f32>, dims: (usize, usize, usize)) -> Self {
        Self {
            id: MessageId::generate(),
            a,
            b,
            dims,
            priority_level: priority::NORMAL,
        }
    }

    fn with_high_priority(mut self) -> Self {
        self.priority_level = priority::HIGH;
        self
    }
}

/// Demonstrates CPU fallback computation for vector addition
fn compute_vector_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Demonstrates CPU fallback computation for matrix multiplication
fn compute_matrix_multiply(a: &[f32], b: &[f32], dims: (usize, usize, usize)) -> Vec<f32> {
    let (m, k, n) = dims;
    let mut c = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Request-Response Pattern Example ===\n");

    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await?;

    // Launch compute kernels
    let vector_kernel = runtime
        .launch(
            "vector_compute",
            LaunchOptions::default().without_auto_activate(),
        )
        .await?;

    let matrix_kernel = runtime
        .launch(
            "matrix_compute",
            LaunchOptions::default()
                .without_auto_activate()
                .with_priority(priority::HIGH),
        )
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

    // ====== Real Computation ======
    println!("\n=== Real CPU Computation (GPU Fallback) ===");
    demonstrate_real_computation().await;

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

    let request = VectorAddRequest::new(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]);

    println!("  Request ID: {}", request.id);
    println!("  Vector A: {:?}", request.a);
    println!("  Vector B: {:?}", request.b);

    // Simulate kernel processing with CPU fallback
    let result = compute_vector_add(&request.a, &request.b);

    let response = VectorAddResponse {
        id: MessageId::generate(),
        result,
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
            let req = VectorAddRequest::new(vec![i as f32; 4], vec![(i + 1) as f32; 4]);
            println!(
                "  Request {}: correlation_id = {:?}",
                i, req.correlation_id
            );
            req
        })
        .collect();

    println!("\n  Correlation IDs allow matching responses to requests");
    println!("  even when responses arrive out of order.");

    // Process all requests
    for (i, req) in requests.iter().enumerate() {
        let result = compute_vector_add(&req.a, &req.b);
        println!("  Response {}: result = {:?}", i, result);
    }
}

async fn demonstrate_priorities() {
    println!("Messages can have different priorities:\n");

    let priorities = [
        (priority::LOW, "LOW (0)"),
        (priority::NORMAL, "NORMAL (64)"),
        (priority::HIGH, "HIGH (128)"),
        (priority::CRITICAL, "CRITICAL (192)"),
    ];

    for (prio, name) in priorities {
        println!("  Priority {}: {}", prio, name);
    }

    println!("\n  Higher priority messages are processed first.");
    println!("  Use CRITICAL sparingly - for time-sensitive operations.");

    // Example: High-priority matrix multiplication
    let urgent_request =
        MatrixMultiplyRequest::new(vec![1.0; 16], vec![1.0; 16], (4, 4, 4)).with_high_priority();

    println!(
        "\n  Created high-priority request: priority = {}",
        urgent_request.priority_level
    );
}

async fn demonstrate_real_computation() {
    println!("Performing real computations on CPU (GPU fallback):\n");

    // Vector addition
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let start = std::time::Instant::now();
    let result = compute_vector_add(&a, &b);
    let elapsed = start.elapsed();

    println!("  Vector Addition:");
    println!("    A = {:?}", a);
    println!("    B = {:?}", b);
    println!("    A + B = {:?}", result);
    println!("    Time: {:?}\n", elapsed);

    // Matrix multiplication (2x3 * 3x2 = 2x2)
    let mat_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let mat_b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

    let start = std::time::Instant::now();
    let result = compute_matrix_multiply(&mat_a, &mat_b, (2, 3, 2));
    let elapsed = start.elapsed();

    println!("  Matrix Multiplication (2x3 * 3x2 = 2x2):");
    println!("    A = {:?}", mat_a);
    println!("    B = {:?}", mat_b);
    println!("    A * B = {:?}", result);
    println!("    Expected: [58, 64, 139, 154]");
    println!("    Time: {:?}", elapsed);
}

async fn demonstrate_timeout_handling() {
    println!("Always use timeouts for receive operations:\n");

    let timeout = Duration::from_secs(5);
    println!("  Timeout: {:?}", timeout);

    // Demonstrate timeout pattern
    println!("\n  Pattern:");
    println!("    match kernel.receive_timeout(timeout).await {{");
    println!("        Ok(response) => handle(response),");
    println!("        Err(RingKernelError::Timeout(_)) => retry_or_fail(),");
    println!("        Err(e) => handle_error(e),");
    println!("    }}");

    println!("\n  Timeouts prevent deadlocks when kernels are slow or stuck");
    println!("  Consider retry logic for transient failures");
}
