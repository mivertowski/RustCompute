//! # Axum Web API Integration
//!
//! This example demonstrates building a REST API that delegates
//! compute-intensive operations to GPU Ring Kernels.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
//! │   Client    │─────▶│  Axum API   │─────▶│   RingKernel│
//! │  (HTTP)     │◀─────│  (CPU)      │◀─────│   (GPU)     │
//! └─────────────┘      └─────────────┘      └─────────────┘
//!
//! POST /api/compute/vector-add
//!   → Parse JSON request
//!   → Send to GPU kernel
//!   → Wait for result
//!   → Return JSON response
//! ```
//!
//! ## Endpoints
//!
//! - `POST /api/compute/vector-add` - Vector addition
//! - `POST /api/compute/matrix-multiply` - Matrix multiplication
//! - `GET /api/health` - Health check
//! - `GET /api/metrics` - Prometheus metrics
//!
//! ## Run this example:
//! ```bash
//! cargo run --example axum_api --features "ringkernel-ecosystem/axum,ringkernel-ecosystem/prometheus"
//! ```

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use ringkernel::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;

// ============ Request/Response Types ============

#[derive(Debug, Deserialize)]
struct VectorAddRequest {
    a: Vec<f32>,
    b: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct VectorAddResponse {
    result: Vec<f32>,
    latency_ms: f64,
}

#[derive(Debug, Deserialize)]
struct MatrixMultiplyRequest {
    a: Vec<f32>,
    b: Vec<f32>,
    m: usize,
    k: usize,
    n: usize,
}

#[derive(Debug, Serialize)]
struct MatrixMultiplyResponse {
    result: Vec<f32>,
    dimensions: (usize, usize),
    latency_ms: f64,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    backend: String,
    active_kernels: usize,
    uptime_secs: f64,
}

#[derive(Debug, Serialize)]
struct MetricsResponse {
    requests_total: u64,
    requests_success: u64,
    requests_failed: u64,
    avg_latency_ms: f64,
}

// ============ Application State ============

struct AppState {
    runtime: Arc<RingKernel>,
    vector_kernel: Arc<KernelHandle>,
    matrix_kernel: Arc<KernelHandle>,
    stats: RequestStats,
    start_time: Instant,
}

#[derive(Default)]
struct RequestStats {
    total: AtomicU64,
    success: AtomicU64,
    failed: AtomicU64,
    total_latency_us: AtomicU64,
}

impl RequestStats {
    fn record_success(&self, latency: Duration) {
        self.total.fetch_add(1, Ordering::Relaxed);
        self.success.fetch_add(1, Ordering::Relaxed);
        self.total_latency_us
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
    }

    fn record_failure(&self) {
        self.total.fetch_add(1, Ordering::Relaxed);
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    fn avg_latency_ms(&self) -> f64 {
        let total = self.success.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let total_us = self.total_latency_us.load(Ordering::Relaxed);
        (total_us as f64 / total as f64) / 1000.0
    }
}

// ============ Error Handling ============

enum AppError {
    KernelError(String),
    ValidationError(String),
    Timeout,
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            AppError::KernelError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e),
            AppError::ValidationError(e) => (StatusCode::BAD_REQUEST, e),
            AppError::Timeout => (StatusCode::GATEWAY_TIMEOUT, "Request timed out".to_string()),
        };

        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

// ============ Handlers ============

async fn vector_add_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<VectorAddRequest>,
) -> Result<Json<VectorAddResponse>, AppError> {
    let start = Instant::now();

    // Validate input
    if request.a.len() != request.b.len() {
        state.stats.record_failure();
        return Err(AppError::ValidationError(
            "Vectors must have the same length".to_string(),
        ));
    }

    if request.a.is_empty() {
        state.stats.record_failure();
        return Err(AppError::ValidationError(
            "Vectors cannot be empty".to_string(),
        ));
    }

    // In a real implementation:
    // state.vector_kernel.send(KernelRequest::VectorAdd { a, b }).await?;
    // let result = state.vector_kernel.receive(timeout).await?;

    // Simulated GPU computation
    let result: Vec<f32> = request
        .a
        .iter()
        .zip(request.b.iter())
        .map(|(a, b)| a + b)
        .collect();

    let latency = start.elapsed();
    state.stats.record_success(latency);

    Ok(Json(VectorAddResponse {
        result,
        latency_ms: latency.as_secs_f64() * 1000.0,
    }))
}

async fn matrix_multiply_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MatrixMultiplyRequest>,
) -> Result<Json<MatrixMultiplyResponse>, AppError> {
    let start = Instant::now();

    // Validate dimensions
    if request.a.len() != request.m * request.k {
        state.stats.record_failure();
        return Err(AppError::ValidationError(format!(
            "Matrix A size {} doesn't match dimensions {}x{}",
            request.a.len(),
            request.m,
            request.k
        )));
    }

    if request.b.len() != request.k * request.n {
        state.stats.record_failure();
        return Err(AppError::ValidationError(format!(
            "Matrix B size {} doesn't match dimensions {}x{}",
            request.b.len(),
            request.k,
            request.n
        )));
    }

    // Simulated GPU matrix multiplication
    let mut result = vec![0.0f32; request.m * request.n];
    for i in 0..request.m {
        for j in 0..request.n {
            for p in 0..request.k {
                result[i * request.n + j] +=
                    request.a[i * request.k + p] * request.b[p * request.n + j];
            }
        }
    }

    let latency = start.elapsed();
    state.stats.record_success(latency);

    Ok(Json(MatrixMultiplyResponse {
        result,
        dimensions: (request.m, request.n),
        latency_ms: latency.as_secs_f64() * 1000.0,
    }))
}

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy",
        backend: format!("{:?}", state.runtime.backend()),
        active_kernels: state.runtime.list_kernels().len(),
        uptime_secs: state.start_time.elapsed().as_secs_f64(),
    })
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> Json<MetricsResponse> {
    Json(MetricsResponse {
        requests_total: state.stats.total.load(Ordering::Relaxed),
        requests_success: state.stats.success.load(Ordering::Relaxed),
        requests_failed: state.stats.failed.load(Ordering::Relaxed),
        avg_latency_ms: state.stats.avg_latency_ms(),
    })
}

// ============ Main ============

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== RingKernel Axum API Example ===\n");

    // Initialize RingKernel runtime
    println!("Initializing RingKernel runtime...");
    let runtime = Arc::new(
        RingKernel::builder()
            .backend(Backend::Cpu)
            .build()
            .await?,
    );

    // Launch compute kernels
    println!("Launching compute kernels...");
    let vector_kernel = Arc::new(
        runtime
            .launch("vector_compute", LaunchOptions::default())
            .await?,
    );
    let matrix_kernel = Arc::new(
        runtime
            .launch("matrix_compute", LaunchOptions::default())
            .await?,
    );

    vector_kernel.activate().await?;
    matrix_kernel.activate().await?;

    // Create application state
    let state = Arc::new(AppState {
        runtime,
        vector_kernel,
        matrix_kernel,
        stats: RequestStats::default(),
        start_time: Instant::now(),
    });

    // Build router
    let app = Router::new()
        .route("/api/compute/vector-add", post(vector_add_handler))
        .route("/api/compute/matrix-multiply", post(matrix_multiply_handler))
        .route("/api/health", get(health_handler))
        .route("/api/metrics", get(metrics_handler))
        .with_state(state);

    // Start server
    let addr = "0.0.0.0:3000";
    println!("\nServer starting on http://{}", addr);
    println!("\nAvailable endpoints:");
    println!("  POST /api/compute/vector-add");
    println!("  POST /api/compute/matrix-multiply");
    println!("  GET  /api/health");
    println!("  GET  /api/metrics");

    println!("\nExample requests:");
    println!(r#"
  curl -X POST http://localhost:3000/api/compute/vector-add \
    -H "Content-Type: application/json" \
    -d '{{"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}}'

  curl http://localhost:3000/api/health
"#);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
