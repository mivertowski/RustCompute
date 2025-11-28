# RingKernel Ecosystem Integration Guide

This guide provides comprehensive documentation for integrating RingKernel with the broader Rust ecosystem. Each integration is optional and can be enabled via feature flags.

## Table of Contents

1. [Overview](#overview)
2. [Feature Flags](#feature-flags)
3. [Web Frameworks](#web-frameworks)
   - [Axum Integration](#axum-integration)
   - [Actix Integration](#actix-integration)
   - [Tower Integration](#tower-integration)
4. [Data Processing](#data-processing)
   - [Apache Arrow](#apache-arrow)
   - [Polars DataFrames](#polars-dataframes)
5. [Machine Learning](#machine-learning)
   - [Candle Integration](#candle-integration)
6. [Distributed Computing](#distributed-computing)
   - [gRPC with Tonic](#grpc-with-tonic)
7. [Configuration & Observability](#configuration--observability)
   - [Configuration Management](#configuration-management)
   - [Tracing Integration](#tracing-integration)
   - [Prometheus Metrics](#prometheus-metrics)
8. [Best Practices](#best-practices)

---

## Overview

The `ringkernel-ecosystem` crate provides optional integrations with popular Rust crates, enabling you to use RingKernel in various contexts:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RingKernel Ecosystem                             │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │    Web      │  │    Data     │  │     ML      │                 │
│  │  Axum       │  │   Arrow     │  │   Candle    │                 │
│  │  Actix      │  │   Polars    │  │             │                 │
│  │  Tower      │  │             │  │             │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│         │                │                │                         │
│         └────────────────┼────────────────┘                         │
│                          ▼                                          │
│              ┌─────────────────────┐                                │
│              │   RingKernel Core   │                                │
│              │   (GPU Runtime)     │                                │
│              └─────────────────────┘                                │
│                          │                                          │
│  ┌───────────────────────┼───────────────────────┐                 │
│  │                       ▼                       │                 │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐    │                 │
│  │  │   gRPC   │  │  Config  │  │ Tracing  │    │                 │
│  │  │  Tonic   │  │   TOML   │  │Prometheus│    │                 │
│  │  └──────────┘  └──────────┘  └──────────┘    │                 │
│  └───────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Feature Flags

Add the integration crate to your `Cargo.toml`:

```toml
[dependencies]
ringkernel = "0.1"
ringkernel-ecosystem = { version = "0.1", features = ["axum", "prometheus"] }
```

Available features:

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `actix` | Actix Web/Actor integration | actix, actix-rt |
| `tower` | Tower Service trait impl | tower |
| `axum` | Axum web framework | axum, tokio |
| `arrow` | Apache Arrow support | arrow |
| `polars` | Polars DataFrame ops | polars |
| `grpc` | gRPC server/client | tonic, prost |
| `candle` | Candle ML framework | candle-core |
| `config` | Configuration loading | config, toml |
| `tracing-integration` | Tracing support | tracing |
| `prometheus` | Prometheus metrics | prometheus |
| `full` | All features | All of above |

---

## Web Frameworks

### Axum Integration

Build high-performance REST APIs that delegate compute to GPU kernels.

#### Basic Setup

```rust
use axum::{Router, routing::post, extract::State};
use ringkernel_ecosystem::axum::{AppState, ApiRequest, health_handler, metrics_handler};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize RingKernel runtime
    let runtime = Arc::new(RingKernel::builder()
        .backend(Backend::Cuda)
        .build()
        .await?);

    // Create application state
    let state = Arc::new(AppState::new(runtime));

    // Build router with GPU-accelerated endpoints
    let app = Router::new()
        .route("/api/compute", post(compute_handler))
        .route("/api/health", get(health_handler))
        .route("/api/metrics", get(metrics_handler))
        .with_state(state);

    axum::serve(listener, app).await?;
    Ok(())
}

async fn compute_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ComputeRequest>,
) -> Result<Json<ComputeResponse>, AppError> {
    // Send to GPU kernel
    let result = state.runtime.process("compute_kernel", request.data).await?;
    Ok(Json(ComputeResponse { result }))
}
```

#### Error Handling

```rust
use ringkernel_ecosystem::axum::AppError;

enum AppError {
    KernelError(String),
    ValidationError(String),
    Timeout,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::KernelError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e),
            AppError::ValidationError(e) => (StatusCode::BAD_REQUEST, e),
            AppError::Timeout => (StatusCode::GATEWAY_TIMEOUT, "Timeout".into()),
        };
        (status, Json(json!({ "error": message }))).into_response()
    }
}
```

### Actix Integration

Bridge Actix actors with GPU kernels.

```rust
use ringkernel_ecosystem::actix::{GpuActorBridge, GpuActorConfig};

// Create bridge between Actix and RingKernel
let bridge = GpuActorBridge::new(runtime, GpuActorConfig::default());

// Use in Actix actor
impl Actor for ComputeActor {
    type Context = Context<Self>;
}

impl Handler<ComputeRequest> for ComputeActor {
    type Result = ResponseFuture<ComputeResponse>;

    fn handle(&mut self, msg: ComputeRequest, _: &mut Context<Self>) -> Self::Result {
        let bridge = self.bridge.clone();
        Box::pin(async move {
            bridge.process("kernel_id", msg.data).await
        })
    }
}
```

### Tower Integration

Use RingKernel as a Tower service for composable middleware.

```rust
use ringkernel_ecosystem::tower::{RingKernelService, ServiceBuilder};
use tower::{Service, ServiceExt};

// Create Tower service
let service = ServiceBuilder::new(runtime)
    .timeout(Duration::from_secs(30))
    .build();

// Use with Tower middleware
let service = ServiceBuilder::new()
    .rate_limit(100, Duration::from_secs(1))
    .service(service);

// Process request
let response = service.ready().await?.call(request).await?;
```

---

## Data Processing

### Apache Arrow

Process Arrow arrays directly on GPU for columnar data analytics.

```rust
use ringkernel_ecosystem::arrow::{ArrowKernelOps, ArrowPipelineBuilder};
use arrow::array::{Float32Array, ArrayRef};

// Process Arrow arrays on GPU
let array = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
let result = runtime.process_arrow(&array, "square_kernel").await?;

// Build processing pipeline
let pipeline = ArrowPipelineBuilder::new(runtime)
    .operation("normalize")
    .operation("transform")
    .operation("aggregate")
    .build();

let result = pipeline.execute(record_batch).await?;
```

#### Chunked Processing for Large Datasets

```rust
use ringkernel_ecosystem::arrow::chunk_array;

// Process large arrays in chunks
let chunks = chunk_array(&large_array, 10000);
let results = Vec::new();

for chunk in chunks {
    let result = runtime.process_arrow(&chunk, "kernel").await?;
    results.push(result);
}
```

### Polars DataFrames

GPU-accelerate Polars DataFrame operations.

```rust
use ringkernel_ecosystem::polars::{GpuOps, DataFrameGpuOps};
use polars::prelude::*;

// GPU operations on Series
let series = Series::new("values", &[1.0f32, 2.0, 3.0]);
let result = series.gpu_add(&other_series, &runtime, "add_kernel").await?;

// DataFrame aggregation on GPU
let aggregator = GpuAggregator::new(runtime);
let stats = aggregator.compute_stats(&df, &["col1", "col2"]).await?;

println!("Mean: {}", stats.mean);
println!("Std:  {}", stats.std);
```

---

## Machine Learning

### Candle Integration

Bridge Candle tensors with GPU Ring Kernels for custom ML operations.

```rust
use ringkernel_ecosystem::candle::{TensorOps, gpu_matmul, CandlePipelineBuilder};
use candle_core::{Tensor, Device};

// GPU tensor operations
let a = Tensor::randn(0f32, 1.0, (1024, 1024), &Device::Cpu)?;
let b = Tensor::randn(0f32, 1.0, (1024, 1024), &Device::Cpu)?;

// Matrix multiplication on GPU via RingKernel
let result = gpu_matmul(&a, &b, &runtime, "matmul_kernel", &config).await?;

// Custom operation pipeline
let pipeline = CandlePipelineBuilder::new(runtime)
    .operation("normalize")
    .operation("attention")
    .operation("projection")
    .build();

let output = pipeline.execute(input_tensor).await?;
```

#### Custom ML Kernels

```rust
// Register custom CUDA kernel for ML operation
runtime.register_kernel("fused_attention", |input| {
    // Custom attention implementation
    // More efficient than separate Q, K, V operations
}).await?;

// Use in inference
let output = tensor.gpu_apply(&runtime, "fused_attention", &config).await?;
```

---

## Distributed Computing

### gRPC with Tonic

Expose GPU kernels over the network for distributed computing.

#### Server

```rust
use ringkernel_ecosystem::grpc::{RingKernelGrpcServer, GrpcConfig, GrpcServerBuilder};

// Build gRPC server
let server = GrpcServerBuilder::new(runtime)
    .timeout(Duration::from_secs(30))
    .max_message_size(16 * 1024 * 1024)
    .enable_reflection(true)
    .enable_health_check(true)
    .build();

// Start serving
tonic::transport::Server::builder()
    .add_service(RingKernelServer::new(server))
    .serve("[::1]:50051".parse()?)
    .await?;
```

#### Client

```rust
use ringkernel_ecosystem::grpc::{RingKernelGrpcClient, ProcessRequest};

let client = RingKernelGrpcClient::new("http://[::1]:50051");

let request = ProcessRequest {
    kernel_id: "compute".to_string(),
    payload: data,
    timeout_ms: Some(30000),
    request_id: Some("req-123".to_string()),
};

let response = client.process(request).await?;
```

#### Proto Definition

```protobuf
syntax = "proto3";
package ringkernel;

service RingKernel {
    rpc Process (ProcessRequest) returns (ProcessResponse);
    rpc GetKernelInfo (KernelInfoRequest) returns (KernelInfoResponse);
    rpc StreamProcess (stream ProcessRequest) returns (stream ProcessResponse);
}
```

---

## Configuration & Observability

### Configuration Management

Load configuration from TOML files and environment variables.

```rust
use ringkernel_ecosystem::config::{RingKernelConfig, ConfigBuilder, load_config};

// Load from file
let config = load_config("config/app.toml")?;

// With environment overrides (RINGKERNEL_BACKEND=cuda)
let config = RingKernelConfig::from_env()?;

// Programmatic configuration
let config = ConfigBuilder::new()
    .backend("cuda")
    .device_id(0)
    .queue_capacity(4096)
    .kernel_timeout(Duration::from_secs(60))
    .memory_pool_size(1024 * 1024 * 1024)
    .build()?;

// Validate configuration
config.validate()?;
```

#### Example TOML Configuration

```toml
backend = "cuda"
device_id = 0
queue_capacity = 2048
telemetry_enabled = true

[kernel]
default_block_size = 256
max_kernels = 64
timeout_ms = 30000

[memory]
pool_size = 268435456  # 256MB
enable_pooling = true

[logging]
level = "info"
format = "json"
```

### Tracing Integration

Automatic instrumentation with the `tracing` crate.

```rust
use ringkernel_ecosystem::tracing_ext::{SpanManager, TracingConfig, Sampler};

// Configure tracing
let config = TracingConfig {
    service_name: "gpu-service".to_string(),
    sample_rate: 0.1, // 10% sampling for high-frequency events
    ..Default::default()
};

let span_manager = SpanManager::new(config);

// Instrument kernel operations
let span = span_manager.create_span("kernel.process", &[
    ("kernel_id", "compute"),
    ("batch_size", "32"),
]);

let _guard = span.enter();
runtime.process("compute", data).await?;
```

#### Sampler for High-Frequency Events

```rust
// Don't trace every single message in high-throughput systems
let sampler = Sampler::new(100); // 1 in 100

if sampler.should_sample() {
    tracing::info!("Processing message");
}
```

### Prometheus Metrics

Export metrics in Prometheus format.

```rust
use ringkernel_ecosystem::metrics::{MetricsRegistry, Counter, Histogram, Gauge};

// Create registry
let registry = MetricsRegistry::new();

// Register metrics
registry.register_kernel("compute_kernel");

// Record metrics
registry.record_success("compute_kernel", Duration::from_micros(500));
registry.record_failure("compute_kernel");

// Export for Prometheus scraping
let metrics_text = registry.render();
```

#### Axum Metrics Endpoint

```rust
async fn metrics_handler(State(state): State<Arc<AppState>>) -> String {
    state.metrics_registry.render()
}

let app = Router::new()
    .route("/metrics", get(metrics_handler));
```

#### Grafana Dashboard

```json
{
  "panels": [
    {
      "title": "Request Rate",
      "targets": [{
        "expr": "rate(ringkernel_requests_total[5m])"
      }]
    },
    {
      "title": "Latency P99",
      "targets": [{
        "expr": "histogram_quantile(0.99, ringkernel_latency_bucket)"
      }]
    }
  ]
}
```

---

## Best Practices

### 1. Feature Selection

Only enable features you need to minimize compile time and binary size:

```toml
# Good: Specific features
ringkernel-ecosystem = { version = "0.1", features = ["axum", "prometheus"] }

# Avoid in production: All features
ringkernel-ecosystem = { version = "0.1", features = ["full"] }
```

### 2. Error Handling

Always handle ecosystem-specific errors:

```rust
use ringkernel_ecosystem::error::{EcosystemError, Result};

match runtime.process_arrow(&array, "kernel").await {
    Ok(result) => handle_result(result),
    Err(EcosystemError::DataConversion(msg)) => log::warn!("Conversion failed: {}", msg),
    Err(EcosystemError::Timeout) => retry_operation(),
    Err(e) => return Err(e.into()),
}
```

### 3. Resource Management

Use appropriate timeouts and pool sizes:

```rust
// Configure based on workload
let config = GrpcConfig {
    timeout: Duration::from_secs(30),
    max_message_size: 16 * 1024 * 1024,
    max_concurrent_streams: 200,
    ..Default::default()
};
```

### 4. Monitoring in Production

Always enable metrics and health checks:

```rust
// Health endpoint
router.route("/health", get(health_handler));

// Metrics endpoint
router.route("/metrics", get(metrics_handler));

// Kubernetes probes
// livenessProbe: /health
// readinessProbe: /health
```

### 5. Configuration Layering

Use environment-specific configuration:

```bash
# Development
RINGKERNEL_BACKEND=cpu cargo run

# Production
RINGKERNEL_BACKEND=cuda \
RINGKERNEL_MEMORY__POOL_SIZE=1073741824 \
cargo run --release
```

---

## Conclusion

The RingKernel ecosystem integrations enable seamless use of GPU acceleration in various contexts. Choose the integrations that fit your use case and follow the best practices for production deployments.

For more examples, see the `examples/ecosystem/` directory.
