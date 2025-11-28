# RingKernel Examples

This directory contains working examples demonstrating RingKernel's capabilities for GPU-native persistent actor programming.

## Overview

RingKernel is a framework for building high-performance, GPU-accelerated applications using a persistent actor model. Unlike traditional GPU programming where kernels are short-lived, RingKernel actors (kernels) persist on the GPU and communicate via lock-free message queues.

## Directory Structure

```
examples/
├── basic/                  # Getting started examples
│   ├── hello_kernel.rs         # Your first kernel
│   └── kernel_states.rs        # Kernel lifecycle management
├── messaging/              # Message passing patterns
│   ├── request_response.rs     # Request-response pattern
│   └── pub_sub.rs              # Topic-based pub/sub
├── web-api/                # Web service integration
│   └── axum_api.rs             # REST API with Axum
├── data-processing/        # Data pipeline examples
│   └── batch_processor.rs      # Batch processing pipeline
├── monitoring/             # Observability examples
│   └── telemetry.rs            # Real-time metrics & histograms
├── ecosystem/              # External integration examples
│   ├── grpc_server.rs          # gRPC server for distributed GPU
│   ├── config_management.rs    # Configuration loading & validation
│   └── ml_pipeline.rs          # ML inference with Candle
└── advanced/               # Advanced patterns
    └── multi_gpu.rs            # Multi-GPU coordination
```

## Quick Start

### Prerequisites

```bash
# Ensure Rust is installed
rustup --version

# Clone the repository
git clone https://github.com/mivertowski/RustCompute.git
cd RustCompute
```

### Running Examples

```bash
# Run a basic example
cargo run --example hello_kernel

# Run with specific features
cargo run --example axum_api --features "ringkernel-ecosystem/axum"

# Run with release optimizations
cargo run --release --example batch_processor
```

## Use Cases

### 1. Real-Time Data Processing

Process streaming data with GPU acceleration while maintaining state:

```rust
// Create a persistent kernel for stream processing
let runtime = RingKernelRuntime::builder()
    .backend(Backend::Cuda)
    .build()
    .await?;

// Launch a stateful processor
runtime.launch_kernel("stream_processor", config).await?;

// Feed data continuously - kernel maintains state between messages
for batch in data_stream {
    runtime.send("stream_processor", batch).await?;
}
```

### 2. GPU-Accelerated Microservices

Build web services that offload computation to GPU:

```rust
// Axum handler delegating to GPU kernel
async fn compute_embeddings(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, AppError> {
    state.runtime.send("embedding_kernel", request).await?;
    let response = state.runtime.receive("embedding_kernel", timeout).await?;
    Ok(Json(response))
}
```

### 3. Multi-GPU Workload Distribution

Distribute work across multiple GPUs with automatic load balancing:

```rust
let coordinator = MultiGpuBuilder::new()
    .load_balancing(LoadBalancingStrategy::LeastLoaded)
    .build();

// Register available GPUs
coordinator.register_device(DeviceInfo::new(0, "RTX 4090", Backend::Cuda));
coordinator.register_device(DeviceInfo::new(1, "RTX 4090", Backend::Cuda));

// Coordinator automatically selects optimal device
let device = coordinator.select_device(&launch_options)?;
```

### 4. Distributed GPU Computing

Expose GPU kernels over the network via gRPC:

```rust
let server = GrpcServerBuilder::new(runtime)
    .timeout(Duration::from_secs(30))
    .max_message_size(16 * 1024 * 1024)
    .build();

// Serve GPU compute over the network
Server::builder()
    .add_service(RingKernelServer::new(server))
    .serve("[::1]:50051".parse()?)
    .await?;
```

## Architecture Concepts

### Persistent Actors

Traditional GPU programming:
```
CPU: Launch kernel → GPU: Execute → CPU: Collect results → GPU: Terminate
     (repeat for each operation)
```

RingKernel model:
```
CPU: Launch kernel → GPU: Persist & Process messages continuously
     └── Send message → GPU processes → Return result (kernel stays alive)
     └── Send message → GPU processes → Return result
     └── ... (kernel maintains state)
```

### Message Queues

Each kernel has input/output ring buffers for zero-copy message passing:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Producer   │────▶│ Ring Buffer  │────▶│   Kernel    │
│   (Host)    │     │  (GPU mem)   │     │   (GPU)     │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    Lock-free SPSC
                    (no synchronization)
```

### Hybrid Logical Clocks

Messages are ordered using Hybrid Logical Clocks (HLC) for causal consistency:

```rust
let clock = HlcClock::new(node_id);
let timestamp = clock.now();  // Combines wall clock + logical counter

// Timestamps enable:
// - Causal ordering of messages
// - Distributed system coordination
// - Debugging and tracing
```

## Performance Considerations

1. **Message Size**: Keep messages small (<64KB) for best throughput
2. **Batch Operations**: Group related operations into single messages
3. **Queue Capacity**: Size queues based on expected message rate
4. **Backend Selection**: CUDA for NVIDIA, Metal for Apple, WebGPU for portability

## Ecosystem Integration Examples

The `ecosystem/` directory demonstrates integration with popular Rust crates:

### gRPC Server (`ecosystem/grpc_server.rs`)

Expose GPU kernels over the network for distributed computing:

```bash
cargo run --example grpc_server --features "ringkernel-ecosystem/grpc"
```

- Protocol buffer serialization
- Bidirectional streaming
- Health checks and metrics
- Load balancing ready

### Configuration Management (`ecosystem/config_management.rs`)

Flexible configuration from multiple sources:

```bash
cargo run --example config_management --features "ringkernel-ecosystem/config"
```

- TOML configuration files
- Environment variable overrides
- Validation and builder patterns
- Multi-environment setup (dev/staging/prod)

### ML Pipeline (`ecosystem/ml_pipeline.rs`)

Build ML inference pipelines with GPU acceleration:

```bash
cargo run --example ml_pipeline --features "ringkernel-ecosystem/candle"
```

- Image preprocessing
- Batch inference
- Custom GPU kernels for ML ops
- Performance benchmarking

## Advanced Examples

### Multi-GPU Coordination (`advanced/multi_gpu.rs`)

Scale across multiple GPUs for maximum performance:

```bash
cargo run --example multi_gpu
```

- GPU discovery and management
- Load balancing strategies
- Data and model parallelism
- Fault tolerance and recovery

## Next Steps

1. Start with `basic/hello_kernel.rs` to understand the fundamentals
2. Explore `messaging/` for communication patterns
3. Check `web-api/` for service integration
4. Review `monitoring/` for production observability
5. Dive into `ecosystem/` for external integrations
6. Check `advanced/` for multi-GPU and complex patterns

## Troubleshooting

### GPU Not Available

If CUDA/Metal is not detected, the runtime falls back to CPU:

```rust
let backend = if cuda_available() {
    Backend::Cuda
} else {
    Backend::Cpu
};
```

### Memory Issues

For large datasets, use streaming:

```rust
for chunk in data.chunks(1024) {
    runtime.send("processor", chunk).await?;
}
```

### Performance Tips

1. **Warm up the GPU**: Run a few iterations before benchmarking
2. **Batch operations**: Group small operations into larger batches
3. **Reuse buffers**: Pre-allocate and reuse memory when possible
4. **Profile first**: Use `nvprof` or Instruments before optimizing

## License

MIT OR Apache-2.0
