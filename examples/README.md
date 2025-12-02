# RingKernel Examples

This directory contains working examples demonstrating RingKernel's capabilities for GPU-native persistent actor programming.

## Overview

RingKernel is a framework for building high-performance, GPU-accelerated applications using a persistent actor model. Unlike traditional GPU programming where kernels are short-lived, RingKernel actors (kernels) persist on the GPU and communicate via lock-free message queues.

All examples in this directory are functional and can be run with the commands shown below.

## Directory Structure

```
examples/
├── basic/                  # Getting started examples
│   ├── hello_kernel.rs         # Runtime, lifecycle, suspend/resume
│   └── kernel_states.rs        # State machine, multi-kernel orchestration
├── messaging/              # Message passing patterns
│   ├── request_response.rs     # Correlation IDs, priorities, CPU fallback
│   └── pub_sub.rs              # Topic wildcards, QoS levels, IoT scenario
├── web-api/                # Web service integration
│   └── axum_api.rs             # REST API with vector/matrix operations
├── data-processing/        # Data pipeline examples
│   └── batch_processor.rs      # Preprocessing, GPU compute, statistics
├── monitoring/             # Observability examples
│   └── telemetry.rs            # Metrics, histograms, alerts, monitoring
├── ecosystem/              # External integration examples
│   ├── grpc_server.rs          # gRPC server with streaming
│   ├── config_management.rs    # TOML, env vars, validation
│   └── ml_pipeline.rs          # ML inference patterns, batching
└── advanced/               # Advanced patterns
    └── multi_gpu.rs            # Load balancing, fault tolerance
```

## Running Examples

All examples are wired to the `ringkernel` crate and can be run with:

```bash
cargo run -p ringkernel --example <example_name>
```

### Basic Examples

```bash
# Runtime creation, kernel lifecycle, suspend/resume
cargo run -p ringkernel --example basic_hello_kernel

# Kernel state machine with multi-kernel pipeline orchestration
cargo run -p ringkernel --example kernel_states
```

### Messaging Examples

```bash
# Request-response with correlation IDs and priorities
# Includes real CPU fallback computation (vector add, matrix multiply)
cargo run -p ringkernel --example request_response

# Topic-based pub/sub with wildcard matching
# Demonstrates QoS levels and IoT sensor network scenario
cargo run -p ringkernel --example pub_sub
```

### Web API Examples

```bash
# REST API with Axum framework
# Endpoints for vector addition and matrix multiplication
cargo run -p ringkernel --example axum_api
```

### Data Processing Examples

```bash
# Batch processing pipeline with preprocessing and statistics
# Shows ~1.2M rows/sec throughput on CPU fallback
cargo run -p ringkernel --example batch_processor
```

### Monitoring Examples

```bash
# Real-time telemetry collection
# Histograms, percentiles, alerts, 3-second monitoring loop
cargo run -p ringkernel --example telemetry
```

### Ecosystem Integration Examples

```bash
# gRPC server patterns for distributed GPU computing
cargo run -p ringkernel --example grpc_server

# Configuration management with TOML, env vars, validation
cargo run -p ringkernel --example config_management

# ML inference pipeline patterns and benchmarking
cargo run -p ringkernel --example ml_pipeline
```

### Advanced Examples

```bash
# Multi-GPU coordination with load balancing strategies
cargo run -p ringkernel --example multi_gpu
```

## Architecture Concepts

### Persistent Kernels

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

### Kernel Lifecycle

Kernels follow a state machine:

```
        ┌──────────┐
        │ Launched │
        └────┬─────┘
             │ activate()
             ▼
        ┌──────────┐
   ┌────│  Active  │────┐
   │    └────┬─────┘    │
   │         │          │ deactivate()
   │         │          ▼
   │         │    ┌────────────┐
   │         │    │Deactivated │
   │         │    └─────┬──────┘
   │         │          │
   │         │ terminate()
   │         ▼          │
   │    ┌──────────┐    │
   └───▶│Terminated│◀───┘
        └──────────┘
```

### Message Queues

Each kernel has input/output ring buffers for communication:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Producer   │────▶│ Ring Buffer  │────▶│   Kernel    │
│   (Host)    │     │ (lock-free)  │     │   (GPU)     │
└─────────────┘     └──────────────┘     └─────────────┘
```

Queue capacity must be a power of 2 (e.g., 1024, 2048, 4096).

### Hybrid Logical Clocks

Messages are timestamped with HLC for causal ordering:

```rust
let clock = HlcClock::new(node_id);
let timestamp = clock.tick();

// Timestamps enable:
// - Causal ordering of messages
// - Distributed system coordination
// - Request-response correlation
```

## Example Highlights

### basic_hello_kernel

Demonstrates:
- Runtime creation with backend selection
- Launching kernels with custom options
- Suspend/resume (deactivate/activate)
- Status queries and metrics
- Clean shutdown

### kernel_states

Demonstrates:
- Multiple kernels in different states simultaneously
- Pipeline orchestration (ingress → processor → egress)
- Graceful shutdown sequence

### request_response

Demonstrates:
- Type-safe message types
- Correlation IDs for tracking
- Priority levels (LOW, NORMAL, HIGH, CRITICAL)
- CPU fallback computation (real matrix multiplication)
- Timeout handling patterns

### pub_sub

Demonstrates:
- PubSubBroker creation and configuration
- Topic subscriptions with wildcards (`*`, `#`)
- QoS levels (AtMostOnce, AtLeastOnce, ExactlyOnce)
- Real-world IoT sensor network scenario

### batch_processor

Demonstrates:
- Data pipeline with preprocessing
- GPU compute simulation with CPU fallback
- Statistics collection (throughput, GPU utilization)
- Batch processing with configurable sizes

### telemetry

Demonstrates:
- TelemetryBuffer (64-byte GPU-friendly structure)
- MetricsCollector for per-kernel tracking
- Latency histograms with percentiles
- Alert subscription system
- Real-time monitoring loop

## Performance Notes

The examples demonstrate patterns rather than peak performance. For production:

1. **Queue Sizing**: Size queues based on expected message rate
2. **Batch Operations**: Group related operations into single messages
3. **Backend Selection**: Use CUDA for NVIDIA GPUs, CPU for development
4. **Profiling**: Use `nvprof` or similar tools before optimizing

Current examples use CPU fallback computation. With GPU execution:
- Vector operations: ~75M elements/sec (verified on RTX 2000 Ada)
- Memory transfers: 7.6 GB/s HtoD, 1.4 GB/s DtoH

## Troubleshooting

### GPU Not Detected

If CUDA is not available, RingKernel falls back to CPU:

```rust
let runtime = RingKernel::builder()
    .backend(Backend::Auto)  // Will use CPU if no GPU found
    .build()
    .await?;

println!("Backend: {:?}", runtime.backend());
```

### Queue Capacity Error

Queue capacity must be a power of 2:

```rust
// Wrong
let options = LaunchOptions::default().with_queue_capacity(1000);

// Correct
let options = LaunchOptions::default().with_queue_capacity(1024);

// Or calculate it
let capacity = desired_size.next_power_of_two();
```

### State Transition Errors

Don't activate an already active kernel:

```rust
// Use without_auto_activate if you want manual control
let options = LaunchOptions::default().without_auto_activate();
let kernel = runtime.launch("processor", options).await?;

// Now safe to call activate
kernel.activate().await?;
```

## License

MIT OR Apache-2.0
