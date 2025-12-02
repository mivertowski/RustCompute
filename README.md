# RingKernel (RustCompute)

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

**GPU-native persistent actor model framework for Rust** - A port of DotCompute's Ring Kernel system.

RingKernel enables GPU-accelerated actor systems with persistent kernels, lock-free message passing, and hybrid logical clocks for causal ordering across distributed GPU computations.

## Current Status

RingKernel is under active development. The core runtime, CPU backend, and CUDA backend are functional. The framework includes working examples demonstrating real-world usage patterns.

**What works today:**
- Runtime creation and kernel lifecycle management
- CPU backend (fully functional for development and testing)
- CUDA backend (verified GPU execution with real PTX kernels)
- Message passing infrastructure (queues, serialization, HLC timestamps)
- Pub/Sub messaging with topic wildcards
- Telemetry and metrics collection
- 11 working examples covering common use cases

**In progress:**
- Metal and WebGPU backends (scaffolded, not yet functional)
- Proc macro code generation for custom kernels
- K2K (kernel-to-kernel) direct messaging

## Features

- **Persistent GPU-resident state** across kernel invocations
- **Lock-free message passing** between host and kernels
- **Hybrid Logical Clocks (HLC)** for temporal ordering and causality
- **Multiple GPU backends**: CUDA (working), Metal, WebGPU, CPU (fallback)
- **Zero-copy serialization** via rkyv
- **Type-safe Rust** with async/await support

## Quick Start

Add RingKernel to your `Cargo.toml`:

```toml
[dependencies]
ringkernel = "0.1"
tokio = { version = "1", features = ["full"] }
```

Basic usage:

```rust
use ringkernel::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create runtime with auto-detected backend
    let runtime = RingKernel::builder()
        .backend(Backend::Auto)
        .build()
        .await?;

    // Launch a kernel (auto-activates by default)
    let kernel = runtime.launch("my_kernel", LaunchOptions::default()).await?;
    println!("Kernel state: {:?}", kernel.state());

    // Check if active
    if kernel.is_active() {
        println!("Kernel is processing");
    }

    // Suspend and resume
    kernel.suspend().await?;
    kernel.resume().await?;

    // Terminate when done
    kernel.terminate().await?;
    runtime.shutdown().await?;

    Ok(())
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Host (CPU)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Application │──│   Runtime   │──│   Message Bridge    │  │
│  │  (async)    │  │  (tokio)    │  │  (DMA transfers)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │ PCIe / Unified Memory
┌────────────────────────────┴────────────────────────────────┐
│                    Device (GPU)                             │
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │ Control Block │  │   Input Queue   │  │ Output Queue  │  │
│  │   (128 B)     │  │  (lock-free)    │  │ (lock-free)   │  │
│  └───────────────┘  └─────────────────┘  └───────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Persistent Kernel                          ││
│  │     (maintains state between messages)                  ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Backends

| Backend | Platform | Status | Notes |
|---------|----------|--------|-------|
| **CPU** | All | **Working** | Full functionality, ideal for development |
| **CUDA** | Linux, Windows | **Working** | Verified GPU execution, requires CUDA toolkit |
| **Metal** | macOS, iOS | Scaffolded | API defined, implementation pending |
| **WebGPU** | Cross-platform | Scaffolded | API defined, implementation pending |

Enable backends via Cargo features:

```toml
[dependencies]
ringkernel = { version = "0.1", features = ["cuda"] }
```

## Examples

RingKernel includes 11 working examples demonstrating real-world patterns:

### Basic Examples
```bash
# Runtime and kernel lifecycle
cargo run -p ringkernel --example basic_hello_kernel

# Kernel state machine and orchestration
cargo run -p ringkernel --example kernel_states
```

### Messaging Examples
```bash
# Request-response with correlation IDs
cargo run -p ringkernel --example request_response

# Topic-based pub/sub with wildcards
cargo run -p ringkernel --example pub_sub
```

### Integration Examples
```bash
# REST API with Axum
cargo run -p ringkernel --example axum_api

# gRPC server patterns
cargo run -p ringkernel --example grpc_server

# Configuration management (TOML, env vars)
cargo run -p ringkernel --example config_management
```

### Data Processing
```bash
# Batch processing pipeline
cargo run -p ringkernel --example batch_processor
```

### Monitoring
```bash
# Telemetry and metrics collection
cargo run -p ringkernel --example telemetry
```

### Advanced
```bash
# ML inference pipeline patterns
cargo run -p ringkernel --example ml_pipeline

# Multi-GPU coordination
cargo run -p ringkernel --example multi_gpu
```

## Core Concepts

### Kernel Lifecycle

Kernels follow a state machine:

```
Launched → Active ⇄ Deactivated → Terminated
```

```rust
let kernel = runtime.launch("processor",
    LaunchOptions::default().without_auto_activate()
).await?;

kernel.activate().await?;   // Start processing
kernel.suspend().await?;    // Pause (alias for deactivate)
kernel.resume().await?;     // Resume (alias for activate)
kernel.terminate().await?;  // Clean shutdown
```

### Launch Options

Configure kernel behavior at launch:

```rust
let options = LaunchOptions::default()
    .with_queue_capacity(4096)      // Input/output queue size (must be power of 2)
    .with_grid_size(4)              // Number of blocks
    .with_block_size(256)           // Threads per block
    .with_shared_memory(16384)      // Shared memory bytes
    .with_priority(priority::HIGH)  // Scheduling hint
    .without_auto_activate();       // Don't activate immediately
```

### Hybrid Logical Clocks

HLC provides causal ordering across distributed operations:

```rust
let clock = HlcClock::new(1); // Node ID = 1

let ts1 = clock.tick();
let ts2 = clock.tick();
assert!(ts1 < ts2); // Total ordering guaranteed

// Update from received message
let remote_ts = HlcTimestamp::new(100, 5, 2);
let updated = clock.update(&remote_ts)?;
assert!(updated > remote_ts); // Causality preserved
```

### Pub/Sub Messaging

Topic-based messaging with wildcard support:

```rust
let broker = PubSubBroker::new(PubSubConfig::default());

// Subscribe with wildcards
let sub = broker.subscribe(kernel_id, Topic::new("sensors/#"));

// Publish
broker.publish(
    Topic::new("sensors/temperature"),
    publisher_id,
    envelope,
    timestamp,
)?;
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `ringkernel` | Main facade crate (re-exports everything) |
| `ringkernel-core` | Core traits, types, and runtime |
| `ringkernel-derive` | Proc macros (in development) |
| `ringkernel-cpu` | CPU backend |
| `ringkernel-cuda` | NVIDIA CUDA backend |
| `ringkernel-metal` | Apple Metal backend (scaffolded) |
| `ringkernel-wgpu` | WebGPU backend (scaffolded) |
| `ringkernel-codegen` | GPU kernel code generation |
| `ringkernel-ecosystem` | Integration utilities |
| `ringkernel-audio-fft` | Example: GPU-accelerated audio processing |

## Running Tests

```bash
# Core tests (no GPU required)
cargo test --workspace --exclude ringkernel-cuda

# With CUDA backend
cargo test --features cuda -p ringkernel-cuda

# GPU execution verification (requires NVIDIA GPU)
cargo test --features cuda -p ringkernel-cuda --test gpu_execution_verify

# Benchmarks
cargo bench --package ringkernel
```

## Documentation

- [Architecture Overview](docs/01-architecture-overview.md)
- [Crate Structure](docs/02-crate-structure.md)
- [Core Abstractions](docs/03-core-abstractions.md)
- [Memory Management](docs/04-memory-management.md)
- [GPU Backends](docs/05-gpu-backends.md)
- [Serialization](docs/06-serialization.md)
- [Proc Macros](docs/07-proc-macros.md)
- [Runtime](docs/08-runtime.md)
- [Testing](docs/09-testing.md)
- [Performance](docs/10-performance.md)
- [Ecosystem Integration](docs/11-ecosystem.md)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Areas where help is needed:

- Metal backend implementation
- WebGPU backend implementation
- Additional examples and documentation
- Performance optimization
- Testing on different hardware

## Acknowledgments

This project is a Rust port of [DotCompute](https://github.com/dotcompute) Ring Kernel system. Thanks to the DotCompute team for the original design and architecture.
