# RingKernel

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://mivertowski.github.io/RustCompute/)

A GPU-native persistent actor model framework for Rust.

RingKernel provides infrastructure for building GPU-accelerated actor systems with persistent kernels, lock-free message passing, and hybrid logical clocks for causal ordering.

## Overview

RingKernel treats GPU compute units as long-running actors that maintain state between invocations. Instead of launching kernels per-operation, kernels persist on the GPU, communicate via lock-free queues, and process messages continuously.

**Key capabilities:**

- Persistent GPU-resident kernels that maintain state across invocations
- Lock-free message queues for host↔GPU and kernel-to-kernel communication
- Hybrid Logical Clocks (HLC) for causal ordering across distributed operations
- Multiple backend support: CPU, CUDA, WebGPU
- Zero-copy serialization via rkyv
- Proc macros for message and kernel definitions

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ringkernel = "0.1"
tokio = { version = "1", features = ["full"] }
```

For GPU backends:

```toml
# NVIDIA CUDA
ringkernel = { version = "0.1", features = ["cuda"] }

# WebGPU (cross-platform)
ringkernel = { version = "0.1", features = ["wgpu"] }
```

## Quick Start

```rust
use ringkernel::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create runtime (uses CPU backend by default)
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await?;

    // Launch a kernel
    let kernel = runtime.launch("processor", LaunchOptions::default()).await?;

    // Kernel is active and ready to process messages
    println!("State: {:?}", kernel.state());

    // Lifecycle management
    kernel.deactivate().await?;
    kernel.activate().await?;
    kernel.terminate().await?;

    runtime.shutdown().await?;
    Ok(())
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Host (CPU)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Application │──│   Runtime   │──│   Message Bridge    │  │
│  │   (async)   │  │   (tokio)   │  │   (DMA transfers)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │ PCIe / Unified Memory
┌────────────────────────────┴────────────────────────────────┐
│                      Device (GPU)                           │
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │ Control Block │  │   Input Queue   │  │ Output Queue  │  │
│  │    (128 B)    │  │  (lock-free)    │  │ (lock-free)   │  │
│  └───────────────┘  └─────────────────┘  └───────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  Persistent Kernel                      ││
│  │        (maintains state between messages)               ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Backends

| Backend | Status | Platforms | Requirements |
|---------|--------|-----------|--------------|
| CPU | Stable | All | None |
| CUDA | Stable | Linux, Windows | NVIDIA GPU, CUDA 12.x |
| WebGPU | Stable | All | Vulkan/Metal/DX12 capable GPU |
| Metal | Planned | macOS, iOS | — |

## Core Concepts

### Kernel Lifecycle

Kernels follow a state machine: `Created → Launched → Active ⇄ Deactivated → Terminated`

```rust
// Manual lifecycle control
let kernel = runtime.launch("worker",
    LaunchOptions::default().without_auto_activate()
).await?;

kernel.activate().await?;    // Start processing
kernel.deactivate().await?;  // Pause
kernel.activate().await?;    // Resume
kernel.terminate().await?;   // Clean shutdown
```

### Launch Options

```rust
let options = LaunchOptions::default()
    .with_queue_capacity(4096)      // Must be power of 2
    .with_block_size(256)           // Threads per block
    .with_priority(priority::HIGH)
    .without_auto_activate();
```

### Kernel-to-Kernel Messaging

Direct communication between kernels:

```rust
// K2K is enabled by default
let runtime = CpuRuntime::new().await?;
let broker = runtime.k2k_broker().unwrap();

// Kernels can send messages directly to each other
let receipt = broker.send(source_id, dest_id, envelope).await?;
```

### Hybrid Logical Clocks

HLC provides causal ordering across distributed operations:

```rust
let clock = HlcClock::new(node_id);

let ts1 = clock.tick();
let ts2 = clock.tick();
assert!(ts1 < ts2);

// Synchronize with remote timestamp
let remote_ts = receive_timestamp();
let synced = clock.update(&remote_ts)?;
```

### Pub/Sub Messaging

Topic-based messaging with wildcard support:

```rust
let broker = PubSubBroker::new(PubSubConfig::default());

// Subscribe with wildcards: * (single level), # (multi-level)
broker.subscribe(kernel_id, Topic::new("sensors/+/temperature"));
broker.subscribe(kernel_id, Topic::new("events/#"));

// Publish
broker.publish(Topic::new("sensors/room1/temperature"), sender, envelope, timestamp)?;
```

### Proc Macros

Define messages and kernels declaratively:

```rust
use ringkernel::prelude::*;

#[derive(RingMessage)]
#[message(type_id = 1)]
struct ComputeRequest {
    #[message(id)]
    id: MessageId,
    data: Vec<f32>,
}

#[derive(GpuType)]
struct Matrix4x4 {
    data: [f32; 16],
}
```

## Examples

The repository includes working examples:

```bash
# Basic kernel lifecycle
cargo run -p ringkernel --example basic_hello_kernel

# Kernel state management
cargo run -p ringkernel --example kernel_states

# Request-response patterns
cargo run -p ringkernel --example request_response

# Pub/sub messaging
cargo run -p ringkernel --example pub_sub

# Kernel-to-kernel messaging
cargo run -p ringkernel --example kernel_to_kernel

# WebGPU backend
cargo run -p ringkernel --example wgpu_hello --features wgpu

# Batch data processing
cargo run -p ringkernel --example batch_processor

# Telemetry and monitoring
cargo run -p ringkernel --example telemetry
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `ringkernel` | Main facade crate |
| `ringkernel-core` | Core traits, types, HLC, K2K, PubSub |
| `ringkernel-derive` | Proc macros (`#[derive(RingMessage)]`, `#[ring_kernel]`) |
| `ringkernel-cpu` | CPU backend |
| `ringkernel-cuda` | NVIDIA CUDA backend |
| `ringkernel-wgpu` | WebGPU backend |
| `ringkernel-codegen` | GPU kernel code generation |
| `ringkernel-cuda-codegen` | Rust-to-CUDA transpiler for writing GPU kernels in Rust DSL |
| `ringkernel-wavesim` | 2D wave simulation demo with tile-based FDTD |

## Testing

```bash
# Run all tests (240+ tests)
cargo test --workspace

# CUDA backend tests (requires NVIDIA GPU)
cargo test -p ringkernel-cuda --test gpu_execution_verify

# WebGPU backend tests (requires GPU)
cargo test -p ringkernel-wgpu --features wgpu-tests -- --ignored

# Benchmarks
cargo bench --package ringkernel
```

## Performance

Benchmarked on NVIDIA hardware:

- Message queue throughput: ~75M operations/sec
- Host-to-device bandwidth: ~7.6 GB/s (PCIe 4.0)
- HLC timestamp generation: <10ns per tick

### WaveSim Showcase Application

The `ringkernel-wavesim` crate demonstrates RingKernel's capabilities with a 2D acoustic wave simulation using tile-based actors. Performance highlights:

| Backend | Grid Size | Steps/sec | Throughput |
|---------|-----------|-----------|------------|
| CPU (SoA + SIMD + Rayon) | 256×256 | 35,418 | 2.3B cells/s |
| CUDA Packed (GPU-only halo) | 128×128 | 100,000+ | 100M+ cells/s |
| CUDA Packed (GPU-only halo) | 256×256 | 112,837 | 7.4M cells/s |
| CUDA Packed (GPU-only halo) | 512×512 | 71,324 | 18.7M cells/s |

**GPU vs CPU speedup at 512×512: 9.9x**

The CUDA Packed backend demonstrates GPU-only halo exchange—all tile communication happens via GPU memory copies with zero host transfers during simulation. This eliminates the traditional bottleneck of host-GPU synchronization for stencil computations.

**Features:**
- Interactive GUI with real-time visualization
- Drawing mode for absorbers and reflectors to create interference patterns
- Multiple backends: CPU (SIMD), CUDA tile-based, CUDA packed

```bash
# Run WaveSim GUI
cargo run -p ringkernel-wavesim --bin wavesim --release --features cuda

# Run WaveSim benchmarks
cargo run -p ringkernel-wavesim --bin bench_packed --release --features cuda
```

See [WaveSim PERFORMANCE.md](crates/ringkernel-wavesim/PERFORMANCE.md) for detailed analysis.

Performance varies significantly by hardware and workload.

## Documentation

**Online Documentation:** [https://mivertowski.github.io/RustCompute/](https://mivertowski.github.io/RustCompute/)

- **[API Reference](https://mivertowski.github.io/RustCompute/api/ringkernel/)** - Complete API documentation generated from source
- **[Guide](https://mivertowski.github.io/RustCompute/guide/01-architecture-overview.md)** - Architecture and usage guides

Detailed documentation is also available in the `docs/` directory:

- [Architecture Overview](docs/01-architecture-overview.md)
- [Core Abstractions](docs/03-core-abstractions.md)
- [Memory Management](docs/04-memory-management.md)
- [GPU Backends](docs/05-gpu-backends.md)

## Rust-to-CUDA Code Generation

Write GPU kernels in Rust and transpile them to CUDA C:

```rust
use ringkernel_cuda_codegen::{transpile_global_kernel, dsl::*};
use syn::parse_quote;

// Write kernel logic in Rust DSL
let kernel: syn::ItemFn = parse_quote! {
    fn my_kernel(data: &mut [f32], n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        data[idx as usize] = data[idx as usize] * 2.0;
    }
};

// Transpile to CUDA C
let cuda_code = transpile_global_kernel(&kernel)?;
// Generates: extern "C" __global__ void my_kernel(float* data, int n) { ... }
```

**Supported DSL Features:**
- Thread/block indices: `thread_idx_x()`, `block_idx_x()`, `block_dim_x()`, `grid_dim_x()`
- Early return: `if cond { return; }`
- Match expressions → switch/case
- Stencil patterns with `GridPos` abstraction
- Automatic type mapping (f32→float, i32→int, etc.)

## Known Limitations

- Metal backend is not yet implemented
- WebGPU lacks 64-bit atomics (WGSL limitation)
- Persistent kernel mode requires CUDA compute capability 7.0+

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome. Priority areas:

- Metal backend implementation
- Additional examples and documentation
- Performance optimization
- Testing on diverse hardware

Please open an issue to discuss significant changes before submitting PRs.
