# RingKernel

[![Crates.io](https://img.shields.io/crates/v/ringkernel-core.svg)](https://crates.io/crates/ringkernel-core)
[![Documentation](https://docs.rs/ringkernel-core/badge.svg)](https://docs.rs/ringkernel-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://mivertowski.github.io/RustCompute/)

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
    .with_cooperative(true)         // Enable grid-wide sync (CUDA cooperative groups)
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

The repository includes 20+ working examples organized by category:

```bash
# === Basic Examples ===
cargo run -p ringkernel --example basic_hello_kernel    # Runtime, lifecycle, suspend/resume
cargo run -p ringkernel --example kernel_states         # State machine, multi-kernel

# === Messaging Examples ===
cargo run -p ringkernel --example request_response      # Correlation IDs, priorities
cargo run -p ringkernel --example pub_sub               # Topic wildcards, QoS
cargo run -p ringkernel --example kernel_to_kernel      # K2K direct messaging
cargo run -p ringkernel --example ping_pong             # K2K broker usage

# === CUDA Codegen Examples ===
cargo run -p ringkernel --example global_kernel         # SAXPY, halo exchange, array init
cargo run -p ringkernel --example stencil_kernel        # FDTD wave, heat diffusion, GridPos
cargo run -p ringkernel --example ring_kernel_codegen   # Persistent kernels, HLC, K2K

# === Advanced Examples ===
cargo run -p ringkernel --example educational_modes     # WaveSim parallel computing modes
cargo run -p ringkernel --example multi_gpu             # Multi-GPU load balancing

# === Integration Examples ===
cargo run -p ringkernel --example axum_api              # REST API with Axum
cargo run -p ringkernel --example grpc_server           # gRPC patterns
cargo run -p ringkernel --example batch_processor       # Data pipelines
cargo run -p ringkernel --example telemetry             # Metrics, alerts

# === WebGPU ===
cargo run -p ringkernel --example wgpu_hello --features wgpu
```

## Showcase Applications

RingKernel includes five comprehensive showcase applications demonstrating GPU-accelerated computing.

| App | Description | Run Command |
|-----|-------------|-------------|
| **WaveSim** | 2D acoustic wave simulation with FDTD | `cargo run -p ringkernel-wavesim --release` |
| **WaveSim3D** | 3D acoustic wave simulation with binaural audio | `cargo run -p ringkernel-wavesim3d --release` |
| **TxMon** | Real-time transaction monitoring | `cargo run -p ringkernel-txmon --release` |
| **AccNet** | Accounting network analytics | `cargo run -p ringkernel-accnet --release` |
| **ProcInt** | Process intelligence with DFG mining | `cargo run -p ringkernel-procint --release` |

See the [Showcase Applications Guide](docs/15-showcase-applications.md) for detailed documentation with screenshots.

## Crate Structure

| Crate | Description |
|-------|-------------|
| `ringkernel` | Main facade crate |
| `ringkernel-core` | Core traits, types, HLC, K2K, PubSub |
| `ringkernel-derive` | Proc macros (`#[derive(RingMessage)]`, `#[ring_kernel]`) |
| `ringkernel-cpu` | CPU backend |
| `ringkernel-cuda` | NVIDIA CUDA backend with cooperative groups support |
| `ringkernel-wgpu` | WebGPU backend |
| `ringkernel-codegen` | GPU kernel code generation |
| `ringkernel-cuda-codegen` | Rust-to-CUDA transpiler for writing GPU kernels in Rust DSL |
| `ringkernel-wgpu-codegen` | Rust-to-WGSL transpiler for writing GPU kernels in Rust DSL (WebGPU) |
| `ringkernel-wavesim` | 2D wave simulation demo with tile-based FDTD and educational modes |
| `ringkernel-wavesim3d` | 3D acoustic wave simulation with binaural audio, block actor backend, and volumetric rendering |
| `ringkernel-txmon` | Transaction monitoring showcase with GPU-accelerated fraud detection |
| `ringkernel-accnet` | Accounting network analytics with fraud detection and GAAP compliance |
| `ringkernel-procint` | Process intelligence with DFG mining, pattern detection, conformance checking |

## Testing

```bash
# Run all tests (520+ tests)
cargo test --workspace

# CUDA backend tests (requires NVIDIA GPU)
cargo test -p ringkernel-cuda --test gpu_execution_verify

# WebGPU backend tests (requires GPU)
cargo test -p ringkernel-wgpu --features wgpu-tests -- --ignored

# Benchmarks
cargo bench --package ringkernel
```

## Performance

Benchmarked on NVIDIA RTX Ada hardware:

**CUDA Codegen Transpiler Performance:**

| Operation | Throughput | Batch Time | vs CPU |
|-----------|------------|------------|--------|
| CUDA Codegen (1M elements) | ~93B elem/sec | 0.5 µs | 12,378x |
| CUDA SAXPY PTX | ~77B elem/sec | 0.6 µs | 10,258x |
| Stencil Pattern Detection | ~15.7M TPS | 262 µs | 2.09x |

**Core System Metrics:**
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
- **Educational simulation modes** for teaching parallel computing concepts

**Educational Modes:**

WaveSim includes animated simulation modes that visually demonstrate the evolution of parallel computing:

| Mode | Era | Description |
|------|-----|-------------|
| Standard | Modern | Full-speed parallel (default) |
| CellByCell | 1950s | Sequential processing, one cell at a time |
| RowByRow | 1970s | Vector processing (Cray-style), row-by-row |
| ChaoticParallel | 1990s | Uncoordinated parallelism showing race conditions |
| SynchronizedParallel | 2000s | Barrier-synchronized parallelism |
| ActorBased | Modern | Tile-based actors with HLC for causal ordering |

Visual indicators highlight processing: green for active cells, yellow for current row (RowByRow), cyan tiles with borders for active tiles (ActorBased).

```bash
# Run WaveSim GUI
cargo run -p ringkernel-wavesim --bin wavesim --release --features cuda

# Run WaveSim benchmarks
cargo run -p ringkernel-wavesim --bin bench_packed --release --features cuda
```

See [WaveSim PERFORMANCE.md](crates/ringkernel-wavesim/PERFORMANCE.md) for detailed analysis.

### TxMon Showcase Application

The `ringkernel-txmon` crate demonstrates GPU-accelerated transaction monitoring for real-time fraud detection:

**Features:**
- Real-time transaction simulation with configurable suspicious rate
- Four detection rules: velocity breach, amount threshold, structured transactions, geographic anomaly
- Three GPU backend approaches: batch kernel, ring kernel (actor model), stencil kernel (pattern detection)
- Interactive GUI with live metrics and high-risk account tracking

**GPU Backend Approaches:**

| Approach | Description | Use Case |
|----------|-------------|----------|
| **Batch Kernel** | One thread per transaction, max throughput | High-volume batch processing |
| **Ring Kernel** | Persistent actor with HLC + K2K messaging | Real-time streaming pipelines |
| **Stencil Kernel** | 2D grid pattern detection (velocity × time) | Network anomaly detection |

```bash
# Run TxMon GUI
cargo run -p ringkernel-txmon --release --features cuda-codegen

# Run GPU benchmark
cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen
```

Performance varies significantly by hardware and workload.

## Documentation

**API Documentation:** [https://docs.rs/ringkernel](https://docs.rs/ringkernel)

**Online Guides:** [https://mivertowski.github.io/RustCompute/](https://mivertowski.github.io/RustCompute/)

- **[API Reference](https://docs.rs/ringkernel)** - Complete API documentation on docs.rs
- **[Guide](https://mivertowski.github.io/RustCompute/guide/01-architecture-overview.html)** - Architecture and usage guides

Detailed documentation is also available in the `docs/` directory:

- [Architecture Overview](docs/01-architecture-overview.md)
- [Core Abstractions](docs/03-core-abstractions.md)
- [Memory Management](docs/04-memory-management.md)
- [GPU Backends](docs/05-gpu-backends.md)

## GPU Code Generation

Write GPU kernels in Rust and transpile them to CUDA C or WGSL. Both transpilers support three kernel types with unified API:

### Backend Selection

```rust
// For NVIDIA GPUs (CUDA)
use ringkernel_cuda_codegen::{transpile_global_kernel, transpile_stencil_kernel, StencilConfig};

// For cross-platform (WebGPU/WGSL)
use ringkernel_wgpu_codegen::{transpile_global_kernel, transpile_stencil_kernel, StencilConfig};
```

The same Rust DSL code works with both backends—just change the import.

### Global Kernels

```rust
use ringkernel_cuda_codegen::{transpile_global_kernel};
use syn::parse_quote;

let kernel: syn::ItemFn = parse_quote! {
    fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        y[idx as usize] = a * x[idx as usize] + y[idx as usize];
    }
};

let cuda_code = transpile_global_kernel(&kernel)?;
// Generates: extern "C" __global__ void saxpy(...) { ... }
```

### Stencil Kernels

```rust
use ringkernel_cuda_codegen::{transpile_stencil_kernel, StencilConfig};

let kernel: syn::ItemFn = parse_quote! {
    fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
        let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p) - 4.0 * p[pos.idx()];
        p_prev[pos.idx()] = 2.0 * p[pos.idx()] - p_prev[pos.idx()] + c2 * lap;
    }
};

let config = StencilConfig::new("fdtd").with_tile_size(16, 16).with_halo(1);
let cuda_code = transpile_stencil_kernel(&kernel, &config)?;
```

### Ring Kernels (Persistent Actor Model)

Generate persistent GPU kernels that process messages in a loop:

```rust
use ringkernel_cuda_codegen::{transpile_ring_kernel, RingKernelConfig};

let handler: syn::ItemFn = parse_quote! {
    fn process(ctx: &RingContext, msg: &Request) -> Response {
        let tid = ctx.global_thread_id();
        ctx.sync_threads();
        Response { value: msg.value * 2.0, id: tid as u64 }
    }
};

let config = RingKernelConfig::new("processor")
    .with_block_size(128)
    .with_queue_capacity(1024)
    .with_hlc(true)   // Hybrid Logical Clocks
    .with_k2k(true);  // Kernel-to-Kernel messaging

let cuda_code = transpile_ring_kernel(&handler, &config)?;
```

**DSL Features:**
- Thread/block indices: `thread_idx_x()`, `block_idx_x()`, `block_dim_x()`, `grid_dim_x()`, `warp_size()`
- Control flow: `if/else`, `match` → switch/case, early `return`
- Loops: `for i in 0..n`, `while cond`, `loop` with `break`/`continue`
- Stencil patterns (2D): `pos.north()`, `pos.south()`, `pos.east()`, `pos.west()`, `pos.at(dx, dy)`
- Stencil patterns (3D): `pos.up()`, `pos.down()`, `pos.at(dx, dy, dz)` for volumetric kernels
- Shared memory: `__shared__` arrays and tiles
- Struct literals: `Point { x: 1.0, y: 2.0 }` → C compound literals
- Reference expressions: `&arr[idx]` → pointer with automatic `->` for field access
- Type inference: Tracks pointer variables for correct accessor generation
- **120+ GPU intrinsics** across 13 categories (atomics, warp ops, sync, math, trig, bit manipulation, memory, timing)

**Ring Kernel Features:**
- Persistent message loop with ControlBlock lifecycle
- RingContext method inlining (`ctx.thread_id()` → `threadIdx.x`)
- HLC operations: `hlc_tick()`, `hlc_now()`, `hlc_update()`
- K2K messaging: `k2k_send()`, `k2k_try_recv()`, `k2k_peek()`
- Queue intrinsics: `enqueue_response()`, `input_queue_empty()`

### WGSL-Specific Notes

The WGSL transpiler handles WebGPU limitations with automatic workarounds:

| Feature | CUDA | WGSL Workaround |
|---------|------|-----------------|
| 64-bit integers | Native `long long` | Emulated as `vec2<u32>` lo/hi pairs |
| 64-bit atomics | `atomicAdd(long long*)` | Helper functions: `atomic_inc_u64()` |
| f64 | Native `double` | Downcast to `f32` with warning |
| Persistent kernels | Persistent GPU loop | Host-driven dispatch loop |
| K2K messaging | Supported | **Not supported** (returns error) |

```rust
use ringkernel_wgpu_codegen::{transpile_ring_kernel, RingKernelConfig};

let config = RingKernelConfig::new("processor")
    .with_workgroup_size(256)
    .with_hlc(true);
    // Note: .with_k2k(true) will return an error in WGPU

let wgsl_code = transpile_ring_kernel(&handler, &config)?;
```

## Known Limitations

- Metal backend is not yet implemented
- WebGPU lacks 64-bit atomics (WGSL limitation)
- Persistent kernel mode requires CUDA compute capability 7.0+
- Cooperative groups (`grid.sync()`) requires CUDA compute capability 6.0+ and `cooperative` feature flag

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome. Priority areas:

- Metal backend implementation
- Additional examples and documentation
- Performance optimization
- Testing on diverse hardware

Please open an issue to discuss significant changes before submitting PRs.
