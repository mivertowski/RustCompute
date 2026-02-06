# RingKernel

[![Crates.io](https://img.shields.io/crates/v/ringkernel-core.svg)](https://crates.io/crates/ringkernel-core)
[![Documentation](https://docs.rs/ringkernel-core/badge.svg)](https://docs.rs/ringkernel-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-1416%20passed-brightgreen.svg)]()

A GPU-native persistent actor model framework for Rust.

RingKernel treats GPU compute units as long-running actors that maintain state between invocations. Instead of launching kernels per-operation, kernels persist on the GPU, communicate via lock-free queues, and process messages continuously.

## Key Capabilities

- **Persistent GPU-resident kernels** that maintain state across invocations
- **Lock-free message queues** for host-GPU and kernel-to-kernel communication
- **Hybrid Logical Clocks (HLC)** for causal ordering across distributed operations
- **Multiple GPU backends**: CPU (always available), CUDA (cudarc 0.18.2), WebGPU (wgpu 27.0), Metal (experimental)
- **Rust-to-GPU transpilers** for writing CUDA and WGSL kernels in a Rust DSL
- **Zero-copy serialization** via rkyv for efficient GPU data transfer
- **Enterprise features**: authentication, rate limiting, TLS/mTLS, multi-tenancy, observability

## Installation

```toml
[dependencies]
ringkernel = "0.4"
tokio = { version = "1.48", features = ["full"] }
```

For GPU backends:

```toml
# NVIDIA CUDA
ringkernel = { version = "0.4", features = ["cuda"] }

# WebGPU (cross-platform)
ringkernel = { version = "0.4", features = ["wgpu"] }
```

## Quick Start

```rust
use ringkernel::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await?;

    // Launch a persistent kernel
    let kernel = runtime.launch("processor", LaunchOptions::default()).await?;
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
Host (CPU)                              Device (GPU)
┌──────────────────────────┐           ┌──────────────────────────┐
│  Application (async)     │           │  Control Block (128 B)   │
│  Runtime (tokio)         │◄─ DMA ──►│  Input Queue (lock-free) │
│  Message Bridge          │           │  Output Queue (lock-free)│
└──────────────────────────┘           │  Persistent Kernel       │
                                       └──────────────────────────┘
```

Kernels follow a state machine: `Created → Launched → Active ⇄ Deactivated → Terminated`

## GPU Code Generation

Write GPU kernels in Rust and transpile to CUDA C or WGSL. The same DSL code works with both backends.

### Global Kernels

```rust
use ringkernel_cuda_codegen::transpile_global_kernel;
use syn::parse_quote;

let kernel: syn::ItemFn = parse_quote! {
    fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        y[idx as usize] = a * x[idx as usize] + y[idx as usize];
    }
};

let cuda_code = transpile_global_kernel(&kernel)?;
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
    .with_envelope_format(true)
    .with_hlc(true)
    .with_k2k(true);

let cuda_code = transpile_ring_kernel(&handler, &config)?;
```

The DSL supports 120+ GPU intrinsics across 13 categories: atomics, warp operations, synchronization, math, trigonometric, memory, and more.

## Messaging

### Kernel-to-Kernel (K2K)

```rust
let runtime = CpuRuntime::new().await?;
let broker = runtime.k2k_broker().unwrap();
let receipt = broker.send(source_id, dest_id, envelope).await?;
```

### Pub/Sub with Wildcards

```rust
let broker = PubSubBroker::new(PubSubConfig::default());
broker.subscribe(kernel_id, Topic::new("sensors/+/temperature"));
broker.publish(Topic::new("sensors/room1/temperature"), sender, envelope, timestamp)?;
```

### Hybrid Logical Clocks

```rust
let clock = HlcClock::new(node_id);
let ts1 = clock.tick();
let ts2 = clock.tick();
assert!(ts1 < ts2);

// Synchronize with remote timestamp
let synced = clock.update(&remote_ts)?;
```

## Proc Macros

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

## Backends

| Backend | Status | Platforms | Requirements |
|---------|--------|-----------|--------------|
| CPU | Stable | All | None |
| CUDA | Stable | Linux, Windows | NVIDIA GPU, CUDA 12.x, cudarc 0.18.2 |
| WebGPU | Stable | All | Vulkan/Metal/DX12 capable GPU, wgpu 27.0 |
| Metal | Experimental | macOS, iOS | Apple GPU, metal-rs 0.31 |

## Enterprise Features

Enable with `features = ["enterprise"]` on `ringkernel-core`:

- **Authentication**: API key, JWT (RS256/HS256), chained providers
- **Authorization**: RBAC with policy evaluation, fine-grained resource rules
- **Rate Limiting**: Token bucket, sliding window, leaky bucket algorithms
- **Multi-tenancy**: Tenant context, resource quotas, usage tracking
- **TLS/mTLS**: Certificate management with hot reload, SNI, session resumption
- **K2K Encryption**: AES-256-GCM and ChaCha20-Poly1305 for kernel-to-kernel messages
- **Observability**: Prometheus export, OTLP tracing, structured logging, alert routing
- **Resilience**: Circuit breaker, degradation manager, kernel watchdog, automatic recovery

## Crate Structure

| Crate | Purpose |
|-------|---------|
| `ringkernel` | Main facade, re-exports core + derive + backends |
| `ringkernel-core` | Core traits, HLC, K2K, PubSub, queues, enterprise features |
| `ringkernel-derive` | Proc macros (`#[derive(RingMessage)]`, `#[ring_kernel]`, `#[derive(GpuType)]`) |
| `ringkernel-cpu` | CPU backend with mock GPU testing infrastructure |
| `ringkernel-cuda` | NVIDIA CUDA backend: cooperative groups, profiling, reduction, memory pools, streams |
| `ringkernel-wgpu` | WebGPU cross-platform backend |
| `ringkernel-metal` | Apple Metal backend (experimental) |
| `ringkernel-codegen` | Shared GPU code generation infrastructure and DSL marker functions |
| `ringkernel-cuda-codegen` | Rust-to-CUDA transpiler (global, stencil, ring kernel types) |
| `ringkernel-wgpu-codegen` | Rust-to-WGSL transpiler |
| `ringkernel-ir` | Unified IR for multi-backend codegen (CUDA/WGSL/MSL) |
| `ringkernel-ecosystem` | Framework integrations: Actix, Axum, Tower, gRPC, Arrow, Polars |
| `ringkernel-cli` | CLI for scaffolding, codegen, and validation |
| `ringkernel-python` | Python bindings via PyO3 with async/await support |
| `ringkernel-montecarlo` | Monte Carlo primitives: Philox RNG, variance reduction techniques |
| `ringkernel-graph` | Graph algorithms: CSR, BFS, SCC, Union-Find, SpMV |

## Showcase Applications

| Application | Description | Command |
|-------------|-------------|---------|
| **WaveSim** | 2D acoustic wave simulation with tile-based FDTD | `cargo run -p ringkernel-wavesim --release` |
| **WaveSim3D** | 3D wave simulation with persistent GPU actors and binaural audio | `cargo run -p ringkernel-wavesim3d --release` |
| **TxMon** | Real-time GPU-accelerated transaction fraud detection | `cargo run -p ringkernel-txmon --release --features cuda-codegen` |
| **AccNet** | Accounting network analytics with fraud and GAAP compliance | `cargo run -p ringkernel-accnet --release` |
| **ProcInt** | Process intelligence with DFG mining and conformance checking | `cargo run -p ringkernel-procint --release` |

## Performance

Benchmarked on NVIDIA RTX Ada hardware:

| Metric | Value |
|--------|-------|
| CUDA codegen throughput | ~93B elem/sec (12,378x vs CPU) |
| Message queue throughput | ~75M ops/sec |
| Host-to-device bandwidth | ~7.6 GB/s (PCIe 4.0) |
| HLC timestamp generation | <10ns per tick |
| Persistent actor inject latency | 0.03 us (11,327x faster than traditional kernel launch) |
| WaveSim GPU vs CPU (512x512) | 9.9x speedup |
| WaveSim3D GPU stencil (64^3) | 78,046 Mcells/s (280x vs CPU) |

## CLI

```bash
cargo install ringkernel-cli

ringkernel new my-gpu-app --template persistent-actor
ringkernel codegen src/kernels/processor.rs --backend cuda,wgsl
ringkernel check --backends all
```

## Python Bindings

```python
import ringkernel
import asyncio

async def main():
    runtime = await ringkernel.RingKernel.create(backend="cpu")
    kernel = await runtime.launch("processor", ringkernel.LaunchOptions())
    await kernel.terminate()
    await runtime.shutdown()

asyncio.run(main())
```

## Testing

```bash
cargo test --workspace                    # 1416 tests, 96 GPU-only ignored
cargo test -p ringkernel-core             # Core: 525+ tests incl. property-based
cargo bench --package ringkernel          # Benchmarks
```

## Examples

```bash
# Basic
cargo run -p ringkernel --example basic_hello_kernel
cargo run -p ringkernel --example kernel_to_kernel

# CUDA codegen
cargo run -p ringkernel --example global_kernel
cargo run -p ringkernel --example stencil_kernel
cargo run -p ringkernel --example ring_kernel_codegen

# WebGPU
cargo run -p ringkernel --example wgpu_hello --features wgpu

# Integration
cargo run -p ringkernel --example axum_api
cargo run -p ringkernel --example grpc_server
```

## Documentation

- **API Reference**: [docs.rs/ringkernel](https://docs.rs/ringkernel)
- **Guides**: [mivertowski.github.io/RustCompute](https://mivertowski.github.io/RustCompute/)

## Known Limitations

- Metal backend is experimental (no persistent kernels yet)
- WebGPU lacks 64-bit atomics (WGSL limitation)
- Persistent kernel mode requires CUDA compute capability 7.0+
- Cooperative groups (`grid.sync()`) requires CC 6.0+ and `cooperative` feature flag

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome. Priority areas: Metal backend completion, additional examples, performance optimization, and hardware testing. Please open an issue to discuss significant changes before submitting PRs.
