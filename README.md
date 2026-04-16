# RingKernel

[![Crates.io](https://img.shields.io/crates/v/ringkernel-core.svg)](https://crates.io/crates/ringkernel-core)
[![Documentation](https://docs.rs/ringkernel-core/badge.svg)](https://docs.rs/ringkernel-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-1496%20passed-brightgreen.svg)]()

A GPU-native persistent actor model framework for Rust.

RingKernel treats GPU thread blocks as long-running actors that maintain state, communicate via lock-free queues, and manage lifecycle (create, destroy, restart, supervise) — all within a single persistent kernel launch. No kernel re-launch overhead. No host round-trip for inter-actor messaging.

## What's Proven (H100 Benchmarks)

Measured on NVIDIA H100 NVL with locked clocks, exclusive compute mode, and statistical rigor (95% CI, Cohen's d, Welch's t-test). Full data in [`docs/benchmarks/ACADEMIC_PROOF.md`](docs/benchmarks/ACADEMIC_PROOF.md).

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Persistent actor command injection | **55 ns** | 8,698x faster than cuLaunchKernel |
| vs CUDA Graphs (best traditional) | **0.2 us total** | 3,005x faster than graph replay |
| Sustained throughput (60 seconds) | **5.54M ops/s** | CV 0.05%, zero degradation |
| Cluster sync (H100 Hopper) | **0.628 us/sync** | 2.98x faster than grid.sync() |
| Zero-copy serialization | **0.544 ns** | Sub-nanosecond pointer cast |
| Actor create (on-GPU) | **163 us** | Erlang-style, no kernel re-launch |
| Actor restart (on-GPU) | **197 us** | State reset + reactivation |
| Streaming pipeline | **610K events/s** | 4-stage GPU-native pipeline |
| WaveSim3D GPU stencil | **78K Mcells/s** | 217.9x vs CPU (40-core EPYC) |
| Async memory alloc | **878 ns** | 116.9x vs cuMemAlloc |

## Key Capabilities

- **Persistent GPU kernels** that run for the entire application lifetime
- **Actor lifecycle on GPU**: create, destroy, restart, supervise — within a single kernel launch
- **Lock-free SPSC queues**: truly lock-free (atomic head/tail, no mutexes), 55 ns per message
- **Thread Block Clusters** (H100): DSMEM messaging, cluster.sync(), Green Contexts
- **Hybrid Logical Clocks** for causal ordering (30 ns/tick)
- **Rust-to-GPU transpilers**: CUDA codegen (155+ intrinsics), WGSL codegen
- **Enterprise features**: auth, rate limiting, TLS, multi-tenancy, NVTX profiling
- **Actor infrastructure**: supervision trees, named registry, backpressure, dead letter queue, memory pressure handling

## Honest Assessment

**What works well:**
- CUDA persistent kernel execution is production-quality
- Lock-free queues are fast and stable (sub-100ns, 0.05% CV over 60s)
- Hopper features (clusters, DSMEM, Green Contexts) are verified on H100
- Actor lifecycle (create/destroy/restart/supervise) works end-to-end on GPU

**What's partial or limited:**
- `CudaRuntime::launch()` fires a real GPU kernel (auto-activate), but the PTX template is generic — custom user kernels require the codegen path or `PersistentSimulation` directly
- WebGPU and Metal backends are scaffolds — they have runtime infrastructure but no persistent kernel support (API limitation, not a bug)
- The unified IR (`ringkernel-ir`) is built but the codegen crates bypass it (direct AST-to-backend transpilation)
- WGSL codegen has 23 unimplemented atomic/shuffle intrinsics (blocked by WebGPU subgroup spec)

**What's missing:**
- Multi-GPU actor communication (P2P, NVLink-aware placement)
- Dynamic work stealing on GPU (host-side scheduler exists, GPU-side is future work)
- Fault tolerance checkpointing (actor state snapshots planned, not implemented)
- Streaming integrations (Kafka, NATS — planned)

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

    // Launch a persistent kernel (auto-activates on GPU)
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
┌──────────────────────────┐           ┌──────────────────────────────────┐
│  Application (async)     │           │  Supervisor (Block 0)            │
│  ActorSupervisor         │◄─ DMA ──►│  Actor Pool (Blocks 1-N)         │
│  ActorRegistry           │           │    ├─ Control Block (256B each)  │
│  FlowController          │           │    ├─ H2K/K2H Queues            │
│  DeadLetterQueue         │           │    ├─ Inter-Actor Buffers        │
│  MemoryPressureMonitor   │           │    └─ K2K Routes (device mem)    │
└──────────────────────────┘           │  grid.sync() / cluster.sync()   │
                                       └──────────────────────────────────┘
```

## GPU Actor Lifecycle

Actors on GPU follow the same lifecycle as Erlang processes:

```
Dormant → Initializing → Active ⇄ Draining → Terminated / Failed
```

```rust
use ringkernel_core::actor::{ActorSupervisor, ActorConfig, RestartPolicy};

let mut supervisor = ActorSupervisor::new(128); // 127 actor slots + 1 supervisor

// Create actors
let config = ActorConfig::named("sensor-reader")
    .with_restart_policy(RestartPolicy::OneForOne {
        max_restarts: 3,
        window: Duration::from_secs(60),
    });
let actor = supervisor.create_actor(&config, None)?;
supervisor.activate_actor(actor)?;

// Create child actors
let child = supervisor.create_actor(&child_config, Some(actor))?;

// Cascading kill (Erlang-style)
supervisor.kill_tree(actor); // Kills actor and all descendants

// Named registry for service discovery
let mut registry = ActorRegistry::new();
registry.register("isa_ontology", kernel_id);
let actor = registry.lookup("standards/isa/*"); // Wildcard patterns
```

## Crate Structure

| Crate | Purpose | Status |
|-------|---------|--------|
| `ringkernel` | Main facade | Stable |
| `ringkernel-core` | Core traits, actor lifecycle, HLC, K2K, queues, enterprise | Stable |
| `ringkernel-derive` | Proc macros (`#[derive(RingMessage)]`, `#[ring_kernel]`) | Stable |
| `ringkernel-cpu` | CPU backend (testing, fallback) | Stable |
| `ringkernel-cuda` | NVIDIA CUDA: persistent kernels, Hopper features, cooperative groups | **Stable, H100-verified** |
| `ringkernel-wgpu` | WebGPU cross-platform backend | Scaffold |
| `ringkernel-metal` | Apple Metal backend | Scaffold |
| `ringkernel-cuda-codegen` | Rust-to-CUDA transpiler (155+ intrinsics) | Stable |
| `ringkernel-wgpu-codegen` | Rust-to-WGSL transpiler | Partial (23 atomic stubs) |
| `ringkernel-ir` | Unified IR for multi-backend codegen | Built, underutilized |
| `ringkernel-ecosystem` | Actix, Axum, Tower, gRPC, Arrow, Polars integrations | Stable |
| `ringkernel-cli` | CLI scaffolding and codegen | Stable |
| `ringkernel-montecarlo` | Philox RNG, variance reduction | Stable |
| `ringkernel-graph` | CSR, BFS, SCC, Union-Find, SpMV | Stable |

## Testing

```bash
cargo test --workspace                    # 1,496 tests, 0 failures
cargo test -p ringkernel-core             # 592 core tests
cargo test -p ringkernel-cuda \
  --features "cuda,cooperative" \
  --release -- --ignored                  # 31 GPU tests on CUDA hardware
cargo bench --package ringkernel          # Criterion benchmarks
```

## Roadmap

### Completed (this release)
- [x] Persistent cooperative kernel execution (CUDA)
- [x] Lock-free SPSC/MPSC queues (truly lock-free, no mutexes)
- [x] Hopper Thread Block Clusters, DSMEM, TMA, Green Contexts
- [x] GPU actor lifecycle (create/destroy/restart/supervise)
- [x] Named actor registry with wildcard service discovery
- [x] Credit-based backpressure and dead letter queue
- [x] GPU memory pressure handling (budgets, mitigation)
- [x] Dynamic scheduling (work stealing protocol)
- [x] Async memory pool (116.9x faster than cuMemAlloc)
- [x] NVTX profiling, Chrome trace export, memory tracking
- [x] Paper-quality H100 benchmarks with statistical analysis

### Next
- [ ] Multi-GPU P2P actor communication (NVLink, NVSHMEM)
- [ ] GPU-side work stealing within persistent kernel
- [ ] State checkpointing and fault recovery
- [ ] Route codegen through unified IR
- [ ] Streaming integrations (Kafka, NATS, Redis Streams)
- [ ] LLM provider bridge (OpenAI, Anthropic, local models)
- [ ] Vector store / GPU-resident embedding index
- [ ] WebGPU event-driven actor mode
- [ ] Blackwell (B200) Cluster Launch Control

## Documentation

- **Academic proof**: [`docs/benchmarks/ACADEMIC_PROOF.md`](docs/benchmarks/ACADEMIC_PROOF.md) — H100 empirical evidence
- **Benchmark data**: [`docs/benchmarks/h100-b200-baseline.md`](docs/benchmarks/h100-b200-baseline.md) — Full tables with CI
- **Gap analysis**: [`docs/superpowers/GAP_ANALYSIS.md`](docs/superpowers/GAP_ANALYSIS.md) — Vision vs reality
- **Methodology**: [`docs/benchmarks/METHODOLOGY.md`](docs/benchmarks/METHODOLOGY.md) — Statistical protocol
- **API Reference**: [docs.rs/ringkernel](https://docs.rs/ringkernel)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
