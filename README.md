# RingKernel v1.1.0

[![Crates.io](https://img.shields.io/crates/v/ringkernel-core.svg)](https://crates.io/crates/ringkernel-core)
[![Documentation](https://docs.rs/ringkernel-core/badge.svg)](https://docs.rs/ringkernel-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-1595%20passed-brightgreen.svg)]()

**Persistent GPU actors for NVIDIA CUDA, proven on H100 and 2x H100 NVL.**

RingKernel treats GPU thread blocks as long-running actors that maintain state, communicate via lock-free queues, and manage lifecycle (create, destroy, restart, supervise) -- all within a single persistent kernel launch. No kernel re-launch overhead. No host round-trip for inter-actor messaging. v1.1 adds multi-GPU migration over NVLink P2P, per-tenant isolation, PROV-O provenance, hot rule reload, and TLA+ formally verified protocols.

## Key Results

### Single-GPU (H100 NVL, v1.0 baseline)

Measured with locked clocks, exclusive compute mode, and statistical rigor (95% CI, Cohen's d, Welch's t-test). Full data in [`docs/benchmarks/ACADEMIC_PROOF.md`](docs/benchmarks/ACADEMIC_PROOF.md).

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Persistent actor command injection | **55 ns** | **8,698x faster** than cuLaunchKernel |
| vs CUDA Graphs (best traditional) | **0.2 us total** | **3,005x faster** than graph replay |
| Sustained throughput (60 seconds) | **5.54M ops/s** | CV 0.05%, zero degradation |
| Cluster sync (H100 Hopper) | **0.628 us/sync** | 2.98x faster than grid.sync() |
| Zero-copy serialization | **0.544 ns** | Sub-nanosecond pointer cast |
| Actor create (on-GPU) | **163 us** | Erlang-style, no kernel re-launch |
| Actor restart (on-GPU) | **197 us** | State reset + reactivation |
| Streaming pipeline | **610K events/s** | 4-stage GPU-native pipeline |
| WaveSim3D GPU stencil | **78K Mcells/s** | 217.9x vs CPU (40-core EPYC) |
| Async memory alloc | **878 ns** | 116.9x vs cuMemAlloc |

### Multi-GPU (2x H100 NVL, v1.1, NVLink NV12 topology)

Full data in [`docs/benchmarks/v1.1-2x-h100-results.md`](docs/benchmarks/v1.1-2x-h100-results.md).

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| K2K SMEM tier latency | **6.7 us** | baseline (intra-block, payload-independent) |
| K2K DSMEM tier latency | **9.0 - 15.3 us** | 1.3 - 2.3x vs SMEM (cross-block within cluster) |
| K2K HBM tier latency | **10.5 - 18.2 us** | 1.5 - 2.7x vs SMEM (cross-cluster via global memory) |
| NVLink P2P migration (16 MiB) | **69 us** | **8.7x faster** than host-staged |
| Multi-GPU K2K sustained bandwidth | **258 GB/s @ 16 MiB** | 81% of 318 GB/s NVLink peak |
| Lifecycle rule overhead (all 5 rules) | **23 ns mean / 30 ns p99** | Flat within measurement noise |
| Cross-tenant leak count | **0** | 13 isolation tests, multiple stress scenarios |
| TLA+ specs verified (distinct states) | **6 / 6** | 24K+ combined, no counterexamples |

## Quick Start

### CPU Backend (default, no GPU required)

```toml
[dependencies]
ringkernel = "1.1"
tokio = { version = "1.48", features = ["full"] }
```

```rust
use ringkernel::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await?;

    // Launch a persistent kernel (auto-activates)
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

### CUDA Backend (requires NVIDIA GPU + CUDA toolkit)

```toml
[dependencies]
ringkernel = { version = "1.1", features = ["cuda"] }
tokio = { version = "1.48", features = ["full"] }
```

```rust
use ringkernel::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let runtime = RingKernel::builder()
        .backend(Backend::Cuda)
        .build()
        .await?;

    let kernel = runtime.launch("gpu_processor", LaunchOptions::default()).await?;
    println!("Running on CUDA: {:?}", kernel.state());

    kernel.terminate().await?;
    runtime.shutdown().await?;
    Ok(())
}
```

## Architecture

```
Host (CPU)                              Device (GPU)
+----------------------------+          +------------------------------------+
|  Application (async)       |          |  Supervisor (Block 0)              |
|  ActorSupervisor           |<- DMA -->|  Actor Pool (Blocks 1-N)           |
|  ActorRegistry             |          |    +- Control Block (256B each)    |
|  FlowController            |          |    +- H2K/K2H Queues              |
|  DeadLetterQueue           |          |    +- Inter-Actor Buffers          |
|  MemoryPressureMonitor     |          |    +- K2K Routes (device mem)      |
+----------------------------+          |  grid.sync() / cluster.sync()      |
                                        +------------------------------------+
```

### Core Components

- **ringkernel-core**: Actor lifecycle, lock-free queues, HLC timestamps, control blocks, K2K messaging (single- and multi-tenant), PubSub, enterprise features, PROV-O provenance, hot rule reload, introspection streaming, incremental/delta checkpoints
- **ringkernel-cuda**: Persistent CUDA kernels, cooperative groups, Thread Block Clusters, DSMEM, async memory pools, NVTX profiling, **multi-GPU runtime with real `cuCtxEnablePeerAccess` + `cuMemcpyPeerAsync`**, NVLink topology probe, migration protocol, hierarchical work stealing (block / cluster / grid), opt-in NVSHMEM bindings
- **ringkernel-cuda-codegen**: Rust-to-CUDA transpiler with 155+ intrinsics (global, stencil, ring, persistent FDTD kernels)
- **ringkernel-ir**: Unified intermediate representation for code generation; supports BF16, FP8 (E4M3/E5M2), FP6 (E3M2/E2M3), and FP4 (E2M1) scalar types with per-type compute-capability gating

### Multi-GPU runtime (v1.1)

```rust
use ringkernel_cuda::multi_gpu::{MultiGpuRuntime, PlacementHint, MigrationPlan};
use ringkernel_core::actor::ActorId;

// Probe NVLink topology, retain 2 primary contexts, wire peer access.
let mgpu = MultiGpuRuntime::new(&[0, 1]).await?;

// NVLink-aware placement: co-locate on connected GPUs.
let a = mgpu.launch("actor-a", PlacementHint::Pinned(0), opts.clone()).await?;
let _b = mgpu.launch("actor-b", PlacementHint::NvlinkPreferred(0), opts).await?;

// 3-phase migration (quiesce, transfer via cuMemcpyPeerAsync, swap).
let report = mgpu.migrate_actor(MigrationPlan::new(0, 1, ActorId(42))).await?;
assert!(report.used_nvlink && report.state_checksum_match);
```

### GPU Actor Lifecycle

Actors on GPU follow the same lifecycle as Erlang processes:

```
Dormant -> Initializing -> Active <-> Draining -> Terminated / Failed
```

```rust
use ringkernel_core::actor::{ActorSupervisor, ActorConfig, RestartPolicy};

let mut supervisor = ActorSupervisor::new(128); // 127 actor slots + 1 supervisor

let config = ActorConfig::named("sensor-reader")
    .with_restart_policy(RestartPolicy::OneForOne {
        max_restarts: 3,
        window: Duration::from_secs(60),
    });
let actor = supervisor.create_actor(&config, None)?;
supervisor.activate_actor(actor)?;

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
| `ringkernel` | Main facade, re-exports everything | Stable |
| `ringkernel-core` | Core traits, actor lifecycle, HLC, K2K, queues, enterprise | Stable |
| `ringkernel-derive` | Proc macros (`#[derive(RingMessage)]`, `#[ring_kernel]`) | Stable |
| `ringkernel-cpu` | CPU backend (testing, fallback) | Stable |
| `ringkernel-cuda` | NVIDIA CUDA: persistent kernels, Hopper features, cooperative groups | **Stable, H100-verified** |
| `ringkernel-cuda-codegen` | Rust-to-CUDA transpiler (155+ intrinsics) | Stable |
| `ringkernel-codegen` | GPU kernel code generation | Stable |
| `ringkernel-ir` | Unified IR for multi-backend codegen | Stable |
| `ringkernel-ecosystem` | Actix, Axum, Tower, gRPC, Arrow, Polars integrations | Stable |
| `ringkernel-cli` | CLI: scaffolding, codegen, compatibility checking | Stable |
| `ringkernel-montecarlo` | Philox RNG, variance reduction, importance sampling | Stable |
| `ringkernel-graph` | CSR, BFS, SCC, Union-Find, SpMV | Stable |
| `ringkernel-audio-fft` | GPU-accelerated audio FFT | Stable |
| `ringkernel-wavesim` | 2D wave simulation with educational modes | Stable |
| `ringkernel-txmon` | GPU transaction monitoring with fraud detection | Stable |
| `ringkernel-accnet` | GPU accounting network visualization | Stable |
| `ringkernel-procint` | GPU process intelligence, DFG mining, conformance checking | Stable |
| `ringkernel-python` | Python bindings | Stable |

## Build Commands

```bash
# Build
cargo build --workspace                          # Build entire workspace
cargo build --workspace --features cuda           # With CUDA backend

# Test
cargo test --workspace                            # Run all tests (1,595 passing)
cargo test -p ringkernel-core                     # Core tests (600+ tests, incl. multi-tenant / provenance / rules / delta checkpoints)
cargo test -p ringkernel-cuda --test gpu_execution_verify  # CUDA GPU tests (requires NVIDIA GPU)
cargo test -p ringkernel-ecosystem --features "persistent,actix,tower,axum,grpc"
cargo bench --package ringkernel                  # Criterion benchmarks

# Examples
cargo run -p ringkernel --example basic_hello_kernel
cargo run -p ringkernel --example kernel_to_kernel
cargo run -p ringkernel --example global_kernel         # CUDA codegen
cargo run -p ringkernel --example stencil_kernel
cargo run -p ringkernel --example ring_kernel_codegen
cargo run -p ringkernel --example educational_modes
cargo run -p ringkernel --example enterprise_runtime
cargo run -p ringkernel --example pagerank_reduction --features cuda

# Applications
cargo run -p ringkernel-txmon --release --features cuda-codegen
cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen
cargo run -p ringkernel-procint --release
cargo run -p ringkernel-procint --bin procint-benchmark --release
cargo run -p ringkernel-ecosystem --example axum_persistent_api --features "axum,persistent"

# CLI
cargo run -p ringkernel-cli -- new my-app --template persistent-actor
cargo run -p ringkernel-cli -- codegen src/kernels/mod.rs --backend cuda
cargo run -p ringkernel-cli -- check --backends all
```

## Performance

### H100 Benchmarks (95% CI)

**Command Injection Latency**

| Model | Per-Command (us) | 95% CI (us) | vs Traditional |
|-------|-----------------|-------------|----------------|
| Traditional (cuLaunchKernel) | 1.583 | +/-6.0 | 1.0x |
| CUDA Graph Replay | 0.547 | +/-12.3 | 2.9x |
| **Persistent Actor** | **0.000** | **+/-0.0** | **8,698x** |

**Lock-Free Queue Throughput**

| Payload | Latency (ns) | 95% CI (ns) | Throughput (Mmsg/s) |
|---------|-------------|-------------|---------------------|
| 64 B | 72.28 | [72.26, 72.32] | 13.83 |
| 256 B | 74.67 | [74.65, 74.70] | 13.39 |
| 1 KB | 82.64 | [82.61, 82.68] | 12.10 |
| 4 KB | 177.50 | [177.47, 177.53] | 5.63 |

**60-Second Sustained Run**: 315M operations, 5.54 Mops/s, CV 0.05%, zero throughput degradation, zero memory growth.

**Hopper Features**: cluster.sync() at 0.628 us/sync (2.98x faster than grid.sync()), DSMEM ring-topology K2K messaging verified across 4-block clusters.

## Documentation

- **Academic paper**: [`docs/paper/main.tex`](docs/paper/main.tex) → `make` → 48-page `main.pdf` (*Persistent GPU Actors*, 13 sections + appendix)
- **v1.1 benchmarks (2x H100 NVL)**: [`docs/benchmarks/v1.1-2x-h100-results.md`](docs/benchmarks/v1.1-2x-h100-results.md)
- **v1.0 single-GPU baseline**: [`docs/benchmarks/h100-b200-baseline.md`](docs/benchmarks/h100-b200-baseline.md)
- **Academic proof**: [`docs/benchmarks/ACADEMIC_PROOF.md`](docs/benchmarks/ACADEMIC_PROOF.md) -- H100 empirical evidence with statistical analysis
- **Methodology**: [`docs/benchmarks/METHODOLOGY.md`](docs/benchmarks/METHODOLOGY.md) -- Statistical protocol (Welch's t-test, Cohen's d)
- **TLA+ / TLC report (v1.1)**: [`docs/verification/v1.1-tlc-report.md`](docs/verification/v1.1-tlc-report.md) -- 6 specs, no counterexamples
- **TLA+ specs**: [`docs/verification/`](docs/verification/) -- hlc, k2k_delivery, migration, multi_gpu_k2k, tenant_isolation, actor_lifecycle
- **Experiment pipeline**: [`docs/paper/experiments/RUNBOOK.md`](docs/paper/experiments/RUNBOOK.md) -- reproducible 6-experiment run_all.sh
- **Architecture**: [`docs/01-architecture-overview.md`](docs/01-architecture-overview.md)
- **API Reference**: [docs.rs/ringkernel](https://docs.rs/ringkernel)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
