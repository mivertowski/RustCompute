# CLAUDE.md

## Project Overview

RingKernel is a GPU-native persistent actor model framework for Rust, focused exclusively on NVIDIA CUDA. It enables GPU-accelerated actor systems with persistent kernels, lock-free message passing, and hybrid logical clocks (HLC) for causal ordering. Proven on H100 (8,698x speedup vs `cuLaunchKernel`) and on 2× H100 NVL (8.7× NVLink P2P migration at 16 MiB; 258 GB/s sustained K2K bandwidth, ~81% of NV12 peak).

## Release Status

- **v1.0.0** (shipped): single-GPU persistent actor framework, H100-verified.
- **v1.1.0** (shipped, 2026-04-20): multi-GPU runtime (real `cuCtxEnablePeerAccess` + `cuMemcpyPeerAsync` on 2× H100 NVL), VynGraph NSAI integration points (PROV-O, multi-tenant K2K, hot rule reload, introspection streaming), 6/6 TLA+ specs pass under TLC with no counterexamples. See [`docs/benchmarks/v1.1-2x-h100-results.md`](docs/benchmarks/v1.1-2x-h100-results.md) and [`docs/verification/v1.1-tlc-report.md`](docs/verification/v1.1-tlc-report.md).
- **v1.2 groundwork** on `main` (not yet tagged): see section below.

## v1.2 Groundwork (on `main`, not yet released)

The items below landed after v1.1.0 and are additive / opt-in. They are the v1.2 roadmap goals implementable on 2× H100 NVL without B200 silicon. See `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for full detail.

- **Hierarchical work stealing** — intra-cluster DSMEM (`cluster_dsmem_work_steal`) and cross-cluster HBM (`grid_hbm_work_steal`) kernels complete the block → cluster → grid stealing hierarchy that started with the v1.1 intra-block `warp_work_steal`.
- **NVSHMEM symmetric-heap bindings** — opt-in `nvshmem` Cargo feature on `ringkernel-cuda`; `multi_gpu::nvshmem::NvshmemHeap` RAII wrapper over the NVSHMEM host ABI (`attach` / `malloc` / `put` / `get` / `barrier_all` / `fence` / `my_pe` / `n_pes`). Bootstrap (MPI / `nvshmrun` / unique-ID) is the caller's responsibility.
- **Blackwell / sm_100 capability queries** — `GpuArchitecture::supports_{cluster_launch_control,fp8,fp6,fp4,nvlink5,tee}`; `from_compute_capability` routes all 10.x / 11.x / 12.x to the Blackwell preset (covers datacenter sm_100/101/103, consumer sm_120, and DGX Spark GB10 sm_121). Rubin has no announced CC yet — callers select `rubin()` explicitly. Codegen stubs compile; runtime paths wait for B200 hardware.
- **Post-Hopper codegen scalar types** — `ScalarType::BF16`, `FP8E4M3`, `FP8E5M2`, `FP6E3M2`, `FP6E2M3`, `FP4E2M1` with per-type `min_compute_capability()`. CUDA / MSL / WGSL lowerings wired; unsupported targets reject at codegen time.
- **SpscQueue hot-path** — `head` / `tail` / producer stats / consumer stats each on their own 128-byte line (was: single line, full cross-core invalidation per op). `update_max_depth` uses `AtomicU64::fetch_max` instead of a CAS loop.
- **Delta checkpoints** — `Checkpoint::delta_from(base, new)`, `applied_with_delta(base, delta)`, and `content_digest()` for ordered identity-based diffs; recorded parent digest catches wrong-base application.
- **HBM tier direct K2K measurement** — `cluster_hbm_k2k` kernel replaces the snapshot/restart proxy used in the initial v1.1 run; Exp 1 now measures all three K2K tiers (SMEM, DSMEM, HBM) directly.
- **Multi-GPU K2K sustained bandwidth** — `paper_multi_gpu_k2k_bw` micro-bench (paper Addendum 6b): 258 GB/s @ 16 MiB on 2× H100 NVL.
- **Workspace dep consolidation** — 8 crates migrated from `version + path` to `{ workspace = true }`; root `[workspace.dependencies]` has a `ringkernel` facade entry.

## Build Commands

```bash
cargo build --workspace                          # Build entire workspace
cargo build --workspace --features cuda           # With CUDA backend
cargo test --workspace                            # Run all tests
cargo test -p ringkernel-core                     # Single crate tests
cargo test -p ringkernel-core test_name           # Single test
cargo test -p ringkernel-cuda --test gpu_execution_verify  # CUDA GPU tests (requires NVIDIA GPU)
cargo test -p ringkernel-ecosystem --features "persistent,actix,tower,axum,grpc"
cargo bench --package ringkernel

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

## Architecture

### Workspace Crates (19)

| Crate | Purpose |
|-------|---------|
| `ringkernel` | Main facade, re-exports everything |
| `ringkernel-core` | Core traits: RingMessage, MessageQueue, HlcTimestamp, ControlBlock, RingContext, RingKernelRuntime, K2K, PubSub, enterprise features |
| `ringkernel-derive` | Proc macros: `#[derive(RingMessage)]`, `#[ring_kernel]`, `#[derive(GpuType)]` |
| `ringkernel-cpu` | CPU backend (always available, testing/fallback) |
| `ringkernel-cuda` | NVIDIA CUDA backend with cooperative groups, profiling, reduction, memory pools, streams |
| `ringkernel-codegen` | GPU kernel code generation |
| `ringkernel-cuda-codegen` | Rust-to-CUDA transpiler (global/stencil/ring/persistent FDTD kernels) |
| `ringkernel-ir` | Unified IR for code generation |
| `ringkernel-ecosystem` | Web framework integrations: Actix, Axum, Tower, gRPC with persistent GPU actors |
| `ringkernel-cli` | CLI: scaffolding, codegen, compatibility checking |
| `ringkernel-audio-fft` | Example: GPU-accelerated audio FFT |
| `ringkernel-wavesim` | Example: 2D wave simulation with educational modes |
| `ringkernel-txmon` | Showcase: GPU transaction monitoring with fraud detection GUI |
| `ringkernel-accnet` | Showcase: GPU accounting network visualization |
| `ringkernel-procint` | Showcase: GPU process intelligence, DFG mining, conformance checking |
| `ringkernel-montecarlo` | Monte Carlo: Philox RNG, antithetic/control variates, importance sampling |
| `ringkernel-graph` | Graph algorithms: CSR, BFS, SCC, Union-Find, SpMV |
| `ringkernel-python` | Python bindings |

### Backend System

Backends implement `RingKernelRuntime`. Selection via features: `cpu` (default), `cuda`.
Auto-detection: `Backend::Auto` tries CUDA -> CPU.

### Feature Flags

**ringkernel:** `cpu` (default), `cuda`

**ringkernel-cuda:** `cooperative` (grid.sync(), requires nvcc), `profiling` (CUDA events, NVTX, Chrome trace)

**ringkernel-core:** `crypto`, `auth`, `rate-limiting`, `alerting`, `tls`, `enterprise` (all combined)

**ringkernel-ecosystem:** `persistent`, `persistent-cuda`, `actix`, `tower`, `axum`, `grpc`, `persistent-full`

### Serialization

Uses rkyv 0.7 for zero-copy. `RingMessage` derive macro generates serialization automatically.

## Important Patterns

### cudarc 0.19.3 API

```rust
// Module loading
let module = device.inner().load_module(ptx)?;
let func = module.load_function("kernel_name")?;

// Kernel launch (builder pattern)
use cudarc::driver::PushKernelArg;
unsafe {
    stream.launch_builder(&func).arg(&input).arg(&output).arg(&scalar).launch(cfg)?;
}

// Cooperative kernel launch
use cudarc::driver::result as cuda_result;
unsafe { cuda_result::launch_cooperative_kernel(func, grid, block, smem, stream, params)?; }
```

**Broken old API (cudarc 0.11):** `device.load_ptx()`, `device.get_func()`, `func.launch(cfg, params)` -- none of these work.

### Common Gotchas

**Queue capacity must be power of 2:**
```rust
let options = LaunchOptions::default().with_queue_capacity(1024);
```

**Avoid double activation** (kernels auto-activate by default):
```rust
// Use without_auto_activate for manual control:
let kernel = runtime.launch("test", LaunchOptions::default().without_auto_activate()).await?;
kernel.activate().await?;
```

**Result type conflict** (prelude exports `Result<T>`):
```rust
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> { ... }
```

## Testing

- Unit tests: `#[cfg(test)]` in each crate's `src/`
- Async tests: `#[tokio::test]`
- GPU tests: `#[ignore]` + feature flag `cuda`
- Property tests: `proptest` for queue/serialization invariants
- Benchmarks: `crates/ringkernel/benches/` with Criterion
- 1,617+ tests across workspace (1,595 tests pass `cargo test --workspace --release --exclude ringkernel-txmon`)

## Publishing

```bash
./scripts/publish.sh --status           # Check status
./scripts/publish.sh <TOKEN>            # Publish (auto-skips published, correct dep order)
./scripts/publish.sh --dry-run          # Dry run
```

Order: Tier 1 (core, cuda-codegen) -> Tier 2 (derive, cpu, cuda, codegen, ecosystem, audio-fft) -> Tier 3 (ringkernel) -> Tier 4 (apps)

## Dependency Versions

| Package | Version | Package | Version |
|---------|---------|---------|---------|
| tokio | 1.48 | cudarc | 0.19.3 |
| rayon | 1.11 | rkyv | 0.7 |
| thiserror | 2.0 | axum | 0.8 |
| tower | 0.5 | tonic/prost | 0.14 |
| iced | 0.13 | egui | 0.31 |
| winit | 0.30 | glam | 0.29 |
| arrow | 54 | polars | 0.46 |
| sha2 | 0.10 | | |
