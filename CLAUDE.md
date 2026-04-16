# CLAUDE.md

## Project Overview

RingKernel is a GPU-native persistent actor model framework for Rust, focused exclusively on NVIDIA CUDA. It enables GPU-accelerated actor systems with persistent kernels, lock-free message passing, and hybrid logical clocks (HLC) for causal ordering. Proven on H100 with 8,698x speedup vs traditional kernel launch.

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
- 1,496+ tests across workspace

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
