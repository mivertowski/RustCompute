# RingKernel H100 GPU VM Session Guide

> **For the Claude Code instance running on the Azure H100 VM.**
> Read this before starting any work.

## Context

RingKernel is a GPU-native persistent actor model framework for Rust. A comprehensive production readiness initiative is underway. Phases 0-4 are complete (crash safety, H100/B200 build support, observability, testing). This VM session handles GPU-dependent work.

## Key Documents

Read these first:
1. **`CLAUDE.md`** — Project architecture, build commands, API patterns, gotchas
2. **`docs/superpowers/specs/2026-03-22-production-readiness-design.md`** — Master spec with checklist (check what's done vs pending)
3. **`docs/superpowers/plans/2026-03-22-production-readiness-phase2.md`** — Phase 2 plan (benchmarks)
4. **`docs/superpowers/plans/2026-03-22-production-readiness-phases3-8.md`** — Phases 5-8 plans
5. **`docs/benchmarks/h100-b200-baseline.md`** — Benchmark template to fill in
6. **`docs/19-cuda-wishlist-persistent-actors.md`** — CUDA wishlist driving Phase 5-6

## What's Already Done (Phases 0-4)

- **Crash safety**: Zero bare `.unwrap()` in production, typed error enums everywhere, `clippy::unwrap_used` lint
- **H100/B200 build**: Architecture auto-detection (`RINGKERNEL_CUDA_ARCH`), cudarc 0.19.3, Blackwell preset, libcu++ atomics default
- **Observability**: println migrated to tracing
- **Testing**: Compile-fail tests, feature matrix CI, unsafe audit for CUDA crate
- **1,447 tests pass, 0 failures**

## Your Tasks (Priority Order)

### Task 1: Validate GPU Environment
```bash
nvidia-smi
nvcc --version
cargo build --workspace --features cuda --release
cargo test --workspace --release
```
Verify the 97 previously-ignored GPU tests now pass.

### Task 2: Run Baseline Benchmarks (Phase 2.4)
Fill in `docs/benchmarks/h100-b200-baseline.md` with actual numbers:
```bash
# Lock clocks first
sudo nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits)
sudo nvidia-smi -c EXCLUSIVE_PROCESS

# Criterion benchmarks
cargo bench --package ringkernel --release

# Application benchmarks
cargo run -p ringkernel-wavesim3d --bin wavesim3d-benchmark --release --features cuda-codegen
cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen
cargo run -p ringkernel-procint --bin procint-benchmark --release
```
Run each 3 times, report median.

### Task 3: Validate persistent.rs Runtime Atomics (Phase 2.3.3)
Test the libcu++ atomics in the persistent kernel runtime:
```bash
cargo test -p ringkernel-cuda --features "cuda,cooperative" --release -- --ignored
```
If any atomics-related tests fail, check `crates/ringkernel-cuda/src/persistent.rs`.

### Task 4: Implement Phase 5 — Hopper Features
This is the main development work. See spec items 5.1-5.5.

**Priority order:**

#### 5.1 Thread Block Clusters
- Add cluster launch attribute (`CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`) to kernel launch config
- Create `ClusterGroup` abstraction wrapping cooperative groups cluster API
- Add cluster size to `LaunchOptions` (default 1, up to 8 portable)
- Key file: `crates/ringkernel-cuda/src/cooperative.rs`
- cudarc sys bindings: `cudarc::driver::sys::CUlaunchAttributeID_enum::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`

#### 5.2 Distributed Shared Memory (DSMEM)
- Add DSMEM message queue for intra-cluster actors
- `map_shared_rank(ptr, block_rank)` wrapper
- DSMEM-backed K2K channel (7x faster than global memory)
- Benchmark: DSMEM K2K vs current global memory K2K
- Fallback to global memory on pre-Hopper

#### 5.3 TMA Async Copy
- TMA 1D bulk async wrapper (no tensor map needed)
- Single-thread message payload transfer
- Barrier-based completion tracking
- Multicast to cluster

#### 5.4 Split-Phase Barriers
- `barrier_arrive()` / `barrier_wait(token)` for producer-consumer actors
- Replace polling-based message detection

#### 5.5 Green Contexts (SM Partitioning)
- `cuGreenCtxCreate` via driver API
- SM reservation for persistent actors
- Context isolation
- Requires CUDA 12.4+ (should be available)

### Task 5: NVTX Profiler Integration (Phase 3.2.1)
```bash
# NVTX should be available with CUDA toolkit
# Check: ls /usr/local/cuda/include/nvtx3/
```
Implement real NVTX integration in `crates/ringkernel-core/src/observability.rs` (currently a stub).

## Architecture Quick Reference

```
ringkernel-core     — Core traits, error types, runtime, shutdown handler
ringkernel-cuda     — CUDA backend (persistent kernels, K2K, reduction, streams)
ringkernel-cuda-codegen — Rust-to-CUDA transpiler (155+ intrinsics)
ringkernel-ir       — Unified IR for multi-backend codegen
```

Key CUDA files:
- `crates/ringkernel-cuda/src/cooperative.rs` — Cooperative groups, grid sync
- `crates/ringkernel-cuda/src/persistent.rs` — Persistent kernel mapped memory
- `crates/ringkernel-cuda/src/reduction.rs` — Reduction with mapped buffers
- `crates/ringkernel-cuda/src/driver_api.rs` — Low-level CUDA driver wrappers
- `crates/ringkernel-cuda/src/launch_config/mode.rs` — GPU architecture presets
- `crates/ringkernel-cuda/build.rs` — Architecture detection, cooperative kernel compilation

## API Patterns (from CLAUDE.md)

```rust
// cudarc 0.19.3 kernel launch
let module = device.inner().load_module(ptx)?;
let func = module.load_function("kernel_name")?;
unsafe {
    stream.launch_builder(&func).arg(&input).arg(&output).launch(cfg)?;
}

// Cooperative kernel launch
use cudarc::driver::result as cuda_result;
unsafe { cuda_result::launch_cooperative_kernel(func, grid, block, smem, stream, params)?; }

// Cluster launch (new for Phase 5) — via sys bindings
use cudarc::driver::sys::CUlaunchAttributeID_enum;
// Set CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION in launch config
```

## Commit Convention

```
feat(cuda): description          # new features
refactor(crate): description     # restructuring
fix(crate): description          # bug fixes
test(crate): description         # test additions
docs: description                # documentation

# Always end with:
Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

## Important Gotchas

- Queue capacity must be power of 2
- `block_dim_x()` returns 1 on CPU (not 0)
- cudarc 0.19.3: `launch_builder(&func).arg(&x).launch(cfg)` — NOT old `func.launch()` API
- Cooperative groups grid size capped ~1024 blocks (device-dependent)
- `RINGKERNEL_CUDA_ARCH` env var controls build target (auto-detected by setup script)
- PTX templates use sm_75 (forward-compatible minimum for cooperative groups)
