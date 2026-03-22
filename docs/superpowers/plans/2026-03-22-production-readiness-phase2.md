# Phase 2: H100/B200 Build Support & Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run correctly on H100 (sm_90) and B200 (sm_100). Upgrade cudarc. Establish performance baselines.

**Architecture:** Replace hardcoded sm_89 PTX target with auto-detection or env-var override. Add GPU architecture presets for Hopper/Blackwell. Upgrade cudarc 0.18.2 -> 0.19.3. Enable libcu++ ordered atomics.

**Tech Stack:** cudarc 0.19.3, CUDA Toolkit 13.x, nvcc multi-arch compilation.

**Spec:** `docs/superpowers/specs/2026-03-22-production-readiness-design.md` (Phase 2, items 2.1-2.4)

---

## Task 1: GPU Architecture Detection & Targeting

**Files:**
- Modify: `crates/ringkernel-cuda/build.rs` — replace hardcoded sm_89
- Modify: `crates/ringkernel-cuda/src/device.rs` — add hopper/blackwell presets

- [ ] **Step 1: Read build.rs around line 119** to understand current sm_89 hardcoding
- [ ] **Step 2: Read device.rs** to find existing `GpuArchitecture` presets (volta, ampere, ada, hopper)
- [ ] **Step 3: Add RINGKERNEL_CUDA_ARCH env var support** in build.rs:
  - Check `RINGKERNEL_CUDA_ARCH` env var first
  - Fall back to `-arch=native` if nvcc supports it (CUDA 12+)
  - Fall back to `-arch=sm_89` as last resort
- [ ] **Step 4: Add multi-arch gencode** for broader compatibility:
  ```
  -gencode arch=compute_75,code=sm_75
  -gencode arch=compute_80,code=sm_80
  -gencode arch=compute_89,code=sm_89
  -gencode arch=compute_90,code=sm_90
  -gencode arch=compute_100,code=sm_100
  ```
- [ ] **Step 5: Add GpuArchitecture::blackwell()** preset in device.rs
- [ ] **Step 6: Add runtime GPU detection** via `cuDeviceGetAttribute` to auto-select architecture
- [ ] **Step 7: Run tests** — `cargo test -p ringkernel-cuda`
- [ ] **Step 8: Commit** — `"feat(cuda): add H100/B200 architecture detection and multi-arch build"`

---

## Task 2: cudarc Upgrade (0.18.2 -> 0.19.3)

**Files:**
- Modify: `Cargo.toml` (workspace) — version bump
- Modify: All CUDA crate source files with breaking API changes

- [ ] **Step 1: Read cudarc 0.19.x changelog** for breaking changes
  Key changes: `get_global` returns `CudaViewMut`, deprecated memcpy methods
- [ ] **Step 2: Bump version** in workspace Cargo.toml
- [ ] **Step 3: Fix `get_global` breaking change** — update all call sites
- [ ] **Step 4: Fix deprecated memcpy methods** — use new unified API
- [ ] **Step 5: Add cluster launch attribute support** via sys-level bindings
- [ ] **Step 6: Update examples and tests** for new API patterns
- [ ] **Step 7: Run full CUDA test suite** — `cargo test -p ringkernel-cuda`
- [ ] **Step 8: Run workspace build** — `cargo build --workspace --features cuda`
- [ ] **Step 9: Commit** — `"feat(cuda): upgrade cudarc to 0.19.3 with cluster launch support"`

---

## Task 3: libcu++ Ordered Atomics

**Files:**
- Modify: `crates/ringkernel-cuda-codegen/src/persistent_fdtd.rs`
- Modify: `crates/ringkernel-cuda-codegen/src/intrinsics.rs`
- Modify: `crates/ringkernel-cuda/src/persistent.rs`

- [ ] **Step 1: Read persistent_fdtd.rs** config to find `use_libcupp_atomics` flag
- [ ] **Step 2: Enable libcupp_atomics by default** in config
- [ ] **Step 3: Add codegen for `cuda::atomic<T, thread_scope_system>`** in intrinsics.rs
- [ ] **Step 4: Migrate persistent.rs** from `atomicAdd` + `__threadfence_system()` to libcu++ atomics
- [ ] **Step 5: Run tests** — `cargo test -p ringkernel-cuda-codegen`
- [ ] **Step 6: Commit** — `"feat(cuda): enable libcu++ ordered atomics for H100/B200"`

---

## Task 4: H100/B200 Baseline Benchmarks

**Files:**
- Modify: `crates/ringkernel/benches/` — update benchmark configs
- Create: `docs/benchmarks/h100-baseline.md` — results document

- [ ] **Step 1: Verify benchmarks compile for H100** — `cargo bench --package ringkernel --no-run`
- [ ] **Step 2: Run message injection latency** benchmark on H100
- [ ] **Step 3: Run queue throughput** benchmark on H100
- [ ] **Step 4: Run K2K halo exchange** benchmark
- [ ] **Step 5: Run cooperative grid sync** benchmark at various grid sizes
- [ ] **Step 6: Run reduction throughput** benchmark
- [ ] **Step 7: Run persistent kernel sustained throughput** (60-second run)
- [ ] **Step 8: Run WaveSim3D, TxMon, ProcInt** application benchmarks
- [ ] **Step 9: Document results** in `docs/benchmarks/h100-baseline.md`
- [ ] **Step 10: Commit** — `"docs: add H100 baseline benchmark results"`

---

## Success Criteria

1. `cargo build --workspace --features cuda` succeeds targeting sm_90 and sm_100
2. cudarc 0.19.3 integrated with all tests passing
3. `RINGKERNEL_CUDA_ARCH=sm_90` builds and runs on H100
4. `RINGKERNEL_CUDA_ARCH=sm_100` builds for B200 (runtime validation when hardware available)
5. Baseline benchmarks documented for H100
6. libcu++ atomics enabled by default in persistent kernel codegen
