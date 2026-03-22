# RingKernel Production Readiness & H100/B200 Validation

**Date:** 2026-03-22
**Status:** Design Spec
**Scope:** All findings from comprehensive codebase audit

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Phase 1: Crash Safety & Error Handling](#phase-1-crash-safety--error-handling)
3. [Phase 2: H100/B200 Build Support & Baseline](#phase-2-h100b200-build-support--baseline)
4. [Phase 3: Observability & Logging](#phase-3-observability--logging)
5. [Phase 4: Testing & Documentation](#phase-4-testing--documentation)
6. [Phase 5: Hopper Feature Integration](#phase-5-hopper-feature-integration)
7. [Phase 6: Blackwell Features](#phase-6-blackwell-features)
8. [Phase 7: Generalization & Ecosystem Expansion](#phase-7-generalization--ecosystem-expansion)
9. [Phase 8: Roadmap Corrections & Metal Backend](#phase-8-roadmap-corrections--metal-backend)
10. [Tracking Checklist](#tracking-checklist)

---

## Executive Summary

A comprehensive audit of RingKernel identified **89 actionable items** across 8 phases. The framework has strong architectural foundations (1,416 tests, SSA-based IR, 3-target codegen) but needs hardening for production use and modernization for H100/B200 GPUs.

**Critical findings:**
- 1,152 `unwrap()` calls in non-test code (crash risk)
- PTX targets sm_89 (Ada); H100/B200 features unused
- cudarc pinned at 0.18.2 (latest: 0.19.3)
- TLS cert loading is a placeholder (security gap)
- 781 raw println/eprintln instead of structured logging
- Hopper/Blackwell features (DSMEM, TMA, Green Contexts, CLC) not leveraged
- Roadmap overclaims 100% completion despite Metal scaffold status

**Approach:** 8 phases, each independently deliverable. Phases 1-2 are blockers for H100/B200 validation. Phases 3-4 enable production deployment. Phases 5-6 unlock hardware-native performance. Phases 7-8 expand scope.

---

## Phase 1: Crash Safety & Error Handling

**Goal:** Eliminate panic-on-error paths in production code. Establish proper error types.
**Priority:** P0 (blocking production use)
**Estimated scope:** ~40 files across 8 crates

### 1.1 Typed Error Enums (replace Result<_, String>)

**Finding:** 188 instances of `Result<T, String>` across the workspace. Errors lose context, cannot be matched programmatically.

**Items:**

| ID | Crate | File(s) | Count | Action |
|----|-------|---------|-------|--------|
| 1.1.1 | ringkernel-accnet | Multiple source files | 16 | Create `AccNetError` enum with thiserror |
| 1.1.2 | ringkernel-wavesim | simulation/*.rs | ~12 | Create `WaveSimError` enum |
| 1.1.3 | ringkernel-wavesim3d | simulation/*.rs | ~10 | Create `WaveSim3dError` enum |
| 1.1.4 | ringkernel-txmon | Multiple files | ~8 | Create `TxMonError` enum |
| 1.1.5 | ringkernel-procint | Multiple files | ~8 | Create `ProcIntError` enum |
| 1.1.6 | ringkernel-cli | commands/*.rs | 9 | Create `CliError` enum |
| 1.1.7 | ringkernel-montecarlo | lib.rs, engines/*.rs | ~6 | Create `MonteCarloError` enum |
| 1.1.8 | ringkernel-audio-fft | Multiple files | ~5 | Create `AudioFftError` enum |
| 1.1.9 | ringkernel-codegen | lib.rs | ~4 | Create `CodegenError` enum |
| 1.1.10 | ringkernel-graph | algorithms/*.rs | ~5 | Create `GraphError` enum |

**Pattern:** Each crate gets a `error.rs` with a `thiserror`-derived enum. Public API functions return `Result<T, CrateError>`.

### 1.2 unwrap() Reduction (1,152 -> <200)

**Finding:** 1,152 `unwrap()` calls in non-test code. Top offenders:

| ID | Crate | Count | Strategy |
|----|-------|-------|----------|
| 1.2.1 | ringkernel-core | 327 | Replace with `?` operator using RingKernelError |
| 1.2.2 | ringkernel-cuda-codegen | 249 | `writeln!().unwrap()` -> return `Result<String>` from codegen fns |
| 1.2.3 | ringkernel-wavesim | 98 | Replace with WaveSimError propagation |
| 1.2.4 | ringkernel-ecosystem | 38 | Replace with typed errors |
| 1.2.5 | ringkernel-cpu | 37 | Replace with RingKernelError propagation |
| 1.2.6 | ringkernel-wgpu-codegen | ~30 | Same pattern as cuda-codegen |
| 1.2.7 | ringkernel-cuda | ~25 | Replace with CudaError propagation |
| 1.2.8 | ringkernel-graph | ~20 | Replace with GraphError propagation |
| 1.2.9 | ringkernel-metal | ~15 | Replace with MetalError propagation |
| 1.2.10 | All remaining crates | ~100 | Case-by-case audit |

**Strategy for codegen crates:** The 249 `writeln!(...).unwrap()` calls in cuda-codegen are formatting operations that shouldn't fail, but the pattern is wrong. Refactor codegen functions to return `Result<String, fmt::Error>` and propagate with `?`.

### 1.3 TLS Certificate Loading Fix

**Finding:** `ringkernel-core/src/tls.rs:607` — `load_certificate()` returns empty vectors for cert_chain and private_key.

| ID | File | Action |
|----|------|--------|
| 1.3.1 | crates/ringkernel-core/src/tls.rs | Implement PEM parsing using rustls-pemfile crate |
| 1.3.2 | crates/ringkernel-core/Cargo.toml | Add rustls-pemfile dependency (feature-gated under `tls`) |
| 1.3.3 | crates/ringkernel-core/src/tls.rs | Add integration tests for cert loading |

### 1.4 Critical Stub Completions

| ID | Stub | File | Action |
|----|------|------|--------|
| 1.4.1 | CloudWatch audit sink | core/src/audit.rs:1053 | Implement with aws-sdk-cloudwatchlogs (feature-gated) |
| 1.4.2 | OTLP span export | core/src/observability.rs:2463 | Implement HTTP export via reqwest (feature-gated) |
| 1.4.3 | CLI input validation | cli/src/commands/mod.rs:30 | Add proper input validation, remove unwrap on chars() |

---

## Phase 2: H100/B200 Build Support & Baseline

**Goal:** Build and run correctly on H100/B200. Establish performance baselines.
**Priority:** P0 (required for hardware validation)

### 2.1 Architecture Detection & Targeting

| ID | File | Action |
|----|------|--------|
| 2.1.1 | crates/ringkernel-cuda/build.rs | Replace hardcoded `sm_89` with runtime detection or env var `RINGKERNEL_CUDA_ARCH` |
| 2.1.2 | crates/ringkernel-cuda/build.rs | Add multi-arch compilation: `-gencode arch=compute_75,code=sm_75 ... arch=compute_100,code=sm_100` |
| 2.1.3 | crates/ringkernel-cuda/src/device.rs | Add `GpuArchitecture::hopper()` preset (sm_90, 228KB shared, 50MB L2, 132 SMs for H100) |
| 2.1.4 | crates/ringkernel-cuda/src/device.rs | Add `GpuArchitecture::blackwell()` preset (sm_100, 228KB shared, 126MB L2, est. 192 SMs for B200) |
| 2.1.5 | crates/ringkernel-cuda/src/device.rs | Add runtime GPU detection via `cuDeviceGetAttribute` to auto-select architecture |

### 2.2 cudarc Upgrade

| ID | Action |
|----|--------|
| 2.2.1 | Upgrade cudarc from 0.18.2 to 0.19.3 in workspace Cargo.toml |
| 2.2.2 | Fix breaking change: `get_global` now returns `CudaViewMut` |
| 2.2.3 | Fix deprecated memcpy methods |
| 2.2.4 | Add cluster launch attribute support via sys-level bindings |
| 2.2.5 | Update all CUDA examples and tests for new API |

### 2.3 libcu++ Ordered Atomics

| ID | File | Action |
|----|------|--------|
| 2.3.1 | crates/ringkernel-cuda-codegen/src/persistent_fdtd.rs | Enable `use_libcupp_atomics` by default |
| 2.3.2 | crates/ringkernel-cuda-codegen/src/intrinsics.rs | Add `cuda::atomic<T, thread_scope_system>` codegen |
| 2.3.3 | crates/ringkernel-cuda/src/persistent.rs | Migrate from `atomicAdd` + `__threadfence_system()` to libcu++ atomics |

### 2.4 Baseline Benchmarks on H100/B200

| ID | Benchmark | Metric |
|----|-----------|--------|
| 2.4.1 | Message injection latency | Target: <0.03us (current Ada baseline) |
| 2.4.2 | Queue throughput | Target: >75M ops/sec |
| 2.4.3 | K2K halo exchange | Measure per-element latency |
| 2.4.4 | Cooperative grid sync | Measure grid.sync() overhead at various grid sizes |
| 2.4.5 | Reduction throughput | Target: >93B elem/sec (current baseline) |
| 2.4.6 | Persistent kernel sustained throughput | 60-second sustained run |
| 2.4.7 | Memory pool allocation rate | O(1) verification at scale |
| 2.4.8 | WaveSim3D end-to-end | Full simulation benchmark |
| 2.4.9 | TxMon fraud detection throughput | Transactions/sec |
| 2.4.10 | ProcInt DFG mining | Events/sec |

---

## Phase 3: Observability & Logging

**Goal:** Replace raw console output with structured logging. Complete profiler integration.
**Priority:** P1 (required for production monitoring)

### 3.1 Replace println/eprintln with tracing (781 instances)

| ID | Crate | Count | Action |
|----|-------|-------|--------|
| 3.1.1 | ringkernel-wavesim | ~120 | Replace with tracing::{info,warn,debug,error} |
| 3.1.2 | ringkernel-wavesim3d | ~100 | Same |
| 3.1.3 | ringkernel-txmon | ~80 | Same |
| 3.1.4 | ringkernel-accnet | ~60 | Same |
| 3.1.5 | ringkernel-procint | ~50 | Same |
| 3.1.6 | ringkernel-cuda | ~40 | Same |
| 3.1.7 | ringkernel-ecosystem | ~30 | Same |
| 3.1.8 | ringkernel-core | ~25 | Same |
| 3.1.9 | ringkernel-cli | ~20 | Same |
| 3.1.10 | All remaining | ~256 | Same |

**Pattern:** Add `tracing` as dependency where not present. Use appropriate log levels:
- `error!` for failures
- `warn!` for degraded operation (stub fallbacks)
- `info!` for lifecycle events (kernel launch, terminate)
- `debug!` for operational details
- `trace!` for hot-path diagnostics

### 3.2 Profiler Stub Completion

| ID | Profiler | File | Action |
|----|----------|------|--------|
| 3.2.1 | NVTX | core/src/observability.rs:2898 | Implement real NVTX integration via nvtx crate (feature-gated) |
| 3.2.2 | RenderDoc | core/src/observability.rs:2914 | Implement RenderDoc API integration (feature-gated) |
| 3.2.3 | Metal System Trace | core/src/observability.rs | Implement Metal GPU signpost API (macOS, feature-gated) |
| 3.2.4 | Chrome trace | cuda/src/profiling/ | Already implemented; add span correlation with tracing |

### 3.3 Metrics Export

| ID | Action |
|----|--------|
| 3.3.1 | Add Prometheus metrics export (feature-gated) for kernel lifecycle, queue depth, memory usage |
| 3.3.2 | Add OpenTelemetry span export for distributed tracing across CPU-GPU boundary |

---

## Phase 4: Testing & Documentation

**Goal:** Close coverage gaps. Enforce documentation standards.
**Priority:** P1 (quality gates for production)

### 4.1 Missing Test Coverage

| ID | Crate | Gap | Action |
|----|-------|-----|--------|
| 4.1.1 | ringkernel-derive | 0 tests | Add integration tests: compile-pass/compile-fail for all proc macros |
| 4.1.2 | ringkernel-python | 0 tests | Add FFI smoke tests, memory safety tests |
| 4.1.3 | ringkernel-metal | 2 ignored tests | Add macOS CI runner or conditional tests |
| 4.1.4 | ringkernel-ecosystem | Minimal | Add feature combination tests |
| 4.1.5 | ringkernel-codegen | 3 tests | Add template substitution edge cases |

### 4.2 Feature Matrix Testing

| ID | Combination | Action |
|----|------------|--------|
| 4.2.1 | `{core only}` | Ensure core compiles and tests pass without any optional features |
| 4.2.2 | `{core + cuda}` | CUDA backend integration |
| 4.2.3 | `{core + wgpu}` | WebGPU backend integration |
| 4.2.4 | `{core + enterprise}` | All enterprise features together |
| 4.2.5 | `{ecosystem + persistent + axum}` | Web framework + GPU integration |
| 4.2.6 | `{all-backends}` | All backends simultaneously |
| 4.2.7 | CI matrix job | Add GitHub Actions matrix for feature combos |

### 4.3 GPU CI Gate

| ID | Action |
|----|--------|
| 4.3.1 | Add self-hosted GPU runner for H100 |
| 4.3.2 | Run ignored GPU tests on every PR (not just manual dispatch) |
| 4.3.3 | Add GPU benchmark regression detection (fail if >10% regression) |

### 4.4 Documentation Enforcement

| ID | Crate | Action |
|----|-------|--------|
| 4.4.1 | ringkernel-core | Add `#![deny(missing_docs)]` |
| 4.4.2 | ringkernel-derive | Add `#![deny(missing_docs)]` |
| 4.4.3 | ringkernel-cuda | Add `#![deny(missing_docs)]` |
| 4.4.4 | ringkernel-wgpu | Add `#![deny(missing_docs)]` |
| 4.4.5 | ringkernel-ir | Add `#![deny(missing_docs)]` |
| 4.4.6 | ringkernel (facade) | Add `#![deny(missing_docs)]` |
| 4.4.7 | All remaining crates | Add `#![warn(missing_docs)]` at minimum |
| 4.4.8 | All crates with unsafe | Add `#![deny(unsafe_op_in_unsafe_fn)]` |

### 4.5 Unsafe Code Audit

| ID | Crate | Count | Action |
|----|-------|-------|--------|
| 4.5.1 | ringkernel-wavesim3d | 48 blocks | Add `// SAFETY:` comments to all |
| 4.5.2 | ringkernel-cuda | 43 blocks | Add `// SAFETY:` comments, verify correctness |
| 4.5.3 | ringkernel-graph | 16 blocks | Add `// SAFETY:` comments |
| 4.5.4 | ringkernel-montecarlo | 11 blocks | Add `// SAFETY:` comments |
| 4.5.5 | ringkernel-txmon | 8 blocks | Add `// SAFETY:` comments |
| 4.5.6 | ringkernel-accnet | 5 blocks | Add `// SAFETY:` comments |
| 4.5.7 | All remaining | ~59 blocks | Audit and document |

---

## Phase 5: Hopper Feature Integration (H100)

**Goal:** Leverage H100-native features for 2-10x performance improvements in actor messaging.
**Priority:** P1 (performance unlock)
**Requires:** Phase 2 complete (build support, cudarc upgrade)

### 5.1 Thread Block Clusters

| ID | Action |
|----|--------|
| 5.1.1 | Add cluster launch attribute to kernel launch config (`CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`) |
| 5.1.2 | Create `ClusterGroup` abstraction in ringkernel-cuda wrapping cooperative groups cluster API |
| 5.1.3 | Add cluster size configuration to `LaunchOptions` (default: 1 for backward compat, up to 8 portable / 16 non-portable) |
| 5.1.4 | Update cooperative kernel PTX templates with cluster intrinsics |
| 5.1.5 | Add cluster-aware actor placement: co-locate communicating actors in same cluster |

### 5.2 Distributed Shared Memory (DSMEM)

| ID | Action |
|----|--------|
| 5.2.1 | Add DSMEM message queue implementation: actors in same cluster exchange messages via shared memory instead of global memory |
| 5.2.2 | Implement `map_shared_rank(ptr, block_rank)` wrapper for cross-block shared memory access |
| 5.2.3 | Add DSMEM-backed K2K channel type (7x faster than global memory K2K) |
| 5.2.4 | Benchmark: measure DSMEM K2K latency vs current global memory K2K |
| 5.2.5 | Add fallback: if cluster size = 1 (pre-Hopper), use global memory K2K transparently |

### 5.3 TMA Async Copy

| ID | Action |
|----|--------|
| 5.3.1 | Add TMA 1D bulk async copy wrapper (no tensor map needed, simplest path) |
| 5.3.2 | Implement single-thread message payload transfer: one thread issues TMA, rest continue processing |
| 5.3.3 | Add barrier-based completion tracking via `barrier_arrive_tx()` |
| 5.3.4 | Implement multicast: broadcast message to all actors in cluster with single TMA op |
| 5.3.5 | Add TMA to host-to-kernel message injection path |

### 5.4 Split-Phase Barriers

| ID | Action |
|----|--------|
| 5.4.1 | Add `barrier_arrive()` / `barrier_wait(token)` wrappers for producer-consumer actor patterns |
| 5.4.2 | Implement mailbox signaling: producer "arrives" when posting message, consumer "waits" for messages |
| 5.4.3 | Replace polling-based message detection with barrier-based notification |

### 5.5 Green Contexts (SM Partitioning)

| ID | Action |
|----|--------|
| 5.5.1 | Add Green Context creation via CUDA 13.1 runtime API |
| 5.5.2 | Implement SM reservation for persistent actor kernels (e.g., reserve 16 of 132 SMs for actors) |
| 5.5.3 | Add `GreenContextConfig` to `LaunchOptions` with SM count, priority |
| 5.5.4 | Implement context isolation: actor kernel runs in dedicated partition, won't be preempted by other work |
| 5.5.5 | Add monitoring: track SM utilization per green context |

---

## Phase 6: Blackwell Features (B200)

**Goal:** Leverage B200-exclusive features for next-gen persistent actor scheduling.
**Priority:** P2 (advanced optimization)
**Requires:** Phase 5 complete (Hopper features as foundation)

### 6.1 Cluster Launch Control (CLC) Work Stealing

| ID | Action |
|----|--------|
| 6.1.1 | Implement CLC work-stealing loop for persistent actors: idle blocks steal work from overloaded ones |
| 6.1.2 | Add `clusterlaunchcontrol_try_cancel` wrapper for dynamic block scheduling |
| 6.1.3 | Implement scheduler warp pattern (warp 0 handles CLC queries, other warps compute) |
| 6.1.4 | Add pipeline depth 3 pattern (CUTLASS-style) for latency hiding |
| 6.1.5 | Benchmark: measure load balancing improvement vs static actor assignment |

### 6.2 Extended Cluster Size

| ID | Action |
|----|--------|
| 6.2.1 | Enable non-portable cluster size 16 on B200 (`cudaFuncAttributeNonPortableClusterSizeAllowed`) |
| 6.2.2 | Exploit 16-block clusters: 16 * 228KB = 3.6MB DSMEM pool for actor communication |
| 6.2.3 | Add cluster topology optimizer: place actor graph in cluster to minimize cross-cluster communication |

### 6.3 TMEM (Tensor Memory)

| ID | Action |
|----|--------|
| 6.3.1 | Evaluate TMEM for per-actor state storage at register speed |
| 6.3.2 | If applicable: implement TMEM-backed actor metadata (actor ID, state machine, clock) |

### 6.4 Blackwell-Specific Optimizations

| ID | Action |
|----|--------|
| 6.4.1 | Leverage 126MB L2 cache (GB200) for persistent kernel working set |
| 6.4.2 | Use `CUDA_DISABLE_PERF_BOOST` env var for consistent actor latency |
| 6.4.3 | Add 32-byte aligned vector types for maximizing bandwidth in message buffers |

---

## Phase 7: Generalization & Ecosystem Expansion

**Goal:** Extract reusable GPU infrastructure, expand integrations.
**Priority:** P2 (community & adoption)
**Source:** RINGKERNEL_GENERALIZATION_PROPOSAL.md (10 modules, ~15K LOC from RustGraph)

### 7.1 Module Extraction (from RustGraph)

| ID | Module | LOC | Independence |
|----|--------|-----|-------------|
| 7.1.1 | Stratified GPU Memory Pool | 800 | 100% |
| 7.1.2 | Multi-Stream Execution Manager | 1,100 | 100% |
| 7.1.3 | PTX Compilation Cache | 430 | 100% |
| 7.1.4 | GPU Profiling/NVTX Integration | 680 | 100% |
| 7.1.5 | Benchmarking Framework | 1,200 | 95% |
| 7.1.6 | Hybrid CPU-GPU Processing | 620 | 100% |
| 7.1.7 | Resource Estimation & Guarding | 550 | 100% |
| 7.1.8 | Lock-Free Message Bus (K2K enhancement) | 1,270 | 95% |
| 7.1.9 | Kernel Mode Selection Framework | 500 | 90% |
| 7.1.10 | GPU Context Manager (enhancement) | ~500 | 100% |

### 7.2 Multi-GPU Support

| ID | Action |
|----|--------|
| 7.2.1 | Implement P2P mapped memory for multi-GPU actor communication (replaces removed multi-device cooperative groups) |
| 7.2.2 | Add NVSHMEM integration for cross-GPU messaging |
| 7.2.3 | Add NVLink-aware actor placement (minimize cross-GPU hops) |
| 7.2.4 | Grace Hopper: leverage unified memory with hardware coherence for zero-overhead host-GPU communication |

### 7.3 CUDA Wishlist Workaround Improvements

Based on docs/19-cuda-wishlist-persistent-actors.md:

| ID | Wishlist Item | Current Workaround | Improvement |
|----|--------------|-------------------|-------------|
| 7.3.1 | Host-Kernel Signaling (P0) | Mapped memory + polling | Replace polling with Green Context + barrier-based notification |
| 7.3.2 | K2K Mailboxes (P0) | Manual halo exchange | DSMEM-backed mailboxes (Phase 5) |
| 7.3.3 | Dynamic Block Scheduling (P1) | Static assignment | CLC work-stealing (Phase 6) |
| 7.3.4 | Persistent Kernel Preemption (P1) | Single kernel monopolizes | Green Context SM partitioning (Phase 5) |
| 7.3.5 | Checkpointing/Migration (P2) | None (total loss) | Add periodic state snapshot to host memory |

### 7.4 New Ecosystem Integrations

| ID | Integration | Purpose |
|----|-------------|---------|
| 7.4.1 | NCCL | Multi-GPU collective operations |
| 7.4.2 | cuDNN | Neural network operations in actors |
| 7.4.3 | cuBLAS | Dense linear algebra for actor compute |
| 7.4.4 | RAPIDS cuDF | GPU DataFrame processing in actor pipelines |

---

## Phase 8: Roadmap Corrections & Metal Backend

**Goal:** Align roadmap claims with reality. Improve Metal where possible.
**Priority:** P3 (honesty & completeness)

### 8.1 Roadmap Corrections

| ID | Item | Current Claim | Reality | Action |
|----|------|--------------|---------|--------|
| 8.1.1 | Metal persistent kernels | Complete | Scaffold only | Mark as "Partial - event-driven only" |
| 8.1.2 | WebGPU ring kernels | Complete | Host-emulated | Mark as "Emulated (host-driven)" |
| 8.1.3 | WebGPU K2K | Complete | Host-mediated | Mark as "Host-mediated (WGSL limitation)" |
| 8.1.4 | VSCode extension | Complete | Unclear status | Verify and update |
| 8.1.5 | GPU Playground | Complete | Unclear status | Verify and update |

### 8.2 Metal Backend Improvements

| ID | Action |
|----|--------|
| 8.2.1 | Document Metal limitations clearly: no persistent kernels, no cooperative groups, no f64 |
| 8.2.2 | Focus Metal on stencil kernels + Apple Silicon unified memory advantage |
| 8.2.3 | Add Metal Performance Shaders (MPS) integration for matrix ops |
| 8.2.4 | Add Metal System Trace profiling |
| 8.2.5 | Expand test coverage for Metal stencil kernels |

### 8.3 WebGPU Improvements

| ID | Action |
|----|--------|
| 8.3.1 | Document WebGPU limitations clearly in user-facing docs |
| 8.3.2 | Add WebGPU subgroup operations (where hardware supports) |
| 8.3.3 | Optimize host-mediated K2K path (batch message transfers) |
| 8.3.4 | Add WebGPU performance benchmarks |

---

## Tracking Checklist

### Phase 1: Crash Safety & Error Handling
- [ ] 1.1.1 AccNet typed errors
- [ ] 1.1.2 WaveSim typed errors
- [ ] 1.1.3 WaveSim3D typed errors
- [ ] 1.1.4 TxMon typed errors
- [ ] 1.1.5 ProcInt typed errors
- [ ] 1.1.6 CLI typed errors
- [ ] 1.1.7 MonteCarlo typed errors
- [ ] 1.1.8 AudioFFT typed errors
- [ ] 1.1.9 Codegen typed errors
- [ ] 1.1.10 Graph typed errors
- [ ] 1.2.1 unwrap reduction: ringkernel-core (327)
- [ ] 1.2.2 unwrap reduction: ringkernel-cuda-codegen (249)
- [ ] 1.2.3 unwrap reduction: ringkernel-wavesim (98)
- [ ] 1.2.4 unwrap reduction: ringkernel-ecosystem (38)
- [ ] 1.2.5 unwrap reduction: ringkernel-cpu (37)
- [ ] 1.2.6 unwrap reduction: ringkernel-wgpu-codegen (~30)
- [ ] 1.2.7 unwrap reduction: ringkernel-cuda (~25)
- [ ] 1.2.8 unwrap reduction: ringkernel-graph (~20)
- [ ] 1.2.9 unwrap reduction: ringkernel-metal (~15)
- [ ] 1.2.10 unwrap reduction: all remaining (~100)
- [ ] 1.3.1 TLS PEM parsing implementation
- [ ] 1.3.2 rustls-pemfile dependency
- [ ] 1.3.3 TLS integration tests
- [ ] 1.4.1 CloudWatch audit sink implementation
- [ ] 1.4.2 OTLP span export implementation
- [ ] 1.4.3 CLI input validation

### Phase 2: H100/B200 Build Support & Baseline
- [ ] 2.1.1 Architecture detection in build.rs
- [ ] 2.1.2 Multi-arch compilation
- [ ] 2.1.3 GpuArchitecture::hopper() preset
- [ ] 2.1.4 GpuArchitecture::blackwell() preset
- [ ] 2.1.5 Runtime GPU detection
- [ ] 2.2.1 cudarc upgrade to 0.19.3
- [ ] 2.2.2 Fix get_global breaking change
- [ ] 2.2.3 Fix deprecated memcpy methods
- [ ] 2.2.4 Cluster launch attribute support
- [ ] 2.2.5 Update examples and tests
- [ ] 2.3.1 Enable libcupp_atomics default
- [ ] 2.3.2 libcu++ codegen support
- [ ] 2.3.3 Migrate persistent.rs atomics
- [ ] 2.4.1-2.4.10 Baseline benchmarks (10 items)

### Phase 3: Observability & Logging
- [ ] 3.1.1-3.1.10 Replace println/eprintln (781 instances across 10 crate groups)
- [ ] 3.2.1 NVTX real integration
- [ ] 3.2.2 RenderDoc integration
- [ ] 3.2.3 Metal System Trace
- [ ] 3.2.4 Chrome trace + tracing correlation
- [ ] 3.3.1 Prometheus metrics export
- [ ] 3.3.2 OpenTelemetry span export

### Phase 4: Testing & Documentation
- [ ] 4.1.1 ringkernel-derive tests
- [ ] 4.1.2 ringkernel-python tests
- [ ] 4.1.3 Metal CI tests
- [ ] 4.1.4 Ecosystem tests
- [ ] 4.1.5 Codegen edge case tests
- [ ] 4.2.1-4.2.7 Feature matrix testing (7 combos + CI job)
- [ ] 4.3.1 H100 self-hosted runner
- [ ] 4.3.2 GPU tests on every PR
- [ ] 4.3.3 Benchmark regression detection
- [ ] 4.4.1-4.4.8 Documentation enforcement (8 crates + unsafe deny)
- [ ] 4.5.1-4.5.7 Unsafe code audit (190 blocks across 7 groups)

### Phase 5: Hopper Feature Integration
- [ ] 5.1.1-5.1.5 Thread Block Clusters (5 items)
- [ ] 5.2.1-5.2.5 Distributed Shared Memory (5 items)
- [ ] 5.3.1-5.3.5 TMA Async Copy (5 items)
- [ ] 5.4.1-5.4.3 Split-Phase Barriers (3 items)
- [ ] 5.5.1-5.5.5 Green Contexts (5 items)

### Phase 6: Blackwell Features
- [ ] 6.1.1-6.1.5 Cluster Launch Control (5 items)
- [ ] 6.2.1-6.2.3 Extended Cluster Size (3 items)
- [ ] 6.3.1-6.3.2 TMEM evaluation (2 items)
- [ ] 6.4.1-6.4.3 Blackwell optimizations (3 items)

### Phase 7: Generalization & Ecosystem
- [ ] 7.1.1-7.1.10 Module extraction from RustGraph (10 modules)
- [ ] 7.2.1-7.2.4 Multi-GPU support (4 items)
- [ ] 7.3.1-7.3.5 Wishlist workaround improvements (5 items)
- [ ] 7.4.1-7.4.4 New ecosystem integrations (4 items)

### Phase 8: Roadmap Corrections & Metal
- [ ] 8.1.1-8.1.5 Roadmap corrections (5 items)
- [ ] 8.2.1-8.2.5 Metal improvements (5 items)
- [ ] 8.3.1-8.3.4 WebGPU improvements (4 items)

---

**Total items: 89 across 8 phases**

**Recommended execution order:**
1. Phase 1 (crash safety) — blocking for production
2. Phase 2 (H100/B200 build) — blocking for hardware validation
3. Phase 4.1-4.2 (critical tests) — parallel with Phase 2
4. Phase 3 (observability) — enables production monitoring
5. Phase 4.3-4.5 (remaining tests, docs, unsafe audit)
6. Phase 5 (Hopper features) — performance unlock
7. Phase 8 (roadmap corrections) — honesty
8. Phase 6 (Blackwell features) — advanced
9. Phase 7 (generalization) — expansion
