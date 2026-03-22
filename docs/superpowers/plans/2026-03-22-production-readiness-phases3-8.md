# Phases 3-8: Production Readiness — Summary Plans

> **For agentic workers:** Each phase below will be expanded into a full implementation plan when the preceding phase is complete. Use this document for tracking and scope overview.

**Spec:** `docs/superpowers/specs/2026-03-22-production-readiness-design.md`

---

## Phase 3: Observability & Logging

**Goal:** Replace 781 raw println/eprintln with structured tracing. Complete profiler stubs.
**Depends on:** Phase 1 (error types must exist before adding log context)
**Estimated tasks:** 14

### Task List
- [ ] 3.1 Add `tracing` dependency to crates missing it (~10 crates)
- [ ] 3.2 Replace println/eprintln in ringkernel-wavesim (~120 instances)
- [ ] 3.3 Replace println/eprintln in ringkernel-wavesim3d (~100 instances)
- [ ] 3.4 Replace println/eprintln in ringkernel-txmon (~80 instances)
- [ ] 3.5 Replace println/eprintln in ringkernel-accnet (~60 instances)
- [ ] 3.6 Replace println/eprintln in ringkernel-procint (~50 instances)
- [ ] 3.7 Replace println/eprintln in ringkernel-cuda (~40 instances)
- [ ] 3.8 Replace println/eprintln in ringkernel-ecosystem (~30 instances)
- [ ] 3.9 Replace println/eprintln in ringkernel-core (~25 instances)
- [ ] 3.10 Replace println/eprintln in ringkernel-cli (~20 instances)
- [ ] 3.11 Replace println/eprintln in all remaining crates (~256 instances)
- [ ] 3.12 Implement real NVTX integration (feature-gated via `nvtx` crate)
- [ ] 3.13 Implement RenderDoc API integration (feature-gated)
- [ ] 3.14 Add Chrome trace + tracing span correlation
- [ ] 3.15 Add Prometheus metrics export (feature-gated)
- [ ] 3.16 Add OpenTelemetry span export for CPU-GPU boundary tracing

---

## Phase 4: Testing & Documentation

**Goal:** Close coverage gaps. Enforce documentation standards. Audit unsafe code.
**Depends on:** Phase 1 (error types), Phase 3 (logging for test diagnostics)
**Estimated tasks:** 20

### Task List
- [ ] 4.1 Add integration tests for ringkernel-derive proc macros (compile-pass/compile-fail)
- [ ] 4.2 Add FFI smoke tests for ringkernel-python
- [ ] 4.3 Add macOS CI runner or conditional tests for ringkernel-metal
- [ ] 4.4 Add feature combination tests for ringkernel-ecosystem
- [ ] 4.5 Add template edge case tests for ringkernel-codegen
- [ ] 4.6 Create GitHub Actions feature matrix CI job
- [ ] 4.7 Test combo: `{core only}` — no optional features
- [ ] 4.8 Test combo: `{core + cuda}` — CUDA backend
- [ ] 4.9 Test combo: `{core + wgpu}` — WebGPU backend
- [ ] 4.10 Test combo: `{core + enterprise}` — all enterprise features
- [ ] 4.11 Test combo: `{ecosystem + persistent + axum}` — web + GPU
- [ ] 4.12 Set up H100 self-hosted GPU CI runner
- [ ] 4.13 Enable GPU tests on every PR (not just manual dispatch)
- [ ] 4.14 Add GPU benchmark regression detection (fail if >10% regression)
- [ ] 4.15 Add `#![deny(missing_docs)]` to 6 core public crates
- [ ] 4.16 Add `#![warn(missing_docs)]` to all remaining crates
- [ ] 4.17 Add `#![deny(unsafe_op_in_unsafe_fn)]` to all crates with unsafe
- [ ] 4.18 Audit and document 48 unsafe blocks in ringkernel-wavesim3d
- [ ] 4.19 Audit and document 43 unsafe blocks in ringkernel-cuda
- [ ] 4.20 Audit and document remaining 99 unsafe blocks across other crates
- [ ] 4.21 Add code coverage metrics (tarpaulin or llvm-cov)

---

## Phase 5: Hopper Feature Integration (H100)

**Goal:** Leverage H100-native features for 2-10x perf improvement in actor messaging.
**Depends on:** Phase 2 (cudarc upgrade, architecture detection)
**Estimated tasks:** 23

### Task List

**Thread Block Clusters:**
- [ ] 5.1 Add cluster launch attribute to kernel launch config
- [ ] 5.2 Create `ClusterGroup` abstraction wrapping cooperative groups cluster API
- [ ] 5.3 Add cluster size configuration to `LaunchOptions`
- [ ] 5.4 Update cooperative kernel PTX templates with cluster intrinsics
- [ ] 5.5 Add cluster-aware actor placement (co-locate communicating actors)

**Distributed Shared Memory (DSMEM):**
- [ ] 5.6 Add DSMEM message queue implementation for intra-cluster actors
- [ ] 5.7 Implement `map_shared_rank(ptr, block_rank)` wrapper
- [ ] 5.8 Add DSMEM-backed K2K channel type
- [ ] 5.9 Benchmark DSMEM K2K vs global memory K2K
- [ ] 5.10 Add transparent fallback to global memory K2K on pre-Hopper GPUs

**TMA Async Copy:**
- [ ] 5.11 Add TMA 1D bulk async copy wrapper
- [ ] 5.12 Implement single-thread async message payload transfer
- [ ] 5.13 Add barrier-based completion tracking via `barrier_arrive_tx()`
- [ ] 5.14 Implement multicast (broadcast to all actors in cluster)
- [ ] 5.15 Add TMA to host-to-kernel message injection path

**Split-Phase Barriers:**
- [ ] 5.16 Add `barrier_arrive()` / `barrier_wait(token)` wrappers
- [ ] 5.17 Implement mailbox signaling with arrive/wait pattern
- [ ] 5.18 Replace polling-based message detection with barrier notification

**Green Contexts (SM Partitioning):**
- [ ] 5.19 Add Green Context creation via CUDA 13.1 runtime API
- [ ] 5.20 Implement SM reservation for persistent actor kernels
- [ ] 5.21 Add `GreenContextConfig` to `LaunchOptions`
- [ ] 5.22 Implement context isolation (dedicated SM partition)
- [ ] 5.23 Add SM utilization monitoring per green context

---

## Phase 6: Blackwell Features (B200)

**Goal:** Leverage B200-exclusive features for next-gen actor scheduling.
**Depends on:** Phase 5 (Hopper features as foundation)
**Estimated tasks:** 13

### Task List

**Cluster Launch Control (CLC):**
- [ ] 6.1 Implement CLC work-stealing loop for persistent actors
- [ ] 6.2 Add `clusterlaunchcontrol_try_cancel` wrapper
- [ ] 6.3 Implement scheduler warp pattern (warp 0 handles CLC, rest compute)
- [ ] 6.4 Add pipeline depth 3 pattern for latency hiding
- [ ] 6.5 Benchmark load balancing improvement vs static assignment

**Extended Clusters:**
- [ ] 6.6 Enable non-portable cluster size 16 on B200
- [ ] 6.7 Exploit 16-block clusters for 3.6MB DSMEM pool
- [ ] 6.8 Add cluster topology optimizer for actor graph placement

**TMEM & Optimizations:**
- [ ] 6.9 Evaluate TMEM for per-actor state storage
- [ ] 6.10 If applicable: implement TMEM-backed actor metadata
- [ ] 6.11 Leverage 126MB L2 cache for persistent kernel working set
- [ ] 6.12 Support `CUDA_DISABLE_PERF_BOOST` for consistent latency
- [ ] 6.13 Add 32-byte aligned vector types for message buffers

---

## Phase 7: Generalization & Ecosystem Expansion

**Goal:** Extract reusable GPU infrastructure. Add multi-GPU. Expand integrations.
**Depends on:** Phases 5-6 (for GPU infrastructure improvements)
**Estimated tasks:** 23

### Task List

**Module Extraction from RustGraph (~15K LOC):**
- [ ] 7.1 Extract Stratified GPU Memory Pool (800 LOC)
- [ ] 7.2 Extract Multi-Stream Execution Manager (1,100 LOC)
- [ ] 7.3 Extract PTX Compilation Cache (430 LOC)
- [ ] 7.4 Extract GPU Profiling/NVTX Integration (680 LOC)
- [ ] 7.5 Extract Benchmarking Framework (1,200 LOC)
- [ ] 7.6 Extract Hybrid CPU-GPU Processing (620 LOC)
- [ ] 7.7 Extract Resource Estimation & Guarding (550 LOC)
- [ ] 7.8 Enhance Lock-Free Message Bus (K2K, 1,270 LOC)
- [ ] 7.9 Extract Kernel Mode Selection Framework (500 LOC)
- [ ] 7.10 Enhance GPU Context Manager

**Multi-GPU Support:**
- [ ] 7.11 Implement P2P mapped memory for multi-GPU actor communication
- [ ] 7.12 Add NVSHMEM integration for cross-GPU messaging
- [ ] 7.13 Add NVLink-aware actor placement
- [ ] 7.14 Grace Hopper unified memory with hardware coherence

**CUDA Wishlist Improvements:**
- [ ] 7.15 Replace polling-based H2K signaling with Green Context + barrier notification
- [ ] 7.16 DSMEM-backed K2K mailboxes (leveraging Phase 5)
- [ ] 7.17 CLC work-stealing for dynamic scheduling (leveraging Phase 6)
- [ ] 7.18 Green Context SM partitioning for preemption isolation
- [ ] 7.19 Add periodic state checkpoint to host memory (fault tolerance)

**Ecosystem Integrations:**
- [ ] 7.20 NCCL integration for multi-GPU collectives
- [ ] 7.21 cuDNN integration for neural network ops in actors
- [ ] 7.22 cuBLAS integration for dense linear algebra
- [ ] 7.23 RAPIDS cuDF integration for GPU DataFrame processing

---

## Phase 8: Roadmap Corrections & Metal/WebGPU Backend

**Goal:** Align roadmap with reality. Improve secondary backends.
**Depends on:** None (can run in parallel with Phases 5-7)
**Estimated tasks:** 14

### Task List

**Roadmap Corrections:**
- [ ] 8.1 Update ROADMAP.md: Metal persistent kernels -> "Partial (event-driven only)"
- [ ] 8.2 Update ROADMAP.md: WebGPU ring kernels -> "Emulated (host-driven)"
- [ ] 8.3 Update ROADMAP.md: WebGPU K2K -> "Host-mediated (WGSL limitation)"
- [ ] 8.4 Verify and update VSCode extension status
- [ ] 8.5 Verify and update GPU Playground status

**Metal Backend:**
- [ ] 8.6 Document Metal limitations in user-facing docs
- [ ] 8.7 Focus Metal docs on stencil kernels + Apple Silicon unified memory
- [ ] 8.8 Add Metal Performance Shaders (MPS) integration
- [ ] 8.9 Add Metal System Trace profiling
- [ ] 8.10 Expand test coverage for Metal stencil kernels

**WebGPU Backend:**
- [ ] 8.11 Document WebGPU limitations in user-facing docs
- [ ] 8.12 Add WebGPU subgroup operations (where hardware supports)
- [ ] 8.13 Optimize host-mediated K2K path (batch message transfers)
- [ ] 8.14 Add WebGPU performance benchmarks

---

## Overall Phase Dependencies

```
Phase 1 (Crash Safety) ──┬──> Phase 3 (Observability) ──> Phase 4 (Testing)
                         │
Phase 2 (H100/B200)  ───┼──> Phase 5 (Hopper) ──> Phase 6 (Blackwell) ──> Phase 7 (Generalization)
                         │
                         └──> Phase 8 (Roadmap Corrections) [independent]
```

**Critical path:** Phase 1 -> Phase 2 -> Phase 5 -> Phase 6 -> Phase 7
**Parallel track:** Phase 3 -> Phase 4 (can run alongside Phase 2)
**Independent:** Phase 8 (anytime)

---

## Grand Total: 89 items across 8 phases

| Phase | Items | Priority | Status |
|-------|-------|----------|--------|
| Phase 1: Crash Safety | 26 | P0 | Planned |
| Phase 2: H100/B200 Build | 18 | P0 | Planned |
| Phase 3: Observability | 16 | P1 | Planned |
| Phase 4: Testing & Docs | 21 | P1 | Planned |
| Phase 5: Hopper Features | 23 | P1 | Planned |
| Phase 6: Blackwell Features | 13 | P2 | Planned |
| Phase 7: Generalization | 23 | P2 | Planned |
| Phase 8: Roadmap/Metal/WebGPU | 14 | P3 | Planned |
| **Total** | **154** | | |

Note: Item count expanded from 89 high-level findings to 154 implementation tasks as items were decomposed into concrete steps.
