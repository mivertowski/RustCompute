# RingKernel Gap Analysis: Vision vs Reality

> Comprehensive audit of what's proven, what's real, what's stubbed,
> and what's needed to finalize the GPU-native actor model.
>
> Conducted 2026-04-16 against commit `b8c60fd`.

---

## Executive Summary

RingKernel is **~75% production-real, ~15% partial, ~10% scaffold**.
The core persistent actor paradigm on CUDA is proven and benchmarked.
However, there are specific gaps between the vision ("GPU-native actor model
with everything that belongs to it") and what's actually shipping today.

**Strongest areas**: CUDA persistent kernel execution, benchmark evidence,
cooperative groups, Hopper features, codegen transpilers, ecosystem integrations.

**Weakest areas**: CudaRuntime→GPU execution gap, CPU-side queue lock-freedom claim,
secondary backends (WebGPU/Metal persistence), IR underutilization, security
feature-gate fallbacks.

---

## 1. What's Real and Proven (with H100 Evidence)

### 1.1 Persistent Kernel Execution ✅ PROVEN
- `PersistentSimulation` launches real cooperative kernels via `cuLaunchCooperativeKernel`
- H2K/K2H queues on mapped memory — truly lock-free (volatile + fences)
- K2K routing and halo exchange on device memory
- Grid.sync() for multi-block synchronization
- **Evidence**: WaveSim3D persistent backend runs 100+ timesteps in single kernel launch
- **Evidence**: 60-second sustained throughput test — zero degradation

### 1.2 Hopper Thread Block Clusters ✅ PROVEN
- `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`
- Cluster.sync() 2.98x faster than grid.sync() (measured on H100)
- DSMEM K2K messaging verified: all 4 blocks exchanged data
- Green Context SM partitioning via `cuGreenCtxCreate`
- Async memory pool: 116.9x faster than cuMemAlloc

### 1.3 Performance Claims ✅ PROVEN
- Persistent actor injection: 75 ns (8,698x vs traditional, 3,005x vs CUDA Graphs)
- GPU stencil: 217.9x vs CPU
- HLC causal ordering: 30 ns/tick
- Zero-copy serialization: 0.544 ns (sub-nanosecond)
- All with 95% CI, statistical significance

### 1.4 Codegen Transpilers ✅ REAL
- CUDA codegen: 13.6K LOC, 155+ intrinsics, generates executable CUDA C
- WGSL codegen: 5.1K LOC, generates valid WebGPU shaders
- Both produce code that compiles and runs

### 1.5 Core Infrastructure ✅ REAL
- CpuRuntime: full backend with real async threading
- K2K Broker: real message router with direct/indirect delivery
- PubSub: topic-based with pattern matching
- Enterprise features: auth, rate-limiting, alerting, TLS — all real implementations
- Memory layout: ControlBlock (128B), MessageHeader (256B), HLC (24B) — verified
- 1,447 tests pass

---

## 2. Critical Gaps to Close

### 2.1 🔴 CudaRuntime.launch() Doesn't Actually Launch GPU Kernels

**The Problem**: `CudaRuntime::launch("my_kernel", opts)` loads PTX and creates
metadata but does NOT call cuLaunchKernel. Users must separately call
`kernel.activate()` to start GPU execution. Meanwhile, `PersistentSimulation`
directly launches cooperative kernels.

This means the primary API — `RingKernelRuntime::launch()` — doesn't produce
GPU execution on the CUDA backend. There's a disconnect between the trait contract
and what actually happens.

**Impact**: Users following the documented API (`runtime.launch()`) get a loaded-
but-not-running kernel. Only users who know to call `PersistentSimulation` directly
get actual persistent GPU execution.

**Fix**: Bridge `CudaRuntime::launch()` to `PersistentSimulation::start()` for
persistent kernel types. When `LaunchOptions` requests a persistent kernel,
`CudaRuntime` should:
1. Generate kernel code via codegen
2. Create PersistentSimulation
3. Call `sim.start()` → `cuLaunchCooperativeKernel`
4. Return a KernelHandle that wraps the persistent simulation
5. `handle.send()` maps to H2K queue injection, `handle.recv()` to K2H polling

This unifies the two execution paths and makes the trait contract truthful.

**Effort**: Medium (1-2 days). The pieces exist; they need wiring.

### 2.2 🔴 CPU-Side SpscQueue Uses Mutex, Not Lock-Free

**The Problem**: `SpscQueue` claims "lock-free message passing" but uses
`parking_lot::Mutex<Option<MessageEnvelope>>` per ring buffer slot. The head/tail
pointers are atomic, but the actual data access is mutex-protected.

The GPU-side queues (H2K/K2H in persistent.rs) ARE truly lock-free — they use
volatile reads/writes with memory fences. But the CPU-side queue benchmarked at
75 ns is measuring Mutex lock/unlock overhead.

**Impact**: The "lock-free" claim in benchmarks is technically inaccurate for the
CPU-side queue. The 75 ns number is real but includes Mutex cost. A truly
lock-free implementation would be faster.

**Fix**: Replace `Vec<Mutex<Option<MessageEnvelope>>>` with a true lock-free
ring buffer using `AtomicPtr` or `UnsafeCell` with acquire/release ordering.
Two approaches:
- **Fixed-size slots with AtomicU8 state flag**: Each slot has an atomic state
  (Empty/Full). Producer CAS Empty→Writing, writes data, sets Full. Consumer
  CAS Full→Reading, reads data, sets Empty.
- **Crossbeam-style**: Use `crossbeam::ArrayQueue` as the underlying buffer.

**Effort**: Small-Medium (1 day). Well-understood data structure.

### 2.3 🟡 WGSL Codegen: 23 Unimplemented Intrinsics

**The Problem**: The WGSL codegen has 23 `unimplemented!()` calls for atomic
operations and warp shuffles. These are blocked by WebGPU subgroup extensions
that aren't universally available yet.

**Impact**: Blocks fine-grained GPU parallelism on WebGPU. Atomic operations
are essential for the lock-free queue protocol on GPU.

**Fix**: Implement using WGSL subgroup extensions where available, fall back to
shared-memory reduction patterns where not. Mark unsupported operations as
compile-time errors with clear messages.

**Effort**: Medium (2-3 days for full coverage).

---

## 3. Important Gaps

### 3.1 🟡 IR Underutilized — Codegen Bypasses It

**The Problem**: `ringkernel-ir` has a full SSA-based IR with three lowering
passes (CUDA, WGSL, MSL), but the codegen crates (`ringkernel-cuda-codegen`,
`ringkernel-wgpu-codegen`) transpile Rust AST directly to backend code,
bypassing the IR entirely.

**Impact**: Loses the key benefit of a unified IR — write one optimization pass,
get improvements on all backends. Currently, each codegen crate has its own
independent optimization logic.

**Fix**: Route the codegen pipeline through the IR:
`Rust AST → IR → optimize → lower_cuda/lower_wgsl/lower_msl → backend code`

This is an architectural refactor, not a bug fix.

**Effort**: Large (1-2 weeks). Requires rewriting codegen front-ends to emit IR
instead of backend code directly.

### 3.2 🟡 WebGPU/Metal: No Persistent Kernel Support

**The Problem**: `ringkernel-wgpu` and `ringkernel-metal` are scaffolds — they
have runtime/kernel/memory frameworks but cannot run persistent kernels because
WebGPU and Metal lack cooperative groups and mapped memory semantics.

**Impact**: The "GPU-native actor model" is CUDA-only for persistent actors.
WebGPU and Metal can only do event-driven (re-launch) patterns.

**Fix**: Two-tier approach:
- **Tier 1 (achievable)**: Implement event-driven actor mode on WebGPU/Metal
  that emulates persistence by rapid kernel re-launch + state preservation
  in device memory. Slower than CUDA persistent but still actor-model semantics.
- **Tier 2 (research)**: When WebGPU gets subgroup extensions and persistent
  compute shader proposals mature, implement true persistence.

**Effort**: Tier 1 = Medium (1 week per backend). Tier 2 = dependent on spec.

### 3.3 🟡 Security: XOR Fallback Without `crypto` Feature

**The Problem**: Without the `crypto` feature flag, memory encryption silently
falls back to XOR-based "encryption" that is NOT cryptographically secure.
There's no warning at compile time or runtime.

**Impact**: Users who don't enable `crypto` get a false sense of security.

**Fix**: Either:
- Make `crypto` a default feature, or
- Add `#[deprecated]` warning on the XOR fallback, or
- Refuse to compile encryption operations without the `crypto` feature

**Effort**: Small (hours).

---

## 4. What's Missing from the Vision

### 4.1 Dynamic Actor Scheduling / Work Stealing
The current model is static: each thread block IS an actor, assigned at launch.
There's no runtime load balancing. If one actor's workload spikes, its SM is
saturated while neighbors idle.

**Needed**: A scheduler warp pattern (warp 0 handles scheduling, rest compute)
or CLC work-stealing (Blackwell Phase 6). This is the #1 feature gap for
production datacenter use.

### 4.2 Fault Tolerance / Checkpointing
No actor state checkpointing exists. A GPU reset = total state loss.
Persistent actors need periodic snapshots to host memory for recovery.

**Needed**: Checkpoint protocol: periodic H2K "snapshot" command → kernel copies
state to mapped buffer → host persists to disk/network.

### 4.3 Multi-GPU Actor Communication
Currently single-GPU only. No P2P mapped memory, no NVLink-aware placement,
no NVSHMEM integration.

**Needed**: Cross-GPU K2K via P2P or NVSHMEM. Actor placement optimizer
considering NVLink topology.

### 4.4 Actor Supervision / Lifecycle Management
No supervisor hierarchy. No "restart actor on failure" policy. No actor
dependency graph.

**Needed**: Erlang-style supervisor trees: if a child actor fails (NaN, infinite
loop), supervisor detects via K2H heartbeat timeout and restarts.

### 4.5 Backpressure / Flow Control
The SPSC queues have fixed capacity. If a producer outpaces a consumer,
messages are dropped silently (or the producer spins).

**Needed**: Explicit backpressure signals: queue full → notify producer →
adaptive rate control.

### 4.6 Observability: Per-Actor Metrics
NVTX profiling exists but there's no per-actor metric collection (messages
processed, latency histogram, queue depth) exposed through the runtime API.

**Needed**: Embed metric counters in the control block. Expose via
`runtime.metrics()` per kernel.

---

## 5. Maturity Matrix

| Component | Status | Evidence Level | Priority to Fix |
|-----------|--------|---------------|----------------|
| **CUDA Persistent Kernel** | ✅ Real | H100 benchmarks | — |
| **Cooperative Groups** | ✅ Real | H100 benchmarks | — |
| **Hopper Clusters + DSMEM** | ✅ Real | H100 tests | — |
| **H2K/K2H Queues (GPU-side)** | ✅ Real, lock-free | Volatile + fences | — |
| **CUDA Codegen** | ✅ Real | 155+ intrinsics | — |
| **Core Traits + CPU Runtime** | ✅ Real | 1,447 tests | — |
| **Enterprise Features** | ✅ Real | Auth/TLS/Rate-limit | — |
| **Ecosystem Integrations** | ✅ Real | Actix/Axum/Tower/gRPC | — |
| **SpscQueue (CPU-side)** | 🟡 Partial | Mutex-based, not lock-free | **HIGH** |
| **CudaRuntime→GPU bridge** | 🟡 Partial | Metadata only, no launch | **HIGH** |
| **WGSL Atomic Intrinsics** | 🟡 Partial | 23 unimplemented!() | Medium |
| **IR Pipeline** | 🟡 Partial | Built but bypassed | Medium |
| **Crypto Fallback** | 🟡 Risky | XOR without warning | Medium |
| **WebGPU Persistence** | 🔵 Scaffold | No persistent kernels | Low (API-limited) |
| **Metal Persistence** | 🔵 Scaffold | No persistent kernels | Low (API-limited) |
| **Dynamic Scheduling** | ❌ Missing | — | **HIGH** |
| **Fault Tolerance** | ❌ Missing | — | **HIGH** |
| **Multi-GPU** | ❌ Missing | — | Medium |
| **Actor Supervision** | ❌ Missing | — | Medium |
| **Backpressure** | ❌ Missing | — | Medium |
| **Per-Actor Metrics** | ❌ Missing | — | Low |

---

## 6. Recommended Next Steps (Priority Order)

### Phase A: Close Critical Gaps (makes the core story airtight)

1. **Make SpscQueue truly lock-free** — Replace Mutex slots with atomic
   state flags. This fixes the "lock-free" claim and may improve the 75 ns
   benchmark to ~30-50 ns.

2. **Bridge CudaRuntime.launch() to PersistentSimulation** — Wire up the
   trait so `runtime.launch("actor", opts.persistent())` actually starts
   a cooperative kernel. This makes the API truthful.

3. **Fix crypto XOR fallback** — Default `crypto` feature or compile error.

### Phase B: Complete the Actor Model (makes it a real actor framework)

4. **Actor supervision** — Heartbeat-based failure detection via K2H timeout.
   Restart policy in LaunchOptions.

5. **Backpressure protocol** — Add `queue_pressure` field to ControlBlock.
   Producer checks before enqueue. Flow control mode in LaunchOptions.

6. **Checkpointing** — H2K "snapshot" command. Periodic state dump to mapped
   memory. Host-side persistence.

### Phase C: Scale and Generalize (makes it datacenter-grade)

7. **Dynamic scheduling** — Scheduler warp pattern for load balancing within
   the persistent kernel.

8. **Multi-GPU K2K** — P2P mapped memory for cross-GPU actor communication.

9. **Route codegen through IR** — Unified optimization pipeline.

10. **WebGPU event-driven actors** — Non-persistent actor mode for portability.

---

## 7. What's Needed for Paper Finality

The academic proof (`ACADEMIC_PROOF.md`) is solid for the **persistent kernel
paradigm claim**. To make it bulletproof:

1. **Disclose the CPU-side queue is Mutex-based** in the paper's threats to
   validity. The GPU-side H2K/K2H queues ARE lock-free; the benchmark numbers
   for those are valid. The CPU-side throughput number (5.54M ops/s) measures
   the host injection path, which includes Mutex overhead.

2. **Clarify the "launch" semantics**: The benchmark compares
   `PersistentSimulation::start()` (real GPU) vs traditional `cuLaunchKernel`,
   not `CudaRuntime::launch()`. This is the honest comparison.

3. **Add a "Future Work" section** covering: dynamic scheduling,
   fault tolerance, multi-GPU, and the CudaRuntime integration gap.

With these disclosures, the paper's claims are defensible and the evidence is real.
