# RingKernel Architecture Analysis

> Current State Assessment as of January 2026

## Executive Summary

RingKernel is a GPU-native persistent actor model framework for Rust that enables GPU-accelerated actor systems with persistent kernels, lock-free message passing, and hybrid logical clocks (HLC) for causal ordering. This document provides a comprehensive analysis of the current implementation state across all backends and subsystems.

---

## Backend Implementation Matrix

| Feature | CUDA | WebGPU | Metal | CPU |
|---------|:----:|:------:|:-----:|:---:|
| **Basic Kernel Execution** | ✅ Complete | ✅ Complete | ⚠️ Scaffold | ✅ Complete |
| **Persistent Kernels** | ✅ Complete | ❌ Not Possible | ❌ Not Started | N/A |
| **H2K Messaging** | ✅ Complete | ❌ N/A | ❌ Not Started | N/A |
| **K2H Messaging** | ✅ Complete | ❌ N/A | ❌ Not Started | N/A |
| **K2K (GPU-side)** | ✅ Complete | ❌ Not Possible | ❌ Not Started | N/A |
| **K2K (Host-side Broker)** | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |
| **Cooperative Groups** | ✅ Complete | ❌ N/A | ❌ N/A | N/A |
| **Mapped Memory** | ✅ Complete | ❌ N/A | ⚠️ Possible | Shared RAM |
| **Code Generation** | ✅ 183 tests | ⚠️ 50 tests | ❌ None | N/A |
| **HLC Timestamps** | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete |

### Legend
- ✅ Complete - Production-ready implementation
- ⚠️ Scaffold/Partial - Framework exists, needs work
- ❌ Not Started/Not Possible - Missing or technically infeasible

---

## CUDA Backend Analysis

### Strengths (Production-Ready)

**Persistent Kernel Architecture** (`ringkernel-cuda/src/persistent.rs` - 1,200+ lines)
- `PersistentSimulation` - Host-side wrapper for managing persistent kernels
- `CudaMappedBuffer<T>` - CPU/GPU visible pinned memory for command queues
- `PersistentControlBlock` (256 bytes) - GPU-resident lifecycle management
- Single kernel launch for entire simulation lifetime
- Grid-wide synchronization via cooperative groups (`cg::grid_group::sync()`)

**Message Passing Infrastructure**
- `H2KMessage` (64 bytes) - Host-to-Kernel commands with 7 command types
- `K2HMessage` (64 bytes) - Kernel-to-Host responses with acknowledgment
- `K2KInboxHeader` - Queue metadata for inter-kernel communication
- `K2KRouteEntry` - Routing table for neighbor block communication
- Lock-free SPSC queues via mapped memory

**Performance Characteristics**
| Operation | Traditional | Persistent | Speedup |
|-----------|-------------|------------|---------|
| Command Injection | 317 µs | 0.03 µs | **11,327x** |
| Single Step | 3.2 µs | 163 µs | 0.02x |
| Mixed Workload (16ms) | 40.5 ms | 15.3 ms | **2.7x** |

### Implementation Details

```rust
// Control Block Structure (256 bytes, mapped memory)
#[repr(C, align(256))]
pub struct PersistentControlBlock {
    pub status: AtomicU32,           // Running/Paused/Terminated
    pub current_step: AtomicU64,     // Simulation step counter
    pub target_step: AtomicU64,      // Steps to execute
    pub h2k_head: AtomicU32,         // H2K queue head pointer
    pub h2k_tail: AtomicU32,         // H2K queue tail pointer
    pub k2h_head: AtomicU32,         // K2H queue head pointer
    pub k2h_tail: AtomicU32,         // K2H queue tail pointer
    // ... physics parameters, sync barriers
}
```

### Known Limitations
1. Cooperative groups limited to ~512 blocks (auto-fallback to software sync)
2. Single-step throughput lower than batch traditional kernels
3. `cuLaunchCooperativeKernel` not directly exposed in cudarc (workaround in place)

---

## WebGPU Backend Analysis

### Current State (Event-Driven Only)

**Capabilities**
- Full kernel execution via wgpu 27.0
- Host-side K2K broker functional
- HLC timestamps supported
- Cross-platform (Vulkan, Metal, DX12)

**Fundamental Limitations**
1. **No Persistent Kernels**: WebGPU execution model requires dispatch/wait cycles
2. **No GPU-side K2K**: Shader language doesn't support cross-workgroup communication
3. **No Cooperative Groups**: No grid-wide synchronization primitive
4. **No 64-bit Atomics**: Emulated with lo/hi u32 pairs

**Code Generation Status** (`ringkernel-wgpu-codegen` - 50+ tests)
```rust
// WGSL limitations requiring workarounds:
- 64-bit atomics: Emulated (lo/hi split)
- f64: Auto-downcast to f32 with warning
- Persistent kernels: Host-driven dispatch loop
- K2K messaging: Not supported
- Warp operations: Limited subgroup support (18 unimplemented)
```

### Unimplemented GPU Intrinsics (18 total)
- `atomic_add`, `atomic_sub`, `atomic_min`, `atomic_max`
- `atomic_exchange`, `atomic_cas`, `atomic_load`, `atomic_store`
- `warp_shuffle`, `warp_ballot`, `warp_all`, `warp_any`
- `lane_id`, `warp_size`

---

## Metal Backend Analysis

### Current State (Scaffolded Only)

**Existing Code** (`ringkernel-metal/src/` - 566 lines)
- Basic `MetalDevice` wrapper
- Kernel state management stubs
- Memory allocation placeholders

**Missing Components**
1. **Persistent Kernel Implementation**: Entire `persistent.rs` equivalent
2. **MSL Code Generation**: No transpiler exists
3. **GPU-side K2K**: Not started
4. **Mapped Memory**: Not implemented (IOSurface/MTLBuffer shared possible)

**Technical Feasibility**
- Metal supports argument buffers for persistent state
- ICB (Indirect Command Buffers) could enable persistence patterns
- Metal has thread group coordination primitives
- Apple Silicon has unified memory architecture

---

## CPU Backend Analysis

### Current State (Complete for Testing)

**Implementation** (`ringkernel-cpu/src/`)
- Full async kernel simulation
- K2K broker fully functional
- HLC timestamps working
- Suitable for CI/CD and development

**Performance**
- Baseline for GPU speedup comparisons
- ~278 Mcells/s with Rayon parallelization
- Used in all tests where GPU hardware unavailable

---

## Code Generation Analysis

### CUDA Codegen (`ringkernel-cuda-codegen`)

**Kernel Types Supported**
| Type | Description | Status |
|------|-------------|--------|
| Global Kernels | Generic CUDA with indices | ✅ Complete |
| Stencil Kernels | GridPos abstraction | ✅ Complete |
| Ring Kernels | Persistent actor model | ✅ Complete |
| Persistent FDTD | True persistent 3D simulation | ✅ Complete |

**DSL Features** (120+ GPU intrinsics)
- Block/grid indices: `block_idx_x()`, `thread_idx_x()`, etc.
- Control flow: `if/else`, `match` → switch, `for`/`while`/`loop`
- Stencil intrinsics (2D/3D): `pos.north()`, `pos.up()`, etc.
- Shared memory: `__shared__` arrays
- Synchronization: `__syncthreads()`, cooperative groups
- HLC operations: `hlc_tick()`, `hlc_update()`, `hlc_now()`
- K2K messaging: `k2k_send_envelope()`, `k2k_try_recv_envelope()`

### WGSL Codegen (`ringkernel-wgpu-codegen`)

**Feature Parity with CUDA**
| Feature | CUDA | WGSL |
|---------|:----:|:----:|
| Global kernels | ✅ | ✅ |
| Stencil kernels | ✅ | ✅ |
| Ring kernels | ✅ | ⚠️ Host-driven |
| Shared memory | ✅ | ✅ |
| 64-bit atomics | ✅ | ⚠️ Emulated |
| f64 support | ✅ | ❌ |
| K2K messaging | ✅ | ❌ |

---

## Ecosystem Integrations Analysis

### Working Integrations

| Integration | Lines | Status | Notes |
|-------------|-------|--------|-------|
| `persistent.rs` | 820 | ✅ Complete | Core trait + mock |
| `cuda_bridge.rs` | 590 | ✅ Complete | CUDA backend impl |
| `actix.rs` | 750 | ✅ Complete | Actor framework |
| `axum.rs` | 900 | ✅ Complete | REST API |
| `tower.rs` | 900 | ✅ Complete | Service middleware |
| `grpc.rs` | 650 | ⚠️ Partial | Streaming incomplete |
| `metrics.rs` | 350 | ✅ Complete | Prometheus |
| `tracing_ext.rs` | 280 | ✅ Complete | Distributed tracing |

### Scaffolded Integrations

| Integration | Status | Missing |
|-------------|--------|---------|
| `arrow.rs` | Framework only | GPU kernel integration |
| `polars.rs` | Framework only | GPU kernel integration |
| `candle.rs` | Framework only | GPU kernel integration |
| WebSocket | Partial | Handler implementation |
| SSE | Partial | Event streaming handler |

---

## Test Coverage Summary

| Crate | Test Count | Notes |
|-------|------------|-------|
| ringkernel-core | 65 | Core abstractions |
| ringkernel-cuda-codegen | 183 | Most comprehensive |
| ringkernel-procint | 77 | DFG/conformance |
| ringkernel-wavesim3d | 72 | 3D simulation |
| ringkernel-wavesim | 63 | 2D simulation |
| ringkernel-wgpu-codegen | 50 | WGSL transpiler |
| ringkernel-txmon | 40 | Transaction monitoring |
| ringkernel-audio-fft | 32 | Audio processing |
| ringkernel-ecosystem | 30 | Web integrations |
| ringkernel-control-block | 29 | Lifecycle management |
| ringkernel-hlc | 16 | Timestamps |
| ringkernel-derive | 14 | Proc macros |
| ringkernel-cpu | 11 | CPU backend |
| ringkernel-k2k | 11 | Kernel messaging |
| ringkernel-cuda | 6 | GPU execution |
| **Total** | **580+** | |

---

## Technical Debt & TODOs

### Critical Priority
1. **CUDA runtime.rs:191** - Track available kernel slots (shader occupation)
2. **CUDA kernel.rs:397** - Correlation tracking in metadata
3. **CUDA persistent.rs:1182** - Software sync fallback optimization

### Code Generation
4. **ring_kernel.rs:720** - Compute checksum for response validation
5. **persistent_fdtd.rs:714** - Calculate energy for stats

### Architecture
6. **multi_gpu.rs:182** - Kernel migration between devices (stub)
7. **wgpu-codegen/shared.rs:50** - Higher-dimensional shared memory arrays

---

## Architectural Recommendations

### Immediate Priorities
1. **Metal Backend**: Full implementation required for Apple ecosystem
2. **SSE/WebSocket**: Complete streaming handlers in ecosystem
3. **Multi-GPU**: Implement kernel migration infrastructure

### Medium-Term
1. **WGSL Atomics**: Implement remaining 18 GPU intrinsics
2. **Arrow/Polars/Candle**: GPU kernel integration
3. **gRPC Streaming**: Complete bidirectional streaming

### Long-Term
1. **Distributed Kernels**: Cross-node K2K messaging
2. **Fault Tolerance**: Checkpoint/restore for persistent kernels
3. **Dynamic Scaling**: Runtime topology reconfiguration

---

## Conclusion

RingKernel has a mature CUDA implementation with production-ready persistent kernels achieving 11,327x faster command injection. The WebGPU backend is functional but limited by language constraints. Metal and several ecosystem integrations require significant development. The codebase has excellent test coverage (580+ tests) and well-documented performance characteristics.
