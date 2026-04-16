---
layout: default
title: Architecture Overview
nav_order: 2
---

# Architecture Overview

## Current Status

RingKernel v1.0.0 is a CUDA-focused persistent GPU actor framework. The core runtime, CPU backend, and NVIDIA CUDA backend are production-quality with H100-verified performance.

**Working today:**
- Runtime creation and kernel lifecycle management
- CPU backend (fully functional, testing and development fallback)
- CUDA backend (persistent kernels, cooperative groups, H100 Hopper features)
- Thread Block Clusters with DSMEM messaging (Hopper SM 9.0)
- cluster.sync() for intra-GPC synchronization (2.98x faster than grid.sync())
- Green Contexts for SM partitioning
- Async memory pools (116.9x faster than cuMemAlloc)
- Lock-free SPSC/MPSC queues (55 ns per message, 0.05% CV over 60 seconds)
- Actor lifecycle on GPU (create, destroy, restart, supervise)
- Message passing infrastructure (queues, serialization, HLC timestamps)
- Pub/Sub messaging with topic wildcards
- K2K (kernel-to-kernel) direct messaging via device memory and DSMEM
- Telemetry and metrics collection with NVTX profiling
- Rust-to-CUDA transpiler with 155+ intrinsics (ringkernel-cuda-codegen)
- Size-stratified memory pools with pressure handling
- Global reduction primitives with multi-phase execution
- Multi-kernel dispatch with domain-based routing
- Queue tiering for throughput-based capacity selection
- Enterprise features (auth, rate limiting, TLS, multi-tenancy)
- 20+ working examples
- 4 showcase applications: WaveSim, TxMon, AccNet, ProcInt
- 1,496+ tests across the workspace

## DotCompute Ring Kernel Architecture

The Ring Kernel system implements a **GPU-native actor model** with persistent state. This is a Rust port of DotCompute's Ring Kernel system.

## Component Mapping

| DotCompute Component | Rust Equivalent | Purpose |
|---------------------|-----------------|---------|
| `IRingKernelRuntime` | `RingKernel` struct | Runtime and kernel lifecycle |
| `IRingKernelMessage` | `trait RingMessage` | Type-safe message protocol |
| `IMessageQueue<T>` | `trait MessageQueue<T>` | Lock-free ring buffer |
| `RingKernelContext` | `struct RingContext` | GPU intrinsics facade |
| `RingKernelControlBlock` | `#[repr(C)] struct ControlBlock` | GPU-resident state (128 bytes) |
| `HlcTimestamp` | `struct HlcTimestamp` | Hybrid Logical Clock |
| `MemoryPackSerializer` | `rkyv` / `zerocopy` derive | Zero-copy serialization |

---

## System Architecture Diagram

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
                                        |  DSMEM (Hopper clusters)           |
                                        +------------------------------------+
```

### Hopper Architecture Features

RingKernel leverages H100/Hopper-specific hardware capabilities:

- **Thread Block Clusters**: Groups of thread blocks co-located on the same GPC, enabling DSMEM-based communication without global memory
- **Distributed Shared Memory (DSMEM)**: Direct shared memory access between blocks in a cluster (~30 cycle latency vs ~400 cycles for global memory)
- **cluster.sync()**: Intra-GPC synchronization at 0.628 us/sync (2.98x faster than grid.sync())
- **Green Contexts**: SM partitioning for persistent actor isolation
- **TMA (Tensor Memory Accelerator)**: Hardware-accelerated bulk data movement
- **Async Memory Pools**: Stream-ordered allocation at 878 ns (116.9x faster than cuMemAlloc)

---

## Kernel Lifecycle State Machine

Kernels follow a deterministic state machine. By default, kernels auto-activate on launch.

```
        +-----------+
        | Launched  |
        +-----+-----+
              | activate()
              v
        +-----------+
   +----+  Active   +----+
   |    +-----+-----+    |
   |          |           | deactivate() / suspend()
   |          |           v
   |          |    +-----------+
   |          |    |Deactivated|
   |          |    +-----+-----+
   |          |          |
   |          | terminate()
   |          v          |
   |    +-----------+    |
   +--->+Terminated +<---+
        +-----------+
```

### API Usage

```rust
// Launch with auto-activation (default)
let kernel = runtime.launch("processor", LaunchOptions::default()).await?;
assert!(kernel.is_active());

// Launch without auto-activation
let kernel = runtime.launch("processor",
    LaunchOptions::default().without_auto_activate()
).await?;
kernel.activate().await?;

// Suspend and resume
kernel.suspend().await?;  // alias for deactivate()
kernel.resume().await?;   // alias for activate()

// Check state
println!("State: {:?}", kernel.state());
println!("Active: {}", kernel.is_active());

// Clean shutdown
kernel.terminate().await?;
```

---

## Message Flow

### Host -> GPU (Input)

```
1. Application calls kernel.send(message)
2. Message serialized via rkyv (zero-copy)
3. Bridge copies to pinned host buffer
4. DMA transfer to GPU input queue
5. Kernel dequeues and processes
```

### GPU -> Host (Output)

```
1. Kernel calls ctx.enqueue_output(response)
2. Message written to GPU output queue
3. Bridge polls/DMA copies to host
4. Deserialized via rkyv
5. Future resolved, application receives response
```

### GPU -> GPU (K2K Messaging)

```
1. Kernel A calls ctx.send_to_kernel("B", msg)
2. Routing table lookup (O(1) hash)
3. Message copied to Kernel B's K2K queue
4. Kernel B calls ctx.try_receive_from_kernel("A")
5. Direct GPU memory access (no PCIe)
```

### Intra-Cluster K2K (Hopper DSMEM)

```
1. Actor A writes to distributed shared memory
2. cluster.sync() ensures visibility
3. Actor B reads from DSMEM (~30 cycle latency)
4. No global memory traffic required
```

---

## Memory Layout Requirements

### Control Block (128 bytes, cache-line aligned)

```rust
#[repr(C, align(128))]
pub struct ControlBlock {
    // Flags (offset 0-15)
    pub is_active: AtomicI32,           // 0: inactive, 1: processing
    pub should_terminate: AtomicI32,    // 0: run, 1: shutdown
    pub has_terminated: AtomicI32,      // 0: running, 1: done, 2: relaunchable
    pub errors_encountered: AtomicI32,

    // Counters (offset 16-31)
    pub messages_processed: AtomicI64,
    pub last_activity_ticks: AtomicI64,

    // Input queue descriptors (offset 32-63)
    pub input_queue_head_ptr: u64,      // Device pointer
    pub input_queue_tail_ptr: u64,
    pub input_queue_buffer_ptr: u64,
    pub input_queue_capacity: i32,
    pub input_queue_message_size: i32,

    // Output queue descriptors (offset 64-95)
    pub output_queue_head_ptr: u64,
    pub output_queue_tail_ptr: u64,
    pub output_queue_buffer_ptr: u64,
    pub output_queue_capacity: i32,
    pub output_queue_message_size: i32,

    // Reserved (offset 96-127)
    _reserved: [u64; 4],
}
```

### Telemetry Buffer (64 bytes, cache-line aligned)

```rust
#[repr(C, align(64))]
pub struct TelemetryBuffer {
    pub messages_processed: AtomicU64,
    pub messages_dropped: AtomicU64,
    pub last_processed_timestamp: AtomicI64,
    pub queue_depth: AtomicI32,
    pub total_latency_nanos: AtomicU64,
    pub max_latency_nanos: AtomicU64,
    pub min_latency_nanos: AtomicU64,
    pub error_code: AtomicI32,
}
```

---

## Backend Abstraction

RingKernel supports two GPU backends through the `Backend` enum:

| Backend | Platform | Status | Notes |
|---------|----------|--------|-------|
| **CPU** | All | **Stable** | Full functionality, ideal for development and testing |
| **CUDA** | Linux, Windows | **Stable, H100-verified** | Persistent kernels, cooperative groups, Hopper features |

### Backend Selection

```rust
// Auto-detect best available backend
let runtime = RingKernel::builder()
    .backend(Backend::Auto)  // CUDA -> CPU
    .build()
    .await?;

// Force CUDA backend
let runtime = RingKernel::builder()
    .backend(Backend::Cuda)
    .build()
    .await?;

// Check active backend
println!("Using backend: {:?}", runtime.backend());
```

### Performance (CUDA backend, H100 NVL)

- Persistent actor injection: 55 ns (8,698x faster than cuLaunchKernel)
- Lock-free queue throughput: 13.83 Mmsg/s (64B payload)
- Sustained throughput: 5.54 Mops/s over 60 seconds (CV 0.05%)
- Cluster sync: 0.628 us (2.98x faster than grid.sync())
- Async memory alloc: 878 ns (116.9x faster than cuMemAlloc)

---

## Next: [Crate Structure](./02-crate-structure.md)
