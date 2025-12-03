---
layout: default
title: Architecture Overview
nav_order: 2
---

# Architecture Overview

## Current Status

RingKernel is under active development. The core runtime, CPU backend, CUDA backend, and WebGPU backend are functional with verified GPU execution.

**Working today:**
- Runtime creation and kernel lifecycle management
- CPU backend (fully functional)
- CUDA backend (verified with real PTX kernels, ~75M elements/sec)
- WebGPU backend (cross-platform via wgpu)
- Message passing infrastructure (queues, serialization, HLC timestamps)
- Pub/Sub messaging with topic wildcards
- K2K (kernel-to-kernel) direct messaging
- Telemetry and metrics collection
- Rust-to-CUDA transpiler (ringkernel-cuda-codegen)
- 11 working examples + WaveSim showcase app

**In progress:**
- Metal backend (scaffolded)

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
┌─────────────────────────────────────────────────────────────────────────┐
│                         HOST (CPU) SIDE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐   │
│  │  Application    │───▶│ RingKernelRuntime│───▶│  Message Bridge   │   │
│  │  (async/await)  │    │  (lifecycle mgmt)│    │  (Host↔GPU DMA)   │   │
│  └─────────────────┘    └──────────────────┘    └─────────┬─────────┘   │
│                                │                          │              │
│                         ┌──────┴──────┐                   │              │
│                         ▼             ▼                   ▼              │
│                 ┌───────────┐  ┌───────────┐    ┌─────────────────┐     │
│                 │  Launch   │  │ Terminate │    │  Serialization  │     │
│                 │  Options  │  │  Handler  │    │  (rkyv/zerocopy)│     │
│                 └───────────┘  └───────────┘    └─────────────────┘     │
│                                                                          │
├──────────────────────────────────PCIe────────────────────────────────────┤
│                                                                          │
│                         DEVICE (GPU) SIDE                                │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    PERSISTENT KERNEL                             │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │    │
│  │  │ Control     │  │ Input Queue │  │ Message Handler         │  │    │
│  │  │ Block       │  │ (lock-free) │  │ (user-defined logic)    │  │    │
│  │  │ (128 bytes) │  │             │  │                         │  │    │
│  │  │ - is_active │  │ head ──────▶│  │ process(ctx, msg) {     │  │    │
│  │  │ - terminate │  │ tail ◀──────│  │   ctx.sync_threads();   │  │    │
│  │  │ - msg_count │  │ buffer[]    │  │   // GPU computation    │  │    │
│  │  │ - errors    │  │             │  │   ctx.enqueue_output(); │  │    │
│  │  └─────────────┘  └─────────────┘  │ }                       │  │    │
│  │                                     └─────────────────────────┘  │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │    │
│  │  │ Telemetry   │  │Output Queue │  │ K2K Messaging           │  │    │
│  │  │ Buffer      │  │ (lock-free) │  │ (kernel-to-kernel)      │  │    │
│  │  │ (64 bytes)  │  │             │  │                         │  │    │
│  │  │ - processed │  │ head ◀──────│  │ send_to_kernel("other") │  │    │
│  │  │ - latency   │  │ tail ──────▶│  │ recv_from_kernel()      │  │    │
│  │  │ - errors    │  │ buffer[]    │  │                         │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Kernel Lifecycle State Machine

Kernels follow a deterministic state machine. By default, kernels auto-activate on launch.

```
        ┌──────────┐
        │ Launched │
        └────┬─────┘
             │ activate()
             ▼
        ┌──────────┐
   ┌────│  Active  │────┐
   │    └────┬─────┘    │
   │         │          │ deactivate() / suspend()
   │         │          ▼
   │         │    ┌────────────┐
   │         │    │Deactivated │
   │         │    └─────┬──────┘
   │         │          │
   │         │ terminate()
   │         ▼          │
   │    ┌──────────┐    │
   └───▶│Terminated│◀───┘
        └──────────┘
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

### Host → GPU (Input)

```
1. Application calls kernel.send(message)
2. Message serialized via rkyv (zero-copy)
3. Bridge copies to pinned host buffer
4. DMA transfer to GPU input queue
5. Kernel dequeues and processes
```

### GPU → Host (Output)

```
1. Kernel calls ctx.enqueue_output(response)
2. Message written to GPU output queue
3. Bridge polls/DMA copies to host
4. Deserialized via rkyv
5. Future resolved, application receives response
```

### GPU → GPU (K2K Messaging)

```
1. Kernel A calls ctx.send_to_kernel("B", msg)
2. Routing table lookup (O(1) hash)
3. Message copied to Kernel B's K2K queue
4. Kernel B calls ctx.try_receive_from_kernel("A")
5. Direct GPU memory access (no PCIe)
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

RingKernel supports multiple GPU backends through the `Backend` enum:

| Backend | Platform | Status | Notes |
|---------|----------|--------|-------|
| **CPU** | All | **Working** | Full functionality, ideal for development |
| **CUDA** | Linux, Windows | **Working** | Verified GPU execution, requires CUDA toolkit |
| **Metal** | macOS, iOS | Scaffolded | API defined, implementation pending |
| **WebGPU** | Cross-platform | **Working** | Via wgpu (Vulkan, Metal, DX12) |

### Backend Selection

```rust
// Auto-detect best available backend
let runtime = RingKernel::builder()
    .backend(Backend::Auto)  // CUDA → Metal → WebGPU → CPU
    .build()
    .await?;

// Force specific backend
let runtime = RingKernel::builder()
    .backend(Backend::Cuda)
    .build()
    .await?;

// Check active backend
println!("Using backend: {:?}", runtime.backend());
```

### Performance (CUDA backend, RTX 2000 Ada)

- Vector operations: ~75M elements/sec
- Memory bandwidth: 7.6 GB/s HtoD, 1.4 GB/s DtoH

---

## Next: [Crate Structure](./02-crate-structure.md)
