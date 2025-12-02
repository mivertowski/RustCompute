# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RingKernel is a GPU-native persistent actor model framework for Rust. It enables GPU-accelerated actor systems with persistent kernels, lock-free message passing, and hybrid logical clocks (HLC) for causal ordering.

## Build Commands

```bash
# Build entire workspace
cargo build --workspace

# Build with specific GPU backend
cargo build --workspace --features cuda
cargo build --workspace --features wgpu

# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p ringkernel-core

# Run a single test
cargo test -p ringkernel-core test_name

# Run CUDA GPU execution tests (requires NVIDIA GPU)
cargo test -p ringkernel-cuda --test gpu_execution_verify

# Run WebGPU tests (requires GPU)
cargo test -p ringkernel-wgpu --features wgpu-tests -- --ignored

# Run benchmarks
cargo bench --package ringkernel

# Run examples
cargo run -p ringkernel --example hello_kernel
cargo run -p ringkernel --example kernel_to_kernel
cargo run -p ringkernel --example wgpu_hello --features wgpu
```

## Architecture

### Workspace Structure

The project is a Cargo workspace with these crates:

- **`ringkernel`** - Main facade crate that re-exports everything; entry point for users
- **`ringkernel-core`** - Core traits and types (RingMessage, MessageQueue, HlcTimestamp, ControlBlock, RingContext, RingKernelRuntime, K2K messaging, PubSub)
- **`ringkernel-derive`** - Proc macros: `#[derive(RingMessage)]`, `#[ring_kernel]`, `#[derive(GpuType)]`
- **`ringkernel-cpu`** - CPU backend implementation (always available, used for testing/fallback)
- **`ringkernel-cuda`** - NVIDIA CUDA backend (feature-gated)
- **`ringkernel-wgpu`** - WebGPU cross-platform backend (feature-gated)
- **`ringkernel-metal`** - Apple Metal backend (feature-gated, macOS only, scaffolded)
- **`ringkernel-codegen`** - GPU kernel code generation
- **`ringkernel-cuda-codegen`** - Rust-to-CUDA transpiler for writing GPU kernels in Rust DSL
- **`ringkernel-ecosystem`** - Integration utilities
- **`ringkernel-audio-fft`** - Example application: GPU-accelerated audio FFT processing
- **`ringkernel-wavesim`** - Example application: 2D acoustic wave simulation with GPU-accelerated FDTD

### Core Abstractions (in ringkernel-core)

- **`RingMessage`** trait - Messages serializable via rkyv for zero-copy GPU transfer
- **`MessageQueue`** - Lock-free ring buffer for host↔GPU message passing
- **`RingKernelRuntime`** trait - Backend-agnostic runtime interface
- **`KernelHandle`** - Handle to manage kernel lifecycle (launch, activate, terminate)
- **`HlcTimestamp`/`HlcClock`** - Hybrid logical clocks for causal ordering
- **`ControlBlock`** - 128-byte GPU-resident structure managing kernel state
- **`RingContext`** - GPU intrinsics facade passed to kernel handlers
- **`K2KBroker`/`K2KEndpoint`** - Kernel-to-kernel direct messaging
- **`PubSubBroker`** - Topic-based publish/subscribe with wildcards

### Backend System

Backends implement `RingKernelRuntime` trait. Selection via features:
- `cpu` (default) - Always available
- `cuda` - Requires NVIDIA GPU + CUDA toolkit
- `wgpu` - Cross-platform via WebGPU (Vulkan, Metal, DX12)
- `metal` - macOS/iOS only (scaffolded)

Auto-detection: `Backend::Auto` tries CUDA → Metal → WebGPU → CPU.

### Proc Macros (ringkernel-derive)

```rust
// Message definition with field annotations
#[derive(RingMessage)]
#[message(type_id = 1)]  // optional explicit type ID
struct MyMessage {
    #[message(id)]          // MessageId field
    id: MessageId,
    #[message(correlation)] // optional correlation tracking
    correlation: CorrelationId,
    #[message(priority)]    // optional priority
    priority: Priority,
    payload: Vec<f32>,
}

// Kernel handler definition
#[ring_kernel(id = "processor", mode = "persistent", block_size = 128)]
async fn handle(ctx: &mut RingContext, msg: MyMessage) -> MyResponse { ... }

// GPU-compatible types
#[derive(GpuType)]
struct Matrix4x4 {
    data: [f32; 16],
}
```

### CUDA Code Generation (ringkernel-cuda-codegen)

Write GPU kernels in Rust DSL and transpile to CUDA C:

```rust
use ringkernel_cuda_codegen::{transpile_global_kernel, transpile_stencil_kernel, StencilConfig, Grid};
use ringkernel_cuda_codegen::dsl::*;

// Generic CUDA kernel (non-stencil)
let kernel_fn: syn::ItemFn = parse_quote! {
    fn exchange_halos(buffer: &mut [f32], copies: &[u32], num_copies: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= num_copies { return; }
        buffer[copies[(idx * 2 + 1) as usize] as usize] =
            buffer[copies[(idx * 2) as usize] as usize];
    }
};
let cuda_code = transpile_global_kernel(&kernel_fn)?;

// Stencil kernel with GridPos abstraction
let stencil_fn: syn::ItemFn = parse_quote! {
    fn fdtd_step(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
        let laplacian = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p)
                        - 4.0 * p[pos.idx()];
        p_prev[pos.idx()] = 2.0 * p[pos.idx()] - p_prev[pos.idx()] + c2 * laplacian;
    }
};
let config = StencilConfig::new("fdtd").with_tile_size(16, 16).with_halo(1);
let cuda_code = transpile_stencil_kernel(&stencil_fn, &config)?;
```

**DSL Features:**
- Block/grid indices: `block_idx_x()`, `thread_idx_x()`, `block_dim_x()`, `grid_dim_x()`, etc.
- Early return: `if cond { return; }`
- Match expressions → switch/case
- Stencil intrinsics: `pos.north(buf)`, `pos.south(buf)`, `pos.east(buf)`, `pos.west(buf)`, `pos.at(buf, dx, dy)`
- Type inference for int vs float expressions

### K2K (Kernel-to-Kernel) Messaging

Direct messaging between kernels without going through the host application:

```rust
// Runtime with K2K enabled (default)
let runtime = CpuRuntime::new().await?;
assert!(runtime.is_k2k_enabled());

// Access the broker
let broker = runtime.k2k_broker().unwrap();

// Register kernels and send messages
let endpoint = broker.register(kernel_id);
endpoint.send(destination_id, envelope).await?;
```

### Serialization

Uses rkyv for zero-copy serialization. Messages must derive `rkyv::Archive`, `rkyv::Serialize`, `rkyv::Deserialize`. The `RingMessage` derive macro generates serialization methods automatically.

### Feature Flags

Main crate (`ringkernel`) features:
- `cpu` (default) - CPU backend
- `cuda` - NVIDIA CUDA backend
- `wgpu` - WebGPU cross-platform backend
- `metal` - Apple Metal backend
- `all-backends` - All GPU backends

## Testing Patterns

- Unit tests in each crate's `src/` using `#[cfg(test)]`
- Integration tests use `#[tokio::test]` for async runtime
- GPU tests use `#[ignore]` attribute and feature flags (`wgpu-tests`, `cuda`)
- Property-based testing via `proptest` for queue/serialization invariants
- Benchmarks in `crates/ringkernel/benches/` using Criterion

### Test Count Summary

240+ tests across the workspace:
- ringkernel-core: 65 tests
- ringkernel-cpu: 11 tests
- ringkernel-cuda: 6 GPU execution tests
- ringkernel-cuda-codegen: 39 tests
- ringkernel-derive: 14 macro tests
- ringkernel-wavesim: 46 tests
- k2k_integration: 11 tests
- control_block: 29 tests
- hlc: 16 tests
- ringkernel-audio-fft: 32 tests
- Plus additional integration and doc tests

## Important Patterns

**Queue capacity must be power of 2:**
```rust
// Correct
let options = LaunchOptions::default().with_queue_capacity(1024);

// Or calculate it
let capacity = desired_size.next_power_of_two();
```

**Avoid double activation:**
```rust
// Kernels auto-activate by default, so this will error:
let kernel = runtime.launch("test", LaunchOptions::default()).await?;
kernel.activate().await?;  // Error: "Active to Active"

// Use without_auto_activate for manual control:
let kernel = runtime.launch("test",
    LaunchOptions::default().without_auto_activate()
).await?;
kernel.activate().await?;  // OK
```

**Result type in examples:**
```rust
// The prelude exports Result<T> which conflicts with std::result::Result<T, E>
// Use explicit type in main functions:
#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // ...
}
```
