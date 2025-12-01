# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RingKernel (RustCompute) is a GPU-native persistent actor model framework for Rust - a port of DotCompute's Ring Kernel system. It enables GPU-accelerated actor systems with persistent kernels, lock-free message passing, and hybrid logical clocks (HLC) for causal ordering.

## Build Commands

```bash
# Build entire workspace
cargo build --workspace

# Build with specific GPU backend
cargo build --workspace --features cuda
cargo build --workspace --features wgpu
cargo build --workspace --features metal

# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p ringkernel-core

# Run a single test
cargo test -p ringkernel-core test_name

# Run benchmarks
cargo bench --package ringkernel

# Run a specific benchmark
cargo bench --package ringkernel --bench serialization

# Run examples
cargo run -p ringkernel --example hello_kernel
cargo run -p ringkernel --example vector_add
cargo run -p ringkernel --example ping_pong
```

## Architecture

### Workspace Structure

The project is a Cargo workspace with these crates:

- **`ringkernel`** - Main facade crate that re-exports everything; entry point for users
- **`ringkernel-core`** - Core traits and types (RingMessage, MessageQueue, HlcTimestamp, ControlBlock, RingContext, RingKernelRuntime)
- **`ringkernel-derive`** - Proc macros: `#[derive(RingMessage)]`, `#[ring_kernel]`, `#[derive(GpuType)]`
- **`ringkernel-cpu`** - CPU backend implementation (always available, used for testing/fallback)
- **`ringkernel-cuda`** - NVIDIA CUDA backend (feature-gated)
- **`ringkernel-metal`** - Apple Metal backend (feature-gated, macOS only)
- **`ringkernel-wgpu`** - WebGPU cross-platform backend (feature-gated)
- **`ringkernel-codegen`** - GPU kernel code generation
- **`ringkernel-ecosystem`** - Integration utilities
- **`ringkernel-audio-fft`** - Example application: GPU-accelerated audio FFT processing

### Core Abstractions (in ringkernel-core)

- **`RingMessage`** trait - Messages serializable via rkyv for zero-copy GPU transfer
- **`MessageQueue`** - Lock-free ring buffer for host↔GPU message passing
- **`RingKernelRuntime`** trait - Backend-agnostic runtime interface
- **`KernelHandle`** - Handle to manage kernel lifecycle (launch, activate, terminate)
- **`HlcTimestamp`/`HlcClock`** - Hybrid logical clocks for causal ordering
- **`ControlBlock`** - 128-byte GPU-resident structure managing kernel state
- **`RingContext`** - GPU intrinsics facade passed to kernel handlers

### Backend System

Backends implement `RingKernelRuntime` trait. Selection via features:
- `cpu` (default) - Always available
- `cuda` - Requires NVIDIA GPU + CUDA toolkit
- `wgpu` - Cross-platform via WebGPU
- `metal` - macOS/iOS only

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
```

### Serialization

Uses rkyv for zero-copy serialization. Messages must derive `rkyv::Archive`, `rkyv::Serialize`, `rkyv::Deserialize`. The `RingMessage` derive macro generates serialization methods automatically.

### Feature Flags

Main crate (`ringkernel`) features:
- `cpu` (default), `cuda`, `wgpu`, `metal`, `all-backends`
- `telemetry`, `k2k-messaging`, `topic-pubsub`, `multi-gpu` (planned)

## Testing Patterns

- Unit tests in each crate's `src/` using `#[cfg(test)]`
- Integration tests use `#[tokio::test]` for async runtime
- Property-based testing via `proptest` for queue/serialization invariants
- Benchmarks in `crates/ringkernel/benches/` using Criterion
