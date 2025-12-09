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
cargo run -p ringkernel --example basic_hello_kernel
cargo run -p ringkernel --example kernel_to_kernel
cargo run -p ringkernel --example wgpu_hello --features wgpu

# Run CUDA codegen examples
cargo run -p ringkernel --example global_kernel
cargo run -p ringkernel --example stencil_kernel
cargo run -p ringkernel --example ring_kernel_codegen

# Run educational modes example
cargo run -p ringkernel --example educational_modes

# Run transaction monitoring GUI
cargo run -p ringkernel-txmon --release --features cuda-codegen

# Run txmon GPU benchmark (requires NVIDIA GPU)
cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen

# Run process intelligence GUI
cargo run -p ringkernel-procint --release

# Run process intelligence benchmark
cargo run -p ringkernel-procint --bin procint-benchmark --release
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
- **`ringkernel-wgpu-codegen`** - Rust-to-WGSL transpiler for writing GPU kernels in Rust DSL (WebGPU backend)
- **`ringkernel-ecosystem`** - Integration utilities
- **`ringkernel-audio-fft`** - Example application: GPU-accelerated audio FFT processing
- **`ringkernel-wavesim`** - Example application: 2D acoustic wave simulation with GPU-accelerated FDTD and educational simulation modes
- **`ringkernel-txmon`** - Showcase application: GPU-accelerated transaction monitoring with real-time fraud detection GUI
- **`ringkernel-accnet`** - Showcase application: GPU-accelerated accounting network visualization
- **`ringkernel-procint`** - Showcase application: GPU-accelerated process intelligence with DFG mining, pattern detection, and conformance checking

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

Write GPU kernels in Rust DSL and transpile to CUDA C. Supports three kernel types:

#### 1. Global Kernels (Generic CUDA)

```rust
use ringkernel_cuda_codegen::{transpile_global_kernel};
use syn::parse_quote;

let kernel_fn: syn::ItemFn = parse_quote! {
    fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        y[idx as usize] = a * x[idx as usize] + y[idx as usize];
    }
};
let cuda_code = transpile_global_kernel(&kernel_fn)?;
```

#### 2. Stencil Kernels (GridPos Abstraction)

```rust
use ringkernel_cuda_codegen::{transpile_stencil_kernel, StencilConfig};

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

#### 3. Ring Kernels (Persistent Actor Model)

```rust
use ringkernel_cuda_codegen::{transpile_ring_kernel, RingKernelConfig};

let handler: syn::ItemFn = parse_quote! {
    fn process_message(ctx: &RingContext, msg: &Request) -> Response {
        let tid = ctx.global_thread_id();
        ctx.sync_threads();
        let result = msg.value * 2.0;
        Response { value: result, id: tid as u64 }
    }
};

let config = RingKernelConfig::new("processor")
    .with_block_size(128)
    .with_queue_capacity(1024)
    .with_hlc(true)   // Enable Hybrid Logical Clocks
    .with_k2k(true);  // Enable Kernel-to-Kernel messaging

let cuda_code = transpile_ring_kernel(&handler, &config)?;
```

**DSL Features:**
- Block/grid indices: `block_idx_x()`, `thread_idx_x()`, `block_dim_x()`, `grid_dim_x()`, `warp_size()`, etc.
- Control flow: `if/else`, `match` → switch/case, early `return`
- Loops: `for i in 0..n`, `while cond`, `loop` with `break`/`continue`
- Stencil intrinsics (2D): `pos.north(buf)`, `pos.south(buf)`, `pos.east(buf)`, `pos.west(buf)`, `pos.at(buf, dx, dy)`
- Stencil intrinsics (3D): `pos.up(buf)`, `pos.down(buf)`, `pos.at(buf, dx, dy, dz)` for volumetric kernels
- Shared memory: `__shared__` arrays and tiles with `SharedMemoryConfig`
- Struct literals: `Point { x: 1.0, y: 2.0 }` → C compound literals
- Reference expressions: `&arr[idx]` → pointer to element with automatic `->` operator for field access
- 120+ GPU intrinsics across 13 categories (synchronization, atomics, math, trig, hyperbolic, exponential, classification, warp, bit manipulation, memory, special, index, timing)

**Ring Kernel Features:**
- Persistent message loop with ControlBlock lifecycle management
- RingContext method inlining (`ctx.thread_id()` → `threadIdx.x`)
- HLC clock operations: `hlc_tick()`, `hlc_update()`, `hlc_now()`
- K2K messaging: `k2k_send()`, `k2k_try_recv()`, `k2k_peek()`, `k2k_pending_count()`
- Queue intrinsics: `enqueue_response()`, `input_queue_empty()`, etc.
- Automatic termination handling and cleanup

### WGSL Code Generation (ringkernel-wgpu-codegen)

Write GPU kernels in Rust DSL and transpile to WGSL for WebGPU. Provides full parity with CUDA codegen:

```rust
use ringkernel_wgpu_codegen::{transpile_global_kernel, transpile_stencil_kernel, StencilConfig};
use syn::parse_quote;

// Global kernel
let kernel_fn: syn::ItemFn = parse_quote! {
    fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        y[idx as usize] = a * x[idx as usize] + y[idx as usize];
    }
};
let wgsl_code = transpile_global_kernel(&kernel_fn)?;

// Stencil kernel (same API as CUDA)
let config = StencilConfig::new("fdtd").with_tile_size(16, 16).with_halo(1);
let wgsl_code = transpile_stencil_kernel(&stencil_fn, &config)?;
```

**WGSL Limitations vs CUDA (handled with workarounds):**
- No 64-bit atomics: Emulated using lo/hi u32 pairs
- No f64: Downcast to f32 with warning
- No persistent kernels: Host-driven dispatch loop emulation
- No K2K messaging: Not supported in WebGPU execution model

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

470+ tests across the workspace:
- ringkernel-core: 65 tests
- ringkernel-cpu: 11 tests
- ringkernel-cuda: 6 GPU execution tests
- ringkernel-cuda-codegen: 171 tests (loops, shared memory, ring kernels, K2K, reference expressions, 120+ GPU intrinsics)
- ringkernel-wgpu-codegen: 50 tests (types, intrinsics, transpiler, validation)
- ringkernel-derive: 14 macro tests
- ringkernel-wavesim: 49 tests (including educational modes)
- ringkernel-txmon: 40 tests (GPU types, batch kernel, stencil kernel, ring kernel backends)
- ringkernel-procint: 77 tests (DFG construction, pattern detection, partial order, conformance checking)
- k2k_integration: 11 tests
- control_block: 29 tests
- hlc: 16 tests
- ringkernel-audio-fft: 32 tests
- Plus additional integration and doc tests

### GPU Benchmark Results (RTX Ada)

The `ringkernel-cuda-codegen` transpiler generates CUDA code that achieves impressive performance:

| Backend | Throughput | Batch Time | Speedup vs CPU |
|---------|------------|------------|----------------|
| CUDA Codegen (1M floats) | ~93B elem/sec | 0.5 µs | 12,378x |
| CUDA SAXPY PTX | ~77B elem/sec | 0.6 µs | 10,258x |
| GPU Stencil (CPU fallback) | ~15.7M TPS | 262 µs | 2.09x |
| CPU Baseline | ~7.5M TPS | 547 µs | 1.00x |

Run benchmark: `cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen`

### WaveSim Educational Modes

The WaveSim application includes educational simulation modes that visually demonstrate the evolution of parallel computing:

```rust
use crate::simulation::{SimulationMode, EducationalProcessor};

// Available modes:
// - Standard: Full-speed parallel (default)
// - CellByCell: 1950s sequential processing
// - RowByRow: 1970s vector processing (Cray-style)
// - ChaoticParallel: 1990s uncoordinated parallelism (shows race conditions)
// - SynchronizedParallel: 2000s barrier-synchronized parallelism
// - ActorBased: Modern tile-based actors with HLC

let mut processor = EducationalProcessor::new(SimulationMode::CellByCell);
processor.cells_per_frame = 16; // Cells processed per animation frame
```

Visual indicators show:
- **Green cells**: Currently being processed
- **Yellow row**: Active row (RowByRow mode)
- **Cyan tile with border**: Active tile (ActorBased mode)

## Publishing

All crates are published to crates.io under the `ringkernel-*` namespace. Use the publish script to handle the dependency order:

```bash
# Check which crates are published vs pending
./scripts/publish.sh --status

# Publish all unpublished crates (auto-skips already published)
./scripts/publish.sh <CRATES_IO_TOKEN>

# Dry run to verify without publishing
./scripts/publish.sh --dry-run
```

The script automatically:
- Skips already-published crates (safe to run multiple times)
- Publishes in correct dependency order (tier by tier)
- Handles crates.io rate limits (~5 crates per 10 minutes)
- Shows status before and after publishing

Crate publishing order:
1. **Tier 1** (no deps): core, cuda-codegen, wgpu-codegen
2. **Tier 2** (depends on core): derive, cpu, cuda, wgpu, metal, codegen, ecosystem, audio-fft
3. **Tier 3** (main crate): ringkernel
4. **Tier 4** (applications): wavesim, txmon, accnet, procint

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
