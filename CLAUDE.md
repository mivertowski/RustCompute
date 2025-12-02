# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RingKernel (RustCompute) is a GPU-native persistent actor model framework for Rust - a port of DotCompute's Ring Kernel system. It enables GPU-accelerated actor systems with persistent kernels, lock-free message passing, and hybrid logical clocks (HLC) for causal ordering.

## Current Status

**Working today:**
- Runtime creation and kernel lifecycle management
- CPU backend (fully functional)
- CUDA backend (verified GPU execution, ~75M elements/sec, 7.6 GB/s HtoD)
- Message passing infrastructure (queues, serialization, HLC timestamps)
- Pub/Sub messaging with topic wildcards
- Telemetry and metrics collection
- 11 working examples demonstrating real-world patterns

**In progress:**
- Metal and WebGPU backends (scaffolded, API defined)
- Proc macro code generation for custom kernels
- K2K (kernel-to-kernel) direct messaging

## Build Commands

```bash
# Build entire workspace
cargo build --workspace

# Build with CUDA backend
cargo build --workspace --features cuda

# Run all tests (no GPU required)
cargo test --workspace --exclude ringkernel-cuda

# Run CUDA tests (requires NVIDIA GPU)
cargo test --features cuda -p ringkernel-cuda

# Verify GPU execution
cargo test --features cuda -p ringkernel-cuda --test gpu_execution_verify

# Run benchmarks
cargo bench --package ringkernel

# Run examples
cargo run -p ringkernel --example basic_hello_kernel
cargo run -p ringkernel --example request_response
cargo run -p ringkernel --example pub_sub
cargo run -p ringkernel --example batch_processor
cargo run -p ringkernel --example telemetry
```

## Architecture

### Workspace Structure

The project is a Cargo workspace with these crates:

| Crate | Status | Description |
|-------|--------|-------------|
| `ringkernel` | Working | Main facade crate (re-exports everything) |
| `ringkernel-core` | Working | Core types: RingKernel, KernelHandle, HLC, PubSub, Telemetry |
| `ringkernel-cpu` | Working | CPU backend (always available) |
| `ringkernel-cuda` | Working | NVIDIA CUDA backend with PTX kernels |
| `ringkernel-metal` | Scaffolded | Apple Metal backend |
| `ringkernel-wgpu` | Scaffolded | WebGPU cross-platform backend |
| `ringkernel-derive` | In Development | Proc macros for message/kernel definitions |
| `ringkernel-codegen` | In Development | GPU kernel code generation |
| `ringkernel-ecosystem` | Working | Integration utilities |
| `ringkernel-audio-fft` | Working | Example: GPU audio FFT processing |

### Core Abstractions (ringkernel-core)

- **`RingKernel`** - Runtime builder and manager
- **`KernelHandle`** - Handle to manage kernel lifecycle
  - `state()`, `is_active()`, `is_terminated()`
  - `activate()`, `deactivate()`, `suspend()`, `resume()`, `terminate()`
- **`LaunchOptions`** - Kernel configuration builder
  - `with_queue_capacity(n)` - Queue size (must be power of 2)
  - `with_grid_size(n)`, `with_block_size(n)` - GPU dimensions
  - `with_priority(p)` - Scheduling hint
  - `without_auto_activate()` - Manual activation control
- **`HlcTimestamp`/`HlcClock`** - Hybrid logical clocks for causal ordering
- **`PubSubBroker`** - Topic-based messaging with wildcards (`*`, `#`)
- **`TelemetryPipeline`/`MetricsCollector`** - Real-time metrics and alerts
- **`priority`** module - Constants: `LOW`, `NORMAL`, `HIGH`, `CRITICAL`

### Backend System

Backends implement the runtime traits. Selection via `Backend` enum:
- `Backend::Auto` - Tries CUDA → Metal → WebGPU → CPU
- `Backend::Cpu` - Always available
- `Backend::Cuda` - Requires NVIDIA GPU + CUDA toolkit

### Important Patterns

**Queue capacity must be power of 2:**
```rust
// Wrong - will panic
let options = LaunchOptions::default().with_queue_capacity(1000);

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

## Examples

11 working examples in `/examples/`:

| Example | Command | Demonstrates |
|---------|---------|--------------|
| `basic_hello_kernel` | `cargo run -p ringkernel --example basic_hello_kernel` | Runtime, lifecycle |
| `kernel_states` | `cargo run -p ringkernel --example kernel_states` | State machine |
| `request_response` | `cargo run -p ringkernel --example request_response` | Correlation IDs, priorities |
| `pub_sub` | `cargo run -p ringkernel --example pub_sub` | Topic wildcards, QoS |
| `axum_api` | `cargo run -p ringkernel --example axum_api` | REST API |
| `batch_processor` | `cargo run -p ringkernel --example batch_processor` | Data pipelines |
| `telemetry` | `cargo run -p ringkernel --example telemetry` | Metrics, alerts |
| `grpc_server` | `cargo run -p ringkernel --example grpc_server` | gRPC patterns |
| `config_management` | `cargo run -p ringkernel --example config_management` | TOML, env vars |
| `ml_pipeline` | `cargo run -p ringkernel --example ml_pipeline` | ML inference |
| `multi_gpu` | `cargo run -p ringkernel --example multi_gpu` | Load balancing |

## Testing Patterns

- Unit tests in each crate's `src/` using `#[cfg(test)]`
- Integration tests use `#[tokio::test]` for async runtime
- GPU execution tests in `ringkernel-cuda/tests/gpu_execution_verify.rs`
- Benchmarks in `crates/ringkernel/benches/` using Criterion
