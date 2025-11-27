# RingKernel (RustCompute)

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

**GPU-native persistent actor model framework for Rust** - A port of DotCompute's Ring Kernel system.

RingKernel enables GPU-accelerated actor systems with persistent kernels, lock-free message passing, and hybrid logical clocks for causal ordering across distributed GPU computations.

## Features

- 🚀 **Persistent GPU-resident state** across kernel invocations
- 🔄 **Lock-free message passing** between kernels (K2K messaging)
- ⏱️ **Hybrid Logical Clocks (HLC)** for temporal ordering and causality
- 🎯 **Multiple GPU backends**: CUDA, Metal, WebGPU, CPU (fallback)
- 📦 **Zero-copy serialization** via rkyv/zerocopy
- 🦀 **Type-safe Rust** with proc macro support

## Quick Start

Add RingKernel to your `Cargo.toml`:

```toml
[dependencies]
ringkernel = "0.1"
tokio = { version = "1", features = ["full"] }
```

Basic usage:

```rust
use ringkernel::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create runtime with auto-detected backend
    let runtime = RingKernel::builder()
        .backend(Backend::Auto)
        .build()
        .await?;

    // Launch a kernel
    let kernel = runtime.launch("my_kernel", LaunchOptions::default()).await?;

    // Kernel is automatically activated
    println!("Kernel state: {:?}", kernel.status().state);

    // Terminate when done
    kernel.terminate().await?;
    runtime.shutdown().await?;

    Ok(())
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Host (CPU)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Application │──│   Runtime   │──│   Message Bridge    │  │
│  │  (async)    │  │  (tokio)    │  │  (DMA transfers)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │ PCIe / Unified Memory
┌────────────────────────────┴────────────────────────────────┐
│                    Device (GPU)                             │
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │ Control Block │  │   Input Queue   │  │ Output Queue  │  │
│  │   (128 B)     │  │  (lock-free)    │  │ (lock-free)   │  │
│  └───────────────┘  └─────────────────┘  └───────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Persistent Kernel                          ││
│  │     (your code runs continuously on GPU)               ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Backends

| Backend | Platform | Persistent Kernels | Status |
|---------|----------|-------------------|--------|
| **CPU** | All | Simulated | ✅ Ready |
| **CUDA** | Linux, Windows | ✅ Native | ✅ Ready |
| **Metal** | macOS, iOS | Event-driven | ✅ Ready |
| **WebGPU** | Cross-platform | Event-driven | ✅ Ready |

Enable backends via Cargo features:

```toml
[dependencies]
ringkernel = { version = "0.1", features = ["cuda", "wgpu"] }
```

## Core Concepts

### Messages

Messages are defined using the `RingMessage` trait or derive macro:

```rust
use ringkernel::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Archive, Serialize, Deserialize)]
struct AddRequest {
    a: Vec<f32>,
    b: Vec<f32>,
}

// Implement RingMessage trait for your types
impl RingMessage for AddRequest {
    fn message_type() -> u64 { 0x1001 }
    fn message_id(&self) -> MessageId { MessageId::generate() }
    // ... other methods
}
```

### Hybrid Logical Clocks

HLC provides causal ordering across distributed GPU kernels:

```rust
use ringkernel::prelude::*;

let clock = HlcClock::new(1); // Node ID = 1

// Generate timestamps
let ts1 = clock.tick();
let ts2 = clock.tick();
assert!(ts1 < ts2); // Total ordering guaranteed

// Update from received message
let remote_ts = HlcTimestamp::new(100, 5, 2);
let updated = clock.update(&remote_ts)?;
assert!(updated > remote_ts); // Causality preserved
```

### Control Block

The 128-byte control block manages kernel lifecycle:

```rust
use ringkernel::prelude::*;

let control = ControlBlock::with_capacities(1024, 1024);
assert!(!control.is_active());
assert!(control.input_queue_empty());
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `ringkernel` | Main facade crate (re-exports everything) |
| `ringkernel-core` | Core traits and types |
| `ringkernel-derive` | Proc macros (`#[ring_kernel]`, `#[derive(RingMessage)]`) |
| `ringkernel-cpu` | CPU backend (testing/fallback) |
| `ringkernel-cuda` | NVIDIA CUDA backend |
| `ringkernel-metal` | Apple Metal backend |
| `ringkernel-wgpu` | WebGPU cross-platform backend |
| `ringkernel-codegen` | GPU kernel code generation |

## Performance Targets

Based on DotCompute benchmarks, targeting:

| Metric | Target | DotCompute |
|--------|--------|------------|
| Serialization | <30ns | 20-50ns |
| Message latency (native) | <500µs | <1ms |
| Startup time | <1ms | ~10ms |
| Binary size | <2MB | 5-10MB |

## Examples

Run the examples:

```bash
# Hello world
cargo run --example hello_kernel

# Vector addition
cargo run --example vector_add

# Ping-pong (K2K messaging)
cargo run --example ping_pong
```

## Running Tests

```bash
# All tests
cargo test --workspace

# With specific backend
cargo test --workspace --features cuda

# Benchmarks
cargo bench --package ringkernel
```

## Documentation

- [Architecture Overview](docs/01-architecture-overview.md)
- [Crate Structure](docs/02-crate-structure.md)
- [Core Abstractions](docs/03-core-abstractions.md)
- [Memory Management](docs/04-memory-management.md)
- [GPU Backends](docs/05-gpu-backends.md)
- [Serialization](docs/06-serialization.md)
- [Proc Macros](docs/07-proc-macros.md)
- [Runtime](docs/08-runtime.md)
- [Testing](docs/09-testing.md)
- [Performance](docs/10-performance.md)
- [Ecosystem Integration](docs/11-ecosystem.md)
- [Migration from DotCompute](docs/12-migration.md)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Acknowledgments

This project is a Rust port of [DotCompute](https://github.com/dotcompute) Ring Kernel system. Thanks to the DotCompute team for the original design and architecture.
