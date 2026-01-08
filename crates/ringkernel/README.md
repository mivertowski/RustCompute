# ringkernel

Main facade crate for the RingKernel GPU-native persistent actor model framework.

## Overview

This crate re-exports the entire RingKernel API, providing a single entry point for users. It combines core abstractions, backend implementations, and derive macros into a unified interface.

## Installation

```toml
[dependencies]
ringkernel = "0.2"
tokio = { version = "1.48", features = ["full"] }
```

For GPU backends:

```toml
# NVIDIA CUDA
ringkernel = { version = "0.2", features = ["cuda"] }

# WebGPU (cross-platform)
ringkernel = { version = "0.2", features = ["wgpu"] }

# All backends
ringkernel = { version = "0.2", features = ["all-backends"] }
```

## Quick Start

```rust
use ringkernel::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create runtime with CPU backend
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await?;

    // Launch a kernel
    let kernel = runtime.launch("processor", LaunchOptions::default()).await?;

    // Kernel lifecycle
    kernel.deactivate().await?;
    kernel.activate().await?;
    kernel.terminate().await?;

    runtime.shutdown().await?;
    Ok(())
}
```

## Features

| Feature | Description |
|---------|-------------|
| `cpu` | CPU backend (default, always available) |
| `cuda` | NVIDIA CUDA backend |
| `wgpu` | WebGPU cross-platform backend |
| `metal` | Apple Metal backend (scaffolded) |
| `all-backends` | Enable all GPU backends |

## Re-exported Crates

- `ringkernel-core` - Core traits and types
- `ringkernel-cpu` - CPU backend
- `ringkernel-cuda` - CUDA backend (optional)
- `ringkernel-wgpu` - WebGPU backend (optional)
- `ringkernel-derive` - Proc macros

## Examples

The crate includes 20+ examples:

```bash
# Basic usage
cargo run -p ringkernel --example basic_hello_kernel

# Messaging patterns
cargo run -p ringkernel --example kernel_to_kernel
cargo run -p ringkernel --example pub_sub

# CUDA code generation
cargo run -p ringkernel --example global_kernel
cargo run -p ringkernel --example stencil_kernel

# WebGPU
cargo run -p ringkernel --example wgpu_hello --features wgpu
```

## License

Apache-2.0
