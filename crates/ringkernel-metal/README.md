# ringkernel-metal

Apple Metal backend for RingKernel.

## Status

**Implemented** - Full Metal backend with event-driven execution model.

## Overview

This crate provides GPU compute support for RingKernel on Apple platforms using the Metal framework. It targets macOS, iOS, and Apple Silicon devices with unified memory architecture.

## Requirements

- macOS 10.15+ or iOS 13+
- Apple Silicon (M1/M2/M3/M4) or compatible AMD GPU
- Rust with `metal` feature enabled

## Features

- **Event-driven kernel execution** via Metal compute shaders
- **MSL (Metal Shading Language)** runtime compilation
- **Unified memory architecture** optimization for Apple Silicon
- **K2K (Kernel-to-Kernel) messaging** via inbox/routing system
- **Halo exchange** for stencil computations (2D/3D grids)
- **HLC (Hybrid Logical Clock)** support for causal ordering
- **Full `RingKernelRuntime` trait** implementation

## Limitations

Metal does not support CUDA-style cooperative groups, so persistent kernels use event-driven execution with host-side dispatch loops. This provides equivalent functionality with different performance characteristics:

| Aspect | CUDA Persistent | Metal Event-Driven |
|--------|-----------------|-------------------|
| Kernel lifetime | Long-running | Per-message dispatch |
| Grid synchronization | `grid.sync()` | Host-driven barriers |
| Command latency | ~0.03µs (mapped memory) | ~10-50µs (dispatch) |
| Best for | Interactive commands | Batch compute |

## Usage

```rust
use ringkernel_metal::{MetalRuntime, is_metal_available};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !is_metal_available() {
        eprintln!("Metal not available");
        return Ok(());
    }

    // Create runtime
    let runtime = MetalRuntime::new().await?;

    // Launch kernel with options
    let kernel = runtime.launch("compute", Default::default()).await?;
    kernel.activate().await?;

    // Send messages (triggers compute dispatch)
    kernel.send_envelope(envelope).await?;

    // Receive results
    let response = kernel.receive_timeout(Duration::from_secs(1)).await?;

    // Clean shutdown
    kernel.terminate().await?;
    runtime.shutdown().await?;
    Ok(())
}
```

## Architecture

### Core Components

- **`MetalDevice`** - Wrapper around `metal::Device` with capability queries
- **`MetalBuffer`** - GPU buffer with `StorageModeShared` for unified memory
- **`MetalKernel`** - Full `KernelHandleInner` implementation with:
  - MSL shader compilation via `new_library_with_source`
  - Compute pipeline creation
  - Event-driven dispatch on message send
  - Control block for lifecycle management
- **`MetalRuntime`** - `RingKernelRuntime` implementation with K2K broker

### K2K Messaging

Kernel-to-kernel messaging uses a routing table and inbox system:

```rust
// K2K structures
MetalK2KInboxHeader  // 64 bytes - inbox state with lock/sequence
MetalK2KRouteEntry   // 32 bytes - route to neighbor
MetalK2KRoutingTable // Routing with 2D (4-neighbor) or 3D (6-neighbor) support
```

### Halo Exchange

For stencil computations, the `MetalHaloExchange` manager handles:

```rust
use ringkernel_metal::{MetalHaloExchange, HaloExchangeConfig};

// Create 2D grid with halo exchange
let config = HaloExchangeConfig::new_2d(
    8, 8,   // grid: 8x8 tiles
    64, 64, // tile: 64x64 cells
    1,      // halo: 1 cell
);
let mut exchange = MetalHaloExchange::new(config);

// Initialize on device
exchange.initialize(&device)?;

// Perform exchange cycle
exchange.exchange(&tile_buffers)?;
```

## MSL Templates

The crate provides MSL templates for common patterns:

- **`RING_KERNEL_MSL_TEMPLATE`** - Base ring kernel with control block, message queues, and HLC
- **`K2K_HALO_EXCHANGE_MSL_TEMPLATE`** - K2K communication for stencil computations

## Platform Support

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon) | ✅ Full support |
| macOS (Intel + AMD) | ✅ Full support |
| iOS | ✅ Supported (untested) |
| Linux/Windows | ❌ Stub only |

On non-macOS platforms, `MetalRuntime::new()` returns `BackendUnavailable`.

## License

Apache-2.0
