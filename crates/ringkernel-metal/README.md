# ringkernel-metal

Apple Metal backend for RingKernel.

## Status

This crate is **scaffolded** - the API is defined but implementation is pending.

## Overview

This crate will provide GPU compute support for RingKernel on Apple platforms using the Metal framework. It targets macOS, iOS, and Apple Silicon devices.

## Requirements (Planned)

- macOS 10.15+ or iOS 13+
- Apple Silicon (M1/M2/M3) or compatible AMD GPU
- Metal feature enabled

## Features (Planned)

- Event-driven kernel execution via Metal compute shaders
- MSL (Metal Shading Language) shader support
- Apple Silicon unified memory architecture optimization
- iOS deployment support

## Limitations

Like WebGPU, Metal does not support CUDA-style cooperative groups, so persistent kernels will use event-driven execution with host-side dispatch loops.

## Usage (API Preview)

```rust
use ringkernel_metal::MetalRuntime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !ringkernel_metal::is_metal_available() {
        eprintln!("Metal not available");
        return Ok(());
    }

    let runtime = MetalRuntime::new().await?;
    let kernel = runtime.launch("compute", Default::default()).await?;

    // Process messages...

    kernel.terminate().await?;
    runtime.shutdown().await?;
    Ok(())
}
```

## Current State

The crate currently provides:
- Stub `MetalRuntime` that returns `BackendUnavailable` errors
- MSL kernel template (`RING_KERNEL_MSL_TEMPLATE`)
- `is_metal_available()` runtime check

## Contributing

Contributions to implement the Metal backend are welcome. Key work items:
1. Device enumeration via `metal::Device`
2. Compute pipeline creation
3. Buffer management with unified memory
4. Kernel dispatch and synchronization

## License

Apache-2.0
