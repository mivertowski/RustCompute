# ringkernel-wgpu

WebGPU backend for RingKernel.

## Overview

This crate provides cross-platform GPU compute support for RingKernel via WebGPU (wgpu). It works across Vulkan, Metal, DX12, and browser environments.

## Requirements

- A GPU supporting Vulkan 1.2, Metal, or DirectX 12
- For browsers: WebGPU-enabled browser (Chrome 113+, Firefox 118+)

## Features

- Cross-platform GPU access (Windows, macOS, Linux, Web)
- WGSL (WebGPU Shading Language) compute shaders
- Event-driven execution model
- Automatic backend selection (Vulkan → Metal → DX12)

## Limitations

Compared to the CUDA backend:

- **No persistent kernels**: WebGPU doesn't support cooperative groups; kernels are dispatched per batch
- **No 64-bit atomics**: Counters use lo/hi u32 pair emulation
- **No kernel-to-kernel messaging**: K2K is not supported in WebGPU's execution model
- **Event-driven only**: Must re-dispatch for continuous processing

## Usage

```rust
use ringkernel_wgpu::WgpuRuntime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !ringkernel_wgpu::is_wgpu_available() {
        eprintln!("No WebGPU adapter found");
        return Ok(());
    }

    let runtime = WgpuRuntime::new().await?;
    let kernel = runtime.launch("compute", Default::default()).await?;

    // Process messages...

    kernel.terminate().await?;
    runtime.shutdown().await?;
    Ok(())
}
```

## Exports

| Type | Description |
|------|-------------|
| `WgpuRuntime` | Main runtime implementing `RingKernelRuntime` |
| `WgpuAdapter` | GPU adapter handle |
| `WgpuKernel` | Compiled compute pipeline |
| `WgpuBuffer` | GPU buffer wrapper |

## Testing

```bash
# Requires WebGPU-capable GPU
cargo test -p ringkernel-wgpu --features wgpu-tests -- --ignored
```

## Browser Support

When targeting WebAssembly, the runtime automatically detects the browser's WebGPU support and configures accordingly.

## License

Apache-2.0
