# ringkernel-cuda

NVIDIA CUDA backend for RingKernel.

## Overview

This crate provides NVIDIA GPU support for RingKernel via the `cudarc` library
(v0.19.3). It implements the `RingKernelRuntime` trait for launching and
managing persistent GPU kernels.

## Requirements

- NVIDIA GPU with Compute Capability 7.0 or higher (Volta, Turing, Ampere, Ada,
  Hopper, Blackwell)
- CUDA Toolkit 12.x or later
- cudarc 0.19.3 (pinned via workspace)
- Linux (native) or Windows (WSL2, with cooperative-group limitations)

## Features

- Persistent kernel execution using cooperative groups
- Lock-free message queues in GPU global memory
- PTX compilation at runtime via NVRTC, with an optional on-disk PTX cache
- NVTX profiling hooks and optional Chrome-trace export
- Multi-GPU runtime: device enumeration, P2P setup, NVLink topology detection
  (via `nvml-wrapper`), and cross-device actor migration
- Stencil kernel loading and execution

## Usage

```rust
use ringkernel_cuda::CudaRuntime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check availability first
    if !ringkernel_cuda::is_cuda_available() {
        eprintln!("No CUDA device found");
        return Ok(());
    }

    let runtime = CudaRuntime::new().await?;
    let kernel = runtime.launch("processor", Default::default()).await?;

    // Process messages...

    kernel.terminate().await?;
    runtime.shutdown().await?;
    Ok(())
}
```

## Stencil Kernel Loading

For pre-transpiled CUDA kernels:

```rust
use ringkernel_cuda::{StencilKernelLoader, LaunchConfig};

let loader = StencilKernelLoader::new(&cuda_device);
let kernel = loader.load_from_source(cuda_source)?;

let config = LaunchConfig {
    grid: (grid_x, grid_y, 1),
    block: (16, 16, 1),
    shared_mem: 0,
};

kernel.launch(&config, &[&input_buf, &output_buf])?;
```

## cudarc 0.19.3 API

This crate uses cudarc 0.19.3 with the builder pattern for kernel launches:

```rust
use cudarc::driver::{CudaModule, CudaFunction, PushKernelArg};

// Load module and function
let module = device.inner().load_module(ptx)?;
let func = module.load_function("kernel_name")?;

// Launch with builder pattern
unsafe {
    stream
        .launch_builder(&func)
        .arg(&input_ptr)
        .arg(&output_ptr)
        .launch(cfg)?;
}
```

For cooperative kernel launches (grid-wide synchronization):

```rust
use cudarc::driver::result as cuda_result;

unsafe {
    cuda_result::launch_cooperative_kernel(
        func, grid_dim, block_dim, shared_mem_bytes, stream, kernel_params
    )?;
}
```

## Exports

| Type | Description |
|------|-------------|
| `CudaRuntime` | Main runtime implementing `RingKernelRuntime` |
| `CudaDevice` | GPU device handle |
| `CudaKernel` | Compiled kernel handle |
| `CudaBuffer` | GPU memory buffer |
| `CudaControlBlock` | GPU-resident kernel state |
| `CudaMessageQueue` | Lock-free queue in GPU memory |
| `StencilKernelLoader` | Loads CUDA stencil kernels |

## Feature flags

| Flag | Description |
|------|-------------|
| `cuda` | Enable the CUDA backend (pulls in `cudarc`). |
| `cooperative` | Grid-wide synchronization via cooperative kernel launch. Requires `nvcc` at build time. |
| `profiling` | NVTX ranges, CUDA events, and Chrome-trace export (adds `serde`/`serde_json`). |
| `ptx-cache` | On-disk PTX cache to eliminate first-tick compilation overhead. |
| `multi-gpu` | NVLink topology probing via `nvml-wrapper` (falls back to `cudarc` P2P otherwise). |
| `nvshmem` | Opt-in NVSHMEM symmetric-heap bindings (v1.2 groundwork). Requires a working NVSHMEM install; set `NVSHMEM_LIB_DIR` if it lives outside `/usr`. |

## v1.1 multi-GPU runtime

Version 1.1 ships a multi-GPU runtime built on this backend:

- Device enumeration and per-device stream management
- P2P enablement between peers that support it
- NVLink topology detection (when the `multi-gpu` feature is enabled)
- Cross-device actor migration through the core runtime's migration primitives

See `ringkernel-cuda`'s `multi_gpu` module for the concrete APIs.

## v1.2 groundwork on `main`

The `main` branch also contains groundwork for v1.2 that is opt-in and not yet
stabilised:

- NVSHMEM bindings behind the `nvshmem` feature for symmetric-heap allocations
- Hierarchical work stealing for persistent-actor scheduling
- Blackwell codegen stubs

These paths compile but are exercised by a limited test set. Treat them as
preview quality.

## Platform notes

**Native Linux**: Full support for persistent kernels using CUDA cooperative
groups.

**WSL2**: Persistent kernels may not work due to cooperative-group
limitations. Falls back to event-driven execution.

**Windows Native**: Not currently supported. Use WSL2.

## Testing

```bash
# Requires NVIDIA GPU
cargo test -p ringkernel-cuda --features cuda
```

## License

Apache-2.0
