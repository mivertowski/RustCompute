# ringkernel-wgpu-codegen

Rust-to-WGSL transpiler for RingKernel GPU kernels.

## Overview

This crate provides transpilation from a restricted Rust DSL to WGSL (WebGPU Shading Language). It shares the same DSL as `ringkernel-cuda-codegen`, allowing the same kernel code to target both CUDA and WebGPU.

## Installation

```toml
[dependencies]
ringkernel-wgpu-codegen = "0.2"
syn = { version = "2.0", features = ["full"] }
```

## Global Kernels

```rust
use ringkernel_wgpu_codegen::transpile_global_kernel;
use syn::parse_quote;

let func: syn::ItemFn = parse_quote! {
    fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        y[idx as usize] = a * x[idx as usize] + y[idx as usize];
    }
};

let wgsl_code = transpile_global_kernel(&func)?;
```

## Stencil Kernels

```rust
use ringkernel_wgpu_codegen::{transpile_stencil_kernel, StencilConfig};

let func: syn::ItemFn = parse_quote! {
    fn heat(temp: &[f32], temp_new: &mut [f32], alpha: f32, pos: GridPos) {
        let t = temp[pos.idx()];
        let neighbors = pos.north(temp) + pos.south(temp)
                      + pos.east(temp) + pos.west(temp);
        temp_new[pos.idx()] = t + alpha * (neighbors - 4.0 * t);
    }
};

let config = StencilConfig::new("heat")
    .with_tile_size(16, 16)
    .with_halo(1);

let wgsl_code = transpile_stencil_kernel(&func, &config)?;
```

## Ring Kernels

Ring kernels in WGPU are emulated using host-driven dispatch loops:

```rust
use ringkernel_wgpu_codegen::{transpile_ring_kernel, RingKernelConfig};

let handler: syn::ItemFn = parse_quote! {
    fn process(value: f32) -> f32 {
        value * 2.0
    }
};

let config = RingKernelConfig::new("processor")
    .with_workgroup_size(256)
    .with_hlc(true);

let wgsl_code = transpile_ring_kernel(&handler, &config)?;
```

## WGSL Limitations

The transpiler handles these WebGPU limitations:

| CUDA Feature | WGSL Workaround |
|--------------|-----------------|
| 64-bit atomics | Lo/hi u32 pairs with manual carry |
| `f64` | Downcast to `f32` (with warning) |
| Persistent kernels | Host-driven dispatch loop |
| Warp operations | Subgroup operations (where available) |
| K2K messaging | Not supported |

## Type Mapping

| Rust Type | WGSL Type |
|-----------|-----------|
| `f32` | `f32` |
| `i32` | `i32` |
| `u32` | `u32` |
| `bool` | `bool` |
| `&[T]` | `array<T>` (storage, read) |
| `&mut [T]` | `array<T>` (storage, read_write) |

## Testing

```bash
cargo test -p ringkernel-wgpu-codegen
```

The crate includes 50 tests covering types, intrinsics, and transpilation.

## License

Apache-2.0
