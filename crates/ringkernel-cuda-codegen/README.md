# ringkernel-cuda-codegen

Rust-to-CUDA transpiler for RingKernel GPU kernels.

## Overview

This crate enables writing GPU kernels in a restricted Rust DSL and transpiling them to CUDA C code. It supports three kernel types:

1. **Global Kernels** - Standard CUDA `__global__` functions
2. **Stencil Kernels** - Tile-based kernels with `GridPos` abstraction
3. **Ring Kernels** - Persistent actor kernels with message loops

## Installation

```toml
[dependencies]
ringkernel-cuda-codegen = "0.1"
syn = { version = "2.0", features = ["full"] }
```

## Global Kernels

For general-purpose CUDA kernels:

```rust
use ringkernel_cuda_codegen::transpile_global_kernel;
use syn::parse_quote;

let func: syn::ItemFn = parse_quote! {
    fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        y[idx as usize] = a * x[idx as usize] + y[idx as usize];
    }
};

let cuda_code = transpile_global_kernel(&func)?;
```

## Stencil Kernels

For grid-based computations with neighbor access:

```rust
use ringkernel_cuda_codegen::{transpile_stencil_kernel, StencilConfig};

let func: syn::ItemFn = parse_quote! {
    fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
        let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p)
                  - 4.0 * p[pos.idx()];
        p_prev[pos.idx()] = 2.0 * p[pos.idx()] - p_prev[pos.idx()] + c2 * lap;
    }
};

let config = StencilConfig::new("fdtd")
    .with_tile_size(16, 16)
    .with_halo(1);

let cuda_code = transpile_stencil_kernel(&func, &config)?;
```

## Ring Kernels

For persistent actor-model kernels:

```rust
use ringkernel_cuda_codegen::{transpile_ring_kernel, RingKernelConfig};

let handler: syn::ItemFn = parse_quote! {
    fn process(ctx: &RingContext, msg: &Request) -> Response {
        let tid = ctx.global_thread_id();
        ctx.sync_threads();
        Response { value: msg.value * 2.0, id: tid as u64 }
    }
};

let config = RingKernelConfig::new("processor")
    .with_block_size(128)
    .with_queue_capacity(1024)
    .with_hlc(true)   // Hybrid Logical Clocks
    .with_k2k(true);  // Kernel-to-kernel messaging

let cuda_code = transpile_ring_kernel(&handler, &config)?;
```

## DSL Reference

### Thread/Block Indices
- `thread_idx_x()`, `thread_idx_y()`, `thread_idx_z()`
- `block_idx_x()`, `block_idx_y()`, `block_idx_z()`
- `block_dim_x()`, `block_dim_y()`, `block_dim_z()`
- `grid_dim_x()`, `grid_dim_y()`, `grid_dim_z()`

### Stencil Intrinsics
- `pos.idx()` - Linear index
- `pos.north(buf)`, `pos.south(buf)`, `pos.east(buf)`, `pos.west(buf)`
- `pos.at(buf, dx, dy)` - Relative offset access

### Synchronization
- `sync_threads()` - Block-level barrier
- `thread_fence()` - Device memory fence
- `thread_fence_block()` - Block memory fence

### Atomics
- `atomic_add(ptr, val)`, `atomic_sub(ptr, val)`
- `atomic_min(ptr, val)`, `atomic_max(ptr, val)`
- `atomic_exchange(ptr, val)`, `atomic_cas(ptr, compare, val)`

### Math Functions
- `sqrt()`, `abs()`, `floor()`, `ceil()`, `round()`
- `sin()`, `cos()`, `tan()`, `exp()`, `log()`
- `powf()`, `min()`, `max()`, `mul_add()`

### Warp Operations
- `warp_shuffle(val, lane)`, `warp_shuffle_up(val, delta)`
- `warp_shuffle_down(val, delta)`, `warp_shuffle_xor(val, mask)`
- `warp_ballot(pred)`, `warp_all(pred)`, `warp_any(pred)`

## Type Mapping

| Rust Type | CUDA Type |
|-----------|-----------|
| `f32` | `float` |
| `f64` | `double` |
| `i32` | `int` |
| `u32` | `unsigned int` |
| `i64` | `long long` |
| `u64` | `unsigned long long` |
| `bool` | `int` |
| `&[T]` | `const T* __restrict__` |
| `&mut [T]` | `T* __restrict__` |

## Testing

```bash
cargo test -p ringkernel-cuda-codegen
```

The crate includes 143 tests covering all kernel types and language features.

## License

Apache-2.0
