---
layout: default
title: WGSL Codegen
nav_order: 15
---

# WGSL Code Generation

The `ringkernel-wgpu-codegen` crate provides a Rust-to-WGSL transpiler that lets you write GPU kernels in a Rust DSL and generate equivalent WGSL (WebGPU Shading Language) code.

## Overview

The transpiler provides **full feature parity with `ringkernel-cuda-codegen`**, supporting the same three kernel types:

1. **Global Kernels** - Standard `@compute` entry points
2. **Stencil Kernels** - Tile-based kernels with `GridPos` abstraction
3. **Ring Kernels** - Persistent actor kernels (host-driven emulation)

The same Rust DSL code works with both CUDA and WGSL backends—just change the import.

## Global Kernels

Generic compute shaders without stencil-specific patterns. Use DSL functions for thread/workgroup indices.

```rust
use ringkernel_wgpu_codegen::transpile_global_kernel;
use syn::parse_quote;

let kernel: syn::ItemFn = parse_quote! {
    fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        y[idx as usize] = a * x[idx as usize] + y[idx as usize];
    }
};

let wgsl_code = transpile_global_kernel(&kernel)?;
```

**Generated WGSL:**
```wgsl
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> a: f32;
@group(0) @binding(3) var<storage, read> n: i32;

@compute @workgroup_size(256, 1, 1)
fn saxpy(
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    var idx: i32 = i32(workgroup_id.x) * i32(256u) + i32(local_invocation_id.x);
    if (idx >= n) { return; }
    y[idx] = a * x[idx] + y[idx];
}
```

## Stencil Kernels

For stencil computations (e.g., FDTD, convolutions), use the `GridPos` abstraction:

```rust
use ringkernel_wgpu_codegen::{transpile_stencil_kernel, StencilConfig};

let kernel: syn::ItemFn = parse_quote! {
    fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
        let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p)
                  - 4.0 * p[pos.idx()];
        p_prev[pos.idx()] = 2.0 * p[pos.idx()] - p_prev[pos.idx()] + c2 * lap;
    }
};

let config = StencilConfig::new("fdtd")
    .with_tile_size(16, 16)
    .with_halo(1);

let wgsl_code = transpile_stencil_kernel(&kernel, &config)?;
```

**Stencil Intrinsics:**
- `pos.idx()` - Current cell linear index
- `pos.north(buf)` - Access cell above (`buf[idx - buffer_width]`)
- `pos.south(buf)` - Access cell below (`buf[idx + buffer_width]`)
- `pos.east(buf)` - Access cell to the right (`buf[idx + 1]`)
- `pos.west(buf)` - Access cell to the left (`buf[idx - 1]`)
- `pos.at(buf, dx, dy)` - Access cell at relative offset

## Ring Kernels

Persistent GPU kernels that process messages in a loop.

**Important:** WebGPU does not support true persistent kernels. Ring kernels are emulated using host-driven dispatch loops—the host re-dispatches the kernel until termination.

```rust
use ringkernel_wgpu_codegen::{transpile_ring_kernel, RingKernelConfig};

let handler: syn::ItemFn = parse_quote! {
    fn process(ctx: &RingContext, msg: &Request) -> Response {
        let tid = ctx.global_thread_id();
        ctx.sync_threads();
        let result = msg.value * 2.0;
        Response { value: result, id: tid as u64 }
    }
};

let config = RingKernelConfig::new("processor")
    .with_workgroup_size(256)
    .with_hlc(true);
    // Note: K2K is NOT supported in WGPU

let wgsl_code = transpile_ring_kernel(&handler, &config)?;
```

### RingKernelConfig Options

| Option | Default | Description |
|--------|---------|-------------|
| `workgroup_size` | 256 | Threads per workgroup |
| `enable_hlc` | false | Enable Hybrid Logical Clocks |
| `enable_k2k` | false | **NOT SUPPORTED** - will return error |
| `max_messages_per_dispatch` | 1024 | Messages processed per dispatch |

### K2K Limitation

Kernel-to-kernel (K2K) messaging is **not supported** in WGPU due to WebGPU's execution model. Attempting to enable K2K will return an error:

```rust
let config = RingKernelConfig::new("processor")
    .with_k2k(true);  // This will error!

let result = transpile_ring_kernel(&handler, &config);
assert!(result.is_err());
// Error: "Kernel-to-kernel (K2K) messaging is not supported in WGPU"
```

Use host-mediated messaging instead for cross-kernel communication.

## DSL Features

### Control Flow

```rust
// If/else
if condition {
    do_something();
} else {
    do_other();
}

// Match (transpiles to switch/case)
match value {
    0 => handle_zero(),
    1 => handle_one(),
    _ => handle_default(),
}

// Early return
if idx >= n { return; }
```

### Loops

```rust
// For loops (range patterns)
for i in 0..n {
    data[i] = 0.0;
}

// Inclusive range
for i in 0..=n {
    process(i);
}

// While loops
while condition {
    iterate();
}

// Infinite loops
loop {
    if done { break; }
    continue;
}
```

**Note:** Loops are forbidden in `Stencil` validation mode but allowed in `Generic` and `RingKernel` modes.

### Shared Memory (Workgroup Variables)

```rust
use ringkernel_wgpu_codegen::{SharedTile, SharedArray};

// 2D tile: var<workgroup> tile: array<array<f32, 16>, 16>;
let tile: SharedTile<f32, 16, 16>;

// 1D array: var<workgroup> cache: array<f32, 256>;
let cache: SharedArray<f32, 256>;
```

### Struct Literals

```rust
// Rust struct literal
let response = Response { value: result, id: tid as u64 };

// Transpiles to WGSL constructor:
// Response(result, u64(tid))
```

### Type Mapping

| Rust Type | WGSL Type | Notes |
|-----------|-----------|-------|
| `f32` | `f32` | Direct mapping |
| `f64` | `f32` | **Warning**: Downcast (WGSL has no f64) |
| `i32` | `i32` | Direct mapping |
| `u32` | `u32` | Direct mapping |
| `i64` | `vec2<i32>` | Emulated as lo/hi pair |
| `u64` | `vec2<u32>` | Emulated as lo/hi pair |
| `bool` | `bool` | Direct mapping |
| `usize` | `u32` | WGSL uses 32-bit addressing |
| `&[T]` | `array<T>` (storage, read) | Storage buffer |
| `&mut [T]` | `array<T>` (storage, read_write) | Storage buffer |

## GPU Intrinsics

### Thread/Workgroup Indices

| Rust DSL | WGSL |
|----------|------|
| `thread_idx_x()` | `local_invocation_id.x` |
| `thread_idx_y()` | `local_invocation_id.y` |
| `thread_idx_z()` | `local_invocation_id.z` |
| `block_idx_x()` | `workgroup_id.x` |
| `block_idx_y()` | `workgroup_id.y` |
| `block_idx_z()` | `workgroup_id.z` |
| `block_dim_x()` | `WORKGROUP_SIZE_X` (constant) |
| `global_thread_id()` | `global_invocation_id.x` |
| `grid_dim_x()` | `num_workgroups.x` |

### Synchronization

| Rust DSL | WGSL |
|----------|------|
| `sync_threads()` | `workgroupBarrier()` |
| `thread_fence()` | `storageBarrier()` |
| `thread_fence_block()` | `workgroupBarrier()` |

### Atomics

| Rust DSL | WGSL |
|----------|------|
| `atomic_add(ptr, val)` | `atomicAdd(ptr, val)` |
| `atomic_sub(ptr, val)` | `atomicSub(ptr, val)` |
| `atomic_min(ptr, val)` | `atomicMin(ptr, val)` |
| `atomic_max(ptr, val)` | `atomicMax(ptr, val)` |
| `atomic_exchange(ptr, val)` | `atomicExchange(ptr, val)` |
| `atomic_cas(ptr, cmp, val)` | `atomicCompareExchangeWeak(ptr, cmp, val)` |
| `atomic_load(ptr)` | `atomicLoad(ptr)` |
| `atomic_store(ptr, val)` | `atomicStore(ptr, val)` |

### Math Functions

| Rust DSL | WGSL |
|----------|------|
| `sqrt(x)` | `sqrt(x)` |
| `rsqrt(x)` | `inverseSqrt(x)` |
| `abs(x)` | `abs(x)` |
| `floor(x)` | `floor(x)` |
| `ceil(x)` | `ceil(x)` |
| `round(x)` | `round(x)` |
| `sin(x)` | `sin(x)` |
| `cos(x)` | `cos(x)` |
| `tan(x)` | `tan(x)` |
| `exp(x)` | `exp(x)` |
| `log(x)` | `log(x)` |
| `powf(x, y)` | `pow(x, y)` |
| `min(a, b)` | `min(a, b)` |
| `max(a, b)` | `max(a, b)` |
| `clamp(x, lo, hi)` | `clamp(x, lo, hi)` |
| `fma(a, b, c)` | `fma(a, b, c)` |
| `mix(a, b, t)` | `mix(a, b, t)` |

### Subgroup Operations (Warp Equivalents)

**Note:** These require the `chromium_experimental_subgroups` extension, which may not be available on all platforms.

| Rust DSL | WGSL |
|----------|------|
| `warp_shuffle(val, lane)` | `subgroupShuffle(val, lane)` |
| `warp_shuffle_up(val, delta)` | `subgroupShuffleUp(val, delta)` |
| `warp_shuffle_down(val, delta)` | `subgroupShuffleDown(val, delta)` |
| `warp_shuffle_xor(val, mask)` | `subgroupShuffleXor(val, mask)` |
| `warp_ballot(pred)` | `subgroupBallot(pred)` |
| `warp_all(pred)` | `subgroupAll(pred)` |
| `warp_any(pred)` | `subgroupAny(pred)` |
| `lane_id()` | `subgroup_invocation_id` |
| `warp_size()` | `subgroup_size` |

## 64-bit Integer Emulation

WGSL 1.0 does not support 64-bit integers or atomics. The transpiler emulates them using lo/hi `u32` pairs:

### Type Representation

```wgsl
// u64 becomes vec2<u32> where:
// - x (lo) = lower 32 bits
// - y (hi) = upper 32 bits
```

### Generated Helper Functions

The transpiler automatically includes these helpers for ring kernels:

```wgsl
// Read 64-bit value from atomic pair
fn read_u64(lo: ptr<storage, atomic<u32>, read_write>,
            hi: ptr<storage, atomic<u32>, read_write>) -> vec2<u32> {
    return vec2<u32>(atomicLoad(lo), atomicLoad(hi));
}

// Atomically increment 64-bit value
fn atomic_inc_u64(lo: ptr<storage, atomic<u32>, read_write>,
                  hi: ptr<storage, atomic<u32>, read_write>) {
    let old_lo = atomicAdd(lo, 1u);
    if (old_lo == 0xFFFFFFFFu) {
        atomicAdd(hi, 1u);
    }
}

// Compare two 64-bit values: returns -1, 0, or 1
fn compare_u64(a: vec2<u32>, b: vec2<u32>) -> i32 {
    if (a.y > b.y) { return 1; }
    if (a.y < b.y) { return -1; }
    if (a.x > b.x) { return 1; }
    if (a.x < b.x) { return -1; }
    return 0;
}
```

## Validation Modes

The transpiler uses validation modes to control what constructs are allowed:

| Mode | Loops | Description |
|------|-------|-------------|
| `Stencil` | Forbidden | Classic stencil kernels (default for `transpile_stencil_kernel`) |
| `Generic` | Allowed | General-purpose kernels (default for `transpile_global_kernel`) |
| `RingKernel` | Required | Persistent actor kernels |

## WGSL vs CUDA Differences

| Feature | CUDA | WGSL |
|---------|------|------|
| Thread index | `threadIdx.x` | `local_invocation_id.x` |
| Block/workgroup | `blockIdx.x` | `workgroup_id.x` |
| Sync | `__syncthreads()` | `workgroupBarrier()` |
| Shared memory | `__shared__` | `var<workgroup>` |
| 64-bit integers | Native | Emulated (vec2) |
| f64 | Native | Not supported |
| Persistent kernels | GPU loop | Host-driven dispatch |
| K2K messaging | Supported | Not supported |

## Testing

The crate includes 50 tests covering all features:

```bash
cargo test -p ringkernel-wgpu-codegen
```

Test categories:
- Type mapping (primitives, slices, 64-bit emulation)
- Expression transpilation (binary ops, method calls, indexing)
- Intrinsics (thread indices, barriers, math)
- Loops (for/while/loop patterns)
- Validation (mode-specific constraints)

## Next: [Migration Guide](./12-migration.md)
