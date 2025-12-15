---
layout: default
title: CUDA Codegen
nav_order: 14
---

# CUDA Code Generation

The `ringkernel-cuda-codegen` crate provides a Rust-to-CUDA transpiler that lets you write GPU kernels in a Rust DSL and generate equivalent CUDA C code.

> **Cross-platform alternative:** For WebGPU/WGSL, see [WGSL Code Generation](./14-wgpu-codegen.md). Both transpilers share the same Rust DSL—just change the import.

## Overview

The transpiler supports three types of kernels:

1. **Global Kernels** - Standard CUDA `__global__` functions
2. **Stencil Kernels** - Tile-based kernels with `GridPos` abstraction
3. **Ring Kernels** - Persistent actor kernels with message loops

## Global Kernels

Generic CUDA kernels without stencil-specific patterns. Use DSL functions for thread/block indices.

```rust
use ringkernel_cuda_codegen::transpile_global_kernel;
use syn::parse_quote;

let kernel: syn::ItemFn = parse_quote! {
    fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        y[idx as usize] = a * x[idx as usize] + y[idx as usize];
    }
};

let cuda_code = transpile_global_kernel(&kernel)?;
```

**Generated CUDA:**
```cuda
extern "C" __global__ void saxpy(
    const float* __restrict__ x,
    float* __restrict__ y,
    float a,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) { return; }
    y[idx] = a * x[idx] + y[idx];
}
```

## Stencil Kernels

For stencil computations (e.g., FDTD, convolutions), use the `GridPos` abstraction:

```rust
use ringkernel_cuda_codegen::{transpile_stencil_kernel, StencilConfig, Grid};

// 2D stencil kernel
let kernel: syn::ItemFn = parse_quote! {
    fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
        let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p)
                  - 4.0 * p[pos.idx()];
        p_prev[pos.idx()] = 2.0 * p[pos.idx()] - p_prev[pos.idx()] + c2 * lap;
    }
};

let config = StencilConfig::new("fdtd")
    .with_grid(Grid::Grid2D)
    .with_tile_size(16, 16)
    .with_halo(1);

let cuda_code = transpile_stencil_kernel(&kernel, &config)?;

// 3D stencil kernel (for volumetric simulations)
let kernel_3d: syn::ItemFn = parse_quote! {
    fn laplacian_3d(p: &[f32], out: &mut [f32], pos: GridPos) {
        let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p)
                  + pos.up(p) + pos.down(p) - 6.0 * p[pos.idx()];
        out[pos.idx()] = lap;
    }
};

let config_3d = StencilConfig::new("laplacian")
    .with_grid(Grid::Grid3D)
    .with_tile_size(8, 8)
    .with_halo(1);
```

**Stencil Intrinsics (2D):**
- `pos.idx()` - Current cell linear index
- `pos.north(buf)` - Access cell above (Y+)
- `pos.south(buf)` - Access cell below (Y-)
- `pos.east(buf)` - Access cell to the right (X+)
- `pos.west(buf)` - Access cell to the left (X-)
- `pos.at(buf, dx, dy)` - Access cell at relative offset

**Stencil Intrinsics (3D):**
- `pos.up(buf)` - Access cell above (Z+)
- `pos.down(buf)` - Access cell below (Z-)
- `pos.at(buf, dx, dy, dz)` - 3D relative offset access

## Ring Kernels

Persistent GPU kernels that process messages in a loop. Ideal for the actor model pattern.

```rust
use ringkernel_cuda_codegen::{transpile_ring_kernel, RingKernelConfig};

let handler: syn::ItemFn = parse_quote! {
    fn process(ctx: &RingContext, msg: &Request) -> Response {
        let tid = ctx.global_thread_id();
        ctx.sync_threads();
        let result = msg.value * 2.0;
        Response { value: result, id: tid as u64 }
    }
};

let config = RingKernelConfig::new("processor")
    .with_block_size(128)
    .with_queue_capacity(1024)
    .with_envelope_format(true)  // 256-byte MessageHeader + payload
    .with_hlc(true)              // Enable Hybrid Logical Clocks
    .with_k2k(true)              // Enable K2K messaging
    .with_kernel_id(1000)        // Kernel identity for routing
    .with_hlc_node_id(42);       // HLC node ID

let cuda_code = transpile_ring_kernel(&handler, &config)?;
```

### RingKernelConfig Options

| Option | Default | Description |
|--------|---------|-------------|
| `block_size` | 128 | Threads per block |
| `queue_capacity` | 1024 | Input/output queue size (must be power of 2) |
| `enable_hlc` | true | Enable Hybrid Logical Clocks |
| `enable_k2k` | false | Enable kernel-to-kernel messaging |
| `use_envelope_format` | true | Use MessageEnvelope format (256-byte header + payload) |
| `kernel_id` | 0 | Kernel identity for K2K routing |
| `hlc_node_id` | 0 | HLC node ID for timestamp generation |
| `message_size` | 64 | Message payload size (not including header) |
| `response_size` | 64 | Response payload size (not including header) |
| `idle_sleep_ns` | 1000 | Nanosleep duration when idle |

### Envelope Format

When `use_envelope_format(true)` is set, the generated kernel uses the `MessageEnvelope` format compatible with the RingKernel runtime:

- **MessageHeader** (256 bytes, 64-byte aligned):
  - Magic number validation (`0x52494E474B45524E` = "RINGKERN")
  - Message ID and correlation ID for request/response tracking
  - Source/destination kernel IDs for routing
  - HLC timestamps for causal ordering
  - Payload size and checksum

- **Generated helper functions**:
  - `message_get_header(envelope_ptr)` - Extract header from envelope
  - `message_get_payload(envelope_ptr)` - Get payload pointer
  - `message_header_validate(header)` - Validate magic number
  - `message_create_response_header(...)` - Create response with correlation

### RingContext Methods

The `RingContext` parameter provides GPU intrinsics that are inlined during transpilation:

| Method | CUDA Equivalent |
|--------|-----------------|
| `ctx.thread_id()` | `threadIdx.x` |
| `ctx.block_id()` | `blockIdx.x` |
| `ctx.global_thread_id()` | `blockIdx.x * blockDim.x + threadIdx.x` |
| `ctx.sync_threads()` | `__syncthreads()` |
| `ctx.lane_id()` | `threadIdx.x % 32` |
| `ctx.warp_id()` | `threadIdx.x / 32` |

### Generated Kernel Structure

Ring kernels generate a complete persistent kernel with:

1. **ControlBlock** - 128-byte aligned struct for lifecycle management
2. **Preamble** - Thread ID calculation, queue constants
3. **Persistent Loop** - Message dequeue, handler execution, response enqueue
4. **Epilogue** - Termination marking, HLC state persistence

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

### Shared Memory

```rust
use ringkernel_cuda_codegen::SharedMemoryConfig;

let mut shared = SharedMemoryConfig::new();
shared.add_array("cache", "float", 256);     // __shared__ float cache[256];
shared.add_tile("tile", "float", 16, 16);    // __shared__ float tile[16][16];
```

### Struct Literals

```rust
// Rust struct literal
let response = Response { value: result, id: tid as u64 };

// Transpiles to C compound literal:
// (Response){ .value = result, .id = (unsigned long long)(tid) }
```

### Reference Expressions

The transpiler handles Rust reference expressions and automatically tracks pointer variables for correct accessor generation:

```rust
// Rust DSL with reference expressions
fn process_batch(transactions: &[GpuTransaction], alerts: &mut [GpuAlert], n: i32) {
    let idx = block_idx_x() * block_dim_x() + thread_idx_x();
    if idx >= n { return; }

    let tx = &transactions[idx as usize];    // Creates pointer
    let alert = &mut alerts[idx as usize];   // Creates mutable pointer

    alert.transaction_id = tx.transaction_id;  // Uses -> automatically
}

// Transpiles to CUDA:
// GpuTransaction* tx = &transactions[(unsigned long long)(idx)];
// GpuAlert* alert = &alerts[(unsigned long long)(idx)];
// alert->transaction_id = tx->transaction_id;
```

**Type Inference for References:**
- `&arr[idx]` where `arr` contains "transaction" → `GpuTransaction*`
- `&arr[idx]` where `arr` contains "profile" → `GpuCustomerProfile*`
- `&arr[idx]` where `arr` contains "alert" → `GpuAlert*`
- Other reference expressions → appropriate pointer type

**Automatic Accessor Selection:**
- Pointer variables use `->` for field access
- Value variables use `.` for field access
- Tracked via `pointer_vars` HashSet during transpilation

### Type Mapping

| Rust Type | CUDA Type |
|-----------|-----------|
| `f32` | `float` |
| `f64` | `double` |
| `i32` | `int` |
| `u32` | `unsigned int` |
| `i64` | `long long` |
| `u64` | `unsigned long long` |
| `bool` | `int` |
| `usize` | `unsigned long long` |
| `&[T]` | `const T* __restrict__` |
| `&mut [T]` | `T* __restrict__` |

## GPU Intrinsics

The transpiler supports **120+ GPU intrinsics** across 13 categories.

### Thread/Block Indices (13 ops)

- `thread_idx_x()`, `thread_idx_y()`, `thread_idx_z()` → `threadIdx.x/y/z`
- `block_idx_x()`, `block_idx_y()`, `block_idx_z()` → `blockIdx.x/y/z`
- `block_dim_x()`, `block_dim_y()`, `block_dim_z()` → `blockDim.x/y/z`
- `grid_dim_x()`, `grid_dim_y()`, `grid_dim_z()` → `gridDim.x/y/z`
- `warp_size()` → `warpSize`

### Synchronization (7 ops)

- `sync_threads()` → `__syncthreads()`
- `sync_threads_count(pred)` → `__syncthreads_count()`
- `sync_threads_and(pred)` → `__syncthreads_and()`
- `sync_threads_or(pred)` → `__syncthreads_or()`
- `thread_fence()` → `__threadfence()`
- `thread_fence_block()` → `__threadfence_block()`
- `thread_fence_system()` → `__threadfence_system()`

### Atomic Operations (11 ops)

- `atomic_add(ptr, val)` → `atomicAdd`
- `atomic_sub(ptr, val)` → `atomicSub`
- `atomic_min(ptr, val)` → `atomicMin`
- `atomic_max(ptr, val)` → `atomicMax`
- `atomic_exchange(ptr, val)` → `atomicExch`
- `atomic_cas(ptr, compare, val)` → `atomicCAS`
- `atomic_and(ptr, val)` → `atomicAnd`
- `atomic_or(ptr, val)` → `atomicOr`
- `atomic_xor(ptr, val)` → `atomicXor`
- `atomic_inc(ptr, val)` → `atomicInc` (increment with wrap)
- `atomic_dec(ptr, val)` → `atomicDec` (decrement with wrap)

### Math Functions (16 ops)

- `sqrt()`, `rsqrt()` - Square root, reciprocal sqrt
- `abs()`, `fabs()` - Absolute value
- `floor()`, `ceil()`, `round()`, `trunc()` - Rounding
- `fma()`, `mul_add()` - Fused multiply-add
- `fmin()`, `fmax()` - Minimum, maximum
- `fmod()`, `remainder()` - Modulo operations
- `copysign()`, `cbrt()`, `hypot()`

### Trigonometric Functions (11 ops)

- `sin()`, `cos()`, `tan()` - Basic trig
- `asin()`, `acos()`, `atan()`, `atan2()` - Inverse trig
- `sincos()` - Combined sine and cosine
- `sinpi()`, `cospi()` - Sin/cos of π*x

### Hyperbolic Functions (6 ops)

- `sinh()`, `cosh()`, `tanh()` - Hyperbolic
- `asinh()`, `acosh()`, `atanh()` - Inverse hyperbolic

### Exponential and Logarithmic (18 ops)

- `exp()`, `exp2()`, `exp10()`, `expm1()` - Exponentials
- `log()`, `ln()`, `log2()`, `log10()`, `log1p()` - Logarithms
- `pow()`, `powf()`, `powi()` - Power
- `ldexp()`, `scalbn()`, `ilogb()` - Exponent manipulation
- `erf()`, `erfc()`, `erfinv()`, `erfcinv()` - Error functions
- `lgamma()`, `tgamma()` - Gamma functions

### Classification Functions (8 ops)

- `is_nan()`, `isnan()` → `isnan`
- `is_infinite()`, `isinf()` → `isinf`
- `is_finite()`, `isfinite()` → `isfinite`
- `is_normal()`, `isnormal()` → `isnormal`
- `signbit()`, `nextafter()`, `fdim()`

### Warp Operations (16 ops)

- `warp_active_mask()` → `__activemask()`
- `warp_shfl(mask, val, lane)` → `__shfl_sync`
- `warp_shfl_up(mask, val, delta)` → `__shfl_up_sync`
- `warp_shfl_down(mask, val, delta)` → `__shfl_down_sync`
- `warp_shfl_xor(mask, val, lane_mask)` → `__shfl_xor_sync`
- `warp_ballot(mask, pred)` → `__ballot_sync`
- `warp_all(mask, pred)` → `__all_sync`
- `warp_any(mask, pred)` → `__any_sync`
- `warp_match_any(mask, val)` → `__match_any_sync` (Volta+)
- `warp_match_all(mask, val)` → `__match_all_sync` (Volta+)
- `warp_reduce_add/min/max/and/or/xor(mask, val)` → `__reduce_*_sync` (SM 8.0+)

### Bit Manipulation (8 ops)

- `popc()`, `popcount()`, `count_ones()` → `__popc`
- `clz()`, `leading_zeros()` → `__clz`
- `ctz()`, `trailing_zeros()` → `__ffs - 1`
- `ffs()` → `__ffs`
- `brev()`, `reverse_bits()` → `__brev`
- `byte_perm()` → `__byte_perm`
- `funnel_shift_left()`, `funnel_shift_right()` → `__funnelshift_l/r`

### Memory Operations (3 ops)

- `ldg(ptr)`, `load_global(ptr)` → `__ldg`
- `prefetch_l1(ptr)` → `__prefetch_l1`
- `prefetch_l2(ptr)` → `__prefetch_l2`

### Special Functions (13 ops)

- `rcp()`, `recip()` → `__frcp_rn` - Fast reciprocal
- `fast_div()` → `__fdividef` - Fast division
- `saturate()`, `clamp_01()` → `__saturatef` - Saturate to [0,1]
- `j0()`, `j1()`, `jn()` - Bessel functions of first kind
- `y0()`, `y1()`, `yn()` - Bessel functions of second kind
- `normcdf()`, `normcdfinv()` - Normal CDF
- `cyl_bessel_i0()`, `cyl_bessel_i1()` - Cylindrical Bessel

### Clock and Timing (3 ops)

- `clock()` → `clock()` - 32-bit clock counter
- `clock64()` → `clock64()` - 64-bit clock counter
- `nanosleep(ns)` → `__nanosleep` - Sleep for nanoseconds

## Ring Kernel Intrinsics

### Control Block Access

- `is_active()` - Check if kernel is active
- `should_terminate()` - Check termination signal
- `mark_terminated()` - Mark kernel as terminated
- `messages_processed()` - Get processed message count

### Queue Operations

- `input_queue_empty()` - Check if input queue is empty
- `output_queue_empty()` - Check if output queue is empty
- `input_queue_size()` - Get input queue size
- `output_queue_size()` - Get output queue size
- `enqueue_response(&resp)` - Enqueue response to output

### HLC Operations

- `hlc_tick()` - Increment logical clock
- `hlc_update(received_ts)` - Update clock with received timestamp
- `hlc_now()` - Get current HLC timestamp

### K2K Operations

**Standard K2K (raw messages):**
- `k2k_send(target_id, &msg)` - Send message to another kernel
- `k2k_try_recv()` - Try to receive K2K message
- `k2k_peek()` - Peek at next message without consuming
- `k2k_has_message()` - Check if K2K messages pending
- `k2k_pending_count()` - Get count of pending messages

**Envelope-aware K2K (when `use_envelope_format(true)`):**
- `k2k_send_envelope(target_id, envelope_ptr)` - Send envelope with headers
- `k2k_try_recv_envelope()` - Receive envelope with header validation
- `k2k_peek_envelope()` - Peek at next envelope without consuming

## Validation Modes

The transpiler uses validation modes to control what constructs are allowed:

| Mode | Loops | Description |
|------|-------|-------------|
| `Stencil` | Forbidden | Classic stencil kernels (default) |
| `Generic` | Allowed | General-purpose kernels |
| `RingKernel` | Required | Persistent actor kernels |

## Examples

The repository includes working examples demonstrating all three kernel types:

```bash
# Global kernels - SAXPY, halo exchange, array initialization
cargo run -p ringkernel --example global_kernel

# Stencil kernels - FDTD wave equation, heat diffusion, box filter
cargo run -p ringkernel --example stencil_kernel

# Ring kernels - Persistent actor model with HLC and K2K
cargo run -p ringkernel --example ring_kernel_codegen
```

See the `/examples/cuda-codegen/` directory for the full source code.

## Testing

The crate includes 183 tests (171 unit + 12 integration) covering all features:

```bash
cargo test -p ringkernel-cuda-codegen
```

**Test Categories:**
- Global kernel transpilation (types, intrinsics, control flow)
- Stencil kernel generation (GridPos, tile configs)
- Ring kernel actors (ControlBlock, message loop, HLC)
- Envelope format (MessageHeader, serialization, K2K)
- 120+ GPU intrinsics across 13 categories

## Benchmark Results

The transpiler generates CUDA code that achieves impressive performance on NVIDIA RTX Ada:

| Operation | Throughput | Batch Time | vs CPU |
|-----------|------------|------------|--------|
| CUDA Codegen (1M floats) | ~93B elem/sec | 0.5 µs | 12,378x |
| CUDA SAXPY PTX | ~77B elem/sec | 0.6 µs | 10,258x |
| Stencil Pattern Detection | ~15.7M TPS | 262 µs | 2.09x |

Run the benchmark:
```bash
cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen
```

## Next: [WGSL Code Generation](./14-wgpu-codegen.md)
