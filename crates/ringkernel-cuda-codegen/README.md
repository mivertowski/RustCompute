# ringkernel-cuda-codegen

Rust-to-CUDA transpiler for RingKernel GPU kernels.

## Overview

This crate enables writing GPU kernels in a restricted Rust DSL and transpiling them to CUDA C code. It supports three kernel types:

1. **Global Kernels** - Standard CUDA `__global__` functions
2. **Stencil Kernels** - Tile-based kernels with `GridPos` abstraction (2D and 3D)
3. **Ring Kernels** - Persistent actor kernels with message loops

## Installation

```toml
[dependencies]
ringkernel-cuda-codegen = "0.2"
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

For grid-based computations with neighbor access (2D and 3D):

```rust
use ringkernel_cuda_codegen::{transpile_stencil_kernel, StencilConfig, Grid};

// 2D stencil
let func: syn::ItemFn = parse_quote! {
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

let cuda_code = transpile_stencil_kernel(&func, &config)?;

// 3D stencil with up/down neighbors
let func_3d: syn::ItemFn = parse_quote! {
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
    .with_hlc(true)           // Hybrid Logical Clocks
    .with_k2k(true)           // Kernel-to-kernel messaging
    .with_envelope_format(true)  // MessageEnvelope serialization
    .with_kernel_id(1)        // Kernel ID for routing
    .with_hlc_node_id(1);     // HLC node ID

let cuda_code = transpile_ring_kernel(&handler, &config)?;
```

### Envelope Format

When `with_envelope_format(true)` is enabled, messages use a standardized `MessageEnvelope` format:

```c
// MessageHeader (256 bytes, cache-aligned)
typedef struct __align__(256) {
    uint32_t magic;           // 0xCAFEBABE
    uint32_t version;         // Protocol version
    uint64_t type_id;         // Message type identifier
    uint64_t envelope_id;     // Unique envelope ID
    uint64_t correlation_id;  // Request/response correlation
    uint64_t source_kernel;   // Source kernel ID
    uint64_t target_kernel;   // Target kernel ID
    uint64_t hlc_wall;        // HLC wall clock
    uint64_t hlc_logical;     // HLC logical counter
    uint32_t hlc_node;        // HLC node ID
    uint32_t priority;        // Message priority
    uint32_t payload_size;    // Payload size in bytes
    uint32_t flags;           // Message flags
    uint8_t reserved[168];    // Padding to 256 bytes
} MessageHeader;

// MessageEnvelope = header + payload
```

This enables:
- Automatic HLC timestamp propagation across kernels
- Kernel-to-kernel routing with source/target tracking
- Correlation ID for request/response patterns
- Priority-based message ordering

## DSL Reference

### Thread/Block Indices
- `thread_idx_x()`, `thread_idx_y()`, `thread_idx_z()` → `threadIdx.x/y/z`
- `block_idx_x()`, `block_idx_y()`, `block_idx_z()` → `blockIdx.x/y/z`
- `block_dim_x()`, `block_dim_y()`, `block_dim_z()` → `blockDim.x/y/z`
- `grid_dim_x()`, `grid_dim_y()`, `grid_dim_z()` → `gridDim.x/y/z`
- `warp_size()` → `warpSize`

### Stencil Intrinsics (2D)
- `pos.idx()` - Linear index
- `pos.north(buf)`, `pos.south(buf)` - Y-axis neighbors
- `pos.east(buf)`, `pos.west(buf)` - X-axis neighbors
- `pos.at(buf, dx, dy)` - Relative offset access

### Stencil Intrinsics (3D)
- `pos.up(buf)`, `pos.down(buf)` - Z-axis neighbors
- `pos.at(buf, dx, dy, dz)` - 3D relative offset access

### Synchronization
- `sync_threads()` → `__syncthreads()` - Block-level barrier
- `sync_threads_count(pred)` → `__syncthreads_count()` - Count threads with predicate
- `sync_threads_and(pred)` → `__syncthreads_and()` - AND of predicate
- `sync_threads_or(pred)` → `__syncthreads_or()` - OR of predicate
- `thread_fence()` → `__threadfence()` - Device memory fence
- `thread_fence_block()` → `__threadfence_block()` - Block memory fence
- `thread_fence_system()` → `__threadfence_system()` - System memory fence

### Atomic Operations (Integer)
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

### Basic Math Functions
- `sqrt()`, `rsqrt()` - Square root, reciprocal sqrt
- `abs()`, `fabs()` - Absolute value
- `floor()`, `ceil()`, `round()`, `trunc()` - Rounding
- `fma()`, `mul_add()` - Fused multiply-add
- `fmin()`, `fmax()` - Minimum, maximum
- `fmod()`, `remainder()` - Modulo operations
- `copysign()` - Copy sign
- `cbrt()` - Cube root
- `hypot()` - Hypotenuse

### Trigonometric Functions
- `sin()`, `cos()`, `tan()` - Basic trig
- `asin()`, `acos()`, `atan()`, `atan2()` - Inverse trig
- `sincos()` - Combined sine and cosine
- `sinpi()`, `cospi()` - Sin/cos of π*x

### Hyperbolic Functions
- `sinh()`, `cosh()`, `tanh()` - Hyperbolic
- `asinh()`, `acosh()`, `atanh()` - Inverse hyperbolic

### Exponential and Logarithmic Functions
- `exp()`, `exp2()`, `exp10()`, `expm1()` - Exponentials
- `log()`, `ln()`, `log2()`, `log10()`, `log1p()` - Logarithms
- `pow()`, `powf()`, `powi()` - Power
- `ldexp()`, `scalbn()` - Load/scale exponent
- `ilogb()` - Extract exponent
- `erf()`, `erfc()`, `erfinv()`, `erfcinv()` - Error functions
- `lgamma()`, `tgamma()` - Gamma functions

### Classification Functions
- `is_nan()`, `isnan()` → `isnan`
- `is_infinite()`, `isinf()` → `isinf`
- `is_finite()`, `isfinite()` → `isfinite`
- `is_normal()`, `isnormal()` → `isnormal`
- `signbit()` - Check sign bit
- `nextafter()` - Next representable value
- `fdim()` - Positive difference

### Warp Operations
- `warp_active_mask()` → `__activemask()` - Active lane mask
- `warp_shfl(mask, val, lane)` → `__shfl_sync` - Shuffle
- `warp_shfl_up(mask, val, delta)` → `__shfl_up_sync`
- `warp_shfl_down(mask, val, delta)` → `__shfl_down_sync`
- `warp_shfl_xor(mask, val, lane_mask)` → `__shfl_xor_sync`
- `warp_ballot(mask, pred)` → `__ballot_sync`
- `warp_all(mask, pred)` → `__all_sync`
- `warp_any(mask, pred)` → `__any_sync`

### Warp Match Operations (Volta+)
- `warp_match_any(mask, val)` → `__match_any_sync`
- `warp_match_all(mask, val)` → `__match_all_sync`

### Warp Reduce Operations (SM 8.0+)
- `warp_reduce_add(mask, val)` → `__reduce_add_sync`
- `warp_reduce_min(mask, val)` → `__reduce_min_sync`
- `warp_reduce_max(mask, val)` → `__reduce_max_sync`
- `warp_reduce_and(mask, val)` → `__reduce_and_sync`
- `warp_reduce_or(mask, val)` → `__reduce_or_sync`
- `warp_reduce_xor(mask, val)` → `__reduce_xor_sync`

### Bit Manipulation
- `popc()`, `popcount()`, `count_ones()` → `__popc` - Population count
- `clz()`, `leading_zeros()` → `__clz` - Count leading zeros
- `ctz()`, `trailing_zeros()` → `__ffs - 1` - Count trailing zeros
- `ffs()` → `__ffs` - Find first set
- `brev()`, `reverse_bits()` → `__brev` - Bit reverse
- `byte_perm()` → `__byte_perm` - Byte permutation
- `funnel_shift_left()` → `__funnelshift_l`
- `funnel_shift_right()` → `__funnelshift_r`

### Memory Operations
- `ldg(ptr)`, `load_global(ptr)` → `__ldg` - Read-only cache load
- `prefetch_l1(ptr)` → `__prefetch_l1` - L1 prefetch
- `prefetch_l2(ptr)` → `__prefetch_l2` - L2 prefetch

### Special Functions
- `rcp()`, `recip()` → `__frcp_rn` - Fast reciprocal
- `fast_div()` → `__fdividef` - Fast division
- `saturate()`, `clamp_01()` → `__saturatef` - Saturate to [0,1]
- `j0()`, `j1()`, `jn()` - Bessel functions of first kind
- `y0()`, `y1()`, `yn()` - Bessel functions of second kind
- `normcdf()`, `normcdfinv()` - Normal CDF
- `cyl_bessel_i0()`, `cyl_bessel_i1()` - Cylindrical Bessel functions

### Clock and Timing
- `clock()` → `clock()` - 32-bit clock counter
- `clock64()` → `clock64()` - 64-bit clock counter
- `nanosleep(ns)` → `__nanosleep` - Sleep for nanoseconds

### RingContext Methods
- `ctx.thread_id()` → `threadIdx.x`
- `ctx.block_id()` → `blockIdx.x`
- `ctx.global_thread_id()` → `(blockIdx.x * blockDim.x + threadIdx.x)`
- `ctx.sync_threads()` → `__syncthreads()`
- `ctx.lane_id()` → `(threadIdx.x % 32)`
- `ctx.warp_id()` → `(threadIdx.x / 32)`

### Ring Kernel Intrinsics
- `is_active()`, `should_terminate()`, `mark_terminated()`
- `messages_processed()`, `input_queue_size()`, `output_queue_size()`
- `input_queue_empty()`, `output_queue_empty()`, `enqueue_response(&resp)`
- `hlc_tick()`, `hlc_update(ts)`, `hlc_now()` - HLC operations

### K2K Messaging (Envelope-Aware)
- `k2k_send(target, &msg)` - Send message to target kernel
- `k2k_send_envelope(&envelope)` - Send full envelope with routing
- `k2k_try_recv()` - Non-blocking receive
- `k2k_try_recv_envelope()` - Receive with envelope metadata
- `k2k_has_message()`, `k2k_peek()`, `k2k_pending_count()`
- `k2k_get_source_kernel()` - Get source kernel ID from envelope
- `k2k_get_correlation_id()` - Get correlation ID for request/response

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

## Intrinsic Count

The transpiler supports **120+ GPU intrinsics** across 13 categories:

| Category | Count | Examples |
|----------|-------|----------|
| Synchronization | 7 | `sync_threads`, `thread_fence` |
| Atomics | 11 | `atomic_add`, `atomic_cas`, `atomic_and` |
| Math | 16 | `sqrt`, `fma`, `cbrt`, `hypot` |
| Trigonometric | 11 | `sin`, `asin`, `atan2`, `sincos` |
| Hyperbolic | 6 | `sinh`, `asinh` |
| Exponential | 18 | `exp`, `log2`, `erf`, `gamma` |
| Classification | 8 | `isnan`, `isfinite`, `signbit` |
| Warp | 16 | `warp_shfl`, `warp_reduce_add`, `warp_match_any` |
| Bit Manipulation | 8 | `popc`, `clz`, `brev`, `funnel_shift_left` |
| Memory | 3 | `ldg`, `prefetch_l1` |
| Special | 13 | `rcp`, `saturate`, `normcdf` |
| Index | 13 | `thread_idx_x`, `warp_size` |
| Timing | 3 | `clock`, `clock64`, `nanosleep` |

## Testing

```bash
cargo test -p ringkernel-cuda-codegen
```

The crate includes 183 tests covering all kernel types, intrinsics, envelope format, and language features.

## License

Apache-2.0
