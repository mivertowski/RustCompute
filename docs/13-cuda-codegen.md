# CUDA Code Generation

The `ringkernel-cuda-codegen` crate provides a Rust-to-CUDA transpiler that lets you write GPU kernels in a Rust DSL and generate equivalent CUDA C code.

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
use ringkernel_cuda_codegen::{transpile_stencil_kernel, StencilConfig};

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

let cuda_code = transpile_stencil_kernel(&kernel, &config)?;
```

**Stencil Intrinsics:**
- `pos.idx()` - Current cell linear index
- `pos.north(buf)` - Access cell above
- `pos.south(buf)` - Access cell below
- `pos.east(buf)` - Access cell to the right
- `pos.west(buf)` - Access cell to the left
- `pos.at(buf, dx, dy)` - Access cell at relative offset

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
    .with_hlc(true)
    .with_k2k(true);

let cuda_code = transpile_ring_kernel(&handler, &config)?;
```

### RingKernelConfig Options

| Option | Default | Description |
|--------|---------|-------------|
| `block_size` | 128 | Threads per block |
| `queue_capacity` | 1024 | Input/output queue size (must be power of 2) |
| `enable_hlc` | true | Enable Hybrid Logical Clocks |
| `enable_k2k` | false | Enable kernel-to-kernel messaging |
| `message_size` | 64 | Message buffer element size |
| `response_size` | 64 | Response buffer element size |
| `idle_sleep_ns` | 1000 | Nanosleep duration when idle |

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

### Thread/Block Indices

- `thread_idx_x()`, `thread_idx_y()`, `thread_idx_z()`
- `block_idx_x()`, `block_idx_y()`, `block_idx_z()`
- `block_dim_x()`, `block_dim_y()`, `block_dim_z()`
- `grid_dim_x()`, `grid_dim_y()`, `grid_dim_z()`

### Synchronization

- `sync_threads()` → `__syncthreads()`
- `thread_fence()` → `__threadfence()`
- `thread_fence_block()` → `__threadfence_block()`

### Atomics

- `atomic_add(ptr, val)` → `atomicAdd(ptr, val)`
- `atomic_sub(ptr, val)` → `atomicSub(ptr, val)`
- `atomic_min(ptr, val)` → `atomicMin(ptr, val)`
- `atomic_max(ptr, val)` → `atomicMax(ptr, val)`
- `atomic_exchange(ptr, val)` → `atomicExch(ptr, val)`
- `atomic_cas(ptr, compare, val)` → `atomicCAS(ptr, compare, val)`

### Math Functions

- `sqrt()`, `abs()`, `floor()`, `ceil()`, `round()`
- `sin()`, `cos()`, `tan()`, `exp()`, `log()`
- `powf()`, `min()`, `max()`, `mul_add()` (FMA)

### Warp Operations

- `warp_shuffle(val, lane)` → `__shfl_sync(FULL_MASK, val, lane)`
- `warp_shuffle_up(val, delta)` → `__shfl_up_sync(...)`
- `warp_shuffle_down(val, delta)` → `__shfl_down_sync(...)`
- `warp_shuffle_xor(val, mask)` → `__shfl_xor_sync(...)`
- `warp_ballot(pred)` → `__ballot_sync(...)`
- `warp_all(pred)` → `__all_sync(...)`
- `warp_any(pred)` → `__any_sync(...)`

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

- `k2k_send(target_id, &msg)` - Send message to another kernel
- `k2k_try_recv()` - Try to receive K2K message
- `k2k_peek()` - Peek at next message without consuming
- `k2k_has_message()` - Check if K2K messages pending
- `k2k_pending_count()` - Get count of pending messages

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

The crate includes 138 tests covering all features:

```bash
cargo test -p ringkernel-cuda-codegen
```

## Next: [Migration Guide](./12-migration.md)
