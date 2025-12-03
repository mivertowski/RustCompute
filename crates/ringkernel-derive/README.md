# ringkernel-derive

Procedural macros for RingKernel.

## Overview

This crate provides derive macros and attribute macros for defining RingKernel messages, handlers, and GPU kernels.

## Macros

### `#[derive(RingMessage)]`

Implements the `RingMessage` trait for message types:

```rust
use ringkernel_derive::RingMessage;

#[derive(RingMessage)]
#[message(type_id = 1)]
struct MyMessage {
    #[message(id)]
    id: MessageId,
    #[message(correlation)]
    correlation: CorrelationId,
    #[message(priority)]
    priority: Priority,
    payload: Vec<f32>,
}
```

**Struct attributes:**
- `#[message(type_id = N)]` - Explicit message type ID (optional, auto-generated from name hash if omitted)

**Field attributes:**
- `#[message(id)]` - Mark as message ID field
- `#[message(correlation)]` - Mark as correlation ID field
- `#[message(priority)]` - Mark as priority field

### `#[ring_kernel]`

Defines a ring kernel handler with metadata:

```rust
use ringkernel_derive::ring_kernel;

#[ring_kernel(id = "processor", mode = "persistent", block_size = 128)]
async fn handle(ctx: &mut RingContext, msg: MyMessage) -> MyResponse {
    MyResponse {
        id: MessageId::generate(),
        result: msg.a + msg.b,
    }
}
```

**Attributes:**
- `id` (required) - Unique kernel identifier
- `mode` - `"persistent"` (default) or `"event_driven"`
- `grid_size` - Number of blocks (default: 1)
- `block_size` - Threads per block (default: 256)
- `publishes_to` - Comma-separated list of target kernel IDs

### `#[derive(GpuType)]`

Marks a type as GPU-compatible for direct transfer:

```rust
use ringkernel_derive::GpuType;

#[derive(Copy, Clone, GpuType)]
#[repr(C)]
struct Matrix4x4 {
    data: [f32; 16],
}
```

Generates compile-time assertions for `Copy` and implements `bytemuck::Pod` and `bytemuck::Zeroable`.

### `#[stencil_kernel]`

Defines a stencil kernel that transpiles to CUDA (requires `cuda-codegen` feature):

```rust
use ringkernel_derive::stencil_kernel;

#[stencil_kernel(id = "fdtd", grid = "2d", tile_size = 16, halo = 1)]
fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
    let curr = p[pos.idx()];
    let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p) - 4.0 * curr;
    p_prev[pos.idx()] = 2.0 * curr - p_prev[pos.idx()] + c2 * lap;
}

// Access generated CUDA source:
assert!(FDTD_CUDA_SOURCE.contains("__global__"));
```

**Attributes:**
- `id` (required) - Kernel identifier
- `grid` - `"1d"`, `"2d"` (default), or `"3d"`
- `tile_size` - Square tile size (default: 16)
- `tile_width` / `tile_height` - Non-square tile dimensions
- `halo` - Stencil radius (default: 1)

## Generated Code

The macros generate:
- Trait implementations (`RingMessage`)
- Handler wrapper functions
- Kernel registration via `inventory`
- CUDA source constants (for `stencil_kernel`)

## Testing

```bash
cargo test -p ringkernel-derive
```

The crate includes 14 tests covering all macro functionality.

## License

Apache-2.0
