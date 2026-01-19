---
layout: default
title: Proc Macros
nav_order: 8
---

# Proc Macros

## Overview

DotCompute uses C# Source Generators for:
- `[RingKernel]` attribute → CUDA kernel generation
- `[MemoryPackable]` → serialization code
- `[RingKernelMessage]` → message infrastructure

Rust equivalent: **Procedural Macros** (proc-macros)

---

## #[derive(RingMessage)]

Generates `RingMessage` trait implementation:

```rust
// crates/ringkernel-derive/src/ring_message.rs

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

#[proc_macro_derive(RingMessage, attributes(message))]
pub fn derive_ring_message(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();

    // Find fields with #[message(...)] attributes
    let (id_field, priority_field, correlation_field) = extract_message_fields(&input);

    let expanded = quote! {
        impl ::ringkernel_core::RingMessage for #name {
            const MESSAGE_TYPE: &'static str = #name_str;

            fn message_id(&self) -> ::uuid::Uuid {
                self.#id_field
            }

            fn set_message_id(&mut self, id: ::uuid::Uuid) {
                self.#id_field = id;
            }

            fn priority(&self) -> u8 {
                self.#priority_field
            }

            fn correlation_id(&self) -> Option<::uuid::Uuid> {
                self.#correlation_field
            }

            fn payload_size(&self) -> usize {
                ::std::mem::size_of::<Self>() // Simplified; real impl uses rkyv
            }
        }
    };

    TokenStream::from(expanded)
}

fn extract_message_fields(input: &DeriveInput) -> (syn::Ident, syn::Ident, syn::Ident) {
    // Parse #[message(id)], #[message(priority)], #[message(correlation)]
    // ... implementation
    todo!()
}
```

### Usage

```rust
use ringkernel::prelude::*;
use rkyv::{Archive, Serialize, Deserialize};

#[derive(RingMessage, Archive, Serialize, Deserialize)]
pub struct ComputeRequest {
    #[message(id)]
    pub id: Uuid,

    #[message(priority = "priority::HIGH")]
    pub priority: u8,

    #[message(correlation)]
    pub correlation_id: Option<Uuid>,

    // Payload
    pub input_data: Vec<f32>,
    pub operation: OperationType,
}
```

### Domain Attribute (v0.3.0)

Assign messages to business domains for automatic type ID allocation:

```rust
use ringkernel::prelude::*;

#[derive(RingMessage)]
#[message(domain = "FraudDetection")]  // Type ID auto-assigned from 1000-1099 range
pub struct SuspiciousTransaction {
    #[message(id)]
    pub id: Uuid,
    pub account_id: u64,
    pub amount: f64,
    pub risk_score: f32,
}

#[derive(RingMessage)]
#[message(domain = "ProcessIntelligence", type_id = 1505)]  // Explicit ID within domain range
pub struct ActivityEvent {
    #[message(id)]
    pub id: Uuid,
    pub case_id: u64,
    pub activity: String,
}
```

Available domains and their type ID ranges:

| Domain | Range | Description |
|--------|-------|-------------|
| `General` | 0-99 | Default, unclassified |
| `GraphAnalytics` | 100-199 | Graph algorithms |
| `StatisticalML` | 200-299 | ML/statistics |
| `Compliance` | 300-399 | Regulatory compliance |
| `RiskManagement` | 400-499 | Risk analysis |
| `OrderMatching` | 500-599 | Trading systems |
| `MarketData` | 600-699 | Market feeds |
| `Settlement` | 700-799 | Trade settlement |
| `Accounting` | 800-899 | Financial accounting |
| `NetworkAnalysis` | 900-999 | Network analysis |
| `FraudDetection` | 1000-1099 | Fraud detection |
| `TimeSeries` | 1100-1199 | Time series analysis |
| `Simulation` | 1200-1299 | Simulations |
| `Banking` | 1300-1399 | Banking operations |
| `BehavioralAnalytics` | 1400-1499 | Behavioral analysis |
| `ProcessIntelligence` | 1500-1599 | Process mining |
| `Clearing` | 1600-1699 | Clearing operations |
| `TreasuryManagement` | 1700-1799 | Treasury |
| `PaymentProcessing` | 1800-1899 | Payments |
| `FinancialAudit` | 1900-1999 | Auditing |
| `Custom` | 10000+ | User-defined |

---

## #[derive(PersistentMessage)] (v0.3.0)

For messages that are dispatched to GPU kernel handlers:

```rust
use ringkernel::prelude::*;

#[derive(PersistentMessage)]
#[persistent_message(handler_id = 1)]
pub struct MatrixMultiplyRequest {
    pub matrix_a: Vec<f32>,
    pub matrix_b: Vec<f32>,
    pub rows_a: u32,
    pub cols_a: u32,
    pub cols_b: u32,
}

#[derive(PersistentMessage)]
#[persistent_message(handler_id = 1, requires_response = true)]
pub struct MatrixMultiplyResponse {
    pub result: Vec<f32>,
    pub compute_time_ns: u64,
}
```

### Attributes

| Attribute | Description |
|-----------|-------------|
| `handler_id = N` | Handler identifier for dispatch routing (required) |
| `requires_response = bool` | Whether the message expects a response (default: false) |
| `domain = "Name"` | Optional domain classification |

### Generated Code

```rust
impl ::ringkernel_core::persistent_message::PersistentMessage for MatrixMultiplyRequest {
    const HANDLER_ID: ::ringkernel_core::persistent_message::HandlerId =
        ::ringkernel_core::persistent_message::HandlerId(1);
    const REQUIRES_RESPONSE: bool = false;

    fn serialize_payload(&self) -> Vec<u8> {
        ::rkyv::to_bytes::<Self>(self).unwrap().to_vec()
    }

    fn deserialize_response(data: &[u8]) -> Option<Self::Response> {
        ::rkyv::from_bytes::<Self::Response>(data).ok()
    }

    type Response = ();
}
```

---

## #[derive(GpuType)] (v0.3.0)

Generate GPU-compatible type definitions for use in kernels:

```rust
use ringkernel::prelude::*;

#[derive(GpuType)]
#[repr(C)]
pub struct Particle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub mass: f32,
    pub charge: f32,
}

#[derive(GpuType)]
#[repr(C, align(16))]
pub struct Matrix4x4 {
    pub data: [f32; 16],
}
```

### Requirements

- Must have `#[repr(C)]` for C-compatible layout
- All fields must be `Copy` types
- Nested structs must also derive `GpuType`
- Supports alignment attributes (`#[repr(C, align(N))]`)

### Generated Code

The macro generates:
- CUDA struct definition with matching layout
- WGSL struct definition with proper alignment
- Size and alignment compile-time assertions
- Serialization helpers for GPU transfer

---

## #[ring_kernel]

The main macro for defining ring kernels:

```rust
// crates/ringkernel-derive/src/ring_kernel.rs

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, AttributeArgs};

#[proc_macro_attribute]
pub fn ring_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as AttributeArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let config = parse_kernel_config(&args);
    let fn_name = &input_fn.sig.ident;
    let kernel_id = config.id.unwrap_or_else(|| fn_name.to_string());

    // Extract input/output types from signature
    let (input_type, output_type) = extract_message_types(&input_fn.sig);

    // Generate kernel registration
    let registration = generate_registration(&kernel_id, &input_type, &output_type);

    // Generate GPU code (CUDA/Metal/WGSL)
    let gpu_code = generate_gpu_kernel(&input_fn, &config);

    let expanded = quote! {
        // Original function (for CPU backend / testing)
        #input_fn

        // Kernel registration
        #registration

        // GPU kernel source (embedded as const)
        #gpu_code
    };

    TokenStream::from(expanded)
}

#[derive(Default)]
struct KernelConfig {
    id: Option<String>,
    mode: RingKernelMode,
    grid_size: Option<(u32, u32, u32)>,
    block_size: Option<(u32, u32, u32)>,
    publishes_to: Vec<String>,
    subscribes_to: Vec<String>,
    named_barriers: Vec<String>,
}

fn parse_kernel_config(args: &AttributeArgs) -> KernelConfig {
    let mut config = KernelConfig::default();

    for arg in args {
        match arg {
            // Parse: id = "kernel_name"
            // Parse: mode = "persistent" | "event_driven"
            // Parse: publishes_to = ["other_kernel"]
            // ...
        }
    }

    config
}
```

### Usage

```rust
use ringkernel::prelude::*;

#[ring_kernel(
    id = "vector_add",
    mode = "persistent",
    block_size = (256, 1, 1),
    publishes_to = ["aggregator"],
)]
pub fn vector_add(ctx: &mut RingContext, req: VectorAddRequest) -> VectorAddResponse {
    ctx.sync_threads();

    let idx = ctx.global_thread_id() as usize;
    if idx < req.a.len() {
        let result = req.a[idx] + req.b[idx];
        // ...
    }

    VectorAddResponse { /* ... */ }
}
```

### Generated Code

```rust
// Kernel registration
inventory::submit! {
    ::ringkernel_core::KernelRegistration {
        id: "vector_add",
        input_type: ::std::any::TypeId::of::<VectorAddRequest>(),
        output_type: ::std::any::TypeId::of::<VectorAddResponse>(),
        mode: ::ringkernel_core::RingKernelMode::Persistent,
        handler: |ctx, bytes| {
            let req: VectorAddRequest = ::rkyv::from_bytes(bytes)?;
            let resp = vector_add(ctx, req);
            Ok(::rkyv::to_bytes(&resp)?)
        },
    }
}

// CUDA source (for CUDA backend)
#[cfg(feature = "cuda")]
pub const VECTOR_ADD_CUDA: &str = r#"
__global__ void vector_add_kernel(
    ControlBlock* control,
    unsigned char* input_buffer,
    cuda::atomic<int>* input_head,
    // ...
) {
    // Generated persistent loop
    while (!control->should_terminate.load()) {
        // Dequeue, deserialize, process, enqueue
    }
}
"#;
```

---

## Context Method Translation

The proc macro translates `RingContext` method calls to GPU intrinsics:

```rust
/// Mapping from Rust methods to GPU intrinsics
const INTRINSIC_MAP: &[(&str, IntrinsicInfo)] = &[
    ("sync_threads", IntrinsicInfo {
        cuda: "__syncthreads()",
        metal: "threadgroup_barrier(mem_flags::mem_threadgroup)",
        wgsl: "workgroupBarrier()",
    }),
    ("sync_warp", IntrinsicInfo {
        cuda: "__syncwarp({mask})",
        metal: "simdgroup_barrier(mem_flags::mem_none)",
        wgsl: "/* no equivalent */",
    }),
    ("atomic_add", IntrinsicInfo {
        cuda: "atomicAdd({ptr}, {value})",
        metal: "atomic_fetch_add_explicit({ptr}, {value}, memory_order_relaxed)",
        wgsl: "atomicAdd({ptr}, {value})",
    }),
    ("now", IntrinsicInfo {
        cuda: "make_hlc_timestamp(clock64(), hlc_logical++, kernel_id)",
        metal: "make_hlc_timestamp(/* metal clock */, hlc_logical++, kernel_id)",
        wgsl: "/* limited timing support */",
    }),
    // ... more mappings
];

fn translate_to_cuda(rust_code: &syn::ItemFn) -> String {
    let mut cuda_code = String::new();

    // Walk AST and translate method calls
    for stmt in &rust_code.block.stmts {
        cuda_code.push_str(&translate_statement(stmt, Backend::Cuda));
    }

    cuda_code
}
```

---

## Compile-Time Validation

```rust
// Validate kernel signature at compile time
fn validate_kernel_signature(sig: &syn::Signature) -> syn::Result<()> {
    // Must have exactly 2 parameters: ctx and message
    if sig.inputs.len() != 2 {
        return Err(syn::Error::new_spanned(
            sig,
            "Ring kernel must have signature: fn(ctx: &mut RingContext, msg: T) -> R"
        ));
    }

    // First param must be &mut RingContext
    // Second param must implement RingMessage
    // Return type must implement RingMessage

    Ok(())
}

// Validate message types at compile time
fn validate_message_types(input: &syn::Type, output: &syn::Type) -> syn::Result<()> {
    // Check that types implement RingMessage + Archive + Serialize + Deserialize
    // This is done via trait bounds, but we can add helpful errors

    Ok(())
}
```

---

## Inventory Pattern for Registration

Use the `inventory` crate for automatic kernel discovery:

```rust
// crates/ringkernel-core/src/registry.rs

use inventory;

/// Kernel registration entry.
pub struct KernelRegistration {
    pub id: &'static str,
    pub input_type: std::any::TypeId,
    pub output_type: std::any::TypeId,
    pub mode: RingKernelMode,
    pub handler: fn(&mut RingContext, &[u8]) -> Result<Vec<u8>>,
}

inventory::collect!(KernelRegistration);

/// Get all registered kernels.
pub fn registered_kernels() -> impl Iterator<Item = &'static KernelRegistration> {
    inventory::iter::<KernelRegistration>()
}

/// Find kernel by ID.
pub fn find_kernel(id: &str) -> Option<&'static KernelRegistration> {
    registered_kernels().find(|k| k.id == id)
}
```

---

## CUDA Code Generation (ringkernel-cuda-codegen)

The `ringkernel-cuda-codegen` crate provides a Rust-to-CUDA transpiler that allows writing GPU kernels in a Rust DSL:

### Generic Kernel Transpilation

```rust
use ringkernel_cuda_codegen::{transpile_global_kernel, dsl::*};
use syn::parse_quote;

// Write kernel in Rust DSL
let kernel: syn::ItemFn = parse_quote! {
    fn exchange_halos(buffer: &mut [f32], copies: &[u32], num_copies: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= num_copies { return; }
        let src = copies[(idx * 2) as usize];
        let dst = copies[(idx * 2 + 1) as usize];
        buffer[dst as usize] = buffer[src as usize];
    }
};

// Transpile to CUDA C
let cuda = transpile_global_kernel(&kernel)?;
```

Generated output:
```cuda
extern "C" __global__ void exchange_halos(
    float* __restrict__ buffer,
    const unsigned int* __restrict__ copies,
    int num_copies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_copies) return;
    int src = copies[idx * 2];
    int dst = copies[idx * 2 + 1];
    buffer[dst] = buffer[src];
}
```

### Stencil Kernel Transpilation

For stencil computations, the transpiler supports a `GridPos` abstraction:

```rust
use ringkernel_cuda_codegen::{transpile_stencil_kernel, StencilConfig};

let stencil: syn::ItemFn = parse_quote! {
    fn fdtd_step(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
        let curr = p[pos.idx()];
        let prev = p_prev[pos.idx()];
        let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p) - 4.0 * curr;
        p_prev[pos.idx()] = 2.0 * curr - prev + c2 * lap;
    }
};

let config = StencilConfig::new("fdtd").with_tile_size(16, 16).with_halo(1);
let cuda = transpile_stencil_kernel(&stencil, &config)?;
```

### Supported DSL Features

| Rust DSL | CUDA Output |
|----------|-------------|
| `thread_idx_x()` | `threadIdx.x` |
| `block_idx_x()` | `blockIdx.x` |
| `block_dim_x()` | `blockDim.x` |
| `grid_dim_x()` | `gridDim.x` |
| `if cond { return; }` | `if (cond) return;` |
| `match expr { ... }` | `switch (expr) { ... }` |
| `pos.north(buf)` | `buf[idx - width]` |
| `pos.south(buf)` | `buf[idx + width]` |
| `sync_threads()` | `__syncthreads()` |

### Type Mapping

| Rust Type | CUDA Type |
|-----------|-----------|
| `f32` | `float` |
| `f64` | `double` |
| `i32` | `int` |
| `u32` | `unsigned int` |
| `i64` | `long long` |
| `&[f32]` | `const float* __restrict__` |
| `&mut [f32]` | `float* __restrict__` |

---

## Next: [Runtime Implementation](./08-runtime.md)
