//! CUDA code generation from Rust DSL for RingKernel stencil kernels.
//!
//! This crate provides transpilation from a restricted Rust DSL to CUDA C code,
//! enabling developers to write GPU kernels in Rust without directly writing CUDA.
//!
//! # Overview
//!
//! The transpiler supports a subset of Rust focused on stencil/grid operations:
//!
//! - Primitive types: `f32`, `f64`, `i32`, `u32`, `i64`, `u64`, `bool`
//! - Array slices: `&[T]`, `&mut [T]`
//! - Arithmetic and comparison operators
//! - Let bindings and if/else expressions
//! - Stencil intrinsics via `GridPos` context
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cuda_codegen::{transpile_stencil_kernel, StencilConfig};
//!
//! let rust_code = r#"
//!     fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
//!         let curr = p[pos.idx()];
//!         let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p) - 4.0 * curr;
//!         p_prev[pos.idx()] = 2.0 * curr - p_prev[pos.idx()] + c2 * lap;
//!     }
//! "#;
//!
//! let config = StencilConfig {
//!     id: "fdtd".to_string(),
//!     grid: Grid::Grid2D,
//!     tile_size: (16, 16),
//!     halo: 1,
//! };
//!
//! let cuda_code = transpile_stencil_kernel(rust_code, &config)?;
//! ```

pub mod dsl;
pub mod handler;
mod intrinsics;
pub mod loops;
pub mod ring_kernel;
pub mod shared;
mod stencil;
mod transpiler;
mod types;
mod validation;

pub use handler::{
    generate_cuda_struct, generate_message_deser, generate_response_ser, ContextMethod,
    HandlerCodegenConfig, HandlerParam, HandlerParamKind, HandlerReturnType, HandlerSignature,
    MessageTypeInfo, MessageTypeRegistry,
};
pub use intrinsics::{GpuIntrinsic, IntrinsicRegistry, RingKernelIntrinsic, StencilIntrinsic};
pub use loops::{LoopPattern, RangeInfo};
pub use ring_kernel::{
    generate_control_block_struct, generate_hlc_struct, generate_k2k_structs, RingKernelConfig,
};
pub use shared::{SharedArray, SharedMemoryConfig, SharedMemoryDecl, SharedTile};
pub use stencil::{Grid, GridPos, StencilConfig, StencilLaunchConfig};
pub use transpiler::{transpile_function, CudaTranspiler, SharedVarInfo};
pub use types::{
    get_slice_element_type, is_control_block_type, is_mutable_reference, is_ring_context_type,
    ring_kernel_type_mapper, CudaType, RingKernelParamKind, TypeMapper,
};
pub use validation::{
    is_simple_assignment, validate_function, validate_function_with_mode,
    validate_stencil_signature, ValidationError, ValidationMode,
};

use thiserror::Error;

/// Errors that can occur during transpilation.
#[derive(Error, Debug)]
pub enum TranspileError {
    /// Failed to parse Rust code.
    #[error("Parse error: {0}")]
    Parse(String),

    /// DSL constraint violation.
    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    /// Unsupported Rust construct.
    #[error("Unsupported construct: {0}")]
    Unsupported(String),

    /// Type mapping failure.
    #[error("Type error: {0}")]
    Type(String),
}

/// Result type for transpilation operations.
pub type Result<T> = std::result::Result<T, TranspileError>;

/// Transpile a Rust stencil kernel function to CUDA C code.
///
/// This is the main entry point for code generation. It takes a parsed
/// Rust function and stencil configuration, validates the DSL constraints,
/// and generates equivalent CUDA C code.
///
/// # Arguments
///
/// * `func` - The parsed Rust function (from syn)
/// * `config` - Stencil kernel configuration
///
/// # Returns
///
/// The generated CUDA C source code as a string.
pub fn transpile_stencil_kernel(func: &syn::ItemFn, config: &StencilConfig) -> Result<String> {
    // Validate DSL constraints
    validate_function(func)?;

    // Create transpiler with stencil config
    let mut transpiler = CudaTranspiler::new(config.clone());

    // Generate CUDA code
    transpiler.transpile_stencil(func)
}

/// Transpile a Rust function to a CUDA `__device__` function.
///
/// This generates a device-callable function (not a kernel) from Rust code.
pub fn transpile_device_function(func: &syn::ItemFn) -> Result<String> {
    validate_function(func)?;
    transpile_function(func)
}

/// Transpile a Rust function to a CUDA `__global__` kernel.
///
/// This generates an externally-callable kernel without stencil-specific patterns.
/// Use DSL functions like `thread_idx_x()`, `block_idx_x()` to access CUDA indices.
///
/// # Example
///
/// ```ignore
/// use ringkernel_cuda_codegen::transpile_global_kernel;
/// use syn::parse_quote;
///
/// let func: syn::ItemFn = parse_quote! {
///     fn exchange_halos(buffer: &mut [f32], copies: &[u32], num_copies: i32) {
///         let idx = block_idx_x() * block_dim_x() + thread_idx_x();
///         if idx >= num_copies { return; }
///         let src = copies[idx * 2] as usize;
///         let dst = copies[idx * 2 + 1] as usize;
///         buffer[dst] = buffer[src];
///     }
/// };
///
/// let cuda = transpile_global_kernel(&func)?;
/// // Generates: extern "C" __global__ void exchange_halos(...) { ... }
/// ```
pub fn transpile_global_kernel(func: &syn::ItemFn) -> Result<String> {
    validate_function(func)?;
    let mut transpiler = CudaTranspiler::new_generic();
    transpiler.transpile_generic_kernel(func)
}

/// Transpile a Rust handler function to a persistent ring kernel.
///
/// Ring kernels are persistent GPU kernels that process messages in a loop.
/// The handler function is embedded within the message processing loop.
///
/// # Example
///
/// ```ignore
/// use ringkernel_cuda_codegen::{transpile_ring_kernel, RingKernelConfig};
/// use syn::parse_quote;
///
/// let handler: syn::ItemFn = parse_quote! {
///     fn process_message(value: f32) -> f32 {
///         value * 2.0
///     }
/// };
///
/// let config = RingKernelConfig::new("processor")
///     .with_block_size(128)
///     .with_hlc(true);
///
/// let cuda = transpile_ring_kernel(&handler, &config)?;
/// // Generates a persistent kernel with message loop wrapping the handler
/// ```
pub fn transpile_ring_kernel(handler: &syn::ItemFn, config: &RingKernelConfig) -> Result<String> {
    // Validate handler with generic mode (loops allowed but not required)
    // The persistent loop is generated by the ring kernel wrapper, not the handler
    validate_function_with_mode(handler, ValidationMode::Generic)?;

    // Create transpiler in generic mode for the handler
    let mut transpiler = CudaTranspiler::with_mode(ValidationMode::Generic);

    // Transpile the handler into a ring kernel wrapper
    transpiler.transpile_ring_kernel(handler, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_simple_function_transpile() {
        let func: syn::ItemFn = parse_quote! {
            fn add(a: f32, b: f32) -> f32 {
                a + b
            }
        };

        let result = transpile_device_function(&func);
        assert!(
            result.is_ok(),
            "Should transpile simple function: {:?}",
            result
        );

        let cuda = result.unwrap();
        assert!(cuda.contains("float"), "Should contain CUDA float type");
        assert!(cuda.contains("a + b"), "Should contain the expression");
    }

    #[test]
    fn test_global_kernel_transpile() {
        let func: syn::ItemFn = parse_quote! {
            fn exchange_halos(buffer: &mut [f32], copies: &[u32], num_copies: i32) {
                let idx = block_idx_x() * block_dim_x() + thread_idx_x();
                if idx >= num_copies {
                    return;
                }
                let src = copies[idx * 2] as usize;
                let dst = copies[idx * 2 + 1] as usize;
                buffer[dst] = buffer[src];
            }
        };

        let result = transpile_global_kernel(&func);
        assert!(
            result.is_ok(),
            "Should transpile global kernel: {:?}",
            result
        );

        let cuda = result.unwrap();
        assert!(
            cuda.contains("extern \"C\" __global__"),
            "Should be global kernel"
        );
        assert!(cuda.contains("exchange_halos"), "Should have kernel name");
        assert!(cuda.contains("blockIdx.x"), "Should contain blockIdx.x");
        assert!(cuda.contains("blockDim.x"), "Should contain blockDim.x");
        assert!(cuda.contains("threadIdx.x"), "Should contain threadIdx.x");
        assert!(cuda.contains("return"), "Should have early return");

        println!("Generated global kernel:\n{}", cuda);
    }

    #[test]
    fn test_stencil_kernel_transpile() {
        let func: syn::ItemFn = parse_quote! {
            fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
                let curr = p[pos.idx()];
                let prev = p_prev[pos.idx()];
                let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p) - 4.0 * curr;
                p_prev[pos.idx()] = (2.0 * curr - prev + c2 * lap);
            }
        };

        let config = StencilConfig {
            id: "fdtd".to_string(),
            grid: Grid::Grid2D,
            tile_size: (16, 16),
            halo: 1,
        };

        let result = transpile_stencil_kernel(&func, &config);
        assert!(
            result.is_ok(),
            "Should transpile stencil kernel: {:?}",
            result
        );

        let cuda = result.unwrap();
        assert!(cuda.contains("__global__"), "Should be a CUDA kernel");
        assert!(cuda.contains("threadIdx"), "Should use thread indices");
    }

    #[test]
    fn test_ring_kernel_transpile() {
        // A simple handler that doubles the input value
        let handler: syn::ItemFn = parse_quote! {
            fn process(value: f32) -> f32 {
                let result = value * 2.0;
                result
            }
        };

        let config = RingKernelConfig::new("processor")
            .with_block_size(128)
            .with_queue_capacity(1024)
            .with_hlc(true);

        let result = transpile_ring_kernel(&handler, &config);
        assert!(result.is_ok(), "Should transpile ring kernel: {:?}", result);

        let cuda = result.unwrap();

        // Check struct definitions
        assert!(
            cuda.contains("struct __align__(128) ControlBlock"),
            "Should have ControlBlock struct"
        );
        assert!(cuda.contains("is_active"), "Should have is_active field");
        assert!(
            cuda.contains("should_terminate"),
            "Should have should_terminate field"
        );
        assert!(
            cuda.contains("messages_processed"),
            "Should have messages_processed field"
        );

        // Check kernel signature
        assert!(
            cuda.contains("extern \"C\" __global__ void ring_kernel_processor"),
            "Should have correct kernel name"
        );
        assert!(
            cuda.contains("ControlBlock* __restrict__ control"),
            "Should have control block param"
        );
        assert!(
            cuda.contains("input_buffer"),
            "Should have input buffer param"
        );
        assert!(
            cuda.contains("output_buffer"),
            "Should have output buffer param"
        );

        // Check preamble
        assert!(
            cuda.contains("int tid = threadIdx.x + blockIdx.x * blockDim.x"),
            "Should have thread id calculation"
        );
        assert!(
            cuda.contains("MSG_SIZE"),
            "Should have message size constant"
        );
        assert!(cuda.contains("hlc_physical"), "Should have HLC variables");
        assert!(
            cuda.contains("hlc_logical"),
            "Should have HLC logical counter"
        );

        // Check message loop
        assert!(cuda.contains("while (true)"), "Should have persistent loop");
        assert!(
            cuda.contains("atomicAdd(&control->should_terminate, 0)"),
            "Should check termination"
        );
        assert!(
            cuda.contains("atomicAdd(&control->is_active, 0)"),
            "Should check is_active"
        );

        // Check handler code is embedded
        assert!(
            cuda.contains("// === USER HANDLER CODE ==="),
            "Should have handler marker"
        );
        assert!(cuda.contains("value * 2.0"), "Should contain handler logic");
        assert!(
            cuda.contains("// === END HANDLER CODE ==="),
            "Should have end marker"
        );

        // Check epilogue
        assert!(
            cuda.contains("atomicExch(&control->has_terminated, 1)"),
            "Should mark terminated"
        );

        println!("Generated ring kernel:\n{}", cuda);
    }

    #[test]
    fn test_ring_kernel_with_k2k() {
        let handler: syn::ItemFn = parse_quote! {
            fn forward(msg: f32) -> f32 {
                msg
            }
        };

        let config = RingKernelConfig::new("forwarder")
            .with_block_size(64)
            .with_k2k(true)
            .with_hlc(true);

        let result = transpile_ring_kernel(&handler, &config);
        assert!(result.is_ok(), "Should transpile K2K kernel: {:?}", result);

        let cuda = result.unwrap();

        // Check K2K structs
        assert!(
            cuda.contains("K2KRoutingTable"),
            "Should have K2K routing table"
        );
        assert!(cuda.contains("K2KRoute"), "Should have K2K route struct");

        // Check K2K params in signature
        assert!(cuda.contains("k2k_routes"), "Should have k2k_routes param");
        assert!(cuda.contains("k2k_inbox"), "Should have k2k_inbox param");
        assert!(cuda.contains("k2k_outbox"), "Should have k2k_outbox param");

        println!("Generated K2K ring kernel:\n{}", cuda);
    }

    #[test]
    fn test_ring_kernel_config_defaults() {
        let config = RingKernelConfig::default();
        assert_eq!(config.block_size, 128);
        assert_eq!(config.queue_capacity, 1024);
        assert!(config.enable_hlc);
        assert!(!config.enable_k2k);
    }

    #[test]
    fn test_ring_kernel_intrinsic_availability() {
        // Test that ring kernel intrinsics are properly exported
        assert!(RingKernelIntrinsic::from_name("is_active").is_some());
        assert!(RingKernelIntrinsic::from_name("should_terminate").is_some());
        assert!(RingKernelIntrinsic::from_name("hlc_tick").is_some());
        assert!(RingKernelIntrinsic::from_name("enqueue_response").is_some());
        assert!(RingKernelIntrinsic::from_name("k2k_send").is_some());
        assert!(RingKernelIntrinsic::from_name("nanosleep").is_some());
    }

    #[test]
    fn test_handler_signature_parsing() {
        let func: syn::ItemFn = parse_quote! {
            fn handle(ctx: &RingContext, msg: &MyMessage) -> MyResponse {
                MyResponse { value: msg.value * 2.0 }
            }
        };

        let mapper = TypeMapper::new();
        let sig = HandlerSignature::parse(&func, &mapper).unwrap();

        assert_eq!(sig.name, "handle");
        assert!(sig.has_context);
        assert!(sig.message_param.is_some());
        assert!(sig.has_response());
    }

    #[test]
    fn test_handler_with_context_methods() {
        // Handler that uses context methods
        let handler: syn::ItemFn = parse_quote! {
            fn process(ctx: &RingContext, value: f32) -> f32 {
                let tid = ctx.thread_id();
                let result = value * 2.0;
                ctx.sync_threads();
                result
            }
        };

        let config = RingKernelConfig::new("with_context")
            .with_block_size(128)
            .with_hlc(true);

        let result = transpile_ring_kernel(&handler, &config);
        assert!(
            result.is_ok(),
            "Should transpile handler with context: {:?}",
            result
        );

        let cuda = result.unwrap();

        // Context methods should be inlined
        assert!(
            cuda.contains("threadIdx.x"),
            "ctx.thread_id() should become threadIdx.x"
        );
        assert!(
            cuda.contains("__syncthreads()"),
            "ctx.sync_threads() should become __syncthreads()"
        );

        println!("Generated handler with context:\n{}", cuda);
    }

    #[test]
    fn test_handler_with_message_param() {
        let handler: syn::ItemFn = parse_quote! {
            fn process_msg(msg: &Message, scale: f32) -> f32 {
                msg.value * scale
            }
        };

        let config = RingKernelConfig::new("msg_handler");
        let result = transpile_ring_kernel(&handler, &config);
        assert!(
            result.is_ok(),
            "Should transpile handler with message: {:?}",
            result
        );

        let cuda = result.unwrap();
        // Should have message deserialization comment
        assert!(cuda.contains("Message"), "Should reference Message type");

        println!("Generated message handler:\n{}", cuda);
    }

    #[test]
    fn test_context_method_mappings() {
        // Test that context methods are properly mapped
        assert!(ContextMethod::from_name("thread_id").is_some());
        assert!(ContextMethod::from_name("sync_threads").is_some());
        assert!(ContextMethod::from_name("global_thread_id").is_some());
        assert!(ContextMethod::from_name("atomic_add").is_some());
        assert!(ContextMethod::from_name("lane_id").is_some());
        assert!(ContextMethod::from_name("warp_id").is_some());

        // Test CUDA output
        assert_eq!(ContextMethod::ThreadId.to_cuda(&[]), "threadIdx.x");
        assert_eq!(ContextMethod::SyncThreads.to_cuda(&[]), "__syncthreads()");
        assert_eq!(
            ContextMethod::GlobalThreadId.to_cuda(&[]),
            "(blockIdx.x * blockDim.x + threadIdx.x)"
        );
    }

    #[test]
    fn test_message_type_registration() {
        let mut registry = MessageTypeRegistry::new();

        registry.register_message(MessageTypeInfo {
            name: "InputMsg".to_string(),
            size: 16,
            fields: vec![
                ("id".to_string(), "unsigned long long".to_string()),
                ("value".to_string(), "float".to_string()),
            ],
        });

        registry.register_response(MessageTypeInfo {
            name: "OutputMsg".to_string(),
            size: 8,
            fields: vec![("result".to_string(), "float".to_string())],
        });

        let structs = registry.generate_structs();
        assert!(structs.contains("struct InputMsg"));
        assert!(structs.contains("struct OutputMsg"));
        assert!(structs.contains("unsigned long long id"));
        assert!(structs.contains("float result"));
    }

    #[test]
    fn test_full_handler_integration() {
        // Complete handler with all features
        let handler: syn::ItemFn = parse_quote! {
            fn full_handler(ctx: &RingContext, msg: &Request) -> Response {
                let tid = ctx.global_thread_id();
                ctx.sync_threads();
                let result = msg.value * 2.0;
                Response { value: result, id: tid as u64 }
            }
        };

        let config = RingKernelConfig::new("full")
            .with_block_size(256)
            .with_queue_capacity(2048)
            .with_hlc(true)
            .with_k2k(false);

        let result = transpile_ring_kernel(&handler, &config);
        assert!(
            result.is_ok(),
            "Should transpile full handler: {:?}",
            result
        );

        let cuda = result.unwrap();

        // Check all expected components
        assert!(cuda.contains("ring_kernel_full"), "Kernel name");
        assert!(cuda.contains("ControlBlock"), "ControlBlock struct");
        assert!(cuda.contains("while (true)"), "Persistent loop");
        assert!(cuda.contains("threadIdx.x"), "Thread index");
        assert!(cuda.contains("__syncthreads()"), "Sync threads");
        assert!(
            cuda.contains("blockIdx.x * blockDim.x + threadIdx.x"),
            "Global thread ID"
        );
        assert!(cuda.contains("has_terminated"), "Termination marking");

        println!("Full handler integration:\n{}", cuda);
    }

    #[test]
    fn test_k2k_handler_integration() {
        // Handler that uses K2K messaging
        let handler: syn::ItemFn = parse_quote! {
            fn k2k_handler(ctx: &RingContext, msg: &InputMsg) -> OutputMsg {
                let tid = ctx.global_thread_id();

                // Process incoming message
                let result = msg.value * 2.0;

                // Build response
                OutputMsg { result: result, source_id: tid as u64 }
            }
        };

        let config = RingKernelConfig::new("k2k_processor")
            .with_block_size(128)
            .with_queue_capacity(1024)
            .with_hlc(true)
            .with_k2k(true);

        let result = transpile_ring_kernel(&handler, &config);
        assert!(result.is_ok(), "Should transpile K2K handler: {:?}", result);

        let cuda = result.unwrap();

        // Check K2K-specific components
        assert!(
            cuda.contains("K2KRoutingTable"),
            "Should have K2KRoutingTable"
        );
        assert!(cuda.contains("K2KRoute"), "Should have K2KRoute struct");
        assert!(
            cuda.contains("K2KInboxHeader"),
            "Should have K2KInboxHeader"
        );
        assert!(cuda.contains("k2k_routes"), "Should have k2k_routes param");
        assert!(cuda.contains("k2k_inbox"), "Should have k2k_inbox param");
        assert!(cuda.contains("k2k_outbox"), "Should have k2k_outbox param");
        assert!(cuda.contains("k2k_send"), "Should have k2k_send function");
        assert!(
            cuda.contains("k2k_try_recv"),
            "Should have k2k_try_recv function"
        );
        assert!(cuda.contains("k2k_peek"), "Should have k2k_peek function");
        assert!(
            cuda.contains("k2k_pending_count"),
            "Should have k2k_pending_count function"
        );

        println!("K2K handler integration:\n{}", cuda);
    }

    #[test]
    fn test_all_kernel_types_comparison() {
        // Stencil kernel
        let stencil_func: syn::ItemFn = parse_quote! {
            fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
                let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p) - 4.0 * p[pos.idx()];
                p_prev[pos.idx()] = 2.0 * p[pos.idx()] - p_prev[pos.idx()] + c2 * lap;
            }
        };

        let stencil_config = StencilConfig::new("fdtd")
            .with_tile_size(16, 16)
            .with_halo(1);

        let stencil_cuda = transpile_stencil_kernel(&stencil_func, &stencil_config).unwrap();
        assert!(
            !stencil_cuda.contains("GridPos"),
            "Stencil should remove GridPos"
        );
        assert!(
            stencil_cuda.contains("buffer_width"),
            "Stencil should have buffer_width"
        );

        // Global kernel
        let global_func: syn::ItemFn = parse_quote! {
            fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
                let idx = block_idx_x() * block_dim_x() + thread_idx_x();
                if idx >= n { return; }
                y[idx as usize] = a * x[idx as usize] + y[idx as usize];
            }
        };

        let global_cuda = transpile_global_kernel(&global_func).unwrap();
        assert!(global_cuda.contains("__global__"), "Global kernel marker");
        assert!(global_cuda.contains("blockIdx.x"), "CUDA block index");

        // Ring kernel
        let ring_func: syn::ItemFn = parse_quote! {
            fn process(msg: f32) -> f32 {
                msg * 2.0
            }
        };

        let ring_config = RingKernelConfig::new("process")
            .with_block_size(128)
            .with_hlc(true);

        let ring_cuda = transpile_ring_kernel(&ring_func, &ring_config).unwrap();
        assert!(
            ring_cuda.contains("ControlBlock"),
            "Ring kernel ControlBlock"
        );
        assert!(ring_cuda.contains("while (true)"), "Persistent loop");
        assert!(ring_cuda.contains("has_terminated"), "Termination");

        println!("=== Stencil Kernel ===\n{}\n", stencil_cuda);
        println!("=== Global Kernel ===\n{}\n", global_cuda);
        println!("=== Ring Kernel ===\n{}\n", ring_cuda);
    }
}
