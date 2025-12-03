//! WGSL code generation from Rust DSL for RingKernel.
//!
//! This crate provides transpilation from a restricted Rust DSL to WGSL (WebGPU Shading Language),
//! enabling developers to write GPU kernels in Rust that target WebGPU-compatible devices.
//!
//! # Overview
//!
//! The transpiler supports the same DSL as `ringkernel-cuda-codegen`:
//!
//! - Primitive types: `f32`, `i32`, `u32`, `bool` (with 64-bit emulation)
//! - Array slices: `&[T]`, `&mut [T]` → storage buffers
//! - Arithmetic and comparison operators
//! - Let bindings and if/else expressions
//! - For/while/loop constructs
//! - Stencil intrinsics via `GridPos` context
//! - Ring kernel support with host-driven persistence emulation
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_wgpu_codegen::{transpile_stencil_kernel, StencilConfig};
//! use syn::parse_quote;
//!
//! let func: syn::ItemFn = parse_quote! {
//!     fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
//!         let curr = p[pos.idx()];
//!         let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p) - 4.0 * curr;
//!         p_prev[pos.idx()] = 2.0 * curr - p_prev[pos.idx()] + c2 * lap;
//!     }
//! };
//!
//! let config = StencilConfig::new("fdtd")
//!     .with_tile_size(16, 16)
//!     .with_halo(1);
//!
//! let wgsl_code = transpile_stencil_kernel(&func, &config)?;
//! ```
//!
//! # WGSL Limitations
//!
//! Compared to CUDA, WGSL has some limitations that are handled with workarounds:
//!
//! - **No 64-bit atomics**: Emulated using lo/hi u32 pairs
//! - **No f64**: Downcast to f32 with a warning
//! - **No persistent kernels**: Emulated with host-driven dispatch loops
//! - **No warp operations**: Mapped to subgroup operations where available
//! - **No kernel-to-kernel messaging**: K2K is not supported in WGPU

pub mod bindings;
pub mod dsl;
pub mod handler;
pub mod intrinsics;
pub mod loops;
pub mod ring_kernel;
pub mod shared;
pub mod stencil;
pub mod transpiler;
pub mod types;
pub mod u64_workarounds;
pub mod validation;

pub use bindings::{AccessMode, BindingLayout, generate_bindings};
pub use dsl::*;
pub use handler::{
    HandlerCodegenConfig, HandlerParam, HandlerParamKind, HandlerReturnType, HandlerSignature,
    WgslContextMethod,
};
pub use intrinsics::{IntrinsicRegistry, WgslIntrinsic};
pub use loops::{LoopPattern, RangeInfo};
pub use ring_kernel::RingKernelConfig;
pub use shared::{SharedArray, SharedMemoryConfig, SharedMemoryDecl, SharedTile};
pub use stencil::{Grid, GridPos, StencilConfig, StencilLaunchConfig};
pub use transpiler::{transpile_function, WgslTranspiler};
pub use types::{AddressSpace, TypeMapper, WgslType};
pub use u64_workarounds::U64Helpers;
pub use validation::{ValidationError, ValidationMode};

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

    /// WGSL-specific limitation.
    #[error("WGSL limitation: {0}")]
    WgslLimitation(String),
}

/// Result type for transpilation operations.
pub type Result<T> = std::result::Result<T, TranspileError>;

/// Transpile a Rust stencil kernel function to WGSL code.
///
/// This is the main entry point for stencil code generation. It takes a parsed
/// Rust function and stencil configuration, validates the DSL constraints,
/// and generates equivalent WGSL code.
///
/// # Arguments
///
/// * `func` - The parsed Rust function (from syn)
/// * `config` - Stencil kernel configuration
///
/// # Returns
///
/// The generated WGSL source code as a string.
///
/// # Example
///
/// ```ignore
/// use ringkernel_wgpu_codegen::{transpile_stencil_kernel, StencilConfig};
/// use syn::parse_quote;
///
/// let func: syn::ItemFn = parse_quote! {
///     fn heat(temp: &[f32], temp_new: &mut [f32], alpha: f32, pos: GridPos) {
///         let t = temp[pos.idx()];
///         let neighbors = pos.north(temp) + pos.south(temp) + pos.east(temp) + pos.west(temp);
///         temp_new[pos.idx()] = t + alpha * (neighbors - 4.0 * t);
///     }
/// };
///
/// let config = StencilConfig::new("heat").with_tile_size(16, 16).with_halo(1);
/// let wgsl = transpile_stencil_kernel(&func, &config)?;
/// ```
pub fn transpile_stencil_kernel(func: &syn::ItemFn, config: &StencilConfig) -> Result<String> {
    // Validate DSL constraints
    validation::validate_function(func)?;

    // Create transpiler with stencil config
    let mut transpiler = WgslTranspiler::new_stencil(config.clone());

    // Generate WGSL code
    transpiler.transpile_stencil(func)
}

/// Transpile a Rust function to a WGSL helper function.
///
/// This generates a callable function (not a compute entry point) from Rust code.
pub fn transpile_device_function(func: &syn::ItemFn) -> Result<String> {
    validation::validate_function(func)?;
    transpile_function(func)
}

/// Transpile a Rust function to a WGSL `@compute` kernel.
///
/// This generates an compute shader entry point without stencil-specific patterns.
/// Use DSL functions like `thread_idx_x()`, `block_idx_x()` to access WGSL indices.
///
/// # Example
///
/// ```ignore
/// use ringkernel_wgpu_codegen::transpile_global_kernel;
/// use syn::parse_quote;
///
/// let func: syn::ItemFn = parse_quote! {
///     fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
///         let idx = block_idx_x() * block_dim_x() + thread_idx_x();
///         if idx >= n { return; }
///         y[idx as usize] = a * x[idx as usize] + y[idx as usize];
///     }
/// };
///
/// let wgsl = transpile_global_kernel(&func)?;
/// // Generates: @compute @workgroup_size(256) fn saxpy(...) { ... }
/// ```
pub fn transpile_global_kernel(func: &syn::ItemFn) -> Result<String> {
    validation::validate_function(func)?;
    let mut transpiler = WgslTranspiler::new_generic();
    transpiler.transpile_global_kernel(func)
}

/// Transpile a Rust handler function to a WGSL ring kernel.
///
/// Ring kernels in WGPU are emulated using host-driven dispatch loops since
/// WebGPU does not support true persistent kernels. The handler function
/// processes a batch of messages per dispatch.
///
/// # Example
///
/// ```ignore
/// use ringkernel_wgpu_codegen::{transpile_ring_kernel, RingKernelConfig};
/// use syn::parse_quote;
///
/// let handler: syn::ItemFn = parse_quote! {
///     fn process(value: f32) -> f32 {
///         value * 2.0
///     }
/// };
///
/// let config = RingKernelConfig::new("processor")
///     .with_workgroup_size(256)
///     .with_hlc(true);
///
/// let wgsl = transpile_ring_kernel(&handler, &config)?;
/// ```
///
/// # WGSL Limitations
///
/// - **No K2K**: Kernel-to-kernel messaging is not supported
/// - **No persistent loop**: Host must re-dispatch until termination
/// - **No 64-bit atomics**: Counters use lo/hi u32 pair emulation
pub fn transpile_ring_kernel(
    handler: &syn::ItemFn,
    config: &RingKernelConfig,
) -> Result<String> {
    // K2K is not supported in WGPU
    if config.enable_k2k {
        return Err(TranspileError::WgslLimitation(
            "Kernel-to-kernel (K2K) messaging is not supported in WGPU. \
             WebGPU does not allow GPU-direct communication between compute dispatches. \
             Use host-mediated messaging instead.".to_string()
        ));
    }

    // Validate handler with generic mode (loops allowed but not required)
    validation::validate_function_with_mode(handler, ValidationMode::Generic)?;

    // Create transpiler in generic mode for the handler
    let mut transpiler = WgslTranspiler::new_ring_kernel(config.clone());

    // Transpile the handler into a ring kernel wrapper
    transpiler.transpile_ring_kernel(handler, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_transpile_error_display() {
        let err = TranspileError::Parse("unexpected token".to_string());
        assert!(err.to_string().contains("Parse error"));

        let err = TranspileError::WgslLimitation("no 64-bit atomics".to_string());
        assert!(err.to_string().contains("WGSL limitation"));
    }

    #[test]
    fn test_k2k_rejected() {
        let handler: syn::ItemFn = parse_quote! {
            fn forward(msg: f32) -> f32 {
                msg
            }
        };

        let config = RingKernelConfig::new("forwarder")
            .with_workgroup_size(64)
            .with_k2k(true);

        let result = transpile_ring_kernel(&handler, &config);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TranspileError::WgslLimitation(_)));
    }
}
