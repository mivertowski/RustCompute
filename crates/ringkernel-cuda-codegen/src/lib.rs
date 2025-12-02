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

mod intrinsics;
mod stencil;
mod transpiler;
mod types;
mod validation;

pub use intrinsics::{GpuIntrinsic, IntrinsicRegistry};
pub use stencil::{Grid, GridPos, StencilConfig};
pub use transpiler::{transpile_function, CudaTranspiler};
pub use types::{CudaType, TypeMapper};
pub use validation::{validate_function, ValidationError};

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
pub fn transpile_stencil_kernel(
    func: &syn::ItemFn,
    config: &StencilConfig,
) -> Result<String> {
    // Validate DSL constraints
    validate_function(func)?;

    // Create transpiler with stencil config
    let mut transpiler = CudaTranspiler::new(config.clone());

    // Generate CUDA code
    transpiler.transpile_stencil(func)
}

/// Transpile a Rust function to CUDA without stencil-specific patterns.
///
/// This is a lower-level API for non-stencil kernels that still want
/// Rust-to-CUDA translation.
pub fn transpile_generic_kernel(func: &syn::ItemFn) -> Result<String> {
    validate_function(func)?;
    transpile_function(func)
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

        let result = transpile_generic_kernel(&func);
        assert!(result.is_ok(), "Should transpile simple function: {:?}", result);

        let cuda = result.unwrap();
        assert!(cuda.contains("float"), "Should contain CUDA float type");
        assert!(cuda.contains("a + b"), "Should contain the expression");
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
        assert!(result.is_ok(), "Should transpile stencil kernel: {:?}", result);

        let cuda = result.unwrap();
        assert!(cuda.contains("__global__"), "Should be a CUDA kernel");
        assert!(cuda.contains("threadIdx"), "Should use thread indices");
    }
}
