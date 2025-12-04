//! CUDA kernel code generation and execution.
//!
//! Transpiles Rust DSL kernels to CUDA and manages GPU execution.

mod codegen;
mod executor;
mod types;

pub use codegen::*;
pub use executor::*;
pub use types::*;
