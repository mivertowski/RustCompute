//! CPU Backend for RingKernel
//!
//! This crate provides a CPU-based implementation of the RingKernel runtime,
//! primarily used for testing and as a fallback when no GPU is available.
//!
//! # Features
//!
//! - Full implementation of the RingKernelRuntime trait
//! - Simulates GPU execution using async tasks
//! - Supports all kernel lifecycle operations
//! - Useful for unit testing and development
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cpu::CpuRuntime;
//! use ringkernel_core::runtime::{RuntimeBuilder, Backend};
//!
//! #[tokio::main]
//! async fn main() {
//!     let runtime = CpuRuntime::new().await.unwrap();
//!     let kernel = runtime.launch("my_kernel", Default::default()).await.unwrap();
//!     kernel.activate().await.unwrap();
//! }
//! ```

#![warn(missing_docs)]

mod kernel;
mod memory;
pub mod mock;
mod runtime;
pub mod simd;

pub use kernel::CpuKernel;
pub use memory::CpuBuffer;
pub use mock::{MockAtomics, MockGpu, MockKernelConfig, MockSharedMemory, MockThread, MockWarp};
pub use runtime::CpuRuntime;
pub use simd::SimdOps;

/// Prelude for convenient imports.
pub mod prelude {
    pub use crate::mock::{
        MockAtomics, MockGpu, MockKernelConfig, MockSharedMemory, MockThread, MockWarp,
    };
    pub use crate::simd::SimdOps;
    pub use crate::CpuKernel;
    pub use crate::CpuRuntime;
}
