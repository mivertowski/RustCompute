//! CUDA Backend for RingKernel
//!
//! This crate provides NVIDIA CUDA GPU support for RingKernel using cudarc.
//!
//! # Features
//!
//! - Persistent kernel execution (cooperative groups)
//! - Lock-free message queues in GPU global memory
//! - PTX compilation via NVRTC
//! - Multi-GPU support
//!
//! # Requirements
//!
//! - NVIDIA GPU with Compute Capability 7.0+
//! - CUDA Toolkit 11.0+
//! - Native Linux (persistent kernels) or WSL2 (event-driven fallback)
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cuda::CudaRuntime;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let runtime = CudaRuntime::new().await?;
//!     let kernel = runtime.launch("vector_add", Default::default()).await?;
//!     kernel.activate().await?;
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]

#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
mod kernel;
#[cfg(feature = "cuda")]
mod memory;
#[cfg(feature = "cuda")]
mod runtime;
#[cfg(feature = "cuda")]
mod stencil;

#[cfg(feature = "cuda")]
pub use device::CudaDevice;
#[cfg(feature = "cuda")]
pub use kernel::CudaKernel;
#[cfg(feature = "cuda")]
pub use memory::{CudaBuffer, CudaControlBlock, CudaMemoryPool, CudaMessageQueue};
#[cfg(feature = "cuda")]
pub use runtime::CudaRuntime;
#[cfg(feature = "cuda")]
pub use stencil::{CompiledStencilKernel, LaunchConfig, StencilKernelLoader};

/// Re-export memory module for advanced usage.
#[cfg(feature = "cuda")]
pub mod memory_exports {
    pub use super::memory::{CudaBuffer, CudaControlBlock, CudaMemoryPool, CudaMessageQueue};
}

// Placeholder implementations when CUDA is not available
#[cfg(not(feature = "cuda"))]
mod stub {
    use async_trait::async_trait;
    use ringkernel_core::error::{Result, RingKernelError};
    use ringkernel_core::runtime::{
        Backend, KernelHandle, KernelId, LaunchOptions, RingKernelRuntime, RuntimeMetrics,
    };

    /// Stub CUDA runtime when CUDA feature is disabled.
    pub struct CudaRuntime;

    impl CudaRuntime {
        /// Create fails when CUDA is not available.
        pub async fn new() -> Result<Self> {
            Err(RingKernelError::BackendUnavailable(
                "CUDA feature not enabled".to_string(),
            ))
        }
    }

    #[async_trait]
    impl RingKernelRuntime for CudaRuntime {
        fn backend(&self) -> Backend {
            Backend::Cuda
        }

        fn is_backend_available(&self, _backend: Backend) -> bool {
            false
        }

        async fn launch(&self, _kernel_id: &str, _options: LaunchOptions) -> Result<KernelHandle> {
            Err(RingKernelError::BackendUnavailable("CUDA".to_string()))
        }

        fn get_kernel(&self, _kernel_id: &KernelId) -> Option<KernelHandle> {
            None
        }

        fn list_kernels(&self) -> Vec<KernelId> {
            vec![]
        }

        fn metrics(&self) -> RuntimeMetrics {
            RuntimeMetrics::default()
        }

        async fn shutdown(&self) -> Result<()> {
            Ok(())
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use stub::CudaRuntime;

/// Check if CUDA is available at runtime.
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Check for CUDA devices
        cudarc::driver::CudaDevice::count()
            .map(|c| c > 0)
            .unwrap_or(false)
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get CUDA device count.
pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        cudarc::driver::CudaDevice::count().unwrap_or(0) as usize
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// PTX kernel source template for persistent ring kernel.
///
/// This is a minimal kernel that immediately marks itself as terminated.
/// Uses PTX 8.0 / sm_89 for Ada Lovelace GPU compatibility (RTX 40xx series).
pub const RING_KERNEL_PTX_TEMPLATE: &str = r#"
.version 8.0
.target sm_89
.address_size 64

.visible .entry ring_kernel_main(
    .param .u64 control_block_ptr,
    .param .u64 input_queue_ptr,
    .param .u64 output_queue_ptr,
    .param .u64 shared_state_ptr
) {
    .reg .u64 %cb_ptr;
    .reg .u32 %one;

    // Load control block pointer
    ld.param.u64 %cb_ptr, [control_block_ptr];

    // Mark as terminated immediately (offset 8)
    mov.u32 %one, 1;
    st.global.u32 [%cb_ptr + 8], %one;

    ret;
}
"#;
