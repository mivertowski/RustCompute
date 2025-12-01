//! # RingKernel
//!
//! GPU-native persistent actor model framework for Rust.
//!
//! RingKernel is a Rust port of DotCompute's Ring Kernel system, enabling
//! GPU-accelerated actor systems with persistent kernels, lock-free message
//! passing, and hybrid logical clocks for causal ordering.
//!
//! ## Features
//!
//! - **Persistent GPU-resident state** across kernel invocations
//! - **Lock-free message passing** between kernels (K2K messaging)
//! - **Hybrid Logical Clocks (HLC)** for temporal ordering
//! - **Multiple GPU backends**: CUDA, Metal, WebGPU
//! - **Type-safe serialization** via rkyv/zerocopy
//!
//! ## Quick Start
//!
//! ```ignore
//! use ringkernel::prelude::*;
//!
//! #[derive(RingMessage)]
//! struct AddRequest {
//!     #[message(id)]
//!     id: MessageId,
//!     a: f32,
//!     b: f32,
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create runtime with auto-detected backend
//!     let runtime = RingKernel::builder()
//!         .backend(Backend::Auto)
//!         .build()
//!         .await?;
//!
//!     // Launch a kernel
//!     let kernel = runtime.launch("adder", LaunchOptions::default()).await?;
//!     kernel.activate().await?;
//!
//!     // Send a message
//!     kernel.send(AddRequest {
//!         id: MessageId::generate(),
//!         a: 1.0,
//!         b: 2.0,
//!     }).await?;
//!
//!     // Receive response
//!     let response = kernel.receive().await?;
//!     println!("Result: {:?}", response);
//!
//!     kernel.terminate().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Backends
//!
//! RingKernel supports multiple GPU backends:
//!
//! - **CPU** - Testing and fallback (always available)
//! - **CUDA** - NVIDIA GPUs (requires `cuda` feature)
//! - **Metal** - Apple GPUs (requires `metal` feature, macOS only)
//! - **WebGPU** - Cross-platform via wgpu (requires `wgpu` feature)
//!
//! Enable backends via Cargo features:
//!
//! ```toml
//! [dependencies]
//! ringkernel = { version = "0.1", features = ["cuda", "wgpu"] }
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    Host (CPU)                           │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
//! │  │ Application │──│   Runtime   │──│  Message Bridge │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────┘  │
//! └──────────────────────────┬──────────────────────────────┘
//!                            │ DMA Transfers
//! ┌──────────────────────────┴──────────────────────────────┐
//! │                   Device (GPU)                          │
//! │  ┌───────────┐  ┌───────────────┐  ┌───────────────┐    │
//! │  │ Control   │  │ Input Queue   │  │ Output Queue  │    │
//! │  │ Block     │  │ (lock-free)   │  │ (lock-free)   │    │
//! │  └───────────┘  └───────────────┘  └───────────────┘    │
//! │  ┌─────────────────────────────────────────────────┐    │
//! │  │         Persistent Kernel (your code)          │    │
//! │  └─────────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────────┘
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(hidden_glob_reexports)]

// Re-export core types
pub use ringkernel_core::*;

// Re-export derive macros
pub use ringkernel_derive::*;

// Re-export CPU backend (always available)
pub use ringkernel_cpu::CpuRuntime;

// Conditional re-exports for GPU backends
#[cfg(feature = "cuda")]
pub use ringkernel_cuda::CudaRuntime;

#[cfg(feature = "wgpu")]
pub use ringkernel_wgpu::WgpuRuntime;

#[cfg(feature = "metal")]
pub use ringkernel_metal::MetalRuntime;

// Re-export codegen
pub use ringkernel_codegen as codegen;

// Use types from the re-exported ringkernel_core
use ringkernel_core::error::Result;
use ringkernel_core::error::RingKernelError;
use ringkernel_core::runtime::Backend;
use ringkernel_core::runtime::KernelHandle;
use ringkernel_core::runtime::KernelId;
use ringkernel_core::runtime::LaunchOptions;
use ringkernel_core::runtime::RingKernelRuntime;
use ringkernel_core::runtime::RuntimeBuilder;
use ringkernel_core::runtime::RuntimeMetrics;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::RingKernel;
    pub use ringkernel_core::prelude::*;
    pub use ringkernel_derive::*;
}

/// Private module for kernel registration (used by proc macros).
#[doc(hidden)]
pub mod __private {
    use ringkernel_core::types::KernelMode;

    /// Kernel registration information.
    pub struct KernelRegistration {
        /// Kernel identifier.
        pub id: &'static str,
        /// Execution mode.
        pub mode: KernelMode,
        /// Grid size.
        pub grid_size: u32,
        /// Block size.
        pub block_size: u32,
    }

    // Use inventory for kernel registration
    inventory::collect!(KernelRegistration);
}

/// Main RingKernel runtime facade.
///
/// This struct provides the primary API for creating and managing
/// GPU-native actor systems.
pub struct RingKernel {
    /// Inner runtime implementation.
    inner: Box<dyn RingKernelRuntime>,
}

impl RingKernel {
    /// Create a new runtime builder.
    pub fn builder() -> RingKernelBuilder {
        RingKernelBuilder::new()
    }

    /// Create a new runtime with default settings.
    pub async fn new() -> Result<Self> {
        Self::builder().build().await
    }

    /// Create with a specific backend.
    pub async fn with_backend(backend: Backend) -> Result<Self> {
        Self::builder().backend(backend).build().await
    }

    /// Get the active backend.
    pub fn backend(&self) -> Backend {
        self.inner.backend()
    }

    /// Check if a backend is available.
    pub fn is_backend_available(&self, backend: Backend) -> bool {
        self.inner.is_backend_available(backend)
    }

    /// Launch a kernel.
    pub async fn launch(&self, kernel_id: &str, options: LaunchOptions) -> Result<KernelHandle> {
        self.inner.launch(kernel_id, options).await
    }

    /// Get a handle to an existing kernel.
    pub fn get_kernel(&self, kernel_id: &KernelId) -> Option<KernelHandle> {
        self.inner.get_kernel(kernel_id)
    }

    /// List all kernel IDs.
    pub fn list_kernels(&self) -> Vec<KernelId> {
        self.inner.list_kernels()
    }

    /// Get runtime metrics.
    pub fn metrics(&self) -> RuntimeMetrics {
        self.inner.metrics()
    }

    /// Shutdown the runtime.
    pub async fn shutdown(self) -> Result<()> {
        self.inner.shutdown().await
    }
}

/// Builder for RingKernel runtime.
pub struct RingKernelBuilder {
    inner: RuntimeBuilder,
}

impl RingKernelBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            inner: RuntimeBuilder::new(),
        }
    }

    /// Set the backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.inner = self.inner.backend(backend);
        self
    }

    /// Set the device index.
    pub fn device(mut self, index: usize) -> Self {
        self.inner = self.inner.device(index);
        self
    }

    /// Enable debug mode.
    pub fn debug(mut self, enable: bool) -> Self {
        self.inner = self.inner.debug(enable);
        self
    }

    /// Enable profiling.
    pub fn profiling(mut self, enable: bool) -> Self {
        self.inner = self.inner.profiling(enable);
        self
    }

    /// Build the runtime.
    pub async fn build(self) -> Result<RingKernel> {
        let backend = self.inner.backend;

        let inner: Box<dyn RingKernelRuntime> = match backend {
            Backend::Auto => self.build_auto().await?,
            Backend::Cpu => Box::new(CpuRuntime::new().await?),
            #[cfg(feature = "cuda")]
            Backend::Cuda => Box::new(ringkernel_cuda::CudaRuntime::new().await?),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => {
                return Err(RingKernelError::BackendUnavailable(
                    "CUDA feature not enabled".to_string(),
                ))
            }
            #[cfg(feature = "metal")]
            Backend::Metal => Box::new(ringkernel_metal::MetalRuntime::new().await?),
            #[cfg(not(feature = "metal"))]
            Backend::Metal => {
                return Err(RingKernelError::BackendUnavailable(
                    "Metal feature not enabled".to_string(),
                ))
            }
            #[cfg(feature = "wgpu")]
            Backend::Wgpu => Box::new(ringkernel_wgpu::WgpuRuntime::new().await?),
            #[cfg(not(feature = "wgpu"))]
            Backend::Wgpu => {
                return Err(RingKernelError::BackendUnavailable(
                    "WebGPU feature not enabled".to_string(),
                ))
            }
        };

        Ok(RingKernel { inner })
    }

    /// Auto-select the best available backend.
    async fn build_auto(self) -> Result<Box<dyn RingKernelRuntime>> {
        // Try CUDA first (best performance on NVIDIA)
        #[cfg(feature = "cuda")]
        if ringkernel_cuda::is_cuda_available() {
            tracing::info!("Auto-selected CUDA backend");
            return Ok(Box::new(ringkernel_cuda::CudaRuntime::new().await?));
        }

        // Try Metal on macOS
        #[cfg(feature = "metal")]
        if ringkernel_metal::is_metal_available() {
            tracing::info!("Auto-selected Metal backend");
            return Ok(Box::new(ringkernel_metal::MetalRuntime::new().await?));
        }

        // Try WebGPU
        #[cfg(feature = "wgpu")]
        if ringkernel_wgpu::is_wgpu_available() {
            tracing::info!("Auto-selected WebGPU backend");
            return Ok(Box::new(ringkernel_wgpu::WgpuRuntime::new().await?));
        }

        // Fall back to CPU
        tracing::info!("Auto-selected CPU backend (no GPU available)");
        Ok(Box::new(CpuRuntime::new().await?))
    }
}

impl Default for RingKernelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Get list of registered kernels from the inventory.
pub fn registered_kernels() -> Vec<&'static str> {
    inventory::iter::<__private::KernelRegistration>
        .into_iter()
        .map(|r| r.id)
        .collect()
}

/// Check availability of backends at runtime.
pub mod availability {
    /// Check if CUDA is available.
    pub fn cuda() -> bool {
        #[cfg(feature = "cuda")]
        {
            ringkernel_cuda::is_cuda_available()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Check if Metal is available.
    pub fn metal() -> bool {
        #[cfg(feature = "metal")]
        {
            ringkernel_metal::is_metal_available()
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }

    /// Check if WebGPU is available.
    pub fn wgpu() -> bool {
        #[cfg(feature = "wgpu")]
        {
            ringkernel_wgpu::is_wgpu_available()
        }
        #[cfg(not(feature = "wgpu"))]
        {
            false
        }
    }

    /// Get list of available backends.
    pub fn available_backends() -> Vec<super::Backend> {
        let mut backends = vec![super::Backend::Cpu];

        if cuda() {
            backends.push(super::Backend::Cuda);
        }
        if metal() {
            backends.push(super::Backend::Metal);
        }
        if wgpu() {
            backends.push(super::Backend::Wgpu);
        }

        backends
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_runtime_creation() {
        let runtime = RingKernel::new().await.unwrap();
        // Default backend is Auto which selects best available (CUDA > Metal > WebGPU > CPU)
        // On systems with CUDA, this will select CUDA; otherwise CPU
        let backend = runtime.backend();
        assert!(
            backend == Backend::Cuda || backend == Backend::Cpu,
            "Expected Cuda or Cpu backend, got {:?}",
            backend
        );
    }

    #[tokio::test]
    async fn test_builder() {
        let runtime = RingKernel::builder()
            .backend(Backend::Cpu)
            .debug(true)
            .build()
            .await
            .unwrap();

        assert_eq!(runtime.backend(), Backend::Cpu);
    }

    #[tokio::test]
    async fn test_launch_kernel() {
        let runtime = RingKernel::new().await.unwrap();

        let kernel = runtime
            .launch("test_kernel", LaunchOptions::default())
            .await
            .unwrap();

        assert_eq!(kernel.id().as_str(), "test_kernel");
    }

    #[test]
    fn test_availability() {
        // CPU should always be available
        let backends = availability::available_backends();
        assert!(backends.contains(&Backend::Cpu));
    }
}
