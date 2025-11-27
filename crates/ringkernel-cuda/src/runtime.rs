//! CUDA runtime implementation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use parking_lot::RwLock;

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::runtime::{
    Backend, KernelHandle, KernelId, LaunchOptions, RingKernelRuntime, RuntimeMetrics,
};

use crate::device::{CudaDevice, enumerate_devices};
use crate::kernel::CudaKernel;
use crate::memory::CudaMemoryPool;
use crate::RING_KERNEL_PTX_TEMPLATE;

/// CUDA runtime for RingKernel.
pub struct CudaRuntime {
    /// Primary device.
    device: CudaDevice,
    /// Memory pool.
    memory_pool: RwLock<CudaMemoryPool>,
    /// Active kernels.
    kernels: RwLock<HashMap<KernelId, Arc<CudaKernel>>>,
    /// Kernel ID counter.
    kernel_counter: AtomicU64,
    /// Total kernels launched.
    total_launched: AtomicU64,
    /// Runtime start time.
    start_time: Instant,
}

impl CudaRuntime {
    /// Create a new CUDA runtime.
    pub async fn new() -> Result<Self> {
        Self::with_device(0).await
    }

    /// Create a runtime with a specific device.
    pub async fn with_device(device_index: usize) -> Result<Self> {
        let device = CudaDevice::new(device_index)?;

        // Check compute capability
        let (major, minor) = device.compute_capability();
        if major < 7 {
            tracing::warn!(
                "CUDA device {} has compute capability {}.{}, persistent kernels require 7.0+",
                device_index,
                major,
                minor
            );
        }

        tracing::info!(
            "Initialized CUDA runtime on device {} ({}) with CC {}.{}",
            device_index,
            device.name(),
            major,
            minor
        );

        let memory_pool = CudaMemoryPool::new(device.clone());

        Ok(Self {
            device,
            memory_pool: RwLock::new(memory_pool),
            kernels: RwLock::new(HashMap::new()),
            kernel_counter: AtomicU64::new(0),
            total_launched: AtomicU64::new(0),
            start_time: Instant::now(),
        })
    }

    /// Get device info.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Check if persistent kernels are supported.
    pub fn supports_persistent_kernels(&self) -> bool {
        self.device.supports_persistent_kernels()
    }
}

#[async_trait]
impl RingKernelRuntime for CudaRuntime {
    fn backend(&self) -> Backend {
        Backend::Cuda
    }

    fn is_backend_available(&self, backend: Backend) -> bool {
        match backend {
            Backend::Cuda => true,
            Backend::Cpu => true, // CPU fallback always available
            _ => false,
        }
    }

    async fn launch(&self, kernel_id: &str, options: LaunchOptions) -> Result<KernelHandle> {
        // Check for duplicate
        let id = KernelId::new(kernel_id);
        if self.kernels.read().contains_key(&id) {
            return Err(RingKernelError::KernelAlreadyExists(kernel_id.to_string()));
        }

        let id_num = self.kernel_counter.fetch_add(1, Ordering::Relaxed);

        // Create kernel
        let mut kernel = CudaKernel::new(kernel_id, id_num, self.device.clone(), options)?;

        // Load PTX (using template for now)
        kernel.load_ptx(RING_KERNEL_PTX_TEMPLATE)?;

        let kernel = Arc::new(kernel);
        self.kernels.write().insert(id.clone(), Arc::clone(&kernel));
        self.total_launched.fetch_add(1, Ordering::Relaxed);

        tracing::info!(kernel_id = %kernel_id, "Launched CUDA kernel");

        Ok(KernelHandle::new(kernel))
    }

    fn get_kernel(&self, kernel_id: &KernelId) -> Option<KernelHandle> {
        self.kernels
            .read()
            .get(kernel_id)
            .map(|k| KernelHandle::new(Arc::clone(k)))
    }

    fn list_kernels(&self) -> Vec<KernelId> {
        self.kernels.read().keys().cloned().collect()
    }

    fn metrics(&self) -> RuntimeMetrics {
        let kernels = self.kernels.read();
        RuntimeMetrics {
            backend: Backend::Cuda,
            active_kernels: kernels.len(),
            total_launched: self.total_launched.load(Ordering::Relaxed) as usize,
            uptime: self.start_time.elapsed(),
            memory_used: 0, // TODO: Track actual GPU memory usage
            memory_total: self.device.total_memory(),
        }
    }

    async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down CUDA runtime");

        // Terminate all kernels
        let kernel_ids: Vec<_> = self.kernels.read().keys().cloned().collect();

        for id in kernel_ids {
            if let Some(kernel) = self.kernels.read().get(&id) {
                if let Err(e) = kernel.terminate().await {
                    tracing::warn!(kernel_id = %id, error = %e, "Failed to terminate kernel");
                }
            }
        }

        // Clear kernels
        self.kernels.write().clear();

        // Synchronize device
        self.device.synchronize()?;

        tracing::info!("CUDA runtime shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires CUDA hardware
    async fn test_cuda_runtime_creation() {
        let runtime = CudaRuntime::new().await.unwrap();
        assert_eq!(runtime.backend(), Backend::Cuda);
    }

    #[tokio::test]
    #[ignore] // Requires CUDA hardware
    async fn test_cuda_kernel_launch() {
        let runtime = CudaRuntime::new().await.unwrap();
        let kernel = runtime.launch("test_kernel", LaunchOptions::default()).await.unwrap();

        assert_eq!(kernel.id(), "test_kernel");
        assert_eq!(kernel.status().state, ringkernel_core::runtime::KernelState::Initialized);

        runtime.shutdown().await.unwrap();
    }
}
