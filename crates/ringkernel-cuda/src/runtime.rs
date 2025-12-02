//! CUDA runtime implementation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use parking_lot::RwLock;

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::k2k::{K2KBroker, K2KBuilder, K2KConfig};
use ringkernel_core::runtime::{
    Backend, KernelHandle, KernelHandleInner, KernelId, LaunchOptions, RingKernelRuntime,
    RuntimeMetrics,
};

use crate::device::CudaDevice;
use crate::kernel::CudaKernel;
use crate::memory::CudaMemoryPool;
use crate::RING_KERNEL_PTX_TEMPLATE;

/// CUDA runtime for RingKernel.
pub struct CudaRuntime {
    /// Primary device.
    device: CudaDevice,
    /// Memory pool.
    #[allow(dead_code)]
    memory_pool: RwLock<CudaMemoryPool>,
    /// Active kernels.
    kernels: RwLock<HashMap<KernelId, Arc<CudaKernel>>>,
    /// Kernel ID counter.
    kernel_counter: AtomicU64,
    /// Total kernels launched.
    total_launched: AtomicU64,
    /// Runtime start time.
    #[allow(dead_code)]
    start_time: Instant,
    /// K2K broker for kernel-to-kernel messaging.
    k2k_broker: Option<Arc<K2KBroker>>,
}

impl CudaRuntime {
    /// Create a new CUDA runtime.
    pub async fn new() -> Result<Self> {
        Self::with_device(0).await
    }

    /// Create a runtime with a specific device.
    pub async fn with_device(device_index: usize) -> Result<Self> {
        Self::with_config(device_index, true).await
    }

    /// Create a runtime with configuration options.
    pub async fn with_config(device_index: usize, enable_k2k: bool) -> Result<Self> {
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
            "Initialized CUDA runtime on device {} ({}) with CC {}.{}, k2k={}",
            device_index,
            device.name(),
            major,
            minor,
            enable_k2k
        );

        let memory_pool = CudaMemoryPool::new(device.clone());

        let k2k_broker = if enable_k2k {
            Some(K2KBuilder::new().build())
        } else {
            None
        };

        Ok(Self {
            device,
            memory_pool: RwLock::new(memory_pool),
            kernels: RwLock::new(HashMap::new()),
            kernel_counter: AtomicU64::new(0),
            total_launched: AtomicU64::new(0),
            start_time: Instant::now(),
            k2k_broker,
        })
    }

    /// Create a runtime with custom K2K configuration.
    pub async fn with_k2k_config(device_index: usize, k2k_config: K2KConfig) -> Result<Self> {
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
            "Initialized CUDA runtime with custom K2K config on device {} ({}) with CC {}.{}",
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
            k2k_broker: Some(K2KBroker::new(k2k_config)),
        })
    }

    /// Get device info.
    #[allow(dead_code)]
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Check if persistent kernels are supported.
    #[allow(dead_code)]
    pub fn supports_persistent_kernels(&self) -> bool {
        self.device.supports_persistent_kernels()
    }

    /// Check if K2K messaging is enabled.
    pub fn is_k2k_enabled(&self) -> bool {
        self.k2k_broker.is_some()
    }

    /// Get the K2K broker (if enabled).
    pub fn k2k_broker(&self) -> Option<&Arc<K2KBroker>> {
        self.k2k_broker.as_ref()
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
            return Err(RingKernelError::KernelAlreadyActive(kernel_id.to_string()));
        }

        let id_num = self.kernel_counter.fetch_add(1, Ordering::Relaxed);

        // Register with K2K broker if enabled
        // Note: CUDA K2K is host-mediated - messages go through host, not GPU-direct
        let _k2k_endpoint = self.k2k_broker.as_ref().map(|broker| broker.register(id.clone()));

        // Create kernel
        let mut kernel = CudaKernel::new(kernel_id, id_num, self.device.clone(), options)?;

        // Load PTX (using template for now)
        kernel.load_ptx(RING_KERNEL_PTX_TEMPLATE)?;

        let kernel = Arc::new(kernel);
        self.kernels.write().insert(id.clone(), Arc::clone(&kernel));
        self.total_launched.fetch_add(1, Ordering::Relaxed);

        tracing::info!(
            kernel_id = %kernel_id,
            k2k = %self.is_k2k_enabled(),
            "Launched CUDA kernel"
        );

        Ok(KernelHandle::new(id, kernel))
    }

    fn get_kernel(&self, kernel_id: &KernelId) -> Option<KernelHandle> {
        self.kernels.read().get(kernel_id).map(|k| {
            let inner: Arc<dyn KernelHandleInner> = Arc::clone(k) as Arc<dyn KernelHandleInner>;
            KernelHandle::new(kernel_id.clone(), inner)
        })
    }

    fn list_kernels(&self) -> Vec<KernelId> {
        self.kernels.read().keys().cloned().collect()
    }

    fn metrics(&self) -> RuntimeMetrics {
        RuntimeMetrics {
            active_kernels: self.kernels.read().len(),
            total_launched: self.total_launched.load(Ordering::Relaxed),
            messages_sent: 0,
            messages_received: 0,
            gpu_memory_used: 0,
            host_memory_used: 0,
        }
    }

    async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down CUDA runtime");

        // Terminate all kernels
        let kernel_ids: Vec<_> = self.kernels.read().keys().cloned().collect();

        for id in kernel_ids.iter() {
            let kernel = self.kernels.read().get(id).cloned();
            if let Some(kernel) = kernel {
                if let Err(e) = kernel.terminate().await {
                    tracing::warn!(kernel_id = %id, error = %e, "Failed to terminate kernel");
                }
            }
            // Unregister from K2K broker
            if let Some(broker) = &self.k2k_broker {
                broker.unregister(id);
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
        let kernel = runtime
            .launch("test_kernel", LaunchOptions::default())
            .await
            .unwrap();

        assert_eq!(kernel.id().as_str(), "test_kernel");
        assert_eq!(
            kernel.status().state,
            ringkernel_core::runtime::KernelState::Launched
        );

        runtime.shutdown().await.unwrap();
    }
}
