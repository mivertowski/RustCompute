//! WebGPU runtime implementation.

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

use crate::adapter::WgpuAdapter;
use crate::kernel::WgpuKernel;
use crate::RING_KERNEL_WGSL_TEMPLATE;

/// WebGPU runtime for RingKernel.
pub struct WgpuRuntime {
    /// WebGPU adapter.
    adapter: Arc<WgpuAdapter>,
    /// Active kernels.
    kernels: RwLock<HashMap<KernelId, Arc<WgpuKernel>>>,
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

impl WgpuRuntime {
    /// Create a new WebGPU runtime.
    pub async fn new() -> Result<Self> {
        Self::with_config(true).await
    }

    /// Create a runtime with configuration options.
    pub async fn with_config(enable_k2k: bool) -> Result<Self> {
        let adapter = WgpuAdapter::new().await?;

        tracing::info!(
            "Initialized WebGPU runtime on {} ({:?}), k2k={}",
            adapter.name(),
            adapter.backend(),
            enable_k2k
        );

        let k2k_broker = if enable_k2k {
            Some(K2KBuilder::new().build())
        } else {
            None
        };

        Ok(Self {
            adapter: Arc::new(adapter),
            kernels: RwLock::new(HashMap::new()),
            kernel_counter: AtomicU64::new(0),
            total_launched: AtomicU64::new(0),
            start_time: Instant::now(),
            k2k_broker,
        })
    }

    /// Create a runtime with custom K2K configuration.
    pub async fn with_k2k_config(k2k_config: K2KConfig) -> Result<Self> {
        let adapter = WgpuAdapter::new().await?;

        tracing::info!(
            "Initialized WebGPU runtime with custom K2K config on {} ({:?})",
            adapter.name(),
            adapter.backend()
        );

        Ok(Self {
            adapter: Arc::new(adapter),
            kernels: RwLock::new(HashMap::new()),
            kernel_counter: AtomicU64::new(0),
            total_launched: AtomicU64::new(0),
            start_time: Instant::now(),
            k2k_broker: Some(K2KBroker::new(k2k_config)),
        })
    }

    /// Get adapter info.
    #[allow(dead_code)]
    pub fn adapter(&self) -> &WgpuAdapter {
        &self.adapter
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
impl RingKernelRuntime for WgpuRuntime {
    fn backend(&self) -> Backend {
        Backend::Wgpu
    }

    fn is_backend_available(&self, backend: Backend) -> bool {
        match backend {
            Backend::Wgpu => true,
            Backend::Cpu => true,
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
        // Note: WebGPU K2K is host-mediated - messages go through host, not GPU-direct
        let _k2k_endpoint = self
            .k2k_broker
            .as_ref()
            .map(|broker| broker.register(id.clone()));

        // Create kernel
        let mut kernel = WgpuKernel::new(kernel_id, id_num, Arc::clone(&self.adapter), options)?;

        // Load shader
        kernel.load_shader(RING_KERNEL_WGSL_TEMPLATE)?;

        let kernel = Arc::new(kernel);
        self.kernels.write().insert(id.clone(), Arc::clone(&kernel));
        self.total_launched.fetch_add(1, Ordering::Relaxed);

        tracing::info!(
            kernel_id = %kernel_id,
            k2k = %self.is_k2k_enabled(),
            "Launched WebGPU kernel"
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
        tracing::info!("Shutting down WebGPU runtime");

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

        // Wait for GPU to finish
        self.adapter.poll(wgpu::Maintain::Wait);

        tracing::info!("WebGPU runtime shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // May not have GPU in CI
    async fn test_wgpu_runtime_creation() {
        let runtime = WgpuRuntime::new().await.unwrap();
        assert_eq!(runtime.backend(), Backend::Wgpu);
    }

    #[tokio::test]
    #[ignore] // May not have GPU in CI
    async fn test_wgpu_kernel_launch() {
        let runtime = WgpuRuntime::new().await.unwrap();
        let kernel = runtime
            .launch("test_kernel", LaunchOptions::default())
            .await
            .unwrap();

        assert_eq!(kernel.id().as_str(), "test_kernel");
        runtime.shutdown().await.unwrap();
    }
}
