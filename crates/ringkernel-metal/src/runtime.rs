//! Metal runtime implementation.

#![cfg(all(target_os = "macos", feature = "metal"))]

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::k2k::{K2KBroker, K2KBuilder, K2KConfig};
use ringkernel_core::runtime::{
    Backend, KernelHandle, KernelHandleInner, KernelId, KernelState, LaunchOptions,
    RingKernelRuntime, RuntimeMetrics,
};

use crate::device::MetalDevice;
use crate::kernel::MetalKernel;
use crate::RING_KERNEL_MSL_TEMPLATE;

/// Metal runtime for RingKernel.
pub struct MetalRuntime {
    /// Metal device.
    device: Arc<MetalDevice>,
    /// Active kernels.
    kernels: RwLock<HashMap<KernelId, Arc<MetalKernel>>>,
    /// Kernel counter for unique IDs.
    kernel_counter: AtomicU64,
    /// Total kernels launched.
    total_launched: AtomicU64,
    /// Runtime start time.
    #[allow(dead_code)]
    start_time: Instant,
    /// K2K broker for kernel-to-kernel messaging.
    k2k_broker: Option<Arc<K2KBroker>>,
}

impl MetalRuntime {
    /// Create a new Metal runtime.
    pub async fn new() -> Result<Self> {
        Self::with_config(true).await
    }

    /// Create a runtime with configuration options.
    pub async fn with_config(enable_k2k: bool) -> Result<Self> {
        let device = MetalDevice::new()?;

        tracing::info!(
            "Initialized Metal runtime on {}, k2k={}",
            device.name(),
            enable_k2k
        );

        let k2k_broker = if enable_k2k {
            Some(K2KBuilder::new().build())
        } else {
            None
        };

        Ok(Self {
            device: Arc::new(device),
            kernels: RwLock::new(HashMap::new()),
            kernel_counter: AtomicU64::new(0),
            total_launched: AtomicU64::new(0),
            start_time: Instant::now(),
            k2k_broker,
        })
    }

    /// Create a runtime with custom K2K configuration.
    pub async fn with_k2k_config(k2k_config: K2KConfig) -> Result<Self> {
        let device = MetalDevice::new()?;

        tracing::info!(
            "Initialized Metal runtime with custom K2K config on {}",
            device.name()
        );

        Ok(Self {
            device: Arc::new(device),
            kernels: RwLock::new(HashMap::new()),
            kernel_counter: AtomicU64::new(0),
            total_launched: AtomicU64::new(0),
            start_time: Instant::now(),
            k2k_broker: Some(K2KBroker::new(k2k_config)),
        })
    }

    /// Get the Metal device.
    pub fn device(&self) -> &MetalDevice {
        &self.device
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
impl RingKernelRuntime for MetalRuntime {
    fn backend(&self) -> Backend {
        Backend::Metal
    }

    fn is_backend_available(&self, backend: Backend) -> bool {
        matches!(backend, Backend::Metal | Backend::Cpu)
    }

    async fn launch(&self, kernel_id: &str, options: LaunchOptions) -> Result<KernelHandle> {
        // Check for duplicate
        let id = KernelId::new(kernel_id);
        if self.kernels.read().contains_key(&id) {
            return Err(RingKernelError::KernelAlreadyActive(kernel_id.to_string()));
        }

        let id_num = self.kernel_counter.fetch_add(1, Ordering::Relaxed);

        // Register with K2K broker if enabled
        let _k2k_endpoint = self
            .k2k_broker
            .as_ref()
            .map(|broker| broker.register(id.clone()));

        // Create kernel
        let mut kernel = MetalKernel::new(kernel_id, id_num, Arc::clone(&self.device), options)?;

        // Load default shader
        kernel.load_shader(RING_KERNEL_MSL_TEMPLATE)?;

        let kernel = Arc::new(kernel);
        self.kernels.write().insert(id.clone(), Arc::clone(&kernel));
        self.total_launched.fetch_add(1, Ordering::Relaxed);

        tracing::info!(
            kernel_id = %kernel_id,
            k2k = %self.is_k2k_enabled(),
            "Launched Metal kernel"
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
        let kernels = self.kernels.read();
        let active = kernels
            .values()
            .filter(|k| k.status().state == KernelState::Active)
            .count();

        RuntimeMetrics {
            active_kernels: active,
            total_launched: self.total_launched.load(Ordering::Relaxed),
            messages_sent: 0,
            messages_received: 0,
            gpu_memory_used: 0,
            host_memory_used: 0,
        }
    }

    async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down Metal runtime");

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

        tracing::info!("Metal runtime shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // May not have Metal on CI
    async fn test_metal_runtime_creation() {
        let runtime = MetalRuntime::new().await.unwrap();
        assert_eq!(runtime.backend(), Backend::Metal);
    }

    #[tokio::test]
    #[ignore] // May not have Metal on CI
    async fn test_metal_kernel_launch() {
        let runtime = MetalRuntime::new().await.unwrap();
        let kernel = runtime
            .launch("test_kernel", LaunchOptions::default())
            .await
            .unwrap();

        assert_eq!(kernel.id().as_str(), "test_kernel");
        runtime.shutdown().await.unwrap();
    }
}
