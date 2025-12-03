//! CPU runtime implementation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use tracing::{debug, info};

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::k2k::{K2KBroker, K2KBuilder, K2KConfig};
use ringkernel_core::runtime::{
    Backend, KernelHandle, KernelHandleInner, KernelId, LaunchOptions, RingKernelRuntime,
    RuntimeMetrics,
};

use crate::kernel::CpuKernel;

/// CPU-based implementation of RingKernelRuntime.
///
/// This runtime executes kernels on the CPU, simulating GPU behavior.
/// It's primarily used for testing and as a fallback when no GPU is available.
pub struct CpuRuntime {
    /// Node ID for HLC.
    node_id: u64,
    /// Active kernels.
    kernels: RwLock<HashMap<KernelId, Arc<CpuKernel>>>,
    /// Total kernels launched.
    total_launched: AtomicU64,
    /// Total messages sent.
    messages_sent: AtomicU64,
    /// Total messages received.
    messages_received: AtomicU64,
    /// Shutdown flag.
    shutdown: RwLock<bool>,
    /// K2K broker for kernel-to-kernel messaging.
    k2k_broker: Option<Arc<K2KBroker>>,
}

impl CpuRuntime {
    /// Create a new CPU runtime.
    pub async fn new() -> Result<Self> {
        Self::with_node_id(1).await
    }

    /// Create a CPU runtime with specific node ID.
    pub async fn with_node_id(node_id: u64) -> Result<Self> {
        Self::with_config(node_id, true).await
    }

    /// Create a CPU runtime with configuration options.
    pub async fn with_config(node_id: u64, enable_k2k: bool) -> Result<Self> {
        info!(
            "Initializing CPU runtime (node_id={}, k2k={})",
            node_id, enable_k2k
        );

        let k2k_broker = if enable_k2k {
            Some(K2KBuilder::new().build())
        } else {
            None
        };

        Ok(Self {
            node_id,
            kernels: RwLock::new(HashMap::new()),
            total_launched: AtomicU64::new(0),
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            shutdown: RwLock::new(false),
            k2k_broker,
        })
    }

    /// Create a CPU runtime with custom K2K configuration.
    pub async fn with_k2k_config(node_id: u64, k2k_config: K2KConfig) -> Result<Self> {
        info!(
            "Initializing CPU runtime with custom K2K config (node_id={})",
            node_id
        );

        Ok(Self {
            node_id,
            kernels: RwLock::new(HashMap::new()),
            total_launched: AtomicU64::new(0),
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            shutdown: RwLock::new(false),
            k2k_broker: Some(K2KBroker::new(k2k_config)),
        })
    }

    /// Get node ID.
    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    /// Check if runtime is shut down.
    pub fn is_shutdown(&self) -> bool {
        *self.shutdown.read()
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
impl RingKernelRuntime for CpuRuntime {
    fn backend(&self) -> Backend {
        Backend::Cpu
    }

    fn is_backend_available(&self, backend: Backend) -> bool {
        matches!(backend, Backend::Cpu | Backend::Auto)
    }

    async fn launch(&self, kernel_id: &str, options: LaunchOptions) -> Result<KernelHandle> {
        if self.is_shutdown() {
            return Err(RingKernelError::BackendError(
                "Runtime is shut down".to_string(),
            ));
        }

        let id = KernelId::new(kernel_id);

        // Check if kernel already exists
        {
            let kernels = self.kernels.read();
            if kernels.contains_key(&id) {
                return Err(RingKernelError::InvalidConfig(format!(
                    "Kernel '{}' already exists",
                    kernel_id
                )));
            }
        }

        debug!(
            "Launching CPU kernel '{}' (grid={}, block={}, k2k={})",
            kernel_id,
            options.grid_size,
            options.block_size,
            self.is_k2k_enabled()
        );

        // Register with K2K broker if enabled
        let k2k_endpoint = self
            .k2k_broker
            .as_ref()
            .map(|broker| broker.register(id.clone()));

        // Create kernel with K2K endpoint
        let kernel = Arc::new(CpuKernel::new_with_k2k(
            id.clone(),
            options.clone(),
            self.node_id,
            k2k_endpoint,
        ));
        kernel.launch();

        // Auto-activate if requested
        if options.auto_activate {
            kernel.activate().await?;
        }

        // Store kernel
        {
            let mut kernels = self.kernels.write();
            kernels.insert(id.clone(), Arc::clone(&kernel));
        }

        self.total_launched.fetch_add(1, Ordering::Relaxed);

        info!("CPU kernel '{}' launched successfully", kernel_id);

        Ok(kernel.handle())
    }

    fn get_kernel(&self, kernel_id: &KernelId) -> Option<KernelHandle> {
        let kernels = self.kernels.read();
        kernels.get(kernel_id).map(|k| k.handle())
    }

    fn list_kernels(&self) -> Vec<KernelId> {
        let kernels = self.kernels.read();
        kernels.keys().cloned().collect()
    }

    fn metrics(&self) -> RuntimeMetrics {
        let kernels = self.kernels.read();
        let active = kernels.values().filter(|k| k.state().is_running()).count();

        RuntimeMetrics {
            active_kernels: active,
            total_launched: self.total_launched.load(Ordering::Relaxed),
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),
            gpu_memory_used: 0,
            host_memory_used: 0,
        }
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down CPU runtime");

        // Mark as shutdown
        *self.shutdown.write() = true;

        // Terminate all kernels
        let kernel_ids: Vec<KernelId> = {
            let kernels = self.kernels.read();
            kernels.keys().cloned().collect()
        };

        for id in kernel_ids.iter() {
            if let Some(kernel) = self.get_kernel(id) {
                if let Err(e) = kernel.terminate().await {
                    debug!("Error terminating kernel '{}': {}", id, e);
                }
            }
            // Unregister from K2K broker
            if let Some(broker) = &self.k2k_broker {
                broker.unregister(id);
            }
        }

        // Clear kernel map
        {
            let mut kernels = self.kernels.write();
            kernels.clear();
        }

        info!("CPU runtime shut down complete");
        Ok(())
    }
}

impl Drop for CpuRuntime {
    fn drop(&mut self) {
        if !self.is_shutdown() {
            // Best effort cleanup
            let kernels = self.kernels.get_mut();
            kernels.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_runtime_creation() {
        let runtime = CpuRuntime::new().await.unwrap();
        assert_eq!(runtime.backend(), Backend::Cpu);
        assert!(runtime.is_backend_available(Backend::Cpu));
        assert!(!runtime.is_backend_available(Backend::Cuda));
    }

    #[tokio::test]
    async fn test_kernel_launch() {
        let runtime = CpuRuntime::new().await.unwrap();

        let handle = runtime
            .launch("test_kernel", LaunchOptions::default())
            .await
            .unwrap();

        assert_eq!(handle.id().as_str(), "test_kernel");

        let status = handle.status();
        assert!(status.state.is_running());
    }

    #[tokio::test]
    async fn test_list_kernels() {
        let runtime = CpuRuntime::new().await.unwrap();

        runtime
            .launch("kernel1", LaunchOptions::default())
            .await
            .unwrap();
        runtime
            .launch("kernel2", LaunchOptions::default())
            .await
            .unwrap();

        let ids = runtime.list_kernels();
        assert_eq!(ids.len(), 2);
    }

    #[tokio::test]
    async fn test_duplicate_kernel() {
        let runtime = CpuRuntime::new().await.unwrap();

        runtime
            .launch("test", LaunchOptions::default())
            .await
            .unwrap();

        let result = runtime.launch("test", LaunchOptions::default()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_shutdown() {
        let runtime = CpuRuntime::new().await.unwrap();

        runtime
            .launch("kernel1", LaunchOptions::default())
            .await
            .unwrap();

        runtime.shutdown().await.unwrap();

        assert!(runtime.is_shutdown());
        assert!(runtime.list_kernels().is_empty());
    }

    #[tokio::test]
    async fn test_metrics() {
        let runtime = CpuRuntime::new().await.unwrap();

        runtime
            .launch("kernel1", LaunchOptions::default())
            .await
            .unwrap();
        runtime
            .launch("kernel2", LaunchOptions::default())
            .await
            .unwrap();

        let metrics = runtime.metrics();
        assert_eq!(metrics.active_kernels, 2);
        assert_eq!(metrics.total_launched, 2);
    }
}
