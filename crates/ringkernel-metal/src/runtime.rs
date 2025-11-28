//! Metal runtime implementation.

#![cfg(all(target_os = "macos", feature = "metal"))]

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::runtime::{
    Backend, KernelHandle, KernelId, KernelState, KernelStatus, LaunchOptions, RingKernelRuntime,
    RuntimeMetrics,
};
use ringkernel_core::telemetry::TelemetryBuffer;

use crate::device::MetalDevice;
use crate::kernel::MetalKernel;

/// Metal runtime for RingKernel.
pub struct MetalRuntime {
    /// Metal device.
    device: Arc<MetalDevice>,
    /// Active kernels.
    kernels: RwLock<HashMap<KernelId, Arc<RwLock<MetalKernel>>>>,
    /// Kernel counter for unique IDs.
    kernel_counter: AtomicU64,
    /// Total kernels launched.
    total_launched: AtomicU64,
}

impl MetalRuntime {
    /// Create a new Metal runtime.
    pub async fn new() -> Result<Self> {
        let device = MetalDevice::new()?;

        Ok(Self {
            device: Arc::new(device),
            kernels: RwLock::new(HashMap::new()),
            kernel_counter: AtomicU64::new(0),
            total_launched: AtomicU64::new(0),
        })
    }
}

#[async_trait]
impl RingKernelRuntime for MetalRuntime {
    fn backend(&self) -> Backend {
        Backend::Metal
    }

    fn is_backend_available(&self, backend: Backend) -> bool {
        matches!(backend, Backend::Metal)
    }

    async fn launch(&self, kernel_id: &str, options: LaunchOptions) -> Result<KernelHandle> {
        let id_num = self.kernel_counter.fetch_add(1, Ordering::Relaxed);

        let kernel = MetalKernel::new(kernel_id, id_num, &self.device, options)?;
        let kernel_id_obj = kernel.id().clone();

        let kernel = Arc::new(RwLock::new(kernel));
        self.kernels
            .write()
            .insert(kernel_id_obj.clone(), Arc::clone(&kernel));

        self.total_launched.fetch_add(1, Ordering::Relaxed);

        Ok(KernelHandle::new(kernel_id_obj, id_num))
    }

    fn get_kernel(&self, kernel_id: &KernelId) -> Option<KernelHandle> {
        self.kernels.read().get(kernel_id).map(|k| {
            let kernel = k.read();
            KernelHandle::new(kernel.id().clone(), kernel.kernel_id())
        })
    }

    fn list_kernels(&self) -> Vec<KernelId> {
        self.kernels.read().keys().cloned().collect()
    }

    fn metrics(&self) -> RuntimeMetrics {
        let kernels = self.kernels.read();
        let active = kernels
            .values()
            .filter(|k| k.read().state() == KernelState::Active)
            .count();

        RuntimeMetrics {
            total_launched: self.total_launched.load(Ordering::Relaxed),
            active_kernels: active as u64,
            total_messages_sent: 0,
            total_messages_received: 0,
            uptime_secs: 0,
        }
    }

    async fn shutdown(&self) -> Result<()> {
        let mut kernels = self.kernels.write();
        for kernel in kernels.values() {
            let _ = kernel.write().terminate();
        }
        kernels.clear();
        Ok(())
    }
}
