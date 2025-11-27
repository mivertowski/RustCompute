//! WebGPU adapter management.

use std::sync::Arc;

use ringkernel_core::error::{Result, RingKernelError};

/// Wrapper around wgpu adapter and device.
pub struct WgpuAdapter {
    /// The wgpu instance.
    #[allow(dead_code)]
    instance: wgpu::Instance,
    /// The selected adapter.
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    /// The device.
    device: Arc<wgpu::Device>,
    /// The command queue.
    queue: Arc<wgpu::Queue>,
    /// Adapter info.
    info: wgpu::AdapterInfo,
}

impl WgpuAdapter {
    /// Create a new WebGPU adapter with default settings.
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                RingKernelError::BackendUnavailable("No WebGPU adapter found".to_string())
            })?;

        let info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("RingKernel Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| {
                RingKernelError::BackendError(format!("Failed to create device: {}", e))
            })?;

        tracing::info!(
            "Created WebGPU adapter: {} ({:?})",
            info.name,
            info.backend
        );

        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            info,
        })
    }

    /// Get the adapter name.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Get the backend type.
    pub fn backend(&self) -> wgpu::Backend {
        self.info.backend
    }

    /// Get device type.
    pub fn device_type(&self) -> wgpu::DeviceType {
        self.info.device_type
    }

    /// Get the wgpu device.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Get the command queue.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Get adapter limits.
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// Poll the device for completed work.
    pub fn poll(&self, maintain: wgpu::Maintain) -> wgpu::MaintainResult {
        self.device.poll(maintain)
    }
}

/// Enumerate available adapters.
#[allow(dead_code)]
pub async fn enumerate_adapters() -> Vec<WgpuAdapterInfo> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

    instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            WgpuAdapterInfo {
                name: info.name.clone(),
                backend: info.backend,
                device_type: info.device_type,
            }
        })
        .collect()
}

/// Information about a WebGPU adapter.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct WgpuAdapterInfo {
    /// Adapter name.
    pub name: String,
    /// Backend type (Vulkan, Metal, DX12, etc.).
    pub backend: wgpu::Backend,
    /// Device type (discrete GPU, integrated, etc.).
    pub device_type: wgpu::DeviceType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // May not have GPU in CI
    async fn test_adapter_creation() {
        let adapter = WgpuAdapter::new().await.unwrap();
        println!("Adapter: {} ({:?})", adapter.name(), adapter.backend());
    }
}
