//! Metal device abstraction.

#![cfg(all(target_os = "macos", feature = "metal"))]

use metal::Device;
use ringkernel_core::error::{Result, RingKernelError};

/// Metal device wrapper for GPU operations.
pub struct MetalDevice {
    /// The underlying Metal device.
    device: Device,
    /// Device name.
    name: String,
}

impl MetalDevice {
    /// Create a new Metal device using the system default.
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| {
            RingKernelError::BackendUnavailable("No Metal device found".to_string())
        })?;

        let name = device.name().to_string();

        Ok(Self { device, name })
    }

    /// Get the underlying Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the device name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Check if the device supports unified memory.
    pub fn has_unified_memory(&self) -> bool {
        self.device.has_unified_memory()
    }

    /// Get the maximum threads per threadgroup.
    pub fn max_threads_per_threadgroup(&self) -> u64 {
        self.device.max_threads_per_threadgroup().width
    }
}
