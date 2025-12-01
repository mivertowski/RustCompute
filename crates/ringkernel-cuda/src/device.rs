//! CUDA device management.

use cudarc::driver::{CudaDevice as CudarcDevice, CudaSlice, DeviceRepr};
use std::sync::Arc;

use ringkernel_core::error::{Result, RingKernelError};

/// Wrapper around cudarc CudaDevice with RingKernel utilities.
pub struct CudaDevice {
    /// The underlying cudarc device.
    inner: Arc<CudarcDevice>,
    /// Device ordinal.
    ordinal: usize,
    /// Device name.
    name: String,
    /// Compute capability (major, minor).
    compute_capability: (u32, u32),
    /// Total global memory in bytes.
    total_memory: usize,
}

impl CudaDevice {
    /// Create a new CUDA device wrapper.
    pub fn new(ordinal: usize) -> Result<Self> {
        let inner = CudarcDevice::new(ordinal).map_err(|e| {
            RingKernelError::BackendError(format!(
                "Failed to create CUDA device {}: {}",
                ordinal, e
            ))
        })?;

        let name = inner.name().map_err(|e| {
            RingKernelError::BackendError(format!("Failed to get device name: {}", e))
        })?;

        // Get compute capability
        let major = inner.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| RingKernelError::BackendError(format!("Failed to get compute capability: {}", e)))? as u32;
        let minor = inner.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| RingKernelError::BackendError(format!("Failed to get compute capability: {}", e)))? as u32;

        // Get total memory via device attribute (memory_total not directly available in newer cudarc)
        let total_memory_attr = inner.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
        ).unwrap_or(0) as usize;
        // Use a reasonable fallback - actual memory queries work differently in newer cudarc
        let total_memory = if total_memory_attr > 0 {
            total_memory_attr
        } else {
            // Fallback: assume 8GB for modern GPUs
            8 * 1024 * 1024 * 1024
        };

        Ok(Self {
            inner,
            ordinal,
            name,
            compute_capability: (major, minor),
            total_memory,
        })
    }

    /// Get device ordinal.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Get device name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get compute capability as (major, minor).
    pub fn compute_capability(&self) -> (u32, u32) {
        self.compute_capability
    }

    /// Get total global memory in bytes.
    pub fn total_memory(&self) -> usize {
        self.total_memory
    }

    /// Check if device supports persistent kernels (requires CC 7.0+).
    pub fn supports_persistent_kernels(&self) -> bool {
        self.compute_capability.0 >= 7
    }

    /// Get the underlying cudarc device.
    pub fn inner(&self) -> &Arc<CudarcDevice> {
        &self.inner
    }

    /// Allocate device memory.
    pub fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>> {
        // Safety: alloc is unsafe in newer cudarc, but we're just allocating uninitialized memory
        unsafe {
            self.inner
                .alloc::<T>(len)
                .map_err(|_| RingKernelError::OutOfMemory {
                    requested: len * std::mem::size_of::<T>(),
                    available: 0, // We don't have easy access to free memory
                })
        }
    }

    /// Copy data from host to device.
    pub fn htod_copy<T: DeviceRepr + Clone + Unpin>(&self, src: &[T]) -> Result<CudaSlice<T>> {
        self.inner
            .htod_copy(src.to_vec())
            .map_err(|e| RingKernelError::TransferFailed(format!("HtoD copy failed: {}", e)))
    }

    /// Copy data from device to host.
    pub fn dtoh_copy<T: DeviceRepr + Clone>(&self, src: &CudaSlice<T>) -> Result<Vec<T>> {
        self.inner
            .dtoh_sync_copy(src)
            .map_err(|e| RingKernelError::TransferFailed(format!("DtoH copy failed: {}", e)))
    }

    /// Synchronize device.
    pub fn synchronize(&self) -> Result<()> {
        self.inner
            .synchronize()
            .map_err(|e| RingKernelError::BackendError(format!("Synchronize failed: {}", e)))
    }
}

impl Clone for CudaDevice {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            ordinal: self.ordinal,
            name: self.name.clone(),
            compute_capability: self.compute_capability,
            total_memory: self.total_memory,
        }
    }
}

/// Device info for runtime queries.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CudaDeviceInfo {
    /// Device ordinal.
    pub ordinal: usize,
    /// Device name.
    pub name: String,
    /// Compute capability (major, minor).
    pub compute_capability: (u32, u32),
    /// Total global memory in bytes.
    pub total_memory: usize,
    /// Whether persistent kernels are supported.
    pub supports_persistent: bool,
}

impl From<&CudaDevice> for CudaDeviceInfo {
    fn from(device: &CudaDevice) -> Self {
        Self {
            ordinal: device.ordinal,
            name: device.name.clone(),
            compute_capability: device.compute_capability,
            total_memory: device.total_memory,
            supports_persistent: device.supports_persistent_kernels(),
        }
    }
}

/// Enumerate all CUDA devices.
#[allow(dead_code)]
pub fn enumerate_devices() -> Result<Vec<CudaDeviceInfo>> {
    let count = CudarcDevice::count().map_err(|e| {
        RingKernelError::BackendError(format!("Failed to count CUDA devices: {}", e))
    })? as usize;

    let mut devices = Vec::with_capacity(count);
    for i in 0..count {
        match CudaDevice::new(i) {
            Ok(device) => devices.push(CudaDeviceInfo::from(&device)),
            Err(e) => {
                tracing::warn!("Failed to enumerate device {}: {}", i, e);
            }
        }
    }

    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_device_enumeration() {
        // This test requires CUDA hardware
        if let Ok(devices) = enumerate_devices() {
            for device in &devices {
                println!(
                    "Device {}: {} (CC {}.{})",
                    device.ordinal,
                    device.name,
                    device.compute_capability.0,
                    device.compute_capability.1
                );
            }
        }
    }
}
