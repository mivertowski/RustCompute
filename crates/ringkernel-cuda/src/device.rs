//! CUDA device management.

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;

/// Type alias for Arc-wrapped CudaStream
type StreamHandle = Arc<CudaStream>;

use ringkernel_core::error::{Result, RingKernelError};

use crate::persistent::CudaMappedBuffer;

/// Wrapper around cudarc CudaContext with RingKernel utilities.
pub struct CudaDevice {
    /// The underlying cudarc context.
    inner: Arc<CudaContext>,
    /// Default stream for operations.
    stream: StreamHandle,
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
        let inner = CudaContext::new(ordinal).map_err(|e| {
            RingKernelError::BackendError(format!(
                "Failed to create CUDA device {}: {}",
                ordinal, e
            ))
        })?;

        let name = inner.name().map_err(|e| {
            RingKernelError::BackendError(format!("Failed to get device name: {}", e))
        })?;

        // Get compute capability
        let (major, minor) = inner.compute_capability().map_err(|e| {
            RingKernelError::BackendError(format!("Failed to get compute capability: {}", e))
        })?;

        // Create default stream
        let stream = inner.default_stream();

        // Get total memory - use a reasonable default for modern GPUs
        // cudarc 0.18 doesn't expose direct memory query easily
        let total_memory = 8 * 1024 * 1024 * 1024; // 8GB default

        Ok(Self {
            inner,
            stream,
            ordinal,
            name,
            compute_capability: (major as u32, minor as u32),
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

    /// Get the underlying cudarc context.
    pub fn inner(&self) -> &Arc<CudaContext> {
        &self.inner
    }

    /// Get the default stream.
    pub fn stream(&self) -> &StreamHandle {
        &self.stream
    }

    /// Allocate device memory.
    pub fn alloc<T: cudarc::driver::DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>> {
        // Safety: Allocating uninitialized GPU memory is safe as long as we don't read
        // from it before initializing. The caller is responsible for initialization.
        unsafe {
            self.stream
                .alloc::<T>(len)
                .map_err(|_| RingKernelError::OutOfMemory {
                    requested: len * std::mem::size_of::<T>(),
                    available: 0,
                })
        }
    }

    /// Copy data from host to device.
    pub fn htod_copy<T: cudarc::driver::DeviceRepr + Clone + Unpin>(
        &self,
        src: &[T],
    ) -> Result<CudaSlice<T>> {
        // Allocate device memory
        let mut dst = self.alloc::<T>(src.len())?;
        // Copy data
        self.stream
            .memcpy_htod(src, &mut dst)
            .map_err(|e| RingKernelError::TransferFailed(format!("HtoD copy failed: {}", e)))?;
        Ok(dst)
    }

    /// Copy data from device to host.
    pub fn dtoh_copy<T: cudarc::driver::DeviceRepr + Clone + Default>(
        &self,
        src: &CudaSlice<T>,
    ) -> Result<Vec<T>> {
        let mut dst = vec![T::default(); src.len()];
        self.stream
            .memcpy_dtoh(src, &mut dst)
            .map_err(|e| RingKernelError::TransferFailed(format!("DtoH copy failed: {}", e)))?;
        Ok(dst)
    }

    /// Allocate mapped memory visible to both CPU and GPU.
    ///
    /// Mapped memory (also called pinned or page-locked memory) can be accessed
    /// by both the CPU and GPU without explicit memory transfers. This is ideal
    /// for:
    /// - Control blocks and status flags
    /// - Lock-free message queues
    /// - Reduction accumulators that need to be read by the host
    ///
    /// # Arguments
    ///
    /// * `len` - Number of elements to allocate
    ///
    /// # Returns
    ///
    /// A `CudaMappedBuffer<T>` that provides both host and device pointers.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use ringkernel_cuda::CudaDevice;
    ///
    /// let device = CudaDevice::new(0)?;
    /// let buffer = device.alloc_mapped::<f64>(1024)?;
    ///
    /// // Write from CPU
    /// buffer.write(0, 42.0);
    ///
    /// // Pass device_ptr() to GPU kernel
    /// let gpu_ptr = buffer.device_ptr();
    /// ```
    pub fn alloc_mapped<T: Copy + Send + Sync>(&self, len: usize) -> Result<CudaMappedBuffer<T>> {
        CudaMappedBuffer::new(self, len)
    }

    /// Check if device supports cooperative groups (requires CC 6.0+).
    ///
    /// Cooperative groups enable grid-wide synchronization via `grid.sync()`,
    /// which is required for efficient reduce-and-broadcast operations.
    pub fn supports_cooperative_groups(&self) -> bool {
        self.compute_capability.0 >= 6
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
            stream: self.inner.default_stream(),
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
    let count = CudaContext::device_count().map_err(|e| {
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
