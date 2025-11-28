//! CPU memory buffer implementation.

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::memory::GpuBuffer;
use std::sync::Arc;

/// CPU-based buffer that simulates GPU memory.
pub struct CpuBuffer {
    /// Buffer data.
    data: Arc<parking_lot::RwLock<Vec<u8>>>,
    /// Buffer size.
    size: usize,
}

impl CpuBuffer {
    /// Create a new CPU buffer.
    pub fn new(size: usize) -> Self {
        Self {
            data: Arc::new(parking_lot::RwLock::new(vec![0u8; size])),
            size,
        }
    }

    /// Create from existing data.
    pub fn from_data(data: Vec<u8>) -> Self {
        let size = data.len();
        Self {
            data: Arc::new(parking_lot::RwLock::new(data)),
            size,
        }
    }

    /// Get read access to buffer data.
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, Vec<u8>> {
        self.data.read()
    }

    /// Get write access to buffer data.
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, Vec<u8>> {
        self.data.write()
    }
}

impl GpuBuffer for CpuBuffer {
    fn size(&self) -> usize {
        self.size
    }

    fn device_ptr(&self) -> usize {
        // Return address of underlying data as "device pointer"
        self.data.read().as_ptr() as usize
    }

    fn copy_from_host(&self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(RingKernelError::TransferFailed(format!(
                "Source ({}) larger than buffer ({})",
                data.len(),
                self.size
            )));
        }

        let mut buf = self.data.write();
        buf[..data.len()].copy_from_slice(data);
        Ok(())
    }

    fn copy_to_host(&self, data: &mut [u8]) -> Result<()> {
        let buf = self.data.read();
        let len = data.len().min(buf.len());
        data[..len].copy_from_slice(&buf[..len]);
        Ok(())
    }
}

impl Clone for CpuBuffer {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            size: self.size,
        }
    }
}

/// CPU device memory allocator.
#[allow(dead_code)]
pub struct CpuDeviceMemory {
    /// Total simulated memory.
    total: usize,
    /// Current allocated memory.
    allocated: std::sync::atomic::AtomicUsize,
}

impl CpuDeviceMemory {
    /// Create a new CPU device memory allocator.
    #[allow(dead_code)]
    pub fn new(total_memory: usize) -> Self {
        Self {
            total: total_memory,
            allocated: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl ringkernel_core::memory::DeviceMemory for CpuDeviceMemory {
    fn allocate(&self, size: usize) -> Result<Box<dyn GpuBuffer>> {
        let current = self.allocated.load(std::sync::atomic::Ordering::Relaxed);
        if current + size > self.total {
            return Err(RingKernelError::OutOfMemory {
                requested: size,
                available: self.total - current,
            });
        }
        self.allocated
            .fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        Ok(Box::new(CpuBuffer::new(size)))
    }

    fn allocate_aligned(&self, size: usize, alignment: usize) -> Result<Box<dyn GpuBuffer>> {
        let aligned_size = ringkernel_core::memory::align::align_up(size, alignment);
        self.allocate(aligned_size)
    }

    fn total_memory(&self) -> usize {
        self.total
    }

    fn free_memory(&self) -> usize {
        let allocated = self.allocated.load(std::sync::atomic::Ordering::Relaxed);
        self.total.saturating_sub(allocated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_buffer() {
        let buffer = CpuBuffer::new(1024);
        assert_eq!(buffer.size(), 1024);

        let data = vec![1u8, 2, 3, 4, 5];
        buffer.copy_from_host(&data).unwrap();

        let mut result = vec![0u8; 5];
        buffer.copy_to_host(&mut result).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_cpu_device_memory() {
        use ringkernel_core::memory::DeviceMemory;

        let mem = CpuDeviceMemory::new(1024 * 1024); // 1MB
        assert_eq!(mem.total_memory(), 1024 * 1024);

        let buf = mem.allocate(1024).unwrap();
        assert_eq!(buf.size(), 1024);
        assert_eq!(mem.free_memory(), 1024 * 1024 - 1024);
    }
}
