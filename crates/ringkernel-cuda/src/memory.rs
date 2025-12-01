//! CUDA memory management for RingKernel.

use std::cell::UnsafeCell;

use cudarc::driver::{CudaSlice, DevicePtr};

use ringkernel_core::control::ControlBlock;
use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::memory::GpuBuffer;
use ringkernel_core::message::MessageHeader;

use crate::device::CudaDevice;

/// CUDA buffer implementing GpuBuffer trait.
///
/// Uses UnsafeCell for interior mutability since cudarc requires &mut for HtoD copies
/// but we need to implement the GpuBuffer trait which takes &self.
pub struct CudaBuffer {
    /// Device memory slice (wrapped in UnsafeCell for interior mutability).
    data: UnsafeCell<CudaSlice<u8>>,
    /// Size in bytes.
    size: usize,
    /// Parent device.
    device: CudaDevice,
}

// Safety: CudaBuffer is only accessed from one thread at a time
// (CUDA operations are inherently serialized on a device)
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// Create a new CUDA buffer.
    pub fn new(device: &CudaDevice, size: usize) -> Result<Self> {
        let data = device.alloc::<u8>(size)?;
        Ok(Self {
            data: UnsafeCell::new(data),
            size,
            device: device.clone(),
        })
    }

    /// Get the device pointer.
    pub fn device_ptr(&self) -> u64 {
        // Safety: we only read the device pointer, not mutate the slice
        unsafe { *(*self.data.get()).device_ptr() }
    }

    /// Get underlying slice for kernel launches.
    pub fn as_slice(&self) -> &CudaSlice<u8> {
        // Safety: caller ensures no concurrent mutation
        unsafe { &*self.data.get() }
    }

    /// Get mutable reference for uploads.
    pub fn as_mut_slice(&mut self) -> &mut CudaSlice<u8> {
        self.data.get_mut()
    }
}

impl GpuBuffer for CudaBuffer {
    fn size(&self) -> usize {
        self.size
    }

    fn device_ptr(&self) -> usize {
        // Safety: we only read the device pointer
        unsafe { *(*self.data.get()).device_ptr() as usize }
    }

    fn copy_from_host(&self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(RingKernelError::AllocationFailed {
                size: data.len(),
                reason: format!("buffer too small: {} > {}", data.len(), self.size),
            });
        }

        // Safety: UnsafeCell allows interior mutability, and CUDA operations
        // are serialized on the device, so concurrent access is not an issue.
        let slice = unsafe { &mut *self.data.get() };
        self.device
            .inner()
            .htod_sync_copy_into(data, slice)
            .map_err(|e| RingKernelError::TransferFailed(format!("HtoD copy failed: {}", e)))?;

        Ok(())
    }

    fn copy_to_host(&self, data: &mut [u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(RingKernelError::AllocationFailed {
                size: data.len(),
                reason: format!("buffer too small: {} > {}", data.len(), self.size),
            });
        }

        // Safety: we only read from the slice
        let slice = unsafe { &*self.data.get() };
        let host_data = self
            .device
            .inner()
            .dtoh_sync_copy(slice)
            .map_err(|e| RingKernelError::TransferFailed(format!("DtoH copy failed: {}", e)))?;

        data[..host_data.len()].copy_from_slice(&host_data);
        Ok(())
    }
}

/// CUDA control block allocation.
pub struct CudaControlBlock {
    /// Device buffer for control block.
    buffer: CudaBuffer,
    /// Device for operations (kept for potential future use).
    #[allow(dead_code)]
    device: CudaDevice,
}

impl CudaControlBlock {
    /// Allocate a new control block on device.
    pub fn new(device: &CudaDevice) -> Result<Self> {
        let buffer = CudaBuffer::new(device, std::mem::size_of::<ControlBlock>())?;

        // Initialize to zeros
        let cb = Self {
            buffer,
            device: device.clone(),
        };
        let zeros = vec![0u8; std::mem::size_of::<ControlBlock>()];
        cb.buffer.copy_from_host(&zeros)?;

        Ok(cb)
    }

    /// Get device pointer for kernel launch.
    pub fn device_ptr(&self) -> u64 {
        self.buffer.device_ptr()
    }

    /// Read control block from device.
    pub fn read(&self) -> Result<ControlBlock> {
        let mut data = [0u8; std::mem::size_of::<ControlBlock>()];
        self.buffer.copy_to_host(&mut data)?;

        // Safety: ControlBlock is POD and we've read the right size
        Ok(unsafe { std::ptr::read(data.as_ptr() as *const ControlBlock) })
    }

    /// Write control block to device.
    pub fn write(&self, cb: &ControlBlock) -> Result<()> {
        let data = unsafe {
            std::slice::from_raw_parts(
                cb as *const ControlBlock as *const u8,
                std::mem::size_of::<ControlBlock>(),
            )
        };
        self.buffer.copy_from_host(data)
    }

    /// Set the active flag.
    pub fn set_active(&self, active: bool) -> Result<()> {
        let mut cb = self.read()?;
        cb.is_active = if active { 1 } else { 0 };
        self.write(&cb)
    }

    /// Set the termination request flag.
    pub fn set_should_terminate(&self, terminate: bool) -> Result<()> {
        let mut cb = self.read()?;
        cb.should_terminate = if terminate { 1 } else { 0 };
        self.write(&cb)
    }
}

/// CUDA message queue buffer.
pub struct CudaMessageQueue {
    /// Device buffer for message headers.
    headers: CudaBuffer,
    /// Device buffer for message payloads.
    payloads: CudaBuffer,
    /// Queue capacity (number of messages).
    capacity: usize,
    /// Maximum payload size per message.
    max_payload_size: usize,
    /// Device reference (kept for potential future use).
    #[allow(dead_code)]
    device: CudaDevice,
}

impl CudaMessageQueue {
    /// Create a new message queue on device.
    pub fn new(device: &CudaDevice, capacity: usize, max_payload_size: usize) -> Result<Self> {
        let header_size = std::mem::size_of::<MessageHeader>() * capacity;
        let payload_size = max_payload_size * capacity;

        let headers = CudaBuffer::new(device, header_size)?;
        let payloads = CudaBuffer::new(device, payload_size)?;

        Ok(Self {
            headers,
            payloads,
            capacity,
            max_payload_size,
            device: device.clone(),
        })
    }

    /// Get headers device pointer.
    pub fn headers_ptr(&self) -> u64 {
        self.headers.device_ptr()
    }

    /// Get payloads device pointer.
    pub fn payloads_ptr(&self) -> u64 {
        self.payloads.device_ptr()
    }

    /// Get capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get max payload size.
    pub fn max_payload_size(&self) -> usize {
        self.max_payload_size
    }

    /// Total memory used.
    pub fn total_size(&self) -> usize {
        self.headers.size() + self.payloads.size()
    }
}

/// Memory pool for CUDA allocations.
pub struct CudaMemoryPool {
    /// Device reference.
    device: CudaDevice,
    /// Pre-allocated control blocks (reserved for pooling).
    #[allow(dead_code)]
    control_blocks: Vec<CudaControlBlock>,
    /// Pre-allocated queues (reserved for pooling).
    #[allow(dead_code)]
    queues: Vec<CudaMessageQueue>,
}

impl CudaMemoryPool {
    /// Create a new memory pool.
    pub fn new(device: CudaDevice) -> Self {
        Self {
            device,
            control_blocks: Vec::new(),
            queues: Vec::new(),
        }
    }

    /// Allocate a control block.
    pub fn alloc_control_block(&mut self) -> Result<CudaControlBlock> {
        CudaControlBlock::new(&self.device)
    }

    /// Allocate a message queue.
    pub fn alloc_queue(
        &mut self,
        capacity: usize,
        max_payload_size: usize,
    ) -> Result<CudaMessageQueue> {
        CudaMessageQueue::new(&self.device, capacity, max_payload_size)
    }

    /// Get device reference.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests require CUDA hardware - marked as ignore
    #[test]
    #[ignore]
    fn test_cuda_buffer() {
        let device = CudaDevice::new(0).unwrap();
        let buffer = CudaBuffer::new(&device, 1024).unwrap();

        // Write data
        let data = vec![42u8; 1024];
        buffer.copy_from_host(&data).unwrap();

        // Read back
        let mut read_data = vec![0u8; 1024];
        buffer.copy_to_host(&mut read_data).unwrap();

        assert_eq!(data, read_data);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_control_block() {
        let device = CudaDevice::new(0).unwrap();
        let cb = CudaControlBlock::new(&device).unwrap();

        // Initially not active
        let state = cb.read().unwrap();
        assert!(!state.is_active());

        // Set active
        cb.set_active(true).unwrap();
        let state = cb.read().unwrap();
        assert!(state.is_active());

        // Deactivate
        cb.set_active(false).unwrap();
        let state = cb.read().unwrap();
        assert!(!state.is_active());
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_control_block_termination() {
        let device = CudaDevice::new(0).unwrap();
        let cb = CudaControlBlock::new(&device).unwrap();

        // Initially not requesting termination
        let state = cb.read().unwrap();
        assert!(!state.should_terminate());

        // Request termination
        cb.set_should_terminate(true).unwrap();
        let state = cb.read().unwrap();
        assert!(state.should_terminate());
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_message_queue() {
        let device = CudaDevice::new(0).unwrap();
        let queue = CudaMessageQueue::new(&device, 64, 4096).unwrap();

        assert_eq!(queue.capacity(), 64);
        assert_eq!(queue.max_payload_size(), 4096);
        assert!(queue.headers_ptr() > 0);
        assert!(queue.payloads_ptr() > 0);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_memory_pool() {
        let device = CudaDevice::new(0).unwrap();
        let mut pool = CudaMemoryPool::new(device);

        // Allocate control block
        let cb = pool.alloc_control_block().unwrap();
        assert!(cb.device_ptr() > 0);

        // Allocate queue
        let queue = pool.alloc_queue(32, 2048).unwrap();
        assert_eq!(queue.capacity(), 32);
    }
}
