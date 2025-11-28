//! Metal memory management.

#![cfg(all(target_os = "macos", feature = "metal"))]

use metal::{Buffer, Device, MTLResourceOptions};
use ringkernel_core::error::{Result, RingKernelError};

/// A Metal buffer for GPU memory.
pub struct MetalBuffer {
    /// The underlying Metal buffer.
    buffer: Buffer,
    /// Buffer size in bytes.
    size: usize,
}

impl MetalBuffer {
    /// Create a new Metal buffer.
    pub fn new(device: &Device, size: usize) -> Result<Self> {
        let options = MTLResourceOptions::StorageModeShared;

        let buffer = device.new_buffer(size as u64, options);

        Ok(Self { buffer, size })
    }

    /// Create a buffer with initial data.
    pub fn with_data(device: &Device, data: &[u8]) -> Result<Self> {
        let options = MTLResourceOptions::StorageModeShared;

        let buffer =
            device.new_buffer_with_data(data.as_ptr() as *const _, data.len() as u64, options);

        Ok(Self {
            buffer,
            size: data.len(),
        })
    }

    /// Get the buffer size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the underlying Metal buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get a mutable slice to the buffer contents.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.contents() as *mut u8, self.size) }
    }

    /// Get a slice to the buffer contents.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buffer.contents() as *const u8, self.size) }
    }

    /// Copy data to the buffer.
    pub fn copy_from_slice(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(RingKernelError::MemoryError(
                "Data exceeds buffer size".to_string(),
            ));
        }

        self.as_mut_slice()[..data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Copy data from the buffer.
    pub fn copy_to_slice(&self, dest: &mut [u8]) -> Result<()> {
        if dest.len() > self.size {
            return Err(RingKernelError::MemoryError(
                "Destination exceeds buffer size".to_string(),
            ));
        }

        dest.copy_from_slice(&self.as_slice()[..dest.len()]);
        Ok(())
    }
}
