//! WebGPU memory management.

use std::sync::Arc;

use ringkernel_core::control::ControlBlock;
use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::memory::GpuBuffer;
use ringkernel_core::message::MessageHeader;

use crate::adapter::WgpuAdapter;

/// WebGPU buffer implementing GpuBuffer trait.
pub struct WgpuBuffer {
    /// The wgpu buffer.
    buffer: wgpu::Buffer,
    /// Size in bytes.
    size: usize,
    /// Reference to device.
    device: Arc<wgpu::Device>,
    /// Reference to queue.
    queue: Arc<wgpu::Queue>,
}

impl WgpuBuffer {
    /// Create a new storage buffer.
    pub fn new(adapter: &WgpuAdapter, size: usize, label: Option<&str>) -> Self {
        let buffer = adapter.device().create_buffer(&wgpu::BufferDescriptor {
            label,
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            size,
            device: Arc::clone(adapter.device()),
            queue: Arc::clone(adapter.queue()),
        }
    }

    /// Create a buffer with initial data.
    pub fn new_init(adapter: &WgpuAdapter, data: &[u8], label: Option<&str>) -> Self {
        let buffer = adapter.device().create_buffer(&wgpu::BufferDescriptor {
            label,
            size: data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        // Copy data
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(data);
        buffer.unmap();

        Self {
            buffer,
            size: data.len(),
            device: Arc::clone(adapter.device()),
            queue: Arc::clone(adapter.queue()),
        }
    }

    /// Get the underlying wgpu buffer.
    #[allow(dead_code)]
    pub fn inner(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Create a binding for this buffer.
    pub fn as_entire_binding(&self) -> wgpu::BindingResource<'_> {
        self.buffer.as_entire_binding()
    }
}

impl GpuBuffer for WgpuBuffer {
    fn size(&self) -> usize {
        self.size
    }

    fn device_ptr(&self) -> usize {
        // WebGPU doesn't expose raw device pointers
        // Return 0 as a placeholder - actual GPU address is managed internally by wgpu
        0
    }

    fn copy_from_host(&self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(RingKernelError::AllocationFailed {
                size: data.len(),
                reason: format!("buffer too small: {} > {}", data.len(), self.size),
            });
        }

        self.queue.write_buffer(&self.buffer, 0, data);
        Ok(())
    }

    fn copy_to_host(&self, data: &mut [u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(RingKernelError::AllocationFailed {
                size: data.len(),
                reason: format!("buffer too small: {} > {}", data.len(), self.size),
            });
        }

        // Create a staging buffer for reading
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: self.size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Encoder"),
            });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.size as u64);
        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| RingKernelError::TransferFailed(format!("Channel error: {}", e)))?
            .map_err(|e| RingKernelError::TransferFailed(format!("Map error: {}", e)))?;

        let mapped = slice.get_mapped_range();
        let copy_len = self.size.min(data.len());
        data[..copy_len].copy_from_slice(&mapped[..copy_len]);
        drop(mapped);
        staging.unmap();

        Ok(())
    }
}

/// WebGPU control block.
pub struct WgpuControlBlock {
    /// Buffer for control block data.
    buffer: WgpuBuffer,
}

impl WgpuControlBlock {
    /// Create a new control block.
    pub fn new(adapter: &WgpuAdapter) -> Self {
        let size = std::mem::size_of::<ControlBlock>();
        let data = vec![0u8; size];
        let buffer = WgpuBuffer::new_init(adapter, &data, Some("Control Block"));

        Self { buffer }
    }

    /// Get the binding resource.
    pub fn as_binding(&self) -> wgpu::BindingResource<'_> {
        self.buffer.as_entire_binding()
    }

    /// Get the underlying buffer.
    #[allow(dead_code)]
    pub fn buffer(&self) -> &WgpuBuffer {
        &self.buffer
    }

    /// Read control block from GPU.
    pub fn read(&self) -> Result<ControlBlock> {
        let mut data = vec![0u8; std::mem::size_of::<ControlBlock>()];
        self.buffer.copy_to_host(&mut data)?;

        // Safety: ControlBlock is POD
        Ok(unsafe { std::ptr::read(data.as_ptr() as *const ControlBlock) })
    }

    /// Write control block to GPU.
    pub fn write(&self, cb: &ControlBlock) -> Result<()> {
        let data = unsafe {
            std::slice::from_raw_parts(
                cb as *const ControlBlock as *const u8,
                std::mem::size_of::<ControlBlock>(),
            )
        };
        self.buffer.copy_from_host(data)
    }
}

/// WebGPU message queue.
pub struct WgpuMessageQueue {
    /// Headers buffer.
    headers: WgpuBuffer,
    /// Payloads buffer.
    payloads: WgpuBuffer,
    /// Queue capacity.
    capacity: usize,
    /// Max payload size.
    max_payload_size: usize,
    /// Current head index (for reading).
    head: std::sync::atomic::AtomicU32,
    /// Current tail index (for writing).
    tail: std::sync::atomic::AtomicU32,
}

impl WgpuMessageQueue {
    /// Create a new message queue.
    pub fn new(adapter: &WgpuAdapter, capacity: usize, max_payload_size: usize) -> Self {
        let header_size = std::mem::size_of::<MessageHeader>() * capacity;
        let payload_size = max_payload_size * capacity;

        Self {
            headers: WgpuBuffer::new(adapter, header_size, Some("Message Headers")),
            payloads: WgpuBuffer::new(adapter, payload_size, Some("Message Payloads")),
            capacity,
            max_payload_size,
            head: std::sync::atomic::AtomicU32::new(0),
            tail: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Get headers binding.
    pub fn headers_binding(&self) -> wgpu::BindingResource<'_> {
        self.headers.as_entire_binding()
    }

    /// Get payloads binding.
    #[allow(dead_code)]
    pub fn payloads_binding(&self) -> wgpu::BindingResource<'_> {
        self.payloads.as_entire_binding()
    }

    /// Get queue capacity.
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current queue size.
    pub fn len(&self) -> usize {
        let head = self.head.load(std::sync::atomic::Ordering::Acquire);
        let tail = self.tail.load(std::sync::atomic::Ordering::Acquire);
        if tail >= head {
            (tail - head) as usize
        } else {
            self.capacity - (head - tail) as usize
        }
    }

    /// Check if queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if queue is full.
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity - 1
    }

    /// Enqueue a message envelope.
    ///
    /// This writes the header and payload to GPU buffers.
    pub fn enqueue(&self, envelope: &ringkernel_core::message::MessageEnvelope) -> Result<()> {
        if self.is_full() {
            return Err(RingKernelError::QueueFull {
                capacity: self.capacity,
            });
        }

        let tail = self.tail.load(std::sync::atomic::Ordering::Acquire) as usize;

        // Write header at the appropriate offset
        let header_offset = tail * std::mem::size_of::<MessageHeader>();
        let header_bytes = envelope.header.as_bytes();

        // Create a write buffer for header
        let mut header_data = vec![0u8; std::mem::size_of::<MessageHeader>()];
        header_data[..header_bytes.len()].copy_from_slice(header_bytes);

        // Write to GPU - we need to write to the specific offset
        // For now, we write the entire header region (this could be optimized)
        let mut all_headers = vec![0u8; self.headers.size()];
        self.headers.copy_to_host(&mut all_headers)?;
        all_headers[header_offset..header_offset + header_data.len()].copy_from_slice(&header_data);
        self.headers.copy_from_host(&all_headers)?;

        // Write payload at the appropriate offset
        if !envelope.payload.is_empty() {
            let payload_offset = tail * self.max_payload_size;
            let payload_len = envelope.payload.len().min(self.max_payload_size);

            let mut all_payloads = vec![0u8; self.payloads.size()];
            self.payloads.copy_to_host(&mut all_payloads)?;
            all_payloads[payload_offset..payload_offset + payload_len]
                .copy_from_slice(&envelope.payload[..payload_len]);
            self.payloads.copy_from_host(&all_payloads)?;
        }

        // Update tail index
        let new_tail = ((tail + 1) % self.capacity) as u32;
        self.tail
            .store(new_tail, std::sync::atomic::Ordering::Release);

        Ok(())
    }

    /// Dequeue a message envelope.
    ///
    /// This reads the header and payload from GPU buffers.
    pub fn dequeue(&self) -> Result<ringkernel_core::message::MessageEnvelope> {
        if self.is_empty() {
            return Err(RingKernelError::QueueEmpty);
        }

        let head = self.head.load(std::sync::atomic::Ordering::Acquire) as usize;

        // Read header at the appropriate offset
        let header_offset = head * std::mem::size_of::<MessageHeader>();
        let mut all_headers = vec![0u8; self.headers.size()];
        self.headers.copy_to_host(&mut all_headers)?;

        let header_bytes =
            &all_headers[header_offset..header_offset + std::mem::size_of::<MessageHeader>()];
        let header = MessageHeader::read_from(header_bytes).ok_or_else(|| {
            RingKernelError::DeserializationError("Failed to read header".to_string())
        })?;

        // Read payload
        let payload_offset = head * self.max_payload_size;
        let payload_size = (header.payload_size as usize).min(self.max_payload_size);

        let payload = if payload_size > 0 {
            let mut all_payloads = vec![0u8; self.payloads.size()];
            self.payloads.copy_to_host(&mut all_payloads)?;
            all_payloads[payload_offset..payload_offset + payload_size].to_vec()
        } else {
            Vec::new()
        };

        // Update head index
        let new_head = ((head + 1) % self.capacity) as u32;
        self.head
            .store(new_head, std::sync::atomic::Ordering::Release);

        Ok(ringkernel_core::message::MessageEnvelope { header, payload })
    }

    /// Try to dequeue without blocking (returns None if empty).
    pub fn try_dequeue(&self) -> Option<ringkernel_core::message::MessageEnvelope> {
        if self.is_empty() {
            None
        } else {
            self.dequeue().ok()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // May not have GPU in CI
    async fn test_wgpu_buffer() {
        let adapter = WgpuAdapter::new().await.unwrap();
        let buffer = WgpuBuffer::new(&adapter, 1024, Some("Test Buffer"));

        // Write data
        let data = vec![42u8; 1024];
        buffer.copy_from_host(&data).unwrap();

        // Read back
        let mut read_data = vec![0u8; 1024];
        buffer.copy_to_host(&mut read_data).unwrap();

        assert_eq!(data, read_data);
    }
}
