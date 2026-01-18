//! WebGPU memory management.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use ringkernel_core::control::ControlBlock;
use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::memory::GpuBuffer;
use ringkernel_core::message::MessageHeader;

use crate::adapter::WgpuAdapter;

// ============================================================================
// Staging Buffer Pool
// ============================================================================

/// Statistics for staging buffer pool.
#[derive(Debug, Clone, Default)]
pub struct StagingPoolStats {
    /// Total acquire calls.
    pub total_acquires: u64,
    /// Cache hits (buffer reused).
    pub cache_hits: u64,
    /// New allocations.
    pub allocations: u64,
    /// Buffers returned to pool.
    pub returns: u64,
    /// Buffers trimmed (evicted).
    pub trimmed: u64,
}

impl StagingPoolStats {
    /// Get hit rate.
    pub fn hit_rate(&self) -> f64 {
        if self.total_acquires == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_acquires as f64
        }
    }
}

/// Entry in the staging buffer pool.
struct StagingEntry {
    /// The staging buffer.
    buffer: wgpu::Buffer,
    /// Size in bytes.
    size: usize,
    /// Last time this buffer was used.
    last_used: Instant,
}

/// Pool of staging buffers for efficient GPU-to-host transfers.
///
/// Instead of creating a new staging buffer for every `copy_to_host` call,
/// this pool maintains reusable buffers that are returned after use.
///
/// # Example
///
/// ```ignore
/// let pool = StagingBufferPool::new(device.clone(), 8);
///
/// // Acquire a staging buffer
/// let guard = pool.acquire(1024)?;
///
/// // Use guard.buffer() for the copy operation
/// encoder.copy_buffer_to_buffer(src, 0, guard.buffer(), 0, size);
///
/// // Buffer automatically returned to pool when guard drops
/// ```
pub struct StagingBufferPool {
    device: Arc<wgpu::Device>,
    buffers: Mutex<Vec<StagingEntry>>,
    max_cached: usize,
    stats: Mutex<StagingPoolStats>,
}

impl StagingBufferPool {
    /// Create a new staging buffer pool.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `max_cached` - Maximum number of buffers to cache
    pub fn new(device: Arc<wgpu::Device>, max_cached: usize) -> Self {
        Self {
            device,
            buffers: Mutex::new(Vec::with_capacity(max_cached)),
            max_cached,
            stats: Mutex::new(StagingPoolStats::default()),
        }
    }

    /// Acquire a staging buffer of at least the requested size.
    ///
    /// Returns a guard that automatically returns the buffer to the pool on drop.
    pub fn acquire(&self, min_size: usize) -> StagingBufferGuard<'_> {
        let mut stats = self.stats.lock();
        stats.total_acquires += 1;

        // Try to find a suitable buffer in the pool
        let mut buffers = self.buffers.lock();

        // Find smallest buffer >= min_size, preferring exact matches
        let mut best_idx = None;
        let mut best_size = usize::MAX;

        for (idx, entry) in buffers.iter().enumerate() {
            if entry.size >= min_size && entry.size < best_size {
                best_size = entry.size;
                best_idx = Some(idx);
                // Prefer exact match
                if entry.size == min_size {
                    break;
                }
            }
        }

        if let Some(idx) = best_idx {
            stats.cache_hits += 1;
            let entry = buffers.remove(idx);
            drop(buffers);
            drop(stats);

            return StagingBufferGuard {
                buffer: Some(entry.buffer),
                size: entry.size,
                pool: self,
            };
        }

        // No suitable buffer found, allocate new one
        stats.allocations += 1;
        drop(buffers);
        drop(stats);

        // Round up to power of 2 for better reuse
        let actual_size = min_size.next_power_of_two().max(4096);

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer (Pooled)"),
            size: actual_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        StagingBufferGuard {
            buffer: Some(buffer),
            size: actual_size,
            pool: self,
        }
    }

    /// Return a buffer to the pool.
    fn return_buffer(&self, buffer: wgpu::Buffer, size: usize) {
        let mut stats = self.stats.lock();
        let mut buffers = self.buffers.lock();

        if buffers.len() < self.max_cached {
            stats.returns += 1;
            buffers.push(StagingEntry {
                buffer,
                size,
                last_used: Instant::now(),
            });
        }
        // If pool is full, buffer is dropped
    }

    /// Trim buffers older than the specified duration.
    pub fn trim(&self, max_age: Duration) {
        let mut stats = self.stats.lock();
        let mut buffers = self.buffers.lock();
        let now = Instant::now();

        let before_len = buffers.len();
        buffers.retain(|entry| now.duration_since(entry.last_used) < max_age);
        let trimmed = before_len - buffers.len();
        stats.trimmed += trimmed as u64;
    }

    /// Clear all cached buffers.
    pub fn clear(&self) {
        let mut buffers = self.buffers.lock();
        buffers.clear();
    }

    /// Get the number of currently cached buffers.
    pub fn cached_count(&self) -> usize {
        self.buffers.lock().len()
    }

    /// Get statistics snapshot.
    pub fn stats(&self) -> StagingPoolStats {
        self.stats.lock().clone()
    }
}

/// RAII guard for a staging buffer from the pool.
///
/// The buffer is automatically returned to the pool when dropped.
pub struct StagingBufferGuard<'a> {
    buffer: Option<wgpu::Buffer>,
    size: usize,
    pool: &'a StagingBufferPool,
}

impl<'a> StagingBufferGuard<'a> {
    /// Get the underlying wgpu buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("buffer should exist")
    }

    /// Get the buffer size.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl<'a> Drop for StagingBufferGuard<'a> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_buffer(buffer, self.size);
        }
    }
}

/// Shared staging buffer pool.
pub type SharedStagingPool = Arc<StagingBufferPool>;

/// Create a shared staging buffer pool.
pub fn create_staging_pool(device: Arc<wgpu::Device>, max_cached: usize) -> SharedStagingPool {
    Arc::new(StagingBufferPool::new(device, max_cached))
}

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
    /// Optional staging buffer pool for efficient reads.
    staging_pool: Option<SharedStagingPool>,
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
            staging_pool: None,
        }
    }

    /// Create a new storage buffer with a staging pool for efficient reads.
    pub fn new_with_pool(
        adapter: &WgpuAdapter,
        size: usize,
        label: Option<&str>,
        staging_pool: SharedStagingPool,
    ) -> Self {
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
            staging_pool: Some(staging_pool),
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
            staging_pool: None,
        }
    }

    /// Create a buffer with initial data and a staging pool.
    pub fn new_init_with_pool(
        adapter: &WgpuAdapter,
        data: &[u8],
        label: Option<&str>,
        staging_pool: SharedStagingPool,
    ) -> Self {
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
            staging_pool: Some(staging_pool),
        }
    }

    /// Set the staging pool for this buffer.
    pub fn set_staging_pool(&mut self, pool: SharedStagingPool) {
        self.staging_pool = Some(pool);
    }

    /// Get the staging pool if set.
    pub fn staging_pool(&self) -> Option<&SharedStagingPool> {
        self.staging_pool.as_ref()
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

        // Use pooled staging buffer if available, otherwise create one
        if let Some(pool) = &self.staging_pool {
            self.copy_to_host_pooled(data, pool)
        } else {
            self.copy_to_host_unpooled(data)
        }
    }
}

impl WgpuBuffer {
    /// Copy to host using a pooled staging buffer.
    fn copy_to_host_pooled(&self, data: &mut [u8], pool: &StagingBufferPool) -> Result<()> {
        // Acquire a staging buffer from the pool
        let staging_guard = pool.acquire(self.size);

        // Copy from GPU buffer to staging
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Encoder (Pooled)"),
            });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, staging_guard.buffer(), 0, self.size as u64);
        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let slice = staging_guard.buffer().slice(..);
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
        staging_guard.buffer().unmap();

        // staging_guard drops here and returns buffer to pool

        Ok(())
    }

    /// Copy to host without pooling (creates a new staging buffer).
    fn copy_to_host_unpooled(&self, data: &mut [u8]) -> Result<()> {
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
