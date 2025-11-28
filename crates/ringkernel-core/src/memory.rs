//! GPU and host memory management abstractions.
//!
//! This module provides RAII wrappers for GPU memory, pinned host memory,
//! and memory pools for efficient allocation.

use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::{Result, RingKernelError};

/// Trait for GPU buffer operations.
pub trait GpuBuffer: Send + Sync {
    /// Get buffer size in bytes.
    fn size(&self) -> usize;

    /// Get device pointer (as usize for FFI compatibility).
    fn device_ptr(&self) -> usize;

    /// Copy data from host to device.
    fn copy_from_host(&self, data: &[u8]) -> Result<()>;

    /// Copy data from device to host.
    fn copy_to_host(&self, data: &mut [u8]) -> Result<()>;
}

/// Trait for device memory allocation.
pub trait DeviceMemory: Send + Sync {
    /// Allocate device memory.
    fn allocate(&self, size: usize) -> Result<Box<dyn GpuBuffer>>;

    /// Allocate device memory with alignment.
    fn allocate_aligned(&self, size: usize, alignment: usize) -> Result<Box<dyn GpuBuffer>>;

    /// Get total device memory.
    fn total_memory(&self) -> usize;

    /// Get free device memory.
    fn free_memory(&self) -> usize;
}

/// Pinned (page-locked) host memory for efficient DMA transfers.
///
/// Pinned memory allows direct DMA transfers between host and device
/// without intermediate copying, significantly improving transfer performance.
pub struct PinnedMemory<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
    _marker: PhantomData<T>,
}

impl<T: Copy> PinnedMemory<T> {
    /// Allocate pinned memory for `count` elements.
    ///
    /// # Safety
    ///
    /// The underlying memory is uninitialized. Caller must ensure
    /// data is initialized before reading.
    pub fn new(count: usize) -> Result<Self> {
        if count == 0 {
            return Err(RingKernelError::InvalidConfig(
                "Cannot allocate zero-sized buffer".to_string(),
            ));
        }

        let layout = Layout::array::<T>(count).map_err(|_| RingKernelError::HostAllocationFailed {
            size: count * std::mem::size_of::<T>(),
        })?;

        // In production, this would use platform-specific pinned allocation
        // (e.g., cuMemAllocHost for CUDA, or mlock for general case)
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            return Err(RingKernelError::HostAllocationFailed { size: layout.size() });
        }

        Ok(Self {
            ptr: NonNull::new(ptr as *mut T).unwrap(),
            len: count,
            layout,
            _marker: PhantomData,
        })
    }

    /// Create pinned memory from a slice, copying the data.
    pub fn from_slice(data: &[T]) -> Result<Self> {
        let mut mem = Self::new(data.len())?;
        mem.as_mut_slice().copy_from_slice(data);
        Ok(mem)
    }

    /// Get slice reference.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get mutable slice reference.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get raw pointer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T: Copy> Drop for PinnedMemory<T> {
    fn drop(&mut self) {
        // In production, this would use platform-specific deallocation
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

// SAFETY: PinnedMemory can be sent between threads
unsafe impl<T: Copy + Send> Send for PinnedMemory<T> {}
unsafe impl<T: Copy + Sync> Sync for PinnedMemory<T> {}

/// Memory pool for efficient allocation/deallocation.
///
/// Memory pools amortize allocation costs by maintaining a free list
/// of pre-allocated buffers.
pub struct MemoryPool {
    /// Pool name for debugging.
    name: String,
    /// Buffer size for this pool.
    buffer_size: usize,
    /// Maximum number of buffers to pool.
    max_buffers: usize,
    /// Free list of buffers.
    free_list: Mutex<Vec<Vec<u8>>>,
    /// Statistics: total allocations.
    total_allocations: AtomicUsize,
    /// Statistics: cache hits.
    cache_hits: AtomicUsize,
    /// Statistics: current pool size.
    pool_size: AtomicUsize,
}

impl MemoryPool {
    /// Create a new memory pool.
    pub fn new(name: impl Into<String>, buffer_size: usize, max_buffers: usize) -> Self {
        Self {
            name: name.into(),
            buffer_size,
            max_buffers,
            free_list: Mutex::new(Vec::with_capacity(max_buffers)),
            total_allocations: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            pool_size: AtomicUsize::new(0),
        }
    }

    /// Allocate a buffer from the pool.
    pub fn allocate(&self) -> PooledBuffer<'_> {
        self.total_allocations.fetch_add(1, Ordering::Relaxed);

        let buffer = {
            let mut free = self.free_list.lock();
            if let Some(buf) = free.pop() {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                self.pool_size.fetch_sub(1, Ordering::Relaxed);
                buf
            } else {
                vec![0u8; self.buffer_size]
            }
        };

        PooledBuffer {
            buffer: Some(buffer),
            pool: self,
        }
    }

    /// Return a buffer to the pool.
    fn return_buffer(&self, mut buffer: Vec<u8>) {
        let mut free = self.free_list.lock();
        if free.len() < self.max_buffers {
            buffer.clear();
            buffer.resize(self.buffer_size, 0);
            free.push(buffer);
            self.pool_size.fetch_add(1, Ordering::Relaxed);
        }
        // If pool is full, buffer is dropped
    }

    /// Get pool name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get current pool size.
    pub fn current_size(&self) -> usize {
        self.pool_size.load(Ordering::Relaxed)
    }

    /// Get cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_allocations.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Pre-allocate buffers.
    pub fn preallocate(&self, count: usize) {
        let count = count.min(self.max_buffers);
        let mut free = self.free_list.lock();
        for _ in free.len()..count {
            free.push(vec![0u8; self.buffer_size]);
            self.pool_size.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A buffer from a memory pool.
///
/// When dropped, the buffer is returned to the pool for reuse.
pub struct PooledBuffer<'a> {
    buffer: Option<Vec<u8>>,
    pool: &'a MemoryPool,
}

impl<'a> PooledBuffer<'a> {
    /// Get slice reference.
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_ref().map(|b| b.as_slice()).unwrap_or(&[])
    }

    /// Get mutable slice reference.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer
            .as_mut()
            .map(|b| b.as_mut_slice())
            .unwrap_or(&mut [])
    }

    /// Get buffer length.
    pub fn len(&self) -> usize {
        self.buffer.as_ref().map(|b| b.len()).unwrap_or(0)
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a> Drop for PooledBuffer<'a> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_buffer(buffer);
        }
    }
}

impl<'a> std::ops::Deref for PooledBuffer<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a> std::ops::DerefMut for PooledBuffer<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Shared memory pool that can be cloned.
pub type SharedMemoryPool = Arc<MemoryPool>;

/// Create a shared memory pool.
pub fn create_pool(name: impl Into<String>, buffer_size: usize, max_buffers: usize) -> SharedMemoryPool {
    Arc::new(MemoryPool::new(name, buffer_size, max_buffers))
}

/// Alignment utilities.
pub mod align {
    /// Cache line size (64 bytes on most modern CPUs).
    pub const CACHE_LINE_SIZE: usize = 64;

    /// GPU cache line size (128 bytes on many GPUs).
    pub const GPU_CACHE_LINE_SIZE: usize = 128;

    /// Align a value up to the next multiple of alignment.
    #[inline]
    pub const fn align_up(value: usize, alignment: usize) -> usize {
        let mask = alignment - 1;
        (value + mask) & !mask
    }

    /// Align a value down to the previous multiple of alignment.
    #[inline]
    pub const fn align_down(value: usize, alignment: usize) -> usize {
        let mask = alignment - 1;
        value & !mask
    }

    /// Check if a value is aligned.
    #[inline]
    pub const fn is_aligned(value: usize, alignment: usize) -> bool {
        value & (alignment - 1) == 0
    }

    /// Get required padding for alignment.
    #[inline]
    pub const fn padding_for(offset: usize, alignment: usize) -> usize {
        let misalignment = offset & (alignment - 1);
        if misalignment == 0 {
            0
        } else {
            alignment - misalignment
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinned_memory() {
        let mut mem = PinnedMemory::<f32>::new(1024).unwrap();
        assert_eq!(mem.len(), 1024);
        assert_eq!(mem.size_bytes(), 1024 * 4);

        // Write some data
        let slice = mem.as_mut_slice();
        for (i, v) in slice.iter_mut().enumerate() {
            *v = i as f32;
        }

        // Read back
        assert_eq!(mem.as_slice()[42], 42.0);
    }

    #[test]
    fn test_pinned_memory_from_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mem = PinnedMemory::from_slice(&data).unwrap();
        assert_eq!(mem.as_slice(), &data[..]);
    }

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new("test", 1024, 10);

        // First allocation should be fresh
        let buf1 = pool.allocate();
        assert_eq!(buf1.len(), 1024);
        drop(buf1);

        // Second allocation should be cached
        let _buf2 = pool.allocate();
        assert_eq!(pool.hit_rate(), 0.5); // 1 hit out of 2 allocations
    }

    #[test]
    fn test_pool_preallocate() {
        let pool = MemoryPool::new("test", 1024, 10);
        pool.preallocate(5);
        assert_eq!(pool.current_size(), 5);

        // All allocations should hit cache
        for _ in 0..5 {
            let _ = pool.allocate();
        }
        assert_eq!(pool.hit_rate(), 1.0);
    }

    #[test]
    fn test_align_up() {
        use align::*;

        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(64, 64), 64);
        assert_eq!(align_up(65, 64), 128);
    }

    #[test]
    fn test_is_aligned() {
        use align::*;

        assert!(is_aligned(0, 64));
        assert!(is_aligned(64, 64));
        assert!(is_aligned(128, 64));
        assert!(!is_aligned(1, 64));
        assert!(!is_aligned(63, 64));
    }

    #[test]
    fn test_padding_for() {
        use align::*;

        assert_eq!(padding_for(0, 64), 0);
        assert_eq!(padding_for(1, 64), 63);
        assert_eq!(padding_for(63, 64), 1);
        assert_eq!(padding_for(64, 64), 0);
    }
}
