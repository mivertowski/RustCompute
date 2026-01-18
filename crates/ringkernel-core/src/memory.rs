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

        let layout =
            Layout::array::<T>(count).map_err(|_| RingKernelError::HostAllocationFailed {
                size: count * std::mem::size_of::<T>(),
            })?;

        // In production, this would use platform-specific pinned allocation
        // (e.g., cuMemAllocHost for CUDA, or mlock for general case)
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            return Err(RingKernelError::HostAllocationFailed {
                size: layout.size(),
            });
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
        self.buffer.as_deref().unwrap_or(&[])
    }

    /// Get mutable slice reference.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_deref_mut().unwrap_or(&mut [])
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
pub fn create_pool(
    name: impl Into<String>,
    buffer_size: usize,
    max_buffers: usize,
) -> SharedMemoryPool {
    Arc::new(MemoryPool::new(name, buffer_size, max_buffers))
}

// ============================================================================
// Size-Stratified Memory Pool
// ============================================================================

/// Size bucket for stratified pooling.
///
/// Provides predefined size classes for efficient multi-size pooling.
/// Allocations are rounded up to the smallest bucket that fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SizeBucket {
    /// Tiny buffers (256 bytes) - metadata, small messages.
    Tiny,
    /// Small buffers (1 KB) - typical message payloads.
    Small,
    /// Medium buffers (4 KB) - page-sized allocations.
    #[default]
    Medium,
    /// Large buffers (16 KB) - batch operations.
    Large,
    /// Huge buffers (64 KB) - large transfers.
    Huge,
}

impl SizeBucket {
    /// All bucket variants in order from smallest to largest.
    pub const ALL: [SizeBucket; 5] = [
        SizeBucket::Tiny,
        SizeBucket::Small,
        SizeBucket::Medium,
        SizeBucket::Large,
        SizeBucket::Huge,
    ];

    /// Get the size in bytes for this bucket.
    pub fn size(&self) -> usize {
        match self {
            Self::Tiny => 256,
            Self::Small => 1024,
            Self::Medium => 4096,
            Self::Large => 16384,
            Self::Huge => 65536,
        }
    }

    /// Find the smallest bucket that fits the requested size.
    ///
    /// Returns `Huge` for any size larger than 16KB.
    pub fn for_size(requested: usize) -> Self {
        if requested <= 256 {
            Self::Tiny
        } else if requested <= 1024 {
            Self::Small
        } else if requested <= 4096 {
            Self::Medium
        } else if requested <= 16384 {
            Self::Large
        } else {
            Self::Huge
        }
    }

    /// Get the next larger bucket, or self if already at largest.
    pub fn upgrade(&self) -> Self {
        match self {
            Self::Tiny => Self::Small,
            Self::Small => Self::Medium,
            Self::Medium => Self::Large,
            Self::Large => Self::Huge,
            Self::Huge => Self::Huge,
        }
    }

    /// Get the next smaller bucket, or self if already at smallest.
    pub fn downgrade(&self) -> Self {
        match self {
            Self::Tiny => Self::Tiny,
            Self::Small => Self::Tiny,
            Self::Medium => Self::Small,
            Self::Large => Self::Medium,
            Self::Huge => Self::Large,
        }
    }
}

impl std::fmt::Display for SizeBucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tiny => write!(f, "Tiny(256B)"),
            Self::Small => write!(f, "Small(1KB)"),
            Self::Medium => write!(f, "Medium(4KB)"),
            Self::Large => write!(f, "Large(16KB)"),
            Self::Huge => write!(f, "Huge(64KB)"),
        }
    }
}

/// Statistics for a stratified memory pool.
#[derive(Debug, Clone, Default)]
pub struct StratifiedPoolStats {
    /// Total allocations across all buckets.
    pub total_allocations: usize,
    /// Total cache hits across all buckets.
    pub total_hits: usize,
    /// Allocations per bucket.
    pub allocations_per_bucket: std::collections::HashMap<SizeBucket, usize>,
    /// Hits per bucket.
    pub hits_per_bucket: std::collections::HashMap<SizeBucket, usize>,
}

impl StratifiedPoolStats {
    /// Calculate overall hit rate.
    pub fn hit_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.total_hits as f64 / self.total_allocations as f64
        }
    }

    /// Get hit rate for a specific bucket.
    pub fn bucket_hit_rate(&self, bucket: SizeBucket) -> f64 {
        let allocs = self
            .allocations_per_bucket
            .get(&bucket)
            .copied()
            .unwrap_or(0);
        let hits = self.hits_per_bucket.get(&bucket).copied().unwrap_or(0);
        if allocs == 0 {
            0.0
        } else {
            hits as f64 / allocs as f64
        }
    }
}

/// Multi-size memory pool with automatic bucket selection.
///
/// Instead of having a single buffer size, this pool maintains separate
/// pools for different size classes. Allocations are rounded up to the
/// smallest bucket that fits.
///
/// # Example
///
/// ```ignore
/// use ringkernel_core::memory::{StratifiedMemoryPool, SizeBucket};
///
/// let pool = StratifiedMemoryPool::new("my_pool");
///
/// // Allocate various sizes - each goes to appropriate bucket
/// let tiny_buf = pool.allocate(100);   // Uses Tiny bucket (256B)
/// let medium_buf = pool.allocate(2000); // Uses Medium bucket (4KB)
///
/// // Check statistics
/// let stats = pool.stats();
/// println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);
/// ```
pub struct StratifiedMemoryPool {
    name: String,
    buckets: std::collections::HashMap<SizeBucket, MemoryPool>,
    max_buffers_per_bucket: usize,
    stats: Mutex<StratifiedPoolStats>,
}

impl StratifiedMemoryPool {
    /// Create a new stratified pool with default settings.
    ///
    /// Creates pools for all bucket sizes with 16 buffers per bucket.
    pub fn new(name: impl Into<String>) -> Self {
        Self::with_capacity(name, 16)
    }

    /// Create a pool with specified max buffers per bucket.
    pub fn with_capacity(name: impl Into<String>, max_buffers_per_bucket: usize) -> Self {
        let name = name.into();
        let mut buckets = std::collections::HashMap::new();

        for bucket in SizeBucket::ALL {
            let pool_name = format!("{}_{}", name, bucket);
            buckets.insert(
                bucket,
                MemoryPool::new(pool_name, bucket.size(), max_buffers_per_bucket),
            );
        }

        Self {
            name,
            buckets,
            max_buffers_per_bucket,
            stats: Mutex::new(StratifiedPoolStats::default()),
        }
    }

    /// Allocate a buffer of at least the requested size.
    ///
    /// The buffer may be larger than requested (rounded up to bucket size).
    pub fn allocate(&self, size: usize) -> StratifiedBuffer<'_> {
        let bucket = SizeBucket::for_size(size);
        self.allocate_bucket(bucket)
    }

    /// Allocate from a specific bucket.
    pub fn allocate_bucket(&self, bucket: SizeBucket) -> StratifiedBuffer<'_> {
        let pool = self.buckets.get(&bucket).expect("bucket pool exists");

        // Track stats before allocation to capture hit
        let was_cached = pool.current_size() > 0;
        let buffer = pool.allocate();

        // Update stats
        {
            let mut stats = self.stats.lock();
            stats.total_allocations += 1;
            *stats.allocations_per_bucket.entry(bucket).or_insert(0) += 1;
            if was_cached {
                stats.total_hits += 1;
                *stats.hits_per_bucket.entry(bucket).or_insert(0) += 1;
            }
        }

        StratifiedBuffer {
            inner: buffer,
            bucket,
            pool: self,
        }
    }

    /// Get pool name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get max buffers per bucket.
    pub fn max_buffers_per_bucket(&self) -> usize {
        self.max_buffers_per_bucket
    }

    /// Get current size of a specific bucket pool.
    pub fn bucket_size(&self, bucket: SizeBucket) -> usize {
        self.buckets
            .get(&bucket)
            .map(|p| p.current_size())
            .unwrap_or(0)
    }

    /// Get total buffers currently pooled across all buckets.
    pub fn total_pooled(&self) -> usize {
        self.buckets.values().map(|p| p.current_size()).sum()
    }

    /// Get statistics snapshot.
    pub fn stats(&self) -> StratifiedPoolStats {
        self.stats.lock().clone()
    }

    /// Pre-allocate buffers for a specific bucket.
    pub fn preallocate(&self, bucket: SizeBucket, count: usize) {
        if let Some(pool) = self.buckets.get(&bucket) {
            pool.preallocate(count);
        }
    }

    /// Pre-allocate buffers for all buckets.
    pub fn preallocate_all(&self, count_per_bucket: usize) {
        for bucket in SizeBucket::ALL {
            self.preallocate(bucket, count_per_bucket);
        }
    }

    /// Shrink all pools to target utilization.
    ///
    /// Removes excess pooled buffers to free memory.
    pub fn shrink_to(&self, target_per_bucket: usize) {
        for pool in self.buckets.values() {
            let mut free_list = pool.free_list.lock();
            while free_list.len() > target_per_bucket {
                free_list.pop();
                pool.pool_size.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

/// A buffer from a stratified memory pool.
///
/// Tracks which bucket it came from for proper return.
pub struct StratifiedBuffer<'a> {
    inner: PooledBuffer<'a>,
    bucket: SizeBucket,
    #[allow(dead_code)]
    pool: &'a StratifiedMemoryPool,
}

impl<'a> StratifiedBuffer<'a> {
    /// Get the size bucket this buffer was allocated from.
    pub fn bucket(&self) -> SizeBucket {
        self.bucket
    }

    /// Get the actual capacity (bucket size, may be larger than requested).
    pub fn capacity(&self) -> usize {
        self.bucket.size()
    }

    /// Get slice reference.
    pub fn as_slice(&self) -> &[u8] {
        self.inner.as_slice()
    }

    /// Get mutable slice reference.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.inner.as_mut_slice()
    }

    /// Get buffer length.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<'a> std::ops::Deref for StratifiedBuffer<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a> std::ops::DerefMut for StratifiedBuffer<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Shared stratified memory pool.
pub type SharedStratifiedPool = Arc<StratifiedMemoryPool>;

/// Create a shared stratified memory pool.
pub fn create_stratified_pool(name: impl Into<String>) -> SharedStratifiedPool {
    Arc::new(StratifiedMemoryPool::new(name))
}

/// Create a shared stratified memory pool with custom capacity.
pub fn create_stratified_pool_with_capacity(
    name: impl Into<String>,
    max_buffers_per_bucket: usize,
) -> SharedStratifiedPool {
    Arc::new(StratifiedMemoryPool::with_capacity(
        name,
        max_buffers_per_bucket,
    ))
}

// ============================================================================
// Memory Pressure Reactions
// ============================================================================

use crate::observability::MemoryPressureLevel;

/// Callback type for memory pressure changes.
pub type PressureCallback = Box<dyn Fn(MemoryPressureLevel) + Send + Sync>;

/// Reaction to memory pressure events.
///
/// Pools can be configured to react to memory pressure by shrinking
/// or invoking custom callbacks.
pub enum PressureReaction {
    /// No automatic reaction to pressure.
    None,
    /// Automatically shrink pool to target utilization.
    ///
    /// The `target_utilization` is a fraction (0.0 to 1.0) of the max
    /// pool size to retain when under pressure.
    Shrink {
        /// Target utilization as fraction of max capacity.
        target_utilization: f64,
    },
    /// Invoke a custom callback on pressure change.
    Callback(PressureCallback),
}

impl std::fmt::Debug for PressureReaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "PressureReaction::None"),
            Self::Shrink { target_utilization } => {
                write!(
                    f,
                    "PressureReaction::Shrink {{ target_utilization: {} }}",
                    target_utilization
                )
            }
            Self::Callback(_) => write!(f, "PressureReaction::Callback(<fn>)"),
        }
    }
}

/// Memory pressure handler for stratified pools.
///
/// Monitors memory pressure levels and triggers configured reactions.
pub struct PressureHandler {
    /// Configured reaction.
    reaction: PressureReaction,
    /// Current pressure level.
    current_level: Mutex<MemoryPressureLevel>,
}

impl PressureHandler {
    /// Create a new pressure handler with the specified reaction.
    pub fn new(reaction: PressureReaction) -> Self {
        Self {
            reaction,
            current_level: Mutex::new(MemoryPressureLevel::Normal),
        }
    }

    /// Create a handler with no reaction.
    pub fn no_reaction() -> Self {
        Self::new(PressureReaction::None)
    }

    /// Create a handler that shrinks to target utilization.
    pub fn shrink_to(target_utilization: f64) -> Self {
        Self::new(PressureReaction::Shrink {
            target_utilization: target_utilization.clamp(0.0, 1.0),
        })
    }

    /// Create a handler with a custom callback.
    pub fn with_callback<F>(callback: F) -> Self
    where
        F: Fn(MemoryPressureLevel) + Send + Sync + 'static,
    {
        Self::new(PressureReaction::Callback(Box::new(callback)))
    }

    /// Get the current pressure level.
    pub fn current_level(&self) -> MemoryPressureLevel {
        *self.current_level.lock()
    }

    /// Handle a pressure level change.
    ///
    /// Returns the number of buffers to retain per bucket (if shrinking).
    pub fn on_pressure_change(
        &self,
        new_level: MemoryPressureLevel,
        max_per_bucket: usize,
    ) -> Option<usize> {
        let old_level = {
            let mut current = self.current_level.lock();
            let old = *current;
            *current = new_level;
            old
        };

        // Only react if pressure increased
        if !Self::is_higher_pressure(new_level, old_level) {
            return None;
        }

        match &self.reaction {
            PressureReaction::None => None,
            PressureReaction::Shrink { target_utilization } => {
                // Calculate target based on pressure level
                let pressure_factor = Self::pressure_severity(new_level);
                let adjusted_target = target_utilization * (1.0 - pressure_factor);
                let target_count = ((max_per_bucket as f64) * adjusted_target) as usize;
                Some(target_count.max(1)) // Keep at least 1
            }
            PressureReaction::Callback(callback) => {
                callback(new_level);
                None
            }
        }
    }

    /// Check if new pressure level is higher than old.
    fn is_higher_pressure(new: MemoryPressureLevel, old: MemoryPressureLevel) -> bool {
        Self::pressure_ordinal(new) > Self::pressure_ordinal(old)
    }

    /// Get ordinal value for pressure level comparison.
    fn pressure_ordinal(level: MemoryPressureLevel) -> u8 {
        match level {
            MemoryPressureLevel::Normal => 0,
            MemoryPressureLevel::Elevated => 1,
            MemoryPressureLevel::Warning => 2,
            MemoryPressureLevel::Critical => 3,
            MemoryPressureLevel::OutOfMemory => 4,
        }
    }

    /// Get severity factor (0.0 to 1.0) for pressure level.
    fn pressure_severity(level: MemoryPressureLevel) -> f64 {
        match level {
            MemoryPressureLevel::Normal => 0.0,
            MemoryPressureLevel::Elevated => 0.2,
            MemoryPressureLevel::Warning => 0.5,
            MemoryPressureLevel::Critical => 0.8,
            MemoryPressureLevel::OutOfMemory => 1.0,
        }
    }
}

impl std::fmt::Debug for PressureHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PressureHandler")
            .field("reaction", &self.reaction)
            .field("current_level", &self.current_level())
            .finish()
    }
}

/// Extension trait for pressure-aware memory pools.
pub trait PressureAwarePool {
    /// Handle a memory pressure change event.
    ///
    /// Returns true if the pool took action (e.g., shrunk).
    fn handle_pressure(&self, level: MemoryPressureLevel) -> bool;

    /// Get current pressure level.
    fn pressure_level(&self) -> MemoryPressureLevel;
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

    // ========================================================================
    // Size-Stratified Pool Tests
    // ========================================================================

    #[test]
    fn test_size_bucket_sizes() {
        assert_eq!(SizeBucket::Tiny.size(), 256);
        assert_eq!(SizeBucket::Small.size(), 1024);
        assert_eq!(SizeBucket::Medium.size(), 4096);
        assert_eq!(SizeBucket::Large.size(), 16384);
        assert_eq!(SizeBucket::Huge.size(), 65536);
    }

    #[test]
    fn test_size_bucket_selection() {
        // Exact boundaries
        assert_eq!(SizeBucket::for_size(0), SizeBucket::Tiny);
        assert_eq!(SizeBucket::for_size(256), SizeBucket::Tiny);
        assert_eq!(SizeBucket::for_size(257), SizeBucket::Small);
        assert_eq!(SizeBucket::for_size(1024), SizeBucket::Small);
        assert_eq!(SizeBucket::for_size(1025), SizeBucket::Medium);
        assert_eq!(SizeBucket::for_size(4096), SizeBucket::Medium);
        assert_eq!(SizeBucket::for_size(4097), SizeBucket::Large);
        assert_eq!(SizeBucket::for_size(16384), SizeBucket::Large);
        assert_eq!(SizeBucket::for_size(16385), SizeBucket::Huge);
        assert_eq!(SizeBucket::for_size(100000), SizeBucket::Huge);
    }

    #[test]
    fn test_size_bucket_upgrade_downgrade() {
        assert_eq!(SizeBucket::Tiny.upgrade(), SizeBucket::Small);
        assert_eq!(SizeBucket::Small.upgrade(), SizeBucket::Medium);
        assert_eq!(SizeBucket::Medium.upgrade(), SizeBucket::Large);
        assert_eq!(SizeBucket::Large.upgrade(), SizeBucket::Huge);
        assert_eq!(SizeBucket::Huge.upgrade(), SizeBucket::Huge); // Max

        assert_eq!(SizeBucket::Tiny.downgrade(), SizeBucket::Tiny); // Min
        assert_eq!(SizeBucket::Small.downgrade(), SizeBucket::Tiny);
        assert_eq!(SizeBucket::Medium.downgrade(), SizeBucket::Small);
        assert_eq!(SizeBucket::Large.downgrade(), SizeBucket::Medium);
        assert_eq!(SizeBucket::Huge.downgrade(), SizeBucket::Large);
    }

    #[test]
    fn test_stratified_pool_allocation() {
        let pool = StratifiedMemoryPool::new("test");

        // Allocate different sizes
        let buf1 = pool.allocate(100); // Tiny
        let buf2 = pool.allocate(500); // Small
        let buf3 = pool.allocate(2000); // Medium

        assert_eq!(buf1.bucket(), SizeBucket::Tiny);
        assert_eq!(buf2.bucket(), SizeBucket::Small);
        assert_eq!(buf3.bucket(), SizeBucket::Medium);

        // Buffers have full bucket capacity
        assert_eq!(buf1.capacity(), 256);
        assert_eq!(buf2.capacity(), 1024);
        assert_eq!(buf3.capacity(), 4096);
    }

    #[test]
    fn test_stratified_pool_reuse() {
        let pool = StratifiedMemoryPool::new("test");

        // First allocation - fresh
        {
            let _buf = pool.allocate(100);
        }
        // Buffer returned to pool

        // Second allocation - should reuse
        {
            let _buf = pool.allocate(100);
        }

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_hits, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_stratified_pool_stats_per_bucket() {
        let pool = StratifiedMemoryPool::new("test");

        // Allocate from different buckets
        let _buf1 = pool.allocate(100); // Tiny
        let _buf2 = pool.allocate(500); // Small
        let _buf3 = pool.allocate(100); // Tiny again

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(
            stats.allocations_per_bucket.get(&SizeBucket::Tiny),
            Some(&2)
        );
        assert_eq!(
            stats.allocations_per_bucket.get(&SizeBucket::Small),
            Some(&1)
        );
    }

    #[test]
    fn test_stratified_pool_preallocate() {
        let pool = StratifiedMemoryPool::new("test");

        pool.preallocate(SizeBucket::Medium, 5);
        assert_eq!(pool.bucket_size(SizeBucket::Medium), 5);
        assert_eq!(pool.bucket_size(SizeBucket::Tiny), 0);

        // All medium allocations should hit cache
        for _ in 0..5 {
            let _buf = pool.allocate(2000);
        }

        let stats = pool.stats();
        assert_eq!(stats.hits_per_bucket.get(&SizeBucket::Medium), Some(&5));
    }

    #[test]
    fn test_stratified_pool_shrink() {
        let pool = StratifiedMemoryPool::new("test");

        // Preallocate then shrink
        pool.preallocate_all(10);
        assert_eq!(pool.total_pooled(), 50); // 5 buckets * 10

        pool.shrink_to(2);
        assert_eq!(pool.total_pooled(), 10); // 5 buckets * 2
    }

    #[test]
    fn test_stratified_buffer_deref() {
        let pool = StratifiedMemoryPool::new("test");

        let mut buf = pool.allocate(100);

        // Write via DerefMut
        buf[0] = 42;
        buf[1] = 43;

        // Read via Deref
        assert_eq!(buf[0], 42);
        assert_eq!(buf[1], 43);
    }

    // ========================================================================
    // Memory Pressure Reaction Tests
    // ========================================================================

    #[test]
    fn test_pressure_handler_no_reaction() {
        let handler = PressureHandler::no_reaction();
        assert_eq!(handler.current_level(), MemoryPressureLevel::Normal);

        let result = handler.on_pressure_change(MemoryPressureLevel::Critical, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_pressure_handler_shrink() {
        let handler = PressureHandler::shrink_to(0.5);

        // Normal -> Critical should trigger shrink
        let result = handler.on_pressure_change(MemoryPressureLevel::Critical, 10);
        assert!(result.is_some());
        // With 0.5 target and 0.8 severity, adjusted = 0.5 * (1.0 - 0.8) = 0.1
        // 10 * 0.1 = 1 -> max(1, 1) = 1
        assert!(result.unwrap() >= 1);
    }

    #[test]
    fn test_pressure_handler_callback() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let handler = PressureHandler::with_callback(move |level| {
            if level == MemoryPressureLevel::Critical {
                called_clone.store(true, Ordering::SeqCst);
            }
        });

        handler.on_pressure_change(MemoryPressureLevel::Critical, 10);
        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_pressure_handler_only_reacts_to_increase() {
        let handler = PressureHandler::shrink_to(0.5);

        // Start at Critical
        handler.on_pressure_change(MemoryPressureLevel::Critical, 10);

        // Going back to Normal should not trigger
        let result = handler.on_pressure_change(MemoryPressureLevel::Normal, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_pressure_handler_level_tracking() {
        let handler = PressureHandler::no_reaction();

        assert_eq!(handler.current_level(), MemoryPressureLevel::Normal);

        handler.on_pressure_change(MemoryPressureLevel::Warning, 10);
        assert_eq!(handler.current_level(), MemoryPressureLevel::Warning);

        handler.on_pressure_change(MemoryPressureLevel::Critical, 10);
        assert_eq!(handler.current_level(), MemoryPressureLevel::Critical);
    }

    #[test]
    fn test_pressure_reaction_debug() {
        let none = PressureReaction::None;
        assert!(format!("{:?}", none).contains("None"));

        let shrink = PressureReaction::Shrink {
            target_utilization: 0.5,
        };
        assert!(format!("{:?}", shrink).contains("0.5"));

        let callback = PressureReaction::Callback(Box::new(|_| {}));
        assert!(format!("{:?}", callback).contains("Callback"));
    }

    #[test]
    fn test_pressure_handler_debug() {
        let handler = PressureHandler::shrink_to(0.3);
        let debug_str = format!("{:?}", handler);
        assert!(debug_str.contains("PressureHandler"));
        assert!(debug_str.contains("Shrink"));
    }

    #[test]
    fn test_pressure_severity_values() {
        // Test that severity increases with pressure level
        let normal = PressureHandler::pressure_severity(MemoryPressureLevel::Normal);
        let elevated = PressureHandler::pressure_severity(MemoryPressureLevel::Elevated);
        let warning = PressureHandler::pressure_severity(MemoryPressureLevel::Warning);
        let critical = PressureHandler::pressure_severity(MemoryPressureLevel::Critical);
        let oom = PressureHandler::pressure_severity(MemoryPressureLevel::OutOfMemory);

        assert!(normal < elevated);
        assert!(elevated < warning);
        assert!(warning < critical);
        assert!(critical < oom);
        assert!(oom <= 1.0);
    }
}
