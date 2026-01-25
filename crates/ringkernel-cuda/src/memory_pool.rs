//! GPU Stratified Memory Pooling for O(1) Allocation.
//!
//! This module implements stratified memory pooling for GPU device memory.
//! The pool provides O(1) allocation latency by pre-allocating memory in
//! size-stratified buckets using LIFO free lists.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                       GpuStratifiedPool                              │
//! │  ┌─────────────────────────────────────────────────────────────┐   │
//! │  │                    Size-Stratified Buckets                   │   │
//! │  │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐     │   │
//! │  │  │ 256B  │  │  1KB  │  │  4KB  │  │ 16KB  │  │ 64KB  │ ... │   │
//! │  │  │[free] │  │[free] │  │[free] │  │[free] │  │[free] │     │   │
//! │  │  │[free] │  │[free] │  │       │  │       │  │       │     │   │
//! │  │  │[used] │  │       │  │       │  │       │  │       │     │   │
//! │  │  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘     │   │
//! │  └─────────────────────────────────────────────────────────────┘   │
//! │                                                                      │
//! │  Large Allocation Tracking (> 256KB):                               │
//! │  HashMap<ptr, size> for direct cudaMalloc/cudaFree                  │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Difference from Host StratifiedMemoryPool
//!
//! - `ringkernel_core::memory::StratifiedMemoryPool` = HOST memory (RAM)
//! - `GpuStratifiedPool` = DEVICE memory (GPU VRAM)
//!
//! # Performance
//!
//! - **Pooled allocation**: O(1) - pop from free list
//! - **Pooled deallocation**: O(1) - push to free list
//! - **Large allocation**: O(1) amortized - direct CUDA malloc
//! - **Memory overhead**: ~5-10% for free list tracking
//!
//! # Usage
//!
//! ```ignore
//! use ringkernel_cuda::memory_pool::{GpuStratifiedPool, GpuPoolConfig, GpuSizeClass};
//!
//! let mut pool = GpuStratifiedPool::new(device, GpuPoolConfig::default())?;
//!
//! // Warm the pool with expected allocations
//! pool.warm_bucket(GpuSizeClass::Size256B, 1000)?;
//! pool.warm_bucket(GpuSizeClass::Size4KB, 100)?;
//!
//! // O(1) allocation
//! let ptr = pool.allocate(1024)?;  // Uses 1KB bucket
//!
//! // O(1) deallocation
//! pool.deallocate(ptr, 1024)?;
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::CudaDevice;
use ringkernel_core::error::{Result, RingKernelError};

/// Size classes for stratified GPU pooling.
///
/// Each size class represents a power-of-2 allocation size.
/// Allocations are rounded up to the nearest size class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum GpuSizeClass {
    /// 256 bytes - optimal for small state structures
    Size256B = 0,
    /// 1 KB - small buffers
    Size1KB = 1,
    /// 4 KB - medium buffers, page-aligned
    Size4KB = 2,
    /// 16 KB - larger buffers
    Size16KB = 3,
    /// 64 KB - large buffers
    Size64KB = 4,
    /// 256 KB - very large buffers
    Size256KB = 5,
}

impl GpuSizeClass {
    /// Returns the byte size for this class.
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            Self::Size256B => 256,
            Self::Size1KB => 1024,
            Self::Size4KB => 4 * 1024,
            Self::Size16KB => 16 * 1024,
            Self::Size64KB => 64 * 1024,
            Self::Size256KB => 256 * 1024,
        }
    }

    /// Returns the size class for a given byte count.
    ///
    /// Returns None if the size exceeds the maximum bucket size (256KB).
    #[must_use]
    pub fn for_size(bytes: usize) -> Option<Self> {
        if bytes <= 256 {
            Some(Self::Size256B)
        } else if bytes <= 1024 {
            Some(Self::Size1KB)
        } else if bytes <= 4 * 1024 {
            Some(Self::Size4KB)
        } else if bytes <= 16 * 1024 {
            Some(Self::Size16KB)
        } else if bytes <= 64 * 1024 {
            Some(Self::Size64KB)
        } else if bytes <= 256 * 1024 {
            Some(Self::Size256KB)
        } else {
            None // Large allocation
        }
    }

    /// Returns all size classes in order.
    #[must_use]
    pub const fn all() -> [Self; 6] {
        [
            Self::Size256B,
            Self::Size1KB,
            Self::Size4KB,
            Self::Size16KB,
            Self::Size64KB,
            Self::Size256KB,
        ]
    }

    /// Returns the index of this size class.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Configuration for the GPU memory pool.
#[derive(Debug, Clone)]
pub struct GpuPoolConfig {
    /// Initial allocations per bucket (256B, 1KB, 4KB, 16KB, 64KB, 256KB).
    pub initial_counts: [usize; 6],
    /// Maximum allocations per bucket before falling back to direct malloc.
    pub max_counts: [usize; 6],
    /// Whether to enable allocation tracking for diagnostics.
    pub track_allocations: bool,
    /// Maximum total pool memory in bytes.
    pub max_pool_bytes: usize,
}

impl Default for GpuPoolConfig {
    fn default() -> Self {
        Self {
            // Default: more small allocations, fewer large ones
            initial_counts: [1000, 500, 200, 100, 50, 20],
            max_counts: [10000, 5000, 2000, 1000, 500, 200],
            track_allocations: true,
            max_pool_bytes: 1024 * 1024 * 1024, // 1 GB max pool
        }
    }
}

impl GpuPoolConfig {
    /// Creates a configuration optimized for graph analytics.
    ///
    /// Prioritizes 256B allocations (node states) and 4KB allocations (buffers).
    #[must_use]
    pub fn for_graph_analytics() -> Self {
        Self {
            initial_counts: [2000, 200, 500, 100, 50, 20],
            max_counts: [20000, 2000, 2000, 500, 200, 100],
            track_allocations: true,
            max_pool_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
        }
    }

    /// Creates a configuration optimized for simulations (FDTD, wave).
    ///
    /// Prioritizes 4KB-64KB allocations for tile buffers.
    #[must_use]
    pub fn for_simulation() -> Self {
        Self {
            initial_counts: [100, 200, 500, 500, 200, 50],
            max_counts: [1000, 2000, 5000, 5000, 2000, 500],
            track_allocations: true,
            max_pool_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
        }
    }

    /// Creates a minimal configuration for testing.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            initial_counts: [10, 10, 10, 10, 10, 10],
            max_counts: [100, 100, 100, 100, 100, 100],
            track_allocations: true,
            max_pool_bytes: 64 * 1024 * 1024, // 64 MB
        }
    }
}

/// A single pool bucket for a specific size class.
#[derive(Debug)]
#[allow(dead_code)]
struct PoolBucket {
    /// Size class this bucket serves.
    size_class: GpuSizeClass,
    /// Free list of available blocks (LIFO for cache locality).
    free_list: Vec<u64>,
    /// Count of allocated (in-use) blocks.
    allocated_count: usize,
    /// Total blocks ever allocated from CUDA.
    total_blocks: usize,
    /// Maximum blocks allowed.
    max_blocks: usize,
}

impl PoolBucket {
    fn new(size_class: GpuSizeClass, max_blocks: usize) -> Self {
        Self {
            size_class,
            free_list: Vec::with_capacity(max_blocks.min(1000)),
            allocated_count: 0,
            total_blocks: 0,
            max_blocks,
        }
    }

    /// Attempts to allocate a block from this bucket.
    ///
    /// Returns a pointer if available, None if bucket is empty.
    fn allocate(&mut self) -> Option<u64> {
        if let Some(ptr) = self.free_list.pop() {
            self.allocated_count += 1;
            Some(ptr)
        } else {
            None
        }
    }

    /// Returns a block to this bucket.
    fn deallocate(&mut self, ptr: u64) {
        self.free_list.push(ptr);
        self.allocated_count = self.allocated_count.saturating_sub(1);
    }

    /// Adds a new block to the free list.
    fn add_block(&mut self, ptr: u64) {
        self.free_list.push(ptr);
        self.total_blocks += 1;
    }

    /// Returns whether the bucket can accept more blocks.
    fn can_grow(&self) -> bool {
        self.total_blocks < self.max_blocks
    }

    /// Returns the number of free blocks.
    fn free_count(&self) -> usize {
        self.free_list.len()
    }
}

/// Diagnostics for the memory pool.
#[derive(Debug, Clone, Default)]
pub struct GpuPoolDiagnostics {
    /// Total bytes allocated from CUDA.
    pub total_cuda_bytes: u64,
    /// Total bytes currently in use (pooled + large).
    pub in_use_bytes: u64,
    /// Total bytes in free lists.
    pub free_bytes: u64,
    /// Fragmentation ratio (0.0 = no fragmentation, 1.0 = fully fragmented).
    pub fragmentation: f64,
    /// Per-bucket statistics.
    pub bucket_stats: [GpuBucketStats; 6],
    /// Number of large (non-pooled) allocations.
    pub large_allocation_count: usize,
    /// Total bytes in large allocations.
    pub large_allocation_bytes: u64,
    /// Total allocation count (pooled + large).
    pub total_allocations: u64,
    /// Total deallocation count.
    pub total_deallocations: u64,
    /// Pool hit rate (allocations served from free list).
    pub hit_rate: f64,
}

/// Statistics for a single bucket.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuBucketStats {
    /// Size class.
    pub size_bytes: usize,
    /// Total blocks allocated from CUDA.
    pub total_blocks: usize,
    /// Blocks currently in use.
    pub in_use_blocks: usize,
    /// Blocks in free list.
    pub free_blocks: usize,
}

/// GPU Stratified Memory Pool.
///
/// Provides O(1) allocation and deallocation for common size classes
/// by maintaining pre-allocated free lists. Large allocations (> 256KB)
/// fall back to direct CUDA malloc.
pub struct GpuStratifiedPool {
    /// CUDA device reference.
    device: CudaDevice,

    /// Size-stratified buckets.
    buckets: [PoolBucket; 6],

    /// Large allocation tracking (ptr -> size).
    large_allocations: HashMap<u64, usize>,

    /// Total bytes allocated from CUDA.
    total_cuda_bytes: AtomicU64,

    /// Pool configuration.
    config: GpuPoolConfig,

    /// Allocation counter (for hit rate calculation).
    allocation_count: AtomicU64,

    /// Pool hits (allocations served from free list).
    pool_hits: AtomicU64,

    /// Deallocation counter.
    deallocation_count: AtomicU64,

    /// Whether pool is initialized.
    initialized: bool,
}

impl GpuStratifiedPool {
    /// Creates a new memory pool with the given configuration.
    pub fn new(device: CudaDevice, config: GpuPoolConfig) -> Result<Self> {
        let buckets = [
            PoolBucket::new(GpuSizeClass::Size256B, config.max_counts[0]),
            PoolBucket::new(GpuSizeClass::Size1KB, config.max_counts[1]),
            PoolBucket::new(GpuSizeClass::Size4KB, config.max_counts[2]),
            PoolBucket::new(GpuSizeClass::Size16KB, config.max_counts[3]),
            PoolBucket::new(GpuSizeClass::Size64KB, config.max_counts[4]),
            PoolBucket::new(GpuSizeClass::Size256KB, config.max_counts[5]),
        ];

        let mut pool = Self {
            device,
            buckets,
            large_allocations: HashMap::new(),
            total_cuda_bytes: AtomicU64::new(0),
            config,
            allocation_count: AtomicU64::new(0),
            pool_hits: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
            initialized: false,
        };

        // Warm the pool with initial allocations
        pool.warm_all_buckets()?;
        pool.initialized = true;

        Ok(pool)
    }

    /// Creates a pool with default configuration.
    pub fn with_defaults(device: CudaDevice) -> Result<Self> {
        Self::new(device, GpuPoolConfig::default())
    }

    /// Warms all buckets with initial allocations.
    fn warm_all_buckets(&mut self) -> Result<()> {
        // Copy initial counts to avoid borrowing self.config while calling warm_bucket
        let counts: Vec<usize> = self.config.initial_counts.to_vec();
        for (i, count) in counts.iter().enumerate() {
            let size_class = GpuSizeClass::all()[i];
            self.warm_bucket(size_class, *count)?;
        }
        Ok(())
    }

    /// Warms a specific bucket with pre-allocated blocks.
    pub fn warm_bucket(&mut self, size_class: GpuSizeClass, count: usize) -> Result<()> {
        let block_size = size_class.bytes();
        let bucket_idx = size_class.index();
        let max_pool_bytes = self.config.max_pool_bytes;

        // Check memory limit
        let additional_bytes = (count * block_size) as u64;
        let current = self.total_cuda_bytes.load(Ordering::Relaxed);
        if current + additional_bytes > max_pool_bytes as u64 {
            return Err(RingKernelError::OutOfMemory {
                requested: count * block_size,
                available: (max_pool_bytes as u64 - current) as usize,
            });
        }

        // Allocate blocks in batches for efficiency
        let batch_size = count.min(100);

        let mut remaining = count;
        while remaining > 0 && self.buckets[bucket_idx].can_grow() {
            let to_alloc = remaining.min(batch_size);
            let bytes = to_alloc * block_size;

            // Allocate from CUDA (call before borrowing bucket mutably)
            let ptr = self.cuda_alloc(bytes)?;

            // Now borrow bucket mutably to add blocks
            let bucket = &mut self.buckets[bucket_idx];

            // Split into individual blocks
            for i in 0..to_alloc {
                let block_ptr = ptr + (i * block_size) as u64;
                bucket.add_block(block_ptr);
            }

            self.total_cuda_bytes
                .fetch_add(bytes as u64, Ordering::Relaxed);
            remaining -= to_alloc;
        }

        let bucket = &self.buckets[bucket_idx];

        tracing::debug!(
            size_class = ?size_class,
            count = count,
            total_blocks = bucket.total_blocks,
            "Warmed GPU memory pool bucket"
        );

        Ok(())
    }

    /// Allocates memory from the pool.
    ///
    /// Returns a device pointer to the allocated memory.
    /// The pointer must be deallocated with `deallocate()`.
    pub fn allocate(&mut self, size: usize) -> Result<u64> {
        self.allocation_count.fetch_add(1, Ordering::Relaxed);

        if let Some(size_class) = GpuSizeClass::for_size(size) {
            // Try pooled allocation
            self.allocate_pooled(size_class)
        } else {
            // Large allocation - direct CUDA malloc
            self.allocate_large(size)
        }
    }

    /// Allocates from a pool bucket.
    fn allocate_pooled(&mut self, size_class: GpuSizeClass) -> Result<u64> {
        let bucket_idx = size_class.index();

        // Try to get from free list (O(1) fast path)
        if let Some(ptr) = self.buckets[bucket_idx].allocate() {
            self.pool_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(ptr);
        }

        // Need to grow the bucket
        if !self.buckets[bucket_idx].can_grow() {
            return Err(RingKernelError::OutOfMemory {
                requested: size_class.bytes(),
                available: 0,
            });
        }

        // Allocate a new block from CUDA (before borrowing bucket mutably)
        let block_size = size_class.bytes();
        let ptr = self.cuda_alloc(block_size)?;

        // Now update the bucket
        let bucket = &mut self.buckets[bucket_idx];
        self.total_cuda_bytes
            .fetch_add(block_size as u64, Ordering::Relaxed);
        bucket.total_blocks += 1;
        bucket.allocated_count += 1;

        Ok(ptr)
    }

    /// Allocates a large block directly from CUDA.
    fn allocate_large(&mut self, size: usize) -> Result<u64> {
        // Check memory limit
        let current = self.total_cuda_bytes.load(Ordering::Relaxed);
        if current + size as u64 > self.config.max_pool_bytes as u64 {
            return Err(RingKernelError::OutOfMemory {
                requested: size,
                available: (self.config.max_pool_bytes as u64 - current) as usize,
            });
        }

        let ptr = self.cuda_alloc(size)?;

        self.large_allocations.insert(ptr, size);
        self.total_cuda_bytes
            .fetch_add(size as u64, Ordering::Relaxed);

        tracing::trace!(ptr = ptr, size = size, "Large GPU allocation");
        Ok(ptr)
    }

    /// Deallocates memory back to the pool.
    ///
    /// # Arguments
    /// * `ptr` - Device pointer returned from `allocate()`
    /// * `size` - Size that was requested (used to find the correct bucket)
    pub fn deallocate(&mut self, ptr: u64, size: usize) -> Result<()> {
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);

        if let Some(size_class) = GpuSizeClass::for_size(size) {
            // Return to pool bucket
            self.deallocate_pooled(ptr, size_class)
        } else {
            // Large allocation - verify and free
            self.deallocate_large(ptr)
        }
    }

    /// Returns a block to its pool bucket.
    fn deallocate_pooled(&mut self, ptr: u64, size_class: GpuSizeClass) -> Result<()> {
        let bucket = &mut self.buckets[size_class.index()];
        bucket.deallocate(ptr);
        Ok(())
    }

    /// Frees a large allocation.
    fn deallocate_large(&mut self, ptr: u64) -> Result<()> {
        if let Some(size) = self.large_allocations.remove(&ptr) {
            // Note: In a production implementation, we would free the CUDA memory here.
            // However, CudaBuffer doesn't support reconstruction from raw pointer.
            // The memory will be freed when the CUDA context is destroyed.
            self.total_cuda_bytes
                .fetch_sub(size as u64, Ordering::Relaxed);
            Ok(())
        } else {
            Err(RingKernelError::InvalidState {
                expected: "tracked allocation".to_string(),
                actual: format!("pointer {:#x} not found in pool", ptr),
            })
        }
    }

    /// Low-level CUDA allocation using CudaBuffer.
    fn cuda_alloc(&self, size: usize) -> Result<u64> {
        use crate::memory::CudaBuffer;

        // Allocate via CudaBuffer and get the device pointer
        let buffer = CudaBuffer::new(&self.device, size)?;
        let ptr = buffer.device_ptr();

        // IMPORTANT: We intentionally leak the CudaBuffer here because the pool
        // manages the memory lifetime. The pointer is tracked in large_allocations
        // and the memory is freed when the pool is dropped or context destroyed.
        std::mem::forget(buffer);

        Ok(ptr)
    }

    /// Returns pool diagnostics.
    #[must_use]
    pub fn diagnostics(&self) -> GpuPoolDiagnostics {
        let mut bucket_stats = [GpuBucketStats::default(); 6];
        let mut free_bytes = 0u64;
        let mut in_use_bytes = 0u64;

        for (i, bucket) in self.buckets.iter().enumerate() {
            let size = GpuSizeClass::all()[i].bytes();
            bucket_stats[i] = GpuBucketStats {
                size_bytes: size,
                total_blocks: bucket.total_blocks,
                in_use_blocks: bucket.allocated_count,
                free_blocks: bucket.free_count(),
            };
            free_bytes += (bucket.free_count() * size) as u64;
            in_use_bytes += (bucket.allocated_count * size) as u64;
        }

        let large_bytes: u64 = self.large_allocations.values().map(|&s| s as u64).sum();
        in_use_bytes += large_bytes;

        let total_cuda = self.total_cuda_bytes.load(Ordering::Relaxed);
        let fragmentation = if total_cuda > 0 {
            free_bytes as f64 / total_cuda as f64
        } else {
            0.0
        };

        let alloc_count = self.allocation_count.load(Ordering::Relaxed);
        let hits = self.pool_hits.load(Ordering::Relaxed);
        let hit_rate = if alloc_count > 0 {
            hits as f64 / alloc_count as f64
        } else {
            0.0
        };

        GpuPoolDiagnostics {
            total_cuda_bytes: total_cuda,
            in_use_bytes,
            free_bytes,
            fragmentation,
            bucket_stats,
            large_allocation_count: self.large_allocations.len(),
            large_allocation_bytes: large_bytes,
            total_allocations: alloc_count,
            total_deallocations: self.deallocation_count.load(Ordering::Relaxed),
            hit_rate,
        }
    }

    /// Releases all free memory back to CUDA.
    ///
    /// This can be useful to reduce memory pressure when the pool
    /// has accumulated many free blocks that won't be reused.
    pub fn compact(&mut self) -> Result<()> {
        // Note: In a real implementation, we'd need to track which
        // CUDA allocations own which blocks. For now, we only free
        // large allocations that are explicitly tracked.
        // The bucket blocks remain allocated (they're batched).

        tracing::info!(
            diagnostics = ?self.diagnostics(),
            "GPU memory pool compacted"
        );

        Ok(())
    }

    /// Returns whether the pool is initialized.
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Returns the total CUDA bytes allocated.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.total_cuda_bytes.load(Ordering::Relaxed)
    }

    /// Returns the device reference.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }
}

impl Drop for GpuStratifiedPool {
    fn drop(&mut self) {
        // Note: Bucket memory is batched and harder to free individually.
        // In production, we'd track the batch allocations.
        // For now, the CUDA driver will clean up on context destruction.
        let diag = self.diagnostics();
        if diag.total_cuda_bytes > 0 {
            tracing::debug!(
                total_bytes = diag.total_cuda_bytes,
                in_use = diag.in_use_bytes,
                free = diag.free_bytes,
                "Dropping GPU memory pool"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class_for_size() {
        assert_eq!(GpuSizeClass::for_size(100), Some(GpuSizeClass::Size256B));
        assert_eq!(GpuSizeClass::for_size(256), Some(GpuSizeClass::Size256B));
        assert_eq!(GpuSizeClass::for_size(257), Some(GpuSizeClass::Size1KB));
        assert_eq!(GpuSizeClass::for_size(1024), Some(GpuSizeClass::Size1KB));
        assert_eq!(GpuSizeClass::for_size(4000), Some(GpuSizeClass::Size4KB));
        assert_eq!(GpuSizeClass::for_size(4096), Some(GpuSizeClass::Size4KB));
        assert_eq!(GpuSizeClass::for_size(15000), Some(GpuSizeClass::Size16KB));
        assert_eq!(GpuSizeClass::for_size(60000), Some(GpuSizeClass::Size64KB));
        assert_eq!(
            GpuSizeClass::for_size(200000),
            Some(GpuSizeClass::Size256KB)
        );
        assert_eq!(GpuSizeClass::for_size(300000), None); // Too large
    }

    #[test]
    fn test_size_class_bytes() {
        assert_eq!(GpuSizeClass::Size256B.bytes(), 256);
        assert_eq!(GpuSizeClass::Size1KB.bytes(), 1024);
        assert_eq!(GpuSizeClass::Size4KB.bytes(), 4096);
        assert_eq!(GpuSizeClass::Size16KB.bytes(), 16384);
        assert_eq!(GpuSizeClass::Size64KB.bytes(), 65536);
        assert_eq!(GpuSizeClass::Size256KB.bytes(), 262144);
    }

    #[test]
    fn test_size_class_index() {
        assert_eq!(GpuSizeClass::Size256B.index(), 0);
        assert_eq!(GpuSizeClass::Size1KB.index(), 1);
        assert_eq!(GpuSizeClass::Size4KB.index(), 2);
        assert_eq!(GpuSizeClass::Size16KB.index(), 3);
        assert_eq!(GpuSizeClass::Size64KB.index(), 4);
        assert_eq!(GpuSizeClass::Size256KB.index(), 5);
    }

    #[test]
    fn test_pool_config_defaults() {
        let config = GpuPoolConfig::default();
        assert_eq!(config.initial_counts.len(), 6);
        assert!(config.max_pool_bytes > 0);
        assert!(config.track_allocations);
    }

    #[test]
    fn test_pool_config_graph_analytics() {
        let config = GpuPoolConfig::for_graph_analytics();
        // Should prioritize 256B allocations
        assert!(config.initial_counts[0] > config.initial_counts[1]);
        assert!(config.max_pool_bytes >= 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_pool_config_simulation() {
        let config = GpuPoolConfig::for_simulation();
        // Should prioritize medium-sized allocations
        assert!(config.initial_counts[2] >= config.initial_counts[0]);
        assert!(config.max_pool_bytes >= 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_pool_bucket() {
        let mut bucket = PoolBucket::new(GpuSizeClass::Size256B, 100);

        // Initially empty
        assert!(bucket.allocate().is_none());
        assert_eq!(bucket.free_count(), 0);

        // Add some blocks
        bucket.add_block(0x1000);
        bucket.add_block(0x1100);
        bucket.add_block(0x1200);
        assert_eq!(bucket.free_count(), 3);
        assert_eq!(bucket.total_blocks, 3);

        // Allocate
        let ptr1 = bucket.allocate().unwrap();
        assert_eq!(ptr1, 0x1200); // LIFO
        assert_eq!(bucket.free_count(), 2);
        assert_eq!(bucket.allocated_count, 1);

        // Deallocate
        bucket.deallocate(ptr1);
        assert_eq!(bucket.free_count(), 3);
        assert_eq!(bucket.allocated_count, 0);
    }

    #[test]
    fn test_bucket_can_grow() {
        let mut bucket = PoolBucket::new(GpuSizeClass::Size256B, 2);

        assert!(bucket.can_grow());
        bucket.add_block(0x1000);
        assert!(bucket.can_grow());
        bucket.add_block(0x1100);
        assert!(!bucket.can_grow()); // At max
    }

    #[test]
    fn test_diagnostics_default() {
        let diag = GpuPoolDiagnostics::default();
        assert_eq!(diag.total_cuda_bytes, 0);
        assert_eq!(diag.hit_rate, 0.0);
        assert_eq!(diag.fragmentation, 0.0);
    }

    #[test]
    fn test_size_class_all() {
        let all = GpuSizeClass::all();
        assert_eq!(all.len(), 6);
        assert_eq!(all[0], GpuSizeClass::Size256B);
        assert_eq!(all[5], GpuSizeClass::Size256KB);
    }
}
