//! CUDA Reduction Buffer Implementation
//!
//! This module provides GPU-accelerated reduction primitives using CUDA.
//! It implements the `ReductionHandle` trait from ringkernel-core.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    ReductionBuffer<T>                        │
//! │  ┌─────────────────────────────────────────────────────────┐ │
//! │  │         MAPPED MEMORY (CPU + GPU visible)               │ │
//! │  │  [slot_0] [slot_1] ... [slot_n]  [barrier_counter]      │ │
//! │  └─────────────────────────────────────────────────────────┘ │
//! │                                                              │
//! │  CPU Access:                    GPU Access:                  │
//! │  • read() / read_combined()     • atomicAdd to device_ptr   │
//! │  • reset() to identity          • __threadfence_system()    │
//! │  • sync_and_read()              • grid.sync() / barrier     │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use ringkernel_cuda::reduction::{ReductionBuffer, ReductionOp};
//!
//! // Create a sum reduction buffer
//! let buffer = ReductionBuffer::<f32>::new(&device, ReductionOp::Sum)?;
//!
//! // Pass device_ptr to kernel
//! let ptr = buffer.device_ptr();
//!
//! // After kernel execution, read result
//! let sum = buffer.sync_and_read()?;
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::atomic::{fence, Ordering};
use std::sync::Mutex;

use cudarc::driver::sys as cuda_sys;

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::reduction::{ReductionConfig, ReductionHandle, ReductionOp, ReductionScalar};

use crate::device::CudaDevice;

/// CUDA-backed reduction buffer using mapped memory.
///
/// This buffer is visible to both CPU and GPU, enabling zero-copy access
/// to reduction results. Uses `cudaHostAllocMapped` for the underlying storage.
///
/// # Memory Layout
///
/// The buffer contains `num_slots` values of type `T`, followed by optional
/// barrier synchronization counters for software grid sync.
///
/// # Thread Safety
///
/// - GPU writes use atomic operations (`atomicAdd`, etc.)
/// - CPU reads use volatile operations after memory fence
/// - Grid-wide sync uses cooperative groups or software barriers
pub struct ReductionBuffer<T: ReductionScalar> {
    /// Host pointer (CPU-accessible).
    host_ptr: *mut T,
    /// Device pointer (GPU-accessible).
    device_ptr: u64,
    /// Number of accumulation slots.
    num_slots: usize,
    /// Reduction operation type.
    op: ReductionOp,
    /// Parent device.
    device: CudaDevice,
    /// Type marker.
    _marker: PhantomData<T>,
}

// Safety: ReductionBuffer uses pinned mapped memory with atomic synchronization
unsafe impl<T: ReductionScalar> Send for ReductionBuffer<T> {}
unsafe impl<T: ReductionScalar> Sync for ReductionBuffer<T> {}

impl<T: ReductionScalar> ReductionBuffer<T> {
    /// Create a new reduction buffer with a single slot.
    ///
    /// # Arguments
    ///
    /// * `device` - CUDA device to allocate on
    /// * `op` - Reduction operation type (determines identity value)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let buffer = ReductionBuffer::<f32>::new(&device, ReductionOp::Sum)?;
    /// ```
    pub fn new(device: &CudaDevice, op: ReductionOp) -> Result<Self> {
        Self::with_slots(device, 1, op)
    }

    /// Create a reduction buffer with multiple slots.
    ///
    /// Multiple slots reduce atomic contention by spreading updates across
    /// several memory locations. Use with `read_combined()` to get final result.
    ///
    /// # Arguments
    ///
    /// * `device` - CUDA device
    /// * `num_slots` - Number of accumulation slots (minimum 1)
    /// * `op` - Reduction operation
    ///
    /// # Performance
    ///
    /// - 1 slot: Highest contention, simplest code
    /// - 4-8 slots: Good balance for most workloads
    /// - 32+ slots: Minimal contention, more host-side work
    pub fn with_slots(device: &CudaDevice, num_slots: usize, op: ReductionOp) -> Result<Self> {
        let num_slots = num_slots.max(1);

        // Ensure device context is active
        let _ = device.inner();

        let size_bytes = num_slots * std::mem::size_of::<T>();
        let mut host_ptr: *mut c_void = ptr::null_mut();

        // Allocate pinned mapped memory
        unsafe {
            let result = cuda_sys::cuMemHostAlloc(
                &mut host_ptr,
                size_bytes,
                cuda_sys::CU_MEMHOSTALLOC_DEVICEMAP | cuda_sys::CU_MEMHOSTALLOC_PORTABLE,
            );

            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::AllocationFailed {
                    size: size_bytes,
                    reason: format!("cuMemHostAlloc failed for reduction buffer: {:?}", result),
                });
            }
        }

        // Get device pointer for GPU access
        let mut device_ptr: u64 = 0;
        unsafe {
            let result = cuda_sys::cuMemHostGetDevicePointer_v2(&mut device_ptr, host_ptr, 0);

            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                // Clean up allocation
                let _ = cuda_sys::cuMemFreeHost(host_ptr);
                return Err(RingKernelError::AllocationFailed {
                    size: size_bytes,
                    reason: format!(
                        "cuMemHostGetDevicePointer failed for reduction buffer: {:?}",
                        result
                    ),
                });
            }
        }

        let buffer = Self {
            host_ptr: host_ptr as *mut T,
            device_ptr,
            num_slots,
            op,
            device: device.clone(),
            _marker: PhantomData,
        };

        // Initialize to identity value
        buffer.reset()?;

        Ok(buffer)
    }

    /// Create a buffer from a configuration.
    pub fn from_config(
        device: &CudaDevice,
        op: ReductionOp,
        config: &ReductionConfig,
    ) -> Result<Self> {
        Self::with_slots(device, config.num_slots, op)
    }

    /// Get device pointer for kernel parameter passing.
    ///
    /// Pass this pointer to your CUDA kernel as a `T*` parameter.
    /// The kernel should use atomic operations to update the value.
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    /// Get number of slots.
    #[inline]
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Get the reduction operation type.
    #[inline]
    pub fn op(&self) -> ReductionOp {
        self.op
    }

    /// Get the parent device.
    #[inline]
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Reset all slots to identity value.
    ///
    /// Call this before each reduction to ensure clean starting state.
    /// Uses volatile writes for GPU visibility.
    pub fn reset(&self) -> Result<()> {
        let identity = T::identity(self.op);

        // Use volatile writes for GPU visibility
        for i in 0..self.num_slots {
            unsafe {
                ptr::write_volatile(self.host_ptr.add(i), identity);
            }
        }

        // Memory fence to ensure writes are visible
        fence(Ordering::SeqCst);

        Ok(())
    }

    /// Read the result from slot 0.
    ///
    /// For single-slot buffers, this is the final result.
    /// For multi-slot buffers, use `read_combined()` instead.
    pub fn read(&self) -> Result<T> {
        self.read_slot(0)
    }

    /// Read the result from a specific slot.
    pub fn read_slot(&self, slot: usize) -> Result<T> {
        if slot >= self.num_slots {
            return Err(RingKernelError::InvalidIndex(slot));
        }

        // Memory fence before reading
        fence(Ordering::SeqCst);

        // Volatile read for GPU visibility
        let value = unsafe { ptr::read_volatile(self.host_ptr.add(slot)) };

        Ok(value)
    }

    /// Read and combine all slots into a single result.
    ///
    /// Uses the reduction operation to combine values from all slots.
    /// This is the preferred method for multi-slot buffers.
    pub fn read_combined(&self) -> Result<T> {
        // Memory fence before reading
        fence(Ordering::SeqCst);

        let mut result = T::identity(self.op);

        for i in 0..self.num_slots {
            let value = unsafe { ptr::read_volatile(self.host_ptr.add(i)) };
            result = T::combine(result, value, self.op);
        }

        Ok(result)
    }

    /// Synchronize device and read result from slot 0.
    ///
    /// Ensures all GPU operations complete before reading.
    /// Use this when you need guaranteed visibility of kernel writes.
    pub fn sync_and_read(&self) -> Result<T> {
        self.device.synchronize()?;
        self.read()
    }

    /// Synchronize device and read combined result.
    ///
    /// Combines `synchronize()` and `read_combined()`.
    pub fn sync_and_read_combined(&self) -> Result<T> {
        self.device.synchronize()?;
        self.read_combined()
    }

    /// Write a value to a specific slot (for testing/initialization).
    ///
    /// Use `reset()` instead for normal operation.
    pub fn write_slot(&self, slot: usize, value: T) -> Result<()> {
        if slot >= self.num_slots {
            return Err(RingKernelError::InvalidIndex(slot));
        }

        unsafe {
            ptr::write_volatile(self.host_ptr.add(slot), value);
        }
        fence(Ordering::SeqCst);

        Ok(())
    }

    /// Get raw host pointer (for advanced use).
    ///
    /// # Safety
    ///
    /// Caller must ensure proper synchronization with GPU operations.
    #[inline]
    pub unsafe fn host_ptr(&self) -> *mut T {
        self.host_ptr
    }
}

impl<T: ReductionScalar> Drop for ReductionBuffer<T> {
    fn drop(&mut self) {
        // Ensure device context is active
        let _ = self.device.inner();

        unsafe {
            let _ = cuda_sys::cuMemFreeHost(self.host_ptr as *mut c_void);
        }
    }
}

impl<T: ReductionScalar> ReductionHandle<T> for ReductionBuffer<T> {
    fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    fn reset(&self) -> Result<()> {
        ReductionBuffer::reset(self)
    }

    fn read(&self) -> Result<T> {
        ReductionBuffer::read(self)
    }

    fn read_combined(&self) -> Result<T> {
        ReductionBuffer::read_combined(self)
    }

    fn sync_and_read(&self) -> Result<T> {
        ReductionBuffer::sync_and_read(self)
    }

    fn op(&self) -> ReductionOp {
        self.op
    }

    fn num_slots(&self) -> usize {
        self.num_slots
    }
}

/// Builder for `ReductionBuffer` with fluent configuration.
pub struct ReductionBufferBuilder<T: ReductionScalar> {
    op: ReductionOp,
    num_slots: usize,
    _marker: PhantomData<T>,
}

impl<T: ReductionScalar> ReductionBufferBuilder<T> {
    /// Create a new builder with the specified operation.
    pub fn new(op: ReductionOp) -> Self {
        Self {
            op,
            num_slots: 1,
            _marker: PhantomData,
        }
    }

    /// Set the number of accumulation slots.
    pub fn with_slots(mut self, num_slots: usize) -> Self {
        self.num_slots = num_slots.max(1);
        self
    }

    /// Build the reduction buffer.
    pub fn build(self, device: &CudaDevice) -> Result<ReductionBuffer<T>> {
        ReductionBuffer::with_slots(device, self.num_slots, self.op)
    }
}

/// Statistics for the reduction buffer cache.
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Number of cache hits (buffer reused).
    pub hits: AtomicU64,
    /// Number of cache misses (new buffer allocated).
    pub misses: AtomicU64,
    /// Number of buffers returned to cache.
    pub returns: AtomicU64,
}

impl CacheStats {
    /// Create new statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get hit count.
    pub fn hits(&self) -> u64 {
        self.hits.load(AtomicOrdering::Relaxed)
    }

    /// Get miss count.
    pub fn misses(&self) -> u64 {
        self.misses.load(AtomicOrdering::Relaxed)
    }

    /// Get return count.
    pub fn returns(&self) -> u64 {
        self.returns.load(AtomicOrdering::Relaxed)
    }

    /// Calculate hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits() as f64;
        let misses = self.misses() as f64;
        let total = hits + misses;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
}

/// Cache key for reduction buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Number of slots.
    pub num_slots: usize,
    /// Reduction operation type.
    pub op: ReductionOp,
}

impl CacheKey {
    /// Create a new cache key.
    pub fn new(num_slots: usize, op: ReductionOp) -> Self {
        Self { num_slots, op }
    }
}

/// Cache for reduction buffers to enable efficient reuse.
///
/// PageRank and other iterative algorithms frequently create and destroy
/// reduction buffers. This cache pools buffers by (num_slots, op) key
/// to avoid repeated allocation of pinned mapped memory.
///
/// # Example
///
/// ```ignore
/// use ringkernel_cuda::reduction::{ReductionBufferCache, ReductionOp};
///
/// let cache = ReductionBufferCache::new(&device, 4);
///
/// // First acquire allocates a new buffer
/// let buffer = cache.acquire::<f32>(4, ReductionOp::Sum)?;
/// // buffer is automatically returned to cache on drop
///
/// // Second acquire reuses the cached buffer
/// let buffer2 = cache.acquire::<f32>(4, ReductionOp::Sum)?;
/// ```
pub struct ReductionBufferCache {
    /// Parent device.
    device: CudaDevice,
    /// Cached buffers keyed by (num_slots, op).
    /// Uses Box<dyn Any + Send> for type erasure.
    cache: Mutex<HashMap<CacheKey, Vec<Box<dyn Any + Send>>>>,
    /// Maximum buffers to cache per key.
    max_per_key: usize,
    /// Cache statistics.
    stats: CacheStats,
}

impl ReductionBufferCache {
    /// Create a new reduction buffer cache.
    ///
    /// # Arguments
    ///
    /// * `device` - CUDA device for buffer allocation
    /// * `max_per_key` - Maximum buffers to cache per (num_slots, op) key
    pub fn new(device: &CudaDevice, max_per_key: usize) -> Self {
        Self {
            device: device.clone(),
            cache: Mutex::new(HashMap::new()),
            max_per_key: max_per_key.max(1),
            stats: CacheStats::new(),
        }
    }

    /// Acquire a reduction buffer, reusing from cache if available.
    ///
    /// The returned `CachedReductionBuffer` will automatically return
    /// the buffer to the cache when dropped.
    ///
    /// # Arguments
    ///
    /// * `num_slots` - Number of accumulation slots
    /// * `op` - Reduction operation type
    ///
    /// # Type Parameters
    ///
    /// * `T` - Scalar type (f32, f64, i32, etc.)
    pub fn acquire<T: ReductionScalar + 'static>(
        &self,
        num_slots: usize,
        op: ReductionOp,
    ) -> Result<CachedReductionBuffer<'_, T>> {
        let key = CacheKey::new(num_slots, op);

        // Try to get from cache
        let buffer = {
            let mut cache = self.cache.lock().unwrap();
            cache.get_mut(&key).and_then(|v| v.pop())
        };

        let buffer = if let Some(boxed) = buffer {
            // Cache hit - downcast the buffer
            self.stats.hits.fetch_add(1, AtomicOrdering::Relaxed);
            match boxed.downcast::<ReductionBuffer<T>>() {
                Ok(buf) => {
                    let buf = *buf;
                    // Reset to identity before returning
                    buf.reset()?;
                    buf
                }
                Err(_) => {
                    // Type mismatch (shouldn't happen with correct usage)
                    // Fall back to allocation
                    self.stats.misses.fetch_add(1, AtomicOrdering::Relaxed);
                    ReductionBuffer::with_slots(&self.device, num_slots, op)?
                }
            }
        } else {
            // Cache miss - allocate new buffer
            self.stats.misses.fetch_add(1, AtomicOrdering::Relaxed);
            ReductionBuffer::with_slots(&self.device, num_slots, op)?
        };

        Ok(CachedReductionBuffer {
            buffer: Some(buffer),
            cache: self,
            key,
        })
    }

    /// Return a buffer to the cache.
    ///
    /// This is called automatically by `CachedReductionBuffer::drop()`.
    fn return_buffer<T: ReductionScalar + 'static>(&self, key: CacheKey, buffer: ReductionBuffer<T>) {
        let mut cache = self.cache.lock().unwrap();
        let entry = cache.entry(key).or_default();

        if entry.len() < self.max_per_key {
            entry.push(Box::new(buffer));
            self.stats.returns.fetch_add(1, AtomicOrdering::Relaxed);
        }
        // If cache is full, buffer is dropped (memory freed)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get the number of cached buffers for a given key.
    pub fn cached_count(&self, num_slots: usize, op: ReductionOp) -> usize {
        let key = CacheKey::new(num_slots, op);
        let cache = self.cache.lock().unwrap();
        cache.get(&key).map(|v| v.len()).unwrap_or(0)
    }

    /// Get the total number of cached buffers across all keys.
    pub fn total_cached(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.values().map(|v| v.len()).sum()
    }

    /// Clear all cached buffers.
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }

    /// Trim cache to at most `max` buffers per key.
    pub fn trim_to(&self, max: usize) {
        let mut cache = self.cache.lock().unwrap();
        for entry in cache.values_mut() {
            while entry.len() > max {
                entry.pop();
            }
        }
    }
}

/// RAII wrapper that returns a reduction buffer to the cache on drop.
///
/// This struct implements `Deref` and `DerefMut` to provide transparent
/// access to the underlying `ReductionBuffer<T>`.
pub struct CachedReductionBuffer<'a, T: ReductionScalar + 'static> {
    /// The wrapped buffer (Option for take-on-drop pattern).
    buffer: Option<ReductionBuffer<T>>,
    /// Reference to parent cache.
    cache: &'a ReductionBufferCache,
    /// Cache key for returning.
    key: CacheKey,
}

impl<T: ReductionScalar + 'static> Deref for CachedReductionBuffer<'_, T> {
    type Target = ReductionBuffer<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().unwrap()
    }
}

impl<T: ReductionScalar + 'static> DerefMut for CachedReductionBuffer<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer.as_mut().unwrap()
    }
}

impl<T: ReductionScalar + 'static> Drop for CachedReductionBuffer<'_, T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.cache.return_buffer(self.key, buffer);
        }
    }
}

impl<T: ReductionScalar + 'static> CachedReductionBuffer<'_, T> {
    /// Get the device pointer for kernel parameter passing.
    pub fn device_ptr(&self) -> u64 {
        self.buffer.as_ref().unwrap().device_ptr()
    }

    /// Get number of slots.
    pub fn num_slots(&self) -> usize {
        self.buffer.as_ref().unwrap().num_slots()
    }

    /// Get the reduction operation.
    pub fn op(&self) -> ReductionOp {
        self.buffer.as_ref().unwrap().op()
    }

    /// Take ownership of the buffer, removing it from cache management.
    ///
    /// After calling this, the buffer will NOT be returned to the cache
    /// when this wrapper is dropped.
    pub fn take(mut self) -> ReductionBuffer<T> {
        self.buffer.take().unwrap()
    }
}

/// Generate CUDA kernel code for block-level sum reduction.
///
/// This produces a device function that can be called from your kernel.
///
/// # Parameters
///
/// * `func_name` - Name of the generated function
/// * `value_type` - CUDA type name (e.g., "float", "double")
/// * `block_size` - Thread block size (must be power of 2)
///
/// # Generated Signature
///
/// ```cuda
/// __device__ {type} {func_name}({type} val, {type}* shared, int block_size);
/// ```
pub fn generate_block_reduce_code(func_name: &str, value_type: &str, _block_size: u32) -> String {
    format!(
        r#"
// Block-level parallel reduction for {value_type}
__device__ {value_type} {func_name}({value_type} val, {value_type}* shared, int block_size) {{
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    // Tree reduction
    #pragma unroll
    for (int s = block_size / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared[tid] += shared[tid + s];
        }}
        __syncthreads();
    }}

    return shared[0];
}}
"#,
        func_name = func_name,
        value_type = value_type
    )
}

/// Generate CUDA kernel code for grid-level reduction with atomic accumulation.
///
/// This produces a device function that performs block-level reduction,
/// then atomically adds to a global accumulator.
///
/// # Parameters
///
/// * `func_name` - Name of the generated function
/// * `value_type` - CUDA type name
/// * `block_size` - Thread block size
///
/// # Generated Signature
///
/// ```cuda
/// __device__ void {func_name}({type} val, {type}* shared, {type}* accumulator);
/// ```
pub fn generate_grid_reduce_code(func_name: &str, value_type: &str, block_size: u32) -> String {
    format!(
        r#"
// Grid-level reduction with atomic accumulation
__device__ void {func_name}({value_type} val, {value_type}* shared, {value_type}* accumulator) {{
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    // Block-level tree reduction
    #pragma unroll
    for (int s = {block_size} / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared[tid] += shared[tid + s];
        }}
        __syncthreads();
    }}

    // Block leader atomically adds to global accumulator
    if (tid == 0) {{
        atomicAdd(accumulator, shared[0]);
    }}
}}
"#,
        func_name = func_name,
        value_type = value_type,
        block_size = block_size
    )
}

/// Generate CUDA kernel code for reduce-and-broadcast pattern.
///
/// This is the key pattern for PageRank dangling node handling:
/// 1. Each thread contributes a value
/// 2. Block-level reduction aggregates within block
/// 3. Block leader atomically adds to global accumulator
/// 4. Grid-wide sync (cooperative or software barrier)
/// 5. All threads read the final result
///
/// # Parameters
///
/// * `func_name` - Name of the generated function
/// * `value_type` - CUDA type name
/// * `block_size` - Thread block size
/// * `use_cooperative` - Use cooperative groups for grid sync
///
/// # Generated Signature
///
/// ```cuda
/// __device__ {type} {func_name}({type} val, {type}* shared, {type}* accumulator,
///                               [cg::grid_group& grid | uint32_t* barrier_counter, uint32_t* barrier_gen]);
/// ```
pub fn generate_reduce_and_broadcast_code(
    func_name: &str,
    value_type: &str,
    block_size: u32,
    use_cooperative: bool,
) -> String {
    let grid_sync_code = if use_cooperative {
        "grid.sync();"
    } else {
        r#"// Software grid barrier
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int gen = *barrier_gen;
            unsigned int arrived = atomicAdd(barrier_counter, 1) + 1;
            if (arrived == gridDim.x * gridDim.y * gridDim.z) {
                *barrier_counter = 0;
                __threadfence();
                atomicAdd(barrier_gen, 1);
            } else {
                while (atomicAdd(barrier_gen, 0) == gen) {
                    __threadfence();
                }
            }
        }
        __syncthreads();"#
    };

    let params = if use_cooperative {
        format!(
            "{} val, {}* shared, {}* accumulator, cg::grid_group& grid",
            value_type, value_type, value_type
        )
    } else {
        format!(
            "{} val, {}* shared, {}* accumulator, unsigned int* barrier_counter, unsigned int* barrier_gen",
            value_type, value_type, value_type
        )
    };

    format!(
        r#"
// Reduce-and-broadcast: all threads get the global sum
__device__ {value_type} {func_name}({params}) {{
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    // Phase 1: Block-level reduction
    #pragma unroll
    for (int s = {block_size} / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared[tid] += shared[tid + s];
        }}
        __syncthreads();
    }}

    // Phase 2: Block leader atomically adds to accumulator
    if (tid == 0) {{
        atomicAdd(accumulator, shared[0]);
    }}

    // Phase 3: Grid-wide synchronization
    {grid_sync_code}

    // Phase 4: All threads read the result
    return *accumulator;
}}
"#,
        func_name = func_name,
        value_type = value_type,
        block_size = block_size,
        params = params,
        grid_sync_code = grid_sync_code
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_block_reduce() {
        let code = generate_block_reduce_code("block_sum", "float", 256);
        assert!(code.contains("__device__ float block_sum"));
        assert!(code.contains("shared[tid] += shared[tid + s]"));
        assert!(code.contains("__syncthreads()"));
    }

    #[test]
    fn test_generate_grid_reduce() {
        let code = generate_grid_reduce_code("grid_sum", "double", 128);
        assert!(code.contains("__device__ void grid_sum"));
        assert!(code.contains("atomicAdd(accumulator, shared[0])"));
    }

    #[test]
    fn test_generate_reduce_and_broadcast_cooperative() {
        let code = generate_reduce_and_broadcast_code("reduce_broadcast", "float", 256, true);
        assert!(code.contains("cg::grid_group& grid"));
        assert!(code.contains("grid.sync()"));
    }

    #[test]
    fn test_generate_reduce_and_broadcast_software() {
        let code = generate_reduce_and_broadcast_code("reduce_broadcast", "float", 256, false);
        assert!(code.contains("barrier_counter"));
        assert!(code.contains("barrier_gen"));
        assert!(code.contains("atomicAdd(barrier_counter, 1)"));
    }

    // --- CacheStats tests ---

    #[test]
    fn test_cache_stats_new() {
        let stats = CacheStats::new();
        assert_eq!(stats.hits(), 0);
        assert_eq!(stats.misses(), 0);
        assert_eq!(stats.returns(), 0);
    }

    #[test]
    fn test_cache_stats_hit_rate_empty() {
        let stats = CacheStats::new();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats::new();
        stats.hits.fetch_add(3, AtomicOrdering::Relaxed);
        stats.misses.fetch_add(1, AtomicOrdering::Relaxed);
        assert!((stats.hit_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_counters() {
        let stats = CacheStats::new();
        stats.hits.fetch_add(10, AtomicOrdering::Relaxed);
        stats.misses.fetch_add(5, AtomicOrdering::Relaxed);
        stats.returns.fetch_add(8, AtomicOrdering::Relaxed);

        assert_eq!(stats.hits(), 10);
        assert_eq!(stats.misses(), 5);
        assert_eq!(stats.returns(), 8);
    }

    // --- CacheKey tests ---

    #[test]
    fn test_cache_key_equality() {
        let key1 = CacheKey::new(4, ReductionOp::Sum);
        let key2 = CacheKey::new(4, ReductionOp::Sum);
        let key3 = CacheKey::new(4, ReductionOp::Max);
        let key4 = CacheKey::new(8, ReductionOp::Sum);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_cache_key_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(CacheKey::new(4, ReductionOp::Sum));
        set.insert(CacheKey::new(4, ReductionOp::Max));
        set.insert(CacheKey::new(8, ReductionOp::Sum));
        set.insert(CacheKey::new(4, ReductionOp::Sum)); // Duplicate

        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_cache_key_different_ops() {
        // Verify all reduction ops create distinct keys
        let ops = [
            ReductionOp::Sum,
            ReductionOp::Min,
            ReductionOp::Max,
            ReductionOp::And,
            ReductionOp::Or,
            ReductionOp::Xor,
            ReductionOp::Product,
        ];

        let keys: Vec<_> = ops.iter().map(|&op| CacheKey::new(1, op)).collect();

        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j], "Keys for {:?} and {:?} should differ", ops[i], ops[j]);
            }
        }
    }
}
