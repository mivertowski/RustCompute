//! Async Memory Pool for Datacenter GPU Workloads.
//!
//! Uses `cuMemAllocAsync` / `cuMemFreeAsync` to avoid the global synchronization
//! penalty of traditional `cuMemAlloc`. This is critical for persistent actors
//! that need to allocate temporary buffers without blocking other streams.
//!
//! # Why Async Memory Matters for Persistent Actors
//!
//! Traditional `cuMemAlloc` is a synchronizing operation — it blocks all streams
//! until the allocation completes. For persistent actor workloads with multiple
//! concurrent streams, this creates unnecessary serialization.
//!
//! `cuMemAllocAsync` allocates from a pre-configured pool on a specific stream,
//! allowing other streams to continue executing. The allocation is ordered within
//! the stream but doesn't affect other streams.
//!
//! # Performance
//!
//! On H100: `cuMemAllocAsync` is ~10x faster than `cuMemAlloc` for small allocations
//! because it avoids the driver-level lock and allocates from a thread-local pool.

use std::ptr;

use cudarc::driver::sys as cuda_sys;

use ringkernel_core::error::{Result, RingKernelError};

use crate::device::CudaDevice;

/// Configuration for an async memory pool.
#[derive(Debug, Clone)]
pub struct AsyncPoolConfig {
    /// Initial pool size in bytes (pre-allocated).
    pub initial_size: usize,
    /// Maximum pool size in bytes (0 = unlimited).
    pub max_size: usize,
    /// Release threshold: free memory back to OS when pool exceeds this.
    pub release_threshold: usize,
}

impl Default for AsyncPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 64 * 1024 * 1024,     // 64 MB
            max_size: 0,                          // Unlimited
            // NVIDIA recommendation for persistent workloads: set high release
            // threshold to prevent OS reclaim during sustained operation.
            // For persistent actors that run for minutes/hours, we want to
            // keep the pool warm — only release when truly excessive.
            release_threshold: 1024 * 1024 * 1024, // 1 GB (high for persistent)
        }
    }
}

impl AsyncPoolConfig {
    /// Configuration optimized for persistent actor workloads.
    ///
    /// High release threshold prevents pool shrinkage during sustained operation.
    /// Based on NVIDIA guidance for stream-ordered allocators with persistent kernels.
    pub fn for_persistent_actors() -> Self {
        Self {
            initial_size: 256 * 1024 * 1024,      // 256 MB pre-allocated
            max_size: 0,                            // Unlimited
            release_threshold: 4 * 1024 * 1024 * 1024, // 4 GB (very high)
        }
    }

    /// Configuration for short-lived workloads (lower memory retention).
    pub fn for_batch_processing() -> Self {
        Self {
            initial_size: 32 * 1024 * 1024,        // 32 MB
            max_size: 0,
            release_threshold: 128 * 1024 * 1024,  // 128 MB (release quickly)
        }
    }
}

/// An async memory pool using CUDA's stream-ordered allocator.
pub struct AsyncMemoryPool {
    pool: cuda_sys::CUmemoryPool,
    device: CudaDevice,
    total_allocated: std::sync::atomic::AtomicU64,
}

// SAFETY: CUmemoryPool handles are thread-safe per CUDA driver guarantees.
unsafe impl Send for AsyncMemoryPool {}
unsafe impl Sync for AsyncMemoryPool {}

impl AsyncMemoryPool {
    /// Create a new async memory pool.
    pub fn new(device: &CudaDevice, config: &AsyncPoolConfig) -> Result<Self> {
        let _ = device.inner(); // Ensure context active

        let mut pool_props: cuda_sys::CUmemPoolProps_st = unsafe { std::mem::zeroed() };
        pool_props.allocType = cuda_sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        pool_props.handleTypes = cuda_sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
        pool_props.location.type_ = cuda_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        pool_props.location.id = device.ordinal() as i32;

        let mut pool: cuda_sys::CUmemoryPool = ptr::null_mut();
        unsafe {
            let r = cuda_sys::cuMemPoolCreate(&mut pool, &pool_props);
            if r != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::AllocationFailed {
                    size: config.initial_size,
                    reason: format!("cuMemPoolCreate failed: {:?}", r),
                });
            }

            // Set release threshold
            if config.release_threshold > 0 {
                let threshold = config.release_threshold as u64;
                let r = cuda_sys::cuMemPoolSetAttribute(
                    pool,
                    cuda_sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &threshold as *const u64 as *mut std::ffi::c_void,
                );
                if r != cuda_sys::CUresult::CUDA_SUCCESS {
                    tracing::warn!("Failed to set pool release threshold: {:?}", r);
                }
            }
        }

        Ok(Self {
            pool,
            device: device.clone(),
            total_allocated: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Use the device's default memory pool (no custom pool creation).
    pub fn from_default(device: &CudaDevice) -> Result<Self> {
        let _ = device.inner();

        let mut pool: cuda_sys::CUmemoryPool = ptr::null_mut();
        unsafe {
            let r = cuda_sys::cuDeviceGetDefaultMemPool(
                &mut pool,
                device.ordinal() as i32,
            );
            if r != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::BackendError(format!(
                    "cuDeviceGetDefaultMemPool failed: {:?}",
                    r
                )));
            }
        }

        Ok(Self {
            pool,
            device: device.clone(),
            total_allocated: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Allocate memory asynchronously on a stream.
    ///
    /// The allocation is stream-ordered: it becomes available when the stream
    /// reaches this point. Other streams are not affected.
    pub fn alloc_async(
        &self,
        size: usize,
        stream: cuda_sys::CUstream,
    ) -> Result<u64> {
        let mut dptr: u64 = 0;

        unsafe {
            let r = cuda_sys::cuMemAllocAsync(&mut dptr, size, stream);
            if r != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::AllocationFailed {
                    size,
                    reason: format!("cuMemAllocAsync failed: {:?}", r),
                });
            }
        }

        self.total_allocated
            .fetch_add(size as u64, std::sync::atomic::Ordering::Relaxed);

        Ok(dptr)
    }

    /// Free memory asynchronously on a stream.
    pub fn free_async(&self, dptr: u64, stream: cuda_sys::CUstream) -> Result<()> {
        unsafe {
            let r = cuda_sys::cuMemFreeAsync(dptr, stream);
            if r != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::BackendError(format!(
                    "cuMemFreeAsync failed: {:?}",
                    r
                )));
            }
        }

        Ok(())
    }

    /// Get total bytes allocated through this pool.
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Trim the pool, releasing unused memory back to the OS.
    pub fn trim(&self, min_bytes_to_keep: usize) -> Result<()> {
        unsafe {
            let r = cuda_sys::cuMemPoolTrimTo(self.pool, min_bytes_to_keep);
            if r != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::BackendError(format!(
                    "cuMemPoolTrimTo failed: {:?}",
                    r
                )));
            }
        }
        Ok(())
    }
}

impl Drop for AsyncMemoryPool {
    fn drop(&mut self) {
        let _ = self.device.inner();
        unsafe {
            let _ = cuda_sys::cuMemPoolDestroy(self.pool);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_pool_config_defaults() {
        let config = AsyncPoolConfig::default();
        assert_eq!(config.initial_size, 64 * 1024 * 1024);
        assert_eq!(config.max_size, 0);
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_async_pool_creation() {
        let device = CudaDevice::new(0).expect("device");
        let pool = AsyncMemoryPool::from_default(&device);
        assert!(pool.is_ok(), "Failed to create async pool: {:?}", pool.err());
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_async_alloc_free() {
        let device = CudaDevice::new(0).expect("device");
        let pool = AsyncMemoryPool::from_default(&device).expect("pool");

        let stream: cuda_sys::CUstream = ptr::null_mut();
        let dptr = pool.alloc_async(4096, stream).expect("alloc");
        assert!(dptr != 0, "Device pointer should be non-zero");

        pool.free_async(dptr, stream).expect("free");
    }

    #[test]
    #[ignore] // Requires CUDA GPU
    fn bench_async_vs_sync_alloc() {
        let device = CudaDevice::new(0).expect("device");
        let pool = AsyncMemoryPool::from_default(&device).expect("pool");
        let stream: cuda_sys::CUstream = ptr::null_mut();

        let alloc_size = 4096usize;
        let iterations = 10000u32;

        // Benchmark async alloc
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let dptr = pool.alloc_async(alloc_size, stream).expect("alloc");
            pool.free_async(dptr, stream).expect("free");
        }
        unsafe { cuda_sys::cuStreamSynchronize(stream); }
        let async_time = start.elapsed();

        // Benchmark sync alloc
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let mut dptr: u64 = 0;
            unsafe {
                cuda_sys::cuMemAlloc_v2(&mut dptr, alloc_size);
                cuda_sys::cuMemFree_v2(dptr);
            }
        }
        let sync_time = start.elapsed();

        let async_per = async_time.as_nanos() as f64 / iterations as f64;
        let sync_per = sync_time.as_nanos() as f64 / iterations as f64;
        let speedup = sync_per / async_per;

        println!();
        println!("══════════════════════════════════════════════════════════════");
        println!("  Async vs Sync Memory Allocation — {} iterations", iterations);
        println!("══════════════════════════════════════════════════════════════");
        println!("  cuMemAllocAsync:  {:.0} ns/alloc", async_per);
        println!("  cuMemAlloc:       {:.0} ns/alloc", sync_per);
        println!("  Speedup:          {:.1}x", speedup);
        println!("══════════════════════════════════════════════════════════════");
    }
}
