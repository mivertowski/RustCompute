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

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::{fence, Ordering};

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
}
