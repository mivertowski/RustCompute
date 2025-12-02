//! Rust DSL functions for writing CUDA kernels.
//!
//! This module provides Rust functions that map to CUDA intrinsics during transpilation.
//! These functions have CPU fallback implementations for testing but are transpiled
//! to the corresponding CUDA operations when used in kernel code.
//!
//! # Thread/Block Index Access
//!
//! ```ignore
//! use ringkernel_cuda_codegen::dsl::*;
//!
//! fn my_kernel(...) {
//!     let tx = thread_idx_x();  // -> threadIdx.x
//!     let bx = block_idx_x();   // -> blockIdx.x
//!     let idx = bx * block_dim_x() + tx;  // Global thread index
//! }
//! ```
//!
//! # Thread Synchronization
//!
//! ```ignore
//! sync_threads();  // -> __syncthreads()
//! ```

/// Get the thread index within a block (x dimension).
/// Transpiles to: `threadIdx.x`
#[inline]
pub fn thread_idx_x() -> i32 {
    // CPU fallback: single-threaded execution uses index 0
    0
}

/// Get the thread index within a block (y dimension).
/// Transpiles to: `threadIdx.y`
#[inline]
pub fn thread_idx_y() -> i32 {
    0
}

/// Get the thread index within a block (z dimension).
/// Transpiles to: `threadIdx.z`
#[inline]
pub fn thread_idx_z() -> i32 {
    0
}

/// Get the block index within a grid (x dimension).
/// Transpiles to: `blockIdx.x`
#[inline]
pub fn block_idx_x() -> i32 {
    0
}

/// Get the block index within a grid (y dimension).
/// Transpiles to: `blockIdx.y`
#[inline]
pub fn block_idx_y() -> i32 {
    0
}

/// Get the block index within a grid (z dimension).
/// Transpiles to: `blockIdx.z`
#[inline]
pub fn block_idx_z() -> i32 {
    0
}

/// Get the block dimension (x dimension).
/// Transpiles to: `blockDim.x`
#[inline]
pub fn block_dim_x() -> i32 {
    1
}

/// Get the block dimension (y dimension).
/// Transpiles to: `blockDim.y`
#[inline]
pub fn block_dim_y() -> i32 {
    1
}

/// Get the block dimension (z dimension).
/// Transpiles to: `blockDim.z`
#[inline]
pub fn block_dim_z() -> i32 {
    1
}

/// Get the grid dimension (x dimension).
/// Transpiles to: `gridDim.x`
#[inline]
pub fn grid_dim_x() -> i32 {
    1
}

/// Get the grid dimension (y dimension).
/// Transpiles to: `gridDim.y`
#[inline]
pub fn grid_dim_y() -> i32 {
    1
}

/// Get the grid dimension (z dimension).
/// Transpiles to: `gridDim.z`
#[inline]
pub fn grid_dim_z() -> i32 {
    1
}

/// Synchronize all threads in a block.
/// Transpiles to: `__syncthreads()`
#[inline]
pub fn sync_threads() {
    // CPU fallback: no-op (single-threaded)
}

/// Thread memory fence.
/// Transpiles to: `__threadfence()`
#[inline]
pub fn thread_fence() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

/// Block-level memory fence.
/// Transpiles to: `__threadfence_block()`
#[inline]
pub fn thread_fence_block() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_indices_default() {
        assert_eq!(thread_idx_x(), 0);
        assert_eq!(thread_idx_y(), 0);
        assert_eq!(thread_idx_z(), 0);
    }

    #[test]
    fn test_block_indices_default() {
        assert_eq!(block_idx_x(), 0);
        assert_eq!(block_idx_y(), 0);
        assert_eq!(block_idx_z(), 0);
    }

    #[test]
    fn test_dimensions_default() {
        assert_eq!(block_dim_x(), 1);
        assert_eq!(block_dim_y(), 1);
        assert_eq!(grid_dim_x(), 1);
    }
}
