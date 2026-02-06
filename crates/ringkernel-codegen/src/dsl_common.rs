//! Common DSL marker functions shared across GPU backends.
//!
//! These functions provide CPU-fallback implementations of GPU intrinsics.
//! During transpilation, the CUDA and WGSL code generators recognize these
//! function names and replace them with their backend-specific equivalents.
//!
//! # Thread/Block Indices
//!
//! | Function | CUDA | WGSL |
//! |----------|------|------|
//! | `thread_idx_x()` | `threadIdx.x` | `local_invocation_id.x` |
//! | `block_idx_x()` | `blockIdx.x` | `workgroup_id.x` |
//! | `block_dim_x()` | `blockDim.x` | `WORKGROUP_SIZE_X` |
//! | `grid_dim_x()` | `gridDim.x` | `num_workgroups.x` |
//!
//! # Math Functions
//!
//! All math functions use Rust's standard library for CPU fallback:
//! `sqrt`, `rsqrt`, `floor`, `ceil`, `round`, `sin`, `cos`, `tan`,
//! `exp`, `log`, `fma`.

// =============================================================================
// Thread/Block Index Functions
// =============================================================================

/// Get the thread index within a block/workgroup (x dimension).
///
/// - CUDA: `threadIdx.x`
/// - WGSL: `local_invocation_id.x`
#[inline(always)]
pub fn thread_idx_x() -> i32 {
    0 // CPU fallback: single-threaded execution
}

/// Get the thread index within a block/workgroup (y dimension).
///
/// - CUDA: `threadIdx.y`
/// - WGSL: `local_invocation_id.y`
#[inline(always)]
pub fn thread_idx_y() -> i32 {
    0
}

/// Get the thread index within a block/workgroup (z dimension).
///
/// - CUDA: `threadIdx.z`
/// - WGSL: `local_invocation_id.z`
#[inline(always)]
pub fn thread_idx_z() -> i32 {
    0
}

/// Get the block/workgroup index (x dimension).
///
/// - CUDA: `blockIdx.x`
/// - WGSL: `workgroup_id.x`
#[inline(always)]
pub fn block_idx_x() -> i32 {
    0
}

/// Get the block/workgroup index (y dimension).
///
/// - CUDA: `blockIdx.y`
/// - WGSL: `workgroup_id.y`
#[inline(always)]
pub fn block_idx_y() -> i32 {
    0
}

/// Get the block/workgroup index (z dimension).
///
/// - CUDA: `blockIdx.z`
/// - WGSL: `workgroup_id.z`
#[inline(always)]
pub fn block_idx_z() -> i32 {
    0
}

/// Get the block/workgroup dimension (x dimension).
///
/// - CUDA: `blockDim.x`
/// - WGSL: `WORKGROUP_SIZE_X`
#[inline(always)]
pub fn block_dim_x() -> i32 {
    1 // CPU fallback: one thread per block
}

/// Get the block/workgroup dimension (y dimension).
///
/// - CUDA: `blockDim.y`
/// - WGSL: `WORKGROUP_SIZE_Y`
#[inline(always)]
pub fn block_dim_y() -> i32 {
    1
}

/// Get the block/workgroup dimension (z dimension).
///
/// - CUDA: `blockDim.z`
/// - WGSL: `WORKGROUP_SIZE_Z`
#[inline(always)]
pub fn block_dim_z() -> i32 {
    1
}

/// Get the grid/dispatch dimension (x dimension).
///
/// - CUDA: `gridDim.x`
/// - WGSL: `num_workgroups.x`
#[inline(always)]
pub fn grid_dim_x() -> i32 {
    1 // CPU fallback: one block in the grid
}

/// Get the grid/dispatch dimension (y dimension).
///
/// - CUDA: `gridDim.y`
/// - WGSL: `num_workgroups.y`
#[inline(always)]
pub fn grid_dim_y() -> i32 {
    1
}

/// Get the grid/dispatch dimension (z dimension).
///
/// - CUDA: `gridDim.z`
/// - WGSL: `num_workgroups.z`
#[inline(always)]
pub fn grid_dim_z() -> i32 {
    1
}

// =============================================================================
// Synchronization Functions
// =============================================================================

/// Synchronize all threads in a block/workgroup.
///
/// - CUDA: `__syncthreads()`
/// - WGSL: `workgroupBarrier()`
#[inline(always)]
pub fn sync_threads() {
    // CPU fallback: no-op (single-threaded)
}

/// Thread/storage memory fence.
///
/// - CUDA: `__threadfence()`
/// - WGSL: `storageBarrier()`
#[inline(always)]
pub fn thread_fence() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
}

/// Block-level/workgroup memory fence.
///
/// - CUDA: `__threadfence_block()`
/// - WGSL: `workgroupBarrier()`
#[inline(always)]
pub fn thread_fence_block() {
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
}

// =============================================================================
// Math Functions (f32)
// =============================================================================

/// Square root.
///
/// - CUDA: `sqrtf(x)`
/// - WGSL: `sqrt(x)`
#[inline(always)]
pub fn sqrt(x: f32) -> f32 {
    x.sqrt()
}

/// Reciprocal square root.
///
/// - CUDA: `rsqrtf(x)`
/// - WGSL: `inverseSqrt(x)`
#[inline(always)]
pub fn rsqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

/// Floor.
///
/// - CUDA: `floorf(x)`
/// - WGSL: `floor(x)`
#[inline(always)]
pub fn floor(x: f32) -> f32 {
    x.floor()
}

/// Ceiling.
///
/// - CUDA: `ceilf(x)`
/// - WGSL: `ceil(x)`
#[inline(always)]
pub fn ceil(x: f32) -> f32 {
    x.ceil()
}

/// Round to nearest.
///
/// - CUDA: `roundf(x)`
/// - WGSL: `round(x)`
#[inline(always)]
pub fn round(x: f32) -> f32 {
    x.round()
}

/// Sine.
///
/// - CUDA: `sinf(x)`
/// - WGSL: `sin(x)`
#[inline(always)]
pub fn sin(x: f32) -> f32 {
    x.sin()
}

/// Cosine.
///
/// - CUDA: `cosf(x)`
/// - WGSL: `cos(x)`
#[inline(always)]
pub fn cos(x: f32) -> f32 {
    x.cos()
}

/// Tangent.
///
/// - CUDA: `tanf(x)`
/// - WGSL: `tan(x)`
#[inline(always)]
pub fn tan(x: f32) -> f32 {
    x.tan()
}

/// Exponential (base e).
///
/// - CUDA: `expf(x)`
/// - WGSL: `exp(x)`
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    x.exp()
}

/// Natural logarithm (base e).
///
/// - CUDA: `logf(x)`
/// - WGSL: `log(x)`
#[inline(always)]
pub fn log(x: f32) -> f32 {
    x.ln()
}

/// Fused multiply-add: `a * b + c`.
///
/// - CUDA: `fmaf(a, b, c)`
/// - WGSL: `fma(a, b, c)`
#[inline(always)]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// Power: `x^y`.
///
/// - CUDA: `powf(x, y)`
/// - WGSL: `pow(x, y)`
#[inline(always)]
pub fn powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_indices() {
        assert_eq!(thread_idx_x(), 0);
        assert_eq!(thread_idx_y(), 0);
        assert_eq!(thread_idx_z(), 0);
    }

    #[test]
    fn test_block_indices() {
        assert_eq!(block_idx_x(), 0);
        assert_eq!(block_idx_y(), 0);
        assert_eq!(block_idx_z(), 0);
    }

    #[test]
    fn test_dimensions() {
        assert_eq!(block_dim_x(), 1);
        assert_eq!(block_dim_y(), 1);
        assert_eq!(block_dim_z(), 1);
        assert_eq!(grid_dim_x(), 1);
        assert_eq!(grid_dim_y(), 1);
        assert_eq!(grid_dim_z(), 1);
    }

    #[test]
    fn test_sync_compiles() {
        sync_threads();
        thread_fence();
        thread_fence_block();
    }

    #[test]
    fn test_math_functions() {
        assert!((sqrt(4.0) - 2.0).abs() < 1e-6);
        assert!((rsqrt(4.0) - 0.5).abs() < 1e-6);
        assert_eq!(floor(3.7), 3.0);
        assert_eq!(ceil(3.2), 4.0);
        assert_eq!(round(3.5), 4.0);
        assert!((sin(0.0)).abs() < 1e-6);
        assert!((cos(0.0) - 1.0).abs() < 1e-6);
        assert!((tan(0.0)).abs() < 1e-6);
        assert!((exp(0.0) - 1.0).abs() < 1e-6);
        assert!((log(1.0)).abs() < 1e-6);
        assert!((fma(2.0, 3.0, 1.0) - 7.0).abs() < 1e-6);
        assert!((powf(2.0, 3.0) - 8.0).abs() < 1e-6);
    }
}
