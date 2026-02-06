//! DSL marker functions for WGSL code generation.
//!
//! These functions are compile-time markers that the transpiler recognizes
//! and converts to WGSL intrinsics. They do nothing at Rust compile time.
//!
//! Common functions (thread indices, basic math, synchronization) are shared with
//! the CUDA backend via `ringkernel_codegen::dsl_common`.
//!
//! # Thread/Workgroup Indices
//!
//! | Function | WGSL Equivalent |
//! |----------|-----------------|
//! | `thread_idx_x()` | `local_invocation_id.x` |
//! | `thread_idx_y()` | `local_invocation_id.y` |
//! | `thread_idx_z()` | `local_invocation_id.z` |
//! | `block_idx_x()` | `workgroup_id.x` |
//! | `block_idx_y()` | `workgroup_id.y` |
//! | `block_idx_z()` | `workgroup_id.z` |
//! | `block_dim_x()` | `WORKGROUP_SIZE_X` (constant) |
//! | `global_thread_id()` | `global_invocation_id.x` |

#![allow(unused_variables)]

// Re-export common DSL functions shared across backends.
// These provide: thread/block indices, sync primitives, and basic math.
pub use ringkernel_codegen::dsl_common::{
    block_dim_x,
    block_dim_y,
    block_dim_z,
    block_idx_x,
    block_idx_y,
    block_idx_z,
    ceil,
    cos,
    exp,
    floor,
    fma,
    grid_dim_x,
    grid_dim_y,
    grid_dim_z,
    log,
    powf,
    round,
    rsqrt,
    sin,
    // Math
    sqrt,
    // Synchronization
    sync_threads,
    tan,
    thread_fence,
    thread_fence_block,
    // Thread/block indices
    thread_idx_x,
    thread_idx_y,
    thread_idx_z,
};

// =============================================================================
// WGSL-Specific Thread/Workgroup Indices
// =============================================================================

/// Get the global thread ID (x-component).
/// Maps to `global_invocation_id.x` in WGSL.
#[inline(always)]
pub fn global_thread_id() -> i32 {
    0
}

/// Get the global thread ID in the y-dimension.
/// Maps to `global_invocation_id.y` in WGSL.
#[inline(always)]
pub fn global_thread_id_y() -> i32 {
    0
}

/// Get the global thread ID in the z-dimension.
/// Maps to `global_invocation_id.z` in WGSL.
#[inline(always)]
pub fn global_thread_id_z() -> i32 {
    0
}

// =============================================================================
// Atomic Operations (32-bit)
// =============================================================================

/// Atomic add. Maps to `atomicAdd(&var, value)` in WGSL.
#[inline(always)]
pub fn atomic_add<T>(_ptr: &T, _val: T) -> T {
    unimplemented!("atomic_add is a transpiler intrinsic")
}

/// Atomic subtract. Maps to `atomicSub(&var, value)` in WGSL.
#[inline(always)]
pub fn atomic_sub<T>(_ptr: &T, _val: T) -> T {
    unimplemented!("atomic_sub is a transpiler intrinsic")
}

/// Atomic minimum. Maps to `atomicMin(&var, value)` in WGSL.
#[inline(always)]
pub fn atomic_min<T>(_ptr: &T, _val: T) -> T {
    unimplemented!("atomic_min is a transpiler intrinsic")
}

/// Atomic maximum. Maps to `atomicMax(&var, value)` in WGSL.
#[inline(always)]
pub fn atomic_max<T>(_ptr: &T, _val: T) -> T {
    unimplemented!("atomic_max is a transpiler intrinsic")
}

/// Atomic exchange. Maps to `atomicExchange(&var, value)` in WGSL.
#[inline(always)]
pub fn atomic_exchange<T>(_ptr: &T, _val: T) -> T {
    unimplemented!("atomic_exchange is a transpiler intrinsic")
}

/// Atomic compare-and-swap. Maps to `atomicCompareExchangeWeak(&var, compare, value)` in WGSL.
/// Returns the old value.
#[inline(always)]
pub fn atomic_cas<T>(_ptr: &T, _compare: T, _val: T) -> T {
    unimplemented!("atomic_cas is a transpiler intrinsic")
}

/// Atomic load. Maps to `atomicLoad(&var)` in WGSL.
#[inline(always)]
pub fn atomic_load<T>(_ptr: &T) -> T {
    unimplemented!("atomic_load is a transpiler intrinsic")
}

/// Atomic store. Maps to `atomicStore(&var, value)` in WGSL.
#[inline(always)]
pub fn atomic_store<T>(_ptr: &T, _val: T) {
    unimplemented!("atomic_store is a transpiler intrinsic")
}

// =============================================================================
// WGSL-Specific Math Functions
// =============================================================================

/// Absolute value. Maps to `abs(x)` in WGSL.
#[inline(always)]
pub fn abs<T: num_traits::Signed>(x: T) -> T {
    x.abs()
}

/// Minimum. Maps to `min(a, b)` in WGSL.
#[inline(always)]
pub fn min<T: Ord>(a: T, b: T) -> T {
    std::cmp::min(a, b)
}

/// Maximum. Maps to `max(a, b)` in WGSL.
#[inline(always)]
pub fn max<T: Ord>(a: T, b: T) -> T {
    std::cmp::max(a, b)
}

/// Clamp. Maps to `clamp(x, lo, hi)` in WGSL.
#[inline(always)]
pub fn clamp<T: Ord>(x: T, lo: T, hi: T) -> T {
    std::cmp::max(lo, std::cmp::min(x, hi))
}

/// Linear interpolation. Maps to `mix(a, b, t)` in WGSL.
#[inline(always)]
pub fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

// =============================================================================
// Subgroup Operations (require WGSL extensions)
// =============================================================================

/// Shuffle within subgroup. Maps to `subgroupShuffle(value, lane)` in WGSL.
/// Requires: `enable chromium_experimental_subgroups;`
#[inline(always)]
pub fn warp_shuffle<T>(_val: T, _lane: u32) -> T {
    unimplemented!("warp_shuffle requires WGSL subgroup extension")
}

/// Shuffle up within subgroup. Maps to `subgroupShuffleUp(value, delta)` in WGSL.
#[inline(always)]
pub fn warp_shuffle_up<T>(_val: T, _delta: u32) -> T {
    unimplemented!("warp_shuffle_up requires WGSL subgroup extension")
}

/// Shuffle down within subgroup. Maps to `subgroupShuffleDown(value, delta)` in WGSL.
#[inline(always)]
pub fn warp_shuffle_down<T>(_val: T, _delta: u32) -> T {
    unimplemented!("warp_shuffle_down requires WGSL subgroup extension")
}

/// Shuffle XOR within subgroup. Maps to `subgroupShuffleXor(value, mask)` in WGSL.
#[inline(always)]
pub fn warp_shuffle_xor<T>(_val: T, _mask: u32) -> T {
    unimplemented!("warp_shuffle_xor requires WGSL subgroup extension")
}

/// Ballot within subgroup. Maps to `subgroupBallot(predicate)` in WGSL.
/// Returns a vec4<u32> where each bit represents a lane's predicate.
#[inline(always)]
pub fn warp_ballot(_pred: bool) -> [u32; 4] {
    unimplemented!("warp_ballot requires WGSL subgroup extension")
}

/// Check if all lanes in subgroup satisfy predicate. Maps to `subgroupAll(predicate)` in WGSL.
#[inline(always)]
pub fn warp_all(_pred: bool) -> bool {
    unimplemented!("warp_all requires WGSL subgroup extension")
}

/// Check if any lane in subgroup satisfies predicate. Maps to `subgroupAny(predicate)` in WGSL.
#[inline(always)]
pub fn warp_any(_pred: bool) -> bool {
    unimplemented!("warp_any requires WGSL subgroup extension")
}

/// Get the lane ID within the subgroup.
/// Maps to `subgroup_invocation_id` builtin in WGSL.
#[inline(always)]
pub fn lane_id() -> u32 {
    unimplemented!("lane_id requires WGSL subgroup extension")
}

/// Get the subgroup size.
/// Maps to `subgroup_size` builtin in WGSL.
#[inline(always)]
pub fn warp_size() -> u32 {
    unimplemented!("warp_size requires WGSL subgroup extension")
}

// Workaround: num_traits isn't in dependencies, so use a simple trait
mod num_traits {
    pub trait Signed {
        fn abs(self) -> Self;
    }

    impl Signed for f32 {
        fn abs(self) -> Self {
            f32::abs(self)
        }
    }

    impl Signed for i32 {
        fn abs(self) -> Self {
            i32::abs(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_functions() {
        // These should just pass through to Rust's implementations
        assert_eq!(sqrt(4.0), 2.0);
        assert_eq!(floor(3.7), 3.0);
        assert_eq!(ceil(3.2), 4.0);
    }

    #[test]
    fn test_thread_indices_compile() {
        // These just need to compile - they return 0 as placeholders
        let _ = thread_idx_x();
        let _ = block_idx_x();
        let _ = block_dim_x();
    }

    #[test]
    fn test_sync_compiles() {
        // These just need to compile
        sync_threads();
        thread_fence();
    }
}
