//! DSL marker functions for WGSL code generation.
//!
//! These functions are compile-time markers that the transpiler recognizes
//! and converts to WGSL intrinsics. They do nothing at Rust compile time.
//!
//! Common functions (thread indices, basic math, synchronization) are shared with
//! the CUDA backend via `ringkernel_codegen::dsl_common`.
//!
//! # Marker Function Pattern
//!
//! Functions in this module fall into two categories:
//!
//! 1. **Functions with CPU fallbacks** (thread indices, math): These return sensible
//!    defaults (e.g., `thread_idx_x()` returns 0) so kernel code can be tested on CPU.
//!    During transpilation, they are replaced with their WGSL equivalents.
//!
//! 2. **Transpiler-only markers** (atomics, subgroup ops): These panic if called directly
//!    on CPU because they have no meaningful single-threaded semantics. They exist solely
//!    as markers for the `ringkernel-wgpu-codegen` transpiler, which converts them to WGSL
//!    intrinsics. If you see a panic from one of these functions, it means the code path
//!    was executed on CPU without transpilation -- use the transpiler to generate WGSL.
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
//!
//! # Atomic Operations
//!
//! Atomic functions (`atomic_add`, `atomic_sub`, `atomic_min`, `atomic_max`,
//! `atomic_exchange`, `atomic_cas`, `atomic_load`, `atomic_store`) are transpiler-only
//! markers. They map to WGSL `atomicAdd`, `atomicSub`, etc. These have no CPU fallback
//! because generic `<T>` atomics cannot be safely emulated without specialization.
//!
//! # Subgroup/Warp Operations
//!
//! Subgroup functions (`warp_shuffle`, `warp_ballot`, `warp_all`, `warp_any`, `lane_id`,
//! `warp_size`) are transpiler-only markers. They map to WGSL subgroup builtins which
//! require the `enable chromium_experimental_subgroups;` directive (or the standardized
//! `enable subgroups;` once ratified). These operations are inherently multi-lane and
//! have no meaningful single-threaded CPU fallback.

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
//
// These are transpiler-only markers with no CPU fallback. They map to WGSL
// atomic builtins during transpilation. Calling them directly on CPU will panic.
// See the module-level documentation for details on the marker pattern.
// =============================================================================

/// Atomic add. Maps to `atomicAdd(&var, value)` in WGSL.
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn atomic_add<T>(_ptr: &T, _val: T) -> T {
    unimplemented!(
        "atomic_add is a WGSL intrinsic -- this function is a transpiler marker and must \
         not be called directly. Use the ringkernel-wgpu-codegen transpiler to convert to WGSL."
    )
}

/// Atomic subtract. Maps to `atomicSub(&var, value)` in WGSL.
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn atomic_sub<T>(_ptr: &T, _val: T) -> T {
    unimplemented!(
        "atomic_sub is a WGSL intrinsic -- this function is a transpiler marker and must \
         not be called directly. Use the ringkernel-wgpu-codegen transpiler to convert to WGSL."
    )
}

/// Atomic minimum. Maps to `atomicMin(&var, value)` in WGSL.
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn atomic_min<T>(_ptr: &T, _val: T) -> T {
    unimplemented!(
        "atomic_min is a WGSL intrinsic -- this function is a transpiler marker and must \
         not be called directly. Use the ringkernel-wgpu-codegen transpiler to convert to WGSL."
    )
}

/// Atomic maximum. Maps to `atomicMax(&var, value)` in WGSL.
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn atomic_max<T>(_ptr: &T, _val: T) -> T {
    unimplemented!(
        "atomic_max is a WGSL intrinsic -- this function is a transpiler marker and must \
         not be called directly. Use the ringkernel-wgpu-codegen transpiler to convert to WGSL."
    )
}

/// Atomic exchange. Maps to `atomicExchange(&var, value)` in WGSL.
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn atomic_exchange<T>(_ptr: &T, _val: T) -> T {
    unimplemented!(
        "atomic_exchange is a WGSL intrinsic -- this function is a transpiler marker and must \
         not be called directly. Use the ringkernel-wgpu-codegen transpiler to convert to WGSL."
    )
}

/// Atomic compare-and-swap. Maps to `atomicCompareExchangeWeak(&var, compare, value)` in WGSL.
/// Returns the old value.
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn atomic_cas<T>(_ptr: &T, _compare: T, _val: T) -> T {
    unimplemented!(
        "atomic_cas is a WGSL intrinsic -- this function is a transpiler marker and must \
         not be called directly. Use the ringkernel-wgpu-codegen transpiler to convert to WGSL."
    )
}

/// Atomic load. Maps to `atomicLoad(&var)` in WGSL.
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn atomic_load<T>(_ptr: &T) -> T {
    unimplemented!(
        "atomic_load is a WGSL intrinsic -- this function is a transpiler marker and must \
         not be called directly. Use the ringkernel-wgpu-codegen transpiler to convert to WGSL."
    )
}

/// Atomic store. Maps to `atomicStore(&var, value)` in WGSL.
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn atomic_store<T>(_ptr: &T, _val: T) {
    unimplemented!(
        "atomic_store is a WGSL intrinsic -- this function is a transpiler marker and must \
         not be called directly. Use the ringkernel-wgpu-codegen transpiler to convert to WGSL."
    )
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
//
// These are transpiler-only markers with no CPU fallback. They map to WGSL
// subgroup builtins during transpilation, which require either
// `enable chromium_experimental_subgroups;` or the standardized
// `enable subgroups;` directive (once ratified in the WebGPU spec).
//
// Calling them directly on CPU will panic because subgroup operations are
// inherently multi-lane and have no meaningful single-threaded semantics.
// =============================================================================

/// Shuffle within subgroup. Maps to `subgroupShuffle(value, lane)` in WGSL.
/// Requires: `enable chromium_experimental_subgroups;` or `enable subgroups;`
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn warp_shuffle<T>(_val: T, _lane: u32) -> T {
    unimplemented!(
        "warp_shuffle is a WGSL subgroup intrinsic (subgroupShuffle) -- this function is a \
         transpiler marker and must not be called directly. Use the ringkernel-wgpu-codegen \
         transpiler to convert to WGSL. Requires `enable subgroups;` in the generated shader."
    )
}

/// Shuffle up within subgroup. Maps to `subgroupShuffleUp(value, delta)` in WGSL.
/// Requires: `enable chromium_experimental_subgroups;` or `enable subgroups;`
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn warp_shuffle_up<T>(_val: T, _delta: u32) -> T {
    unimplemented!(
        "warp_shuffle_up is a WGSL subgroup intrinsic (subgroupShuffleUp) -- this function is a \
         transpiler marker and must not be called directly. Use the ringkernel-wgpu-codegen \
         transpiler to convert to WGSL. Requires `enable subgroups;` in the generated shader."
    )
}

/// Shuffle down within subgroup. Maps to `subgroupShuffleDown(value, delta)` in WGSL.
/// Requires: `enable chromium_experimental_subgroups;` or `enable subgroups;`
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn warp_shuffle_down<T>(_val: T, _delta: u32) -> T {
    unimplemented!(
        "warp_shuffle_down is a WGSL subgroup intrinsic (subgroupShuffleDown) -- this function \
         is a transpiler marker and must not be called directly. Use the ringkernel-wgpu-codegen \
         transpiler to convert to WGSL. Requires `enable subgroups;` in the generated shader."
    )
}

/// Shuffle XOR within subgroup. Maps to `subgroupShuffleXor(value, mask)` in WGSL.
/// Requires: `enable chromium_experimental_subgroups;` or `enable subgroups;`
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn warp_shuffle_xor<T>(_val: T, _mask: u32) -> T {
    unimplemented!(
        "warp_shuffle_xor is a WGSL subgroup intrinsic (subgroupShuffleXor) -- this function \
         is a transpiler marker and must not be called directly. Use the ringkernel-wgpu-codegen \
         transpiler to convert to WGSL. Requires `enable subgroups;` in the generated shader."
    )
}

/// Ballot within subgroup. Maps to `subgroupBallot(predicate)` in WGSL.
/// Returns a vec4<u32> where each bit represents a lane's predicate.
/// Requires: `enable chromium_experimental_subgroups;` or `enable subgroups;`
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn warp_ballot(_pred: bool) -> [u32; 4] {
    unimplemented!(
        "warp_ballot is a WGSL subgroup intrinsic (subgroupBallot) -- this function is a \
         transpiler marker and must not be called directly. Use the ringkernel-wgpu-codegen \
         transpiler to convert to WGSL. Requires `enable subgroups;` in the generated shader."
    )
}

/// Check if all lanes in subgroup satisfy predicate. Maps to `subgroupAll(predicate)` in WGSL.
/// Requires: `enable chromium_experimental_subgroups;` or `enable subgroups;`
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn warp_all(_pred: bool) -> bool {
    unimplemented!(
        "warp_all is a WGSL subgroup intrinsic (subgroupAll) -- this function is a \
         transpiler marker and must not be called directly. Use the ringkernel-wgpu-codegen \
         transpiler to convert to WGSL. Requires `enable subgroups;` in the generated shader."
    )
}

/// Check if any lane in subgroup satisfies predicate. Maps to `subgroupAny(predicate)` in WGSL.
/// Requires: `enable chromium_experimental_subgroups;` or `enable subgroups;`
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn warp_any(_pred: bool) -> bool {
    unimplemented!(
        "warp_any is a WGSL subgroup intrinsic (subgroupAny) -- this function is a \
         transpiler marker and must not be called directly. Use the ringkernel-wgpu-codegen \
         transpiler to convert to WGSL. Requires `enable subgroups;` in the generated shader."
    )
}

/// Get the lane ID within the subgroup.
/// Maps to `subgroup_invocation_id` builtin in WGSL.
/// Requires: `enable chromium_experimental_subgroups;` or `enable subgroups;`
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn lane_id() -> u32 {
    unimplemented!(
        "lane_id is a WGSL subgroup intrinsic (subgroup_invocation_id) -- this function is a \
         transpiler marker and must not be called directly. Use the ringkernel-wgpu-codegen \
         transpiler to convert to WGSL. Requires `enable subgroups;` in the generated shader."
    )
}

/// Get the subgroup size.
/// Maps to `subgroup_size` builtin in WGSL.
/// Requires: `enable chromium_experimental_subgroups;` or `enable subgroups;`
///
/// # Panics
/// Always panics when called directly. This is a transpiler marker -- use
/// the `ringkernel-wgpu-codegen` transpiler to convert to WGSL.
#[inline(always)]
pub fn warp_size() -> u32 {
    unimplemented!(
        "warp_size is a WGSL subgroup intrinsic (subgroup_size) -- this function is a \
         transpiler marker and must not be called directly. Use the ringkernel-wgpu-codegen \
         transpiler to convert to WGSL. Requires `enable subgroups;` in the generated shader."
    )
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
