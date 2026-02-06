//! Rust DSL functions for writing CUDA kernels.
//!
//! This module provides Rust functions that map to CUDA intrinsics during transpilation.
//! These functions have CPU fallback implementations for testing but are transpiled
//! to the corresponding CUDA operations when used in kernel code.
//!
//! Common functions (thread indices, basic math, synchronization) are shared with
//! the WGSL backend via `ringkernel_codegen::dsl_common`.
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
//!
//! # Math Functions
//!
//! All standard math functions are available with CPU fallbacks:
//! - Trigonometric: sin, cos, tan, asin, acos, atan, atan2
//! - Hyperbolic: sinh, cosh, tanh, asinh, acosh, atanh
//! - Exponential: exp, exp2, exp10, expm1, log, log2, log10, log1p
//! - Power: pow, sqrt, rsqrt, cbrt
//! - Rounding: floor, ceil, round, trunc
//! - Comparison: fmin, fmax, fdim, copysign
//!
//! # Warp Operations
//!
//! ```ignore
//! let mask = warp_active_mask();       // Get active lane mask
//! let result = warp_reduce_add(mask, value);  // Warp-level sum
//! let shuffled = warp_shfl(mask, value, lane); // Shuffle
//! ```
//!
//! # Bit Manipulation
//!
//! ```ignore
//! let bits = popc(x);      // Count set bits
//! let zeros = clz(x);      // Count leading zeros
//! let rev = brev(x);       // Reverse bits
//! ```

use std::sync::atomic::{fence, Ordering};

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

// ============================================================================
// CUDA-Specific Thread/Block Functions
// ============================================================================

/// Get the warp size (always 32 on NVIDIA GPUs).
/// Transpiles to: `warpSize`
#[inline]
pub fn warp_size() -> i32 {
    32
}

// ============================================================================
// CUDA-Specific Synchronization Functions
// ============================================================================

/// Synchronize threads and count predicate.
/// Transpiles to: `__syncthreads_count(predicate)`
#[inline]
pub fn sync_threads_count(predicate: bool) -> i32 {
    if predicate {
        1
    } else {
        0
    }
}

/// Synchronize threads with AND of predicate.
/// Transpiles to: `__syncthreads_and(predicate)`
#[inline]
pub fn sync_threads_and(predicate: bool) -> i32 {
    if predicate {
        1
    } else {
        0
    }
}

/// Synchronize threads with OR of predicate.
/// Transpiles to: `__syncthreads_or(predicate)`
#[inline]
pub fn sync_threads_or(predicate: bool) -> i32 {
    if predicate {
        1
    } else {
        0
    }
}

/// System-wide memory fence.
/// Transpiles to: `__threadfence_system()`
#[inline]
pub fn thread_fence_system() {
    fence(Ordering::SeqCst);
}

// ============================================================================
// Atomic Operations (CPU fallbacks - not thread-safe!)
// ============================================================================

/// Atomic add. Transpiles to: `atomicAdd(addr, val)`
/// WARNING: CPU fallback is NOT thread-safe!
#[inline]
pub fn atomic_add(addr: &mut i32, val: i32) -> i32 {
    let old = *addr;
    *addr += val;
    old
}

/// Atomic add for f32. Transpiles to: `atomicAdd(addr, val)`
#[inline]
pub fn atomic_add_f32(addr: &mut f32, val: f32) -> f32 {
    let old = *addr;
    *addr += val;
    old
}

/// Atomic subtract. Transpiles to: `atomicSub(addr, val)`
#[inline]
pub fn atomic_sub(addr: &mut i32, val: i32) -> i32 {
    let old = *addr;
    *addr -= val;
    old
}

/// Atomic minimum. Transpiles to: `atomicMin(addr, val)`
#[inline]
pub fn atomic_min(addr: &mut i32, val: i32) -> i32 {
    let old = *addr;
    *addr = old.min(val);
    old
}

/// Atomic maximum. Transpiles to: `atomicMax(addr, val)`
#[inline]
pub fn atomic_max(addr: &mut i32, val: i32) -> i32 {
    let old = *addr;
    *addr = old.max(val);
    old
}

/// Atomic exchange. Transpiles to: `atomicExch(addr, val)`
#[inline]
pub fn atomic_exchange(addr: &mut i32, val: i32) -> i32 {
    let old = *addr;
    *addr = val;
    old
}

/// Atomic compare and swap. Transpiles to: `atomicCAS(addr, compare, val)`
#[inline]
pub fn atomic_cas(addr: &mut i32, compare: i32, val: i32) -> i32 {
    let old = *addr;
    if old == compare {
        *addr = val;
    }
    old
}

/// Atomic AND. Transpiles to: `atomicAnd(addr, val)`
#[inline]
pub fn atomic_and(addr: &mut i32, val: i32) -> i32 {
    let old = *addr;
    *addr &= val;
    old
}

/// Atomic OR. Transpiles to: `atomicOr(addr, val)`
#[inline]
pub fn atomic_or(addr: &mut i32, val: i32) -> i32 {
    let old = *addr;
    *addr |= val;
    old
}

/// Atomic XOR. Transpiles to: `atomicXor(addr, val)`
#[inline]
pub fn atomic_xor(addr: &mut i32, val: i32) -> i32 {
    let old = *addr;
    *addr ^= val;
    old
}

/// Atomic increment with wrap. Transpiles to: `atomicInc(addr, val)`
#[inline]
pub fn atomic_inc(addr: &mut u32, val: u32) -> u32 {
    let old = *addr;
    *addr = if old >= val { 0 } else { old + 1 };
    old
}

/// Atomic decrement with wrap. Transpiles to: `atomicDec(addr, val)`
#[inline]
pub fn atomic_dec(addr: &mut u32, val: u32) -> u32 {
    let old = *addr;
    *addr = if old == 0 || old > val { val } else { old - 1 };
    old
}

// ============================================================================
// CUDA-Specific Math Functions
// ============================================================================

/// Absolute value for f32. Transpiles to: `fabsf(x)`
#[inline]
pub fn fabs(x: f32) -> f32 {
    x.abs()
}

/// Truncate toward zero. Transpiles to: `truncf(x)`
#[inline]
pub fn trunc(x: f32) -> f32 {
    x.trunc()
}

/// Minimum. Transpiles to: `fminf(a, b)`
#[inline]
pub fn fmin(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// Maximum. Transpiles to: `fmaxf(a, b)`
#[inline]
pub fn fmax(a: f32, b: f32) -> f32 {
    a.max(b)
}

/// Floating-point modulo. Transpiles to: `fmodf(x, y)`
#[inline]
pub fn fmod(x: f32, y: f32) -> f32 {
    x % y
}

/// Remainder. Transpiles to: `remainderf(x, y)`
#[inline]
pub fn remainder(x: f32, y: f32) -> f32 {
    x - (x / y).round() * y
}

/// Copy sign. Transpiles to: `copysignf(x, y)`
#[inline]
pub fn copysign(x: f32, y: f32) -> f32 {
    x.copysign(y)
}

/// Cube root. Transpiles to: `cbrtf(x)`
#[inline]
pub fn cbrt(x: f32) -> f32 {
    x.cbrt()
}

/// Hypotenuse. Transpiles to: `hypotf(x, y)`
#[inline]
pub fn hypot(x: f32, y: f32) -> f32 {
    x.hypot(y)
}

// ============================================================================
// CUDA-Specific Trigonometric Functions
// ============================================================================

/// Arcsine. Transpiles to: `asinf(x)`
#[inline]
pub fn asin(x: f32) -> f32 {
    x.asin()
}

/// Arccosine. Transpiles to: `acosf(x)`
#[inline]
pub fn acos(x: f32) -> f32 {
    x.acos()
}

/// Arctangent. Transpiles to: `atanf(x)`
#[inline]
pub fn atan(x: f32) -> f32 {
    x.atan()
}

/// Two-argument arctangent. Transpiles to: `atan2f(y, x)`
#[inline]
pub fn atan2(y: f32, x: f32) -> f32 {
    y.atan2(x)
}

/// Sine and cosine together. Transpiles to: `sincosf(x, &s, &c)`
#[inline]
pub fn sincos(x: f32) -> (f32, f32) {
    (x.sin(), x.cos())
}

/// Sine of pi*x. Transpiles to: `sinpif(x)`
#[inline]
pub fn sinpi(x: f32) -> f32 {
    (x * std::f32::consts::PI).sin()
}

/// Cosine of pi*x. Transpiles to: `cospif(x)`
#[inline]
pub fn cospi(x: f32) -> f32 {
    (x * std::f32::consts::PI).cos()
}

// ============================================================================
// CUDA-Specific Hyperbolic Functions
// ============================================================================

/// Hyperbolic sine. Transpiles to: `sinhf(x)`
#[inline]
pub fn sinh(x: f32) -> f32 {
    x.sinh()
}

/// Hyperbolic cosine. Transpiles to: `coshf(x)`
#[inline]
pub fn cosh(x: f32) -> f32 {
    x.cosh()
}

/// Hyperbolic tangent. Transpiles to: `tanhf(x)`
#[inline]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// Inverse hyperbolic sine. Transpiles to: `asinhf(x)`
#[inline]
pub fn asinh(x: f32) -> f32 {
    x.asinh()
}

/// Inverse hyperbolic cosine. Transpiles to: `acoshf(x)`
#[inline]
pub fn acosh(x: f32) -> f32 {
    x.acosh()
}

/// Inverse hyperbolic tangent. Transpiles to: `atanhf(x)`
#[inline]
pub fn atanh(x: f32) -> f32 {
    x.atanh()
}

// ============================================================================
// CUDA-Specific Exponential and Logarithmic Functions
// ============================================================================

/// Exponential (base 2). Transpiles to: `exp2f(x)`
#[inline]
pub fn exp2(x: f32) -> f32 {
    x.exp2()
}

/// Exponential (base 10). Transpiles to: `exp10f(x)`
#[inline]
pub fn exp10(x: f32) -> f32 {
    (x * std::f32::consts::LN_10).exp()
}

/// exp(x) - 1 (accurate for small x). Transpiles to: `expm1f(x)`
#[inline]
pub fn expm1(x: f32) -> f32 {
    x.exp_m1()
}

/// Logarithm (base 2). Transpiles to: `log2f(x)`
#[inline]
pub fn log2(x: f32) -> f32 {
    x.log2()
}

/// Logarithm (base 10). Transpiles to: `log10f(x)`
#[inline]
pub fn log10(x: f32) -> f32 {
    x.log10()
}

/// log(1 + x) (accurate for small x). Transpiles to: `log1pf(x)`
#[inline]
pub fn log1p(x: f32) -> f32 {
    x.ln_1p()
}

/// Power. Transpiles to: `powf(x, y)`
#[inline]
pub fn pow(x: f32, y: f32) -> f32 {
    x.powf(y)
}

/// Load exponent. Transpiles to: `ldexpf(x, exp)`
#[inline]
pub fn ldexp(x: f32, exp: i32) -> f32 {
    x * 2.0_f32.powi(exp)
}

/// Scale by power of 2. Transpiles to: `scalbnf(x, n)`
#[inline]
pub fn scalbn(x: f32, n: i32) -> f32 {
    x * 2.0_f32.powi(n)
}

/// Extract exponent. Transpiles to: `ilogbf(x)`
#[inline]
pub fn ilogb(x: f32) -> i32 {
    if x == 0.0 {
        i32::MIN
    } else if x.is_infinite() {
        i32::MAX
    } else {
        x.abs().log2().floor() as i32
    }
}

/// Error function. Transpiles to: `erff(x)`
#[inline]
pub fn erf(x: f32) -> f32 {
    // Approximation using Horner form
    let a1 = 0.254_829_6_f32;
    let a2 = -0.284_496_74_f32;
    let a3 = 1.421_413_7_f32;
    let a4 = -1.453_152_f32;
    let a5 = 1.061_405_4_f32;
    let p = 0.327_591_1_f32;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Complementary error function. Transpiles to: `erfcf(x)`
#[inline]
pub fn erfc(x: f32) -> f32 {
    1.0 - erf(x)
}

// ============================================================================
// Classification and Comparison Functions
// ============================================================================

/// Check if NaN. Transpiles to: `isnan(x)`
#[inline]
pub fn is_nan(x: f32) -> bool {
    x.is_nan()
}

/// Check if infinite. Transpiles to: `isinf(x)`
#[inline]
pub fn is_infinite(x: f32) -> bool {
    x.is_infinite()
}

/// Check if finite. Transpiles to: `isfinite(x)`
#[inline]
pub fn is_finite(x: f32) -> bool {
    x.is_finite()
}

/// Check if normal. Transpiles to: `isnormal(x)`
#[inline]
pub fn is_normal(x: f32) -> bool {
    x.is_normal()
}

/// Check sign bit. Transpiles to: `signbit(x)`
#[inline]
pub fn signbit(x: f32) -> bool {
    x.is_sign_negative()
}

/// Next representable value. Transpiles to: `nextafterf(x, y)`
#[inline]
pub fn nextafter(x: f32, y: f32) -> f32 {
    if x == y {
        y
    } else if y > x {
        f32::from_bits(x.to_bits() + 1)
    } else {
        f32::from_bits(x.to_bits() - 1)
    }
}

/// Floating-point difference. Transpiles to: `fdimf(x, y)`
#[inline]
pub fn fdim(x: f32, y: f32) -> f32 {
    if x > y {
        x - y
    } else {
        0.0
    }
}

// ============================================================================
// Warp-Level Operations
// ============================================================================

/// Get active thread mask. Transpiles to: `__activemask()`
#[inline]
pub fn warp_active_mask() -> u32 {
    1 // CPU fallback: only one thread active
}

/// Warp ballot. Transpiles to: `__ballot_sync(mask, predicate)`
#[inline]
pub fn warp_ballot(_mask: u32, predicate: bool) -> u32 {
    if predicate {
        1
    } else {
        0
    }
}

/// Warp all predicate. Transpiles to: `__all_sync(mask, predicate)`
#[inline]
pub fn warp_all(_mask: u32, predicate: bool) -> bool {
    predicate
}

/// Warp any predicate. Transpiles to: `__any_sync(mask, predicate)`
#[inline]
pub fn warp_any(_mask: u32, predicate: bool) -> bool {
    predicate
}

/// Warp shuffle. Transpiles to: `__shfl_sync(mask, val, lane)`
#[inline]
pub fn warp_shfl<T: Copy>(_mask: u32, val: T, _lane: i32) -> T {
    val // CPU fallback: return same value
}

/// Warp shuffle up. Transpiles to: `__shfl_up_sync(mask, val, delta)`
#[inline]
pub fn warp_shfl_up<T: Copy>(_mask: u32, val: T, _delta: u32) -> T {
    val
}

/// Warp shuffle down. Transpiles to: `__shfl_down_sync(mask, val, delta)`
#[inline]
pub fn warp_shfl_down<T: Copy>(_mask: u32, val: T, _delta: u32) -> T {
    val
}

/// Warp shuffle XOR. Transpiles to: `__shfl_xor_sync(mask, val, lane_mask)`
#[inline]
pub fn warp_shfl_xor<T: Copy>(_mask: u32, val: T, _lane_mask: i32) -> T {
    val
}

/// Warp reduce add. Transpiles to: `__reduce_add_sync(mask, val)`
#[inline]
pub fn warp_reduce_add(_mask: u32, val: i32) -> i32 {
    val // CPU: single thread, no reduction needed
}

/// Warp reduce min. Transpiles to: `__reduce_min_sync(mask, val)`
#[inline]
pub fn warp_reduce_min(_mask: u32, val: i32) -> i32 {
    val
}

/// Warp reduce max. Transpiles to: `__reduce_max_sync(mask, val)`
#[inline]
pub fn warp_reduce_max(_mask: u32, val: i32) -> i32 {
    val
}

/// Warp reduce AND. Transpiles to: `__reduce_and_sync(mask, val)`
#[inline]
pub fn warp_reduce_and(_mask: u32, val: u32) -> u32 {
    val
}

/// Warp reduce OR. Transpiles to: `__reduce_or_sync(mask, val)`
#[inline]
pub fn warp_reduce_or(_mask: u32, val: u32) -> u32 {
    val
}

/// Warp reduce XOR. Transpiles to: `__reduce_xor_sync(mask, val)`
#[inline]
pub fn warp_reduce_xor(_mask: u32, val: u32) -> u32 {
    val
}

/// Warp match any. Transpiles to: `__match_any_sync(mask, val)`
#[inline]
pub fn warp_match_any(_mask: u32, _val: u32) -> u32 {
    1 // CPU: single thread always matches itself
}

/// Warp match all. Transpiles to: `__match_all_sync(mask, val, pred)`
#[inline]
pub fn warp_match_all(_mask: u32, _val: u32) -> (u32, bool) {
    (1, true) // CPU: single thread, trivially all match
}

// ============================================================================
// Bit Manipulation Functions
// ============================================================================

/// Population count (count set bits). Transpiles to: `__popc(x)`
#[inline]
pub fn popc(x: u32) -> i32 {
    x.count_ones() as i32
}

/// Population count (i32 version).
#[inline]
pub fn popcount(x: i32) -> i32 {
    (x as u32).count_ones() as i32
}

/// Count leading zeros. Transpiles to: `__clz(x)`
#[inline]
pub fn clz(x: u32) -> i32 {
    x.leading_zeros() as i32
}

/// Count leading zeros (i32 version).
#[inline]
pub fn leading_zeros(x: i32) -> i32 {
    (x as u32).leading_zeros() as i32
}

/// Count trailing zeros. Transpiles to: `__ffs(x) - 1`
#[inline]
pub fn ctz(x: u32) -> i32 {
    if x == 0 {
        32
    } else {
        x.trailing_zeros() as i32
    }
}

/// Count trailing zeros (i32 version).
#[inline]
pub fn trailing_zeros(x: i32) -> i32 {
    if x == 0 {
        32
    } else {
        (x as u32).trailing_zeros() as i32
    }
}

/// Find first set bit (1-indexed, 0 if none). Transpiles to: `__ffs(x)`
#[inline]
pub fn ffs(x: u32) -> i32 {
    if x == 0 {
        0
    } else {
        (x.trailing_zeros() + 1) as i32
    }
}

/// Bit reverse. Transpiles to: `__brev(x)`
#[inline]
pub fn brev(x: u32) -> u32 {
    x.reverse_bits()
}

/// Bit reverse (i32 version).
#[inline]
pub fn reverse_bits(x: i32) -> i32 {
    (x as u32).reverse_bits() as i32
}

/// Byte permutation. Transpiles to: `__byte_perm(x, y, s)`
#[inline]
pub fn byte_perm(x: u32, y: u32, s: u32) -> u32 {
    let bytes = [
        (x & 0xFF) as u8,
        ((x >> 8) & 0xFF) as u8,
        ((x >> 16) & 0xFF) as u8,
        ((x >> 24) & 0xFF) as u8,
        (y & 0xFF) as u8,
        ((y >> 8) & 0xFF) as u8,
        ((y >> 16) & 0xFF) as u8,
        ((y >> 24) & 0xFF) as u8,
    ];
    let b0 = bytes[(s & 0x7) as usize] as u32;
    let b1 = bytes[((s >> 4) & 0x7) as usize] as u32;
    let b2 = bytes[((s >> 8) & 0x7) as usize] as u32;
    let b3 = bytes[((s >> 12) & 0x7) as usize] as u32;
    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

/// Funnel shift left. Transpiles to: `__funnelshift_l(lo, hi, shift)`
#[inline]
pub fn funnel_shift_left(lo: u32, hi: u32, shift: u32) -> u32 {
    let shift = shift & 31;
    if shift == 0 {
        lo
    } else {
        (hi << shift) | (lo >> (32 - shift))
    }
}

/// Funnel shift right. Transpiles to: `__funnelshift_r(lo, hi, shift)`
#[inline]
pub fn funnel_shift_right(lo: u32, hi: u32, shift: u32) -> u32 {
    let shift = shift & 31;
    if shift == 0 {
        lo
    } else {
        (lo >> shift) | (hi << (32 - shift))
    }
}

// ============================================================================
// Memory Operations
// ============================================================================

/// Read-only cache load. Transpiles to: `__ldg(ptr)`
#[inline]
pub fn ldg<T: Copy>(ptr: &T) -> T {
    *ptr
}

/// Load from global memory (alias for ldg).
#[inline]
pub fn load_global<T: Copy>(ptr: &T) -> T {
    *ptr
}

/// Prefetch to L1 cache. Transpiles to: `__prefetch_l1(ptr)`
#[inline]
pub fn prefetch_l1<T>(_ptr: &T) {
    // CPU fallback: no-op
}

/// Prefetch to L2 cache. Transpiles to: `__prefetch_l2(ptr)`
#[inline]
pub fn prefetch_l2<T>(_ptr: &T) {
    // CPU fallback: no-op
}

// ============================================================================
// Special Functions
// ============================================================================

/// Fast reciprocal. Transpiles to: `__frcp_rn(x)`
#[inline]
pub fn rcp(x: f32) -> f32 {
    1.0 / x
}

/// Fast division. Transpiles to: `__fdividef(x, y)`
#[inline]
pub fn fast_div(x: f32, y: f32) -> f32 {
    x / y
}

/// Saturate to [0, 1]. Transpiles to: `__saturatef(x)`
#[inline]
pub fn saturate(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Clamp to [0, 1] (alias for saturate).
#[inline]
pub fn clamp_01(x: f32) -> f32 {
    saturate(x)
}

// ============================================================================
// Clock and Timing
// ============================================================================

/// Read clock counter. Transpiles to: `clock()`
#[inline]
pub fn clock() -> u32 {
    // CPU fallback: use std time
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(0)
}

/// Read 64-bit clock counter. Transpiles to: `clock64()`
#[inline]
pub fn clock64() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Nanosleep. Transpiles to: `__nanosleep(ns)`
#[inline]
pub fn nanosleep(ns: u32) {
    std::thread::sleep(std::time::Duration::from_nanos(ns as u64));
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
        assert_eq!(warp_size(), 32);
    }

    #[test]
    fn test_math_functions() {
        assert!((sqrt(4.0) - 2.0).abs() < 1e-6);
        assert!((rsqrt(4.0) - 0.5).abs() < 1e-6);
        assert!((sin(0.0)).abs() < 1e-6);
        assert!((cos(0.0) - 1.0).abs() < 1e-6);
        assert!((exp(0.0) - 1.0).abs() < 1e-6);
        assert!((log(1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_trigonometric_functions() {
        let pi = std::f32::consts::PI;
        assert!((sin(pi / 2.0) - 1.0).abs() < 1e-6);
        assert!((cos(pi) + 1.0).abs() < 1e-6);
        assert!((tan(0.0)).abs() < 1e-6);
        assert!((asin(1.0) - pi / 2.0).abs() < 1e-6);
        assert!((atan2(1.0, 1.0) - pi / 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_hyperbolic_functions() {
        assert!((sinh(0.0)).abs() < 1e-6);
        assert!((cosh(0.0) - 1.0).abs() < 1e-6);
        assert!((tanh(0.0)).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_functions() {
        assert!((exp2(3.0) - 8.0).abs() < 1e-6);
        assert!((log2(8.0) - 3.0).abs() < 1e-6);
        assert!((log10(100.0) - 2.0).abs() < 1e-6);
        assert!((pow(2.0, 3.0) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_classification_functions() {
        assert!(is_nan(f32::NAN));
        assert!(!is_nan(1.0));
        assert!(is_infinite(f32::INFINITY));
        assert!(!is_infinite(1.0));
        assert!(is_finite(1.0));
        assert!(!is_finite(f32::INFINITY));
    }

    #[test]
    fn test_bit_manipulation() {
        assert_eq!(popc(0b1010_1010), 4);
        assert_eq!(clz(1u32), 31);
        assert_eq!(clz(0x8000_0000u32), 0);
        assert_eq!(ctz(0b1000), 3);
        assert_eq!(ffs(0b1000), 4);
        assert_eq!(brev(1u32), 0x8000_0000);
    }

    #[test]
    fn test_warp_operations() {
        assert_eq!(warp_active_mask(), 1);
        assert_eq!(warp_ballot(0xFFFF_FFFF, true), 1);
        assert!(warp_all(0xFFFF_FFFF, true));
        assert!(warp_any(0xFFFF_FFFF, true));
        assert_eq!(warp_reduce_add(0xFFFF_FFFF, 5), 5);
    }

    #[test]
    fn test_special_functions() {
        assert!((rcp(2.0) - 0.5).abs() < 1e-6);
        assert!((fast_div(10.0, 2.0) - 5.0).abs() < 1e-6);
        assert_eq!(saturate(-1.0), 0.0);
        assert_eq!(saturate(0.5), 0.5);
        assert_eq!(saturate(2.0), 1.0);
    }

    #[test]
    fn test_atomic_operations() {
        let mut val = 10;
        assert_eq!(atomic_add(&mut val, 5), 10);
        assert_eq!(val, 15);

        let mut val = 10;
        assert_eq!(atomic_sub(&mut val, 3), 10);
        assert_eq!(val, 7);

        let mut val = 10;
        assert_eq!(atomic_cas(&mut val, 10, 20), 10);
        assert_eq!(val, 20);
    }

    #[test]
    fn test_funnel_shift() {
        assert_eq!(funnel_shift_left(0xFFFF_0000, 0x0000_FFFF, 16), 0xFFFF_FFFF);
        assert_eq!(
            funnel_shift_right(0xFFFF_0000, 0x0000_FFFF, 16),
            0xFFFF_FFFF
        );
    }

    #[test]
    fn test_byte_perm() {
        let x = 0x04030201u32;
        let y = 0x08070605u32;
        // Select bytes 0, 1, 2, 3 from x
        assert_eq!(byte_perm(x, y, 0x3210), 0x04030201);
        // Select bytes 4, 5, 6, 7 from y
        assert_eq!(byte_perm(x, y, 0x7654), 0x08070605);
    }
}
