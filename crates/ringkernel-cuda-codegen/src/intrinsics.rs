//! GPU intrinsic mapping for CUDA code generation.
//!
//! This module provides mappings from high-level Rust operations to
//! CUDA intrinsics and built-in functions.

use std::collections::HashMap;

/// GPU intrinsic operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuIntrinsic {
    // === Synchronization ===
    /// Thread synchronization within a block.
    SyncThreads,
    /// Thread fence (memory ordering across device).
    ThreadFence,
    /// Thread fence within block.
    ThreadFenceBlock,
    /// Thread fence across system.
    ThreadFenceSystem,
    /// Synchronize threads with predicate.
    SyncThreadsCount,
    /// Synchronize threads with AND predicate.
    SyncThreadsAnd,
    /// Synchronize threads with OR predicate.
    SyncThreadsOr,

    // === Atomic Operations (Integer) ===
    /// Atomic add.
    AtomicAdd,
    /// Atomic subtract.
    AtomicSub,
    /// Atomic minimum.
    AtomicMin,
    /// Atomic maximum.
    AtomicMax,
    /// Atomic exchange.
    AtomicExch,
    /// Atomic compare-and-swap.
    AtomicCas,
    /// Atomic bitwise AND.
    AtomicAnd,
    /// Atomic bitwise OR.
    AtomicOr,
    /// Atomic bitwise XOR.
    AtomicXor,
    /// Atomic increment (with wrap).
    AtomicInc,
    /// Atomic decrement (with wrap).
    AtomicDec,

    // === Basic Math Functions ===
    /// Square root.
    Sqrt,
    /// Reciprocal square root.
    Rsqrt,
    /// Absolute value (integer).
    Abs,
    /// Absolute value (floating point).
    Fabs,
    /// Floor.
    Floor,
    /// Ceiling.
    Ceil,
    /// Round to nearest.
    Round,
    /// Truncate toward zero.
    Trunc,
    /// Fused multiply-add.
    Fma,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
    /// Floating-point modulo.
    Fmod,
    /// Remainder.
    Remainder,
    /// Copy sign.
    Copysign,
    /// Cube root.
    Cbrt,
    /// Hypotenuse.
    Hypot,

    // === Trigonometric Functions ===
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Tangent.
    Tan,
    /// Arcsine.
    Asin,
    /// Arccosine.
    Acos,
    /// Arctangent.
    Atan,
    /// Two-argument arctangent.
    Atan2,
    /// Sine and cosine (combined).
    Sincos,
    /// Sine of pi*x.
    Sinpi,
    /// Cosine of pi*x.
    Cospi,

    // === Hyperbolic Functions ===
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Hyperbolic tangent.
    Tanh,
    /// Inverse hyperbolic sine.
    Asinh,
    /// Inverse hyperbolic cosine.
    Acosh,
    /// Inverse hyperbolic tangent.
    Atanh,

    // === Exponential and Logarithmic Functions ===
    /// Exponential (base e).
    Exp,
    /// Exponential (base 2).
    Exp2,
    /// Exponential (base 10).
    Exp10,
    /// exp(x) - 1 (accurate for small x).
    Expm1,
    /// Natural logarithm (base e).
    Log,
    /// Logarithm (base 2).
    Log2,
    /// Logarithm (base 10).
    Log10,
    /// log(1 + x) (accurate for small x).
    Log1p,
    /// Power.
    Pow,
    /// Load exponent.
    Ldexp,
    /// Scale by power of 2.
    Scalbn,
    /// Extract exponent.
    Ilogb,
    /// Logarithm of gamma function.
    Lgamma,
    /// Gamma function.
    Tgamma,
    /// Error function.
    Erf,
    /// Complementary error function.
    Erfc,
    /// Inverse error function.
    Erfinv,
    /// Inverse complementary error function.
    Erfcinv,

    // === Classification and Comparison ===
    /// Check if NaN.
    Isnan,
    /// Check if infinite.
    Isinf,
    /// Check if finite.
    Isfinite,
    /// Check if normal.
    Isnormal,
    /// Check sign bit.
    Signbit,
    /// Next representable value.
    Nextafter,
    /// Floating-point difference.
    Fdim,
    /// Not-a-Number.
    Nan,

    // === Warp-Level Operations ===
    /// Warp shuffle.
    WarpShfl,
    /// Warp shuffle up.
    WarpShflUp,
    /// Warp shuffle down.
    WarpShflDown,
    /// Warp shuffle XOR.
    WarpShflXor,
    /// Get active thread mask.
    WarpActiveMask,
    /// Warp ballot.
    WarpBallot,
    /// Warp all predicate.
    WarpAll,
    /// Warp any predicate.
    WarpAny,
    /// Warp match any.
    WarpMatchAny,
    /// Warp match all.
    WarpMatchAll,
    /// Warp reduce add.
    WarpReduceAdd,
    /// Warp reduce min.
    WarpReduceMin,
    /// Warp reduce max.
    WarpReduceMax,
    /// Warp reduce AND.
    WarpReduceAnd,
    /// Warp reduce OR.
    WarpReduceOr,
    /// Warp reduce XOR.
    WarpReduceXor,

    // === Bit Manipulation ===
    /// Population count (count set bits).
    Popc,
    /// Count leading zeros.
    Clz,
    /// Count trailing zeros (via ffs).
    Ctz,
    /// Find first set bit.
    Ffs,
    /// Bit reverse.
    Brev,
    /// Byte permute.
    BytePerm,
    /// Funnel shift left.
    FunnelShiftLeft,
    /// Funnel shift right.
    FunnelShiftRight,

    // === Memory Operations ===
    /// Read-only cache load.
    Ldg,
    /// Prefetch L1.
    PrefetchL1,
    /// Prefetch L2.
    PrefetchL2,

    // === Special Functions ===
    /// Reciprocal.
    Rcp,
    /// Division (fast).
    Fdividef,
    /// Saturate to [0,1].
    Saturate,
    /// Bessel J0.
    J0,
    /// Bessel J1.
    J1,
    /// Bessel Jn.
    Jn,
    /// Bessel Y0.
    Y0,
    /// Bessel Y1.
    Y1,
    /// Bessel Yn.
    Yn,
    /// Normal CDF.
    Normcdf,
    /// Inverse normal CDF.
    Normcdfinv,
    /// Cylindrical Bessel I0.
    CylBesselI0,
    /// Cylindrical Bessel I1.
    CylBesselI1,

    // === CUDA Thread/Block Indices ===
    /// Thread index X.
    ThreadIdxX,
    /// Thread index Y.
    ThreadIdxY,
    /// Thread index Z.
    ThreadIdxZ,
    /// Block index X.
    BlockIdxX,
    /// Block index Y.
    BlockIdxY,
    /// Block index Z.
    BlockIdxZ,
    /// Block dimension X.
    BlockDimX,
    /// Block dimension Y.
    BlockDimY,
    /// Block dimension Z.
    BlockDimZ,
    /// Grid dimension X.
    GridDimX,
    /// Grid dimension Y.
    GridDimY,
    /// Grid dimension Z.
    GridDimZ,
    /// Warp size (always 32).
    WarpSize,

    // === Clock and Timing ===
    /// Read clock counter.
    Clock,
    /// Read 64-bit clock counter.
    Clock64,
    /// Nanosleep.
    Nanosleep,
}

impl GpuIntrinsic {
    /// Convert to CUDA function/intrinsic name.
    pub fn to_cuda_string(&self) -> &'static str {
        match self {
            // Synchronization
            GpuIntrinsic::SyncThreads => "__syncthreads()",
            GpuIntrinsic::ThreadFence => "__threadfence()",
            GpuIntrinsic::ThreadFenceBlock => "__threadfence_block()",
            GpuIntrinsic::ThreadFenceSystem => "__threadfence_system()",
            GpuIntrinsic::SyncThreadsCount => "__syncthreads_count",
            GpuIntrinsic::SyncThreadsAnd => "__syncthreads_and",
            GpuIntrinsic::SyncThreadsOr => "__syncthreads_or",

            // Atomic operations
            GpuIntrinsic::AtomicAdd => "atomicAdd",
            GpuIntrinsic::AtomicSub => "atomicSub",
            GpuIntrinsic::AtomicMin => "atomicMin",
            GpuIntrinsic::AtomicMax => "atomicMax",
            GpuIntrinsic::AtomicExch => "atomicExch",
            GpuIntrinsic::AtomicCas => "atomicCAS",
            GpuIntrinsic::AtomicAnd => "atomicAnd",
            GpuIntrinsic::AtomicOr => "atomicOr",
            GpuIntrinsic::AtomicXor => "atomicXor",
            GpuIntrinsic::AtomicInc => "atomicInc",
            GpuIntrinsic::AtomicDec => "atomicDec",

            // Basic math
            GpuIntrinsic::Sqrt => "sqrtf",
            GpuIntrinsic::Rsqrt => "rsqrtf",
            GpuIntrinsic::Abs => "abs",
            GpuIntrinsic::Fabs => "fabsf",
            GpuIntrinsic::Floor => "floorf",
            GpuIntrinsic::Ceil => "ceilf",
            GpuIntrinsic::Round => "roundf",
            GpuIntrinsic::Trunc => "truncf",
            GpuIntrinsic::Fma => "fmaf",
            GpuIntrinsic::Min => "fminf",
            GpuIntrinsic::Max => "fmaxf",
            GpuIntrinsic::Fmod => "fmodf",
            GpuIntrinsic::Remainder => "remainderf",
            GpuIntrinsic::Copysign => "copysignf",
            GpuIntrinsic::Cbrt => "cbrtf",
            GpuIntrinsic::Hypot => "hypotf",

            // Trigonometric
            GpuIntrinsic::Sin => "sinf",
            GpuIntrinsic::Cos => "cosf",
            GpuIntrinsic::Tan => "tanf",
            GpuIntrinsic::Asin => "asinf",
            GpuIntrinsic::Acos => "acosf",
            GpuIntrinsic::Atan => "atanf",
            GpuIntrinsic::Atan2 => "atan2f",
            GpuIntrinsic::Sincos => "sincosf",
            GpuIntrinsic::Sinpi => "sinpif",
            GpuIntrinsic::Cospi => "cospif",

            // Hyperbolic
            GpuIntrinsic::Sinh => "sinhf",
            GpuIntrinsic::Cosh => "coshf",
            GpuIntrinsic::Tanh => "tanhf",
            GpuIntrinsic::Asinh => "asinhf",
            GpuIntrinsic::Acosh => "acoshf",
            GpuIntrinsic::Atanh => "atanhf",

            // Exponential and logarithmic
            GpuIntrinsic::Exp => "expf",
            GpuIntrinsic::Exp2 => "exp2f",
            GpuIntrinsic::Exp10 => "exp10f",
            GpuIntrinsic::Expm1 => "expm1f",
            GpuIntrinsic::Log => "logf",
            GpuIntrinsic::Log2 => "log2f",
            GpuIntrinsic::Log10 => "log10f",
            GpuIntrinsic::Log1p => "log1pf",
            GpuIntrinsic::Pow => "powf",
            GpuIntrinsic::Ldexp => "ldexpf",
            GpuIntrinsic::Scalbn => "scalbnf",
            GpuIntrinsic::Ilogb => "ilogbf",
            GpuIntrinsic::Lgamma => "lgammaf",
            GpuIntrinsic::Tgamma => "tgammaf",
            GpuIntrinsic::Erf => "erff",
            GpuIntrinsic::Erfc => "erfcf",
            GpuIntrinsic::Erfinv => "erfinvf",
            GpuIntrinsic::Erfcinv => "erfcinvf",

            // Classification and comparison
            GpuIntrinsic::Isnan => "isnan",
            GpuIntrinsic::Isinf => "isinf",
            GpuIntrinsic::Isfinite => "isfinite",
            GpuIntrinsic::Isnormal => "isnormal",
            GpuIntrinsic::Signbit => "signbit",
            GpuIntrinsic::Nextafter => "nextafterf",
            GpuIntrinsic::Fdim => "fdimf",
            GpuIntrinsic::Nan => "nanf",

            // Warp-level operations
            GpuIntrinsic::WarpShfl => "__shfl_sync",
            GpuIntrinsic::WarpShflUp => "__shfl_up_sync",
            GpuIntrinsic::WarpShflDown => "__shfl_down_sync",
            GpuIntrinsic::WarpShflXor => "__shfl_xor_sync",
            GpuIntrinsic::WarpActiveMask => "__activemask()",
            GpuIntrinsic::WarpBallot => "__ballot_sync",
            GpuIntrinsic::WarpAll => "__all_sync",
            GpuIntrinsic::WarpAny => "__any_sync",
            GpuIntrinsic::WarpMatchAny => "__match_any_sync",
            GpuIntrinsic::WarpMatchAll => "__match_all_sync",
            GpuIntrinsic::WarpReduceAdd => "__reduce_add_sync",
            GpuIntrinsic::WarpReduceMin => "__reduce_min_sync",
            GpuIntrinsic::WarpReduceMax => "__reduce_max_sync",
            GpuIntrinsic::WarpReduceAnd => "__reduce_and_sync",
            GpuIntrinsic::WarpReduceOr => "__reduce_or_sync",
            GpuIntrinsic::WarpReduceXor => "__reduce_xor_sync",

            // Bit manipulation
            GpuIntrinsic::Popc => "__popc",
            GpuIntrinsic::Clz => "__clz",
            GpuIntrinsic::Ctz => "__ffs", // ffs returns 1 + ctz, but commonly used
            GpuIntrinsic::Ffs => "__ffs",
            GpuIntrinsic::Brev => "__brev",
            GpuIntrinsic::BytePerm => "__byte_perm",
            GpuIntrinsic::FunnelShiftLeft => "__funnelshift_l",
            GpuIntrinsic::FunnelShiftRight => "__funnelshift_r",

            // Memory operations
            GpuIntrinsic::Ldg => "__ldg",
            GpuIntrinsic::PrefetchL1 => "__prefetch_l1",
            GpuIntrinsic::PrefetchL2 => "__prefetch_l2",

            // Special functions
            GpuIntrinsic::Rcp => "__frcp_rn",
            GpuIntrinsic::Fdividef => "__fdividef",
            GpuIntrinsic::Saturate => "__saturatef",
            GpuIntrinsic::J0 => "j0f",
            GpuIntrinsic::J1 => "j1f",
            GpuIntrinsic::Jn => "jnf",
            GpuIntrinsic::Y0 => "y0f",
            GpuIntrinsic::Y1 => "y1f",
            GpuIntrinsic::Yn => "ynf",
            GpuIntrinsic::Normcdf => "normcdff",
            GpuIntrinsic::Normcdfinv => "normcdfinvf",
            GpuIntrinsic::CylBesselI0 => "cyl_bessel_i0f",
            GpuIntrinsic::CylBesselI1 => "cyl_bessel_i1f",

            // Thread/block indices
            GpuIntrinsic::ThreadIdxX => "threadIdx.x",
            GpuIntrinsic::ThreadIdxY => "threadIdx.y",
            GpuIntrinsic::ThreadIdxZ => "threadIdx.z",
            GpuIntrinsic::BlockIdxX => "blockIdx.x",
            GpuIntrinsic::BlockIdxY => "blockIdx.y",
            GpuIntrinsic::BlockIdxZ => "blockIdx.z",
            GpuIntrinsic::BlockDimX => "blockDim.x",
            GpuIntrinsic::BlockDimY => "blockDim.y",
            GpuIntrinsic::BlockDimZ => "blockDim.z",
            GpuIntrinsic::GridDimX => "gridDim.x",
            GpuIntrinsic::GridDimY => "gridDim.y",
            GpuIntrinsic::GridDimZ => "gridDim.z",
            GpuIntrinsic::WarpSize => "warpSize",

            // Clock and timing
            GpuIntrinsic::Clock => "clock()",
            GpuIntrinsic::Clock64 => "clock64()",
            GpuIntrinsic::Nanosleep => "__nanosleep",
        }
    }

    /// Check if this intrinsic is a value (no parentheses needed).
    pub fn is_value_intrinsic(&self) -> bool {
        matches!(
            self,
            GpuIntrinsic::ThreadIdxX
                | GpuIntrinsic::ThreadIdxY
                | GpuIntrinsic::ThreadIdxZ
                | GpuIntrinsic::BlockIdxX
                | GpuIntrinsic::BlockIdxY
                | GpuIntrinsic::BlockIdxZ
                | GpuIntrinsic::BlockDimX
                | GpuIntrinsic::BlockDimY
                | GpuIntrinsic::BlockDimZ
                | GpuIntrinsic::GridDimX
                | GpuIntrinsic::GridDimY
                | GpuIntrinsic::GridDimZ
                | GpuIntrinsic::WarpSize
        )
    }

    /// Check if this intrinsic is a zero-argument function (ends with ()).
    pub fn is_zero_arg_function(&self) -> bool {
        matches!(
            self,
            GpuIntrinsic::SyncThreads
                | GpuIntrinsic::ThreadFence
                | GpuIntrinsic::ThreadFenceBlock
                | GpuIntrinsic::ThreadFenceSystem
                | GpuIntrinsic::WarpActiveMask
                | GpuIntrinsic::Clock
                | GpuIntrinsic::Clock64
        )
    }

    /// Check if this intrinsic requires a mask argument (warp operations).
    pub fn requires_mask(&self) -> bool {
        matches!(
            self,
            GpuIntrinsic::WarpShfl
                | GpuIntrinsic::WarpShflUp
                | GpuIntrinsic::WarpShflDown
                | GpuIntrinsic::WarpShflXor
                | GpuIntrinsic::WarpBallot
                | GpuIntrinsic::WarpAll
                | GpuIntrinsic::WarpAny
                | GpuIntrinsic::WarpMatchAny
                | GpuIntrinsic::WarpMatchAll
                | GpuIntrinsic::WarpReduceAdd
                | GpuIntrinsic::WarpReduceMin
                | GpuIntrinsic::WarpReduceMax
                | GpuIntrinsic::WarpReduceAnd
                | GpuIntrinsic::WarpReduceOr
                | GpuIntrinsic::WarpReduceXor
        )
    }

    /// Get the category of this intrinsic for documentation purposes.
    pub fn category(&self) -> &'static str {
        match self {
            GpuIntrinsic::SyncThreads
            | GpuIntrinsic::ThreadFence
            | GpuIntrinsic::ThreadFenceBlock
            | GpuIntrinsic::ThreadFenceSystem
            | GpuIntrinsic::SyncThreadsCount
            | GpuIntrinsic::SyncThreadsAnd
            | GpuIntrinsic::SyncThreadsOr => "synchronization",

            GpuIntrinsic::AtomicAdd
            | GpuIntrinsic::AtomicSub
            | GpuIntrinsic::AtomicMin
            | GpuIntrinsic::AtomicMax
            | GpuIntrinsic::AtomicExch
            | GpuIntrinsic::AtomicCas
            | GpuIntrinsic::AtomicAnd
            | GpuIntrinsic::AtomicOr
            | GpuIntrinsic::AtomicXor
            | GpuIntrinsic::AtomicInc
            | GpuIntrinsic::AtomicDec => "atomic",

            GpuIntrinsic::Sqrt
            | GpuIntrinsic::Rsqrt
            | GpuIntrinsic::Abs
            | GpuIntrinsic::Fabs
            | GpuIntrinsic::Floor
            | GpuIntrinsic::Ceil
            | GpuIntrinsic::Round
            | GpuIntrinsic::Trunc
            | GpuIntrinsic::Fma
            | GpuIntrinsic::Min
            | GpuIntrinsic::Max
            | GpuIntrinsic::Fmod
            | GpuIntrinsic::Remainder
            | GpuIntrinsic::Copysign
            | GpuIntrinsic::Cbrt
            | GpuIntrinsic::Hypot => "math",

            GpuIntrinsic::Sin
            | GpuIntrinsic::Cos
            | GpuIntrinsic::Tan
            | GpuIntrinsic::Asin
            | GpuIntrinsic::Acos
            | GpuIntrinsic::Atan
            | GpuIntrinsic::Atan2
            | GpuIntrinsic::Sincos
            | GpuIntrinsic::Sinpi
            | GpuIntrinsic::Cospi => "trigonometric",

            GpuIntrinsic::Sinh
            | GpuIntrinsic::Cosh
            | GpuIntrinsic::Tanh
            | GpuIntrinsic::Asinh
            | GpuIntrinsic::Acosh
            | GpuIntrinsic::Atanh => "hyperbolic",

            GpuIntrinsic::Exp
            | GpuIntrinsic::Exp2
            | GpuIntrinsic::Exp10
            | GpuIntrinsic::Expm1
            | GpuIntrinsic::Log
            | GpuIntrinsic::Log2
            | GpuIntrinsic::Log10
            | GpuIntrinsic::Log1p
            | GpuIntrinsic::Pow
            | GpuIntrinsic::Ldexp
            | GpuIntrinsic::Scalbn
            | GpuIntrinsic::Ilogb
            | GpuIntrinsic::Lgamma
            | GpuIntrinsic::Tgamma
            | GpuIntrinsic::Erf
            | GpuIntrinsic::Erfc
            | GpuIntrinsic::Erfinv
            | GpuIntrinsic::Erfcinv => "exponential",

            GpuIntrinsic::Isnan
            | GpuIntrinsic::Isinf
            | GpuIntrinsic::Isfinite
            | GpuIntrinsic::Isnormal
            | GpuIntrinsic::Signbit
            | GpuIntrinsic::Nextafter
            | GpuIntrinsic::Fdim
            | GpuIntrinsic::Nan => "classification",

            GpuIntrinsic::WarpShfl
            | GpuIntrinsic::WarpShflUp
            | GpuIntrinsic::WarpShflDown
            | GpuIntrinsic::WarpShflXor
            | GpuIntrinsic::WarpActiveMask
            | GpuIntrinsic::WarpBallot
            | GpuIntrinsic::WarpAll
            | GpuIntrinsic::WarpAny
            | GpuIntrinsic::WarpMatchAny
            | GpuIntrinsic::WarpMatchAll
            | GpuIntrinsic::WarpReduceAdd
            | GpuIntrinsic::WarpReduceMin
            | GpuIntrinsic::WarpReduceMax
            | GpuIntrinsic::WarpReduceAnd
            | GpuIntrinsic::WarpReduceOr
            | GpuIntrinsic::WarpReduceXor => "warp",

            GpuIntrinsic::Popc
            | GpuIntrinsic::Clz
            | GpuIntrinsic::Ctz
            | GpuIntrinsic::Ffs
            | GpuIntrinsic::Brev
            | GpuIntrinsic::BytePerm
            | GpuIntrinsic::FunnelShiftLeft
            | GpuIntrinsic::FunnelShiftRight => "bit",

            GpuIntrinsic::Ldg | GpuIntrinsic::PrefetchL1 | GpuIntrinsic::PrefetchL2 => "memory",

            GpuIntrinsic::Rcp
            | GpuIntrinsic::Fdividef
            | GpuIntrinsic::Saturate
            | GpuIntrinsic::J0
            | GpuIntrinsic::J1
            | GpuIntrinsic::Jn
            | GpuIntrinsic::Y0
            | GpuIntrinsic::Y1
            | GpuIntrinsic::Yn
            | GpuIntrinsic::Normcdf
            | GpuIntrinsic::Normcdfinv
            | GpuIntrinsic::CylBesselI0
            | GpuIntrinsic::CylBesselI1 => "special",

            GpuIntrinsic::ThreadIdxX
            | GpuIntrinsic::ThreadIdxY
            | GpuIntrinsic::ThreadIdxZ
            | GpuIntrinsic::BlockIdxX
            | GpuIntrinsic::BlockIdxY
            | GpuIntrinsic::BlockIdxZ
            | GpuIntrinsic::BlockDimX
            | GpuIntrinsic::BlockDimY
            | GpuIntrinsic::BlockDimZ
            | GpuIntrinsic::GridDimX
            | GpuIntrinsic::GridDimY
            | GpuIntrinsic::GridDimZ
            | GpuIntrinsic::WarpSize => "index",

            GpuIntrinsic::Clock | GpuIntrinsic::Clock64 | GpuIntrinsic::Nanosleep => "timing",
        }
    }
}

/// Registry for mapping Rust function names to GPU intrinsics.
#[derive(Debug)]
pub struct IntrinsicRegistry {
    mappings: HashMap<String, GpuIntrinsic>,
}

impl Default for IntrinsicRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl IntrinsicRegistry {
    /// Create a new registry with default mappings.
    pub fn new() -> Self {
        let mut mappings = HashMap::new();

        // === Synchronization ===
        mappings.insert("sync_threads".to_string(), GpuIntrinsic::SyncThreads);
        mappings.insert("thread_fence".to_string(), GpuIntrinsic::ThreadFence);
        mappings.insert(
            "thread_fence_block".to_string(),
            GpuIntrinsic::ThreadFenceBlock,
        );
        mappings.insert(
            "thread_fence_system".to_string(),
            GpuIntrinsic::ThreadFenceSystem,
        );
        mappings.insert(
            "sync_threads_count".to_string(),
            GpuIntrinsic::SyncThreadsCount,
        );
        mappings.insert("sync_threads_and".to_string(), GpuIntrinsic::SyncThreadsAnd);
        mappings.insert("sync_threads_or".to_string(), GpuIntrinsic::SyncThreadsOr);

        // === Atomic operations ===
        mappings.insert("atomic_add".to_string(), GpuIntrinsic::AtomicAdd);
        mappings.insert("atomic_sub".to_string(), GpuIntrinsic::AtomicSub);
        mappings.insert("atomic_min".to_string(), GpuIntrinsic::AtomicMin);
        mappings.insert("atomic_max".to_string(), GpuIntrinsic::AtomicMax);
        mappings.insert("atomic_exchange".to_string(), GpuIntrinsic::AtomicExch);
        mappings.insert("atomic_exch".to_string(), GpuIntrinsic::AtomicExch);
        mappings.insert("atomic_cas".to_string(), GpuIntrinsic::AtomicCas);
        mappings.insert("atomic_compare_swap".to_string(), GpuIntrinsic::AtomicCas);
        mappings.insert("atomic_and".to_string(), GpuIntrinsic::AtomicAnd);
        mappings.insert("atomic_or".to_string(), GpuIntrinsic::AtomicOr);
        mappings.insert("atomic_xor".to_string(), GpuIntrinsic::AtomicXor);
        mappings.insert("atomic_inc".to_string(), GpuIntrinsic::AtomicInc);
        mappings.insert("atomic_dec".to_string(), GpuIntrinsic::AtomicDec);

        // === Basic math functions ===
        mappings.insert("sqrt".to_string(), GpuIntrinsic::Sqrt);
        mappings.insert("rsqrt".to_string(), GpuIntrinsic::Rsqrt);
        mappings.insert("abs".to_string(), GpuIntrinsic::Fabs);
        mappings.insert("fabs".to_string(), GpuIntrinsic::Fabs);
        mappings.insert("floor".to_string(), GpuIntrinsic::Floor);
        mappings.insert("ceil".to_string(), GpuIntrinsic::Ceil);
        mappings.insert("round".to_string(), GpuIntrinsic::Round);
        mappings.insert("trunc".to_string(), GpuIntrinsic::Trunc);
        mappings.insert("mul_add".to_string(), GpuIntrinsic::Fma);
        mappings.insert("fma".to_string(), GpuIntrinsic::Fma);
        mappings.insert("min".to_string(), GpuIntrinsic::Min);
        mappings.insert("max".to_string(), GpuIntrinsic::Max);
        mappings.insert("fmin".to_string(), GpuIntrinsic::Min);
        mappings.insert("fmax".to_string(), GpuIntrinsic::Max);
        mappings.insert("fmod".to_string(), GpuIntrinsic::Fmod);
        mappings.insert("remainder".to_string(), GpuIntrinsic::Remainder);
        mappings.insert("copysign".to_string(), GpuIntrinsic::Copysign);
        mappings.insert("cbrt".to_string(), GpuIntrinsic::Cbrt);
        mappings.insert("hypot".to_string(), GpuIntrinsic::Hypot);

        // === Trigonometric functions ===
        mappings.insert("sin".to_string(), GpuIntrinsic::Sin);
        mappings.insert("cos".to_string(), GpuIntrinsic::Cos);
        mappings.insert("tan".to_string(), GpuIntrinsic::Tan);
        mappings.insert("asin".to_string(), GpuIntrinsic::Asin);
        mappings.insert("acos".to_string(), GpuIntrinsic::Acos);
        mappings.insert("atan".to_string(), GpuIntrinsic::Atan);
        mappings.insert("atan2".to_string(), GpuIntrinsic::Atan2);
        mappings.insert("sincos".to_string(), GpuIntrinsic::Sincos);
        mappings.insert("sinpi".to_string(), GpuIntrinsic::Sinpi);
        mappings.insert("cospi".to_string(), GpuIntrinsic::Cospi);

        // === Hyperbolic functions ===
        mappings.insert("sinh".to_string(), GpuIntrinsic::Sinh);
        mappings.insert("cosh".to_string(), GpuIntrinsic::Cosh);
        mappings.insert("tanh".to_string(), GpuIntrinsic::Tanh);
        mappings.insert("asinh".to_string(), GpuIntrinsic::Asinh);
        mappings.insert("acosh".to_string(), GpuIntrinsic::Acosh);
        mappings.insert("atanh".to_string(), GpuIntrinsic::Atanh);

        // === Exponential and logarithmic ===
        mappings.insert("exp".to_string(), GpuIntrinsic::Exp);
        mappings.insert("exp2".to_string(), GpuIntrinsic::Exp2);
        mappings.insert("exp10".to_string(), GpuIntrinsic::Exp10);
        mappings.insert("expm1".to_string(), GpuIntrinsic::Expm1);
        mappings.insert("ln".to_string(), GpuIntrinsic::Log);
        mappings.insert("log".to_string(), GpuIntrinsic::Log);
        mappings.insert("log2".to_string(), GpuIntrinsic::Log2);
        mappings.insert("log10".to_string(), GpuIntrinsic::Log10);
        mappings.insert("log1p".to_string(), GpuIntrinsic::Log1p);
        mappings.insert("powf".to_string(), GpuIntrinsic::Pow);
        mappings.insert("powi".to_string(), GpuIntrinsic::Pow);
        mappings.insert("pow".to_string(), GpuIntrinsic::Pow);
        mappings.insert("ldexp".to_string(), GpuIntrinsic::Ldexp);
        mappings.insert("scalbn".to_string(), GpuIntrinsic::Scalbn);
        mappings.insert("ilogb".to_string(), GpuIntrinsic::Ilogb);
        mappings.insert("lgamma".to_string(), GpuIntrinsic::Lgamma);
        mappings.insert("tgamma".to_string(), GpuIntrinsic::Tgamma);
        mappings.insert("gamma".to_string(), GpuIntrinsic::Tgamma);
        mappings.insert("erf".to_string(), GpuIntrinsic::Erf);
        mappings.insert("erfc".to_string(), GpuIntrinsic::Erfc);
        mappings.insert("erfinv".to_string(), GpuIntrinsic::Erfinv);
        mappings.insert("erfcinv".to_string(), GpuIntrinsic::Erfcinv);

        // === Classification and comparison ===
        mappings.insert("is_nan".to_string(), GpuIntrinsic::Isnan);
        mappings.insert("isnan".to_string(), GpuIntrinsic::Isnan);
        mappings.insert("is_infinite".to_string(), GpuIntrinsic::Isinf);
        mappings.insert("isinf".to_string(), GpuIntrinsic::Isinf);
        mappings.insert("is_finite".to_string(), GpuIntrinsic::Isfinite);
        mappings.insert("isfinite".to_string(), GpuIntrinsic::Isfinite);
        mappings.insert("is_normal".to_string(), GpuIntrinsic::Isnormal);
        mappings.insert("isnormal".to_string(), GpuIntrinsic::Isnormal);
        mappings.insert("is_sign_negative".to_string(), GpuIntrinsic::Signbit);
        mappings.insert("signbit".to_string(), GpuIntrinsic::Signbit);
        mappings.insert("nextafter".to_string(), GpuIntrinsic::Nextafter);
        mappings.insert("fdim".to_string(), GpuIntrinsic::Fdim);
        mappings.insert("nan".to_string(), GpuIntrinsic::Nan);

        // === Warp operations ===
        mappings.insert("warp_shfl".to_string(), GpuIntrinsic::WarpShfl);
        mappings.insert("warp_shuffle".to_string(), GpuIntrinsic::WarpShfl);
        mappings.insert("warp_shfl_up".to_string(), GpuIntrinsic::WarpShflUp);
        mappings.insert("warp_shuffle_up".to_string(), GpuIntrinsic::WarpShflUp);
        mappings.insert("warp_shfl_down".to_string(), GpuIntrinsic::WarpShflDown);
        mappings.insert("warp_shuffle_down".to_string(), GpuIntrinsic::WarpShflDown);
        mappings.insert("warp_shfl_xor".to_string(), GpuIntrinsic::WarpShflXor);
        mappings.insert("warp_shuffle_xor".to_string(), GpuIntrinsic::WarpShflXor);
        mappings.insert("warp_active_mask".to_string(), GpuIntrinsic::WarpActiveMask);
        mappings.insert("active_mask".to_string(), GpuIntrinsic::WarpActiveMask);
        mappings.insert("warp_ballot".to_string(), GpuIntrinsic::WarpBallot);
        mappings.insert("ballot".to_string(), GpuIntrinsic::WarpBallot);
        mappings.insert("warp_all".to_string(), GpuIntrinsic::WarpAll);
        mappings.insert("warp_any".to_string(), GpuIntrinsic::WarpAny);
        mappings.insert("warp_match_any".to_string(), GpuIntrinsic::WarpMatchAny);
        mappings.insert("warp_match_all".to_string(), GpuIntrinsic::WarpMatchAll);
        mappings.insert("warp_reduce_add".to_string(), GpuIntrinsic::WarpReduceAdd);
        mappings.insert("warp_reduce_min".to_string(), GpuIntrinsic::WarpReduceMin);
        mappings.insert("warp_reduce_max".to_string(), GpuIntrinsic::WarpReduceMax);
        mappings.insert("warp_reduce_and".to_string(), GpuIntrinsic::WarpReduceAnd);
        mappings.insert("warp_reduce_or".to_string(), GpuIntrinsic::WarpReduceOr);
        mappings.insert("warp_reduce_xor".to_string(), GpuIntrinsic::WarpReduceXor);

        // === Bit manipulation ===
        mappings.insert("popc".to_string(), GpuIntrinsic::Popc);
        mappings.insert("popcount".to_string(), GpuIntrinsic::Popc);
        mappings.insert("count_ones".to_string(), GpuIntrinsic::Popc);
        mappings.insert("clz".to_string(), GpuIntrinsic::Clz);
        mappings.insert("leading_zeros".to_string(), GpuIntrinsic::Clz);
        mappings.insert("ctz".to_string(), GpuIntrinsic::Ctz);
        mappings.insert("trailing_zeros".to_string(), GpuIntrinsic::Ctz);
        mappings.insert("ffs".to_string(), GpuIntrinsic::Ffs);
        mappings.insert("brev".to_string(), GpuIntrinsic::Brev);
        mappings.insert("reverse_bits".to_string(), GpuIntrinsic::Brev);
        mappings.insert("byte_perm".to_string(), GpuIntrinsic::BytePerm);
        mappings.insert(
            "funnel_shift_left".to_string(),
            GpuIntrinsic::FunnelShiftLeft,
        );
        mappings.insert(
            "funnel_shift_right".to_string(),
            GpuIntrinsic::FunnelShiftRight,
        );

        // === Memory operations ===
        mappings.insert("ldg".to_string(), GpuIntrinsic::Ldg);
        mappings.insert("load_global".to_string(), GpuIntrinsic::Ldg);
        mappings.insert("prefetch_l1".to_string(), GpuIntrinsic::PrefetchL1);
        mappings.insert("prefetch_l2".to_string(), GpuIntrinsic::PrefetchL2);

        // === Special functions ===
        mappings.insert("rcp".to_string(), GpuIntrinsic::Rcp);
        mappings.insert("recip".to_string(), GpuIntrinsic::Rcp);
        mappings.insert("fdividef".to_string(), GpuIntrinsic::Fdividef);
        mappings.insert("fast_div".to_string(), GpuIntrinsic::Fdividef);
        mappings.insert("saturate".to_string(), GpuIntrinsic::Saturate);
        mappings.insert("clamp_01".to_string(), GpuIntrinsic::Saturate);
        mappings.insert("j0".to_string(), GpuIntrinsic::J0);
        mappings.insert("j1".to_string(), GpuIntrinsic::J1);
        mappings.insert("jn".to_string(), GpuIntrinsic::Jn);
        mappings.insert("y0".to_string(), GpuIntrinsic::Y0);
        mappings.insert("y1".to_string(), GpuIntrinsic::Y1);
        mappings.insert("yn".to_string(), GpuIntrinsic::Yn);
        mappings.insert("normcdf".to_string(), GpuIntrinsic::Normcdf);
        mappings.insert("norm_cdf".to_string(), GpuIntrinsic::Normcdf);
        mappings.insert("normcdfinv".to_string(), GpuIntrinsic::Normcdfinv);
        mappings.insert("norm_cdf_inv".to_string(), GpuIntrinsic::Normcdfinv);
        mappings.insert("cyl_bessel_i0".to_string(), GpuIntrinsic::CylBesselI0);
        mappings.insert("cyl_bessel_i1".to_string(), GpuIntrinsic::CylBesselI1);

        // === Thread/block indices ===
        mappings.insert("thread_idx_x".to_string(), GpuIntrinsic::ThreadIdxX);
        mappings.insert("thread_idx_y".to_string(), GpuIntrinsic::ThreadIdxY);
        mappings.insert("thread_idx_z".to_string(), GpuIntrinsic::ThreadIdxZ);
        mappings.insert("block_idx_x".to_string(), GpuIntrinsic::BlockIdxX);
        mappings.insert("block_idx_y".to_string(), GpuIntrinsic::BlockIdxY);
        mappings.insert("block_idx_z".to_string(), GpuIntrinsic::BlockIdxZ);
        mappings.insert("block_dim_x".to_string(), GpuIntrinsic::BlockDimX);
        mappings.insert("block_dim_y".to_string(), GpuIntrinsic::BlockDimY);
        mappings.insert("block_dim_z".to_string(), GpuIntrinsic::BlockDimZ);
        mappings.insert("grid_dim_x".to_string(), GpuIntrinsic::GridDimX);
        mappings.insert("grid_dim_y".to_string(), GpuIntrinsic::GridDimY);
        mappings.insert("grid_dim_z".to_string(), GpuIntrinsic::GridDimZ);
        mappings.insert("warp_size".to_string(), GpuIntrinsic::WarpSize);

        // === Clock and timing ===
        mappings.insert("clock".to_string(), GpuIntrinsic::Clock);
        mappings.insert("clock64".to_string(), GpuIntrinsic::Clock64);
        mappings.insert("nanosleep".to_string(), GpuIntrinsic::Nanosleep);

        Self { mappings }
    }

    /// Look up an intrinsic by Rust function name.
    pub fn lookup(&self, name: &str) -> Option<&GpuIntrinsic> {
        self.mappings.get(name)
    }

    /// Register a custom intrinsic mapping.
    pub fn register(&mut self, rust_name: &str, intrinsic: GpuIntrinsic) {
        self.mappings.insert(rust_name.to_string(), intrinsic);
    }

    /// Check if a name is a known intrinsic.
    pub fn is_intrinsic(&self, name: &str) -> bool {
        self.mappings.contains_key(name)
    }
}

/// Stencil-specific intrinsics for neighbor access.
///
/// These are special intrinsics that the transpiler handles
/// differently based on stencil configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum StencilIntrinsic {
    /// Get current cell index: `pos.idx()`
    Index,
    /// Access north neighbor: `pos.north(buf)`
    North,
    /// Access south neighbor: `pos.south(buf)`
    South,
    /// Access east neighbor: `pos.east(buf)`
    East,
    /// Access west neighbor: `pos.west(buf)`
    West,
    /// Access neighbor at offset: `pos.at(buf, dx, dy)`
    At,
    /// 3D: Access neighbor above: `pos.up(buf)`
    Up,
    /// 3D: Access neighbor below: `pos.down(buf)`
    Down,
}

impl StencilIntrinsic {
    /// Parse a method name to stencil intrinsic.
    pub fn from_method_name(name: &str) -> Option<Self> {
        match name {
            "idx" => Some(StencilIntrinsic::Index),
            "north" => Some(StencilIntrinsic::North),
            "south" => Some(StencilIntrinsic::South),
            "east" => Some(StencilIntrinsic::East),
            "west" => Some(StencilIntrinsic::West),
            "at" => Some(StencilIntrinsic::At),
            "up" => Some(StencilIntrinsic::Up),
            "down" => Some(StencilIntrinsic::Down),
            _ => None,
        }
    }
}

/// Ring kernel intrinsics for persistent actor kernels.
///
/// These intrinsics provide access to control block state, queue operations,
/// and HLC (Hybrid Logical Clock) functionality within ring kernel handlers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingKernelIntrinsic {
    // === Control Block Access ===
    /// Check if kernel is active: `is_active()`
    IsActive,
    /// Check if termination requested: `should_terminate()`
    ShouldTerminate,
    /// Mark kernel as terminated: `mark_terminated()`
    MarkTerminated,
    /// Get messages processed count: `messages_processed()`
    GetMessagesProcessed,

    // === Queue Operations ===
    /// Get input queue size: `input_queue_size()`
    InputQueueSize,
    /// Get output queue size: `output_queue_size()`
    OutputQueueSize,
    /// Check if input queue empty: `input_queue_empty()`
    InputQueueEmpty,
    /// Check if output queue empty: `output_queue_empty()`
    OutputQueueEmpty,
    /// Enqueue a response: `enqueue_response(&response)`
    EnqueueResponse,

    // === HLC Operations ===
    /// Increment HLC logical counter: `hlc_tick()`
    HlcTick,
    /// Update HLC with received timestamp: `hlc_update(received_ts)`
    HlcUpdate,
    /// Get current HLC timestamp: `hlc_now()`
    HlcNow,

    // === K2K Operations ===
    /// Send message to another kernel: `k2k_send(target_id, &msg)`
    K2kSend,
    /// Try to receive K2K message: `k2k_try_recv()`
    K2kTryRecv,
    /// Check for K2K messages: `k2k_has_message()`
    K2kHasMessage,
    /// Peek at next K2K message without consuming: `k2k_peek()`
    K2kPeek,
    /// Get number of pending K2K messages: `k2k_pending_count()`
    K2kPendingCount,

    // === Timing ===
    /// Sleep for nanoseconds: `nanosleep(ns)`
    Nanosleep,
}

impl RingKernelIntrinsic {
    /// Get the CUDA code for this intrinsic.
    pub fn to_cuda(&self, args: &[String]) -> String {
        match self {
            Self::IsActive => "atomicAdd(&control->is_active, 0) != 0".to_string(),
            Self::ShouldTerminate => "atomicAdd(&control->should_terminate, 0) != 0".to_string(),
            Self::MarkTerminated => "atomicExch(&control->has_terminated, 1)".to_string(),
            Self::GetMessagesProcessed => "atomicAdd(&control->messages_processed, 0)".to_string(),

            Self::InputQueueSize => {
                "(atomicAdd(&control->input_head, 0) - atomicAdd(&control->input_tail, 0))"
                    .to_string()
            }
            Self::OutputQueueSize => {
                "(atomicAdd(&control->output_head, 0) - atomicAdd(&control->output_tail, 0))"
                    .to_string()
            }
            Self::InputQueueEmpty => {
                "(atomicAdd(&control->input_head, 0) == atomicAdd(&control->input_tail, 0))"
                    .to_string()
            }
            Self::OutputQueueEmpty => {
                "(atomicAdd(&control->output_head, 0) == atomicAdd(&control->output_tail, 0))"
                    .to_string()
            }
            Self::EnqueueResponse => {
                if !args.is_empty() {
                    format!(
                        "{{ unsigned long long _out_idx = atomicAdd(&control->output_head, 1) & control->output_mask; \
                         memcpy(&output_buffer[_out_idx * RESP_SIZE], {}, RESP_SIZE); }}",
                        args[0]
                    )
                } else {
                    "/* enqueue_response requires response pointer */".to_string()
                }
            }

            Self::HlcTick => "hlc_logical++".to_string(),
            Self::HlcUpdate => {
                if !args.is_empty() {
                    format!(
                        "{{ if ({} > hlc_physical) {{ hlc_physical = {}; hlc_logical = 0; }} else {{ hlc_logical++; }} }}",
                        args[0], args[0]
                    )
                } else {
                    "hlc_logical++".to_string()
                }
            }
            Self::HlcNow => "(hlc_physical << 32) | (hlc_logical & 0xFFFFFFFF)".to_string(),

            Self::K2kSend => {
                if args.len() >= 2 {
                    // k2k_send(target_id, msg_ptr) -> k2k_send(k2k_routes, target_id, msg_ptr, sizeof(*msg_ptr))
                    format!(
                        "k2k_send(k2k_routes, {}, {}, sizeof(*{}))",
                        args[0], args[1], args[1]
                    )
                } else {
                    "/* k2k_send requires target_id and msg_ptr */".to_string()
                }
            }
            Self::K2kTryRecv => "k2k_try_recv(k2k_inbox)".to_string(),
            Self::K2kHasMessage => "k2k_has_message(k2k_inbox)".to_string(),
            Self::K2kPeek => "k2k_peek(k2k_inbox)".to_string(),
            Self::K2kPendingCount => "k2k_pending_count(k2k_inbox)".to_string(),

            Self::Nanosleep => {
                if !args.is_empty() {
                    format!("__nanosleep({})", args[0])
                } else {
                    "__nanosleep(1000)".to_string()
                }
            }
        }
    }

    /// Parse a function name to get the intrinsic.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "is_active" | "is_kernel_active" => Some(Self::IsActive),
            "should_terminate" => Some(Self::ShouldTerminate),
            "mark_terminated" => Some(Self::MarkTerminated),
            "messages_processed" | "get_messages_processed" => Some(Self::GetMessagesProcessed),

            "input_queue_size" => Some(Self::InputQueueSize),
            "output_queue_size" => Some(Self::OutputQueueSize),
            "input_queue_empty" => Some(Self::InputQueueEmpty),
            "output_queue_empty" => Some(Self::OutputQueueEmpty),
            "enqueue_response" | "enqueue" => Some(Self::EnqueueResponse),

            "hlc_tick" => Some(Self::HlcTick),
            "hlc_update" => Some(Self::HlcUpdate),
            "hlc_now" => Some(Self::HlcNow),

            "k2k_send" => Some(Self::K2kSend),
            "k2k_try_recv" => Some(Self::K2kTryRecv),
            "k2k_has_message" => Some(Self::K2kHasMessage),
            "k2k_peek" => Some(Self::K2kPeek),
            "k2k_pending_count" | "k2k_pending" => Some(Self::K2kPendingCount),

            "nanosleep" => Some(Self::Nanosleep),

            _ => None,
        }
    }

    /// Check if this intrinsic requires the control block.
    pub fn requires_control_block(&self) -> bool {
        matches!(
            self,
            Self::IsActive
                | Self::ShouldTerminate
                | Self::MarkTerminated
                | Self::GetMessagesProcessed
                | Self::InputQueueSize
                | Self::OutputQueueSize
                | Self::InputQueueEmpty
                | Self::OutputQueueEmpty
                | Self::EnqueueResponse
        )
    }

    /// Check if this intrinsic requires HLC state.
    pub fn requires_hlc(&self) -> bool {
        matches!(self, Self::HlcTick | Self::HlcUpdate | Self::HlcNow)
    }

    /// Check if this intrinsic requires K2K support.
    pub fn requires_k2k(&self) -> bool {
        matches!(
            self,
            Self::K2kSend
                | Self::K2kTryRecv
                | Self::K2kHasMessage
                | Self::K2kPeek
                | Self::K2kPendingCount
        )
    }
}

impl StencilIntrinsic {
    /// Get the index offset for 2D stencil (relative to buffer_width).
    ///
    /// Returns (row_offset, col_offset) where final offset is:
    /// `row_offset * buffer_width + col_offset`
    pub fn get_offset_2d(&self) -> Option<(i32, i32)> {
        match self {
            StencilIntrinsic::Index => Some((0, 0)),
            StencilIntrinsic::North => Some((-1, 0)),
            StencilIntrinsic::South => Some((1, 0)),
            StencilIntrinsic::East => Some((0, 1)),
            StencilIntrinsic::West => Some((0, -1)),
            StencilIntrinsic::At => None, // Requires runtime offset
            StencilIntrinsic::Up | StencilIntrinsic::Down => None, // 3D only
        }
    }

    /// Get the index offset for 3D stencil.
    ///
    /// Returns (z_offset, row_offset, col_offset) where final offset is:
    /// `z_offset * buffer_slice + row_offset * buffer_width + col_offset`
    pub fn get_offset_3d(&self) -> Option<(i32, i32, i32)> {
        match self {
            StencilIntrinsic::Index => Some((0, 0, 0)),
            StencilIntrinsic::North => Some((0, -1, 0)),
            StencilIntrinsic::South => Some((0, 1, 0)),
            StencilIntrinsic::East => Some((0, 0, 1)),
            StencilIntrinsic::West => Some((0, 0, -1)),
            StencilIntrinsic::Up => Some((-1, 0, 0)),
            StencilIntrinsic::Down => Some((1, 0, 0)),
            StencilIntrinsic::At => None, // Requires runtime offset
        }
    }

    /// Check if this is a 3D-only intrinsic.
    pub fn is_3d_only(&self) -> bool {
        matches!(self, StencilIntrinsic::Up | StencilIntrinsic::Down)
    }

    /// Generate CUDA index expression for 2D stencil.
    ///
    /// # Arguments
    /// * `buffer_name` - Name of the buffer variable
    /// * `buffer_width` - Width expression (e.g., "18" for tile_size + 2*halo)
    /// * `idx_var` - Name of the current index variable
    pub fn to_cuda_index_2d(&self, buffer_name: &str, buffer_width: &str, idx_var: &str) -> String {
        match self {
            StencilIntrinsic::Index => format!("{}[{}]", buffer_name, idx_var),
            StencilIntrinsic::North => {
                format!("{}[{} - {}]", buffer_name, idx_var, buffer_width)
            }
            StencilIntrinsic::South => {
                format!("{}[{} + {}]", buffer_name, idx_var, buffer_width)
            }
            StencilIntrinsic::East => format!("{}[{} + 1]", buffer_name, idx_var),
            StencilIntrinsic::West => format!("{}[{} - 1]", buffer_name, idx_var),
            StencilIntrinsic::At => {
                // This should be handled specially with provided offsets
                format!("{}[{}]", buffer_name, idx_var)
            }
            _ => format!("{}[{}]", buffer_name, idx_var),
        }
    }

    /// Generate CUDA index expression for 3D stencil.
    ///
    /// # Arguments
    /// * `buffer_name` - Name of the buffer variable
    /// * `buffer_width` - Width expression
    /// * `buffer_slice` - Slice size expression (width * height)
    /// * `idx_var` - Name of the current index variable
    pub fn to_cuda_index_3d(
        &self,
        buffer_name: &str,
        buffer_width: &str,
        buffer_slice: &str,
        idx_var: &str,
    ) -> String {
        match self {
            StencilIntrinsic::Index => format!("{}[{}]", buffer_name, idx_var),
            StencilIntrinsic::North => {
                format!("{}[{} - {}]", buffer_name, idx_var, buffer_width)
            }
            StencilIntrinsic::South => {
                format!("{}[{} + {}]", buffer_name, idx_var, buffer_width)
            }
            StencilIntrinsic::East => format!("{}[{} + 1]", buffer_name, idx_var),
            StencilIntrinsic::West => format!("{}[{} - 1]", buffer_name, idx_var),
            StencilIntrinsic::Up => {
                format!("{}[{} - {}]", buffer_name, idx_var, buffer_slice)
            }
            StencilIntrinsic::Down => {
                format!("{}[{} + {}]", buffer_name, idx_var, buffer_slice)
            }
            StencilIntrinsic::At => {
                // This should be handled specially with provided offsets
                format!("{}[{}]", buffer_name, idx_var)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intrinsic_lookup() {
        let registry = IntrinsicRegistry::new();

        assert_eq!(
            registry.lookup("sync_threads"),
            Some(&GpuIntrinsic::SyncThreads)
        );
        assert_eq!(registry.lookup("sqrt"), Some(&GpuIntrinsic::Sqrt));
        assert_eq!(registry.lookup("unknown_func"), None);
    }

    #[test]
    fn test_intrinsic_cuda_output() {
        assert_eq!(
            GpuIntrinsic::SyncThreads.to_cuda_string(),
            "__syncthreads()"
        );
        assert_eq!(GpuIntrinsic::AtomicAdd.to_cuda_string(), "atomicAdd");
        assert_eq!(GpuIntrinsic::Sqrt.to_cuda_string(), "sqrtf");
    }

    #[test]
    fn test_stencil_intrinsic_parsing() {
        assert_eq!(
            StencilIntrinsic::from_method_name("north"),
            Some(StencilIntrinsic::North)
        );
        assert_eq!(
            StencilIntrinsic::from_method_name("idx"),
            Some(StencilIntrinsic::Index)
        );
        assert_eq!(StencilIntrinsic::from_method_name("unknown"), None);
    }

    #[test]
    fn test_stencil_cuda_index() {
        let north = StencilIntrinsic::North;
        assert_eq!(
            north.to_cuda_index_2d("p", "buffer_width", "idx"),
            "p[idx - buffer_width]"
        );

        let east = StencilIntrinsic::East;
        assert_eq!(east.to_cuda_index_2d("p", "18", "idx"), "p[idx + 1]");
    }

    #[test]
    fn test_stencil_offset() {
        assert_eq!(StencilIntrinsic::North.get_offset_2d(), Some((-1, 0)));
        assert_eq!(StencilIntrinsic::East.get_offset_2d(), Some((0, 1)));
        assert_eq!(StencilIntrinsic::Index.get_offset_2d(), Some((0, 0)));
    }

    #[test]
    fn test_ring_kernel_intrinsic_lookup() {
        assert_eq!(
            RingKernelIntrinsic::from_name("is_active"),
            Some(RingKernelIntrinsic::IsActive)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("should_terminate"),
            Some(RingKernelIntrinsic::ShouldTerminate)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("hlc_tick"),
            Some(RingKernelIntrinsic::HlcTick)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("enqueue_response"),
            Some(RingKernelIntrinsic::EnqueueResponse)
        );
        assert_eq!(RingKernelIntrinsic::from_name("unknown"), None);
    }

    #[test]
    fn test_ring_kernel_intrinsic_cuda_output() {
        assert!(RingKernelIntrinsic::IsActive
            .to_cuda(&[])
            .contains("is_active"));
        assert!(RingKernelIntrinsic::ShouldTerminate
            .to_cuda(&[])
            .contains("should_terminate"));
        assert!(RingKernelIntrinsic::HlcTick
            .to_cuda(&[])
            .contains("hlc_logical"));
        assert!(RingKernelIntrinsic::InputQueueEmpty
            .to_cuda(&[])
            .contains("input_head"));
    }

    #[test]
    fn test_ring_kernel_queue_intrinsics() {
        let enqueue = RingKernelIntrinsic::EnqueueResponse;
        let cuda = enqueue.to_cuda(&["&response".to_string()]);
        assert!(cuda.contains("output_head"));
        assert!(cuda.contains("memcpy"));
    }

    #[test]
    fn test_k2k_intrinsics() {
        // Test k2k_send
        let send = RingKernelIntrinsic::K2kSend;
        let cuda = send.to_cuda(&["target_id".to_string(), "&msg".to_string()]);
        assert!(cuda.contains("k2k_send"));
        assert!(cuda.contains("k2k_routes"));
        assert!(cuda.contains("target_id"));

        // Test k2k_try_recv
        assert_eq!(
            RingKernelIntrinsic::K2kTryRecv.to_cuda(&[]),
            "k2k_try_recv(k2k_inbox)"
        );

        // Test k2k_has_message
        assert_eq!(
            RingKernelIntrinsic::K2kHasMessage.to_cuda(&[]),
            "k2k_has_message(k2k_inbox)"
        );

        // Test k2k_peek
        assert_eq!(
            RingKernelIntrinsic::K2kPeek.to_cuda(&[]),
            "k2k_peek(k2k_inbox)"
        );

        // Test k2k_pending_count
        assert_eq!(
            RingKernelIntrinsic::K2kPendingCount.to_cuda(&[]),
            "k2k_pending_count(k2k_inbox)"
        );
    }

    #[test]
    fn test_k2k_intrinsic_lookup() {
        assert_eq!(
            RingKernelIntrinsic::from_name("k2k_send"),
            Some(RingKernelIntrinsic::K2kSend)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("k2k_try_recv"),
            Some(RingKernelIntrinsic::K2kTryRecv)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("k2k_has_message"),
            Some(RingKernelIntrinsic::K2kHasMessage)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("k2k_peek"),
            Some(RingKernelIntrinsic::K2kPeek)
        );
        assert_eq!(
            RingKernelIntrinsic::from_name("k2k_pending_count"),
            Some(RingKernelIntrinsic::K2kPendingCount)
        );
    }

    #[test]
    fn test_intrinsic_requirements() {
        // K2K intrinsics require K2K
        assert!(RingKernelIntrinsic::K2kSend.requires_k2k());
        assert!(RingKernelIntrinsic::K2kTryRecv.requires_k2k());
        assert!(RingKernelIntrinsic::K2kPeek.requires_k2k());
        assert!(!RingKernelIntrinsic::HlcTick.requires_k2k());

        // HLC intrinsics require HLC
        assert!(RingKernelIntrinsic::HlcTick.requires_hlc());
        assert!(RingKernelIntrinsic::HlcNow.requires_hlc());
        assert!(!RingKernelIntrinsic::K2kSend.requires_hlc());

        // Control block intrinsics require control block
        assert!(RingKernelIntrinsic::IsActive.requires_control_block());
        assert!(RingKernelIntrinsic::EnqueueResponse.requires_control_block());
        assert!(!RingKernelIntrinsic::HlcTick.requires_control_block());
    }

    // === NEW INTRINSIC TESTS ===

    #[test]
    fn test_new_atomic_intrinsics() {
        let registry = IntrinsicRegistry::new();

        // Test bitwise atomics
        assert_eq!(
            registry.lookup("atomic_and"),
            Some(&GpuIntrinsic::AtomicAnd)
        );
        assert_eq!(registry.lookup("atomic_or"), Some(&GpuIntrinsic::AtomicOr));
        assert_eq!(
            registry.lookup("atomic_xor"),
            Some(&GpuIntrinsic::AtomicXor)
        );
        assert_eq!(
            registry.lookup("atomic_inc"),
            Some(&GpuIntrinsic::AtomicInc)
        );
        assert_eq!(
            registry.lookup("atomic_dec"),
            Some(&GpuIntrinsic::AtomicDec)
        );

        // Test CUDA output
        assert_eq!(GpuIntrinsic::AtomicAnd.to_cuda_string(), "atomicAnd");
        assert_eq!(GpuIntrinsic::AtomicOr.to_cuda_string(), "atomicOr");
        assert_eq!(GpuIntrinsic::AtomicXor.to_cuda_string(), "atomicXor");
        assert_eq!(GpuIntrinsic::AtomicInc.to_cuda_string(), "atomicInc");
        assert_eq!(GpuIntrinsic::AtomicDec.to_cuda_string(), "atomicDec");
    }

    #[test]
    fn test_trigonometric_intrinsics() {
        let registry = IntrinsicRegistry::new();

        // Test inverse trig
        assert_eq!(registry.lookup("asin"), Some(&GpuIntrinsic::Asin));
        assert_eq!(registry.lookup("acos"), Some(&GpuIntrinsic::Acos));
        assert_eq!(registry.lookup("atan"), Some(&GpuIntrinsic::Atan));
        assert_eq!(registry.lookup("atan2"), Some(&GpuIntrinsic::Atan2));

        // Test CUDA output
        assert_eq!(GpuIntrinsic::Asin.to_cuda_string(), "asinf");
        assert_eq!(GpuIntrinsic::Acos.to_cuda_string(), "acosf");
        assert_eq!(GpuIntrinsic::Atan.to_cuda_string(), "atanf");
        assert_eq!(GpuIntrinsic::Atan2.to_cuda_string(), "atan2f");
    }

    #[test]
    fn test_hyperbolic_intrinsics() {
        let registry = IntrinsicRegistry::new();

        // Test hyperbolic functions
        assert_eq!(registry.lookup("sinh"), Some(&GpuIntrinsic::Sinh));
        assert_eq!(registry.lookup("cosh"), Some(&GpuIntrinsic::Cosh));
        assert_eq!(registry.lookup("tanh"), Some(&GpuIntrinsic::Tanh));
        assert_eq!(registry.lookup("asinh"), Some(&GpuIntrinsic::Asinh));
        assert_eq!(registry.lookup("acosh"), Some(&GpuIntrinsic::Acosh));
        assert_eq!(registry.lookup("atanh"), Some(&GpuIntrinsic::Atanh));

        // Test CUDA output
        assert_eq!(GpuIntrinsic::Sinh.to_cuda_string(), "sinhf");
        assert_eq!(GpuIntrinsic::Cosh.to_cuda_string(), "coshf");
        assert_eq!(GpuIntrinsic::Tanh.to_cuda_string(), "tanhf");
    }

    #[test]
    fn test_exponential_logarithmic_intrinsics() {
        let registry = IntrinsicRegistry::new();

        // Test exp variants
        assert_eq!(registry.lookup("exp2"), Some(&GpuIntrinsic::Exp2));
        assert_eq!(registry.lookup("exp10"), Some(&GpuIntrinsic::Exp10));
        assert_eq!(registry.lookup("expm1"), Some(&GpuIntrinsic::Expm1));

        // Test log variants
        assert_eq!(registry.lookup("log2"), Some(&GpuIntrinsic::Log2));
        assert_eq!(registry.lookup("log10"), Some(&GpuIntrinsic::Log10));
        assert_eq!(registry.lookup("log1p"), Some(&GpuIntrinsic::Log1p));

        // Test CUDA output
        assert_eq!(GpuIntrinsic::Exp2.to_cuda_string(), "exp2f");
        assert_eq!(GpuIntrinsic::Log2.to_cuda_string(), "log2f");
        assert_eq!(GpuIntrinsic::Log10.to_cuda_string(), "log10f");
    }

    #[test]
    fn test_classification_intrinsics() {
        let registry = IntrinsicRegistry::new();

        // Test classification functions
        assert_eq!(registry.lookup("is_nan"), Some(&GpuIntrinsic::Isnan));
        assert_eq!(registry.lookup("isnan"), Some(&GpuIntrinsic::Isnan));
        assert_eq!(registry.lookup("is_infinite"), Some(&GpuIntrinsic::Isinf));
        assert_eq!(registry.lookup("is_finite"), Some(&GpuIntrinsic::Isfinite));
        assert_eq!(registry.lookup("is_normal"), Some(&GpuIntrinsic::Isnormal));
        assert_eq!(registry.lookup("signbit"), Some(&GpuIntrinsic::Signbit));

        // Test CUDA output
        assert_eq!(GpuIntrinsic::Isnan.to_cuda_string(), "isnan");
        assert_eq!(GpuIntrinsic::Isinf.to_cuda_string(), "isinf");
        assert_eq!(GpuIntrinsic::Isfinite.to_cuda_string(), "isfinite");
    }

    #[test]
    fn test_warp_reduce_intrinsics() {
        let registry = IntrinsicRegistry::new();

        // Test warp reduce operations
        assert_eq!(
            registry.lookup("warp_reduce_add"),
            Some(&GpuIntrinsic::WarpReduceAdd)
        );
        assert_eq!(
            registry.lookup("warp_reduce_min"),
            Some(&GpuIntrinsic::WarpReduceMin)
        );
        assert_eq!(
            registry.lookup("warp_reduce_max"),
            Some(&GpuIntrinsic::WarpReduceMax)
        );
        assert_eq!(
            registry.lookup("warp_reduce_and"),
            Some(&GpuIntrinsic::WarpReduceAnd)
        );
        assert_eq!(
            registry.lookup("warp_reduce_or"),
            Some(&GpuIntrinsic::WarpReduceOr)
        );
        assert_eq!(
            registry.lookup("warp_reduce_xor"),
            Some(&GpuIntrinsic::WarpReduceXor)
        );

        // Test CUDA output
        assert_eq!(
            GpuIntrinsic::WarpReduceAdd.to_cuda_string(),
            "__reduce_add_sync"
        );
        assert_eq!(
            GpuIntrinsic::WarpReduceMin.to_cuda_string(),
            "__reduce_min_sync"
        );
        assert_eq!(
            GpuIntrinsic::WarpReduceMax.to_cuda_string(),
            "__reduce_max_sync"
        );
    }

    #[test]
    fn test_warp_match_intrinsics() {
        let registry = IntrinsicRegistry::new();

        assert_eq!(
            registry.lookup("warp_match_any"),
            Some(&GpuIntrinsic::WarpMatchAny)
        );
        assert_eq!(
            registry.lookup("warp_match_all"),
            Some(&GpuIntrinsic::WarpMatchAll)
        );

        assert_eq!(
            GpuIntrinsic::WarpMatchAny.to_cuda_string(),
            "__match_any_sync"
        );
        assert_eq!(
            GpuIntrinsic::WarpMatchAll.to_cuda_string(),
            "__match_all_sync"
        );
    }

    #[test]
    fn test_bit_manipulation_intrinsics() {
        let registry = IntrinsicRegistry::new();

        // Test bit manipulation
        assert_eq!(registry.lookup("popc"), Some(&GpuIntrinsic::Popc));
        assert_eq!(registry.lookup("popcount"), Some(&GpuIntrinsic::Popc));
        assert_eq!(registry.lookup("count_ones"), Some(&GpuIntrinsic::Popc));
        assert_eq!(registry.lookup("clz"), Some(&GpuIntrinsic::Clz));
        assert_eq!(registry.lookup("leading_zeros"), Some(&GpuIntrinsic::Clz));
        assert_eq!(registry.lookup("ctz"), Some(&GpuIntrinsic::Ctz));
        assert_eq!(registry.lookup("ffs"), Some(&GpuIntrinsic::Ffs));
        assert_eq!(registry.lookup("brev"), Some(&GpuIntrinsic::Brev));
        assert_eq!(registry.lookup("reverse_bits"), Some(&GpuIntrinsic::Brev));

        // Test CUDA output
        assert_eq!(GpuIntrinsic::Popc.to_cuda_string(), "__popc");
        assert_eq!(GpuIntrinsic::Clz.to_cuda_string(), "__clz");
        assert_eq!(GpuIntrinsic::Ffs.to_cuda_string(), "__ffs");
        assert_eq!(GpuIntrinsic::Brev.to_cuda_string(), "__brev");
    }

    #[test]
    fn test_funnel_shift_intrinsics() {
        let registry = IntrinsicRegistry::new();

        assert_eq!(
            registry.lookup("funnel_shift_left"),
            Some(&GpuIntrinsic::FunnelShiftLeft)
        );
        assert_eq!(
            registry.lookup("funnel_shift_right"),
            Some(&GpuIntrinsic::FunnelShiftRight)
        );

        assert_eq!(
            GpuIntrinsic::FunnelShiftLeft.to_cuda_string(),
            "__funnelshift_l"
        );
        assert_eq!(
            GpuIntrinsic::FunnelShiftRight.to_cuda_string(),
            "__funnelshift_r"
        );
    }

    #[test]
    fn test_memory_intrinsics() {
        let registry = IntrinsicRegistry::new();

        assert_eq!(registry.lookup("ldg"), Some(&GpuIntrinsic::Ldg));
        assert_eq!(registry.lookup("load_global"), Some(&GpuIntrinsic::Ldg));
        assert_eq!(
            registry.lookup("prefetch_l1"),
            Some(&GpuIntrinsic::PrefetchL1)
        );
        assert_eq!(
            registry.lookup("prefetch_l2"),
            Some(&GpuIntrinsic::PrefetchL2)
        );

        assert_eq!(GpuIntrinsic::Ldg.to_cuda_string(), "__ldg");
        assert_eq!(GpuIntrinsic::PrefetchL1.to_cuda_string(), "__prefetch_l1");
        assert_eq!(GpuIntrinsic::PrefetchL2.to_cuda_string(), "__prefetch_l2");
    }

    #[test]
    fn test_clock_intrinsics() {
        let registry = IntrinsicRegistry::new();

        assert_eq!(registry.lookup("clock"), Some(&GpuIntrinsic::Clock));
        assert_eq!(registry.lookup("clock64"), Some(&GpuIntrinsic::Clock64));
        assert_eq!(registry.lookup("nanosleep"), Some(&GpuIntrinsic::Nanosleep));

        assert_eq!(GpuIntrinsic::Clock.to_cuda_string(), "clock()");
        assert_eq!(GpuIntrinsic::Clock64.to_cuda_string(), "clock64()");
        assert_eq!(GpuIntrinsic::Nanosleep.to_cuda_string(), "__nanosleep");
    }

    #[test]
    fn test_special_function_intrinsics() {
        let registry = IntrinsicRegistry::new();

        assert_eq!(registry.lookup("rcp"), Some(&GpuIntrinsic::Rcp));
        assert_eq!(registry.lookup("recip"), Some(&GpuIntrinsic::Rcp));
        assert_eq!(registry.lookup("saturate"), Some(&GpuIntrinsic::Saturate));
        assert_eq!(registry.lookup("clamp_01"), Some(&GpuIntrinsic::Saturate));

        assert_eq!(GpuIntrinsic::Rcp.to_cuda_string(), "__frcp_rn");
        assert_eq!(GpuIntrinsic::Saturate.to_cuda_string(), "__saturatef");
    }

    #[test]
    fn test_intrinsic_categories() {
        // Test category assignment
        assert_eq!(GpuIntrinsic::SyncThreads.category(), "synchronization");
        assert_eq!(GpuIntrinsic::AtomicAdd.category(), "atomic");
        assert_eq!(GpuIntrinsic::Sqrt.category(), "math");
        assert_eq!(GpuIntrinsic::Sin.category(), "trigonometric");
        assert_eq!(GpuIntrinsic::Sinh.category(), "hyperbolic");
        assert_eq!(GpuIntrinsic::Exp.category(), "exponential");
        assert_eq!(GpuIntrinsic::Isnan.category(), "classification");
        assert_eq!(GpuIntrinsic::WarpShfl.category(), "warp");
        assert_eq!(GpuIntrinsic::Popc.category(), "bit");
        assert_eq!(GpuIntrinsic::Ldg.category(), "memory");
        assert_eq!(GpuIntrinsic::Rcp.category(), "special");
        assert_eq!(GpuIntrinsic::ThreadIdxX.category(), "index");
        assert_eq!(GpuIntrinsic::Clock.category(), "timing");
    }

    #[test]
    fn test_intrinsic_flags() {
        // Test is_value_intrinsic
        assert!(GpuIntrinsic::ThreadIdxX.is_value_intrinsic());
        assert!(GpuIntrinsic::BlockDimX.is_value_intrinsic());
        assert!(GpuIntrinsic::WarpSize.is_value_intrinsic());
        assert!(!GpuIntrinsic::Sin.is_value_intrinsic());
        assert!(!GpuIntrinsic::AtomicAdd.is_value_intrinsic());

        // Test is_zero_arg_function
        assert!(GpuIntrinsic::SyncThreads.is_zero_arg_function());
        assert!(GpuIntrinsic::ThreadFence.is_zero_arg_function());
        assert!(GpuIntrinsic::WarpActiveMask.is_zero_arg_function());
        assert!(GpuIntrinsic::Clock.is_zero_arg_function());
        assert!(!GpuIntrinsic::Sin.is_zero_arg_function());

        // Test requires_mask
        assert!(GpuIntrinsic::WarpShfl.requires_mask());
        assert!(GpuIntrinsic::WarpBallot.requires_mask());
        assert!(GpuIntrinsic::WarpReduceAdd.requires_mask());
        assert!(!GpuIntrinsic::Sin.requires_mask());
        assert!(!GpuIntrinsic::AtomicAdd.requires_mask());
    }

    #[test]
    fn test_3d_stencil_intrinsics() {
        assert_eq!(
            StencilIntrinsic::from_method_name("up"),
            Some(StencilIntrinsic::Up)
        );
        assert_eq!(
            StencilIntrinsic::from_method_name("down"),
            Some(StencilIntrinsic::Down)
        );

        // Test 3D only flag
        assert!(StencilIntrinsic::Up.is_3d_only());
        assert!(StencilIntrinsic::Down.is_3d_only());
        assert!(!StencilIntrinsic::North.is_3d_only());
        assert!(!StencilIntrinsic::East.is_3d_only());
        assert!(!StencilIntrinsic::Index.is_3d_only());

        // Test 3D offsets
        assert_eq!(StencilIntrinsic::Up.get_offset_3d(), Some((-1, 0, 0)));
        assert_eq!(StencilIntrinsic::Down.get_offset_3d(), Some((1, 0, 0)));
        assert_eq!(StencilIntrinsic::North.get_offset_3d(), Some((0, -1, 0)));
        assert_eq!(StencilIntrinsic::South.get_offset_3d(), Some((0, 1, 0)));
        assert_eq!(StencilIntrinsic::East.get_offset_3d(), Some((0, 0, 1)));
        assert_eq!(StencilIntrinsic::West.get_offset_3d(), Some((0, 0, -1)));

        // Test 3D index generation
        let up = StencilIntrinsic::Up;
        assert_eq!(up.to_cuda_index_3d("p", "18", "324", "idx"), "p[idx - 324]");

        let down = StencilIntrinsic::Down;
        assert_eq!(
            down.to_cuda_index_3d("p", "18", "324", "idx"),
            "p[idx + 324]"
        );
    }

    #[test]
    fn test_sync_intrinsics() {
        let registry = IntrinsicRegistry::new();

        assert_eq!(
            registry.lookup("sync_threads_count"),
            Some(&GpuIntrinsic::SyncThreadsCount)
        );
        assert_eq!(
            registry.lookup("sync_threads_and"),
            Some(&GpuIntrinsic::SyncThreadsAnd)
        );
        assert_eq!(
            registry.lookup("sync_threads_or"),
            Some(&GpuIntrinsic::SyncThreadsOr)
        );

        assert_eq!(
            GpuIntrinsic::SyncThreadsCount.to_cuda_string(),
            "__syncthreads_count"
        );
        assert_eq!(
            GpuIntrinsic::SyncThreadsAnd.to_cuda_string(),
            "__syncthreads_and"
        );
        assert_eq!(
            GpuIntrinsic::SyncThreadsOr.to_cuda_string(),
            "__syncthreads_or"
        );
    }

    #[test]
    fn test_math_extras() {
        let registry = IntrinsicRegistry::new();

        assert_eq!(registry.lookup("trunc"), Some(&GpuIntrinsic::Trunc));
        assert_eq!(registry.lookup("cbrt"), Some(&GpuIntrinsic::Cbrt));
        assert_eq!(registry.lookup("hypot"), Some(&GpuIntrinsic::Hypot));
        assert_eq!(registry.lookup("copysign"), Some(&GpuIntrinsic::Copysign));
        assert_eq!(registry.lookup("fmod"), Some(&GpuIntrinsic::Fmod));

        assert_eq!(GpuIntrinsic::Trunc.to_cuda_string(), "truncf");
        assert_eq!(GpuIntrinsic::Cbrt.to_cuda_string(), "cbrtf");
        assert_eq!(GpuIntrinsic::Hypot.to_cuda_string(), "hypotf");
    }
}
