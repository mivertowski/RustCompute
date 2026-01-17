//! Global Reduction Primitives
//!
//! This module provides traits and types for GPU-accelerated reduction operations.
//! Reductions aggregate values across all GPU threads using operations like sum,
//! min, max, etc.
//!
//! # Use Cases
//!
//! - **PageRank**: Sum dangling node contributions across all nodes
//! - **Graph algorithms**: Compute convergence metrics, global norms
//! - **Scientific computing**: Vector norms, dot products, energy calculations
//!
//! # Architecture
//!
//! Reductions use a hierarchical approach for efficiency:
//! 1. **Warp-level**: Use shuffle instructions for fast intra-warp reduction
//! 2. **Block-level**: Tree reduction in shared memory with `__syncthreads()`
//! 3. **Grid-level**: Atomic accumulation from block leaders, then broadcast
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::reduction::{ReductionOp, GlobalReduction};
//!
//! // In kernel code (DSL):
//! let my_contrib = if out_degree[idx] == 0 { rank } else { 0.0 };
//! let dangling_sum = reduce_and_broadcast(my_contrib, &accumulator);
//! let new_rank = base + damping * (incoming + dangling_sum / n);
//! ```

use std::fmt::Debug;

/// Reduction operation types.
///
/// Each operation has an identity value that serves as the neutral element:
/// - Sum: 0 (a + 0 = a)
/// - Min: MAX (min(a, MAX) = a)
/// - Max: MIN (max(a, MIN) = a)
/// - And: all bits set (-1 for signed, MAX for unsigned)
/// - Or: 0 (a | 0 = a)
/// - Xor: 0 (a ^ 0 = a)
/// - Product: 1 (a * 1 = a)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    /// Sum of all values.
    Sum,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
    /// Product of all values.
    Product,
}

impl ReductionOp {
    /// Get the CUDA atomic function name for this operation.
    #[must_use]
    pub fn atomic_name(&self) -> &'static str {
        match self {
            ReductionOp::Sum => "atomicAdd",
            ReductionOp::Min => "atomicMin",
            ReductionOp::Max => "atomicMax",
            ReductionOp::And => "atomicAnd",
            ReductionOp::Or => "atomicOr",
            ReductionOp::Xor => "atomicXor",
            ReductionOp::Product => "atomicMul", // Requires custom implementation
        }
    }

    /// Get the WGSL atomic function name for this operation.
    #[must_use]
    pub fn wgsl_atomic_name(&self) -> Option<&'static str> {
        match self {
            ReductionOp::Sum => Some("atomicAdd"),
            ReductionOp::Min => Some("atomicMin"),
            ReductionOp::Max => Some("atomicMax"),
            ReductionOp::And => Some("atomicAnd"),
            ReductionOp::Or => Some("atomicOr"),
            ReductionOp::Xor => Some("atomicXor"),
            ReductionOp::Product => None, // Not supported in WGSL
        }
    }

    /// Get the C operator for this reduction (for code generation).
    #[must_use]
    pub fn c_operator(&self) -> &'static str {
        match self {
            ReductionOp::Sum => "+",
            ReductionOp::Min => "min",
            ReductionOp::Max => "max",
            ReductionOp::And => "&",
            ReductionOp::Or => "|",
            ReductionOp::Xor => "^",
            ReductionOp::Product => "*",
        }
    }

    /// Check if this operation is commutative.
    #[must_use]
    pub const fn is_commutative(&self) -> bool {
        true // All supported operations are commutative
    }

    /// Check if this operation is associative.
    #[must_use]
    pub const fn is_associative(&self) -> bool {
        // Note: floating-point sum/product are not strictly associative
        // due to rounding, but we treat them as such for parallel reduction
        true
    }
}

impl std::fmt::Display for ReductionOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReductionOp::Sum => write!(f, "sum"),
            ReductionOp::Min => write!(f, "min"),
            ReductionOp::Max => write!(f, "max"),
            ReductionOp::And => write!(f, "and"),
            ReductionOp::Or => write!(f, "or"),
            ReductionOp::Xor => write!(f, "xor"),
            ReductionOp::Product => write!(f, "product"),
        }
    }
}

/// Trait for scalar types that support reduction operations.
///
/// Implementors must provide identity values for each reduction operation.
/// The identity value is the neutral element such that `op(x, identity) = x`.
pub trait ReductionScalar: Copy + Send + Sync + Debug + Default + 'static {
    /// Get the identity value for the given reduction operation.
    fn identity(op: ReductionOp) -> Self;

    /// Combine two values according to the reduction operation.
    fn combine(a: Self, b: Self, op: ReductionOp) -> Self;

    /// Size in bytes.
    fn size_bytes() -> usize {
        std::mem::size_of::<Self>()
    }

    /// CUDA type name for code generation.
    fn cuda_type() -> &'static str;

    /// WGSL type name for code generation.
    fn wgsl_type() -> &'static str;
}

impl ReductionScalar for f32 {
    fn identity(op: ReductionOp) -> Self {
        match op {
            ReductionOp::Sum | ReductionOp::Or | ReductionOp::Xor => 0.0,
            ReductionOp::Min => f32::INFINITY,
            ReductionOp::Max => f32::NEG_INFINITY,
            ReductionOp::Product | ReductionOp::And => 1.0,
        }
    }

    fn combine(a: Self, b: Self, op: ReductionOp) -> Self {
        match op {
            ReductionOp::Sum => a + b,
            ReductionOp::Min => a.min(b),
            ReductionOp::Max => a.max(b),
            ReductionOp::Product => a * b,
            // Bitwise ops on floats: use bit representation
            ReductionOp::And => f32::from_bits(a.to_bits() & b.to_bits()),
            ReductionOp::Or => f32::from_bits(a.to_bits() | b.to_bits()),
            ReductionOp::Xor => f32::from_bits(a.to_bits() ^ b.to_bits()),
        }
    }

    fn cuda_type() -> &'static str {
        "float"
    }

    fn wgsl_type() -> &'static str {
        "f32"
    }
}

impl ReductionScalar for f64 {
    fn identity(op: ReductionOp) -> Self {
        match op {
            ReductionOp::Sum | ReductionOp::Or | ReductionOp::Xor => 0.0,
            ReductionOp::Min => f64::INFINITY,
            ReductionOp::Max => f64::NEG_INFINITY,
            ReductionOp::Product | ReductionOp::And => 1.0,
        }
    }

    fn combine(a: Self, b: Self, op: ReductionOp) -> Self {
        match op {
            ReductionOp::Sum => a + b,
            ReductionOp::Min => a.min(b),
            ReductionOp::Max => a.max(b),
            ReductionOp::Product => a * b,
            ReductionOp::And => f64::from_bits(a.to_bits() & b.to_bits()),
            ReductionOp::Or => f64::from_bits(a.to_bits() | b.to_bits()),
            ReductionOp::Xor => f64::from_bits(a.to_bits() ^ b.to_bits()),
        }
    }

    fn cuda_type() -> &'static str {
        "double"
    }

    fn wgsl_type() -> &'static str {
        "f32" // WGSL doesn't have f64, fallback to f32
    }
}

impl ReductionScalar for i32 {
    fn identity(op: ReductionOp) -> Self {
        match op {
            ReductionOp::Sum | ReductionOp::Or | ReductionOp::Xor => 0,
            ReductionOp::Min => i32::MAX,
            ReductionOp::Max => i32::MIN,
            ReductionOp::Product => 1,
            ReductionOp::And => -1, // All bits set
        }
    }

    fn combine(a: Self, b: Self, op: ReductionOp) -> Self {
        match op {
            ReductionOp::Sum => a.wrapping_add(b),
            ReductionOp::Min => a.min(b),
            ReductionOp::Max => a.max(b),
            ReductionOp::Product => a.wrapping_mul(b),
            ReductionOp::And => a & b,
            ReductionOp::Or => a | b,
            ReductionOp::Xor => a ^ b,
        }
    }

    fn cuda_type() -> &'static str {
        "int"
    }

    fn wgsl_type() -> &'static str {
        "i32"
    }
}

impl ReductionScalar for i64 {
    fn identity(op: ReductionOp) -> Self {
        match op {
            ReductionOp::Sum | ReductionOp::Or | ReductionOp::Xor => 0,
            ReductionOp::Min => i64::MAX,
            ReductionOp::Max => i64::MIN,
            ReductionOp::Product => 1,
            ReductionOp::And => -1,
        }
    }

    fn combine(a: Self, b: Self, op: ReductionOp) -> Self {
        match op {
            ReductionOp::Sum => a.wrapping_add(b),
            ReductionOp::Min => a.min(b),
            ReductionOp::Max => a.max(b),
            ReductionOp::Product => a.wrapping_mul(b),
            ReductionOp::And => a & b,
            ReductionOp::Or => a | b,
            ReductionOp::Xor => a ^ b,
        }
    }

    fn cuda_type() -> &'static str {
        "long long"
    }

    fn wgsl_type() -> &'static str {
        "i32" // WGSL doesn't have i64
    }
}

impl ReductionScalar for u32 {
    fn identity(op: ReductionOp) -> Self {
        match op {
            ReductionOp::Sum | ReductionOp::Or | ReductionOp::Xor => 0,
            ReductionOp::Min | ReductionOp::And => u32::MAX,
            ReductionOp::Max => 0,
            ReductionOp::Product => 1,
        }
    }

    fn combine(a: Self, b: Self, op: ReductionOp) -> Self {
        match op {
            ReductionOp::Sum => a.wrapping_add(b),
            ReductionOp::Min => a.min(b),
            ReductionOp::Max => a.max(b),
            ReductionOp::Product => a.wrapping_mul(b),
            ReductionOp::And => a & b,
            ReductionOp::Or => a | b,
            ReductionOp::Xor => a ^ b,
        }
    }

    fn cuda_type() -> &'static str {
        "unsigned int"
    }

    fn wgsl_type() -> &'static str {
        "u32"
    }
}

impl ReductionScalar for u64 {
    fn identity(op: ReductionOp) -> Self {
        match op {
            ReductionOp::Sum | ReductionOp::Or | ReductionOp::Xor => 0,
            ReductionOp::Min | ReductionOp::And => u64::MAX,
            ReductionOp::Max => 0,
            ReductionOp::Product => 1,
        }
    }

    fn combine(a: Self, b: Self, op: ReductionOp) -> Self {
        match op {
            ReductionOp::Sum => a.wrapping_add(b),
            ReductionOp::Min => a.min(b),
            ReductionOp::Max => a.max(b),
            ReductionOp::Product => a.wrapping_mul(b),
            ReductionOp::And => a & b,
            ReductionOp::Or => a | b,
            ReductionOp::Xor => a ^ b,
        }
    }

    fn cuda_type() -> &'static str {
        "unsigned long long"
    }

    fn wgsl_type() -> &'static str {
        "u32" // WGSL doesn't have u64
    }
}

/// Configuration for reduction operations.
#[derive(Debug, Clone)]
pub struct ReductionConfig {
    /// Number of reduction slots (for parallel accumulation).
    ///
    /// Multiple slots reduce atomic contention by spreading updates
    /// across several memory locations. The final result is computed
    /// by combining all slots on the host.
    pub num_slots: usize,

    /// Use cooperative groups for grid-wide synchronization.
    ///
    /// Requires compute capability 6.0+ (Pascal or newer).
    /// When disabled, falls back to software barriers or multi-launch.
    pub use_cooperative: bool,

    /// Use software barrier when cooperative groups unavailable.
    ///
    /// Software barriers use atomic counters in global memory.
    /// This works on all devices but has higher latency.
    pub use_software_barrier: bool,

    /// Shared memory size per block for reduction (bytes).
    ///
    /// Should be at least `block_size * sizeof(T)` for full reduction.
    /// Default: 0 (auto-calculate based on block size).
    pub shared_mem_bytes: usize,
}

impl Default for ReductionConfig {
    fn default() -> Self {
        Self {
            num_slots: 1,
            use_cooperative: true,
            use_software_barrier: true,
            shared_mem_bytes: 0,
        }
    }
}

impl ReductionConfig {
    /// Create a new reduction config with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of accumulation slots.
    #[must_use]
    pub fn with_slots(mut self, num_slots: usize) -> Self {
        self.num_slots = num_slots.max(1);
        self
    }

    /// Enable or disable cooperative groups.
    #[must_use]
    pub fn with_cooperative(mut self, enabled: bool) -> Self {
        self.use_cooperative = enabled;
        self
    }

    /// Enable or disable software barrier fallback.
    #[must_use]
    pub fn with_software_barrier(mut self, enabled: bool) -> Self {
        self.use_software_barrier = enabled;
        self
    }

    /// Set explicit shared memory size.
    #[must_use]
    pub fn with_shared_mem(mut self, bytes: usize) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }
}

/// Handle to a reduction buffer for streaming operations.
///
/// This trait abstracts over backend-specific reduction buffer implementations,
/// allowing the same code to work with CUDA, WebGPU, or CPU backends.
pub trait ReductionHandle<T: ReductionScalar>: Send + Sync {
    /// Get device pointer for kernel parameter passing.
    fn device_ptr(&self) -> u64;

    /// Reset buffer to identity value.
    fn reset(&self) -> crate::error::Result<()>;

    /// Read the current reduction result from slot 0.
    fn read(&self) -> crate::error::Result<T>;

    /// Read and combine all slots into a single result.
    fn read_combined(&self) -> crate::error::Result<T>;

    /// Synchronize device and read result.
    ///
    /// Ensures all GPU operations complete before reading.
    fn sync_and_read(&self) -> crate::error::Result<T>;

    /// Get the reduction operation type.
    fn op(&self) -> ReductionOp;

    /// Get number of slots.
    fn num_slots(&self) -> usize;
}

/// Trait for GPU runtimes that support global reduction operations.
///
/// Implemented by backend-specific runtimes (CUDA, WebGPU, etc.) to provide
/// efficient reduction primitives.
pub trait GlobalReduction: Send + Sync {
    /// Create a reduction buffer for the specified type and operation.
    fn create_reduction_buffer<T: ReductionScalar>(
        &self,
        op: ReductionOp,
        config: &ReductionConfig,
    ) -> crate::error::Result<Box<dyn ReductionHandle<T>>>;

    /// Check if cooperative groups are supported.
    fn supports_cooperative(&self) -> bool;

    /// Check if grid-wide reduction is available.
    fn supports_grid_reduction(&self) -> bool;

    /// Get minimum compute capability for cooperative groups.
    ///
    /// Returns (major, minor) version tuple, or None if not applicable.
    fn cooperative_compute_capability(&self) -> Option<(u32, u32)> {
        Some((6, 0)) // Default: Pascal or newer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduction_op_display() {
        assert_eq!(format!("{}", ReductionOp::Sum), "sum");
        assert_eq!(format!("{}", ReductionOp::Min), "min");
        assert_eq!(format!("{}", ReductionOp::Max), "max");
    }

    #[test]
    fn test_f32_identity() {
        assert_eq!(f32::identity(ReductionOp::Sum), 0.0);
        assert_eq!(f32::identity(ReductionOp::Min), f32::INFINITY);
        assert_eq!(f32::identity(ReductionOp::Max), f32::NEG_INFINITY);
        assert_eq!(f32::identity(ReductionOp::Product), 1.0);
    }

    #[test]
    fn test_f32_combine() {
        assert_eq!(f32::combine(2.0, 3.0, ReductionOp::Sum), 5.0);
        assert_eq!(f32::combine(2.0, 3.0, ReductionOp::Min), 2.0);
        assert_eq!(f32::combine(2.0, 3.0, ReductionOp::Max), 3.0);
        assert_eq!(f32::combine(2.0, 3.0, ReductionOp::Product), 6.0);
    }

    #[test]
    fn test_i32_identity() {
        assert_eq!(i32::identity(ReductionOp::Sum), 0);
        assert_eq!(i32::identity(ReductionOp::Min), i32::MAX);
        assert_eq!(i32::identity(ReductionOp::Max), i32::MIN);
        assert_eq!(i32::identity(ReductionOp::And), -1);
        assert_eq!(i32::identity(ReductionOp::Or), 0);
    }

    #[test]
    fn test_u32_combine() {
        assert_eq!(u32::combine(5, 3, ReductionOp::Sum), 8);
        assert_eq!(u32::combine(5, 3, ReductionOp::Min), 3);
        assert_eq!(u32::combine(5, 3, ReductionOp::Max), 5);
        assert_eq!(u32::combine(0b1100, 0b1010, ReductionOp::And), 0b1000);
        assert_eq!(u32::combine(0b1100, 0b1010, ReductionOp::Or), 0b1110);
        assert_eq!(u32::combine(0b1100, 0b1010, ReductionOp::Xor), 0b0110);
    }

    #[test]
    fn test_reduction_config_builder() {
        let config = ReductionConfig::new()
            .with_slots(4)
            .with_cooperative(false)
            .with_shared_mem(4096);

        assert_eq!(config.num_slots, 4);
        assert!(!config.use_cooperative);
        assert_eq!(config.shared_mem_bytes, 4096);
    }

    #[test]
    fn test_cuda_type_names() {
        assert_eq!(f32::cuda_type(), "float");
        assert_eq!(f64::cuda_type(), "double");
        assert_eq!(i32::cuda_type(), "int");
        assert_eq!(i64::cuda_type(), "long long");
        assert_eq!(u32::cuda_type(), "unsigned int");
        assert_eq!(u64::cuda_type(), "unsigned long long");
    }

    #[test]
    fn test_atomic_names() {
        assert_eq!(ReductionOp::Sum.atomic_name(), "atomicAdd");
        assert_eq!(ReductionOp::Min.atomic_name(), "atomicMin");
        assert_eq!(ReductionOp::Max.atomic_name(), "atomicMax");
    }
}
