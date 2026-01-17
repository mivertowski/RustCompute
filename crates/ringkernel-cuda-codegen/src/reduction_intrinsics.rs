//! Reduction Intrinsics for CUDA Code Generation
//!
//! This module provides DSL intrinsics for efficient parallel reductions that
//! transpile to optimized CUDA code. Reductions aggregate values across threads
//! using operations like sum, min, max, etc.
//!
//! # Reduction Hierarchy
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Grid-Level Reduction                        │
//! │  ┌───────────────────────────────────────────────────────────┐  │
//! │  │                    Block 0                                │  │
//! │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │  │
//! │  │  │ Warp 0  │ │ Warp 1  │ │ Warp 2  │ │ Warp N  │         │  │
//! │  │  │ shuffle │ │ shuffle │ │ shuffle │ │ shuffle │         │  │
//! │  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘         │  │
//! │  │       └──────────────┴──────────┴──────────┘             │  │
//! │  │                      │ shared memory                     │  │
//! │  │                      ▼                                   │  │
//! │  │              block_reduce_sum()                          │  │
//! │  │                      │                                   │  │
//! │  └──────────────────────┼───────────────────────────────────┘  │
//! │                         ▼ atomicAdd                            │
//! │  ┌─────────────────────────────────────────────────────────┐  │
//! │  │              Global Accumulator (mapped memory)          │  │
//! │  └─────────────────────────────────────────────────────────┘  │
//! │                         │ grid.sync() / barrier                │
//! │                         ▼                                      │
//! │              All threads read final result                     │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! // Rust DSL
//! fn pagerank_phase1(ranks: &[f64], out_degree: &[u32], dangling_sum: &mut f64, n: u32) {
//!     let idx = global_thread_idx();
//!     if idx >= n { return; }
//!
//!     let contrib = if out_degree[idx] == 0 { ranks[idx] } else { 0.0 };
//!     reduce_and_broadcast(contrib, dangling_sum);  // All threads get sum
//! }
//! ```

use std::fmt;

/// Reduction operation types for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ReductionOp {
    /// Sum: `a + b`
    #[default]
    Sum,
    /// Minimum: `min(a, b)`
    Min,
    /// Maximum: `max(a, b)`
    Max,
    /// Bitwise AND: `a & b`
    And,
    /// Bitwise OR: `a | b`
    Or,
    /// Bitwise XOR: `a ^ b`
    Xor,
    /// Product: `a * b`
    Product,
}

impl ReductionOp {
    /// Get the binary operator for this reduction.
    pub fn operator(&self) -> &'static str {
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

    /// Get the CUDA atomic function name.
    pub fn atomic_fn(&self) -> &'static str {
        match self {
            ReductionOp::Sum => "atomicAdd",
            ReductionOp::Min => "atomicMin",
            ReductionOp::Max => "atomicMax",
            ReductionOp::And => "atomicAnd",
            ReductionOp::Or => "atomicOr",
            ReductionOp::Xor => "atomicXor",
            ReductionOp::Product => "atomicMul", // Custom implementation
        }
    }

    /// Get the identity value expression for this operation.
    pub fn identity(&self, ty: &str) -> String {
        match (self, ty) {
            (ReductionOp::Sum, _) => "0".to_string(),
            (ReductionOp::Min, "float") => "INFINITY".to_string(),
            (ReductionOp::Min, "double") => "HUGE_VAL".to_string(),
            (ReductionOp::Min, "int") => "INT_MAX".to_string(),
            (ReductionOp::Min, "long long") => "LLONG_MAX".to_string(),
            (ReductionOp::Min, "unsigned int") => "UINT_MAX".to_string(),
            (ReductionOp::Min, "unsigned long long") => "ULLONG_MAX".to_string(),
            (ReductionOp::Max, "float") => "-INFINITY".to_string(),
            (ReductionOp::Max, "double") => "-HUGE_VAL".to_string(),
            (ReductionOp::Max, "int") => "INT_MIN".to_string(),
            (ReductionOp::Max, "long long") => "LLONG_MIN".to_string(),
            (ReductionOp::Max, _) => "0".to_string(),
            (ReductionOp::And, _) => "~0".to_string(),
            (ReductionOp::Or, _) => "0".to_string(),
            (ReductionOp::Xor, _) => "0".to_string(),
            (ReductionOp::Product, _) => "1".to_string(),
            _ => "0".to_string(),
        }
    }

    /// Get the combination expression for two values.
    pub fn combine_expr(&self, a: &str, b: &str) -> String {
        match self {
            ReductionOp::Sum => format!("{} + {}", a, b),
            ReductionOp::Min => format!("min({}, {})", a, b),
            ReductionOp::Max => format!("max({}, {})", a, b),
            ReductionOp::And => format!("{} & {}", a, b),
            ReductionOp::Or => format!("{} | {}", a, b),
            ReductionOp::Xor => format!("{} ^ {}", a, b),
            ReductionOp::Product => format!("{} * {}", a, b),
        }
    }
}

impl fmt::Display for ReductionOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

/// Configuration for reduction code generation.
#[derive(Debug, Clone)]
pub struct ReductionCodegenConfig {
    /// Thread block size (must be power of 2).
    pub block_size: u32,
    /// Value type (CUDA type name).
    pub value_type: String,
    /// Reduction operation.
    pub op: ReductionOp,
    /// Use cooperative groups for grid-wide sync.
    pub use_cooperative: bool,
    /// Generate helper function definitions.
    pub generate_helpers: bool,
}

impl Default for ReductionCodegenConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            value_type: "float".to_string(),
            op: ReductionOp::Sum,
            use_cooperative: true,
            generate_helpers: true,
        }
    }
}

impl ReductionCodegenConfig {
    /// Create a new config with the given block size.
    pub fn new(block_size: u32) -> Self {
        Self {
            block_size,
            ..Default::default()
        }
    }

    /// Set the value type.
    pub fn with_type(mut self, ty: &str) -> Self {
        self.value_type = ty.to_string();
        self
    }

    /// Set the reduction operation.
    pub fn with_op(mut self, op: ReductionOp) -> Self {
        self.op = op;
        self
    }

    /// Enable or disable cooperative groups.
    pub fn with_cooperative(mut self, enabled: bool) -> Self {
        self.use_cooperative = enabled;
        self
    }
}

/// Reduction intrinsic types for the DSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionIntrinsic {
    /// Block-level sum reduction: `block_reduce_sum(value, shared_mem)`
    BlockReduceSum,
    /// Block-level min reduction: `block_reduce_min(value, shared_mem)`
    BlockReduceMin,
    /// Block-level max reduction: `block_reduce_max(value, shared_mem)`
    BlockReduceMax,
    /// Block-level AND reduction: `block_reduce_and(value, shared_mem)`
    BlockReduceAnd,
    /// Block-level OR reduction: `block_reduce_or(value, shared_mem)`
    BlockReduceOr,

    /// Grid-level sum with atomic accumulation: `grid_reduce_sum(value, shared, accumulator)`
    GridReduceSum,
    /// Grid-level min with atomic accumulation: `grid_reduce_min(value, shared, accumulator)`
    GridReduceMin,
    /// Grid-level max with atomic accumulation: `grid_reduce_max(value, shared, accumulator)`
    GridReduceMax,

    /// Atomic accumulate to buffer: `atomic_accumulate(accumulator, value)`
    AtomicAccumulate,
    /// Atomic accumulate with fetch: `atomic_accumulate_fetch(accumulator, value)`
    AtomicAccumulateFetch,

    /// Read from reduction buffer (broadcast): `broadcast_read(buffer)`
    BroadcastRead,

    /// Full reduce-and-broadcast: `reduce_and_broadcast(value, shared, accumulator)`
    /// Returns the global result to all threads.
    ReduceAndBroadcast,
}

impl ReductionIntrinsic {
    /// Parse intrinsic from DSL function name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "block_reduce_sum" => Some(ReductionIntrinsic::BlockReduceSum),
            "block_reduce_min" => Some(ReductionIntrinsic::BlockReduceMin),
            "block_reduce_max" => Some(ReductionIntrinsic::BlockReduceMax),
            "block_reduce_and" => Some(ReductionIntrinsic::BlockReduceAnd),
            "block_reduce_or" => Some(ReductionIntrinsic::BlockReduceOr),
            "grid_reduce_sum" => Some(ReductionIntrinsic::GridReduceSum),
            "grid_reduce_min" => Some(ReductionIntrinsic::GridReduceMin),
            "grid_reduce_max" => Some(ReductionIntrinsic::GridReduceMax),
            "atomic_accumulate" => Some(ReductionIntrinsic::AtomicAccumulate),
            "atomic_accumulate_fetch" => Some(ReductionIntrinsic::AtomicAccumulateFetch),
            "broadcast_read" => Some(ReductionIntrinsic::BroadcastRead),
            "reduce_and_broadcast" => Some(ReductionIntrinsic::ReduceAndBroadcast),
            _ => None,
        }
    }

    /// Get the reduction operation for this intrinsic.
    pub fn op(&self) -> Option<ReductionOp> {
        match self {
            ReductionIntrinsic::BlockReduceSum | ReductionIntrinsic::GridReduceSum => {
                Some(ReductionOp::Sum)
            }
            ReductionIntrinsic::BlockReduceMin | ReductionIntrinsic::GridReduceMin => {
                Some(ReductionOp::Min)
            }
            ReductionIntrinsic::BlockReduceMax | ReductionIntrinsic::GridReduceMax => {
                Some(ReductionOp::Max)
            }
            ReductionIntrinsic::BlockReduceAnd => Some(ReductionOp::And),
            ReductionIntrinsic::BlockReduceOr => Some(ReductionOp::Or),
            _ => None,
        }
    }
}

// =============================================================================
// Code Generation Functions
// =============================================================================

/// Generate CUDA helper functions for reduction operations.
///
/// This should be included at the top of kernels that use reduction intrinsics.
pub fn generate_reduction_helpers(config: &ReductionCodegenConfig) -> String {
    let mut code = String::new();

    // Includes
    if config.use_cooperative {
        code.push_str("#include <cooperative_groups.h>\n");
        code.push_str("namespace cg = cooperative_groups;\n\n");
    }
    code.push_str("#include <limits.h>\n");
    code.push_str("#include <math.h>\n\n");

    // Block-level reduction function
    code.push_str(&generate_block_reduce_fn(
        &config.value_type,
        config.block_size,
        &config.op,
    ));

    // Grid-level reduction function
    code.push_str(&generate_grid_reduce_fn(
        &config.value_type,
        config.block_size,
        &config.op,
    ));

    // Reduce-and-broadcast function
    code.push_str(&generate_reduce_and_broadcast_fn(
        &config.value_type,
        config.block_size,
        &config.op,
        config.use_cooperative,
    ));

    // Software barrier (if not using cooperative groups)
    if !config.use_cooperative {
        code.push_str(&generate_software_barrier());
    }

    code
}

/// Generate a block-level reduction function.
fn generate_block_reduce_fn(ty: &str, block_size: u32, op: &ReductionOp) -> String {
    let combine = op.combine_expr("shared[tid]", "shared[tid + s]");

    format!(
        r#"
// Block-level {op} reduction using shared memory
__device__ {ty} __block_reduce_{op}({ty} val, {ty}* shared) {{
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    // Tree reduction
    #pragma unroll
    for (int s = {block_size} / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared[tid] = {combine};
        }}
        __syncthreads();
    }}

    return shared[0];
}}

"#,
        ty = ty,
        op = op,
        block_size = block_size,
        combine = combine
    )
}

/// Generate a grid-level reduction function with atomic accumulation.
fn generate_grid_reduce_fn(ty: &str, block_size: u32, op: &ReductionOp) -> String {
    let atomic_fn = op.atomic_fn();
    let combine = op.combine_expr("shared[tid]", "shared[tid + s]");

    format!(
        r#"
// Grid-level {op} reduction with atomic accumulation
__device__ void __grid_reduce_{op}({ty} val, {ty}* shared, {ty}* accumulator) {{
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    // Block-level tree reduction
    #pragma unroll
    for (int s = {block_size} / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared[tid] = {combine};
        }}
        __syncthreads();
    }}

    // Block leader atomically accumulates
    if (tid == 0) {{
        {atomic_fn}(accumulator, shared[0]);
    }}
}}

"#,
        ty = ty,
        op = op,
        block_size = block_size,
        combine = combine,
        atomic_fn = atomic_fn
    )
}

/// Generate a reduce-and-broadcast function.
fn generate_reduce_and_broadcast_fn(
    ty: &str,
    block_size: u32,
    op: &ReductionOp,
    use_cooperative: bool,
) -> String {
    let atomic_fn = op.atomic_fn();
    let combine = op.combine_expr("shared[tid]", "shared[tid + s]");

    let grid_sync = if use_cooperative {
        "    cg::grid_group grid = cg::this_grid();\n    grid.sync();"
    } else {
        "    __software_grid_sync(&__barrier_counter, &__barrier_gen, gridDim.x * gridDim.y * gridDim.z);"
    };

    let params = if use_cooperative {
        format!("{} val, {}* shared, {}* accumulator", ty, ty, ty)
    } else {
        format!(
            "{} val, {}* shared, {}* accumulator, unsigned int* __barrier_counter, unsigned int* __barrier_gen",
            ty, ty, ty
        )
    };

    format!(
        r#"
// Reduce-and-broadcast: all threads get the global {op} result
__device__ {ty} __reduce_and_broadcast_{op}({params}) {{
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    // Phase 1: Block-level reduction
    #pragma unroll
    for (int s = {block_size} / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared[tid] = {combine};
        }}
        __syncthreads();
    }}

    // Phase 2: Block leader atomically accumulates
    if (tid == 0) {{
        {atomic_fn}(accumulator, shared[0]);
    }}

    // Phase 3: Grid-wide synchronization
{grid_sync}

    // Phase 4: All threads read the result
    return *accumulator;
}}

"#,
        ty = ty,
        op = op,
        block_size = block_size,
        combine = combine,
        atomic_fn = atomic_fn,
        params = params,
        grid_sync = grid_sync
    )
}

/// Generate a software grid barrier for non-cooperative kernels.
fn generate_software_barrier() -> String {
    r#"
// Software grid barrier using global memory atomics
// Use when cooperative groups are unavailable
__device__ void __software_grid_sync(
    volatile unsigned int* barrier_counter,
    volatile unsigned int* barrier_gen,
    unsigned int num_blocks
) {
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int gen = *barrier_gen;
        unsigned int arrived = atomicAdd((unsigned int*)barrier_counter, 1) + 1;

        if (arrived == num_blocks) {
            // Last block to arrive: reset counter and increment generation
            *barrier_counter = 0;
            __threadfence();
            atomicAdd((unsigned int*)barrier_gen, 1);
        } else {
            // Wait for all blocks
            while (atomicAdd((unsigned int*)barrier_gen, 0) == gen) {
                __threadfence();
            }
        }
    }

    __syncthreads();
}

"#
    .to_string()
}

/// Transpile a reduction intrinsic call to CUDA code.
///
/// # Arguments
///
/// * `intrinsic` - The reduction intrinsic being called
/// * `args` - The transpiled argument expressions
/// * `config` - Reduction configuration
///
/// # Returns
///
/// The CUDA expression or statement for this intrinsic.
pub fn transpile_reduction_call(
    intrinsic: ReductionIntrinsic,
    args: &[String],
    config: &ReductionCodegenConfig,
) -> Result<String, String> {
    match intrinsic {
        ReductionIntrinsic::BlockReduceSum => {
            validate_args(args, 2, "block_reduce_sum")?;
            Ok(format!("__block_reduce_sum({}, {})", args[0], args[1]))
        }
        ReductionIntrinsic::BlockReduceMin => {
            validate_args(args, 2, "block_reduce_min")?;
            Ok(format!("__block_reduce_min({}, {})", args[0], args[1]))
        }
        ReductionIntrinsic::BlockReduceMax => {
            validate_args(args, 2, "block_reduce_max")?;
            Ok(format!("__block_reduce_max({}, {})", args[0], args[1]))
        }
        ReductionIntrinsic::BlockReduceAnd => {
            validate_args(args, 2, "block_reduce_and")?;
            Ok(format!("__block_reduce_and({}, {})", args[0], args[1]))
        }
        ReductionIntrinsic::BlockReduceOr => {
            validate_args(args, 2, "block_reduce_or")?;
            Ok(format!("__block_reduce_or({}, {})", args[0], args[1]))
        }
        ReductionIntrinsic::GridReduceSum => {
            validate_args(args, 3, "grid_reduce_sum")?;
            Ok(format!(
                "__grid_reduce_sum({}, {}, {})",
                args[0], args[1], args[2]
            ))
        }
        ReductionIntrinsic::GridReduceMin => {
            validate_args(args, 3, "grid_reduce_min")?;
            Ok(format!(
                "__grid_reduce_min({}, {}, {})",
                args[0], args[1], args[2]
            ))
        }
        ReductionIntrinsic::GridReduceMax => {
            validate_args(args, 3, "grid_reduce_max")?;
            Ok(format!(
                "__grid_reduce_max({}, {}, {})",
                args[0], args[1], args[2]
            ))
        }
        ReductionIntrinsic::AtomicAccumulate => {
            validate_args(args, 2, "atomic_accumulate")?;
            Ok(format!("atomicAdd({}, {})", args[0], args[1]))
        }
        ReductionIntrinsic::AtomicAccumulateFetch => {
            validate_args(args, 2, "atomic_accumulate_fetch")?;
            Ok(format!("atomicAdd({}, {})", args[0], args[1]))
        }
        ReductionIntrinsic::BroadcastRead => {
            validate_args(args, 1, "broadcast_read")?;
            Ok(format!("(*{})", args[0]))
        }
        ReductionIntrinsic::ReduceAndBroadcast => {
            if config.use_cooperative {
                validate_args(args, 3, "reduce_and_broadcast")?;
                Ok(format!(
                    "__reduce_and_broadcast_sum({}, {}, {})",
                    args[0], args[1], args[2]
                ))
            } else {
                validate_args(args, 5, "reduce_and_broadcast (software barrier)")?;
                Ok(format!(
                    "__reduce_and_broadcast_sum({}, {}, {}, {}, {})",
                    args[0], args[1], args[2], args[3], args[4]
                ))
            }
        }
    }
}

fn validate_args(args: &[String], expected: usize, name: &str) -> Result<(), String> {
    if args.len() != expected {
        Err(format!(
            "{} requires {} arguments, got {}",
            name,
            expected,
            args.len()
        ))
    } else {
        Ok(())
    }
}

/// Generate inline reduction code (without helper function call).
///
/// Use this when you want the reduction logic inlined in a specific location.
pub fn generate_inline_block_reduce(
    value_expr: &str,
    shared_array: &str,
    result_var: &str,
    ty: &str,
    block_size: u32,
    op: &ReductionOp,
) -> String {
    let combine = op.combine_expr(
        &format!("{}[__tid]", shared_array),
        &format!("{}[__tid + __s]", shared_array),
    );

    format!(
        r#"{{
    int __tid = threadIdx.x;
    {shared_array}[__tid] = {value_expr};
    __syncthreads();

    #pragma unroll
    for (int __s = {block_size} / 2; __s > 0; __s >>= 1) {{
        if (__tid < __s) {{
            {shared_array}[__tid] = {combine};
        }}
        __syncthreads();
    }}

    {ty} {result_var} = {shared_array}[0];
}}"#,
        value_expr = value_expr,
        shared_array = shared_array,
        result_var = result_var,
        ty = ty,
        block_size = block_size,
        combine = combine
    )
}

/// Generate inline grid reduction with atomic accumulation.
pub fn generate_inline_grid_reduce(
    value_expr: &str,
    shared_array: &str,
    accumulator: &str,
    _ty: &str,
    block_size: u32,
    op: &ReductionOp,
) -> String {
    let atomic_fn = op.atomic_fn();
    let combine = op.combine_expr(
        &format!("{}[__tid]", shared_array),
        &format!("{}[__tid + __s]", shared_array),
    );

    format!(
        r#"{{
    int __tid = threadIdx.x;
    {shared_array}[__tid] = {value_expr};
    __syncthreads();

    #pragma unroll
    for (int __s = {block_size} / 2; __s > 0; __s >>= 1) {{
        if (__tid < __s) {{
            {shared_array}[__tid] = {combine};
        }}
        __syncthreads();
    }}

    if (__tid == 0) {{
        {atomic_fn}({accumulator}, {shared_array}[0]);
    }}
}}"#,
        value_expr = value_expr,
        shared_array = shared_array,
        accumulator = accumulator,
        block_size = block_size,
        combine = combine,
        atomic_fn = atomic_fn
    )
}

/// Generate inline reduce-and-broadcast code.
#[allow(clippy::too_many_arguments)]
pub fn generate_inline_reduce_and_broadcast(
    value_expr: &str,
    shared_array: &str,
    accumulator: &str,
    result_var: &str,
    ty: &str,
    block_size: u32,
    op: &ReductionOp,
    use_cooperative: bool,
) -> String {
    let atomic_fn = op.atomic_fn();
    let combine = op.combine_expr(
        &format!("{}[__tid]", shared_array),
        &format!("{}[__tid + __s]", shared_array),
    );

    let grid_sync = if use_cooperative {
        "    cg::grid_group __grid = cg::this_grid();\n    __grid.sync();"
    } else {
        "    __software_grid_sync(&__barrier_counter, &__barrier_gen, gridDim.x * gridDim.y * gridDim.z);"
    };

    format!(
        r#"{{
    int __tid = threadIdx.x;
    {shared_array}[__tid] = {value_expr};
    __syncthreads();

    // Block-level reduction
    #pragma unroll
    for (int __s = {block_size} / 2; __s > 0; __s >>= 1) {{
        if (__tid < __s) {{
            {shared_array}[__tid] = {combine};
        }}
        __syncthreads();
    }}

    // Atomic accumulation
    if (__tid == 0) {{
        {atomic_fn}({accumulator}, {shared_array}[0]);
    }}

    // Grid synchronization
{grid_sync}

    // Broadcast: all threads read result
    {ty} {result_var} = *{accumulator};
}}"#,
        value_expr = value_expr,
        shared_array = shared_array,
        accumulator = accumulator,
        result_var = result_var,
        ty = ty,
        block_size = block_size,
        combine = combine,
        atomic_fn = atomic_fn,
        grid_sync = grid_sync
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduction_op_combine() {
        assert_eq!(ReductionOp::Sum.combine_expr("a", "b"), "a + b");
        assert_eq!(ReductionOp::Min.combine_expr("a", "b"), "min(a, b)");
        assert_eq!(ReductionOp::Max.combine_expr("a", "b"), "max(a, b)");
        assert_eq!(ReductionOp::And.combine_expr("a", "b"), "a & b");
    }

    #[test]
    fn test_reduction_op_identity() {
        assert_eq!(ReductionOp::Sum.identity("float"), "0");
        assert_eq!(ReductionOp::Min.identity("float"), "INFINITY");
        assert_eq!(ReductionOp::Max.identity("float"), "-INFINITY");
        assert_eq!(ReductionOp::Product.identity("int"), "1");
    }

    #[test]
    fn test_intrinsic_from_name() {
        assert_eq!(
            ReductionIntrinsic::from_name("block_reduce_sum"),
            Some(ReductionIntrinsic::BlockReduceSum)
        );
        assert_eq!(
            ReductionIntrinsic::from_name("grid_reduce_min"),
            Some(ReductionIntrinsic::GridReduceMin)
        );
        assert_eq!(
            ReductionIntrinsic::from_name("reduce_and_broadcast"),
            Some(ReductionIntrinsic::ReduceAndBroadcast)
        );
        assert_eq!(ReductionIntrinsic::from_name("not_an_intrinsic"), None);
    }

    #[test]
    fn test_generate_block_reduce() {
        let code = generate_block_reduce_fn("float", 256, &ReductionOp::Sum);
        assert!(code.contains("__device__ float __block_reduce_sum"));
        assert!(code.contains("shared[tid] = shared[tid] + shared[tid + s]"));
        assert!(code.contains("__syncthreads()"));
    }

    #[test]
    fn test_generate_grid_reduce() {
        let code = generate_grid_reduce_fn("double", 128, &ReductionOp::Max);
        assert!(code.contains("__device__ void __grid_reduce_max"));
        assert!(code.contains("atomicMax(accumulator, shared[0])"));
    }

    #[test]
    fn test_generate_reduce_and_broadcast_cooperative() {
        let code = generate_reduce_and_broadcast_fn("float", 256, &ReductionOp::Sum, true);
        assert!(code.contains("cg::grid_group grid = cg::this_grid()"));
        assert!(code.contains("grid.sync()"));
    }

    #[test]
    fn test_generate_reduce_and_broadcast_software() {
        let code = generate_reduce_and_broadcast_fn("float", 256, &ReductionOp::Sum, false);
        assert!(code.contains("__barrier_counter"));
        assert!(code.contains("__software_grid_sync"));
    }

    #[test]
    fn test_inline_block_reduce() {
        let code = generate_inline_block_reduce(
            "my_value",
            "shared_mem",
            "result",
            "float",
            256,
            &ReductionOp::Sum,
        );
        assert!(code.contains("shared_mem[__tid] = my_value"));
        assert!(code.contains("float result = shared_mem[0]"));
    }

    #[test]
    fn test_transpile_reduction_call() {
        let config = ReductionCodegenConfig::default();

        let result = transpile_reduction_call(
            ReductionIntrinsic::BlockReduceSum,
            &["val".to_string(), "shared".to_string()],
            &config,
        );
        assert_eq!(result.unwrap(), "__block_reduce_sum(val, shared)");

        let result = transpile_reduction_call(
            ReductionIntrinsic::AtomicAccumulate,
            &["&accum".to_string(), "val".to_string()],
            &config,
        );
        assert_eq!(result.unwrap(), "atomicAdd(&accum, val)");
    }

    #[test]
    fn test_reduction_helpers() {
        let config = ReductionCodegenConfig::new(256)
            .with_type("float")
            .with_op(ReductionOp::Sum)
            .with_cooperative(true);

        let helpers = generate_reduction_helpers(&config);
        assert!(helpers.contains("#include <cooperative_groups.h>"));
        assert!(helpers.contains("__block_reduce_sum"));
        assert!(helpers.contains("__grid_reduce_sum"));
        assert!(helpers.contains("__reduce_and_broadcast_sum"));
    }
}
