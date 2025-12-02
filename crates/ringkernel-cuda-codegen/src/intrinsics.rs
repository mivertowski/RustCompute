//! GPU intrinsic mapping for CUDA code generation.
//!
//! This module provides mappings from high-level Rust operations to
//! CUDA intrinsics and built-in functions.

use std::collections::HashMap;

/// GPU intrinsic operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuIntrinsic {
    /// Thread synchronization.
    SyncThreads,
    /// Thread fence (memory ordering).
    ThreadFence,
    ThreadFenceBlock,
    ThreadFenceSystem,

    /// Atomic operations.
    AtomicAdd,
    AtomicSub,
    AtomicMin,
    AtomicMax,
    AtomicExch,
    AtomicCas,

    /// Math functions.
    Sqrt,
    Rsqrt,
    Abs,
    Fabs,
    Floor,
    Ceil,
    Round,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Pow,
    Fma,
    Min,
    Max,

    /// Warp-level operations.
    WarpShfl,
    WarpShflUp,
    WarpShflDown,
    WarpShflXor,
    WarpActiveMask,
    WarpBallot,
    WarpAll,
    WarpAny,
}

impl GpuIntrinsic {
    /// Convert to CUDA function/intrinsic name.
    pub fn to_cuda_string(&self) -> &'static str {
        match self {
            GpuIntrinsic::SyncThreads => "__syncthreads()",
            GpuIntrinsic::ThreadFence => "__threadfence()",
            GpuIntrinsic::ThreadFenceBlock => "__threadfence_block()",
            GpuIntrinsic::ThreadFenceSystem => "__threadfence_system()",
            GpuIntrinsic::AtomicAdd => "atomicAdd",
            GpuIntrinsic::AtomicSub => "atomicSub",
            GpuIntrinsic::AtomicMin => "atomicMin",
            GpuIntrinsic::AtomicMax => "atomicMax",
            GpuIntrinsic::AtomicExch => "atomicExch",
            GpuIntrinsic::AtomicCas => "atomicCAS",
            GpuIntrinsic::Sqrt => "sqrtf",
            GpuIntrinsic::Rsqrt => "rsqrtf",
            GpuIntrinsic::Abs => "abs",
            GpuIntrinsic::Fabs => "fabsf",
            GpuIntrinsic::Floor => "floorf",
            GpuIntrinsic::Ceil => "ceilf",
            GpuIntrinsic::Round => "roundf",
            GpuIntrinsic::Sin => "sinf",
            GpuIntrinsic::Cos => "cosf",
            GpuIntrinsic::Tan => "tanf",
            GpuIntrinsic::Exp => "expf",
            GpuIntrinsic::Log => "logf",
            GpuIntrinsic::Pow => "powf",
            GpuIntrinsic::Fma => "fmaf",
            GpuIntrinsic::Min => "fminf",
            GpuIntrinsic::Max => "fmaxf",
            GpuIntrinsic::WarpShfl => "__shfl_sync",
            GpuIntrinsic::WarpShflUp => "__shfl_up_sync",
            GpuIntrinsic::WarpShflDown => "__shfl_down_sync",
            GpuIntrinsic::WarpShflXor => "__shfl_xor_sync",
            GpuIntrinsic::WarpActiveMask => "__activemask()",
            GpuIntrinsic::WarpBallot => "__ballot_sync",
            GpuIntrinsic::WarpAll => "__all_sync",
            GpuIntrinsic::WarpAny => "__any_sync",
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

        // Synchronization
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

        // Atomics (common naming)
        mappings.insert("atomic_add".to_string(), GpuIntrinsic::AtomicAdd);
        mappings.insert("atomic_sub".to_string(), GpuIntrinsic::AtomicSub);
        mappings.insert("atomic_min".to_string(), GpuIntrinsic::AtomicMin);
        mappings.insert("atomic_max".to_string(), GpuIntrinsic::AtomicMax);
        mappings.insert("atomic_exchange".to_string(), GpuIntrinsic::AtomicExch);
        mappings.insert("atomic_cas".to_string(), GpuIntrinsic::AtomicCas);

        // Math functions (Rust std naming)
        mappings.insert("sqrt".to_string(), GpuIntrinsic::Sqrt);
        mappings.insert("abs".to_string(), GpuIntrinsic::Fabs);
        mappings.insert("floor".to_string(), GpuIntrinsic::Floor);
        mappings.insert("ceil".to_string(), GpuIntrinsic::Ceil);
        mappings.insert("round".to_string(), GpuIntrinsic::Round);
        mappings.insert("sin".to_string(), GpuIntrinsic::Sin);
        mappings.insert("cos".to_string(), GpuIntrinsic::Cos);
        mappings.insert("tan".to_string(), GpuIntrinsic::Tan);
        mappings.insert("exp".to_string(), GpuIntrinsic::Exp);
        mappings.insert("ln".to_string(), GpuIntrinsic::Log);
        mappings.insert("log".to_string(), GpuIntrinsic::Log);
        mappings.insert("powf".to_string(), GpuIntrinsic::Pow);
        mappings.insert("powi".to_string(), GpuIntrinsic::Pow);
        mappings.insert("mul_add".to_string(), GpuIntrinsic::Fma);
        mappings.insert("min".to_string(), GpuIntrinsic::Min);
        mappings.insert("max".to_string(), GpuIntrinsic::Max);

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
        assert_eq!(GpuIntrinsic::SyncThreads.to_cuda_string(), "__syncthreads()");
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
}
