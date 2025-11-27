//! Core type definitions for GPU thread identification and coordination.

use bytemuck::{Pod, Zeroable};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

/// Thread ID within a block (0 to block_size - 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[derive(AsBytes, FromBytes, FromZeroes, Pod, Zeroable)]
#[repr(C)]
pub struct ThreadId {
    /// X dimension
    pub x: u32,
    /// Y dimension
    pub y: u32,
    /// Z dimension
    pub z: u32,
}

impl ThreadId {
    /// Create a new 1D thread ID.
    pub const fn new_1d(x: u32) -> Self {
        Self { x, y: 0, z: 0 }
    }

    /// Create a new 2D thread ID.
    pub const fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 0 }
    }

    /// Create a new 3D thread ID.
    pub const fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Get linear index for 1D launch.
    pub const fn linear(&self) -> u32 {
        self.x
    }

    /// Get linear index for given block dimensions.
    pub const fn linear_for_dim(&self, dim_x: u32, dim_y: u32) -> u32 {
        self.x + self.y * dim_x + self.z * dim_x * dim_y
    }
}

/// Block ID within a grid (0 to grid_size - 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[derive(AsBytes, FromBytes, FromZeroes, Pod, Zeroable)]
#[repr(C)]
pub struct BlockId {
    /// X dimension
    pub x: u32,
    /// Y dimension
    pub y: u32,
    /// Z dimension
    pub z: u32,
}

impl BlockId {
    /// Create a new 1D block ID.
    pub const fn new_1d(x: u32) -> Self {
        Self { x, y: 0, z: 0 }
    }

    /// Create a new 2D block ID.
    pub const fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 0 }
    }

    /// Create a new 3D block ID.
    pub const fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Get linear index for given grid dimensions.
    pub const fn linear_for_dim(&self, dim_x: u32, dim_y: u32) -> u32 {
        self.x + self.y * dim_x + self.z * dim_x * dim_y
    }
}

/// Global thread ID across all blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[derive(AsBytes, FromBytes, FromZeroes, Pod, Zeroable)]
#[repr(C)]
pub struct GlobalThreadId {
    /// X dimension
    pub x: u32,
    /// Y dimension
    pub y: u32,
    /// Z dimension
    pub z: u32,
}

impl GlobalThreadId {
    /// Create a new global thread ID.
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Calculate global thread ID from block and thread IDs.
    pub const fn from_block_thread(
        block: BlockId,
        thread: ThreadId,
        block_dim: Dim3,
    ) -> Self {
        Self {
            x: block.x * block_dim.x + thread.x,
            y: block.y * block_dim.y + thread.y,
            z: block.z * block_dim.z + thread.z,
        }
    }

    /// Get linear index for given grid dimensions.
    pub const fn linear_for_dim(&self, dim_x: u32, dim_y: u32) -> u32 {
        self.x + self.y * dim_x + self.z * dim_x * dim_y
    }
}

/// Warp ID within a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[derive(AsBytes, FromBytes, FromZeroes, Pod, Zeroable)]
#[repr(C)]
pub struct WarpId(pub u32);

impl WarpId {
    /// Warp size (32 threads per warp on NVIDIA GPUs).
    pub const WARP_SIZE: u32 = 32;

    /// Calculate warp ID from thread ID.
    pub const fn from_thread_linear(thread_linear: u32) -> Self {
        Self(thread_linear / Self::WARP_SIZE)
    }

    /// Get lane ID within warp (0-31).
    pub const fn lane_id(thread_linear: u32) -> u32 {
        thread_linear % Self::WARP_SIZE
    }
}

/// 3D dimension specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(AsBytes, FromBytes, FromZeroes, Pod, Zeroable)]
#[repr(C)]
pub struct Dim3 {
    /// X dimension
    pub x: u32,
    /// Y dimension
    pub y: u32,
    /// Z dimension
    pub z: u32,
}

impl Dim3 {
    /// Create a 1D dimension.
    pub const fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Create a 2D dimension.
    pub const fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Create a 3D dimension.
    pub const fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total number of elements.
    pub const fn total(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl Default for Dim3 {
    fn default() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }
}

/// Memory fence scope for synchronization operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FenceScope {
    /// Thread-local fence (compiler barrier).
    Thread = 0,
    /// Block-scope fence (shared memory visible).
    Block = 1,
    /// Device-scope fence (all GPU threads).
    Device = 2,
    /// System-scope fence (CPU + GPU visible).
    System = 3,
}

/// Memory ordering for atomic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemoryOrder {
    /// Relaxed ordering (no synchronization).
    Relaxed = 0,
    /// Acquire ordering (reads synchronized).
    Acquire = 1,
    /// Release ordering (writes synchronized).
    Release = 2,
    /// Acquire-release ordering.
    AcquireRelease = 3,
    /// Sequential consistency.
    SeqCst = 4,
}

/// Kernel execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KernelMode {
    /// Persistent kernel (runs until explicitly terminated).
    /// Supported on native Linux with CUDA.
    #[default]
    Persistent,
    /// Event-driven mode (kernel relaunched per message batch).
    /// Used on WSL2, Metal, and WebGPU.
    EventDriven,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_id_linear() {
        let tid = ThreadId::new_3d(2, 3, 1);
        // For block dim (8, 4, 2): index = 2 + 3*8 + 1*8*4 = 2 + 24 + 32 = 58
        assert_eq!(tid.linear_for_dim(8, 4), 58);
    }

    #[test]
    fn test_global_thread_id() {
        let block = BlockId::new_1d(2);
        let thread = ThreadId::new_1d(5);
        let block_dim = Dim3::new_1d(32);

        let global = GlobalThreadId::from_block_thread(block, thread, block_dim);
        assert_eq!(global.x, 2 * 32 + 5);
    }

    #[test]
    fn test_warp_id() {
        assert_eq!(WarpId::from_thread_linear(0).0, 0);
        assert_eq!(WarpId::from_thread_linear(31).0, 0);
        assert_eq!(WarpId::from_thread_linear(32).0, 1);
        assert_eq!(WarpId::lane_id(35), 3);
    }

    #[test]
    fn test_dim3_total() {
        assert_eq!(Dim3::new_1d(256).total(), 256);
        assert_eq!(Dim3::new_2d(16, 16).total(), 256);
        assert_eq!(Dim3::new_3d(8, 8, 4).total(), 256);
    }
}
