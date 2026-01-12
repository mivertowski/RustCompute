//! Unified GPU backend trait for tile-based FDTD simulation.
//!
//! This module provides an abstraction layer over CUDA and WGPU backends,
//! enabling GPU-resident state with minimal host transfers.
//!
//! ## Design
//!
//! The key optimization is keeping simulation state on the GPU:
//! - Pressure buffers persist on GPU across steps (no per-step upload/download)
//! - Only halo data (64 bytes per tile edge) transfers for K2K messaging
//! - Full grid readback only when GUI needs to render
//!
//! ## Buffer Layout
//!
//! Each tile uses an 18x18 buffer (324 floats = 1,296 bytes):
//!
//! ```text
//! +---+----------------+---+
//! | NW|   North Halo   |NE |  <- Row 0 (from neighbor)
//! +---+----------------+---+
//! |   |                |   |
//! | W |   16x16 Tile   | E |  <- Rows 1-16 (owned)
//! |   |    Interior    |   |
//! +---+----------------+---+
//! | SW|   South Halo   |SE |  <- Row 17 (from neighbor)
//! +---+----------------+---+
//! ```

use ringkernel_core::error::Result;

/// Edge direction for halo exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Edge {
    /// North edge (top row of interior)
    North = 0,
    /// South edge (bottom row of interior)
    South = 1,
    /// West edge (left column of interior)
    West = 2,
    /// East edge (right column of interior)
    East = 3,
}

/// Boundary condition type for domain edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum BoundaryCondition {
    /// Absorbing boundary - pressure goes to zero (default).
    /// Simulates infinite domain where waves exit and don't return.
    #[default]
    Absorbing = 0,
    /// Reflecting boundary - mirrors interior values to halo.
    /// Simulates a hard wall where waves bounce back.
    Reflecting = 1,
    /// Periodic boundary - wraps around to opposite edge.
    /// Simulates an infinite repeating pattern.
    Periodic = 2,
}

impl Edge {
    /// Get the opposite edge (for K2K routing).
    pub fn opposite(self) -> Self {
        match self {
            Edge::North => Edge::South,
            Edge::South => Edge::North,
            Edge::West => Edge::East,
            Edge::East => Edge::West,
        }
    }

    /// All edges in order.
    pub const ALL: [Edge; 4] = [Edge::North, Edge::South, Edge::West, Edge::East];
}

/// Parameters for FDTD computation.
#[derive(Debug, Clone, Copy)]
pub struct FdtdParams {
    /// Tile interior size (e.g., 16 for 16x16 tiles).
    pub tile_size: u32,
    /// Courant number squared (c² = (speed * dt / dx)²).
    pub c2: f32,
    /// Damping factor (1.0 - damping_coefficient).
    pub damping: f32,
}

impl FdtdParams {
    /// Create new FDTD parameters.
    pub fn new(tile_size: u32, c2: f32, damping: f32) -> Self {
        Self {
            tile_size,
            c2,
            damping,
        }
    }
}

/// Unified trait for GPU backends (CUDA and WGPU).
///
/// Implementations must provide GPU-resident tile buffers with:
/// - Double-buffered pressure (ping-pong)
/// - Halo staging buffers for K2K exchange
/// - Minimal transfer operations
pub trait TileGpuBackend: Send + Sync {
    /// GPU buffer type for this backend.
    type Buffer: Send + Sync;

    /// Create GPU buffers for a tile.
    ///
    /// Returns buffers for:
    /// - pressure: 18x18 f32 buffer (current state)
    /// - pressure_prev: 18x18 f32 buffer (previous state)
    /// - halo_staging: 4 × 16 f32 buffers (one per edge)
    fn create_tile_buffers(&self, tile_size: u32) -> Result<TileGpuBuffers<Self::Buffer>>;

    /// Upload initial pressure data (one-time, at tile creation).
    fn upload_initial_state(
        &self,
        buffers: &TileGpuBuffers<Self::Buffer>,
        pressure: &[f32],
        pressure_prev: &[f32],
    ) -> Result<()>;

    /// Execute FDTD step entirely on GPU (no host transfer).
    ///
    /// Reads from current buffer, writes to previous buffer (ping-pong).
    fn fdtd_step(&self, buffers: &TileGpuBuffers<Self::Buffer>, params: &FdtdParams) -> Result<()>;

    /// Extract halo from GPU to host (small transfer for K2K).
    ///
    /// Returns 16 f32 values for the specified edge.
    fn extract_halo(&self, buffers: &TileGpuBuffers<Self::Buffer>, edge: Edge) -> Result<Vec<f32>>;

    /// Inject neighbor halo from K2K message into GPU buffer.
    ///
    /// Takes 16 f32 values to write to the specified halo region.
    fn inject_halo(
        &self,
        buffers: &TileGpuBuffers<Self::Buffer>,
        edge: Edge,
        data: &[f32],
    ) -> Result<()>;

    /// Swap ping-pong buffers (pointer swap, no data movement).
    fn swap_buffers(&self, buffers: &mut TileGpuBuffers<Self::Buffer>);

    /// Read full tile pressure for visualization.
    ///
    /// Only called when GUI needs to render (once per frame).
    /// Returns 16x16 interior values (not the full 18x18 buffer).
    fn read_interior_pressure(&self, buffers: &TileGpuBuffers<Self::Buffer>) -> Result<Vec<f32>>;

    /// Apply boundary condition for a domain edge tile.
    ///
    /// For tiles at the domain boundary (without a neighbor on one or more edges),
    /// this method applies the specified boundary condition to the halo region:
    /// - Absorbing: Sets halo to zero (waves exit and don't return)
    /// - Reflecting: Mirrors interior values to halo (hard wall)
    /// - Periodic: Would need opposite edge data (handled separately)
    ///
    /// This is a GPU operation - more efficient than CPU-side manipulation.
    fn apply_boundary(
        &self,
        buffers: &TileGpuBuffers<Self::Buffer>,
        edge: Edge,
        condition: BoundaryCondition,
    ) -> Result<()>;

    /// Synchronize GPU operations.
    fn synchronize(&self) -> Result<()>;
}

/// GPU buffers for a single tile.
///
/// Uses generic Buffer type to support both CUDA and WGPU backends.
pub struct TileGpuBuffers<B> {
    /// Current pressure buffer (18x18 with halos).
    pub pressure_a: B,
    /// Previous pressure buffer (18x18 with halos).
    pub pressure_b: B,
    /// Staging buffers for halo extraction (4 edges × 16 cells).
    pub halo_north: B,
    pub halo_south: B,
    pub halo_west: B,
    pub halo_east: B,
    /// Which buffer is "current" (true = A, false = B).
    pub current_is_a: bool,
    /// Tile interior size.
    pub tile_size: u32,
    /// Buffer width including halos.
    pub buffer_width: u32,
}

impl<B> TileGpuBuffers<B> {
    /// Get buffer width (tile_size + 2 for halos).
    pub fn buffer_width(&self) -> u32 {
        self.buffer_width
    }

    /// Get total buffer size in f32 elements.
    pub fn buffer_size(&self) -> usize {
        (self.buffer_width * self.buffer_width) as usize
    }

    /// Get halo buffer for an edge.
    pub fn halo_buffer(&self, edge: Edge) -> &B {
        match edge {
            Edge::North => &self.halo_north,
            Edge::South => &self.halo_south,
            Edge::West => &self.halo_west,
            Edge::East => &self.halo_east,
        }
    }

    /// Get mutable halo buffer for an edge.
    pub fn halo_buffer_mut(&mut self, edge: Edge) -> &mut B {
        match edge {
            Edge::North => &mut self.halo_north,
            Edge::South => &mut self.halo_south,
            Edge::West => &mut self.halo_west,
            Edge::East => &mut self.halo_east,
        }
    }
}

/// Calculate buffer index for interior coordinates (0-indexed).
#[inline(always)]
pub fn buffer_index(local_x: usize, local_y: usize, buffer_width: usize) -> usize {
    // Add 1 to account for halo border
    (local_y + 1) * buffer_width + (local_x + 1)
}

/// Calculate halo row index (row 0 or row buffer_width-1).
#[inline(always)]
pub fn halo_row_start(edge: Edge, buffer_width: usize) -> usize {
    match edge {
        Edge::North => 1,                                     // Row 0, cols 1..tile_size+1
        Edge::South => (buffer_width - 1) * buffer_width + 1, // Last row, cols 1..tile_size+1
        Edge::West | Edge::East => unreachable!(),
    }
}

/// Calculate halo column index.
#[inline(always)]
pub fn halo_col_index(edge: Edge, y: usize, buffer_width: usize) -> usize {
    match edge {
        Edge::West => (y + 1) * buffer_width, // Col 0
        Edge::East => (y + 1) * buffer_width + buffer_width - 1, // Last col
        Edge::North | Edge::South => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_opposite() {
        assert_eq!(Edge::North.opposite(), Edge::South);
        assert_eq!(Edge::South.opposite(), Edge::North);
        assert_eq!(Edge::East.opposite(), Edge::West);
        assert_eq!(Edge::West.opposite(), Edge::East);
    }

    #[test]
    fn test_buffer_index() {
        // 18x18 buffer for 16x16 tile
        let bw = 18;

        // Interior (0,0) should be at buffer position (1,1) = row 1, col 1
        assert_eq!(buffer_index(0, 0, bw), 18 + 1);
        assert_eq!(buffer_index(0, 0, bw), 19);

        // Interior (15,15) should be at buffer position (16,16)
        assert_eq!(buffer_index(15, 15, bw), 16 * 18 + 16);
        assert_eq!(buffer_index(15, 15, bw), 304);
    }

    #[test]
    fn test_fdtd_params() {
        let params = FdtdParams::new(16, 0.25, 0.99);
        assert_eq!(params.tile_size, 16);
        assert_eq!(params.c2, 0.25);
        assert_eq!(params.damping, 0.99);
    }
}
