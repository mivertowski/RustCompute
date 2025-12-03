//! Stencil kernel support for WGSL code generation.
//!
//! Provides GridPos abstraction and stencil kernel configuration.

/// Configuration for stencil kernel generation.
#[derive(Debug, Clone)]
pub struct StencilConfig {
    /// Kernel name.
    pub name: String,
    /// Tile width (workgroup size X).
    pub tile_width: u32,
    /// Tile height (workgroup size Y).
    pub tile_height: u32,
    /// Halo size for neighbor access.
    pub halo: u32,
    /// Whether to use shared memory for the tile.
    pub use_shared_memory: bool,
}

impl StencilConfig {
    /// Create a new stencil configuration.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tile_width: 16,
            tile_height: 16,
            halo: 1,
            use_shared_memory: true,
        }
    }

    /// Set tile size.
    pub fn with_tile_size(mut self, width: u32, height: u32) -> Self {
        self.tile_width = width;
        self.tile_height = height;
        self
    }

    /// Set halo size.
    pub fn with_halo(mut self, halo: u32) -> Self {
        self.halo = halo;
        self
    }

    /// Disable shared memory usage.
    pub fn without_shared_memory(mut self) -> Self {
        self.use_shared_memory = false;
        self
    }

    /// Get the buffer width including halo.
    pub fn buffer_width(&self) -> u32 {
        self.tile_width + 2 * self.halo
    }

    /// Get the buffer height including halo.
    pub fn buffer_height(&self) -> u32 {
        self.tile_height + 2 * self.halo
    }

    /// Get the workgroup size annotation.
    pub fn workgroup_size_annotation(&self) -> String {
        format!(
            "@workgroup_size({}, {}, 1)",
            self.tile_width, self.tile_height
        )
    }
}

/// Launch configuration for stencil kernels.
#[derive(Debug, Clone)]
pub struct StencilLaunchConfig {
    /// Grid dimensions.
    pub grid: Grid,
    /// Block dimensions (derived from StencilConfig).
    pub block_width: u32,
    pub block_height: u32,
}

impl StencilLaunchConfig {
    /// Create a new launch configuration.
    pub fn new(grid: Grid, config: &StencilConfig) -> Self {
        Self {
            grid,
            block_width: config.tile_width,
            block_height: config.tile_height,
        }
    }
}

/// Grid dimensions for kernel dispatch.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    /// Width in cells.
    pub width: u32,
    /// Height in cells.
    pub height: u32,
    /// Depth (for 3D grids).
    pub depth: u32,
}

impl Grid {
    /// Create a 2D grid.
    pub fn new_2d(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            depth: 1,
        }
    }

    /// Create a 3D grid.
    pub fn new_3d(width: u32, height: u32, depth: u32) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    /// Calculate number of workgroups needed for this grid.
    pub fn workgroups(&self, tile_width: u32, tile_height: u32) -> (u32, u32, u32) {
        let x = (self.width + tile_width - 1) / tile_width;
        let y = (self.height + tile_height - 1) / tile_height;
        (x, y, self.depth)
    }
}

/// Grid position context passed to stencil kernels.
///
/// This is a compile-time marker type. The transpiler recognizes method calls
/// on `GridPos` and converts them to WGSL index calculations.
#[derive(Debug, Clone, Copy)]
pub struct GridPos {
    /// X coordinate in the grid.
    pub x: i32,
    /// Y coordinate in the grid.
    pub y: i32,
    /// Linear index in the buffer.
    pub idx: usize,
    /// Buffer stride (width).
    pub stride: usize,
}

impl GridPos {
    /// Get the linear index for the current position.
    ///
    /// Maps to: `idx` (local variable in generated WGSL)
    #[inline(always)]
    pub fn idx(&self) -> usize {
        self.idx
    }

    /// Get the x coordinate.
    #[inline(always)]
    pub fn x(&self) -> i32 {
        self.x
    }

    /// Get the y coordinate.
    #[inline(always)]
    pub fn y(&self) -> i32 {
        self.y
    }

    /// Access the north neighbor (y - 1).
    ///
    /// Maps to: `buffer[idx - buffer_width]`
    #[inline(always)]
    pub fn north<T: Copy>(&self, buffer: &[T]) -> T {
        buffer[self.idx - self.stride]
    }

    /// Access the south neighbor (y + 1).
    ///
    /// Maps to: `buffer[idx + buffer_width]`
    #[inline(always)]
    pub fn south<T: Copy>(&self, buffer: &[T]) -> T {
        buffer[self.idx + self.stride]
    }

    /// Access the east neighbor (x + 1).
    ///
    /// Maps to: `buffer[idx + 1]`
    #[inline(always)]
    pub fn east<T: Copy>(&self, buffer: &[T]) -> T {
        buffer[self.idx + 1]
    }

    /// Access the west neighbor (x - 1).
    ///
    /// Maps to: `buffer[idx - 1]`
    #[inline(always)]
    pub fn west<T: Copy>(&self, buffer: &[T]) -> T {
        buffer[self.idx - 1]
    }

    /// Access a neighbor at relative offset (dx, dy).
    ///
    /// Maps to: `buffer[idx + dy * buffer_width + dx]`
    #[inline(always)]
    pub fn at<T: Copy>(&self, buffer: &[T], dx: i32, dy: i32) -> T {
        let offset = (dy as isize * self.stride as isize + dx as isize) as usize;
        buffer[self.idx.wrapping_add(offset)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stencil_config() {
        let config = StencilConfig::new("heat")
            .with_tile_size(16, 16)
            .with_halo(1);

        assert_eq!(config.name, "heat");
        assert_eq!(config.tile_width, 16);
        assert_eq!(config.tile_height, 16);
        assert_eq!(config.halo, 1);
        assert_eq!(config.buffer_width(), 18);
        assert_eq!(config.buffer_height(), 18);
    }

    #[test]
    fn test_grid_workgroups() {
        let grid = Grid::new_2d(256, 256);
        let (wx, wy, wz) = grid.workgroups(16, 16);
        assert_eq!(wx, 16);
        assert_eq!(wy, 16);
        assert_eq!(wz, 1);
    }

    #[test]
    fn test_workgroup_size_annotation() {
        let config = StencilConfig::new("test").with_tile_size(8, 8);
        assert_eq!(config.workgroup_size_annotation(), "@workgroup_size(8, 8, 1)");
    }
}
