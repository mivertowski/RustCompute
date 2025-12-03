//! Stencil-specific patterns and configuration for CUDA code generation.
//!
//! This module provides the configuration types and patterns for generating
//! stencil/grid kernels that follow the WaveSim FDTD style.

/// Grid dimensionality for stencil kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grid {
    /// 1D grid (line).
    Grid1D,
    /// 2D grid (plane).
    Grid2D,
    /// 3D grid (volume).
    Grid3D,
}

impl Grid {
    /// Parse from string representation.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "1d" | "grid1d" => Some(Grid::Grid1D),
            "2d" | "grid2d" => Some(Grid::Grid2D),
            "3d" | "grid3d" => Some(Grid::Grid3D),
            _ => None,
        }
    }

    /// Get the number of dimensions.
    pub fn dimensions(&self) -> usize {
        match self {
            Grid::Grid1D => 1,
            Grid::Grid2D => 2,
            Grid::Grid3D => 3,
        }
    }
}

/// Configuration for a stencil kernel.
#[derive(Debug, Clone)]
pub struct StencilConfig {
    /// Kernel identifier.
    pub id: String,
    /// Grid dimensionality.
    pub grid: Grid,
    /// Tile/block size (width, height) for 2D, or (x, y, z) for 3D.
    pub tile_size: (usize, usize),
    /// Halo/ghost cell width (stencil radius).
    pub halo: usize,
}

impl Default for StencilConfig {
    fn default() -> Self {
        Self {
            id: "stencil_kernel".to_string(),
            grid: Grid::Grid2D,
            tile_size: (16, 16),
            halo: 1,
        }
    }
}

impl StencilConfig {
    /// Create a new stencil configuration.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    /// Set the grid dimensionality.
    pub fn with_grid(mut self, grid: Grid) -> Self {
        self.grid = grid;
        self
    }

    /// Set the tile size.
    pub fn with_tile_size(mut self, width: usize, height: usize) -> Self {
        self.tile_size = (width, height);
        self
    }

    /// Set the halo width.
    pub fn with_halo(mut self, halo: usize) -> Self {
        self.halo = halo;
        self
    }

    /// Get the buffer width including halos.
    pub fn buffer_width(&self) -> usize {
        self.tile_size.0 + 2 * self.halo
    }

    /// Get the buffer height including halos (for 2D+).
    pub fn buffer_height(&self) -> usize {
        self.tile_size.1 + 2 * self.halo
    }

    /// Generate the CUDA kernel preamble with thread index calculations.
    pub fn generate_preamble(&self) -> String {
        match self.grid {
            Grid::Grid1D => self.generate_1d_preamble(),
            Grid::Grid2D => self.generate_2d_preamble(),
            Grid::Grid3D => self.generate_3d_preamble(),
        }
    }

    fn generate_1d_preamble(&self) -> String {
        let tile_size = self.tile_size.0;
        let buffer_width = self.buffer_width();

        format!(
            r#"    int lx = threadIdx.x;
    if (lx >= {tile_size}) return;

    int buffer_width = {buffer_width};
    int idx = lx + {halo};
"#,
            tile_size = tile_size,
            buffer_width = buffer_width,
            halo = self.halo,
        )
    }

    fn generate_2d_preamble(&self) -> String {
        let (tile_w, tile_h) = self.tile_size;
        let buffer_width = self.buffer_width();

        format!(
            r#"    int lx = threadIdx.x;
    int ly = threadIdx.y;
    if (lx >= {tile_w} || ly >= {tile_h}) return;

    int buffer_width = {buffer_width};
    int idx = (ly + {halo}) * buffer_width + (lx + {halo});
"#,
            tile_w = tile_w,
            tile_h = tile_h,
            buffer_width = buffer_width,
            halo = self.halo,
        )
    }

    fn generate_3d_preamble(&self) -> String {
        let (tile_w, tile_h) = self.tile_size;
        let buffer_width = self.buffer_width();
        let buffer_height = self.buffer_height();

        format!(
            r#"    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int lz = threadIdx.z;
    if (lx >= {tile_w} || ly >= {tile_h}) return;

    int buffer_width = {buffer_width};
    int buffer_height = {buffer_height};
    int buffer_slice = buffer_width * buffer_height;
    int idx = (lz + {halo}) * buffer_slice + (ly + {halo}) * buffer_width + (lx + {halo});
"#,
            tile_w = tile_w,
            tile_h = tile_h,
            buffer_width = buffer_width,
            buffer_height = buffer_height,
            halo = self.halo,
        )
    }

    /// Generate launch bounds for the kernel.
    pub fn generate_launch_bounds(&self) -> String {
        let threads = self.tile_size.0 * self.tile_size.1;
        format!("__launch_bounds__({threads})")
    }
}

/// Context for grid position within a stencil kernel.
///
/// This is a marker type that gets translated to CUDA index calculations.
/// In Rust, it provides the API; in generated CUDA, it becomes inline code.
#[derive(Debug, Clone, Copy)]
pub struct GridPos {
    // Phantom - this struct is never actually instantiated in GPU code
    _private: (),
}

impl GridPos {
    /// Get the current cell's linear index.
    ///
    /// In CUDA: `idx`
    #[inline]
    pub fn idx(&self) -> usize {
        // Placeholder - actual implementation is generated
        0
    }

    /// Access the north neighbor (y - 1).
    ///
    /// In CUDA: `buf[idx - buffer_width]`
    #[inline]
    pub fn north<T: Copy>(&self, _buf: &[T]) -> T {
        // Placeholder - actual implementation is generated
        unsafe { std::mem::zeroed() }
    }

    /// Access the south neighbor (y + 1).
    ///
    /// In CUDA: `buf[idx + buffer_width]`
    #[inline]
    pub fn south<T: Copy>(&self, _buf: &[T]) -> T {
        unsafe { std::mem::zeroed() }
    }

    /// Access the east neighbor (x + 1).
    ///
    /// In CUDA: `buf[idx + 1]`
    #[inline]
    pub fn east<T: Copy>(&self, _buf: &[T]) -> T {
        unsafe { std::mem::zeroed() }
    }

    /// Access the west neighbor (x - 1).
    ///
    /// In CUDA: `buf[idx - 1]`
    #[inline]
    pub fn west<T: Copy>(&self, _buf: &[T]) -> T {
        unsafe { std::mem::zeroed() }
    }

    /// Access a neighbor at arbitrary offset.
    ///
    /// In CUDA: `buf[idx + dy * buffer_width + dx]`
    #[inline]
    pub fn at<T: Copy>(&self, _buf: &[T], _dx: i32, _dy: i32) -> T {
        unsafe { std::mem::zeroed() }
    }

    /// Access the neighbor above (z - 1, 3D only).
    ///
    /// In CUDA: `buf[idx - buffer_slice]`
    #[inline]
    pub fn up<T: Copy>(&self, _buf: &[T]) -> T {
        unsafe { std::mem::zeroed() }
    }

    /// Access the neighbor below (z + 1, 3D only).
    ///
    /// In CUDA: `buf[idx + buffer_slice]`
    #[inline]
    pub fn down<T: Copy>(&self, _buf: &[T]) -> T {
        unsafe { std::mem::zeroed() }
    }
}

/// Launch configuration for stencil kernels.
#[derive(Debug, Clone)]
pub struct StencilLaunchConfig {
    /// Block dimensions.
    pub block_dim: (u32, u32, u32),
    /// Grid dimensions.
    pub grid_dim: (u32, u32, u32),
    /// Shared memory size in bytes.
    pub shared_mem: u32,
}

impl StencilLaunchConfig {
    /// Create launch config for a 2D stencil.
    pub fn for_2d_grid(grid_width: usize, grid_height: usize, tile_size: (usize, usize)) -> Self {
        let tiles_x = grid_width.div_ceil(tile_size.0);
        let tiles_y = grid_height.div_ceil(tile_size.1);

        Self {
            block_dim: (tile_size.0 as u32, tile_size.1 as u32, 1),
            grid_dim: (tiles_x as u32, tiles_y as u32, 1),
            shared_mem: 0,
        }
    }

    /// Create launch config for packed tile execution (one block per tile).
    pub fn for_packed_tiles(num_tiles: usize, tile_size: (usize, usize)) -> Self {
        Self {
            block_dim: (tile_size.0 as u32, tile_size.1 as u32, 1),
            grid_dim: (num_tiles as u32, 1, 1),
            shared_mem: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_parsing() {
        assert_eq!(Grid::parse("2d"), Some(Grid::Grid2D));
        assert_eq!(Grid::parse("Grid3D"), Some(Grid::Grid3D));
        assert_eq!(Grid::parse("1D"), Some(Grid::Grid1D));
        assert_eq!(Grid::parse("invalid"), None);
    }

    #[test]
    fn test_stencil_config_defaults() {
        let config = StencilConfig::default();
        assert_eq!(config.tile_size, (16, 16));
        assert_eq!(config.halo, 1);
        assert_eq!(config.grid, Grid::Grid2D);
    }

    #[test]
    fn test_buffer_dimensions() {
        let config = StencilConfig::new("test")
            .with_tile_size(16, 16)
            .with_halo(1);

        assert_eq!(config.buffer_width(), 18);
        assert_eq!(config.buffer_height(), 18);
    }

    #[test]
    fn test_2d_preamble_generation() {
        let config = StencilConfig::new("fdtd")
            .with_grid(Grid::Grid2D)
            .with_tile_size(16, 16)
            .with_halo(1);

        let preamble = config.generate_preamble();

        assert!(preamble.contains("threadIdx.x"));
        assert!(preamble.contains("threadIdx.y"));
        assert!(preamble.contains("buffer_width = 18"));
        assert!(preamble.contains("if (lx >= 16 || ly >= 16) return;"));
    }

    #[test]
    fn test_1d_preamble_generation() {
        let config = StencilConfig::new("blur")
            .with_grid(Grid::Grid1D)
            .with_tile_size(256, 1)
            .with_halo(2);

        let preamble = config.generate_preamble();

        assert!(preamble.contains("threadIdx.x"));
        assert!(!preamble.contains("threadIdx.y"));
        assert!(preamble.contains("buffer_width = 260")); // 256 + 2*2
    }

    #[test]
    fn test_launch_config_2d() {
        let config = StencilLaunchConfig::for_2d_grid(256, 256, (16, 16));

        assert_eq!(config.block_dim, (16, 16, 1));
        assert_eq!(config.grid_dim, (16, 16, 1)); // 256/16 = 16 tiles each dim
    }

    #[test]
    fn test_launch_config_packed() {
        let config = StencilLaunchConfig::for_packed_tiles(100, (16, 16));

        assert_eq!(config.block_dim, (16, 16, 1));
        assert_eq!(config.grid_dim, (100, 1, 1));
    }
}
