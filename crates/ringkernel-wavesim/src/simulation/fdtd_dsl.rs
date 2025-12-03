//! FDTD kernel defined using the Rust DSL.
//!
//! This module demonstrates how to define stencil kernels in pure Rust
//! that get transpiled to CUDA at compile time.
//!
//! The generated CUDA code is equivalent to the handwritten version in
//! `shaders/fdtd_tile.cu`, but is written in a high-level Rust DSL.

#[cfg(feature = "cuda-codegen")]
use ringkernel_cuda_codegen::{transpile_stencil_kernel, Grid, StencilConfig};

// Re-export or define GridPos
#[cfg(feature = "cuda-codegen")]
pub use ringkernel_cuda_codegen::GridPos;

/// CPU-side GridPos for when cuda-codegen is not enabled.
///
/// This provides a fully functional CPU implementation that mirrors
/// the CUDA version's semantics for testing and verification.
#[cfg(not(feature = "cuda-codegen"))]
#[derive(Debug, Clone, Copy)]
pub struct GridPos {
    /// Current linear index into the buffer.
    index: usize,
    /// Buffer width (tile_size + 2 * halo).
    buffer_width: usize,
}

#[cfg(not(feature = "cuda-codegen"))]
impl GridPos {
    /// Create a new GridPos for CPU execution.
    ///
    /// # Arguments
    /// * `index` - Current cell's linear index
    /// * `buffer_width` - Width of the buffer including halos
    pub fn new(index: usize, buffer_width: usize) -> Self {
        Self {
            index,
            buffer_width,
        }
    }

    /// Create a GridPos from 2D coordinates within the interior.
    ///
    /// # Arguments
    /// * `lx` - Local x coordinate (0..tile_size)
    /// * `ly` - Local y coordinate (0..tile_size)
    /// * `tile_size` - Size of the interior tile
    /// * `halo` - Halo width
    pub fn from_local(lx: usize, ly: usize, tile_size: usize, halo: usize) -> Self {
        let buffer_width = tile_size + 2 * halo;
        let index = (ly + halo) * buffer_width + (lx + halo);
        Self {
            index,
            buffer_width,
        }
    }

    /// Get the current cell's linear index.
    #[inline]
    pub fn idx(&self) -> usize {
        self.index
    }

    /// Access the north neighbor (y - 1).
    #[inline]
    pub fn north<T: Copy>(&self, buf: &[T]) -> T {
        buf[self.index - self.buffer_width]
    }

    /// Access the south neighbor (y + 1).
    #[inline]
    pub fn south<T: Copy>(&self, buf: &[T]) -> T {
        buf[self.index + self.buffer_width]
    }

    /// Access the east neighbor (x + 1).
    #[inline]
    pub fn east<T: Copy>(&self, buf: &[T]) -> T {
        buf[self.index + 1]
    }

    /// Access the west neighbor (x - 1).
    #[inline]
    pub fn west<T: Copy>(&self, buf: &[T]) -> T {
        buf[self.index - 1]
    }

    /// Access a neighbor at arbitrary offset.
    #[inline]
    pub fn at<T: Copy>(&self, buf: &[T], dx: i32, dy: i32) -> T {
        let offset = dy as isize * self.buffer_width as isize + dx as isize;
        buf[(self.index as isize + offset) as usize]
    }

    /// Access the neighbor above (z - 1, 3D only).
    ///
    /// For 2D grids, this is equivalent to `north`.
    #[inline]
    pub fn up<T: Copy>(&self, buf: &[T]) -> T {
        self.north(buf)
    }

    /// Access the neighbor below (z + 1, 3D only).
    ///
    /// For 2D grids, this is equivalent to `south`.
    #[inline]
    pub fn down<T: Copy>(&self, buf: &[T]) -> T {
        self.south(buf)
    }
}

/// FDTD wave equation kernel for 16x16 tiles with 1-cell halo.
///
/// This is a pure Rust implementation of the FDTD step that can be
/// used for CPU fallback and verification.
pub fn fdtd_wave_step_cpu(
    pressure: &[f32],
    pressure_prev: &mut [f32],
    c2: f32,
    damping: f32,
    idx: usize,
    buffer_width: usize,
) {
    let p = pressure[idx];
    let p_prev = pressure_prev[idx];

    // Compute discrete Laplacian using neighbor access
    let laplacian = pressure[idx - buffer_width]  // North
        + pressure[idx + buffer_width]             // South
        + pressure[idx + 1]                        // East
        + pressure[idx - 1]                        // West
        - 4.0 * p;

    // FDTD update: p_new = 2*p - p_prev + c²*laplacian
    let p_new = 2.0 * p - p_prev + c2 * laplacian;

    // Apply damping and store result
    pressure_prev[idx] = p_new * damping;
}

/// FDTD wave equation kernel using the GridPos DSL.
///
/// This version uses the same API as the CUDA-transpiled version,
/// demonstrating that the Rust DSL code works identically on CPU.
///
/// The function signature matches what `#[stencil_kernel]` would generate:
/// ```ignore
/// fn fdtd_wave_step(
///     pressure: &[f32],
///     pressure_prev: &mut [f32],
///     c2: f32,
///     damping: f32,
///     pos: GridPos,
/// )
/// ```
#[cfg(not(feature = "cuda-codegen"))]
pub fn fdtd_wave_step_dsl(
    pressure: &[f32],
    pressure_prev: &mut [f32],
    c2: f32,
    damping: f32,
    pos: GridPos,
) {
    let p = pressure[pos.idx()];
    let p_prev = pressure_prev[pos.idx()];
    let laplacian =
        pos.north(pressure) + pos.south(pressure) + pos.east(pressure) + pos.west(pressure)
            - 4.0 * p;
    let p_new = 2.0 * p - p_prev + c2 * laplacian;
    pressure_prev[pos.idx()] = p_new * damping;
}

/// Run FDTD on an entire tile using the DSL kernel.
///
/// This simulates what CUDA would do: run the kernel for each interior cell.
#[cfg(not(feature = "cuda-codegen"))]
pub fn fdtd_tile_step_dsl(
    pressure: &[f32],
    pressure_prev: &mut [f32],
    c2: f32,
    damping: f32,
    tile_size: usize,
    halo: usize,
) {
    for ly in 0..tile_size {
        for lx in 0..tile_size {
            let pos = GridPos::from_local(lx, ly, tile_size, halo);
            fdtd_wave_step_dsl(pressure, pressure_prev, c2, damping, pos);
        }
    }
}

/// Generate CUDA source for the FDTD kernel using the DSL transpiler.
///
/// This function demonstrates using the transpiler directly to generate
/// CUDA code from a Rust function AST.
#[cfg(feature = "cuda-codegen")]
pub fn generate_fdtd_cuda() -> String {
    use syn::parse_quote;

    // Define the kernel as a syn AST
    let kernel_fn: syn::ItemFn = parse_quote! {
        fn fdtd_wave_step(
            pressure: &[f32],
            pressure_prev: &mut [f32],
            c2: f32,
            damping: f32,
            pos: GridPos,
        ) {
            let p = pressure[pos.idx()];
            let p_prev = pressure_prev[pos.idx()];
            let laplacian = pos.north(pressure)
                + pos.south(pressure)
                + pos.east(pressure)
                + pos.west(pressure)
                - 4.0 * p;
            let p_new = 2.0 * p - p_prev + c2 * laplacian;
            pressure_prev[pos.idx()] = p_new * damping;
        }
    };

    // Configure the stencil
    let config = StencilConfig::new("fdtd_wave_step")
        .with_grid(Grid::Grid2D)
        .with_tile_size(16, 16)
        .with_halo(1);

    // Transpile to CUDA
    match transpile_stencil_kernel(&kernel_fn, &config) {
        Ok(cuda) => cuda,
        Err(e) => format!("// Transpilation error: {}", e),
    }
}

#[cfg(not(feature = "cuda-codegen"))]
pub fn generate_fdtd_cuda() -> String {
    "// CUDA codegen not enabled".to_string()
}

/// Get the handwritten CUDA source for comparison.
pub fn handwritten_fdtd_cuda() -> &'static str {
    include_str!("../shaders/fdtd_tile.cu")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fdtd_step() {
        // Create a simple 4x4 buffer (with 1-cell halo = 6x6 total)
        let buffer_width = 6;
        let mut pressure = vec![0.0f32; 36];
        let mut pressure_prev = vec![0.0f32; 36];

        // Set center cell to 1.0
        pressure[14] = 1.0; // (2,2) in 6x6 buffer

        // Run one FDTD step on the center cell
        fdtd_wave_step_cpu(&pressure, &mut pressure_prev, 0.25, 0.99, 14, buffer_width);

        // The Laplacian of a single spike is -4 (since neighbors are 0)
        // p_new = 2*1 - 0 + 0.25*(-4) = 2 - 1 = 1.0
        // With damping: 1.0 * 0.99 = 0.99
        assert!((pressure_prev[14] - 0.99).abs() < 0.01);
    }

    #[test]
    #[cfg(feature = "cuda-codegen")]
    fn test_generated_cuda_source() {
        let source = generate_fdtd_cuda();

        // Check that it looks like valid CUDA
        assert!(
            source.contains("extern \"C\" __global__"),
            "Should have CUDA kernel declaration"
        );
        assert!(source.contains("threadIdx.x"), "Should use thread index X");
        assert!(source.contains("threadIdx.y"), "Should use thread index Y");
        assert!(
            source.contains("buffer_width = 18"),
            "Should have correct buffer width (16 + 2*1)"
        );

        println!("Generated CUDA:\n{}", source);
    }

    #[test]
    #[cfg(feature = "cuda-codegen")]
    fn test_cuda_source_structure() {
        let source = generate_fdtd_cuda();

        // Verify key structural elements
        assert!(
            source.contains("if (lx >= 16 || ly >= 16) return;"),
            "Should have bounds check"
        );
        assert!(
            source.contains("float p ="),
            "Should have pressure variable"
        );

        // Verify stencil access pattern (using buffer_width constant)
        assert!(
            source.contains("- 18") || source.contains("- buffer_width"),
            "Should access north neighbor"
        );
        assert!(
            source.contains("+ 18") || source.contains("+ buffer_width"),
            "Should access south neighbor"
        );
    }

    #[test]
    #[cfg(not(feature = "cuda-codegen"))]
    fn test_grid_pos_neighbor_access() {
        // Create a 4x4 interior with 1-cell halo = 6x6 buffer
        // Layout (indices):
        //  0  1  2  3  4  5
        //  6  7  8  9 10 11
        // 12 13 14 15 16 17
        // 18 19 20 21 22 23
        // 24 25 26 27 28 29
        // 30 31 32 33 34 35

        let buffer: Vec<f32> = (0..36).map(|i| i as f32).collect();
        let pos = GridPos::new(14, 6); // Center at (2,2)

        assert_eq!(pos.idx(), 14);
        assert_eq!(pos.north(&buffer), 8.0); // 14 - 6 = 8
        assert_eq!(pos.south(&buffer), 20.0); // 14 + 6 = 20
        assert_eq!(pos.east(&buffer), 15.0); // 14 + 1 = 15
        assert_eq!(pos.west(&buffer), 13.0); // 14 - 1 = 13
    }

    #[test]
    #[cfg(not(feature = "cuda-codegen"))]
    fn test_grid_pos_from_local() {
        // 16x16 tile with halo=1 -> 18x18 buffer
        let pos = GridPos::from_local(0, 0, 16, 1);
        // index = (0+1)*18 + (0+1) = 19
        assert_eq!(pos.idx(), 19);

        let pos = GridPos::from_local(15, 15, 16, 1);
        // index = (15+1)*18 + (15+1) = 16*18 + 16 = 288 + 16 = 304
        assert_eq!(pos.idx(), 304);
    }

    #[test]
    #[cfg(not(feature = "cuda-codegen"))]
    fn test_grid_pos_at_offset() {
        let buffer: Vec<f32> = (0..36).map(|i| i as f32).collect();
        let pos = GridPos::new(14, 6);

        // at(0, 0) should be the same as idx
        assert_eq!(pos.at(&buffer, 0, 0), buffer[14]);

        // at(1, 0) should be east
        assert_eq!(pos.at(&buffer, 1, 0), buffer[15]);

        // at(-1, 0) should be west
        assert_eq!(pos.at(&buffer, -1, 0), buffer[13]);

        // at(0, -1) should be north
        assert_eq!(pos.at(&buffer, 0, -1), buffer[8]);

        // at(0, 1) should be south
        assert_eq!(pos.at(&buffer, 0, 1), buffer[20]);

        // Diagonal: northeast
        assert_eq!(pos.at(&buffer, 1, -1), buffer[9]);
    }

    #[test]
    #[cfg(not(feature = "cuda-codegen"))]
    fn test_dsl_matches_cpu() {
        // Verify that the DSL version produces the same results as the direct CPU version
        let buffer_width = 6;
        let tile_size = 4;
        let halo = 1;

        let mut pressure1 = vec![0.0f32; 36];
        let mut pressure1_prev = vec![0.0f32; 36];
        let mut pressure2 = vec![0.0f32; 36];
        let mut pressure2_prev = vec![0.0f32; 36];

        // Set center cell to 1.0 in both
        pressure1[14] = 1.0;
        pressure2[14] = 1.0;

        let c2 = 0.25f32;
        let damping = 0.99f32;

        // Run CPU version for all interior cells
        for ly in 0..tile_size {
            for lx in 0..tile_size {
                let idx = (ly + halo) * buffer_width + (lx + halo);
                fdtd_wave_step_cpu(
                    &pressure1,
                    &mut pressure1_prev,
                    c2,
                    damping,
                    idx,
                    buffer_width,
                );
            }
        }

        // Run DSL version
        fdtd_tile_step_dsl(
            &pressure2,
            &mut pressure2_prev,
            c2,
            damping,
            tile_size,
            halo,
        );

        // Compare results
        for i in 0..36 {
            assert!(
                (pressure1_prev[i] - pressure2_prev[i]).abs() < 1e-6,
                "Mismatch at index {}: CPU={}, DSL={}",
                i,
                pressure1_prev[i],
                pressure2_prev[i]
            );
        }
    }
}
