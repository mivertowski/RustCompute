//! Grid management for the acoustic wave simulation.
//!
//! Uses Structure of Arrays (SoA) layout for cache-friendly memory access
//! and SIMD-ready data organization.

use super::AcousticParams;
use rayon::prelude::*;

/// Cell type for drawing mode.
///
/// Determines how a cell interacts with the wave simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CellType {
    /// Normal cell - wave propagates freely.
    #[default]
    Normal,
    /// Absorber - fully absorbs waves (0% reflection).
    Absorber,
    /// Reflector - fully reflects waves (100% reflection, like a wall).
    Reflector,
}

/// The main simulation grid containing all cells.
///
/// Uses SoA (Structure of Arrays) memory layout for cache-friendly
/// sequential access and SIMD compatibility. Pressure arrays are
/// double-buffered for in-place FDTD time-stepping.
pub struct SimulationGrid {
    /// Grid width (number of columns).
    pub width: u32,
    /// Grid height (number of rows).
    pub height: u32,

    /// Current pressure values (width * height, row-major order).
    pressure: Vec<f32>,
    /// Previous pressure values for time-stepping.
    pressure_prev: Vec<f32>,

    /// Boundary cell flags (true if cell is on grid boundary).
    is_boundary: Vec<bool>,
    /// Reflection coefficients per cell (0 = absorb, 1 = reflect).
    reflection_coeff: Vec<f32>,
    /// Cell types for drawing mode (Normal, Absorber, Reflector).
    cell_types: Vec<CellType>,

    /// Acoustic simulation parameters.
    pub params: AcousticParams,
}

impl SimulationGrid {
    /// Create a new simulation grid.
    ///
    /// # Arguments
    /// * `width` - Number of columns
    /// * `height` - Number of rows
    /// * `params` - Acoustic simulation parameters
    pub fn new(width: u32, height: u32, params: AcousticParams) -> Self {
        let size = (width * height) as usize;

        let mut grid = Self {
            width,
            height,
            pressure: vec![0.0; size],
            pressure_prev: vec![0.0; size],
            is_boundary: vec![false; size],
            reflection_coeff: vec![1.0; size],
            cell_types: vec![CellType::Normal; size],
            params,
        };
        grid.initialize_boundaries();
        grid
    }

    /// Initialize boundary flags and reflection coefficients.
    fn initialize_boundaries(&mut self) {
        let width = self.width as usize;
        let height = self.height as usize;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let is_boundary = x == 0 || y == 0 || x == width - 1 || y == height - 1;
                self.is_boundary[idx] = is_boundary;
                self.reflection_coeff[idx] = if is_boundary { 0.95 } else { 1.0 };
            }
        }
    }

    /// Convert (x, y) coordinates to linear index.
    #[inline(always)]
    fn idx(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.width as usize) + (x as usize)
    }

    /// Perform one simulation step using FDTD scheme.
    ///
    /// Uses the 2D wave equation: ∂²p/∂t² = c²∇²p
    ///
    /// Discretized as:
    /// p[n+1] = 2*p[n] - p[n-1] + c²*(p_N + p_S + p_E + p_W - 4*p)
    pub fn step(&mut self) {
        let width = self.width as usize;
        let height = self.height as usize;
        let c2 = self.params.courant_number().powi(2);
        let damping = 1.0 - self.params.damping;

        // Process interior cells in parallel using rayon
        self.step_interior_parallel(width, height, c2, damping);

        // Process boundary cells with reflection (sequential, small workload)
        self.step_boundaries(c2, damping);

        // Swap buffers
        std::mem::swap(&mut self.pressure, &mut self.pressure_prev);
    }

    /// Process interior cells - uses parallel or sequential based on grid size.
    ///
    /// For small grids, sequential processing is faster due to lower overhead.
    /// For larger grids (>=512), parallel processing provides benefits.
    #[inline]
    fn step_interior_parallel(&mut self, width: usize, height: usize, c2: f32, damping: f32) {
        // Threshold: only use parallel for grids with enough work per thread
        // 512x512 = 262K cells, good balance of work vs overhead
        const PARALLEL_THRESHOLD: usize = 512;

        if width >= PARALLEL_THRESHOLD || height >= PARALLEL_THRESHOLD {
            self.step_interior_parallel_impl(width, height, c2, damping);
        } else {
            self.step_interior_sequential(width, height, c2, damping);
        }
    }

    /// Sequential implementation for interior cells (faster for small grids).
    #[inline]
    fn step_interior_sequential(&mut self, width: usize, height: usize, c2: f32, damping: f32) {
        for y in 1..height - 1 {
            let row_start = y * width;

            #[cfg(feature = "simd")]
            {
                Self::fdtd_row_simd_inline(
                    &self.pressure,
                    &mut self.pressure_prev[row_start..row_start + width],
                    width,
                    row_start,
                    c2,
                    damping,
                );
            }

            #[cfg(not(feature = "simd"))]
            {
                for x in 1..width - 1 {
                    let idx = row_start + x;

                    let p_curr = self.pressure[idx];
                    let p_prev = self.pressure_prev[idx];
                    let p_north = self.pressure[idx - width];
                    let p_south = self.pressure[idx + width];
                    let p_west = self.pressure[idx - 1];
                    let p_east = self.pressure[idx + 1];

                    let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
                    let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;

                    self.pressure_prev[idx] = p_new * damping;
                }
            }
        }
    }

    /// Parallel implementation for interior cells using rayon.
    ///
    /// Each row is processed independently in parallel.
    /// Uses SIMD for intra-row processing when available.
    #[inline]
    fn step_interior_parallel_impl(&mut self, width: usize, height: usize, c2: f32, damping: f32) {
        // Shared reference to pressure (read-only)
        let pressure = &self.pressure;

        // Split pressure_prev into rows for parallel mutable access
        // We skip row 0 (boundary) and process interior rows only
        let interior_start = width; // Skip first row
        let interior_end = (height - 1) * width; // Skip last row

        // Get mutable slice of interior rows in pressure_prev
        let pressure_prev_interior = &mut self.pressure_prev[interior_start..interior_end];

        // Process rows in parallel using par_chunks_mut
        pressure_prev_interior
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row_offset, prev_row)| {
                let y = row_offset + 1; // Actual y coordinate (offset by 1 for skipped boundary)
                let row_start = y * width;

                #[cfg(feature = "simd")]
                {
                    // SIMD version - process 8 cells at a time
                    Self::fdtd_row_simd_inline(pressure, prev_row, width, row_start, c2, damping);
                }

                #[cfg(not(feature = "simd"))]
                {
                    // Scalar version
                    for x in 1..width - 1 {
                        let idx = row_start + x;
                        let local_idx = x; // Index within prev_row

                        let p_curr = pressure[idx];
                        let p_prev = prev_row[local_idx];
                        let p_north = pressure[idx - width];
                        let p_south = pressure[idx + width];
                        let p_west = pressure[idx - 1];
                        let p_east = pressure[idx + 1];

                        let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
                        let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;

                        prev_row[local_idx] = p_new * damping;
                    }
                }
            });
    }

    /// SIMD implementation for a single row (inlined for parallel use).
    #[cfg(feature = "simd")]
    #[inline]
    fn fdtd_row_simd_inline(
        pressure: &[f32],
        prev_row: &mut [f32],
        width: usize,
        row_start: usize,
        c2: f32,
        damping: f32,
    ) {
        use std::simd::f32x8;

        let c2_vec = f32x8::splat(c2);
        let damping_vec = f32x8::splat(damping);
        let four = f32x8::splat(4.0);
        let two = f32x8::splat(2.0);

        let mut x = 1;
        while x + 8 <= width - 1 {
            let idx = row_start + x;
            let local_idx = x;

            // Load vectors
            let p_curr = f32x8::from_slice(&pressure[idx..idx + 8]);
            let p_prev = f32x8::from_slice(&prev_row[local_idx..local_idx + 8]);
            let p_north = f32x8::from_slice(&pressure[idx - width..idx - width + 8]);
            let p_south = f32x8::from_slice(&pressure[idx + width..idx + width + 8]);
            let p_west = f32x8::from_slice(&pressure[idx - 1..idx - 1 + 8]);
            let p_east = f32x8::from_slice(&pressure[idx + 1..idx + 1 + 8]);

            // FDTD computation
            let laplacian = p_north + p_south + p_east + p_west - four * p_curr;
            let p_new = two * p_curr - p_prev + c2_vec * laplacian;
            let p_damped = p_new * damping_vec;

            // Store result
            prev_row[local_idx..local_idx + 8].copy_from_slice(&p_damped.to_array());

            x += 8;
        }

        // Scalar fallback for remaining cells
        while x < width - 1 {
            let idx = row_start + x;
            let local_idx = x;

            let p_curr = pressure[idx];
            let p_prev = prev_row[local_idx];
            let p_north = pressure[idx - width];
            let p_south = pressure[idx + width];
            let p_west = pressure[idx - 1];
            let p_east = pressure[idx + 1];

            let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
            let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;

            prev_row[local_idx] = p_new * damping;
            x += 1;
        }
    }

    /// Process boundary cells with reflection boundary conditions.
    fn step_boundaries(&mut self, c2: f32, damping: f32) {
        let width = self.width as usize;
        let height = self.height as usize;

        // Top and bottom rows
        for x in 0..width {
            // Top row (y = 0)
            let idx = x;
            let p_curr = self.pressure[idx];
            let p_prev = self.pressure_prev[idx];
            let refl = self.reflection_coeff[idx];

            let p_south = self.pressure[idx + width];
            let p_west = if x > 0 {
                self.pressure[idx - 1]
            } else {
                p_curr * refl
            };
            let p_east = if x < width - 1 {
                self.pressure[idx + 1]
            } else {
                p_curr * refl
            };
            let p_north = p_curr * refl; // Reflect at boundary

            let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
            let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;
            self.pressure_prev[idx] = p_new * damping * refl;

            // Bottom row (y = height - 1)
            let idx = (height - 1) * width + x;
            let p_curr = self.pressure[idx];
            let p_prev = self.pressure_prev[idx];
            let refl = self.reflection_coeff[idx];

            let p_north = self.pressure[idx - width];
            let p_west = if x > 0 {
                self.pressure[idx - 1]
            } else {
                p_curr * refl
            };
            let p_east = if x < width - 1 {
                self.pressure[idx + 1]
            } else {
                p_curr * refl
            };
            let p_south = p_curr * refl; // Reflect at boundary

            let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
            let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;
            self.pressure_prev[idx] = p_new * damping * refl;
        }

        // Left and right columns (excluding corners already processed)
        for y in 1..height - 1 {
            // Left column (x = 0)
            let idx = y * width;
            let p_curr = self.pressure[idx];
            let p_prev = self.pressure_prev[idx];
            let refl = self.reflection_coeff[idx];

            let p_north = self.pressure[idx - width];
            let p_south = self.pressure[idx + width];
            let p_east = self.pressure[idx + 1];
            let p_west = p_curr * refl; // Reflect at boundary

            let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
            let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;
            self.pressure_prev[idx] = p_new * damping * refl;

            // Right column (x = width - 1)
            let idx = y * width + width - 1;
            let p_curr = self.pressure[idx];
            let p_prev = self.pressure_prev[idx];
            let refl = self.reflection_coeff[idx];

            let p_north = self.pressure[idx - width];
            let p_south = self.pressure[idx + width];
            let p_west = self.pressure[idx - 1];
            let p_east = p_curr * refl; // Reflect at boundary

            let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
            let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;
            self.pressure_prev[idx] = p_new * damping * refl;
        }
    }

    /// Inject an impulse at the given grid position.
    pub fn inject_impulse(&mut self, x: u32, y: u32, amplitude: f32) {
        if x < self.width && y < self.height {
            let idx = self.idx(x, y);
            self.pressure[idx] += amplitude;
        }
    }

    /// Get the pressure grid as a 2D vector for visualization.
    ///
    /// Returns a Vec of rows, where each row is a Vec of pressure values.
    pub fn get_pressure_grid(&self) -> Vec<Vec<f32>> {
        let width = self.width as usize;
        let height = self.height as usize;
        let mut grid = vec![vec![0.0; width]; height];

        for (y, row) in grid.iter_mut().enumerate().take(height) {
            for (x, cell) in row.iter_mut().enumerate().take(width) {
                *cell = self.pressure[y * width + x];
            }
        }
        grid
    }

    /// Get the pressure buffer as a flat slice (row-major order).
    ///
    /// This is more efficient than `get_pressure_grid()` for bulk operations.
    #[inline]
    pub fn pressure_slice(&self) -> &[f32] {
        &self.pressure
    }

    /// Get the maximum absolute pressure in the grid.
    pub fn max_pressure(&self) -> f32 {
        self.pressure.iter().map(|p| p.abs()).fold(0.0, f32::max)
    }

    /// Get total energy in the system (sum of squared pressures).
    pub fn total_energy(&self) -> f32 {
        self.pressure.iter().map(|p| p * p).sum()
    }

    /// Reset all cells to initial state.
    pub fn reset(&mut self) {
        self.pressure.fill(0.0);
        self.pressure_prev.fill(0.0);
    }

    /// Resize the grid.
    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        let size = (new_width * new_height) as usize;
        self.width = new_width;
        self.height = new_height;
        self.pressure = vec![0.0; size];
        self.pressure_prev = vec![0.0; size];
        self.is_boundary = vec![false; size];
        self.reflection_coeff = vec![1.0; size];
        self.cell_types = vec![CellType::Normal; size];
        self.initialize_boundaries();
    }

    /// Update acoustic parameters.
    pub fn set_params(&mut self, params: AcousticParams) {
        self.params = params;
    }

    /// Update speed of sound.
    pub fn set_speed_of_sound(&mut self, speed: f32) {
        self.params.set_speed_of_sound(speed);
    }

    /// Update cell size in meters.
    pub fn set_cell_size(&mut self, size: f32) {
        self.params.set_cell_size(size);
    }

    /// Get the number of cells in the grid.
    pub fn cell_count(&self) -> usize {
        self.pressure.len()
    }

    /// Get pressure at a specific cell (for compatibility with old API).
    pub fn get_pressure(&self, x: u32, y: u32) -> Option<f32> {
        if x < self.width && y < self.height {
            Some(self.pressure[self.idx(x, y)])
        } else {
            None
        }
    }

    /// Check if a cell is a boundary.
    pub fn is_boundary(&self, x: u32, y: u32) -> bool {
        if x < self.width && y < self.height {
            self.is_boundary[self.idx(x, y)]
        } else {
            false
        }
    }

    /// Set the cell type at the given position.
    ///
    /// This affects how the cell interacts with waves:
    /// - Normal: waves propagate freely
    /// - Absorber: waves are absorbed (0% reflection)
    /// - Reflector: waves are reflected (100% reflection)
    pub fn set_cell_type(&mut self, x: u32, y: u32, cell_type: CellType) {
        if x < self.width && y < self.height {
            let idx = self.idx(x, y);
            self.cell_types[idx] = cell_type;

            // Update reflection coefficient based on cell type
            self.reflection_coeff[idx] = match cell_type {
                CellType::Normal => {
                    // Normal cells at boundary get slight absorption, interior gets full reflection
                    if self.is_boundary[idx] {
                        0.95
                    } else {
                        1.0
                    }
                }
                CellType::Absorber => 0.0,  // Full absorption
                CellType::Reflector => 1.0, // Full reflection
            };

            // Mark reflectors/absorbers as boundary-like for FDTD calculation
            // (they need special handling in the step function)
            if cell_type != CellType::Normal {
                self.is_boundary[idx] = true;
            } else {
                // Restore original boundary status
                let is_edge = x == 0 || y == 0 || x == self.width - 1 || y == self.height - 1;
                self.is_boundary[idx] = is_edge;
            }
        }
    }

    /// Get the cell type at the given position.
    pub fn get_cell_type(&self, x: u32, y: u32) -> Option<CellType> {
        if x < self.width && y < self.height {
            Some(self.cell_types[self.idx(x, y)])
        } else {
            None
        }
    }

    /// Get all cell types as a flat slice (row-major order).
    pub fn cell_types_slice(&self) -> &[CellType] {
        &self.cell_types
    }

    /// Clear all user-defined cell types (reset to Normal).
    ///
    /// This keeps the grid boundaries intact.
    pub fn clear_cell_types(&mut self) {
        for i in 0..self.cell_types.len() {
            self.cell_types[i] = CellType::Normal;
        }
        // Re-initialize boundary reflection coefficients
        self.initialize_boundaries();
    }

    /// Get mutable access to pressure buffers for educational mode processing.
    ///
    /// Returns (pressure, pressure_prev, width, height, c2, damping)
    pub fn get_buffers_mut(&mut self) -> (&mut Vec<f32>, &mut Vec<f32>, usize, usize, f32, f32) {
        let width = self.width as usize;
        let height = self.height as usize;
        let c2 = self.params.courant_number().powi(2);
        let damping = 1.0 - self.params.damping;
        (
            &mut self.pressure,
            &mut self.pressure_prev,
            width,
            height,
            c2,
            damping,
        )
    }

    /// Swap the pressure buffers (called when educational step completes).
    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.pressure, &mut self.pressure_prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let params = AcousticParams::new(343.0, 1.0);
        let grid = SimulationGrid::new(8, 8, params);

        assert_eq!(grid.width, 8);
        assert_eq!(grid.height, 8);
        assert_eq!(grid.cell_count(), 64);
    }

    #[test]
    fn test_boundary_cells() {
        let params = AcousticParams::new(343.0, 1.0);
        let grid = SimulationGrid::new(4, 4, params);

        // Corner cells should be boundaries
        assert!(grid.is_boundary(0, 0));
        assert!(grid.is_boundary(3, 3));

        // Edge cells should be boundaries
        assert!(grid.is_boundary(1, 0));
        assert!(grid.is_boundary(0, 1));

        // Interior cells should not be boundaries
        assert!(!grid.is_boundary(1, 1));
        assert!(!grid.is_boundary(2, 2));
    }

    #[test]
    fn test_impulse_injection() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = SimulationGrid::new(8, 8, params);

        grid.inject_impulse(4, 4, 1.0);

        assert_eq!(grid.get_pressure(4, 4), Some(1.0));
    }

    #[test]
    fn test_wave_propagation() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = SimulationGrid::new(8, 8, params);

        // Inject impulse at center
        grid.inject_impulse(4, 4, 1.0);

        // Run several steps
        for _ in 0..10 {
            grid.step();
        }

        // Wave should have propagated to neighbors
        let neighbor_pressure = grid.get_pressure(4, 5).unwrap();
        assert!(
            neighbor_pressure.abs() > 0.0,
            "Wave should have propagated to neighbor"
        );
    }

    #[test]
    fn test_pressure_grid() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = SimulationGrid::new(4, 4, params);

        grid.inject_impulse(2, 2, 0.5);

        let pressure_grid = grid.get_pressure_grid();
        assert_eq!(pressure_grid.len(), 4);
        assert_eq!(pressure_grid[0].len(), 4);
        assert_eq!(pressure_grid[2][2], 0.5);
    }

    #[test]
    fn test_reset() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = SimulationGrid::new(4, 4, params);

        grid.inject_impulse(2, 2, 1.0);
        grid.step();

        grid.reset();

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(grid.get_pressure(x, y), Some(0.0));
            }
        }
    }

    #[test]
    fn test_resize() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = SimulationGrid::new(4, 4, params);

        grid.resize(8, 6);

        assert_eq!(grid.width, 8);
        assert_eq!(grid.height, 6);
        assert_eq!(grid.cell_count(), 48);
    }

    #[test]
    fn test_energy_conservation() {
        let params = AcousticParams::new(343.0, 1.0).with_damping(0.0);
        let mut grid = SimulationGrid::new(32, 32, params);

        // Set boundaries to full reflection for this test
        for i in 0..grid.reflection_coeff.len() {
            grid.reflection_coeff[i] = 1.0;
        }

        grid.inject_impulse(16, 16, 1.0);
        let initial_energy = grid.total_energy();

        // Run simulation (short duration to avoid numerical issues)
        for _ in 0..20 {
            grid.step();
        }

        let final_energy = grid.total_energy();

        // With no damping and full reflection, energy should be well conserved
        // (allowing 20% loss for numerical dissipation)
        assert!(
            final_energy > 0.8 * initial_energy,
            "Energy should be roughly conserved: initial={}, final={}",
            initial_energy,
            final_energy
        );
    }

    #[test]
    fn test_pressure_slice() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = SimulationGrid::new(4, 4, params);

        grid.inject_impulse(2, 2, 0.5);

        let slice = grid.pressure_slice();
        assert_eq!(slice.len(), 16);
        // Cell (2, 2) is at index 2*4 + 2 = 10
        assert_eq!(slice[10], 0.5);
    }
}
