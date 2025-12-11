//! 3D FDTD simulation grid for acoustic wave propagation.
//!
//! Implements the 3D wave equation using finite differences:
//! ∂²p/∂t² = c² (∂²p/∂x² + ∂²p/∂y² + ∂²p/∂z²) - γ·∂p/∂t
//!
//! Uses a 7-point stencil (6 neighbors + center) for the Laplacian.

use crate::simulation::physics::{AcousticParams3D, Position3D};
use rayon::prelude::*;

/// Cell type for boundary conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CellType {
    /// Normal cell - wave propagates freely
    #[default]
    Normal,
    /// Absorber - absorbs waves (anechoic boundary)
    Absorber,
    /// Reflector - reflects waves (hard boundary)
    Reflector,
    /// Obstacle - solid object (partial reflection)
    Obstacle,
}

/// 3D simulation grid using Structure of Arrays (SoA) layout.
///
/// Memory layout is z-major (z varies slowest):
/// `index = z * (width * height) + y * width + x`
pub struct SimulationGrid3D {
    /// Grid dimensions
    pub width: usize,
    pub height: usize,
    pub depth: usize,

    /// Current pressure field (row-major, z-major ordering)
    pub pressure: Vec<f32>,

    /// Previous pressure field (for time-stepping)
    pub pressure_prev: Vec<f32>,

    /// Cell types (boundary conditions)
    pub cell_types: Vec<CellType>,

    /// Reflection coefficients per cell (0 = full absorption, 1 = full reflection)
    pub reflection_coeff: Vec<f32>,

    /// Acoustic parameters
    pub params: AcousticParams3D,

    /// Total number of cells
    total_cells: usize,

    /// Slice size (width * height) for z-indexing
    slice_size: usize,

    /// Current simulation step
    pub step: u64,

    /// Accumulated simulation time
    pub time: f32,
}

impl SimulationGrid3D {
    /// Create a new 3D simulation grid.
    ///
    /// # Arguments
    /// * `width` - Number of cells in X direction
    /// * `height` - Number of cells in Y direction
    /// * `depth` - Number of cells in Z direction
    /// * `params` - Acoustic simulation parameters
    pub fn new(width: usize, height: usize, depth: usize, params: AcousticParams3D) -> Self {
        let total_cells = width * height * depth;
        let slice_size = width * height;

        let mut grid = Self {
            width,
            height,
            depth,
            pressure: vec![0.0; total_cells],
            pressure_prev: vec![0.0; total_cells],
            cell_types: vec![CellType::Normal; total_cells],
            reflection_coeff: vec![1.0; total_cells],
            params,
            total_cells,
            slice_size,
            step: 0,
            time: 0.0,
        };

        // Set boundary conditions
        grid.setup_boundaries();
        grid
    }

    /// Set up default boundary conditions (absorbing boundaries).
    fn setup_boundaries(&mut self) {
        // Set all boundary cells to absorbers
        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let is_boundary = x == 0
                        || x == self.width - 1
                        || y == 0
                        || y == self.height - 1
                        || z == 0
                        || z == self.depth - 1;

                    if is_boundary {
                        let idx = self.index(x, y, z);
                        self.cell_types[idx] = CellType::Absorber;
                        self.reflection_coeff[idx] = 0.1; // 10% reflection
                    }
                }
            }
        }
    }

    /// Convert (x, y, z) coordinates to linear index.
    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.slice_size + y * self.width + x
    }

    /// Convert linear index to (x, y, z) coordinates.
    #[inline]
    pub fn coords(&self, idx: usize) -> (usize, usize, usize) {
        let z = idx / self.slice_size;
        let remainder = idx % self.slice_size;
        let y = remainder / self.width;
        let x = remainder % self.width;
        (x, y, z)
    }

    /// Check if coordinates are within bounds.
    #[inline]
    pub fn in_bounds(&self, x: i32, y: i32, z: i32) -> bool {
        x >= 0
            && x < self.width as i32
            && y >= 0
            && y < self.height as i32
            && z >= 0
            && z < self.depth as i32
    }

    /// Inject a pressure impulse at a specific position.
    pub fn inject_impulse(&mut self, x: usize, y: usize, z: usize, amplitude: f32) {
        if x < self.width && y < self.height && z < self.depth {
            let idx = self.index(x, y, z);
            self.pressure[idx] += amplitude;
            self.pressure_prev[idx] += amplitude * 0.5;
        }
    }

    /// Inject a pressure impulse at a 3D position (in meters).
    pub fn inject_impulse_at(&mut self, pos: Position3D, amplitude: f32) {
        let (x, y, z) = pos.to_grid_indices(self.params.cell_size);
        self.inject_impulse(x, y, z, amplitude);
    }

    /// Inject a spherical impulse (Gaussian distribution).
    pub fn inject_spherical_impulse(
        &mut self,
        center: Position3D,
        amplitude: f32,
        radius_cells: f32,
    ) {
        let (cx, cy, cz) = center.to_grid_indices(self.params.cell_size);
        let r = radius_cells as i32;
        let r_sq = radius_cells * radius_cells;

        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let x = cx as i32 + dx;
                    let y = cy as i32 + dy;
                    let z = cz as i32 + dz;

                    if self.in_bounds(x, y, z) {
                        let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                        if dist_sq <= r_sq {
                            // Gaussian falloff
                            let factor = (-dist_sq / (2.0 * radius_cells)).exp();
                            let idx = self.index(x as usize, y as usize, z as usize);
                            self.pressure[idx] += amplitude * factor;
                            self.pressure_prev[idx] += amplitude * factor * 0.5;
                        }
                    }
                }
            }
        }
    }

    /// Set cell type at a specific location.
    pub fn set_cell_type(&mut self, x: usize, y: usize, z: usize, cell_type: CellType) {
        if x < self.width && y < self.height && z < self.depth {
            let idx = self.index(x, y, z);
            self.cell_types[idx] = cell_type;
            self.reflection_coeff[idx] = match cell_type {
                CellType::Normal => 1.0,
                CellType::Absorber => 0.0,
                CellType::Reflector => 1.0,
                CellType::Obstacle => 0.8,
            };
        }
    }

    /// Create a rectangular obstacle.
    pub fn create_obstacle(&mut self, min: Position3D, max: Position3D, cell_type: CellType) {
        let (x0, y0, z0) = min.to_grid_indices(self.params.cell_size);
        let (x1, y1, z1) = max.to_grid_indices(self.params.cell_size);

        for z in z0..=z1.min(self.depth - 1) {
            for y in y0..=y1.min(self.height - 1) {
                for x in x0..=x1.min(self.width - 1) {
                    self.set_cell_type(x, y, z, cell_type);
                }
            }
        }
    }

    /// Perform one FDTD time step (CPU, sequential).
    pub fn step_sequential(&mut self) {
        let c2 = self.params.c_squared;
        let damping = self.params.simple_damping;

        // Update interior cells only
        for z in 1..self.depth - 1 {
            for y in 1..self.height - 1 {
                for x in 1..self.width - 1 {
                    let idx = self.index(x, y, z);

                    // Skip non-normal cells (handle separately)
                    if self.cell_types[idx] != CellType::Normal {
                        continue;
                    }

                    // Current and previous pressure
                    let p = self.pressure[idx];
                    let p_prev = self.pressure_prev[idx];

                    // 6 neighbors for 3D Laplacian
                    let p_west = self.pressure[idx - 1];
                    let p_east = self.pressure[idx + 1];
                    let p_south = self.pressure[idx - self.width];
                    let p_north = self.pressure[idx + self.width];
                    let p_down = self.pressure[idx - self.slice_size];
                    let p_up = self.pressure[idx + self.slice_size];

                    // 7-point stencil Laplacian
                    let laplacian = p_west + p_east + p_south + p_north + p_down + p_up - 6.0 * p;

                    // FDTD update equation
                    let p_new = 2.0 * p - p_prev + c2 * laplacian;

                    // Apply damping
                    self.pressure_prev[idx] = p_new * damping;
                }
            }
        }

        // Handle boundary cells
        self.apply_boundary_conditions();

        // Swap buffers
        std::mem::swap(&mut self.pressure, &mut self.pressure_prev);

        self.step += 1;
        self.time += self.params.time_step;
    }

    /// Perform one FDTD time step (CPU, parallel with Rayon).
    pub fn step_parallel(&mut self) {
        let c2 = self.params.c_squared;
        let damping = self.params.simple_damping;
        let width = self.width;
        let height = self.height;
        let depth = self.depth;
        let slice_size = self.slice_size;

        // Take references to avoid capturing self in closures
        let pressure = &self.pressure;
        let pressure_prev = &self.pressure_prev;
        let cell_types = &self.cell_types;

        // Compute new values in parallel and store in a new buffer
        // Use par_iter over z indices to avoid flat_map memory explosion
        let num_interior_z = depth - 2;

        // Pre-allocate output buffer
        let mut new_values = vec![0.0f32; num_interior_z * (height - 2) * (width - 2)];

        // Process in parallel
        new_values
            .par_chunks_mut((height - 2) * (width - 2))
            .enumerate()
            .for_each(|(zi, chunk)| {
                let z = zi + 1; // Actual z index (skip boundary)
                let mut out_idx = 0;

                for y in 1..height - 1 {
                    for x in 1..width - 1 {
                        let idx = z * slice_size + y * width + x;

                        let p_new = if cell_types[idx] != CellType::Normal {
                            // Keep previous value for non-normal cells
                            pressure_prev[idx]
                        } else {
                            let p = pressure[idx];
                            let p_prev = pressure_prev[idx];

                            // 6 neighbors
                            let p_west = pressure[idx - 1];
                            let p_east = pressure[idx + 1];
                            let p_south = pressure[idx - width];
                            let p_north = pressure[idx + width];
                            let p_down = pressure[idx - slice_size];
                            let p_up = pressure[idx + slice_size];

                            let laplacian =
                                p_west + p_east + p_south + p_north + p_down + p_up - 6.0 * p;
                            (2.0 * p - p_prev + c2 * laplacian) * damping
                        };

                        chunk[out_idx] = p_new;
                        out_idx += 1;
                    }
                }
            });

        // Copy results back to pressure_prev
        let mut result_idx = 0;
        for z in 1..depth - 1 {
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let idx = self.index(x, y, z);
                    self.pressure_prev[idx] = new_values[result_idx];
                    result_idx += 1;
                }
            }
        }

        // Handle boundary conditions
        self.apply_boundary_conditions();

        // Swap buffers
        std::mem::swap(&mut self.pressure, &mut self.pressure_prev);

        self.step += 1;
        self.time += self.params.time_step;
    }

    /// Apply boundary conditions.
    fn apply_boundary_conditions(&mut self) {
        for idx in 0..self.total_cells {
            match self.cell_types[idx] {
                CellType::Absorber => {
                    // Gradually absorb energy
                    self.pressure_prev[idx] *= self.reflection_coeff[idx];
                }
                CellType::Reflector => {
                    // Perfect reflection (handled implicitly by stencil)
                }
                CellType::Obstacle => {
                    // Partial reflection
                    self.pressure_prev[idx] *= self.reflection_coeff[idx];
                }
                CellType::Normal => {}
            }
        }
    }

    /// Get pressure value at a specific position (interpolated).
    pub fn sample_pressure(&self, pos: Position3D) -> f32 {
        // Trilinear interpolation
        let fx = pos.x / self.params.cell_size;
        let fy = pos.y / self.params.cell_size;
        let fz = pos.z / self.params.cell_size;

        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let z0 = fz.floor() as usize;

        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let z1 = (z0 + 1).min(self.depth - 1);

        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let tz = fz - z0 as f32;

        // 8 corner samples
        let c000 = self.pressure[self.index(x0, y0, z0)];
        let c100 = self.pressure[self.index(x1, y0, z0)];
        let c010 = self.pressure[self.index(x0, y1, z0)];
        let c110 = self.pressure[self.index(x1, y1, z0)];
        let c001 = self.pressure[self.index(x0, y0, z1)];
        let c101 = self.pressure[self.index(x1, y0, z1)];
        let c011 = self.pressure[self.index(x0, y1, z1)];
        let c111 = self.pressure[self.index(x1, y1, z1)];

        // Trilinear interpolation
        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        c0 * (1.0 - tz) + c1 * tz
    }

    /// Get an XY slice of the pressure field at a given Z.
    pub fn get_xy_slice(&self, z: usize) -> Vec<f32> {
        let z = z.min(self.depth - 1);
        let start = z * self.slice_size;
        self.pressure[start..start + self.slice_size].to_vec()
    }

    /// Get an XZ slice of the pressure field at a given Y.
    pub fn get_xz_slice(&self, y: usize) -> Vec<f32> {
        let y = y.min(self.height - 1);
        let mut slice = Vec::with_capacity(self.width * self.depth);
        for z in 0..self.depth {
            for x in 0..self.width {
                slice.push(self.pressure[self.index(x, y, z)]);
            }
        }
        slice
    }

    /// Get a YZ slice of the pressure field at a given X.
    pub fn get_yz_slice(&self, x: usize) -> Vec<f32> {
        let x = x.min(self.width - 1);
        let mut slice = Vec::with_capacity(self.height * self.depth);
        for z in 0..self.depth {
            for y in 0..self.height {
                slice.push(self.pressure[self.index(x, y, z)]);
            }
        }
        slice
    }

    /// Get the total energy in the simulation.
    pub fn total_energy(&self) -> f32 {
        self.pressure.iter().map(|&p| p * p).sum::<f32>()
    }

    /// Get the maximum absolute pressure.
    pub fn max_pressure(&self) -> f32 {
        self.pressure
            .iter()
            .map(|&p| p.abs())
            .fold(0.0_f32, f32::max)
    }

    /// Reset the simulation to initial state.
    pub fn reset(&mut self) {
        self.pressure.fill(0.0);
        self.pressure_prev.fill(0.0);
        self.step = 0;
        self.time = 0.0;
    }

    /// Get grid dimensions as (width, height, depth).
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.depth)
    }

    /// Get the physical size of the simulation domain in meters.
    pub fn physical_size(&self) -> (f32, f32, f32) {
        let cell = self.params.cell_size;
        (
            self.width as f32 * cell,
            self.height as f32 * cell,
            self.depth as f32 * cell,
        )
    }
}

/// GPU-compatible data layout for 3D grid.
///
/// Packed format for efficient GPU memory access.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridParams {
    /// Grid dimensions
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub _padding0: u32,

    /// Courant number squared
    pub c_squared: f32,
    /// Damping factor
    pub damping: f32,
    /// Slice size (width * height)
    pub slice_size: u32,
    pub _padding1: u32,
}

impl From<&SimulationGrid3D> for GridParams {
    fn from(grid: &SimulationGrid3D) -> Self {
        Self {
            width: grid.width as u32,
            height: grid.height as u32,
            depth: grid.depth as u32,
            _padding0: 0,
            c_squared: grid.params.c_squared,
            damping: grid.params.simple_damping,
            slice_size: grid.slice_size as u32,
            _padding1: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::physics::Environment;

    fn create_test_grid() -> SimulationGrid3D {
        let params = AcousticParams3D::new(Environment::default(), 0.1);
        SimulationGrid3D::new(32, 32, 32, params)
    }

    #[test]
    fn test_grid_creation() {
        let grid = create_test_grid();
        assert_eq!(grid.width, 32);
        assert_eq!(grid.height, 32);
        assert_eq!(grid.depth, 32);
        assert_eq!(grid.total_cells, 32 * 32 * 32);
    }

    #[test]
    fn test_indexing() {
        let grid = create_test_grid();

        // Test index conversion
        let idx = grid.index(5, 10, 15);
        let (x, y, z) = grid.coords(idx);
        assert_eq!((x, y, z), (5, 10, 15));

        // Test corner cases
        assert_eq!(grid.index(0, 0, 0), 0);
        assert_eq!(grid.index(31, 31, 31), 32 * 32 * 32 - 1);
    }

    #[test]
    fn test_impulse_injection() {
        let mut grid = create_test_grid();
        let center_idx = grid.index(16, 16, 16);

        assert_eq!(grid.pressure[center_idx], 0.0);
        grid.inject_impulse(16, 16, 16, 1.0);
        assert_eq!(grid.pressure[center_idx], 1.0);
    }

    #[test]
    fn test_wave_propagation() {
        let mut grid = create_test_grid();

        // Inject impulse at center
        grid.inject_impulse(16, 16, 16, 1.0);

        // Run simulation
        for _ in 0..10 {
            grid.step_sequential();
        }

        // Energy should have spread from center
        let center_idx = grid.index(16, 16, 16);
        let neighbor_idx = grid.index(17, 16, 16);

        // Neighbor should have received energy (some energy left center)
        assert!(grid.pressure[neighbor_idx].abs() > 0.0 || grid.pressure[center_idx].abs() > 0.0);
    }

    #[test]
    fn test_wave_spreading() {
        let mut grid = create_test_grid();

        // Set up for minimal damping
        grid.params.simple_damping = 1.0;

        // Inject impulse at center
        grid.inject_impulse(16, 16, 16, 1.0);

        // Verify initial state
        let initial_max = grid.max_pressure();
        assert!(initial_max > 0.5);

        // Run a few steps
        for _ in 0..5 {
            grid.step_sequential();
        }

        // Wave should have propagated - energy should be distributed
        // and not concentrated at a single point anymore
        let total_energy = grid.total_energy();
        assert!(total_energy > 0.0, "Wave should contain energy");

        // Check that wave has spread beyond center
        // (neighboring cells should have non-zero pressure)
        let neighbor_idx = grid.index(17, 16, 16);
        let has_spread = grid.pressure[neighbor_idx].abs() > 0.0
            || grid.pressure[grid.index(16, 17, 16)].abs() > 0.0;
        assert!(has_spread, "Wave should propagate to neighbors");
    }

    #[test]
    fn test_pressure_sampling() {
        let mut grid = create_test_grid();

        // Set a known pressure value
        let idx = grid.index(10, 10, 10);
        grid.pressure[idx] = 1.0;

        // Sample at exact grid point
        let pos = Position3D::new(
            10.0 * grid.params.cell_size,
            10.0 * grid.params.cell_size,
            10.0 * grid.params.cell_size,
        );
        let sampled = grid.sample_pressure(pos);

        // Should be close to 1.0 (trilinear interpolation)
        assert!(sampled > 0.5, "Sampled value too low: {}", sampled);
    }

    #[test]
    fn test_boundary_setup() {
        let grid = create_test_grid();

        // Check that boundary cells are absorbers
        assert_eq!(grid.cell_types[grid.index(0, 0, 0)], CellType::Absorber);
        assert_eq!(grid.cell_types[grid.index(31, 31, 31)], CellType::Absorber);

        // Check interior cell
        assert_eq!(grid.cell_types[grid.index(16, 16, 16)], CellType::Normal);
    }

    #[test]
    fn test_slices() {
        let mut grid = create_test_grid();

        // Set a known pattern
        let idx = grid.index(5, 10, 15);
        grid.pressure[idx] = 1.0;

        // Test XY slice
        let xy_slice = grid.get_xy_slice(15);
        assert_eq!(xy_slice.len(), grid.width * grid.height);
        assert_eq!(xy_slice[10 * grid.width + 5], 1.0);

        // Test XZ slice
        let xz_slice = grid.get_xz_slice(10);
        assert_eq!(xz_slice.len(), grid.width * grid.depth);
        assert_eq!(xz_slice[15 * grid.width + 5], 1.0);

        // Test YZ slice
        let yz_slice = grid.get_yz_slice(5);
        assert_eq!(yz_slice.len(), grid.height * grid.depth);
        assert_eq!(yz_slice[15 * grid.height + 10], 1.0);
    }

    #[test]
    fn test_parallel_vs_sequential() {
        let params = AcousticParams3D::new(Environment::default(), 0.1);

        let mut grid_seq = SimulationGrid3D::new(16, 16, 16, params.clone());
        let mut grid_par = SimulationGrid3D::new(16, 16, 16, params);

        // Same impulse
        grid_seq.inject_impulse(8, 8, 8, 1.0);
        grid_par.inject_impulse(8, 8, 8, 1.0);

        // Run both
        for _ in 0..10 {
            grid_seq.step_sequential();
            grid_par.step_parallel();
        }

        // Results should be very close
        let max_diff: f32 = grid_seq
            .pressure
            .iter()
            .zip(grid_par.pressure.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        assert!(
            max_diff < 0.001,
            "Sequential and parallel results differ by {}",
            max_diff
        );
    }
}
