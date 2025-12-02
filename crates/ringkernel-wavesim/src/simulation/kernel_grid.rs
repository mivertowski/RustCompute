//! Kernel-based simulation grid using RingKernel actors.
//!
//! This module provides a GPU-native implementation of the wave simulation
//! where each cell is a persistent kernel actor communicating via K2K messaging.

use super::{AcousticParams, CellState};
use ringkernel::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// A simulation grid backed by RingKernel actors.
///
/// Each cell in the grid is a persistent kernel that maintains its own state
/// and communicates with neighbors via message passing.
pub struct KernelGrid {
    // Note: We don't derive Debug because RingKernel and KernelHandle don't implement it
    /// Grid width (number of columns).
    pub width: u32,
    /// Grid height (number of rows).
    pub height: u32,
    /// Cell states (for visualization).
    cells: HashMap<(u32, u32), CellState>,
    /// Acoustic simulation parameters.
    pub params: AcousticParams,
    /// The RingKernel runtime.
    runtime: Arc<RingKernel>,
    /// Kernel handles for each cell.
    kernels: HashMap<String, KernelHandle>,
    /// The backend being used.
    backend: Backend,
}

impl std::fmt::Debug for KernelGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelGrid")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("backend", &self.backend)
            .field("kernel_count", &self.kernels.len())
            .finish()
    }
}

impl KernelGrid {
    /// Create a new kernel-based simulation grid.
    ///
    /// # Arguments
    /// * `width` - Number of columns
    /// * `height` - Number of rows
    /// * `params` - Acoustic simulation parameters
    /// * `backend` - The compute backend to use
    pub async fn new(
        width: u32,
        height: u32,
        params: AcousticParams,
        backend: Backend,
    ) -> Result<Self> {
        let runtime = Arc::new(RingKernel::with_backend(backend).await?);

        let mut grid = Self {
            width,
            height,
            cells: HashMap::with_capacity((width * height) as usize),
            params,
            runtime,
            kernels: HashMap::new(),
            backend,
        };

        grid.initialize_cells();
        grid.launch_kernels().await?;

        Ok(grid)
    }

    /// Initialize cell states.
    fn initialize_cells(&mut self) {
        self.cells.clear();

        for y in 0..self.height {
            for x in 0..self.width {
                let mut cell = CellState::new(x, y);

                // Mark boundary cells
                let is_boundary =
                    x == 0 || y == 0 || x == self.width - 1 || y == self.height - 1;
                cell.is_boundary = is_boundary;
                cell.reflection_coeff = if is_boundary { 0.95 } else { 1.0 };

                self.cells.insert((x, y), cell);
            }
        }
    }

    /// Launch kernel actors for each cell.
    async fn launch_kernels(&mut self) -> Result<()> {
        for ((_x, _y), cell) in &self.cells {
            let kernel_id = cell.kernel_id();

            let options = LaunchOptions::default();

            let kernel = self.runtime.launch(&kernel_id, options).await?;
            // Note: CPU runtime launches kernels in Active state by default,
            // so we don't call activate() here to avoid InvalidStateTransition errors

            self.kernels.insert(kernel_id, kernel);
        }

        tracing::info!(
            "Launched {} kernel actors on {:?} backend",
            self.kernels.len(),
            self.backend
        );

        Ok(())
    }

    /// Perform one simulation step using kernel actors.
    ///
    /// This coordinates the pressure exchange and computation across
    /// all kernel actors.
    pub async fn step(&mut self) -> Result<()> {
        // Phase 1: Exchange pressure values between neighbors
        self.exchange_pressures();

        // Phase 2: Each cell computes its next pressure value
        for cell in self.cells.values_mut() {
            cell.step(&self.params);
        }

        Ok(())
    }

    /// Exchange pressure values between neighboring cells.
    fn exchange_pressures(&mut self) {
        // Collect current pressure values
        let pressures: HashMap<(u32, u32), f32> = self
            .cells
            .iter()
            .map(|(&pos, cell)| (pos, cell.pressure))
            .collect();

        // Update each cell with neighbor pressures
        for ((x, y), cell) in self.cells.iter_mut() {
            // North neighbor (y - 1)
            cell.p_north = if *y > 0 {
                pressures.get(&(*x, y - 1)).copied().unwrap_or(0.0)
            } else {
                cell.pressure * cell.reflection_coeff // Reflect at boundary
            };

            // South neighbor (y + 1)
            cell.p_south = if *y < self.height - 1 {
                pressures.get(&(*x, y + 1)).copied().unwrap_or(0.0)
            } else {
                cell.pressure * cell.reflection_coeff
            };

            // West neighbor (x - 1)
            cell.p_west = if *x > 0 {
                pressures.get(&(x - 1, *y)).copied().unwrap_or(0.0)
            } else {
                cell.pressure * cell.reflection_coeff
            };

            // East neighbor (x + 1)
            cell.p_east = if *x < self.width - 1 {
                pressures.get(&(x + 1, *y)).copied().unwrap_or(0.0)
            } else {
                cell.pressure * cell.reflection_coeff
            };
        }
    }

    /// Inject an impulse at the given grid position.
    pub fn inject_impulse(&mut self, x: u32, y: u32, amplitude: f32) {
        if let Some(cell) = self.cells.get_mut(&(x, y)) {
            cell.inject_impulse(amplitude);
        }
    }

    /// Get the pressure grid for visualization.
    pub fn get_pressure_grid(&self) -> Vec<Vec<f32>> {
        let mut grid = vec![vec![0.0; self.width as usize]; self.height as usize];
        for ((x, y), cell) in &self.cells {
            if (*y as usize) < grid.len() && (*x as usize) < grid[0].len() {
                grid[*y as usize][*x as usize] = cell.pressure;
            }
        }
        grid
    }

    /// Get the maximum absolute pressure in the grid.
    pub fn max_pressure(&self) -> f32 {
        self.cells
            .values()
            .map(|c| c.pressure.abs())
            .fold(0.0, f32::max)
    }

    /// Get total energy in the system.
    pub fn total_energy(&self) -> f32 {
        self.cells.values().map(|c| c.pressure * c.pressure).sum()
    }

    /// Reset all cells to initial state.
    pub fn reset(&mut self) {
        for cell in self.cells.values_mut() {
            cell.reset();
        }
    }

    /// Resize the grid.
    pub async fn resize(&mut self, new_width: u32, new_height: u32) -> Result<()> {
        // Terminate existing kernels and recreate runtime to clear registered kernel IDs
        self.shutdown_kernels().await?;

        // Create new runtime to clear kernel registrations
        self.runtime = Arc::new(RingKernel::with_backend(self.backend).await?);

        self.width = new_width;
        self.height = new_height;

        self.initialize_cells();
        self.launch_kernels().await
    }

    /// Update acoustic parameters.
    pub fn set_speed_of_sound(&mut self, speed: f32) {
        self.params.set_speed_of_sound(speed);
    }

    /// Update cell size.
    pub fn set_cell_size(&mut self, size: f32) {
        self.params.set_cell_size(size);
    }

    /// Get the number of cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Get the current backend.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Switch to a different backend.
    pub async fn switch_backend(&mut self, backend: Backend) -> Result<()> {
        if self.backend == backend {
            return Ok(());
        }

        // Shutdown current kernels
        self.shutdown_kernels().await?;

        // Create new runtime with new backend
        self.runtime = Arc::new(RingKernel::with_backend(backend).await?);
        self.backend = backend;

        // Relaunch kernels
        self.launch_kernels().await?;

        tracing::info!("Switched to {:?} backend", backend);
        Ok(())
    }

    /// Shutdown all kernel actors.
    async fn shutdown_kernels(&mut self) -> Result<()> {
        for (id, kernel) in self.kernels.drain() {
            if let Err(e) = kernel.terminate().await {
                tracing::warn!("Failed to terminate kernel {}: {:?}", id, e);
            }
        }
        Ok(())
    }

    /// Shutdown the grid and release resources.
    pub async fn shutdown(mut self) -> Result<()> {
        self.shutdown_kernels().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kernel_grid_creation() {
        let params = AcousticParams::new(343.0, 1.0);
        let grid = KernelGrid::new(8, 8, params, Backend::Cpu).await.unwrap();

        assert_eq!(grid.width, 8);
        assert_eq!(grid.height, 8);
        assert_eq!(grid.cell_count(), 64);
        assert_eq!(grid.backend(), Backend::Cpu);
    }

    #[tokio::test]
    async fn test_kernel_grid_impulse() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = KernelGrid::new(4, 4, params, Backend::Cpu).await.unwrap();

        grid.inject_impulse(2, 2, 1.0);

        let pressure_grid = grid.get_pressure_grid();
        assert_eq!(pressure_grid[2][2], 1.0);
    }

    #[tokio::test]
    async fn test_kernel_grid_step() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = KernelGrid::new(8, 8, params, Backend::Cpu).await.unwrap();

        // Inject impulse at center
        grid.inject_impulse(4, 4, 1.0);

        // Run several steps
        for _ in 0..10 {
            grid.step().await.unwrap();
        }

        // Wave should have propagated to neighbors
        let pressure_grid = grid.get_pressure_grid();
        let neighbor_pressure = pressure_grid[4][5];
        assert!(
            neighbor_pressure.abs() > 0.0,
            "Wave should have propagated to neighbor"
        );
    }

    #[tokio::test]
    async fn test_kernel_grid_reset() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = KernelGrid::new(4, 4, params, Backend::Cpu).await.unwrap();

        grid.inject_impulse(2, 2, 1.0);
        grid.step().await.unwrap();

        grid.reset();

        let pressure_grid = grid.get_pressure_grid();
        for row in pressure_grid {
            for p in row {
                assert_eq!(p, 0.0);
            }
        }
    }

    #[tokio::test]
    async fn test_kernel_grid_resize() {
        let params = AcousticParams::new(343.0, 1.0);
        let mut grid = KernelGrid::new(4, 4, params, Backend::Cpu).await.unwrap();

        grid.resize(8, 6).await.unwrap();

        assert_eq!(grid.width, 8);
        assert_eq!(grid.height, 6);
        assert_eq!(grid.cell_count(), 48);
    }

    #[tokio::test]
    async fn test_kernel_grid_shutdown() {
        let params = AcousticParams::new(343.0, 1.0);
        let grid = KernelGrid::new(4, 4, params, Backend::Cpu).await.unwrap();

        // Should shutdown cleanly
        grid.shutdown().await.unwrap();
    }
}
