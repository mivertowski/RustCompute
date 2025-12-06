//! 3D acoustic wave simulation module.
//!
//! Provides:
//! - Physics parameters with realistic acoustic modeling
//! - 3D FDTD simulation grid (CPU and GPU backends)
//! - GPU-accelerated stencil operations

pub mod grid3d;
pub mod physics;

#[cfg(feature = "cuda")]
pub mod gpu_backend;

#[cfg(feature = "cuda-codegen")]
pub mod kernels;

pub use grid3d::{CellType, GridParams, SimulationGrid3D};
pub use physics::{
    AcousticParams3D, AtmosphericAbsorption, Environment, Medium, MediumProperties,
    MultiBandDamping, Orientation3D, Position3D,
};

/// Simulation engine that manages grid updates and GPU/CPU dispatch.
pub struct SimulationEngine {
    /// 3D simulation grid
    pub grid: SimulationGrid3D,
    /// Use GPU backend if available
    pub use_gpu: bool,
    /// GPU backend
    #[cfg(feature = "cuda")]
    pub gpu: Option<gpu_backend::GpuBackend3D>,
}

impl SimulationEngine {
    /// Create a new simulation engine with CPU backend.
    pub fn new_cpu(
        width: usize,
        height: usize,
        depth: usize,
        params: AcousticParams3D,
    ) -> Self {
        Self {
            grid: SimulationGrid3D::new(width, height, depth, params),
            use_gpu: false,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    /// Create a new simulation engine with GPU backend.
    #[cfg(feature = "cuda")]
    pub fn new_gpu(
        width: usize,
        height: usize,
        depth: usize,
        params: AcousticParams3D,
    ) -> Result<Self, gpu_backend::GpuError> {
        let grid = SimulationGrid3D::new(width, height, depth, params);
        let gpu = gpu_backend::GpuBackend3D::new(&grid)?;

        Ok(Self {
            grid,
            use_gpu: true,
            gpu: Some(gpu),
        })
    }

    /// Perform one simulation step.
    pub fn step(&mut self) {
        #[cfg(feature = "cuda")]
        if self.use_gpu {
            if let Some(ref mut gpu) = self.gpu {
                if gpu.step(&mut self.grid).is_ok() {
                    return;
                }
            }
        }

        // Fall back to parallel CPU
        self.grid.step_parallel();
    }

    /// Perform multiple simulation steps.
    pub fn step_n(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Reset the simulation.
    pub fn reset(&mut self) {
        self.grid.reset();

        #[cfg(feature = "cuda")]
        if let Some(ref mut gpu) = self.gpu {
            let _ = gpu.reset(&self.grid);
        }
    }

    /// Inject an impulse at the given position.
    pub fn inject_impulse(&mut self, x: usize, y: usize, z: usize, amplitude: f32) {
        self.grid.inject_impulse(x, y, z, amplitude);

        #[cfg(feature = "cuda")]
        if let Some(ref mut gpu) = self.gpu {
            let _ = gpu.upload_pressure(&self.grid);
        }
    }

    /// Get current simulation time.
    pub fn time(&self) -> f32 {
        self.grid.time
    }

    /// Get current step count.
    pub fn step_count(&self) -> u64 {
        self.grid.step
    }

    /// Get grid dimensions.
    pub fn dimensions(&self) -> (usize, usize, usize) {
        self.grid.dimensions()
    }

    /// Enable or disable GPU acceleration.
    #[cfg(feature = "cuda")]
    pub fn set_use_gpu(&mut self, use_gpu: bool) {
        if use_gpu && self.gpu.is_none() {
            if let Ok(gpu) = gpu_backend::GpuBackend3D::new(&self.grid) {
                self.gpu = Some(gpu);
            }
        }
        self.use_gpu = use_gpu && self.gpu.is_some();
    }

    /// Check if GPU is being used.
    pub fn is_using_gpu(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.use_gpu && self.gpu.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

/// Configuration for creating a simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Grid width (cells)
    pub width: usize,
    /// Grid height (cells)
    pub height: usize,
    /// Grid depth (cells)
    pub depth: usize,
    /// Cell size (meters)
    pub cell_size: f32,
    /// Environment settings
    pub environment: Environment,
    /// Use GPU if available
    pub prefer_gpu: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            depth: 64,
            cell_size: 0.05, // 5cm cells
            environment: Environment::default(),
            prefer_gpu: true,
        }
    }
}

impl SimulationConfig {
    /// Create a configuration for a small room (10m x 10m x 3m).
    pub fn small_room() -> Self {
        Self {
            width: 200,
            height: 60,
            depth: 200,
            cell_size: 0.05, // 5cm cells
            environment: Environment::default(),
            prefer_gpu: true,
        }
    }

    /// Create a configuration for a medium room (20m x 20m x 5m).
    pub fn medium_room() -> Self {
        Self {
            width: 200,
            height: 50,
            depth: 200,
            cell_size: 0.1, // 10cm cells
            environment: Environment::default(),
            prefer_gpu: true,
        }
    }

    /// Create a configuration for a large space (50m x 50m x 10m).
    pub fn large_space() -> Self {
        Self {
            width: 250,
            height: 50,
            depth: 250,
            cell_size: 0.2, // 20cm cells
            environment: Environment::default(),
            prefer_gpu: true,
        }
    }

    /// Create a configuration for underwater simulation.
    pub fn underwater() -> Self {
        Self {
            environment: Environment::default().with_medium(Medium::Water),
            ..Self::medium_room()
        }
    }

    /// Set the environment.
    pub fn with_environment(mut self, env: Environment) -> Self {
        self.environment = env;
        self
    }

    /// Build the simulation engine.
    pub fn build(self) -> SimulationEngine {
        let params = AcousticParams3D::new(self.environment, self.cell_size);

        #[cfg(feature = "cuda")]
        if self.prefer_gpu {
            if let Ok(engine) =
                SimulationEngine::new_gpu(self.width, self.height, self.depth, params.clone())
            {
                return engine;
            }
        }

        SimulationEngine::new_cpu(self.width, self.height, self.depth, params)
    }

    /// Get the physical dimensions of the simulation space.
    pub fn physical_dimensions(&self) -> (f32, f32, f32) {
        (
            self.width as f32 * self.cell_size,
            self.height as f32 * self.cell_size,
            self.depth as f32 * self.cell_size,
        )
    }

    /// Get the maximum accurately simulated frequency.
    pub fn max_frequency(&self, speed_of_sound: f32) -> f32 {
        speed_of_sound / (10.0 * self.cell_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_engine_cpu() {
        let mut engine = SimulationEngine::new_cpu(
            32,
            32,
            32,
            AcousticParams3D::default(),
        );

        engine.inject_impulse(16, 16, 16, 1.0);
        engine.step();

        assert_eq!(engine.step_count(), 1);
        assert!(engine.time() > 0.0);
    }

    #[test]
    fn test_simulation_config() {
        let config = SimulationConfig::default();
        let (w, h, d) = config.physical_dimensions();

        assert!(w > 0.0);
        assert!(h > 0.0);
        assert!(d > 0.0);
    }

    #[test]
    fn test_config_presets() {
        let small = SimulationConfig::small_room();
        let medium = SimulationConfig::medium_room();
        let large = SimulationConfig::large_space();
        let water = SimulationConfig::underwater();

        assert!(small.width < medium.width || small.cell_size < medium.cell_size);
        assert!(medium.width < large.width || medium.cell_size < large.cell_size);
        assert_eq!(water.environment.medium, Medium::Water);
    }
}
