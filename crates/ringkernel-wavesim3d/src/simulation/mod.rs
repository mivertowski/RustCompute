//! 3D acoustic wave simulation module.
//!
//! Provides:
//! - Physics parameters with realistic acoustic modeling
//! - 3D FDTD simulation grid (CPU and GPU backends)
//! - GPU-accelerated stencil operations
//! - **Actor-based GPU simulation** (cell-as-actor paradigm)
//!
//! # Computation Methods
//!
//! The simulation supports two GPU computation methods:
//!
//! ## Stencil Method (Default)
//! Traditional GPU stencil computation where each thread reads neighbor values
//! from shared memory. Fast and efficient for regular grids.
//!
//! ## Actor Method
//! Novel cell-as-actor paradigm where each spatial cell is an independent actor.
//! Actors communicate via message passing (halo exchange) instead of shared memory.
//! Uses HLC (Hybrid Logical Clocks) for temporal alignment.
//!
//! ```ignore
//! use ringkernel_wavesim3d::simulation::{ComputationMethod, SimulationConfig};
//!
//! let config = SimulationConfig::default()
//!     .with_computation_method(ComputationMethod::Actor);
//! let engine = config.build();
//! ```

pub mod grid3d;
pub mod physics;

#[cfg(feature = "cuda")]
pub mod gpu_backend;

#[cfg(feature = "cuda")]
pub mod actor_backend;

pub use grid3d::{CellType, GridParams, SimulationGrid3D};
pub use physics::{
    AcousticParams3D, AtmosphericAbsorption, Environment, Medium, MediumProperties,
    MultiBandDamping, Orientation3D, Position3D,
};

#[cfg(feature = "cuda")]
pub use actor_backend::{
    ActorBackendConfig, ActorError, ActorStats, CellActorState, Direction3D, HaloMessage,
};

/// Computation method for GPU-accelerated simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputationMethod {
    /// Traditional stencil-based GPU computation.
    /// Each thread reads neighbor values from global/shared memory.
    /// Fast and efficient for regular grids.
    #[default]
    Stencil,

    /// Actor-based GPU computation (cell-as-actor paradigm).
    /// Each spatial cell is an independent actor that:
    /// - Holds its own state (pressure, cell type)
    /// - Communicates with neighbors via message passing
    /// - Uses HLC for temporal alignment
    ///
    /// This method demonstrates the actor model on GPUs but may have
    /// higher overhead than stencil computation.
    Actor,
}

/// Simulation engine that manages grid updates and GPU/CPU dispatch.
pub struct SimulationEngine {
    /// 3D simulation grid
    pub grid: SimulationGrid3D,
    /// Use GPU backend if available
    pub use_gpu: bool,
    /// Computation method for GPU
    pub computation_method: ComputationMethod,
    /// Stencil GPU backend
    #[cfg(feature = "cuda")]
    pub gpu: Option<gpu_backend::GpuBackend3D>,
    /// Actor GPU backend
    #[cfg(feature = "cuda")]
    pub actor_gpu: Option<actor_backend::ActorGpuBackend3D>,
}

impl SimulationEngine {
    /// Create a new simulation engine with CPU backend.
    pub fn new_cpu(width: usize, height: usize, depth: usize, params: AcousticParams3D) -> Self {
        Self {
            grid: SimulationGrid3D::new(width, height, depth, params),
            use_gpu: false,
            computation_method: ComputationMethod::Stencil,
            #[cfg(feature = "cuda")]
            gpu: None,
            #[cfg(feature = "cuda")]
            actor_gpu: None,
        }
    }

    /// Create a new simulation engine with GPU backend (stencil method).
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
            computation_method: ComputationMethod::Stencil,
            gpu: Some(gpu),
            actor_gpu: None,
        })
    }

    /// Create a new simulation engine with GPU backend using actor model.
    ///
    /// This uses the cell-as-actor paradigm where each spatial cell is an
    /// independent actor that communicates with neighbors via message passing.
    #[cfg(feature = "cuda")]
    pub fn new_gpu_actor(
        width: usize,
        height: usize,
        depth: usize,
        params: AcousticParams3D,
        actor_config: actor_backend::ActorBackendConfig,
    ) -> Result<Self, actor_backend::ActorError> {
        let grid = SimulationGrid3D::new(width, height, depth, params);
        let actor_gpu = actor_backend::ActorGpuBackend3D::new(&grid, actor_config)?;

        Ok(Self {
            grid,
            use_gpu: true,
            computation_method: ComputationMethod::Actor,
            gpu: None,
            actor_gpu: Some(actor_gpu),
        })
    }

    /// Perform one simulation step.
    pub fn step(&mut self) {
        #[cfg(feature = "cuda")]
        if self.use_gpu {
            match self.computation_method {
                ComputationMethod::Stencil => {
                    if let Some(ref mut gpu) = self.gpu {
                        if gpu.step(&mut self.grid).is_ok() {
                            return;
                        }
                    }
                }
                ComputationMethod::Actor => {
                    if let Some(ref mut actor_gpu) = self.actor_gpu {
                        if actor_gpu.step(&mut self.grid, 1).is_ok() {
                            return;
                        }
                    }
                }
            }
        }

        // Fall back to sequential CPU (parallel can cause issues with GUI event loops)
        self.grid.step_sequential();
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
        {
            if let Some(ref mut gpu) = self.gpu {
                let _ = gpu.reset(&self.grid);
            }
            if let Some(ref mut actor_gpu) = self.actor_gpu {
                let _ = actor_gpu.reset(&self.grid);
            }
        }
    }

    /// Inject an impulse at the given position.
    pub fn inject_impulse(&mut self, x: usize, y: usize, z: usize, amplitude: f32) {
        self.grid.inject_impulse(x, y, z, amplitude);

        #[cfg(feature = "cuda")]
        {
            if let Some(ref mut gpu) = self.gpu {
                let _ = gpu.upload_pressure(&self.grid);
            }
            if let Some(ref mut actor_gpu) = self.actor_gpu {
                let _ = actor_gpu.upload_pressure(&self.grid);
            }
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
        if use_gpu {
            match self.computation_method {
                ComputationMethod::Stencil => {
                    if self.gpu.is_none() {
                        if let Ok(gpu) = gpu_backend::GpuBackend3D::new(&self.grid) {
                            self.gpu = Some(gpu);
                        }
                    }
                    self.use_gpu = self.gpu.is_some();
                }
                ComputationMethod::Actor => {
                    if self.actor_gpu.is_none() {
                        if let Ok(actor_gpu) = actor_backend::ActorGpuBackend3D::new(
                            &self.grid,
                            actor_backend::ActorBackendConfig::default(),
                        ) {
                            self.actor_gpu = Some(actor_gpu);
                        }
                    }
                    self.use_gpu = self.actor_gpu.is_some();
                }
            }
        } else {
            self.use_gpu = false;
        }
    }

    /// Set the computation method for GPU.
    #[cfg(feature = "cuda")]
    pub fn set_computation_method(&mut self, method: ComputationMethod) {
        self.computation_method = method;
        // Re-initialize the appropriate backend if GPU is in use
        if self.use_gpu {
            self.set_use_gpu(true);
        }
    }

    /// Check if GPU is being used.
    pub fn is_using_gpu(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            match self.computation_method {
                ComputationMethod::Stencil => self.use_gpu && self.gpu.is_some(),
                ComputationMethod::Actor => self.use_gpu && self.actor_gpu.is_some(),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Get the current computation method.
    pub fn computation_method(&self) -> ComputationMethod {
        self.computation_method
    }

    /// Download pressure data from GPU to CPU grid (if using GPU).
    #[cfg(feature = "cuda")]
    pub fn sync_from_gpu(&mut self) {
        if self.use_gpu {
            match self.computation_method {
                ComputationMethod::Stencil => {
                    if let Some(ref gpu) = self.gpu {
                        let _ = gpu.download_pressure(&mut self.grid);
                    }
                }
                ComputationMethod::Actor => {
                    if let Some(ref actor_gpu) = self.actor_gpu {
                        let _ = actor_gpu.download_pressure(&mut self.grid);
                    }
                }
            }
        }
    }

    /// Get actor backend statistics (if using actor method).
    #[cfg(feature = "cuda")]
    pub fn actor_stats(&self) -> Option<actor_backend::ActorStats> {
        self.actor_gpu.as_ref().map(|gpu| gpu.stats())
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
    /// Computation method for GPU
    pub computation_method: ComputationMethod,
    /// Actor backend configuration (for Actor method)
    #[cfg(feature = "cuda")]
    pub actor_config: actor_backend::ActorBackendConfig,
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
            computation_method: ComputationMethod::Stencil,
            #[cfg(feature = "cuda")]
            actor_config: actor_backend::ActorBackendConfig::default(),
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
            ..Default::default()
        }
    }

    /// Create a configuration for a medium room (20m x 20m x 5m).
    pub fn medium_room() -> Self {
        Self {
            width: 200,
            height: 50,
            depth: 200,
            cell_size: 0.1, // 10cm cells
            ..Default::default()
        }
    }

    /// Create a configuration for a large space (50m x 50m x 10m).
    pub fn large_space() -> Self {
        Self {
            width: 250,
            height: 50,
            depth: 250,
            cell_size: 0.2, // 20cm cells
            ..Default::default()
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

    /// Set the computation method.
    ///
    /// - `Stencil`: Traditional GPU stencil computation (default)
    /// - `Actor`: Cell-as-actor paradigm with message-based halo exchange
    pub fn with_computation_method(mut self, method: ComputationMethod) -> Self {
        self.computation_method = method;
        self
    }

    /// Set the actor backend configuration (only used with Actor method).
    #[cfg(feature = "cuda")]
    pub fn with_actor_config(mut self, config: actor_backend::ActorBackendConfig) -> Self {
        self.actor_config = config;
        self
    }

    /// Build the simulation engine.
    pub fn build(self) -> SimulationEngine {
        let params = AcousticParams3D::new(self.environment, self.cell_size);

        #[cfg(feature = "cuda")]
        if self.prefer_gpu {
            match self.computation_method {
                ComputationMethod::Stencil => {
                    if let Ok(engine) = SimulationEngine::new_gpu(
                        self.width,
                        self.height,
                        self.depth,
                        params.clone(),
                    ) {
                        return engine;
                    }
                }
                ComputationMethod::Actor => {
                    if let Ok(engine) = SimulationEngine::new_gpu_actor(
                        self.width,
                        self.height,
                        self.depth,
                        params.clone(),
                        self.actor_config,
                    ) {
                        return engine;
                    }
                }
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
        let mut engine = SimulationEngine::new_cpu(32, 32, 32, AcousticParams3D::default());

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

    #[test]
    fn test_computation_method_default() {
        let config = SimulationConfig::default();
        assert_eq!(config.computation_method, ComputationMethod::Stencil);
    }

    #[test]
    fn test_computation_method_actor() {
        let config = SimulationConfig::default().with_computation_method(ComputationMethod::Actor);
        assert_eq!(config.computation_method, ComputationMethod::Actor);
    }

    #[test]
    fn test_engine_computation_method() {
        let engine = SimulationEngine::new_cpu(16, 16, 16, AcousticParams3D::default());
        assert_eq!(engine.computation_method(), ComputationMethod::Stencil);
    }
}
