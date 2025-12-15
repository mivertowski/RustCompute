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

#[cfg(feature = "cuda")]
pub mod block_actor_backend;

#[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
pub mod persistent_backend;

#[cfg(feature = "cuda-codegen")]
pub mod kernels3d;

pub use grid3d::{CellType, GridParams, SimulationGrid3D};
pub use physics::{
    AcousticParams3D, AtmosphericAbsorption, Environment, Medium, MediumProperties,
    MultiBandDamping, Orientation3D, Position3D,
};

#[cfg(feature = "cuda")]
pub use actor_backend::{
    ActorBackendConfig, ActorError, ActorStats, CellActorState, Direction3D, HaloMessage,
};

#[cfg(feature = "cuda")]
pub use block_actor_backend::{
    BlockActorConfig, BlockActorError, BlockActorGpuBackend, BlockActorStats,
};

#[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
pub use persistent_backend::{PersistentBackend, PersistentBackendConfig, PersistentBackendError, PersistentBackendStats};

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

    /// Block-based actor computation (hybrid approach).
    /// Each 8×8×8 block of cells is a single actor (512 cells per actor).
    /// - Intra-block: Fast stencil computation with shared memory
    /// - Inter-block: Double-buffered message passing (no atomics)
    ///
    /// Combines actor model benefits with stencil performance.
    /// Expected: 10-50× faster than per-cell Actor method.
    BlockActor,

    /// Truly persistent GPU actor paradigm.
    /// A single kernel runs for the entire simulation lifetime:
    /// - H2K (Host→Kernel) command messaging via mapped memory
    /// - K2H (Kernel→Host) response messaging
    /// - K2K halo exchange between blocks on device memory
    /// - Grid-wide synchronization via cooperative groups
    ///
    /// No kernel launch overhead per step - commands control simulation
    /// from host while GPU runs continuously.
    Persistent,
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
    /// Block actor GPU backend
    #[cfg(feature = "cuda")]
    pub block_actor_gpu: Option<block_actor_backend::BlockActorGpuBackend>,
    /// Persistent GPU backend (true GPU actor with single kernel launch)
    #[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
    pub persistent_gpu: Option<persistent_backend::PersistentBackend>,
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
            #[cfg(feature = "cuda")]
            block_actor_gpu: None,
            #[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
            persistent_gpu: None,
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
            block_actor_gpu: None,
            #[cfg(feature = "cuda-codegen")]
            persistent_gpu: None,
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
            block_actor_gpu: None,
            #[cfg(feature = "cuda-codegen")]
            persistent_gpu: None,
        })
    }

    /// Create a new simulation engine with GPU backend using block actor model.
    ///
    /// This uses a hybrid approach where each 8×8×8 block of cells is a single actor.
    /// Intra-block computation uses fast stencil, inter-block uses double-buffered messaging.
    #[cfg(feature = "cuda")]
    pub fn new_gpu_block_actor(
        width: usize,
        height: usize,
        depth: usize,
        params: AcousticParams3D,
        config: block_actor_backend::BlockActorConfig,
    ) -> Result<Self, block_actor_backend::BlockActorError> {
        let grid = SimulationGrid3D::new(width, height, depth, params);
        let block_actor_gpu = block_actor_backend::BlockActorGpuBackend::new(&grid, config)?;

        Ok(Self {
            grid,
            use_gpu: true,
            computation_method: ComputationMethod::BlockActor,
            gpu: None,
            actor_gpu: None,
            block_actor_gpu: Some(block_actor_gpu),
            #[cfg(feature = "cuda-codegen")]
            persistent_gpu: None,
        })
    }

    /// Create a new simulation engine with persistent GPU backend.
    ///
    /// This uses the truly persistent GPU actor paradigm where a single kernel
    /// runs for the entire simulation lifetime. Commands are sent via H2K queue
    /// and the kernel processes them continuously.
    #[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
    pub fn new_gpu_persistent(
        width: usize,
        height: usize,
        depth: usize,
        params: AcousticParams3D,
        config: persistent_backend::PersistentBackendConfig,
    ) -> Result<Self, persistent_backend::PersistentBackendError> {
        let grid = SimulationGrid3D::new(width, height, depth, params);
        let persistent_gpu = persistent_backend::PersistentBackend::new(&grid, config)?;

        Ok(Self {
            grid,
            use_gpu: true,
            computation_method: ComputationMethod::Persistent,
            #[cfg(feature = "cuda")]
            gpu: None,
            #[cfg(feature = "cuda")]
            actor_gpu: None,
            #[cfg(feature = "cuda")]
            block_actor_gpu: None,
            persistent_gpu: Some(persistent_gpu),
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
                ComputationMethod::BlockActor => {
                    if let Some(ref mut block_actor_gpu) = self.block_actor_gpu {
                        // Use fused kernel for better performance (single kernel launch)
                        if block_actor_gpu.step_fused(&mut self.grid, 1).is_ok() {
                            return;
                        }
                    }
                }
                #[cfg(feature = "cuda-codegen")]
                ComputationMethod::Persistent => {
                    if let Some(ref mut persistent_gpu) = self.persistent_gpu {
                        // Persistent kernel runs continuously - just request one step
                        if persistent_gpu.step(&mut self.grid, 1).is_ok() {
                            return;
                        }
                    }
                }
                #[cfg(not(feature = "cuda-codegen"))]
                ComputationMethod::Persistent => {
                    // Fall through to CPU when cuda-codegen not enabled
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
            if let Some(ref mut block_actor_gpu) = self.block_actor_gpu {
                let _ = block_actor_gpu.reset(&self.grid);
            }
            #[cfg(feature = "cuda-codegen")]
            if let Some(ref mut persistent_gpu) = self.persistent_gpu {
                let _ = persistent_gpu.reset(&self.grid);
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
            if let Some(ref mut block_actor_gpu) = self.block_actor_gpu {
                let _ = block_actor_gpu.upload_pressure(&self.grid);
            }
            #[cfg(feature = "cuda-codegen")]
            if let Some(ref mut persistent_gpu) = self.persistent_gpu {
                // Persistent kernel receives impulse via H2K message queue
                let _ = persistent_gpu.inject_impulse(x as u32, y as u32, z as u32, amplitude);
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
                ComputationMethod::BlockActor => {
                    if self.block_actor_gpu.is_none() {
                        if let Ok(block_actor_gpu) = block_actor_backend::BlockActorGpuBackend::new(
                            &self.grid,
                            block_actor_backend::BlockActorConfig::default(),
                        ) {
                            self.block_actor_gpu = Some(block_actor_gpu);
                        }
                    }
                    self.use_gpu = self.block_actor_gpu.is_some();
                }
                #[cfg(feature = "cuda-codegen")]
                ComputationMethod::Persistent => {
                    if self.persistent_gpu.is_none() {
                        if let Ok(persistent_gpu) = persistent_backend::PersistentBackend::new(
                            &self.grid,
                            persistent_backend::PersistentBackendConfig::default(),
                        ) {
                            self.persistent_gpu = Some(persistent_gpu);
                        }
                    }
                    self.use_gpu = self.persistent_gpu.is_some();
                }
                #[cfg(not(feature = "cuda-codegen"))]
                ComputationMethod::Persistent => {
                    // Persistent method requires cuda-codegen feature
                    self.use_gpu = false;
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
                ComputationMethod::BlockActor => self.use_gpu && self.block_actor_gpu.is_some(),
                #[cfg(feature = "cuda-codegen")]
                ComputationMethod::Persistent => self.use_gpu && self.persistent_gpu.is_some(),
                #[cfg(not(feature = "cuda-codegen"))]
                ComputationMethod::Persistent => false,
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
                ComputationMethod::BlockActor => {
                    if let Some(ref block_actor_gpu) = self.block_actor_gpu {
                        let _ = block_actor_gpu.download_pressure(&mut self.grid);
                    }
                }
                #[cfg(feature = "cuda-codegen")]
                ComputationMethod::Persistent => {
                    if let Some(ref persistent_gpu) = self.persistent_gpu {
                        let _ = persistent_gpu.download_pressure(&mut self.grid);
                    }
                }
                #[cfg(not(feature = "cuda-codegen"))]
                ComputationMethod::Persistent => {}
            }
        }
    }

    /// Get actor backend statistics (if using actor method).
    #[cfg(feature = "cuda")]
    pub fn actor_stats(&self) -> Option<actor_backend::ActorStats> {
        self.actor_gpu.as_ref().map(|gpu| gpu.stats())
    }

    /// Get block actor backend statistics (if using block actor method).
    #[cfg(feature = "cuda")]
    pub fn block_actor_stats(&self) -> Option<block_actor_backend::BlockActorStats> {
        self.block_actor_gpu.as_ref().map(|gpu| gpu.stats())
    }

    /// Get persistent backend statistics (if using persistent method).
    #[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
    pub fn persistent_stats(&self) -> Option<persistent_backend::PersistentBackendStats> {
        self.persistent_gpu.as_ref().map(|gpu| gpu.stats())
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
    /// Block actor backend configuration (for BlockActor method)
    #[cfg(feature = "cuda")]
    pub block_actor_config: block_actor_backend::BlockActorConfig,
    /// Persistent backend configuration (for Persistent method)
    #[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
    pub persistent_config: persistent_backend::PersistentBackendConfig,
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
            #[cfg(feature = "cuda")]
            block_actor_config: block_actor_backend::BlockActorConfig::default(),
            #[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
            persistent_config: persistent_backend::PersistentBackendConfig::default(),
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
                ComputationMethod::BlockActor => {
                    if let Ok(engine) = SimulationEngine::new_gpu_block_actor(
                        self.width,
                        self.height,
                        self.depth,
                        params.clone(),
                        self.block_actor_config,
                    ) {
                        return engine;
                    }
                }
                #[cfg(feature = "cuda-codegen")]
                ComputationMethod::Persistent => {
                    if let Ok(engine) = SimulationEngine::new_gpu_persistent(
                        self.width,
                        self.height,
                        self.depth,
                        params.clone(),
                        self.persistent_config,
                    ) {
                        return engine;
                    }
                }
                #[cfg(not(feature = "cuda-codegen"))]
                ComputationMethod::Persistent => {
                    // Persistent method requires cuda-codegen feature
                    // Fall through to CPU
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
