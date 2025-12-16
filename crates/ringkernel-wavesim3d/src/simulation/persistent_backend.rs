//! Persistent GPU Backend for WaveSim3D
//!
//! This module provides a truly persistent GPU actor backend where a single kernel
//! runs for the entire simulation lifetime. Commands are sent via H2K (Host-to-Kernel)
//! message queue, and the kernel processes them continuously without relaunching.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         HOST (CPU)                              │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │              PersistentBackend                           │   │
//! │  │  • step() → H2K RunSteps command                         │   │
//! │  │  • inject_impulse() → H2K InjectImpulse command          │   │
//! │  │  • poll() → K2H responses                                │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! │                          │                                      │
//! │  ┌───────────────────────▼───────────────────────────────────┐  │
//! │  │          MAPPED MEMORY (CPU + GPU visible)                │  │
//! │  │  [ControlBlock] [H2K Queue] [K2H Queue]                   │  │
//! │  └───────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │ PCIe (commands only)
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    GPU (PERSISTENT KERNEL)                      │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                 COORDINATOR (Block 0)                    │   │
//! │  │  • Process H2K commands • Send K2H responses            │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │                          │ grid.sync()                          │
//! │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
//! │  │Block 1 │ │Block 2 │ │Block 3 │ │  ...   │ │Block N │       │
//! │  │ Actor  │ │ Actor  │ │ Actor  │ │        │ │ Actor  │       │
//! │  └───┬────┘ └───┬────┘ └───┬────┘ └────────┘ └───┬────┘       │
//! │      └──────────┴──────────┼────────────────────┘              │
//! │                    K2K HALO EXCHANGE                            │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │  [Pressure A] [Pressure B] - Double-buffered simulation │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use ringkernel_core::error::RingKernelError;
use ringkernel_cuda::persistent::{
    PersistentSimulation, PersistentSimulationConfig, PersistentSimulationStats, ResponseType,
};
use ringkernel_cuda::CudaDevice;
use ringkernel_cuda_codegen::persistent_fdtd::{
    generate_persistent_fdtd_kernel, PersistentFdtdConfig,
};
use std::process::Command;
use std::time::Duration;

use super::grid3d::SimulationGrid3D;

/// Error type for persistent backend operations.
#[derive(Debug)]
pub enum PersistentBackendError {
    /// CUDA device creation failed.
    DeviceError(String),
    /// Simulation creation failed.
    SimulationError(String),
    /// Kernel compilation failed.
    CompilationError(String),
    /// Kernel launch failed.
    LaunchError(String),
    /// Command send failed.
    CommandError(String),
    /// Pressure readback failed.
    ReadbackError(String),
}

impl std::fmt::Display for PersistentBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistentBackendError::DeviceError(s) => write!(f, "CUDA device error: {}", s),
            PersistentBackendError::SimulationError(s) => write!(f, "Simulation error: {}", s),
            PersistentBackendError::CompilationError(s) => {
                write!(f, "Kernel compilation error: {}", s)
            }
            PersistentBackendError::LaunchError(s) => write!(f, "Kernel launch error: {}", s),
            PersistentBackendError::CommandError(s) => write!(f, "Command error: {}", s),
            PersistentBackendError::ReadbackError(s) => write!(f, "Readback error: {}", s),
        }
    }
}

impl std::error::Error for PersistentBackendError {}

/// Configuration for the persistent backend.
#[derive(Debug, Clone)]
pub struct PersistentBackendConfig {
    /// Tile size per dimension (cells per block).
    pub tile_size: (usize, usize, usize),
    /// Use cooperative groups for grid synchronization.
    pub use_cooperative: bool,
    /// Progress report interval (steps).
    pub progress_interval: u64,
    /// H2K queue capacity.
    pub h2k_capacity: usize,
    /// K2H queue capacity.
    pub k2h_capacity: usize,
}

impl Default for PersistentBackendConfig {
    fn default() -> Self {
        Self {
            tile_size: (8, 8, 8),
            use_cooperative: true,
            progress_interval: 100,
            h2k_capacity: 256,
            k2h_capacity: 256,
        }
    }
}

impl PersistentBackendConfig {
    /// Set tile size.
    pub fn with_tile_size(mut self, tx: usize, ty: usize, tz: usize) -> Self {
        self.tile_size = (tx, ty, tz);
        self
    }

    /// Enable/disable cooperative groups.
    pub fn with_cooperative(mut self, use_coop: bool) -> Self {
        self.use_cooperative = use_coop;
        self
    }

    /// Set progress reporting interval.
    pub fn with_progress_interval(mut self, interval: u64) -> Self {
        self.progress_interval = interval;
        self
    }
}

/// Statistics from the persistent backend.
#[derive(Debug, Clone, Default)]
pub struct PersistentBackendStats {
    /// Current simulation step.
    pub current_step: u64,
    /// Steps remaining to process.
    pub steps_remaining: u64,
    /// Total energy in simulation.
    pub total_energy: f32,
    /// Messages processed by kernel.
    pub messages_processed: u64,
    /// K2K messages sent.
    pub k2k_sent: u64,
    /// K2K messages received.
    pub k2k_received: u64,
    /// Kernel has terminated.
    pub has_terminated: bool,
    /// Kernel is running.
    pub is_running: bool,
}

impl From<PersistentSimulationStats> for PersistentBackendStats {
    fn from(stats: PersistentSimulationStats) -> Self {
        Self {
            current_step: stats.current_step,
            steps_remaining: stats.steps_remaining,
            total_energy: stats.total_energy,
            messages_processed: stats.messages_processed,
            k2k_sent: stats.k2k_sent,
            k2k_received: stats.k2k_received,
            has_terminated: stats.has_terminated,
            is_running: false, // Updated separately
        }
    }
}

/// Persistent GPU backend for wavesim3d.
///
/// This backend launches a single persistent kernel that runs for the entire
/// simulation lifetime. All operations (steps, impulses, etc.) are sent as
/// commands through the H2K message queue.
pub struct PersistentBackend {
    /// The underlying persistent simulation.
    simulation: PersistentSimulation,
    /// Configuration.
    config: PersistentBackendConfig,
    /// PTX code for the kernel.
    ptx: String,
    /// Kernel has been started.
    started: bool,
}

impl PersistentBackend {
    /// Create a new persistent backend for the given grid.
    pub fn new(
        grid: &SimulationGrid3D,
        config: PersistentBackendConfig,
    ) -> Result<Self, PersistentBackendError> {
        // Create CUDA device
        let device = CudaDevice::new(0)
            .map_err(|e: RingKernelError| PersistentBackendError::DeviceError(e.to_string()))?;

        let (width, height, depth) = grid.dimensions();

        // Create simulation config
        let sim_config = PersistentSimulationConfig::new(width, height, depth)
            .with_tile_size(config.tile_size.0, config.tile_size.1, config.tile_size.2)
            .with_acoustics(
                grid.params.medium.speed_of_sound,
                grid.params.cell_size,
                grid.params.simple_damping,
            );

        // Check grid fits cooperative limits and auto-fallback to software sync
        let total_blocks = sim_config.total_blocks();
        let use_cooperative = if config.use_cooperative {
            // Query actual cooperative limit from device
            // Typical limits: 48-512 blocks depending on GPU and kernel
            // If we exceed, fall back to software sync
            let estimated_max_blocks = 512; // Conservative estimate
            if total_blocks > estimated_max_blocks {
                eprintln!(
                    "Note: Grid requires {} blocks, exceeds estimated cooperative limit {}. Using software sync.",
                    total_blocks, estimated_max_blocks
                );
                false
            } else {
                true
            }
        } else {
            false
        };

        // Generate CUDA kernel code with appropriate sync method
        let kernel_config = PersistentFdtdConfig::new("persistent_fdtd3d")
            .with_tile_size(config.tile_size.0, config.tile_size.1, config.tile_size.2)
            .with_cooperative(use_cooperative)
            .with_progress_interval(config.progress_interval);

        let cuda_code = generate_persistent_fdtd_kernel(&kernel_config);

        // Compile to PTX using nvcc
        let ptx = compile_cuda_to_ptx(&cuda_code, config.use_cooperative)
            .map_err(|e| PersistentBackendError::CompilationError(e))?;

        // Create persistent simulation
        let simulation = PersistentSimulation::new(&device, sim_config)
            .map_err(|e| PersistentBackendError::SimulationError(e.to_string()))?;

        Ok(Self {
            simulation,
            config,
            ptx,
            started: false,
        })
    }

    /// Start the persistent kernel.
    ///
    /// This launches the kernel which will run until shutdown() is called.
    pub fn start(&mut self) -> Result<(), PersistentBackendError> {
        if self.started {
            return Ok(());
        }

        self.simulation
            .start(&self.ptx, "persistent_fdtd3d")
            .map_err(|e| PersistentBackendError::LaunchError(e.to_string()))?;

        self.started = true;
        Ok(())
    }

    /// Run N simulation steps.
    ///
    /// This sends a RunSteps command to the kernel and waits for completion.
    pub fn step(
        &mut self,
        grid: &mut SimulationGrid3D,
        steps: usize,
    ) -> Result<(), PersistentBackendError> {
        // Auto-start if not started
        if !self.started {
            self.start()?;
        }

        // Get initial step count
        let initial_step = self.simulation.stats().current_step;

        // Send run steps command
        self.simulation
            .run_steps(steps as u64)
            .map_err(|e| PersistentBackendError::CommandError(e.to_string()))?;

        // Wait for the kernel to acknowledge the command (steps_remaining becomes > 0)
        // Then wait for completion (current_step reaches target)
        let target_step = initial_step + steps as u64;
        let timeout = Duration::from_secs(10);
        let start = std::time::Instant::now();
        let mut command_acknowledged = false;

        loop {
            let stats = self.simulation.stats();

            // First, wait for command to be acknowledged
            if !command_acknowledged {
                if stats.steps_remaining > 0 || stats.current_step > initial_step {
                    command_acknowledged = true;
                }
            }

            // Once acknowledged, wait for completion
            if command_acknowledged && stats.current_step >= target_step {
                break;
            }

            // Check for termination error
            if stats.has_terminated {
                return Err(PersistentBackendError::CommandError(
                    "Kernel terminated unexpectedly".to_string(),
                ));
            }

            // Check for timeout
            if start.elapsed() > timeout {
                return Err(PersistentBackendError::CommandError(format!(
                    "Timeout waiting for steps. Acknowledged: {}, current: {}, target: {}",
                    command_acknowledged, stats.current_step, target_step
                )));
            }

            // Poll for responses (don't block)
            let _ = self.simulation.poll_responses();

            // Brief sleep to avoid busy-waiting
            std::thread::sleep(Duration::from_micros(100));
        }

        // Update grid step count
        grid.step += steps as u64;
        grid.time += steps as f32 * grid.params.time_step;

        Ok(())
    }

    /// Inject an impulse at the given position.
    pub fn inject_impulse(
        &mut self,
        x: u32,
        y: u32,
        z: u32,
        amplitude: f32,
    ) -> Result<(), PersistentBackendError> {
        // Auto-start if not started
        if !self.started {
            self.start()?;
        }

        self.simulation
            .inject_impulse(x, y, z, amplitude)
            .map_err(|e| PersistentBackendError::CommandError(e.to_string()))?;

        Ok(())
    }

    /// Reset the simulation state.
    pub fn reset(&mut self, _grid: &SimulationGrid3D) -> Result<(), PersistentBackendError> {
        // For persistent kernel, reset means shutting down and restarting
        if self.started {
            self.shutdown()?;
        }

        // Will be restarted on next step() call
        Ok(())
    }

    /// Download pressure data from GPU to CPU grid.
    pub fn download_pressure(
        &self,
        grid: &mut SimulationGrid3D,
    ) -> Result<(), PersistentBackendError> {
        if !self.started {
            return Ok(()); // Nothing to download
        }

        let (width, height, depth) = grid.dimensions();
        let total_cells = width * height * depth;

        // Read into grid's pressure buffer
        if grid.pressure.len() != total_cells {
            return Err(PersistentBackendError::ReadbackError(
                "Grid pressure buffer size mismatch".to_string(),
            ));
        }

        self.simulation
            .read_pressure(&mut grid.pressure)
            .map_err(|e| PersistentBackendError::ReadbackError(e.to_string()))?;

        Ok(())
    }

    /// Get current statistics.
    pub fn stats(&self) -> PersistentBackendStats {
        let mut stats = PersistentBackendStats::from(self.simulation.stats());
        stats.is_running = self.started && self.simulation.is_running();
        stats
    }

    /// Poll for K2H responses.
    pub fn poll_responses(&self) -> Vec<(ResponseType, u64, u64, f32)> {
        self.simulation
            .poll_responses()
            .into_iter()
            .map(|msg| {
                let resp_type = match msg.resp_type {
                    0 => ResponseType::Ack,
                    1 => ResponseType::Progress,
                    2 => ResponseType::Error,
                    3 => ResponseType::Terminated,
                    4 => ResponseType::Energy,
                    _ => ResponseType::Ack,
                };
                (resp_type, msg.step, msg.steps_remaining, msg.energy)
            })
            .collect()
    }

    /// Gracefully shut down the persistent kernel.
    pub fn shutdown(&mut self) -> Result<(), PersistentBackendError> {
        if !self.started {
            return Ok(());
        }

        self.simulation
            .shutdown()
            .map_err(|e| PersistentBackendError::CommandError(e.to_string()))?;

        self.started = false;
        Ok(())
    }

    /// Check if the kernel is running.
    pub fn is_running(&self) -> bool {
        self.started && self.simulation.is_running()
    }
}

impl Drop for PersistentBackend {
    fn drop(&mut self) {
        // Ensure graceful shutdown
        let _ = self.shutdown();
    }
}

/// Compile CUDA code to PTX using nvcc.
fn compile_cuda_to_ptx(cuda_code: &str, use_cooperative: bool) -> Result<String, String> {
    // Write to temp file
    let temp_dir = std::env::temp_dir();
    let cuda_file = temp_dir.join("persistent_fdtd_wavesim.cu");
    let ptx_file = temp_dir.join("persistent_fdtd_wavesim.ptx");

    std::fs::write(&cuda_file, cuda_code)
        .map_err(|e| format!("Failed to write CUDA file: {}", e))?;

    // Compile with nvcc
    // Use -arch=native to automatically detect the GPU architecture
    // This ensures compatibility with newer CUDA versions that dropped older sm_* support
    let mut args = vec![
        "-ptx".to_string(),
        "-o".to_string(),
        ptx_file.to_string_lossy().to_string(),
        cuda_file.to_string_lossy().to_string(),
        "-arch=native".to_string(),
        "-std=c++17".to_string(),
    ];

    if use_cooperative {
        args.push("-rdc=true".to_string()); // Required for cooperative groups
    }

    let output = Command::new("nvcc")
        .args(&args)
        .output()
        .map_err(|e| format!("Failed to run nvcc: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "nvcc compilation failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    std::fs::read_to_string(&ptx_file).map_err(|e| format!("Failed to read PTX: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = PersistentBackendConfig::default();
        assert_eq!(config.tile_size, (8, 8, 8));
        assert!(config.use_cooperative);
        assert_eq!(config.progress_interval, 100);
    }

    #[test]
    fn test_config_builder() {
        let config = PersistentBackendConfig::default()
            .with_tile_size(4, 4, 4)
            .with_cooperative(false)
            .with_progress_interval(50);

        assert_eq!(config.tile_size, (4, 4, 4));
        assert!(!config.use_cooperative);
        assert_eq!(config.progress_interval, 50);
    }

    #[test]
    fn test_error_display() {
        let err = PersistentBackendError::DeviceError("test".to_string());
        assert!(err.to_string().contains("CUDA device error"));

        let err = PersistentBackendError::CompilationError("failed".to_string());
        assert!(err.to_string().contains("compilation error"));
    }
}
