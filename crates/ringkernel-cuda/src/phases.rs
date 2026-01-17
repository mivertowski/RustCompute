//! Multi-Phase Kernel Execution Support
//!
//! This module provides infrastructure for executing multi-phase GPU kernels
//! with inter-phase synchronization and reductions. This is essential for
//! algorithms like PageRank that require global reductions between computation
//! phases.
//!
//! # Architecture
//!
//! ```text
//! Phase 1: Accumulate              Phase 2: Apply
//! ┌─────────────────────┐          ┌─────────────────────┐
//! │ Each thread:        │          │ Each thread:        │
//! │ if out_degree == 0: │  sync    │ new_rank = base +   │
//! │   contrib = rank    │ ──────>  │   d*(sum + dangling │
//! │ block_reduce(contrib)          │      / N)           │
//! │ atomicAdd(dangling) │          │                     │
//! └─────────────────────┘          └─────────────────────┘
//! ```
//!
//! # Synchronization Modes
//!
//! - **Cooperative**: Single launch with `grid.sync()` (CC 6.0+, lowest latency)
//! - **SoftwareBarrier**: Single launch with atomic counter barrier (universal)
//! - **MultiLaunch**: Separate kernel launches with implicit stream sync (always works)
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cuda::phases::{MultiPhaseExecutor, MultiPhaseConfig, SyncMode, KernelPhase};
//!
//! let config = MultiPhaseConfig::new()
//!     .with_sync_mode(SyncMode::Cooperative)
//!     .add_phase(KernelPhase::new("accumulate", "pagerank_accumulate"))
//!     .add_phase(KernelPhase::new("apply", "pagerank_apply"));
//!
//! let executor = MultiPhaseExecutor::new(&device, config)?;
//! executor.execute(&params)?;
//! ```

use std::fmt;
use std::marker::PhantomData;

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::reduction::{ReductionOp, ReductionScalar};

use crate::device::CudaDevice;
use crate::reduction::ReductionBuffer;

// ============================================================================
// SYNCHRONIZATION MODES
// ============================================================================

/// Synchronization mode between kernel phases.
///
/// The choice of sync mode affects both performance and hardware compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SyncMode {
    /// Use cooperative groups for grid-wide synchronization.
    ///
    /// - Requires Compute Capability 6.0+ (Pascal and newer)
    /// - Lowest latency (~20µs for grid.sync())
    /// - Single kernel launch for all phases
    /// - Requires cooperative launch API
    #[default]
    Cooperative,

    /// Use software barrier with atomic counters.
    ///
    /// - Works on any GPU with atomics (CC 3.0+)
    /// - Moderate latency (~50-100µs depending on grid size)
    /// - Single kernel launch for all phases
    /// - No special launch requirements
    SoftwareBarrier,

    /// Use separate kernel launches with stream synchronization.
    ///
    /// - Works on all GPUs
    /// - Highest latency (~200-500µs per sync point)
    /// - Multiple kernel launches
    /// - Most compatible fallback
    MultiLaunch,
}

impl SyncMode {
    /// Check if this sync mode is supported on the given device.
    pub fn is_supported(&self, device: &CudaDevice) -> bool {
        match self {
            SyncMode::Cooperative => device.supports_cooperative_groups(),
            SyncMode::SoftwareBarrier => true, // Universal support with atomics
            SyncMode::MultiLaunch => true,     // Always supported
        }
    }

    /// Get the best available sync mode for a device.
    pub fn best_for_device(device: &CudaDevice) -> Self {
        if device.supports_cooperative_groups() {
            SyncMode::Cooperative
        } else {
            SyncMode::SoftwareBarrier
        }
    }

    /// Generate CUDA code for the synchronization barrier.
    pub fn barrier_code(&self) -> &'static str {
        match self {
            SyncMode::Cooperative => {
                r#"    cg::grid_group __grid = cg::this_grid();
    __grid.sync();"#
            }
            SyncMode::SoftwareBarrier => {
                r#"    __software_grid_sync(&__barrier_counter, &__barrier_gen, gridDim.x * gridDim.y * gridDim.z);"#
            }
            SyncMode::MultiLaunch => {
                // No barrier needed - sync happens between launches
                "    // (sync handled by separate kernel launches)"
            }
        }
    }
}

impl fmt::Display for SyncMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyncMode::Cooperative => write!(f, "Cooperative (grid.sync)"),
            SyncMode::SoftwareBarrier => write!(f, "Software Barrier (atomic counter)"),
            SyncMode::MultiLaunch => write!(f, "Multi-Launch (stream sync)"),
        }
    }
}

// ============================================================================
// KERNEL PHASE DEFINITION
// ============================================================================

/// Definition of a single kernel phase.
///
/// Each phase represents a distinct computation step that may require
/// synchronization with other phases.
#[derive(Debug, Clone)]
pub struct KernelPhase {
    /// Human-readable name for this phase.
    pub name: String,
    /// CUDA function name to invoke.
    pub func_name: String,
    /// Grid dimensions (blocks).
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (threads per block).
    pub block_dim: (u32, u32, u32),
    /// Shared memory size in bytes.
    pub shared_mem_bytes: u32,
    /// Whether this phase performs a reduction.
    pub has_reduction: bool,
}

impl KernelPhase {
    /// Create a new kernel phase.
    pub fn new(name: &str, func_name: &str) -> Self {
        Self {
            name: name.to_string(),
            func_name: func_name.to_string(),
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
            has_reduction: false,
        }
    }

    /// Set grid dimensions.
    pub fn with_grid_dim(mut self, x: u32, y: u32, z: u32) -> Self {
        self.grid_dim = (x, y, z);
        self
    }

    /// Set grid dimensions for 1D launch.
    pub fn with_grid_dim_1d(mut self, x: u32) -> Self {
        self.grid_dim = (x, 1, 1);
        self
    }

    /// Set block dimensions.
    pub fn with_block_dim(mut self, x: u32, y: u32, z: u32) -> Self {
        self.block_dim = (x, y, z);
        self
    }

    /// Set block dimensions for 1D launch.
    pub fn with_block_dim_1d(mut self, x: u32) -> Self {
        self.block_dim = (x, 1, 1);
        self
    }

    /// Set shared memory size.
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Mark this phase as performing a reduction.
    pub fn with_reduction(mut self) -> Self {
        self.has_reduction = true;
        self
    }

    /// Calculate total number of threads.
    pub fn total_threads(&self) -> u64 {
        let grid_size = self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64;
        let block_size =
            self.block_dim.0 as u64 * self.block_dim.1 as u64 * self.block_dim.2 as u64;
        grid_size * block_size
    }

    /// Calculate total number of blocks.
    pub fn total_blocks(&self) -> u32 {
        self.grid_dim.0 * self.grid_dim.1 * self.grid_dim.2
    }
}

// ============================================================================
// INTER-PHASE REDUCTION
// ============================================================================

/// Configuration for a reduction that occurs between phases.
///
/// This is used when phase N produces values that need to be reduced
/// before phase N+1 can use the aggregate result.
#[derive(Debug, Clone)]
pub struct InterPhaseReduction<T: ReductionScalar> {
    /// The reduction operation to perform.
    pub op: ReductionOp,
    /// Index of the phase that produces the values to reduce.
    pub source_phase: usize,
    /// Index of the phase that consumes the reduced result.
    pub target_phase: usize,
    /// Name of this reduction (for debugging).
    pub name: String,
    /// Marker for the scalar type.
    _marker: PhantomData<T>,
}

impl<T: ReductionScalar> InterPhaseReduction<T> {
    /// Create a new inter-phase reduction.
    pub fn new(op: ReductionOp, source_phase: usize, target_phase: usize) -> Self {
        Self {
            op,
            source_phase,
            target_phase,
            name: format!("{:?}_reduction_{}_to_{}", op, source_phase, target_phase),
            _marker: PhantomData,
        }
    }

    /// Set a custom name for this reduction.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }
}

// ============================================================================
// MULTI-PHASE CONFIGURATION
// ============================================================================

/// Configuration for multi-phase kernel execution.
#[derive(Debug, Clone)]
pub struct MultiPhaseConfig {
    /// Synchronization mode between phases.
    pub sync_mode: SyncMode,
    /// Ordered list of kernel phases.
    pub phases: Vec<KernelPhase>,
    /// Whether to auto-select the best sync mode for the device.
    pub auto_select_sync: bool,
}

impl Default for MultiPhaseConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiPhaseConfig {
    /// Create a new multi-phase configuration.
    pub fn new() -> Self {
        Self {
            sync_mode: SyncMode::default(),
            phases: Vec::new(),
            auto_select_sync: true,
        }
    }

    /// Set the synchronization mode.
    pub fn with_sync_mode(mut self, mode: SyncMode) -> Self {
        self.sync_mode = mode;
        self.auto_select_sync = false;
        self
    }

    /// Enable automatic sync mode selection based on device capabilities.
    pub fn with_auto_sync(mut self) -> Self {
        self.auto_select_sync = true;
        self
    }

    /// Add a phase to the execution sequence.
    pub fn add_phase(mut self, phase: KernelPhase) -> Self {
        self.phases.push(phase);
        self
    }

    /// Add multiple phases at once.
    pub fn with_phases(mut self, phases: Vec<KernelPhase>) -> Self {
        self.phases = phases;
        self
    }

    /// Get the number of phases.
    pub fn num_phases(&self) -> usize {
        self.phases.len()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.phases.is_empty() {
            return Err(RingKernelError::InvalidConfig(
                "MultiPhaseConfig has no phases defined".to_string(),
            ));
        }

        for (i, phase) in self.phases.iter().enumerate() {
            if phase.func_name.is_empty() {
                return Err(RingKernelError::InvalidConfig(format!(
                    "Phase {} has empty function name",
                    i
                )));
            }
            if phase.block_dim.0 == 0 || phase.block_dim.1 == 0 || phase.block_dim.2 == 0 {
                return Err(RingKernelError::InvalidConfig(format!(
                    "Phase {} has invalid block dimensions",
                    i
                )));
            }
            if phase.grid_dim.0 == 0 || phase.grid_dim.1 == 0 || phase.grid_dim.2 == 0 {
                return Err(RingKernelError::InvalidConfig(format!(
                    "Phase {} has invalid grid dimensions",
                    i
                )));
            }
        }

        Ok(())
    }
}

// ============================================================================
// MULTI-PHASE EXECUTOR
// ============================================================================

/// Executor for multi-phase kernel sequences.
///
/// The executor handles:
/// - Phase sequencing and synchronization
/// - Reduction buffer management
/// - Sync mode selection and fallback
pub struct MultiPhaseExecutor {
    /// The CUDA device.
    device: CudaDevice,
    /// Configuration.
    config: MultiPhaseConfig,
    /// Effective sync mode after auto-selection.
    effective_sync_mode: SyncMode,
    /// Software barrier counter (for SoftwareBarrier mode).
    barrier_counter: Option<ReductionBuffer<u32>>,
    /// Barrier generation counter (for SoftwareBarrier mode).
    barrier_generation: Option<ReductionBuffer<u32>>,
}

impl MultiPhaseExecutor {
    /// Create a new multi-phase executor.
    pub fn new(device: &CudaDevice, config: MultiPhaseConfig) -> Result<Self> {
        config.validate()?;

        // Determine effective sync mode
        let effective_sync_mode = if config.auto_select_sync {
            SyncMode::best_for_device(device)
        } else if !config.sync_mode.is_supported(device) {
            return Err(RingKernelError::InvalidConfig(format!(
                "Sync mode {} not supported on device {} (CC {}.{})",
                config.sync_mode,
                device.name(),
                device.compute_capability().0,
                device.compute_capability().1
            )));
        } else {
            config.sync_mode
        };

        // Allocate barrier resources for software barrier mode
        let (barrier_counter, barrier_generation) =
            if effective_sync_mode == SyncMode::SoftwareBarrier {
                let counter = ReductionBuffer::new(device, ReductionOp::Sum)?;
                let generation = ReductionBuffer::new(device, ReductionOp::Sum)?;
                (Some(counter), Some(generation))
            } else {
                (None, None)
            };

        Ok(Self {
            device: device.clone(),
            config,
            effective_sync_mode,
            barrier_counter,
            barrier_generation,
        })
    }

    /// Get the effective synchronization mode.
    pub fn sync_mode(&self) -> SyncMode {
        self.effective_sync_mode
    }

    /// Get the configuration.
    pub fn config(&self) -> &MultiPhaseConfig {
        &self.config
    }

    /// Get the device.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Get barrier counter device pointer (for software barrier mode).
    pub fn barrier_counter_ptr(&self) -> Option<u64> {
        self.barrier_counter.as_ref().map(|b| b.device_ptr())
    }

    /// Get barrier generation device pointer (for software barrier mode).
    pub fn barrier_generation_ptr(&self) -> Option<u64> {
        self.barrier_generation.as_ref().map(|b| b.device_ptr())
    }

    /// Reset barrier counters before execution.
    pub fn reset_barriers(&self) -> Result<()> {
        if let Some(ref counter) = self.barrier_counter {
            counter.reset()?;
        }
        if let Some(ref gen) = self.barrier_generation {
            gen.reset()?;
        }
        Ok(())
    }

    /// Generate CUDA code for the software barrier function.
    ///
    /// This function implements a grid-wide barrier using atomic counters.
    /// It must be included in the kernel source when using SoftwareBarrier mode.
    pub fn software_barrier_code() -> &'static str {
        r#"
// Software grid-wide barrier using atomic counters
// Based on GPU Computing Gems Jade Edition, Chapter 39
__device__ void __software_grid_sync(
    unsigned int* barrier_counter,
    unsigned int* barrier_generation,
    unsigned int num_blocks
) {
    // Thread 0 of each block participates in the barrier
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        // Read current generation
        unsigned int gen = *barrier_generation;

        // Signal arrival
        unsigned int arrived = atomicAdd(barrier_counter, 1);

        if (arrived == num_blocks - 1) {
            // Last block to arrive: reset counter and advance generation
            *barrier_counter = 0;
            __threadfence();
            atomicAdd(barrier_generation, 1);
        } else {
            // Wait for generation to advance
            while (*barrier_generation == gen) {
                // Spin
            }
        }
    }

    // Ensure all threads in block see the barrier completion
    __syncthreads();
}
"#
    }

    /// Generate the complete preamble code for multi-phase kernels.
    ///
    /// This includes:
    /// - Cooperative groups header (if needed)
    /// - Software barrier function (if needed)
    /// - Common helper macros
    pub fn preamble_code(&self) -> String {
        let mut code = String::new();

        // Cooperative groups header
        if self.effective_sync_mode == SyncMode::Cooperative {
            code.push_str("#include <cooperative_groups.h>\n");
            code.push_str("namespace cg = cooperative_groups;\n\n");
        }

        // Software barrier
        if self.effective_sync_mode == SyncMode::SoftwareBarrier {
            code.push_str(Self::software_barrier_code());
        }

        code
    }
}

// ============================================================================
// PHASE EXECUTION STATS
// ============================================================================

/// Statistics from executing a multi-phase kernel.
#[derive(Debug, Clone, Default)]
pub struct PhaseExecutionStats {
    /// Total execution time across all phases.
    pub total_time_us: u64,
    /// Time spent in each phase.
    pub phase_times_us: Vec<u64>,
    /// Time spent in synchronization.
    pub sync_time_us: u64,
    /// Number of phases executed.
    pub phases_executed: usize,
    /// Sync mode used.
    pub sync_mode: SyncMode,
}

impl PhaseExecutionStats {
    /// Create new empty stats.
    pub fn new(num_phases: usize, sync_mode: SyncMode) -> Self {
        Self {
            total_time_us: 0,
            phase_times_us: vec![0; num_phases],
            sync_time_us: 0,
            phases_executed: 0,
            sync_mode,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_mode_display() {
        assert_eq!(
            format!("{}", SyncMode::Cooperative),
            "Cooperative (grid.sync)"
        );
        assert_eq!(
            format!("{}", SyncMode::SoftwareBarrier),
            "Software Barrier (atomic counter)"
        );
        assert_eq!(
            format!("{}", SyncMode::MultiLaunch),
            "Multi-Launch (stream sync)"
        );
    }

    #[test]
    fn test_sync_mode_barrier_code() {
        assert!(SyncMode::Cooperative.barrier_code().contains("grid.sync"));
        assert!(SyncMode::SoftwareBarrier
            .barrier_code()
            .contains("__software_grid_sync"));
        assert!(SyncMode::MultiLaunch
            .barrier_code()
            .contains("separate kernel launches"));
    }

    #[test]
    fn test_kernel_phase_builder() {
        let phase = KernelPhase::new("accumulate", "pagerank_accumulate")
            .with_grid_dim_1d(1024)
            .with_block_dim_1d(256)
            .with_shared_mem(256 * 8) // 256 doubles
            .with_reduction();

        assert_eq!(phase.name, "accumulate");
        assert_eq!(phase.func_name, "pagerank_accumulate");
        assert_eq!(phase.grid_dim, (1024, 1, 1));
        assert_eq!(phase.block_dim, (256, 1, 1));
        assert_eq!(phase.shared_mem_bytes, 2048);
        assert!(phase.has_reduction);
        assert_eq!(phase.total_threads(), 1024 * 256);
        assert_eq!(phase.total_blocks(), 1024);
    }

    #[test]
    fn test_kernel_phase_3d() {
        let phase = KernelPhase::new("stencil", "apply_stencil")
            .with_grid_dim(8, 8, 8)
            .with_block_dim(8, 8, 8);

        assert_eq!(phase.grid_dim, (8, 8, 8));
        assert_eq!(phase.block_dim, (8, 8, 8));
        assert_eq!(phase.total_threads(), 512 * 512);
        assert_eq!(phase.total_blocks(), 512);
    }

    #[test]
    fn test_inter_phase_reduction() {
        let reduction: InterPhaseReduction<f64> =
            InterPhaseReduction::new(ReductionOp::Sum, 0, 1).with_name("dangling_sum");

        assert_eq!(reduction.op, ReductionOp::Sum);
        assert_eq!(reduction.source_phase, 0);
        assert_eq!(reduction.target_phase, 1);
        assert_eq!(reduction.name, "dangling_sum");
    }

    #[test]
    fn test_multi_phase_config_builder() {
        let config = MultiPhaseConfig::new()
            .with_sync_mode(SyncMode::Cooperative)
            .add_phase(KernelPhase::new("phase1", "kernel_phase1"))
            .add_phase(KernelPhase::new("phase2", "kernel_phase2"));

        assert_eq!(config.sync_mode, SyncMode::Cooperative);
        assert_eq!(config.num_phases(), 2);
        assert!(!config.auto_select_sync);
    }

    #[test]
    fn test_multi_phase_config_validation() {
        // Empty config should fail
        let empty_config = MultiPhaseConfig::new();
        assert!(empty_config.validate().is_err());

        // Valid config should pass
        let valid_config =
            MultiPhaseConfig::new().add_phase(KernelPhase::new("phase1", "kernel_phase1"));
        assert!(valid_config.validate().is_ok());

        // Invalid block dim should fail
        let mut invalid_phase = KernelPhase::new("bad", "bad_kernel");
        invalid_phase.block_dim = (0, 1, 1);
        let invalid_config = MultiPhaseConfig::new().add_phase(invalid_phase);
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_software_barrier_code_generation() {
        let code = MultiPhaseExecutor::software_barrier_code();
        assert!(code.contains("__software_grid_sync"));
        assert!(code.contains("atomicAdd"));
        assert!(code.contains("barrier_counter"));
        assert!(code.contains("barrier_generation"));
        assert!(code.contains("__threadfence"));
        assert!(code.contains("__syncthreads"));
    }

    #[test]
    fn test_phase_execution_stats() {
        let stats = PhaseExecutionStats::new(3, SyncMode::Cooperative);
        assert_eq!(stats.phase_times_us.len(), 3);
        assert_eq!(stats.phases_executed, 0);
        assert_eq!(stats.sync_mode, SyncMode::Cooperative);
    }

    #[test]
    fn test_sync_mode_default() {
        let mode = SyncMode::default();
        assert_eq!(mode, SyncMode::Cooperative);
    }

    #[test]
    fn test_multi_phase_config_with_phases() {
        let phases = vec![
            KernelPhase::new("a", "kernel_a"),
            KernelPhase::new("b", "kernel_b"),
            KernelPhase::new("c", "kernel_c"),
        ];

        let config = MultiPhaseConfig::new().with_phases(phases);
        assert_eq!(config.num_phases(), 3);
        assert_eq!(config.phases[0].name, "a");
        assert_eq!(config.phases[1].name, "b");
        assert_eq!(config.phases[2].name, "c");
    }

    #[test]
    fn test_auto_sync_mode() {
        let config = MultiPhaseConfig::new().with_auto_sync();
        assert!(config.auto_select_sync);
    }
}
