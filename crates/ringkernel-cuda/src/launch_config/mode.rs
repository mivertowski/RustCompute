//! Kernel mode selection based on workload characteristics.

/// Kernel execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelMode {
    /// Element-centric: 1 thread per element.
    /// Best for simple, uniform workloads.
    #[default]
    ElementCentric,

    /// Structure-of-Arrays for coalesced memory access.
    /// 2-4x improvement from eliminating strided access.
    SoA,

    /// Work-item centric with load balancing.
    /// Best for irregular workloads with varying per-element work.
    WorkItemCentric,

    /// Tiled/blocked processing for cache optimization.
    /// Processes data in cache-sized chunks.
    Tiled {
        /// Tile dimensions (x, y, z).
        tile_size: (u32, u32, u32),
    },

    /// Warp-cooperative processing.
    /// Multiple threads per element for high-degree elements.
    WarpCooperative,

    /// Automatic mode selection based on workload profile.
    Auto,
}

impl KernelMode {
    /// Creates a Tiled mode with the given tile size.
    #[must_use]
    pub fn tiled(x: u32, y: u32, z: u32) -> Self {
        KernelMode::Tiled {
            tile_size: (x, y, z),
        }
    }

    /// Creates a Tiled mode with a 2D tile size.
    #[must_use]
    pub fn tiled_2d(x: u32, y: u32) -> Self {
        KernelMode::Tiled {
            tile_size: (x, y, 1),
        }
    }

    /// Creates a Tiled mode with a 1D tile size.
    #[must_use]
    pub fn tiled_1d(x: u32) -> Self {
        KernelMode::Tiled {
            tile_size: (x, 1, 1),
        }
    }

    /// Returns the tile size if this is a Tiled mode.
    #[must_use]
    pub fn tile_size(&self) -> Option<(u32, u32, u32)> {
        match self {
            KernelMode::Tiled { tile_size } => Some(*tile_size),
            _ => None,
        }
    }
}

/// Memory access pattern for workload analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccessPattern {
    /// Coalesced, sequential access.
    /// Threads access adjacent memory locations.
    #[default]
    Coalesced,

    /// Stencil access pattern.
    /// Each thread accesses a neighborhood of elements.
    Stencil {
        /// Stencil radius in each dimension.
        radius: u32,
    },

    /// Irregular/random access pattern.
    /// Access locations are data-dependent.
    Irregular,

    /// Reduction pattern.
    /// Threads cooperatively reduce values.
    Reduction,

    /// Scatter pattern.
    /// Each thread writes to a computed location.
    Scatter,

    /// Gather pattern.
    /// Each thread reads from multiple computed locations.
    Gather,
}

impl AccessPattern {
    /// Creates a stencil pattern with the given radius.
    #[must_use]
    pub fn stencil(radius: u32) -> Self {
        AccessPattern::Stencil { radius }
    }

    /// Returns the stencil radius if this is a Stencil pattern.
    #[must_use]
    pub fn stencil_radius(&self) -> Option<u32> {
        match self {
            AccessPattern::Stencil { radius } => Some(*radius),
            _ => None,
        }
    }
}

/// Workload profile for kernel mode selection.
#[derive(Debug, Clone)]
pub struct WorkloadProfile {
    /// Number of elements to process.
    pub element_count: usize,
    /// Bytes per element.
    pub bytes_per_element: usize,
    /// Memory access pattern.
    pub access_pattern: AccessPattern,
    /// Compute intensity (FLOPs per byte).
    pub compute_intensity: f32,
    /// Maximum per-element work variation (for load balancing decisions).
    pub max_work_variation: f32,
    /// Whether elements can be processed independently.
    pub independent_elements: bool,
}

impl WorkloadProfile {
    /// Creates a new workload profile.
    #[must_use]
    pub fn new(element_count: usize, bytes_per_element: usize) -> Self {
        Self {
            element_count,
            bytes_per_element,
            access_pattern: AccessPattern::default(),
            compute_intensity: 1.0,
            max_work_variation: 1.0,
            independent_elements: true,
        }
    }

    /// Builder method to set the access pattern.
    #[must_use]
    pub fn with_access_pattern(mut self, pattern: AccessPattern) -> Self {
        self.access_pattern = pattern;
        self
    }

    /// Builder method to set compute intensity.
    #[must_use]
    pub fn with_compute_intensity(mut self, intensity: f32) -> Self {
        self.compute_intensity = intensity;
        self
    }

    /// Builder method to set work variation.
    #[must_use]
    pub fn with_work_variation(mut self, variation: f32) -> Self {
        self.max_work_variation = variation;
        self
    }

    /// Builder method to set element independence.
    #[must_use]
    pub fn with_independent_elements(mut self, independent: bool) -> Self {
        self.independent_elements = independent;
        self
    }

    /// Returns total working set size in bytes.
    #[must_use]
    pub fn working_set_bytes(&self) -> usize {
        self.element_count * self.bytes_per_element
    }

    /// Returns whether workload has high work variation.
    #[must_use]
    pub fn is_irregular(&self) -> bool {
        self.max_work_variation > 10.0 || matches!(self.access_pattern, AccessPattern::Irregular)
    }
}

/// GPU architecture characteristics for kernel optimization.
#[derive(Debug, Clone, Copy)]
pub struct GpuArchitecture {
    /// L2 cache size in bytes.
    pub l2_cache_bytes: usize,
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
    /// Maximum threads per SM.
    pub max_threads_per_sm: u32,
    /// Shared memory per SM in bytes.
    pub shared_mem_per_sm: usize,
    /// Compute capability (major, minor).
    pub compute_capability: (u32, u32),
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
    /// Warp size (typically 32).
    pub warp_size: u32,
}

impl Default for GpuArchitecture {
    fn default() -> Self {
        // Default to Ampere-class GPU
        Self {
            l2_cache_bytes: 40 * 1024 * 1024, // 40 MB
            sm_count: 84,
            max_threads_per_sm: 2048,
            shared_mem_per_sm: 164 * 1024, // 164 KB
            compute_capability: (8, 0),
            max_threads_per_block: 1024,
            warp_size: 32,
        }
    }
}

impl GpuArchitecture {
    /// Creates an architecture profile for common GPU classes.
    #[must_use]
    pub fn volta() -> Self {
        Self {
            l2_cache_bytes: 6 * 1024 * 1024, // 6 MB
            sm_count: 80,
            max_threads_per_sm: 2048,
            shared_mem_per_sm: 96 * 1024, // 96 KB configurable
            compute_capability: (7, 0),
            max_threads_per_block: 1024,
            warp_size: 32,
        }
    }

    /// Creates an architecture profile for Ampere GPUs.
    #[must_use]
    pub fn ampere() -> Self {
        Self::default()
    }

    /// Creates an architecture profile for Ada Lovelace GPUs.
    #[must_use]
    pub fn ada() -> Self {
        Self {
            l2_cache_bytes: 72 * 1024 * 1024, // 72 MB
            sm_count: 128,
            max_threads_per_sm: 2048,
            shared_mem_per_sm: 100 * 1024, // 100 KB
            compute_capability: (8, 9),
            max_threads_per_block: 1024,
            warp_size: 32,
        }
    }

    /// Creates an architecture profile for Hopper GPUs.
    #[must_use]
    pub fn hopper() -> Self {
        Self {
            l2_cache_bytes: 50 * 1024 * 1024, // 50 MB
            sm_count: 132,
            max_threads_per_sm: 2048,
            shared_mem_per_sm: 228 * 1024, // 228 KB
            compute_capability: (9, 0),
            max_threads_per_block: 1024,
            warp_size: 32,
        }
    }

    /// Returns total GPU threads available.
    #[must_use]
    pub fn total_threads(&self) -> u32 {
        self.sm_count * self.max_threads_per_sm
    }

    /// Returns whether cooperative groups are supported.
    #[must_use]
    pub fn supports_cooperative_groups(&self) -> bool {
        self.compute_capability.0 >= 6
    }
}

/// Kernel mode selector based on workload profile and GPU architecture.
#[derive(Debug, Clone)]
pub struct KernelModeSelector {
    /// GPU architecture.
    architecture: GpuArchitecture,
    /// L2 cache threshold multiplier for tiled mode.
    l2_threshold_multiplier: f32,
    /// Work variation threshold for load balancing.
    load_balance_threshold: f32,
}

impl KernelModeSelector {
    /// Creates a new selector for the given architecture.
    #[must_use]
    pub fn new(architecture: GpuArchitecture) -> Self {
        Self {
            architecture,
            l2_threshold_multiplier: 2.0,
            load_balance_threshold: 10.0,
        }
    }

    /// Creates a selector with default Ampere architecture.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(GpuArchitecture::default())
    }

    /// Builder method to set L2 threshold multiplier.
    #[must_use]
    pub fn with_l2_threshold(mut self, multiplier: f32) -> Self {
        self.l2_threshold_multiplier = multiplier;
        self
    }

    /// Builder method to set load balance threshold.
    #[must_use]
    pub fn with_load_balance_threshold(mut self, threshold: f32) -> Self {
        self.load_balance_threshold = threshold;
        self
    }

    /// Selects the optimal kernel mode for a workload.
    #[must_use]
    pub fn select(&self, profile: &WorkloadProfile) -> KernelMode {
        // Check for high work variation (needs load balancing)
        if profile.max_work_variation > self.load_balance_threshold {
            return KernelMode::WorkItemCentric;
        }

        // Check for stencil access pattern
        if let AccessPattern::Stencil { radius } = profile.access_pattern {
            let tile_size = self.tile_size_for_stencil(radius);
            return KernelMode::Tiled { tile_size };
        }

        // Check working set vs L2 cache
        let working_set = profile.working_set_bytes();
        let l2_size = self.architecture.l2_cache_bytes;

        if working_set > (l2_size as f32 * self.l2_threshold_multiplier) as usize {
            // Large working set: use tiled for cache blocking
            return KernelMode::tiled_1d(256);
        }

        if working_set > l2_size {
            // Moderate overflow: SoA for bandwidth efficiency
            return KernelMode::SoA;
        }

        // Working set fits in L2
        if matches!(profile.access_pattern, AccessPattern::Reduction) {
            return KernelMode::WarpCooperative;
        }

        if profile.element_count > 1000 {
            KernelMode::SoA
        } else {
            KernelMode::ElementCentric
        }
    }

    /// Computes tile size for stencil operations.
    fn tile_size_for_stencil(&self, radius: u32) -> (u32, u32, u32) {
        // Tile size should account for halo region
        let effective_tile = 16 - 2 * radius;
        let tile = effective_tile.max(4);
        (tile, tile, 1)
    }

    /// Returns recommended block size for the given mode.
    #[must_use]
    pub fn recommended_block_size(&self, mode: KernelMode) -> (u32, u32, u32) {
        match mode {
            KernelMode::ElementCentric => (256, 1, 1),
            KernelMode::SoA => (256, 1, 1),
            KernelMode::WorkItemCentric => (128, 1, 1),
            KernelMode::Tiled { tile_size } => tile_size,
            KernelMode::WarpCooperative => (128, 1, 1),
            KernelMode::Auto => (256, 1, 1),
        }
    }

    /// Returns recommended grid size for the given mode and element count.
    #[must_use]
    pub fn recommended_grid_size(&self, mode: KernelMode, element_count: usize) -> (u32, u32, u32) {
        let (bx, by, bz) = self.recommended_block_size(mode);
        let threads_per_block = bx * by * bz;

        // Calculate grid size to cover all elements
        let blocks_needed = (element_count as u32 + threads_per_block - 1) / threads_per_block;

        // Limit to reasonable grid dimensions
        let max_grid_dim = 65535;
        let gx = blocks_needed.min(max_grid_dim);
        let gy = ((blocks_needed + max_grid_dim - 1) / max_grid_dim).min(max_grid_dim);
        let gz = 1;

        (gx, gy, gz)
    }

    /// Returns the GPU architecture.
    #[must_use]
    pub fn architecture(&self) -> &GpuArchitecture {
        &self.architecture
    }

    /// Returns a complete launch configuration.
    #[must_use]
    pub fn launch_config(&self, profile: &WorkloadProfile) -> LaunchConfig {
        let mode = self.select(profile);
        let block_dim = self.recommended_block_size(mode);
        let grid_dim = self.recommended_grid_size(mode, profile.element_count);

        // Calculate shared memory requirements
        let shared_mem = match mode {
            KernelMode::Tiled { tile_size } => {
                let (tx, ty, tz) = tile_size;
                (tx * ty * tz * profile.bytes_per_element as u32) as usize
            }
            KernelMode::WarpCooperative => {
                // Shared memory for warp-level reduction
                block_dim.0 as usize * std::mem::size_of::<f32>()
            }
            _ => 0,
        };

        LaunchConfig {
            mode,
            grid_dim,
            block_dim,
            shared_mem_bytes: shared_mem,
            stream_id: None,
        }
    }
}

/// Complete kernel launch configuration.
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Selected kernel mode.
    pub mode: KernelMode,
    /// Grid dimensions (blocks).
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (threads).
    pub block_dim: (u32, u32, u32),
    /// Shared memory in bytes.
    pub shared_mem_bytes: usize,
    /// Optional stream ID.
    pub stream_id: Option<usize>,
}

impl LaunchConfig {
    /// Creates a simple 1D launch configuration.
    #[must_use]
    pub fn simple_1d(element_count: usize, threads_per_block: u32) -> Self {
        let blocks = (element_count as u32 + threads_per_block - 1) / threads_per_block;
        Self {
            mode: KernelMode::ElementCentric,
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
            stream_id: None,
        }
    }

    /// Builder method to set stream ID.
    #[must_use]
    pub fn with_stream(mut self, stream_id: usize) -> Self {
        self.stream_id = Some(stream_id);
        self
    }

    /// Builder method to set shared memory.
    #[must_use]
    pub fn with_shared_mem(mut self, bytes: usize) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Returns total threads.
    #[must_use]
    pub fn total_threads(&self) -> u64 {
        let blocks = self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64;
        let threads_per_block =
            self.block_dim.0 as u64 * self.block_dim.1 as u64 * self.block_dim.2 as u64;
        blocks * threads_per_block
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_mode_tiled() {
        let mode = KernelMode::tiled(16, 16, 1);
        assert_eq!(mode.tile_size(), Some((16, 16, 1)));
    }

    #[test]
    fn test_access_pattern_stencil() {
        let pattern = AccessPattern::stencil(1);
        assert_eq!(pattern.stencil_radius(), Some(1));
    }

    #[test]
    fn test_workload_profile() {
        let profile = WorkloadProfile::new(1_000_000, 64)
            .with_access_pattern(AccessPattern::Coalesced)
            .with_compute_intensity(10.0);

        assert_eq!(profile.element_count, 1_000_000);
        assert_eq!(profile.bytes_per_element, 64);
        assert_eq!(profile.working_set_bytes(), 64_000_000);
    }

    #[test]
    fn test_gpu_architecture_defaults() {
        let arch = GpuArchitecture::default();
        assert!(arch.l2_cache_bytes > 0);
        assert!(arch.sm_count > 0);
        assert!(arch.supports_cooperative_groups());
    }

    #[test]
    fn test_kernel_mode_selector_simple() {
        let selector = KernelModeSelector::with_defaults();

        // Small workload should use ElementCentric or SoA
        let small_profile = WorkloadProfile::new(100, 8);
        let mode = selector.select(&small_profile);
        assert!(matches!(mode, KernelMode::ElementCentric));
    }

    #[test]
    fn test_kernel_mode_selector_large() {
        let selector = KernelModeSelector::with_defaults();

        // Large workload should use Tiled or SoA
        let large_profile = WorkloadProfile::new(10_000_000, 64);
        let mode = selector.select(&large_profile);
        assert!(matches!(mode, KernelMode::Tiled { .. } | KernelMode::SoA));
    }

    #[test]
    fn test_kernel_mode_selector_stencil() {
        let selector = KernelModeSelector::with_defaults();

        let stencil_profile =
            WorkloadProfile::new(1_000_000, 4).with_access_pattern(AccessPattern::stencil(1));

        let mode = selector.select(&stencil_profile);
        assert!(matches!(mode, KernelMode::Tiled { .. }));
    }

    #[test]
    fn test_kernel_mode_selector_irregular() {
        let selector = KernelModeSelector::with_defaults();

        let irregular_profile = WorkloadProfile::new(1_000_000, 64).with_work_variation(100.0);

        let mode = selector.select(&irregular_profile);
        assert_eq!(mode, KernelMode::WorkItemCentric);
    }

    #[test]
    fn test_recommended_block_size() {
        let selector = KernelModeSelector::with_defaults();

        let (bx, by, bz) = selector.recommended_block_size(KernelMode::ElementCentric);
        assert_eq!(bx, 256);
        assert_eq!(by, 1);
        assert_eq!(bz, 1);
    }

    #[test]
    fn test_recommended_grid_size() {
        let selector = KernelModeSelector::with_defaults();

        let (gx, _, _) = selector.recommended_grid_size(KernelMode::ElementCentric, 1_000_000);
        // 1M elements / 256 threads = 3907 blocks (rounded up)
        assert!(gx >= 3906);
    }

    #[test]
    fn test_launch_config() {
        let selector = KernelModeSelector::with_defaults();
        let profile = WorkloadProfile::new(1_000_000, 64);

        let config = selector.launch_config(&profile);
        assert!(config.total_threads() >= 1_000_000);
    }

    #[test]
    fn test_launch_config_simple() {
        let config = LaunchConfig::simple_1d(1000, 256);
        assert_eq!(config.grid_dim.0, 4); // ceil(1000/256) = 4
        assert_eq!(config.block_dim.0, 256);
    }
}
