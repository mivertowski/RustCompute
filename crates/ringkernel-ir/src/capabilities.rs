//! Backend capabilities for IR code generation.
//!
//! Tracks what features are available on different GPU backends.

use std::collections::HashSet;

/// Capability flags for GPU features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CapabilityFlag {
    /// 64-bit floating point (f64).
    Float64,
    /// 64-bit integers.
    Int64,
    /// 64-bit atomics.
    Atomic64,
    /// Cooperative groups / grid sync.
    CooperativeGroups,
    /// Subgroup/warp operations.
    Subgroups,
    /// Subgroup shuffle.
    SubgroupShuffle,
    /// Subgroup vote.
    SubgroupVote,
    /// Subgroup reduce.
    SubgroupReduce,
    /// Shared memory.
    SharedMemory,
    /// Dynamic shared memory.
    DynamicSharedMemory,
    /// Indirect command buffers.
    IndirectCommands,
    /// Persistent kernels.
    PersistentKernels,
    /// Half precision (f16).
    Float16,
    /// Tensor cores / matrix ops.
    TensorCores,
    /// Ray tracing.
    RayTracing,
    /// Bindless textures.
    BindlessTextures,
    /// Unified memory.
    UnifiedMemory,
    /// Multi-GPU support.
    MultiGpu,
}

impl std::fmt::Display for CapabilityFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CapabilityFlag::Float64 => write!(f, "float64"),
            CapabilityFlag::Int64 => write!(f, "int64"),
            CapabilityFlag::Atomic64 => write!(f, "atomic64"),
            CapabilityFlag::CooperativeGroups => write!(f, "cooperative_groups"),
            CapabilityFlag::Subgroups => write!(f, "subgroups"),
            CapabilityFlag::SubgroupShuffle => write!(f, "subgroup_shuffle"),
            CapabilityFlag::SubgroupVote => write!(f, "subgroup_vote"),
            CapabilityFlag::SubgroupReduce => write!(f, "subgroup_reduce"),
            CapabilityFlag::SharedMemory => write!(f, "shared_memory"),
            CapabilityFlag::DynamicSharedMemory => write!(f, "dynamic_shared_memory"),
            CapabilityFlag::IndirectCommands => write!(f, "indirect_commands"),
            CapabilityFlag::PersistentKernels => write!(f, "persistent_kernels"),
            CapabilityFlag::Float16 => write!(f, "float16"),
            CapabilityFlag::TensorCores => write!(f, "tensor_cores"),
            CapabilityFlag::RayTracing => write!(f, "ray_tracing"),
            CapabilityFlag::BindlessTextures => write!(f, "bindless_textures"),
            CapabilityFlag::UnifiedMemory => write!(f, "unified_memory"),
            CapabilityFlag::MultiGpu => write!(f, "multi_gpu"),
        }
    }
}

/// Set of capabilities required or available.
#[derive(Debug, Clone, Default)]
pub struct Capabilities {
    flags: HashSet<CapabilityFlag>,
}

impl Capabilities {
    /// Create empty capabilities.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with specific flags.
    pub fn with_flags(flags: impl IntoIterator<Item = CapabilityFlag>) -> Self {
        Self {
            flags: flags.into_iter().collect(),
        }
    }

    /// Add a capability.
    pub fn add(&mut self, flag: CapabilityFlag) {
        self.flags.insert(flag);
    }

    /// Remove a capability.
    pub fn remove(&mut self, flag: CapabilityFlag) {
        self.flags.remove(&flag);
    }

    /// Check if capability is present.
    pub fn has(&self, flag: CapabilityFlag) -> bool {
        self.flags.contains(&flag)
    }

    /// Check if all required capabilities are satisfied.
    pub fn satisfies(&self, required: &Capabilities) -> bool {
        required.flags.iter().all(|f| self.flags.contains(f))
    }

    /// Get missing capabilities.
    pub fn missing(&self, required: &Capabilities) -> Vec<CapabilityFlag> {
        required
            .flags
            .iter()
            .filter(|f| !self.flags.contains(f))
            .copied()
            .collect()
    }

    /// Merge with another set.
    pub fn merge(&mut self, other: &Capabilities) {
        self.flags.extend(&other.flags);
    }

    /// Get all flags.
    pub fn flags(&self) -> &HashSet<CapabilityFlag> {
        &self.flags
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }
}

/// Backend-specific capabilities.
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Backend name.
    pub name: String,
    /// Available capabilities.
    pub capabilities: Capabilities,
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
    /// Maximum shared memory per block (bytes).
    pub max_shared_memory: u32,
    /// Warp/wavefront size.
    pub warp_size: u32,
    /// Maximum registers per thread.
    pub max_registers: u32,
}

impl BackendCapabilities {
    /// Create CUDA capabilities (SM 8.0+).
    pub fn cuda_sm80() -> Self {
        Self {
            name: "CUDA SM 8.0".to_string(),
            capabilities: Capabilities::with_flags([
                CapabilityFlag::Float64,
                CapabilityFlag::Int64,
                CapabilityFlag::Atomic64,
                CapabilityFlag::CooperativeGroups,
                CapabilityFlag::Subgroups,
                CapabilityFlag::SubgroupShuffle,
                CapabilityFlag::SubgroupVote,
                CapabilityFlag::SubgroupReduce,
                CapabilityFlag::SharedMemory,
                CapabilityFlag::DynamicSharedMemory,
                CapabilityFlag::PersistentKernels,
                CapabilityFlag::Float16,
                CapabilityFlag::TensorCores,
                CapabilityFlag::UnifiedMemory,
            ]),
            max_threads_per_block: 1024,
            max_shared_memory: 163840, // 160 KB
            warp_size: 32,
            max_registers: 255,
        }
    }

    /// Create WebGPU capabilities (baseline).
    pub fn wgpu_baseline() -> Self {
        Self {
            name: "WebGPU Baseline".to_string(),
            capabilities: Capabilities::with_flags([
                CapabilityFlag::SharedMemory,
                CapabilityFlag::Float16,
            ]),
            max_threads_per_block: 256,
            max_shared_memory: 16384, // 16 KB
            warp_size: 32,            // Varies by hardware
            max_registers: 128,
        }
    }

    /// Create WebGPU capabilities with subgroups.
    pub fn wgpu_with_subgroups() -> Self {
        let mut caps = Self::wgpu_baseline();
        caps.name = "WebGPU with Subgroups".to_string();
        caps.capabilities.add(CapabilityFlag::Subgroups);
        caps.capabilities.add(CapabilityFlag::SubgroupVote);
        caps
    }

    /// Create Metal capabilities (Apple Silicon).
    pub fn metal_apple_silicon() -> Self {
        Self {
            name: "Metal Apple Silicon".to_string(),
            capabilities: Capabilities::with_flags([
                CapabilityFlag::Int64,
                CapabilityFlag::Subgroups,
                CapabilityFlag::SubgroupShuffle,
                CapabilityFlag::SubgroupVote,
                CapabilityFlag::SubgroupReduce,
                CapabilityFlag::SharedMemory,
                CapabilityFlag::DynamicSharedMemory,
                CapabilityFlag::IndirectCommands,
                CapabilityFlag::Float16,
                CapabilityFlag::UnifiedMemory,
            ]),
            max_threads_per_block: 1024,
            max_shared_memory: 32768, // 32 KB
            warp_size: 32,            // SIMD width
            max_registers: 256,
        }
    }

    /// Check if backend supports required capabilities.
    pub fn supports(&self, required: &Capabilities) -> bool {
        self.capabilities.satisfies(required)
    }

    /// Get unsupported capabilities.
    pub fn unsupported(&self, required: &Capabilities) -> Vec<CapabilityFlag> {
        self.capabilities.missing(required)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capabilities_add_has() {
        let mut caps = Capabilities::new();
        assert!(!caps.has(CapabilityFlag::Float64));

        caps.add(CapabilityFlag::Float64);
        assert!(caps.has(CapabilityFlag::Float64));
    }

    #[test]
    fn test_capabilities_satisfies() {
        let available = Capabilities::with_flags([
            CapabilityFlag::Float64,
            CapabilityFlag::Int64,
            CapabilityFlag::SharedMemory,
        ]);

        let required1 = Capabilities::with_flags([CapabilityFlag::Float64]);
        assert!(available.satisfies(&required1));

        let required2 = Capabilities::with_flags([CapabilityFlag::CooperativeGroups]);
        assert!(!available.satisfies(&required2));
    }

    #[test]
    fn test_capabilities_missing() {
        let available = Capabilities::with_flags([CapabilityFlag::Float64]);
        let required = Capabilities::with_flags([CapabilityFlag::Float64, CapabilityFlag::Int64]);

        let missing = available.missing(&required);
        assert_eq!(missing.len(), 1);
        assert!(missing.contains(&CapabilityFlag::Int64));
    }

    #[test]
    fn test_cuda_capabilities() {
        let cuda = BackendCapabilities::cuda_sm80();
        assert!(cuda.capabilities.has(CapabilityFlag::Float64));
        assert!(cuda.capabilities.has(CapabilityFlag::CooperativeGroups));
        assert!(cuda.capabilities.has(CapabilityFlag::PersistentKernels));
    }

    #[test]
    fn test_wgpu_capabilities() {
        let wgpu = BackendCapabilities::wgpu_baseline();
        assert!(!wgpu.capabilities.has(CapabilityFlag::Float64));
        assert!(wgpu.capabilities.has(CapabilityFlag::SharedMemory));
    }

    #[test]
    fn test_metal_capabilities() {
        let metal = BackendCapabilities::metal_apple_silicon();
        assert!(metal.capabilities.has(CapabilityFlag::UnifiedMemory));
        assert!(!metal.capabilities.has(CapabilityFlag::Float64)); // Metal doesn't support f64
    }

    #[test]
    fn test_backend_supports() {
        let cuda = BackendCapabilities::cuda_sm80();
        let wgpu = BackendCapabilities::wgpu_baseline();

        let requires_f64 = Capabilities::with_flags([CapabilityFlag::Float64]);
        assert!(cuda.supports(&requires_f64));
        assert!(!wgpu.supports(&requires_f64));
    }
}
