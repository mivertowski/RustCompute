//! Intrinsic registry for WGSL code generation.
//!
//! Maps Rust DSL function calls to WGSL intrinsics.

use std::collections::HashMap;

/// WGSL intrinsic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgslIntrinsic {
    // Thread/workgroup indices
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,
    GlobalInvocationIdX,
    GlobalInvocationIdY,
    GlobalInvocationIdZ,
    NumWorkgroupsX,
    NumWorkgroupsY,
    NumWorkgroupsZ,

    // Workgroup size constants
    WorkgroupSizeX,
    WorkgroupSizeY,
    WorkgroupSizeZ,

    // Synchronization
    WorkgroupBarrier,
    StorageBarrier,

    // Atomics
    AtomicAdd,
    AtomicSub,
    AtomicMin,
    AtomicMax,
    AtomicExchange,
    AtomicCompareExchangeWeak,
    AtomicLoad,
    AtomicStore,

    // Math - single argument
    Sqrt,
    InverseSqrt,
    Abs,
    Floor,
    Ceil,
    Round,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,

    // Math - multiple arguments
    Pow,
    Min,
    Max,
    Clamp,
    Fma,
    Mix,

    // Subgroup operations (require extensions)
    SubgroupShuffle,
    SubgroupShuffleUp,
    SubgroupShuffleDown,
    SubgroupShuffleXor,
    SubgroupBallot,
    SubgroupAll,
    SubgroupAny,
    SubgroupInvocationId,
    SubgroupSize,
}

impl WgslIntrinsic {
    /// Get the WGSL code representation for this intrinsic.
    pub fn to_wgsl(&self) -> &'static str {
        match self {
            // These are builtins accessed via variables, not function calls
            WgslIntrinsic::LocalInvocationIdX => "local_invocation_id.x",
            WgslIntrinsic::LocalInvocationIdY => "local_invocation_id.y",
            WgslIntrinsic::LocalInvocationIdZ => "local_invocation_id.z",
            WgslIntrinsic::WorkgroupIdX => "workgroup_id.x",
            WgslIntrinsic::WorkgroupIdY => "workgroup_id.y",
            WgslIntrinsic::WorkgroupIdZ => "workgroup_id.z",
            WgslIntrinsic::GlobalInvocationIdX => "global_invocation_id.x",
            WgslIntrinsic::GlobalInvocationIdY => "global_invocation_id.y",
            WgslIntrinsic::GlobalInvocationIdZ => "global_invocation_id.z",
            WgslIntrinsic::NumWorkgroupsX => "num_workgroups.x",
            WgslIntrinsic::NumWorkgroupsY => "num_workgroups.y",
            WgslIntrinsic::NumWorkgroupsZ => "num_workgroups.z",

            // Workgroup size constants (substituted at transpile time)
            WgslIntrinsic::WorkgroupSizeX => "WORKGROUP_SIZE_X",
            WgslIntrinsic::WorkgroupSizeY => "WORKGROUP_SIZE_Y",
            WgslIntrinsic::WorkgroupSizeZ => "WORKGROUP_SIZE_Z",

            // Synchronization
            WgslIntrinsic::WorkgroupBarrier => "workgroupBarrier()",
            WgslIntrinsic::StorageBarrier => "storageBarrier()",

            // Atomics - these need arguments, so just return the function name
            WgslIntrinsic::AtomicAdd => "atomicAdd",
            WgslIntrinsic::AtomicSub => "atomicSub",
            WgslIntrinsic::AtomicMin => "atomicMin",
            WgslIntrinsic::AtomicMax => "atomicMax",
            WgslIntrinsic::AtomicExchange => "atomicExchange",
            WgslIntrinsic::AtomicCompareExchangeWeak => "atomicCompareExchangeWeak",
            WgslIntrinsic::AtomicLoad => "atomicLoad",
            WgslIntrinsic::AtomicStore => "atomicStore",

            // Math functions
            WgslIntrinsic::Sqrt => "sqrt",
            WgslIntrinsic::InverseSqrt => "inverseSqrt",
            WgslIntrinsic::Abs => "abs",
            WgslIntrinsic::Floor => "floor",
            WgslIntrinsic::Ceil => "ceil",
            WgslIntrinsic::Round => "round",
            WgslIntrinsic::Sin => "sin",
            WgslIntrinsic::Cos => "cos",
            WgslIntrinsic::Tan => "tan",
            WgslIntrinsic::Exp => "exp",
            WgslIntrinsic::Log => "log",
            WgslIntrinsic::Pow => "pow",
            WgslIntrinsic::Min => "min",
            WgslIntrinsic::Max => "max",
            WgslIntrinsic::Clamp => "clamp",
            WgslIntrinsic::Fma => "fma",
            WgslIntrinsic::Mix => "mix",

            // Subgroup operations
            WgslIntrinsic::SubgroupShuffle => "subgroupShuffle",
            WgslIntrinsic::SubgroupShuffleUp => "subgroupShuffleUp",
            WgslIntrinsic::SubgroupShuffleDown => "subgroupShuffleDown",
            WgslIntrinsic::SubgroupShuffleXor => "subgroupShuffleXor",
            WgslIntrinsic::SubgroupBallot => "subgroupBallot",
            WgslIntrinsic::SubgroupAll => "subgroupAll",
            WgslIntrinsic::SubgroupAny => "subgroupAny",
            WgslIntrinsic::SubgroupInvocationId => "subgroup_invocation_id",
            WgslIntrinsic::SubgroupSize => "subgroup_size",
        }
    }

    /// Check if this intrinsic requires the subgroup extension.
    pub fn requires_subgroup_extension(&self) -> bool {
        matches!(
            self,
            WgslIntrinsic::SubgroupShuffle
                | WgslIntrinsic::SubgroupShuffleUp
                | WgslIntrinsic::SubgroupShuffleDown
                | WgslIntrinsic::SubgroupShuffleXor
                | WgslIntrinsic::SubgroupBallot
                | WgslIntrinsic::SubgroupAll
                | WgslIntrinsic::SubgroupAny
                | WgslIntrinsic::SubgroupInvocationId
                | WgslIntrinsic::SubgroupSize
        )
    }
}

/// Registry mapping Rust DSL function names to WGSL intrinsics.
pub struct IntrinsicRegistry {
    mappings: HashMap<&'static str, WgslIntrinsic>,
}

impl Default for IntrinsicRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl IntrinsicRegistry {
    /// Create a new intrinsic registry with all standard mappings.
    pub fn new() -> Self {
        let mut mappings = HashMap::new();

        // Thread/workgroup indices
        mappings.insert("thread_idx_x", WgslIntrinsic::LocalInvocationIdX);
        mappings.insert("thread_idx_y", WgslIntrinsic::LocalInvocationIdY);
        mappings.insert("thread_idx_z", WgslIntrinsic::LocalInvocationIdZ);
        mappings.insert("block_idx_x", WgslIntrinsic::WorkgroupIdX);
        mappings.insert("block_idx_y", WgslIntrinsic::WorkgroupIdY);
        mappings.insert("block_idx_z", WgslIntrinsic::WorkgroupIdZ);
        mappings.insert("global_thread_id", WgslIntrinsic::GlobalInvocationIdX);
        mappings.insert("global_thread_id_y", WgslIntrinsic::GlobalInvocationIdY);
        mappings.insert("global_thread_id_z", WgslIntrinsic::GlobalInvocationIdZ);
        mappings.insert("grid_dim_x", WgslIntrinsic::NumWorkgroupsX);
        mappings.insert("grid_dim_y", WgslIntrinsic::NumWorkgroupsY);
        mappings.insert("grid_dim_z", WgslIntrinsic::NumWorkgroupsZ);

        // Workgroup size
        mappings.insert("block_dim_x", WgslIntrinsic::WorkgroupSizeX);
        mappings.insert("block_dim_y", WgslIntrinsic::WorkgroupSizeY);
        mappings.insert("block_dim_z", WgslIntrinsic::WorkgroupSizeZ);

        // Synchronization
        mappings.insert("sync_threads", WgslIntrinsic::WorkgroupBarrier);
        mappings.insert("thread_fence", WgslIntrinsic::StorageBarrier);
        mappings.insert("thread_fence_block", WgslIntrinsic::WorkgroupBarrier);

        // Atomics
        mappings.insert("atomic_add", WgslIntrinsic::AtomicAdd);
        mappings.insert("atomic_sub", WgslIntrinsic::AtomicSub);
        mappings.insert("atomic_min", WgslIntrinsic::AtomicMin);
        mappings.insert("atomic_max", WgslIntrinsic::AtomicMax);
        mappings.insert("atomic_exchange", WgslIntrinsic::AtomicExchange);
        mappings.insert("atomic_cas", WgslIntrinsic::AtomicCompareExchangeWeak);
        mappings.insert("atomic_load", WgslIntrinsic::AtomicLoad);
        mappings.insert("atomic_store", WgslIntrinsic::AtomicStore);

        // Math functions
        mappings.insert("sqrt", WgslIntrinsic::Sqrt);
        mappings.insert("rsqrt", WgslIntrinsic::InverseSqrt);
        mappings.insert("abs", WgslIntrinsic::Abs);
        mappings.insert("floor", WgslIntrinsic::Floor);
        mappings.insert("ceil", WgslIntrinsic::Ceil);
        mappings.insert("round", WgslIntrinsic::Round);
        mappings.insert("sin", WgslIntrinsic::Sin);
        mappings.insert("cos", WgslIntrinsic::Cos);
        mappings.insert("tan", WgslIntrinsic::Tan);
        mappings.insert("exp", WgslIntrinsic::Exp);
        mappings.insert("log", WgslIntrinsic::Log);
        mappings.insert("powf", WgslIntrinsic::Pow);
        mappings.insert("min", WgslIntrinsic::Min);
        mappings.insert("max", WgslIntrinsic::Max);
        mappings.insert("clamp", WgslIntrinsic::Clamp);
        mappings.insert("fma", WgslIntrinsic::Fma);
        mappings.insert("mix", WgslIntrinsic::Mix);

        // Subgroup operations
        mappings.insert("warp_shuffle", WgslIntrinsic::SubgroupShuffle);
        mappings.insert("warp_shuffle_up", WgslIntrinsic::SubgroupShuffleUp);
        mappings.insert("warp_shuffle_down", WgslIntrinsic::SubgroupShuffleDown);
        mappings.insert("warp_shuffle_xor", WgslIntrinsic::SubgroupShuffleXor);
        mappings.insert("warp_ballot", WgslIntrinsic::SubgroupBallot);
        mappings.insert("warp_all", WgslIntrinsic::SubgroupAll);
        mappings.insert("warp_any", WgslIntrinsic::SubgroupAny);
        mappings.insert("lane_id", WgslIntrinsic::SubgroupInvocationId);
        mappings.insert("warp_size", WgslIntrinsic::SubgroupSize);

        Self { mappings }
    }

    /// Look up an intrinsic by Rust DSL function name.
    pub fn lookup(&self, name: &str) -> Option<WgslIntrinsic> {
        self.mappings.get(name).copied()
    }

    /// Check if a function name is a known intrinsic.
    pub fn is_intrinsic(&self, name: &str) -> bool {
        self.mappings.contains_key(name)
    }

    /// Get all intrinsics that require the subgroup extension.
    pub fn subgroup_intrinsics(&self) -> Vec<(&'static str, WgslIntrinsic)> {
        self.mappings
            .iter()
            .filter(|(_, intrinsic)| intrinsic.requires_subgroup_extension())
            .map(|(&name, &intrinsic)| (name, intrinsic))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_lookup() {
        let registry = IntrinsicRegistry::new();
        assert_eq!(
            registry.lookup("thread_idx_x"),
            Some(WgslIntrinsic::LocalInvocationIdX)
        );
        assert_eq!(
            registry.lookup("sync_threads"),
            Some(WgslIntrinsic::WorkgroupBarrier)
        );
        assert_eq!(registry.lookup("unknown_function"), None);
    }

    #[test]
    fn test_intrinsic_wgsl_output() {
        assert_eq!(
            WgslIntrinsic::LocalInvocationIdX.to_wgsl(),
            "local_invocation_id.x"
        );
        assert_eq!(
            WgslIntrinsic::WorkgroupBarrier.to_wgsl(),
            "workgroupBarrier()"
        );
        assert_eq!(WgslIntrinsic::Sqrt.to_wgsl(), "sqrt");
    }

    #[test]
    fn test_subgroup_extension_detection() {
        assert!(WgslIntrinsic::SubgroupShuffle.requires_subgroup_extension());
        assert!(!WgslIntrinsic::Sqrt.requires_subgroup_extension());
        assert!(!WgslIntrinsic::WorkgroupBarrier.requires_subgroup_extension());
    }
}
