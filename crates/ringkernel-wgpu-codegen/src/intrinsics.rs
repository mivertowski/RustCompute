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
    // Basic subgroup queries
    SubgroupInvocationId,
    SubgroupSize,

    // Vote/ballot operations
    SubgroupAll,
    SubgroupAny,
    SubgroupBallot,
    SubgroupElect,

    // Shuffle operations
    SubgroupShuffle,
    SubgroupShuffleUp,
    SubgroupShuffleDown,
    SubgroupShuffleXor,
    SubgroupBroadcast,
    SubgroupBroadcastFirst,

    // Arithmetic reductions
    SubgroupAdd,
    SubgroupMul,
    SubgroupMin,
    SubgroupMax,
    SubgroupAnd,
    SubgroupOr,
    SubgroupXor,

    // Inclusive/exclusive scans
    SubgroupInclusiveAdd,
    SubgroupExclusiveAdd,
    SubgroupInclusiveMul,
    SubgroupExclusiveMul,
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

            // Subgroup operations - basic queries
            WgslIntrinsic::SubgroupInvocationId => "subgroup_invocation_id",
            WgslIntrinsic::SubgroupSize => "subgroup_size",

            // Subgroup vote/ballot operations
            WgslIntrinsic::SubgroupAll => "subgroupAll",
            WgslIntrinsic::SubgroupAny => "subgroupAny",
            WgslIntrinsic::SubgroupBallot => "subgroupBallot",
            WgslIntrinsic::SubgroupElect => "subgroupElect",

            // Subgroup shuffle operations
            WgslIntrinsic::SubgroupShuffle => "subgroupShuffle",
            WgslIntrinsic::SubgroupShuffleUp => "subgroupShuffleUp",
            WgslIntrinsic::SubgroupShuffleDown => "subgroupShuffleDown",
            WgslIntrinsic::SubgroupShuffleXor => "subgroupShuffleXor",
            WgslIntrinsic::SubgroupBroadcast => "subgroupBroadcast",
            WgslIntrinsic::SubgroupBroadcastFirst => "subgroupBroadcastFirst",

            // Subgroup arithmetic reductions
            WgslIntrinsic::SubgroupAdd => "subgroupAdd",
            WgslIntrinsic::SubgroupMul => "subgroupMul",
            WgslIntrinsic::SubgroupMin => "subgroupMin",
            WgslIntrinsic::SubgroupMax => "subgroupMax",
            WgslIntrinsic::SubgroupAnd => "subgroupAnd",
            WgslIntrinsic::SubgroupOr => "subgroupOr",
            WgslIntrinsic::SubgroupXor => "subgroupXor",

            // Subgroup scan operations
            WgslIntrinsic::SubgroupInclusiveAdd => "subgroupInclusiveAdd",
            WgslIntrinsic::SubgroupExclusiveAdd => "subgroupExclusiveAdd",
            WgslIntrinsic::SubgroupInclusiveMul => "subgroupInclusiveMul",
            WgslIntrinsic::SubgroupExclusiveMul => "subgroupExclusiveMul",
        }
    }

    /// Check if this intrinsic requires the subgroup extension.
    pub fn requires_subgroup_extension(&self) -> bool {
        matches!(
            self,
            // Basic queries
            WgslIntrinsic::SubgroupInvocationId
                | WgslIntrinsic::SubgroupSize
                // Vote/ballot
                | WgslIntrinsic::SubgroupAll
                | WgslIntrinsic::SubgroupAny
                | WgslIntrinsic::SubgroupBallot
                | WgslIntrinsic::SubgroupElect
                // Shuffle
                | WgslIntrinsic::SubgroupShuffle
                | WgslIntrinsic::SubgroupShuffleUp
                | WgslIntrinsic::SubgroupShuffleDown
                | WgslIntrinsic::SubgroupShuffleXor
                | WgslIntrinsic::SubgroupBroadcast
                | WgslIntrinsic::SubgroupBroadcastFirst
                // Arithmetic
                | WgslIntrinsic::SubgroupAdd
                | WgslIntrinsic::SubgroupMul
                | WgslIntrinsic::SubgroupMin
                | WgslIntrinsic::SubgroupMax
                | WgslIntrinsic::SubgroupAnd
                | WgslIntrinsic::SubgroupOr
                | WgslIntrinsic::SubgroupXor
                // Scans
                | WgslIntrinsic::SubgroupInclusiveAdd
                | WgslIntrinsic::SubgroupExclusiveAdd
                | WgslIntrinsic::SubgroupInclusiveMul
                | WgslIntrinsic::SubgroupExclusiveMul
        )
    }

    /// Check if this is a subgroup builtin variable (not a function).
    pub fn is_subgroup_builtin(&self) -> bool {
        matches!(
            self,
            WgslIntrinsic::SubgroupInvocationId | WgslIntrinsic::SubgroupSize
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

        // Subgroup operations - basic queries
        mappings.insert("lane_id", WgslIntrinsic::SubgroupInvocationId);
        mappings.insert("subgroup_id", WgslIntrinsic::SubgroupInvocationId);
        mappings.insert(
            "subgroup_invocation_id",
            WgslIntrinsic::SubgroupInvocationId,
        );
        mappings.insert("warp_size", WgslIntrinsic::SubgroupSize);
        mappings.insert("subgroup_size", WgslIntrinsic::SubgroupSize);

        // Subgroup vote/ballot operations
        mappings.insert("subgroup_all", WgslIntrinsic::SubgroupAll);
        mappings.insert("warp_all", WgslIntrinsic::SubgroupAll);
        mappings.insert("subgroup_any", WgslIntrinsic::SubgroupAny);
        mappings.insert("warp_any", WgslIntrinsic::SubgroupAny);
        mappings.insert("subgroup_ballot", WgslIntrinsic::SubgroupBallot);
        mappings.insert("warp_ballot", WgslIntrinsic::SubgroupBallot);
        mappings.insert("subgroup_elect", WgslIntrinsic::SubgroupElect);
        mappings.insert("warp_elect", WgslIntrinsic::SubgroupElect);

        // Subgroup shuffle operations
        mappings.insert("subgroup_shuffle", WgslIntrinsic::SubgroupShuffle);
        mappings.insert("warp_shuffle", WgslIntrinsic::SubgroupShuffle);
        mappings.insert("subgroup_shuffle_up", WgslIntrinsic::SubgroupShuffleUp);
        mappings.insert("warp_shuffle_up", WgslIntrinsic::SubgroupShuffleUp);
        mappings.insert("subgroup_shuffle_down", WgslIntrinsic::SubgroupShuffleDown);
        mappings.insert("warp_shuffle_down", WgslIntrinsic::SubgroupShuffleDown);
        mappings.insert("subgroup_shuffle_xor", WgslIntrinsic::SubgroupShuffleXor);
        mappings.insert("warp_shuffle_xor", WgslIntrinsic::SubgroupShuffleXor);
        mappings.insert("subgroup_broadcast", WgslIntrinsic::SubgroupBroadcast);
        mappings.insert("warp_broadcast", WgslIntrinsic::SubgroupBroadcast);
        mappings.insert(
            "subgroup_broadcast_first",
            WgslIntrinsic::SubgroupBroadcastFirst,
        );
        mappings.insert(
            "warp_broadcast_first",
            WgslIntrinsic::SubgroupBroadcastFirst,
        );

        // Subgroup arithmetic reductions
        mappings.insert("subgroup_add", WgslIntrinsic::SubgroupAdd);
        mappings.insert("warp_reduce_add", WgslIntrinsic::SubgroupAdd);
        mappings.insert("subgroup_mul", WgslIntrinsic::SubgroupMul);
        mappings.insert("warp_reduce_mul", WgslIntrinsic::SubgroupMul);
        mappings.insert("subgroup_min", WgslIntrinsic::SubgroupMin);
        mappings.insert("warp_reduce_min", WgslIntrinsic::SubgroupMin);
        mappings.insert("subgroup_max", WgslIntrinsic::SubgroupMax);
        mappings.insert("warp_reduce_max", WgslIntrinsic::SubgroupMax);
        mappings.insert("subgroup_and", WgslIntrinsic::SubgroupAnd);
        mappings.insert("warp_reduce_and", WgslIntrinsic::SubgroupAnd);
        mappings.insert("subgroup_or", WgslIntrinsic::SubgroupOr);
        mappings.insert("warp_reduce_or", WgslIntrinsic::SubgroupOr);
        mappings.insert("subgroup_xor", WgslIntrinsic::SubgroupXor);
        mappings.insert("warp_reduce_xor", WgslIntrinsic::SubgroupXor);

        // Subgroup scan operations
        mappings.insert(
            "subgroup_inclusive_add",
            WgslIntrinsic::SubgroupInclusiveAdd,
        );
        mappings.insert("warp_prefix_sum", WgslIntrinsic::SubgroupInclusiveAdd);
        mappings.insert(
            "subgroup_exclusive_add",
            WgslIntrinsic::SubgroupExclusiveAdd,
        );
        mappings.insert("warp_exclusive_sum", WgslIntrinsic::SubgroupExclusiveAdd);
        mappings.insert(
            "subgroup_inclusive_mul",
            WgslIntrinsic::SubgroupInclusiveMul,
        );
        mappings.insert(
            "subgroup_exclusive_mul",
            WgslIntrinsic::SubgroupExclusiveMul,
        );

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
        assert!(WgslIntrinsic::SubgroupAdd.requires_subgroup_extension());
        assert!(WgslIntrinsic::SubgroupInclusiveAdd.requires_subgroup_extension());
        assert!(!WgslIntrinsic::Sqrt.requires_subgroup_extension());
        assert!(!WgslIntrinsic::WorkgroupBarrier.requires_subgroup_extension());
    }

    #[test]
    fn test_subgroup_operations_mappings() {
        let registry = IntrinsicRegistry::new();

        // Vote/ballot operations
        assert_eq!(
            registry.lookup("subgroup_all"),
            Some(WgslIntrinsic::SubgroupAll)
        );
        assert_eq!(
            registry.lookup("warp_all"),
            Some(WgslIntrinsic::SubgroupAll)
        );
        assert_eq!(
            registry.lookup("subgroup_any"),
            Some(WgslIntrinsic::SubgroupAny)
        );
        assert_eq!(
            registry.lookup("subgroup_ballot"),
            Some(WgslIntrinsic::SubgroupBallot)
        );
        assert_eq!(
            registry.lookup("subgroup_elect"),
            Some(WgslIntrinsic::SubgroupElect)
        );

        // Shuffle operations
        assert_eq!(
            registry.lookup("subgroup_shuffle"),
            Some(WgslIntrinsic::SubgroupShuffle)
        );
        assert_eq!(
            registry.lookup("warp_shuffle"),
            Some(WgslIntrinsic::SubgroupShuffle)
        );
        assert_eq!(
            registry.lookup("subgroup_shuffle_xor"),
            Some(WgslIntrinsic::SubgroupShuffleXor)
        );
        assert_eq!(
            registry.lookup("subgroup_broadcast"),
            Some(WgslIntrinsic::SubgroupBroadcast)
        );
        assert_eq!(
            registry.lookup("subgroup_broadcast_first"),
            Some(WgslIntrinsic::SubgroupBroadcastFirst)
        );

        // Arithmetic reductions
        assert_eq!(
            registry.lookup("subgroup_add"),
            Some(WgslIntrinsic::SubgroupAdd)
        );
        assert_eq!(
            registry.lookup("warp_reduce_add"),
            Some(WgslIntrinsic::SubgroupAdd)
        );
        assert_eq!(
            registry.lookup("subgroup_min"),
            Some(WgslIntrinsic::SubgroupMin)
        );
        assert_eq!(
            registry.lookup("subgroup_max"),
            Some(WgslIntrinsic::SubgroupMax)
        );

        // Scan operations
        assert_eq!(
            registry.lookup("subgroup_inclusive_add"),
            Some(WgslIntrinsic::SubgroupInclusiveAdd)
        );
        assert_eq!(
            registry.lookup("warp_prefix_sum"),
            Some(WgslIntrinsic::SubgroupInclusiveAdd)
        );
        assert_eq!(
            registry.lookup("subgroup_exclusive_add"),
            Some(WgslIntrinsic::SubgroupExclusiveAdd)
        );
    }

    #[test]
    fn test_subgroup_wgsl_output() {
        // Vote/ballot
        assert_eq!(WgslIntrinsic::SubgroupAll.to_wgsl(), "subgroupAll");
        assert_eq!(WgslIntrinsic::SubgroupAny.to_wgsl(), "subgroupAny");
        assert_eq!(WgslIntrinsic::SubgroupBallot.to_wgsl(), "subgroupBallot");
        assert_eq!(WgslIntrinsic::SubgroupElect.to_wgsl(), "subgroupElect");

        // Shuffle
        assert_eq!(WgslIntrinsic::SubgroupShuffle.to_wgsl(), "subgroupShuffle");
        assert_eq!(
            WgslIntrinsic::SubgroupShuffleXor.to_wgsl(),
            "subgroupShuffleXor"
        );
        assert_eq!(
            WgslIntrinsic::SubgroupBroadcast.to_wgsl(),
            "subgroupBroadcast"
        );

        // Arithmetic
        assert_eq!(WgslIntrinsic::SubgroupAdd.to_wgsl(), "subgroupAdd");
        assert_eq!(WgslIntrinsic::SubgroupMin.to_wgsl(), "subgroupMin");
        assert_eq!(WgslIntrinsic::SubgroupMax.to_wgsl(), "subgroupMax");

        // Scans
        assert_eq!(
            WgslIntrinsic::SubgroupInclusiveAdd.to_wgsl(),
            "subgroupInclusiveAdd"
        );
        assert_eq!(
            WgslIntrinsic::SubgroupExclusiveAdd.to_wgsl(),
            "subgroupExclusiveAdd"
        );

        // Builtins
        assert_eq!(
            WgslIntrinsic::SubgroupInvocationId.to_wgsl(),
            "subgroup_invocation_id"
        );
        assert_eq!(WgslIntrinsic::SubgroupSize.to_wgsl(), "subgroup_size");
    }

    #[test]
    fn test_subgroup_builtin_detection() {
        assert!(WgslIntrinsic::SubgroupInvocationId.is_subgroup_builtin());
        assert!(WgslIntrinsic::SubgroupSize.is_subgroup_builtin());
        assert!(!WgslIntrinsic::SubgroupAdd.is_subgroup_builtin());
        assert!(!WgslIntrinsic::SubgroupShuffle.is_subgroup_builtin());
    }

    #[test]
    fn test_subgroup_intrinsics_list() {
        let registry = IntrinsicRegistry::new();
        let subgroup_ops = registry.subgroup_intrinsics();

        // Should have all the subgroup operations
        assert!(subgroup_ops.len() > 20);

        // All should require subgroup extension
        for (_, intrinsic) in &subgroup_ops {
            assert!(
                intrinsic.requires_subgroup_extension(),
                "Intrinsic {:?} should require subgroup extension",
                intrinsic
            );
        }
    }
}
