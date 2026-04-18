//! Hot-swappable compiled rule artifacts.
//!
//! Per `docs/superpowers/specs/2026-04-17-v1.1-vyngraph-gaps.md` section 3.3,
//! this module lets RingKernel accept opaque compiled rule artifacts (PTX +
//! metadata) and hot-swap them atomically without runtime restart.
//!
//! ## Design philosophy
//!
//! RingKernel stays **rule-format-agnostic**. Callers such as VynGraph own
//! OWL 2 RL / SHACL parsing and compile rules to PTX using our existing
//! `ringkernel-cuda-codegen` pipeline. RingKernel receives the compiled
//! artifact via [`CompiledRule`] and manages versioning, validation,
//! rollback, and the atomic swap state machine.
//!
//! ## Artifact lifecycle
//!
//! ```text
//! CompiledRule  ─register_rule()─►  RuleStatus::Registered
//!      │                                    │
//!      │         reload_rule()              │
//!      ▼                                    ▼
//! (new version) ─pre_stage/quiesce/swap─► RuleStatus::Active
//!                                              │
//!      prior version: Superseded(new_ver)      │
//!                                              │
//!                       rollback_rule() ◄──────┤
//!      current version: Rolledback                │
//!      prior version: Active                      │
//! ```
//!
//! ## Guarantees
//!
//! - Version monotonicity (downgrades rejected unless explicit rollback)
//! - Bounded history (FIFO eviction beyond `max_history`)
//! - Validation-before-swap (compute cap, dependencies, signature)
//! - Pluggable swap backend (`NoopSwapBackend` for tests, CUDA in production)
//!
//! ## Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use ringkernel_core::rules::{
//!     ActorConfig, CompiledRule, NoopSwapBackend, RuleMetadata, RuleRegistry,
//! };
//!
//! # async fn example() {
//! let registry = RuleRegistry::new(5, Arc::new(NoopSwapBackend));
//! let rule = CompiledRule {
//!     rule_id: "gaap-consolidation".into(),
//!     version: 1,
//!     ptx: b".version 8.0\n.target sm_90\n".to_vec(),
//!     compute_cap: "sm_90".into(),
//!     depends_on: vec![],
//!     signature: None,
//!     actor_config: ActorConfig::default(),
//!     metadata: RuleMetadata::default(),
//! };
//! let handle = registry.register_rule(rule, "sm_90").await.unwrap();
//! assert_eq!(handle.version, 1);
//! # }
//! ```
//!
//! [`HotReloadManager::rule_registry()`] exposes the registry for use by
//! existing multi-GPU hot-reload plumbing.
//!
//! [`HotReloadManager::rule_registry()`]: crate::multi_gpu::HotReloadManager::rule_registry

use std::time::{Duration, SystemTime};

pub mod registry;

pub use registry::{
    NoopSwapBackend, RuleRegistry, RuleSwapBackend, SignatureVerifier,
};

/// A compiled rule artifact ready for GPU hot-swap.
///
/// RingKernel does not inspect `ptx` beyond validating compute capability,
/// dependencies and (optionally) signature. The caller owns semantic
/// correctness of the compilation.
#[derive(Debug, Clone)]
pub struct CompiledRule {
    /// Caller-scoped rule set identifier (e.g. `"gaap-consolidation"`).
    pub rule_id: String,
    /// Monotonically increasing version; later versions must be strictly
    /// greater than the currently active version.
    pub version: u64,
    /// Compiled PTX bytes for the actor kernel.
    pub ptx: Vec<u8>,
    /// Required compute capability, e.g. `"sm_90"` for H100.
    pub compute_cap: String,
    /// Other `rule_id`s that must already be registered before this rule
    /// can be installed. Used for inference-rule dependency graphs.
    pub depends_on: Vec<String>,
    /// Optional integrity signature (format is verifier-specific).
    pub signature: Option<Vec<u8>>,
    /// Actor launch configuration.
    pub actor_config: ActorConfig,
    /// Opaque metadata passed through for audit/logging. RingKernel does
    /// not interpret any of these fields.
    pub metadata: RuleMetadata,
}

/// Launch configuration for the rule's actor kernel.
#[derive(Debug, Clone)]
pub struct ActorConfig {
    /// CUDA block dimensions `(x, y, z)`.
    pub block_dim: (u32, u32, u32),
    /// CUDA grid dimensions `(x, y, z)`.
    pub grid_dim: (u32, u32, u32),
    /// Dynamic shared-memory bytes to allocate per block.
    pub shared_mem_bytes: u32,
    /// Maximum number of in-flight messages this actor accepts.
    pub max_in_flight: u32,
}

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            block_dim: (1, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
            max_in_flight: 1024,
        }
    }
}

/// Opaque metadata attached to a compiled rule.
///
/// All fields are optional and none of them influence the swap state
/// machine. They exist solely for audit trails, observability, and
/// attribution. Callers are free to ignore them or fill them in as they
/// see fit; RingKernel passes them through unchanged.
#[derive(Debug, Clone, Default)]
pub struct RuleMetadata {
    /// Human-readable description of the source language, e.g.
    /// `"OWL 2 RL"`, `"SHACL"`, `"custom DSL"`. Opaque to RingKernel.
    pub source_language: Option<String>,
    /// SHA-256 of the rule source text, for audit reproducibility.
    pub source_hash: Option<[u8; 32]>,
    /// When the rule was compiled.
    pub compiled_at: Option<SystemTime>,
    /// Version string of the compiler that produced this artifact.
    pub compiler_version: Option<String>,
    /// Principal who authored / compiled the rule.
    pub author: Option<String>,
}

/// Lightweight handle returned after a successful registry operation.
#[derive(Debug, Clone)]
pub struct RuleHandle {
    /// Rule identifier.
    pub rule_id: String,
    /// Rule version.
    pub version: u64,
    /// Lifecycle status of this specific version.
    pub status: RuleStatus,
    /// When the version was registered with the registry.
    pub registered_at: SystemTime,
}

/// Lifecycle status of a specific rule version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleStatus {
    /// Loaded and validated but not yet the active version.
    Registered,
    /// Currently executing on the device.
    Active,
    /// Being drained ahead of a swap.
    Quiescing,
    /// Replaced by the specified newer version.
    Superseded(u64),
    /// Rolled back away from (prior `Active` version the user chose to revert).
    Rolledback,
    /// Validation or swap backend failed; this version is unusable.
    Failed,
}

/// Report emitted after a successful reload (or rollback).
#[derive(Debug, Clone)]
pub struct ReloadReport {
    /// Rule identifier.
    pub rule_id: String,
    /// Version we moved away from (0 if this was the initial activation).
    pub from_version: u64,
    /// Version that is now `Active`.
    pub to_version: u64,
    /// Time spent draining the old actor.
    pub quiesce_duration: Duration,
    /// Time spent performing the atomic pointer swap.
    pub swap_duration: Duration,
    /// Messages that were in-flight during the swap window
    /// (as reported by the swap backend).
    pub messages_in_flight_during_swap: u64,
    /// Whether the previous version is still retained in history and can
    /// be the target of a subsequent rollback.
    pub rollback_available: bool,
}

/// Errors produced by the rule registry.
#[derive(Debug, thiserror::Error)]
pub enum RuleError {
    /// No such rule in the registry.
    #[error("rule not found: {0}")]
    NotFound(String),

    /// Incoming version is not strictly newer than the current active version.
    #[error("version downgrade rejected: current={current}, proposed={proposed}")]
    VersionDowngrade {
        /// Currently active version.
        current: u64,
        /// Version the caller tried to install.
        proposed: u64,
    },

    /// Rule targets a compute capability the device does not meet.
    #[error("compute capability mismatch: rule={required}, device={available}")]
    ComputeCapMismatch {
        /// Compute cap the rule requires.
        required: String,
        /// Compute cap the device actually has.
        available: String,
    },

    /// Rule depends on another rule that is not registered.
    #[error("dependency missing: {0}")]
    MissingDependency(String),

    /// Signature check did not succeed.
    #[error("signature verification failed")]
    InvalidSignature,

    /// Caller asked to roll back to a version no longer in history.
    #[error("rollback target not in history: version={0}")]
    RollbackTargetMissing(u64),

    /// No version is currently active — nothing to roll back from.
    #[error("no active version to rollback")]
    NoActiveVersion,

    /// Quiesce window elapsed before the actor finished draining.
    #[error("quiesce timeout after {0:?}")]
    QuiesceTimeout(Duration),

    /// Swap backend refused the operation (wraps backend-specific detail).
    #[error("swap backend error: {0}")]
    BackendError(String),

    /// Version was already registered and we do not allow re-register of
    /// the same `(rule_id, version)` tuple.
    #[error("duplicate version: rule={rule_id}, version={version}")]
    DuplicateVersion {
        /// Rule identifier.
        rule_id: String,
        /// Version that was already present.
        version: u64,
    },
}
