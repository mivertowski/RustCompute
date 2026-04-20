//! 3-phase actor migration protocol (spec §3.1).
//!
//! Migrating a live persistent actor from one GPU to another without
//! dropping in-flight messages proceeds in three phases:
//!
//! 1. **Quiesce** — the source GPU stops accepting new messages for the
//!    actor, drains in-flight traffic into a staging buffer and emits a
//!    state snapshot.
//! 2. **Transfer** — state and the staged messages stream over NVLink /
//!    P2P to the target GPU (or fall back to host pinned memory when no
//!    NVLink link exists between the pair).
//! 3. **Swap** — the broker atomically rewires the routing table so that
//!    new senders hit the target GPU; the target drains the staging
//!    buffer before taking new traffic; the source actor is terminated.
//!
//! On hardware we would drive these phases through H2K commands and
//! device-side kernels. On the host today (no multi-GPU hardware yet) we
//! simulate the phases: the registry moves the actor, the staging buffer
//! captures a synthetic "in-flight" snapshot and the report is populated
//! with real byte counts, real NVLink detection and a real CRC32 match
//! check. That lets every invariant be exercised by unit tests.
//!
//! The public orchestration API lives on [`MultiGpuRuntime::migrate_actor`]
//! / [`MultiGpuRuntime::migrate_batch`] / [`MultiGpuRuntime::rebalance`].

use std::time::{Duration, Instant};

use ringkernel_core::actor::ActorId;
use ringkernel_core::error::{Result, RingKernelError};

use super::staging::StagingBuffer;

/// Error type specific to multi-GPU migration and runtime operations.
///
/// These wrap into [`RingKernelError::MigrationFailed`] /
/// [`RingKernelError::MultiGpuError`] when crossing the public API
/// boundary — internally we carry structured variants to make the unit
/// tests precise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiGpuError {
    /// Requested GPU index is not known to the runtime.
    UnknownGpu(u32),
    /// Peer access is not available between the two GPUs (no NVLink /
    /// no P2P path, or hardware missing).
    PeerAccessNotAvailable {
        /// Source GPU ordinal.
        from: u32,
        /// Destination GPU ordinal.
        to: u32,
    },
    /// The actor was not found on the expected source GPU.
    ActorNotOnSource {
        /// Actor identifier.
        actor_id: ActorId,
        /// GPU the caller claimed it was on.
        expected_gpu: u32,
    },
    /// The migration controller's budget would be exceeded.
    BudgetExhausted {
        /// Current in-flight bytes.
        in_flight_bytes: u64,
        /// Budget cap.
        budget_bytes: u64,
    },
    /// Too many concurrent migrations.
    ConcurrencyLimitReached {
        /// Current concurrent migrations.
        current: u32,
        /// Configured cap.
        limit: u32,
    },
    /// Rate limit exceeded.
    RateLimited {
        /// Per-second cap.
        limit_per_sec: u32,
    },
    /// CRC32 state checksum mismatched between source and target.
    ChecksumMismatch {
        /// Source-side CRC32.
        source_crc: u32,
        /// Target-side CRC32.
        target_crc: u32,
    },
    /// Staging buffer overflowed without a disk-spill target configured.
    StagingOverflow {
        /// Number of bytes the caller tried to stage.
        attempted: usize,
        /// Remaining capacity at the time of the push.
        available: usize,
    },
}

impl std::fmt::Display for MultiGpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownGpu(gpu) => write!(f, "unknown GPU ordinal: {gpu}"),
            Self::PeerAccessNotAvailable { from, to } => write!(
                f,
                "peer access not available between GPU {from} and GPU {to}"
            ),
            Self::ActorNotOnSource {
                actor_id,
                expected_gpu,
            } => write!(
                f,
                "actor {actor_id} is not registered on GPU {expected_gpu}"
            ),
            Self::BudgetExhausted {
                in_flight_bytes,
                budget_bytes,
            } => write!(
                f,
                "migration buffer budget exhausted: {in_flight_bytes}/{budget_bytes} bytes in flight"
            ),
            Self::ConcurrencyLimitReached { current, limit } => write!(
                f,
                "migration concurrency limit reached: {current}/{limit}"
            ),
            Self::RateLimited { limit_per_sec } => write!(
                f,
                "migration rate limit exceeded ({limit_per_sec}/s)"
            ),
            Self::ChecksumMismatch {
                source_crc,
                target_crc,
            } => write!(
                f,
                "migration checksum mismatch: source={source_crc:#x} target={target_crc:#x}"
            ),
            Self::StagingOverflow {
                attempted,
                available,
            } => write!(
                f,
                "staging buffer overflow: attempted {attempted} bytes, {available} available"
            ),
        }
    }
}

impl std::error::Error for MultiGpuError {}

impl From<MultiGpuError> for RingKernelError {
    fn from(err: MultiGpuError) -> Self {
        match &err {
            MultiGpuError::PeerAccessNotAvailable { .. } | MultiGpuError::UnknownGpu(_) => {
                RingKernelError::MultiGpuError(err.to_string())
            }
            MultiGpuError::ActorNotOnSource { .. }
            | MultiGpuError::ChecksumMismatch { .. }
            | MultiGpuError::StagingOverflow { .. } => {
                RingKernelError::MigrationFailed(err.to_string())
            }
            MultiGpuError::BudgetExhausted { .. }
            | MultiGpuError::ConcurrencyLimitReached { .. }
            | MultiGpuError::RateLimited { .. } => {
                RingKernelError::MigrationFailed(err.to_string())
            }
        }
    }
}

/// Default quiesce window from spec §3.1 (10 ms).
pub const DEFAULT_QUIESCE_WINDOW: Duration = Duration::from_millis(10);

/// Default in-flight buffer slots (64K, power of 2 — spec §3.1).
pub const DEFAULT_IN_FLIGHT_BUFFER_SLOTS: u32 = 65_536;

/// Default slot size (100 B — nominal message size from spec).
pub const DEFAULT_IN_FLIGHT_SLOT_SIZE: usize = 100;

/// Plan for migrating a single actor from one GPU to another.
///
/// Construct via [`MigrationPlan::new`] for the defaults from the spec
/// (10 ms quiesce, 64K in-flight slots) and then tweak via the builder
/// methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationPlan {
    /// Source GPU ordinal.
    pub source_gpu: u32,
    /// Target GPU ordinal.
    pub target_gpu: u32,
    /// Actor to migrate.
    pub actor_id: ActorId,
    /// Quiesce window — how long the source GPU tries to drain new
    /// messages into the staging buffer before forcibly completing
    /// Phase 1.
    pub quiesce_window: Duration,
    /// Number of slots in the per-actor in-flight staging buffer.
    /// Must be a power of two; the staging buffer enforces this at
    /// construction time.
    pub in_flight_buffer_slots: u32,
}

impl MigrationPlan {
    /// Create a plan with the defaults from spec §3.1.
    pub fn new(source_gpu: u32, target_gpu: u32, actor_id: ActorId) -> Self {
        Self {
            source_gpu,
            target_gpu,
            actor_id,
            quiesce_window: DEFAULT_QUIESCE_WINDOW,
            in_flight_buffer_slots: DEFAULT_IN_FLIGHT_BUFFER_SLOTS,
        }
    }

    /// Override the quiesce window.
    pub fn with_quiesce_window(mut self, window: Duration) -> Self {
        self.quiesce_window = window;
        self
    }

    /// Override the in-flight buffer slot count (will be rounded up to
    /// a power of two by the staging buffer).
    pub fn with_in_flight_slots(mut self, slots: u32) -> Self {
        self.in_flight_buffer_slots = slots;
        self
    }

    /// Estimated buffer budget (in bytes) this plan would consume.
    pub fn estimated_bytes(&self) -> u64 {
        self.in_flight_buffer_slots as u64 * DEFAULT_IN_FLIGHT_SLOT_SIZE as u64
    }
}

/// Phase-by-phase duration breakdown for a migration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PhaseDurations {
    /// Time spent quiescing the source actor (Phase 1).
    pub quiesce: Duration,
    /// Time spent transferring state + staging buffer (Phase 2).
    pub transfer: Duration,
    /// Time spent swapping the routing table (Phase 3).
    pub swap: Duration,
}

impl PhaseDurations {
    /// Total migration wall-clock time.
    pub fn total(&self) -> Duration {
        self.quiesce + self.transfer + self.swap
    }
}

/// Final report describing what happened during a single migration.
#[derive(Debug, Clone)]
pub struct MigrationReport {
    /// Migrated actor.
    pub actor_id: ActorId,
    /// Source GPU.
    pub source_gpu: u32,
    /// Target GPU.
    pub target_gpu: u32,
    /// Phase-by-phase durations.
    pub phase_durations: PhaseDurations,
    /// Messages transferred through the staging buffer.
    pub messages_transferred: u64,
    /// Bytes transferred through the staging buffer.
    pub bytes_transferred: u64,
    /// `true` when the source/target pair shared a direct NVLink (P2P)
    /// link; `false` when migration fell back to pinned host memory.
    pub used_nvlink: bool,
    /// `true` when pre- and post-migration CRC32 checksums matched.
    pub state_checksum_match: bool,
}

/// Strategy for rebalancing actors across GPUs.
#[derive(Debug, Clone)]
pub enum RebalanceStrategy {
    /// Move actors from overloaded GPUs to under-utilised ones.
    LoadBalance {
        /// Target load imbalance (max load / min load) below which no
        /// further migration is attempted. Typical value: `1.1`.
        target_imbalance: f32,
    },
    /// Co-locate actors that communicate frequently over NVLink.
    /// NB: we can't actually measure K2K traffic without hardware in the
    /// current implementation, so this strategy degrades to
    /// "pick pairs connected by ≥ min_nvlink_bandwidth_gbps" and
    /// consolidates actors onto the lower-ordinal side of each link.
    CommunicationAware {
        /// Minimum NVLink bandwidth (GB/s) required to treat a link as
        /// "communication-aware" for the purposes of consolidation.
        min_nvlink_bandwidth_gbps: u32,
    },
    /// Execute an explicit list of migrations.
    Explicit(Vec<MigrationPlan>),
}

/// Execute the quiesce phase. This drains the actor's in-flight messages
/// into the provided staging buffer, applying a time cap equal to
/// `quiesce_window` (we do not actually sleep — the phase ends as soon
/// as draining completes, which is how it would behave on hardware).
///
/// Returns `(messages, bytes, duration)` drained.
pub(crate) fn run_quiesce_phase(
    staging: &mut StagingBuffer,
    in_flight_samples: &[Vec<u8>],
    quiesce_window: Duration,
) -> Result<(u64, u64, Duration)> {
    let start = Instant::now();
    let mut messages = 0u64;
    let mut bytes = 0u64;

    for msg in in_flight_samples {
        // Stop accepting once the quiesce window has elapsed — on
        // hardware the source kernel would reach this state via a
        // deadline timer; we mirror that here so tests can exercise
        // partial drains by passing a tiny window.
        if start.elapsed() >= quiesce_window {
            break;
        }
        staging.push(msg)?;
        messages += 1;
        bytes += msg.len() as u64;
    }

    Ok((messages, bytes, start.elapsed()))
}

/// Execute the transfer phase. On hardware this would stream the
/// staging buffer and actor state across NVLink (or fall back to
/// pinned host memory). In the no-hardware implementation we
/// compute a CRC32 over the staging buffer and treat the transfer
/// as a simple memcpy.
///
/// Returns `(crc_source, duration)`.
pub(crate) fn run_transfer_phase(staging: &StagingBuffer) -> (u32, Duration) {
    let start = Instant::now();
    let crc = staging.checksum();
    (crc, start.elapsed())
}

/// Execute the swap phase. This is the routing-table swap + actor
/// activation on the target; on the host it's essentially the
/// registry move, which runs in sub-microsecond.
pub(crate) fn run_swap_phase_duration() -> Duration {
    // We don't actually lock anything here — callers hold the
    // registry write lock during the swap.
    Duration::from_nanos(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_plan_defaults_match_spec() {
        let plan = MigrationPlan::new(0, 1, ActorId(42));
        assert_eq!(plan.source_gpu, 0);
        assert_eq!(plan.target_gpu, 1);
        assert_eq!(plan.actor_id, ActorId(42));
        assert_eq!(plan.quiesce_window, DEFAULT_QUIESCE_WINDOW);
        assert_eq!(plan.in_flight_buffer_slots, DEFAULT_IN_FLIGHT_BUFFER_SLOTS);
    }

    #[test]
    fn migration_plan_builder_overrides() {
        let plan = MigrationPlan::new(0, 1, ActorId(0))
            .with_quiesce_window(Duration::from_millis(50))
            .with_in_flight_slots(1024);
        assert_eq!(plan.quiesce_window, Duration::from_millis(50));
        assert_eq!(plan.in_flight_buffer_slots, 1024);
    }

    #[test]
    fn phase_durations_total_sums_components() {
        let p = PhaseDurations {
            quiesce: Duration::from_millis(10),
            transfer: Duration::from_millis(30),
            swap: Duration::from_millis(1),
        };
        assert_eq!(p.total(), Duration::from_millis(41));
    }

    #[test]
    fn multi_gpu_error_converts_to_ring_kernel_error() {
        let err = MultiGpuError::PeerAccessNotAvailable { from: 0, to: 1 };
        let rk: RingKernelError = err.into();
        assert!(matches!(rk, RingKernelError::MultiGpuError(_)));

        let err = MultiGpuError::ChecksumMismatch {
            source_crc: 1,
            target_crc: 2,
        };
        let rk: RingKernelError = err.into();
        assert!(matches!(rk, RingKernelError::MigrationFailed(_)));

        let err = MultiGpuError::BudgetExhausted {
            in_flight_bytes: 10,
            budget_bytes: 5,
        };
        let rk: RingKernelError = err.into();
        assert!(matches!(rk, RingKernelError::MigrationFailed(_)));
    }

    #[test]
    fn estimated_bytes_matches_slot_count_times_slot_size() {
        let plan = MigrationPlan::new(0, 1, ActorId(0)).with_in_flight_slots(1024);
        assert_eq!(
            plan.estimated_bytes(),
            1024u64 * DEFAULT_IN_FLIGHT_SLOT_SIZE as u64
        );
    }
}
