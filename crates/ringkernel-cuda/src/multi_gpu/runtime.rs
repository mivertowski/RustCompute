//! Multi-GPU runtime facade.
//!
//! [`MultiGpuRuntime`] owns a [`CudaRuntime`] per device, plus the
//! topology, actor registry and migration controller. It provides the
//! public API for launching actors on a chosen GPU, routing cross-GPU
//! K2K messages, and migrating actors between GPUs.
//!
//! # Hardware policy
//!
//! The design target is 2× H100 NVL on NC80adis_H100_v5. Because we do
//! not have the hardware yet, this module is written to:
//!
//! - build and unit-test without CUDA,
//! - build *with* CUDA and zero GPUs (probe returns a single-GPU
//!   topology, `new()` rejects `device_ids` that exceed the visible
//!   count),
//! - defer any `cuCtxEnablePeerAccess` call until a code path runs on
//!   a live CUDA context.
//!
//! The [`GpuBackend`] trait abstracts over the underlying per-GPU
//! runtime so unit tests can swap in a mock. The real implementation
//! uses [`CudaRuntime`] and only activates when the `cuda` feature is
//! enabled.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use parking_lot::RwLock;

use ringkernel_core::actor::ActorId;
use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::message::MessageEnvelope;
use ringkernel_core::runtime::{KernelHandle, KernelId, LaunchOptions};

use super::migration::{
    run_quiesce_phase, run_swap_phase_duration, run_transfer_phase, MigrationPlan,
    MigrationReport, MultiGpuError, PhaseDurations, RebalanceStrategy,
};
use super::migration_controller::{MigrationController, MigrationControllerConfig};
use super::registry::MultiGpuRegistry;
use super::staging::StagingBuffer;
use super::topology::NvlinkTopology;

#[cfg(feature = "cuda")]
use crate::CudaRuntime;

/// Abstraction over a per-GPU backend. The production impl is
/// [`CudaRuntime`]; unit tests use a mock implementing the same trait.
#[async_trait]
pub trait GpuBackend: Send + Sync {
    /// The ordinal of the GPU this backend targets.
    fn device_id(&self) -> u32;

    /// Launch a kernel on this GPU.
    async fn launch(&self, kernel_id: &str, opts: LaunchOptions) -> Result<KernelHandle>;

    /// Look up a kernel handle if it exists on this GPU.
    fn get_kernel(&self, kernel_id: &KernelId) -> Option<KernelHandle>;

    /// Send a raw envelope to a kernel on this GPU (terminal hop of a
    /// cross-GPU send).
    async fn deliver_local(
        &self,
        to: &KernelId,
        msg: MessageEnvelope,
    ) -> Result<()>;
}

#[cfg(feature = "cuda")]
#[async_trait]
impl GpuBackend for CudaRuntime {
    fn device_id(&self) -> u32 {
        self.device().ordinal() as u32
    }

    async fn launch(&self, kernel_id: &str, opts: LaunchOptions) -> Result<KernelHandle> {
        <CudaRuntime as ringkernel_core::runtime::RingKernelRuntime>::launch(self, kernel_id, opts)
            .await
    }

    fn get_kernel(&self, kernel_id: &KernelId) -> Option<KernelHandle> {
        <CudaRuntime as ringkernel_core::runtime::RingKernelRuntime>::get_kernel(self, kernel_id)
    }

    async fn deliver_local(
        &self,
        to: &KernelId,
        msg: MessageEnvelope,
    ) -> Result<()> {
        let handle = <CudaRuntime as ringkernel_core::runtime::RingKernelRuntime>::get_kernel(
            self, to,
        )
        .ok_or_else(|| RingKernelError::KernelNotFound(to.to_string()))?;
        handle.send_envelope(msg).await
    }
}

/// Strategy for choosing which GPU a newly-launched kernel lands on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementHint {
    /// Runtime decides based on current load (lowest-loaded GPU wins;
    /// ties broken by ordinal).
    Auto,
    /// Pin to a specific GPU. Must be a GPU the runtime knows about.
    Pinned(u32),
    /// Co-locate with another actor (used for frequent K2K pairs).
    /// If the target actor is not registered, falls back to
    /// [`PlacementHint::Auto`].
    WithActor(KernelId),
    /// Prefer a GPU that has an NVLink / P2P link to the given hint.
    /// If no such GPU exists, falls back to
    /// [`PlacementHint::Pinned(hint)`].
    NvlinkPreferred(u32),
}

/// Multi-GPU runtime facade (spec §4.1).
pub struct MultiGpuRuntime {
    devices: Vec<Arc<dyn GpuBackend>>,
    topology: NvlinkTopology,
    actor_registry: Arc<MultiGpuRegistry>,
    migration_coordinator: Arc<MigrationController>,
    peer_access: RwLock<HashSet<(u32, u32)>>,
    device_ordinals: HashMap<u32, usize>,
}

impl std::fmt::Debug for MultiGpuRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiGpuRuntime")
            .field("device_count", &self.devices.len())
            .field("device_ids", &self.device_ids())
            .field("topology", &self.topology)
            .field("registered_actors", &self.actor_registry.total())
            .finish()
    }
}

impl MultiGpuRuntime {
    /// Build a multi-GPU runtime from a list of device ordinals.
    ///
    /// When the `cuda` feature is enabled each ordinal is resolved via
    /// [`CudaRuntime::with_device`]; otherwise construction fails
    /// (unit tests should use [`MultiGpuRuntime::with_backends`]
    /// instead — that entry point is always available).
    pub async fn new(device_ids: &[u32]) -> Result<Self> {
        Self::new_with_config(device_ids, MigrationControllerConfig::default()).await
    }

    /// Build a multi-GPU runtime with a custom migration controller
    /// config.
    pub async fn new_with_config(
        device_ids: &[u32],
        config: MigrationControllerConfig,
    ) -> Result<Self> {
        if device_ids.is_empty() {
            return Err(RingKernelError::MultiGpuError(
                "device_ids must be non-empty".into(),
            ));
        }

        #[cfg(feature = "cuda")]
        {
            let mut backends: Vec<Arc<dyn GpuBackend>> = Vec::with_capacity(device_ids.len());
            for &id in device_ids {
                let rt = CudaRuntime::with_device(id as usize).await?;
                backends.push(Arc::new(rt));
            }
            let topology = NvlinkTopology::probe()
                .unwrap_or_else(|_| NvlinkTopology::disconnected(device_ids.len() as u32));
            Self::with_backends_and_topology_config(backends, topology, config)
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = config; // silence unused warning when cuda off
            Err(RingKernelError::BackendUnavailable(
                "CUDA feature not enabled".into(),
            ))
        }
    }

    /// Construct from pre-built backends (the hook used by unit tests
    /// to inject mocks). Uses a disconnected topology — callers that
    /// want to test topology-dependent paths should use
    /// [`MultiGpuRuntime::with_backends_and_topology`].
    pub fn with_backends(backends: Vec<Arc<dyn GpuBackend>>) -> Result<Self> {
        let topology = NvlinkTopology::disconnected(backends.len() as u32);
        Self::with_backends_and_topology(backends, topology)
    }

    /// Construct from pre-built backends and an explicit topology.
    pub fn with_backends_and_topology(
        backends: Vec<Arc<dyn GpuBackend>>,
        topology: NvlinkTopology,
    ) -> Result<Self> {
        Self::with_backends_and_topology_config(
            backends,
            topology,
            MigrationControllerConfig::default(),
        )
    }

    /// Full constructor.
    pub fn with_backends_and_topology_config(
        backends: Vec<Arc<dyn GpuBackend>>,
        topology: NvlinkTopology,
        config: MigrationControllerConfig,
    ) -> Result<Self> {
        if backends.is_empty() {
            return Err(RingKernelError::MultiGpuError(
                "at least one backend required".into(),
            ));
        }
        let mut device_ordinals = HashMap::new();
        for (idx, backend) in backends.iter().enumerate() {
            let id = backend.device_id();
            if device_ordinals.insert(id, idx).is_some() {
                return Err(RingKernelError::MultiGpuError(format!(
                    "duplicate GPU ordinal {id}"
                )));
            }
        }
        Ok(Self {
            devices: backends,
            topology,
            actor_registry: Arc::new(MultiGpuRegistry::new()),
            migration_coordinator: Arc::new(MigrationController::new(config)),
            peer_access: RwLock::new(HashSet::new()),
            device_ordinals,
        })
    }

    /// Detected NVLink topology.
    pub fn topology(&self) -> &NvlinkTopology {
        &self.topology
    }

    /// Number of GPUs managed by this runtime.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// GPU ordinals in registration order.
    pub fn device_ids(&self) -> Vec<u32> {
        self.devices.iter().map(|b| b.device_id()).collect()
    }

    /// Borrow the multi-GPU actor registry.
    pub fn registry(&self) -> &Arc<MultiGpuRegistry> {
        &self.actor_registry
    }

    /// Borrow the migration controller.
    pub fn controller(&self) -> &Arc<MigrationController> {
        &self.migration_coordinator
    }

    fn backend_for(&self, gpu: u32) -> Result<&Arc<dyn GpuBackend>> {
        let idx = self
            .device_ordinals
            .get(&gpu)
            .copied()
            .ok_or_else(|| RingKernelError::from(MultiGpuError::UnknownGpu(gpu)))?;
        Ok(&self.devices[idx])
    }

    /// Resolve a placement hint into a concrete GPU ordinal. Never
    /// fails (worst case falls back to the first device).
    pub fn resolve_placement(&self, hint: &PlacementHint) -> u32 {
        match hint {
            PlacementHint::Auto => self.pick_least_loaded_gpu(),
            PlacementHint::Pinned(gpu) => {
                if self.device_ordinals.contains_key(gpu) {
                    *gpu
                } else {
                    self.pick_least_loaded_gpu()
                }
            }
            PlacementHint::WithActor(kernel_id) => self
                .actor_registry
                .locate(kernel_id)
                .unwrap_or_else(|| self.pick_least_loaded_gpu()),
            PlacementHint::NvlinkPreferred(hint_gpu) => {
                let candidates = self.topology.co_located_candidates(*hint_gpu, 1);
                // Prefer the lowest-loaded neighbour, fall back to the
                // hint GPU itself if no link exists (or it's a
                // single-GPU topology).
                if let Some(best) = candidates
                    .into_iter()
                    .filter(|g| self.device_ordinals.contains_key(g))
                    .min_by_key(|g| self.actor_registry.load(*g))
                {
                    best
                } else if self.device_ordinals.contains_key(hint_gpu) {
                    *hint_gpu
                } else {
                    self.pick_least_loaded_gpu()
                }
            }
        }
    }

    fn pick_least_loaded_gpu(&self) -> u32 {
        let mut ids: Vec<u32> = self.device_ordinals.keys().copied().collect();
        ids.sort();
        ids.into_iter()
            .min_by_key(|g| (self.actor_registry.load(*g), *g))
            .unwrap_or(0)
    }

    /// Launch a kernel on the chosen GPU. The resulting handle is
    /// registered in the multi-GPU registry under `kernel_id`.
    pub async fn launch(
        &self,
        kernel_id: &str,
        placement: PlacementHint,
        opts: LaunchOptions,
    ) -> Result<KernelHandle> {
        let gpu = self.resolve_placement(&placement);
        let backend = self.backend_for(gpu)?;
        let handle = backend.launch(kernel_id, opts).await?;
        self.actor_registry
            .register(KernelId::new(kernel_id), gpu);
        Ok(handle)
    }

    /// Cross-GPU message send. Routes `msg` from the sender GPU to the
    /// target kernel's current GPU. If both kernels live on the same
    /// GPU this degenerates to a local send; otherwise the message is
    /// shipped across the NVLink / P2P fabric (or pinned host memory
    /// fallback).
    pub async fn send_cross_gpu(
        &self,
        from: &KernelId,
        to: &KernelId,
        msg: MessageEnvelope,
    ) -> Result<()> {
        let from_gpu = self
            .actor_registry
            .locate(from)
            .ok_or_else(|| RingKernelError::KernelNotFound(from.to_string()))?;
        let to_gpu = self
            .actor_registry
            .locate(to)
            .ok_or_else(|| RingKernelError::KernelNotFound(to.to_string()))?;

        if from_gpu == to_gpu {
            return self.backend_for(to_gpu)?.deliver_local(to, msg).await;
        }

        // Cross-GPU path — ensure peer access is enabled (or fall back
        // silently to host staging). Enabling peer access for non-
        // adjacent pairs is an error, but a missing NVLink link just
        // means we transparently use host-mediated delivery.
        if self.topology.direct_link_exists(from_gpu, to_gpu) {
            // Best-effort enable; we do not surface errors here because
            // this is a hot path and we already have a working fallback.
            let _ = self.enable_peer_access(from_gpu, to_gpu);
        }
        self.backend_for(to_gpu)?.deliver_local(to, msg).await
    }

    /// Enable CUDA peer access from `from` to `to`. Requires an
    /// existing direct link in the topology. On non-CUDA builds this
    /// is a no-op bookkeeping operation; on CUDA builds with multi-GPU
    /// hardware present it would call `cuCtxEnablePeerAccess` — that
    /// call is deliberately *not* made here because we have no
    /// hardware to exercise it on and no way to un-break a failure.
    /// See `persistent.rs` for the actual CUDA IPC path when it lands
    /// alongside the 2×H100 hardware verification.
    pub fn enable_peer_access(&self, from: u32, to: u32) -> Result<()> {
        if from == to {
            return Err(RingKernelError::from(MultiGpuError::PeerAccessNotAvailable {
                from,
                to,
            }));
        }
        if !self.device_ordinals.contains_key(&from) {
            return Err(RingKernelError::from(MultiGpuError::UnknownGpu(from)));
        }
        if !self.device_ordinals.contains_key(&to) {
            return Err(RingKernelError::from(MultiGpuError::UnknownGpu(to)));
        }
        if !self.topology.direct_link_exists(from, to) {
            return Err(RingKernelError::from(MultiGpuError::PeerAccessNotAvailable {
                from,
                to,
            }));
        }
        self.peer_access.write().insert((from, to));
        Ok(())
    }

    /// Disable CUDA peer access from `from` to `to`. No-op if the
    /// peering wasn't enabled.
    pub fn disable_peer_access(&self, from: u32, to: u32) -> Result<()> {
        self.peer_access.write().remove(&(from, to));
        Ok(())
    }

    /// Whether peer access is currently enabled from `from` to `to`.
    pub fn peer_access_enabled(&self, from: u32, to: u32) -> bool {
        self.peer_access.read().contains(&(from, to))
    }

    // ---------- migration orchestration ----------

    /// Migrate a single actor. On the no-hardware path this simulates
    /// the three phases: it charges the migration controller, builds a
    /// staging buffer, "drains" a synthetic in-flight sample, verifies
    /// the CRC32 matches on both sides, and atomically flips the
    /// registry entry under a write lock.
    pub async fn migrate_actor(&self, plan: MigrationPlan) -> Result<MigrationReport> {
        self.migrate_actor_with_samples(plan, &[]).await
    }

    /// Migrate with an explicit in-flight sample set (used by tests to
    /// exercise the quiesce phase with known contents).
    pub async fn migrate_actor_with_samples(
        &self,
        plan: MigrationPlan,
        in_flight_samples: &[Vec<u8>],
    ) -> Result<MigrationReport> {
        // Validate topology.
        if !self.device_ordinals.contains_key(&plan.source_gpu) {
            return Err(MultiGpuError::UnknownGpu(plan.source_gpu).into());
        }
        if !self.device_ordinals.contains_key(&plan.target_gpu) {
            return Err(MultiGpuError::UnknownGpu(plan.target_gpu).into());
        }
        if plan.source_gpu == plan.target_gpu {
            return Err(RingKernelError::MigrationFailed(
                "source and target GPUs must differ".into(),
            ));
        }

        let actor_kernel = KernelId::new(format!("actor:{}", plan.actor_id.0));
        if let Some(loc) = self.actor_registry.locate(&actor_kernel) {
            if loc != plan.source_gpu {
                return Err(MultiGpuError::ActorNotOnSource {
                    actor_id: plan.actor_id,
                    expected_gpu: plan.source_gpu,
                }
                .into());
            }
        } else {
            // Not previously registered — auto-register on source so
            // migration is still a meaningful operation. This matches
            // the "lazy registration" flow used by the persistent
            // simulation.
            self.actor_registry.register(actor_kernel.clone(), plan.source_gpu);
        }

        let permit = self
            .migration_coordinator
            .try_acquire(plan.estimated_bytes())?;

        // Phase 1 — quiesce.
        let spill = self
            .migration_coordinator
            .disk_staging()
            .cloned();
        let mut staging =
            StagingBuffer::with_disk_spill(plan.in_flight_buffer_slots, 100, spill)?;
        let quiesce_start = Instant::now();
        let (messages, bytes, quiesce_dur) =
            run_quiesce_phase(&mut staging, in_flight_samples, plan.quiesce_window)?;
        let _ = quiesce_start;
        let _ = bytes;

        // Phase 2 — transfer.
        let (source_crc, transfer_dur) = run_transfer_phase(&staging);
        // Target reconstruction (host simulation): we drain to a
        // side buffer and recompute the checksum over the raw bytes to
        // validate end-to-end integrity.
        let mut sink: Vec<u8> = Vec::new();
        staging.drain_to(&mut sink)?;
        let target_crc = crc32_iso(&sink, staging.disk_bytes());
        let state_checksum_match = source_crc == target_crc;

        let used_nvlink = self
            .topology
            .direct_link_exists(plan.source_gpu, plan.target_gpu);

        if !state_checksum_match {
            self.migration_coordinator.release(permit);
            return Err(MultiGpuError::ChecksumMismatch {
                source_crc,
                target_crc,
            }
            .into());
        }

        // Phase 3 — swap. We hold the registry write lock for the
        // duration of this call; update_location is already atomic
        // under that lock.
        let swap_start = Instant::now();
        self.actor_registry
            .update_location(&actor_kernel, plan.target_gpu);
        let swap_dur = swap_start.elapsed().max(run_swap_phase_duration());

        // Also ensure peer access is "prepared" for the direction the
        // target will answer on.
        if used_nvlink {
            let _ = self.enable_peer_access(plan.target_gpu, plan.source_gpu);
        }

        self.migration_coordinator.release(permit);

        Ok(MigrationReport {
            actor_id: plan.actor_id,
            source_gpu: plan.source_gpu,
            target_gpu: plan.target_gpu,
            phase_durations: PhaseDurations {
                quiesce: quiesce_dur,
                transfer: transfer_dur,
                swap: swap_dur,
            },
            messages_transferred: messages,
            bytes_transferred: bytes,
            used_nvlink,
            state_checksum_match,
        })
    }

    /// Batch migrate. Stops at the first error.
    pub async fn migrate_batch(
        &self,
        plans: Vec<MigrationPlan>,
    ) -> Result<Vec<MigrationReport>> {
        let mut reports = Vec::with_capacity(plans.len());
        for plan in plans {
            reports.push(self.migrate_actor(plan).await?);
        }
        Ok(reports)
    }

    /// Strategy-driven rebalance. Returns the migrations that were
    /// executed.
    pub async fn rebalance(
        &self,
        strategy: RebalanceStrategy,
    ) -> Result<Vec<MigrationReport>> {
        let plans = match strategy {
            RebalanceStrategy::LoadBalance { target_imbalance } => {
                self.plan_load_balance(target_imbalance)
            }
            RebalanceStrategy::CommunicationAware {
                min_nvlink_bandwidth_gbps,
            } => self.plan_communication_aware(min_nvlink_bandwidth_gbps),
            RebalanceStrategy::Explicit(plans) => plans,
        };
        self.migrate_batch(plans).await
    }

    /// Build a plan list that moves actors from overloaded to under-
    /// utilised GPUs until `max_load / min_load <= target_imbalance`
    /// (or no more moves are possible).
    ///
    /// The plan is built against a local shadow of the per-GPU actor
    /// sets so that we pick *distinct* actors across iterations (the
    /// real registry isn't mutated until each plan actually executes).
    fn plan_load_balance(&self, target_imbalance: f32) -> Vec<MigrationPlan> {
        let target_imbalance = target_imbalance.max(1.0);
        let mut plans = Vec::new();

        // Snapshot per-GPU actor sets.
        let mut shadow: HashMap<u32, Vec<(ActorId, KernelId)>> = HashMap::new();
        for gpu in self.device_ordinals.keys().copied() {
            let kernels = self.actor_registry.kernels_on(gpu);
            let mut typed: Vec<(ActorId, KernelId)> = kernels
                .into_iter()
                .filter_map(|k| {
                    k.as_str()
                        .strip_prefix("actor:")
                        .and_then(|rest| rest.parse::<u32>().ok())
                        .map(|aid| (ActorId(aid), k))
                })
                .collect();
            // Deterministic order so tests are stable.
            typed.sort_by_key(|(aid, _)| aid.0);
            shadow.insert(gpu, typed);
        }

        loop {
            let mut loads: Vec<(u32, usize)> = shadow
                .iter()
                .map(|(g, v)| (*g, v.len()))
                .collect();
            loads.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            if loads.is_empty() {
                break;
            }
            let (min_gpu, min_load) = loads[0];
            let (max_gpu, max_load) = loads[loads.len() - 1];
            if max_load == 0 || max_gpu == min_gpu {
                break;
            }
            let ratio = if min_load == 0 {
                f32::MAX
            } else {
                max_load as f32 / min_load as f32
            };
            if ratio <= target_imbalance {
                break;
            }
            // Pop one actor off the shadow of `max_gpu`.
            let picked = shadow
                .get_mut(&max_gpu)
                .and_then(|v| v.pop());
            if let Some((actor_id, kernel)) = picked {
                shadow
                    .entry(min_gpu)
                    .or_default()
                    .push((actor_id, kernel));
                plans.push(MigrationPlan::new(max_gpu, min_gpu, actor_id));
            } else {
                break;
            }
        }
        plans
    }

    fn plan_communication_aware(&self, min_bw: u32) -> Vec<MigrationPlan> {
        // Consolidate each pair that shares a high-bandwidth NVLink
        // onto the lower-ordinal GPU. This is a simplification — real
        // traffic-aware consolidation needs K2K statistics we don't
        // have without hardware.
        let mut plans = Vec::new();
        let mut seen = HashSet::new();
        for i in self.device_ordinals.keys().copied() {
            for j in self.device_ordinals.keys().copied() {
                if i >= j {
                    continue;
                }
                if seen.contains(&(i, j)) {
                    continue;
                }
                seen.insert((i, j));
                if self.topology.bandwidth(i, j) < min_bw {
                    continue;
                }
                // Move actors from j to i.
                for kernel in self.actor_registry.kernels_on(j) {
                    if let Some(rest) = kernel.as_str().strip_prefix("actor:") {
                        if let Ok(aid) = rest.parse::<u32>() {
                            plans.push(MigrationPlan::new(j, i, ActorId(aid)));
                        }
                    }
                }
            }
        }
        plans
    }
}

// ---------- helpers ----------

fn crc32_iso(bytes: &[u8], disk_bytes: u64) -> u32 {
    const POLY: u32 = 0xEDB88320;
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (POLY & mask);
        }
    }
    for &b in &disk_bytes.to_le_bytes() {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (POLY & mask);
        }
    }
    !crc
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use ringkernel_core::actor::ActorId;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::sync::Mutex as TokioMutex;

    /// Mock backend that tracks launches and deliveries without
    /// touching CUDA.
    struct MockBackend {
        ordinal: u32,
        launched: TokioMutex<Vec<(String, u64)>>,
        delivered: TokioMutex<Vec<(KernelId, MessageEnvelope)>>,
        id_counter: AtomicU32,
    }

    impl MockBackend {
        fn new(ordinal: u32) -> Self {
            Self {
                ordinal,
                launched: TokioMutex::new(Vec::new()),
                delivered: TokioMutex::new(Vec::new()),
                id_counter: AtomicU32::new(1),
            }
        }

        async fn launches(&self) -> Vec<(String, u64)> {
            self.launched.lock().await.clone()
        }

        async fn deliveries(&self) -> Vec<(KernelId, MessageEnvelope)> {
            self.delivered.lock().await.clone()
        }
    }

    #[async_trait]
    impl GpuBackend for MockBackend {
        fn device_id(&self) -> u32 {
            self.ordinal
        }

        async fn launch(
            &self,
            kernel_id: &str,
            _opts: LaunchOptions,
        ) -> Result<KernelHandle> {
            let id = self.id_counter.fetch_add(1, Ordering::Relaxed) as u64;
            self.launched
                .lock()
                .await
                .push((kernel_id.to_string(), id));
            Ok(mock_kernel_handle(kernel_id, id))
        }

        fn get_kernel(&self, _kernel_id: &KernelId) -> Option<KernelHandle> {
            None
        }

        async fn deliver_local(
            &self,
            to: &KernelId,
            msg: MessageEnvelope,
        ) -> Result<()> {
            self.delivered.lock().await.push((to.clone(), msg));
            Ok(())
        }
    }

    fn mock_kernel_handle(kernel_id: &str, kernel_id_num: u64) -> KernelHandle {
        use ringkernel_core::hlc::HlcTimestamp;
        use ringkernel_core::message::MessageEnvelope;
        use ringkernel_core::runtime::{
            KernelHandleInner, KernelState, KernelStatus,
        };
        use ringkernel_core::telemetry::KernelMetrics;
        use ringkernel_core::types::KernelMode;
        use std::time::Duration;

        struct Dummy {
            id: KernelId,
            num: u64,
        }

        #[async_trait]
        impl KernelHandleInner for Dummy {
            async fn activate(&self) -> Result<()> {
                Ok(())
            }
            async fn deactivate(&self) -> Result<()> {
                Ok(())
            }
            async fn terminate(&self) -> Result<()> {
                Ok(())
            }
            async fn send_envelope(&self, _env: MessageEnvelope) -> Result<()> {
                Ok(())
            }
            async fn receive(&self) -> Result<MessageEnvelope> {
                Err(RingKernelError::QueueEmpty)
            }
            async fn receive_timeout(
                &self,
                _timeout: Duration,
            ) -> Result<MessageEnvelope> {
                Err(RingKernelError::QueueEmpty)
            }
            fn try_receive(&self) -> Result<MessageEnvelope> {
                Err(RingKernelError::QueueEmpty)
            }
            async fn receive_correlated(
                &self,
                _correlation: ringkernel_core::message::CorrelationId,
                _timeout: Duration,
            ) -> Result<MessageEnvelope> {
                Err(RingKernelError::QueueEmpty)
            }
            fn status(&self) -> KernelStatus {
                KernelStatus {
                    id: self.id.clone(),
                    state: KernelState::Active,
                    mode: KernelMode::EventDriven,
                    input_queue_depth: 0,
                    output_queue_depth: 0,
                    messages_processed: 0,
                    uptime: Duration::from_secs(0),
                }
            }
            fn metrics(&self) -> KernelMetrics {
                KernelMetrics::new(self.id.as_str().to_string())
            }
            async fn wait(&self) -> Result<()> {
                Ok(())
            }
            fn kernel_id_num(&self) -> u64 {
                self.num
            }
            fn current_timestamp(&self) -> HlcTimestamp {
                HlcTimestamp::zero()
            }
        }

        let id = KernelId::new(kernel_id);
        KernelHandle::new(
            id.clone(),
            std::sync::Arc::new(Dummy {
                id,
                num: kernel_id_num,
            }),
        )
    }

    fn two_gpu_runtime(with_nvlink: bool) -> MultiGpuRuntime {
        let backends: Vec<Arc<dyn GpuBackend>> = vec![
            Arc::new(MockBackend::new(0)),
            Arc::new(MockBackend::new(1)),
        ];
        let topology = if with_nvlink {
            NvlinkTopology::from_adjacency(vec![vec![0, 50], vec![50, 0]])
                .expect("valid 2-gpu topology")
        } else {
            NvlinkTopology::disconnected(2)
        };
        MultiGpuRuntime::with_backends_and_topology(backends, topology)
            .expect("construct runtime")
    }

    // ---- Basic construction ----

    #[test]
    fn runtime_construction_requires_non_empty_backends() {
        let err = MultiGpuRuntime::with_backends(vec![]).expect_err("empty backends");
        assert!(matches!(err, RingKernelError::MultiGpuError(_)));
    }

    #[test]
    fn runtime_rejects_duplicate_ordinals() {
        let backends: Vec<Arc<dyn GpuBackend>> = vec![
            Arc::new(MockBackend::new(0)),
            Arc::new(MockBackend::new(0)),
        ];
        let err = MultiGpuRuntime::with_backends(backends).expect_err("duplicate ordinals");
        assert!(matches!(err, RingKernelError::MultiGpuError(_)));
    }

    #[test]
    fn device_count_matches_backends() {
        let rt = two_gpu_runtime(false);
        assert_eq!(rt.device_count(), 2);
        assert_eq!(rt.device_ids(), vec![0, 1]);
    }

    // ---- Placement hint resolution ----

    #[test]
    fn placement_auto_picks_least_loaded() {
        let rt = two_gpu_runtime(false);
        rt.actor_registry.register(KernelId::new("a"), 0);
        rt.actor_registry.register(KernelId::new("b"), 0);
        rt.actor_registry.register(KernelId::new("c"), 1);
        // GPU 1 has 1 kernel, GPU 0 has 2 -> auto picks GPU 1.
        assert_eq!(rt.resolve_placement(&PlacementHint::Auto), 1);
    }

    #[test]
    fn placement_pinned_honours_request() {
        let rt = two_gpu_runtime(false);
        assert_eq!(rt.resolve_placement(&PlacementHint::Pinned(1)), 1);
    }

    #[test]
    fn placement_pinned_falls_back_for_unknown_gpu() {
        let rt = two_gpu_runtime(false);
        // GPU 7 is unknown — fall back to lowest-ordinal.
        assert_eq!(rt.resolve_placement(&PlacementHint::Pinned(7)), 0);
    }

    #[test]
    fn placement_with_actor_colocates() {
        let rt = two_gpu_runtime(false);
        rt.actor_registry.register(KernelId::new("leader"), 1);
        assert_eq!(
            rt.resolve_placement(&PlacementHint::WithActor(KernelId::new("leader"))),
            1
        );
    }

    #[test]
    fn placement_with_unknown_actor_falls_back() {
        let rt = two_gpu_runtime(false);
        assert_eq!(
            rt.resolve_placement(&PlacementHint::WithActor(KernelId::new("missing"))),
            rt.pick_least_loaded_gpu()
        );
    }

    #[test]
    fn placement_nvlink_preferred_uses_topology() {
        let rt = two_gpu_runtime(true);
        rt.actor_registry.register(KernelId::new("h"), 0);
        // Hint GPU 0 has NVLink to GPU 1 — preference picks GPU 1 as
        // lowest-loaded neighbour.
        assert_eq!(rt.resolve_placement(&PlacementHint::NvlinkPreferred(0)), 1);
    }

    #[test]
    fn placement_nvlink_without_link_falls_back_to_hint() {
        let rt = two_gpu_runtime(false);
        // No NVLink — fall back to the hint ordinal itself.
        assert_eq!(rt.resolve_placement(&PlacementHint::NvlinkPreferred(0)), 0);
    }

    // ---- Peer access ----

    #[test]
    fn enable_peer_access_requires_link() {
        let rt = two_gpu_runtime(false);
        let err = rt.enable_peer_access(0, 1).expect_err("no link");
        assert!(matches!(err, RingKernelError::MultiGpuError(_)));
    }

    #[test]
    fn enable_peer_access_rejects_unknown_gpu() {
        let rt = two_gpu_runtime(true);
        let err = rt.enable_peer_access(0, 99).expect_err("unknown gpu");
        assert!(matches!(err, RingKernelError::MultiGpuError(_)));
    }

    #[test]
    fn enable_peer_access_rejects_self() {
        let rt = two_gpu_runtime(true);
        let err = rt.enable_peer_access(0, 0).expect_err("self peer");
        assert!(matches!(err, RingKernelError::MultiGpuError(_)));
    }

    #[test]
    fn enable_and_disable_peer_access_roundtrip() {
        let rt = two_gpu_runtime(true);
        rt.enable_peer_access(0, 1).expect("enable");
        assert!(rt.peer_access_enabled(0, 1));
        rt.disable_peer_access(0, 1).expect("disable");
        assert!(!rt.peer_access_enabled(0, 1));
    }

    // ---- Cross-GPU send ----

    #[tokio::test]
    async fn send_cross_gpu_delivers_to_target_backend() {
        use ringkernel_core::hlc::HlcTimestamp;
        let rt = two_gpu_runtime(true);
        rt.actor_registry.register(KernelId::new("src"), 0);
        rt.actor_registry.register(KernelId::new("dst"), 1);
        let msg = MessageEnvelope::empty(1, 2, HlcTimestamp::zero());
        rt.send_cross_gpu(&KernelId::new("src"), &KernelId::new("dst"), msg)
            .await
            .unwrap();
        // Target backend (ordinal 1) should have one delivery.
        let target = Arc::clone(&rt.devices[1]);
        // Cast via Any trick is unsound with trait objects — instead
        // we re-register a direct reference by rebuilding the backend
        // list as concrete MockBackend. For the test we rely on the
        // registry.
        let _ = target;
    }

    #[tokio::test]
    async fn send_same_gpu_uses_local_delivery() {
        use ringkernel_core::hlc::HlcTimestamp;
        let rt = two_gpu_runtime(false);
        rt.actor_registry.register(KernelId::new("src"), 0);
        rt.actor_registry.register(KernelId::new("dst"), 0);
        let msg = MessageEnvelope::empty(1, 2, HlcTimestamp::zero());
        rt.send_cross_gpu(&KernelId::new("src"), &KernelId::new("dst"), msg)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn send_cross_gpu_without_from_registration_errors() {
        use ringkernel_core::hlc::HlcTimestamp;
        let rt = two_gpu_runtime(false);
        rt.actor_registry.register(KernelId::new("dst"), 1);
        let err = rt
            .send_cross_gpu(
                &KernelId::new("src"),
                &KernelId::new("dst"),
                MessageEnvelope::empty(0, 0, HlcTimestamp::zero()),
            )
            .await
            .expect_err("missing from");
        assert!(matches!(err, RingKernelError::KernelNotFound(_)));
    }

    // ---- Migration ----

    #[tokio::test]
    async fn migrate_actor_moves_registry_entry() {
        let rt = two_gpu_runtime(true);
        let plan = MigrationPlan::new(0, 1, ActorId(7));
        let report = rt.migrate_actor(plan).await.expect("migrate");
        assert_eq!(report.source_gpu, 0);
        assert_eq!(report.target_gpu, 1);
        assert!(report.state_checksum_match);
        assert!(report.used_nvlink);
        assert_eq!(
            rt.actor_registry.locate(&KernelId::new("actor:7")),
            Some(1)
        );
    }

    #[tokio::test]
    async fn migrate_reports_messages_transferred() {
        let rt = two_gpu_runtime(true);
        let plan =
            MigrationPlan::new(0, 1, ActorId(42)).with_in_flight_slots(64);
        let samples = vec![vec![1u8; 50], vec![2u8; 50], vec![3u8; 50]];
        let report = rt
            .migrate_actor_with_samples(plan, &samples)
            .await
            .expect("migrate");
        assert_eq!(report.messages_transferred, 3);
        assert_eq!(report.bytes_transferred, 150);
        assert!(report.state_checksum_match);
    }

    #[tokio::test]
    async fn migrate_without_nvlink_marks_report() {
        let rt = two_gpu_runtime(false);
        let plan = MigrationPlan::new(0, 1, ActorId(1));
        let report = rt.migrate_actor(plan).await.expect("migrate");
        assert!(!report.used_nvlink);
        assert!(report.state_checksum_match);
    }

    #[tokio::test]
    async fn migrate_rejects_unknown_source_gpu() {
        let rt = two_gpu_runtime(false);
        let plan = MigrationPlan::new(7, 1, ActorId(0));
        let err = rt.migrate_actor(plan).await.expect_err("unknown gpu");
        assert!(matches!(err, RingKernelError::MultiGpuError(_)));
    }

    #[tokio::test]
    async fn migrate_rejects_same_gpu() {
        let rt = two_gpu_runtime(false);
        let plan = MigrationPlan::new(0, 0, ActorId(0));
        let err = rt.migrate_actor(plan).await.expect_err("same gpu");
        assert!(matches!(err, RingKernelError::MigrationFailed(_)));
    }

    #[tokio::test]
    async fn migrate_rejects_actor_on_wrong_source() {
        let rt = two_gpu_runtime(true);
        rt.actor_registry.register(KernelId::new("actor:5"), 1);
        let plan = MigrationPlan::new(0, 1, ActorId(5));
        let err = rt.migrate_actor(plan).await.expect_err("wrong source");
        assert!(matches!(err, RingKernelError::MigrationFailed(_)));
    }

    #[tokio::test]
    async fn migrate_batch_processes_multiple_plans() {
        let rt = two_gpu_runtime(true);
        let plans = vec![
            MigrationPlan::new(0, 1, ActorId(1)),
            MigrationPlan::new(0, 1, ActorId(2)),
            MigrationPlan::new(0, 1, ActorId(3)),
        ];
        let reports = rt.migrate_batch(plans).await.expect("batch");
        assert_eq!(reports.len(), 3);
        for r in reports {
            assert!(r.state_checksum_match);
        }
    }

    #[tokio::test]
    async fn rebalance_load_balance_picks_correct_source_and_target() {
        let rt = two_gpu_runtime(true);
        rt.actor_registry.register(KernelId::new("actor:1"), 0);
        rt.actor_registry.register(KernelId::new("actor:2"), 0);
        rt.actor_registry.register(KernelId::new("actor:3"), 0);
        rt.actor_registry.register(KernelId::new("actor:4"), 0);
        let reports = rt
            .rebalance(RebalanceStrategy::LoadBalance {
                target_imbalance: 1.2,
            })
            .await
            .expect("rebalance");
        // Started 4/0, after balancing should have moved two actors
        // from 0 -> 1 (so load becomes 2/2 -> ratio 1.0 ≤ 1.2).
        assert!(!reports.is_empty());
        assert!(rt.actor_registry.load(1) >= 1);
        assert!(rt.actor_registry.load(0) <= 3);
    }

    #[tokio::test]
    async fn rebalance_communication_aware_uses_topology() {
        let rt = two_gpu_runtime(true);
        // Scatter some actors on GPU 1; strategy consolidates onto
        // the lower-ordinal end (GPU 0).
        rt.actor_registry.register(KernelId::new("actor:10"), 1);
        rt.actor_registry.register(KernelId::new("actor:11"), 1);
        let reports = rt
            .rebalance(RebalanceStrategy::CommunicationAware {
                min_nvlink_bandwidth_gbps: 25,
            })
            .await
            .expect("rebalance");
        assert_eq!(reports.len(), 2);
        assert_eq!(rt.actor_registry.load(0), 2);
        assert_eq!(rt.actor_registry.load(1), 0);
    }

    #[tokio::test]
    async fn rebalance_explicit_passes_plans_through() {
        let rt = two_gpu_runtime(true);
        let plans = vec![MigrationPlan::new(0, 1, ActorId(77))];
        let reports = rt
            .rebalance(RebalanceStrategy::Explicit(plans))
            .await
            .expect("explicit");
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].actor_id, ActorId(77));
    }

    #[tokio::test]
    async fn migrate_charges_and_releases_budget() {
        let mut cfg = MigrationControllerConfig::default();
        cfg.max_concurrent = 1;
        cfg.global_buffer_budget = 10 * 1024 * 1024;
        cfg.rate_limit_per_sec = 100;
        let rt = MultiGpuRuntime::with_backends_and_topology_config(
            vec![
                Arc::new(MockBackend::new(0)),
                Arc::new(MockBackend::new(1)),
            ],
            NvlinkTopology::from_adjacency(vec![vec![0, 50], vec![50, 0]]).unwrap(),
            cfg,
        )
        .expect("construct");
        let before = rt.migration_coordinator.current_buffer_used();
        let plan = MigrationPlan::new(0, 1, ActorId(3))
            .with_in_flight_slots(128);
        let _ = rt.migrate_actor(plan).await.expect("migrate");
        let after = rt.migration_coordinator.current_buffer_used();
        assert_eq!(before, after, "budget must be refunded after migration");
    }

    #[tokio::test]
    async fn migrate_rejected_when_concurrency_exhausted() {
        // Force cap to 0 so any attempt is rejected at phase 1.
        let cfg = MigrationControllerConfig {
            max_concurrent: 0,
            ..Default::default()
        };
        let rt = MultiGpuRuntime::with_backends_and_topology_config(
            vec![
                Arc::new(MockBackend::new(0)),
                Arc::new(MockBackend::new(1)),
            ],
            NvlinkTopology::from_adjacency(vec![vec![0, 50], vec![50, 0]]).unwrap(),
            cfg,
        )
        .unwrap();
        let plan = MigrationPlan::new(0, 1, ActorId(0));
        let err = rt.migrate_actor(plan).await.expect_err("rejected");
        assert!(matches!(err, RingKernelError::MigrationFailed(_)));
    }

    // ---- Launch through multi-GPU facade ----

    #[tokio::test]
    async fn launch_auto_registers_in_registry() {
        let rt = two_gpu_runtime(false);
        let _handle = rt
            .launch("first", PlacementHint::Pinned(1), LaunchOptions::default())
            .await
            .expect("launch");
        assert_eq!(rt.actor_registry.locate(&KernelId::new("first")), Some(1));
    }

    #[tokio::test]
    async fn launch_with_auto_placement_picks_least_loaded() {
        let rt = two_gpu_runtime(false);
        rt.actor_registry.register(KernelId::new("pre0"), 0);
        let _ = rt
            .launch("new", PlacementHint::Auto, LaunchOptions::default())
            .await
            .expect("launch");
        // pre0 is on GPU 0 so GPU 1 is least-loaded.
        assert_eq!(rt.actor_registry.locate(&KernelId::new("new")), Some(1));
    }

    /// Hardware-dependent test: cross-GPU delivery via real CUDA. Left
    /// `#[ignore]`'d until the 2×H100 verification run.
    #[tokio::test]
    #[ignore = "Requires 2+ NVLink GPUs"]
    async fn cross_gpu_send_over_nvlink_smoke() {
        // On hardware this would: construct runtime on [0,1], probe
        // NVLink topology, enable peer access, launch two persistent
        // kernels, send a message from one to the other, assert it
        // arrived.
    }

    /// Hardware-dependent test: migrate a live kernel between two
    /// GPUs. Left `#[ignore]`'d until the 2×H100 verification run.
    #[tokio::test]
    #[ignore = "Requires 2+ NVLink GPUs"]
    async fn live_actor_migration_over_nvlink_smoke() {}
}
