//! Multi-GPU actor registry.
//!
//! Tracks which actor lives on which GPU. This is the single source of
//! truth that the K2K broker consults to route cross-GPU messages and
//! that the migration orchestrator updates when an actor moves from one
//! GPU to another.
//!
//! Invariants:
//! - Every actor has **exactly one** GPU location at any point in time
//!   (Phase 3 of migration flips this atomically under a write lock).
//! - Per-GPU sets are kept in sync with the location map (a [`register`]
//!   insert + membership check is atomic under the write lock).
//! - Registrations are idempotent: re-registering an actor on the same
//!   GPU is a no-op; re-registering it on a *different* GPU is treated
//!   as a migration (location is updated, old GPU set is cleaned up).
//!
//! [`register`]: MultiGpuRegistry::register

use std::collections::{HashMap, HashSet};

use parking_lot::RwLock;

use ringkernel_core::runtime::KernelId;

/// Tracks which kernels are running on which GPU.
#[derive(Default)]
pub struct MultiGpuRegistry {
    /// Reverse lookup: kernel → GPU.
    actor_locations: RwLock<HashMap<KernelId, u32>>,
    /// Forward lookup: GPU → set of kernels.
    gpu_kernels: RwLock<HashMap<u32, HashSet<KernelId>>>,
}

impl MultiGpuRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a kernel on a GPU. If the kernel is already registered
    /// on a different GPU, the old entry is cleaned up atomically
    /// (equivalent to calling [`MultiGpuRegistry::update_location`]).
    pub fn register(&self, kernel_id: KernelId, gpu: u32) {
        let mut locations = self.actor_locations.write();
        let mut per_gpu = self.gpu_kernels.write();

        // If already on `gpu`, no-op.
        if let Some(existing) = locations.get(&kernel_id) {
            if *existing == gpu {
                per_gpu.entry(gpu).or_default().insert(kernel_id.clone());
                return;
            }
            // Moving GPU — pull the id off the old per-gpu set.
            if let Some(set) = per_gpu.get_mut(existing) {
                set.remove(&kernel_id);
            }
        }

        locations.insert(kernel_id.clone(), gpu);
        per_gpu.entry(gpu).or_default().insert(kernel_id);
    }

    /// Deregister a kernel. No-op if the kernel was never registered.
    pub fn deregister(&self, kernel_id: &KernelId) {
        let mut locations = self.actor_locations.write();
        let mut per_gpu = self.gpu_kernels.write();
        if let Some(gpu) = locations.remove(kernel_id) {
            if let Some(set) = per_gpu.get_mut(&gpu) {
                set.remove(kernel_id);
            }
        }
    }

    /// Look up the GPU for a kernel. Returns `None` if the kernel is
    /// not registered.
    pub fn locate(&self, kernel_id: &KernelId) -> Option<u32> {
        self.actor_locations.read().get(kernel_id).copied()
    }

    /// All kernels currently registered on a given GPU. Order is not
    /// meaningful; callers that need deterministic ordering should
    /// sort the result.
    pub fn kernels_on(&self, gpu: u32) -> Vec<KernelId> {
        self.gpu_kernels
            .read()
            .get(&gpu)
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Number of kernels registered on a GPU.
    pub fn load(&self, gpu: u32) -> usize {
        self.gpu_kernels
            .read()
            .get(&gpu)
            .map(|s| s.len())
            .unwrap_or(0)
    }

    /// Atomically update the GPU for a kernel. This is the post-Phase-3
    /// migration hook. Returns the previous GPU (if any).
    pub fn update_location(&self, kernel_id: &KernelId, new_gpu: u32) -> Option<u32> {
        let mut locations = self.actor_locations.write();
        let mut per_gpu = self.gpu_kernels.write();

        let prev = locations.insert(kernel_id.clone(), new_gpu);
        if let Some(old) = prev {
            if old != new_gpu {
                if let Some(set) = per_gpu.get_mut(&old) {
                    set.remove(kernel_id);
                }
            }
        }
        per_gpu
            .entry(new_gpu)
            .or_default()
            .insert(kernel_id.clone());
        prev
    }

    /// Total number of distinct registered kernels.
    pub fn total(&self) -> usize {
        self.actor_locations.read().len()
    }

    /// Load distribution across GPUs, as a sorted Vec<(gpu_id, load)>.
    /// Useful for rebalancing strategies.
    pub fn load_distribution(&self) -> Vec<(u32, usize)> {
        let mut out: Vec<_> = self
            .gpu_kernels
            .read()
            .iter()
            .map(|(gpu, set)| (*gpu, set.len()))
            .collect();
        out.sort_by_key(|(gpu, _)| *gpu);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_locate() {
        let reg = MultiGpuRegistry::new();
        reg.register(KernelId::new("actor1"), 0);
        assert_eq!(reg.locate(&KernelId::new("actor1")), Some(0));
        assert_eq!(reg.load(0), 1);
        assert_eq!(reg.total(), 1);
    }

    #[test]
    fn deregister_removes_from_both_indices() {
        let reg = MultiGpuRegistry::new();
        reg.register(KernelId::new("a"), 0);
        reg.register(KernelId::new("b"), 0);
        reg.deregister(&KernelId::new("a"));
        assert_eq!(reg.locate(&KernelId::new("a")), None);
        assert_eq!(reg.load(0), 1);
        assert_eq!(reg.kernels_on(0), vec![KernelId::new("b")]);
    }

    #[test]
    fn update_location_moves_between_gpus() {
        let reg = MultiGpuRegistry::new();
        reg.register(KernelId::new("x"), 0);
        let prev = reg.update_location(&KernelId::new("x"), 1);
        assert_eq!(prev, Some(0));
        assert_eq!(reg.locate(&KernelId::new("x")), Some(1));
        assert_eq!(reg.load(0), 0);
        assert_eq!(reg.load(1), 1);
    }

    #[test]
    fn re_registering_on_same_gpu_is_idempotent() {
        let reg = MultiGpuRegistry::new();
        reg.register(KernelId::new("x"), 0);
        reg.register(KernelId::new("x"), 0);
        assert_eq!(reg.load(0), 1);
        assert_eq!(reg.total(), 1);
    }

    #[test]
    fn re_registering_on_different_gpu_migrates() {
        let reg = MultiGpuRegistry::new();
        reg.register(KernelId::new("x"), 0);
        reg.register(KernelId::new("x"), 1);
        assert_eq!(reg.locate(&KernelId::new("x")), Some(1));
        assert_eq!(reg.load(0), 0);
        assert_eq!(reg.load(1), 1);
    }

    #[test]
    fn load_distribution_returns_per_gpu_counts_sorted() {
        let reg = MultiGpuRegistry::new();
        reg.register(KernelId::new("a"), 0);
        reg.register(KernelId::new("b"), 0);
        reg.register(KernelId::new("c"), 1);
        let dist = reg.load_distribution();
        assert_eq!(dist, vec![(0, 2), (1, 1)]);
    }

    #[test]
    fn deregister_missing_kernel_is_noop() {
        let reg = MultiGpuRegistry::new();
        reg.deregister(&KernelId::new("nope"));
        assert_eq!(reg.total(), 0);
    }

    #[test]
    fn kernels_on_unknown_gpu_returns_empty() {
        let reg = MultiGpuRegistry::new();
        assert!(reg.kernels_on(7).is_empty());
        assert_eq!(reg.load(7), 0);
    }
}
