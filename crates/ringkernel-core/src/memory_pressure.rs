//! GPU Memory Pressure Handling — FR-005
//!
//! Automatic memory pressure response for GPU actor systems:
//! - Per-actor memory budgets with soft/hard limits
//! - Pressure signals broadcast to all actors
//! - Configurable mitigation strategies
//! - OOM recovery via supervisor restart with reduced allocation

use std::collections::HashMap;
use std::fmt;

use crate::actor::ActorId;

/// Memory pressure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PressureLevel {
    /// Normal operation (< 60% usage).
    Normal,
    /// Warning level (60-80% usage). Non-essential allocations should be avoided.
    Warning,
    /// Critical level (80-95% usage). Active mitigation required.
    Critical,
    /// OOM imminent (>95% usage). Emergency measures.
    Emergency,
}

impl PressureLevel {
    /// Compute pressure level from usage fraction (0.0 - 1.0).
    pub fn from_usage(fraction: f64) -> Self {
        if fraction >= 0.95 {
            Self::Emergency
        } else if fraction >= 0.80 {
            Self::Critical
        } else if fraction >= 0.60 {
            Self::Warning
        } else {
            Self::Normal
        }
    }
}

impl fmt::Display for PressureLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::Warning => write!(f, "warning"),
            Self::Critical => write!(f, "critical"),
            Self::Emergency => write!(f, "emergency"),
        }
    }
}

/// Mitigation strategy for memory pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MitigationStrategy {
    /// Reduce queue depths (discard oldest buffered messages to DLQ).
    ReduceQueues,
    /// Spill cold data to host memory (GPU → CPU fallback).
    SpillToHost,
    /// Checkpoint and deactivate lowest-priority actors.
    DeactivateActors,
    /// Request garbage collection from actors (custom handler).
    RequestGc,
    /// Do nothing (let OOM happen naturally).
    None,
}

/// Per-actor memory budget.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Soft limit: warning emitted when exceeded.
    pub soft_limit_bytes: u64,
    /// Hard limit: enforced maximum.
    pub hard_limit_bytes: u64,
    /// Current allocation.
    pub current_bytes: u64,
    /// Peak allocation observed.
    pub peak_bytes: u64,
    /// Mitigation strategy when budget exceeded.
    pub mitigation: MitigationStrategy,
}

impl MemoryBudget {
    /// Create a budget with soft and hard limits.
    pub fn new(soft_limit: u64, hard_limit: u64) -> Self {
        Self {
            soft_limit_bytes: soft_limit,
            hard_limit_bytes: hard_limit,
            current_bytes: 0,
            peak_bytes: 0,
            mitigation: MitigationStrategy::ReduceQueues,
        }
    }

    /// Record an allocation.
    pub fn alloc(&mut self, bytes: u64) -> AllocationResult {
        let new_total = self.current_bytes + bytes;

        if new_total > self.hard_limit_bytes {
            return AllocationResult::Denied {
                requested: bytes,
                available: self.hard_limit_bytes.saturating_sub(self.current_bytes),
            };
        }

        self.current_bytes = new_total;
        if new_total > self.peak_bytes {
            self.peak_bytes = new_total;
        }

        if new_total > self.soft_limit_bytes {
            AllocationResult::GrantedWithWarning {
                usage_fraction: new_total as f64 / self.hard_limit_bytes as f64,
            }
        } else {
            AllocationResult::Granted
        }
    }

    /// Record a deallocation.
    pub fn dealloc(&mut self, bytes: u64) {
        self.current_bytes = self.current_bytes.saturating_sub(bytes);
    }

    /// Current usage as fraction (0.0 - 1.0).
    pub fn usage_fraction(&self) -> f64 {
        if self.hard_limit_bytes == 0 {
            return 0.0;
        }
        self.current_bytes as f64 / self.hard_limit_bytes as f64
    }

    /// Current pressure level.
    pub fn pressure_level(&self) -> PressureLevel {
        PressureLevel::from_usage(self.usage_fraction())
    }
}

/// Result of a memory allocation request.
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationResult {
    /// Allocation granted, within soft limit.
    Granted,
    /// Allocation granted, but soft limit exceeded (warning).
    GrantedWithWarning {
        /// Fraction of memory budget currently in use (0.0 to 1.0).
        usage_fraction: f64,
    },
    /// Allocation denied — would exceed hard limit.
    Denied {
        /// Number of bytes requested.
        requested: u64,
        /// Number of bytes available.
        available: u64,
    },
}

/// Global memory pressure monitor for all actors.
pub struct MemoryPressureMonitor {
    /// Per-actor budgets.
    budgets: HashMap<ActorId, MemoryBudget>,
    /// Total GPU memory available.
    total_gpu_memory: u64,
    /// Total currently allocated across all actors.
    total_allocated: u64,
    /// Default budget for actors without explicit config.
    default_budget: MemoryBudget,
}

impl MemoryPressureMonitor {
    /// Create a monitor for a GPU with the given total memory.
    pub fn new(total_gpu_memory: u64) -> Self {
        let per_actor_default = total_gpu_memory / 8; // Default: 1/8 of GPU memory
        Self {
            budgets: HashMap::new(),
            total_gpu_memory,
            total_allocated: 0,
            default_budget: MemoryBudget::new(
                (per_actor_default as f64 * 0.8) as u64,
                per_actor_default,
            ),
        }
    }

    /// Set a budget for a specific actor.
    pub fn set_budget(&mut self, actor: ActorId, budget: MemoryBudget) {
        self.budgets.insert(actor, budget);
    }

    /// Request allocation from an actor's budget.
    pub fn request_alloc(&mut self, actor: ActorId, bytes: u64) -> AllocationResult {
        let budget = self
            .budgets
            .entry(actor)
            .or_insert_with(|| self.default_budget.clone());

        let result = budget.alloc(bytes);
        if matches!(
            result,
            AllocationResult::Granted | AllocationResult::GrantedWithWarning { .. }
        ) {
            self.total_allocated += bytes;
        }
        result
    }

    /// Record a deallocation.
    pub fn record_dealloc(&mut self, actor: ActorId, bytes: u64) {
        if let Some(budget) = self.budgets.get_mut(&actor) {
            budget.dealloc(bytes);
        }
        self.total_allocated = self.total_allocated.saturating_sub(bytes);
    }

    /// Get the global pressure level across all actors.
    pub fn global_pressure(&self) -> PressureLevel {
        if self.total_gpu_memory == 0 {
            return PressureLevel::Normal;
        }
        PressureLevel::from_usage(self.total_allocated as f64 / self.total_gpu_memory as f64)
    }

    /// Get actors that are above their soft limit.
    pub fn actors_over_budget(&self) -> Vec<(ActorId, f64)> {
        self.budgets
            .iter()
            .filter(|(_, b)| b.current_bytes > b.soft_limit_bytes)
            .map(|(&id, b)| (id, b.usage_fraction()))
            .collect()
    }

    /// Get actors sorted by memory usage (descending).
    pub fn actors_by_usage(&self) -> Vec<(ActorId, u64)> {
        let mut actors: Vec<_> = self
            .budgets
            .iter()
            .map(|(&id, b)| (id, b.current_bytes))
            .collect();
        actors.sort_by_key(|a| std::cmp::Reverse(a.1));
        actors
    }

    /// Get an actor's budget.
    pub fn get_budget(&self, actor: ActorId) -> Option<&MemoryBudget> {
        self.budgets.get(&actor)
    }

    /// Get total allocated across all actors.
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated
    }

    /// Get total GPU memory.
    pub fn total_gpu_memory(&self) -> u64 {
        self.total_gpu_memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_levels() {
        assert_eq!(PressureLevel::from_usage(0.3), PressureLevel::Normal);
        assert_eq!(PressureLevel::from_usage(0.7), PressureLevel::Warning);
        assert_eq!(PressureLevel::from_usage(0.9), PressureLevel::Critical);
        assert_eq!(PressureLevel::from_usage(0.96), PressureLevel::Emergency);
    }

    #[test]
    fn test_memory_budget() {
        let mut budget = MemoryBudget::new(800, 1000);

        assert_eq!(budget.alloc(500), AllocationResult::Granted);
        assert_eq!(budget.current_bytes, 500);

        // Exceeds soft limit
        let result = budget.alloc(400);
        assert!(matches!(
            result,
            AllocationResult::GrantedWithWarning { .. }
        ));
        assert_eq!(budget.current_bytes, 900);

        // Exceeds hard limit
        let result = budget.alloc(200);
        assert!(matches!(result, AllocationResult::Denied { .. }));
        assert_eq!(budget.current_bytes, 900); // Unchanged

        budget.dealloc(500);
        assert_eq!(budget.current_bytes, 400);
        assert_eq!(budget.peak_bytes, 900); // Peak preserved
    }

    #[test]
    fn test_monitor_global_pressure() {
        let mut monitor = MemoryPressureMonitor::new(1000);

        monitor.set_budget(ActorId(1), MemoryBudget::new(400, 500));
        monitor.set_budget(ActorId(2), MemoryBudget::new(400, 500));

        monitor.request_alloc(ActorId(1), 300);
        monitor.request_alloc(ActorId(2), 300);

        assert_eq!(monitor.total_allocated(), 600);
        assert_eq!(monitor.global_pressure(), PressureLevel::Warning);
    }

    #[test]
    fn test_actors_over_budget() {
        let mut monitor = MemoryPressureMonitor::new(10000);
        monitor.set_budget(ActorId(1), MemoryBudget::new(100, 200));
        monitor.set_budget(ActorId(2), MemoryBudget::new(100, 200));

        monitor.request_alloc(ActorId(1), 150); // Over soft
        monitor.request_alloc(ActorId(2), 50); // Under soft

        let over = monitor.actors_over_budget();
        assert_eq!(over.len(), 1);
        assert_eq!(over[0].0, ActorId(1));
    }

    #[test]
    fn test_actors_by_usage() {
        let mut monitor = MemoryPressureMonitor::new(10000);
        monitor.set_budget(ActorId(1), MemoryBudget::new(500, 1000));
        monitor.set_budget(ActorId(2), MemoryBudget::new(500, 1000));

        monitor.request_alloc(ActorId(1), 100);
        monitor.request_alloc(ActorId(2), 300);

        let sorted = monitor.actors_by_usage();
        assert_eq!(sorted[0].0, ActorId(2)); // Highest usage first
        assert_eq!(sorted[1].0, ActorId(1));
    }
}
