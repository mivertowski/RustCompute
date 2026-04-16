//! Dynamic Actor Scheduling — Work Stealing Protocol
//!
//! Provides load balancing for persistent GPU actors via a work stealing protocol.
//! Without dynamic scheduling, each actor (thread block) processes only its own
//! message queue. If one actor's workload spikes while neighbors are idle,
//! the busy actor becomes a bottleneck.
//!
//! # Scheduler Warp Pattern
//!
//! Within each thread block of the persistent kernel:
//! - **Warp 0**: Scheduler warp — monitors queue depth, steals work from overloaded
//!   neighbors, redistributes messages
//! - **Warps 1-N**: Compute warps — process messages from the local work queue
//!
//! ```text
//! ┌─── Block (Actor) ───────────────────────────────────┐
//! │ Warp 0 [SCHEDULER]                                   │
//! │ ├─ Monitor local queue depth                         │
//! │ ├─ If depth < steal_threshold:                       │
//! │ │   └─ Steal from busiest neighbor via K2K           │
//! │ ├─ If depth > share_threshold:                       │
//! │ │   └─ Offer work to least-busy neighbor             │
//! │ └─ Update load metrics in shared memory              │
//! │                                                      │
//! │ Warps 1-7 [COMPUTE]                                  │
//! │ ├─ Dequeue message from local work queue             │
//! │ ├─ Process message (user handler)                    │
//! │ └─ Enqueue response to output queue                  │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! # Work Stealing Protocol
//!
//! 1. Each block publishes its queue depth to a shared load table (global or DSMEM)
//! 2. Scheduler warp compares local depth with neighbor depths
//! 3. If local depth < `steal_threshold` and a neighbor has depth > `share_threshold`:
//!    a. Scheduler warp atomically reserves N messages from neighbor's queue
//!    b. Messages are copied via K2K channel (DSMEM for cluster, global for cross-cluster)
//!    c. Both blocks update their queue depths
//! 4. Grid sync (or cluster sync) ensures load table consistency
//!
//! # Load Table Layout (in mapped/global memory)
//!
//! ```text
//! load_table[block_id] = {
//!     queue_depth: u32,    // Current input queue depth
//!     capacity: u32,       // Queue capacity
//!     messages_processed: u64,  // Throughput indicator
//!     steal_requests: u32, // Pending steal requests
//!     offer_count: u32,    // Messages offered to steal
//! }
//! ```

use std::fmt;

/// Configuration for dynamic actor scheduling.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Queue depth below which the scheduler tries to steal work.
    pub steal_threshold: u32,
    /// Queue depth above which the scheduler offers work to neighbors.
    pub share_threshold: u32,
    /// Maximum messages to steal in one operation.
    pub max_steal_batch: u32,
    /// Number of neighbor blocks to check for work stealing.
    pub steal_neighborhood: u32,
    /// Enable/disable dynamic scheduling (can be toggled at runtime).
    pub enabled: bool,
    /// Scheduling strategy.
    pub strategy: SchedulingStrategy,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            steal_threshold: 4,
            share_threshold: 64,
            max_steal_batch: 16,
            steal_neighborhood: 4,
            enabled: true,
            strategy: SchedulingStrategy::WorkStealing,
        }
    }
}

impl SchedulerConfig {
    /// Create a static (no scheduling) configuration.
    ///
    /// This is the current default behavior: each thread block processes its
    /// own fixed work queue. No load balancing occurs.
    pub fn static_scheduling() -> Self {
        Self {
            enabled: false,
            strategy: SchedulingStrategy::Static,
            ..Default::default()
        }
    }

    /// Create a work-stealing configuration with the given threshold.
    ///
    /// When an actor's queue depth falls below `steal_threshold`, its scheduler
    /// warp will attempt to steal messages from the busiest neighbor.
    pub fn work_stealing(steal_threshold: u32) -> Self {
        Self {
            steal_threshold,
            strategy: SchedulingStrategy::WorkStealing,
            ..Default::default()
        }
    }

    /// Create a round-robin configuration.
    ///
    /// Messages are distributed from a global work queue to blocks in
    /// round-robin order, using a global atomic counter for index assignment.
    pub fn round_robin() -> Self {
        Self {
            strategy: SchedulingStrategy::RoundRobin,
            ..Default::default()
        }
    }

    /// Create a priority-based configuration with the given number of levels.
    ///
    /// Messages are bucketed into priority sub-queues (0 = lowest, levels-1 = highest).
    /// The scheduler warp dequeues from the highest-priority non-empty sub-queue first.
    pub fn priority(levels: u32) -> Self {
        let levels = levels.clamp(1, 16);
        Self {
            strategy: SchedulingStrategy::Priority { levels },
            ..Default::default()
        }
    }

    /// Set the steal threshold.
    pub fn with_steal_threshold(mut self, threshold: u32) -> Self {
        self.steal_threshold = threshold;
        self
    }

    /// Set the share threshold.
    pub fn with_share_threshold(mut self, threshold: u32) -> Self {
        self.share_threshold = threshold;
        self
    }

    /// Set the maximum steal batch size.
    pub fn with_max_steal_batch(mut self, batch: u32) -> Self {
        self.max_steal_batch = batch;
        self
    }

    /// Set the number of neighbor blocks to check for work stealing.
    pub fn with_steal_neighborhood(mut self, neighborhood: u32) -> Self {
        self.steal_neighborhood = neighborhood;
        self
    }

    /// Enable or disable the scheduler.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Check if this configuration uses dynamic scheduling (anything other than Static).
    pub fn is_dynamic(&self) -> bool {
        self.enabled && self.strategy != SchedulingStrategy::Static
    }
}

/// Work item for the scheduler.
///
/// Represents a unit of work that can be assigned to any actor (thread block).
/// The scheduler warp uses these to track pending work in the global work queue.
///
/// Layout (16 bytes): message_id (8) + actor_id (4) + priority (4).
/// Fields ordered largest-first to avoid padding.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkItem {
    /// Unique message identifier for tracking.
    pub message_id: u64,
    /// Actor ID that owns (or should process) this work item.
    pub actor_id: u32,
    /// Priority level (0 = lowest). Used by `Priority` strategy.
    pub priority: u32,
}

impl WorkItem {
    /// Create a new work item.
    pub fn new(actor_id: u32, message_id: u64, priority: u32) -> Self {
        Self {
            message_id,
            actor_id,
            priority,
        }
    }
}

impl fmt::Display for WorkItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WorkItem(actor={}, msg={}, pri={})",
            self.actor_id, self.message_id, self.priority
        )
    }
}

/// Configuration for the scheduler warp pattern in CUDA codegen.
///
/// This controls how the scheduler warp (warp 0 by default) is generated
/// within each persistent kernel thread block. When enabled, the codegen
/// produces a split: warp 0 handles work distribution, remaining warps
/// do message processing.
#[derive(Debug, Clone)]
pub struct SchedulerWarpConfig {
    /// Which warp handles scheduling (default: 0).
    pub scheduler_warp_id: u32,
    /// The scheduling parameters.
    pub scheduler: SchedulerConfig,
    /// Size of the global work queue (number of WorkItem slots).
    /// Must be power of 2. Used for round-robin and priority strategies.
    pub work_queue_capacity: usize,
    /// Polling interval in nanoseconds for the scheduler warp
    /// when no work is available (default: 1000ns).
    pub poll_interval_ns: u32,
}

impl Default for SchedulerWarpConfig {
    fn default() -> Self {
        Self {
            scheduler_warp_id: 0,
            scheduler: SchedulerConfig::default(),
            work_queue_capacity: 1024,
            poll_interval_ns: 1000,
        }
    }
}

impl SchedulerWarpConfig {
    /// Create a new scheduler warp config with the given strategy.
    pub fn new(scheduler: SchedulerConfig) -> Self {
        Self {
            scheduler,
            ..Default::default()
        }
    }

    /// Create a static (disabled) scheduler warp config.
    /// Codegen will produce the original non-split kernel.
    pub fn disabled() -> Self {
        Self {
            scheduler: SchedulerConfig::static_scheduling(),
            ..Default::default()
        }
    }

    /// Set the scheduler warp ID.
    pub fn with_scheduler_warp(mut self, warp_id: u32) -> Self {
        self.scheduler_warp_id = warp_id;
        self
    }

    /// Set the global work queue capacity.
    pub fn with_work_queue_capacity(mut self, capacity: usize) -> Self {
        debug_assert!(
            capacity.is_power_of_two(),
            "Work queue capacity must be power of 2"
        );
        self.work_queue_capacity = capacity;
        self
    }

    /// Set the poll interval for the scheduler warp.
    pub fn with_poll_interval_ns(mut self, ns: u32) -> Self {
        self.poll_interval_ns = ns;
        self
    }

    /// Check if the scheduler warp pattern should be generated.
    pub fn is_enabled(&self) -> bool {
        self.scheduler.is_dynamic()
    }
}

/// Scheduling strategy for persistent actors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingStrategy {
    /// No dynamic scheduling — each actor processes its own queue only.
    Static,
    /// Work stealing — idle actors steal from busy neighbors.
    WorkStealing,
    /// Work sharing — busy actors proactively share with idle neighbors.
    WorkSharing,
    /// Hybrid — combines stealing and sharing based on load imbalance.
    Hybrid,
    /// Round-robin: a central work queue distributes to blocks in round-robin order.
    /// The scheduler warp in each block pulls from a global atomic counter.
    RoundRobin,
    /// Priority-based: actors have priority levels, higher priority served first.
    /// Messages are dequeued from the highest-priority non-empty sub-queue.
    Priority {
        /// Number of priority levels (1-16). Messages are bucketed into sub-queues.
        levels: u32,
    },
}

impl fmt::Display for SchedulingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Static => write!(f, "static"),
            Self::WorkStealing => write!(f, "work-stealing"),
            Self::WorkSharing => write!(f, "work-sharing"),
            Self::Hybrid => write!(f, "hybrid"),
            Self::RoundRobin => write!(f, "round-robin"),
            Self::Priority { levels } => write!(f, "priority({})", levels),
        }
    }
}

/// Per-actor load entry in the shared load table.
///
/// This structure is stored in mapped memory (or global memory on GPU)
/// so both the scheduler warp and the host can read/write it.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadEntry {
    /// Current input queue depth.
    pub queue_depth: u32,
    /// Queue capacity.
    pub capacity: u32,
    /// Total messages processed (monotonic counter for throughput).
    pub messages_processed: u64,
    /// Pending steal requests (atomically incremented by thieves).
    pub steal_requests: u32,
    /// Messages available for stealing (set by owner when depth > share_threshold).
    pub offer_count: u32,
    /// Load score: queue_depth * 256 / capacity (0-255, for fast comparison).
    pub load_score: u32,
    /// Padding to 32 bytes.
    pub _pad: u32,
}

impl LoadEntry {
    /// Compute the load score (0-255) from queue depth and capacity.
    pub fn compute_load_score(&mut self) {
        if self.capacity > 0 {
            self.load_score = ((self.queue_depth as u64 * 255) / self.capacity as u64) as u32;
        } else {
            self.load_score = 0;
        }
    }

    /// Check if this actor is overloaded (should offer work).
    pub fn is_overloaded(&self, threshold: u32) -> bool {
        self.queue_depth > threshold
    }

    /// Check if this actor is underloaded (should steal work).
    pub fn is_underloaded(&self, threshold: u32) -> bool {
        self.queue_depth < threshold
    }
}

/// The load table containing entries for all actors.
///
/// This lives in mapped memory so both host and GPU can access it.
pub struct LoadTable {
    entries: Vec<LoadEntry>,
}

impl LoadTable {
    /// Create a new load table for `num_actors` actors.
    pub fn new(num_actors: usize) -> Self {
        Self {
            entries: vec![LoadEntry::default(); num_actors],
        }
    }

    /// Get a reference to an entry.
    pub fn get(&self, actor_id: u32) -> Option<&LoadEntry> {
        self.entries.get(actor_id as usize)
    }

    /// Get a mutable reference to an entry.
    pub fn get_mut(&mut self, actor_id: u32) -> Option<&mut LoadEntry> {
        self.entries.get_mut(actor_id as usize)
    }

    /// Find the most loaded actor (best target for work stealing FROM).
    pub fn most_loaded(&self) -> Option<(u32, &LoadEntry)> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.queue_depth > 0)
            .max_by_key(|(_, e)| e.queue_depth)
            .map(|(i, e)| (i as u32, e))
    }

    /// Find the least loaded actor (best target for work sharing TO).
    pub fn least_loaded(&self) -> Option<(u32, &LoadEntry)> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.capacity > 0)
            .min_by_key(|(_, e)| e.queue_depth)
            .map(|(i, e)| (i as u32, e))
    }

    /// Compute load imbalance ratio (max_depth / min_depth).
    /// Returns 1.0 for perfectly balanced, higher = more imbalanced.
    pub fn imbalance_ratio(&self) -> f64 {
        let active: Vec<&LoadEntry> = self.entries.iter().filter(|e| e.capacity > 0).collect();
        if active.is_empty() {
            return 1.0;
        }

        let max = active.iter().map(|e| e.queue_depth).max().unwrap_or(0);
        let min = active.iter().map(|e| e.queue_depth).min().unwrap_or(0);

        if min == 0 {
            if max == 0 { 1.0 } else { f64::INFINITY }
        } else {
            max as f64 / min as f64
        }
    }

    /// Compute a work stealing plan: which actors should steal from which.
    ///
    /// Returns a list of (thief_id, victim_id, count) tuples.
    /// For `RoundRobin` and `Priority` strategies, this returns an empty plan
    /// since those use a central work queue rather than peer-to-peer stealing.
    pub fn compute_steal_plan(&self, config: &SchedulerConfig) -> Vec<StealOp> {
        if !config.enabled || config.strategy == SchedulingStrategy::Static {
            return Vec::new();
        }

        // Round-robin and priority use a central queue, not peer stealing.
        if matches!(
            config.strategy,
            SchedulingStrategy::RoundRobin | SchedulingStrategy::Priority { .. }
        ) {
            return Vec::new();
        }

        let mut ops = Vec::new();

        // Find underloaded actors (potential thieves)
        let thieves: Vec<u32> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.is_underloaded(config.steal_threshold) && e.capacity > 0)
            .map(|(i, _)| i as u32)
            .collect();

        // Find overloaded actors (potential victims)
        let mut victims: Vec<(u32, u32)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.is_overloaded(config.share_threshold))
            .map(|(i, e)| (i as u32, e.queue_depth - config.share_threshold))
            .collect();

        // Sort victims by excess load (descending)
        victims.sort_by(|a, b| b.1.cmp(&a.1));

        // Match thieves to victims
        let mut victim_idx = 0;
        for thief in &thieves {
            if victim_idx >= victims.len() {
                break;
            }

            let (victim_id, available) = &mut victims[victim_idx];
            if *available == 0 {
                victim_idx += 1;
                continue;
            }

            let steal_count = (*available).min(config.max_steal_batch);
            ops.push(StealOp {
                thief: *thief,
                victim: *victim_id,
                count: steal_count,
            });

            *available -= steal_count;
            if *available == 0 {
                victim_idx += 1;
            }
        }

        ops
    }

    /// Get all entries as a slice.
    pub fn entries(&self) -> &[LoadEntry] {
        &self.entries
    }

    /// Number of actors in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A single work-stealing operation.
#[derive(Debug, Clone, Copy)]
pub struct StealOp {
    /// Actor that will receive stolen work.
    pub thief: u32,
    /// Actor that work will be stolen from.
    pub victim: u32,
    /// Number of messages to transfer.
    pub count: u32,
}

impl fmt::Display for StealOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "steal {} msgs: actor {} ← actor {}",
            self.count, self.thief, self.victim
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_config_defaults() {
        let config = SchedulerConfig::default();
        assert_eq!(config.steal_threshold, 4);
        assert_eq!(config.share_threshold, 64);
        assert!(config.enabled);
    }

    #[test]
    fn test_load_entry_score() {
        let mut entry = LoadEntry {
            queue_depth: 50,
            capacity: 100,
            ..Default::default()
        };
        entry.compute_load_score();
        assert_eq!(entry.load_score, 127); // 50/100 * 255 ≈ 127
    }

    #[test]
    fn test_load_table_most_least_loaded() {
        let mut table = LoadTable::new(4);
        table.entries[0] = LoadEntry { queue_depth: 10, capacity: 100, ..Default::default() };
        table.entries[1] = LoadEntry { queue_depth: 90, capacity: 100, ..Default::default() };
        table.entries[2] = LoadEntry { queue_depth: 50, capacity: 100, ..Default::default() };
        table.entries[3] = LoadEntry { queue_depth: 5, capacity: 100, ..Default::default() };

        let (most_id, most) = table.most_loaded().unwrap();
        assert_eq!(most_id, 1);
        assert_eq!(most.queue_depth, 90);

        let (least_id, least) = table.least_loaded().unwrap();
        assert_eq!(least_id, 3);
        assert_eq!(least.queue_depth, 5);
    }

    #[test]
    fn test_imbalance_ratio() {
        let mut table = LoadTable::new(4);
        // Balanced: all depth 50
        for e in &mut table.entries {
            e.queue_depth = 50;
            e.capacity = 100;
        }
        assert!((table.imbalance_ratio() - 1.0).abs() < 0.01);

        // Imbalanced: 10 vs 100
        table.entries[0].queue_depth = 10;
        table.entries[1].queue_depth = 100;
        assert!((table.imbalance_ratio() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_steal_plan_static_disabled() {
        let table = LoadTable::new(4);
        let config = SchedulerConfig {
            strategy: SchedulingStrategy::Static,
            ..Default::default()
        };
        let plan = table.compute_steal_plan(&config);
        assert!(plan.is_empty());
    }

    #[test]
    fn test_steal_plan_work_stealing() {
        let mut table = LoadTable::new(4);
        // Actor 0: idle (depth 2, below steal threshold 4)
        table.entries[0] = LoadEntry { queue_depth: 2, capacity: 100, ..Default::default() };
        // Actor 1: overloaded (depth 80, above share threshold 64)
        table.entries[1] = LoadEntry { queue_depth: 80, capacity: 100, ..Default::default() };
        // Actor 2: normal
        table.entries[2] = LoadEntry { queue_depth: 30, capacity: 100, ..Default::default() };
        // Actor 3: idle
        table.entries[3] = LoadEntry { queue_depth: 1, capacity: 100, ..Default::default() };

        let config = SchedulerConfig::default();
        let plan = table.compute_steal_plan(&config);

        assert!(!plan.is_empty(), "Should produce steal operations");
        // Actor 0 and 3 should steal from actor 1
        assert!(plan.iter().all(|op| op.victim == 1));
        assert!(plan.iter().any(|op| op.thief == 0 || op.thief == 3));
    }

    #[test]
    fn test_steal_plan_respects_max_batch() {
        let mut table = LoadTable::new(2);
        table.entries[0] = LoadEntry { queue_depth: 0, capacity: 100, ..Default::default() };
        table.entries[1] = LoadEntry { queue_depth: 100, capacity: 100, ..Default::default() };

        let config = SchedulerConfig {
            max_steal_batch: 8,
            ..Default::default()
        };
        let plan = table.compute_steal_plan(&config);

        assert!(!plan.is_empty());
        for op in &plan {
            assert!(op.count <= 8, "Steal count {} exceeds max batch 8", op.count);
        }
    }

    #[test]
    fn test_load_entry_size() {
        assert_eq!(
            std::mem::size_of::<LoadEntry>(),
            32,
            "LoadEntry must be 32 bytes for GPU cache efficiency"
        );
    }

    #[test]
    fn test_work_item_size() {
        assert_eq!(
            std::mem::size_of::<WorkItem>(),
            16,
            "WorkItem must be 16 bytes for GPU cache efficiency"
        );
    }

    #[test]
    fn test_work_item_display() {
        let item = WorkItem::new(3, 42, 2);
        let s = format!("{}", item);
        assert!(s.contains("actor=3"));
        assert!(s.contains("msg=42"));
        assert!(s.contains("pri=2"));
    }

    #[test]
    fn test_scheduler_config_static() {
        let config = SchedulerConfig::static_scheduling();
        assert!(!config.enabled);
        assert_eq!(config.strategy, SchedulingStrategy::Static);
        assert!(!config.is_dynamic());
    }

    #[test]
    fn test_scheduler_config_work_stealing() {
        let config = SchedulerConfig::work_stealing(8);
        assert_eq!(config.steal_threshold, 8);
        assert_eq!(config.strategy, SchedulingStrategy::WorkStealing);
        assert!(config.is_dynamic());
    }

    #[test]
    fn test_scheduler_config_round_robin() {
        let config = SchedulerConfig::round_robin();
        assert_eq!(config.strategy, SchedulingStrategy::RoundRobin);
        assert!(config.is_dynamic());
    }

    #[test]
    fn test_scheduler_config_priority() {
        let config = SchedulerConfig::priority(4);
        assert_eq!(config.strategy, SchedulingStrategy::Priority { levels: 4 });
        assert!(config.is_dynamic());
    }

    #[test]
    fn test_scheduler_config_priority_clamped() {
        let config = SchedulerConfig::priority(100);
        assert_eq!(config.strategy, SchedulingStrategy::Priority { levels: 16 });
    }

    #[test]
    fn test_scheduler_config_builder_chain() {
        let config = SchedulerConfig::work_stealing(10)
            .with_share_threshold(80)
            .with_max_steal_batch(32)
            .with_steal_neighborhood(6);

        assert_eq!(config.steal_threshold, 10);
        assert_eq!(config.share_threshold, 80);
        assert_eq!(config.max_steal_batch, 32);
        assert_eq!(config.steal_neighborhood, 6);
    }

    #[test]
    fn test_scheduler_warp_config_default() {
        let config = SchedulerWarpConfig::default();
        assert_eq!(config.scheduler_warp_id, 0);
        assert_eq!(config.work_queue_capacity, 1024);
        assert_eq!(config.poll_interval_ns, 1000);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_scheduler_warp_config_disabled() {
        let config = SchedulerWarpConfig::disabled();
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_scheduler_warp_config_builder() {
        let config = SchedulerWarpConfig::new(SchedulerConfig::round_robin())
            .with_scheduler_warp(1)
            .with_work_queue_capacity(2048)
            .with_poll_interval_ns(500);

        assert_eq!(config.scheduler_warp_id, 1);
        assert_eq!(config.work_queue_capacity, 2048);
        assert_eq!(config.poll_interval_ns, 500);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(format!("{}", SchedulingStrategy::Static), "static");
        assert_eq!(format!("{}", SchedulingStrategy::WorkStealing), "work-stealing");
        assert_eq!(format!("{}", SchedulingStrategy::WorkSharing), "work-sharing");
        assert_eq!(format!("{}", SchedulingStrategy::Hybrid), "hybrid");
        assert_eq!(format!("{}", SchedulingStrategy::RoundRobin), "round-robin");
        assert_eq!(
            format!("{}", SchedulingStrategy::Priority { levels: 4 }),
            "priority(4)"
        );
    }

    #[test]
    fn test_steal_plan_round_robin_empty() {
        let mut table = LoadTable::new(4);
        table.entries[0] = LoadEntry { queue_depth: 2, capacity: 100, ..Default::default() };
        table.entries[1] = LoadEntry { queue_depth: 80, capacity: 100, ..Default::default() };

        let config = SchedulerConfig::round_robin();
        let plan = table.compute_steal_plan(&config);
        assert!(plan.is_empty(), "Round-robin should not produce steal ops");
    }

    #[test]
    fn test_steal_plan_priority_empty() {
        let mut table = LoadTable::new(4);
        table.entries[0] = LoadEntry { queue_depth: 2, capacity: 100, ..Default::default() };
        table.entries[1] = LoadEntry { queue_depth: 80, capacity: 100, ..Default::default() };

        let config = SchedulerConfig::priority(4);
        let plan = table.compute_steal_plan(&config);
        assert!(plan.is_empty(), "Priority should not produce steal ops");
    }
}
