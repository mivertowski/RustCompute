//! GPU Actor Lifecycle Model
//!
//! Implements the full actor model lifecycle on GPU hardware:
//! - **Create**: Activate a dormant actor slot from the pool
//! - **Destroy**: Deactivate an actor, return its slot to the pool
//! - **Restart**: Destroy + reinitialize state + Create
//! - **Supervise**: Parent-child tree with failure detection and restart policies
//!
//! # Architecture
//!
//! On a GPU, thread blocks are fixed at kernel launch time — you can't spawn
//! new blocks dynamically. The solution is a **pool-based** design:
//!
//! ```text
//! ┌─── Persistent Kernel (N blocks pre-allocated at launch) ──────────┐
//! │                                                                    │
//! │  Block 0: SUPERVISOR                                               │
//! │  ├─ Actor registry (who is active, who is dormant)                 │
//! │  ├─ Free list (available actor slots)                              │
//! │  ├─ Supervision tree (parent-child relationships)                  │
//! │  └─ Heartbeat monitor (detect actor failures)                      │
//! │                                                                    │
//! │  Block 1: ACTIVE actor "sensor-reader"                             │
//! │  ├─ Processing messages from H2K queue                             │
//! │  ├─ Created child: Block 3                                         │
//! │  └─ Heartbeat: last_seen = 1.2ms ago                               │
//! │                                                                    │
//! │  Block 2: DORMANT (in free pool)                                   │
//! │  └─ Zero cost when idle — just checks is_active flag               │
//! │                                                                    │
//! │  Block 3: ACTIVE actor "data-processor" (child of Block 1)         │
//! │  ├─ Processing K2K messages from Block 1                           │
//! │  └─ Heartbeat: last_seen = 0.5ms ago                               │
//! │                                                                    │
//! │  Block 4-N: DORMANT (in free pool)                                 │
//! │                                                                    │
//! │  "Create actor" = supervisor activates dormant block               │
//! │  "Kill actor"   = supervisor deactivates, returns to free pool     │
//! │  "Restart"      = destroy + reinit state + create                  │
//! └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Comparison with Erlang/Akka
//!
//! | Erlang/Akka | RingKernel GPU |
//! |-------------|----------------|
//! | `spawn(Fun)` | `supervisor.create_actor(config)` → activates dormant block |
//! | `Pid ! Message` | K2K channel or H2K queue injection |
//! | `exit(Pid, Reason)` | `supervisor.destroy_actor(id)` → deactivates block |
//! | Supervisor tree | Parent-child tree in mapped memory |
//! | `one_for_one` restart | Heartbeat timeout → destroy + create |
//! | Process isolation | Separate control block + queue per block |

use std::fmt;
use std::time::Duration;

/// Unique identifier for a GPU actor within a persistent kernel.
///
/// This maps to a thread block index in the persistent kernel grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActorId(pub u32);

impl fmt::Display for ActorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "actor:{}", self.0)
    }
}

/// State of a GPU actor slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ActorState {
    /// Slot is unused and available for allocation.
    Dormant = 0,
    /// Actor is initializing (setting up state before becoming active).
    Initializing = 1,
    /// Actor is active and processing messages.
    Active = 2,
    /// Actor is draining its queue before shutdown.
    Draining = 3,
    /// Actor has terminated (waiting to be reclaimed by supervisor).
    Terminated = 4,
    /// Actor encountered a fatal error.
    Failed = 5,
}

impl ActorState {
    /// Create from raw u32 value (for reading from mapped memory).
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Dormant),
            1 => Some(Self::Initializing),
            2 => Some(Self::Active),
            3 => Some(Self::Draining),
            4 => Some(Self::Terminated),
            5 => Some(Self::Failed),
            _ => None,
        }
    }

    /// Check if the actor is alive (initializing or active).
    pub fn is_alive(self) -> bool {
        matches!(self, Self::Initializing | Self::Active)
    }
}

/// Restart policy for supervised actors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartPolicy {
    /// Never restart — if the actor fails, it stays failed.
    Permanent,
    /// Restart on failure, up to `max_restarts` within `window`.
    OneForOne {
        /// Maximum restart attempts within the time window.
        max_restarts: u32,
        /// Time window for counting restarts.
        window: Duration,
    },
    /// Restart the failed actor and all its siblings (children of the same parent).
    OneForAll {
        /// Maximum restart attempts within the time window.
        max_restarts: u32,
        /// Time window for counting restarts.
        window: Duration,
    },
    /// Restart the failed actor and all actors that were started after it.
    RestForOne {
        /// Maximum restart attempts within the time window.
        max_restarts: u32,
        /// Time window for counting restarts.
        window: Duration,
    },
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self::OneForOne {
            max_restarts: 3,
            window: Duration::from_secs(60),
        }
    }
}

/// Configuration for creating a new GPU actor.
#[derive(Debug, Clone)]
pub struct ActorConfig {
    /// Human-readable name for the actor.
    pub name: String,
    /// Message queue capacity for this actor.
    pub queue_capacity: u32,
    /// Restart policy if the actor fails.
    pub restart_policy: RestartPolicy,
    /// Heartbeat interval — supervisor checks liveness at this rate.
    pub heartbeat_interval: Duration,
    /// Heartbeat timeout — actor is considered failed if no heartbeat for this long.
    pub heartbeat_timeout: Duration,
    /// Initial state data to copy into the actor's shared memory.
    pub initial_state: Option<Vec<u8>>,
}

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            queue_capacity: 1024,
            restart_policy: RestartPolicy::default(),
            heartbeat_interval: Duration::from_millis(100),
            heartbeat_timeout: Duration::from_millis(500),
            initial_state: None,
        }
    }
}

impl ActorConfig {
    /// Create a config with a name.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set the restart policy.
    pub fn with_restart_policy(mut self, policy: RestartPolicy) -> Self {
        self.restart_policy = policy;
        self
    }

    /// Set heartbeat parameters.
    pub fn with_heartbeat(mut self, interval: Duration, timeout: Duration) -> Self {
        self.heartbeat_interval = interval;
        self.heartbeat_timeout = timeout;
        self
    }

    /// Set the queue capacity.
    pub fn with_queue_capacity(mut self, capacity: u32) -> Self {
        self.queue_capacity = capacity;
        self
    }
}

/// Entry in the supervision tree.
///
/// Stored in mapped memory so both CPU and GPU can access it.
/// Fixed 64-byte size for GPU cache efficiency.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct SupervisionEntry {
    /// Actor ID (thread block index).
    pub actor_id: u32,
    /// Current state (see ActorState).
    pub state: u32,
    /// Parent actor ID (0 = root / no parent).
    pub parent_id: u32,
    /// Restart count within current window.
    pub restart_count: u32,
    /// Last heartbeat timestamp (nanoseconds since kernel start).
    pub last_heartbeat_ns: u64,
    /// Restart window start timestamp.
    pub restart_window_start_ns: u64,
    /// Maximum restarts allowed in window.
    pub max_restarts: u32,
    /// Restart window duration in nanoseconds.
    pub restart_window_ns: u64,
    /// Padding to 64 bytes.
    pub _pad: [u8; 8],
}

impl SupervisionEntry {
    /// Create a new dormant entry.
    pub fn dormant(actor_id: u32) -> Self {
        Self {
            actor_id,
            state: ActorState::Dormant as u32,
            parent_id: 0,
            restart_count: 0,
            last_heartbeat_ns: 0,
            restart_window_start_ns: 0,
            max_restarts: 3,
            restart_window_ns: 60_000_000_000, // 60 seconds
            _pad: [0; 8],
        }
    }

    /// Check if this entry is available for allocation.
    pub fn is_available(&self) -> bool {
        self.state == ActorState::Dormant as u32
    }

    /// Get the actor state.
    pub fn actor_state(&self) -> ActorState {
        ActorState::from_u32(self.state).unwrap_or(ActorState::Failed)
    }
}

/// The supervisor manages the actor pool lifecycle.
///
/// This is the host-side component. The GPU-side supervisor runs as
/// block 0 in the persistent kernel and mirrors this state.
pub struct ActorSupervisor {
    /// Supervision tree entries (one per thread block in the grid).
    entries: Vec<SupervisionEntry>,
    /// Free list (indices of dormant actors).
    free_list: Vec<u32>,
    /// Total actor slots (= grid size of persistent kernel).
    capacity: u32,
    /// Number of currently active actors.
    active_count: u32,
}

impl ActorSupervisor {
    /// Create a supervisor for a persistent kernel with `grid_size` blocks.
    ///
    /// Block 0 is reserved for the supervisor itself.
    /// Blocks 1..grid_size are available as actor slots.
    pub fn new(grid_size: u32) -> Self {
        let mut entries = Vec::with_capacity(grid_size as usize);
        let mut free_list = Vec::with_capacity(grid_size as usize);

        for i in 0..grid_size {
            entries.push(SupervisionEntry::dormant(i));
            if i > 0 {
                // Block 0 = supervisor, blocks 1+ = actor pool
                free_list.push(i);
            }
        }

        Self {
            entries,
            free_list,
            capacity: grid_size - 1, // exclude supervisor block
            active_count: 0,
        }
    }

    /// Create a new actor, returning its ID.
    ///
    /// Allocates a dormant slot from the free pool, initializes it with
    /// the given config, and sets its state to Initializing.
    pub fn create_actor(
        &mut self,
        config: &ActorConfig,
        parent_id: Option<ActorId>,
    ) -> Result<ActorId, ActorError> {
        let slot = self.free_list.pop().ok_or(ActorError::PoolExhausted {
            capacity: self.capacity,
            active: self.active_count,
        })?;

        let entry = &mut self.entries[slot as usize];
        entry.state = ActorState::Initializing as u32;
        entry.parent_id = parent_id.map(|p| p.0).unwrap_or(0);
        entry.restart_count = 0;
        entry.last_heartbeat_ns = 0;

        match config.restart_policy {
            RestartPolicy::OneForOne {
                max_restarts,
                window,
            }
            | RestartPolicy::OneForAll {
                max_restarts,
                window,
            }
            | RestartPolicy::RestForOne {
                max_restarts,
                window,
            } => {
                entry.max_restarts = max_restarts;
                entry.restart_window_ns = window.as_nanos() as u64;
            }
            RestartPolicy::Permanent => {
                entry.max_restarts = 0;
                entry.restart_window_ns = 0;
            }
        }

        self.active_count += 1;

        Ok(ActorId(slot))
    }

    /// Activate an actor (transition from Initializing to Active).
    pub fn activate_actor(&mut self, id: ActorId) -> Result<(), ActorError> {
        let entry = self
            .entries
            .get_mut(id.0 as usize)
            .ok_or(ActorError::InvalidId(id))?;

        if entry.state != ActorState::Initializing as u32 {
            return Err(ActorError::InvalidStateTransition {
                actor: id,
                from: entry.actor_state(),
                to: ActorState::Active,
            });
        }

        entry.state = ActorState::Active as u32;
        Ok(())
    }

    /// Destroy an actor, returning its slot to the free pool.
    pub fn destroy_actor(&mut self, id: ActorId) -> Result<(), ActorError> {
        let entry = self
            .entries
            .get_mut(id.0 as usize)
            .ok_or(ActorError::InvalidId(id))?;

        if entry.state == ActorState::Dormant as u32 {
            return Err(ActorError::InvalidStateTransition {
                actor: id,
                from: ActorState::Dormant,
                to: ActorState::Terminated,
            });
        }

        entry.state = ActorState::Dormant as u32;
        entry.parent_id = 0;
        entry.restart_count = 0;
        entry.last_heartbeat_ns = 0;

        self.free_list.push(id.0);
        self.active_count = self.active_count.saturating_sub(1);

        Ok(())
    }

    /// Restart an actor (destroy + re-create with same config).
    ///
    /// Returns the new ActorId (may be the same slot if available).
    pub fn restart_actor(
        &mut self,
        id: ActorId,
        config: &ActorConfig,
    ) -> Result<ActorId, ActorError> {
        let parent_id = {
            let entry = self
                .entries
                .get(id.0 as usize)
                .ok_or(ActorError::InvalidId(id))?;

            // Check restart budget
            if entry.restart_count >= entry.max_restarts && entry.max_restarts > 0 {
                return Err(ActorError::MaxRestartsExceeded {
                    actor: id,
                    restarts: entry.restart_count,
                    max: entry.max_restarts,
                });
            }

            if entry.parent_id > 0 {
                Some(ActorId(entry.parent_id))
            } else {
                None
            }
        };

        // Increment restart count before destroy (to preserve it)
        let restart_count = self.entries[id.0 as usize].restart_count;

        self.destroy_actor(id)?;
        let new_id = self.create_actor(config, parent_id)?;

        // Restore restart count (incremented)
        self.entries[new_id.0 as usize].restart_count = restart_count + 1;

        Ok(new_id)
    }

    /// Record a heartbeat from an actor.
    pub fn heartbeat(&mut self, id: ActorId, timestamp_ns: u64) {
        if let Some(entry) = self.entries.get_mut(id.0 as usize) {
            entry.last_heartbeat_ns = timestamp_ns;
        }
    }

    /// Check for actors that have missed their heartbeat deadline.
    ///
    /// Returns a list of actor IDs that should be restarted.
    pub fn check_heartbeats(&self, now_ns: u64, timeout_ns: u64) -> Vec<ActorId> {
        self.entries
            .iter()
            .filter(|e| {
                e.actor_state().is_alive()
                    && e.last_heartbeat_ns > 0
                    && (now_ns - e.last_heartbeat_ns) > timeout_ns
            })
            .map(|e| ActorId(e.actor_id))
            .collect()
    }

    /// Get children of a given actor.
    pub fn children_of(&self, parent: ActorId) -> Vec<ActorId> {
        self.entries
            .iter()
            .filter(|e| e.parent_id == parent.0 && e.actor_state().is_alive())
            .map(|e| ActorId(e.actor_id))
            .collect()
    }

    /// Get the supervision entry for an actor.
    pub fn get(&self, id: ActorId) -> Option<&SupervisionEntry> {
        self.entries.get(id.0 as usize)
    }

    /// Number of currently active actors.
    pub fn active_count(&self) -> u32 {
        self.active_count
    }

    /// Number of available (dormant) actor slots.
    pub fn available_count(&self) -> u32 {
        self.free_list.len() as u32
    }

    /// Total capacity (excluding supervisor block).
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Get all supervision entries as a slice (for copying to mapped memory).
    pub fn entries(&self) -> &[SupervisionEntry] {
        &self.entries
    }

    // === FR-001: Cascading Termination & Escalation ===

    /// Cascading kill: destroy an actor and all its descendants.
    ///
    /// Walks the supervision tree depth-first, destroying children
    /// before parents. Returns the list of destroyed actor IDs.
    pub fn kill_tree(&mut self, root: ActorId) -> Vec<ActorId> {
        let mut destroyed = Vec::new();
        self.kill_tree_recursive(root, &mut destroyed);
        destroyed
    }

    fn kill_tree_recursive(&mut self, id: ActorId, destroyed: &mut Vec<ActorId>) {
        // First, recursively kill all children
        let children = self.children_of(id);
        for child in children {
            self.kill_tree_recursive(child, destroyed);
        }

        // Then destroy this actor
        if self.destroy_actor(id).is_ok() {
            destroyed.push(id);
        }
    }

    /// Handle a failed actor according to its parent's restart policy.
    ///
    /// Returns a list of actions taken (for logging/auditing).
    pub fn handle_failure(
        &mut self,
        failed_id: ActorId,
        config: &ActorConfig,
    ) -> Vec<SupervisionAction> {
        let mut actions = Vec::new();

        let (parent_id, policy) = {
            let entry = match self.get(failed_id) {
                Some(e) => e,
                None => return actions,
            };
            let parent = if entry.parent_id > 0 {
                Some(ActorId(entry.parent_id))
            } else {
                None
            };
            (parent, config.restart_policy)
        };

        // Mark as failed
        if let Some(entry) = self.entries.get_mut(failed_id.0 as usize) {
            entry.state = ActorState::Failed as u32;
        }
        actions.push(SupervisionAction::MarkedFailed(failed_id));

        match policy {
            RestartPolicy::Permanent => {
                // No restart — escalate to parent
                actions.push(SupervisionAction::Escalated {
                    failed: failed_id,
                    escalated_to: parent_id,
                });
            }

            RestartPolicy::OneForOne { .. } => {
                // Restart only the failed actor
                match self.restart_actor(failed_id, config) {
                    Ok(new_id) => {
                        actions.push(SupervisionAction::Restarted {
                            old_id: failed_id,
                            new_id,
                        });
                    }
                    Err(ActorError::MaxRestartsExceeded { .. }) => {
                        // Budget exhausted — escalate to parent
                        actions.push(SupervisionAction::Escalated {
                            failed: failed_id,
                            escalated_to: parent_id,
                        });
                    }
                    Err(_) => {
                        actions.push(SupervisionAction::Escalated {
                            failed: failed_id,
                            escalated_to: parent_id,
                        });
                    }
                }
            }

            RestartPolicy::OneForAll { .. } => {
                // Restart the failed actor AND all its siblings
                if let Some(parent) = parent_id {
                    let siblings = self.children_of(parent);
                    for sibling in siblings {
                        let _ = self.destroy_actor(sibling);
                        actions.push(SupervisionAction::DestroyedSibling(sibling));
                    }
                    // Re-create all (including the failed one)
                    // Note: caller is responsible for re-creating with proper configs
                    actions.push(SupervisionAction::AllSiblingsDestroyed { parent });
                }
            }

            RestartPolicy::RestForOne { .. } => {
                // Restart the failed actor and all actors started after it
                if let Some(parent) = parent_id {
                    let siblings = self.children_of(parent);
                    let mut found = false;
                    for sibling in siblings {
                        if sibling == failed_id {
                            found = true;
                        }
                        if found {
                            let _ = self.destroy_actor(sibling);
                            actions.push(SupervisionAction::DestroyedSibling(sibling));
                        }
                    }
                }
            }
        }

        actions
    }

    /// Get the depth of the supervision tree from root to the given actor.
    pub fn depth(&self, id: ActorId) -> u32 {
        let mut depth = 0;
        let mut current = id;
        while let Some(entry) = self.get(current) {
            if entry.parent_id == 0 {
                break;
            }
            current = ActorId(entry.parent_id);
            depth += 1;
            if depth > 100 {
                break; // Safety: prevent infinite loop on corrupted tree
            }
        }
        depth
    }

    /// Produce a textual visualization of the supervision tree.
    pub fn tree_view(&self) -> String {
        let mut out = String::new();
        out.push_str("Supervision Tree:\n");

        // Find root actors (no parent)
        let roots: Vec<ActorId> = self
            .entries
            .iter()
            .filter(|e| e.parent_id == 0 && e.actor_state().is_alive())
            .map(|e| ActorId(e.actor_id))
            .collect();

        for root in roots {
            self.tree_view_recursive(root, &mut out, 0);
        }

        out
    }

    fn tree_view_recursive(&self, id: ActorId, out: &mut String, indent: usize) {
        if let Some(entry) = self.get(id) {
            let prefix = "  ".repeat(indent);
            let state = match entry.actor_state() {
                ActorState::Active => "ACTIVE",
                ActorState::Dormant => "DORMANT",
                ActorState::Initializing => "INIT",
                ActorState::Failed => "FAILED",
                ActorState::Terminated => "TERM",
                ActorState::Draining => "DRAIN",
            };
            out.push_str(&format!(
                "{}[{}] actor:{} state={} restarts={} msgs={}\n",
                prefix,
                if indent == 0 { "R" } else { "C" },
                id.0,
                state,
                entry.restart_count,
                entry.last_heartbeat_ns
            ));

            let children = self.children_of(id);
            for child in children {
                self.tree_view_recursive(child, out, indent + 1);
            }
        }
    }
}

/// Action taken by the supervisor in response to a failure.
#[derive(Debug, Clone)]
pub enum SupervisionAction {
    /// Actor marked as failed.
    MarkedFailed(ActorId),
    /// Actor was restarted.
    Restarted {
        /// The original actor ID before restart.
        old_id: ActorId,
        /// The new actor ID after restart.
        new_id: ActorId,
    },
    /// A sibling was destroyed (OneForAll/RestForOne).
    DestroyedSibling(ActorId),
    /// All siblings destroyed, parent needs to re-create them.
    AllSiblingsDestroyed {
        /// Parent actor that owns the destroyed siblings.
        parent: ActorId,
    },
    /// Failure escalated to parent supervisor.
    Escalated {
        /// Actor that failed.
        failed: ActorId,
        /// Parent supervisor the failure was escalated to.
        escalated_to: Option<ActorId>,
    },
}

/// Errors from actor lifecycle operations.
#[derive(Debug, Clone)]
pub enum ActorError {
    /// No available actor slots in the pool.
    PoolExhausted {
        /// Total pool capacity.
        capacity: u32,
        /// Number of currently active actors.
        active: u32,
    },
    /// Invalid actor ID.
    InvalidId(ActorId),
    /// Invalid state transition.
    InvalidStateTransition {
        /// Actor that attempted the invalid transition.
        actor: ActorId,
        /// Current state of the actor.
        from: ActorState,
        /// Attempted target state.
        to: ActorState,
    },
    /// Actor has exceeded its maximum restart budget.
    MaxRestartsExceeded {
        /// Actor that exceeded its restart budget.
        actor: ActorId,
        /// Number of restarts attempted.
        restarts: u32,
        /// Maximum allowed restarts.
        max: u32,
    },
}

impl fmt::Display for ActorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PoolExhausted { capacity, active } => write!(
                f,
                "Actor pool exhausted: {}/{} slots active",
                active, capacity
            ),
            Self::InvalidId(id) => write!(f, "Invalid actor ID: {}", id),
            Self::InvalidStateTransition { actor, from, to } => write!(
                f,
                "Invalid state transition for {}: {:?} → {:?}",
                actor, from, to
            ),
            Self::MaxRestartsExceeded {
                actor,
                restarts,
                max,
            } => write!(
                f,
                "Actor {} exceeded max restarts: {}/{}",
                actor, restarts, max
            ),
        }
    }
}

impl std::error::Error for ActorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_lifecycle_create_destroy() {
        let mut supervisor = ActorSupervisor::new(8); // 7 actor slots + 1 supervisor

        assert_eq!(supervisor.capacity(), 7);
        assert_eq!(supervisor.active_count(), 0);
        assert_eq!(supervisor.available_count(), 7);

        // Create an actor
        let config = ActorConfig::named("test-actor");
        let id = supervisor.create_actor(&config, None).unwrap();
        assert_eq!(supervisor.active_count(), 1);
        assert_eq!(supervisor.available_count(), 6);

        // Activate it
        supervisor.activate_actor(id).unwrap();
        let entry = supervisor.get(id).unwrap();
        assert_eq!(entry.actor_state(), ActorState::Active);

        // Destroy it
        supervisor.destroy_actor(id).unwrap();
        assert_eq!(supervisor.active_count(), 0);
        assert_eq!(supervisor.available_count(), 7);
    }

    #[test]
    fn test_actor_parent_child() {
        let mut supervisor = ActorSupervisor::new(8);

        let parent_config = ActorConfig::named("parent");
        let parent = supervisor.create_actor(&parent_config, None).unwrap();
        supervisor.activate_actor(parent).unwrap();

        // Create children
        let child_config = ActorConfig::named("child");
        let child1 = supervisor
            .create_actor(&child_config, Some(parent))
            .unwrap();
        let child2 = supervisor
            .create_actor(&child_config, Some(parent))
            .unwrap();
        supervisor.activate_actor(child1).unwrap();
        supervisor.activate_actor(child2).unwrap();

        let children = supervisor.children_of(parent);
        assert_eq!(children.len(), 2);
        assert!(children.contains(&child1));
        assert!(children.contains(&child2));
    }

    #[test]
    fn test_actor_restart() {
        let mut supervisor = ActorSupervisor::new(8);
        let config =
            ActorConfig::named("restartable").with_restart_policy(RestartPolicy::OneForOne {
                max_restarts: 3,
                window: Duration::from_secs(60),
            });

        let id = supervisor.create_actor(&config, None).unwrap();
        supervisor.activate_actor(id).unwrap();

        // Restart 3 times (should succeed)
        for i in 0..3 {
            let new_id = supervisor.restart_actor(id, &config).unwrap();
            supervisor.activate_actor(new_id).unwrap();
            let entry = supervisor.get(new_id).unwrap();
            assert_eq!(entry.restart_count, i + 1);
        }
    }

    #[test]
    fn test_actor_max_restarts_exceeded() {
        let mut supervisor = ActorSupervisor::new(8);
        let config = ActorConfig::named("fragile").with_restart_policy(RestartPolicy::OneForOne {
            max_restarts: 1,
            window: Duration::from_secs(60),
        });

        let id = supervisor.create_actor(&config, None).unwrap();
        supervisor.activate_actor(id).unwrap();

        // First restart: OK
        let new_id = supervisor.restart_actor(id, &config).unwrap();
        supervisor.activate_actor(new_id).unwrap();

        // Second restart: should fail
        let result = supervisor.restart_actor(new_id, &config);
        assert!(matches!(
            result,
            Err(ActorError::MaxRestartsExceeded { .. })
        ));
    }

    #[test]
    fn test_pool_exhaustion() {
        let mut supervisor = ActorSupervisor::new(4); // 3 actor slots
        let config = ActorConfig::named("actor");

        // Fill the pool
        for _ in 0..3 {
            let id = supervisor.create_actor(&config, None).unwrap();
            supervisor.activate_actor(id).unwrap();
        }

        // Next create should fail
        let result = supervisor.create_actor(&config, None);
        assert!(matches!(result, Err(ActorError::PoolExhausted { .. })));
    }

    #[test]
    fn test_heartbeat_failure_detection() {
        let mut supervisor = ActorSupervisor::new(8);
        let config = ActorConfig::named("monitored");

        let id1 = supervisor.create_actor(&config, None).unwrap();
        let id2 = supervisor.create_actor(&config, None).unwrap();
        supervisor.activate_actor(id1).unwrap();
        supervisor.activate_actor(id2).unwrap();

        // Actor 1 sends heartbeat, actor 2 doesn't
        supervisor.heartbeat(id1, 1_000_000); // 1ms
        supervisor.heartbeat(id2, 1_000_000); // 1ms

        // Check at 600ms — both should be fine (timeout is 500ms but we'll set a custom one)
        let timeout_ns = 500_000_000; // 500ms

        // Advance time: actor 1 sends again at 400ms, actor 2 is silent
        supervisor.heartbeat(id1, 400_000_000);

        // Check at 600ms — actor 2 should be timed out
        let failed = supervisor.check_heartbeats(600_000_000, timeout_ns);
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0], id2);
    }

    #[test]
    fn test_supervision_entry_size() {
        assert_eq!(
            std::mem::size_of::<SupervisionEntry>(),
            64,
            "SupervisionEntry must be exactly 64 bytes for GPU cache efficiency"
        );
    }

    #[test]
    fn test_actor_state_roundtrip() {
        for state in [
            ActorState::Dormant,
            ActorState::Initializing,
            ActorState::Active,
            ActorState::Draining,
            ActorState::Terminated,
            ActorState::Failed,
        ] {
            let raw = state as u32;
            let recovered = ActorState::from_u32(raw).unwrap();
            assert_eq!(recovered, state);
        }
    }

    // === FR-001: Cascading termination & escalation tests ===

    #[test]
    fn test_cascading_kill_tree() {
        let mut sup = ActorSupervisor::new(16);
        let config = ActorConfig::named("node");

        // Build a tree: root → child1, child2; child1 → grandchild1
        let root = sup.create_actor(&config, None).unwrap();
        sup.activate_actor(root).unwrap();

        let child1 = sup.create_actor(&config, Some(root)).unwrap();
        sup.activate_actor(child1).unwrap();

        let child2 = sup.create_actor(&config, Some(root)).unwrap();
        sup.activate_actor(child2).unwrap();

        let grandchild1 = sup.create_actor(&config, Some(child1)).unwrap();
        sup.activate_actor(grandchild1).unwrap();

        assert_eq!(sup.active_count(), 4);

        // Kill the root — should cascade to all descendants
        let destroyed = sup.kill_tree(root);
        assert_eq!(destroyed.len(), 4);
        assert_eq!(sup.active_count(), 0);
        assert_eq!(sup.available_count(), 15); // All back in pool
    }

    #[test]
    fn test_handle_failure_one_for_one() {
        let mut sup = ActorSupervisor::new(8);
        let config = ActorConfig::named("worker").with_restart_policy(RestartPolicy::OneForOne {
            max_restarts: 2,
            window: Duration::from_secs(60),
        });

        let parent = sup
            .create_actor(&ActorConfig::named("parent"), None)
            .unwrap();
        sup.activate_actor(parent).unwrap();

        let child = sup.create_actor(&config, Some(parent)).unwrap();
        sup.activate_actor(child).unwrap();

        // Simulate failure
        let actions = sup.handle_failure(child, &config);
        assert!(actions
            .iter()
            .any(|a| matches!(a, SupervisionAction::Restarted { .. })));
    }

    #[test]
    fn test_handle_failure_escalation() {
        let mut sup = ActorSupervisor::new(8);
        let config = ActorConfig::named("fragile").with_restart_policy(RestartPolicy::Permanent);

        let parent = sup
            .create_actor(&ActorConfig::named("parent"), None)
            .unwrap();
        sup.activate_actor(parent).unwrap();

        let child = sup.create_actor(&config, Some(parent)).unwrap();
        sup.activate_actor(child).unwrap();

        // Permanent policy → failure escalates to parent
        let actions = sup.handle_failure(child, &config);
        assert!(actions
            .iter()
            .any(|a| matches!(a, SupervisionAction::Escalated { .. })));
    }

    #[test]
    fn test_tree_depth() {
        let mut sup = ActorSupervisor::new(16);
        let config = ActorConfig::named("node");

        let root = sup.create_actor(&config, None).unwrap();
        sup.activate_actor(root).unwrap();
        assert_eq!(sup.depth(root), 0);

        let child = sup.create_actor(&config, Some(root)).unwrap();
        sup.activate_actor(child).unwrap();
        assert_eq!(sup.depth(child), 1);

        let grandchild = sup.create_actor(&config, Some(child)).unwrap();
        sup.activate_actor(grandchild).unwrap();
        assert_eq!(sup.depth(grandchild), 2);
    }

    #[test]
    fn test_tree_view() {
        let mut sup = ActorSupervisor::new(8);
        let config = ActorConfig::named("actor");

        let root = sup.create_actor(&config, None).unwrap();
        sup.activate_actor(root).unwrap();

        let child1 = sup.create_actor(&config, Some(root)).unwrap();
        sup.activate_actor(child1).unwrap();

        let child2 = sup.create_actor(&config, Some(root)).unwrap();
        sup.activate_actor(child2).unwrap();

        let view = sup.tree_view();
        assert!(view.contains("ACTIVE"));
        assert!(view.contains("actor:"));
    }
}
