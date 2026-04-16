//! Graceful Shutdown with Drain Mode — FR-010
//!
//! Coordinated shutdown with message preservation:
//! - Drain mode: stop accepting new messages, process all in-flight
//! - Grace period: configurable timeout, then hard kill
//! - Dependency ordering: leaf actors first, parents last
//! - Checkpoint on drain: automatically snapshot state before termination

use std::time::{Duration, Instant};

use crate::actor::{ActorId, ActorState, ActorSupervisor};

/// Shutdown phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownPhase {
    /// Normal operation.
    Running,
    /// Draining: no new messages accepted, processing in-flight.
    Draining,
    /// Grace period expired, hard-killing remaining actors.
    ForceKilling,
    /// Shutdown complete.
    Terminated,
}

/// Configuration for graceful shutdown.
#[derive(Debug, Clone)]
pub struct ShutdownConfig {
    /// Maximum time to wait for actors to drain.
    pub drain_timeout: Duration,
    /// Whether to checkpoint actor state before termination.
    pub checkpoint_on_drain: bool,
    /// Whether to process remaining queue messages before terminating.
    pub process_in_flight: bool,
    /// Shutdown ordering strategy.
    pub ordering: ShutdownOrdering,
}

impl Default for ShutdownConfig {
    fn default() -> Self {
        Self {
            drain_timeout: Duration::from_secs(30),
            checkpoint_on_drain: true,
            process_in_flight: true,
            ordering: ShutdownOrdering::LeafFirst,
        }
    }
}

/// Strategy for ordering actor shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownOrdering {
    /// Shut down leaf actors first, then parents (bottom-up).
    /// Ensures children finish before parents.
    LeafFirst,
    /// Shut down parents first (top-down, cascading).
    ParentFirst,
    /// Shut down all actors simultaneously.
    Parallel,
}

/// Manages the graceful shutdown process.
pub struct ShutdownCoordinator {
    /// Current phase.
    phase: ShutdownPhase,
    /// Configuration.
    config: ShutdownConfig,
    /// When shutdown was initiated.
    started_at: Option<Instant>,
    /// Actors that have been told to drain.
    draining_actors: Vec<ActorId>,
    /// Actors that have confirmed drain complete.
    drained_actors: Vec<ActorId>,
    /// Actors that were force-killed.
    force_killed: Vec<ActorId>,
}

impl ShutdownCoordinator {
    /// Create a new shutdown coordinator.
    pub fn new(config: ShutdownConfig) -> Self {
        Self {
            phase: ShutdownPhase::Running,
            config,
            started_at: None,
            draining_actors: Vec::new(),
            drained_actors: Vec::new(),
            force_killed: Vec::new(),
        }
    }

    /// Initiate graceful shutdown.
    ///
    /// Returns the ordered list of actors to drain.
    pub fn initiate(&mut self, supervisor: &ActorSupervisor) -> Vec<ActorId> {
        self.phase = ShutdownPhase::Draining;
        self.started_at = Some(Instant::now());

        // Determine shutdown order
        let actors = self.compute_shutdown_order(supervisor);
        self.draining_actors = actors.clone();

        tracing::info!(
            phase = "draining",
            actors = actors.len(),
            timeout_secs = self.config.drain_timeout.as_secs(),
            "Initiating graceful shutdown"
        );

        actors
    }

    /// Mark an actor as drained (finished processing in-flight messages).
    pub fn mark_drained(&mut self, actor: ActorId) {
        self.drained_actors.push(actor);
    }

    /// Check if the drain timeout has expired.
    pub fn is_timeout_expired(&self) -> bool {
        self.started_at
            .map(|start| start.elapsed() >= self.config.drain_timeout)
            .unwrap_or(false)
    }

    /// Advance the shutdown state machine.
    ///
    /// Returns the current phase and any actors that need force-killing.
    pub fn tick(&mut self) -> (ShutdownPhase, Vec<ActorId>) {
        match self.phase {
            ShutdownPhase::Running => (self.phase, Vec::new()),

            ShutdownPhase::Draining => {
                // Check if all actors have drained
                let all_drained = self.draining_actors.iter().all(|a| {
                    self.drained_actors.contains(a)
                });

                if all_drained {
                    self.phase = ShutdownPhase::Terminated;
                    tracing::info!("All actors drained, shutdown complete");
                    (self.phase, Vec::new())
                } else if self.is_timeout_expired() {
                    // Force kill remaining
                    self.phase = ShutdownPhase::ForceKilling;
                    let remaining: Vec<ActorId> = self.draining_actors
                        .iter()
                        .filter(|a| !self.drained_actors.contains(a))
                        .copied()
                        .collect();

                    tracing::warn!(
                        remaining = remaining.len(),
                        "Drain timeout expired, force-killing remaining actors"
                    );

                    self.force_killed = remaining.clone();
                    self.phase = ShutdownPhase::Terminated;
                    (ShutdownPhase::ForceKilling, remaining)
                } else {
                    (self.phase, Vec::new())
                }
            }

            ShutdownPhase::ForceKilling => {
                self.phase = ShutdownPhase::Terminated;
                (self.phase, Vec::new())
            }

            ShutdownPhase::Terminated => (self.phase, Vec::new()),
        }
    }

    /// Current shutdown phase.
    pub fn phase(&self) -> ShutdownPhase {
        self.phase
    }

    /// Elapsed time since shutdown was initiated.
    pub fn elapsed(&self) -> Option<Duration> {
        self.started_at.map(|s| s.elapsed())
    }

    /// Get shutdown report.
    pub fn report(&self) -> ShutdownReport {
        ShutdownReport {
            phase: self.phase,
            total_actors: self.draining_actors.len(),
            drained: self.drained_actors.len(),
            force_killed: self.force_killed.len(),
            elapsed: self.elapsed(),
            checkpoint_enabled: self.config.checkpoint_on_drain,
        }
    }

    /// Compute the shutdown order based on the configured strategy.
    fn compute_shutdown_order(&self, supervisor: &ActorSupervisor) -> Vec<ActorId> {
        let mut order: Vec<ActorId> = supervisor
            .entries()
            .iter()
            .filter(|e| e.actor_state().is_alive())
            .map(|e| ActorId(e.actor_id))
            .collect();

        match self.config.ordering {
            ShutdownOrdering::LeafFirst => {
                // Sort by depth (deepest first = leaves first)
                order.sort_by(|a, b| {
                    let da = supervisor.depth(*a);
                    let db = supervisor.depth(*b);
                    db.cmp(&da) // Descending depth = leaves first
                });
            }
            ShutdownOrdering::ParentFirst => {
                // Sort by depth (shallowest first = parents first)
                order.sort_by(|a, b| {
                    let da = supervisor.depth(*a);
                    let db = supervisor.depth(*b);
                    da.cmp(&db)
                });
            }
            ShutdownOrdering::Parallel => {
                // No sorting needed
            }
        }

        order
    }
}

/// Report of shutdown results.
#[derive(Debug, Clone)]
pub struct ShutdownReport {
    /// Final phase.
    pub phase: ShutdownPhase,
    /// Total actors that were draining.
    pub total_actors: usize,
    /// Successfully drained.
    pub drained: usize,
    /// Force-killed after timeout.
    pub force_killed: usize,
    /// Time taken.
    pub elapsed: Option<Duration>,
    /// Whether checkpointing was enabled.
    pub checkpoint_enabled: bool,
}

impl std::fmt::Display for ShutdownReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Shutdown: {} actors, {} drained, {} force-killed, {:?} elapsed",
            self.total_actors,
            self.drained,
            self.force_killed,
            self.elapsed.unwrap_or_default()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::ActorConfig;

    #[test]
    fn test_graceful_shutdown_all_drain() {
        let mut supervisor = ActorSupervisor::new(8);
        let config = ActorConfig::named("worker");

        let a1 = supervisor.create_actor(&config, None).unwrap();
        supervisor.activate_actor(a1).unwrap();
        let a2 = supervisor.create_actor(&config, None).unwrap();
        supervisor.activate_actor(a2).unwrap();

        let mut coord = ShutdownCoordinator::new(ShutdownConfig::default());
        let actors = coord.initiate(&supervisor);
        assert_eq!(actors.len(), 2);
        assert_eq!(coord.phase(), ShutdownPhase::Draining);

        // Simulate actors draining
        coord.mark_drained(a1);
        coord.mark_drained(a2);

        let (phase, force) = coord.tick();
        assert_eq!(phase, ShutdownPhase::Terminated);
        assert!(force.is_empty());

        let report = coord.report();
        assert_eq!(report.drained, 2);
        assert_eq!(report.force_killed, 0);
    }

    #[test]
    fn test_shutdown_timeout_force_kill() {
        let mut supervisor = ActorSupervisor::new(8);
        let config = ActorConfig::named("worker");

        let a1 = supervisor.create_actor(&config, None).unwrap();
        supervisor.activate_actor(a1).unwrap();

        let mut coord = ShutdownCoordinator::new(ShutdownConfig {
            drain_timeout: Duration::from_millis(1), // Very short
            ..Default::default()
        });

        coord.initiate(&supervisor);
        // Don't mark_drained — simulate timeout

        std::thread::sleep(Duration::from_millis(5));

        let (phase, force_killed) = coord.tick();
        assert_eq!(phase, ShutdownPhase::ForceKilling);
        assert_eq!(force_killed.len(), 1);
        assert_eq!(force_killed[0], a1);
    }

    #[test]
    fn test_leaf_first_ordering() {
        let mut supervisor = ActorSupervisor::new(8);
        let config = ActorConfig::named("node");

        let root = supervisor.create_actor(&config, None).unwrap();
        supervisor.activate_actor(root).unwrap();
        let child = supervisor.create_actor(&config, Some(root)).unwrap();
        supervisor.activate_actor(child).unwrap();
        let grandchild = supervisor.create_actor(&config, Some(child)).unwrap();
        supervisor.activate_actor(grandchild).unwrap();

        let coord = ShutdownCoordinator::new(ShutdownConfig {
            ordering: ShutdownOrdering::LeafFirst,
            ..Default::default()
        });

        let order = coord.compute_shutdown_order(&supervisor);
        // Grandchild should be first (deepest)
        assert_eq!(order[0], grandchild);
        // Root should be last (shallowest)
        assert_eq!(*order.last().unwrap(), root);
    }

    #[test]
    fn test_shutdown_report_display() {
        let report = ShutdownReport {
            phase: ShutdownPhase::Terminated,
            total_actors: 5,
            drained: 4,
            force_killed: 1,
            elapsed: Some(Duration::from_secs(2)),
            checkpoint_enabled: true,
        };
        let s = format!("{}", report);
        assert!(s.contains("5 actors"));
        assert!(s.contains("4 drained"));
        assert!(s.contains("1 force-killed"));
    }
}
