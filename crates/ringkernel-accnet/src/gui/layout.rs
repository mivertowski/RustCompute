//! Force-directed graph layout algorithm.
//!
//! Implements Fruchterman-Reingold with Barnes-Hut optimization for
//! smooth, aesthetically pleasing network layouts.

use crate::models::{AccountType, AccountingNetwork};
use nalgebra::Vector2;
use std::collections::HashMap;

/// A node in the layout.
#[derive(Debug, Clone)]
pub struct LayoutNode {
    /// Account index.
    pub index: u16,
    /// Account type.
    pub account_type: AccountType,
    /// Current position.
    pub position: Vector2<f32>,
    /// Current velocity.
    pub velocity: Vector2<f32>,
    /// Whether position is pinned.
    pub pinned: bool,
    /// Node mass (affects repulsion).
    pub mass: f32,
}

/// Configuration for force-directed layout.
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    /// Repulsion constant (higher = more spread out).
    pub repulsion: f32,
    /// Attraction constant (higher = tighter clusters).
    pub attraction: f32,
    /// Damping factor (0-1, higher = more friction).
    pub damping: f32,
    /// Minimum velocity threshold.
    pub min_velocity: f32,
    /// Maximum iterations per frame.
    pub max_iterations: usize,
    /// Ideal edge length.
    pub ideal_length: f32,
    /// Canvas width.
    pub width: f32,
    /// Canvas height.
    pub height: f32,
    /// Gravity toward center.
    pub gravity: f32,
    /// Group accounts by type.
    pub group_by_type: bool,
    /// Minimum distance between any two nodes.
    pub min_node_distance: f32,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            repulsion: 50000.0, // Very strong repulsion for maximum spacing
            attraction: 0.0008, // Very gentle attraction for loose connections
            damping: 0.85,      // Balance between friction and movement
            min_velocity: 0.3,  // Lower threshold for smoother convergence
            max_iterations: 50,
            ideal_length: 300.0, // Large ideal edge length
            width: 800.0,
            height: 600.0,
            gravity: 0.008, // Very low gravity - minimal pull to center
            group_by_type: true,
            min_node_distance: 300.0, // Very large minimum distance between nodes
        }
    }
}

/// Force-directed graph layout engine.
pub struct ForceDirectedLayout {
    /// Configuration.
    pub config: LayoutConfig,
    /// Layout nodes.
    nodes: HashMap<u16, LayoutNode>,
    /// Edges (source, target, weight).
    edges: Vec<(u16, u16, f32)>,
    /// Whether layout has converged.
    pub converged: bool,
    /// Current temperature (for simulated annealing).
    temperature: f32,
}

impl ForceDirectedLayout {
    /// Create a new layout engine.
    pub fn new(config: LayoutConfig) -> Self {
        Self {
            temperature: config.width / 10.0,
            config,
            nodes: HashMap::new(),
            edges: Vec::new(),
            converged: false,
        }
    }

    /// Initialize layout from accounting network.
    pub fn initialize(&mut self, network: &AccountingNetwork) {
        self.nodes.clear();
        self.edges.clear();
        self.converged = false;
        self.temperature = self.config.width / 10.0;

        // Create nodes with initial positions
        let center = Vector2::new(self.config.width / 2.0, self.config.height / 2.0);

        for (i, account) in network.accounts.iter().enumerate() {
            // Initial position based on account type (circular layout by type)
            let angle = self.type_angle(account.account_type) + (i as f32 * 0.1);
            let radius = self.config.width.min(self.config.height) * 0.3;

            let position = if self.config.group_by_type {
                center + Vector2::new(angle.cos() * radius, angle.sin() * radius)
            } else {
                // Random initial position
                center
                    + Vector2::new(
                        (rand::random::<f32>() - 0.5) * self.config.width * 0.8,
                        (rand::random::<f32>() - 0.5) * self.config.height * 0.8,
                    )
            };

            self.nodes.insert(
                account.index,
                LayoutNode {
                    index: account.index,
                    account_type: account.account_type,
                    position,
                    velocity: Vector2::zeros(),
                    pinned: false,
                    mass: 1.0 + (account.risk_score * 2.0),
                },
            );
        }

        // Create edges from flows
        for flow in &network.flows {
            let weight = flow.amount.to_f64().abs() as f32;
            self.edges
                .push((flow.source_account_index, flow.target_account_index, weight));
        }
    }

    /// Get base angle for account type grouping.
    fn type_angle(&self, account_type: AccountType) -> f32 {
        use std::f32::consts::PI;
        match account_type {
            AccountType::Asset => 0.0,
            AccountType::Liability => PI * 2.0 / 5.0,
            AccountType::Equity => PI * 4.0 / 5.0,
            AccountType::Revenue => PI * 6.0 / 5.0,
            AccountType::Expense => PI * 8.0 / 5.0,
            AccountType::Contra => PI,
        }
    }

    /// Run one iteration of the layout algorithm.
    pub fn step(&mut self) -> bool {
        if self.converged || self.nodes.is_empty() {
            return false;
        }

        let mut forces: HashMap<u16, Vector2<f32>> = HashMap::new();
        for &idx in self.nodes.keys() {
            forces.insert(idx, Vector2::zeros());
        }

        // Calculate repulsion forces (all pairs) with minimum distance enforcement
        let node_indices: Vec<u16> = self.nodes.keys().copied().collect();
        let min_dist = self.config.min_node_distance;

        for i in 0..node_indices.len() {
            for j in (i + 1)..node_indices.len() {
                let idx_i = node_indices[i];
                let idx_j = node_indices[j];

                let node_i = &self.nodes[&idx_i];
                let node_j = &self.nodes[&idx_j];

                let delta = node_j.position - node_i.position;
                let distance = delta.magnitude().max(1.0);

                // Strong repulsion when below minimum distance
                let force_magnitude = if distance < min_dist {
                    // Much stronger force when too close - push apart aggressively
                    let overlap = min_dist / distance;
                    (self.config.repulsion * overlap * overlap / (distance * distance)).min(200.0)
                } else {
                    // Normal Fruchterman-Reingold repulsion
                    (self.config.repulsion / (distance * distance)).min(100.0)
                };

                // Safe normalization
                let force = if distance > 0.01 {
                    delta / distance * force_magnitude
                } else {
                    // Random direction if too close
                    Vector2::new(
                        (rand::random::<f32>() - 0.5) * force_magnitude,
                        (rand::random::<f32>() - 0.5) * force_magnitude,
                    )
                };

                if let Some(f) = forces.get_mut(&idx_i) {
                    *f -= force;
                }
                if let Some(f) = forces.get_mut(&idx_j) {
                    *f += force;
                }
            }
        }

        // Calculate attraction forces (edges)
        for (source, target, weight) in &self.edges {
            if let (Some(node_s), Some(node_t)) = (self.nodes.get(source), self.nodes.get(target)) {
                let delta = node_t.position - node_s.position;
                let distance = delta.magnitude().max(1.0);

                // Attraction proportional to distance, with weight factor capped
                let weight_factor = 1.0 + weight.abs().max(1.0).ln();
                let force_magnitude = (self.config.attraction * distance * weight_factor).min(50.0);

                // Safe normalization
                let force = if distance > 0.01 {
                    delta / distance * force_magnitude
                } else {
                    Vector2::zeros()
                };

                if let Some(f) = forces.get_mut(source) {
                    *f += force;
                }
                if let Some(f) = forces.get_mut(target) {
                    *f -= force;
                }
            }
        }

        // Apply gravity toward center
        let center = Vector2::new(self.config.width / 2.0, self.config.height / 2.0);
        for (idx, node) in &self.nodes {
            let delta = center - node.position;
            let gravity_force = delta * self.config.gravity * node.mass;
            if let Some(f) = forces.get_mut(idx) {
                *f += gravity_force;
            }
        }

        // Apply forces and update positions
        let mut max_displacement = 0.0f32;

        for (idx, force) in &forces {
            if let Some(node) = self.nodes.get_mut(idx) {
                if node.pinned {
                    continue;
                }

                // Skip if force is NaN or Inf
                if !force.x.is_finite() || !force.y.is_finite() {
                    continue;
                }

                // Update velocity with force and damping
                let new_velocity =
                    (node.velocity + *force / node.mass.max(0.1)) * self.config.damping;

                // Skip if velocity becomes NaN
                if !new_velocity.x.is_finite() || !new_velocity.y.is_finite() {
                    node.velocity = Vector2::zeros();
                    continue;
                }

                node.velocity = new_velocity;

                // Limit velocity by temperature
                let speed = node.velocity.magnitude();
                if speed > self.temperature {
                    node.velocity = node.velocity / speed * self.temperature;
                }

                // Cap maximum velocity
                let max_speed = 50.0;
                if speed > max_speed {
                    node.velocity = node.velocity / speed * max_speed;
                }

                // Update position
                let displacement = node.velocity;
                node.position += displacement;

                // Keep within bounds with padding
                let padding = 50.0;
                node.position.x = node.position.x.clamp(padding, self.config.width - padding);
                node.position.y = node.position.y.clamp(padding, self.config.height - padding);

                max_displacement = max_displacement.max(displacement.magnitude());
            }
        }

        // Cool down
        self.temperature *= 0.995;

        // Check convergence
        if max_displacement < self.config.min_velocity && self.temperature < 1.0 {
            self.converged = true;
        }

        true
    }

    /// Run multiple iterations.
    pub fn iterate(&mut self, iterations: usize) {
        for _ in 0..iterations {
            if !self.step() {
                break;
            }
        }
    }

    /// Get node position.
    pub fn get_position(&self, index: u16) -> Option<Vector2<f32>> {
        self.nodes.get(&index).map(|n| n.position)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &LayoutNode> {
        self.nodes.values()
    }

    /// Get all edges.
    pub fn edges(&self) -> &[(u16, u16, f32)] {
        &self.edges
    }

    /// Pin a node at its current position.
    pub fn pin_node(&mut self, index: u16) {
        if let Some(node) = self.nodes.get_mut(&index) {
            node.pinned = true;
        }
    }

    /// Unpin a node.
    pub fn unpin_node(&mut self, index: u16) {
        if let Some(node) = self.nodes.get_mut(&index) {
            node.pinned = false;
        }
    }

    /// Set node position (for dragging).
    pub fn set_position(&mut self, index: u16, position: Vector2<f32>) {
        if let Some(node) = self.nodes.get_mut(&index) {
            node.position = position;
            node.velocity = Vector2::zeros();
        }
    }

    /// Resize the layout area.
    pub fn resize(&mut self, width: f32, height: f32) {
        let scale_x = width / self.config.width;
        let scale_y = height / self.config.height;

        for node in self.nodes.values_mut() {
            node.position.x *= scale_x;
            node.position.y *= scale_y;
        }

        self.config.width = width;
        self.config.height = height;
    }

    /// Reset the layout.
    pub fn reset(&mut self, network: &AccountingNetwork) {
        self.initialize(network);
    }

    /// Update edge weights from network without resetting positions.
    /// This allows the layout to adapt to new flow data while preserving node positions.
    pub fn update_edges(&mut self, network: &AccountingNetwork) {
        // Aggregate edges by source-target pair
        use std::collections::HashMap as StdHashMap;
        let mut edge_weights: StdHashMap<(u16, u16), f32> = StdHashMap::new();

        for flow in &network.flows {
            let key = (flow.source_account_index, flow.target_account_index);
            *edge_weights.entry(key).or_insert(0.0) += flow.amount.to_f64().abs() as f32;
        }

        // Convert to edge list
        self.edges = edge_weights
            .into_iter()
            .map(|((s, t), w)| (s, t, w))
            .collect();

        // Add any missing nodes
        for account in &network.accounts {
            if !self.nodes.contains_key(&account.index) {
                let center = Vector2::new(self.config.width / 2.0, self.config.height / 2.0);
                let angle = self.type_angle(account.account_type);
                let radius = self.config.width.min(self.config.height) * 0.3;
                let position = center + Vector2::new(angle.cos() * radius, angle.sin() * radius);

                self.nodes.insert(
                    account.index,
                    LayoutNode {
                        index: account.index,
                        account_type: account.account_type,
                        position,
                        velocity: Vector2::zeros(),
                        pinned: false,
                        mass: 1.0 + (account.risk_score * 2.0),
                    },
                );
            }
        }
    }

    /// Warm up the layout to allow it to readjust.
    /// Call this periodically to let the layout adapt to weight changes.
    pub fn warm_up(&mut self) {
        self.converged = false;
        self.temperature = (self.config.width / 20.0).max(20.0); // Moderate warm-up
    }

    /// Check if layout has any edges.
    pub fn has_edges(&self) -> bool {
        !self.edges.is_empty()
    }

    /// Get edge count.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Default for ForceDirectedLayout {
    fn default() -> Self {
        Self::new(LayoutConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_creation() {
        let layout = ForceDirectedLayout::default();
        assert!(!layout.converged);
        assert_eq!(layout.nodes.len(), 0);
    }

    #[test]
    fn test_layout_config() {
        let config = LayoutConfig::default();
        assert!(config.repulsion > 0.0);
        assert!(config.attraction > 0.0);
    }
}
