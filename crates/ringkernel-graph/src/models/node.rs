//! Node types for graph algorithms.
//!
//! This module provides strongly-typed wrappers for graph concepts:
//! - [`NodeId`]: Unique identifier for graph vertices
//! - [`Distance`]: BFS distance/level from source
//! - [`ComponentId`]: Connected component identifier

use bytemuck::{Pod, Zeroable};

/// Node identifier (vertex ID).
///
/// Using a newtype prevents mixing up node IDs with other integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Maximum valid node ID.
    pub const MAX: NodeId = NodeId(u32::MAX - 1);

    /// Invalid/sentinel node ID.
    pub const INVALID: NodeId = NodeId(u32::MAX);

    /// Create a new node ID.
    pub const fn new(id: u32) -> Self {
        NodeId(id)
    }

    /// Check if this is a valid node ID.
    pub const fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }

    /// Get the inner value.
    pub const fn get(&self) -> u32 {
        self.0
    }
}

impl From<u32> for NodeId {
    fn from(id: u32) -> Self {
        NodeId(id)
    }
}

impl From<usize> for NodeId {
    fn from(id: usize) -> Self {
        NodeId(id as u32)
    }
}

impl From<NodeId> for usize {
    fn from(id: NodeId) -> Self {
        id.0 as usize
    }
}

// SAFETY: NodeId is #[repr(transparent)] over u32
unsafe impl Zeroable for NodeId {}
unsafe impl Pod for NodeId {}

/// Distance from source in BFS.
///
/// Also known as "level" or "hop count".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Distance(pub u32);

impl Distance {
    /// Infinity (unreachable).
    pub const INFINITY: Distance = Distance(u32::MAX);

    /// Zero distance (source node).
    pub const ZERO: Distance = Distance(0);

    /// Create a new distance.
    pub const fn new(d: u32) -> Self {
        Distance(d)
    }

    /// Check if node is reachable.
    pub const fn is_reachable(&self) -> bool {
        self.0 != u32::MAX
    }

    /// Get the inner value.
    pub const fn get(&self) -> u32 {
        self.0
    }

    /// Increment distance by 1, saturating at infinity.
    pub const fn increment(&self) -> Self {
        if self.0 == u32::MAX {
            Distance::INFINITY
        } else {
            Distance(self.0.saturating_add(1))
        }
    }
}

impl From<u32> for Distance {
    fn from(d: u32) -> Self {
        Distance(d)
    }
}

// SAFETY: Distance is #[repr(transparent)] over u32
unsafe impl Zeroable for Distance {}
unsafe impl Pod for Distance {}

/// Connected component identifier.
///
/// Nodes in the same component have the same ComponentId.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ComponentId(pub u32);

impl ComponentId {
    /// Unassigned component.
    pub const UNASSIGNED: ComponentId = ComponentId(u32::MAX);

    /// Create a new component ID.
    pub const fn new(id: u32) -> Self {
        ComponentId(id)
    }

    /// Check if component is assigned.
    pub const fn is_assigned(&self) -> bool {
        self.0 != u32::MAX
    }

    /// Get the inner value.
    pub const fn get(&self) -> u32 {
        self.0
    }
}

impl From<u32> for ComponentId {
    fn from(id: u32) -> Self {
        ComponentId(id)
    }
}

impl From<NodeId> for ComponentId {
    fn from(id: NodeId) -> Self {
        ComponentId(id.0)
    }
}

// SAFETY: ComponentId is #[repr(transparent)] over u32
unsafe impl Zeroable for ComponentId {}
unsafe impl Pod for ComponentId {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_basics() {
        let node = NodeId::new(42);
        assert_eq!(node.get(), 42);
        assert!(node.is_valid());
        assert!(!NodeId::INVALID.is_valid());
    }

    #[test]
    fn test_node_id_conversions() {
        let node: NodeId = 100u32.into();
        assert_eq!(node.get(), 100);

        let idx: usize = node.into();
        assert_eq!(idx, 100);
    }

    #[test]
    fn test_distance_basics() {
        let d = Distance::new(5);
        assert_eq!(d.get(), 5);
        assert!(d.is_reachable());
        assert!(!Distance::INFINITY.is_reachable());
    }

    #[test]
    fn test_distance_increment() {
        let d = Distance::new(5);
        assert_eq!(d.increment().get(), 6);

        // Saturates at infinity
        assert_eq!(Distance::INFINITY.increment(), Distance::INFINITY);
    }

    #[test]
    fn test_component_id_basics() {
        let c = ComponentId::new(3);
        assert_eq!(c.get(), 3);
        assert!(c.is_assigned());
        assert!(!ComponentId::UNASSIGNED.is_assigned());
    }

    #[test]
    fn test_pod_traits() {
        // Verify types are Pod-compatible
        let nodes = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let bytes = bytemuck::bytes_of(&nodes);
        assert_eq!(bytes.len(), 12); // 3 * 4 bytes

        let distances = [Distance::ZERO, Distance::new(1), Distance::INFINITY];
        let bytes = bytemuck::bytes_of(&distances);
        assert_eq!(bytes.len(), 12);
    }
}
