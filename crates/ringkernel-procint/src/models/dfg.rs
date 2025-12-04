//! Directly-Follows Graph (DFG) structures for GPU processing.

use super::ActivityId;
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

/// GPU-compatible DFG node (64 bytes, cache-line aligned).
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct GpuDFGNode {
    /// Activity identifier.
    pub activity_id: u32,
    /// Number of events for this activity.
    pub event_count: u32,
    /// Total duration across all events (ms).
    pub total_duration_ms: u64,
    /// Minimum duration (ms).
    pub min_duration_ms: u32,
    /// Maximum duration (ms).
    pub max_duration_ms: u32,
    /// Average duration (ms).
    pub avg_duration_ms: f32,
    /// Standard deviation of duration.
    pub std_duration_ms: f32,
    /// First occurrence timestamp.
    pub first_seen_ms: u64,
    /// Last occurrence timestamp.
    pub last_seen_ms: u64,
    /// Is this a start activity?
    pub is_start: u8,
    /// Is this an end activity?
    pub is_end: u8,
    /// Node flags.
    pub flags: u8,
    /// Padding.
    pub _padding: u8,
    /// In-degree (number of incoming edges).
    pub incoming_count: u16,
    /// Out-degree (number of outgoing edges).
    pub outgoing_count: u16,
}

// Verify size
const _: () = assert!(std::mem::size_of::<GpuDFGNode>() == 64);

/// GPU-compatible DFG edge (64 bytes, cache-line aligned).
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct GpuDFGEdge {
    /// Source activity ID.
    pub source_activity: u32,
    /// Target activity ID.
    pub target_activity: u32,
    /// Transition frequency.
    pub frequency: u32,
    /// Number of unique cases with this transition.
    pub unique_cases: u32,
    /// Total transition duration (ms).
    pub total_duration_ms: u64,
    /// Minimum transition duration (ms).
    pub min_duration_ms: u32,
    /// Maximum transition duration (ms).
    pub max_duration_ms: u32,
    /// Average transition duration (ms).
    pub avg_duration_ms: f32,
    /// Standard deviation of duration.
    pub std_duration_ms: f32,
    /// Transition probability (frequency / source outgoing).
    pub probability: f32,
    /// Edge flags (is_loop, is_self_loop, is_back_edge).
    pub flags: u32,
    /// Reserved.
    pub _reserved: [u8; 16],
}

// Verify size
const _: () = assert!(std::mem::size_of::<GpuDFGEdge>() == 64);

impl GpuDFGEdge {
    /// Check if this is a self-loop (same source and target).
    pub fn is_self_loop(&self) -> bool {
        self.source_activity == self.target_activity
    }

    /// Check if this is marked as a loop.
    pub fn is_loop(&self) -> bool {
        (self.flags & 0x01) != 0
    }

    /// Check if this is a back edge.
    pub fn is_back_edge(&self) -> bool {
        (self.flags & 0x02) != 0
    }
}

/// GPU-compatible DFG graph header (128 bytes).
#[derive(Debug, Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C, align(128))]
pub struct GpuDFGGraph {
    /// Number of nodes (activities).
    pub node_count: u32,
    /// Number of edges (transitions).
    pub edge_count: u32,
    /// Total number of events processed.
    pub total_events: u64,
    /// Total number of cases (traces).
    pub total_cases: u32,
    /// Number of unique activities.
    pub unique_activities: u32,
    /// Average trace length.
    pub avg_trace_length: f32,
    /// Maximum trace length.
    pub max_trace_length: u32,
    /// Number of start activities.
    pub start_activity_count: u32,
    /// Number of end activities.
    pub end_activity_count: u32,
    /// Graph density (edges / (nodes * (nodes - 1))).
    pub graph_density: f32,
    /// Number of loops detected.
    pub loop_count: u32,
    /// Earliest event timestamp.
    pub timestamp_min: u64,
    /// Latest event timestamp.
    pub timestamp_max: u64,
    /// Average activity duration.
    pub avg_activity_duration: f32,
    /// Reserved.
    pub _reserved: [u8; 56],
}

// Verify size
const _: () = assert!(std::mem::size_of::<GpuDFGGraph>() == 128);

impl Default for GpuDFGGraph {
    fn default() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            total_events: 0,
            total_cases: 0,
            unique_activities: 0,
            avg_trace_length: 0.0,
            max_trace_length: 0,
            start_activity_count: 0,
            end_activity_count: 0,
            graph_density: 0.0,
            loop_count: 0,
            timestamp_min: 0,
            timestamp_max: 0,
            avg_activity_duration: 0.0,
            _reserved: [0; 56],
        }
    }
}

/// High-level DFG representation with all components.
#[derive(Debug, Clone, Default)]
pub struct DFGGraph {
    /// Graph header/metadata.
    pub header: GpuDFGGraph,
    /// Activity nodes.
    pub nodes: Vec<GpuDFGNode>,
    /// Transition edges.
    pub edges: Vec<GpuDFGEdge>,
    /// Activity ID to node index mapping.
    pub node_index: HashMap<ActivityId, usize>,
    /// Edge key (source, target) to edge index mapping.
    pub edge_index: HashMap<(ActivityId, ActivityId), usize>,
}

impl DFGGraph {
    /// Create a new empty DFG.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &[GpuDFGNode] {
        &self.nodes
    }

    /// Get all edges.
    pub fn edges(&self) -> &[GpuDFGEdge] {
        &self.edges
    }

    /// Get node count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Create from GPU buffers.
    pub fn from_gpu(nodes: Vec<GpuDFGNode>, edges: Vec<GpuDFGEdge>) -> Self {
        let mut dfg = Self {
            header: GpuDFGGraph {
                node_count: nodes.len() as u32,
                edge_count: edges.len() as u32,
                ..Default::default()
            },
            nodes,
            edges,
            node_index: HashMap::new(),
            edge_index: HashMap::new(),
        };

        // Build indices
        for (i, node) in dfg.nodes.iter().enumerate() {
            dfg.node_index.insert(node.activity_id, i);
        }
        for (i, edge) in dfg.edges.iter().enumerate() {
            dfg.edge_index
                .insert((edge.source_activity, edge.target_activity), i);
        }

        dfg
    }

    /// Update a node by activity ID.
    pub fn update_node(
        &mut self,
        activity_id: ActivityId,
        event_count: u32,
        avg_duration: f32,
        _cost: f32,
    ) {
        self.add_or_update_node(activity_id, |node| {
            node.event_count = event_count;
            node.avg_duration_ms = avg_duration;
            // Note: cost is stored in reserved or separate tracking
        });
    }

    /// Update an edge.
    pub fn update_edge(
        &mut self,
        source: ActivityId,
        target: ActivityId,
        frequency: u32,
        avg_duration: f32,
    ) {
        self.add_or_update_edge(source, target, |edge| {
            edge.frequency = frequency;
            edge.avg_duration_ms = avg_duration;
        });
    }

    /// Get node by activity ID.
    pub fn get_node(&self, activity_id: ActivityId) -> Option<&GpuDFGNode> {
        self.node_index.get(&activity_id).map(|&i| &self.nodes[i])
    }

    /// Get mutable node by activity ID.
    pub fn get_node_mut(&mut self, activity_id: ActivityId) -> Option<&mut GpuDFGNode> {
        self.node_index
            .get(&activity_id)
            .map(|&i| &mut self.nodes[i])
    }

    /// Get edge by source and target.
    pub fn get_edge(&self, source: ActivityId, target: ActivityId) -> Option<&GpuDFGEdge> {
        self.edge_index
            .get(&(source, target))
            .map(|&i| &self.edges[i])
    }

    /// Add or update a node.
    pub fn add_or_update_node(
        &mut self,
        activity_id: ActivityId,
        update_fn: impl FnOnce(&mut GpuDFGNode),
    ) {
        if let Some(&idx) = self.node_index.get(&activity_id) {
            update_fn(&mut self.nodes[idx]);
        } else {
            let idx = self.nodes.len();
            let mut node = GpuDFGNode {
                activity_id,
                ..Default::default()
            };
            update_fn(&mut node);
            self.nodes.push(node);
            self.node_index.insert(activity_id, idx);
            self.header.node_count = self.nodes.len() as u32;
        }
    }

    /// Add or update an edge.
    pub fn add_or_update_edge(
        &mut self,
        source: ActivityId,
        target: ActivityId,
        update_fn: impl FnOnce(&mut GpuDFGEdge),
    ) {
        let key = (source, target);
        if let Some(&idx) = self.edge_index.get(&key) {
            update_fn(&mut self.edges[idx]);
        } else {
            let idx = self.edges.len();
            let mut edge = GpuDFGEdge {
                source_activity: source,
                target_activity: target,
                ..Default::default()
            };
            update_fn(&mut edge);
            self.edges.push(edge);
            self.edge_index.insert(key, idx);
            self.header.edge_count = self.edges.len() as u32;
        }
    }

    /// Get start activities.
    pub fn start_activities(&self) -> impl Iterator<Item = &GpuDFGNode> {
        self.nodes.iter().filter(|n| n.is_start != 0)
    }

    /// Get end activities.
    pub fn end_activities(&self) -> impl Iterator<Item = &GpuDFGNode> {
        self.nodes.iter().filter(|n| n.is_end != 0)
    }

    /// Get outgoing edges for an activity.
    pub fn outgoing_edges(&self, activity_id: ActivityId) -> impl Iterator<Item = &GpuDFGEdge> {
        self.edges
            .iter()
            .filter(move |e| e.source_activity == activity_id)
    }

    /// Get incoming edges for an activity.
    pub fn incoming_edges(&self, activity_id: ActivityId) -> impl Iterator<Item = &GpuDFGEdge> {
        self.edges
            .iter()
            .filter(move |e| e.target_activity == activity_id)
    }

    /// Calculate edge probabilities.
    pub fn calculate_probabilities(&mut self) {
        // First, calculate outgoing totals per node
        let mut outgoing_totals: HashMap<ActivityId, u32> = HashMap::new();
        for edge in &self.edges {
            *outgoing_totals.entry(edge.source_activity).or_insert(0) += edge.frequency;
        }

        // Then update edge probabilities
        for edge in &mut self.edges {
            if let Some(&total) = outgoing_totals.get(&edge.source_activity) {
                edge.probability = if total > 0 {
                    edge.frequency as f32 / total as f32
                } else {
                    0.0
                };
            }
        }
    }

    /// Update node degrees from edges.
    pub fn update_degrees(&mut self) {
        // Reset degrees
        for node in &mut self.nodes {
            node.incoming_count = 0;
            node.outgoing_count = 0;
        }

        // First collect the counts to avoid borrow issues
        let mut incoming: HashMap<ActivityId, u16> = HashMap::new();
        let mut outgoing: HashMap<ActivityId, u16> = HashMap::new();

        for edge in &self.edges {
            *outgoing.entry(edge.source_activity).or_insert(0) += 1;
            *incoming.entry(edge.target_activity).or_insert(0) += 1;
        }

        // Then update nodes
        for node in &mut self.nodes {
            if let Some(&count) = outgoing.get(&node.activity_id) {
                node.outgoing_count = count;
            }
            if let Some(&count) = incoming.get(&node.activity_id) {
                node.incoming_count = count;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfg_node_size() {
        assert_eq!(std::mem::size_of::<GpuDFGNode>(), 64);
    }

    #[test]
    fn test_dfg_edge_size() {
        assert_eq!(std::mem::size_of::<GpuDFGEdge>(), 64);
    }

    #[test]
    fn test_dfg_header_size() {
        assert_eq!(std::mem::size_of::<GpuDFGGraph>(), 128);
    }

    #[test]
    fn test_dfg_graph_operations() {
        let mut dfg = DFGGraph::new();

        dfg.add_or_update_node(1, |n| {
            n.event_count = 10;
            n.is_start = 1;
        });

        dfg.add_or_update_node(2, |n| {
            n.event_count = 8;
            n.is_end = 1;
        });

        dfg.add_or_update_edge(1, 2, |e| {
            e.frequency = 8;
        });

        assert_eq!(dfg.nodes.len(), 2);
        assert_eq!(dfg.edges.len(), 1);

        dfg.calculate_probabilities();
        assert_eq!(dfg.edges[0].probability, 1.0);
    }
}
