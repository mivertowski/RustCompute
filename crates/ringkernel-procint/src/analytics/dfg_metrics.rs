//! DFG metrics calculation.
//!
//! Computes graph metrics from Directly-Follows Graphs.

use crate::models::{DFGGraph, GpuDFGEdge, GpuDFGNode};

/// DFG metrics calculator.
#[derive(Debug, Default)]
pub struct DFGMetricsCalculator {
    /// Last computed metrics.
    pub metrics: DFGMetrics,
}

impl DFGMetricsCalculator {
    /// Create a new calculator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate metrics from a DFG.
    pub fn calculate(&mut self, dfg: &DFGGraph) -> &DFGMetrics {
        let nodes = dfg.nodes();
        let edges = dfg.edges();

        self.metrics = DFGMetrics {
            node_count: nodes.len(),
            edge_count: edges.len(),
            total_events: nodes.iter().map(|n| n.event_count as u64).sum(),
            avg_events_per_node: self.avg_events_per_node(nodes),
            density: self.graph_density(nodes.len(), edges.len()),
            avg_degree: self.avg_degree(nodes, edges),
            max_in_degree: self.max_in_degree(nodes),
            max_out_degree: self.max_out_degree(nodes),
            avg_duration_ms: self.avg_duration(nodes),
            total_cost: 0.0, // Cost tracking moved to separate analytics
            bottleneck_score: self.bottleneck_score(nodes),
            parallelism_factor: self.parallelism_factor(edges),
        };

        &self.metrics
    }

    fn avg_events_per_node(&self, nodes: &[GpuDFGNode]) -> f32 {
        if nodes.is_empty() {
            return 0.0;
        }
        let total: u64 = nodes.iter().map(|n| n.event_count as u64).sum();
        total as f32 / nodes.len() as f32
    }

    fn graph_density(&self, node_count: usize, edge_count: usize) -> f32 {
        if node_count <= 1 {
            return 0.0;
        }
        let max_edges = node_count * (node_count - 1);
        edge_count as f32 / max_edges as f32
    }

    fn avg_degree(&self, nodes: &[GpuDFGNode], edges: &[GpuDFGEdge]) -> f32 {
        if nodes.is_empty() {
            return 0.0;
        }
        // Total degree = 2 * edges (each edge contributes to in and out degree)
        2.0 * edges.len() as f32 / nodes.len() as f32
    }

    fn max_in_degree(&self, nodes: &[GpuDFGNode]) -> u32 {
        nodes
            .iter()
            .map(|n| n.incoming_count as u32)
            .max()
            .unwrap_or(0)
    }

    fn max_out_degree(&self, nodes: &[GpuDFGNode]) -> u32 {
        nodes
            .iter()
            .map(|n| n.outgoing_count as u32)
            .max()
            .unwrap_or(0)
    }

    fn avg_duration(&self, nodes: &[GpuDFGNode]) -> f32 {
        if nodes.is_empty() {
            return 0.0;
        }
        let total: f32 = nodes.iter().map(|n| n.avg_duration_ms).sum();
        total / nodes.len() as f32
    }

    fn bottleneck_score(&self, nodes: &[GpuDFGNode]) -> f32 {
        // Score based on variance in throughput rates (using degree counts)
        if nodes.is_empty() {
            return 0.0;
        }

        let rates: Vec<f32> = nodes
            .iter()
            .filter(|n| n.event_count > 0 && n.incoming_count > 0)
            .map(|n| n.outgoing_count as f32 / n.incoming_count.max(1) as f32)
            .collect();

        if rates.is_empty() {
            return 0.0;
        }

        let mean: f32 = rates.iter().sum::<f32>() / rates.len() as f32;
        let variance: f32 =
            rates.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rates.len() as f32;

        // Normalize variance to 0-1 score
        (variance / (variance + 1.0)).min(1.0)
    }

    fn parallelism_factor(&self, edges: &[GpuDFGEdge]) -> f32 {
        // Estimate parallelism from split/join patterns
        if edges.is_empty() {
            return 1.0;
        }

        // Count activities with multiple outgoing edges
        let mut out_counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for edge in edges {
            *out_counts.entry(edge.source_activity).or_insert(0) += 1;
        }

        let split_count = out_counts.values().filter(|&&c| c > 1).count();
        1.0 + split_count as f32 * 0.5
    }

    /// Get current metrics.
    pub fn metrics(&self) -> &DFGMetrics {
        &self.metrics
    }
}

/// Computed DFG metrics.
#[derive(Debug, Clone, Default)]
pub struct DFGMetrics {
    /// Number of nodes (activities).
    pub node_count: usize,
    /// Number of edges (transitions).
    pub edge_count: usize,
    /// Total events processed.
    pub total_events: u64,
    /// Average events per node.
    pub avg_events_per_node: f32,
    /// Graph density (edges / max_edges).
    pub density: f32,
    /// Average node degree.
    pub avg_degree: f32,
    /// Maximum in-degree.
    pub max_in_degree: u32,
    /// Maximum out-degree.
    pub max_out_degree: u32,
    /// Average activity duration (ms).
    pub avg_duration_ms: f32,
    /// Total cost across all activities.
    pub total_cost: f64,
    /// Bottleneck score (0-1).
    pub bottleneck_score: f32,
    /// Parallelism factor.
    pub parallelism_factor: f32,
}

impl DFGMetrics {
    /// Get complexity score (combination of density and degree).
    pub fn complexity_score(&self) -> f32 {
        self.density * 0.4 + (self.avg_degree / 10.0).min(1.0) * 0.3 + self.bottleneck_score * 0.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_calculation() {
        let mut dfg = DFGGraph::default();

        // Add some test data
        dfg.update_node(1, 100, 5000.0, 100.0);
        dfg.update_node(2, 80, 3000.0, 80.0);
        dfg.update_edge(1, 2, 50, 2000.0);

        let mut calc = DFGMetricsCalculator::new();
        let metrics = calc.calculate(&dfg);

        assert!(metrics.node_count > 0);
        assert!(metrics.total_events > 0);
    }
}
