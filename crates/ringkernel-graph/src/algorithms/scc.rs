//! Strongly Connected Components (SCC) algorithms.
//!
//! A strongly connected component is a maximal subgraph where every node
//! can reach every other node. This module provides:
//!
//! - Tarjan's algorithm (sequential, O(V+E))
//! - Kosaraju's algorithm (can be parallelized)

use crate::models::{ComponentId, CsrMatrix, NodeId};
use crate::Result;

/// SCC configuration.
#[derive(Debug, Clone, Default)]
pub struct SccConfig {
    /// Return components in topological order.
    pub topological_order: bool,
}

impl SccConfig {
    /// Create new SCC configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return components in topological order.
    pub fn with_topological_order(mut self) -> Self {
        self.topological_order = true;
        self
    }
}

/// Tarjan's SCC algorithm (sequential).
///
/// Uses DFS with lowlink values to find SCCs in O(V+E) time.
/// Returns component ID for each node.
pub fn scc_tarjan(adj: &CsrMatrix) -> Result<Vec<ComponentId>> {
    scc_tarjan_with_config(adj, &SccConfig::default())
}

/// Tarjan's algorithm with configuration.
pub fn scc_tarjan_with_config(adj: &CsrMatrix, _config: &SccConfig) -> Result<Vec<ComponentId>> {
    if adj.num_rows == 0 {
        return Ok(Vec::new());
    }

    let n = adj.num_rows;
    let mut component = vec![ComponentId::UNASSIGNED; n];
    let mut index = vec![u32::MAX; n]; // Discovery index
    let mut lowlink = vec![u32::MAX; n];
    let mut on_stack = vec![false; n];
    let mut stack = Vec::new();

    let mut current_index = 0u32;
    let mut current_component = 0u32;

    // Need to handle non-recursive implementation to avoid stack overflow
    for start in 0..n {
        if index[start] != u32::MAX {
            continue;
        }

        // Iterative DFS with explicit stack
        let mut dfs_stack: Vec<(usize, usize)> = vec![(start, 0)];

        while let Some(&(v, neighbor_idx)) = dfs_stack.last() {
            if neighbor_idx == 0 {
                // First visit to this node
                index[v] = current_index;
                lowlink[v] = current_index;
                current_index += 1;
                stack.push(v);
                on_stack[v] = true;
            }

            let neighbors = adj.neighbors(NodeId(v as u32));

            if neighbor_idx < neighbors.len() {
                let w = neighbors[neighbor_idx] as usize;
                dfs_stack.last_mut().unwrap().1 += 1;

                if index[w] == u32::MAX {
                    // Not visited, push to DFS stack
                    dfs_stack.push((w, 0));
                } else if on_stack[w] {
                    // Back edge to node in current SCC
                    lowlink[v] = lowlink[v].min(index[w]);
                }
            } else {
                // All neighbors processed
                dfs_stack.pop();

                // Update parent's lowlink
                if let Some(&(parent, _)) = dfs_stack.last() {
                    lowlink[parent] = lowlink[parent].min(lowlink[v]);
                }

                // Check if v is root of an SCC
                if lowlink[v] == index[v] {
                    // Pop all nodes in this SCC
                    loop {
                        let w = stack.pop().unwrap();
                        on_stack[w] = false;
                        component[w] = ComponentId::new(current_component);
                        if w == v {
                            break;
                        }
                    }
                    current_component += 1;
                }
            }
        }
    }

    Ok(component)
}

/// Kosaraju's SCC algorithm.
///
/// Two-pass algorithm:
/// 1. DFS on original graph, record finish order
/// 2. DFS on transposed graph in reverse finish order
///
/// More parallelizable than Tarjan's.
pub fn scc_kosaraju(adj: &CsrMatrix) -> Result<Vec<ComponentId>> {
    scc_kosaraju_with_config(adj, &SccConfig::default())
}

/// Kosaraju's algorithm with configuration.
pub fn scc_kosaraju_with_config(adj: &CsrMatrix, _config: &SccConfig) -> Result<Vec<ComponentId>> {
    if adj.num_rows == 0 {
        return Ok(Vec::new());
    }

    let n = adj.num_rows;

    // Pass 1: DFS on original graph, record finish order
    let mut visited = vec![false; n];
    let mut finish_order = Vec::with_capacity(n);

    for start in 0..n {
        if !visited[start] {
            dfs_finish_order(adj, start, &mut visited, &mut finish_order);
        }
    }

    // Create transposed graph
    let transposed = adj.transpose();

    // Pass 2: DFS on transposed graph in reverse finish order
    let mut component = vec![ComponentId::UNASSIGNED; n];
    let mut current_component = 0u32;
    visited.fill(false);

    for &node in finish_order.iter().rev() {
        if !visited[node] {
            dfs_assign_component(
                &transposed,
                node,
                &mut visited,
                &mut component,
                ComponentId::new(current_component),
            );
            current_component += 1;
        }
    }

    Ok(component)
}

/// DFS to record finish order.
fn dfs_finish_order(
    adj: &CsrMatrix,
    start: usize,
    visited: &mut [bool],
    finish_order: &mut Vec<usize>,
) {
    let mut stack = vec![(start, false)]; // (node, processed)

    while let Some((node, processed)) = stack.pop() {
        if processed {
            finish_order.push(node);
            continue;
        }

        if visited[node] {
            continue;
        }
        visited[node] = true;

        // Push this node again to record finish time after children
        stack.push((node, true));

        for &neighbor in adj.neighbors(NodeId(node as u32)) {
            let w = neighbor as usize;
            if !visited[w] {
                stack.push((w, false));
            }
        }
    }
}

/// DFS to assign component IDs.
fn dfs_assign_component(
    adj: &CsrMatrix,
    start: usize,
    visited: &mut [bool],
    component: &mut [ComponentId],
    comp_id: ComponentId,
) {
    let mut stack = vec![start];

    while let Some(node) = stack.pop() {
        if visited[node] {
            continue;
        }
        visited[node] = true;
        component[node] = comp_id;

        for &neighbor in adj.neighbors(NodeId(node as u32)) {
            let w = neighbor as usize;
            if !visited[w] {
                stack.push(w);
            }
        }
    }
}

/// Get number of SCCs from component assignment.
pub fn count_components(components: &[ComponentId]) -> usize {
    if components.is_empty() {
        return 0;
    }
    components
        .iter()
        .filter(|c| c.is_assigned())
        .map(|c| c.0)
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0)
}

/// Get nodes in each component.
pub fn get_component_members(components: &[ComponentId]) -> Vec<Vec<NodeId>> {
    let num_components = count_components(components);
    let mut members = vec![Vec::new(); num_components];

    for (node_id, &comp_id) in components.iter().enumerate() {
        if comp_id.is_assigned() {
            members[comp_id.0 as usize].push(NodeId(node_id as u32));
        }
    }

    members
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_node() {
        let adj = CsrMatrix::empty(1);
        let components = scc_tarjan(&adj).unwrap();
        assert_eq!(components.len(), 1);
        assert!(components[0].is_assigned());
    }

    #[test]
    fn test_line_graph_sccs() {
        // Line graph: 0 -> 1 -> 2 -> 3
        // Each node is its own SCC (no cycles)
        let edges = [(0, 1), (1, 2), (2, 3)];
        let adj = CsrMatrix::from_edges(4, &edges);

        let components = scc_tarjan(&adj).unwrap();
        assert_eq!(count_components(&components), 4);
    }

    #[test]
    fn test_cycle_scc() {
        // Cycle: 0 -> 1 -> 2 -> 0
        // All nodes in same SCC
        let edges = [(0, 1), (1, 2), (2, 0)];
        let adj = CsrMatrix::from_edges(3, &edges);

        let components = scc_tarjan(&adj).unwrap();
        assert_eq!(count_components(&components), 1);
        assert_eq!(components[0], components[1]);
        assert_eq!(components[1], components[2]);
    }

    #[test]
    fn test_two_sccs() {
        // Two cycles: 0 <-> 1, 2 <-> 3, with 1 -> 2
        let edges = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)];
        let adj = CsrMatrix::from_edges(4, &edges);

        let components = scc_tarjan(&adj).unwrap();
        assert_eq!(count_components(&components), 2);

        // 0 and 1 should be in same component
        assert_eq!(components[0], components[1]);
        // 2 and 3 should be in same component
        assert_eq!(components[2], components[3]);
        // Different components
        assert_ne!(components[0], components[2]);
    }

    #[test]
    fn test_kosaraju_same_as_tarjan() {
        // More complex graph
        let edges = [
            (0, 1),
            (1, 2),
            (2, 0), // Cycle 0-1-2
            (2, 3),
            (3, 4),
            (4, 3), // Cycle 3-4
            (5, 6),
            (6, 5), // Cycle 5-6
        ];
        let adj = CsrMatrix::from_edges(7, &edges);

        let tarjan = scc_tarjan(&adj).unwrap();
        let kosaraju = scc_kosaraju(&adj).unwrap();

        // Same number of components
        assert_eq!(count_components(&tarjan), count_components(&kosaraju));

        // Same groupings (component IDs may differ)
        let tarjan_members = get_component_members(&tarjan);
        let kosaraju_members = get_component_members(&kosaraju);

        assert_eq!(tarjan_members.len(), kosaraju_members.len());

        // Sort members within each component for comparison
        let mut tarjan_sorted: Vec<_> = tarjan_members
            .iter()
            .map(|v| {
                let mut v = v.clone();
                v.sort();
                v
            })
            .collect();
        tarjan_sorted.sort();

        let mut kosaraju_sorted: Vec<_> = kosaraju_members
            .iter()
            .map(|v| {
                let mut v = v.clone();
                v.sort();
                v
            })
            .collect();
        kosaraju_sorted.sort();

        assert_eq!(tarjan_sorted, kosaraju_sorted);
    }

    #[test]
    fn test_disconnected_graph() {
        // Two disconnected nodes
        let adj = CsrMatrix::empty(2);

        let components = scc_tarjan(&adj).unwrap();
        assert_eq!(count_components(&components), 2);
        assert_ne!(components[0], components[1]);
    }

    #[test]
    fn test_get_component_members() {
        let edges = [(0, 1), (1, 0), (2, 3), (3, 2)];
        let adj = CsrMatrix::from_edges(4, &edges);

        let components = scc_tarjan(&adj).unwrap();
        let members = get_component_members(&components);

        assert_eq!(members.len(), 2);

        // Check that each component has 2 members
        for comp in &members {
            assert_eq!(comp.len(), 2);
        }
    }

    #[test]
    fn test_empty_graph() {
        let adj = CsrMatrix::empty(0);
        let components = scc_tarjan(&adj).unwrap();
        assert!(components.is_empty());
    }
}
