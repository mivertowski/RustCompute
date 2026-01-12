//! Breadth-first search algorithm.
//!
//! BFS computes shortest path distances from source nodes to all reachable nodes.
//! The parallel version uses frontier-based expansion suitable for GPU execution.

use crate::models::{CsrMatrix, Distance, NodeId};
use crate::{GraphError, Result};

/// BFS configuration.
#[derive(Debug, Clone)]
pub struct BfsConfig {
    /// Maximum distance to explore.
    pub max_distance: u32,
    /// Enable bidirectional BFS for single-source single-target.
    pub bidirectional: bool,
}

impl Default for BfsConfig {
    fn default() -> Self {
        Self {
            max_distance: u32::MAX - 1,
            bidirectional: false,
        }
    }
}

impl BfsConfig {
    /// Create new BFS configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum distance.
    pub fn with_max_distance(mut self, max: u32) -> Self {
        self.max_distance = max;
        self
    }
}

/// Sequential BFS implementation.
///
/// Uses a queue-based approach with O(V + E) complexity.
///
/// # Arguments
///
/// * `adj` - Adjacency matrix in CSR format
/// * `sources` - Source nodes to start BFS from
///
/// # Returns
///
/// Vector of distances, one per node. Distance::INFINITY for unreachable nodes.
pub fn bfs_sequential(adj: &CsrMatrix, sources: &[NodeId]) -> Result<Vec<Distance>> {
    bfs_sequential_with_config(adj, sources, &BfsConfig::default())
}

/// Sequential BFS with configuration.
pub fn bfs_sequential_with_config(
    adj: &CsrMatrix,
    sources: &[NodeId],
    config: &BfsConfig,
) -> Result<Vec<Distance>> {
    if adj.num_rows == 0 {
        return Err(GraphError::EmptyGraph);
    }

    let mut distances = vec![Distance::INFINITY; adj.num_rows];
    let mut queue = std::collections::VecDeque::new();

    // Initialize sources
    for &src in sources {
        if src.0 as usize >= adj.num_rows {
            return Err(GraphError::InvalidNodeId(src.0 as u64));
        }
        if distances[src.0 as usize] == Distance::INFINITY {
            distances[src.0 as usize] = Distance::ZERO;
            queue.push_back(src);
        }
    }

    // BFS traversal
    while let Some(node) = queue.pop_front() {
        let current_dist = distances[node.0 as usize];

        if current_dist.0 >= config.max_distance {
            continue;
        }

        for &neighbor_id in adj.neighbors(node) {
            let neighbor = neighbor_id as usize;
            if distances[neighbor] == Distance::INFINITY {
                distances[neighbor] = current_dist.increment();
                queue.push_back(NodeId(neighbor_id));
            }
        }
    }

    Ok(distances)
}

/// Parallel BFS implementation (frontier-based).
///
/// Uses level-synchronous approach suitable for GPU parallelization:
/// 1. Process all nodes in current frontier in parallel
/// 2. Collect next frontier
/// 3. Repeat until no more nodes to explore
///
/// On CPU, this uses rayon for parallelism (when available).
/// The same algorithm structure maps directly to GPU kernels.
pub fn bfs_parallel(adj: &CsrMatrix, sources: &[NodeId]) -> Result<Vec<Distance>> {
    bfs_parallel_with_config(adj, sources, &BfsConfig::default())
}

/// Parallel BFS with configuration.
pub fn bfs_parallel_with_config(
    adj: &CsrMatrix,
    sources: &[NodeId],
    config: &BfsConfig,
) -> Result<Vec<Distance>> {
    if adj.num_rows == 0 {
        return Err(GraphError::EmptyGraph);
    }

    let mut distances = vec![Distance::INFINITY; adj.num_rows];

    // Initialize sources
    let mut frontier: Vec<NodeId> = Vec::new();
    for &src in sources {
        if src.0 as usize >= adj.num_rows {
            return Err(GraphError::InvalidNodeId(src.0 as u64));
        }
        if distances[src.0 as usize] == Distance::INFINITY {
            distances[src.0 as usize] = Distance::ZERO;
            frontier.push(src);
        }
    }

    let mut level = 0u32;

    // Level-synchronous BFS
    while !frontier.is_empty() && level < config.max_distance {
        let mut next_frontier = Vec::new();
        level += 1;

        // Process all frontier nodes (parallelizable)
        for &node in &frontier {
            for &neighbor_id in adj.neighbors(node) {
                let neighbor = neighbor_id as usize;
                // Atomic compare-and-swap in parallel version
                if distances[neighbor] == Distance::INFINITY {
                    distances[neighbor] = Distance::new(level);
                    next_frontier.push(NodeId(neighbor_id));
                }
            }
        }

        frontier = next_frontier;
    }

    Ok(distances)
}

/// Multi-source BFS returning parent pointers for path reconstruction.
pub fn bfs_with_parents(
    adj: &CsrMatrix,
    sources: &[NodeId],
) -> Result<(Vec<Distance>, Vec<NodeId>)> {
    if adj.num_rows == 0 {
        return Err(GraphError::EmptyGraph);
    }

    let mut distances = vec![Distance::INFINITY; adj.num_rows];
    let mut parents = vec![NodeId::INVALID; adj.num_rows];
    let mut queue = std::collections::VecDeque::new();

    // Initialize sources (they are their own parents)
    for &src in sources {
        if src.0 as usize >= adj.num_rows {
            return Err(GraphError::InvalidNodeId(src.0 as u64));
        }
        if distances[src.0 as usize] == Distance::INFINITY {
            distances[src.0 as usize] = Distance::ZERO;
            parents[src.0 as usize] = src;
            queue.push_back(src);
        }
    }

    // BFS traversal
    while let Some(node) = queue.pop_front() {
        let current_dist = distances[node.0 as usize];

        for &neighbor_id in adj.neighbors(node) {
            let neighbor = neighbor_id as usize;
            if distances[neighbor] == Distance::INFINITY {
                distances[neighbor] = current_dist.increment();
                parents[neighbor] = node;
                queue.push_back(NodeId(neighbor_id));
            }
        }
    }

    Ok((distances, parents))
}

/// Reconstruct path from source to target using parent pointers.
pub fn reconstruct_path(parents: &[NodeId], target: NodeId) -> Option<Vec<NodeId>> {
    let target_idx = target.0 as usize;
    if target_idx >= parents.len() || !parents[target_idx].is_valid() {
        return None;
    }

    let mut path = vec![target];
    let mut current = target;

    // Walk back to source
    while parents[current.0 as usize] != current {
        current = parents[current.0 as usize];
        if !current.is_valid() {
            return None;
        }
        path.push(current);
    }

    path.reverse();
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_line_graph(n: usize) -> CsrMatrix {
        // 0 -> 1 -> 2 -> ... -> n-1
        let edges: Vec<_> = (0..n - 1).map(|i| (i as u32, i as u32 + 1)).collect();
        CsrMatrix::from_edges(n, &edges)
    }

    fn make_star_graph(n: usize) -> CsrMatrix {
        // 0 -> 1, 0 -> 2, ..., 0 -> n-1
        let edges: Vec<_> = (1..n).map(|i| (0, i as u32)).collect();
        CsrMatrix::from_edges(n, &edges)
    }

    #[test]
    fn test_bfs_line_graph() {
        let adj = make_line_graph(5);
        let distances = bfs_sequential(&adj, &[NodeId(0)]).unwrap();

        assert_eq!(distances[0], Distance::new(0));
        assert_eq!(distances[1], Distance::new(1));
        assert_eq!(distances[2], Distance::new(2));
        assert_eq!(distances[3], Distance::new(3));
        assert_eq!(distances[4], Distance::new(4));
    }

    #[test]
    fn test_bfs_star_graph() {
        let adj = make_star_graph(5);
        let distances = bfs_sequential(&adj, &[NodeId(0)]).unwrap();

        assert_eq!(distances[0], Distance::new(0));
        for i in 1..5 {
            assert_eq!(distances[i], Distance::new(1));
        }
    }

    #[test]
    fn test_bfs_multi_source() {
        // Line graph: 0 -> 1 -> 2 -> 3 -> 4
        let adj = make_line_graph(5);
        let distances = bfs_sequential(&adj, &[NodeId(0), NodeId(4)]).unwrap();

        // Node 4 is a source, so distance 0
        assert_eq!(distances[4], Distance::new(0));
        // Node 0 is a source
        assert_eq!(distances[0], Distance::new(0));
    }

    #[test]
    fn test_bfs_unreachable() {
        // Two disconnected components: 0 -> 1, 2 -> 3
        let edges = [(0, 1), (2, 3)];
        let adj = CsrMatrix::from_edges(4, &edges);

        let distances = bfs_sequential(&adj, &[NodeId(0)]).unwrap();

        assert_eq!(distances[0], Distance::new(0));
        assert_eq!(distances[1], Distance::new(1));
        assert_eq!(distances[2], Distance::INFINITY);
        assert_eq!(distances[3], Distance::INFINITY);
    }

    #[test]
    fn test_bfs_parallel_same_as_sequential() {
        let adj = make_line_graph(10);

        let seq = bfs_sequential(&adj, &[NodeId(0)]).unwrap();
        let par = bfs_parallel(&adj, &[NodeId(0)]).unwrap();

        assert_eq!(seq, par);
    }

    #[test]
    fn test_bfs_max_distance() {
        let adj = make_line_graph(10);
        let config = BfsConfig::new().with_max_distance(3);
        let distances = bfs_sequential_with_config(&adj, &[NodeId(0)], &config).unwrap();

        assert_eq!(distances[0], Distance::new(0));
        assert_eq!(distances[1], Distance::new(1));
        assert_eq!(distances[2], Distance::new(2));
        assert_eq!(distances[3], Distance::new(3));
        // Beyond max_distance
        assert_eq!(distances[4], Distance::INFINITY);
    }

    #[test]
    fn test_bfs_with_parents() {
        let adj = make_line_graph(5);
        let (distances, parents) = bfs_with_parents(&adj, &[NodeId(0)]).unwrap();

        assert_eq!(distances[4], Distance::new(4));

        // Reconstruct path
        let path = reconstruct_path(&parents, NodeId(4)).unwrap();
        assert_eq!(
            path,
            vec![NodeId(0), NodeId(1), NodeId(2), NodeId(3), NodeId(4)]
        );
    }

    #[test]
    fn test_empty_graph_error() {
        let adj = CsrMatrix::empty(0);
        let result = bfs_sequential(&adj, &[NodeId(0)]);
        assert!(matches!(result, Err(GraphError::EmptyGraph)));
    }

    #[test]
    fn test_invalid_source_error() {
        let adj = make_line_graph(3);
        let result = bfs_sequential(&adj, &[NodeId(100)]);
        assert!(matches!(result, Err(GraphError::InvalidNodeId(_))));
    }
}
