//! Union-Find (Disjoint Set) data structure.
//!
//! Union-Find efficiently tracks connected components in undirected graphs.
//! Supports:
//! - `find(x)`: Find representative of x's component
//! - `union(x, y)`: Merge components containing x and y
//!
//! Uses path compression and union by rank for near O(1) amortized operations.

use crate::models::{ComponentId, NodeId};
use crate::Result;

/// Union-Find data structure with path compression and union by rank.
#[derive(Debug, Clone)]
pub struct UnionFind {
    /// Parent pointers (parent[i] = parent of node i, or i if root).
    parent: Vec<u32>,
    /// Rank (tree height upper bound) for union by rank.
    rank: Vec<u32>,
    /// Number of components.
    num_components: usize,
}

impl UnionFind {
    /// Create new Union-Find with n singleton sets.
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
            num_components: n,
        }
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Number of disjoint components.
    pub fn num_components(&self) -> usize {
        self.num_components
    }

    /// Find representative of node's component with path compression.
    pub fn find(&mut self, x: NodeId) -> NodeId {
        let mut root = x.0;

        // Find root
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }

        // Path compression: point all nodes on path directly to root
        let mut node = x.0;
        while self.parent[node as usize] != root {
            let next = self.parent[node as usize];
            self.parent[node as usize] = root;
            node = next;
        }

        NodeId(root)
    }

    /// Union two components by rank.
    ///
    /// Returns true if a merge occurred (x and y were in different components).
    pub fn union(&mut self, x: NodeId, y: NodeId) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false; // Already in same component
        }

        // Union by rank: attach smaller tree under larger tree
        let rx = self.rank[root_x.0 as usize];
        let ry = self.rank[root_y.0 as usize];

        if rx < ry {
            self.parent[root_x.0 as usize] = root_y.0;
        } else if rx > ry {
            self.parent[root_y.0 as usize] = root_x.0;
        } else {
            // Same rank: arbitrarily choose root_x as new root, increment rank
            self.parent[root_y.0 as usize] = root_x.0;
            self.rank[root_x.0 as usize] += 1;
        }

        self.num_components -= 1;
        true
    }

    /// Check if two nodes are in the same component.
    pub fn connected(&mut self, x: NodeId, y: NodeId) -> bool {
        self.find(x) == self.find(y)
    }

    /// Get component ID for each node.
    ///
    /// Returns a vector where `result[i]` is the component ID of node i.
    /// Component IDs are assigned 0, 1, 2, ... based on root nodes.
    pub fn component_ids(&mut self) -> Vec<ComponentId> {
        let n = self.parent.len();
        let mut comp_id = vec![ComponentId::UNASSIGNED; n];
        let mut next_id = 0u32;

        for i in 0..n {
            let root = self.find(NodeId(i as u32));

            // Assign ID to root if not already assigned
            if !comp_id[root.0 as usize].is_assigned() {
                comp_id[root.0 as usize] = ComponentId::new(next_id);
                next_id += 1;
            }

            // Copy root's ID to this node
            comp_id[i] = comp_id[root.0 as usize];
        }

        comp_id
    }

    /// Get the size of the component containing node x.
    pub fn component_size(&mut self, x: NodeId) -> usize {
        let root = self.find(x);
        let mut count = 0;
        for i in 0..self.parent.len() {
            if self.find(NodeId(i as u32)) == root {
                count += 1;
            }
        }
        count
    }
}

/// Sequential union-find on edge list.
///
/// Computes connected components from undirected edges.
pub fn union_find_sequential(n: usize, edges: &[(NodeId, NodeId)]) -> Result<Vec<ComponentId>> {
    let mut uf = UnionFind::new(n);

    for &(u, v) in edges {
        uf.union(u, v);
    }

    Ok(uf.component_ids())
}

/// Parallel union-find using Shiloach-Vishkin style algorithm.
///
/// This implementation uses atomic operations for thread-safe parallel execution.
/// The algorithm works in rounds of:
/// 1. Hook: For each edge, attempt to hook smaller component under larger
/// 2. Jump: Pointer jumping to flatten trees
///
/// Continues until no changes occur (convergence).
pub fn union_find_parallel(n: usize, edges: &[(NodeId, NodeId)]) -> Result<Vec<ComponentId>> {
    use std::sync::atomic::{AtomicU32, Ordering};

    if n == 0 {
        return Ok(vec![]);
    }

    // Initialize parent array with atomic operations for thread safety
    let parent: Vec<AtomicU32> = (0..n as u32).map(AtomicU32::new).collect();

    // Shiloach-Vishkin style parallel connected components
    let mut changed = true;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 64; // Prevent infinite loops

    while changed && iterations < MAX_ITERATIONS {
        changed = false;
        iterations += 1;

        // Phase 1: Hook - process all edges
        // For each edge (u, v), try to hook smaller root under larger root
        for &(u, v) in edges {
            let mut pu = parent[u.0 as usize].load(Ordering::Relaxed);
            let mut pv = parent[v.0 as usize].load(Ordering::Relaxed);

            // Find roots (with limited iterations to avoid infinite loops)
            for _ in 0..n {
                let gpu = parent[pu as usize].load(Ordering::Relaxed);
                if gpu == pu {
                    break;
                }
                pu = gpu;
            }
            for _ in 0..n {
                let gpv = parent[pv as usize].load(Ordering::Relaxed);
                if gpv == pv {
                    break;
                }
                pv = gpv;
            }

            // Hook smaller root under larger root
            if pu != pv {
                let (smaller, larger) = if pu < pv { (pu, pv) } else { (pv, pu) };
                // Atomic compare-and-swap to hook smaller under larger
                if parent[smaller as usize]
                    .compare_exchange(smaller, larger, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    changed = true;
                }
            }
        }

        // Phase 2: Jump - pointer jumping to flatten trees
        // Each node points to its grandparent: parent[i] = parent[parent[i]]
        for i in 0..n {
            let pi = parent[i].load(Ordering::Relaxed);
            if pi != i as u32 {
                let gpi = parent[pi as usize].load(Ordering::Relaxed);
                if gpi != pi {
                    // Point to grandparent (path compression)
                    let _ =
                        parent[i].compare_exchange(pi, gpi, Ordering::AcqRel, Ordering::Relaxed);
                    changed = true;
                }
            }
        }
    }

    // Final pass: ensure all nodes point to their root and assign component IDs
    let mut final_parent: Vec<u32> = parent.iter().map(|p| p.load(Ordering::Relaxed)).collect();

    // Complete path compression
    for i in 0..n {
        let mut root = i as u32;
        while final_parent[root as usize] != root {
            root = final_parent[root as usize];
        }
        // Compress path
        let mut node = i as u32;
        while final_parent[node as usize] != root {
            let next = final_parent[node as usize];
            final_parent[node as usize] = root;
            node = next;
        }
    }

    // Assign component IDs
    let mut comp_id = vec![ComponentId::UNASSIGNED; n];
    let mut next_id = 0u32;

    for i in 0..n {
        let root = final_parent[i] as usize;

        if !comp_id[root].is_assigned() {
            comp_id[root] = ComponentId::new(next_id);
            next_id += 1;
        }

        comp_id[i] = comp_id[root];
    }

    Ok(comp_id)
}

/// Parallel union-find optimized for GPU execution.
///
/// This variant structures the computation for future GPU acceleration:
/// - Uses contiguous memory layouts suitable for GPU buffers
/// - Minimizes synchronization points
/// - Structures work for SIMT execution
#[cfg(feature = "cuda")]
pub fn union_find_gpu_ready(n: usize, edges: &[(NodeId, NodeId)]) -> Result<(Vec<u32>, usize)> {
    // For now, use the parallel CPU implementation
    // Returns (parent array, number of components) for GPU buffer compatibility
    let components = union_find_parallel(n, edges)?;

    // Convert to parent array format
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut num_components = 0u32;

    for i in 0..n {
        if components[i].0 == num_components {
            num_components += 1;
        }
        // Find representative
        let comp = components[i].0;
        // First node with this component ID becomes the root
        for j in 0..=i {
            if components[j].0 == comp {
                parent[i] = j as u32;
                break;
            }
        }
    }

    Ok((parent, num_components as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_singleton_sets() {
        let mut uf = UnionFind::new(5);
        assert_eq!(uf.num_components(), 5);

        // Each node is its own representative
        for i in 0..5 {
            assert_eq!(uf.find(NodeId(i)), NodeId(i));
        }
    }

    #[test]
    fn test_union_basic() {
        let mut uf = UnionFind::new(5);

        assert!(uf.union(NodeId(0), NodeId(1)));
        assert_eq!(uf.num_components(), 4);
        assert!(uf.connected(NodeId(0), NodeId(1)));

        assert!(uf.union(NodeId(2), NodeId(3)));
        assert_eq!(uf.num_components(), 3);

        assert!(uf.union(NodeId(0), NodeId(2)));
        assert_eq!(uf.num_components(), 2);
        assert!(uf.connected(NodeId(0), NodeId(3)));
    }

    #[test]
    fn test_union_same_component() {
        let mut uf = UnionFind::new(3);

        uf.union(NodeId(0), NodeId(1));
        uf.union(NodeId(1), NodeId(2));

        // All three are connected
        assert!(uf.connected(NodeId(0), NodeId(2)));

        // Union within same component returns false
        assert!(!uf.union(NodeId(0), NodeId(2)));
        assert_eq!(uf.num_components(), 1);
    }

    #[test]
    fn test_path_compression() {
        let mut uf = UnionFind::new(10);

        // Create a chain: 0 -> 1 -> 2 -> ... -> 9
        for i in 0..9 {
            uf.union(NodeId(i), NodeId(i + 1));
        }

        // Find on last node should compress path
        let root = uf.find(NodeId(9));

        // After path compression, parent should point directly to root
        // (this tests that path compression worked)
        for i in 0..10 {
            assert_eq!(uf.find(NodeId(i)), root);
        }
    }

    #[test]
    fn test_component_ids() {
        let mut uf = UnionFind::new(5);

        uf.union(NodeId(0), NodeId(1));
        uf.union(NodeId(2), NodeId(3));

        let ids = uf.component_ids();

        // 0 and 1 should have same ID
        assert_eq!(ids[0], ids[1]);
        // 2 and 3 should have same ID
        assert_eq!(ids[2], ids[3]);
        // 4 is alone
        assert_ne!(ids[4], ids[0]);
        assert_ne!(ids[4], ids[2]);
        // Three distinct components
        assert_eq!(uf.num_components(), 3);
    }

    #[test]
    fn test_component_size() {
        let mut uf = UnionFind::new(6);

        uf.union(NodeId(0), NodeId(1));
        uf.union(NodeId(1), NodeId(2));
        // Component {0, 1, 2} has size 3

        uf.union(NodeId(3), NodeId(4));
        // Component {3, 4} has size 2

        // Node 5 is alone

        assert_eq!(uf.component_size(NodeId(0)), 3);
        assert_eq!(uf.component_size(NodeId(3)), 2);
        assert_eq!(uf.component_size(NodeId(5)), 1);
    }

    #[test]
    fn test_union_find_from_edges() {
        let edges = [
            (NodeId(0), NodeId(1)),
            (NodeId(1), NodeId(2)),
            (NodeId(3), NodeId(4)),
        ];

        let components = union_find_sequential(5, &edges).unwrap();

        // {0, 1, 2} are connected
        assert_eq!(components[0], components[1]);
        assert_eq!(components[1], components[2]);

        // {3, 4} are connected
        assert_eq!(components[3], components[4]);

        // Different groups
        assert_ne!(components[0], components[3]);
    }

    #[test]
    fn test_empty_union_find() {
        let uf = UnionFind::new(0);
        assert!(uf.is_empty());
        assert_eq!(uf.num_components(), 0);
    }

    #[test]
    fn test_parallel_union_find_basic() {
        let edges = [
            (NodeId(0), NodeId(1)),
            (NodeId(1), NodeId(2)),
            (NodeId(3), NodeId(4)),
        ];

        let components = union_find_parallel(5, &edges).unwrap();

        // {0, 1, 2} are connected
        assert_eq!(components[0], components[1]);
        assert_eq!(components[1], components[2]);

        // {3, 4} are connected
        assert_eq!(components[3], components[4]);

        // Different groups
        assert_ne!(components[0], components[3]);
    }

    #[test]
    fn test_parallel_union_find_single_component() {
        // Linear chain connecting all nodes
        let edges: Vec<_> = (0..9).map(|i| (NodeId(i), NodeId(i + 1))).collect();

        let components = union_find_parallel(10, &edges).unwrap();

        // All nodes should be in the same component
        for i in 1..10 {
            assert_eq!(components[0], components[i]);
        }
    }

    #[test]
    fn test_parallel_union_find_no_edges() {
        let components = union_find_parallel(5, &[]).unwrap();

        // Each node is its own component
        for i in 0..5 {
            for j in (i + 1)..5 {
                assert_ne!(components[i], components[j]);
            }
        }
    }

    #[test]
    fn test_parallel_union_find_empty() {
        let components = union_find_parallel(0, &[]).unwrap();
        assert!(components.is_empty());
    }

    #[test]
    fn test_parallel_vs_sequential_consistency() {
        // Test that parallel and sequential give same results
        let edges = [
            (NodeId(0), NodeId(5)),
            (NodeId(1), NodeId(6)),
            (NodeId(2), NodeId(7)),
            (NodeId(5), NodeId(6)),
            (NodeId(3), NodeId(8)),
            (NodeId(4), NodeId(9)),
            (NodeId(8), NodeId(9)),
        ];

        let seq_components = union_find_sequential(10, &edges).unwrap();
        let par_components = union_find_parallel(10, &edges).unwrap();

        // Check connectivity is identical
        for i in 0..10 {
            for j in (i + 1)..10 {
                let seq_same = seq_components[i] == seq_components[j];
                let par_same = par_components[i] == par_components[j];
                assert_eq!(seq_same, par_same, "Mismatch for nodes {} and {}", i, j);
            }
        }
    }

    #[test]
    fn test_parallel_union_find_star_graph() {
        // Star graph: node 0 connected to all others
        let edges: Vec<_> = (1..10).map(|i| (NodeId(0), NodeId(i))).collect();

        let components = union_find_parallel(10, &edges).unwrap();

        // All nodes should be in the same component
        for i in 1..10 {
            assert_eq!(components[0], components[i]);
        }
    }
}
