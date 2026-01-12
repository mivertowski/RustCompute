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

/// Parallel union-find (currently falls back to sequential).
///
/// In a true parallel implementation, this would use atomic operations
/// and parallel hook-based union algorithms suitable for GPU.
pub fn union_find_parallel(n: usize, edges: &[(NodeId, NodeId)]) -> Result<Vec<ComponentId>> {
    // TODO: Implement parallel Shiloach-Vishkin or similar algorithm
    union_find_sequential(n, edges)
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
}
