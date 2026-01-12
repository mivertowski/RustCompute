//! Compressed Sparse Row (CSR) matrix format.
//!
//! CSR is an efficient format for sparse matrices/graphs that enables:
//! - O(1) access to row start/end positions
//! - O(degree) iteration over neighbors
//! - Cache-friendly sequential access patterns
//!
//! Memory layout:
//! - `row_ptr[i]` = starting index in col_idx for row i
//! - `col_idx[row_ptr[i]..row_ptr[i+1]]` = column indices (neighbors) of row i
//! - `values` (optional) = edge weights

use super::node::NodeId;
use crate::{GraphError, Result};

/// Compressed Sparse Row matrix for graph adjacency.
///
/// For a graph with N nodes and M edges:
/// - `row_ptr`: N+1 elements, where row_ptr[i] is the start of row i's edges
/// - `col_idx`: M elements, the column indices (neighbor node IDs)
/// - `values`: Optional M elements for weighted graphs
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// Number of rows (nodes).
    pub num_rows: usize,
    /// Number of columns (typically equals num_rows for square adjacency).
    pub num_cols: usize,
    /// Row pointers (length = num_rows + 1).
    pub row_ptr: Vec<u64>,
    /// Column indices (length = nnz).
    pub col_idx: Vec<u32>,
    /// Optional edge values/weights.
    pub values: Option<Vec<f64>>,
}

impl CsrMatrix {
    /// Create an empty CSR matrix.
    pub fn empty(num_nodes: usize) -> Self {
        Self {
            num_rows: num_nodes,
            num_cols: num_nodes,
            row_ptr: vec![0; num_nodes + 1],
            col_idx: Vec::new(),
            values: None,
        }
    }

    /// Create CSR from edge list.
    ///
    /// # Arguments
    ///
    /// * `num_nodes` - Number of nodes in the graph
    /// * `edges` - List of (source, destination) pairs
    ///
    /// # Example
    ///
    /// ```
    /// use ringkernel_graph::CsrMatrix;
    ///
    /// // Graph: 0 -> 1 -> 2
    /// let csr = CsrMatrix::from_edges(3, &[(0, 1), (1, 2)]);
    /// assert_eq!(csr.num_nonzeros(), 2);
    /// ```
    pub fn from_edges(num_nodes: usize, edges: &[(u32, u32)]) -> Self {
        CsrMatrixBuilder::new(num_nodes).with_edges(edges).build()
    }

    /// Create CSR from edge list with weights.
    pub fn from_weighted_edges(num_nodes: usize, edges: &[(u32, u32, f64)]) -> Self {
        CsrMatrixBuilder::new(num_nodes)
            .with_weighted_edges(edges)
            .build()
    }

    /// Number of non-zero entries (edges).
    pub fn num_nonzeros(&self) -> usize {
        self.col_idx.len()
    }

    /// Check if matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.col_idx.is_empty()
    }

    /// Get the degree (number of outgoing edges) of a node.
    pub fn degree(&self, node: NodeId) -> usize {
        let i = node.0 as usize;
        if i >= self.num_rows {
            return 0;
        }
        (self.row_ptr[i + 1] - self.row_ptr[i]) as usize
    }

    /// Get neighbors of a node.
    pub fn neighbors(&self, node: NodeId) -> &[u32] {
        let i = node.0 as usize;
        if i >= self.num_rows {
            return &[];
        }
        let start = self.row_ptr[i] as usize;
        let end = self.row_ptr[i + 1] as usize;
        &self.col_idx[start..end]
    }

    /// Get neighbors with weights (returns empty if unweighted).
    pub fn weighted_neighbors(&self, node: NodeId) -> Vec<(NodeId, f64)> {
        let i = node.0 as usize;
        if i >= self.num_rows {
            return Vec::new();
        }
        let start = self.row_ptr[i] as usize;
        let end = self.row_ptr[i + 1] as usize;

        let neighbors = &self.col_idx[start..end];
        match &self.values {
            Some(vals) => neighbors
                .iter()
                .zip(&vals[start..end])
                .map(|(&col, &w)| (NodeId(col), w))
                .collect(),
            None => neighbors.iter().map(|&col| (NodeId(col), 1.0)).collect(),
        }
    }

    /// Check if edge exists from src to dst.
    pub fn has_edge(&self, src: NodeId, dst: NodeId) -> bool {
        self.neighbors(src).contains(&dst.0)
    }

    /// Validate CSR structure.
    pub fn validate(&self) -> Result<()> {
        // Check row_ptr length
        if self.row_ptr.len() != self.num_rows + 1 {
            return Err(GraphError::InvalidCsr(format!(
                "row_ptr length {} != num_rows + 1 = {}",
                self.row_ptr.len(),
                self.num_rows + 1
            )));
        }

        // Check row_ptr is non-decreasing
        for i in 0..self.num_rows {
            if self.row_ptr[i] > self.row_ptr[i + 1] {
                return Err(GraphError::InvalidCsr(format!(
                    "row_ptr not monotonic at index {}",
                    i
                )));
            }
        }

        // Check final row_ptr matches col_idx length
        let nnz = *self.row_ptr.last().unwrap_or(&0) as usize;
        if nnz != self.col_idx.len() {
            return Err(GraphError::InvalidCsr(format!(
                "row_ptr[-1] = {} != col_idx.len() = {}",
                nnz,
                self.col_idx.len()
            )));
        }

        // Check values length if present
        if let Some(ref vals) = self.values {
            if vals.len() != self.col_idx.len() {
                return Err(GraphError::InvalidCsr(format!(
                    "values.len() = {} != col_idx.len() = {}",
                    vals.len(),
                    self.col_idx.len()
                )));
            }
        }

        // Check col_idx values are in bounds
        for &col in &self.col_idx {
            if col as usize >= self.num_cols {
                return Err(GraphError::InvalidCsr(format!(
                    "col_idx {} >= num_cols {}",
                    col, self.num_cols
                )));
            }
        }

        Ok(())
    }

    /// Create transpose (reverse graph).
    pub fn transpose(&self) -> Self {
        let mut builder = CsrMatrixBuilder::new(self.num_cols);

        // Count in-degrees for new row_ptr
        let mut counts = vec![0u64; self.num_cols];
        for &col in &self.col_idx {
            counts[col as usize] += 1;
        }

        // Build transposed edges
        for row in 0..self.num_rows {
            let start = self.row_ptr[row] as usize;
            let end = self.row_ptr[row + 1] as usize;
            for (i, &col) in self.col_idx[start..end].iter().enumerate() {
                let weight = self.values.as_ref().map(|v| v[start + i]);
                builder.edges.push((col, row as u32, weight));
            }
        }

        builder.build()
    }
}

/// Builder for CSR matrices.
#[derive(Debug, Default)]
pub struct CsrMatrixBuilder {
    num_nodes: usize,
    edges: Vec<(u32, u32, Option<f64>)>,
}

impl CsrMatrixBuilder {
    /// Create new builder with given number of nodes.
    pub fn new(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            edges: Vec::new(),
        }
    }

    /// Add edges from slice.
    pub fn with_edges(mut self, edges: &[(u32, u32)]) -> Self {
        for &(src, dst) in edges {
            self.edges.push((src, dst, None));
        }
        self
    }

    /// Add weighted edges from slice.
    pub fn with_weighted_edges(mut self, edges: &[(u32, u32, f64)]) -> Self {
        for &(src, dst, w) in edges {
            self.edges.push((src, dst, Some(w)));
        }
        self
    }

    /// Add a single edge.
    pub fn add_edge(&mut self, src: u32, dst: u32) {
        self.edges.push((src, dst, None));
    }

    /// Add a weighted edge.
    pub fn add_weighted_edge(&mut self, src: u32, dst: u32, weight: f64) {
        self.edges.push((src, dst, Some(weight)));
    }

    /// Build the CSR matrix.
    pub fn build(mut self) -> CsrMatrix {
        // Sort edges by source
        self.edges.sort_by_key(|e| e.0);

        let has_weights = self.edges.iter().any(|e| e.2.is_some());

        // Build row_ptr
        let mut row_ptr = vec![0u64; self.num_nodes + 1];
        for &(src, _, _) in &self.edges {
            if (src as usize) < self.num_nodes {
                row_ptr[src as usize + 1] += 1;
            }
        }

        // Cumulative sum
        for i in 1..=self.num_nodes {
            row_ptr[i] += row_ptr[i - 1];
        }

        // Build col_idx and values
        let col_idx: Vec<u32> = self.edges.iter().map(|e| e.1).collect();
        let values = if has_weights {
            Some(self.edges.iter().map(|e| e.2.unwrap_or(1.0)).collect())
        } else {
            None
        };

        CsrMatrix {
            num_rows: self.num_nodes,
            num_cols: self.num_nodes,
            row_ptr,
            col_idx,
            values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_matrix() {
        let csr = CsrMatrix::empty(5);
        assert_eq!(csr.num_rows, 5);
        assert_eq!(csr.num_nonzeros(), 0);
        assert!(csr.is_empty());
    }

    #[test]
    fn test_from_edges() {
        // 0 -> 1 -> 2
        //      |
        //      v
        //      3
        let edges = [(0, 1), (1, 2), (1, 3)];
        let csr = CsrMatrix::from_edges(4, &edges);

        assert_eq!(csr.num_rows, 4);
        assert_eq!(csr.num_nonzeros(), 3);
        assert!(csr.validate().is_ok());
    }

    #[test]
    fn test_neighbors() {
        let edges = [(0, 1), (0, 2), (1, 2)];
        let csr = CsrMatrix::from_edges(3, &edges);

        let n0 = csr.neighbors(NodeId(0));
        assert_eq!(n0.len(), 2);
        assert!(n0.contains(&1));
        assert!(n0.contains(&2));

        let n1 = csr.neighbors(NodeId(1));
        assert_eq!(n1.len(), 1);
        assert!(n1.contains(&2));

        let n2 = csr.neighbors(NodeId(2));
        assert!(n2.is_empty());
    }

    #[test]
    fn test_degree() {
        let edges = [(0, 1), (0, 2), (0, 3), (1, 2)];
        let csr = CsrMatrix::from_edges(4, &edges);

        assert_eq!(csr.degree(NodeId(0)), 3);
        assert_eq!(csr.degree(NodeId(1)), 1);
        assert_eq!(csr.degree(NodeId(2)), 0);
        assert_eq!(csr.degree(NodeId(3)), 0);
    }

    #[test]
    fn test_has_edge() {
        let edges = [(0, 1), (1, 2)];
        let csr = CsrMatrix::from_edges(3, &edges);

        assert!(csr.has_edge(NodeId(0), NodeId(1)));
        assert!(csr.has_edge(NodeId(1), NodeId(2)));
        assert!(!csr.has_edge(NodeId(0), NodeId(2)));
        assert!(!csr.has_edge(NodeId(2), NodeId(0)));
    }

    #[test]
    fn test_weighted_edges() {
        let edges = [(0, 1, 1.5), (0, 2, 2.5), (1, 2, 3.0)];
        let csr = CsrMatrix::from_weighted_edges(3, &edges);

        assert!(csr.values.is_some());

        let neighbors = csr.weighted_neighbors(NodeId(0));
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&(NodeId(1), 1.5)));
        assert!(neighbors.contains(&(NodeId(2), 2.5)));
    }

    #[test]
    fn test_transpose() {
        // 0 -> 1 -> 2
        let edges = [(0, 1), (1, 2)];
        let csr = CsrMatrix::from_edges(3, &edges);
        let transposed = csr.transpose();

        // Should now be: 1 -> 0, 2 -> 1
        assert!(transposed.has_edge(NodeId(1), NodeId(0)));
        assert!(transposed.has_edge(NodeId(2), NodeId(1)));
        assert!(!transposed.has_edge(NodeId(0), NodeId(1)));
    }

    #[test]
    fn test_builder() {
        let mut builder = CsrMatrixBuilder::new(4);
        builder.add_edge(0, 1);
        builder.add_edge(0, 2);
        builder.add_weighted_edge(1, 3, 2.5);

        let csr = builder.build();
        assert_eq!(csr.num_nonzeros(), 3);
        assert!(csr.values.is_some());
    }

    #[test]
    fn test_validation() {
        // Valid matrix
        let csr = CsrMatrix::from_edges(3, &[(0, 1), (1, 2)]);
        assert!(csr.validate().is_ok());

        // Invalid: col_idx out of bounds
        let invalid = CsrMatrix {
            num_rows: 3,
            num_cols: 3,
            row_ptr: vec![0, 1, 2, 2],
            col_idx: vec![1, 10], // 10 is out of bounds
            values: None,
        };
        assert!(invalid.validate().is_err());
    }
}
