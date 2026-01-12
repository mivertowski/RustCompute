//! GPU-accelerated graph algorithm primitives for RingKernel.
//!
//! This crate provides high-performance graph algorithms optimized for
//! parallel execution on GPUs. It includes:
//!
//! - **CSR Matrix**: Compressed Sparse Row format for efficient graph storage
//! - **BFS**: Parallel breadth-first search with multiple source support
//! - **SCC**: Strongly connected components via forward-backward algorithm
//! - **Union-Find**: Parallel disjoint set data structure
//! - **SpMV**: Sparse matrix-vector multiplication
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_graph::{CsrMatrix, bfs_parallel, NodeId};
//!
//! // Build adjacency list: 0 -> 1 -> 2
//! let matrix = CsrMatrix::from_edges(3, &[(0, 1), (1, 2)]);
//!
//! // Run BFS from node 0
//! let distances = bfs_parallel(&matrix, &[NodeId(0)]).await?;
//! assert_eq!(distances[0].0, 0);  // Distance to self
//! assert_eq!(distances[1].0, 1);  // Distance to node 1
//! assert_eq!(distances[2].0, 2);  // Distance to node 2
//! ```

pub mod algorithms;
pub mod models;

/// GPU-accelerated implementations (requires `cuda` feature).
#[cfg(feature = "cuda")]
pub mod gpu;

// Re-export main types
pub use algorithms::bfs::{bfs_parallel, bfs_sequential, BfsConfig};
pub use algorithms::scc::{scc_kosaraju, scc_tarjan, SccConfig};
pub use algorithms::spmv::{spmv, spmv_parallel, SpmvConfig};
pub use algorithms::union_find::{union_find_parallel, union_find_sequential, UnionFind};
pub use models::csr::{CsrMatrix, CsrMatrixBuilder};
pub use models::node::{ComponentId, Distance, NodeId};

/// Graph algorithm error types.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    /// Invalid node ID.
    #[error("Invalid node ID: {0}")]
    InvalidNodeId(u64),

    /// Matrix dimension mismatch.
    #[error("Matrix dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Empty graph.
    #[error("Empty graph")]
    EmptyGraph,

    /// Invalid CSR format.
    #[error("Invalid CSR format: {0}")]
    InvalidCsr(String),

    /// Algorithm error.
    #[error("Algorithm error: {0}")]
    AlgorithmError(String),
}

/// Result type for graph operations.
pub type Result<T> = std::result::Result<T, GraphError>;
