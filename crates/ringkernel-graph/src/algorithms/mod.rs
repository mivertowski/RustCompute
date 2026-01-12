//! Graph algorithms.
//!
//! This module provides parallel graph algorithms:
//! - [`bfs`]: Breadth-first search
//! - [`scc`]: Strongly connected components
//! - [`union_find`]: Disjoint set data structure
//! - [`spmv`]: Sparse matrix-vector multiplication

pub mod bfs;
pub mod scc;
pub mod spmv;
pub mod union_find;

pub use bfs::{bfs_parallel, bfs_sequential, BfsConfig};
pub use scc::{scc_kosaraju, scc_tarjan, SccConfig};
pub use spmv::{spmv, spmv_parallel, SpmvConfig};
pub use union_find::{union_find_parallel, union_find_sequential, UnionFind};
