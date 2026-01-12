//! Graph data models.
//!
//! This module provides the core data structures for representing graphs:
//! - [`CsrMatrix`]: Compressed Sparse Row format for adjacency matrices
//! - [`NodeId`], [`Distance`], [`ComponentId`]: Graph node types

pub mod csr;
pub mod node;

pub use csr::{CsrMatrix, CsrMatrixBuilder};
pub use node::{ComponentId, Distance, NodeId};
