//! GPU kernels for accounting network analysis.
//!
//! This module contains Rust DSL implementations that can be transpiled
//! to CUDA or WGSL for GPU execution.
//!
//! ## Kernel Categories
//!
//! 1. **Journal Transformation** - Methods A-E for converting entries to flows
//! 2. **Network Analysis** - Suspense detection, GAAP violations, fraud patterns
//! 3. **Temporal Analysis** - Seasonality, trends, behavioral anomalies
//!
//! ## Method Distribution (Ivertowski et al., 2024)
//!
//! | Method | Description | Frequency |
//! |--------|-------------|-----------|
//! | A | 1-to-1 debit→credit | 60.68% |
//! | B | n-to-n bijective | 16.63% |
//! | C | n-to-m partition | 11.00% |
//! | D | Higher aggregate | 11.00% |
//! | E | Decomposition | 0.69% |

pub mod transformation;
pub mod analysis;
pub mod temporal;

pub use transformation::*;
pub use analysis::*;
pub use temporal::*;
