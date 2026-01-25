//! # Hybrid CPU-GPU Processing
//!
//! Adaptive processing that routes workloads to CPU or GPU based on size
//! and characteristics. Small/irregular workloads use CPU (via Rayon),
//! while bulk parallel operations use GPU.
//!
//! ## Decision Heuristics
//!
//! | Workload Size | GPU Overhead | Decision |
//! |---------------|--------------|----------|
//! | < threshold   | High         | CPU      |
//! | >= threshold  | Amortized    | GPU      |
//!
//! For irregular access patterns (sparse updates, random traversal),
//! CPU may outperform GPU even at larger scales.
//!
//! ## Example
//!
//! ```ignore
//! use ringkernel_core::hybrid::*;
//!
//! // Define your workload
//! struct MyWorkload { data: Vec<f32> }
//!
//! impl HybridWorkload for MyWorkload {
//!     type Result = Vec<f32>;
//!
//!     fn workload_size(&self) -> usize { self.data.len() }
//!     fn execute_cpu(&self) -> Self::Result { /* ... */ }
//!     fn execute_gpu(&self) -> Result<Self::Result, HybridError> { /* ... */ }
//! }
//!
//! // Create dispatcher and execute
//! let dispatcher = HybridDispatcher::new(HybridConfig::default());
//! let result = dispatcher.execute(&workload);
//! ```

mod config;
mod dispatcher;
mod error;
mod stats;
mod traits;

pub use config::{HybridConfig, HybridConfigBuilder, ProcessingMode};
pub use dispatcher::HybridDispatcher;
pub use error::{HybridError, HybridResult};
pub use stats::{HybridStats, HybridStatsSnapshot};
pub use traits::HybridWorkload;
