//! CUDA GPU backends for transaction monitoring.
//!
//! This module provides multiple GPU-accelerated approaches:
//!
//! 1. **Actor-based Ring Kernel** (`ring_kernel`) - Persistent GPU kernels that
//!    continuously process transaction streams with HLC timestamps and K2K messaging.
//!    Best for: Low-latency streaming with complex inter-kernel communication.
//!
//! 2. **Parallel Batch Kernel** (`batch_kernel`) - High-throughput batch processing
//!    where each thread handles one transaction independently.
//!    Best for: Maximum throughput on large batches.
//!
//! 3. **Stencil Pattern Detection** (`stencil_kernel`) - Uses GridPos for detecting
//!    spatial patterns in transaction networks (e.g., circular trading, layering).
//!    Best for: Network-based anomaly detection.

mod types;
mod codegen;

pub mod ring_kernel;
pub mod batch_kernel;
pub mod stencil_kernel;

pub use types::*;
pub use codegen::*;
