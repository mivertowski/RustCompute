//! Multi-GPU runtime infrastructure.
//!
//! This module provides the primitives needed to coordinate persistent
//! actors across multiple CUDA devices:
//!
//! - [`topology`]: NVLink / P2P topology detection (adjacency matrix,
//!   bandwidth, shortest-hop path) used for placement decisions.
//! - [`runtime`]: multi-device `CudaRuntime` facade that dispatches launches
//!   across devices (spec section 4.1).
//! - [`registry`]: per-GPU actor registry — tracks which actor lives on
//!   which device (used for routing and rebalance decisions).
//! - [`migration`]: 3-phase actor migration protocol (quiesce, transfer,
//!   swap) for rebalancing actors across GPUs (spec section 3.1).
//! - [`migration_controller`]: rate limiting and buffer-budget accounting
//!   that backpressures migration bursts.
//! - [`migration_kernels`]: Rust wrappers for the GPU-side migration kernels
//!   (state capture/restore/CRC32, in-flight queue drain).
//! - [`staging`]: in-flight message staging buffer with optional disk
//!   spill for ultra-burst workloads.
//!
//! The implementations here are designed to **compile and unit-test
//! without CUDA hardware**. Any call that would require a live GPU
//! (e.g. `cuCtxEnablePeerAccess`) is deferred to the point where a GPU
//! path actually executes; single-GPU code paths work on any host.
//!
//! See `docs/superpowers/specs/2026-04-17-v1.1-vyngraph-gaps.md` for the
//! full specification.

pub mod migration;
pub mod migration_controller;
pub mod migration_kernels;
pub mod registry;
pub mod runtime;
pub mod staging;
pub mod topology;

pub use migration::{
    MigrationPlan, MigrationReport, MultiGpuError, PhaseDurations, RebalanceStrategy,
};
pub use migration_controller::{MigrationController, MigrationControllerConfig, MigrationPermit};
pub use migration_kernels::{CaptureResult, DrainResult, MigrationKernels};
pub use registry::MultiGpuRegistry;
pub use runtime::{MultiGpuRuntime, PlacementHint};
pub use staging::StagingBuffer;
pub use topology::NvlinkTopology;
