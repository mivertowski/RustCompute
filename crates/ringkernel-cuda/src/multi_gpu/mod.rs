//! Multi-GPU runtime infrastructure.
//!
//! This module provides the primitives needed to coordinate persistent
//! actors across multiple CUDA devices:
//!
//! - [`topology`]: NVLink / P2P topology detection (adjacency matrix,
//!   bandwidth, shortest-hop path) used for placement decisions.
//! - `runtime`: multi-device `CudaRuntime` facade that dispatches launches
//!   across devices (coming in v1.1 — see spec section 4.1).
//! - `migration`: 3-phase actor migration protocol (quiesce, transfer, swap)
//!   for rebalancing actors across GPUs (coming in v1.1 — see spec section 3.1).
//!
//! Today only `topology` is wired up; the other submodules will be added as
//! the v1.1 multi-GPU work lands.

pub mod topology;

// Coming in v1.1 — see docs/superpowers/specs/2026-04-17-v1.1-vyngraph-gaps.md
// pub mod runtime;
// pub mod migration;
