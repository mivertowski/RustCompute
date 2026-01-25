//! Kernel mode selection and launch configuration.
//!
//! This module provides abstractions for selecting optimal kernel execution modes
//! and computing launch configurations based on workload characteristics and
//! GPU architecture.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cuda::launch_config::{
//!     KernelMode, WorkloadProfile, AccessPattern, GpuArchitecture, KernelModeSelector,
//! };
//!
//! let arch = GpuArchitecture::from_device(&device);
//! let selector = KernelModeSelector::new(arch);
//!
//! let profile = WorkloadProfile::new(1_000_000, 64)
//!     .with_access_pattern(AccessPattern::Coalesced);
//!
//! let mode = selector.select(&profile);
//! let block_size = selector.recommended_block_size(mode);
//! let grid_size = selector.recommended_grid_size(mode, profile.element_count);
//! ```

mod mode;

pub use mode::{
    AccessPattern, GpuArchitecture, KernelMode, KernelModeSelector, LaunchConfig, WorkloadProfile,
};
