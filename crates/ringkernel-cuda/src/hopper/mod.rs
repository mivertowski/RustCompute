//! Hopper (H100) Architecture Features — Phase 5
//!
//! This module provides Hopper-specific GPU features:
//! - **Thread Block Clusters**: Group thread blocks for DSMEM access and cluster-level sync
//! - **Distributed Shared Memory (DSMEM)**: Shared memory accessible across blocks in a cluster
//! - **TMA Async Copy**: Tensor Memory Accelerator for efficient bulk data movement
//! - **Green Contexts**: SM partitioning for resource isolation
//!
//! All features require compute capability 9.0+ (H100/H200) and gracefully fall back
//! on older architectures.

pub mod async_mem;
pub mod cluster;
pub mod dsmem;
pub mod green_ctx;
pub mod lifecycle;
pub mod tma;

use ringkernel_core::error::{Result, RingKernelError};

use crate::device::CudaDevice;

/// Check if the device supports Hopper features (compute capability 9.0+).
pub fn check_hopper_support(device: &CudaDevice) -> Result<()> {
    let (major, minor) = device.compute_capability();
    if major < 9 {
        return Err(RingKernelError::NotSupported(format!(
            "Hopper features require compute capability 9.0+, device has {}.{}",
            major, minor
        )));
    }
    Ok(())
}

/// Check if the device supports cluster launch.
pub fn supports_cluster_launch(device: &CudaDevice) -> bool {
    let (major, _) = device.compute_capability();
    major >= 9
}

/// Maximum portable cluster size (works across all Hopper devices).
pub const MAX_PORTABLE_CLUSTER_SIZE: u32 = 8;

/// Maximum cluster size on Blackwell (B200).
pub const MAX_BLACKWELL_CLUSTER_SIZE: u32 = 16;
