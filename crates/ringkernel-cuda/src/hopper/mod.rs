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
// Green Contexts (CUDA 12.4+) — cudarc only exposes `CUgreenCtx` and its FFI
// functions behind these version features (matching NVIDIA's own
// introduction of the API in CUDA 12.4). Gating this at compile time, not
// just at runtime via `check_hopper_support()` below, so the crate still
// builds on any CUDA Toolkit older than 12.4 — which otherwise fails to
// compile at all, on any GPU, regardless of architecture.
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000",
    feature = "cuda-13010",
    feature = "cuda-13020"
))]
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

#[cfg(test)]
mod pascal_compat_tests {
    //! Confirms Hopper-only feature checks degrade gracefully (return an
    //! `Err`/`false`, not a panic or UB) on pre-Hopper hardware, now that the
    //! compile-time gating fix above lets this module build at all on CUDA
    //! Toolkits older than 12.4. Requires real GPU hardware — run with:
    //!   cargo test -p ringkernel-cuda --features cuda -- --ignored pascal_compat
    use super::*;
    use crate::device::CudaDevice;

    #[test]
    #[ignore] // Requires CUDA hardware
    fn pascal_compat_check_hopper_support_degrades_gracefully() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        let (major, minor) = device.compute_capability();
        println!("Device compute capability: {major}.{minor}");

        let result = check_hopper_support(&device);
        if major < 9 {
            assert!(
                result.is_err(),
                "check_hopper_support returned Ok on pre-Hopper hardware"
            );
        } else {
            assert!(result.is_ok());
        }
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn pascal_compat_supports_cluster_launch_degrades_gracefully() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        let (major, _) = device.compute_capability();

        let supported = supports_cluster_launch(&device);
        if major < 9 {
            assert!(
                !supported,
                "supports_cluster_launch returned true on pre-Hopper hardware"
            );
        }
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn pascal_compat_dsmem_degrades_gracefully() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        let (major, _) = device.compute_capability();

        let available = dsmem::is_dsmem_available(&device, 4);
        if major < 9 {
            assert!(
                !available,
                "is_dsmem_available returned true on pre-Hopper hardware"
            );
        }
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn pascal_compat_cluster_size_degrades_to_one() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        let (major, _) = device.compute_capability();

        let max_cluster = cluster::query_max_cluster_size(&device, std::ptr::null_mut())
            .expect("query_max_cluster_size should not error");
        if major < 9 {
            assert_eq!(
                max_cluster, 1,
                "query_max_cluster_size did not degrade to 1 on pre-Hopper hardware"
            );
        }
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn pascal_compat_cooperative_groups_still_available() {
        // Cooperative groups (grid.sync()) have a lower floor (CC 6.0+,
        // Pascal) than Hopper-specific features (CC 9.0+) — confirms the
        // gating logic distinguishes per-feature floors correctly, rather
        // than treating "not Hopper" as "nothing works".
        let device = CudaDevice::new(0).expect("Failed to create device");
        let (major, _) = device.compute_capability();

        let coop = device.supports_cooperative_groups();
        if major >= 6 {
            assert!(
                coop,
                "supports_cooperative_groups returned false on CC 6.0+ hardware"
            );
        }
    }
}
