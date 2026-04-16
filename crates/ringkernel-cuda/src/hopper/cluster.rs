//! Thread Block Cluster support for Hopper (H100) and newer GPUs.
//!
//! Thread Block Clusters group multiple thread blocks that can:
//! - Access each other's shared memory (DSMEM)
//! - Synchronize at cluster granularity (faster than grid sync)
//! - Be co-scheduled on the same GPC for low-latency communication
//!
//! # Usage
//!
//! ```ignore
//! use ringkernel_cuda::hopper::cluster::{ClusterLaunchConfig, launch_kernel_with_cluster};
//!
//! let config = ClusterLaunchConfig {
//!     grid_dim: (16, 1, 1),
//!     block_dim: (256, 1, 1),
//!     cluster_dim: (4, 1, 1),  // 4 blocks per cluster
//!     shared_mem_bytes: 0,
//! };
//!
//! unsafe { launch_kernel_with_cluster(func, &config, &mut params, stream)?; }
//! ```

use std::ffi::c_void;
use std::mem;
use std::ptr;

use cudarc::driver::sys as cuda_sys;

use ringkernel_core::error::{Result, RingKernelError};

use crate::device::CudaDevice;
use crate::driver_api::{DirectPtxModule, KernelParams};

// Include build-time generated cluster kernel PTX
include!(concat!(env!("OUT_DIR"), "/cluster_kernels.rs"));

/// Available pre-compiled cluster kernels.
pub mod kernels {
    /// Cluster sync test kernel.
    pub const CLUSTER_TEST_SYNC: &str = "cluster_test_sync";
    /// DSMEM K2K messaging kernel.
    pub const CLUSTER_DSMEM_K2K: &str = "cluster_dsmem_k2k";
    /// Persistent actor kernel with cluster support.
    pub const CLUSTER_PERSISTENT_ACTOR: &str = "cluster_persistent_actor";
}

/// Check if cluster kernels were compiled at build time.
pub fn has_cluster_kernel_support() -> bool {
    HAS_CLUSTER_KERNEL_SUPPORT
}

/// Get the pre-compiled cluster kernel PTX.
pub fn cluster_kernel_ptx() -> &'static str {
    CLUSTER_KERNEL_PTX
}

/// Configuration for cluster-aware kernel launch.
#[derive(Debug, Clone)]
pub struct ClusterLaunchConfig {
    /// Grid dimensions (number of blocks in x, y, z).
    /// Must be divisible by cluster_dim in each dimension.
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (threads per block in x, y, z).
    pub block_dim: (u32, u32, u32),
    /// Cluster dimensions (blocks per cluster in x, y, z).
    /// Product must be <= 8 for portable code (up to 16 on Blackwell).
    /// Default: (1, 1, 1) = no clustering.
    pub cluster_dim: (u32, u32, u32),
    /// Shared memory per block in bytes.
    pub shared_mem_bytes: u32,
    /// Cluster scheduling policy preference.
    pub scheduling_policy: ClusterSchedulingPolicy,
}

impl Default for ClusterLaunchConfig {
    fn default() -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            cluster_dim: (1, 1, 1),
            shared_mem_bytes: 0,
            scheduling_policy: ClusterSchedulingPolicy::Default,
        }
    }
}

impl ClusterLaunchConfig {
    /// Total number of blocks per cluster.
    pub fn blocks_per_cluster(&self) -> u32 {
        self.cluster_dim.0 * self.cluster_dim.1 * self.cluster_dim.2
    }

    /// Total number of clusters in the grid.
    pub fn num_clusters(&self) -> u32 {
        let clusters_x = self.grid_dim.0 / self.cluster_dim.0.max(1);
        let clusters_y = self.grid_dim.1 / self.cluster_dim.1.max(1);
        let clusters_z = self.grid_dim.2 / self.cluster_dim.2.max(1);
        clusters_x * clusters_y * clusters_z
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        let blocks_per_cluster = self.blocks_per_cluster();

        if blocks_per_cluster > super::MAX_PORTABLE_CLUSTER_SIZE {
            return Err(RingKernelError::InvalidConfig(format!(
                "Cluster size {} exceeds portable maximum {}. Use <= 8 blocks per cluster.",
                blocks_per_cluster,
                super::MAX_PORTABLE_CLUSTER_SIZE
            )));
        }

        if blocks_per_cluster > 0 && !blocks_per_cluster.is_power_of_two() {
            return Err(RingKernelError::InvalidConfig(format!(
                "Cluster size {} must be a power of 2",
                blocks_per_cluster
            )));
        }

        // Grid must be divisible by cluster dimensions
        if self.grid_dim.0 % self.cluster_dim.0 != 0
            || self.grid_dim.1 % self.cluster_dim.1 != 0
            || self.grid_dim.2 % self.cluster_dim.2 != 0
        {
            return Err(RingKernelError::InvalidConfig(format!(
                "Grid dimensions {:?} must be divisible by cluster dimensions {:?}",
                self.grid_dim, self.cluster_dim
            )));
        }

        Ok(())
    }
}

/// Cluster scheduling policy preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterSchedulingPolicy {
    /// Let the driver choose the scheduling policy.
    Default,
    /// Spread clusters across GPCs for maximum bandwidth.
    Spread,
    /// Pack clusters on the same GPC for low-latency DSMEM access.
    ClusterOnGpc,
}

impl ClusterSchedulingPolicy {
    fn to_cuda(&self) -> cuda_sys::CUclusterSchedulingPolicy {
        match self {
            Self::Default => cuda_sys::CUclusterSchedulingPolicy::CU_CLUSTER_SCHEDULING_POLICY_DEFAULT,
            Self::Spread => cuda_sys::CUclusterSchedulingPolicy::CU_CLUSTER_SCHEDULING_POLICY_SPREAD,
            Self::ClusterOnGpc => cuda_sys::CUclusterSchedulingPolicy::CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING,
        }
    }
}

/// Launch a kernel with Thread Block Cluster configuration via `cuLaunchKernelEx`.
///
/// This uses the CUDA 12.0+ extended launch API to set cluster dimensions
/// as launch attributes, enabling DSMEM access and cluster-level sync.
///
/// # Safety
///
/// - `func` must be a valid CUfunction handle
/// - `kernel_params` must contain valid pointers matching the kernel signature
/// - Grid dimensions must be valid and divisible by cluster dimensions
/// - The device must support compute capability 9.0+
pub unsafe fn launch_kernel_with_cluster(
    func: cuda_sys::CUfunction,
    config: &ClusterLaunchConfig,
    kernel_params: &mut [*mut c_void],
    stream: cuda_sys::CUstream,
) -> Result<()> {
    config.validate()?;

    let blocks_per_cluster = config.blocks_per_cluster();

    // If cluster size is 1, fall back to standard cooperative launch
    if blocks_per_cluster <= 1 {
        return launch_standard(func, config, kernel_params, stream);
    }

    // Build launch attributes array
    let mut attrs: Vec<cuda_sys::CUlaunchAttribute_st> = Vec::new();

    // Attribute 1: Cluster dimension
    let mut cluster_attr: cuda_sys::CUlaunchAttribute_st = mem::zeroed();
    cluster_attr.id = cuda_sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    cluster_attr.value.clusterDim.x = config.cluster_dim.0;
    cluster_attr.value.clusterDim.y = config.cluster_dim.1;
    cluster_attr.value.clusterDim.z = config.cluster_dim.2;
    attrs.push(cluster_attr);

    // Attribute 2: Scheduling policy (if not default)
    if config.scheduling_policy != ClusterSchedulingPolicy::Default {
        let mut policy_attr: cuda_sys::CUlaunchAttribute_st = mem::zeroed();
        policy_attr.id = cuda_sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
        policy_attr.value.clusterSchedulingPolicyPreference = config.scheduling_policy.to_cuda();
        attrs.push(policy_attr);
    }

    // Attribute 3: Cooperative flag (enables grid.sync within cluster)
    let mut coop_attr: cuda_sys::CUlaunchAttribute_st = mem::zeroed();
    coop_attr.id = cuda_sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
    coop_attr.value.cooperative = 1;
    attrs.push(coop_attr);

    // Build CUlaunchConfig
    let launch_config = cuda_sys::CUlaunchConfig_st {
        gridDimX: config.grid_dim.0,
        gridDimY: config.grid_dim.1,
        gridDimZ: config.grid_dim.2,
        blockDimX: config.block_dim.0,
        blockDimY: config.block_dim.1,
        blockDimZ: config.block_dim.2,
        sharedMemBytes: config.shared_mem_bytes,
        hStream: stream,
        attrs: attrs.as_mut_ptr(),
        numAttrs: attrs.len() as u32,
    };

    let result = cuda_sys::cuLaunchKernelEx(
        &launch_config as *const _,
        func,
        kernel_params.as_mut_ptr(),
        ptr::null_mut(), // extra
    );

    if result != cuda_sys::CUresult::CUDA_SUCCESS {
        return Err(RingKernelError::LaunchFailed(format!(
            "cuLaunchKernelEx with cluster config failed: {:?} \
             (grid={:?}, block={:?}, cluster={:?})",
            result, config.grid_dim, config.block_dim, config.cluster_dim
        )));
    }

    Ok(())
}

/// Fallback to standard kernel launch (no clustering).
unsafe fn launch_standard(
    func: cuda_sys::CUfunction,
    config: &ClusterLaunchConfig,
    kernel_params: &mut [*mut c_void],
    stream: cuda_sys::CUstream,
) -> Result<()> {
    let result = cuda_sys::cuLaunchKernel(
        func,
        config.grid_dim.0,
        config.grid_dim.1,
        config.grid_dim.2,
        config.block_dim.0,
        config.block_dim.1,
        config.block_dim.2,
        config.shared_mem_bytes,
        stream,
        kernel_params.as_mut_ptr(),
        ptr::null_mut(),
    );

    if result != cuda_sys::CUresult::CUDA_SUCCESS {
        return Err(RingKernelError::LaunchFailed(format!(
            "cuLaunchKernel failed: {:?}",
            result
        )));
    }

    Ok(())
}

/// A cooperative kernel with cluster launch capability.
///
/// Extends `DirectCooperativeKernel` with Thread Block Cluster support.
pub struct ClusterKernel {
    /// The loaded PTX module (kept alive for the lifetime of the kernel).
    _module: DirectPtxModule,
    /// Raw CUfunction handle.
    func: cuda_sys::CUfunction,
    /// Block size for occupancy calculations.
    #[allow(dead_code)]
    block_size: u32,
    /// Maximum concurrent blocks for cooperative launch.
    max_blocks: u32,
    /// Function name (for debugging).
    func_name: String,
    /// Whether the device supports cluster launch.
    has_cluster_support: bool,
}

// SAFETY: Same guarantees as DirectCooperativeKernel — CUfunction handles are
// derived from CUmodule and share thread-safety guarantees. All mutable state
// is initialized at construction and read-only thereafter.
unsafe impl Send for ClusterKernel {}
unsafe impl Sync for ClusterKernel {}

impl ClusterKernel {
    /// Load a kernel with cluster launch capability from PTX source.
    pub fn from_ptx(
        device: &CudaDevice,
        ptx: &str,
        func_name: &str,
        block_size: u32,
    ) -> Result<Self> {
        let module = DirectPtxModule::load_ptx(device, ptx)?;
        let func = module.get_function(func_name)?;

        // Query max concurrent blocks
        let mut max_blocks_per_sm: i32 = 0;
        let result = unsafe {
            cuda_sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut max_blocks_per_sm,
                func,
                block_size as i32,
                0,
            )
        };

        if result != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(RingKernelError::BackendError(format!(
                "Occupancy query failed: {:?}",
                result
            )));
        }

        let num_sms = device
            .inner()
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| RingKernelError::BackendError(format!("Failed to get SM count: {}", e)))?;

        let max_blocks = (max_blocks_per_sm as u32) * (num_sms as u32);
        let has_cluster_support = super::supports_cluster_launch(device);

        Ok(Self {
            _module: module,
            func,
            block_size,
            max_blocks,
            func_name: func_name.to_string(),
            has_cluster_support,
        })
    }

    /// Launch with cluster configuration.
    ///
    /// If the device doesn't support clusters, falls back to standard launch
    /// with cluster_dim ignored.
    ///
    /// # Safety
    ///
    /// - `kernel_params` must contain valid pointers matching the kernel signature
    pub unsafe fn launch(
        &self,
        config: &ClusterLaunchConfig,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        let total_blocks = config.grid_dim.0 * config.grid_dim.1 * config.grid_dim.2;
        if total_blocks > self.max_blocks {
            return Err(RingKernelError::LaunchFailed(format!(
                "Grid size {} exceeds max concurrent blocks {}",
                total_blocks, self.max_blocks
            )));
        }

        let stream: cuda_sys::CUstream = ptr::null_mut();

        if self.has_cluster_support && config.blocks_per_cluster() > 1 {
            launch_kernel_with_cluster(self.func, config, kernel_params, stream)
        } else {
            launch_standard(self.func, config, kernel_params, stream)
        }
    }

    /// Launch with typed parameters.
    ///
    /// # Safety
    ///
    /// - All parameter types must match the kernel signature exactly
    pub unsafe fn launch_params<T: KernelParams>(
        &self,
        config: &ClusterLaunchConfig,
        params: &mut T,
    ) -> Result<()> {
        let mut param_ptrs = params.as_param_ptrs();
        self.launch(config, &mut param_ptrs)
    }

    /// Synchronize after launch.
    pub fn synchronize(&self) -> Result<()> {
        let result = unsafe { cuda_sys::cuCtxSynchronize() };
        if result != cuda_sys::CUresult::CUDA_SUCCESS {
            return Err(RingKernelError::BackendError(format!(
                "cuCtxSynchronize failed: {:?}",
                result
            )));
        }
        Ok(())
    }

    /// Get maximum concurrent blocks.
    pub fn max_concurrent_blocks(&self) -> u32 {
        self.max_blocks
    }

    /// Whether the device supports cluster launch.
    pub fn has_cluster_support(&self) -> bool {
        self.has_cluster_support
    }

    /// Get the function name.
    pub fn func_name(&self) -> &str {
        &self.func_name
    }

    /// Get the raw CUfunction handle.
    pub fn raw_function(&self) -> cuda_sys::CUfunction {
        self.func
    }
}

/// Query the maximum cluster size supported by the device for a given kernel.
pub fn query_max_cluster_size(
    device: &CudaDevice,
    _func: cuda_sys::CUfunction,
) -> Result<u32> {
    if !super::supports_cluster_launch(device) {
        return Ok(1);
    }

    // On Hopper, the max portable cluster size is 8
    // On Blackwell, it can be up to 16
    let (major, _) = device.compute_capability();
    if major >= 10 {
        Ok(super::MAX_BLACKWELL_CLUSTER_SIZE)
    } else {
        Ok(super::MAX_PORTABLE_CLUSTER_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_config_defaults() {
        let config = ClusterLaunchConfig::default();
        assert_eq!(config.blocks_per_cluster(), 1);
        assert_eq!(config.num_clusters(), 1);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cluster_config_validation() {
        let config = ClusterLaunchConfig {
            grid_dim: (16, 1, 1),
            block_dim: (256, 1, 1),
            cluster_dim: (4, 1, 1),
            ..Default::default()
        };
        assert_eq!(config.blocks_per_cluster(), 4);
        assert_eq!(config.num_clusters(), 4);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cluster_config_not_power_of_two() {
        let config = ClusterLaunchConfig {
            grid_dim: (9, 1, 1),
            block_dim: (256, 1, 1),
            cluster_dim: (3, 1, 1),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cluster_config_too_large() {
        let config = ClusterLaunchConfig {
            grid_dim: (16, 1, 1),
            block_dim: (256, 1, 1),
            cluster_dim: (16, 1, 1),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cluster_config_grid_not_divisible() {
        let config = ClusterLaunchConfig {
            grid_dim: (15, 1, 1),
            block_dim: (256, 1, 1),
            cluster_dim: (4, 1, 1),
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_scheduling_policy_conversion() {
        assert_eq!(
            ClusterSchedulingPolicy::Default.to_cuda(),
            cuda_sys::CUclusterSchedulingPolicy::CU_CLUSTER_SCHEDULING_POLICY_DEFAULT,
        );
    }

    #[test]
    #[ignore] // Requires H100 GPU
    fn test_cluster_kernel_loading() {
        use crate::cooperative;

        if !cooperative::has_cooperative_support() {
            return;
        }

        let device = CudaDevice::new(0).expect("Failed to create device");
        if !super::super::supports_cluster_launch(&device) {
            return;
        }

        let ptx = cooperative::cooperative_kernel_ptx();
        let kernel = ClusterKernel::from_ptx(&device, ptx, "coop_test_grid_sync", 256);
        match kernel {
            Ok(k) => {
                assert!(k.has_cluster_support());
                assert!(k.max_concurrent_blocks() > 0);
            }
            Err(e) => {
                tracing::warn!(%e, "Failed to load cluster kernel");
            }
        }
    }
}
