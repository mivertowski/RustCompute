//! Cooperative groups kernel launch support.
//!
//! This module provides `cuLaunchCooperativeKernel` functionality for
//! kernels that use grid-wide synchronization via cooperative groups.
//!
//! # Requirements
//!
//! - NVIDIA GPU with compute capability 6.0+ (Pascal or newer)
//! - nvcc available at build time (for PTX compilation)
//! - Grid size must fit within device's max concurrent block limit
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cuda::cooperative::{CooperativeKernel, CooperativeLaunchConfig};
//!
//! let kernel = CooperativeKernel::from_precompiled(&device, "coop_persistent_fdtd", 256)?;
//! let max_blocks = kernel.max_concurrent_blocks();
//!
//! let config = CooperativeLaunchConfig {
//!     grid_dim: (num_blocks, 1, 1),
//!     block_dim: (256, 1, 1),
//!     ..Default::default()
//! };
//!
//! unsafe { kernel.launch(&config, &mut params)?; }
//! ```

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::result as cuda_result;
use cudarc::driver::sys as cuda_sys;
use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig};

use ringkernel_core::error::{Result, RingKernelError};

use crate::device::CudaDevice;

// Include build-time generated PTX (or stub)
include!(concat!(env!("OUT_DIR"), "/cooperative_kernels.rs"));

/// Launch a kernel cooperatively using the CUDA driver API.
///
/// This is a thin wrapper around cudarc's `result::launch_cooperative_kernel`,
/// which calls `cuLaunchCooperativeKernel` directly, enabling true grid-wide
/// synchronization via `grid.sync()` in cooperative groups.
///
/// # Safety
///
/// - `func` must be a valid CUfunction handle
/// - `kernel_params` must contain valid pointers matching the kernel signature
/// - Grid size must not exceed the device's cooperative launch limits
/// - The stream must be valid
pub unsafe fn launch_cooperative_kernel(
    func: cuda_sys::CUfunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: cuda_sys::CUstream,
    kernel_params: &mut [*mut c_void],
) -> std::result::Result<(), cudarc::driver::DriverError> {
    // Use cudarc's native cooperative launch (added in 0.18.0)
    cuda_result::launch_cooperative_kernel(
        func,
        grid_dim,
        block_dim,
        shared_mem_bytes,
        stream,
        kernel_params,
    )
}

/// Check if cooperative groups support is available.
///
/// Returns true if nvcc was found at build time and cooperative kernels were compiled.
pub fn has_cooperative_support() -> bool {
    HAS_COOPERATIVE_SUPPORT
}

/// Get pre-compiled cooperative kernel PTX.
///
/// Returns empty string if nvcc was not available at build time.
pub fn cooperative_kernel_ptx() -> &'static str {
    COOPERATIVE_KERNEL_PTX
}

/// Get the build message about cooperative support.
pub fn cooperative_build_message() -> &'static str {
    COOPERATIVE_BUILD_MESSAGE
}

/// Configuration for cooperative kernel launch.
#[derive(Debug, Clone)]
pub struct CooperativeLaunchConfig {
    /// Grid dimensions (number of blocks in x, y, z).
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (threads per block in x, y, z).
    pub block_dim: (u32, u32, u32),
    /// Shared memory per block in bytes.
    pub shared_mem_bytes: u32,
}

impl Default for CooperativeLaunchConfig {
    fn default() -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Check if a device supports cooperative groups.
pub fn check_cooperative_support(device: &CudaDevice) -> Result<()> {
    let (major, minor) = device.compute_capability();

    // Cooperative groups require compute capability 6.0+ (Pascal)
    if major < 6 {
        return Err(RingKernelError::NotSupported(format!(
            "Cooperative groups require compute capability 6.0+, device has {}.{}",
            major, minor
        )));
    }

    // Check if cooperative launch is supported
    let supports_coop = device
        .inner()
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH)
        .unwrap_or(0);

    if supports_coop == 0 {
        return Err(RingKernelError::NotSupported(
            "Device does not support cooperative launch".to_string(),
        ));
    }

    Ok(())
}

/// High-level wrapper for cooperative kernel loading and launching.
///
/// Handles PTX loading, occupancy queries, and cooperative launch.
/// Uses cudarc 0.18.2's native `launch_cooperative` support.
pub struct CooperativeKernel {
    /// Underlying cudarc context.
    context: Arc<CudaContext>,
    /// Loaded kernel function (for occupancy queries and typed launch).
    func: CudaFunction,
    /// Raw CUfunction handle for raw parameter cooperative launch.
    /// Note: CudaFunction.cu_function is pub(crate) in cudarc, so we extract it
    /// via transmute. This is version-dependent but safer than bypassing cudarc entirely.
    cu_func: cuda_sys::CUfunction,
    /// Module name for identification.
    #[allow(dead_code)]
    module_name: String,
    /// Function name.
    func_name: String,
    /// Block size this kernel was configured for.
    block_size: u32,
    /// Maximum concurrent blocks for cooperative launch.
    max_blocks: u32,
}

impl CooperativeKernel {
    /// Load a cooperative kernel from PTX source.
    ///
    /// # Arguments
    ///
    /// * `device` - CUDA device to load kernel on
    /// * `ptx` - PTX source code
    /// * `module_name` - Name for the loaded module (must be 'static)
    /// * `func_name` - Name of the kernel function (must be 'static)
    /// * `block_size` - Block size (threads per block) for occupancy calculation
    ///
    /// Note: module_name and func_name must be 'static because cudarc caches
    /// these names internally. Use from_precompiled() for the common case.
    pub fn from_ptx(
        device: &CudaDevice,
        ptx: &str,
        module_name: &'static str,
        func_name: &'static str,
        block_size: u32,
    ) -> Result<Self> {
        // Check cooperative support
        check_cooperative_support(device)?;

        if ptx.is_empty() {
            return Err(RingKernelError::NotSupported(
                "Cooperative kernel PTX is empty (nvcc not available at build time)".to_string(),
            ));
        }

        // Load PTX using cudarc 0.18's module API
        let ptx_code = cudarc::nvrtc::Ptx::from_src(ptx);
        let module = device
            .inner()
            .load_module(ptx_code)
            .map_err(|e| RingKernelError::LaunchFailed(format!("PTX load failed: {}", e)))?;

        // Get function handle from module
        let func = module
            .load_function(func_name)
            .map_err(|e| RingKernelError::KernelNotFound(format!("{}: {}", func_name, e)))?;

        // Get raw CUfunction handle for raw parameter cooperative launch.
        // CudaFunction.cu_function is pub(crate) in cudarc 0.18, so we extract it via transmute.
        // This is version-dependent but cudarc's internal layout is: { cu_function: CUfunction, ... }
        let cu_func = unsafe {
            #[repr(C)]
            struct CudaFunctionRepr {
                cu_function: cuda_sys::CUfunction,
                // Other fields follow but we only need the first one
            }

            let func_repr: &CudaFunctionRepr = std::mem::transmute(&func);
            func_repr.cu_function
        };

        // Calculate max concurrent blocks using occupancy API
        let max_blocks_per_sm = func
            .occupancy_max_active_blocks_per_multiprocessor(block_size, 0, None)
            .map_err(|e| {
                RingKernelError::BackendError(format!("Failed to query occupancy: {:?}", e))
            })?;

        // Get number of SMs
        let num_sms = device
            .inner()
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_err(|e| RingKernelError::BackendError(format!("Failed to get SM count: {}", e)))?;

        // Max grid size = max_blocks_per_sm * num_sms
        let max_blocks = max_blocks_per_sm * (num_sms as u32);

        Ok(Self {
            context: Arc::clone(device.inner()),
            func,
            cu_func,
            module_name: module_name.to_string(),
            func_name: func_name.to_string(),
            block_size,
            max_blocks,
        })
    }

    /// Load from pre-compiled cooperative kernels.
    ///
    /// Uses PTX embedded at build time by build.rs.
    /// Use constants from the `kernels` module for func_name.
    pub fn from_precompiled(
        device: &CudaDevice,
        func_name: &'static str,
        block_size: u32,
    ) -> Result<Self> {
        if !HAS_COOPERATIVE_SUPPORT {
            return Err(RingKernelError::NotSupported(format!(
                "Cooperative groups not available: {}",
                COOPERATIVE_BUILD_MESSAGE
            )));
        }

        Self::from_ptx(
            device,
            COOPERATIVE_KERNEL_PTX,
            "coop_kernels",
            func_name,
            block_size,
        )
    }

    /// Get maximum grid size for cooperative launch.
    ///
    /// Returns the maximum number of blocks that can run concurrently.
    /// Grid size must not exceed this value for cooperative launch.
    pub fn max_concurrent_blocks(&self) -> u32 {
        self.max_blocks
    }

    /// Get the block size this kernel was configured for.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Get the function name.
    pub fn func_name(&self) -> &str {
        &self.func_name
    }

    /// Launch the kernel cooperatively using raw parameters.
    ///
    /// This performs a true cooperative launch via `cuLaunchCooperativeKernel`,
    /// where all blocks must be able to run concurrently. Grid-wide synchronization
    /// via `grid.sync()` is guaranteed to work.
    ///
    /// Uses cudarc 0.18.2's native cooperative launch support.
    ///
    /// # Safety
    ///
    /// - `kernel_params` must contain valid pointers matching the kernel signature
    /// - Each element must point to the corresponding kernel argument
    /// - Grid size must not exceed `max_concurrent_blocks()`
    /// - The kernel must have been compiled with cooperative groups support
    pub unsafe fn launch(
        &self,
        config: &CooperativeLaunchConfig,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        // Validate grid size
        let total_blocks = config.grid_dim.0 * config.grid_dim.1 * config.grid_dim.2;
        if total_blocks > self.max_blocks {
            return Err(RingKernelError::LaunchFailed(format!(
                "Grid size {} exceeds max concurrent blocks {} for cooperative launch. \
                 Reduce grid size or use standard kernel launch.",
                total_blocks, self.max_blocks
            )));
        }

        // Use cudarc's native cooperative launch (cudarc 0.18+)
        // Get the default stream for cooperative launch
        let stream: cuda_sys::CUstream = std::ptr::null_mut();

        cuda_result::launch_cooperative_kernel(
            self.cu_func,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            stream,
            kernel_params,
        )
        .map_err(|e| {
            RingKernelError::LaunchFailed(format!("Cooperative kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }

    /// Launch with raw kernel parameters (alias for `launch`).
    ///
    /// This is kept for backwards compatibility. Use `launch()` directly.
    ///
    /// # Safety
    ///
    /// Same as `launch`.
    #[inline]
    pub unsafe fn launch_cooperative_raw(
        &self,
        config: &CooperativeLaunchConfig,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        self.launch(config, kernel_params)
    }

    /// Synchronize device after launch.
    pub fn synchronize(&self) -> Result<()> {
        self.context
            .synchronize()
            .map_err(|e| RingKernelError::BackendError(format!("Synchronize failed: {}", e)))
    }

    /// Get the underlying CudaFunction for advanced use cases.
    pub fn cuda_function(&self) -> &CudaFunction {
        &self.func
    }

    /// Get the cudarc launch config from our config.
    pub fn to_launch_config(config: &CooperativeLaunchConfig) -> LaunchConfig {
        LaunchConfig {
            grid_dim: config.grid_dim,
            block_dim: config.block_dim,
            shared_mem_bytes: config.shared_mem_bytes,
        }
    }
}

/// Available pre-compiled cooperative kernels.
pub mod kernels {
    /// Block-based FDTD kernel with cooperative groups.
    pub const COOP_PERSISTENT_FDTD: &str = "coop_persistent_fdtd";

    /// Generic cooperative ring kernel entry point.
    pub const COOP_RING_KERNEL_ENTRY: &str = "coop_ring_kernel_entry";

    /// Test kernel for grid sync functionality.
    pub const COOP_TEST_GRID_SYNC: &str = "coop_test_grid_sync";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooperative_support_detection() {
        let has_support = has_cooperative_support();
        let message = cooperative_build_message();
        println!("Cooperative support: {}", has_support);
        println!("Build message: {}", message);

        // If we have support, PTX should not be empty
        if has_support {
            assert!(!cooperative_kernel_ptx().is_empty());
        }
    }

    #[test]
    fn test_default_config() {
        let config = CooperativeLaunchConfig::default();
        assert_eq!(config.grid_dim, (1, 1, 1));
        assert_eq!(config.block_dim, (256, 1, 1));
        assert_eq!(config.shared_mem_bytes, 0);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cooperative_kernel_loading() {
        if !has_cooperative_support() {
            println!("Skipping: cooperative support not available");
            return;
        }

        let device = CudaDevice::new(0).expect("Failed to create device");

        // Check compute capability
        let (major, _minor) = device.compute_capability();
        if major < 6 {
            println!("Skipping: device doesn't support cooperative groups");
            return;
        }

        let kernel =
            CooperativeKernel::from_precompiled(&device, kernels::COOP_PERSISTENT_FDTD, 256);

        match kernel {
            Ok(k) => {
                println!("Loaded cooperative kernel: {}", k.func_name());
                println!("Max concurrent blocks: {}", k.max_concurrent_blocks());
            }
            Err(e) => {
                println!("Failed to load: {}", e);
            }
        }
    }
}
