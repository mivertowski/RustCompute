//! Direct CUDA driver API access for safe cooperative kernel launch.
//!
//! This module provides direct PTX module loading via the CUDA driver API,
//! enabling full control over cooperative kernel launch without relying on
//! cudarc's internal CudaFunction struct layout.
//!
//! While cudarc 0.18+ provides `launch_cooperative`, this module is useful for:
//! - Direct PTX loading with `cuModuleLoadData` (cleaner than going through cudarc)
//! - Avoiding the transmute hack to extract CUfunction handles
//! - Full control over persistent kernel lifecycles

use std::ffi::{c_void, CString};
use std::ptr;

use cudarc::driver::result as cuda_result;
use cudarc::driver::sys as cuda_sys;

use ringkernel_core::error::{Result, RingKernelError};

use crate::device::CudaDevice;

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

/// Launch a kernel cooperatively using cudarc's result module.
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
unsafe fn launch_cooperative_kernel(
    func: cuda_sys::CUfunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: cuda_sys::CUstream,
    kernel_params: &mut [*mut c_void],
) -> std::result::Result<(), cuda_result::DriverError> {
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

/// A PTX module loaded directly via the CUDA driver API.
///
/// This provides safe access to kernel functions without going through cudarc's
/// `CudaFunction`, which has a non-`repr(C)` layout that makes extracting the
/// raw `CUfunction` handle fragile.
pub struct DirectPtxModule {
    /// Raw CUDA module handle.
    module: cuda_sys::CUmodule,
}

impl DirectPtxModule {
    /// Load a PTX module from source.
    ///
    /// # Arguments
    ///
    /// * `device` - CUDA device (ensures context is active)
    /// * `ptx` - PTX source code (must be null-terminated or this function adds it)
    pub fn load_ptx(device: &CudaDevice, ptx: &str) -> Result<Self> {
        // Ensure device context is active
        let _ = device.inner();

        // PTX must be null-terminated for CUDA driver API
        let ptx_cstring = if let Some(stripped) = ptx.strip_suffix('\0') {
            CString::new(stripped)
        } else {
            CString::new(ptx)
        }
        .map_err(|e| {
            RingKernelError::LaunchFailed(format!("Invalid PTX (contains null byte): {}", e))
        })?;

        let mut module: cuda_sys::CUmodule = ptr::null_mut();

        // Load the PTX using driver API
        unsafe {
            let result = cuda_sys::cuModuleLoadData(&mut module, ptx_cstring.as_ptr() as *const _);

            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::LaunchFailed(format!(
                    "cuModuleLoadData failed: {:?}",
                    result
                )));
            }
        }

        Ok(Self { module })
    }

    /// Get a function handle from the loaded module.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the kernel function in the PTX
    pub fn get_function(&self, name: &str) -> Result<cuda_sys::CUfunction> {
        let name_cstring = CString::new(name).map_err(|e| {
            RingKernelError::KernelNotFound(format!("Invalid function name: {}", e))
        })?;

        let mut func: cuda_sys::CUfunction = ptr::null_mut();

        unsafe {
            let result =
                cuda_sys::cuModuleGetFunction(&mut func, self.module, name_cstring.as_ptr());

            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::KernelNotFound(format!(
                    "cuModuleGetFunction failed for '{}': {:?}",
                    name, result
                )));
            }
        }

        Ok(func)
    }
}

impl Drop for DirectPtxModule {
    fn drop(&mut self) {
        // Unload the module (ignore errors during cleanup)
        unsafe {
            let _ = cuda_sys::cuModuleUnload(self.module);
        }
    }
}

/// A cooperative kernel loaded via direct driver API.
///
/// This provides true cooperative launch capability without relying on cudarc's
/// internal `CudaFunction` struct layout.
pub struct DirectCooperativeKernel {
    /// The loaded PTX module.
    _module: DirectPtxModule,
    /// Raw CUfunction handle.
    func: cuda_sys::CUfunction,
    /// Block size for occupancy calculations.
    block_size: u32,
    /// Maximum concurrent blocks for this kernel.
    max_blocks: u32,
    /// Function name (for debugging).
    func_name: String,
}

impl DirectCooperativeKernel {
    /// Load a cooperative kernel from PTX source.
    ///
    /// # Arguments
    ///
    /// * `device` - CUDA device to load the kernel on
    /// * `ptx` - PTX source code
    /// * `func_name` - Name of the kernel function in the PTX
    /// * `block_size` - Block size (threads per block) for occupancy calculation
    pub fn from_ptx(
        device: &CudaDevice,
        ptx: &str,
        func_name: &str,
        block_size: u32,
    ) -> Result<Self> {
        // Load module via direct driver API
        let module = DirectPtxModule::load_ptx(device, ptx)?;
        let func = module.get_function(func_name)?;

        // Query max concurrent blocks for cooperative launch
        let max_blocks = Self::query_max_blocks(device, func, block_size)?;

        Ok(Self {
            _module: module,
            func,
            block_size,
            max_blocks,
            func_name: func_name.to_string(),
        })
    }

    /// Query maximum concurrent blocks for cooperative launch.
    fn query_max_blocks(
        device: &CudaDevice,
        func: cuda_sys::CUfunction,
        block_size: u32,
    ) -> Result<u32> {
        // Query max blocks per SM
        let mut max_blocks_per_sm: i32 = 0;

        unsafe {
            let result = cuda_sys::cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut max_blocks_per_sm,
                func,
                block_size as i32,
                0, // dynamic shared memory
            );

            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::BackendError(format!(
                    "cuOccupancyMaxActiveBlocksPerMultiprocessor failed: {:?}",
                    result
                )));
            }
        }

        // Get number of SMs
        let num_sms = device
            .inner()
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| RingKernelError::BackendError(format!("Failed to get SM count: {}", e)))?;

        Ok((max_blocks_per_sm as u32) * (num_sms as u32))
    }

    /// Get maximum concurrent blocks for cooperative launch.
    pub fn max_concurrent_blocks(&self) -> u32 {
        self.max_blocks
    }

    /// Get block size this kernel was configured with.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Get the function name.
    pub fn func_name(&self) -> &str {
        &self.func_name
    }

    /// Get the raw CUfunction handle.
    ///
    /// This is safe because we own the handle and the module is kept alive.
    pub fn raw_function(&self) -> cuda_sys::CUfunction {
        self.func
    }

    /// Launch the kernel cooperatively with raw parameters.
    ///
    /// This calls `cuLaunchCooperativeKernel` directly, enabling true grid-wide
    /// synchronization via `grid.sync()` in cooperative groups.
    ///
    /// # Safety
    ///
    /// - `kernel_params` must contain valid pointers matching the kernel signature
    /// - Each element must point to the corresponding kernel argument
    /// - Grid size must not exceed `max_concurrent_blocks()`
    pub unsafe fn launch_cooperative(
        &self,
        config: &CooperativeLaunchConfig,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        // Validate grid size
        let total_blocks = config.grid_dim.0 * config.grid_dim.1 * config.grid_dim.2;
        if total_blocks > self.max_blocks {
            return Err(RingKernelError::LaunchFailed(format!(
                "Grid size {} exceeds max concurrent blocks {} for cooperative launch",
                total_blocks, self.max_blocks
            )));
        }

        // Use default stream (null)
        let stream: cuda_sys::CUstream = ptr::null_mut();

        // Call true cooperative launch
        launch_cooperative_kernel(
            self.func,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            stream,
            kernel_params,
        )
        .map_err(|e| {
            RingKernelError::LaunchFailed(format!("Cooperative launch failed: {:?}", e))
        })?;

        Ok(())
    }

    /// Launch cooperatively with typed parameters.
    ///
    /// Helper method that builds the parameter array from individual arguments.
    ///
    /// # Safety
    ///
    /// - All parameter types must match the kernel signature exactly
    /// - Pointer parameters must be valid device pointers
    pub unsafe fn launch_cooperative_params<T: KernelParams>(
        &self,
        config: &CooperativeLaunchConfig,
        params: &mut T,
    ) -> Result<()> {
        let mut param_ptrs = params.as_param_ptrs();
        self.launch_cooperative(config, &mut param_ptrs)
    }

    /// Synchronize after launch.
    pub fn synchronize(&self) -> Result<()> {
        unsafe {
            let result = cuda_sys::cuCtxSynchronize();
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::BackendError(format!(
                    "cuCtxSynchronize failed: {:?}",
                    result
                )));
            }
        }
        Ok(())
    }
}

/// Trait for kernel parameters that can be converted to a pointer array.
///
/// Implement this for your kernel's parameter struct to use typed launch.
pub trait KernelParams {
    /// Convert parameters to an array of pointers for CUDA driver API.
    fn as_param_ptrs(&mut self) -> Vec<*mut c_void>;
}

/// Common kernel parameters for ring kernels.
///
/// Standard layout: (control_ptr, input_ptr, output_ptr, shared_ptr)
#[repr(C)]
pub struct RingKernelParams {
    /// Device pointer to control block.
    pub control_ptr: u64,
    /// Device pointer to input queue.
    pub input_ptr: u64,
    /// Device pointer to output queue.
    pub output_ptr: u64,
    /// Device pointer to shared state.
    pub shared_ptr: u64,
}

impl KernelParams for RingKernelParams {
    fn as_param_ptrs(&mut self) -> Vec<*mut c_void> {
        vec![
            &mut self.control_ptr as *mut u64 as *mut c_void,
            &mut self.input_ptr as *mut u64 as *mut c_void,
            &mut self.output_ptr as *mut u64 as *mut c_void,
            &mut self.shared_ptr as *mut u64 as *mut c_void,
        ]
    }
}

/// Extended kernel parameters for ring kernels with K2K messaging.
///
/// Layout: (control_ptr, input_ptr, output_ptr, shared_ptr, k2k_routes_ptr, k2k_inbox_ptr, k2k_outbox_ptr)
#[repr(C)]
pub struct RingKernelK2KParams {
    /// Device pointer to control block.
    pub control_ptr: u64,
    /// Device pointer to input queue.
    pub input_ptr: u64,
    /// Device pointer to output queue.
    pub output_ptr: u64,
    /// Device pointer to shared state.
    pub shared_ptr: u64,
    /// Device pointer to K2K routing table.
    pub k2k_routes_ptr: u64,
    /// Device pointer to K2K inbox buffer.
    pub k2k_inbox_ptr: u64,
    /// Device pointer to K2K outbox buffer.
    pub k2k_outbox_ptr: u64,
}

impl KernelParams for RingKernelK2KParams {
    fn as_param_ptrs(&mut self) -> Vec<*mut c_void> {
        vec![
            &mut self.control_ptr as *mut u64 as *mut c_void,
            &mut self.input_ptr as *mut u64 as *mut c_void,
            &mut self.output_ptr as *mut u64 as *mut c_void,
            &mut self.shared_ptr as *mut u64 as *mut c_void,
            &mut self.k2k_routes_ptr as *mut u64 as *mut c_void,
            &mut self.k2k_inbox_ptr as *mut u64 as *mut c_void,
            &mut self.k2k_outbox_ptr as *mut u64 as *mut c_void,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_kernel_params_layout() {
        let mut params = RingKernelParams {
            control_ptr: 0x1000,
            input_ptr: 0x2000,
            output_ptr: 0x3000,
            shared_ptr: 0x4000,
        };

        let ptrs = params.as_param_ptrs();
        assert_eq!(ptrs.len(), 4);

        // Verify pointers point to correct values
        unsafe {
            assert_eq!(*(ptrs[0] as *const u64), 0x1000);
            assert_eq!(*(ptrs[1] as *const u64), 0x2000);
            assert_eq!(*(ptrs[2] as *const u64), 0x3000);
            assert_eq!(*(ptrs[3] as *const u64), 0x4000);
        }
    }

    #[test]
    fn test_k2k_params_layout() {
        let mut params = RingKernelK2KParams {
            control_ptr: 0x1000,
            input_ptr: 0x2000,
            output_ptr: 0x3000,
            shared_ptr: 0x4000,
            k2k_routes_ptr: 0x5000,
            k2k_inbox_ptr: 0x6000,
            k2k_outbox_ptr: 0x7000,
        };

        let ptrs = params.as_param_ptrs();
        assert_eq!(ptrs.len(), 7);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_direct_ptx_module() {
        // Simple PTX that compiles (minimal valid kernel)
        let ptx = r#"
.version 7.0
.target sm_60
.address_size 64

.visible .entry test_kernel(
    .param .u64 param
)
{
    ret;
}
"#;

        let device = CudaDevice::new(0).expect("Failed to create device");
        let module = DirectPtxModule::load_ptx(&device, ptx).expect("Failed to load PTX");
        let func = module
            .get_function("test_kernel")
            .expect("Failed to get function");
        assert!(!func.is_null());
    }
}
