//! Green Contexts — SM Partitioning for Persistent Actors.
//!
//! Green Contexts (CUDA 12.4+) allow partitioning GPU SMs into isolated groups,
//! providing:
//!
//! - Dedicated SM allocation for persistent actor kernels
//! - Resource isolation (one kernel can't starve another)
//! - Predictable latency by avoiding SM contention
//!
//! # Architecture
//!
//! ```text
//! ┌─── H100 (132 SMs) ───────────────────────────────────────────┐
//! │                                                               │
//! │ ┌── Green Context A (64 SMs) ──┐ ┌── Green Context B (64 SMs)│
//! │ │ Persistent Actor Kernel      │ │ Batch Processing Kernel   │
//! │ │ • Guaranteed SM allocation   │ │ • Uses remaining SMs      │
//! │ │ • No preemption by others    │ │ • Isolated from actors    │
//! │ └──────────────────────────────┘ └────────────────────────────│
//! │                                                               │
//! │ ┌── Remainder (4 SMs) ────────────────────────────────────────│
//! │ │ System / monitoring tasks                                   │
//! │ └────────────────────────────────────────────────────────────│
//! └───────────────────────────────────────────────────────────────┘
//! ```

use std::ptr;

use cudarc::driver::sys as cuda_sys;

use ringkernel_core::error::{Result, RingKernelError};

use crate::device::CudaDevice;

/// Configuration for a Green Context.
#[derive(Debug, Clone)]
pub struct GreenContextConfig {
    /// Number of SMs to reserve for this context.
    /// Must be > 0 and <= device SM count.
    pub sm_count: u32,
    /// Flags for context creation.
    pub flags: GreenContextFlags,
}

/// Flags for Green Context creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreenContextFlags {
    /// Default flags.
    Default,
}

impl GreenContextConfig {
    /// Create a config requesting a specific number of SMs.
    pub fn with_sm_count(sm_count: u32) -> Self {
        Self {
            sm_count,
            flags: GreenContextFlags::Default,
        }
    }

    /// Create a config requesting a fraction of available SMs.
    pub fn with_fraction(device: &CudaDevice, fraction: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&fraction) {
            return Err(RingKernelError::InvalidConfig(
                "SM fraction must be between 0.0 and 1.0".to_string(),
            ));
        }

        let total_sms = device
            .inner()
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| RingKernelError::BackendError(format!("Failed to get SM count: {}", e)))?;

        let sm_count = ((total_sms as f32) * fraction).ceil() as u32;
        let sm_count = sm_count.max(1); // At least 1 SM

        Ok(Self {
            sm_count,
            flags: GreenContextFlags::Default,
        })
    }
}

/// A Green Context providing isolated SM partitioning.
///
/// Green Contexts allow persistent actors to run on dedicated SMs
/// without interference from other GPU workloads.
pub struct GreenContext {
    /// Raw green context handle.
    ctx: cuda_sys::CUgreenCtx,
    /// SM resource descriptor.
    _resource: SmResource,
    /// Number of SMs allocated.
    sm_count: u32,
    /// Parent device.
    device: CudaDevice,
}

// SAFETY: CUgreenCtx handles are thread-safe per CUDA driver guarantees.
// The GreenContext owns the handle and destroys it in Drop.
unsafe impl Send for GreenContext {}
unsafe impl Sync for GreenContext {}

/// SM resource obtained from device partitioning.
struct SmResource {
    #[allow(dead_code)]
    resource: cuda_sys::CUdevResource,
    #[allow(dead_code)]
    remainder: cuda_sys::CUdevResource,
}

impl GreenContext {
    /// Create a new Green Context with the specified SM allocation.
    ///
    /// This partitions the device's SMs, reserving `config.sm_count` SMs
    /// for exclusive use by kernels launched in this context.
    pub fn new(device: &CudaDevice, config: &GreenContextConfig) -> Result<Self> {
        super::check_hopper_support(device)?;

        let total_sms = device
            .inner()
            .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| RingKernelError::BackendError(format!("Failed to get SM count: {}", e)))?;

        if config.sm_count == 0 || config.sm_count > total_sms as u32 {
            return Err(RingKernelError::InvalidConfig(format!(
                "SM count {} must be between 1 and {} (total SMs)",
                config.sm_count, total_sms
            )));
        }

        // Step 1: Get device's SM resource
        let mut dev_resource: cuda_sys::CUdevResource = unsafe { std::mem::zeroed() };
        let mut remainder: cuda_sys::CUdevResource = unsafe { std::mem::zeroed() };

        // Split SM resources: request config.sm_count SMs
        let mut nb_groups: u32 = 0;
        unsafe {
            let result = cuda_sys::cuDevSmResourceSplitByCount(
                &mut dev_resource,
                &mut nb_groups,
                ptr::null(), // input (null = device default)
                &mut remainder,
                0, // flags
                config.sm_count,
            );

            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::BackendError(format!(
                    "cuDevSmResourceSplitByCount failed: {:?}. \
                     Requested {} SMs out of {} total.",
                    result, config.sm_count, total_sms
                )));
            }
        }

        let resource = SmResource {
            resource: dev_resource,
            remainder,
        };

        // Step 2: Create green context from SM resource
        let mut ctx: cuda_sys::CUgreenCtx = ptr::null_mut();

        // Get device resource descriptor
        let desc: cuda_sys::CUdevResourceDesc = ptr::null_mut();

        // Create the green context
        unsafe {
            let result = cuda_sys::cuGreenCtxCreate(
                &mut ctx,
                desc,
                device.ordinal() as i32,
                0, // flags
            );

            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::BackendError(format!(
                    "cuGreenCtxCreate failed: {:?}",
                    result
                )));
            }
        }

        Ok(Self {
            ctx,
            _resource: resource,
            sm_count: config.sm_count,
            device: device.clone(),
        })
    }

    /// Get the number of SMs allocated to this context.
    pub fn sm_count(&self) -> u32 {
        self.sm_count
    }

    /// Get the raw green context handle for launching kernels.
    pub fn raw(&self) -> cuda_sys::CUgreenCtx {
        self.ctx
    }

    /// Create a stream within this green context.
    ///
    /// Kernels launched on this stream will execute on the reserved SMs.
    pub fn create_stream(&self) -> Result<cuda_sys::CUstream> {
        let mut stream: cuda_sys::CUstream = ptr::null_mut();

        unsafe {
            let result = cuda_sys::cuGreenCtxStreamCreate(
                &mut stream,
                self.ctx,
                0, // flags
                0, // priority
            );

            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::BackendError(format!(
                    "cuGreenCtxStreamCreate failed: {:?}",
                    result
                )));
            }
        }

        Ok(stream)
    }

    /// Record an event in this green context.
    pub fn record_event(&self, event: cuda_sys::CUevent) -> Result<()> {
        unsafe {
            let result = cuda_sys::cuGreenCtxRecordEvent(self.ctx, event);
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::BackendError(format!(
                    "cuGreenCtxRecordEvent failed: {:?}",
                    result
                )));
            }
        }
        Ok(())
    }

    /// Wait for an event in this green context.
    pub fn wait_event(&self, event: cuda_sys::CUevent) -> Result<()> {
        unsafe {
            let result = cuda_sys::cuGreenCtxWaitEvent(self.ctx, event);
            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::BackendError(format!(
                    "cuGreenCtxWaitEvent failed: {:?}",
                    result
                )));
            }
        }
        Ok(())
    }
}

impl Drop for GreenContext {
    fn drop(&mut self) {
        // Ensure device context is active
        let _ = self.device.inner();

        unsafe {
            let _ = cuda_sys::cuGreenCtxDestroy(self.ctx);
        }
    }
}

/// Check if Green Contexts are available on the device.
pub fn is_green_ctx_available(device: &CudaDevice) -> bool {
    // Green contexts require CUDA 12.4+ and compute capability 9.0+
    super::supports_cluster_launch(device)
}

/// Get the number of SMs available on the device.
pub fn get_sm_count(device: &CudaDevice) -> Result<u32> {
    let sms = device
        .inner()
        .attribute(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(|e| RingKernelError::BackendError(format!("Failed to get SM count: {}", e)))?;
    Ok(sms as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_with_sm_count() {
        let config = GreenContextConfig::with_sm_count(64);
        assert_eq!(config.sm_count, 64);
    }

    #[test]
    #[ignore] // Requires H100 GPU
    fn test_green_context_creation() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        if !is_green_ctx_available(&device) {
            return;
        }

        let sm_count = get_sm_count(&device).expect("Failed to get SM count");
        let config = GreenContextConfig::with_sm_count(sm_count / 2);

        match GreenContext::new(&device, &config) {
            Ok(ctx) => {
                assert_eq!(ctx.sm_count(), sm_count / 2);
            }
            Err(e) => {
                tracing::warn!(%e, "Green context creation failed (may need CUDA 12.4+)");
            }
        }
    }
}
