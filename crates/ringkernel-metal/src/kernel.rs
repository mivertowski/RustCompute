//! Metal kernel implementation.

#![cfg(all(target_os = "macos", feature = "metal"))]

use metal::{ComputeCommandEncoder, ComputePipelineState, MTLSize};
use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::runtime::{KernelId, KernelState, LaunchOptions};
use ringkernel_core::telemetry::TelemetryBuffer;

use crate::device::MetalDevice;
use crate::memory::MetalBuffer;

/// A Metal compute kernel.
pub struct MetalKernel {
    /// Kernel identifier.
    id: KernelId,
    /// Kernel numeric ID.
    kernel_id: u64,
    /// Launch options.
    options: LaunchOptions,
    /// Current state.
    state: KernelState,
    /// Compute pipeline.
    pipeline: Option<ComputePipelineState>,
    /// Control block buffer.
    control_block: Option<MetalBuffer>,
    /// Input queue buffer.
    input_queue: Option<MetalBuffer>,
    /// Output queue buffer.
    output_queue: Option<MetalBuffer>,
    /// Telemetry.
    telemetry: TelemetryBuffer,
}

impl MetalKernel {
    /// Create a new Metal kernel.
    pub fn new(
        id: &str,
        kernel_id: u64,
        _device: &MetalDevice,
        options: LaunchOptions,
    ) -> Result<Self> {
        Ok(Self {
            id: KernelId::new(id),
            kernel_id,
            options,
            state: KernelState::Created,
            pipeline: None,
            control_block: None,
            input_queue: None,
            output_queue: None,
            telemetry: TelemetryBuffer::new(),
        })
    }

    /// Get the kernel ID.
    pub fn id(&self) -> &KernelId {
        &self.id
    }

    /// Get the numeric kernel ID.
    pub fn kernel_id(&self) -> u64 {
        self.kernel_id
    }

    /// Get the current state.
    pub fn state(&self) -> KernelState {
        self.state
    }

    /// Activate the kernel.
    pub fn activate(&mut self) -> Result<()> {
        if self.state != KernelState::Created && self.state != KernelState::Inactive {
            return Err(RingKernelError::InvalidState {
                expected: "Created or Inactive".to_string(),
                actual: format!("{:?}", self.state),
            });
        }
        self.state = KernelState::Active;
        Ok(())
    }

    /// Deactivate the kernel.
    pub fn deactivate(&mut self) -> Result<()> {
        if self.state != KernelState::Active {
            return Err(RingKernelError::InvalidState {
                expected: "Active".to_string(),
                actual: format!("{:?}", self.state),
            });
        }
        self.state = KernelState::Inactive;
        Ok(())
    }

    /// Terminate the kernel.
    pub fn terminate(&mut self) -> Result<()> {
        self.state = KernelState::Terminated;
        Ok(())
    }

    /// Get telemetry.
    pub fn telemetry(&self) -> TelemetryBuffer {
        self.telemetry
    }

    /// Get launch options.
    pub fn options(&self) -> &LaunchOptions {
        &self.options
    }
}
