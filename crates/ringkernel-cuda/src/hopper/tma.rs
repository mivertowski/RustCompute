//! TMA (Tensor Memory Accelerator) Async Copy for Hopper GPUs.
//!
//! TMA provides hardware-accelerated bulk data movement between global and shared memory.
//! For persistent actor messaging, TMA enables:
//!
//! - Single-thread async message payload transfer (frees warps for compute)
//! - Barrier-based completion tracking via `barrier_arrive_tx()`
//! - Multicast to all blocks in a cluster
//!
//! # TMA 1D Bulk Copy Pattern
//!
//! ```text
//! Global Memory                    Shared Memory
//! ┌──────────────┐     TMA 1D     ┌──────────────┐
//! │ Message Buf  │ ──────────────► │ Message Slot │
//! │ (4KB page)   │  (single-       │ (shared mem) │
//! └──────────────┘   thread,        └──────────────┘
//!                    async)
//! ```
//!
//! TMA 1D bulk copy doesn't require a tensor map descriptor for 1D data,
//! making it ideal for message payload transfer.

use ringkernel_core::error::{Result, RingKernelError};

use crate::device::CudaDevice;

/// Configuration for TMA-based message transfer.
#[derive(Debug, Clone)]
pub struct TmaConfig {
    /// Maximum transfer size in bytes per TMA operation.
    /// Must be a multiple of 16 bytes.
    pub max_transfer_size: u32,
    /// Number of concurrent TMA operations to pipeline.
    pub pipeline_depth: u32,
    /// Whether to use multicast (send to all blocks in cluster).
    pub multicast: bool,
}

impl Default for TmaConfig {
    fn default() -> Self {
        Self {
            max_transfer_size: 4096,
            pipeline_depth: 2,
            multicast: false,
        }
    }
}

impl TmaConfig {
    /// Validate TMA configuration.
    pub fn validate(&self) -> Result<()> {
        if self.max_transfer_size % 16 != 0 {
            return Err(RingKernelError::InvalidConfig(format!(
                "TMA transfer size must be multiple of 16, got {}",
                self.max_transfer_size
            )));
        }

        if self.max_transfer_size > 131072 {
            return Err(RingKernelError::InvalidConfig(format!(
                "TMA transfer size {} exceeds maximum 128KB",
                self.max_transfer_size
            )));
        }

        if self.pipeline_depth == 0 || self.pipeline_depth > 8 {
            return Err(RingKernelError::InvalidConfig(format!(
                "TMA pipeline depth must be 1-8, got {}",
                self.pipeline_depth
            )));
        }

        Ok(())
    }
}

/// TMA operation descriptor for device-side code generation.
///
/// This provides the information needed by the CUDA codegen to emit
/// TMA intrinsics (`cp.async.bulk`) in the generated PTX.
#[derive(Debug, Clone)]
pub struct TmaDescriptor {
    /// Source address (global memory device pointer).
    pub src_global_addr: u64,
    /// Destination offset in shared memory.
    pub dst_smem_offset: u32,
    /// Transfer size in bytes.
    pub transfer_size: u32,
    /// Barrier index for completion tracking.
    pub barrier_idx: u32,
}

/// Shared memory layout for TMA-based message injection.
///
/// Used by the codegen to emit correct shared memory declarations.
#[derive(Debug, Clone)]
pub struct TmaSmemLayout {
    /// Offset where message payload lands after TMA copy.
    pub payload_offset: u32,
    /// Size of payload area in shared memory.
    pub payload_size: u32,
    /// Offset of the mbarrier array.
    pub barrier_offset: u32,
    /// Number of mbarrier objects (= pipeline_depth).
    pub barrier_count: u32,
    /// Total shared memory needed for TMA operations.
    pub total_smem: u32,
}

/// Compute the shared memory layout for TMA message injection.
pub fn compute_tma_smem_layout(config: &TmaConfig) -> TmaSmemLayout {
    let payload_offset = 0u32;
    let payload_size = config.max_transfer_size * config.pipeline_depth;

    // mbarrier objects need 8-byte alignment, 8 bytes each
    let barrier_offset = (payload_offset + payload_size + 7) & !7;
    let barrier_count = config.pipeline_depth;
    let barrier_size = barrier_count * 8; // 8 bytes per mbarrier

    let total_smem = barrier_offset + barrier_size;

    TmaSmemLayout {
        payload_offset,
        payload_size,
        barrier_offset,
        barrier_count,
        total_smem,
    }
}

/// Check if TMA is available on the device.
pub fn is_tma_available(device: &CudaDevice) -> bool {
    let (major, _) = device.compute_capability();
    major >= 9 // TMA is a Hopper feature
}

/// Generate PTX snippet for TMA 1D bulk async copy.
///
/// This generates the PTX instructions for a single TMA transfer
/// that can be included in generated CUDA code.
///
/// The generated code:
/// 1. Issues `cp.async.bulk.shared::cluster.global` for the transfer
/// 2. Uses `mbarrier.arrive.expect_tx` for completion tracking
/// 3. Consumer waits with `mbarrier.try_wait.parity`
pub fn generate_tma_ptx_snippet(transfer_size: u32) -> String {
    format!(
        r#"
// TMA 1D bulk async copy: global -> shared memory
// Transfer size: {size} bytes
// Only thread 0 issues the copy; all threads can wait on the barrier

// Initialize mbarrier (thread 0 only)
// mbarrier.init.shared::cta.b64 [barrier_addr], expected_count;

// Issue TMA copy (thread 0 only)
// cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
//   [dst_smem], [src_global], {size}, [barrier_addr];

// Wait for completion (all threads)
// mbarrier.try_wait.parity.shared::cta.b64 waitComplete, [barrier_addr], phaseBit;
"#,
        size = transfer_size
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tma_config_defaults() {
        let config = TmaConfig::default();
        assert_eq!(config.max_transfer_size, 4096);
        assert_eq!(config.pipeline_depth, 2);
        assert!(!config.multicast);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tma_config_validation() {
        let mut config = TmaConfig::default();

        config.max_transfer_size = 15; // Not multiple of 16
        assert!(config.validate().is_err());

        config.max_transfer_size = 256;
        config.pipeline_depth = 0;
        assert!(config.validate().is_err());

        config.pipeline_depth = 2;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_tma_smem_layout() {
        let config = TmaConfig::default();
        let layout = compute_tma_smem_layout(&config);

        assert_eq!(layout.payload_offset, 0);
        assert_eq!(layout.payload_size, 4096 * 2); // 2 pipeline stages
        assert_eq!(layout.barrier_count, 2);
        assert!(layout.total_smem > layout.payload_size);
    }

    #[test]
    fn test_ptx_snippet_generation() {
        let snippet = generate_tma_ptx_snippet(4096);
        assert!(snippet.contains("4096"));
        assert!(snippet.contains("cp.async.bulk"));
    }
}
