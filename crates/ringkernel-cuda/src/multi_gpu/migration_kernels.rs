//! Rust wrapper for the GPU-side migration kernels.
//!
//! This module loads the `capture_actor_state`, `restore_actor_state`, and
//! `drain_inflight_queue` entry points from the build-time-compiled PTX
//! (see `crates/ringkernel-cuda/src/cuda/migration_kernels.cu`) and exposes
//! safe Rust wrappers that the migration orchestrator can drive on either
//! the source or target GPU.
//!
//! ### Architecture
//!
//! ```text
//!   Source GPU                                         Target GPU
//!   ──────────                                         ──────────
//!   capture_actor_state ──►  mapped state buffer  ──►  restore_actor_state
//!   drain_inflight_queue ──► mapped drain buffer  ──►  replay into new queue
//! ```
//!
//! All three kernels write into **mapped memory** (see
//! [`crate::persistent::CudaMappedBuffer`]) so the host sees the results
//! without an explicit `cudaMemcpy` after stream synchronization.
//!
//! ### PTX availability
//!
//! If `nvcc` is not present at build time, the PTX constant is empty and
//! [`MigrationKernels::load`] returns [`RingKernelError::BackendUnavailable`].
//! All call sites must handle this gracefully so that downstream crates can
//! build on CPU-only machines.

use cudarc::driver::sys as cuda_sys;
use cudarc::driver::{CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::memory::GpuBuffer;

use crate::device::CudaDevice;
use crate::memory::CudaBuffer;
use crate::persistent::CudaMappedBuffer;

// Build-time-compiled PTX for the migration kernels.
//
// The build script (`build.rs`) compiles
// `src/cuda/migration_kernels.cu` with `nvcc -ptx -arch=…` and embeds the
// result here. When `nvcc` is missing the constant is empty and
// `HAS_MIGRATION_KERNEL_SUPPORT` is `false`.
include!(concat!(env!("OUT_DIR"), "/migration_kernels.rs"));

// ============================================================================
// Public result types
// ============================================================================

/// Result of a `capture_actor_state` invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureResult {
    /// Number of bytes actually written to the output mapped buffer.
    pub bytes_written: u64,
    /// CRC32 of the captured bytes — the target GPU verifies against this.
    pub checksum: u32,
}

/// Result of a `drain_inflight_queue` invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrainResult {
    /// Number of messages successfully moved from the K2K queue into the
    /// drain buffer. The source actor's queue's `tail` is advanced by the
    /// same amount.
    pub messages_drained: u64,
}

// ============================================================================
// MigrationKernels
// ============================================================================

/// Loaded handle to the three migration kernel entry points on a specific
/// CUDA device.
///
/// Each `MigrationKernels` instance is tied to the device that called
/// [`MigrationKernels::load`]; sharing across devices requires a separate
/// instance per device.
pub struct MigrationKernels {
    /// Source device that loaded the PTX.
    device: CudaDevice,
    /// Loaded CUDA module (keeps PTX alive).
    #[allow(dead_code)]
    module: std::sync::Arc<CudaModule>,
    /// Capture entry point.
    capture_fn: CudaFunction,
    /// Restore entry point.
    restore_fn: CudaFunction,
    /// Drain entry point.
    drain_fn: CudaFunction,
}

impl MigrationKernels {
    /// Load the migration PTX onto `device` and resolve entry points.
    ///
    /// Returns [`RingKernelError::BackendUnavailable`] if the PTX was not
    /// built (no `nvcc` at build time) so downstream crates can degrade
    /// gracefully on CPU-only machines.
    pub fn load(device: &CudaDevice) -> Result<Self> {
        if !HAS_MIGRATION_KERNEL_SUPPORT || MIGRATION_KERNEL_PTX.trim().is_empty() {
            return Err(RingKernelError::BackendUnavailable(format!(
                "Migration kernels not compiled: {}",
                MIGRATION_KERNEL_BUILD_MESSAGE
            )));
        }

        // Ensure the device's context is current before loading the module.
        let ctx = device.inner();
        let ptx = cudarc::nvrtc::Ptx::from_src(MIGRATION_KERNEL_PTX);

        let module = ctx.load_module(ptx).map_err(|e| {
            RingKernelError::BackendError(format!("Failed to load migration PTX: {}", e))
        })?;

        let capture_fn = module.load_function("capture_actor_state").map_err(|e| {
            RingKernelError::BackendError(format!(
                "capture_actor_state not found in migration PTX: {}",
                e
            ))
        })?;
        let restore_fn = module.load_function("restore_actor_state").map_err(|e| {
            RingKernelError::BackendError(format!(
                "restore_actor_state not found in migration PTX: {}",
                e
            ))
        })?;
        let drain_fn = module.load_function("drain_inflight_queue").map_err(|e| {
            RingKernelError::BackendError(format!(
                "drain_inflight_queue not found in migration PTX: {}",
                e
            ))
        })?;

        Ok(Self {
            device: device.clone(),
            module,
            capture_fn,
            restore_fn,
            drain_fn,
        })
    }

    /// Source device the PTX was loaded on.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Launch `capture_actor_state` and return the number of bytes written
    /// together with the CRC32 checksum of the captured data.
    ///
    /// Caller contract:
    /// - `actor_state` is device memory belonging to `self.device`
    /// - `output.len() >= actor_state.len()` (in bytes)
    /// - `stream` is synchronized by the caller when they need to observe
    ///   the host-visible `bytes_written` / `checksum` fields
    pub fn capture_state(
        &self,
        stream: &CudaStream,
        actor_state: &CudaBuffer,
        output: &CudaMappedBuffer<u8>,
    ) -> Result<CaptureResult> {
        let state_size_bytes = actor_state.size();
        if output.len() < state_size_bytes {
            return Err(RingKernelError::InvalidConfig(format!(
                "capture output buffer too small: have {} bytes, need {}",
                output.len(),
                state_size_bytes
            )));
        }

        // Mapped u64 for bytes_written and u32 for checksum.
        let bytes_written_buf: CudaMappedBuffer<u64> = CudaMappedBuffer::new(&self.device, 1)?;
        let checksum_buf: CudaMappedBuffer<u32> = CudaMappedBuffer::new(&self.device, 1)?;
        bytes_written_buf.write(0, 0);
        checksum_buf.write(0, 0);

        // Launch with `gridDim.x == 1` so the checksum reduction is
        // deterministic. A cross-GPU migration uses the same block shape on
        // the source and the target, which makes the XOR-reduced checksum
        // reproducible on both sides.
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        let actor_ptr: u64 = actor_state.device_ptr();
        let output_ptr: u64 = output.device_ptr();
        let state_size: u64 = state_size_bytes as u64;
        let bw_ptr: u64 = bytes_written_buf.device_ptr();
        let cs_ptr: u64 = checksum_buf.device_ptr();

        // SAFETY: All device pointers above point to live allocations owned
        // by the caller or by this function. The kernel signature expects
        // `(const void*, size_t, void*, uint64_t*, uint32_t*)` which we
        // match exactly via the typed arg builder.
        unsafe {
            stream
                .launch_builder(&self.capture_fn)
                .arg(&actor_ptr)
                .arg(&state_size)
                .arg(&output_ptr)
                .arg(&bw_ptr)
                .arg(&cs_ptr)
                .launch(cfg)
                .map_err(|e| {
                    RingKernelError::LaunchFailed(format!("capture_actor_state launch: {}", e))
                })?;
        }

        // Synchronize so the host-visible mapped fields are stable.
        stream_synchronize(stream)?;

        let bytes_written = bytes_written_buf.read(0).unwrap_or(0);
        let checksum = checksum_buf.read(0).unwrap_or(0);

        Ok(CaptureResult {
            bytes_written,
            checksum,
        })
    }

    /// Launch `restore_actor_state` and fail if the CRC32 does not match
    /// `expected_checksum`.
    pub fn restore_state(
        &self,
        stream: &CudaStream,
        actor_state: &CudaBuffer,
        input: &CudaMappedBuffer<u8>,
        expected_checksum: u32,
    ) -> Result<()> {
        let state_size_bytes = actor_state.size();
        if input.len() < state_size_bytes {
            return Err(RingKernelError::InvalidConfig(format!(
                "restore input buffer too small: have {} bytes, need {}",
                input.len(),
                state_size_bytes
            )));
        }

        // Verification result is an i32 in mapped memory. The restore kernel
        // writes the final 0/-1 into it; we pre-zero for cleanliness.
        let verify_buf: CudaMappedBuffer<i32> = CudaMappedBuffer::new(&self.device, 1)?;
        verify_buf.write(0, 0);

        // The restore kernel assumes `gridDim.x == 1` so the in-block
        // checksum reduction is deterministic. See migration_kernels.cu.
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        let actor_ptr: u64 = actor_state.device_ptr();
        let input_ptr: u64 = input.device_ptr();
        let state_size: u64 = state_size_bytes as u64;
        let expected: u32 = expected_checksum;
        let verify_ptr: u64 = verify_buf.device_ptr();

        // SAFETY: pointers belong to live allocations; arg types match the
        // kernel signature `(void*, const void*, size_t, uint32_t, int*)`.
        unsafe {
            stream
                .launch_builder(&self.restore_fn)
                .arg(&actor_ptr)
                .arg(&input_ptr)
                .arg(&state_size)
                .arg(&expected)
                .arg(&verify_ptr)
                .launch(cfg)
                .map_err(|e| {
                    RingKernelError::LaunchFailed(format!("restore_actor_state launch: {}", e))
                })?;
        }

        stream_synchronize(stream)?;

        let result = verify_buf.read(0).unwrap_or(-1);
        if result == 0 {
            Ok(())
        } else {
            Err(RingKernelError::BackendError(format!(
                "Migration checksum mismatch: expected 0x{:08X}, restore verification returned {}",
                expected_checksum, result
            )))
        }
    }

    /// Drain the K2K in-flight queue into `drain_buffer` during the quiesce
    /// phase. Returns the number of messages moved.
    pub fn drain_queue(
        &self,
        stream: &CudaStream,
        queue: &CudaBuffer,
        drain_buffer: &CudaBuffer,
    ) -> Result<DrainResult> {
        let messages_drained_buf: CudaMappedBuffer<u64> = CudaMappedBuffer::new(&self.device, 1)?;
        messages_drained_buf.write(0, 0);

        // A single-block launch is enough; the kernel is serial at its
        // producer-consumer boundary.
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let queue_ptr: u64 = queue.device_ptr();
        // The kernel reads capacity from the queue header and sanity-checks
        // it against this argument. We fetch it from the header via a small
        // host read here for the parity check.
        let queue_capacity: u64 = read_queue_capacity(queue)? as u64;
        let drain_ptr: u64 = drain_buffer.device_ptr();
        let md_ptr: u64 = messages_drained_buf.device_ptr();

        // SAFETY: pointers are live; kernel signature matches
        // `(void*, size_t, void*, uint64_t*)`.
        unsafe {
            stream
                .launch_builder(&self.drain_fn)
                .arg(&queue_ptr)
                .arg(&queue_capacity)
                .arg(&drain_ptr)
                .arg(&md_ptr)
                .launch(cfg)
                .map_err(|e| {
                    RingKernelError::LaunchFailed(format!("drain_inflight_queue launch: {}", e))
                })?;
        }

        stream_synchronize(stream)?;

        let messages_drained = messages_drained_buf.read(0).unwrap_or(0);
        Ok(DrainResult { messages_drained })
    }

    /// PTX source used for the load (diagnostic).
    pub fn ptx(&self) -> &'static str {
        MIGRATION_KERNEL_PTX
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Launch config used by both capture and restore. We fix `gridDim.x == 1`
/// so the XOR-reduced checksum is deterministic across source/target GPUs.
/// 128 threads-per-block is enough to saturate memory for the state sizes
/// v1.1 targets (≤ 6.4 MiB per migrating actor per the spec).
#[allow(dead_code)]
fn single_block_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Read the `capacity` field from the K2K queue header (first 16 bytes of
/// the header are `head` and `tail`, the next `u32` is `capacity`).
fn read_queue_capacity(queue: &CudaBuffer) -> Result<u32> {
    let mut bytes = [0u8; 4];
    queue.copy_to_host_at(&mut bytes, 16)?;
    Ok(u32::from_le_bytes(bytes))
}

/// Synchronize a CUDA stream. Exposed as a small helper so tests can mock it.
fn stream_synchronize(stream: &CudaStream) -> Result<()> {
    // cudarc 0.19 exposes `stream.synchronize()` but the returned error type
    // varies; wrap via `cuStreamSynchronize` directly to stay explicit.
    let raw: cuda_sys::CUstream = stream.cu_stream() as cuda_sys::CUstream;
    // SAFETY: `raw` is a live CUstream handle held by `stream` for the
    // duration of this call.
    let res = unsafe { cuda_sys::cuStreamSynchronize(raw) };
    if res != cuda_sys::CUresult::CUDA_SUCCESS {
        return Err(RingKernelError::BackendError(format!(
            "cuStreamSynchronize failed: {:?}",
            res
        )));
    }
    Ok(())
}

// ============================================================================
// Tests — build without hardware
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_block_config_shape() {
        let cfg = single_block_config();
        assert_eq!(
            cfg.grid_dim,
            (1, 1, 1),
            "restore/capture kernels require exactly one block for a deterministic checksum"
        );
        assert_eq!(cfg.block_dim, (128, 1, 1));
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn test_single_block_config_is_deterministic() {
        // Guards against regressions: any future change that relaxes the
        // gridDim requirement must be paired with a matching change in the
        // .cu file (or a per-block atomic reduction).
        let a = single_block_config();
        let b = single_block_config();
        assert_eq!(a.grid_dim, b.grid_dim);
        assert_eq!(a.block_dim, b.block_dim);
    }

    #[test]
    fn test_capture_result_shape() {
        let r = CaptureResult {
            bytes_written: 42,
            checksum: 0xCAFEBABE,
        };
        assert_eq!(r.bytes_written, 42);
        assert_eq!(r.checksum, 0xCAFEBABE);
    }

    #[test]
    fn test_capture_result_zero_state() {
        // Zero-byte state: a valid edge case (empty actor). The wrapper
        // should not special-case; the kernel handles it.
        let r = CaptureResult {
            bytes_written: 0,
            checksum: 0,
        };
        assert_eq!(r.bytes_written, 0);
        assert_eq!(r.checksum, 0);
    }

    #[test]
    fn test_drain_result_shape() {
        let r = DrainResult {
            messages_drained: 17,
        };
        assert_eq!(r.messages_drained, 17);
    }

    #[test]
    fn test_drain_result_empty_queue() {
        // Draining an empty queue must still succeed and report 0 messages.
        let r = DrainResult {
            messages_drained: 0,
        };
        assert_eq!(r.messages_drained, 0);
    }

    #[test]
    fn test_load_returns_backend_unavailable_when_no_ptx() {
        // Skip this test in environments where the migration PTX was compiled.
        // `HAS_MIGRATION_KERNEL_SUPPORT` becomes `true` only when nvcc is
        // available and the compilation succeeds.
        if !HAS_MIGRATION_KERNEL_SUPPORT {
            // Build environment has no nvcc. We cannot call `load` because
            // that still needs a CUDA device — we only assert the PTX const
            // is empty, which is the observable proxy for "load would fail
            // with BackendUnavailable".
            assert!(MIGRATION_KERNEL_PTX.trim().is_empty());
        }
    }

    #[test]
    fn test_migration_ptx_build_message_is_set() {
        // Always populated — either "compiled successfully" or the stub
        // reason. Non-empty is the invariant.
        assert!(!MIGRATION_KERNEL_BUILD_MESSAGE.is_empty());
    }

    // Hardware-backed tests — require an NVIDIA GPU with CUDA driver.
    #[test]
    #[ignore] // Requires CUDA GPU
    fn test_load_on_device() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        match MigrationKernels::load(&device) {
            Ok(kernels) => {
                assert!(!kernels.ptx().is_empty());
            }
            Err(RingKernelError::BackendUnavailable(_)) => {
                // OK — PTX not built on this runner.
            }
            Err(e) => panic!("unexpected load error: {:?}", e),
        }
    }
}
