//! Error types for the process intelligence crate.

use thiserror::Error;

/// Errors that can occur in the process intelligence crate.
#[derive(Debug, Error)]
pub enum ProcIntError {
    /// CUDA feature is not compiled in.
    #[error("CUDA feature not enabled. Build with --features cuda")]
    CudaNotEnabled,

    /// No CUDA device is available.
    #[error("no CUDA device available")]
    NoCudaDevice,

    /// A required kernel was not loaded before execution.
    #[error("kernel not loaded: {0}")]
    KernelNotLoaded(String),

    /// CUDA kernel code generation (transpilation) failed.
    #[error("codegen failed for {kernel}: {reason}")]
    Codegen {
        /// Which kernel failed to transpile.
        kernel: &'static str,
        /// The underlying transpiler error message.
        reason: String,
    },

    /// NVRTC compilation of CUDA source to PTX failed.
    #[error("NVRTC compilation failed for {kernel}: {reason}")]
    NvrtcCompilation {
        /// Kernel that failed to compile.
        kernel: String,
        /// The underlying compiler error message.
        reason: String,
    },

    /// Loading a compiled PTX module onto the device failed.
    #[error("failed to load PTX for {kernel} (func: {func}): {reason}")]
    PtxLoad {
        /// Kernel module name.
        kernel: String,
        /// Entry-point function name.
        func: String,
        /// The underlying driver error message.
        reason: String,
    },

    /// A host-to-device memory transfer failed.
    #[error("host-to-device transfer failed ({buffer}): {reason}")]
    HostToDevice {
        /// Which buffer was being transferred.
        buffer: &'static str,
        /// The underlying driver error message.
        reason: String,
    },

    /// A device-to-host memory transfer failed.
    #[error("device-to-host transfer failed ({buffer}): {reason}")]
    DeviceToHost {
        /// Which buffer was being transferred.
        buffer: &'static str,
        /// The underlying driver error message.
        reason: String,
    },

    /// A GPU kernel launch failed.
    #[error("kernel launch failed ({kernel}): {reason}")]
    KernelLaunch {
        /// Which kernel failed to launch.
        kernel: &'static str,
        /// The underlying driver error message.
        reason: String,
    },

    /// GPU device synchronization failed.
    #[error("device synchronization failed: {0}")]
    DeviceSync(String),
}
