//! GPU Actor Lifecycle Kernel — embedded PTX and constants.
//!
//! This module provides the pre-compiled actor lifecycle kernel that
//! demonstrates the full GPU actor model: supervisor, dynamic create/destroy,
//! heartbeat monitoring, restart, and per-actor metrics.

// Include build-time generated lifecycle kernel PTX
include!(concat!(env!("OUT_DIR"), "/actor_lifecycle_kernel.rs"));

/// Kernel entry point name.
pub const KERNEL_NAME: &str = "actor_lifecycle_kernel";

/// Check if the lifecycle kernel was compiled at build time.
pub fn has_lifecycle_kernel() -> bool {
    HAS_LIFECYCLE_KERNEL
}

/// Get the pre-compiled lifecycle kernel PTX.
pub fn lifecycle_kernel_ptx() -> &'static str {
    LIFECYCLE_KERNEL_PTX
}
