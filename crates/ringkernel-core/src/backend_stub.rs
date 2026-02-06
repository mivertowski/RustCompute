//! Macro for generating unavailable backend stubs.
//!
//! When a backend feature (e.g., `cuda`, `wgpu`, `metal`) is disabled,
//! the corresponding crate still needs to expose a stub `Runtime` type
//! that implements `RingKernelRuntime` but returns errors for all operations.
//!
//! This macro eliminates the ~50 lines of identical boilerplate per backend.

/// Generate a stub runtime for an unavailable backend.
///
/// Creates a struct that implements `RingKernelRuntime` with all methods
/// returning `BackendUnavailable` errors.
///
/// # Example
///
/// ```ignore
/// ringkernel_core::unavailable_backend!(CudaRuntime, Backend::Cuda, "CUDA");
/// ```
#[macro_export]
macro_rules! unavailable_backend {
    ($runtime:ident, $backend:expr, $name:expr) => {
        /// Stub runtime when the backend feature is disabled.
        pub struct $runtime;

        impl $runtime {
            /// Create fails when backend is not available.
            pub async fn new() -> $crate::error::Result<Self> {
                Err($crate::error::RingKernelError::BackendUnavailable(
                    concat!($name, " feature not enabled").to_string(),
                ))
            }
        }

        #[async_trait::async_trait]
        impl $crate::runtime::RingKernelRuntime for $runtime {
            fn backend(&self) -> $crate::runtime::Backend {
                $backend
            }

            fn is_backend_available(&self, _backend: $crate::runtime::Backend) -> bool {
                false
            }

            async fn launch(
                &self,
                _kernel_id: &str,
                _options: $crate::runtime::LaunchOptions,
            ) -> $crate::error::Result<$crate::runtime::KernelHandle> {
                Err($crate::error::RingKernelError::BackendUnavailable(
                    $name.to_string(),
                ))
            }

            fn get_kernel(
                &self,
                _kernel_id: &$crate::runtime::KernelId,
            ) -> Option<$crate::runtime::KernelHandle> {
                None
            }

            fn list_kernels(&self) -> Vec<$crate::runtime::KernelId> {
                vec![]
            }

            fn metrics(&self) -> $crate::runtime::RuntimeMetrics {
                $crate::runtime::RuntimeMetrics::default()
            }

            async fn shutdown(&self) -> $crate::error::Result<()> {
                Ok(())
            }
        }
    };
}
