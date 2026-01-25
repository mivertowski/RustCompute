//! Resource guard module for preventing system overload.
//!
//! This module provides safeguards to prevent out-of-memory conditions
//! during large-scale operations.
//!
//! # Example
//!
//! ```
//! use ringkernel_core::resource::{ResourceGuard, MemoryEstimate};
//!
//! let guard = ResourceGuard::new();
//! let estimate = MemoryEstimate::new()
//!     .with_primary(1024 * 1024)  // 1 MB
//!     .with_auxiliary(512 * 1024); // 512 KB
//!
//! if !guard.can_allocate(estimate.total_bytes()) {
//!     panic!("Insufficient memory");
//! }
//!
//! // Or use the trait-based estimator
//! struct MyWorkload { elements: usize }
//! impl ringkernel_core::resource::MemoryEstimator for MyWorkload {
//!     fn estimate(&self) -> MemoryEstimate {
//!         MemoryEstimate::new().with_primary((self.elements * 64) as u64)
//!     }
//!     fn name(&self) -> &str { "MyWorkload" }
//! }
//! ```

mod error;
mod estimate;
mod guard;
mod system;

pub use error::{ResourceError, ResourceResult};
pub use estimate::{LinearEstimator, MemoryEstimate, MemoryEstimator};
pub use guard::{global_guard, ReservationGuard, ResourceGuard};
pub use system::{get_available_memory, get_total_memory};

/// Default maximum memory usage (4 GB).
pub const DEFAULT_MAX_MEMORY_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Safety margin for system memory (leave 1 GB free).
pub const SYSTEM_MEMORY_MARGIN: u64 = 1024 * 1024 * 1024;
