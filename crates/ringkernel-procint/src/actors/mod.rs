//! Actor runtime for GPU kernel coordination.
//!
//! Provides pipeline orchestration for process intelligence kernels.

mod coordinator;
mod messages;
mod runtime;

pub use coordinator::*;
pub use messages::*;
pub use runtime::*;
