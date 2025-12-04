//! Core data structures for process intelligence.
//!
//! GPU-aligned structures for efficient kernel processing.

mod activity;
mod conformance;
mod dfg;
mod event;
mod partial_order;
mod patterns;
mod trace;

pub use activity::*;
pub use conformance::*;
pub use dfg::*;
pub use event::*;
pub use partial_order::*;
pub use patterns::*;
pub use trace::*;
