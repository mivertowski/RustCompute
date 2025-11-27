//! # RingKernel Core
//!
//! Core traits and types for the RingKernel GPU-native persistent actor system.
//!
//! This crate provides the foundational abstractions for building GPU-accelerated
//! actor systems with persistent kernels, lock-free message passing, and hybrid
//! logical clocks for temporal ordering.
//!
//! ## Core Abstractions
//!
//! - [`RingMessage`] - Trait for messages between kernels
//! - [`MessageQueue`] - Lock-free ring buffer for message passing
//! - [`RingKernelRuntime`] - Backend-agnostic runtime management
//! - [`RingContext`] - GPU intrinsics facade for kernel handlers
//! - [`HlcTimestamp`] - Hybrid Logical Clock for causal ordering
//!
//! ## Example
//!
//! ```ignore
//! use ringkernel_core::prelude::*;
//!
//! #[derive(RingMessage)]
//! struct MyMessage {
//!     #[message(id)]
//!     id: MessageId,
//!     payload: Vec<f32>,
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod context;
pub mod control;
pub mod error;
pub mod hlc;
pub mod memory;
pub mod message;
pub mod queue;
pub mod runtime;
pub mod telemetry;
pub mod types;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::context::*;
    pub use crate::control::*;
    pub use crate::error::*;
    pub use crate::hlc::*;
    pub use crate::memory::*;
    pub use crate::message::*;
    pub use crate::queue::*;
    pub use crate::runtime::*;
    pub use crate::telemetry::*;
    pub use crate::types::*;
}

// Re-exports for convenience
pub use context::RingContext;
pub use control::ControlBlock;
pub use error::{RingKernelError, Result};
pub use hlc::HlcTimestamp;
pub use memory::{DeviceMemory, GpuBuffer, MemoryPool, PinnedMemory};
pub use message::{MessageHeader, MessageId, Priority, RingMessage};
pub use queue::{MessageQueue, QueueStats};
pub use runtime::{
    Backend, KernelHandle, KernelId, KernelState, KernelStatus, LaunchOptions, RingKernelRuntime,
};
pub use telemetry::TelemetryBuffer;
pub use types::{BlockId, GlobalThreadId, ThreadId, WarpId};
