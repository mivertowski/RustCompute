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
pub mod k2k;
pub mod memory;
pub mod message;
pub mod multi_gpu;
pub mod pubsub;
pub mod queue;
pub mod runtime;
pub mod telemetry;
pub mod telemetry_pipeline;
pub mod types;
pub mod checkpoint;

/// Private module for proc macro integration.
/// Not part of the public API - exposed for macro-generated code only.
#[doc(hidden)]
pub mod __private;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::context::*;
    pub use crate::control::*;
    pub use crate::error::*;
    pub use crate::hlc::*;
    pub use crate::k2k::{
        DeliveryStatus, K2KBroker, K2KBuilder, K2KConfig, K2KEndpoint, K2KMessage,
    };
    pub use crate::memory::*;
    pub use crate::message::{
        priority, CorrelationId, MessageEnvelope, MessageHeader, MessageId, Priority, RingMessage,
    };
    pub use crate::multi_gpu::{
        DeviceInfo, DeviceStatus, LoadBalancingStrategy, MultiGpuBuilder, MultiGpuCoordinator,
    };
    pub use crate::pubsub::{PubSubBroker, PubSubBuilder, Publication, QoS, Subscription, Topic};
    pub use crate::queue::*;
    pub use crate::runtime::*;
    pub use crate::telemetry::*;
    pub use crate::telemetry_pipeline::{
        MetricsCollector, MetricsSnapshot, TelemetryAlert, TelemetryConfig, TelemetryEvent,
        TelemetryPipeline,
    };
    pub use crate::types::*;
}

// Re-exports for convenience
pub use context::RingContext;
pub use control::ControlBlock;
pub use error::{Result, RingKernelError};
pub use hlc::HlcTimestamp;
pub use memory::{DeviceMemory, GpuBuffer, MemoryPool, PinnedMemory};
pub use message::{priority, MessageHeader, MessageId, Priority, RingMessage};
pub use queue::{MessageQueue, QueueStats};
pub use runtime::{
    Backend, KernelHandle, KernelId, KernelState, KernelStatus, LaunchOptions, RingKernelRuntime,
};
pub use telemetry::TelemetryBuffer;
pub use types::{BlockId, GlobalThreadId, ThreadId, WarpId};
