//! Ecosystem integrations for RingKernel.
//!
//! This crate provides optional integrations with popular Rust ecosystem
//! libraries for actors, web frameworks, data processing, and observability.
//!
//! # Feature Flags
//!
//! All integrations are opt-in via feature flags:
//!
//! - `actix` - Actix actor framework bridge
//! - `tower` - Tower service middleware integration
//! - `axum` - Axum web framework integration
//! - `arrow` - Apache Arrow data processing
//! - `polars` - Polars DataFrame operations
//! - `grpc` - gRPC server via Tonic
//! - `candle` - Candle ML framework bridge
//! - `config` - Configuration file management
//! - `tracing-integration` - Enhanced tracing support
//! - `prometheus` - Prometheus metrics export
//! - `full` - All integrations enabled
//!
//! # Example
//!
//! ```toml
//! [dependencies]
//! ringkernel-ecosystem = { version = "0.1", features = ["axum", "prometheus"] }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;

#[cfg(feature = "persistent")]
pub mod persistent;

#[cfg(feature = "persistent-cuda")]
pub mod cuda_bridge;

#[cfg(feature = "actix")]
pub mod actix;

#[cfg(feature = "tower")]
pub mod tower;

#[cfg(feature = "axum")]
pub mod axum;

#[cfg(feature = "arrow")]
pub mod arrow;

#[cfg(feature = "polars")]
pub mod polars;

#[cfg(feature = "grpc")]
pub mod grpc;

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "config")]
pub mod config;

#[cfg(feature = "tracing-integration")]
pub mod tracing_ext;

#[cfg(feature = "prometheus")]
pub mod metrics;

/// Prelude for convenient imports.
///
/// Note: Each integration module defines its own `RuntimeHandle` trait with
/// different bounds. To avoid conflicts, import the specific `RuntimeHandle`
/// from the module you're using (e.g., `use ringkernel_ecosystem::axum::RuntimeHandle`).
pub mod prelude {
    pub use crate::error::*;

    #[cfg(feature = "persistent")]
    pub use crate::persistent::*;

    // Re-export commonly used types, excluding RuntimeHandle to avoid conflicts
    #[cfg(feature = "actix")]
    pub use crate::actix::{
        BridgeHealth, GpuActorBridge, GpuActorConfig, GpuActorExt, GpuRequest, GpuResponse,
        HealthCheck, TypedGpuRequest,
    };
    #[cfg(all(feature = "actix", feature = "persistent"))]
    pub use crate::actix::{
        GetStatsCmd, GpuPersistentActor, GpuPersistentActorExt, InjectCmd, PauseCmd,
        PersistentActorConfig, PersistentResponseMsg, ResumeCmd, RunStepsCmd, ShutdownCmd,
        Subscribe, Unsubscribe,
    };

    #[cfg(all(feature = "tower", feature = "persistent"))]
    pub use crate::tower::{
        PersistentKernelLayer, PersistentKernelService, PersistentServiceBuilder,
        PersistentServiceConfig, PersistentServiceRequest, PersistentServiceResponse,
    };
    #[cfg(feature = "tower")]
    pub use crate::tower::{
        RingKernelLayer, RingKernelService, ServiceBuilder, ServiceConfig, ServiceRequest,
        ServiceResponse,
    };

    #[cfg(feature = "axum")]
    pub use crate::axum::{
        gpu_handler, health_handler, ApiRequest, ApiResponse, AppState, AxumConfig, HealthResponse,
        KernelHealth, RequestStats, RouterBuilder, StatsSnapshot,
    };
    #[cfg(all(feature = "axum", feature = "persistent"))]
    pub use crate::axum::{
        impulse_handler, pause_handler, persistent_health_handler, persistent_stats_handler,
        resume_handler, step_handler, CommandResponse, ImpulseRequest, PersistentAxumConfig,
        PersistentGpuState, PersistentHealthResponse, PersistentStatsResponse, StepRequest,
    };

    #[cfg(feature = "persistent-cuda")]
    pub use crate::cuda_bridge::{CudaPersistentHandle, CudaPersistentHandleBuilder};

    #[cfg(feature = "arrow")]
    pub use crate::arrow::*;

    #[cfg(feature = "polars")]
    pub use crate::polars::*;

    #[cfg(feature = "grpc")]
    pub use crate::grpc::*;

    #[cfg(feature = "candle")]
    pub use crate::candle::*;

    #[cfg(feature = "config")]
    pub use crate::config::*;

    #[cfg(feature = "prometheus")]
    pub use crate::metrics::*;
}
