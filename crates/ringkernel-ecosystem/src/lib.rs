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
pub mod prelude {
    pub use crate::error::*;

    #[cfg(feature = "actix")]
    pub use crate::actix::*;

    #[cfg(feature = "tower")]
    pub use crate::tower::*;

    #[cfg(feature = "axum")]
    pub use crate::axum::*;

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
