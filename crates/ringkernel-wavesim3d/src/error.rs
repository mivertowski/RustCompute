//! Unified error types for the WaveSim3D crate.
//!
//! This module provides a top-level [`WaveSim3dError`] enum that wraps all
//! subsystem-specific error types, enabling callers to use a single error
//! type across the entire crate.

use thiserror::Error;

use crate::audio::WavError;
use crate::visualization::renderer::RendererError;

/// Top-level error type for the WaveSim3D crate.
///
/// Wraps all subsystem errors into a single enum for convenient
/// propagation with `?`.
#[derive(Debug, Error)]
pub enum WaveSim3dError {
    /// Audio subsystem error (WAV I/O).
    #[error(transparent)]
    Wav(#[from] WavError),

    /// Visualization / rendering error.
    #[error(transparent)]
    Renderer(#[from] RendererError),

    /// GPU stencil backend error.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Gpu(#[from] crate::simulation::gpu_backend::GpuError),

    /// GPU cell-actor backend error.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Actor(#[from] crate::simulation::actor_backend::ActorError),

    /// GPU block-actor backend error.
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    BlockActor(#[from] crate::simulation::block_actor_backend::BlockActorError),

    /// Persistent GPU backend error.
    #[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
    #[error(transparent)]
    PersistentBackend(#[from] crate::simulation::persistent_backend::PersistentBackendError),
}
