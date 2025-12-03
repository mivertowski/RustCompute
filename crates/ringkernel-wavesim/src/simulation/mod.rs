//! Simulation core for 2D acoustic wave propagation.

mod cell;
mod grid;
mod kernel_grid;
mod physics;
mod tile_grid;
pub mod educational;

// FDTD kernel defined in Rust DSL (transpiled to CUDA at compile time)
pub mod fdtd_dsl;

// All CUDA kernels generated from Rust DSL
pub mod kernels;

#[cfg(feature = "simd")]
mod simd;

// GPU backend abstraction and implementations
pub mod gpu_backend;

#[cfg(feature = "wgpu")]
mod gpu_compute;

#[cfg(feature = "wgpu")]
pub mod wgpu_compute;

#[cfg(feature = "cuda")]
pub mod cuda_compute;

#[cfg(feature = "cuda")]
pub mod cuda_packed;

pub use cell::{CellState, Direction};
pub use grid::{CellType, SimulationGrid};
pub use kernel_grid::KernelGrid;
pub use physics::AcousticParams;
pub use tile_grid::{TileKernelGrid, TileActor, HaloDirection, DEFAULT_TILE_SIZE, GpuPersistentBackend};
pub use educational::{SimulationMode, ProcessingState, EducationalProcessor, TileMessage, StepResult};

// GPU backend trait and types
pub use gpu_backend::{Edge, FdtdParams, TileGpuBackend, TileGpuBuffers};

#[cfg(feature = "wgpu")]
pub use gpu_compute::{TileGpuCompute, TileGpuComputePool, TileBuffers, TileFdtdParams, init_wgpu};

#[cfg(feature = "wgpu")]
pub use wgpu_compute::WgpuTileBackend;

#[cfg(feature = "cuda")]
pub use cuda_compute::CudaTileBackend;

#[cfg(feature = "cuda")]
pub use cuda_packed::CudaPackedBackend;
