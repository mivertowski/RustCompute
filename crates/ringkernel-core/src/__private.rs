//! Private module for proc macro integration.
//!
//! This module contains types that are used by the `ringkernel-derive` proc macros
//! but are not part of the public API. They are exposed for macro-generated code only.

use crate::types::KernelMode;

/// Kernel registration information collected by `#[ring_kernel]` macro.
///
/// This struct is used with the `inventory` crate to collect kernel registrations
/// at compile time. It should not be used directly by user code.
#[derive(Debug, Clone)]
pub struct KernelRegistration {
    /// Unique kernel identifier.
    pub id: &'static str,
    /// Execution mode (persistent or event-driven).
    pub mode: KernelMode,
    /// Grid size (number of blocks).
    pub grid_size: u32,
    /// Block size (threads per block).
    pub block_size: u32,
    /// Target kernels this kernel publishes to (for K2K routing).
    pub publishes_to: &'static [&'static str],
}

// Register the type with inventory for static collection
inventory::collect!(KernelRegistration);

/// Get all registered kernels.
pub fn registered_kernels() -> impl Iterator<Item = &'static KernelRegistration> {
    inventory::iter::<KernelRegistration>()
}

/// Stencil kernel registration information collected by `#[stencil_kernel]` macro.
///
/// This struct stores the generated CUDA source code and stencil configuration
/// for runtime compilation and execution.
#[derive(Debug, Clone)]
pub struct StencilKernelRegistration {
    /// Unique kernel identifier.
    pub id: &'static str,
    /// Grid dimensionality ("1d", "2d", or "3d").
    pub grid: &'static str,
    /// Tile width (block X dimension).
    pub tile_width: u32,
    /// Tile height (block Y dimension).
    pub tile_height: u32,
    /// Halo/ghost cell width (stencil radius).
    pub halo: u32,
    /// Generated CUDA C source code.
    pub cuda_source: &'static str,
}

// Register stencil kernels with inventory
inventory::collect!(StencilKernelRegistration);

/// Get all registered stencil kernels.
pub fn registered_stencil_kernels() -> impl Iterator<Item = &'static StencilKernelRegistration> {
    inventory::iter::<StencilKernelRegistration>()
}

/// Find a stencil kernel registration by ID.
pub fn find_stencil_kernel(id: &str) -> Option<&'static StencilKernelRegistration> {
    registered_stencil_kernels().find(|k| k.id == id)
}
