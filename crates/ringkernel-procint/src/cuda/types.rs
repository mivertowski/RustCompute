//! GPU-compatible type definitions for CUDA kernels.

#![allow(missing_docs)]

use crate::models::{
    ConformanceResult, GpuDFGEdge, GpuDFGGraph, GpuDFGNode, GpuObjectEvent, GpuPartialOrderTrace,
    GpuPatternMatch,
};

/// GPU buffer wrapper for type safety.
#[derive(Debug)]
pub struct GpuBuffer<T> {
    /// Number of elements.
    pub len: usize,
    /// Element type marker.
    _marker: std::marker::PhantomData<T>,
    /// Raw pointer (null on CPU, valid on GPU).
    #[cfg(feature = "cuda")]
    pub ptr: *mut T,
}

impl<T> Default for GpuBuffer<T> {
    fn default() -> Self {
        Self {
            len: 0,
            _marker: std::marker::PhantomData,
            #[cfg(feature = "cuda")]
            ptr: std::ptr::null_mut(),
        }
    }
}

impl<T> GpuBuffer<T> {
    /// Create a new empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            len: capacity,
            _marker: std::marker::PhantomData,
            #[cfg(feature = "cuda")]
            ptr: std::ptr::null_mut(),
        }
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Type aliases for common GPU buffers.
pub type EventBuffer = GpuBuffer<GpuObjectEvent>;
pub type DFGNodeBuffer = GpuBuffer<GpuDFGNode>;
pub type DFGEdgeBuffer = GpuBuffer<GpuDFGEdge>;
pub type DFGGraphBuffer = GpuBuffer<GpuDFGGraph>;
pub type PatternBuffer = GpuBuffer<GpuPatternMatch>;
pub type ConformanceBuffer = GpuBuffer<ConformanceResult>;
pub type PartialOrderBuffer = GpuBuffer<GpuPartialOrderTrace>;

/// Kernel launch configuration.
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    /// Grid dimensions.
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions.
    pub block_dim: (u32, u32, u32),
    /// Shared memory size in bytes.
    pub shared_mem_bytes: u32,
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

impl LaunchConfig {
    /// Create 1D launch configuration.
    pub fn linear(num_elements: u32, block_size: u32) -> Self {
        let grid_size = num_elements.div_ceil(block_size);
        Self {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Create 2D launch configuration.
    pub fn grid_2d(width: u32, height: u32, tile_size: u32) -> Self {
        Self {
            grid_dim: (width.div_ceil(tile_size), height.div_ceil(tile_size), 1),
            block_dim: (tile_size, tile_size, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Set shared memory size.
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_launch_config_linear() {
        let config = LaunchConfig::linear(10000, 256);
        assert_eq!(config.grid_dim.0, 40); // ceil(10000/256)
        assert_eq!(config.block_dim.0, 256);
    }

    #[test]
    fn test_launch_config_2d() {
        let config = LaunchConfig::grid_2d(1024, 768, 16);
        assert_eq!(config.grid_dim.0, 64);
        assert_eq!(config.grid_dim.1, 48);
    }
}
