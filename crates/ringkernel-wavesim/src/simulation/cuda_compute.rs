//! CUDA backend implementation for tile-based FDTD simulation.
//!
//! This module implements the `TileGpuBackend` trait for NVIDIA CUDA GPUs,
//! providing GPU-resident tile buffers with minimal host transfers.
//!
//! ## Features
//!
//! - PTX compilation via NVRTC
//! - Persistent GPU buffers (pressure stays on GPU)
//! - Halo-only transfers (64 bytes per edge per step)
//! - Full grid readback only when rendering
//!
//! ## Kernel Source
//!
//! When the `cuda-codegen` feature is enabled, kernels are generated from Rust DSL.
//! Otherwise, handwritten CUDA source from `shaders/fdtd_tile.cu` is used.

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::memory::GpuBuffer;
use ringkernel_cuda::{CudaBuffer, CudaDevice};

use super::gpu_backend::{Edge, FdtdParams, TileGpuBackend, TileGpuBuffers};

/// CUDA kernel source for tile FDTD.
///
/// Uses generated DSL kernels when `cuda-codegen` feature is enabled,
/// otherwise falls back to handwritten CUDA source.
#[cfg(feature = "cuda-codegen")]
fn get_cuda_source() -> String {
    super::kernels::generate_tile_kernels()
}

#[cfg(not(feature = "cuda-codegen"))]
fn get_cuda_source() -> String {
    include_str!("../shaders/fdtd_tile.cu").to_string()
}

/// Module name for loaded kernels.
const MODULE_NAME: &str = "fdtd_tile";

/// Kernel function names.
const FN_FDTD_STEP: &str = "fdtd_tile_step";
const FN_EXTRACT_HALO: &str = "extract_halo";
const FN_INJECT_HALO: &str = "inject_halo";
const FN_READ_INTERIOR: &str = "read_interior";

/// CUDA tile GPU backend.
///
/// Manages CUDA device, compiled kernels, and provides the `TileGpuBackend` implementation.
pub struct CudaTileBackend {
    /// CUDA device.
    device: CudaDevice,
    /// Tile size (interior cells per side).
    tile_size: u32,
}

impl CudaTileBackend {
    /// Create a new CUDA tile backend.
    ///
    /// Compiles the FDTD kernels using NVRTC.
    pub fn new(tile_size: u32) -> Result<Self> {
        Self::with_device(0, tile_size)
    }

    /// Create a CUDA tile backend on a specific device.
    pub fn with_device(ordinal: usize, tile_size: u32) -> Result<Self> {
        let device = CudaDevice::new(ordinal)?;

        tracing::info!(
            "CUDA tile backend: {} (CC {}.{})",
            device.name(),
            device.compute_capability().0,
            device.compute_capability().1
        );

        // Compile CUDA source to PTX and load module
        let cuda_source = get_cuda_source();
        let ptx = cudarc::nvrtc::compile_ptx(&cuda_source).map_err(|e| {
            RingKernelError::BackendError(format!("NVRTC compilation failed: {}", e))
        })?;

        device.inner().load_ptx(
            ptx,
            MODULE_NAME,
            &[FN_FDTD_STEP, FN_EXTRACT_HALO, FN_INJECT_HALO, FN_READ_INTERIOR],
        ).map_err(|e| {
            RingKernelError::BackendError(format!("Failed to load PTX module: {}", e))
        })?;

        Ok(Self {
            device,
            tile_size,
        })
    }

    /// Get buffer width (tile_size + 2 for halos).
    pub fn buffer_width(&self) -> u32 {
        self.tile_size + 2
    }

    /// Get buffer size in bytes (18x18 floats for 16x16 tile).
    pub fn buffer_size_bytes(&self) -> usize {
        let bw = self.buffer_width() as usize;
        bw * bw * std::mem::size_of::<f32>()
    }

    /// Get halo buffer size in bytes (16 floats).
    pub fn halo_size_bytes(&self) -> usize {
        self.tile_size as usize * std::mem::size_of::<f32>()
    }

    /// Get interior buffer size in bytes (16x16 floats).
    pub fn interior_size_bytes(&self) -> usize {
        (self.tile_size * self.tile_size) as usize * std::mem::size_of::<f32>()
    }

    /// Get device reference.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }
}

impl TileGpuBackend for CudaTileBackend {
    type Buffer = CudaBuffer;

    fn create_tile_buffers(&self, tile_size: u32) -> Result<TileGpuBuffers<Self::Buffer>> {
        let buffer_width = tile_size + 2;
        let buffer_size = (buffer_width * buffer_width) as usize * std::mem::size_of::<f32>();
        let halo_size = tile_size as usize * std::mem::size_of::<f32>();

        // Allocate pressure buffers (18x18 for 16x16 tile)
        let pressure_a = CudaBuffer::new(&self.device, buffer_size)?;
        let pressure_b = CudaBuffer::new(&self.device, buffer_size)?;

        // Allocate halo staging buffers (16 floats each)
        let halo_north = CudaBuffer::new(&self.device, halo_size)?;
        let halo_south = CudaBuffer::new(&self.device, halo_size)?;
        let halo_west = CudaBuffer::new(&self.device, halo_size)?;
        let halo_east = CudaBuffer::new(&self.device, halo_size)?;

        // Initialize buffers to zero
        let zeros_buffer = vec![0u8; buffer_size];
        let zeros_halo = vec![0u8; halo_size];

        pressure_a.copy_from_host(&zeros_buffer)?;
        pressure_b.copy_from_host(&zeros_buffer)?;
        halo_north.copy_from_host(&zeros_halo)?;
        halo_south.copy_from_host(&zeros_halo)?;
        halo_west.copy_from_host(&zeros_halo)?;
        halo_east.copy_from_host(&zeros_halo)?;

        Ok(TileGpuBuffers {
            pressure_a,
            pressure_b,
            halo_north,
            halo_south,
            halo_west,
            halo_east,
            current_is_a: true,
            tile_size,
            buffer_width,
        })
    }

    fn upload_initial_state(
        &self,
        buffers: &TileGpuBuffers<Self::Buffer>,
        pressure: &[f32],
        pressure_prev: &[f32],
    ) -> Result<()> {
        // Convert f32 slices to byte slices
        let pressure_bytes: &[u8] = bytemuck::cast_slice(pressure);
        let pressure_prev_bytes: &[u8] = bytemuck::cast_slice(pressure_prev);

        buffers.pressure_a.copy_from_host(pressure_bytes)?;
        buffers.pressure_b.copy_from_host(pressure_prev_bytes)?;

        Ok(())
    }

    fn fdtd_step(
        &self,
        buffers: &TileGpuBuffers<Self::Buffer>,
        params: &FdtdParams,
    ) -> Result<()> {
        use cudarc::driver::LaunchAsync;

        // Ping-pong: read from current, write to previous
        let (current, prev) = if buffers.current_is_a {
            (&buffers.pressure_a, &buffers.pressure_b)
        } else {
            (&buffers.pressure_b, &buffers.pressure_a)
        };

        // Get kernel function
        let kernel_fn = self.device.inner()
            .get_func(MODULE_NAME, FN_FDTD_STEP)
            .ok_or_else(|| RingKernelError::BackendError("FDTD kernel not found".to_string()))?;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        // Get device pointers
        let current_ptr = current.device_ptr();
        let prev_ptr = prev.device_ptr();

        // Launch kernel
        unsafe {
            kernel_fn.launch(cfg, (
                current_ptr,
                prev_ptr,
                params.c2,
                params.damping,
            ))
        }.map_err(|e| {
            RingKernelError::BackendError(format!("FDTD kernel launch failed: {}", e))
        })?;

        Ok(())
    }

    fn extract_halo(
        &self,
        buffers: &TileGpuBuffers<Self::Buffer>,
        edge: Edge,
    ) -> Result<Vec<f32>> {
        use cudarc::driver::LaunchAsync;

        // Get current pressure buffer
        let current = if buffers.current_is_a {
            &buffers.pressure_a
        } else {
            &buffers.pressure_b
        };

        // Get staging buffer for this edge
        let staging = buffers.halo_buffer(edge);

        // Get kernel function
        let kernel_fn = self.device.inner()
            .get_func(MODULE_NAME, FN_EXTRACT_HALO)
            .ok_or_else(|| RingKernelError::BackendError("Extract halo kernel not found".to_string()))?;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (16, 1, 1),
            shared_mem_bytes: 0,
        };

        let current_ptr = current.device_ptr();
        let staging_ptr = staging.device_ptr();
        let edge_val = edge as i32;

        // Launch kernel
        unsafe {
            kernel_fn.launch(cfg, (current_ptr, staging_ptr, edge_val))
        }.map_err(|e| {
            RingKernelError::BackendError(format!("Extract halo kernel launch failed: {}", e))
        })?;

        // Synchronize and read back
        self.device.synchronize()?;

        let mut halo = vec![0u8; self.halo_size_bytes()];
        staging.copy_to_host(&mut halo)?;

        // Convert bytes to f32
        Ok(bytemuck::cast_slice(&halo).to_vec())
    }

    fn inject_halo(
        &self,
        buffers: &TileGpuBuffers<Self::Buffer>,
        edge: Edge,
        data: &[f32],
    ) -> Result<()> {
        use cudarc::driver::LaunchAsync;

        // Get current pressure buffer
        let current = if buffers.current_is_a {
            &buffers.pressure_a
        } else {
            &buffers.pressure_b
        };

        // Get staging buffer for this edge
        let staging = buffers.halo_buffer(edge);

        // Upload halo data to staging buffer
        let data_bytes: &[u8] = bytemuck::cast_slice(data);
        staging.copy_from_host(data_bytes)?;

        // Get kernel function
        let kernel_fn = self.device.inner()
            .get_func(MODULE_NAME, FN_INJECT_HALO)
            .ok_or_else(|| RingKernelError::BackendError("Inject halo kernel not found".to_string()))?;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (16, 1, 1),
            shared_mem_bytes: 0,
        };

        let current_ptr = current.device_ptr();
        let staging_ptr = staging.device_ptr();
        let edge_val = edge as i32;

        // Launch kernel
        unsafe {
            kernel_fn.launch(cfg, (current_ptr, staging_ptr, edge_val))
        }.map_err(|e| {
            RingKernelError::BackendError(format!("Inject halo kernel launch failed: {}", e))
        })?;

        Ok(())
    }

    fn swap_buffers(&self, buffers: &mut TileGpuBuffers<Self::Buffer>) {
        buffers.current_is_a = !buffers.current_is_a;
    }

    fn read_interior_pressure(
        &self,
        buffers: &TileGpuBuffers<Self::Buffer>,
    ) -> Result<Vec<f32>> {
        use cudarc::driver::LaunchAsync;

        // Get current pressure buffer
        let current = if buffers.current_is_a {
            &buffers.pressure_a
        } else {
            &buffers.pressure_b
        };

        // Create temporary buffer for readback (16x16 = 256 floats)
        let output_buffer = CudaBuffer::new(&self.device, self.interior_size_bytes())?;

        // Get kernel function
        let kernel_fn = self.device.inner()
            .get_func(MODULE_NAME, FN_READ_INTERIOR)
            .ok_or_else(|| RingKernelError::BackendError("Read interior kernel not found".to_string()))?;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let current_ptr = current.device_ptr();
        let output_ptr = output_buffer.device_ptr();

        // Launch kernel
        unsafe {
            kernel_fn.launch(cfg, (current_ptr, output_ptr))
        }.map_err(|e| {
            RingKernelError::BackendError(format!("Read interior kernel launch failed: {}", e))
        })?;

        // Synchronize and read back
        self.device.synchronize()?;

        let mut output = vec![0u8; self.interior_size_bytes()];
        output_buffer.copy_to_host(&mut output)?;

        // Convert bytes to f32
        Ok(bytemuck::cast_slice(&output).to_vec())
    }

    fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_backend_creation() {
        let backend = CudaTileBackend::new(16).unwrap();
        assert_eq!(backend.tile_size, 16);
        assert_eq!(backend.buffer_width(), 18);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_buffer_creation() {
        let backend = CudaTileBackend::new(16).unwrap();
        let buffers = backend.create_tile_buffers(16).unwrap();

        assert_eq!(buffers.tile_size, 16);
        assert_eq!(buffers.buffer_width, 18);
        assert!(buffers.current_is_a);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_fdtd_step() {
        let backend = CudaTileBackend::new(16).unwrap();
        let mut buffers = backend.create_tile_buffers(16).unwrap();

        // Create test data with impulse at center
        let buffer_size = 18 * 18;
        let mut pressure = vec![0.0f32; buffer_size];
        let pressure_prev = vec![0.0f32; buffer_size];

        // Set center cell to 1.0 (interior center is at buffer position 9,9)
        let center_idx = 9 * 18 + 9;
        pressure[center_idx] = 1.0;

        // Upload initial state
        backend.upload_initial_state(&buffers, &pressure, &pressure_prev).unwrap();

        // Run FDTD step
        let params = FdtdParams::new(16, 0.25, 0.99);
        backend.fdtd_step(&buffers, &params).unwrap();
        backend.swap_buffers(&mut buffers);

        // Read back result
        let result = backend.read_interior_pressure(&buffers).unwrap();

        // Center should have decreased, neighbors should have increased
        let center_interior = 8 * 16 + 8; // Interior center
        assert!(result[center_interior].abs() < 1.0, "Center should have decreased");
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_halo_exchange() {
        let backend = CudaTileBackend::new(16).unwrap();
        let buffers = backend.create_tile_buffers(16).unwrap();

        // Create test data with gradient
        let buffer_size = 18 * 18;
        let mut pressure = vec![0.0f32; buffer_size];

        // Fill first interior row with values 1..16
        for x in 0..16 {
            let idx = 1 * 18 + (x + 1);
            pressure[idx] = (x + 1) as f32;
        }

        // Upload
        backend.upload_initial_state(&buffers, &pressure, &vec![0.0f32; buffer_size]).unwrap();

        // Extract north edge
        let halo = backend.extract_halo(&buffers, Edge::North).unwrap();

        assert_eq!(halo.len(), 16);
        for (i, &v) in halo.iter().enumerate() {
            assert_eq!(v, (i + 1) as f32, "Halo mismatch at {}", i);
        }
    }
}
