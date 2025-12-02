//! Packed CUDA backend for GPU-only tile-based FDTD simulation.
//!
//! This backend keeps ALL tile data in a single packed GPU buffer, enabling
//! GPU-only halo exchange with ZERO host transfers during simulation steps.
//!
//! ## Key Features
//!
//! - **Zero host transfers**: Halo exchange happens entirely on GPU
//! - **Two kernel launches per step**: Exchange halos + FDTD compute
//! - **Massive parallelism**: All tiles computed in single kernel launch
//! - **Precomputed halo routing**: Copy indices computed once at init
//!
//! ## Memory Layout
//!
//! All tiles packed contiguously in GPU memory:
//! ```text
//! [Tile(0,0) 18×18][Tile(1,0) 18×18][Tile(0,1) 18×18]...
//! ```
//!
//! Each tile is `(tile_size + 2)²` floats (18×18 = 324 floats for 16×16 tile).

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::memory::GpuBuffer;
use ringkernel_cuda::{CudaBuffer, CudaDevice};

/// CUDA kernel source for packed tile FDTD.
///
/// Uses generated DSL kernels when `cuda-codegen` feature is enabled,
/// otherwise falls back to handwritten CUDA source.
#[cfg(feature = "cuda-codegen")]
fn get_cuda_packed_source() -> String {
    super::kernels::generate_packed_kernels()
}

#[cfg(not(feature = "cuda-codegen"))]
fn get_cuda_packed_source() -> String {
    include_str!("../shaders/fdtd_packed.cu").to_string()
}

/// Module name for loaded kernels.
const MODULE_NAME: &str = "fdtd_packed";

/// Kernel function names.
const FN_EXCHANGE_HALOS: &str = "exchange_all_halos";
const FN_FDTD_ALL_TILES: &str = "fdtd_all_tiles";
const FN_READ_ALL_INTERIORS: &str = "read_all_interiors";
const FN_INJECT_IMPULSE: &str = "inject_impulse";
const FN_APPLY_BOUNDARY: &str = "apply_boundary_conditions";

/// Packed CUDA backend for tile-based FDTD simulation.
///
/// All tiles are stored in a single contiguous GPU buffer, enabling
/// GPU-only halo exchange without any host transfers.
pub struct CudaPackedBackend {
    // Note: Cannot derive Debug due to CudaDevice/CudaBuffer not implementing it
    /// CUDA device.
    device: CudaDevice,

    /// Grid dimensions in tiles.
    tiles_x: u32,
    tiles_y: u32,

    /// Tile dimensions.
    tile_size: u32,
    buffer_width: u32,

    /// Grid dimensions in cells.
    grid_width: u32,
    grid_height: u32,

    /// Packed pressure buffers (all tiles contiguous).
    /// Buffer A: current pressure state
    /// Buffer B: previous pressure state (ping-pong)
    packed_a: CudaBuffer,
    packed_b: CudaBuffer,

    /// Which buffer is current (true = A, false = B).
    current_is_a: bool,

    /// Halo copy list on GPU.
    /// Format: [src0, dst0, src1, dst1, ...] as u32 indices into packed buffer.
    halo_copies: CudaBuffer,
    num_halo_copies: u32,

    /// Output buffer for visualization readback.
    output_buffer: CudaBuffer,

    /// Staging buffer for tile uploads.
    staging_buffer: CudaBuffer,

    /// FDTD parameters.
    c2: f32,
    damping: f32,
}

impl std::fmt::Debug for CudaPackedBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaPackedBackend")
            .field("tiles_x", &self.tiles_x)
            .field("tiles_y", &self.tiles_y)
            .field("tile_size", &self.tile_size)
            .field("grid_width", &self.grid_width)
            .field("grid_height", &self.grid_height)
            .field("num_halo_copies", &self.num_halo_copies)
            .field("c2", &self.c2)
            .field("damping", &self.damping)
            .finish_non_exhaustive()
    }
}

impl CudaPackedBackend {
    /// Create a new packed CUDA backend.
    ///
    /// # Arguments
    /// * `grid_width` - Width of the simulation grid in cells
    /// * `grid_height` - Height of the simulation grid in cells
    /// * `tile_size` - Size of each tile's interior (typically 16)
    pub fn new(grid_width: u32, grid_height: u32, tile_size: u32) -> Result<Self> {
        Self::with_device(0, grid_width, grid_height, tile_size)
    }

    /// Create a packed CUDA backend on a specific device.
    pub fn with_device(
        ordinal: usize,
        grid_width: u32,
        grid_height: u32,
        tile_size: u32,
    ) -> Result<Self> {
        let device = CudaDevice::new(ordinal)?;

        tracing::info!(
            "CUDA packed backend: {} (CC {}.{})",
            device.name(),
            device.compute_capability().0,
            device.compute_capability().1
        );

        // Calculate tile grid dimensions
        let tiles_x = (grid_width + tile_size - 1) / tile_size;
        let tiles_y = (grid_height + tile_size - 1) / tile_size;
        let buffer_width = tile_size + 2;
        let tile_buffer_size = buffer_width * buffer_width;
        let num_tiles = tiles_x * tiles_y;
        let total_buffer_size = (num_tiles * tile_buffer_size) as usize * std::mem::size_of::<f32>();

        tracing::info!(
            "Packed layout: {}x{} tiles, {} total floats ({:.1} KB)",
            tiles_x,
            tiles_y,
            num_tiles * tile_buffer_size,
            total_buffer_size as f64 / 1024.0
        );

        // Compile CUDA source to PTX and load module
        let cuda_source = get_cuda_packed_source();
        let ptx = cudarc::nvrtc::compile_ptx(&cuda_source).map_err(|e| {
            RingKernelError::BackendError(format!("NVRTC compilation failed: {}", e))
        })?;

        device
            .inner()
            .load_ptx(
                ptx,
                MODULE_NAME,
                &[
                    FN_EXCHANGE_HALOS,
                    FN_FDTD_ALL_TILES,
                    FN_READ_ALL_INTERIORS,
                    FN_INJECT_IMPULSE,
                    FN_APPLY_BOUNDARY,
                ],
            )
            .map_err(|e| RingKernelError::BackendError(format!("Failed to load PTX: {}", e)))?;

        // Allocate packed pressure buffers
        let packed_a = CudaBuffer::new(&device, total_buffer_size)?;
        let packed_b = CudaBuffer::new(&device, total_buffer_size)?;

        // Initialize to zero
        let zeros = vec![0u8; total_buffer_size];
        packed_a.copy_from_host(&zeros)?;
        packed_b.copy_from_host(&zeros)?;

        // Compute halo copy list
        let halo_copy_list = Self::compute_halo_copies(tiles_x, tiles_y, tile_size, buffer_width);
        let num_halo_copies = halo_copy_list.len() as u32;

        tracing::info!(
            "Halo copy list: {} copies ({:.1} KB on GPU)",
            num_halo_copies,
            (num_halo_copies as usize * 2 * std::mem::size_of::<u32>()) as f64 / 1024.0
        );

        // Upload halo copy list to GPU
        let halo_copy_bytes: Vec<u8> = halo_copy_list
            .iter()
            .flat_map(|(src, dst)| {
                let mut bytes = Vec::with_capacity(8);
                bytes.extend_from_slice(&src.to_ne_bytes());
                bytes.extend_from_slice(&dst.to_ne_bytes());
                bytes
            })
            .collect();

        let halo_copies = CudaBuffer::new(&device, halo_copy_bytes.len())?;
        halo_copies.copy_from_host(&halo_copy_bytes)?;

        // Allocate output buffer for visualization
        let output_size = (grid_width * grid_height) as usize * std::mem::size_of::<f32>();
        let output_buffer = CudaBuffer::new(&device, output_size)?;

        // Allocate staging buffer for tile uploads (one tile at a time)
        let staging_size = tile_buffer_size as usize * std::mem::size_of::<f32>();
        let staging_buffer = CudaBuffer::new(&device, staging_size)?;

        Ok(Self {
            device,
            tiles_x,
            tiles_y,
            tile_size,
            buffer_width,
            grid_width,
            grid_height,
            packed_a,
            packed_b,
            current_is_a: true,
            halo_copies,
            num_halo_copies,
            output_buffer,
            staging_buffer,
            c2: 0.25,      // Default Courant number squared
            damping: 0.99, // Default damping
        })
    }

    /// Compute the list of halo copy operations.
    ///
    /// Returns a list of (src_index, dst_index) pairs for copying halo data
    /// between adjacent tiles.
    fn compute_halo_copies(
        tiles_x: u32,
        tiles_y: u32,
        tile_size: u32,
        buffer_width: u32,
    ) -> Vec<(u32, u32)> {
        let tile_buffer_size = buffer_width * buffer_width;
        let mut copies = Vec::new();

        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let tile_offset = (ty * tiles_x + tx) * tile_buffer_size;

                // Copy this tile's edges to neighbor's halos

                // North edge → South halo of tile above
                // Extract row 1 (first interior row) → inject to row 17 (south halo) of north neighbor
                if ty > 0 {
                    let north_tile_offset = ((ty - 1) * tiles_x + tx) * tile_buffer_size;
                    for x in 0..tile_size {
                        let src = tile_offset + 1 * buffer_width + (x + 1);
                        let dst = north_tile_offset + (buffer_width - 1) * buffer_width + (x + 1);
                        copies.push((src, dst));
                    }
                }

                // South edge → North halo of tile below
                // Extract row 16 (last interior row) → inject to row 0 (north halo) of south neighbor
                if ty < tiles_y - 1 {
                    let south_tile_offset = ((ty + 1) * tiles_x + tx) * tile_buffer_size;
                    for x in 0..tile_size {
                        let src = tile_offset + tile_size * buffer_width + (x + 1);
                        let dst = south_tile_offset + 0 * buffer_width + (x + 1);
                        copies.push((src, dst));
                    }
                }

                // West edge → East halo of tile to the left
                // Extract col 1 (first interior col) → inject to col 17 (east halo) of west neighbor
                if tx > 0 {
                    let west_tile_offset = (ty * tiles_x + (tx - 1)) * tile_buffer_size;
                    for y in 0..tile_size {
                        let src = tile_offset + (y + 1) * buffer_width + 1;
                        let dst = west_tile_offset + (y + 1) * buffer_width + (buffer_width - 1);
                        copies.push((src, dst));
                    }
                }

                // East edge → West halo of tile to the right
                // Extract col 16 (last interior col) → inject to col 0 (west halo) of east neighbor
                if tx < tiles_x - 1 {
                    let east_tile_offset = (ty * tiles_x + (tx + 1)) * tile_buffer_size;
                    for y in 0..tile_size {
                        let src = tile_offset + (y + 1) * buffer_width + tile_size;
                        let dst = east_tile_offset + (y + 1) * buffer_width + 0;
                        copies.push((src, dst));
                    }
                }
            }
        }

        copies
    }

    /// Set FDTD parameters.
    pub fn set_params(&mut self, c2: f32, damping: f32) {
        self.c2 = c2;
        self.damping = damping;
    }

    /// Upload initial pressure data for a single tile.
    pub fn upload_tile(&self, tile_x: u32, tile_y: u32, pressure: &[f32]) -> Result<()> {
        use cudarc::driver::LaunchAsync;

        let pressure_bytes: &[u8] = bytemuck::cast_slice(pressure);
        self.staging_buffer.copy_from_host(pressure_bytes)?;

        let kernel = self
            .device
            .inner()
            .get_func(MODULE_NAME, "upload_tile_data")
            .ok_or_else(|| RingKernelError::BackendError("upload_tile_data not found".into()))?;

        let packed_ptr = if self.current_is_a {
            self.packed_a.device_ptr()
        } else {
            self.packed_b.device_ptr()
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (self.buffer_width, self.buffer_width, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    packed_ptr,
                    self.staging_buffer.device_ptr(),
                    tile_x as i32,
                    tile_y as i32,
                    self.tiles_x as i32,
                    self.buffer_width as i32,
                ),
            )
        }
        .map_err(|e| RingKernelError::BackendError(format!("Upload failed: {}", e)))?;

        Ok(())
    }

    /// Upload initial state for all tiles from a Vec of tile buffers.
    pub fn upload_all_tiles(&self, tiles: &[(u32, u32, Vec<f32>)]) -> Result<()> {
        for (tx, ty, pressure) in tiles {
            self.upload_tile(*tx, *ty, pressure)?;
        }
        // Also copy to the other buffer for proper ping-pong initialization
        let size = self.packed_a.size();
        let mut data = vec![0u8; size];
        if self.current_is_a {
            self.packed_a.copy_to_host(&mut data)?;
            self.packed_b.copy_from_host(&data)?;
        } else {
            self.packed_b.copy_to_host(&mut data)?;
            self.packed_a.copy_from_host(&data)?;
        }
        Ok(())
    }

    /// Perform one simulation step (GPU-only, no host transfers).
    ///
    /// This executes:
    /// 1. Halo exchange kernel (copies all edges between tiles)
    /// 2. Boundary conditions kernel (handles grid edges)
    /// 3. FDTD kernel (computes wave equation for all tiles)
    /// 4. Buffer swap (just flips a flag)
    pub fn step(&mut self) -> Result<()> {
        use cudarc::driver::LaunchAsync;

        let (curr_ptr, prev_ptr) = if self.current_is_a {
            (self.packed_a.device_ptr(), self.packed_b.device_ptr())
        } else {
            (self.packed_b.device_ptr(), self.packed_a.device_ptr())
        };

        // Phase 1: Exchange all halos (single kernel launch)
        if self.num_halo_copies > 0 {
            let exchange_kernel = self
                .device
                .inner()
                .get_func(MODULE_NAME, FN_EXCHANGE_HALOS)
                .ok_or_else(|| {
                    RingKernelError::BackendError("exchange_all_halos not found".into())
                })?;

            let threads_per_block = 256u32;
            let num_blocks = (self.num_halo_copies + threads_per_block - 1) / threads_per_block;

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                exchange_kernel.launch(
                    cfg,
                    (
                        curr_ptr,
                        self.halo_copies.device_ptr(),
                        self.num_halo_copies as i32,
                    ),
                )
            }
            .map_err(|e| RingKernelError::BackendError(format!("Halo exchange failed: {}", e)))?;
        }

        // Phase 2: Apply boundary conditions
        let boundary_kernel = self
            .device
            .inner()
            .get_func(MODULE_NAME, FN_APPLY_BOUNDARY)
            .ok_or_else(|| {
                RingKernelError::BackendError("apply_boundary_conditions not found".into())
            })?;

        let max_edge_threads =
            std::cmp::max(self.tiles_x, self.tiles_y) as u32 * self.tile_size;

        let boundary_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (4, 1, 1), // 4 edges
            block_dim: (max_edge_threads.min(1024), 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            boundary_kernel.launch(
                boundary_cfg,
                (
                    curr_ptr,
                    self.tiles_x as i32,
                    self.tiles_y as i32,
                    self.tile_size as i32,
                    self.buffer_width as i32,
                    0.95f32, // reflection coefficient (slight absorption)
                ),
            )
        }
        .map_err(|e| RingKernelError::BackendError(format!("Boundary failed: {}", e)))?;

        // Phase 3: Compute FDTD for all tiles (single kernel launch)
        let fdtd_kernel = self
            .device
            .inner()
            .get_func(MODULE_NAME, FN_FDTD_ALL_TILES)
            .ok_or_else(|| RingKernelError::BackendError("fdtd_all_tiles not found".into()))?;

        let fdtd_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (self.tiles_x, self.tiles_y, 1),
            block_dim: (self.tile_size, self.tile_size, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            fdtd_kernel.launch(
                fdtd_cfg,
                (
                    curr_ptr,
                    prev_ptr,
                    self.tiles_x as i32,
                    self.tiles_y as i32,
                    self.tile_size as i32,
                    self.buffer_width as i32,
                    self.c2,
                    self.damping,
                ),
            )
        }
        .map_err(|e| RingKernelError::BackendError(format!("FDTD failed: {}", e)))?;

        // Phase 4: Swap buffers (just flip the flag, no data movement)
        self.current_is_a = !self.current_is_a;

        Ok(())
    }

    /// Run N steps without any host interaction.
    pub fn step_batch(&mut self, count: u32) -> Result<()> {
        for _ in 0..count {
            self.step()?;
        }
        // Synchronize once at the end
        self.device.synchronize()?;
        Ok(())
    }

    /// Inject an impulse at the given global grid position.
    pub fn inject_impulse(&self, x: u32, y: u32, amplitude: f32) -> Result<()> {
        use cudarc::driver::LaunchAsync;

        if x >= self.grid_width || y >= self.grid_height {
            return Ok(());
        }

        let tile_x = x / self.tile_size;
        let tile_y = y / self.tile_size;
        let local_x = x % self.tile_size;
        let local_y = y % self.tile_size;

        let kernel = self
            .device
            .inner()
            .get_func(MODULE_NAME, FN_INJECT_IMPULSE)
            .ok_or_else(|| RingKernelError::BackendError("inject_impulse not found".into()))?;

        let curr_ptr = if self.current_is_a {
            self.packed_a.device_ptr()
        } else {
            self.packed_b.device_ptr()
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    curr_ptr,
                    tile_x as i32,
                    tile_y as i32,
                    local_x as i32,
                    local_y as i32,
                    self.tiles_x as i32,
                    self.buffer_width as i32,
                    amplitude,
                ),
            )
        }
        .map_err(|e| RingKernelError::BackendError(format!("Impulse failed: {}", e)))?;

        Ok(())
    }

    /// Read the full pressure grid for visualization.
    ///
    /// This is the only operation that transfers data from GPU to host.
    /// Call this only when you need to render the simulation.
    pub fn read_pressure_grid(&self) -> Result<Vec<f32>> {
        use cudarc::driver::LaunchAsync;

        let kernel = self
            .device
            .inner()
            .get_func(MODULE_NAME, FN_READ_ALL_INTERIORS)
            .ok_or_else(|| RingKernelError::BackendError("read_all_interiors not found".into()))?;

        let curr_ptr = if self.current_is_a {
            self.packed_a.device_ptr()
        } else {
            self.packed_b.device_ptr()
        };

        let threads = 16u32;
        let blocks_x = (self.grid_width + threads - 1) / threads;
        let blocks_y = (self.grid_height + threads - 1) / threads;

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks_x, blocks_y, 1),
            block_dim: (threads, threads, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    curr_ptr,
                    self.output_buffer.device_ptr(),
                    self.tiles_x as i32,
                    self.tiles_y as i32,
                    self.tile_size as i32,
                    self.buffer_width as i32,
                    self.grid_width as i32,
                    self.grid_height as i32,
                ),
            )
        }
        .map_err(|e| RingKernelError::BackendError(format!("Read failed: {}", e)))?;

        // Synchronize and read back
        self.device.synchronize()?;

        let output_size = (self.grid_width * self.grid_height) as usize * std::mem::size_of::<f32>();
        let mut output_bytes = vec![0u8; output_size];
        self.output_buffer.copy_to_host(&mut output_bytes)?;

        Ok(bytemuck::cast_slice(&output_bytes).to_vec())
    }

    /// Synchronize the GPU (wait for all operations to complete).
    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
    }

    /// Get grid dimensions.
    pub fn grid_size(&self) -> (u32, u32) {
        (self.grid_width, self.grid_height)
    }

    /// Get tile grid dimensions.
    pub fn tile_grid_size(&self) -> (u32, u32) {
        (self.tiles_x, self.tiles_y)
    }

    /// Get total number of tiles.
    pub fn tile_count(&self) -> u32 {
        self.tiles_x * self.tiles_y
    }

    /// Get number of halo copy operations per step.
    pub fn halo_copy_count(&self) -> u32 {
        self.num_halo_copies
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_packed_backend_creation() {
        let backend = CudaPackedBackend::new(64, 64, 16).unwrap();
        assert_eq!(backend.tiles_x, 4);
        assert_eq!(backend.tiles_y, 4);
        assert_eq!(backend.tile_count(), 16);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_packed_backend_step() {
        let mut backend = CudaPackedBackend::new(32, 32, 16).unwrap();
        backend.set_params(0.25, 0.99);

        // Inject impulse
        backend.inject_impulse(16, 16, 1.0).unwrap();

        // Run some steps
        for _ in 0..10 {
            backend.step().unwrap();
        }

        // Read back
        let pressure = backend.read_pressure_grid().unwrap();
        assert_eq!(pressure.len(), 32 * 32);

        // Check that wave has propagated (center should have changed)
        let center = pressure[16 * 32 + 16];
        assert!(center.abs() < 1.0, "Wave should have propagated");
    }

    #[test]
    fn test_halo_copy_computation() {
        // 2x2 tile grid
        let copies = CudaPackedBackend::compute_halo_copies(2, 2, 16, 18);

        // Each internal edge has 16 copies
        // 2x2 grid has: 2 horizontal edges + 2 vertical edges = 4 edges
        // Each edge = 16 copies, but bidirectional = 2x
        // Actually: tile(0,0) has east and south neighbors
        //           tile(1,0) has west and south neighbors
        //           tile(0,1) has east and north neighbors
        //           tile(1,1) has west and north neighbors
        // Total = 8 edge connections × 16 cells = 128 copies
        // But we copy in one direction per edge pair, so 4 edges × 16 = 64 each direction
        // Hmm, let me recalculate:
        // tile(0,0): south (16), east (16) = 32
        // tile(1,0): south (16) = 16 (west already covered by 0,0's east)
        // tile(0,1): east (16) = 16 (north already covered by 0,0's south)
        // tile(1,1): nothing new
        // Total = 32 + 16 + 16 = 64? No wait, each tile copies its OWN edges...

        // Let me trace through:
        // (0,0): ty>0? no. ty<1? yes, south to (0,1) = 16. tx>0? no. tx<1? yes, east to (1,0) = 16. Total: 32
        // (1,0): ty>0? no. ty<1? yes, south to (1,1) = 16. tx>0? yes, west to (0,0) = 16. tx<1? no. Total: 32
        // (0,1): ty>0? yes, north to (0,0) = 16. ty<1? no. tx>0? no. tx<1? yes, east to (1,1) = 16. Total: 32
        // (1,1): ty>0? yes, north to (1,0) = 16. ty<1? no. tx>0? yes, west to (0,1) = 16. tx<1? no. Total: 32
        // Grand total: 128 copies

        assert_eq!(copies.len(), 128);
    }
}
