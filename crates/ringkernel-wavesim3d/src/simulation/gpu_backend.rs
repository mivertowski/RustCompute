//! GPU backend for 3D acoustic wave simulation.
//!
//! Uses CUDA for NVIDIA GPUs with a packed buffer layout
//! for efficient memory access and halo exchange.

use super::grid3d::{GridParams, SimulationGrid3D};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

/// GPU error types.
#[derive(Debug)]
pub enum GpuError {
    DeviceError(String),
    CompileError(String),
    LaunchError(String),
    MemoryError(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::DeviceError(msg) => write!(f, "GPU device error: {}", msg),
            GpuError::CompileError(msg) => write!(f, "GPU compile error: {}", msg),
            GpuError::LaunchError(msg) => write!(f, "GPU launch error: {}", msg),
            GpuError::MemoryError(msg) => write!(f, "GPU memory error: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

/// CUDA kernel source for 3D FDTD.
const FDTD_3D_KERNEL: &str = r#"
extern "C" __global__ void fdtd_3d_step(
    const float* __restrict__ pressure,
    float* __restrict__ pressure_prev,
    const unsigned char* __restrict__ cell_types,
    const float* __restrict__ reflection_coeff,
    unsigned int width,
    unsigned int height,
    unsigned int depth,
    unsigned int slice_size,
    float c_squared,
    float damping
) {
    // Calculate global 3D indices
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Bounds check (skip boundary cells)
    if (x < 1 || x >= width - 1 ||
        y < 1 || y >= height - 1 ||
        z < 1 || z >= depth - 1) {
        return;
    }

    // Linear index
    unsigned int idx = z * slice_size + y * width + x;

    // Skip non-normal cells (handled separately)
    if (cell_types[idx] != 0) {
        return;
    }

    // Current and previous pressure
    float p = pressure[idx];
    float p_prev = pressure_prev[idx];

    // 6-neighbor stencil for 3D Laplacian
    float p_west  = pressure[idx - 1];
    float p_east  = pressure[idx + 1];
    float p_south = pressure[idx - width];
    float p_north = pressure[idx + width];
    float p_down  = pressure[idx - slice_size];
    float p_up    = pressure[idx + slice_size];

    // 7-point stencil Laplacian
    float laplacian = p_west + p_east + p_south + p_north + p_down + p_up - 6.0f * p;

    // FDTD update equation
    float p_new = 2.0f * p - p_prev + c_squared * laplacian;

    // Apply damping
    pressure_prev[idx] = p_new * damping;
}

extern "C" __global__ void apply_boundary_conditions(
    float* __restrict__ pressure_prev,
    const unsigned char* __restrict__ cell_types,
    const float* __restrict__ reflection_coeff,
    unsigned int total_cells
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_cells) {
        return;
    }

    unsigned char cell_type = cell_types[idx];

    // Apply boundary conditions based on cell type
    // 1 = Absorber, 2 = Reflector, 3 = Obstacle
    if (cell_type == 1 || cell_type == 3) {
        pressure_prev[idx] *= reflection_coeff[idx];
    }
}

extern "C" __global__ void swap_buffers(
    float* __restrict__ a,
    float* __restrict__ b,
    unsigned int total_cells
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_cells) {
        return;
    }

    float temp = a[idx];
    a[idx] = b[idx];
    b[idx] = temp;
}

extern "C" __global__ void inject_impulse(
    float* __restrict__ pressure,
    float* __restrict__ pressure_prev,
    unsigned int x,
    unsigned int y,
    unsigned int z,
    unsigned int width,
    unsigned int slice_size,
    float amplitude
) {
    unsigned int idx = z * slice_size + y * width + x;
    pressure[idx] += amplitude;
    pressure_prev[idx] += amplitude * 0.5f;
}

extern "C" __global__ void inject_spherical_impulse(
    float* __restrict__ pressure,
    float* __restrict__ pressure_prev,
    unsigned int cx,
    unsigned int cy,
    unsigned int cz,
    unsigned int width,
    unsigned int height,
    unsigned int depth,
    unsigned int slice_size,
    float amplitude,
    float radius
) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) {
        return;
    }

    int dx = (int)x - (int)cx;
    int dy = (int)y - (int)cy;
    int dz = (int)z - (int)cz;

    float dist_sq = (float)(dx * dx + dy * dy + dz * dz);
    float r_sq = radius * radius;

    if (dist_sq <= r_sq) {
        // Gaussian falloff
        float factor = expf(-dist_sq / (2.0f * radius));
        unsigned int idx = z * slice_size + y * width + x;
        pressure[idx] += amplitude * factor;
        pressure_prev[idx] += amplitude * factor * 0.5f;
    }
}
"#;

/// GPU backend for 3D FDTD simulation.
pub struct GpuBackend3D {
    /// CUDA context
    ctx: Arc<CudaContext>,
    /// CUDA stream
    stream: Arc<CudaStream>,
    /// CUDA module
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    /// Kernel functions
    fn_fdtd_step: CudaFunction,
    fn_boundary: CudaFunction,
    fn_swap: CudaFunction,
    fn_impulse: CudaFunction,
    fn_spherical_impulse: CudaFunction,
    /// Pressure buffer
    pressure: CudaSlice<f32>,
    /// Previous pressure buffer
    pressure_prev: CudaSlice<f32>,
    /// Cell types buffer
    cell_types: CudaSlice<u8>,
    /// Reflection coefficients buffer
    reflection_coeff: CudaSlice<f32>,
    /// Grid parameters
    params: GridParams,
    /// Total number of cells
    total_cells: usize,
    /// Whether buffers are swapped (for tracking)
    swapped: bool,
}

impl GpuBackend3D {
    /// Create a new GPU backend from a CPU grid.
    pub fn new(grid: &SimulationGrid3D) -> Result<Self, GpuError> {
        // Initialize CUDA context and stream (cudarc 0.18.2 API)
        let ctx = CudaContext::new(0).map_err(|e| GpuError::DeviceError(e.to_string()))?;
        let stream = ctx.default_stream();

        // Compile CUDA kernel
        let ptx = cudarc::nvrtc::compile_ptx(FDTD_3D_KERNEL)
            .map_err(|e| GpuError::CompileError(e.to_string()))?;

        // Load module and functions (cudarc 0.18.2 API)
        let module = ctx
            .load_module(ptx)
            .map_err(|e| GpuError::CompileError(e.to_string()))?;

        let fn_fdtd_step = module
            .load_function("fdtd_3d_step")
            .map_err(|e| GpuError::CompileError(e.to_string()))?;
        let fn_boundary = module
            .load_function("apply_boundary_conditions")
            .map_err(|e| GpuError::CompileError(e.to_string()))?;
        let fn_swap = module
            .load_function("swap_buffers")
            .map_err(|e| GpuError::CompileError(e.to_string()))?;
        let fn_impulse = module
            .load_function("inject_impulse")
            .map_err(|e| GpuError::CompileError(e.to_string()))?;
        let fn_spherical_impulse = module
            .load_function("inject_spherical_impulse")
            .map_err(|e| GpuError::CompileError(e.to_string()))?;

        let total_cells = grid.width * grid.height * grid.depth;
        let params = GridParams::from(grid);

        // Allocate GPU buffers (cudarc 0.18.2 API)
        let mut pressure = unsafe {
            stream
                .alloc::<f32>(grid.pressure.len())
                .map_err(|e| GpuError::MemoryError(e.to_string()))?
        };
        stream
            .memcpy_htod(&grid.pressure, &mut pressure)
            .map_err(|e| GpuError::MemoryError(e.to_string()))?;

        let mut pressure_prev = unsafe {
            stream
                .alloc::<f32>(grid.pressure_prev.len())
                .map_err(|e| GpuError::MemoryError(e.to_string()))?
        };
        stream
            .memcpy_htod(&grid.pressure_prev, &mut pressure_prev)
            .map_err(|e| GpuError::MemoryError(e.to_string()))?;

        // Convert cell types to u8
        let cell_types_u8: Vec<u8> = grid
            .cell_types
            .iter()
            .map(|ct| match ct {
                super::grid3d::CellType::Normal => 0u8,
                super::grid3d::CellType::Absorber => 1u8,
                super::grid3d::CellType::Reflector => 2u8,
                super::grid3d::CellType::Obstacle => 3u8,
            })
            .collect();

        let mut cell_types = unsafe {
            stream
                .alloc::<u8>(cell_types_u8.len())
                .map_err(|e| GpuError::MemoryError(e.to_string()))?
        };
        stream
            .memcpy_htod(&cell_types_u8, &mut cell_types)
            .map_err(|e| GpuError::MemoryError(e.to_string()))?;

        let mut reflection_coeff = unsafe {
            stream
                .alloc::<f32>(grid.reflection_coeff.len())
                .map_err(|e| GpuError::MemoryError(e.to_string()))?
        };
        stream
            .memcpy_htod(&grid.reflection_coeff, &mut reflection_coeff)
            .map_err(|e| GpuError::MemoryError(e.to_string()))?;

        Ok(Self {
            ctx,
            stream,
            module,
            fn_fdtd_step,
            fn_boundary,
            fn_swap,
            fn_impulse,
            fn_spherical_impulse,
            pressure,
            pressure_prev,
            cell_types,
            reflection_coeff,
            params,
            total_cells,
            swapped: false,
        })
    }

    /// Perform one FDTD step on the GPU.
    pub fn step(&mut self, grid: &mut SimulationGrid3D) -> Result<(), GpuError> {
        // Calculate launch configuration for 3D kernel
        let block_size = (8u32, 8u32, 4u32); // 256 threads per block
        let grid_size = (
            self.params.width.div_ceil(block_size.0),
            self.params.height.div_ceil(block_size.1),
            self.params.depth.div_ceil(block_size.2),
        );

        let config_3d = LaunchConfig {
            block_dim: block_size,
            grid_dim: grid_size,
            shared_mem_bytes: 0,
        };

        // Launch FDTD kernel (cudarc 0.18.2 API)
        unsafe {
            self.stream
                .launch_builder(&self.fn_fdtd_step)
                .arg(&self.pressure)
                .arg(&self.pressure_prev)
                .arg(&self.cell_types)
                .arg(&self.reflection_coeff)
                .arg(&self.params.width)
                .arg(&self.params.height)
                .arg(&self.params.depth)
                .arg(&self.params.slice_size)
                .arg(&self.params.c_squared)
                .arg(&self.params.damping)
                .launch(config_3d)
                .map_err(|e| GpuError::LaunchError(e.to_string()))?;
        }

        // Apply boundary conditions
        let boundary_blocks = self.total_cells.div_ceil(256);
        let config_1d = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (boundary_blocks as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let total_cells_u32 = self.total_cells as u32;
        unsafe {
            self.stream
                .launch_builder(&self.fn_boundary)
                .arg(&self.pressure_prev)
                .arg(&self.cell_types)
                .arg(&self.reflection_coeff)
                .arg(&total_cells_u32)
                .launch(config_1d)
                .map_err(|e| GpuError::LaunchError(e.to_string()))?;
        }

        // Swap buffers
        std::mem::swap(&mut self.pressure, &mut self.pressure_prev);
        self.swapped = !self.swapped;

        // Update grid state
        grid.step += 1;
        grid.time += grid.params.time_step;

        Ok(())
    }

    /// Download pressure data from GPU to CPU grid.
    pub fn download_pressure(&self, grid: &mut SimulationGrid3D) -> Result<(), GpuError> {
        self.stream
            .memcpy_dtoh(&self.pressure, &mut grid.pressure)
            .map_err(|e| GpuError::MemoryError(e.to_string()))?;
        Ok(())
    }

    /// Upload pressure data from CPU grid to GPU.
    pub fn upload_pressure(&mut self, grid: &SimulationGrid3D) -> Result<(), GpuError> {
        self.stream
            .memcpy_htod(&grid.pressure, &mut self.pressure)
            .map_err(|e| GpuError::MemoryError(e.to_string()))?;

        self.stream
            .memcpy_htod(&grid.pressure_prev, &mut self.pressure_prev)
            .map_err(|e| GpuError::MemoryError(e.to_string()))?;

        Ok(())
    }

    /// Reset the GPU buffers to match the CPU grid.
    pub fn reset(&mut self, grid: &SimulationGrid3D) -> Result<(), GpuError> {
        self.upload_pressure(grid)?;
        self.swapped = false;
        Ok(())
    }

    /// Inject an impulse directly on the GPU.
    pub fn inject_impulse(
        &mut self,
        x: u32,
        y: u32,
        z: u32,
        amplitude: f32,
    ) -> Result<(), GpuError> {
        let config = LaunchConfig {
            block_dim: (1, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_impulse)
                .arg(&self.pressure)
                .arg(&self.pressure_prev)
                .arg(&x)
                .arg(&y)
                .arg(&z)
                .arg(&self.params.width)
                .arg(&self.params.slice_size)
                .arg(&amplitude)
                .launch(config)
                .map_err(|e| GpuError::LaunchError(e.to_string()))?;
        }

        Ok(())
    }

    /// Inject a spherical impulse on the GPU.
    pub fn inject_spherical_impulse(
        &mut self,
        cx: u32,
        cy: u32,
        cz: u32,
        amplitude: f32,
        radius: f32,
    ) -> Result<(), GpuError> {
        let block_size = (8u32, 8u32, 4u32);
        let grid_size = (
            self.params.width.div_ceil(block_size.0),
            self.params.height.div_ceil(block_size.1),
            self.params.depth.div_ceil(block_size.2),
        );

        let config = LaunchConfig {
            block_dim: block_size,
            grid_dim: grid_size,
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_spherical_impulse)
                .arg(&self.pressure)
                .arg(&self.pressure_prev)
                .arg(&cx)
                .arg(&cy)
                .arg(&cz)
                .arg(&self.params.width)
                .arg(&self.params.height)
                .arg(&self.params.depth)
                .arg(&self.params.slice_size)
                .arg(&amplitude)
                .arg(&radius)
                .launch(config)
                .map_err(|e| GpuError::LaunchError(e.to_string()))?;
        }

        Ok(())
    }

    /// Get the current GPU memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        // Each f32 buffer + u8 cell_types
        self.total_cells * (4 + 4 + 1 + 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::physics::{AcousticParams3D, Environment};

    fn create_test_grid() -> SimulationGrid3D {
        let params = AcousticParams3D::new(Environment::default(), 0.1);
        SimulationGrid3D::new(32, 32, 32, params)
    }

    #[test]
    #[ignore = "Requires CUDA GPU"]
    fn test_gpu_backend_creation() {
        let grid = create_test_grid();
        let gpu = GpuBackend3D::new(&grid);
        assert!(gpu.is_ok());
    }

    #[test]
    #[ignore = "Requires CUDA GPU"]
    fn test_gpu_step() {
        let mut grid = create_test_grid();
        grid.inject_impulse(16, 16, 16, 1.0);

        let mut gpu = GpuBackend3D::new(&grid).unwrap();

        // Run a few steps
        for _ in 0..10 {
            gpu.step(&mut grid).unwrap();
        }

        // Download and verify
        gpu.download_pressure(&mut grid).unwrap();

        // Energy should have spread
        assert!(grid.total_energy() > 0.0);
    }

    #[test]
    #[ignore = "Requires CUDA GPU"]
    fn test_gpu_impulse_injection() {
        let grid = create_test_grid();
        let mut gpu = GpuBackend3D::new(&grid).unwrap();

        // Inject on GPU
        gpu.inject_impulse(16, 16, 16, 1.0).unwrap();

        // Download and verify
        let mut grid_copy = create_test_grid();
        gpu.download_pressure(&mut grid_copy).unwrap();

        let center_pressure = grid_copy.pressure[grid_copy.index(16, 16, 16)];
        assert!(center_pressure > 0.5);
    }
}
