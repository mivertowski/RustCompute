//! GPU compute shader for tile-based FDTD simulation.
//!
//! This module provides GPU-accelerated FDTD computation for the tile actor grid.
//! Each tile dispatches a compute shader to process its interior cells in parallel,
//! while K2K messaging handles the halo exchange between tiles.
//!
//! ## Architecture
//!
//! ```text
//! +------------------+
//! | TileKernelGrid   |
//! |  - K2K Broker    |
//! +------------------+
//!         |
//!    +----+----+
//!    |         |
//! +------+ +------+
//! | Tile | | Tile |  <- Each tile has halo buffers
//! |Actor | |Actor |
//! +------+ +------+
//!    |         |
//! +------+ +------+
//! | GPU  | | GPU  |  <- GPU computes FDTD for interiors
//! |FDTD  | |FDTD  |
//! +------+ +------+
//! ```

use ringkernel_core::error::{Result, RingKernelError};
use std::sync::Arc;

/// WGSL shader source for tile FDTD computation.
const TILE_FDTD_SHADER: &str = r#"
// Tile FDTD compute shader
// Processes a single tile's interior using the halo buffers

struct TileParams {
    tile_size: u32,      // Interior size (e.g., 16)
    buffer_width: u32,   // Including halos (e.g., 18)
    c2: f32,             // Courant number squared
    damping: f32,        // Damping factor (1 - damping_coefficient)
}

@group(0) @binding(0) var<uniform> params: TileParams;
@group(0) @binding(1) var<storage, read> pressure: array<f32>;
@group(0) @binding(2) var<storage, read> pressure_prev: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressure_out: array<f32>;

@compute @workgroup_size(16, 16)
fn fdtd_step(@builtin(global_invocation_id) gid: vec3u) {
    let local_x = gid.x;
    let local_y = gid.y;
    let tile_size = params.tile_size;
    let bw = params.buffer_width;

    // Skip if outside interior
    if (local_x >= tile_size || local_y >= tile_size) {
        return;
    }

    // Buffer index (account for halo offset of 1)
    let idx = (local_y + 1u) * bw + (local_x + 1u);

    // Current and previous pressure
    let p_curr = pressure[idx];
    let p_prev = pressure_prev[idx];

    // Neighbor access (halos provide boundary values)
    let p_north = pressure[idx - bw];
    let p_south = pressure[idx + bw];
    let p_west = pressure[idx - 1u];
    let p_east = pressure[idx + 1u];

    // FDTD computation: Laplacian and time stepping
    let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
    let p_new = 2.0 * p_curr - p_prev + params.c2 * laplacian;

    // Apply damping
    pressure_out[idx] = p_new * params.damping;
}
"#;

/// Parameters for the FDTD shader (must match WGSL struct layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TileFdtdParams {
    pub tile_size: u32,
    pub buffer_width: u32,
    pub c2: f32,
    pub damping: f32,
}

/// GPU compute resources for a tile.
pub struct TileGpuCompute {
    /// WGPU device.
    device: Arc<wgpu::Device>,
    /// WGPU queue.
    queue: Arc<wgpu::Queue>,
    /// Compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Parameters buffer.
    params_buffer: wgpu::Buffer,
    /// Pressure buffer (with halos).
    pressure_buffer: wgpu::Buffer,
    /// Previous pressure buffer.
    pressure_prev_buffer: wgpu::Buffer,
    /// Output buffer (written by shader).
    pressure_out_buffer: wgpu::Buffer,
    /// Staging buffer for readback.
    staging_buffer: wgpu::Buffer,
    /// Tile size.
    tile_size: u32,
    /// Buffer width including halos.
    buffer_width: u32,
}

impl TileGpuCompute {
    /// Create GPU compute resources for a tile.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, tile_size: u32) -> Result<Self> {
        let buffer_width = tile_size + 2; // Add 1-cell halo on each side
        let buffer_size = (buffer_width * buffer_width) as usize * std::mem::size_of::<f32>();

        // Create shader module
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tile FDTD Shader"),
            source: wgpu::ShaderSource::Wgsl(TILE_FDTD_SHADER.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tile FDTD Bind Group Layout"),
            entries: &[
                // Params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Pressure (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Pressure prev (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Pressure out (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tile FDTD Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tile FDTD Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "fdtd_step",
        });

        // Create buffers
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Params Buffer"),
            size: std::mem::size_of::<TileFdtdParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pressure_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Pressure Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pressure_prev_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Pressure Prev Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pressure_out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Pressure Out Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Staging Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            params_buffer,
            pressure_buffer,
            pressure_prev_buffer,
            pressure_out_buffer,
            staging_buffer,
            tile_size,
            buffer_width,
        })
    }

    /// Dispatch FDTD compute for this tile.
    ///
    /// Takes the tile's pressure buffer (with halos already applied)
    /// and computes the FDTD step for all interior cells.
    pub fn compute_fdtd(
        &self,
        pressure: &[f32],
        pressure_prev: &[f32],
        c2: f32,
        damping: f32,
    ) -> Vec<f32> {
        let buffer_size = (self.buffer_width * self.buffer_width) as usize;

        // Upload data
        let params = TileFdtdParams {
            tile_size: self.tile_size,
            buffer_width: self.buffer_width,
            c2,
            damping,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&self.pressure_buffer, 0, bytemuck::cast_slice(pressure));
        self.queue.write_buffer(
            &self.pressure_prev_buffer,
            0,
            bytemuck::cast_slice(pressure_prev),
        );

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile FDTD Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.pressure_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.pressure_prev_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.pressure_out_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tile FDTD Encoder"),
            });

        // Dispatch compute
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tile FDTD Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch enough workgroups to cover tile interior
            // Workgroup size is 16x16, so for 16x16 tile we need 1x1 workgroups
            let workgroups_x = (self.tile_size + 15) / 16;
            let workgroups_y = (self.tile_size + 15) / 16;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.pressure_out_buffer,
            0,
            &self.staging_buffer,
            0,
            (buffer_size * std::mem::size_of::<f32>()) as u64,
        );

        // Submit and wait
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map staging buffer");

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();

        result
    }
}

/// Shared GPU compute pool for multiple tiles.
///
/// Instead of each tile having its own GPU resources, tiles share
/// a compute pool to minimize GPU memory usage and maximize throughput.
pub struct TileGpuComputePool {
    /// WGPU device.
    device: Arc<wgpu::Device>,
    /// WGPU queue.
    queue: Arc<wgpu::Queue>,
    /// Compute pipeline (shared).
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout (shared).
    bind_group_layout: wgpu::BindGroupLayout,
    /// Tile size.
    tile_size: u32,
    /// Buffer width including halos.
    buffer_width: u32,
    /// Params buffer (uniform, shared).
    params_buffer: wgpu::Buffer,
}

impl TileGpuComputePool {
    /// Create a new GPU compute pool.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, tile_size: u32) -> Result<Self> {
        let buffer_width = tile_size + 2;

        // Create shader module
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tile FDTD Shader"),
            source: wgpu::ShaderSource::Wgsl(TILE_FDTD_SHADER.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tile FDTD Bind Group Layout"),
            entries: &[
                // Params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Pressure (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Pressure prev (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Pressure out (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tile FDTD Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tile FDTD Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "fdtd_step",
        });

        // Create params buffer
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Params Buffer"),
            size: std::mem::size_of::<TileFdtdParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            tile_size,
            buffer_width,
            params_buffer,
        })
    }

    /// Create buffers for a single tile.
    pub fn create_tile_buffers(&self) -> TileBuffers {
        let buffer_size =
            (self.buffer_width * self.buffer_width) as usize * std::mem::size_of::<f32>();

        let pressure_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Pressure Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pressure_prev_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Pressure Prev Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pressure_out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Pressure Out Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tile Staging Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        TileBuffers {
            pressure: pressure_buffer,
            pressure_prev: pressure_prev_buffer,
            pressure_out: pressure_out_buffer,
            staging: staging_buffer,
        }
    }

    /// Dispatch FDTD compute for a tile.
    pub fn compute_fdtd(
        &self,
        tile_buffers: &TileBuffers,
        pressure: &[f32],
        pressure_prev: &[f32],
        c2: f32,
        damping: f32,
    ) -> Vec<f32> {
        let buffer_size = (self.buffer_width * self.buffer_width) as usize;

        // Upload data
        let params = TileFdtdParams {
            tile_size: self.tile_size,
            buffer_width: self.buffer_width,
            c2,
            damping,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        self.queue
            .write_buffer(&tile_buffers.pressure, 0, bytemuck::cast_slice(pressure));
        self.queue.write_buffer(
            &tile_buffers.pressure_prev,
            0,
            bytemuck::cast_slice(pressure_prev),
        );

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tile FDTD Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tile_buffers.pressure.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_buffers.pressure_prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tile_buffers.pressure_out.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tile FDTD Encoder"),
            });

        // Dispatch compute
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tile FDTD Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = (self.tile_size + 15) / 16;
            let workgroups_y = (self.tile_size + 15) / 16;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &tile_buffers.pressure_out,
            0,
            &tile_buffers.staging,
            0,
            (buffer_size * std::mem::size_of::<f32>()) as u64,
        );

        // Submit and wait
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = tile_buffers.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("Failed to map staging buffer");

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        tile_buffers.staging.unmap();

        result
    }

    /// Get the device reference.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Get the queue reference.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
}

/// GPU buffers for a single tile.
pub struct TileBuffers {
    /// Current pressure buffer.
    pub pressure: wgpu::Buffer,
    /// Previous pressure buffer.
    pub pressure_prev: wgpu::Buffer,
    /// Output pressure buffer.
    pub pressure_out: wgpu::Buffer,
    /// Staging buffer for readback.
    pub staging: wgpu::Buffer,
}

/// Initialize WGPU adapter for tile GPU compute.
pub async fn init_wgpu() -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| {
            RingKernelError::BackendUnavailable("No WebGPU adapter found".to_string())
        })?;

    let info = adapter.get_info();
    tracing::info!("Tile GPU Compute: {} ({:?})", info.name, info.backend);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("WaveSim Tile GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .map_err(|e| RingKernelError::BackendError(format!("Failed to create device: {}", e)))?;

    Ok((Arc::new(device), Arc::new(queue)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // May not have GPU
    async fn test_tile_gpu_compute() {
        let (device, queue) = init_wgpu().await.unwrap();
        let pool = TileGpuComputePool::new(device, queue, 16).unwrap();
        let buffers = pool.create_tile_buffers();

        // Create test data (18x18 buffer for 16x16 tile with halos)
        let buffer_size = 18 * 18;
        let mut pressure = vec![0.0f32; buffer_size];
        let pressure_prev = vec![0.0f32; buffer_size];

        // Set center cell to 1.0
        let center_idx = 9 * 18 + 9; // Center of 18x18 buffer
        pressure[center_idx] = 1.0;

        let result = pool.compute_fdtd(&buffers, &pressure, &pressure_prev, 0.25, 0.99);

        // Result should show pressure spreading
        assert!(
            result[center_idx].abs() < 1.0,
            "Center should have decreased"
        );
        assert!(
            result[center_idx - 18].abs() > 0.0,
            "North should have increased"
        );
    }

    #[tokio::test]
    #[ignore] // May not have GPU
    async fn test_tile_single_compute() {
        let (device, queue) = init_wgpu().await.unwrap();
        let compute = TileGpuCompute::new(device, queue, 16).unwrap();

        let buffer_size = 18 * 18;
        let mut pressure = vec![0.0f32; buffer_size];
        let pressure_prev = vec![0.0f32; buffer_size];

        let center_idx = 9 * 18 + 9;
        pressure[center_idx] = 1.0;

        let result = compute.compute_fdtd(&pressure, &pressure_prev, 0.25, 0.99);

        assert!(result[center_idx].abs() < 1.0);
    }
}
