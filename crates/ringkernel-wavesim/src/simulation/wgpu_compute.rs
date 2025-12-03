//! WGPU backend implementation for tile-based FDTD simulation.
//!
//! This module implements the `TileGpuBackend` trait for WebGPU,
//! providing GPU-resident tile buffers with minimal host transfers.
//!
//! ## Features
//!
//! - Cross-platform GPU compute (Vulkan, Metal, DX12)
//! - Persistent GPU buffers (pressure stays on GPU)
//! - Halo-only transfers (64 bytes per edge per step)
//! - Full grid readback only when rendering

use ringkernel_core::error::{Result, RingKernelError};
use std::sync::Arc;

use super::gpu_backend::{Edge, FdtdParams, TileGpuBackend, TileGpuBuffers};

/// WGSL shader source for tile FDTD.
const WGSL_FDTD_SHADER: &str = r#"
// Tile FDTD compute shader
// Processes a single tile's interior using the 18x18 buffer (16x16 interior + 1-cell halo)

struct Params {
    tile_size: u32,      // Interior size (16)
    buffer_width: u32,   // Including halos (18)
    c2: f32,             // Courant number squared
    damping: f32,        // Damping factor (1 - damping_coefficient)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pressure: array<f32>;
@group(0) @binding(2) var<storage, read_write> pressure_out: array<f32>;

@compute @workgroup_size(16, 16)
fn fdtd_step(@builtin(global_invocation_id) gid: vec3u) {
    let lx = gid.x;
    let ly = gid.y;
    let bw = params.buffer_width;

    // Skip if outside interior
    if (lx >= params.tile_size || ly >= params.tile_size) {
        return;
    }

    // Buffer index (account for halo offset of 1)
    let idx = (ly + 1u) * bw + (lx + 1u);

    // Current pressure
    let p_curr = pressure[idx];

    // Previous pressure is stored in output buffer (ping-pong)
    let p_prev = pressure_out[idx];

    // Neighbor access (halos provide boundary values)
    let p_north = pressure[idx - bw];
    let p_south = pressure[idx + bw];
    let p_west = pressure[idx - 1u];
    let p_east = pressure[idx + 1u];

    // FDTD computation: Laplacian and time stepping
    let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
    let p_new = 2.0 * p_curr - p_prev + params.c2 * laplacian;

    // Apply damping and write to output
    pressure_out[idx] = p_new * params.damping;
}

// Halo extraction shader
struct HaloParams {
    edge: u32,           // 0=North, 1=South, 2=West, 3=East
    buffer_width: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> halo_params: HaloParams;
@group(0) @binding(1) var<storage, read> src_pressure: array<f32>;
@group(0) @binding(2) var<storage, read_write> halo_out: array<f32>;

@compute @workgroup_size(16)
fn extract_halo(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= 16u) { return; }

    let bw = halo_params.buffer_width;
    var idx: u32;

    switch (halo_params.edge) {
        case 0u: {  // North - extract row 1 (first interior row)
            idx = 1u * bw + (i + 1u);
        }
        case 1u: {  // South - extract row 16 (last interior row)
            idx = 16u * bw + (i + 1u);
        }
        case 2u: {  // West - extract col 1 (first interior col)
            idx = (i + 1u) * bw + 1u;
        }
        default: {  // East - extract col 16 (last interior col)
            idx = (i + 1u) * bw + 16u;
        }
    }

    halo_out[i] = src_pressure[idx];
}

// Halo injection shader
@compute @workgroup_size(16)
fn inject_halo(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= 16u) { return; }

    let bw = halo_params.buffer_width;
    var idx: u32;

    switch (halo_params.edge) {
        case 0u: {  // North - inject to row 0 (halo row)
            idx = 0u * bw + (i + 1u);
        }
        case 1u: {  // South - inject to row 17 (halo row)
            idx = 17u * bw + (i + 1u);
        }
        case 2u: {  // West - inject to col 0 (halo col)
            idx = (i + 1u) * bw + 0u;
        }
        default: {  // East - inject to col 17 (halo col)
            idx = (i + 1u) * bw + 17u;
        }
    }

    // halo_out is actually the input here, src_pressure is the destination
    // But we need to swap the bindings conceptually
    // Actually we read from halo_out (binding 2) and write to src_pressure (binding 1)
    // Let's use a different layout for injection
}
"#;

/// Separate shader for halo injection (needs different storage qualifiers)
const WGSL_INJECT_SHADER: &str = r#"
struct HaloParams {
    edge: u32,
    buffer_width: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> halo_params: HaloParams;
@group(0) @binding(1) var<storage, read_write> dst_pressure: array<f32>;
@group(0) @binding(2) var<storage, read> halo_in: array<f32>;

@compute @workgroup_size(16)
fn inject_halo(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= 16u) { return; }

    let bw = halo_params.buffer_width;
    var idx: u32;

    switch (halo_params.edge) {
        case 0u: { idx = 0u * bw + (i + 1u); }
        case 1u: { idx = 17u * bw + (i + 1u); }
        case 2u: { idx = (i + 1u) * bw + 0u; }
        default: { idx = (i + 1u) * bw + 17u; }
    }

    dst_pressure[idx] = halo_in[i];
}
"#;

/// Shader for reading interior to linear buffer
const WGSL_READ_INTERIOR_SHADER: &str = r#"
struct ReadParams {
    buffer_width: u32,
    tile_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> read_params: ReadParams;
@group(0) @binding(1) var<storage, read> src_pressure: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16)
fn read_interior(@builtin(global_invocation_id) gid: vec3u) {
    let lx = gid.x;
    let ly = gid.y;

    if (lx >= read_params.tile_size || ly >= read_params.tile_size) {
        return;
    }

    let bw = read_params.buffer_width;
    let src_idx = (ly + 1u) * bw + (lx + 1u);
    let dst_idx = ly * read_params.tile_size + lx;

    output[dst_idx] = src_pressure[src_idx];
}
"#;

/// Parameters for FDTD shader (must match WGSL struct layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WgpuFdtdParams {
    tile_size: u32,
    buffer_width: u32,
    c2: f32,
    damping: f32,
}

/// Parameters for halo shaders.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WgpuHaloParams {
    edge: u32,
    buffer_width: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Parameters for read interior shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WgpuReadParams {
    buffer_width: u32,
    tile_size: u32,
    _pad0: u32,
    _pad1: u32,
}

/// WGPU buffer wrapper for tile GPU buffers.
pub struct WgpuBuffer {
    /// The WGPU buffer.
    buffer: wgpu::Buffer,
    /// Size in bytes.
    size: usize,
}

impl WgpuBuffer {
    /// Create a new WGPU buffer.
    fn new(device: &wgpu::Device, size: usize, usage: wgpu::BufferUsages, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });
        Self { buffer, size }
    }

    /// Get the underlying buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get buffer size.
    pub fn size(&self) -> usize {
        self.size
    }
}

/// WGPU tile GPU backend.
pub struct WgpuTileBackend {
    /// WGPU device.
    device: Arc<wgpu::Device>,
    /// WGPU queue.
    queue: Arc<wgpu::Queue>,
    /// FDTD compute pipeline.
    fdtd_pipeline: wgpu::ComputePipeline,
    /// FDTD bind group layout.
    fdtd_bind_group_layout: wgpu::BindGroupLayout,
    /// Extract halo pipeline.
    extract_pipeline: wgpu::ComputePipeline,
    /// Extract bind group layout.
    extract_bind_group_layout: wgpu::BindGroupLayout,
    /// Inject halo pipeline.
    inject_pipeline: wgpu::ComputePipeline,
    /// Inject bind group layout.
    inject_bind_group_layout: wgpu::BindGroupLayout,
    /// Read interior pipeline.
    read_pipeline: wgpu::ComputePipeline,
    /// Read bind group layout.
    read_bind_group_layout: wgpu::BindGroupLayout,
    /// Tile size.
    tile_size: u32,
}

impl WgpuTileBackend {
    /// Create a new WGPU tile backend.
    pub async fn new(tile_size: u32) -> Result<Self> {
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
        tracing::info!("WGPU tile backend: {} ({:?})", info.name, info.backend);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("WaveSim WGPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| {
                RingKernelError::BackendError(format!("Failed to create device: {}", e))
            })?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Create FDTD pipeline
        let fdtd_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FDTD Shader"),
            source: wgpu::ShaderSource::Wgsl(WGSL_FDTD_SHADER.into()),
        });

        let fdtd_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("FDTD Bind Group Layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

        let fdtd_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FDTD Pipeline Layout"),
            bind_group_layouts: &[&fdtd_bind_group_layout],
            push_constant_ranges: &[],
        });

        let fdtd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FDTD Pipeline"),
            layout: Some(&fdtd_pipeline_layout),
            module: &fdtd_module,
            entry_point: "fdtd_step",
        });

        // Create extract halo pipeline
        let extract_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Extract Bind Group Layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

        let extract_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Extract Pipeline Layout"),
                bind_group_layouts: &[&extract_bind_group_layout],
                push_constant_ranges: &[],
            });

        let extract_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Extract Halo Pipeline"),
            layout: Some(&extract_pipeline_layout),
            module: &fdtd_module,
            entry_point: "extract_halo",
        });

        // Create inject halo pipeline
        let inject_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Inject Shader"),
            source: wgpu::ShaderSource::Wgsl(WGSL_INJECT_SHADER.into()),
        });

        let inject_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Inject Bind Group Layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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
                ],
            });

        let inject_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Inject Pipeline Layout"),
                bind_group_layouts: &[&inject_bind_group_layout],
                push_constant_ranges: &[],
            });

        let inject_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Inject Halo Pipeline"),
            layout: Some(&inject_pipeline_layout),
            module: &inject_module,
            entry_point: "inject_halo",
        });

        // Create read interior pipeline
        let read_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Read Interior Shader"),
            source: wgpu::ShaderSource::Wgsl(WGSL_READ_INTERIOR_SHADER.into()),
        });

        let read_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Read Bind Group Layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

        let read_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Read Pipeline Layout"),
            bind_group_layouts: &[&read_bind_group_layout],
            push_constant_ranges: &[],
        });

        let read_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Read Interior Pipeline"),
            layout: Some(&read_pipeline_layout),
            module: &read_module,
            entry_point: "read_interior",
        });

        Ok(Self {
            device,
            queue,
            fdtd_pipeline,
            fdtd_bind_group_layout,
            extract_pipeline,
            extract_bind_group_layout,
            inject_pipeline,
            inject_bind_group_layout,
            read_pipeline,
            read_bind_group_layout,
            tile_size,
        })
    }

    /// Get device reference.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get queue reference.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

impl TileGpuBackend for WgpuTileBackend {
    type Buffer = WgpuBuffer;

    fn create_tile_buffers(&self, tile_size: u32) -> Result<TileGpuBuffers<Self::Buffer>> {
        let buffer_width = tile_size + 2;
        let buffer_size = (buffer_width * buffer_width) as usize * std::mem::size_of::<f32>();
        let halo_size = tile_size as usize * std::mem::size_of::<f32>();

        // Pressure buffers need COPY_DST for upload and COPY_SRC for potential readback
        let pressure_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let pressure_a = WgpuBuffer::new(&self.device, buffer_size, pressure_usage, "Pressure A");
        let pressure_b = WgpuBuffer::new(&self.device, buffer_size, pressure_usage, "Pressure B");

        // Halo buffers for staging
        let halo_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let halo_north = WgpuBuffer::new(&self.device, halo_size, halo_usage, "Halo North");
        let halo_south = WgpuBuffer::new(&self.device, halo_size, halo_usage, "Halo South");
        let halo_west = WgpuBuffer::new(&self.device, halo_size, halo_usage, "Halo West");
        let halo_east = WgpuBuffer::new(&self.device, halo_size, halo_usage, "Halo East");

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
        self.queue.write_buffer(
            buffers.pressure_a.buffer(),
            0,
            bytemuck::cast_slice(pressure),
        );
        self.queue.write_buffer(
            buffers.pressure_b.buffer(),
            0,
            bytemuck::cast_slice(pressure_prev),
        );
        Ok(())
    }

    fn fdtd_step(&self, buffers: &TileGpuBuffers<Self::Buffer>, params: &FdtdParams) -> Result<()> {
        let (current, prev) = if buffers.current_is_a {
            (&buffers.pressure_a, &buffers.pressure_b)
        } else {
            (&buffers.pressure_b, &buffers.pressure_a)
        };

        // Create params buffer
        let gpu_params = WgpuFdtdParams {
            tile_size: params.tile_size,
            buffer_width: params.tile_size + 2,
            c2: params.c2,
            damping: params.damping,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FDTD Params"),
                contents: bytemuck::bytes_of(&gpu_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FDTD Bind Group"),
            layout: &self.fdtd_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: current.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: prev.buffer().as_entire_binding(),
                },
            ],
        });

        // Encode and submit
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FDTD Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FDTD Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fdtd_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn extract_halo(&self, buffers: &TileGpuBuffers<Self::Buffer>, edge: Edge) -> Result<Vec<f32>> {
        let current = if buffers.current_is_a {
            &buffers.pressure_a
        } else {
            &buffers.pressure_b
        };

        let halo_buffer = buffers.halo_buffer(edge);

        // Create params buffer
        let params = WgpuHaloParams {
            edge: edge as u32,
            buffer_width: buffers.buffer_width,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Extract Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Extract Bind Group"),
            layout: &self.extract_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: current.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: halo_buffer.buffer().as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Extract Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Extract Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.extract_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Copy to staging buffer for readback
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Halo Staging"),
            size: halo_buffer.size() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            halo_buffer.buffer(),
            0,
            &staging,
            0,
            halo_buffer.size() as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back
        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .unwrap()
            .map_err(|e| RingKernelError::TransferFailed(format!("Map failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    fn inject_halo(
        &self,
        buffers: &TileGpuBuffers<Self::Buffer>,
        edge: Edge,
        data: &[f32],
    ) -> Result<()> {
        let current = if buffers.current_is_a {
            &buffers.pressure_a
        } else {
            &buffers.pressure_b
        };

        let halo_buffer = buffers.halo_buffer(edge);

        // Upload halo data
        self.queue
            .write_buffer(halo_buffer.buffer(), 0, bytemuck::cast_slice(data));

        // Create params buffer
        let params = WgpuHaloParams {
            edge: edge as u32,
            buffer_width: buffers.buffer_width,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Inject Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Inject Bind Group"),
            layout: &self.inject_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: current.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: halo_buffer.buffer().as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Inject Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Inject Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inject_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn swap_buffers(&self, buffers: &mut TileGpuBuffers<Self::Buffer>) {
        buffers.current_is_a = !buffers.current_is_a;
    }

    fn read_interior_pressure(&self, buffers: &TileGpuBuffers<Self::Buffer>) -> Result<Vec<f32>> {
        let current = if buffers.current_is_a {
            &buffers.pressure_a
        } else {
            &buffers.pressure_b
        };

        let interior_size = (self.tile_size * self.tile_size) as usize * std::mem::size_of::<f32>();

        // Create output buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Interior Output"),
            size: interior_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params
        let params = WgpuReadParams {
            buffer_width: buffers.buffer_width,
            tile_size: self.tile_size,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Read Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Read Bind Group"),
            layout: &self.read_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: current.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Read Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.read_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Copy to staging
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Interior Staging"),
            size: interior_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, interior_size as u64);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back
        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .unwrap()
            .map_err(|e| RingKernelError::TransferFailed(format!("Map failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    fn synchronize(&self) -> Result<()> {
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
}

// Need to import BufferInitDescriptor
use wgpu::util::DeviceExt;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires GPU
    async fn test_wgpu_backend_creation() {
        let backend = WgpuTileBackend::new(16).await.unwrap();
        assert_eq!(backend.tile_size, 16);
    }

    #[tokio::test]
    #[ignore] // Requires GPU
    async fn test_wgpu_buffer_creation() {
        let backend = WgpuTileBackend::new(16).await.unwrap();
        let buffers = backend.create_tile_buffers(16).unwrap();

        assert_eq!(buffers.tile_size, 16);
        assert_eq!(buffers.buffer_width, 18);
        assert!(buffers.current_is_a);
    }
}
