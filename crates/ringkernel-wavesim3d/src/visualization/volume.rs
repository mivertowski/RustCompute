//! Volumetric rendering for 3D wave simulation.
//!
//! Uses ray marching to render the pressure field as a translucent volume.

use crate::simulation::SimulationGrid3D;
use half::f16;
use wgpu::util::DeviceExt;

/// Ray marching shader for volume rendering
/// Uses a cube proxy geometry to constrain rendering to the volume bounds
const VOLUME_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_pos: vec4<f32>,
    grid_size: vec4<f32>,
};

struct VolumeParams {
    grid_dims: vec4<u32>,      // x, y, z dimensions, padding
    density_scale: f32,
    step_size: f32,
    max_steps: u32,
    threshold: f32,
};

// Group 0: Camera bind group (shared with other render passes)
@group(0) @binding(0) var<uniform> camera: CameraUniform;
// Group 1: Volume-specific bind group
@group(1) @binding(0) var<uniform> params: VolumeParams;
@group(1) @binding(1) var volume_texture: texture_3d<f32>;
@group(1) @binding(2) var volume_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

// Cube vertices for proxy geometry (36 vertices for 12 triangles)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Unit cube vertices (will be scaled by grid_size)
    var cube_vertices = array<vec3<f32>, 36>(
        // Front face
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
        // Back face
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
        // Top face
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 0.0),
        // Bottom face
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0),
        // Right face
        vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0),
        // Left face
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0),
        vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 1.0, 0.0),
    );

    // Scale cube to grid size
    let local_pos = cube_vertices[vertex_index];
    let world_pos = local_pos * camera.grid_size.xyz;

    var output: VertexOutput;
    output.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    output.world_pos = world_pos;

    return output;
}

// Ray-box intersection
fn intersect_box(ray_origin: vec3<f32>, ray_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / ray_dir;
    let t1 = (box_min - ray_origin) * inv_dir;
    let t2 = (box_max - ray_origin) * inv_dir;

    let tmin = min(t1, t2);
    let tmax = max(t1, t2);

    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);

    return vec2<f32>(t_near, t_far);
}

// Color mapping: blue (negative) - white (zero) - red (positive)
fn pressure_to_color(pressure: f32) -> vec4<f32> {
    let p = clamp(pressure, -1.0, 1.0);

    if (p >= 0.0) {
        // White to Red
        return vec4<f32>(1.0, 1.0 - p, 1.0 - p, abs(p) * params.density_scale);
    } else {
        // Blue to White
        return vec4<f32>(1.0 + p, 1.0 + p, 1.0, abs(p) * params.density_scale);
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray_origin = camera.camera_pos.xyz;
    // Ray direction from camera through the cube surface point
    let ray_dir = normalize(in.world_pos - ray_origin);

    // Box bounds (0 to grid_size)
    let box_min = vec3<f32>(0.0, 0.0, 0.0);
    let box_max = camera.grid_size.xyz;

    // Find intersection with volume box
    let t = intersect_box(ray_origin, ray_dir, box_min, box_max);

    if (t.x > t.y || t.y < 0.0) {
        // No intersection - transparent
        discard;
    }

    // Clamp to positive t values (start from where ray enters the volume)
    let t_start = max(t.x, 0.0);
    let t_end = t.y;

    // Ray marching through the volume
    var accumulated_color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var current_t = t_start;
    let step = params.step_size;
    var steps: u32 = 0u;

    loop {
        if (current_t >= t_end || steps >= params.max_steps || accumulated_color.a >= 0.95) {
            break;
        }

        let pos = ray_origin + ray_dir * current_t;

        // Convert to normalized texture coordinates [0, 1]
        let uvw = pos / box_max;

        // Sample pressure from 3D texture
        let pressure = textureSample(volume_texture, volume_sampler, uvw).r;

        // Only render if above threshold
        if (abs(pressure) > params.threshold) {
            let sample_color = pressure_to_color(pressure);

            // Front-to-back compositing
            let alpha = sample_color.a * (1.0 - accumulated_color.a);
            accumulated_color = vec4<f32>(
                accumulated_color.rgb + sample_color.rgb * alpha,
                accumulated_color.a + alpha
            );
        }

        current_t = current_t + step;
        steps = steps + 1u;
    }

    // Blend with background
    if (accumulated_color.a < 0.01) {
        discard;
    }

    return accumulated_color;
}
"#;

/// Parameters for volume rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeParams {
    /// Grid dimensions (x, y, z, padding)
    pub grid_dims: [u32; 4],
    /// Density scale for alpha
    pub density_scale: f32,
    /// Step size for ray marching
    pub step_size: f32,
    /// Maximum ray marching steps
    pub max_steps: u32,
    /// Minimum pressure threshold
    pub threshold: f32,
}

impl Default for VolumeParams {
    fn default() -> Self {
        Self {
            grid_dims: [64, 32, 64, 0],
            density_scale: 2.0,
            step_size: 0.05,
            max_steps: 256,
            threshold: 0.01,
        }
    }
}

/// Volume renderer using ray marching
pub struct VolumeRenderer {
    /// Render pipeline
    pipeline: wgpu::RenderPipeline,
    /// Volume texture
    volume_texture: wgpu::Texture,
    /// Volume texture view
    #[allow(dead_code)]
    volume_view: wgpu::TextureView,
    /// Texture sampler
    #[allow(dead_code)]
    sampler: wgpu::Sampler,
    /// Parameters uniform buffer
    params_buffer: wgpu::Buffer,
    /// Bind group
    bind_group: wgpu::BindGroup,
    /// Bind group layout
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// Volume parameters
    pub params: VolumeParams,
    /// Grid dimensions
    dimensions: (usize, usize, usize),
    /// Maximum pressure for normalization
    max_pressure: f32,
}

impl VolumeRenderer {
    /// Create a new volume renderer
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        dimensions: (usize, usize, usize),
    ) -> Self {
        let (width, height, depth) = dimensions;

        // Create 3D texture for volume data
        // Using R16Float which supports filtering (linear sampling)
        let volume_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("volume_texture"),
            size: wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: depth as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let volume_view = volume_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("volume_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let params = VolumeParams {
            grid_dims: [width as u32, height as u32, depth as u32, 0],
            ..Default::default()
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("volume_params_buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout for volume-specific bindings
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("volume_bind_group_layout"),
            entries: &[
                // VolumeParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3D texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("volume_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&volume_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("volume_pipeline_layout"),
            bind_group_layouts: &[camera_bind_group_layout, &bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("volume_shader"),
            source: wgpu::ShaderSource::Wgsl(VOLUME_SHADER.into()),
        });

        // Create pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("volume_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            volume_texture,
            volume_view,
            sampler,
            params_buffer,
            bind_group,
            bind_group_layout,
            params,
            dimensions,
            max_pressure: 1.0,
        }
    }

    /// Update the volume texture from simulation grid
    pub fn update_volume(&mut self, queue: &wgpu::Queue, grid: &SimulationGrid3D) {
        let (width, height, depth) = self.dimensions;

        // Update max pressure for normalization
        self.max_pressure = grid.max_pressure().max(0.001);

        // Copy and normalize pressure data to f16 format (as raw bytes)
        let mut data: Vec<u8> = Vec::with_capacity(width * height * depth * 2);
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let idx = z * (width * height) + y * width + x;
                    let pressure = grid.pressure[idx] / self.max_pressure;
                    let f16_val = f16::from_f32(pressure);
                    data.extend_from_slice(&f16_val.to_le_bytes());
                }
            }
        }

        // Upload to texture (f16 = 2 bytes per pixel)
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.volume_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((width * 2) as u32), // 2 bytes per f16 pixel
                rows_per_image: Some(height as u32),
            },
            wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: depth as u32,
            },
        );
    }

    /// Update volume parameters
    pub fn update_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Render the volume
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.bind_group, &[]);
        render_pass.draw(0..36, 0..1); // 36 vertices for cube (6 faces * 2 triangles * 3 vertices)
    }

    /// Set density scale
    pub fn set_density_scale(&mut self, scale: f32) {
        self.params.density_scale = scale;
    }

    /// Set step size
    pub fn set_step_size(&mut self, step: f32) {
        self.params.step_size = step;
    }

    /// Set threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.params.threshold = threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_params_default() {
        let params = VolumeParams::default();
        assert_eq!(params.max_steps, 256);
        assert!(params.density_scale > 0.0);
    }
}
