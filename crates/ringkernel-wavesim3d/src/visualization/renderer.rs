//! Main 3D renderer for wave simulation visualization.
//!
//! Combines slice rendering, markers, and UI into a complete visualization system.

use super::camera::{Camera3D, CameraController};
use super::slice::SliceRenderer;
use super::volume::VolumeRenderer;
use super::{CameraUniform, ColorMap, GridLines, HeadWireframe, MarkerSphere, Vertex3D};
use crate::simulation::physics::Position3D;
use wgpu::util::DeviceExt;

/// Visualization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisualizationMode {
    /// Single slice view
    SingleSlice,
    /// Multiple orthogonal slices
    MultiSlice,
    /// Volume rendering (ray marching)
    #[default]
    VolumeRender,
    /// Isosurface rendering
    Isosurface,
}

/// Rendering configuration.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Visualization mode
    pub mode: VisualizationMode,
    /// Color map for pressure values
    pub color_map: ColorMap,
    /// Background color
    pub background_color: [f32; 4],
    /// Show bounding box
    pub show_bounding_box: bool,
    /// Show floor grid
    pub show_floor_grid: bool,
    /// Show source markers
    pub show_sources: bool,
    /// Show listener head
    pub show_listener: bool,
    /// Slice opacity
    pub slice_opacity: f32,
    /// Auto-scale pressure colors
    pub auto_scale: bool,
    /// Manual max pressure (if not auto-scaling)
    pub max_pressure: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            mode: VisualizationMode::VolumeRender,
            color_map: ColorMap::BlueWhiteRed,
            background_color: [0.1, 0.1, 0.15, 1.0],
            show_bounding_box: true,
            show_floor_grid: true,
            show_sources: true,
            show_listener: true,
            slice_opacity: 0.85,
            auto_scale: true,
            max_pressure: 1.0,
        }
    }
}

/// WGSL shader for basic 3D rendering.
const SHADER_SOURCE: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_pos: vec4<f32>,
    grid_size: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tex_coord: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) normal: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    out.world_pos = in.position;
    out.normal = in.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let normal = normalize(in.normal);
    let diffuse = max(dot(normal, light_dir), 0.3);

    var color = in.color;
    color = vec4<f32>(color.rgb * diffuse, color.a);

    return color;
}

@fragment
fn fs_main_line(in: VertexOutput) -> @location(0) vec4<f32> {
    // No lighting for lines
    return in.color;
}
"#;

/// Main 3D renderer.
pub struct Renderer3D {
    /// WGPU device
    device: wgpu::Device,
    /// WGPU queue
    queue: wgpu::Queue,
    /// Surface configuration
    surface_config: wgpu::SurfaceConfiguration,
    /// Surface
    surface: wgpu::Surface<'static>,
    /// Render pipeline for triangles
    triangle_pipeline: wgpu::RenderPipeline,
    /// Render pipeline for lines
    line_pipeline: wgpu::RenderPipeline,
    /// Camera uniform buffer
    camera_buffer: wgpu::Buffer,
    /// Camera bind group
    camera_bind_group: wgpu::BindGroup,
    /// Depth texture view (cached)
    depth_texture_view: wgpu::TextureView,
    /// Cached slice vertex buffer
    slice_buffer: Option<wgpu::Buffer>,
    /// Cached slice vertex count
    slice_vertex_count: u32,
    /// Cached line vertex buffer
    line_buffer: Option<wgpu::Buffer>,
    /// Cached line vertex count
    line_vertex_count: u32,
    /// Cached marker vertex buffer
    marker_buffer: Option<wgpu::Buffer>,
    /// Cached marker vertex count
    marker_vertex_count: u32,
    /// Volume renderer
    volume_renderer: Option<VolumeRenderer>,
    /// Grid dimensions (cells)
    #[allow(dead_code)]
    grid_dimensions: (usize, usize, usize),
    /// Camera bind group layout (for volume renderer)
    #[allow(dead_code)]
    camera_bind_group_layout: wgpu::BindGroupLayout,
    /// Camera
    pub camera: Camera3D,
    /// Camera controller
    pub camera_controller: CameraController,
    /// Slice renderer
    pub slice_renderer: SliceRenderer,
    /// Render configuration
    pub config: RenderConfig,
    /// Grid physical size
    grid_size: (f32, f32, f32),
    /// Source markers
    sources: Vec<MarkerSphere>,
    /// Listener head
    listener: Option<HeadWireframe>,
}

impl Renderer3D {
    /// Create a new renderer.
    pub async fn new(
        window: &winit::window::Window,
        grid_size: (f32, f32, f32),
        grid_dimensions: (usize, usize, usize),
    ) -> Result<Self, RendererError> {
        let size = window.inner_size();

        // Create WGPU instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Create surface - use unsafe to get 'static lifetime
        // SAFETY: The window handle outlives the surface. The window is valid
        // for the duration of the renderer's lifetime.
        let surface = unsafe {
            instance
                .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(window).unwrap())
                .map_err(|e| RendererError::SurfaceError(e.to_string()))?
        };

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| RendererError::AdapterError(e.to_string()))?;

        // Request device
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: Some("wavesim3d_device"),
                ..Default::default()
            })
            .await
            .map_err(|e| RendererError::DeviceError(e.to_string()))?;

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wavesim3d_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        // Create camera uniform buffer
        let camera_uniform = CameraUniform::new();
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create bind group
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create triangle render pipeline
        let triangle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("triangle_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex3D::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for transparent slices
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create line render pipeline
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex3D::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main_line"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Set up camera
        let mut camera = Camera3D::for_grid(grid_size);
        camera.set_aspect(size.width as f32 / size.height as f32);
        let camera_controller = CameraController::from_camera(&camera);

        // Create initial depth texture
        let depth_texture_view = Self::create_depth_texture_static(&device, &surface_config);

        // Create volume renderer
        let volume_renderer = VolumeRenderer::new(
            &device,
            surface_config.format,
            &bind_group_layout,
            grid_dimensions,
        );

        Ok(Self {
            device,
            queue,
            surface_config,
            surface,
            triangle_pipeline,
            line_pipeline,
            camera_buffer,
            camera_bind_group,
            depth_texture_view,
            slice_buffer: None,
            slice_vertex_count: 0,
            line_buffer: None,
            line_vertex_count: 0,
            marker_buffer: None,
            marker_vertex_count: 0,
            volume_renderer: Some(volume_renderer),
            grid_dimensions,
            camera_bind_group_layout: bind_group_layout,
            camera,
            camera_controller,
            slice_renderer: SliceRenderer::new(),
            config: RenderConfig::default(),
            grid_size,
            sources: Vec::new(),
            listener: None,
        })
    }

    /// Resize the render surface.
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
            self.camera
                .set_aspect(new_size.width as f32 / new_size.height as f32);
            // Recreate depth texture for new size
            self.depth_texture_view =
                Self::create_depth_texture_static(&self.device, &self.surface_config);
        }
    }

    /// Add a source marker.
    pub fn add_source(&mut self, position: Position3D, color: [f32; 4]) {
        self.sources.push(MarkerSphere::new(position, 0.1, color));
    }

    /// Clear source markers.
    pub fn clear_sources(&mut self) {
        self.sources.clear();
    }

    /// Set the listener head position.
    pub fn set_listener(&mut self, position: Position3D, scale: f32) {
        self.listener = Some(HeadWireframe::new(position, scale));
    }

    /// Clear the listener.
    pub fn clear_listener(&mut self) {
        self.listener = None;
    }

    /// Update the camera uniform buffer.
    fn update_camera_uniform(&self) {
        let mut uniform = CameraUniform::new();
        uniform.update(&self.camera, self.grid_size);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    /// Create depth texture (static method for use in constructor).
    fn create_depth_texture_static(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Update a cached vertex buffer, recreating only if size changed significantly.
    fn update_vertex_buffer_static(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[Vertex3D],
        buffer: &mut Option<wgpu::Buffer>,
        vertex_count: &mut u32,
        label: &str,
    ) {
        let new_count = vertices.len() as u32;
        let data = bytemuck::cast_slice(vertices);
        let required_size = data.len() as u64;

        // Check if we need to recreate the buffer
        let needs_recreate = match buffer {
            Some(ref existing) => existing.size() < required_size,
            None => true,
        };

        if needs_recreate && !vertices.is_empty() {
            // Allocate with some extra capacity to reduce reallocations
            let alloc_size = (required_size as f64 * 1.5) as u64;
            *buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: alloc_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        // Write data to buffer
        if let Some(ref buf) = buffer {
            if !vertices.is_empty() {
                queue.write_buffer(buf, 0, data);
            }
        }

        *vertex_count = new_count;
    }

    /// Render a frame.
    pub fn render(
        &mut self,
        grid: &crate::simulation::SimulationGrid3D,
    ) -> Result<(), RendererError> {
        // Get surface texture
        let output = self
            .surface
            .get_current_texture()
            .map_err(|e| RendererError::SurfaceError(e.to_string()))?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Update camera
        self.update_camera_uniform();

        // Update max pressure
        if self.config.auto_scale {
            self.slice_renderer.set_max_pressure(grid.max_pressure());
        } else {
            self.slice_renderer
                .set_max_pressure(self.config.max_pressure);
        }

        // Generate vertices
        let slice_vertices = self.slice_renderer.generate_vertices(grid, self.grid_size);

        let mut line_vertices = Vec::new();

        // Bounding box
        if self.config.show_bounding_box {
            let grid_lines = GridLines::new(self.grid_size);
            line_vertices.extend(grid_lines.generate_box());
        }

        // Floor grid
        if self.config.show_floor_grid {
            let grid_lines = GridLines::new(self.grid_size);
            line_vertices.extend(grid_lines.generate_floor_grid());
        }

        // Source markers
        let mut marker_vertices = Vec::new();
        if self.config.show_sources {
            for source in &self.sources {
                marker_vertices.extend(source.generate_vertices(16));
            }
        }

        // Listener head
        if self.config.show_listener {
            if let Some(ref head) = self.listener {
                line_vertices.extend(head.generate_vertices());
            }
        }

        // Update cached vertex buffers (recreate only if size changed significantly)
        Self::update_vertex_buffer_static(
            &self.device,
            &self.queue,
            &slice_vertices,
            &mut self.slice_buffer,
            &mut self.slice_vertex_count,
            "slice_vertex_buffer",
        );
        Self::update_vertex_buffer_static(
            &self.device,
            &self.queue,
            &line_vertices,
            &mut self.line_buffer,
            &mut self.line_vertex_count,
            "line_vertex_buffer",
        );
        Self::update_vertex_buffer_static(
            &self.device,
            &self.queue,
            &marker_vertices,
            &mut self.marker_buffer,
            &mut self.marker_vertex_count,
            "marker_vertex_buffer",
        );

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        // Update volume texture if in volume mode
        if self.config.mode == VisualizationMode::VolumeRender {
            if let Some(ref mut volume_renderer) = self.volume_renderer {
                volume_renderer.update_volume(&self.queue, grid);
                volume_renderer.update_params(&self.queue);
            }
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.config.background_color[0] as f64,
                            g: self.config.background_color[1] as f64,
                            b: self.config.background_color[2] as f64,
                            a: self.config.background_color[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw based on visualization mode
            match self.config.mode {
                VisualizationMode::VolumeRender => {
                    // Draw volume
                    if let Some(ref volume_renderer) = self.volume_renderer {
                        volume_renderer.render(&mut render_pass, &self.camera_bind_group);
                    }
                }
                _ => {
                    // Draw slices (slice modes)
                    if self.slice_vertex_count > 0 {
                        if let Some(ref buffer) = self.slice_buffer {
                            render_pass.set_pipeline(&self.triangle_pipeline);
                            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                            render_pass.set_vertex_buffer(0, buffer.slice(..));
                            render_pass.draw(0..self.slice_vertex_count, 0..1);
                        }
                    }
                }
            }

            // Draw markers
            if self.marker_vertex_count > 0 {
                if let Some(ref buffer) = self.marker_buffer {
                    render_pass.set_pipeline(&self.triangle_pipeline);
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, buffer.slice(..));
                    render_pass.draw(0..self.marker_vertex_count, 0..1);
                }
            }

            // Draw lines
            if self.line_vertex_count > 0 {
                if let Some(ref buffer) = self.line_buffer {
                    render_pass.set_pipeline(&self.line_pipeline);
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, buffer.slice(..));
                    render_pass.draw(0..self.line_vertex_count, 0..1);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Get the device for external rendering.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get the queue for external rendering.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get the surface format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }
}

/// Renderer error types.
#[derive(Debug)]
pub enum RendererError {
    SurfaceError(String),
    AdapterError(String),
    DeviceError(String),
    ShaderError(String),
}

impl std::fmt::Display for RendererError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RendererError::SurfaceError(msg) => write!(f, "Surface error: {}", msg),
            RendererError::AdapterError(msg) => write!(f, "Adapter error: {}", msg),
            RendererError::DeviceError(msg) => write!(f, "Device error: {}", msg),
            RendererError::ShaderError(msg) => write!(f, "Shader error: {}", msg),
        }
    }
}

impl std::error::Error for RendererError {}
