//! 3D visualization module for acoustic wave simulation.
//!
//! Provides:
//! - 3D slice rendering (XY, XZ, YZ planes)
//! - Volume rendering with ray marching
//! - Interactive camera controls
//! - Source and listener visualization

pub mod camera;
pub mod renderer;
pub mod slice;
pub mod volume;

pub use camera::{Camera3D, CameraController, MouseButton};
pub use renderer::{RenderConfig, Renderer3D, VisualizationMode};
pub use slice::{SliceAxis, SliceConfig, SliceRenderer};
pub use volume::{VolumeParams, VolumeRenderer};

use crate::simulation::physics::Position3D;
use glam::Mat4;

/// Color mapping for pressure values.
#[derive(Debug, Clone, Copy)]
pub enum ColorMap {
    /// Blue (negative) - White - Red (positive)
    BlueWhiteRed,
    /// Black - Hot (fire colors)
    Hot,
    /// Purple - White - Green
    Diverging,
    /// Grayscale
    Grayscale,
    /// Cool to warm (blue - red through white)
    CoolWarm,
}

impl ColorMap {
    /// Map a pressure value (-1 to 1) to RGBA color.
    pub fn map(&self, value: f32, max_abs: f32) -> [f32; 4] {
        let normalized = (value / max_abs.max(0.001)).clamp(-1.0, 1.0);

        match self {
            ColorMap::BlueWhiteRed => {
                if normalized >= 0.0 {
                    // White to Red
                    [1.0, 1.0 - normalized, 1.0 - normalized, 1.0]
                } else {
                    // Blue to White
                    [1.0 + normalized, 1.0 + normalized, 1.0, 1.0]
                }
            }
            ColorMap::Hot => {
                let abs_val = normalized.abs();
                if abs_val < 0.33 {
                    [abs_val * 3.0, 0.0, 0.0, 1.0]
                } else if abs_val < 0.66 {
                    [1.0, (abs_val - 0.33) * 3.0, 0.0, 1.0]
                } else {
                    [1.0, 1.0, (abs_val - 0.66) * 3.0, 1.0]
                }
            }
            ColorMap::Diverging => {
                if normalized >= 0.0 {
                    [1.0 - normalized * 0.5, 1.0, 1.0 - normalized, 1.0]
                } else {
                    [1.0, 1.0 + normalized * 0.5, 1.0 + normalized, 1.0]
                }
            }
            ColorMap::Grayscale => {
                let gray = (normalized + 1.0) * 0.5;
                [gray, gray, gray, 1.0]
            }
            ColorMap::CoolWarm => {
                // Moreland's CoolWarm colormap
                let t = (normalized + 1.0) * 0.5;
                let r = 0.230 + t * (0.706 - 0.230);
                let g = 0.299 + t * (0.016 - 0.299);
                let b = 0.754 + t * (0.150 - 0.754);
                [r, g, b, 1.0]
            }
        }
    }

    /// Map with alpha based on magnitude.
    pub fn map_with_alpha(&self, value: f32, max_abs: f32, alpha_scale: f32) -> [f32; 4] {
        let mut color = self.map(value, max_abs);
        color[3] = (value.abs() / max_abs.max(0.001)).clamp(0.0, 1.0) * alpha_scale;
        color
    }
}

/// Vertex for 3D rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex3D {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub normal: [f32; 3],
    pub tex_coord: [f32; 2],
}

impl Vertex3D {
    pub fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self {
            position,
            color,
            normal: [0.0, 1.0, 0.0],
            tex_coord: [0.0, 0.0],
        }
    }

    pub fn with_normal(mut self, normal: [f32; 3]) -> Self {
        self.normal = normal;
        self
    }

    pub fn with_tex_coord(mut self, tex_coord: [f32; 2]) -> Self {
        self.tex_coord = tex_coord;
        self
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 40,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Uniform buffer for camera/projection data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub grid_size: [f32; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            view: Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.0, 0.0, 1.0],
            grid_size: [1.0, 1.0, 1.0, 1.0],
        }
    }

    pub fn update(&mut self, camera: &Camera3D, grid_size: (f32, f32, f32)) {
        self.view_proj = camera.view_projection_matrix().to_cols_array_2d();
        self.view = camera.view_matrix().to_cols_array_2d();
        let pos = camera.position();
        self.camera_pos = [pos.x, pos.y, pos.z, 1.0];
        self.grid_size = [grid_size.0, grid_size.1, grid_size.2, 1.0];
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::new()
    }
}

/// Marker sphere for sources and listeners.
pub struct MarkerSphere {
    pub position: Position3D,
    pub radius: f32,
    pub color: [f32; 4],
}

impl MarkerSphere {
    pub fn new(position: Position3D, radius: f32, color: [f32; 4]) -> Self {
        Self {
            position,
            radius,
            color,
        }
    }

    /// Generate sphere vertices.
    pub fn generate_vertices(&self, segments: u32) -> Vec<Vertex3D> {
        let mut vertices = Vec::new();

        for i in 0..=segments {
            let lat = std::f32::consts::PI * i as f32 / segments as f32;
            let sin_lat = lat.sin();
            let cos_lat = lat.cos();

            for j in 0..=segments {
                let lon = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
                let sin_lon = lon.sin();
                let cos_lon = lon.cos();

                let x = self.position.x + self.radius * sin_lat * cos_lon;
                let y = self.position.y + self.radius * cos_lat;
                let z = self.position.z + self.radius * sin_lat * sin_lon;

                let nx = sin_lat * cos_lon;
                let ny = cos_lat;
                let nz = sin_lat * sin_lon;

                vertices.push(
                    Vertex3D::new([x, y, z], self.color)
                        .with_normal([nx, ny, nz])
                        .with_tex_coord([j as f32 / segments as f32, i as f32 / segments as f32]),
                );
            }
        }

        vertices
    }

    /// Generate sphere indices.
    pub fn generate_indices(&self, segments: u32) -> Vec<u32> {
        let mut indices = Vec::new();
        let row_length = segments + 1;

        for i in 0..segments {
            for j in 0..segments {
                let a = i * row_length + j;
                let b = a + row_length;

                indices.push(a);
                indices.push(b);
                indices.push(a + 1);

                indices.push(a + 1);
                indices.push(b);
                indices.push(b + 1);
            }
        }

        indices
    }
}

/// Grid lines for visualization.
pub struct GridLines {
    pub size: (f32, f32, f32),
    pub divisions: (u32, u32, u32),
    pub color: [f32; 4],
}

impl GridLines {
    pub fn new(size: (f32, f32, f32)) -> Self {
        Self {
            size,
            divisions: (10, 10, 10),
            color: [0.3, 0.3, 0.3, 1.0],
        }
    }

    /// Generate grid line vertices (XY plane at z=0).
    pub fn generate_floor_grid(&self) -> Vec<Vertex3D> {
        let mut vertices = Vec::new();
        let (sx, _, sz) = self.size;
        let (dx, _, dz) = self.divisions;

        // X-direction lines
        for i in 0..=dz {
            let z = i as f32 * sz / dz as f32;
            vertices.push(Vertex3D::new([0.0, 0.0, z], self.color));
            vertices.push(Vertex3D::new([sx, 0.0, z], self.color));
        }

        // Z-direction lines
        for i in 0..=dx {
            let x = i as f32 * sx / dx as f32;
            vertices.push(Vertex3D::new([x, 0.0, 0.0], self.color));
            vertices.push(Vertex3D::new([x, 0.0, sz], self.color));
        }

        vertices
    }

    /// Generate a bounding box wireframe.
    pub fn generate_box(&self) -> Vec<Vertex3D> {
        let mut vertices = Vec::new();
        let (sx, sy, sz) = self.size;

        // Bottom face
        vertices.extend([
            Vertex3D::new([0.0, 0.0, 0.0], self.color),
            Vertex3D::new([sx, 0.0, 0.0], self.color),
            Vertex3D::new([sx, 0.0, 0.0], self.color),
            Vertex3D::new([sx, 0.0, sz], self.color),
            Vertex3D::new([sx, 0.0, sz], self.color),
            Vertex3D::new([0.0, 0.0, sz], self.color),
            Vertex3D::new([0.0, 0.0, sz], self.color),
            Vertex3D::new([0.0, 0.0, 0.0], self.color),
        ]);

        // Top face
        vertices.extend([
            Vertex3D::new([0.0, sy, 0.0], self.color),
            Vertex3D::new([sx, sy, 0.0], self.color),
            Vertex3D::new([sx, sy, 0.0], self.color),
            Vertex3D::new([sx, sy, sz], self.color),
            Vertex3D::new([sx, sy, sz], self.color),
            Vertex3D::new([0.0, sy, sz], self.color),
            Vertex3D::new([0.0, sy, sz], self.color),
            Vertex3D::new([0.0, sy, 0.0], self.color),
        ]);

        // Vertical edges
        vertices.extend([
            Vertex3D::new([0.0, 0.0, 0.0], self.color),
            Vertex3D::new([0.0, sy, 0.0], self.color),
            Vertex3D::new([sx, 0.0, 0.0], self.color),
            Vertex3D::new([sx, sy, 0.0], self.color),
            Vertex3D::new([sx, 0.0, sz], self.color),
            Vertex3D::new([sx, sy, sz], self.color),
            Vertex3D::new([0.0, 0.0, sz], self.color),
            Vertex3D::new([0.0, sy, sz], self.color),
        ]);

        vertices
    }
}

/// Head wireframe for visualization.
pub struct HeadWireframe {
    pub position: Position3D,
    pub scale: f32,
    pub yaw: f32,
}

impl HeadWireframe {
    pub fn new(position: Position3D, scale: f32) -> Self {
        Self {
            position,
            scale,
            yaw: 0.0,
        }
    }

    /// Generate simplified head shape vertices (ellipsoid + ears).
    pub fn generate_vertices(&self) -> Vec<Vertex3D> {
        let mut vertices = Vec::new();
        let head_color = [0.2, 0.8, 0.2, 1.0];
        let ear_color = [0.8, 0.2, 0.2, 1.0];

        // Generate ellipsoid for head
        let segments = 16u32;
        let head_width = 0.15 * self.scale;
        let head_height = 0.22 * self.scale;
        let head_depth = 0.18 * self.scale;

        for i in 0..=segments {
            let lat = std::f32::consts::PI * i as f32 / segments as f32;

            for j in 0..segments {
                let lon1 = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
                let lon2 = 2.0 * std::f32::consts::PI * (j + 1) as f32 / segments as f32;

                let y1 = self.position.y + head_height * lat.cos();
                let x1 = self.position.x + head_width * lat.sin() * lon1.cos();
                let z1 = self.position.z + head_depth * lat.sin() * lon1.sin();

                let x2 = self.position.x + head_width * lat.sin() * lon2.cos();
                let z2 = self.position.z + head_depth * lat.sin() * lon2.sin();

                vertices.push(Vertex3D::new([x1, y1, z1], head_color));
                vertices.push(Vertex3D::new([x2, y1, z2], head_color));
            }
        }

        // Ear markers
        let ear_offset = 0.085 * self.scale;
        let ear_radius = 0.03 * self.scale;

        // Left ear
        for i in 0..=8 {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / 8.0;
            let x = self.position.x - ear_offset;
            let y = self.position.y + ear_radius * angle.cos();
            let z = self.position.z + ear_radius * angle.sin();
            vertices.push(Vertex3D::new([x, y, z], ear_color));
        }

        // Right ear
        for i in 0..=8 {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / 8.0;
            let x = self.position.x + ear_offset;
            let y = self.position.y + ear_radius * angle.cos();
            let z = self.position.z + ear_radius * angle.sin();
            vertices.push(Vertex3D::new([x, y, z], ear_color));
        }

        // Nose direction indicator
        let nose_length = 0.15 * self.scale;
        vertices.push(Vertex3D::new(
            [self.position.x, self.position.y, self.position.z],
            [1.0, 1.0, 0.0, 1.0],
        ));
        vertices.push(Vertex3D::new(
            [
                self.position.x,
                self.position.y,
                self.position.z + nose_length,
            ],
            [1.0, 1.0, 0.0, 1.0],
        ));

        vertices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_map() {
        let map = ColorMap::BlueWhiteRed;

        // Zero should be white
        let white = map.map(0.0, 1.0);
        assert!((white[0] - 1.0).abs() < 0.01);
        assert!((white[1] - 1.0).abs() < 0.01);
        assert!((white[2] - 1.0).abs() < 0.01);

        // Positive should be reddish
        let red = map.map(1.0, 1.0);
        assert!(red[0] > red[1]);
        assert!(red[0] > red[2]);

        // Negative should be bluish
        let blue = map.map(-1.0, 1.0);
        assert!(blue[2] > blue[0]);
        assert!(blue[2] > blue[1]);
    }

    #[test]
    fn test_sphere_generation() {
        let sphere = MarkerSphere::new(Position3D::origin(), 1.0, [1.0, 0.0, 0.0, 1.0]);

        let vertices = sphere.generate_vertices(8);
        let indices = sphere.generate_indices(8);

        assert!(!vertices.is_empty());
        assert!(!indices.is_empty());
    }

    #[test]
    fn test_grid_lines() {
        let grid = GridLines::new((10.0, 5.0, 10.0));

        let floor = grid.generate_floor_grid();
        let bbox = grid.generate_box();

        assert!(!floor.is_empty());
        assert!(!bbox.is_empty());
        // Bounding box should have 24 vertices (12 edges * 2 endpoints)
        assert_eq!(bbox.len(), 24);
    }
}
