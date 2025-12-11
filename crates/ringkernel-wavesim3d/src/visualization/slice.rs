//! Slice rendering for 3D volume visualization.
//!
//! Renders 2D slices through the 3D pressure field.

use super::{ColorMap, Vertex3D};
use crate::simulation::SimulationGrid3D;

/// Which axis the slice is perpendicular to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SliceAxis {
    /// XY plane (perpendicular to Z)
    #[default]
    XY,
    /// XZ plane (perpendicular to Y)
    XZ,
    /// YZ plane (perpendicular to X)
    YZ,
}

/// Configuration for a slice through the volume.
#[derive(Debug, Clone)]
pub struct SliceConfig {
    /// Which axis the slice is perpendicular to
    pub axis: SliceAxis,
    /// Position along the axis (0.0 to 1.0)
    pub position: f32,
    /// Opacity of the slice
    pub opacity: f32,
    /// Color map for rendering
    pub color_map: ColorMap,
    /// Whether to show grid lines on the slice
    pub show_grid: bool,
}

impl Default for SliceConfig {
    fn default() -> Self {
        Self {
            axis: SliceAxis::XY,
            position: 0.5,
            opacity: 0.8,
            color_map: ColorMap::BlueWhiteRed,
            show_grid: false,
        }
    }
}

impl SliceConfig {
    pub fn new(axis: SliceAxis, position: f32) -> Self {
        Self {
            axis,
            position: position.clamp(0.0, 1.0),
            ..Default::default()
        }
    }
}

/// Renderer for 2D slices through 3D data.
pub struct SliceRenderer {
    /// Active slices
    pub slices: Vec<SliceConfig>,
    /// Maximum pressure value for normalization
    max_pressure: f32,
}

impl Default for SliceRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl SliceRenderer {
    pub fn new() -> Self {
        Self {
            slices: vec![SliceConfig::default()],
            max_pressure: 1.0,
        }
    }

    /// Add a slice configuration.
    pub fn add_slice(&mut self, config: SliceConfig) {
        self.slices.push(config);
    }

    /// Remove a slice by index.
    pub fn remove_slice(&mut self, index: usize) {
        if index < self.slices.len() {
            self.slices.remove(index);
        }
    }

    /// Clear all slices.
    pub fn clear(&mut self) {
        self.slices.clear();
    }

    /// Set the maximum pressure for color normalization.
    pub fn set_max_pressure(&mut self, max: f32) {
        self.max_pressure = max.max(0.001);
    }

    /// Generate vertices for all slices.
    pub fn generate_vertices(
        &self,
        grid: &SimulationGrid3D,
        grid_physical_size: (f32, f32, f32),
    ) -> Vec<Vertex3D> {
        let mut vertices = Vec::new();

        for config in &self.slices {
            vertices.extend(self.generate_slice_vertices(grid, grid_physical_size, config));
        }

        vertices
    }

    /// Generate vertices for a single slice.
    pub fn generate_slice_vertices(
        &self,
        grid: &SimulationGrid3D,
        grid_physical_size: (f32, f32, f32),
        config: &SliceConfig,
    ) -> Vec<Vertex3D> {
        let (phys_w, phys_h, phys_d) = grid_physical_size;
        let (w, h, d) = (grid.width, grid.height, grid.depth);

        match config.axis {
            SliceAxis::XY => {
                let z_idx = ((d as f32 - 1.0) * config.position) as usize;
                let z_pos = phys_d * config.position;
                let slice_data = grid.get_xy_slice(z_idx);

                self.create_quad_grid(
                    &slice_data,
                    w,
                    h,
                    |x, y| [x * phys_w / w as f32, y * phys_h / h as f32, z_pos],
                    [0.0, 0.0, 1.0],
                    config,
                )
            }
            SliceAxis::XZ => {
                let y_idx = ((h as f32 - 1.0) * config.position) as usize;
                let y_pos = phys_h * config.position;
                let slice_data = grid.get_xz_slice(y_idx);

                self.create_quad_grid(
                    &slice_data,
                    w,
                    d,
                    |x, z| [x * phys_w / w as f32, y_pos, z * phys_d / d as f32],
                    [0.0, 1.0, 0.0],
                    config,
                )
            }
            SliceAxis::YZ => {
                let x_idx = ((w as f32 - 1.0) * config.position) as usize;
                let x_pos = phys_w * config.position;
                let slice_data = grid.get_yz_slice(x_idx);

                self.create_quad_grid(
                    &slice_data,
                    h,
                    d,
                    |y, z| [x_pos, y * phys_h / h as f32, z * phys_d / d as f32],
                    [1.0, 0.0, 0.0],
                    config,
                )
            }
        }
    }

    /// Create a grid of quads from slice data.
    fn create_quad_grid<F>(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
        pos_fn: F,
        normal: [f32; 3],
        config: &SliceConfig,
    ) -> Vec<Vertex3D>
    where
        F: Fn(f32, f32) -> [f32; 3],
    {
        let mut vertices = Vec::with_capacity((width - 1) * (height - 1) * 6);

        for y in 0..height - 1 {
            for x in 0..width - 1 {
                // Four corners of the quad
                let idx00 = y * width + x;
                let idx10 = y * width + x + 1;
                let idx01 = (y + 1) * width + x;
                let idx11 = (y + 1) * width + x + 1;

                // Get pressure values and colors
                let p00 = data.get(idx00).copied().unwrap_or(0.0);
                let p10 = data.get(idx10).copied().unwrap_or(0.0);
                let p01 = data.get(idx01).copied().unwrap_or(0.0);
                let p11 = data.get(idx11).copied().unwrap_or(0.0);

                let mut c00 = config.color_map.map(p00, self.max_pressure);
                let mut c10 = config.color_map.map(p10, self.max_pressure);
                let mut c01 = config.color_map.map(p01, self.max_pressure);
                let mut c11 = config.color_map.map(p11, self.max_pressure);

                // Apply opacity
                c00[3] *= config.opacity;
                c10[3] *= config.opacity;
                c01[3] *= config.opacity;
                c11[3] *= config.opacity;

                // Positions
                let pos00 = pos_fn(x as f32, y as f32);
                let pos10 = pos_fn(x as f32 + 1.0, y as f32);
                let pos01 = pos_fn(x as f32, y as f32 + 1.0);
                let pos11 = pos_fn(x as f32 + 1.0, y as f32 + 1.0);

                // Two triangles for the quad
                vertices.push(Vertex3D::new(pos00, c00).with_normal(normal));
                vertices.push(Vertex3D::new(pos10, c10).with_normal(normal));
                vertices.push(Vertex3D::new(pos01, c01).with_normal(normal));

                vertices.push(Vertex3D::new(pos10, c10).with_normal(normal));
                vertices.push(Vertex3D::new(pos11, c11).with_normal(normal));
                vertices.push(Vertex3D::new(pos01, c01).with_normal(normal));
            }
        }

        vertices
    }

    /// Generate grid line vertices for slices.
    pub fn generate_grid_lines(
        &self,
        grid_physical_size: (f32, f32, f32),
        divisions: u32,
    ) -> Vec<Vertex3D> {
        let mut vertices = Vec::new();
        let (phys_w, phys_h, phys_d) = grid_physical_size;
        let line_color = [0.5, 0.5, 0.5, 0.3];

        for config in &self.slices {
            if !config.show_grid {
                continue;
            }

            match config.axis {
                SliceAxis::XY => {
                    let z = phys_d * config.position;
                    for i in 0..=divisions {
                        let t = i as f32 / divisions as f32;
                        // Horizontal lines
                        vertices.push(Vertex3D::new([0.0, t * phys_h, z], line_color));
                        vertices.push(Vertex3D::new([phys_w, t * phys_h, z], line_color));
                        // Vertical lines
                        vertices.push(Vertex3D::new([t * phys_w, 0.0, z], line_color));
                        vertices.push(Vertex3D::new([t * phys_w, phys_h, z], line_color));
                    }
                }
                SliceAxis::XZ => {
                    let y = phys_h * config.position;
                    for i in 0..=divisions {
                        let t = i as f32 / divisions as f32;
                        // X lines
                        vertices.push(Vertex3D::new([0.0, y, t * phys_d], line_color));
                        vertices.push(Vertex3D::new([phys_w, y, t * phys_d], line_color));
                        // Z lines
                        vertices.push(Vertex3D::new([t * phys_w, y, 0.0], line_color));
                        vertices.push(Vertex3D::new([t * phys_w, y, phys_d], line_color));
                    }
                }
                SliceAxis::YZ => {
                    let x = phys_w * config.position;
                    for i in 0..=divisions {
                        let t = i as f32 / divisions as f32;
                        // Y lines
                        vertices.push(Vertex3D::new([x, 0.0, t * phys_d], line_color));
                        vertices.push(Vertex3D::new([x, phys_h, t * phys_d], line_color));
                        // Z lines
                        vertices.push(Vertex3D::new([x, t * phys_h, 0.0], line_color));
                        vertices.push(Vertex3D::new([x, t * phys_h, phys_d], line_color));
                    }
                }
            }
        }

        vertices
    }
}

/// Multi-slice configuration for volume exploration.
pub struct MultiSliceView {
    /// XY slice position (0-1)
    pub xy_position: f32,
    /// XZ slice position (0-1)
    pub xz_position: f32,
    /// YZ slice position (0-1)
    pub yz_position: f32,
    /// Show XY slice
    pub show_xy: bool,
    /// Show XZ slice
    pub show_xz: bool,
    /// Show YZ slice
    pub show_yz: bool,
    /// Color map
    pub color_map: ColorMap,
    /// Opacity
    pub opacity: f32,
}

impl Default for MultiSliceView {
    fn default() -> Self {
        Self {
            xy_position: 0.5,
            xz_position: 0.5,
            yz_position: 0.5,
            show_xy: true,
            show_xz: false,
            show_yz: false,
            color_map: ColorMap::BlueWhiteRed,
            opacity: 0.8,
        }
    }
}

impl MultiSliceView {
    /// Convert to slice renderer configuration.
    pub fn to_slice_configs(&self) -> Vec<SliceConfig> {
        let mut configs = Vec::new();

        if self.show_xy {
            configs.push(SliceConfig {
                axis: SliceAxis::XY,
                position: self.xy_position,
                opacity: self.opacity,
                color_map: self.color_map,
                show_grid: false,
            });
        }

        if self.show_xz {
            configs.push(SliceConfig {
                axis: SliceAxis::XZ,
                position: self.xz_position,
                opacity: self.opacity,
                color_map: self.color_map,
                show_grid: false,
            });
        }

        if self.show_yz {
            configs.push(SliceConfig {
                axis: SliceAxis::YZ,
                position: self.yz_position,
                opacity: self.opacity,
                color_map: self.color_map,
                show_grid: false,
            });
        }

        configs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::physics::{AcousticParams3D, Environment};

    fn create_test_grid() -> SimulationGrid3D {
        let params = AcousticParams3D::new(Environment::default(), 0.1);
        SimulationGrid3D::new(16, 16, 16, params)
    }

    #[test]
    fn test_slice_config() {
        let config = SliceConfig::default();
        assert_eq!(config.axis, SliceAxis::XY);
        assert!((config.position - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_slice_renderer() {
        let grid = create_test_grid();
        let renderer = SliceRenderer::new();

        let vertices = renderer.generate_vertices(&grid, (1.6, 1.6, 1.6));
        assert!(!vertices.is_empty());
    }

    #[test]
    fn test_multi_slice_view() {
        let view = MultiSliceView::default();
        let configs = view.to_slice_configs();

        // Default should have only XY slice
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].axis, SliceAxis::XY);
    }

    #[test]
    fn test_all_axes() {
        let grid = create_test_grid();
        let mut renderer = SliceRenderer::new();

        renderer.clear();
        renderer.add_slice(SliceConfig::new(SliceAxis::XY, 0.5));
        renderer.add_slice(SliceConfig::new(SliceAxis::XZ, 0.5));
        renderer.add_slice(SliceConfig::new(SliceAxis::YZ, 0.5));

        let vertices = renderer.generate_vertices(&grid, (1.6, 1.6, 1.6));
        assert!(vertices.len() > 100); // Should have many vertices
    }
}
