//! Canvas widget for rendering the pressure grid.

use super::app::Message;
use crate::simulation::CellType;
use iced::mouse;
use iced::widget::canvas::{self, Cache, Geometry};
use iced::{Color, Point, Rectangle, Renderer, Size, Theme};

/// Default tile size for visualization (matches CUDA Packed backend).
const TILE_SIZE: usize = 16;

/// Canvas for visualizing the pressure grid.
pub struct GridCanvas {
    /// Cache for the rendered grid.
    cache: Cache,
    /// Current pressure values.
    pressure_grid: Vec<Vec<f32>>,
    /// Cell types for drawing mode.
    cell_types: Vec<Vec<CellType>>,
    /// Maximum pressure for color scaling.
    max_pressure: f32,
    /// Whether to show tile boundaries.
    show_tiles: bool,
    /// Cells currently being processed (for educational modes).
    processing_cells: Vec<(u32, u32)>,
    /// Active tiles being processed (for educational modes).
    active_tiles: Vec<(u32, u32)>,
    /// Current row being processed (for RowByRow mode).
    current_row: Option<u32>,
}

impl GridCanvas {
    /// Create a new canvas with the given pressure grid.
    pub fn new(pressure_grid: Vec<Vec<f32>>) -> Self {
        let rows = pressure_grid.len();
        let cols = if rows > 0 { pressure_grid[0].len() } else { 0 };
        Self {
            cache: Cache::new(),
            pressure_grid,
            cell_types: vec![vec![CellType::Normal; cols]; rows],
            max_pressure: 1.0,
            show_tiles: true,
            processing_cells: Vec::new(),
            active_tiles: Vec::new(),
            current_row: None,
        }
    }

    /// Set whether to show tile boundaries.
    #[allow(dead_code)]
    pub fn set_show_tiles(&mut self, show: bool) {
        if self.show_tiles != show {
            self.show_tiles = show;
            self.cache.clear();
        }
    }

    /// Update the pressure grid and invalidate the cache.
    pub fn update_pressure(&mut self, pressure_grid: Vec<Vec<f32>>) {
        // Update max pressure for auto-scaling
        let max = pressure_grid
            .iter()
            .flat_map(|row| row.iter())
            .fold(0.0f32, |acc, &p| acc.max(p.abs()));

        // Use adaptive scaling with a minimum threshold
        self.max_pressure = max.max(0.1);

        // Resize cell_types if grid size changed
        let rows = pressure_grid.len();
        let cols = if rows > 0 { pressure_grid[0].len() } else { 0 };
        if self.cell_types.len() != rows || (rows > 0 && self.cell_types[0].len() != cols) {
            self.cell_types = vec![vec![CellType::Normal; cols]; rows];
        }

        self.pressure_grid = pressure_grid;
        self.cache.clear();
    }

    /// Update cell types and invalidate the cache.
    pub fn update_cell_types(&mut self, cell_types: Vec<Vec<CellType>>) {
        self.cell_types = cell_types;
        self.cache.clear();
    }

    /// Set the current processing state for educational mode visualization.
    pub fn set_processing_state(
        &mut self,
        cells: Vec<(u32, u32)>,
        tiles: Vec<(u32, u32)>,
        row: Option<u32>,
    ) {
        self.processing_cells = cells;
        self.active_tiles = tiles;
        self.current_row = row;
        self.cache.clear();
    }

    /// Convert pressure value to color.
    ///
    /// Uses a diverging color scheme:
    /// - Negative pressure: Blue
    /// - Zero pressure: Black/Dark Gray
    /// - Positive pressure: Red/Orange
    fn pressure_to_color(&self, pressure: f32) -> Color {
        // Normalize to [-1, 1] range using adaptive scaling
        let normalized = (pressure / self.max_pressure).clamp(-1.0, 1.0);

        if normalized >= 0.0 {
            // Positive: Black -> Red -> Yellow
            let t = normalized;
            if t < 0.5 {
                // Black to Red
                let t2 = t * 2.0;
                Color::from_rgb(t2, 0.0, 0.0)
            } else {
                // Red to Yellow
                let t2 = (t - 0.5) * 2.0;
                Color::from_rgb(1.0, t2, 0.0)
            }
        } else {
            // Negative: Black -> Blue -> Cyan
            let t = -normalized;
            if t < 0.5 {
                // Black to Blue
                let t2 = t * 2.0;
                Color::from_rgb(0.0, 0.0, t2)
            } else {
                // Blue to Cyan
                let t2 = (t - 0.5) * 2.0;
                Color::from_rgb(0.0, t2, 1.0)
            }
        }
    }
}

impl canvas::Program<Message> for GridCanvas {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
            // Background
            frame.fill_rectangle(
                Point::ORIGIN,
                bounds.size(),
                Color::from_rgb(0.05, 0.05, 0.05),
            );

            if self.pressure_grid.is_empty() || self.pressure_grid[0].is_empty() {
                return;
            }

            let rows = self.pressure_grid.len();
            let cols = self.pressure_grid[0].len();

            let cell_width = bounds.width / cols as f32;
            let cell_height = bounds.height / rows as f32;

            // Draw cells
            for (y, row) in self.pressure_grid.iter().enumerate() {
                for (x, &pressure) in row.iter().enumerate() {
                    // Get cell type if available
                    let cell_type = self
                        .cell_types
                        .get(y)
                        .and_then(|r| r.get(x))
                        .copied()
                        .unwrap_or(CellType::Normal);

                    let color = match cell_type {
                        CellType::Normal => self.pressure_to_color(pressure),
                        CellType::Absorber => {
                            // Dark purple for absorbers - blend with pressure
                            let base = self.pressure_to_color(pressure);
                            Color::from_rgb(base.r * 0.3 + 0.2, base.g * 0.1, base.b * 0.3 + 0.3)
                        }
                        CellType::Reflector => {
                            // Bright white/gray for reflectors - blend with pressure
                            let base = self.pressure_to_color(pressure);
                            Color::from_rgb(
                                base.r * 0.5 + 0.5,
                                base.g * 0.5 + 0.5,
                                base.b * 0.5 + 0.5,
                            )
                        }
                    };

                    frame.fill_rectangle(
                        Point::new(x as f32 * cell_width, y as f32 * cell_height),
                        Size::new(cell_width, cell_height),
                        color,
                    );
                }
            }

            use iced::widget::canvas::path::Builder;
            use iced::widget::canvas::Stroke;

            // Draw tile boundaries (always visible, slightly more prominent)
            if self.show_tiles {
                let tile_color = Color::from_rgba(1.0, 1.0, 1.0, 0.25);
                let mut tile_path = Builder::new();

                // Calculate tile boundaries
                let tiles_x = cols.div_ceil(TILE_SIZE);
                let tiles_y = rows.div_ceil(TILE_SIZE);

                // Vertical tile lines
                for tx in 1..tiles_x {
                    let x = tx * TILE_SIZE;
                    if x < cols {
                        let px = x as f32 * cell_width;
                        tile_path.move_to(Point::new(px, 0.0));
                        tile_path.line_to(Point::new(px, bounds.height));
                    }
                }

                // Horizontal tile lines
                for ty in 1..tiles_y {
                    let y = ty * TILE_SIZE;
                    if y < rows {
                        let py = y as f32 * cell_height;
                        tile_path.move_to(Point::new(0.0, py));
                        tile_path.line_to(Point::new(bounds.width, py));
                    }
                }

                frame.stroke(
                    &tile_path.build(),
                    Stroke::default().with_color(tile_color).with_width(1.5),
                );
            }

            // Draw cell grid lines (subtle, only when zoomed in)
            let grid_color = Color::from_rgba(1.0, 1.0, 1.0, 0.08);

            // Only draw cell grid lines if cells are large enough
            if cell_width > 12.0 && cell_height > 12.0 {
                let mut path = Builder::new();

                // Vertical lines (skip tile boundaries to avoid double-drawing)
                for x in 1..cols {
                    if self.show_tiles && x % TILE_SIZE == 0 {
                        continue; // Skip tile boundaries
                    }
                    let px = x as f32 * cell_width;
                    path.move_to(Point::new(px, 0.0));
                    path.line_to(Point::new(px, bounds.height));
                }

                // Horizontal lines (skip tile boundaries)
                for y in 1..rows {
                    if self.show_tiles && y % TILE_SIZE == 0 {
                        continue; // Skip tile boundaries
                    }
                    let py = y as f32 * cell_height;
                    path.move_to(Point::new(0.0, py));
                    path.line_to(Point::new(bounds.width, py));
                }

                frame.stroke(
                    &path.build(),
                    Stroke::default().with_color(grid_color).with_width(1.0),
                );
            }

            // Draw processing indicators for educational modes
            // Highlight processing cells with a bright green overlay
            if !self.processing_cells.is_empty() {
                let processing_color = Color::from_rgba(0.0, 1.0, 0.3, 0.6);
                for (cx, cy) in &self.processing_cells {
                    let px = *cx as f32 * cell_width;
                    let py = *cy as f32 * cell_height;
                    frame.fill_rectangle(
                        Point::new(px, py),
                        Size::new(cell_width, cell_height),
                        processing_color,
                    );
                }
            }

            // Highlight current row being processed (for RowByRow mode)
            if let Some(row) = self.current_row {
                let row_color = Color::from_rgba(1.0, 0.8, 0.0, 0.3);
                let py = row as f32 * cell_height;
                frame.fill_rectangle(
                    Point::new(0.0, py),
                    Size::new(bounds.width, cell_height),
                    row_color,
                );
            }

            // Highlight active tiles (for ActorBased mode)
            // Note: tiles in educational mode start at cell (1,1) to skip boundary
            if !self.active_tiles.is_empty() {
                let tile_highlight = Color::from_rgba(0.0, 0.8, 1.0, 0.25);
                let tile_border = Color::from_rgba(0.0, 1.0, 1.0, 0.8);

                for (tx, ty) in &self.active_tiles {
                    // Tile position: starts at cell 1 (boundary offset) + tile_index * tile_size
                    let start_cell_x = 1 + *tx as usize * TILE_SIZE;
                    let start_cell_y = 1 + *ty as usize * TILE_SIZE;
                    let px = start_cell_x as f32 * cell_width;
                    let py = start_cell_y as f32 * cell_height;

                    // Clamp tile size to not exceed grid interior
                    let end_cell_x = (start_cell_x + TILE_SIZE).min(cols - 1);
                    let end_cell_y = (start_cell_y + TILE_SIZE).min(rows - 1);
                    let tile_w = (end_cell_x - start_cell_x) as f32 * cell_width;
                    let tile_h = (end_cell_y - start_cell_y) as f32 * cell_height;

                    // Fill
                    frame.fill_rectangle(
                        Point::new(px, py),
                        Size::new(tile_w, tile_h),
                        tile_highlight,
                    );

                    // Border
                    let mut tile_border_path = Builder::new();
                    tile_border_path.move_to(Point::new(px, py));
                    tile_border_path.line_to(Point::new(px + tile_w, py));
                    tile_border_path.line_to(Point::new(px + tile_w, py + tile_h));
                    tile_border_path.line_to(Point::new(px, py + tile_h));
                    tile_border_path.line_to(Point::new(px, py));

                    frame.stroke(
                        &tile_border_path.build(),
                        Stroke::default().with_color(tile_border).with_width(2.5),
                    );
                }
            }
        });

        vec![geometry]
    }

    fn update(
        &self,
        _state: &mut Self::State,
        event: &canvas::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<canvas::Action<Message>> {
        match event {
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                if let Some(position) = cursor.position_in(bounds) {
                    let x = position.x / bounds.width;
                    let y = position.y / bounds.height;

                    return Some(canvas::Action::publish(Message::CanvasClick(x, y)));
                }
            }
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Right)) => {
                if let Some(position) = cursor.position_in(bounds) {
                    let x = position.x / bounds.width;
                    let y = position.y / bounds.height;

                    return Some(canvas::Action::publish(Message::CanvasRightClick(x, y)));
                }
            }
            _ => {}
        }

        None
    }
}
