//! Canvas widget for rendering the pressure grid.

use super::app::Message;
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
    /// Maximum pressure for color scaling.
    max_pressure: f32,
    /// Whether to show tile boundaries.
    show_tiles: bool,
}

impl GridCanvas {
    /// Create a new canvas with the given pressure grid.
    pub fn new(pressure_grid: Vec<Vec<f32>>) -> Self {
        Self {
            cache: Cache::new(),
            pressure_grid,
            max_pressure: 1.0,
            show_tiles: true,
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
        self.pressure_grid = pressure_grid;
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
                    let color = self.pressure_to_color(pressure);

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
                let tiles_x = (cols + TILE_SIZE - 1) / TILE_SIZE;
                let tiles_y = (rows + TILE_SIZE - 1) / TILE_SIZE;

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
        });

        vec![geometry]
    }

    fn update(
        &self,
        _state: &mut Self::State,
        event: canvas::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> (canvas::event::Status, Option<Message>) {
        if let canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) = event {
            if let Some(position) = cursor.position_in(bounds) {
                // Convert to normalized coordinates [0, 1]
                let x = position.x / bounds.width;
                let y = position.y / bounds.height;

                return (
                    canvas::event::Status::Captured,
                    Some(Message::CanvasClick(x, y)),
                );
            }
        }

        (canvas::event::Status::Ignored, None)
    }
}
