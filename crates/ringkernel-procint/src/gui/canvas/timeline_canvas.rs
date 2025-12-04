//! Timeline canvas for partial order visualization.
//!
//! Shows concurrent activities and precedence relationships.

use super::Camera;
use crate::gui::Theme;
use crate::models::GpuPartialOrderTrace;
use eframe::egui::{self, Color32, Pos2, Rect, Rounding, Stroke, Vec2};
use std::collections::HashMap;

/// Timeline canvas for partial order visualization.
pub struct TimelineCanvas {
    /// Camera for pan/zoom.
    pub camera: Camera,
    /// Activity colors.
    activity_colors: HashMap<u32, Color32>,
    /// Activity names.
    activity_names: HashMap<u32, String>,
    /// Selected trace.
    selected_trace: Option<u64>,
    /// Pixels per second (base scale before zoom).
    pixels_per_sec: f32,
}

impl Default for TimelineCanvas {
    fn default() -> Self {
        Self::new()
    }
}

impl TimelineCanvas {
    /// Create a new timeline canvas.
    pub fn new() -> Self {
        Self {
            camera: Camera::default(),
            activity_colors: HashMap::new(),
            activity_names: HashMap::new(),
            selected_trace: None,
            pixels_per_sec: 0.01, // 0.01 pixels per second (activities are minutes/hours long)
        }
    }

    /// Set activity names.
    pub fn set_activity_names(&mut self, names: HashMap<u32, String>) {
        self.activity_names = names;
    }

    /// Generate colors for activities.
    fn ensure_colors(&mut self, activities: &[u32]) {
        let palette = [
            Color32::from_rgb(59, 130, 246), // Blue
            Color32::from_rgb(34, 197, 94),  // Green
            Color32::from_rgb(239, 68, 68),  // Red
            Color32::from_rgb(168, 85, 247), // Purple
            Color32::from_rgb(234, 179, 8),  // Yellow
            Color32::from_rgb(236, 72, 153), // Pink
            Color32::from_rgb(20, 184, 166), // Teal
            Color32::from_rgb(249, 115, 22), // Orange
        ];

        for (i, &activity_id) in activities.iter().enumerate() {
            self.activity_colors
                .entry(activity_id)
                .or_insert_with(|| palette[i % palette.len()]);
        }
    }

    /// Format time in human-readable format.
    fn format_time(seconds: f32) -> String {
        if seconds >= 3600.0 {
            let hours = seconds / 3600.0;
            format!("{:.1}h", hours)
        } else if seconds >= 60.0 {
            let mins = seconds / 60.0;
            format!("{:.1}m", mins)
        } else {
            format!("{:.0}s", seconds)
        }
    }

    /// Render the timeline.
    pub fn render(
        &mut self,
        ui: &mut egui::Ui,
        theme: &Theme,
        traces: &[GpuPartialOrderTrace],
        rect: Rect,
    ) {
        let painter = ui.painter_at(rect);

        // Handle input
        self.handle_input(ui, rect);

        // Update camera
        self.camera.update(ui.ctx().input(|i| i.stable_dt));

        // Draw background grid (with camera transform)
        self.draw_grid(&painter, theme, rect);

        // Draw header with trace count
        painter.text(
            Pos2::new(rect.left() + 10.0, rect.top() + 15.0),
            egui::Align2::LEFT_CENTER,
            format!("Partial Order Timeline - {} traces", traces.len()),
            egui::FontId::proportional(14.0),
            theme.text,
        );

        // Draw traces
        let trace_height = 80.0;
        let header_height = 40.0;
        let mut y_offset = header_height;

        if traces.is_empty() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No partial order traces yet.\nStart the pipeline to see execution patterns.",
                egui::FontId::proportional(14.0),
                theme.text_muted,
            );
        } else {
            for trace in traces.iter().take(6) {
                // Limit to 6 traces for performance
                if trace.activity_count > 0 {
                    self.draw_trace(&painter, theme, trace, rect, y_offset, trace_height);
                    y_offset += trace_height + 10.0;
                }
            }
        }

        // Draw time axis at the top
        self.draw_time_axis(&painter, theme, rect);
    }

    /// Handle input events.
    fn handle_input(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let response = ui.interact(
            rect,
            ui.id().with("timeline_canvas"),
            egui::Sense::click_and_drag(),
        );

        // Pan
        if response.dragged() {
            self.camera.pan_by(response.drag_delta());
        }

        // Zoom with scroll wheel
        let scroll = ui.ctx().input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            if let Some(pointer) = ui.ctx().input(|i| i.pointer.hover_pos()) {
                if rect.contains(pointer) {
                    // Zoom centered on mouse position
                    let zoom_factor = 1.0 + scroll * 0.002;
                    let new_zoom = (self.camera.zoom * zoom_factor).clamp(0.1, 10.0);

                    // Adjust offset to zoom toward mouse position
                    let mouse_rel = pointer.x - rect.left();
                    let scale_change = new_zoom / self.camera.zoom;
                    self.camera.offset.x =
                        mouse_rel - (mouse_rel - self.camera.offset.x) * scale_change;

                    self.camera.zoom = new_zoom;
                }
            }
        }

        // Reset on double-click
        if response.double_clicked() {
            self.camera.reset();
            self.pixels_per_sec = 0.01;
        }
    }

    /// Draw background grid.
    fn draw_grid(&self, painter: &egui::Painter, theme: &Theme, rect: Rect) {
        let grid_color = theme.panel_bg.linear_multiply(1.5);

        // Calculate time range visible
        let effective_scale = self.pixels_per_sec * self.camera.zoom;

        // Grid step in seconds (adaptive based on pixels per grid line)
        // Aim for ~100 pixels between grid lines
        let target_pixels = 100.0;
        let raw_step = target_pixels / effective_scale;

        // Round to nice values: 1min, 5min, 10min, 30min, 1h, 2h, 6h, 12h, 1day
        let grid_step_secs = if raw_step < 60.0 {
            60.0 // 1 minute
        } else if raw_step < 300.0 {
            300.0 // 5 minutes
        } else if raw_step < 600.0 {
            600.0 // 10 minutes
        } else if raw_step < 1800.0 {
            1800.0 // 30 minutes
        } else if raw_step < 3600.0 {
            3600.0 // 1 hour
        } else if raw_step < 7200.0 {
            7200.0 // 2 hours
        } else if raw_step < 21600.0 {
            21600.0 // 6 hours
        } else if raw_step < 43200.0 {
            43200.0 // 12 hours
        } else {
            86400.0 // 1 day
        };

        let start_time_secs = (-self.camera.offset.x / effective_scale).max(0.0);
        let end_time_secs = start_time_secs + rect.width() / effective_scale;

        // Vertical lines (time markers)
        let first_line = (start_time_secs / grid_step_secs).floor() as i64 * grid_step_secs as i64;
        let mut t = first_line as f32;

        while t <= end_time_secs {
            let x = rect.left() + self.camera.offset.x + t * effective_scale;
            if x >= rect.left() && x <= rect.right() {
                painter.line_segment(
                    [Pos2::new(x, rect.top() + 30.0), Pos2::new(x, rect.bottom())],
                    Stroke::new(1.0, grid_color),
                );
            }
            t += grid_step_secs;
        }

        // Horizontal lines (trace separators)
        for i in 0..8 {
            let y = rect.top() + 40.0 + i as f32 * 90.0;
            if y < rect.bottom() {
                painter.line_segment(
                    [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
                    Stroke::new(1.0, grid_color),
                );
            }
        }
    }

    /// Draw a single trace with actual time-based positioning.
    fn draw_trace(
        &mut self,
        painter: &egui::Painter,
        theme: &Theme,
        trace: &GpuPartialOrderTrace,
        rect: Rect,
        y_offset: f32,
        height: f32,
    ) {
        // Collect activities
        let activities: Vec<u32> = trace.activity_ids[..trace.activity_count as usize].to_vec();
        if activities.is_empty() {
            return;
        }
        self.ensure_colors(&activities);

        // Calculate total duration in seconds from timestamps
        let total_duration_ms = trace
            .end_time
            .physical_ms
            .saturating_sub(trace.start_time.physical_ms);
        let total_duration_secs = (total_duration_ms as f32 / 1000.0).max(1.0);

        // Layout parameters
        let label_width = 90.0;
        let row_height = (height / activities.len() as f32).min(14.0);
        let effective_scale = self.pixels_per_sec * self.camera.zoom;
        let timeline_x_start = rect.left() + label_width;

        // Draw case ID label
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top() + y_offset),
            egui::Align2::LEFT_CENTER,
            format!("Case {}", trace.case_id % 10000),
            egui::FontId::proportional(10.0),
            theme.text,
        );

        // Draw activity count
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top() + y_offset + 12.0),
            egui::Align2::LEFT_CENTER,
            format!("{} acts", activities.len()),
            egui::FontId::proportional(9.0),
            theme.text_muted,
        );

        // Draw concurrency indicator
        if trace.max_width > 1 {
            painter.text(
                Pos2::new(rect.left() + 5.0, rect.top() + y_offset + 24.0),
                egui::Align2::LEFT_CENTER,
                format!("⇉{}", trace.max_width),
                egui::FontId::proportional(9.0),
                Color32::YELLOW,
            );
        }

        // Draw total duration
        painter.text(
            Pos2::new(rect.left() + label_width - 5.0, rect.top() + y_offset),
            egui::Align2::RIGHT_CENTER,
            Self::format_time(total_duration_secs),
            egui::FontId::proportional(9.0),
            theme.text_muted,
        );

        let bars_y_start = rect.top() + y_offset + 5.0;

        // Draw timeline background
        let timeline_rect = Rect::from_min_size(
            Pos2::new(timeline_x_start, bars_y_start),
            Vec2::new(
                rect.width() - label_width - 10.0,
                activities.len() as f32 * row_height,
            ),
        );
        painter.rect_filled(
            timeline_rect,
            Rounding::same(2.0),
            theme.panel_bg.linear_multiply(0.6),
        );

        // Draw activity bars positioned by actual time (with camera transform)
        for (i, &activity_id) in activities.iter().enumerate() {
            let color = self
                .activity_colors
                .get(&activity_id)
                .copied()
                .unwrap_or(theme.accent);
            let row_y = bars_y_start + i as f32 * row_height;

            // Get timing data (in seconds)
            let start_secs = trace.activity_start_secs[i] as f32;
            let duration_secs = trace.activity_duration_secs[i] as f32;

            // Calculate bar position with camera offset
            let bar_x = timeline_x_start + self.camera.offset.x + start_secs * effective_scale;
            let bar_w = (duration_secs * effective_scale).max(8.0); // Minimum 8px for visibility

            // Skip if bar is completely off-screen
            if bar_x + bar_w < rect.left() || bar_x > rect.right() {
                continue;
            }

            // Clip bar to visible area
            let clipped_x = bar_x.max(timeline_x_start);
            let clipped_w = (bar_x + bar_w).min(rect.right() - 5.0) - clipped_x;

            if clipped_w <= 0.0 {
                continue;
            }

            let bar_rect = Rect::from_min_size(
                Pos2::new(clipped_x, row_y + 1.0),
                Vec2::new(clipped_w, row_height - 2.0),
            );

            // Check if concurrent
            let is_concurrent = self.has_concurrent_activities(trace, i);
            let fill_color = if is_concurrent {
                color
            } else {
                color.linear_multiply(0.7)
            };

            painter.rect_filled(bar_rect, Rounding::same(2.0), fill_color);

            // Yellow border for concurrent activities
            if is_concurrent {
                painter.rect_stroke(
                    bar_rect,
                    Rounding::same(2.0),
                    Stroke::new(1.5, Color32::YELLOW),
                );
            }

            // Activity label inside bar (if wide enough)
            let name = self
                .activity_names
                .get(&activity_id)
                .map(|n| if n.len() > 8 { &n[..8] } else { n })
                .unwrap_or("?");

            if clipped_w > 40.0 && bar_rect.left() >= timeline_x_start {
                painter.text(
                    Pos2::new(bar_rect.left() + 3.0, bar_rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    name,
                    egui::FontId::proportional(8.0),
                    Color32::WHITE,
                );
            }

            // Duration label at right of bar (if space)
            if clipped_w > 60.0 {
                painter.text(
                    Pos2::new(bar_rect.right() - 2.0, bar_rect.center().y),
                    egui::Align2::RIGHT_CENTER,
                    Self::format_time(duration_secs),
                    egui::FontId::proportional(7.0),
                    Color32::WHITE.linear_multiply(0.8),
                );
            }
        }

        // Draw arrows for direct precedence relationships
        for i in 0..activities.len() {
            for j in 0..activities.len() {
                if i != j && trace.precedes(i, j) {
                    // Check if direct precedence (no intermediate activity)
                    let is_direct = !(0..activities.len())
                        .any(|k| k != i && k != j && trace.precedes(i, k) && trace.precedes(k, j));

                    if is_direct {
                        let start_i = trace.activity_start_secs[i] as f32;
                        let dur_i = trace.activity_duration_secs[i] as f32;
                        let start_j = trace.activity_start_secs[j] as f32;

                        let x1 = timeline_x_start
                            + self.camera.offset.x
                            + (start_i + dur_i) * effective_scale;
                        let x2 =
                            timeline_x_start + self.camera.offset.x + start_j * effective_scale;
                        let y1 = bars_y_start + i as f32 * row_height + row_height / 2.0;
                        let y2 = bars_y_start + j as f32 * row_height + row_height / 2.0;

                        // Only draw if both endpoints are visible
                        if x1 >= timeline_x_start && x2 <= rect.right() {
                            painter.line_segment(
                                [Pos2::new(x1, y1), Pos2::new(x2, y2)],
                                Stroke::new(1.0, theme.accent.linear_multiply(0.3)),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Compute parallel execution layers for activities.
    /// Activities that can run in parallel are assigned the same layer.
    #[allow(dead_code)]
    fn compute_parallel_layers(&self, trace: &GpuPartialOrderTrace) -> Vec<usize> {
        let n = trace.activity_count as usize;
        let mut layers = vec![0usize; n];

        // Topological sort with layer assignment
        // Layer[i] = max(Layer[j] + 1) for all j that precede i
        for _ in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if i != j && trace.precedes(j, i) {
                        layers[i] = layers[i].max(layers[j] + 1);
                    }
                }
            }
        }

        layers
    }

    /// Check if an activity has any concurrent activities.
    fn has_concurrent_activities(&self, trace: &GpuPartialOrderTrace, idx: usize) -> bool {
        for j in 0..trace.activity_count as usize {
            if idx != j && trace.is_concurrent(idx, j) {
                return true;
            }
        }
        false
    }

    /// Draw time axis.
    fn draw_time_axis(&self, painter: &egui::Painter, theme: &Theme, rect: Rect) {
        let axis_y = rect.top() + 32.0;
        let label_width = 90.0;
        let timeline_x_start = rect.left() + label_width;

        // Calculate time range visible
        let effective_scale = self.pixels_per_sec * self.camera.zoom;

        // Time step for labels (adaptive based on pixels per label)
        // Aim for ~100 pixels between labels
        let target_pixels = 100.0;
        let raw_step = target_pixels / effective_scale;

        // Round to nice values
        let time_step_secs = if raw_step < 60.0 {
            60.0 // 1 minute
        } else if raw_step < 300.0 {
            300.0 // 5 minutes
        } else if raw_step < 600.0 {
            600.0 // 10 minutes
        } else if raw_step < 1800.0 {
            1800.0 // 30 minutes
        } else if raw_step < 3600.0 {
            3600.0 // 1 hour
        } else if raw_step < 7200.0 {
            7200.0 // 2 hours
        } else if raw_step < 21600.0 {
            21600.0 // 6 hours
        } else if raw_step < 43200.0 {
            43200.0 // 12 hours
        } else {
            86400.0 // 1 day
        };

        let start_time_secs = (-self.camera.offset.x / effective_scale).max(0.0);
        let end_time_secs = start_time_secs + (rect.width() - label_width) / effective_scale;

        // Axis line
        painter.line_segment(
            [
                Pos2::new(timeline_x_start, axis_y),
                Pos2::new(rect.right(), axis_y),
            ],
            Stroke::new(1.0, theme.text_muted),
        );

        // Time labels
        let first_label = (start_time_secs / time_step_secs).floor() as i64 * time_step_secs as i64;
        let mut t = first_label as f32;

        while t <= end_time_secs {
            let x = timeline_x_start + self.camera.offset.x + t * effective_scale;
            if x >= timeline_x_start && x <= rect.right() - 20.0 {
                // Tick mark
                painter.line_segment(
                    [Pos2::new(x, axis_y - 3.0), Pos2::new(x, axis_y + 3.0)],
                    Stroke::new(1.0, theme.text_muted),
                );

                // Label
                painter.text(
                    Pos2::new(x, axis_y - 8.0),
                    egui::Align2::CENTER_BOTTOM,
                    Self::format_time(t),
                    egui::FontId::proportional(9.0),
                    theme.text_muted,
                );
            }
            t += time_step_secs;
        }
    }

    /// Reset canvas.
    pub fn reset(&mut self) {
        self.camera.reset();
        self.activity_colors.clear();
        self.selected_trace = None;
        self.pixels_per_sec = 0.01;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_creation() {
        let canvas = TimelineCanvas::new();
        assert!(canvas.activity_colors.is_empty());
    }

    #[test]
    fn test_format_time() {
        assert_eq!(TimelineCanvas::format_time(30.0), "30s");
        assert_eq!(TimelineCanvas::format_time(90.0), "1.5m");
        assert_eq!(TimelineCanvas::format_time(3600.0), "1.0h");
        assert_eq!(TimelineCanvas::format_time(7200.0), "2.0h");
    }
}
