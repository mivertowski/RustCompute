//! Network graph canvas with interactive visualization.
//!
//! Renders the accounting network as an interactive force-directed graph
//! with flow particle animations.

use eframe::egui::{self, Color32, Pos2, Rect, Response, Sense, Stroke, Vec2};
use nalgebra::Vector2;
use uuid::Uuid;

use super::animation::ParticleSystem;
use super::layout::{ForceDirectedLayout, LayoutConfig};
use super::theme::AccNetTheme;
use crate::models::AccountingNetwork;

/// Interactive network canvas.
pub struct NetworkCanvas {
    /// Force-directed layout engine.
    pub layout: ForceDirectedLayout,
    /// Particle animation system.
    pub particles: ParticleSystem,
    /// Visual theme.
    pub theme: AccNetTheme,
    /// Canvas zoom level.
    pub zoom: f32,
    /// Canvas pan offset.
    pub pan: Vec2,
    /// Currently selected node.
    pub selected_node: Option<u16>,
    /// Currently hovered node.
    pub hovered_node: Option<u16>,
    /// Node being dragged.
    dragging_node: Option<u16>,
    /// Show labels.
    pub show_labels: bool,
    /// Show risk indicators.
    pub show_risk: bool,
    /// Animate layout.
    pub animate_layout: bool,
    /// Last frame time for delta calculation.
    last_frame: std::time::Instant,
}

impl NetworkCanvas {
    /// Create a new network canvas.
    pub fn new() -> Self {
        Self {
            layout: ForceDirectedLayout::new(LayoutConfig::default()),
            particles: ParticleSystem::new(2000),
            theme: AccNetTheme::dark(),
            zoom: 1.0,
            pan: Vec2::ZERO,
            selected_node: None,
            hovered_node: None,
            dragging_node: None,
            show_labels: true,
            show_risk: true,
            animate_layout: true,
            last_frame: std::time::Instant::now(),
        }
    }

    /// Initialize from a network.
    pub fn initialize(&mut self, network: &AccountingNetwork) {
        self.layout.initialize(network);
        self.refresh_particles(network);
    }

    /// Refresh particles without resetting layout.
    /// Call this periodically to update flow animations while preserving node positions.
    pub fn refresh_particles(&mut self, network: &AccountingNetwork) {
        self.particles.clear();

        // Queue recent flows for particle animation (limit to last 500 for performance)
        let flow_count = network.flows.len();
        let start_idx = if flow_count > 500 {
            flow_count - 500
        } else {
            0
        };

        for flow in &network.flows[start_idx..] {
            let suspicious = flow.is_anomalous();
            let color = if suspicious {
                self.theme.flow_suspicious
            } else {
                self.theme.flow_normal
            };
            self.particles.queue_flow(
                flow.source_account_index,
                flow.target_account_index,
                Uuid::new_v4(),
                color,
                suspicious,
            );
        }
    }

    /// Update layout and animations.
    pub fn update(&mut self) {
        let now = std::time::Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Update layout
        if self.animate_layout && !self.layout.converged {
            self.layout.iterate(5);
        }

        // Update particles
        self.particles.update(dt);
    }

    /// Render the canvas.
    pub fn show(&mut self, ui: &mut egui::Ui, network: &AccountingNetwork) -> Response {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());

        let rect = response.rect;

        // Only resize layout when canvas size actually changes
        let current_width = self.layout.config.width;
        let current_height = self.layout.config.height;
        if (rect.width() - current_width).abs() > 1.0
            || (rect.height() - current_height).abs() > 1.0
        {
            self.layout.resize(rect.width(), rect.height());
        }

        // Handle input
        self.handle_input(ui, &response, rect);

        // Draw background
        painter.rect_filled(rect, 0.0, self.theme.canvas_bg);

        // Draw grid
        self.draw_grid(&painter, rect);

        // Transform helper
        let transform = |pos: Vector2<f32>| -> Pos2 {
            let x = rect.left() + (pos.x + self.pan.x) * self.zoom;
            let y = rect.top() + (pos.y + self.pan.y) * self.zoom;
            Pos2::new(x, y)
        };

        // Calculate max edge weight for normalization
        let max_weight = self
            .layout
            .edges()
            .iter()
            .map(|(_, _, w)| *w)
            .fold(1.0f32, f32::max);

        // Calculate center for edge bundling
        let center = {
            let (sum_x, sum_y, count) = self
                .layout
                .nodes()
                .fold((0.0f32, 0.0f32, 0usize), |(sx, sy, c), node| {
                    (sx + node.position.x, sy + node.position.y, c + 1)
                });
            if count > 0 {
                Vector2::new(sum_x / count as f32, sum_y / count as f32)
            } else {
                Vector2::new(
                    self.layout.config.width / 2.0,
                    self.layout.config.height / 2.0,
                )
            }
        };

        // Draw edges with futuristic styling and edge bundling
        for (source_idx, target_idx, weight) in self.layout.edges() {
            if let (Some(src_pos), Some(tgt_pos)) = (
                self.layout.get_position(*source_idx),
                self.layout.get_position(*target_idx),
            ) {
                // Check if this edge involves suspicious activity
                let suspicious = network.flows.iter().any(|f| {
                    f.source_account_index == *source_idx
                        && f.target_account_index == *target_idx
                        && f.is_anomalous()
                });

                // Get futuristic color with transparency
                let color = self
                    .theme
                    .edge_color_futuristic(*weight, suspicious, false, max_weight);

                // Thin edges: 0.5 to 2.0 based on weight
                let thickness = (0.5 + (*weight / max_weight).sqrt() * 1.5).min(2.0);

                // Edge bundling: curve edges toward center
                let bundle_strength = 0.15; // How much to curve toward center
                let ctrl = Vector2::new(
                    src_pos.x
                        + (tgt_pos.x - src_pos.x) * 0.5
                        + (center.x - (src_pos.x + tgt_pos.x) * 0.5) * bundle_strength,
                    src_pos.y
                        + (tgt_pos.y - src_pos.y) * 0.5
                        + (center.y - (src_pos.y + tgt_pos.y) * 0.5) * bundle_strength,
                );

                // Draw curved edge (quadratic bezier)
                self.draw_curved_edge(
                    &painter,
                    transform(src_pos),
                    transform(ctrl),
                    transform(tgt_pos),
                    Stroke::new(thickness, color),
                );
            }
        }

        // Draw particles
        for particle in self.particles.particles() {
            if let (Some(src_pos), Some(tgt_pos)) = (
                self.layout.get_position(particle.source),
                self.layout.get_position(particle.target),
            ) {
                let pos = particle.position(src_pos, tgt_pos);
                let screen_pos = transform(pos);

                // Glow effect
                let glow_color = Color32::from_rgba_unmultiplied(
                    particle.color.r(),
                    particle.color.g(),
                    particle.color.b(),
                    80,
                );
                painter.circle_filled(screen_pos, particle.size * 2.0 * self.zoom, glow_color);
                painter.circle_filled(screen_pos, particle.size * self.zoom, particle.color);
            }
        }

        // Draw nodes
        self.hovered_node = None;
        for node in self.layout.nodes() {
            let pos = transform(node.position);
            let radius = self.theme.node_radius * self.zoom;

            // Check hover
            let mouse_pos = response.hover_pos().unwrap_or(Pos2::ZERO);
            let distance = (pos - mouse_pos).length();
            if distance < radius {
                self.hovered_node = Some(node.index);
            }

            // Node color
            let base_color = self.theme.account_color(node.account_type);
            let color = if Some(node.index) == self.selected_node {
                self.theme.accent
            } else if Some(node.index) == self.hovered_node {
                Color32::from_rgb(
                    (base_color.r() as u16 + 40).min(255) as u8,
                    (base_color.g() as u16 + 40).min(255) as u8,
                    (base_color.b() as u16 + 40).min(255) as u8,
                )
            } else {
                base_color
            };

            // Draw node
            painter.circle_filled(pos, radius, color);
            painter.circle_stroke(pos, radius, Stroke::new(2.0, Color32::WHITE));

            // Risk indicator
            if self.show_risk {
                if let Some(account) = network.accounts.iter().find(|a| a.index == node.index) {
                    if account.risk_score > 0.5 {
                        let risk_color = self.severity_color(account.risk_score);
                        painter.circle_filled(
                            Pos2::new(pos.x + radius * 0.7, pos.y - radius * 0.7),
                            6.0 * self.zoom,
                            risk_color,
                        );
                    }
                }
            }

            // Label
            if self.show_labels && self.zoom > 0.5 {
                if let Some(metadata) = network.account_metadata.get(&node.index) {
                    let label_pos = Pos2::new(pos.x, pos.y + radius + 10.0);
                    painter.text(
                        label_pos,
                        egui::Align2::CENTER_TOP,
                        &metadata.name,
                        egui::FontId::proportional(12.0 * self.zoom),
                        self.theme.text_primary,
                    );
                }
            }
        }

        // Draw legend
        self.draw_legend(&painter, rect, network);

        response
    }

    /// Handle user input.
    fn handle_input(&mut self, ui: &egui::Ui, response: &Response, rect: Rect) {
        // Pan with middle mouse or right drag
        if response.dragged_by(egui::PointerButton::Middle)
            || response.dragged_by(egui::PointerButton::Secondary)
        {
            self.pan += response.drag_delta() / self.zoom;
        }

        // Zoom with scroll
        if response.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                let zoom_delta = 1.0 + scroll * 0.001;
                self.zoom = (self.zoom * zoom_delta).clamp(0.1, 5.0);
            }
        }

        // Node selection with left click
        if response.clicked() {
            self.selected_node = self.hovered_node;
        }

        // Node dragging
        if response.drag_started_by(egui::PointerButton::Primary) && self.hovered_node.is_some() {
            self.dragging_node = self.hovered_node;
            if let Some(idx) = self.dragging_node {
                self.layout.pin_node(idx);
            }
        }

        if let Some(idx) = self.dragging_node {
            if response.dragged_by(egui::PointerButton::Primary) {
                let mouse_pos = response.hover_pos().unwrap_or(Pos2::ZERO);
                let canvas_pos = Vector2::new(
                    (mouse_pos.x - rect.left()) / self.zoom - self.pan.x,
                    (mouse_pos.y - rect.top()) / self.zoom - self.pan.y,
                );
                self.layout.set_position(idx, canvas_pos);
            }

            if response.drag_stopped() {
                self.layout.unpin_node(idx);
                self.dragging_node = None;
            }
        }
    }

    /// Draw background grid.
    fn draw_grid(&self, painter: &egui::Painter, rect: Rect) {
        let grid_spacing = 50.0 * self.zoom;
        let grid_color = self.theme.grid_color;

        // Vertical lines
        let start_x = rect.left() + (self.pan.x * self.zoom) % grid_spacing;
        let mut x = start_x;
        while x < rect.right() {
            painter.line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                Stroke::new(1.0, grid_color),
            );
            x += grid_spacing;
        }

        // Horizontal lines
        let start_y = rect.top() + (self.pan.y * self.zoom) % grid_spacing;
        let mut y = start_y;
        while y < rect.bottom() {
            painter.line_segment(
                [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
                Stroke::new(1.0, grid_color),
            );
            y += grid_spacing;
        }
    }

    /// Draw a curved edge using quadratic Bezier curve.
    fn draw_curved_edge(
        &self,
        painter: &egui::Painter,
        src: Pos2,
        ctrl: Pos2,
        tgt: Pos2,
        stroke: Stroke,
    ) {
        // Approximate quadratic Bezier with line segments for smooth curve
        let segments = 16;
        let mut prev = src;

        for i in 1..=segments {
            let t = i as f32 / segments as f32;
            // Quadratic Bezier: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
            let u = 1.0 - t;
            let x = u * u * src.x + 2.0 * u * t * ctrl.x + t * t * tgt.x;
            let y = u * u * src.y + 2.0 * u * t * ctrl.y + t * t * tgt.y;
            let curr = Pos2::new(x, y);

            painter.line_segment([prev, curr], stroke);
            prev = curr;
        }
    }

    /// Draw an arrow from src to tgt.
    #[allow(dead_code)]
    fn draw_arrow(&self, painter: &egui::Painter, src: Pos2, tgt: Pos2, stroke: Stroke) {
        // Line
        painter.line_segment([src, tgt], stroke);

        // Arrowhead
        let dir = (tgt - src).normalized();
        let perp = Vec2::new(-dir.y, dir.x);
        let arrow_size = 8.0 * self.zoom;
        let tip = tgt - dir * (self.theme.node_radius * self.zoom);

        let arrow_points = [
            tip,
            tip - dir * arrow_size + perp * arrow_size * 0.5,
            tip - dir * arrow_size - perp * arrow_size * 0.5,
        ];

        painter.add(egui::Shape::convex_polygon(
            arrow_points.to_vec(),
            stroke.color,
            Stroke::NONE,
        ));
    }

    /// Get color for severity level.
    fn severity_color(&self, score: f32) -> Color32 {
        if score > 0.8 {
            self.theme.alert_critical
        } else if score > 0.6 {
            self.theme.alert_high
        } else if score > 0.4 {
            self.theme.alert_medium
        } else {
            self.theme.alert_low
        }
    }

    /// Draw the enhanced legend with account types, flow types, and status indicators.
    fn draw_legend(&self, painter: &egui::Painter, rect: Rect, network: &AccountingNetwork) {
        let legend_x = rect.right() - 130.0;
        let legend_y = rect.top() + 15.0;
        let section_spacing = 12.0;
        let line_height = 16.0;

        // Account types section
        let account_entries = [
            ("Asset", self.theme.asset_color),
            ("Liability", self.theme.liability_color),
            ("Equity", self.theme.equity_color),
            ("Revenue", self.theme.revenue_color),
            ("Expense", self.theme.expense_color),
        ];

        // Flow types section
        let flow_entries = [
            ("Normal Flow", self.theme.flow_normal),
            ("Suspicious", self.theme.flow_suspicious),
        ];

        // Status indicators section
        let status_entries = [
            ("Low Risk", self.theme.alert_low),
            ("Medium Risk", self.theme.alert_medium),
            ("High Risk", self.theme.alert_high),
            ("Critical", self.theme.alert_critical),
        ];

        // Calculate total height
        let total_height = 18.0 + // "Accounts" header
            account_entries.len() as f32 * line_height +
            section_spacing +
            18.0 + // "Flows" header
            flow_entries.len() as f32 * line_height +
            section_spacing +
            18.0 + // "Status" header
            status_entries.len() as f32 * line_height +
            section_spacing +
            45.0; // Stats section

        // Background
        let bg_rect = Rect::from_min_size(
            Pos2::new(legend_x - 10.0, legend_y - 5.0),
            Vec2::new(125.0, total_height),
        );
        painter.rect_filled(
            bg_rect,
            6.0,
            Color32::from_rgba_unmultiplied(15, 15, 25, 220),
        );
        painter.rect_stroke(
            bg_rect,
            6.0,
            Stroke::new(1.0, Color32::from_rgb(60, 60, 80)),
        );

        let mut y = legend_y;

        // Account types header
        painter.text(
            Pos2::new(legend_x, y),
            egui::Align2::LEFT_TOP,
            "Accounts",
            egui::FontId::proportional(11.0),
            self.theme.text_secondary,
        );
        y += 14.0;

        // Account type entries
        for (label, color) in &account_entries {
            painter.circle_filled(Pos2::new(legend_x + 5.0, y + 6.0), 5.0, *color);
            painter.text(
                Pos2::new(legend_x + 16.0, y),
                egui::Align2::LEFT_TOP,
                *label,
                egui::FontId::proportional(10.0),
                self.theme.text_primary,
            );
            y += line_height;
        }

        y += section_spacing - 4.0;

        // Flow types header
        painter.text(
            Pos2::new(legend_x, y),
            egui::Align2::LEFT_TOP,
            "Flows",
            egui::FontId::proportional(11.0),
            self.theme.text_secondary,
        );
        y += 14.0;

        // Flow type entries (with line indicator instead of circle)
        for (label, color) in &flow_entries {
            painter.line_segment(
                [
                    Pos2::new(legend_x, y + 6.0),
                    Pos2::new(legend_x + 12.0, y + 6.0),
                ],
                Stroke::new(3.0, *color),
            );
            painter.text(
                Pos2::new(legend_x + 18.0, y),
                egui::Align2::LEFT_TOP,
                *label,
                egui::FontId::proportional(10.0),
                self.theme.text_primary,
            );
            y += line_height;
        }

        y += section_spacing - 4.0;

        // Status indicators header
        painter.text(
            Pos2::new(legend_x, y),
            egui::Align2::LEFT_TOP,
            "Risk Level",
            egui::FontId::proportional(11.0),
            self.theme.text_secondary,
        );
        y += 14.0;

        // Status entries (with small square indicator)
        for (label, color) in &status_entries {
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(legend_x, y + 2.0), Vec2::new(10.0, 10.0)),
                2.0,
                *color,
            );
            painter.text(
                Pos2::new(legend_x + 16.0, y),
                egui::Align2::LEFT_TOP,
                *label,
                egui::FontId::proportional(10.0),
                self.theme.text_primary,
            );
            y += line_height;
        }

        y += section_spacing;

        // Live stats section
        painter.line_segment(
            [Pos2::new(legend_x, y), Pos2::new(legend_x + 100.0, y)],
            Stroke::new(1.0, Color32::from_rgb(60, 60, 80)),
        );
        y += 8.0;

        // Node count
        painter.text(
            Pos2::new(legend_x, y),
            egui::Align2::LEFT_TOP,
            format!("Nodes: {}", network.accounts.len()),
            egui::FontId::proportional(10.0),
            self.theme.text_secondary,
        );
        y += 14.0;

        // Edge count
        painter.text(
            Pos2::new(legend_x, y),
            egui::Align2::LEFT_TOP,
            format!("Edges: {}", network.flows.len()),
            egui::FontId::proportional(10.0),
            self.theme.text_secondary,
        );
        y += 14.0;

        // Particle count
        painter.text(
            Pos2::new(legend_x, y),
            egui::Align2::LEFT_TOP,
            format!("Particles: {}", self.particles.count()),
            egui::FontId::proportional(10.0),
            self.theme.text_secondary,
        );
    }

    /// Reset view.
    pub fn reset_view(&mut self) {
        self.zoom = 1.0;
        self.pan = Vec2::ZERO;
    }
}

impl Default for NetworkCanvas {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canvas_creation() {
        let canvas = NetworkCanvas::new();
        assert_eq!(canvas.zoom, 1.0);
        assert!(canvas.show_labels);
    }
}
