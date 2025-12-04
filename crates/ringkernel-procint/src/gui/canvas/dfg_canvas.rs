//! Force-directed DFG canvas.
//!
//! Renders Directly-Follows Graph with interactive layout.

use super::{Camera, NodePosition, TokenAnimation};
use crate::gui::Theme;
use crate::models::{DFGGraph, GpuDFGEdge, GpuDFGNode, GpuPatternMatch, PatternType};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};
use std::collections::HashMap;

/// DFG canvas for visualization.
pub struct DfgCanvas {
    /// Camera for pan/zoom.
    pub camera: Camera,
    /// Node positions.
    positions: HashMap<u32, NodePosition>,
    /// Activity names.
    activity_names: HashMap<u32, String>,
    /// Token animation.
    pub tokens: TokenAnimation,
    /// Selected node.
    selected_node: Option<u32>,
    /// Hovered node.
    hovered_node: Option<u32>,
    /// Layout iteration count.
    layout_iterations: u32,
    /// Patterns to highlight.
    highlight_patterns: Vec<GpuPatternMatch>,
}

impl Default for DfgCanvas {
    fn default() -> Self {
        Self::new()
    }
}

impl DfgCanvas {
    /// Create a new DFG canvas.
    pub fn new() -> Self {
        Self {
            camera: Camera::default(),
            positions: HashMap::new(),
            activity_names: HashMap::new(),
            tokens: TokenAnimation::new(),
            selected_node: None,
            hovered_node: None,
            layout_iterations: 0,
            highlight_patterns: Vec::new(),
        }
    }

    /// Set activity names for display.
    pub fn set_activity_names(&mut self, names: HashMap<u32, String>) {
        self.activity_names = names;
    }

    /// Set patterns to highlight.
    pub fn set_highlight_patterns(&mut self, patterns: Vec<GpuPatternMatch>) {
        self.highlight_patterns = patterns;
    }

    /// Update layout from DFG.
    pub fn update_layout(&mut self, dfg: &DFGGraph) {
        // Initialize positions for new nodes
        for node in dfg.nodes() {
            if !self.positions.contains_key(&node.activity_id) {
                let angle = self.positions.len() as f32 * 0.618 * std::f32::consts::TAU;
                let radius = 100.0 + self.positions.len() as f32 * 30.0;
                self.positions.insert(
                    node.activity_id,
                    NodePosition::new(angle.cos() * radius, angle.sin() * radius),
                );
            }
        }

        // Run force-directed layout iterations
        self.apply_forces(dfg);
    }

    /// Apply force-directed layout.
    fn apply_forces(&mut self, dfg: &DFGGraph) {
        let nodes = dfg.nodes();
        let edges = dfg.edges();

        if nodes.is_empty() {
            return;
        }

        // Force parameters
        let repulsion = 5000.0;
        let attraction = 0.01;
        let damping = 0.9;
        let dt = 0.3;

        // Collect node IDs
        let node_ids: Vec<u32> = nodes.iter().map(|n| n.activity_id).collect();

        // Apply repulsion between all pairs
        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                let id1 = node_ids[i];
                let id2 = node_ids[j];

                if let (Some(p1), Some(p2)) = (self.positions.get(&id1), self.positions.get(&id2)) {
                    let dx = p2.x - p1.x;
                    let dy = p2.y - p1.y;
                    let dist_sq = (dx * dx + dy * dy).max(100.0);
                    let force = repulsion / dist_sq;
                    let dist = dist_sq.sqrt();

                    let fx = force * dx / dist;
                    let fy = force * dy / dist;

                    if let Some(p) = self.positions.get_mut(&id1) {
                        p.vx -= fx;
                        p.vy -= fy;
                    }
                    if let Some(p) = self.positions.get_mut(&id2) {
                        p.vx += fx;
                        p.vy += fy;
                    }
                }
            }
        }

        // Apply attraction along edges
        for edge in edges {
            if let (Some(p1), Some(p2)) = (
                self.positions.get(&edge.source_activity),
                self.positions.get(&edge.target_activity),
            ) {
                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);

                // Target distance based on edge frequency
                let target_dist = 150.0 - (edge.frequency as f32).log2().min(50.0);
                let force = attraction * (dist - target_dist);

                let fx = force * dx / dist;
                let fy = force * dy / dist;

                let src = edge.source_activity;
                let tgt = edge.target_activity;

                if let Some(p) = self.positions.get_mut(&src) {
                    p.vx += fx;
                    p.vy += fy;
                }
                if let Some(p) = self.positions.get_mut(&tgt) {
                    p.vx -= fx;
                    p.vy -= fy;
                }
            }
        }

        // Update positions
        for pos in self.positions.values_mut() {
            pos.vx *= damping;
            pos.vy *= damping;
            pos.x += pos.vx * dt;
            pos.y += pos.vy * dt;
        }

        self.layout_iterations += 1;
    }

    /// Render the canvas.
    pub fn render(&mut self, ui: &mut egui::Ui, theme: &Theme, dfg: &DFGGraph, rect: Rect) {
        let painter = ui.painter_at(rect);
        let center = rect.center();

        // Handle input
        self.handle_input(ui, rect);

        // Update camera
        self.camera.update(ui.ctx().input(|i| i.stable_dt));

        // Update tokens
        self.tokens.update(ui.ctx().input(|i| i.stable_dt));

        // Draw edges
        for edge in dfg.edges() {
            self.draw_edge(&painter, theme, edge, center, dfg);
        }

        // Draw tokens
        for token in self.tokens.active_tokens() {
            if let (Some(src_pos), Some(tgt_pos)) = (
                self.positions.get(&token.source_activity),
                self.positions.get(&token.target_activity),
            ) {
                let p1 = self.camera.world_to_screen(src_pos.pos(), center);
                let p2 = self.camera.world_to_screen(tgt_pos.pos(), center);
                let pos = p1 + (p2 - p1) * token.progress;

                // Glow effect
                painter.circle_filled(pos, 8.0, theme.token.linear_multiply(0.3));
                painter.circle_filled(pos, 5.0, theme.token);
            }
        }

        // Draw nodes
        for node in dfg.nodes() {
            self.draw_node(&painter, theme, node, center);
        }

        // Draw highlight patterns
        self.draw_pattern_highlights(&painter, theme, center);
    }

    /// Handle input events.
    fn handle_input(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let response = ui.interact(
            rect,
            ui.id().with("dfg_canvas"),
            egui::Sense::click_and_drag(),
        );

        // Pan
        if response.dragged() {
            self.camera.pan_by(response.drag_delta());
        }

        // Zoom
        let scroll = ui.ctx().input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            if let Some(pointer) = ui.ctx().input(|i| i.pointer.hover_pos()) {
                if rect.contains(pointer) {
                    self.camera.zoom_by(scroll * 0.01, pointer, rect.center());
                }
            }
        }

        // Reset on double-click
        if response.double_clicked() {
            self.camera.reset();
        }

        // Check node hover
        self.hovered_node = None;
        if let Some(pointer) = ui.ctx().input(|i| i.pointer.hover_pos()) {
            if rect.contains(pointer) {
                let world_pos = self.camera.screen_to_world(pointer, rect.center());
                for (id, pos) in &self.positions {
                    let dist = (Pos2::new(pos.x, pos.y) - world_pos).length();
                    if dist < 30.0 / self.camera.zoom {
                        self.hovered_node = Some(*id);
                        break;
                    }
                }
            }
        }

        // Select on click
        if response.clicked() {
            self.selected_node = self.hovered_node;
        }
    }

    /// Draw a single node.
    fn draw_node(&self, painter: &egui::Painter, theme: &Theme, node: &GpuDFGNode, center: Pos2) {
        if let Some(pos) = self.positions.get(&node.activity_id) {
            let screen_pos = self.camera.world_to_screen(pos.pos(), center);

            // Node size based on event count
            let base_size = 20.0;
            let size = base_size + (node.event_count as f32).log2().min(20.0) * 2.0;
            let size = size * self.camera.zoom;

            // Node color
            let color = if node.is_start != 0 {
                theme.node_start
            } else if node.is_end != 0 {
                theme.node_end
            } else {
                theme.node_default
            };

            // Highlight if selected or hovered
            let is_highlighted = self.selected_node == Some(node.activity_id)
                || self.hovered_node == Some(node.activity_id);

            if is_highlighted {
                painter.circle_filled(screen_pos, size + 4.0, color.linear_multiply(0.3));
            }

            // Node circle
            painter.circle_filled(screen_pos, size, color);
            painter.circle_stroke(
                screen_pos,
                size,
                Stroke::new(2.0, Color32::WHITE.linear_multiply(0.3)),
            );

            // Node label
            let name = self
                .activity_names
                .get(&node.activity_id)
                .cloned()
                .unwrap_or_else(|| format!("A{}", node.activity_id));

            let label_pos = screen_pos + Vec2::new(0.0, size + 12.0);
            painter.text(
                label_pos,
                egui::Align2::CENTER_TOP,
                &name,
                egui::FontId::proportional(11.0 * self.camera.zoom.sqrt()),
                theme.text,
            );

            // Event count badge
            if node.event_count > 0 {
                let badge_pos = screen_pos + Vec2::new(size * 0.7, -size * 0.7);
                painter.circle_filled(badge_pos, 10.0 * self.camera.zoom, theme.accent);
                painter.text(
                    badge_pos,
                    egui::Align2::CENTER_CENTER,
                    format!("{}", node.event_count),
                    egui::FontId::proportional(8.0 * self.camera.zoom),
                    Color32::WHITE,
                );
            }
        }
    }

    /// Draw a single edge.
    fn draw_edge(
        &self,
        painter: &egui::Painter,
        theme: &Theme,
        edge: &GpuDFGEdge,
        center: Pos2,
        dfg: &DFGGraph,
    ) {
        if let (Some(src_pos), Some(tgt_pos)) = (
            self.positions.get(&edge.source_activity),
            self.positions.get(&edge.target_activity),
        ) {
            let p1 = self.camera.world_to_screen(src_pos.pos(), center);
            let p2 = self.camera.world_to_screen(tgt_pos.pos(), center);

            // Edge width based on frequency
            let width = 1.0 + (edge.frequency as f32).log2().min(4.0);

            // Edge color based on duration
            let avg_duration = dfg.nodes().iter().map(|n| n.avg_duration_ms).sum::<f32>()
                / dfg.nodes().len().max(1) as f32;
            let color = theme.edge_duration_color(edge.avg_duration_ms, avg_duration);

            // Draw edge
            painter.line_segment([p1, p2], Stroke::new(width, color.linear_multiply(0.7)));

            // Draw arrowhead
            let dir = (p2 - p1).normalized();
            let arrow_size = 8.0 * self.camera.zoom;
            let arrow_pos = p2 - dir * 25.0 * self.camera.zoom;

            let perp = Vec2::new(-dir.y, dir.x);
            let arrow_p1 = arrow_pos - dir * arrow_size + perp * arrow_size * 0.5;
            let arrow_p2 = arrow_pos - dir * arrow_size - perp * arrow_size * 0.5;

            painter.add(egui::Shape::convex_polygon(
                vec![arrow_pos, arrow_p1, arrow_p2],
                color,
                Stroke::NONE,
            ));

            // Draw frequency label on edge
            if edge.frequency > 1 {
                let mid = p1 + (p2 - p1) * 0.5;
                let label_offset = perp * 12.0;
                painter.text(
                    mid + label_offset,
                    egui::Align2::CENTER_CENTER,
                    format!("{}", edge.frequency),
                    egui::FontId::proportional(9.0),
                    theme.text_muted,
                );
            }
        }
    }

    /// Draw pattern highlights.
    fn draw_pattern_highlights(&self, painter: &egui::Painter, theme: &Theme, center: Pos2) {
        for pattern in &self.highlight_patterns {
            let pattern_type = pattern.get_pattern_type();
            let color = match pattern_type {
                PatternType::Bottleneck => theme.bottleneck,
                PatternType::Loop | PatternType::Rework => theme.loop_highlight,
                PatternType::LongRunning => theme.warning,
                _ => theme.accent,
            };

            // Highlight each activity in the pattern
            for &activity_id in pattern.activities() {
                if let Some(pos) = self.positions.get(&activity_id) {
                    let screen_pos = self.camera.world_to_screen(pos.pos(), center);
                    let size = 35.0 * self.camera.zoom;

                    // Pulsing glow effect
                    let time = painter.ctx().input(|i| i.time);
                    let alpha = ((time * 2.0).sin() * 0.3 + 0.5) as f32;

                    painter.circle_stroke(
                        screen_pos,
                        size,
                        Stroke::new(3.0, color.linear_multiply(alpha)),
                    );
                    painter.circle_stroke(
                        screen_pos,
                        size + 5.0,
                        Stroke::new(2.0, color.linear_multiply(alpha * 0.5)),
                    );
                }
            }
        }
    }

    /// Reset layout.
    pub fn reset(&mut self) {
        self.positions.clear();
        self.layout_iterations = 0;
        self.tokens.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canvas_creation() {
        let canvas = DfgCanvas::new();
        assert!(canvas.positions.is_empty());
    }
}
