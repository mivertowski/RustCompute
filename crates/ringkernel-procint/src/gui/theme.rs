//! Theme and styling for the process intelligence GUI.

use eframe::egui::{self, Color32, Rounding, Stroke, Vec2};

/// Application theme colors.
pub struct Theme {
    /// Background color.
    pub background: Color32,
    /// Panel background.
    pub panel_bg: Color32,
    /// Primary accent color.
    pub accent: Color32,
    /// Secondary accent.
    pub accent_secondary: Color32,
    /// Text color.
    pub text: Color32,
    /// Muted text.
    pub text_muted: Color32,
    /// Success color.
    pub success: Color32,
    /// Warning color.
    pub warning: Color32,
    /// Error color.
    pub error: Color32,
    /// Node default color.
    pub node_default: Color32,
    /// Node start color.
    pub node_start: Color32,
    /// Node end color.
    pub node_end: Color32,
    /// Edge default color.
    pub edge_default: Color32,
    /// Edge fast color.
    pub edge_fast: Color32,
    /// Edge slow color.
    pub edge_slow: Color32,
    /// Token color.
    pub token: Color32,
    /// Bottleneck highlight.
    pub bottleneck: Color32,
    /// Loop highlight.
    pub loop_highlight: Color32,
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

impl Theme {
    /// Dark theme (default).
    pub fn dark() -> Self {
        Self {
            background: Color32::from_rgb(18, 18, 24),
            panel_bg: Color32::from_rgb(28, 28, 36),
            accent: Color32::from_rgb(99, 102, 241), // Indigo
            accent_secondary: Color32::from_rgb(139, 92, 246), // Purple
            text: Color32::from_rgb(229, 231, 235),
            text_muted: Color32::from_rgb(156, 163, 175),
            success: Color32::from_rgb(34, 197, 94),
            warning: Color32::from_rgb(234, 179, 8),
            error: Color32::from_rgb(239, 68, 68),
            node_default: Color32::from_rgb(59, 130, 246), // Blue
            node_start: Color32::from_rgb(34, 197, 94),    // Green
            node_end: Color32::from_rgb(239, 68, 68),      // Red
            edge_default: Color32::from_rgb(100, 116, 139),
            edge_fast: Color32::from_rgb(34, 197, 94),
            edge_slow: Color32::from_rgb(239, 68, 68),
            token: Color32::from_rgb(250, 204, 21), // Yellow
            bottleneck: Color32::from_rgb(239, 68, 68),
            loop_highlight: Color32::from_rgb(59, 130, 246),
        }
    }

    /// Light theme.
    pub fn light() -> Self {
        Self {
            background: Color32::from_rgb(249, 250, 251),
            panel_bg: Color32::from_rgb(255, 255, 255),
            accent: Color32::from_rgb(79, 70, 229),
            accent_secondary: Color32::from_rgb(124, 58, 237),
            text: Color32::from_rgb(17, 24, 39),
            text_muted: Color32::from_rgb(107, 114, 128),
            success: Color32::from_rgb(22, 163, 74),
            warning: Color32::from_rgb(202, 138, 4),
            error: Color32::from_rgb(220, 38, 38),
            node_default: Color32::from_rgb(37, 99, 235),
            node_start: Color32::from_rgb(22, 163, 74),
            node_end: Color32::from_rgb(220, 38, 38),
            edge_default: Color32::from_rgb(148, 163, 184),
            edge_fast: Color32::from_rgb(22, 163, 74),
            edge_slow: Color32::from_rgb(220, 38, 38),
            token: Color32::from_rgb(234, 179, 8),
            bottleneck: Color32::from_rgb(220, 38, 38),
            loop_highlight: Color32::from_rgb(37, 99, 235),
        }
    }

    /// Apply theme to egui context.
    pub fn apply(&self, ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();

        // Set colors
        style.visuals.dark_mode = self.background.r() < 128;
        style.visuals.override_text_color = Some(self.text);
        style.visuals.panel_fill = self.panel_bg;
        style.visuals.window_fill = self.panel_bg;
        style.visuals.extreme_bg_color = self.background;

        // Widgets
        style.visuals.widgets.noninteractive.bg_fill = self.panel_bg;
        style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(38, 38, 48);
        style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(48, 48, 58);
        style.visuals.widgets.active.bg_fill = self.accent;

        // Selection
        style.visuals.selection.bg_fill = self.accent.linear_multiply(0.3);
        style.visuals.selection.stroke = Stroke::new(1.0, self.accent);

        // Rounding
        style.visuals.window_rounding = Rounding::same(8.0);
        style.visuals.menu_rounding = Rounding::same(6.0);

        // Spacing
        style.spacing.item_spacing = Vec2::new(8.0, 6.0);
        style.spacing.button_padding = Vec2::new(12.0, 6.0);

        ctx.set_style(style);
    }

    /// Get color for fitness value.
    pub fn fitness_color(&self, fitness: f32) -> Color32 {
        if fitness >= 0.95 {
            self.success
        } else if fitness >= 0.80 {
            Color32::from_rgb(34, 211, 238) // Cyan
        } else if fitness >= 0.50 {
            self.warning
        } else {
            self.error
        }
    }

    /// Get color for edge duration.
    pub fn edge_duration_color(&self, duration_ms: f32, avg_duration_ms: f32) -> Color32 {
        let ratio = duration_ms / avg_duration_ms.max(1.0);
        if ratio < 0.5 {
            self.edge_fast
        } else if ratio > 2.0 {
            self.edge_slow
        } else {
            self.edge_default
        }
    }

    /// Get pattern severity color.
    pub fn severity_color(&self, severity: crate::models::PatternSeverity) -> Color32 {
        match severity {
            crate::models::PatternSeverity::Info => self.accent,
            crate::models::PatternSeverity::Warning => self.warning,
            crate::models::PatternSeverity::Critical => self.error,
        }
    }
}

/// Styled panel with rounded corners.
pub fn styled_panel(ui: &mut egui::Ui, theme: &Theme, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::none()
        .fill(theme.panel_bg)
        .rounding(Rounding::same(8.0))
        .inner_margin(egui::Margin::same(12.0))
        .stroke(Stroke::new(1.0, Color32::from_rgb(38, 38, 48)))
        .show(ui, |ui| {
            add_contents(ui);
        });
}

/// Section header with accent underline.
pub fn section_header(ui: &mut egui::Ui, theme: &Theme, title: &str) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(title).strong().size(14.0));
    });
    ui.add_space(2.0);
    let rect = ui.available_rect_before_wrap();
    ui.painter().line_segment(
        [
            egui::pos2(rect.left(), rect.top()),
            egui::pos2(rect.left() + 60.0, rect.top()),
        ],
        Stroke::new(2.0, theme.accent),
    );
    ui.add_space(8.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theme_creation() {
        let dark = Theme::dark();
        assert!(dark.background.r() < 50);

        let light = Theme::light();
        assert!(light.background.r() > 200);
    }

    #[test]
    fn test_fitness_color() {
        let theme = Theme::dark();
        assert_eq!(theme.fitness_color(0.98), theme.success);
        assert_eq!(theme.fitness_color(0.30), theme.error);
    }
}
