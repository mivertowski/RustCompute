//! Visual theme for the accounting network GUI.

use eframe::egui::{self, Color32, Rounding, Stroke};

/// Color theme for the application.
#[derive(Debug, Clone)]
pub struct AccNetTheme {
    /// Background color for the main canvas.
    pub canvas_bg: Color32,
    /// Grid line color.
    pub grid_color: Color32,
    /// Panel background.
    pub panel_bg: Color32,
    /// Text color.
    pub text_primary: Color32,
    /// Secondary text color.
    pub text_secondary: Color32,
    /// Accent color for highlights.
    pub accent: Color32,

    // Account type colors
    /// Asset account color.
    pub asset_color: Color32,
    /// Liability account color.
    pub liability_color: Color32,
    /// Equity account color.
    pub equity_color: Color32,
    /// Revenue account color.
    pub revenue_color: Color32,
    /// Expense account color.
    pub expense_color: Color32,
    /// Contra account color.
    pub contra_color: Color32,

    // Alert colors
    /// Low severity.
    pub alert_low: Color32,
    /// Medium severity.
    pub alert_medium: Color32,
    /// High severity.
    pub alert_high: Color32,
    /// Critical severity.
    pub alert_critical: Color32,

    // Flow colors
    /// Normal flow.
    pub flow_normal: Color32,
    /// Suspicious flow.
    pub flow_suspicious: Color32,
    /// Fraudulent flow.
    pub flow_fraud: Color32,

    // UI
    /// Node radius.
    pub node_radius: f32,
    /// Edge width.
    pub edge_width: f32,
    /// Panel rounding.
    pub panel_rounding: Rounding,
}

impl Default for AccNetTheme {
    fn default() -> Self {
        Self::dark()
    }
}

impl AccNetTheme {
    /// Create a dark theme (default).
    pub fn dark() -> Self {
        Self {
            canvas_bg: Color32::from_rgb(18, 18, 24),
            grid_color: Color32::from_rgb(40, 40, 50),
            panel_bg: Color32::from_rgb(28, 28, 36),
            text_primary: Color32::from_rgb(240, 240, 245),
            text_secondary: Color32::from_rgb(140, 140, 155),
            accent: Color32::from_rgb(100, 180, 255),

            // Account types - professional palette
            asset_color: Color32::from_rgb(66, 165, 245), // Blue
            liability_color: Color32::from_rgb(239, 83, 80), // Red
            equity_color: Color32::from_rgb(102, 187, 106), // Green
            revenue_color: Color32::from_rgb(255, 202, 40), // Amber
            expense_color: Color32::from_rgb(171, 71, 188), // Purple
            contra_color: Color32::from_rgb(158, 158, 158), // Gray

            // Alerts
            alert_low: Color32::from_rgb(100, 180, 255),
            alert_medium: Color32::from_rgb(255, 202, 40),
            alert_high: Color32::from_rgb(255, 152, 0),
            alert_critical: Color32::from_rgb(244, 67, 54),

            // Flows
            flow_normal: Color32::from_rgb(100, 255, 180),
            flow_suspicious: Color32::from_rgb(255, 202, 40),
            flow_fraud: Color32::from_rgb(244, 67, 54),

            // UI
            node_radius: 20.0,
            edge_width: 2.0,
            panel_rounding: Rounding::same(8.0),
        }
    }

    /// Create a light theme.
    pub fn light() -> Self {
        Self {
            canvas_bg: Color32::from_rgb(245, 245, 250),
            grid_color: Color32::from_rgb(220, 220, 230),
            panel_bg: Color32::from_rgb(255, 255, 255),
            text_primary: Color32::from_rgb(30, 30, 40),
            text_secondary: Color32::from_rgb(100, 100, 115),
            accent: Color32::from_rgb(25, 118, 210),

            // Account types
            asset_color: Color32::from_rgb(25, 118, 210),
            liability_color: Color32::from_rgb(211, 47, 47),
            equity_color: Color32::from_rgb(56, 142, 60),
            revenue_color: Color32::from_rgb(255, 160, 0),
            expense_color: Color32::from_rgb(142, 36, 170),
            contra_color: Color32::from_rgb(117, 117, 117),

            // Alerts
            alert_low: Color32::from_rgb(25, 118, 210),
            alert_medium: Color32::from_rgb(255, 160, 0),
            alert_high: Color32::from_rgb(230, 81, 0),
            alert_critical: Color32::from_rgb(198, 40, 40),

            // Flows
            flow_normal: Color32::from_rgb(0, 150, 136),
            flow_suspicious: Color32::from_rgb(255, 160, 0),
            flow_fraud: Color32::from_rgb(198, 40, 40),

            // UI
            node_radius: 20.0,
            edge_width: 2.0,
            panel_rounding: Rounding::same(8.0),
        }
    }

    /// Get color for account type.
    pub fn account_color(&self, account_type: crate::models::AccountType) -> Color32 {
        use crate::models::AccountType;
        match account_type {
            AccountType::Asset => self.asset_color,
            AccountType::Liability => self.liability_color,
            AccountType::Equity => self.equity_color,
            AccountType::Revenue => self.revenue_color,
            AccountType::Expense => self.expense_color,
            AccountType::Contra => self.contra_color,
        }
    }

    /// Get stroke for edges.
    pub fn edge_stroke(&self, suspicious: bool, fraud: bool) -> Stroke {
        let color = if fraud {
            self.flow_fraud
        } else if suspicious {
            self.flow_suspicious
        } else {
            self.flow_normal
        };
        Stroke::new(self.edge_width, color)
    }

    /// Get futuristic edge color with transparency based on weight and flow type.
    /// Returns (color, alpha) tuple.
    pub fn edge_color_futuristic(
        &self,
        weight: f32,
        suspicious: bool,
        fraud: bool,
        max_weight: f32,
    ) -> Color32 {
        // Normalize weight for alpha calculation
        let normalized = (weight / max_weight.max(1.0)).min(1.0);

        // Base color selection
        let base = if fraud {
            self.flow_fraud
        } else if suspicious {
            self.flow_suspicious
        } else {
            // Gradient from cyan to blue based on weight
            let t = normalized;
            Color32::from_rgb(
                (60.0 + t * 40.0) as u8,  // 60-100
                (180.0 + t * 40.0) as u8, // 180-220
                (220.0 + t * 35.0) as u8, // 220-255
            )
        };

        // Alpha: lighter for low weight, more visible for high weight
        // Range from 40 (very transparent) to 160 (semi-transparent)
        let alpha = (40.0 + normalized * 120.0) as u8;

        Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), alpha)
    }

    /// Apply theme to egui context.
    pub fn apply(&self, ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();

        style.visuals.dark_mode = self.canvas_bg.r() < 128;
        style.visuals.panel_fill = self.panel_bg;
        style.visuals.window_fill = self.panel_bg;
        style.visuals.widgets.noninteractive.fg_stroke.color = self.text_primary;
        style.visuals.widgets.inactive.fg_stroke.color = self.text_secondary;
        style.visuals.selection.bg_fill = self.accent;
        style.visuals.window_rounding = self.panel_rounding;

        ctx.set_style(style);
    }
}
