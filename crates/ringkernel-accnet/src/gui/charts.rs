//! Chart and visualization widgets for the analytics dashboard.
//!
//! Provides reusable chart components:
//! - Histogram for Benford's Law analysis
//! - Bar charts for pattern breakdowns
//! - Donut charts for distributions
//! - Enhanced sparklines with tooltips

use eframe::egui::{self, Color32, Pos2, Rect, Response, Sense, Stroke, Vec2};
use std::f32::consts::PI;

use super::theme::AccNetTheme;

/// Benford's Law expected distribution for digits 1-9.
pub const BENFORD_EXPECTED: [f64; 9] = [
    0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046,
];

/// Histogram chart widget.
pub struct Histogram {
    /// Data values (counts or frequencies).
    pub values: Vec<f64>,
    /// Labels for each bar.
    pub labels: Vec<String>,
    /// Expected values (optional, for comparison overlay).
    pub expected: Option<Vec<f64>>,
    /// Bar color.
    pub bar_color: Color32,
    /// Expected overlay color.
    pub expected_color: Color32,
    /// Title.
    pub title: String,
    /// Height of the chart.
    pub height: f32,
}

impl Histogram {
    /// Create a new histogram.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            values: Vec::new(),
            labels: Vec::new(),
            expected: None,
            bar_color: Color32::from_rgb(100, 150, 230),
            expected_color: Color32::from_rgb(255, 180, 100),
            title: title.into(),
            height: 120.0,
        }
    }

    /// Create a Benford's Law histogram.
    pub fn benford(actual_counts: [usize; 9]) -> Self {
        let total: usize = actual_counts.iter().sum();
        let total_f = total.max(1) as f64;

        let values: Vec<f64> = actual_counts.iter().map(|&c| c as f64 / total_f).collect();
        let labels: Vec<String> = (1..=9).map(|d| d.to_string()).collect();
        let expected = Some(BENFORD_EXPECTED.to_vec());

        Self {
            values,
            labels,
            expected,
            bar_color: Color32::from_rgb(100, 180, 130),
            expected_color: Color32::from_rgb(255, 100, 100),
            title: "Benford's Law Distribution".to_string(),
            height: 100.0,
        }
    }

    /// Set the height.
    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// Render the histogram.
    pub fn show(&self, ui: &mut egui::Ui, theme: &AccNetTheme) -> Response {
        let width = ui.available_width();
        let (response, painter) = ui.allocate_painter(
            Vec2::new(width, self.height + 30.0),
            Sense::hover(),
        );
        let rect = response.rect;

        // Title
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top()),
            egui::Align2::LEFT_TOP,
            &self.title,
            egui::FontId::proportional(11.0),
            theme.text_secondary,
        );

        // Chart area
        let chart_rect = Rect::from_min_max(
            Pos2::new(rect.left() + 20.0, rect.top() + 18.0),
            Pos2::new(rect.right() - 5.0, rect.bottom() - 15.0),
        );

        // Background
        painter.rect_filled(chart_rect, 2.0, Color32::from_rgb(25, 25, 35));

        if self.values.is_empty() {
            painter.text(
                chart_rect.center(),
                egui::Align2::CENTER_CENTER,
                "No data",
                egui::FontId::proportional(10.0),
                theme.text_secondary,
            );
            return response;
        }

        let n = self.values.len();
        let bar_width = (chart_rect.width() - 10.0) / n as f32;
        let gap = bar_width * 0.15;
        let max_val = self.values.iter().copied().fold(0.0_f64, f64::max)
            .max(self.expected.as_ref().map(|e| e.iter().copied().fold(0.0_f64, f64::max)).unwrap_or(0.0))
            .max(0.01);

        // Draw bars
        for (i, &val) in self.values.iter().enumerate() {
            let x = chart_rect.left() + 5.0 + i as f32 * bar_width;
            let bar_height = (val / max_val) as f32 * (chart_rect.height() - 5.0);

            let bar_rect = Rect::from_min_max(
                Pos2::new(x + gap / 2.0, chart_rect.bottom() - bar_height),
                Pos2::new(x + bar_width - gap / 2.0, chart_rect.bottom()),
            );

            painter.rect_filled(bar_rect, 2.0, self.bar_color);

            // Expected value marker (horizontal line)
            if let Some(ref expected) = self.expected {
                if i < expected.len() {
                    let expected_y = chart_rect.bottom() - (expected[i] / max_val) as f32 * (chart_rect.height() - 5.0);
                    painter.line_segment(
                        [
                            Pos2::new(x + gap / 2.0, expected_y),
                            Pos2::new(x + bar_width - gap / 2.0, expected_y),
                        ],
                        Stroke::new(2.0, self.expected_color),
                    );
                }
            }

            // Label
            if i < self.labels.len() {
                painter.text(
                    Pos2::new(x + bar_width / 2.0, chart_rect.bottom() + 2.0),
                    egui::Align2::CENTER_TOP,
                    &self.labels[i],
                    egui::FontId::proportional(9.0),
                    theme.text_secondary,
                );
            }
        }

        // Y-axis scale
        painter.text(
            Pos2::new(chart_rect.left() - 2.0, chart_rect.top()),
            egui::Align2::RIGHT_TOP,
            format!("{:.0}%", max_val * 100.0),
            egui::FontId::proportional(8.0),
            theme.text_secondary,
        );

        response
    }
}

/// Horizontal bar chart for category breakdowns.
pub struct BarChart {
    /// Category names and values.
    pub data: Vec<(String, f64, Color32)>,
    /// Title.
    pub title: String,
    /// Height per bar.
    pub bar_height: f32,
}

impl BarChart {
    /// Create a new bar chart.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            data: Vec::new(),
            title: title.into(),
            bar_height: 18.0,
        }
    }

    /// Add a data point.
    pub fn add(mut self, label: impl Into<String>, value: f64, color: Color32) -> Self {
        self.data.push((label.into(), value, color));
        self
    }

    /// Render the bar chart.
    pub fn show(&self, ui: &mut egui::Ui, theme: &AccNetTheme) -> Response {
        let width = ui.available_width();
        let total_height = 18.0 + self.data.len() as f32 * (self.bar_height + 4.0);

        let (response, painter) = ui.allocate_painter(
            Vec2::new(width, total_height),
            Sense::hover(),
        );
        let rect = response.rect;

        // Title
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top()),
            egui::Align2::LEFT_TOP,
            &self.title,
            egui::FontId::proportional(11.0),
            theme.text_secondary,
        );

        if self.data.is_empty() {
            return response;
        }

        let max_val = self.data.iter().map(|(_, v, _)| *v).fold(0.0_f64, f64::max).max(1.0);
        let label_width = 80.0;
        let bar_area_width = width - label_width - 45.0;

        for (i, (label, value, color)) in self.data.iter().enumerate() {
            let y = rect.top() + 18.0 + i as f32 * (self.bar_height + 4.0);

            // Label
            painter.text(
                Pos2::new(rect.left() + label_width - 5.0, y + self.bar_height / 2.0),
                egui::Align2::RIGHT_CENTER,
                label,
                egui::FontId::proportional(10.0),
                theme.text_primary,
            );

            // Bar
            let bar_width = ((*value / max_val) as f32 * bar_area_width).max(2.0);
            let bar_rect = Rect::from_min_size(
                Pos2::new(rect.left() + label_width, y),
                Vec2::new(bar_width, self.bar_height),
            );
            painter.rect_filled(bar_rect, 2.0, *color);

            // Value
            painter.text(
                Pos2::new(rect.left() + label_width + bar_width + 5.0, y + self.bar_height / 2.0),
                egui::Align2::LEFT_CENTER,
                format!("{}", *value as usize),
                egui::FontId::proportional(10.0),
                theme.text_secondary,
            );
        }

        response
    }
}

/// Donut/pie chart for distributions.
pub struct DonutChart {
    /// Segments: (label, value, color).
    pub segments: Vec<(String, f64, Color32)>,
    /// Title.
    pub title: String,
    /// Outer radius.
    pub radius: f32,
    /// Inner radius (hole).
    pub inner_radius: f32,
}

impl DonutChart {
    /// Create a new donut chart.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            segments: Vec::new(),
            title: title.into(),
            radius: 45.0,
            inner_radius: 25.0,
        }
    }

    /// Add a segment.
    pub fn add(mut self, label: impl Into<String>, value: f64, color: Color32) -> Self {
        self.segments.push((label.into(), value, color));
        self
    }

    /// Render the donut chart.
    pub fn show(&self, ui: &mut egui::Ui, theme: &AccNetTheme) -> Response {
        let width = ui.available_width();
        let height = self.radius * 2.0 + 50.0;

        let (response, painter) = ui.allocate_painter(
            Vec2::new(width, height),
            Sense::hover(),
        );
        let rect = response.rect;

        // Title
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top()),
            egui::Align2::LEFT_TOP,
            &self.title,
            egui::FontId::proportional(11.0),
            theme.text_secondary,
        );

        if self.segments.is_empty() {
            return response;
        }

        let total: f64 = self.segments.iter().map(|(_, v, _)| *v).sum();
        if total == 0.0 {
            return response;
        }

        let center = Pos2::new(rect.left() + self.radius + 10.0, rect.top() + self.radius + 18.0);
        let mut start_angle = -PI / 2.0; // Start at top

        // Draw segments using triangles for proper rendering
        for (_label, value, color) in &self.segments {
            let sweep_angle = (*value / total) as f32 * 2.0 * PI;

            if sweep_angle > 0.01 {
                // Draw arc using small triangular segments
                let steps = (sweep_angle * 30.0).max(8.0) as usize;

                for i in 0..steps {
                    let angle1 = start_angle + sweep_angle * (i as f32 / steps as f32);
                    let angle2 = start_angle + sweep_angle * ((i + 1) as f32 / steps as f32);

                    // Outer triangle
                    let outer1 = Pos2::new(
                        center.x + self.radius * angle1.cos(),
                        center.y + self.radius * angle1.sin(),
                    );
                    let outer2 = Pos2::new(
                        center.x + self.radius * angle2.cos(),
                        center.y + self.radius * angle2.sin(),
                    );
                    let inner1 = Pos2::new(
                        center.x + self.inner_radius * angle1.cos(),
                        center.y + self.inner_radius * angle1.sin(),
                    );
                    let inner2 = Pos2::new(
                        center.x + self.inner_radius * angle2.cos(),
                        center.y + self.inner_radius * angle2.sin(),
                    );

                    // Two triangles to form a quad
                    painter.add(egui::Shape::convex_polygon(
                        vec![outer1, outer2, inner2, inner1],
                        *color,
                        Stroke::NONE,
                    ));
                }

                // Draw segment border
                let end_angle = start_angle + sweep_angle;
                painter.line_segment(
                    [
                        Pos2::new(center.x + self.inner_radius * start_angle.cos(), center.y + self.inner_radius * start_angle.sin()),
                        Pos2::new(center.x + self.radius * start_angle.cos(), center.y + self.radius * start_angle.sin()),
                    ],
                    Stroke::new(1.0, Color32::from_rgb(50, 50, 60)),
                );
                painter.line_segment(
                    [
                        Pos2::new(center.x + self.inner_radius * end_angle.cos(), center.y + self.inner_radius * end_angle.sin()),
                        Pos2::new(center.x + self.radius * end_angle.cos(), center.y + self.radius * end_angle.sin()),
                    ],
                    Stroke::new(1.0, Color32::from_rgb(50, 50, 60)),
                );
            }

            start_angle += sweep_angle;
        }

        // Center text (total)
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            format!("{}", total as usize),
            egui::FontId::proportional(12.0),
            theme.text_primary,
        );

        // Legend (to the right)
        let legend_x = center.x + self.radius + 20.0;
        let legend_y_start = rect.top() + 18.0;

        for (i, (label, value, color)) in self.segments.iter().enumerate() {
            let y = legend_y_start + i as f32 * 14.0;

            // Color dot
            painter.circle_filled(Pos2::new(legend_x, y + 5.0), 4.0, *color);

            // Label and value
            let pct = if total > 0.0 { (*value / total * 100.0) as usize } else { 0 };
            painter.text(
                Pos2::new(legend_x + 10.0, y),
                egui::Align2::LEFT_TOP,
                format!("{} ({}%)", label, pct),
                egui::FontId::proportional(9.0),
                theme.text_secondary,
            );
        }

        response
    }
}

/// Sparkline with area fill and gradient.
pub struct Sparkline {
    /// Data points.
    pub values: Vec<f32>,
    /// Max points to keep.
    pub max_points: usize,
    /// Line color.
    pub color: Color32,
    /// Title.
    pub title: String,
    /// Height.
    pub height: f32,
    /// Show min/max labels.
    pub show_labels: bool,
}

impl Sparkline {
    /// Create a new sparkline.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            values: Vec::new(),
            max_points: 100,
            color: Color32::from_rgb(100, 200, 150),
            title: title.into(),
            height: 50.0,
            show_labels: true,
        }
    }

    /// Add a value.
    pub fn push(&mut self, value: f32) {
        self.values.push(value);
        if self.values.len() > self.max_points {
            self.values.remove(0);
        }
    }

    /// Set color.
    pub fn color(mut self, color: Color32) -> Self {
        self.color = color;
        self
    }

    /// Render the sparkline.
    pub fn show(&self, ui: &mut egui::Ui, theme: &AccNetTheme) -> Response {
        let width = ui.available_width();
        let (response, painter) = ui.allocate_painter(
            Vec2::new(width, self.height + 16.0),
            Sense::hover(),
        );
        let rect = response.rect;

        // Title
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top()),
            egui::Align2::LEFT_TOP,
            &self.title,
            egui::FontId::proportional(11.0),
            theme.text_secondary,
        );

        // Chart area
        let chart_rect = Rect::from_min_max(
            Pos2::new(rect.left() + 5.0, rect.top() + 14.0),
            Pos2::new(rect.right() - 5.0, rect.bottom()),
        );

        // Background
        painter.rect_filled(chart_rect, 2.0, Color32::from_rgb(25, 25, 35));

        if self.values.len() < 2 {
            return response;
        }

        let min_val = self.values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = self.values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = (max_val - min_val).max(0.01);

        let n = self.values.len();
        let points: Vec<Pos2> = self.values
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let x = chart_rect.left() + (i as f32 / (n - 1) as f32) * chart_rect.width();
                let y = chart_rect.bottom() - ((val - min_val) / range) * (chart_rect.height() - 4.0) - 2.0;
                Pos2::new(x, y)
            })
            .collect();

        // Area fill
        let mut area_points = points.clone();
        area_points.push(Pos2::new(chart_rect.right(), chart_rect.bottom()));
        area_points.push(Pos2::new(chart_rect.left(), chart_rect.bottom()));

        let fill_color = Color32::from_rgba_unmultiplied(
            self.color.r(),
            self.color.g(),
            self.color.b(),
            40,
        );
        painter.add(egui::Shape::convex_polygon(
            area_points,
            fill_color,
            Stroke::NONE,
        ));

        // Line
        for i in 0..(points.len() - 1) {
            painter.line_segment([points[i], points[i + 1]], Stroke::new(2.0, self.color));
        }

        // Current value dot
        if let Some(last) = points.last() {
            painter.circle_filled(*last, 3.0, self.color);
        }

        // Labels
        if self.show_labels && !self.values.is_empty() {
            let current = self.values.last().unwrap_or(&0.0);
            painter.text(
                Pos2::new(chart_rect.right() - 3.0, chart_rect.top() + 3.0),
                egui::Align2::RIGHT_TOP,
                format!("{:.1}", current),
                egui::FontId::proportional(9.0),
                self.color,
            );
        }

        response
    }
}

/// Method distribution chart for A-E transformation methods.
pub struct MethodDistribution {
    /// Counts for each method.
    pub counts: [usize; 5],
    /// Whether to show explanation text.
    pub show_explanation: bool,
}

impl MethodDistribution {
    /// Create from counts.
    pub fn new(counts: [usize; 5]) -> Self {
        Self { counts, show_explanation: true }
    }

    /// Set whether to show explanation.
    pub fn with_explanation(mut self, show: bool) -> Self {
        self.show_explanation = show;
        self
    }

    /// Render the distribution.
    pub fn show(&self, ui: &mut egui::Ui, theme: &AccNetTheme) -> Response {
        let labels = ["A", "B", "C", "D", "E"];
        let descriptions = [
            "1:1 direct mapping",      // A
            "n:n amount match",        // B
            "n:m partition",           // C
            "Higher aggregate",        // D
            "Decomposition",           // E
        ];
        let colors = [
            Color32::from_rgb(100, 180, 230), // A - blue
            Color32::from_rgb(150, 200, 130), // B - green
            Color32::from_rgb(230, 180, 100), // C - orange
            Color32::from_rgb(200, 130, 180), // D - purple
            Color32::from_rgb(180, 130, 130), // E - red
        ];

        let total: usize = self.counts.iter().sum();
        let width = ui.available_width();
        let base_height = if self.show_explanation { 95.0 } else { 60.0 };

        let (response, painter) = ui.allocate_painter(
            Vec2::new(width, base_height),
            Sense::hover(),
        );
        let rect = response.rect;

        // Title
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top()),
            egui::Align2::LEFT_TOP,
            "Transformation Methods",
            egui::FontId::proportional(11.0),
            theme.text_secondary,
        );

        let bar_y = rect.top() + 18.0;
        let bar_height = 20.0;
        let bar_width = width - 10.0;

        // Background
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(rect.left() + 5.0, bar_y), Vec2::new(bar_width, bar_height)),
            4.0,
            Color32::from_rgb(40, 40, 50),
        );

        if total == 0 {
            return response;
        }

        // Stacked bar
        let mut x = rect.left() + 5.0;
        for (i, &count) in self.counts.iter().enumerate() {
            let segment_width = (count as f32 / total as f32) * bar_width;
            if segment_width > 1.0 {
                painter.rect_filled(
                    Rect::from_min_size(Pos2::new(x, bar_y), Vec2::new(segment_width, bar_height)),
                    if i == 0 { 4.0 } else if i == 4 { 4.0 } else { 0.0 },
                    colors[i],
                );
                x += segment_width;
            }
        }

        // Legend below
        let legend_y = bar_y + bar_height + 5.0;
        let legend_spacing = bar_width / 5.0;

        for (i, label) in labels.iter().enumerate() {
            let x = rect.left() + 5.0 + legend_spacing * (i as f32 + 0.5);
            let pct = if total > 0 { self.counts[i] * 100 / total } else { 0 };

            painter.circle_filled(Pos2::new(x - 15.0, legend_y + 5.0), 4.0, colors[i]);
            painter.text(
                Pos2::new(x - 8.0, legend_y),
                egui::Align2::LEFT_TOP,
                format!("{}: {}%", label, pct),
                egui::FontId::proportional(9.0),
                theme.text_secondary,
            );
        }

        // Explanation text below legend
        if self.show_explanation {
            let exp_y = legend_y + 18.0;
            let exp_spacing = bar_width / 5.0;

            for (i, desc) in descriptions.iter().enumerate() {
                let x = rect.left() + 5.0 + exp_spacing * (i as f32 + 0.5);
                painter.text(
                    Pos2::new(x, exp_y),
                    egui::Align2::CENTER_TOP,
                    *desc,
                    egui::FontId::proportional(7.5),
                    Color32::from_rgb(120, 120, 135),
                );
            }
        }

        response
    }
}

/// Horizontal bar chart for account balances (in thousands/millions).
pub struct BalanceBarChart {
    /// Category names and values.
    pub data: Vec<(String, f64, Color32)>,
    /// Title.
    pub title: String,
    /// Height per bar.
    pub bar_height: f32,
}

impl BalanceBarChart {
    /// Create a new balance bar chart.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            data: Vec::new(),
            title: title.into(),
            bar_height: 14.0,
        }
    }

    /// Add a data point (value in raw units, will be formatted).
    pub fn add(mut self, label: impl Into<String>, value: f64, color: Color32) -> Self {
        self.data.push((label.into(), value, color));
        self
    }

    /// Format value with K/M suffix.
    fn format_value(value: f64) -> String {
        if value >= 1_000_000.0 {
            format!("{:.1}M", value / 1_000_000.0)
        } else if value >= 1_000.0 {
            format!("{:.1}K", value / 1_000.0)
        } else {
            format!("{:.0}", value)
        }
    }

    /// Render the balance bar chart.
    pub fn show(&self, ui: &mut egui::Ui, theme: &AccNetTheme) -> Response {
        let width = ui.available_width();
        let total_height = 18.0 + self.data.len() as f32 * (self.bar_height + 3.0);

        let (response, painter) = ui.allocate_painter(
            Vec2::new(width, total_height),
            Sense::hover(),
        );
        let rect = response.rect;

        // Title
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top()),
            egui::Align2::LEFT_TOP,
            &self.title,
            egui::FontId::proportional(11.0),
            theme.text_secondary,
        );

        if self.data.is_empty() {
            return response;
        }

        let max_val = self.data.iter().map(|(_, v, _)| *v).fold(0.0_f64, f64::max).max(1.0);
        let label_width = 55.0;
        let value_width = 40.0;
        let bar_area_width = width - label_width - value_width - 15.0;

        for (i, (label, value, color)) in self.data.iter().enumerate() {
            let y = rect.top() + 18.0 + i as f32 * (self.bar_height + 3.0);

            // Label
            painter.text(
                Pos2::new(rect.left() + label_width - 3.0, y + self.bar_height / 2.0),
                egui::Align2::RIGHT_CENTER,
                label,
                egui::FontId::proportional(9.0),
                theme.text_primary,
            );

            // Bar background
            let bar_bg = Rect::from_min_size(
                Pos2::new(rect.left() + label_width, y),
                Vec2::new(bar_area_width, self.bar_height),
            );
            painter.rect_filled(bar_bg, 2.0, Color32::from_rgb(35, 35, 45));

            // Bar fill
            let bar_width = ((*value / max_val) as f32 * bar_area_width).max(2.0);
            let bar_rect = Rect::from_min_size(
                Pos2::new(rect.left() + label_width, y),
                Vec2::new(bar_width, self.bar_height),
            );
            painter.rect_filled(bar_rect, 2.0, *color);

            // Value
            painter.text(
                Pos2::new(rect.left() + label_width + bar_area_width + 5.0, y + self.bar_height / 2.0),
                egui::Align2::LEFT_CENTER,
                Self::format_value(*value),
                egui::FontId::proportional(9.0),
                theme.text_secondary,
            );
        }

        response
    }
}

/// Live stats ticker showing animated counters.
pub struct LiveTicker {
    /// Label-value pairs.
    pub items: Vec<(String, String, Color32)>,
}

impl LiveTicker {
    /// Create a new ticker.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Add an item.
    pub fn add(mut self, label: impl Into<String>, value: impl Into<String>, color: Color32) -> Self {
        self.items.push((label.into(), value.into(), color));
        self
    }

    /// Render the ticker.
    pub fn show(&self, ui: &mut egui::Ui, _theme: &AccNetTheme) -> Response {
        let width = ui.available_width();
        let (response, painter) = ui.allocate_painter(
            Vec2::new(width, 30.0),
            Sense::hover(),
        );
        let rect = response.rect;

        if self.items.is_empty() {
            return response;
        }

        let spacing = width / self.items.len() as f32;

        for (i, (label, value, color)) in self.items.iter().enumerate() {
            let x = rect.left() + spacing * (i as f32 + 0.5);

            // Value (large)
            painter.text(
                Pos2::new(x, rect.top()),
                egui::Align2::CENTER_TOP,
                value,
                egui::FontId::proportional(16.0),
                *color,
            );

            // Label (small)
            painter.text(
                Pos2::new(x, rect.top() + 18.0),
                egui::Align2::CENTER_TOP,
                label,
                egui::FontId::proportional(9.0),
                Color32::from_rgb(150, 150, 160),
            );
        }

        response
    }
}

impl Default for LiveTicker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benford_histogram() {
        let counts = [30, 18, 12, 10, 8, 7, 6, 5, 4];
        let hist = Histogram::benford(counts);
        assert_eq!(hist.values.len(), 9);
        assert_eq!(hist.labels.len(), 9);
    }

    #[test]
    fn test_sparkline() {
        let mut spark = Sparkline::new("Test");
        for i in 0..150 {
            spark.push(i as f32);
        }
        assert_eq!(spark.values.len(), 100);
    }

    #[test]
    fn test_method_distribution() {
        let dist = MethodDistribution::new([100, 50, 30, 15, 5]);
        assert_eq!(dist.counts.iter().sum::<usize>(), 200);
    }
}
