//! UI panels for controls, analytics, and alerts.

use super::theme::AccNetTheme;
use crate::analytics::AnalyticsSnapshot;
use crate::fabric::{Alert, AlertSeverity, PipelineConfig};
use crate::models::AccountingNetwork;
use eframe::egui::{self, Color32, RichText, Ui};

/// Control panel for simulation settings.
pub struct ControlPanel {
    /// Pipeline configuration.
    pub config: PipelineConfig,
    /// Whether simulation is running.
    pub running: bool,
    /// Selected company archetype index.
    pub archetype_index: usize,
    /// Previous archetype index (to detect changes).
    prev_archetype_index: usize,
    /// Simulation speed multiplier.
    pub speed: f32,
    /// Anomaly injection rate.
    pub anomaly_rate: f32,
    /// Flag to signal reset needed.
    pub needs_reset: bool,
}

impl ControlPanel {
    /// Create a new control panel.
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
            running: false,
            archetype_index: 0,
            prev_archetype_index: 0,
            speed: 1.0,
            anomaly_rate: 0.05,
            needs_reset: false,
        }
    }

    /// Render the control panel.
    pub fn show(&mut self, ui: &mut Ui, theme: &AccNetTheme) {
        ui.vertical(|ui| {
            ui.heading(RichText::new("Simulation Control").color(theme.text_primary));
            ui.separator();

            // Play/Pause
            ui.horizontal(|ui| {
                let button_text = if self.running { "Pause" } else { "Start" };
                if ui.button(RichText::new(button_text).size(16.0)).clicked() {
                    self.running = !self.running;
                }

                if ui.button("Reset").clicked() {
                    self.running = false;
                    self.needs_reset = true;
                }
            });

            ui.add_space(10.0);

            // Company archetype
            ui.label(RichText::new("Company Type").color(theme.text_secondary));
            egui::ComboBox::from_id_salt("archetype")
                .selected_text(match self.archetype_index {
                    0 => "Retail",
                    1 => "SaaS",
                    2 => "Manufacturing",
                    3 => "Professional Services",
                    4 => "Financial Services",
                    _ => "Retail",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.archetype_index, 0, "Retail");
                    ui.selectable_value(&mut self.archetype_index, 1, "SaaS");
                    ui.selectable_value(&mut self.archetype_index, 2, "Manufacturing");
                    ui.selectable_value(&mut self.archetype_index, 3, "Professional Services");
                    ui.selectable_value(&mut self.archetype_index, 4, "Financial Services");
                });

            // Detect archetype change and trigger reset
            if self.archetype_index != self.prev_archetype_index {
                self.prev_archetype_index = self.archetype_index;
                self.needs_reset = true;
            }

            ui.add_space(10.0);

            // Speed slider
            ui.label(RichText::new("Speed").color(theme.text_secondary));
            ui.add(egui::Slider::new(&mut self.speed, 0.1..=10.0).logarithmic(true));

            ui.add_space(10.0);

            // Anomaly rate
            ui.label(RichText::new("Anomaly Injection Rate").color(theme.text_secondary));
            ui.add(egui::Slider::new(&mut self.anomaly_rate, 0.0..=0.3).show_value(true));

            ui.add_space(10.0);

            // Batch size
            ui.label(RichText::new("Entries per Batch").color(theme.text_secondary));
            let mut batch = self.config.batch_size as f32;
            ui.add(egui::Slider::new(&mut batch, 10.0..=1000.0).logarithmic(true));
            self.config.batch_size = batch as usize;
        });
    }
}

impl Default for ControlPanel {
    fn default() -> Self {
        Self::new()
    }
}

/// Analytics panel showing real-time metrics.
pub struct AnalyticsPanel {
    /// Current snapshot.
    snapshot: Option<AnalyticsSnapshot>,
    /// Historical risk values for sparkline.
    risk_history: Vec<f32>,
    /// Max history points.
    max_history: usize,
}

impl AnalyticsPanel {
    /// Create a new analytics panel.
    pub fn new() -> Self {
        Self {
            snapshot: None,
            risk_history: Vec::new(),
            max_history: 100,
        }
    }

    /// Update with new snapshot.
    pub fn update(&mut self, snapshot: AnalyticsSnapshot) {
        self.risk_history.push(snapshot.overall_risk);
        if self.risk_history.len() > self.max_history {
            self.risk_history.remove(0);
        }
        self.snapshot = Some(snapshot);
    }

    /// Render the analytics panel.
    pub fn show(&mut self, ui: &mut Ui, network: &AccountingNetwork, theme: &AccNetTheme) {
        ui.vertical(|ui| {
            ui.heading(RichText::new("Analytics").color(theme.text_primary));
            ui.separator();

            // Overall risk gauge
            if let Some(ref snapshot) = self.snapshot {
                ui.label(RichText::new("Overall Risk").color(theme.text_secondary));
                self.draw_risk_gauge(ui, snapshot.overall_risk, theme);

                ui.add_space(10.0);

                // Metrics grid
                ui.horizontal(|ui| {
                    self.metric_box(
                        ui,
                        "Accounts",
                        &network.accounts.len().to_string(),
                        theme.asset_color,
                    );
                    self.metric_box(
                        ui,
                        "Flows",
                        &network.flows.len().to_string(),
                        theme.flow_normal,
                    );
                });

                ui.horizontal(|ui| {
                    self.metric_box(
                        ui,
                        "Suspense",
                        &snapshot.suspense_accounts.to_string(),
                        theme.alert_medium,
                    );
                    self.metric_box(
                        ui,
                        "GAAP",
                        &snapshot.gaap_violations.to_string(),
                        theme.alert_high,
                    );
                });

                ui.horizontal(|ui| {
                    self.metric_box(
                        ui,
                        "Fraud",
                        &snapshot.fraud_patterns.to_string(),
                        theme.alert_critical,
                    );
                    let health = format!("{:.0}%", snapshot.network_health.coverage * 100.0);
                    self.metric_box(ui, "Health", &health, theme.equity_color);
                });

                ui.add_space(10.0);

                // Sparkline
                ui.label(RichText::new("Risk Trend").color(theme.text_secondary));
                self.draw_sparkline(ui, theme);
            } else {
                ui.label("No data yet...");
            }
        });
    }

    /// Draw a risk gauge.
    fn draw_risk_gauge(&self, ui: &mut Ui, risk: f32, theme: &AccNetTheme) {
        let (rect, _response) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 30.0), egui::Sense::hover());

        let painter = ui.painter();

        // Background
        painter.rect_filled(rect, 4.0, Color32::from_rgb(40, 40, 50));

        // Fill
        let fill_width = rect.width() * risk;
        let fill_rect = egui::Rect::from_min_size(rect.min, egui::vec2(fill_width, rect.height()));
        let fill_color = self.risk_color(risk, theme);
        painter.rect_filled(fill_rect, 4.0, fill_color);

        // Text
        let text = format!("{:.1}%", risk * 100.0);
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            text,
            egui::FontId::proportional(14.0),
            Color32::WHITE,
        );
    }

    /// Draw a metric box.
    fn metric_box(&self, ui: &mut Ui, label: &str, value: &str, color: Color32) {
        let (rect, _response) =
            ui.allocate_exact_size(egui::vec2(90.0, 55.0), egui::Sense::hover());

        let painter = ui.painter();

        // Background
        painter.rect_filled(
            rect,
            4.0,
            Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 30),
        );
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(
                1.0,
                Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 60),
            ),
        );

        // Value (large, centered)
        painter.text(
            egui::Pos2::new(rect.center().x, rect.top() + 15.0),
            egui::Align2::CENTER_TOP,
            value,
            egui::FontId::proportional(20.0),
            color,
        );

        // Label (smaller, at bottom with padding)
        painter.text(
            egui::Pos2::new(rect.center().x, rect.bottom() - 6.0),
            egui::Align2::CENTER_BOTTOM,
            label,
            egui::FontId::proportional(9.0),
            Color32::from_rgb(160, 160, 175),
        );
    }

    /// Draw sparkline of risk history.
    fn draw_sparkline(&self, ui: &mut Ui, theme: &AccNetTheme) {
        let (rect, _response) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 40.0), egui::Sense::hover());

        if self.risk_history.is_empty() {
            return;
        }

        let painter = ui.painter();

        // Background
        painter.rect_filled(rect, 2.0, Color32::from_rgb(30, 30, 40));

        // Draw line
        let n = self.risk_history.len();
        let points: Vec<egui::Pos2> = self
            .risk_history
            .iter()
            .enumerate()
            .map(|(i, &risk)| {
                let x = rect.left() + (i as f32 / n.max(1) as f32) * rect.width();
                let y = rect.bottom() - risk * rect.height();
                egui::Pos2::new(x, y)
            })
            .collect();

        if points.len() >= 2 {
            for i in 0..(points.len() - 1) {
                painter.line_segment(
                    [points[i], points[i + 1]],
                    egui::Stroke::new(2.0, theme.accent),
                );
            }
        }
    }

    /// Get color for risk level.
    fn risk_color(&self, risk: f32, theme: &AccNetTheme) -> Color32 {
        if risk > 0.7 {
            theme.alert_critical
        } else if risk > 0.5 {
            theme.alert_high
        } else if risk > 0.3 {
            theme.alert_medium
        } else {
            theme.alert_low
        }
    }
}

impl Default for AnalyticsPanel {
    fn default() -> Self {
        Self::new()
    }
}

/// Alerts panel showing detected anomalies.
pub struct AlertsPanel {
    /// Alert history.
    alerts: Vec<Alert>,
    /// Maximum alerts to keep.
    max_alerts: usize,
    /// Filter by severity.
    pub min_severity: AlertSeverity,
}

impl AlertsPanel {
    /// Create a new alerts panel.
    pub fn new() -> Self {
        Self {
            alerts: Vec::new(),
            max_alerts: 100,
            min_severity: AlertSeverity::Low,
        }
    }

    /// Add an alert.
    pub fn add_alert(&mut self, alert: Alert) {
        self.alerts.insert(0, alert);
        if self.alerts.len() > self.max_alerts {
            self.alerts.pop();
        }
    }

    /// Clear all alerts.
    pub fn clear(&mut self) {
        self.alerts.clear();
    }

    /// Render the alerts panel.
    pub fn show(&mut self, ui: &mut Ui, theme: &AccNetTheme) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.heading(RichText::new("Alerts").color(theme.text_primary));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Clear").clicked() {
                        self.clear();
                    }
                });
            });
            ui.separator();

            // Severity filter
            ui.horizontal(|ui| {
                ui.label("Min Severity:");
                egui::ComboBox::from_id_salt("severity_filter")
                    .selected_text(format!("{:?}", self.min_severity))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.min_severity, AlertSeverity::Low, "Low");
                        ui.selectable_value(
                            &mut self.min_severity,
                            AlertSeverity::Medium,
                            "Medium",
                        );
                        ui.selectable_value(&mut self.min_severity, AlertSeverity::High, "High");
                        ui.selectable_value(
                            &mut self.min_severity,
                            AlertSeverity::Critical,
                            "Critical",
                        );
                    });
            });

            ui.add_space(5.0);

            // Scrollable alert list
            egui::ScrollArea::vertical()
                .max_height(300.0)
                .show(ui, |ui| {
                    for alert in &self.alerts {
                        if self.should_show(alert) {
                            self.render_alert(ui, alert, theme);
                        }
                    }

                    if self.alerts.is_empty() {
                        ui.label(RichText::new("No alerts").color(theme.text_secondary));
                    }
                });
        });
    }

    /// Check if alert should be shown based on filter.
    fn should_show(&self, alert: &Alert) -> bool {
        alert.severity as u8 >= self.min_severity as u8
    }

    /// Render a single alert.
    fn render_alert(&self, ui: &mut Ui, alert: &Alert, theme: &AccNetTheme) {
        let color = match alert.severity {
            AlertSeverity::Info => theme.text_secondary,
            AlertSeverity::Low => theme.alert_low,
            AlertSeverity::Medium => theme.alert_medium,
            AlertSeverity::High => theme.alert_high,
            AlertSeverity::Critical => theme.alert_critical,
        };

        egui::Frame::none()
            .fill(Color32::from_rgba_unmultiplied(
                color.r(),
                color.g(),
                color.b(),
                20,
            ))
            .inner_margin(8.0)
            .outer_margin(2.0)
            .rounding(4.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    // Severity indicator
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(4.0, 40.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 2.0, color);

                    ui.vertical(|ui| {
                        // Alert type as title
                        ui.label(RichText::new(&alert.alert_type).strong().color(color));

                        // Message
                        ui.label(
                            RichText::new(&alert.message)
                                .small()
                                .color(theme.text_secondary),
                        );

                        // Timestamp
                        let time = chrono::DateTime::from_timestamp(
                            (alert.timestamp.physical / 1000) as i64,
                            0,
                        )
                        .map(|dt| dt.format("%H:%M:%S").to_string())
                        .unwrap_or_else(|| "Unknown".to_string());
                        ui.label(RichText::new(time).small().color(theme.text_secondary));
                    });
                });
            });
    }
}

impl Default for AlertsPanel {
    fn default() -> Self {
        Self::new()
    }
}

/// Educational overlay for explaining concepts.
pub struct EducationalOverlay {
    /// Whether overlay is visible.
    pub visible: bool,
    /// Current topic.
    pub topic: EducationalTopic,
}

/// Educational topics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EducationalTopic {
    /// Double-entry bookkeeping.
    DoubleEntry,
    /// Fraud patterns.
    FraudPatterns,
    /// GAAP compliance.
    GaapCompliance,
    /// Benford's Law.
    BenfordsLaw,
    /// Network metrics.
    NetworkMetrics,
}

impl EducationalOverlay {
    /// Create a new overlay.
    pub fn new() -> Self {
        Self {
            visible: false,
            topic: EducationalTopic::DoubleEntry,
        }
    }

    /// Render the overlay.
    pub fn show(&mut self, ui: &mut Ui, theme: &AccNetTheme) {
        if !self.visible {
            return;
        }

        egui::Window::new("Learn")
            .collapsible(true)
            .resizable(true)
            .default_width(400.0)
            .show(ui.ctx(), |ui| {
                self.show_content(ui, theme);
            });
    }

    /// Render the overlay content (called from app.rs window).
    pub fn show_content(&mut self, ui: &mut Ui, theme: &AccNetTheme) {
        // Topic tabs
        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.topic == EducationalTopic::DoubleEntry, "Double Entry")
                .clicked()
            {
                self.topic = EducationalTopic::DoubleEntry;
            }
            if ui
                .selectable_label(self.topic == EducationalTopic::FraudPatterns, "Fraud")
                .clicked()
            {
                self.topic = EducationalTopic::FraudPatterns;
            }
            if ui
                .selectable_label(self.topic == EducationalTopic::GaapCompliance, "GAAP")
                .clicked()
            {
                self.topic = EducationalTopic::GaapCompliance;
            }
            if ui
                .selectable_label(self.topic == EducationalTopic::BenfordsLaw, "Benford")
                .clicked()
            {
                self.topic = EducationalTopic::BenfordsLaw;
            }
        });

        ui.separator();

        match self.topic {
            EducationalTopic::DoubleEntry => self.show_double_entry(ui, theme),
            EducationalTopic::FraudPatterns => self.show_fraud_patterns(ui, theme),
            EducationalTopic::GaapCompliance => self.show_gaap(ui, theme),
            EducationalTopic::BenfordsLaw => self.show_benford(ui, theme),
            EducationalTopic::NetworkMetrics => self.show_metrics(ui, theme),
        }
    }

    fn show_double_entry(&self, ui: &mut Ui, theme: &AccNetTheme) {
        ui.label(
            RichText::new("Double-Entry Bookkeeping")
                .heading()
                .color(theme.text_primary),
        );
        ui.add_space(5.0);
        ui.label("Every financial transaction is recorded in at least two accounts:");
        ui.label("- One account is debited (left side)");
        ui.label("- One account is credited (right side)");
        ui.add_space(10.0);
        ui.label(RichText::new("The Accounting Equation:").strong());
        ui.label("Assets = Liabilities + Equity");
        ui.add_space(10.0);
        ui.label("In this visualization, each node is an account, and edges represent money flowing between accounts.");
    }

    fn show_fraud_patterns(&self, ui: &mut Ui, theme: &AccNetTheme) {
        ui.label(
            RichText::new("Fraud Pattern Detection")
                .heading()
                .color(theme.text_primary),
        );
        ui.add_space(5.0);
        ui.label("Common fraud patterns we detect:");
        ui.add_space(5.0);

        let patterns = [
            (
                "Circular Flows",
                "Money cycling through accounts back to origin",
            ),
            (
                "Benford Violation",
                "Digit distribution doesn't match natural patterns",
            ),
            (
                "Threshold Clustering",
                "Amounts clustered just below approval limits",
            ),
            ("Round Amounts", "Excessive use of round numbers"),
            ("Timing Anomalies", "Unusual transaction timing patterns"),
        ];

        for (name, desc) in patterns {
            ui.horizontal(|ui| {
                ui.label(RichText::new(name).strong().color(theme.alert_high));
                ui.label(RichText::new(format!(" - {}", desc)).color(theme.text_secondary));
            });
        }
    }

    fn show_gaap(&self, ui: &mut Ui, theme: &AccNetTheme) {
        ui.label(
            RichText::new("GAAP Compliance")
                .heading()
                .color(theme.text_primary),
        );
        ui.add_space(5.0);
        ui.label("Generally Accepted Accounting Principles (GAAP) violations:");
        ui.add_space(5.0);

        let violations = [
            ("Revenue → Cash", "Revenue should flow through A/R first"),
            (
                "Revenue → Expense",
                "Revenue shouldn't directly offset expenses",
            ),
            ("Asset → Equity", "Direct transfers bypass income statement"),
        ];

        for (name, desc) in violations {
            ui.horizontal(|ui| {
                ui.label(RichText::new(name).strong().color(theme.alert_medium));
                ui.label(RichText::new(format!(" - {}", desc)).color(theme.text_secondary));
            });
        }
    }

    fn show_benford(&self, ui: &mut Ui, theme: &AccNetTheme) {
        ui.label(
            RichText::new("Benford's Law")
                .heading()
                .color(theme.text_primary),
        );
        ui.add_space(5.0);
        ui.label("In naturally occurring datasets, leading digits follow a specific distribution:");
        ui.add_space(5.0);

        let expected = [
            ("1", "30.1%"),
            ("2", "17.6%"),
            ("3", "12.5%"),
            ("4", "9.7%"),
            ("5", "7.9%"),
            ("6", "6.7%"),
            ("7", "5.8%"),
            ("8", "5.1%"),
            ("9", "4.6%"),
        ];

        ui.horizontal_wrapped(|ui| {
            for (digit, pct) in expected {
                ui.label(RichText::new(format!("{}: {}", digit, pct)).monospace());
            }
        });

        ui.add_space(10.0);
        ui.label("Fraudulent data often violates this distribution because humans tend to fabricate 'random' numbers differently.");
    }

    fn show_metrics(&self, ui: &mut Ui, theme: &AccNetTheme) {
        ui.label(
            RichText::new("Network Metrics")
                .heading()
                .color(theme.text_primary),
        );
        ui.add_space(5.0);
        ui.label("We analyze the accounting network using graph theory:");
        ui.add_space(5.0);

        let metrics = [
            ("PageRank", "Identifies central/important accounts"),
            ("Density", "How interconnected the network is"),
            ("Clustering", "Groups of tightly connected accounts"),
        ];

        for (name, desc) in metrics {
            ui.horizontal(|ui| {
                ui.label(RichText::new(name).strong().color(theme.accent));
                ui.label(RichText::new(format!(" - {}", desc)).color(theme.text_secondary));
            });
        }
    }
}

impl Default for EducationalOverlay {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_panel() {
        let panel = ControlPanel::new();
        assert!(!panel.running);
    }

    #[test]
    fn test_analytics_panel() {
        let panel = AnalyticsPanel::new();
        assert!(panel.snapshot.is_none());
    }

    #[test]
    fn test_alerts_panel() {
        let panel = AlertsPanel::new();
        assert!(panel.alerts.is_empty());
    }
}
