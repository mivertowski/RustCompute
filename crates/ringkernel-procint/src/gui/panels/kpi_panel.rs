//! KPI dashboard panel.

use crate::analytics::{HealthStatus, ProcessKPIs};
use crate::gui::widgets::{conformance_gauge, kpi_card, sparkline};
use crate::gui::{section_header, styled_panel, Theme};
use eframe::egui::Ui;
use std::collections::VecDeque;

/// KPI dashboard panel.
pub struct KpiPanel {
    /// Throughput history for sparkline.
    throughput_history: VecDeque<f32>,
    /// Fitness history for sparkline.
    fitness_history: VecDeque<f32>,
    /// Max history points.
    max_history: usize,
}

impl Default for KpiPanel {
    fn default() -> Self {
        Self {
            throughput_history: VecDeque::with_capacity(50),
            fitness_history: VecDeque::with_capacity(50),
            max_history: 50,
        }
    }
}

impl KpiPanel {
    /// Update history with new KPIs.
    pub fn update(&mut self, kpis: &ProcessKPIs) {
        self.throughput_history
            .push_back(kpis.events_per_second as f32);
        if self.throughput_history.len() > self.max_history {
            self.throughput_history.pop_front();
        }

        self.fitness_history.push_back(kpis.avg_fitness);
        if self.fitness_history.len() > self.max_history {
            self.fitness_history.pop_front();
        }
    }

    /// Render the panel.
    pub fn render(&mut self, ui: &mut Ui, theme: &Theme, kpis: &ProcessKPIs) {
        styled_panel(ui, theme, |ui| {
            section_header(ui, theme, "KEY METRICS");

            // Throughput
            ui.horizontal(|ui| {
                kpi_card(ui, theme, "Throughput", &kpis.format_throughput(), None);
            });

            // Sparkline for throughput
            let history: Vec<f32> = self.throughput_history.iter().copied().collect();
            if !history.is_empty() {
                sparkline(ui, theme, &history, ui.available_width(), 30.0);
            }

            ui.add_space(12.0);

            // Conformance gauge
            ui.horizontal(|ui| {
                conformance_gauge(ui, theme, kpis.avg_fitness, 80.0);
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new("Fitness")
                            .size(11.0)
                            .color(theme.text_muted),
                    );
                    ui.label(
                        egui::RichText::new(kpis.format_fitness())
                            .size(16.0)
                            .strong(),
                    );

                    let health = kpis.health_status();
                    let health_color = match health {
                        HealthStatus::Excellent => theme.success,
                        HealthStatus::Good => theme.accent,
                        HealthStatus::Warning => theme.warning,
                        HealthStatus::Critical => theme.error,
                    };
                    ui.label(
                        egui::RichText::new(health.name())
                            .size(11.0)
                            .color(health_color),
                    );
                });
            });

            ui.add_space(12.0);

            // Duration
            kpi_card(ui, theme, "Avg Duration", &kpis.format_duration(), None);

            ui.add_space(8.0);

            // Cases and patterns
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new("Cases")
                            .size(11.0)
                            .color(theme.text_muted),
                    );
                    ui.label(
                        egui::RichText::new(format!("{}", kpis.cases_completed))
                            .size(14.0)
                            .strong(),
                    );
                });
                ui.add_space(20.0);
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new("Patterns")
                            .size(11.0)
                            .color(theme.text_muted),
                    );
                    ui.label(
                        egui::RichText::new(format!("{}", kpis.pattern_count))
                            .size(14.0)
                            .strong(),
                    );
                });
            });
        });
    }
}
