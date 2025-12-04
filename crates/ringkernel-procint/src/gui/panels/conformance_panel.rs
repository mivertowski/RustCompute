//! Conformance checking results panel.

use crate::gui::widgets::labeled_progress;
use crate::gui::{section_header, styled_panel, Theme};
use crate::kernels::ConformanceCheckResult;
use eframe::egui::{self, Ui};

/// Conformance results panel.
#[derive(Default)]
pub struct ConformancePanel {
    /// Last conformance results.
    pub last_results: Option<ConformanceCheckResult>,
}

impl ConformancePanel {
    /// Update with new results.
    pub fn update(&mut self, results: ConformanceCheckResult) {
        self.last_results = Some(results);
    }

    /// Render the panel.
    pub fn render(&mut self, ui: &mut Ui, theme: &Theme) {
        styled_panel(ui, theme, |ui| {
            section_header(ui, theme, "CONFORMANCE");

            if let Some(results) = &self.last_results {
                let dist = results.distribution();
                let total = dist.total();

                // Fitness score
                let avg_fitness = results.avg_fitness();
                let fitness_color = theme.fitness_color(avg_fitness);

                ui.horizontal(|ui| {
                    ui.label("Fitness:");
                    ui.label(
                        egui::RichText::new(format!("{:.1}%", avg_fitness * 100.0))
                            .size(16.0)
                            .strong()
                            .color(fitness_color),
                    );
                });

                ui.add_space(8.0);

                // Compliance distribution
                if total > 0 {
                    let percentages = dist.percentages();

                    labeled_progress(
                        ui,
                        theme,
                        "Fully",
                        percentages[0] / 100.0,
                        Some(theme.success),
                    );
                    labeled_progress(
                        ui,
                        theme,
                        "Mostly",
                        percentages[1] / 100.0,
                        Some(egui::Color32::from_rgb(34, 211, 238)),
                    );
                    labeled_progress(
                        ui,
                        theme,
                        "Partial",
                        percentages[2] / 100.0,
                        Some(theme.warning),
                    );
                    labeled_progress(ui, theme, "Non", percentages[3] / 100.0, Some(theme.error));
                }

                ui.add_space(8.0);

                // Summary counts
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("✓ {}", results.conformant_count()))
                            .color(theme.success),
                    );
                    ui.label(
                        egui::RichText::new(format!("✗ {}", results.non_conformant_count()))
                            .color(theme.error),
                    );
                });

                ui.add_space(8.0);

                // Deviations
                ui.collapsing("Deviations", |ui| {
                    let mut missing_total = 0u16;
                    let mut extra_total = 0u16;

                    for result in &results.results {
                        missing_total += result.missing_count;
                        extra_total += result.extra_count;
                    }

                    ui.label(format!("Missing activities: {}", missing_total));
                    ui.label(format!("Extra activities: {}", extra_total));
                    ui.label(format!("Traces checked: {}", results.results.len()));
                });
            } else {
                ui.label(
                    egui::RichText::new("No conformance data yet")
                        .size(11.0)
                        .color(theme.text_muted),
                );
            }
        });
    }
}
