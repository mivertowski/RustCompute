//! Patterns display panel.

use crate::analytics::PatternAggregator;
use crate::gui::widgets::pattern_badge;
use crate::gui::{section_header, styled_panel, Theme};
use crate::models::PatternType;
use eframe::egui::{self, Ui};

/// Patterns display panel.
pub struct PatternsPanel;

impl Default for PatternsPanel {
    fn default() -> Self {
        Self
    }
}

impl PatternsPanel {
    /// Render the panel.
    pub fn render(&mut self, ui: &mut Ui, theme: &Theme, aggregator: &PatternAggregator) {
        styled_panel(ui, theme, |ui| {
            section_header(ui, theme, "DETECTED PATTERNS");

            let dist = aggregator.distribution();

            // Pattern badges
            ui.horizontal_wrapped(|ui| {
                if dist.bottleneck > 0 {
                    pattern_badge(ui, "Bottleneck", theme.error, dist.bottleneck);
                }
                if dist.loop_count > 0 {
                    pattern_badge(ui, "Loop", theme.warning, dist.loop_count);
                }
                if dist.rework > 0 {
                    pattern_badge(ui, "Rework", theme.warning, dist.rework);
                }
                if dist.long_running > 0 {
                    pattern_badge(ui, "Long-Running", theme.accent, dist.long_running);
                }
                if dist.circular > 0 {
                    pattern_badge(ui, "Circular", theme.error, dist.circular);
                }
            });

            if dist.total() == 0 {
                ui.label(
                    egui::RichText::new("No patterns detected")
                        .size(11.0)
                        .color(theme.text_muted),
                );
            }

            ui.add_space(12.0);

            // Recent patterns list
            ui.collapsing("Recent Patterns", |ui| {
                let recent = aggregator.top_patterns(5);

                if recent.is_empty() {
                    ui.label(
                        egui::RichText::new("None yet")
                            .size(11.0)
                            .color(theme.text_muted),
                    );
                } else {
                    for pattern in recent {
                        ui.horizontal(|ui| {
                            let (icon, color) = match pattern.pattern_type {
                                PatternType::Bottleneck => ("⚠", theme.error),
                                PatternType::Loop => ("↺", theme.warning),
                                PatternType::Rework => ("↩", theme.warning),
                                PatternType::LongRunning => ("⏱", theme.accent),
                                PatternType::Circular => ("⟳", theme.error),
                                _ => ("•", theme.text_muted),
                            };

                            ui.label(egui::RichText::new(icon).color(color));
                            ui.label(egui::RichText::new(pattern.pattern_type.name()).size(11.0));
                            ui.label(
                                egui::RichText::new(format!(
                                    "({:.0}%)",
                                    pattern.confidence * 100.0
                                ))
                                .size(10.0)
                                .color(theme.text_muted),
                            );
                        });
                    }
                }
            });

            ui.add_space(8.0);

            // Statistics
            ui.collapsing("Statistics", |ui| {
                ui.label(format!("Total detected: {}", aggregator.total_detected));
                ui.label(format!(
                    "Critical: {}",
                    aggregator.count_by_severity(crate::models::PatternSeverity::Critical)
                ));
                ui.label(format!(
                    "Warnings: {}",
                    aggregator.count_by_severity(crate::models::PatternSeverity::Warning)
                ));
            });
        });
    }
}
