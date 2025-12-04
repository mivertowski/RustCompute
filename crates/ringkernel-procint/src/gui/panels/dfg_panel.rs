//! DFG statistics panel.

use crate::analytics::DFGMetrics;
use crate::gui::{section_header, styled_panel, Theme};
use eframe::egui::{self, Ui};

/// DFG statistics panel.
pub struct DfgPanel;

impl Default for DfgPanel {
    fn default() -> Self {
        Self
    }
}

impl DfgPanel {
    /// Render the panel.
    pub fn render(&mut self, ui: &mut Ui, theme: &Theme, metrics: &DFGMetrics) {
        styled_panel(ui, theme, |ui| {
            section_header(ui, theme, "DFG METRICS");

            // Basic counts
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new("Nodes")
                            .size(11.0)
                            .color(theme.text_muted),
                    );
                    ui.label(
                        egui::RichText::new(format!("{}", metrics.node_count))
                            .size(16.0)
                            .strong(),
                    );
                });
                ui.add_space(20.0);
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new("Edges")
                            .size(11.0)
                            .color(theme.text_muted),
                    );
                    ui.label(
                        egui::RichText::new(format!("{}", metrics.edge_count))
                            .size(16.0)
                            .strong(),
                    );
                });
            });

            ui.add_space(12.0);

            // Graph properties
            ui.collapsing("Graph Properties", |ui| {
                ui.label(format!("Density: {:.3}", metrics.density));
                ui.label(format!("Avg Degree: {:.2}", metrics.avg_degree));
                ui.label(format!("Max In-Degree: {}", metrics.max_in_degree));
                ui.label(format!("Max Out-Degree: {}", metrics.max_out_degree));
                ui.label(format!("Parallelism: {:.2}", metrics.parallelism_factor));
            });

            ui.add_space(8.0);

            // Performance metrics
            ui.collapsing("Performance", |ui| {
                ui.label(format!("Total Events: {}", metrics.total_events));
                ui.label(format!(
                    "Avg Events/Node: {:.1}",
                    metrics.avg_events_per_node
                ));
                ui.label(format!("Avg Duration: {:.0}ms", metrics.avg_duration_ms));
                ui.label(format!("Total Cost: ${:.2}", metrics.total_cost));
            });

            ui.add_space(8.0);

            // Bottleneck indicator
            let bottleneck_color = if metrics.bottleneck_score > 0.7 {
                theme.error
            } else if metrics.bottleneck_score > 0.3 {
                theme.warning
            } else {
                theme.success
            };

            ui.horizontal(|ui| {
                ui.label("Bottleneck Risk:");
                ui.label(
                    egui::RichText::new(format!("{:.0}%", metrics.bottleneck_score * 100.0))
                        .color(bottleneck_color),
                );
            });

            // Complexity score
            let complexity = metrics.complexity_score();
            ui.horizontal(|ui| {
                ui.label("Complexity:");
                ui.label(egui::RichText::new(format!("{:.2}", complexity)).color(
                    if complexity > 0.7 {
                        theme.warning
                    } else {
                        theme.text
                    },
                ));
            });
        });
    }
}
