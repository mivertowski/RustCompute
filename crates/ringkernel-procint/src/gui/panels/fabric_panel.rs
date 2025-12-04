//! Data fabric control panel.

use crate::actors::PipelineCoordinator;
use crate::fabric::SectorTemplate;
use crate::gui::widgets::{sector_selector, status_dot};
use crate::gui::{section_header, styled_panel, Theme};
use eframe::egui::{self, Ui};

/// Data fabric control panel.
pub struct FabricPanel {
    /// Current sector.
    pub sector: SectorTemplate,
    /// Events per second.
    pub events_per_second: u32,
    /// Deviation rate.
    pub deviation_rate: f32,
}

impl Default for FabricPanel {
    fn default() -> Self {
        use crate::fabric::HealthcareConfig;
        Self {
            sector: SectorTemplate::Healthcare(HealthcareConfig::default()),
            events_per_second: 10000,
            deviation_rate: 0.1,
        }
    }
}

impl FabricPanel {
    /// Render the panel.
    pub fn render(&mut self, ui: &mut Ui, theme: &Theme, coordinator: &mut PipelineCoordinator) {
        styled_panel(ui, theme, |ui| {
            section_header(ui, theme, "DATA FABRIC");

            // Sector selector
            ui.horizontal(|ui| {
                if sector_selector(ui, &mut self.sector) {
                    coordinator.set_sector(self.sector.clone());
                }
            });
            ui.add_space(8.0);

            // Events per second slider
            ui.horizontal(|ui| {
                ui.label("Events/sec:");
                if ui
                    .add(
                        egui::Slider::new(&mut self.events_per_second, 1000..=100000)
                            .logarithmic(true),
                    )
                    .changed()
                {
                    let mut config = coordinator.pipeline().config().generator.clone();
                    config.events_per_second = self.events_per_second;
                    coordinator.pipeline_mut().set_generator_config(config);
                }
            });

            // Deviation rate slider
            ui.horizontal(|ui| {
                ui.label("Deviation:");
                if ui
                    .add(egui::Slider::new(&mut self.deviation_rate, 0.0..=0.5).show_value(true))
                    .changed()
                {
                    let mut config = coordinator.pipeline().config().generator.clone();
                    config.deviation_rate = self.deviation_rate;
                    coordinator.pipeline_mut().set_generator_config(config);
                }
            });

            ui.add_space(12.0);

            // Status indicator
            ui.horizontal(|ui| {
                let is_running = coordinator.stats().is_running;
                let color = if is_running {
                    theme.success
                } else {
                    theme.text_muted
                };
                status_dot(ui, color, is_running);
                ui.label(if is_running { "Running" } else { "Paused" });
            });

            ui.add_space(8.0);

            // Control buttons
            ui.horizontal(|ui| {
                let is_running = coordinator.stats().is_running;

                if is_running {
                    if ui.button("⏸ Pause").clicked() {
                        coordinator.handle_control(crate::actors::ControlMessage::Pause);
                    }
                } else if ui.button("▶ Start").clicked() {
                    coordinator.handle_control(crate::actors::ControlMessage::Start);
                }

                if ui.button("↻ Reset").clicked() {
                    coordinator.handle_control(crate::actors::ControlMessage::Reset);
                }
            });

            ui.add_space(12.0);

            // Sector info
            ui.collapsing("Sector Info", |ui| {
                ui.label(format!("Name: {}", self.sector.name()));
                ui.label(format!("Activities: {}", self.sector.activities().len()));
                ui.label(format!("Transitions: {}", self.sector.transitions().len()));
            });
        });
    }
}
