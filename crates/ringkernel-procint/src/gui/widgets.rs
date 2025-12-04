//! Custom widgets for process intelligence GUI.

use super::Theme;
use eframe::egui::{self, Color32, Pos2, Rect, Rounding, Stroke, Vec2};

/// KPI display widget.
pub fn kpi_card(ui: &mut egui::Ui, theme: &Theme, label: &str, value: &str, trend: Option<f32>) {
    egui::Frame::none()
        .fill(Color32::from_rgb(38, 38, 48))
        .rounding(Rounding::same(6.0))
        .inner_margin(egui::Margin::same(10.0))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new(label)
                        .size(11.0)
                        .color(theme.text_muted),
                );
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(value).size(20.0).strong());
                    if let Some(trend) = trend {
                        let (icon, color) = if trend > 0.0 {
                            ("↑", theme.success)
                        } else if trend < 0.0 {
                            ("↓", theme.error)
                        } else {
                            ("→", theme.text_muted)
                        };
                        ui.label(egui::RichText::new(icon).size(14.0).color(color));
                    }
                });
            });
        });
}

/// Progress bar with label.
pub fn labeled_progress(
    ui: &mut egui::Ui,
    theme: &Theme,
    label: &str,
    progress: f32,
    color: Option<Color32>,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).size(12.0));
        ui.add_space(8.0);
        let bar_width = ui.available_width() - 50.0;
        let (rect, _) = ui.allocate_exact_size(Vec2::new(bar_width, 16.0), egui::Sense::hover());

        // Background
        ui.painter()
            .rect_filled(rect, Rounding::same(4.0), Color32::from_rgb(38, 38, 48));

        // Fill
        let fill_width = rect.width() * progress.clamp(0.0, 1.0);
        let fill_rect = Rect::from_min_size(rect.min, Vec2::new(fill_width, rect.height()));
        ui.painter().rect_filled(
            fill_rect,
            Rounding::same(4.0),
            color.unwrap_or(theme.accent),
        );

        ui.add_space(4.0);
        ui.label(egui::RichText::new(format!("{:.0}%", progress * 100.0)).size(11.0));
    });
}

/// Status indicator dot.
pub fn status_dot(ui: &mut egui::Ui, color: Color32, pulsing: bool) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), egui::Sense::hover());
    let center = rect.center();
    let radius = 4.0;

    if pulsing {
        let time = ui.ctx().input(|i| i.time);
        let alpha = ((time * 3.0).sin() * 0.5 + 0.5) as f32;
        let pulse_color = color.linear_multiply(0.3 + alpha * 0.7);
        ui.painter()
            .circle_filled(center, radius + 2.0, pulse_color.linear_multiply(0.3));
    }

    ui.painter().circle_filled(center, radius, color);
}

/// Mini sparkline chart.
pub fn sparkline(ui: &mut egui::Ui, theme: &Theme, values: &[f32], width: f32, height: f32) {
    if values.is_empty() {
        return;
    }

    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, height), egui::Sense::hover());

    let max_val = values.iter().copied().fold(f32::MIN, f32::max).max(1.0);
    let min_val = values.iter().copied().fold(f32::MAX, f32::min);
    let range = (max_val - min_val).max(1.0);

    let points: Vec<Pos2> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = rect.left() + (i as f32 / values.len().max(1) as f32) * rect.width();
            let y = rect.bottom() - ((v - min_val) / range) * rect.height();
            Pos2::new(x, y)
        })
        .collect();

    // Draw line
    if points.len() > 1 {
        for window in points.windows(2) {
            ui.painter()
                .line_segment([window[0], window[1]], Stroke::new(1.5, theme.accent));
        }
    }

    // Draw dots at endpoints
    if let Some(first) = points.first() {
        ui.painter().circle_filled(*first, 2.0, theme.accent);
    }
    if let Some(last) = points.last() {
        ui.painter().circle_filled(*last, 3.0, theme.accent);
    }
}

/// Pattern badge.
pub fn pattern_badge(ui: &mut egui::Ui, pattern_type: &str, severity_color: Color32, count: u64) {
    egui::Frame::none()
        .fill(severity_color.linear_multiply(0.2))
        .rounding(Rounding::same(4.0))
        .inner_margin(egui::Margin::symmetric(8.0, 4.0))
        .stroke(Stroke::new(1.0, severity_color))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(pattern_type)
                        .size(11.0)
                        .color(severity_color),
                );
                if count > 1 {
                    ui.label(
                        egui::RichText::new(format!("×{}", count))
                            .size(10.0)
                            .color(severity_color.linear_multiply(0.7)),
                    );
                }
            });
        });
}

/// Sector selector dropdown.
pub fn sector_selector(ui: &mut egui::Ui, current: &mut crate::fabric::SectorTemplate) -> bool {
    use crate::fabric::{
        FinanceConfig, HealthcareConfig, IncidentConfig, ManufacturingConfig, SectorTemplate,
    };

    let sectors = [
        SectorTemplate::Healthcare(HealthcareConfig::default()),
        SectorTemplate::Manufacturing(ManufacturingConfig::default()),
        SectorTemplate::Finance(FinanceConfig::default()),
        SectorTemplate::IncidentManagement(IncidentConfig::default()),
    ];

    let mut changed = false;

    egui::ComboBox::from_label("Sector")
        .selected_text(current.name())
        .show_ui(ui, |ui| {
            for sector in &sectors {
                if ui
                    .selectable_value(current, sector.clone(), sector.name())
                    .clicked()
                {
                    changed = true;
                }
            }
        });

    changed
}

/// Conformance gauge (circular progress).
pub fn conformance_gauge(ui: &mut egui::Ui, theme: &Theme, fitness: f32, size: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::hover());
    let center = rect.center();
    let radius = size / 2.0 - 4.0;

    // Background arc
    let stroke_width = 6.0;
    ui.painter().circle_stroke(
        center,
        radius,
        Stroke::new(stroke_width, Color32::from_rgb(38, 38, 48)),
    );

    // Fitness arc
    let color = theme.fitness_color(fitness);

    // Draw arc segments
    let segments = 32;
    let start_angle = -std::f32::consts::FRAC_PI_2;
    for i in 0..segments {
        let t = i as f32 / segments as f32;
        if t > fitness {
            break;
        }
        let a1 = start_angle + t * std::f32::consts::TAU;
        let a2 = start_angle + (i + 1) as f32 / segments as f32 * std::f32::consts::TAU;
        let p1 = center + Vec2::new(a1.cos(), a1.sin()) * radius;
        let p2 = center + Vec2::new(a2.cos(), a2.sin()) * radius;
        ui.painter()
            .line_segment([p1, p2], Stroke::new(stroke_width, color));
    }

    // Center text
    let text = format!("{:.0}%", fitness * 100.0);
    ui.painter().text(
        center,
        egui::Align2::CENTER_CENTER,
        text,
        egui::FontId::proportional(size / 4.0),
        theme.text,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theme() {
        let theme = Theme::dark();
        assert!(theme.fitness_color(0.95) == theme.success);
    }
}
