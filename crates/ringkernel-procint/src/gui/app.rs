//! Main application state and rendering.

use super::canvas::{DfgCanvas, TimelineCanvas};
use super::panels::{ConformancePanel, DfgPanel, FabricPanel, KpiPanel, PatternsPanel};
use super::Theme;
use crate::actors::PipelineCoordinator;
use crate::cuda::GpuStatus;
use crate::fabric::{PipelineConfig, SectorTemplate};
use crate::models::DFGGraph;
use eframe::egui::{self, Color32, Pos2, Rect};
use std::collections::HashMap;

/// View mode for the main canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    Dfg,
    Timeline,
    Variations,
    Split,
}

/// Main application state.
pub struct ProcessIntelligenceApp {
    /// Theme.
    pub theme: Theme,
    /// Pipeline coordinator.
    pub coordinator: PipelineCoordinator,
    /// DFG canvas.
    pub dfg_canvas: DfgCanvas,
    /// Timeline canvas.
    pub timeline_canvas: TimelineCanvas,
    /// Current view mode.
    pub view_mode: ViewMode,
    /// Fabric control panel.
    pub fabric_panel: FabricPanel,
    /// KPI panel.
    pub kpi_panel: KpiPanel,
    /// Patterns panel.
    pub patterns_panel: PatternsPanel,
    /// Conformance panel.
    pub conformance_panel: ConformancePanel,
    /// DFG panel.
    pub dfg_panel: DfgPanel,
    /// Current DFG.
    dfg: DFGGraph,
    /// Partial order traces.
    partial_orders: Vec<crate::models::GpuPartialOrderTrace>,
    /// Frame counter.
    frame_count: u64,
    /// Show settings.
    show_settings: bool,
    /// Last sector (to detect changes).
    last_sector: SectorTemplate,
}

impl Default for ProcessIntelligenceApp {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessIntelligenceApp {
    /// Create a new application.
    pub fn new() -> Self {
        use crate::fabric::HealthcareConfig;
        let initial_sector = SectorTemplate::Healthcare(HealthcareConfig::default());
        let coordinator =
            PipelineCoordinator::new(initial_sector.clone(), PipelineConfig::default());

        let mut dfg_canvas = DfgCanvas::new();

        // Set activity names from sector
        let sector = coordinator.pipeline().sector();
        let names: HashMap<u32, String> = sector
            .activities()
            .iter()
            .enumerate()
            .map(|(i, a)| ((i + 1) as u32, a.name.to_string()))
            .collect();
        dfg_canvas.set_activity_names(names.clone());

        let mut timeline_canvas = TimelineCanvas::new();
        timeline_canvas.set_activity_names(names);

        Self {
            theme: Theme::dark(),
            coordinator,
            dfg_canvas,
            timeline_canvas,
            view_mode: ViewMode::Dfg,
            fabric_panel: FabricPanel::default(),
            kpi_panel: KpiPanel::default(),
            patterns_panel: PatternsPanel,
            conformance_panel: ConformancePanel::default(),
            dfg_panel: DfgPanel,
            dfg: DFGGraph::default(),
            partial_orders: Vec::new(),
            frame_count: 0,
            show_settings: false,
            last_sector: initial_sector,
        }
    }

    /// Run the application.
    pub fn run(self) -> eframe::Result<()> {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1400.0, 900.0])
                .with_min_inner_size([1000.0, 600.0])
                .with_title("RingKernel Process Intelligence"),
            ..Default::default()
        };

        eframe::run_native(
            "Process Intelligence",
            options,
            Box::new(|cc| {
                // Apply theme
                let app = self;
                app.theme.apply(&cc.egui_ctx);
                Ok(Box::new(app))
            }),
        )
    }
}

impl eframe::App for ProcessIntelligenceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frame_count += 1;

        // Check for sector changes and reset views
        if self.fabric_panel.sector != self.last_sector {
            self.last_sector = self.fabric_panel.sector.clone();
            self.reset_views();
        }

        // Process pipeline tick
        if self.coordinator.stats().is_running {
            if let Some(batch) = self.coordinator.tick() {
                // Update DFG from coordinator
                self.dfg = self.coordinator.current_dfg().clone();
                self.partial_orders = self.coordinator.current_partial_orders().to_vec();

                // Update conformance panel
                if let Some(conformance) = self.coordinator.current_conformance() {
                    self.conformance_panel.update(conformance.clone());
                }

                // Spawn tokens for animation
                // Use actual DFG edges for token animation
                if batch.event_count > 0
                    && self.frame_count.is_multiple_of(5)
                    && !self.dfg.edges().is_empty()
                {
                    let edge_idx = (self.frame_count as usize) % self.dfg.edges().len();
                    let edge = &self.dfg.edges()[edge_idx];
                    self.dfg_canvas.tokens.spawn(
                        batch.batch_id,
                        edge.source_activity,
                        edge.target_activity,
                    );
                }
            }

            // Update KPI panel
            let kpis = self.coordinator.analytics().kpi_tracker.kpis();
            self.kpi_panel.update(kpis);

            // Request continuous repaint
            ctx.request_repaint();
        }

        // Update canvases
        self.dfg_canvas.update_layout(&self.dfg);

        // Top panel
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            self.render_top_bar(ui);
        });

        // Bottom status bar
        egui::TopBottomPanel::bottom("status_bar")
            .min_height(28.0)
            .show(ctx, |ui| {
                self.render_status_bar(ui);
            });

        // Left side panel
        egui::SidePanel::left("left_panel")
            .min_width(220.0)
            .max_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.fabric_panel
                        .render(ui, &self.theme, &mut self.coordinator);
                    ui.add_space(12.0);
                    let kpis = self.coordinator.analytics().kpi_tracker.kpis();
                    self.kpi_panel.render(ui, &self.theme, kpis);
                });
            });

        // Right side panel
        egui::SidePanel::right("right_panel")
            .min_width(200.0)
            .max_width(260.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.patterns_panel.render(
                        ui,
                        &self.theme,
                        &self.coordinator.analytics().pattern_aggregator,
                    );
                    ui.add_space(12.0);
                    self.conformance_panel.render(ui, &self.theme);
                    ui.add_space(12.0);
                    let metrics = self.coordinator.analytics().dfg_metrics.metrics();
                    self.dfg_panel.render(ui, &self.theme, metrics);
                });
            });

        // Central panel (canvas)
        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_canvas(ui);
        });
    }
}

impl ProcessIntelligenceApp {
    /// Render top bar.
    fn render_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(
                egui::RichText::new("RingKernel Process Intelligence")
                    .strong()
                    .color(self.theme.accent),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Settings button
                if ui.button("⚙").clicked() {
                    self.show_settings = !self.show_settings;
                }

                ui.separator();

                // View mode buttons
                ui.selectable_value(&mut self.view_mode, ViewMode::Dfg, "DFG");
                ui.selectable_value(&mut self.view_mode, ViewMode::Timeline, "Timeline");
                ui.selectable_value(&mut self.view_mode, ViewMode::Variations, "Variants");
                ui.selectable_value(&mut self.view_mode, ViewMode::Split, "Split");
            });
        });
    }

    /// Render status bar.
    fn render_status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let stats = self.coordinator.stats();
            let kpis = self.coordinator.analytics().kpi_tracker.kpis();
            let gpu_stats = self.coordinator.gpu_stats();

            ui.label(format!("Events: {}", stats.total_events));
            ui.separator();
            ui.label(format!("TPS: {}", kpis.format_throughput()));
            ui.separator();

            // GPU status with color indicator - now uses actual execution stats
            let (gpu_text, gpu_color) = if gpu_stats.is_gpu_active() {
                let launches = gpu_stats.total_launches();
                if launches > 0 {
                    (
                        format!("CUDA ({} kernels)", launches),
                        Color32::from_rgb(0, 200, 100),
                    )
                } else {
                    ("CUDA (ready)".to_string(), Color32::from_rgb(0, 200, 100))
                }
            } else {
                match self.coordinator.dfg_gpu_status() {
                    GpuStatus::CudaReady => ("CUDA (compiling)".to_string(), Color32::YELLOW),
                    GpuStatus::CudaPending => ("CUDA (init)".to_string(), Color32::YELLOW),
                    GpuStatus::CudaError => ("CUDA (err)".to_string(), Color32::RED),
                    GpuStatus::CpuFallback => {
                        ("CPU fallback".to_string(), Color32::from_rgb(100, 150, 200))
                    }
                    GpuStatus::CudaNotCompiled => (
                        "CPU (build with --features cuda)".to_string(),
                        Color32::from_rgb(180, 180, 180),
                    ),
                }
            };
            ui.colored_label(gpu_color, format!("GPU: {}", gpu_text));

            // Show GPU throughput if active
            if gpu_stats.is_gpu_active() && gpu_stats.total_launches() > 0 {
                let throughput = gpu_stats.throughput();
                if throughput > 0.0 {
                    ui.label(format!("| {:.0} elem/s", throughput));
                }
            }
            ui.separator();

            ui.label(format!(
                "Patterns: {}",
                self.coordinator
                    .analytics()
                    .pattern_aggregator
                    .total_detected
            ));

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!("Sector: {}", self.fabric_panel.sector.name()));
            });
        });
    }

    /// Render main canvas.
    fn render_canvas(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();

        // Background
        ui.painter()
            .rect_filled(rect, egui::Rounding::ZERO, self.theme.background);

        match self.view_mode {
            ViewMode::Dfg => {
                self.dfg_canvas.render(ui, &self.theme, &self.dfg, rect);
            }
            ViewMode::Timeline => {
                self.timeline_canvas
                    .render(ui, &self.theme, &self.partial_orders, rect);
            }
            ViewMode::Variations => {
                self.render_variations(ui, rect);
            }
            ViewMode::Split => {
                let mid = rect.center().x;
                let left_rect = Rect::from_min_max(rect.min, Pos2::new(mid - 2.0, rect.max.y));
                let right_rect = Rect::from_min_max(Pos2::new(mid + 2.0, rect.min.y), rect.max);

                self.dfg_canvas
                    .render(ui, &self.theme, &self.dfg, left_rect);

                // Divider
                ui.painter().line_segment(
                    [Pos2::new(mid, rect.top()), Pos2::new(mid, rect.bottom())],
                    egui::Stroke::new(2.0, Color32::from_rgb(38, 38, 48)),
                );

                self.timeline_canvas
                    .render(ui, &self.theme, &self.partial_orders, right_rect);
            }
        }
    }

    /// Reset all views (called when sector changes).
    fn reset_views(&mut self) {
        // Reset DFG
        self.dfg = DFGGraph::default();
        self.partial_orders.clear();

        // Reset canvases
        self.dfg_canvas.reset();
        self.timeline_canvas.reset();

        // Update activity names for new sector
        let sector = self.coordinator.pipeline().sector();
        let names: HashMap<u32, String> = sector
            .activities()
            .iter()
            .enumerate()
            .map(|(i, a)| ((i + 1) as u32, a.name.to_string()))
            .collect();
        self.dfg_canvas.set_activity_names(names.clone());
        self.timeline_canvas.set_activity_names(names);

        // Reset conformance panel
        self.conformance_panel = ConformancePanel::default();
    }

    /// Render process variations view with graph and bar chart.
    fn render_variations(&self, ui: &mut egui::Ui, rect: Rect) {
        let painter = ui.painter_at(rect);

        // Get variations sorted by count
        let mut variations: Vec<_> = self.coordinator.variations().values().collect();
        variations.sort_by_key(|a| std::cmp::Reverse(a.count));

        // Get activity names and colors
        let sector = self.coordinator.pipeline().sector();
        let activity_names: HashMap<u32, String> = sector
            .activities()
            .iter()
            .enumerate()
            .map(|(i, a)| ((i + 1) as u32, a.name.to_string()))
            .collect();

        let activity_colors = self.get_activity_colors(&activity_names);

        let total_cases: u64 = variations.iter().map(|v| v.count).sum();

        // Show "no data" message if empty
        if variations.is_empty() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No process variations yet.\nStart the pipeline to collect data.",
                egui::FontId::proportional(14.0),
                self.theme.text_muted,
            );
            return;
        }

        // Split view: top area for variant graph, bottom for bar chart
        let graph_height = (rect.height() * 0.45).min(250.0);
        let graph_rect = Rect::from_min_size(rect.min, egui::Vec2::new(rect.width(), graph_height));
        let chart_rect = Rect::from_min_max(
            Pos2::new(rect.left(), rect.top() + graph_height + 10.0),
            rect.max,
        );

        // Draw section divider
        painter.line_segment(
            [
                Pos2::new(rect.left() + 20.0, graph_rect.bottom() + 5.0),
                Pos2::new(rect.right() - 20.0, graph_rect.bottom() + 5.0),
            ],
            egui::Stroke::new(1.0, self.theme.panel_bg.linear_multiply(1.5)),
        );

        // === TOP: Process Variant Graph ===
        self.render_variant_graph(
            &painter,
            graph_rect,
            &variations,
            &activity_names,
            &activity_colors,
            total_cases,
        );

        // === BOTTOM: Bar Chart ===
        self.render_variant_bar_chart(
            &painter,
            chart_rect,
            &variations,
            &activity_names,
            total_cases,
        );
    }

    /// Get consistent colors for activities.
    fn get_activity_colors(&self, activity_names: &HashMap<u32, String>) -> HashMap<u32, Color32> {
        let palette = [
            Color32::from_rgb(59, 130, 246), // Blue
            Color32::from_rgb(34, 197, 94),  // Green
            Color32::from_rgb(239, 68, 68),  // Red
            Color32::from_rgb(168, 85, 247), // Purple
            Color32::from_rgb(234, 179, 8),  // Yellow
            Color32::from_rgb(236, 72, 153), // Pink
            Color32::from_rgb(20, 184, 166), // Teal
            Color32::from_rgb(249, 115, 22), // Orange
        ];

        activity_names
            .keys()
            .enumerate()
            .map(|(i, &id)| (id, palette[i % palette.len()]))
            .collect()
    }

    /// Render process variant graph (like Celonis/ProM style).
    fn render_variant_graph(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        variations: &[&crate::actors::ProcessVariation],
        activity_names: &HashMap<u32, String>,
        activity_colors: &HashMap<u32, Color32>,
        total_cases: u64,
    ) {
        // Title
        painter.text(
            Pos2::new(rect.left() + 20.0, rect.top() + 15.0),
            egui::Align2::LEFT_CENTER,
            "Process Variant Graph",
            egui::FontId::proportional(14.0),
            self.theme.text,
        );

        painter.text(
            Pos2::new(rect.right() - 20.0, rect.top() + 15.0),
            egui::Align2::RIGHT_CENTER,
            format!("{} variants, {} cases", variations.len(), total_cases),
            egui::FontId::proportional(10.0),
            self.theme.text_muted,
        );

        // Calculate node positions - show top 5 variants as horizontal flows
        let graph_y_start = rect.top() + 35.0;
        let graph_height = rect.height() - 45.0;
        let row_height = (graph_height / 5.0).min(40.0);
        let node_radius = 14.0;
        let margin_x = 80.0;
        let graph_width = rect.width() - margin_x * 2.0;

        for (var_idx, var) in variations.iter().take(5).enumerate() {
            let y = graph_y_start + var_idx as f32 * row_height + row_height / 2.0;
            let num_activities = var.activities.len().max(1);
            let spacing = graph_width / (num_activities + 1) as f32;

            let percentage = if total_cases > 0 {
                var.count as f32 / total_cases as f32
            } else {
                0.0
            };

            // Draw variant label
            painter.text(
                Pos2::new(rect.left() + 10.0, y),
                egui::Align2::LEFT_CENTER,
                format!("#{} ({:.0}%)", var_idx + 1, percentage * 100.0),
                egui::FontId::proportional(9.0),
                self.theme.text_muted,
            );

            // Draw edges first (so nodes are on top)
            for i in 0..var.activities.len().saturating_sub(1) {
                let x1 = rect.left() + margin_x + (i + 1) as f32 * spacing + node_radius;
                let x2 = rect.left() + margin_x + (i + 2) as f32 * spacing - node_radius;

                // Edge thickness based on variant frequency
                let edge_width = (1.0 + percentage * 4.0).min(3.0);

                painter.line_segment(
                    [Pos2::new(x1, y), Pos2::new(x2, y)],
                    egui::Stroke::new(edge_width, self.theme.accent.linear_multiply(0.6)),
                );

                // Arrow head
                let arrow_size = 5.0;
                painter.line_segment(
                    [Pos2::new(x2 - arrow_size, y - arrow_size), Pos2::new(x2, y)],
                    egui::Stroke::new(edge_width, self.theme.accent.linear_multiply(0.6)),
                );
                painter.line_segment(
                    [Pos2::new(x2 - arrow_size, y + arrow_size), Pos2::new(x2, y)],
                    egui::Stroke::new(edge_width, self.theme.accent.linear_multiply(0.6)),
                );
            }

            // Draw activity nodes
            for (i, &activity_id) in var.activities.iter().enumerate() {
                let x = rect.left() + margin_x + (i + 1) as f32 * spacing;
                let color = activity_colors
                    .get(&activity_id)
                    .copied()
                    .unwrap_or(self.theme.accent);

                // Node circle
                painter.circle_filled(Pos2::new(x, y), node_radius, color);

                // Conformant variant indicator (border)
                if !var.is_conformant {
                    painter.circle_stroke(
                        Pos2::new(x, y),
                        node_radius + 2.0,
                        egui::Stroke::new(2.0, self.theme.error),
                    );
                }

                // Activity abbreviation
                let name = activity_names
                    .get(&activity_id)
                    .map(|n| n.chars().take(2).collect::<String>())
                    .unwrap_or_else(|| format!("{}", activity_id));

                painter.text(
                    Pos2::new(x, y),
                    egui::Align2::CENTER_CENTER,
                    name,
                    egui::FontId::proportional(8.0),
                    Color32::WHITE,
                );
            }

            // Conformance indicator at end
            let conf_x = rect.right() - 25.0;
            let (conf_symbol, conf_color) = if var.is_conformant {
                ("✓", self.theme.success)
            } else {
                ("✗", self.theme.error)
            };
            painter.text(
                Pos2::new(conf_x, y),
                egui::Align2::CENTER_CENTER,
                conf_symbol,
                egui::FontId::proportional(12.0),
                conf_color,
            );
        }
    }

    /// Render variant bar chart (Pareto-style).
    fn render_variant_bar_chart(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        variations: &[&crate::actors::ProcessVariation],
        activity_names: &HashMap<u32, String>,
        total_cases: u64,
    ) {
        // Title
        painter.text(
            Pos2::new(rect.left() + 20.0, rect.top() + 10.0),
            egui::Align2::LEFT_CENTER,
            "Variant Distribution",
            egui::FontId::proportional(14.0),
            self.theme.text,
        );

        let chart_y_start = rect.top() + 30.0;
        let chart_height = rect.height() - 40.0;
        let row_height = (chart_height / 12.0).min(28.0);
        let bar_x_start = rect.left() + 160.0;
        let max_bar_width = rect.width() - 240.0;

        // Cumulative percentage for Pareto line
        let mut cumulative = 0.0;

        for (i, var) in variations.iter().take(12).enumerate() {
            let y = chart_y_start + i as f32 * row_height + row_height / 2.0;

            let percentage = if total_cases > 0 {
                var.count as f32 / total_cases as f32
            } else {
                0.0
            };
            cumulative += percentage;

            // Rank
            painter.text(
                Pos2::new(rect.left() + 15.0, y),
                egui::Align2::LEFT_CENTER,
                format!("#{}", i + 1),
                egui::FontId::proportional(10.0),
                self.theme.text_muted,
            );

            // Count
            painter.text(
                Pos2::new(rect.left() + 45.0, y),
                egui::Align2::LEFT_CENTER,
                format!("{}", var.count),
                egui::FontId::proportional(10.0),
                self.theme.text,
            );

            // Activity sequence preview
            let activity_seq: String = var
                .activities
                .iter()
                .take(4)
                .map(|id| {
                    activity_names
                        .get(id)
                        .map(|n| n.chars().take(2).collect::<String>())
                        .unwrap_or_else(|| "?".to_string())
                })
                .collect::<Vec<_>>()
                .join("→");

            let seq_display = if var.activities.len() > 4 {
                format!("{}…", activity_seq)
            } else {
                activity_seq
            };

            painter.text(
                Pos2::new(rect.left() + 85.0, y),
                egui::Align2::LEFT_CENTER,
                seq_display,
                egui::FontId::proportional(8.0),
                self.theme.text_muted,
            );

            // Bar
            let bar_width = max_bar_width * percentage;
            let bar_rect = Rect::from_min_size(
                Pos2::new(bar_x_start, y - row_height * 0.35),
                egui::Vec2::new(bar_width.max(2.0), row_height * 0.7),
            );

            let bar_color = if var.is_conformant {
                self.theme.success.linear_multiply(0.7)
            } else {
                self.theme.error.linear_multiply(0.7)
            };
            painter.rect_filled(bar_rect, egui::Rounding::same(3.0), bar_color);

            // Percentage label
            painter.text(
                Pos2::new(bar_x_start + bar_width + 5.0, y),
                egui::Align2::LEFT_CENTER,
                format!("{:.1}%", percentage * 100.0),
                egui::FontId::proportional(9.0),
                self.theme.text,
            );

            // Cumulative percentage (Pareto)
            painter.text(
                Pos2::new(rect.right() - 40.0, y),
                egui::Align2::LEFT_CENTER,
                format!("Σ{:.0}%", cumulative * 100.0),
                egui::FontId::proportional(8.0),
                self.theme.text_muted,
            );

            // Duration
            painter.text(
                Pos2::new(rect.right() - 10.0, y),
                egui::Align2::RIGHT_CENTER,
                format!("{:.0}ms", var.avg_duration_ms),
                egui::FontId::proportional(8.0),
                self.theme.text_muted,
            );
        }

        // Show "more variants" indicator if needed
        if variations.len() > 12 {
            let y = chart_y_start + 12.0 * row_height + 5.0;
            painter.text(
                Pos2::new(rect.center().x, y),
                egui::Align2::CENTER_TOP,
                format!("... and {} more variants", variations.len() - 12),
                egui::FontId::proportional(10.0),
                self.theme.text_muted,
            );
        }
    }
}
