//! Main application window.

use eframe::egui::{self, CentralPanel, RichText, SidePanel, TopBottomPanel};
use uuid::Uuid;

use super::canvas::NetworkCanvas;
use super::dashboard::AnalyticsDashboard;
use super::panels::{AlertsPanel, AnalyticsPanel, ControlPanel, EducationalOverlay};
use super::theme::AccNetTheme;
use crate::actors::{CoordinatorConfig, GpuActorRuntime};
use crate::analytics::AnalyticsEngine;
use crate::cuda::{AnalysisRuntime, Backend};
use crate::fabric::{
    Alert, AlertSeverity, CompanyArchetype, DataFabricPipeline, GeneratorConfig, PipelineConfig,
};
use crate::models::{AccountingNetwork, HybridTimestamp};

/// Main application state.
pub struct AccNetApp {
    /// Network canvas.
    canvas: NetworkCanvas,
    /// Control panel.
    controls: ControlPanel,
    /// Analytics panel.
    analytics: AnalyticsPanel,
    /// Analytics dashboard with charts.
    dashboard: AnalyticsDashboard,
    /// Alerts panel.
    alerts: AlertsPanel,
    /// Educational overlay.
    education: EducationalOverlay,
    /// Visual theme.
    theme: AccNetTheme,

    /// Current accounting network.
    network: AccountingNetwork,
    /// Data fabric pipeline.
    pipeline: Option<DataFabricPipeline>,
    /// Analytics engine.
    engine: AnalyticsEngine,
    /// GPU-accelerated analysis runtime.
    analysis_runtime: AnalysisRuntime,
    /// GPU-native actor runtime for ring kernel analytics.
    actor_runtime: GpuActorRuntime,

    /// Frame counter.
    frame_count: u64,
    /// Batches processed.
    batches_processed: u64,
    /// Show debug info.
    show_debug: bool,
    /// Show dashboard.
    show_dashboard: bool,
    /// Last layout update time.
    last_layout_update: std::time::Instant,
}

impl AccNetApp {
    /// Create a new application.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Apply theme
        let theme = AccNetTheme::dark();
        theme.apply(&cc.egui_ctx);

        // Initialize network
        let entity_id = Uuid::new_v4();
        let network = AccountingNetwork::new(entity_id, 2024, 1);

        // Initialize GPU runtime with CUDA preference (auto-fallback to CPU)
        let analysis_runtime = AnalysisRuntime::with_backend(Backend::Auto);
        if analysis_runtime.is_cuda_active() {
            let status = analysis_runtime.status();
            eprintln!(
                "AccNet: GPU acceleration active - {} (CC {}.{})",
                status.cuda_device_name.as_deref().unwrap_or("Unknown"),
                status.cuda_compute_capability.map(|c| c.0).unwrap_or(0),
                status.cuda_compute_capability.map(|c| c.1).unwrap_or(0)
            );
        } else {
            eprintln!("AccNet: CPU backend (CUDA not available)");
        }

        // Initialize GPU-native actor runtime for ring kernel analytics
        let actor_runtime = GpuActorRuntime::new(CoordinatorConfig::default());
        if actor_runtime.is_gpu_active() {
            let status = actor_runtime.status();
            eprintln!(
                "AccNet: Ring Kernel Actor System active - {} kernels",
                status.kernels_launched
            );
        } else {
            eprintln!("AccNet: Ring Kernel Actor System (CPU fallback)");
        }

        Self {
            canvas: NetworkCanvas::new(),
            controls: ControlPanel::new(),
            analytics: AnalyticsPanel::new(),
            dashboard: AnalyticsDashboard::new(),
            alerts: AlertsPanel::new(),
            education: EducationalOverlay::new(),
            theme,
            network,
            pipeline: None,
            engine: AnalyticsEngine::new(),
            analysis_runtime,
            actor_runtime,
            frame_count: 0,
            batches_processed: 0,
            show_debug: false,
            show_dashboard: true,
            last_layout_update: std::time::Instant::now(),
        }
    }

    /// Initialize the pipeline with current settings.
    fn init_pipeline(&mut self) {
        let archetype = match self.controls.archetype_index {
            0 => CompanyArchetype::retail_standard(),
            1 => CompanyArchetype::saas_standard(),
            2 => CompanyArchetype::manufacturing_standard(),
            3 => CompanyArchetype::professional_services_standard(),
            4 => CompanyArchetype::financial_services_standard(),
            _ => CompanyArchetype::retail_standard(),
        };

        // Create pipeline configuration
        let mut config = PipelineConfig::default();
        config.batch_size = self.controls.config.batch_size;
        config.inject_anomalies = self.controls.anomaly_rate > 0.0;
        config.anomaly_config.injection_rate = self.controls.anomaly_rate as f64;

        // Create generator config
        let gen_config = GeneratorConfig {
            seed: Some(42),
            ..Default::default()
        };

        // Create new pipeline
        self.pipeline = Some(DataFabricPipeline::new(archetype, gen_config, config));

        // Get the network from pipeline
        if let Some(ref pipeline) = self.pipeline {
            self.network = pipeline.network().clone();
        }

        // Initialize canvas with network
        self.canvas.initialize(&self.network);
        self.batches_processed = 0;
    }

    /// Process pipeline events.
    fn process_pipeline(&mut self) {
        if !self.controls.running {
            return;
        }

        if self.pipeline.is_none() {
            self.init_pipeline();
        }

        let Some(pipeline) = &mut self.pipeline else {
            return;
        };

        // Generate batches based on speed
        let batches_to_process = (self.controls.speed).ceil() as usize;

        for _ in 0..batches_to_process {
            // Process one tick of the pipeline
            let flows = pipeline.tick();

            if !flows.is_empty() {
                self.batches_processed += 1;
            }
        }

        // Copy network state from pipeline
        self.network = pipeline.network().clone();

        // Run analysis using GPU-accelerated runtime (CUDA if available)
        let analysis = self.analysis_runtime.analyze(&self.network);

        // Also run ring kernel actor-based analytics
        let actor_result = self.actor_runtime.analyze(&self.network);

        // Use actor results for Benford analysis if available
        if actor_result.benford_distribution.iter().sum::<u32>() > 0 {
            // Update dashboard with actor-based Benford analysis
            // (The actor runtime provides more accurate parallel analysis)
        }

        // Update network statistics
        self.network.statistics.suspense_account_count = analysis.stats.suspense_count;
        self.network.statistics.gaap_violation_count = analysis.stats.gaap_violation_count;
        self.network.statistics.fraud_pattern_count = analysis.stats.fraud_pattern_count;

        // Generate alerts for detected issues
        for violation in &analysis.gaap_violations {
            self.alerts.add_alert(Alert {
                id: Uuid::new_v4(),
                severity: match violation.severity {
                    crate::models::ViolationSeverity::Low => AlertSeverity::Low,
                    crate::models::ViolationSeverity::Medium => AlertSeverity::Medium,
                    crate::models::ViolationSeverity::High => AlertSeverity::High,
                    crate::models::ViolationSeverity::Critical => AlertSeverity::Critical,
                },
                alert_type: format!("GAAP: {:?}", violation.violation_type),
                message: violation.violation_type.description().to_string(),
                accounts: vec![violation.source_account, violation.target_account],
                amount: Some(violation.amount),
                timestamp: HybridTimestamp::now(),
            });
        }

        for pattern in &analysis.fraud_patterns {
            let severity = if pattern.risk_score > 0.8 {
                AlertSeverity::Critical
            } else if pattern.risk_score > 0.5 {
                AlertSeverity::High
            } else {
                AlertSeverity::Medium
            };

            self.alerts.add_alert(Alert {
                id: Uuid::new_v4(),
                severity,
                alert_type: format!("Fraud: {:?}", pattern.pattern_type),
                message: pattern.pattern_type.description().to_string(),
                accounts: pattern
                    .involved_accounts
                    .iter()
                    .copied()
                    .filter(|&x| x != 0)
                    .collect(),
                amount: Some(pattern.amount),
                timestamp: HybridTimestamp::now(),
            });
        }

        // Update analytics snapshot
        let snapshot = self.engine.analyze(&self.network);
        self.analytics.update(snapshot);

        // Only update particles periodically, don't reinitialize layout
        // This preserves node positions while refreshing flow animations
        if self.batches_processed % 100 == 0 && self.batches_processed > 0 {
            self.canvas.refresh_particles(&self.network);
        }
    }

    /// Render the top menu bar.
    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Reset").clicked() {
                    self.controls.running = false;
                    self.pipeline = None;
                    self.network = AccountingNetwork::new(Uuid::new_v4(), 2024, 1);
                    self.canvas.initialize(&self.network);
                    self.alerts.clear();
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Exit").clicked() {
                    std::process::exit(0);
                }
            });

            ui.menu_button("View", |ui| {
                ui.checkbox(&mut self.canvas.show_labels, "Show Labels");
                ui.checkbox(&mut self.canvas.show_risk, "Show Risk Indicators");
                ui.checkbox(&mut self.canvas.animate_layout, "Animate Layout");
                ui.separator();
                ui.checkbox(&mut self.show_dashboard, "Show Dashboard");
                ui.separator();
                if ui.button("Reset View").clicked() {
                    self.canvas.reset_view();
                    ui.close_menu();
                }
            });

            ui.menu_button("Theme", |ui| {
                if ui.button("Dark").clicked() {
                    self.theme = AccNetTheme::dark();
                    self.theme.apply(ui.ctx());
                    self.canvas.theme = self.theme.clone();
                    ui.close_menu();
                }
                if ui.button("Light").clicked() {
                    self.theme = AccNetTheme::light();
                    self.theme.apply(ui.ctx());
                    self.canvas.theme = self.theme.clone();
                    ui.close_menu();
                }
            });

            ui.menu_button("Help", |ui| {
                ui.checkbox(&mut self.education.visible, "Show Tutorial");
                ui.separator();
                if ui.button("About").clicked() {
                    // Show about dialog
                    ui.close_menu();
                }
            });

            // Spacer
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.show_debug, "Debug");

                // Backend indicator - shows actual GPU execution status
                let (backend_text, backend_color) = if self.analysis_runtime.is_cuda_active() {
                    let status = self.analysis_runtime.status();
                    let device = status.cuda_device_name.as_deref().unwrap_or("GPU");
                    (
                        format!("GPU: {}", device),
                        egui::Color32::from_rgb(100, 220, 100),
                    )
                } else {
                    match self.analysis_runtime.backend() {
                        Backend::Cuda => (
                            "CUDA (fallback CPU)".to_string(),
                            egui::Color32::from_rgb(220, 180, 100),
                        ),
                        Backend::Cpu => ("CPU".to_string(), egui::Color32::from_rgb(180, 180, 180)),
                        Backend::Auto => {
                            ("AUTO".to_string(), egui::Color32::from_rgb(180, 180, 180))
                        }
                    }
                };
                ui.label(RichText::new(backend_text).color(backend_color).small());
                ui.separator();

                ui.label(
                    RichText::new(format!(
                        "Batches: {} | Flows: {} | FPS: {:.1}",
                        self.batches_processed,
                        self.network.flows.len(),
                        ui.ctx().input(|i| 1.0 / i.predicted_dt)
                    ))
                    .small(),
                );
            });
        });
    }
}

impl eframe::App for AccNetApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frame_count += 1;

        // Check if reset is needed (e.g., company type changed)
        if self.controls.needs_reset {
            self.controls.needs_reset = false;
            self.controls.running = false;
            self.pipeline = None;
            self.network = AccountingNetwork::new(Uuid::new_v4(), 2024, 1);
            self.canvas.initialize(&self.network);
            self.dashboard.reset();
            self.alerts.clear();
            self.batches_processed = 0;
            self.last_layout_update = std::time::Instant::now();
        }

        // Process pipeline
        self.process_pipeline();

        // Periodically update layout edges and warm up (every 5 seconds)
        let now = std::time::Instant::now();
        if now.duration_since(self.last_layout_update).as_secs() >= 5 && self.controls.running {
            self.canvas.layout.update_edges(&self.network);
            self.canvas.layout.warm_up();
            self.last_layout_update = now;
        }

        // Update canvas animations
        self.canvas.update();

        // Top menu bar
        TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            self.menu_bar(ui);
        });

        // Left panel - Controls
        SidePanel::left("controls")
            .default_width(200.0)
            .resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.controls.show(ui, &self.theme);
                });
            });

        // Right panel - Analytics & Alerts
        SidePanel::right("analytics")
            .default_width(280.0)
            .resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.analytics.show(ui, &self.network, &self.theme);
                    ui.add_space(15.0);
                    self.alerts.show(ui, &self.theme);
                    ui.add_space(15.0);
                    ui.separator();
                    ui.add_space(5.0);
                    // Extended analytics with heatmaps, rankings, patterns
                    self.dashboard.update(&self.network, self.frame_count);
                    self.dashboard.show(ui, &self.network, &self.theme);
                });
            });

        // Bottom panel - Dashboard with histograms and charts
        if self.show_dashboard {
            egui::TopBottomPanel::bottom("dashboard")
                .default_height(180.0)
                .resizable(true)
                .show(ctx, |ui| {
                    // Update dashboard
                    self.dashboard.update(&self.network, self.frame_count);

                    ui.horizontal(|ui| {
                        ui.heading(RichText::new("Analytics Dashboard").size(14.0));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("x").clicked() {
                                self.show_dashboard = false;
                            }
                        });
                    });
                    ui.separator();

                    egui::ScrollArea::horizontal().show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // Benford's histogram
                            ui.vertical(|ui| {
                                ui.set_width(180.0);
                                let hist = super::charts::Histogram::benford(
                                    self.dashboard.benford_counts(),
                                );
                                hist.show(ui, &self.theme);
                            });

                            ui.separator();

                            // Account distribution donut
                            ui.vertical(|ui| {
                                ui.set_width(200.0);
                                let counts = self.dashboard.account_type_counts();
                                let chart = super::charts::DonutChart::new("Account Types")
                                    .add("Asset", counts[0] as f64, self.theme.asset_color)
                                    .add("Liability", counts[1] as f64, self.theme.liability_color)
                                    .add("Equity", counts[2] as f64, self.theme.equity_color)
                                    .add("Revenue", counts[3] as f64, self.theme.revenue_color)
                                    .add("Expense", counts[4] as f64, self.theme.expense_color);
                                chart.show(ui, &self.theme);
                            });

                            ui.separator();

                            // Method distribution with explanations
                            ui.vertical(|ui| {
                                ui.set_width(220.0);
                                let dist = super::charts::MethodDistribution::new(
                                    self.dashboard.method_counts(),
                                )
                                .with_explanation(true);
                                dist.show(ui, &self.theme);
                            });

                            ui.separator();

                            // Account balances bar chart
                            ui.vertical(|ui| {
                                ui.set_width(180.0);
                                let balances = self.dashboard.account_type_balances();
                                let chart = super::charts::BalanceBarChart::new("Account Balances")
                                    .add("Asset", balances[0], self.theme.asset_color)
                                    .add("Liability", balances[1], self.theme.liability_color)
                                    .add("Equity", balances[2], self.theme.equity_color)
                                    .add("Revenue", balances[3], self.theme.revenue_color)
                                    .add("Expense", balances[4], self.theme.expense_color);
                                chart.show(ui, &self.theme);
                            });

                            ui.separator();

                            // Issues bar chart
                            ui.vertical(|ui| {
                                ui.set_width(180.0);
                                let chart = super::charts::BarChart::new("Detected Issues")
                                    .add(
                                        "Suspense",
                                        self.network.statistics.suspense_account_count as f64,
                                        self.theme.alert_medium,
                                    )
                                    .add(
                                        "GAAP",
                                        self.network.statistics.gaap_violation_count as f64,
                                        self.theme.alert_high,
                                    )
                                    .add(
                                        "Fraud",
                                        self.network.statistics.fraud_pattern_count as f64,
                                        self.theme.alert_critical,
                                    );
                                chart.show(ui, &self.theme);
                            });

                            ui.separator();

                            // Flow rate sparkline
                            ui.vertical(|ui| {
                                ui.set_width(150.0);
                                let mut sparkline = super::charts::Sparkline::new("Flow Rate");
                                sparkline.color = self.theme.flow_normal;
                                for val in self.dashboard.flow_rate_history() {
                                    sparkline.push(*val);
                                }
                                sparkline.show(ui, &self.theme);

                                ui.add_space(10.0);

                                let mut risk_sparkline =
                                    super::charts::Sparkline::new("Risk Trend");
                                risk_sparkline.color = self.theme.alert_high;
                                for val in self.dashboard.risk_history() {
                                    risk_sparkline.push(*val * 100.0);
                                }
                                risk_sparkline.show(ui, &self.theme);
                            });
                        });
                    });
                });
        }

        // Central panel - Network Canvas
        CentralPanel::default().show(ctx, |ui| {
            self.canvas.show(ui, &self.network);
        });

        // Educational overlay (uses egui Window which takes ctx)
        if self.education.visible {
            egui::Window::new("Learn")
                .collapsible(true)
                .resizable(true)
                .default_width(400.0)
                .show(ctx, |ui| {
                    self.education.show_content(ui, &self.theme);
                });
        }

        // Debug window
        if self.show_debug {
            egui::Window::new("Debug Info")
                .collapsible(true)
                .show(ctx, |ui| {
                    ui.label(format!("Frame: {}", self.frame_count));
                    ui.label(format!("Accounts: {}", self.network.accounts.len()));
                    ui.label(format!("Flows: {}", self.network.flows.len()));
                    ui.label(format!("Particles: {}", self.canvas.particles.count()));
                    ui.label(format!(
                        "Layout converged: {}",
                        self.canvas.layout.converged
                    ));
                    ui.label(format!("Selected node: {:?}", self.canvas.selected_node));
                    ui.label(format!("Hovered node: {:?}", self.canvas.hovered_node));

                    ui.separator();
                    ui.heading("Ring Kernel Actor System");
                    let status = self.actor_runtime.status();
                    ui.label(format!("GPU Active: {}", status.cuda_active));
                    ui.label(format!("Kernels: {}", status.kernels_launched));
                    ui.label(format!("Messages Processed: {}", status.messages_processed));
                    ui.label(format!(
                        "Snapshots: {}",
                        status.coordinator_stats.snapshots_processed
                    ));
                    ui.label(format!(
                        "Avg Processing: {:.2} µs",
                        status.coordinator_stats.avg_processing_time_us
                    ));
                });
        }

        // Request repaint for animation
        ctx.request_repaint();
    }
}

/// Run the application.
pub fn run() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Accounting Network Analytics"),
        ..Default::default()
    };

    eframe::run_native(
        "AccNet",
        options,
        Box::new(|cc| Ok(Box::new(AccNetApp::new(cc)))),
    )
}

#[cfg(test)]
mod tests {
    // App tests would require a graphics context
}
