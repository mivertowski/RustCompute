//! Analytics dashboard with live visualizations.
//!
//! Provides a comprehensive view of network health and detected patterns
//! using charts, histograms, heatmaps, and real-time metrics.

use eframe::egui::{self, Color32, RichText, Ui};
use std::collections::{HashMap, VecDeque};

use super::charts::{BarChart, DonutChart, Histogram, LiveTicker, MethodDistribution, Sparkline};
use super::heatmaps::{ActivityHeatmap, CorrelationHeatmap, RiskHeatmap};
use super::rankings::{AmountDistribution, PatternStatsPanel, TopAccountsPanel};
use super::theme::AccNetTheme;
use crate::models::{AccountType, AccountingNetwork, SolvingMethod};

/// Analytics dashboard with live visualizations.
pub struct AnalyticsDashboard {
    /// Benford digit counts (digits 1-9).
    benford_counts: [usize; 9],
    /// Fraud pattern counts by type.
    fraud_counts: [usize; 8],
    /// GAAP violation counts by type.
    gaap_counts: [usize; 6],
    /// Account type counts.
    account_type_counts: [usize; 5],
    /// Account type balances (accumulated values).
    account_type_balances: [f64; 5],
    /// Method distribution.
    method_counts: [usize; 5],
    /// Transaction volume history.
    volume_history: VecDeque<f32>,
    /// Risk score history.
    risk_history: VecDeque<f32>,
    /// Flow rate history (flows per tick).
    flow_rate_history: VecDeque<f32>,
    /// Max history points.
    max_history: usize,
    /// Total flows processed.
    total_flows: usize,
    /// Total entries processed.
    #[allow(dead_code)]
    total_entries: usize,
    /// Last update frame.
    last_update_frame: u64,
    /// Account metadata for display names.
    account_metadata: HashMap<u16, String>,
    /// Current view mode for dashboard sections.
    expanded_sections: DashboardSections,
}

/// Tracks which dashboard sections are expanded/collapsed.
#[derive(Default)]
pub struct DashboardSections {
    /// Show rankings panel.
    pub show_rankings: bool,
    /// Show heatmaps.
    pub show_heatmaps: bool,
    /// Show pattern detection.
    pub show_patterns: bool,
    /// Show amount distribution.
    pub show_amounts: bool,
}

impl AnalyticsDashboard {
    /// Create a new dashboard.
    pub fn new() -> Self {
        Self {
            benford_counts: [0; 9],
            fraud_counts: [0; 8],
            gaap_counts: [0; 6],
            account_type_counts: [0; 5],
            account_type_balances: [0.0; 5],
            method_counts: [0; 5],
            volume_history: VecDeque::with_capacity(100),
            risk_history: VecDeque::with_capacity(100),
            flow_rate_history: VecDeque::with_capacity(100),
            max_history: 100,
            total_flows: 0,
            total_entries: 0,
            last_update_frame: 0,
            account_metadata: HashMap::new(),
            expanded_sections: DashboardSections {
                show_rankings: true,
                show_heatmaps: true,
                show_patterns: true,
                show_amounts: true,
            },
        }
    }

    /// Set account metadata for display.
    pub fn set_account_metadata(&mut self, metadata: HashMap<u16, String>) {
        self.account_metadata = metadata;
    }

    /// Update the dashboard from network state.
    pub fn update(&mut self, network: &AccountingNetwork, frame: u64) {
        // Avoid updating too frequently
        if frame == self.last_update_frame {
            return;
        }
        self.last_update_frame = frame;

        // Reset counts
        self.benford_counts = [0; 9];
        self.fraud_counts = [0; 8];
        self.gaap_counts = [0; 6];
        self.account_type_counts = [0; 5];
        self.account_type_balances = [0.0; 5];
        self.method_counts = [0; 5];

        // Count account types and sum balances
        for account in &network.accounts {
            let idx = match account.account_type {
                AccountType::Asset => 0,
                AccountType::Liability => 1,
                AccountType::Equity => 2,
                AccountType::Revenue => 3,
                AccountType::Expense => 4,
                AccountType::Contra => 0, // Count contra with assets for simplicity
            };
            self.account_type_counts[idx] += 1;
            self.account_type_balances[idx] += account.closing_balance.to_f64().abs();
        }

        // Analyze flows for Benford's Law
        for flow in &network.flows {
            let amount = flow.amount.to_f64().abs();
            if amount >= 1.0 {
                // Get first digit
                let first_digit = get_first_digit(amount);
                if (1..=9).contains(&first_digit) {
                    self.benford_counts[(first_digit - 1) as usize] += 1;
                }
            }

            // Count solving methods
            let method_idx = match flow.method_used {
                SolvingMethod::MethodA => 0,
                SolvingMethod::MethodB => 1,
                SolvingMethod::MethodC => 2,
                SolvingMethod::MethodD => 3,
                SolvingMethod::MethodE => 4,
                SolvingMethod::Pending => continue, // Skip pending flows in count
            };
            self.method_counts[method_idx] += 1;
        }

        // Count fraud patterns (from network statistics)
        // We'll derive counts from network patterns if available
        self.fraud_counts[0] = network.statistics.fraud_pattern_count / 3; // Rough estimate
        self.fraud_counts[2] = self.calculate_benford_violations();

        // Update totals
        let new_flows = network.flows.len();
        let flow_rate = (new_flows as i64 - self.total_flows as i64).max(0) as f32;
        self.total_flows = new_flows;

        // Update history
        self.flow_rate_history.push_back(flow_rate);
        if self.flow_rate_history.len() > self.max_history {
            self.flow_rate_history.pop_front();
        }

        // Risk score from network
        let risk = self.calculate_risk_score(network);
        self.risk_history.push_back(risk);
        if self.risk_history.len() > self.max_history {
            self.risk_history.pop_front();
        }

        // Volume (cumulative flows)
        let volume = network
            .flows
            .iter()
            .map(|f| f.amount.to_f64().abs())
            .sum::<f64>() as f32
            / 1000.0; // In thousands
        self.volume_history.push_back(volume);
        if self.volume_history.len() > self.max_history {
            self.volume_history.pop_front();
        }
    }

    /// Calculate Benford violation count.
    fn calculate_benford_violations(&self) -> usize {
        let total: usize = self.benford_counts.iter().sum();
        if total < 50 {
            return 0;
        }

        // Chi-squared test
        let expected = [
            0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046,
        ];
        let mut chi_sq = 0.0;

        for (i, &count) in self.benford_counts.iter().enumerate() {
            let observed = count as f64 / total as f64;
            let exp = expected[i];
            chi_sq += (observed - exp).powi(2) / exp;
        }

        // Critical value at 0.05 significance with 8 df is 15.507
        if chi_sq > 15.507 {
            1
        } else {
            0
        }
    }

    /// Calculate overall risk score.
    /// Uses a bounded calculation that provides meaningful variation between 0-100%.
    fn calculate_risk_score(&self, network: &AccountingNetwork) -> f32 {
        let n = network.accounts.len().max(1) as f32;
        let flows = network.flows.len().max(1) as f32;

        // Suspense: proportion of accounts flagged as suspense (0-1 range)
        // Cap at 20% suspense accounts = max risk from this factor
        let suspense_ratio = (network.statistics.suspense_account_count as f32 / n).min(0.2) / 0.2;

        // GAAP violations: per 100 flows, capped at 10%
        // Having 10 violations per 100 flows = max risk from this factor
        let violation_ratio =
            (network.statistics.gaap_violation_count as f32 / flows * 100.0).min(10.0) / 10.0;

        // Fraud patterns: per 100 flows, capped at 5%
        // Having 5 fraud patterns per 100 flows = max risk from this factor
        let fraud_ratio =
            (network.statistics.fraud_pattern_count as f32 / flows * 100.0).min(5.0) / 5.0;

        // Benford's Law anomaly (binary: 0 or 1)
        let benford_risk = self.calculate_benford_violations() as f32;

        // Confidence: lower average confidence = higher risk
        let avg_confidence = network.statistics.avg_confidence as f32;
        let confidence_risk = (1.0 - avg_confidence).clamp(0.0, 1.0);

        // Weighted combination - each factor contributes proportionally
        // Weights sum to 1.0
        let risk = 0.20 * suspense_ratio
            + 0.25 * violation_ratio
            + 0.30 * fraud_ratio
            + 0.10 * benford_risk
            + 0.15 * confidence_risk;

        risk.clamp(0.0, 1.0)
    }

    /// Render the full dashboard.
    pub fn show(&mut self, ui: &mut Ui, network: &AccountingNetwork, theme: &AccNetTheme) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            // Live ticker at top
            self.show_ticker(ui, network, theme);
            ui.add_space(10.0);

            // Benford's Law histogram
            self.show_benford(ui, theme);
            ui.add_space(15.0);

            // Account type distribution
            self.show_account_distribution(ui, theme);
            ui.add_space(15.0);

            // Method distribution
            self.show_method_distribution(ui, theme);
            ui.add_space(15.0);

            // Fraud pattern breakdown
            self.show_fraud_breakdown(ui, network, theme);
            ui.add_space(15.0);

            // Flow rate sparkline
            self.show_flow_rate(ui, theme);
            ui.add_space(10.0);

            // Risk trend sparkline
            self.show_risk_trend(ui, theme);
            ui.add_space(15.0);

            // === NEW ANALYTICS SECTIONS ===

            // Section header with toggle
            ui.separator();
            ui.add_space(5.0);

            // Pattern Detection (Collapsible)
            let patterns_header = if self.expanded_sections.show_patterns {
                "▼ Pattern Detection"
            } else {
                "▶ Pattern Detection"
            };
            if ui
                .selectable_label(
                    self.expanded_sections.show_patterns,
                    RichText::new(patterns_header)
                        .color(theme.text_primary)
                        .size(12.0),
                )
                .clicked()
            {
                self.expanded_sections.show_patterns = !self.expanded_sections.show_patterns;
            }
            if self.expanded_sections.show_patterns {
                ui.add_space(5.0);
                self.show_pattern_detection(ui, network, theme);
                ui.add_space(10.0);
            }

            // Top Accounts Rankings (Collapsible)
            let rankings_header = if self.expanded_sections.show_rankings {
                "▼ Top Accounts"
            } else {
                "▶ Top Accounts"
            };
            if ui
                .selectable_label(
                    self.expanded_sections.show_rankings,
                    RichText::new(rankings_header)
                        .color(theme.text_primary)
                        .size(12.0),
                )
                .clicked()
            {
                self.expanded_sections.show_rankings = !self.expanded_sections.show_rankings;
            }
            if self.expanded_sections.show_rankings {
                ui.add_space(5.0);
                self.show_top_accounts(ui, network, theme);
                ui.add_space(10.0);
            }

            // Amount Distribution (Collapsible)
            let amounts_header = if self.expanded_sections.show_amounts {
                "▼ Amount Distribution"
            } else {
                "▶ Amount Distribution"
            };
            if ui
                .selectable_label(
                    self.expanded_sections.show_amounts,
                    RichText::new(amounts_header)
                        .color(theme.text_primary)
                        .size(12.0),
                )
                .clicked()
            {
                self.expanded_sections.show_amounts = !self.expanded_sections.show_amounts;
            }
            if self.expanded_sections.show_amounts {
                ui.add_space(5.0);
                self.show_amount_distribution(ui, network, theme);
                ui.add_space(10.0);
            }

            // Heatmaps (Collapsible)
            let heatmaps_header = if self.expanded_sections.show_heatmaps {
                "▼ Heatmaps"
            } else {
                "▶ Heatmaps"
            };
            if ui
                .selectable_label(
                    self.expanded_sections.show_heatmaps,
                    RichText::new(heatmaps_header)
                        .color(theme.text_primary)
                        .size(12.0),
                )
                .clicked()
            {
                self.expanded_sections.show_heatmaps = !self.expanded_sections.show_heatmaps;
            }
            if self.expanded_sections.show_heatmaps {
                ui.add_space(5.0);
                self.show_heatmaps(ui, network, theme);
                ui.add_space(10.0);
            }
        });
    }

    /// Show pattern detection panel.
    fn show_pattern_detection(
        &self,
        ui: &mut Ui,
        network: &AccountingNetwork,
        theme: &AccNetTheme,
    ) {
        let stats = PatternStatsPanel::from_network(network);
        stats.show(ui, theme);
    }

    /// Show top accounts rankings.
    fn show_top_accounts(&self, ui: &mut Ui, network: &AccountingNetwork, theme: &AccNetTheme) {
        // Build account name map from network metadata
        let account_names: HashMap<u16, String> = network
            .account_metadata
            .iter()
            .map(|(&idx, meta)| (idx, meta.name.clone()))
            .collect();

        // Show tabs for different ranking types
        ui.horizontal(|ui| {
            ui.label(RichText::new("By:").small().color(theme.text_secondary));
        });

        // PageRank ranking (most influential accounts)
        let pagerank_panel = TopAccountsPanel::by_pagerank(network, &account_names);
        pagerank_panel.show(ui, theme);

        ui.add_space(10.0);

        // Risk ranking
        let risk_panel = TopAccountsPanel::by_risk(network, &account_names);
        risk_panel.show(ui, theme);
    }

    /// Show amount distribution.
    fn show_amount_distribution(
        &self,
        ui: &mut Ui,
        network: &AccountingNetwork,
        theme: &AccNetTheme,
    ) {
        let dist = AmountDistribution::from_network(network);
        dist.show(ui, theme);
    }

    /// Show heatmaps.
    fn show_heatmaps(&self, ui: &mut Ui, network: &AccountingNetwork, theme: &AccNetTheme) {
        // Build account name map from network metadata
        let account_names: HashMap<u16, String> = network
            .account_metadata
            .iter()
            .map(|(&idx, meta)| (idx, meta.name.clone()))
            .collect();

        // Activity heatmap by account type
        let activity = ActivityHeatmap::from_network_by_type(network);
        activity.show(ui, theme);

        ui.add_space(10.0);

        // Correlation heatmap (top accounts)
        if network.accounts.len() >= 5 {
            let correlation = CorrelationHeatmap::from_network(network, 8, &account_names);
            correlation.show(ui, theme);

            ui.add_space(10.0);

            // Risk factor heatmap
            let risk = RiskHeatmap::from_network(network, 6, &account_names);
            risk.show(ui, theme);
        }
    }

    /// Show the live ticker.
    fn show_ticker(&self, ui: &mut Ui, network: &AccountingNetwork, theme: &AccNetTheme) {
        let ticker = LiveTicker::new()
            .add(
                "Accounts",
                network.accounts.len().to_string(),
                theme.asset_color,
            )
            .add("Flows", network.flows.len().to_string(), theme.flow_normal)
            .add(
                "Alerts",
                network.statistics.gaap_violation_count.to_string(),
                theme.alert_high,
            )
            .add(
                "Fraud",
                network.statistics.fraud_pattern_count.to_string(),
                theme.alert_critical,
            );

        ticker.show(ui, theme);
    }

    /// Show Benford's Law histogram.
    fn show_benford(&self, ui: &mut Ui, theme: &AccNetTheme) {
        let histogram = Histogram::benford(self.benford_counts);
        histogram.show(ui, theme);

        // Violation indicator
        let total: usize = self.benford_counts.iter().sum();
        if total >= 50 {
            let violations = self.calculate_benford_violations();
            let (text, color) = if violations > 0 {
                ("Distribution anomaly detected", theme.alert_high)
            } else {
                ("Distribution normal", theme.alert_low)
            };
            ui.horizontal(|ui| {
                ui.add_space(5.0);
                ui.label(RichText::new(text).small().color(color));
            });
        }
    }

    /// Show account type distribution.
    fn show_account_distribution(&self, ui: &mut Ui, theme: &AccNetTheme) {
        let chart = DonutChart::new("Account Types")
            .add(
                "Asset",
                self.account_type_counts[0] as f64,
                theme.asset_color,
            )
            .add(
                "Liability",
                self.account_type_counts[1] as f64,
                theme.liability_color,
            )
            .add(
                "Equity",
                self.account_type_counts[2] as f64,
                theme.equity_color,
            )
            .add(
                "Revenue",
                self.account_type_counts[3] as f64,
                theme.revenue_color,
            )
            .add(
                "Expense",
                self.account_type_counts[4] as f64,
                theme.expense_color,
            );

        chart.show(ui, theme);
    }

    /// Show method distribution.
    fn show_method_distribution(&self, ui: &mut Ui, theme: &AccNetTheme) {
        let dist = MethodDistribution::new(self.method_counts);
        dist.show(ui, theme);
    }

    /// Show fraud pattern breakdown.
    fn show_fraud_breakdown(&self, ui: &mut Ui, network: &AccountingNetwork, theme: &AccNetTheme) {
        let fraud_count = network.statistics.fraud_pattern_count;
        let gaap_count = network.statistics.gaap_violation_count;
        let suspense = network.statistics.suspense_account_count;

        let chart = BarChart::new("Detected Issues")
            .add("Suspense Accts", suspense as f64, theme.alert_medium)
            .add("GAAP Violations", gaap_count as f64, theme.alert_high)
            .add("Fraud Patterns", fraud_count as f64, theme.alert_critical)
            .add(
                "Benford Anomaly",
                self.calculate_benford_violations() as f64,
                Color32::from_rgb(200, 100, 200),
            );

        chart.show(ui, theme);
    }

    /// Show flow rate sparkline.
    fn show_flow_rate(&self, ui: &mut Ui, theme: &AccNetTheme) {
        let mut sparkline = Sparkline::new("Flow Rate (per tick)");
        sparkline.color = theme.flow_normal;
        for &val in &self.flow_rate_history {
            sparkline.push(val);
        }
        sparkline.show(ui, theme);
    }

    /// Show risk trend sparkline.
    fn show_risk_trend(&self, ui: &mut Ui, theme: &AccNetTheme) {
        let mut sparkline = Sparkline::new("Risk Score Trend");
        sparkline.color = theme.alert_high;
        for &val in &self.risk_history {
            sparkline.push(val * 100.0); // Convert to percentage
        }
        sparkline.show(ui, theme);
    }

    /// Reset the dashboard.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Get Benford digit counts.
    pub fn benford_counts(&self) -> [usize; 9] {
        self.benford_counts
    }

    /// Get account type counts.
    pub fn account_type_counts(&self) -> [usize; 5] {
        self.account_type_counts
    }

    /// Get account type balances (accumulated values).
    pub fn account_type_balances(&self) -> [f64; 5] {
        self.account_type_balances
    }

    /// Get method counts.
    pub fn method_counts(&self) -> [usize; 5] {
        self.method_counts
    }

    /// Get flow rate history.
    pub fn flow_rate_history(&self) -> &VecDeque<f32> {
        &self.flow_rate_history
    }

    /// Get risk history.
    pub fn risk_history(&self) -> &VecDeque<f32> {
        &self.risk_history
    }
}

impl Default for AnalyticsDashboard {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the first digit of a number.
fn get_first_digit(mut n: f64) -> u8 {
    n = n.abs();
    while n >= 10.0 {
        n /= 10.0;
    }
    while n < 1.0 && n > 0.0 {
        n *= 10.0;
    }
    n as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_digit() {
        assert_eq!(get_first_digit(123.45), 1);
        assert_eq!(get_first_digit(9876.0), 9);
        assert_eq!(get_first_digit(0.0045), 4);
        assert_eq!(get_first_digit(5.0), 5);
    }

    #[test]
    fn test_dashboard_creation() {
        let dashboard = AnalyticsDashboard::new();
        assert_eq!(dashboard.total_flows, 0);
    }
}
