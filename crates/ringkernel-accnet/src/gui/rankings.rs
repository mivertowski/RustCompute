//! Account ranking panels for analytics dashboard.
//!
//! Shows top accounts by various metrics:
//! - PageRank (influence in the network)
//! - Centrality (degree/betweenness)
//! - Risk score (composite anomaly indicator)

use eframe::egui::{self, Color32, Pos2, Rect, Response, Sense, Vec2};
use std::collections::HashMap;

use super::theme::AccNetTheme;
use crate::models::{AccountFlags, AccountType, AccountingNetwork};

/// A ranked account entry.
#[derive(Clone)]
pub struct RankedAccount {
    /// Account index.
    pub index: u16,
    /// Display name or code.
    pub name: String,
    /// Account type.
    pub account_type: AccountType,
    /// Metric value.
    pub value: f64,
    /// Risk level (0-1).
    pub risk: f32,
}

/// Top accounts ranking panel.
pub struct TopAccountsPanel {
    /// Title of the panel.
    pub title: String,
    /// Ranked accounts.
    pub accounts: Vec<RankedAccount>,
    /// Maximum accounts to show.
    pub max_display: usize,
    /// Bar color based on account type.
    pub use_type_colors: bool,
}

impl TopAccountsPanel {
    /// Create empty panel.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            accounts: Vec::new(),
            max_display: 10,
            use_type_colors: true,
        }
    }

    /// Create top accounts by PageRank.
    pub fn by_pagerank(network: &AccountingNetwork, metadata: &HashMap<u16, String>) -> Self {
        let pagerank = network.compute_pagerank(20, 0.85);

        let mut accounts: Vec<RankedAccount> = pagerank
            .iter()
            .enumerate()
            .filter_map(|(idx, &rank)| {
                if idx < network.accounts.len() {
                    let acc = &network.accounts[idx];
                    let name = metadata
                        .get(&acc.index)
                        .cloned()
                        .unwrap_or_else(|| format!("#{}", acc.index));
                    let risk = Self::calculate_account_risk(acc);
                    Some(RankedAccount {
                        index: acc.index,
                        name,
                        account_type: acc.account_type,
                        value: rank,
                        risk,
                    })
                } else {
                    None
                }
            })
            .collect();

        accounts.sort_by(|a, b| {
            b.value
                .partial_cmp(&a.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        accounts.truncate(10);

        Self {
            title: "Top Accounts by PageRank".to_string(),
            accounts,
            max_display: 10,
            use_type_colors: true,
        }
    }

    /// Create top accounts by degree centrality.
    pub fn by_centrality(network: &AccountingNetwork, metadata: &HashMap<u16, String>) -> Self {
        let max_possible = (network.accounts.len().saturating_sub(1) * 2) as f64;

        let mut accounts: Vec<RankedAccount> = network
            .accounts
            .iter()
            .map(|acc| {
                let degree = (acc.in_degree + acc.out_degree) as f64;
                let centrality = if max_possible > 0.0 {
                    degree / max_possible
                } else {
                    0.0
                };
                let name = metadata
                    .get(&acc.index)
                    .cloned()
                    .unwrap_or_else(|| format!("#{}", acc.index));
                let risk = Self::calculate_account_risk(acc);
                RankedAccount {
                    index: acc.index,
                    name,
                    account_type: acc.account_type,
                    value: centrality,
                    risk,
                }
            })
            .collect();

        accounts.sort_by(|a, b| {
            b.value
                .partial_cmp(&a.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        accounts.truncate(10);

        Self {
            title: "Top Accounts by Centrality".to_string(),
            accounts,
            max_display: 10,
            use_type_colors: true,
        }
    }

    /// Create top accounts by risk score.
    pub fn by_risk(network: &AccountingNetwork, metadata: &HashMap<u16, String>) -> Self {
        let mut accounts: Vec<RankedAccount> = network
            .accounts
            .iter()
            .map(|acc| {
                let risk = Self::calculate_account_risk(acc);
                let name = metadata
                    .get(&acc.index)
                    .cloned()
                    .unwrap_or_else(|| format!("#{}", acc.index));
                RankedAccount {
                    index: acc.index,
                    name,
                    account_type: acc.account_type,
                    value: risk as f64,
                    risk,
                }
            })
            .collect();

        accounts.sort_by(|a, b| {
            b.value
                .partial_cmp(&a.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        accounts.truncate(10);

        Self {
            title: "Top Accounts by Risk".to_string(),
            accounts,
            max_display: 10,
            use_type_colors: false, // Use risk colors instead
        }
    }

    /// Create top accounts by transaction volume.
    pub fn by_volume(network: &AccountingNetwork, metadata: &HashMap<u16, String>) -> Self {
        let max_volume = network
            .accounts
            .iter()
            .map(|a| a.transaction_count as f64)
            .fold(1.0, f64::max);

        let mut accounts: Vec<RankedAccount> = network
            .accounts
            .iter()
            .map(|acc| {
                let volume = acc.transaction_count as f64 / max_volume;
                let name = metadata
                    .get(&acc.index)
                    .cloned()
                    .unwrap_or_else(|| format!("#{}", acc.index));
                let risk = Self::calculate_account_risk(acc);
                RankedAccount {
                    index: acc.index,
                    name,
                    account_type: acc.account_type,
                    value: volume,
                    risk,
                }
            })
            .collect();

        accounts.sort_by(|a, b| {
            b.value
                .partial_cmp(&a.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        accounts.truncate(10);

        Self {
            title: "Top Accounts by Volume".to_string(),
            accounts,
            max_display: 10,
            use_type_colors: true,
        }
    }

    /// Calculate composite risk score for an account.
    fn calculate_account_risk(acc: &crate::models::AccountNode) -> f32 {
        let mut risk = 0.0f32;

        // Suspense account flag = high base risk
        if acc.flags.has(AccountFlags::IS_SUSPENSE_ACCOUNT) {
            risk += 0.4;
        }

        // GAAP violation flag
        if acc.flags.has(AccountFlags::HAS_GAAP_VIOLATION) {
            risk += 0.25;
        }

        // Fraud pattern flag
        if acc.flags.has(AccountFlags::HAS_FRAUD_PATTERN) {
            risk += 0.3;
        }

        // Anomaly flag
        if acc.flags.has(AccountFlags::HAS_ANOMALY) {
            risk += 0.2;
        }

        // High degree concentration
        let degree = (acc.in_degree + acc.out_degree) as f32;
        if degree > 50.0 {
            risk += 0.1;
        }

        risk.min(1.0)
    }

    /// Render the ranking panel.
    pub fn show(&self, ui: &mut egui::Ui, theme: &AccNetTheme) -> Response {
        let width = ui.available_width();
        let row_height = 18.0;
        let header_height = 20.0;
        let total_height =
            header_height + self.accounts.len().min(self.max_display) as f32 * row_height + 10.0;

        let (response, painter) =
            ui.allocate_painter(Vec2::new(width, total_height), Sense::hover());
        let rect = response.rect;

        // Title
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top()),
            egui::Align2::LEFT_TOP,
            &self.title,
            egui::FontId::proportional(11.0),
            theme.text_secondary,
        );

        if self.accounts.is_empty() {
            painter.text(
                Pos2::new(rect.center().x, rect.top() + 40.0),
                egui::Align2::CENTER_CENTER,
                "No data",
                egui::FontId::proportional(10.0),
                theme.text_secondary,
            );
            return response;
        }

        let max_value = self.accounts.iter().map(|a| a.value).fold(0.001, f64::max);

        let rank_width = 20.0;
        let name_width = 70.0;
        let bar_area_start = rect.left() + rank_width + name_width;
        let bar_area_width = width - rank_width - name_width - 50.0;

        for (i, account) in self.accounts.iter().take(self.max_display).enumerate() {
            let y = rect.top() + header_height + i as f32 * row_height;

            // Rank number
            painter.text(
                Pos2::new(rect.left() + 5.0, y + row_height / 2.0),
                egui::Align2::LEFT_CENTER,
                format!("{}.", i + 1),
                egui::FontId::proportional(9.0),
                theme.text_secondary,
            );

            // Account name
            painter.text(
                Pos2::new(rect.left() + rank_width, y + row_height / 2.0),
                egui::Align2::LEFT_CENTER,
                &account.name[..account.name.len().min(10)],
                egui::FontId::proportional(9.0),
                theme.text_primary,
            );

            // Bar
            let bar_width = ((account.value / max_value) as f32 * bar_area_width).max(3.0);
            let bar_color = if self.use_type_colors {
                Self::type_color(account.account_type, theme)
            } else {
                Self::risk_color(account.risk)
            };

            let bar_rect = Rect::from_min_size(
                Pos2::new(bar_area_start, y + 2.0),
                Vec2::new(bar_width, row_height - 4.0),
            );
            painter.rect_filled(bar_rect, 2.0, bar_color);

            // Value
            painter.text(
                Pos2::new(bar_area_start + bar_area_width + 5.0, y + row_height / 2.0),
                egui::Align2::LEFT_CENTER,
                format!("{:.3}", account.value),
                egui::FontId::proportional(8.0),
                theme.text_secondary,
            );

            // Risk indicator dot
            let risk_x = rect.right() - 10.0;
            let risk_color = Self::risk_color(account.risk);
            painter.circle_filled(Pos2::new(risk_x, y + row_height / 2.0), 4.0, risk_color);
        }

        response
    }

    fn type_color(account_type: AccountType, theme: &AccNetTheme) -> Color32 {
        match account_type {
            AccountType::Asset => theme.asset_color,
            AccountType::Liability => theme.liability_color,
            AccountType::Equity => theme.equity_color,
            AccountType::Revenue => theme.revenue_color,
            AccountType::Expense => theme.expense_color,
            AccountType::Contra => Color32::from_rgb(150, 150, 150),
        }
    }

    fn risk_color(risk: f32) -> Color32 {
        if risk < 0.25 {
            Color32::from_rgb(80, 180, 100) // Green - low risk
        } else if risk < 0.5 {
            Color32::from_rgb(180, 180, 80) // Yellow - medium
        } else if risk < 0.75 {
            Color32::from_rgb(220, 140, 60) // Orange - elevated
        } else {
            Color32::from_rgb(200, 60, 60) // Red - high risk
        }
    }
}

/// Pattern statistics panel showing detected anomalies.
pub struct PatternStatsPanel {
    /// Circular flows detected.
    pub circular_flows: usize,
    /// Velocity anomalies (rapid multi-hop).
    pub velocity_anomalies: usize,
    /// Timing anomalies (off-hours).
    pub timing_anomalies: usize,
    /// Amount clustering (structuring).
    pub amount_clustering: usize,
    /// Dormant account reactivations.
    pub dormant_reactivations: usize,
    /// Round amount anomalies.
    pub round_amounts: usize,
}

impl PatternStatsPanel {
    /// Create from network analysis.
    pub fn from_network(network: &AccountingNetwork) -> Self {
        // Analyze flows for patterns
        let mut circular_flows = 0;
        let mut velocity_anomalies = 0;
        let mut amount_clustering = 0;
        let mut dormant_reactivations = 0;
        let mut round_amounts = 0;

        // Check for circular flow flags
        for flow in &network.flows {
            if flow.flags.has(crate::models::FlowFlags::IS_CIRCULAR) {
                circular_flows += 1;
            }
            if flow.flags.has(crate::models::FlowFlags::IS_ANOMALOUS) {
                velocity_anomalies += 1;
            }

            // Check for round amounts (potential structuring)
            let amount = flow.amount.to_f64().abs();
            if amount > 100.0 && amount % 1000.0 == 0.0 {
                round_amounts += 1;
            }

            // Check for amount clustering around thresholds
            let threshold_distances = [
                (amount - 9999.0).abs(),
                (amount - 10000.0).abs(),
                (amount - 4999.0).abs(),
                (amount - 5000.0).abs(),
            ];
            if threshold_distances.iter().any(|&d| d < 100.0) {
                amount_clustering += 1;
            }
        }

        // Check for dormant reactivations
        for acc in &network.accounts {
            if acc.flags.has(AccountFlags::IS_DORMANT) && acc.transaction_count > 0 {
                dormant_reactivations += 1;
            }
        }

        // Estimate timing anomalies from fraud pattern count
        let timing_anomalies = network.statistics.fraud_pattern_count / 4;

        Self {
            circular_flows,
            velocity_anomalies,
            timing_anomalies,
            amount_clustering,
            dormant_reactivations,
            round_amounts,
        }
    }

    /// Render the pattern stats panel.
    pub fn show(&self, ui: &mut egui::Ui, theme: &AccNetTheme) -> Response {
        let width = ui.available_width();
        let patterns = [
            (
                "Circular Flows",
                self.circular_flows,
                "⟲",
                Color32::from_rgb(200, 80, 80),
            ),
            (
                "Velocity Anomalies",
                self.velocity_anomalies,
                "⚡",
                Color32::from_rgb(220, 160, 60),
            ),
            (
                "Timing Anomalies",
                self.timing_anomalies,
                "⏰",
                Color32::from_rgb(180, 100, 180),
            ),
            (
                "Amount Clustering",
                self.amount_clustering,
                "▣",
                Color32::from_rgb(100, 160, 200),
            ),
            (
                "Dormant Reactivated",
                self.dormant_reactivations,
                "💤",
                Color32::from_rgb(140, 140, 180),
            ),
            (
                "Round Amounts",
                self.round_amounts,
                "○",
                Color32::from_rgb(160, 200, 160),
            ),
        ];

        let row_height = 18.0;
        let total_height = 20.0 + patterns.len() as f32 * row_height + 5.0;

        let (response, painter) =
            ui.allocate_painter(Vec2::new(width, total_height), Sense::hover());
        let rect = response.rect;

        // Title
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top()),
            egui::Align2::LEFT_TOP,
            "Detected Patterns",
            egui::FontId::proportional(11.0),
            theme.text_secondary,
        );

        let _total: usize = patterns.iter().map(|(_, c, _, _)| c).sum();
        let max_count = patterns
            .iter()
            .map(|(_, c, _, _)| *c)
            .max()
            .unwrap_or(1)
            .max(1);

        for (i, (name, count, icon, color)) in patterns.iter().enumerate() {
            let y = rect.top() + 18.0 + i as f32 * row_height;

            // Icon
            painter.text(
                Pos2::new(rect.left() + 10.0, y + row_height / 2.0),
                egui::Align2::LEFT_CENTER,
                *icon,
                egui::FontId::proportional(10.0),
                *color,
            );

            // Name
            painter.text(
                Pos2::new(rect.left() + 25.0, y + row_height / 2.0),
                egui::Align2::LEFT_CENTER,
                *name,
                egui::FontId::proportional(9.0),
                theme.text_primary,
            );

            // Mini bar
            let bar_start = rect.left() + 120.0;
            let bar_max_width = width - 160.0;
            let bar_width = (*count as f32 / max_count as f32 * bar_max_width).max(2.0);

            let bar_rect = Rect::from_min_size(
                Pos2::new(bar_start, y + 4.0),
                Vec2::new(bar_width, row_height - 8.0),
            );
            painter.rect_filled(bar_rect, 2.0, *color);

            // Count
            painter.text(
                Pos2::new(rect.right() - 25.0, y + row_height / 2.0),
                egui::Align2::RIGHT_CENTER,
                count.to_string(),
                egui::FontId::proportional(9.0),
                if *count > 0 {
                    *color
                } else {
                    theme.text_secondary
                },
            );
        }

        response
    }
}

/// Amount distribution histogram showing transaction amounts.
pub struct AmountDistribution {
    /// Histogram buckets (count per range).
    pub buckets: Vec<usize>,
    /// Bucket labels.
    pub labels: Vec<String>,
    /// Title.
    pub title: String,
}

impl AmountDistribution {
    /// Create from network flows.
    pub fn from_network(network: &AccountingNetwork) -> Self {
        // Define amount ranges (log scale for Benford compliance check)
        let ranges = [
            (0.0, 100.0, "<100"),
            (100.0, 500.0, "100-500"),
            (500.0, 1000.0, "500-1K"),
            (1000.0, 5000.0, "1K-5K"),
            (5000.0, 10000.0, "5K-10K"),
            (10000.0, 50000.0, "10K-50K"),
            (50000.0, 100000.0, "50K-100K"),
            (100000.0, f64::MAX, ">100K"),
        ];

        let mut buckets = vec![0usize; ranges.len()];

        for flow in &network.flows {
            let amount = flow.amount.to_f64().abs();
            for (i, &(min, max, _)) in ranges.iter().enumerate() {
                if amount >= min && amount < max {
                    buckets[i] += 1;
                    break;
                }
            }
        }

        Self {
            buckets,
            labels: ranges.iter().map(|(_, _, l)| l.to_string()).collect(),
            title: "Transaction Amount Distribution".to_string(),
        }
    }

    /// Render the distribution.
    pub fn show(&self, ui: &mut egui::Ui, theme: &AccNetTheme) -> Response {
        let width = ui.available_width();
        let height = 90.0;

        let (response, painter) = ui.allocate_painter(Vec2::new(width, height), Sense::hover());
        let rect = response.rect;

        // Title
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top()),
            egui::Align2::LEFT_TOP,
            &self.title,
            egui::FontId::proportional(11.0),
            theme.text_secondary,
        );

        if self.buckets.is_empty() {
            return response;
        }

        let chart_left = rect.left() + 5.0;
        let chart_top = rect.top() + 18.0;
        let chart_width = width - 10.0;
        let chart_height = height - 35.0;

        let max_count = *self.buckets.iter().max().unwrap_or(&1).max(&1);
        let bar_width = chart_width / self.buckets.len() as f32;
        let gap = bar_width * 0.1;

        // Background
        painter.rect_filled(
            Rect::from_min_size(
                Pos2::new(chart_left, chart_top),
                Vec2::new(chart_width, chart_height),
            ),
            2.0,
            Color32::from_rgb(25, 25, 35),
        );

        // Bars
        for (i, &count) in self.buckets.iter().enumerate() {
            let x = chart_left + i as f32 * bar_width;
            let bar_height = (count as f32 / max_count as f32) * (chart_height - 5.0);

            let bar_rect = Rect::from_min_max(
                Pos2::new(x + gap, chart_top + chart_height - bar_height),
                Pos2::new(x + bar_width - gap, chart_top + chart_height),
            );

            // Gradient color based on position (small to large amounts)
            let t = i as f32 / (self.buckets.len() - 1) as f32;
            let color = Color32::from_rgb(
                (80.0 + 120.0 * t) as u8,
                (180.0 - 60.0 * t) as u8,
                (200.0 - 100.0 * t) as u8,
            );

            painter.rect_filled(bar_rect, 2.0, color);

            // Label
            if i < self.labels.len() {
                painter.text(
                    Pos2::new(x + bar_width / 2.0, chart_top + chart_height + 3.0),
                    egui::Align2::CENTER_TOP,
                    &self.labels[i],
                    egui::FontId::proportional(7.0),
                    theme.text_secondary,
                );
            }
        }

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_risk_color() {
        let low = TopAccountsPanel::risk_color(0.1);
        let high = TopAccountsPanel::risk_color(0.9);
        assert_ne!(low, high);
    }

    #[test]
    fn test_pattern_stats() {
        let network = AccountingNetwork::new(Uuid::new_v4(), 2024, 1);
        let stats = PatternStatsPanel::from_network(&network);
        assert_eq!(stats.circular_flows, 0);
    }
}
