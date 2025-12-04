//! Heatmap visualizations for account activity and correlations.
//!
//! Provides grid-based heatmaps that reveal patterns in:
//! - Account activity over time
//! - Account-to-account flow correlations
//! - Temporal patterns (hour of day, day of week)

use eframe::egui::{self, Color32, Pos2, Rect, Response, Sense, Stroke, Vec2};
use std::collections::HashMap;

use super::theme::AccNetTheme;
use crate::models::{AccountType, AccountingNetwork};

/// Color gradient for heatmaps (cold to hot).
pub struct HeatmapGradient {
    /// Gradient stops: (position 0-1, color).
    pub stops: Vec<(f32, Color32)>,
}

impl HeatmapGradient {
    /// Default cold-to-hot gradient (blue -> cyan -> green -> yellow -> red).
    pub fn thermal() -> Self {
        Self {
            stops: vec![
                (0.0, Color32::from_rgb(20, 30, 60)),    // Dark blue
                (0.2, Color32::from_rgb(40, 80, 160)),   // Blue
                (0.4, Color32::from_rgb(60, 180, 180)),  // Cyan
                (0.5, Color32::from_rgb(80, 200, 100)),  // Green
                (0.7, Color32::from_rgb(220, 200, 60)),  // Yellow
                (0.85, Color32::from_rgb(240, 120, 40)), // Orange
                (1.0, Color32::from_rgb(200, 40, 40)),   // Red
            ],
        }
    }

    /// Correlation gradient (negative blue, zero gray, positive red).
    pub fn correlation() -> Self {
        Self {
            stops: vec![
                (0.0, Color32::from_rgb(40, 80, 200)), // Strong negative (blue)
                (0.35, Color32::from_rgb(80, 120, 180)), // Weak negative
                (0.5, Color32::from_rgb(60, 60, 70)),  // Zero (gray)
                (0.65, Color32::from_rgb(180, 100, 80)), // Weak positive
                (1.0, Color32::from_rgb(200, 50, 50)), // Strong positive (red)
            ],
        }
    }

    /// Risk gradient (green safe -> yellow caution -> red danger).
    pub fn risk() -> Self {
        Self {
            stops: vec![
                (0.0, Color32::from_rgb(60, 160, 80)),  // Safe (green)
                (0.3, Color32::from_rgb(120, 180, 80)), // Low risk
                (0.5, Color32::from_rgb(200, 200, 60)), // Medium (yellow)
                (0.7, Color32::from_rgb(220, 140, 50)), // Elevated
                (0.85, Color32::from_rgb(200, 80, 50)), // High
                (1.0, Color32::from_rgb(180, 40, 40)),  // Critical (red)
            ],
        }
    }

    /// Get color for a value in [0, 1].
    pub fn sample(&self, t: f32) -> Color32 {
        let t = t.clamp(0.0, 1.0);

        // Find surrounding stops
        let mut prev = (0.0_f32, self.stops[0].1);
        for &(pos, color) in &self.stops {
            if t <= pos {
                // Interpolate between prev and current
                let range = pos - prev.0;
                if range < 0.001 {
                    return color;
                }
                let local_t = (t - prev.0) / range;
                return Self::lerp_color(prev.1, color, local_t);
            }
            prev = (pos, color);
        }
        self.stops.last().map(|s| s.1).unwrap_or(Color32::WHITE)
    }

    fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
        Color32::from_rgb(
            (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t) as u8,
            (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t) as u8,
            (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t) as u8,
        )
    }
}

/// Account activity heatmap - shows activity by account type and time.
pub struct ActivityHeatmap {
    /// Activity data: [account_type][time_bucket] = activity_level (0-1).
    pub data: Vec<Vec<f32>>,
    /// Row labels (account types or account names).
    pub row_labels: Vec<String>,
    /// Column labels (time periods).
    pub col_labels: Vec<String>,
    /// Title.
    pub title: String,
    /// Gradient for coloring.
    pub gradient: HeatmapGradient,
    /// Cell size.
    pub cell_size: f32,
}

impl ActivityHeatmap {
    /// Create from network data, grouping by account type.
    pub fn from_network_by_type(network: &AccountingNetwork) -> Self {
        let type_names = ["Asset", "Liability", "Equity", "Revenue", "Expense"];
        let mut data = vec![vec![0.0f32; 10]; 5]; // 5 types x 10 time buckets

        // Calculate activity by account type
        let mut max_activity = 1.0f32;
        for account in &network.accounts {
            let type_idx = match account.account_type {
                AccountType::Asset | AccountType::Contra => 0,
                AccountType::Liability => 1,
                AccountType::Equity => 2,
                AccountType::Revenue => 3,
                AccountType::Expense => 4,
            };

            // Distribute activity across time buckets based on transaction_count
            let activity = account.transaction_count as f32;
            let bucket = (account.index as usize) % 10; // Distribute by index for demo
            data[type_idx][bucket] += activity;
            max_activity = max_activity.max(data[type_idx][bucket]);
        }

        // Normalize to 0-1
        for row in &mut data {
            for cell in row {
                *cell /= max_activity.max(1.0);
            }
        }

        Self {
            data,
            row_labels: type_names.iter().map(|s| s.to_string()).collect(),
            col_labels: (1..=10).map(|i| format!("T{}", i)).collect(),
            title: "Account Type Activity".to_string(),
            gradient: HeatmapGradient::thermal(),
            cell_size: 18.0,
        }
    }

    /// Create from network showing individual account activity.
    pub fn from_network_top_accounts(network: &AccountingNetwork, top_n: usize) -> Self {
        // Get top N accounts by transaction count
        let mut accounts: Vec<_> = network.accounts.iter().enumerate().collect();
        accounts.sort_by(|a, b| b.1.transaction_count.cmp(&a.1.transaction_count));
        accounts.truncate(top_n);

        let mut data = vec![vec![0.0f32; 8]; accounts.len()];
        let mut row_labels = Vec::new();

        for (row_idx, (_, account)) in accounts.iter().enumerate() {
            // Use account code as label, or index if no metadata
            row_labels.push(format!("#{}", account.index));

            // Simulate activity distribution (in real app, use actual temporal data)
            let base_activity = account.transaction_count as f32;
            for (col, cell) in data[row_idx].iter_mut().enumerate().take(8) {
                // Create some variation based on account properties
                let variation = ((account.index as f32 + col as f32) * 0.7).sin() * 0.3 + 0.7;
                *cell = base_activity * variation;
            }
        }

        // Normalize
        let max_val = data
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .fold(0.0f32, f32::max)
            .max(1.0);

        for row in &mut data {
            for cell in row {
                *cell /= max_val;
            }
        }

        Self {
            data,
            row_labels,
            col_labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Avg"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            title: "Top Account Activity by Day".to_string(),
            gradient: HeatmapGradient::thermal(),
            cell_size: 16.0,
        }
    }

    /// Render the heatmap.
    pub fn show(&self, ui: &mut egui::Ui, theme: &AccNetTheme) -> Response {
        let rows = self.data.len();
        let cols = if rows > 0 { self.data[0].len() } else { 0 };

        let label_width = 60.0;
        let header_height = 20.0;
        let width = label_width + cols as f32 * self.cell_size + 10.0;
        let height = header_height + rows as f32 * self.cell_size + 25.0;

        let (response, painter) = ui.allocate_painter(
            Vec2::new(width.min(ui.available_width()), height),
            Sense::hover(),
        );
        let rect = response.rect;

        // Title
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top()),
            egui::Align2::LEFT_TOP,
            &self.title,
            egui::FontId::proportional(11.0),
            theme.text_secondary,
        );

        if rows == 0 || cols == 0 {
            return response;
        }

        let grid_left = rect.left() + label_width;
        let grid_top = rect.top() + header_height;

        // Column headers
        for (i, label) in self.col_labels.iter().enumerate() {
            let x = grid_left + (i as f32 + 0.5) * self.cell_size;
            painter.text(
                Pos2::new(x, grid_top - 3.0),
                egui::Align2::CENTER_BOTTOM,
                label,
                egui::FontId::proportional(7.0),
                theme.text_secondary,
            );
        }

        // Render grid
        for (row_idx, row_data) in self.data.iter().enumerate() {
            let y = grid_top + row_idx as f32 * self.cell_size;

            // Row label
            if row_idx < self.row_labels.len() {
                painter.text(
                    Pos2::new(grid_left - 3.0, y + self.cell_size / 2.0),
                    egui::Align2::RIGHT_CENTER,
                    &self.row_labels[row_idx],
                    egui::FontId::proportional(8.0),
                    theme.text_secondary,
                );
            }

            // Cells
            for (col_idx, &value) in row_data.iter().enumerate() {
                let x = grid_left + col_idx as f32 * self.cell_size;
                let cell_rect = Rect::from_min_size(
                    Pos2::new(x + 1.0, y + 1.0),
                    Vec2::new(self.cell_size - 2.0, self.cell_size - 2.0),
                );

                let color = self.gradient.sample(value);
                painter.rect_filled(cell_rect, 2.0, color);
            }
        }

        // Grid border
        let grid_rect = Rect::from_min_size(
            Pos2::new(grid_left, grid_top),
            Vec2::new(cols as f32 * self.cell_size, rows as f32 * self.cell_size),
        );
        painter.rect_stroke(
            grid_rect,
            0.0,
            Stroke::new(1.0, Color32::from_rgb(60, 60, 70)),
        );

        response
    }
}

/// Account correlation heatmap - shows flow relationships between accounts.
pub struct CorrelationHeatmap {
    /// Correlation matrix: [from_account][to_account] = correlation (-1 to 1).
    pub data: Vec<Vec<f32>>,
    /// Account labels.
    pub labels: Vec<String>,
    /// Title.
    pub title: String,
    /// Gradient for coloring.
    pub gradient: HeatmapGradient,
    /// Cell size.
    pub cell_size: f32,
}

impl CorrelationHeatmap {
    /// Create from network flow data.
    pub fn from_network(
        network: &AccountingNetwork,
        top_n: usize,
        account_names: &HashMap<u16, String>,
    ) -> Self {
        // Get top N accounts by degree
        let mut accounts: Vec<_> = network.accounts.iter().enumerate().collect();
        accounts.sort_by(|a, b| {
            let deg_a = a.1.in_degree + a.1.out_degree;
            let deg_b = b.1.in_degree + b.1.out_degree;
            deg_b.cmp(&deg_a)
        });
        accounts.truncate(top_n);

        let n = accounts.len();
        let mut data = vec![vec![0.0f32; n]; n];
        let mut labels = Vec::new();

        // Map original indices to matrix indices
        let index_map: HashMap<u16, usize> = accounts
            .iter()
            .enumerate()
            .map(|(i, (_, acc))| (acc.index, i))
            .collect();

        for (_, acc) in &accounts {
            let name = account_names
                .get(&acc.index)
                .cloned()
                .unwrap_or_else(|| format!("#{}", acc.index));
            // Truncate long names for display
            let short_name: String = name.chars().take(8).collect();
            labels.push(short_name);
        }

        // Build correlation from flows
        let mut flow_counts: HashMap<(u16, u16), usize> = HashMap::new();
        let mut max_flow = 1usize;

        for flow in &network.flows {
            if index_map.contains_key(&flow.source_account_index)
                && index_map.contains_key(&flow.target_account_index)
            {
                let key = (flow.source_account_index, flow.target_account_index);
                let count = flow_counts.entry(key).or_insert(0);
                *count += 1;
                max_flow = max_flow.max(*count);
            }
        }

        // Fill correlation matrix
        for ((from, to), count) in flow_counts {
            if let (Some(&i), Some(&j)) = (index_map.get(&from), index_map.get(&to)) {
                // Normalize to 0-1, then shift to -0.5 to 0.5 for visualization
                // (showing flow as positive correlation)
                let normalized = count as f32 / max_flow as f32;
                data[i][j] = normalized;
                // Symmetric for visualization
                data[j][i] = normalized * 0.8; // Slightly lower for reverse
            }
        }

        // Diagonal = 1.0 (self-correlation)
        for (i, row) in data.iter_mut().enumerate().take(n) {
            row[i] = 1.0;
        }

        Self {
            data,
            labels,
            title: "Account Flow Correlation".to_string(),
            gradient: HeatmapGradient::correlation(),
            cell_size: 14.0,
        }
    }

    /// Render the correlation heatmap.
    pub fn show(&self, ui: &mut egui::Ui, theme: &AccNetTheme) -> Response {
        let n = self.data.len();
        if n == 0 {
            let (response, _) = ui.allocate_painter(Vec2::new(100.0, 40.0), Sense::hover());
            return response;
        }

        let label_width = 35.0;
        let header_height = 35.0;
        let width = label_width + n as f32 * self.cell_size + 10.0;
        let height = header_height + n as f32 * self.cell_size + 25.0;

        let (response, painter) = ui.allocate_painter(
            Vec2::new(width.min(ui.available_width()), height),
            Sense::hover(),
        );
        let rect = response.rect;

        // Title
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top()),
            egui::Align2::LEFT_TOP,
            &self.title,
            egui::FontId::proportional(11.0),
            theme.text_secondary,
        );

        let grid_left = rect.left() + label_width;
        let grid_top = rect.top() + header_height;

        // Column headers (rotated labels)
        for (i, label) in self.labels.iter().enumerate() {
            let x = grid_left + (i as f32 + 0.5) * self.cell_size;
            painter.text(
                Pos2::new(x, grid_top - 3.0),
                egui::Align2::CENTER_BOTTOM,
                label,
                egui::FontId::proportional(7.0),
                theme.text_secondary,
            );
        }

        // Render grid
        for (i, row) in self.data.iter().enumerate() {
            let y = grid_top + i as f32 * self.cell_size;

            // Row label
            if i < self.labels.len() {
                painter.text(
                    Pos2::new(grid_left - 2.0, y + self.cell_size / 2.0),
                    egui::Align2::RIGHT_CENTER,
                    &self.labels[i],
                    egui::FontId::proportional(7.0),
                    theme.text_secondary,
                );
            }

            // Cells
            for (j, &value) in row.iter().enumerate() {
                let x = grid_left + j as f32 * self.cell_size;
                let cell_rect = Rect::from_min_size(
                    Pos2::new(x + 0.5, y + 0.5),
                    Vec2::new(self.cell_size - 1.0, self.cell_size - 1.0),
                );

                let color = self.gradient.sample(value);
                painter.rect_filled(cell_rect, 1.0, color);
            }
        }

        response
    }
}

/// Risk heatmap showing account risk levels by various factors.
pub struct RiskHeatmap {
    /// Risk data: [account][risk_factor] = risk_level (0-1).
    pub data: Vec<Vec<f32>>,
    /// Account labels.
    pub account_labels: Vec<String>,
    /// Risk factor labels.
    pub factor_labels: Vec<String>,
    /// Title.
    pub title: String,
    /// Gradient.
    pub gradient: HeatmapGradient,
    /// Cell size.
    pub cell_size: f32,
}

impl RiskHeatmap {
    /// Create from network analyzing multiple risk factors.
    pub fn from_network(
        network: &AccountingNetwork,
        top_n: usize,
        account_names: &HashMap<u16, String>,
    ) -> Self {
        let factors = ["Suspense", "Centrality", "Volume", "Balance", "Anomaly"];

        // Get accounts with highest total risk indicators
        let mut account_risks: Vec<(usize, f32, &crate::models::AccountNode)> = network
            .accounts
            .iter()
            .enumerate()
            .map(|(i, acc)| {
                let suspense_risk = if acc
                    .flags
                    .has(crate::models::AccountFlags::IS_SUSPENSE_ACCOUNT)
                {
                    0.8
                } else {
                    0.0
                };
                let degree_risk = (acc.in_degree + acc.out_degree) as f32 / 100.0;
                let total = suspense_risk + degree_risk;
                (i, total, acc)
            })
            .collect();

        account_risks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        account_risks.truncate(top_n);

        let mut data = Vec::new();
        let mut account_labels = Vec::new();

        let max_degree = network
            .accounts
            .iter()
            .map(|a| (a.in_degree + a.out_degree) as f32)
            .fold(1.0f32, f32::max);

        let max_volume = network
            .accounts
            .iter()
            .map(|a| a.transaction_count as f32)
            .fold(1.0f32, f32::max);

        let max_balance = network
            .accounts
            .iter()
            .map(|a| a.closing_balance.to_f64().abs() as f32)
            .fold(1.0f32, f32::max);

        for (_, _, acc) in account_risks {
            let name = account_names
                .get(&acc.index)
                .cloned()
                .unwrap_or_else(|| format!("#{}", acc.index));
            // Truncate long names for display
            let short_name: String = name.chars().take(10).collect();
            account_labels.push(short_name);

            let row = vec![
                // Suspense risk
                if acc
                    .flags
                    .has(crate::models::AccountFlags::IS_SUSPENSE_ACCOUNT)
                {
                    0.9
                } else {
                    0.1
                },
                // Centrality risk (high degree = higher risk)
                ((acc.in_degree + acc.out_degree) as f32 / max_degree).min(1.0),
                // Volume risk (high transaction count = watch closely)
                (acc.transaction_count as f32 / max_volume).min(1.0),
                // Balance concentration risk
                (acc.closing_balance.to_f64().abs() as f32 / max_balance).min(1.0),
                // Anomaly flag risk
                if acc.flags.has(crate::models::AccountFlags::HAS_ANOMALY) {
                    0.85
                } else {
                    0.15
                },
            ];
            data.push(row);
        }

        Self {
            data,
            account_labels,
            factor_labels: factors.iter().map(|s| s.to_string()).collect(),
            title: "Account Risk Factors".to_string(),
            gradient: HeatmapGradient::risk(),
            cell_size: 20.0,
        }
    }

    /// Render the risk heatmap.
    pub fn show(&self, ui: &mut egui::Ui, theme: &AccNetTheme) -> Response {
        let rows = self.data.len();
        let cols = self.factor_labels.len();

        if rows == 0 || cols == 0 {
            let (response, _) = ui.allocate_painter(Vec2::new(100.0, 40.0), Sense::hover());
            return response;
        }

        let label_width = 40.0;
        let header_height = 22.0;
        let width = label_width + cols as f32 * self.cell_size + 10.0;
        let height = header_height + rows as f32 * self.cell_size + 25.0;

        let (response, painter) = ui.allocate_painter(
            Vec2::new(width.min(ui.available_width()), height),
            Sense::hover(),
        );
        let rect = response.rect;

        // Title
        painter.text(
            Pos2::new(rect.left() + 5.0, rect.top()),
            egui::Align2::LEFT_TOP,
            &self.title,
            egui::FontId::proportional(11.0),
            theme.text_secondary,
        );

        let grid_left = rect.left() + label_width;
        let grid_top = rect.top() + header_height;

        // Column headers
        for (i, label) in self.factor_labels.iter().enumerate() {
            let x = grid_left + (i as f32 + 0.5) * self.cell_size;
            painter.text(
                Pos2::new(x, grid_top - 2.0),
                egui::Align2::CENTER_BOTTOM,
                &label[..label.len().min(4)], // Truncate to 4 chars
                egui::FontId::proportional(7.0),
                theme.text_secondary,
            );
        }

        // Render cells
        for (i, row) in self.data.iter().enumerate() {
            let y = grid_top + i as f32 * self.cell_size;

            // Row label
            if i < self.account_labels.len() {
                painter.text(
                    Pos2::new(grid_left - 2.0, y + self.cell_size / 2.0),
                    egui::Align2::RIGHT_CENTER,
                    &self.account_labels[i],
                    egui::FontId::proportional(7.0),
                    theme.text_secondary,
                );
            }

            for (j, &value) in row.iter().enumerate() {
                let x = grid_left + j as f32 * self.cell_size;
                let cell_rect = Rect::from_min_size(
                    Pos2::new(x + 1.0, y + 1.0),
                    Vec2::new(self.cell_size - 2.0, self.cell_size - 2.0),
                );

                let color = self.gradient.sample(value);
                painter.rect_filled(cell_rect, 2.0, color);
            }
        }

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_thermal() {
        let g = HeatmapGradient::thermal();
        let c0 = g.sample(0.0);
        let c1 = g.sample(1.0);
        assert_ne!(c0, c1);
    }

    #[test]
    fn test_gradient_bounds() {
        let g = HeatmapGradient::thermal();
        // Should not panic on out-of-bounds
        let _ = g.sample(-0.5);
        let _ = g.sample(1.5);
    }
}
