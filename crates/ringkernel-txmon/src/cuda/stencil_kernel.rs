//! Stencil-based pattern detection for transaction network analysis.
//!
//! This module uses stencil computation patterns to detect complex
//! transaction patterns like circular trading, layering, and velocity
//! anomalies across customer networks.
//!
//! ## Concept
//!
//! Transaction networks can be represented as 2D grids where:
//! - **Velocity Grid**: X = customer_id % width, Y = time_bucket
//! - **Network Grid**: X = source_customer % width, Y = dest_customer % height
//!
//! Stencil operations then detect patterns by examining neighbor relationships:
//!
//! ```text
//! Velocity Anomaly Detection (5-point stencil):
//!
//!              [t-1]
//!               |
//!    [c-1] -- [c,t] -- [c+1]
//!               |
//!              [t+1]
//!
//! - Temporal diff: [c,t] - [c,t-1] (sudden spike?)
//! - Spatial diff: [c,t] vs avg([c-1,t], [c+1,t]) (outlier among neighbors?)
//! ```
//!
//! ```text
//! Circular Trading Detection (network grid):
//!
//!    A -> B (row A, col B)
//!    B -> C (row B, col C)
//!    C -> A (row C, col A) <- forms cycle!
//!
//! High symmetry + coordinated timing = suspicious
//! ```

use super::types::*;

/// Configuration for stencil-based pattern detection.
#[derive(Debug, Clone)]
pub struct StencilPatternConfig {
    /// Grid width (number of customer buckets).
    pub grid_width: usize,
    /// Grid height (number of time buckets or customer buckets).
    pub grid_height: usize,
    /// Tile size for GPU processing.
    pub tile_size: (usize, usize),
    /// Halo size (stencil radius).
    pub halo: usize,
    /// Velocity anomaly threshold.
    pub velocity_threshold: f32,
    /// Circular pattern threshold.
    pub circular_threshold: f32,
    /// Time bucket duration in milliseconds.
    pub time_bucket_ms: u64,
}

impl Default for StencilPatternConfig {
    fn default() -> Self {
        Self {
            grid_width: 256,
            grid_height: 256,
            tile_size: (16, 16),
            halo: 1,
            velocity_threshold: 2.0,
            circular_threshold: 0.7,
            time_bucket_ms: 60_000, // 1 minute
        }
    }
}

/// Velocity grid for temporal pattern analysis.
///
/// Each cell contains transaction count for a (customer_bucket, time_bucket) pair.
#[derive(Clone)]
pub struct VelocityGrid {
    pub data: Vec<u32>,
    pub width: usize,
    pub height: usize,
    pub base_timestamp: u64,
    pub bucket_ms: u64,
}

impl VelocityGrid {
    /// Create a new velocity grid.
    pub fn new(width: usize, height: usize, bucket_ms: u64) -> Self {
        Self {
            data: vec![0; width * height],
            width,
            height,
            base_timestamp: 0,
            bucket_ms,
        }
    }

    /// Reset the grid.
    pub fn reset(&mut self, base_timestamp: u64) {
        self.data.fill(0);
        self.base_timestamp = base_timestamp;
    }

    /// Add a transaction to the grid.
    pub fn add_transaction(&mut self, customer_id: u64, timestamp: u64) {
        let x = (customer_id as usize) % self.width;
        let y = ((timestamp - self.base_timestamp) / self.bucket_ms) as usize;

        if y < self.height {
            let idx = y * self.width + x;
            self.data[idx] = self.data[idx].saturating_add(1);
        }
    }

    /// Get transaction count at position.
    pub fn get(&self, x: usize, y: usize) -> u32 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0
        }
    }

    /// Get grid index for position.
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
}

/// Network adjacency grid for circular pattern detection.
///
/// Cell (x, y) represents transactions from customer_bucket x to customer_bucket y.
#[derive(Clone)]
pub struct NetworkGrid {
    pub cells: Vec<NetworkCell>,
    pub width: usize,
    pub height: usize,
}

impl NetworkGrid {
    /// Create a new network grid.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            cells: vec![NetworkCell::default(); width * height],
            width,
            height,
        }
    }

    /// Reset the grid.
    pub fn reset(&mut self) {
        for cell in &mut self.cells {
            *cell = NetworkCell::default();
        }
    }

    /// Add a transaction to the network.
    pub fn add_transaction(
        &mut self,
        from_customer: u64,
        to_customer: u64,
        amount: u64,
        is_inbound: bool,
    ) {
        let x = (from_customer as usize) % self.width;
        let y = (to_customer as usize) % self.height;
        let idx = y * self.width + x;

        let cell = &mut self.cells[idx];
        cell.total_amount = cell.total_amount.saturating_add(amount);
        cell.tx_count = cell.tx_count.saturating_add(1);

        if is_inbound {
            cell.inbound_count = cell.inbound_count.saturating_add(1);
        } else {
            cell.outbound_count = cell.outbound_count.saturating_add(1);
        }
    }

    /// Get cell at position.
    pub fn get(&self, x: usize, y: usize) -> &NetworkCell {
        &self.cells[y * self.width + x]
    }

    /// Get mutable cell at position.
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut NetworkCell {
        &mut self.cells[y * self.width + x]
    }
}

/// Anomaly score result from stencil computation.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnomalyScore {
    pub customer_bucket: usize,
    pub time_bucket: usize,
    pub velocity_score: f32,
    pub pattern_score: f32,
    pub combined_score: f32,
}

/// CPU implementation of velocity anomaly detection stencil.
///
/// This mirrors the GPU stencil kernel logic for testing and fallback.
pub fn detect_velocity_anomalies_cpu(
    current: &VelocityGrid,
    previous: &VelocityGrid,
    threshold: f32,
) -> Vec<AnomalyScore> {
    let mut anomalies = Vec::new();

    for y in 1..current.height - 1 {
        for x in 1..current.width - 1 {
            let curr = current.get(x, y) as f32;
            let prev = previous.get(x, y) as f32;

            // Temporal neighbors
            let north = current.get(x, y - 1) as f32;
            let south = current.get(x, y + 1) as f32;

            // Spatial neighbors
            let west = current.get(x - 1, y) as f32;
            let east = current.get(x + 1, y) as f32;

            // Calculate temporal acceleration
            let temporal_diff = curr - prev;

            // Calculate spatial anomaly
            let neighbor_avg = (north + south + west + east) / 4.0;
            let spatial_diff = curr - neighbor_avg;

            // Combined score
            let score = (temporal_diff * 0.6) + (spatial_diff * 0.4);

            if score > threshold {
                anomalies.push(AnomalyScore {
                    customer_bucket: x,
                    time_bucket: y,
                    velocity_score: score,
                    pattern_score: 0.0,
                    combined_score: score,
                });
            }
        }
    }

    anomalies
}

/// CPU implementation of circular pattern detection stencil.
pub fn detect_circular_patterns_cpu(
    current: &NetworkGrid,
    previous: &NetworkGrid,
    threshold: f32,
) -> Vec<AnomalyScore> {
    let mut patterns = Vec::new();

    for y in 1..current.height - 1 {
        for x in 1..current.width - 1 {
            let cell = current.get(x, y);

            if cell.tx_count == 0 {
                continue;
            }

            let inbound = cell.inbound_count as f32;
            let outbound = cell.outbound_count as f32;
            let total = inbound + outbound;

            // Symmetry score
            let symmetry = if total > 0.0 {
                1.0 - ((inbound - outbound).abs() / total)
            } else {
                0.0
            };

            // Concentration score
            let concentration = if cell.tx_count > 0 {
                1.0 - (cell.unique_counterparties as f32 / cell.tx_count as f32).min(1.0)
            } else {
                0.0
            };

            // Velocity score
            let prev_cell = previous.get(x, y);
            let velocity = (cell.tx_count as f32 - prev_cell.tx_count as f32).max(0.0);

            // Neighbor coordination
            let north = current.get(x, y - 1);
            let south = current.get(x, y + 1);
            let avg_neighbor = (north.total_amount + south.total_amount) as f32 / 2.0;

            let coordination = if cell.total_amount > 0 {
                (1.0 - (cell.total_amount as f32 - avg_neighbor).abs() / cell.total_amount as f32)
                    .max(0.0)
            } else {
                0.0
            };

            // Combined pattern score
            let score =
                (symmetry * 0.3) + (concentration * 0.3) + (velocity * 0.2) + (coordination * 0.2);

            if score > threshold {
                patterns.push(AnomalyScore {
                    customer_bucket: x,
                    time_bucket: y,
                    velocity_score: velocity,
                    pattern_score: score,
                    combined_score: score,
                });
            }
        }
    }

    patterns
}

/// Stencil pattern detection backend.
pub struct StencilPatternBackend {
    config: StencilPatternConfig,
    velocity_grid_current: VelocityGrid,
    velocity_grid_previous: VelocityGrid,
    network_grid_current: NetworkGrid,
    network_grid_previous: NetworkGrid,
    anomaly_scores: Vec<f32>,
}

impl StencilPatternBackend {
    /// Create a new stencil pattern backend.
    pub fn new(config: StencilPatternConfig) -> Self {
        let velocity_current =
            VelocityGrid::new(config.grid_width, config.grid_height, config.time_bucket_ms);
        let velocity_previous =
            VelocityGrid::new(config.grid_width, config.grid_height, config.time_bucket_ms);
        let network_current = NetworkGrid::new(config.grid_width, config.grid_height);
        let network_previous = NetworkGrid::new(config.grid_width, config.grid_height);
        let scores_size = config.grid_width * config.grid_height;

        Self {
            config,
            velocity_grid_current: velocity_current,
            velocity_grid_previous: velocity_previous,
            network_grid_current: network_current,
            network_grid_previous: network_previous,
            anomaly_scores: vec![0.0; scores_size],
        }
    }

    /// Add transactions to the grids.
    pub fn add_transactions(&mut self, transactions: &[GpuTransaction]) {
        for tx in transactions {
            // Add to velocity grid
            self.velocity_grid_current
                .add_transaction(tx.customer_id, tx.timestamp);

            // Add to network grid (if there's a destination)
            if tx.destination_id != 0 && tx.destination_id != tx.customer_id {
                self.network_grid_current.add_transaction(
                    tx.customer_id,
                    tx.destination_id,
                    tx.amount_cents,
                    false,
                );
            }
        }
    }

    /// Advance to next time window.
    pub fn advance_window(&mut self, new_base_timestamp: u64) {
        // Swap current to previous
        std::mem::swap(
            &mut self.velocity_grid_current,
            &mut self.velocity_grid_previous,
        );
        std::mem::swap(
            &mut self.network_grid_current,
            &mut self.network_grid_previous,
        );

        // Reset current grids
        self.velocity_grid_current.reset(new_base_timestamp);
        self.network_grid_current.reset();
    }

    /// Detect velocity anomalies (CPU fallback).
    pub fn detect_velocity_anomalies(&self) -> Vec<AnomalyScore> {
        detect_velocity_anomalies_cpu(
            &self.velocity_grid_current,
            &self.velocity_grid_previous,
            self.config.velocity_threshold,
        )
    }

    /// Detect circular patterns (CPU fallback).
    pub fn detect_circular_patterns(&self) -> Vec<AnomalyScore> {
        detect_circular_patterns_cpu(
            &self.network_grid_current,
            &self.network_grid_previous,
            self.config.circular_threshold,
        )
    }

    /// Get all detected anomalies.
    pub fn detect_all(&self) -> PatternDetectionResult {
        let velocity_anomalies = self.detect_velocity_anomalies();
        let circular_patterns = self.detect_circular_patterns();

        PatternDetectionResult {
            velocity_anomalies,
            circular_patterns,
            grid_width: self.config.grid_width,
            grid_height: self.config.grid_height,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &StencilPatternConfig {
        &self.config
    }
}

/// Results from pattern detection.
#[derive(Debug, Clone)]
pub struct PatternDetectionResult {
    pub velocity_anomalies: Vec<AnomalyScore>,
    pub circular_patterns: Vec<AnomalyScore>,
    pub grid_width: usize,
    pub grid_height: usize,
}

impl PatternDetectionResult {
    /// Get total number of detected patterns.
    pub fn total_patterns(&self) -> usize {
        self.velocity_anomalies.len() + self.circular_patterns.len()
    }

    /// Get top N anomalies by score.
    pub fn top_anomalies(&self, n: usize) -> Vec<&AnomalyScore> {
        let mut all: Vec<_> = self
            .velocity_anomalies
            .iter()
            .chain(self.circular_patterns.iter())
            .collect();

        all.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        all.into_iter().take(n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_grid() {
        let mut grid = VelocityGrid::new(16, 16, 1000);
        grid.reset(0);

        grid.add_transaction(5, 500); // bucket (5, 0)
        grid.add_transaction(5, 1500); // bucket (5, 1)
        grid.add_transaction(5, 500); // bucket (5, 0) again

        assert_eq!(grid.get(5, 0), 2);
        assert_eq!(grid.get(5, 1), 1);
        assert_eq!(grid.get(0, 0), 0);
    }

    #[test]
    fn test_network_grid() {
        let mut grid = NetworkGrid::new(16, 16);

        grid.add_transaction(1, 2, 1000, false); // 1 -> 2
        grid.add_transaction(2, 1, 1000, true); // 2 -> 1

        let cell_1_2 = grid.get(1 % 16, 2 % 16);
        assert_eq!(cell_1_2.tx_count, 1);
        assert_eq!(cell_1_2.outbound_count, 1);
    }

    #[test]
    fn test_velocity_anomaly_detection() {
        let mut current = VelocityGrid::new(8, 8, 1000);
        let mut previous = VelocityGrid::new(8, 8, 1000);

        current.reset(0);
        previous.reset(0);

        // Create a spike at (4, 4)
        current.data[4 * 8 + 4] = 100; // Very high
        current.data[3 * 8 + 4] = 2; // Neighbors low
        current.data[5 * 8 + 4] = 2;
        current.data[4 * 8 + 3] = 2;
        current.data[4 * 8 + 5] = 2;

        let anomalies = detect_velocity_anomalies_cpu(&current, &previous, 10.0);

        assert!(!anomalies.is_empty());
        assert!(anomalies
            .iter()
            .any(|a| a.customer_bucket == 4 && a.time_bucket == 4));
    }

    #[test]
    fn test_stencil_backend() {
        let config = StencilPatternConfig {
            grid_width: 16,
            grid_height: 16,
            ..Default::default()
        };

        let mut backend = StencilPatternBackend::new(config);

        let transactions = vec![
            GpuTransaction {
                customer_id: 1,
                timestamp: 100,
                amount_cents: 1000,
                ..Default::default()
            },
            GpuTransaction {
                customer_id: 2,
                timestamp: 200,
                amount_cents: 2000,
                ..Default::default()
            },
        ];

        backend.add_transactions(&transactions);

        let result = backend.detect_all();
        assert_eq!(result.grid_width, 16);
        assert_eq!(result.grid_height, 16);
    }
}
