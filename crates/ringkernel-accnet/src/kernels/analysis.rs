//! Network analysis kernels for fraud detection and GAAP compliance.
//!
//! These kernels analyze the accounting network graph to detect:
//! - Suspense accounts
//! - GAAP violations
//! - Fraud patterns (circular flows, Benford violations, etc.)

use crate::models::{
    AccountType, AccountingNetwork, FraudPattern, FraudPatternType, GaapViolation,
    GaapViolationType,
};

/// Configuration for analysis kernels.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Block size for GPU dispatch.
    pub block_size: u32,
    /// Suspense score threshold.
    pub suspense_threshold: f32,
    /// Enable Benford's Law analysis.
    pub benford_enabled: bool,
    /// Chi-squared threshold for Benford violation.
    pub benford_chi_sq_threshold: f64,
    /// Enable circular flow detection.
    pub circular_detection_enabled: bool,
    /// Maximum cycle length to detect.
    pub max_cycle_length: usize,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            suspense_threshold: 0.7,
            benford_enabled: true,
            benford_chi_sq_threshold: 15.507, // Chi-squared critical value for 8 DOF at α=0.05
            circular_detection_enabled: true,
            max_cycle_length: 10,
        }
    }
}

/// Result of network analysis.
#[derive(Debug, Clone, Default)]
pub struct AnalysisResult {
    /// Detected suspense accounts (index, score).
    pub suspense_accounts: Vec<(u16, f32)>,
    /// Detected GAAP violations.
    pub gaap_violations: Vec<GaapViolation>,
    /// Detected fraud patterns.
    pub fraud_patterns: Vec<FraudPattern>,
    /// Analysis statistics.
    pub stats: AnalysisStats,
}

/// Statistics from analysis.
#[derive(Debug, Clone, Default)]
pub struct AnalysisStats {
    /// Accounts analyzed.
    pub accounts_analyzed: usize,
    /// Flows analyzed.
    pub flows_analyzed: usize,
    /// Suspense accounts found.
    pub suspense_count: usize,
    /// GAAP violations found.
    pub gaap_violation_count: usize,
    /// Fraud patterns found.
    pub fraud_pattern_count: usize,
}

/// Network analysis kernel dispatcher.
pub struct AnalysisKernel {
    config: AnalysisConfig,
}

impl AnalysisKernel {
    /// Create a new analysis kernel.
    pub fn new(config: AnalysisConfig) -> Self {
        Self { config }
    }

    /// Analyze the network (CPU fallback).
    pub fn analyze(&self, network: &AccountingNetwork) -> AnalysisResult {
        let mut result = AnalysisResult::default();

        // Suspense detection
        for account in &network.accounts {
            if account.suspense_score >= self.config.suspense_threshold {
                result
                    .suspense_accounts
                    .push((account.index, account.suspense_score));
            }
        }

        // GAAP violation detection
        result.gaap_violations = self.detect_gaap_violations(network);

        // Fraud pattern detection
        if self.config.benford_enabled {
            if let Some(pattern) = self.check_benford_violation(network) {
                result.fraud_patterns.push(pattern);
            }
        }

        if self.config.circular_detection_enabled {
            result
                .fraud_patterns
                .extend(self.detect_circular_flows(network));
        }

        // Update stats
        result.stats.accounts_analyzed = network.accounts.len();
        result.stats.flows_analyzed = network.flows.len();
        result.stats.suspense_count = result.suspense_accounts.len();
        result.stats.gaap_violation_count = result.gaap_violations.len();
        result.stats.fraud_pattern_count = result.fraud_patterns.len();

        result
    }

    /// Detect GAAP violations.
    fn detect_gaap_violations(&self, network: &AccountingNetwork) -> Vec<GaapViolation> {
        let mut violations = Vec::new();

        // Check for Revenue→Cash direct (should go through A/R)
        for flow in &network.flows {
            let source = network.accounts.get(flow.source_account_index as usize);
            let target = network.accounts.get(flow.target_account_index as usize);

            if let (Some(src), Some(tgt)) = (source, target) {
                // Revenue directly to Cash
                if src.account_type == AccountType::Revenue
                    && tgt.account_type == AccountType::Asset
                {
                    violations.push(GaapViolation::new(
                        GaapViolationType::RevenueToCashDirect,
                        flow.source_account_index,
                        flow.target_account_index,
                        flow.amount,
                        flow.journal_entry_id,
                    ));
                }

                // Revenue to Expense (improper offset)
                if src.account_type == AccountType::Revenue
                    && tgt.account_type == AccountType::Expense
                {
                    violations.push(GaapViolation::new(
                        GaapViolationType::RevenueToExpense,
                        flow.source_account_index,
                        flow.target_account_index,
                        flow.amount,
                        flow.journal_entry_id,
                    ));
                }
            }
        }

        violations
    }

    /// Check for Benford's Law violations.
    fn check_benford_violation(&self, network: &AccountingNetwork) -> Option<FraudPattern> {
        // Count first digits from flow amounts
        let mut digit_counts = [0u32; 9];
        let mut total = 0u32;

        for flow in &network.flows {
            let amount = flow.amount.abs();
            if amount.mantissa > 0 {
                let first_digit = Self::first_digit(amount.mantissa.unsigned_abs());
                if first_digit >= 1 && first_digit <= 9 {
                    digit_counts[(first_digit - 1) as usize] += 1;
                    total += 1;
                }
            }
        }

        if total < 100 {
            return None; // Not enough data
        }

        let chi_sq = crate::models::benford_chi_squared(&digit_counts, total);

        if chi_sq > self.config.benford_chi_sq_threshold {
            let mut pattern = FraudPattern::new(FraudPatternType::BenfordViolation);
            pattern.risk_score = (chi_sq / 50.0).min(1.0) as f32;
            Some(pattern)
        } else {
            None
        }
    }

    /// Get first significant digit.
    fn first_digit(mut n: u128) -> u32 {
        while n >= 10 {
            n /= 10;
        }
        n as u32
    }

    /// Detect circular flows (simplified DFS-based).
    fn detect_circular_flows(&self, network: &AccountingNetwork) -> Vec<FraudPattern> {
        let mut patterns = Vec::new();
        let n = network.accounts.len();

        if n == 0 {
            return patterns;
        }

        // Build adjacency list
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        for flow in &network.flows {
            let src = flow.source_account_index as usize;
            let tgt = flow.target_account_index as usize;
            if src < n && tgt < n {
                adj[src].push(tgt);
            }
        }

        // Simple cycle detection using DFS
        let mut visited = vec![false; n];
        let mut rec_stack = vec![false; n];

        for start in 0..n {
            if !visited[start] {
                let mut path = Vec::new();
                if self.has_cycle(
                    &adj,
                    start,
                    &mut visited,
                    &mut rec_stack,
                    &mut path,
                    self.config.max_cycle_length,
                ) {
                    let mut pattern = FraudPattern::new(FraudPatternType::CircularFlow);
                    pattern.account_count = path.len() as u16;
                    for (i, &idx) in path.iter().enumerate().take(8) {
                        pattern.involved_accounts[i] = idx as u16;
                    }
                    patterns.push(pattern);
                }
            }
        }

        patterns
    }

    /// DFS helper for cycle detection.
    fn has_cycle(
        &self,
        adj: &[Vec<usize>],
        node: usize,
        visited: &mut [bool],
        rec_stack: &mut [bool],
        path: &mut Vec<usize>,
        max_len: usize,
    ) -> bool {
        visited[node] = true;
        rec_stack[node] = true;
        path.push(node);

        if path.len() > max_len {
            path.pop();
            rec_stack[node] = false;
            return false;
        }

        for &neighbor in &adj[node] {
            if !visited[neighbor] {
                if self.has_cycle(adj, neighbor, visited, rec_stack, path, max_len) {
                    return true;
                }
            } else if rec_stack[neighbor] {
                return true;
            }
        }

        path.pop();
        rec_stack[node] = false;
        false
    }
}

impl Default for AnalysisKernel {
    fn default() -> Self {
        Self::new(AnalysisConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_kernel_creation() {
        let kernel = AnalysisKernel::default();
        assert!(kernel.config.benford_enabled);
    }

    #[test]
    fn test_first_digit() {
        assert_eq!(AnalysisKernel::first_digit(12345), 1);
        assert_eq!(AnalysisKernel::first_digit(999), 9);
        assert_eq!(AnalysisKernel::first_digit(5), 5);
    }
}
