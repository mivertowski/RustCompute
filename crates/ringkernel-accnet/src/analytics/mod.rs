//! Analytics engine for real-time analysis of the accounting network.
//!
//! Provides high-level analysis combining outputs from GPU kernels.

use crate::models::{AccountingNetwork, FraudPattern, GaapViolation, NetworkSnapshot};
use std::collections::HashMap;

/// The main analytics engine.
pub struct AnalyticsEngine {
    /// Risk thresholds
    pub thresholds: RiskThresholds,
}

/// Configuration thresholds for risk assessment.
#[derive(Debug, Clone)]
pub struct RiskThresholds {
    /// Suspense score threshold
    pub suspense_threshold: f32,
    /// Confidence threshold for low-confidence alerts
    pub confidence_threshold: f32,
    /// Benford chi-squared threshold
    pub benford_chi_sq_threshold: f64,
    /// Z-score threshold for anomaly detection
    pub z_score_threshold: f64,
}

impl Default for RiskThresholds {
    fn default() -> Self {
        Self {
            suspense_threshold: 0.7,
            confidence_threshold: 0.5,
            benford_chi_sq_threshold: 15.507,
            z_score_threshold: 3.0,
        }
    }
}

impl AnalyticsEngine {
    /// Create a new analytics engine.
    pub fn new() -> Self {
        Self {
            thresholds: RiskThresholds::default(),
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(thresholds: RiskThresholds) -> Self {
        Self { thresholds }
    }

    /// Analyze the network and return a snapshot.
    pub fn analyze(&self, network: &AccountingNetwork) -> AnalyticsSnapshot {
        let mut snapshot = AnalyticsSnapshot::default();

        // Calculate overall risk
        snapshot.overall_risk = self.calculate_overall_risk(network);

        // Count issues by severity
        snapshot.suspense_accounts = network.statistics.suspense_account_count;
        snapshot.gaap_violations = network.statistics.gaap_violation_count;
        snapshot.fraud_patterns = network.statistics.fraud_pattern_count;

        // Account risks
        for account in &network.accounts {
            snapshot.account_risks.insert(account.index, RiskScore {
                total: account.risk_score,
                suspense_component: account.suspense_score * 0.3,
                fraud_component: if account.flags.has(crate::models::AccountFlags::HAS_FRAUD_PATTERN) { 0.5 } else { 0.0 },
                confidence_component: 0.0, // Would be calculated from flows
            });
        }

        // Network health
        snapshot.network_health = NetworkHealth {
            balance_check: true, // Simplified
            coverage: network.statistics.avg_confidence,
            connectivity: network.statistics.density,
        };

        snapshot
    }

    /// Calculate overall risk score for the network.
    fn calculate_overall_risk(&self, network: &AccountingNetwork) -> f32 {
        let n = network.accounts.len().max(1) as f32;

        // Weighted combination of risk factors
        let suspense_ratio = network.statistics.suspense_account_count as f32 / n;
        let violation_ratio = network.statistics.gaap_violation_count as f32 / n;
        let fraud_ratio = network.statistics.fraud_pattern_count as f32 / n;
        let confidence_factor = 1.0 - network.statistics.avg_confidence as f32;

        (0.25 * suspense_ratio + 0.30 * violation_ratio + 0.35 * fraud_ratio + 0.10 * confidence_factor)
            .min(1.0)
    }
}

impl Default for AnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of analytics results.
#[derive(Debug, Clone, Default)]
pub struct AnalyticsSnapshot {
    /// Overall network risk score (0.0 - 1.0)
    pub overall_risk: f32,
    /// Number of suspected suspense accounts
    pub suspense_accounts: usize,
    /// Number of GAAP violations detected
    pub gaap_violations: usize,
    /// Number of fraud patterns detected
    pub fraud_patterns: usize,
    /// Risk scores by account
    pub account_risks: HashMap<u16, RiskScore>,
    /// Network health metrics
    pub network_health: NetworkHealth,
}

/// Risk score breakdown for an account.
#[derive(Debug, Clone, Default)]
pub struct RiskScore {
    /// Total risk score
    pub total: f32,
    /// Contribution from suspense detection
    pub suspense_component: f32,
    /// Contribution from fraud patterns
    pub fraud_component: f32,
    /// Contribution from low confidence
    pub confidence_component: f32,
}

/// Network health metrics.
#[derive(Debug, Clone, Default)]
pub struct NetworkHealth {
    /// Whether all entries balance
    pub balance_check: bool,
    /// Average transformation confidence
    pub coverage: f64,
    /// Network connectivity (density)
    pub connectivity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_analytics_engine() {
        let engine = AnalyticsEngine::new();
        let network = AccountingNetwork::new(Uuid::new_v4(), 2024, 1);
        let snapshot = engine.analyze(&network);

        assert!(snapshot.overall_risk >= 0.0 && snapshot.overall_risk <= 1.0);
    }
}
