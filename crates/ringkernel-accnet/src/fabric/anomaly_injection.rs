//! Anomaly injection for testing fraud and violation detection.
//!
//! This module allows configurable injection of various anomaly patterns
//! into the synthetic data stream for testing detection algorithms.

use crate::models::{
    Decimal128, FraudPatternType, GaapViolationType, HybridTimestamp,
    JournalEntry, JournalLineItem, LineType, SolvingMethod,
};
use rand::prelude::*;
use std::collections::HashMap;
use uuid::Uuid;

/// Configuration for anomaly injection.
#[derive(Debug, Clone)]
pub struct AnomalyInjectionConfig {
    /// Overall injection rate (0.0 - 1.0)
    pub injection_rate: f64,

    /// Fraud pattern injection settings
    pub fraud_patterns: Vec<FraudPatternConfig>,

    /// GAAP violation injection settings
    pub gaap_violations: Vec<GaapViolationConfig>,

    /// Timing anomaly injection settings
    pub timing_anomalies: TimingAnomalyConfig,

    /// Amount anomaly injection settings
    pub amount_anomalies: AmountAnomalyConfig,

    /// Whether injected anomalies should be labeled (for training/evaluation)
    pub label_anomalies: bool,
}

impl Default for AnomalyInjectionConfig {
    fn default() -> Self {
        Self {
            injection_rate: 0.02, // 2% of transactions
            fraud_patterns: vec![
                FraudPatternConfig::circular_flow(0.25),
                FraudPatternConfig::threshold_clustering(0.20),
                FraudPatternConfig::round_amounts(0.15),
                FraudPatternConfig::velocity(0.15),
                FraudPatternConfig::dormant_activation(0.10),
                FraudPatternConfig::unusual_pairing(0.15),
            ],
            gaap_violations: vec![
                GaapViolationConfig::new(GaapViolationType::RevenueToCashDirect, 0.30),
                GaapViolationConfig::new(GaapViolationType::ExpenseToAsset, 0.25),
                GaapViolationConfig::new(GaapViolationType::CashToRevenue, 0.20),
                GaapViolationConfig::new(GaapViolationType::RevenueToExpense, 0.10),
                GaapViolationConfig::new(GaapViolationType::UnbalancedEntry, 0.15),
            ],
            timing_anomalies: TimingAnomalyConfig::default(),
            amount_anomalies: AmountAnomalyConfig::default(),
            label_anomalies: true,
        }
    }
}

impl AnomalyInjectionConfig {
    /// Create a configuration with no anomaly injection.
    pub fn disabled() -> Self {
        Self {
            injection_rate: 0.0,
            ..Default::default()
        }
    }

    /// Create a high anomaly rate for testing (10%).
    pub fn high_rate() -> Self {
        Self {
            injection_rate: 0.10,
            ..Default::default()
        }
    }

    /// Validate that probability distributions sum correctly.
    pub fn validate(&self) -> Result<(), String> {
        let fraud_total: f64 = self.fraud_patterns.iter().map(|p| p.probability).sum();
        if (fraud_total - 1.0).abs() > 0.01 {
            return Err(format!("Fraud pattern probabilities must sum to 1.0, got {}", fraud_total));
        }

        let gaap_total: f64 = self.gaap_violations.iter().map(|v| v.probability).sum();
        if (gaap_total - 1.0).abs() > 0.01 {
            return Err(format!("GAAP violation probabilities must sum to 1.0, got {}", gaap_total));
        }

        Ok(())
    }
}

/// Configuration for a specific fraud pattern.
#[derive(Debug, Clone)]
pub struct FraudPatternConfig {
    /// Type of fraud pattern
    pub pattern_type: FraudPatternType,
    /// Probability of this pattern (within fraud injections)
    pub probability: f64,
    /// Number of accounts involved (min, max)
    pub account_count: (u8, u8),
    /// Amount range for fraudulent transactions
    pub amount_range: (f64, f64),
}

impl FraudPatternConfig {
    /// Create a circular flow pattern configuration.
    pub fn circular_flow(probability: f64) -> Self {
        Self {
            pattern_type: FraudPatternType::CircularFlow,
            probability,
            account_count: (3, 5),
            amount_range: (10000.0, 100000.0),
        }
    }

    /// Create a threshold clustering pattern configuration.
    pub fn threshold_clustering(probability: f64) -> Self {
        Self {
            pattern_type: FraudPatternType::ThresholdClustering,
            probability,
            account_count: (2, 2),
            amount_range: (9000.0, 9999.0), // Just below $10k threshold
        }
    }

    /// Create a round amounts pattern configuration.
    pub fn round_amounts(probability: f64) -> Self {
        Self {
            pattern_type: FraudPatternType::RoundAmounts,
            probability,
            account_count: (2, 2),
            amount_range: (1000.0, 50000.0),
        }
    }

    /// Create a high velocity pattern configuration.
    pub fn velocity(probability: f64) -> Self {
        Self {
            pattern_type: FraudPatternType::HighVelocity,
            probability,
            account_count: (3, 6),
            amount_range: (5000.0, 50000.0),
        }
    }

    /// Create a dormant activation pattern configuration.
    pub fn dormant_activation(probability: f64) -> Self {
        Self {
            pattern_type: FraudPatternType::DormantActivation,
            probability,
            account_count: (2, 2),
            amount_range: (10000.0, 500000.0),
        }
    }

    /// Create an unusual pairing pattern configuration.
    pub fn unusual_pairing(probability: f64) -> Self {
        Self {
            pattern_type: FraudPatternType::UnusualPairing,
            probability,
            account_count: (2, 2),
            amount_range: (5000.0, 100000.0),
        }
    }
}

/// Configuration for a GAAP violation.
#[derive(Debug, Clone)]
pub struct GaapViolationConfig {
    /// Type of violation
    pub violation_type: GaapViolationType,
    /// Probability of this violation (within GAAP injections)
    pub probability: f64,
}

impl GaapViolationConfig {
    /// Create a new GAAP violation configuration.
    pub fn new(violation_type: GaapViolationType, probability: f64) -> Self {
        Self {
            violation_type,
            probability,
        }
    }
}

/// Configuration for timing-based anomalies.
#[derive(Debug, Clone)]
pub struct TimingAnomalyConfig {
    /// Inject after-hours entries
    pub after_hours: bool,
    /// Inject weekend entries
    pub weekend_entries: bool,
    /// Inject holiday entries
    pub holiday_entries: bool,
    /// Inject month-end manipulation
    pub month_end_manipulation: bool,
}

impl Default for TimingAnomalyConfig {
    fn default() -> Self {
        Self {
            after_hours: true,
            weekend_entries: true,
            holiday_entries: false,
            month_end_manipulation: true,
        }
    }
}

/// Configuration for amount-based anomalies.
#[derive(Debug, Clone)]
pub struct AmountAnomalyConfig {
    /// Inject round amount anomalies
    pub round_amounts: bool,
    /// Inject Benford's Law violations
    pub benford_violations: bool,
    /// Inject outlier amounts
    pub outliers: bool,
    /// Outlier multiplier (e.g., 10x normal)
    pub outlier_multiplier: f64,
}

impl Default for AmountAnomalyConfig {
    fn default() -> Self {
        Self {
            round_amounts: true,
            benford_violations: true,
            outliers: true,
            outlier_multiplier: 10.0,
        }
    }
}

/// Injector that modifies transactions to create anomalies.
pub struct AnomalyInjector {
    /// Configuration
    config: AnomalyInjectionConfig,
    /// Random number generator
    rng: StdRng,
    /// Account type mapping (index -> is_asset, is_revenue, etc.)
    account_types: HashMap<u16, AccountTypeInfo>,
    /// Injection statistics
    stats: InjectionStats,
    /// Pending circular flow entries
    pending_circular_flows: Vec<CircularFlowState>,
    /// Dormant accounts (haven't been used recently)
    dormant_accounts: Vec<u16>,
}

/// Information about an account's type for violation detection.
#[derive(Debug, Clone, Copy)]
pub struct AccountTypeInfo {
    pub is_asset: bool,
    pub is_liability: bool,
    pub is_revenue: bool,
    pub is_expense: bool,
    pub is_equity: bool,
    pub is_cash: bool,
    pub is_suspense: bool,
}

impl Default for AccountTypeInfo {
    fn default() -> Self {
        Self {
            is_asset: false,
            is_liability: false,
            is_revenue: false,
            is_expense: false,
            is_equity: false,
            is_cash: false,
            is_suspense: false,
        }
    }
}

/// State for multi-entry circular flow injection.
#[derive(Debug, Clone)]
struct CircularFlowState {
    /// Accounts in the circle
    accounts: Vec<u16>,
    /// Current position in the circle
    current_position: usize,
    /// Amount being circulated
    amount: Decimal128,
    /// Entries remaining
    remaining: usize,
}

/// Statistics about injected anomalies.
#[derive(Debug, Clone, Default)]
pub struct InjectionStats {
    /// Total entries processed
    pub entries_processed: u64,
    /// Total anomalies injected
    pub anomalies_injected: u64,
    /// Fraud patterns by type
    pub fraud_patterns: HashMap<FraudPatternType, u32>,
    /// GAAP violations by type
    pub gaap_violations: HashMap<GaapViolationType, u32>,
    /// Timing anomalies
    pub timing_anomalies: u32,
    /// Amount anomalies
    pub amount_anomalies: u32,
}

/// Result of anomaly injection.
#[derive(Debug, Clone)]
pub struct InjectionResult {
    /// Modified entry (or original if not modified)
    pub entry: JournalEntry,
    /// Modified debit lines
    pub debit_lines: Vec<JournalLineItem>,
    /// Modified credit lines
    pub credit_lines: Vec<JournalLineItem>,
    /// Whether an anomaly was injected
    pub anomaly_injected: bool,
    /// Label for the anomaly (if labeling is enabled)
    pub anomaly_label: Option<AnomalyLabel>,
}

/// Label describing an injected anomaly.
#[derive(Debug, Clone)]
pub enum AnomalyLabel {
    /// Fraud pattern
    FraudPattern(FraudPatternType),
    /// GAAP violation
    GaapViolation(GaapViolationType),
    /// Timing anomaly
    TimingAnomaly(String),
    /// Amount anomaly
    AmountAnomaly(String),
}

impl AnomalyInjector {
    /// Create a new anomaly injector.
    pub fn new(config: AnomalyInjectionConfig, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| rand::thread_rng().gen());
        Self {
            config,
            rng: StdRng::seed_from_u64(seed),
            account_types: HashMap::new(),
            stats: InjectionStats::default(),
            pending_circular_flows: Vec::new(),
            dormant_accounts: Vec::new(),
        }
    }

    /// Register an account's type information.
    pub fn register_account(&mut self, index: u16, info: AccountTypeInfo) {
        self.account_types.insert(index, info);
    }

    /// Mark an account as dormant.
    pub fn mark_dormant(&mut self, index: u16) {
        if !self.dormant_accounts.contains(&index) {
            self.dormant_accounts.push(index);
        }
    }

    /// Process an entry and potentially inject an anomaly.
    pub fn process(
        &mut self,
        entry: JournalEntry,
        debit_lines: Vec<JournalLineItem>,
        credit_lines: Vec<JournalLineItem>,
    ) -> InjectionResult {
        self.stats.entries_processed += 1;

        // Check if we should inject an anomaly
        if self.config.injection_rate <= 0.0 || self.rng.gen::<f64>() > self.config.injection_rate {
            return InjectionResult {
                entry,
                debit_lines,
                credit_lines,
                anomaly_injected: false,
                anomaly_label: None,
            };
        }

        // Decide what type of anomaly to inject
        let anomaly_type: f64 = self.rng.gen();

        if anomaly_type < 0.5 {
            // Fraud pattern (50% of anomalies)
            self.inject_fraud_pattern(entry, debit_lines, credit_lines)
        } else if anomaly_type < 0.8 {
            // GAAP violation (30% of anomalies)
            self.inject_gaap_violation(entry, debit_lines, credit_lines)
        } else if anomaly_type < 0.9 {
            // Timing anomaly (10% of anomalies)
            self.inject_timing_anomaly(entry, debit_lines, credit_lines)
        } else {
            // Amount anomaly (10% of anomalies)
            self.inject_amount_anomaly(entry, debit_lines, credit_lines)
        }
    }

    /// Inject a fraud pattern.
    fn inject_fraud_pattern(
        &mut self,
        mut entry: JournalEntry,
        mut debit_lines: Vec<JournalLineItem>,
        mut credit_lines: Vec<JournalLineItem>,
    ) -> InjectionResult {
        // Select fraud pattern type
        let pattern_type = self.select_fraud_pattern();

        let label = match pattern_type {
            FraudPatternType::ThresholdClustering => {
                // Modify amount to be just below threshold
                let threshold = 10000.0;
                let new_amount = Decimal128::from_f64(
                    threshold - self.rng.gen_range(1.0..999.0)
                );

                for line in &mut debit_lines {
                    line.amount = new_amount;
                }
                for line in &mut credit_lines {
                    line.amount = new_amount;
                }
                entry.total_debits = new_amount;
                entry.total_credits = new_amount;

                Some(AnomalyLabel::FraudPattern(FraudPatternType::ThresholdClustering))
            }

            FraudPatternType::RoundAmounts => {
                // Make amount suspiciously round
                let round_amounts = [1000.0, 5000.0, 10000.0, 25000.0, 50000.0, 100000.0];
                let new_amount = Decimal128::from_f64(
                    round_amounts[self.rng.gen_range(0..round_amounts.len())]
                );

                for line in &mut debit_lines {
                    line.amount = new_amount;
                }
                for line in &mut credit_lines {
                    line.amount = new_amount;
                }
                entry.total_debits = new_amount;
                entry.total_credits = new_amount;

                Some(AnomalyLabel::FraudPattern(FraudPatternType::RoundAmounts))
            }

            FraudPatternType::UnusualPairing => {
                // Create an implausible account pairing
                // Find a revenue account and expense account
                if let (Some(revenue_idx), Some(expense_idx)) = self.find_unusual_pair() {
                    if !debit_lines.is_empty() {
                        debit_lines[0].account_index = revenue_idx; // Revenue as debit is unusual
                    }
                    if !credit_lines.is_empty() {
                        credit_lines[0].account_index = expense_idx; // Expense as credit is unusual
                    }
                    Some(AnomalyLabel::FraudPattern(FraudPatternType::UnusualPairing))
                } else {
                    None
                }
            }

            _ => {
                // Other patterns require multi-entry injection (simplified here)
                Some(AnomalyLabel::FraudPattern(pattern_type))
            }
        };

        if label.is_some() {
            self.stats.anomalies_injected += 1;
            *self.stats.fraud_patterns.entry(pattern_type).or_insert(0) += 1;
        }

        InjectionResult {
            entry,
            debit_lines,
            credit_lines,
            anomaly_injected: label.is_some(),
            anomaly_label: if self.config.label_anomalies { label } else { None },
        }
    }

    /// Inject a GAAP violation.
    fn inject_gaap_violation(
        &mut self,
        mut entry: JournalEntry,
        mut debit_lines: Vec<JournalLineItem>,
        mut credit_lines: Vec<JournalLineItem>,
    ) -> InjectionResult {
        let violation_type = self.select_gaap_violation();

        let label = match violation_type {
            GaapViolationType::UnbalancedEntry => {
                // Make entry unbalanced
                if !credit_lines.is_empty() {
                    let adjustment = Decimal128::from_f64(
                        self.rng.gen_range(100.0..1000.0)
                    );
                    credit_lines[0].amount = credit_lines[0].amount + adjustment;
                    entry.total_credits = entry.total_credits + adjustment;
                    entry.flags.0 &= !crate::models::JournalEntryFlags::IS_BALANCED;
                }
                Some(AnomalyLabel::GaapViolation(GaapViolationType::UnbalancedEntry))
            }

            GaapViolationType::RevenueToCashDirect => {
                // Find revenue and cash accounts
                if let (Some(revenue_idx), Some(cash_idx)) = self.find_revenue_cash_pair() {
                    if !debit_lines.is_empty() {
                        debit_lines[0].account_index = cash_idx;
                    }
                    if !credit_lines.is_empty() {
                        credit_lines[0].account_index = revenue_idx;
                    }
                    Some(AnomalyLabel::GaapViolation(GaapViolationType::RevenueToCashDirect))
                } else {
                    None
                }
            }

            _ => {
                // Other violations need specific account pairs
                Some(AnomalyLabel::GaapViolation(violation_type))
            }
        };

        if label.is_some() {
            self.stats.anomalies_injected += 1;
            *self.stats.gaap_violations.entry(violation_type).or_insert(0) += 1;
        }

        InjectionResult {
            entry,
            debit_lines,
            credit_lines,
            anomaly_injected: label.is_some(),
            anomaly_label: if self.config.label_anomalies { label } else { None },
        }
    }

    /// Inject a timing anomaly.
    fn inject_timing_anomaly(
        &mut self,
        mut entry: JournalEntry,
        debit_lines: Vec<JournalLineItem>,
        credit_lines: Vec<JournalLineItem>,
    ) -> InjectionResult {
        // Modify timestamp to be after hours
        // Set hour to 23 (11 PM)
        let ms_per_day = 86_400_000u64;
        let ms_per_hour = 3_600_000u64;
        let day_start = (entry.posting_date.physical / ms_per_day) * ms_per_day;
        entry.posting_date.physical = day_start + 23 * ms_per_hour + self.rng.gen_range(0..ms_per_hour);

        self.stats.anomalies_injected += 1;
        self.stats.timing_anomalies += 1;

        InjectionResult {
            entry,
            debit_lines,
            credit_lines,
            anomaly_injected: true,
            anomaly_label: if self.config.label_anomalies {
                Some(AnomalyLabel::TimingAnomaly("after_hours".to_string()))
            } else {
                None
            },
        }
    }

    /// Inject an amount anomaly.
    fn inject_amount_anomaly(
        &mut self,
        mut entry: JournalEntry,
        mut debit_lines: Vec<JournalLineItem>,
        mut credit_lines: Vec<JournalLineItem>,
    ) -> InjectionResult {
        // Create an outlier amount
        let multiplier = self.config.amount_anomalies.outlier_multiplier;
        let current = entry.total_debits.to_f64();
        let new_amount = Decimal128::from_f64(current * multiplier);

        for line in &mut debit_lines {
            line.amount = Decimal128::from_f64(line.amount.to_f64() * multiplier);
        }
        for line in &mut credit_lines {
            line.amount = Decimal128::from_f64(line.amount.to_f64() * multiplier);
        }
        entry.total_debits = new_amount;
        entry.total_credits = new_amount;

        self.stats.anomalies_injected += 1;
        self.stats.amount_anomalies += 1;

        InjectionResult {
            entry,
            debit_lines,
            credit_lines,
            anomaly_injected: true,
            anomaly_label: if self.config.label_anomalies {
                Some(AnomalyLabel::AmountAnomaly("outlier".to_string()))
            } else {
                None
            },
        }
    }

    /// Select a fraud pattern based on configured probabilities.
    fn select_fraud_pattern(&mut self) -> FraudPatternType {
        let r: f64 = self.rng.gen();
        let mut cumulative = 0.0;

        for config in &self.config.fraud_patterns {
            cumulative += config.probability;
            if r < cumulative {
                return config.pattern_type;
            }
        }

        FraudPatternType::RoundAmounts // Default
    }

    /// Select a GAAP violation based on configured probabilities.
    fn select_gaap_violation(&mut self) -> GaapViolationType {
        let r: f64 = self.rng.gen();
        let mut cumulative = 0.0;

        for config in &self.config.gaap_violations {
            cumulative += config.probability;
            if r < cumulative {
                return config.violation_type;
            }
        }

        GaapViolationType::UnbalancedEntry // Default
    }

    /// Find an unusual account pairing (revenue-expense).
    fn find_unusual_pair(&self) -> (Option<u16>, Option<u16>) {
        let revenue = self.account_types.iter()
            .find(|(_, info)| info.is_revenue)
            .map(|(&idx, _)| idx);
        let expense = self.account_types.iter()
            .find(|(_, info)| info.is_expense)
            .map(|(&idx, _)| idx);
        (revenue, expense)
    }

    /// Find revenue and cash accounts.
    fn find_revenue_cash_pair(&self) -> (Option<u16>, Option<u16>) {
        let revenue = self.account_types.iter()
            .find(|(_, info)| info.is_revenue)
            .map(|(&idx, _)| idx);
        let cash = self.account_types.iter()
            .find(|(_, info)| info.is_cash)
            .map(|(&idx, _)| idx);
        (revenue, cash)
    }

    /// Get injection statistics.
    pub fn stats(&self) -> &InjectionStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = InjectionStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = AnomalyInjectionConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.injection_rate > 0.0);
    }

    #[test]
    fn test_injector_creation() {
        let config = AnomalyInjectionConfig::default();
        let injector = AnomalyInjector::new(config, Some(42));
        assert_eq!(injector.stats().entries_processed, 0);
    }

    #[test]
    fn test_disabled_injection() {
        let config = AnomalyInjectionConfig::disabled();
        let mut injector = AnomalyInjector::new(config, Some(42));

        let entry = JournalEntry::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            HybridTimestamp::now(),
        );

        let result = injector.process(entry, vec![], vec![]);
        assert!(!result.anomaly_injected);
    }

    #[test]
    fn test_fraud_pattern_selection() {
        let config = AnomalyInjectionConfig {
            injection_rate: 1.0, // Always inject
            ..Default::default()
        };
        let mut injector = AnomalyInjector::new(config, Some(42));

        // Process multiple entries and verify injections happen
        for _ in 0..100 {
            let entry = JournalEntry::new(
                Uuid::new_v4(),
                Uuid::new_v4(),
                HybridTimestamp::now(),
            );
            let debit = JournalLineItem::debit(0, Decimal128::from_f64(1000.0), 1);
            let credit = JournalLineItem::credit(1, Decimal128::from_f64(1000.0), 2);

            injector.process(entry, vec![debit], vec![credit]);
        }

        assert!(injector.stats().anomalies_injected > 0);
    }
}
