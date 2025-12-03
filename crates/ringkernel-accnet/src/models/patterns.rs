//! Pattern definitions for fraud detection and GAAP compliance.
//!
//! These structures define the rules and detected patterns used by
//! the analysis kernels.

use super::{AccountType, Decimal128, HybridTimestamp};
use rkyv::{Archive, Deserialize, Serialize};
use uuid::Uuid;

/// Types of fraud patterns that can be detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[repr(u8)]
pub enum FraudPatternType {
    /// Money flowing in a circle: A → B → C → A
    CircularFlow = 0,

    /// Self-loop: A → B and B → A within short timeframe
    SelfLoop = 1,

    /// First digit distribution violates Benford's Law
    BenfordViolation = 2,

    /// Transactions clustered just below approval thresholds
    ThresholdClustering = 3,

    /// Entries posted outside business hours
    AfterHoursEntry = 4,

    /// Rapid multi-hop money movement (potential kiting)
    HighVelocity = 5,

    /// Implausible account pairings (e.g., Payroll → Fixed Assets)
    UnusualPairing = 6,

    /// Sudden activity on long-dormant accounts
    DormantActivation = 7,

    /// Round amounts (possible fabrication)
    RoundAmounts = 8,

    /// Duplicate transactions (possible double-payment)
    DuplicateTransaction = 9,

    /// Split transactions to avoid detection
    StructuredTransactions = 10,

    /// Unusual reversal patterns
    ReversalAnomaly = 11,
}

impl FraudPatternType {
    /// Risk weight for this pattern type (0.0 - 1.0).
    pub fn risk_weight(&self) -> f32 {
        match self {
            FraudPatternType::CircularFlow => 0.95,
            FraudPatternType::HighVelocity => 0.90,
            FraudPatternType::ThresholdClustering => 0.85,
            FraudPatternType::StructuredTransactions => 0.85,
            FraudPatternType::DormantActivation => 0.80,
            FraudPatternType::UnusualPairing => 0.75,
            FraudPatternType::BenfordViolation => 0.70,
            FraudPatternType::AfterHoursEntry => 0.60,
            FraudPatternType::RoundAmounts => 0.50,
            FraudPatternType::SelfLoop => 0.65,
            FraudPatternType::DuplicateTransaction => 0.55,
            FraudPatternType::ReversalAnomaly => 0.60,
        }
    }

    /// Human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            FraudPatternType::CircularFlow => "Circular money flow detected (A→B→C→A)",
            FraudPatternType::SelfLoop => "Bidirectional flow between accounts",
            FraudPatternType::BenfordViolation => "Amount distribution violates Benford's Law",
            FraudPatternType::ThresholdClustering => "Amounts clustered below approval threshold",
            FraudPatternType::AfterHoursEntry => "Entry posted outside business hours",
            FraudPatternType::HighVelocity => "Rapid multi-hop money movement",
            FraudPatternType::UnusualPairing => "Implausible account combination",
            FraudPatternType::DormantActivation => "Dormant account suddenly activated",
            FraudPatternType::RoundAmounts => "Suspicious round-number amounts",
            FraudPatternType::DuplicateTransaction => "Potential duplicate transaction",
            FraudPatternType::StructuredTransactions => "Structured to avoid detection",
            FraudPatternType::ReversalAnomaly => "Unusual reversal pattern",
        }
    }

    /// Icon for UI.
    pub fn icon(&self) -> &'static str {
        match self {
            FraudPatternType::CircularFlow => "🔄",
            FraudPatternType::SelfLoop => "↔️",
            FraudPatternType::BenfordViolation => "📊",
            FraudPatternType::ThresholdClustering => "📍",
            FraudPatternType::AfterHoursEntry => "🌙",
            FraudPatternType::HighVelocity => "⚡",
            FraudPatternType::UnusualPairing => "❓",
            FraudPatternType::DormantActivation => "💤",
            FraudPatternType::RoundAmounts => "🔢",
            FraudPatternType::DuplicateTransaction => "📋",
            FraudPatternType::StructuredTransactions => "✂️",
            FraudPatternType::ReversalAnomaly => "↩️",
        }
    }
}

/// A detected fraud pattern instance.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct FraudPattern {
    /// Pattern identifier
    pub id: Uuid,
    /// Type of pattern
    pub pattern_type: FraudPatternType,
    /// Risk score (0.0 - 1.0)
    pub risk_score: f32,
    /// Total amount involved
    pub amount: Decimal128,
    /// Number of accounts involved
    pub account_count: u16,
    /// Number of transactions in the pattern
    pub transaction_count: u16,
    /// Time span of the pattern (days)
    pub timeframe_days: u16,
    /// Padding
    pub _pad: u16,
    /// First timestamp
    pub first_seen: HybridTimestamp,
    /// Last timestamp
    pub last_seen: HybridTimestamp,
    /// Involved account indices (up to 8)
    pub involved_accounts: [u16; 8],
}

impl FraudPattern {
    pub fn new(pattern_type: FraudPatternType) -> Self {
        Self {
            id: Uuid::new_v4(),
            pattern_type,
            risk_score: pattern_type.risk_weight(),
            amount: Decimal128::ZERO,
            account_count: 0,
            transaction_count: 0,
            timeframe_days: 0,
            _pad: 0,
            first_seen: HybridTimestamp::zero(),
            last_seen: HybridTimestamp::zero(),
            involved_accounts: [u16::MAX; 8],
        }
    }

    /// Add an account to the pattern.
    pub fn add_account(&mut self, account_index: u16) {
        for i in 0..8 {
            if self.involved_accounts[i] == u16::MAX {
                self.involved_accounts[i] = account_index;
                self.account_count += 1;
                break;
            }
        }
    }

    /// Get involved accounts as a vector.
    pub fn get_involved_accounts(&self) -> Vec<u16> {
        self.involved_accounts
            .iter()
            .filter(|&&idx| idx != u16::MAX)
            .copied()
            .collect()
    }
}

/// GAAP violation severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[repr(u8)]
pub enum ViolationSeverity {
    /// Minor: Unusual but not necessarily wrong
    Low = 0,
    /// Moderate: Needs review
    Medium = 1,
    /// Significant: Likely error or policy violation
    High = 2,
    /// Critical: Definite violation requiring immediate action
    Critical = 3,
}

impl ViolationSeverity {
    pub fn color(&self) -> [u8; 3] {
        match self {
            ViolationSeverity::Low => [255, 235, 59],     // Yellow
            ViolationSeverity::Medium => [255, 152, 0],  // Orange
            ViolationSeverity::High => [244, 67, 54],    // Red
            ViolationSeverity::Critical => [183, 28, 28], // Dark red
        }
    }
}

/// Types of GAAP violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[repr(u8)]
pub enum GaapViolationType {
    /// Revenue → Cash (should use Receivable)
    RevenueToCashDirect = 0,

    /// Revenue → Expense (impossible under accounting equation)
    RevenueToExpense = 1,

    /// Cash → Revenue (backward flow)
    CashToRevenue = 2,

    /// Expense → Asset (capitalization bypass)
    ExpenseToAsset = 3,

    /// Liability → Revenue (debt forgiveness misclassification)
    LiabilityToRevenue = 4,

    /// COGS without Inventory movement
    CogsWithoutInventory = 5,

    /// Direct increase to Accumulated Depreciation
    AccumDepreciationIncrease = 6,

    /// Direct modification of Retained Earnings (except closing)
    RetainedEarningsModification = 7,

    /// Intercompany imbalance
    IntercompanyImbalance = 8,

    /// Unbalanced entry
    UnbalancedEntry = 9,
}

impl GaapViolationType {
    /// Default severity for this violation type.
    pub fn default_severity(&self) -> ViolationSeverity {
        match self {
            GaapViolationType::RevenueToExpense => ViolationSeverity::Critical,
            GaapViolationType::UnbalancedEntry => ViolationSeverity::Critical,
            GaapViolationType::RetainedEarningsModification => ViolationSeverity::High,
            GaapViolationType::AccumDepreciationIncrease => ViolationSeverity::High,
            GaapViolationType::RevenueToCashDirect => ViolationSeverity::Medium,
            GaapViolationType::CashToRevenue => ViolationSeverity::Medium,
            GaapViolationType::LiabilityToRevenue => ViolationSeverity::High,
            GaapViolationType::ExpenseToAsset => ViolationSeverity::Medium,
            GaapViolationType::CogsWithoutInventory => ViolationSeverity::Medium,
            GaapViolationType::IntercompanyImbalance => ViolationSeverity::Low,
        }
    }

    /// Description for UI.
    pub fn description(&self) -> &'static str {
        match self {
            GaapViolationType::RevenueToCashDirect => "Revenue directly to Cash (bypass A/R)",
            GaapViolationType::RevenueToExpense => "Revenue to Expense (accounting equation violation)",
            GaapViolationType::CashToRevenue => "Cash to Revenue (backward flow)",
            GaapViolationType::ExpenseToAsset => "Expense to Asset (improper capitalization)",
            GaapViolationType::LiabilityToRevenue => "Liability to Revenue (misclassification)",
            GaapViolationType::CogsWithoutInventory => "COGS without Inventory movement",
            GaapViolationType::AccumDepreciationIncrease => "Direct Accum. Depreciation increase",
            GaapViolationType::RetainedEarningsModification => "Direct Retained Earnings modification",
            GaapViolationType::IntercompanyImbalance => "Intercompany accounts don't balance",
            GaapViolationType::UnbalancedEntry => "Debits ≠ Credits",
        }
    }

    /// Check if a flow between two account types constitutes this violation.
    pub fn matches(&self, source_type: AccountType, target_type: AccountType) -> bool {
        match self {
            GaapViolationType::RevenueToCashDirect => {
                source_type == AccountType::Revenue && target_type == AccountType::Asset
            }
            GaapViolationType::RevenueToExpense => {
                source_type == AccountType::Revenue && target_type == AccountType::Expense
            }
            GaapViolationType::CashToRevenue => {
                source_type == AccountType::Asset && target_type == AccountType::Revenue
            }
            GaapViolationType::ExpenseToAsset => {
                source_type == AccountType::Expense && target_type == AccountType::Asset
            }
            GaapViolationType::LiabilityToRevenue => {
                source_type == AccountType::Liability && target_type == AccountType::Revenue
            }
            _ => false, // Other violations need more context
        }
    }
}

/// A GAAP violation rule for detection.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct GaapViolationRule {
    /// Rule identifier
    pub rule_id: u32,
    /// Violation type
    pub violation_type: GaapViolationType,
    /// Source account type (if applicable)
    pub source_type: Option<AccountType>,
    /// Target account type (if applicable)
    pub target_type: Option<AccountType>,
    /// Severity level
    pub severity: ViolationSeverity,
    /// Minimum amount to trigger (0 = any)
    pub min_amount: f64,
    /// Rule name hash (for display lookup)
    pub rule_name_hash: u64,
}

/// A detected GAAP violation instance.
#[derive(Debug, Clone)]
pub struct GaapViolation {
    /// Unique identifier
    pub id: Uuid,
    /// Type of violation
    pub violation_type: GaapViolationType,
    /// Severity
    pub severity: ViolationSeverity,
    /// Source account index
    pub source_account: u16,
    /// Target account index
    pub target_account: u16,
    /// Amount involved
    pub amount: Decimal128,
    /// Journal entry that caused the violation
    pub journal_entry_id: Uuid,
    /// When detected
    pub detected_at: HybridTimestamp,
    /// Description
    pub description: String,
}

impl GaapViolation {
    pub fn new(
        violation_type: GaapViolationType,
        source: u16,
        target: u16,
        amount: Decimal128,
        journal_entry_id: Uuid,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            violation_type,
            severity: violation_type.default_severity(),
            source_account: source,
            target_account: target,
            amount,
            journal_entry_id,
            detected_at: HybridTimestamp::now(),
            description: violation_type.description().to_string(),
        }
    }
}

/// Benford's Law expected first-digit distribution.
pub const BENFORD_EXPECTED: [f64; 9] = [
    0.301, // 1
    0.176, // 2
    0.125, // 3
    0.097, // 4
    0.079, // 5
    0.067, // 6
    0.058, // 7
    0.051, // 8
    0.046, // 9
];

/// Calculate chi-squared statistic for Benford's Law compliance.
pub fn benford_chi_squared(observed_counts: &[u32; 9], total: u32) -> f64 {
    if total == 0 {
        return 0.0;
    }

    let mut chi_sq = 0.0;
    for i in 0..9 {
        let expected = BENFORD_EXPECTED[i] * total as f64;
        let observed = observed_counts[i] as f64;
        if expected > 0.0 {
            chi_sq += (observed - expected).powi(2) / expected;
        }
    }
    chi_sq
}

/// Critical value for chi-squared with 8 degrees of freedom (p=0.05).
pub const BENFORD_CHI_SQ_CRITICAL: f64 = 15.507;

/// Check if a chi-squared value indicates Benford violation.
pub fn is_benford_violation(chi_squared: f64) -> bool {
    chi_squared > BENFORD_CHI_SQ_CRITICAL
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benford_perfect_distribution() {
        // Perfectly matching Benford's Law
        let observed = [301, 176, 125, 97, 79, 67, 58, 51, 46];
        let chi_sq = benford_chi_squared(&observed, 1000);
        assert!(chi_sq < BENFORD_CHI_SQ_CRITICAL);
    }

    #[test]
    fn test_benford_uniform_violation() {
        // Uniform distribution (clearly not Benford)
        let observed = [111, 111, 111, 111, 111, 111, 111, 111, 112];
        let chi_sq = benford_chi_squared(&observed, 1000);
        assert!(is_benford_violation(chi_sq));
    }

    #[test]
    fn test_gaap_violation_matching() {
        assert!(GaapViolationType::RevenueToExpense.matches(
            AccountType::Revenue,
            AccountType::Expense
        ));
        assert!(!GaapViolationType::RevenueToExpense.matches(
            AccountType::Asset,
            AccountType::Expense
        ));
    }

    #[test]
    fn test_fraud_pattern_accounts() {
        let mut pattern = FraudPattern::new(FraudPatternType::CircularFlow);
        pattern.add_account(0);
        pattern.add_account(1);
        pattern.add_account(2);

        let accounts = pattern.get_involved_accounts();
        assert_eq!(accounts, vec![0, 1, 2]);
    }
}
