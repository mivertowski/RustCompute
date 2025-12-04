//! Account node representation for the accounting network graph.
//!
//! Each account is a node in the directed graph, with edges representing
//! monetary flows between accounts.

use super::Decimal128;
use rkyv::{Archive, Deserialize, Serialize};
use uuid::Uuid;

/// The five fundamental account types in double-entry bookkeeping.
/// Each has a "normal balance" side (debit or credit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[repr(u8)]
pub enum AccountType {
    /// Assets: What the company owns (Cash, Inventory, Equipment)
    /// Normal balance: Debit (increases with debits)
    Asset = 0,

    /// Liabilities: What the company owes (Accounts Payable, Loans)
    /// Normal balance: Credit (increases with credits)
    Liability = 1,

    /// Equity: Owner's stake (Common Stock, Retained Earnings)
    /// Normal balance: Credit
    Equity = 2,

    /// Revenue: Income from operations (Sales, Service Revenue)
    /// Normal balance: Credit
    Revenue = 3,

    /// Expense: Costs of operations (Salaries, Rent, Utilities)
    /// Normal balance: Debit
    Expense = 4,

    /// Contra accounts: Offset their parent account type
    /// (Accumulated Depreciation, Sales Returns)
    Contra = 5,
}

impl AccountType {
    /// Returns the normal balance side for this account type.
    pub fn normal_balance(&self) -> BalanceSide {
        match self {
            AccountType::Asset | AccountType::Expense => BalanceSide::Debit,
            AccountType::Liability | AccountType::Equity | AccountType::Revenue => {
                BalanceSide::Credit
            }
            AccountType::Contra => BalanceSide::Credit, // Usually contra-asset
        }
    }

    /// Returns true if debits increase this account's balance.
    pub fn debit_increases(&self) -> bool {
        matches!(self, AccountType::Asset | AccountType::Expense)
    }

    /// Returns a color for visualization.
    pub fn color(&self) -> [u8; 3] {
        match self {
            AccountType::Asset => [100, 149, 237],   // Cornflower blue
            AccountType::Liability => [255, 99, 71], // Tomato red
            AccountType::Equity => [50, 205, 50],    // Lime green
            AccountType::Revenue => [255, 215, 0],   // Gold
            AccountType::Expense => [255, 140, 0],   // Dark orange
            AccountType::Contra => [148, 0, 211],    // Dark violet
        }
    }

    /// Returns a display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            AccountType::Asset => "Asset",
            AccountType::Liability => "Liability",
            AccountType::Equity => "Equity",
            AccountType::Revenue => "Revenue",
            AccountType::Expense => "Expense",
            AccountType::Contra => "Contra",
        }
    }

    /// Returns an icon character for visualization.
    pub fn icon(&self) -> char {
        match self {
            AccountType::Asset => '●',     // Solid circle
            AccountType::Liability => '○', // Empty circle
            AccountType::Equity => '▣',    // Square with fill
            AccountType::Revenue => '◆',   // Diamond
            AccountType::Expense => '◇',   // Empty diamond
            AccountType::Contra => '◐',    // Half-filled circle
        }
    }
}

/// Which side of the accounting equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[repr(u8)]
pub enum BalanceSide {
    /// Debit side (left side of T-account).
    Debit = 0,
    /// Credit side (right side of T-account).
    Credit = 1,
}

/// Semantic flags for account behavior analysis.
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct AccountSemantics {
    /// Bit flags for semantic roles
    pub flags: u32,
    /// Typical transactions per month
    pub typical_frequency: f32,
    /// Log-scale average transaction size (0-100)
    pub avg_amount_scale: f32,
}

impl AccountSemantics {
    /// Flag: Cash or cash equivalent account.
    pub const IS_CASH: u32 = 1 << 0;
    /// Flag: Accounts receivable.
    pub const IS_RECEIVABLE: u32 = 1 << 1;
    /// Flag: Accounts payable.
    pub const IS_PAYABLE: u32 = 1 << 2;
    /// Flag: Revenue account.
    pub const IS_REVENUE: u32 = 1 << 3;
    /// Flag: Expense account.
    pub const IS_EXPENSE: u32 = 1 << 4;
    /// Flag: Inventory account.
    pub const IS_INVENTORY: u32 = 1 << 5;
    /// Flag: VAT/sales tax account.
    pub const IS_VAT: u32 = 1 << 6;
    /// Flag: Suspense/clearing account.
    pub const IS_SUSPENSE: u32 = 1 << 7;
    /// Flag: Intercompany account.
    pub const IS_INTERCOMPANY: u32 = 1 << 8;
    /// Flag: Depreciation account.
    pub const IS_DEPRECIATION: u32 = 1 << 9;
    /// Flag: Cost of goods sold account.
    pub const IS_COGS: u32 = 1 << 10;
    /// Flag: Payroll account.
    pub const IS_PAYROLL: u32 = 1 << 11;

    /// Check if this is a cash account.
    pub fn is_cash(&self) -> bool {
        self.flags & Self::IS_CASH != 0
    }
    /// Check if this is a suspense account.
    pub fn is_suspense(&self) -> bool {
        self.flags & Self::IS_SUSPENSE != 0
    }
    /// Check if this is a revenue account.
    pub fn is_revenue(&self) -> bool {
        self.flags & Self::IS_REVENUE != 0
    }
    /// Check if this is an expense account.
    pub fn is_expense(&self) -> bool {
        self.flags & Self::IS_EXPENSE != 0
    }
}

/// A single account node in the accounting network.
/// GPU-aligned to 128 bytes for efficient memory access.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[repr(C, align(128))]
pub struct AccountNode {
    // === Identity (32 bytes) ===
    /// Unique identifier
    pub id: Uuid,
    /// Account code hash (for fast lookup)
    pub code_hash: u64,
    /// Index in the network's account array (0-255 for GPU)
    pub index: u16,
    /// Account type classification
    pub account_type: AccountType,
    /// Account class ID (for hierarchy)
    pub class_id: u8,
    /// Account subclass ID
    pub subclass_id: u8,
    /// Padding for alignment
    pub _pad1: [u8; 3],

    // === Account metadata (variable, stored separately) ===
    // code: String - stored in auxiliary structure
    // name: String - stored in auxiliary structure

    // === Balances (32 bytes) ===
    /// Balance at period start
    pub opening_balance: Decimal128,
    /// Balance at period end
    pub closing_balance: Decimal128,

    // === Activity (32 bytes) ===
    /// Total debit activity in period
    pub total_debits: Decimal128,
    /// Total credit activity in period
    pub total_credits: Decimal128,

    // === Graph metrics (16 bytes) ===
    /// Number of incoming edges (accounts that flow TO this account)
    pub in_degree: u16,
    /// Number of outgoing edges (accounts that flow FROM this account)
    pub out_degree: u16,
    /// Betweenness centrality (0.0 - 1.0)
    pub betweenness_centrality: f32,
    /// PageRank score
    pub pagerank: f32,
    /// Clustering coefficient
    pub clustering_coefficient: f32,

    // === Analysis results (12 bytes) ===
    /// Suspense account confidence (0.0 - 1.0)
    pub suspense_score: f32,
    /// Risk score from anomaly detection
    pub risk_score: f32,
    /// Transaction count in period
    pub transaction_count: u32,

    // === Flags (4 bytes) ===
    /// Account property flags.
    pub flags: AccountFlags,
}

/// Bit flags for account properties.
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(transparent)]
pub struct AccountFlags(pub u32);

impl AccountFlags {
    /// Flag: Identified as a suspense account.
    pub const IS_SUSPENSE_ACCOUNT: u32 = 1 << 0;
    /// Flag: Cash or cash equivalent.
    pub const IS_CASH_ACCOUNT: u32 = 1 << 1;
    /// Flag: Revenue account.
    pub const IS_REVENUE_ACCOUNT: u32 = 1 << 2;
    /// Flag: Expense account.
    pub const IS_EXPENSE_ACCOUNT: u32 = 1 << 3;
    /// Flag: Intercompany account.
    pub const IS_INTERCOMPANY: u32 = 1 << 4;
    /// Flag: Flagged for audit review.
    pub const FLAGGED_FOR_AUDIT: u32 = 1 << 5;
    /// Flag: Has GAAP violation.
    pub const HAS_GAAP_VIOLATION: u32 = 1 << 6;
    /// Flag: Involved in fraud pattern.
    pub const HAS_FRAUD_PATTERN: u32 = 1 << 7;
    /// Flag: Dormant account (no recent activity).
    pub const IS_DORMANT: u32 = 1 << 8;
    /// Flag: Has detected anomaly.
    pub const HAS_ANOMALY: u32 = 1 << 9;

    /// Create a new empty flags instance.
    pub fn new() -> Self {
        Self(0)
    }

    /// Set a flag.
    pub fn set(&mut self, flag: u32) {
        self.0 |= flag;
    }

    /// Clear a flag.
    pub fn clear(&mut self, flag: u32) {
        self.0 &= !flag;
    }

    /// Check if a flag is set.
    pub fn has(&self, flag: u32) -> bool {
        self.0 & flag != 0
    }
}

impl AccountNode {
    /// Create a new account node with default values.
    pub fn new(id: Uuid, account_type: AccountType, index: u16) -> Self {
        Self {
            id,
            code_hash: 0,
            index,
            account_type,
            class_id: 0,
            subclass_id: 0,
            _pad1: [0; 3],
            opening_balance: Decimal128::ZERO,
            closing_balance: Decimal128::ZERO,
            total_debits: Decimal128::ZERO,
            total_credits: Decimal128::ZERO,
            in_degree: 0,
            out_degree: 0,
            betweenness_centrality: 0.0,
            pagerank: 0.0,
            clustering_coefficient: 0.0,
            suspense_score: 0.0,
            risk_score: 0.0,
            transaction_count: 0,
            flags: AccountFlags::new(),
        }
    }

    /// Calculate the net change for this period.
    pub fn net_change(&self) -> Decimal128 {
        self.closing_balance - self.opening_balance
    }

    /// Calculate total activity (debits + credits).
    pub fn total_activity(&self) -> Decimal128 {
        Decimal128::from_f64(self.total_debits.to_f64().abs() + self.total_credits.to_f64().abs())
    }

    /// Calculate balance ratio (balance / activity) - low = suspense indicator.
    pub fn balance_ratio(&self) -> f64 {
        let activity = self.total_activity().to_f64();
        if activity > 0.0 {
            self.closing_balance.to_f64().abs() / activity
        } else {
            1.0 // No activity = not suspense
        }
    }

    /// Check if this account has high centrality (hub in the network).
    pub fn is_hub(&self) -> bool {
        self.in_degree + self.out_degree > 10 || self.betweenness_centrality > 0.1
    }
}

/// Auxiliary structure for account string data.
#[derive(Debug, Clone)]
pub struct AccountMetadata {
    /// Account code (e.g., "1100")
    pub code: String,
    /// Account name (e.g., "Cash and Cash Equivalents")
    pub name: String,
    /// Description
    pub description: String,
    /// Parent account ID for hierarchy
    pub parent_id: Option<Uuid>,
    /// Semantic properties
    pub semantics: AccountSemantics,
}

impl AccountMetadata {
    /// Create new account metadata with code and name.
    pub fn new(code: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            name: name.into(),
            description: String::new(),
            parent_id: None,
            semantics: AccountSemantics::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_type_normal_balance() {
        assert_eq!(AccountType::Asset.normal_balance(), BalanceSide::Debit);
        assert_eq!(AccountType::Liability.normal_balance(), BalanceSide::Credit);
        assert_eq!(AccountType::Revenue.normal_balance(), BalanceSide::Credit);
        assert_eq!(AccountType::Expense.normal_balance(), BalanceSide::Debit);
    }

    #[test]
    fn test_decimal128_arithmetic() {
        let a = Decimal128::from_f64(100.50);
        let b = Decimal128::from_f64(25.25);
        let sum = a + b;
        assert!((sum.to_f64() - 125.75).abs() < 0.01);
    }

    #[test]
    fn test_account_node_size() {
        // Ensure GPU alignment (may be larger with rkyv metadata)
        let size = std::mem::size_of::<AccountNode>();
        assert!(
            size >= 128,
            "AccountNode should be at least 128 bytes, got {}",
            size
        );
        assert!(
            size.is_multiple_of(128),
            "AccountNode should be 128-byte aligned, got {}",
            size
        );
    }
}
