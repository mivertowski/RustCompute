//! Journal entry structures for double-entry bookkeeping.
//!
//! A journal entry records a business transaction with balanced debits and credits.
//! These entries are transformed into network flows using Methods A-E.

use super::{AccountType, Decimal128, HybridTimestamp};
use rkyv::{Archive, Deserialize, Serialize};
use uuid::Uuid;

/// The transformation method used to convert a journal entry to flows.
/// Based on Ivertowski et al. (2024) methodology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[repr(u8)]
pub enum SolvingMethod {
    /// Method A: 1-to-1 mapping (60.68% of entries)
    /// Single debit → single credit, confidence = 1.0
    MethodA = 0,

    /// Method B: n-to-n bijective mapping (16.63% of entries)
    /// Equal debit/credit counts, match by amount
    /// Confidence = 1.0 (distinct) or 1/n (duplicates)
    MethodB = 1,

    /// Method C: n-to-m partition (11% of entries)
    /// Unequal distribution, subset sum or VAT disaggregation
    /// Confidence varies by match quality
    MethodC = 2,

    /// Method D: Higher aggregate (11% of entries)
    /// Account class level matching when detail fails
    /// Confidence = 1.0 at aggregate level
    MethodD = 3,

    /// Method E: Decomposition with shadow bookings (0.76% of entries)
    /// Last resort - greedy allocation
    /// Confidence = 1/decomposition_steps
    MethodE = 4,

    /// Not yet processed
    Pending = 255,
}

impl SolvingMethod {
    /// Expected percentage of entries using this method.
    pub fn expected_ratio(&self) -> f64 {
        match self {
            SolvingMethod::MethodA => 0.6068,
            SolvingMethod::MethodB => 0.1663,
            SolvingMethod::MethodC => 0.11,
            SolvingMethod::MethodD => 0.11,
            SolvingMethod::MethodE => 0.0076,
            SolvingMethod::Pending => 0.0,
        }
    }

    /// Base confidence for this method.
    pub fn base_confidence(&self) -> f32 {
        match self {
            SolvingMethod::MethodA => 1.0,
            SolvingMethod::MethodB => 1.0, // May be reduced for duplicates
            SolvingMethod::MethodC => 0.85,
            SolvingMethod::MethodD => 1.0,
            SolvingMethod::MethodE => 0.5, // Divided by steps
            SolvingMethod::Pending => 0.0,
        }
    }

    /// Display name for UI.
    pub fn display_name(&self) -> &'static str {
        match self {
            SolvingMethod::MethodA => "A: 1-to-1",
            SolvingMethod::MethodB => "B: n-to-n",
            SolvingMethod::MethodC => "C: n-to-m",
            SolvingMethod::MethodD => "D: Aggregate",
            SolvingMethod::MethodE => "E: Decompose",
            SolvingMethod::Pending => "Pending",
        }
    }

    /// Color for visualization.
    pub fn color(&self) -> [u8; 3] {
        match self {
            SolvingMethod::MethodA => [0, 200, 83],    // Green - best
            SolvingMethod::MethodB => [100, 181, 246], // Blue
            SolvingMethod::MethodC => [255, 193, 7],   // Amber
            SolvingMethod::MethodD => [255, 152, 0],   // Orange
            SolvingMethod::MethodE => [244, 67, 54],   // Red - worst
            SolvingMethod::Pending => [158, 158, 158], // Gray
        }
    }
}

/// A journal entry header (ISO 21378:2019 compliant).
/// GPU-aligned to 128 bytes.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[repr(C, align(128))]
pub struct JournalEntry {
    // === Identity (32 bytes) ===
    /// Unique entry identifier
    pub id: Uuid,
    /// Entity (company) ID
    pub entity_id: Uuid,

    // === Document reference (16 bytes) ===
    /// Document number hash
    pub document_number_hash: u64,
    /// Source system identifier
    pub source_system_id: u32,
    /// Batch number for bulk imports
    pub batch_id: u32,

    // === Temporal (16 bytes) ===
    /// When the entry was posted
    pub posting_date: HybridTimestamp,

    // === Line item counts (8 bytes) ===
    /// Total line items
    pub line_count: u16,
    /// Number of debit lines
    pub debit_line_count: u16,
    /// Number of credit lines
    pub credit_line_count: u16,
    /// Index of first line in the line array
    pub first_line_index: u16,

    // === Amounts (32 bytes) ===
    /// Sum of all debit amounts
    pub total_debits: Decimal128,
    /// Sum of all credit amounts (should equal total_debits)
    pub total_credits: Decimal128,

    // === Transformation (8 bytes) ===
    /// Method used to transform to flows
    pub solving_method: SolvingMethod,
    /// Average confidence across generated flows
    pub average_confidence: f32,
    /// Number of flows generated
    pub flow_count: u16,
    /// Padding
    pub _pad: u8,

    // === Flags (4 bytes) ===
    /// Entry property flags.
    pub flags: JournalEntryFlags,

    // === Reserved (12 bytes) ===
    /// Reserved for future use.
    pub _reserved: [u8; 12],
}

/// Bit flags for journal entry properties.
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(transparent)]
pub struct JournalEntryFlags(pub u32);

impl JournalEntryFlags {
    /// Flag: Entry is balanced (debits = credits).
    pub const IS_BALANCED: u32 = 1 << 0;
    /// Flag: Entry has been transformed to flows.
    pub const IS_TRANSFORMED: u32 = 1 << 1;
    /// Flag: Contains decomposed/shadow values.
    pub const HAS_DECOMPOSED_VALUES: u32 = 1 << 2;
    /// Flag: Uses higher aggregate matching.
    pub const USES_HIGHER_AGGREGATE: u32 = 1 << 3;
    /// Flag: Flagged for audit review.
    pub const FLAGGED_FOR_AUDIT: u32 = 1 << 4;
    /// Flag: Reversing entry.
    pub const IS_REVERSING: u32 = 1 << 5;
    /// Flag: Recurring entry.
    pub const IS_RECURRING: u32 = 1 << 6;
    /// Flag: Adjustment entry.
    pub const IS_ADJUSTMENT: u32 = 1 << 7;
    /// Flag: Contains VAT lines.
    pub const HAS_VAT: u32 = 1 << 8;
    /// Flag: Intercompany transaction.
    pub const IS_INTERCOMPANY: u32 = 1 << 9;

    /// Create new flags (balanced by default).
    pub fn new() -> Self {
        Self(Self::IS_BALANCED) // Entries should be balanced by default
    }

    /// Check if entry is balanced.
    pub fn is_balanced(&self) -> bool {
        self.0 & Self::IS_BALANCED != 0
    }
    /// Check if entry has been transformed.
    pub fn is_transformed(&self) -> bool {
        self.0 & Self::IS_TRANSFORMED != 0
    }
    /// Check if entry is flagged for audit.
    pub fn flagged_for_audit(&self) -> bool {
        self.0 & Self::FLAGGED_FOR_AUDIT != 0
    }
}

/// A single line item in a journal entry.
/// GPU-aligned to 64 bytes.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct JournalLineItem {
    // === Identity (20 bytes) ===
    /// Line item ID
    pub id: Uuid,
    /// Parent journal entry ID reference index
    pub journal_entry_index: u32,

    // === Account reference (8 bytes) ===
    /// Account ID (references AccountNode)
    pub account_index: u16,
    /// Line number within entry (1-based)
    pub line_number: u16,
    /// Debit (0) or Credit (1)
    pub line_type: LineType,
    /// Padding
    pub _pad1: [u8; 3],

    // === Amount (16 bytes) ===
    /// Monetary amount (positive for debit, negative for credit by convention)
    pub amount: Decimal128,

    // === Confidence and matching (8 bytes) ===
    /// Confidence score (1.0 = original, <1.0 = estimated/decomposed)
    pub confidence: f32,
    /// Index of matched line (for Method A/B/C), u16::MAX if unmatched
    pub matched_line_index: u16,
    /// Flags
    pub flags: LineItemFlags,
    /// Padding
    pub _pad2: u8,

    // === Reserved (12 bytes) ===
    /// Reserved for future use.
    pub _reserved: [u8; 12],
}

/// Line type: Debit or Credit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[repr(u8)]
pub enum LineType {
    /// Debit line (left side of entry).
    Debit = 0,
    /// Credit line (right side of entry).
    Credit = 1,
}

impl LineType {
    /// Check if this is a debit line.
    pub fn is_debit(&self) -> bool {
        matches!(self, LineType::Debit)
    }
    /// Check if this is a credit line.
    pub fn is_credit(&self) -> bool {
        matches!(self, LineType::Credit)
    }
}

/// Bit flags for line item properties.
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(transparent)]
pub struct LineItemFlags(pub u8);

impl LineItemFlags {
    /// Flag: Shadow booking (Method E decomposition).
    pub const IS_SHADOW_BOOKING: u8 = 1 << 0;
    /// Flag: Higher aggregate line (Method D).
    pub const IS_HIGHER_AGGREGATE: u8 = 1 << 1;
    /// Flag: VAT/tax line.
    pub const IS_VAT_LINE: u8 = 1 << 2;
    /// Flag: Rounding adjustment line.
    pub const IS_ROUNDING_ADJUSTMENT: u8 = 1 << 3;
    /// Flag: Line has been matched.
    pub const IS_MATCHED: u8 = 1 << 4;
}

impl JournalEntry {
    /// Create a new journal entry.
    pub fn new(id: Uuid, entity_id: Uuid, posting_date: HybridTimestamp) -> Self {
        Self {
            id,
            entity_id,
            document_number_hash: 0,
            source_system_id: 0,
            batch_id: 0,
            posting_date,
            line_count: 0,
            debit_line_count: 0,
            credit_line_count: 0,
            first_line_index: 0,
            total_debits: Decimal128::ZERO,
            total_credits: Decimal128::ZERO,
            solving_method: SolvingMethod::Pending,
            average_confidence: 0.0,
            flow_count: 0,
            _pad: 0,
            flags: JournalEntryFlags::new(),
            _reserved: [0; 12],
        }
    }

    /// Check if the entry is balanced (debits = credits).
    pub fn is_balanced(&self) -> bool {
        (self.total_debits.to_f64() - self.total_credits.to_f64()).abs() < 0.01
    }

    /// Determine which solving method should be used.
    pub fn determine_method(&self) -> SolvingMethod {
        if self.debit_line_count == 1 && self.credit_line_count == 1 {
            SolvingMethod::MethodA
        } else if self.debit_line_count == self.credit_line_count {
            SolvingMethod::MethodB
        } else {
            SolvingMethod::MethodC
        }
    }
}

impl JournalLineItem {
    /// Create a new debit line.
    pub fn debit(account_index: u16, amount: Decimal128, line_number: u16) -> Self {
        Self {
            id: Uuid::new_v4(),
            journal_entry_index: 0,
            account_index,
            line_number,
            line_type: LineType::Debit,
            _pad1: [0; 3],
            amount,
            confidence: 1.0,
            matched_line_index: u16::MAX,
            flags: LineItemFlags(0),
            _pad2: 0,
            _reserved: [0; 12],
        }
    }

    /// Create a new credit line.
    pub fn credit(account_index: u16, amount: Decimal128, line_number: u16) -> Self {
        Self {
            id: Uuid::new_v4(),
            journal_entry_index: 0,
            account_index,
            line_number,
            line_type: LineType::Credit,
            _pad1: [0; 3],
            amount,
            confidence: 1.0,
            matched_line_index: u16::MAX,
            flags: LineItemFlags(0),
            _pad2: 0,
            _reserved: [0; 12],
        }
    }

    /// Check if this is a debit line.
    pub fn is_debit(&self) -> bool {
        self.line_type.is_debit()
    }

    /// Check if this is a credit line.
    pub fn is_credit(&self) -> bool {
        self.line_type.is_credit()
    }

    /// Check if this line has been matched to another.
    pub fn is_matched(&self) -> bool {
        self.matched_line_index != u16::MAX
    }
}

/// Common booking patterns for pattern recognition and confidence boosting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BookingPatternType {
    /// Cash receipt from customer (DR Cash, CR A/R or Revenue)
    CashReceipt = 0,
    /// Cash payment to vendor (DR A/P or Expense, CR Cash)
    CashPayment = 1,
    /// Sales transaction (DR A/R, CR Revenue, possibly CR VAT)
    SalesRevenue = 2,
    /// Purchase transaction (DR Expense/Inventory, CR A/P)
    Purchase = 3,
    /// Payroll entry (DR Salary Expense, CR Cash/Payroll Payable)
    Payroll = 4,
    /// Depreciation (DR Depreciation Expense, CR Accumulated Depreciation)
    Depreciation = 5,
    /// Accrual entry (DR Expense, CR Accrued Liability)
    Accrual = 6,
    /// Reversal entry (opposite of original)
    Reversal = 7,
    /// Intercompany transfer
    Intercompany = 8,
    /// VAT settlement
    VatSettlement = 9,
    /// Bank reconciliation
    BankReconciliation = 10,
    /// Unknown/other pattern
    Unknown = 255,
}

impl BookingPatternType {
    /// Expected account type for debit side.
    pub fn expected_debit_type(&self) -> Option<AccountType> {
        match self {
            BookingPatternType::CashReceipt => Some(AccountType::Asset), // Cash
            BookingPatternType::CashPayment => Some(AccountType::Liability), // A/P
            BookingPatternType::SalesRevenue => Some(AccountType::Asset), // A/R
            BookingPatternType::Purchase => Some(AccountType::Expense),
            BookingPatternType::Payroll => Some(AccountType::Expense),
            BookingPatternType::Depreciation => Some(AccountType::Expense),
            BookingPatternType::Accrual => Some(AccountType::Expense),
            _ => None,
        }
    }

    /// Expected account type for credit side.
    pub fn expected_credit_type(&self) -> Option<AccountType> {
        match self {
            BookingPatternType::CashReceipt => Some(AccountType::Revenue),
            BookingPatternType::CashPayment => Some(AccountType::Asset), // Cash
            BookingPatternType::SalesRevenue => Some(AccountType::Revenue),
            BookingPatternType::Purchase => Some(AccountType::Liability), // A/P
            BookingPatternType::Payroll => Some(AccountType::Asset),      // Cash
            BookingPatternType::Depreciation => Some(AccountType::Contra), // Accum Depr
            BookingPatternType::Accrual => Some(AccountType::Liability),
            _ => None,
        }
    }

    /// Confidence boost when pattern is matched.
    pub fn confidence_boost(&self) -> f32 {
        match self {
            BookingPatternType::CashReceipt => 0.20,
            BookingPatternType::CashPayment => 0.20,
            BookingPatternType::SalesRevenue => 0.15,
            BookingPatternType::Purchase => 0.15,
            BookingPatternType::Payroll => 0.25,
            BookingPatternType::Depreciation => 0.25,
            _ => 0.10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_journal_entry_size() {
        let size = std::mem::size_of::<JournalEntry>();
        assert!(
            size >= 128,
            "JournalEntry should be at least 128 bytes, got {}",
            size
        );
        assert!(
            size % 128 == 0,
            "JournalEntry should be 128-byte aligned, got {}",
            size
        );
    }

    #[test]
    fn test_line_item_size() {
        let size = std::mem::size_of::<JournalLineItem>();
        assert!(
            size >= 64,
            "JournalLineItem should be at least 64 bytes, got {}",
            size
        );
        assert!(
            size % 64 == 0,
            "JournalLineItem should be 64-byte aligned, got {}",
            size
        );
    }

    #[test]
    fn test_method_determination() {
        let mut entry = JournalEntry::new(Uuid::new_v4(), Uuid::new_v4(), HybridTimestamp::now());

        // 1 debit, 1 credit -> Method A
        entry.debit_line_count = 1;
        entry.credit_line_count = 1;
        assert_eq!(entry.determine_method(), SolvingMethod::MethodA);

        // 3 debits, 3 credits -> Method B
        entry.debit_line_count = 3;
        entry.credit_line_count = 3;
        assert_eq!(entry.determine_method(), SolvingMethod::MethodB);

        // 2 debits, 5 credits -> Method C
        entry.debit_line_count = 2;
        entry.credit_line_count = 5;
        assert_eq!(entry.determine_method(), SolvingMethod::MethodC);
    }
}
