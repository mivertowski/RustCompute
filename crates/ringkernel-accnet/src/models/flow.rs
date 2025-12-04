//! Transaction flow representation - edges in the accounting network graph.
//!
//! A flow represents monetary movement from one account to another,
//! derived from journal entry transformation.

use super::{Decimal128, HybridTimestamp, SolvingMethod};
use rkyv::{Archive, Deserialize, Serialize};
use uuid::Uuid;

/// A directed edge in the accounting network representing monetary flow.
/// GPU-aligned to 64 bytes.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct TransactionFlow {
    // === Edge endpoints (4 bytes) ===
    /// Source account index (debited account)
    pub source_account_index: u16,
    /// Target account index (credited account)
    pub target_account_index: u16,

    // === Amount (16 bytes) ===
    /// Monetary amount transferred
    pub amount: Decimal128,

    // === Provenance (20 bytes) ===
    /// Original journal entry ID
    pub journal_entry_id: Uuid,
    /// Line item index in the journal entry
    pub debit_line_index: u16,
    /// Corresponding credit line index
    pub credit_line_index: u16,

    // === Temporal (8 bytes) ===
    /// When this flow was recorded
    pub timestamp: HybridTimestamp,

    // === Quality metrics (8 bytes) ===
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Transformation method used
    pub method_used: SolvingMethod,
    /// Flags
    pub flags: FlowFlags,
    /// Padding
    pub _pad: [u8; 2],

    // === Reserved (8 bytes) ===
    /// Reserved for future use.
    pub _reserved: [u8; 8],
}

/// Bit flags for flow properties.
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(transparent)]
pub struct FlowFlags(pub u8);

impl FlowFlags {
    /// Flag: Flow derived from shadow bookings (Method E).
    pub const HAS_SHADOW_BOOKINGS: u8 = 1 << 0;
    /// Flag: Flow uses higher aggregate matching (Method D).
    pub const USES_HIGHER_AGGREGATE: u8 = 1 << 1;
    /// Flag: Flow flagged for audit review.
    pub const FLAGGED_FOR_AUDIT: u8 = 1 << 2;
    /// Flag: Flow is a reversal of another transaction.
    pub const IS_REVERSAL: u8 = 1 << 3;
    /// Flag: Flow is part of a circular pattern.
    pub const IS_CIRCULAR: u8 = 1 << 4;
    /// Flag: Flow detected as anomalous.
    pub const IS_ANOMALOUS: u8 = 1 << 5;
    /// Flag: Flow violates GAAP rules.
    pub const IS_GAAP_VIOLATION: u8 = 1 << 6;
    /// Flag: Flow is part of a fraud pattern.
    pub const IS_FRAUD_PATTERN: u8 = 1 << 7;

    /// Create new empty flags.
    pub fn new() -> Self {
        Self(0)
    }

    /// Check if a flag is set.
    pub fn has(&self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    /// Set a flag.
    pub fn set(&mut self, flag: u8) {
        self.0 |= flag;
    }

    /// Clear a flag.
    pub fn clear(&mut self, flag: u8) {
        self.0 &= !flag;
    }
}

impl TransactionFlow {
    /// Create a new transaction flow.
    pub fn new(
        source: u16,
        target: u16,
        amount: Decimal128,
        journal_entry_id: Uuid,
        timestamp: HybridTimestamp,
    ) -> Self {
        Self {
            source_account_index: source,
            target_account_index: target,
            amount,
            journal_entry_id,
            debit_line_index: 0,
            credit_line_index: 0,
            timestamp,
            confidence: 1.0,
            method_used: SolvingMethod::MethodA,
            flags: FlowFlags::new(),
            _pad: [0; 2],
            _reserved: [0; 8],
        }
    }

    /// Create a flow with full provenance.
    pub fn with_provenance(
        source: u16,
        target: u16,
        amount: Decimal128,
        journal_entry_id: Uuid,
        debit_line_index: u16,
        credit_line_index: u16,
        timestamp: HybridTimestamp,
        method: SolvingMethod,
        confidence: f32,
    ) -> Self {
        Self {
            source_account_index: source,
            target_account_index: target,
            amount,
            journal_entry_id,
            debit_line_index,
            credit_line_index,
            timestamp,
            confidence,
            method_used: method,
            flags: FlowFlags::new(),
            _pad: [0; 2],
            _reserved: [0; 8],
        }
    }

    /// Check if this flow is a self-loop (same source and target).
    pub fn is_self_loop(&self) -> bool {
        self.source_account_index == self.target_account_index
    }

    /// Check if this flow has high confidence.
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.9
    }

    /// Check if this flow is flagged as anomalous.
    pub fn is_anomalous(&self) -> bool {
        self.flags.has(FlowFlags::IS_ANOMALOUS)
            || self.flags.has(FlowFlags::IS_CIRCULAR)
            || self.flags.has(FlowFlags::IS_FRAUD_PATTERN)
            || self.flags.has(FlowFlags::IS_GAAP_VIOLATION)
    }
}

/// Aggregated flow statistics between two accounts.
/// Used for visualization edge weights.
#[derive(Debug, Clone, Default)]
pub struct AggregatedFlow {
    /// Source account index
    pub source: u16,
    /// Target account index
    pub target: u16,
    /// Total amount transferred
    pub total_amount: f64,
    /// Number of individual transactions
    pub transaction_count: u32,
    /// Average confidence across transactions
    pub avg_confidence: f32,
    /// Earliest transaction
    pub first_timestamp: HybridTimestamp,
    /// Latest transaction
    pub last_timestamp: HybridTimestamp,
    /// Method distribution
    pub method_counts: [u32; 5], // Methods A-E
    /// Number of flagged transactions
    pub flagged_count: u32,
}

impl AggregatedFlow {
    /// Create a new aggregated flow between two accounts.
    pub fn new(source: u16, target: u16) -> Self {
        Self {
            source,
            target,
            ..Default::default()
        }
    }

    /// Add a flow to this aggregation.
    pub fn add(&mut self, flow: &TransactionFlow) {
        self.total_amount += flow.amount.to_f64();
        self.transaction_count += 1;

        // Update running average confidence
        let n = self.transaction_count as f32;
        self.avg_confidence = self.avg_confidence * (n - 1.0) / n + flow.confidence / n;

        // Update timestamps
        if self.transaction_count == 1 {
            self.first_timestamp = flow.timestamp;
            self.last_timestamp = flow.timestamp;
        } else {
            if flow.timestamp < self.first_timestamp {
                self.first_timestamp = flow.timestamp;
            }
            if flow.timestamp > self.last_timestamp {
                self.last_timestamp = flow.timestamp;
            }
        }

        // Update method distribution
        let method_idx = flow.method_used as usize;
        if method_idx < 5 {
            self.method_counts[method_idx] += 1;
        }

        // Count flagged
        if flow.is_anomalous() {
            self.flagged_count += 1;
        }
    }

    /// Get the dominant solving method for this flow.
    pub fn dominant_method(&self) -> SolvingMethod {
        let max_idx = self
            .method_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        match max_idx {
            0 => SolvingMethod::MethodA,
            1 => SolvingMethod::MethodB,
            2 => SolvingMethod::MethodC,
            3 => SolvingMethod::MethodD,
            4 => SolvingMethod::MethodE,
            _ => SolvingMethod::MethodA,
        }
    }

    /// Calculate risk score based on flags and confidence.
    pub fn risk_score(&self) -> f32 {
        let flag_ratio = self.flagged_count as f32 / self.transaction_count.max(1) as f32;
        let confidence_factor = 1.0 - self.avg_confidence;
        0.6 * flag_ratio + 0.4 * confidence_factor
    }
}

/// Flow direction for analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowDirection {
    /// Money flowing into an account
    Inflow,
    /// Money flowing out of an account
    Outflow,
    /// Both directions (for graph traversal)
    Both,
}

/// Edge in the graph for traversal algorithms.
#[derive(Debug, Clone, Copy)]
pub struct GraphEdge {
    /// Source node index.
    pub from: u16,
    /// Target node index.
    pub to: u16,
    /// Edge weight (amount, frequency, or custom metric).
    pub weight: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_flow_size() {
        let size = std::mem::size_of::<TransactionFlow>();
        assert!(
            size >= 64,
            "TransactionFlow should be at least 64 bytes, got {}",
            size
        );
        assert!(
            size % 64 == 0,
            "TransactionFlow should be 64-byte aligned, got {}",
            size
        );
    }

    #[test]
    fn test_aggregated_flow() {
        let mut agg = AggregatedFlow::new(0, 1);

        let flow1 = TransactionFlow::new(
            0,
            1,
            Decimal128::from_f64(100.0),
            Uuid::new_v4(),
            HybridTimestamp::now(),
        );

        let mut flow2 = TransactionFlow::new(
            0,
            1,
            Decimal128::from_f64(200.0),
            Uuid::new_v4(),
            HybridTimestamp::now(),
        );
        flow2.method_used = SolvingMethod::MethodB;

        agg.add(&flow1);
        agg.add(&flow2);

        assert_eq!(agg.transaction_count, 2);
        assert!((agg.total_amount - 300.0).abs() < 0.01);
        // With equal counts (1 A, 1 B), max_by_key returns the last one with max count
        let dominant = agg.dominant_method();
        assert!(dominant == SolvingMethod::MethodA || dominant == SolvingMethod::MethodB);
    }

    #[test]
    fn test_flow_flags() {
        let mut flow = TransactionFlow::new(
            0,
            1,
            Decimal128::from_f64(100.0),
            Uuid::new_v4(),
            HybridTimestamp::now(),
        );

        assert!(!flow.is_anomalous());

        flow.flags.set(FlowFlags::IS_CIRCULAR);
        assert!(flow.is_anomalous());
    }
}
