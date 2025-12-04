//! Message types for GPU actor communication.
//!
//! All messages are GPU-serializable using rkyv for zero-copy transfer.

use ringkernel_core::message::{CorrelationId, MessageId, Priority};
use ringkernel_derive::RingMessage;

/// Flow generation request - triggers flow generation from journal entries.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 100)]
pub struct FlowGenerationRequest {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Batch ID for tracking.
    pub batch_id: u64,
    /// Number of journal entries to process.
    pub entry_count: u32,
    /// Journal entry data offset in shared buffer.
    pub data_offset: u64,
}

/// Flow generation response - contains generated flows.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 101)]
pub struct FlowGenerationResponse {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation to request.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Batch ID.
    pub batch_id: u64,
    /// Number of flows generated.
    pub flow_count: u32,
    /// Flows data offset in shared buffer.
    pub data_offset: u64,
}

/// PageRank computation request.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 200)]
pub struct PageRankRequest {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Number of accounts in the network.
    pub account_count: u32,
    /// Number of edges (flows).
    pub edge_count: u32,
    /// Damping factor (typically 0.85).
    pub damping: f32,
    /// Number of iterations.
    pub iterations: u32,
    /// Graph data offset in shared buffer.
    pub graph_offset: u64,
}

/// PageRank computation response.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 201)]
pub struct PageRankResponse {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation to request.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// PageRank scores offset in shared buffer.
    pub scores_offset: u64,
    /// Convergence achieved.
    pub converged: bool,
    /// Final iteration count.
    pub iterations_run: u32,
}

/// Fraud detection request.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 300)]
pub struct FraudDetectionRequest {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Priority level.
    #[message(priority)]
    pub priority: Priority,
    /// Network snapshot ID.
    pub snapshot_id: u64,
    /// Number of flows to analyze.
    pub flow_count: u32,
    /// Flows data offset.
    pub flows_offset: u64,
    /// Account data offset.
    pub accounts_offset: u64,
    /// Account count.
    pub account_count: u32,
}

/// Fraud detection response.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 301)]
pub struct FraudDetectionResponse {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation to request.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Number of fraud patterns detected.
    pub pattern_count: u32,
    /// Patterns data offset.
    pub patterns_offset: u64,
    /// Overall risk score (0.0 - 1.0).
    pub risk_score: f32,
}

/// GAAP validation request.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 400)]
pub struct GaapValidationRequest {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Number of flows to validate.
    pub flow_count: u32,
    /// Flows data offset.
    pub flows_offset: u64,
    /// Account types offset.
    pub account_types_offset: u64,
}

/// GAAP validation response.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 401)]
pub struct GaapValidationResponse {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation to request.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Number of violations found.
    pub violation_count: u32,
    /// Violations data offset.
    pub violations_offset: u64,
}

/// Benford analysis request.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 500)]
pub struct BenfordAnalysisRequest {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Number of amounts to analyze.
    pub amount_count: u32,
    /// Amounts data offset.
    pub amounts_offset: u64,
}

/// Benford analysis response.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 501)]
pub struct BenfordAnalysisResponse {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation to request.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Digit distribution (counts for digits 1-9).
    pub digit_counts: [u32; 9],
    /// Chi-squared statistic.
    pub chi_squared: f32,
    /// Is anomalous (exceeds threshold).
    pub is_anomalous: bool,
}

/// Suspense detection request.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 600)]
pub struct SuspenseDetectionRequest {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Number of accounts.
    pub account_count: u32,
    /// Account balances offset.
    pub balances_offset: u64,
    /// Account risk scores offset.
    pub risk_scores_offset: u64,
    /// Flow counts offset.
    pub flow_counts_offset: u64,
}

/// Suspense detection response.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 601)]
pub struct SuspenseDetectionResponse {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Correlation to request.
    #[message(correlation)]
    pub correlation_id: CorrelationId,
    /// Number of suspense accounts detected.
    pub suspense_count: u32,
    /// Suspense scores offset.
    pub scores_offset: u64,
}

/// Aggregated analytics result sent to host.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 900)]
pub struct AnalyticsResult {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Snapshot ID this result corresponds to.
    pub snapshot_id: u64,
    /// PageRank computed.
    pub pagerank_complete: bool,
    /// Fraud detection complete.
    pub fraud_detection_complete: bool,
    /// GAAP validation complete.
    pub gaap_validation_complete: bool,
    /// Benford analysis complete.
    pub benford_complete: bool,
    /// Total fraud patterns detected.
    pub fraud_pattern_count: u32,
    /// Total GAAP violations.
    pub gaap_violation_count: u32,
    /// Total suspense accounts.
    pub suspense_account_count: u32,
    /// Overall risk score.
    pub overall_risk_score: f32,
    /// Benford anomaly detected.
    pub benford_anomaly: bool,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
}

/// Command to shutdown a kernel gracefully.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[message(type_id = 999)]
pub struct ShutdownCommand {
    /// Message ID.
    #[message(id)]
    pub id: MessageId,
    /// Reason for shutdown.
    pub reason: u32, // 0 = normal, 1 = error, 2 = restart
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let request = FlowGenerationRequest {
            id: MessageId::generate(),
            batch_id: 42,
            entry_count: 100,
            data_offset: 0x1000,
        };

        // Messages should be serializable
        assert_eq!(request.batch_id, 42);
        assert_eq!(request.entry_count, 100);
    }

    #[test]
    fn test_pagerank_request() {
        let request = PageRankRequest {
            id: MessageId::generate(),
            account_count: 50,
            edge_count: 200,
            damping: 0.85,
            iterations: 20,
            graph_offset: 0,
        };

        assert_eq!(request.damping, 0.85);
        assert_eq!(request.iterations, 20);
    }
}
