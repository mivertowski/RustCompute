//! Analytics coordinator for orchestrating GPU kernel actors.
//!
//! The coordinator manages the lifecycle of all analytics kernels
//! and orchestrates the flow of data through the pipeline.

use std::time::{Duration, Instant};

use ringkernel_core::MessageId;

use super::messages::*;

/// Kernel identifiers for the analytics pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalyticsKernelId {
    /// PageRank computation kernel.
    PageRank,
    /// Fraud detection kernel.
    FraudDetector,
    /// GAAP validation kernel.
    GaapValidator,
    /// Benford analysis kernel.
    BenfordAnalyzer,
    /// Suspense detection kernel.
    SuspenseDetector,
    /// Results aggregator kernel.
    ResultsAggregator,
}

impl AnalyticsKernelId {
    /// Get the kernel name for launching.
    pub fn name(&self) -> &'static str {
        match self {
            Self::PageRank => "pagerank_actor",
            Self::FraudDetector => "fraud_detector_actor",
            Self::GaapValidator => "gaap_validator_actor",
            Self::BenfordAnalyzer => "benford_analyzer_actor",
            Self::SuspenseDetector => "suspense_detector_actor",
            Self::ResultsAggregator => "results_aggregator_actor",
        }
    }

    /// All kernel IDs in launch order.
    pub fn all() -> &'static [Self] {
        &[
            Self::PageRank,
            Self::FraudDetector,
            Self::GaapValidator,
            Self::BenfordAnalyzer,
            Self::SuspenseDetector,
            Self::ResultsAggregator,
        ]
    }
}

/// Configuration for the analytics coordinator.
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// PageRank damping factor.
    pub pagerank_damping: f32,
    /// PageRank iterations.
    pub pagerank_iterations: u32,
    /// Velocity threshold for fraud detection.
    pub velocity_threshold: f32,
    /// Round amount threshold.
    pub round_amount_threshold: f64,
    /// Queue capacity for each kernel.
    pub queue_capacity: usize,
    /// Block size for kernels.
    pub block_size: u32,
    /// Enable K2K messaging.
    pub enable_k2k: bool,
    /// Enable HLC timestamps.
    pub enable_hlc: bool,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            pagerank_damping: 0.85,
            pagerank_iterations: 20,
            velocity_threshold: 10.0,
            round_amount_threshold: 1000.0,
            queue_capacity: 256,
            block_size: 256,
            enable_k2k: true,
            enable_hlc: true,
        }
    }
}

/// Pipeline state tracking.
#[derive(Debug, Default)]
pub struct PipelineState {
    /// Current snapshot being processed.
    pub current_snapshot_id: u64,
    /// PageRank complete for current snapshot.
    pub pagerank_complete: bool,
    /// Fraud detection complete.
    pub fraud_detection_complete: bool,
    /// GAAP validation complete.
    pub gaap_validation_complete: bool,
    /// Benford analysis complete.
    pub benford_complete: bool,
    /// Suspense detection complete.
    pub suspense_complete: bool,
    /// Processing start time.
    pub start_time: Option<Instant>,
    /// Results collected.
    pub fraud_pattern_count: u32,
    /// GAAP violations.
    pub gaap_violation_count: u32,
    /// Suspense accounts.
    pub suspense_account_count: u32,
    /// Benford anomaly detected.
    pub benford_anomaly: bool,
}

impl PipelineState {
    /// Check if all analytics are complete.
    pub fn is_complete(&self) -> bool {
        self.pagerank_complete
            && self.fraud_detection_complete
            && self.gaap_validation_complete
            && self.benford_complete
            && self.suspense_complete
    }

    /// Get processing duration.
    pub fn processing_time(&self) -> Option<Duration> {
        self.start_time.map(|t| t.elapsed())
    }

    /// Reset for new snapshot.
    pub fn reset(&mut self, snapshot_id: u64) {
        self.current_snapshot_id = snapshot_id;
        self.pagerank_complete = false;
        self.fraud_detection_complete = false;
        self.gaap_validation_complete = false;
        self.benford_complete = false;
        self.suspense_complete = false;
        self.start_time = Some(Instant::now());
        self.fraud_pattern_count = 0;
        self.gaap_violation_count = 0;
        self.suspense_account_count = 0;
        self.benford_anomaly = false;
    }
}

/// Analytics pipeline coordinator.
///
/// Manages GPU kernel actors and orchestrates the analytics pipeline.
pub struct AnalyticsCoordinator {
    /// Configuration.
    pub config: CoordinatorConfig,
    /// Pipeline state.
    pub state: PipelineState,
    /// Next snapshot ID.
    next_snapshot_id: u64,
    /// Processing statistics.
    pub stats: CoordinatorStats,
}

/// Coordinator statistics.
#[derive(Debug, Default, Clone)]
pub struct CoordinatorStats {
    /// Total snapshots processed.
    pub snapshots_processed: u64,
    /// Total processing time (microseconds).
    pub total_processing_time_us: u64,
    /// Average processing time (microseconds).
    pub avg_processing_time_us: f64,
    /// Total fraud patterns detected.
    pub total_fraud_patterns: u64,
    /// Total GAAP violations.
    pub total_gaap_violations: u64,
    /// Total suspense accounts flagged.
    pub total_suspense_accounts: u64,
}

impl AnalyticsCoordinator {
    /// Create a new coordinator.
    pub fn new(config: CoordinatorConfig) -> Self {
        Self {
            config,
            state: PipelineState::default(),
            next_snapshot_id: 1,
            stats: CoordinatorStats::default(),
        }
    }

    /// Start processing a new network snapshot.
    pub fn begin_snapshot(&mut self) -> u64 {
        let snapshot_id = self.next_snapshot_id;
        self.next_snapshot_id += 1;
        self.state.reset(snapshot_id);
        snapshot_id
    }

    /// Create a PageRank request.
    pub fn create_pagerank_request(
        &self,
        account_count: u32,
        edge_count: u32,
        graph_offset: u64,
    ) -> PageRankRequest {
        PageRankRequest {
            id: MessageId::generate(),
            account_count,
            edge_count,
            damping: self.config.pagerank_damping,
            iterations: self.config.pagerank_iterations,
            graph_offset,
        }
    }

    /// Create a fraud detection request.
    pub fn create_fraud_detection_request(
        &self,
        flow_count: u32,
        flows_offset: u64,
        accounts_offset: u64,
        account_count: u32,
    ) -> FraudDetectionRequest {
        FraudDetectionRequest {
            id: MessageId::generate(),
            priority: ringkernel_core::Priority::High,
            snapshot_id: self.state.current_snapshot_id,
            flow_count,
            flows_offset,
            accounts_offset,
            account_count,
        }
    }

    /// Create a GAAP validation request.
    pub fn create_gaap_validation_request(
        &self,
        flow_count: u32,
        flows_offset: u64,
        account_types_offset: u64,
    ) -> GaapValidationRequest {
        GaapValidationRequest {
            id: MessageId::generate(),
            flow_count,
            flows_offset,
            account_types_offset,
        }
    }

    /// Create a Benford analysis request.
    pub fn create_benford_analysis_request(
        &self,
        amount_count: u32,
        amounts_offset: u64,
    ) -> BenfordAnalysisRequest {
        BenfordAnalysisRequest {
            id: MessageId::generate(),
            amount_count,
            amounts_offset,
        }
    }

    /// Create a suspense detection request.
    pub fn create_suspense_detection_request(
        &self,
        account_count: u32,
        balances_offset: u64,
        risk_scores_offset: u64,
        flow_counts_offset: u64,
    ) -> SuspenseDetectionRequest {
        SuspenseDetectionRequest {
            id: MessageId::generate(),
            account_count,
            balances_offset,
            risk_scores_offset,
            flow_counts_offset,
        }
    }

    /// Handle PageRank response.
    pub fn handle_pagerank_response(&mut self, _response: PageRankResponse) {
        self.state.pagerank_complete = true;
        // PageRank scores are stored at response.scores_offset
    }

    /// Handle fraud detection response.
    pub fn handle_fraud_response(&mut self, response: FraudDetectionResponse) {
        self.state.fraud_detection_complete = true;
        self.state.fraud_pattern_count = response.pattern_count;
    }

    /// Handle GAAP validation response.
    pub fn handle_gaap_response(&mut self, response: GaapValidationResponse) {
        self.state.gaap_validation_complete = true;
        self.state.gaap_violation_count = response.violation_count;
    }

    /// Handle Benford analysis response.
    pub fn handle_benford_response(&mut self, response: BenfordAnalysisResponse) {
        self.state.benford_complete = true;
        self.state.benford_anomaly = response.is_anomalous;
    }

    /// Handle suspense detection response.
    pub fn handle_suspense_response(&mut self, response: SuspenseDetectionResponse) {
        self.state.suspense_complete = true;
        self.state.suspense_account_count = response.suspense_count;
    }

    /// Finalize the current snapshot.
    pub fn finalize_snapshot(&mut self) -> AnalyticsResult {
        let processing_time = self
            .state
            .processing_time()
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);

        // Update statistics
        self.stats.snapshots_processed += 1;
        self.stats.total_processing_time_us += processing_time;
        self.stats.avg_processing_time_us =
            self.stats.total_processing_time_us as f64 / self.stats.snapshots_processed as f64;
        self.stats.total_fraud_patterns += self.state.fraud_pattern_count as u64;
        self.stats.total_gaap_violations += self.state.gaap_violation_count as u64;
        self.stats.total_suspense_accounts += self.state.suspense_account_count as u64;

        // Calculate overall risk score
        let fraud_risk = (self.state.fraud_pattern_count as f32 / 100.0).min(1.0);
        let gaap_risk = (self.state.gaap_violation_count as f32 / 50.0).min(1.0);
        let suspense_risk = (self.state.suspense_account_count as f32 / 20.0).min(1.0);
        let benford_risk = if self.state.benford_anomaly { 0.5 } else { 0.0 };

        let overall_risk =
            (fraud_risk * 0.35 + gaap_risk * 0.25 + suspense_risk * 0.25 + benford_risk * 0.15)
                .min(1.0);

        AnalyticsResult {
            id: MessageId::generate(),
            snapshot_id: self.state.current_snapshot_id,
            pagerank_complete: self.state.pagerank_complete,
            fraud_detection_complete: self.state.fraud_detection_complete,
            gaap_validation_complete: self.state.gaap_validation_complete,
            benford_complete: self.state.benford_complete,
            fraud_pattern_count: self.state.fraud_pattern_count,
            gaap_violation_count: self.state.gaap_violation_count,
            suspense_account_count: self.state.suspense_account_count,
            overall_risk_score: overall_risk,
            benford_anomaly: self.state.benford_anomaly,
            processing_time_us: processing_time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let coord = AnalyticsCoordinator::new(CoordinatorConfig::default());
        assert_eq!(coord.config.pagerank_damping, 0.85);
        assert_eq!(coord.config.pagerank_iterations, 20);
    }

    #[test]
    fn test_begin_snapshot() {
        let mut coord = AnalyticsCoordinator::new(CoordinatorConfig::default());
        let id1 = coord.begin_snapshot();
        let id2 = coord.begin_snapshot();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_pipeline_state() {
        let mut state = PipelineState::default();
        state.reset(1);

        assert!(!state.is_complete());

        state.pagerank_complete = true;
        state.fraud_detection_complete = true;
        state.gaap_validation_complete = true;
        state.benford_complete = true;
        state.suspense_complete = true;

        assert!(state.is_complete());
    }

    #[test]
    fn test_create_requests() {
        let coord = AnalyticsCoordinator::new(CoordinatorConfig::default());

        let pr_req = coord.create_pagerank_request(100, 500, 0);
        assert_eq!(pr_req.account_count, 100);
        assert_eq!(pr_req.edge_count, 500);
        assert_eq!(pr_req.damping, 0.85);

        let fraud_req = coord.create_fraud_detection_request(500, 0, 1000, 100);
        assert_eq!(fraud_req.flow_count, 500);
    }

    #[test]
    fn test_finalize_snapshot() {
        let mut coord = AnalyticsCoordinator::new(CoordinatorConfig::default());
        coord.begin_snapshot();

        // Simulate responses
        coord.state.pagerank_complete = true;
        coord.state.fraud_detection_complete = true;
        coord.state.fraud_pattern_count = 5;
        coord.state.gaap_validation_complete = true;
        coord.state.gaap_violation_count = 3;
        coord.state.benford_complete = true;
        coord.state.benford_anomaly = false;
        coord.state.suspense_complete = true;
        coord.state.suspense_account_count = 2;

        let result = coord.finalize_snapshot();
        assert_eq!(result.snapshot_id, 1);
        assert_eq!(result.fraud_pattern_count, 5);
        assert_eq!(result.gaap_violation_count, 3);
        assert_eq!(result.suspense_account_count, 2);
        assert!(result.overall_risk_score > 0.0);
        assert!(result.overall_risk_score <= 1.0);
    }

    #[test]
    fn test_kernel_ids() {
        assert_eq!(AnalyticsKernelId::PageRank.name(), "pagerank_actor");
        assert_eq!(
            AnalyticsKernelId::FraudDetector.name(),
            "fraud_detector_actor"
        );
        assert_eq!(AnalyticsKernelId::all().len(), 6);
    }
}
