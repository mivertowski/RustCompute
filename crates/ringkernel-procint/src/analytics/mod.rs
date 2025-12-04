//! Analytics engines for process intelligence.
//!
//! Aggregates results from GPU kernels into actionable insights.

mod dfg_metrics;
mod kpi_tracking;
mod pattern_analysis;

pub use dfg_metrics::*;
pub use kpi_tracking::*;
pub use pattern_analysis::*;

/// Main analytics engine coordinating all analysis components.
#[derive(Debug)]
pub struct AnalyticsEngine {
    /// DFG metrics calculator.
    pub dfg_metrics: DFGMetricsCalculator,
    /// Pattern aggregator.
    pub pattern_aggregator: PatternAggregator,
    /// KPI tracker.
    pub kpi_tracker: KPITracker,
}

impl Default for AnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalyticsEngine {
    /// Create a new analytics engine.
    pub fn new() -> Self {
        Self {
            dfg_metrics: DFGMetricsCalculator::new(),
            pattern_aggregator: PatternAggregator::new(),
            kpi_tracker: KPITracker::new(),
        }
    }
}
