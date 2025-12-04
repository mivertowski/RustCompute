//! Actor message types for process intelligence pipeline.

use crate::models::{
    ConformanceResult, GpuDFGEdge, GpuDFGNode, GpuObjectEvent, GpuPartialOrderTrace,
    GpuPatternMatch,
};

/// Message types for the processing pipeline.
#[derive(Debug, Clone)]
pub enum PipelineMessage {
    /// New events to process.
    Events(EventBatch),
    /// DFG update notification.
    DfgUpdate(DfgUpdateMessage),
    /// Pattern detection results.
    PatternsDetected(PatternBatchMessage),
    /// Conformance check results.
    ConformanceResults(ConformanceBatchMessage),
    /// Partial order derivation results.
    PartialOrders(PartialOrderBatchMessage),
    /// Pipeline control.
    Control(ControlMessage),
}

/// Batch of events to process.
#[derive(Debug, Clone)]
pub struct EventBatch {
    /// Batch ID.
    pub batch_id: u64,
    /// Events in the batch.
    pub events: Vec<GpuObjectEvent>,
    /// Timestamp when batch was created.
    pub timestamp_ms: u64,
}

impl EventBatch {
    /// Create a new event batch.
    pub fn new(batch_id: u64, events: Vec<GpuObjectEvent>) -> Self {
        Self {
            batch_id,
            events,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    /// Get batch size.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

/// DFG update message.
#[derive(Debug, Clone)]
pub struct DfgUpdateMessage {
    /// Batch ID that triggered update.
    pub batch_id: u64,
    /// Updated nodes.
    pub nodes: Vec<GpuDFGNode>,
    /// Updated edges.
    pub edges: Vec<GpuDFGEdge>,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
}

/// Pattern batch message.
#[derive(Debug, Clone)]
pub struct PatternBatchMessage {
    /// Batch ID.
    pub batch_id: u64,
    /// Detected patterns.
    pub patterns: Vec<GpuPatternMatch>,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
}

/// Conformance batch message.
#[derive(Debug, Clone)]
pub struct ConformanceBatchMessage {
    /// Batch ID.
    pub batch_id: u64,
    /// Conformance results.
    pub results: Vec<ConformanceResult>,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
}

/// Partial order batch message.
#[derive(Debug, Clone)]
pub struct PartialOrderBatchMessage {
    /// Batch ID.
    pub batch_id: u64,
    /// Derived partial orders.
    pub traces: Vec<GpuPartialOrderTrace>,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
}

/// Pipeline control messages.
#[derive(Debug, Clone, Copy)]
pub enum ControlMessage {
    /// Start processing.
    Start,
    /// Pause processing.
    Pause,
    /// Stop processing.
    Stop,
    /// Reset all state.
    Reset,
    /// Request statistics.
    GetStats,
}

/// Pipeline statistics.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total events processed.
    pub total_events: u64,
    /// Total batches processed.
    pub batches_processed: u64,
    /// Total patterns detected.
    pub patterns_detected: u64,
    /// Total conformance checks.
    pub conformance_checks: u64,
    /// Average batch processing time (us).
    pub avg_batch_time_us: f64,
    /// Events per second.
    pub events_per_second: f64,
    /// Is running.
    pub is_running: bool,
}

impl PipelineStats {
    /// Update with new batch metrics.
    pub fn record_batch(&mut self, events: u64, processing_time_us: u64) {
        self.total_events += events;
        self.batches_processed += 1;

        // Update rolling average
        let alpha = 0.1;
        self.avg_batch_time_us =
            self.avg_batch_time_us * (1.0 - alpha) + processing_time_us as f64 * alpha;

        // Update throughput estimate
        if processing_time_us > 0 {
            let batch_eps = events as f64 * 1_000_000.0 / processing_time_us as f64;
            self.events_per_second = self.events_per_second * (1.0 - alpha) + batch_eps * alpha;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::HybridTimestamp;

    #[test]
    fn test_event_batch() {
        let events = vec![GpuObjectEvent {
            event_id: 1,
            object_id: 100,
            activity_id: 1,
            timestamp: HybridTimestamp::now(),
            ..Default::default()
        }];

        let batch = EventBatch::new(1, events);
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_pipeline_stats() {
        let mut stats = PipelineStats::default();
        stats.record_batch(1000, 1000); // 1000 events in 1ms = 1M eps
        assert!(stats.events_per_second > 0.0);
    }
}
