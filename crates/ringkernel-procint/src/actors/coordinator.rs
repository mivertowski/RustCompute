//! Pipeline coordinator for process intelligence.
//!
//! Orchestrates the flow of events through GPU kernels.

use super::{
    ControlMessage, DfgUpdateMessage, PartialOrderBatchMessage, PatternBatchMessage,
    PipelineMessage, PipelineStats,
};
use crate::analytics::AnalyticsEngine;
use crate::cuda::GpuStatus;
use crate::fabric::{PipelineConfig, ProcessingPipeline, SectorTemplate};
use crate::kernels::{
    ConformanceKernel, DfgConstructionKernel, PartialOrderKernel, PatternConfig,
    PatternDetectionKernel,
};
use crate::models::ProcessModel;
use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver, Sender};

/// A process variation (unique trace pattern).
#[derive(Debug, Clone)]
pub struct ProcessVariation {
    /// Signature (activity sequence as string).
    pub signature: String,
    /// Activity IDs in order.
    pub activities: Vec<u32>,
    /// Number of cases with this variation.
    pub count: u64,
    /// Average duration.
    pub avg_duration_ms: f32,
    /// Is conformant to model.
    pub is_conformant: bool,
}

/// GPU statistics report.
#[derive(Debug, Clone, Default)]
pub struct GpuStatsReport {
    /// DFG kernel launches.
    pub dfg_kernel_launches: u64,
    /// Pattern detection kernel launches.
    pub pattern_kernel_launches: u64,
    /// Total GPU execution time in microseconds.
    pub total_gpu_time_us: u64,
    /// Total elements processed on GPU.
    pub total_elements_gpu: u64,
    /// Total bytes transferred to/from GPU.
    pub bytes_transferred: u64,
    /// Whether DFG kernel is using GPU.
    pub dfg_using_gpu: bool,
    /// Whether pattern kernel is using GPU.
    pub pattern_using_gpu: bool,
}

impl GpuStatsReport {
    /// Check if any GPU kernels are active.
    pub fn is_gpu_active(&self) -> bool {
        self.dfg_using_gpu || self.pattern_using_gpu
    }

    /// Get GPU throughput in elements per second.
    pub fn throughput(&self) -> f64 {
        if self.total_gpu_time_us > 0 {
            self.total_elements_gpu as f64 * 1_000_000.0 / self.total_gpu_time_us as f64
        } else {
            0.0
        }
    }

    /// Get total kernel launches.
    pub fn total_launches(&self) -> u64 {
        self.dfg_kernel_launches + self.pattern_kernel_launches
    }
}

/// Pipeline coordinator managing all processing stages.
pub struct PipelineCoordinator {
    /// Data generation pipeline.
    pipeline: ProcessingPipeline,
    /// DFG construction kernel.
    dfg_kernel: DfgConstructionKernel,
    /// Pattern detection kernel.
    pattern_kernel: PatternDetectionKernel,
    /// Partial order kernel.
    partial_order_kernel: PartialOrderKernel,
    /// Conformance kernel.
    conformance_kernel: ConformanceKernel,
    /// Analytics engine.
    analytics: AnalyticsEngine,
    /// Pipeline statistics.
    stats: PipelineStats,
    /// Message sender for results.
    result_sender: Option<Sender<PipelineMessage>>,
    /// Next batch ID.
    next_batch_id: u64,
    /// Current DFG (updated each tick).
    current_dfg: crate::models::DFGGraph,
    /// Current patterns.
    current_patterns: Vec<crate::models::GpuPatternMatch>,
    /// Current partial orders.
    current_partial_orders: Vec<crate::models::GpuPartialOrderTrace>,
    /// Current conformance results.
    current_conformance: Option<crate::kernels::ConformanceCheckResult>,
    /// Process variations (unique trace signatures).
    variations: std::collections::HashMap<String, ProcessVariation>,
}

impl PipelineCoordinator {
    /// Create a new pipeline coordinator.
    pub fn new(sector: SectorTemplate, config: PipelineConfig) -> Self {
        // Create a reference model from the sector for conformance checking
        let model = Self::create_model_from_sector(&sector);
        let pipeline = ProcessingPipeline::new(sector, config);

        Self {
            pipeline,
            dfg_kernel: DfgConstructionKernel::default(),
            pattern_kernel: PatternDetectionKernel::new(PatternConfig::default()),
            partial_order_kernel: PartialOrderKernel::default(),
            conformance_kernel: ConformanceKernel::new(model),
            analytics: AnalyticsEngine::new(),
            stats: PipelineStats::default(),
            result_sender: None,
            next_batch_id: 1,
            current_dfg: crate::models::DFGGraph::default(),
            current_patterns: Vec::new(),
            current_partial_orders: Vec::new(),
            current_conformance: None,
            variations: HashMap::new(),
        }
    }

    /// Create a process model from sector template.
    fn create_model_from_sector(sector: &SectorTemplate) -> ProcessModel {
        use crate::models::ProcessModelType;

        let mut model = ProcessModel::new(1, sector.name(), ProcessModelType::DFG);
        let registry = sector.build_registry();

        // Add transitions from sector, converting names to IDs
        for trans in sector.transitions() {
            if let (Some(source), Some(target)) = (
                registry.get_by_name(trans.source),
                registry.get_by_name(trans.target),
            ) {
                model.add_transition(source.id, target.id);
            }
        }

        // Add start/end activities
        for name in sector.start_activities() {
            if let Some(activity) = registry.get_by_name(name) {
                model.start_activities.push(activity.id);
            }
        }
        for name in sector.end_activities() {
            if let Some(activity) = registry.get_by_name(name) {
                model.end_activities.push(activity.id);
            }
        }

        model
    }

    /// Set reference model for conformance checking.
    pub fn with_conformance_model(mut self, model: ProcessModel) -> Self {
        self.conformance_kernel = ConformanceKernel::new(model);
        self
    }

    /// Create result channel.
    pub fn create_result_channel(&mut self) -> Receiver<PipelineMessage> {
        let (sender, receiver) = channel();
        self.result_sender = Some(sender);
        receiver
    }

    /// Get current statistics.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get analytics engine.
    pub fn analytics(&self) -> &AnalyticsEngine {
        &self.analytics
    }

    /// Get mutable analytics engine.
    pub fn analytics_mut(&mut self) -> &mut AnalyticsEngine {
        &mut self.analytics
    }

    /// Get current DFG graph.
    pub fn current_dfg(&self) -> &crate::models::DFGGraph {
        &self.current_dfg
    }

    /// Get current detected patterns.
    pub fn current_patterns(&self) -> &[crate::models::GpuPatternMatch] {
        &self.current_patterns
    }

    /// Get current partial orders.
    pub fn current_partial_orders(&self) -> &[crate::models::GpuPartialOrderTrace] {
        &self.current_partial_orders
    }

    /// Get current conformance results.
    pub fn current_conformance(&self) -> Option<&crate::kernels::ConformanceCheckResult> {
        self.current_conformance.as_ref()
    }

    /// Get process variations.
    pub fn variations(&self) -> &HashMap<String, ProcessVariation> {
        &self.variations
    }

    /// Get GPU status for DFG kernel.
    pub fn dfg_gpu_status(&self) -> GpuStatus {
        self.dfg_kernel.gpu_status()
    }

    /// Get GPU status for pattern detection kernel.
    pub fn pattern_gpu_status(&self) -> GpuStatus {
        self.pattern_kernel.gpu_status()
    }

    /// Check if DFG kernel is using GPU.
    pub fn is_dfg_using_gpu(&self) -> bool {
        self.dfg_kernel.is_using_gpu()
    }

    /// Check if pattern kernel is using GPU.
    pub fn is_pattern_using_gpu(&self) -> bool {
        self.pattern_kernel.is_using_gpu()
    }

    /// Get combined GPU stats.
    pub fn gpu_stats(&self) -> GpuStatsReport {
        let dfg_stats = self.dfg_kernel.gpu_stats();
        let pattern_stats = self.pattern_kernel.gpu_stats();

        GpuStatsReport {
            dfg_kernel_launches: dfg_stats.kernel_launches,
            pattern_kernel_launches: pattern_stats.kernel_launches,
            total_gpu_time_us: dfg_stats.total_gpu_time_us + pattern_stats.total_gpu_time_us,
            total_elements_gpu: dfg_stats.total_elements_gpu + pattern_stats.total_elements_gpu,
            bytes_transferred: dfg_stats.bytes_to_gpu
                + dfg_stats.bytes_from_gpu
                + pattern_stats.bytes_to_gpu
                + pattern_stats.bytes_from_gpu,
            dfg_using_gpu: self.dfg_kernel.is_using_gpu(),
            pattern_using_gpu: self.pattern_kernel.is_using_gpu(),
        }
    }

    /// Handle control message.
    pub fn handle_control(&mut self, msg: ControlMessage) {
        match msg {
            ControlMessage::Start => {
                self.pipeline.start();
                self.stats.is_running = true;
            }
            ControlMessage::Pause => {
                self.pipeline.stop();
                self.stats.is_running = false;
            }
            ControlMessage::Stop => {
                self.pipeline.stop();
                self.stats.is_running = false;
            }
            ControlMessage::Reset => {
                self.reset();
            }
            ControlMessage::GetStats => {
                // Stats are available via stats() method
            }
        }
    }

    /// Process a single tick (generate + process batch).
    pub fn tick(&mut self) -> Option<ProcessedBatch> {
        if !self.pipeline.is_running() {
            return None;
        }

        let start = std::time::Instant::now();

        // Generate events
        let events = self.pipeline.generate_batch();
        if events.is_empty() {
            return None;
        }

        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        // Process through DFG kernel
        let dfg_result = self.dfg_kernel.process(&events);

        // Update analytics
        self.analytics.dfg_metrics.calculate(&dfg_result.dfg);

        // Detect patterns
        let pattern_result = self.pattern_kernel.detect(dfg_result.dfg.nodes());
        self.analytics
            .pattern_aggregator
            .add_patterns(&pattern_result.patterns);

        // Derive partial orders
        let partial_order_result = self.partial_order_kernel.derive(&events);

        // Conformance checking
        let conformance_results = self.conformance_kernel.check(&events);

        // Update variations from partial orders
        self.update_variations(&partial_order_result.traces, &conformance_results);

        // Update KPIs
        let total_time = start.elapsed().as_micros() as u64;
        self.stats.record_batch(events.len() as u64, total_time);

        // Update KPI tracker
        let avg_duration = dfg_result
            .dfg
            .nodes()
            .iter()
            .map(|n| n.avg_duration_ms)
            .sum::<f32>()
            / dfg_result.dfg.nodes().len().max(1) as f32;

        let fitness = conformance_results.avg_fitness();

        self.analytics
            .kpi_tracker
            .update(events.len() as u64, avg_duration, fitness);
        self.analytics
            .kpi_tracker
            .set_pattern_count(pattern_result.patterns.len() as u64);

        // Store current results for access via getters
        self.current_dfg = dfg_result.dfg.clone();
        self.current_patterns = pattern_result.patterns.clone();
        self.current_partial_orders = partial_order_result.traces.clone();
        self.current_conformance = Some(conformance_results);

        // Send results if channel exists
        if let Some(sender) = &self.result_sender {
            let _ = sender.send(PipelineMessage::DfgUpdate(DfgUpdateMessage {
                batch_id,
                nodes: self.current_dfg.nodes().to_vec(),
                edges: self.current_dfg.edges().to_vec(),
                processing_time_us: dfg_result.total_time_us,
            }));

            let _ = sender.send(PipelineMessage::PatternsDetected(PatternBatchMessage {
                batch_id,
                patterns: self.current_patterns.clone(),
                processing_time_us: pattern_result.total_time_us,
            }));

            let _ = sender.send(PipelineMessage::PartialOrders(PartialOrderBatchMessage {
                batch_id,
                traces: self.current_partial_orders.clone(),
                processing_time_us: partial_order_result.total_time_us,
            }));
        }

        Some(ProcessedBatch {
            batch_id,
            event_count: events.len(),
            dfg_nodes: dfg_result.dfg.nodes().len(),
            dfg_edges: dfg_result.dfg.edges().len(),
            patterns_detected: pattern_result.patterns.len(),
            partial_orders: partial_order_result.traces.len(),
            processing_time_us: total_time,
        })
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.pipeline.stop();
        self.pipeline.reset_stats();
        self.analytics = AnalyticsEngine::new();
        self.stats = PipelineStats::default();
        self.next_batch_id = 1;
        self.current_dfg = crate::models::DFGGraph::default();
        self.current_patterns.clear();
        self.current_partial_orders.clear();
        self.current_conformance = None;
        self.variations.clear();
        self.dfg_kernel = DfgConstructionKernel::default();
        self.partial_order_kernel = PartialOrderKernel::default();
    }

    /// Change sector.
    pub fn set_sector(&mut self, sector: SectorTemplate) {
        // Update conformance model for new sector
        let model = Self::create_model_from_sector(&sector);
        self.conformance_kernel = ConformanceKernel::new(model);
        self.pipeline.set_sector(sector);
        self.reset();
    }

    /// Update process variations from traces.
    fn update_variations(
        &mut self,
        traces: &[crate::models::GpuPartialOrderTrace],
        conformance: &crate::kernels::ConformanceCheckResult,
    ) {
        for trace in traces {
            // Create signature from activity sequence
            let activities: Vec<u32> = trace.activity_ids[..trace.activity_count as usize].to_vec();
            let signature: String = activities
                .iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join("->");

            // Check if this case is conformant
            let is_conformant = conformance
                .results
                .iter()
                .find(|r| r.trace_id == trace.case_id)
                .map(|r| r.is_conformant())
                .unwrap_or(true);

            // Update or create variation
            let duration = (trace.end_time.physical_ms - trace.start_time.physical_ms) as f32;

            if let Some(var) = self.variations.get_mut(&signature) {
                // Update existing
                let total_duration = var.avg_duration_ms * var.count as f32 + duration;
                var.count += 1;
                var.avg_duration_ms = total_duration / var.count as f32;
            } else {
                // Create new
                self.variations.insert(
                    signature.clone(),
                    ProcessVariation {
                        signature,
                        activities,
                        count: 1,
                        avg_duration_ms: duration,
                        is_conformant,
                    },
                );
            }
        }
    }

    /// Get pipeline reference.
    pub fn pipeline(&self) -> &ProcessingPipeline {
        &self.pipeline
    }

    /// Get mutable pipeline reference.
    pub fn pipeline_mut(&mut self) -> &mut ProcessingPipeline {
        &mut self.pipeline
    }
}

/// Summary of a processed batch.
#[derive(Debug, Clone)]
pub struct ProcessedBatch {
    /// Batch ID.
    pub batch_id: u64,
    /// Number of events processed.
    pub event_count: usize,
    /// Number of DFG nodes.
    pub dfg_nodes: usize,
    /// Number of DFG edges.
    pub dfg_edges: usize,
    /// Number of patterns detected.
    pub patterns_detected: usize,
    /// Number of partial orders derived.
    pub partial_orders: usize,
    /// Total processing time in microseconds.
    pub processing_time_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::{FinanceConfig, HealthcareConfig, ManufacturingConfig, PipelineConfig};

    #[test]
    fn test_coordinator_creation() {
        let coord = PipelineCoordinator::new(
            SectorTemplate::Healthcare(HealthcareConfig::default()),
            PipelineConfig::default(),
        );
        assert!(!coord.stats().is_running);
    }

    #[test]
    fn test_coordinator_lifecycle() {
        let mut coord = PipelineCoordinator::new(
            SectorTemplate::Manufacturing(ManufacturingConfig::default()),
            PipelineConfig::default(),
        );

        coord.handle_control(ControlMessage::Start);
        assert!(coord.stats().is_running);

        coord.handle_control(ControlMessage::Pause);
        assert!(!coord.stats().is_running);
    }

    #[test]
    fn test_tick_processing() {
        let mut coord = PipelineCoordinator::new(
            SectorTemplate::Finance(FinanceConfig::default()),
            PipelineConfig::default(),
        );

        coord.handle_control(ControlMessage::Start);
        let result = coord.tick();

        assert!(result.is_some());
        let batch = result.unwrap();
        assert!(batch.event_count > 0);
    }
}
