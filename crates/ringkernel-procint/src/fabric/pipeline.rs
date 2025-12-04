//! Streaming pipeline for event processing.
//!
//! Connects event generation to GPU kernels and analytics.

use crate::fabric::{GeneratorConfig, ProcessEventGenerator, SectorTemplate};
use crate::models::GpuObjectEvent;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Generator configuration.
    pub generator: GeneratorConfig,
    /// Batch size for GPU processing.
    pub gpu_batch_size: usize,
    /// Enable DFG construction.
    pub enable_dfg: bool,
    /// Enable pattern detection.
    pub enable_patterns: bool,
    /// Enable conformance checking.
    pub enable_conformance: bool,
    /// Enable partial order derivation.
    pub enable_partial_order: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            generator: GeneratorConfig::default(),
            gpu_batch_size: 4096,
            enable_dfg: true,
            enable_patterns: true,
            enable_conformance: true,
            enable_partial_order: true,
        }
    }
}

/// Pipeline statistics.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total events processed.
    pub events_processed: u64,
    /// Total batches processed.
    pub batches_processed: u64,
    /// DFG updates.
    pub dfg_updates: u64,
    /// Patterns detected.
    pub patterns_detected: u64,
    /// Conformance checks performed.
    pub conformance_checks: u64,
    /// Average batch processing time (microseconds).
    pub avg_batch_time_us: f64,
}

/// Streaming pipeline for process intelligence.
pub struct ProcessingPipeline {
    /// Sector template.
    sector: SectorTemplate,
    /// Pipeline configuration.
    config: PipelineConfig,
    /// Event generator.
    generator: ProcessEventGenerator,
    /// Running flag.
    running: Arc<AtomicBool>,
    /// Event counter.
    event_count: Arc<AtomicU64>,
    /// Batch counter.
    batch_count: Arc<AtomicU64>,
}

impl ProcessingPipeline {
    /// Create a new pipeline.
    pub fn new(sector: SectorTemplate, config: PipelineConfig) -> Self {
        let generator = ProcessEventGenerator::new(sector.clone(), config.generator.clone());

        Self {
            sector,
            config,
            generator,
            running: Arc::new(AtomicBool::new(false)),
            event_count: Arc::new(AtomicU64::new(0)),
            batch_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get the sector template.
    pub fn sector(&self) -> &SectorTemplate {
        &self.sector
    }

    /// Get the configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Check if pipeline is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Start the pipeline.
    pub fn start(&self) {
        self.running.store(true, Ordering::Relaxed);
    }

    /// Stop the pipeline.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Generate next batch of events.
    pub fn generate_batch(&mut self) -> Vec<GpuObjectEvent> {
        if !self.is_running() {
            return Vec::new();
        }

        let events = self.generator.generate_batch(self.config.gpu_batch_size);
        self.event_count
            .fetch_add(events.len() as u64, Ordering::Relaxed);
        self.batch_count.fetch_add(1, Ordering::Relaxed);
        events
    }

    /// Get current statistics.
    pub fn stats(&self) -> PipelineStats {
        let generator_stats = self.generator.stats();
        PipelineStats {
            events_processed: self.event_count.load(Ordering::Relaxed),
            batches_processed: self.batch_count.load(Ordering::Relaxed),
            dfg_updates: generator_stats.total_events,
            patterns_detected: generator_stats.bottleneck_count
                + generator_stats.rework_count
                + generator_stats.long_running_count,
            conformance_checks: generator_stats.cases_completed,
            avg_batch_time_us: 0.0, // Will be updated by actual processing
        }
    }

    /// Get generator statistics.
    pub fn generator_stats(&self) -> &crate::fabric::GeneratorStats {
        self.generator.stats()
    }

    /// Get estimated throughput.
    pub fn throughput(&self) -> f32 {
        self.generator.throughput()
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.event_count.store(0, Ordering::Relaxed);
        self.batch_count.store(0, Ordering::Relaxed);
    }

    /// Change sector template.
    pub fn set_sector(&mut self, sector: SectorTemplate) {
        self.sector = sector.clone();
        self.generator = ProcessEventGenerator::new(sector, self.config.generator.clone());
        self.reset_stats();
    }

    /// Update generator configuration.
    pub fn set_generator_config(&mut self, config: GeneratorConfig) {
        self.config.generator = config.clone();
        self.generator = ProcessEventGenerator::new(self.sector.clone(), config);
    }
}

/// Pipeline builder for convenient setup.
#[derive(Debug, Default)]
pub struct PipelineBuilder {
    sector: Option<SectorTemplate>,
    config: PipelineConfig,
}

impl PipelineBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set sector template.
    pub fn with_sector(mut self, sector: SectorTemplate) -> Self {
        self.sector = Some(sector);
        self
    }

    /// Set events per second.
    pub fn with_events_per_second(mut self, eps: u32) -> Self {
        self.config.generator.events_per_second = eps;
        self
    }

    /// Set GPU batch size.
    pub fn with_gpu_batch_size(mut self, size: usize) -> Self {
        self.config.gpu_batch_size = size;
        self
    }

    /// Enable/disable DFG construction.
    pub fn with_dfg(mut self, enabled: bool) -> Self {
        self.config.enable_dfg = enabled;
        self
    }

    /// Enable/disable pattern detection.
    pub fn with_patterns(mut self, enabled: bool) -> Self {
        self.config.enable_patterns = enabled;
        self
    }

    /// Enable/disable conformance checking.
    pub fn with_conformance(mut self, enabled: bool) -> Self {
        self.config.enable_conformance = enabled;
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> ProcessingPipeline {
        let sector = self.sector.unwrap_or_default();
        ProcessingPipeline::new(sector, self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::{HealthcareConfig, ManufacturingConfig};

    #[test]
    fn test_pipeline_creation() {
        let pipeline = PipelineBuilder::new()
            .with_sector(SectorTemplate::Healthcare(HealthcareConfig::default()))
            .with_events_per_second(10000)
            .build();

        assert!(!pipeline.is_running());
        assert_eq!(pipeline.sector().name(), "Healthcare");
    }

    #[test]
    fn test_pipeline_start_stop() {
        let pipeline =
            ProcessingPipeline::new(SectorTemplate::default(), PipelineConfig::default());

        assert!(!pipeline.is_running());
        pipeline.start();
        assert!(pipeline.is_running());
        pipeline.stop();
        assert!(!pipeline.is_running());
    }

    #[test]
    fn test_generate_batch() {
        let mut pipeline = PipelineBuilder::new()
            .with_sector(SectorTemplate::Manufacturing(ManufacturingConfig::default()))
            .with_gpu_batch_size(100)
            .build();

        pipeline.start();
        let batch = pipeline.generate_batch();
        assert!(!batch.is_empty());

        let stats = pipeline.stats();
        assert!(stats.events_processed > 0);
    }
}
