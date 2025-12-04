//! Streaming data pipeline connecting generation to analysis.
//!
//! The pipeline orchestrates data flow from generation through
//! transformation and analysis, emitting events for visualization.

use super::{
    AccountTypeInfo, AnomalyInjectionConfig, AnomalyInjector, ChartOfAccountsTemplate,
    CompanyArchetype, GeneratorConfig, TransactionGenerator,
};
use crate::models::{
    AccountingNetwork, Decimal128, FraudPattern, GaapViolation, HybridTimestamp, NetworkSnapshot,
    TemporalAlert, TransactionFlow,
};
use std::time::Duration;
use tokio::sync::broadcast;
use uuid::Uuid;

/// Configuration for the data pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// How often to emit batches
    pub tick_duration: Duration,
    /// Entries per batch
    pub batch_size: usize,
    /// Channel buffer size
    pub channel_buffer: usize,
    /// Enable anomaly injection
    pub inject_anomalies: bool,
    /// Anomaly injection configuration
    pub anomaly_config: AnomalyInjectionConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            tick_duration: Duration::from_millis(100),
            batch_size: 50,
            channel_buffer: 1000,
            inject_anomalies: true,
            anomaly_config: AnomalyInjectionConfig::default(),
        }
    }
}

impl PipelineConfig {
    /// Create a fast configuration for testing.
    pub fn fast() -> Self {
        Self {
            tick_duration: Duration::from_millis(10),
            batch_size: 100,
            ..Default::default()
        }
    }

    /// Create a slow configuration for educational demos.
    pub fn educational() -> Self {
        Self {
            tick_duration: Duration::from_millis(500),
            batch_size: 5,
            ..Default::default()
        }
    }
}

/// Events emitted by the pipeline.
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    /// New journal entries were generated.
    EntriesGenerated {
        /// Number of entries generated.
        count: usize,
        /// Timestamp of generation.
        timestamp: HybridTimestamp,
    },

    /// Entries were transformed into flows.
    FlowsCreated {
        /// Generated transaction flows.
        flows: Vec<TransactionFlow>,
        /// Timestamp of transformation.
        timestamp: HybridTimestamp,
    },

    /// Network was updated with new data
    NetworkUpdated(NetworkSnapshot),

    /// Anomaly was detected
    AnomalyDetected(Alert),

    /// Fraud pattern was identified
    FraudPatternDetected(FraudPattern),

    /// GAAP violation was found
    GaapViolationDetected(GaapViolation),

    /// Temporal anomaly was detected
    TemporalAnomalyDetected(TemporalAlert),

    /// Pipeline statistics update
    StatsUpdated(PipelineStats),

    /// Pipeline paused
    Paused,

    /// Pipeline resumed
    Resumed,

    /// Pipeline stopped
    Stopped,
}

/// An alert from the analysis engine.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Unique identifier
    pub id: Uuid,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert type
    pub alert_type: String,
    /// Human-readable message
    pub message: String,
    /// Involved account indices
    pub accounts: Vec<u16>,
    /// Amount involved (if applicable)
    pub amount: Option<Decimal128>,
    /// When the alert was raised
    pub timestamp: HybridTimestamp,
}

/// Alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational only
    Info,
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Critical - immediate attention needed
    Critical,
}

impl AlertSeverity {
    /// Get color for this severity.
    pub fn color(&self) -> [u8; 3] {
        match self {
            AlertSeverity::Info => [100, 181, 246],   // Light blue
            AlertSeverity::Low => [255, 235, 59],     // Yellow
            AlertSeverity::Medium => [255, 152, 0],   // Orange
            AlertSeverity::High => [244, 67, 54],     // Red
            AlertSeverity::Critical => [183, 28, 28], // Dark red
        }
    }

    /// Get icon for this severity.
    pub fn icon(&self) -> &'static str {
        match self {
            AlertSeverity::Info => "ℹ️",
            AlertSeverity::Low => "⚠️",
            AlertSeverity::Medium => "🔶",
            AlertSeverity::High => "🔴",
            AlertSeverity::Critical => "🚨",
        }
    }
}

/// Pipeline statistics.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total entries generated
    pub entries_generated: u64,
    /// Total flows created
    pub flows_created: u64,
    /// Total anomalies detected
    pub anomalies_detected: u64,
    /// Entries per second (recent)
    pub entries_per_second: f64,
    /// Flows per second (recent)
    pub flows_per_second: f64,
    /// Method distribution
    pub method_distribution: [u32; 5],
    /// Pipeline running time (seconds)
    pub running_time_seconds: f64,
}

/// The data fabric pipeline.
pub struct DataFabricPipeline {
    /// Entity ID for this pipeline
    entity_id: Uuid,
    /// Transaction generator
    generator: TransactionGenerator,
    /// Anomaly injector
    injector: Option<AnomalyInjector>,
    /// The accounting network being built
    network: AccountingNetwork,
    /// Configuration
    config: PipelineConfig,
    /// Event broadcaster
    event_sender: broadcast::Sender<PipelineEvent>,
    /// Running state
    is_running: bool,
    /// Paused state
    is_paused: bool,
    /// Statistics
    stats: PipelineStats,
    /// Start time
    start_time: Option<std::time::Instant>,
}

impl DataFabricPipeline {
    /// Create a new pipeline.
    pub fn new(
        archetype: CompanyArchetype,
        generator_config: GeneratorConfig,
        pipeline_config: PipelineConfig,
    ) -> Self {
        let entity_id = Uuid::new_v4();
        let generator = TransactionGenerator::new(archetype.clone(), generator_config);

        // Initialize network with chart of accounts
        let coa = ChartOfAccountsTemplate::for_archetype(&archetype);
        let mut network = AccountingNetwork::new(entity_id, 2024, 1);

        // Add accounts from template
        for account_def in &coa.accounts {
            let (node, metadata) = account_def.to_account(network.accounts.len() as u16);
            network.add_account(node, metadata);
        }

        // Create anomaly injector if enabled
        let injector = if pipeline_config.inject_anomalies {
            let mut inj = AnomalyInjector::new(pipeline_config.anomaly_config.clone(), None);

            // Register account types
            for (idx, def) in coa.accounts.iter().enumerate() {
                use crate::models::AccountType;
                let info = AccountTypeInfo {
                    is_asset: def.account_type == AccountType::Asset,
                    is_liability: def.account_type == AccountType::Liability,
                    is_revenue: def.account_type == AccountType::Revenue,
                    is_expense: def.account_type == AccountType::Expense,
                    is_equity: def.account_type == AccountType::Equity,
                    is_cash: def.semantics & crate::models::AccountSemantics::IS_CASH != 0,
                    is_suspense: def.semantics & crate::models::AccountSemantics::IS_SUSPENSE != 0,
                };
                inj.register_account(idx as u16, info);
            }

            Some(inj)
        } else {
            None
        };

        let (event_sender, _) = broadcast::channel(pipeline_config.channel_buffer);

        Self {
            entity_id,
            generator,
            injector,
            network,
            config: pipeline_config,
            event_sender,
            is_running: false,
            is_paused: false,
            stats: PipelineStats::default(),
            start_time: None,
        }
    }

    /// Subscribe to pipeline events.
    pub fn subscribe(&self) -> broadcast::Receiver<PipelineEvent> {
        self.event_sender.subscribe()
    }

    /// Get the current network snapshot.
    pub fn network_snapshot(&self) -> NetworkSnapshot {
        self.network.snapshot()
    }

    /// Get the full network (for analysis).
    pub fn network(&self) -> &AccountingNetwork {
        &self.network
    }

    /// Get mutable network access.
    pub fn network_mut(&mut self) -> &mut AccountingNetwork {
        &mut self.network
    }

    /// Get current statistics.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Check if pipeline is running.
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Check if pipeline is paused.
    pub fn is_paused(&self) -> bool {
        self.is_paused
    }

    /// Process one batch of data.
    /// Returns the generated flows.
    pub fn tick(&mut self) -> Vec<TransactionFlow> {
        if self.is_paused {
            return Vec::new();
        }

        // Record start time if first tick
        if self.start_time.is_none() {
            self.start_time = Some(std::time::Instant::now());
        }

        // Generate entries
        let entries = self.generator.generate_batch(self.config.batch_size);
        let entry_count = entries.len();
        self.stats.entries_generated += entry_count as u64;

        // Emit generation event
        let _ = self.event_sender.send(PipelineEvent::EntriesGenerated {
            count: entry_count,
            timestamp: HybridTimestamp::now(),
        });

        // Process each entry through injection and transformation
        let mut all_flows = Vec::new();

        for entry in entries {
            // Optionally inject anomalies
            let (final_entry, debit_lines, credit_lines, _anomaly_label) =
                if let Some(ref mut injector) = self.injector {
                    let result =
                        injector.process(entry.entry, entry.debit_lines, entry.credit_lines);

                    if result.anomaly_injected {
                        self.stats.anomalies_detected += 1;

                        // Emit anomaly alert
                        if let Some(ref label) = result.anomaly_label {
                            let alert = self.create_alert_from_label(label, &result.entry);
                            let _ = self
                                .event_sender
                                .send(PipelineEvent::AnomalyDetected(alert));
                        }
                    }

                    (
                        result.entry,
                        result.debit_lines,
                        result.credit_lines,
                        result.anomaly_label,
                    )
                } else {
                    (entry.entry, entry.debit_lines, entry.credit_lines, None)
                };

            // Transform to flows
            let flows = self.transform_to_flows(&final_entry, &debit_lines, &credit_lines);
            self.stats.flows_created += flows.len() as u64;

            // Update method distribution
            let method_idx = final_entry.solving_method as usize;
            if method_idx < 5 {
                self.stats.method_distribution[method_idx] += 1;
            }

            // Add flows to network
            for flow in &flows {
                self.network.add_flow(flow.clone());
            }

            all_flows.extend(flows);
        }

        // Emit flows event
        if !all_flows.is_empty() {
            let _ = self.event_sender.send(PipelineEvent::FlowsCreated {
                flows: all_flows.clone(),
                timestamp: HybridTimestamp::now(),
            });
        }

        // Update network statistics
        self.network.update_statistics();

        // Emit network update
        let _ = self
            .event_sender
            .send(PipelineEvent::NetworkUpdated(self.network.snapshot()));

        // Update timing stats
        if let Some(start) = self.start_time {
            self.stats.running_time_seconds = start.elapsed().as_secs_f64();
            if self.stats.running_time_seconds > 0.0 {
                self.stats.entries_per_second =
                    self.stats.entries_generated as f64 / self.stats.running_time_seconds;
                self.stats.flows_per_second =
                    self.stats.flows_created as f64 / self.stats.running_time_seconds;
            }
        }

        all_flows
    }

    /// Transform a journal entry to transaction flows.
    fn transform_to_flows(
        &self,
        entry: &crate::models::JournalEntry,
        debit_lines: &[crate::models::JournalLineItem],
        credit_lines: &[crate::models::JournalLineItem],
    ) -> Vec<TransactionFlow> {
        use crate::models::SolvingMethod;

        match entry.solving_method {
            SolvingMethod::MethodA => {
                // 1-to-1: single flow
                if let (Some(debit), Some(credit)) = (debit_lines.first(), credit_lines.first()) {
                    vec![TransactionFlow::with_provenance(
                        debit.account_index,
                        credit.account_index,
                        debit.amount,
                        entry.id,
                        0,
                        0,
                        entry.posting_date,
                        SolvingMethod::MethodA,
                        1.0,
                    )]
                } else {
                    Vec::new()
                }
            }

            SolvingMethod::MethodB => {
                // n-to-n: match by position (simplified)
                let n = debit_lines.len().min(credit_lines.len());
                (0..n)
                    .map(|i| {
                        TransactionFlow::with_provenance(
                            debit_lines[i].account_index,
                            credit_lines[i].account_index,
                            debit_lines[i].amount,
                            entry.id,
                            i as u16,
                            i as u16,
                            entry.posting_date,
                            SolvingMethod::MethodB,
                            1.0,
                        )
                    })
                    .collect()
            }

            _ => {
                // For C/D/E, create flows from each debit to proportional credits
                let total_credit: f64 = credit_lines.iter().map(|c| c.amount.to_f64()).sum();

                if total_credit == 0.0 {
                    return Vec::new();
                }

                let mut flows = Vec::new();
                for debit in debit_lines {
                    let debit_amount = debit.amount.to_f64();
                    for credit in credit_lines {
                        let credit_ratio = credit.amount.to_f64() / total_credit;
                        let flow_amount = Decimal128::from_f64(debit_amount * credit_ratio);
                        let confidence = entry.average_confidence * credit_ratio as f32;

                        flows.push(TransactionFlow::with_provenance(
                            debit.account_index,
                            credit.account_index,
                            flow_amount,
                            entry.id,
                            0,
                            0,
                            entry.posting_date,
                            entry.solving_method,
                            confidence,
                        ));
                    }
                }
                flows
            }
        }
    }

    /// Create an alert from an anomaly label.
    fn create_alert_from_label(
        &self,
        label: &super::AnomalyLabel,
        entry: &crate::models::JournalEntry,
    ) -> Alert {
        let (alert_type, message, severity) = match label {
            super::AnomalyLabel::FraudPattern(pattern) => {
                let severity = match pattern {
                    crate::models::FraudPatternType::CircularFlow => AlertSeverity::Critical,
                    crate::models::FraudPatternType::HighVelocity => AlertSeverity::High,
                    crate::models::FraudPatternType::ThresholdClustering => AlertSeverity::High,
                    _ => AlertSeverity::Medium,
                };
                (
                    format!("Fraud: {:?}", pattern),
                    pattern.description().to_string(),
                    severity,
                )
            }
            super::AnomalyLabel::GaapViolation(violation) => {
                let severity = match violation.default_severity() {
                    crate::models::ViolationSeverity::Critical => AlertSeverity::Critical,
                    crate::models::ViolationSeverity::High => AlertSeverity::High,
                    crate::models::ViolationSeverity::Medium => AlertSeverity::Medium,
                    crate::models::ViolationSeverity::Low => AlertSeverity::Low,
                };
                (
                    format!("GAAP: {:?}", violation),
                    violation.description().to_string(),
                    severity,
                )
            }
            super::AnomalyLabel::TimingAnomaly(desc) => (
                "Timing".to_string(),
                format!("Timing anomaly: {}", desc),
                AlertSeverity::Medium,
            ),
            super::AnomalyLabel::AmountAnomaly(desc) => (
                "Amount".to_string(),
                format!("Amount anomaly: {}", desc),
                AlertSeverity::Medium,
            ),
        };

        Alert {
            id: Uuid::new_v4(),
            severity,
            alert_type,
            message,
            accounts: vec![],
            amount: Some(entry.total_debits),
            timestamp: entry.posting_date,
        }
    }

    /// Pause the pipeline.
    pub fn pause(&mut self) {
        self.is_paused = true;
        let _ = self.event_sender.send(PipelineEvent::Paused);
    }

    /// Resume the pipeline.
    pub fn resume(&mut self) {
        self.is_paused = false;
        let _ = self.event_sender.send(PipelineEvent::Resumed);
    }

    /// Stop the pipeline.
    pub fn stop(&mut self) {
        self.is_running = false;
        let _ = self.event_sender.send(PipelineEvent::Stopped);
    }

    /// Reset the pipeline (clear network and stats).
    pub fn reset(&mut self) {
        self.network = AccountingNetwork::new(self.entity_id, 2024, 1);

        // Re-add accounts from chart of accounts
        // (In a real implementation, we'd store the CoA template)

        self.stats = PipelineStats::default();
        self.start_time = None;

        if let Some(ref mut injector) = self.injector {
            injector.reset_stats();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let archetype = CompanyArchetype::retail_standard();
        let gen_config = GeneratorConfig::default();
        let pipe_config = PipelineConfig::default();

        let pipeline = DataFabricPipeline::new(archetype, gen_config, pipe_config);
        assert!(!pipeline.is_running());
        assert!(!pipeline.is_paused());
    }

    #[test]
    fn test_pipeline_tick() {
        let archetype = CompanyArchetype::retail_standard();
        let gen_config = GeneratorConfig {
            seed: Some(42),
            ..Default::default()
        };
        let pipe_config = PipelineConfig {
            batch_size: 10,
            inject_anomalies: false,
            ..Default::default()
        };

        let mut pipeline = DataFabricPipeline::new(archetype, gen_config, pipe_config);

        let flows = pipeline.tick();
        assert!(!flows.is_empty());
        assert!(pipeline.stats().entries_generated > 0);
        assert!(pipeline.stats().flows_created > 0);
    }

    #[test]
    fn test_pipeline_pause_resume() {
        let archetype = CompanyArchetype::retail_standard();
        let gen_config = GeneratorConfig::default();
        let pipe_config = PipelineConfig::default();

        let mut pipeline = DataFabricPipeline::new(archetype, gen_config, pipe_config);

        // Generate some data
        pipeline.tick();
        let initial_count = pipeline.stats().entries_generated;

        // Pause - should generate nothing
        pipeline.pause();
        assert!(pipeline.is_paused());
        pipeline.tick();
        assert_eq!(pipeline.stats().entries_generated, initial_count);

        // Resume - should generate again
        pipeline.resume();
        assert!(!pipeline.is_paused());
        pipeline.tick();
        assert!(pipeline.stats().entries_generated > initial_count);
    }

    #[test]
    fn test_pipeline_with_anomalies() {
        let archetype = CompanyArchetype::retail_standard();
        let gen_config = GeneratorConfig {
            seed: Some(42),
            ..Default::default()
        };
        let pipe_config = PipelineConfig {
            batch_size: 100,
            inject_anomalies: true,
            anomaly_config: AnomalyInjectionConfig {
                injection_rate: 0.5, // High rate for testing
                ..Default::default()
            },
            ..Default::default()
        };

        let mut pipeline = DataFabricPipeline::new(archetype, gen_config, pipe_config);

        // Generate multiple batches
        for _ in 0..10 {
            pipeline.tick();
        }

        // Should have detected some anomalies
        assert!(pipeline.stats().anomalies_detected > 0);
    }
}
