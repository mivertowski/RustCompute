//! Real-time telemetry pipeline for streaming metrics.
//!
//! This module provides a real-time metrics pipeline that allows
//! subscribers to receive continuous updates about kernel performance.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

use crate::runtime::KernelId;
use crate::telemetry::{KernelMetrics, LatencyHistogram, TelemetryBuffer};

/// Configuration for the telemetry pipeline.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Collection interval in milliseconds.
    pub collection_interval_ms: u64,
    /// Maximum number of samples to retain in history.
    pub max_history_samples: usize,
    /// Channel buffer size for subscribers.
    pub channel_buffer_size: usize,
    /// Whether to enable latency histograms.
    pub enable_histograms: bool,
    /// Alert threshold for message drop rate (0.0-1.0).
    pub drop_rate_alert_threshold: f64,
    /// Alert threshold for average latency in microseconds.
    pub latency_alert_threshold_us: u64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: 100,
            max_history_samples: 1000,
            channel_buffer_size: 256,
            enable_histograms: true,
            drop_rate_alert_threshold: 0.01,    // 1%
            latency_alert_threshold_us: 10_000, // 10ms
        }
    }
}

/// A telemetry event that is broadcast to subscribers.
#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    /// Periodic metrics snapshot.
    MetricsSnapshot(MetricsSnapshot),
    /// Alert when thresholds are exceeded.
    Alert(TelemetryAlert),
    /// Kernel state change event.
    KernelStateChange {
        /// Kernel identifier.
        kernel_id: KernelId,
        /// Previous state.
        previous: String,
        /// New state.
        new: String,
    },
}

/// A snapshot of metrics at a point in time.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Timestamp when snapshot was taken.
    pub timestamp: Instant,
    /// Metrics for each kernel.
    pub kernel_metrics: HashMap<KernelId, KernelMetrics>,
    /// Aggregate metrics across all kernels.
    pub aggregate: AggregateMetrics,
}

/// Aggregate metrics across all kernels.
#[derive(Debug, Clone, Default)]
pub struct AggregateMetrics {
    /// Total messages processed across all kernels.
    pub total_messages_processed: u64,
    /// Total messages dropped across all kernels.
    pub total_messages_dropped: u64,
    /// Average latency across all kernels.
    pub avg_latency_us: f64,
    /// Minimum latency across all kernels.
    pub min_latency_us: u64,
    /// Maximum latency across all kernels.
    pub max_latency_us: u64,
    /// Total throughput (messages/sec).
    pub throughput: f64,
    /// Number of active kernels.
    pub active_kernels: usize,
    /// Total GPU memory used.
    pub total_gpu_memory: u64,
}

/// An alert when telemetry thresholds are exceeded.
#[derive(Debug, Clone)]
pub struct TelemetryAlert {
    /// Alert severity.
    pub severity: AlertSeverity,
    /// Alert type.
    pub alert_type: AlertType,
    /// Human-readable message.
    pub message: String,
    /// Related kernel (if applicable).
    pub kernel_id: Option<KernelId>,
    /// Timestamp when alert was generated.
    pub timestamp: Instant,
}

/// Alert severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    /// Informational.
    Info,
    /// Warning - may indicate a problem.
    Warning,
    /// Error - action required.
    Error,
    /// Critical - immediate action required.
    Critical,
}

/// Type of alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertType {
    /// Message drop rate exceeded threshold.
    HighDropRate,
    /// Latency exceeded threshold.
    HighLatency,
    /// Queue approaching capacity.
    QueueNearFull,
    /// Kernel error occurred.
    KernelError,
    /// Memory pressure detected.
    MemoryPressure,
}

/// Real-time telemetry pipeline.
///
/// Collects metrics from kernels and broadcasts them to subscribers.
pub struct TelemetryPipeline {
    /// Configuration.
    config: TelemetryConfig,
    /// Running state.
    running: Arc<AtomicBool>,
    /// Broadcast sender for events.
    sender: broadcast::Sender<TelemetryEvent>,
    /// Registered metrics sources.
    sources: Arc<RwLock<HashMap<KernelId, Arc<dyn MetricsSource>>>>,
    /// Historical snapshots.
    history: Arc<RwLock<Vec<MetricsSnapshot>>>,
    /// Pipeline start time.
    start_time: Instant,
    /// Sequence number for events.
    #[allow(dead_code)]
    sequence: AtomicU64,
}

/// Trait for providing metrics from a kernel.
pub trait MetricsSource: Send + Sync {
    /// Get current metrics.
    fn get_metrics(&self) -> KernelMetrics;

    /// Get kernel ID.
    fn kernel_id(&self) -> &KernelId;

    /// Check if kernel is active.
    fn is_active(&self) -> bool;
}

impl TelemetryPipeline {
    /// Create a new telemetry pipeline.
    pub fn new(config: TelemetryConfig) -> Self {
        let (sender, _) = broadcast::channel(config.channel_buffer_size);

        Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            sender,
            sources: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
            sequence: AtomicU64::new(0),
        }
    }

    /// Subscribe to telemetry events.
    pub fn subscribe(&self) -> broadcast::Receiver<TelemetryEvent> {
        self.sender.subscribe()
    }

    /// Register a metrics source.
    pub fn register_source(&self, source: Arc<dyn MetricsSource>) {
        let kernel_id = source.kernel_id().clone();
        self.sources.write().insert(kernel_id, source);
    }

    /// Unregister a metrics source.
    pub fn unregister_source(&self, kernel_id: &KernelId) {
        self.sources.write().remove(kernel_id);
    }

    /// Start the telemetry collection loop.
    pub fn start(&self) -> tokio::task::JoinHandle<()> {
        self.running.store(true, Ordering::Release);

        let running = Arc::clone(&self.running);
        let sources = Arc::clone(&self.sources);
        let history = Arc::clone(&self.history);
        let sender = self.sender.clone();
        let config = self.config.clone();
        let start_time = self.start_time;

        tokio::spawn(async move {
            let interval = Duration::from_millis(config.collection_interval_ms);

            while running.load(Ordering::Acquire) {
                // Collect metrics
                let snapshot = Self::collect_snapshot(&sources, start_time, &config);

                // Check for alerts
                let alerts = Self::check_alerts(&snapshot, &config);

                // Store in history
                {
                    let mut hist = history.write();
                    hist.push(snapshot.clone());
                    if hist.len() > config.max_history_samples {
                        hist.remove(0);
                    }
                }

                // Broadcast snapshot
                let _ = sender.send(TelemetryEvent::MetricsSnapshot(snapshot));

                // Broadcast alerts
                for alert in alerts {
                    let _ = sender.send(TelemetryEvent::Alert(alert));
                }

                tokio::time::sleep(interval).await;
            }
        })
    }

    /// Stop the telemetry collection loop.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    /// Get the latest snapshot.
    pub fn latest_snapshot(&self) -> Option<MetricsSnapshot> {
        self.history.read().last().cloned()
    }

    /// Get historical snapshots.
    pub fn history(&self) -> Vec<MetricsSnapshot> {
        self.history.read().clone()
    }

    /// Get aggregate metrics over a time range.
    pub fn aggregate_over(&self, duration: Duration) -> Option<AggregateMetrics> {
        let history = self.history.read();
        let cutoff = Instant::now() - duration;

        let relevant: Vec<_> = history.iter().filter(|s| s.timestamp >= cutoff).collect();

        if relevant.is_empty() {
            return None;
        }

        let mut aggregate = AggregateMetrics::default();

        for snapshot in &relevant {
            aggregate.total_messages_processed += snapshot.aggregate.total_messages_processed;
            aggregate.total_messages_dropped += snapshot.aggregate.total_messages_dropped;
            aggregate.min_latency_us = aggregate
                .min_latency_us
                .min(snapshot.aggregate.min_latency_us);
            aggregate.max_latency_us = aggregate
                .max_latency_us
                .max(snapshot.aggregate.max_latency_us);
        }

        // Average metrics
        let count = relevant.len() as f64;
        aggregate.avg_latency_us = relevant
            .iter()
            .map(|s| s.aggregate.avg_latency_us)
            .sum::<f64>()
            / count;
        aggregate.throughput = relevant.iter().map(|s| s.aggregate.throughput).sum::<f64>() / count;
        aggregate.active_kernels = relevant
            .last()
            .map(|s| s.aggregate.active_kernels)
            .unwrap_or(0);
        aggregate.total_gpu_memory = relevant
            .last()
            .map(|s| s.aggregate.total_gpu_memory)
            .unwrap_or(0);

        Some(aggregate)
    }

    /// Collect a metrics snapshot.
    fn collect_snapshot(
        sources: &RwLock<HashMap<KernelId, Arc<dyn MetricsSource>>>,
        start_time: Instant,
        _config: &TelemetryConfig,
    ) -> MetricsSnapshot {
        let sources = sources.read();
        let mut kernel_metrics = HashMap::new();
        let mut aggregate = AggregateMetrics::default();

        let elapsed = start_time.elapsed().as_secs_f64();

        for (kernel_id, source) in sources.iter() {
            if source.is_active() {
                aggregate.active_kernels += 1;
            }

            let metrics = source.get_metrics();

            aggregate.total_messages_processed += metrics.telemetry.messages_processed;
            aggregate.total_messages_dropped += metrics.telemetry.messages_dropped;
            aggregate.total_gpu_memory += metrics.gpu_memory_used;

            if metrics.telemetry.min_latency_us < aggregate.min_latency_us
                || aggregate.min_latency_us == 0
            {
                aggregate.min_latency_us = metrics.telemetry.min_latency_us;
            }
            if metrics.telemetry.max_latency_us > aggregate.max_latency_us {
                aggregate.max_latency_us = metrics.telemetry.max_latency_us;
            }

            kernel_metrics.insert(kernel_id.clone(), metrics);
        }

        // Calculate averages
        if !kernel_metrics.is_empty() {
            aggregate.avg_latency_us = kernel_metrics
                .values()
                .map(|m| m.telemetry.avg_latency_us())
                .sum::<f64>()
                / kernel_metrics.len() as f64;

            if elapsed > 0.0 {
                aggregate.throughput = aggregate.total_messages_processed as f64 / elapsed;
            }
        }

        MetricsSnapshot {
            timestamp: Instant::now(),
            kernel_metrics,
            aggregate,
        }
    }

    /// Check for alert conditions.
    fn check_alerts(snapshot: &MetricsSnapshot, config: &TelemetryConfig) -> Vec<TelemetryAlert> {
        let mut alerts = Vec::new();

        for (kernel_id, metrics) in &snapshot.kernel_metrics {
            // Check drop rate
            let drop_rate = metrics.telemetry.drop_rate();
            if drop_rate > config.drop_rate_alert_threshold {
                alerts.push(TelemetryAlert {
                    severity: if drop_rate > 0.1 {
                        AlertSeverity::Critical
                    } else if drop_rate > 0.05 {
                        AlertSeverity::Error
                    } else {
                        AlertSeverity::Warning
                    },
                    alert_type: AlertType::HighDropRate,
                    message: format!(
                        "Kernel {} drop rate is {:.2}%",
                        kernel_id,
                        drop_rate * 100.0
                    ),
                    kernel_id: Some(kernel_id.clone()),
                    timestamp: Instant::now(),
                });
            }

            // Check latency
            let avg_latency = metrics.telemetry.avg_latency_us() as u64;
            if avg_latency > config.latency_alert_threshold_us {
                alerts.push(TelemetryAlert {
                    severity: if avg_latency > config.latency_alert_threshold_us * 10 {
                        AlertSeverity::Critical
                    } else if avg_latency > config.latency_alert_threshold_us * 5 {
                        AlertSeverity::Error
                    } else {
                        AlertSeverity::Warning
                    },
                    alert_type: AlertType::HighLatency,
                    message: format!("Kernel {} average latency is {}µs", kernel_id, avg_latency),
                    kernel_id: Some(kernel_id.clone()),
                    timestamp: Instant::now(),
                });
            }

            // Check for errors
            if metrics.telemetry.last_error != 0 {
                alerts.push(TelemetryAlert {
                    severity: AlertSeverity::Error,
                    alert_type: AlertType::KernelError,
                    message: format!(
                        "Kernel {} reported error code {}",
                        kernel_id, metrics.telemetry.last_error
                    ),
                    kernel_id: Some(kernel_id.clone()),
                    timestamp: Instant::now(),
                });
            }
        }

        alerts
    }
}

/// Metrics collector that aggregates metrics from multiple kernels.
#[derive(Default)]
pub struct MetricsCollector {
    /// Per-kernel telemetry.
    kernel_telemetry: RwLock<HashMap<KernelId, TelemetryBuffer>>,
    /// Per-kernel histograms.
    kernel_histograms: RwLock<HashMap<KernelId, LatencyHistogram>>,
    /// Start time.
    start_time: RwLock<Option<Instant>>,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            kernel_telemetry: RwLock::new(HashMap::new()),
            kernel_histograms: RwLock::new(HashMap::new()),
            start_time: RwLock::new(Some(Instant::now())),
        }
    }

    /// Record a message processed event.
    pub fn record_message_processed(&self, kernel_id: &KernelId, latency_us: u64) {
        let mut telemetry = self.kernel_telemetry.write();
        let entry = telemetry.entry(kernel_id.clone()).or_default();

        entry.messages_processed += 1;
        entry.total_latency_us += latency_us;
        entry.min_latency_us = entry.min_latency_us.min(latency_us);
        entry.max_latency_us = entry.max_latency_us.max(latency_us);

        // Record in histogram
        let mut histograms = self.kernel_histograms.write();
        let histogram = histograms.entry(kernel_id.clone()).or_default();
        histogram.record(latency_us);
    }

    /// Record a message dropped event.
    pub fn record_message_dropped(&self, kernel_id: &KernelId) {
        let mut telemetry = self.kernel_telemetry.write();
        let entry = telemetry.entry(kernel_id.clone()).or_default();
        entry.messages_dropped += 1;
    }

    /// Record an error.
    pub fn record_error(&self, kernel_id: &KernelId, error_code: u32) {
        let mut telemetry = self.kernel_telemetry.write();
        let entry = telemetry.entry(kernel_id.clone()).or_default();
        entry.last_error = error_code;
    }

    /// Get telemetry for a kernel.
    pub fn get_telemetry(&self, kernel_id: &KernelId) -> Option<TelemetryBuffer> {
        self.kernel_telemetry.read().get(kernel_id).copied()
    }

    /// Get histogram for a kernel.
    pub fn get_histogram(&self, kernel_id: &KernelId) -> Option<LatencyHistogram> {
        self.kernel_histograms.read().get(kernel_id).cloned()
    }

    /// Get aggregate telemetry across all kernels.
    pub fn get_aggregate(&self) -> TelemetryBuffer {
        let telemetry = self.kernel_telemetry.read();
        let mut aggregate = TelemetryBuffer::new();

        for buffer in telemetry.values() {
            aggregate.merge(buffer);
        }

        aggregate
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.kernel_telemetry.write().clear();
        self.kernel_histograms.write().clear();
        *self.start_time.write() = Some(Instant::now());
    }

    /// Get elapsed time since start.
    pub fn elapsed(&self) -> Duration {
        self.start_time
            .read()
            .map(|t| t.elapsed())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_config_default() {
        let config = TelemetryConfig::default();
        assert_eq!(config.collection_interval_ms, 100);
        assert_eq!(config.max_history_samples, 1000);
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        let kernel_id = KernelId::new("test");

        collector.record_message_processed(&kernel_id, 100);
        collector.record_message_processed(&kernel_id, 200);
        collector.record_message_dropped(&kernel_id);

        let telemetry = collector.get_telemetry(&kernel_id).unwrap();
        assert_eq!(telemetry.messages_processed, 2);
        assert_eq!(telemetry.messages_dropped, 1);
        assert_eq!(telemetry.min_latency_us, 100);
        assert_eq!(telemetry.max_latency_us, 200);
    }

    #[test]
    fn test_aggregate_metrics() {
        let collector = MetricsCollector::new();

        let kernel1 = KernelId::new("kernel1");
        let kernel2 = KernelId::new("kernel2");

        collector.record_message_processed(&kernel1, 100);
        collector.record_message_processed(&kernel2, 200);

        let aggregate = collector.get_aggregate();
        assert_eq!(aggregate.messages_processed, 2);
    }
}
