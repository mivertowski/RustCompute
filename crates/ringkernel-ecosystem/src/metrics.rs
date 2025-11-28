//! Prometheus metrics integration for RingKernel.
//!
//! This module provides Prometheus metrics export for monitoring
//! RingKernel performance in production deployments.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ecosystem::metrics::{MetricsRegistry, init_metrics};
//!
//! let registry = init_metrics()?;
//! registry.record_message_sent("kernel1");
//! ```

use parking_lot::RwLock;
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec,
    IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry, TextEncoder,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::error::{EcosystemError, Result};

/// Default latency buckets in seconds.
const DEFAULT_LATENCY_BUCKETS: &[f64] = &[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0];

/// Default size buckets in bytes.
const DEFAULT_SIZE_BUCKETS: &[f64] = &[
    64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1048576.0,
];

/// Configuration for metrics collection.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Prefix for all metric names.
    pub prefix: String,
    /// Custom latency buckets.
    pub latency_buckets: Vec<f64>,
    /// Custom size buckets.
    pub size_buckets: Vec<f64>,
    /// Enable per-kernel metrics.
    pub per_kernel_metrics: bool,
    /// Enable histogram metrics (more expensive).
    pub enable_histograms: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            prefix: "ringkernel".to_string(),
            latency_buckets: DEFAULT_LATENCY_BUCKETS.to_vec(),
            size_buckets: DEFAULT_SIZE_BUCKETS.to_vec(),
            per_kernel_metrics: true,
            enable_histograms: true,
        }
    }
}

/// RingKernel metrics registry.
pub struct MetricsRegistry {
    /// Prometheus registry.
    registry: Registry,
    /// Configuration.
    config: MetricsConfig,
    /// Core metrics.
    core: CoreMetrics,
    /// Kernel-specific metrics.
    kernel_metrics: RwLock<HashMap<String, KernelMetrics>>,
}

/// Core metrics shared across all kernels.
struct CoreMetrics {
    /// Total messages sent.
    messages_sent: IntCounterVec,
    /// Total messages received.
    messages_received: IntCounterVec,
    /// Total messages dropped.
    messages_dropped: IntCounterVec,
    /// Message latency histogram.
    message_latency: HistogramVec,
    /// Message size histogram.
    message_size: HistogramVec,
    /// Active kernels gauge.
    active_kernels: IntGauge,
    /// Total kernels launched.
    kernels_launched: IntCounter,
    /// Memory usage gauge.
    memory_usage: IntGaugeVec,
    /// Queue depth gauge.
    queue_depth: IntGaugeVec,
    /// Errors counter.
    errors: IntCounterVec,
}

/// Per-kernel metrics.
pub struct KernelMetrics {
    /// Kernel ID.
    pub kernel_id: String,
    /// Messages processed.
    pub messages_processed: IntCounter,
    /// Average latency gauge.
    pub avg_latency: Gauge,
    /// Current queue depth.
    pub queue_depth: IntGauge,
    /// Kernel state (0=inactive, 1=active, 2=suspended, 3=error).
    pub state: IntGauge,
}

impl MetricsRegistry {
    /// Create a new metrics registry with default configuration.
    pub fn new() -> Result<Self> {
        Self::with_config(MetricsConfig::default())
    }

    /// Create a new metrics registry with custom configuration.
    pub fn with_config(config: MetricsConfig) -> Result<Self> {
        let registry = Registry::new();
        let core = create_core_metrics(&registry, &config)?;

        Ok(Self {
            registry,
            config,
            core,
            kernel_metrics: RwLock::new(HashMap::new()),
        })
    }

    /// Get the underlying Prometheus registry.
    pub fn prometheus_registry(&self) -> &Registry {
        &self.registry
    }

    /// Record a message sent event.
    pub fn record_message_sent(&self, kernel_id: &str) {
        self.core
            .messages_sent
            .with_label_values(&[kernel_id])
            .inc();
    }

    /// Record a message received event.
    pub fn record_message_received(&self, kernel_id: &str) {
        self.core
            .messages_received
            .with_label_values(&[kernel_id])
            .inc();
    }

    /// Record a message dropped event.
    pub fn record_message_dropped(&self, kernel_id: &str, reason: &str) {
        self.core
            .messages_dropped
            .with_label_values(&[kernel_id, reason])
            .inc();
    }

    /// Record message latency.
    pub fn record_message_latency(&self, kernel_id: &str, duration: Duration) {
        if self.config.enable_histograms {
            self.core
                .message_latency
                .with_label_values(&[kernel_id])
                .observe(duration.as_secs_f64());
        }
    }

    /// Record message size.
    pub fn record_message_size(&self, kernel_id: &str, size: usize) {
        if self.config.enable_histograms {
            self.core
                .message_size
                .with_label_values(&[kernel_id])
                .observe(size as f64);
        }
    }

    /// Set active kernel count.
    pub fn set_active_kernels(&self, count: i64) {
        self.core.active_kernels.set(count);
    }

    /// Record kernel launch.
    pub fn record_kernel_launch(&self) {
        self.core.kernels_launched.inc();
        self.core.active_kernels.inc();
    }

    /// Record kernel shutdown.
    pub fn record_kernel_shutdown(&self) {
        self.core.active_kernels.dec();
    }

    /// Set memory usage for a kernel.
    pub fn set_memory_usage(&self, kernel_id: &str, memory_type: &str, bytes: i64) {
        self.core
            .memory_usage
            .with_label_values(&[kernel_id, memory_type])
            .set(bytes);
    }

    /// Set queue depth for a kernel.
    pub fn set_queue_depth(&self, kernel_id: &str, queue_type: &str, depth: i64) {
        self.core
            .queue_depth
            .with_label_values(&[kernel_id, queue_type])
            .set(depth);
    }

    /// Record an error.
    pub fn record_error(&self, kernel_id: &str, error_type: &str) {
        self.core
            .errors
            .with_label_values(&[kernel_id, error_type])
            .inc();
    }

    /// Register a kernel for per-kernel metrics.
    pub fn register_kernel(&self, kernel_id: &str) -> Result<()> {
        if !self.config.per_kernel_metrics {
            return Ok(());
        }

        let mut metrics = self.kernel_metrics.write();
        if metrics.contains_key(kernel_id) {
            return Ok(());
        }

        let prefix = &self.config.prefix;

        let messages_processed = IntCounter::new(
            format!("{}_kernel_{}_messages_processed", prefix, kernel_id),
            format!("Messages processed by kernel {}", kernel_id),
        )
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;

        let avg_latency = Gauge::new(
            format!("{}_kernel_{}_avg_latency_seconds", prefix, kernel_id),
            format!("Average latency for kernel {}", kernel_id),
        )
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;

        let queue_depth = IntGauge::new(
            format!("{}_kernel_{}_queue_depth", prefix, kernel_id),
            format!("Queue depth for kernel {}", kernel_id),
        )
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;

        let state = IntGauge::new(
            format!("{}_kernel_{}_state", prefix, kernel_id),
            format!("State of kernel {} (0=inactive, 1=active, 2=suspended, 3=error)", kernel_id),
        )
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;

        self.registry
            .register(Box::new(messages_processed.clone()))
            .map_err(|e| EcosystemError::Internal(e.to_string()))?;
        self.registry
            .register(Box::new(avg_latency.clone()))
            .map_err(|e| EcosystemError::Internal(e.to_string()))?;
        self.registry
            .register(Box::new(queue_depth.clone()))
            .map_err(|e| EcosystemError::Internal(e.to_string()))?;
        self.registry
            .register(Box::new(state.clone()))
            .map_err(|e| EcosystemError::Internal(e.to_string()))?;

        metrics.insert(
            kernel_id.to_string(),
            KernelMetrics {
                kernel_id: kernel_id.to_string(),
                messages_processed,
                avg_latency,
                queue_depth,
                state,
            },
        );

        Ok(())
    }

    /// Unregister a kernel.
    pub fn unregister_kernel(&self, kernel_id: &str) {
        self.kernel_metrics.write().remove(kernel_id);
    }

    /// Update kernel-specific metrics.
    pub fn update_kernel_metrics(&self, kernel_id: &str, processed: u64, latency: f64, depth: i64) {
        let metrics = self.kernel_metrics.read();
        if let Some(km) = metrics.get(kernel_id) {
            km.messages_processed.inc_by(processed);
            km.avg_latency.set(latency);
            km.queue_depth.set(depth);
        }
    }

    /// Set kernel state.
    pub fn set_kernel_state(&self, kernel_id: &str, state: KernelState) {
        let metrics = self.kernel_metrics.read();
        if let Some(km) = metrics.get(kernel_id) {
            km.state.set(state as i64);
        }
    }

    /// Export metrics as Prometheus text format.
    pub fn export(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder
            .encode(&metric_families, &mut buffer)
            .map_err(|e| EcosystemError::Internal(e.to_string()))?;
        String::from_utf8(buffer).map_err(|e| EcosystemError::Internal(e.to_string()))
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new().expect("Failed to create default metrics registry")
    }
}

/// Kernel state for metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum KernelState {
    /// Kernel is inactive.
    Inactive = 0,
    /// Kernel is active.
    Active = 1,
    /// Kernel is suspended.
    Suspended = 2,
    /// Kernel is in error state.
    Error = 3,
}

fn create_core_metrics(registry: &Registry, config: &MetricsConfig) -> Result<CoreMetrics> {
    let prefix = &config.prefix;

    let messages_sent = IntCounterVec::new(
        Opts::new(
            format!("{}_messages_sent_total", prefix),
            "Total messages sent to kernels",
        ),
        &["kernel_id"],
    )
    .map_err(|e| EcosystemError::Internal(e.to_string()))?;

    let messages_received = IntCounterVec::new(
        Opts::new(
            format!("{}_messages_received_total", prefix),
            "Total messages received from kernels",
        ),
        &["kernel_id"],
    )
    .map_err(|e| EcosystemError::Internal(e.to_string()))?;

    let messages_dropped = IntCounterVec::new(
        Opts::new(
            format!("{}_messages_dropped_total", prefix),
            "Total messages dropped",
        ),
        &["kernel_id", "reason"],
    )
    .map_err(|e| EcosystemError::Internal(e.to_string()))?;

    let message_latency = HistogramVec::new(
        HistogramOpts::new(
            format!("{}_message_latency_seconds", prefix),
            "Message round-trip latency",
        )
        .buckets(config.latency_buckets.clone()),
        &["kernel_id"],
    )
    .map_err(|e| EcosystemError::Internal(e.to_string()))?;

    let message_size = HistogramVec::new(
        HistogramOpts::new(
            format!("{}_message_size_bytes", prefix),
            "Message size in bytes",
        )
        .buckets(config.size_buckets.clone()),
        &["kernel_id"],
    )
    .map_err(|e| EcosystemError::Internal(e.to_string()))?;

    let active_kernels = IntGauge::new(
        format!("{}_active_kernels", prefix),
        "Number of active kernels",
    )
    .map_err(|e| EcosystemError::Internal(e.to_string()))?;

    let kernels_launched = IntCounter::new(
        format!("{}_kernels_launched_total", prefix),
        "Total kernels launched",
    )
    .map_err(|e| EcosystemError::Internal(e.to_string()))?;

    let memory_usage = IntGaugeVec::new(
        Opts::new(
            format!("{}_memory_usage_bytes", prefix),
            "Memory usage in bytes",
        ),
        &["kernel_id", "memory_type"],
    )
    .map_err(|e| EcosystemError::Internal(e.to_string()))?;

    let queue_depth = IntGaugeVec::new(
        Opts::new(format!("{}_queue_depth", prefix), "Current queue depth"),
        &["kernel_id", "queue_type"],
    )
    .map_err(|e| EcosystemError::Internal(e.to_string()))?;

    let errors = IntCounterVec::new(
        Opts::new(format!("{}_errors_total", prefix), "Total errors"),
        &["kernel_id", "error_type"],
    )
    .map_err(|e| EcosystemError::Internal(e.to_string()))?;

    // Register all metrics
    registry
        .register(Box::new(messages_sent.clone()))
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;
    registry
        .register(Box::new(messages_received.clone()))
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;
    registry
        .register(Box::new(messages_dropped.clone()))
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;
    registry
        .register(Box::new(message_latency.clone()))
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;
    registry
        .register(Box::new(message_size.clone()))
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;
    registry
        .register(Box::new(active_kernels.clone()))
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;
    registry
        .register(Box::new(kernels_launched.clone()))
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;
    registry
        .register(Box::new(memory_usage.clone()))
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;
    registry
        .register(Box::new(queue_depth.clone()))
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;
    registry
        .register(Box::new(errors.clone()))
        .map_err(|e| EcosystemError::Internal(e.to_string()))?;

    Ok(CoreMetrics {
        messages_sent,
        messages_received,
        messages_dropped,
        message_latency,
        message_size,
        active_kernels,
        kernels_launched,
        memory_usage,
        queue_depth,
        errors,
    })
}

/// Initialize default metrics registry.
pub fn init_metrics() -> Result<Arc<MetricsRegistry>> {
    Ok(Arc::new(MetricsRegistry::new()?))
}

/// Initialize metrics with custom configuration.
pub fn init_metrics_with_config(config: MetricsConfig) -> Result<Arc<MetricsRegistry>> {
    Ok(Arc::new(MetricsRegistry::with_config(config)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_config_default() {
        let config = MetricsConfig::default();
        assert_eq!(config.prefix, "ringkernel");
        assert!(config.per_kernel_metrics);
        assert!(config.enable_histograms);
    }

    #[test]
    fn test_metrics_registry_creation() {
        let registry = MetricsRegistry::new().unwrap();
        assert_eq!(registry.config.prefix, "ringkernel");
    }

    #[test]
    fn test_record_metrics() {
        let registry = MetricsRegistry::new().unwrap();

        registry.record_message_sent("kernel1");
        registry.record_message_received("kernel1");
        registry.record_message_dropped("kernel1", "queue_full");
        registry.record_message_latency("kernel1", Duration::from_micros(500));
        registry.record_message_size("kernel1", 1024);
        registry.record_error("kernel1", "timeout");
    }

    #[test]
    fn test_kernel_metrics_registration() {
        let registry = MetricsRegistry::new().unwrap();

        registry.register_kernel("kernel1").unwrap();
        registry.update_kernel_metrics("kernel1", 100, 0.001, 50);
        registry.set_kernel_state("kernel1", KernelState::Active);
        registry.unregister_kernel("kernel1");
    }

    #[test]
    fn test_export_metrics() {
        let registry = MetricsRegistry::new().unwrap();
        registry.record_message_sent("test");

        let output = registry.export().unwrap();
        assert!(output.contains("ringkernel_messages_sent_total"));
    }

    #[test]
    fn test_kernel_state_values() {
        assert_eq!(KernelState::Inactive as i64, 0);
        assert_eq!(KernelState::Active as i64, 1);
        assert_eq!(KernelState::Suspended as i64, 2);
        assert_eq!(KernelState::Error as i64, 3);
    }
}
