//! Observability infrastructure for RingKernel.
//!
//! This module provides production-ready observability features:
//!
//! - **OpenTelemetry Integration** - Distributed tracing and metrics
//! - **Prometheus Exporter** - Metrics in Prometheus exposition format
//! - **Grafana Dashboard** - JSON templates for visualization
//!
//! ## Usage
//!
//! ```ignore
//! use ringkernel_core::observability::{PrometheusExporter, GrafanaDashboard};
//!
//! // Create Prometheus exporter
//! let exporter = PrometheusExporter::new();
//! exporter.register_collector(metrics_collector);
//!
//! // Get Prometheus metrics
//! let metrics = exporter.render();
//! println!("{}", metrics);
//!
//! // Generate Grafana dashboard JSON
//! let dashboard = GrafanaDashboard::new("RingKernel Metrics")
//!     .add_kernel_panel()
//!     .add_latency_panel()
//!     .add_throughput_panel()
//!     .build();
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use crate::telemetry_pipeline::MetricsCollector;

// ============================================================================
// OpenTelemetry-Compatible Span/Trace Types
// ============================================================================

/// A trace ID compatible with OpenTelemetry W3C Trace Context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceId(pub u128);

impl TraceId {
    /// Generate a new random trace ID.
    pub fn new() -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        std::thread::current().id().hash(&mut hasher);
        let high = hasher.finish() as u128;
        hasher.write_u64(high as u64);
        let low = hasher.finish() as u128;
        Self((high << 64) | low)
    }

    /// Parse from hex string.
    pub fn from_hex(hex: &str) -> Option<Self> {
        u128::from_str_radix(hex, 16).ok().map(Self)
    }

    /// Convert to hex string.
    pub fn to_hex(&self) -> String {
        format!("{:032x}", self.0)
    }
}

impl Default for TraceId {
    fn default() -> Self {
        Self::new()
    }
}

/// A span ID compatible with OpenTelemetry W3C Trace Context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId(pub u64);

impl SpanId {
    /// Generate a new random span ID.
    pub fn new() -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        std::process::id().hash(&mut hasher);
        Self(hasher.finish())
    }

    /// Parse from hex string.
    pub fn from_hex(hex: &str) -> Option<Self> {
        u64::from_str_radix(hex, 16).ok().map(Self)
    }

    /// Convert to hex string.
    pub fn to_hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

impl Default for SpanId {
    fn default() -> Self {
        Self::new()
    }
}

/// Span kind (OpenTelemetry compatible).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    /// Internal operation.
    Internal,
    /// Server-side span (receiving request).
    Server,
    /// Client-side span (sending request).
    Client,
    /// Producer span (async message send).
    Producer,
    /// Consumer span (async message receive).
    Consumer,
}

/// Span status (OpenTelemetry compatible).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpanStatus {
    /// Unset status.
    Unset,
    /// Operation completed successfully.
    Ok,
    /// Operation failed with error message.
    Error {
        /// Error message describing what went wrong.
        message: String,
    },
}

/// An OpenTelemetry-compatible span.
#[derive(Debug, Clone)]
pub struct Span {
    /// Trace ID.
    pub trace_id: TraceId,
    /// Span ID.
    pub span_id: SpanId,
    /// Parent span ID (if any).
    pub parent_span_id: Option<SpanId>,
    /// Span name.
    pub name: String,
    /// Span kind.
    pub kind: SpanKind,
    /// Start time.
    pub start_time: Instant,
    /// End time (if completed).
    pub end_time: Option<Instant>,
    /// Status.
    pub status: SpanStatus,
    /// Attributes (key-value pairs).
    pub attributes: HashMap<String, AttributeValue>,
    /// Events recorded during span.
    pub events: Vec<SpanEvent>,
}

/// Attribute value types.
#[derive(Debug, Clone)]
pub enum AttributeValue {
    /// String value.
    String(String),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// String array.
    StringArray(Vec<String>),
}

impl From<&str> for AttributeValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

impl From<String> for AttributeValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<i64> for AttributeValue {
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}

impl From<f64> for AttributeValue {
    fn from(f: f64) -> Self {
        Self::Float(f)
    }
}

impl From<bool> for AttributeValue {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

/// An event that occurred during a span.
#[derive(Debug, Clone)]
pub struct SpanEvent {
    /// Event name.
    pub name: String,
    /// Timestamp.
    pub timestamp: Instant,
    /// Event attributes.
    pub attributes: HashMap<String, AttributeValue>,
}

impl Span {
    /// Create a new span.
    pub fn new(name: impl Into<String>, kind: SpanKind) -> Self {
        Self {
            trace_id: TraceId::new(),
            span_id: SpanId::new(),
            parent_span_id: None,
            name: name.into(),
            kind,
            start_time: Instant::now(),
            end_time: None,
            status: SpanStatus::Unset,
            attributes: HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Create a child span.
    pub fn child(&self, name: impl Into<String>, kind: SpanKind) -> Self {
        Self {
            trace_id: self.trace_id,
            span_id: SpanId::new(),
            parent_span_id: Some(self.span_id),
            name: name.into(),
            kind,
            start_time: Instant::now(),
            end_time: None,
            status: SpanStatus::Unset,
            attributes: HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Set an attribute.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: impl Into<AttributeValue>) {
        self.attributes.insert(key.into(), value.into());
    }

    /// Add an event.
    pub fn add_event(&mut self, name: impl Into<String>) {
        self.events.push(SpanEvent {
            name: name.into(),
            timestamp: Instant::now(),
            attributes: HashMap::new(),
        });
    }

    /// Add an event with attributes.
    pub fn add_event_with_attributes(
        &mut self,
        name: impl Into<String>,
        attributes: HashMap<String, AttributeValue>,
    ) {
        self.events.push(SpanEvent {
            name: name.into(),
            timestamp: Instant::now(),
            attributes,
        });
    }

    /// Set status to OK.
    pub fn set_ok(&mut self) {
        self.status = SpanStatus::Ok;
    }

    /// Set error status.
    pub fn set_error(&mut self, message: impl Into<String>) {
        self.status = SpanStatus::Error {
            message: message.into(),
        };
    }

    /// End the span.
    pub fn end(&mut self) {
        self.end_time = Some(Instant::now());
    }

    /// Get span duration.
    pub fn duration(&self) -> Duration {
        self.end_time
            .unwrap_or_else(Instant::now)
            .duration_since(self.start_time)
    }

    /// Check if span is ended.
    pub fn is_ended(&self) -> bool {
        self.end_time.is_some()
    }
}

// ============================================================================
// Span Builder
// ============================================================================

/// Builder for creating spans with fluent API.
pub struct SpanBuilder {
    name: String,
    kind: SpanKind,
    parent: Option<(TraceId, SpanId)>,
    attributes: HashMap<String, AttributeValue>,
}

impl SpanBuilder {
    /// Create a new span builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: SpanKind::Internal,
            parent: None,
            attributes: HashMap::new(),
        }
    }

    /// Set span kind.
    pub fn kind(mut self, kind: SpanKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set parent span.
    pub fn parent(mut self, parent: &Span) -> Self {
        self.parent = Some((parent.trace_id, parent.span_id));
        self
    }

    /// Set attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<AttributeValue>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Build the span.
    pub fn build(self) -> Span {
        let mut span = Span::new(self.name, self.kind);
        if let Some((trace_id, parent_id)) = self.parent {
            span.trace_id = trace_id;
            span.parent_span_id = Some(parent_id);
        }
        span.attributes = self.attributes;
        span
    }
}

// ============================================================================
// Prometheus Metrics Exporter
// ============================================================================

/// Prometheus metric type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Counter (monotonically increasing).
    Counter,
    /// Gauge (can go up or down).
    Gauge,
    /// Histogram (distribution of values).
    Histogram,
    /// Summary (quantiles).
    Summary,
}

/// A Prometheus metric definition.
#[derive(Debug, Clone)]
pub struct MetricDefinition {
    /// Metric name.
    pub name: String,
    /// Metric type.
    pub metric_type: MetricType,
    /// Help text.
    pub help: String,
    /// Label names.
    pub labels: Vec<String>,
}

/// A single metric sample.
#[derive(Debug, Clone)]
pub struct MetricSample {
    /// Metric name.
    pub name: String,
    /// Label values (in order matching definition).
    pub label_values: Vec<String>,
    /// Sample value.
    pub value: f64,
    /// Timestamp (optional).
    pub timestamp_ms: Option<u64>,
}

/// Prometheus metrics exporter.
pub struct PrometheusExporter {
    /// Metric definitions.
    definitions: RwLock<Vec<MetricDefinition>>,
    /// Registered collectors.
    collectors: RwLock<Vec<Arc<dyn PrometheusCollector>>>,
    /// Custom metrics (for direct registration).
    custom_metrics: RwLock<HashMap<String, CustomMetric>>,
    /// Export timestamp.
    export_count: AtomicU64,
}

/// A custom registered metric.
struct CustomMetric {
    definition: MetricDefinition,
    samples: Vec<MetricSample>,
}

/// Trait for collecting Prometheus metrics.
pub trait PrometheusCollector: Send + Sync {
    /// Get metric definitions.
    fn definitions(&self) -> Vec<MetricDefinition>;

    /// Collect current metric samples.
    fn collect(&self) -> Vec<MetricSample>;
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            definitions: RwLock::new(Vec::new()),
            collectors: RwLock::new(Vec::new()),
            custom_metrics: RwLock::new(HashMap::new()),
            export_count: AtomicU64::new(0),
        })
    }

    /// Register a collector.
    pub fn register_collector(&self, collector: Arc<dyn PrometheusCollector>) {
        let defs = collector.definitions();
        self.definitions.write().extend(defs);
        self.collectors.write().push(collector);
    }

    /// Register a counter metric.
    pub fn register_counter(&self, name: &str, help: &str, labels: &[&str]) {
        let def = MetricDefinition {
            name: name.to_string(),
            metric_type: MetricType::Counter,
            help: help.to_string(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
        };
        self.custom_metrics.write().insert(
            name.to_string(),
            CustomMetric {
                definition: def,
                samples: Vec::new(),
            },
        );
    }

    /// Register a gauge metric.
    pub fn register_gauge(&self, name: &str, help: &str, labels: &[&str]) {
        let def = MetricDefinition {
            name: name.to_string(),
            metric_type: MetricType::Gauge,
            help: help.to_string(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
        };
        self.custom_metrics.write().insert(
            name.to_string(),
            CustomMetric {
                definition: def,
                samples: Vec::new(),
            },
        );
    }

    /// Register a histogram metric.
    pub fn register_histogram(&self, name: &str, help: &str, labels: &[&str]) {
        let def = MetricDefinition {
            name: name.to_string(),
            metric_type: MetricType::Histogram,
            help: help.to_string(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
        };
        self.custom_metrics.write().insert(
            name.to_string(),
            CustomMetric {
                definition: def,
                samples: Vec::new(),
            },
        );
    }

    /// Set a metric value.
    pub fn set_metric(&self, name: &str, value: f64, label_values: &[&str]) {
        let mut metrics = self.custom_metrics.write();
        if let Some(metric) = metrics.get_mut(name) {
            let sample = MetricSample {
                name: name.to_string(),
                label_values: label_values.iter().map(|s| s.to_string()).collect(),
                value,
                timestamp_ms: None,
            };
            // Find and replace existing sample with same labels, or add new
            let existing = metric
                .samples
                .iter_mut()
                .find(|s| s.label_values == sample.label_values);
            if let Some(existing) = existing {
                existing.value = value;
            } else {
                metric.samples.push(sample);
            }
        }
    }

    /// Increment a counter.
    pub fn inc_counter(&self, name: &str, label_values: &[&str]) {
        self.add_counter(name, 1.0, label_values);
    }

    /// Add to a counter.
    pub fn add_counter(&self, name: &str, delta: f64, label_values: &[&str]) {
        let mut metrics = self.custom_metrics.write();
        if let Some(metric) = metrics.get_mut(name) {
            let label_vec: Vec<String> = label_values.iter().map(|s| s.to_string()).collect();
            let existing = metric
                .samples
                .iter_mut()
                .find(|s| s.label_values == label_vec);
            if let Some(existing) = existing {
                existing.value += delta;
            } else {
                metric.samples.push(MetricSample {
                    name: name.to_string(),
                    label_values: label_vec,
                    value: delta,
                    timestamp_ms: None,
                });
            }
        }
    }

    /// Render metrics in Prometheus exposition format.
    pub fn render(&self) -> String {
        self.export_count.fetch_add(1, Ordering::Relaxed);

        let mut output = String::new();

        // Collect from registered collectors
        let collectors = self.collectors.read();
        for collector in collectors.iter() {
            let defs = collector.definitions();
            let samples = collector.collect();

            for def in &defs {
                // Write TYPE and HELP
                writeln!(output, "# HELP {} {}", def.name, def.help).unwrap();
                writeln!(
                    output,
                    "# TYPE {} {}",
                    def.name,
                    match def.metric_type {
                        MetricType::Counter => "counter",
                        MetricType::Gauge => "gauge",
                        MetricType::Histogram => "histogram",
                        MetricType::Summary => "summary",
                    }
                )
                .unwrap();

                // Write samples for this metric
                for sample in samples.iter().filter(|s| s.name == def.name) {
                    Self::write_sample(&mut output, &def.labels, sample);
                }
            }
        }

        // Collect custom metrics
        let custom = self.custom_metrics.read();
        for metric in custom.values() {
            writeln!(
                output,
                "# HELP {} {}",
                metric.definition.name, metric.definition.help
            )
            .unwrap();
            writeln!(
                output,
                "# TYPE {} {}",
                metric.definition.name,
                match metric.definition.metric_type {
                    MetricType::Counter => "counter",
                    MetricType::Gauge => "gauge",
                    MetricType::Histogram => "histogram",
                    MetricType::Summary => "summary",
                }
            )
            .unwrap();

            for sample in &metric.samples {
                Self::write_sample(&mut output, &metric.definition.labels, sample);
            }
        }

        output
    }

    fn write_sample(output: &mut String, labels: &[String], sample: &MetricSample) {
        if labels.is_empty() || sample.label_values.is_empty() {
            writeln!(output, "{} {}", sample.name, sample.value).unwrap();
        } else {
            let label_pairs: Vec<String> = labels
                .iter()
                .zip(sample.label_values.iter())
                .map(|(k, v)| format!("{}=\"{}\"", k, v))
                .collect();
            writeln!(
                output,
                "{}{{{}}} {}",
                sample.name,
                label_pairs.join(","),
                sample.value
            )
            .unwrap();
        }
    }

    /// Get export count.
    pub fn export_count(&self) -> u64 {
        self.export_count.load(Ordering::Relaxed)
    }
}

impl Default for PrometheusExporter {
    fn default() -> Self {
        Self {
            definitions: RwLock::new(Vec::new()),
            collectors: RwLock::new(Vec::new()),
            custom_metrics: RwLock::new(HashMap::new()),
            export_count: AtomicU64::new(0),
        }
    }
}

// ============================================================================
// RingKernel Prometheus Collector
// ============================================================================

/// Prometheus collector for RingKernel metrics.
pub struct RingKernelCollector {
    /// Metrics collector to read from.
    collector: Arc<MetricsCollector>,
}

impl RingKernelCollector {
    /// Create a new RingKernel collector.
    pub fn new(collector: Arc<MetricsCollector>) -> Arc<Self> {
        Arc::new(Self { collector })
    }
}

impl PrometheusCollector for RingKernelCollector {
    fn definitions(&self) -> Vec<MetricDefinition> {
        vec![
            MetricDefinition {
                name: "ringkernel_messages_processed_total".to_string(),
                metric_type: MetricType::Counter,
                help: "Total number of messages processed by kernels".to_string(),
                labels: vec!["kernel_id".to_string()],
            },
            MetricDefinition {
                name: "ringkernel_messages_dropped_total".to_string(),
                metric_type: MetricType::Counter,
                help: "Total number of messages dropped by kernels".to_string(),
                labels: vec!["kernel_id".to_string()],
            },
            MetricDefinition {
                name: "ringkernel_latency_us".to_string(),
                metric_type: MetricType::Gauge,
                help: "Current average message latency in microseconds".to_string(),
                labels: vec!["kernel_id".to_string(), "stat".to_string()],
            },
            MetricDefinition {
                name: "ringkernel_throughput".to_string(),
                metric_type: MetricType::Gauge,
                help: "Current message throughput per second".to_string(),
                labels: vec!["kernel_id".to_string()],
            },
        ]
    }

    fn collect(&self) -> Vec<MetricSample> {
        let aggregate = self.collector.get_aggregate();
        let elapsed = self.collector.elapsed().as_secs_f64().max(1.0);

        vec![
            MetricSample {
                name: "ringkernel_messages_processed_total".to_string(),
                label_values: vec!["aggregate".to_string()],
                value: aggregate.messages_processed as f64,
                timestamp_ms: None,
            },
            MetricSample {
                name: "ringkernel_messages_dropped_total".to_string(),
                label_values: vec!["aggregate".to_string()],
                value: aggregate.messages_dropped as f64,
                timestamp_ms: None,
            },
            MetricSample {
                name: "ringkernel_latency_us".to_string(),
                label_values: vec!["aggregate".to_string(), "avg".to_string()],
                value: aggregate.avg_latency_us(),
                timestamp_ms: None,
            },
            MetricSample {
                name: "ringkernel_latency_us".to_string(),
                label_values: vec!["aggregate".to_string(), "min".to_string()],
                value: aggregate.min_latency_us as f64,
                timestamp_ms: None,
            },
            MetricSample {
                name: "ringkernel_latency_us".to_string(),
                label_values: vec!["aggregate".to_string(), "max".to_string()],
                value: aggregate.max_latency_us as f64,
                timestamp_ms: None,
            },
            MetricSample {
                name: "ringkernel_throughput".to_string(),
                label_values: vec!["aggregate".to_string()],
                value: aggregate.messages_processed as f64 / elapsed,
                timestamp_ms: None,
            },
        ]
    }
}

// ============================================================================
// Grafana Dashboard Generator
// ============================================================================

/// Grafana panel type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelType {
    /// Time series graph.
    Graph,
    /// Single stat / gauge.
    Stat,
    /// Table.
    Table,
    /// Heatmap.
    Heatmap,
    /// Bar gauge.
    BarGauge,
}

/// A Grafana panel definition.
#[derive(Debug, Clone)]
pub struct GrafanaPanel {
    /// Panel title.
    pub title: String,
    /// Panel type.
    pub panel_type: PanelType,
    /// PromQL query expressions.
    pub queries: Vec<String>,
    /// Grid position.
    pub grid_pos: (u32, u32, u32, u32), // x, y, w, h
    /// Unit (for display).
    pub unit: Option<String>,
}

/// Grafana dashboard builder.
pub struct GrafanaDashboard {
    /// Dashboard title.
    title: String,
    /// Dashboard description.
    description: String,
    /// Panels.
    panels: Vec<GrafanaPanel>,
    /// Refresh interval.
    refresh: String,
    /// Time range.
    time_from: String,
    /// Tags.
    tags: Vec<String>,
}

impl GrafanaDashboard {
    /// Create a new dashboard builder.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            description: String::new(),
            panels: Vec::new(),
            refresh: "5s".to_string(),
            time_from: "now-1h".to_string(),
            tags: vec!["ringkernel".to_string()],
        }
    }

    /// Set description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set refresh interval.
    pub fn refresh(mut self, interval: impl Into<String>) -> Self {
        self.refresh = interval.into();
        self
    }

    /// Set time range.
    pub fn time_from(mut self, from: impl Into<String>) -> Self {
        self.time_from = from.into();
        self
    }

    /// Add a tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add a custom panel.
    pub fn panel(mut self, panel: GrafanaPanel) -> Self {
        self.panels.push(panel);
        self
    }

    /// Add kernel throughput panel.
    pub fn add_throughput_panel(mut self) -> Self {
        self.panels.push(GrafanaPanel {
            title: "Message Throughput".to_string(),
            panel_type: PanelType::Graph,
            queries: vec!["rate(ringkernel_messages_processed_total[1m])".to_string()],
            grid_pos: (0, 0, 12, 8),
            unit: Some("msg/s".to_string()),
        });
        self
    }

    /// Add latency panel.
    pub fn add_latency_panel(mut self) -> Self {
        self.panels.push(GrafanaPanel {
            title: "Message Latency".to_string(),
            panel_type: PanelType::Graph,
            queries: vec![
                "ringkernel_latency_us{stat=\"avg\"}".to_string(),
                "ringkernel_latency_us{stat=\"max\"}".to_string(),
            ],
            grid_pos: (12, 0, 12, 8),
            unit: Some("µs".to_string()),
        });
        self
    }

    /// Add kernel status panel.
    pub fn add_kernel_status_panel(mut self) -> Self {
        self.panels.push(GrafanaPanel {
            title: "Active Kernels".to_string(),
            panel_type: PanelType::Stat,
            queries: vec!["count(ringkernel_messages_processed_total)".to_string()],
            grid_pos: (0, 8, 6, 4),
            unit: None,
        });
        self
    }

    /// Add drop rate panel.
    pub fn add_drop_rate_panel(mut self) -> Self {
        self.panels.push(GrafanaPanel {
            title: "Message Drop Rate".to_string(),
            panel_type: PanelType::Graph,
            queries: vec![
                "rate(ringkernel_messages_dropped_total[1m]) / rate(ringkernel_messages_processed_total[1m])".to_string(),
            ],
            grid_pos: (6, 8, 6, 4),
            unit: Some("percentunit".to_string()),
        });
        self
    }

    /// Add multi-GPU panel.
    pub fn add_multi_gpu_panel(mut self) -> Self {
        self.panels.push(GrafanaPanel {
            title: "GPU Memory Usage".to_string(),
            panel_type: PanelType::BarGauge,
            queries: vec!["ringkernel_gpu_memory_used_bytes".to_string()],
            grid_pos: (12, 8, 12, 4),
            unit: Some("bytes".to_string()),
        });
        self
    }

    /// Add all standard panels.
    pub fn add_standard_panels(self) -> Self {
        self.add_throughput_panel()
            .add_latency_panel()
            .add_kernel_status_panel()
            .add_drop_rate_panel()
            .add_multi_gpu_panel()
    }

    /// Build dashboard JSON.
    pub fn build(&self) -> String {
        let panels_json: Vec<String> = self
            .panels
            .iter()
            .enumerate()
            .map(|(i, panel)| {
                let queries_json: Vec<String> = panel
                    .queries
                    .iter()
                    .enumerate()
                    .map(|(j, q)| {
                        format!(
                            r#"{{
                        "expr": "{}",
                        "refId": "{}",
                        "legendFormat": "{{}}"
                    }}"#,
                            q,
                            (b'A' + j as u8) as char
                        )
                    })
                    .collect();

                let unit_field = panel
                    .unit
                    .as_ref()
                    .map(|u| format!(r#""unit": "{}","#, u))
                    .unwrap_or_default();

                format!(
                    r#"{{
                    "id": {},
                    "title": "{}",
                    "type": "{}",
                    "gridPos": {{"x": {}, "y": {}, "w": {}, "h": {}}},
                    {}
                    "targets": [{}],
                    "datasource": {{"type": "prometheus", "uid": "${{datasource}}"}}
                }}"#,
                    i + 1,
                    panel.title,
                    match panel.panel_type {
                        PanelType::Graph => "timeseries",
                        PanelType::Stat => "stat",
                        PanelType::Table => "table",
                        PanelType::Heatmap => "heatmap",
                        PanelType::BarGauge => "bargauge",
                    },
                    panel.grid_pos.0,
                    panel.grid_pos.1,
                    panel.grid_pos.2,
                    panel.grid_pos.3,
                    unit_field,
                    queries_json.join(",")
                )
            })
            .collect();

        let tags_json: Vec<String> = self.tags.iter().map(|t| format!(r#""{}""#, t)).collect();

        format!(
            r#"{{
                "title": "{}",
                "description": "{}",
                "tags": [{}],
                "refresh": "{}",
                "time": {{"from": "{}", "to": "now"}},
                "templating": {{
                    "list": [
                        {{
                            "name": "datasource",
                            "type": "datasource",
                            "query": "prometheus"
                        }},
                        {{
                            "name": "kernel_id",
                            "type": "query",
                            "query": "label_values(ringkernel_messages_processed_total, kernel_id)",
                            "multi": true,
                            "includeAll": true
                        }}
                    ]
                }},
                "panels": [{}]
            }}"#,
            self.title,
            self.description,
            tags_json.join(","),
            self.refresh,
            self.time_from,
            panels_json.join(",")
        )
    }
}

// ============================================================================
// Observability Context
// ============================================================================

/// Global observability context for managing spans and metrics.
pub struct ObservabilityContext {
    /// Active spans.
    active_spans: RwLock<HashMap<SpanId, Span>>,
    /// Completed spans (for export).
    completed_spans: RwLock<Vec<Span>>,
    /// Max completed spans to retain.
    max_completed: usize,
    /// Prometheus exporter.
    prometheus: Arc<PrometheusExporter>,
}

impl ObservabilityContext {
    /// Create a new observability context.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            active_spans: RwLock::new(HashMap::new()),
            completed_spans: RwLock::new(Vec::new()),
            max_completed: 10000,
            prometheus: PrometheusExporter::new(),
        })
    }

    /// Start a new span.
    pub fn start_span(&self, name: impl Into<String>, kind: SpanKind) -> Span {
        let span = Span::new(name, kind);
        self.active_spans.write().insert(span.span_id, span.clone());
        span
    }

    /// Start a child span.
    pub fn start_child_span(&self, parent: &Span, name: impl Into<String>, kind: SpanKind) -> Span {
        let span = parent.child(name, kind);
        self.active_spans.write().insert(span.span_id, span.clone());
        span
    }

    /// End a span.
    pub fn end_span(&self, mut span: Span) {
        span.end();
        self.active_spans.write().remove(&span.span_id);

        let mut completed = self.completed_spans.write();
        completed.push(span);
        if completed.len() > self.max_completed {
            completed.remove(0);
        }
    }

    /// Get Prometheus exporter.
    pub fn prometheus(&self) -> &Arc<PrometheusExporter> {
        &self.prometheus
    }

    /// Export completed spans (for sending to trace backends).
    pub fn export_spans(&self) -> Vec<Span> {
        self.completed_spans.write().drain(..).collect()
    }

    /// Get active span count.
    pub fn active_span_count(&self) -> usize {
        self.active_spans.read().len()
    }
}

impl Default for ObservabilityContext {
    fn default() -> Self {
        Self {
            active_spans: RwLock::new(HashMap::new()),
            completed_spans: RwLock::new(Vec::new()),
            max_completed: 10000,
            prometheus: PrometheusExporter::new(),
        }
    }
}

// ============================================================================
// GPU Profiler Integration Stubs
// ============================================================================

/// GPU profiler backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuProfilerBackend {
    /// NVIDIA Nsight Systems/Compute.
    Nsight,
    /// RenderDoc (cross-platform).
    RenderDoc,
    /// PIX for Windows.
    Pix,
    /// Apple Metal System Trace.
    MetalSystemTrace,
    /// AMD Radeon GPU Profiler.
    Rgp,
    /// Custom profiler.
    Custom,
}

/// GPU profiler marker color.
#[derive(Debug, Clone, Copy)]
pub struct ProfilerColor {
    /// Red component (0-255).
    pub r: u8,
    /// Green component (0-255).
    pub g: u8,
    /// Blue component (0-255).
    pub b: u8,
    /// Alpha component (0-255).
    pub a: u8,
}

impl ProfilerColor {
    /// Create a new color.
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Red color.
    pub const RED: Self = Self::new(255, 0, 0);
    /// Green color.
    pub const GREEN: Self = Self::new(0, 255, 0);
    /// Blue color.
    pub const BLUE: Self = Self::new(0, 0, 255);
    /// Yellow color.
    pub const YELLOW: Self = Self::new(255, 255, 0);
    /// Cyan color.
    pub const CYAN: Self = Self::new(0, 255, 255);
    /// Magenta color.
    pub const MAGENTA: Self = Self::new(255, 0, 255);
    /// Orange color.
    pub const ORANGE: Self = Self::new(255, 165, 0);
}

/// GPU profiler range handle for scoped profiling.
pub struct ProfilerRange {
    /// Range name.
    #[allow(dead_code)]
    name: String,
    /// Backend being used.
    #[allow(dead_code)]
    backend: GpuProfilerBackend,
    /// Start time.
    start: Instant,
}

impl ProfilerRange {
    /// Create a new profiler range (internal use).
    fn new(name: impl Into<String>, backend: GpuProfilerBackend) -> Self {
        Self {
            name: name.into(),
            backend,
            start: Instant::now(),
        }
    }

    /// Create a stub profiler range for external profiler implementations.
    ///
    /// This is used by custom profiler implementations (like CUDA NVTX) that
    /// manage their own range lifecycle but need to return a ProfilerRange
    /// for API compatibility.
    pub fn stub(name: impl Into<String>, backend: GpuProfilerBackend) -> Self {
        Self::new(name, backend)
    }

    /// Get elapsed duration.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Drop for ProfilerRange {
    fn drop(&mut self) {
        // In a real implementation, this would call the profiler API to end the range
        // e.g., nvtxRangePop() for NVTX
    }
}

/// Trait for GPU profiler integration.
///
/// Implement this trait to integrate with specific GPU profiling tools.
/// The default implementation is a no-op for when no profiler is attached.
pub trait GpuProfiler: Send + Sync {
    /// Check if the profiler is available and attached.
    fn is_available(&self) -> bool {
        false
    }

    /// Get the profiler backend type.
    fn backend(&self) -> GpuProfilerBackend;

    /// Start a profiler capture session.
    fn start_capture(&self) -> Result<(), ProfilerError> {
        Ok(())
    }

    /// End a profiler capture session.
    fn end_capture(&self) -> Result<(), ProfilerError> {
        Ok(())
    }

    /// Trigger a frame/dispatch capture.
    fn trigger_capture(&self) -> Result<(), ProfilerError> {
        Ok(())
    }

    /// Push a named range onto the profiler stack.
    fn push_range(&self, name: &str, _color: ProfilerColor) -> ProfilerRange {
        ProfilerRange::new(name, self.backend())
    }

    /// Pop the current range from the profiler stack.
    fn pop_range(&self) {}

    /// Insert an instantaneous marker.
    fn mark(&self, _name: &str, _color: ProfilerColor) {}

    /// Set a per-thread name for the profiler.
    fn set_thread_name(&self, _name: &str) {}

    /// Add a message to the profiler output.
    fn message(&self, _text: &str) {}

    /// Register a GPU memory allocation.
    fn register_allocation(&self, _ptr: u64, _size: usize, _name: &str) {}

    /// Unregister a GPU memory allocation.
    fn unregister_allocation(&self, _ptr: u64) {}
}

/// Profiler error type.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProfilerError {
    /// Profiler is not available.
    #[error("GPU profiler not available")]
    NotAvailable,
    /// Profiler is not attached.
    #[error("GPU profiler not attached")]
    NotAttached,
    /// Capture already in progress.
    #[error("Capture already in progress")]
    CaptureInProgress,
    /// No capture in progress.
    #[error("No capture in progress")]
    NoCaptureInProgress,
    /// Backend-specific error.
    #[error("Profiler error: {0}")]
    Backend(String),
}

/// Null profiler implementation (no-op).
pub struct NullProfiler;

impl GpuProfiler for NullProfiler {
    fn backend(&self) -> GpuProfilerBackend {
        GpuProfilerBackend::Custom
    }
}

/// NVTX (NVIDIA Tools Extension) profiler stub.
///
/// When the real NVTX library is available, this integrates with
/// Nsight Systems and Nsight Compute.
pub struct NvtxProfiler {
    /// Whether NVTX is available.
    available: bool,
    /// Whether a capture is in progress.
    capture_in_progress: std::sync::atomic::AtomicBool,
}

impl NvtxProfiler {
    /// Create a new NVTX profiler.
    ///
    /// In a real implementation, this would check for libnvtx availability.
    pub fn new() -> Self {
        Self {
            available: false, // Would check nvtxInitialize() in real impl
            capture_in_progress: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Check if NVTX library is loaded.
    pub fn is_nvtx_loaded(&self) -> bool {
        // In real implementation: check if libnvtx is dynamically loaded
        self.available
    }
}

impl Default for NvtxProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuProfiler for NvtxProfiler {
    fn is_available(&self) -> bool {
        self.available
    }

    fn backend(&self) -> GpuProfilerBackend {
        GpuProfilerBackend::Nsight
    }

    fn start_capture(&self) -> Result<(), ProfilerError> {
        if !self.available {
            return Err(ProfilerError::NotAvailable);
        }
        if self.capture_in_progress.swap(true, Ordering::SeqCst) {
            return Err(ProfilerError::CaptureInProgress);
        }
        // Real impl: nvtxRangePushA("Capture")
        Ok(())
    }

    fn end_capture(&self) -> Result<(), ProfilerError> {
        if !self.capture_in_progress.swap(false, Ordering::SeqCst) {
            return Err(ProfilerError::NoCaptureInProgress);
        }
        // Real impl: nvtxRangePop()
        Ok(())
    }

    fn push_range(&self, name: &str, _color: ProfilerColor) -> ProfilerRange {
        // Real impl: nvtxRangePushA(name) with color attribute
        ProfilerRange::new(name, self.backend())
    }

    fn pop_range(&self) {
        // Real impl: nvtxRangePop()
    }

    fn mark(&self, _name: &str, _color: ProfilerColor) {
        // Real impl: nvtxMarkA(name) with color
    }

    fn set_thread_name(&self, _name: &str) {
        // Real impl: nvtxNameOsThread(thread_id, name)
    }
}

/// RenderDoc profiler stub.
///
/// Integrates with RenderDoc for GPU frame capture and debugging.
pub struct RenderDocProfiler {
    /// Whether RenderDoc is attached.
    attached: bool,
}

impl RenderDocProfiler {
    /// Create a new RenderDoc profiler.
    ///
    /// In a real implementation, this would use the RenderDoc in-app API.
    pub fn new() -> Self {
        Self {
            attached: false, // Would check RENDERDOC_GetAPI in real impl
        }
    }

    /// Check if RenderDoc is attached to the process.
    pub fn is_attached(&self) -> bool {
        // Real impl: check RENDERDOC_API_VERSION via GetAPI
        self.attached
    }

    /// Get RenderDoc capture file path.
    pub fn get_capture_path(&self) -> Option<String> {
        // Real impl: RENDERDOC_GetCapture
        None
    }

    /// Launch RenderDoc UI.
    pub fn launch_ui(&self) -> Result<(), ProfilerError> {
        if !self.attached {
            return Err(ProfilerError::NotAttached);
        }
        // Real impl: RENDERDOC_LaunchReplayUI
        Ok(())
    }
}

impl Default for RenderDocProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuProfiler for RenderDocProfiler {
    fn is_available(&self) -> bool {
        self.attached
    }

    fn backend(&self) -> GpuProfilerBackend {
        GpuProfilerBackend::RenderDoc
    }

    fn trigger_capture(&self) -> Result<(), ProfilerError> {
        if !self.attached {
            return Err(ProfilerError::NotAttached);
        }
        // Real impl: RENDERDOC_TriggerCapture
        Ok(())
    }

    fn start_capture(&self) -> Result<(), ProfilerError> {
        if !self.attached {
            return Err(ProfilerError::NotAttached);
        }
        // Real impl: RENDERDOC_StartFrameCapture
        Ok(())
    }

    fn end_capture(&self) -> Result<(), ProfilerError> {
        // Real impl: RENDERDOC_EndFrameCapture
        Ok(())
    }

    fn set_thread_name(&self, _name: &str) {
        // Real impl: RENDERDOC_SetCaptureOptionStr
    }
}

/// Metal System Trace profiler stub (macOS).
///
/// Integrates with Xcode Instruments for Metal GPU profiling.
#[cfg(target_os = "macos")]
pub struct MetalProfiler {
    /// Whether Metal profiling is available.
    available: bool,
}

#[cfg(target_os = "macos")]
impl MetalProfiler {
    /// Create a new Metal profiler.
    pub fn new() -> Self {
        Self { available: true }
    }
}

#[cfg(target_os = "macos")]
impl Default for MetalProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(target_os = "macos")]
impl GpuProfiler for MetalProfiler {
    fn is_available(&self) -> bool {
        self.available
    }

    fn backend(&self) -> GpuProfilerBackend {
        GpuProfilerBackend::MetalSystemTrace
    }

    fn push_range(&self, name: &str, _color: ProfilerColor) -> ProfilerRange {
        // Real impl: MTLCommandBuffer.pushDebugGroup(name)
        ProfilerRange::new(name, self.backend())
    }

    fn pop_range(&self) {
        // Real impl: MTLCommandBuffer.popDebugGroup()
    }

    fn mark(&self, _name: &str, _color: ProfilerColor) {
        // Real impl: MTLCommandBuffer.insertDebugSignpost(name)
    }
}

/// GPU profiler manager for selecting and using profilers.
pub struct GpuProfilerManager {
    /// Active profiler.
    profiler: Arc<dyn GpuProfiler>,
    /// Enabled state.
    enabled: std::sync::atomic::AtomicBool,
}

impl GpuProfilerManager {
    /// Create a new profiler manager with auto-detection.
    pub fn new() -> Self {
        // Try to detect available profiler
        let nvtx = NvtxProfiler::new();
        if nvtx.is_available() {
            return Self {
                profiler: Arc::new(nvtx),
                enabled: std::sync::atomic::AtomicBool::new(true),
            };
        }

        let renderdoc = RenderDocProfiler::new();
        if renderdoc.is_available() {
            return Self {
                profiler: Arc::new(renderdoc),
                enabled: std::sync::atomic::AtomicBool::new(true),
            };
        }

        // Fallback to null profiler
        Self {
            profiler: Arc::new(NullProfiler),
            enabled: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Create with a specific profiler.
    pub fn with_profiler(profiler: Arc<dyn GpuProfiler>) -> Self {
        let enabled = profiler.is_available();
        Self {
            profiler,
            enabled: std::sync::atomic::AtomicBool::new(enabled),
        }
    }

    /// Check if profiling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Enable or disable profiling.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get the profiler backend.
    pub fn backend(&self) -> GpuProfilerBackend {
        self.profiler.backend()
    }

    /// Start a profiled scope.
    pub fn scope(&self, name: &str) -> ProfilerScope<'_> {
        ProfilerScope::new(name, &*self.profiler, self.is_enabled())
    }

    /// Start a profiled scope with color.
    pub fn scope_colored(&self, name: &str, color: ProfilerColor) -> ProfilerScope<'_> {
        ProfilerScope::new_colored(name, &*self.profiler, self.is_enabled(), color)
    }

    /// Insert a marker.
    pub fn mark(&self, name: &str) {
        if self.is_enabled() {
            self.profiler.mark(name, ProfilerColor::CYAN);
        }
    }

    /// Get access to the underlying profiler.
    pub fn profiler(&self) -> &dyn GpuProfiler {
        &*self.profiler
    }
}

impl Default for GpuProfilerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII scope for profiler ranges.
pub struct ProfilerScope<'a> {
    profiler: &'a dyn GpuProfiler,
    enabled: bool,
}

impl<'a> ProfilerScope<'a> {
    fn new(name: &str, profiler: &'a dyn GpuProfiler, enabled: bool) -> Self {
        if enabled {
            profiler.push_range(name, ProfilerColor::CYAN);
        }
        Self { profiler, enabled }
    }

    fn new_colored(
        name: &str,
        profiler: &'a dyn GpuProfiler,
        enabled: bool,
        color: ProfilerColor,
    ) -> Self {
        if enabled {
            profiler.push_range(name, color);
        }
        Self { profiler, enabled }
    }
}

impl<'a> Drop for ProfilerScope<'a> {
    fn drop(&mut self) {
        if self.enabled {
            self.profiler.pop_range();
        }
    }
}

/// Macro for scoped GPU profiling.
///
/// # Example
///
/// ```ignore
/// use ringkernel_core::gpu_profile;
///
/// fn compute_kernel() {
///     gpu_profile!(profiler, "compute_kernel", {
///         // GPU work here
///     });
/// }
/// ```
#[macro_export]
macro_rules! gpu_profile {
    ($profiler:expr, $name:expr) => {
        let _scope = $profiler.scope($name);
    };
    ($profiler:expr, $name:expr, $color:expr) => {
        let _scope = $profiler.scope_colored($name, $color);
    };
}

// ============================================================================
// GPU Memory Dashboard
// ============================================================================

/// GPU memory allocation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMemoryType {
    /// Device-local memory (fastest, GPU only).
    DeviceLocal,
    /// Host-visible memory (accessible from CPU).
    HostVisible,
    /// Host-coherent memory (no explicit flush needed).
    HostCoherent,
    /// Mapped memory (CPU-GPU shared).
    Mapped,
    /// Queue buffers for message passing.
    QueueBuffer,
    /// Control block memory.
    ControlBlock,
    /// Shared memory (block-local).
    SharedMemory,
}

/// A tracked GPU memory allocation.
#[derive(Debug, Clone)]
pub struct GpuMemoryAllocation {
    /// Unique allocation ID.
    pub id: u64,
    /// Allocation name/label.
    pub name: String,
    /// Size in bytes.
    pub size: usize,
    /// Memory type.
    pub memory_type: GpuMemoryType,
    /// Device index (for multi-GPU).
    pub device_index: u32,
    /// Kernel ID (if associated with a kernel).
    pub kernel_id: Option<String>,
    /// Allocation timestamp.
    pub allocated_at: Instant,
    /// Whether the allocation is currently in use.
    pub in_use: bool,
}

/// GPU memory pool statistics.
#[derive(Debug, Clone, Default)]
pub struct GpuMemoryPoolStats {
    /// Pool name.
    pub name: String,
    /// Total capacity in bytes.
    pub capacity: usize,
    /// Currently allocated bytes.
    pub allocated: usize,
    /// Peak allocated bytes.
    pub peak_allocated: usize,
    /// Number of active allocations.
    pub allocation_count: u32,
    /// Number of allocations since creation.
    pub total_allocations: u64,
    /// Number of deallocations since creation.
    pub total_deallocations: u64,
    /// Fragmentation ratio (0.0 = none, 1.0 = fully fragmented).
    pub fragmentation: f32,
}

impl GpuMemoryPoolStats {
    /// Get utilization percentage.
    pub fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            0.0
        } else {
            (self.allocated as f32 / self.capacity as f32) * 100.0
        }
    }
}

/// Per-device GPU memory statistics.
#[derive(Debug, Clone, Default)]
pub struct GpuDeviceMemoryStats {
    /// Device index.
    pub device_index: u32,
    /// Device name.
    pub device_name: String,
    /// Total device memory in bytes.
    pub total_memory: u64,
    /// Free device memory in bytes.
    pub free_memory: u64,
    /// Memory used by RingKernel.
    pub ringkernel_used: u64,
    /// Memory used by other applications.
    pub other_used: u64,
    /// Memory pool statistics.
    pub pools: Vec<GpuMemoryPoolStats>,
}

impl GpuDeviceMemoryStats {
    /// Get used memory in bytes.
    pub fn used_memory(&self) -> u64 {
        self.total_memory - self.free_memory
    }

    /// Get utilization percentage.
    pub fn utilization(&self) -> f32 {
        if self.total_memory == 0 {
            0.0
        } else {
            (self.used_memory() as f32 / self.total_memory as f32) * 100.0
        }
    }
}

/// GPU Memory Dashboard for monitoring and visualization.
///
/// Provides real-time GPU memory tracking with allocation history,
/// per-kernel usage, and memory pressure alerts.
///
/// # Example
///
/// ```ignore
/// use ringkernel_core::observability::GpuMemoryDashboard;
///
/// let dashboard = GpuMemoryDashboard::new();
///
/// // Track an allocation
/// dashboard.track_allocation(
///     1,
///     "input_queue",
///     65536,
///     GpuMemoryType::QueueBuffer,
///     0,
///     Some("processor_kernel"),
/// );
///
/// // Get current stats
/// let stats = dashboard.get_device_stats(0);
/// println!("GPU 0 utilization: {:.1}%", stats.utilization());
///
/// // Generate Grafana panel JSON
/// let panel = dashboard.grafana_panel();
/// ```
pub struct GpuMemoryDashboard {
    /// Active allocations.
    allocations: RwLock<HashMap<u64, GpuMemoryAllocation>>,
    /// Per-device statistics.
    device_stats: RwLock<HashMap<u32, GpuDeviceMemoryStats>>,
    /// Memory pressure thresholds.
    thresholds: GpuMemoryThresholds,
    /// Allocation counter for unique IDs.
    allocation_counter: AtomicU64,
    /// Total bytes allocated.
    total_allocated: AtomicU64,
    /// Peak bytes allocated.
    peak_allocated: AtomicU64,
}

/// Memory pressure thresholds for alerts.
#[derive(Debug, Clone)]
pub struct GpuMemoryThresholds {
    /// Warning threshold (percentage).
    pub warning: f32,
    /// Critical threshold (percentage).
    pub critical: f32,
    /// Maximum allocation size before warning (bytes).
    pub max_allocation_size: usize,
    /// Maximum number of allocations before warning.
    pub max_allocation_count: u32,
}

impl Default for GpuMemoryThresholds {
    fn default() -> Self {
        Self {
            warning: 75.0,
            critical: 90.0,
            max_allocation_size: 1024 * 1024 * 1024, // 1 GB
            max_allocation_count: 10000,
        }
    }
}

/// Memory pressure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressureLevel {
    /// Memory usage is normal.
    Normal,
    /// Memory usage is elevated (approaching warning threshold).
    Elevated,
    /// Memory usage is at warning level.
    Warning,
    /// Memory usage is critical.
    Critical,
    /// Out of memory.
    OutOfMemory,
}

impl GpuMemoryDashboard {
    /// Create a new GPU memory dashboard.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            allocations: RwLock::new(HashMap::new()),
            device_stats: RwLock::new(HashMap::new()),
            thresholds: GpuMemoryThresholds::default(),
            allocation_counter: AtomicU64::new(1),
            total_allocated: AtomicU64::new(0),
            peak_allocated: AtomicU64::new(0),
        })
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(thresholds: GpuMemoryThresholds) -> Arc<Self> {
        Arc::new(Self {
            allocations: RwLock::new(HashMap::new()),
            device_stats: RwLock::new(HashMap::new()),
            thresholds,
            allocation_counter: AtomicU64::new(1),
            total_allocated: AtomicU64::new(0),
            peak_allocated: AtomicU64::new(0),
        })
    }

    /// Track a new GPU memory allocation.
    pub fn track_allocation(
        &self,
        id: u64,
        name: impl Into<String>,
        size: usize,
        memory_type: GpuMemoryType,
        device_index: u32,
        kernel_id: Option<&str>,
    ) {
        let allocation = GpuMemoryAllocation {
            id,
            name: name.into(),
            size,
            memory_type,
            device_index,
            kernel_id: kernel_id.map(String::from),
            allocated_at: Instant::now(),
            in_use: true,
        };

        self.allocations.write().insert(id, allocation);

        // Update totals
        let new_total = self
            .total_allocated
            .fetch_add(size as u64, Ordering::Relaxed)
            + size as u64;
        let mut peak = self.peak_allocated.load(Ordering::Relaxed);
        while new_total > peak {
            match self.peak_allocated.compare_exchange_weak(
                peak,
                new_total,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => peak = current,
            }
        }
    }

    /// Generate a new unique allocation ID.
    pub fn next_allocation_id(&self) -> u64 {
        self.allocation_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Track deallocation.
    pub fn track_deallocation(&self, id: u64) {
        let mut allocations = self.allocations.write();
        if let Some(alloc) = allocations.remove(&id) {
            self.total_allocated
                .fetch_sub(alloc.size as u64, Ordering::Relaxed);
        }
    }

    /// Mark an allocation as no longer in use (but not freed).
    pub fn mark_unused(&self, id: u64) {
        let mut allocations = self.allocations.write();
        if let Some(alloc) = allocations.get_mut(&id) {
            alloc.in_use = false;
        }
    }

    /// Register a GPU device.
    pub fn register_device(&self, device_index: u32, name: impl Into<String>, total_memory: u64) {
        let stats = GpuDeviceMemoryStats {
            device_index,
            device_name: name.into(),
            total_memory,
            free_memory: total_memory,
            ringkernel_used: 0,
            other_used: 0,
            pools: Vec::new(),
        };
        self.device_stats.write().insert(device_index, stats);
    }

    /// Update device memory statistics.
    pub fn update_device_stats(&self, device_index: u32, free_memory: u64, ringkernel_used: u64) {
        let mut stats = self.device_stats.write();
        if let Some(device) = stats.get_mut(&device_index) {
            device.free_memory = free_memory;
            device.ringkernel_used = ringkernel_used;
            device.other_used = device
                .total_memory
                .saturating_sub(free_memory + ringkernel_used);
        }
    }

    /// Get device statistics.
    pub fn get_device_stats(&self, device_index: u32) -> Option<GpuDeviceMemoryStats> {
        self.device_stats.read().get(&device_index).cloned()
    }

    /// Get all device statistics.
    pub fn get_all_device_stats(&self) -> Vec<GpuDeviceMemoryStats> {
        self.device_stats.read().values().cloned().collect()
    }

    /// Get all active allocations.
    pub fn get_allocations(&self) -> Vec<GpuMemoryAllocation> {
        self.allocations.read().values().cloned().collect()
    }

    /// Get allocations for a specific kernel.
    pub fn get_kernel_allocations(&self, kernel_id: &str) -> Vec<GpuMemoryAllocation> {
        self.allocations
            .read()
            .values()
            .filter(|a| a.kernel_id.as_deref() == Some(kernel_id))
            .cloned()
            .collect()
    }

    /// Get total allocated memory.
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated.load(Ordering::Relaxed)
    }

    /// Get peak allocated memory.
    pub fn peak_allocated(&self) -> u64 {
        self.peak_allocated.load(Ordering::Relaxed)
    }

    /// Get allocation count.
    pub fn allocation_count(&self) -> usize {
        self.allocations.read().len()
    }

    /// Check memory pressure level for a device.
    pub fn check_pressure(&self, device_index: u32) -> MemoryPressureLevel {
        let stats = self.device_stats.read();
        if let Some(device) = stats.get(&device_index) {
            let utilization = device.utilization();
            if device.free_memory == 0 {
                MemoryPressureLevel::OutOfMemory
            } else if utilization >= self.thresholds.critical {
                MemoryPressureLevel::Critical
            } else if utilization >= self.thresholds.warning {
                MemoryPressureLevel::Warning
            } else if utilization >= self.thresholds.warning * 0.8 {
                MemoryPressureLevel::Elevated
            } else {
                MemoryPressureLevel::Normal
            }
        } else {
            MemoryPressureLevel::Normal
        }
    }

    /// Generate Grafana dashboard panel for GPU memory.
    pub fn grafana_panel(&self) -> GrafanaPanel {
        GrafanaPanel {
            title: "GPU Memory Usage".to_string(),
            panel_type: PanelType::BarGauge,
            queries: vec![
                "ringkernel_gpu_memory_allocated_bytes".to_string(),
                "ringkernel_gpu_memory_peak_bytes".to_string(),
            ],
            grid_pos: (0, 0, 12, 8),
            unit: Some("bytes".to_string()),
        }
    }

    /// Generate Prometheus metrics for GPU memory.
    pub fn prometheus_metrics(&self) -> String {
        let mut output = String::new();

        // Total allocated
        writeln!(output, "# HELP ringkernel_gpu_memory_allocated_bytes Current GPU memory allocated by RingKernel").unwrap();
        writeln!(output, "# TYPE ringkernel_gpu_memory_allocated_bytes gauge").unwrap();
        writeln!(
            output,
            "ringkernel_gpu_memory_allocated_bytes {}",
            self.total_allocated()
        )
        .unwrap();

        // Peak allocated
        writeln!(
            output,
            "# HELP ringkernel_gpu_memory_peak_bytes Peak GPU memory allocated by RingKernel"
        )
        .unwrap();
        writeln!(output, "# TYPE ringkernel_gpu_memory_peak_bytes gauge").unwrap();
        writeln!(
            output,
            "ringkernel_gpu_memory_peak_bytes {}",
            self.peak_allocated()
        )
        .unwrap();

        // Allocation count
        writeln!(
            output,
            "# HELP ringkernel_gpu_memory_allocation_count Number of active GPU allocations"
        )
        .unwrap();
        writeln!(
            output,
            "# TYPE ringkernel_gpu_memory_allocation_count gauge"
        )
        .unwrap();
        writeln!(
            output,
            "ringkernel_gpu_memory_allocation_count {}",
            self.allocation_count()
        )
        .unwrap();

        // Per-device stats
        let device_stats = self.device_stats.read();
        for device in device_stats.values() {
            writeln!(
                output,
                "ringkernel_gpu_device_memory_total_bytes{{device=\"{}\"}} {}",
                device.device_name, device.total_memory
            )
            .unwrap();
            writeln!(
                output,
                "ringkernel_gpu_device_memory_free_bytes{{device=\"{}\"}} {}",
                device.device_name, device.free_memory
            )
            .unwrap();
            writeln!(
                output,
                "ringkernel_gpu_device_memory_used_bytes{{device=\"{}\"}} {}",
                device.device_name,
                device.used_memory()
            )
            .unwrap();
            writeln!(
                output,
                "ringkernel_gpu_device_utilization{{device=\"{}\"}} {:.2}",
                device.device_name,
                device.utilization()
            )
            .unwrap();
        }

        output
    }

    /// Generate a memory summary report.
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        writeln!(report, "=== GPU Memory Dashboard ===").unwrap();
        writeln!(report, "Total Allocated: {} bytes", self.total_allocated()).unwrap();
        writeln!(report, "Peak Allocated: {} bytes", self.peak_allocated()).unwrap();
        writeln!(report, "Active Allocations: {}", self.allocation_count()).unwrap();
        writeln!(report).unwrap();

        // Device summary
        let device_stats = self.device_stats.read();
        for device in device_stats.values() {
            writeln!(
                report,
                "--- Device {} ({}) ---",
                device.device_index, device.device_name
            )
            .unwrap();
            writeln!(
                report,
                "  Total: {} MB",
                device.total_memory / (1024 * 1024)
            )
            .unwrap();
            writeln!(report, "  Free: {} MB", device.free_memory / (1024 * 1024)).unwrap();
            writeln!(
                report,
                "  RingKernel: {} MB",
                device.ringkernel_used / (1024 * 1024)
            )
            .unwrap();
            writeln!(report, "  Utilization: {:.1}%", device.utilization()).unwrap();
            writeln!(
                report,
                "  Pressure: {:?}",
                self.check_pressure(device.device_index)
            )
            .unwrap();
        }

        // Top allocations by size
        let allocations = self.allocations.read();
        let mut sorted_allocs: Vec<_> = allocations.values().collect();
        sorted_allocs.sort_by_key(|a| std::cmp::Reverse(a.size));

        if !sorted_allocs.is_empty() {
            writeln!(report).unwrap();
            writeln!(report, "--- Top 10 Allocations ---").unwrap();
            for (i, alloc) in sorted_allocs.iter().take(10).enumerate() {
                writeln!(
                    report,
                    "  {}. {} - {} bytes ({:?})",
                    i + 1,
                    alloc.name,
                    alloc.size,
                    alloc.memory_type
                )
                .unwrap();
            }
        }

        report
    }
}

impl Default for GpuMemoryDashboard {
    fn default() -> Self {
        Self {
            allocations: RwLock::new(HashMap::new()),
            device_stats: RwLock::new(HashMap::new()),
            thresholds: GpuMemoryThresholds::default(),
            allocation_counter: AtomicU64::new(1),
            total_allocated: AtomicU64::new(0),
            peak_allocated: AtomicU64::new(0),
        }
    }
}

// ============================================================================
// OTLP (OpenTelemetry Protocol) Exporter
// ============================================================================

/// OTLP transport protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OtlpTransport {
    /// HTTP with JSON encoding (default, no extra dependencies).
    #[default]
    HttpJson,
    /// HTTP with Protobuf encoding (requires protobuf support).
    HttpProtobuf,
    /// gRPC transport (requires tonic).
    Grpc,
}

/// Configuration for OTLP exporter.
#[derive(Debug, Clone)]
pub struct OtlpConfig {
    /// OTLP endpoint URL (e.g., "http://localhost:4318/v1/traces").
    pub endpoint: String,
    /// Transport protocol.
    pub transport: OtlpTransport,
    /// Service name for resource attributes.
    pub service_name: String,
    /// Service version.
    pub service_version: String,
    /// Service instance ID.
    pub service_instance_id: Option<String>,
    /// Additional resource attributes.
    pub resource_attributes: Vec<(String, String)>,
    /// Export batch size.
    pub batch_size: usize,
    /// Export interval.
    pub export_interval: Duration,
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum retry attempts.
    pub max_retries: u32,
    /// Retry delay (base for exponential backoff).
    pub retry_delay: Duration,
    /// Optional authorization header.
    pub authorization: Option<String>,
}

impl Default for OtlpConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:4318/v1/traces".to_string(),
            transport: OtlpTransport::HttpJson,
            service_name: "ringkernel".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            service_instance_id: None,
            resource_attributes: Vec::new(),
            batch_size: 512,
            export_interval: Duration::from_secs(5),
            timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
            authorization: None,
        }
    }
}

impl OtlpConfig {
    /// Create a new OTLP configuration.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Default::default()
        }
    }

    /// Set the service name.
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Set the service version.
    pub fn with_service_version(mut self, version: impl Into<String>) -> Self {
        self.service_version = version.into();
        self
    }

    /// Set the service instance ID.
    pub fn with_instance_id(mut self, id: impl Into<String>) -> Self {
        self.service_instance_id = Some(id.into());
        self
    }

    /// Add a resource attribute.
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.resource_attributes.push((key.into(), value.into()));
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set the export interval.
    pub fn with_export_interval(mut self, interval: Duration) -> Self {
        self.export_interval = interval;
        self
    }

    /// Set the authorization header.
    pub fn with_authorization(mut self, auth: impl Into<String>) -> Self {
        self.authorization = Some(auth.into());
        self
    }

    /// Configure for Jaeger OTLP endpoint.
    pub fn jaeger(endpoint: impl Into<String>) -> Self {
        Self::new(endpoint).with_service_name("ringkernel")
    }

    /// Configure for Honeycomb.
    pub fn honeycomb(api_key: impl Into<String>) -> Self {
        Self::new("https://api.honeycomb.io/v1/traces")
            .with_authorization(format!("x-honeycomb-team {}", api_key.into()))
    }

    /// Configure for Grafana Cloud.
    pub fn grafana_cloud(instance_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        let instance = instance_id.into();
        Self::new("https://otlp-gateway-prod-us-central-0.grafana.net/otlp/v1/traces")
            .with_authorization(format!("Basic {}", api_key.into()))
            .with_attribute("grafana.instance", instance)
    }
}

/// OTLP export result.
#[derive(Debug, Clone)]
pub struct OtlpExportResult {
    /// Number of spans exported.
    pub spans_exported: usize,
    /// Whether the export succeeded.
    pub success: bool,
    /// Error message if export failed.
    pub error: Option<String>,
    /// Export duration.
    pub duration: Duration,
    /// Number of retry attempts.
    pub retry_count: u32,
}

/// Statistics for the OTLP exporter.
#[derive(Debug, Clone, Default)]
pub struct OtlpExporterStats {
    /// Total spans exported.
    pub total_spans_exported: u64,
    /// Total export attempts.
    pub total_exports: u64,
    /// Successful exports.
    pub successful_exports: u64,
    /// Failed exports.
    pub failed_exports: u64,
    /// Total retry attempts.
    pub total_retries: u64,
    /// Spans currently in buffer.
    pub buffered_spans: usize,
    /// Last export time.
    pub last_export: Option<Instant>,
    /// Last error message.
    pub last_error: Option<String>,
}

/// OTLP span exporter for sending traces to OTLP-compatible backends.
///
/// Supports HTTP/JSON transport with automatic batching and retries.
pub struct OtlpExporter {
    config: OtlpConfig,
    buffer: RwLock<Vec<Span>>,
    stats: RwLock<OtlpExporterStats>,
}

impl OtlpExporter {
    /// Create a new OTLP exporter with the given configuration.
    pub fn new(config: OtlpConfig) -> Self {
        Self {
            config,
            buffer: RwLock::new(Vec::new()),
            stats: RwLock::new(OtlpExporterStats::default()),
        }
    }

    /// Create an exporter for a local Jaeger instance.
    pub fn jaeger_local() -> Self {
        Self::new(OtlpConfig::jaeger("http://localhost:4318/v1/traces"))
    }

    /// Get the exporter configuration.
    pub fn config(&self) -> &OtlpConfig {
        &self.config
    }

    /// Get current statistics.
    pub fn stats(&self) -> OtlpExporterStats {
        self.stats.read().clone()
    }

    /// Add a span to the export buffer.
    pub fn export_span(&self, span: Span) {
        let mut buffer = self.buffer.write();
        buffer.push(span);

        let should_flush = buffer.len() >= self.config.batch_size;
        drop(buffer);

        if should_flush {
            let _ = self.flush();
        }
    }

    /// Add multiple spans to the export buffer.
    pub fn export_spans(&self, spans: Vec<Span>) {
        let mut buffer = self.buffer.write();
        buffer.extend(spans);

        let should_flush = buffer.len() >= self.config.batch_size;
        drop(buffer);

        if should_flush {
            let _ = self.flush();
        }
    }

    /// Get the number of buffered spans.
    pub fn buffered_count(&self) -> usize {
        self.buffer.read().len()
    }

    /// Flush all buffered spans to the OTLP endpoint.
    pub fn flush(&self) -> OtlpExportResult {
        let spans: Vec<Span> = {
            let mut buffer = self.buffer.write();
            std::mem::take(&mut *buffer)
        };

        if spans.is_empty() {
            return OtlpExportResult {
                spans_exported: 0,
                success: true,
                error: None,
                duration: Duration::ZERO,
                retry_count: 0,
            };
        }

        let start = Instant::now();
        let result = self.send_spans(&spans);
        let duration = start.elapsed();

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_exports += 1;
            stats.last_export = Some(Instant::now());

            if result.success {
                stats.successful_exports += 1;
                stats.total_spans_exported += spans.len() as u64;
            } else {
                stats.failed_exports += 1;
                stats.last_error = result.error.clone();
                // Put spans back in buffer for retry
                let mut buffer = self.buffer.write();
                buffer.extend(spans);
            }
            stats.total_retries += result.retry_count as u64;
            stats.buffered_spans = self.buffer.read().len();
        }

        OtlpExportResult {
            spans_exported: if result.success {
                result.spans_exported
            } else {
                0
            },
            duration,
            ..result
        }
    }

    /// Send spans to the OTLP endpoint.
    fn send_spans(&self, spans: &[Span]) -> OtlpExportResult {
        // Without the alerting feature (reqwest), we can only buffer spans
        #[cfg(not(feature = "alerting"))]
        {
            tracing::debug!(
                span_count = spans.len(),
                endpoint = %self.config.endpoint,
                "OTLP stub: would export spans (enable 'alerting' feature for HTTP export)"
            );
            OtlpExportResult {
                spans_exported: spans.len(),
                success: true,
                error: None,
                duration: Duration::ZERO,
                retry_count: 0,
            }
        }

        #[cfg(feature = "alerting")]
        {
            self.send_spans_http(spans)
        }
    }

    /// Send spans via HTTP (requires alerting feature).
    #[cfg(feature = "alerting")]
    fn send_spans_http(&self, spans: &[Span]) -> OtlpExportResult {
        let payload = self.build_otlp_json(spans);

        let client = reqwest::blocking::Client::builder()
            .timeout(self.config.timeout)
            .build();

        let client = match client {
            Ok(c) => c,
            Err(e) => {
                return OtlpExportResult {
                    spans_exported: 0,
                    success: false,
                    error: Some(format!("Failed to create HTTP client: {}", e)),
                    duration: Duration::ZERO,
                    retry_count: 0,
                };
            }
        };

        let mut retry_count = 0;
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            let mut request = client
                .post(&self.config.endpoint)
                .header("Content-Type", "application/json")
                .body(payload.clone());

            if let Some(auth) = &self.config.authorization {
                request = request.header("Authorization", auth);
            }

            match request.send() {
                Ok(response) => {
                    if response.status().is_success() {
                        return OtlpExportResult {
                            spans_exported: spans.len(),
                            success: true,
                            error: None,
                            duration: Duration::ZERO,
                            retry_count,
                        };
                    } else {
                        last_error = Some(format!(
                            "HTTP {}: {}",
                            response.status(),
                            response.status().as_str()
                        ));
                    }
                }
                Err(e) => {
                    last_error = Some(format!("Request failed: {}", e));
                }
            }

            if attempt < self.config.max_retries {
                retry_count += 1;
                std::thread::sleep(self.config.retry_delay * (1 << attempt));
            }
        }

        OtlpExportResult {
            spans_exported: 0,
            success: false,
            error: last_error,
            duration: Duration::ZERO,
            retry_count,
        }
    }

    /// Build OTLP JSON payload.
    #[cfg(feature = "alerting")]
    fn build_otlp_json(&self, spans: &[Span]) -> String {
        use std::fmt::Write;

        let mut json = String::with_capacity(4096);

        // Resource spans structure
        json.push_str(r#"{"resourceSpans":[{"resource":{"attributes":["#);

        // Service name
        let _ = write!(
            json,
            r#"{{"key":"service.name","value":{{"stringValue":"{}"}}}}"#,
            escape_json_str(&self.config.service_name)
        );

        // Service version
        let _ = write!(
            json,
            r#",{{"key":"service.version","value":{{"stringValue":"{}"}}}}"#,
            escape_json_str(&self.config.service_version)
        );

        // Instance ID
        if let Some(instance_id) = &self.config.service_instance_id {
            let _ = write!(
                json,
                r#",{{"key":"service.instance.id","value":{{"stringValue":"{}"}}}}"#,
                escape_json_str(instance_id)
            );
        }

        // Additional attributes
        for (key, value) in &self.config.resource_attributes {
            let _ = write!(
                json,
                r#",{{"key":"{}","value":{{"stringValue":"{}"}}}}"#,
                escape_json_str(key),
                escape_json_str(value)
            );
        }

        json.push_str(r#"]},"scopeSpans":[{"scope":{"name":"ringkernel"},"spans":["#);

        // Add spans
        let mut first = true;
        for span in spans {
            if !first {
                json.push(',');
            }
            first = false;
            self.span_to_json(&mut json, span);
        }

        json.push_str("]}]}]}");
        json
    }

    /// Convert a span to OTLP JSON format.
    #[cfg(feature = "alerting")]
    fn span_to_json(&self, json: &mut String, span: &Span) {
        use std::fmt::Write;

        let _ = write!(
            json,
            r#"{{"traceId":"{}","spanId":"{}""#,
            span.trace_id.to_hex(),
            span.span_id.to_hex()
        );

        if let Some(parent) = span.parent_span_id {
            let _ = write!(json, r#","parentSpanId":"{}""#, parent.to_hex());
        }

        let _ = write!(
            json,
            r#","name":"{}","kind":{}"#,
            escape_json_str(&span.name),
            match span.kind {
                SpanKind::Internal => 1,
                SpanKind::Server => 2,
                SpanKind::Client => 3,
                SpanKind::Producer => 4,
                SpanKind::Consumer => 5,
            }
        );

        // Convert timestamps to nanoseconds since epoch
        let start_nanos = span.start_time.elapsed().as_nanos();
        let end_nanos = span
            .end_time
            .map(|t| t.elapsed().as_nanos())
            .unwrap_or(start_nanos);

        // Note: These are approximate since we use Instant, not SystemTime
        let _ = write!(
            json,
            r#","startTimeUnixNano":"{}","endTimeUnixNano":"{}""#,
            start_nanos, end_nanos
        );

        // Status
        let _ = write!(
            json,
            r#","status":{{"code":{}}}"#,
            match &span.status {
                SpanStatus::Unset => 0,
                SpanStatus::Ok => 1,
                SpanStatus::Error { .. } => 2,
            }
        );

        // Attributes
        if !span.attributes.is_empty() {
            json.push_str(r#","attributes":["#);
            let mut first = true;
            for (key, value) in &span.attributes {
                if !first {
                    json.push(',');
                }
                first = false;
                let _ = write!(
                    json,
                    r#"{{"key":"{}","value":{}}}"#,
                    escape_json_str(key),
                    attribute_value_to_json(value)
                );
            }
            json.push(']');
        }

        // Events
        if !span.events.is_empty() {
            json.push_str(r#","events":["#);
            let mut first = true;
            for event in &span.events {
                if !first {
                    json.push(',');
                }
                first = false;
                let _ = write!(
                    json,
                    r#"{{"name":"{}","timeUnixNano":"{}"}}"#,
                    escape_json_str(&event.name),
                    event.timestamp.elapsed().as_nanos()
                );
            }
            json.push(']');
        }

        json.push('}');
    }
}

/// Helper to escape JSON strings.
#[cfg(feature = "alerting")]
fn escape_json_str(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Convert AttributeValue to OTLP JSON format.
#[cfg(feature = "alerting")]
fn attribute_value_to_json(value: &AttributeValue) -> String {
    match value {
        AttributeValue::String(s) => format!(r#"{{"stringValue":"{}"}}"#, escape_json_str(s)),
        AttributeValue::Int(i) => format!(r#"{{"intValue":"{}"}}"#, i),
        AttributeValue::Float(f) => format!(r#"{{"doubleValue":{}}}"#, f),
        AttributeValue::Bool(b) => format!(r#"{{"boolValue":{}}}"#, b),
        AttributeValue::StringArray(arr) => {
            let values: Vec<String> = arr
                .iter()
                .map(|s| format!(r#"{{"stringValue":"{}"}}"#, escape_json_str(s)))
                .collect();
            format!(r#"{{"arrayValue":{{"values":[{}]}}}}"#, values.join(","))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::KernelId;

    #[test]
    fn test_trace_id_generation() {
        let id1 = TraceId::new();
        let id2 = TraceId::new();
        assert_ne!(id1.0, id2.0);
    }

    #[test]
    fn test_trace_id_hex() {
        let id = TraceId(0x123456789abcdef0123456789abcdef0);
        let hex = id.to_hex();
        assert_eq!(hex.len(), 32);
        let parsed = TraceId::from_hex(&hex).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_span_creation() {
        let span = Span::new("test_operation", SpanKind::Internal);
        assert!(!span.is_ended());
        assert_eq!(span.name, "test_operation");
    }

    #[test]
    fn test_span_child() {
        let parent = Span::new("parent", SpanKind::Server);
        let child = parent.child("child", SpanKind::Internal);

        assert_eq!(child.trace_id, parent.trace_id);
        assert_eq!(child.parent_span_id, Some(parent.span_id));
    }

    #[test]
    fn test_span_attributes() {
        let mut span = Span::new("test", SpanKind::Internal);
        span.set_attribute("string_key", "value");
        span.set_attribute("int_key", 42i64);
        span.set_attribute("bool_key", true);

        assert_eq!(span.attributes.len(), 3);
    }

    #[test]
    fn test_span_events() {
        let mut span = Span::new("test", SpanKind::Internal);
        span.add_event("event1");
        span.add_event("event2");

        assert_eq!(span.events.len(), 2);
    }

    #[test]
    fn test_span_builder() {
        let parent = Span::new("parent", SpanKind::Server);
        let span = SpanBuilder::new("child")
            .kind(SpanKind::Client)
            .parent(&parent)
            .attribute("key", "value")
            .build();

        assert_eq!(span.trace_id, parent.trace_id);
        assert_eq!(span.kind, SpanKind::Client);
        assert!(span.attributes.contains_key("key"));
    }

    #[test]
    fn test_prometheus_exporter() {
        let exporter = PrometheusExporter::new();
        exporter.register_counter("test_counter", "A test counter", &["label1"]);
        exporter.register_gauge("test_gauge", "A test gauge", &[]);

        exporter.inc_counter("test_counter", &["value1"]);
        exporter.inc_counter("test_counter", &["value1"]);
        exporter.set_metric("test_gauge", 42.0, &[]);

        let output = exporter.render();
        assert!(output.contains("test_counter"));
        assert!(output.contains("test_gauge"));
    }

    #[test]
    fn test_grafana_dashboard() {
        let dashboard = GrafanaDashboard::new("Test Dashboard")
            .description("A test dashboard")
            .add_throughput_panel()
            .add_latency_panel()
            .build();

        assert!(dashboard.contains("Test Dashboard"));
        assert!(dashboard.contains("Message Throughput"));
        assert!(dashboard.contains("Message Latency"));
    }

    #[test]
    fn test_observability_context() {
        let ctx = ObservabilityContext::new();

        let span = ctx.start_span("test", SpanKind::Internal);
        assert_eq!(ctx.active_span_count(), 1);

        ctx.end_span(span);
        assert_eq!(ctx.active_span_count(), 0);

        let exported = ctx.export_spans();
        assert_eq!(exported.len(), 1);
    }

    #[test]
    fn test_ringkernel_collector() {
        let collector = Arc::new(MetricsCollector::new());
        let kernel_id = KernelId::new("test");

        collector.record_message_processed(&kernel_id, 100);
        collector.record_message_processed(&kernel_id, 200);

        let prom_collector = RingKernelCollector::new(collector);
        let defs = prom_collector.definitions();
        let samples = prom_collector.collect();

        assert!(!defs.is_empty());
        assert!(!samples.is_empty());
    }

    // GPU Profiler tests

    #[test]
    fn test_profiler_color() {
        let color = ProfilerColor::new(128, 64, 32);
        assert_eq!(color.r, 128);
        assert_eq!(color.g, 64);
        assert_eq!(color.b, 32);
        assert_eq!(color.a, 255);

        assert_eq!(ProfilerColor::RED.r, 255);
        assert_eq!(ProfilerColor::GREEN.g, 255);
        assert_eq!(ProfilerColor::BLUE.b, 255);
    }

    #[test]
    fn test_null_profiler() {
        let profiler = NullProfiler;
        assert!(!profiler.is_available());
        assert_eq!(profiler.backend(), GpuProfilerBackend::Custom);

        // All operations should be no-ops
        assert!(profiler.start_capture().is_ok());
        assert!(profiler.end_capture().is_ok());
        assert!(profiler.trigger_capture().is_ok());

        let range = profiler.push_range("test", ProfilerColor::RED);
        let _elapsed = range.elapsed(); // Just verify it doesn't panic
        profiler.pop_range();
        profiler.mark("marker", ProfilerColor::BLUE);
        profiler.set_thread_name("thread");
    }

    #[test]
    fn test_nvtx_profiler_stub() {
        let profiler = NvtxProfiler::new();
        assert_eq!(profiler.backend(), GpuProfilerBackend::Nsight);

        // Not available by default (stub)
        assert!(!profiler.is_available());
        assert!(!profiler.is_nvtx_loaded());

        // Start capture should fail when not available
        assert!(matches!(
            profiler.start_capture(),
            Err(ProfilerError::NotAvailable)
        ));
    }

    #[test]
    fn test_renderdoc_profiler_stub() {
        let profiler = RenderDocProfiler::new();
        assert_eq!(profiler.backend(), GpuProfilerBackend::RenderDoc);

        // Not attached by default (stub)
        assert!(!profiler.is_available());
        assert!(!profiler.is_attached());
        assert!(profiler.get_capture_path().is_none());

        // Launch UI should fail when not attached
        assert!(matches!(
            profiler.launch_ui(),
            Err(ProfilerError::NotAttached)
        ));
    }

    #[test]
    fn test_gpu_profiler_manager() {
        let manager = GpuProfilerManager::new();

        // Default should be null profiler (since stubs report unavailable)
        assert!(!manager.is_enabled());
        assert_eq!(manager.backend(), GpuProfilerBackend::Custom);

        // Can enable/disable
        manager.set_enabled(true);
        assert!(manager.is_enabled());
        manager.set_enabled(false);
        assert!(!manager.is_enabled());
    }

    #[test]
    fn test_profiler_scope() {
        let manager = GpuProfilerManager::new();

        // Scopes should work even when profiler is not available
        {
            let _scope = manager.scope("test_scope");
            // Scope automatically pops on drop
        }

        {
            let _scope = manager.scope_colored("colored_scope", ProfilerColor::ORANGE);
        }

        // Mark should also work
        manager.mark("test_marker");
    }

    #[test]
    fn test_profiler_with_custom() {
        let custom_profiler = Arc::new(NullProfiler);
        let manager = GpuProfilerManager::with_profiler(custom_profiler);

        assert_eq!(manager.backend(), GpuProfilerBackend::Custom);
    }

    #[test]
    fn test_profiler_range_elapsed() {
        let range = ProfilerRange::new("test", GpuProfilerBackend::Custom);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = range.elapsed();
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    fn test_profiler_error_display() {
        let err = ProfilerError::NotAvailable;
        assert!(err.to_string().contains("not available"));

        let err = ProfilerError::NotAttached;
        assert!(err.to_string().contains("not attached"));

        let err = ProfilerError::CaptureInProgress;
        assert!(err.to_string().contains("in progress"));

        let err = ProfilerError::Backend("test error".to_string());
        assert!(err.to_string().contains("test error"));
    }

    // GPU Memory Dashboard tests

    #[test]
    fn test_gpu_memory_dashboard_creation() {
        let dashboard = GpuMemoryDashboard::new();
        assert_eq!(dashboard.total_allocated(), 0);
        assert_eq!(dashboard.peak_allocated(), 0);
        assert_eq!(dashboard.allocation_count(), 0);
    }

    #[test]
    fn test_gpu_memory_allocation_tracking() {
        let dashboard = GpuMemoryDashboard::new();

        // Track an allocation
        dashboard.track_allocation(
            1,
            "test_buffer",
            65536,
            GpuMemoryType::DeviceLocal,
            0,
            Some("test_kernel"),
        );

        assert_eq!(dashboard.total_allocated(), 65536);
        assert_eq!(dashboard.peak_allocated(), 65536);
        assert_eq!(dashboard.allocation_count(), 1);

        // Track another allocation
        dashboard.track_allocation(
            2,
            "queue_buffer",
            1024,
            GpuMemoryType::QueueBuffer,
            0,
            Some("test_kernel"),
        );

        assert_eq!(dashboard.total_allocated(), 66560);
        assert_eq!(dashboard.peak_allocated(), 66560);
        assert_eq!(dashboard.allocation_count(), 2);

        // Deallocate first buffer
        dashboard.track_deallocation(1);
        assert_eq!(dashboard.total_allocated(), 1024);
        assert_eq!(dashboard.peak_allocated(), 66560); // Peak should remain
        assert_eq!(dashboard.allocation_count(), 1);
    }

    #[test]
    fn test_gpu_memory_device_stats() {
        let dashboard = GpuMemoryDashboard::new();

        // Register a device
        dashboard.register_device(0, "NVIDIA RTX 4090", 24 * 1024 * 1024 * 1024); // 24 GB

        let stats = dashboard.get_device_stats(0).unwrap();
        assert_eq!(stats.device_index, 0);
        assert_eq!(stats.device_name, "NVIDIA RTX 4090");
        assert_eq!(stats.total_memory, 24 * 1024 * 1024 * 1024);
        assert_eq!(stats.utilization(), 0.0);

        // Update device stats
        let used = 8 * 1024 * 1024 * 1024; // 8 GB used
        let free = 16 * 1024 * 1024 * 1024; // 16 GB free
        dashboard.update_device_stats(0, free, used);

        let stats = dashboard.get_device_stats(0).unwrap();
        assert!(stats.utilization() > 30.0 && stats.utilization() < 35.0);
    }

    #[test]
    fn test_gpu_memory_pressure_levels() {
        let dashboard = GpuMemoryDashboard::new();

        // Register a device with 1 GB
        dashboard.register_device(0, "Test GPU", 1024 * 1024 * 1024);

        // Normal usage (50%)
        dashboard.update_device_stats(0, 512 * 1024 * 1024, 256 * 1024 * 1024);
        assert_eq!(dashboard.check_pressure(0), MemoryPressureLevel::Normal);

        // Warning level (80%)
        dashboard.update_device_stats(0, 200 * 1024 * 1024, 600 * 1024 * 1024);
        assert_eq!(dashboard.check_pressure(0), MemoryPressureLevel::Warning);

        // Critical level (95%)
        dashboard.update_device_stats(0, 50 * 1024 * 1024, 900 * 1024 * 1024);
        assert_eq!(dashboard.check_pressure(0), MemoryPressureLevel::Critical);

        // OOM
        dashboard.update_device_stats(0, 0, 1024 * 1024 * 1024);
        assert_eq!(
            dashboard.check_pressure(0),
            MemoryPressureLevel::OutOfMemory
        );
    }

    #[test]
    fn test_gpu_memory_kernel_allocations() {
        let dashboard = GpuMemoryDashboard::new();

        // Track allocations for different kernels
        dashboard.track_allocation(
            1,
            "buf1",
            1000,
            GpuMemoryType::DeviceLocal,
            0,
            Some("kernel_a"),
        );
        dashboard.track_allocation(
            2,
            "buf2",
            2000,
            GpuMemoryType::DeviceLocal,
            0,
            Some("kernel_a"),
        );
        dashboard.track_allocation(
            3,
            "buf3",
            3000,
            GpuMemoryType::DeviceLocal,
            0,
            Some("kernel_b"),
        );

        let kernel_a_allocs = dashboard.get_kernel_allocations("kernel_a");
        assert_eq!(kernel_a_allocs.len(), 2);

        let kernel_b_allocs = dashboard.get_kernel_allocations("kernel_b");
        assert_eq!(kernel_b_allocs.len(), 1);

        let kernel_c_allocs = dashboard.get_kernel_allocations("kernel_c");
        assert_eq!(kernel_c_allocs.len(), 0);
    }

    #[test]
    fn test_gpu_memory_prometheus_metrics() {
        let dashboard = GpuMemoryDashboard::new();
        dashboard.track_allocation(1, "buf", 1000, GpuMemoryType::DeviceLocal, 0, None);
        dashboard.register_device(0, "GPU0", 1024 * 1024 * 1024);

        let metrics = dashboard.prometheus_metrics();
        assert!(metrics.contains("ringkernel_gpu_memory_allocated_bytes"));
        assert!(metrics.contains("ringkernel_gpu_memory_peak_bytes"));
        assert!(metrics.contains("ringkernel_gpu_memory_allocation_count"));
    }

    #[test]
    fn test_gpu_memory_summary_report() {
        let dashboard = GpuMemoryDashboard::new();
        dashboard.track_allocation(
            1,
            "large_buffer",
            1024 * 1024,
            GpuMemoryType::DeviceLocal,
            0,
            None,
        );
        dashboard.register_device(0, "GPU0", 1024 * 1024 * 1024);

        let report = dashboard.summary_report();
        assert!(report.contains("GPU Memory Dashboard"));
        assert!(report.contains("large_buffer"));
    }

    #[test]
    fn test_gpu_memory_pool_stats() {
        let pool_stats = GpuMemoryPoolStats {
            name: "default".to_string(),
            capacity: 1024 * 1024,
            allocated: 512 * 1024,
            peak_allocated: 768 * 1024,
            allocation_count: 10,
            total_allocations: 100,
            total_deallocations: 90,
            fragmentation: 0.1,
        };

        assert!(pool_stats.utilization() > 49.0 && pool_stats.utilization() < 51.0);
    }

    #[test]
    fn test_gpu_memory_types() {
        // Ensure all memory types are distinct
        let types = [
            GpuMemoryType::DeviceLocal,
            GpuMemoryType::HostVisible,
            GpuMemoryType::HostCoherent,
            GpuMemoryType::Mapped,
            GpuMemoryType::QueueBuffer,
            GpuMemoryType::ControlBlock,
            GpuMemoryType::SharedMemory,
        ];

        for (i, t1) in types.iter().enumerate() {
            for (j, t2) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(t1, t2);
                }
            }
        }
    }

    #[test]
    fn test_gpu_memory_grafana_panel() {
        let dashboard = GpuMemoryDashboard::new();
        let panel = dashboard.grafana_panel();

        assert_eq!(panel.title, "GPU Memory Usage");
        assert_eq!(panel.panel_type, PanelType::BarGauge);
        assert!(!panel.queries.is_empty());
    }

    #[test]
    fn test_gpu_memory_allocation_id_generation() {
        let dashboard = GpuMemoryDashboard::new();

        let id1 = dashboard.next_allocation_id();
        let id2 = dashboard.next_allocation_id();
        let id3 = dashboard.next_allocation_id();

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    // OTLP Exporter tests

    #[test]
    fn test_otlp_config_default() {
        let config = OtlpConfig::default();
        assert_eq!(config.endpoint, "http://localhost:4318/v1/traces");
        assert_eq!(config.transport, OtlpTransport::HttpJson);
        assert_eq!(config.service_name, "ringkernel");
        assert_eq!(config.batch_size, 512);
    }

    #[test]
    fn test_otlp_config_builder() {
        let config = OtlpConfig::new("http://example.com/v1/traces")
            .with_service_name("my-service")
            .with_service_version("1.0.0")
            .with_instance_id("instance-1")
            .with_attribute("env", "production")
            .with_batch_size(100);

        assert_eq!(config.endpoint, "http://example.com/v1/traces");
        assert_eq!(config.service_name, "my-service");
        assert_eq!(config.service_version, "1.0.0");
        assert_eq!(config.service_instance_id, Some("instance-1".to_string()));
        assert_eq!(config.resource_attributes.len(), 1);
        assert_eq!(config.batch_size, 100);
    }

    #[test]
    fn test_otlp_config_jaeger() {
        let config = OtlpConfig::jaeger("http://jaeger:4318/v1/traces");
        assert_eq!(config.endpoint, "http://jaeger:4318/v1/traces");
        assert_eq!(config.service_name, "ringkernel");
    }

    #[test]
    fn test_otlp_config_honeycomb() {
        let config = OtlpConfig::honeycomb("my-api-key");
        assert_eq!(config.endpoint, "https://api.honeycomb.io/v1/traces");
        assert_eq!(
            config.authorization,
            Some("x-honeycomb-team my-api-key".to_string())
        );
    }

    #[test]
    fn test_otlp_exporter_creation() {
        let exporter = OtlpExporter::new(OtlpConfig::default());
        assert_eq!(exporter.buffered_count(), 0);
        assert_eq!(exporter.config().service_name, "ringkernel");
    }

    #[test]
    fn test_otlp_exporter_jaeger_local() {
        let exporter = OtlpExporter::jaeger_local();
        assert_eq!(
            exporter.config().endpoint,
            "http://localhost:4318/v1/traces"
        );
    }

    #[test]
    fn test_otlp_exporter_buffering() {
        let config = OtlpConfig::default().with_batch_size(10);
        let exporter = OtlpExporter::new(config);

        // Create a test span using the constructor
        let span = Span::new("test_span", SpanKind::Internal);

        // Add spans
        for _ in 0..5 {
            exporter.export_span(span.clone());
        }

        assert_eq!(exporter.buffered_count(), 5);
    }

    #[test]
    fn test_otlp_exporter_flush_empty() {
        let exporter = OtlpExporter::new(OtlpConfig::default());

        let result = exporter.flush();
        assert!(result.success);
        assert_eq!(result.spans_exported, 0);
    }

    #[test]
    fn test_otlp_exporter_stats() {
        let exporter = OtlpExporter::new(OtlpConfig::default());

        // Initial stats
        let stats = exporter.stats();
        assert_eq!(stats.total_exports, 0);
        assert_eq!(stats.total_spans_exported, 0);
        assert_eq!(stats.buffered_spans, 0);
    }

    #[test]
    fn test_otlp_transport_default() {
        let transport = OtlpTransport::default();
        assert_eq!(transport, OtlpTransport::HttpJson);
    }
}
