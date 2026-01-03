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
            let existing = metric.samples.iter_mut().find(|s| s.label_values == sample.label_values);
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
            let existing = metric.samples.iter_mut().find(|s| s.label_values == label_vec);
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
            writeln!(output, "# HELP {} {}", metric.definition.name, metric.definition.help).unwrap();
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
            writeln!(output, "{}{{{}}} {}", sample.name, label_pairs.join(","), sample.value).unwrap();
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
            queries: vec![
                "rate(ringkernel_messages_processed_total[1m])".to_string(),
            ],
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
            queries: vec![
                "count(ringkernel_messages_processed_total)".to_string(),
            ],
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
            queries: vec![
                "ringkernel_gpu_memory_used_bytes".to_string(),
            ],
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
        let panels_json: Vec<String> = self.panels.iter().enumerate().map(|(i, panel)| {
            let queries_json: Vec<String> = panel.queries.iter().enumerate().map(|(j, q)| {
                format!(
                    r#"{{
                        "expr": "{}",
                        "refId": "{}",
                        "legendFormat": "{{}}"
                    }}"#,
                    q,
                    (b'A' + j as u8) as char
                )
            }).collect();

            let unit_field = panel.unit.as_ref()
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
        }).collect();

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
}
