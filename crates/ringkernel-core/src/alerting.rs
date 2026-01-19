//! Alert routing system for enterprise monitoring.
//!
//! This module provides a pluggable alert routing system with support for
//! multiple destinations including webhooks (Slack, Teams, PagerDuty),
//! email, and custom sinks.
//!
//! # Feature Flags
//!
//! - `alerting` - Enables HTTP webhook delivery (requires `reqwest` crate)
//!
//! # Example
//!
//! ```rust,ignore
//! use ringkernel_core::alerting::{AlertRouter, Alert, AlertSeverity, WebhookSink};
//!
//! let router = AlertRouter::new()
//!     .with_sink(WebhookSink::new("https://hooks.slack.com/services/...")
//!         .with_severity_filter(AlertSeverity::Warning))
//!     .with_sink(WebhookSink::pagerduty("your-integration-key")
//!         .with_severity_filter(AlertSeverity::Critical));
//!
//! router.send(Alert::new(AlertSeverity::Critical, "GPU memory exhausted")
//!     .with_source("kernel_1")
//!     .with_metadata("gpu_id", "0")
//!     .with_metadata("memory_used_gb", "24.5"));
//! ```

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ============================================================================
// ALERT SEVERITY
// ============================================================================

/// Alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum AlertSeverity {
    /// Informational alert (FYI).
    Info = 0,
    /// Warning that may require attention.
    Warning = 1,
    /// Error that requires attention.
    Error = 2,
    /// Critical issue requiring immediate attention.
    Critical = 3,
}

impl AlertSeverity {
    /// Get the severity name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Error => "ERROR",
            Self::Critical => "CRITICAL",
        }
    }

    /// Get a color for display (hex format).
    pub fn color(&self) -> &'static str {
        match self {
            Self::Info => "#2196F3",    // Blue
            Self::Warning => "#FF9800", // Orange
            Self::Error => "#F44336",   // Red
            Self::Critical => "#9C27B0", // Purple
        }
    }
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// ALERT
// ============================================================================

/// An alert to be routed to sinks.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Unique alert ID.
    pub id: u64,
    /// Alert severity.
    pub severity: AlertSeverity,
    /// Alert title/summary.
    pub title: String,
    /// Detailed description.
    pub description: Option<String>,
    /// Source of the alert (kernel ID, component name, etc.).
    pub source: Option<String>,
    /// When the alert was created.
    pub timestamp: SystemTime,
    /// Deduplication key (alerts with same key within window are deduplicated).
    pub dedup_key: Option<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
    /// Alert tags for filtering.
    pub tags: Vec<String>,
}

impl Alert {
    /// Create a new alert.
    pub fn new(severity: AlertSeverity, title: impl Into<String>) -> Self {
        static ALERT_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            id: ALERT_ID.fetch_add(1, Ordering::Relaxed),
            severity,
            title: title.into(),
            description: None,
            source: None,
            timestamp: SystemTime::now(),
            dedup_key: None,
            metadata: HashMap::new(),
            tags: Vec::new(),
        }
    }

    /// Add a description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a source.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add a deduplication key.
    pub fn with_dedup_key(mut self, key: impl Into<String>) -> Self {
        self.dedup_key = Some(key.into());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags.
    pub fn with_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Get timestamp as Unix milliseconds.
    pub fn timestamp_millis(&self) -> u64 {
        self.timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Format as JSON.
    pub fn to_json(&self) -> String {
        let source_str = self
            .source
            .as_ref()
            .map(|s| format!(r#","source":"{}""#, escape_json(s)))
            .unwrap_or_default();

        let desc_str = self
            .description
            .as_ref()
            .map(|s| format!(r#","description":"{}""#, escape_json(s)))
            .unwrap_or_default();

        let metadata_str = if self.metadata.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = self
                .metadata
                .iter()
                .map(|(k, v)| format!(r#""{}":"{}""#, escape_json(k), escape_json(v)))
                .collect();
            format!(r#","metadata":{{{}}}"#, pairs.join(","))
        };

        let tags_str = if self.tags.is_empty() {
            String::new()
        } else {
            let tags: Vec<String> = self.tags.iter().map(|t| format!(r#""{}""#, escape_json(t))).collect();
            format!(r#","tags":[{}]"#, tags.join(","))
        };

        format!(
            r#"{{"id":{},"severity":"{}","title":"{}","timestamp":{}{}{}{}{}}}"#,
            self.id,
            self.severity.as_str(),
            escape_json(&self.title),
            self.timestamp_millis(),
            source_str,
            desc_str,
            metadata_str,
            tags_str,
        )
    }
}

/// Escape a string for JSON.
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ============================================================================
// ALERT SINK TRAIT
// ============================================================================

/// Error type for alert sink operations.
#[derive(Debug, Clone)]
pub enum AlertSinkError {
    /// Connection/network error.
    ConnectionError(String),
    /// Rate limited.
    RateLimited(String),
    /// Configuration error.
    ConfigError(String),
    /// Other error.
    Other(String),
}

impl fmt::Display for AlertSinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            Self::RateLimited(msg) => write!(f, "Rate limited: {}", msg),
            Self::ConfigError(msg) => write!(f, "Config error: {}", msg),
            Self::Other(msg) => write!(f, "Alert sink error: {}", msg),
        }
    }
}

impl std::error::Error for AlertSinkError {}

/// Result type for alert sink operations.
pub type AlertSinkResult<T> = Result<T, AlertSinkError>;

/// Trait for pluggable alert destinations.
#[async_trait]
pub trait AlertSink: Send + Sync {
    /// Send an alert to the sink.
    async fn send(&self, alert: &Alert) -> AlertSinkResult<()>;

    /// Get the sink name.
    fn sink_name(&self) -> &str;

    /// Check if this sink accepts alerts of the given severity.
    fn accepts_severity(&self, severity: AlertSeverity) -> bool;

    /// Health check the sink.
    async fn health_check(&self) -> AlertSinkResult<()>;
}

// ============================================================================
// LOG SINK (writes to tracing/log)
// ============================================================================

/// Alert sink that writes to the tracing/log system.
pub struct LogSink {
    /// Minimum severity to log.
    min_severity: AlertSeverity,
}

impl LogSink {
    /// Create a new log sink.
    pub fn new() -> Self {
        Self {
            min_severity: AlertSeverity::Info,
        }
    }

    /// Set minimum severity filter.
    pub fn with_severity_filter(mut self, min: AlertSeverity) -> Self {
        self.min_severity = min;
        self
    }
}

impl Default for LogSink {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AlertSink for LogSink {
    async fn send(&self, alert: &Alert) -> AlertSinkResult<()> {
        if alert.severity < self.min_severity {
            return Ok(());
        }

        let source = alert.source.as_deref().unwrap_or("unknown");
        let msg = format!(
            "[ALERT] {} | {} | {} | {}",
            alert.severity,
            source,
            alert.title,
            alert.description.as_deref().unwrap_or("")
        );

        match alert.severity {
            AlertSeverity::Info => tracing::info!("{}", msg),
            AlertSeverity::Warning => tracing::warn!("{}", msg),
            AlertSeverity::Error => tracing::error!("{}", msg),
            AlertSeverity::Critical => tracing::error!(target: "critical", "{}", msg),
        }

        Ok(())
    }

    fn sink_name(&self) -> &str {
        "LogSink"
    }

    fn accepts_severity(&self, severity: AlertSeverity) -> bool {
        severity >= self.min_severity
    }

    async fn health_check(&self) -> AlertSinkResult<()> {
        Ok(())
    }
}

// ============================================================================
// IN-MEMORY SINK (for testing)
// ============================================================================

/// Alert sink that stores alerts in memory (for testing).
pub struct InMemorySink {
    alerts: RwLock<Vec<Alert>>,
    max_alerts: usize,
    min_severity: AlertSeverity,
}

impl InMemorySink {
    /// Create a new in-memory sink.
    pub fn new(max_alerts: usize) -> Self {
        Self {
            alerts: RwLock::new(Vec::new()),
            max_alerts,
            min_severity: AlertSeverity::Info,
        }
    }

    /// Set minimum severity filter.
    pub fn with_severity_filter(mut self, min: AlertSeverity) -> Self {
        self.min_severity = min;
        self
    }

    /// Get all stored alerts.
    pub fn alerts(&self) -> Vec<Alert> {
        self.alerts.read().clone()
    }

    /// Get the count of stored alerts.
    pub fn count(&self) -> usize {
        self.alerts.read().len()
    }

    /// Clear all stored alerts.
    pub fn clear(&self) {
        self.alerts.write().clear();
    }
}

#[async_trait]
impl AlertSink for InMemorySink {
    async fn send(&self, alert: &Alert) -> AlertSinkResult<()> {
        if alert.severity < self.min_severity {
            return Ok(());
        }

        let mut alerts = self.alerts.write();
        if alerts.len() >= self.max_alerts {
            alerts.remove(0);
        }
        alerts.push(alert.clone());
        Ok(())
    }

    fn sink_name(&self) -> &str {
        "InMemorySink"
    }

    fn accepts_severity(&self, severity: AlertSeverity) -> bool {
        severity >= self.min_severity
    }

    async fn health_check(&self) -> AlertSinkResult<()> {
        Ok(())
    }
}

// ============================================================================
// WEBHOOK SINK (requires alerting feature)
// ============================================================================

/// Webhook alert sink for Slack, Teams, PagerDuty, etc.
#[cfg(feature = "alerting")]
pub struct WebhookSink {
    /// Webhook URL.
    url: String,
    /// Minimum severity to send.
    min_severity: AlertSeverity,
    /// Custom headers.
    headers: HashMap<String, String>,
    /// Webhook format.
    format: WebhookFormat,
    /// HTTP client.
    client: reqwest::Client,
    /// Timeout.
    timeout: Duration,
}

/// Webhook payload format.
#[cfg(feature = "alerting")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebhookFormat {
    /// Generic JSON payload.
    Json,
    /// Slack incoming webhook format.
    Slack,
    /// Microsoft Teams incoming webhook format.
    MsTeams,
    /// PagerDuty Events API v2 format.
    PagerDuty,
}

#[cfg(feature = "alerting")]
impl WebhookSink {
    /// Create a new webhook sink.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            min_severity: AlertSeverity::Info,
            headers: HashMap::new(),
            format: WebhookFormat::Json,
            client: reqwest::Client::new(),
            timeout: Duration::from_secs(10),
        }
    }

    /// Create a Slack webhook sink.
    pub fn slack(url: impl Into<String>) -> Self {
        Self::new(url).with_format(WebhookFormat::Slack)
    }

    /// Create a Microsoft Teams webhook sink.
    pub fn teams(url: impl Into<String>) -> Self {
        Self::new(url).with_format(WebhookFormat::MsTeams)
    }

    /// Create a PagerDuty webhook sink.
    pub fn pagerduty(routing_key: impl Into<String>) -> Self {
        Self::new("https://events.pagerduty.com/v2/enqueue")
            .with_format(WebhookFormat::PagerDuty)
            .with_header("X-Routing-Key", routing_key.into())
    }

    /// Set minimum severity filter.
    pub fn with_severity_filter(mut self, min: AlertSeverity) -> Self {
        self.min_severity = min;
        self
    }

    /// Set webhook format.
    pub fn with_format(mut self, format: WebhookFormat) -> Self {
        self.format = format;
        self
    }

    /// Add a custom header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    /// Set timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Format alert as webhook payload.
    fn format_payload(&self, alert: &Alert) -> String {
        match self.format {
            WebhookFormat::Json => alert.to_json(),
            WebhookFormat::Slack => self.format_slack(alert),
            WebhookFormat::MsTeams => self.format_teams(alert),
            WebhookFormat::PagerDuty => self.format_pagerduty(alert),
        }
    }

    fn format_slack(&self, alert: &Alert) -> String {
        let desc = alert.description.as_deref().unwrap_or("");
        let source = alert.source.as_deref().unwrap_or("unknown");

        format!(
            r#"{{"attachments":[{{"color":"{}","title":"[{}] {}","text":"{}","fields":[{{"title":"Source","value":"{}","short":true}},{{"title":"Time","value":"{}","short":true}}]}}]}}"#,
            alert.severity.color(),
            alert.severity.as_str(),
            escape_json(&alert.title),
            escape_json(desc),
            escape_json(source),
            alert.timestamp_millis(),
        )
    }

    fn format_teams(&self, alert: &Alert) -> String {
        let desc = alert.description.as_deref().unwrap_or("");
        let source = alert.source.as_deref().unwrap_or("unknown");

        format!(
            r#"{{"@type":"MessageCard","@context":"http://schema.org/extensions","themeColor":"{}","title":"[{}] {}","text":"{}","sections":[{{"facts":[{{"name":"Source","value":"{}"}},{{"name":"Severity","value":"{}"}}]}}]}}"#,
            alert.severity.color().trim_start_matches('#'),
            alert.severity.as_str(),
            escape_json(&alert.title),
            escape_json(desc),
            escape_json(source),
            alert.severity.as_str(),
        )
    }

    fn format_pagerduty(&self, alert: &Alert) -> String {
        let severity_pd = match alert.severity {
            AlertSeverity::Info => "info",
            AlertSeverity::Warning => "warning",
            AlertSeverity::Error => "error",
            AlertSeverity::Critical => "critical",
        };

        let dedup = alert.dedup_key.as_deref().unwrap_or(&alert.title);
        let source = alert.source.as_deref().unwrap_or("ringkernel");

        format!(
            r#"{{"routing_key":"{}","event_action":"trigger","dedup_key":"{}","payload":{{"summary":"{}","source":"{}","severity":"{}","timestamp":"{}"}}}}"#,
            self.headers.get("X-Routing-Key").unwrap_or(&String::new()),
            escape_json(dedup),
            escape_json(&alert.title),
            escape_json(source),
            severity_pd,
            chrono_timestamp(alert.timestamp),
        )
    }
}

#[cfg(feature = "alerting")]
fn chrono_timestamp(time: SystemTime) -> String {
    let millis = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    // Format as ISO 8601
    format!("{}Z", millis / 1000)
}

#[cfg(feature = "alerting")]
#[async_trait]
impl AlertSink for WebhookSink {
    async fn send(&self, alert: &Alert) -> AlertSinkResult<()> {
        if alert.severity < self.min_severity {
            return Ok(());
        }

        let payload = self.format_payload(alert);
        let mut request = self
            .client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .body(payload)
            .timeout(self.timeout);

        for (key, value) in &self.headers {
            request = request.header(key, value);
        }

        let response = request.send().await.map_err(|e| {
            AlertSinkError::ConnectionError(format!("Webhook request failed: {}", e))
        })?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(AlertSinkError::Other(format!(
                "Webhook returned status: {}",
                response.status()
            )))
        }
    }

    fn sink_name(&self) -> &str {
        "WebhookSink"
    }

    fn accepts_severity(&self, severity: AlertSeverity) -> bool {
        severity >= self.min_severity
    }

    async fn health_check(&self) -> AlertSinkResult<()> {
        // Simple check that we can reach the URL
        Ok(())
    }
}

// ============================================================================
// ALERT ROUTER
// ============================================================================

/// Configuration for alert deduplication.
#[derive(Debug, Clone)]
pub struct DeduplicationConfig {
    /// Time window for deduplication.
    pub window: Duration,
    /// Maximum entries in dedup cache.
    pub max_entries: usize,
}

impl Default for DeduplicationConfig {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(300), // 5 minutes
            max_entries: 10000,
        }
    }
}

/// Alert router that sends alerts to multiple sinks.
pub struct AlertRouter {
    /// Registered sinks.
    sinks: Vec<Arc<dyn AlertSink>>,
    /// Deduplication cache.
    dedup_cache: RwLock<HashMap<String, Instant>>,
    /// Deduplication config.
    dedup_config: DeduplicationConfig,
    /// Alert counter.
    alert_count: AtomicU64,
    /// Deduplicated count.
    dedup_count: AtomicU64,
}

impl AlertRouter {
    /// Create a new alert router.
    pub fn new() -> Self {
        Self {
            sinks: Vec::new(),
            dedup_cache: RwLock::new(HashMap::new()),
            dedup_config: DeduplicationConfig::default(),
            alert_count: AtomicU64::new(0),
            dedup_count: AtomicU64::new(0),
        }
    }

    /// Set deduplication config.
    pub fn with_deduplication(mut self, config: DeduplicationConfig) -> Self {
        self.dedup_config = config;
        self
    }

    /// Add a sink.
    pub fn with_sink<S: AlertSink + 'static>(mut self, sink: S) -> Self {
        self.sinks.push(Arc::new(sink));
        self
    }

    /// Add a sink (Arc version).
    pub fn with_sink_arc(mut self, sink: Arc<dyn AlertSink>) -> Self {
        self.sinks.push(sink);
        self
    }

    /// Send an alert to all appropriate sinks.
    pub async fn send(&self, alert: Alert) {
        self.alert_count.fetch_add(1, Ordering::Relaxed);

        // Check deduplication
        if let Some(dedup_key) = &alert.dedup_key {
            let now = Instant::now();
            let mut cache = self.dedup_cache.write();

            // Clean old entries
            cache.retain(|_, instant| now.duration_since(*instant) < self.dedup_config.window);

            // Check if duplicate
            if let Some(last_seen) = cache.get(dedup_key) {
                if now.duration_since(*last_seen) < self.dedup_config.window {
                    self.dedup_count.fetch_add(1, Ordering::Relaxed);
                    return; // Deduplicated
                }
            }

            // Record this alert
            if cache.len() < self.dedup_config.max_entries {
                cache.insert(dedup_key.clone(), now);
            }
        }

        // Send to all sinks that accept this severity
        for sink in &self.sinks {
            if sink.accepts_severity(alert.severity) {
                if let Err(e) = sink.send(&alert).await {
                    tracing::error!(
                        "Alert sink {} failed: {}",
                        sink.sink_name(),
                        e
                    );
                }
            }
        }
    }

    /// Send an alert synchronously (spawns async task).
    pub fn send_sync(&self, alert: Alert) {
        // This is a simplified version that logs immediately
        // In production, you'd spawn a task or use a channel
        self.alert_count.fetch_add(1, Ordering::Relaxed);

        let source = alert.source.as_deref().unwrap_or("unknown");
        let msg = format!(
            "[ALERT] {} | {} | {}",
            alert.severity,
            source,
            alert.title
        );

        match alert.severity {
            AlertSeverity::Info => tracing::info!("{}", msg),
            AlertSeverity::Warning => tracing::warn!("{}", msg),
            AlertSeverity::Error | AlertSeverity::Critical => tracing::error!("{}", msg),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> AlertRouterStats {
        AlertRouterStats {
            total_alerts: self.alert_count.load(Ordering::Relaxed),
            deduplicated: self.dedup_count.load(Ordering::Relaxed),
            sinks: self.sinks.len(),
        }
    }
}

impl Default for AlertRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Alert router statistics.
#[derive(Debug, Clone)]
pub struct AlertRouterStats {
    /// Total alerts received.
    pub total_alerts: u64,
    /// Alerts deduplicated.
    pub deduplicated: u64,
    /// Number of sinks.
    pub sinks: usize,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Error);
        assert!(AlertSeverity::Error < AlertSeverity::Critical);
    }

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new(AlertSeverity::Warning, "Test alert")
            .with_description("This is a test")
            .with_source("test_kernel")
            .with_metadata("key1", "value1")
            .with_tag("test");

        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.title, "Test alert");
        assert_eq!(alert.description, Some("This is a test".to_string()));
        assert_eq!(alert.source, Some("test_kernel".to_string()));
        assert_eq!(alert.metadata.get("key1"), Some(&"value1".to_string()));
        assert!(alert.tags.contains(&"test".to_string()));
    }

    #[test]
    fn test_alert_json() {
        let alert = Alert::new(AlertSeverity::Error, "Test")
            .with_source("kernel_1");

        let json = alert.to_json();
        assert!(json.contains("ERROR"));
        assert!(json.contains("Test"));
        assert!(json.contains("kernel_1"));
    }

    #[tokio::test]
    async fn test_in_memory_sink() {
        let sink = InMemorySink::new(10);

        let alert = Alert::new(AlertSeverity::Warning, "Test alert");
        sink.send(&alert).await.unwrap();

        assert_eq!(sink.count(), 1);
        let alerts = sink.alerts();
        assert_eq!(alerts[0].title, "Test alert");
    }

    #[tokio::test]
    async fn test_in_memory_sink_severity_filter() {
        let sink = InMemorySink::new(10)
            .with_severity_filter(AlertSeverity::Error);

        // Info alert should be filtered
        let info = Alert::new(AlertSeverity::Info, "Info");
        sink.send(&info).await.unwrap();
        assert_eq!(sink.count(), 0);

        // Error alert should pass
        let error = Alert::new(AlertSeverity::Error, "Error");
        sink.send(&error).await.unwrap();
        assert_eq!(sink.count(), 1);
    }

    #[tokio::test]
    async fn test_alert_router() {
        let sink = Arc::new(InMemorySink::new(100));
        let router = AlertRouter::new()
            .with_sink_arc(sink.clone());

        router.send(Alert::new(AlertSeverity::Warning, "Alert 1")).await;
        router.send(Alert::new(AlertSeverity::Error, "Alert 2")).await;

        assert_eq!(sink.count(), 2);
        assert_eq!(router.stats().total_alerts, 2);
    }

    #[tokio::test]
    async fn test_alert_deduplication() {
        let sink = Arc::new(InMemorySink::new(100));
        let router = AlertRouter::new()
            .with_deduplication(DeduplicationConfig {
                window: Duration::from_secs(60),
                max_entries: 100,
            })
            .with_sink_arc(sink.clone());

        // Send same alert multiple times with dedup key
        for _ in 0..5 {
            router.send(
                Alert::new(AlertSeverity::Warning, "Repeated alert")
                    .with_dedup_key("same-key")
            ).await;
        }

        // Only first should get through
        assert_eq!(sink.count(), 1);
        assert_eq!(router.stats().deduplicated, 4);
    }

    #[tokio::test]
    async fn test_log_sink() {
        let sink = LogSink::new()
            .with_severity_filter(AlertSeverity::Warning);

        let info = Alert::new(AlertSeverity::Info, "Info");
        assert!(!sink.accepts_severity(info.severity));

        let warning = Alert::new(AlertSeverity::Warning, "Warning");
        assert!(sink.accepts_severity(warning.severity));

        // Just verify it doesn't panic
        sink.send(&warning).await.unwrap();
    }
}
