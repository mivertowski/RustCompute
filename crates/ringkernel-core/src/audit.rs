//! Audit logging for enterprise security and compliance.
//!
//! This module provides comprehensive audit logging for GPU kernel operations,
//! enabling security monitoring, compliance reporting, and forensic analysis.
//!
//! # Features
//!
//! - Structured audit events with timestamps
//! - Multiple output sinks (file, syslog, custom)
//! - Tamper-evident log chains with checksums
//! - Async-safe audit trail generation
//! - Retention policies and log rotation
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::audit::{AuditLogger, AuditEvent, AuditLevel};
//!
//! let logger = AuditLogger::new()
//!     .with_file_sink("/var/log/ringkernel/audit.log")
//!     .with_retention(Duration::from_days(90))
//!     .build()?;
//!
//! logger.log(AuditEvent::kernel_launched("processor", "cuda"));
//! ```

use std::collections::VecDeque;
use std::fmt;
use std::io::Write;
use std::net::UdpSocket;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};

use crate::hlc::HlcTimestamp;

// ============================================================================
// AUDIT LEVELS
// ============================================================================

/// Audit event severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum AuditLevel {
    /// Informational events (kernel start/stop, config changes).
    Info = 0,
    /// Warning events (degraded performance, retries).
    Warning = 1,
    /// Security-relevant events (authentication, authorization).
    Security = 2,
    /// Critical events (failures, violations).
    Critical = 3,
    /// Compliance-relevant events (data access, retention).
    Compliance = 4,
}

impl AuditLevel {
    /// Get the level name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Security => "SECURITY",
            Self::Critical => "CRITICAL",
            Self::Compliance => "COMPLIANCE",
        }
    }
}

impl fmt::Display for AuditLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// AUDIT EVENT TYPES
// ============================================================================

/// Types of audit events.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AuditEventType {
    // Kernel lifecycle events
    /// Kernel was launched.
    KernelLaunched,
    /// Kernel was terminated.
    KernelTerminated,
    /// Kernel was migrated to another device.
    KernelMigrated,
    /// Kernel checkpoint was created.
    KernelCheckpointed,
    /// Kernel was restored from checkpoint.
    KernelRestored,

    // Message events
    /// Message was sent.
    MessageSent,
    /// Message was received.
    MessageReceived,
    /// Message delivery failed.
    MessageFailed,

    // Security events
    /// Authentication attempt.
    AuthenticationAttempt,
    /// Authorization check.
    AuthorizationCheck,
    /// Configuration change.
    ConfigurationChange,
    /// Security policy violation.
    SecurityViolation,

    // Resource events
    /// GPU memory allocated.
    MemoryAllocated,
    /// GPU memory deallocated.
    MemoryDeallocated,
    /// Resource limit exceeded.
    ResourceLimitExceeded,

    // Health events
    /// Health check performed.
    HealthCheck,
    /// Circuit breaker state changed.
    CircuitBreakerStateChange,
    /// Degradation level changed.
    DegradationChange,

    /// Custom event type for user-defined audit events.
    Custom(String),
}

impl AuditEventType {
    /// Get the event type name.
    pub fn as_str(&self) -> &str {
        match self {
            Self::KernelLaunched => "kernel_launched",
            Self::KernelTerminated => "kernel_terminated",
            Self::KernelMigrated => "kernel_migrated",
            Self::KernelCheckpointed => "kernel_checkpointed",
            Self::KernelRestored => "kernel_restored",
            Self::MessageSent => "message_sent",
            Self::MessageReceived => "message_received",
            Self::MessageFailed => "message_failed",
            Self::AuthenticationAttempt => "authentication_attempt",
            Self::AuthorizationCheck => "authorization_check",
            Self::ConfigurationChange => "configuration_change",
            Self::SecurityViolation => "security_violation",
            Self::MemoryAllocated => "memory_allocated",
            Self::MemoryDeallocated => "memory_deallocated",
            Self::ResourceLimitExceeded => "resource_limit_exceeded",
            Self::HealthCheck => "health_check",
            Self::CircuitBreakerStateChange => "circuit_breaker_state_change",
            Self::DegradationChange => "degradation_change",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl fmt::Display for AuditEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// AUDIT EVENT
// ============================================================================

/// A structured audit event.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    /// Unique event ID.
    pub id: u64,
    /// Event timestamp (wall clock).
    pub timestamp: SystemTime,
    /// HLC timestamp for causal ordering.
    pub hlc: Option<HlcTimestamp>,
    /// Event level.
    pub level: AuditLevel,
    /// Event type.
    pub event_type: AuditEventType,
    /// Actor/component that generated the event.
    pub actor: String,
    /// Target resource or kernel.
    pub target: Option<String>,
    /// Event description.
    pub description: String,
    /// Additional metadata as key-value pairs.
    pub metadata: Vec<(String, String)>,
    /// Previous event checksum (for tamper detection).
    pub prev_checksum: Option<u64>,
    /// This event's checksum.
    pub checksum: u64,
}

impl AuditEvent {
    /// Create a new audit event.
    pub fn new(
        level: AuditLevel,
        event_type: AuditEventType,
        actor: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        let id = next_event_id();
        let timestamp = SystemTime::now();
        let actor = actor.into();
        let description = description.into();

        let mut event = Self {
            id,
            timestamp,
            hlc: None,
            level,
            event_type,
            actor,
            target: None,
            description,
            metadata: Vec::new(),
            prev_checksum: None,
            checksum: 0,
        };

        event.checksum = event.compute_checksum();
        event
    }

    /// Add an HLC timestamp.
    pub fn with_hlc(mut self, hlc: HlcTimestamp) -> Self {
        self.hlc = Some(hlc);
        self.checksum = self.compute_checksum();
        self
    }

    /// Add a target resource.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self.checksum = self.compute_checksum();
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self.checksum = self.compute_checksum();
        self
    }

    /// Set the previous checksum for chain integrity.
    pub fn with_prev_checksum(mut self, checksum: u64) -> Self {
        self.prev_checksum = Some(checksum);
        self.checksum = self.compute_checksum();
        self
    }

    /// Compute a checksum for this event.
    fn compute_checksum(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.id.hash(&mut hasher);
        self.timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .hash(&mut hasher);
        self.level.as_str().hash(&mut hasher);
        self.event_type.as_str().hash(&mut hasher);
        self.actor.hash(&mut hasher);
        self.target.hash(&mut hasher);
        self.description.hash(&mut hasher);
        for (k, v) in &self.metadata {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        self.prev_checksum.hash(&mut hasher);
        hasher.finish()
    }

    /// Verify the event checksum.
    pub fn verify_checksum(&self) -> bool {
        self.checksum == self.compute_checksum()
    }

    // Helper constructors for common events

    /// Create a kernel launched event.
    pub fn kernel_launched(kernel_id: impl Into<String>, backend: impl Into<String>) -> Self {
        Self::new(
            AuditLevel::Info,
            AuditEventType::KernelLaunched,
            "runtime",
            format!("Kernel launched on {}", backend.into()),
        )
        .with_target(kernel_id)
    }

    /// Create a kernel terminated event.
    pub fn kernel_terminated(kernel_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(
            AuditLevel::Info,
            AuditEventType::KernelTerminated,
            "runtime",
            format!("Kernel terminated: {}", reason.into()),
        )
        .with_target(kernel_id)
    }

    /// Create a security violation event.
    pub fn security_violation(actor: impl Into<String>, violation: impl Into<String>) -> Self {
        Self::new(
            AuditLevel::Security,
            AuditEventType::SecurityViolation,
            actor,
            violation,
        )
    }

    /// Create a configuration change event.
    pub fn config_change(
        actor: impl Into<String>,
        config_key: impl Into<String>,
        old_value: impl Into<String>,
        new_value: impl Into<String>,
    ) -> Self {
        Self::new(
            AuditLevel::Compliance,
            AuditEventType::ConfigurationChange,
            actor,
            format!("Configuration changed: {}", config_key.into()),
        )
        .with_metadata("old_value", old_value)
        .with_metadata("new_value", new_value)
    }

    /// Create a health check event.
    pub fn health_check(kernel_id: impl Into<String>, status: impl Into<String>) -> Self {
        Self::new(
            AuditLevel::Info,
            AuditEventType::HealthCheck,
            "health_checker",
            format!("Health check: {}", status.into()),
        )
        .with_target(kernel_id)
    }

    /// Format as JSON.
    pub fn to_json(&self) -> String {
        let timestamp = self
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        let hlc_str = self
            .hlc
            .map(|h| {
                format!(
                    r#","hlc":{{"wall":{},"logical":{}}}"#,
                    h.physical, h.logical
                )
            })
            .unwrap_or_default();

        let target_str = self
            .target
            .as_ref()
            .map(|t| format!(r#","target":"{}""#, escape_json(t)))
            .unwrap_or_default();

        let prev_checksum_str = self
            .prev_checksum
            .map(|c| format!(r#","prev_checksum":{}"#, c))
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

        format!(
            r#"{{"id":{},"timestamp":{}{},"level":"{}","event_type":"{}","actor":"{}"{}"description":"{}"{}"checksum":{}{}}}"#,
            self.id,
            timestamp,
            hlc_str,
            self.level.as_str(),
            self.event_type.as_str(),
            escape_json(&self.actor),
            target_str,
            escape_json(&self.description),
            metadata_str,
            self.checksum,
            prev_checksum_str,
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

// Global event ID counter
static EVENT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_event_id() -> u64 {
    EVENT_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

// ============================================================================
// AUDIT SINK TRAIT
// ============================================================================

/// Trait for audit log output sinks.
pub trait AuditSink: Send + Sync {
    /// Write an audit event to the sink.
    fn write(&self, event: &AuditEvent) -> std::io::Result<()>;

    /// Flush any buffered events.
    fn flush(&self) -> std::io::Result<()>;

    /// Close the sink.
    fn close(&self) -> std::io::Result<()>;
}

/// File-based audit sink.
pub struct FileSink {
    path: PathBuf,
    writer: Mutex<Option<std::fs::File>>,
    max_size: u64,
    current_size: AtomicU64,
}

impl FileSink {
    /// Create a new file sink.
    pub fn new(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        let metadata = file.metadata()?;

        Ok(Self {
            path,
            writer: Mutex::new(Some(file)),
            max_size: 100 * 1024 * 1024, // 100 MB default
            current_size: AtomicU64::new(metadata.len()),
        })
    }

    /// Set the maximum file size before rotation.
    pub fn with_max_size(mut self, size: u64) -> Self {
        self.max_size = size;
        self
    }

    /// Rotate the log file if needed.
    fn rotate_if_needed(&self) -> std::io::Result<()> {
        if self.current_size.load(Ordering::Relaxed) >= self.max_size {
            let mut writer = self.writer.lock();
            if let Some(file) = writer.take() {
                drop(file);

                // Rename current file with timestamp
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let rotated_path = self.path.with_extension(format!("log.{}", timestamp));
                std::fs::rename(&self.path, rotated_path)?;

                // Create new file
                let new_file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path)?;
                *writer = Some(new_file);
                self.current_size.store(0, Ordering::Relaxed);
            }
        }
        Ok(())
    }
}

impl AuditSink for FileSink {
    fn write(&self, event: &AuditEvent) -> std::io::Result<()> {
        self.rotate_if_needed()?;

        let json = event.to_json();
        let line = format!("{}\n", json);
        let len = line.len() as u64;

        let mut writer = self.writer.lock();
        if let Some(file) = writer.as_mut() {
            file.write_all(line.as_bytes())?;
            self.current_size.fetch_add(len, Ordering::Relaxed);
        }
        Ok(())
    }

    fn flush(&self) -> std::io::Result<()> {
        let mut writer = self.writer.lock();
        if let Some(file) = writer.as_mut() {
            file.flush()?;
        }
        Ok(())
    }

    fn close(&self) -> std::io::Result<()> {
        let mut writer = self.writer.lock();
        if let Some(file) = writer.take() {
            drop(file);
        }
        Ok(())
    }
}

/// In-memory audit sink for testing.
#[derive(Default)]
pub struct MemorySink {
    events: Mutex<VecDeque<AuditEvent>>,
    max_events: usize,
}

impl MemorySink {
    /// Create a new memory sink.
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Mutex::new(VecDeque::with_capacity(max_events)),
            max_events,
        }
    }

    /// Get all stored events.
    pub fn events(&self) -> Vec<AuditEvent> {
        self.events.lock().iter().cloned().collect()
    }

    /// Get the count of events.
    pub fn len(&self) -> usize {
        self.events.lock().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.events.lock().is_empty()
    }

    /// Clear all events.
    pub fn clear(&self) {
        self.events.lock().clear();
    }
}

impl AuditSink for MemorySink {
    fn write(&self, event: &AuditEvent) -> std::io::Result<()> {
        let mut events = self.events.lock();
        if events.len() >= self.max_events {
            events.pop_front();
        }
        events.push_back(event.clone());
        Ok(())
    }

    fn flush(&self) -> std::io::Result<()> {
        Ok(())
    }

    fn close(&self) -> std::io::Result<()> {
        Ok(())
    }
}

// ============================================================================
// SYSLOG SINK (RFC 5424)
// ============================================================================

/// Syslog facility codes (RFC 5424).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SyslogFacility {
    /// Kernel messages.
    Kern = 0,
    /// User-level messages.
    User = 1,
    /// Security/authorization messages.
    Auth = 4,
    /// Security/authorization messages (private).
    AuthPriv = 10,
    /// Local use 0.
    Local0 = 16,
    /// Local use 1.
    Local1 = 17,
    /// Local use 2.
    Local2 = 18,
    /// Local use 3.
    Local3 = 19,
    /// Local use 4.
    Local4 = 20,
    /// Local use 5.
    Local5 = 21,
    /// Local use 6.
    Local6 = 22,
    /// Local use 7.
    Local7 = 23,
}

/// Syslog severity codes (RFC 5424).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SyslogSeverity {
    /// System is unusable.
    Emergency = 0,
    /// Action must be taken immediately.
    Alert = 1,
    /// Critical conditions.
    Critical = 2,
    /// Error conditions.
    Error = 3,
    /// Warning conditions.
    Warning = 4,
    /// Normal but significant condition.
    Notice = 5,
    /// Informational messages.
    Informational = 6,
    /// Debug-level messages.
    Debug = 7,
}

impl From<AuditLevel> for SyslogSeverity {
    fn from(level: AuditLevel) -> Self {
        match level {
            AuditLevel::Info => SyslogSeverity::Informational,
            AuditLevel::Warning => SyslogSeverity::Warning,
            AuditLevel::Security => SyslogSeverity::Notice,
            AuditLevel::Critical => SyslogSeverity::Error,
            AuditLevel::Compliance => SyslogSeverity::Notice,
        }
    }
}

/// Configuration for syslog sink.
#[derive(Debug, Clone)]
pub struct SyslogConfig {
    /// Syslog server address (e.g., "127.0.0.1:514").
    pub server_addr: String,
    /// Facility code.
    pub facility: SyslogFacility,
    /// Application name (APP-NAME in RFC 5424).
    pub app_name: String,
    /// Process ID (PROCID in RFC 5424).
    pub procid: Option<String>,
    /// Message ID (MSGID in RFC 5424).
    pub msgid: Option<String>,
    /// Use RFC 5424 format (true) or BSD format (false).
    pub rfc5424: bool,
}

impl Default for SyslogConfig {
    fn default() -> Self {
        Self {
            server_addr: "127.0.0.1:514".to_string(),
            facility: SyslogFacility::Local0,
            app_name: "ringkernel".to_string(),
            procid: None,
            msgid: None,
            rfc5424: true,
        }
    }
}

/// RFC 5424 syslog sink for remote audit log forwarding.
pub struct SyslogSink {
    config: SyslogConfig,
    socket: Mutex<Option<UdpSocket>>,
    hostname: String,
}

impl SyslogSink {
    /// Create a new syslog sink with the given configuration.
    pub fn new(config: SyslogConfig) -> std::io::Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        socket.connect(&config.server_addr)?;

        // Get hostname
        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("HOST"))
            .unwrap_or_else(|_| "localhost".to_string());

        Ok(Self {
            config,
            socket: Mutex::new(Some(socket)),
            hostname,
        })
    }

    /// Create a syslog sink with default configuration.
    pub fn with_server(server_addr: impl Into<String>) -> std::io::Result<Self> {
        Self::new(SyslogConfig {
            server_addr: server_addr.into(),
            ..Default::default()
        })
    }

    /// Format an audit event as RFC 5424 syslog message.
    fn format_rfc5424(&self, event: &AuditEvent) -> String {
        let severity: SyslogSeverity = event.level.into();
        let priority = (self.config.facility as u8) * 8 + (severity as u8);

        // RFC 5424 timestamp format
        let timestamp = event
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let secs = timestamp.as_secs();
        let millis = timestamp.subsec_millis();

        // Simple ISO 8601 format (we don't have chrono, so approximate)
        let epoch_days = secs / 86400;
        let day_secs = secs % 86400;
        let hours = day_secs / 3600;
        let minutes = (day_secs % 3600) / 60;
        let seconds = day_secs % 60;

        // Approximate date calculation (not accounting for leap years perfectly)
        let year = 1970 + (epoch_days / 365);
        let day_of_year = epoch_days % 365;
        let month = (day_of_year / 30).min(11) + 1;
        let day = (day_of_year % 30) + 1;

        let timestamp_str = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
            year, month, day, hours, minutes, seconds, millis
        );

        let procid = self.config.procid.as_deref().unwrap_or("-");
        let msgid = self.config.msgid.as_deref().unwrap_or("-");

        // Structured data (SD-ELEMENT)
        let sd = format!(
            "[ringkernel@12345 level=\"{}\" event_type=\"{}\" actor=\"{}\" checksum=\"{}\"]",
            event.level.as_str(),
            event.event_type.as_str(),
            event.actor,
            event.checksum
        );

        format!(
            "<{}>{} {} {} {} {} {} {} {}",
            priority,
            1, // version
            timestamp_str,
            self.hostname,
            self.config.app_name,
            procid,
            msgid,
            sd,
            event.description
        )
    }

    /// Format an audit event as BSD syslog message.
    fn format_bsd(&self, event: &AuditEvent) -> String {
        let severity: SyslogSeverity = event.level.into();
        let priority = (self.config.facility as u8) * 8 + (severity as u8);

        let timestamp = event
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let secs = timestamp.as_secs();

        // BSD syslog timestamp format (Mmm dd hh:mm:ss)
        let months = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];
        let epoch_days = secs / 86400;
        let day_secs = secs % 86400;
        let hours = day_secs / 3600;
        let minutes = (day_secs % 3600) / 60;
        let seconds = day_secs % 60;

        let day_of_year = epoch_days % 365;
        let month_idx = ((day_of_year / 30) as usize).min(11);
        let day = (day_of_year % 30) + 1;

        let timestamp_str = format!(
            "{} {:2} {:02}:{:02}:{:02}",
            months[month_idx], day, hours, minutes, seconds
        );

        format!(
            "<{}>{} {} {}: [{}] {}",
            priority,
            timestamp_str,
            self.hostname,
            self.config.app_name,
            event.event_type.as_str(),
            event.description
        )
    }
}

impl AuditSink for SyslogSink {
    fn write(&self, event: &AuditEvent) -> std::io::Result<()> {
        let message = if self.config.rfc5424 {
            self.format_rfc5424(event)
        } else {
            self.format_bsd(event)
        };

        let socket = self.socket.lock();
        if let Some(ref sock) = *socket {
            sock.send(message.as_bytes())?;
        }
        Ok(())
    }

    fn flush(&self) -> std::io::Result<()> {
        Ok(())
    }

    fn close(&self) -> std::io::Result<()> {
        let mut socket = self.socket.lock();
        *socket = None;
        Ok(())
    }
}

// ============================================================================
// ELASTICSEARCH SINK (requires alerting feature for reqwest)
// ============================================================================

/// Configuration for Elasticsearch audit sink.
#[cfg(feature = "alerting")]
#[derive(Debug, Clone)]
pub struct ElasticsearchConfig {
    /// Elasticsearch URL (e.g., "http://localhost:9200").
    pub url: String,
    /// Index name or pattern (e.g., "ringkernel-audit-{date}").
    pub index_pattern: String,
    /// Optional authentication (Basic auth).
    pub auth: Option<(String, String)>,
    /// Batch size before flushing.
    pub batch_size: usize,
    /// Request timeout.
    pub timeout: Duration,
}

#[cfg(feature = "alerting")]
impl Default for ElasticsearchConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:9200".to_string(),
            index_pattern: "ringkernel-audit".to_string(),
            auth: None,
            batch_size: 100,
            timeout: Duration::from_secs(30),
        }
    }
}

/// Elasticsearch sink for direct indexing of audit events.
#[cfg(feature = "alerting")]
pub struct ElasticsearchSink {
    config: ElasticsearchConfig,
    client: reqwest::blocking::Client,
    buffer: Mutex<Vec<String>>,
}

#[cfg(feature = "alerting")]
impl ElasticsearchSink {
    /// Create a new Elasticsearch sink.
    pub fn new(config: ElasticsearchConfig) -> Result<Self, reqwest::Error> {
        let client = reqwest::blocking::Client::builder()
            .timeout(config.timeout)
            .build()?;

        Ok(Self {
            config,
            client,
            buffer: Mutex::new(Vec::new()),
        })
    }

    /// Get the index name for an event.
    fn get_index(&self, event: &AuditEvent) -> String {
        let timestamp = event
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let secs = timestamp.as_secs();

        // Calculate date components
        let epoch_days = secs / 86400;
        let year = 1970 + (epoch_days / 365);
        let day_of_year = epoch_days % 365;
        let month = (day_of_year / 30).min(11) + 1;
        let day = (day_of_year % 30) + 1;

        let date_str = format!("{:04}.{:02}.{:02}", year, month, day);

        self.config
            .index_pattern
            .replace("{date}", &date_str)
            .replace("{year}", &format!("{:04}", year))
            .replace("{month}", &format!("{:02}", month))
            .replace("{day}", &format!("{:02}", day))
    }

    /// Convert an audit event to Elasticsearch document JSON.
    fn to_es_document(&self, event: &AuditEvent) -> String {
        let timestamp_millis = event
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        // Build metadata as nested JSON
        let metadata_json = if event.metadata.is_empty() {
            "{}".to_string()
        } else {
            let pairs: Vec<String> = event
                .metadata
                .iter()
                .map(|(k, v)| format!(r#""{}":"{}""#, escape_json(k), escape_json(v)))
                .collect();
            format!("{{{}}}", pairs.join(","))
        };

        let hlc_json = event
            .hlc
            .map(|h| format!(r#","hlc":{{"physical":{},"logical":{}}}"#, h.physical, h.logical))
            .unwrap_or_default();

        let target_json = event
            .target
            .as_ref()
            .map(|t| format!(r#","target":"{}""#, escape_json(t)))
            .unwrap_or_default();

        format!(
            r#"{{"@timestamp":{},"id":{},"level":"{}","event_type":"{}","actor":"{}"{}{}"description":"{}","metadata":{},"checksum":{}}}"#,
            timestamp_millis,
            event.id,
            event.level.as_str(),
            event.event_type.as_str(),
            escape_json(&event.actor),
            target_json,
            hlc_json,
            escape_json(&event.description),
            metadata_json,
            event.checksum
        )
    }

    /// Flush the buffer to Elasticsearch using bulk API.
    fn flush_buffer(&self) -> std::io::Result<()> {
        let documents: Vec<String> = {
            let mut buffer = self.buffer.lock();
            std::mem::take(&mut *buffer)
        };

        if documents.is_empty() {
            return Ok(());
        }

        // Build bulk request body
        let mut bulk_body = String::new();
        for doc in documents {
            // Action line
            bulk_body.push_str(&format!(
                r#"{{"index":{{"_index":"{}"}}}}"#,
                self.config.index_pattern.replace("{date}", "current")
            ));
            bulk_body.push('\n');
            // Document line
            bulk_body.push_str(&doc);
            bulk_body.push('\n');
        }

        let url = format!("{}/_bulk", self.config.url);
        let mut request = self.client.post(&url).body(bulk_body).header(
            reqwest::header::CONTENT_TYPE,
            "application/x-ndjson",
        );

        if let Some((user, pass)) = &self.config.auth {
            request = request.basic_auth(user, Some(pass));
        }

        request.send().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, format!("ES request failed: {}", e))
        })?;

        Ok(())
    }
}

#[cfg(feature = "alerting")]
impl AuditSink for ElasticsearchSink {
    fn write(&self, event: &AuditEvent) -> std::io::Result<()> {
        let doc = self.to_es_document(event);

        let should_flush = {
            let mut buffer = self.buffer.lock();
            buffer.push(doc);
            buffer.len() >= self.config.batch_size
        };

        if should_flush {
            self.flush_buffer()?;
        }

        Ok(())
    }

    fn flush(&self) -> std::io::Result<()> {
        self.flush_buffer()
    }

    fn close(&self) -> std::io::Result<()> {
        self.flush_buffer()
    }
}

// ============================================================================
// CLOUDWATCH LOGS SINK (stub - requires aws-sdk-cloudwatchlogs)
// ============================================================================

/// Configuration for CloudWatch Logs sink.
#[derive(Debug, Clone)]
pub struct CloudWatchConfig {
    /// Log group name.
    pub log_group: String,
    /// Log stream name.
    pub log_stream: String,
    /// AWS region.
    pub region: String,
    /// Batch size before flushing.
    pub batch_size: usize,
}

impl Default for CloudWatchConfig {
    fn default() -> Self {
        Self {
            log_group: "/ringkernel/audit".to_string(),
            log_stream: "default".to_string(),
            region: "us-east-1".to_string(),
            batch_size: 100,
        }
    }
}

/// CloudWatch Logs sink for AWS-native audit logging.
///
/// Note: This is a stub implementation. For production use, enable the
/// `cloudwatch` feature and use the AWS SDK for CloudWatch Logs.
pub struct CloudWatchSink {
    config: CloudWatchConfig,
    buffer: Mutex<Vec<(u64, String)>>, // (timestamp_ms, message)
    sequence_token: Mutex<Option<String>>,
}

impl CloudWatchSink {
    /// Create a new CloudWatch Logs sink.
    pub fn new(config: CloudWatchConfig) -> Self {
        Self {
            config,
            buffer: Mutex::new(Vec::new()),
            sequence_token: Mutex::new(None),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &CloudWatchConfig {
        &self.config
    }

    /// Get the current buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer.lock().len()
    }
}

impl AuditSink for CloudWatchSink {
    fn write(&self, event: &AuditEvent) -> std::io::Result<()> {
        let timestamp_ms = event
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let message = event.to_json();

        let should_flush = {
            let mut buffer = self.buffer.lock();
            buffer.push((timestamp_ms, message));
            buffer.len() >= self.config.batch_size
        };

        if should_flush {
            self.flush()?;
        }

        Ok(())
    }

    fn flush(&self) -> std::io::Result<()> {
        let events: Vec<(u64, String)> = {
            let mut buffer = self.buffer.lock();
            std::mem::take(&mut *buffer)
        };

        if events.is_empty() {
            return Ok(());
        }

        // Note: In a real implementation, this would use aws-sdk-cloudwatchlogs
        // to call PutLogEvents. For now, we log to stderr as a fallback.
        eprintln!(
            "[CloudWatch stub] Would send {} events to {}/{}",
            events.len(),
            self.config.log_group,
            self.config.log_stream
        );

        Ok(())
    }

    fn close(&self) -> std::io::Result<()> {
        self.flush()
    }
}

// ============================================================================
// AUDIT LOGGER
// ============================================================================

/// Configuration for the audit logger.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Minimum level to log.
    pub min_level: AuditLevel,
    /// Whether to include checksums.
    pub enable_checksums: bool,
    /// Buffer size before flushing.
    pub buffer_size: usize,
    /// Flush interval.
    pub flush_interval: Duration,
    /// Retention period.
    pub retention: Duration,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            min_level: AuditLevel::Info,
            enable_checksums: true,
            buffer_size: 100,
            flush_interval: Duration::from_secs(5),
            retention: Duration::from_secs(90 * 24 * 60 * 60), // 90 days
        }
    }
}

/// Builder for AuditLogger.
pub struct AuditLoggerBuilder {
    config: AuditConfig,
    sinks: Vec<Arc<dyn AuditSink>>,
}

impl AuditLoggerBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: AuditConfig::default(),
            sinks: Vec::new(),
        }
    }

    /// Set the minimum log level.
    pub fn with_min_level(mut self, level: AuditLevel) -> Self {
        self.config.min_level = level;
        self
    }

    /// Add a file sink.
    pub fn with_file_sink(mut self, path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let sink = Arc::new(FileSink::new(path)?);
        self.sinks.push(sink);
        Ok(self)
    }

    /// Add a memory sink.
    pub fn with_memory_sink(mut self, max_events: usize) -> Self {
        let sink = Arc::new(MemorySink::new(max_events));
        self.sinks.push(sink);
        self
    }

    /// Add a custom sink.
    pub fn with_sink(mut self, sink: Arc<dyn AuditSink>) -> Self {
        self.sinks.push(sink);
        self
    }

    /// Add a syslog sink.
    pub fn with_syslog_sink(mut self, config: SyslogConfig) -> std::io::Result<Self> {
        let sink = Arc::new(SyslogSink::new(config)?);
        self.sinks.push(sink);
        Ok(self)
    }

    /// Add a syslog sink with just a server address.
    pub fn with_syslog(mut self, server_addr: impl Into<String>) -> std::io::Result<Self> {
        let sink = Arc::new(SyslogSink::with_server(server_addr)?);
        self.sinks.push(sink);
        Ok(self)
    }

    /// Add a CloudWatch Logs sink.
    pub fn with_cloudwatch_sink(mut self, config: CloudWatchConfig) -> Self {
        let sink = Arc::new(CloudWatchSink::new(config));
        self.sinks.push(sink);
        self
    }

    /// Add an Elasticsearch sink (requires `alerting` feature).
    #[cfg(feature = "alerting")]
    pub fn with_elasticsearch_sink(
        mut self,
        config: ElasticsearchConfig,
    ) -> Result<Self, reqwest::Error> {
        let sink = Arc::new(ElasticsearchSink::new(config)?);
        self.sinks.push(sink);
        Ok(self)
    }

    /// Set the retention period.
    pub fn with_retention(mut self, retention: Duration) -> Self {
        self.config.retention = retention;
        self
    }

    /// Enable or disable checksums.
    pub fn with_checksums(mut self, enable: bool) -> Self {
        self.config.enable_checksums = enable;
        self
    }

    /// Build the logger.
    pub fn build(self) -> AuditLogger {
        AuditLogger {
            config: self.config,
            sinks: self.sinks,
            last_checksum: AtomicU64::new(0),
            event_count: AtomicU64::new(0),
            buffer: RwLock::new(Vec::new()),
        }
    }
}

impl Default for AuditLoggerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The main audit logger.
pub struct AuditLogger {
    config: AuditConfig,
    sinks: Vec<Arc<dyn AuditSink>>,
    last_checksum: AtomicU64,
    event_count: AtomicU64,
    buffer: RwLock<Vec<AuditEvent>>,
}

impl AuditLogger {
    /// Create a new logger builder.
    pub fn builder() -> AuditLoggerBuilder {
        AuditLoggerBuilder::new()
    }

    /// Create a simple in-memory logger for testing.
    pub fn in_memory(max_events: usize) -> Self {
        AuditLoggerBuilder::new()
            .with_memory_sink(max_events)
            .build()
    }

    /// Log an audit event.
    pub fn log(&self, mut event: AuditEvent) {
        // Check level
        if event.level < self.config.min_level {
            return;
        }

        // Add chain checksum if enabled
        if self.config.enable_checksums {
            let prev = self.last_checksum.load(Ordering::Acquire);
            event = event.with_prev_checksum(prev);
            self.last_checksum.store(event.checksum, Ordering::Release);
        }

        // Write to all sinks
        for sink in &self.sinks {
            if let Err(e) = sink.write(&event) {
                eprintln!("Audit sink error: {}", e);
            }
        }

        self.event_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Log a kernel launch event.
    pub fn log_kernel_launched(&self, kernel_id: &str, backend: &str) {
        self.log(AuditEvent::kernel_launched(kernel_id, backend));
    }

    /// Log a kernel termination event.
    pub fn log_kernel_terminated(&self, kernel_id: &str, reason: &str) {
        self.log(AuditEvent::kernel_terminated(kernel_id, reason));
    }

    /// Log a security violation.
    pub fn log_security_violation(&self, actor: &str, violation: &str) {
        self.log(AuditEvent::security_violation(actor, violation));
    }

    /// Log a configuration change.
    pub fn log_config_change(&self, actor: &str, key: &str, old_value: &str, new_value: &str) {
        self.log(AuditEvent::config_change(actor, key, old_value, new_value));
    }

    /// Get the total event count.
    pub fn event_count(&self) -> u64 {
        self.event_count.load(Ordering::Relaxed)
    }

    /// Buffer an event for batch processing.
    ///
    /// Events buffered with this method can be flushed with `flush_buffered`.
    pub fn buffer_event(&self, event: AuditEvent) {
        let mut buffer = self.buffer.write();
        buffer.push(event);
    }

    /// Flush all buffered events to sinks.
    pub fn flush_buffered(&self) -> std::io::Result<()> {
        let events: Vec<AuditEvent> = {
            let mut buffer = self.buffer.write();
            std::mem::take(&mut *buffer)
        };

        for mut event in events {
            // Add chain checksum if enabled
            if self.config.enable_checksums {
                let prev = self.last_checksum.load(Ordering::Acquire);
                event = event.with_prev_checksum(prev);
                self.last_checksum.store(event.checksum, Ordering::Release);
            }

            // Write to all sinks
            for sink in &self.sinks {
                sink.write(&event)?;
            }

            self.event_count.fetch_add(1, Ordering::Relaxed);
        }

        self.flush()
    }

    /// Get the count of buffered events.
    pub fn buffered_count(&self) -> usize {
        self.buffer.read().len()
    }

    /// Flush all sinks.
    pub fn flush(&self) -> std::io::Result<()> {
        for sink in &self.sinks {
            sink.flush()?;
        }
        Ok(())
    }

    /// Close all sinks.
    pub fn close(&self) -> std::io::Result<()> {
        for sink in &self.sinks {
            sink.close()?;
        }
        Ok(())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_event_creation() {
        let event = AuditEvent::new(
            AuditLevel::Info,
            AuditEventType::KernelLaunched,
            "runtime",
            "Kernel launched",
        );

        assert_eq!(event.level, AuditLevel::Info);
        assert_eq!(event.event_type, AuditEventType::KernelLaunched);
        assert_eq!(event.actor, "runtime");
        assert!(event.checksum != 0);
    }

    #[test]
    fn test_audit_event_checksum() {
        let event = AuditEvent::kernel_launched("test_kernel", "cuda");
        assert!(event.verify_checksum());

        // Modifying the event should invalidate the checksum
        let mut modified = event.clone();
        modified.description = "Modified".to_string();
        assert!(!modified.verify_checksum());
    }

    #[test]
    fn test_audit_event_chain() {
        let event1 = AuditEvent::kernel_launched("k1", "cuda");
        let event2 = AuditEvent::kernel_launched("k2", "cuda").with_prev_checksum(event1.checksum);

        assert_eq!(event2.prev_checksum, Some(event1.checksum));
    }

    #[test]
    fn test_audit_event_json() {
        let event = AuditEvent::kernel_launched("test", "cuda")
            .with_metadata("gpu_id", "0")
            .with_metadata("memory_mb", "8192");

        let json = event.to_json();
        assert!(json.contains("kernel_launched"));
        assert!(json.contains("test"));
        assert!(json.contains("cuda"));
        assert!(json.contains("gpu_id"));
    }

    #[test]
    fn test_memory_sink() {
        let sink = MemorySink::new(10);

        let event = AuditEvent::kernel_launched("test", "cuda");
        sink.write(&event).unwrap();

        assert_eq!(sink.len(), 1);
        assert!(!sink.is_empty());

        let events = sink.events();
        assert_eq!(events[0].event_type, AuditEventType::KernelLaunched);
    }

    #[test]
    fn test_memory_sink_rotation() {
        let sink = MemorySink::new(3);

        for i in 0..5 {
            let event = AuditEvent::new(
                AuditLevel::Info,
                AuditEventType::Custom(format!("event_{}", i)),
                "test",
                format!("Event {}", i),
            );
            sink.write(&event).unwrap();
        }

        // Should only keep the last 3
        assert_eq!(sink.len(), 3);
        let events = sink.events();
        assert_eq!(
            events[0].event_type,
            AuditEventType::Custom("event_2".to_string())
        );
    }

    #[test]
    fn test_audit_logger() {
        let logger = AuditLogger::in_memory(100);

        logger.log_kernel_launched("k1", "cuda");
        logger.log_kernel_terminated("k1", "shutdown");
        logger.log_security_violation("user", "unauthorized access");

        assert_eq!(logger.event_count(), 3);
    }

    #[test]
    fn test_audit_level_ordering() {
        assert!(AuditLevel::Info < AuditLevel::Warning);
        assert!(AuditLevel::Warning < AuditLevel::Security);
        assert!(AuditLevel::Security < AuditLevel::Critical);
        assert!(AuditLevel::Critical < AuditLevel::Compliance);
    }

    #[test]
    fn test_audit_event_helpers() {
        let event = AuditEvent::config_change("admin", "max_kernels", "10", "20");
        assert_eq!(event.level, AuditLevel::Compliance);
        assert_eq!(event.metadata.len(), 2);

        let health = AuditEvent::health_check("kernel_1", "healthy");
        assert_eq!(health.event_type, AuditEventType::HealthCheck);
    }

    #[test]
    fn test_syslog_severity_conversion() {
        assert_eq!(
            SyslogSeverity::from(AuditLevel::Info),
            SyslogSeverity::Informational
        );
        assert_eq!(
            SyslogSeverity::from(AuditLevel::Warning),
            SyslogSeverity::Warning
        );
        assert_eq!(
            SyslogSeverity::from(AuditLevel::Security),
            SyslogSeverity::Notice
        );
        assert_eq!(
            SyslogSeverity::from(AuditLevel::Critical),
            SyslogSeverity::Error
        );
    }

    #[test]
    fn test_syslog_config_default() {
        let config = SyslogConfig::default();
        assert_eq!(config.server_addr, "127.0.0.1:514");
        assert_eq!(config.facility, SyslogFacility::Local0);
        assert_eq!(config.app_name, "ringkernel");
        assert!(config.rfc5424);
    }

    #[test]
    fn test_cloudwatch_config_default() {
        let config = CloudWatchConfig::default();
        assert_eq!(config.log_group, "/ringkernel/audit");
        assert_eq!(config.log_stream, "default");
        assert_eq!(config.region, "us-east-1");
        assert_eq!(config.batch_size, 100);
    }

    #[test]
    fn test_cloudwatch_sink_buffering() {
        let config = CloudWatchConfig {
            batch_size: 5,
            ..Default::default()
        };
        let sink = CloudWatchSink::new(config);

        // Write 3 events (below batch size)
        for i in 0..3 {
            let event = AuditEvent::new(
                AuditLevel::Info,
                AuditEventType::Custom(format!("event_{}", i)),
                "test",
                format!("Event {}", i),
            );
            sink.write(&event).unwrap();
        }

        assert_eq!(sink.buffer_size(), 3);
    }

    #[test]
    fn test_syslog_facility_values() {
        assert_eq!(SyslogFacility::Kern as u8, 0);
        assert_eq!(SyslogFacility::User as u8, 1);
        assert_eq!(SyslogFacility::Auth as u8, 4);
        assert_eq!(SyslogFacility::Local0 as u8, 16);
        assert_eq!(SyslogFacility::Local7 as u8, 23);
    }

    #[test]
    fn test_syslog_severity_values() {
        assert_eq!(SyslogSeverity::Emergency as u8, 0);
        assert_eq!(SyslogSeverity::Alert as u8, 1);
        assert_eq!(SyslogSeverity::Critical as u8, 2);
        assert_eq!(SyslogSeverity::Error as u8, 3);
        assert_eq!(SyslogSeverity::Warning as u8, 4);
        assert_eq!(SyslogSeverity::Notice as u8, 5);
        assert_eq!(SyslogSeverity::Informational as u8, 6);
        assert_eq!(SyslogSeverity::Debug as u8, 7);
    }
}
