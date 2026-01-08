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
}
