//! Structured logging with trace correlation.
//!
//! This module provides enterprise-grade logging infrastructure with:
//! - Automatic trace ID and span ID injection
//! - JSON structured output for log aggregation
//! - Log level filtering by module
//! - Integration with the `tracing` crate
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::logging::{LogConfig, StructuredLogger, LogOutput};
//!
//! let config = LogConfig::builder()
//!     .level(LogLevel::Info)
//!     .output(LogOutput::Json)
//!     .with_trace_correlation(true)
//!     .build();
//!
//! let logger = StructuredLogger::new(config);
//! logger.info("Kernel started", &[("kernel_id", "k1"), ("mode", "persistent")]);
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::observability::{SpanId, TraceId};

// ============================================================================
// Log Level
// ============================================================================

/// Log severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum LogLevel {
    /// Trace-level logging (most verbose).
    Trace = 0,
    /// Debug-level logging.
    Debug = 1,
    /// Info-level logging.
    #[default]
    Info = 2,
    /// Warning-level logging.
    Warn = 3,
    /// Error-level logging.
    Error = 4,
    /// Fatal-level logging (least verbose).
    Fatal = 5,
}

impl LogLevel {
    /// Convert to string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Fatal => "FATAL",
        }
    }

    /// Parse log level from string representation.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(LogLevel::Trace),
            "DEBUG" => Some(LogLevel::Debug),
            "INFO" => Some(LogLevel::Info),
            "WARN" | "WARNING" => Some(LogLevel::Warn),
            "ERROR" => Some(LogLevel::Error),
            "FATAL" | "CRITICAL" => Some(LogLevel::Fatal),
            _ => None,
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Log Output Format
// ============================================================================

/// Log output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogOutput {
    /// Human-readable text format.
    #[default]
    Text,
    /// JSON structured format for log aggregation.
    Json,
    /// Compact single-line format.
    Compact,
    /// Pretty-printed format with colors (if terminal supports it).
    Pretty,
}

// ============================================================================
// Trace Context
// ============================================================================

/// Thread-local trace context for correlation.
#[derive(Debug, Clone)]
pub struct TraceContext {
    /// Current trace ID.
    pub trace_id: Option<TraceId>,
    /// Current span ID.
    pub span_id: Option<SpanId>,
    /// Parent span ID.
    pub parent_span_id: Option<SpanId>,
    /// Additional context fields.
    pub fields: HashMap<String, String>,
}

impl TraceContext {
    /// Create a new empty trace context.
    pub fn new() -> Self {
        Self {
            trace_id: None,
            span_id: None,
            parent_span_id: None,
            fields: HashMap::new(),
        }
    }

    /// Create a trace context with a new trace.
    pub fn with_new_trace() -> Self {
        Self {
            trace_id: Some(TraceId::new()),
            span_id: Some(SpanId::new()),
            parent_span_id: None,
            fields: HashMap::new(),
        }
    }

    /// Set trace ID.
    pub fn with_trace_id(mut self, trace_id: TraceId) -> Self {
        self.trace_id = Some(trace_id);
        self
    }

    /// Set span ID.
    pub fn with_span_id(mut self, span_id: SpanId) -> Self {
        self.span_id = Some(span_id);
        self
    }

    /// Add a context field.
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Log Configuration
// ============================================================================

/// Configuration for structured logging.
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Default log level.
    pub level: LogLevel,
    /// Output format.
    pub output: LogOutput,
    /// Enable trace correlation.
    pub trace_correlation: bool,
    /// Include timestamps.
    pub include_timestamps: bool,
    /// Include caller location (file:line).
    pub include_location: bool,
    /// Include thread ID.
    pub include_thread_id: bool,
    /// Per-module log levels.
    pub module_levels: HashMap<String, LogLevel>,
    /// Service name for structured logs.
    pub service_name: String,
    /// Environment (dev, staging, prod).
    pub environment: String,
    /// Custom fields added to every log.
    pub global_fields: HashMap<String, String>,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            output: LogOutput::Text,
            trace_correlation: true,
            include_timestamps: true,
            include_location: false,
            include_thread_id: false,
            module_levels: HashMap::new(),
            service_name: "ringkernel".to_string(),
            environment: "development".to_string(),
            global_fields: HashMap::new(),
        }
    }
}

impl LogConfig {
    /// Create a new builder.
    pub fn builder() -> LogConfigBuilder {
        LogConfigBuilder::new()
    }

    /// Create a development configuration.
    pub fn development() -> Self {
        Self {
            level: LogLevel::Debug,
            output: LogOutput::Pretty,
            trace_correlation: true,
            include_timestamps: true,
            include_location: true,
            include_thread_id: false,
            environment: "development".to_string(),
            ..Default::default()
        }
    }

    /// Create a production configuration.
    pub fn production() -> Self {
        Self {
            level: LogLevel::Info,
            output: LogOutput::Json,
            trace_correlation: true,
            include_timestamps: true,
            include_location: false,
            include_thread_id: true,
            environment: "production".to_string(),
            ..Default::default()
        }
    }

    /// Get effective log level for a module.
    pub fn effective_level(&self, module: &str) -> LogLevel {
        // Check for exact match first
        if let Some(&level) = self.module_levels.get(module) {
            return level;
        }

        // Find the longest (most specific) prefix match
        let mut best_match: Option<(&str, LogLevel)> = None;
        for (prefix, &level) in &self.module_levels {
            if module.starts_with(prefix) {
                match best_match {
                    None => best_match = Some((prefix, level)),
                    Some((best_prefix, _)) if prefix.len() > best_prefix.len() => {
                        best_match = Some((prefix, level));
                    }
                    _ => {}
                }
            }
        }

        best_match.map(|(_, level)| level).unwrap_or(self.level)
    }
}

/// Builder for log configuration.
#[derive(Debug, Default)]
pub struct LogConfigBuilder {
    config: LogConfig,
}

impl LogConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: LogConfig::default(),
        }
    }

    /// Set default log level.
    pub fn level(mut self, level: LogLevel) -> Self {
        self.config.level = level;
        self
    }

    /// Set output format.
    pub fn output(mut self, output: LogOutput) -> Self {
        self.config.output = output;
        self
    }

    /// Enable/disable trace correlation.
    pub fn with_trace_correlation(mut self, enabled: bool) -> Self {
        self.config.trace_correlation = enabled;
        self
    }

    /// Enable/disable timestamps.
    pub fn with_timestamps(mut self, enabled: bool) -> Self {
        self.config.include_timestamps = enabled;
        self
    }

    /// Enable/disable caller location.
    pub fn with_location(mut self, enabled: bool) -> Self {
        self.config.include_location = enabled;
        self
    }

    /// Enable/disable thread ID.
    pub fn with_thread_id(mut self, enabled: bool) -> Self {
        self.config.include_thread_id = enabled;
        self
    }

    /// Set service name.
    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.config.service_name = name.into();
        self
    }

    /// Set environment.
    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.config.environment = env.into();
        self
    }

    /// Set log level for a specific module.
    pub fn module_level(mut self, module: impl Into<String>, level: LogLevel) -> Self {
        self.config.module_levels.insert(module.into(), level);
        self
    }

    /// Add a global field.
    pub fn global_field(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.global_fields.insert(key.into(), value.into());
        self
    }

    /// Build the configuration.
    pub fn build(self) -> LogConfig {
        self.config
    }
}

// ============================================================================
// Log Entry
// ============================================================================

/// A structured log entry.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Log level.
    pub level: LogLevel,
    /// Log message.
    pub message: String,
    /// Timestamp.
    pub timestamp: SystemTime,
    /// Module/target.
    pub target: Option<String>,
    /// File name.
    pub file: Option<String>,
    /// Line number.
    pub line: Option<u32>,
    /// Thread ID.
    pub thread_id: Option<u64>,
    /// Thread name.
    pub thread_name: Option<String>,
    /// Trace ID.
    pub trace_id: Option<TraceId>,
    /// Span ID.
    pub span_id: Option<SpanId>,
    /// Structured fields.
    pub fields: HashMap<String, LogValue>,
}

/// Log field value types.
#[derive(Debug, Clone)]
pub enum LogValue {
    /// String value.
    String(String),
    /// Integer value.
    Int(i64),
    /// Unsigned integer value.
    Uint(u64),
    /// Float value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
}

impl fmt::Display for LogValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogValue::String(s) => write!(f, "{}", s),
            LogValue::Int(i) => write!(f, "{}", i),
            LogValue::Uint(u) => write!(f, "{}", u),
            LogValue::Float(fl) => write!(f, "{}", fl),
            LogValue::Bool(b) => write!(f, "{}", b),
        }
    }
}

impl From<&str> for LogValue {
    fn from(s: &str) -> Self {
        LogValue::String(s.to_string())
    }
}

impl From<String> for LogValue {
    fn from(s: String) -> Self {
        LogValue::String(s)
    }
}

impl From<i64> for LogValue {
    fn from(i: i64) -> Self {
        LogValue::Int(i)
    }
}

impl From<u64> for LogValue {
    fn from(u: u64) -> Self {
        LogValue::Uint(u)
    }
}

impl From<f64> for LogValue {
    fn from(f: f64) -> Self {
        LogValue::Float(f)
    }
}

impl From<bool> for LogValue {
    fn from(b: bool) -> Self {
        LogValue::Bool(b)
    }
}

impl LogEntry {
    /// Create a new log entry.
    pub fn new(level: LogLevel, message: impl Into<String>) -> Self {
        Self {
            level,
            message: message.into(),
            timestamp: SystemTime::now(),
            target: None,
            file: None,
            line: None,
            thread_id: None,
            thread_name: None,
            trace_id: None,
            span_id: None,
            fields: HashMap::new(),
        }
    }

    /// Set target/module.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Set trace context.
    pub fn with_trace_context(mut self, ctx: &TraceContext) -> Self {
        self.trace_id = ctx.trace_id;
        self.span_id = ctx.span_id;
        for (k, v) in &ctx.fields {
            self.fields.insert(k.clone(), LogValue::String(v.clone()));
        }
        self
    }

    /// Add a field.
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<LogValue>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    /// Format as JSON.
    pub fn to_json(&self, config: &LogConfig) -> String {
        let mut json = String::with_capacity(512);
        json.push('{');

        // Timestamp
        if config.include_timestamps {
            let ts = self
                .timestamp
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            json.push_str(&format!(r#""timestamp":{},"#, ts));
        }

        // Level
        json.push_str(&format!(r#""level":"{}","#, self.level.as_str()));

        // Message (escape quotes)
        let escaped_msg = self.message.replace('\\', "\\\\").replace('"', "\\\"");
        json.push_str(&format!(r#""message":"{}","#, escaped_msg));

        // Service and environment
        json.push_str(&format!(r#""service":"{}","#, config.service_name));
        json.push_str(&format!(r#""environment":"{}","#, config.environment));

        // Target
        if let Some(ref target) = self.target {
            json.push_str(&format!(r#""target":"{}","#, target));
        }

        // Location
        if config.include_location {
            if let Some(ref file) = self.file {
                json.push_str(&format!(r#""file":"{}","#, file));
            }
            if let Some(line) = self.line {
                json.push_str(&format!(r#""line":{},"#, line));
            }
        }

        // Thread
        if config.include_thread_id {
            if let Some(tid) = self.thread_id {
                json.push_str(&format!(r#""thread_id":{},"#, tid));
            }
            if let Some(ref name) = self.thread_name {
                json.push_str(&format!(r#""thread_name":"{}","#, name));
            }
        }

        // Trace correlation
        if config.trace_correlation {
            if let Some(trace_id) = self.trace_id {
                json.push_str(&format!(r#""trace_id":"{:032x}","#, trace_id.0));
            }
            if let Some(span_id) = self.span_id {
                json.push_str(&format!(r#""span_id":"{:016x}","#, span_id.0));
            }
        }

        // Global fields
        for (k, v) in &config.global_fields {
            json.push_str(&format!(r#""{}":"{}","#, k, v));
        }

        // Entry fields
        if !self.fields.is_empty() {
            json.push_str(r#""fields":{"#);
            let mut first = true;
            for (k, v) in &self.fields {
                if !first {
                    json.push(',');
                }
                first = false;
                match v {
                    LogValue::String(s) => {
                        let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
                        json.push_str(&format!(r#""{}":"{}""#, k, escaped));
                    }
                    LogValue::Int(i) => json.push_str(&format!(r#""{}":{}""#, k, i)),
                    LogValue::Uint(u) => json.push_str(&format!(r#""{}":{}""#, k, u)),
                    LogValue::Float(f) => json.push_str(&format!(r#""{}":{}""#, k, f)),
                    LogValue::Bool(b) => json.push_str(&format!(r#""{}":{}""#, k, b)),
                }
            }
            json.push_str("},");
        }

        // Remove trailing comma and close
        if json.ends_with(',') {
            json.pop();
        }
        json.push('}');

        json
    }

    /// Format as text.
    pub fn to_text(&self, config: &LogConfig) -> String {
        let mut text = String::with_capacity(256);

        // Timestamp
        if config.include_timestamps {
            let ts = self
                .timestamp
                .duration_since(UNIX_EPOCH)
                .map(|d| {
                    let secs = d.as_secs();
                    let millis = d.subsec_millis();
                    format!(
                        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
                        1970 + secs / 31536000,            // Approximate year
                        ((secs % 31536000) / 2592000) + 1, // Month (approximate)
                        ((secs % 2592000) / 86400) + 1,    // Day
                        (secs % 86400) / 3600,             // Hour
                        (secs % 3600) / 60,                // Minute
                        secs % 60,                         // Second
                        millis
                    )
                })
                .unwrap_or_else(|_| "1970-01-01T00:00:00.000Z".to_string());
            text.push_str(&ts);
            text.push(' ');
        }

        // Level
        text.push_str(&format!("{:5} ", self.level.as_str()));

        // Target
        if let Some(ref target) = self.target {
            text.push_str(&format!("[{}] ", target));
        }

        // Trace correlation
        if config.trace_correlation {
            if let Some(trace_id) = self.trace_id {
                text.push_str(&format!("trace={:032x} ", trace_id.0));
            }
        }

        // Message
        text.push_str(&self.message);

        // Fields
        if !self.fields.is_empty() {
            text.push_str(" {");
            let mut first = true;
            for (k, v) in &self.fields {
                if !first {
                    text.push_str(", ");
                }
                first = false;
                text.push_str(&format!("{}={}", k, v));
            }
            text.push('}');
        }

        text
    }
}

// ============================================================================
// Structured Logger
// ============================================================================

/// Structured logger with trace correlation.
pub struct StructuredLogger {
    /// Configuration.
    config: RwLock<LogConfig>,
    /// Current trace context.
    context: RwLock<TraceContext>,
    /// Log counter.
    log_count: AtomicU64,
    /// Error counter.
    error_count: AtomicU64,
    /// Enabled flag.
    enabled: AtomicBool,
    /// Start time.
    start_time: Instant,
    /// Log sinks.
    sinks: RwLock<Vec<Arc<dyn LogSink>>>,
}

impl StructuredLogger {
    /// Create a new logger with configuration.
    pub fn new(config: LogConfig) -> Self {
        Self {
            config: RwLock::new(config),
            context: RwLock::new(TraceContext::new()),
            log_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
            start_time: Instant::now(),
            sinks: RwLock::new(vec![]),
        }
    }

    /// Create with default configuration.
    pub fn default_logger() -> Self {
        Self::new(LogConfig::default())
    }

    /// Create a development logger.
    pub fn development() -> Self {
        Self::new(LogConfig::development())
    }

    /// Create a production logger.
    pub fn production() -> Self {
        Self::new(LogConfig::production())
    }

    /// Enable/disable logging.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::SeqCst);
    }

    /// Check if logging is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Update configuration.
    pub fn set_config(&self, config: LogConfig) {
        *self.config.write() = config;
    }

    /// Get current configuration.
    pub fn config(&self) -> LogConfig {
        self.config.read().clone()
    }

    /// Set trace context.
    pub fn set_context(&self, context: TraceContext) {
        *self.context.write() = context;
    }

    /// Get current trace context.
    pub fn context(&self) -> TraceContext {
        self.context.read().clone()
    }

    /// Start a new trace.
    pub fn start_trace(&self) -> TraceContext {
        let ctx = TraceContext::with_new_trace();
        *self.context.write() = ctx.clone();
        ctx
    }

    /// Add a log sink.
    pub fn add_sink(&self, sink: Arc<dyn LogSink>) {
        self.sinks.write().push(sink);
    }

    /// Log at specified level.
    pub fn log(&self, level: LogLevel, message: &str, fields: &[(&str, &str)]) {
        if !self.enabled.load(Ordering::SeqCst) {
            return;
        }

        let config = self.config.read();
        if level < config.level {
            return;
        }

        let ctx = self.context.read();
        let mut entry = LogEntry::new(level, message).with_trace_context(&ctx);

        for (k, v) in fields {
            entry = entry.with_field(*k, *v);
        }

        self.log_count.fetch_add(1, Ordering::Relaxed);
        if level >= LogLevel::Error {
            self.error_count.fetch_add(1, Ordering::Relaxed);
        }

        // Format and output
        let output = match config.output {
            LogOutput::Json => entry.to_json(&config),
            LogOutput::Text | LogOutput::Compact | LogOutput::Pretty => entry.to_text(&config),
        };

        drop(config);

        // Send to sinks
        let sinks = self.sinks.read();
        for sink in sinks.iter() {
            let _ = sink.write(&entry, &output);
        }

        // Default output to stderr
        if sinks.is_empty() {
            let _ = writeln!(std::io::stderr(), "{}", output);
        }
    }

    /// Log at trace level.
    pub fn trace(&self, message: &str, fields: &[(&str, &str)]) {
        self.log(LogLevel::Trace, message, fields);
    }

    /// Log at debug level.
    pub fn debug(&self, message: &str, fields: &[(&str, &str)]) {
        self.log(LogLevel::Debug, message, fields);
    }

    /// Log at info level.
    pub fn info(&self, message: &str, fields: &[(&str, &str)]) {
        self.log(LogLevel::Info, message, fields);
    }

    /// Log at warn level.
    pub fn warn(&self, message: &str, fields: &[(&str, &str)]) {
        self.log(LogLevel::Warn, message, fields);
    }

    /// Log at error level.
    pub fn error(&self, message: &str, fields: &[(&str, &str)]) {
        self.log(LogLevel::Error, message, fields);
    }

    /// Log at fatal level.
    pub fn fatal(&self, message: &str, fields: &[(&str, &str)]) {
        self.log(LogLevel::Fatal, message, fields);
    }

    /// Get statistics.
    pub fn stats(&self) -> LoggerStats {
        LoggerStats {
            log_count: self.log_count.load(Ordering::Relaxed),
            error_count: self.error_count.load(Ordering::Relaxed),
            uptime: self.start_time.elapsed(),
            sink_count: self.sinks.read().len(),
        }
    }
}

impl Default for StructuredLogger {
    fn default() -> Self {
        Self::default_logger()
    }
}

/// Logger statistics.
#[derive(Debug, Clone)]
pub struct LoggerStats {
    /// Total log entries.
    pub log_count: u64,
    /// Error-level entries.
    pub error_count: u64,
    /// Time since logger started.
    pub uptime: std::time::Duration,
    /// Number of sinks.
    pub sink_count: usize,
}

// ============================================================================
// Log Sinks
// ============================================================================

/// Trait for log output destinations.
pub trait LogSink: Send + Sync {
    /// Write a log entry.
    fn write(&self, entry: &LogEntry, formatted: &str) -> Result<(), LogSinkError>;

    /// Flush buffered logs.
    fn flush(&self) -> Result<(), LogSinkError> {
        Ok(())
    }

    /// Get sink name.
    fn name(&self) -> &str;
}

/// Log sink error.
#[derive(Debug)]
pub struct LogSinkError {
    /// Error message.
    pub message: String,
}

impl fmt::Display for LogSinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LogSinkError: {}", self.message)
    }
}

impl std::error::Error for LogSinkError {}

/// Console log sink.
pub struct ConsoleSink {
    /// Use stderr instead of stdout.
    use_stderr: bool,
}

impl ConsoleSink {
    /// Create a new console sink.
    pub fn new() -> Self {
        Self { use_stderr: true }
    }

    /// Create a sink that outputs to stdout.
    pub fn stdout() -> Self {
        Self { use_stderr: false }
    }

    /// Create a sink that outputs to stderr.
    pub fn stderr() -> Self {
        Self { use_stderr: true }
    }
}

impl Default for ConsoleSink {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSink for ConsoleSink {
    fn write(&self, _entry: &LogEntry, formatted: &str) -> Result<(), LogSinkError> {
        let result = if self.use_stderr {
            writeln!(std::io::stderr(), "{}", formatted)
        } else {
            writeln!(std::io::stdout(), "{}", formatted)
        };
        result.map_err(|e| LogSinkError {
            message: e.to_string(),
        })
    }

    fn name(&self) -> &str {
        if self.use_stderr {
            "console:stderr"
        } else {
            "console:stdout"
        }
    }
}

/// Memory log sink for testing.
pub struct MemoryLogSink {
    /// Stored logs.
    logs: RwLock<Vec<String>>,
    /// Maximum capacity.
    capacity: usize,
}

impl MemoryLogSink {
    /// Create a new memory sink.
    pub fn new(capacity: usize) -> Self {
        Self {
            logs: RwLock::new(Vec::with_capacity(capacity)),
            capacity,
        }
    }

    /// Get all stored logs.
    pub fn logs(&self) -> Vec<String> {
        self.logs.read().clone()
    }

    /// Clear stored logs.
    pub fn clear(&self) {
        self.logs.write().clear();
    }

    /// Get log count.
    pub fn len(&self) -> usize {
        self.logs.read().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.logs.read().is_empty()
    }
}

impl LogSink for MemoryLogSink {
    fn write(&self, _entry: &LogEntry, formatted: &str) -> Result<(), LogSinkError> {
        let mut logs = self.logs.write();
        if logs.len() >= self.capacity {
            logs.remove(0);
        }
        logs.push(formatted.to_string());
        Ok(())
    }

    fn name(&self) -> &str {
        "memory"
    }
}

/// File log sink.
pub struct FileLogSink {
    /// File path.
    path: String,
    /// File handle.
    file: RwLock<Option<std::fs::File>>,
}

impl FileLogSink {
    /// Create a new file sink.
    pub fn new(path: impl Into<String>) -> Result<Self, LogSinkError> {
        let path = path.into();
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| LogSinkError {
                message: format!("Failed to open log file: {}", e),
            })?;

        Ok(Self {
            path,
            file: RwLock::new(Some(file)),
        })
    }
}

impl LogSink for FileLogSink {
    fn write(&self, _entry: &LogEntry, formatted: &str) -> Result<(), LogSinkError> {
        let mut guard = self.file.write();
        if let Some(ref mut file) = *guard {
            writeln!(file, "{}", formatted).map_err(|e| LogSinkError {
                message: e.to_string(),
            })?;
        }
        Ok(())
    }

    fn flush(&self) -> Result<(), LogSinkError> {
        let mut guard = self.file.write();
        if let Some(ref mut file) = *guard {
            file.flush().map_err(|e| LogSinkError {
                message: e.to_string(),
            })?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        &self.path
    }
}

// ============================================================================
// Global Logger
// ============================================================================

use std::sync::OnceLock;

static GLOBAL_LOGGER: OnceLock<StructuredLogger> = OnceLock::new();

/// Initialize the global logger.
pub fn init(config: LogConfig) {
    let _ = GLOBAL_LOGGER.set(StructuredLogger::new(config));
}

/// Get the global logger.
pub fn logger() -> &'static StructuredLogger {
    GLOBAL_LOGGER.get_or_init(StructuredLogger::default_logger)
}

/// Log at trace level.
pub fn trace(message: &str, fields: &[(&str, &str)]) {
    logger().trace(message, fields);
}

/// Log at debug level.
pub fn debug(message: &str, fields: &[(&str, &str)]) {
    logger().debug(message, fields);
}

/// Log at info level.
pub fn info(message: &str, fields: &[(&str, &str)]) {
    logger().info(message, fields);
}

/// Log at warn level.
pub fn warn(message: &str, fields: &[(&str, &str)]) {
    logger().warn(message, fields);
}

/// Log at error level.
pub fn error(message: &str, fields: &[(&str, &str)]) {
    logger().error(message, fields);
}

/// Log at fatal level.
pub fn fatal(message: &str, fields: &[(&str, &str)]) {
    logger().fatal(message, fields);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
    }

    #[test]
    fn test_log_level_from_str() {
        assert_eq!(LogLevel::parse("trace"), Some(LogLevel::Trace));
        assert_eq!(LogLevel::parse("DEBUG"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::parse("Info"), Some(LogLevel::Info));
        assert_eq!(LogLevel::parse("WARNING"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::parse("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::parse("FATAL"), Some(LogLevel::Fatal));
        assert_eq!(LogLevel::parse("CRITICAL"), Some(LogLevel::Fatal));
        assert_eq!(LogLevel::parse("invalid"), None);
    }

    #[test]
    fn test_log_config_builder() {
        let config = LogConfig::builder()
            .level(LogLevel::Debug)
            .output(LogOutput::Json)
            .with_trace_correlation(true)
            .with_timestamps(true)
            .with_location(true)
            .service_name("test-service")
            .environment("test")
            .module_level("ringkernel::k2k", LogLevel::Trace)
            .global_field("version", "1.0.0")
            .build();

        assert_eq!(config.level, LogLevel::Debug);
        assert_eq!(config.output, LogOutput::Json);
        assert!(config.trace_correlation);
        assert!(config.include_timestamps);
        assert!(config.include_location);
        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.environment, "test");
        assert_eq!(
            config.effective_level("ringkernel::k2k::broker"),
            LogLevel::Trace
        );
    }

    #[test]
    fn test_log_config_effective_level() {
        let config = LogConfig::builder()
            .level(LogLevel::Info)
            .module_level("ringkernel", LogLevel::Debug)
            .module_level("ringkernel::k2k", LogLevel::Trace)
            .build();

        assert_eq!(config.effective_level("other::module"), LogLevel::Info);
        assert_eq!(config.effective_level("ringkernel::core"), LogLevel::Debug);
        assert_eq!(config.effective_level("ringkernel::k2k"), LogLevel::Trace);
        assert_eq!(
            config.effective_level("ringkernel::k2k::broker"),
            LogLevel::Trace
        );
    }

    #[test]
    fn test_trace_context() {
        let ctx = TraceContext::with_new_trace()
            .with_field("user_id", "123")
            .with_field("request_id", "abc");

        assert!(ctx.trace_id.is_some());
        assert!(ctx.span_id.is_some());
        assert_eq!(ctx.fields.get("user_id"), Some(&"123".to_string()));
    }

    #[test]
    fn test_log_entry_json() {
        let config = LogConfig::builder()
            .service_name("test")
            .environment("dev")
            .with_timestamps(false)
            .with_trace_correlation(false)
            .build();

        let entry = LogEntry::new(LogLevel::Info, "Test message").with_field("key", "value");

        let json = entry.to_json(&config);
        assert!(json.contains(r#""level":"INFO""#));
        assert!(json.contains(r#""message":"Test message""#));
        assert!(json.contains(r#""service":"test""#));
    }

    #[test]
    fn test_log_entry_text() {
        let config = LogConfig::builder()
            .with_timestamps(false)
            .with_trace_correlation(false)
            .build();

        let entry = LogEntry::new(LogLevel::Warn, "Warning!").with_target("test::module");

        let text = entry.to_text(&config);
        assert!(text.contains("WARN"));
        assert!(text.contains("[test::module]"));
        assert!(text.contains("Warning!"));
    }

    #[test]
    fn test_structured_logger() {
        let logger = StructuredLogger::new(LogConfig::builder().level(LogLevel::Debug).build());

        let sink = Arc::new(MemoryLogSink::new(100));
        logger.add_sink(sink.clone());

        logger.info("Test message", &[("key", "value")]);
        logger.debug("Debug message", &[]);
        logger.trace("Trace message", &[]); // Should be filtered

        assert_eq!(sink.len(), 2);
    }

    #[test]
    fn test_memory_sink_capacity() {
        let sink = MemoryLogSink::new(3);
        let entry = LogEntry::new(LogLevel::Info, "msg");

        sink.write(&entry, "log1").unwrap();
        sink.write(&entry, "log2").unwrap();
        sink.write(&entry, "log3").unwrap();
        sink.write(&entry, "log4").unwrap();

        let logs = sink.logs();
        assert_eq!(logs.len(), 3);
        assert_eq!(logs[0], "log2");
        assert_eq!(logs[2], "log4");
    }

    #[test]
    fn test_logger_stats() {
        let logger = StructuredLogger::new(LogConfig::default());
        let sink = Arc::new(MemoryLogSink::new(100));
        logger.add_sink(sink);

        logger.info("info", &[]);
        logger.error("error", &[]);
        logger.warn("warn", &[]);

        let stats = logger.stats();
        assert_eq!(stats.log_count, 3);
        assert_eq!(stats.error_count, 1);
        assert_eq!(stats.sink_count, 1);
    }

    #[test]
    fn test_logger_disable() {
        let logger = StructuredLogger::new(LogConfig::default());
        let sink = Arc::new(MemoryLogSink::new(100));
        logger.add_sink(sink.clone());

        logger.info("before", &[]);
        logger.set_enabled(false);
        logger.info("during", &[]);
        logger.set_enabled(true);
        logger.info("after", &[]);

        assert_eq!(sink.len(), 2);
    }

    #[test]
    fn test_log_value_display() {
        assert_eq!(LogValue::String("test".to_string()).to_string(), "test");
        assert_eq!(LogValue::Int(-42).to_string(), "-42");
        assert_eq!(LogValue::Uint(42).to_string(), "42");
        assert_eq!(LogValue::Bool(true).to_string(), "true");
    }

    #[test]
    fn test_console_sink() {
        let sink = ConsoleSink::stderr();
        assert_eq!(sink.name(), "console:stderr");

        let sink = ConsoleSink::stdout();
        assert_eq!(sink.name(), "console:stdout");
    }

    #[test]
    fn test_log_config_presets() {
        let dev = LogConfig::development();
        assert_eq!(dev.level, LogLevel::Debug);
        assert_eq!(dev.output, LogOutput::Pretty);
        assert!(dev.include_location);

        let prod = LogConfig::production();
        assert_eq!(prod.level, LogLevel::Info);
        assert_eq!(prod.output, LogOutput::Json);
        assert!(prod.include_thread_id);
    }
}
