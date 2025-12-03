//! Enhanced tracing support for RingKernel.
//!
//! This module provides tracing integration with structured logging,
//! span management, and GPU-specific instrumentation.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ecosystem::tracing_ext::{init_tracing, TracingConfig};
//!
//! init_tracing(TracingConfig::default())?;
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, span, warn, Level, Span};

/// Configuration for tracing.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Log level.
    pub level: Level,
    /// Enable span timing.
    pub enable_timing: bool,
    /// Enable GPU-specific spans.
    pub enable_gpu_spans: bool,
    /// Sample rate for high-frequency events (0.0 to 1.0).
    pub sample_rate: f64,
    /// Enable JSON output format.
    pub json_output: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            level: Level::INFO,
            enable_timing: true,
            enable_gpu_spans: true,
            sample_rate: 1.0,
            json_output: false,
        }
    }
}

/// Span manager for GPU operations.
#[derive(Default)]
pub struct SpanManager {
    /// Active spans.
    active_spans: RwLock<HashMap<String, SpanInfo>>,
    /// Completed span statistics.
    stats: SpanStats,
}

/// Information about an active span.
struct SpanInfo {
    /// Span handle.
    #[allow(dead_code)]
    span: Span,
    /// Start time.
    start_time: Instant,
    /// Kernel ID.
    kernel_id: String,
    /// Operation type.
    operation: String,
}

/// Statistics for completed spans.
#[derive(Default)]
pub struct SpanStats {
    /// Total spans created.
    pub total_spans: AtomicU64,
    /// Total span duration in microseconds.
    pub total_duration_us: AtomicU64,
    /// Spans by operation type.
    operation_counts: RwLock<HashMap<String, u64>>,
}

impl SpanStats {
    /// Record a completed span.
    pub fn record(&self, operation: &str, duration: Duration) {
        self.total_spans.fetch_add(1, Ordering::Relaxed);
        self.total_duration_us
            .fetch_add(duration.as_micros() as u64, Ordering::Relaxed);

        let mut counts = self.operation_counts.write();
        *counts.entry(operation.to_string()).or_insert(0) += 1;
    }

    /// Get average span duration in microseconds.
    pub fn avg_duration_us(&self) -> f64 {
        let total = self.total_spans.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            self.total_duration_us.load(Ordering::Relaxed) as f64 / total as f64
        }
    }

    /// Get operation counts.
    pub fn operation_counts(&self) -> HashMap<String, u64> {
        self.operation_counts.read().clone()
    }
}

impl SpanManager {
    /// Create a new span manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a new span for a GPU operation.
    pub fn start_span(&self, id: &str, kernel_id: &str, operation: &str) -> SpanGuard {
        let span = span!(
            Level::INFO,
            "ringkernel.operation",
            kernel_id = kernel_id,
            operation = operation,
            span_id = id
        );

        let info = SpanInfo {
            span: span.clone(),
            start_time: Instant::now(),
            kernel_id: kernel_id.to_string(),
            operation: operation.to_string(),
        };

        self.active_spans.write().insert(id.to_string(), info);

        SpanGuard {
            id: id.to_string(),
            operation: operation.to_string(),
            manager: self,
            span,
        }
    }

    /// Complete a span.
    fn complete_span(&self, id: &str, operation: &str) {
        if let Some(info) = self.active_spans.write().remove(id) {
            let duration = info.start_time.elapsed();
            self.stats.record(operation, duration);
        }
    }

    /// Get span statistics.
    pub fn stats(&self) -> &SpanStats {
        &self.stats
    }

    /// Get number of active spans.
    pub fn active_count(&self) -> usize {
        self.active_spans.read().len()
    }
}

/// Guard that automatically completes a span when dropped.
pub struct SpanGuard<'a> {
    id: String,
    operation: String,
    manager: &'a SpanManager,
    span: Span,
}

impl<'a> SpanGuard<'a> {
    /// Get the span.
    pub fn span(&self) -> &Span {
        &self.span
    }

    /// Record an event in the span.
    pub fn event(&self, message: &str) {
        let _enter = self.span.enter();
        info!(message);
    }

    /// Record a warning in the span.
    pub fn warning(&self, message: &str) {
        let _enter = self.span.enter();
        warn!(message);
    }

    /// Record an error in the span.
    pub fn error(&self, message: &str) {
        let _enter = self.span.enter();
        error!(message);
    }
}

impl<'a> Drop for SpanGuard<'a> {
    fn drop(&mut self) {
        self.manager.complete_span(&self.id, &self.operation);
    }
}

/// Log GPU kernel events.
pub fn log_kernel_event(kernel_id: &str, event: &str, details: Option<&str>) {
    if let Some(details) = details {
        info!(
            kernel_id = kernel_id,
            event = event,
            details = details,
            "GPU kernel event"
        );
    } else {
        info!(kernel_id = kernel_id, event = event, "GPU kernel event");
    }
}

/// Log message send operation.
pub fn log_message_send(kernel_id: &str, message_id: &str, priority: u8, size: usize) {
    debug!(
        kernel_id = kernel_id,
        message_id = message_id,
        priority = priority,
        size = size,
        "Message sent to kernel"
    );
}

/// Log message receive operation.
pub fn log_message_receive(kernel_id: &str, message_id: &str, latency_us: u64) {
    debug!(
        kernel_id = kernel_id,
        message_id = message_id,
        latency_us = latency_us,
        "Message received from kernel"
    );
}

/// Log error.
pub fn log_error(kernel_id: &str, error: &str, context: Option<&str>) {
    if let Some(context) = context {
        error!(
            kernel_id = kernel_id,
            error = error,
            context = context,
            "Kernel error"
        );
    } else {
        error!(kernel_id = kernel_id, error = error, "Kernel error");
    }
}

/// Instrumentation helpers for GPU operations.
pub mod instrument {
    use super::*;

    /// Instrument an async function with timing.
    pub async fn timed<F, T>(name: &str, f: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();
        let result = f.await;
        let duration = start.elapsed();

        debug!(
            operation = name,
            duration_us = duration.as_micros() as u64,
            "Operation completed"
        );

        result
    }

    /// Create a span for kernel launch.
    pub fn kernel_launch_span(kernel_id: &str) -> Span {
        span!(Level::INFO, "ringkernel.launch", kernel_id = kernel_id)
    }

    /// Create a span for message processing.
    pub fn message_span(kernel_id: &str, message_id: &str) -> Span {
        span!(
            Level::DEBUG,
            "ringkernel.message",
            kernel_id = kernel_id,
            message_id = message_id
        )
    }

    /// Create a span for GPU memory operations.
    pub fn memory_span(operation: &str, size: usize) -> Span {
        span!(
            Level::TRACE,
            "ringkernel.memory",
            operation = operation,
            size = size
        )
    }
}

/// Sampling helper for high-frequency events.
pub struct Sampler {
    rate: f64,
    counter: AtomicU64,
}

impl Sampler {
    /// Create a new sampler with the given rate.
    pub fn new(rate: f64) -> Self {
        Self {
            rate: rate.clamp(0.0, 1.0),
            counter: AtomicU64::new(0),
        }
    }

    /// Check if this event should be sampled.
    pub fn should_sample(&self) -> bool {
        if self.rate >= 1.0 {
            return true;
        }
        if self.rate <= 0.0 {
            return false;
        }

        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let threshold = (1.0 / self.rate) as u64;
        count % threshold == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_config_default() {
        let config = TracingConfig::default();
        assert_eq!(config.level, Level::INFO);
        assert!(config.enable_timing);
        assert!(config.enable_gpu_spans);
        assert_eq!(config.sample_rate, 1.0);
    }

    #[test]
    fn test_span_stats() {
        let stats = SpanStats::default();
        stats.record("send", Duration::from_micros(100));
        stats.record("receive", Duration::from_micros(200));

        assert_eq!(stats.total_spans.load(Ordering::Relaxed), 2);
        assert_eq!(stats.avg_duration_us(), 150.0);
    }

    #[test]
    fn test_span_manager() {
        let manager = SpanManager::new();
        assert_eq!(manager.active_count(), 0);

        {
            let _guard = manager.start_span("span1", "kernel1", "send");
            assert_eq!(manager.active_count(), 1);
        }

        assert_eq!(manager.active_count(), 0);
        assert_eq!(manager.stats().total_spans.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_sampler() {
        let sampler = Sampler::new(1.0);
        assert!(sampler.should_sample());

        let sampler_half = Sampler::new(0.5);
        let mut sampled = 0;
        for _ in 0..10 {
            if sampler_half.should_sample() {
                sampled += 1;
            }
        }
        assert!(sampled > 0 && sampled <= 10);
    }
}
