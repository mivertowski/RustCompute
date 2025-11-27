//! Telemetry and metrics collection for kernel monitoring.
//!
//! This module provides structures for collecting performance metrics
//! from GPU kernels, including throughput, latency, and error tracking.

/// Telemetry buffer (64 bytes, cache-line aligned).
///
/// This structure is updated by the GPU kernel and read by the host
/// for monitoring and debugging purposes.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct TelemetryBuffer {
    /// Total messages processed successfully.
    pub messages_processed: u64,
    /// Total messages dropped (queue full, timeout, etc.).
    pub messages_dropped: u64,
    /// Sum of processing latencies in microseconds.
    pub total_latency_us: u64,
    /// Minimum processing latency in microseconds.
    pub min_latency_us: u64,
    /// Maximum processing latency in microseconds.
    pub max_latency_us: u64,
    /// Current input queue depth.
    pub input_queue_depth: u32,
    /// Current output queue depth.
    pub output_queue_depth: u32,
    /// Last error code (0 = no error).
    pub last_error: u32,
    /// Reserved for alignment (pad to 64 bytes).
    pub _reserved: [u32; 3],
}

// Verify size at compile time
const _: () = assert!(std::mem::size_of::<TelemetryBuffer>() == 64);

impl TelemetryBuffer {
    /// Create a new telemetry buffer.
    pub const fn new() -> Self {
        Self {
            messages_processed: 0,
            messages_dropped: 0,
            total_latency_us: 0,
            min_latency_us: u64::MAX,
            max_latency_us: 0,
            input_queue_depth: 0,
            output_queue_depth: 0,
            last_error: 0,
            _reserved: [0; 3],
        }
    }

    /// Calculate average latency in microseconds.
    pub fn avg_latency_us(&self) -> f64 {
        if self.messages_processed == 0 {
            0.0
        } else {
            self.total_latency_us as f64 / self.messages_processed as f64
        }
    }

    /// Get throughput (messages per second) given elapsed time.
    pub fn throughput(&self, elapsed_secs: f64) -> f64 {
        if elapsed_secs <= 0.0 {
            0.0
        } else {
            self.messages_processed as f64 / elapsed_secs
        }
    }

    /// Get drop rate (0.0 to 1.0).
    pub fn drop_rate(&self) -> f64 {
        let total = self.messages_processed + self.messages_dropped;
        if total == 0 {
            0.0
        } else {
            self.messages_dropped as f64 / total as f64
        }
    }

    /// Reset all counters to initial state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Merge another telemetry buffer into this one.
    pub fn merge(&mut self, other: &TelemetryBuffer) {
        self.messages_processed += other.messages_processed;
        self.messages_dropped += other.messages_dropped;
        self.total_latency_us += other.total_latency_us;
        self.min_latency_us = self.min_latency_us.min(other.min_latency_us);
        self.max_latency_us = self.max_latency_us.max(other.max_latency_us);
        // Queue depths are point-in-time, take latest
        self.input_queue_depth = other.input_queue_depth;
        self.output_queue_depth = other.output_queue_depth;
        // Last error takes the most recent non-zero
        if other.last_error != 0 {
            self.last_error = other.last_error;
        }
    }
}

/// Extended metrics for detailed monitoring.
#[derive(Debug, Clone)]
pub struct KernelMetrics {
    /// Basic telemetry from GPU.
    pub telemetry: TelemetryBuffer,

    /// Kernel identifier.
    pub kernel_id: String,

    /// Timestamp when metrics were collected.
    pub collected_at: std::time::Instant,

    /// Time since kernel was launched.
    pub uptime: std::time::Duration,

    /// Number of kernel invocations (for event-driven mode).
    pub invocations: u64,

    /// Total bytes transferred to device.
    pub bytes_to_device: u64,

    /// Total bytes transferred from device.
    pub bytes_from_device: u64,

    /// GPU memory usage in bytes.
    pub gpu_memory_used: u64,

    /// Host memory usage in bytes.
    pub host_memory_used: u64,
}

impl Default for KernelMetrics {
    fn default() -> Self {
        Self {
            telemetry: TelemetryBuffer::default(),
            kernel_id: String::new(),
            collected_at: std::time::Instant::now(),
            uptime: std::time::Duration::ZERO,
            invocations: 0,
            bytes_to_device: 0,
            bytes_from_device: 0,
            gpu_memory_used: 0,
            host_memory_used: 0,
        }
    }
}

impl KernelMetrics {
    /// Create new metrics for a kernel.
    pub fn new(kernel_id: impl Into<String>) -> Self {
        Self {
            kernel_id: kernel_id.into(),
            ..Default::default()
        }
    }

    /// Calculate transfer bandwidth (bytes/sec).
    pub fn transfer_bandwidth(&self) -> f64 {
        let total_bytes = self.bytes_to_device + self.bytes_from_device;
        let secs = self.uptime.as_secs_f64();
        if secs > 0.0 {
            total_bytes as f64 / secs
        } else {
            0.0
        }
    }

    /// Get summary as a formatted string.
    pub fn summary(&self) -> String {
        format!(
            "Kernel {} - Processed: {}, Dropped: {}, Avg Latency: {:.2}µs, Throughput: {:.2}/s",
            self.kernel_id,
            self.telemetry.messages_processed,
            self.telemetry.messages_dropped,
            self.telemetry.avg_latency_us(),
            self.telemetry.throughput(self.uptime.as_secs_f64())
        )
    }
}

/// Histogram for latency distribution.
#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    /// Bucket boundaries in microseconds.
    pub buckets: Vec<u64>,
    /// Counts for each bucket.
    pub counts: Vec<u64>,
    /// Count of values above last bucket.
    pub overflow: u64,
}

impl LatencyHistogram {
    /// Create a new histogram with default buckets.
    pub fn new() -> Self {
        // Default buckets: 1µs, 10µs, 100µs, 1ms, 10ms, 100ms, 1s
        Self::with_buckets(vec![1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
    }

    /// Create with custom bucket boundaries.
    pub fn with_buckets(buckets: Vec<u64>) -> Self {
        let counts = vec![0; buckets.len()];
        Self {
            buckets,
            counts,
            overflow: 0,
        }
    }

    /// Record a latency value.
    pub fn record(&mut self, value_us: u64) {
        for (i, &boundary) in self.buckets.iter().enumerate() {
            if value_us <= boundary {
                self.counts[i] += 1;
                return;
            }
        }
        self.overflow += 1;
    }

    /// Get total count.
    pub fn total(&self) -> u64 {
        self.counts.iter().sum::<u64>() + self.overflow
    }

    /// Get percentile value.
    pub fn percentile(&self, p: f64) -> u64 {
        let total = self.total();
        if total == 0 {
            return 0;
        }

        let target = (total as f64 * p / 100.0).ceil() as u64;
        let mut cumulative = 0u64;

        for (i, &count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return self.buckets[i];
            }
        }

        // Return last bucket boundary + 1 for overflow
        self.buckets.last().map(|b| b + 1).unwrap_or(0)
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_buffer_size() {
        assert_eq!(std::mem::size_of::<TelemetryBuffer>(), 64);
    }

    #[test]
    fn test_avg_latency() {
        let mut tb = TelemetryBuffer::new();
        assert_eq!(tb.avg_latency_us(), 0.0);

        tb.messages_processed = 10;
        tb.total_latency_us = 1000;
        assert_eq!(tb.avg_latency_us(), 100.0);
    }

    #[test]
    fn test_throughput() {
        let mut tb = TelemetryBuffer::new();
        tb.messages_processed = 1000;

        assert_eq!(tb.throughput(1.0), 1000.0);
        assert_eq!(tb.throughput(2.0), 500.0);
        assert_eq!(tb.throughput(0.0), 0.0);
    }

    #[test]
    fn test_drop_rate() {
        let mut tb = TelemetryBuffer::new();
        tb.messages_processed = 90;
        tb.messages_dropped = 10;

        assert!((tb.drop_rate() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_merge() {
        let mut tb1 = TelemetryBuffer::new();
        tb1.messages_processed = 100;
        tb1.min_latency_us = 10;
        tb1.max_latency_us = 100;

        let mut tb2 = TelemetryBuffer::new();
        tb2.messages_processed = 50;
        tb2.min_latency_us = 5;
        tb2.max_latency_us = 200;

        tb1.merge(&tb2);

        assert_eq!(tb1.messages_processed, 150);
        assert_eq!(tb1.min_latency_us, 5);
        assert_eq!(tb1.max_latency_us, 200);
    }

    #[test]
    fn test_histogram_percentile() {
        let mut hist = LatencyHistogram::with_buckets(vec![10, 50, 100, 500]);

        // Record some values
        for _ in 0..80 {
            hist.record(5); // <= 10
        }
        for _ in 0..15 {
            hist.record(30); // <= 50
        }
        for _ in 0..5 {
            hist.record(200); // <= 500
        }

        assert_eq!(hist.percentile(50.0), 10); // p50 = 10µs bucket
        assert_eq!(hist.percentile(90.0), 50); // p90 = 50µs bucket
        assert_eq!(hist.percentile(99.0), 500); // p99 = 500µs bucket
    }
}
