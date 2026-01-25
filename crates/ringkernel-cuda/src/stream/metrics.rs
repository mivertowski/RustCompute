//! Stream overlap metrics for compute/transfer analysis.

/// Metrics for analyzing compute/transfer overlap efficiency.
#[derive(Debug, Clone, Default)]
pub struct OverlapMetrics {
    /// Total compute time (nanoseconds).
    pub compute_ns: u64,
    /// Total transfer time (nanoseconds).
    pub transfer_ns: u64,
    /// Overlapped time (compute running while transfer active).
    pub overlap_ns: u64,
    /// Number of overlapped operations.
    pub overlap_count: u64,
}

impl OverlapMetrics {
    /// Creates new metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records compute time.
    pub fn record_compute(&mut self, ns: u64) {
        self.compute_ns += ns;
    }

    /// Records transfer time.
    pub fn record_transfer(&mut self, ns: u64) {
        self.transfer_ns += ns;
    }

    /// Records overlap time.
    pub fn record_overlap(&mut self, ns: u64) {
        self.overlap_ns += ns;
        self.overlap_count += 1;
    }

    /// Calculates the overlap efficiency (0.0 = no overlap, 1.0 = perfect overlap).
    #[must_use]
    pub fn overlap_efficiency(&self) -> f64 {
        let max_overlap = self.compute_ns.min(self.transfer_ns);
        if max_overlap == 0 {
            return 0.0;
        }
        self.overlap_ns as f64 / max_overlap as f64
    }

    /// Calculates the effective throughput improvement from overlapping.
    /// Returns the ratio of (sequential time) / (overlapped time).
    #[must_use]
    pub fn throughput_improvement(&self) -> f64 {
        let sequential = self.compute_ns + self.transfer_ns;
        let overlapped = sequential.saturating_sub(self.overlap_ns);
        if overlapped == 0 {
            return 1.0;
        }
        sequential as f64 / overlapped as f64
    }

    /// Returns compute time in milliseconds.
    #[must_use]
    pub fn compute_ms(&self) -> f64 {
        self.compute_ns as f64 / 1_000_000.0
    }

    /// Returns transfer time in milliseconds.
    #[must_use]
    pub fn transfer_ms(&self) -> f64 {
        self.transfer_ns as f64 / 1_000_000.0
    }

    /// Returns overlap time in milliseconds.
    #[must_use]
    pub fn overlap_ms(&self) -> f64 {
        self.overlap_ns as f64 / 1_000_000.0
    }

    /// Merges another metrics instance into this one.
    pub fn merge(&mut self, other: &OverlapMetrics) {
        self.compute_ns += other.compute_ns;
        self.transfer_ns += other.transfer_ns;
        self.overlap_ns += other.overlap_ns;
        self.overlap_count += other.overlap_count;
    }
}

impl std::fmt::Display for OverlapMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Overlap Metrics: compute={:.2}ms, transfer={:.2}ms, overlap={:.2}ms ({:.1}% efficiency, {:.2}x throughput)",
            self.compute_ms(),
            self.transfer_ms(),
            self.overlap_ms(),
            self.overlap_efficiency() * 100.0,
            self.throughput_improvement()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlap_metrics_default() {
        let metrics = OverlapMetrics::default();
        assert_eq!(metrics.compute_ns, 0);
        assert_eq!(metrics.transfer_ns, 0);
        assert_eq!(metrics.overlap_ns, 0);
        assert_eq!(metrics.overlap_count, 0);
    }

    #[test]
    fn test_overlap_metrics_efficiency() {
        let mut metrics = OverlapMetrics::default();

        // No overlap case
        metrics.compute_ns = 1_000_000; // 1ms
        metrics.transfer_ns = 500_000; // 0.5ms
        metrics.overlap_ns = 0;
        assert!((metrics.overlap_efficiency() - 0.0).abs() < 0.001);

        // Perfect overlap case
        metrics.overlap_ns = 500_000; // Transfer fully overlapped with compute
        assert!((metrics.overlap_efficiency() - 1.0).abs() < 0.001);

        // Partial overlap case
        metrics.overlap_ns = 250_000; // 50% of transfer time overlapped
        assert!((metrics.overlap_efficiency() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_overlap_metrics_throughput() {
        let mut metrics = OverlapMetrics::default();

        // No overlap: sequential time = 1.5ms, overlapped = 1.5ms
        metrics.compute_ns = 1_000_000;
        metrics.transfer_ns = 500_000;
        metrics.overlap_ns = 0;
        assert!((metrics.throughput_improvement() - 1.0).abs() < 0.001);

        // Full overlap: sequential = 1.5ms, overlapped = 1.0ms
        metrics.overlap_ns = 500_000;
        assert!((metrics.throughput_improvement() - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_overlap_metrics_merge() {
        let mut metrics1 = OverlapMetrics {
            compute_ns: 1000,
            transfer_ns: 500,
            overlap_ns: 200,
            overlap_count: 1,
        };

        let metrics2 = OverlapMetrics {
            compute_ns: 2000,
            transfer_ns: 1000,
            overlap_ns: 400,
            overlap_count: 2,
        };

        metrics1.merge(&metrics2);

        assert_eq!(metrics1.compute_ns, 3000);
        assert_eq!(metrics1.transfer_ns, 1500);
        assert_eq!(metrics1.overlap_ns, 600);
        assert_eq!(metrics1.overlap_count, 3);
    }

    #[test]
    fn test_overlap_metrics_display() {
        let metrics = OverlapMetrics {
            compute_ns: 1_000_000,
            transfer_ns: 500_000,
            overlap_ns: 250_000,
            overlap_count: 1,
        };
        let display = format!("{}", metrics);
        assert!(display.contains("compute=1.00ms"));
        assert!(display.contains("transfer=0.50ms"));
        assert!(display.contains("50.0% efficiency"));
    }
}
