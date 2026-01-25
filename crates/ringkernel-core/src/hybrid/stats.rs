//! Statistics for hybrid processing decisions.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Statistics for hybrid processing decisions.
///
/// Thread-safe via atomic operations.
#[derive(Debug, Default)]
pub struct HybridStats {
    /// Total CPU executions.
    cpu_executions: AtomicU64,
    /// Total GPU executions.
    gpu_executions: AtomicU64,
    /// Total CPU time (nanoseconds).
    cpu_time_ns: AtomicU64,
    /// Total GPU time (nanoseconds).
    gpu_time_ns: AtomicU64,
    /// Total elements processed on CPU.
    cpu_elements: AtomicU64,
    /// Total elements processed on GPU.
    gpu_elements: AtomicU64,
    /// Crossover threshold learned from measurements.
    learned_threshold: AtomicU64,
}

impl HybridStats {
    /// Creates new empty statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a CPU execution.
    pub fn record_cpu_execution(&self, duration: Duration, elements: usize) {
        self.cpu_executions.fetch_add(1, Ordering::Relaxed);
        self.cpu_time_ns
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.cpu_elements
            .fetch_add(elements as u64, Ordering::Relaxed);
    }

    /// Records a GPU execution.
    pub fn record_gpu_execution(&self, duration: Duration, elements: usize) {
        self.gpu_executions.fetch_add(1, Ordering::Relaxed);
        self.gpu_time_ns
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.gpu_elements
            .fetch_add(elements as u64, Ordering::Relaxed);
    }

    /// Updates the learned threshold.
    pub fn set_learned_threshold(&self, threshold: usize) {
        self.learned_threshold
            .store(threshold as u64, Ordering::Relaxed);
    }

    /// Gets the learned threshold.
    #[must_use]
    pub fn learned_threshold(&self) -> usize {
        self.learned_threshold.load(Ordering::Relaxed) as usize
    }

    /// Gets the total CPU executions.
    #[must_use]
    pub fn cpu_executions(&self) -> u64 {
        self.cpu_executions.load(Ordering::Relaxed)
    }

    /// Gets the total GPU executions.
    #[must_use]
    pub fn gpu_executions(&self) -> u64 {
        self.gpu_executions.load(Ordering::Relaxed)
    }

    /// Gets the average CPU time per execution.
    #[must_use]
    pub fn avg_cpu_time(&self) -> Duration {
        let execs = self.cpu_executions.load(Ordering::Relaxed);
        if execs == 0 {
            return Duration::ZERO;
        }
        let total_ns = self.cpu_time_ns.load(Ordering::Relaxed);
        Duration::from_nanos(total_ns / execs)
    }

    /// Gets the average GPU time per execution.
    #[must_use]
    pub fn avg_gpu_time(&self) -> Duration {
        let execs = self.gpu_executions.load(Ordering::Relaxed);
        if execs == 0 {
            return Duration::ZERO;
        }
        let total_ns = self.gpu_time_ns.load(Ordering::Relaxed);
        Duration::from_nanos(total_ns / execs)
    }

    /// Gets the CPU/GPU execution ratio.
    #[must_use]
    pub fn cpu_gpu_ratio(&self) -> f32 {
        let cpu = self.cpu_executions.load(Ordering::Relaxed) as f32;
        let gpu = self.gpu_executions.load(Ordering::Relaxed) as f32;
        if gpu == 0.0 {
            return f32::INFINITY;
        }
        cpu / gpu
    }

    /// Gets the average CPU throughput (elements per second).
    #[must_use]
    pub fn cpu_throughput(&self) -> f64 {
        let total_ns = self.cpu_time_ns.load(Ordering::Relaxed);
        let total_elements = self.cpu_elements.load(Ordering::Relaxed);
        if total_ns == 0 {
            return 0.0;
        }
        (total_elements as f64) / (total_ns as f64 / 1_000_000_000.0)
    }

    /// Gets the average GPU throughput (elements per second).
    #[must_use]
    pub fn gpu_throughput(&self) -> f64 {
        let total_ns = self.gpu_time_ns.load(Ordering::Relaxed);
        let total_elements = self.gpu_elements.load(Ordering::Relaxed);
        if total_ns == 0 {
            return 0.0;
        }
        (total_elements as f64) / (total_ns as f64 / 1_000_000_000.0)
    }

    /// Creates a snapshot of the current statistics.
    #[must_use]
    pub fn snapshot(&self) -> HybridStatsSnapshot {
        HybridStatsSnapshot {
            cpu_executions: self.cpu_executions.load(Ordering::Relaxed),
            gpu_executions: self.gpu_executions.load(Ordering::Relaxed),
            cpu_time_ns: self.cpu_time_ns.load(Ordering::Relaxed),
            gpu_time_ns: self.gpu_time_ns.load(Ordering::Relaxed),
            cpu_elements: self.cpu_elements.load(Ordering::Relaxed),
            gpu_elements: self.gpu_elements.load(Ordering::Relaxed),
            learned_threshold: self.learned_threshold.load(Ordering::Relaxed) as usize,
        }
    }

    /// Resets all statistics to zero.
    pub fn reset(&self) {
        self.cpu_executions.store(0, Ordering::Relaxed);
        self.gpu_executions.store(0, Ordering::Relaxed);
        self.cpu_time_ns.store(0, Ordering::Relaxed);
        self.gpu_time_ns.store(0, Ordering::Relaxed);
        self.cpu_elements.store(0, Ordering::Relaxed);
        self.gpu_elements.store(0, Ordering::Relaxed);
    }
}

/// A point-in-time snapshot of hybrid processing statistics.
#[derive(Debug, Clone)]
pub struct HybridStatsSnapshot {
    /// Total CPU executions.
    pub cpu_executions: u64,
    /// Total GPU executions.
    pub gpu_executions: u64,
    /// Total CPU time (nanoseconds).
    pub cpu_time_ns: u64,
    /// Total GPU time (nanoseconds).
    pub gpu_time_ns: u64,
    /// Total elements processed on CPU.
    pub cpu_elements: u64,
    /// Total elements processed on GPU.
    pub gpu_elements: u64,
    /// Learned threshold.
    pub learned_threshold: usize,
}

impl HybridStatsSnapshot {
    /// Total executions across both backends.
    #[must_use]
    pub fn total_executions(&self) -> u64 {
        self.cpu_executions + self.gpu_executions
    }

    /// GPU utilization percentage (0.0-100.0).
    #[must_use]
    pub fn gpu_utilization(&self) -> f64 {
        let total = self.total_executions();
        if total == 0 {
            return 0.0;
        }
        (self.gpu_executions as f64 / total as f64) * 100.0
    }

    /// Average CPU time per execution.
    #[must_use]
    pub fn avg_cpu_time(&self) -> Duration {
        if self.cpu_executions == 0 {
            return Duration::ZERO;
        }
        Duration::from_nanos(self.cpu_time_ns / self.cpu_executions)
    }

    /// Average GPU time per execution.
    #[must_use]
    pub fn avg_gpu_time(&self) -> Duration {
        if self.gpu_executions == 0 {
            return Duration::ZERO;
        }
        Duration::from_nanos(self.gpu_time_ns / self.gpu_executions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_new() {
        let stats = HybridStats::new();
        assert_eq!(stats.cpu_executions(), 0);
        assert_eq!(stats.gpu_executions(), 0);
    }

    #[test]
    fn test_record_cpu_execution() {
        let stats = HybridStats::new();
        stats.record_cpu_execution(Duration::from_millis(100), 1000);

        assert_eq!(stats.cpu_executions(), 1);
        assert_eq!(stats.avg_cpu_time(), Duration::from_millis(100));
    }

    #[test]
    fn test_record_gpu_execution() {
        let stats = HybridStats::new();
        stats.record_gpu_execution(Duration::from_millis(50), 10000);

        assert_eq!(stats.gpu_executions(), 1);
        assert_eq!(stats.avg_gpu_time(), Duration::from_millis(50));
    }

    #[test]
    fn test_cpu_gpu_ratio() {
        let stats = HybridStats::new();
        stats.record_cpu_execution(Duration::from_millis(100), 1000);
        stats.record_cpu_execution(Duration::from_millis(100), 1000);
        stats.record_gpu_execution(Duration::from_millis(50), 10000);

        assert!((stats.cpu_gpu_ratio() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_snapshot() {
        let stats = HybridStats::new();
        stats.record_cpu_execution(Duration::from_millis(100), 1000);
        stats.record_gpu_execution(Duration::from_millis(50), 10000);
        stats.set_learned_threshold(5000);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.cpu_executions, 1);
        assert_eq!(snapshot.gpu_executions, 1);
        assert_eq!(snapshot.learned_threshold, 5000);
        assert!((snapshot.gpu_utilization() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset() {
        let stats = HybridStats::new();
        stats.record_cpu_execution(Duration::from_millis(100), 1000);
        stats.reset();

        assert_eq!(stats.cpu_executions(), 0);
    }
}
