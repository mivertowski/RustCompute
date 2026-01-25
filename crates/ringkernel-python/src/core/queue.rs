//! Queue types for RingKernel Python bindings.
//!
//! Provides queue statistics and monitoring.

use pyo3::prelude::*;
use pyo3::types::PyType;
use ringkernel_core::queue::{
    PartitionedQueueStats, QueueHealth, QueueMetrics, QueueMonitor, QueueStats, QueueTier,
};

/// Statistics snapshot for a message queue.
///
/// Provides insight into queue utilization and performance.
///
/// Example:
///     >>> stats = kernel.queue_stats()
///     >>> print(f"Depth: {stats.depth}/{stats.max_depth}")
///     >>> print(f"Throughput: {stats.enqueued} enqueued, {stats.dequeued} dequeued")
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyQueueStats {
    inner: QueueStats,
}

#[pymethods]
impl PyQueueStats {
    /// Number of messages enqueued (lifetime).
    #[getter]
    fn enqueued(&self) -> u64 {
        self.inner.enqueued
    }

    /// Number of messages dequeued (lifetime).
    #[getter]
    fn dequeued(&self) -> u64 {
        self.inner.dequeued
    }

    /// Number of messages dropped due to full queue.
    #[getter]
    fn dropped(&self) -> u64 {
        self.inner.dropped
    }

    /// Current queue depth (messages waiting).
    #[getter]
    fn depth(&self) -> u64 {
        self.inner.depth
    }

    /// Maximum queue depth observed.
    #[getter]
    fn max_depth(&self) -> u64 {
        self.inner.max_depth
    }

    /// Calculate current utilization as a fraction (0.0-1.0).
    ///
    /// Args:
    ///     capacity: Queue capacity.
    ///
    /// Returns:
    ///     Utilization fraction.
    fn utilization(&self, capacity: usize) -> f64 {
        if capacity == 0 {
            0.0
        } else {
            self.inner.depth as f64 / capacity as f64
        }
    }

    /// Calculate drop rate as a fraction.
    ///
    /// Returns:
    ///     Fraction of messages dropped (0.0-1.0).
    fn drop_rate(&self) -> f64 {
        let total = self.inner.enqueued + self.inner.dropped;
        if total == 0 {
            0.0
        } else {
            self.inner.dropped as f64 / total as f64
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "QueueStats(depth={}, enqueued={}, dequeued={}, dropped={})",
            self.inner.depth, self.inner.enqueued, self.inner.dequeued, self.inner.dropped
        )
    }
}

impl From<QueueStats> for PyQueueStats {
    fn from(inner: QueueStats) -> Self {
        Self { inner }
    }
}

/// Queue capacity tier for sizing.
///
/// Tiers provide standard capacity levels for different workloads.
///
/// Tiers:
///     - SMALL: 256 messages
///     - MEDIUM: 1024 messages
///     - LARGE: 4096 messages
///     - EXTRA_LARGE: 16384 messages
#[pyclass(frozen, eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyQueueTier {
    /// 256 messages.
    Small = 256,
    /// 1024 messages.
    Medium = 1024,
    /// 4096 messages.
    Large = 4096,
    /// 16384 messages.
    ExtraLarge = 16384,
}

#[pymethods]
impl PyQueueTier {
    /// Get the capacity for this tier.
    #[getter]
    fn capacity(&self) -> usize {
        match self {
            Self::Small => 256,
            Self::Medium => 1024,
            Self::Large => 4096,
            Self::ExtraLarge => 16384,
        }
    }

    /// Select a tier based on throughput requirements.
    ///
    /// Args:
    ///     messages_per_second: Expected message rate.
    ///     headroom_ms: Target headroom in milliseconds (default: 100).
    ///
    /// Returns:
    ///     Appropriate tier for the workload.
    #[classmethod]
    #[pyo3(signature = (messages_per_second, headroom_ms=100))]
    fn for_throughput(
        _cls: &Bound<'_, PyType>,
        messages_per_second: u64,
        headroom_ms: u64,
    ) -> Self {
        QueueTier::for_throughput(messages_per_second, headroom_ms).into()
    }

    /// Get the next larger tier.
    ///
    /// Returns:
    ///     The next tier up, or self if already largest.
    fn upgrade(&self) -> Self {
        let tier: QueueTier = (*self).into();
        tier.upgrade().into()
    }

    /// Get the next smaller tier.
    ///
    /// Returns:
    ///     The next tier down, or self if already smallest.
    fn downgrade(&self) -> Self {
        let tier: QueueTier = (*self).into();
        tier.downgrade().into()
    }

    fn __repr__(&self) -> String {
        match self {
            Self::Small => "QueueTier.Small".to_string(),
            Self::Medium => "QueueTier.Medium".to_string(),
            Self::Large => "QueueTier.Large".to_string(),
            Self::ExtraLarge => "QueueTier.ExtraLarge".to_string(),
        }
    }
}

impl From<QueueTier> for PyQueueTier {
    fn from(tier: QueueTier) -> Self {
        match tier {
            QueueTier::Small => Self::Small,
            QueueTier::Medium => Self::Medium,
            QueueTier::Large => Self::Large,
            QueueTier::ExtraLarge => Self::ExtraLarge,
        }
    }
}

impl From<PyQueueTier> for QueueTier {
    fn from(tier: PyQueueTier) -> Self {
        match tier {
            PyQueueTier::Small => Self::Small,
            PyQueueTier::Medium => Self::Medium,
            PyQueueTier::Large => Self::Large,
            PyQueueTier::ExtraLarge => Self::ExtraLarge,
        }
    }
}

/// Queue health status.
///
/// Indicates whether a queue is operating normally or under stress.
#[pyclass(frozen, eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyQueueHealth {
    /// Queue is operating normally.
    Healthy = 0,
    /// Queue utilization is high (warning threshold exceeded).
    Warning = 1,
    /// Queue is near capacity (critical threshold exceeded).
    Critical = 2,
}

#[pymethods]
impl PyQueueHealth {
    fn __repr__(&self) -> String {
        match self {
            Self::Healthy => "QueueHealth.Healthy".to_string(),
            Self::Warning => "QueueHealth.Warning".to_string(),
            Self::Critical => "QueueHealth.Critical".to_string(),
        }
    }
}

impl From<QueueHealth> for PyQueueHealth {
    fn from(health: QueueHealth) -> Self {
        match health {
            QueueHealth::Healthy => Self::Healthy,
            QueueHealth::Warning => Self::Warning,
            QueueHealth::Critical => Self::Critical,
        }
    }
}

/// Queue monitor for health tracking.
///
/// Monitors queue utilization and suggests tier upgrades.
///
/// Example:
///     >>> monitor = QueueMonitor(warning_threshold=0.75, critical_threshold=0.9)
///     >>> health = monitor.check(queue)
///     >>> if health == QueueHealth.Critical:
///     ...     print("Queue overloaded!")
#[pyclass]
#[derive(Clone)]
pub struct PyQueueMonitor {
    inner: QueueMonitor,
}

#[pymethods]
impl PyQueueMonitor {
    /// Create a new queue monitor.
    ///
    /// Args:
    ///     warning_threshold: Utilization fraction for warning (default: 0.75).
    ///     critical_threshold: Utilization fraction for critical (default: 0.90).
    ///
    /// Returns:
    ///     A new QueueMonitor.
    #[new]
    #[pyo3(signature = (warning_threshold=0.75, critical_threshold=0.90))]
    fn new(warning_threshold: f64, critical_threshold: f64) -> Self {
        Self {
            inner: QueueMonitor::new(warning_threshold, critical_threshold),
        }
    }

    /// Create a monitor with default thresholds.
    ///
    /// Default: warning=0.75, critical=0.90.
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: QueueMonitor::default(),
        }
    }

    /// Warning threshold.
    #[getter]
    fn warning_threshold(&self) -> f64 {
        self.inner.warning_threshold
    }

    /// Critical threshold.
    #[getter]
    fn critical_threshold(&self) -> f64 {
        self.inner.critical_threshold
    }

    /// Calculate utilization for a queue.
    ///
    /// Args:
    ///     depth: Current queue depth.
    ///     capacity: Queue capacity.
    ///
    /// Returns:
    ///     Utilization as a fraction (0.0-1.0).
    fn utilization(&self, depth: u64, capacity: usize) -> f64 {
        if capacity == 0 {
            0.0
        } else {
            depth as f64 / capacity as f64
        }
    }

    /// Calculate utilization as a percentage.
    ///
    /// Args:
    ///     depth: Current queue depth.
    ///     capacity: Queue capacity.
    ///
    /// Returns:
    ///     Utilization as a percentage (0-100).
    fn utilization_percent(&self, depth: u64, capacity: usize) -> f64 {
        self.utilization(depth, capacity) * 100.0
    }

    /// Check queue health based on utilization.
    ///
    /// Args:
    ///     depth: Current queue depth.
    ///     capacity: Queue capacity.
    ///
    /// Returns:
    ///     Health status.
    fn check_health(&self, depth: u64, capacity: usize) -> PyQueueHealth {
        let util = self.utilization(depth, capacity);
        if util >= self.inner.critical_threshold {
            PyQueueHealth::Critical
        } else if util >= self.inner.warning_threshold {
            PyQueueHealth::Warning
        } else {
            PyQueueHealth::Healthy
        }
    }

    /// Suggest a tier upgrade if needed.
    ///
    /// Args:
    ///     depth: Current queue depth.
    ///     capacity: Queue capacity.
    ///     current_tier: Current queue tier.
    ///
    /// Returns:
    ///     Suggested tier if upgrade needed, None otherwise.
    fn suggest_upgrade(
        &self,
        depth: u64,
        capacity: usize,
        current_tier: PyQueueTier,
    ) -> Option<PyQueueTier> {
        let util = self.utilization(depth, capacity);
        if util >= self.inner.warning_threshold && current_tier != PyQueueTier::ExtraLarge {
            Some(current_tier.upgrade())
        } else {
            None
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "QueueMonitor(warning={:.0}%, critical={:.0}%)",
            self.inner.warning_threshold * 100.0,
            self.inner.critical_threshold * 100.0
        )
    }
}

/// Complete queue metrics snapshot.
///
/// Combines health status, statistics, and tier information.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyQueueMetrics {
    /// Health status.
    #[pyo3(get)]
    health: PyQueueHealth,
    /// Current utilization (0.0-1.0).
    #[pyo3(get)]
    utilization: f64,
    /// Queue statistics.
    #[pyo3(get)]
    stats: PyQueueStats,
    /// Current tier (if known).
    #[pyo3(get)]
    tier: Option<PyQueueTier>,
    /// Suggested tier upgrade (if any).
    #[pyo3(get)]
    suggested_upgrade: Option<PyQueueTier>,
}

#[pymethods]
impl PyQueueMetrics {
    fn __repr__(&self) -> String {
        format!(
            "QueueMetrics(health={:?}, utilization={:.1}%, depth={})",
            self.health,
            self.utilization * 100.0,
            self.stats.depth()
        )
    }
}

impl From<QueueMetrics> for PyQueueMetrics {
    fn from(metrics: QueueMetrics) -> Self {
        Self {
            health: metrics.health.into(),
            utilization: metrics.utilization,
            stats: metrics.stats.into(),
            tier: metrics.tier.map(Into::into),
            suggested_upgrade: metrics.suggested_upgrade.map(Into::into),
        }
    }
}

/// Statistics for partitioned queues.
///
/// Provides aggregate and per-partition metrics for load balancing analysis.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyPartitionedQueueStats {
    inner: PartitionedQueueStats,
}

#[pymethods]
impl PyPartitionedQueueStats {
    /// Aggregate statistics across all partitions.
    #[getter]
    fn total(&self) -> PyQueueStats {
        self.inner.total.clone().into()
    }

    /// Number of partitions.
    #[getter]
    fn partition_count(&self) -> usize {
        self.inner.partition_count
    }

    /// Get statistics for a specific partition.
    ///
    /// Args:
    ///     partition: Partition index.
    ///
    /// Returns:
    ///     Statistics for the partition, or None if out of range.
    fn partition_stats(&self, partition: usize) -> Option<PyQueueStats> {
        self.inner
            .partition_stats
            .get(partition)
            .cloned()
            .map(Into::into)
    }

    /// Calculate load imbalance ratio.
    ///
    /// Returns:
    ///     Ratio of max partition depth to average (1.0 = perfect balance).
    fn load_imbalance(&self) -> f64 {
        self.inner.load_imbalance()
    }

    /// Get maximum partition utilization.
    ///
    /// Args:
    ///     capacity_per_partition: Capacity of each partition.
    ///
    /// Returns:
    ///     Maximum utilization across all partitions.
    fn max_partition_utilization(&self, capacity_per_partition: usize) -> f64 {
        self.inner.max_partition_utilization(capacity_per_partition)
    }

    fn __repr__(&self) -> String {
        format!(
            "PartitionedQueueStats(partitions={}, imbalance={:.2})",
            self.inner.partition_count,
            self.inner.load_imbalance()
        )
    }
}

impl From<PartitionedQueueStats> for PyPartitionedQueueStats {
    fn from(inner: PartitionedQueueStats) -> Self {
        Self { inner }
    }
}

/// Register queue types with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create queue submodule
    let queue = PyModule::new_bound(m.py(), "queue")?;
    queue.add_class::<PyQueueStats>()?;
    queue.add_class::<PyQueueTier>()?;
    queue.add_class::<PyQueueHealth>()?;
    queue.add_class::<PyQueueMonitor>()?;
    queue.add_class::<PyQueueMetrics>()?;
    queue.add_class::<PyPartitionedQueueStats>()?;

    // Add to parent module
    m.add_submodule(&queue)?;

    // Also add commonly-used types at top level
    m.add_class::<PyQueueStats>()?;
    m.add_class::<PyQueueTier>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_tier_capacity() {
        assert_eq!(PyQueueTier::Small.capacity(), 256);
        assert_eq!(PyQueueTier::Large.capacity(), 4096);
    }

    #[test]
    fn test_queue_tier_upgrade() {
        assert_eq!(PyQueueTier::Small.upgrade(), PyQueueTier::Medium);
        assert_eq!(PyQueueTier::ExtraLarge.upgrade(), PyQueueTier::ExtraLarge);
    }

    #[test]
    fn test_monitor_health() {
        let monitor = PyQueueMonitor::new(0.75, 0.90);
        assert_eq!(monitor.check_health(50, 100), PyQueueHealth::Healthy);
        assert_eq!(monitor.check_health(80, 100), PyQueueHealth::Warning);
        assert_eq!(monitor.check_health(95, 100), PyQueueHealth::Critical);
    }
}
