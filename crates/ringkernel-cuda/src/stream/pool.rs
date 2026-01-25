//! Stream pool for multi-workload concurrent execution.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::manager::StreamId;

/// Stream pool for managing concurrent workload execution.
///
/// Maps workload IDs to dedicated streams for optimal parallelism.
#[derive(Debug)]
pub struct StreamPool {
    /// Workload-to-stream assignment.
    workload_streams: HashMap<u32, StreamId>,
    /// Stream utilization tracking (kernel launches per stream).
    stream_utilization: Vec<AtomicU64>,
    /// Total kernel launches.
    total_launches: AtomicU64,
    /// Stream pool creation time.
    created_at: Instant,
}

impl StreamPool {
    /// Creates a new stream pool.
    #[must_use]
    pub fn new(num_streams: usize) -> Self {
        let mut stream_utilization = Vec::with_capacity(num_streams);
        for _ in 0..num_streams {
            stream_utilization.push(AtomicU64::new(0));
        }

        Self {
            workload_streams: HashMap::new(),
            stream_utilization,
            total_launches: AtomicU64::new(0),
            created_at: Instant::now(),
        }
    }

    /// Assigns a workload to a specific stream.
    pub fn assign_workload(&mut self, workload_id: u32, stream_id: StreamId) {
        self.workload_streams.insert(workload_id, stream_id);
    }

    /// Gets the stream assigned to a workload.
    #[must_use]
    pub fn stream_for_workload(&self, workload_id: u32) -> Option<StreamId> {
        self.workload_streams.get(&workload_id).copied()
    }

    /// Records a kernel launch on a stream.
    pub fn record_launch(&self, stream_id: StreamId) {
        self.total_launches.fetch_add(1, Ordering::Relaxed);
        if let StreamId::Compute(idx) = stream_id {
            if idx < self.stream_utilization.len() {
                self.stream_utilization[idx].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Returns stream utilization statistics.
    #[must_use]
    pub fn stats(&self) -> StreamPoolStats {
        let total = self.total_launches.load(Ordering::Relaxed);
        let per_stream: Vec<u64> = self
            .stream_utilization
            .iter()
            .map(|u| u.load(Ordering::Relaxed))
            .collect();
        let elapsed_secs = self.created_at.elapsed().as_secs_f64();

        StreamPoolStats {
            total_launches: total,
            per_stream_launches: per_stream,
            launches_per_second: if elapsed_secs > 0.0 {
                total as f64 / elapsed_secs
            } else {
                0.0
            },
            workload_count: self.workload_streams.len(),
        }
    }

    /// Selects the least-utilized stream for load balancing.
    #[must_use]
    pub fn least_utilized_stream(&self) -> StreamId {
        let (min_idx, _) = self
            .stream_utilization
            .iter()
            .enumerate()
            .min_by_key(|(_, u)| u.load(Ordering::Relaxed))
            .unwrap_or((0, &self.stream_utilization[0]));
        StreamId::Compute(min_idx)
    }

    /// Assigns a workload to the least-utilized stream.
    pub fn assign_to_least_utilized(&mut self, workload_id: u32) -> StreamId {
        let stream_id = self.least_utilized_stream();
        self.assign_workload(workload_id, stream_id);
        stream_id
    }

    /// Returns the number of streams in the pool.
    #[must_use]
    pub fn num_streams(&self) -> usize {
        self.stream_utilization.len()
    }

    /// Returns the number of assigned workloads.
    #[must_use]
    pub fn num_workloads(&self) -> usize {
        self.workload_streams.len()
    }

    /// Resets utilization counters.
    pub fn reset_utilization(&self) {
        self.total_launches.store(0, Ordering::Relaxed);
        for util in &self.stream_utilization {
            util.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for StreamPool {
    fn default() -> Self {
        Self::new(4)
    }
}

/// Statistics for stream pool utilization.
#[derive(Debug, Clone)]
pub struct StreamPoolStats {
    /// Total kernel launches across all streams.
    pub total_launches: u64,
    /// Kernel launches per stream.
    pub per_stream_launches: Vec<u64>,
    /// Average launches per second.
    pub launches_per_second: f64,
    /// Number of workloads assigned to streams.
    pub workload_count: usize,
}

impl StreamPoolStats {
    /// Returns the most utilized stream index.
    #[must_use]
    pub fn most_utilized_stream(&self) -> Option<usize> {
        self.per_stream_launches
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
    }

    /// Returns the least utilized stream index.
    #[must_use]
    pub fn least_utilized_stream(&self) -> Option<usize> {
        self.per_stream_launches
            .iter()
            .enumerate()
            .min_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
    }

    /// Returns the utilization balance ratio (0.0 = all on one stream, 1.0 = perfectly balanced).
    #[must_use]
    pub fn balance_ratio(&self) -> f64 {
        if self.per_stream_launches.is_empty() || self.total_launches == 0 {
            return 1.0;
        }

        let expected = self.total_launches as f64 / self.per_stream_launches.len() as f64;
        let variance: f64 = self
            .per_stream_launches
            .iter()
            .map(|&count| (count as f64 - expected).powi(2))
            .sum::<f64>()
            / self.per_stream_launches.len() as f64;

        let max_variance = expected.powi(2) * (self.per_stream_launches.len() as f64 - 1.0);
        if max_variance == 0.0 {
            return 1.0;
        }

        1.0 - (variance / max_variance).sqrt()
    }
}

impl std::fmt::Display for StreamPoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StreamPool: {} launches ({:.1}/s), {} workloads, {:.1}% balanced",
            self.total_launches,
            self.launches_per_second,
            self.workload_count,
            self.balance_ratio() * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_pool_creation() {
        let pool = StreamPool::new(4);
        let stats = pool.stats();
        assert_eq!(stats.total_launches, 0);
        assert_eq!(stats.per_stream_launches.len(), 4);
    }

    #[test]
    fn test_stream_pool_workload_assignment() {
        let mut pool = StreamPool::new(4);

        // Assign workloads to streams
        pool.assign_workload(1, StreamId::Compute(0));
        pool.assign_workload(2, StreamId::Compute(1));
        pool.assign_workload(8, StreamId::Compute(2));

        assert_eq!(pool.stream_for_workload(1), Some(StreamId::Compute(0)));
        assert_eq!(pool.stream_for_workload(2), Some(StreamId::Compute(1)));
        assert_eq!(pool.stream_for_workload(8), Some(StreamId::Compute(2)));
        assert_eq!(pool.stream_for_workload(99), None);
    }

    #[test]
    fn test_stream_pool_utilization() {
        let pool = StreamPool::new(4);

        // Record some launches
        pool.record_launch(StreamId::Compute(0));
        pool.record_launch(StreamId::Compute(0));
        pool.record_launch(StreamId::Compute(1));
        pool.record_launch(StreamId::Compute(2));

        let stats = pool.stats();
        assert_eq!(stats.total_launches, 4);
        assert_eq!(stats.per_stream_launches[0], 2);
        assert_eq!(stats.per_stream_launches[1], 1);
        assert_eq!(stats.per_stream_launches[2], 1);
        assert_eq!(stats.per_stream_launches[3], 0);
    }

    #[test]
    fn test_stream_pool_least_utilized() {
        let pool = StreamPool::new(4);

        // Stream 0 has 3 launches, stream 1 has 1, streams 2,3 have 0
        pool.record_launch(StreamId::Compute(0));
        pool.record_launch(StreamId::Compute(0));
        pool.record_launch(StreamId::Compute(0));
        pool.record_launch(StreamId::Compute(1));

        // Least utilized should be stream 2 or 3 (both have 0)
        let least = pool.least_utilized_stream();
        match least {
            StreamId::Compute(idx) => assert!(idx >= 2, "Expected stream 2 or 3"),
            _ => panic!("Expected compute stream"),
        }
    }

    #[test]
    fn test_stream_pool_reset() {
        let pool = StreamPool::new(4);

        pool.record_launch(StreamId::Compute(0));
        pool.record_launch(StreamId::Compute(1));

        assert_eq!(pool.stats().total_launches, 2);

        pool.reset_utilization();

        assert_eq!(pool.stats().total_launches, 0);
    }

    #[test]
    fn test_stream_pool_stats_balance() {
        let pool = StreamPool::new(4);

        // Perfectly balanced (1 launch each)
        pool.record_launch(StreamId::Compute(0));
        pool.record_launch(StreamId::Compute(1));
        pool.record_launch(StreamId::Compute(2));
        pool.record_launch(StreamId::Compute(3));

        let stats = pool.stats();
        assert!((stats.balance_ratio() - 1.0).abs() < 0.001);
    }
}
