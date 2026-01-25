//! Hybrid CPU-GPU dispatcher.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::config::{HybridConfig, ProcessingMode};
use super::error::{HybridError, HybridResult};
use super::stats::HybridStats;
use super::traits::HybridWorkload;

/// Dispatcher for routing workloads between CPU and GPU.
///
/// The dispatcher uses the configured `ProcessingMode` to decide where to
/// execute each workload. In `Adaptive` mode, it learns the optimal threshold
/// from runtime measurements.
pub struct HybridDispatcher {
    /// Configuration.
    config: HybridConfig,
    /// Execution statistics.
    stats: Arc<HybridStats>,
    /// Adaptive threshold (updated based on measurements).
    adaptive_threshold: AtomicUsize,
}

impl HybridDispatcher {
    /// Creates a new hybrid dispatcher.
    #[must_use]
    pub fn new(config: HybridConfig) -> Self {
        let initial_threshold = match config.mode {
            ProcessingMode::Hybrid { gpu_threshold } => gpu_threshold,
            _ => 10_000,
        };

        Self {
            config,
            stats: Arc::new(HybridStats::new()),
            adaptive_threshold: AtomicUsize::new(initial_threshold),
        }
    }

    /// Creates a dispatcher with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(HybridConfig::default())
    }

    /// Returns whether GPU should be used for the given workload size.
    #[must_use]
    pub fn should_use_gpu(&self, workload_size: usize) -> bool {
        if !self.config.gpu_available {
            return false;
        }

        match self.config.mode {
            ProcessingMode::GpuOnly => true,
            ProcessingMode::CpuOnly => false,
            ProcessingMode::Hybrid { gpu_threshold } => workload_size >= gpu_threshold,
            ProcessingMode::Adaptive => {
                workload_size >= self.adaptive_threshold.load(Ordering::Relaxed)
            }
        }
    }

    /// Executes a workload using the appropriate backend.
    ///
    /// Returns the result and records execution statistics.
    pub fn execute<W: HybridWorkload>(&self, workload: &W) -> HybridResult<W::Result> {
        let size = workload.workload_size();

        // Check workload size limit
        if self.config.max_workload_size > 0 && size > self.config.max_workload_size {
            return Err(HybridError::WorkloadTooLarge {
                requested: size,
                maximum: self.config.max_workload_size,
            });
        }

        let use_gpu = self.should_use_gpu(size) && workload.supports_gpu();

        if use_gpu {
            let start = Instant::now();
            let result = workload.execute_gpu()?;
            let elapsed = start.elapsed();
            self.stats.record_gpu_execution(elapsed, size);
            Ok(result)
        } else {
            let start = Instant::now();
            let result = workload.execute_cpu();
            let elapsed = start.elapsed();
            self.stats.record_cpu_execution(elapsed, size);
            Ok(result)
        }
    }

    /// Executes a workload and measures both backends for comparison.
    ///
    /// In `Adaptive` mode, this updates the threshold based on measurements.
    /// Returns the result from the faster backend.
    pub fn execute_measured<W: HybridWorkload>(&self, workload: &W) -> HybridResult<W::Result>
    where
        W::Result: Clone,
    {
        let size = workload.workload_size();

        if !self.config.gpu_available || !workload.supports_gpu() {
            let start = Instant::now();
            let result = workload.execute_cpu();
            let elapsed = start.elapsed();
            self.stats.record_cpu_execution(elapsed, size);
            return Ok(result);
        }

        // Execute on CPU
        let cpu_start = Instant::now();
        let cpu_result = workload.execute_cpu();
        let cpu_time = cpu_start.elapsed();

        // Execute on GPU
        let gpu_start = Instant::now();
        let gpu_result = workload.execute_gpu()?;
        let gpu_time = gpu_start.elapsed();

        // Update adaptive threshold
        self.update_adaptive_threshold(size, cpu_time, gpu_time);

        // Record whichever was faster
        if gpu_time < cpu_time {
            self.stats.record_gpu_execution(gpu_time, size);
            Ok(gpu_result)
        } else {
            self.stats.record_cpu_execution(cpu_time, size);
            Ok(cpu_result)
        }
    }

    /// Updates the adaptive threshold based on runtime measurements.
    pub fn update_adaptive_threshold(
        &self,
        _workload_size: usize,
        cpu_time: Duration,
        gpu_time: Duration,
    ) {
        if !matches!(self.config.mode, ProcessingMode::Adaptive) {
            return;
        }

        let current = self.adaptive_threshold.load(Ordering::Relaxed);
        let ratio = cpu_time.as_nanos() as f32 / gpu_time.as_nanos().max(1) as f32;

        let new_threshold = if ratio > 1.5 {
            // GPU significantly faster - lower threshold
            let adjustment = (current as f32 * self.config.learning_rate) as usize;
            current
                .saturating_sub(adjustment)
                .max(self.config.min_adaptive_threshold)
        } else if ratio < 0.7 {
            // CPU significantly faster - raise threshold
            let adjustment = (current as f32 * self.config.learning_rate) as usize;
            current
                .saturating_add(adjustment)
                .min(self.config.max_adaptive_threshold)
        } else {
            current
        };

        self.adaptive_threshold
            .store(new_threshold, Ordering::Relaxed);
        self.stats.set_learned_threshold(new_threshold);
    }

    /// Forces execution on CPU regardless of mode.
    pub fn execute_cpu<W: HybridWorkload>(&self, workload: &W) -> W::Result {
        let start = Instant::now();
        let result = workload.execute_cpu();
        let elapsed = start.elapsed();
        self.stats
            .record_cpu_execution(elapsed, workload.workload_size());
        result
    }

    /// Forces execution on GPU regardless of mode.
    pub fn execute_gpu<W: HybridWorkload>(&self, workload: &W) -> HybridResult<W::Result> {
        if !self.config.gpu_available {
            return Err(HybridError::GpuNotAvailable);
        }

        let start = Instant::now();
        let result = workload.execute_gpu()?;
        let elapsed = start.elapsed();
        self.stats
            .record_gpu_execution(elapsed, workload.workload_size());
        Ok(result)
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &HybridConfig {
        &self.config
    }

    /// Returns the execution statistics.
    #[must_use]
    pub fn stats(&self) -> &HybridStats {
        &self.stats
    }

    /// Returns a shared reference to the statistics.
    #[must_use]
    pub fn stats_arc(&self) -> Arc<HybridStats> {
        Arc::clone(&self.stats)
    }

    /// Returns the current adaptive threshold.
    #[must_use]
    pub fn adaptive_threshold(&self) -> usize {
        self.adaptive_threshold.load(Ordering::Relaxed)
    }

    /// Manually sets the adaptive threshold.
    pub fn set_adaptive_threshold(&self, threshold: usize) {
        let clamped = threshold
            .max(self.config.min_adaptive_threshold)
            .min(self.config.max_adaptive_threshold);
        self.adaptive_threshold.store(clamped, Ordering::Relaxed);
        self.stats.set_learned_threshold(clamped);
    }
}

/// Result of a hybrid execution.
#[derive(Debug, Clone)]
pub struct HybridExecutionResult<T> {
    /// The result value.
    pub value: T,
    /// Execution time.
    pub execution_time: Duration,
    /// Whether GPU was used.
    pub used_gpu: bool,
    /// Workload size.
    pub workload_size: usize,
}

impl<T> HybridExecutionResult<T> {
    /// Creates a new execution result.
    pub fn new(value: T, execution_time: Duration, used_gpu: bool, workload_size: usize) -> Self {
        Self {
            value,
            execution_time,
            used_gpu,
            workload_size,
        }
    }

    /// Returns throughput in elements per second.
    #[must_use]
    pub fn throughput(&self) -> f64 {
        if self.execution_time.is_zero() {
            return 0.0;
        }
        self.workload_size as f64 / self.execution_time.as_secs_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestWorkload {
        size: usize,
        supports_gpu: bool,
    }

    impl HybridWorkload for TestWorkload {
        type Result = usize;

        fn workload_size(&self) -> usize {
            self.size
        }

        fn execute_cpu(&self) -> Self::Result {
            self.size * 2
        }

        fn execute_gpu(&self) -> HybridResult<Self::Result> {
            Ok(self.size * 3)
        }

        fn supports_gpu(&self) -> bool {
            self.supports_gpu
        }
    }

    #[test]
    fn test_dispatcher_new() {
        let dispatcher = HybridDispatcher::new(HybridConfig::default());
        assert!(!dispatcher.config().gpu_available);
    }

    #[test]
    fn test_should_use_gpu_hybrid() {
        let config = HybridConfig::builder()
            .mode(ProcessingMode::Hybrid {
                gpu_threshold: 1000,
            })
            .gpu_available(true)
            .build();
        let dispatcher = HybridDispatcher::new(config);

        assert!(!dispatcher.should_use_gpu(500));
        assert!(dispatcher.should_use_gpu(1000));
        assert!(dispatcher.should_use_gpu(5000));
    }

    #[test]
    fn test_should_use_gpu_cpu_only() {
        let config = HybridConfig::cpu_only();
        let dispatcher = HybridDispatcher::new(config);

        assert!(!dispatcher.should_use_gpu(1_000_000));
    }

    #[test]
    fn test_should_use_gpu_gpu_only() {
        let config = HybridConfig::gpu_only();
        let dispatcher = HybridDispatcher::new(config);

        assert!(dispatcher.should_use_gpu(1));
    }

    #[test]
    fn test_execute_cpu_path() {
        let config = HybridConfig::cpu_only();
        let dispatcher = HybridDispatcher::new(config);

        let workload = TestWorkload {
            size: 100,
            supports_gpu: true,
        };
        let result = dispatcher.execute(&workload).unwrap();

        assert_eq!(result, 200); // CPU result
        assert_eq!(dispatcher.stats().cpu_executions(), 1);
    }

    #[test]
    fn test_execute_gpu_path() {
        let config = HybridConfig::gpu_only();
        let dispatcher = HybridDispatcher::new(config);

        let workload = TestWorkload {
            size: 100,
            supports_gpu: true,
        };
        let result = dispatcher.execute(&workload).unwrap();

        assert_eq!(result, 300); // GPU result
        assert_eq!(dispatcher.stats().gpu_executions(), 1);
    }

    #[test]
    fn test_execute_falls_back_if_gpu_unsupported() {
        let config = HybridConfig::gpu_only();
        let dispatcher = HybridDispatcher::new(config);

        let workload = TestWorkload {
            size: 100,
            supports_gpu: false,
        };
        let result = dispatcher.execute(&workload).unwrap();

        assert_eq!(result, 200); // CPU result
        assert_eq!(dispatcher.stats().cpu_executions(), 1);
    }

    #[test]
    fn test_workload_too_large() {
        let config = HybridConfig::builder().max_workload_size(100).build();
        let dispatcher = HybridDispatcher::new(config);

        let workload = TestWorkload {
            size: 1000,
            supports_gpu: true,
        };
        let result = dispatcher.execute(&workload);

        assert!(matches!(
            result,
            Err(HybridError::WorkloadTooLarge {
                requested: 1000,
                maximum: 100
            })
        ));
    }

    #[test]
    fn test_adaptive_threshold_update_gpu_faster() {
        let config = HybridConfig::builder()
            .mode(ProcessingMode::Adaptive)
            .gpu_available(true)
            .learning_rate(0.5)
            .build();
        let dispatcher = HybridDispatcher::new(config);

        let initial = dispatcher.adaptive_threshold();

        // GPU significantly faster
        dispatcher.update_adaptive_threshold(
            5000,
            Duration::from_millis(100),
            Duration::from_millis(10),
        );

        assert!(dispatcher.adaptive_threshold() < initial);
    }

    #[test]
    fn test_adaptive_threshold_update_cpu_faster() {
        let config = HybridConfig::builder()
            .mode(ProcessingMode::Adaptive)
            .gpu_available(true)
            .learning_rate(0.5)
            .build();
        let dispatcher = HybridDispatcher::new(config);

        let initial = dispatcher.adaptive_threshold();

        // CPU significantly faster
        dispatcher.update_adaptive_threshold(
            5000,
            Duration::from_millis(10),
            Duration::from_millis(100),
        );

        assert!(dispatcher.adaptive_threshold() > initial);
    }

    #[test]
    fn test_set_adaptive_threshold_clamping() {
        let config = HybridConfig::builder()
            .mode(ProcessingMode::Adaptive)
            .min_adaptive_threshold(100)
            .max_adaptive_threshold(10000)
            .build();
        let dispatcher = HybridDispatcher::new(config);

        dispatcher.set_adaptive_threshold(50);
        assert_eq!(dispatcher.adaptive_threshold(), 100);

        dispatcher.set_adaptive_threshold(50000);
        assert_eq!(dispatcher.adaptive_threshold(), 10000);
    }

    #[test]
    fn test_execution_result_throughput() {
        let result = HybridExecutionResult::new(42, Duration::from_secs(1), false, 1000);
        assert!((result.throughput() - 1000.0).abs() < f64::EPSILON);
    }
}
