//! Configuration for hybrid CPU-GPU processing.

/// Processing mode for hybrid execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingMode {
    /// Always use GPU (if available).
    GpuOnly,
    /// Always use CPU (parallel via Rayon or similar).
    CpuOnly,
    /// Automatically select based on workload size.
    Hybrid {
        /// Minimum elements to use GPU (smaller workloads use CPU).
        gpu_threshold: usize,
    },
    /// Adaptive mode that learns from runtime measurements.
    Adaptive,
}

impl Default for ProcessingMode {
    fn default() -> Self {
        ProcessingMode::Hybrid {
            gpu_threshold: 10_000, // 10K elements default threshold
        }
    }
}

impl ProcessingMode {
    /// Creates a Hybrid mode with the specified threshold.
    #[must_use]
    pub fn hybrid(gpu_threshold: usize) -> Self {
        ProcessingMode::Hybrid { gpu_threshold }
    }

    /// Returns the GPU threshold if in Hybrid mode.
    #[must_use]
    pub fn threshold(&self) -> Option<usize> {
        match self {
            ProcessingMode::Hybrid { gpu_threshold } => Some(*gpu_threshold),
            _ => None,
        }
    }
}

/// Configuration for hybrid processing.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Processing mode.
    pub mode: ProcessingMode,
    /// Number of CPU threads (0 = auto-detect from Rayon).
    pub cpu_threads: usize,
    /// Whether GPU is available.
    pub gpu_available: bool,
    /// Adaptive learning rate (0.0-1.0).
    pub learning_rate: f32,
    /// Maximum workload size (0 = unlimited).
    pub max_workload_size: usize,
    /// Minimum adaptive threshold (prevents going too low).
    pub min_adaptive_threshold: usize,
    /// Maximum adaptive threshold (prevents going too high).
    pub max_adaptive_threshold: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            mode: ProcessingMode::default(),
            cpu_threads: 0,
            gpu_available: false,
            learning_rate: 0.1,
            max_workload_size: 0,
            min_adaptive_threshold: 1_000,
            max_adaptive_threshold: 1_000_000,
        }
    }
}

impl HybridConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a CPU-only configuration.
    #[must_use]
    pub fn cpu_only() -> Self {
        Self {
            mode: ProcessingMode::CpuOnly,
            gpu_available: false,
            ..Default::default()
        }
    }

    /// Creates a GPU-only configuration.
    #[must_use]
    pub fn gpu_only() -> Self {
        Self {
            mode: ProcessingMode::GpuOnly,
            gpu_available: true,
            ..Default::default()
        }
    }

    /// Creates an adaptive configuration that learns optimal thresholds.
    #[must_use]
    pub fn adaptive() -> Self {
        Self {
            mode: ProcessingMode::Adaptive,
            gpu_available: true,
            learning_rate: 0.1,
            ..Default::default()
        }
    }

    /// Creates a configuration for small workloads (low threshold).
    #[must_use]
    pub fn for_small_workloads() -> Self {
        Self {
            mode: ProcessingMode::Hybrid {
                gpu_threshold: 1_000,
            },
            gpu_available: true,
            ..Default::default()
        }
    }

    /// Creates a configuration for large workloads (high threshold).
    #[must_use]
    pub fn for_large_workloads() -> Self {
        Self {
            mode: ProcessingMode::Hybrid {
                gpu_threshold: 100_000,
            },
            gpu_available: true,
            ..Default::default()
        }
    }

    /// Returns a builder for custom configuration.
    #[must_use]
    pub fn builder() -> HybridConfigBuilder {
        HybridConfigBuilder::new()
    }
}

/// Builder for `HybridConfig`.
#[derive(Debug, Clone)]
pub struct HybridConfigBuilder {
    config: HybridConfig,
}

impl HybridConfigBuilder {
    /// Creates a new builder with default values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: HybridConfig::default(),
        }
    }

    /// Sets the processing mode.
    #[must_use]
    pub fn mode(mut self, mode: ProcessingMode) -> Self {
        self.config.mode = mode;
        self
    }

    /// Sets whether GPU is available.
    #[must_use]
    pub fn gpu_available(mut self, available: bool) -> Self {
        self.config.gpu_available = available;
        self
    }

    /// Sets the number of CPU threads (0 = auto).
    #[must_use]
    pub fn cpu_threads(mut self, threads: usize) -> Self {
        self.config.cpu_threads = threads;
        self
    }

    /// Sets the adaptive learning rate.
    #[must_use]
    pub fn learning_rate(mut self, rate: f32) -> Self {
        self.config.learning_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Sets the maximum workload size.
    #[must_use]
    pub fn max_workload_size(mut self, size: usize) -> Self {
        self.config.max_workload_size = size;
        self
    }

    /// Sets the minimum adaptive threshold.
    #[must_use]
    pub fn min_adaptive_threshold(mut self, threshold: usize) -> Self {
        self.config.min_adaptive_threshold = threshold;
        self
    }

    /// Sets the maximum adaptive threshold.
    #[must_use]
    pub fn max_adaptive_threshold(mut self, threshold: usize) -> Self {
        self.config.max_adaptive_threshold = threshold;
        self
    }

    /// Builds the configuration.
    #[must_use]
    pub fn build(self) -> HybridConfig {
        self.config
    }
}

impl Default for HybridConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processing_mode_default() {
        let mode = ProcessingMode::default();
        assert!(matches!(
            mode,
            ProcessingMode::Hybrid {
                gpu_threshold: 10_000
            }
        ));
    }

    #[test]
    fn test_processing_mode_hybrid() {
        let mode = ProcessingMode::hybrid(5000);
        assert_eq!(mode.threshold(), Some(5000));
    }

    #[test]
    fn test_config_cpu_only() {
        let config = HybridConfig::cpu_only();
        assert_eq!(config.mode, ProcessingMode::CpuOnly);
        assert!(!config.gpu_available);
    }

    #[test]
    fn test_config_gpu_only() {
        let config = HybridConfig::gpu_only();
        assert_eq!(config.mode, ProcessingMode::GpuOnly);
        assert!(config.gpu_available);
    }

    #[test]
    fn test_config_builder() {
        let config = HybridConfig::builder()
            .mode(ProcessingMode::Adaptive)
            .gpu_available(true)
            .learning_rate(0.5)
            .max_workload_size(1_000_000)
            .build();

        assert_eq!(config.mode, ProcessingMode::Adaptive);
        assert!(config.gpu_available);
        assert!((config.learning_rate - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.max_workload_size, 1_000_000);
    }

    #[test]
    fn test_learning_rate_clamping() {
        let config = HybridConfig::builder().learning_rate(2.0).build();
        assert!((config.learning_rate - 1.0).abs() < f32::EPSILON);

        let config = HybridConfig::builder().learning_rate(-0.5).build();
        assert!(config.learning_rate.abs() < f32::EPSILON);
    }
}
