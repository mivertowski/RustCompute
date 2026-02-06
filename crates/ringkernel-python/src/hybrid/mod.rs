//! Hybrid CPU-GPU dispatcher Python bindings.
//!
//! Provides intelligent workload routing between CPU and GPU.

use pyo3::prelude::*;
use pyo3::types::PyType;

/// Processing mode for hybrid dispatch.
#[pyclass(frozen, eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyProcessingMode {
    /// Force CPU-only execution.
    CpuOnly = 0,
    /// Force GPU-only execution.
    GpuOnly = 1,
    /// Use GPU above threshold, CPU below.
    Hybrid = 2,
    /// Learn optimal threshold automatically.
    Adaptive = 3,
}

#[pymethods]
impl PyProcessingMode {
    fn __repr__(&self) -> String {
        match self {
            Self::CpuOnly => "ProcessingMode.CpuOnly".to_string(),
            Self::GpuOnly => "ProcessingMode.GpuOnly".to_string(),
            Self::Hybrid => "ProcessingMode.Hybrid".to_string(),
            Self::Adaptive => "ProcessingMode.Adaptive".to_string(),
        }
    }
}

/// Configuration for hybrid dispatch.
///
/// Example:
///     >>> config = HybridConfig(mode=ProcessingMode.Adaptive)
///     >>> config = HybridConfig.adaptive()  # Preset
#[pyclass]
#[derive(Clone)]
pub struct PyHybridConfig {
    mode: PyProcessingMode,
    gpu_threshold: usize,
    learning_rate: f32,
    gpu_available: bool,
}

#[pymethods]
impl PyHybridConfig {
    /// Create a new hybrid config.
    ///
    /// Args:
    ///     mode: Processing mode.
    ///     gpu_threshold: Minimum size for GPU execution (default: 10000).
    ///     learning_rate: Adaptive learning rate (default: 0.1).
    ///     gpu_available: Whether GPU is available (default: False).
    #[new]
    #[pyo3(signature = (mode=PyProcessingMode::CpuOnly, gpu_threshold=10000, learning_rate=0.1, gpu_available=false))]
    fn new(
        mode: PyProcessingMode,
        gpu_threshold: usize,
        learning_rate: f32,
        gpu_available: bool,
    ) -> Self {
        Self {
            mode,
            gpu_threshold,
            learning_rate,
            gpu_available,
        }
    }

    /// Create CPU-only config.
    #[classmethod]
    fn cpu_only(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(PyProcessingMode::CpuOnly, 10000, 0.1, false)
    }

    /// Create GPU-only config (requires GPU).
    #[classmethod]
    fn gpu_only(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(PyProcessingMode::GpuOnly, 0, 0.1, true)
    }

    /// Create adaptive config (learns optimal threshold).
    #[classmethod]
    fn adaptive(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(PyProcessingMode::Adaptive, 10000, 0.1, true)
    }

    #[getter]
    fn mode(&self) -> PyProcessingMode {
        self.mode
    }

    #[getter]
    fn gpu_threshold(&self) -> usize {
        self.gpu_threshold
    }

    #[getter]
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    #[getter]
    fn gpu_available(&self) -> bool {
        self.gpu_available
    }

    fn __repr__(&self) -> String {
        format!(
            "HybridConfig(mode={:?}, threshold={}, gpu={})",
            self.mode, self.gpu_threshold, self.gpu_available
        )
    }
}

/// Statistics for hybrid dispatch.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyHybridStats {
    /// Number of CPU executions.
    #[pyo3(get)]
    cpu_count: u64,
    /// Number of GPU executions.
    #[pyo3(get)]
    gpu_count: u64,
    /// Current adaptive threshold.
    #[pyo3(get)]
    adaptive_threshold: Option<usize>,
}

#[pymethods]
impl PyHybridStats {
    /// Total execution count.
    fn total(&self) -> u64 {
        self.cpu_count + self.gpu_count
    }

    /// GPU usage ratio.
    fn gpu_ratio(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            0.0
        } else {
            self.gpu_count as f64 / total as f64
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HybridStats(cpu={}, gpu={}, ratio={:.1}%)",
            self.cpu_count,
            self.gpu_count,
            self.gpu_ratio() * 100.0
        )
    }
}

/// Hybrid CPU-GPU dispatcher.
///
/// Routes workloads to CPU or GPU based on size and configuration.
///
/// Example:
///     >>> config = HybridConfig.adaptive()
///     >>> dispatcher = HybridDispatcher(config)
///     >>> should_use_gpu = dispatcher.should_use_gpu(workload_size=50000)
#[pyclass]
pub struct PyHybridDispatcher {
    config: PyHybridConfig,
    cpu_count: std::sync::atomic::AtomicU64,
    gpu_count: std::sync::atomic::AtomicU64,
    adaptive_threshold: std::sync::atomic::AtomicUsize,
}

#[pymethods]
impl PyHybridDispatcher {
    /// Create a new hybrid dispatcher.
    ///
    /// Args:
    ///     config: Hybrid configuration.
    #[new]
    fn new(config: &PyHybridConfig) -> Self {
        Self {
            config: config.clone(),
            cpu_count: std::sync::atomic::AtomicU64::new(0),
            gpu_count: std::sync::atomic::AtomicU64::new(0),
            adaptive_threshold: std::sync::atomic::AtomicUsize::new(config.gpu_threshold),
        }
    }

    /// Create with default config.
    #[classmethod]
    fn with_defaults(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(&PyHybridConfig::cpu_only(_cls))
    }

    /// Check if GPU should be used for given workload size.
    ///
    /// Args:
    ///     workload_size: Size of the workload (e.g., element count).
    ///
    /// Returns:
    ///     True if GPU should be used.
    fn should_use_gpu(&self, workload_size: usize) -> bool {
        match self.config.mode {
            PyProcessingMode::CpuOnly => false,
            PyProcessingMode::GpuOnly => self.config.gpu_available,
            PyProcessingMode::Hybrid | PyProcessingMode::Adaptive => {
                let threshold = self
                    .adaptive_threshold
                    .load(std::sync::atomic::Ordering::Relaxed);
                self.config.gpu_available && workload_size >= threshold
            }
        }
    }

    /// Record a CPU execution.
    fn record_cpu_execution(&self) {
        self.cpu_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a GPU execution.
    fn record_gpu_execution(&self) {
        self.gpu_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Update adaptive threshold based on timing.
    ///
    /// Args:
    ///     workload_size: Size of the workload.
    ///     cpu_time_ms: CPU execution time in milliseconds.
    ///     gpu_time_ms: GPU execution time in milliseconds.
    fn update_adaptive_threshold(&self, workload_size: usize, cpu_time_ms: f64, gpu_time_ms: f64) {
        if self.config.mode != PyProcessingMode::Adaptive {
            return;
        }

        let current = self
            .adaptive_threshold
            .load(std::sync::atomic::Ordering::Relaxed);

        // If GPU was faster, lower threshold; if CPU was faster, raise it
        let new_threshold = if gpu_time_ms < cpu_time_ms {
            // GPU was faster, prefer GPU for smaller workloads
            ((1.0 - self.config.learning_rate) * current as f32
                + self.config.learning_rate * (workload_size as f32 * 0.8)) as usize
        } else {
            // CPU was faster, prefer CPU for larger workloads
            ((1.0 - self.config.learning_rate) * current as f32
                + self.config.learning_rate * (workload_size as f32 * 1.2)) as usize
        };

        self.adaptive_threshold
            .store(new_threshold.max(100), std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current adaptive threshold.
    #[getter]
    fn adaptive_threshold(&self) -> usize {
        self.adaptive_threshold
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get dispatcher statistics.
    fn stats(&self) -> PyHybridStats {
        PyHybridStats {
            cpu_count: self.cpu_count.load(std::sync::atomic::Ordering::Relaxed),
            gpu_count: self.gpu_count.load(std::sync::atomic::Ordering::Relaxed),
            adaptive_threshold: if self.config.mode == PyProcessingMode::Adaptive {
                Some(
                    self.adaptive_threshold
                        .load(std::sync::atomic::Ordering::Relaxed),
                )
            } else {
                None
            },
        }
    }

    /// Get configuration.
    fn config(&self) -> PyHybridConfig {
        self.config.clone()
    }

    fn __repr__(&self) -> String {
        let stats = self.stats();
        format!(
            "HybridDispatcher(mode={:?}, cpu={}, gpu={})",
            self.config.mode, stats.cpu_count, stats.gpu_count
        )
    }
}

/// Register hybrid types with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create hybrid submodule
    let hybrid = PyModule::new(m.py(), "hybrid")?;
    hybrid.add_class::<PyProcessingMode>()?;
    hybrid.add_class::<PyHybridConfig>()?;
    hybrid.add_class::<PyHybridStats>()?;
    hybrid.add_class::<PyHybridDispatcher>()?;

    // Add to parent module
    m.add_submodule(&hybrid)?;

    // Also add commonly-used types at top level
    m.add_class::<PyHybridDispatcher>()?;
    m.add_class::<PyProcessingMode>()?;

    Ok(())
}
