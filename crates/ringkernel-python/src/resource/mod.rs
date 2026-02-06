//! Resource management Python bindings.
//!
//! Provides memory estimation and enforcement for GPU workloads.

use pyo3::prelude::*;
use pyo3::types::PyType;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::error::{ErrorKind, PyRingKernelError};

/// Default maximum memory (4 GB).
const DEFAULT_MAX_MEMORY: u64 = 4 * 1024 * 1024 * 1024;

/// System memory safety margin (1 GB).
const SYSTEM_MARGIN: u64 = 1024 * 1024 * 1024;

/// Memory estimate for a workload.
///
/// Describes expected memory usage with confidence level.
///
/// Example:
///     >>> estimate = MemoryEstimate(
///     ...     primary_bytes=1_000_000,
///     ...     auxiliary_bytes=500_000,
///     ...     peak_bytes=1_800_000,
///     ...     confidence=0.9
///     ... )
#[pyclass]
#[derive(Clone)]
pub struct PyMemoryEstimate {
    /// Primary allocation (main data structures).
    #[pyo3(get, set)]
    primary_bytes: u64,
    /// Auxiliary allocations (temporaries, buffers).
    #[pyo3(get, set)]
    auxiliary_bytes: u64,
    /// Peak memory during operation.
    #[pyo3(get, set)]
    peak_bytes: u64,
    /// Confidence in estimate (0.0-1.0).
    #[pyo3(get, set)]
    confidence: f32,
}

#[pymethods]
impl PyMemoryEstimate {
    /// Create a new memory estimate.
    #[new]
    #[pyo3(signature = (primary_bytes=0, auxiliary_bytes=0, peak_bytes=0, confidence=1.0))]
    fn new(primary_bytes: u64, auxiliary_bytes: u64, peak_bytes: u64, confidence: f32) -> Self {
        Self {
            primary_bytes,
            auxiliary_bytes,
            peak_bytes: if peak_bytes > 0 {
                peak_bytes
            } else {
                primary_bytes + auxiliary_bytes
            },
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Total steady-state memory (primary + auxiliary).
    fn total_bytes(&self) -> u64 {
        self.primary_bytes + self.auxiliary_bytes
    }

    /// Create with only primary bytes.
    #[classmethod]
    fn with_primary(_cls: &Bound<'_, PyType>, bytes: u64) -> Self {
        Self::new(bytes, 0, bytes, 1.0)
    }

    /// Create for a data structure with element count and size.
    #[classmethod]
    fn for_elements(_cls: &Bound<'_, PyType>, count: usize, bytes_per_element: usize) -> Self {
        let bytes = (count * bytes_per_element) as u64;
        Self::new(bytes, 0, bytes, 1.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "MemoryEstimate(primary={}B, aux={}B, peak={}B, conf={:.0}%)",
            self.primary_bytes,
            self.auxiliary_bytes,
            self.peak_bytes,
            self.confidence * 100.0
        )
    }
}

/// A reservation guard for guaranteed memory allocation.
///
/// The reservation is released when the guard is dropped or committed.
///
/// Example:
///     >>> with guard.reserve(1_000_000) as reservation:
///     ...     # Memory is reserved
///     ...     do_work()
///     ...     reservation.commit()  # Convert to permanent allocation
///     # Released if not committed
#[pyclass]
pub struct PyReservationGuard {
    guard: Arc<PyResourceGuard>,
    bytes: u64,
    committed: AtomicBool,
    released: AtomicBool,
}

#[pymethods]
impl PyReservationGuard {
    /// Reserved bytes.
    #[getter]
    fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Check if reservation has been committed.
    fn is_committed(&self) -> bool {
        self.committed.load(Ordering::Relaxed)
    }

    /// Convert reservation to permanent allocation.
    ///
    /// After committing, the memory remains allocated and won't be
    /// released when the guard is dropped.
    fn commit(&self) {
        if !self.committed.swap(true, Ordering::Relaxed) {
            // Move from reserved to allocated
            self.guard.reserved.fetch_sub(self.bytes, Ordering::Relaxed);
            self.guard.current.fetch_add(self.bytes, Ordering::Relaxed);
        }
    }

    /// Explicitly release the reservation.
    fn release(&self) {
        if !self.released.swap(true, Ordering::Relaxed) && !self.committed.load(Ordering::Relaxed) {
            self.guard.reserved.fetch_sub(self.bytes, Ordering::Relaxed);
        }
    }

    fn __enter__(self_: PyRef<'_, Self>) -> PyRef<'_, Self> {
        self_
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        self.release();
        false // Don't suppress exceptions
    }

    fn __repr__(&self) -> String {
        format!(
            "ReservationGuard(bytes={}, committed={})",
            self.bytes,
            self.committed.load(Ordering::Relaxed)
        )
    }
}

impl Drop for PyReservationGuard {
    fn drop(&mut self) {
        if !self.released.load(Ordering::Relaxed) && !self.committed.load(Ordering::Relaxed) {
            self.guard.reserved.fetch_sub(self.bytes, Ordering::Relaxed);
        }
    }
}

/// Resource guard for memory limit enforcement.
///
/// Tracks memory usage and prevents over-allocation.
///
/// Example:
///     >>> guard = ResourceGuard(max_memory_bytes=4_000_000_000)
///     >>> if guard.can_allocate(1_000_000):
///     ...     # Safe to allocate
///     ...     pass
///     >>> guard.record_allocation(1_000_000)
///     >>> guard.record_deallocation(1_000_000)
#[pyclass]
pub struct PyResourceGuard {
    max_memory: AtomicU64,
    current: AtomicU64,
    reserved: AtomicU64,
    safety_margin: f64,
    enforce_limits: AtomicBool,
}

#[pymethods]
impl PyResourceGuard {
    /// Create a new resource guard.
    ///
    /// Args:
    ///     max_memory_bytes: Maximum allowed memory (default: 4GB).
    ///     safety_margin: Fraction of max to keep free (0.0-0.9, default: 0.1).
    #[new]
    #[pyo3(signature = (max_memory_bytes=None, safety_margin=0.1))]
    fn new(max_memory_bytes: Option<u64>, safety_margin: f64) -> Self {
        Self {
            max_memory: AtomicU64::new(max_memory_bytes.unwrap_or(DEFAULT_MAX_MEMORY)),
            current: AtomicU64::new(0),
            reserved: AtomicU64::new(0),
            safety_margin: safety_margin.clamp(0.0, 0.9),
            enforce_limits: AtomicBool::new(true),
        }
    }

    /// Create an unguarded instance (no limits enforced).
    ///
    /// Useful for testing or when limits are managed externally.
    #[classmethod]
    fn unguarded(_cls: &Bound<'_, PyType>) -> Self {
        let guard = Self::new(None, 0.0);
        guard.enforce_limits.store(false, Ordering::Relaxed);
        guard
    }

    /// Global default guard (singleton-like access).
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(None, 0.1)
    }

    /// Maximum memory limit.
    #[getter]
    fn max_memory(&self) -> u64 {
        self.max_memory.load(Ordering::Relaxed)
    }

    /// Set maximum memory limit.
    #[setter]
    fn set_max_memory(&self, bytes: u64) {
        self.max_memory.store(bytes, Ordering::Relaxed);
    }

    /// Current tracked memory usage.
    #[getter]
    fn current_memory(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }

    /// Currently reserved memory.
    #[getter]
    fn reserved_memory(&self) -> u64 {
        self.reserved.load(Ordering::Relaxed)
    }

    /// Available memory (max - current - reserved - margin).
    fn available_memory(&self) -> u64 {
        let max = self.max_memory.load(Ordering::Relaxed);
        let current = self.current.load(Ordering::Relaxed);
        let reserved = self.reserved.load(Ordering::Relaxed);
        let margin = (max as f64 * self.safety_margin) as u64;

        max.saturating_sub(current)
            .saturating_sub(reserved)
            .saturating_sub(margin)
    }

    /// Check if allocation is possible.
    ///
    /// Args:
    ///     bytes: Number of bytes to allocate.
    ///
    /// Returns:
    ///     True if allocation would succeed.
    fn can_allocate(&self, bytes: u64) -> bool {
        if !self.enforce_limits.load(Ordering::Relaxed) {
            return true;
        }
        bytes <= self.available_memory()
    }

    /// Record an allocation.
    ///
    /// Args:
    ///     bytes: Number of bytes allocated.
    fn record_allocation(&self, bytes: u64) {
        self.current.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record a deallocation.
    ///
    /// Args:
    ///     bytes: Number of bytes freed.
    fn record_deallocation(&self, bytes: u64) {
        self.current.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Reserve memory for future allocation.
    ///
    /// Args:
    ///     bytes: Number of bytes to reserve.
    ///
    /// Returns:
    ///     ReservationGuard on success.
    ///
    /// Raises:
    ///     ReservationError: If reservation would exceed limits.
    fn reserve(self_: PyRef<'_, Self>, bytes: u64) -> PyResult<PyReservationGuard> {
        if !self_.can_allocate(bytes) {
            return Err(PyRingKernelError::new(
                ErrorKind::ReservationFailed,
                format!(
                    "Cannot reserve {} bytes (available: {})",
                    bytes,
                    self_.available_memory()
                ),
            )
            .into_py_err());
        }

        self_.reserved.fetch_add(bytes, Ordering::Relaxed);

        // Get Arc<Self> from PyRef
        let guard = Arc::new(PyResourceGuard {
            max_memory: AtomicU64::new(self_.max_memory.load(Ordering::Relaxed)),
            current: AtomicU64::new(self_.current.load(Ordering::Relaxed)),
            reserved: AtomicU64::new(self_.reserved.load(Ordering::Relaxed)),
            safety_margin: self_.safety_margin,
            enforce_limits: AtomicBool::new(self_.enforce_limits.load(Ordering::Relaxed)),
        });

        Ok(PyReservationGuard {
            guard,
            bytes,
            committed: AtomicBool::new(false),
            released: AtomicBool::new(false),
        })
    }

    /// Validate a memory estimate.
    ///
    /// Args:
    ///     estimate: Memory estimate to validate.
    ///
    /// Raises:
    ///     MemoryLimitError: If estimate exceeds available memory.
    fn validate(&self, estimate: &PyMemoryEstimate) -> PyResult<()> {
        if !self.enforce_limits.load(Ordering::Relaxed) {
            return Ok(());
        }

        let available = self.available_memory();
        if estimate.peak_bytes > available {
            return Err(PyRingKernelError::new(
                ErrorKind::MemoryLimit,
                format!(
                    "Estimate peak {} bytes exceeds available {} bytes",
                    estimate.peak_bytes, available
                ),
            )
            .into_py_err());
        }
        Ok(())
    }

    /// Calculate maximum safe element count.
    ///
    /// Args:
    ///     bytes_per_element: Size of each element.
    ///
    /// Returns:
    ///     Maximum number of elements that can be safely allocated.
    fn max_safe_elements(&self, bytes_per_element: usize) -> usize {
        if bytes_per_element == 0 {
            return 0;
        }
        (self.available_memory() / bytes_per_element as u64) as usize
    }

    /// Enable or disable limit enforcement.
    fn set_enforce_limits(&self, enforce: bool) {
        self.enforce_limits.store(enforce, Ordering::Relaxed);
    }

    /// Check if limits are being enforced.
    fn is_enforcing(&self) -> bool {
        self.enforce_limits.load(Ordering::Relaxed)
    }

    /// Get utilization as a fraction.
    fn utilization(&self) -> f64 {
        let max = self.max_memory.load(Ordering::Relaxed);
        if max == 0 {
            return 0.0;
        }
        let used = self.current.load(Ordering::Relaxed) + self.reserved.load(Ordering::Relaxed);
        used as f64 / max as f64
    }

    fn __repr__(&self) -> String {
        format!(
            "ResourceGuard(used={}B, max={}B, util={:.1}%)",
            self.current.load(Ordering::Relaxed),
            self.max_memory.load(Ordering::Relaxed),
            self.utilization() * 100.0
        )
    }
}

/// Get total system memory (if available).
///
/// Returns:
///     Total physical memory in bytes, or None if unavailable.
#[pyfunction]
fn get_total_memory() -> Option<u64> {
    // Try to get from /proc/meminfo on Linux
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return Some(kb * 1024);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Get available system memory (if available).
///
/// Returns:
///     Available memory in bytes, or None if unavailable.
#[pyfunction]
fn get_available_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return Some(kb * 1024);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Register resource types with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create resource submodule
    let resource = PyModule::new(m.py(), "resource")?;
    resource.add_class::<PyMemoryEstimate>()?;
    resource.add_class::<PyReservationGuard>()?;
    resource.add_class::<PyResourceGuard>()?;
    resource.add_function(wrap_pyfunction!(get_total_memory, &resource)?)?;
    resource.add_function(wrap_pyfunction!(get_available_memory, &resource)?)?;

    // Add constants
    resource.add("DEFAULT_MAX_MEMORY", DEFAULT_MAX_MEMORY)?;
    resource.add("SYSTEM_MARGIN", SYSTEM_MARGIN)?;

    // Add to parent module
    m.add_submodule(&resource)?;

    // Also add commonly-used types at top level
    m.add_class::<PyResourceGuard>()?;
    m.add_class::<PyMemoryEstimate>()?;

    Ok(())
}
