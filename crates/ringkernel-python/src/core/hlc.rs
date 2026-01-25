//! Hybrid Logical Clock (HLC) Python bindings.
//!
//! Provides causal ordering for distributed kernel messages.

use pyo3::prelude::*;
use pyo3::types::PyType;
use ringkernel_core::hlc::{HlcClock, HlcTimestamp};
use std::sync::Arc;

use crate::error::PyRingKernelError;

/// A Hybrid Logical Clock timestamp for causal ordering.
///
/// HLC timestamps combine wall-clock time with logical counters to provide
/// a total ordering of events even when physical clocks are imperfect.
///
/// Timestamps are immutable and can be compared for ordering.
///
/// Example:
///     >>> ts1 = HlcTimestamp.now(node_id=1)
///     >>> ts2 = HlcTimestamp.now(node_id=1)
///     >>> assert ts2 > ts1  # Later timestamp is greater
#[pyclass(frozen, eq, ord)]
#[derive(Clone, Debug)]
pub struct PyHlcTimestamp {
    inner: HlcTimestamp,
}

#[pymethods]
impl PyHlcTimestamp {
    /// Create a new HLC timestamp with explicit values.
    ///
    /// Args:
    ///     physical: Physical time in microseconds since epoch.
    ///     logical: Logical counter for events at same physical time.
    ///     node_id: Unique identifier for the node/kernel.
    ///
    /// Returns:
    ///     A new HlcTimestamp instance.
    #[new]
    fn new(physical: u64, logical: u64, node_id: u64) -> Self {
        Self {
            inner: HlcTimestamp::new(physical, logical, node_id),
        }
    }

    /// Create a timestamp representing the current wall-clock time.
    ///
    /// Args:
    ///     node_id: Unique identifier for the node/kernel.
    ///
    /// Returns:
    ///     A timestamp with current physical time, logical=0.
    #[classmethod]
    fn now(_cls: &Bound<'_, PyType>, node_id: u64) -> Self {
        Self {
            inner: HlcTimestamp::now(node_id),
        }
    }

    /// Create a zero timestamp (minimum possible value).
    ///
    /// Returns:
    ///     A timestamp with all fields set to zero.
    #[classmethod]
    fn zero(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: HlcTimestamp::zero(),
        }
    }

    /// Physical time component in microseconds since epoch.
    #[getter]
    fn physical(&self) -> u64 {
        self.inner.physical
    }

    /// Logical counter for events at the same physical time.
    #[getter]
    fn logical(&self) -> u64 {
        self.inner.logical
    }

    /// Node identifier that created this timestamp.
    #[getter]
    fn node_id(&self) -> u64 {
        self.inner.node_id
    }

    /// Check if this is a zero timestamp.
    ///
    /// Returns:
    ///     True if all fields are zero.
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Get physical time as microseconds.
    ///
    /// Returns:
    ///     Physical time in microseconds since epoch.
    fn as_micros(&self) -> u64 {
        self.inner.as_micros()
    }

    /// Get physical time as milliseconds.
    ///
    /// Returns:
    ///     Physical time in milliseconds since epoch.
    fn as_millis(&self) -> u64 {
        self.inner.as_millis()
    }

    /// Pack timestamp into a single u128 for atomic comparison.
    ///
    /// Format: physical (64 bits) | logical (48 bits) | node_id (16 bits)
    ///
    /// Returns:
    ///     Packed 128-bit integer.
    fn pack(&self) -> u128 {
        self.inner.pack()
    }

    /// Unpack a u128 back into a timestamp.
    ///
    /// Args:
    ///     packed: A 128-bit integer from pack().
    ///
    /// Returns:
    ///     The unpacked HlcTimestamp.
    #[classmethod]
    fn unpack(_cls: &Bound<'_, PyType>, packed: u128) -> Self {
        Self {
            inner: HlcTimestamp::unpack(packed),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HlcTimestamp(physical={}, logical={}, node_id={})",
            self.inner.physical, self.inner.logical, self.inner.node_id
        )
    }

    fn __str__(&self) -> String {
        format!(
            "HLC({}.{}.{})",
            self.inner.physical, self.inner.logical, self.inner.node_id
        )
    }

    fn __hash__(&self) -> u64 {
        // Use packed representation for hashing
        let packed = self.inner.pack();
        (packed >> 64) as u64 ^ packed as u64
    }
}

impl PartialEq for PyHlcTimestamp {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for PyHlcTimestamp {}

impl PartialOrd for PyHlcTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PyHlcTimestamp {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.inner.cmp(&other.inner)
    }
}

impl From<HlcTimestamp> for PyHlcTimestamp {
    fn from(inner: HlcTimestamp) -> Self {
        Self { inner }
    }
}

impl From<PyHlcTimestamp> for HlcTimestamp {
    fn from(py_ts: PyHlcTimestamp) -> Self {
        py_ts.inner
    }
}

/// A Hybrid Logical Clock for generating causally-ordered timestamps.
///
/// The clock maintains state and ensures timestamps are always strictly
/// increasing, even when the wall clock goes backwards.
///
/// Example:
///     >>> clock = HlcClock(node_id=1)
///     >>> ts1 = clock.tick()  # Get next timestamp
///     >>> ts2 = clock.tick()  # Always greater than ts1
///     >>> assert ts2 > ts1
///
///     >>> # Update from received message
///     >>> received = HlcTimestamp(physical=..., logical=..., node_id=2)
///     >>> merged = clock.update(received)  # Merge causality
#[pyclass]
pub struct PyHlcClock {
    inner: Arc<HlcClock>,
}

#[pymethods]
impl PyHlcClock {
    /// Create a new HLC clock for a node.
    ///
    /// Args:
    ///     node_id: Unique identifier for this node/kernel.
    ///     max_drift_ms: Maximum allowed clock drift in milliseconds (default: 60000).
    ///
    /// Returns:
    ///     A new HlcClock instance.
    #[new]
    #[pyo3(signature = (node_id, max_drift_ms=None))]
    fn new(node_id: u64, max_drift_ms: Option<u64>) -> Self {
        let clock = match max_drift_ms {
            Some(drift) => HlcClock::with_max_drift(node_id, drift),
            None => HlcClock::new(node_id),
        };
        Self {
            inner: Arc::new(clock),
        }
    }

    /// The node ID for this clock.
    #[getter]
    fn node_id(&self) -> u64 {
        self.inner.node_id()
    }

    /// Read the current clock value without advancing.
    ///
    /// Returns:
    ///     The current timestamp.
    fn now(&self) -> PyHlcTimestamp {
        self.inner.now().into()
    }

    /// Advance the clock and return a new timestamp.
    ///
    /// This is guaranteed to return a timestamp strictly greater than
    /// any previously returned timestamp.
    ///
    /// Returns:
    ///     A new, strictly increasing timestamp.
    fn tick(&self) -> PyHlcTimestamp {
        self.inner.tick().into()
    }

    /// Update the clock from a received timestamp.
    ///
    /// This merges the received timestamp's causality with the local
    /// clock, ensuring the returned timestamp is greater than both
    /// the local clock and the received timestamp.
    ///
    /// Args:
    ///     received: A timestamp from another node.
    ///
    /// Returns:
    ///     A new timestamp reflecting merged causality.
    ///
    /// Raises:
    ///     RingKernelError: If clock skew exceeds the maximum drift.
    fn update(&self, received: &PyHlcTimestamp) -> PyResult<PyHlcTimestamp> {
        self.inner
            .update(&received.inner)
            .map(PyHlcTimestamp::from)
            .map_err(|e| PyRingKernelError::from(e).into_py_err())
    }

    fn __repr__(&self) -> String {
        format!("HlcClock(node_id={})", self.inner.node_id())
    }
}

/// Register HLC types with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create hlc submodule
    let hlc = PyModule::new_bound(m.py(), "hlc")?;
    hlc.add_class::<PyHlcTimestamp>()?;
    hlc.add_class::<PyHlcClock>()?;

    // Add to parent module
    m.add_submodule(&hlc)?;

    // Also add at top level for convenience
    m.add_class::<PyHlcTimestamp>()?;
    m.add_class::<PyHlcClock>()?;

    Ok(())
}
