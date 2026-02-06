//! Kernel-to-Kernel (K2K) messaging Python bindings.
//!
//! Direct messaging between GPU kernels without host intervention.

use pyo3::prelude::*;
use pyo3::types::PyType;
use ringkernel_core::k2k::{DeliveryReceipt, DeliveryStatus, K2KBroker, K2KConfig, K2KStats};
use std::sync::Arc;

use crate::core::hlc::PyHlcTimestamp;
use crate::core::message::PyMessageId;

/// Configuration for K2K messaging.
#[pyclass]
#[derive(Clone)]
pub struct PyK2KConfig {
    inner: K2KConfig,
}

#[pymethods]
impl PyK2KConfig {
    /// Create a new K2K configuration.
    #[new]
    #[pyo3(signature = (max_pending_messages=1024, delivery_timeout_ms=5000, enable_tracing=false, max_hops=8))]
    fn new(
        max_pending_messages: usize,
        delivery_timeout_ms: u64,
        enable_tracing: bool,
        max_hops: u8,
    ) -> Self {
        Self {
            inner: K2KConfig {
                max_pending_messages,
                delivery_timeout_ms,
                enable_tracing,
                max_hops,
            },
        }
    }

    /// Create default configuration.
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: K2KConfig::default(),
        }
    }

    #[getter]
    fn max_pending_messages(&self) -> usize {
        self.inner.max_pending_messages
    }

    #[getter]
    fn delivery_timeout_ms(&self) -> u64 {
        self.inner.delivery_timeout_ms
    }

    #[getter]
    fn enable_tracing(&self) -> bool {
        self.inner.enable_tracing
    }

    #[getter]
    fn max_hops(&self) -> u8 {
        self.inner.max_hops
    }

    fn __repr__(&self) -> String {
        format!(
            "K2KConfig(max_pending={}, timeout_ms={}, tracing={}, max_hops={})",
            self.inner.max_pending_messages,
            self.inner.delivery_timeout_ms,
            self.inner.enable_tracing,
            self.inner.max_hops
        )
    }
}

impl From<K2KConfig> for PyK2KConfig {
    fn from(inner: K2KConfig) -> Self {
        Self { inner }
    }
}

impl From<PyK2KConfig> for K2KConfig {
    fn from(py_config: PyK2KConfig) -> Self {
        py_config.inner
    }
}

/// Message delivery status.
#[pyclass(frozen, eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyDeliveryStatus {
    /// Message successfully delivered.
    Delivered = 0,
    /// Message pending delivery.
    Pending = 1,
    /// Destination not found.
    NotFound = 2,
    /// Destination queue is full.
    QueueFull = 3,
    /// Delivery timed out.
    Timeout = 4,
    /// Maximum routing hops exceeded.
    MaxHopsExceeded = 5,
}

#[pymethods]
impl PyDeliveryStatus {
    /// Check if delivery was successful.
    fn is_success(&self) -> bool {
        matches!(self, Self::Delivered)
    }

    /// Check if message is still pending.
    fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }

    /// Check if delivery failed.
    fn is_failure(&self) -> bool {
        matches!(
            self,
            Self::NotFound | Self::QueueFull | Self::Timeout | Self::MaxHopsExceeded
        )
    }

    fn __repr__(&self) -> String {
        match self {
            Self::Delivered => "DeliveryStatus.Delivered".to_string(),
            Self::Pending => "DeliveryStatus.Pending".to_string(),
            Self::NotFound => "DeliveryStatus.NotFound".to_string(),
            Self::QueueFull => "DeliveryStatus.QueueFull".to_string(),
            Self::Timeout => "DeliveryStatus.Timeout".to_string(),
            Self::MaxHopsExceeded => "DeliveryStatus.MaxHopsExceeded".to_string(),
        }
    }
}

impl From<DeliveryStatus> for PyDeliveryStatus {
    fn from(status: DeliveryStatus) -> Self {
        match status {
            DeliveryStatus::Delivered => Self::Delivered,
            DeliveryStatus::Pending => Self::Pending,
            DeliveryStatus::NotFound => Self::NotFound,
            DeliveryStatus::QueueFull => Self::QueueFull,
            DeliveryStatus::Timeout => Self::Timeout,
            DeliveryStatus::MaxHopsExceeded => Self::MaxHopsExceeded,
        }
    }
}

/// Receipt for a sent message.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyDeliveryReceipt {
    inner: DeliveryReceipt,
}

#[pymethods]
impl PyDeliveryReceipt {
    /// Message ID.
    #[getter]
    fn message_id(&self) -> PyMessageId {
        self.inner.message_id.into()
    }

    /// Source kernel ID.
    #[getter]
    fn source(&self) -> String {
        self.inner.source.as_str().to_string()
    }

    /// Destination kernel ID.
    #[getter]
    fn destination(&self) -> String {
        self.inner.destination.as_str().to_string()
    }

    /// Delivery status.
    #[getter]
    fn status(&self) -> PyDeliveryStatus {
        self.inner.status.into()
    }

    /// Timestamp when status was recorded.
    #[getter]
    fn timestamp(&self) -> PyHlcTimestamp {
        self.inner.timestamp.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "DeliveryReceipt(msg={}, status={:?}, dst={})",
            self.inner.message_id, self.inner.status, self.inner.destination
        )
    }
}

impl From<DeliveryReceipt> for PyDeliveryReceipt {
    fn from(inner: DeliveryReceipt) -> Self {
        Self { inner }
    }
}

/// K2K messaging statistics.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyK2KStats {
    inner: K2KStats,
}

#[pymethods]
impl PyK2KStats {
    /// Number of registered endpoints.
    #[getter]
    fn registered_endpoints(&self) -> usize {
        self.inner.registered_endpoints
    }

    /// Total messages delivered.
    #[getter]
    fn messages_delivered(&self) -> u64 {
        self.inner.messages_delivered
    }

    /// Number of configured routes.
    #[getter]
    fn routes_configured(&self) -> usize {
        self.inner.routes_configured
    }

    fn __repr__(&self) -> String {
        format!(
            "K2KStats(endpoints={}, delivered={}, routes={})",
            self.inner.registered_endpoints,
            self.inner.messages_delivered,
            self.inner.routes_configured
        )
    }
}

impl From<K2KStats> for PyK2KStats {
    fn from(inner: K2KStats) -> Self {
        Self { inner }
    }
}

/// Central broker for K2K messaging.
#[pyclass]
pub struct PyK2KBroker {
    pub(crate) inner: Arc<K2KBroker>,
}

impl PyK2KBroker {
    /// Create from an existing Arc<K2KBroker>.
    pub(crate) fn from_arc(inner: Arc<K2KBroker>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyK2KBroker {
    /// Create a new K2K broker.
    #[new]
    fn new(config: &PyK2KConfig) -> Self {
        Self {
            inner: K2KBroker::new(config.inner.clone()),
        }
    }

    /// Check if a kernel is registered.
    fn is_registered(&self, kernel_id: &str) -> bool {
        use ringkernel_core::runtime::KernelId;
        self.inner.is_registered(&KernelId::new(kernel_id))
    }

    /// Get list of all registered kernel IDs.
    fn registered_kernels(&self) -> Vec<String> {
        self.inner
            .registered_kernels()
            .into_iter()
            .map(|k| k.as_str().to_string())
            .collect()
    }

    /// Add a routing entry.
    fn add_route(&self, destination: &str, next_hop: &str) {
        use ringkernel_core::runtime::KernelId;
        self.inner
            .add_route(KernelId::new(destination), KernelId::new(next_hop));
    }

    /// Remove a routing entry.
    fn remove_route(&self, destination: &str) {
        use ringkernel_core::runtime::KernelId;
        self.inner.remove_route(&KernelId::new(destination));
    }

    /// Get messaging statistics.
    fn stats(&self) -> PyK2KStats {
        self.inner.stats().into()
    }

    fn __repr__(&self) -> String {
        let stats = self.inner.stats();
        format!(
            "K2KBroker(endpoints={}, delivered={})",
            stats.registered_endpoints, stats.messages_delivered
        )
    }
}

/// Register K2K types with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create k2k submodule
    let k2k = PyModule::new(m.py(), "k2k")?;
    k2k.add_class::<PyK2KConfig>()?;
    k2k.add_class::<PyDeliveryStatus>()?;
    k2k.add_class::<PyDeliveryReceipt>()?;
    k2k.add_class::<PyK2KStats>()?;
    k2k.add_class::<PyK2KBroker>()?;

    // Add to parent module
    m.add_submodule(&k2k)?;

    // Also add commonly-used types at top level
    m.add_class::<PyK2KBroker>()?;
    m.add_class::<PyK2KConfig>()?;

    Ok(())
}
