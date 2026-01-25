//! Message types for RingKernel Python bindings.
//!
//! Provides message identifiers, envelopes, and priority handling.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};
use ringkernel_core::message::{
    CorrelationId, MessageEnvelope, MessageHeader, MessageId, Priority,
};

use crate::core::hlc::PyHlcTimestamp;
use crate::error::{ErrorKind, PyRingKernelError};

/// A unique identifier for a message.
///
/// Message IDs are 64-bit values that uniquely identify each message
/// in the system. They can be auto-generated or explicitly set.
///
/// Example:
///     >>> msg_id = MessageId.generate()  # Auto-generate
///     >>> msg_id = MessageId(12345)  # Explicit value
///     >>> print(msg_id)  # msg:0000000000003039
#[pyclass(frozen, eq, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyMessageId {
    inner: MessageId,
}

#[pymethods]
impl PyMessageId {
    /// Create a message ID from an explicit value.
    ///
    /// Args:
    ///     id: The 64-bit message ID.
    #[new]
    fn new(id: u64) -> Self {
        Self {
            inner: MessageId::new(id),
        }
    }

    /// Generate a unique message ID.
    ///
    /// Uses atomic incrementing to ensure uniqueness.
    ///
    /// Returns:
    ///     A new unique MessageId.
    #[classmethod]
    fn generate(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: MessageId::generate(),
        }
    }

    /// Get the raw 64-bit value.
    #[getter]
    fn value(&self) -> u64 {
        self.inner.inner()
    }

    fn __repr__(&self) -> String {
        format!("MessageId({})", self.inner.inner())
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __int__(&self) -> u64 {
        self.inner.inner()
    }
}

impl From<MessageId> for PyMessageId {
    fn from(inner: MessageId) -> Self {
        Self { inner }
    }
}

impl From<PyMessageId> for MessageId {
    fn from(py_id: PyMessageId) -> Self {
        py_id.inner
    }
}

/// A correlation ID for request-response tracking.
///
/// Correlation IDs link responses back to their original requests.
/// They can be None to indicate no correlation.
///
/// Example:
///     >>> corr = CorrelationId.generate()
///     >>> corr = CorrelationId.none()  # No correlation
///     >>> if corr.is_some():
///     ...     print(f"Correlation: {corr.value}")
#[pyclass(frozen, eq, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyCorrelationId {
    inner: CorrelationId,
}

#[pymethods]
impl PyCorrelationId {
    /// Create a correlation ID from an explicit value.
    ///
    /// Args:
    ///     id: The 64-bit correlation ID.
    #[new]
    fn new(id: u64) -> Self {
        Self {
            inner: CorrelationId::new(id),
        }
    }

    /// Generate a unique correlation ID.
    ///
    /// Returns:
    ///     A new unique CorrelationId.
    #[classmethod]
    fn generate(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: CorrelationId::generate(),
        }
    }

    /// Create a None correlation ID.
    ///
    /// Returns:
    ///     A correlation ID indicating no correlation.
    #[classmethod]
    fn none(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: CorrelationId::none(),
        }
    }

    /// Check if this correlation ID has a value.
    ///
    /// Returns:
    ///     True if this is not a None correlation ID.
    fn is_some(&self) -> bool {
        self.inner.is_some()
    }

    /// Get the raw 64-bit value.
    ///
    /// Returns None if this is a None correlation ID.
    #[getter]
    fn value(&self) -> Option<u64> {
        if self.inner.is_some() {
            Some(self.inner.0)
        } else {
            None
        }
    }

    fn __repr__(&self) -> String {
        if self.inner.is_some() {
            format!("CorrelationId({})", self.inner.0)
        } else {
            "CorrelationId.none()".to_string()
        }
    }

    fn __bool__(&self) -> bool {
        self.inner.is_some()
    }
}

impl From<CorrelationId> for PyCorrelationId {
    fn from(inner: CorrelationId) -> Self {
        Self { inner }
    }
}

impl From<PyCorrelationId> for CorrelationId {
    fn from(py_id: PyCorrelationId) -> Self {
        py_id.inner
    }
}

/// Message priority levels.
///
/// Higher priority messages are processed before lower priority ones.
///
/// Priority levels:
///     - LOW (0): Background/batch operations
///     - NORMAL (1): Default priority
///     - HIGH (2): Time-sensitive operations
///     - CRITICAL (3): Must be processed immediately
#[pyclass(frozen, eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyPriority {
    /// Background/batch operations (priority 0).
    Low = 0,
    /// Default priority (priority 1).
    Normal = 1,
    /// Time-sensitive operations (priority 2).
    High = 2,
    /// Must be processed immediately (priority 3).
    Critical = 3,
}

#[pymethods]
impl PyPriority {
    /// Create a priority from a numeric value.
    ///
    /// Args:
    ///     value: Priority level (0-3).
    ///
    /// Returns:
    ///     The corresponding Priority enum value.
    ///
    /// Raises:
    ///     ValueError: If value is not 0-3.
    #[classmethod]
    fn from_int(_cls: &Bound<'_, PyType>, value: u8) -> PyResult<Self> {
        match value {
            0 => Ok(Self::Low),
            1 => Ok(Self::Normal),
            2 => Ok(Self::High),
            3 => Ok(Self::Critical),
            _ => Err(PyRingKernelError::new(
                ErrorKind::InvalidArgument,
                format!("Invalid priority value: {} (must be 0-3)", value),
            )
            .into_py_err()),
        }
    }

    /// Get the numeric value of this priority.
    #[getter]
    fn value(&self) -> u8 {
        *self as u8
    }

    fn __repr__(&self) -> String {
        match self {
            Self::Low => "Priority.Low".to_string(),
            Self::Normal => "Priority.Normal".to_string(),
            Self::High => "Priority.High".to_string(),
            Self::Critical => "Priority.Critical".to_string(),
        }
    }

    fn __int__(&self) -> u8 {
        *self as u8
    }
}

impl From<Priority> for PyPriority {
    fn from(p: Priority) -> Self {
        match p {
            Priority::Low => Self::Low,
            Priority::Normal => Self::Normal,
            Priority::High => Self::High,
            Priority::Critical => Self::Critical,
        }
    }
}

impl From<PyPriority> for Priority {
    fn from(p: PyPriority) -> Self {
        match p {
            PyPriority::Low => Self::Low,
            PyPriority::Normal => Self::Normal,
            PyPriority::High => Self::High,
            PyPriority::Critical => Self::Critical,
        }
    }
}

/// A message header containing metadata.
///
/// Headers are 256 bytes and contain routing information, timestamps,
/// and validation data. They are cache-line aligned for GPU efficiency.
///
/// Example:
///     >>> header = MessageHeader.new(
///     ...     type_id=1,
///     ...     source_kernel=1000,
///     ...     dest_kernel=1001,
///     ...     payload_size=64,
///     ...     timestamp=HlcTimestamp.now(node_id=1)
///     ... )
#[pyclass]
#[derive(Clone)]
pub struct PyMessageHeader {
    inner: MessageHeader,
}

#[pymethods]
impl PyMessageHeader {
    /// Create a new message header.
    ///
    /// Args:
    ///     type_id: Message type discriminator.
    ///     source_kernel: Source kernel ID.
    ///     dest_kernel: Destination kernel ID.
    ///     payload_size: Size of the payload in bytes.
    ///     timestamp: HLC timestamp for the message.
    ///
    /// Returns:
    ///     A new MessageHeader.
    #[classmethod]
    fn new(
        _cls: &Bound<'_, PyType>,
        type_id: u64,
        source_kernel: u64,
        dest_kernel: u64,
        payload_size: u64,
        timestamp: &PyHlcTimestamp,
    ) -> Self {
        Self {
            inner: MessageHeader::new(
                type_id,
                source_kernel,
                dest_kernel,
                payload_size as usize,
                timestamp.clone().into(),
            ),
        }
    }

    /// Magic number for validation.
    #[getter]
    fn magic(&self) -> u64 {
        self.inner.magic
    }

    /// Message ID.
    #[getter]
    fn message_id(&self) -> PyMessageId {
        self.inner.message_id.into()
    }

    /// Correlation ID for request-response tracking.
    #[getter]
    fn correlation_id(&self) -> PyCorrelationId {
        self.inner.correlation_id.into()
    }

    /// Source kernel ID.
    #[getter]
    fn source_kernel(&self) -> u64 {
        self.inner.source_kernel
    }

    /// Destination kernel ID.
    #[getter]
    fn dest_kernel(&self) -> u64 {
        self.inner.dest_kernel
    }

    /// Message type discriminator.
    #[getter]
    fn message_type(&self) -> u64 {
        self.inner.message_type
    }

    /// Message priority.
    #[getter]
    fn priority(&self) -> PyPriority {
        match self.inner.priority {
            0 => PyPriority::Low,
            1 => PyPriority::Normal,
            2 => PyPriority::High,
            3 => PyPriority::Critical,
            _ => PyPriority::Normal, // Default for invalid values
        }
    }

    /// Payload size in bytes.
    #[getter]
    fn payload_size(&self) -> u64 {
        self.inner.payload_size
    }

    /// HLC timestamp.
    #[getter]
    fn timestamp(&self) -> PyHlcTimestamp {
        self.inner.timestamp.into()
    }

    /// Validate the header.
    ///
    /// Checks magic number, version, and payload size.
    ///
    /// Returns:
    ///     True if the header is valid.
    fn validate(&self) -> bool {
        self.inner.validate()
    }

    /// Set correlation ID.
    fn with_correlation(&mut self, correlation_id: &PyCorrelationId) {
        self.inner = self.inner.with_correlation(correlation_id.inner);
    }

    /// Set priority.
    fn with_priority(&mut self, priority: PyPriority) {
        self.inner = self.inner.with_priority(priority.into());
    }

    fn __repr__(&self) -> String {
        format!(
            "MessageHeader(type={}, src={}, dst={}, size={})",
            self.inner.message_type,
            self.inner.source_kernel,
            self.inner.dest_kernel,
            self.inner.payload_size
        )
    }
}

impl From<MessageHeader> for PyMessageHeader {
    fn from(inner: MessageHeader) -> Self {
        Self { inner }
    }
}

impl From<PyMessageHeader> for MessageHeader {
    fn from(py_header: PyMessageHeader) -> Self {
        py_header.inner
    }
}

/// A message envelope containing header and payload.
///
/// Envelopes wrap message data for transport between kernels.
/// The header contains routing/metadata, payload contains the actual data.
///
/// Example:
///     >>> envelope = MessageEnvelope.from_bytes(header, payload_bytes)
///     >>> header = envelope.header
///     >>> data = envelope.payload
#[pyclass]
#[derive(Clone)]
pub struct PyMessageEnvelope {
    inner: MessageEnvelope,
}

#[pymethods]
impl PyMessageEnvelope {
    /// Create an envelope from header and payload.
    ///
    /// Args:
    ///     header: Message header.
    ///     payload: Raw payload bytes.
    ///
    /// Returns:
    ///     A new MessageEnvelope.
    #[classmethod]
    fn from_bytes(
        _cls: &Bound<'_, PyType>,
        header: &PyMessageHeader,
        payload: &Bound<'_, PyBytes>,
    ) -> Self {
        Self {
            inner: MessageEnvelope {
                header: header.inner,
                payload: payload.as_bytes().to_vec(),
            },
        }
    }

    /// Create an empty envelope for testing.
    ///
    /// Args:
    ///     source_kernel: Source kernel ID.
    ///     dest_kernel: Destination kernel ID.
    ///     timestamp: HLC timestamp.
    ///
    /// Returns:
    ///     An empty MessageEnvelope.
    #[classmethod]
    fn empty(
        _cls: &Bound<'_, PyType>,
        source_kernel: u64,
        dest_kernel: u64,
        timestamp: &PyHlcTimestamp,
    ) -> Self {
        Self {
            inner: MessageEnvelope::empty(source_kernel, dest_kernel, timestamp.clone().into()),
        }
    }

    /// Get the message header.
    #[getter]
    fn header(&self) -> PyMessageHeader {
        self.inner.header.into()
    }

    /// Get the payload bytes.
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.payload)
    }

    /// Total size in bytes (header + payload).
    #[getter]
    fn total_size(&self) -> usize {
        self.inner.total_size()
    }

    /// Serialize the envelope to bytes.
    ///
    /// Returns:
    ///     The envelope as a contiguous byte buffer.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.to_bytes())
    }

    /// Deserialize an envelope from bytes.
    ///
    /// Args:
    ///     data: Serialized envelope bytes.
    ///
    /// Returns:
    ///     The deserialized MessageEnvelope.
    ///
    /// Raises:
    ///     RingKernelError: If deserialization fails.
    #[classmethod]
    fn from_raw_bytes(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        MessageEnvelope::from_bytes(data.as_bytes())
            .map(|inner| Self { inner })
            .map_err(|e| PyRingKernelError::from(e).into_py_err())
    }

    fn __repr__(&self) -> String {
        format!(
            "MessageEnvelope(type={}, size={})",
            self.inner.header.message_type,
            self.inner.total_size()
        )
    }
}

impl From<MessageEnvelope> for PyMessageEnvelope {
    fn from(inner: MessageEnvelope) -> Self {
        Self { inner }
    }
}

impl From<PyMessageEnvelope> for MessageEnvelope {
    fn from(py_env: PyMessageEnvelope) -> Self {
        py_env.inner
    }
}

/// Maximum payload size in bytes.
pub const MAX_PAYLOAD_SIZE: usize = MessageHeader::MAX_PAYLOAD_SIZE;

/// Register message types with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes
    m.add_class::<PyMessageId>()?;
    m.add_class::<PyCorrelationId>()?;
    m.add_class::<PyPriority>()?;
    m.add_class::<PyMessageHeader>()?;
    m.add_class::<PyMessageEnvelope>()?;

    // Add constants
    m.add("MAX_PAYLOAD_SIZE", MAX_PAYLOAD_SIZE)?;
    m.add("MESSAGE_MAGIC", MessageHeader::MAGIC)?;
    m.add("MESSAGE_VERSION", MessageHeader::VERSION)?;

    Ok(())
}
