//! RingKernel runtime Python bindings.
//!
//! Provides the main runtime and kernel handle types.

use pyo3::prelude::*;
use pyo3::types::PyType;
use std::sync::Arc;
use std::time::Duration;

use ringkernel_core::runtime::{
    Backend, KernelHandle, KernelId, KernelState, KernelStatus, LaunchOptions, RingKernelRuntime,
    RuntimeMetrics,
};
use ringkernel_core::types::KernelMode;
use ringkernel_cpu::CpuRuntime;

use crate::core::k2k::PyK2KBroker;
use crate::core::message::PyMessageEnvelope;
use crate::error::{ErrorKind, PyRingKernelError};

/// Kernel execution mode.
#[pyclass(frozen, eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyKernelMode {
    /// Persistent kernel that runs until terminated.
    Persistent = 0,
    /// Event-driven kernel that processes messages and exits.
    EventDriven = 1,
}

#[pymethods]
impl PyKernelMode {
    fn __repr__(&self) -> String {
        match self {
            Self::Persistent => "KernelMode.Persistent".to_string(),
            Self::EventDriven => "KernelMode.EventDriven".to_string(),
        }
    }
}

impl From<KernelMode> for PyKernelMode {
    fn from(mode: KernelMode) -> Self {
        match mode {
            KernelMode::Persistent => Self::Persistent,
            KernelMode::EventDriven => Self::EventDriven,
        }
    }
}

impl From<PyKernelMode> for KernelMode {
    fn from(mode: PyKernelMode) -> Self {
        match mode {
            PyKernelMode::Persistent => Self::Persistent,
            PyKernelMode::EventDriven => Self::EventDriven,
        }
    }
}

/// Kernel lifecycle state.
///
/// The kernel transitions through these states during its lifecycle:
///
/// Created → Launched → Active → Terminated
///                ↓       ↑↓
///            Deactivated
#[pyclass(frozen, eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyKernelState {
    /// Kernel is created but not launched.
    Created = 0,
    /// Kernel is launched and initializing.
    Launched = 1,
    /// Kernel is active and processing messages.
    Active = 2,
    /// Kernel is deactivated (paused).
    Deactivated = 3,
    /// Kernel is terminating.
    Terminating = 4,
    /// Kernel has terminated.
    Terminated = 5,
}

#[pymethods]
impl PyKernelState {
    /// Check if kernel can be activated.
    fn can_activate(&self) -> bool {
        matches!(self, Self::Launched | Self::Deactivated)
    }

    /// Check if kernel can be deactivated.
    fn can_deactivate(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Check if kernel can be terminated.
    fn can_terminate(&self) -> bool {
        matches!(self, Self::Active | Self::Deactivated | Self::Launched)
    }

    /// Check if kernel is running (can process messages).
    fn is_running(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Check if kernel has finished.
    fn is_finished(&self) -> bool {
        matches!(self, Self::Terminated)
    }

    fn __repr__(&self) -> String {
        match self {
            Self::Created => "KernelState.Created".to_string(),
            Self::Launched => "KernelState.Launched".to_string(),
            Self::Active => "KernelState.Active".to_string(),
            Self::Deactivated => "KernelState.Deactivated".to_string(),
            Self::Terminating => "KernelState.Terminating".to_string(),
            Self::Terminated => "KernelState.Terminated".to_string(),
        }
    }
}

impl From<KernelState> for PyKernelState {
    fn from(state: KernelState) -> Self {
        match state {
            KernelState::Created => Self::Created,
            KernelState::Launched => Self::Launched,
            KernelState::Active => Self::Active,
            KernelState::Deactivated => Self::Deactivated,
            KernelState::Terminating => Self::Terminating,
            KernelState::Terminated => Self::Terminated,
        }
    }
}

/// Kernel launch options.
///
/// Configure kernel execution parameters before launching.
///
/// Example:
///     >>> options = LaunchOptions()
///     >>> options = options.with_queue_capacity(2048)
///     >>> options = options.with_block_size(512)
///     >>> kernel = await runtime.launch("processor", options)
#[pyclass]
#[derive(Clone)]
pub struct PyLaunchOptions {
    inner: LaunchOptions,
}

#[pymethods]
impl PyLaunchOptions {
    /// Create default launch options.
    #[new]
    fn new() -> Self {
        Self {
            inner: LaunchOptions::default(),
        }
    }

    /// Create options for a single block with specified thread count.
    #[classmethod]
    fn single_block(_cls: &Bound<'_, PyType>, threads: u32) -> Self {
        Self {
            inner: LaunchOptions::single_block(threads),
        }
    }

    /// Create options for a multi-block grid.
    #[classmethod]
    fn multi_block(_cls: &Bound<'_, PyType>, grid_size: u32, block_size: u32) -> Self {
        Self {
            inner: LaunchOptions::multi_block(grid_size, block_size),
        }
    }

    /// Execution mode.
    #[getter]
    fn mode(&self) -> PyKernelMode {
        self.inner.mode.into()
    }

    /// Grid size (number of blocks).
    #[getter]
    fn grid_size(&self) -> u32 {
        self.inner.grid_size
    }

    /// Block size (threads per block).
    #[getter]
    fn block_size(&self) -> u32 {
        self.inner.block_size
    }

    /// Input queue capacity.
    #[getter]
    fn input_queue_capacity(&self) -> usize {
        self.inner.input_queue_capacity
    }

    /// Output queue capacity.
    #[getter]
    fn output_queue_capacity(&self) -> usize {
        self.inner.output_queue_capacity
    }

    /// Whether kernel auto-activates after launch.
    #[getter]
    fn auto_activate(&self) -> bool {
        self.inner.auto_activate
    }

    /// Whether cooperative groups are enabled.
    #[getter]
    fn cooperative(&self) -> bool {
        self.inner.cooperative
    }

    /// Whether K2K messaging is enabled.
    #[getter]
    fn enable_k2k(&self) -> bool {
        self.inner.enable_k2k
    }

    /// Set execution mode.
    fn with_mode(&self, mode: PyKernelMode) -> Self {
        Self {
            inner: LaunchOptions {
                mode: mode.into(),
                ..self.inner.clone()
            },
        }
    }

    /// Set queue capacity for both input and output.
    fn with_queue_capacity(&self, capacity: usize) -> Self {
        Self {
            inner: self.inner.clone().with_queue_capacity(capacity),
        }
    }

    /// Set input queue capacity.
    fn with_input_queue_capacity(&self, capacity: usize) -> Self {
        Self {
            inner: self.inner.clone().with_input_queue_capacity(capacity),
        }
    }

    /// Set output queue capacity.
    fn with_output_queue_capacity(&self, capacity: usize) -> Self {
        Self {
            inner: self.inner.clone().with_output_queue_capacity(capacity),
        }
    }

    /// Set grid size.
    fn with_grid_size(&self, grid_size: u32) -> Self {
        Self {
            inner: self.inner.clone().with_grid_size(grid_size),
        }
    }

    /// Set block size.
    fn with_block_size(&self, block_size: u32) -> Self {
        Self {
            inner: self.inner.clone().with_block_size(block_size),
        }
    }

    /// Disable auto-activation.
    fn without_auto_activate(&self) -> Self {
        Self {
            inner: self.inner.clone().without_auto_activate(),
        }
    }

    /// Enable cooperative groups.
    fn with_cooperative(&self, enable: bool) -> Self {
        Self {
            inner: self.inner.clone().with_cooperative(enable),
        }
    }

    /// Enable K2K messaging.
    fn with_k2k(&self, enable: bool) -> Self {
        Self {
            inner: self.inner.clone().with_k2k(enable),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LaunchOptions(grid={}, block={}, queues={})",
            self.inner.grid_size, self.inner.block_size, self.inner.input_queue_capacity
        )
    }
}

impl From<LaunchOptions> for PyLaunchOptions {
    fn from(inner: LaunchOptions) -> Self {
        Self { inner }
    }
}

impl From<PyLaunchOptions> for LaunchOptions {
    fn from(opts: PyLaunchOptions) -> Self {
        opts.inner
    }
}

/// Kernel status snapshot.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyKernelStatus {
    /// Kernel ID.
    #[pyo3(get)]
    id: String,
    /// Current state.
    #[pyo3(get)]
    state: PyKernelState,
    /// Execution mode.
    #[pyo3(get)]
    mode: PyKernelMode,
    /// Messages in input queue.
    #[pyo3(get)]
    input_queue_depth: usize,
    /// Messages in output queue.
    #[pyo3(get)]
    output_queue_depth: usize,
    /// Total messages processed.
    #[pyo3(get)]
    messages_processed: u64,
    /// Uptime in seconds.
    #[pyo3(get)]
    uptime_secs: f64,
}

#[pymethods]
impl PyKernelStatus {
    fn __repr__(&self) -> String {
        format!(
            "KernelStatus(id='{}', state={:?}, processed={})",
            self.id, self.state, self.messages_processed
        )
    }
}

impl From<KernelStatus> for PyKernelStatus {
    fn from(status: KernelStatus) -> Self {
        Self {
            id: status.id.as_str().to_string(),
            state: status.state.into(),
            mode: status.mode.into(),
            input_queue_depth: status.input_queue_depth,
            output_queue_depth: status.output_queue_depth,
            messages_processed: status.messages_processed,
            uptime_secs: status.uptime.as_secs_f64(),
        }
    }
}

/// Runtime metrics snapshot.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyRuntimeMetrics {
    /// Active kernel count.
    #[pyo3(get)]
    active_kernels: usize,
    /// Total kernels launched.
    #[pyo3(get)]
    total_launched: u64,
    /// Total messages sent.
    #[pyo3(get)]
    messages_sent: u64,
    /// Total messages received.
    #[pyo3(get)]
    messages_received: u64,
    /// GPU memory used (bytes).
    #[pyo3(get)]
    gpu_memory_used: u64,
    /// Host memory used (bytes).
    #[pyo3(get)]
    host_memory_used: u64,
}

#[pymethods]
impl PyRuntimeMetrics {
    fn __repr__(&self) -> String {
        format!(
            "RuntimeMetrics(kernels={}, launched={}, sent={}, recv={})",
            self.active_kernels, self.total_launched, self.messages_sent, self.messages_received
        )
    }
}

impl From<RuntimeMetrics> for PyRuntimeMetrics {
    fn from(metrics: RuntimeMetrics) -> Self {
        Self {
            active_kernels: metrics.active_kernels,
            total_launched: metrics.total_launched,
            messages_sent: metrics.messages_sent,
            messages_received: metrics.messages_received,
            gpu_memory_used: metrics.gpu_memory_used,
            host_memory_used: metrics.host_memory_used,
        }
    }
}

/// Handle to a launched kernel.
///
/// Provides methods to interact with a running kernel.
///
/// Example:
///     >>> kernel = await runtime.launch("processor", options)
///     >>> await kernel.send(envelope)
///     >>> response = await kernel.receive(timeout=1.0)
///     >>> await kernel.terminate()
#[pyclass]
pub struct PyKernelHandle {
    inner: KernelHandle,
}

#[pymethods]
impl PyKernelHandle {
    /// Kernel identifier.
    #[getter]
    fn id(&self) -> String {
        self.inner.id().as_str().to_string()
    }

    /// Current kernel state.
    #[getter]
    fn state(&self) -> PyKernelState {
        self.inner.status().state.into()
    }

    /// Check if kernel is active.
    fn is_active(&self) -> bool {
        self.inner.status().state.is_running()
    }

    /// Get kernel status.
    fn status(&self) -> PyKernelStatus {
        self.inner.status().into()
    }

    /// Activate the kernel.
    fn activate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let handle = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            handle
                .activate()
                .await
                .map_err(|e| PyRingKernelError::from(e).into_py_err())
        })
    }

    /// Deactivate the kernel (pause).
    fn deactivate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let handle = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            handle
                .deactivate()
                .await
                .map_err(|e| PyRingKernelError::from(e).into_py_err())
        })
    }

    /// Terminate the kernel.
    fn terminate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let handle = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            handle
                .terminate()
                .await
                .map_err(|e| PyRingKernelError::from(e).into_py_err())
        })
    }

    /// Send a message envelope to the kernel.
    fn send<'py>(
        &self,
        py: Python<'py>,
        envelope: &PyMessageEnvelope,
    ) -> PyResult<Bound<'py, PyAny>> {
        let handle = self.inner.clone();
        let env = envelope.clone().into();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            handle
                .send_envelope(env)
                .await
                .map_err(|e| PyRingKernelError::from(e).into_py_err())
        })
    }

    /// Receive a message envelope (blocking).
    fn receive<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let handle = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            handle
                .receive()
                .await
                .map(PyMessageEnvelope::from)
                .map_err(|e| PyRingKernelError::from(e).into_py_err())
        })
    }

    /// Receive with timeout.
    ///
    /// Args:
    ///     timeout: Timeout in seconds.
    ///
    /// Returns:
    ///     MessageEnvelope if received within timeout.
    ///
    /// Raises:
    ///     TimeoutError: If no message received within timeout.
    #[pyo3(signature = (timeout=1.0))]
    fn receive_timeout<'py>(&self, py: Python<'py>, timeout: f64) -> PyResult<Bound<'py, PyAny>> {
        let handle = self.inner.clone();
        let duration = Duration::from_secs_f64(timeout);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            handle
                .receive_timeout(duration)
                .await
                .map(PyMessageEnvelope::from)
                .map_err(|e| PyRingKernelError::from(e).into_py_err())
        })
    }

    /// Try to receive (non-blocking).
    ///
    /// Returns:
    ///     MessageEnvelope if available, None otherwise.
    fn try_receive(&self) -> Option<PyMessageEnvelope> {
        self.inner.try_receive().ok().map(PyMessageEnvelope::from)
    }

    /// Wait for kernel to terminate.
    fn wait<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let handle = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            handle
                .wait()
                .await
                .map_err(|e| PyRingKernelError::from(e).into_py_err())
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "KernelHandle(id='{}', state={:?})",
            self.inner.id().as_str(),
            self.inner.status().state
        )
    }
}

impl From<KernelHandle> for PyKernelHandle {
    fn from(inner: KernelHandle) -> Self {
        Self { inner }
    }
}

/// GPU backend type.
#[pyclass(frozen, eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyBackend {
    /// Auto-select best available backend.
    Auto = 0,
    /// CPU backend (for testing).
    Cpu = 1,
    /// NVIDIA CUDA backend.
    Cuda = 2,
    /// Apple Metal backend.
    Metal = 3,
    /// WebGPU backend.
    Wgpu = 4,
}

#[pymethods]
impl PyBackend {
    fn __repr__(&self) -> String {
        match self {
            Self::Auto => "Backend.Auto".to_string(),
            Self::Cpu => "Backend.Cpu".to_string(),
            Self::Cuda => "Backend.Cuda".to_string(),
            Self::Metal => "Backend.Metal".to_string(),
            Self::Wgpu => "Backend.Wgpu".to_string(),
        }
    }
}

impl From<Backend> for PyBackend {
    fn from(backend: Backend) -> Self {
        match backend {
            Backend::Auto => Self::Auto,
            Backend::Cpu => Self::Cpu,
            Backend::Cuda => Self::Cuda,
            Backend::Metal => Self::Metal,
            Backend::Wgpu => Self::Wgpu,
        }
    }
}

impl From<PyBackend> for Backend {
    fn from(backend: PyBackend) -> Self {
        match backend {
            PyBackend::Auto => Self::Auto,
            PyBackend::Cpu => Self::Cpu,
            PyBackend::Cuda => Self::Cuda,
            PyBackend::Metal => Self::Metal,
            PyBackend::Wgpu => Self::Wgpu,
        }
    }
}

/// Main RingKernel runtime.
///
/// The runtime manages kernel lifecycle, message passing, and resource allocation.
///
/// Example (async):
///     >>> runtime = await RingKernel.create(backend="cpu")
///     >>> kernel = await runtime.launch("processor")
///     >>> await kernel.send(envelope)
///     >>> await runtime.shutdown()
///
/// Example (sync):
///     >>> runtime = RingKernel.create_sync(backend="cpu")
///     >>> kernel = runtime.launch_sync("processor")
///     >>> runtime.shutdown_sync()
///
/// Example (context manager):
///     >>> async with await RingKernel.create() as runtime:
///     ...     kernel = await runtime.launch("processor")
///     ...     # Runtime auto-shuts down on exit
#[pyclass]
pub struct PyRingKernel {
    // Using CpuRuntime for now; can be made generic later
    inner: Arc<CpuRuntime>,
    tokio_runtime: Option<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyRingKernel {
    /// Create a new RingKernel runtime (async).
    ///
    /// Args:
    ///     backend: Backend to use ("auto", "cpu", "cuda", "metal", "webgpu").
    ///     node_id: Node ID for HLC (default: 1).
    ///     enable_k2k: Enable K2K messaging (default: True).
    ///
    /// Returns:
    ///     A new RingKernel runtime.
    #[classmethod]
    #[pyo3(signature = (backend="cpu", node_id=1, enable_k2k=true))]
    fn create<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        backend: &str,
        node_id: u64,
        enable_k2k: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Currently only CPU backend is implemented
        if backend != "cpu" && backend != "auto" {
            return Err(PyRingKernelError::new(
                ErrorKind::CudaNotAvailable,
                format!("Backend '{}' not available, use 'cpu'", backend),
            )
            .into_py_err());
        }

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let inner = CpuRuntime::with_config(node_id, enable_k2k)
                .await
                .map_err(|e| PyRingKernelError::from(e).into_py_err())?;
            Ok(PyRingKernel {
                inner: Arc::new(inner),
                tokio_runtime: None,
            })
        })
    }

    /// Create a new RingKernel runtime (sync).
    ///
    /// This creates an internal tokio runtime for blocking operations.
    ///
    /// Args:
    ///     backend: Backend to use ("auto", "cpu").
    ///     node_id: Node ID for HLC (default: 1).
    ///     enable_k2k: Enable K2K messaging (default: True).
    ///
    /// Returns:
    ///     A new RingKernel runtime.
    #[classmethod]
    #[pyo3(signature = (backend="cpu", node_id=1, enable_k2k=true))]
    fn create_sync(
        _cls: &Bound<'_, PyType>,
        backend: &str,
        node_id: u64,
        enable_k2k: bool,
    ) -> PyResult<Self> {
        if backend != "cpu" && backend != "auto" {
            return Err(PyRingKernelError::new(
                ErrorKind::CudaNotAvailable,
                format!("Backend '{}' not available, use 'cpu'", backend),
            )
            .into_py_err());
        }

        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            PyRingKernelError::new(
                ErrorKind::Runtime,
                format!("Failed to create runtime: {}", e),
            )
            .into_py_err()
        })?;

        let inner = rt.block_on(async {
            CpuRuntime::with_config(node_id, enable_k2k)
                .await
                .map_err(|e| PyRingKernelError::from(e).into_py_err())
        })?;

        Ok(PyRingKernel {
            inner: Arc::new(inner),
            tokio_runtime: Some(rt),
        })
    }

    /// Get the backend type.
    #[getter]
    fn backend(&self) -> PyBackend {
        self.inner.backend().into()
    }

    /// Node ID for HLC.
    #[getter]
    fn node_id(&self) -> u64 {
        self.inner.node_id()
    }

    /// Check if runtime is shut down.
    #[getter]
    fn is_shutdown(&self) -> bool {
        self.inner.is_shutdown()
    }

    /// Check if K2K messaging is enabled.
    #[getter]
    fn is_k2k_enabled(&self) -> bool {
        self.inner.is_k2k_enabled()
    }

    /// Launch a kernel (async).
    ///
    /// Args:
    ///     kernel_id: Unique identifier for the kernel.
    ///     options: Launch options (default: LaunchOptions()).
    ///
    /// Returns:
    ///     KernelHandle for interacting with the kernel.
    #[pyo3(signature = (kernel_id, options=None))]
    fn launch<'py>(
        &self,
        py: Python<'py>,
        kernel_id: &str,
        options: Option<&PyLaunchOptions>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        let id = kernel_id.to_string();
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .launch(&id, opts)
                .await
                .map(PyKernelHandle::from)
                .map_err(|e| PyRingKernelError::from(e).into_py_err())
        })
    }

    /// Launch a kernel (sync).
    #[pyo3(signature = (kernel_id, options=None))]
    fn launch_sync(
        &self,
        kernel_id: &str,
        options: Option<&PyLaunchOptions>,
    ) -> PyResult<PyKernelHandle> {
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();

        let result = if let Some(rt) = &self.tokio_runtime {
            rt.block_on(self.inner.launch(kernel_id, opts))
        } else {
            // Try to get the current runtime
            let handle = tokio::runtime::Handle::try_current().map_err(|_| {
                PyRingKernelError::new(
                    ErrorKind::Runtime,
                    "No tokio runtime available. Use create_sync() or run within async context",
                )
                .into_py_err()
            })?;
            handle.block_on(self.inner.launch(kernel_id, opts))
        };

        result
            .map(PyKernelHandle::from)
            .map_err(|e| PyRingKernelError::from(e).into_py_err())
    }

    /// Get a handle to an existing kernel.
    ///
    /// Args:
    ///     kernel_id: Kernel identifier.
    ///
    /// Returns:
    ///     KernelHandle if found, None otherwise.
    fn get_kernel(&self, kernel_id: &str) -> Option<PyKernelHandle> {
        self.inner
            .get_kernel(&KernelId::new(kernel_id))
            .map(PyKernelHandle::from)
    }

    /// List all kernel IDs.
    fn list_kernels(&self) -> Vec<String> {
        self.inner
            .list_kernels()
            .into_iter()
            .map(|id| id.as_str().to_string())
            .collect()
    }

    /// Get runtime metrics.
    fn metrics(&self) -> PyRuntimeMetrics {
        self.inner.metrics().into()
    }

    /// Shutdown the runtime (async).
    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .shutdown()
                .await
                .map_err(|e| PyRingKernelError::from(e).into_py_err())
        })
    }

    /// Shutdown the runtime (sync).
    fn shutdown_sync(&self) -> PyResult<()> {
        let result = if let Some(rt) = &self.tokio_runtime {
            rt.block_on(self.inner.shutdown())
        } else {
            let handle = tokio::runtime::Handle::try_current().map_err(|_| {
                PyRingKernelError::new(ErrorKind::Runtime, "No tokio runtime available")
                    .into_py_err()
            })?;
            handle.block_on(self.inner.shutdown())
        };

        result.map_err(|e| PyRingKernelError::from(e).into_py_err())
    }

    /// Get the K2K broker (if K2K is enabled).
    fn k2k_broker(&self) -> Option<PyK2KBroker> {
        self.inner
            .k2k_broker()
            .map(|broker| PyK2KBroker::from_arc(Arc::clone(broker)))
    }

    // Context manager support
    fn __aenter__<'py>(self_: PyRef<'py, Self>) -> PyRef<'py, Self> {
        self_
    }

    #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
    fn __aexit__<'py>(
        &self,
        py: Python<'py>,
        _exc_type: Option<&Bound<'py, PyAny>>,
        _exc_val: Option<&Bound<'py, PyAny>>,
        _exc_tb: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.shutdown(py)
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
    ) -> PyResult<bool> {
        self.shutdown_sync()?;
        Ok(false)
    }

    fn __repr__(&self) -> String {
        let metrics = self.inner.metrics();
        format!(
            "RingKernel(backend={:?}, kernels={}, k2k={})",
            self.inner.backend(),
            metrics.active_kernels,
            self.inner.is_k2k_enabled()
        )
    }
}

/// Register runtime types with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKernelMode>()?;
    m.add_class::<PyKernelState>()?;
    m.add_class::<PyLaunchOptions>()?;
    m.add_class::<PyKernelStatus>()?;
    m.add_class::<PyRuntimeMetrics>()?;
    m.add_class::<PyKernelHandle>()?;
    m.add_class::<PyBackend>()?;
    m.add_class::<PyRingKernel>()?;

    Ok(())
}
