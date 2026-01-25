//! CUDA Python bindings for RingKernel.
//!
//! Provides GPU device management, memory pooling, stream orchestration, and profiling.

use pyo3::prelude::*;
use pyo3::types::PyType;

use ringkernel_cuda::{
    cuda_device_count as rust_cuda_device_count, is_cuda_available as rust_is_cuda_available,
    CudaDevice as RustCudaDevice, GpuPoolConfig as RustGpuPoolConfig, GpuPoolDiagnostics,
    GpuSizeClass, OverlapMetrics as RustOverlapMetrics, StreamConfig as RustStreamConfig, StreamId,
    StreamPoolStats as RustStreamPoolStats,
};

use crate::error::PyRingKernelError;

// =============================================================================
// Device Types
// =============================================================================

/// Check if CUDA is available on this system.
#[pyfunction]
fn is_cuda_available() -> bool {
    rust_is_cuda_available()
}

/// Get the number of CUDA devices.
#[pyfunction]
fn cuda_device_count() -> usize {
    rust_cuda_device_count()
}

/// Information about a CUDA device.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyCudaDeviceInfo {
    /// Device ordinal number.
    #[pyo3(get)]
    pub ordinal: usize,
    /// Device name.
    #[pyo3(get)]
    pub name: String,
    /// Compute capability (major, minor).
    #[pyo3(get)]
    pub compute_capability: (u32, u32),
    /// Total memory in bytes.
    #[pyo3(get)]
    pub total_memory: usize,
    /// Whether the device supports persistent kernels.
    #[pyo3(get)]
    pub supports_persistent: bool,
}

#[pymethods]
impl PyCudaDeviceInfo {
    /// Get total memory in megabytes.
    fn total_memory_mb(&self) -> f64 {
        self.total_memory as f64 / (1024.0 * 1024.0)
    }

    /// Get total memory in gigabytes.
    fn total_memory_gb(&self) -> f64 {
        self.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "CudaDeviceInfo(ordinal={}, name='{}', cc={}.{}, memory={:.1}GB)",
            self.ordinal,
            self.name,
            self.compute_capability.0,
            self.compute_capability.1,
            self.total_memory_gb()
        )
    }
}

/// Enumerate all CUDA devices.
#[pyfunction]
fn enumerate_devices() -> PyResult<Vec<PyCudaDeviceInfo>> {
    let count = rust_cuda_device_count();
    let mut devices = Vec::with_capacity(count);
    for ordinal in 0..count {
        match RustCudaDevice::new(ordinal) {
            Ok(device) => {
                devices.push(PyCudaDeviceInfo {
                    ordinal,
                    name: device.name().to_string(),
                    compute_capability: device.compute_capability(),
                    total_memory: device.total_memory(),
                    supports_persistent: device.supports_persistent_kernels(),
                });
            }
            Err(e) => return Err(PyRingKernelError::from(e).into_py_err()),
        }
    }
    Ok(devices)
}

/// A CUDA device for GPU computation.
#[pyclass]
pub struct PyCudaDevice {
    inner: RustCudaDevice,
}

#[pymethods]
impl PyCudaDevice {
    /// Create a new CUDA device wrapper.
    ///
    /// Args:
    ///     ordinal: Device ordinal (0 for first GPU).
    ///
    /// Returns:
    ///     A CudaDevice instance.
    ///
    /// Raises:
    ///     CudaDeviceError: If device initialization fails.
    #[new]
    #[pyo3(signature = (ordinal=0))]
    fn new(ordinal: usize) -> PyResult<Self> {
        RustCudaDevice::new(ordinal)
            .map(|inner| Self { inner })
            .map_err(|e| PyRingKernelError::from(e).into_py_err())
    }

    /// Device ordinal number.
    #[getter]
    fn ordinal(&self) -> usize {
        self.inner.ordinal()
    }

    /// Device name.
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Compute capability as (major, minor).
    #[getter]
    fn compute_capability(&self) -> (u32, u32) {
        self.inner.compute_capability()
    }

    /// Total global memory in bytes.
    #[getter]
    fn total_memory(&self) -> usize {
        self.inner.total_memory()
    }

    /// Total memory in megabytes.
    fn total_memory_mb(&self) -> f64 {
        self.inner.total_memory() as f64 / (1024.0 * 1024.0)
    }

    /// Check if this device supports persistent kernels (CC 7.0+).
    fn supports_persistent_kernels(&self) -> bool {
        self.inner.supports_persistent_kernels()
    }

    /// Check if this device supports cooperative groups (CC 6.0+).
    fn supports_cooperative_groups(&self) -> bool {
        self.inner.supports_cooperative_groups()
    }

    /// Synchronize the device (wait for all operations to complete).
    fn synchronize(&self) -> PyResult<()> {
        self.inner
            .synchronize()
            .map_err(|e| PyRingKernelError::from(e).into_py_err())
    }

    fn __repr__(&self) -> String {
        format!(
            "CudaDevice(ordinal={}, name='{}', cc={}.{}, memory={:.1}GB)",
            self.inner.ordinal(),
            self.inner.name(),
            self.inner.compute_capability().0,
            self.inner.compute_capability().1,
            self.total_memory_mb() / 1024.0
        )
    }
}

// =============================================================================
// Memory Pool Types
// =============================================================================

/// Size class for stratified memory pooling.
#[pyclass(frozen, eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyGpuSizeClass {
    /// 256 bytes.
    Size256B = 0,
    /// 1 KB.
    Size1KB = 1,
    /// 4 KB.
    Size4KB = 2,
    /// 16 KB.
    Size16KB = 3,
    /// 64 KB.
    Size64KB = 4,
    /// 256 KB.
    Size256KB = 5,
}

#[pymethods]
impl PyGpuSizeClass {
    /// Get the byte size for this class.
    fn bytes(&self) -> usize {
        match self {
            Self::Size256B => 256,
            Self::Size1KB => 1024,
            Self::Size4KB => 4096,
            Self::Size16KB => 16384,
            Self::Size64KB => 65536,
            Self::Size256KB => 262144,
        }
    }

    /// Get the appropriate size class for a given byte count.
    #[classmethod]
    fn for_size(_cls: &Bound<'_, PyType>, bytes: usize) -> Option<Self> {
        GpuSizeClass::for_size(bytes).map(|c| match c {
            GpuSizeClass::Size256B => Self::Size256B,
            GpuSizeClass::Size1KB => Self::Size1KB,
            GpuSizeClass::Size4KB => Self::Size4KB,
            GpuSizeClass::Size16KB => Self::Size16KB,
            GpuSizeClass::Size64KB => Self::Size64KB,
            GpuSizeClass::Size256KB => Self::Size256KB,
        })
    }

    fn __repr__(&self) -> String {
        format!("GpuSizeClass.Size{}B", self.bytes())
    }
}

impl From<PyGpuSizeClass> for GpuSizeClass {
    fn from(py_class: PyGpuSizeClass) -> Self {
        match py_class {
            PyGpuSizeClass::Size256B => Self::Size256B,
            PyGpuSizeClass::Size1KB => Self::Size1KB,
            PyGpuSizeClass::Size4KB => Self::Size4KB,
            PyGpuSizeClass::Size16KB => Self::Size16KB,
            PyGpuSizeClass::Size64KB => Self::Size64KB,
            PyGpuSizeClass::Size256KB => Self::Size256KB,
        }
    }
}

/// Configuration for GPU memory pool.
#[pyclass]
#[derive(Clone)]
pub struct PyGpuPoolConfig {
    inner: RustGpuPoolConfig,
}

#[pymethods]
impl PyGpuPoolConfig {
    /// Create default configuration.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustGpuPoolConfig::default(),
        }
    }

    /// Configuration optimized for graph analytics.
    #[classmethod]
    fn for_graph_analytics(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustGpuPoolConfig::for_graph_analytics(),
        }
    }

    /// Configuration optimized for simulations.
    #[classmethod]
    fn for_simulation(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustGpuPoolConfig::for_simulation(),
        }
    }

    /// Minimal configuration for testing.
    #[classmethod]
    fn minimal(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustGpuPoolConfig::minimal(),
        }
    }

    /// Whether allocation tracking is enabled.
    #[getter]
    fn track_allocations(&self) -> bool {
        self.inner.track_allocations
    }

    /// Maximum pool memory in bytes.
    #[getter]
    fn max_pool_bytes(&self) -> usize {
        self.inner.max_pool_bytes
    }

    fn __repr__(&self) -> String {
        format!(
            "GpuPoolConfig(tracking={}, max_bytes={})",
            self.inner.track_allocations, self.inner.max_pool_bytes
        )
    }
}

/// Per-bucket statistics for memory pool.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyGpuBucketStats {
    /// Size class in bytes.
    #[pyo3(get)]
    pub size_bytes: usize,
    /// Total blocks allocated from CUDA.
    #[pyo3(get)]
    pub total_blocks: usize,
    /// Blocks currently in use.
    #[pyo3(get)]
    pub in_use_blocks: usize,
    /// Blocks in free list.
    #[pyo3(get)]
    pub free_blocks: usize,
}

#[pymethods]
impl PyGpuBucketStats {
    /// Utilization ratio (0.0-1.0).
    fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.in_use_blocks as f64 / self.total_blocks as f64
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GpuBucketStats(size={}B, total={}, in_use={}, free={})",
            self.size_bytes, self.total_blocks, self.in_use_blocks, self.free_blocks
        )
    }
}

/// Diagnostics for GPU memory pool.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyGpuPoolDiagnostics {
    /// Total bytes allocated from CUDA.
    #[pyo3(get)]
    pub total_cuda_bytes: u64,
    /// Bytes currently in use.
    #[pyo3(get)]
    pub in_use_bytes: u64,
    /// Bytes in free lists.
    #[pyo3(get)]
    pub free_bytes: u64,
    /// Fragmentation ratio (0.0-1.0).
    #[pyo3(get)]
    pub fragmentation: f64,
    /// Large allocation count (non-pooled).
    #[pyo3(get)]
    pub large_allocation_count: usize,
    /// Large allocation bytes.
    #[pyo3(get)]
    pub large_allocation_bytes: u64,
    /// Lifetime allocation count.
    #[pyo3(get)]
    pub total_allocations: u64,
    /// Lifetime deallocation count.
    #[pyo3(get)]
    pub total_deallocations: u64,
    /// Pool hit rate (0.0-1.0).
    #[pyo3(get)]
    pub hit_rate: f64,
    /// Per-bucket statistics.
    bucket_stats: Vec<PyGpuBucketStats>,
}

#[pymethods]
impl PyGpuPoolDiagnostics {
    /// Get per-bucket statistics.
    fn bucket_stats(&self) -> Vec<PyGpuBucketStats> {
        self.bucket_stats.clone()
    }

    /// Get utilization ratio.
    fn utilization(&self) -> f64 {
        if self.total_cuda_bytes == 0 {
            0.0
        } else {
            self.in_use_bytes as f64 / self.total_cuda_bytes as f64
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GpuPoolDiagnostics(used={}B, total={}B, util={:.1}%, hit_rate={:.1}%)",
            self.in_use_bytes,
            self.total_cuda_bytes,
            self.utilization() * 100.0,
            self.hit_rate * 100.0
        )
    }
}

impl From<GpuPoolDiagnostics> for PyGpuPoolDiagnostics {
    fn from(diag: GpuPoolDiagnostics) -> Self {
        Self {
            total_cuda_bytes: diag.total_cuda_bytes,
            in_use_bytes: diag.in_use_bytes,
            free_bytes: diag.free_bytes,
            fragmentation: diag.fragmentation,
            large_allocation_count: diag.large_allocation_count,
            large_allocation_bytes: diag.large_allocation_bytes,
            total_allocations: diag.total_allocations,
            total_deallocations: diag.total_deallocations,
            hit_rate: diag.hit_rate,
            bucket_stats: diag
                .bucket_stats
                .iter()
                .map(|s| PyGpuBucketStats {
                    size_bytes: s.size_bytes,
                    total_blocks: s.total_blocks,
                    in_use_blocks: s.in_use_blocks,
                    free_blocks: s.free_blocks,
                })
                .collect(),
        }
    }
}

// =============================================================================
// Stream Types
// =============================================================================

/// Stream identifier.
#[pyclass(frozen, eq)]
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PyStreamId {
    /// Compute stream with index.
    Compute { index: usize },
    /// Dedicated transfer stream.
    Transfer {},
    /// Default stream (stream 0).
    Default {},
}

#[pymethods]
impl PyStreamId {
    /// Create a compute stream ID.
    #[classmethod]
    fn compute(_cls: &Bound<'_, PyType>, index: usize) -> Self {
        Self::Compute { index }
    }

    /// Create a transfer stream ID.
    #[classmethod]
    fn transfer(_cls: &Bound<'_, PyType>) -> Self {
        Self::Transfer {}
    }

    /// Create a default stream ID.
    #[classmethod]
    fn default_stream(_cls: &Bound<'_, PyType>) -> Self {
        Self::Default {}
    }

    fn __repr__(&self) -> String {
        match self {
            Self::Compute { index } => format!("StreamId.Compute({})", index),
            Self::Transfer {} => "StreamId.Transfer".to_string(),
            Self::Default {} => "StreamId.Default".to_string(),
        }
    }
}

impl From<PyStreamId> for StreamId {
    fn from(py_id: PyStreamId) -> Self {
        match py_id {
            PyStreamId::Compute { index } => StreamId::Compute(index),
            PyStreamId::Transfer {} => StreamId::Transfer,
            PyStreamId::Default {} => StreamId::Default,
        }
    }
}

impl From<StreamId> for PyStreamId {
    fn from(id: StreamId) -> Self {
        match id {
            StreamId::Compute(index) => PyStreamId::Compute { index },
            StreamId::Transfer => PyStreamId::Transfer {},
            StreamId::Default => PyStreamId::Default {},
        }
    }
}

/// Stream manager configuration.
#[pyclass]
#[derive(Clone)]
pub struct PyStreamConfig {
    inner: RustStreamConfig,
}

#[pymethods]
impl PyStreamConfig {
    /// Create default configuration (4 compute + transfer).
    #[new]
    fn new() -> Self {
        Self {
            inner: RustStreamConfig::default(),
        }
    }

    /// Minimal configuration (1 stream, no transfer).
    #[classmethod]
    fn minimal(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustStreamConfig::minimal(),
        }
    }

    /// Performance configuration (4 streams + transfer + graphs).
    #[classmethod]
    fn performance(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustStreamConfig::performance(),
        }
    }

    /// Configuration for simulation workloads.
    #[classmethod]
    fn for_simulation(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            inner: RustStreamConfig::for_simulation(),
        }
    }

    /// Number of compute streams.
    #[getter]
    fn num_compute_streams(&self) -> usize {
        self.inner.num_compute_streams
    }

    /// Whether transfer stream is enabled.
    #[getter]
    fn use_transfer_stream(&self) -> bool {
        self.inner.use_transfer_stream
    }

    /// Whether graph capture is enabled.
    #[getter]
    fn enable_graph_capture(&self) -> bool {
        self.inner.enable_graph_capture
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamConfig(compute={}, transfer={}, graphs={})",
            self.inner.num_compute_streams,
            self.inner.use_transfer_stream,
            self.inner.enable_graph_capture
        )
    }
}

/// Compute/transfer overlap metrics.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyOverlapMetrics {
    /// Total compute time in nanoseconds.
    #[pyo3(get)]
    pub compute_ns: u64,
    /// Total transfer time in nanoseconds.
    #[pyo3(get)]
    pub transfer_ns: u64,
    /// Overlapped time in nanoseconds.
    #[pyo3(get)]
    pub overlap_ns: u64,
    /// Number of overlapped operations.
    #[pyo3(get)]
    pub overlap_count: u64,
}

#[pymethods]
impl PyOverlapMetrics {
    /// Compute time in milliseconds.
    fn compute_ms(&self) -> f64 {
        self.compute_ns as f64 / 1_000_000.0
    }

    /// Transfer time in milliseconds.
    fn transfer_ms(&self) -> f64 {
        self.transfer_ns as f64 / 1_000_000.0
    }

    /// Overlap time in milliseconds.
    fn overlap_ms(&self) -> f64 {
        self.overlap_ns as f64 / 1_000_000.0
    }

    /// Overlap efficiency (0.0-1.0).
    fn overlap_efficiency(&self) -> f64 {
        let total = self.compute_ns + self.transfer_ns;
        if total == 0 {
            0.0
        } else {
            self.overlap_ns as f64 / total as f64
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OverlapMetrics(compute={:.2}ms, transfer={:.2}ms, overlap={:.2}ms, efficiency={:.1}%)",
            self.compute_ms(),
            self.transfer_ms(),
            self.overlap_ms(),
            self.overlap_efficiency() * 100.0
        )
    }
}

impl From<RustOverlapMetrics> for PyOverlapMetrics {
    fn from(m: RustOverlapMetrics) -> Self {
        Self {
            compute_ns: m.compute_ns,
            transfer_ns: m.transfer_ns,
            overlap_ns: m.overlap_ns,
            overlap_count: m.overlap_count,
        }
    }
}

/// Stream pool statistics.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct PyStreamPoolStats {
    /// Total kernel launches.
    #[pyo3(get)]
    pub total_launches: u64,
    /// Launches per stream.
    #[pyo3(get)]
    pub per_stream_launches: Vec<u64>,
    /// Average throughput.
    #[pyo3(get)]
    pub launches_per_second: f64,
    /// Number of assigned workloads.
    #[pyo3(get)]
    pub workload_count: usize,
}

#[pymethods]
impl PyStreamPoolStats {
    /// Get the index of the most utilized stream.
    fn most_utilized_stream(&self) -> Option<usize> {
        self.per_stream_launches
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
    }

    /// Get the index of the least utilized stream.
    fn least_utilized_stream(&self) -> Option<usize> {
        self.per_stream_launches
            .iter()
            .enumerate()
            .min_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
    }

    /// Balance ratio (0.0-1.0, higher is better).
    fn balance_ratio(&self) -> f64 {
        if self.per_stream_launches.is_empty() {
            return 1.0;
        }
        let max = *self.per_stream_launches.iter().max().unwrap_or(&1) as f64;
        let min = *self.per_stream_launches.iter().min().unwrap_or(&0) as f64;
        if max == 0.0 {
            1.0
        } else {
            min / max
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamPoolStats(launches={}, streams={}, balance={:.2})",
            self.total_launches,
            self.per_stream_launches.len(),
            self.balance_ratio()
        )
    }
}

impl From<RustStreamPoolStats> for PyStreamPoolStats {
    fn from(s: RustStreamPoolStats) -> Self {
        Self {
            total_launches: s.total_launches,
            per_stream_launches: s.per_stream_launches,
            launches_per_second: s.launches_per_second,
            workload_count: s.workload_count,
        }
    }
}

// =============================================================================
// Module Registration
// =============================================================================

/// Register CUDA types with the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create cuda submodule
    let cuda = PyModule::new_bound(m.py(), "cuda")?;

    // Device functions
    cuda.add_function(wrap_pyfunction!(is_cuda_available, &cuda)?)?;
    cuda.add_function(wrap_pyfunction!(cuda_device_count, &cuda)?)?;
    cuda.add_function(wrap_pyfunction!(enumerate_devices, &cuda)?)?;

    // Device types
    cuda.add_class::<PyCudaDeviceInfo>()?;
    cuda.add_class::<PyCudaDevice>()?;

    // Memory pool types
    cuda.add_class::<PyGpuSizeClass>()?;
    cuda.add_class::<PyGpuPoolConfig>()?;
    cuda.add_class::<PyGpuBucketStats>()?;
    cuda.add_class::<PyGpuPoolDiagnostics>()?;

    // Stream types
    cuda.add_class::<PyStreamId>()?;
    cuda.add_class::<PyStreamConfig>()?;
    cuda.add_class::<PyOverlapMetrics>()?;
    cuda.add_class::<PyStreamPoolStats>()?;

    // Add submodule
    m.add_submodule(&cuda)?;

    // Add commonly-used types at top level
    m.add_class::<PyCudaDevice>()?;
    m.add_function(wrap_pyfunction!(is_cuda_available, m)?)?;

    Ok(())
}
