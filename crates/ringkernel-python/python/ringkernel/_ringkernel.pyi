"""Type stubs for ringkernel._ringkernel."""

from typing import Optional, List, Awaitable, ContextManager, AsyncContextManager
from enum import IntEnum

__version__: str

# Constants
MAX_PAYLOAD_SIZE: int
MESSAGE_MAGIC: int
MESSAGE_VERSION: int

# =============================================================================
# Core Runtime
# =============================================================================

class Backend(IntEnum):
    """GPU backend type."""
    Auto = 0
    Cpu = 1
    Cuda = 2
    Metal = 3
    Wgpu = 4

class KernelMode(IntEnum):
    """Kernel execution mode."""
    Persistent = 0
    EventDriven = 1

class KernelState(IntEnum):
    """Kernel lifecycle state."""
    Created = 0
    Launched = 1
    Active = 2
    Deactivated = 3
    Terminating = 4
    Terminated = 5

    def can_activate(self) -> bool: ...
    def can_deactivate(self) -> bool: ...
    def can_terminate(self) -> bool: ...
    def is_running(self) -> bool: ...
    def is_finished(self) -> bool: ...

class LaunchOptions:
    """Kernel launch options."""
    def __init__(self) -> None: ...

    @classmethod
    def single_block(cls, threads: int) -> LaunchOptions: ...

    @classmethod
    def multi_block(cls, grid_size: int, block_size: int) -> LaunchOptions: ...

    @property
    def mode(self) -> KernelMode: ...
    @property
    def grid_size(self) -> int: ...
    @property
    def block_size(self) -> int: ...
    @property
    def input_queue_capacity(self) -> int: ...
    @property
    def output_queue_capacity(self) -> int: ...
    @property
    def auto_activate(self) -> bool: ...
    @property
    def cooperative(self) -> bool: ...
    @property
    def enable_k2k(self) -> bool: ...

    def with_mode(self, mode: KernelMode) -> LaunchOptions: ...
    def with_queue_capacity(self, capacity: int) -> LaunchOptions: ...
    def with_input_queue_capacity(self, capacity: int) -> LaunchOptions: ...
    def with_output_queue_capacity(self, capacity: int) -> LaunchOptions: ...
    def with_grid_size(self, grid_size: int) -> LaunchOptions: ...
    def with_block_size(self, block_size: int) -> LaunchOptions: ...
    def without_auto_activate(self) -> LaunchOptions: ...
    def with_cooperative(self, enable: bool) -> LaunchOptions: ...
    def with_k2k(self, enable: bool) -> LaunchOptions: ...

class KernelStatus:
    """Kernel status snapshot."""
    @property
    def id(self) -> str: ...
    @property
    def state(self) -> KernelState: ...
    @property
    def mode(self) -> KernelMode: ...
    @property
    def input_queue_depth(self) -> int: ...
    @property
    def output_queue_depth(self) -> int: ...
    @property
    def messages_processed(self) -> int: ...
    @property
    def uptime_secs(self) -> float: ...

class RuntimeMetrics:
    """Runtime metrics snapshot."""
    @property
    def active_kernels(self) -> int: ...
    @property
    def total_launched(self) -> int: ...
    @property
    def messages_sent(self) -> int: ...
    @property
    def messages_received(self) -> int: ...
    @property
    def gpu_memory_used(self) -> int: ...
    @property
    def host_memory_used(self) -> int: ...

class KernelHandle:
    """Handle to a launched kernel."""
    @property
    def id(self) -> str: ...
    @property
    def state(self) -> KernelState: ...

    def is_active(self) -> bool: ...
    def status(self) -> KernelStatus: ...

    def activate(self) -> Awaitable[None]: ...
    def deactivate(self) -> Awaitable[None]: ...
    def terminate(self) -> Awaitable[None]: ...
    def send(self, envelope: MessageEnvelope) -> Awaitable[None]: ...
    def receive(self) -> Awaitable[MessageEnvelope]: ...
    def receive_timeout(self, timeout: float = 1.0) -> Awaitable[MessageEnvelope]: ...
    def try_receive(self) -> Optional[MessageEnvelope]: ...
    def wait(self) -> Awaitable[None]: ...

class RingKernel(ContextManager['RingKernel'], AsyncContextManager['RingKernel']):
    """Main RingKernel runtime."""

    @classmethod
    def create(
        cls,
        backend: str = "cpu",
        node_id: int = 1,
        enable_k2k: bool = True,
    ) -> Awaitable[RingKernel]: ...

    @classmethod
    def create_sync(
        cls,
        backend: str = "cpu",
        node_id: int = 1,
        enable_k2k: bool = True,
    ) -> RingKernel: ...

    @property
    def backend(self) -> Backend: ...
    @property
    def node_id(self) -> int: ...
    @property
    def is_shutdown(self) -> bool: ...
    @property
    def is_k2k_enabled(self) -> bool: ...

    def launch(
        self,
        kernel_id: str,
        options: Optional[LaunchOptions] = None,
    ) -> Awaitable[KernelHandle]: ...

    def launch_sync(
        self,
        kernel_id: str,
        options: Optional[LaunchOptions] = None,
    ) -> KernelHandle: ...

    def get_kernel(self, kernel_id: str) -> Optional[KernelHandle]: ...
    def list_kernels(self) -> List[str]: ...
    def metrics(self) -> RuntimeMetrics: ...

    def shutdown(self) -> Awaitable[None]: ...
    def shutdown_sync(self) -> None: ...

    def k2k_broker(self) -> Optional[K2KBroker]: ...

# =============================================================================
# Messages
# =============================================================================

class MessageId:
    """Unique message identifier."""
    def __init__(self, id: int) -> None: ...

    @classmethod
    def generate(cls) -> MessageId: ...

    @property
    def value(self) -> int: ...

    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __int__(self) -> int: ...

class CorrelationId:
    """Correlation ID for request-response tracking."""
    def __init__(self, id: int) -> None: ...

    @classmethod
    def generate(cls) -> CorrelationId: ...

    @classmethod
    def none(cls) -> CorrelationId: ...

    def is_some(self) -> bool: ...

    @property
    def value(self) -> Optional[int]: ...

    def __bool__(self) -> bool: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

class Priority(IntEnum):
    """Message priority levels."""
    Low = 0
    Normal = 1
    High = 2
    Critical = 3

    @classmethod
    def from_int(cls, value: int) -> Priority: ...

    @property
    def value(self) -> int: ...

class MessageHeader:
    """Message header containing metadata."""

    @classmethod
    def new(
        cls,
        type_id: int,
        source_kernel: int,
        dest_kernel: int,
        payload_size: int,
        timestamp: HlcTimestamp,
    ) -> MessageHeader: ...

    @property
    def magic(self) -> int: ...
    @property
    def message_id(self) -> MessageId: ...
    @property
    def correlation_id(self) -> CorrelationId: ...
    @property
    def source_kernel(self) -> int: ...
    @property
    def dest_kernel(self) -> int: ...
    @property
    def message_type(self) -> int: ...
    @property
    def priority(self) -> Priority: ...
    @property
    def payload_size(self) -> int: ...
    @property
    def timestamp(self) -> HlcTimestamp: ...

    def validate(self) -> bool: ...
    def with_correlation(self, correlation_id: CorrelationId) -> None: ...
    def with_priority(self, priority: Priority) -> None: ...

class MessageEnvelope:
    """Message envelope containing header and payload."""

    @classmethod
    def from_bytes(cls, header: MessageHeader, payload: bytes) -> MessageEnvelope: ...

    @classmethod
    def empty(
        cls,
        source_kernel: int,
        dest_kernel: int,
        timestamp: HlcTimestamp,
    ) -> MessageEnvelope: ...

    @classmethod
    def from_raw_bytes(cls, data: bytes) -> MessageEnvelope: ...

    @property
    def header(self) -> MessageHeader: ...
    @property
    def payload(self) -> bytes: ...
    @property
    def total_size(self) -> int: ...

    def to_bytes(self) -> bytes: ...

# =============================================================================
# HLC
# =============================================================================

class HlcTimestamp:
    """Hybrid Logical Clock timestamp."""
    def __init__(self, physical: int, logical: int, node_id: int) -> None: ...

    @classmethod
    def now(cls, node_id: int) -> HlcTimestamp: ...

    @classmethod
    def zero(cls) -> HlcTimestamp: ...

    @classmethod
    def unpack(cls, packed: int) -> HlcTimestamp: ...

    @property
    def physical(self) -> int: ...
    @property
    def logical(self) -> int: ...
    @property
    def node_id(self) -> int: ...

    def is_zero(self) -> bool: ...
    def as_micros(self) -> int: ...
    def as_millis(self) -> int: ...
    def pack(self) -> int: ...

    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __lt__(self, other: HlcTimestamp) -> bool: ...
    def __le__(self, other: HlcTimestamp) -> bool: ...
    def __gt__(self, other: HlcTimestamp) -> bool: ...
    def __ge__(self, other: HlcTimestamp) -> bool: ...

class HlcClock:
    """Hybrid Logical Clock for generating timestamps."""
    def __init__(self, node_id: int, max_drift_ms: Optional[int] = None) -> None: ...

    @property
    def node_id(self) -> int: ...

    def now(self) -> HlcTimestamp: ...
    def tick(self) -> HlcTimestamp: ...
    def update(self, received: HlcTimestamp) -> HlcTimestamp: ...

# =============================================================================
# K2K
# =============================================================================

class K2KConfig:
    """K2K messaging configuration."""
    def __init__(
        self,
        max_pending_messages: int = 1024,
        delivery_timeout_ms: int = 5000,
        enable_tracing: bool = False,
        max_hops: int = 8,
    ) -> None: ...

    @classmethod
    def default(cls) -> K2KConfig: ...

    @property
    def max_pending_messages(self) -> int: ...
    @property
    def delivery_timeout_ms(self) -> int: ...
    @property
    def enable_tracing(self) -> bool: ...
    @property
    def max_hops(self) -> int: ...

class DeliveryStatus(IntEnum):
    """Message delivery status."""
    Delivered = 0
    Pending = 1
    NotFound = 2
    QueueFull = 3
    Timeout = 4
    MaxHopsExceeded = 5

    def is_success(self) -> bool: ...
    def is_pending(self) -> bool: ...
    def is_failure(self) -> bool: ...

class DeliveryReceipt:
    """Receipt for a sent message."""
    @property
    def message_id(self) -> MessageId: ...
    @property
    def source(self) -> str: ...
    @property
    def destination(self) -> str: ...
    @property
    def status(self) -> DeliveryStatus: ...
    @property
    def timestamp(self) -> HlcTimestamp: ...

class K2KStats:
    """K2K messaging statistics."""
    @property
    def registered_endpoints(self) -> int: ...
    @property
    def messages_delivered(self) -> int: ...
    @property
    def routes_configured(self) -> int: ...

class K2KBroker:
    """Central broker for K2K messaging."""
    def __init__(self, config: K2KConfig) -> None: ...

    def is_registered(self, kernel_id: str) -> bool: ...
    def registered_kernels(self) -> List[str]: ...
    def add_route(self, destination: str, next_hop: str) -> None: ...
    def remove_route(self, destination: str) -> None: ...
    def stats(self) -> K2KStats: ...

# =============================================================================
# Queue
# =============================================================================

class QueueStats:
    """Queue statistics snapshot."""
    @property
    def enqueued(self) -> int: ...
    @property
    def dequeued(self) -> int: ...
    @property
    def dropped(self) -> int: ...
    @property
    def depth(self) -> int: ...
    @property
    def max_depth(self) -> int: ...

    def utilization(self, capacity: int) -> float: ...
    def drop_rate(self) -> float: ...

class QueueTier(IntEnum):
    """Queue capacity tier."""
    Small = 256
    Medium = 1024
    Large = 4096
    ExtraLarge = 16384

    @classmethod
    def for_throughput(
        cls,
        messages_per_second: int,
        headroom_ms: int = 100,
    ) -> QueueTier: ...

    @property
    def capacity(self) -> int: ...

    def upgrade(self) -> QueueTier: ...
    def downgrade(self) -> QueueTier: ...

class QueueHealth(IntEnum):
    """Queue health status."""
    Healthy = 0
    Warning = 1
    Critical = 2

class QueueMonitor:
    """Queue health monitor."""
    def __init__(
        self,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.90,
    ) -> None: ...

    @classmethod
    def default(cls) -> QueueMonitor: ...

    @property
    def warning_threshold(self) -> float: ...
    @property
    def critical_threshold(self) -> float: ...

    def utilization(self, depth: int, capacity: int) -> float: ...
    def utilization_percent(self, depth: int, capacity: int) -> float: ...
    def check_health(self, depth: int, capacity: int) -> QueueHealth: ...
    def suggest_upgrade(
        self,
        depth: int,
        capacity: int,
        current_tier: QueueTier,
    ) -> Optional[QueueTier]: ...

class QueueMetrics:
    """Complete queue metrics snapshot."""
    @property
    def health(self) -> QueueHealth: ...
    @property
    def utilization(self) -> float: ...
    @property
    def stats(self) -> QueueStats: ...
    @property
    def tier(self) -> Optional[QueueTier]: ...
    @property
    def suggested_upgrade(self) -> Optional[QueueTier]: ...

class PartitionedQueueStats:
    """Statistics for partitioned queues."""
    @property
    def total(self) -> QueueStats: ...
    @property
    def partition_count(self) -> int: ...

    def partition_stats(self, partition: int) -> Optional[QueueStats]: ...
    def load_imbalance(self) -> float: ...
    def max_partition_utilization(self, capacity_per_partition: int) -> float: ...

# =============================================================================
# Hybrid
# =============================================================================

class ProcessingMode(IntEnum):
    """Processing mode for hybrid dispatch."""
    CpuOnly = 0
    GpuOnly = 1
    Hybrid = 2
    Adaptive = 3

class HybridConfig:
    """Hybrid dispatcher configuration."""
    def __init__(
        self,
        mode: ProcessingMode = ProcessingMode.CpuOnly,
        gpu_threshold: int = 10000,
        learning_rate: float = 0.1,
        gpu_available: bool = False,
    ) -> None: ...

    @classmethod
    def cpu_only(cls) -> HybridConfig: ...

    @classmethod
    def gpu_only(cls) -> HybridConfig: ...

    @classmethod
    def adaptive(cls) -> HybridConfig: ...

    @property
    def mode(self) -> ProcessingMode: ...
    @property
    def gpu_threshold(self) -> int: ...
    @property
    def learning_rate(self) -> float: ...
    @property
    def gpu_available(self) -> bool: ...

class HybridStats:
    """Hybrid dispatcher statistics."""
    @property
    def cpu_count(self) -> int: ...
    @property
    def gpu_count(self) -> int: ...
    @property
    def adaptive_threshold(self) -> Optional[int]: ...

    def total(self) -> int: ...
    def gpu_ratio(self) -> float: ...

class HybridDispatcher:
    """Hybrid CPU-GPU dispatcher."""
    def __init__(self, config: HybridConfig) -> None: ...

    @classmethod
    def with_defaults(cls) -> HybridDispatcher: ...

    @property
    def adaptive_threshold(self) -> int: ...

    def should_use_gpu(self, workload_size: int) -> bool: ...
    def record_cpu_execution(self) -> None: ...
    def record_gpu_execution(self) -> None: ...
    def update_adaptive_threshold(
        self,
        workload_size: int,
        cpu_time_ms: float,
        gpu_time_ms: float,
    ) -> None: ...
    def stats(self) -> HybridStats: ...
    def config(self) -> HybridConfig: ...

# =============================================================================
# Resource
# =============================================================================

class MemoryEstimate:
    """Memory estimate for a workload."""
    primary_bytes: int
    auxiliary_bytes: int
    peak_bytes: int
    confidence: float

    def __init__(
        self,
        primary_bytes: int = 0,
        auxiliary_bytes: int = 0,
        peak_bytes: int = 0,
        confidence: float = 1.0,
    ) -> None: ...

    @classmethod
    def with_primary(cls, bytes: int) -> MemoryEstimate: ...

    @classmethod
    def for_elements(cls, count: int, bytes_per_element: int) -> MemoryEstimate: ...

    def total_bytes(self) -> int: ...

class ReservationGuard(ContextManager['ReservationGuard']):
    """Reservation guard for guaranteed memory allocation."""
    @property
    def bytes(self) -> int: ...

    def is_committed(self) -> bool: ...
    def commit(self) -> None: ...
    def release(self) -> None: ...

class ResourceGuard(ContextManager['ResourceGuard']):
    """Resource guard for memory limit enforcement."""
    def __init__(
        self,
        max_memory_bytes: Optional[int] = None,
        safety_margin: float = 0.1,
    ) -> None: ...

    @classmethod
    def unguarded(cls) -> ResourceGuard: ...

    @classmethod
    def default(cls) -> ResourceGuard: ...

    @property
    def max_memory(self) -> int: ...
    @max_memory.setter
    def max_memory(self, bytes: int) -> None: ...
    @property
    def current_memory(self) -> int: ...
    @property
    def reserved_memory(self) -> int: ...

    def available_memory(self) -> int: ...
    def can_allocate(self, bytes: int) -> bool: ...
    def record_allocation(self, bytes: int) -> None: ...
    def record_deallocation(self, bytes: int) -> None: ...
    def reserve(self, bytes: int) -> ReservationGuard: ...
    def validate(self, estimate: MemoryEstimate) -> None: ...
    def max_safe_elements(self, bytes_per_element: int) -> int: ...
    def set_enforce_limits(self, enforce: bool) -> None: ...
    def is_enforcing(self) -> bool: ...
    def utilization(self) -> float: ...

def get_total_memory() -> Optional[int]: ...
def get_available_memory() -> Optional[int]: ...

# =============================================================================
# Exceptions
# =============================================================================

class RingKernelError(Exception):
    """Base exception for all RingKernel errors."""
    pass

class MemoryLimitError(RingKernelError):
    """Memory limit exceeded."""
    pass

class KernelError(RingKernelError):
    """Kernel operation error."""
    pass

class KernelStateError(KernelError):
    """Invalid kernel state transition."""
    pass

class CudaError(RingKernelError):
    """CUDA operation failed."""
    pass

class CudaDeviceError(CudaError):
    """CUDA device not available."""
    pass

class CudaMemoryError(CudaError):
    """CUDA memory operation failed."""
    pass

class QueueError(RingKernelError):
    """Queue operation failed."""
    pass

class QueueFullError(QueueError):
    """Queue is full."""
    pass

class QueueEmptyError(QueueError):
    """Queue is empty."""
    pass

class K2KError(RingKernelError):
    """K2K messaging error."""
    pass

class K2KDeliveryError(K2KError):
    """Message delivery failed."""
    pass

class BenchmarkError(RingKernelError):
    """Benchmark operation failed."""
    pass

class HybridError(RingKernelError):
    """Hybrid dispatch error."""
    pass

class GpuNotAvailableError(HybridError):
    """GPU not available."""
    pass

class ResourceError(RingKernelError):
    """Resource management error."""
    pass

class ReservationError(ResourceError):
    """Memory reservation failed."""
    pass

# =============================================================================
# Submodules
# =============================================================================

class hlc:
    HlcTimestamp = HlcTimestamp
    HlcClock = HlcClock

class k2k:
    K2KConfig = K2KConfig
    K2KBroker = K2KBroker
    K2KStats = K2KStats
    DeliveryStatus = DeliveryStatus
    DeliveryReceipt = DeliveryReceipt

class queue:
    QueueStats = QueueStats
    QueueTier = QueueTier
    QueueHealth = QueueHealth
    QueueMonitor = QueueMonitor
    QueueMetrics = QueueMetrics
    PartitionedQueueStats = PartitionedQueueStats

class hybrid:
    ProcessingMode = ProcessingMode
    HybridConfig = HybridConfig
    HybridStats = HybridStats
    HybridDispatcher = HybridDispatcher

class resource:
    MemoryEstimate = MemoryEstimate
    ReservationGuard = ReservationGuard
    ResourceGuard = ResourceGuard
    DEFAULT_MAX_MEMORY: int
    SYSTEM_MARGIN: int
    def get_total_memory() -> Optional[int]: ...
    def get_available_memory() -> Optional[int]: ...

class exceptions:
    RingKernelError = RingKernelError
    MemoryLimitError = MemoryLimitError
    KernelError = KernelError
    KernelStateError = KernelStateError
    CudaError = CudaError
    CudaDeviceError = CudaDeviceError
    CudaMemoryError = CudaMemoryError
    QueueError = QueueError
    QueueFullError = QueueFullError
    QueueEmptyError = QueueEmptyError
    K2KError = K2KError
    K2KDeliveryError = K2KDeliveryError
    BenchmarkError = BenchmarkError
    HybridError = HybridError
    GpuNotAvailableError = GpuNotAvailableError
    ResourceError = ResourceError
    ReservationError = ReservationError

# =============================================================================
# CUDA Module (optional)
# =============================================================================

def is_cuda_available() -> bool:
    """Check if CUDA is available on this system."""
    ...

class PyCudaDeviceInfo:
    """Information about a CUDA device."""
    ordinal: int
    name: str
    compute_capability: tuple[int, int]
    total_memory: int
    supports_persistent: bool

    def total_memory_mb(self) -> float: ...
    def total_memory_gb(self) -> float: ...

class PyCudaDevice:
    """A CUDA device for GPU computation."""
    def __init__(self, ordinal: int = 0) -> None: ...

    @property
    def ordinal(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def compute_capability(self) -> tuple[int, int]: ...
    @property
    def total_memory(self) -> int: ...

    def total_memory_mb(self) -> float: ...
    def supports_persistent_kernels(self) -> bool: ...
    def supports_cooperative_groups(self) -> bool: ...
    def synchronize(self) -> None: ...

class PyGpuSizeClass:
    """Size class for stratified memory pooling."""
    Size256B: int
    Size1KB: int
    Size4KB: int
    Size16KB: int
    Size64KB: int
    Size256KB: int

    def bytes(self) -> int: ...

    @classmethod
    def for_size(cls, bytes: int) -> Optional[PyGpuSizeClass]: ...

class PyGpuPoolConfig:
    """Configuration for GPU memory pool."""
    def __init__(self) -> None: ...

    @classmethod
    def for_graph_analytics(cls) -> PyGpuPoolConfig: ...

    @classmethod
    def for_simulation(cls) -> PyGpuPoolConfig: ...

    @classmethod
    def minimal(cls) -> PyGpuPoolConfig: ...

    @property
    def track_allocations(self) -> bool: ...
    @property
    def max_pool_bytes(self) -> int: ...

class PyGpuPoolDiagnostics:
    """Diagnostics for GPU memory pool."""
    total_cuda_bytes: int
    in_use_bytes: int
    free_bytes: int
    fragmentation: float
    large_allocation_count: int
    large_allocation_bytes: int
    total_allocations: int
    total_deallocations: int
    hit_rate: float

    def bucket_stats(self) -> List[PyGpuBucketStats]: ...
    def utilization(self) -> float: ...

class PyGpuBucketStats:
    """Per-bucket statistics for memory pool."""
    size_bytes: int
    total_blocks: int
    in_use_blocks: int
    free_blocks: int

    def utilization(self) -> float: ...

class PyStreamId:
    """Stream identifier."""
    @classmethod
    def compute(cls, index: int) -> PyStreamId: ...

    @classmethod
    def transfer(cls) -> PyStreamId: ...

    @classmethod
    def default_stream(cls) -> PyStreamId: ...

class PyStreamConfig:
    """Stream manager configuration."""
    def __init__(self) -> None: ...

    @classmethod
    def minimal(cls) -> PyStreamConfig: ...

    @classmethod
    def performance(cls) -> PyStreamConfig: ...

    @classmethod
    def for_simulation(cls) -> PyStreamConfig: ...

    @property
    def num_compute_streams(self) -> int: ...
    @property
    def use_transfer_stream(self) -> bool: ...
    @property
    def enable_graph_capture(self) -> bool: ...

class PyOverlapMetrics:
    """Compute/transfer overlap metrics."""
    compute_ns: int
    transfer_ns: int
    overlap_ns: int
    overlap_count: int

    def compute_ms(self) -> float: ...
    def transfer_ms(self) -> float: ...
    def overlap_ms(self) -> float: ...
    def overlap_efficiency(self) -> float: ...

class PyStreamPoolStats:
    """Stream pool statistics."""
    total_launches: int
    per_stream_launches: List[int]
    launches_per_second: float
    workload_count: int

    def most_utilized_stream(self) -> Optional[int]: ...
    def least_utilized_stream(self) -> Optional[int]: ...
    def balance_ratio(self) -> float: ...

class cuda:
    """CUDA submodule."""
    def is_cuda_available() -> bool: ...
    def cuda_device_count() -> int: ...
    def enumerate_devices() -> List[PyCudaDeviceInfo]: ...

    PyCudaDeviceInfo = PyCudaDeviceInfo
    PyCudaDevice = PyCudaDevice
    PyGpuSizeClass = PyGpuSizeClass
    PyGpuPoolConfig = PyGpuPoolConfig
    PyGpuPoolDiagnostics = PyGpuPoolDiagnostics
    PyGpuBucketStats = PyGpuBucketStats
    PyStreamId = PyStreamId
    PyStreamConfig = PyStreamConfig
    PyOverlapMetrics = PyOverlapMetrics
    PyStreamPoolStats = PyStreamPoolStats

# =============================================================================
# Benchmark Module (optional)
# =============================================================================

class PyBenchmarkConfig:
    """Configuration for benchmark execution."""
    def __init__(
        self,
        warmup_iterations: int = 5,
        measurement_iterations: int = 10,
        regression_threshold: float = 0.10,
    ) -> None: ...

    @classmethod
    def quick(cls) -> PyBenchmarkConfig: ...

    @classmethod
    def comprehensive(cls) -> PyBenchmarkConfig: ...

    @classmethod
    def ci(cls) -> PyBenchmarkConfig: ...

    def with_warmup(self, iterations: int) -> PyBenchmarkConfig: ...
    def with_measurements(self, iterations: int) -> PyBenchmarkConfig: ...
    def with_sizes(self, sizes: List[int]) -> PyBenchmarkConfig: ...
    def with_regression_threshold(self, threshold: float) -> PyBenchmarkConfig: ...
    def with_timeout_secs(self, seconds: float) -> PyBenchmarkConfig: ...

    @property
    def warmup_iterations(self) -> int: ...
    @property
    def measurement_iterations(self) -> int: ...
    @property
    def regression_threshold(self) -> float: ...
    @property
    def workload_sizes(self) -> List[int]: ...

class PyWorkloadConfig:
    """Configuration for individual workload execution."""
    def __init__(self, size: int) -> None: ...

    def with_convergence_threshold(self, threshold: float) -> PyWorkloadConfig: ...
    def with_max_iterations(self, max: int) -> PyWorkloadConfig: ...
    def with_param(self, key: str, value: str) -> PyWorkloadConfig: ...

    @property
    def size(self) -> int: ...
    @property
    def convergence_threshold(self) -> float: ...
    @property
    def max_iterations(self) -> int: ...

class PyBenchmarkResult:
    """Single benchmark execution result."""
    def __init__(self, workload_id: str, size: int, total_time_secs: float) -> None: ...

    @classmethod
    def from_measurements(
        cls,
        workload_id: str,
        size: int,
        measurement_times_secs: List[float],
        iterations: Optional[int] = None,
        converged: Optional[bool] = None,
    ) -> PyBenchmarkResult: ...

    def with_metric(self, key: str, value: float) -> PyBenchmarkResult: ...

    @property
    def workload_id(self) -> str: ...
    @property
    def size(self) -> int: ...
    @property
    def throughput_ops(self) -> float: ...
    @property
    def total_time_secs(self) -> float: ...
    @property
    def iterations(self) -> Optional[int]: ...
    @property
    def converged(self) -> Optional[bool]: ...

    def throughput_mops(self) -> float: ...
    def total_time_ms(self) -> float: ...
    def throughput_stddev(self) -> float: ...
    def custom_metrics(self) -> dict[str, float]: ...

class PyRegressionStatus:
    """Regression status classification."""
    Improved: int
    Unchanged: int
    Regressed: int

    def symbol(self) -> str: ...
    def text(self) -> str: ...

class PyRegressionEntry:
    """Single regression comparison entry."""
    workload_id: str
    size: int
    current_throughput: float
    baseline_throughput: float
    percent_change: float
    status: PyRegressionStatus

class PyRegressionReport:
    """Complete regression analysis report."""
    @property
    def regression_count(self) -> int: ...
    @property
    def improvement_count(self) -> int: ...
    @property
    def unchanged_count(self) -> int: ...
    @property
    def threshold(self) -> float: ...
    @property
    def overall_status(self) -> PyRegressionStatus: ...

    def has_regressions(self) -> bool: ...
    def total_comparisons(self) -> int: ...
    def entries(self) -> List[PyRegressionEntry]: ...
    def worst_regression(self) -> Optional[PyRegressionEntry]: ...
    def best_improvement(self) -> Optional[PyRegressionEntry]: ...
    def summary(self) -> str: ...

class PyBenchmarkBaseline:
    """Baseline for regression detection."""
    @classmethod
    def from_results(cls, results: List[PyBenchmarkResult], version: str) -> PyBenchmarkBaseline: ...

    @property
    def version(self) -> str: ...

    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
    def get(self, workload_id: str, size: int) -> Optional[PyBenchmarkResult]: ...

class PyBenchmarkSuite:
    """Main benchmark suite for running and tracking benchmarks."""
    def __init__(self, config: PyBenchmarkConfig) -> None: ...

    @classmethod
    def with_defaults(cls) -> PyBenchmarkSuite: ...

    def config(self) -> PyBenchmarkConfig: ...
    def results(self) -> List[PyBenchmarkResult]: ...
    def add_result(self, result: PyBenchmarkResult) -> None: ...
    def create_baseline(self, version: str) -> PyBenchmarkBaseline: ...
    def set_baseline(self, baseline: PyBenchmarkBaseline) -> None: ...
    def compare_to_baseline(self) -> Optional[PyRegressionReport]: ...
    def scaling_metrics_for(self, workload_id: str) -> PyScalingMetrics: ...
    def generate_markdown_report(self) -> str: ...
    def generate_json_export(self) -> str: ...
    def generate_latex_table(self) -> str: ...
    def clear_results(self) -> None: ...
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...

class PyConfidenceInterval:
    """Statistical confidence interval."""
    lower: float
    upper: float
    confidence_level: float

    @classmethod
    def from_values(cls, values: List[float]) -> PyConfidenceInterval: ...

    @classmethod
    def from_values_with_confidence(
        cls,
        values: List[float],
        confidence_level: float,
    ) -> PyConfidenceInterval: ...

    def width(self) -> float: ...
    def midpoint(self) -> float: ...

class PyDetailedStatistics:
    """Detailed statistics with percentiles."""
    count: int
    mean: float
    std_dev: float
    min: float
    max: float
    median: float
    p5: float
    p25: float
    p75: float
    p95: float
    p99: float

    @classmethod
    def from_values(cls, values: List[float]) -> PyDetailedStatistics: ...

    def coefficient_of_variation(self) -> float: ...
    def iqr(self) -> float: ...

class PyScalingMetrics:
    """Scaling behavior analysis."""
    exponent: float
    r_squared: float
    data_points: int

    @classmethod
    def from_sizes_and_throughputs(
        cls,
        sizes: List[int],
        throughputs: List[float],
    ) -> PyScalingMetrics: ...

    def scaling_quality(self) -> str: ...

class benchmark:
    """Benchmark submodule."""
    PyBenchmarkConfig = PyBenchmarkConfig
    PyWorkloadConfig = PyWorkloadConfig
    PyBenchmarkResult = PyBenchmarkResult
    PyBenchmarkBaseline = PyBenchmarkBaseline
    PyBenchmarkSuite = PyBenchmarkSuite
    PyConfidenceInterval = PyConfidenceInterval
    PyDetailedStatistics = PyDetailedStatistics
    PyScalingMetrics = PyScalingMetrics
    PyRegressionStatus = PyRegressionStatus
    PyRegressionEntry = PyRegressionEntry
    PyRegressionReport = PyRegressionReport
