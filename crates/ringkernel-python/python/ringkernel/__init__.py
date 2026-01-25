"""
RingKernel - GPU-native persistent actor model framework.

This package provides Python bindings for RingKernel, enabling GPU-accelerated
actor systems with persistent kernels and lock-free message passing.

Example:
    >>> import asyncio
    >>> import ringkernel
    >>>
    >>> async def main():
    ...     # Create runtime
    ...     runtime = await ringkernel.RingKernel.create(backend="cpu")
    ...
    ...     # Launch kernel
    ...     options = ringkernel.LaunchOptions().with_queue_capacity(2048)
    ...     kernel = await runtime.launch("processor", options)
    ...
    ...     # Interact with kernel
    ...     print(f"Kernel state: {kernel.state}")
    ...
    ...     # Cleanup
    ...     await kernel.terminate()
    ...     await runtime.shutdown()
    >>>
    >>> asyncio.run(main())

Sync usage:
    >>> runtime = ringkernel.RingKernel.create_sync(backend="cpu")
    >>> kernel = runtime.launch_sync("processor")
    >>> kernel.terminate_sync() # if needed
    >>> runtime.shutdown_sync()
"""

from ringkernel._ringkernel import (
    MAX_PAYLOAD_SIZE,
    MESSAGE_MAGIC,
    MESSAGE_VERSION,
    CudaError,
    KernelError,
    MemoryLimitError,
    PyBackend as Backend,
    PyCorrelationId as CorrelationId,
    PyHlcClock as HlcClock,
    # HLC
    PyHlcTimestamp as HlcTimestamp,
    # Hybrid (types exported at top level)
    PyHybridDispatcher as HybridDispatcher,
    # K2K (types exported at top level)
    PyK2KBroker as K2KBroker,
    PyK2KConfig as K2KConfig,
    PyKernelHandle as KernelHandle,
    PyKernelMode as KernelMode,
    PyKernelState as KernelState,
    PyKernelStatus as KernelStatus,
    PyLaunchOptions as LaunchOptions,
    PyMemoryEstimate as MemoryEstimate,
    PyMessageEnvelope as MessageEnvelope,
    PyMessageHeader as MessageHeader,
    # Messages
    PyMessageId as MessageId,
    PyPriority as Priority,
    PyProcessingMode as ProcessingMode,
    # Queue (types exported at top level)
    PyQueueStats as QueueStats,
    PyQueueTier as QueueTier,
    # Resource
    PyResourceGuard as ResourceGuard,
    # Core runtime (aliased for Pythonic naming)
    PyRingKernel as RingKernel,
    PyRuntimeMetrics as RuntimeMetrics,
    # Exceptions
    RingKernelError,
    # Version
    __version__,
    # CUDA
    is_cuda_available,
)

# Import optional CUDA device at top level for convenience
try:
    from ringkernel._ringkernel import PyCudaDevice as CudaDevice
except ImportError:
    CudaDevice = None

# Re-export submodules for convenience
from ringkernel._ringkernel import exceptions, hlc, hybrid, k2k, queue, resource

# Additional types from submodules (aliased for convenience)
# These are also available via submodules: ringkernel.k2k.PyDeliveryStatus, etc.
K2KStats = k2k.PyK2KStats
DeliveryStatus = k2k.PyDeliveryStatus
DeliveryReceipt = k2k.PyDeliveryReceipt
QueueHealth = queue.PyQueueHealth
QueueMonitor = queue.PyQueueMonitor
QueueMetrics = queue.PyQueueMetrics
HybridConfig = hybrid.PyHybridConfig
HybridStats = hybrid.PyHybridStats

# Optional submodules
try:
    from ringkernel._ringkernel import cuda
except ImportError:
    cuda = None

try:
    from ringkernel._ringkernel import benchmark
except ImportError:
    benchmark = None

__all__ = [
    # Version
    "__version__",

    # Core runtime
    "RingKernel",
    "KernelHandle",
    "LaunchOptions",
    "KernelState",
    "KernelMode",
    "KernelStatus",
    "RuntimeMetrics",
    "Backend",

    # Messages
    "MessageId",
    "CorrelationId",
    "Priority",
    "MessageHeader",
    "MessageEnvelope",
    "MAX_PAYLOAD_SIZE",
    "MESSAGE_MAGIC",
    "MESSAGE_VERSION",

    # HLC
    "HlcTimestamp",
    "HlcClock",

    # K2K
    "K2KBroker",
    "K2KConfig",
    "K2KStats",
    "DeliveryStatus",
    "DeliveryReceipt",

    # Queue
    "QueueStats",
    "QueueTier",
    "QueueHealth",
    "QueueMonitor",
    "QueueMetrics",

    # Hybrid
    "HybridDispatcher",
    "ProcessingMode",
    "HybridConfig",
    "HybridStats",

    # Resource
    "ResourceGuard",
    "MemoryEstimate",

    # Exceptions
    "RingKernelError",
    "CudaError",
    "KernelError",
    "MemoryLimitError",

    # CUDA
    "is_cuda_available",
    "CudaDevice",

    # Submodules
    "hlc",
    "k2k",
    "queue",
    "hybrid",
    "resource",
    "exceptions",
    "cuda",
    "benchmark",
]
