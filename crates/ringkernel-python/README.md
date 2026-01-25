# ringkernel

Python bindings for the RingKernel GPU-native persistent actor framework.

## Features

- **GPU-Accelerated Actors**: Persistent GPU kernels with lock-free message passing
- **Hybrid Logical Clocks**: Causal ordering for distributed systems
- **K2K Messaging**: Direct kernel-to-kernel communication
- **Hybrid CPU/GPU Dispatch**: Intelligent workload routing with adaptive thresholds
- **Memory Management**: Resource guards and stratified GPU memory pools
- **Benchmarking**: Comprehensive benchmark suite with regression detection
- **CUDA Support**: Full NVIDIA GPU acceleration (optional feature)

## Installation

```bash
pip install ringkernel
```

For development:

```bash
cd crates/ringkernel-python
pip install maturin
maturin develop --features cuda,benchmark
```

## Quick Start

### Async Usage (Recommended)

```python
import asyncio
import ringkernel

async def main():
    # Create runtime with CPU backend
    runtime = await ringkernel.RingKernel.create(backend="cpu")

    # Launch kernel with custom options
    options = (
        ringkernel.LaunchOptions()
        .with_queue_capacity(2048)
        .with_block_size(256)
        .with_k2k(True)
    )
    kernel = await runtime.launch("processor", options)

    # Check kernel state
    print(f"Kernel ID: {kernel.id}")
    print(f"Kernel state: {kernel.state}")
    print(f"Is active: {kernel.is_active()}")

    # Get kernel status snapshot
    status = kernel.status()
    print(f"Messages processed: {status.messages_processed}")

    # Cleanup
    await kernel.terminate()
    await runtime.shutdown()

asyncio.run(main())
```

### Sync Usage

```python
import ringkernel

# Create runtime
runtime = ringkernel.RingKernel.create_sync(backend="cpu")

# Launch kernel
kernel = runtime.launch_sync("processor")

# Check state
print(f"Kernel {kernel.id} is active: {kernel.is_active()}")

# List all kernels
print(f"Active kernels: {runtime.list_kernels()}")

# View runtime metrics
metrics = runtime.metrics()
print(f"Total launched: {metrics.total_launched}")

# Cleanup
runtime.shutdown_sync()
```

### Working with Messages

```python
import ringkernel

# Create HLC timestamp
clock = ringkernel.HlcClock(node_id=1)
timestamp = clock.tick()

# Create message header
header = ringkernel.MessageHeader.new(
    type_id=1,
    source_kernel=0,
    dest_kernel=1000,
    payload_size=4,
    timestamp=timestamp,
)

# Create envelope with payload
payload = b"\x01\x02\x03\x04"
envelope = ringkernel.MessageEnvelope.from_bytes(header, payload)

# Access envelope properties
print(f"Header: {envelope.header}")
print(f"Payload: {envelope.payload}")
print(f"Total size: {envelope.total_size}")

# Serialize/deserialize
raw = envelope.to_bytes()
restored = ringkernel.MessageEnvelope.from_raw_bytes(raw)
```

## Hybrid Logical Clock (HLC)

HLC provides causal ordering for distributed kernel messages:

```python
from ringkernel import HlcClock, HlcTimestamp

# Create a clock for node 1
clock = HlcClock(node_id=1)

# Generate timestamps (always strictly increasing)
ts1 = clock.tick()
ts2 = clock.tick()
assert ts2 > ts1

# Read current time without advancing
current = clock.now()

# Create timestamp from components
ts = HlcTimestamp(physical=1000000, logical=5, node_id=1)
print(f"Physical: {ts.physical}")
print(f"Logical: {ts.logical}")
print(f"Node ID: {ts.node_id}")

# Timestamps from current time
ts_now = HlcTimestamp.now(node_id=42)
ts_zero = HlcTimestamp.zero()

# Update from received message (merge causality)
received = HlcTimestamp(physical=2000000, logical=0, node_id=2)
merged = clock.update(received)
assert merged > received

# Pack/unpack for atomic operations
packed = ts1.pack()  # 128-bit integer
unpacked = HlcTimestamp.unpack(packed)

# Time conversions
micros = ts1.as_micros()
millis = ts1.as_millis()
```

## K2K (Kernel-to-Kernel) Messaging

Configure kernel-to-kernel messaging through the broker:

```python
from ringkernel import K2KBroker, K2KConfig, K2KStats

# Create configuration
config = K2KConfig(
    max_pending_messages=2048,
    delivery_timeout_ms=10000,
    enable_tracing=True,
    max_hops=16,
)
# Or use defaults
config = K2KConfig.default()

# Create broker
broker = K2KBroker(config)

# Add routing rules
broker.add_route(destination="kernel_b", next_hop="kernel_a")

# Check registration (kernels are registered through runtime)
kernels = broker.registered_kernels()
print(f"Registered: {kernels}")

# View statistics
stats = broker.stats()
print(f"Endpoints: {stats.registered_endpoints}")
print(f"Messages delivered: {stats.messages_delivered}")
print(f"Routes configured: {stats.routes_configured}")

# Delivery status tracking
from ringkernel import DeliveryStatus

status = DeliveryStatus.Delivered
print(f"Success: {status.is_success()}")
print(f"Pending: {status.is_pending()}")
print(f"Failure: {status.is_failure()}")
```

## Hybrid Dispatcher

Route workloads between CPU and GPU based on size:

```python
from ringkernel import HybridDispatcher, HybridConfig, ProcessingMode

# CPU-only mode
config = HybridConfig.cpu_only()

# GPU-only mode (requires GPU)
config = HybridConfig.gpu_only()

# Adaptive mode (learns optimal threshold)
config = HybridConfig.adaptive()

# Custom configuration
config = HybridConfig(
    mode=ProcessingMode.Hybrid,
    gpu_threshold=10000,  # Use GPU above this size
    learning_rate=0.1,
    gpu_available=True,
)

# Create dispatcher
dispatcher = HybridDispatcher(config)

# Check routing decision
workload_size = 50000
if dispatcher.should_use_gpu(workload_size):
    result = execute_on_gpu(data)
    dispatcher.record_gpu_execution()
else:
    result = execute_on_cpu(data)
    dispatcher.record_cpu_execution()

# Update adaptive threshold based on timing
dispatcher.update_adaptive_threshold(
    workload_size=50000,
    cpu_time_ms=100.0,
    gpu_time_ms=10.0,
)

# View current threshold
print(f"Adaptive threshold: {dispatcher.adaptive_threshold}")

# View statistics
stats = dispatcher.stats()
print(f"CPU executions: {stats.cpu_count}")
print(f"GPU executions: {stats.gpu_count}")
print(f"Total: {stats.total()}")
print(f"GPU ratio: {stats.gpu_ratio():.1%}")
```

## Resource Guard

Prevent out-of-memory errors with resource tracking:

```python
from ringkernel import ResourceGuard, MemoryEstimate

# Create guard with 4GB limit and 10% safety margin
guard = ResourceGuard(max_memory_bytes=4 * 1024**3, safety_margin=0.1)

# Or use defaults (80% of system memory)
guard = ResourceGuard.default()

# Unguarded mode (no limits)
guard = ResourceGuard.unguarded()

# Check before allocation
requested = 1_000_000
if guard.can_allocate(requested):
    buffer = allocate(requested)
    guard.record_allocation(requested)

# Track deallocation
guard.record_deallocation(requested)

# View memory state
print(f"Max memory: {guard.max_memory}")
print(f"Current: {guard.current_memory}")
print(f"Reserved: {guard.reserved_memory}")
print(f"Available: {guard.available_memory()}")
print(f"Utilization: {guard.utilization():.1%}")

# Reserve memory (RAII pattern)
with guard.reserve(1_000_000) as reservation:
    print(f"Reserved: {reservation.bytes}")
    do_work()
    reservation.commit()  # Convert to permanent allocation
# Auto-released if not committed or on exception

# Memory estimation
estimate = MemoryEstimate(
    primary_bytes=1_000_000,
    auxiliary_bytes=500_000,
    peak_bytes=2_000_000,
    confidence=0.9,
)
print(f"Total: {estimate.total_bytes()}")  # primary + auxiliary

# Validate workload fits
guard.validate(estimate)  # Raises MemoryLimitError if insufficient

# Calculate safe element count
max_elements = guard.max_safe_elements(bytes_per_element=8)
```

## Queue Monitoring

Monitor queue health and get capacity recommendations:

```python
from ringkernel import QueueMonitor, QueueTier, QueueHealth, QueueStats

# Create monitor with thresholds
monitor = QueueMonitor(warning_threshold=0.75, critical_threshold=0.90)

# Or use defaults (75% warning, 90% critical)
monitor = QueueMonitor.default()

# Check utilization
util = monitor.utilization(depth=800, capacity=1000)
print(f"Utilization: {util:.1%}")

# Check health status
health = monitor.check_health(depth=800, capacity=1000)
if health == QueueHealth.Healthy:
    print("Queue is healthy")
elif health == QueueHealth.Warning:
    print("Queue utilization is high")
elif health == QueueHealth.Critical:
    print("Queue is near capacity!")

# Queue tiers with capacity
print(f"Small: {QueueTier.Small.capacity}")      # 256
print(f"Medium: {QueueTier.Medium.capacity}")    # 1024
print(f"Large: {QueueTier.Large.capacity}")      # 4096
print(f"ExtraLarge: {QueueTier.ExtraLarge.capacity}")  # 16384

# Get tier recommendation based on throughput
tier = QueueTier.for_throughput(
    messages_per_second=10000,
    headroom_ms=100,
)
print(f"Recommended: {tier}")

# Tier upgrades/downgrades
next_tier = QueueTier.Medium.upgrade()    # Large
prev_tier = QueueTier.Large.downgrade()   # Medium

# Get upgrade suggestion
suggested = monitor.suggest_upgrade(
    depth=900,
    capacity=1000,
    current_tier=QueueTier.Medium,
)
if suggested:
    print(f"Consider upgrading to: {suggested}")
```

## CUDA Support

GPU device management and memory pooling (requires `cuda` feature):

```python
import ringkernel

# Check CUDA availability
if ringkernel.is_cuda_available():
    from ringkernel.cuda import (
        cuda_device_count,
        enumerate_devices,
        PyCudaDevice,
        PyGpuPoolConfig,
        PyStreamConfig,
    )

    # Enumerate devices
    print(f"CUDA devices: {cuda_device_count()}")
    for info in enumerate_devices():
        print(f"  {info.name} (CC {info.compute_capability})")
        print(f"    Memory: {info.total_memory / 1e9:.1f} GB")
        print(f"    Persistent: {info.supports_persistent}")

    # Create device wrapper
    device = PyCudaDevice(ordinal=0)
    print(f"Device: {device.name}")
    print(f"Compute capability: {device.compute_capability}")
    print(f"Total memory: {device.total_memory_mb():.0f} MB")
    print(f"Supports persistent: {device.supports_persistent_kernels()}")
    print(f"Supports cooperative: {device.supports_cooperative_groups()}")

    # Synchronize device
    device.synchronize()

    # GPU memory pool configuration
    config = PyGpuPoolConfig.for_graph_analytics()  # 256B-heavy workloads
    config = PyGpuPoolConfig.for_simulation()       # Larger allocations
    config = PyGpuPoolConfig.minimal()              # Testing

    print(f"Track allocations: {config.track_allocations}")
    print(f"Max pool bytes: {config.max_pool_bytes}")

    # Stream configuration
    stream_config = PyStreamConfig.performance()  # 4 compute + transfer
    stream_config = PyStreamConfig.minimal()      # 1 stream
    stream_config = PyStreamConfig.for_simulation()

    print(f"Compute streams: {stream_config.num_compute_streams}")
    print(f"Transfer stream: {stream_config.use_transfer_stream}")
    print(f"Graph capture: {stream_config.enable_graph_capture}")
```

## Benchmarking

Comprehensive benchmark suite with regression detection (requires `benchmark` feature):

```python
from ringkernel.benchmark import (
    PyBenchmarkConfig,
    PyBenchmarkSuite,
    PyBenchmarkResult,
    PyConfidenceInterval,
    PyDetailedStatistics,
    PyScalingMetrics,
)

# Configuration presets
config = PyBenchmarkConfig.quick()          # 1 warmup, 3 measurements
config = PyBenchmarkConfig.comprehensive()  # 5 warmup, 10 measurements
config = PyBenchmarkConfig.ci()             # 2 warmup, 5 measurements

# Custom configuration
config = PyBenchmarkConfig(
    warmup_iterations=5,
    measurement_iterations=10,
    regression_threshold=0.10,  # 10% threshold
)

# Builder pattern
config = (
    PyBenchmarkConfig.quick()
    .with_warmup(3)
    .with_measurements(10)
    .with_sizes([1000, 10000, 100000])
    .with_regression_threshold(0.15)
    .with_timeout_secs(30.0)
)

# Create suite
suite = PyBenchmarkSuite(config)

# Create result from single run
result = PyBenchmarkResult(
    workload_id="matrix_multiply",
    size=1000,
    total_time_secs=0.5,
)
print(f"Throughput: {result.throughput_ops} ops/s")
print(f"Throughput: {result.throughput_mops()} Mops/s")
print(f"Time: {result.total_time_ms()} ms")

# Create result from multiple measurements
result = PyBenchmarkResult.from_measurements(
    workload_id="matrix_multiply",
    size=1000,
    measurement_times_secs=[0.48, 0.52, 0.49, 0.51, 0.50],
    iterations=100,      # Optional: for iterative algorithms
    converged=True,      # Optional: convergence status
)
print(f"Stddev: {result.throughput_stddev()}")

# Add custom metrics
result = result.with_metric("memory_mb", 256.5)
print(f"Custom metrics: {result.custom_metrics()}")

# Add results to suite
suite.add_result(result)

# Generate reports
markdown = suite.generate_markdown_report()
json_str = suite.generate_json_export()
latex = suite.generate_latex_table()

print(markdown)

# Create baseline for regression detection
baseline = suite.create_baseline("v1.0.0")
print(f"Baseline version: {baseline.version}")
print(f"Baseline results: {len(baseline)}")

# Compare against baseline
suite.set_baseline(baseline)
report = suite.compare_to_baseline()
if report:
    print(f"Regressions: {report.regression_count}")
    print(f"Improvements: {report.improvement_count}")
    print(f"Unchanged: {report.unchanged_count}")
    print(f"Has regressions: {report.has_regressions()}")
    print(f"Summary: {report.summary()}")

    # Get worst/best
    worst = report.worst_regression()
    best = report.best_improvement()
    if worst:
        print(f"Worst: {worst.workload_id} ({worst.percent_change:+.1%})")

# Statistical analysis
stats = PyDetailedStatistics.from_values([0.48, 0.52, 0.49, 0.51, 0.50])
print(f"Mean: {stats.mean}")
print(f"Std dev: {stats.std_dev}")
print(f"Median: {stats.median}")
print(f"P95: {stats.p95}")
print(f"IQR: {stats.iqr()}")
print(f"CV: {stats.coefficient_of_variation()}")

# Confidence intervals
ci = PyConfidenceInterval.from_values([0.48, 0.52, 0.49, 0.51, 0.50])
print(f"95% CI: [{ci.lower}, {ci.upper}]")
print(f"Width: {ci.width()}")

# Scaling analysis
scaling = PyScalingMetrics.from_sizes_and_throughputs(
    sizes=[1000, 10000, 100000],
    throughputs=[1e6, 8e6, 50e6],
)
print(f"Scaling exponent: {scaling.exponent}")
print(f"R-squared: {scaling.r_squared}")
print(f"Quality: {scaling.scaling_quality()}")  # Excellent/Good/Fair/Poor
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `RingKernel` | Main runtime for managing GPU kernels |
| `KernelHandle` | Handle for interacting with launched kernels |
| `LaunchOptions` | Kernel launch configuration |
| `KernelState` | Kernel lifecycle state enum |
| `KernelMode` | Kernel execution mode (Persistent/EventDriven) |
| `KernelStatus` | Kernel status snapshot |
| `RuntimeMetrics` | Runtime-wide metrics |
| `Backend` | GPU backend enum (Auto/Cpu/Cuda/Metal/Wgpu) |

### Message Types

| Type | Description |
|------|-------------|
| `MessageId` | Unique message identifier |
| `CorrelationId` | Request-response correlation |
| `Priority` | Message priority (Low/Normal/High/Critical) |
| `MessageHeader` | 256-byte message header |
| `MessageEnvelope` | Header + payload wrapper |

### HLC Types

| Type | Description |
|------|-------------|
| `HlcTimestamp` | Hybrid logical clock timestamp |
| `HlcClock` | Clock for generating timestamps |

### K2K Types

| Type | Description |
|------|-------------|
| `K2KBroker` | Message routing broker |
| `K2KConfig` | Broker configuration |
| `K2KStats` | Broker statistics |
| `DeliveryStatus` | Message delivery status |
| `DeliveryReceipt` | Delivery confirmation |

### Hybrid Dispatch

| Type | Description |
|------|-------------|
| `HybridDispatcher` | CPU/GPU workload router |
| `HybridConfig` | Dispatcher configuration |
| `HybridStats` | Execution statistics |
| `ProcessingMode` | Routing mode enum |

### Resource Management

| Type | Description |
|------|-------------|
| `ResourceGuard` | Memory limit enforcement |
| `MemoryEstimate` | Workload memory prediction |

### Queue Monitoring

| Type | Description |
|------|-------------|
| `QueueStats` | Queue statistics |
| `QueueTier` | Queue capacity tier |
| `QueueHealth` | Health status enum |
| `QueueMonitor` | Health monitoring |
| `QueueMetrics` | Complete metrics snapshot |

### CUDA Types (feature-gated)

| Type | Description |
|------|-------------|
| `PyCudaDevice` | CUDA device wrapper |
| `PyCudaDeviceInfo` | Device information |
| `PyGpuPoolConfig` | Memory pool configuration |
| `PyGpuPoolDiagnostics` | Pool statistics |
| `PyGpuSizeClass` | Memory size classes |
| `PyStreamConfig` | Stream configuration |
| `PyStreamId` | Stream identifier |

### Benchmark Types (feature-gated)

| Type | Description |
|------|-------------|
| `PyBenchmarkConfig` | Benchmark configuration |
| `PyBenchmarkSuite` | Benchmark orchestration |
| `PyBenchmarkResult` | Single result |
| `PyBenchmarkBaseline` | Regression baseline |
| `PyRegressionReport` | Regression analysis |
| `PyConfidenceInterval` | Statistical CI |
| `PyDetailedStatistics` | Percentile statistics |
| `PyScalingMetrics` | Scaling analysis |

### Exceptions

| Exception | Description |
|-----------|-------------|
| `RingKernelError` | Base exception |
| `CudaError` | CUDA operation failed |
| `KernelError` | Kernel operation error |
| `MemoryLimitError` | Memory limit exceeded |

## Building from Source

Requirements:
- Rust 1.70+
- Python 3.8+
- maturin (`pip install maturin`)

```bash
# Development build (CPU only)
maturin develop

# Development build with all features
maturin develop --features cuda,benchmark

# Release build
maturin build --release

# Release wheel with CUDA support
maturin build --release --features cuda,benchmark

# Run tests
pytest tests/ -v
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `cuda` | NVIDIA CUDA GPU support |
| `benchmark` | Benchmarking with regression detection |
| `cuda-profiling` | GPU profiling (NVTX, events) |
| `numpy` | NumPy array interop |

## License

Apache-2.0
