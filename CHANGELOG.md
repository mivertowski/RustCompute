# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-01-25

### Added

#### GPU Infrastructure Generalization (RustGraph → RingKernel)

This release extracts ~7,000+ lines of proven GPU infrastructure from RustGraph into RingKernel, making these capabilities available to all RingKernel users.

#### Python Bindings (`ringkernel-python`) - **NEW CRATE**

- **PyO3-based Python wrapper** providing Pythonic access to RingKernel
  - Full async/await support with `pyo3-async-runtimes` and tokio integration
  - Sync fallbacks for all async operations (`create_sync`, `launch_sync`, etc.)
  - Type stubs (`.pyi` files) for IDE support and static type checking
  - Python 3.8+ compatibility via `abi3-py38`

- **Core Runtime API**:
  - `RingKernel.create()` / `create_sync()` - Create runtime with backend selection
  - `KernelHandle` - Launch, activate, deactivate, terminate kernels
  - `LaunchOptions` - Configure queue capacity, block size, priority
  - `MessageId`, `MessageEnvelope` - Message handling primitives
  - `HlcTimestamp`, `HlcClock` - Hybrid Logical Clock support
  - `K2KBroker`, `K2KEndpoint` - Kernel-to-kernel messaging
  - `QueueStats` - Queue monitoring and statistics

- **CUDA Support** (feature-gated via `cuda`):
  - `CudaDevice` - Device enumeration and properties
  - `GpuMemoryPool` - Stratified GPU memory pool management
  - `StreamManager` - Multi-stream execution management
  - `ProfilingSession` - GPU profiling and metrics collection

- **Benchmark Framework** (feature-gated via `benchmark`):
  - `BenchmarkSuite`, `BenchmarkConfig` - Comprehensive benchmarking
  - `BenchmarkResult` - Results with throughput and timing
  - Regression detection with baseline comparison
  - Multiple report formats (Markdown, JSON, LaTeX)

- **Hybrid Dispatcher**:
  - `HybridDispatcher` - Automatic CPU/GPU workload routing
  - `HybridConfig`, `ProcessingMode` - Configuration with adaptive thresholds
  - `HybridStats` - Execution statistics and threshold learning

- **Resource Management**:
  - `ResourceGuard` - Memory limit enforcement with safety margins
  - `ReservationGuard` - RAII wrapper for guaranteed allocations
  - `MemoryEstimate` - Workload memory estimation

```python
import ringkernel
import asyncio

async def main():
    runtime = await ringkernel.RingKernel.create(backend="cpu")
    kernel = await runtime.launch("processor", ringkernel.LaunchOptions())
    await kernel.terminate()
    await runtime.shutdown()

asyncio.run(main())
```

#### PTX Compilation Cache (`ringkernel-cuda/src/compile/`) - **NEW MODULE**

- **`PtxCache`** - Disk-based PTX compilation cache for faster kernel loading
  - SHA-256 content-based hashing for cache keys
  - Compute capability-aware caching (separate cache per GPU architecture)
  - Thread-safe with atomic file operations
  - Environment variable support: `RINGKERNEL_PTX_CACHE_DIR`
  - `PtxCacheStats` for hit/miss tracking
  - `PtxCacheError` with descriptive error types
  - Default cache location: `~/.cache/ringkernel/ptx/`

```rust
use ringkernel_cuda::compile::{PtxCache, PtxCacheStats};

let cache = PtxCache::new()?;  // Uses default directory
let hash = PtxCache::hash_source(cuda_source);

// Check cache first
if let Some(ptx) = cache.get(&hash, "sm_89")? {
    // Use cached PTX
} else {
    let ptx = compile_ptx(cuda_source)?;
    cache.put(&hash, "sm_89", &ptx)?;
}

println!("Cache stats: {:?}", cache.stats());
```

#### GPU Stratified Memory Pool (`ringkernel-cuda/src/memory_pool.rs`) - **NEW FILE**

- **`GpuStratifiedPool`** - Size-stratified memory pool for GPU VRAM
  - 6 size classes: 256B, 1KB, 4KB, 16KB, 64KB, 256KB
  - O(1) allocation from free lists per bucket
  - Large allocation fallback for oversized requests
  - Thread-safe with atomic counters
  - `GpuPoolConfig` with presets: `for_graph_analytics()`, `for_simulation()`
  - `GpuPoolDiagnostics` for monitoring utilization
  - `warm_bucket()` for pre-allocation
  - `compact()` for memory defragmentation

```rust
use ringkernel_cuda::memory_pool::{GpuStratifiedPool, GpuPoolConfig, GpuSizeClass};

let config = GpuPoolConfig::for_graph_analytics();  // 256B-heavy
let mut pool = GpuStratifiedPool::new(&device, config)?;

// Warm the small buffer bucket
pool.warm_bucket(GpuSizeClass::Size1KB, 100)?;

// Allocate (O(1) for pooled sizes)
let ptr = pool.allocate(512)?;  // Uses 1KB bucket
pool.deallocate(ptr, 512)?;

println!("Diagnostics: {:?}", pool.diagnostics());
```

#### Multi-Stream Execution Manager (`ringkernel-cuda/src/stream/`) - **NEW MODULE**

- **`StreamManager`** - Multi-stream CUDA execution for compute/transfer overlap
  - Configurable compute streams (1-8) with priority support
  - Dedicated transfer stream for async DMA
  - Event-based inter-stream synchronization
  - `StreamConfig` with presets: `minimal()`, `performance()`
  - `StreamId` enum: `Compute(usize)`, `Transfer`, `Default`
  - `record_event()` / `stream_wait_event()` for dependencies
  - `event_elapsed_ms()` for timing measurements

- **`StreamPool`** - Load-balanced stream assignment
  - `assign_workload()` for explicit assignment
  - `least_utilized()` for automatic load balancing
  - Utilization tracking with atomic counters
  - `StreamPoolStats` for monitoring

- **`OverlapMetrics`** - Compute/transfer overlap measurement
  - Overlap ratio calculation
  - Transfer/compute time tracking

```rust
use ringkernel_cuda::stream::{StreamManager, StreamConfig, StreamId};

let config = StreamConfig::performance();  // 4 compute + transfer
let mut manager = StreamManager::new(&device, config)?;

// Launch kernel on compute stream
let compute_stream = manager.cuda_stream(StreamId::Compute(0))?;
// ... launch kernel ...

// Record event for synchronization
manager.record_event("kernel_done", StreamId::Compute(0))?;

// Transfer stream waits for kernel
manager.stream_wait_event(StreamId::Transfer, "kernel_done")?;

// Timing
let elapsed = manager.event_elapsed_ms("start", "kernel_done")?;
```

#### Benchmark Framework (`ringkernel-core/src/benchmark/`) - **NEW MODULE**

- **`Benchmarkable` trait** - Generic interface for benchmarkable workloads
  - `name()` / `code()` for identification
  - `execute()` for workload execution
  - Supports custom workload sizes

- **`BenchmarkSuite`** - Comprehensive benchmark orchestration
  - `run()` / `run_all_sizes()` for execution
  - Baseline comparison with `set_baseline()` / `compare_to_baseline()`
  - Multiple report formats: Markdown, LaTeX, JSON

- **`BenchmarkConfig`** - Benchmark configuration
  - Warmup/measurement iterations
  - Convergence thresholds
  - Configurable workload sizes
  - Presets: `quick()`, `comprehensive()`, `ci()`

- **`BenchmarkResult`** - Detailed benchmark results
  - Throughput (ops/s), total time, iterations
  - Per-measurement timing data
  - Custom metrics support
  - Convergence tracking

- **`RegressionReport`** - Performance regression detection
  - Per-workload comparison to baseline
  - Status: Regression, Improvement, Unchanged
  - Configurable threshold (default: 5%)

- **`Statistics`** - Statistical analysis utilities
  - `ConfidenceInterval` with configurable confidence level
  - `DetailedStatistics`: mean, std_dev, min, max, percentiles (p5, p25, median, p75, p95, p99)
  - `ScalingMetrics` for analyzing algorithmic scaling (exponent, R²)

```rust
use ringkernel_core::benchmark::{BenchmarkSuite, BenchmarkConfig, Benchmarkable};

struct MyWorkload;
impl Benchmarkable for MyWorkload {
    fn name(&self) -> &str { "MyWorkload" }
    fn code(&self) -> &str { "MW" }
    fn execute(&self, config: &WorkloadConfig) -> BenchmarkResult {
        // ... run workload ...
    }
}

let config = BenchmarkConfig::comprehensive()
    .with_sizes(vec![1000, 10_000, 100_000]);
let mut suite = BenchmarkSuite::new(config);

suite.run_all_sizes(&MyWorkload);

// Generate reports
println!("{}", suite.generate_markdown_report());
println!("{}", suite.generate_latex_table());

// Regression detection
let baseline = suite.create_baseline("v1.0");
suite.set_baseline(baseline);
if let Some(report) = suite.compare_to_baseline() {
    println!("Regressions: {}", report.regression_count);
}
```

#### Hybrid CPU-GPU Dispatcher (`ringkernel-core/src/hybrid/`) - **NEW MODULE**

- **`HybridDispatcher`** - Intelligent CPU/GPU workload routing
  - Automatic threshold-based routing
  - Adaptive threshold learning from execution times
  - Configurable learning rate
  - Fallback to CPU when GPU unavailable

- **`HybridWorkload` trait** - Workload interface for hybrid execution
  - `execute_cpu()` / `execute_gpu()` implementations
  - `workload_size()` for routing decisions
  - `supports_gpu()` for capability detection
  - `memory_estimate()` for resource planning

- **`ProcessingMode`** - Routing mode configuration
  - `GpuOnly` - Always use GPU
  - `CpuOnly` - Always use CPU
  - `Hybrid { gpu_threshold }` - Size-based routing
  - `Adaptive` - Learn optimal threshold

- **`HybridConfig`** - Dispatcher configuration
  - Learning rate, initial threshold, min/max thresholds
  - GPU availability flag
  - Presets: `cpu_only()`, `gpu_only()`, `adaptive()`, `for_small_workloads()`, `for_large_workloads()`

- **`HybridStats`** - Execution statistics
  - CPU/GPU execution counts and times
  - Adaptive threshold history
  - `cpu_gpu_ratio()` for balance analysis

```rust
use ringkernel_core::hybrid::{HybridDispatcher, HybridConfig, HybridWorkload, ProcessingMode};

struct MatrixMultiply { size: usize, /* ... */ }
impl HybridWorkload for MatrixMultiply {
    type Result = Matrix;
    fn workload_size(&self) -> usize { self.size * self.size }
    fn execute_cpu(&self) -> Matrix { /* CPU impl */ }
    fn execute_gpu(&self) -> HybridResult<Matrix> { /* GPU impl */ }
}

let config = HybridConfig::adaptive()
    .with_initial_threshold(10_000)
    .with_learning_rate(0.1);
let dispatcher = HybridDispatcher::new(config);

let workload = MatrixMultiply { size: 1000 };

// Automatic routing based on size and learned threshold
let result = dispatcher.execute(&workload);

// Check stats
let stats = dispatcher.stats().snapshot();
println!("GPU executions: {}, CPU executions: {}", stats.gpu_executions, stats.cpu_executions);
```

#### Resource Guard (`ringkernel-core/src/resource/`) - **NEW MODULE**

- **`ResourceGuard`** - Memory limit enforcement with reservations
  - Configurable maximum memory
  - Safety margin (default: 30%)
  - Reservation system for guaranteed allocations
  - `can_allocate()` for pre-flight checks
  - `reserve()` returns `ReservationGuard` RAII wrapper
  - `max_safe_elements()` for capacity planning
  - `unguarded()` for unlimited allocation mode
  - `global_guard()` singleton for process-wide limits

- **`MemoryEstimator` trait** - Workload memory estimation
  - `estimate()` returns `MemoryEstimate`
  - `name()` for identification

- **`MemoryEstimate`** - Detailed memory requirements
  - Primary, auxiliary, and peak bytes
  - Confidence level (0.0-1.0)
  - `total_bytes()` / `peak_bytes()` helpers
  - Builder pattern with `with_primary()`, `with_auxiliary()`, etc.

- **`LinearEstimator`** - Simple linear memory estimator
  - Bytes per element + fixed overhead

- **System utilities**:
  - `get_total_memory()` - System RAM
  - `get_available_memory()` - Free RAM
  - `get_memory_utilization()` - Current usage percentage

```rust
use ringkernel_core::resource::{ResourceGuard, MemoryEstimate, MemoryEstimator};

let guard = ResourceGuard::with_max_memory(4 * 1024 * 1024 * 1024);  // 4 GB

// Check before allocating
if guard.can_allocate(1024 * 1024 * 1024) {
    // Safe to allocate 1 GB
}

// Reserve memory with RAII guard
let reservation = guard.reserve(512 * 1024 * 1024)?;
// ... use reserved memory ...
// Automatically released when reservation drops

// Calculate safe element count
let max_elements = guard.max_safe_elements(64);  // 64 bytes per element
println!("Can safely process {} elements", max_elements);
```

#### Kernel Mode Selection (`ringkernel-cuda/src/launch_config/`) - **NEW MODULE**

- **`KernelMode`** - Execution mode selection
  - `ElementCentric` - One thread per element (default)
  - `SoA` - Structure-of-Arrays for coalesced access
  - `WorkItemCentric` - Load-balanced work distribution
  - `Tiled { tile_size }` - Tiled execution with configurable tile dimensions
  - `WarpCooperative` - Warp-level parallelism
  - `Auto` - Automatic selection based on workload

- **`AccessPattern`** - Memory access pattern hints
  - `Coalesced` - Sequential access
  - `Stencil { radius }` - Stencil patterns with halo
  - `Irregular` - Random access
  - `Reduction` - Reduction operations
  - `Scatter` / `Gather` - Indirect access

- **`WorkloadProfile`** - Workload characteristics
  - Element count, bytes per element
  - Access pattern, compute intensity
  - Builder pattern for configuration

- **`GpuArchitecture`** - GPU capability profiles
  - L2 cache size, SM count, max threads/SM
  - Shared memory per SM
  - Compute capability
  - Presets: `volta()`, `ampere()`, `ada()`, `hopper()`

- **`KernelModeSelector`** - Intelligent mode selection
  - `select()` chooses optimal mode for workload
  - `recommended_block_size()` per mode
  - `recommended_grid_size()` for element count
  - `launch_config()` returns complete `LaunchConfig`

- **`LaunchConfig`** - Complete kernel launch configuration
  - Grid dimensions, block dimensions
  - Shared memory bytes
  - `simple_1d()` / `simple_2d()` helpers

```rust
use ringkernel_cuda::launch_config::{
    KernelModeSelector, WorkloadProfile, AccessPattern, GpuArchitecture,
};

let arch = GpuArchitecture::ada();  // RTX 40xx
let selector = KernelModeSelector::new(arch);

let profile = WorkloadProfile::new(1_000_000, 64)
    .with_access_pattern(AccessPattern::Stencil { radius: 1 })
    .with_compute_intensity(0.8);

let mode = selector.select(&profile);  // Returns Tiled for stencil
let config = selector.launch_config(mode, profile.element_count);

println!("Grid: {:?}, Block: {:?}", config.grid_dim, config.block_dim);
```

#### Partitioned Queues (`ringkernel-core/src/queue.rs`)

- **`PartitionedQueue`** - Multi-partition queue for reduced contention
  - Hash-based message routing by source kernel ID
  - Configurable partition count (rounded to power of 2)
  - `try_enqueue()` routes to appropriate partition
  - `try_dequeue_any()` round-robin across partitions
  - `try_dequeue_partition()` for targeted dequeue
  - `partition_for()` returns partition index for source

- **`PartitionedQueueStats`** - Partition-level statistics
  - Per-partition message counts
  - `load_imbalance()` metric (max/avg ratio)
  - Total message count across all partitions

```rust
use ringkernel_core::queue::PartitionedQueue;

let queue = PartitionedQueue::new(4, 1024);  // 4 partitions, 1024 capacity each

// Enqueue routes based on source kernel ID
queue.try_enqueue(envelope)?;  // Uses envelope.header.source_kernel for routing

// Dequeue from any partition (round-robin)
if let Some(msg) = queue.try_dequeue_any() {
    // Process message
}

// Check load balance
let stats = queue.stats();
println!("Load imbalance: {:.2}x", stats.load_imbalance());
```

### Changed

- **Test Coverage** - Increased from 900+ to 950+ tests
  - 12 PTX cache tests
  - 15 GPU memory pool tests
  - 18 stream manager tests
  - 28 benchmark framework tests
  - 27 hybrid dispatcher tests
  - 23 resource guard tests
  - 12 kernel mode selection tests
  - 7 partitioned queue tests

- **Dependencies** - Added `sha2 = "0.10"` for PTX cache hashing

### Fixed

- Fixed `source_id` → `source_kernel` field name in queue tests
- Fixed floating point precision in `max_safe_elements` test
- Fixed `RingKernelError::InvalidState` struct variant usage in memory pool
- Removed unused `GpuBuffer` import in memory pool

## [0.3.2] - 2026-01-20

### Added

#### GPU Profiling Infrastructure

- **CUDA Profiling Module** (`ringkernel-cuda/src/profiling/`) - **NEW MODULE**
  - Feature-gated via `profiling` feature flag
  - Comprehensive GPU profiling capabilities for performance analysis

- **CUDA Event Wrappers** (`profiling/events.rs`)
  - `CudaEvent` - RAII wrapper for CUDA events with timing support
  - `CudaEventFlags` - Event configuration (blocking sync, disable timing, interprocess)
  - `GpuTimer` - Start/stop timer using CUDA events with microsecond precision
  - `GpuTimerPool` - Pool of reusable timers with interior mutability for concurrent access

- **NVTX Integration** (`profiling/nvtx.rs`)
  - `CudaNvtxProfiler` - Real NVTX profiler using cudarc's nvtx module
  - Timeline visualization in Nsight Systems and Nsight Compute
  - `NvtxCategory` - Predefined categories (Kernel, Transfer, Memory, Sync, Queue, User)
  - `NvtxRange` - RAII wrapper for automatic range end on drop
  - `NvtxPayload` - Typed payloads for markers (I32, I64, U32, U64, F32, F64)
  - Implements `GpuProfiler` trait for integration with ringkernel-core

- **Kernel Metrics** (`profiling/metrics.rs`)
  - `KernelMetrics` - Execution metadata (grid/block dims, GPU time, occupancy, registers)
  - `TransferMetrics` - Memory transfer stats with bandwidth calculation
  - `TransferDirection` - HostToDevice, DeviceToHost, DeviceToDevice
  - `ProfilingSession` - Collects kernel and transfer events with timestamps
  - `KernelAttributes` - Query kernel attributes via cuFuncGetAttribute

- **Memory Tracking** (`profiling/memory_tracker.rs`)
  - `CudaMemoryTracker` - Track GPU memory allocations with timing
  - `TrackedAllocation` - Allocation metadata (ptr, size, kind, label, timestamp)
  - `CudaMemoryKind` - Device, Pinned, Mapped, Managed memory types
  - Peak usage tracking and allocation statistics
  - Integration with `GpuMemoryDashboard` from ringkernel-core

- **Chrome Trace Export** (`profiling/chrome_trace.rs`)
  - `GpuTraceEvent` - Chrome trace format event structure
  - `GpuEventArgs` - Rich event metadata (grid/block dims, occupancy, bandwidth)
  - `GpuChromeTraceBuilder` - Build Chrome trace JSON from profiling sessions
  - Support for kernel events, transfer events, NVTX ranges, memory allocations
  - Process/thread naming for multi-GPU and multi-stream visualization
  - Compatible with chrome://tracing, Perfetto UI, and Nsight Systems

### Changed

- **Dependencies** - Added `nvtx` feature to cudarc dependency
- **ringkernel-cuda/Cargo.toml** - Added optional `serde` and `serde_json` for Chrome trace export

### Fixed

- Added `ProfilerRange::stub()` public constructor in ringkernel-core for external profiler implementations

## [0.3.1] - 2026-01-19

### Added

#### Enterprise Security Features

- **Real Cryptography** (`ringkernel-core/src/security.rs`)
  - AES-256-GCM and ChaCha20-Poly1305 encryption algorithms
  - Proper nonce generation with `rand::thread_rng()`
  - Key derivation using Argon2id and HKDF-SHA256
  - Secure memory wiping with `zeroize` crate
  - Feature-gated via `crypto` feature flag

- **Secrets Management** (`ringkernel-core/src/secrets.rs`) - **NEW FILE**
  - `SecretStore` trait for pluggable secret backends
  - `InMemorySecretStore` for development/testing
  - `EnvVarSecretStore` for environment variable secrets
  - `CachedSecretStore` with TTL-based caching
  - `ChainedSecretStore` for fallback chains
  - `KeyRotationManager` for automatic key rotation
  - `SecretKey` and `SecretValue` types with secure memory handling

- **Authentication Framework** (`ringkernel-core/src/auth.rs`) - **NEW FILE**
  - `AuthProvider` trait for pluggable authentication
  - `ApiKeyAuth` for simple API key validation
  - `JwtAuth` for JWT token validation (RS256/HS256) - requires `auth` feature
  - `ChainedAuthProvider` for fallback authentication chains
  - `AuthContext` with identity and credential management
  - `Credentials` enum: ApiKey, Bearer, Basic, Certificate

- **Role-Based Access Control** (`ringkernel-core/src/rbac.rs`) - **NEW FILE**
  - `Role` enum: Admin, Operator, Developer, Viewer, Custom
  - `Permission` enum: Read, Write, Execute, Admin, Custom
  - `RbacPolicy` with subject-role-permission bindings
  - `PolicyEvaluator` with deny-by-default evaluation
  - `ResourceRule` for fine-grained resource access control

- **Multi-Tenancy Support** (`ringkernel-core/src/tenancy.rs`) - **NEW FILE**
  - `TenantContext` for request scoping with tenant ID
  - `TenantRegistry` for managing tenant configurations
  - `ResourceQuota` with limits for memory, kernels, message rate
  - `ResourceUsage` tracking with quota enforcement
  - `QuotaUtilization` for monitoring tenant resource usage

#### Enterprise Observability

- **OpenTelemetry OTLP Export** (`ringkernel-core/src/observability.rs`)
  - `OtlpExporter` for sending spans to OTLP endpoints
  - `OtlpConfig` with endpoint, headers, and transport configuration
  - Batch export with configurable interval and queue size
  - HTTP and gRPC transport options via `OtlpTransport` enum
  - Automatic retry with exponential backoff
  - `OtlpExporterStats` for monitoring export success/failure

- **Structured Logging** (`ringkernel-core/src/logging.rs`) - **NEW FILE**
  - `StructuredLogger` with multi-sink support
  - `LogLevel`: Trace, Debug, Info, Warn, Error, Fatal
  - `LogOutput`: Text, Json, Compact, Pretty
  - `TraceContext` for automatic trace_id/span_id injection
  - `LogConfig` with builder pattern and presets (development, production)
  - Built-in sinks: `ConsoleSink`, `MemoryLogSink`, `FileLogSink`
  - JSON structured output for log aggregation
  - Global logger functions: `init()`, `info()`, `error()`, etc.

- **Alert Routing System** (`ringkernel-core/src/alerting.rs`) - **NEW FILE**
  - `AlertSink` trait for pluggable alert destinations
  - `AlertRouter` for routing alerts based on severity
  - `WebhookSink` for Slack, Teams, PagerDuty (requires `alerting` feature)
  - `LogSink` and `InMemorySink` for testing/debugging
  - `DeduplicationConfig` for alert deduplication with time windows
  - `AlertSeverity`: Info, Warning, Error, Critical
  - `AlertRouterStats` for monitoring alert delivery

- **Remote Audit Sinks** (`ringkernel-core/src/audit.rs`)
  - `SyslogSink` for RFC 5424 syslog with configurable facility/severity
  - `CloudWatchSink` for AWS CloudWatch Logs integration
  - `ElasticsearchSink` for direct Elasticsearch indexing (requires `alerting` feature)
  - Async batch sending with configurable flush intervals

#### Enterprise Rate Limiting

- **Rate Limiting** (`ringkernel-core/src/rate_limiting.rs`) - **NEW FILE**
  - `RateLimiter` with pluggable algorithms
  - `RateLimitAlgorithm`: TokenBucket, SlidingWindow, LeakyBucket
  - `RateLimitConfig` with burst, window size, and rate configuration
  - `RateLimiterBuilder` with fluent configuration API
  - `RateLimitGuard` RAII wrapper for rate-limited operations
  - `SharedRateLimiter` for distributed rate limiting
  - `RateLimiterExt` trait for easy integration
  - `RateLimiterStatsSnapshot` for monitoring
  - Feature-gated via `rate-limiting` feature flag

#### Network Security

- **TLS Support** (`ringkernel-core/src/tls.rs`) - **NEW FILE**
  - `TlsConfig` with builder pattern for server/client configuration
  - `TlsAcceptor` for server-side TLS with rustls
  - `TlsConnector` for client-side TLS connections
  - `CertificateStore` with automatic rotation and hot reload
  - `SniResolver` for multi-domain certificate selection
  - mTLS (mutual TLS) with client certificate validation
  - `TlsVersion` enum: Tls12, Tls13
  - `TlsSessionInfo` for connection metadata
  - Feature-gated via `tls` feature flag

- **K2K Message Encryption** (`ringkernel-core/src/k2k.rs`)
  - `K2KEncryptor` for kernel-to-kernel message encryption
  - `K2KEncryptionConfig` with algorithm and key configuration
  - `K2KEncryptionAlgorithm`: Aes256Gcm, ChaCha20Poly1305
  - `EncryptedK2KMessage` with nonce and authentication tag
  - `EncryptedK2KEndpoint` wrapper for transparent encryption
  - `EncryptedK2KBuilder` for fluent endpoint creation
  - `K2KKeyMaterial` with secure key handling
  - Forward secrecy support with ephemeral keys
  - Feature-gated via `crypto` feature flag

#### Operational Excellence

- **Operation Timeouts** (`ringkernel-core/src/timeout.rs`) - **NEW FILE**
  - `Timeout` wrapper for async operations with deadlines
  - `Deadline` for absolute timeout tracking
  - `CancellationToken` for cooperative cancellation
  - `OperationContext` with deadline propagation
  - `timeout()` and `timeout_named()` helper functions
  - `with_timeout()` and `with_timeout_named()` for futures
  - `TimeoutStats` and `TimeoutStatsSnapshot` for monitoring

- **Automatic Recovery** (`ringkernel-core/src/health.rs`)
  - `RecoveryPolicy` enum: Restart, Migrate, Checkpoint, Notify, Escalate, Circuit
  - `FailureType` enum: Timeout, Crash, DeviceError, ResourceExhausted, QueueOverflow, StateCorruption
  - `RecoveryConfig` with builder pattern and per-failure-type policies
  - `RecoveryManager` for coordinating recovery actions
  - `RecoveryAction` with retry tracking and timestamps
  - `RecoveryResult` with success/failure details
  - `RecoveryStatsSnapshot` for monitoring recovery attempts
  - Automatic escalation after max retries exceeded
  - Configurable cooldown periods between recovery attempts

### Changed

- **Feature Flags** - New enterprise feature flags in `ringkernel-core/Cargo.toml`:
  - `crypto` - Real cryptography (AES-GCM, ChaCha20, Argon2)
  - `auth` - JWT authentication support
  - `rate-limiting` - Governor-based rate limiting
  - `alerting` - Webhook alerts via reqwest
  - `tls` - TLS support via rustls
  - `enterprise` - Combined feature enabling all enterprise features

- **Test Coverage** - Increased from 825+ to 900+ tests
  - 14 crypto tests for K2K encryption
  - 14 logging tests for structured logging
  - 15 recovery tests for automatic recovery
  - 13 TLS tests for certificate management
  - Plus tests for secrets, auth, RBAC, tenancy, rate limiting, alerting

### Fixed

- Fixed SpanStatus pattern matching for OTLP export
- Fixed AttributeValue JSON serialization in observability
- Fixed TraceId/SpanId Display formatting with hex output
- Fixed reqwest blocking feature for webhook alerts

## [0.3.0] - 2026-01-17

### Added

#### Multi-Kernel Dispatch and Persistent Message Routing

- **`#[derive(PersistentMessage)]` macro** (`ringkernel-derive`)
  - Automatic `handler_id` generation for GPU kernel dispatch
  - Inline payload serialization with response tracking
  - Compile-time handler registration

- **`KernelDispatcher`** (`ringkernel-core/src/dispatcher.rs`) - **NEW FILE**
  - Type-based message routing via K2K broker
  - `DispatcherBuilder` with fluent configuration API
  - `DispatcherConfig` for routing behavior customization
  - `DispatcherMetrics` for observability (messages dispatched, errors, latency)

- **CUDA Handler Dispatch Code Generator** (`ringkernel-cuda-codegen/src/ring_kernel.rs`)
  - `CudaDispatchTable` for handler registration
  - Switch-based dispatch code generation
  - `ExtendedH2KMessage` struct generation for typed payloads

- **Queue Tiering System** (`ringkernel-core/src/queue.rs`)
  - `QueueTier` enum: Small (256), Medium (1024), Large (4096), ExtraLarge (16384)
  - `QueueFactory` for creating appropriately-sized message queues
  - `QueueMonitor` for queue health checking with configurable thresholds
  - `QueueMetrics` for observability (enqueue/dequeue counts, peak depth)
  - `for_throughput()` method for automatic tier selection based on message rate

- **Persistent Message Infrastructure** (`ringkernel-core/src/persistent_message.rs`) - **NEW FILE**
  - `PersistentMessage` trait for GPU-dispatchable messages
  - `DispatchTable` for runtime handler registration
  - `HandlerId` type for type-safe handler identification

#### CUDA NVRTC Compilation

- **`compile_ptx()` function** (`ringkernel-cuda/src/lib.rs`)
  - Wraps `cudarc::nvrtc::compile_ptx` for downstream crates
  - Compile CUDA source to PTX without direct cudarc dependency
  - Returns PTX string or compilation error

#### Memory Pool Management

- **Size-Stratified Memory Pool** (`ringkernel-core/src/memory.rs`)
  - `SizeBucket` enum: Tiny (256B), Small (1KB), Medium (4KB), Large (16KB), Huge (64KB)
  - `StratifiedMemoryPool` - Multi-bucket pool with automatic size selection
  - `StratifiedBuffer` - RAII wrapper that returns buffers to correct bucket on drop
  - `StratifiedPoolStats` - Per-bucket allocation statistics with hit rate tracking
  - `create_stratified_pool()` and `create_stratified_pool_with_capacity()` helpers

- **WebGPU Staging Buffer Pool** (`ringkernel-wgpu/src/memory.rs`)
  - `StagingBufferPool` - Reusable staging buffer cache for GPU-to-host transfers
  - `StagingBufferGuard` - RAII wrapper for automatic buffer return
  - `StagingPoolStats` - Cache hit/miss tracking for staging buffers
  - `WgpuBuffer` extended with optional staging pool integration

- **CUDA Reduction Buffer Cache** (`ringkernel-cuda/src/reduction.rs`)
  - `ReductionBufferCache` - Cache keyed by (num_slots, ReductionOp) for buffer reuse
  - `CachedReductionBuffer<T>` - RAII wrapper with `Deref`/`DerefMut` for transparent access
  - `CacheStats` - Hit/miss counters with hit rate calculation
  - `CacheKey` - Hashable key type for cache lookup

- **Analytics Context Manager** (`ringkernel-core/src/analytics_context.rs`) - **NEW FILE**
  - `AnalyticsContext` - Grouped buffer lifecycle for analytics operations (DFG, BFS, pattern detection)
  - `AllocationHandle` - Type-safe opaque handle to allocations
  - `ContextStats` - Peak/current bytes, allocation counts, typed allocation tracking
  - `AnalyticsContextBuilder` - Fluent builder with preallocation support
  - `allocate_typed<T>()` for type-safe buffer allocation with automatic sizing

- **Memory Pressure Reactions** (`ringkernel-core/src/memory.rs`)
  - `PressureReaction` enum: None, Shrink (with target utilization), Callback
  - `PressureHandler` - Monitors pressure levels and triggers configured reactions
  - `PressureAwarePool` trait - Extension for pressure-aware memory pools
  - Severity-based shrink calculation (Normal → Elevated → Warning → Critical → OutOfMemory)

#### Global Reduction Primitives

- **`ringkernel-core/src/reduction.rs`** - Core reduction traits
  - `ReductionOp` enum: Sum, Min, Max, And, Or, Xor, Product
  - `ReductionScalar` trait for type-safe reduction with identity values
  - `ReductionConfig` for configuring reduction behavior
  - `ReductionHandle` trait for streaming operations
  - `GlobalReduction` trait for backend-agnostic reduction interface

- **`ringkernel-cuda/src/reduction.rs`** - CUDA reduction implementation
  - `ReductionBuffer<T>` using mapped memory (CPU+GPU visible)
  - Zero-copy host read of reduction results
  - Multi-slot support for reduced contention
  - Block-then-atomic pattern for efficient grid reductions
  - Helper code generation: `generate_block_reduce_code()`, `generate_grid_reduce_code()`, `generate_reduce_and_broadcast_code()`

- **`ringkernel-cuda/src/phases.rs`** - Multi-phase kernel execution
  - `SyncMode` enum: Cooperative, SoftwareBarrier, MultiLaunch
  - `KernelPhase` struct for phase metadata
  - `InterPhaseReduction<T>` for reduction between phases
  - `MultiPhaseConfig` for phase sequencing
  - `MultiPhaseExecutor` for orchestrating phase execution
  - `PhaseExecutionStats` for performance tracking

- **`ringkernel-cuda-codegen/src/reduction_intrinsics.rs`** - Codegen for reductions
  - `generate_reduction_helpers()` for cooperative groups support
  - `generate_inline_reduce_and_broadcast()` for inline reduction code
  - `ReductionCodegenConfig` for configuring code generation

- **New codegen intrinsics** in `GpuIntrinsic` enum:
  - Block-level: `BlockReduceSum`, `BlockReduceMin`, `BlockReduceMax`, `BlockReduceAnd`, `BlockReduceOr`
  - Grid-level: `GridReduceSum`, `GridReduceMin`, `GridReduceMax`
  - Combined: `ReduceAndBroadcast`

- **Ring kernel reduction support** via `KernelReductionConfig`:
  - `with_reduction()` builder method on `RingKernelConfig`
  - `with_sum_reduction()` convenience method
  - Automatic reduction boilerplate generation

- **`pagerank_reduction` example** demonstrating PageRank with dangling node handling
  - Triangle graph (no dangling), star graph (75% dangling), chain with sink examples
  - Generated CUDA kernel code visualization

#### CudaDevice Enhancements

- `alloc_mapped<T>()` method for mapped memory allocation
- `supports_cooperative_groups()` method for capability detection

#### Metal Backend Scaffold
- **`ringkernel-metal`** - Apple Metal backend implementation (scaffold)
  - `MetalRuntime` with compute command queue management
  - `MetalBuffer` for GPU buffer allocation and mapping
  - `MetalPipeline` for compute pipeline state
  - Fence-based synchronization (Metal lacks cooperative groups)
  - MSL kernel compilation via metal-rs 0.31
  - Note: True persistent kernels not yet implemented (requires host-driven dispatch)

#### CUDA Enhancements
- **Correlation Tracking** - Request/response message matching via `CorrelationId`
  - `receive_with_correlation()` with timeout support
  - `HashMap<CorrelationId, oneshot::Sender>` for pending correlations
- **Kernel Slot Management** - `SlotAllocator` for K2K route management
  - BitSet-based slot allocation with `allocate()`/`release()`
  - Prevents slot collisions in multi-kernel topologies
- **Cooperative Kernel Fallback** - Software synchronization when grid exceeds limits
  - Automatic fallback to barrier-based sync using atomics
  - `cuLaunchCooperativeKernel` integration via cudarc 0.18.2

#### IR Node Lowering
- **CUDA Backend** - Full messaging and HLC node implementation
  - `K2HEnqueue`, `H2KDequeue`, `H2KIsEmpty` - Host↔Kernel queues
  - `K2KSend`, `K2KRecv`, `K2KTryRecv` - Kernel-to-kernel messaging
  - `HlcNow`, `HlcTick`, `HlcUpdate` - Hybrid logical clock operations
- **MSL Backend** - Metal shading language equivalents
  - Same 9 node types with Metal-specific implementations

#### Persistent FDTD Enhancements
- **Energy Calculation** - Parallel reduction for total field energy
  - `block_reduce_energy()` device function with shared memory
  - E = Σ(p²) computed at progress intervals
  - `atomicAdd` for cross-block accumulation
- **Message Checksum** - CRC32 integrity verification
  - Checksum computation in ring kernel response messages
  - Optional bypass for performance-critical paths

#### WGSL Code Generation
- **Higher-Dimensional Shared Memory** - 2D, 3D, and 4D+ support
  - `SharedTile::new_3d()` for 3D nested arrays
  - 3D generates: `array<array<array<T, X>, Y>, Z>`
  - 4D+ uses linearized indexing with formula generation
  - `SharedVolume<T, X, Y, Z>` marker type for type safety

#### Graph Algorithms
- **Parallel Union-Find** - Shiloach-Vishkin algorithm implementation
  - GPU-accelerated connected components
  - Parallel pointer jumping for path compression

#### Audio Processing
- **Proper Resampling** - Linear interpolation + windowed sinc
  - `LinearResampler` for low-overhead conversion
  - `SincResampler` for high-quality audio
  - Sample rate conversion 44.1kHz ↔ 48kHz

#### Simulation Backends
- **GPU Boundary Reflection** - CUDA kernel for boundary conditions
  - Support for absorbing, reflecting, and periodic boundaries
  - Integrated with tile-based GPU actor system
- **True Cooperative Launch** - `step_cooperative()` with `grid.sync()`
  - Uses `CooperativeLaunchConfig` and `PersistentParams`
  - Grid-wide synchronization without fallback

#### Accounting
- **Industry Chart of Accounts Templates** - Realistic account structures
  - `manufacturing_standard()` - Raw Materials, WIP, Finished Goods, Direct Labor/Materials/Overhead
  - `professional_services_standard()` - Unbilled Receivables, WIP-Billable, Client Retainers
  - `financial_services_standard()` - Trading Securities, Loans Receivable, Customer Deposits, Custody Assets

#### New Crates

- **`ringkernel-montecarlo`** - GPU-accelerated Monte Carlo primitives for variance reduction
  - **Philox RNG** - Counter-based PRNG with `GpuRng` trait (stateless, GPU-friendly)
  - **Antithetic Variates** - Variance reduction using negatively correlated samples
  - **Control Variates** - Variance reduction using correlated variables with known expectations
  - **Importance Sampling** - Self-normalized estimator with exponential tilting for rare events
  - 16 tests covering all algorithms

- **`ringkernel-graph`** - GPU-accelerated graph algorithm primitives
  - **CSR Matrix** - Compressed Sparse Row format with builder pattern
  - **BFS** - Sequential and parallel breadth-first search with multi-source support
  - **SCC** - Strongly connected components via Tarjan and Kosaraju algorithms
  - **Union-Find** - Parallel disjoint set with path compression and union by rank
  - **SpMV** - Sparse matrix-vector multiplication with power iteration
  - Node types: `NodeId`, `Distance`, `ComponentId` with Pod traits
  - 51 tests covering all algorithms

#### Domain System (FR-1)

- **`Domain` enum** - 20 business domain classifications with type ID ranges
  - GraphAnalytics (100-199), StatisticalML (200-299), Compliance (300-399)
  - RiskManagement (400-499), OrderMatching (500-599), MarketData (600-699)
  - Settlement (700-799), Accounting (800-899), NetworkAnalysis (900-999)
  - FraudDetection (1000-1099), TimeSeries (1100-1199), Simulation (1200-1299)
  - Banking (1300-1399), BehavioralAnalytics (1400-1499), ProcessIntelligence (1500-1599)
  - Clearing (1600-1699), TreasuryManagement (1700-1799), PaymentProcessing (1800-1899)
  - FinancialAudit (1900-1999), Custom (10000+)
- **`DomainMessage` trait** - Domain-aware messages with automatic type ID calculation
- **`#[derive(RingMessage)]`** extended with `domain` attribute

#### RingContext Extensions (FR-2)

- **Metrics Types** - `MetricType`, `MetricsEntry`, `ContextMetricsBuffer`
- **Alert Types** - `AlertSeverity`, `KernelAlertType`, `AlertRouting`, `KernelAlert`
- **RingContext methods**:
  - `domain()`, `set_domain()` - Domain association
  - `record_latency()`, `record_throughput()`, `record_counter()`, `record_gauge()` - Metrics collection
  - `flush_metrics()` - Retrieve and clear metrics buffer
  - `emit_alert()`, `alert_if_slow()` - Alert emission

#### K2K Message Registry (FR-3)

- **`K2KMessageRegistration`** - Compile-time message type registration
- **`K2KTypeRegistry`** - Runtime registry with `discover()`, `is_routable()`, `get_category()`
- **`#[derive(RingMessage)]`** extended with `k2k_routable` and `category` attributes
- Integration with `inventory` crate for automatic registration

#### ControlBlock State Helpers (FR-4)

- **`EmbeddedState` trait** - For 24-byte states that fit in ControlBlock._reserved
- **`StateDescriptor`** - 24-byte header for external state references
- **`ControlBlockStateHelper`** - Read/write embedded state from ControlBlock
- **`GpuState` trait** - For larger states with serialization support
- **`#[derive(ControlBlockState)]`** - Derive macro for embedded state types

## [0.2.0] - 2025-01-08

### Added

#### New Crates

- **`ringkernel-ir`** - Unified Intermediate Representation for multi-backend code generation
  - SSA-based IR capturing GPU-specific operations
  - Architecture: Rust DSL → IR → CUDA/WGSL/MSL backends
  - `IrBuilder` fluent API for constructing kernel IR
  - Optimization passes: constant folding, dead code elimination, algebraic simplification
  - `BackendCapabilities` trait for querying backend support
  - `Validator` with configurable validation levels
  - Pretty-printing and IR visualization

- **`ringkernel-cli`** - Command-line tool for project scaffolding and kernel code generation
  - `ringkernel new <name>` - Create new projects with templates (basic, persistent-actor, wavesim, enterprise)
  - `ringkernel init` - Initialize RingKernel in existing projects
  - `ringkernel codegen <file>` - Generate CUDA/WGSL/MSL from Rust DSL
  - `ringkernel check` - Validate kernel compatibility across backends
  - `ringkernel completions` - Generate shell completions (bash, zsh, fish, PowerShell)
  - Colored terminal output with progress indicators

#### Enterprise Runtime Features

- **`RuntimeBuilder`** - Fluent builder for enterprise runtime configuration
  - Presets: `development()`, `production()`, `high_performance()`
  - Automatic component initialization based on configuration

- **`RingKernelContext`** - Unified runtime managing all enterprise features
  - Centralized access to health, metrics, multi-GPU, and migration components
  - Lifecycle management with state machine

- **`ConfigBuilder`** - Nested configuration system with builder pattern
  - Environment variable overrides
  - TOML/YAML configuration file support

- **`LifecycleState`** - Runtime state machine
  - States: `Initializing` → `Running` → `Draining` → `ShuttingDown` → `Stopped`
  - Graceful shutdown with drain timeout

- **Health & Resilience**
  - `HealthChecker` - Liveness/readiness probes with async health checks
  - `CircuitBreaker` - Fault tolerance with automatic recovery (Closed/Open/HalfOpen states)
  - `DegradationManager` - Graceful degradation with 5 levels (Normal → Critical)
  - `KernelWatchdog` - Stale kernel detection with configurable heartbeat monitoring

- **Observability**
  - `PrometheusExporter` - Export metrics in Prometheus format
  - `ObservabilityContext` - Distributed tracing with span management
  - GPU memory dashboard with pressure alerts

- **Multi-GPU**
  - `MultiGpuCoordinator` - Device selection with load balancing strategies (RoundRobin, LeastLoaded, Random)
  - `KernelMigrator` - Live kernel migration between GPUs using checkpoints
  - `GpuTopology` - NVLink/PCIe topology discovery

- **`ShutdownReport`** - Final statistics on graceful shutdown

#### Security Module

- **`MemoryEncryption`** - GPU memory encryption
  - Algorithms: AES-256-GCM, AES-128-GCM, ChaCha20-Poly1305, XChaCha20-Poly1305
  - Key derivation: HKDF-SHA256, HKDF-SHA384, Argon2id, PBKDF2-SHA256
  - Automatic key rotation with configurable interval
  - Encrypt control blocks, message queues, and kernel state

- **`KernelSandbox`** - Kernel isolation and resource control
  - `ResourceLimits` - Memory, execution time, message rate, K2K connections
  - `SandboxPolicy` - K2K ACLs (allow/deny lists), memory access levels
  - Presets: `restrictive()` for untrusted kernels, `permissive()` for trusted
  - Violation detection and recording

- **`ComplianceReporter`** - Audit-ready compliance documentation
  - Standards: SOC2, GDPR, HIPAA, PCI-DSS, ISO 27001, FedRAMP, NIST CSF
  - Export formats: JSON, HTML, Markdown, PDF, CSV
  - Automatic compliance check generation with evidence and recommendations

#### ML Framework Bridges

- **`PyTorchBridge`** - Bidirectional tensor interop with PyTorch
  - Data types: Float16/32/64, BFloat16, Int8/32/64, UInt8, Bool
  - Device management (CPU, CUDA)
  - Pinned memory support

- **`OnnxExecutor`** - Load and execute ONNX models on GPU ring kernels
  - Model loading from file or memory
  - Input/output tensor management
  - Execution providers configuration

- **`HuggingFacePipeline`** - Integration with Hugging Face Transformers
  - Text classification, generation, and embedding pipelines
  - Model caching and configuration

#### Developer Experience

- **Hot Reload** - Kernel hot reload with state preservation
  - File system watcher for kernel source changes
  - State checkpointing during reload

- **GPU Memory Dashboard** - Real-time memory monitoring
  - Pressure alerts with configurable thresholds
  - Per-kernel memory breakdown

- **Mock GPU Testing** (`ringkernel-cpu/src/mock.rs`)
  - `MockGpuDevice` for testing GPU code without hardware
  - Deterministic execution for reproducible tests
  - Memory allocation tracking

- **Fuzzing Infrastructure** (5 fuzz targets)
  - Message serialization fuzzing
  - Queue operations fuzzing
  - HLC timestamp fuzzing
  - IR validation fuzzing
  - Codegen fuzzing

- **CI GPU Testing Workflow**
  - GitHub Actions with GPU runner support
  - Automated CUDA and WebGPU test execution

- **Interactive Tutorials** (4 tutorials)
  - `01-hello-kernel` - Basic kernel lifecycle
  - `02-message-passing` - Request/response patterns
  - `03-k2k-messaging` - Kernel-to-kernel communication
  - `04-persistent-actors` - Persistent GPU actors

- **VSCode Extension Scaffolding**
  - Syntax highlighting for RingKernel DSL
  - Code completion support

#### Additional Features

- **SIMD Optimizations** (`ringkernel-cpu/src/simd.rs`)
  - Vectorized stencil operations
  - SIMD-accelerated reductions

- **Subgroup Operations** (WGSL backend)
  - `subgroupAdd`, `subgroupMul`, `subgroupMin`, `subgroupMax`
  - Broadcast and shuffle operations

- **Metal K2K Halo Exchange** - Kernel-to-kernel communication on Metal backend

- **Optimization Passes** (ringkernel-ir)
  - `ConstantFolding` - Compile-time constant evaluation
  - `DeadCodeElimination` - Remove unused values
  - `DeadBlockElimination` - Remove unreachable blocks
  - `AlgebraicSimplification` - Simplify arithmetic expressions

### Changed

- **API Changes**
  - Renamed `RuntimeMetrics` → `ContextMetrics`

- **Test Coverage**
  - Increased from 580+ to 700+ tests across workspace

### Fixed

- Various clippy warnings across all crates
- HLC test using `tick()` instead of read-only `now()`
- Tutorial code formatting for educational clarity

## [0.1.3] - 2025-12-14

### Added

#### Cooperative Groups Support
- **Grid-wide GPU synchronization** via CUDA cooperative groups (`grid.sync()`)
- **`cuLaunchCooperativeKernel` driver API interop** - Direct FFI calls to CUDA driver for true cooperative launch
- **Build-time PTX compilation** - `build.rs` with nvcc detection and automatic kernel compilation
- **`cooperative` feature flag** for `ringkernel-cuda` and `ringkernel-wavesim3d`
- **`cooperative` field in `LaunchOptions`** for cooperative launch mode

#### Block Actor Backend (WaveSim3D)
- **8×8×8 block-based actor model** - Hybrid approach combining stencil and actor patterns
  - Intra-block: Fast stencil computation with shared memory
  - Inter-block: Double-buffered message passing (no atomics)
- **`BlockActorGpuBackend`** with `step_fused()` for single-kernel-launch execution
- **Performance**: 8,165 Mcells/s (59.6× faster than per-cell actors)
- **Grid size validation** with `max_cooperative_blocks` (144 on RTX 4090)

#### New Computation Method
- **`ComputationMethod::BlockActor`** - Third GPU computation method for wavesim3d
  - Combines actor model benefits with stencil performance
  - 10-50× faster than per-cell Actor method

### Changed
- Added `CooperativeKernel` wrapper in `ringkernel-cuda::cooperative` module
- Added cooperative kernel infrastructure to wavesim3d benchmark

#### Dependency Updates
- **tokio**: 1.35 → 1.48 (improved task scheduling, better cancellation handling)
- **thiserror**: 1.0 → 2.0 (updated derive macros)
- **wgpu**: 0.19 → 27.0 (Arc-based resource tracking, 40%+ performance improvement)
  - Migrated to new Instance/Adapter/Device creation API
  - Updated pipeline descriptors with `entry_point: Option<&str>`, `compilation_options`, `cache`
  - Renamed `ImageCopyTexture` → `TexelCopyTextureInfo`, `ImageDataLayout` → `TexelCopyBufferLayout`
  - Updated `device.poll()` to use `PollType::wait_indefinitely()`
- **winit**: 0.29 → 0.30 (new window creation API)
- **egui/egui-wgpu/egui-winit**: 0.27 → 0.31 (updated for wgpu 27 compatibility)
- **glam**: 0.27 → 0.29 (linear algebra updates)
- **metal**: 0.27 → 0.31 (Apple GPU backend updates)
- **axum**: 0.7 → 0.8 (improved routing, better error handling)
- **tower**: 0.4 → 0.5 (service abstraction updates)
- **tonic**: 0.11 → 0.14 (better gRPC streaming, improved health checking)
- **prost**: 0.12 → 0.14 (protobuf updates to match tonic)
- **actix-rt**: 2.9 → 2.10
- **rayon**: 1.10 → 1.11 (requires MSRV 1.80)
- **arrow**: 52 → 54 (columnar data updates)
- **polars**: 0.39 → 0.46 (DataFrame updates)

#### Deferred Updates
- **iced**: Kept at 0.13 (0.14 requires major application API rewrite)
- **rkyv**: Kept at 0.7 (0.8 has incompatible data format, requires significant migration)

## [0.1.2] - 2025-12-11

### Added

#### New Crate
- **WaveSim3D** (`ringkernel-wavesim3d`) - 3D acoustic wave simulation with realistic physics
  - Full 3D FDTD (Finite-Difference Time-Domain) wave propagation solver
  - Binaural audio rendering with HRTF (Head-Related Transfer Function) support
  - Volumetric ray marching visualization for real-time 3D pressure field rendering
  - GPU-native actor system for distributed 3D wave simulation
  - Support for multiple sound sources with frequency-dependent propagation
  - Material absorption modeling with frequency-dependent coefficients
  - Interactive 3D camera controls and visualization modes

#### CUDA Codegen Intrinsics Expansion
- Expanded GPU intrinsics from ~45 to **120+ operations** across 13 categories
- **Atomic Operations** (11 ops): `atomic_add`, `atomic_sub`, `atomic_min`, `atomic_max`, `atomic_exchange`, `atomic_cas`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_inc`, `atomic_dec`
- **Synchronization** (7 ops): `sync_threads`, `sync_threads_count`, `sync_threads_and`, `sync_threads_or`, `thread_fence`, `thread_fence_block`, `thread_fence_system`
- **Trigonometric** (11 ops): `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sincos`, `sinpi`, `cospi`
- **Hyperbolic** (6 ops): `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- **Exponential/Logarithmic** (18 ops): `exp`, `exp2`, `exp10`, `expm1`, `log`, `ln`, `log2`, `log10`, `log1p`, `pow`, `ldexp`, `scalbn`, `ilogb`, `erf`, `erfc`, `erfinv`, `erfcinv`, `lgamma`, `tgamma`
- **Classification** (8 ops): `is_nan`, `is_infinite`, `is_finite`, `is_normal`, `signbit`, `nextafter`, `fdim`
- **Warp Operations** (16 ops): `warp_active_mask`, `warp_shfl`, `warp_shfl_up`, `warp_shfl_down`, `warp_shfl_xor`, `warp_ballot`, `warp_all`, `warp_any`, `warp_match_any`, `warp_match_all`, `warp_reduce_add/min/max/and/or/xor`
- **Bit Manipulation** (8 ops): `popc`, `clz`, `ctz`, `ffs`, `brev`, `byte_perm`, `funnel_shift_left`, `funnel_shift_right`
- **Memory Operations** (3 ops): `ldg`, `prefetch_l1`, `prefetch_l2`
- **Special Functions** (13 ops): `rcp`, `fast_div`, `saturate`, `j0`, `j1`, `jn`, `y0`, `y1`, `yn`, `normcdf`, `normcdfinv`, `cyl_bessel_i0`, `cyl_bessel_i1`
- **Timing** (3 ops): `clock`, `clock64`, `nanosleep`
- **3D Stencil Intrinsics**: `pos.up(buf)`, `pos.down(buf)`, `pos.at(buf, dx, dy, dz)` for volumetric kernels

### Changed
- Added `required-features` to CUDA-only wavesim binaries to fix build without CUDA
- Updated GitHub Actions release workflow with proper feature flags and Ubuntu version
- Updated ringkernel-cuda-codegen tests from 143 to 171 tests

### Fixed
- Fixed release workflow feature flags for showcase applications
- Fixed Ubuntu version compatibility in CI/CD pipeline

## [0.1.1] - 2025-12-04

### Added

#### New Showcase Applications
- **AccNet** (`ringkernel-accnet`) - GPU-accelerated accounting network analytics
  - Network visualization with force-directed graph layout
  - Fraud detection: circular flows, threshold clustering, Benford's Law violations
  - GAAP compliance checking for accounting rule violations
  - Temporal analysis for seasonality, trends, and behavioral anomalies
  - GPU kernels: Suspense detection, GAAP violation, Benford analysis, PageRank
- **ProcInt** (`ringkernel-procint`) - GPU-accelerated process intelligence
  - DFG (Directly-Follows Graph) mining from event streams
  - Pattern detection: bottlenecks, loops, rework, long-running activities
  - Conformance checking with fitness and precision metrics
  - Timeline view with partial order traces and concurrent activity visualization
  - Multi-sector templates: Healthcare, Manufacturing, Finance, IT
  - GPU kernels: DFG construction, pattern detection, partial order derivation, conformance checking

### Changed
- Updated showcase documentation with AccNet and ProcInt sections
- Updated CI workflow to exclude CUDA tests on runners without GPU hardware

### Fixed
- Fixed 14 clippy warnings in ringkernel-accnet (needless_range_loop, manual_range_contains, clamp patterns, etc.)
- Fixed benchmark API compatibility in ringkernel-accnet
- Fixed code formatting issues across showcase applications

## [0.1.0] - 2025-12-03

### Added

#### Core Framework
- GPU-native persistent actor model with `RingKernelRuntime` trait
- Lock-free `MessageQueue` (SPSC ring buffer) for host-GPU message passing
- `ControlBlock` - 128-byte GPU-resident structure for kernel lifecycle management
- `RingContext` - GPU intrinsics facade for kernel handlers
- Hybrid Logical Clocks (`HlcTimestamp`, `HlcClock`) for causal ordering across distributed kernels
- `KernelHandle` for managing kernel lifecycle (launch, activate, terminate)

#### Messaging
- `RingMessage` trait with zero-copy serialization via rkyv
- Kernel-to-Kernel (K2K) direct messaging with `K2KBroker` and `K2KEndpoint`
- Topic-based Publish/Subscribe with wildcard support via `PubSubBroker`
- Message correlation tracking and priority support

#### Procedural Macros (`ringkernel-derive`)
- `#[derive(RingMessage)]` - Automatic message serialization with field annotations
- `#[ring_kernel]` - Kernel handler definition with configuration
- `#[derive(GpuType)]` - GPU-compatible type generation

#### Backend Support
- **CPU Backend** (`ringkernel-cpu`) - Always available for testing and fallback
- **CUDA Backend** (`ringkernel-cuda`) - NVIDIA GPU support via cudarc
- **WebGPU Backend** (`ringkernel-wgpu`) - Cross-platform GPU support (Vulkan, Metal, DX12)
- **Metal Backend** (`ringkernel-metal`) - Apple GPU support (scaffolded)
- Auto-detection with `Backend::Auto` (tries CUDA → Metal → WebGPU → CPU)

#### Code Generation
- **CUDA Codegen** (`ringkernel-cuda-codegen`) - Rust DSL to CUDA C transpiler
  - Global kernels with block/grid indices
  - Stencil kernels with `GridPos` abstraction and tiled shared memory
  - Ring kernels for persistent actor model with HLC and K2K support
  - 45+ GPU intrinsics (atomics, warp ops, sync, math)
- **WGSL Codegen** (`ringkernel-wgpu-codegen`) - Rust DSL to WGSL transpiler
  - Full parity with CUDA codegen for portable shaders
  - 64-bit emulation via lo/hi u32 pairs
  - Subgroup operations support

#### Ecosystem Integrations (`ringkernel-ecosystem`)
- Actor framework integrations (Actix, Tower)
- Web framework integrations (Axum)
- Data processing (Arrow, Polars)
- gRPC support (Tonic)
- Machine learning (Candle)
- Configuration management
- Metrics and observability (Prometheus, tracing)

#### Example Applications
- **WaveSim** (`ringkernel-wavesim`) - Interactive 2D acoustic wave simulation
  - FDTD solver with GPU acceleration
  - Educational modes demonstrating parallel computing evolution
  - Multiple backends (CPU, CUDA, WebGPU)
- **TxMon** (`ringkernel-txmon`) - Real-time transaction monitoring
  - GPU-accelerated fraud detection patterns
  - Structuring detection, velocity checks, PEP monitoring
  - Interactive GUI with real-time visualization
- **Audio FFT** (`ringkernel-audio-fft`) - GPU-accelerated audio processing
  - Direct/ambience source separation
  - Real-time FFT processing with actor model

### Performance
- CUDA codegen achieves ~93B elem/sec on RTX Ada (12,378x vs CPU)
- Lock-free message queue with sub-microsecond latency
- Zero-copy serialization for GPU transfer

### Documentation
- Comprehensive README files for all crates
- CLAUDE.md with build commands and architecture overview
- Code examples for all major features

[Unreleased]: https://github.com/mivertowski/RustCompute/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/mivertowski/RustCompute/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/mivertowski/RustCompute/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/mivertowski/RustCompute/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/mivertowski/RustCompute/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mivertowski/RustCompute/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/mivertowski/RustCompute/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/mivertowski/RustCompute/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/mivertowski/RustCompute/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mivertowski/RustCompute/releases/tag/v0.1.0
