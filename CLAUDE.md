# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RingKernel is a GPU-native persistent actor model framework for Rust. It enables GPU-accelerated actor systems with persistent kernels, lock-free message passing, and hybrid logical clocks (HLC) for causal ordering.

## Build Commands

```bash
# Build entire workspace
cargo build --workspace

# Build with specific GPU backend
cargo build --workspace --features cuda
cargo build --workspace --features wgpu

# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p ringkernel-core

# Run a single test
cargo test -p ringkernel-core test_name

# Run CUDA GPU execution tests (requires NVIDIA GPU)
cargo test -p ringkernel-cuda --test gpu_execution_verify

# Run WebGPU tests (requires GPU)
cargo test -p ringkernel-wgpu --features wgpu-tests -- --ignored

# Run benchmarks
cargo bench --package ringkernel

# Run examples
cargo run -p ringkernel --example basic_hello_kernel
cargo run -p ringkernel --example kernel_to_kernel
cargo run -p ringkernel --example wgpu_hello --features wgpu

# Run CUDA codegen examples
cargo run -p ringkernel --example global_kernel
cargo run -p ringkernel --example stencil_kernel
cargo run -p ringkernel --example ring_kernel_codegen

# Run educational modes example
cargo run -p ringkernel --example educational_modes

# Run transaction monitoring GUI
cargo run -p ringkernel-txmon --release --features cuda-codegen

# Run txmon GPU benchmark (requires NVIDIA GPU)
cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen

# Run process intelligence GUI
cargo run -p ringkernel-procint --release

# Run process intelligence benchmark
cargo run -p ringkernel-procint --bin procint-benchmark --release

# Run wavesim3d with cooperative groups (requires nvcc)
cargo run -p ringkernel-wavesim3d --release --features cooperative

# Run wavesim3d throughput benchmark (GPU actor comparison)
cargo run -p ringkernel-wavesim3d --bin wavesim3d-benchmark --release --features cuda-codegen

# Run wavesim3d interactive benchmark (persistent vs traditional latency)
cargo run -p ringkernel-wavesim3d --bin interactive-benchmark --release --features cuda-codegen

# Run wavesim3d persistent actor test
cargo run -p ringkernel-wavesim3d --example test_persistent --release --features cuda-codegen

# Run ecosystem tests
cargo test -p ringkernel-ecosystem --features "persistent,actix,tower,axum,grpc"

# Run ecosystem example (Axum REST API)
cargo run -p ringkernel-ecosystem --example axum_persistent_api --features "axum,persistent"

# RingKernel CLI tool
cargo run -p ringkernel-cli -- new my-app --template persistent-actor
cargo run -p ringkernel-cli -- codegen src/kernels/mod.rs --backend cuda,wgsl
cargo run -p ringkernel-cli -- check --backends all
cargo run -p ringkernel-cli -- init --backends cuda
```

## Architecture

### Workspace Structure

The project is a Cargo workspace with these crates:

- **`ringkernel`** - Main facade crate that re-exports everything; entry point for users
- **`ringkernel-core`** - Core traits and types (RingMessage, MessageQueue, HlcTimestamp, ControlBlock, RingContext, RingKernelRuntime, K2K messaging, PubSub)
- **`ringkernel-derive`** - Proc macros: `#[derive(RingMessage)]`, `#[ring_kernel]`, `#[derive(GpuType)]`
- **`ringkernel-cpu`** - CPU backend implementation (always available, used for testing/fallback)
- **`ringkernel-cuda`** - NVIDIA CUDA backend with cooperative groups and GPU profiling support (feature-gated)
- **`ringkernel-wgpu`** - WebGPU cross-platform backend (feature-gated)
- **`ringkernel-metal`** - Apple Metal backend (feature-gated, macOS only, scaffold implementation with runtime/buffer/pipeline)
- **`ringkernel-codegen`** - GPU kernel code generation
- **`ringkernel-cuda-codegen`** - Rust-to-CUDA transpiler for writing GPU kernels in Rust DSL
- **`ringkernel-wgpu-codegen`** - Rust-to-WGSL transpiler for writing GPU kernels in Rust DSL (WebGPU backend)
- **`ringkernel-ir`** - Unified Intermediate Representation for multi-backend code generation (CUDA/WGSL/MSL)
- **`ringkernel-ecosystem`** - Ecosystem integrations with **persistent GPU actor support** (Actix `GpuPersistentActor`, Axum REST/SSE, Tower `PersistentKernelService`, gRPC streaming, ML bridges)
- **`ringkernel-cli`** - CLI tool for project scaffolding, kernel code generation, and compatibility checking
- **`ringkernel-audio-fft`** - Example application: GPU-accelerated audio FFT processing
- **`ringkernel-wavesim`** - Example application: 2D acoustic wave simulation with GPU-accelerated FDTD, tile-based ring kernel actors, and educational simulation modes
- **`ringkernel-wavesim3d`** - Example application: 3D acoustic wave simulation with binaural audio, **persistent GPU actors** (H2K/K2H messaging, K2K halo exchange, cooperative groups), and volumetric ray marching visualization
- **`ringkernel-txmon`** - Showcase application: GPU-accelerated transaction monitoring with real-time fraud detection GUI
- **`ringkernel-accnet`** - Showcase application: GPU-accelerated accounting network visualization
- **`ringkernel-procint`** - Showcase application: GPU-accelerated process intelligence with DFG mining, pattern detection, and conformance checking
- **`ringkernel-montecarlo`** - Monte Carlo primitives: Philox RNG, antithetic variates, control variates, importance sampling
- **`ringkernel-graph`** - Graph algorithms: CSR matrix, BFS, SCC (Tarjan/Kosaraju), Union-Find, SpMV

### Core Abstractions (in ringkernel-core)

- **`RingMessage`** trait - Messages serializable via rkyv for zero-copy GPU transfer
- **`MessageQueue`** - Lock-free ring buffer for host↔GPU message passing
- **`RingKernelRuntime`** trait - Backend-agnostic runtime interface
- **`KernelHandle`** - Handle to manage kernel lifecycle (launch, activate, terminate)
- **`HlcTimestamp`/`HlcClock`** - Hybrid logical clocks for causal ordering
- **`ControlBlock`** - 128-byte GPU-resident structure managing kernel state
- **`RingContext`** - GPU intrinsics facade passed to kernel handlers
- **`K2KBroker`/`K2KEndpoint`** - Kernel-to-kernel direct messaging
- **`PubSubBroker`** - Topic-based publish/subscribe with wildcards
- **`ReductionOp`** - Reduction operations (Sum, Min, Max, And, Or, Xor, Product)
- **`ReductionScalar`** trait - Type-safe reduction with identity values
- **`GlobalReduction`** trait - Backend-agnostic reduction interface
- **`StratifiedMemoryPool`** - Size-stratified buffer pooling with automatic bucket selection
- **`AnalyticsContext`** - Grouped buffer lifecycle for analytics operations

### Memory Pool Management (in ringkernel-core)

Efficient buffer reuse for analytics workloads:

- **`SizeBucket`** - Size classes: Tiny (256B), Small (1KB), Medium (4KB), Large (16KB), Huge (64KB)
- **`StratifiedMemoryPool`** - Multi-bucket pool with automatic size selection
- **`StratifiedBuffer`** - RAII wrapper that returns to correct bucket on drop
- **`StratifiedPoolStats`** - Per-bucket allocation statistics with hit rate
- **`PressureHandler`** - Memory pressure monitoring with configurable reactions
- **`PressureReaction`** - None, Shrink (with target utilization), or Callback

```rust
use ringkernel_core::memory::{StratifiedMemoryPool, SizeBucket};

let pool = StratifiedMemoryPool::new("analytics");
let buf = pool.allocate(2000);  // Uses Medium bucket (4KB)
// Buffer returned to pool on drop

let stats = pool.stats();
println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);
```

### Analytics Context (in ringkernel-core)

Grouped buffer lifecycle for analytics operations:

- **`AnalyticsContext`** - Groups buffers for a single analytics operation
- **`AllocationHandle`** - Type-safe opaque handle to allocations
- **`ContextStats`** - Peak/current bytes, allocation counts
- **`AnalyticsContextBuilder`** - Fluent builder with preallocation support

```rust
use ringkernel_core::analytics_context::AnalyticsContext;

let mut ctx = AnalyticsContext::new("bfs_traversal");
let frontier = ctx.allocate(1024);
let visited = ctx.allocate_typed::<u32>(256);
// All buffers released when ctx drops
```

### Global Reduction Primitives (in ringkernel-cuda)

GPU-accelerated global reductions for algorithms like PageRank with dangling nodes:

- **`ReductionBuffer<T>`** - Mapped memory buffer for zero-copy reduction results
- **`ReductionBufferBuilder`** - Builder pattern for creating reduction buffers
- **`ReductionBufferCache`** - Cache for buffer reuse keyed by (num_slots, op)
- **`CachedReductionBuffer<T>`** - RAII wrapper that returns buffer to cache on drop
- **Multi-phase kernel execution** via `phases.rs`:
  - **`SyncMode`** - Cooperative (grid.sync()), SoftwareBarrier (atomics), MultiLaunch
  - **`KernelPhase`** - Phase metadata (name, function, dimensions)
  - **`InterPhaseReduction<T>`** - Reduction between kernel phases
  - **`MultiPhaseConfig`** - Phase sequencing configuration
  - **`MultiPhaseExecutor`** - Orchestrates multi-phase execution

```rust
use ringkernel_cuda::reduction::{ReductionBuffer, ReductionBufferBuilder};
use ringkernel_cuda::phases::{MultiPhaseConfig, SyncMode};

// Create reduction buffer for dangling node sum
let buffer = ReductionBufferBuilder::new()
    .with_op(ReductionOp::Sum)
    .with_slots(4)
    .build(&device)?;

// Multi-phase PageRank kernel
let config = MultiPhaseConfig::new()
    .with_sync_mode(SyncMode::Cooperative)  // grid.sync() on CC 6.0+
    .with_phase("accumulate", accumulate_fn)
    .with_phase("apply", apply_fn);
```

Run the PageRank example: `cargo run -p ringkernel --example pagerank_reduction --features cuda`

### PTX Compilation Cache (in ringkernel-cuda)

Disk-based PTX caching for faster kernel loading:

- **`PtxCache`** - SHA-256 content-based caching with compute capability awareness
- **`PtxCacheStats`** - Hit/miss statistics for cache performance monitoring
- **`PtxCacheError`** - Descriptive error types for cache operations
- Default location: `~/.cache/ringkernel/ptx/`
- Environment variable: `RINGKERNEL_PTX_CACHE_DIR`

```rust
use ringkernel_cuda::compile::{PtxCache, PtxCacheStats};

let cache = PtxCache::new()?;
let hash = PtxCache::hash_source(cuda_source);

if let Some(ptx) = cache.get(&hash, "sm_89")? {
    // Use cached PTX
} else {
    let ptx = compile_ptx(cuda_source)?;
    cache.put(&hash, "sm_89", &ptx)?;
}
```

### GPU Stratified Memory Pool (in ringkernel-cuda)

Size-stratified GPU VRAM pooling with O(1) allocation:

- **`GpuStratifiedPool`** - 6 size classes (256B to 256KB) with free lists
- **`GpuPoolConfig`** - Configuration with presets: `for_graph_analytics()`, `for_simulation()`
- **`GpuSizeClass`** - Size class enum for bucket selection
- **`GpuPoolDiagnostics`** - Utilization monitoring and statistics

```rust
use ringkernel_cuda::memory_pool::{GpuStratifiedPool, GpuPoolConfig, GpuSizeClass};

let config = GpuPoolConfig::for_graph_analytics();
let mut pool = GpuStratifiedPool::new(&device, config)?;
pool.warm_bucket(GpuSizeClass::Size1KB, 100)?;  // Pre-allocate

let ptr = pool.allocate(512)?;  // O(1) from free list
pool.deallocate(ptr, 512)?;
```

### Multi-Stream Execution (in ringkernel-cuda)

CUDA stream management for compute/transfer overlap:

- **`StreamManager`** - Multi-stream with compute and transfer streams
- **`StreamConfig`** - Configuration with presets: `minimal()`, `performance()`
- **`StreamId`** - `Compute(usize)`, `Transfer`, `Default`
- **`StreamPool`** - Load-balanced stream assignment
- **`OverlapMetrics`** - Compute/transfer overlap measurement

```rust
use ringkernel_cuda::stream::{StreamManager, StreamConfig, StreamId};

let manager = StreamManager::new(&device, StreamConfig::performance())?;
manager.record_event("kernel_done", StreamId::Compute(0))?;
manager.stream_wait_event(StreamId::Transfer, "kernel_done")?;
```

### Benchmark Framework (in ringkernel-core)

Comprehensive benchmarking with regression detection (feature-gated via `benchmark`):

- **`Benchmarkable` trait** - Generic interface for workloads
- **`BenchmarkSuite`** - Orchestration with multiple report formats
- **`BenchmarkConfig`** - Presets: `quick()`, `comprehensive()`, `ci()`
- **`BenchmarkResult`** - Throughput, timing, custom metrics
- **`RegressionReport`** - Baseline comparison with status tracking
- **`Statistics`** - ConfidenceInterval, DetailedStatistics, ScalingMetrics

```rust
use ringkernel_core::benchmark::{BenchmarkSuite, BenchmarkConfig, Benchmarkable};

let mut suite = BenchmarkSuite::new(BenchmarkConfig::comprehensive());
suite.run_all_sizes(&MyWorkload);

println!("{}", suite.generate_markdown_report());
if let Some(report) = suite.compare_to_baseline() {
    println!("Regressions: {}", report.regression_count);
}
```

### Hybrid CPU-GPU Dispatcher (in ringkernel-core)

Intelligent workload routing with adaptive thresholds:

- **`HybridDispatcher`** - Automatic CPU/GPU routing with learning
- **`HybridWorkload` trait** - `execute_cpu()` / `execute_gpu()` interface
- **`ProcessingMode`** - `GpuOnly`, `CpuOnly`, `Hybrid`, `Adaptive`
- **`HybridConfig`** - Presets: `cpu_only()`, `gpu_only()`, `adaptive()`
- **`HybridStats`** - Execution counts and adaptive threshold history

```rust
use ringkernel_core::hybrid::{HybridDispatcher, HybridConfig};

let dispatcher = HybridDispatcher::new(HybridConfig::adaptive());
let result = dispatcher.execute(&workload);  // Automatic routing
```

### Resource Guard (in ringkernel-core)

Memory limit enforcement with reservations:

- **`ResourceGuard`** - Configurable limits with safety margin
- **`ReservationGuard`** - RAII wrapper for guaranteed allocations
- **`MemoryEstimator` trait** - Workload memory estimation
- **`MemoryEstimate`** - Primary, auxiliary, peak bytes with confidence
- **`LinearEstimator`** - Simple linear estimator
- System utilities: `get_total_memory()`, `get_available_memory()`

```rust
use ringkernel_core::resource::{ResourceGuard, MemoryEstimate};

let guard = ResourceGuard::with_max_memory(4 * 1024 * 1024 * 1024);
let reservation = guard.reserve(512 * 1024 * 1024)?;
// Automatically released on drop
```

### Kernel Mode Selection (in ringkernel-cuda)

Intelligent kernel launch configuration:

- **`KernelMode`** - `ElementCentric`, `SoA`, `Tiled`, `WarpCooperative`, `Auto`
- **`AccessPattern`** - `Coalesced`, `Stencil`, `Irregular`, `Reduction`, `Scatter`, `Gather`
- **`WorkloadProfile`** - Element count, bytes per element, access pattern
- **`GpuArchitecture`** - Presets: `volta()`, `ampere()`, `ada()`, `hopper()`
- **`KernelModeSelector`** - Optimal mode selection and launch config generation
- **`LaunchConfig`** - Complete kernel launch configuration

```rust
use ringkernel_cuda::launch_config::{KernelModeSelector, WorkloadProfile, AccessPattern};

let selector = KernelModeSelector::with_defaults();
let profile = WorkloadProfile::new(1_000_000, 64)
    .with_access_pattern(AccessPattern::Stencil { radius: 1 });
let config = selector.launch_config(selector.select(&profile), 1_000_000);
```

### Partitioned Queues (in ringkernel-core)

Multi-partition message queues for reduced contention:

- **`PartitionedQueue`** - Hash-based routing by source kernel ID
- **`PartitionedQueueStats`** - Per-partition statistics with load imbalance metric

```rust
use ringkernel_core::queue::PartitionedQueue;

let queue = PartitionedQueue::new(4, 1024);  // 4 partitions
queue.try_enqueue(envelope)?;  // Routed by source_kernel
let msg = queue.try_dequeue_any();  // Round-robin dequeue
```

### Enterprise Features (in ringkernel-core)

The following enterprise-grade features provide production-ready infrastructure:

- **`RingKernelContext`** - Unified runtime managing all enterprise features
- **`RuntimeBuilder`** - Fluent builder with `development()`, `production()`, `high_performance()` presets
- **`ConfigBuilder`** - Unified configuration system with nested builders

**Health & Resilience:**
- **`HealthChecker`** - Liveness/readiness probes with async health checks
- **`CircuitBreaker`** - Fault tolerance with automatic recovery
- **`DegradationManager`** - Graceful degradation with 5 levels (Normal → Critical)
- **`KernelWatchdog`** - Stale kernel detection with heartbeat monitoring
- **`RecoveryManager`** - Automatic recovery with configurable policies per failure type

**Observability:**
- **`PrometheusExporter`** - Prometheus metrics export
- **`OtlpExporter`** - OpenTelemetry OTLP export to Jaeger/Honeycomb/Datadog
- **`ObservabilityContext`** - Distributed tracing with spans
- **`StructuredLogger`** - Multi-sink logging with trace correlation (JSON/Text output)
- **`AlertRouter`** - Alert routing with deduplication and severity-based routing

**Security (feature-gated via `crypto`, `auth`, `tls`):**
- **`MemoryEncryption`** - AES-256-GCM and ChaCha20-Poly1305 encryption
- **`K2KEncryptor`** - Kernel-to-kernel message encryption with forward secrecy
- **`SecretStore`** - Pluggable secrets management with key rotation
- **`TlsConfig`/`TlsAcceptor`/`TlsConnector`** - TLS/mTLS with rustls and cert rotation
- **`KernelSandbox`** - Kernel isolation and resource control

**Authentication & Authorization (feature-gated via `auth`):**
- **`ApiKeyAuth`** - Simple API key validation
- **`JwtAuth`** - JWT token validation (RS256/HS256)
- **`ChainedAuthProvider`** - Fallback authentication chains
- **`RbacPolicy`/`PolicyEvaluator`** - Role-based access control with deny-by-default

**Multi-tenancy:**
- **`TenantContext`** - Request scoping with tenant ID
- **`TenantRegistry`** - Tenant configuration management
- **`ResourceQuota`** - Per-tenant limits (memory, kernels, message rate)

**Rate Limiting (feature-gated via `rate-limiting`):**
- **`RateLimiter`** - TokenBucket, SlidingWindow, LeakyBucket algorithms
- **`RateLimiterBuilder`** - Fluent configuration API
- **`SharedRateLimiter`** - Distributed rate limiting

**Multi-GPU:**
- **`MultiGpuCoordinator`** - Device selection with load balancing strategies
- **`KernelMigrator`** - Live kernel migration between GPUs using checkpoints
- **`GpuTopology`** - NVLink/PCIe topology discovery

**Lifecycle:**
- **`LifecycleState`** - Initializing → Running → Draining → ShuttingDown → Stopped
- **`ShutdownReport`** - Final statistics on graceful shutdown
- **`Timeout`/`Deadline`** - Operation timeouts with deadline propagation

```rust
// Enterprise runtime usage
use ringkernel_core::prelude::*;

let runtime = RuntimeBuilder::new()
    .production()  // or .development() or .high_performance()
    .build()?;

runtime.start()?;  // Transition to Running state

// Run health monitoring
let result = runtime.run_health_check_cycle();
println!("Health: {:?}, Circuit: {:?}", result.status, result.circuit_state);

// Circuit breaker protection
let guard = CircuitGuard::new(&runtime, "operation");
guard.execute(|| { /* protected operation */ })?;

// Graceful shutdown
let report = runtime.complete_shutdown()?;
println!("Uptime: {:?}", report.total_uptime);
```

Run the enterprise demo: `cargo run -p ringkernel --example enterprise_runtime`

### Backend System

Backends implement `RingKernelRuntime` trait. Selection via features:
- `cpu` (default) - Always available
- `cuda` - Requires NVIDIA GPU + CUDA toolkit
- `wgpu` - Cross-platform via WebGPU (Vulkan, Metal, DX12)
- `metal` - macOS/iOS only (scaffold with runtime/buffer/pipeline, no true persistent kernels yet)

Auto-detection: `Backend::Auto` tries CUDA → Metal → WebGPU → CPU.

### Proc Macros (ringkernel-derive)

```rust
// Message definition with field annotations
#[derive(RingMessage)]
#[message(type_id = 1)]  // optional explicit type ID
struct MyMessage {
    #[message(id)]          // MessageId field
    id: MessageId,
    #[message(correlation)] // optional correlation tracking
    correlation: CorrelationId,
    #[message(priority)]    // optional priority
    priority: Priority,
    payload: Vec<f32>,
}

// Kernel handler definition
#[ring_kernel(id = "processor", mode = "persistent", block_size = 128)]
async fn handle(ctx: &mut RingContext, msg: MyMessage) -> MyResponse { ... }

// GPU-compatible types
#[derive(GpuType)]
struct Matrix4x4 {
    data: [f32; 16],
}
```

### CUDA Code Generation (ringkernel-cuda-codegen)

Write GPU kernels in Rust DSL and transpile to CUDA C. Supports three kernel types:

#### 1. Global Kernels (Generic CUDA)

```rust
use ringkernel_cuda_codegen::{transpile_global_kernel};
use syn::parse_quote;

let kernel_fn: syn::ItemFn = parse_quote! {
    fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        y[idx as usize] = a * x[idx as usize] + y[idx as usize];
    }
};
let cuda_code = transpile_global_kernel(&kernel_fn)?;
```

#### 2. Stencil Kernels (GridPos Abstraction)

```rust
use ringkernel_cuda_codegen::{transpile_stencil_kernel, StencilConfig};

let stencil_fn: syn::ItemFn = parse_quote! {
    fn fdtd_step(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
        let laplacian = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p)
                        - 4.0 * p[pos.idx()];
        p_prev[pos.idx()] = 2.0 * p[pos.idx()] - p_prev[pos.idx()] + c2 * laplacian;
    }
};
let config = StencilConfig::new("fdtd").with_tile_size(16, 16).with_halo(1);
let cuda_code = transpile_stencil_kernel(&stencil_fn, &config)?;
```

#### 3. Ring Kernels (Persistent Actor Model)

```rust
use ringkernel_cuda_codegen::{transpile_ring_kernel, RingKernelConfig};

let handler: syn::ItemFn = parse_quote! {
    fn process_message(ctx: &RingContext, msg: &Request) -> Response {
        let tid = ctx.global_thread_id();
        ctx.sync_threads();
        let result = msg.value * 2.0;
        Response { value: result, id: tid as u64 }
    }
};

let config = RingKernelConfig::new("processor")
    .with_block_size(128)
    .with_queue_capacity(1024)
    .with_envelope_format(true)  // 256-byte MessageHeader + payload
    .with_hlc(true)              // Enable Hybrid Logical Clocks
    .with_k2k(true)              // Enable Kernel-to-Kernel messaging
    .with_kernel_id(1000)        // Kernel identity for routing
    .with_hlc_node_id(42);       // HLC node ID

let cuda_code = transpile_ring_kernel(&handler, &config)?;
```

**Envelope Format Features (New):**
- `MessageHeader` (256 bytes, 64-byte aligned) matching Rust `#[repr(C)]` layout
- Magic number validation (`0x52494E474B45524E` = "RINGKERN")
- HLC timestamp propagation from incoming messages
- Correlation ID preservation for request/response tracking
- Source/destination kernel routing

**DSL Features:**
- Block/grid indices: `block_idx_x()`, `thread_idx_x()`, `block_dim_x()`, `grid_dim_x()`, `warp_size()`, etc.
- Control flow: `if/else`, `match` → switch/case, early `return`
- Loops: `for i in 0..n`, `while cond`, `loop` with `break`/`continue`
- Stencil intrinsics (2D): `pos.north(buf)`, `pos.south(buf)`, `pos.east(buf)`, `pos.west(buf)`, `pos.at(buf, dx, dy)`
- Stencil intrinsics (3D): `pos.up(buf)`, `pos.down(buf)`, `pos.at(buf, dx, dy, dz)` for volumetric kernels
- Shared memory: `__shared__` arrays and tiles with `SharedMemoryConfig`
- Struct literals: `Point { x: 1.0, y: 2.0 }` → C compound literals
- Reference expressions: `&arr[idx]` → pointer to element with automatic `->` operator for field access
- 120+ GPU intrinsics across 13 categories (synchronization, atomics, math, trig, hyperbolic, exponential, classification, warp, bit manipulation, memory, special, index, timing)

**Ring Kernel Features:**
- Persistent message loop with ControlBlock lifecycle management
- Envelope-based message serialization with 256-byte headers
- RingContext method inlining (`ctx.thread_id()` → `threadIdx.x`)
- HLC clock operations: `hlc_tick()`, `hlc_update()`, `hlc_now()`
- K2K messaging with envelope format: `k2k_send_envelope()`, `k2k_try_recv_envelope()`, `k2k_peek_envelope()`
- Queue intrinsics: `enqueue_response()`, `input_queue_empty()`, etc.
- Message header validation and correlation ID tracking
- Automatic termination handling and cleanup

#### 4. Persistent FDTD Kernels (True Persistent GPU Actors)

The `ringkernel-cuda-codegen` crate includes `persistent_fdtd.rs` for generating truly persistent GPU kernels that run for the entire simulation lifetime:

```rust
use ringkernel_cuda_codegen::persistent_fdtd::{generate_persistent_fdtd_kernel, PersistentFdtdConfig};

let config = PersistentFdtdConfig::new("persistent_fdtd3d")
    .with_tile_size(8, 8, 8)
    .with_cooperative(true)       // Use cooperative groups for grid.sync()
    .with_progress_interval(100); // Report progress every 100 steps

let cuda_code = generate_persistent_fdtd_kernel(&config);
// Compile with: nvcc -ptx -arch=native -std=c++17 -rdc=true
```

**Persistent GPU Actor Architecture:**
- **H2K/K2H Messaging** - Host↔Kernel lock-free SPSC queues via CUDA mapped memory
- **K2K Halo Exchange** - Kernel-to-kernel tile boundary communication on device memory
- **Cooperative Groups** - Grid-wide synchronization via `cg::grid_group::sync()`
- **PersistentControlBlock** (256 bytes) - Lifecycle management in mapped memory

**Key Structures:**
- `PersistentControlBlock` - 256-byte control block with step counters, physics params, sync barriers
- `H2KMessage` (64 bytes) - Commands: RunSteps, Terminate, InjectImpulse, GetProgress
- `K2HMessage` (64 bytes) - Responses: Ack, Progress, Error, Terminated, Energy
- `K2KRouteEntry` - Neighbor block IDs for 3D halo exchange routing

**Runtime Integration (`ringkernel-cuda/src/persistent.rs`):**
```rust
use ringkernel_cuda::persistent::{PersistentSimulation, PersistentSimulationConfig};

let config = PersistentSimulationConfig::new(width, height, depth)
    .with_tile_size(8, 8, 8)
    .with_acoustics(speed_of_sound, cell_size, damping);

let mut sim = PersistentSimulation::new(&device, config)?;
sim.start(&ptx, "persistent_fdtd3d")?;  // Single kernel launch
sim.run_steps(100)?;                     // Send command via control block
let stats = sim.stats();                 // Read from mapped memory
sim.shutdown()?;                         // Graceful termination
```

**Throughput Benchmark Results (RTX Ada, 64³ grid):**
| Method | Throughput | vs CPU |
|--------|-----------|--------|
| GPU Stencil | 78,046 Mcells/s | 280.6x |
| GPU Persistent | 18.2 Mcells/s | 1.2x |
| CPU (Rayon) | 278 Mcells/s | 1.0x |

**Interactive Benchmark Results (Persistent vs Traditional):**
| Operation | Traditional | Persistent | Winner |
|-----------|-------------|------------|--------|
| **Inject (send command)** | 317 µs | 0.03 µs | **Persistent 11,327x** |
| Query (read state) | 0.01 µs | 0.01 µs | Tie |
| Single step (compute) | 3.2 µs | 163 µs | Traditional 51x |
| **Mixed workload** | 40.5 ms | 15.3 ms | **Persistent 2.7x** |

**When to Use Each Approach:**

| Use Case | Best Approach | Why |
|----------|---------------|-----|
| Batch compute (1000s of steps) | Traditional | Lower per-step overhead |
| Interactive commands | **Persistent** | 11,327x faster command injection |
| Real-time GUI (60 FPS) | **Persistent** | 2.7x more ops per 16ms frame |
| Dynamic topology | **Persistent** | No kernel relaunch needed |
| Pure FDTD/matrix math | Traditional | Maximum throughput |

The persistent actor model excels at **interactive command latency**—commands are written to mapped memory without kernel launch overhead. Traditional kernels excel at **batch compute** (running thousands of steps at once).

### WGSL Code Generation (ringkernel-wgpu-codegen)

Write GPU kernels in Rust DSL and transpile to WGSL for WebGPU. Provides full parity with CUDA codegen:

```rust
use ringkernel_wgpu_codegen::{transpile_global_kernel, transpile_stencil_kernel, StencilConfig};
use syn::parse_quote;

// Global kernel
let kernel_fn: syn::ItemFn = parse_quote! {
    fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        y[idx as usize] = a * x[idx as usize] + y[idx as usize];
    }
};
let wgsl_code = transpile_global_kernel(&kernel_fn)?;

// Stencil kernel (same API as CUDA)
let config = StencilConfig::new("fdtd").with_tile_size(16, 16).with_halo(1);
let wgsl_code = transpile_stencil_kernel(&stencil_fn, &config)?;
```

**WGSL Limitations vs CUDA (handled with workarounds):**
- No 64-bit atomics: Emulated using lo/hi u32 pairs
- No f64: Downcast to f32 with warning
- No persistent kernels: Host-driven dispatch loop emulation
- No K2K messaging: Not supported in WebGPU execution model

### K2K (Kernel-to-Kernel) Messaging

Direct messaging between kernels without going through the host application:

```rust
// Runtime with K2K enabled (default)
let runtime = CpuRuntime::new().await?;
assert!(runtime.is_k2k_enabled());

// Access the broker
let broker = runtime.k2k_broker().unwrap();

// Register kernels and send messages
let endpoint = broker.register(kernel_id);
endpoint.send(destination_id, envelope).await?;
```

### Serialization

Uses rkyv for zero-copy serialization. Messages must derive `rkyv::Archive`, `rkyv::Serialize`, `rkyv::Deserialize`. The `RingMessage` derive macro generates serialization methods automatically.

### Feature Flags

Main crate (`ringkernel`) features:
- `cpu` (default) - CPU backend
- `cuda` - NVIDIA CUDA backend
- `wgpu` - WebGPU cross-platform backend
- `metal` - Apple Metal backend
- `all-backends` - All GPU backends

CUDA-specific features (`ringkernel-cuda`):
- `cooperative` - Enable CUDA cooperative groups for grid-wide synchronization (`grid.sync()`). Requires nvcc at build time for PTX compilation.
- `profiling` - GPU profiling infrastructure (CUDA events, NVTX, memory tracking, Chrome trace export). Requires nvToolsExt library.

Core crate (`ringkernel-core`) enterprise features:
- `crypto` - Real cryptography (AES-256-GCM, ChaCha20-Poly1305, Argon2)
- `auth` - JWT authentication support (jsonwebtoken crate)
- `rate-limiting` - Governor-based rate limiting
- `alerting` - Webhook alerts via reqwest
- `tls` - TLS support via rustls
- `enterprise` - Combined feature enabling all enterprise features

Ecosystem crate (`ringkernel-ecosystem`) features:
- `persistent` - Core persistent GPU kernel traits (backend-agnostic)
- `persistent-cuda` - CUDA implementation of `PersistentHandle` via `CudaPersistentHandle`
- `actix` - Actix actor framework bridge with `GpuPersistentActor`
- `tower` - Tower service middleware with `PersistentKernelService`
- `axum` - Axum web framework with `PersistentGpuState` and REST/SSE endpoints
- `grpc` - gRPC server with streaming RPCs via Tonic
- `persistent-full` - Full persistent ecosystem (CUDA + all web frameworks)

### Ecosystem Integrations (ringkernel-ecosystem)

The ecosystem crate provides web framework integrations with **11,327x faster command injection** (~0.03µs vs ~317µs):

```rust
// Actix actor with persistent GPU kernel
use ringkernel_ecosystem::actix::{GpuPersistentActor, RunStepsCmd, InjectCmd};
let actor = GpuPersistentActor::new(handle, config).start();
actor.send(RunStepsCmd::new(1000)).await?;

// Axum REST API
use ringkernel_ecosystem::axum::{PersistentGpuState, PersistentAxumConfig};
let state = PersistentGpuState::new(handle, PersistentAxumConfig::default());
let app = Router::new().merge(state.routes());

// Tower service with middleware
use ringkernel_ecosystem::tower::{PersistentKernelService, PersistentServiceBuilder};
let service = PersistentServiceBuilder::new(handle).build();

// CUDA bridge for real GPU kernels
use ringkernel_ecosystem::cuda_bridge::CudaPersistentHandle;
let handle = CudaPersistentHandle::new(simulation, "fdtd_3d");
```

## Testing Patterns

- Unit tests in each crate's `src/` using `#[cfg(test)]`
- Integration tests use `#[tokio::test]` for async runtime
- GPU tests use `#[ignore]` attribute and feature flags (`wgpu-tests`, `cuda`)
- Property-based testing via `proptest` for queue/serialization invariants
- Benchmarks in `crates/ringkernel/benches/` using Criterion

### Test Count Summary

950+ tests across the workspace:
- ringkernel-core: 538 tests (including memory pool, analytics context, pressure reactions, enterprise security, auth, RBAC, tenancy, rate limiting, TLS, logging, alerting, recovery, benchmark framework, hybrid dispatcher, resource guard, partitioned queues)
- ringkernel-cpu: 11 tests
- ringkernel-cuda: 52+ tests (reduction cache, phases, K2K, persistent actors, PTX cache, GPU memory pool, stream manager, kernel mode selection)
- ringkernel-cuda-codegen: 190+ tests (loops, shared memory, ring kernels, K2K, envelope format, energy calculation, checksums, 120+ GPU intrinsics)
- ringkernel-wgpu-codegen: 55+ tests (types, intrinsics, transpiler, validation, 2D/3D/4D shared memory)
- ringkernel-ir: 40+ tests (IR nodes, CUDA lowering, MSL lowering, messaging nodes, HLC nodes)
- ringkernel-derive: 14 macro tests
- ringkernel-ecosystem: 30 tests (persistent handle, CUDA bridge, Actix, Axum, Tower, gRPC integrations)
- ringkernel-wavesim: 63 tests (including educational modes, tile actor kernels, envelope format)
- ringkernel-wavesim3d: 75+ tests (3D FDTD, binaural audio, volumetric rendering, block actor kernels, cooperative launch)
- ringkernel-txmon: 40 tests (GPU types, batch kernel, stencil kernel, ring kernel backends)
- ringkernel-procint: 77 tests (DFG construction, pattern detection, partial order, conformance checking)
- ringkernel-accnet: 25+ tests (chart of accounts, industry templates, GAAP compliance)
- ringkernel-montecarlo: 16 tests (Philox RNG, antithetic variates, control variates, importance sampling)
- ringkernel-graph: 55+ tests (CSR matrix, BFS, SCC algorithms, Union-Find with Shiloach-Vishkin, SpMV, power iteration)
- k2k_integration: 11 tests
- control_block: 29 tests
- hlc: 16 tests
- ringkernel-audio-fft: 35+ tests (including resampling tests)
- Plus additional integration and doc tests

### GPU Benchmark Results (RTX Ada)

The `ringkernel-cuda-codegen` transpiler generates CUDA code that achieves impressive performance:

| Backend | Throughput | Batch Time | Speedup vs CPU |
|---------|------------|------------|----------------|
| CUDA Codegen (1M floats) | ~93B elem/sec | 0.5 µs | 12,378x |
| CUDA SAXPY PTX | ~77B elem/sec | 0.6 µs | 10,258x |
| GPU Stencil (CPU fallback) | ~15.7M TPS | 262 µs | 2.09x |
| CPU Baseline | ~7.5M TPS | 547 µs | 1.00x |

Run benchmark: `cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen`

### WaveSim Educational Modes

The WaveSim application includes educational simulation modes that visually demonstrate the evolution of parallel computing:

```rust
use crate::simulation::{SimulationMode, EducationalProcessor};

// Available modes:
// - Standard: Full-speed parallel (default)
// - CellByCell: 1950s sequential processing
// - RowByRow: 1970s vector processing (Cray-style)
// - ChaoticParallel: 1990s uncoordinated parallelism (shows race conditions)
// - SynchronizedParallel: 2000s barrier-synchronized parallelism
// - ActorBased: Modern tile-based actors with HLC

let mut processor = EducationalProcessor::new(SimulationMode::CellByCell);
processor.cells_per_frame = 16; // Cells processed per animation frame
```

Visual indicators show:
- **Green cells**: Currently being processed
- **Yellow row**: Active row (RowByRow mode)
- **Cyan tile with border**: Active tile (ActorBased mode)

## Publishing

All crates are published to crates.io under the `ringkernel-*` namespace. Use the publish script to handle the dependency order:

```bash
# Check which crates are published vs pending
./scripts/publish.sh --status

# Publish all unpublished crates (auto-skips already published)
./scripts/publish.sh <CRATES_IO_TOKEN>

# Dry run to verify without publishing
./scripts/publish.sh --dry-run
```

The script automatically:
- Skips already-published crates (safe to run multiple times)
- Publishes in correct dependency order (tier by tier)
- Handles crates.io rate limits (~5 crates per 10 minutes)
- Shows status before and after publishing

Crate publishing order:
1. **Tier 1** (no deps): core, cuda-codegen, wgpu-codegen
2. **Tier 2** (depends on core): derive, cpu, cuda, wgpu, metal, codegen, ecosystem, audio-fft
3. **Tier 3** (main crate): ringkernel
4. **Tier 4** (applications): wavesim, wavesim3d, txmon, accnet, procint

## Important Patterns

**cudarc 0.18.2 API (Updated):**

The CUDA backend uses cudarc 0.18.2 with the following patterns:

```rust
// Module loading (NEW in 0.18.2)
let module = device.inner().load_module(ptx)?;  // Returns Arc<CudaModule>
let func = module.load_function("kernel_name")?; // Load specific function

// Kernel launch with builder pattern (NEW in 0.18.2)
use cudarc::driver::PushKernelArg;
unsafe {
    stream
        .launch_builder(&func)
        .arg(&input_ptr)
        .arg(&output_ptr)
        .arg(&scalar_param)
        .launch(cfg)?;
}

// Cooperative kernel launch (uses cudarc's result module)
use cudarc::driver::result as cuda_result;
unsafe {
    cuda_result::launch_cooperative_kernel(
        func, grid_dim, block_dim, shared_mem_bytes, stream, kernel_params
    )?;
}
```

Old API (cudarc 0.11) patterns that no longer work:
- `device.load_ptx(ptx, module_name, &[func_names])` → Use `load_module()` + `load_function()`
- `device.get_func(module_name, fn_name)` → Store functions at construction time
- `func.launch(cfg, params)` → Use `stream.launch_builder(&func).arg(...).launch(cfg)`

**Queue capacity must be power of 2:**
```rust
// Correct
let options = LaunchOptions::default().with_queue_capacity(1024);

// Or calculate it
let capacity = desired_size.next_power_of_two();
```

**Avoid double activation:**
```rust
// Kernels auto-activate by default, so this will error:
let kernel = runtime.launch("test", LaunchOptions::default()).await?;
kernel.activate().await?;  // Error: "Active to Active"

// Use without_auto_activate for manual control:
let kernel = runtime.launch("test",
    LaunchOptions::default().without_auto_activate()
).await?;
kernel.activate().await?;  // OK
```

**Result type in examples:**
```rust
// The prelude exports Result<T> which conflicts with std::result::Result<T, E>
// Use explicit type in main functions:
#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // ...
}
```

**wgpu 27.0 API (Updated):**

The WebGPU backend uses wgpu 27.0 with Arc-based resource tracking:

```rust
// Instance creation (takes reference now)
let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
    backends: wgpu::Backends::all(),
    ..Default::default()
});

// Adapter request (returns Result, not Option)
let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptions { ... })
    .await
    .map_err(|e| format!("No adapter: {}", e))?;

// Device request (use ..Default::default() for new fields)
let (device, queue) = adapter
    .request_device(&wgpu::DeviceDescriptor {
        label: Some("device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()  // memory_hints, etc.
    })
    .await?;

// Pipeline creation (entry_point is now Option, add compilation_options and cache)
let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs_main"),  // Option<&str> now
        compilation_options: Default::default(),
        buffers: &[...],
    },
    fragment: Some(wgpu::FragmentState {
        entry_point: Some("fs_main"),
        compilation_options: Default::default(),
        ...
    }),
    cache: None,  // New required field
    ...
});

// Texture copy (renamed types)
queue.write_texture(
    wgpu::TexelCopyTextureInfo { ... },  // was ImageCopyTexture
    &data,
    wgpu::TexelCopyBufferLayout { ... }, // was ImageDataLayout
    size,
);

// Device polling (returns Result now)
let _ = device.poll(wgpu::PollType::wait_indefinitely());
```

## Dependency Versions

Key workspace dependencies (as of v0.4.0):

| Category | Package | Version | Notes |
|----------|---------|---------|-------|
| **Runtime** | tokio | 1.48 | Full async runtime |
| **Runtime** | rayon | 1.11 | Parallel iterators |
| **Error** | thiserror | 2.0 | Derive macros |
| **GPU** | cudarc | 0.18.2 | CUDA bindings |
| **GPU** | wgpu | 27.0 | WebGPU (Arc-based) |
| **GPU** | metal | 0.31 | Apple Metal |
| **Crypto** | sha2 | 0.10 | PTX cache hashing |
| **Web** | axum | 0.8 | HTTP framework |
| **Web** | tower | 0.5 | Service abstractions |
| **gRPC** | tonic | 0.14 | gRPC framework |
| **gRPC** | prost | 0.14 | Protobuf |
| **Serialize** | rkyv | 0.7 | Zero-copy |
| **GUI** | iced | 0.13 | Elm-style GUI |
| **GUI** | egui | 0.31 | Immediate mode GUI |
| **GUI** | winit | 0.30 | Window management |
| **Math** | glam | 0.29 | Linear algebra |
| **Data** | arrow | 54 | Columnar data |
| **Data** | polars | 0.46 | DataFrames |
