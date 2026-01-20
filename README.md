# RingKernel

[![Crates.io](https://img.shields.io/crates/v/ringkernel-core.svg)](https://crates.io/crates/ringkernel-core)
[![Documentation](https://docs.rs/ringkernel-core/badge.svg)](https://docs.rs/ringkernel-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://mivertowski.github.io/RustCompute/)

A GPU-native persistent actor model framework for Rust.

RingKernel provides infrastructure for building GPU-accelerated actor systems with persistent kernels, lock-free message passing, and hybrid logical clocks for causal ordering.

## Overview

RingKernel treats GPU compute units as long-running actors that maintain state between invocations. Instead of launching kernels per-operation, kernels persist on the GPU, communicate via lock-free queues, and process messages continuously.

**Key capabilities:**

- Persistent GPU-resident kernels that maintain state across invocations
- Lock-free message queues for host↔GPU and kernel-to-kernel communication
- Hybrid Logical Clocks (HLC) for causal ordering across distributed operations
- Multiple backend support: CPU, CUDA, WebGPU
- Zero-copy serialization via rkyv
- Proc macros for message and kernel definitions

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ringkernel = "0.3"
tokio = { version = "1.48", features = ["full"] }
```

For GPU backends:

```toml
# NVIDIA CUDA
ringkernel = { version = "0.3", features = ["cuda"] }

# WebGPU (cross-platform)
ringkernel = { version = "0.3", features = ["wgpu"] }
```

## Quick Start

```rust
use ringkernel::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create runtime (uses CPU backend by default)
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await?;

    // Launch a kernel
    let kernel = runtime.launch("processor", LaunchOptions::default()).await?;

    // Kernel is active and ready to process messages
    println!("State: {:?}", kernel.state());

    // Lifecycle management
    kernel.deactivate().await?;
    kernel.activate().await?;
    kernel.terminate().await?;

    runtime.shutdown().await?;
    Ok(())
}
```

## CLI Tool

RingKernel includes a CLI for project scaffolding and code generation:

```bash
# Install the CLI
cargo install ringkernel-cli

# Create a new project with persistent actor template
ringkernel new my-gpu-app --template persistent-actor

# Generate CUDA and WGSL code from kernel file
ringkernel codegen src/kernels/processor.rs --backend cuda,wgsl

# Check all kernels for backend compatibility
ringkernel check --backends all

# Generate shell completions
ringkernel completions bash > ~/.bash_completion.d/ringkernel
```

Templates available: `basic`, `persistent-actor`, `wavesim`, `enterprise`

## Enterprise Runtime

For production deployments, use the enterprise runtime with built-in health checking, circuit breakers, and observability:

```rust
use ringkernel::prelude::*;
use ringkernel_core::runtime_context::RuntimeBuilder;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create production-ready runtime
    let runtime = RuntimeBuilder::new()
        .production()  // Pre-configured for production
        .build()?;

    runtime.start()?;  // Transition to Running state

    // Health monitoring
    let health = runtime.run_health_check_cycle();
    println!("Status: {:?}, Circuit: {:?}", health.status, health.circuit_state);

    // Graceful shutdown with drain
    let report = runtime.complete_shutdown()?;
    println!("Uptime: {:?}, Kernels migrated: {}", report.total_uptime, report.migrated_kernels);

    Ok(())
}
```

Enterprise features include:
- **Health & Resilience**: HealthChecker, CircuitBreaker, DegradationManager, KernelWatchdog, RecoveryManager
- **Observability**: PrometheusExporter, OtlpExporter, StructuredLogger with trace correlation, AlertRouter
- **Multi-GPU**: MultiGpuCoordinator, KernelMigrator, GpuTopology discovery
- **Security**: MemoryEncryption (AES-256-GCM, ChaCha20), KernelSandbox, K2K encryption, TLS/mTLS
- **Authentication**: ApiKeyAuth, JwtAuth, ChainedAuthProvider, RBAC with PolicyEvaluator
- **Multi-tenancy**: TenantContext, ResourceQuota, QuotaUtilization monitoring
- **Rate Limiting**: TokenBucket, SlidingWindow, LeakyBucket algorithms
- **Secrets Management**: SecretStore trait, KeyRotationManager, CachedSecretStore

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Host (CPU)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Application │──│   Runtime   │──│   Message Bridge    │  │
│  │   (async)   │  │   (tokio)   │  │   (DMA transfers)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │ PCIe / Unified Memory
┌────────────────────────────┴────────────────────────────────┐
│                      Device (GPU)                           │
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │ Control Block │  │   Input Queue   │  │ Output Queue  │  │
│  │    (128 B)    │  │  (lock-free)    │  │ (lock-free)   │  │
│  └───────────────┘  └─────────────────┘  └───────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  Persistent Kernel                      ││
│  │        (maintains state between messages)               ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Backends

| Backend | Status | Platforms | Requirements |
|---------|--------|-----------|--------------|
| CPU | Stable | All | None |
| CUDA | Stable | Linux, Windows | NVIDIA GPU, CUDA 12.x, cudarc 0.18.2 |
| WebGPU | Stable | All | Vulkan/Metal/DX12 capable GPU, wgpu 27.0 |
| Metal | Scaffold | macOS, iOS | Apple GPU, metal-rs 0.31 |

## Core Concepts

### Kernel Lifecycle

Kernels follow a state machine: `Created → Launched → Active ⇄ Deactivated → Terminated`

```rust
// Manual lifecycle control
let kernel = runtime.launch("worker",
    LaunchOptions::default().without_auto_activate()
).await?;

kernel.activate().await?;    // Start processing
kernel.deactivate().await?;  // Pause
kernel.activate().await?;    // Resume
kernel.terminate().await?;   // Clean shutdown
```

### Launch Options

```rust
let options = LaunchOptions::default()
    .with_queue_capacity(4096)      // Must be power of 2
    .with_block_size(256)           // Threads per block
    .with_priority(priority::HIGH)
    .with_cooperative(true)         // Enable grid-wide sync (CUDA cooperative groups)
    .without_auto_activate();
```

### Global Reduction Primitives

GPU-accelerated global reductions for algorithms like PageRank with dangling nodes:

```rust
use ringkernel_cuda::reduction::{ReductionBuffer, ReductionBufferBuilder};
use ringkernel_core::reduction::{ReductionOp, ReductionConfig};

// Create a reduction buffer for summing dangling node contributions
let buffer = ReductionBufferBuilder::new()
    .with_op(ReductionOp::Sum)
    .with_slots(4)  // Multiple slots reduce contention
    .build(&device)?;

// In kernel: block-level reduction + atomic accumulation
// grid_reduce_sum(local_value, shared_mem, buffer.device_ptr())
```

Multi-phase kernel support with inter-phase synchronization:

```rust
use ringkernel_cuda::phases::{MultiPhaseConfig, SyncMode};

let config = MultiPhaseConfig::new()
    .with_sync_mode(SyncMode::Cooperative)  // Uses grid.sync()
    .with_phase("accumulate", accumulate_fn)
    .with_phase("apply", apply_fn);

// Cooperative groups for CC 6.0+, software barrier fallback for older GPUs
```

### Memory Pool Management

Efficient buffer reuse for analytics workloads:

```rust
use ringkernel_core::prelude::*;

// Size-stratified pool with automatic bucket selection
let pool = StratifiedMemoryPool::new("analytics");

// Allocate various sizes - each goes to appropriate bucket
let tiny_buf = pool.allocate(100);    // Uses Tiny bucket (256B)
let medium_buf = pool.allocate(2000); // Uses Medium bucket (4KB)

// Check statistics
let stats = pool.stats();
println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);
```

Analytics context for grouped buffer lifecycle:

```rust
use ringkernel_core::analytics_context::AnalyticsContext;

// Create context for a BFS operation
let mut ctx = AnalyticsContext::new("bfs_traversal");

// Allocate buffers - all released when context drops
let frontier = ctx.allocate(1024);
let visited = ctx.allocate_typed::<u32>(256);

// Check stats
println!("Peak memory: {} bytes", ctx.stats().peak_bytes);
```

CUDA reduction buffer cache for PageRank-style algorithms:

```rust
use ringkernel_cuda::{ReductionBufferCache, ReductionOp};

let cache = ReductionBufferCache::new(&device, 4);

// First acquire allocates, subsequent acquires reuse
let buffer = cache.acquire::<f32>(4, ReductionOp::Sum)?;
// Buffer automatically returned to cache on drop
```

### Multi-Kernel Dispatch

Route messages to GPU kernels based on message type:

```rust
use ringkernel_core::prelude::*;

// Define a persistent message with automatic handler ID
#[derive(PersistentMessage)]
struct ComputeTask {
    data: Vec<f32>,
}

// Create dispatcher with K2K broker
let dispatcher = DispatcherBuilder::new(k2k_broker)
    .with_default_priority(Priority::Normal)
    .build();

// Dispatch routes to appropriate kernel based on message type
dispatcher.dispatch(kernel_id, envelope).await?;

// Check metrics
let metrics = dispatcher.metrics();
println!("Dispatched: {}, Errors: {}", metrics.messages_dispatched, metrics.errors);
```

### Queue Tiering

Select queue capacity based on expected throughput:

```rust
use ringkernel_core::queue::{QueueTier, QueueFactory, QueueMonitor};

// Automatic tier selection based on message rate
let tier = QueueTier::for_throughput(10_000, 100); // 10k msg/s, 100ms buffer
// Returns QueueTier::Medium (1024 capacity)

// Manual tier selection
let queue = QueueFactory::create_spsc(QueueTier::Large); // 4096 capacity

// Monitor queue health
let monitor = QueueMonitor::new(queue, QueueMonitorConfig::default());
let health = monitor.check_health();
println!("Depth: {}, Healthy: {}", health.current_depth, health.is_healthy);
```

Available tiers: `Small` (256), `Medium` (1024), `Large` (4096), `ExtraLarge` (16384).

### Kernel-to-Kernel Messaging

Direct communication between kernels:

```rust
// K2K is enabled by default
let runtime = CpuRuntime::new().await?;
let broker = runtime.k2k_broker().unwrap();

// Kernels can send messages directly to each other
let receipt = broker.send(source_id, dest_id, envelope).await?;
```

### Hybrid Logical Clocks

HLC provides causal ordering across distributed operations:

```rust
let clock = HlcClock::new(node_id);

let ts1 = clock.tick();
let ts2 = clock.tick();
assert!(ts1 < ts2);

// Synchronize with remote timestamp
let remote_ts = receive_timestamp();
let synced = clock.update(&remote_ts)?;
```

### Pub/Sub Messaging

Topic-based messaging with wildcard support:

```rust
let broker = PubSubBroker::new(PubSubConfig::default());

// Subscribe with wildcards: * (single level), # (multi-level)
broker.subscribe(kernel_id, Topic::new("sensors/+/temperature"));
broker.subscribe(kernel_id, Topic::new("events/#"));

// Publish
broker.publish(Topic::new("sensors/room1/temperature"), sender, envelope, timestamp)?;
```

### Proc Macros

Define messages and kernels declaratively:

```rust
use ringkernel::prelude::*;

#[derive(RingMessage)]
#[message(type_id = 1)]
struct ComputeRequest {
    #[message(id)]
    id: MessageId,
    data: Vec<f32>,
}

#[derive(GpuType)]
struct Matrix4x4 {
    data: [f32; 16],
}
```

### Domain System

Organize messages by business domain with automatic type ID allocation:

```rust
use ringkernel::prelude::*;

// 20 predefined domains with reserved type ID ranges
#[derive(RingMessage)]
#[message(domain = "FraudDetection")]  // Type IDs 1000-1099
struct SuspiciousTransaction {
    #[message(id)]
    id: MessageId,
    amount: f64,
    risk_score: f32,
}

#[derive(RingMessage)]
#[message(domain = "ProcessIntelligence")]  // Type IDs 1500-1599
struct ActivityEvent {
    #[message(id)]
    id: MessageId,
    case_id: u64,
    activity: String,
}

// Check domain at runtime
let domain = Domain::from_type_id(1050);  // Returns Some(Domain::FraudDetection)
```

Available domains: `GraphAnalytics`, `StatisticalML`, `Compliance`, `RiskManagement`, `OrderMatching`, `MarketData`, `Settlement`, `Accounting`, `NetworkAnalysis`, `FraudDetection`, `TimeSeries`, `Simulation`, `Banking`, `BehavioralAnalytics`, `ProcessIntelligence`, `Clearing`, `TreasuryManagement`, `PaymentProcessing`, `FinancialAudit`, `Custom`.

## Examples

The repository includes 20+ working examples organized by category:

```bash
# === Basic Examples ===
cargo run -p ringkernel --example basic_hello_kernel    # Runtime, lifecycle, suspend/resume
cargo run -p ringkernel --example kernel_states         # State machine, multi-kernel

# === Messaging Examples ===
cargo run -p ringkernel --example request_response      # Correlation IDs, priorities
cargo run -p ringkernel --example pub_sub               # Topic wildcards, QoS
cargo run -p ringkernel --example kernel_to_kernel      # K2K direct messaging
cargo run -p ringkernel --example ping_pong             # K2K broker usage

# === CUDA Codegen Examples ===
cargo run -p ringkernel --example global_kernel         # SAXPY, halo exchange, array init
cargo run -p ringkernel --example stencil_kernel        # FDTD wave, heat diffusion, GridPos
cargo run -p ringkernel --example ring_kernel_codegen   # Persistent kernels, HLC, K2K

# === Advanced Examples ===
cargo run -p ringkernel --example educational_modes     # WaveSim parallel computing modes
cargo run -p ringkernel --example multi_gpu             # Multi-GPU load balancing

# === Integration Examples ===
cargo run -p ringkernel --example axum_api              # REST API with Axum
cargo run -p ringkernel --example grpc_server           # gRPC patterns
cargo run -p ringkernel --example batch_processor       # Data pipelines
cargo run -p ringkernel --example telemetry             # Metrics, alerts

# === WebGPU ===
cargo run -p ringkernel --example wgpu_hello --features wgpu
```

## Showcase Applications

RingKernel includes five comprehensive showcase applications demonstrating GPU-accelerated computing.

| App | Description | Run Command |
|-----|-------------|-------------|
| **WaveSim** | 2D acoustic wave simulation with FDTD | `cargo run -p ringkernel-wavesim --release` |
| **WaveSim3D** | 3D acoustic wave simulation with binaural audio | `cargo run -p ringkernel-wavesim3d --release` |
| **TxMon** | Real-time transaction monitoring | `cargo run -p ringkernel-txmon --release` |
| **AccNet** | Accounting network analytics | `cargo run -p ringkernel-accnet --release` |
| **ProcInt** | Process intelligence with DFG mining | `cargo run -p ringkernel-procint --release` |

See the [Showcase Applications Guide](docs/15-showcase-applications.md) for detailed documentation with screenshots.

## Crate Structure

| Crate | Description |
|-------|-------------|
| `ringkernel` | Main facade crate |
| `ringkernel-core` | Core traits, types, HLC, K2K, PubSub, enterprise runtime, security |
| `ringkernel-derive` | Proc macros (`#[derive(RingMessage)]`, `#[ring_kernel]`) |
| `ringkernel-ir` | Unified IR for multi-backend code generation (CUDA/WGSL/MSL) |
| `ringkernel-cli` | CLI tool for scaffolding, codegen, and validation |
| `ringkernel-cpu` | CPU backend with mock GPU testing |
| `ringkernel-cuda` | NVIDIA CUDA backend with cooperative groups support (cudarc 0.18.2) |
| `ringkernel-wgpu` | WebGPU backend |
| `ringkernel-codegen` | GPU kernel code generation |
| `ringkernel-cuda-codegen` | Rust-to-CUDA transpiler for writing GPU kernels in Rust DSL |
| `ringkernel-wgpu-codegen` | Rust-to-WGSL transpiler for writing GPU kernels in Rust DSL (WebGPU) |
| `ringkernel-ecosystem` | Framework integrations (Actix, Axum, Tower, gRPC) + ML bridges |
| `ringkernel-wavesim` | 2D wave simulation with tile-based FDTD, ring kernel actors with K2K messaging, and educational modes |
| `ringkernel-wavesim3d` | 3D acoustic wave simulation with binaural audio, persistent GPU actors (H2K/K2H messaging, K2K halo exchange, cooperative groups), and volumetric rendering |
| `ringkernel-txmon` | Transaction monitoring showcase with GPU-accelerated fraud detection |
| `ringkernel-accnet` | Accounting network analytics with fraud detection and GAAP compliance |
| `ringkernel-procint` | Process intelligence with DFG mining, pattern detection, conformance checking |
| `ringkernel-montecarlo` | Monte Carlo primitives: Philox RNG, antithetic/control variates, importance sampling |
| `ringkernel-graph` | Graph algorithms: CSR matrix, BFS, SCC (Tarjan/Kosaraju), Union-Find, SpMV |

## Testing

```bash
# Run all tests (900+ tests)
cargo test --workspace

# CUDA backend tests (requires NVIDIA GPU)
cargo test -p ringkernel-cuda --test gpu_execution_verify

# WebGPU backend tests (requires GPU)
cargo test -p ringkernel-wgpu --features wgpu-tests -- --ignored

# Benchmarks
cargo bench --package ringkernel
```

## Performance

Benchmarked on NVIDIA RTX Ada hardware:

**CUDA Codegen Transpiler Performance:**

| Operation | Throughput | Batch Time | vs CPU |
|-----------|------------|------------|--------|
| CUDA Codegen (1M elements) | ~93B elem/sec | 0.5 µs | 12,378x |
| CUDA SAXPY PTX | ~77B elem/sec | 0.6 µs | 10,258x |
| Stencil Pattern Detection | ~15.7M TPS | 262 µs | 2.09x |

**Core System Metrics:**
- Message queue throughput: ~75M operations/sec
- Host-to-device bandwidth: ~7.6 GB/s (PCIe 4.0)
- HLC timestamp generation: <10ns per tick

### WaveSim Showcase Application

The `ringkernel-wavesim` crate demonstrates RingKernel's capabilities with a 2D acoustic wave simulation using tile-based actors. Performance highlights:

| Backend | Grid Size | Steps/sec | Throughput |
|---------|-----------|-----------|------------|
| CPU (SoA + SIMD + Rayon) | 256×256 | 35,418 | 2.3B cells/s |
| CUDA Packed (GPU-only halo) | 128×128 | 100,000+ | 100M+ cells/s |
| CUDA Packed (GPU-only halo) | 256×256 | 112,837 | 7.4M cells/s |
| CUDA Packed (GPU-only halo) | 512×512 | 71,324 | 18.7M cells/s |

**GPU vs CPU speedup at 512×512: 9.9x**

The CUDA Packed backend demonstrates GPU-only halo exchange—all tile communication happens via GPU memory copies with zero host transfers during simulation. This eliminates the traditional bottleneck of host-GPU synchronization for stencil computations.

**Features:**
- Interactive GUI with real-time visualization
- Drawing mode for absorbers and reflectors to create interference patterns
- Multiple backends: CPU (SIMD), CUDA tile-based, CUDA packed
- **Educational simulation modes** for teaching parallel computing concepts

**Educational Modes:**

WaveSim includes animated simulation modes that visually demonstrate the evolution of parallel computing:

| Mode | Era | Description |
|------|-----|-------------|
| Standard | Modern | Full-speed parallel (default) |
| CellByCell | 1950s | Sequential processing, one cell at a time |
| RowByRow | 1970s | Vector processing (Cray-style), row-by-row |
| ChaoticParallel | 1990s | Uncoordinated parallelism showing race conditions |
| SynchronizedParallel | 2000s | Barrier-synchronized parallelism |
| ActorBased | Modern | Tile-based actors with HLC for causal ordering |

Visual indicators highlight processing: green for active cells, yellow for current row (RowByRow), cyan tiles with borders for active tiles (ActorBased).

```bash
# Run WaveSim GUI
cargo run -p ringkernel-wavesim --bin wavesim --release --features cuda

# Run WaveSim benchmarks
cargo run -p ringkernel-wavesim --bin bench_packed --release --features cuda
```

See [WaveSim PERFORMANCE.md](crates/ringkernel-wavesim/PERFORMANCE.md) for detailed analysis.

### WaveSim3D Showcase Application

The `ringkernel-wavesim3d` crate provides 3D acoustic wave simulation with binaural audio, demonstrating **truly persistent GPU actors**:

**GPU Actor Benchmark Results (64³ grid, 100 steps):**

| Method | Time | Throughput | vs CPU |
|--------|------|-----------|--------|
| CPU (Rayon) | 94 ms | 278 Mcells/s | 1.0x |
| **GPU Stencil** | 0.34 ms | **78,046 Mcells/s** | **280.6x** |
| GPU Block Actor | 1,748 ms | 15 Mcells/s | 0.1x |
| **GPU Persistent** | 76 ms | 18.2 Mcells/s | **1.2x** |
| GPU Actor (per-cell) | 2,772 ms | 9.5 Mcells/s | 0.03x |

**Key Features:**
- **Persistent GPU Actors** - Single kernel launch runs for entire simulation lifetime
- **H2K/K2H Messaging** - Host↔Kernel communication via mapped memory queues
- **K2K Halo Exchange** - Kernel-to-kernel tile boundary communication on device
- **Cooperative Groups** - Grid-wide synchronization via `grid.sync()`
- **Binaural Audio** - Real-time 3D spatial audio from pressure field

The persistent actor model's value is in complex scenarios with dynamic topology, irregular communication patterns, or long-running interactive simulations—not raw compute-bound FDTD where traditional stencil kernels excel.

**Interactive Benchmark Results (Persistent vs Traditional):**

| Operation | Traditional | Persistent | Winner |
|-----------|-------------|------------|--------|
| **Inject (send command)** | 317 µs | 0.03 µs | **Persistent 11,327x** |
| Query (read state) | 0.01 µs | 0.01 µs | Tie |
| Single step (compute) | 3.2 µs | 163 µs | Traditional 51x |
| **Mixed workload** | 40.5 ms | 15.3 ms | **Persistent 2.7x** |

**Key Insight:** Persistent actors excel at **interactive command latency**—commands are written to mapped memory without kernel launch overhead. Traditional kernels excel at **batch compute** (running thousands of steps at once). For real-time applications at 60 FPS (16.67ms/frame), persistent actors allow **2.7x more interactive operations per frame**.

```bash
# Run WaveSim3D benchmark
cargo run -p ringkernel-wavesim3d --bin wavesim3d-benchmark --release --features cuda-codegen

# Run interactive benchmark (persistent vs traditional)
cargo run -p ringkernel-wavesim3d --bin interactive-benchmark --release --features cuda-codegen

# Run WaveSim3D GUI
cargo run -p ringkernel-wavesim3d --bin wavesim3d --release --features cuda
```

### TxMon Showcase Application

The `ringkernel-txmon` crate demonstrates GPU-accelerated transaction monitoring for real-time fraud detection:

**Features:**
- Real-time transaction simulation with configurable suspicious rate
- Four detection rules: velocity breach, amount threshold, structured transactions, geographic anomaly
- Three GPU backend approaches: batch kernel, ring kernel (actor model), stencil kernel (pattern detection)
- Interactive GUI with live metrics and high-risk account tracking

**GPU Backend Approaches:**

| Approach | Description | Use Case |
|----------|-------------|----------|
| **Batch Kernel** | One thread per transaction, max throughput | High-volume batch processing |
| **Ring Kernel** | Persistent actor with HLC + K2K messaging | Real-time streaming pipelines |
| **Stencil Kernel** | 2D grid pattern detection (velocity × time) | Network anomaly detection |

```bash
# Run TxMon GUI
cargo run -p ringkernel-txmon --release --features cuda-codegen

# Run GPU benchmark
cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen
```

Performance varies significantly by hardware and workload.

## GPU Profiling

RingKernel v0.3.2 includes comprehensive GPU profiling infrastructure. Enable with the `profiling` feature:

```toml
[dependencies]
ringkernel-cuda = { version = "0.3", features = ["profiling"] }
```

### GPU Timer and Events

```rust
use ringkernel_cuda::profiling::{GpuTimer, CudaNvtxProfiler, ProfilingSession};

// GPU-side timing with CUDA events
let mut timer = GpuTimer::new()?;
timer.start(stream)?;
// ... kernel execution ...
timer.stop(stream)?;
println!("Kernel time: {:.3} ms", timer.elapsed_ms()?);

// NVTX profiling for Nsight Systems/Compute
let profiler = CudaNvtxProfiler::new();
{
    let _range = profiler.push_range("compute_phase", ProfilerColor::CYAN);
    // ... kernel execution ...
} // Range automatically ends

// Export to Chrome trace format
let session = ProfilingSession::new();
// ... record kernel/transfer events ...
let builder = GpuChromeTraceBuilder::from_session(&session);
std::fs::write("gpu_trace.json", builder.build())?;
```

Features include:
- **CUDA Events**: GPU-side timing without CPU overhead
- **NVTX Integration**: Timeline visualization in Nsight Systems/Compute
- **Kernel Metrics**: Grid/block dims, occupancy, registers per thread
- **Memory Tracking**: Allocation profiling with leak detection
- **Chrome Trace Export**: GPU timeline visualization in chrome://tracing

## Enterprise Security

RingKernel v0.3.2 includes comprehensive enterprise security features. Enable with the `enterprise` feature:

```toml
[dependencies]
ringkernel-core = { version = "0.3", features = ["enterprise"] }
```

### Authentication & Authorization

```rust
use ringkernel_core::prelude::*;

// API Key authentication
let auth = ApiKeyAuth::new()
    .add_key("sk-prod-abc123", Identity::new("service-a"))
    .add_key("sk-prod-xyz789", Identity::new("service-b"));

let identity = auth.authenticate(&Credentials::ApiKey("sk-prod-abc123".into())).await?;

// RBAC policy evaluation
let policy = RbacPolicy::new()
    .grant(Subject::User("alice".into()), Role::Admin)
    .grant(Subject::User("bob".into()), Role::Developer);

let evaluator = PolicyEvaluator::new(policy);
assert!(evaluator.check(&Subject::User("alice".into()), Permission::Admin));
```

### Rate Limiting

```rust
use ringkernel_core::prelude::*;

// Token bucket rate limiter
let limiter = RateLimiterBuilder::new()
    .algorithm(RateLimitAlgorithm::TokenBucket)
    .rate(1000)  // 1000 requests per second
    .burst(100)  // Allow burst of 100
    .build();

// Check rate limit
match limiter.acquire() {
    Ok(guard) => { /* proceed with operation */ },
    Err(RateLimitError::Exceeded { retry_after }) => {
        println!("Rate limited, retry after {:?}", retry_after);
    }
}
```

### TLS & Encrypted K2K

```rust
use ringkernel_core::prelude::*;

// Server TLS configuration
let tls_config = TlsConfigBuilder::new()
    .with_cert_file("server.crt")
    .with_key_file("server.key")
    .with_client_auth(ClientAuth::Required)  // mTLS
    .build()?;

// Encrypted kernel-to-kernel messaging
let encryptor = K2KEncryptor::new(K2KEncryptionConfig {
    algorithm: K2KEncryptionAlgorithm::Aes256Gcm,
    key: K2KKeyMaterial::generate()?,
});

let endpoint = EncryptedK2KBuilder::new(broker, kernel_id)
    .with_encryptor(encryptor)
    .build();
```

### Structured Logging with Tracing

```rust
use ringkernel_core::prelude::*;

// Initialize structured logger
let config = StructuredLogConfigBuilder::new()
    .production()  // JSON output, Info level
    .build();

let logger = StructuredLogger::new(config);

// Log with trace context
logger.info("Processing request")
    .field("request_id", request_id)
    .field("user_id", user_id)
    .with_trace(TraceContext::current())
    .emit();
```

### Multi-tenancy

```rust
use ringkernel_core::prelude::*;

// Configure tenant with resource quotas
let registry = TenantRegistry::new();
registry.register("tenant-a", ResourceQuota {
    max_memory_bytes: 1024 * 1024 * 1024,  // 1 GB
    max_kernels: 10,
    max_message_rate: 10_000,  // per second
})?;

// Execute with tenant context
let ctx = TenantContext::new("tenant-a");
ctx.track_memory(1024 * 1024)?;  // Track 1 MB usage
let utilization = ctx.quota_utilization();
```

## Documentation

**API Documentation:** [https://docs.rs/ringkernel](https://docs.rs/ringkernel)

**Online Guides:** [https://mivertowski.github.io/RustCompute/](https://mivertowski.github.io/RustCompute/)

- **[API Reference](https://docs.rs/ringkernel)** - Complete API documentation on docs.rs
- **[Guide](https://mivertowski.github.io/RustCompute/guide/01-architecture-overview.html)** - Architecture and usage guides

Detailed documentation is also available in the `docs/` directory:

- [Architecture Overview](docs/01-architecture-overview.md)
- [Core Abstractions](docs/03-core-abstractions.md)
- [Memory Management](docs/04-memory-management.md)
- [GPU Backends](docs/05-gpu-backends.md)

## GPU Code Generation

Write GPU kernels in Rust and transpile them to CUDA C or WGSL. Both transpilers support three kernel types with unified API.

### CUDA PTX Compilation

Compile CUDA source to PTX without directly depending on cudarc:

```rust
use ringkernel_cuda::compile_ptx;

let cuda_source = r#"
    extern "C" __global__ void add(float* a, float* b, float* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) c[idx] = a[idx] + b[idx];
    }
"#;

let ptx = compile_ptx(cuda_source)?;
// Load PTX into CUDA module and execute
```

### Backend Selection

```rust
// For NVIDIA GPUs (CUDA)
use ringkernel_cuda_codegen::{transpile_global_kernel, transpile_stencil_kernel, StencilConfig};

// For cross-platform (WebGPU/WGSL)
use ringkernel_wgpu_codegen::{transpile_global_kernel, transpile_stencil_kernel, StencilConfig};
```

The same Rust DSL code works with both backends—just change the import.

### Global Kernels

```rust
use ringkernel_cuda_codegen::{transpile_global_kernel};
use syn::parse_quote;

let kernel: syn::ItemFn = parse_quote! {
    fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        y[idx as usize] = a * x[idx as usize] + y[idx as usize];
    }
};

let cuda_code = transpile_global_kernel(&kernel)?;
// Generates: extern "C" __global__ void saxpy(...) { ... }
```

### Stencil Kernels

```rust
use ringkernel_cuda_codegen::{transpile_stencil_kernel, StencilConfig};

let kernel: syn::ItemFn = parse_quote! {
    fn fdtd(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
        let lap = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p) - 4.0 * p[pos.idx()];
        p_prev[pos.idx()] = 2.0 * p[pos.idx()] - p_prev[pos.idx()] + c2 * lap;
    }
};

let config = StencilConfig::new("fdtd").with_tile_size(16, 16).with_halo(1);
let cuda_code = transpile_stencil_kernel(&kernel, &config)?;
```

### Ring Kernels (Persistent Actor Model)

Generate persistent GPU kernels that process messages in a loop:

```rust
use ringkernel_cuda_codegen::{transpile_ring_kernel, RingKernelConfig};

let handler: syn::ItemFn = parse_quote! {
    fn process(ctx: &RingContext, msg: &Request) -> Response {
        let tid = ctx.global_thread_id();
        ctx.sync_threads();
        Response { value: msg.value * 2.0, id: tid as u64 }
    }
};

let config = RingKernelConfig::new("processor")
    .with_block_size(128)
    .with_queue_capacity(1024)
    .with_envelope_format(true)  // 256-byte MessageHeader + payload
    .with_hlc(true)              // Hybrid Logical Clocks
    .with_k2k(true)              // Kernel-to-Kernel messaging
    .with_kernel_id(1000)        // Kernel identity for routing
    .with_hlc_node_id(42);       // HLC node ID

let cuda_code = transpile_ring_kernel(&handler, &config)?;
```

**DSL Features:**
- Thread/block indices: `thread_idx_x()`, `block_idx_x()`, `block_dim_x()`, `grid_dim_x()`, `warp_size()`
- Control flow: `if/else`, `match` → switch/case, early `return`
- Loops: `for i in 0..n`, `while cond`, `loop` with `break`/`continue`
- Stencil patterns (2D): `pos.north()`, `pos.south()`, `pos.east()`, `pos.west()`, `pos.at(dx, dy)`
- Stencil patterns (3D): `pos.up()`, `pos.down()`, `pos.at(dx, dy, dz)` for volumetric kernels
- Shared memory: `__shared__` arrays and tiles
- Struct literals: `Point { x: 1.0, y: 2.0 }` → C compound literals
- Reference expressions: `&arr[idx]` → pointer with automatic `->` for field access
- Type inference: Tracks pointer variables for correct accessor generation
- **120+ GPU intrinsics** across 13 categories (atomics, warp ops, sync, math, trig, bit manipulation, memory, timing)

**Ring Kernel Features:**
- Persistent message loop with ControlBlock lifecycle
- **Envelope-based message serialization** with 256-byte MessageHeader
- Message header validation (magic number, correlation ID)
- RingContext method inlining (`ctx.thread_id()` → `threadIdx.x`)
- HLC operations with timestamp propagation: `hlc_tick()`, `hlc_now()`, `hlc_update()`
- K2K messaging with envelope format: `k2k_send_envelope()`, `k2k_try_recv_envelope()`
- Queue intrinsics: `enqueue_response()`, `input_queue_empty()`

### WGSL-Specific Notes

The WGSL transpiler handles WebGPU limitations with automatic workarounds:

| Feature | CUDA | WGSL Workaround |
|---------|------|-----------------|
| 64-bit integers | Native `long long` | Emulated as `vec2<u32>` lo/hi pairs |
| 64-bit atomics | `atomicAdd(long long*)` | Helper functions: `atomic_inc_u64()` |
| f64 | Native `double` | Downcast to `f32` with warning |
| Persistent kernels | Persistent GPU loop | Host-driven dispatch loop |
| K2K messaging | Supported | **Not supported** (returns error) |

```rust
use ringkernel_wgpu_codegen::{transpile_ring_kernel, RingKernelConfig};

let config = RingKernelConfig::new("processor")
    .with_workgroup_size(256)
    .with_hlc(true);
    // Note: .with_k2k(true) will return an error in WGPU

let wgsl_code = transpile_ring_kernel(&handler, &config)?;
```

## Known Limitations

- Metal backend is scaffolded but not production-ready (no true persistent kernels yet)
- WebGPU lacks 64-bit atomics (WGSL limitation)
- Persistent kernel mode requires CUDA compute capability 7.0+
- Cooperative groups (`grid.sync()`) requires CUDA compute capability 6.0+ and `cooperative` feature flag
- Cooperative kernel launch limited to 48-512 concurrent blocks (GPU-dependent); larger grids fall back to software synchronization

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome. Priority areas:

- Metal backend completion (persistent kernels, compute pipelines)
- Additional examples and documentation
- Performance optimization
- Testing on diverse hardware

Please open an issue to discuss significant changes before submitting PRs.
