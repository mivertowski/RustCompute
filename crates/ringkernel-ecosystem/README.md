# ringkernel-ecosystem

Ecosystem integrations for RingKernel.

## Overview

This crate provides optional integrations with popular Rust ecosystem libraries for actors, web frameworks, data processing, and observability. All integrations are opt-in via feature flags.

**Key Feature**: Persistent GPU kernel integration enables **11,327x faster command injection** (~0.03µs vs ~317µs) compared to traditional kernel launch patterns.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `persistent` | Core persistent GPU kernel traits (backend-agnostic) |
| `persistent-cuda` | CUDA implementation of `PersistentHandle` |
| `actix` | Actix actor framework bridge with `GpuPersistentActor` |
| `tower` | Tower service middleware with `PersistentKernelService` |
| `axum` | Axum web framework with persistent GPU state |
| `axum-ws` | WebSocket support for Axum streaming |
| `axum-sse` | Server-Sent Events support for Axum streaming |
| `grpc` | gRPC server with streaming RPCs via Tonic |
| `arrow` | Apache Arrow data processing |
| `polars` | Polars DataFrame operations |
| `candle` | Candle ML framework bridge |
| `config` | Configuration file management |
| `tracing-integration` | Enhanced tracing support |
| `prometheus` | Prometheus metrics export |
| `persistent-full` | Full persistent ecosystem (CUDA + all web frameworks) |
| `full` | All integrations enabled |

## Installation

```toml
[dependencies]
ringkernel-ecosystem = { version = "0.2", features = ["axum", "persistent"] }
```

## Persistent GPU Integration

The persistent GPU integration leverages RingKernel's persistent actor model for ultra-low-latency command injection:

| Operation | Traditional | Persistent | Speedup |
|-----------|-------------|------------|---------|
| Inject command | 317 µs | 0.03 µs | **11,327x** |
| Mixed workload | 40.5 ms | 15.3 ms | **2.7x** |

### Core Trait

```rust
use ringkernel_ecosystem::persistent::PersistentHandle;

#[async_trait]
pub trait PersistentHandle: Send + Sync + 'static {
    fn kernel_id(&self) -> &str;
    fn is_running(&self) -> bool;
    fn send_command(&self, cmd: PersistentCommand) -> Result<CommandId>;
    fn poll_responses(&self) -> Vec<PersistentResponse>;
    fn stats(&self) -> PersistentStats;
    async fn wait_for_command(&self, cmd_id: CommandId, timeout: Duration) -> Result<()>;
    async fn shutdown(&self) -> Result<()>;
}
```

### Actix Integration

GPU-backed actor with persistent kernel:

```rust
use ringkernel_ecosystem::actix::{GpuPersistentActor, PersistentActorConfig};

let handle: Arc<dyn PersistentHandle> = create_persistent_handle();
let actor = GpuPersistentActor::new(handle, PersistentActorConfig::default()).start();

// Ultra-low-latency commands (~0.03µs)
actor.send(RunStepsCmd::new(1000)).await?;
actor.send(InjectCmd::new(10, 10, 0, 1.0)).await?;
```

### Axum Integration

REST API with SSE streaming:

```rust
use ringkernel_ecosystem::axum::{PersistentGpuState, PersistentAxumConfig};

let state = PersistentGpuState::new(handle, PersistentAxumConfig::default());

let app = Router::new()
    .merge(state.routes())  // POST /api/step, /api/impulse, GET /api/stats
    .route("/api/events", get(sse_handler::<MyHandle>))  // SSE streaming
    .with_state(state);
```

### Tower Integration

Tower service with middleware support:

```rust
use ringkernel_ecosystem::tower::{PersistentKernelService, PersistentServiceBuilder};
use tower::ServiceBuilder;

let service = ServiceBuilder::new()
    .timeout(Duration::from_secs(5))
    .rate_limit(10000, Duration::from_secs(1))
    .service(PersistentKernelService::new(handle));
```

### gRPC Integration

Unary and streaming RPCs:

```rust
use ringkernel_ecosystem::grpc::{PersistentGrpcService, PersistentGrpcConfig};

let service = PersistentGrpcService::new(handle, PersistentGrpcConfig::default());

// Unary RPC
let response = service.run_steps(PersistentStepRequest { count: 100, request_id: None }).await?;

// Server-streaming RPC
let stream = service.stream_responses();
while let Some(response) = stream.next().await {
    // Handle real-time kernel responses
}
```

### CUDA Bridge

For NVIDIA GPU support, use `CudaPersistentHandle`:

```rust
use ringkernel_cuda::persistent::{PersistentSimulation, PersistentSimulationConfig};
use ringkernel_cuda::CudaDevice;
use ringkernel_ecosystem::cuda_bridge::CudaPersistentHandle;

// Create CUDA simulation
let device = CudaDevice::new(0)?;
let config = PersistentSimulationConfig::new(64, 64, 64).with_tile_size(8, 8, 8);
let simulation = PersistentSimulation::new(&device, config)?;

// Create ecosystem handle
let handle = CudaPersistentHandle::new(simulation, "fdtd_3d");
handle.start(&ptx, "persistent_fdtd3d")?;

// Use with any ecosystem integration
let actor = GpuPersistentActor::new(Arc::new(handle), config).start();
```

## Examples

Run the Axum REST API example:

```bash
cargo run -p ringkernel-ecosystem --example axum_persistent_api --features "axum,persistent"
```

This demonstrates:
- Setting up Axum routes for persistent kernel control
- REST endpoints for step execution, impulse injection, and stats
- Graceful shutdown handling

## Traditional Integration

For applications not using persistent kernels:

### Axum

```rust
use ringkernel_ecosystem::axum::{AppState, gpu_handler};

let app = Router::new()
    .route("/process", post(gpu_handler::<MyRuntime, Request, Response>))
    .with_state(AppState::new(runtime));
```

### Prometheus Metrics

```rust
use ringkernel_ecosystem::metrics::PrometheusExporter;

let exporter = PrometheusExporter::new();
exporter.register_runtime_metrics(&runtime);
```

### Configuration

```rust
use ringkernel_ecosystem::config::ConfigManager;

let config = ConfigManager::load("config.toml")?;
let runtime = config.create_runtime().await?;
```

## Testing

```bash
# Basic tests
cargo test -p ringkernel-ecosystem

# With persistent features
cargo test -p ringkernel-ecosystem --features "persistent,actix,tower"

# All features
cargo test -p ringkernel-ecosystem --features full
```

## License

Apache-2.0
