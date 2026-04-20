---
layout: default
title: Ecosystem Integration
nav_order: 12
---

# Ecosystem Integration

## Actor Frameworks

### Actix Integration

```rust
use actix::prelude::*;

/// Bridge between Actix actors and Ring Kernels
pub struct GpuActorBridge {
    runtime: Arc<RingKernelRuntime>,
    kernel_id: String,
}

impl Actor for GpuActorBridge {
    type Context = Context<Self>;
}

/// Forward messages to GPU kernel
impl<T: RingMessage + Message<Result = R>, R: RingMessage> Handler<T> for GpuActorBridge {
    type Result = ResponseFuture<R>;

    fn handle(&mut self, msg: T, _ctx: &mut Self::Context) -> Self::Result {
        let runtime = self.runtime.clone();
        let kernel_id = self.kernel_id.clone();

        Box::pin(async move {
            runtime.send(&kernel_id, msg).await?;
            runtime.receive(&kernel_id, Duration::from_secs(30)).await
        })
    }
}

// Usage
let gpu_actor = GpuActorBridge {
    runtime: runtime.clone(),
    kernel_id: "vector_add".to_string(),
}.start();

let result: VectorAddResponse = gpu_actor.send(VectorAddRequest {
    a: vec![1.0, 2.0],
    b: vec![3.0, 4.0],
    ..Default::default()
}).await??;
```

### Tokio Actors (Tower)

```rust
use tower::{Service, ServiceBuilder};

/// Ring Kernel as a Tower service
pub struct RingKernelService<Req, Resp> {
    runtime: Arc<RingKernelRuntime>,
    kernel_id: String,
    _marker: PhantomData<(Req, Resp)>,
}

impl<Req, Resp> Service<Req> for RingKernelService<Req, Resp>
where
    Req: RingMessage,
    Resp: RingMessage,
{
    type Response = Resp;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Resp>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: Req) -> Self::Future {
        let runtime = self.runtime.clone();
        let kernel_id = self.kernel_id.clone();

        Box::pin(async move {
            runtime.send(&kernel_id, req).await?;
            runtime.receive(&kernel_id, Duration::from_secs(30)).await
        })
    }
}

// Usage with Tower middleware
let service = ServiceBuilder::new()
    .timeout(Duration::from_secs(5))
    .rate_limit(1000, Duration::from_secs(1))
    .service(RingKernelService::<VectorAddRequest, VectorAddResponse>::new(
        runtime.clone(),
        "vector_add".to_string(),
    ));
```

---

## Web Frameworks

### Axum Integration

```rust
use axum::{extract::State, routing::post, Json, Router};

struct AppState {
    runtime: Arc<RingKernelRuntime>,
}

async fn vector_add(
    State(state): State<Arc<AppState>>,
    Json(request): Json<VectorAddApiRequest>,
) -> Result<Json<VectorAddApiResponse>, AppError> {
    let kernel_request = VectorAddRequest {
        id: Uuid::new_v4(),
        priority: priority::NORMAL,
        correlation_id: None,
        a: request.a,
        b: request.b,
    };

    state.runtime.send("vector_add", kernel_request).await?;

    let response: VectorAddResponse = state.runtime
        .receive("vector_add", Duration::from_secs(30))
        .await?;

    Ok(Json(VectorAddApiResponse {
        result: response.result,
        latency_ns: response.timestamp.physical(),
    }))
}

#[tokio::main]
async fn main() {
    let runtime = RingKernelRuntime::builder()
        .backend(BackendType::Cuda)
        .build()
        .await
        .unwrap();

    // Launch and activate kernel
    let kernel = runtime.launch("vector_add", Dim3::linear(1), Dim3::linear(256), Default::default())
        .await.unwrap();
    kernel.activate().await.unwrap();

    let state = Arc::new(AppState { runtime });

    let app = Router::new()
        .route("/api/vector-add", post(vector_add))
        .with_state(state);

    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

---

## Data Processing Pipelines

### Apache Arrow Integration

```rust
use arrow::array::{Float32Array, ArrayRef};
use arrow::record_batch::RecordBatch;

/// Process Arrow arrays on GPU
pub async fn process_arrow_batch(
    runtime: &RingKernelRuntime,
    batch: RecordBatch,
) -> Result<RecordBatch> {
    let a = batch.column(0).as_any().downcast_ref::<Float32Array>().unwrap();
    let b = batch.column(1).as_any().downcast_ref::<Float32Array>().unwrap();

    let request = VectorAddRequest {
        id: Uuid::new_v4(),
        priority: priority::NORMAL,
        correlation_id: None,
        a: a.values().to_vec(),
        b: b.values().to_vec(),
    };

    runtime.send("vector_add", request).await?;
    let response: VectorAddResponse = runtime.receive("vector_add", Duration::from_secs(30)).await?;

    let result_array: ArrayRef = Arc::new(Float32Array::from(response.result));

    RecordBatch::try_new(
        batch.schema(),
        vec![result_array],
    ).map_err(Into::into)
}
```

### Polars Integration

```rust
use polars::prelude::*;

/// GPU-accelerated Polars operations
pub trait GpuOps {
    fn gpu_add(self, other: &Series, runtime: &RingKernelRuntime) -> Result<Series>;
}

impl GpuOps for Series {
    fn gpu_add(self, other: &Series, runtime: &RingKernelRuntime) -> Result<Series> {
        let a = self.f32()?.cont_slice()?;
        let b = other.f32()?.cont_slice()?;

        let request = VectorAddRequest {
            id: Uuid::new_v4(),
            priority: priority::NORMAL,
            correlation_id: None,
            a: a.to_vec(),
            b: b.to_vec(),
        };

        // Blocking for simplicity (use async in production)
        let rt = tokio::runtime::Handle::current();
        let response: VectorAddResponse = rt.block_on(async {
            runtime.send("vector_add", request).await?;
            runtime.receive("vector_add", Duration::from_secs(30)).await
        })?;

        Ok(Series::new("result", response.result))
    }
}
```

---

## Distributed Computing

### gRPC with Tonic

```rust
use tonic::{Request, Response, Status};

pub mod ringkernel_proto {
    tonic::include_proto!("ringkernel");
}

use ringkernel_proto::ring_kernel_server::{RingKernel, RingKernelServer};

pub struct RingKernelGrpc {
    runtime: Arc<RingKernelRuntime>,
}

#[tonic::async_trait]
impl RingKernel for RingKernelGrpc {
    async fn process(
        &self,
        request: Request<ProcessRequest>,
    ) -> Result<Response<ProcessResponse>, Status> {
        let req = request.into_inner();

        let kernel_req = VectorAddRequest {
            id: Uuid::new_v4(),
            priority: priority::NORMAL,
            correlation_id: None,
            a: req.a,
            b: req.b,
        };

        self.runtime.send(&req.kernel_id, kernel_req).await
            .map_err(|e| Status::internal(e.to_string()))?;

        let response: VectorAddResponse = self.runtime
            .receive(&req.kernel_id, Duration::from_secs(30))
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(ProcessResponse {
            result: response.result,
        }))
    }
}
```

---

## VynGraph NSAI Integration Points (v1.1)

v1.1 adds four integration points for neuro-symbolic AI use cases where each K2K envelope must carry provenance, tenant audit metadata, or participate in a hot-reloadable rule system. All four are part of `ringkernel-core` and usable outside VynGraph. The full specification with design rationale, ABI, and test matrix is in [`docs/superpowers/specs/2026-04-17-v1.1-vyngraph-gaps.md`](superpowers/specs/2026-04-17-v1.1-vyngraph-gaps.md); the sections below summarize what ships.

### PROV-O Provenance Headers

Every K2K envelope can optionally carry a `ProvenanceHeader` implementing the W3C PROV-O vocabulary with all eight relation kinds (`wasGeneratedBy`, `used`, `wasInformedBy`, `wasDerivedFrom`, `wasAttributedTo`, `wasAssociatedWith`, `actedOnBehalfOf`, `hadMember`). Headers are opt-in per send, support bounded-depth chain walk with cycle detection, and expose an ECDSA / P-256 signature verification hook. A builder API keeps the common case (single generation + single attribution) to two fluent calls.

```rust
let header = ProvenanceHeader::builder()
    .generated_by(activity_id)
    .attributed_to(agent_id)
    .build();
runtime.send_with_provenance(dest, msg, header).await?;
```

22 subtests cover chain walk depth bound, cycle detection, signature validation, and round-trip attach/detach on the envelope.

### Multi-Tenant K2K Isolation

`ringkernel-core/src/k2k/tenant.rs` adds per-tenant sub-brokers, `AuditTag { org_id, engagement_id }` metadata on every message, and cross-tenant send rejection. Per-tenant quotas (`max_concurrent_kernels`, `gpu_memory`, `messages_per_sec`, per-engagement cost budgets) are enforced at enqueue time; rejections write to the configurable audit sink.

```rust
broker.register_tenant_kernel(tenant_id, kernel_id, AuditTag { org_id, engagement_id })?;
broker.send(dest, msg).await?;                        // same-tenant: fast path
broker.send_with_audit(dest, msg, override_tag).await?; // override audit on send
```

Single-tenant deployments use `LegacyTenant::Unspecified` and retain the v1.0 fast path unchanged. v1.1 measured **0 cross-tenant leaks** across 13 subtests (including rejection audit records, engagement-cost accumulation separation, and registration moves between tenants); the TLA+ spec `tenant_isolation.tla` independently proves the invariant.

### Hot Rule Reload

Inference-rule engines (Rete, forward chaining, and custom) can be live-reloaded via the `CompiledRule` artifact API. Version numbers are strictly monotonic — equal or lower versions are rejected — and activation quiesces in-flight evaluators before swapping the rule set. Rollback is supported by re-submitting an earlier artifact with a new version header.

```rust
registry.register_rule(CompiledRule { version: 2, bytes, signature, .. })?;  // activates
registry.register_rule(CompiledRule { version: 1, .. })?;                    // rejected
registry.register_rule(CompiledRule { version: 3, .. /* = v1 bytes */ })?;  // rollback OK
```

32 subtests cover version-bump rejection policy, signature verification, rollback semantics, quiescence of in-flight evaluators, and dependency compilation.

### Live Introspection Streaming

`IntrospectionStream` emits per-kernel telemetry through a drop-tolerant ring buffer with EWMA decay on the consumer side. Designed for high-frequency observability without back-pressuring the host path; consumers that cannot keep up see older samples overwritten rather than blocking the producer.

---

## Machine Learning

### Candle Integration

```rust
use candle_core::{Device, Tensor};

/// Bridge between Candle tensors and Ring Kernels
pub async fn gpu_tensor_op(
    runtime: &RingKernelRuntime,
    a: &Tensor,
    b: &Tensor,
) -> Result<Tensor> {
    let a_vec = a.to_vec1::<f32>()?;
    let b_vec = b.to_vec1::<f32>()?;

    let request = VectorAddRequest {
        id: Uuid::new_v4(),
        priority: priority::NORMAL,
        correlation_id: None,
        a: a_vec,
        b: b_vec,
    };

    runtime.send("vector_add", request).await?;
    let response: VectorAddResponse = runtime.receive("vector_add", Duration::from_secs(30)).await?;

    Tensor::from_vec(response.result, a.shape(), &Device::Cpu)
        .map_err(Into::into)
}
```

---

## Configuration Management

### Config Crate Integration

```rust
use config::{Config, ConfigError, File};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct RingKernelConfig {
    pub backend: String,
    pub device_id: usize,
    pub queue_capacity: usize,
    pub poll_interval_us: u64,
    pub telemetry_enabled: bool,
}

impl RingKernelConfig {
    pub fn load() -> Result<Self, ConfigError> {
        Config::builder()
            .add_source(File::with_name("config/ringkernel"))
            .add_source(config::Environment::with_prefix("RINGKERNEL"))
            .build()?
            .try_deserialize()
    }

    pub fn to_runtime_config(&self) -> RuntimeConfig {
        RuntimeConfig {
            default_queue_capacity: self.queue_capacity,
            bridge_poll_interval: Duration::from_micros(self.poll_interval_us),
            enable_telemetry: self.telemetry_enabled,
            ..Default::default()
        }
    }
}
```

---

## Observability

### Tracing Integration

```rust
use tracing::{info, instrument, span, Level};

impl RingKernelRuntime {
    #[instrument(skip(self, message))]
    pub async fn send<T: RingMessage>(&self, kernel_id: &str, message: T) -> Result<()> {
        let span = span!(Level::DEBUG, "ringkernel.send", kernel_id, message_id = %message.message_id());
        let _guard = span.enter();

        info!(priority = message.priority(), "Sending message to kernel");

        // ... implementation
    }
}
```

### Prometheus Metrics

```rust
use prometheus::{Counter, Histogram, Registry};

lazy_static! {
    static ref MESSAGES_SENT: Counter = Counter::new(
        "ringkernel_messages_sent_total",
        "Total messages sent to kernels"
    ).unwrap();

    static ref MESSAGE_LATENCY: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "ringkernel_message_latency_seconds",
            "Message round-trip latency"
        ).buckets(vec![0.0001, 0.001, 0.01, 0.1, 1.0])
    ).unwrap();
}
```

---

## ML Framework Bridges (v0.2.0+)

RingKernel provides bridges to popular ML frameworks for tensor interop and model execution:

### PyTorch Bridge

```rust
use ringkernel_ecosystem::ml_bridge::{PyTorchBridge, PyTorchDType, PyTorchConfig};

// Create bridge with configuration
let config = PyTorchConfig {
    default_device: "cuda:0".to_string(),
    enable_cuda: true,
    memory_pool_size: 1024 * 1024 * 1024, // 1GB
    use_pinned_memory: true,
};
let bridge = PyTorchBridge::with_config(config);

// Convert data to PyTorch tensor format
let tensor = bridge.to_pytorch(
    &data_bytes,
    &[batch_size, channels, height, width],
    PyTorchDType::Float32,
)?;

// Move to GPU
let gpu_tensor = tensor.to_device("cuda:0");

// Convert back from PyTorch format
let output_data = bridge.from_pytorch(&gpu_tensor)?;
```

### ONNX Executor

```rust
use ringkernel_ecosystem::ml_bridge::{OnnxExecutor, OnnxConfig};

// Create executor
let executor = OnnxExecutor::new(&runtime)?;

// Load model
executor.load_model("models/resnet50.onnx")?;

// Prepare inputs
let inputs = vec![
    ("input", input_tensor),
];

// Execute inference
let outputs = executor.run(&inputs).await?;

// Get output tensor
let predictions = outputs.get("output").unwrap();
```

### Hugging Face Pipeline

```rust
use ringkernel_ecosystem::ml_bridge::{HuggingFacePipeline, PipelineConfig};

// Text classification
let classifier = HuggingFacePipeline::text_classification(
    &runtime,
    "bert-base-uncased",
)?;
let result = classifier.run("This is a great product!").await?;
println!("Sentiment: {:?}, Score: {:.2}", result.label, result.score);

// Text generation
let generator = HuggingFacePipeline::text_generation(
    &runtime,
    "gpt2",
)?;
let text = generator.generate("Once upon a time", 100).await?;

// Embeddings
let embedder = HuggingFacePipeline::embeddings(
    &runtime,
    "sentence-transformers/all-MiniLM-L6-v2",
)?;
let embedding = embedder.embed("Hello, world!").await?;
```

### Data Type Mapping

| PyTorch | RingKernel | Size |
|---------|------------|------|
| `torch.float32` | `PyTorchDType::Float32` | 4 bytes |
| `torch.float64` | `PyTorchDType::Float64` | 8 bytes |
| `torch.float16` | `PyTorchDType::Float16` | 2 bytes |
| `torch.bfloat16` | `PyTorchDType::BFloat16` | 2 bytes |
| `torch.int32` | `PyTorchDType::Int32` | 4 bytes |
| `torch.int64` | `PyTorchDType::Int64` | 8 bytes |
| `torch.int8` | `PyTorchDType::Int8` | 1 byte |
| `torch.uint8` | `PyTorchDType::UInt8` | 1 byte |
| `torch.bool` | `PyTorchDType::Bool` | 1 byte |

---

## Next: [Migration Guide](./12-migration.md)
