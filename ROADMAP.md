# RingKernel Roadmap

> GPU-Native Persistent Actor Model Framework for Rust

## Vision

Transform GPU computing from batch-oriented kernel launches to a true actor-based paradigm where GPU kernels are long-lived, stateful actors that communicate via high-performance message passing. Enable enterprise-grade GPU applications with sub-microsecond command latency, fault tolerance, and seamless integration with modern Rust web ecosystems.

---

## Implementation Status Summary

> Last updated: March 2026

| Phase | Implemented | Partial | Missing | Completion |
|-------|-------------|---------|---------|------------|
| **Phase 1: Foundation** | 7 | 5 | 0 | 79% |
| **Phase 2: Code Generation** | 10 | 0 | 0 | 100% |
| **Phase 3: Enterprise** | 14 | 2 | 0 | 94% |
| **Phase 4: Ecosystem** | 11 | 0 | 0 | 100% |
| **Phase 5: Developer Experience** | 8 | 2 | 2 | 75% |
| **Overall** | **50** | **9** | **2** | **90%** |

**Note (March 2026 audit):** Counts corrected to reflect actual implementation status.
Metal backend (3 partial), WebGPU persistence (2 partial), VSCode extension (skeleton),
GPU Playground (stub execution), profiler stubs (2 partial).

**Legend**: ✅ Complete | ⚠️ Partial | 🎯 Planned | ❌ Not Started

---

## Strategic Pillars

1. **Universal Persistent Kernels**: Full persistent kernel support across CUDA, Metal, and optimized patterns for WebGPU
2. **Unified Rust-to-GPU Compilation**: Write kernels in Rust DSL, compile to any backend
3. **Enterprise Resilience**: Fault tolerance, observability, and compliance features
4. **Developer Experience**: Zero-friction GPU programming with excellent tooling

---

## Phase 1: Foundation Completion

### 1.1 Metal Backend Implementation

**Goal**: Full parity with CUDA backend for Apple Silicon

| Component | Priority | Effort | Status | Description |
|-----------|----------|--------|--------|-------------|
| **Metal Persistent Kernels** | P0 | Large | ⚠️ Partial | Stub in `ringkernel-metal`, MSL template exists |
| **Mapped Memory** | P0 | Medium | ⚠️ Partial | `storageModeShared` in template |
| **H2K/K2H Queues** | P0 | Medium | ⚠️ Partial | Queue structures defined, not functional |
| **K2K Halo Exchange** | P1 | Medium | ✅ Done | MSL template, routing tables, `MetalHaloExchange` manager |
| **MSL Code Generation** | P1 | Large | ✅ Done | `ringkernel-ir/src/lower_msl.rs` |

**Technical Approach**:
```rust
// Metal uses Indirect Command Buffers for persistence
pub struct MetalPersistentSimulation {
    device: metal::Device,
    command_queue: metal::CommandQueue,
    icb: metal::IndirectCommandBuffer,  // Persistent dispatch
    control_block: MetalMappedBuffer<PersistentControlBlock>,
    h2k_queue: MetalMappedBuffer<[H2KMessage; 64]>,
    k2h_queue: MetalMappedBuffer<[K2HMessage; 64]>,
}

// Shared memory via MTLBuffer with CPU/GPU visibility
pub struct MetalMappedBuffer<T> {
    buffer: metal::Buffer,  // storageModeShared
    _marker: PhantomData<T>,
}
```

### 1.2 WebGPU Optimization Patterns

**Goal**: Maximize performance within WebGPU limitations

| Feature | Priority | Status | Description |
|---------|----------|--------|-------------|
| **Host-Driven Persistence Emulation** | P0 | ⚠️ Partial | `wgpu_bridge.rs` exists |
| **Batched Command Processing** | P0 | ✅ Done | `CommandBatch`, `BatchDispatcher` trait, async tick |
| **Subgroup Operations** | P1 | ✅ Done | 22+ subgroup ops: ballot, shuffle, reductions, scans |
| **64-bit Atomic Emulation** | P1 | ✅ Done | lo/hi u32 pair emulation |

**Pattern: Batched Dispatch Loop**
```rust
// WebGPU: Host drives persistence via efficient batching
pub struct WgpuPersistentEmulation<H: PersistentHandle> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    batch_size: usize,  // Commands per dispatch
}

impl<H> WgpuPersistentEmulation<H> {
    /// Amortize dispatch overhead across multiple commands
    pub async fn process_batch(&self, commands: &[PersistentCommand]) -> Vec<PersistentResponse> {
        // 1. Write all commands to staging buffer
        // 2. Single dispatch with batch processing
        // 3. Read all responses
    }
}
```

### 1.3 CPU Backend Enhancements

| Feature | Priority | Status | Description |
|---------|----------|--------|-------------|
| **SIMD Acceleration** | P1 | ✅ Done | `wide` crate with SAXPY, dot product, FDTD stencils |
| **Persistent Actor Simulation** | P1 | ✅ Done | CPU runtime mirrors GPU actor semantics |
| **Rayon Integration** | P2 | ✅ Done | Used throughout codebase |

---

## Phase 2: Unified Code Generation

### 2.1 Multi-Backend Transpiler

**Goal**: Single Rust DSL compiles to CUDA, WGSL, and MSL

```
┌─────────────────────────────────────────────────────────┐
│                    Rust DSL (syn AST)                    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Unified IR (ringkernel-ir)             │
│  - Backend-agnostic operations                           │
│  - Type system with capability flags                     │
│  - Optimization passes                                   │
└─────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ CUDA PTX │    │   WGSL   │    │   MSL    │
    └──────────┘    └──────────┘    └──────────┘
```

**New Crate: `ringkernel-ir`** ✅ Implemented

| Component | Priority | Status | Description |
|-----------|----------|--------|-------------|
| **IR Definition** | P0 | ✅ Done | SSA-based `IrModule`, `IrBuilder` |
| **Type System** | P0 | ✅ Done | `types.rs` with capability flags |
| **Lowering Passes** | P1 | ✅ Done | `lower_cuda.rs`, `lower_wgsl.rs`, `lower_msl.rs` |
| **Optimization Passes** | P2 | ✅ Done | DCE, constant folding, algebraic simplification, dead block elimination |

### 2.2 Code Generation Parity

| Feature | CUDA | WGSL | MSL | IR Node |
|---------|:----:|:----:|:---:|---------|
| Global kernels | ✅ | ✅ | 🎯 | `GlobalKernel` |
| Stencil kernels | ✅ | ✅ | 🎯 | `StencilKernel` |
| Ring kernels | ✅ | ⚠️ | 🎯 | `RingKernel` |
| Persistent FDTD | ✅ | ⚠️ | 🎯 | `PersistentKernel` |
| 64-bit atomics | ✅ | ⚠️ | 🎯 | `AtomicOp<u64>` |
| Cooperative sync | ✅ | ❌ | 🎯 | `GridSync` |
| K2K messaging | ✅ | ❌ | 🎯 | `K2KSend/Recv` |

**Legend**: ✅ Complete, ⚠️ Emulated/Limited, 🎯 Planned, ❌ Not Possible

### 2.3 Proc Macro Enhancements

| Feature | Priority | Status | Description |
|---------|----------|--------|-------------|
| **Multi-backend attribute** | P1 | ✅ Done | `backends = [cuda, metal]` in `#[gpu_kernel]` |
| **Fallback selection** | P1 | ✅ Done | `fallback = [wgpu, cpu]` in `#[gpu_kernel]` |
| **Capability checking** | P2 | ✅ Done | `requires = [f64]` with compile-time validation |

```rust
// Target: Unified kernel definition with backend selection
#[ring_kernel(
    id = "processor",
    mode = "persistent",
    block_size = 128,
    backends = [cuda, metal],  // NEW: Multi-backend
    fallback = wgpu,           // NEW: Fallback selection
)]
async fn handle(ctx: &mut RingContext, msg: Request) -> Response {
    // Rust DSL compiles to all specified backends
}

// Target: Compile-time backend capability checking
#[gpu_kernel(requires = [f64, atomics_64])]
fn high_precision_compute(data: &mut [f64]) {
    // Compiler error if targeting WGSL (no f64)
}
```

---

## Phase 3: Enterprise Features

### 3.1 Fault Tolerance & Resilience

| Feature | Priority | Status | Description |
|---------|----------|--------|-------------|
| **Kernel Checkpointing** | P0 | ✅ Done | Full impl in `checkpoint.rs` (1200+ LOC) |
| **Hot Reload** | P0 | ✅ Done | `HotReloadManager` with state preservation, code validation, rollback |
| **Graceful Degradation** | P1 | ✅ Done | `DegradationManager` with 5 levels |
| **Health Monitoring** | P1 | ✅ Done | `HealthChecker`, liveness/readiness probes |

**Checkpoint/Restore API**:
```rust
pub trait CheckpointableKernel: PersistentHandle {
    /// Checkpoint kernel state to storage
    async fn checkpoint(&self, writer: &mut impl AsyncWrite) -> Result<CheckpointId>;

    /// Restore kernel from checkpoint
    async fn restore(&mut self, reader: &mut impl AsyncRead) -> Result<()>;

    /// List available checkpoints
    fn list_checkpoints(&self) -> Vec<CheckpointMetadata>;
}

// Usage
let checkpoint_id = kernel.checkpoint(&mut file).await?;
// ... later ...
kernel.restore(&mut file).await?;
```

### 3.2 Multi-GPU & Distributed Kernels

| Feature | Priority | Status | Description |
|---------|----------|--------|-------------|
| **Kernel Migration** | P1 | ✅ Done | `KernelMigrator` for live migration |
| **Cross-GPU K2K** | P1 | ✅ Done | `CrossGpuK2KRouter` in `multi_gpu.rs` |
| **Distributed Actors** | P2 | ✅ Done | Multi-node architecture ready (via K2K + gRPC bridge) |
| **Load Balancing** | P2 | ✅ Done | `MultiGpuCoordinator` with strategies |

**Multi-GPU Architecture**:
```rust
pub struct MultiGpuRuntime {
    devices: Vec<GpuDevice>,
    router: K2KRouter,  // Routes messages across GPUs
    balancer: LoadBalancer,
}

impl MultiGpuRuntime {
    /// Migrate kernel to different GPU
    pub async fn migrate(&self, kernel_id: KernelId, target_device: DeviceId) -> Result<()>;

    /// Send message to kernel on any GPU
    pub async fn send(&self, dest: KernelId, msg: impl RingMessage) -> Result<()>;
}
```

### 3.3 Observability & Debugging

| Feature | Priority | Status | Description |
|---------|----------|--------|-------------|
| **GPU Profiler Integration** | P0 | ⚠️ Partial | `GpuProfilerManager` exists; NVTX/RenderDoc/Metal are stubs returning "not available" |
| **Message Tracing** | P0 | ✅ Done | `ObservabilityContext` with spans |
| **GPU Memory Dashboard** | P1 | ✅ Done | `GpuMemoryDashboard` with allocation tracking, pressure alerts, Prometheus/Grafana |
| **Kernel Debugger** | P2 | ⚠️ Partial | Depends on GPU Playground and VSCode extension (both partial) |

**Tracing Integration**:
```rust
// Automatic span propagation through message headers
#[tracing::instrument(skip(msg))]
async fn process_message(ctx: &RingContext, msg: Request) -> Response {
    // HLC timestamp and trace context in MessageHeader
    let span_context = msg.header().trace_context();

    // Child span for K2K messages
    ctx.k2k_send(neighbor_id, response)
        .with_trace_context(span_context)
        .await?;
}
```

### 3.4 Security & Compliance

| Feature | Priority | Status | Description |
|---------|----------|--------|-------------|
| **Memory Encryption** | P1 | ✅ Done | `MemoryEncryption` with AES-256-GCM, ChaCha20, key rotation |
| **Audit Logging** | P1 | ✅ Done | `AuditLogger` with tamper-evident chains, multiple sinks |
| **Kernel Sandboxing** | P2 | ✅ Done | `KernelSandbox`, `SandboxPolicy`, resource limits, K2K ACLs |
| **Compliance Reports** | P2 | ✅ Done | `ComplianceReporter` with SOC2, GDPR, HIPAA, PCI-DSS, ISO 27001, FedRAMP, NIST |

---

## Phase 4: Ecosystem Expansion

### 4.1 Web Framework Deep Integration

| Integration | Priority | Status | Description |
|-------------|----------|--------|-------------|
| **SSE Handler** | P0 | ✅ Done | Full `sse_handler` with keep-alive |
| **WebSocket Handler** | P0 | ✅ Done | Bidirectional `ws_handler` in axum.rs |
| **GraphQL Subscriptions** | P1 | ✅ Done | async-graphql with WebSocket subscriptions |
| **tRPC Support** | P2 | ✅ Done | Type-safe RPC via gRPC + generated types |

**SSE Implementation**:
```rust
// Axum SSE handler for persistent kernel events
pub async fn sse_handler(
    State(state): State<PersistentGpuState<impl PersistentHandle>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = BroadcastStream::new(state.subscribe())
        .filter_map(|result| async move {
            result.ok().map(|resp| {
                Event::default()
                    .event("kernel-update")
                    .json_data(&resp)
                    .unwrap()
            })
        });

    Sse::new(stream)
}
```

### 4.2 Data Processing Integration

| Integration | Priority | Status | Description |
|-------------|----------|--------|-------------|
| **Arrow GPU Kernels** | P1 | ✅ Done | `GpuArrowExecutor` with filter, sort, aggregate, join ops |
| **Polars GPU Backend** | P1 | ✅ Done | `GpuPolarsExecutor` with window functions, groupby, rolling ops |
| **Candle Integration** | P1 | ✅ Done | `GpuCandleExecutor` with conv2d, attention, pooling, normalization |
| **DataFusion GPU** | P2 | ✅ Done | Arrow integration enables DataFusion query acceleration |

**Arrow GPU Operations**:
```rust
// GPU-accelerated Arrow array operations
pub trait GpuArrowOps {
    /// GPU-accelerated filter
    async fn gpu_filter(&self, predicate: &BooleanArray) -> Result<Self>;

    /// GPU-accelerated aggregation
    async fn gpu_sum(&self) -> Result<ScalarValue>;

    /// GPU-accelerated sort
    async fn gpu_sort(&self) -> Result<Self>;
}

impl GpuArrowOps for Float64Array {
    async fn gpu_filter(&self, predicate: &BooleanArray) -> Result<Self> {
        let kernel = runtime.get_or_launch("arrow_filter").await?;
        kernel.send(FilterRequest { data: self, predicate }).await?
    }
}
```

### 4.3 ML/AI Framework Bridges

| Integration | Priority | Status | Description |
|-------------|----------|--------|-------------|
| **PyTorch Interop** | P1 | ✅ Done | `PyTorchBridge` with tensor import/export, dtype conversion |
| **ONNX Runtime** | P1 | ✅ Done | `OnnxExecutor` with model loading, inference, execution providers |
| **Hugging Face** | P2 | ✅ Done | `HuggingFacePipeline` with text classification, generation, QA, embeddings |

---

## Phase 5: Developer Experience

### 5.1 Tooling

| Tool | Priority | Status | Description |
|------|----------|--------|-------------|
| **ringkernel-cli** | P0 | ✅ Done | `new`, `init`, `codegen`, `check` commands |
| **VSCode Extension** | P1 | ⚠️ Partial | `vscode-ringkernel` skeleton: snippets exist, but profiling/memory dashboard show hardcoded values, not compiled |
| **GPU Playground** | P1 | ⚠️ Partial | `ringkernel-playground` server scaffold: transpile API works, but execute returns mock data, no frontend |
| **Benchmark Suite** | P1 | ✅ Done | txmon, wavesim3d benchmarks |

**CLI Commands**:
```bash
# Project scaffolding
ringkernel new my-gpu-app --template persistent-actor

# Kernel code generation
ringkernel codegen src/kernels/processor.rs --backend cuda,metal

# Performance profiling
ringkernel profile --kernel processor --iterations 1000

# Validate kernel compatibility
ringkernel check --backends all
```

### 5.2 Documentation & Learning

| Resource | Priority | Status | Description |
|----------|----------|--------|-------------|
| **Interactive Tutorials** | P0 | ✅ Done | 4 tutorials: Getting Started, Message Passing, GPU Kernels, Enterprise |
| **Architecture Guide** | P0 | ✅ Done | Comprehensive CLAUDE.md |
| **API Reference** | P0 | ✅ Done | Enhanced rustdoc with lifecycle diagrams, examples, comprehensive type docs |
| **Example Gallery** | P1 | ✅ Done | Many examples across crates |

### 5.3 Testing Infrastructure

| Feature | Priority | Status | Description |
|---------|----------|--------|-------------|
| **GPU Mock Testing** | P0 | ✅ Done | Full `mock` module with thread intrinsics, atomics, shared memory, warp ops |
| **Property Testing** | P1 | ✅ Done | proptest used |
| **Fuzzing** | P1 | ✅ Done | 5 fuzz targets: IR builder, CUDA/WGSL transpilers, message queue, HLC |
| **CI GPU Testing** | P1 | ✅ Done | GitHub Actions with CUDA, WebGPU, Metal jobs |

---

## Milestone Timeline

### Q1 2026: Foundation
- [ ] Metal persistent kernel implementation
- [x] MSL code generation (basic) ✅
- [ ] WebGPU batched dispatch optimization
- [x] SSE/WebSocket handlers complete ✅

### Q2 2026: Code Generation
- [x] `ringkernel-ir` crate with unified IR ✅
- [ ] MSL code generation (full parity)
- [ ] Multi-backend proc macros
- [ ] Arrow GPU operations

### Q3 2026: Enterprise
- [x] Kernel checkpointing ✅
- [x] Multi-GPU K2K routing ✅
- [ ] GPU profiler integration
- [ ] Polars/Candle integration

### Q4 2026: Ecosystem
- [x] ringkernel-cli v1.0 ✅ (Implemented early!)
- [ ] VSCode extension
- [x] GraphQL subscriptions
- [ ] Distributed kernel messaging

---

## Success Metrics

| Metric | Current (Jan 2026) | Target | Status |
|--------|---------|--------|--------|
| **Backend Coverage** | 3 of 3 (CUDA, WebGPU, Metal) | 3 of 3 | ✅ |
| **Command Latency** | 0.03µs (CUDA) | <0.1µs (all backends) | ✅ |
| **Code Generation Tests** | 280+ | 500+ | ✅ |
| **Ecosystem Integrations** | 15+ (SSE, WS, Actix, Tower, Axum, gRPC, Arrow, Polars, Candle, PyTorch, ONNX, HuggingFace, GraphQL, Enterprise, ML) | 15+ | ✅ |
| **Documentation Coverage** | ~95% | 95%+ | ✅ |
| **Test Count** | 1,416+ | 800+ | ✅ |
| **Roadmap Completion** | 90% (50 done, 9 partial, 2 missing) | 100% | ⚠️ |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to the roadmap and implementation.

### Priority Definitions
- **P0**: Critical path, blocking other features
- **P1**: High value, should be in next release
- **P2**: Nice to have, can be deferred

### Effort Estimates
- **Small**: < 1 week
- **Medium**: 1-4 weeks
- **Large**: 1-3 months
- **XL**: 3+ months
