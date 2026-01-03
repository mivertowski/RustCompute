# RingKernel Roadmap

> GPU-Native Persistent Actor Model Framework for Rust

## Vision

Transform GPU computing from batch-oriented kernel launches to a true actor-based paradigm where GPU kernels are long-lived, stateful actors that communicate via high-performance message passing. Enable enterprise-grade GPU applications with sub-microsecond command latency, fault tolerance, and seamless integration with modern Rust web ecosystems.

---

## Implementation Status Summary

> Last updated: January 2026

| Phase | Implemented | Partial | Missing | Completion |
|-------|-------------|---------|---------|------------|
| **Phase 1: Foundation** | 5 | 4 | 3 | ~58% |
| **Phase 2: Code Generation** | 6 | 1 | 2 | ~72% |
| **Phase 3: Enterprise** | 8 | 1 | 7 | ~53% |
| **Phase 4: Ecosystem** | 2 | 3 | 6 | ~32% |
| **Phase 5: Developer Experience** | 5 | 2 | 4 | ~55% |
| **Overall** | **26** | **11** | **22** | **~54%** |

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
| **K2K Halo Exchange** | P1 | Medium | ❌ | Not implemented |
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
| **Batched Command Processing** | P0 | ❌ | Not implemented |
| **Subgroup Operations** | P1 | ❌ | Not implemented |
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
| **SIMD Acceleration** | P1 | ❌ | `portable_simd` not integrated |
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
| **Optimization Passes** | P2 | ⚠️ Basic | Validation exists, DCE/folding planned |

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
| **Multi-backend attribute** | P1 | ❌ | `backends = [cuda, metal]` not implemented |
| **Fallback selection** | P1 | ❌ | `fallback = wgpu` not implemented |
| **Capability checking** | P2 | ❌ | `requires = [f64]` not implemented |

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
| **Hot Reload** | P0 | ⚠️ Basic | Basic support in `multi_gpu.rs` |
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
| **Distributed Actors** | P2 | ❌ | Cross-node not implemented |
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
| **GPU Profiler Integration** | P0 | ❌ | Nsight/RenderDoc not integrated |
| **Message Tracing** | P0 | ✅ Done | `ObservabilityContext` with spans |
| **GPU Memory Dashboard** | P1 | ❌ | Not implemented |
| **Kernel Debugger** | P2 | ❌ | Not implemented |

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
| **Memory Encryption** | P1 | ❌ | Not implemented |
| **Audit Logging** | P1 | ❌ | Not implemented |
| **Kernel Sandboxing** | P2 | ❌ | Not implemented |
| **Compliance Reports** | P2 | ❌ | Not implemented |

---

## Phase 4: Ecosystem Expansion

### 4.1 Web Framework Deep Integration

| Integration | Priority | Status | Description |
|-------------|----------|--------|-------------|
| **SSE Handler** | P0 | ✅ Done | Full `sse_handler` with keep-alive |
| **WebSocket Handler** | P0 | ✅ Done | Bidirectional `ws_handler` in axum.rs |
| **GraphQL Subscriptions** | P1 | ❌ | async-graphql not integrated |
| **tRPC Support** | P2 | ❌ | Not implemented |

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
| **Arrow GPU Kernels** | P1 | ⚠️ Basic | Feature exists, GPU ops limited |
| **Polars GPU Backend** | P1 | ⚠️ Basic | Feature exists, GPU ops limited |
| **Candle Integration** | P1 | ⚠️ Basic | Feature exists, tensor bridge |
| **DataFusion GPU** | P2 | ❌ | Not implemented |

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
| **PyTorch Interop** | P1 | ❌ | Not implemented |
| **ONNX Runtime** | P1 | ❌ | Not implemented |
| **Hugging Face** | P2 | ❌ | Not implemented |

---

## Phase 5: Developer Experience

### 5.1 Tooling

| Tool | Priority | Status | Description |
|------|----------|--------|-------------|
| **ringkernel-cli** | P0 | ✅ Done | `new`, `init`, `codegen`, `check` commands |
| **VSCode Extension** | P1 | ❌ | Not implemented |
| **GPU Playground** | P1 | ❌ | Not implemented |
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
| **Interactive Tutorials** | P0 | ❌ | Not implemented |
| **Architecture Guide** | P0 | ✅ Done | Comprehensive CLAUDE.md |
| **API Reference** | P0 | ⚠️ ~60% | rustdoc exists, incomplete |
| **Example Gallery** | P1 | ✅ Done | Many examples across crates |

### 5.3 Testing Infrastructure

| Feature | Priority | Status | Description |
|---------|----------|--------|-------------|
| **GPU Mock Testing** | P0 | ⚠️ Partial | CPU backend as mock |
| **Property Testing** | P1 | ✅ Done | proptest used |
| **Fuzzing** | P1 | ❌ | Not implemented |
| **CI GPU Testing** | P1 | ❌ | Manual tests with `--ignored` |

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
- [ ] GraphQL subscriptions
- [ ] Distributed kernel messaging

---

## Success Metrics

| Metric | Current (Jan 2026) | Target |
|--------|---------|--------|
| **Backend Coverage** | 1.5 of 3 (CUDA full, Metal partial) | 3 of 3 |
| **Command Latency** | 0.03µs (CUDA) | <0.1µs (all backends) |
| **Code Generation Tests** | 233+ | 500+ |
| **Ecosystem Integrations** | 8 (SSE, WS, Actix, Tower, Axum, gRPC, Arrow, Polars) | 15+ |
| **Documentation Coverage** | ~60% | 95%+ |
| **Test Count** | 580+ | 800+ |
| **Roadmap Completion** | ~52% | 100% |

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
