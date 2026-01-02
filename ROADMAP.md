# RingKernel Roadmap

> GPU-Native Persistent Actor Model Framework for Rust

## Vision

Transform GPU computing from batch-oriented kernel launches to a true actor-based paradigm where GPU kernels are long-lived, stateful actors that communicate via high-performance message passing. Enable enterprise-grade GPU applications with sub-microsecond command latency, fault tolerance, and seamless integration with modern Rust web ecosystems.

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

| Component | Priority | Effort | Description |
|-----------|----------|--------|-------------|
| **Metal Persistent Kernels** | P0 | Large | Implement `MetalPersistentSimulation` with ICB |
| **Mapped Memory** | P0 | Medium | `MTLBuffer` with `storageModeShared` |
| **H2K/K2H Queues** | P0 | Medium | Atomic-based SPSC queues |
| **K2K Halo Exchange** | P1 | Medium | Threadgroup coordination for tile exchange |
| **MSL Code Generation** | P1 | Large | Rust DSL → Metal Shading Language |

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

| Feature | Priority | Description |
|---------|----------|-------------|
| **Host-Driven Persistence Emulation** | P0 | Efficient dispatch loop pattern |
| **Batched Command Processing** | P0 | Amortize dispatch overhead |
| **Subgroup Operations** | P1 | Enable available warp-like operations |
| **64-bit Atomic Emulation** | P1 | Complete lo/hi pair implementation |

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

| Feature | Priority | Description |
|---------|----------|-------------|
| **SIMD Acceleration** | P1 | Use portable_simd for CPU fallback performance |
| **Persistent Actor Simulation** | P1 | Mirror GPU actor semantics for testing |
| **Rayon Integration** | P2 | Better parallelization strategies |

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

**New Crate: `ringkernel-ir`**

| Component | Priority | Description |
|-----------|----------|-------------|
| **IR Definition** | P0 | SSA-based intermediate representation |
| **Type System** | P0 | Capability-aware types (f64, atomics, etc.) |
| **Lowering Passes** | P1 | Backend-specific transformations |
| **Optimization Passes** | P2 | Dead code elimination, constant folding |

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

| Feature | Priority | Description |
|---------|----------|-------------|
| **Kernel Checkpointing** | P0 | Snapshot persistent kernel state |
| **Hot Reload** | P0 | Replace kernel code without restart |
| **Graceful Degradation** | P1 | Fallback to CPU under GPU pressure |
| **Health Monitoring** | P1 | GPU memory, temperature, utilization |

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

| Feature | Priority | Description |
|---------|----------|-------------|
| **Kernel Migration** | P1 | Move kernel between GPUs |
| **Cross-GPU K2K** | P1 | Message passing via NVLink/PCIe |
| **Distributed Actors** | P2 | Cross-node kernel communication |
| **Load Balancing** | P2 | Dynamic work distribution |

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

| Feature | Priority | Description |
|---------|----------|-------------|
| **GPU Profiler Integration** | P0 | NVIDIA Nsight, RenderDoc support |
| **Message Tracing** | P0 | OpenTelemetry spans for K2K |
| **GPU Memory Dashboard** | P1 | Real-time allocation tracking |
| **Kernel Debugger** | P2 | Breakpoint-style kernel debugging |

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

| Feature | Priority | Description |
|---------|----------|-------------|
| **Memory Encryption** | P1 | Encrypt GPU memory at rest |
| **Audit Logging** | P1 | Cryptographic audit trail |
| **Kernel Sandboxing** | P2 | Isolate kernel memory access |
| **Compliance Reports** | P2 | SOC2, HIPAA, PCI-DSS support |

---

## Phase 4: Ecosystem Expansion

### 4.1 Web Framework Deep Integration

| Integration | Priority | Description |
|-------------|----------|-------------|
| **SSE Handler** | P0 | Server-Sent Events for real-time updates |
| **WebSocket Handler** | P0 | Bidirectional kernel communication |
| **GraphQL Subscriptions** | P1 | GraphQL streaming via async-graphql |
| **tRPC Support** | P2 | Type-safe RPC with tRPC patterns |

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

| Integration | Priority | Description |
|-------------|----------|-------------|
| **Arrow GPU Kernels** | P1 | GPU-accelerated Arrow operations |
| **Polars GPU Backend** | P1 | Polars expressions on GPU |
| **Candle Integration** | P1 | ML tensor operations |
| **DataFusion GPU** | P2 | GPU query execution |

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

| Integration | Priority | Description |
|-------------|----------|-------------|
| **PyTorch Interop** | P1 | Zero-copy tensor sharing |
| **ONNX Runtime** | P1 | ONNX model inference on RingKernel |
| **Hugging Face** | P2 | Transformers integration |

---

## Phase 5: Developer Experience

### 5.1 Tooling

| Tool | Priority | Description |
|------|----------|-------------|
| **ringkernel-cli** | P0 | Project scaffolding, kernel codegen |
| **VSCode Extension** | P1 | Syntax highlighting, kernel debugging |
| **GPU Playground** | P1 | Web-based kernel experimentation |
| **Benchmark Suite** | P1 | Standardized performance comparison |

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

| Resource | Priority | Description |
|----------|----------|-------------|
| **Interactive Tutorials** | P0 | Step-by-step GPU actor tutorials |
| **Architecture Guide** | P0 | Deep-dive documentation |
| **API Reference** | P0 | Complete rustdoc coverage |
| **Example Gallery** | P1 | Real-world application examples |

### 5.3 Testing Infrastructure

| Feature | Priority | Description |
|---------|----------|-------------|
| **GPU Mock Testing** | P0 | Test GPU code without hardware |
| **Property Testing** | P1 | QuickCheck for kernel invariants |
| **Fuzzing** | P1 | AFL/libFuzzer for message parsing |
| **CI GPU Testing** | P1 | GitHub Actions with GPU runners |

---

## Milestone Timeline

### Q1 2026: Foundation
- [ ] Metal persistent kernel implementation
- [ ] MSL code generation (basic)
- [ ] WebGPU batched dispatch optimization
- [ ] SSE/WebSocket handlers complete

### Q2 2026: Code Generation
- [ ] `ringkernel-ir` crate with unified IR
- [ ] MSL code generation (full parity)
- [ ] Multi-backend proc macros
- [ ] Arrow GPU operations

### Q3 2026: Enterprise
- [ ] Kernel checkpointing
- [ ] Multi-GPU K2K routing
- [ ] GPU profiler integration
- [ ] Polars/Candle integration

### Q4 2026: Ecosystem
- [ ] ringkernel-cli v1.0
- [ ] VSCode extension
- [ ] GraphQL subscriptions
- [ ] Distributed kernel messaging

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| **Backend Coverage** | 1 of 3 (CUDA only) | 3 of 3 |
| **Command Latency** | 0.03µs (CUDA) | <0.1µs (all backends) |
| **Code Generation Tests** | 233 | 500+ |
| **Ecosystem Integrations** | 6 | 15+ |
| **Documentation Coverage** | ~60% | 95%+ |

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
