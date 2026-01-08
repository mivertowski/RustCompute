# Implementation Plan

> Phased Implementation Guide for RingKernel Roadmap

## Overview

This document provides a detailed, actionable implementation plan for the RingKernel roadmap. Each phase is broken down into sprints with specific deliverables, dependencies, effort estimates, and acceptance criteria.

---

## Implementation Principles

### Development Philosophy
1. **Test-First**: Write tests before implementation
2. **Incremental Delivery**: Ship working features frequently
3. **API Stability**: Core traits stabilize early, implementations evolve
4. **Backward Compatibility**: Maintain compatibility within major versions

### Code Quality Standards
- Minimum 80% test coverage for new code
- All public APIs documented with examples
- Clippy lints at `pedantic` level
- Benchmarks for performance-critical paths

### Review Process
- All changes require code review
- Performance changes require benchmark comparison
- API changes require RFC document
- Security-sensitive changes require security review

---

## Phase 1: Foundation Completion (Q1 2026)

### Sprint 1.1: Metal Backend Core (Weeks 1-4)

**Goal**: Basic Metal kernel execution with mapped memory

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 1.1.1 Metal device enumeration | S | None | - |
| 1.1.2 MTLBuffer allocation with storageModeShared | M | 1.1.1 | - |
| 1.1.3 MetalMappedBuffer<T> implementation | M | 1.1.2 | - |
| 1.1.4 Basic compute pipeline creation | M | 1.1.1 | - |
| 1.1.5 Kernel launch and synchronization | M | 1.1.4 | - |
| 1.1.6 Unit tests for Metal primitives | M | 1.1.1-5 | - |

**Effort Key**: S = Small (< 3 days), M = Medium (3-7 days), L = Large (1-3 weeks)

#### Deliverables
- [ ] `MetalDevice` with capability detection
- [ ] `MetalMappedBuffer<T>` for CPU/GPU shared memory
- [ ] `MetalComputePipeline` wrapper
- [ ] 15+ unit tests passing

#### Acceptance Criteria
```rust
#[test]
fn test_metal_mapped_buffer() {
    let device = MetalDevice::new()?;
    let buffer: MetalMappedBuffer<[f32; 1024]> = device.create_mapped_buffer()?;

    // Write from CPU
    buffer.as_mut_slice()[0] = 42.0;

    // GPU can read (verified by kernel)
    // CPU can read GPU writes
    assert_eq!(buffer.as_slice()[0], 42.0);
}
```

---

### Sprint 1.2: Metal Persistent Kernels (Weeks 5-8)

**Goal**: Implement persistent kernel architecture for Metal

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 1.2.1 PersistentControlBlock for Metal | M | 1.1.3 | - |
| 1.2.2 H2K queue implementation | L | 1.2.1 | - |
| 1.2.3 K2H queue implementation | L | 1.2.1 | - |
| 1.2.4 Indirect Command Buffer setup | L | 1.1.4 | - |
| 1.2.5 MetalPersistentSimulation | L | 1.2.1-4 | - |
| 1.2.6 Lifecycle management (pause/resume/terminate) | M | 1.2.5 | - |
| 1.2.7 Integration tests | L | 1.2.1-6 | - |

#### Deliverables
- [ ] `MetalPersistentSimulation` matching CUDA API
- [ ] H2K/K2H SPSC queues with atomics
- [ ] Indirect Command Buffer persistence pattern
- [ ] 25+ tests passing

#### Acceptance Criteria
```rust
#[tokio::test]
async fn test_metal_persistent_kernel() {
    let device = MetalDevice::new()?;
    let config = PersistentConfig::new(64, 64, 64);
    let mut sim = MetalPersistentSimulation::new(&device, config)?;

    sim.start(&metal_lib, "persistent_kernel")?;
    sim.run_steps(100)?;

    let stats = sim.stats();
    assert_eq!(stats.current_step, 100);

    sim.shutdown()?;
}
```

---

### Sprint 1.3: Metal K2K Messaging (Weeks 9-10)

**Goal**: Inter-kernel communication on Metal

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 1.3.1 K2KInboxHeader for Metal | M | 1.2.5 | - |
| 1.3.2 K2KRouteEntry and routing table | M | 1.3.1 | - |
| 1.3.3 Threadgroup-based halo exchange | L | 1.3.2 | - |
| 1.3.4 K2K integration tests | M | 1.3.1-3 | - |

#### Deliverables
- [ ] K2K messaging between threadgroups
- [ ] Halo exchange for stencil patterns
- [ ] 10+ K2K tests passing

---

### Sprint 1.4: WebGPU Optimization (Weeks 9-12)

**Goal**: Optimize WebGPU for persistence emulation

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 1.4.1 WgpuPersistentEmulation design | M | None | - |
| 1.4.2 Batched command processing | L | 1.4.1 | - |
| 1.4.3 Efficient dispatch loop | M | 1.4.2 | - |
| 1.4.4 64-bit atomic emulation (complete) | L | None | - |
| 1.4.5 Subgroup operations (where available) | M | None | - |
| 1.4.6 Performance benchmarks | M | 1.4.1-5 | - |

#### Deliverables
- [ ] `WgpuPersistentEmulation` with batching
- [ ] Complete 64-bit atomic emulation
- [ ] Subgroup operation support detection
- [ ] Benchmark showing <100µs per batch

---

### Sprint 1.5: Ecosystem Streaming (Weeks 11-12)

**Goal**: Complete SSE and WebSocket handlers

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 1.5.1 SSE handler implementation | M | None | - |
| 1.5.2 SSE event formatting | S | 1.5.1 | - |
| 1.5.3 WebSocket handler implementation | M | None | - |
| 1.5.4 Bidirectional WebSocket protocol | M | 1.5.3 | - |
| 1.5.5 Integration tests with test client | M | 1.5.1-4 | - |

#### Deliverables
- [ ] `/api/events` SSE endpoint
- [ ] `/api/ws` WebSocket endpoint
- [ ] Example client implementations
- [ ] 15+ integration tests

---

## Phase 2: Unified Code Generation (Q2 2026)

### Sprint 2.1: IR Foundation (Weeks 1-4)

**Goal**: Create `ringkernel-ir` crate with core IR

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 2.1.1 IR node definitions (SSA-based) | L | None | - |
| 2.1.2 Type system with capability flags | L | 2.1.1 | - |
| 2.1.3 IR builder API | M | 2.1.1 | - |
| 2.1.4 IR pretty printer (debugging) | M | 2.1.1 | - |
| 2.1.5 IR validation passes | M | 2.1.1-2 | - |
| 2.1.6 Unit tests for IR | L | 2.1.1-5 | - |

#### IR Node Types
```rust
pub enum IrNode {
    // Values
    Constant(ConstantValue),
    Parameter(ParameterId, IrType),
    BinaryOp(BinaryOpKind, Box<IrNode>, Box<IrNode>),
    UnaryOp(UnaryOpKind, Box<IrNode>),

    // Control Flow
    Block(Vec<IrNode>),
    If(Box<IrNode>, Box<IrNode>, Option<Box<IrNode>>),
    Loop(Box<IrNode>),
    Break,
    Continue,
    Return(Option<Box<IrNode>>),

    // GPU-Specific
    ThreadId(Dimension),
    BlockId(Dimension),
    GridSync,
    ThreadgroupBarrier,
    AtomicOp(AtomicOpKind, Box<IrNode>, Box<IrNode>),

    // Memory
    Load(Box<IrNode>, IrType),
    Store(Box<IrNode>, Box<IrNode>),
    SharedAlloc(IrType, usize),

    // Messaging
    K2KSend(Box<IrNode>, Box<IrNode>),
    K2KRecv(IrType),
    H2KDequeue,
    K2HEnqueue(Box<IrNode>),
}
```

#### Deliverables
- [ ] `ringkernel-ir` crate with IR definitions
- [ ] Type system with `Capabilities` flags
- [ ] IR builder with ergonomic API
- [ ] 50+ unit tests

---

### Sprint 2.2: CUDA Lowering (Weeks 5-6)

**Goal**: Lower IR to CUDA PTX

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 2.2.1 IR → CUDA AST lowering | L | 2.1.1-5 | - |
| 2.2.2 CUDA-specific optimizations | M | 2.2.1 | - |
| 2.2.3 Integration with existing cuda-codegen | M | 2.2.1 | - |
| 2.2.4 Comparison tests (old vs new) | M | 2.2.3 | - |

#### Deliverables
- [ ] `IrToCuda` lowering pass
- [ ] Byte-identical output with legacy codegen
- [ ] 30+ comparison tests

---

### Sprint 2.3: WGSL Lowering (Weeks 7-8)

**Goal**: Lower IR to WGSL

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 2.3.1 IR → WGSL AST lowering | L | 2.1.1-5 | - |
| 2.3.2 64-bit atomic emulation in IR | M | 2.3.1 | - |
| 2.3.3 f64 → f32 downcast pass | M | 2.3.1 | - |
| 2.3.4 Integration with existing wgpu-codegen | M | 2.3.1 | - |
| 2.3.5 Comparison tests | M | 2.3.4 | - |

#### Deliverables
- [ ] `IrToWgsl` lowering pass
- [ ] Automatic capability-based transformations
- [ ] 30+ comparison tests

---

### Sprint 2.4: MSL Lowering (Weeks 9-12)

**Goal**: Lower IR to Metal Shading Language

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 2.4.1 IR → MSL AST lowering | L | 2.1.1-5 | - |
| 2.4.2 Metal-specific memory model | M | 2.4.1 | - |
| 2.4.3 Threadgroup coordination | M | 2.4.1 | - |
| 2.4.4 Argument buffer generation | M | 2.4.1 | - |
| 2.4.5 MSL compilation integration | M | 2.4.1-4 | - |
| 2.4.6 Cross-backend parity tests | L | 2.2-2.4 | - |

#### Deliverables
- [ ] `IrToMsl` lowering pass
- [ ] Complete MSL code generation
- [ ] 50+ tests ensuring parity

---

### Sprint 2.5: Multi-Backend Proc Macros (Weeks 11-12)

**Goal**: Unified kernel definition with backend selection

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 2.5.1 `backends` attribute parsing | M | 2.2-2.4 | - |
| 2.5.2 `fallback` attribute parsing | M | 2.5.1 | - |
| 2.5.3 Compile-time capability checking | L | 2.5.1 | - |
| 2.5.4 Multi-backend code generation | L | 2.5.1-3 | - |
| 2.5.5 Error message improvements | M | 2.5.1-4 | - |

#### Deliverables
- [ ] `#[ring_kernel(backends = [cuda, metal])]`
- [ ] `#[gpu_kernel(requires = [f64])]`
- [ ] Compile-time backend validation
- [ ] Clear error messages

---

## Phase 3: Enterprise Features (Q3 2026)

### Sprint 3.1: Kernel Checkpointing (Weeks 1-4)

**Goal**: Snapshot and restore persistent kernel state

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 3.1.1 CheckpointableKernel trait | M | None | - |
| 3.1.2 Checkpoint binary format | M | 3.1.1 | - |
| 3.1.3 GPU memory serialization | L | 3.1.2 | - |
| 3.1.4 Queue state serialization | M | 3.1.2 | - |
| 3.1.5 Checkpoint compression (optional) | M | 3.1.2 | - |
| 3.1.6 Restore implementation | L | 3.1.3-4 | - |
| 3.1.7 Checkpoint storage backends | M | 3.1.2 | - |
| 3.1.8 Integration tests | L | 3.1.1-7 | - |

#### Deliverables
- [ ] `CheckpointableKernel` trait
- [ ] File-based checkpoint storage
- [ ] S3/GCS checkpoint storage
- [ ] Checkpoint < 1s for 1GB state

---

### Sprint 3.2: Multi-GPU Support (Weeks 5-8)

**Goal**: Cross-GPU kernel coordination

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 3.2.1 GPU topology discovery | M | None | - |
| 3.2.2 NVLink/PCIe detection | M | 3.2.1 | - |
| 3.2.3 MultiGpuRuntime | L | 3.2.1-2 | - |
| 3.2.4 Cross-GPU K2K router | L | 3.2.3 | - |
| 3.2.5 Kernel migration | L | 3.2.3 | - |
| 3.2.6 Load balancing | M | 3.2.3 | - |
| 3.2.7 Multi-GPU benchmarks | M | 3.2.1-6 | - |

#### Deliverables
- [ ] `GpuTopology` with interconnect info
- [ ] `MultiGpuRuntime` with K2K routing
- [ ] Kernel migration between GPUs
- [ ] Benchmark showing near-linear scaling

---

### Sprint 3.3: Observability (Weeks 9-10)

**Goal**: Production observability infrastructure

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 3.3.1 OpenTelemetry tracing integration | M | None | - |
| 3.3.2 Trace context in MessageHeader | M | 3.3.1 | - |
| 3.3.3 NVIDIA Nsight markers | M | None | - |
| 3.3.4 Prometheus metrics enhancement | M | None | - |
| 3.3.5 Grafana dashboard templates | M | 3.3.4 | - |

#### Deliverables
- [ ] Trace propagation through K2K
- [ ] NVTX integration for profiling
- [ ] Grafana dashboard JSON
- [ ] Jaeger trace visualization

---

### Sprint 3.4: Health & Resilience (Weeks 11-12)

**Goal**: Production health monitoring

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 3.4.1 HealthCheck trait | M | None | - |
| 3.4.2 Built-in health checks | M | 3.4.1 | - |
| 3.4.3 Graceful degradation | M | 3.4.1 | - |
| 3.4.4 Hot reload implementation | L | None | - |
| 3.4.5 Health endpoint for Kubernetes | S | 3.4.1 | - |

#### Deliverables
- [ ] Health monitoring with alerting
- [ ] CPU fallback under pressure
- [ ] Hot reload with <100ms downtime
- [ ] Kubernetes readiness/liveness probes

---

## Phase 4: Ecosystem Expansion (Q4 2026)

### Sprint 4.1: Data Processing Integration (Weeks 1-4)

**Goal**: GPU-accelerated data processing

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 4.1.1 Arrow GPU kernels | L | None | - |
| 4.1.2 GpuArrowOps trait implementation | L | 4.1.1 | - |
| 4.1.3 Polars GPU backend | L | 4.1.1 | - |
| 4.1.4 Candle tensor operations | L | None | - |
| 4.1.5 DataFusion GPU executor | L | 4.1.1 | - |

#### Deliverables
- [ ] Arrow filter/sum/sort on GPU
- [ ] Polars expressions GPU-accelerated
- [ ] Candle model inference
- [ ] DataFusion query GPU execution

---

### Sprint 4.2: CLI & Tooling (Weeks 5-8)

**Goal**: Developer CLI and tooling

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 4.2.1 ringkernel-cli crate | M | None | - |
| 4.2.2 `new` command with templates | M | 4.2.1 | - |
| 4.2.3 `codegen` command | M | 4.2.1 | - |
| 4.2.4 `check` command | M | 4.2.1 | - |
| 4.2.5 `profile` command | L | 4.2.1 | - |
| 4.2.6 `watch` mode | M | 4.2.1 | - |
| 4.2.7 VSCode extension | L | None | - |

#### Deliverables
- [ ] `ringkernel` CLI v1.0
- [ ] Project templates
- [ ] VSCode extension with IntelliSense
- [ ] Integrated profiler

---

### Sprint 4.3: Documentation (Weeks 9-12)

**Goal**: Comprehensive documentation

#### Tasks

| Task | Effort | Dependencies | Assignee |
|------|--------|--------------|----------|
| 4.3.1 Interactive tutorials (5) | L | None | - |
| 4.3.2 Architecture guide | L | None | - |
| 4.3.3 API reference completion | L | None | - |
| 4.3.4 Example gallery (10+) | L | None | - |
| 4.3.5 Video tutorials (3) | M | 4.3.1 | - |

#### Deliverables
- [ ] 5 interactive tutorials
- [ ] Complete architecture guide
- [ ] 95% rustdoc coverage
- [ ] 10+ real-world examples

---

## Resource Requirements

### Team Structure

| Role | Count | Responsibility |
|------|-------|----------------|
| GPU Systems Engineer | 2 | Backend implementation |
| Compiler Engineer | 1 | Code generation, IR |
| Platform Engineer | 1 | CI/CD, tooling |
| DevRel Engineer | 1 | Documentation, examples |

### Infrastructure

| Resource | Purpose |
|----------|---------|
| NVIDIA GPU CI runners | CUDA testing |
| Apple Silicon CI runners | Metal testing |
| Multi-GPU test machines | Scaling tests |
| Cloud storage | Checkpoint testing |

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Metal ICB limitations | Medium | High | Research alternative persistence patterns |
| WebGPU subgroup support | Medium | Medium | Feature detection, fallback paths |
| Multi-GPU NVLink complexity | Low | High | Start with PCIe-only path |
| IR design iterations | High | Medium | Prototype with subset of features |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Metal backend delays | Medium | High | Parallel work on WebGPU optimization |
| IR complexity underestimated | Medium | High | MVP IR first, features later |
| External dependency breaks | Low | Medium | Pin versions, vendor critical deps |

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Metal persistent kernels pass 50+ tests
- [ ] WebGPU batch latency < 100µs
- [ ] SSE/WebSocket handlers in production use

### Phase 2 Complete When:
- [ ] IR compiles identical code to legacy codegen
- [ ] MSL generation produces working Metal shaders
- [ ] Multi-backend proc macros documented

### Phase 3 Complete When:
- [ ] Checkpoint/restore < 1s for 1GB state
- [ ] Multi-GPU shows >80% linear scaling
- [ ] Traces visible in Jaeger

### Phase 4 Complete When:
- [ ] Arrow GPU operations benchmarked
- [ ] CLI has 100+ downloads/week
- [ ] Documentation rated >4.5/5

---

## Appendix: Effort Estimation Guidelines

| Category | Small (S) | Medium (M) | Large (L) |
|----------|-----------|------------|-----------|
| Definition | < 3 days | 3-7 days | 1-3 weeks |
| Complexity | Single concern | Multiple components | System-wide |
| Testing | Unit tests | Integration tests | E2E + benchmarks |
| Review | 1 reviewer | 2 reviewers | Team review |
| Documentation | API docs | Usage examples | Architecture docs |
