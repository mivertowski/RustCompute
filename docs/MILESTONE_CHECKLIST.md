# Milestone Checklist

> Trackable Milestones with Acceptance Criteria

## How to Use This Document

Each milestone contains:
- **Objective**: What we're trying to achieve
- **Deliverables**: Concrete outputs
- **Acceptance Criteria**: How we know it's done
- **Verification Steps**: Commands to verify completion
- **Dependencies**: What must be complete first

Mark items with:
- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[!]` Blocked

---

## Phase 1: Foundation Completion (Q1 2026)

### Milestone 1.1: Metal Backend Core
**Target Date**: End of Week 4

#### Objective
Implement basic Metal kernel execution with CPU/GPU shared memory.

#### Deliverables
- [ ] `ringkernel-metal/src/device.rs` - Device enumeration and capability detection
- [ ] `ringkernel-metal/src/buffer.rs` - MetalMappedBuffer implementation
- [ ] `ringkernel-metal/src/pipeline.rs` - Compute pipeline creation
- [ ] `ringkernel-metal/src/runtime.rs` - RingKernelRuntime implementation
- [ ] `ringkernel-metal/tests/` - 15+ unit tests

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Device enumeration returns all Metal GPUs | [ ] | |
| MetalMappedBuffer allows CPU read/write | [ ] | |
| MetalMappedBuffer allows GPU read/write | [ ] | |
| Basic compute kernel launches successfully | [ ] | |
| Kernel output matches expected values | [ ] | |
| All tests pass on Apple Silicon | [ ] | |
| All tests pass on Intel Mac (if available) | [ ] | |

#### Verification Steps
```bash
# Build Metal backend
cargo build --package ringkernel-metal --features metal

# Run Metal tests
cargo test --package ringkernel-metal --features metal

# Verify device detection
cargo run --package ringkernel-metal --example list_devices --features metal

# Expected output:
# Metal Device 0: Apple M1 Pro (unified memory: 16GB)
```

#### Dependencies
- None (starting point)

---

### Milestone 1.2: Metal Persistent Kernels
**Target Date**: End of Week 8

#### Objective
Implement persistent kernel architecture matching CUDA capabilities.

#### Deliverables
- [ ] `ringkernel-metal/src/persistent.rs` - MetalPersistentSimulation (800+ lines)
- [ ] `ringkernel-metal/src/control_block.rs` - PersistentControlBlock for Metal
- [ ] `ringkernel-metal/src/queue.rs` - H2K/K2H queue implementations
- [ ] `ringkernel-metal/tests/persistent_*.rs` - 25+ integration tests

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| PersistentControlBlock accessible from CPU and GPU | [ ] | |
| H2K queue successfully delivers commands | [ ] | |
| K2H queue successfully delivers responses | [ ] | |
| Kernel runs for 10,000+ steps without relaunch | [ ] | |
| Pause/Resume works correctly | [ ] | |
| Graceful shutdown terminates kernel | [ ] | |
| Command injection latency < 1µs | [ ] | |

#### Verification Steps
```bash
# Run persistent kernel tests
cargo test --package ringkernel-metal --features metal persistent

# Run lifecycle test
cargo test --package ringkernel-metal --features metal test_persistent_lifecycle

# Benchmark command injection
cargo bench --package ringkernel-metal --features metal -- command_injection

# Expected: p50 < 1µs, p99 < 10µs
```

#### Dependencies
- Milestone 1.1 complete

---

### Milestone 1.3: Metal K2K Messaging
**Target Date**: End of Week 10

#### Objective
Enable inter-kernel communication on Metal.

#### Deliverables
- [ ] `ringkernel-metal/src/k2k.rs` - K2K infrastructure
- [ ] Threadgroup-based halo exchange implementation
- [ ] 10+ K2K integration tests

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| K2K messages route between threadgroups | [ ] | |
| Halo exchange works for 3D stencil | [ ] | |
| No data corruption under stress | [ ] | |
| K2K latency < 10µs | [ ] | |

#### Verification Steps
```bash
# Run K2K tests
cargo test --package ringkernel-metal --features metal k2k

# Run halo exchange stress test
cargo test --package ringkernel-metal --features metal --release test_halo_stress

# Verify correctness
cargo run --package ringkernel-metal --example k2k_verify --features metal
```

#### Dependencies
- Milestone 1.2 complete

---

### Milestone 1.4: WebGPU Optimization
**Target Date**: End of Week 12

#### Objective
Optimize WebGPU for persistence emulation with batched dispatch.

#### Deliverables
- [ ] `ringkernel-wgpu/src/persistent_emulation.rs` - WgpuPersistentEmulation
- [ ] Complete 64-bit atomic emulation
- [ ] Subgroup operation detection and usage
- [ ] Performance benchmarks

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Batched dispatch processes 100 commands | [ ] | |
| Per-batch latency < 100µs | [ ] | |
| 64-bit atomics work correctly | [ ] | |
| Subgroup ops used when available | [ ] | |
| Cross-platform tests pass (Vulkan, DX12) | [ ] | |

#### Verification Steps
```bash
# Run WebGPU tests
cargo test --package ringkernel-wgpu --features wgpu-tests -- --ignored

# Run on specific backend
WGPU_BACKEND=vulkan cargo test --package ringkernel-wgpu --features wgpu-tests

# Benchmark batched dispatch
cargo bench --package ringkernel-wgpu --features wgpu-tests -- batch_dispatch
```

#### Dependencies
- None (parallel work)

---

### Milestone 1.5: Ecosystem Streaming
**Target Date**: End of Week 12

#### Objective
Complete SSE and WebSocket handlers for real-time kernel updates.

#### Deliverables
- [ ] `ringkernel-ecosystem/src/axum.rs` - SSE handler
- [ ] `ringkernel-ecosystem/src/axum.rs` - WebSocket handler
- [ ] Example client implementations
- [ ] 15+ integration tests

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| SSE endpoint streams kernel events | [ ] | |
| WebSocket allows bidirectional commands | [ ] | |
| Connection handles 1000+ events | [ ] | |
| Reconnection works correctly | [ ] | |
| Example React client works | [ ] | |

#### Verification Steps
```bash
# Run ecosystem tests
cargo test --package ringkernel-ecosystem --features "axum,persistent"

# Run SSE example
cargo run --package ringkernel-ecosystem --example sse_server --features "axum,persistent"

# Test with curl
curl -N http://localhost:3000/api/events
```

#### Dependencies
- None (parallel work)

---

## Phase 2: Unified Code Generation (Q2 2026)

### Milestone 2.1: IR Foundation
**Target Date**: End of Week 4

#### Objective
Create `ringkernel-ir` crate with SSA-based intermediate representation.

#### Deliverables
- [ ] `ringkernel-ir/src/node.rs` - IR node definitions
- [ ] `ringkernel-ir/src/types.rs` - Type system with capabilities
- [ ] `ringkernel-ir/src/builder.rs` - IR builder API
- [ ] `ringkernel-ir/src/validate.rs` - Validation passes
- [ ] `ringkernel-ir/src/print.rs` - Pretty printer
- [ ] 50+ unit tests

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| All IR nodes defined and documented | [ ] | |
| Type system captures GPU capabilities | [ ] | |
| Builder produces valid IR | [ ] | |
| Validator catches invalid IR | [ ] | |
| Pretty printer outputs readable IR | [ ] | |

#### Verification Steps
```bash
# Build IR crate
cargo build --package ringkernel-ir

# Run IR tests
cargo test --package ringkernel-ir

# Verify IR pretty printing
cargo run --package ringkernel-ir --example ir_printer
```

#### Dependencies
- None (starting point)

---

### Milestone 2.2: CUDA IR Lowering
**Target Date**: End of Week 6

#### Objective
Lower IR to CUDA PTX, matching legacy codegen output.

#### Deliverables
- [ ] `ringkernel-ir/src/lower/cuda.rs` - IR to CUDA lowering
- [ ] Integration with existing cuda-codegen
- [ ] 30+ comparison tests

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| IR lowers to valid CUDA | [ ] | |
| Output matches legacy codegen | [ ] | |
| All 183 existing tests pass | [ ] | |
| Performance equivalent to legacy | [ ] | |

#### Verification Steps
```bash
# Run comparison tests
cargo test --package ringkernel-ir compare_cuda

# Diff generated CUDA
diff <(cargo run --package ringkernel-ir --example lower_saxpy_ir) \
     <(cargo run --package ringkernel-cuda-codegen --example saxpy)
```

#### Dependencies
- Milestone 2.1 complete

---

### Milestone 2.3: WGSL IR Lowering
**Target Date**: End of Week 8

#### Objective
Lower IR to WGSL with automatic capability-based transformations.

#### Deliverables
- [ ] `ringkernel-ir/src/lower/wgsl.rs` - IR to WGSL lowering
- [ ] 64-bit atomic transformation pass
- [ ] f64 to f32 downcast pass
- [ ] 30+ comparison tests

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| IR lowers to valid WGSL | [ ] | |
| 64-bit atomics auto-transformed | [ ] | |
| f64 auto-downcast with warning | [ ] | |
| All 50 existing tests pass | [ ] | |

#### Verification Steps
```bash
# Run comparison tests
cargo test --package ringkernel-ir compare_wgsl

# Verify automatic transformation
cargo run --package ringkernel-ir --example transform_atomics
```

#### Dependencies
- Milestone 2.1 complete

---

### Milestone 2.4: MSL IR Lowering
**Target Date**: End of Week 12

#### Objective
Lower IR to Metal Shading Language.

#### Deliverables
- [ ] `ringkernel-ir/src/lower/msl.rs` - IR to MSL lowering
- [ ] Metal memory model handling
- [ ] Threadgroup coordination generation
- [ ] 50+ tests

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| IR lowers to valid MSL | [ ] | |
| Generated MSL compiles with Metal | [ ] | |
| Kernels produce correct results | [ ] | |
| Feature parity with CUDA codegen | [ ] | |

#### Verification Steps
```bash
# Run MSL lowering tests
cargo test --package ringkernel-ir lower_msl

# Compile generated MSL
cargo run --package ringkernel-ir --example compile_msl --features metal
```

#### Dependencies
- Milestone 2.1 complete
- Milestone 1.2 complete (for testing)

---

### Milestone 2.5: Multi-Backend Proc Macros
**Target Date**: End of Week 12

#### Objective
Enable unified kernel definitions with backend selection.

#### Deliverables
- [ ] `backends` attribute in `#[ring_kernel]`
- [ ] `fallback` attribute for graceful degradation
- [ ] Compile-time capability checking
- [ ] Clear error messages

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| `backends = [cuda, metal]` generates both | [ ] | |
| `fallback = wgpu` works correctly | [ ] | |
| `requires = [f64]` errors on WGSL | [ ] | |
| Error messages are actionable | [ ] | |

#### Verification Steps
```rust
// This should compile and generate CUDA + Metal:
#[ring_kernel(backends = [cuda, metal], fallback = wgpu)]
fn example(ctx: &RingContext) -> u32 { 42 }

// This should error at compile time:
#[ring_kernel(backends = [wgpu], requires = [f64])]
fn example(data: &[f64]) {}
// Error: WebGPU does not support f64
```

#### Dependencies
- Milestones 2.2, 2.3, 2.4 complete

---

## Phase 3: Enterprise Features (Q3 2026)

### Milestone 3.1: Kernel Checkpointing
**Target Date**: End of Week 4

#### Objective
Enable snapshot and restore of persistent kernel state.

#### Deliverables
- [ ] `CheckpointableKernel` trait
- [ ] Binary checkpoint format
- [ ] File storage backend
- [ ] S3/GCS storage backends
- [ ] Checkpoint/restore tests

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Checkpoint captures full state | [ ] | |
| Restore recovers exact state | [ ] | |
| 1GB checkpoint completes in < 1s | [ ] | |
| Compressed checkpoints work | [ ] | |
| Cloud storage backends work | [ ] | |

#### Verification Steps
```bash
# Run checkpoint tests
cargo test --package ringkernel-enterprise checkpoint

# Benchmark checkpoint time
cargo bench --package ringkernel-enterprise -- checkpoint_1gb

# Test S3 backend
AWS_REGION=us-west-2 cargo test --package ringkernel-enterprise checkpoint_s3
```

#### Dependencies
- Phase 1 complete

---

### Milestone 3.2: Multi-GPU Support
**Target Date**: End of Week 8

#### Objective
Enable cross-GPU kernel coordination.

#### Deliverables
- [ ] GPU topology discovery
- [ ] MultiGpuRuntime
- [ ] Cross-GPU K2K router
- [ ] Kernel migration
- [ ] Load balancing

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Topology detection finds all GPUs | [ ] | |
| NVLink connections detected | [ ] | |
| K2K works across GPUs | [ ] | |
| Migration preserves state | [ ] | |
| 80% linear scaling on 4 GPUs | [ ] | |

#### Verification Steps
```bash
# Run on multi-GPU machine
cargo test --package ringkernel-multi-gpu

# Benchmark scaling
cargo bench --package ringkernel-multi-gpu -- scaling

# Verify topology
cargo run --package ringkernel-multi-gpu --example show_topology
```

#### Dependencies
- Phase 1 complete

---

### Milestone 3.3: Observability
**Target Date**: End of Week 10

#### Objective
Production-grade observability infrastructure.

#### Deliverables
- [ ] OpenTelemetry integration
- [ ] NVIDIA Nsight markers
- [ ] Enhanced Prometheus metrics
- [ ] Grafana dashboard

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Traces visible in Jaeger | [ ] | |
| NVTX markers in Nsight | [ ] | |
| Prometheus scrape works | [ ] | |
| Grafana dashboard loads | [ ] | |

#### Verification Steps
```bash
# Start tracing
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
cargo run --package ringkernel --example traced_kernel

# View in Jaeger
open http://localhost:16686

# Scrape metrics
curl http://localhost:9090/metrics
```

#### Dependencies
- None (parallel work)

---

### Milestone 3.4: Health & Resilience
**Target Date**: End of Week 12

#### Objective
Production health monitoring and resilience.

#### Deliverables
- [ ] HealthCheck trait and built-in checks
- [ ] Graceful degradation
- [ ] Hot reload implementation
- [ ] Kubernetes integration

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Health checks detect failures | [ ] | |
| CPU fallback works under pressure | [ ] | |
| Hot reload < 100ms downtime | [ ] | |
| K8s probes respond correctly | [ ] | |

#### Verification Steps
```bash
# Test health checks
cargo test --package ringkernel-enterprise health

# Test hot reload
cargo test --package ringkernel-enterprise hot_reload

# K8s probe check
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz
```

#### Dependencies
- None (parallel work)

---

## Phase 4: Ecosystem Expansion (Q4 2026)

### Milestone 4.1: Data Processing Integration
**Target Date**: End of Week 4

#### Objective
GPU-accelerated data processing with Arrow, Polars, Candle.

#### Deliverables
- [ ] Arrow GPU kernels
- [ ] Polars GPU backend
- [ ] Candle integration
- [ ] Benchmarks vs CPU

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Arrow filter/sort on GPU | [ ] | |
| Polars expressions GPU-accelerated | [ ] | |
| Candle inference works | [ ] | |
| 10x speedup vs CPU | [ ] | |

#### Verification Steps
```bash
# Run integration tests
cargo test --package ringkernel-data --features "arrow,polars,candle"

# Benchmark
cargo bench --package ringkernel-data -- arrow_filter
```

#### Dependencies
- Phase 2 complete

---

### Milestone 4.2: CLI & Tooling
**Target Date**: End of Week 8

#### Objective
Developer CLI and VSCode extension.

#### Deliverables
- [ ] `ringkernel` CLI v1.0
- [ ] Project templates
- [ ] VSCode extension
- [ ] Integrated profiler

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| `ringkernel new` creates project | [ ] | |
| `ringkernel codegen` generates code | [ ] | |
| `ringkernel check` validates | [ ] | |
| VSCode shows diagnostics | [ ] | |

#### Verification Steps
```bash
# Install CLI
cargo install ringkernel-cli

# Create project
ringkernel new my-app --template persistent-actor
cd my-app
cargo build

# Generate code
ringkernel codegen src/kernels/processor.rs --backend cuda,metal
```

#### Dependencies
- Phase 2 complete

---

### Milestone 4.3: Documentation
**Target Date**: End of Week 12

#### Objective
Comprehensive documentation suite.

#### Deliverables
- [ ] 5 interactive tutorials
- [ ] Architecture guide
- [ ] 95% API coverage
- [ ] 10+ examples
- [ ] 3 video tutorials

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Tutorials run without errors | [ ] | |
| Architecture guide complete | [ ] | |
| All public APIs documented | [ ] | |
| Examples compile and run | [ ] | |

#### Verification Steps
```bash
# Build documentation
cargo doc --workspace --no-deps

# Check coverage
cargo doc-coverage --workspace

# Run tutorial code
cd docs/tutorials/01-hello-gpu
cargo run
```

#### Dependencies
- All previous milestones (for accuracy)

---

## Summary Dashboard

### Phase 1 Progress

| Milestone | Target | Status | Blockers |
|-----------|--------|--------|----------|
| 1.1 Metal Core | Week 4 | [ ] | |
| 1.2 Metal Persistent | Week 8 | [ ] | 1.1 |
| 1.3 Metal K2K | Week 10 | [ ] | 1.2 |
| 1.4 WebGPU Opt | Week 12 | [ ] | |
| 1.5 Streaming | Week 12 | [ ] | |

### Phase 2 Progress

| Milestone | Target | Status | Blockers |
|-----------|--------|--------|----------|
| 2.1 IR Foundation | Week 4 | [ ] | |
| 2.2 CUDA Lowering | Week 6 | [ ] | 2.1 |
| 2.3 WGSL Lowering | Week 8 | [ ] | 2.1 |
| 2.4 MSL Lowering | Week 12 | [ ] | 2.1, 1.2 |
| 2.5 Multi-Backend | Week 12 | [ ] | 2.2-2.4 |

### Phase 3 Progress

| Milestone | Target | Status | Blockers |
|-----------|--------|--------|----------|
| 3.1 Checkpointing | Week 4 | [ ] | Phase 1 |
| 3.2 Multi-GPU | Week 8 | [ ] | Phase 1 |
| 3.3 Observability | Week 10 | [ ] | |
| 3.4 Resilience | Week 12 | [ ] | |

### Phase 4 Progress

| Milestone | Target | Status | Blockers |
|-----------|--------|--------|----------|
| 4.1 Data Processing | Week 4 | [ ] | Phase 2 |
| 4.2 CLI/Tooling | Week 8 | [ ] | Phase 2 |
| 4.3 Documentation | Week 12 | [ ] | All |

---

## Appendix: Milestone Template

```markdown
### Milestone X.Y: [Name]
**Target Date**: End of Week N

#### Objective
[One sentence describing the goal]

#### Deliverables
- [ ] Deliverable 1
- [ ] Deliverable 2

#### Acceptance Criteria

| Criterion | Status | Notes |
|-----------|--------|-------|
| Criterion 1 | [ ] | |
| Criterion 2 | [ ] | |

#### Verification Steps
```bash
# Commands to verify completion
```

#### Dependencies
- [List of prerequisite milestones]
```
