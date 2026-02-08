# RingKernel Production Readiness Roadmap

> A comprehensive analysis and strategic plan for taking RingKernel from a feature-complete
> framework (v0.4.2) to a battle-tested, production-grade GPU actor system.

**Date**: February 2026
**Current version**: 0.4.2
**Target**: v1.0.0 stable release

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Assessment](#current-state-assessment)
3. [Gap Analysis](#gap-analysis)
4. [Roadmap Phases](#roadmap-phases)
   - [Phase A: Safety & Correctness](#phase-a-safety--correctness-q1-q2-2026)
   - [Phase B: Operational Maturity](#phase-b-operational-maturity-q2-q3-2026)
   - [Phase C: Performance Hardening](#phase-c-performance-hardening-q3-2026)
   - [Phase D: Ecosystem & API Stability](#phase-d-ecosystem--api-stability-q3-q4-2026)
   - [Phase E: Enterprise Deployment](#phase-e-enterprise-deployment-q4-2026-q1-2027)
5. [Detailed Work Items](#detailed-work-items)
6. [Risk Register](#risk-register)
7. [Success Criteria for v1.0](#success-criteria-for-v10)
8. [Research References](#research-references)

---

## Executive Summary

RingKernel has achieved 100% feature completion across its original 5-phase roadmap, delivering
a GPU-native persistent actor model with ~195K lines of Rust across 22 crates, 950+ tests,
5 fuzz targets, and multi-backend support (CUDA, WebGPU, Metal, CPU). The framework is
**feature-rich but pre-production**.

This roadmap addresses the gap between "feature-complete" and "production-ready" by focusing on:

1. **Safety verification** of ~290 `unsafe` blocks across 46 files via Miri/Kani integration
2. **Operational observability** using OpenTelemetry, structured metrics, and health endpoints
3. **API stability** through semver enforcement and a deliberate path to 1.0
4. **Deployment infrastructure** with containerization, GPU passthrough, and orchestration
5. **Performance baselines** with regression detection and continuous benchmarking

The estimated timeline is 9-12 months to reach a production-stable v1.0.0 release.

---

## Current State Assessment

### Codebase Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Total Rust LOC | ~195,000 | Large, well-structured workspace |
| Workspace crates | 22 (+ tutorials, fuzz) | Good modularity |
| Test count | 950+ (`#[test]` annotations) | Moderate coverage |
| Fuzz targets | 5 (IR, CUDA/WGSL transpilers, message queue, HLC) | Good foundation |
| `unsafe` blocks | ~290 across 46 files | Requires systematic audit |
| Doc comments | ~400+ across codebase | Moderate; gaps in newer crates |
| CI workflows | 4 (ci, gpu-tests, release, docs) | Good structure |
| Benchmark suites | Criterion benches + application benchmarks | Needs regression tracking |
| Version | 0.4.2 | Pre-1.0, no semver stability guarantee |

### Architecture Strengths

- **Clean workspace structure**: Well-separated concerns across core, backends, codegen, ecosystem
- **Feature flag discipline**: Backend selection via Cargo features with auto-detection fallback
- **Zero-copy serialization**: rkyv 0.7 with validation enabled (`bytecheck`, `strict` mode)
- **Multi-backend codegen**: Unified IR (`ringkernel-ir`) lowers to CUDA PTX, WGSL, and MSL
- **Enterprise features**: Checkpoint/restore, multi-GPU, security (AES-256-GCM, audit logging)
- **CI/CD pipeline**: GitHub Actions with fmt, clippy, test, doc, MSRV (1.75), GPU-specific workflows
- **Fuzz testing**: 5 targets covering critical paths (transpilers, message queue, HLC, IR)
- **Release automation**: Multi-platform release workflow (Linux/Windows, CPU/CUDA)

### Identified Gaps

| Area | Current State | Gap |
|------|--------------|-----|
| **Unsafe audit** | ~290 unsafe blocks, no systematic review | No Miri/Kani CI integration |
| **Observability** | `tracing` crate used, basic spans | No OpenTelemetry export, no metrics pipeline |
| **Containerization** | No Docker/k8s configs | Missing production deployment path |
| **API stability** | v0.4.2, no semver checking | No `cargo-semver-checks` in CI |
| **Error recovery** | `thiserror` 2.0 typed errors | No circuit breaker, backpressure signals |
| **Load testing** | Application benchmarks exist | No sustained load / soak tests |
| **Dependency audit** | No `cargo-audit` in CI | Unmonitored supply chain |
| **Documentation** | 19 architecture docs, tutorials | Missing operations guide, runbooks |
| **Graceful shutdown** | `DegradationManager` exists | No signal handling / drain protocol |
| **Memory pressure** | `GpuMemoryDashboard` exists | No automatic OOM recovery |

---

## Gap Analysis

### 1. Safety & Soundness

**The ~290 `unsafe` blocks are the highest-priority production risk.** They span:

- **CUDA driver calls** (`ringkernel-cuda`): kernel launch, memory allocation, stream management (~99 occurrences)
- **GPU memory mapping** (`ringkernel-metal`, `ringkernel-wgpu`): buffer access, pointer casts
- **Zero-copy deserialization** (`ringkernel-core`): rkyv archive access
- **Lock-free queue operations** (`ringkernel-core`): atomic operations, memory ordering
- **FFI boundaries** (`ringkernel-cuda`, `ringkernel-metal`): cudarc, metal-rs interop
- **Proc macro code generation** (`ringkernel-derive`): generated unsafe code paths

**Industry context**: A 2024 mixed-methods study on unsafe Rust found that developers "rarely
audited their dependencies and relied on ad-hoc reasoning" for unsafe code correctness.
AWS has launched a formal initiative to verify the safety of the Rust standard library
using Kani, signaling that model-checking unsafe code is becoming industry standard practice.

### 2. Observability & Operations

The framework uses `tracing` for structured logging but lacks:

- **OpenTelemetry integration**: No OTLP exporter for traces, metrics, or logs
- **Four Golden Signals**: Latency, traffic, errors, saturation not exposed as metrics
- **Health check endpoints**: Enterprise `HealthChecker` exists but no standard HTTP endpoints
- **Structured alerting**: `alerting` feature flag exists but no Prometheus/Grafana integration
- **Distributed tracing context propagation**: Message headers support trace context but no
  end-to-end demo with Jaeger/Tempo

**Industry context**: The 2025 production readiness consensus recommends OpenTelemetry as the
standard instrumentation layer, with `tracing-opentelemetry` bridging Rust's tracing ecosystem
to vendor-agnostic backends.

### 3. Deployment & Containerization

No Docker images, Helm charts, or Kubernetes manifests exist. The release workflow builds
binaries but doesn't package them for container orchestration.

**Industry context**: The NVIDIA Container Toolkit is the standard for GPU workloads in
containers. wgpu requires Vulkan ICD loaders inside containers. CubeCL (Burn's compute
backend) has demonstrated CUDA + wgpu from a single Rust binary in containers.

### 4. API Stability & Semver

At v0.4.2, the project is pre-1.0 with no formal semver enforcement. The `cargo-semver-checks`
tool should be integrated into CI before any 1.0 declaration.

**Industry context**: The Rust ecosystem strongly recommends not staying at 0.x indefinitely,
as it reduces semver expressivity from three categories to two. The Cargo SemVer Compatibility
Guide and RFC 1105 provide detailed definitions of breaking vs. non-breaking changes.

### 5. Lock-Free Queue Reliability

The lock-free message queue is a critical path for all actor communication. While it has
fuzz testing, it needs:

- **Linearizability proof or testing**: Loom-based concurrency testing
- **Boundary condition handling**: Recent GPU queue research (BACQ, 2025) shows that
  boundary-aware designs significantly improve reliability under high contention
- **Backpressure signaling**: No mechanism to signal queue saturation to producers

---

## Roadmap Phases

### Phase A: Safety & Correctness (Q1-Q2 2026)

**Goal**: Achieve verifiable safety for all `unsafe` code and critical data structures.

| # | Work Item | Priority | Effort | Description |
|---|-----------|----------|--------|-------------|
| A1 | **Unsafe code audit** | P0 | Large | Systematic review of all ~290 unsafe blocks; document safety invariants |
| A2 | **Miri CI integration** | P0 | Medium | Add Miri test runs for core crate, message queue, HLC |
| A3 | **Kani verification harnesses** | P1 | Large | Model-check lock-free queue, memory mapping, archive access |
| A4 | **Loom concurrency testing** | P0 | Medium | Test lock-free queue and HLC under all possible thread interleavings |
| A5 | **`cargo-audit` in CI** | P0 | Small | Add dependency vulnerability scanning to CI pipeline |
| A6 | **rkyv migration plan** | P1 | Medium | Evaluate rkyv 0.8 (pre-1.0) or pin 0.7 with safety wrappers |
| A7 | **SAFETY.md documentation** | P1 | Small | Document all unsafe invariants, audit status, and verification coverage |
| A8 | **Clippy strictness** | P1 | Small | Enable `clippy::pedantic` and `clippy::nursery` lint groups |

**Key risks**: rkyv 0.7 has known alignment issues (RUSTSEC-2021-0054); `AlignedVec` must be
used consistently. The lock-free queue needs formal testing beyond fuzz (Loom provides
exhaustive scheduling exploration).

### Phase B: Operational Maturity (Q2-Q3 2026)

**Goal**: Make RingKernel observable, operable, and recoverable in production.

| # | Work Item | Priority | Effort | Description |
|---|-----------|----------|--------|-------------|
| B1 | **OpenTelemetry integration** | P0 | Large | Add `tracing-opentelemetry` bridge; OTLP exporter for traces + metrics |
| B2 | **Prometheus metrics endpoint** | P0 | Medium | Expose four golden signals + GPU-specific metrics (memory, utilization, queue depth) |
| B3 | **Health check HTTP endpoints** | P0 | Small | `/healthz`, `/readyz`, `/livez` per Kubernetes conventions |
| B4 | **Graceful shutdown protocol** | P0 | Medium | Signal handling (SIGTERM/SIGINT), kernel drain, checkpoint-on-shutdown |
| B5 | **Circuit breaker pattern** | P1 | Medium | Automatic fallback when GPU backends fail; CPU degradation path |
| B6 | **Backpressure signaling** | P1 | Medium | Queue saturation detection with producer flow control |
| B7 | **Structured error taxonomy** | P1 | Medium | Classify errors: retriable vs. fatal, user-facing vs. internal |
| B8 | **Operations runbook** | P1 | Medium | Document common failure modes, debugging procedures, recovery steps |
| B9 | **Log correlation** | P2 | Small | Ensure trace IDs propagate through K2K messages and across GPU boundaries |
| B10 | **GPU OOM recovery** | P1 | Medium | Automatic memory pressure relief: eviction, checkpoint-to-disk, degradation |

**Key insight**: The existing `GpuMemoryDashboard` and `DegradationManager` provide a
foundation; this phase wires them into standard observability pipelines and adds missing
recovery paths.

### Phase C: Performance Hardening (Q3 2026)

**Goal**: Establish performance baselines, detect regressions, and optimize critical paths.

| # | Work Item | Priority | Effort | Description |
|---|-----------|----------|--------|-------------|
| C1 | **Continuous benchmarking** | P0 | Medium | Integrate Criterion results into CI with regression detection (e.g., `bencher.dev` or `codspeed`) |
| C2 | **Soak testing framework** | P0 | Medium | 24-hour sustained load tests for message throughput, memory stability |
| C3 | **Lock-free queue optimization** | P1 | Large | Evaluate BACQ-style warp-level optimizations for GPU-side queues |
| C4 | **Memory pool tuning** | P1 | Medium | Profile and optimize CUDA memory pool allocation patterns |
| C5 | **Latency percentile tracking** | P1 | Small | p50/p95/p99/p999 latency histograms for message send/receive |
| C6 | **GPU kernel occupancy profiling** | P1 | Medium | Automated occupancy analysis integrated with `GpuProfilerManager` |
| C7 | **Zero-copy path audit** | P2 | Medium | Verify zero-copy is maintained end-to-end; identify hidden copies |
| C8 | **Compile time optimization** | P2 | Medium | Profile and reduce workspace build times; evaluate `cargo-nextest` |

**Research context**: The 2025 BACQ paper (Polak et al., Applied Sciences) demonstrates that
boundary-aware concurrent queue designs on GPUs achieve superior throughput through warp-level
coordination and virtual caching layers, directly applicable to RingKernel's GPU-side queues.

### Phase D: Ecosystem & API Stability (Q3-Q4 2026)

**Goal**: Stabilize public APIs and prepare the crate ecosystem for 1.0.

| # | Work Item | Priority | Effort | Description |
|---|-----------|----------|--------|-------------|
| D1 | **`cargo-semver-checks` in CI** | P0 | Small | Automated semver compliance on every PR |
| D2 | **API review: `ringkernel-core`** | P0 | Large | Audit all public types, traits, and functions for 1.0 stability |
| D3 | **API review: `ringkernel-derive`** | P0 | Medium | Stabilize proc macro attributes and generated code shape |
| D4 | **API review: backend traits** | P0 | Medium | Finalize `RingKernelRuntime`, `PersistentHandle`, `ControlBlock` |
| D5 | **Migration guide (0.x → 1.0)** | P1 | Medium | Document all breaking changes and migration paths |
| D6 | **Deprecation protocol** | P1 | Small | Mark items for removal with `#[deprecated]` before 1.0 |
| D7 | **Feature flag audit** | P1 | Small | Ensure no feature flag combination produces unsound code |
| D8 | **`#[doc(hidden)]` audit** | P2 | Small | Verify internal-only items are not accidentally public |
| D9 | **MSRV policy** | P1 | Small | Decide on MSRV policy for 1.x (currently 1.75) |
| D10 | **Dependency minimum versions** | P2 | Medium | Test with `-Z minimal-versions` to verify lower bounds |

**Key principle**: Per the Cargo SemVer Compatibility Guide and Effective Rust (Item 21),
reaching 1.0 is a commitment that the public API is stable. The pre-1.0 period is the
time to make breaking changes freely.

### Phase E: Enterprise Deployment (Q4 2026 - Q1 2027)

**Goal**: Provide production deployment artifacts and operational infrastructure.

| # | Work Item | Priority | Effort | Description |
|---|-----------|----------|--------|-------------|
| E1 | **Docker images** | P0 | Medium | Multi-stage Dockerfiles: CUDA base + Rust builder + slim runtime |
| E2 | **NVIDIA Container Toolkit integration** | P0 | Small | Document and test GPU passthrough with `--gpus` flag |
| E3 | **Helm chart** | P1 | Medium | Kubernetes deployment with GPU node selector, resource limits |
| E4 | **Docker Compose profiles** | P1 | Small | Development (CPU), staging (GPU), production profiles |
| E5 | **GPU device scheduling** | P1 | Medium | Kubernetes `nvidia.com/gpu` resource requests; multi-GPU pod affinity |
| E6 | **Configuration management** | P1 | Medium | Environment-based config with validation; 12-factor compliance |
| E7 | **Secret management** | P1 | Small | Integration with Kubernetes secrets, Vault, or AWS Secrets Manager |
| E8 | **Horizontal scaling guide** | P2 | Medium | Document multi-node deployment with gRPC K2K bridge |
| E9 | **Disaster recovery playbook** | P2 | Medium | Checkpoint-based DR with RTO/RPO targets |
| E10 | **Compliance validation** | P2 | Large | End-to-end validation of SOC2/GDPR/HIPAA compliance features |

**Industry context**: PhoenixOS (SOSP '25) demonstrates OS-level concurrent GPU
checkpoint/restore for fault tolerance and process migration. NVIDIA's GTC 2025 session
on HPC GPU fault tolerance provides cluster-level strategies. These inform the checkpoint
and migration features already in RingKernel.

---

## Detailed Work Items

### A1: Unsafe Code Audit — Detailed Plan

The ~290 `unsafe` blocks fall into these categories:

| Category | Count (approx.) | Risk Level | Verification Strategy |
|----------|-----------------|------------|----------------------|
| CUDA driver FFI (`cudarc` calls) | ~100 | Medium | Wrapper review + integration tests |
| GPU memory mapping / pointer casts | ~40 | High | Miri (where possible) + Kani harnesses |
| Lock-free atomics / memory ordering | ~30 | Critical | Loom exhaustive testing |
| rkyv zero-copy archive access | ~20 | High | Validation-gated access patterns |
| Proc macro generated code | ~15 | Medium | Generated code review + expanded test matrix |
| SIMD intrinsics (CPU backend) | ~15 | Low | Miri + standard test coverage |
| Transmutes / pointer arithmetic | ~20 | High | Case-by-case Kani proofs |
| Other (FFI, alignment) | ~50 | Medium | Documentation + defensive wrappers |

**Deliverable**: Each `unsafe` block annotated with a `// SAFETY:` comment explaining
the invariants relied upon, per Rust API Guidelines.

### B1: OpenTelemetry Integration — Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   RingKernel Application                 │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ tracing  │  │ metrics  │  │ GPU Profiler Manager  │  │
│  │ spans    │  │ counters │  │ (NVTX, events)        │  │
│  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘  │
│       │              │                   │              │
│       ▼              ▼                   ▼              │
│  ┌─────────────────────────────────────────────────┐    │
│  │        tracing-opentelemetry bridge              │    │
│  │     + opentelemetry-otlp exporter                │    │
│  └────────────────────┬────────────────────────────┘    │
└───────────────────────┼─────────────────────────────────┘
                        │ OTLP/gRPC
                        ▼
              ┌─────────────────┐
              │  OTel Collector  │
              └────┬───────┬────┘
                   │       │
          ┌────────┘       └────────┐
          ▼                         ▼
   ┌──────────────┐         ┌──────────────┐
   │   Prometheus  │         │ Grafana Tempo │
   │   (metrics)   │         │  (traces)     │
   └──────────────┘         └──────────────┘
```

**Proposed metrics**:

| Metric | Type | Description |
|--------|------|-------------|
| `ringkernel_message_send_duration_seconds` | Histogram | H2K message send latency |
| `ringkernel_message_recv_duration_seconds` | Histogram | K2H message receive latency |
| `ringkernel_queue_depth` | Gauge | Current queue occupancy per kernel |
| `ringkernel_queue_capacity_ratio` | Gauge | Queue fill ratio (saturation signal) |
| `ringkernel_kernel_active_count` | Gauge | Number of active GPU kernels |
| `ringkernel_gpu_memory_allocated_bytes` | Gauge | GPU memory in use |
| `ringkernel_gpu_memory_pool_hit_ratio` | Gauge | Memory pool cache hit rate |
| `ringkernel_checkpoint_duration_seconds` | Histogram | Checkpoint operation latency |
| `ringkernel_k2k_messages_total` | Counter | Inter-kernel messages by route |
| `ringkernel_errors_total` | Counter | Errors by category and severity |

### E1: Docker Images — Multi-Stage Build

```dockerfile
# === Builder stage ===
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04 AS builder
RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
WORKDIR /build
COPY . .
RUN cargo build --release --features cuda

# === Runtime stage ===
FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/ringkernel-app /usr/local/bin/
EXPOSE 8080 4317
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8080/healthz || exit 1
ENTRYPOINT ["ringkernel-app"]
```

---

## Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | rkyv 0.7 has known soundness issues | Medium | High | Pin version, use `AlignedVec` everywhere, evaluate 0.8 migration |
| R2 | Lock-free queue has undiscovered race conditions | Low | Critical | Loom testing, linearizability verification |
| R3 | `cudarc` 0.18 API changes break builds | Medium | Medium | Pin version, wrap in abstraction layer |
| R4 | GPU driver bugs cause silent data corruption | Low | High | Checksums on critical data paths, ABFT techniques |
| R5 | wgpu in containers fails to find GPU adapter | Medium | Medium | Document Vulkan ICD requirements, provide fallback to CPU |
| R6 | 1.0 API proves too restrictive | Medium | Medium | Generous use of `#[non_exhaustive]`, extension traits |
| R7 | MSRV 1.75 becomes limiting | Low | Low | Evaluate bumping to 1.80+ for 1.0 |
| R8 | Performance regression goes undetected | Medium | Medium | Continuous benchmarking in CI with alerting |
| R9 | Enterprise features (encryption, audit) not validated by auditor | Medium | High | Engage third-party security audit pre-1.0 |
| R10 | Metal backend incomplete at 1.0 | High | Medium | Mark as experimental in 1.0; stable in 1.1 |

---

## Success Criteria for v1.0

### Hard Requirements (Must-Have)

- [ ] All `unsafe` blocks have `// SAFETY:` comments and are covered by Miri or Kani
- [ ] Loom tests pass for lock-free queue under all interleavings
- [ ] `cargo-audit` reports zero known vulnerabilities
- [ ] `cargo-semver-checks` integrated in CI; public API frozen
- [ ] OpenTelemetry traces and metrics exportable to standard backends
- [ ] Health check endpoints (`/healthz`, `/readyz`) available
- [ ] Graceful shutdown with kernel drain and optional checkpoint
- [ ] Docker images published for CPU and CUDA variants
- [ ] Operations runbook covering top 10 failure modes
- [ ] 24-hour soak test passes without memory leaks or panics
- [ ] All clippy warnings resolved (including `pedantic` group)
- [ ] Documentation coverage >95% for public API items

### Soft Requirements (Should-Have)

- [ ] Helm chart for Kubernetes deployment with GPU scheduling
- [ ] Continuous benchmarking with regression alerting
- [ ] Third-party security audit completed
- [ ] Migration guide from 0.4.x to 1.0
- [ ] Metal backend at feature parity with CUDA (or marked experimental)
- [ ] Python bindings (`ringkernel-python`) stable and published to PyPI
- [ ] p99 latency < 1μs for H2K message send (CUDA backend)
- [ ] Example production deployment architecture document

### Aspirational (Nice-to-Have for 1.0, otherwise 1.1)

- [ ] WebAssembly/WebGPU deployment path documented
- [ ] GPU-native distributed consensus (Raft on GPU)
- [ ] Automatic GPU kernel hot-patching in production
- [ ] PhoenixOS-style concurrent checkpoint/restore integration
- [ ] CubeCL interoperability for kernel portability

---

## Milestone Timeline

```
2026 Q1          Q2              Q3              Q4          2027 Q1
  │               │               │               │               │
  ├─ Phase A ─────┤               │               │               │
  │ Safety &      ├─ Phase B ─────┤               │               │
  │ Correctness   │ Operational   ├─ Phase C ─────┤               │
  │               │ Maturity      │ Performance   │               │
  │               │               ├─ Phase D ─────┤               │
  │               │               │ API Stability │               │
  │               │               │               ├─ Phase E ─────┤
  │               │               │               │ Enterprise    │
  │               │               │               │ Deployment    │
  │               │               │               │               │
  ▼               ▼               ▼               ▼               ▼
v0.5.0          v0.6.0          v0.8.0          v0.9.0-rc      v1.0.0
(safety)        (observability) (performance)   (API freeze)   (stable)
```

### v0.5.0 — Safety Milestone (Q1 2026)
- Unsafe audit complete with SAFETY.md
- Miri and Loom integrated in CI
- `cargo-audit` + `cargo-deny` in CI
- Clippy pedantic clean

### v0.6.0 — Observability Milestone (Q2 2026)
- OpenTelemetry integration
- Prometheus metrics endpoint
- Health check endpoints
- Graceful shutdown protocol

### v0.8.0 — Performance Milestone (Q3 2026)
- Continuous benchmarking in CI
- Soak test framework
- Latency percentile tracking
- Memory pool optimization

### v0.9.0-rc — API Freeze (Q4 2026)
- `cargo-semver-checks` enforced
- Public API reviewed and documented
- Migration guide from 0.4.x
- Feature flag audit complete

### v1.0.0 — Stable Release (Q1 2027)
- Docker images published
- Helm chart available
- Operations runbook complete
- All hard requirements met

---

## Research References

This roadmap was informed by the following research and industry resources:

### GPU Fault Tolerance & Checkpoint/Restore
- [PhoenixOS: Concurrent OS-level GPU Checkpoint and Restore](https://arxiv.org/html/2405.12079) — SOSP '25
- [FT K-Means: High-Performance K-Means on GPU with Fault Tolerance](https://arxiv.org/html/2408.01391v1) — Algorithm-based fault tolerance
- [Build a Customizable HPC Platform With Enhanced GPU Fault Tolerance](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72096/) — NVIDIA GTC 2025
- [Characterizing GPU Resilience and Impact on AI/HPC Systems](https://arxiv.org/html/2503.11901v1) — 2025

### Lock-Free GPU Queues
- [Boundary-Aware Concurrent Queue (BACQ)](https://www.mdpi.com/2076-3417/15/4/1834) — Applied Sciences, 2025
- [Toward Concurrent Lock-Free Queues on GPUs](https://www.researchgate.net/publication/275603769_Toward_Concurrent_Lock-Free_Queues_on_GPUs)
- [IPC: Shared Memory vs. Message Queues Performance Benchmarking](https://howtech.substack.com/p/ipc-mechanisms-shared-memory-vs-message) — 2025

### Rust Safety Verification
- [Rust Auditing Tools in 2025](https://markaicode.com/rust-auditing-tools-2025-automated-security-scanning/) — Miri, Kani, Rudra
- [Making Unsafe Rust a Little Safer](https://blog.colinbreck.com/making-unsafe-rust-a-little-safer-tools-for-verifying-unsafe-code/)
- [Surveying the Rust Verification Landscape](https://arxiv.org/html/2410.01981v1) — Comprehensive tool survey
- [Kani Rust Verifier: Safety-Critical Tool Submission](https://github.com/rustfoundation/safety-critical-rust-consortium/issues/380)
- [Verify the Safety of the Rust Standard Library](https://aws.amazon.com/blogs/opensource/verify-the-safety-of-the-rust-standard-library/) — AWS/Kani initiative
- [Sherlock Rust Security & Auditing Guide 2026](https://sherlock.xyz/post/rust-security-auditing-guide-2026)

### Rust Production Observability
- [Rust Real-World Observability](https://medium.com/@wedevare/rust-real-world-observability-health-checks-metrics-tracing-and-logs-fd229ea8ec96) — Health checks, metrics, tracing
- [Monitor Data Pipelines in Rust Using OpenTelemetry](https://www.shuttle.dev/blog/2025/09/23/monitor-data-pipelines-in-rust)
- [How to Monitor Rust Applications with OpenTelemetry](https://www.datadoghq.com/blog/monitor-rust-otel/) — Datadog
- [Implementing OpenTelemetry in Rust](https://signoz.io/blog/opentelemetry-rust/) — SigNoz
- [Production Readiness Checklist: 7 Key Steps for 2025](https://goreplay.org/blog/production-readiness-checklist-20250808133113/)

### Rust Actor Frameworks in Production
- [Ractor: Rust Actor Framework](https://github.com/slawlor/ractor) — Erlang/OTP-inspired, used at Meta
- [Comparing Rust Actor Libraries](https://tqwewe.com/blog/comparing-rust-actor-libraries/) — Actix, Coerce, Kameo, Ractor, Xtra
- [Quickwit Actor Framework](https://quickwit.io/blog/quickwit-actor-framework) — Custom framework with backpressure
- [Rust in Distributed Systems, 2025 Edition](https://disant.medium.com/rust-in-distributed-systems-2025-edition-175d95f825d6)

### API Stability & Semver
- [Cargo SemVer Compatibility Guide](https://doc.rust-lang.org/cargo/reference/semver.html) — Official reference
- [RFC 1105: API Evolution](https://rust-lang.github.io/rfcs/1105-api-evolution.html) — Foundational RFC
- [Effective Rust — Item 21: Semantic Versioning](https://effective-rust.com/semver.html)
- [cargo-semver-checks](https://crates.io/crates/cargo-semver-checks) — Automated compliance tool

### GPU Containerization & Deployment
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Docker GPU Support](https://docs.docker.com/desktop/gpu/)
- [Rust GPU: Running on Every GPU](https://rust-gpu.github.io/blog/2025/07/25/rust-on-every-gpu/) — Cross-platform GPU compute
- [CubeCL: Multi-Platform GPU Compute](https://github.com/tracel-ai/cubecl) — CUDA + wgpu from single codebase

### Zero-Copy Serialization Safety
- [rkyv Validation Documentation](https://rkyv.org/validation.html)
- [RUSTSEC-2021-0054: rkyv Archives May Contain Uninitialized Memory](https://rustsec.org/advisories/RUSTSEC-2021-0054.html)
- [rkyv FAQ: Alignment and Safety](https://rkyv.org/faq.html)

---

## Appendix: Priority Definitions

| Priority | Definition | Timeline |
|----------|-----------|----------|
| **P0** | Blocks production deployment; must resolve | Current phase |
| **P1** | High value; important for production confidence | Next phase |
| **P2** | Improves quality; can defer to post-1.0 | Backlog |

## Appendix: Effort Estimates

| Estimate | Duration | Description |
|----------|----------|-------------|
| **Small** | < 1 week | Configuration, docs, small tool integration |
| **Medium** | 1-4 weeks | Feature implementation, significant testing |
| **Large** | 1-3 months | Major subsystem work, cross-cutting concerns |
