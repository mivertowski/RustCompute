# RingKernel Roadmap

> GPU-Native Persistent Actor Model Framework for Rust -- NVIDIA CUDA Focus

## Vision

Transform GPU computing from batch-oriented kernel launches to a true actor-based paradigm where GPU kernels are long-lived, stateful actors that communicate via high-performance message passing. RingKernel focuses exclusively on NVIDIA CUDA, leveraging Hopper and future architectures for maximum persistent actor performance.

---

## v1.0.0 -- Completed (April 2026)

**H100-verified persistent GPU actor framework.**

### Core Runtime
- [x] Persistent cooperative kernel execution (CUDA)
- [x] Lock-free SPSC/MPSC queues (truly lock-free, no mutexes)
- [x] GPU actor lifecycle (create/destroy/restart/supervise)
- [x] Named actor registry with wildcard service discovery
- [x] Credit-based backpressure and dead letter queue
- [x] GPU memory pressure handling (budgets, mitigation)
- [x] Dynamic scheduling (work stealing protocol)
- [x] Hybrid Logical Clocks for causal ordering (30 ns/tick)

### CUDA + Hopper Features
- [x] Thread Block Clusters with DSMEM messaging
- [x] cluster.sync() for intra-GPC synchronization (2.98x faster than grid.sync())
- [x] Green Contexts for SM partitioning
- [x] TMA (Tensor Memory Accelerator) integration
- [x] Async memory pool (116.9x faster than cuMemAlloc)
- [x] NVTX profiling, Chrome trace export, memory tracking
- [x] Cooperative groups with grid-wide synchronization

### Code Generation
- [x] Rust-to-CUDA transpiler (155+ intrinsics)
- [x] Global, stencil, ring, and persistent FDTD kernel modes
- [x] Unified IR (ringkernel-ir) with optimization passes (DCE, constant folding, algebraic simplification)

### Enterprise
- [x] Kernel checkpointing with state preservation
- [x] Hot reload with rollback
- [x] Graceful degradation (5 levels)
- [x] Health monitoring with liveness/readiness probes
- [x] Memory encryption (AES-256-GCM, ChaCha20)
- [x] Audit logging with tamper-evident chains
- [x] Kernel sandboxing with resource limits
- [x] Compliance reporting (SOC2, GDPR, HIPAA, PCI-DSS)

### Ecosystem
- [x] Actix, Axum, Tower, gRPC integrations with persistent GPU actors
- [x] SSE and WebSocket handlers for real-time GPU events
- [x] Arrow and Polars GPU operation support
- [x] CLI scaffolding, codegen, and compatibility checking

### Benchmarks (H100 NVL)
- [x] 8,698x faster than traditional cuLaunchKernel
- [x] 3,005x faster than CUDA Graph replay
- [x] 5.54M ops/s sustained throughput (CV 0.05%, 60 seconds)
- [x] 0.628 us cluster.sync() (2.98x vs grid.sync())
- [x] 0.544 ns zero-copy serialization
- [x] Paper-quality benchmarks with statistical analysis (95% CI, Cohen's d, Welch's t-test)

---

## v1.1 -- Multi-GPU + VynGraph NSAI -- Completed (April 2026)

**2x H100 NVL-verified. See `docs/benchmarks/v1.1-2x-h100-results.md`.**

### Multi-GPU P2P Communication
- [x] NVLink-aware actor placement (`PlacementHint::NvlinkPreferred` uses `NvlinkTopology::probe`)
- [x] P2P direct memory access via `cuCtxEnablePeerAccess` + `cuMemcpyPeerAsync`
- [x] Multi-GPU actor migration with 3-phase transfer (8.7x faster than host-stage at 16 MiB)
- [x] Load balancing across GPU pool (LoadBalance + CommunicationAware rebalance strategies)

### VynGraph NSAI Integration Points
- [x] PROV-O provenance header (8 relation kinds, chain walk, signature hook)
- [x] Multi-tenant K2K isolation (per-tenant sub-brokers, audit sink, quota enforcement)
- [x] Live introspection streaming (EWMA, drop-tolerant ring)
- [x] Hot rule reload (`CompiledRule`, version-monotonic, quiescence under load)

### Formal Verification
- [x] 6 TLA+ specs (hlc, k2k_delivery, migration, multi_gpu_k2k, tenant_isolation, actor_lifecycle)
- [x] TLC model-checking pipeline (no counterexamples)

### Added late in v1.1

- [x] **HBM-tier direct measurement** -- new `cluster_hbm_k2k` kernel, included as paper Exp 1 hbm tier
- [x] **Multi-GPU K2K sustained bandwidth** micro-bench (paper Addendum 6b): 258 GB/s @ 16 MiB (~81% of 318 GB/s peak)
- [x] **Incremental/delta checkpoints** -- `Checkpoint::delta_from` / `applied_with_delta` / `content_digest`
- [x] **Intra-block warp work stealing** -- `warp_work_steal` kernel + audit tests

---

## v1.2 -- Blackwell prep + hierarchical work stealing + NVSHMEM (in progress on `main`)

**v1.2 groundwork is on `main` but not yet tagged** -- see the `[Unreleased]` section of `CHANGELOG.md` for full detail.

### Landed

- [x] **Intra-cluster DSMEM work stealing** -- `cluster_dsmem_work_steal` CUDA kernel; blocks atomically share a DSMEM-hosted counter via `cluster.map_shared_rank`
- [x] **Cross-cluster HBM work stealing** -- `grid_hbm_work_steal` CUDA kernel; completes the block -> cluster -> grid stealing hierarchy
- [x] **NVSHMEM symmetric-heap bindings** -- opt-in `nvshmem` feature on `ringkernel-cuda`, `NvshmemHeap` RAII wrapper over the NVSHMEM host ABI; bootstrap (MPI / `nvshmrun` / unique-ID) is caller's responsibility
- [x] **Blackwell / sm_100 capability queries** -- `GpuArchitecture::supports_{cluster_launch_control,fp8,fp6,fp4,nvlink5,tee}`
- [x] **Post-Hopper codegen types** -- `ScalarType::BF16`, `FP8E4M3`, `FP8E5M2`, `FP6E3M2`, `FP6E2M3`, `FP4E2M1` with per-type `min_compute_capability()`; CUDA / MSL / WGSL lowerings wired
- [x] **Rubin preset** -- `GpuArchitecture::rubin()` placeholder (compute cap 12.x)

### Still open for v1.2 (hardware / integration blockers)

- [ ] **Blackwell runtime validation** -- requires B200 hardware; codegen stubs compile, runtime paths untested
- [ ] **4- / 8-GPU linear scaling benchmarks** -- bound by NC80adis having only 2 GPUs
- [ ] **NVSHMEM end-to-end smoke** -- requires dual-process bootstrap (mpirun or unique-ID); the wrapper handles post-bootstrap operations but the launch harness is still manual
- [ ] **Sub-50 ns command injection on B200** -- needs B200 silicon
- [ ] **20+ Mmsg/s lock-free queue** -- optimization path; current sustained is 5.10 Mops/s

---

## v1.3 -- Streaming and Integration (Target: Q1 2027)

### Streaming Integrations
- [ ] Kafka consumer/producer with GPU-resident processing
- [ ] NATS persistent actor subscriptions
- [ ] Redis Streams GPU bridge
- [ ] gRPC streaming with persistent actor backends

### AI/ML Integration
- [ ] LLM provider bridge (OpenAI, Anthropic, local models) with GPU-resident tokenization
- [ ] Vector store / GPU-resident embedding index
- [ ] Candle model inference as persistent actors
- [ ] PyTorch interop for training loop GPU actors

### Observability
- [ ] GPU profiler integration (Nsight Systems, Nsight Compute)
- [ ] Distributed tracing across multi-GPU actor systems
- [ ] Prometheus metrics exporter for persistent actor health
- [ ] Grafana dashboard templates

---

## Success Metrics

| Metric | v1.0 | v1.1 (Achieved) | v1.2 (groundwork on `main`) |
|--------|------|-----------------|-----------------------------|
| Command latency | 55 ns (H100) | 23 ns mean / 30 ns p99 on all 5 lifecycle rules | <50 ns (B200 target, needs silicon) |
| Sustained throughput | 5.54 Mops/s | 5.10 Mops/s (CV 0.66%, flat over 4x60 s) | 20+ Mops/s (queue opt path) |
| NVLink P2P migration | n/a | 8.7x vs host-stage @ 16 MiB | 4/8-GPU scaling (hardware-bound) |
| Multi-GPU K2K bandwidth | n/a | 258 GB/s @ 16 MiB (81% of NV12 peak) | NVSHMEM symmetric heap (bootstrap wired) |
| Cross-tenant leaks | n/a | 0 across 13 isolation tests | same baseline |
| TLA+ specs verified | n/a | 6 / 6 (no counterexamples) | same, plus capability query tests |
| Test count | 1,496+ | 1,590 (stable Rust 1.95) | 1,595 |
| GPU architectures | Hopper (H100) | Hopper multi-GPU (NV12) | Hopper + Blackwell codegen (FP4/FP6/FP8) |
| Multi-GPU support | Single GPU | 2-8 GPUs | 2-16 GPUs |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to the roadmap and implementation.

### Priority Definitions
- **P0**: Critical path, blocking other features
- **P1**: High value, should be in next release
- **P2**: Nice to have, can be deferred
