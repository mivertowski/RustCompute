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

## v1.1 -- Multi-GPU (Target: Q3 2026)

### Multi-GPU P2P Communication
- [ ] NVLink-aware actor placement (co-locate communicating actors on connected GPUs)
- [ ] P2P direct memory access between persistent actors on different GPUs
- [ ] NVSHMEM integration for symmetric heap across GPU cluster
- [ ] Multi-GPU actor migration with state transfer
- [ ] Load balancing across GPU pool with topology awareness

### GPU-Side Work Stealing
- [ ] Intra-kernel work stealing within persistent actors
- [ ] Dynamic task redistribution without host involvement
- [ ] Hierarchical stealing: intra-block -> intra-cluster -> cross-cluster

### State Checkpointing
- [ ] Periodic GPU actor state snapshots to host memory
- [ ] Incremental checkpoint (delta-only) for large actor state
- [ ] Checkpoint-based fault recovery with actor restart

---

## v1.2 -- Blackwell / B200 (Target: Q4 2026)

### Blackwell Architecture Support
- [ ] Cluster Launch Control for dynamic cluster sizing
- [ ] Enhanced DSMEM bandwidth (2x vs Hopper)
- [ ] FP4/FP6 precision for ML inference actors
- [ ] Confidential Computing (TEE) for secure persistent actors
- [ ] NVLink 5.0 inter-GPU actor communication

### Performance Targets
- [ ] Sub-50 ns command injection on B200
- [ ] 20+ Mmsg/s lock-free queue throughput
- [ ] Multi-GPU linear scaling benchmarks (2, 4, 8 GPUs)

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

| Metric | v1.0 (Achieved) | v1.1 Target | v1.2 Target |
|--------|-----------------|-------------|-------------|
| Command latency | 55 ns (H100) | 55 ns (multi-GPU) | <50 ns (B200) |
| Sustained throughput | 5.54 Mops/s | 20+ Mops/s (multi-GPU) | 50+ Mops/s |
| Test count | 1,496+ | 1,800+ | 2,000+ |
| GPU architectures | Hopper (H100) | Hopper + Ampere | Hopper + Blackwell |
| Multi-GPU support | Single GPU | 2-8 GPUs | 2-16 GPUs |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to the roadmap and implementation.

### Priority Definitions
- **P0**: Critical path, blocking other features
- **P1**: High value, should be in next release
- **P2**: Nice to have, can be deferred
