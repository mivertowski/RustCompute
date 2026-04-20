# RingKernel Documentation

GPU-native persistent actor framework for NVIDIA CUDA. This directory holds the reference documentation, benchmark data, formal specifications, and the academic paper.

## Top-level

| Path | Description |
|------|-------------|
| [`../CLAUDE.md`](../CLAUDE.md) | Architecture summary, build commands, release status, v1.2 groundwork. Start here. |
| [`../ROADMAP.md`](../ROADMAP.md) | Release phases: v1.0 shipped, v1.1 shipped, v1.2 groundwork on `main`, v1.3 target. |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Full release notes including the `[Unreleased]` v1.2 groundwork section. |
| [`../README.md`](../README.md) | User-facing overview with headline numbers and quick-start. |

## Reference documentation

Numbered guides written for end users and contributors. Stable across releases; updated when APIs change.

| Section | Topic |
|---------|-------|
| [01 — Architecture Overview](01-architecture-overview.md) | Host/device split, control blocks, lifecycle state machine, backend selection. |
| [02 — Crate Structure](02-crate-structure.md) | 19-crate workspace layout, dependency directions, facade re-exports. |
| [03 — Core Abstractions](03-core-abstractions.md) | `RingMessage`, `MessageQueue`, `HlcTimestamp`, `ControlBlock`, `RingContext`. |
| [04 — Memory Management](04-memory-management.md) | Pools, pinned buffers, pressure handling, queue tiering. |
| [05 — GPU Backends](05-gpu-backends.md) | CUDA and CPU backend traits, feature gating. |
| [06 — Serialization](06-serialization.md) | rkyv zero-copy, derive macro code paths. |
| [07 — Proc Macros](07-proc-macros.md) | `#[derive(RingMessage)]`, `#[ring_kernel]`, `#[derive(GpuType)]`. |
| [08 — Runtime](08-runtime.md) | `RingKernelRuntime`, bridge, async task model. |
| [09 — Testing](09-testing.md) | Unit, proptest, GPU-gated, and paper-aligned integration tests. |
| [10 — Performance](10-performance.md) | Measurement methodology, criterion benches, sustained-throughput protocol. |
| [11 — Ecosystem Integration](11-ecosystem.md) | Actix, Tower, Axum, gRPC, Arrow, Polars, Candle bridges. v1.1: PROV-O, multi-tenant K2K, hot rule reload. |
| [12 — Migration Guide](12-migration.md) | API-level migration notes across 0.x → 1.x. |
| [13 — CUDA Codegen](13-cuda-codegen.md) | Rust-to-CUDA transpiler (global / stencil / ring / persistent FDTD), 155+ intrinsics. |
| [15 — Showcase Applications](15-showcase-applications.md) | WaveSim, TxMon, AccNet, ProcInt. |
| [16 — CLI Tool](16-cli-tool.md) | `ringkernel-cli`: scaffolding, codegen, compat check. |
| [17 — Security](17-security.md) | Encryption, sandboxing, RBAC, TLS, rate limiting, multi-tenant isolation and audit. |
| [18 — ML Bridges](18-ml-bridges.md) | PyTorch, ONNX, Hugging Face pipelines (via ecosystem), Candle interop. |
| [19 — CUDA Wishlist / Persistent Actors](19-cuda-wishlist-persistent-actors.md) | Hardware asks for NVIDIA; context for the CUDA design decisions. |

## Benchmarks

Raw and summarized benchmark data. Numbers in the README are derived from these files.

| Path | Hardware | Contents |
|------|----------|----------|
| [benchmarks/METHODOLOGY.md](benchmarks/METHODOLOGY.md) | n/a | Statistical protocol: 95% CI, Welch's t-test, Cohen's d, locked-clock preflight. |
| [benchmarks/h100-b200-baseline.md](benchmarks/h100-b200-baseline.md) | 1× H100 NVL | v1.0 single-GPU baseline: command injection, queue throughput, sustained ops/s. |
| [benchmarks/ACADEMIC_PROOF.md](benchmarks/ACADEMIC_PROOF.md) | 1× H100 NVL | Empirical evidence with full statistical analysis; paper-quality. |
| [benchmarks/v1.1-2x-h100-results.md](benchmarks/v1.1-2x-h100-results.md) | 2× H100 NVL (NC80adis_H100_v5) | v1.1 multi-GPU run: tier latencies, snapshot/restart, lifecycle, 60 s sustained, NVLink migration, K2K sustained bandwidth. |

## Formal verification

TLA+ specifications of the persistent actor protocol and their TLC model-checking results.

| Path | Contents |
|------|----------|
| [verification/README.md](verification/README.md) | Overview of the six specifications and how to run them. |
| `verification/hlc.{tla,cfg}` | Hybrid Logical Clocks (physical monotonicity, per-node order, causal ordering). |
| `verification/k2k_delivery.{tla,cfg}` | Single-GPU K2K: FIFO-per-sender, bounded queue, no duplicates. |
| `verification/migration.{tla,cfg}` | 3-phase actor migration: atomic swap, FIFO preserved, `ChecksumMatch` after captured-snapshot fix. |
| `verification/multi_gpu_k2k.{tla,cfg}` | Cross-GPU K2K routing: NVLink adjacency, bounded latency. |
| `verification/tenant_isolation.{tla,cfg}` | Per-tenant K2K partitioning: cross-tenant sends always rejected. |
| `verification/actor_lifecycle.{tla,cfg}` | Actor state machine: supervisor termination, restart preserves snapshot. |
| [verification/v1.1-tlc-report.md](verification/v1.1-tlc-report.md) | TLC 2.19 run results: **6/6 specs OK, no counterexamples**. |
| `verification/tlc.sh` | Wrapper script for running a single spec or all of them. |

## Academic paper

*Persistent GPU Actors* — 13-section paper with appendix, ~48 pages built.

| Path | Contents |
|------|----------|
| [paper/main.tex](paper/main.tex) | Top-level LaTeX source. `make` in `docs/paper/` produces `main.pdf`. |
| `paper/sections/` | Per-section `.tex` files (abstract, introduction, model, k2k, memory, migration, supervision, tenancy, implementation, evaluation, discussion, related, conclusion, appendix). |
| `paper/references.bib` | Bibliography. |
| [paper/experiments/RUNBOOK.md](paper/experiments/RUNBOOK.md) | Reproducible 6-experiment pipeline (`run_all.sh`). |
| `paper/experiments/{01..06}` | Per-experiment `run.sh` + `extract.py`. Results archived to `paper/experiments/results/<timestamp>/`. |
| `paper/figures/` | Figure sources and figure-input data (`data/`). |

## Internal planning (`superpowers/`)

**Not user-facing.** Session handovers, production-readiness plans, and feature specs used by the maintainers and AI assistants. Listed here for transparency, not for external consumption.

| Path | Purpose |
|------|---------|
| `superpowers/specs/` | Feature specifications (v1.0.0 release plan, v1.1 VynGraph gaps, production-readiness design). |
| `superpowers/plans/` | Phased implementation plans (production-readiness phases 1–8). |
| `superpowers/GAP_ANALYSIS.md` | Recurring gap analysis against the roadmap. |
| `superpowers/GPU_VM_SESSION_GUIDE*.md` | Remote-GPU session runbooks (v1.0, v1.1, v1.2). |
| `superpowers/H100_SESSION_HANDOVER.md` | Per-session state carry-over for H100 runs. |

## Other artifacts

| Path | Purpose |
|------|---------|
| `cuda-wishlist-persistent-actors.{tex,pdf}` | Standalone write-up of CUDA hardware asks. |
| `ringkernel-executive-overview.{tex,pdf}` | Short executive overview (1-pager expanded). |
| `_edu.md` | Educational-mode notes for the WaveSim examples. |
| `roadmap/RINGKERNEL_GENERALIZATION_PROPOSAL.md` | Proposal for generalizing the runtime beyond CUDA; historical. |
| `screenshots/` | Static screenshots used in examples. |
