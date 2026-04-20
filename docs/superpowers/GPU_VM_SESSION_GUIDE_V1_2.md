# RingKernel v1.2 — Next Session Guide

> **For the Claude Code instance running on the next Azure GPU VM.**
> Read this before starting any work.

## Current state (as of 2026-04-20)

v1.1.0 validated on 2× H100 NVL (NC80adis_H100_v5) — all 9 success
criteria green. v1.2 groundwork landed on `main` but **not tagged**.
See `CHANGELOG.md` `[Unreleased]` section for the full v1.2 delta.

| Artifact | Where |
|----------|-------|
| v1.1 benchmarks | `docs/benchmarks/v1.1-2x-h100-results.md` |
| v1.1 TLC report | `docs/verification/v1.1-tlc-report.md` |
| Paper PDF | `docs/paper/main.pdf` (48 pages) |
| Experiment pipeline | `docs/paper/experiments/run_all.sh` |
| v1.2 CHANGELOG draft | `CHANGELOG.md` `[Unreleased]` |
| v1.2 progress | `ROADMAP.md` § "v1.2 -- Blackwell prep..." |

## What's already done for v1.2

Implemented and tested on 2× H100 NVL:

1. **Intra-cluster DSMEM work stealing** (`cluster_dsmem_work_steal`)
2. **Cross-cluster HBM work stealing** (`grid_hbm_work_steal`) — `tests/hierarchical_work_steal.rs`
3. **NVSHMEM symmetric-heap bindings** — `ringkernel-cuda::multi_gpu::nvshmem`, opt-in `nvshmem` feature
4. **Blackwell codegen stubs** — `GpuArchitecture::{blackwell, rubin}`, `supports_{fp4, fp6, fp8, nvlink5, tee, cluster_launch_control}`, `ScalarType::{BF16, FP8E4M3, FP8E5M2, FP6E3M2, FP6E2M3, FP4E2M1}`

Regression: 1,595 tests pass, 0 failures.

## What still needs hardware / external systems

These items are **infrastructure-bound** and cannot be finished on
the 2× H100 NVL SKU:

1. **Blackwell runtime validation** — requires B200 silicon. Codegen
   emits `__nv_fp4_*` / `__nv_fp6_*` type names which CUDA 12.9+
   resolves; older toolkits emit unresolved identifiers.
2. **4-GPU / 8-GPU linear scaling benchmarks** — NC80adis has 2
   GPUs. Next tier is ND96 ($80k/month — user has ruled out).
   Alternative: A10 instances (cheaper, less relevant for hero data),
   or wait for NVLink-enabled 4-GPU SKUs at consumer price points.
3. **NVSHMEM end-to-end smoke** — requires dual-process bootstrap
   (`mpirun -np 2` or unique-ID flow). The wrapper handles
   post-bootstrap calls but the launch harness is still manual. A
   3-process smoke test via `nvshmrun` on a 2-GPU VM would close
   this (~30 min once MPI / NVSHMEM are both installed).
4. **Sub-50 ns command injection target** — needs B200 silicon.
5. **20+ Mmsg/s lock-free queue** — optimization path. Current
   sustained on H100 NVL: 5.10 Mops/s. Requires queue algorithm
   changes (possibly LL-SC intrinsics, batched dequeue, or
   prefetch-heavy layout).

## Primary goals for a v1.2 session

When hardware is available, in priority order:

### P0 — Tag v1.1.0

The user held off tagging v1.1.0 specifically so the v1.2 groundwork
could land first. Before v1.2 work starts, confirm with the user
whether to:
 (a) tag v1.1.0 from the v1.1.0 commit (revert-less history), or
 (b) tag v1.2.0 directly and fold the v1.2 groundwork into the
     v1.1.0 CHANGELOG for a combined release.

Either path needs:
- `./scripts/publish.sh --status` to confirm what's already on crates.io
- `git tag v1.1.0 <sha>` or `git tag v1.2.0 HEAD`
- `./scripts/publish.sh <TOKEN>` to push the release

### P1 — NVSHMEM end-to-end smoke

```bash
# On the VM with libnvshmem3-dev-cuda-12:
sudo apt-get install -y libmpi-dev openmpi-bin   # or mpich
mpirun -np 2 cargo test -p ringkernel-cuda \
    --features "cuda,multi-gpu,nvshmem" --release \
    --lib multi_gpu::nvshmem::tests::attach_and_query_pe \
    -- --ignored --nocapture
```

Expected: both PEs attach, print `my_pe` / `n_pes`, barrier, malloc
1 KiB, free. If this passes, NVSHMEM integration is validated.

### P2 — B200 runtime validation (when silicon arrives)

The codegen path already emits `sm_100` PTX via the multi-arch
fallback. Priority checks on B200:

1. `paper_tier_latency` with `TIERS` extended to include FP6/FP4
   tensor ops (new kernel needed)
2. Cluster Launch Control test — dynamic cluster-size changes
3. NVLink 5 bandwidth saturation (expect ~900 GB/s vs 318 GB/s on
   H100 NVL)
4. TEE / confidential computing smoke test for persistent actors

### P3 — Lock-free queue optimization

The 20+ Mmsg/s target requires a focused optimization pass. Likely
wins (validate on H100 first before B200):

- Batched dequeue (16-message batches)
- Prefetch-ahead in the producer
- LL-SC / WFE intrinsics on the consumer
- Cache-line-padded queue head/tail to avoid false sharing

## Decision points to escalate

1. **Tag first or fold into v1.2.0?** — user preference.
2. **B200 hardware acquisition** — who pays and when?
3. **Install MPI for NVSHMEM smoke?** — small cost but modifies VM
   image; user should confirm.
4. **v1.3 streaming integrations** — `ringkernel-ecosystem` already
   has Actix/Axum/Tower/gRPC; Kafka / NATS / Redis are additive.
   Priority vs B200 work?

## Cost-efficient workflow (same as v1.1)

```bash
# Start
az vm start --resource-group ringkernel-gpu --name ringkernel-h100v2

# Work, commit, push
# ...

# Stop when idle
az vm deallocate --resource-group ringkernel-gpu --name ringkernel-h100v2

# Final cleanup
az group delete --name ringkernel-gpu --yes
```

## Starting prompt for the next session

> Read `docs/superpowers/GPU_VM_SESSION_GUIDE_V1_2.md` first. Confirm
> with the user whether to (a) tag v1.1.0 from its original commit or
> (b) fold v1.2 groundwork into a v1.2.0 release, then proceed with
> the priority list in that guide. Target: close out NVSHMEM smoke
> and begin B200 prep if hardware is available.
