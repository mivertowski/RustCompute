# RingKernel v1.1 — NC80adis_H100_v5 Session Guide

> **For the Claude Code instance running on the Azure NC80adis_H100_v5 VM (2× H100 NVL).**
> Read this before starting any work.

## Context

RingKernel v1.0.0 shipped (CUDA-focused, H100-verified, 8,698x vs traditional launch, 11 CI jobs green). Tag: `v1.0.0`, commit history on `main`.

**v1.1 adds multi-GPU + VynGraph NSAI integration.** All pre-hardware code is landed on `main`. Your job: validate it on 2× H100 NVL hardware and finalize features that need NVLink.

**Budget-conscious mandate:** The user approved NC80adis_H100_v5 (2× H100, ~$8/hr) over ND96 ($80k/month out of budget). Work efficiently — deallocate the VM when idle. Target session length: minimize GPU-hours.

**Total budget for full hardware run: ~4 hours of GPU time** (3h experiments + 1h buffer). The paper team has pre-written the experiment pipeline — see `docs/paper/experiments/RUNBOOK.md`.

## Primary Goal

**Formally prove the GPU-native persistent actor paradigm works across 2× H100 with NVLink**, producing:
1. Passing test matrix (8 items, spec §5.3)
2. TLC model checking reports for all 6 TLA+ specs
3. Benchmark numbers with statistical rigor (see `docs/benchmarks/METHODOLOGY.md`)
4. Honest gap documentation if anything doesn't hold

## Key Documents — Read First

1. **`CLAUDE.md`** — project architecture, build commands, gotchas
2. **`docs/superpowers/specs/2026-04-17-v1.1-vyngraph-gaps.md`** — v1.1 master spec (5 gaps, 3-phase migration, formal verification plan, 8-test hardware matrix in §5.3)
3. **`docs/paper/experiments/RUNBOOK.md`** — pre-written 6-experiment pipeline by the paper team (CSV → pgfplots figures) — **primary runbook for Task 4/6**
4. **`docs/paper/main.tex` + `docs/paper/sections/`** — academic paper being validated by these runs ("Persistent GPU Actors" — 13 sections)
5. **`docs/benchmarks/METHODOLOGY.md`** — statistical protocol (CI, Cohen's d, MAD outlier detection)
6. **`docs/benchmarks/h100-b200-baseline.md`** — v1.0 single-GPU baseline (reference only)
7. **`docs/verification/README.md`** — TLA+ model checking instructions (Experiment 5)
8. **`docs/superpowers/GPU_VM_SESSION_GUIDE.md`** — prior H100 session guide (context on what worked last time)

## What's Already Done on `main`

| Component | Status | Key commit |
|-----------|--------|------------|
| Core runtime (H2K/K2H mapped memory, persistent kernels) | v1.0 proven | earlier |
| PROV-O provenance (8 relations, envelope opt-in) | Done | `f6f1d6b` |
| NVLink topology detection (probe, bandwidth, paths) | Done | `38e5cbd` |
| Multi-tenant K2K isolation (per-tenant sub-brokers) | Done | `e33ffd3` |
| Live introspection streaming (IntrospectionStream, EWMA) | Done | `fe1535e` |
| Hot rule reload (CompiledRule artifact API) | Done | `a1226a5` |
| Multi-GPU runtime facade + migration orchestrator | Done (simulated) | `84f0b61` |
| GPU-side tenant enforcement + migration kernels | Done | `febb724` |
| TLA+ specs (6 models: hlc, k2k, migration, multi_gpu, tenants, lifecycle) | Done | `fd95dd8` |
| Academic paper draft (13 sections + appendix) | Done | `019b0d2`..`a92f42c` |
| Paper experiment pipeline (6 experiments + run_all.sh) | Done | `620d9dc` |
| Paper-aligned integration tests (tier_latency, lifecycle, snapshot_restart, nvlink_migration) | Done | `620d9dc` |
| Tests | **1,612+ passing, 0 failures, paper experiments hardware-gated** | |

## Hardware-Phase Work (Your Job)

The pre-hardware code uses **bookkeeping and simulation** where it needs GPU operations. Your task: replace simulated paths with real CUDA calls, then validate.

### Task 1 — VM Setup & Sanity Check

```bash
# After SSH login:
cd ~
git clone https://github.com/mivertowski/RustCompute.git RingKernel
cd RingKernel

# Setup (Rust, CUDA, etc.)
./scripts/setup-gpu-vm.sh

# Verify NVLink is exposed
nvidia-smi topo -m      # Expect "NV18" (or similar NVX) between GPU 0 and GPU 1
nvidia-smi nvlink --status  # Expect both links "Active"

# Lock clocks for consistent benchmarks
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)
sudo nvidia-smi -c EXCLUSIVE_PROCESS

# Build with multi-GPU feature
export RINGKERNEL_CUDA_ARCH=sm_90
cargo build --workspace --features "cuda,cooperative,multi-gpu" --release

# Run all non-hardware tests (should match baseline: 1,612 pass)
cargo test --workspace --release
```

**If NVLink is NOT exposed on NC80adis:** That's fine for correctness testing — fallback is PCIe Gen5. Note it in the test report; migration speed will be lower but all correctness properties still hold.

### Task 2 — Enable Peer Access (Real CUDA P2P)

File: `crates/ringkernel-cuda/src/multi_gpu/runtime.rs`

Currently `enable_peer_access` is bookkeeping-only:
```rust
pub fn enable_peer_access(&self, from: u32, to: u32) -> Result<()> {
    // Currently: validate topology + insert into HashSet
    // TODO (hardware phase): call cuCtxEnablePeerAccess
}
```

**Wire the real call:**
```rust
use cudarc::driver::sys;

pub fn enable_peer_access(&self, from: u32, to: u32) -> Result<()> {
    self.validate_pair(from, to)?;

    // SAFETY: cuCtxEnablePeerAccess requires contexts on both devices
    unsafe {
        let target_ctx = self.devices[to as usize].context();
        let result = sys::cuCtxEnablePeerAccess(target_ctx, 0);
        // Handle CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED gracefully
    }

    self.peer_access.write().insert((from, to));
    Ok(())
}
```

Add `disable_peer_access` mirror using `cuCtxDisablePeerAccess`. Run the 3 `#[ignore]` multi_gpu runtime tests.

### Task 3 — Real NVLink P2P Transfers for Migration

File: `crates/ringkernel-cuda/src/multi_gpu/migration.rs` (Phase 2 "Transfer")

Currently simulated on host. Replace with `cuMemcpyPeer` (or `cudaMemcpyPeerAsync` via stream) once peer access is enabled.

The staging buffer CRC32 should match pre- and post-transfer — the existing test asserts this on simulated path; verify it holds with real P2P.

### Task 4 — Run the Paper Experiment Suite

The paper team has written a full 6-experiment pipeline. **Use it first** — it produces paper figures + reproducibility manifest + benchmark CSVs all at once.

```bash
cd docs/paper/experiments
./run_all.sh                    # runs all 6 in recommended order
# OR selectively:
./run_all.sh --skip=4,6         # skip long + 2-GPU experiments
```

The 6 experiments:

| # | Experiment | Wall-clock | GPUs | Paper § |
|---|------------|-----------|------|---------|
| 1 | K2K tier latency (SMEM/DSMEM/HBM × payload) | ~25 min | 1 | §6.1, §10 |
| 2 | Snapshot/restart | ~15 min | 1 | §10.5 |
| 3 | Lifecycle overhead (create/destroy/restart) | ~15 min | 1 | §4 |
| 4 | Sustained timeseries (60s) | ~70 min | 1 | §10.2 |
| 5 | TLC state-space stats | ~20 min | 0 | §4.4, §6.7 |
| 6 | NVLink P2P migration | ~30 min | 2 | §9.3 |

Corresponding Rust tests:
- `crates/ringkernel-cuda/tests/paper_tier_latency.rs`
- `crates/ringkernel-cuda/tests/paper_snapshot_restart.rs`
- `crates/ringkernel-cuda/tests/paper_lifecycle_overhead.rs`
- `crates/ringkernel-cuda/tests/paper_nvlink_migration.rs`

Results land in `docs/paper/experiments/results/<timestamp>/` with:
- `manifest.json` — commit, driver, CUDA, Rust, GPU info
- `nvidia_smi_full.txt` — full GPU snapshot
- `ecc_pre.csv` / `ecc_post.csv` — ECC error counts (non-zero = invalid trial)
- `<exp>/raw.log` + `<exp>/*.csv` — per-experiment outputs
- `docs/paper/figures/data/*.csv` — figure templates consume these

### Task 4b — Additional v1.1 Gap Validation Tests

Beyond the paper experiments, validate the 4 VynGraph gaps that don't have paper equivalents:

```bash
# Multi-tenant isolation — 4 tenants, 1000 cross-tenant attempts
cargo test -p ringkernel-core --release k2k::tests::multi_tenant -- --nocapture

# Provenance chain — 10-step NSAI attribution
cargo test -p ringkernel-core --release provenance -- --nocapture

# Rule reload under load — swap at 100K msg/s, <1s quiescence
cargo test -p ringkernel-core --release rules::registry::tests -- --nocapture

# Full stress (all features + 60s sustained) — if not covered by Exp 4
# Write a custom integration test if needed; model after paper_snapshot_restart.rs
```

All pass criteria (correctness) can be validated on CPU / single-GPU paths. Only multi-GPU migration and NVLink K2K **require** 2-GPU hardware.

Document results in `docs/benchmarks/v1.1-2x-h100-results.md` with 95% CI. Cross-reference paper's §10 evaluation section.

### Task 5 — TLC Model Checking (Paper Experiment 5)

The paper pipeline runs this too (no GPUs needed, can run in parallel with pre-flight build):

```bash
cd docs/paper/experiments/05-tlc-stats
./run.sh ./out
# Outputs: out/tlc-stats.csv + out/raw/<spec>.log per model
```

Or manually:

```bash
cd docs/verification/
./tlc.sh       # native TLC (needs Java 17 + tla2tools.jar)
# OR use Docker: docker run --rm -v $(pwd):/workspace pmer/tla ...
```

All 6 models should terminate in seconds-to-minutes with no counterexamples. If any model produces a counterexample, **STOP** and treat it as a real bug — do not ship v1.1.

Expected state space sizes (from spec bounds):
- `hlc.tla` — ~10³ states
- `k2k_delivery.tla` — ~10⁴ states
- `migration.tla` — ~10⁵ states (largest)
- `multi_gpu_k2k.tla` — ~10⁴ states
- `tenant_isolation.tla` — ~10³ states
- `actor_lifecycle.tla` — ~10⁴ states

Write report at `docs/verification/v1.1-tlc-report.md`:
- Spec name, distinct states explored, runtime, invariants checked, counterexamples (expect 0)

### Task 6 — Paper Figure Generation (After Experiments Land)

The paper team's `run_all.sh` writes CSVs to `docs/paper/figures/data/`. Build figure PDFs:

```bash
cd docs/paper/figures
pdflatex tier-latency.tex        # → tier-latency.pdf
pdflatex snapshot-restart.tex
pdflatex lifecycle-cost.tex
pdflatex sustained-cv.tex
pdflatex migration-cost.tex
# tlc-stats.tex is a table, included directly in main.tex
```

Then rebuild the paper:

```bash
cd docs/paper
make                             # → main.pdf
```

Review `main.pdf` — all figure references should resolve, no "??" citations.

### Task 6b — Extra Benchmarks (If Time Remains)

Not covered by paper but nice for CHANGELOG:
- Tenant isolation overhead (single vs 4 tenants) — broker throughput comparison
- Provenance overhead (with/without) — message round-trip with and without ProvenanceHeader

Each: 100 samples × 10 trials, compute 95% CI, Cohen's d. Append to `docs/benchmarks/v1.1-2x-h100-results.md`.

### Task 7 — Update CHANGELOG and ROADMAP

```markdown
## [1.1.0] - 2026-04-??

### Headline Results (2× H100 NVL via NVLink)

- Multi-GPU actor migration: <X>ms for <Y>M messages (zero loss)
- NVLink K2K: <X>us p99 latency (<Y>x vs host-mediated)
- Tenant isolation: 0 cross-tenant leaks in <N> attempts
- Formal properties proven: <list>

### Added / Changed / Removed
...
```

## Decision Points (Escalate to User If Unclear)

1. **NVLink absent on NC80adis** — proceed with PCIe fallback, note degradation in report
2. **TLC finds a counterexample** — STOP; file as bug, attach trace
3. **Migration kernels fail on sm_90 but work on sm_75** — investigate, likely cooperative groups edge case
4. **Benchmark shows regression vs v1.0** — don't ship; bisect the commit
5. **Cross-tenant leak detected** — CRITICAL — STOP the release

## Memory & Safety Notes

- `CudaContext` is not `Send` by default; multi-GPU runtime uses `Arc<CudaRuntime>` per device
- P2P direct memory access requires both contexts to have enabled peer access (symmetric)
- `cuMemcpyPeer` is asynchronous; await via stream sync before marking transfer complete
- `K2KRouteEntry` is now 72 bytes (was 64) — anything that size_of::<> compares needs updating

## Gotchas

- `rust-toolchain.toml` is set to `stable` (was `nightly` pre-v1.0) — if a crate needs nightly, add opt-in feature
- `clippy::unwrap_used` is `-D warnings` on lib/bins; tests allow it
- Local Cargo.toml has `.cargo/audit.toml` with documented ignores — keep them, don't silently widen
- `cargo-audit` CI runs; security regressions will block merge
- `rustsec/audit-check@v2` was replaced with direct `cargo audit` due to permissions issue
- `SIMD` feature is opt-in now (needs nightly) — don't re-enable by default

## Commit Convention

Same as v1.0:
```
feat(area): description
fix(area): description
docs(area): description
test(area): description

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

Breaking changes: `feat!:` prefix.

## Cost-Efficient Workflow

```bash
# Start session
az vm start --resource-group ringkernel-gpu --name ringkernel-h100v2

# Work, commit, push frequently (you can iterate offline between sessions)
# ...

# End session (stops compute billing, keeps disk for next time)
az vm deallocate --resource-group ringkernel-gpu --name ringkernel-h100v2

# Final cleanup (delete VM, disk, everything)
az group delete --name ringkernel-gpu --yes
```

**Per-hour cost discipline:** Build and unit tests are <2 min. Reserve GPU time for actual hardware validation runs. Between runs, commit + deallocate.

## Success Criteria

The user will consider v1.1 shippable when:

1. ✅ 6/6 paper experiments complete with CSV outputs in `docs/paper/figures/data/`
2. ✅ 6/6 TLC models pass with no counterexamples (Experiment 5 output)
3. ✅ No regression vs v1.0 single-GPU benchmarks (Experiment 1/4 vs v1.0 baseline)
4. ✅ Cross-tenant leak count = 0 across all tests
5. ✅ Paper figures built (`make` in `docs/paper/`) — `main.pdf` produces
6. ✅ CHANGELOG.md has v1.1.0 entry with concrete numbers from experiments
7. ✅ `cargo test --workspace` green on stable Rust 1.95+
8. ✅ `cargo clippy --workspace --lib --bins -- -D warnings` green
9. ✅ `cargo audit` green (or any new advisories justified in `.cargo/audit.toml`)
10. ✅ Tag `v1.1.0`, push, publish via `./scripts/publish.sh`

## Starting Prompt for the Other Claude

Paste this as the first message on the VM:

> Read `docs/superpowers/GPU_VM_SESSION_GUIDE_V1_1.md` and `docs/paper/experiments/RUNBOOK.md` first. Follow the task priority order:
>
> 1. **Task 1** — VM setup & sanity check (`./scripts/setup-gpu-vm.sh`, verify NVLink)
> 2. **Task 2** — Wire real `cuCtxEnablePeerAccess` in `crates/ringkernel-cuda/src/multi_gpu/runtime.rs`
> 3. **Task 3** — Wire real `cuMemcpyPeer` in `crates/ringkernel-cuda/src/multi_gpu/migration.rs`
> 4. **Task 5** first — TLC model checking (CPU, can run during the build)
> 5. **Task 4** — `cd docs/paper/experiments && ./run_all.sh` (6 experiments, ~3h total)
> 6. **Task 4b** — v1.1 gap validation tests (tenant, provenance, rules)
> 7. **Task 6** — Rebuild paper figures & PDF
> 8. **Task 7** — CHANGELOG + ROADMAP update, tag v1.1.0, publish
>
> All results must follow the statistical methodology in `docs/benchmarks/METHODOLOGY.md`. Experiment outputs land in `docs/paper/experiments/results/<timestamp>/`.
>
> **Budget: ~4 hours of GPU time total.** NC80adis_H100_v5 is $8/hr. Deallocate between runs (`az vm deallocate --resource-group ringkernel-gpu --name ringkernel-h100v2`).
>
> **STOP conditions** (see guide § "Decision Points"): cross-tenant leak, TLC counterexample, benchmark regression vs v1.0. Do not ship v1.1 until all 9 success criteria pass.

Good luck. The v1.0 handover worked — aim for the same outcome.
