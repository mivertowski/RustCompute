# RingKernel v1.1 — NC80adis_H100_v5 Session Guide

> **For the Claude Code instance running on the Azure NC80adis_H100_v5 VM (2× H100 NVL).**
> Read this before starting any work.

## Context

RingKernel v1.0.0 shipped (CUDA-focused, H100-verified, 8,698x vs traditional launch, 11 CI jobs green). Tag: `v1.0.0`, commit history on `main`.

**v1.1 adds multi-GPU + VynGraph NSAI integration.** All pre-hardware code is landed on `main`. Your job: validate it on 2× H100 NVL hardware and finalize features that need NVLink.

**Budget-conscious mandate:** The user approved NC80adis_H100_v5 (2× H100, ~$8/hr) over ND96 ($80k/month out of budget). Work efficiently — deallocate the VM when idle. Target session length: minimize GPU-hours.

## Primary Goal

**Formally prove the GPU-native persistent actor paradigm works across 2× H100 with NVLink**, producing:
1. Passing test matrix (8 items, spec §5.3)
2. TLC model checking reports for all 6 TLA+ specs
3. Benchmark numbers with statistical rigor (see `docs/benchmarks/METHODOLOGY.md`)
4. Honest gap documentation if anything doesn't hold

## Key Documents — Read First

1. **`CLAUDE.md`** — project architecture, build commands, gotchas
2. **`docs/superpowers/specs/2026-04-17-v1.1-vyngraph-gaps.md`** — v1.1 master spec (5 gaps, 3-phase migration, formal verification plan, 8-test hardware matrix in §5.3)
3. **`docs/benchmarks/METHODOLOGY.md`** — statistical protocol (CI, Cohen's d, MAD outlier detection)
4. **`docs/benchmarks/h100-b200-baseline.md`** — v1.0 single-GPU baseline (reference only)
5. **`docs/verification/README.md`** — TLA+ model checking instructions
6. **`docs/superpowers/GPU_VM_SESSION_GUIDE.md`** — prior H100 session guide (for context on what worked last time)

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
| Tests | **1,612 passing, 0 failures, 97 hardware-gated** | |

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

### Task 4 — Run the 8-Test Hardware Matrix

From spec §5.3:

```bash
# Each test: 3 trials for statistical rigor
cargo test -p ringkernel-cuda --features "cuda,cooperative,multi-gpu" --release \
  --test multi_gpu_migration_proof -- --ignored --test-threads=1

cargo test -p ringkernel-cuda --features "cuda,cooperative,multi-gpu" --release \
  --test multi_gpu_nvlink_k2k_proof -- --ignored --test-threads=1
# ... etc
```

**You may need to write some of these integration tests** — check `crates/ringkernel-cuda/tests/` for existing proof tests (actor_lifecycle_proof, streaming_pipeline_proof) as templates. Required tests per spec §5.3:

1. Migration 1M msgs — move actor with 1M in-flight — <100ms, zero loss
2. Migration loop — 100 back-and-forth migrations — no leak, checksum stable
3. NVLink K2K latency — cross-GPU latency — p99 < 5us
4. NVLink K2K throughput — >10M/s sustained 60s
5. Multi-tenant isolation — 4 tenants, 1000 cross-tenant attempts — 0 leaks, all audited
6. Provenance chain — 10-step NSAI chain — PROV-O attribution verified
7. Rule reload under load — swap at 100K msg/s — <1s quiescence, no loss
8. Full stress — all features + 60s sustained — all invariants hold

Document results in `docs/benchmarks/v1.1-2x-h100-results.md` with 95% CI.

### Task 5 — TLC Model Checking

```bash
cd docs/verification/

# Option A: Native TLC (install Java 17 + tla2tools.jar)
./tlc.sh

# Option B: Docker
docker run --rm -v $(pwd):/workspace pmer/tla \
  tlc /workspace/migration.tla -config /workspace/migration.cfg
```

Run each spec. Bounded state spaces are sized to complete in seconds/minutes. Document any invariant violations — those are real bugs to file.

Write a report: `docs/verification/v1.1-tlc-report.md` with:
- Spec name
- State space explored (distinct states, queue size)
- Invariants checked (all should be OK)
- Runtime
- Any counterexamples found

### Task 6 — Benchmarks (Paper-Quality)

Use the academic harness from v1.0:
```bash
./scripts/run-academic-benchmarks.sh
```

Then run multi-GPU-specific benchmarks (you may need to add them — model after existing criterion files in `crates/ringkernel/benches/`):
- Cross-GPU K2K latency vs single-GPU K2K
- Migration latency vs actor size
- Tenant isolation overhead (single vs 4 tenants)
- Provenance overhead (with/without)

Each: 100 samples × 10 trials, compute 95% CI, Cohen's d where comparing. Fill in `docs/benchmarks/v1.1-2x-h100-results.md`.

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

1. ✅ 8/8 hardware matrix tests pass with 95% CI documented
2. ✅ 6/6 TLC models pass with no counterexamples
3. ✅ No regression vs v1.0 single-GPU benchmarks
4. ✅ Cross-tenant leak count = 0 across all tests
5. ✅ CHANGELOG.md has v1.1.0 entry with concrete numbers
6. ✅ `cargo test --workspace` green on stable Rust 1.95+
7. ✅ `cargo clippy --workspace --lib --bins -- -D warnings` green
8. ✅ `cargo audit` green (or any new advisories justified in `.cargo/audit.toml`)
9. ✅ Tag `v1.1.0`, push, publish via `./scripts/publish.sh`

## Starting Prompt for the Other Claude

Paste this as the first message on the VM:

> Read `docs/superpowers/GPU_VM_SESSION_GUIDE_V1_1.md` and follow the task priority order. Start with Task 1 (VM setup), then Task 2 (enable real peer access), then Task 3 (wire real NVLink P2P transfers for migration). After that, run Task 4 (8-test matrix) and Task 5 (TLC models) in parallel if possible. All results must follow the statistical methodology in `docs/benchmarks/METHODOLOGY.md`. Be cost-conscious — this is NC80adis_H100_v5 at ~$8/hr. Deallocate the VM when idle.

Good luck. The v1.0 handover worked — aim for the same outcome.
