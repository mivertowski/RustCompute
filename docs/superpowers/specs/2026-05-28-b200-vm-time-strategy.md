# B200 VM Time Strategy

> **Date:** 2026-05-28
> **Status:** Approved (user delegated remaining decisions autonomously)
> **Next:** Implementation plan via `superpowers:writing-plans`, then execution via `superpowers:subagent-driven-development`.

## Purpose

Plan how to spend a generous Lambda Cloud B200 budget (~12h on `gpu_1x_b200` + ~4h on `gpu_8x_b200`, ~$150–200) to:

1. Validate the v1.2 Blackwell groundwork on real silicon (P0–P5 from `docs/superpowers/GPU_VM_SESSION_GUIDE_B200.md`).
2. Collect paper-grade B200 numbers for Addendum 7 of `docs/paper/`.
3. Reach ship-readiness for a single combined `v1.2.0` tag that claims *both* v1.2 groundwork *and* Blackwell validation.

The user picked "Generous budget", "Both paper + ship-ready", and "Hold v1.2.0 until Blackwell is green" in brainstorming. The user delegated subsequent decisions and asked for subagent-driven implementation mode.

## Strategy choice

**Approach B (hybrid prep + sequential phased VM)** was selected over Approach A (heavy offline prep) and Approach C (minimal prep) because:

- Generous budget means VM-minute optimization is not the binding constraint, so heavy offline prep (A) over-invests in the wrong place.
- "Both goals" requires tests to be *correct*, not just *present*, so improvisation (C) is too risky for FP4/FP6/FP8 codegen (the area with zero silicon time so far).
- B puts dev-time on harnesses/scripts/docs (cheap here) and VM-time on real `nvcc -arch=sm_100` iteration (irreplaceable on silicon).

## Architecture: four phases

```
Phase 0 — OFFLINE PREP             (here, ~2h)
   pushes branch `b200-validation` with harness + skeletons
       ↓
Phase 1 — 1× B200 ON LAMBDA        (~10h, ~$35)
   P0 environment validation, P1 cluster launch control,
   P2 FP4/FP6/FP8 kernel smoke, P4 sub-50ns inject, P5 TEE-I/O
   STOP instance after Phase 1; persistent FS retains state.
       ↓
       [HARD GATE — user confirms before booking 8×]
       ↓
Phase 2 — 8× B200 ON LAMBDA        (~3h, ~$90)
   topology check, P3 NVLink 5 peer bandwidth, P4 multi-GPU repeat
       ↓
Phase 3 — OFFLINE WRAP             (here, ~2h)
   CHANGELOG [1.2.0], Addendum 7, README compat matrix, tag v1.2.0
```

Phases are explicitly *sequential*. Phase 1 must demonstrate that v1.2 stubs run on a single B200 before any 8× billing starts. Stopping the 1× instance between phases lets the user review Phase 1 results without burn rate.

## Components

### Phase 0 deliverables (offline, here)

| File | Purpose |
|---|---|
| `crates/ringkernel-cuda/tests/blackwell_cluster_launch.rs` | Rust harness for `cuLaunchKernelEx` + dynamic cluster dims. CUDA kernel body filled in on VM. |
| `crates/ringkernel-cuda/tests/blackwell_fp_scalar_smoke.rs` | Harness for FP4 / FP6 / FP8 roundtrip; kernel bodies drafted on VM against real nvcc errors. |
| `crates/ringkernel-cuda/tests/blackwell_tee_io_probe.rs` | CC (Confidential Computing) mode probe; may be a binary if a test crate can't run under CC. |
| `scripts/b200-collect-paper-numbers.sh` | Wraps Criterion runs with locked-clocks + exclusive mode; emits CSV + summary. |
| `scripts/b200-multi-gpu-runner.sh` | Phase 2 wrapper: topology check, `paper_multi_gpu_k2k_bw` sweep, P4 repeat. |
| `analysis/b200_h100_compare.py` | Generates comparison plots and markdown tables for Addendum 7. |
| `docs/paper/sections/Addendum_07_Blackwell.md` | Skeleton with `<TBD>` placeholders for each P-level result. |
| _(CHANGELOG.md untouched in Phase 0)_ | Stays at `[Unreleased]` until Phase 3, so an aborted trip doesn't strand a half-moved entry. |

### Phase 1 + 2 deliverables (on VM)

| Artifact | Notes |
|---|---|
| Filled-in `blackwell_cluster_launch.rs` kernel body | `cuLaunchKernelEx` with `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`, reads back a sentinel value. |
| Filled-in `blackwell_fp_scalar_smoke.rs` kernel bodies | One kernel per scalar (FP4E2M1, FP6E3M2, FP6E2M3, FP8E4M3, FP8E5M2); reference values verified vs CUDA reference. |
| `logs/b200-validate-*.log` | Every `scripts/b200-validate.sh` run preserved. |
| `benchmark_results/b200/*` | Per-experiment Criterion outputs + `system_info.json` + `nvidia_smi_full.txt`. |
| Updated `docs/benchmarks/h100-b200-baseline.md` | New "B200" columns alongside the H100 numbers. |

### Phase 3 deliverables (offline, here)

| Artifact | Notes |
|---|---|
| `CHANGELOG.md` `[1.2.0]` entry | First time the v1.2 groundwork moves out of `[Unreleased]`. Dated. Adds Blackwell-validation lines. |
| `docs/paper/sections/Addendum_07_Blackwell.md` | Filled in with measured numbers, 95% CIs, comparison vs H100. |
| `README.md` GPU compat matrix | "Blackwell-runtime-validated" stamp on sm_100. |
| Git tag `v1.2.0` | Local tag created; **not** pushed without explicit user approval. |

## Data flow

Three artifact streams cross the offline ↔ VM boundary:

1. **Code** — dev → `git push origin b200-validation` → VM `git pull` → fill in → VM `git push` → dev `git pull` → polish → `git tag`.
2. **Logs** — VM writes to `logs/b200-validate-*.log`; committed to the repo for reproducibility.
3. **Bench data** — VM writes to `target/criterion/` + `benchmark_results/b200/`. Criterion `estimates.json` files are committed (small, paper-relevant); raw `raw.csv` stays on the persistent FS (large, regeneratable).

The persistent FS at `/home/ubuntu/persistent/RingKernel/` is the authoritative working copy on the VM. `CARGO_HOME` and `CARGO_TARGET_DIR` point at the persistent FS so subsequent sessions skip the ~15-minute cold build.

## Error handling / escalation gates

The autonomous run MUST stop and surface to the user at these gates:

| Gate | When | Action |
|---|---|---|
| **G1** | Any step of `b200-validate.sh` fails on first run | Stop. Report which step (driver / CUDA version / GPU detection / regression). Ask the user how to proceed. |
| **G2** | P2 FP4 or FP6 kernel stalls >2h with no compile/runtime success | Capture full nvcc stderr. Stop. Ask whether to skip P2, narrow scope to FP8 only, or extend. |
| **G3** | **Between Phase 1 and Phase 2** | Always stop. Summarize Phase 1 results (which P-levels green, any new tests committed, $ spent). Ask user to confirm before booking 8× B200. |
| **G4** | Phase 2 NVLink topology is not what we expect | Capture `nvidia-smi topo -m`. If fabric is 4-way or partial-mesh instead of 8-way full mesh, report adjusted bandwidth target and ask whether to proceed. |
| **G5** | Before `git tag v1.2.0` is created | Prepare tag commit locally. Do not push. Ask the user to review Addendum 7 + CHANGELOG, then confirm tag. |

The user is in control at every gate. Defaulting to "stop and ask" is correct because each gate either spends significant additional money (G3, G4) or makes hard-to-reverse changes (G5).

## Verification criteria — "the trip succeeded"

1. `b200-validate.sh` log shows green for all 6 steps on Phase 1 startup.
2. `cargo test -p ringkernel-cuda --features cuda --release` includes the new `blackwell_*` tests and all pass.
3. Each of P3, P4, P2 produces a Criterion estimate with CV < 5% over ≥ 100 samples — or the result carries a documented exception explaining the variance.
4. B200 numbers either improve on or match the H100 baseline in `docs/benchmarks/h100-b200-baseline.md`. Any regression is flagged in Addendum 7.
5. All previously passing tests in the workspace remain green (no Hopper regressions). Validated by `cargo test --workspace --release --exclude ringkernel-txmon` on the VM.
6. README compat matrix, CHANGELOG `[1.2.0]` header, and `h100-b200-baseline.md` all reference the same git SHA for the validation run.

## Subagent orchestration

The user specifically requested subagent-driven implementation mode. Tasks decompose cleanly:

### Phase 0 (parallel, all here)

Dispatched in a single message with multiple `Agent` tool calls:

- **Agent A — Rust test harness scaffolding**: writes the three new `tests/blackwell_*.rs` files using the existing H100 cluster tests (`crates/ringkernel-cuda/tests/cluster_*`) as templates. Leaves CUDA kernel bodies as `// TODO(on-VM)` blocks.
- **Agent B — Shell + analysis scripts**: writes `scripts/b200-collect-paper-numbers.sh`, `scripts/b200-multi-gpu-runner.sh`, and `analysis/b200_h100_compare.py`. Pure scripting, no Rust touched.
- **Agent C — Doc skeletons**: writes `docs/paper/sections/Addendum_07_Blackwell.md` skeleton with `<TBD>` markers. **Does not touch CHANGELOG.md** — that stays at `[Unreleased]` until Phase 3 so an aborted trip doesn't leave a half-moved changelog.

These agents are independent (no shared files, no sequential dependencies) so parallel dispatch is appropriate.

### Phase 1 + 2 (sequential, on VM — user-run)

Subagents cannot SSH into a Lambda instance. The user runs the VM session themselves following the runbook (see "VM runbook" below). The runbook is the artifact this strategy produces for human execution.

### Phase 3 (parallel, all here, after VM session returns)

Dispatched in a single message:

- **Agent D — Paper addendum integration**: takes the committed `benchmark_results/b200/*` and fills in `Addendum_07_Blackwell.md` using the markdown tables from `analysis/b200_h100_compare.py`.
- **Agent E — CHANGELOG + README polish**: moves the v1.2 groundwork block from `[Unreleased]` to a dated `[1.2.0]` entry, appends Blackwell-validation lines, updates the README compat matrix, and ensures `h100-b200-baseline.md` adds the B200 columns.
- **Agent F — Tag checklist**: verifies the verification criteria, prepares the `git tag v1.2.0` commit locally, reports back to the user for G5 confirmation. Does NOT push.

## VM runbook (handed to the user for Phase 1 + 2)

This will live in `docs/superpowers/GPU_VM_SESSION_GUIDE_B200.md` as an addendum, or be regenerated as a fresh file. The runbook captures:

### Phase 1 (1× B200)

```bash
# 0. SSH in, tmux, persistent FS
ssh ubuntu@<lambda-1x-b200-ip>
tmux new -s rk

# 1. Repo + env
cd /home/ubuntu/persistent
[ -d RingKernel ] && (cd RingKernel && git fetch && git checkout b200-validation && git pull --ff-only) \
                  || git clone -b b200-validation https://github.com/mivertowski/RustCompute.git RingKernel
cd RingKernel
export CARGO_HOME=/home/ubuntu/persistent/cargo-cache
export CARGO_TARGET_DIR=/home/ubuntu/persistent/RingKernel/target
./scripts/setup-gpu-vm.sh   # idempotent; installs rustup if missing

# 2. P0 — environment validation
./scripts/b200-validate.sh                 # green on all 6 steps; stop and ask if anything fails

# 3. P1 — cluster launch control test (fill in kernel body)
$EDITOR crates/ringkernel-cuda/tests/blackwell_cluster_launch.rs
cargo test -p ringkernel-cuda --features cuda --release --test blackwell_cluster_launch -- --nocapture

# 4. P2 — FP4 / FP6 / FP8 kernel smoke (fill in kernel bodies, iterate on nvcc errors)
$EDITOR crates/ringkernel-cuda/tests/blackwell_fp_scalar_smoke.rs
cargo test -p ringkernel-cuda --features cuda --release --test blackwell_fp_scalar_smoke -- --nocapture

# 5. P4 — sub-50ns command inject (paper-grade)
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc <b200-base-clock>     # filled in once we see what the device reports
sudo nvidia-smi -c EXCLUSIVE_PROCESS
./scripts/b200-collect-paper-numbers.sh paper_command_inject

# 6. P5 — TEE-I/O probe (may need instance reboot to enable CC mode)
cargo test -p ringkernel-cuda --features cuda --release --test blackwell_tee_io_probe -- --nocapture

# 7. Regression — Hopper tests still pass
cargo test --workspace --release --exclude ringkernel-txmon

# 8. Commit results, push, terminate instance
git add logs/ benchmark_results/b200/ crates/ringkernel-cuda/tests/blackwell_*.rs
git commit -m "feat(b200): Phase 1 single-GPU Blackwell validation"
git push
# Lambda dashboard → Terminate
```

### Phase 2 (8× B200)

```bash
# 0. SSH into the 8× instance, attach the same persistent FS
ssh ubuntu@<lambda-8x-b200-ip>
tmux new -s rk
cd /home/ubuntu/persistent/RingKernel

# 1. Topology check (GATE G4)
nvidia-smi topo -m | tee logs/b200-topo-$(date -u +%Y%m%dT%H%M%SZ).txt
# Expect 8× B200 in NVLink 5 full mesh. If not, stop and report.

# 2. P3 — NVLink 5 peer bandwidth
./scripts/b200-multi-gpu-runner.sh p3

# 3. P4 repeat on multi-GPU host
./scripts/b200-multi-gpu-runner.sh p4

# 4. Commit, push, terminate
git add benchmark_results/b200/multi-gpu/ logs/b200-topo-*.txt
git commit -m "feat(b200): Phase 2 multi-GPU NVLink 5 bandwidth"
git push
# Lambda dashboard → Terminate
```

## Out of scope for this trip

- NVSHMEM symmetric-heap *runtime* validation. The bindings are wired (`crates/ringkernel-cuda/src/multi_gpu/nvshmem.rs`), but proper validation needs MPI bootstrap, which the Lambda 8× B200 instance may or may not have configured. Deferred until either we get hands-on MPI on a Lambda 8× instance or move to a Slurm-managed cluster.
- Rubin (sm_12.x) validation. No silicon yet; presets compile but routing is `from_compute_capability(12, _)`. Captured in `GpuArchitecture::rubin()` as a placeholder.
- Cross-vendor (Metal, WGSL) Blackwell-tier scalar support. Outside this trip's scope.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Lambda CUDA is < 12.9 | Low | Blocks P2 entirely (FP4/FP6 names won't resolve) | `b200-validate.sh` step 0 hard-fails early. Recovery: `apt install cuda-toolkit-12-9` or pick a different image. |
| FP4 kernel iteration takes >2h | Medium | Eats into P3/P4 budget | G2 escalates after 2h. Fallback: narrow P2 to FP8 only, keep FP4/FP6 as compile-only evidence. |
| Lambda 8× instance has reduced NVLink fabric | Low-Medium | P3 bandwidth target needs adjustment | G4 captures `nvidia-smi topo -m` and reports adjusted target. |
| CC (TEE) mode needs instance reboot we can't trigger | Medium | P5 stays at "probe only" | Acceptable — P5 has no benchmark target anyway. |
| Persistent FS not auto-mounted on 8× instance | Low | Phase 2 needs to re-clone | Acceptable — first session of Phase 2 incurs ~15-min cold build. |

## Open decisions (deferred to user during execution)

These are flagged in the B200 guide and remain open until execution:

1. **Paper Addendum 7 scope** — single section or split into 7a (single-GPU B200) + 7b (multi-GPU NVLink 5)? Defer to Phase 3.
2. **Crates.io publish strategy after tag** — do we publish v1.2.0 to crates.io immediately, or hold for a maintenance window? Defer to post-tag.
3. **CC mode probe success criterion** — at minimum, "a persistent actor launches and runs to completion under CC mode". Anything beyond that is exploratory.

## Acceptance

This spec is accepted when the user reviews and approves OR (as in this case) delegates approval autonomously. The next step is `superpowers:writing-plans` to produce the implementation plan, then `superpowers:subagent-driven-development` to execute Phase 0.
