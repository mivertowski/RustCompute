# RingKernel B200 Session Guide (Lambda)

> **For the Claude Code instance working with a Lambda B200 VM.**
> Read this before starting any work. Supersedes `GPU_VM_SESSION_GUIDE_V1_2.md` when the target is Blackwell on Lambda rather than Hopper on Azure.

## Prerequisites on the VM

Lambda's on-demand `gpu_1x_b200` (or `gpu_8x_b200`) instances ship with:

- Ubuntu 22.04 LTS + Lambda Stack (NVIDIA driver, CUDA toolkit, cuDNN, PyTorch pre-installed).
- `nvidia-smi`, `nvcc`, `gcc`, `git`, `curl` on PATH.
- No `rustup` by default — `scripts/setup-gpu-vm.sh` installs it.
- No `cargo` build cache — use a persistent filesystem if you plan to stop/start.

**Check before committing to a session:** `nvcc --version` must report **≥ 12.9**. FP4/FP6 type names (`__nv_fp4_e2m1`, `__nv_fp6_e3m2`, etc.) are not resolvable on older toolkits; `scripts/b200-validate.sh` hard-fails early if CUDA is too old.

## Persistent filesystem layout

Attach a Lambda persistent filesystem at `/home/ubuntu/persistent` (the exact mount point is set in the Lambda dashboard when creating the FS). Recommended layout:

```
/home/ubuntu/persistent/
  RingKernel/           # git checkout
  target/               # bind-mounted or symlinked into RingKernel/target
  cargo-cache/          # CARGO_HOME to keep registry downloads across sessions
  logs/                 # validation + benchmark logs accumulate here
```

Export across sessions:

```bash
export CARGO_HOME=/home/ubuntu/persistent/cargo-cache
export CARGO_TARGET_DIR=/home/ubuntu/persistent/RingKernel/target
```

First-session-only cost: ~15 min for a clean `cargo build --release --features cuda`. Subsequent sessions reuse the `target/` cache and re-link in seconds.

## First session — from cold VM to green validation

```bash
# 1. SSH in
ssh ubuntu@<lambda-instance-ip>

# 2. Bring persistent FS into view (if not auto-mounted)
ls /home/ubuntu/persistent/

# 3. Clone (first session) or pull (subsequent)
cd /home/ubuntu/persistent
[ -d RingKernel ] && (cd RingKernel && git pull --ff-only) \
                  || git clone https://github.com/mivertowski/RustCompute.git RingKernel
cd RingKernel

# 4. Install rust + build (setup-gpu-vm.sh skips driver/CUDA because Lambda Stack has them)
./scripts/setup-gpu-vm.sh

# 5. Validate Blackwell paths
./scripts/b200-validate.sh
```

If step 5 is green, we have a confirmed-Blackwell working environment and can start writing the dedicated on-device tests.

## Priority order once the VM is live

### P0 — `b200-validate.sh` must pass
All 6 steps green. If any step fails, stop and fix before moving on. This script is the single source of truth for "the environment is ready."

### P1 — Dedicated Cluster Launch Control test
Write a `cargo test` that invokes `cuLaunchKernelEx` with dynamic cluster dims and verifies a kernel executes. The existing `supports_cluster_launch_control()` query is compile-time only; the P1 target is a runtime-exercising test that calls the path and reads the result back. Expected location: `crates/ringkernel-cuda/tests/blackwell_cluster_launch.rs`. Use the H100 `tests/cluster_*` tests as a structural template.

### P2 — FP4 / FP6 / FP8 on-device smoke
Write a kernel in `ringkernel-cuda-codegen` that uses `ScalarType::FP4E2M1` through `ScalarType::FP8E5M2`, launch it, read back a tensor, assert the rounding/range matches the CUDA reference. This is the first test that exercises `ringkernel-ir::lower_cuda` → `__nv_fp4_e2m1` → nvcc → sm_100 PTX → launched kernel end-to-end.

### P3 — NVLink 5 peer bandwidth (8× B200 only)
Single-GPU B200 cannot exercise this. On `gpu_8x_b200`, re-run `examples/paper_multi_gpu_k2k_bw.rs` and compare against the H100 NVL baseline (258 GB/s at 16 MiB, ~81% of NV12 318 GB/s peak). Target: > 700 GB/s sustained (~78% of the 900 GB/s NVLink 5 per-GPU aggregate), logged with 95% CI for the paper addendum.

### P4 — Sub-50 ns command injection
Needs B200. The H100 number is 55 ns. If NVLink 5 + cluster.sync Blackwell-tier timings hold, we expect 30–45 ns. Re-run `paper_command_inject` with locked clocks.

### P5 — TEE-I/O surface
`supports_tee()` is exposed but not exercised. Interactive work: enable CC (Confidential Computing) mode on one B200, verify a persistent actor still launches and runs to completion. No benchmark target yet.

## Cost-efficient workflow

```bash
# Start (billing resumes):
# Lambda dashboard → Instances → Launch (or API: lambda-cloud launch)

# Stop when idle (billing pauses for instance compute; persistent FS keeps billing):
# Lambda dashboard → Terminate instance
# Persistent FS retains the repo, target/, and logs for the next session.
```

Keep a long-running `tmux` session so SSH disconnects don't kill in-flight builds:

```bash
tmux new -s rk
# ... work ...
# Detach: Ctrl-b d
# Re-attach: tmux attach -t rk
```

## Decision points to escalate to the user

1. **Single-GPU vs 8-GPU launch?** — default to `gpu_1x_b200` for P1/P2/P5; request `gpu_8x_b200` only for P3.
2. **Tag v1.2.0 before or after Blackwell validation?** — current `main` has v1.2 groundwork untagged since 2026-04-20. Two options: (a) tag v1.2.0 as "Blackwell-stubs-compile, Hopper-validated" now and v1.3.0 once runtime paths are Blackwell-validated, or (b) hold v1.2.0 until Blackwell is green and ship one combined release.
3. **Paper addendum scope** — does B200 data get folded into the main paper or a separate Addendum 7? Matches the v1.1 Addendum 6/6b pattern.

## Starting prompt for the B200 session

> Read `docs/superpowers/GPU_VM_SESSION_GUIDE_B200.md`. Run `./scripts/b200-validate.sh` and report which steps (if any) fail. If all six pass, start P1 (dedicated Cluster Launch Control test) and check with me before moving to P2. Do not launch `gpu_8x_b200` without asking — single GPU is the default.
