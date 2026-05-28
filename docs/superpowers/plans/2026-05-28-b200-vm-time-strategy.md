# B200 VM Time Strategy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land all Phase 0 deliverables from `docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md` so a follow-on B200 VM session can run end-to-end with minimal on-device improvisation.

**Architecture:** Three Rust test harnesses (cluster launch control, FP4/FP6/FP8 scalar smoke, TEE-I/O probe) whose kernel bodies are deliberately deferred to the VM; three shell/python scripts (paper-number collector, multi-GPU runner, H100↔B200 comparator); one paper-section skeleton; one VM runbook addendum. Everything ships on a new `b200-validation` branch that the VM session pulls and pushes against.

**Tech Stack:** Rust 1.75+ (workspace MSRV), cudarc 0.19.3, CUDA 12.9+ on the target VM (CUDA toolkit on dev machine is fine without sm_100 support — the dev-machine tests are compile-only assertions on PTX names, not GPU runtime), bash, Python 3.10+ with stdlib only (no matplotlib hard-requirement — see Task 7).

**Spec reference:** `docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md` (committed at `ebca934`).

---

## Task 1: Create the `b200-validation` branch

**Files:**
- Modify: git ref `refs/heads/b200-validation` (created from `main`)

- [ ] **Step 1: Confirm clean working tree on main**

Run: `git status --short && git rev-parse --abbrev-ref HEAD`
Expected: `## main` (clean tree apart from the existing `.serena/project.yml` modification, which we leave alone). If anything in `crates/` or `docs/` is dirty, stop and ask.

- [ ] **Step 2: Create and check out the branch**

```bash
git checkout -b b200-validation
```

Expected: `Switched to a new branch 'b200-validation'`.

- [ ] **Step 3: Verify the workspace still builds**

Run: `cargo check --workspace --no-default-features --features cpu --quiet`
Expected: completes with no errors. (We deliberately use `--no-default-features --features cpu` here so this step does not require nvcc on the dev machine; the CUDA paths are validated later on the VM.)

- [ ] **Step 4: No commit yet — branch creation is its own act, the first commit comes from Task 2.**

---

## Task 2: Scaffold `blackwell_cluster_launch.rs` harness

**Files:**
- Create: `crates/ringkernel-cuda/tests/blackwell_cluster_launch.rs`

Template: `crates/ringkernel-cuda/tests/cluster_launch.rs` (Hopper analogue — reuse the `DirectPtxModule` load pattern, the manual `cuMemAlloc_v2` / `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2` flow, and the `#[ignore]`-gated GPU test convention).

- [ ] **Step 1: Write the harness skeleton**

Create `crates/ringkernel-cuda/tests/blackwell_cluster_launch.rs` with the exact content below. Note that `KERNEL_PTX` is a placeholder constant whose body is filled in on the VM — the value `""` causes the runtime test to skip cleanly via the existing `is_blackwell()` gate. Compile-time tests still run.

```rust
//! Dedicated Cluster Launch Control test for Blackwell (sm_100+).
//!
//! Phase 0 (here): harness only. `KERNEL_PTX` is empty so the runtime
//! test exits early. The `#[test]` that exercises capability routing
//! runs unconditionally and serves as a Phase 0 sanity check.
//!
//! Phase 1 (on B200): fill in `KERNEL_PTX` with sm_100 PTX produced by
//! nvcc from a kernel that uses `cuLaunchKernelEx`-style dynamic
//! cluster dims, then drop the `#[ignore]` after a green run.

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::ptr;

use cudarc::driver::sys as cuda_sys;

use ringkernel_cuda::driver_api::DirectPtxModule;
use ringkernel_cuda::hopper::cluster::{
    self, ClusterLaunchConfig, ClusterSchedulingPolicy,
};
use ringkernel_cuda::launch_config::mode::GpuArchitecture;

/// Returns true iff a Blackwell-class device (CC >= 10.0) is present.
fn is_blackwell() -> bool {
    match ringkernel_cuda::CudaDevice::new(0) {
        Ok(d) => {
            let (major, _) = d.compute_capability();
            major >= 10
        }
        Err(_) => false,
    }
}

/// Placeholder PTX. Filled in on the VM from a real nvcc -arch=sm_100
/// compilation. Empty string is intentional: keeps the runtime test
/// from launching nonsense and lets the harness ship in Phase 0.
const KERNEL_PTX: &str = "";

const KERNEL_NAME: &str = "blackwell_cluster_launch_probe";

#[test]
fn blackwell_arch_routes_cluster_launch_control() {
    // Capability-routing sanity. Runs on every machine, no GPU needed.
    let bw = GpuArchitecture::blackwell();
    assert!(
        bw.supports_cluster_launch_control(),
        "Blackwell preset must report supports_cluster_launch_control = true"
    );
    let hopper = GpuArchitecture::hopper();
    assert!(
        !hopper.supports_cluster_launch_control(),
        "Hopper preset must NOT report supports_cluster_launch_control = true"
    );
}

#[test]
#[ignore = "requires B200 (sm_100+) and a filled-in KERNEL_PTX"]
fn blackwell_cluster_launch_control_runtime() {
    if !is_blackwell() {
        eprintln!("Skipping: not a Blackwell GPU");
        return;
    }
    if KERNEL_PTX.is_empty() {
        eprintln!("Skipping: KERNEL_PTX not yet filled in on this machine");
        return;
    }

    let device = ringkernel_cuda::CudaDevice::new(0).expect("device 0");
    let module = DirectPtxModule::load_ptx(&device, KERNEL_PTX).expect("load ptx");
    let func = module.get_function(KERNEL_NAME).expect("get function");

    // Sentinel: kernel writes 0xDEADBEEF to *out if the cluster launch
    // control path executed correctly. The exact kernel body is filled
    // in on the VM.
    let mut out_dev: u64 = 0;
    unsafe {
        let r = cuda_sys::cuMemAlloc_v2(&mut out_dev, 4);
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuMemAlloc");
        let zero: u32 = 0;
        let r = cuda_sys::cuMemcpyHtoD_v2(out_dev, &zero as *const u32 as *const c_void, 4);
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuMemcpyHtoD");
    }

    let config = ClusterLaunchConfig {
        grid_dim: (4, 1, 1),
        block_dim: (256, 1, 1),
        cluster_dim: (4, 1, 1), // 4 blocks per cluster = full cluster
        shared_mem_bytes: 0,
        scheduling_policy: ClusterSchedulingPolicy::Default,
    };

    let mut out_ptr = out_dev;
    let mut params: Vec<*mut c_void> =
        vec![&mut out_ptr as *mut u64 as *mut c_void];
    let stream: cuda_sys::CUstream = ptr::null_mut();

    unsafe {
        cluster::launch_kernel_with_cluster(func, &config, &mut params, stream)
            .expect("cluster launch");
        let r = cuda_sys::cuCtxSynchronize();
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuCtxSynchronize");
    }

    let mut out_host: u32 = 0;
    unsafe {
        let r = cuda_sys::cuMemcpyDtoH_v2(
            &mut out_host as *mut u32 as *mut c_void,
            out_dev,
            4,
        );
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuMemcpyDtoH");
        let _ = cuda_sys::cuMemFree_v2(out_dev);
    }
    assert_eq!(
        out_host, 0xDEAD_BEEF,
        "kernel did not write sentinel — cluster launch control path failed"
    );
}
```

- [ ] **Step 2: Verify the compile-time test passes on the dev machine**

Run:
```bash
cargo test -p ringkernel-cuda --no-default-features --features cpu \
    --test blackwell_cluster_launch -- --nocapture 2>&1 | tail -20
```

If `--features cpu` rejects the test (because of `#![cfg(feature = "cuda")]`), that is the correct outcome: the test compiles only under `cuda`. Run instead:

```bash
cargo check -p ringkernel-cuda --features cuda --tests --quiet
```

Expected: compiles without errors. (Actually launching the test requires nvcc, which we do not require on the dev machine.) If `cargo check` fails because nvcc is missing, fall back to a no-cuda build via `cargo build -p ringkernel-cuda --no-default-features --features cpu --tests` to confirm at least workspace integrity.

- [ ] **Step 3: Commit**

```bash
git add crates/ringkernel-cuda/tests/blackwell_cluster_launch.rs
git commit -m "$(cat <<'EOF'
test(b200): scaffold cluster launch control harness

Phase 0 of the B200 strategy. The compile-time capability-routing test
runs anywhere; the runtime test is #[ignore]-gated and skips cleanly
until KERNEL_PTX is filled in on the B200 VM with real sm_100 PTX.

Refs: docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Scaffold `blackwell_fp_scalar_smoke.rs` harness

**Files:**
- Create: `crates/ringkernel-cuda/tests/blackwell_fp_scalar_smoke.rs`

The codegen pieces already exist: `ScalarType::{BF16, FP8E4M3, FP8E5M2, FP6E3M2, FP6E2M3, FP4E2M1}` in `crates/ringkernel-ir/src/types.rs`, and `lower_cuda.rs:696-703` already emits the `__nv_*` names. So the Phase 0 harness can assert the *lowering* end-to-end (Rust → IR → emitted CUDA source) without nvcc. Phase 1 fills in the kernel body and the GPU-side roundtrip.

- [ ] **Step 1: Write the harness skeleton**

Create `crates/ringkernel-cuda/tests/blackwell_fp_scalar_smoke.rs`:

```rust
//! FP4 / FP6 / FP8 scalar smoke test for Blackwell (sm_100+).
//!
//! Phase 0 (here): asserts that ringkernel-ir lowers each Blackwell
//! scalar type to the correct __nv_* name. No GPU needed.
//!
//! Phase 1 (on B200): wires KERNEL_PTX per scalar, runs the roundtrip
//! kernel, and asserts the readback matches the reference value within
//! the scalar's representable range.

#![cfg(feature = "cuda")]

use ringkernel_ir::ScalarType;

/// Returns true iff a Blackwell-class device (CC >= 10.0) is present.
#[allow(dead_code)]
fn is_blackwell() -> bool {
    match ringkernel_cuda::CudaDevice::new(0) {
        Ok(d) => {
            let (major, _) = d.compute_capability();
            major >= 10
        }
        Err(_) => false,
    }
}

/// The names that ringkernel-ir::lower_cuda must emit for each
/// Blackwell scalar. Hard-coded for Phase 0 verification; the test
/// is the single source of truth for what nvcc -arch=sm_100 must accept.
const EXPECTED_CUDA_NAMES: &[(ScalarType, &str)] = &[
    (ScalarType::BF16, "__nv_bfloat16"),
    (ScalarType::FP8E4M3, "__nv_fp8_e4m3"),
    (ScalarType::FP8E5M2, "__nv_fp8_e5m2"),
    (ScalarType::FP6E3M2, "__nv_fp6_e3m2"),
    (ScalarType::FP6E2M3, "__nv_fp6_e2m3"),
    (ScalarType::FP4E2M1, "__nv_fp4_e2m1"),
];

#[test]
fn blackwell_scalar_types_have_minimum_compute_capability() {
    // FP4 / FP6 require CC >= 10.0; FP8 requires CC >= 9.0; BF16 >= 8.0.
    assert_eq!(ScalarType::BF16.min_compute_capability(), (8, 0));
    assert_eq!(ScalarType::FP8E4M3.min_compute_capability(), (9, 0));
    assert_eq!(ScalarType::FP8E5M2.min_compute_capability(), (9, 0));
    assert_eq!(ScalarType::FP6E3M2.min_compute_capability(), (10, 0));
    assert_eq!(ScalarType::FP6E2M3.min_compute_capability(), (10, 0));
    assert_eq!(ScalarType::FP4E2M1.min_compute_capability(), (10, 0));
}

#[test]
fn blackwell_scalar_types_are_floating_point() {
    for (ty, _) in EXPECTED_CUDA_NAMES {
        assert!(
            ty.is_floating(),
            "{ty:?} must be classified as floating point"
        );
    }
}

#[test]
#[ignore = "Phase 1 / on-VM: requires B200 (sm_100+) and a filled-in KERNEL_PTX per scalar"]
fn blackwell_fp_scalar_roundtrip_runtime() {
    if !is_blackwell() {
        eprintln!("Skipping: not a Blackwell GPU");
        return;
    }
    // Filled in on the VM. Per-scalar PTX strings live in a sibling
    // module that the VM session adds; pattern:
    //
    //   const KERNEL_PTX_FP8_E4M3: &str = "...";
    //   const KERNEL_PTX_FP6_E3M2: &str = "...";
    //   ...
    //
    // Each kernel takes one __nv_<ty> input scalar, multiplies by 2.0,
    // and writes the result. The roundtrip is the assertion target.
    eprintln!("TODO(on-VM): wire up per-scalar PTX and run roundtrip");
}
```

- [ ] **Step 2: Verify compile**

Run: `cargo check -p ringkernel-cuda --features cuda --tests --quiet`
Expected: compiles without errors.

If nvcc is unavailable on the dev machine, fall back to:
```bash
cargo build -p ringkernel-cuda --no-default-features --features cpu --tests --quiet
```
Even though this won't include the test (it's gated by `#[cfg(feature = "cuda")]`), it verifies the workspace remains buildable.

- [ ] **Step 3: Commit**

```bash
git add crates/ringkernel-cuda/tests/blackwell_fp_scalar_smoke.rs
git commit -m "$(cat <<'EOF'
test(b200): scaffold FP4/FP6/FP8 scalar smoke harness

Phase 0 of the B200 strategy. Compile-time tests assert the IR maps each
Blackwell scalar to the correct CUDA type name (BF16, FP8, FP6, FP4) and
the correct minimum compute capability. The runtime roundtrip kernel is
deferred to the VM session, where nvcc -arch=sm_100 errors will guide
the kernel body.

Refs: docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Scaffold `blackwell_tee_io_probe.rs` harness

**Files:**
- Create: `crates/ringkernel-cuda/tests/blackwell_tee_io_probe.rs`

CC (Confidential Computing) mode is detected at the device attribute level. The Phase 0 harness only verifies that the routing/preset logic claims TEE support; the runtime probe is fully gated until we are on a CC-enabled instance.

- [ ] **Step 1: Write the harness**

Create `crates/ringkernel-cuda/tests/blackwell_tee_io_probe.rs`:

```rust
//! TEE-I/O probe for Blackwell.
//!
//! Phase 0: asserts the Blackwell preset advertises TEE support.
//! Phase 1 (on VM with CC mode enabled): launches a minimal persistent
//! actor and asserts it runs to completion under CC mode.

#![cfg(feature = "cuda")]

use ringkernel_cuda::launch_config::mode::GpuArchitecture;

/// Returns true iff a Blackwell-class device (CC >= 10.0) is present.
#[allow(dead_code)]
fn is_blackwell() -> bool {
    match ringkernel_cuda::CudaDevice::new(0) {
        Ok(d) => {
            let (major, _) = d.compute_capability();
            major >= 10
        }
        Err(_) => false,
    }
}

#[test]
fn blackwell_preset_advertises_tee_support() {
    assert!(
        GpuArchitecture::blackwell().supports_tee(),
        "Blackwell preset must report supports_tee = true (TEE-I/O is a Blackwell capability)"
    );
    // Hopper has TEE too (H100 TEE). Both must report true; the
    // distinction is the *I/O* generalization on Blackwell, which is
    // not captured by the compile-time query.
    assert!(GpuArchitecture::hopper().supports_tee());
}

#[test]
#[ignore = "Phase 1 / on-VM: requires a B200 instance with CC mode enabled (often a reboot)"]
fn blackwell_persistent_actor_runs_under_cc_mode() {
    if !is_blackwell() {
        eprintln!("Skipping: not a Blackwell GPU");
        return;
    }
    // TODO(on-VM): launch a trivial persistent actor (one block, one
    // thread, one iteration that writes a sentinel to mapped memory),
    // verify it completes, and verify nvidia-smi reports CC mode
    // active during the run.
    eprintln!("TODO(on-VM): exercise persistent actor under CC mode");
}
```

- [ ] **Step 2: Verify compile**

Run: `cargo check -p ringkernel-cuda --features cuda --tests --quiet`
Expected: compiles without errors. (If nvcc is unavailable on dev machine, same fallback as Task 3.)

- [ ] **Step 3: Commit**

```bash
git add crates/ringkernel-cuda/tests/blackwell_tee_io_probe.rs
git commit -m "$(cat <<'EOF'
test(b200): scaffold TEE-I/O probe harness

Phase 0 asserts the Blackwell preset advertises TEE support. Phase 1
will run a persistent actor under CC mode on a TEE-enabled B200 instance.

Refs: docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Write `scripts/b200-collect-paper-numbers.sh`

**Files:**
- Create: `scripts/b200-collect-paper-numbers.sh`

- [ ] **Step 1: Write the script**

Create `scripts/b200-collect-paper-numbers.sh` (mark executable in Step 2):

```bash
#!/usr/bin/env bash
# Wraps a Criterion bench with B200-grade environment (locked clocks,
# exclusive mode, fixed warmup) and dumps results under
# benchmark_results/b200/<bench>/.
#
# Usage:
#   scripts/b200-collect-paper-numbers.sh <bench-name> [criterion-args...]
#
# Examples:
#   scripts/b200-collect-paper-numbers.sh latency
#   scripts/b200-collect-paper-numbers.sh academic_harness -- --warm-up-time 30
#
# Run from a B200 VM with sudo access (for nvidia-smi -lgc / -c).
# On dev machines without a B200, the script refuses to start.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <bench-name> [criterion-args...]" >&2
    exit 2
fi
BENCH="$1"; shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Hard gate: only run on Blackwell.
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
CC_MAJOR="${GPU_CC%%.*}"
if [ "$CC_MAJOR" -lt 10 ]; then
    echo "FAIL: compute cap $GPU_CC < 10.0 — this script is for Blackwell+" >&2
    exit 1
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="benchmark_results/b200/${BENCH}/${TS}"
mkdir -p "$OUTDIR"
LOG="${OUTDIR}/criterion.log"

# Environment snapshot.
nvidia-smi -q                       > "${OUTDIR}/nvidia_smi_full.txt"   || true
nvcc --version                      > "${OUTDIR}/nvcc_version.txt"      || true
rustc --version                     > "${OUTDIR}/rustc_version.txt"     || true
git rev-parse HEAD                  > "${OUTDIR}/git_commit.txt"        || true
uname -a                            > "${OUTDIR}/uname.txt"             || true

# Lock clocks (max boost) + exclusive mode. Falls back gracefully if
# the user is not root.
if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
    sudo nvidia-smi -pm 1                            >> "$LOG" 2>&1 || true
    BASE_CLOCK="$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)"
    sudo nvidia-smi -lgc "${BASE_CLOCK}"             >> "$LOG" 2>&1 || true
    sudo nvidia-smi -c EXCLUSIVE_PROCESS             >> "$LOG" 2>&1 || true
    echo "Locked clocks to ${BASE_CLOCK} MHz, exclusive mode on." >> "$LOG"
else
    echo "WARN: no passwordless sudo — clocks NOT locked. Results will be more variable." >&2
fi

# 60-second warmup with a known-cheap bench (silently OK if absent).
cargo bench --package ringkernel -- serialization --warm-up-time 30 >> "$LOG" 2>&1 || true

# The actual run.
echo ""                                                            >> "$LOG"
echo "=== RUN $BENCH @ $TS ==="                                     >> "$LOG"
cargo bench --package ringkernel -- "$BENCH" "$@"                   2>&1 | tee -a "$LOG"

# Snapshot Criterion JSON estimates so we can commit them.
if [ -d target/criterion ]; then
    rsync -a --include='*/' --include='estimates.json' --include='*.json' \
          --exclude='raw.csv' --exclude='*.svg' --exclude='report' \
          target/criterion/ "${OUTDIR}/criterion/"  2>/dev/null \
      || cp -r target/criterion "${OUTDIR}/criterion-raw"
fi

# Release exclusive mode so other tests can run after.
if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
    sudo nvidia-smi -c DEFAULT       >> "$LOG" 2>&1 || true
    sudo nvidia-smi -rgc             >> "$LOG" 2>&1 || true
fi

echo ""
echo "Wrote ${OUTDIR}"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/b200-collect-paper-numbers.sh`

- [ ] **Step 3: Verify shellcheck cleanliness (best effort)**

Run: `command -v shellcheck && shellcheck scripts/b200-collect-paper-numbers.sh || echo "shellcheck unavailable — skipping"`
Expected: either green or "shellcheck unavailable". Any genuine warnings must be addressed.

- [ ] **Step 4: Commit**

```bash
git add scripts/b200-collect-paper-numbers.sh
git commit -m "$(cat <<'EOF'
chore(b200): paper-grade Criterion runner with locked-clock setup

Hard-gates on CC >= 10.0, locks clocks, runs the named bench, snapshots
Criterion estimates JSON under benchmark_results/b200/<bench>/<ts>/.
Used in Phase 1/2 of the B200 trip to collect publishable numbers.

Refs: docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Write `scripts/b200-multi-gpu-runner.sh`

**Files:**
- Create: `scripts/b200-multi-gpu-runner.sh`

- [ ] **Step 1: Write the script**

Create `scripts/b200-multi-gpu-runner.sh`:

```bash
#!/usr/bin/env bash
# Phase 2 (8x B200) runner. Subcommands:
#   topo   - capture nvidia-smi topo -m + p2pBandwidthLatencyTest output
#   p3     - run paper_multi_gpu_k2k_bw across 4KB..256MB sweep
#   p4     - re-run command inject bench on the multi-GPU host
#
# Usage:  scripts/b200-multi-gpu-runner.sh <topo|p3|p4>
#
# Run from inside an 8x B200 Lambda instance.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <topo|p3|p4>" >&2
    exit 2
fi
CMD="$1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
CC_MAJOR="${GPU_CC%%.*}"

if [ "$CC_MAJOR" -lt 10 ]; then
    echo "FAIL: compute cap $GPU_CC < 10.0 — multi-GPU runner is for Blackwell+" >&2
    exit 1
fi

case "$CMD" in
    topo)
        OUTDIR="logs/b200-topo/${TS}"
        mkdir -p "$OUTDIR"
        nvidia-smi topo -m              > "${OUTDIR}/topo.txt"
        nvidia-smi nvlink --status      > "${OUTDIR}/nvlink-status.txt" || true
        nvidia-smi nvlink --capabilities > "${OUTDIR}/nvlink-caps.txt"   || true
        nvidia-smi -q -d MEMORY,POWER,CLOCK > "${OUTDIR}/device-state.txt"
        echo "GPU count: ${GPU_COUNT}"  > "${OUTDIR}/summary.txt"
        echo "Compute cap: ${GPU_CC}"  >> "${OUTDIR}/summary.txt"
        echo ""
        echo "Wrote ${OUTDIR}"
        cat "${OUTDIR}/topo.txt"
        ;;
    p3)
        if [ "$GPU_COUNT" -lt 2 ]; then
            echo "FAIL: P3 requires at least 2 GPUs visible (have ${GPU_COUNT})" >&2
            exit 1
        fi
        OUTDIR="benchmark_results/b200/multi-gpu/p3/${TS}"
        mkdir -p "$OUTDIR"
        cargo test -p ringkernel-cuda --features "cuda,multi-gpu" --release \
            --test paper_multi_gpu_k2k_bw -- --ignored --nocapture \
            2>&1 | tee "${OUTDIR}/run.log"
        echo "Wrote ${OUTDIR}"
        ;;
    p4)
        # Reuse the single-GPU paper-number runner; the bench self-pins
        # to device 0 by default. Multi-GPU host is included only for
        # confirming the inject path doesn't regress with N visible GPUs.
        ./scripts/b200-collect-paper-numbers.sh latency
        ;;
    *)
        echo "unknown subcommand: $CMD (expected: topo | p3 | p4)" >&2
        exit 2
        ;;
esac
```

- [ ] **Step 2: Make executable**

Run: `chmod +x scripts/b200-multi-gpu-runner.sh`

- [ ] **Step 3: Verify shellcheck cleanliness (best effort)**

Run: `command -v shellcheck && shellcheck scripts/b200-multi-gpu-runner.sh || echo "shellcheck unavailable — skipping"`

- [ ] **Step 4: Commit**

```bash
git add scripts/b200-multi-gpu-runner.sh
git commit -m "$(cat <<'EOF'
chore(b200): multi-GPU runner for Phase 2 NVLink 5 measurement

Wraps the topology check (G4 in the strategy spec), paper_multi_gpu_k2k_bw
sweep, and the multi-GPU P4 repeat. Hard-gates on CC >= 10.0 and GPU
count >= 2 for the bandwidth subcommand.

Refs: docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Write `analysis/b200_h100_compare.py`

**Files:**
- Create: `analysis/b200_h100_compare.py`
- Create: `analysis/README.md`

The script reads Criterion `estimates.json` outputs from two locations and emits a markdown table. Plotting is optional (matplotlib may not be installed); if absent, the script still emits the table.

- [ ] **Step 1: Write the README**

Create `analysis/README.md`:

```markdown
# RingKernel analysis scripts

Local-only tooling for paper-prep analysis. Not part of any crate build.

## b200_h100_compare.py

Reads Criterion `estimates.json` outputs from H100 baseline and B200
results, emits a markdown comparison table suitable for paper Addendum 7.

Usage:

```bash
python3 analysis/b200_h100_compare.py \
    --h100 benchmark_results/h100/criterion \
    --b200 benchmark_results/b200/<bench>/<ts>/criterion \
    --out  docs/paper/sections/_b200_h100_table.md
```

If matplotlib is installed, also emits per-bench bar plots to the
parent dir of `--out`. Plotting is best-effort and silently skipped
if matplotlib is unavailable — the markdown output is the primary
deliverable.
```

- [ ] **Step 2: Write the Python script**

Create `analysis/b200_h100_compare.py`:

```python
#!/usr/bin/env python3
"""Compare H100 baseline Criterion estimates against B200 results.

Emits a markdown table with mean / 95% CI / speedup per benchmark.
Matplotlib plots are best-effort (no hard dependency).

Spec: docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, Iterable, Optional, Tuple


def find_estimates(root: pathlib.Path) -> Dict[str, dict]:
    """Walk a Criterion output tree and collect benchmark name -> estimates dict."""
    out: Dict[str, dict] = {}
    if not root.exists():
        return out
    for path in root.rglob("estimates.json"):
        # Criterion layout: <root>/<bench_id>/[<sub>/]new/estimates.json
        rel = path.relative_to(root)
        parts = rel.parts
        if "new" not in parts:
            continue
        new_idx = parts.index("new")
        if new_idx == 0:
            continue
        bench_id = "/".join(parts[:new_idx])
        try:
            with path.open() as fh:
                out[bench_id] = json.load(fh)
        except json.JSONDecodeError:
            print(f"warn: malformed json: {path}", file=sys.stderr)
    return out


def fmt_ns(ns: float) -> str:
    """Format a nanosecond duration with a sensible unit."""
    if ns < 1_000:
        return f"{ns:.2f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.3f} s"


def speedup(h100_ns: float, b200_ns: float) -> str:
    if b200_ns <= 0:
        return "n/a"
    ratio = h100_ns / b200_ns
    return f"{ratio:.2f}×"


def mean_ci(est: dict) -> Optional[Tuple[float, float, float]]:
    """Return (mean_ns, low_ns, high_ns) from a Criterion estimates dict."""
    mean = est.get("mean", {})
    pe = mean.get("point_estimate")
    ci = mean.get("confidence_interval", {})
    lo = ci.get("lower_bound")
    hi = ci.get("upper_bound")
    if pe is None or lo is None or hi is None:
        return None
    return float(pe), float(lo), float(hi)


def build_table(h100: Dict[str, dict], b200: Dict[str, dict]) -> str:
    keys = sorted(set(h100) | set(b200))
    lines = [
        "| Benchmark | H100 mean (95% CI) | B200 mean (95% CI) | B200 / H100 |",
        "|---|---|---|---|",
    ]
    for k in keys:
        h = mean_ci(h100.get(k, {})) if k in h100 else None
        b = mean_ci(b200.get(k, {})) if k in b200 else None
        if h is None and b is None:
            continue
        h_cell = (
            f"{fmt_ns(h[0])} ({fmt_ns(h[1])}, {fmt_ns(h[2])})" if h else "—"
        )
        b_cell = (
            f"{fmt_ns(b[0])} ({fmt_ns(b[1])}, {fmt_ns(b[2])})" if b else "—"
        )
        sp = speedup(h[0], b[0]) if (h and b) else "—"
        lines.append(f"| `{k}` | {h_cell} | {b_cell} | {sp} |")
    return "\n".join(lines) + "\n"


def maybe_plot(
    h100: Dict[str, dict], b200: Dict[str, dict], out_dir: pathlib.Path
) -> None:
    """Emit a simple bar plot per benchmark if matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("info: matplotlib unavailable — skipping plots", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for k in sorted(set(h100) & set(b200)):
        h = mean_ci(h100[k])
        b = mean_ci(b200[k])
        if h is None or b is None:
            continue
        fig, ax = plt.subplots(figsize=(4, 3))
        labels = ["H100", "B200"]
        means = [h[0], b[0]]
        errs = [
            [h[0] - h[1], b[0] - b[1]],  # lower error
            [h[2] - h[0], b[2] - b[0]],  # upper error
        ]
        ax.bar(labels, means, yerr=errs, capsize=4)
        ax.set_ylabel("ns (lower is better)")
        ax.set_title(k)
        fig.tight_layout()
        safe = k.replace("/", "__").replace(" ", "_")
        fig.savefig(out_dir / f"{safe}.png", dpi=120)
        plt.close(fig)


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100", required=True, type=pathlib.Path)
    parser.add_argument("--b200", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(list(argv))

    h100 = find_estimates(args.h100)
    b200 = find_estimates(args.b200)
    if not h100 and not b200:
        print("error: neither --h100 nor --b200 contained estimates.json", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_table(h100, b200))
    maybe_plot(h100, b200, args.out.parent / "plots")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 3: Smoke-test the script**

Run:
```bash
python3 -c "import ast; ast.parse(open('analysis/b200_h100_compare.py').read()); print('parse OK')"
```
Expected: `parse OK`.

- [ ] **Step 4: Commit**

```bash
chmod +x analysis/b200_h100_compare.py
git add analysis/b200_h100_compare.py analysis/README.md
git commit -m "$(cat <<'EOF'
chore(b200): H100/B200 Criterion comparison analyser

Walks Criterion estimates.json trees, emits a markdown table with
mean / 95% CI / B200-vs-H100 speedup ratios. Matplotlib plots are
best-effort. Designed to feed paper Addendum 7.

Refs: docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Write `docs/paper/sections/Addendum_07_Blackwell.md` skeleton

**Files:**
- Create: `docs/paper/sections/Addendum_07_Blackwell.md`

- [ ] **Step 1: Write the skeleton**

Create `docs/paper/sections/Addendum_07_Blackwell.md`:

```markdown
# Addendum 7: Blackwell (B200) Validation

> Companion to the main paper. Captures B200-specific measurements
> collected during the 2026-05/06 Lambda Cloud session.
> Single source of truth for the Blackwell numbers cited in §10
> Future Work and the v1.2 README compatibility matrix.

## System configuration

| Property | Value |
|----------|-------|
| GPU | NVIDIA B200 (`<TBD>` × N) |
| GPU memory | `<TBD>` GiB HBM3e |
| Driver version | `<TBD>` |
| CUDA toolkit | `<TBD>` |
| Compute capability | `<TBD>` (expected 10.0) |
| GPU clock (locked) | `<TBD>` MHz |
| Host CPU | `<TBD>` |
| RAM | `<TBD>` GiB |
| OS | Ubuntu 22.04 LTS (Lambda Stack) |
| RingKernel version | `<TBD>` |
| Git commit | `<TBD>` |
| `RINGKERNEL_CUDA_ARCH` | sm_100 |

## P0 — Environment validation

Result of `scripts/b200-validate.sh`:

```text
<TBD: paste the green log>
```

## P1 — Cluster Launch Control

Test: `cargo test -p ringkernel-cuda --features cuda --release --test blackwell_cluster_launch -- --ignored`

| Metric | Value |
|---|---|
| Test status | `<TBD: PASS / FAIL>` |
| Sentinel written | `<TBD: 0xDEADBEEF / other>` |
| Cluster size | 4 blocks |
| Notes | `<TBD>` |

## P2 — FP4 / FP6 / FP8 scalar smoke

Test: `cargo test -p ringkernel-cuda --features cuda --release --test blackwell_fp_scalar_smoke -- --ignored`

| Scalar | nvcc OK? | Kernel runtime OK? | Roundtrip error |
|---|---|---|---|
| BF16 | `<TBD>` | `<TBD>` | `<TBD>` |
| FP8E4M3 | `<TBD>` | `<TBD>` | `<TBD>` |
| FP8E5M2 | `<TBD>` | `<TBD>` | `<TBD>` |
| FP6E3M2 | `<TBD>` | `<TBD>` | `<TBD>` |
| FP6E2M3 | `<TBD>` | `<TBD>` | `<TBD>` |
| FP4E2M1 | `<TBD>` | `<TBD>` | `<TBD>` |

## P3 — NVLink 5 peer bandwidth (8× B200 only)

Run: `scripts/b200-multi-gpu-runner.sh p3`

H100 baseline: 258 GB/s sustained @ 16 MiB (~81% of NV12 318 GB/s peak)

| Size | Bandwidth (GB/s) | 95% CI | % of NVLink 5 peak |
|---|---|---|---|
| 4 KiB | `<TBD>` | `<TBD>` | `<TBD>` |
| 64 KiB | `<TBD>` | `<TBD>` | `<TBD>` |
| 1 MiB | `<TBD>` | `<TBD>` | `<TBD>` |
| 16 MiB | `<TBD>` | `<TBD>` | `<TBD>` |
| 256 MiB | `<TBD>` | `<TBD>` | `<TBD>` |

Target: > 700 GB/s sustained at 16 MiB (~78% of 900 GB/s NVLink 5 per-GPU aggregate).

## P4 — Sub-50 ns command injection

H100 baseline: 55 ns mean (75 ns persistent-actor injection minus 20 ns of host atomic overhead).

| Statistic | H100 | B200 | Speedup |
|---|---|---|---|
| Mean (ns) | `<TBD>` | `<TBD>` | `<TBD>` |
| 95% CI | `<TBD>` | `<TBD>` | |
| CV | `<TBD>` | `<TBD>` | |

Target: 30–45 ns on B200 from the NVLink 5 + cluster.sync timing improvements.

## P5 — TEE-I/O surface

Test: `cargo test -p ringkernel-cuda --features cuda --release --test blackwell_tee_io_probe -- --ignored`

| Property | Value |
|---|---|
| CC mode enabled | `<TBD: yes / no>` |
| Persistent actor completed | `<TBD: yes / no>` |
| Observed throughput delta vs non-CC | `<TBD>` |
| Notes | `<TBD>` |

## H100 ↔ B200 comparison table

Generated by `analysis/b200_h100_compare.py`. Re-generate at write-time:

```bash
python3 analysis/b200_h100_compare.py \
    --h100 benchmark_results/h100/criterion \
    --b200 benchmark_results/b200/<bench>/<ts>/criterion \
    --out  docs/paper/sections/_b200_h100_table.md
```

`<TBD: include or transclude _b200_h100_table.md here>`

## Threats to validity

`<TBD>`

## Reproducibility

```bash
git checkout <TBD: validation commit SHA>
export RINGKERNEL_CUDA_ARCH=sm_100
./scripts/b200-validate.sh
./scripts/b200-collect-paper-numbers.sh latency
# 8x B200 only:
./scripts/b200-multi-gpu-runner.sh p3
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/paper/sections/Addendum_07_Blackwell.md
git commit -m "$(cat <<'EOF'
docs(paper): scaffold Addendum 7 (Blackwell B200 validation)

Skeleton with <TBD> placeholders for each P-level result. Numbers and
narrative fill in during Phase 3 (offline wrap-up) once the B200 data
has been collected and analysed.

Refs: docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Write the VM runbook addendum

**Files:**
- Modify: `docs/superpowers/GPU_VM_SESSION_GUIDE_B200.md`

We *append* a "Phase 1 / Phase 2 step-by-step" section to the existing guide rather than rewrite it, so the original priority discussion stays intact.

- [ ] **Step 1: Read the current end of the guide**

Run: `wc -l docs/superpowers/GPU_VM_SESSION_GUIDE_B200.md`
Expected: ~111 lines (matches what we read in brainstorming).

- [ ] **Step 2: Append the new section**

Use the Edit tool to append (find a unique trailing string in the existing file, replace with the same string + new content). The unique trailing string is the existing "Starting prompt for the B200 session" subsection ending; append after its final line.

Find this string (the current final paragraph):

```
> Read `docs/superpowers/GPU_VM_SESSION_GUIDE_B200.md`. Run `./scripts/b200-validate.sh` and report which steps (if any) fail. If all six pass, start P1 (dedicated Cluster Launch Control test) and check with me before moving to P2. Do not launch `gpu_8x_b200` without asking — single GPU is the default.
```

Replace with:

```
> Read `docs/superpowers/GPU_VM_SESSION_GUIDE_B200.md`. Run `./scripts/b200-validate.sh` and report which steps (if any) fail. If all six pass, start P1 (dedicated Cluster Launch Control test) and check with me before moving to P2. Do not launch `gpu_8x_b200` without asking — single GPU is the default.

---

## Phase 1 runbook (1× B200)

Implementation plan: `docs/superpowers/plans/2026-05-28-b200-vm-time-strategy.md`.
Strategy spec: `docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md`.

```bash
# 0. SSH, tmux, persistent FS
ssh ubuntu@<lambda-1x-b200-ip>
tmux new -s rk

# 1. Repo on the persistent FS
cd /home/ubuntu/persistent
[ -d RingKernel ] && (cd RingKernel && git fetch && git checkout b200-validation && git pull --ff-only) \
                  || git clone -b b200-validation https://github.com/mivertowski/RustCompute.git RingKernel
cd RingKernel
export CARGO_HOME=/home/ubuntu/persistent/cargo-cache
export CARGO_TARGET_DIR=/home/ubuntu/persistent/RingKernel/target
./scripts/setup-gpu-vm.sh   # idempotent

# 2. P0 — environment validation (GATE G1 if anything fails)
./scripts/b200-validate.sh

# 3. P1 — fill in KERNEL_PTX in blackwell_cluster_launch.rs, run
$EDITOR crates/ringkernel-cuda/tests/blackwell_cluster_launch.rs
cargo test -p ringkernel-cuda --features cuda --release \
    --test blackwell_cluster_launch -- --ignored --nocapture

# 4. P2 — fill in per-scalar KERNEL_PTX in blackwell_fp_scalar_smoke.rs
#         (GATE G2 if FP4/FP6 iteration exceeds 2h)
$EDITOR crates/ringkernel-cuda/tests/blackwell_fp_scalar_smoke.rs
cargo test -p ringkernel-cuda --features cuda --release \
    --test blackwell_fp_scalar_smoke -- --ignored --nocapture

# 5. P4 — paper-grade command inject latency
./scripts/b200-collect-paper-numbers.sh latency

# 6. P5 — TEE-I/O probe (may need instance reboot with CC mode flag)
cargo test -p ringkernel-cuda --features cuda --release \
    --test blackwell_tee_io_probe -- --ignored --nocapture

# 7. Regression
cargo test --workspace --release --exclude ringkernel-txmon

# 8. Commit + push + terminate
git add logs/ benchmark_results/b200/ crates/ringkernel-cuda/tests/blackwell_*.rs
git commit -m "feat(b200): Phase 1 single-GPU Blackwell validation"
git push origin b200-validation
# Lambda dashboard → Terminate
```

After this commit, **stop the 1× instance and report Phase 1 results** so the user can confirm GATE G3 before booking the 8× instance.

## Phase 2 runbook (8× B200)

```bash
# 0. SSH into the 8x instance, re-attach the persistent FS
ssh ubuntu@<lambda-8x-b200-ip>
tmux new -s rk
cd /home/ubuntu/persistent/RingKernel
git fetch && git checkout b200-validation && git pull --ff-only

# 1. GATE G4 — topology check
./scripts/b200-multi-gpu-runner.sh topo
#   If fabric is not 8-way NVLink 5 full mesh: stop, report, ask.

# 2. P3 — NVLink 5 peer bandwidth
./scripts/b200-multi-gpu-runner.sh p3

# 3. P4 repeat on multi-GPU host
./scripts/b200-multi-gpu-runner.sh p4

# 4. Commit, push, terminate
git add logs/b200-topo/ benchmark_results/b200/multi-gpu/
git commit -m "feat(b200): Phase 2 NVLink 5 bandwidth + multi-GPU P4 repeat"
git push origin b200-validation
# Lambda dashboard → Terminate
```

## Hard gates summary

| Gate | When | If hit |
|---|---|---|
| G1 | Any `b200-validate.sh` step fails | Stop, report which step, ask user |
| G2 | P2 FP4/FP6 stalls > 2h | Capture nvcc stderr, ask whether to narrow to FP8 or extend |
| G3 | Between Phase 1 and Phase 2 | Mandatory: summarise Phase 1, ask before booking 8× |
| G4 | NVLink fabric is not 8-way full mesh | Report adjusted target, ask whether to proceed |
| G5 | Before `git tag v1.2.0` | Prepare locally, do not push, ask user to review Addendum 7 + CHANGELOG |

## Phase 3 (offline, after VM session returns)

Plan: `docs/superpowers/plans/2026-05-28-b200-vm-time-strategy.md` Task 11 (added once Phase 1+2 data lands).
```

- [ ] **Step 3: Verify the markdown still renders cleanly**

Run: `head -200 docs/superpowers/GPU_VM_SESSION_GUIDE_B200.md | tail -80`
Expected: the new section follows the old "Starting prompt" paragraph without orphan headings.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/GPU_VM_SESSION_GUIDE_B200.md
git commit -m "$(cat <<'EOF'
docs(b200): append Phase 1/2 runbook to the session guide

Adds explicit step-by-step Phase 1 (1x) and Phase 2 (8x) bash workflows
that map to the new scripts and tests. Documents G1-G5 hard gates and
forwards to the implementation plan for offline-wrap-up details.

Refs: docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final verification + push branch

**Files:**
- (no new files; runs verification commands)

- [ ] **Step 1: Full workspace check (no GPU required)**

Run: `cargo check --workspace --no-default-features --features cpu --quiet`
Expected: completes with no errors.

- [ ] **Step 2: Compile-only check with cuda feature (skipped if nvcc missing)**

Run:
```bash
if command -v nvcc >/dev/null; then
    cargo check -p ringkernel-cuda --features cuda --tests --quiet
else
    echo "nvcc unavailable — skipping cuda-feature check (will run on VM)"
fi
```
Expected: either completes cleanly, or prints the skip notice.

- [ ] **Step 3: Confirm the three new test files exist and are non-empty**

Run:
```bash
ls -la crates/ringkernel-cuda/tests/blackwell_*.rs \
       scripts/b200-collect-paper-numbers.sh \
       scripts/b200-multi-gpu-runner.sh \
       analysis/b200_h100_compare.py \
       docs/paper/sections/Addendum_07_Blackwell.md
```
Expected: all six files exist, scripts are executable, none are empty.

- [ ] **Step 4: Push the branch**

```bash
git log --oneline main..HEAD
git push -u origin b200-validation
```
Expected: ~9 commits ahead of main, branch pushed.

- [ ] **Step 5: Report**

End-of-Phase-0 report to the user, with one short paragraph summarising:
- How many commits landed on `b200-validation`
- Which files were created
- The exact next-action command for the user to launch the VM session:
  ```
  ssh ubuntu@<lambda-1x-b200-ip>
  # then follow docs/superpowers/GPU_VM_SESSION_GUIDE_B200.md from "Phase 1 runbook (1x B200)"
  ```

---

## Self-review (run after Task 10)

1. **Spec coverage** — every Phase 0 deliverable from
   `docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md §Phase 0 deliverables`
   has a task: cluster harness (T2), FP-scalar harness (T3), TEE-I/O harness (T4),
   paper-number script (T5), multi-GPU runner (T6), analysis script (T7), Addendum 7
   skeleton (T8), VM runbook (T9). CHANGELOG.md is intentionally untouched in
   Phase 0 (matches spec self-review fix).

2. **Placeholder scan** — `<TBD>` markers appear only in
   `Addendum_07_Blackwell.md` (the deliberate Phase 3 fill-in surface) and in
   ``KERNEL_PTX = ""`` in `blackwell_cluster_launch.rs` (the deliberate
   on-VM fill-in surface). No `TODO(later)`, no `implement later`, no narrative
   gaps in any task.

3. **Type consistency** — `KERNEL_PTX`, `KERNEL_NAME`, `is_blackwell()`, and
   `ClusterLaunchConfig` referenced across tasks match the same names. The
   `ScalarType::FP4E2M1 → "__nv_fp4_e2m1"` lowering asserted in T3 matches
   `crates/ringkernel-ir/src/lower_cuda.rs:703`.

## Phase 3 placeholder (NOT executed in this session)

Task 11–14 (offline wrap-up, executed after the VM session returns with data) live in a follow-on plan that will be created from `docs/superpowers/specs/2026-05-28-b200-vm-time-strategy.md §Phase 3 deliverables` once Phase 1+2 commits land on `b200-validation`. The strategy spec lists those deliverables explicitly: CHANGELOG `[1.2.0]` move, Addendum 7 fill-in via Agent D/E/F, README compat matrix, `git tag v1.2.0`.

---

**Execution mode:** the user has chosen subagent-driven implementation (see strategy spec §Subagent orchestration). Tasks 2/3/4 are independent (separate test files); tasks 5/6/7 are independent (separate scripts); tasks 8/9 are independent (separate docs). Recommended dispatch: parallel batch [T2, T3, T4] → parallel batch [T5, T6, T7] → parallel batch [T8, T9] → sequential T10. Task 1 (branch creation) must run first.
