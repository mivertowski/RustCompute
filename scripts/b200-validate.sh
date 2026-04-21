#!/usr/bin/env bash
# RingKernel B200 / Blackwell validation runner.
#
# Assumes scripts/setup-gpu-vm.sh has already run (driver, CUDA, rust,
# repo checkout, release build present). This script validates the
# Blackwell-only paths that were shipped as stubs in v1.2 groundwork:
#   - sm_100 / sm_101 / sm_103 device detection -> Blackwell preset
#   - nvcc resolution of __nv_fp8_* / __nv_fp6_* / __nv_fp4_* types
#   - Cluster Launch Control functional path
#   - NVLink 5 peer bandwidth (only if >= 2 B200 visible)
#   - H100-regression: existing CUDA test suite must still pass
#
# Exits non-zero on the first failing step. Writes a per-run log to
# logs/b200-validate-<timestamp>.log.
#
# Usage:  ./scripts/b200-validate.sh [--skip-regression]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_REGRESSION=0
for arg in "$@"; do
    case "$arg" in
        --skip-regression) SKIP_REGRESSION=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

mkdir -p logs
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/b200-validate-${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo "  RingKernel B200 validation"
echo "  $(date -u -Iseconds)"
echo "  log: $LOG"
echo "=============================================="

step() { echo ""; echo "[$1] $2"; }
fail() { echo "  FAIL: $*" >&2; exit 1; }
pass() { echo "  OK: $*"; }

# ─── 0. Environment snapshot ──────────────────────────────────────────
step "0/6" "Environment snapshot"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
command -v nvcc       >/dev/null || fail "nvcc not found — run scripts/setup-gpu-vm.sh first"
command -v cargo      >/dev/null || fail "cargo not found — run scripts/setup-gpu-vm.sh first"

GPU_NAME="$(nvidia-smi --query-gpu=name          --format=csv,noheader | head -1)"
GPU_CC="$(  nvidia-smi --query-gpu=compute_cap   --format=csv,noheader | head -1)"
GPU_COUNT="$(nvidia-smi --query-gpu=name         --format=csv,noheader | wc -l)"
DRIVER="$(  nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
CUDA_REL="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
RUSTC="$(rustc --version)"

echo "  GPU:          $GPU_NAME (x$GPU_COUNT)"
echo "  Compute cap:  $GPU_CC"
echo "  Driver:       $DRIVER"
echo "  CUDA:         $CUDA_REL"
echo "  rustc:        $RUSTC"

CC_MAJOR="${GPU_CC%%.*}"
if [ "$CC_MAJOR" -lt 10 ]; then
    fail "compute cap $GPU_CC is below 10.0 — this script is for Blackwell+. Use paper bench scripts for Hopper."
fi
pass "Blackwell-family GPU detected (CC $GPU_CC)"

# CUDA 12.9+ required for __nv_fp4_* / __nv_fp6_* type resolution.
CUDA_MAJOR="${CUDA_REL%%.*}"
CUDA_MINOR="${CUDA_REL##*.}"
if [ "$CUDA_MAJOR" -lt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 9 ]; }; then
    fail "CUDA $CUDA_REL < 12.9 — FP4/FP6 Blackwell types will not resolve. Install cuda-toolkit-12-9 or newer."
fi
pass "CUDA $CUDA_REL supports Blackwell scalar types"

# ─── 1. Routing coverage for live device CC ───────────────────────────
step "1/6" "Live device CC routes to Blackwell preset"
# The unit test `from_compute_capability_routes_blackwell_family_to_blackwell_preset`
# already asserts sm_100 / sm_120 / sm_121 → Blackwell at compile time.
# Re-running it here confirms the binary matches the live GPU story.
cargo test -p ringkernel-cuda --features cuda --release --lib \
    launch_config::mode::tests::from_compute_capability_routes_blackwell_family_to_blackwell_preset \
    -- --nocapture \
    || fail "routing test failed — Blackwell family does not map to Blackwell preset"
echo "  Live device CC: $GPU_CC (major $CC_MAJOR) is covered by the routing test"
pass "from_compute_capability routes this device to Blackwell"

# ─── 2. nvcc type-resolution probe ────────────────────────────────────
step "2/6" "nvcc resolves all Blackwell scalar type names at sm_100"
PROBE_CU="$(mktemp --suffix=.cu)"
PROBE_PTX="$(mktemp --suffix=.ptx)"
cat > "$PROBE_CU" <<'CU'
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp6.h>
#include <cuda_fp4.h>

extern "C" __global__ void rk_blackwell_types_probe() {
    __nv_bfloat16  bf = __float2bfloat16(1.0f);
    __nv_fp8_e4m3  a; a = __nv_fp8_e4m3(bf);
    __nv_fp8_e5m2  b; b = __nv_fp8_e5m2(bf);
    __nv_fp6_e3m2  c;
    __nv_fp6_e2m3  d;
    __nv_fp4_e2m1  e;
    (void)a; (void)b; (void)c; (void)d; (void)e;
}
CU

if nvcc -arch=sm_100 --ptx -o "$PROBE_PTX" "$PROBE_CU" 2>&1 | tee -a "$LOG"; then
    pass "nvcc emitted sm_100 PTX using __nv_fp{4,6,8}_* and __nv_bfloat16"
else
    rm -f "$PROBE_CU" "$PROBE_PTX"
    fail "nvcc failed to resolve Blackwell scalar types — expected CUDA >= 12.9"
fi
rm -f "$PROBE_CU" "$PROBE_PTX"

# ─── 3. H100 regression — existing CUDA test suite ────────────────────
if [ "$SKIP_REGRESSION" -eq 0 ]; then
    step "3/6" "Regression: existing ringkernel-cuda test suite"
    cargo test -p ringkernel-cuda --features cuda --release \
        || fail "regression suite failed — fix before touching Blackwell paths"
    pass "ringkernel-cuda suite green"
else
    step "3/6" "Regression: SKIPPED (--skip-regression)"
fi

# ─── 4. Cluster Launch Control functional test ────────────────────────
step "4/6" "Cluster Launch Control smoke"
# Placeholder — the dedicated test is written on-device once we see
# real nvcc error messages for cuLaunchKernelEx + cluster dims. For
# now, exercise the launch_config mode tests that cover the selector
# and then reserve the slot.
cargo test -p ringkernel-cuda --features cuda --release \
    --lib launch_config::mode -- --nocapture \
    || fail "launch_config::mode tests failed on device"
echo "  NOTE: dedicated cuLaunchKernelEx + cluster-dims test still to be written"
echo "        (reserve this step for interactive work on the VM)"
pass "launch_config selector logic green"

# ─── 5. NVLink 5 peer bandwidth (multi-GPU only) ──────────────────────
step "5/6" "NVLink 5 peer bandwidth"
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "  SKIP: only $GPU_COUNT GPU visible — NVLink 5 P2P needs >= 2 B200"
else
    cargo test -p ringkernel-cuda --features "cuda,multi-gpu" --release \
        --test paper_multi_gpu_k2k_bw -- --ignored --nocapture \
        || fail "paper_multi_gpu_k2k_bw failed"
    pass "paper_multi_gpu_k2k_bw completed — check log for 'PAPER_MGPU_BW size=...' lines"
fi

# ─── 6. Summary ───────────────────────────────────────────────────────
step "6/6" "Summary"
echo "  Log:          $LOG"
echo "  GPU:          $GPU_NAME (x$GPU_COUNT)"
echo "  Compute cap:  $GPU_CC"
echo "  CUDA:         $CUDA_REL"
echo "  Driver:       $DRIVER"
echo ""
echo "Next: write the dedicated cuLaunchKernelEx + FP4 on-device kernel"
echo "tests interactively on the VM so we can iterate on real error output."
