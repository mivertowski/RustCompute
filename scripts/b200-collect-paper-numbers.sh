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
