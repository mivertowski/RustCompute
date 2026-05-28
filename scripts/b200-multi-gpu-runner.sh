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
