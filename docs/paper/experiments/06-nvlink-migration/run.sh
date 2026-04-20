#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-./out}"; mkdir -p "${OUT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
FIGDATA="${REPO_ROOT}/docs/paper/figures/data"
cd "${REPO_ROOT}"
LOG="${OUT}/raw.log"
CSV="${OUT}/migration.csv"

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
if [[ "${GPU_COUNT}" -lt 2 ]]; then
    echo "SKIP: requires >= 2 GPUs (found ${GPU_COUNT})" | tee "${LOG}"
    echo "path,size_bytes,n,mean_ns,stddev_ns,p50_ns,p95_ns,p99_ns,ci_halfwidth" > "${CSV}"
    exit 0
fi

echo "==> paper_nvlink_migration (~30 min)..."
cargo test -p ringkernel-cuda --features "cuda,multi-gpu" --release \
    --test paper_nvlink_migration -- --ignored --nocapture 2>&1 | tee "${LOG}"
python3 "$(dirname "${BASH_SOURCE[0]}")/extract.py" "${LOG}" "${FIGDATA}" > "${CSV}"
echo "==> Done: ${CSV} + ${FIGDATA}/migration_{p2p,host}.csv"
