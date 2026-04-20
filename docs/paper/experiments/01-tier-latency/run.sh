#!/usr/bin/env bash
# Experiment 1: K2K Tier Latency
set -euo pipefail
OUT="${1:-./out}"; mkdir -p "${OUT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
FIGDATA="${REPO_ROOT}/docs/paper/figures/data"
cd "${REPO_ROOT}"

LOG="${OUT}/raw.log"
CSV="${OUT}/tier_latency.csv"

echo "==> paper_tier_latency (~25 min)..."
cargo test -p ringkernel-cuda --features "cuda,cooperative" --release \
    --test paper_tier_latency -- --ignored --nocapture 2>&1 | tee "${LOG}"

echo "==> Extracting CSVs (long + per-tier)..."
python3 "$(dirname "${BASH_SOURCE[0]}")/extract.py" "${LOG}" "${FIGDATA}" > "${CSV}"
echo "    long form: ${CSV}"
echo "    figure data: ${FIGDATA}/tier_latency_{smem,dsmem,hbm}.csv"
