#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-./out}"; mkdir -p "${OUT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
FIGDATA="${REPO_ROOT}/docs/paper/figures/data"
cd "${REPO_ROOT}"

CSV="${OUT}/sustained.csv"

for trial in 1 2 3 4; do
    LOG="${OUT}/trial${trial}.log"
    echo "==> Trial ${trial}/4 (60s sustained)..."
    cargo test -p ringkernel-cuda --features cuda --release \
        --test sustained_throughput test_sustained_throughput_60s \
        -- --ignored --nocapture 2>&1 | tee "${LOG}"
done

python3 "$(dirname "${BASH_SOURCE[0]}")/extract.py" \
    "${OUT}"/trial*.log "--wide=${FIGDATA}" > "${CSV}"
echo "==> Done: ${CSV} + ${FIGDATA}/sustained_t{1,2,3,4}.csv"
