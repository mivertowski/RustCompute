#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-./out}"; mkdir -p "${OUT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
FIGDATA="${REPO_ROOT}/docs/paper/figures/data"
cd "${REPO_ROOT}"
LOG="${OUT}/raw.log"
CSV="${OUT}/snapshot_restart.csv"
echo "==> paper_snapshot_restart (~15 min)..."
cargo test -p ringkernel-cuda --features cuda --release \
    --test paper_snapshot_restart -- --ignored --nocapture 2>&1 | tee "${LOG}"
python3 "$(dirname "${BASH_SOURCE[0]}")/extract.py" "${LOG}" "${FIGDATA}" > "${CSV}"
echo "==> Done: ${CSV} + ${FIGDATA}/snapshot_restart_wide.csv"
