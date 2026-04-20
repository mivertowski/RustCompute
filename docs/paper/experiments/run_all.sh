#!/usr/bin/env bash
# Top-level runner for the six paper experiments.
# See RUNBOOK.md for the prerequisites (clock locking, exclusive mode).

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS="${EXP_DIR}/results/${TIMESTAMP}"

ARCHIVE_ONLY=0
SKIP_LIST=""
for arg in "$@"; do
    case "$arg" in
        --archive-only)  ARCHIVE_ONLY=1 ;;
        --skip=*)        SKIP_LIST="${arg#--skip=}" ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--archive-only] [--skip=N1,N2,...]

  --archive-only      Skip experiments; just collect existing results.
  --skip=N1,N2,...    Skip experiments by number (comma-separated).

Experiments (numbered to match RUNBOOK.md):
  1: K2K tier latency           (~25 min, 1 GPU)
  2: Snapshot/restart           (~15 min, 1 GPU)
  3: Lifecycle overhead         (~15 min, 1 GPU)
  4: Sustained timeseries       (~70 min, 1 GPU)
  5: TLC state-space stats      (~20 min, 0 GPUs)
  6: NVLink P2P migration       (~30 min, 2 GPUs)

Results land in ${EXP_DIR}/results/<timestamp>/.
EOF
            exit 0 ;;
        *) echo "unknown arg: $arg"; exit 1 ;;
    esac
done

skipped() { case ",${SKIP_LIST}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

mkdir -p "${RESULTS}"
echo "==> Results directory: ${RESULTS}"

# ─── Manifest ────────────────────────────────────────────────────────
echo "==> Capturing system manifest"
cat > "${RESULTS}/manifest.json" <<EOF
{
  "timestamp":     "$(date -Iseconds)",
  "hostname":      "$(hostname)",
  "git_commit":    "$(cd "${REPO_ROOT}" && git rev-parse HEAD)",
  "git_describe":  "$(cd "${REPO_ROOT}" && git describe --tags --always --dirty)",
  "rust_version":  "$(rustc --version)",
  "cuda_version":  "$(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',' || echo unknown)",
  "driver":        "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)",
  "gpu_name":      "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
  "gpu_count":     $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
}
EOF

nvidia-smi -q                                        > "${RESULTS}/nvidia_smi_full.txt" 2>&1
nvidia-smi nvlink --status                           > "${RESULTS}/nvlink_status.txt" 2>&1 || true
nvidia-smi topo -m                                   > "${RESULTS}/nvidia_topo.txt" 2>&1 || true
nvidia-smi --query-gpu=ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total \
    --format=csv                                     > "${RESULTS}/ecc_pre.csv"

if [[ "${ARCHIVE_ONLY}" -eq 1 ]]; then
    echo "==> Archive-only mode; skipping experiments."
else

# ─── Build (warm cache for all experiments) ──────────────────────────
echo "==> Building (release, cuda+cooperative+multi-gpu)"
cd "${REPO_ROOT}"
cargo build --workspace --features cuda --release --exclude ringkernel-txmon \
    2>&1 | tail -3 | tee "${RESULTS}/build.log"
cargo build -p ringkernel-cuda --features "cuda,cooperative,multi-gpu" --release \
    2>&1 | tail -3 | tee -a "${RESULTS}/build.log"

# ─── Experiment dispatcher ───────────────────────────────────────────
run_exp() {
    local n="$1"; local name="$2"; local dir="$3"
    if skipped "$n"; then echo "==> Skipping experiment $n ($name)"; return; fi
    echo ""
    echo "============================================================"
    echo "  Experiment $n: $name"
    echo "============================================================"
    local out="${RESULTS}/exp${n}_${name}"
    mkdir -p "$out"
    if [[ -x "${EXP_DIR}/${dir}/run.sh" ]]; then
        ( cd "${REPO_ROOT}" && "${EXP_DIR}/${dir}/run.sh" "$out" ) 2>&1 | tee "${out}/stdout.log"
    else
        echo "ERROR: ${EXP_DIR}/${dir}/run.sh not found or not executable"
        return 1
    fi
}

run_exp 5 "tlc-stats"            "05-tlc-stats"
run_exp 1 "tier-latency"         "01-tier-latency"
run_exp 2 "snapshot-restart"     "02-snapshot-restart"
run_exp 3 "lifecycle-overhead"   "03-lifecycle-overhead"
run_exp 4 "sustained-timeseries" "04-sustained-timeseries"
run_exp 6 "nvlink-migration"     "06-nvlink-migration"

fi # archive-only

# ─── Post-flight ─────────────────────────────────────────────────────
nvidia-smi --query-gpu=ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total \
    --format=csv                                     > "${RESULTS}/ecc_post.csv"

# ─── Archive ──────────────────────────────────────────────────────────
ARCHIVE="${EXP_DIR}/results/results_${TIMESTAMP}.tar.gz"
tar czf "${ARCHIVE}" -C "${EXP_DIR}/results" "${TIMESTAMP}"
echo ""
echo "==> Archive: ${ARCHIVE}"

# ─── Summary ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Run complete."
echo "  Results dir: ${RESULTS}"
echo "  Archive:     ${ARCHIVE}"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Copy CSVs into docs/paper/figures/data/"
echo "  2. Build figures: cd docs/paper && make"
echo "  3. Diff ECC counters: diff ${RESULTS}/ecc_pre.csv ${RESULTS}/ecc_post.csv"
echo "     (any non-zero delta invalidates the run)"
