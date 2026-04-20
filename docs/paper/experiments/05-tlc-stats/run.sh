#!/usr/bin/env bash
# Experiment 5: TLC state-space statistics.
# Runs each TLA+ spec under its .cfg, parses TLC's summary, writes CSV.

set -euo pipefail

OUT="${1:-./out}"
mkdir -p "${OUT}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
VER_DIR="${REPO_ROOT}/docs/verification"

CSV="${OUT}/tlc_stats.csv"
echo "spec,bound_constants,distinct_states,states_generated,wall_clock_s,invariants_checked,result" > "${CSV}"

# Specs to run. Each spec's .cfg is in ../verification/.
SPECS=(actor_lifecycle hlc k2k_delivery tenant_isolation migration multi_gpu_k2k)

# Resolve TLC: prefer tla2tools.jar, fall back to docker.
if [[ -n "${TLC_CMD:-}" ]]; then
    TLC="${TLC_CMD}"
elif [[ -f /opt/tla2tools.jar ]]; then
    TLC="java -XX:+UseParallelGC -jar /opt/tla2tools.jar"
elif command -v docker &>/dev/null; then
    TLC="docker run --rm -v ${VER_DIR}:/spec -w /spec tlaplus/tlaplus java -jar /opt/TLA+Tools/tla2tools.jar"
else
    echo "ERROR: no TLC available. Install java + tla2tools.jar OR docker."
    echo "Then re-run with TLC_CMD env if needed."
    exit 1
fi

cd "${VER_DIR}"

for spec in "${SPECS[@]}"; do
    [[ -f "${spec}.tla" ]] || { echo "WARN: ${spec}.tla missing, skipping"; continue; }
    [[ -f "${spec}.cfg" ]] || { echo "WARN: ${spec}.cfg missing, skipping"; continue; }

    echo "==> Running TLC on ${spec}.tla..."
    LOG="${OUT}/tlc_${spec}.log"
    START=$(date +%s)
    set +e
    ${TLC} -config "${spec}.cfg" "${spec}.tla" > "${LOG}" 2>&1
    RC=$?
    set -e
    ELAPSED=$(($(date +%s) - START))

    # Parse TLC summary.
    distinct=$(grep -oE "[0-9]+ distinct states" "${LOG}" | head -1 | awk '{print $1}' | tr -d ',' || echo 0)
    generated=$(grep -oE "[0-9]+ states generated" "${LOG}" | head -1 | awk '{print $1}' | tr -d ',' || echo 0)
    invariants=$(grep -cE "^Invariant " "${spec}.cfg" || echo 0)
    bounds=$(grep -E "^CONSTANTS|^CONSTANT" "${spec}.cfg" | tr '\n' ',' | sed 's/CONSTANTS\?//g; s/  */ /g; s/^ //; s/,$//')

    if [[ ${RC} -eq 0 ]] && grep -q "Model checking completed" "${LOG}"; then
        result="OK"
    elif grep -q "Invariant.*violated\|FAILED\|Error" "${LOG}"; then
        result="FAIL"
    else
        result="UNKNOWN"
    fi

    echo "${spec},\"${bounds}\",${distinct},${generated},${ELAPSED},${invariants},${result}" >> "${CSV}"
    echo "    distinct=${distinct} generated=${generated} elapsed=${ELAPSED}s result=${result}"
done

echo ""
echo "==> CSV: ${CSV}"
echo "==> Logs: ${OUT}/tlc_*.log"
