#!/usr/bin/env bash
# Runs TLC model checker on all TLA+ specs in this directory.
#
# Usage:
#   ./tlc.sh                  # run every spec
#   ./tlc.sh hlc              # run only hlc.tla
#   ./tlc.sh hlc migration    # run a subset
#
# Environment:
#   TLC_JAR     absolute path to tla2tools.jar (defaults to ./tla2tools.jar)
#   TLC_CMD     override the full Java command (for docker or custom JVM)
#   TLC_WORKERS number of worker threads (defaults to number of CPUs)
#
# Exit status is non-zero on the first failing spec.

set -euo pipefail

cd "$(dirname "$0")"

TLC_JAR="${TLC_JAR:-$PWD/tla2tools.jar}"
TLC_WORKERS="${TLC_WORKERS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

if [[ -n "${TLC_CMD:-}" ]]; then
    TLC_RUN=("$TLC_CMD")
else
    if ! command -v java >/dev/null 2>&1; then
        echo "error: java not on PATH. Install Java 11+ or set TLC_CMD." >&2
        exit 2
    fi
    if [[ ! -f "$TLC_JAR" ]]; then
        cat >&2 <<EOF
error: $TLC_JAR not found.

Download from https://github.com/tlaplus/tlaplus/releases/latest and
place tla2tools.jar alongside this script, or set TLC_JAR to its path,
or set TLC_CMD to a docker invocation such as:

  export TLC_CMD="docker run --rm -v \$PWD:/spec -w /spec tlaplus/tlaplus \\
      java -jar /opt/TLA+Tools/tla2tools.jar"
EOF
        exit 2
    fi
    TLC_RUN=(java -XX:+UseParallelGC -jar "$TLC_JAR")
fi

ALL_SPECS=(hlc k2k_delivery migration multi_gpu_k2k tenant_isolation actor_lifecycle)

if [[ $# -gt 0 ]]; then
    SPECS=("$@")
else
    SPECS=("${ALL_SPECS[@]}")
fi

fail=0
for spec in "${SPECS[@]}"; do
    tla="${spec}.tla"
    cfg="${spec}.cfg"
    if [[ ! -f "$tla" || ! -f "$cfg" ]]; then
        echo "skip: $spec (missing $tla or $cfg)" >&2
        continue
    fi
    echo "===================================================================="
    echo "Running TLC on $tla (config: $cfg, workers: $TLC_WORKERS)"
    echo "===================================================================="
    if ! "${TLC_RUN[@]}" tlc2.TLC -workers "$TLC_WORKERS" -config "$cfg" "$tla"; then
        echo "FAILED: $spec" >&2
        fail=1
    else
        echo "OK: $spec"
    fi
    echo
done

exit "$fail"
