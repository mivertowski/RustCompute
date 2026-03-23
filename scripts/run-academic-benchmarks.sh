#!/usr/bin/env bash
# RingKernel Academic Benchmark Suite
# Produces paper-quality results with statistical rigor.
#
# Prerequisites:
#   - NVIDIA GPU with CUDA toolkit
#   - Exclusive compute mode: sudo nvidia-smi -c EXCLUSIVE_PROCESS
#   - Locked clocks: sudo nvidia-smi -lgc <max_clock>
#   - Performance CPU governor: sudo cpupower frequency-set -g performance
#
# Output: benchmark_results/ directory with CSV, JSON, and summary files

set -euo pipefail

RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_DIR}/${TIMESTAMP}"

echo "=============================================="
echo "  RingKernel Academic Benchmark Suite"
echo "  $(date)"
echo "=============================================="

# ─── Pre-flight checks ───────────────────────────────────────────────
echo ""
echo "[Pre-flight] Checking environment..."

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA GPU required."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
GPU_CLOCK=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader | head -1)
CUDA_VER=$(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',' || echo "unknown")
RUST_VER=$(rustc --version)
GIT_COMMIT=$(git rev-parse --short HEAD)

echo "  GPU:          ${GPU_NAME}"
echo "  Compute Cap:  ${GPU_CC}"
echo "  Memory:       ${GPU_MEM}"
echo "  Clock:        ${GPU_CLOCK}"
echo "  CUDA:         ${CUDA_VER}"
echo "  Rust:         ${RUST_VER}"
echo "  Commit:       ${GIT_COMMIT}"

# ─── Setup ────────────────────────────────────────────────────────────
mkdir -p "${RUN_DIR}"

# Save system info
cat > "${RUN_DIR}/system_info.json" << SYSEOF
{
  "gpu": "${GPU_NAME}",
  "driver": "${GPU_DRIVER}",
  "compute_capability": "${GPU_CC}",
  "gpu_memory": "${GPU_MEM}",
  "gpu_clock": "${GPU_CLOCK}",
  "cuda": "${CUDA_VER}",
  "rust": "${RUST_VER}",
  "git_commit": "${GIT_COMMIT}",
  "ringkernel_cuda_arch": "${RINGKERNEL_CUDA_ARCH:-auto}",
  "timestamp": "$(date -Iseconds)",
  "hostname": "$(hostname)"
}
SYSEOF

nvidia-smi -q > "${RUN_DIR}/nvidia_smi_full.txt"
cat /proc/cpuinfo > "${RUN_DIR}/cpuinfo.txt"
free -h > "${RUN_DIR}/meminfo.txt"
uname -a > "${RUN_DIR}/uname.txt"

echo "  Results dir:  ${RUN_DIR}"

# ─── Build ────────────────────────────────────────────────────────────
echo ""
echo "[Build] Compiling with CUDA (release)..."
cargo build --workspace --features cuda --release 2>&1 | tail -1
cargo build -p ringkernel-cuda --features "cuda,cooperative" --release 2>&1 | tail -1

# ─── Warmup ───────────────────────────────────────────────────────────
echo ""
echo "[Warmup] 60-second GPU warmup..."
# Run a quick benchmark to warm up GPU (thermal stabilization)
timeout 60 cargo bench --package ringkernel -- serialization --warm-up-time 30 2>/dev/null || true
echo "  Warmup complete."

# ─── Experiment 1: Core microbenchmarks (Criterion) ───────────────────
echo ""
echo "[Experiment 1] Core microbenchmarks (Criterion)..."
cargo bench --package ringkernel 2>&1 | tee "${RUN_DIR}/criterion_output.txt"
# Criterion saves detailed results in target/criterion/
cp -r target/criterion "${RUN_DIR}/criterion_reports" 2>/dev/null || true

# ─── Experiment 2: GPU execution verification ─────────────────────────
echo ""
echo "[Experiment 2] GPU execution tests..."
cargo test -p ringkernel-cuda --features cuda --release -- --ignored 2>&1 | tee "${RUN_DIR}/gpu_tests.txt"
cargo test -p ringkernel-cuda --features "cuda,cooperative" --release --test gpu_execution_verify 2>&1 | tee -a "${RUN_DIR}/gpu_tests.txt"

# ─── Experiment 3: WaveSim3D benchmarks ───────────────────────────────
echo ""
echo "[Experiment 3] WaveSim3D application benchmark..."
if cargo build -p ringkernel-wavesim3d --bin wavesim3d-benchmark --release --features cuda-codegen 2>/dev/null; then
    for trial in $(seq 1 3); do
        echo "  Trial ${trial}/3..."
        cargo run -p ringkernel-wavesim3d --bin wavesim3d-benchmark --release --features cuda-codegen 2>&1 \
            | tee "${RUN_DIR}/wavesim3d_trial${trial}.txt"
    done
else
    echo "  SKIP: wavesim3d-benchmark build failed (cuda-codegen feature required)"
fi

# ─── Experiment 4: TxMon benchmarks ──────────────────────────────────
echo ""
echo "[Experiment 4] TxMon application benchmark..."
if cargo build -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen 2>/dev/null; then
    for trial in $(seq 1 3); do
        echo "  Trial ${trial}/3..."
        cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen 2>&1 \
            | tee "${RUN_DIR}/txmon_trial${trial}.txt"
    done
else
    echo "  SKIP: txmon-benchmark build failed"
fi

# ─── Experiment 5: ProcInt benchmarks ────────────────────────────────
echo ""
echo "[Experiment 5] ProcInt application benchmark..."
if cargo build -p ringkernel-procint --bin procint-benchmark --release 2>/dev/null; then
    for trial in $(seq 1 3); do
        echo "  Trial ${trial}/3..."
        cargo run -p ringkernel-procint --bin procint-benchmark --release 2>&1 \
            | tee "${RUN_DIR}/procint_trial${trial}.txt"
    done
else
    echo "  SKIP: procint-benchmark build failed"
fi

# ─── Summary ──────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Benchmark suite complete!"
echo "  Results:   ${RUN_DIR}/"
echo "  Criterion: ${RUN_DIR}/criterion_reports/"
echo "  GPU tests: ${RUN_DIR}/gpu_tests.txt"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Review results in ${RUN_DIR}/"
echo "  2. Fill in docs/benchmarks/h100-b200-baseline.md"
echo "  3. Export Criterion HTML: open target/criterion/report/index.html"
echo "  4. Archive: tar czf benchmark_results_${TIMESTAMP}.tar.gz ${RUN_DIR}/"
