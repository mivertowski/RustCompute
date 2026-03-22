#!/usr/bin/env bash
# RingKernel GPU VM Setup Script
# Target: Azure ND H100 v5 (or any NVIDIA GPU VM with Ubuntu 22.04/24.04)
#
# Usage:
#   curl -sSf https://raw.githubusercontent.com/.../scripts/setup-gpu-vm.sh | bash
#   OR
#   chmod +x setup-gpu-vm.sh && ./setup-gpu-vm.sh
#
# Prerequisites:
#   - Ubuntu 22.04 or 24.04
#   - NVIDIA GPU (H100/A100/etc.)
#   - sudo access

set -euo pipefail

CUDA_VERSION="12.6"  # Latest stable with CUDA 13.x compat
RUST_TOOLCHAIN="stable"

echo "=============================================="
echo "  RingKernel GPU VM Setup"
echo "  Target: H100/B200 validation"
echo "=============================================="

# ─── 1. System update ────────────────────────────────────────────────
echo ""
echo "[1/8] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    curl \
    wget \
    htop \
    tmux \
    cmake \
    linux-headers-$(uname -r)

# ─── 2. NVIDIA driver (if not already installed) ─────────────────────
echo ""
echo "[2/8] Checking NVIDIA driver..."
if command -v nvidia-smi &>/dev/null; then
    echo "  NVIDIA driver already installed:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
else
    echo "  Installing NVIDIA driver..."
    sudo apt-get install -y -qq nvidia-driver-550
    echo "  ⚠ REBOOT REQUIRED after driver install. Re-run this script after reboot."
    exit 0
fi

# ─── 3. CUDA Toolkit ─────────────────────────────────────────────────
echo ""
echo "[3/8] Checking CUDA toolkit..."
if command -v nvcc &>/dev/null; then
    echo "  CUDA already installed: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
else
    echo "  Installing CUDA toolkit ${CUDA_VERSION}..."
    # Use NVIDIA's package repo
    DISTRO="ubuntu$(lsb_release -rs | tr -d '.')"
    ARCH="x86_64"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt-get update -qq
    sudo apt-get install -y -qq "cuda-toolkit-${CUDA_VERSION/./-}"
    rm /tmp/cuda-keyring.deb

    # Add to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
fi

# ─── 4. Rust toolchain ───────────────────────────────────────────────
echo ""
echo "[4/8] Checking Rust toolchain..."
if command -v rustup &>/dev/null; then
    echo "  Rust already installed: $(rustc --version)"
    rustup update $RUST_TOOLCHAIN --no-self-update 2>/dev/null || true
else
    echo "  Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_TOOLCHAIN
    source "$HOME/.cargo/env"
fi

# Ensure clippy and rustfmt
rustup component add clippy rustfmt 2>/dev/null || true

# ─── 5. Clone or update repo ─────────────────────────────────────────
echo ""
echo "[5/8] Setting up RingKernel repository..."
REPO_DIR="${HOME}/RingKernel"

if [ -d "$REPO_DIR" ]; then
    echo "  Repository exists, pulling latest..."
    cd "$REPO_DIR"
    git pull --ff-only || echo "  ⚠ Pull failed, using existing state"
else
    echo "  Cloning repository..."
    git clone https://github.com/RustCompute/RustCompute.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

# ─── 6. Build with CUDA ──────────────────────────────────────────────
echo ""
echo "[6/8] Building RingKernel with CUDA support..."

# Detect GPU architecture
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | sed 's/\(.\)\(.\)/sm_\1\2/')
echo "  Detected GPU architecture: ${GPU_ARCH}"
export RINGKERNEL_CUDA_ARCH="${GPU_ARCH}"

# Build workspace with CUDA
cargo build --workspace --features cuda --release 2>&1 | tail -3

# Build with cooperative groups
echo "  Building cooperative groups support..."
cargo build -p ringkernel-cuda --features "cuda,cooperative" --release 2>&1 | tail -3

echo "  Build complete."

# ─── 7. Run tests ────────────────────────────────────────────────────
echo ""
echo "[7/8] Running test suite..."

# CPU tests (should all pass)
echo "  Running CPU tests..."
cargo test --workspace --release 2>&1 | grep -E "^test result:" | awk '{passed+=$4; failed+=$6; ignored+=$8} END {print "  CPU tests: " passed " passed, " failed " failed, " ignored " ignored"}'

# GPU tests (previously ignored)
echo "  Running CUDA GPU tests..."
cargo test -p ringkernel-cuda --features cuda --release -- --ignored 2>&1 | grep -E "^test result:" | awk '{passed+=$4; failed+=$6; ignored+=$8} END {print "  GPU tests: " passed " passed, " failed " failed, " ignored " ignored"}'

# CUDA codegen GPU tests
echo "  Running CUDA codegen GPU tests..."
cargo test -p ringkernel-cuda --features "cuda,cooperative" --release --test gpu_execution_verify 2>&1 | grep -E "^test result:" | tail -1

# ─── 8. System info ──────────────────────────────────────────────────
echo ""
echo "[8/8] System information:"
echo "=============================================="
echo "  GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Driver:       $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "  CUDA:         $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
echo "  Compute Cap:  $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
echo "  Memory:       $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo "  Rust:         $(rustc --version)"
echo "  OS:           $(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
echo "  CPU:          $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "  RAM:          $(free -h | grep Mem | awk '{print $2}')"
echo "  Arch target:  ${RINGKERNEL_CUDA_ARCH}"
echo "=============================================="

echo ""
echo "Setup complete! Next steps:"
echo ""
echo "  # Run H100 baselines:"
echo "  cd ${REPO_DIR}"
echo "  export RINGKERNEL_CUDA_ARCH=${GPU_ARCH}"
echo "  cargo bench --package ringkernel --release"
echo ""
echo "  # Run application benchmarks:"
echo "  cargo run -p ringkernel-wavesim3d --bin wavesim3d-benchmark --release --features cuda-codegen"
echo "  cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen"
echo "  cargo run -p ringkernel-procint --bin procint-benchmark --release"
echo ""
echo "  # Lock GPU clocks for consistent benchmarks:"
echo "  sudo nvidia-smi -lgc \$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1)"
echo ""
echo "  # Set exclusive compute mode:"
echo "  sudo nvidia-smi -c EXCLUSIVE_PROCESS"
