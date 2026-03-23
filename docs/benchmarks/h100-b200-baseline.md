# H100/B200 Baseline Benchmarks — Academic Edition

> Paper-quality benchmark results for validating the GPU-native persistent actor paradigm.
> See `METHODOLOGY.md` for statistical protocol and experimental design.

## System Configuration

| Property | Value |
|----------|-------|
| GPU | |
| GPU Memory | |
| Driver Version | |
| CUDA Toolkit | |
| Compute Capability | |
| GPU Clock (locked) | |
| Memory Clock | |
| ECC Status | |
| CPU | |
| RAM | |
| OS / Kernel | |
| Rust Version | |
| cudarc Version | 0.19.3 |
| RingKernel Version | |
| Git Commit | |
| RINGKERNEL_CUDA_ARCH | |

## Prerequisites

```bash
export RINGKERNEL_CUDA_ARCH=sm_90  # H100
cargo build --workspace --features cuda --release
cargo build -p ringkernel-cuda --features "cuda,cooperative" --release

# Lock clocks and set exclusive mode
sudo nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits)
sudo nvidia-smi -c EXCLUSIVE_PROCESS
sudo cpupower frequency-set -g performance 2>/dev/null || true

# 5-minute GPU warmup
cargo bench --package ringkernel -- --warm-up-time 300 2>/dev/null || true
```

---

## Experiment 1: Launch Overhead — The Fundamental Advantage

**Claim**: Persistent actor injection latency is 3-4 orders of magnitude lower than traditional kernel launch.

**Protocol**: 10,000 commands × 10 trials. CUDA event timing (device-side).

### Traditional Model (cudaMemcpy + cudaLaunchKernel + cudaSync)

| Statistic | Value | Unit |
|-----------|-------|------|
| n | | |
| Mean | | us |
| Median | | us |
| Std Dev | | us |
| 95% CI | [, ] | us |
| p50 / p95 / p99 / p99.9 | / / / | us |
| Outliers (MAD > 3.5) | | |

### Persistent Actor Model (mapped memory store)

| Statistic | Value | Unit |
|-----------|-------|------|
| n | | |
| Mean | | us |
| Median | | us |
| Std Dev | | us |
| 95% CI | [, ] | us |
| p50 / p95 / p99 / p99.9 | / / / | us |
| Outliers (MAD > 3.5) | | |

### Comparison

| Metric | Value |
|--------|-------|
| Speedup (mean) | x |
| Speedup 95% CI | [, ] x |
| Cohen's d | (effect size) |
| Welch's t-test p-value | |
| Statistically significant | YES/NO |

---

## Experiment 2: Message Passing Throughput

**Claim**: Lock-free K2K messaging achieves near-memory-bandwidth throughput.

### Throughput by Payload Size

| Payload | Messages/sec | Bandwidth (GB/s) | Queue Utilization |
|---------|-------------|-------------------|-------------------|
| 64 B | | | |
| 256 B | | | |
| 1 KB | | | |
| 4 KB | | | |

### K2K vs Host-Mediated Comparison

| Method | Latency (us) | Throughput (Mmsg/s) | Cohen's d | p-value |
|--------|-------------|---------------------|-----------|---------|
| K2K (GPU-direct) | | | | |
| Host-mediated | | | | |
| Speedup | | x | | |

---

## Experiment 3: Actor Scalability

**Claim**: Actor throughput scales linearly with actor count up to cooperative group limits.

### Strong Scaling (fixed total work = 1M messages)

| Actors | Time (ms) | Throughput (Mmsg/s) | Efficiency | Grid Sync (us) |
|--------|----------|---------------------|------------|-----------------|
| 1 | | | 1.00 | |
| 4 | | | | |
| 16 | | | | |
| 64 | | | | |
| 256 | | | | |
| 1024 | | | | |

### Weak Scaling (1M messages per actor)

| Actors | Time (ms) | Per-Actor Throughput | Efficiency |
|--------|----------|---------------------|------------|
| 1 | | | 1.00 |
| 4 | | | |
| 16 | | | |
| 64 | | | |
| 256 | | | |
| 1024 | | | |

---

## Experiment 4: Sustained Throughput Under Load

**Claim**: Persistent actors maintain stable throughput over 60 seconds without degradation.

**Protocol**: 60-second continuous run at 90% of peak. 1-second measurement windows.

| Window | Throughput (Mmsg/s) | p50 Latency (us) | p99 Latency (us) | Memory (MB) |
|--------|---------------------|-------------------|-------------------|-------------|
| 0-5s | | | | |
| 5-10s | | | | |
| 10-20s | | | | |
| 20-30s | | | | |
| 30-40s | | | | |
| 40-50s | | | | |
| 50-60s | | | | |

| Metric | Value |
|--------|-------|
| Mean throughput | Mmsg/s |
| CV (throughput) | % |
| Max p99 spike | us |
| Memory growth | MB |
| Degradation | YES/NO |

---

## Experiment 5: Mixed Workload — The Real-World Case

**Claim**: Persistent actors outperform traditional re-launch under mixed workloads.

### Workload Profiles

| Profile | Compute | Inject | Query | Trad. (ops/s) | Persist. (ops/s) | Speedup |
|---------|---------|--------|-------|---------------|------------------|---------|
| Compute-heavy (90/5/5) | 90% | 5% | 5% | | | x |
| Balanced (33/33/33) | 33% | 33% | 33% | | | x |
| Comm-heavy (10/45/45) | 10% | 45% | 45% | | | x |

**Key insight**: Traditional model wins for pure compute (no launch overhead). Persistent actors win when communication frequency increases.

### Crossover Analysis

| Comm. Frequency | Traditional (ops/s) | Persistent (ops/s) | Winner |
|-----------------|--------------------|--------------------|--------|
| 1 msg/100 steps | | | |
| 1 msg/10 steps | | | |
| 1 msg/step | | | |
| 10 msg/step | | | |

---

## Experiment 6: Memory Overhead Analysis

**Claim**: Actor infrastructure adds <5% memory overhead.

| Component | Bytes per Actor | Notes |
|-----------|----------------|-------|
| Control block | 128 | Aligned, mapped |
| Message queue (header) | | |
| Message queue (slots) | | Capacity × slot size |
| HLC timestamp | 24 | Per-actor clock |
| K2K routing table | | Per-route |
| **Total infrastructure** | | |
| **Useful computation memory** | | |
| **Overhead ratio** | | % |

---

## Experiment 7: Fault Tolerance

**Claim**: Actor-based fault isolation prevents cascading failures.

| Scenario | Healthy Actor Throughput | Recovery Time | State Corruption |
|----------|------------------------|---------------|------------------|
| No fault (baseline) | Mmsg/s | N/A | None |
| 1 actor NaN injection | | ms | |
| 10% actors stalled | | ms | |
| Queue overflow | | ms | |

---

## Experiment 8: H100 Architecture Advantage

**Claim**: H100-specific features provide measurable improvement.

### K2K Messaging: Global Memory vs DSMEM vs TMA

| Method | Latency (us) | Bandwidth (GB/s) | Speedup vs Global |
|--------|-------------|-------------------|--------------------|
| Global memory (current) | | | 1.0x |
| DSMEM (cluster) | | | x |
| TMA async | | | x |

### Grid Sync: Full Grid vs Cluster Sync

| Method | Overhead (us) | Actors Synced | Efficiency |
|--------|--------------|---------------|------------|
| grid.sync() (full) | | all | |
| cluster.sync() (local) | | cluster only | |
| Speedup | x | | |

---

## Application Benchmarks

### WaveSim3D (Stencil Computation)

| Backend | Grid Size | Steps/sec | Mcells/s | Speedup vs CPU |
|---------|-----------|-----------|----------|----------------|
| CPU (Rayon) | 64³ | | | 1.0x |
| GPU Stencil | 64³ | | | x |
| Persistent Actor | 64³ | | | x |
| CPU (Rayon) | 128³ | | | 1.0x |
| GPU Stencil | 128³ | | | x |
| Persistent Actor | 128³ | | | x |

### TxMon (Fraud Detection)

| Backend | TPS | Detection Latency (us) | Speedup |
|---------|-----|----------------------|---------|
| CPU | | | 1.0x |
| GPU Batch | | | x |
| Persistent Actor | | | x |

### ProcInt (Process Mining)

| Backend | Events/sec | DFG Nodes | Conformance/s | Speedup |
|---------|-----------|-----------|---------------|---------|
| CPU | | | | 1.0x |
| GPU | | | | x |

---

## Data Files

All raw data exported to `benchmark_results/`:
- `exp1_launch_overhead_*.csv` — Raw per-iteration latencies
- `exp2_throughput_*.csv` — Message passing measurements
- `exp3_scalability_*.csv` — Actor scaling data
- `exp4_sustained_*.csv` — 60-second time series
- `exp5_mixed_*.csv` — Mixed workload results
- `*_comparison_*.json` — Statistical comparisons with p-values
- `*_system_info.json` — System configuration snapshot

## Reproducibility

```bash
# Record system state
nvidia-smi -q > benchmark_results/nvidia_smi_full.txt
nvcc --version > benchmark_results/nvcc_version.txt
rustc --version > benchmark_results/rustc_version.txt
git rev-parse HEAD > benchmark_results/git_commit.txt
cat /proc/cpuinfo > benchmark_results/cpuinfo.txt
free -h > benchmark_results/meminfo.txt
uname -a > benchmark_results/uname.txt

# Run complete benchmark suite
./scripts/run-academic-benchmarks.sh
```
