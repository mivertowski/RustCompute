# H100/B200 Baseline Benchmarks

> Run these benchmarks on dedicated H100/B200 instances to validate the persistent actor pattern.

## Prerequisites

```bash
# Set architecture target
export RINGKERNEL_CUDA_ARCH=sm_90  # H100
# export RINGKERNEL_CUDA_ARCH=sm_100  # B200

# Build with CUDA + cooperative groups
cargo build --workspace --features cuda --release
cargo build -p ringkernel-cuda --features "cuda,cooperative" --release
```

## Benchmark Suite

### 1. Message Injection Latency

```bash
cargo bench --package ringkernel -- injection_latency
```

| Metric | Ada (RTX 4090) | H100 | B200 |
|--------|---------------|------|------|
| Mean latency | 0.03 us | | |
| P99 latency | | | |
| Throughput (msg/s) | | | |

### 2. Queue Throughput

```bash
cargo bench --package ringkernel -- queue_throughput
```

| Metric | Ada | H100 | B200 |
|--------|-----|------|------|
| Ops/sec | 75M | | |
| SPSC throughput | | | |
| MPMC throughput | | | |

### 3. K2K Halo Exchange

```bash
cargo run -p ringkernel-wavesim3d --bin wavesim3d-benchmark --release --features cuda-codegen
```

| Metric | Ada | H100 | B200 |
|--------|-----|------|------|
| Per-element latency | | | |
| Halo bandwidth (GB/s) | | | |
| Grid sync overhead | | | |

### 4. Cooperative Grid Sync

```bash
cargo run -p ringkernel-wavesim3d --bin interactive-benchmark --release --features cuda-codegen
```

| Grid Size | Ada (us) | H100 (us) | B200 (us) |
|-----------|---------|-----------|-----------|
| 64 blocks | | | |
| 256 blocks | | | |
| 1024 blocks | | | |
| Max blocks | | | |

### 5. Reduction Throughput

```bash
cargo bench --package ringkernel -- reduction
```

| Metric | Ada | H100 | B200 |
|--------|-----|------|------|
| Elem/sec | 93B | | |
| Block reduce | | | |
| Grid reduce | | | |

### 6. Persistent Kernel Sustained Throughput

```bash
# 60-second sustained run
cargo run -p ringkernel --example pagerank_reduction --features cuda --release
```

| Metric | Ada | H100 | B200 |
|--------|-----|------|------|
| Sustained ops/sec | | | |
| Mean latency (60s) | | | |
| P99 latency (60s) | | | |
| Memory usage (MB) | | | |

### 7. Memory Pool Allocation

```bash
cargo bench --package ringkernel -- memory_pool
```

| Metric | Ada | H100 | B200 |
|--------|-----|------|------|
| Alloc rate (ops/s) | | | |
| Fragmentation % | | | |
| Pool warmup (ms) | | | |

### 8. WaveSim3D End-to-End

```bash
cargo run -p ringkernel-wavesim3d --bin wavesim3d-benchmark --release --features cuda-codegen
```

| Metric | Ada | H100 | B200 |
|--------|-----|------|------|
| Steps/sec | | | |
| Grid size | | | |
| Total time (s) | | | |

### 9. TxMon Fraud Detection

```bash
cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen
```

| Metric | Ada | H100 | B200 |
|--------|-----|------|------|
| Transactions/sec | | | |
| Detection latency | | | |
| False positive rate | | | |

### 10. ProcInt DFG Mining

```bash
cargo run -p ringkernel-procint --bin procint-benchmark --release
```

| Metric | Ada | H100 | B200 |
|--------|-----|------|------|
| Events/sec | | | |
| DFG nodes discovered | | | |
| Conformance checks/s | | | |

## System Information

```bash
# Record before benchmarking
nvidia-smi
nvcc --version
uname -a
cat /proc/cpuinfo | grep "model name" | head -1
free -h
```

| Property | Value |
|----------|-------|
| GPU | |
| Driver Version | |
| CUDA Version | |
| CPU | |
| RAM | |
| OS | |

## Notes

- All benchmarks should be run with `--release` profile
- Run 3 times and report median
- Ensure GPU is in exclusive compute mode: `nvidia-smi -c EXCLUSIVE_PROCESS`
- Lock GPU clocks for consistent results: `nvidia-smi -lgc <max_clock>`
- Monitor GPU temperature: ensure no thermal throttling during sustained tests
