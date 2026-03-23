# RingKernel Benchmark Methodology

> Academic-grade benchmark methodology for validating the GPU-native persistent actor paradigm.

## 1. Thesis Statement

RingKernel demonstrates that **persistent GPU kernels operating as actors** with lock-free message passing achieve fundamentally different performance characteristics compared to the traditional kernel-launch model — specifically:

1. **Sub-microsecond command injection** (0.03 us vs 50-300 us traditional launch overhead)
2. **Zero-copy inter-kernel messaging** via mapped memory and K2K channels
3. **Sustained throughput without re-launch overhead** under mixed workloads
4. **Linear scalability** of actor count within cooperative group limits

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Levels | Rationale |
|----------|--------|-----------|
| **Execution model** | Traditional (re-launch), Persistent (actor) | Core comparison |
| **Message rate** | 1K, 10K, 100K, 1M msg/s | Throughput scaling |
| **Actor count** | 1, 4, 16, 64, 256, 1024 | Scalability |
| **Message payload** | 64B, 256B, 1KB, 4KB | Payload sensitivity |
| **Grid size** | 32, 128, 512, max blocks | Resource pressure |
| **GPU architecture** | Ada (sm_89), Hopper (sm_90) | Hardware generality |

### 2.2 Dependent Variables

| Metric | Unit | Measurement Method |
|--------|------|-------------------|
| **Injection latency** | microseconds | CUDA events (device-side) |
| **End-to-end latency** | microseconds | Host clock bracketing |
| **Throughput** | messages/second | Count / wall-clock time |
| **Tail latency** | p50, p95, p99, p99.9 | Sorted measurement array |
| **SM utilization** | percent | NVTX + Nsight Compute |
| **Memory bandwidth** | GB/s | CUDA profiler |
| **Power consumption** | watts | nvidia-smi sampling |

### 2.3 Control Variables

| Control | Method |
|---------|--------|
| GPU clock frequency | Lock with `nvidia-smi -lgc <max>` |
| Compute mode | Exclusive process: `nvidia-smi -c EXCLUSIVE_PROCESS` |
| CPU governor | Performance mode: `cpupower frequency-set -g performance` |
| Thermal state | 5-minute warmup before measurement |
| OS interference | Disable GUI, minimize background services |
| Memory state | Fresh allocation per trial (no reuse) |
| ECC | Report ECC status (may affect bandwidth) |

## 3. Statistical Protocol

### 3.1 Trial Structure

```
[Warmup: 100 iterations, discarded]
[Measurement: 1000 iterations, recorded]
× 10 independent trials (fresh process each)
= 10,000 measurements per configuration
```

### 3.2 Reporting Requirements

For each metric, report:

| Statistic | Description |
|-----------|-------------|
| **n** | Number of measurements |
| **Mean** | Arithmetic mean |
| **Median** | 50th percentile |
| **Std Dev** | Standard deviation |
| **95% CI** | Confidence interval: mean ± 1.96 × (σ / √n) |
| **CV** | Coefficient of variation: σ / mean |
| **Min / Max** | Range |
| **p50 / p95 / p99 / p99.9** | Latency percentiles |

### 3.3 Comparative Analysis

When comparing Traditional vs Persistent:
- **Speedup**: geometric mean of per-trial speedups
- **Effect size**: Cohen's d = (μ₁ - μ₂) / s_pooled
- **Statistical significance**: Welch's t-test (unequal variances), report p-value
- **Practical significance**: report absolute difference alongside relative speedup

### 3.4 Outlier Treatment

- **Detection**: Modified Z-score (MAD-based), threshold = 3.5
- **Reporting**: Report both with and without outliers
- **Do NOT silently remove outliers** — they may represent real system behavior (GC, preemption)

## 4. Experiment Catalog

### Experiment 1: Launch Overhead — The Fundamental Advantage

**Hypothesis**: Persistent actor injection latency is 3-4 orders of magnitude lower than traditional kernel launch.

**Method**:
```
Traditional:  for each command { cudaMemcpyHtoD → cudaLaunchKernel → cudaDeviceSynchronize }
Persistent:   for each command { write_to_mapped_memory (1 store) }
```

**Measurements**: 10,000 commands, report per-command latency distribution.

**Expected result**: Traditional ~50-300 us, Persistent ~0.01-0.1 us.

**Why it matters**: Eliminates the kernel launch tax that dominates fine-grained GPU workloads.

### Experiment 2: Message Passing Throughput

**Hypothesis**: Lock-free K2K messaging achieves near-memory-bandwidth throughput.

**Method**: Producer-consumer actor pair exchanging messages of varying payload sizes.

**Configurations**:
- Payload: 64B, 256B, 1KB, 4KB
- Queue depth: 256, 1024, 4096
- K2K vs host-mediated comparison

**Measurements**: Messages/second, bytes/second, queue utilization.

### Experiment 3: Actor Scalability

**Hypothesis**: Actor throughput scales linearly with actor count up to cooperative group limits.

**Method**: Identical actors processing independent message streams.

**Configurations**: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 actors.

**Measurements**: Per-actor throughput, aggregate throughput, grid sync overhead.

**Strong scaling**: Fixed total work, increasing actor count.
**Weak scaling**: Fixed per-actor work, increasing actor count.

### Experiment 4: Sustained Throughput Under Load

**Hypothesis**: Persistent actors maintain stable throughput over extended periods without degradation.

**Method**: 60-second continuous run at 90% of peak injection rate.

**Measurements**: 1-second windowed throughput, latency percentiles per window, memory pressure.

**Report**: Time series plot of throughput and latency.

### Experiment 5: Mixed Workload — The Real-World Case

**Hypothesis**: Persistent actors outperform traditional re-launch under mixed read/write/compute workloads.

**Method**: Alternating inject (write), query (read), and compute (step) operations at varying ratios.

**Configurations**:
- Compute-heavy: 90% compute, 5% inject, 5% query
- Balanced: 33% each
- Communication-heavy: 10% compute, 45% inject, 45% query

**Measurements**: Ops/second by type, total throughput, latency by operation type.

### Experiment 6: Memory Overhead Analysis

**Hypothesis**: Actor infrastructure (control blocks, queues, HLC) adds <5% memory overhead vs raw computation.

**Method**: Measure total GPU memory with and without actor infrastructure for equivalent computation.

**Measurements**: Bytes per actor, queue overhead, control block overhead, total vs useful memory ratio.

### Experiment 7: Fault Tolerance — Graceful Degradation

**Hypothesis**: Actor-based fault isolation prevents single-actor failures from corrupting global state.

**Method**: Inject faults (NaN values, infinite loops) into individual actors, measure system-level impact.

**Measurements**: Healthy actor throughput during fault, recovery time, state corruption events.

### Experiment 8: Hopper Architecture Advantage (H100-specific)

**Hypothesis**: H100 features (DSMEM, Thread Block Clusters) provide measurable improvement over global-memory K2K.

**Method**: Compare K2K messaging via:
a) Global memory (current implementation)
b) Distributed shared memory (Phase 5 DSMEM)
c) TMA async copy (Phase 5)

**Measurements**: Message latency, bandwidth, SM utilization per method.

## 5. Data Export Format

### 5.1 Raw Data (CSV)

Every measurement produces a CSV with columns:
```csv
experiment,configuration,trial,iteration,metric,value,unit,timestamp
exp1_launch_overhead,traditional,1,1,latency_us,52.3,microseconds,2026-03-25T10:00:00Z
exp1_launch_overhead,persistent,1,1,latency_us,0.028,microseconds,2026-03-25T10:00:00Z
```

### 5.2 Summary Statistics (JSON)

```json
{
  "experiment": "exp1_launch_overhead",
  "configuration": "persistent",
  "gpu": "H100",
  "metric": "injection_latency_us",
  "n": 10000,
  "mean": 0.031,
  "median": 0.028,
  "std_dev": 0.012,
  "ci_95_lower": 0.029,
  "ci_95_upper": 0.033,
  "p50": 0.028,
  "p95": 0.045,
  "p99": 0.067,
  "p999": 0.112,
  "min": 0.019,
  "max": 0.298,
  "cv": 0.387,
  "outliers_removed": 3,
  "timestamp": "2026-03-25T10:00:00Z",
  "system": {
    "gpu": "NVIDIA H100 80GB HBM3",
    "driver": "550.x",
    "cuda": "12.6",
    "compute_cap": "9.0",
    "gpu_clock_mhz": 1980,
    "mem_clock_mhz": 2619,
    "ecc": true
  }
}
```

### 5.3 Comparative Summary (for paper tables)

```json
{
  "comparison": "traditional_vs_persistent",
  "metric": "injection_latency_us",
  "traditional": {"mean": 52.3, "ci_95": [48.1, 56.5]},
  "persistent": {"mean": 0.031, "ci_95": [0.029, 0.033]},
  "speedup": 1687.1,
  "speedup_ci_95": [1467.2, 1948.3],
  "cohens_d": 8.92,
  "p_value": 1.2e-47,
  "significant": true
}
```

## 6. Reproducibility Checklist

Before publishing results:

- [ ] GPU driver version recorded
- [ ] CUDA toolkit version recorded
- [ ] GPU clocks locked (report frequency)
- [ ] Exclusive compute mode enabled
- [ ] CPU governor set to performance
- [ ] 5-minute GPU warmup completed
- [ ] ECC status recorded
- [ ] Ambient temperature noted
- [ ] Background process list captured
- [ ] RingKernel git commit hash recorded
- [ ] cudarc version recorded
- [ ] Rust compiler version recorded
- [ ] OS version and kernel recorded
- [ ] Raw data exported as CSV
- [ ] Summary statistics computed
- [ ] Outlier analysis performed
- [ ] Multiple trials completed (minimum 10)

## 7. Visualization Standards

### For Papers
- **Box plots** for latency distributions (show median, IQR, whiskers, outliers)
- **CDF plots** for tail latency analysis
- **Bar charts with error bars** (95% CI) for throughput comparisons
- **Line plots** for scalability (x: actor count, y: throughput)
- **Time series** for sustained throughput experiments
- **Heatmaps** for parameter sensitivity (payload × queue depth)

### Format
- Vector graphics (SVG/PDF)
- Consistent color scheme: Traditional = gray, Persistent = blue, DSMEM = green
- Font size ≥ 8pt in final print
- Include data points alongside trend lines

## 8. Claims Framework

Each performance claim must be supported by:

1. **Measurement**: Raw data with statistical summary
2. **Comparison**: Against a well-understood baseline
3. **Conditions**: Exact hardware/software configuration
4. **Limitations**: When the claim does NOT hold
5. **Reproducibility**: Steps to independently verify
