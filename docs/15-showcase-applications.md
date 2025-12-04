---
layout: default
title: Showcase Applications
nav_order: 16
---

# Showcase Applications

RingKernel includes four comprehensive showcase applications demonstrating GPU-accelerated computing with the actor model. Each showcases different aspects of the framework.

---

## WaveSim - 2D Acoustic Wave Simulation

![WaveSim Screenshot](screenshots/wavesim.png)

**Interactive wave propagation simulator** implementing the Finite-Difference Time-Domain (FDTD) method.

### Key Features

- **Tile-based Actor Model**: 16x16 cell tiles as actors with K2K messaging for halo exchange
- **Educational Modes**: Visualize computing paradigms from 1950s sequential to modern parallel
- **Multi-backend**: CPU (SoA + SIMD + Rayon), CUDA, and WGPU

### Performance

| Backend | 256x256 | 512x512 |
|---------|---------|---------|
| CPU SimulationGrid | 35,418 steps/s | 7,229 steps/s |
| **CUDA Packed** | **112,837 steps/s** | **71,324 steps/s** |

GPU vs CPU speedup: **3.1x at 256x256, 9.9x at 512x512**

### Run It

```bash
cargo run -p ringkernel-wavesim --release
```

Click anywhere on the canvas to inject wave impulses.

---

## TxMon - Transaction Monitoring

![TxMon Screenshot](screenshots/txmon.png)

**Real-time transaction monitoring** for banking/AML compliance scenarios with GPU-accelerated rule evaluation.

### Key Features

- **Transaction Factory**: Configurable synthetic generation with realistic patterns
- **Compliance Rules**: Velocity breach, amount threshold, structuring detection
- **Three GPU Approaches**:
  - Batch Kernel: Maximum throughput (~93B elem/sec)
  - Ring Kernel: Persistent actors with HLC and K2K messaging
  - Stencil Kernel: Pattern detection in transaction networks

### Performance

| Operation | Throughput |
|-----------|------------|
| Batch Kernel | ~93B elem/sec |
| Pattern Detection | ~15.7M TPS |

### Run It

```bash
cargo run -p ringkernel-txmon --release
```

---

## AccNet - Accounting Network Analytics

![AccNet Screenshot](screenshots/accnet.png)

**GPU-accelerated accounting network analysis** transforming double-entry bookkeeping into graph analytics.

### Key Features

- **Network Visualization**: Interactive graph showing account relationships and money flows
- **Fraud Detection**: Circular flows, threshold clustering, Benford's Law violations
- **GAAP Compliance**: Automated detection of accounting rule violations
- **Temporal Analysis**: Seasonality, trends, behavioral anomalies

### GPU Kernels

1. **Suspense Detection**: Identifies suspicious clearing accounts
2. **GAAP Violation**: Checks for improper account pairings
3. **Benford Analysis**: Statistical analysis of first-digit distribution
4. **PageRank**: Network centrality and influence analysis

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│   Data Fabric   │────>│  GPU Kernels     │────>│  Visualization │
│ (Synthetic Gen) │     │ (CUDA/WGSL)      │     │  (egui Canvas) │
└─────────────────┘     └──────────────────┘     └────────────────┘
```

### Run It

```bash
# CPU backend
cargo run -p ringkernel-accnet --release

# With CUDA GPU acceleration
cargo run -p ringkernel-accnet --release --features cuda
```

---

## ProcInt - Process Intelligence

![ProcInt Screenshot](screenshots/procint.png)

**GPU-accelerated process mining** for discovering, analyzing, and monitoring business processes from event logs.

### Key Features

- **DFG Mining**: Directly-Follows Graph construction from event streams with GPU acceleration
- **Pattern Detection**: Bottlenecks, loops, rework, and long-running activities
- **Conformance Checking**: Fitness and precision metrics against reference models
- **Timeline View**: Partial order traces with concurrent activity visualization
- **Multi-Sector**: Healthcare, Manufacturing, Finance, and IT process templates

### GPU Kernels

1. **DFG Construction**: Builds directly-follows graph from event stream (batch kernel)
2. **Pattern Detection**: Identifies process patterns like bottlenecks and loops
3. **Partial Order Derivation**: Builds precedence matrix from interval events (stencil kernel)
4. **Conformance Checking**: Validates traces against reference models

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│   Data Fabric   │────>│  GPU Kernels     │────>│  Visualization │
│ (Event Stream)  │     │ (DFG/Pattern)    │     │  (egui Canvas) │
└─────────────────┘     └──────────────────┘     └────────────────┘
        │                       │                       │
   Sector Templates      4 Kernel Types           Force-directed
   Anomaly Injection     CPU/CUDA fallback        DFG + Timeline
```

### Performance

| Operation | Throughput |
|-----------|------------|
| DFG Construction | ~47K events/sec |
| Pattern Detection | ~3.8M elements/sec |

### Run It

```bash
# CPU backend
cargo run -p ringkernel-procint --release

# Run benchmark
cargo run -p ringkernel-procint --bin procint-benchmark --release
```

---

## Common Patterns Across Showcases

All four applications demonstrate RingKernel's core capabilities:

| Pattern | WaveSim | TxMon | AccNet | ProcInt |
|---------|---------|-------|--------|---------|
| GPU Actor Model | Tile actors | Ring kernels | Analysis actors | DFG/Pattern kernels |
| K2K Messaging | Halo exchange | Multi-stage pipeline | Network analysis | Kernel coordination |
| Real-time GUI | egui + glow | egui dashboard | egui graph canvas | egui DFG + Timeline |
| Multi-backend | CPU/CUDA/WGPU | CPU/CUDA | CPU/CUDA | CPU/CUDA |
| HLC Timestamps | Tile ordering | Transaction ordering | Event ordering | Trace ordering |

### Build All Showcases

```bash
# All showcases, CPU only
cargo build -p ringkernel-wavesim -p ringkernel-txmon -p ringkernel-accnet -p ringkernel-procint --release

# With CUDA support
cargo build -p ringkernel-wavesim -p ringkernel-txmon -p ringkernel-accnet -p ringkernel-procint --release --features cuda
```

---

## Next: [CUDA Code Generation](./13-cuda-codegen.md)
