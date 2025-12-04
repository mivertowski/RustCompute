# ringkernel-procint

GPU-accelerated process intelligence and mining for RingKernel.

## Features

- **DFG Construction**: Real-time Directly-Follows Graph building from event streams
- **Pattern Detection**: Identify bottlenecks, loops, rework, and anomalies
- **Partial Order Mining**: Discover concurrent activity relationships
- **Conformance Checking**: Validate traces against reference models
- **Interactive Visualization**: Force-directed DFG canvas with token animation

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│   Data Fabric   │────▶│   GPU Kernels    │────▶│  Visualization │
│ (Event Stream)  │     │ (CUDA/CPU)       │     │  (egui Canvas) │
└─────────────────┘     └──────────────────┘     └────────────────┘
        │                        │                       │
        ▼                        ▼                       ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│ Sector Templates│     │ DFG Construction │     │ Force Layout   │
│ Event Generator │     │ Pattern Detection│     │ Token Animation│
│ Anomaly Inject  │     │ Conformance Check│     │ Timeline View  │
└─────────────────┘     └──────────────────┘     └────────────────┘
```

## Sector Templates

Four industry-specific process templates with realistic transition probabilities:

- **Healthcare**: Patient journey (Registration → Triage → Examination → Diagnosis → Treatment → Discharge)
- **Manufacturing**: Production workflow (Order → Planning → Production → QC → Packaging → Shipped)
- **Finance**: Loan approval (Application → Verification → Credit Check → Approval → Disbursement)
- **IT Incident**: ITIL process (Report → Classify → Assign → Investigate → Resolve → Close)

## GPU Kernels

### DFG Construction (Batch Kernel)
Builds Directly-Follows Graphs from event streams:
- Node frequency counting
- Edge transition tracking
- Duration statistics (min/max/avg)
- Performance: ~2M+ events/sec

### Pattern Detection (Batch Kernel)
Identifies process anti-patterns:
- **Bottleneck**: High incoming, low outgoing rate
- **Loop**: Self-transitions or cycles
- **Rework**: Activity repetition patterns
- **Long-running**: Duration exceeds threshold

### Partial Order Derivation (Stencil Kernel)
Derives concurrency relationships:
- 16x16 bit precedence matrix
- Interval-based ordering
- Transitive closure computation
- Width (max parallelism) calculation

### Conformance Checking (Batch Kernel)
Validates traces against reference models:
- Fitness scoring
- Precision measurement
- Deviation counting
- Alignment calculation

## Usage

### Library

```rust
use ringkernel_procint::prelude::*;

// Create event generator for healthcare sector
let config = GeneratorConfig::default();
let mut generator = ProcessEventGenerator::new(
    SectorTemplate::Healthcare(HealthcareConfig::default()),
    config,
);

// Generate events and build DFG
let events = generator.generate_batch(1000);
let mut kernel = DfgConstructionKernel::new(64).with_cpu_only();
let result = kernel.process(&events);

// Detect patterns
let mut pattern_kernel = PatternDetectionKernel::new(PatternConfig::default());
let patterns = pattern_kernel.detect(result.dfg.nodes());
```

### GUI Application

```bash
# CPU-only mode (default)
cargo run -p ringkernel-procint --bin procint --release

# With CUDA GPU acceleration (requires NVIDIA GPU + CUDA toolkit)
cargo run -p ringkernel-procint --bin procint --release --features cuda
```

### Benchmark

```bash
# CPU benchmark
cargo run -p ringkernel-procint --bin procint-benchmark --release

# GPU benchmark (requires CUDA)
cargo run -p ringkernel-procint --bin procint-benchmark --release --features cuda
```

## GPU Data Structures

All structures are cache-line aligned for optimal GPU memory access:

| Structure | Size | Description |
|-----------|------|-------------|
| GpuObjectEvent | 128 bytes | Process event with HLC timestamp |
| GpuDFGNode | 64 bytes | Activity node with statistics |
| GpuDFGEdge | 64 bytes | Transition edge with frequency |
| GpuDFGGraph | 128 bytes | Graph header/metadata |
| GpuPartialOrderTrace | 256 bytes | Trace with 16x16 precedence matrix |
| GpuPatternMatch | 64 bytes | Detected pattern instance |
| GpuProcessTrace | 64 bytes | Compact trace representation |

## Feature Flags

```toml
[features]
default = []
cuda = ["ringkernel-cuda", "ringkernel-cuda-codegen", "cudarc"]
```

## Dependencies

- `ringkernel-core` - Core actor runtime
- `eframe`/`egui` - GUI framework
- `rkyv` - Zero-copy serialization
- `rand`/`rand_distr` - Random number generation

## License

Apache-2.0
