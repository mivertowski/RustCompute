# ringkernel-txmon

GPU-accelerated real-time transaction monitoring showcase for RingKernel.

![TxMon Screenshot](../../docs/screenshots/txmon.png)

## Overview

TxMon demonstrates RingKernel's capabilities for high-throughput financial transaction monitoring, suitable for banking/AML compliance scenarios. It includes a transaction factory, compliance rule engine, and real-time GUI.

## Features

- **Transaction Factory**: Configurable synthetic transaction generation with realistic patterns
- **Compliance Rules**: Velocity breach, amount threshold, structuring detection, geographic anomaly
- **Real-time GUI**: Live transaction feed, alerts panel, statistics dashboard
- **Multi-backend**: CPU and optional CUDA GPU acceleration

## GPU Backend Approaches

The `cuda` module provides three different GPU-accelerated approaches:

### 1. Batch Kernel (High Throughput)

Each thread processes one transaction independently. Best for maximum throughput on large batches.

```rust
// ~93B elem/sec on RTX Ada
let alerts = batch_kernel.process(&transactions).await?;
```

### 2. Ring Kernel (Actor-based Streaming)

Persistent GPU kernels with HLC timestamps and K2K messaging. Best for low-latency streaming with complex multi-stage pipelines.

```rust
let kernel = ring_kernel.launch("monitor", config).await?;
kernel.send(transaction).await?;
```

### 3. Stencil Kernel (Pattern Detection)

Uses GridPos for spatial pattern detection in transaction networks. Detects circular trading, velocity anomalies, and coordinated activity patterns.

```rust
let config = StencilConfig::new("pattern_detect")
    .with_tile_size(16, 16);
let patterns = stencil_kernel.detect(&network).await?;
```

## Usage

```rust
use ringkernel_txmon::{TransactionGenerator, MonitoringEngine, GeneratorConfig};

// Create transaction generator
let config = GeneratorConfig {
    transactions_per_second: 10000,
    fraud_ratio: 0.01,  // 1% fraudulent
    ..Default::default()
};
let generator = TransactionGenerator::new(config);

// Create monitoring engine
let engine = MonitoringEngine::new(MonitoringConfig::default());

// Process transactions
for tx in generator.generate_batch(1000) {
    if let Some(alert) = engine.evaluate(&tx) {
        println!("Alert: {:?}", alert);
    }
}
```

## Alert Types

| Type | Description |
|------|-------------|
| `VelocityBreach` | Too many transactions in time window |
| `AmountThreshold` | Single transaction exceeds limit |
| `Structuring` | Pattern of transactions just below threshold |
| `GeographicAnomaly` | Unusual geographic pattern |

## Run GUI

```bash
cargo run -p ringkernel-txmon --bin txmon
```

## Benchmark

```bash
cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen
```

Typical results on RTX 4090:

| Operation | Throughput |
|-----------|------------|
| Batch Kernel | ~93B elem/sec |
| Pattern Detection | ~15.7M TPS |

## Testing

```bash
cargo test -p ringkernel-txmon
```

The crate includes 40+ tests covering transaction generation, rule evaluation, and GPU kernels.

## License

Apache-2.0
