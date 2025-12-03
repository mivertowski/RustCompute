# ringkernel-ecosystem

Ecosystem integrations for RingKernel.

## Overview

This crate provides optional integrations with popular Rust ecosystem libraries for actors, web frameworks, data processing, and observability. All integrations are opt-in via feature flags.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `actix` | Actix actor framework bridge |
| `tower` | Tower service middleware integration |
| `axum` | Axum web framework integration |
| `arrow` | Apache Arrow data processing |
| `polars` | Polars DataFrame operations |
| `grpc` | gRPC server via Tonic |
| `candle` | Candle ML framework bridge |
| `config` | Configuration file management |
| `tracing-integration` | Enhanced tracing support |
| `prometheus` | Prometheus metrics export |
| `full` | All integrations enabled |

## Installation

```toml
[dependencies]
ringkernel-ecosystem = { version = "0.1", features = ["axum", "prometheus"] }
```

## Usage

```rust
use ringkernel_ecosystem::prelude::*;

// Features are conditionally available based on enabled features
```

### Axum Integration

```rust
use ringkernel_ecosystem::axum::RingKernelLayer;

let app = Router::new()
    .route("/process", post(handler))
    .layer(RingKernelLayer::new(runtime));
```

### Prometheus Metrics

```rust
use ringkernel_ecosystem::metrics::PrometheusExporter;

let exporter = PrometheusExporter::new();
exporter.register_runtime_metrics(&runtime);
```

### Configuration

```rust
use ringkernel_ecosystem::config::ConfigManager;

let config = ConfigManager::load("config.toml")?;
let runtime = config.create_runtime().await?;
```

## Testing

```bash
cargo test -p ringkernel-ecosystem
cargo test -p ringkernel-ecosystem --features full
```

## License

Apache-2.0
