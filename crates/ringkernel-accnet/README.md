# ringkernel-accnet

GPU-accelerated accounting network analytics with real-time visualization.

![AccNet Screenshot](../../docs/screenshots/accnet.png)

## Overview

AccNet transforms traditional double-entry bookkeeping into a graph representation, enabling advanced analytics powered by GPU compute. The application demonstrates RingKernel's capabilities for financial network analysis, fraud detection, and compliance monitoring.

## Features

- **Network Visualization**: Interactive graph showing account relationships and money flows
- **Fraud Detection**: Circular flows, threshold clustering, Benford's Law violations
- **GAAP Compliance**: Automated detection of accounting rule violations
- **Temporal Analysis**: Seasonality, trends, behavioral anomalies
- **Multi-backend**: CPU and optional CUDA GPU acceleration

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│   Data Fabric   │────>│  GPU Kernels     │────>│  Visualization │
│ (Synthetic Gen) │     │ (CUDA/WGSL)      │     │  (egui Canvas) │
└─────────────────┘     └──────────────────┘     └────────────────┘
        │                        │                       │
        ▼                        ▼                       ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│ Journal Entries │     │ Network Analysis │     │ Graph Layout   │
│ Transaction Gen │     │ Fraud Detection  │     │ Flow Animation │
│ Anomaly Inject  │     │ Temporal Analysis│     │ Analytics UI   │
└─────────────────┘     └──────────────────┘     └────────────────┘
```

## GPU Kernels

### 1. Suspense Detection

Identifies suspicious clearing accounts by analyzing balance patterns, risk scores, and flow imbalances.

```rust
// Each thread processes one account
fn suspense_detection(
    balance_debit: &[f64],
    balance_credit: &[f64],
    risk_scores: &[f32],
    suspense_scores: &mut [f32],
    n_accounts: i32
)
```

### 2. GAAP Violation Detection

Checks for improper account pairings that violate Generally Accepted Accounting Principles.

```rust
// Each thread processes one flow
fn gaap_violation(
    flow_source: &[u16],
    flow_target: &[u16],
    account_types: &[u8],
    violation_flags: &mut [u8],
    n_flows: i32
)
```

### 3. Benford Analysis

Statistical analysis of first-digit distribution to detect fabricated transactions.

```rust
// Each thread processes one amount
fn benford_analysis(
    amounts: &[f64],
    digit_counts: &mut [u32],  // Atomic updates
    n_amounts: i32
)
```

## Fraud Pattern Types

| Pattern | Risk Weight | Description |
|---------|-------------|-------------|
| CircularFlow | 0.95 | Money flowing in a circle: A → B → C → A |
| HighVelocity | 0.90 | Rapid multi-hop money movement |
| ThresholdClustering | 0.85 | Amounts clustered below approval threshold |
| StructuredTransactions | 0.85 | Split transactions to avoid detection |
| DormantActivation | 0.80 | Sudden activity on long-dormant accounts |
| UnusualPairing | 0.75 | Implausible account combinations |
| BenfordViolation | 0.70 | Amount distribution violates Benford's Law |
| SelfLoop | 0.65 | Bidirectional flow between accounts |
| AfterHoursEntry | 0.60 | Entry posted outside business hours |

## Usage

```rust
use ringkernel_accnet::prelude::*;

// Create a network
let mut network = AccountingNetwork::new(entity_id, 2024, 1);

// Add accounts
let cash = network.add_account(
    AccountNode::new(Uuid::new_v4(), AccountType::Asset, 0),
    AccountMetadata::new("1100", "Cash")
);

// Add flows
network.add_flow(TransactionFlow::new(
    source, target, amount, journal_id, timestamp
));

// Run analysis
network.calculate_pagerank(0.85, 20);
```

## Run GUI

```bash
# CPU backend
cargo run -p ringkernel-accnet --release

# With CUDA GPU acceleration
cargo run -p ringkernel-accnet --release --features cuda
```

## Benchmark

```bash
cargo run -p ringkernel-accnet --bin accnet-benchmark --release --features cuda
```

## Testing

```bash
cargo test -p ringkernel-accnet
```

## License

Apache-2.0
