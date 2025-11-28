# RingKernel Benchmark Suite

This benchmark suite validates the performance claims made in the RingKernel README.

## Performance Targets

| Metric | Target | DotCompute Reference |
|--------|--------|----------------------|
| Serialization | <30ns | 20-50ns |
| Message latency (native) | <500µs | <1ms |
| Startup time | <1ms | ~10ms |
| Binary size | <2MB | 5-10MB |

## Benchmark Files

### Core Benchmarks

| Benchmark | Description | Claims Validated |
|-----------|-------------|------------------|
| `serialization.rs` | Header and HLC serialization via zerocopy | Serialization <30ns |
| `message_queue.rs` | Lock-free queue enqueue/dequeue operations | Lock-free messaging |
| `startup.rs` | Runtime and kernel startup times | Startup time <1ms |
| `latency.rs` | End-to-end message latency | Message latency <500µs |
| `memory_layout.rs` | Memory layout validation (128B control block, 256B header) | Memory layout claims |
| `hlc.rs` | Hybrid Logical Clock performance and ordering | HLC causal ordering |
| `k2k_messaging.rs` | Kernel-to-kernel messaging performance | K2K messaging |
| `claims_validation.rs` | Comprehensive validation of ALL README claims | All claims |

## Running Benchmarks

### Run All Benchmarks

```bash
# From project root
cargo bench --package ringkernel

# Or use the runner script
./scripts/run_benchmarks.sh
```

### Run Specific Benchmark

```bash
# Run only serialization benchmarks
cargo bench --package ringkernel --bench serialization

# Run claims validation
cargo bench --package ringkernel --bench claims_validation
```

### Quick Validation

```bash
./scripts/run_benchmarks.sh quick
```

### Full Claims Validation

```bash
./scripts/run_benchmarks.sh claims
```

### Binary Size Validation

```bash
./scripts/check_binary_size.sh
```

## Benchmark Categories

### 1. Serialization Benchmarks (`serialization.rs`)

Validates the <30ns serialization target using zerocopy.

- `header_zerocopy`: Zero-copy header serialization
- `header_roundtrip`: Full serialize/deserialize cycle
- `timestamp_now`: HLC timestamp creation
- `timestamp_pack_unpack`: HLC pack/unpack operations

### 2. Startup Benchmarks (`startup.rs`)

Validates the <1ms startup time target.

- `cpu_runtime_create`: Runtime initialization
- `single_kernel`: Kernel launch time
- `runtime_and_kernel`: Full system startup
- `multi_kernel_startup`: Multiple kernel startup

### 3. Latency Benchmarks (`latency.rs`)

Validates the <500µs message latency target.

- `envelope_creation`: Message envelope creation
- `queue_roundtrip`: Single message round-trip
- `end_to_end`: Full message path latency
- `burst_latency`: Batch message latency

### 4. Memory Layout Benchmarks (`memory_layout.rs`)

Validates memory layout claims.

- `control_block_128_bytes`: Control block is exactly 128 bytes
- `control_block_128_aligned`: Control block is 128-byte aligned
- `message_header_256_bytes`: Message header is exactly 256 bytes
- `zero_copy_serialization`: Validates zero-copy works correctly

### 5. HLC Benchmarks (`hlc.rs`)

Validates HLC causal ordering guarantees.

- `timestamp_creation`: Timestamp creation performance
- `clock_operations`: Clock tick and update
- `ordering`: Timestamp comparison and sorting
- `causality`: Cross-node causal ordering

### 6. K2K Messaging Benchmarks (`k2k_messaging.rs`)

Validates kernel-to-kernel messaging performance.

- `message_creation`: K2K message creation
- `broker_operations`: K2K broker operations
- `endpoint_operations`: K2K endpoint send/receive
- `lock_free_queue`: Underlying queue operations

### 7. Claims Validation (`claims_validation.rs`)

Comprehensive benchmark that validates ALL README claims:

- `CLAIM_1_serialization_under_30ns`
- `CLAIM_2_message_latency_under_500us`
- `CLAIM_3_startup_under_1ms`
- `CLAIM_4_memory_layout`
- `CLAIM_5_hlc_causal_ordering`
- `CLAIM_6_lock_free_messaging`
- `CLAIM_7_zero_copy_serialization`

## Interpreting Results

### Criterion Output

Criterion outputs timing statistics including:
- **time**: Mean execution time
- **thrpt**: Throughput (for throughput benchmarks)
- **change**: Performance change from baseline

### Target Validation

Look for benchmarks in the `CLAIM_*` groups. These explicitly test against the README targets.

Example output interpretation:
```
CLAIM_1_serialization_under_30ns/message_header_as_bytes
                        time:   [4.2341 ns 4.2584 ns 4.2856 ns]
```

In this example, 4.26ns is well under the 30ns target.

### Performance Comparison

The `claims_validation` benchmark provides a comprehensive view:

```
VALIDATION_SUMMARY/all_claims_combined
                        time:   [1.2345 µs 1.2456 µs 1.2567 µs]
```

## Continuous Integration

These benchmarks can be integrated into CI:

```yaml
- name: Run Benchmarks
  run: |
    cargo bench --package ringkernel --bench claims_validation
    ./scripts/check_binary_size.sh
```

## Adding New Benchmarks

1. Create a new `.rs` file in `benches/`
2. Add `[[bench]]` entry to `Cargo.toml`
3. Use Criterion framework
4. Include validation assertions where appropriate

Example template:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_my_feature(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_feature");

    group.bench_function("operation", |b| {
        b.iter(|| {
            // Benchmark code
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_my_feature);
criterion_main!(benches);
```
