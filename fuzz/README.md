# RingKernel Fuzzing Infrastructure

This directory contains fuzzing targets for the RingKernel project using `cargo-fuzz` with libFuzzer.

## Prerequisites

Install cargo-fuzz:
```bash
cargo install cargo-fuzz
```

You also need a nightly toolchain for fuzzing:
```bash
rustup toolchain install nightly
```

## Available Fuzz Targets

| Target | Description |
|--------|-------------|
| `fuzz_ir_builder` | Fuzzes IR building operations to find crashes in IR construction |
| `fuzz_cuda_transpiler` | Fuzzes CUDA code generation with random Rust-like input |
| `fuzz_wgsl_transpiler` | Fuzzes WGSL code generation with random Rust-like input |
| `fuzz_message_queue` | Fuzzes lock-free message queue operations |
| `fuzz_hlc` | Fuzzes Hybrid Logical Clock implementation and invariants |

## Running Fuzz Tests

From the repository root:

```bash
# Run a specific fuzz target
cargo +nightly fuzz run fuzz_ir_builder

# Run with a time limit (e.g., 60 seconds)
cargo +nightly fuzz run fuzz_ir_builder -- -max_total_time=60

# Run with multiple jobs
cargo +nightly fuzz run fuzz_ir_builder -- -jobs=4

# Run with sanitizers enabled (recommended)
RUSTFLAGS="-Zsanitizer=address" cargo +nightly fuzz run fuzz_ir_builder
```

## Reproducing Crashes

When a crash is found, it will be saved to `fuzz/artifacts/<target_name>/`. To reproduce:

```bash
cargo +nightly fuzz run fuzz_ir_builder fuzz/artifacts/fuzz_ir_builder/crash-<hash>
```

## Corpus Management

Corpus files are stored in `fuzz/corpus/<target_name>/`. To minimize the corpus:

```bash
cargo +nightly fuzz cmin fuzz_ir_builder
```

## Coverage

To generate coverage reports:

```bash
cargo +nightly fuzz coverage fuzz_ir_builder
# View coverage in target/x86_64-unknown-linux-gnu/coverage/
```

## Adding New Fuzz Targets

1. Create a new file in `fuzz/fuzz_targets/`
2. Add a `[[bin]]` section to `fuzz/Cargo.toml`
3. Use the `fuzz_target!` macro from `libfuzzer-sys`

Example:
```rust
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Your fuzzing logic here
});
```

## CI Integration

For continuous fuzzing in CI:

```yaml
# .github/workflows/fuzz.yml
name: Fuzz Tests
on:
  schedule:
    - cron: '0 0 * * *'  # Daily

jobs:
  fuzz:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
      - run: cargo install cargo-fuzz
      - run: cargo +nightly fuzz run fuzz_ir_builder -- -max_total_time=300
      - run: cargo +nightly fuzz run fuzz_message_queue -- -max_total_time=300
      - run: cargo +nightly fuzz run fuzz_hlc -- -max_total_time=300
```

## Notes

- Fuzzing requires the nightly toolchain due to libFuzzer integration
- The transpiler fuzz targets generate random Rust-like code and only call the transpiler if `syn` successfully parses it
- Message queue and HLC fuzzers verify invariants (monotonicity, ordering) and will panic on violations
