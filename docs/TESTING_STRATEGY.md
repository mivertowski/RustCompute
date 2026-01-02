# Testing Strategy

> Comprehensive Testing Plan for RingKernel

## Overview

This document defines the testing strategy for RingKernel, covering unit tests, integration tests, performance benchmarks, and quality assurance processes across all backends and phases.

---

## Testing Pyramid

```
                    ┌─────────────────┐
                    │    E2E Tests    │  ← 5% (Full system validation)
                    │   (10 tests)    │
                   ─┴─────────────────┴─
                  ┌───────────────────────┐
                  │  Integration Tests    │  ← 20% (Cross-component)
                  │     (150 tests)       │
                 ─┴───────────────────────┴─
                ┌─────────────────────────────┐
                │      Component Tests        │  ← 30% (Backend/crate level)
                │        (300 tests)          │
               ─┴─────────────────────────────┴─
              ┌───────────────────────────────────┐
              │          Unit Tests               │  ← 45% (Function level)
              │          (600 tests)              │
             ─┴───────────────────────────────────┴─
```

### Target Test Distribution

| Level | Count | Purpose |
|-------|-------|---------|
| Unit | 600+ | Individual functions, invariants |
| Component | 300+ | Crate-level integration |
| Integration | 150+ | Cross-crate, backend switching |
| E2E | 10+ | Full application scenarios |

---

## Test Categories

### 1. Unit Tests

Located in `src/` alongside code using `#[cfg(test)]`.

```rust
// Example: ringkernel-core/src/hlc.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hlc_tick_increments_logical() {
        let mut clock = HlcClock::new(1);
        let ts1 = clock.tick();
        let ts2 = clock.tick();
        assert!(ts2 > ts1);
    }

    #[test]
    fn hlc_update_merges_timestamps() {
        let mut clock = HlcClock::new(1);
        let remote = HlcTimestamp { physical: 1000, logical: 5, node_id: 2 };
        let merged = clock.update(remote);
        assert!(merged.physical >= remote.physical);
    }
}
```

**Coverage Targets by Crate**:

| Crate | Target | Current |
|-------|--------|---------|
| ringkernel-core | 90% | 85% |
| ringkernel-cuda-codegen | 85% | 80% |
| ringkernel-wgpu-codegen | 85% | 75% |
| ringkernel-ecosystem | 80% | 70% |
| ringkernel-derive | 80% | 75% |

### 2. Component Tests

Located in `tests/` directory of each crate.

```rust
// Example: ringkernel-cuda/tests/persistent_simulation.rs
use ringkernel_cuda::persistent::*;

#[tokio::test]
#[ignore] // Requires GPU
async fn persistent_simulation_lifecycle() {
    let device = CudaDevice::new(0).unwrap();
    let config = PersistentSimulationConfig::new(64, 64, 64);
    let mut sim = PersistentSimulation::new(&device, config).unwrap();

    // Test lifecycle
    sim.start(&ptx, "test_kernel").unwrap();
    assert_eq!(sim.status(), KernelStatus::Active);

    sim.run_steps(100).unwrap();
    assert_eq!(sim.stats().current_step, 100);

    sim.pause().unwrap();
    assert_eq!(sim.status(), KernelStatus::Paused);

    sim.resume().unwrap();
    assert_eq!(sim.status(), KernelStatus::Active);

    sim.shutdown().unwrap();
    assert_eq!(sim.status(), KernelStatus::Terminated);
}
```

### 3. Integration Tests

Cross-crate testing in workspace-level `tests/` or dedicated test crates.

```rust
// Example: tests/integration/cuda_ecosystem.rs
use ringkernel::prelude::*;
use ringkernel_ecosystem::axum::*;

#[tokio::test]
async fn cuda_kernel_with_axum_rest_api() {
    // Launch persistent kernel
    let runtime = CudaRuntime::new().await.unwrap();
    let kernel = runtime.launch_persistent("processor", Default::default()).await.unwrap();

    // Create Axum state
    let state = PersistentGpuState::new(kernel, Default::default());
    let app = axum::Router::new().merge(state.routes());

    // Test REST API
    let client = TestClient::new(app);

    let resp = client.post("/api/step")
        .json(&StepRequest { count: 100 })
        .send()
        .await;

    assert_eq!(resp.status(), 200);

    let stats: StatsResponse = client.get("/api/stats").send().await.json().await;
    assert_eq!(stats.current_step, 100);
}
```

### 4. E2E Tests

Full application scenarios with real GPU hardware.

```rust
// Example: e2e/wavesim3d_scenario.rs
#[tokio::test]
#[ignore] // Requires GPU + GUI
async fn wavesim3d_full_simulation() {
    // Start simulation
    let app = WaveSim3dApp::new(AppConfig::default()).await.unwrap();

    // Run for 1000 steps
    for _ in 0..1000 {
        app.step().await.unwrap();
    }

    // Verify energy conservation (within tolerance)
    let energy = app.total_energy();
    assert!((energy - 1.0).abs() < 0.01);

    // Verify no NaN/Inf in output
    assert!(app.field_data().iter().all(|v| v.is_finite()));
}
```

---

## Backend-Specific Testing

### CUDA Testing

**Requirements**:
- NVIDIA GPU with compute capability 7.0+
- CUDA Toolkit 12.0+
- cudarc 0.18.2

**Test Configuration**:
```rust
// tests/cuda_common.rs
pub fn skip_if_no_cuda() {
    if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
        eprintln!("Skipping: CUDA not available");
        return;
    }
}

pub fn cuda_device() -> CudaDevice {
    skip_if_no_cuda();
    CudaDevice::new(0).expect("CUDA device")
}
```

**Running CUDA Tests**:
```bash
# Run all CUDA tests
cargo test --package ringkernel-cuda --features cuda

# Run specific GPU test
cargo test --package ringkernel-cuda --test gpu_execution_verify

# Run with specific device
CUDA_VISIBLE_DEVICES=0 cargo test --features cuda
```

### Metal Testing

**Requirements**:
- macOS 12+ or iOS 15+
- Apple Silicon or AMD GPU
- Metal 3.0+ support

**Test Configuration**:
```rust
// tests/metal_common.rs
pub fn skip_if_no_metal() {
    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("Skipping: Metal only available on macOS");
        return;
    }
}

pub fn metal_device() -> MetalDevice {
    skip_if_no_metal();
    MetalDevice::system_default().expect("Metal device")
}
```

**Running Metal Tests**:
```bash
# Run all Metal tests (macOS only)
cargo test --package ringkernel-metal --features metal

# Run with specific device
MTL_DEVICE_WRAPPER_TYPE=1 cargo test --features metal
```

### WebGPU Testing

**Requirements**:
- Vulkan, Metal, or DX12 support
- wgpu 27.0 compatible drivers

**Test Configuration**:
```rust
// tests/wgpu_common.rs
pub async fn wgpu_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("No adapter");

    adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("Device")
}
```

**Running WebGPU Tests**:
```bash
# Run all WebGPU tests
cargo test --package ringkernel-wgpu --features wgpu-tests -- --ignored

# Force specific backend
WGPU_BACKEND=vulkan cargo test --features wgpu-tests
WGPU_BACKEND=metal cargo test --features wgpu-tests
WGPU_BACKEND=dx12 cargo test --features wgpu-tests
```

### CPU Backend Testing

Always available, used for CI without GPU.

```bash
# Run CPU-only tests (default)
cargo test --workspace

# Explicitly use CPU backend
RINGKERNEL_BACKEND=cpu cargo test --workspace
```

---

## Mock Testing

### Mock GPU Framework

```rust
// ringkernel-testing/src/mock.rs
pub struct MockGpu {
    memory: HashMap<u64, Vec<u8>>,
    compute_units: u32,
    latency_ns: u64,
    operations: Vec<MockOperation>,
}

impl MockGpu {
    pub fn new() -> Self {
        Self {
            memory: HashMap::new(),
            compute_units: 80,
            latency_ns: 100,
            operations: Vec::new(),
        }
    }

    pub fn with_latency(mut self, ns: u64) -> Self {
        self.latency_ns = ns;
        self
    }

    pub fn allocate(&mut self, size: usize) -> u64 {
        let addr = self.memory.len() as u64 * 0x1000;
        self.memory.insert(addr, vec![0; size]);
        self.operations.push(MockOperation::Allocate { addr, size });
        addr
    }

    pub fn write(&mut self, addr: u64, data: &[u8]) {
        if let Some(mem) = self.memory.get_mut(&addr) {
            mem[..data.len()].copy_from_slice(data);
        }
        self.operations.push(MockOperation::Write { addr, size: data.len() });
    }

    pub fn dispatch(&mut self, kernel: &str, grid: (u32, u32, u32)) {
        std::thread::sleep(Duration::from_nanos(self.latency_ns));
        self.operations.push(MockOperation::Dispatch {
            kernel: kernel.to_string(),
            grid,
        });
    }

    pub fn operations(&self) -> &[MockOperation] {
        &self.operations
    }
}
```

### Mock Persistent Handle

```rust
pub struct MockPersistentHandle {
    status: Arc<AtomicU32>,
    current_step: Arc<AtomicU64>,
    commands: Arc<Mutex<Vec<PersistentCommand>>>,
    responses: Arc<Mutex<VecDeque<PersistentResponse>>>,
}

impl PersistentHandle for MockPersistentHandle {
    async fn send_command(&self, cmd: PersistentCommand) -> Result<CommandId> {
        self.commands.lock().push(cmd.clone());

        match cmd {
            PersistentCommand::RunSteps { count } => {
                self.current_step.fetch_add(count, Ordering::SeqCst);
            }
            PersistentCommand::Terminate => {
                self.status.store(KernelStatus::Terminated as u32, Ordering::SeqCst);
            }
            _ => {}
        }

        Ok(CommandId::new())
    }

    async fn poll_responses(&self) -> Result<Vec<PersistentResponse>> {
        Ok(self.responses.lock().drain(..).collect())
    }
}
```

### Using Mocks in Tests

```rust
#[tokio::test]
async fn test_axum_with_mock_gpu() {
    let mock = MockPersistentHandle::new();
    let state = PersistentGpuState::new(mock.clone(), Default::default());
    let app = axum::Router::new().merge(state.routes());

    let client = TestClient::new(app);

    // Test step endpoint
    client.post("/api/step")
        .json(&StepRequest { count: 50 })
        .send()
        .await;

    // Verify mock received command
    let commands = mock.commands();
    assert!(matches!(commands[0], PersistentCommand::RunSteps { count: 50 }));
}
```

---

## Property-Based Testing

### Using proptest

```rust
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn message_header_roundtrip(
        type_id in 0u32..1000,
        payload_len in 0u32..10000,
        priority in 0u8..255,
    ) {
        let header = MessageHeader::new(type_id, payload_len, priority);
        let bytes = header.to_bytes();
        let decoded = MessageHeader::from_bytes(&bytes).unwrap();

        prop_assert_eq!(header.type_id, decoded.type_id);
        prop_assert_eq!(header.payload_len, decoded.payload_len);
        prop_assert_eq!(header.priority, decoded.priority);
    }

    #[test]
    fn hlc_always_monotonic(ops in prop::collection::vec(hlc_op_strategy(), 1..100)) {
        let mut clock = HlcClock::new(1);
        let mut prev = HlcTimestamp::default();

        for op in ops {
            let ts = match op {
                HlcOp::Tick => clock.tick(),
                HlcOp::Update(received) => clock.update(received),
            };

            prop_assert!(ts > prev, "HLC must be monotonic: {:?} <= {:?}", ts, prev);
            prev = ts;
        }
    }

    #[test]
    fn queue_preserves_order(messages in prop::collection::vec(any::<u64>(), 1..100)) {
        let queue = MessageQueue::new(128);

        for msg in &messages {
            queue.enqueue(*msg).unwrap();
        }

        for expected in &messages {
            let actual = queue.dequeue().unwrap();
            prop_assert_eq!(*expected, actual);
        }
    }
}

fn hlc_op_strategy() -> impl Strategy<Value = HlcOp> {
    prop_oneof![
        Just(HlcOp::Tick),
        (0u64..1000000, 0u32..1000, 0u32..100)
            .prop_map(|(p, l, n)| HlcOp::Update(HlcTimestamp {
                physical: p,
                logical: l,
                node_id: n,
            }))
    ]
}
```

### Invariant Testing

```rust
/// Test queue invariants under concurrent access
#[test]
fn queue_invariants_under_stress() {
    let queue = Arc::new(MessageQueue::new(1024));
    let produced = Arc::new(AtomicU64::new(0));
    let consumed = Arc::new(AtomicU64::new(0));

    // Producer threads
    let producers: Vec<_> = (0..4).map(|_| {
        let q = queue.clone();
        let p = produced.clone();
        std::thread::spawn(move || {
            for i in 0..10000 {
                if q.enqueue(i).is_ok() {
                    p.fetch_add(1, Ordering::SeqCst);
                }
            }
        })
    }).collect();

    // Consumer threads
    let consumers: Vec<_> = (0..4).map(|_| {
        let q = queue.clone();
        let c = consumed.clone();
        std::thread::spawn(move || {
            for _ in 0..10000 {
                if q.dequeue().is_some() {
                    c.fetch_add(1, Ordering::SeqCst);
                }
            }
        })
    }).collect();

    for p in producers { p.join().unwrap(); }
    for c in consumers { c.join().unwrap(); }

    // Invariant: consumed <= produced
    assert!(consumed.load(Ordering::SeqCst) <= produced.load(Ordering::SeqCst));

    // Invariant: queue length = produced - consumed
    assert_eq!(
        queue.len(),
        (produced.load(Ordering::SeqCst) - consumed.load(Ordering::SeqCst)) as usize
    );
}
```

---

## Fuzzing

### Message Parsing Fuzz Targets

```rust
// fuzz/fuzz_targets/message_header.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use ringkernel_core::message::MessageHeader;

fuzz_target!(|data: &[u8]| {
    // Should never panic
    let _ = MessageHeader::try_from_bytes(data);
});
```

```rust
// fuzz/fuzz_targets/h2k_message.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use ringkernel_cuda::persistent::H2KMessage;

fuzz_target!(|data: &[u8]| {
    if data.len() >= std::mem::size_of::<H2KMessage>() {
        let msg: H2KMessage = unsafe {
            std::ptr::read(data.as_ptr() as *const H2KMessage)
        };
        // Validation should handle any input
        let _ = msg.validate();
        let _ = msg.command_type();
    }
});
```

### Running Fuzz Tests

```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# List fuzz targets
cargo +nightly fuzz list

# Run fuzzer for 60 seconds
cargo +nightly fuzz run message_header -- -max_total_time=60

# Run with corpus
cargo +nightly fuzz run message_header fuzz/corpus/message_header/

# Minimize crash
cargo +nightly fuzz tmin message_header crash-abc123
```

---

## Performance Benchmarks

### Criterion Benchmarks

```rust
// benches/persistent_kernel.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_command_injection(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let kernel = rt.block_on(async {
        let runtime = CudaRuntime::new().await.unwrap();
        runtime.launch_persistent("benchmark", Default::default()).await.unwrap()
    });

    c.bench_function("h2k_command_injection", |b| {
        b.iter(|| {
            rt.block_on(kernel.send_command(PersistentCommand::RunSteps { count: 1 }))
        })
    });
}

fn benchmark_step_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("step_throughput");
    for size in [32, 64, 128, 256].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let kernel = rt.block_on(async {
                let config = PersistentConfig::new(size, size, size);
                let runtime = CudaRuntime::new().await.unwrap();
                runtime.launch_persistent("fdtd", config).await.unwrap()
            });

            b.iter(|| {
                rt.block_on(kernel.run_steps(100))
            })
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_command_injection, benchmark_step_throughput);
criterion_main!(benches);
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench --package ringkernel

# Run specific benchmark
cargo bench --package ringkernel -- command_injection

# Save baseline
cargo bench --package ringkernel -- --save-baseline main

# Compare to baseline
cargo bench --package ringkernel -- --baseline main

# Generate HTML report
cargo bench --package ringkernel -- --plotting-backend plotters
open target/criterion/report/index.html
```

### Performance Regression Detection

```yaml
# .github/workflows/bench.yml
name: Benchmarks
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4

      - name: Run benchmarks
        run: cargo bench --package ringkernel -- --save-baseline pr

      - name: Compare to main
        run: |
          git fetch origin main
          git checkout origin/main -- target/criterion
          cargo bench --package ringkernel -- --baseline main --load-baseline pr

      - name: Check for regressions
        run: |
          # Fail if any benchmark regressed >10%
          python scripts/check_bench_regression.py target/criterion
```

---

## CI/CD Testing Pipeline

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [main]
  pull_request:

env:
  CARGO_TERM_COLOR: always

jobs:
  # Fast unit tests (no GPU)
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2

      - name: Run unit tests
        run: cargo test --workspace --lib

  # Component tests (no GPU)
  component-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2

      - name: Run component tests
        run: cargo test --workspace --tests --exclude ringkernel-cuda --exclude ringkernel-metal

  # GPU tests (requires self-hosted runner)
  gpu-tests:
    runs-on: [self-hosted, gpu, cuda]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Run CUDA tests
        run: cargo test --package ringkernel-cuda --features cuda
        env:
          CUDA_VISIBLE_DEVICES: 0

  # macOS Metal tests
  metal-tests:
    runs-on: macos-14  # Apple Silicon
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Run Metal tests
        run: cargo test --package ringkernel-metal --features metal

  # Code coverage
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: llvm-tools-preview
      - uses: taiki-e/install-action@cargo-llvm-cov

      - name: Generate coverage
        run: cargo llvm-cov --workspace --lcov --output-path lcov.info

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: lcov.info

  # Clippy lints
  clippy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy

      - name: Run clippy
        run: cargo clippy --workspace --all-targets -- -D warnings

  # Documentation
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Build docs
        run: cargo doc --workspace --no-deps
        env:
          RUSTDOCFLAGS: -D warnings
```

---

## Test Data Management

### Fixtures

```rust
// tests/fixtures/mod.rs
pub fn sample_grid_64() -> Vec<f32> {
    vec![0.0; 64 * 64 * 64]
}

pub fn sample_impulse() -> ImpulseData {
    ImpulseData {
        position: (32, 32, 32),
        amplitude: 1.0,
        frequency: 440.0,
    }
}

pub fn sample_ptx() -> &'static str {
    include_str!("../fixtures/test_kernel.ptx")
}
```

### Golden Files

```rust
// tests/codegen/golden.rs
#[test]
fn cuda_codegen_matches_golden() {
    let kernel_fn = parse_quote! {
        fn saxpy(x: &[f32], y: &mut [f32], a: f32) {
            let idx = thread_idx_x();
            y[idx] = a * x[idx] + y[idx];
        }
    };

    let generated = transpile_global_kernel(&kernel_fn).unwrap();
    let golden = include_str!("golden/saxpy.cu");

    assert_eq!(generated.trim(), golden.trim());
}
```

---

## Test Quality Metrics

### Coverage Thresholds

| Crate | Minimum | Target |
|-------|---------|--------|
| ringkernel-core | 80% | 90% |
| ringkernel-cuda | 70% | 85% |
| ringkernel-cuda-codegen | 80% | 90% |
| ringkernel-wgpu-codegen | 75% | 85% |
| ringkernel-ecosystem | 75% | 85% |

### Mutation Testing

```bash
# Install cargo-mutants
cargo install cargo-mutants

# Run mutation testing
cargo mutants --package ringkernel-core

# Check mutation score
# Target: >70% mutants killed
```

---

## Appendix: Test Naming Conventions

```rust
// Unit tests: test_<function>_<scenario>
#[test]
fn test_hlc_tick_increments_logical() { }

#[test]
fn test_hlc_update_with_future_timestamp() { }

// Integration tests: test_<component>_<behavior>
#[test]
fn test_cuda_runtime_launches_kernel() { }

// Property tests: prop_<invariant>
proptest! {
    fn prop_queue_fifo_order(messages: Vec<u64>) { }
}

// Benchmark: bench_<operation>_<variant>
fn bench_command_injection_persistent() { }
fn bench_command_injection_traditional() { }
```
