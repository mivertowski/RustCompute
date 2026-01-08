---
layout: default
title: Testing
nav_order: 10
---

# Testing Strategy

## Test Categories

Following DotCompute's testing structure:

| Category | Location | GPU Required | Purpose |
|----------|----------|--------------|---------|
| Unit | `tests/unit/` | No | Component isolation |
| Integration | `tests/integration/` | Optional | Cross-component |
| Hardware | `tests/hardware/` | Yes | GPU-specific |
| Benchmarks | `benches/` | Yes | Performance |

## Test Count Summary

700+ tests across the workspace:
- `ringkernel-core`: 65 tests (queue, HLC, control block, K2K, PubSub, enterprise runtime)
- `ringkernel-cpu`: 11 tests (CPU backend, mock GPU)
- `ringkernel-cuda`: 6 GPU execution tests
- `ringkernel-cuda-codegen`: 183 tests (transpiler, intrinsics, loops, shared memory, ring kernels, K2K, envelope format)
- `ringkernel-wgpu-codegen`: 50 tests (types, intrinsics, transpiler, validation)
- `ringkernel-derive`: 14 macro tests
- `ringkernel-ecosystem`: 30 tests (persistent handle, Actix, Axum, Tower, gRPC)
- `ringkernel-wavesim`: 63 tests (simulation, kernels, grid, educational modes)
- `ringkernel-wavesim3d`: 72 tests (3D FDTD, binaural, volumetric rendering)
- `ringkernel-txmon`: 40 tests (GPU types, batch kernel, stencil, ring kernel)
- `ringkernel-procint`: 77 tests (DFG, pattern detection, conformance)
- `ringkernel-audio-fft`: 32 tests
- `ringkernel-ir`: 40+ tests (IR builder, validation, lowering)
- Plus integration tests (K2K, control block, HLC)

---

## Unit Tests

### Message Serialization

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rkyv::{to_bytes, from_bytes};

    #[test]
    fn test_vector_add_request_serialization() {
        let request = VectorAddRequest {
            id: Uuid::new_v4(),
            priority: priority::NORMAL,
            correlation_id: None,
            a: vec![1.0, 2.0, 3.0],
            b: vec![4.0, 5.0, 6.0],
        };

        // Serialize
        let bytes = to_bytes::<_, 1024>(&request).unwrap();

        // Deserialize
        let archived = rkyv::check_archived_root::<VectorAddRequest>(&bytes).unwrap();

        assert_eq!(archived.a.len(), 3);
        assert_eq!(archived.b.len(), 3);
        assert_eq!(archived.priority, priority::NORMAL);
    }

    #[test]
    fn test_message_size_within_limits() {
        let request = VectorAddRequest {
            id: Uuid::new_v4(),
            priority: priority::NORMAL,
            correlation_id: None,
            a: vec![0.0; 10000],
            b: vec![0.0; 10000],
        };

        let bytes = to_bytes::<_, 262144>(&request).unwrap();
        assert!(bytes.len() <= MAX_MESSAGE_SIZE);
    }
}
```

### Control Block

```rust
#[test]
fn test_control_block_size_and_alignment() {
    assert_eq!(std::mem::size_of::<ControlBlock>(), 128);
    assert_eq!(std::mem::align_of::<ControlBlock>(), 128);
}

#[test]
fn test_control_block_field_offsets() {
    use std::mem::offset_of;

    assert_eq!(offset_of!(ControlBlock, is_active), 0);
    assert_eq!(offset_of!(ControlBlock, should_terminate), 4);
    assert_eq!(offset_of!(ControlBlock, has_terminated), 8);
    assert_eq!(offset_of!(ControlBlock, messages_processed), 16);
    assert_eq!(offset_of!(ControlBlock, input_queue_head_ptr), 32);
    assert_eq!(offset_of!(ControlBlock, output_queue_head_ptr), 64);
}
```

### HLC Timestamp

```rust
#[test]
fn test_hlc_ordering() {
    let ts1 = HlcTimestamp::new(100, 0, 1);
    let ts2 = HlcTimestamp::new(100, 1, 1);
    let ts3 = HlcTimestamp::new(101, 0, 1);

    assert!(ts1 < ts2);
    assert!(ts2 < ts3);
}

#[test]
fn test_hlc_merge() {
    let mut local = HlcTimestamp::new(100, 5, 1);
    let remote = HlcTimestamp::new(100, 10, 2);

    local.merge(&remote);

    assert_eq!(local.physical(), 100);
    assert_eq!(local.logical(), 11); // max(5, 10) + 1
}
```

---

## Integration Tests

### CPU Backend

```rust
#[tokio::test]
async fn test_cpu_kernel_lifecycle() {
    let runtime = RingKernelRuntime::builder()
        .backend(BackendType::Cpu)
        .build()
        .await
        .unwrap();

    // Launch
    let kernel = runtime
        .launch("echo", Dim3::linear(1), Dim3::linear(1), LaunchOptions::default())
        .await
        .unwrap();

    // Activate
    kernel.activate().await.unwrap();
    assert_eq!(kernel.status().await.unwrap(), KernelStatus::Active);

    // Send/receive
    let request = EchoRequest { value: 42 };
    kernel.send(request).await.unwrap();

    let response: EchoResponse = kernel.receive(Duration::from_secs(1)).await.unwrap();
    assert_eq!(response.value, 42);

    // Terminate
    kernel.terminate().await.unwrap();
}

#[tokio::test]
async fn test_deactivate_preserves_state() {
    let runtime = RingKernelRuntime::builder()
        .backend(BackendType::Cpu)
        .build()
        .await
        .unwrap();

    let kernel = runtime
        .launch("counter", Dim3::linear(1), Dim3::linear(1), LaunchOptions::default())
        .await
        .unwrap();

    kernel.activate().await.unwrap();

    // Process some messages
    for i in 0..10 {
        kernel.send(IncrementRequest { amount: 1 }).await.unwrap();
    }

    // Deactivate
    kernel.deactivate().await.unwrap();

    // Reactivate
    kernel.activate().await.unwrap();

    // State should be preserved
    let status: CounterStatus = kernel.call(GetStatusRequest {}).await.unwrap();
    assert_eq!(status.count, 10);

    kernel.terminate().await.unwrap();
}
```

---

## Hardware Tests

### CUDA Backend

```rust
/// Skip test if no CUDA GPU available
fn require_cuda() -> bool {
    CudaBackend::new().map(|b| b.device_count() > 0).unwrap_or(false)
}

#[tokio::test]
#[ignore = "requires CUDA GPU"]
async fn test_cuda_vector_add() {
    if !require_cuda() {
        eprintln!("Skipping: no CUDA GPU");
        return;
    }

    let runtime = RingKernelRuntime::builder()
        .backend(BackendType::Cuda)
        .build()
        .await
        .unwrap();

    let kernel = runtime
        .launch(
            "vector_add",
            Dim3::linear(1),
            Dim3::linear(256),
            LaunchOptions::default(),
        )
        .await
        .unwrap();

    kernel.activate().await.unwrap();

    let request = VectorAddRequest {
        id: Uuid::new_v4(),
        priority: priority::NORMAL,
        correlation_id: None,
        a: vec![1.0, 2.0, 3.0, 4.0],
        b: vec![5.0, 6.0, 7.0, 8.0],
    };

    kernel.send(request).await.unwrap();

    let response: VectorAddResponse = kernel.receive(Duration::from_secs(5)).await.unwrap();

    assert_eq!(response.result, vec![6.0, 8.0, 10.0, 12.0]);

    kernel.terminate().await.unwrap();
}

#[tokio::test]
#[ignore = "requires CUDA GPU"]
async fn test_cuda_message_throughput() {
    if !require_cuda() {
        return;
    }

    let runtime = RingKernelRuntime::builder()
        .backend(BackendType::Cuda)
        .build()
        .await
        .unwrap();

    let kernel = runtime
        .launch("echo", Dim3::linear(1), Dim3::linear(256), LaunchOptions {
            queue_capacity: 16384,
            ..Default::default()
        })
        .await
        .unwrap();

    kernel.activate().await.unwrap();

    let start = Instant::now();
    let message_count = 10000;

    for i in 0..message_count {
        kernel.send(EchoRequest { value: i }).await.unwrap();
    }

    for _ in 0..message_count {
        let _: EchoResponse = kernel.receive(Duration::from_secs(1)).await.unwrap();
    }

    let elapsed = start.elapsed();
    let throughput = message_count as f64 / elapsed.as_secs_f64();

    println!("Throughput: {:.0} messages/sec", throughput);
    assert!(throughput > 10000.0, "Expected >10k msg/s, got {}", throughput);

    kernel.terminate().await.unwrap();
}
```

### WSL2 Compatibility

```rust
#[tokio::test]
#[ignore = "requires CUDA GPU"]
async fn test_wsl2_event_driven_mode() {
    if !require_cuda() || !is_wsl2() {
        return;
    }

    // On WSL2, persistent mode doesn't work reliably
    // Test that event-driven mode works

    let runtime = RingKernelRuntime::builder()
        .backend(BackendType::Cuda)
        .config(RuntimeConfig {
            // Force event-driven mode
            ..Default::default()
        })
        .build()
        .await
        .unwrap();

    let kernel = runtime
        .launch("vector_add", Dim3::linear(1), Dim3::linear(256), LaunchOptions {
            mode: RingKernelMode::EventDriven,
            ..Default::default()
        })
        .await
        .unwrap();

    kernel.activate().await.unwrap();

    // Should work even on WSL2
    let request = VectorAddRequest {
        id: Uuid::new_v4(),
        priority: priority::NORMAL,
        correlation_id: None,
        a: vec![1.0, 2.0],
        b: vec![3.0, 4.0],
    };

    kernel.send(request).await.unwrap();

    let response: VectorAddResponse = kernel.receive(Duration::from_secs(10)).await.unwrap();
    assert_eq!(response.result, vec![4.0, 6.0]);

    kernel.terminate().await.unwrap();
}
```

---

## Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_serialization_roundtrip(
        a in prop::collection::vec(any::<f32>(), 0..1000),
        b in prop::collection::vec(any::<f32>(), 0..1000),
    ) {
        let request = VectorAddRequest {
            id: Uuid::new_v4(),
            priority: priority::NORMAL,
            correlation_id: None,
            a: a.clone(),
            b: b.clone(),
        };

        let bytes = rkyv::to_bytes::<_, 262144>(&request).unwrap();
        let archived = rkyv::check_archived_root::<VectorAddRequest>(&bytes).unwrap();

        prop_assert_eq!(archived.a.as_slice(), a.as_slice());
        prop_assert_eq!(archived.b.as_slice(), b.as_slice());
    }

    #[test]
    fn test_hlc_always_advances(
        timestamps in prop::collection::vec((0u64..1000, 0u32..100, 0u32..10), 1..100)
    ) {
        let mut hlc = HlcTimestamp::new(0, 0, 0);

        for (physical, logical, node_id) in timestamps {
            let received = HlcTimestamp::new(physical, logical, node_id);
            let before = hlc.clone();
            hlc.merge(&received);

            // HLC must always advance
            prop_assert!(hlc > before || hlc == before);
        }
    }
}
```

---

## Mock GPU Testing (v0.2.0+)

Test GPU code without hardware using the mock module:

```rust
use ringkernel_cpu::mock::{MockGpuDevice, MockKernel, MockExecutionConfig};

#[test]
fn test_kernel_with_mock_gpu() {
    // Create mock device
    let device = MockGpuDevice::new()
        .with_memory_mb(8192)
        .with_compute_capability(8, 6);

    // Create mock kernel
    let kernel = MockKernel::new("processor")
        .with_block_size(256)
        .with_grid_size(1024);

    // Configure execution behavior
    let config = MockExecutionConfig::new()
        .deterministic(true)  // Reproducible results
        .simulate_latency(Duration::from_micros(100));

    // Execute
    let result = device.execute(&kernel, &input, config)?;

    // Verify - exact same results every time
    assert_eq!(result, expected_output);
}

#[test]
fn test_memory_allocation_tracking() {
    let device = MockGpuDevice::new();

    // Allocate buffers
    let buf1 = device.allocate::<f32>(1024)?;
    let buf2 = device.allocate::<f32>(2048)?;

    // Check memory usage
    let stats = device.memory_stats();
    assert_eq!(stats.allocated_bytes, (1024 + 2048) * 4);
    assert_eq!(stats.allocation_count, 2);

    // Detect leaks
    drop(buf1);
    let stats = device.memory_stats();
    assert_eq!(stats.allocation_count, 1);
}
```

### Mock Execution Modes

```rust
// Deterministic - Same output for same input
let config = MockExecutionConfig::deterministic();

// Randomized - Simulates non-deterministic GPU behavior
let config = MockExecutionConfig::randomized(seed);

// Fault injection - Test error handling
let config = MockExecutionConfig::with_fault(
    MockFault::OutOfMemory { after_allocations: 5 }
);
```

---

## Fuzzing Infrastructure (v0.2.0+)

RingKernel includes 5 fuzz targets for security and robustness testing:

### Running Fuzz Tests

```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# List fuzz targets
cargo +nightly fuzz list

# Run a specific target
cargo +nightly fuzz run fuzz_message_serialization

# Run with a time limit
cargo +nightly fuzz run fuzz_queue_operations -- -max_total_time=300
```

### Fuzz Targets

| Target | Description |
|--------|-------------|
| `fuzz_message_serialization` | Test rkyv serialization with arbitrary data |
| `fuzz_queue_operations` | Queue enqueue/dequeue sequences |
| `fuzz_hlc_timestamps` | HLC merge and ordering operations |
| `fuzz_ir_validation` | IR construction and validation |
| `fuzz_codegen` | Code generation from arbitrary IR |

### Writing Fuzz Tests

```rust
// fuzz/fuzz_targets/fuzz_message_serialization.rs
#![no_main]

use libfuzzer_sys::fuzz_target;
use ringkernel_core::message::*;

fuzz_target!(|data: &[u8]| {
    // Try to deserialize arbitrary bytes
    if let Ok(archived) = rkyv::check_archived_root::<TestMessage>(data) {
        // If valid, verify roundtrip
        let deserialized = archived.deserialize(&mut rkyv::Infallible).unwrap();
        let reserialized = rkyv::to_bytes::<_, 4096>(&deserialized).unwrap();
        assert_eq!(data, &reserialized[..]);
    }
});
```

### Corpus Management

```bash
# Minimize corpus
cargo +nightly fuzz cmin fuzz_message_serialization

# Merge corpus from multiple runs
cargo +nightly fuzz merge fuzz_message_serialization corpus1/ corpus2/

# Check coverage
cargo +nightly fuzz coverage fuzz_message_serialization
```

---

## CI Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --lib --all-features

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --test '*' --features cpu

  hardware-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --test '*' --features cuda -- --ignored
```

---

## Next: [Performance Considerations](./10-performance.md)
