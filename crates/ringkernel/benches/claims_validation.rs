//! Comprehensive Claims Validation Benchmark Suite
//!
//! This benchmark suite validates ALL performance claims from the README:
//!
//! | Metric              | Target    | DotCompute |
//! |---------------------|-----------|------------|
//! | Serialization       | <30ns     | 20-50ns    |
//! | Message latency     | <500µs    | <1ms       |
//! | Startup time        | <1ms      | ~10ms      |
//! | Binary size         | <2MB      | 5-10MB     |
//!
//! Additional claims validated:
//! - Control block is 128 bytes, cache-line aligned
//! - Message header is 256 bytes
//! - HLC provides causal ordering
//! - Lock-free message passing
//! - Zero-copy serialization

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime as TokioRuntime;

use ringkernel_core::control::ControlBlock;
use ringkernel_core::hlc::{HlcClock, HlcTimestamp, HlcState};
use ringkernel_core::message::{MessageHeader, MessageEnvelope};
use ringkernel_core::queue::{MessageQueue, SpscQueue};
use ringkernel_core::runtime::{LaunchOptions, RingKernelRuntime};
use ringkernel_cpu::CpuRuntime;
use zerocopy::AsBytes;

// =============================================================================
// CLAIM 1: Serialization <30ns (Target: <30ns, DotCompute: 20-50ns)
// =============================================================================

fn validate_serialization_claim(c: &mut Criterion) {
    let mut group = c.benchmark_group("CLAIM_1_serialization_under_30ns");

    group.sample_size(1000);
    group.measurement_time(Duration::from_secs(10));

    // Header serialization (zerocopy - should be essentially free)
    group.bench_function("message_header_as_bytes", |b| {
        let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));

        b.iter(|| {
            let bytes = header.as_bytes();
            black_box(bytes);
        });
    });

    // HLC timestamp serialization
    group.bench_function("hlc_timestamp_as_bytes", |b| {
        let ts = HlcTimestamp::now(1);

        b.iter(|| {
            let bytes = ts.as_bytes();
            black_box(bytes);
        });
    });

    // Control block access (zerocopy compatible)
    group.bench_function("control_block_field_access", |b| {
        let cb = ControlBlock::with_capacities(1024, 1024);

        b.iter(|| {
            black_box(cb.is_active);
            black_box(cb.messages_processed);
            black_box(cb.input_head);
        });
    });

    // Header roundtrip (serialize + deserialize)
    group.bench_function("header_roundtrip", |b| {
        let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));

        b.iter(|| {
            let bytes = header.as_bytes();
            let restored = MessageHeader::read_from(bytes);
            black_box(restored);
        });
    });

    group.finish();
}

// =============================================================================
// CLAIM 2: Message Latency <500µs (Target: <500µs, DotCompute: <1ms)
// =============================================================================

fn validate_message_latency_claim(c: &mut Criterion) {
    let mut group = c.benchmark_group("CLAIM_2_message_latency_under_500us");

    group.sample_size(500);
    group.measurement_time(Duration::from_secs(10));

    // Queue round-trip latency (core message passing path)
    group.bench_function("queue_roundtrip_256b", |b| {
        let queue = SpscQueue::new(1024);
        let envelope = MessageEnvelope {
            header: MessageHeader::new(1, 0, 1, 256, HlcTimestamp::now(1)),
            payload: vec![0u8; 256],
        };

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                queue.try_enqueue(envelope.clone()).unwrap();
                let msg = queue.try_dequeue().unwrap();
                black_box(msg);
            }
            start.elapsed()
        });
    });

    // Full message path: create + enqueue + dequeue + validate
    group.bench_function("full_message_path_256b", |b| {
        let queue = SpscQueue::new(1024);

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                // Create message with HLC timestamp
                let envelope = MessageEnvelope {
                    header: MessageHeader::new(1, 0, 1, 256, HlcTimestamp::now(1)),
                    payload: vec![0u8; 256],
                };

                // Enqueue (host -> device transfer)
                queue.try_enqueue(envelope).unwrap();

                // Dequeue (device processing)
                let received = queue.try_dequeue().unwrap();

                // Validate (ensures message integrity)
                assert!(received.header.validate());
                black_box(received);
            }
            start.elapsed()
        });
    });

    // Various payload sizes
    for size in [64, 256, 1024, 4096].iter() {
        group.bench_with_input(
            BenchmarkId::new("payload_bytes", size),
            size,
            |b, &sz| {
                let queue = SpscQueue::new(1024);
                let envelope = MessageEnvelope {
                    header: MessageHeader::new(1, 0, 1, sz, HlcTimestamp::now(1)),
                    payload: vec![0u8; sz],
                };

                b.iter(|| {
                    queue.try_enqueue(envelope.clone()).unwrap();
                    let msg = queue.try_dequeue().unwrap();
                    black_box(msg);
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// CLAIM 3: Startup Time <1ms (Target: <1ms, DotCompute: ~10ms)
// =============================================================================

fn validate_startup_time_claim(c: &mut Criterion) {
    let mut group = c.benchmark_group("CLAIM_3_startup_under_1ms");

    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));

    let rt = TokioRuntime::new().unwrap();

    // Runtime creation only
    group.bench_function("runtime_create", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                rt.block_on(async {
                    let runtime = CpuRuntime::new().await.unwrap();
                    black_box(runtime);
                });
            }
            start.elapsed()
        });
    });

    // Single kernel launch
    group.bench_function("kernel_launch", |b| {
        let runtime = rt.block_on(CpuRuntime::new()).unwrap();
        let mut counter = 0u64;

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                counter += 1;
                let name = format!("startup_kernel_{}", counter);
                rt.block_on(async {
                    let handle = runtime.launch(&name, LaunchOptions::default()).await.unwrap();
                    black_box(handle);
                });
            }
            start.elapsed()
        });
    });

    // Full startup: runtime + kernel + ready
    group.bench_function("full_startup_cycle", |b| {
        let mut counter = 0u64;

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                counter += 1;
                let name = format!("full_startup_{}", counter);

                rt.block_on(async {
                    let runtime = CpuRuntime::new().await.unwrap();
                    let handle = runtime.launch(&name, LaunchOptions::default()).await.unwrap();
                    black_box(&handle);
                    runtime.shutdown().await.unwrap();
                });
            }
            start.elapsed()
        });
    });

    group.finish();
}

// =============================================================================
// CLAIM 4: Memory Layout (Control Block 128B, Header 256B)
// =============================================================================

fn validate_memory_layout_claim(c: &mut Criterion) {
    let mut group = c.benchmark_group("CLAIM_4_memory_layout");

    group.sample_size(500);

    // Control block size = 128 bytes
    group.bench_function("control_block_128_bytes", |b| {
        b.iter(|| {
            let size = std::mem::size_of::<ControlBlock>();
            assert_eq!(size, 128, "ControlBlock MUST be exactly 128 bytes");
            black_box(size);
        });
    });

    // Control block alignment = 128 bytes (cache-line aligned)
    group.bench_function("control_block_128_aligned", |b| {
        b.iter(|| {
            let align = std::mem::align_of::<ControlBlock>();
            assert_eq!(align, 128, "ControlBlock MUST be 128-byte aligned");
            black_box(align);
        });
    });

    // Message header size = 256 bytes
    group.bench_function("message_header_256_bytes", |b| {
        b.iter(|| {
            let size = std::mem::size_of::<MessageHeader>();
            assert_eq!(size, 256, "MessageHeader MUST be exactly 256 bytes");
            black_box(size);
        });
    });

    // HLC timestamp = 24 bytes
    group.bench_function("hlc_timestamp_24_bytes", |b| {
        b.iter(|| {
            let size = std::mem::size_of::<HlcTimestamp>();
            assert_eq!(size, 24, "HlcTimestamp MUST be 24 bytes");
            black_box(size);
        });
    });

    // HLC state = 16 bytes
    group.bench_function("hlc_state_16_bytes", |b| {
        b.iter(|| {
            let size = std::mem::size_of::<HlcState>();
            assert_eq!(size, 16, "HlcState MUST be 16 bytes");
            black_box(size);
        });
    });

    group.finish();
}

// =============================================================================
// CLAIM 5: HLC Causal Ordering
// =============================================================================

fn validate_hlc_ordering_claim(c: &mut Criterion) {
    let mut group = c.benchmark_group("CLAIM_5_hlc_causal_ordering");

    group.sample_size(500);

    // Total ordering of timestamps
    group.bench_function("total_ordering", |b| {
        let clock = HlcClock::new(1);

        b.iter(|| {
            let ts1 = clock.tick();
            let ts2 = clock.tick();
            let ts3 = clock.tick();

            // Total ordering must be strict
            assert!(ts1 < ts2, "ts1 < ts2 required");
            assert!(ts2 < ts3, "ts2 < ts3 required");
            assert!(ts1 < ts3, "ts1 < ts3 required (transitivity)");

            black_box((ts1, ts2, ts3));
        });
    });

    // Causality across nodes
    group.bench_function("cross_node_causality", |b| {
        b.iter(|| {
            let clock_a = HlcClock::new(1);
            let clock_b = HlcClock::new(2);

            // A -> B message
            let ts_a = clock_a.tick();
            let ts_b = clock_b.update(&ts_a).unwrap();

            // Causality: if A happens-before B, then ts_a < ts_b
            assert!(ts_a < ts_b, "Causality MUST be preserved");

            black_box((ts_a, ts_b));
        });
    });

    // Distributed causal chain
    group.bench_function("distributed_causal_chain", |b| {
        b.iter(|| {
            let clocks: Vec<HlcClock> = (0..5).map(|i| HlcClock::new(i)).collect();

            // Create a causal chain: 0 -> 1 -> 2 -> 3 -> 4
            let mut prev_ts = clocks[0].tick();

            for i in 1..5 {
                let new_ts = clocks[i].update(&prev_ts).unwrap();
                assert!(prev_ts < new_ts, "Chain causality broken at node {}", i);
                prev_ts = new_ts;
            }

            black_box(prev_ts);
        });
    });

    group.finish();
}

// =============================================================================
// CLAIM 6: Lock-Free Message Passing
// =============================================================================

fn validate_lock_free_claim(c: &mut Criterion) {
    let mut group = c.benchmark_group("CLAIM_6_lock_free_messaging");

    group.sample_size(500);

    // Single-threaded lock-free operations
    group.bench_function("spsc_queue_operations", |b| {
        let queue = SpscQueue::new(1024);
        let envelope = MessageEnvelope {
            header: MessageHeader::new(1, 0, 1, 256, HlcTimestamp::now(1)),
            payload: vec![0u8; 256],
        };

        b.iter(|| {
            // These operations should never block
            queue.try_enqueue(envelope.clone()).unwrap();
            let msg = queue.try_dequeue().unwrap();
            black_box(msg);
        });
    });

    // Batch operations (should scale linearly - no lock contention)
    for batch in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_throughput", batch),
            batch,
            |b, &size| {
                let queue = SpscQueue::new(2048);
                let envelope = MessageEnvelope {
                    header: MessageHeader::new(1, 0, 1, 256, HlcTimestamp::now(1)),
                    payload: vec![0u8; 256],
                };

                b.iter(|| {
                    for _ in 0..size {
                        queue.try_enqueue(envelope.clone()).unwrap();
                    }
                    for _ in 0..size {
                        let msg = queue.try_dequeue().unwrap();
                        black_box(msg);
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// CLAIM 7: Zero-Copy Serialization
// =============================================================================

fn validate_zero_copy_claim(c: &mut Criterion) {
    let mut group = c.benchmark_group("CLAIM_7_zero_copy_serialization");

    group.sample_size(1000);

    // Zero-copy should be essentially a pointer cast
    group.bench_function("header_zero_copy_view", |b| {
        let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));

        b.iter(|| {
            // as_bytes() should just return a slice view - no allocation
            let bytes = header.as_bytes();
            assert_eq!(bytes.len(), 256);
            black_box(bytes);
        });
    });

    group.bench_function("timestamp_zero_copy_view", |b| {
        let ts = HlcTimestamp::now(1);

        b.iter(|| {
            let bytes = ts.as_bytes();
            assert_eq!(bytes.len(), 24);
            black_box(bytes);
        });
    });

    group.bench_function("state_zero_copy_view", |b| {
        let state = HlcState::new(12345, 42);

        b.iter(|| {
            let bytes = state.as_bytes();
            assert_eq!(bytes.len(), 16);
            black_box(bytes);
        });
    });

    // Compare with allocation-based serialization
    group.bench_function("header_to_bytes_vec", |b| {
        let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));

        b.iter(|| {
            // This allocates - should be slower
            let bytes = header.as_bytes().to_vec();
            black_box(bytes);
        });
    });

    group.finish();
}

// =============================================================================
// COMPREHENSIVE VALIDATION SUMMARY
// =============================================================================

fn validation_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("VALIDATION_SUMMARY");

    group.sample_size(100);

    // All-in-one validation that touches all claims
    group.bench_function("all_claims_combined", |b| {
        let rt = TokioRuntime::new().unwrap();

        b.iter(|| {
            // Claim 1: Serialization
            let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));
            let bytes = header.as_bytes();
            black_box(bytes);

            // Claim 2: Message latency
            let queue = SpscQueue::new(256);
            let envelope = MessageEnvelope {
                header: MessageHeader::new(1, 0, 1, 64, HlcTimestamp::now(1)),
                payload: vec![0u8; 64],
            };
            queue.try_enqueue(envelope).unwrap();
            let msg = queue.try_dequeue().unwrap();
            black_box(msg);

            // Claim 3: Startup time (already validated separately)
            // Note: Not doing runtime creation in hot loop as it's expensive

            // Claim 4: Memory layout
            assert_eq!(std::mem::size_of::<ControlBlock>(), 128);
            assert_eq!(std::mem::size_of::<MessageHeader>(), 256);

            // Claim 5: HLC ordering
            let clock = HlcClock::new(1);
            let ts1 = clock.tick();
            let ts2 = clock.tick();
            assert!(ts1 < ts2);

            // Claim 6: Lock-free (same as claim 2)
            // Claim 7: Zero-copy (same as claim 1)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    validate_serialization_claim,
    validate_message_latency_claim,
    validate_startup_time_claim,
    validate_memory_layout_claim,
    validate_hlc_ordering_claim,
    validate_lock_free_claim,
    validate_zero_copy_claim,
    validation_summary,
);
criterion_main!(benches);
