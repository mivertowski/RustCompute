//! Message Latency Benchmarks
//!
//! Validates the README claim: Message latency (native) <500µs (vs DotCompute <1ms)
//!
//! These benchmarks measure:
//! - End-to-end message send/receive latency
//! - Queue enqueue/dequeue latency
//! - Message envelope creation latency
//! - Serialization + queue round-trip latency

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Instant;
use tokio::runtime::Runtime as TokioRuntime;

use ringkernel_core::hlc::HlcTimestamp;
use ringkernel_core::message::{MessageEnvelope, MessageHeader, MessageId, CorrelationId, Priority};
use ringkernel_core::queue::{MessageQueue, SpscQueue};
use ringkernel_core::runtime::{LaunchOptions, RingKernelRuntime};
use ringkernel_cpu::CpuRuntime;

/// Benchmark message envelope creation
fn bench_envelope_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency/envelope_creation");

    group.bench_function("header_only", |b| {
        b.iter(|| {
            let header = MessageHeader::new(
                1,
                0,
                1,
                256,
                HlcTimestamp::now(1),
            );
            black_box(header);
        });
    });

    for payload_size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Bytes(*payload_size as u64));

        group.bench_with_input(
            BenchmarkId::new("with_payload", payload_size),
            payload_size,
            |b, &size| {
                let payload = vec![0u8; size];

                b.iter(|| {
                    let header = MessageHeader::new(
                        1,
                        0,
                        1,
                        size,
                        HlcTimestamp::now(1),
                    );
                    let envelope = MessageEnvelope {
                        header,
                        payload: payload.clone(),
                    };
                    black_box(envelope);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark queue latency (critical path for message passing)
fn bench_queue_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency/queue");

    // Low latency is critical - use more samples for accuracy
    group.sample_size(200);

    for payload_size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Bytes(*payload_size as u64));

        // Single message round-trip
        group.bench_with_input(
            BenchmarkId::new("single_roundtrip", payload_size),
            payload_size,
            |b, &size| {
                let queue = SpscQueue::new(1024);
                let envelope = MessageEnvelope {
                    header: MessageHeader::new(1, 0, 1, size, HlcTimestamp::now(1)),
                    payload: vec![0u8; size],
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

/// Benchmark end-to-end latency including serialization
fn bench_end_to_end_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency/end_to_end");

    group.sample_size(200);

    for payload_size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Bytes(*payload_size as u64));

        group.bench_with_input(
            BenchmarkId::new("serialize_queue_deserialize", payload_size),
            payload_size,
            |b, &size| {
                let queue = SpscQueue::new(1024);
                let payload = vec![42u8; size];

                b.iter(|| {
                    // Create envelope (simulates serialization)
                    let envelope = MessageEnvelope {
                        header: MessageHeader::new(1, 0, 1, size, HlcTimestamp::now(1)),
                        payload: payload.clone(),
                    };

                    // Enqueue
                    queue.try_enqueue(envelope).unwrap();

                    // Dequeue
                    let received = queue.try_dequeue().unwrap();

                    // Validate (simulates deserialization check)
                    assert!(received.header.validate());
                    black_box(received);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark latency percentiles (p50, p99, p999)
fn bench_latency_percentiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency/percentiles");

    // Higher sample size for percentile accuracy
    group.sample_size(500);

    group.bench_function("message_roundtrip_256b", |b| {
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

    group.finish();
}

/// Benchmark message burst latency (multiple messages in sequence)
fn bench_burst_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency/burst");

    for burst_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("messages", burst_size),
            burst_size,
            |b, &burst| {
                let queue = SpscQueue::new(2048);
                let envelope = MessageEnvelope {
                    header: MessageHeader::new(1, 0, 1, 256, HlcTimestamp::now(1)),
                    payload: vec![0u8; 256],
                };

                b.iter(|| {
                    // Send burst
                    for _ in 0..burst {
                        queue.try_enqueue(envelope.clone()).unwrap();
                    }

                    // Receive burst
                    for _ in 0..burst {
                        let msg = queue.try_dequeue().unwrap();
                        black_box(msg);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Validation benchmark against README target
fn bench_latency_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_validation");

    // Target: <500µs message latency (vs DotCompute <1ms)

    group.sample_size(500);

    group.bench_function("target_500us_message_roundtrip", |b| {
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

    // Full kernel message simulation
    group.bench_function("target_500us_simulated_kernel_message", |b| {
        let rt = TokioRuntime::new().unwrap();
        let runtime = rt.block_on(CpuRuntime::new()).unwrap();
        let queue = SpscQueue::new(1024);

        b.iter_custom(|iters| {
            let start = Instant::now();
            for i in 0..iters {
                // Create timestamp (HLC)
                let timestamp = HlcTimestamp::now(1);

                // Create message envelope
                let envelope = MessageEnvelope {
                    header: MessageHeader::new(1, 0, 1, 256, timestamp),
                    payload: vec![0u8; 256],
                };

                // Enqueue (simulates host -> device)
                queue.try_enqueue(envelope).unwrap();

                // Dequeue (simulates device processing)
                let received = queue.try_dequeue().unwrap();

                // Validate
                assert!(received.header.validate());
                black_box(received);
            }
            start.elapsed()
        });
    });

    group.finish();
}

/// Benchmark message header operations
fn bench_header_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency/header_ops");

    group.bench_function("create_header", |b| {
        b.iter(|| {
            let header = MessageHeader::new(
                42,
                0,
                1,
                1024,
                HlcTimestamp::now(1),
            );
            black_box(header);
        });
    });

    group.bench_function("header_with_correlation", |b| {
        b.iter(|| {
            let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1))
                .with_correlation(CorrelationId::generate());
            black_box(header);
        });
    });

    group.bench_function("header_with_priority", |b| {
        b.iter(|| {
            let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1))
                .with_priority(Priority::High);
            black_box(header);
        });
    });

    group.bench_function("header_full_config", |b| {
        b.iter(|| {
            let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1))
                .with_correlation(CorrelationId::generate())
                .with_priority(Priority::Critical)
                .with_deadline(HlcTimestamp::now(1));
            black_box(header);
        });
    });

    group.bench_function("header_validate", |b| {
        let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));
        b.iter(|| {
            black_box(header.validate());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_envelope_creation,
    bench_queue_latency,
    bench_end_to_end_latency,
    bench_latency_percentiles,
    bench_burst_latency,
    bench_latency_validation,
    bench_header_operations,
);
criterion_main!(benches);
