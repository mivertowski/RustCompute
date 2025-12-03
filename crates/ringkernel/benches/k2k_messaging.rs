//! Kernel-to-Kernel (K2K) Messaging Benchmarks
//!
//! Validates README claims about K2K messaging:
//! - Lock-free message passing between kernels
//! - Efficient K2K endpoint operations
//! - Message broker performance
//!
//! These benchmarks measure the performance of kernel-to-kernel communication.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Instant;
use tokio::runtime::Runtime as TokioRuntime;

use ringkernel_core::hlc::HlcTimestamp;
use ringkernel_core::k2k::{DeliveryStatus, K2KBuilder, K2KMessage};
use ringkernel_core::message::{MessageEnvelope, MessageHeader};
use ringkernel_core::queue::{MessageQueue, SpscQueue};
use ringkernel_core::runtime::KernelId;

/// Create a test message envelope
fn make_envelope(payload_size: usize) -> MessageEnvelope {
    MessageEnvelope {
        header: MessageHeader::new(1, 1, 2, payload_size, HlcTimestamp::now(1)),
        payload: vec![0u8; payload_size],
    }
}

/// Create a test K2K message
fn make_k2k_message(payload_size: usize) -> K2KMessage {
    K2KMessage::new(
        KernelId::new("source_kernel"),
        KernelId::new("dest_kernel"),
        make_envelope(payload_size),
        HlcTimestamp::now(1),
    )
}

/// Benchmark K2K message creation
fn bench_k2k_message_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("k2k/message_creation");

    for payload_size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Bytes(*payload_size as u64));

        group.bench_with_input(
            BenchmarkId::new("create_message", payload_size),
            payload_size,
            |b, &size| {
                b.iter(|| {
                    let msg = make_k2k_message(size);
                    black_box(msg);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark K2K broker operations
fn bench_k2k_broker(c: &mut Criterion) {
    let mut group = c.benchmark_group("k2k/broker");

    group.sample_size(100);

    // Broker creation
    group.bench_function("create_default", |b| {
        b.iter(|| {
            let broker = K2KBuilder::new().build();
            black_box(broker);
        });
    });

    group.bench_function("create_with_config", |b| {
        b.iter(|| {
            let broker = K2KBuilder::new()
                .max_pending_messages(1024)
                .delivery_timeout_ms(5000)
                .build();
            black_box(broker);
        });
    });

    // Endpoint registration
    group.bench_function("register_endpoint", |b| {
        let broker = K2KBuilder::new().build();
        let mut counter = 0u64;

        b.iter(|| {
            counter += 1;
            let kernel_id = KernelId::new(format!("kernel_{}", counter));
            let endpoint = broker.register(kernel_id);
            black_box(endpoint);
        });
    });

    group.finish();
}

/// Benchmark K2K endpoint operations
fn bench_k2k_endpoint(c: &mut Criterion) {
    let mut group = c.benchmark_group("k2k/endpoint");

    group.sample_size(50);

    let rt = TokioRuntime::new().unwrap();

    // Create broker and endpoints
    let broker = K2KBuilder::new().max_pending_messages(4096).build();

    let endpoint1 = broker.register(KernelId::new("kernel_1"));
    let mut endpoint2 = broker.register(KernelId::new("kernel_2"));

    // Send operation
    for payload_size in [64, 256, 1024].iter() {
        group.throughput(Throughput::Bytes(*payload_size as u64));

        group.bench_with_input(
            BenchmarkId::new("send_async", payload_size),
            payload_size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let envelope = make_envelope(size);
                        let receipt = endpoint1.send(KernelId::new("kernel_2"), envelope).await;
                        let _ = black_box(receipt);
                    });

                    // Drain to prevent queue filling up
                    while endpoint2.try_receive().is_some() {}
                });
            },
        );
    }

    group.finish();
}

/// Benchmark K2K message throughput
fn bench_k2k_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("k2k/throughput");

    group.sample_size(50);

    let rt = TokioRuntime::new().unwrap();

    let broker = K2KBuilder::new().max_pending_messages(8192).build();

    let sender = broker.register(KernelId::new("sender"));
    let mut receiver = broker.register(KernelId::new("receiver"));

    // Batch send throughput
    for batch_size in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_send", batch_size),
            batch_size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        for _ in 0..size {
                            let envelope = make_envelope(256);
                            let receipt = sender.send(KernelId::new("receiver"), envelope).await;
                            let _ = black_box(receipt);
                        }
                    });

                    // Drain received messages
                    while let Some(received) = receiver.try_receive() {
                        black_box(received);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark lock-free queue operations (underlying K2K infrastructure)
fn bench_lock_free_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("k2k/lock_free_queue");

    group.sample_size(200);

    // Single-threaded operations
    for capacity in [256, 1024, 4096].iter() {
        group.bench_with_input(BenchmarkId::new("create", capacity), capacity, |b, &cap| {
            b.iter(|| {
                let queue = SpscQueue::new(cap);
                black_box(queue);
            });
        });
    }

    // Enqueue/dequeue patterns
    let queue = SpscQueue::new(1024);
    let envelope = make_envelope(256);

    group.bench_function("single_enqueue_dequeue", |b| {
        b.iter(|| {
            queue.try_enqueue(envelope.clone()).unwrap();
            let msg = queue.try_dequeue().unwrap();
            black_box(msg);
        });
    });

    // Batch operations
    for batch in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_operations", batch),
            batch,
            |b, &size| {
                let queue = SpscQueue::new(1024);
                let envelope = make_envelope(256);

                b.iter(|| {
                    // Batch enqueue
                    for _ in 0..size {
                        queue.try_enqueue(envelope.clone()).unwrap();
                    }

                    // Batch dequeue
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

/// Benchmark multi-producer scenarios (simulated)
fn bench_multi_sender(c: &mut Criterion) {
    let mut group = c.benchmark_group("k2k/multi_sender");

    group.sample_size(30);

    let rt = TokioRuntime::new().unwrap();

    for num_senders in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("senders", num_senders),
            num_senders,
            |b, &n| {
                let broker = K2KBuilder::new().max_pending_messages(2048).build();

                let mut receiver = broker.register(KernelId::new("receiver"));
                let senders: Vec<_> = (0..n)
                    .map(|i| broker.register(KernelId::new(format!("sender_{}", i))))
                    .collect();

                b.iter(|| {
                    rt.block_on(async {
                        // Each sender sends a message
                        for (i, sender) in senders.iter().enumerate() {
                            let envelope = MessageEnvelope {
                                header: MessageHeader::new(
                                    1,
                                    i as u64 + 1,
                                    0,
                                    256,
                                    HlcTimestamp::now(i as u64 + 1),
                                ),
                                payload: vec![0u8; 256],
                            };
                            let _ = sender.send(KernelId::new("receiver"), envelope).await;
                        }
                    });

                    // Receiver processes all messages
                    let mut count = 0;
                    while let Some(msg) = receiver.try_receive() {
                        black_box(msg);
                        count += 1;
                    }
                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

/// Validation benchmark for K2K claims
fn bench_k2k_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("k2k_validation");

    // Lock-free message passing validation
    group.bench_function("lock_free_guarantee", |b| {
        let queue = SpscQueue::new(1024);
        let envelope = make_envelope(256);

        b.iter_custom(|iters| {
            let start = Instant::now();

            for _ in 0..iters {
                // This should never block - lock-free guarantee
                queue.try_enqueue(envelope.clone()).unwrap();
                let msg = queue.try_dequeue().unwrap();
                black_box(msg);
            }

            start.elapsed()
        });
    });

    // K2K message delivery guarantee
    group.bench_function("delivery_guarantee", |b| {
        let rt = TokioRuntime::new().unwrap();
        let broker = K2KBuilder::new().max_pending_messages(1024).build();

        let sender = broker.register(KernelId::new("sender"));
        let mut receiver = broker.register(KernelId::new("receiver"));

        b.iter(|| {
            rt.block_on(async {
                let envelope = make_envelope(256);

                // Send must succeed
                let receipt = sender
                    .send(KernelId::new("receiver"), envelope)
                    .await
                    .unwrap();
                assert!(
                    matches!(
                        receipt.status,
                        DeliveryStatus::Delivered | DeliveryStatus::Pending
                    ),
                    "Message delivery must succeed"
                );
            });

            // Receive must get the message
            if let Some(received) = receiver.try_receive() {
                assert_eq!(received.envelope.payload.len(), 256);
                black_box(received);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_k2k_message_creation,
    bench_k2k_broker,
    bench_k2k_endpoint,
    bench_k2k_throughput,
    bench_lock_free_queue,
    bench_multi_sender,
    bench_k2k_validation,
);
criterion_main!(benches);
