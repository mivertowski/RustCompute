//! Message Queue Benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use ringkernel_core::hlc::HlcTimestamp;
use ringkernel_core::message::{MessageEnvelope, MessageHeader};
use ringkernel_core::queue::{MessageQueue, SpscQueue};

fn make_envelope(size: usize) -> MessageEnvelope {
    MessageEnvelope {
        header: MessageHeader::new(1, 0, 1, size, HlcTimestamp::now(1)),
        payload: vec![0u8; size],
    }
}

fn bench_enqueue_dequeue(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_queue");

    for size in [64, 256, 1024, 4096].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_function(format!("enqueue_dequeue_{}", size), |b| {
            let queue = SpscQueue::new(1024);
            let envelope = make_envelope(*size);

            b.iter(|| {
                queue.try_enqueue(envelope.clone()).unwrap();
                black_box(queue.try_dequeue().unwrap());
            });
        });
    }

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("queue_throughput");

    group.bench_function("batch_1000_messages", |b| {
        let queue = SpscQueue::new(2048);
        let envelope = make_envelope(256);

        b.iter(|| {
            // Enqueue batch
            for _ in 0..1000 {
                queue.try_enqueue(envelope.clone()).unwrap();
            }

            // Dequeue batch
            for _ in 0..1000 {
                black_box(queue.try_dequeue().unwrap());
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_enqueue_dequeue, bench_throughput);
criterion_main!(benches);
