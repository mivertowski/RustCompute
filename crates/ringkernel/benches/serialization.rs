//! Serialization Benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use ringkernel_core::hlc::HlcTimestamp;
use ringkernel_core::message::MessageHeader;
use zerocopy::AsBytes;

fn bench_header_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    group.throughput(Throughput::Bytes(256)); // MessageHeader is 256 bytes

    group.bench_function("header_zerocopy", |b| {
        let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));

        b.iter(|| {
            let bytes = header.as_bytes();
            black_box(bytes);
        });
    });

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

fn bench_timestamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("hlc");

    group.bench_function("timestamp_now", |b| {
        b.iter(|| {
            black_box(HlcTimestamp::now(1));
        });
    });

    group.bench_function("timestamp_pack_unpack", |b| {
        let ts = HlcTimestamp::now(1);

        b.iter(|| {
            let packed = ts.pack();
            black_box(HlcTimestamp::unpack(packed));
        });
    });

    group.finish();
}

use zerocopy::FromBytes;

criterion_group!(benches, bench_header_serialization, bench_timestamp);
criterion_main!(benches);
