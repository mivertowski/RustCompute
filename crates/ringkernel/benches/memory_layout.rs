//! Memory Layout Benchmarks
//!
//! Validates README claims about memory layout:
//! - Control block: 128 bytes, cache-line aligned
//! - Message header: 256 bytes, cache-line aligned
//! - Zero-copy serialization via rkyv/zerocopy
//!
//! These benchmarks verify that memory operations are efficient and
//! the claimed memory layouts are correct and performant.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use ringkernel_core::control::ControlBlock;
use ringkernel_core::hlc::{HlcState, HlcTimestamp};
use ringkernel_core::message::MessageHeader;
use zerocopy::AsBytes;

/// Compile-time layout assertions
const _: () = {
    assert!(std::mem::size_of::<ControlBlock>() == 128);
    assert!(std::mem::align_of::<ControlBlock>() == 128);
    assert!(std::mem::size_of::<MessageHeader>() == 256);
    assert!(std::mem::align_of::<MessageHeader>() == 64);
    assert!(std::mem::size_of::<HlcTimestamp>() == 24);
    assert!(std::mem::size_of::<HlcState>() == 16);
};

/// Benchmark control block operations
fn bench_control_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout/control_block");

    // Verify size claims
    group.bench_function("size_verification", |b| {
        b.iter(|| {
            let size = std::mem::size_of::<ControlBlock>();
            assert_eq!(size, 128, "Control block must be 128 bytes");
            black_box(size);
        });
    });

    group.bench_function("alignment_verification", |b| {
        b.iter(|| {
            let align = std::mem::align_of::<ControlBlock>();
            assert_eq!(align, 128, "Control block must be 128-byte aligned");
            black_box(align);
        });
    });

    // Creation benchmarks
    group.bench_function("create_new", |b| {
        b.iter(|| {
            let cb = ControlBlock::new();
            black_box(cb);
        });
    });

    group.bench_function("create_with_capacities", |b| {
        b.iter(|| {
            let cb = ControlBlock::with_capacities(1024, 1024);
            black_box(cb);
        });
    });

    // Access patterns
    group.bench_function("read_lifecycle_flags", |b| {
        let cb = ControlBlock::with_capacities(1024, 1024);
        b.iter(|| {
            black_box(cb.is_active());
            black_box(cb.should_terminate());
            black_box(cb.has_terminated());
        });
    });

    group.bench_function("read_queue_state", |b| {
        let cb = ControlBlock::with_capacities(1024, 1024);
        b.iter(|| {
            black_box(cb.input_queue_size());
            black_box(cb.output_queue_size());
            black_box(cb.input_queue_empty());
            black_box(cb.output_queue_empty());
        });
    });

    // Memory copy (DMA simulation)
    group.throughput(Throughput::Bytes(128));
    group.bench_function("memcpy_128b", |b| {
        let src = ControlBlock::with_capacities(1024, 1024);
        let mut dst = ControlBlock::new();

        b.iter(|| {
            // Simulate DMA transfer (copy 128 bytes)
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &src as *const ControlBlock,
                    &mut dst as *mut ControlBlock,
                    1,
                );
            }
            black_box(&dst);
        });
    });

    group.finish();
}

/// Benchmark message header operations
fn bench_message_header(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout/message_header");

    // Verify size claims
    group.bench_function("size_verification", |b| {
        b.iter(|| {
            let size = std::mem::size_of::<MessageHeader>();
            assert_eq!(size, 256, "Message header must be 256 bytes");
            black_box(size);
        });
    });

    group.bench_function("alignment_verification", |b| {
        b.iter(|| {
            let align = std::mem::align_of::<MessageHeader>();
            assert_eq!(align, 64, "Message header must be 64-byte aligned");
            black_box(align);
        });
    });

    // Creation benchmarks
    group.bench_function("create_new", |b| {
        b.iter(|| {
            let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));
            black_box(header);
        });
    });

    group.bench_function("create_default", |b| {
        b.iter(|| {
            let header = MessageHeader::default();
            black_box(header);
        });
    });

    // Zero-copy serialization (zerocopy crate)
    group.throughput(Throughput::Bytes(256));
    group.bench_function("as_bytes_zerocopy", |b| {
        let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));

        b.iter(|| {
            let bytes = header.as_bytes();
            black_box(bytes);
        });
    });

    group.bench_function("read_from_bytes", |b| {
        let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));
        let bytes = header.as_bytes().to_vec();

        b.iter(|| {
            let restored = MessageHeader::read_from(&bytes);
            black_box(restored);
        });
    });

    // Roundtrip (serialize + deserialize)
    group.bench_function("roundtrip", |b| {
        let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));

        b.iter(|| {
            let bytes = header.as_bytes();
            let restored = MessageHeader::read_from(bytes);
            black_box(restored);
        });
    });

    // Memory copy (DMA simulation)
    group.bench_function("memcpy_256b", |b| {
        let src = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));
        let mut dst = MessageHeader::default();

        b.iter(|| {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &src as *const MessageHeader,
                    &mut dst as *mut MessageHeader,
                    1,
                );
            }
            black_box(&dst);
        });
    });

    group.finish();
}

/// Benchmark HLC timestamp memory operations
fn bench_hlc_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout/hlc");

    group.bench_function("timestamp_size", |b| {
        b.iter(|| {
            let size = std::mem::size_of::<HlcTimestamp>();
            assert_eq!(size, 24, "HlcTimestamp must be 24 bytes");
            black_box(size);
        });
    });

    group.bench_function("state_size", |b| {
        b.iter(|| {
            let size = std::mem::size_of::<HlcState>();
            assert_eq!(size, 16, "HlcState must be 16 bytes");
            black_box(size);
        });
    });

    // Timestamp as bytes (zerocopy)
    group.throughput(Throughput::Bytes(24));
    group.bench_function("timestamp_as_bytes", |b| {
        let ts = HlcTimestamp::now(1);

        b.iter(|| {
            let bytes = ts.as_bytes();
            black_box(bytes);
        });
    });

    // State as bytes
    group.throughput(Throughput::Bytes(16));
    group.bench_function("state_as_bytes", |b| {
        let state = HlcState::new(12345678901234, 42);

        b.iter(|| {
            let bytes = state.as_bytes();
            black_box(bytes);
        });
    });

    group.finish();
}

/// Benchmark cache-line access patterns
fn bench_cache_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout/cache_efficiency");

    // Allocate array of control blocks
    group.bench_function("control_block_array_access", |b| {
        let blocks: Vec<ControlBlock> = (0..100)
            .map(|_| ControlBlock::with_capacities(1024, 1024))
            .collect();

        b.iter(|| {
            let mut sum: u64 = 0;
            for block in &blocks {
                sum = sum.wrapping_add(block.messages_processed);
            }
            black_box(sum);
        });
    });

    // Sequential vs random access
    group.bench_function("sequential_access", |b| {
        let blocks: Vec<ControlBlock> = (0..1000)
            .map(|_| ControlBlock::with_capacities(1024, 1024))
            .collect();

        b.iter(|| {
            let mut sum: u64 = 0;
            for block in &blocks {
                sum = sum.wrapping_add(block.messages_processed);
            }
            black_box(sum);
        });
    });

    group.finish();
}

/// Benchmark bulk memory operations
fn bench_bulk_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout/bulk_ops");

    for count in [10, 100, 1000].iter() {
        group.throughput(Throughput::Bytes((128 * count) as u64));

        group.bench_with_input(
            BenchmarkId::new("control_block_batch_copy", count),
            count,
            |b, &n| {
                let src: Vec<ControlBlock> = (0..n)
                    .map(|_| ControlBlock::with_capacities(1024, 1024))
                    .collect();
                let mut dst: Vec<ControlBlock> = vec![ControlBlock::new(); n];

                b.iter(|| {
                    dst.copy_from_slice(&src);
                    black_box(&dst);
                });
            },
        );
    }

    for count in [10, 100, 1000].iter() {
        group.throughput(Throughput::Bytes((256 * count) as u64));

        group.bench_with_input(
            BenchmarkId::new("message_header_batch_copy", count),
            count,
            |b, &n| {
                let src: Vec<MessageHeader> = (0..n)
                    .map(|_| MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1)))
                    .collect();
                let mut dst: Vec<MessageHeader> = vec![MessageHeader::default(); n];

                b.iter(|| {
                    dst.copy_from_slice(&src);
                    black_box(&dst);
                });
            },
        );
    }

    group.finish();
}

/// Validation benchmark for memory layout claims
fn bench_memory_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_validation");

    group.bench_function("validate_all_sizes", |b| {
        b.iter(|| {
            // Control block: 128 bytes
            let cb_size = std::mem::size_of::<ControlBlock>();
            assert_eq!(cb_size, 128, "ControlBlock size mismatch");

            // Control block: 128-byte aligned
            let cb_align = std::mem::align_of::<ControlBlock>();
            assert_eq!(cb_align, 128, "ControlBlock alignment mismatch");

            // Message header: 256 bytes
            let mh_size = std::mem::size_of::<MessageHeader>();
            assert_eq!(mh_size, 256, "MessageHeader size mismatch");

            // Message header: 64-byte aligned (cache-line)
            let mh_align = std::mem::align_of::<MessageHeader>();
            assert_eq!(mh_align, 64, "MessageHeader alignment mismatch");

            // HLC timestamp: 24 bytes
            let ts_size = std::mem::size_of::<HlcTimestamp>();
            assert_eq!(ts_size, 24, "HlcTimestamp size mismatch");

            // HLC state: 16 bytes
            let state_size = std::mem::size_of::<HlcState>();
            assert_eq!(state_size, 16, "HlcState size mismatch");

            black_box((cb_size, cb_align, mh_size, mh_align, ts_size, state_size));
        });
    });

    // Zero-copy verification
    group.bench_function("validate_zero_copy_serialization", |b| {
        let header = MessageHeader::new(42, 0, 1, 1024, HlcTimestamp::now(1));

        b.iter(|| {
            // Zero-copy should be essentially free (just returns a slice)
            let bytes = header.as_bytes();
            assert_eq!(bytes.len(), 256);

            // Read back without allocation
            let restored = MessageHeader::read_from(bytes);
            assert!(restored.is_some());
            assert!(restored.unwrap().validate());

            black_box(bytes);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_control_block,
    bench_message_header,
    bench_hlc_memory,
    bench_cache_efficiency,
    bench_bulk_operations,
    bench_memory_validation,
);
criterion_main!(benches);
