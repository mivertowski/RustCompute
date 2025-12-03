//! Hybrid Logical Clock Benchmarks
//!
//! Validates README claims about HLC:
//! - Causal ordering across distributed GPU kernels
//! - Total ordering of timestamps
//! - Bounded drift from real time
//!
//! These benchmarks measure HLC performance critical for message ordering.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use ringkernel_core::hlc::{HlcClock, HlcState, HlcTimestamp};

/// Benchmark timestamp creation
fn bench_timestamp_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hlc/timestamp");

    group.bench_function("now", |b| {
        b.iter(|| {
            let ts = HlcTimestamp::now(1);
            black_box(ts);
        });
    });

    group.bench_function("new_const", |b| {
        b.iter(|| {
            let ts = HlcTimestamp::new(12345678901234, 42, 1);
            black_box(ts);
        });
    });

    group.bench_function("zero", |b| {
        b.iter(|| {
            let ts = HlcTimestamp::zero();
            black_box(ts);
        });
    });

    group.finish();
}

/// Benchmark clock operations
fn bench_clock_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("hlc/clock");

    group.bench_function("create", |b| {
        b.iter(|| {
            let clock = HlcClock::new(1);
            black_box(clock);
        });
    });

    group.bench_function("tick", |b| {
        let clock = HlcClock::new(1);

        b.iter(|| {
            let ts = clock.tick();
            black_box(ts);
        });
    });

    group.bench_function("now_read", |b| {
        let clock = HlcClock::new(1);

        b.iter(|| {
            let ts = clock.now();
            black_box(ts);
        });
    });

    // Update from received message
    group.bench_function("update", |b| {
        let clock = HlcClock::new(1);
        let received = HlcTimestamp::now(2);

        b.iter(|| {
            let result = clock.update(&received);
            let _ = black_box(result);
        });
    });

    group.finish();
}

/// Benchmark timestamp comparison (ordering)
fn bench_timestamp_ordering(c: &mut Criterion) {
    let mut group = c.benchmark_group("hlc/ordering");

    group.bench_function("compare_timestamps", |b| {
        let ts1 = HlcTimestamp::new(100, 0, 1);
        let ts2 = HlcTimestamp::new(100, 1, 1);

        b.iter(|| {
            black_box(ts1 < ts2);
            black_box(ts1 > ts2);
            black_box(ts1 == ts2);
        });
    });

    group.bench_function("sort_timestamps", |b| {
        let mut timestamps: Vec<HlcTimestamp> = (0..100)
            .map(|i| HlcTimestamp::new(1000 + (i % 10) as u64, i as u64, (i % 3) as u64))
            .collect();

        b.iter(|| {
            timestamps.sort();
            black_box(&timestamps);
        });
    });

    group.bench_function("max_timestamp", |b| {
        let clock = HlcClock::new(1);
        let timestamps: Vec<HlcTimestamp> = (0..100).map(|_| clock.tick()).collect();

        b.iter(|| {
            let max = timestamps.iter().max();
            black_box(max);
        });
    });

    group.finish();
}

/// Benchmark pack/unpack operations
fn bench_pack_unpack(c: &mut Criterion) {
    let mut group = c.benchmark_group("hlc/pack_unpack");

    group.bench_function("pack", |b| {
        let ts = HlcTimestamp::now(1);

        b.iter(|| {
            let packed = ts.pack();
            black_box(packed);
        });
    });

    group.bench_function("unpack", |b| {
        let ts = HlcTimestamp::now(1);
        let packed = ts.pack();

        b.iter(|| {
            let unpacked = HlcTimestamp::unpack(packed);
            black_box(unpacked);
        });
    });

    group.bench_function("roundtrip", |b| {
        let ts = HlcTimestamp::now(1);

        b.iter(|| {
            let packed = ts.pack();
            let unpacked = HlcTimestamp::unpack(packed);
            black_box(unpacked);
        });
    });

    group.finish();
}

/// Benchmark concurrent HLC operations
fn bench_concurrent_hlc(c: &mut Criterion) {
    let mut group = c.benchmark_group("hlc/concurrent");

    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(10));

    // Single clock, multiple tick calls
    group.bench_function("single_clock_contention", |b| {
        let clock = Arc::new(HlcClock::new(1));

        b.iter_custom(|iters| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let clock = Arc::clone(&clock);
                    let per_thread = iters / 4;
                    thread::spawn(move || {
                        for _ in 0..per_thread {
                            let ts = clock.tick();
                            black_box(ts);
                        }
                    })
                })
                .collect();

            let start = Instant::now();
            for handle in handles {
                handle.join().unwrap();
            }
            start.elapsed()
        });
    });

    // Multiple clocks simulating distributed nodes
    for num_nodes in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("distributed_nodes", num_nodes),
            num_nodes,
            |b, &nodes| {
                let clocks: Vec<Arc<HlcClock>> = (0..nodes)
                    .map(|i| Arc::new(HlcClock::new(i as u64)))
                    .collect();

                b.iter(|| {
                    // Each node generates a timestamp
                    let timestamps: Vec<HlcTimestamp> =
                        clocks.iter().map(|clock| clock.tick()).collect();

                    // Simulate message exchange - each node updates from others
                    for (i, clock) in clocks.iter().enumerate() {
                        for (j, ts) in timestamps.iter().enumerate() {
                            if i != j {
                                let _ = clock.update(ts);
                            }
                        }
                    }

                    black_box(&timestamps);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark causal ordering properties
fn bench_causality(c: &mut Criterion) {
    let mut group = c.benchmark_group("hlc/causality");

    // Verify causality is maintained
    group.bench_function("causal_chain", |b| {
        let clock1 = HlcClock::new(1);
        let clock2 = HlcClock::new(2);

        b.iter(|| {
            // Event A on node 1
            let ts_a = clock1.tick();

            // Message sent to node 2
            let ts_b = clock2.update(&ts_a).unwrap();

            // Verify causality
            assert!(ts_a < ts_b, "Causality must be preserved");

            black_box((ts_a, ts_b));
        });
    });

    // Verify happens-before relationship
    group.bench_function("happens_before_chain", |b| {
        let clocks: Vec<HlcClock> = (0..5).map(|i| HlcClock::new(i as u64)).collect();

        b.iter(|| {
            let mut prev_ts = clocks[0].tick();

            for clock in clocks.iter().skip(1) {
                let new_ts = clock.update(&prev_ts).unwrap();
                assert!(prev_ts < new_ts, "Happens-before must be maintained");
                prev_ts = new_ts;
            }

            black_box(prev_ts);
        });
    });

    group.finish();
}

/// Benchmark HLC state operations (GPU-side compact state)
fn bench_hlc_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("hlc/state");

    group.bench_function("create", |b| {
        b.iter(|| {
            let state = HlcState::new(12345678901234, 42);
            black_box(state);
        });
    });

    group.bench_function("to_timestamp", |b| {
        let state = HlcState::new(12345678901234, 42);

        b.iter(|| {
            let ts = state.to_timestamp(1);
            black_box(ts);
        });
    });

    group.bench_function("from_timestamp", |b| {
        let ts = HlcTimestamp::now(1);

        b.iter(|| {
            let state = HlcState::from_timestamp(&ts);
            black_box(state);
        });
    });

    group.bench_function("roundtrip", |b| {
        let ts = HlcTimestamp::now(1);

        b.iter(|| {
            let state = HlcState::from_timestamp(&ts);
            let restored = state.to_timestamp(ts.node_id);
            assert_eq!(ts.physical, restored.physical);
            assert_eq!(ts.logical, restored.logical);
            black_box(restored);
        });
    });

    group.finish();
}

/// Benchmark throughput of HLC tick operations
fn bench_hlc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("hlc/throughput");

    for batch_size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("tick_batch", batch_size),
            batch_size,
            |b, &size| {
                let clock = HlcClock::new(1);

                b.iter(|| {
                    for _ in 0..size {
                        let ts = clock.tick();
                        black_box(ts);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Validation benchmark for HLC claims
fn bench_hlc_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hlc_validation");

    // Total ordering must be maintained
    group.bench_function("total_ordering_guarantee", |b| {
        let clock = HlcClock::new(1);

        b.iter(|| {
            let ts1 = clock.tick();
            let ts2 = clock.tick();
            let ts3 = clock.tick();

            // Strict ordering must hold
            assert!(ts1 < ts2, "ts1 < ts2 must hold");
            assert!(ts2 < ts3, "ts2 < ts3 must hold");
            assert!(ts1 < ts3, "ts1 < ts3 must hold (transitivity)");

            black_box((ts1, ts2, ts3));
        });
    });

    // Causality across nodes must be preserved
    group.bench_function("causality_guarantee", |b| {
        b.iter(|| {
            let clock_a = HlcClock::new(1);
            let clock_b = HlcClock::new(2);

            // A sends message to B
            let ts_a = clock_a.tick();
            let ts_b = clock_b.update(&ts_a).unwrap();

            // Causality: ts_a -> ts_b means ts_a < ts_b
            assert!(ts_a < ts_b, "Causality must be preserved");

            // B sends message back to A
            let ts_a2 = clock_a.update(&ts_b).unwrap();

            // Full causal chain
            assert!(ts_b < ts_a2, "Return message must be causally after");
            assert!(ts_a < ts_a2, "Original event must be before final event");

            black_box((ts_a, ts_b, ts_a2));
        });
    });

    // Bounded drift (HLC should stay close to wall clock)
    group.bench_function("bounded_drift", |b| {
        let clock = HlcClock::new(1);

        b.iter(|| {
            let wall_before = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;

            let ts = clock.tick();

            let wall_after = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;

            // HLC physical should be within wall clock bounds (with some tolerance)
            assert!(
                ts.physical >= wall_before.saturating_sub(1000),
                "HLC should not be too far in the past"
            );
            assert!(
                ts.physical <= wall_after + 1000,
                "HLC should not be too far in the future"
            );

            black_box(ts);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_timestamp_creation,
    bench_clock_operations,
    bench_timestamp_ordering,
    bench_pack_unpack,
    bench_concurrent_hlc,
    bench_causality,
    bench_hlc_state,
    bench_hlc_throughput,
    bench_hlc_validation,
);
criterion_main!(benches);
