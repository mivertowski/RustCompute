//! Runtime Startup Benchmarks
//!
//! Validates the README claim: Startup time <1ms (vs DotCompute ~10ms)
//!
//! These benchmarks measure:
//! - CPU runtime initialization time
//! - Kernel launch time
//! - Full system startup (runtime + kernel)
//! - Backend detection time

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Instant;
use tokio::runtime::Runtime as TokioRuntime;

use ringkernel_core::runtime::{LaunchOptions, RingKernelRuntime};
use ringkernel_cpu::CpuRuntime;

/// Benchmark CPU runtime creation
fn bench_runtime_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("startup");

    // Configure for fast benchmarks (startup should be quick)
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(5));

    let rt = TokioRuntime::new().unwrap();

    group.bench_function("cpu_runtime_create", |b| {
        b.iter(|| {
            rt.block_on(async {
                let runtime = CpuRuntime::new().await.unwrap();
                black_box(runtime);
            });
        });
    });

    group.bench_function("cpu_runtime_with_node_id", |b| {
        b.iter(|| {
            rt.block_on(async {
                let runtime = CpuRuntime::with_node_id(42).await.unwrap();
                black_box(runtime);
            });
        });
    });

    group.finish();
}

/// Benchmark kernel launch time
fn bench_kernel_launch(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_launch");

    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(5));

    let rt = TokioRuntime::new().unwrap();

    // Single kernel launch
    group.bench_function("single_kernel", |b| {
        let runtime = rt.block_on(CpuRuntime::new()).unwrap();
        let mut counter = 0u64;

        b.iter(|| {
            counter += 1;
            let kernel_name = format!("bench_kernel_{}", counter);
            rt.block_on(async {
                let handle = runtime.launch(&kernel_name, LaunchOptions::default()).await.unwrap();
                black_box(handle);
            });
        });
    });

    // Kernel with custom options
    group.bench_function("kernel_with_options", |b| {
        let runtime = rt.block_on(CpuRuntime::new()).unwrap();
        let mut counter = 0u64;

        let options = LaunchOptions {
            grid_size: 256,
            block_size: 64,
            auto_activate: true,
            ..Default::default()
        };

        b.iter(|| {
            counter += 1;
            let kernel_name = format!("bench_opts_kernel_{}", counter);
            rt.block_on(async {
                let handle = runtime.launch(&kernel_name, options.clone()).await.unwrap();
                black_box(handle);
            });
        });
    });

    group.finish();
}

/// Benchmark full system startup (runtime + kernel)
fn bench_full_startup(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_startup");

    group.sample_size(30);
    group.measurement_time(std::time::Duration::from_secs(10));

    let rt = TokioRuntime::new().unwrap();

    group.bench_function("runtime_and_kernel", |b| {
        let mut counter = 0u64;

        b.iter(|| {
            counter += 1;
            let kernel_name = format!("full_startup_kernel_{}", counter);

            rt.block_on(async {
                let runtime = CpuRuntime::new().await.unwrap();
                let handle = runtime.launch(&kernel_name, LaunchOptions::default()).await.unwrap();
                black_box(handle);
                runtime.shutdown().await.unwrap();
            });
        });
    });

    // Multiple kernels startup
    for num_kernels in [1, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("multi_kernel_startup", num_kernels),
            num_kernels,
            |b, &num| {
                let mut counter = 0u64;

                b.iter(|| {
                    counter += 1;
                    rt.block_on(async {
                        let runtime = CpuRuntime::new().await.unwrap();

                        for i in 0..num {
                            let kernel_name = format!("multi_kernel_{}_{}", counter, i);
                            let handle = runtime.launch(&kernel_name, LaunchOptions::default()).await.unwrap();
                            black_box(handle);
                        }

                        runtime.shutdown().await.unwrap();
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark shutdown time
fn bench_shutdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("shutdown");

    group.sample_size(50);

    let rt = TokioRuntime::new().unwrap();

    // Empty runtime shutdown
    group.bench_function("empty_runtime", |b| {
        b.iter(|| {
            rt.block_on(async {
                let runtime = CpuRuntime::new().await.unwrap();
                runtime.shutdown().await.unwrap();
            });
        });
    });

    // Runtime with kernels shutdown
    for num_kernels in [1, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("with_kernels", num_kernels),
            num_kernels,
            |b, &num| {
                let mut counter = 0u64;

                b.iter(|| {
                    counter += 1;
                    rt.block_on(async {
                        let runtime = CpuRuntime::new().await.unwrap();

                        for i in 0..num {
                            let kernel_name = format!("shutdown_kernel_{}_{}", counter, i);
                            runtime.launch(&kernel_name, LaunchOptions::default()).await.unwrap();
                        }

                        runtime.shutdown().await.unwrap();
                    });
                });
            },
        );
    }

    group.finish();
}

/// Measure startup times and report against targets
fn bench_startup_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("startup_validation");

    // This benchmark validates against the README claim
    // Target: <1ms startup (vs DotCompute ~10ms)

    group.sample_size(100);

    let rt = TokioRuntime::new().unwrap();

    group.bench_function("target_1ms_runtime_create", |b| {
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

    group.bench_function("target_1ms_full_startup", |b| {
        let mut counter = 0u64;

        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                counter += 1;
                let kernel_name = format!("validation_kernel_{}", counter);

                rt.block_on(async {
                    let runtime = CpuRuntime::new().await.unwrap();
                    let handle = runtime.launch(&kernel_name, LaunchOptions::default()).await.unwrap();
                    black_box(handle);
                    runtime.shutdown().await.unwrap();
                });
            }
            start.elapsed()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_runtime_creation,
    bench_kernel_launch,
    bench_full_startup,
    bench_shutdown,
    bench_startup_validation,
);
criterion_main!(benches);
