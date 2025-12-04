//! Kernel performance benchmarks.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use ringkernel_procint::fabric::{
    FinanceConfig, GeneratorConfig, HealthcareConfig, ManufacturingConfig, ProcessEventGenerator,
    SectorTemplate,
};
use ringkernel_procint::kernels::{
    DfgConstructionKernel, PartialOrderKernel, PatternConfig, PatternDetectionKernel,
};

fn bench_dfg_construction(c: &mut Criterion) {
    let config = GeneratorConfig {
        events_per_second: 100_000,
        concurrent_cases: 100,
        ..Default::default()
    };
    let mut generator = ProcessEventGenerator::new(
        SectorTemplate::Healthcare(HealthcareConfig::default()),
        config,
    );
    let mut kernel = DfgConstructionKernel::new(64).with_cpu_only();

    let batch_sizes = [1000, 5000, 10000];

    let mut group = c.benchmark_group("DFG Construction");

    for size in batch_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(format!("batch_{}", size), |b| {
            b.iter(|| {
                let events = generator.generate_batch(size);
                let result = kernel.process(black_box(&events));
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_pattern_detection(c: &mut Criterion) {
    let config = GeneratorConfig {
        events_per_second: 100_000,
        concurrent_cases: 100,
        ..Default::default()
    };
    let mut generator = ProcessEventGenerator::new(
        SectorTemplate::Manufacturing(ManufacturingConfig::default()),
        config,
    );
    let mut dfg_kernel = DfgConstructionKernel::new(64).with_cpu_only();
    let mut pattern_kernel = PatternDetectionKernel::new(PatternConfig::default()).with_cpu_only();

    // Pre-generate DFG
    let events = generator.generate_batch(10000);
    let dfg_result = dfg_kernel.process(&events);
    let nodes = dfg_result.dfg.nodes().to_vec();

    let mut group = c.benchmark_group("Pattern Detection");
    group.throughput(Throughput::Elements(nodes.len() as u64));

    group.bench_function("detect_patterns", |b| {
        b.iter(|| {
            let result = pattern_kernel.detect(black_box(&nodes));
            black_box(result)
        })
    });

    group.finish();
}

fn bench_partial_order(c: &mut Criterion) {
    let config = GeneratorConfig {
        events_per_second: 100_000,
        concurrent_cases: 50,
        ..Default::default()
    };
    let mut generator =
        ProcessEventGenerator::new(SectorTemplate::Finance(FinanceConfig::default()), config);
    let mut kernel = PartialOrderKernel::new().with_cpu_only();

    let batch_sizes = [1000, 5000];

    let mut group = c.benchmark_group("Partial Order Derivation");

    for size in batch_sizes {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(format!("batch_{}", size), |b| {
            b.iter(|| {
                let events = generator.generate_batch(size);
                let result = kernel.derive(black_box(&events));
                black_box(result)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dfg_construction,
    bench_pattern_detection,
    bench_partial_order,
);
criterion_main!(benches);
