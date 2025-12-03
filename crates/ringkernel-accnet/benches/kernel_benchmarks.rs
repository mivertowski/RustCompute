//! Benchmarks for accounting network kernels.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ringkernel_accnet::prelude::*;
use ringkernel_accnet::kernels::{TransformationKernel, AnalysisKernel, TemporalKernel};

fn benchmark_transformation(c: &mut Criterion) {
    let kernel = TransformationKernel::default();

    let mut group = c.benchmark_group("transformation");

    for size in [100, 1000, 10000] {
        // Generate test entries
        let archetype = CompanyArchetype::retail_default();
        let template = ChartOfAccountsTemplate::retail_standard();
        let mut generator = TransactionGenerator::new(archetype, template, GeneratorConfig::default());
        let entries = generator.generate_batch(size);

        group.bench_with_input(
            BenchmarkId::new("journal_transform", size),
            &entries,
            |b, entries| {
                b.iter(|| kernel.transform(black_box(entries)))
            },
        );
    }

    group.finish();
}

fn benchmark_analysis(c: &mut Criterion) {
    let transform_kernel = TransformationKernel::default();
    let analysis_kernel = AnalysisKernel::default();

    let mut group = c.benchmark_group("analysis");

    for size in [100, 500, 1000] {
        // Generate test network
        let archetype = CompanyArchetype::retail_default();
        let template = ChartOfAccountsTemplate::retail_standard();
        let mut generator = TransactionGenerator::new(archetype, template, GeneratorConfig::default());
        let entries = generator.generate_batch(size);
        let result = transform_kernel.transform(&entries);

        let entity_id = uuid::Uuid::new_v4();
        let mut network = AccountingNetwork::new(entity_id, 2024, 1);

        for (idx, def) in template.accounts.iter().enumerate() {
            let node = AccountNode::new(uuid::Uuid::new_v4(), def.account_type, idx as u16);
            let metadata = AccountMetadata::new(&def.code, &def.name);
            network.add_account(node, metadata);
        }

        for flow in result.flows {
            network.add_flow(flow);
        }

        group.bench_with_input(
            BenchmarkId::new("network_analysis", size),
            &network,
            |b, network| {
                b.iter(|| analysis_kernel.analyze(black_box(network)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_transformation, benchmark_analysis);
criterion_main!(benches);
