//! Interactive Benchmark: Persistent vs Traditional GPU Actors
//!
//! This benchmark demonstrates where persistent GPU actors excel:
//! - Interactive command latency (not raw throughput)
//! - Fine-grained operations (not bulk batch processing)
//! - Query-heavy workloads (not pure compute)
//!
//! The key insight: Traditional kernels have ~50-100µs overhead per command
//! (cudaMemcpy + launch + sync), while persistent actors have ~5-15µs
//! (mapped memory write + poll).

use std::time::{Duration, Instant};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Interactive Benchmark: Persistent vs Traditional GPU Actors          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    println!("This benchmark demonstrates when persistent GPU actors excel:\n");
    println!("  • Interactive command latency (user input → GPU response)");
    println!("  • Fine-grained operations (many small commands)");
    println!("  • Query-heavy workloads (frequent state inspection)\n");

    // Run traditional baseline
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  TRADITIONAL APPROACH (kernel launch per command)");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    #[allow(unused_variables)]
    let traditional_results = run_traditional_benchmark();

    // Run persistent benchmark
    #[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
    {
        println!("\n═══════════════════════════════════════════════════════════════════════════");
        println!("  PERSISTENT APPROACH (commands to running kernel)");
        println!("═══════════════════════════════════════════════════════════════════════════\n");

        let persistent_results = run_persistent_benchmark();

        // Compare results
        println!("\n═══════════════════════════════════════════════════════════════════════════");
        println!("  COMPARISON SUMMARY");
        println!("═══════════════════════════════════════════════════════════════════════════\n");

        print_comparison(&traditional_results, &persistent_results);
    }

    #[cfg(not(all(feature = "cuda", feature = "cuda-codegen")))]
    {
        println!("\n⚠️  Persistent benchmark requires --features cuda-codegen");
        println!("   Run with: cargo run --bin interactive-benchmark --release --features cuda-codegen\n");
    }
}

/// Results from a benchmark run
#[derive(Debug, Clone)]
#[allow(dead_code)] // Used conditionally with cuda-codegen feature
struct BenchmarkResults {
    /// Name of the approach
    name: String,
    /// Latency for single-cell query (read one value)
    query_latency: LatencyStats,
    /// Latency for impulse injection
    inject_latency: LatencyStats,
    /// Latency for running a single simulation step
    single_step_latency: LatencyStats,
    /// Throughput for mixed interactive workload
    mixed_workload_time: Duration,
    /// Number of operations in mixed workload
    mixed_workload_ops: usize,
}

#[derive(Debug, Clone)]
struct LatencyStats {
    min: Duration,
    max: Duration,
    mean: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    samples: usize,
}

impl LatencyStats {
    fn from_samples(mut samples: Vec<Duration>) -> Self {
        samples.sort();
        let n = samples.len();
        let sum: Duration = samples.iter().sum();

        Self {
            min: samples[0],
            max: samples[n - 1],
            mean: sum / n as u32,
            p50: samples[n / 2],
            p95: samples[n * 95 / 100],
            p99: samples[n * 99 / 100],
            samples: n,
        }
    }

    fn print(&self, name: &str) {
        println!("  {}: ", name);
        println!(
            "    mean: {:>8.2}µs | p50: {:>8.2}µs | p95: {:>8.2}µs | p99: {:>8.2}µs",
            self.mean.as_secs_f64() * 1e6,
            self.p50.as_secs_f64() * 1e6,
            self.p95.as_secs_f64() * 1e6,
            self.p99.as_secs_f64() * 1e6,
        );
        println!(
            "    min:  {:>8.2}µs | max: {:>8.2}µs | samples: {}",
            self.min.as_secs_f64() * 1e6,
            self.max.as_secs_f64() * 1e6,
            self.samples
        );
    }
}

/// Traditional approach: Launch kernel for each command
fn run_traditional_benchmark() -> BenchmarkResults {
    use ringkernel_wavesim3d::simulation::{ComputationMethod, SimulationConfig};

    let width = 64;
    let height = 64;
    let depth = 64;
    let num_samples = 100;

    println!("Configuration:");
    println!("  Grid: {}×{}×{}", width, height, depth);
    println!("  Samples per test: {}", num_samples);
    println!();

    // Create stencil-based engine (traditional kernel-per-step)
    let prefer_gpu = cfg!(feature = "cuda");
    let config = SimulationConfig {
        width,
        height,
        depth,
        cell_size: 0.1,
        prefer_gpu,
        computation_method: ComputationMethod::Stencil,
        ..Default::default()
    };

    let mut engine = config.build();

    println!("  Using GPU: {}", engine.is_using_gpu());

    // Warm up
    engine.step_n(10);

    // 1. Query latency (read single cell)
    println!("Testing query latency (read single cell value)...");
    let query_samples: Vec<Duration> = (0..num_samples)
        .map(|i| {
            let x = (i * 7) % width;
            let y = (i * 11) % height;
            let z = (i * 13) % depth;

            let start = Instant::now();
            // Traditional: must read entire grid or do cudaMemcpy for single value
            let _value = engine.pressure_at(x, y, z);
            start.elapsed()
        })
        .collect();

    let query_latency = LatencyStats::from_samples(query_samples);
    query_latency.print("Query latency");

    // 2. Inject latency (write impulse)
    println!("\nTesting inject latency (write single impulse)...");
    let inject_samples: Vec<Duration> = (0..num_samples)
        .map(|i| {
            let x = (i * 7) % width;
            let y = (i * 11) % height;
            let z = (i * 13) % depth;

            let start = Instant::now();
            engine.inject_impulse(x, y, z, 0.1);
            start.elapsed()
        })
        .collect();

    let inject_latency = LatencyStats::from_samples(inject_samples);
    inject_latency.print("Inject latency");

    // 3. Single step latency
    println!("\nTesting single step latency...");
    let step_samples: Vec<Duration> = (0..num_samples)
        .map(|_| {
            let start = Instant::now();
            engine.step();
            start.elapsed()
        })
        .collect();

    let single_step_latency = LatencyStats::from_samples(step_samples);
    single_step_latency.print("Single step latency");

    // 4. Mixed interactive workload
    // Simulates real interactive use: inject, step, query, repeat
    println!("\nTesting mixed interactive workload...");
    println!("  Pattern: inject → step(5) → query → repeat 100x");

    let mixed_ops = 100;
    let mixed_start = Instant::now();
    for i in 0..mixed_ops {
        let x = (i * 7) % width;
        let y = (i * 11) % height;
        let z = (i * 13) % depth;

        // Inject impulse
        engine.inject_impulse(x, y, z, 0.1);
        // Run 5 steps
        engine.step_n(5);
        // Query result
        let _value = engine.pressure_at(width / 2, height / 2, depth / 2);
    }
    let mixed_workload_time = mixed_start.elapsed();

    println!(
        "  Mixed workload: {} ops in {:.2}ms ({:.2}µs/op)",
        mixed_ops * 3, // inject + step + query per iteration
        mixed_workload_time.as_secs_f64() * 1000.0,
        mixed_workload_time.as_secs_f64() * 1e6 / (mixed_ops * 3) as f64
    );

    BenchmarkResults {
        name: "Traditional (kernel-per-command)".to_string(),
        query_latency,
        inject_latency,
        single_step_latency,
        mixed_workload_time,
        mixed_workload_ops: mixed_ops * 3,
    }
}

/// Persistent approach: Commands to already-running kernel
#[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
fn run_persistent_benchmark() -> BenchmarkResults {
    use ringkernel_wavesim3d::simulation::{
        grid3d::SimulationGrid3D,
        persistent_backend::{PersistentBackend, PersistentBackendConfig},
        AcousticParams3D, Environment,
    };

    let width = 24; // Smaller grid for cooperative limits
    let height = 24;
    let depth = 24;
    let num_samples = 100;

    println!("Configuration:");
    println!("  Grid: {}×{}×{} (cooperative limit)", width, height, depth);
    println!("  Samples per test: {}", num_samples);
    println!();

    // Create grid and persistent backend
    let params = AcousticParams3D::new(Environment::default(), 0.1);
    let mut grid = SimulationGrid3D::new(width, height, depth, params);

    let config = PersistentBackendConfig::default();
    let mut backend = match PersistentBackend::new(&grid, config) {
        Ok(b) => b,
        Err(e) => {
            println!("Failed to create persistent backend: {}", e);
            return BenchmarkResults {
                name: "Persistent (ERROR)".to_string(),
                query_latency: LatencyStats::from_samples(vec![Duration::ZERO]),
                inject_latency: LatencyStats::from_samples(vec![Duration::ZERO]),
                single_step_latency: LatencyStats::from_samples(vec![Duration::ZERO]),
                mixed_workload_time: Duration::ZERO,
                mixed_workload_ops: 0,
            };
        }
    };

    // Start kernel
    if let Err(e) = backend.start() {
        println!("Failed to start kernel: {}", e);
        return BenchmarkResults {
            name: "Persistent (ERROR)".to_string(),
            query_latency: LatencyStats::from_samples(vec![Duration::ZERO]),
            inject_latency: LatencyStats::from_samples(vec![Duration::ZERO]),
            single_step_latency: LatencyStats::from_samples(vec![Duration::ZERO]),
            mixed_workload_time: Duration::ZERO,
            mixed_workload_ops: 0,
        };
    }

    // Warm up
    let _ = backend.step(&mut grid, 10);

    // 1. Query latency (read stats - equivalent to reading state)
    println!("Testing query latency (read kernel state)...");
    let query_samples: Vec<Duration> = (0..num_samples)
        .map(|_| {
            let start = Instant::now();
            // Persistent: just read from mapped memory
            let _stats = backend.stats();
            start.elapsed()
        })
        .collect();

    let query_latency = LatencyStats::from_samples(query_samples);
    query_latency.print("Query latency");

    // 2. Inject latency (send inject command)
    println!("\nTesting inject latency (send inject command)...");
    let inject_samples: Vec<Duration> = (0..num_samples)
        .map(|i| {
            let x = ((i * 7) % width) as u32;
            let y = ((i * 11) % height) as u32;
            let z = ((i * 13) % depth) as u32;

            let start = Instant::now();
            let _ = backend.inject_impulse(x, y, z, 0.1);
            start.elapsed()
        })
        .collect();

    let inject_latency = LatencyStats::from_samples(inject_samples);
    inject_latency.print("Inject latency");

    // 3. Single step latency
    println!("\nTesting single step latency...");
    let step_samples: Vec<Duration> = (0..num_samples)
        .map(|_| {
            let start = Instant::now();
            let _ = backend.step(&mut grid, 1);
            start.elapsed()
        })
        .collect();

    let single_step_latency = LatencyStats::from_samples(step_samples);
    single_step_latency.print("Single step latency");

    // 4. Mixed interactive workload
    println!("\nTesting mixed interactive workload...");
    println!("  Pattern: inject → step(5) → query → repeat 100x");

    let mixed_ops = 100;
    let mixed_start = Instant::now();
    for i in 0..mixed_ops {
        let x = ((i * 7) % width) as u32;
        let y = ((i * 11) % height) as u32;
        let z = ((i * 13) % depth) as u32;

        // Inject impulse
        let _ = backend.inject_impulse(x, y, z, 0.1);
        // Run 5 steps
        let _ = backend.step(&mut grid, 5);
        // Query result
        let _stats = backend.stats();
    }
    let mixed_workload_time = mixed_start.elapsed();

    println!(
        "  Mixed workload: {} ops in {:.2}ms ({:.2}µs/op)",
        mixed_ops * 3,
        mixed_workload_time.as_secs_f64() * 1000.0,
        mixed_workload_time.as_secs_f64() * 1e6 / (mixed_ops * 3) as f64
    );

    // Shutdown
    let _ = backend.shutdown();

    BenchmarkResults {
        name: "Persistent (single kernel)".to_string(),
        query_latency,
        inject_latency,
        single_step_latency,
        mixed_workload_time,
        mixed_workload_ops: mixed_ops * 3,
    }
}

#[allow(dead_code)]
fn print_comparison(traditional: &BenchmarkResults, persistent: &BenchmarkResults) {
    println!("┌─────────────────────────┬──────────────────┬──────────────────┬──────────┐");
    println!("│ Operation               │ Traditional      │ Persistent       │ Speedup  │");
    println!("├─────────────────────────┼──────────────────┼──────────────────┼──────────┤");

    // Query latency
    let trad_query = traditional.query_latency.mean.as_secs_f64() * 1e6;
    let pers_query = persistent.query_latency.mean.as_secs_f64() * 1e6;
    let query_speedup = trad_query / pers_query;
    println!(
        "│ Query (mean)            │ {:>12.2}µs   │ {:>12.2}µs   │ {:>6.1}x  │",
        trad_query, pers_query, query_speedup
    );

    // Inject latency
    let trad_inject = traditional.inject_latency.mean.as_secs_f64() * 1e6;
    let pers_inject = persistent.inject_latency.mean.as_secs_f64() * 1e6;
    let inject_speedup = trad_inject / pers_inject;
    println!(
        "│ Inject (mean)           │ {:>12.2}µs   │ {:>12.2}µs   │ {:>6.1}x  │",
        trad_inject, pers_inject, inject_speedup
    );

    // Single step latency
    let trad_step = traditional.single_step_latency.mean.as_secs_f64() * 1e6;
    let pers_step = persistent.single_step_latency.mean.as_secs_f64() * 1e6;
    let step_speedup = trad_step / pers_step;
    println!(
        "│ Single step (mean)      │ {:>12.2}µs   │ {:>12.2}µs   │ {:>6.1}x  │",
        trad_step, pers_step, step_speedup
    );

    // Mixed workload
    let trad_mixed = traditional.mixed_workload_time.as_secs_f64() * 1000.0;
    let pers_mixed = persistent.mixed_workload_time.as_secs_f64() * 1000.0;
    let mixed_speedup = trad_mixed / pers_mixed;
    println!(
        "│ Mixed workload (total)  │ {:>12.2}ms   │ {:>12.2}ms   │ {:>6.1}x  │",
        trad_mixed, pers_mixed, mixed_speedup
    );

    println!("└─────────────────────────┴──────────────────┴──────────────────┴──────────┘");

    println!("\n📊 Analysis:\n");

    if query_speedup > 1.5 {
        println!(
            "  ✓ Query operations: Persistent is {:.1}x faster",
            query_speedup
        );
        println!("    → Mapped memory reads avoid cudaMemcpy overhead\n");
    }

    if inject_speedup > 1.5 {
        println!(
            "  ✓ Inject operations: Persistent is {:.1}x faster",
            inject_speedup
        );
        println!("    → Commands written to mapped memory, no kernel launch\n");
    }

    if step_speedup > 1.0 {
        println!(
            "  ✓ Single steps: Persistent is {:.1}x faster",
            step_speedup
        );
    } else {
        println!(
            "  → Single steps: Traditional is {:.1}x faster",
            1.0 / step_speedup
        );
        println!("    (Expected: polling overhead in persistent approach)\n");
    }

    println!("\n🎯 Key Insight:\n");
    println!("  Persistent GPU actors excel at INTERACTIVE workloads where:");
    println!("  • Many small operations (not bulk batches)");
    println!("  • Low latency matters (real-time applications)");
    println!("  • State queries are frequent\n");

    println!("  Traditional kernels excel at BATCH workloads where:");
    println!("  • Few large operations (run 10000 steps at once)");
    println!("  • Throughput matters more than latency");
    println!("  • No mid-computation queries needed\n");

    // Frame budget analysis
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  REAL-TIME FRAME BUDGET ANALYSIS (60 FPS = 16.67ms per frame)");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let frame_budget_ms = 16.67;

    let trad_ops_per_frame = (frame_budget_ms / trad_mixed) * traditional.mixed_workload_ops as f64;
    let pers_ops_per_frame = (frame_budget_ms / pers_mixed) * persistent.mixed_workload_ops as f64;

    println!("  Traditional approach:");
    println!(
        "    Max interactive ops per 16ms frame: {:.0}",
        trad_ops_per_frame
    );

    println!("\n  Persistent approach:");
    println!(
        "    Max interactive ops per 16ms frame: {:.0}",
        pers_ops_per_frame
    );

    if pers_ops_per_frame > trad_ops_per_frame {
        println!(
            "\n  → Persistent allows {:.1}x more interactivity per frame!",
            pers_ops_per_frame / trad_ops_per_frame
        );
    }
}
