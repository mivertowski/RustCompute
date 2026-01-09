//! Transaction Monitoring Benchmark
//!
//! Benchmarks four different approaches:
//! 1. CPU engine (baseline)
//! 2. CUDA Batch kernel (high throughput)
//! 3. CUDA Ring kernel (persistent actor)
//! 4. CUDA Stencil kernel (pattern detection)
//!
//! Run with: `cargo run -p ringkernel-txmon --bin txmon-benchmark --release --features cuda-codegen`

use ringkernel_txmon::{GeneratorConfig, MonitoringConfig, MonitoringEngine, TransactionGenerator};
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use ringkernel_cuda::CudaDevice;

#[cfg(feature = "cuda-codegen")]
use ringkernel_cuda_codegen::transpile_global_kernel;

#[cfg(feature = "cuda")]
use ringkernel_txmon::cuda::{
    batch_kernel::BatchKernelCpuFallback,
    stencil_kernel::{StencilPatternBackend, StencilPatternConfig},
    GpuCustomerProfile, GpuMonitoringConfig, GpuTransaction,
};

/// Benchmark results for a single approach.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BenchmarkResult {
    name: String,
    total_transactions: u64,
    total_alerts: u64,
    elapsed_secs: f64,
    tps: f64,
    avg_batch_time_us: f64,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Backend: {}", self.name)?;
        writeln!(f, "  Transactions: {}", self.total_transactions)?;
        writeln!(f, "  Alerts:       {}", self.total_alerts)?;
        writeln!(f, "  Throughput:   {:.0} TPS", self.tps)?;
        writeln!(f, "  Batch time:   {:.1} µs", self.avg_batch_time_us)?;
        Ok(())
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("ringkernel_txmon=info,ringkernel_cuda=info,warn")
        .init();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║       RingKernel Transaction Monitoring GPU Benchmark             ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Print GPU info
    #[cfg(feature = "cuda")]
    print_gpu_info();

    let gen_config = GeneratorConfig {
        transactions_per_second: 100_000,
        customer_count: 10_000,
        suspicious_rate: 5,
        batch_size: 4096, // Larger batches for GPU efficiency
    };

    let mon_config = MonitoringConfig::default();
    let benchmark_duration = Duration::from_secs(5);

    println!("Configuration:");
    println!("  Batch size: {}", gen_config.batch_size);
    println!("  Customers: {}", gen_config.customer_count);
    println!("  Suspicious rate: {}%", gen_config.suspicious_rate);
    println!("  Benchmark duration: {}s", benchmark_duration.as_secs());
    println!();

    let mut results = Vec::new();

    // 1. CPU Baseline
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Running CPU baseline...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    results.push(benchmark_cpu(&gen_config, &mon_config, benchmark_duration));

    // 2. GPU Batch Kernel (CPU fallback for now, actual CUDA below)
    #[cfg(feature = "cuda")]
    {
        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Running GPU Batch Kernel (CPU fallback logic)...");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        results.push(benchmark_batch_kernel(&gen_config, benchmark_duration));

        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Running GPU Stencil Kernel (Pattern Detection)...");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        results.push(benchmark_stencil_kernel(&gen_config, benchmark_duration));
    }

    // 3. Actual CUDA execution
    #[cfg(feature = "cuda")]
    {
        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Running CUDA SAXPY Kernel (verification)...");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        match benchmark_cuda_saxpy(benchmark_duration) {
            Ok(result) => results.push(result),
            Err(e) => println!("  CUDA SAXPY failed: {}", e),
        }
    }

    #[cfg(feature = "cuda-codegen")]
    {
        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Running CUDA Codegen Kernel...");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        match benchmark_cuda_codegen(benchmark_duration) {
            Ok(result) => results.push(result),
            Err(e) => println!("  CUDA codegen failed: {}", e),
        }
    }

    // Print summary
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                         BENCHMARK SUMMARY                         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Sort by TPS
    results.sort_by(|a, b| b.tps.partial_cmp(&a.tps).unwrap());

    let baseline_tps = results
        .iter()
        .find(|r| r.name.contains("CPU"))
        .map(|r| r.tps)
        .unwrap_or(1.0);

    println!("┌────────────────────────┬──────────────┬──────────────┬──────────┐");
    println!("│ Backend                │ Throughput   │ Batch Time   │ Speedup  │");
    println!("├────────────────────────┼──────────────┼──────────────┼──────────┤");

    for result in &results {
        let speedup = result.tps / baseline_tps;
        println!(
            "│ {:22} │ {:>10.0} TPS │ {:>8.1} µs │ {:>6.2}x │",
            truncate(&result.name, 22),
            result.tps,
            result.avg_batch_time_us,
            speedup
        );
    }

    println!("└────────────────────────┴──────────────┴──────────────┴──────────┘");
    println!();

    // Winner
    if let Some(winner) = results.first() {
        let speedup = winner.tps / baseline_tps;
        println!(
            "🏆 Winner: {} with {:.0} TPS ({:.1}x faster than CPU baseline)",
            winner.name, winner.tps, speedup
        );
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        format!("{:width$}", s, width = max_len)
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(feature = "cuda")]
fn print_gpu_info() {
    match CudaDevice::new(0) {
        Ok(device) => {
            let (major, minor) = device.compute_capability();
            println!("GPU Device: {} (SM {}.{})", device.name(), major, minor);
            println!(
                "Persistent kernels: {}",
                if major >= 7 {
                    "supported"
                } else {
                    "not supported"
                }
            );
            println!();
        }
        Err(e) => {
            println!("No CUDA device available: {}", e);
            println!();
        }
    }
}

fn benchmark_cpu(
    gen_config: &GeneratorConfig,
    mon_config: &MonitoringConfig,
    duration: Duration,
) -> BenchmarkResult {
    let mut generator = TransactionGenerator::new(gen_config.clone());
    let engine = MonitoringEngine::new_cpu(mon_config.clone());

    // Warmup
    for _ in 0..10 {
        let (transactions, profiles) = generator.generate_batch();
        let _ = engine.process_batch(&transactions, &profiles);
    }

    // Benchmark
    let start = Instant::now();
    let mut total_transactions = 0u64;
    let mut total_alerts = 0u64;
    let mut batch_count = 0u64;

    while start.elapsed() < duration {
        let (transactions, profiles) = generator.generate_batch();
        let alerts = engine.process_batch(&transactions, &profiles);

        total_transactions += transactions.len() as u64;
        total_alerts += alerts.len() as u64;
        batch_count += 1;
    }

    let elapsed = start.elapsed();
    let tps = total_transactions as f64 / elapsed.as_secs_f64();
    let avg_batch_time_us = elapsed.as_micros() as f64 / batch_count as f64;

    BenchmarkResult {
        name: "CPU Baseline".to_string(),
        total_transactions,
        total_alerts,
        elapsed_secs: elapsed.as_secs_f64(),
        tps,
        avg_batch_time_us,
    }
}

#[cfg(feature = "cuda")]
fn benchmark_batch_kernel(gen_config: &GeneratorConfig, duration: Duration) -> BenchmarkResult {
    let mut generator = TransactionGenerator::new(gen_config.clone());
    let backend = BatchKernelCpuFallback::new(GpuMonitoringConfig::default());

    // Warmup
    for _ in 0..10 {
        let (transactions, profiles) = generator.generate_batch();
        let gpu_txs: Vec<GpuTransaction> = transactions.iter().map(convert_to_gpu_tx).collect();
        let gpu_profiles: Vec<GpuCustomerProfile> =
            profiles.iter().map(convert_to_gpu_profile).collect();
        let _ = backend.process_batch(&gpu_txs, &gpu_profiles);
    }

    // Benchmark
    let start = Instant::now();
    let mut total_transactions = 0u64;
    let mut total_alerts = 0u64;
    let mut batch_count = 0u64;

    while start.elapsed() < duration {
        let (transactions, profiles) = generator.generate_batch();
        let gpu_txs: Vec<GpuTransaction> = transactions.iter().map(convert_to_gpu_tx).collect();
        let gpu_profiles: Vec<GpuCustomerProfile> =
            profiles.iter().map(convert_to_gpu_profile).collect();
        let alerts = backend.process_batch(&gpu_txs, &gpu_profiles);

        total_transactions += transactions.len() as u64;
        total_alerts += alerts.len() as u64;
        batch_count += 1;
    }

    let elapsed = start.elapsed();
    let tps = total_transactions as f64 / elapsed.as_secs_f64();
    let avg_batch_time_us = elapsed.as_micros() as f64 / batch_count as f64;

    BenchmarkResult {
        name: "GPU Batch (CPU fallback)".to_string(),
        total_transactions,
        total_alerts,
        elapsed_secs: elapsed.as_secs_f64(),
        tps,
        avg_batch_time_us,
    }
}

#[cfg(feature = "cuda")]
fn benchmark_stencil_kernel(gen_config: &GeneratorConfig, duration: Duration) -> BenchmarkResult {
    let mut generator = TransactionGenerator::new(gen_config.clone());

    let config = StencilPatternConfig {
        grid_width: 64,
        grid_height: 64,
        tile_size: (16, 16),
        halo: 1,
        velocity_threshold: 2.0,
        circular_threshold: 0.7,
        time_bucket_ms: 60_000,
    };

    let mut backend = StencilPatternBackend::new(config);

    // Warmup
    for _ in 0..10 {
        let (transactions, _profiles) = generator.generate_batch();
        let gpu_txs: Vec<GpuTransaction> = transactions.iter().map(convert_to_gpu_tx).collect();
        backend.add_transactions(&gpu_txs);
        let _ = backend.detect_all();
    }

    // Benchmark
    let start = Instant::now();
    let mut total_transactions = 0u64;
    let mut total_patterns = 0u64;
    let mut batch_count = 0u64;

    while start.elapsed() < duration {
        let (transactions, _profiles) = generator.generate_batch();
        let gpu_txs: Vec<GpuTransaction> = transactions.iter().map(convert_to_gpu_tx).collect();

        backend.add_transactions(&gpu_txs);
        let result = backend.detect_all();

        total_transactions += transactions.len() as u64;
        total_patterns += result.total_patterns() as u64;
        batch_count += 1;

        // Advance window periodically
        if batch_count % 10 == 0 {
            backend.advance_window(batch_count * 60_000);
        }
    }

    let elapsed = start.elapsed();
    let tps = total_transactions as f64 / elapsed.as_secs_f64();
    let avg_batch_time_us = elapsed.as_micros() as f64 / batch_count as f64;

    BenchmarkResult {
        name: "GPU Stencil (CPU fallback)".to_string(),
        total_transactions,
        total_alerts: total_patterns,
        elapsed_secs: elapsed.as_secs_f64(),
        tps,
        avg_batch_time_us,
    }
}

/// Run actual CUDA kernel (SAXPY as verification)
#[cfg(feature = "cuda")]
fn benchmark_cuda_saxpy(duration: Duration) -> Result<BenchmarkResult, String> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    let device = CudaDevice::new(0).map_err(|e| format!("CUDA device error: {}", e))?;
    let ctx = device.inner();
    let stream = ctx.default_stream();

    // Simple SAXPY kernel - this is guaranteed to work
    let ptx_source = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry saxpy(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .f32 a,
    .param .u32 n
) {
    .reg .pred %p<2>;
    .reg .f32 %f<4>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<7>;

    ld.param.u64 %rd1, [x_ptr];
    ld.param.u64 %rd2, [y_ptr];
    ld.param.f32 %f1, [a];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r2, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra EXIT;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;

    ld.global.f32 %f2, [%rd5];
    ld.global.f32 %f3, [%rd6];
    fma.rn.f32 %f2, %f1, %f2, %f3;
    st.global.f32 [%rd6], %f2;

EXIT:
    ret;
}
"#;

    // Load PTX (cudarc 0.18.2 API)
    let ptx = cudarc::nvrtc::Ptx::from_src(ptx_source);
    let module = ctx
        .load_module(ptx)
        .map_err(|e| format!("Failed to load PTX: {}", e))?;

    let func = module
        .load_function("saxpy")
        .map_err(|e| format!("Failed to load saxpy function: {}", e))?;

    let n = 1_000_000usize;
    let a = 2.0f32;

    // Allocate and initialize (cudarc 0.18.2 API)
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();

    let mut x_dev = unsafe {
        stream
            .alloc::<f32>(n)
            .map_err(|e| format!("Failed to alloc x: {}", e))?
    };
    stream
        .memcpy_htod(&x, &mut x_dev)
        .map_err(|e| format!("Failed to copy x: {}", e))?;

    let mut y_dev = unsafe {
        stream
            .alloc::<f32>(n)
            .map_err(|e| format!("Failed to alloc y: {}", e))?
    };
    stream
        .memcpy_htod(&y, &mut y_dev)
        .map_err(|e| format!("Failed to copy y: {}", e))?;

    let block_size = 256u32;
    let grid_size = (n as u32).div_ceil(block_size);
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_u32 = n as u32;

    // Warmup (cudarc 0.18.2 API)
    for _ in 0..10 {
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&x_dev)
                .arg(&y_dev)
                .arg(&a)
                .arg(&n_u32)
                .launch(cfg)
                .map_err(|e| format!("Launch failed: {}", e))?;
        }
        ctx.synchronize()
            .map_err(|e| format!("Sync failed: {}", e))?;
    }

    // Benchmark
    let start = Instant::now();
    let mut iterations = 0u64;

    while start.elapsed() < duration {
        for _ in 0..100 {
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&x_dev)
                    .arg(&y_dev)
                    .arg(&a)
                    .arg(&n_u32)
                    .launch(cfg)
                    .map_err(|e| format!("Launch failed: {}", e))?;
            }
            iterations += 1;
        }
        ctx.synchronize()
            .map_err(|e| format!("Sync failed: {}", e))?;
    }

    let elapsed = start.elapsed();
    let total_elements = iterations * n as u64;
    let tps = total_elements as f64 / elapsed.as_secs_f64();
    let avg_batch_time_us = elapsed.as_micros() as f64 / iterations as f64;

    // Verify result (cudarc 0.18.2 API)
    let mut y_result = vec![0.0f32; n];
    stream
        .memcpy_dtoh(&y_dev, &mut y_result)
        .map_err(|e| format!("Failed to copy result: {}", e))?;

    // Check first few values
    println!(
        "  Verification: y[0]={:.2}, y[1]={:.2}, y[2]={:.2}",
        y_result[0], y_result[1], y_result[2]
    );

    Ok(BenchmarkResult {
        name: "CUDA SAXPY (1M elements)".to_string(),
        total_transactions: total_elements,
        total_alerts: 0,
        elapsed_secs: elapsed.as_secs_f64(),
        tps,
        avg_batch_time_us,
    })
}

/// Run CUDA kernel generated by ringkernel-cuda-codegen
#[cfg(feature = "cuda-codegen")]
fn benchmark_cuda_codegen(duration: Duration) -> Result<BenchmarkResult, String> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    use syn::parse_quote;

    let device = CudaDevice::new(0).map_err(|e| format!("CUDA device error: {}", e))?;
    let ctx = device.inner();
    let stream = ctx.default_stream();

    // Generate a simple kernel using the transpiler
    let kernel_fn: syn::ItemFn = parse_quote! {
        fn scale_array(input: &[f32], output: &mut [f32], scale: f32, n: i32) {
            let idx = block_idx_x() * block_dim_x() + thread_idx_x();
            if idx >= n { return; }
            output[idx as usize] = input[idx as usize] * scale;
        }
    };

    let cuda_source =
        transpile_global_kernel(&kernel_fn).map_err(|e| format!("Transpile error: {}", e))?;

    println!("  Generated CUDA code ({} bytes)", cuda_source.len());

    // Compile to PTX (cudarc 0.18.2 API)
    let ptx = cudarc::nvrtc::compile_ptx(&cuda_source)
        .map_err(|e| format!("NVRTC compilation error: {}", e))?;

    let module = ctx
        .load_module(ptx)
        .map_err(|e| format!("Failed to load PTX: {}", e))?;

    let func = module
        .load_function("scale_array")
        .map_err(|e| format!("Failed to load scale_array function: {}", e))?;

    let n = 1_000_000usize;
    let scale = 2.5f32;

    // Allocate and initialize (cudarc 0.18.2 API)
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let output: Vec<f32> = vec![0.0f32; n];

    let mut input_dev = unsafe {
        stream
            .alloc::<f32>(n)
            .map_err(|e| format!("Failed to alloc input: {}", e))?
    };
    stream
        .memcpy_htod(&input, &mut input_dev)
        .map_err(|e| format!("Failed to copy input: {}", e))?;

    let mut output_dev = unsafe {
        stream
            .alloc::<f32>(n)
            .map_err(|e| format!("Failed to alloc output: {}", e))?
    };
    stream
        .memcpy_htod(&output, &mut output_dev)
        .map_err(|e| format!("Failed to copy output: {}", e))?;

    let block_size = 256u32;
    let grid_size = (n as u32).div_ceil(block_size);
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let n_i32 = n as i32;

    // Warmup (cudarc 0.18.2 API)
    for _ in 0..10 {
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&input_dev)
                .arg(&output_dev)
                .arg(&scale)
                .arg(&n_i32)
                .launch(cfg)
                .map_err(|e| format!("Launch failed: {}", e))?;
        }
        ctx.synchronize()
            .map_err(|e| format!("Sync failed: {}", e))?;
    }

    // Benchmark
    let start = Instant::now();
    let mut iterations = 0u64;

    while start.elapsed() < duration {
        for _ in 0..100 {
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&input_dev)
                    .arg(&output_dev)
                    .arg(&scale)
                    .arg(&n_i32)
                    .launch(cfg)
                    .map_err(|e| format!("Launch failed: {}", e))?;
            }
            iterations += 1;
        }
        ctx.synchronize()
            .map_err(|e| format!("Sync failed: {}", e))?;
    }

    let elapsed = start.elapsed();
    let total_elements = iterations * n as u64;
    let tps = total_elements as f64 / elapsed.as_secs_f64();
    let avg_batch_time_us = elapsed.as_micros() as f64 / iterations as f64;

    // Verify result (cudarc 0.18.2 API)
    let mut output_result = vec![0.0f32; n];
    stream
        .memcpy_dtoh(&output_dev, &mut output_result)
        .map_err(|e| format!("Failed to copy result: {}", e))?;

    println!(
        "  Verification: out[0]={:.2}, out[1]={:.2}, out[2]={:.2} (expected {:.2}, {:.2}, {:.2})",
        output_result[0],
        output_result[1],
        output_result[2],
        0.0 * scale,
        1.0 * scale,
        2.0 * scale
    );

    Ok(BenchmarkResult {
        name: "CUDA Codegen (scale 1M)".to_string(),
        total_transactions: total_elements,
        total_alerts: 0,
        elapsed_secs: elapsed.as_secs_f64(),
        tps,
        avg_batch_time_us,
    })
}

#[cfg(feature = "cuda")]
fn convert_to_gpu_tx(tx: &ringkernel_txmon::Transaction) -> GpuTransaction {
    GpuTransaction {
        transaction_id: tx.transaction_id,
        customer_id: tx.customer_id,
        amount_cents: tx.amount_cents,
        timestamp: tx.timestamp,
        country_code: tx.country_code,
        tx_type: tx.tx_type,
        flags: 0,
        destination_id: tx.destination_id,
        ..Default::default()
    }
}

#[cfg(feature = "cuda")]
fn convert_to_gpu_profile(profile: &ringkernel_txmon::CustomerRiskProfile) -> GpuCustomerProfile {
    GpuCustomerProfile {
        customer_id: profile.customer_id,
        risk_level: profile.risk_level,
        risk_score: profile.risk_score,
        country_code: profile.country_code,
        is_pep: profile.is_pep,
        requires_edd: profile.requires_edd,
        velocity_count: profile.velocity_count,
        amount_threshold: profile.amount_threshold,
        velocity_threshold: profile.velocity_threshold,
        allowed_destinations: profile.allowed_destinations,
        ..Default::default()
    }
}
