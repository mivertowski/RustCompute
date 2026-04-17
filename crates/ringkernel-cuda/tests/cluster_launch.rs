//! Integration tests for Hopper Thread Block Cluster launch.
//!
//! These tests verify that cluster kernels can be loaded and executed
//! on H100 GPUs, and benchmark cluster sync vs grid sync.

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use cudarc::driver::sys as cuda_sys;

use ringkernel_cuda::driver_api::DirectPtxModule;
use ringkernel_cuda::hopper::cluster;

fn is_hopper() -> bool {
    // CudaDevice::new handles driver initialization internally
    match ringkernel_cuda::CudaDevice::new(0) {
        Ok(d) => {
            let (major, _) = d.compute_capability();
            major >= 9
        }
        Err(_) => false,
    }
}

#[test]
fn test_cluster_kernel_ptx_available() {
    assert!(
        cluster::has_cluster_kernel_support(),
        "Cluster kernels should be compiled for sm_90"
    );

    let ptx = cluster::cluster_kernel_ptx();
    assert!(!ptx.is_empty(), "Cluster PTX should not be empty");
    assert!(ptx.contains("cluster_test_sync"));
    assert!(ptx.contains("cluster_dsmem_k2k"));
    assert!(ptx.contains("cluster_persistent_actor"));
}

#[test]
#[ignore] // Requires H100 GPU
fn test_cluster_sync_kernel_execution() {
    if !is_hopper() {
        println!("Skipping: not a Hopper GPU");
        return;
    }

    let device = ringkernel_cuda::CudaDevice::new(0).expect("Failed to create device");
    let ptx = cluster::cluster_kernel_ptx();
    let module = DirectPtxModule::load_ptx(&device, ptx).expect("Failed to load cluster PTX");
    let func = module
        .get_function("cluster_test_sync")
        .expect("Failed to get function");

    // Allocate counter via cuMemAlloc
    let mut counter_dev: u64 = 0;
    unsafe {
        let r = cuda_sys::cuMemAlloc_v2(&mut counter_dev, std::mem::size_of::<u32>());
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuMemAlloc failed");
        let zero: u32 = 0;
        let r = cuda_sys::cuMemcpyHtoD_v2(counter_dev, &zero as *const u32 as *const c_void, 4);
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuMemcpyHtoD failed");
    }

    let iterations: u32 = 10;
    let block_size: u32 = 256;
    let cluster_size: u32 = 4;
    let num_blocks: u32 = cluster_size * 2; // 2 clusters

    let config = cluster::ClusterLaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        cluster_dim: (cluster_size, 1, 1),
        shared_mem_bytes: 0,
        scheduling_policy: cluster::ClusterSchedulingPolicy::Default,
    };

    let mut counter_ptr = counter_dev;
    let mut iters = iterations;
    let mut params: Vec<*mut c_void> = vec![
        &mut counter_ptr as *mut u64 as *mut c_void,
        &mut iters as *mut u32 as *mut c_void,
    ];

    let stream: cuda_sys::CUstream = ptr::null_mut();

    unsafe {
        cluster::launch_kernel_with_cluster(func, &config, &mut params, stream)
            .expect("Cluster kernel launch failed");

        let r = cuda_sys::cuCtxSynchronize();
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "sync failed: {:?}", r);
    }

    // Read counter back
    let mut counter_host: u32 = 0;
    unsafe {
        let r =
            cuda_sys::cuMemcpyDtoH_v2(&mut counter_host as *mut u32 as *mut c_void, counter_dev, 4);
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuMemcpyDtoH failed");
        let _ = cuda_sys::cuMemFree_v2(counter_dev);
    }

    let expected = num_blocks * block_size * iterations;
    println!(
        "Cluster sync test: counter={}, expected={}, blocks={}, cluster_size={}",
        counter_host, expected, num_blocks, cluster_size
    );
    assert_eq!(counter_host, expected);
}

#[test]
#[ignore] // Requires H100 GPU
fn test_cluster_dsmem_k2k_execution() {
    if !is_hopper() {
        println!("Skipping: not a Hopper GPU");
        return;
    }

    let device = ringkernel_cuda::CudaDevice::new(0).expect("Failed to create device");
    let ptx = cluster::cluster_kernel_ptx();
    let module = DirectPtxModule::load_ptx(&device, ptx).expect("Failed to load cluster PTX");
    let func = module
        .get_function("cluster_dsmem_k2k")
        .expect("Failed to get function");

    let block_size: u32 = 256;
    let cluster_size: u32 = 4;
    let num_blocks: u32 = cluster_size;
    let message_size_floats: u32 = 64;
    let rounds: u32 = 100;

    // Allocate results buffer
    let results_bytes = (num_blocks as usize) * std::mem::size_of::<f32>();
    let mut results_dev: u64 = 0;
    unsafe {
        let r = cuda_sys::cuMemAlloc_v2(&mut results_dev, results_bytes);
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuMemAlloc failed");
        let r = cuda_sys::cuMemsetD8_v2(results_dev, 0, results_bytes);
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuMemsetD8 failed");
    }

    let config = cluster::ClusterLaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        cluster_dim: (cluster_size, 1, 1),
        shared_mem_bytes: message_size_floats * 4,
        scheduling_policy: cluster::ClusterSchedulingPolicy::ClusterOnGpc,
    };

    let mut results_ptr = results_dev;
    let mut msg_size = message_size_floats;
    let mut rounds_val = rounds;
    let mut params: Vec<*mut c_void> = vec![
        &mut results_ptr as *mut u64 as *mut c_void,
        &mut msg_size as *mut u32 as *mut c_void,
        &mut rounds_val as *mut u32 as *mut c_void,
    ];

    let stream: cuda_sys::CUstream = ptr::null_mut();

    unsafe {
        cluster::launch_kernel_with_cluster(func, &config, &mut params, stream)
            .expect("DSMEM K2K kernel launch failed");

        let r = cuda_sys::cuCtxSynchronize();
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "sync failed: {:?}", r);
    }

    let mut results_host = vec![0.0f32; num_blocks as usize];
    unsafe {
        let r = cuda_sys::cuMemcpyDtoH_v2(
            results_host.as_mut_ptr() as *mut c_void,
            results_dev,
            results_bytes,
        );
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuMemcpyDtoH failed");
        let _ = cuda_sys::cuMemFree_v2(results_dev);
    }

    println!(
        "DSMEM K2K results ({} rounds, cluster_size={}):",
        rounds, cluster_size
    );
    for (i, val) in results_host.iter().enumerate() {
        println!("  Block {}: sum = {:.2}", i, val);
    }

    // All blocks should have non-zero results from DSMEM exchange
    for (i, val) in results_host.iter().enumerate() {
        assert!(
            *val != 0.0,
            "Block {} result should be non-zero after DSMEM exchange",
            i
        );
    }
}

#[test]
#[ignore] // Requires H100 GPU
fn bench_cluster_sync_vs_grid_sync() {
    if !is_hopper() {
        println!("Skipping: not a Hopper GPU");
        return;
    }

    let device = ringkernel_cuda::CudaDevice::new(0).expect("Failed to create device");
    let block_size: u32 = 256;
    let cluster_size: u32 = 4;
    let iterations: u32 = 1000;
    let trials: u32 = 20;

    // ─── Benchmark 1: Cluster sync ───
    let cluster_ptx = cluster::cluster_kernel_ptx();
    let cluster_module = DirectPtxModule::load_ptx(&device, cluster_ptx).expect("load PTX");
    let cluster_func = cluster_module
        .get_function("cluster_test_sync")
        .expect("get func");

    let mut cluster_times = Vec::with_capacity(trials as usize);

    for _ in 0..trials {
        let mut counter_dev: u64 = 0;
        unsafe {
            cuda_sys::cuMemAlloc_v2(&mut counter_dev, 4);
            cuda_sys::cuMemsetD8_v2(counter_dev, 0, 4);
        }

        let config = cluster::ClusterLaunchConfig {
            grid_dim: (cluster_size, 1, 1),
            block_dim: (block_size, 1, 1),
            cluster_dim: (cluster_size, 1, 1),
            shared_mem_bytes: 0,
            scheduling_policy: cluster::ClusterSchedulingPolicy::ClusterOnGpc,
        };

        let mut cptr = counter_dev;
        let mut iters = iterations;
        let mut params: Vec<*mut c_void> = vec![
            &mut cptr as *mut u64 as *mut c_void,
            &mut iters as *mut u32 as *mut c_void,
        ];
        let stream: cuda_sys::CUstream = ptr::null_mut();

        let start = Instant::now();
        unsafe {
            cluster::launch_kernel_with_cluster(cluster_func, &config, &mut params, stream)
                .expect("launch");
            cuda_sys::cuCtxSynchronize();
        }
        cluster_times.push(start.elapsed().as_nanos() as f64 / 1000.0); // us
        unsafe {
            cuda_sys::cuMemFree_v2(counter_dev);
        }
    }

    // ─── Benchmark 2: Grid sync (cooperative launch, same block count) ───
    let mut grid_times = Vec::with_capacity(trials as usize);

    if ringkernel_cuda::cooperative::has_cooperative_support() {
        let coop_ptx = ringkernel_cuda::cooperative::cooperative_kernel_ptx();
        let coop_module = DirectPtxModule::load_ptx(&device, coop_ptx).expect("load coop PTX");
        let coop_func = coop_module
            .get_function("coop_test_grid_sync")
            .expect("get coop func");

        for _ in 0..trials {
            let mut counter_dev: u64 = 0;
            unsafe {
                cuda_sys::cuMemAlloc_v2(&mut counter_dev, 4);
                cuda_sys::cuMemsetD8_v2(counter_dev, 0, 4);
            }

            let mut cptr = counter_dev;
            let mut iters = iterations;
            let mut params: Vec<*mut c_void> = vec![
                &mut cptr as *mut u64 as *mut c_void,
                &mut iters as *mut u32 as *mut c_void,
            ];
            let stream: cuda_sys::CUstream = ptr::null_mut();

            let start = Instant::now();
            unsafe {
                cudarc::driver::result::launch_cooperative_kernel(
                    coop_func,
                    (cluster_size, 1, 1),
                    (block_size, 1, 1),
                    0,
                    stream,
                    &mut params,
                )
                .expect("coop launch");
                cuda_sys::cuCtxSynchronize();
            }
            grid_times.push(start.elapsed().as_nanos() as f64 / 1000.0);
            unsafe {
                cuda_sys::cuMemFree_v2(counter_dev);
            }
        }
    }

    // ─── Statistics ───
    let stats = |times: &[f64]| -> (f64, f64, f64, f64) {
        let n = times.len() as f64;
        let mean = times.iter().sum::<f64>() / n;
        let mut sorted = times.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        let ci95 = 1.96 * std_dev / n.sqrt();
        (mean, median, std_dev, ci95)
    };

    let (c_mean, c_med, c_sd, c_ci) = stats(&cluster_times);

    println!();
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Cluster Sync vs Grid Sync — H100 Benchmark");
    println!(
        "  {} sync iterations × {} trials, {} blocks × {} threads",
        iterations, trials, cluster_size, block_size
    );
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Cluster sync:");
    println!("    mean   = {:.1} us  (95% CI: ±{:.1})", c_mean, c_ci);
    println!("    median = {:.1} us", c_med);
    println!("    stddev = {:.1} us", c_sd);
    println!("    per-sync = {:.3} us", c_mean / iterations as f64);

    if !grid_times.is_empty() {
        let (g_mean, g_med, g_sd, g_ci) = stats(&grid_times);
        let speedup = g_mean / c_mean;

        println!("  Grid sync (cooperative):");
        println!("    mean   = {:.1} us  (95% CI: ±{:.1})", g_mean, g_ci);
        println!("    median = {:.1} us", g_med);
        println!("    stddev = {:.1} us", g_sd);
        println!("    per-sync = {:.3} us", g_mean / iterations as f64);
        println!("  ────────────────────────────────────────────");
        println!("  Cluster sync speedup: {:.2}x", speedup);
        println!(
            "  Per-sync latency: cluster={:.3} us, grid={:.3} us",
            c_mean / iterations as f64,
            g_mean / iterations as f64
        );
    }
    println!("══════════════════════════════════════════════════════════════════");
}
