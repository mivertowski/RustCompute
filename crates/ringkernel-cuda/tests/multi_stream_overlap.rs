//! Multi-Stream Overlapped Execution Benchmark
//!
//! Tests overlapping compute and H2K communication using multiple CUDA streams.
//! This is a key datacenter pattern for hiding PCIe latency.

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use cudarc::driver::sys as cuda_sys;

use ringkernel_cuda::driver_api::DirectPtxModule;

fn is_cuda_available() -> bool {
    ringkernel_cuda::CudaDevice::new(0).is_ok()
}

#[test]
#[ignore] // Requires CUDA GPU
fn bench_single_stream_vs_multi_stream() {
    if !is_cuda_available() {
        println!("Skipping: no CUDA device");
        return;
    }

    let device = ringkernel_cuda::CudaDevice::new(0).expect("device");

    // Use the cooperative kernel PTX for a real workload
    let ptx = ringkernel_cuda::cooperative::cooperative_kernel_ptx();
    if ptx.is_empty() {
        println!("Skipping: no cooperative PTX");
        return;
    }

    let module = DirectPtxModule::load_ptx(&device, ptx).expect("load PTX");
    let func = module.get_function("coop_test_grid_sync").expect("get func");

    let block_size: u32 = 256;
    let num_blocks: u32 = 4;
    let iterations: u32 = 100;
    let commands: u32 = 50;
    let trials: u32 = 20;

    // ─── Single stream baseline ───
    let mut single_stream_times = Vec::with_capacity(trials as usize);

    for _ in 0..trials {
        let mut stream: cuda_sys::CUstream = ptr::null_mut();
        unsafe { cuda_sys::cuStreamCreate(&mut stream, 0); }

        let start = Instant::now();
        for _ in 0..commands {
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

            unsafe {
                cuda_sys::cuLaunchKernel(
                    func, num_blocks, 1, 1, block_size, 1, 1, 0, stream,
                    params.as_mut_ptr(), ptr::null_mut(),
                );
            }
            unsafe { cuda_sys::cuMemFree_v2(counter_dev); }
        }
        unsafe {
            cuda_sys::cuStreamSynchronize(stream);
        }
        single_stream_times.push(start.elapsed().as_nanos() as f64 / 1000.0);
        unsafe { cuda_sys::cuStreamDestroy_v2(stream); }
    }

    // ─── Multi-stream (4 streams, round-robin) ───
    let num_streams = 4u32;
    let mut multi_stream_times = Vec::with_capacity(trials as usize);

    for _ in 0..trials {
        let mut streams: Vec<cuda_sys::CUstream> = Vec::new();
        for _ in 0..num_streams {
            let mut s: cuda_sys::CUstream = ptr::null_mut();
            unsafe { cuda_sys::cuStreamCreate(&mut s, cuda_sys::CUstream_flags_enum::CU_STREAM_NON_BLOCKING as u32); }
            streams.push(s);
        }

        let start = Instant::now();
        for cmd in 0..commands {
            let stream_idx = (cmd % num_streams) as usize;

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

            unsafe {
                cuda_sys::cuLaunchKernel(
                    func, num_blocks, 1, 1, block_size, 1, 1, 0, streams[stream_idx],
                    params.as_mut_ptr(), ptr::null_mut(),
                );
            }
            unsafe { cuda_sys::cuMemFree_v2(counter_dev); }
        }

        // Sync all streams
        for s in &streams {
            unsafe { cuda_sys::cuStreamSynchronize(*s); }
        }
        multi_stream_times.push(start.elapsed().as_nanos() as f64 / 1000.0);

        for s in streams {
            unsafe { cuda_sys::cuStreamDestroy_v2(s); }
        }
    }

    // ─── Statistics ───
    let stats = |times: &[f64]| -> (f64, f64, f64) {
        let n = times.len() as f64;
        let mean = times.iter().sum::<f64>() / n;
        let mut sorted = times.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let ci95 = 1.96 * variance.sqrt() / n.sqrt();
        (mean, median, ci95)
    };

    let (s_mean, s_med, s_ci) = stats(&single_stream_times);
    let (m_mean, m_med, m_ci) = stats(&multi_stream_times);

    println!();
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Single Stream vs Multi-Stream ({} streams) — H100", num_streams);
    println!("  {} commands × {} trials, {} blocks × {} threads × {} iters",
        commands, trials, num_blocks, block_size, iterations);
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Single stream:  mean={:.1} us, median={:.1} us (±{:.1})", s_mean, s_med, s_ci);
    println!("  Multi-stream:   mean={:.1} us, median={:.1} us (±{:.1})", m_mean, m_med, m_ci);
    println!("  Speedup:        {:.2}x", s_mean / m_mean);
    println!("  Per-command:    single={:.2} us, multi={:.2} us",
        s_mean / commands as f64, m_mean / commands as f64);
    println!("══════════════════════════════════════════════════════════════════");
}
