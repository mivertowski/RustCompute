//! Paper Addendum — Cross-GPU K2K Sustained Bandwidth
//!
//! Complements Experiment 6 (which measures one-shot P2P migration
//! latency) by reporting *sustained* cross-GPU K2K throughput over
//! the NVLink P2P path. For each payload size {4 KiB, 64 KiB,
//! 1 MiB, 16 MiB} we issue `N_TRANSFERS` back-to-back
//! `cuMemcpyPeerAsync` operations on a single stream with a
//! trailing `cuStreamSynchronize`, then compute the effective
//! bandwidth. The P2P path is the same one driven by
//! `MultiGpuRuntime::migrate_actor` — this micro-benchmark
//! validates that the peer-access wiring sustains high bandwidth
//! under back-to-back traffic, not just single-shot latency.
//!
//! Output format (parseable by extract.py):
//!
//!     PAPER_MGPU_BW size=<bytes> n=<transfers> total_ns=<ns> bw_gbs=<gb_per_s>
//!
//! Run with:
//!     cargo test -p ringkernel-cuda --features "cuda,multi-gpu" --release \
//!         --test paper_multi_gpu_k2k_bw -- --ignored --nocapture
//!
//! Requires 2 GPUs. Skips cleanly on a single-GPU host.

#![cfg(feature = "cuda")]

use std::ptr;
use std::time::Instant;

use cudarc::driver::sys as cuda_sys;

const N_WARMUP: usize = 32;
const N_TRANSFERS: usize = 256;
const SIZES: &[usize] = &[4 << 10, 64 << 10, 1 << 20, 16 << 20];

#[test]
#[ignore] // Requires 2 GPUs
fn paper_multi_gpu_k2k_bw() {
    unsafe {
        if cuda_sys::cuInit(0) != cuda_sys::CUresult::CUDA_SUCCESS {
            eprintln!("SKIP: cuInit failed");
            return;
        }
        let mut count: i32 = 0;
        if cuda_sys::cuDeviceGetCount(&mut count) != cuda_sys::CUresult::CUDA_SUCCESS || count < 2 {
            eprintln!("SKIP: requires >= 2 GPUs (found {count})");
            return;
        }

        let mut dev0: cuda_sys::CUdevice = 0;
        let mut dev1: cuda_sys::CUdevice = 0;
        if cuda_sys::cuDeviceGet(&mut dev0, 0) != cuda_sys::CUresult::CUDA_SUCCESS {
            return;
        }
        if cuda_sys::cuDeviceGet(&mut dev1, 1) != cuda_sys::CUresult::CUDA_SUCCESS {
            return;
        }

        let mut ctx0: cuda_sys::CUcontext = ptr::null_mut();
        let mut ctx1: cuda_sys::CUcontext = ptr::null_mut();
        if cuda_sys::cuDevicePrimaryCtxRetain(&mut ctx0, dev0) != cuda_sys::CUresult::CUDA_SUCCESS {
            return;
        }
        if cuda_sys::cuDevicePrimaryCtxRetain(&mut ctx1, dev1) != cuda_sys::CUresult::CUDA_SUCCESS {
            cuda_sys::cuDevicePrimaryCtxRelease_v2(dev0);
            return;
        }

        let mut p2p_supported: i32 = 0;
        cuda_sys::cuDeviceCanAccessPeer(&mut p2p_supported, 0, 1);
        if p2p_supported == 0 {
            eprintln!("SKIP: P2P not supported between GPU 0 and GPU 1");
            cuda_sys::cuDevicePrimaryCtxRelease_v2(dev0);
            cuda_sys::cuDevicePrimaryCtxRelease_v2(dev1);
            return;
        }

        cuda_sys::cuCtxSetCurrent(ctx0);
        let _ = cuda_sys::cuCtxEnablePeerAccess(ctx1, 0);
        cuda_sys::cuCtxSetCurrent(ctx1);
        let _ = cuda_sys::cuCtxEnablePeerAccess(ctx0, 0);

        eprintln!("# Cross-GPU K2K Sustained Bandwidth — 2 × H100 NVL");
        eprintln!("# N_WARMUP={N_WARMUP}, N_TRANSFERS={N_TRANSFERS}");
        eprintln!();

        for &size in SIZES {
            cuda_sys::cuCtxSetCurrent(ctx0);
            let mut src: cuda_sys::CUdeviceptr = 0;
            if cuda_sys::cuMemAlloc_v2(&mut src, size) != cuda_sys::CUresult::CUDA_SUCCESS {
                eprintln!("# SKIP size={size}: cuMemAlloc(src)");
                continue;
            }
            cuda_sys::cuMemsetD8_v2(src, 0xAB, size);
            let mut stream: cuda_sys::CUstream = ptr::null_mut();
            cuda_sys::cuStreamCreate(&mut stream, 0);

            cuda_sys::cuCtxSetCurrent(ctx1);
            let mut dst: cuda_sys::CUdeviceptr = 0;
            if cuda_sys::cuMemAlloc_v2(&mut dst, size) != cuda_sys::CUresult::CUDA_SUCCESS {
                cuda_sys::cuCtxSetCurrent(ctx0);
                cuda_sys::cuMemFree_v2(src);
                cuda_sys::cuStreamDestroy_v2(stream);
                eprintln!("# SKIP size={size}: cuMemAlloc(dst)");
                continue;
            }

            cuda_sys::cuCtxSetCurrent(ctx0);
            // Warmup
            for _ in 0..N_WARMUP {
                cuda_sys::cuMemcpyPeerAsync(dst, ctx1, src, ctx0, size, stream);
            }
            cuda_sys::cuStreamSynchronize(stream);

            // Measured burst: N_TRANSFERS back-to-back copies.
            let t0 = Instant::now();
            for _ in 0..N_TRANSFERS {
                cuda_sys::cuMemcpyPeerAsync(dst, ctx1, src, ctx0, size, stream);
            }
            cuda_sys::cuStreamSynchronize(stream);
            let total_ns = t0.elapsed().as_nanos() as u64;

            let total_bytes = (size as u64) * (N_TRANSFERS as u64);
            let bw_gbs = (total_bytes as f64) / (total_ns as f64);
            println!(
                "PAPER_MGPU_BW size={size} n={N_TRANSFERS} total_ns={total_ns} bw_gbs={bw_gbs:.3}"
            );

            cuda_sys::cuCtxSetCurrent(ctx0);
            cuda_sys::cuMemFree_v2(src);
            cuda_sys::cuStreamDestroy_v2(stream);
            cuda_sys::cuCtxSetCurrent(ctx1);
            cuda_sys::cuMemFree_v2(dst);
        }

        cuda_sys::cuDevicePrimaryCtxRelease_v2(dev0);
        cuda_sys::cuDevicePrimaryCtxRelease_v2(dev1);
    }
}
