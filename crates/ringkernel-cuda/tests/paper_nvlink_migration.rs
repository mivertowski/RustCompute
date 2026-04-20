//! Paper Experiment 6 — NVLink P2P vs Host-Stage Migration Cost
//!
//! Validates the qualitative claim of §9.3 ("NVLink is roughly an order
//! of magnitude faster than host-staged fallback") with concrete numbers
//! on a 2×H100 system.
//!
//! Two paths are exercised:
//!   - "p2p":   NVLink-enabled cuMemcpyPeerAsync between GPU 0 and GPU 1
//!   - "host":  cuMemcpyDtoHAsync (GPU0 -> pinned host) + cuMemcpyHtoDAsync (host -> GPU1)
//!
//! Per state size {1 KiB, 4 KiB, 64 KiB, 1 MiB, 16 MiB} the test runs
//! N_WARMUP + N_TRIALS iterations of each path and prints:
//!
//!     PAPER_MIGRATION path=<p2p|host> size=<bytes> trial=<i> ns=<ns>
//!
//! Requires 2 GPUs. Skips cleanly with a SKIP message on a single-GPU host.
//! Uses raw cuda-sys (does not depend on CudaDevice's private context).
//!
//! Run with:
//!     cargo test -p ringkernel-cuda --features "cuda,multi-gpu" --release \
//!         --test paper_nvlink_migration -- --ignored --nocapture

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use cudarc::driver::sys as cuda_sys;

const N_WARMUP: usize = 50;
const N_TRIALS: usize = 200;
const SIZES: &[usize] = &[1 << 10, 4 << 10, 64 << 10, 1 << 20, 16 << 20];

/// Initialize CUDA, retain primary contexts on devices 0 and 1, return them.
/// Returns None if fewer than 2 devices are available.
unsafe fn setup_dual_gpu() -> Option<(cuda_sys::CUcontext, cuda_sys::CUcontext)> {
    if cuda_sys::cuInit(0) != cuda_sys::CUresult::CUDA_SUCCESS {
        return None;
    }

    let mut count: i32 = 0;
    if cuda_sys::cuDeviceGetCount(&mut count) != cuda_sys::CUresult::CUDA_SUCCESS {
        return None;
    }
    if count < 2 {
        eprintln!("SKIP: requires >= 2 GPUs (found {count})");
        return None;
    }

    let mut dev0: cuda_sys::CUdevice = 0;
    let mut dev1: cuda_sys::CUdevice = 0;
    if cuda_sys::cuDeviceGet(&mut dev0, 0) != cuda_sys::CUresult::CUDA_SUCCESS {
        return None;
    }
    if cuda_sys::cuDeviceGet(&mut dev1, 1) != cuda_sys::CUresult::CUDA_SUCCESS {
        return None;
    }

    let mut ctx0: cuda_sys::CUcontext = ptr::null_mut();
    let mut ctx1: cuda_sys::CUcontext = ptr::null_mut();
    if cuda_sys::cuDevicePrimaryCtxRetain(&mut ctx0, dev0) != cuda_sys::CUresult::CUDA_SUCCESS {
        return None;
    }
    if cuda_sys::cuDevicePrimaryCtxRetain(&mut ctx1, dev1) != cuda_sys::CUresult::CUDA_SUCCESS {
        cuda_sys::cuDevicePrimaryCtxRelease_v2(dev0);
        return None;
    }

    Some((ctx0, ctx1))
}

#[test]
#[ignore] // Requires 2 GPUs
fn paper_nvlink_migration() {
    unsafe {
        let (ctx0, ctx1) = match setup_dual_gpu() {
            Some(c) => c,
            None => return,
        };

        // Probe peer access GPU0 -> GPU1.
        let mut p2p_supported: i32 = 0;
        let _ = cuda_sys::cuDeviceCanAccessPeer(&mut p2p_supported, 0, 1);

        let p2p_enabled = if p2p_supported != 0 {
            cuda_sys::cuCtxSetCurrent(ctx0);
            let _ = cuda_sys::cuCtxEnablePeerAccess(ctx1, 0);
            cuda_sys::cuCtxSetCurrent(ctx1);
            let _ = cuda_sys::cuCtxEnablePeerAccess(ctx0, 0);
            true
        } else {
            false
        };

        eprintln!("# NVLink P2P vs Host-Stage Migration — 2x GPU");
        eprintln!("# P2P enabled: {p2p_enabled}");
        eprintln!("# Warmup: {N_WARMUP}, Trials: {N_TRIALS}");
        eprintln!();

        for &size in SIZES {
            // src on GPU0
            cuda_sys::cuCtxSetCurrent(ctx0);
            let mut src: u64 = 0;
            cuda_sys::cuMemAlloc_v2(&mut src, size);
            cuda_sys::cuMemsetD8_v2(src, 0xAB, size);
            let mut s0: cuda_sys::CUstream = ptr::null_mut();
            cuda_sys::cuStreamCreate(&mut s0, 0);

            // dst on GPU1
            cuda_sys::cuCtxSetCurrent(ctx1);
            let mut dst: u64 = 0;
            cuda_sys::cuMemAlloc_v2(&mut dst, size);
            let mut s1: cuda_sys::CUstream = ptr::null_mut();
            cuda_sys::cuStreamCreate(&mut s1, 0);

            // Pinned host buffer (allocated in current ctx; usable from both).
            let mut host_buf: *mut c_void = ptr::null_mut();
            cuda_sys::cuMemAllocHost_v2(&mut host_buf, size);

            // ── Path A: NVLink P2P ──────────────────────────────────
            if p2p_enabled {
                cuda_sys::cuCtxSetCurrent(ctx0);
                for _ in 0..N_WARMUP {
                    cuda_sys::cuMemcpyPeerAsync(dst, ctx1, src, ctx0, size, s0);
                    cuda_sys::cuStreamSynchronize(s0);
                }
                for trial in 0..N_TRIALS {
                    let t = Instant::now();
                    cuda_sys::cuMemcpyPeerAsync(dst, ctx1, src, ctx0, size, s0);
                    cuda_sys::cuStreamSynchronize(s0);
                    let ns = t.elapsed().as_nanos();
                    println!("PAPER_MIGRATION path=p2p size={size} trial={trial} ns={ns}");
                }
            } else {
                eprintln!("# size={size}: SKIP p2p (NVLink not available)");
            }

            // ── Path B: Host-staged ────────────────────────────────
            for _ in 0..N_WARMUP {
                cuda_sys::cuCtxSetCurrent(ctx0);
                cuda_sys::cuMemcpyDtoHAsync_v2(host_buf, src, size, s0);
                cuda_sys::cuStreamSynchronize(s0);
                cuda_sys::cuCtxSetCurrent(ctx1);
                cuda_sys::cuMemcpyHtoDAsync_v2(dst, host_buf, size, s1);
                cuda_sys::cuStreamSynchronize(s1);
            }
            for trial in 0..N_TRIALS {
                let t = Instant::now();
                cuda_sys::cuCtxSetCurrent(ctx0);
                cuda_sys::cuMemcpyDtoHAsync_v2(host_buf, src, size, s0);
                cuda_sys::cuStreamSynchronize(s0);
                cuda_sys::cuCtxSetCurrent(ctx1);
                cuda_sys::cuMemcpyHtoDAsync_v2(dst, host_buf, size, s1);
                cuda_sys::cuStreamSynchronize(s1);
                let ns = t.elapsed().as_nanos();
                println!("PAPER_MIGRATION path=host size={size} trial={trial} ns={ns}");
            }

            // Cleanup
            cuda_sys::cuCtxSetCurrent(ctx0);
            cuda_sys::cuStreamDestroy_v2(s0);
            cuda_sys::cuMemFree_v2(src);
            cuda_sys::cuCtxSetCurrent(ctx1);
            cuda_sys::cuStreamDestroy_v2(s1);
            cuda_sys::cuMemFree_v2(dst);
            cuda_sys::cuMemFreeHost(host_buf);
        }

        // Release contexts
        cuda_sys::cuDevicePrimaryCtxRelease_v2(0);
        cuda_sys::cuDevicePrimaryCtxRelease_v2(1);
    }
}
