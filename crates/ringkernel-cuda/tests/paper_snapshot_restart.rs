//! Paper Experiment 2 — Snapshot / Restart Wall-Clock
//!
//! Measures the end-to-end host-side wall-clock cost of the snapshot --
//! restart protocol described in §10.5. For each actor-state size in
//! {1 KiB, 4 KiB, 16 KiB, 64 KiB} we issue the protocol N_TRIALS times
//! and print per-trial breakdowns:
//!
//!     PAPER_SNAPSHOT_RESTART size=<bytes> trial=<i> phase=capture ns=<ns>
//!     PAPER_SNAPSHOT_RESTART size=<bytes> trial=<i> phase=copy    ns=<ns>
//!     PAPER_SNAPSHOT_RESTART size=<bytes> trial=<i> phase=ack     ns=<ns>
//!     PAPER_SNAPSHOT_RESTART size=<bytes> trial=<i> phase=total   ns=<ns>
//!
//! Run with:
//!     cargo test -p ringkernel-cuda --features cuda --release \
//!         --test paper_snapshot_restart -- --ignored --nocapture

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use cudarc::driver::sys as cuda_sys;

const N_WARMUP: usize = 50;
const N_TRIALS: usize = 200;
const STATE_SIZES: &[usize] = &[1 << 10, 4 << 10, 16 << 10, 64 << 10]; // 1, 4, 16, 64 KiB

#[test]
#[ignore] // Requires GPU
fn paper_snapshot_restart() {
    let device = match ringkernel_cuda::CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: no CUDA device: {e}");
            return;
        }
    };

    eprintln!("# Snapshot/Restart Wall-Clock — single GPU");
    eprintln!("# Warmup: {N_WARMUP}, Trials: {N_TRIALS}");
    eprintln!("# Phases: capture (HLC + flag write), copy (live -> snap), ack (event sync)");
    eprintln!();

    for &size in STATE_SIZES {
        // Allocate live + snap regions on the device.
        let mut live: u64 = 0;
        let mut snap: u64 = 0;
        let mut control_flag: u64 = 0;
        unsafe {
            let r = cuda_sys::cuMemAlloc_v2(&mut live, size);
            assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
            let r = cuda_sys::cuMemAlloc_v2(&mut snap, size);
            assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
            let r = cuda_sys::cuMemAlloc_v2(&mut control_flag, 4);
            assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
            // Initialize live region with deterministic bytes.
            let r = cuda_sys::cuMemsetD8_v2(live, 0xAB, size);
            assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
        }

        // Create stream + event for ordering.
        let mut stream: cuda_sys::CUstream = ptr::null_mut();
        let mut event: cuda_sys::CUevent = ptr::null_mut();
        unsafe {
            let r = cuda_sys::cuStreamCreate(&mut stream, 0);
            assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
            let r = cuda_sys::cuEventCreate(&mut event, 0);
            assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
        }

        // Warmup
        for _ in 0..N_WARMUP {
            unsafe {
                let zero: u32 = 0;
                cuda_sys::cuMemcpyHtoDAsync_v2(
                    control_flag,
                    &zero as *const _ as *const c_void,
                    4,
                    stream,
                );
                cuda_sys::cuMemcpyDtoDAsync_v2(snap, live, size, stream);
                cuda_sys::cuEventRecord(event, stream);
                cuda_sys::cuEventSynchronize(event);
            }
        }

        for trial in 0..N_TRIALS {
            // Phase 1: capture (write quiesce flag + read HLC; we model both as a 4B mapped write here).
            let t0 = Instant::now();
            unsafe {
                let one: u32 = 1;
                let r = cuda_sys::cuMemcpyHtoDAsync_v2(
                    control_flag,
                    &one as *const _ as *const c_void,
                    4,
                    stream,
                );
                assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
            }
            let t_capture = t0.elapsed().as_nanos();

            // Phase 2: copy (live -> snap on device-side).
            let t1 = Instant::now();
            unsafe {
                let r = cuda_sys::cuMemcpyDtoDAsync_v2(snap, live, size, stream);
                assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
            }
            let t_copy = t1.elapsed().as_nanos();

            // Phase 3: ack (event record + sync).
            let t2 = Instant::now();
            unsafe {
                cuda_sys::cuEventRecord(event, stream);
                cuda_sys::cuEventSynchronize(event);
            }
            let t_ack = t2.elapsed().as_nanos();

            let total = t0.elapsed().as_nanos();

            println!(
                "PAPER_SNAPSHOT_RESTART size={size} trial={trial} phase=capture ns={t_capture}"
            );
            println!("PAPER_SNAPSHOT_RESTART size={size} trial={trial} phase=copy    ns={t_copy}");
            println!("PAPER_SNAPSHOT_RESTART size={size} trial={trial} phase=ack     ns={t_ack}");
            println!("PAPER_SNAPSHOT_RESTART size={size} trial={trial} phase=total   ns={total}");
        }

        unsafe {
            cuda_sys::cuEventDestroy_v2(event);
            cuda_sys::cuStreamDestroy_v2(stream);
            cuda_sys::cuMemFree_v2(live);
            cuda_sys::cuMemFree_v2(snap);
            cuda_sys::cuMemFree_v2(control_flag);
        }
    }
}
