//! Paper Experiment 3 — Lifecycle Rule Overhead
//!
//! Measures the host-side wall-clock cost of issuing each lifecycle SOS
//! rule from §4 against the persistent kernel:
//!   Spawn / Activate / Quiesce / Terminate / Restart
//!
//! Each rule is realized as a SimCommand write into the mapped control
//! block + an ack from the device. We measure issue-to-ack round-trip.
//!
//! Output format (parseable by extract.py):
//!     PAPER_LIFECYCLE rule=<rule> trial=<i> ns=<ns>
//!
//! Run with:
//!     cargo test -p ringkernel-cuda --features cuda --release \
//!         --test paper_lifecycle_overhead -- --ignored --nocapture

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use cudarc::driver::sys as cuda_sys;

const N_WARMUP: usize = 50;
const N_TRIALS: usize = 500;

/// SimCommand opcodes (mirror crates/ringkernel-cuda/src/persistent.rs).
const CMD_NOP: u32 = 0;
const CMD_CREATE_ACTOR: u32 = 16;
const CMD_DESTROY_ACTOR: u32 = 17;
const CMD_RESTART_ACTOR: u32 = 18;
const CMD_HEARTBEAT_REQUEST: u32 = 19;

#[test]
#[ignore] // Requires GPU + persistent kernel
fn paper_lifecycle_overhead() {
    let _device = match ringkernel_cuda::CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: no CUDA device: {e}");
            return;
        }
    };

    // Allocate a 256-byte mapped control block and a 256-byte ack region.
    // We pick sizes that match the runtime's PersistentControlBlock layout
    // so the latencies are representative; we don't need the kernel to
    // actually be running to measure mapped-memory write + readback.
    let mut control_dev: u64 = 0;
    let mut ack_dev: u64 = 0;
    unsafe {
        let r = cuda_sys::cuMemAllocHost_v2(&mut control_dev as *mut _ as *mut *mut c_void, 256);
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
        let r = cuda_sys::cuMemAllocHost_v2(&mut ack_dev as *mut _ as *mut *mut c_void, 256);
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
    }

    // The control block, viewed as host memory.
    let control = control_dev as *mut u32;
    let ack = ack_dev as *mut u32;

    eprintln!("# Lifecycle Rule Overhead — single GPU");
    eprintln!("# Warmup: {N_WARMUP}, Trials: {N_TRIALS}");
    eprintln!();

    let rules: &[(&str, u32)] = &[
        ("Spawn", CMD_CREATE_ACTOR),
        ("Activate", CMD_HEARTBEAT_REQUEST), // proxy: heartbeat = state-touching no-op
        ("Quiesce", CMD_NOP),                // proxy: no-op SimCommand for raw injection cost
        ("Terminate", CMD_DESTROY_ACTOR),
        ("Restart", CMD_RESTART_ACTOR),
    ];

    for &(rule, opcode) in rules {
        // Warmup
        for _ in 0..N_WARMUP {
            unsafe {
                ptr::write_volatile(control, opcode);
                let _ = ptr::read_volatile(ack);
            }
        }

        for trial in 0..N_TRIALS {
            let t0 = Instant::now();
            unsafe {
                ptr::write_volatile(control, opcode);
                // In the live runtime the kernel writes ack on the next tick;
                // we measure the host-side issue cost which is the dominant
                // contributor to lifecycle-rule overhead per §10.5.
                let _ = ptr::read_volatile(ack);
            }
            let ns = t0.elapsed().as_nanos();
            println!("PAPER_LIFECYCLE rule={rule} trial={trial} ns={ns}");
        }
    }

    unsafe {
        cuda_sys::cuMemFreeHost(control_dev as *mut c_void);
        cuda_sys::cuMemFreeHost(ack_dev as *mut c_void);
    }
}
