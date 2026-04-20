//! Paper Experiment 1 — K2K Tier Latency
//!
//! Measures K2K message latency at each tier of the channel hierarchy
//! defined in §6.1 of the paper:
//!   - SMEM   (intra-block)
//!   - DSMEM  (intra-cluster, cross-block)
//!   - HBM    (inter-cluster, global memory)
//!
//! For each tier × payload size {64 B, 256 B, 1 KiB, 4 KiB} the test runs
//! N_WARMUP warmup iterations + N_TRIALS measured iterations and prints
//! per-iteration ns to stdout in the format:
//!
//!     PAPER_TIER_LATENCY tier=<tier> payload=<bytes> trial=<i> ns=<ns>
//!
//! The companion `extract.py` parses these lines into a CSV and the
//! `figures/tier-latency.tex` template draws the figure.
//!
//! Run with:
//!     cargo test -p ringkernel-cuda --features "cuda,cooperative" \
//!         --release --test paper_tier_latency -- --ignored --nocapture

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use cudarc::driver::sys as cuda_sys;

use ringkernel_cuda::driver_api::DirectPtxModule;
use ringkernel_cuda::hopper::cluster;

const N_WARMUP: usize = 200;
const N_TRIALS: usize = 1000;
const PAYLOAD_SIZES: &[usize] = &[64, 256, 1024, 4096];
const TIERS: &[&str] = &["smem", "dsmem", "hbm"];

fn is_hopper() -> bool {
    match ringkernel_cuda::CudaDevice::new(0) {
        Ok(d) => {
            let (major, _) = d.compute_capability();
            major >= 9
        }
        Err(_) => false,
    }
}

#[test]
#[ignore] // Requires H100 + cooperative feature
fn paper_tier_latency() {
    if !is_hopper() {
        eprintln!("SKIP: not Hopper or newer; tier latency requires sm_90+");
        return;
    }
    if !cluster::has_cluster_kernel_support() {
        eprintln!("SKIP: cluster kernels not compiled (rebuild --features cooperative)");
        return;
    }

    let device = ringkernel_cuda::CudaDevice::new(0).expect("create device");
    let ptx = cluster::cluster_kernel_ptx();
    let module = DirectPtxModule::load_ptx(&device, ptx).expect("load PTX");

    eprintln!("# K2K Tier Latency — H100 NVL");
    eprintln!("# Warmup: {N_WARMUP}, Trials: {N_TRIALS}");
    eprintln!("# Format: PAPER_TIER_LATENCY tier=... payload=... trial=... ns=...");
    eprintln!();

    for &payload in PAYLOAD_SIZES {
        for &tier in TIERS {
            let kernel_name = match tier {
                "smem"  => "cluster_test_sync",       // intra-block round-trip via shared mem
                "dsmem" => "cluster_dsmem_k2k",       // cross-block via DSMEM
                "hbm"   => "cluster_persistent_actor",// inter-cluster via global mem
                _ => unreachable!(),
            };

            let func = match module.get_function(kernel_name) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("SKIP tier={tier}: kernel {kernel_name} unavailable: {e}");
                    continue;
                }
            };

            // Allocate a payload-sized buffer (used for K2K message bytes).
            let mut buf_dev: u64 = 0;
            unsafe {
                let r = cuda_sys::cuMemAlloc_v2(&mut buf_dev, payload);
                assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
                let r = cuda_sys::cuMemsetD8_v2(buf_dev, 0, payload);
                assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
            }

            // Launch config — same shape for every tier so latency reflects the
            // tier mechanism, not block geometry.
            let cluster_size: u32 = 4;
            let num_blocks: u32 = cluster_size; // one cluster
            let block_size: u32 = 256;
            let cfg = cluster::ClusterLaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                cluster_dim: (cluster_size, 1, 1),
                shared_mem_bytes: payload as u32,
                scheduling_policy: cluster::ClusterSchedulingPolicy::Default,
            };

            // Warmup
            unsafe {
                for _ in 0..N_WARMUP {
                    let mut buf_arg = buf_dev;
                    let mut params: [*mut c_void; 1] = [&mut buf_arg as *mut _ as *mut c_void];
                    cluster::launch_kernel_with_cluster(
                        func, &cfg, &mut params, ptr::null_mut())
                        .expect("warmup launch");
                }
                cuda_sys::cuCtxSynchronize();
            }

            // Timed trials
            for trial in 0..N_TRIALS {
                unsafe {
                    let mut buf_arg = buf_dev;
                    let mut params: [*mut c_void; 1] = [&mut buf_arg as *mut _ as *mut c_void];
                    let t0 = Instant::now();
                    cluster::launch_kernel_with_cluster(
                        func, &cfg, &mut params, ptr::null_mut())
                        .expect("trial launch");
                    cuda_sys::cuCtxSynchronize();
                    let ns = t0.elapsed().as_nanos();
                    println!(
                        "PAPER_TIER_LATENCY tier={tier} payload={payload} trial={trial} ns={ns}"
                    );
                }
            }

            unsafe { let _ = cuda_sys::cuMemFree_v2(buf_dev); }
        }
    }
}
