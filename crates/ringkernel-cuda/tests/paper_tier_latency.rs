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
// Three K2K tiers measured directly:
//   smem  — cluster_test_sync     (shared memory, intra-block)
//   dsmem — cluster_dsmem_k2k     (distributed shared memory, intra-cluster)
//   hbm   — cluster_hbm_k2k       (global memory, inter-cluster via grid.sync)
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

    // Each tier uses the kernel whose synchronization primitive is that
    // tier. cluster_test_sync exercises only intra-block + cluster sync
    // (SMEM reads/writes stay on-chip); cluster_dsmem_k2k exchanges via
    // distributed shared memory (DSMEM, cluster-wide); we reuse
    // cluster_dsmem_k2k at HBM tier with the message size forcing an
    // L2/HBM round-trip (too large for DSMEM). This is a conservative
    // operational definition of "per-tier latency" — it measures
    // host-to-cluster-sync round-trip at each payload, which the paper
    // uses to show the tier hierarchy's wall-clock impact.
    for &payload in PAYLOAD_SIZES {
        for &tier in TIERS {
            let kernel_name = match tier {
                "smem" => "cluster_test_sync",
                "dsmem" => "cluster_dsmem_k2k",
                "hbm" => "cluster_hbm_k2k",
                _ => unreachable!(),
            };

            let func = match module.get_function(kernel_name) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("SKIP tier={tier}: kernel {kernel_name} unavailable: {e}");
                    continue;
                }
            };

            // Cluster launch geometry. H100 NVL supports cluster size 2;
            // we keep it small so every payload size fits inside SM
            // shared memory when needed (DSMEM / SMEM tiers).
            let cluster_size: u32 = 2;
            let num_blocks: u32 = cluster_size; // one cluster per launch
            let block_size: u32 = 128;

            // Shared memory bytes per block: SMEM / DSMEM use up to
            // `payload`; HBM tier deliberately sets shared_mem_bytes=0
            // and routes the payload through a HBM buffer to force a
            // global-memory round-trip.
            // SMEM/DSMEM use shared memory for the message; HBM uses a
            // global buffer and does not need per-block SMEM.
            let smem_bytes = if tier == "hbm" { 0 } else { payload as u32 };

            let cfg = cluster::ClusterLaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                cluster_dim: (cluster_size, 1, 1),
                shared_mem_bytes: smem_bytes,
                scheduling_policy: cluster::ClusterSchedulingPolicy::Default,
            };

            // Build kernel-specific parameter tuples. Each kernel has
            // its own signature — reuse via a variant match keeps the
            // outer timing loop identical.
            let msg_size_floats: u32 = (payload / 4).max(1) as u32;
            let rounds: u32 = 1;

            // Allocations per tier.
            let mut counter_dev: u64 = 0;
            let mut results_dev: u64 = 0;
            let mut hbm_buf_dev: u64 = 0;
            unsafe {
                match tier {
                    "smem" => {
                        let r = cuda_sys::cuMemAlloc_v2(
                            &mut counter_dev,
                            std::mem::size_of::<u32>(),
                        );
                        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
                        let r = cuda_sys::cuMemsetD8_v2(
                            counter_dev,
                            0,
                            std::mem::size_of::<u32>(),
                        );
                        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
                    }
                    "dsmem" => {
                        let results_bytes =
                            (num_blocks as usize) * std::mem::size_of::<f32>();
                        let r = cuda_sys::cuMemAlloc_v2(&mut results_dev, results_bytes);
                        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
                        let r = cuda_sys::cuMemsetD8_v2(results_dev, 0, results_bytes);
                        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
                    }
                    "hbm" => {
                        // hbm_buf: one slot per block × payload bytes.
                        let hbm_bytes =
                            (num_blocks as usize) * (payload.max(4));
                        let r = cuda_sys::cuMemAlloc_v2(&mut hbm_buf_dev, hbm_bytes);
                        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
                        let r = cuda_sys::cuMemsetD8_v2(hbm_buf_dev, 0, hbm_bytes);
                        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
                        // results: one float per block.
                        let results_bytes =
                            (num_blocks as usize) * std::mem::size_of::<f32>();
                        let r = cuda_sys::cuMemAlloc_v2(&mut results_dev, results_bytes);
                        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
                        let r = cuda_sys::cuMemsetD8_v2(results_dev, 0, results_bytes);
                        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS);
                    }
                    _ => unreachable!(),
                }
            }

            // Helper to build fresh param slice per launch (kernel
            // parameters are passed by reference so they must live for
            // the duration of the call). HBM tier uses
            // `cuLaunchCooperativeKernel` because `cluster_hbm_k2k`
            // relies on `grid.sync()` for inter-cluster visibility;
            // the cluster-config-based launcher is not valid for that.
            let launch_once = |_trial_idx: usize| -> std::result::Result<(), String> {
                let mut counter_arg = counter_dev;
                let mut results_arg = results_dev;
                let mut hbm_arg = hbm_buf_dev;
                let mut iters_arg: u32 = 1;
                let mut msg_arg: u32 = msg_size_floats;
                let mut rounds_arg: u32 = rounds;
                unsafe {
                    let mut params: Vec<*mut c_void> = match tier {
                        "smem" => vec![
                            &mut counter_arg as *mut u64 as *mut c_void,
                            &mut iters_arg as *mut u32 as *mut c_void,
                        ],
                        "dsmem" => vec![
                            &mut results_arg as *mut u64 as *mut c_void,
                            &mut msg_arg as *mut u32 as *mut c_void,
                            &mut rounds_arg as *mut u32 as *mut c_void,
                        ],
                        "hbm" => vec![
                            &mut hbm_arg as *mut u64 as *mut c_void,
                            &mut results_arg as *mut u64 as *mut c_void,
                            &mut msg_arg as *mut u32 as *mut c_void,
                            &mut rounds_arg as *mut u32 as *mut c_void,
                        ],
                        _ => unreachable!(),
                    };
                    if tier == "hbm" {
                        cudarc::driver::result::launch_cooperative_kernel(
                            func,
                            cfg.grid_dim,
                            cfg.block_dim,
                            smem_bytes,
                            ptr::null_mut(),
                            &mut params,
                        )
                        .map_err(|e| format!("cuLaunchCooperativeKernel: {e:?}"))
                    } else {
                        cluster::launch_kernel_with_cluster(
                            func,
                            &cfg,
                            &mut params,
                            ptr::null_mut(),
                        )
                        .map_err(|e| format!("{e}"))
                    }
                }
            };

            // Warmup
            for i in 0..N_WARMUP {
                launch_once(i).expect("warmup launch");
            }
            unsafe {
                cuda_sys::cuCtxSynchronize();
            }

            // Timed trials
            for trial in 0..N_TRIALS {
                let t0 = Instant::now();
                launch_once(trial).expect("trial launch");
                unsafe {
                    cuda_sys::cuCtxSynchronize();
                }
                let ns = t0.elapsed().as_nanos();
                println!(
                    "PAPER_TIER_LATENCY tier={tier} payload={payload} trial={trial} ns={ns}"
                );
            }

            unsafe {
                if counter_dev != 0 {
                    let _ = cuda_sys::cuMemFree_v2(counter_dev);
                }
                if results_dev != 0 {
                    let _ = cuda_sys::cuMemFree_v2(results_dev);
                }
                if hbm_buf_dev != 0 {
                    let _ = cuda_sys::cuMemFree_v2(hbm_buf_dev);
                }
            }
        }
    }
}
