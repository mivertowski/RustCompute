//! Dedicated Cluster Launch Control test for Blackwell (sm_100+).
//!
//! Phase 0 (here): harness only. `KERNEL_PTX` is empty so the runtime
//! test exits early. The `#[test]` that exercises capability routing
//! runs unconditionally and serves as a Phase 0 sanity check.
//!
//! Phase 1 (on B200): fill in `KERNEL_PTX` with sm_100 PTX produced by
//! nvcc from a kernel that uses `cuLaunchKernelEx`-style dynamic
//! cluster dims, then drop the `#[ignore]` after a green run.

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::ptr;

use cudarc::driver::sys as cuda_sys;

use ringkernel_cuda::driver_api::DirectPtxModule;
use ringkernel_cuda::hopper::cluster::{
    self, ClusterLaunchConfig, ClusterSchedulingPolicy,
};
use ringkernel_cuda::launch_config::GpuArchitecture;

/// Returns true iff a Blackwell-class device (CC >= 10.0) is present.
fn is_blackwell() -> bool {
    match ringkernel_cuda::CudaDevice::new(0) {
        Ok(d) => {
            let (major, _) = d.compute_capability();
            major >= 10
        }
        Err(_) => false,
    }
}

/// Placeholder PTX. Filled in on the VM from a real nvcc -arch=sm_100
/// compilation. Empty string is intentional: keeps the runtime test
/// from launching nonsense and lets the harness ship in Phase 0.
const KERNEL_PTX: &str = "";

const KERNEL_NAME: &str = "blackwell_cluster_launch_probe";

#[test]
fn blackwell_arch_routes_cluster_launch_control() {
    // Capability-routing sanity. Runs on every machine, no GPU needed.
    let bw = GpuArchitecture::blackwell();
    assert!(
        bw.supports_cluster_launch_control(),
        "Blackwell preset must report supports_cluster_launch_control = true"
    );
    let hopper = GpuArchitecture::hopper();
    assert!(
        !hopper.supports_cluster_launch_control(),
        "Hopper preset must NOT report supports_cluster_launch_control = true"
    );
}

#[test]
#[ignore = "requires B200 (sm_100+) and a filled-in KERNEL_PTX"]
fn blackwell_cluster_launch_control_runtime() {
    if !is_blackwell() {
        eprintln!("Skipping: not a Blackwell GPU");
        return;
    }
    if KERNEL_PTX.is_empty() {
        eprintln!("Skipping: KERNEL_PTX not yet filled in on this machine");
        return;
    }

    let device = ringkernel_cuda::CudaDevice::new(0).expect("device 0");
    let module = DirectPtxModule::load_ptx(&device, KERNEL_PTX).expect("load ptx");
    let func = module.get_function(KERNEL_NAME).expect("get function");

    // Sentinel: kernel writes 0xDEADBEEF to *out if the cluster launch
    // control path executed correctly. The exact kernel body is filled
    // in on the VM.
    let mut out_dev: u64 = 0;
    unsafe {
        let r = cuda_sys::cuMemAlloc_v2(&mut out_dev, 4);
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuMemAlloc");
        let zero: u32 = 0;
        let r = cuda_sys::cuMemcpyHtoD_v2(out_dev, &zero as *const u32 as *const c_void, 4);
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuMemcpyHtoD");
    }

    let config = ClusterLaunchConfig {
        grid_dim: (4, 1, 1),
        block_dim: (256, 1, 1),
        cluster_dim: (4, 1, 1), // 4 blocks per cluster = full cluster
        shared_mem_bytes: 0,
        scheduling_policy: ClusterSchedulingPolicy::Default,
    };

    let mut out_ptr = out_dev;
    let mut params: Vec<*mut c_void> =
        vec![&mut out_ptr as *mut u64 as *mut c_void];
    let stream: cuda_sys::CUstream = ptr::null_mut();

    unsafe {
        cluster::launch_kernel_with_cluster(func, &config, &mut params, stream)
            .expect("cluster launch");
        let r = cuda_sys::cuCtxSynchronize();
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuCtxSynchronize");
    }

    let mut out_host: u32 = 0;
    unsafe {
        let r = cuda_sys::cuMemcpyDtoH_v2(
            &mut out_host as *mut u32 as *mut c_void,
            out_dev,
            4,
        );
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuMemcpyDtoH");
        let _ = cuda_sys::cuMemFree_v2(out_dev);
    }
    assert_eq!(
        out_host, 0xDEAD_BEEF,
        "kernel did not write sentinel — cluster launch control path failed"
    );
}
