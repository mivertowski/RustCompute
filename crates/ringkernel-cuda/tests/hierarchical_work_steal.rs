//! Tests for the hierarchical work-stealing primitives added in v1.2:
//!   - `cluster_dsmem_work_steal`: blocks within a cluster steal from
//!     a DSMEM counter hosted by block 0.
//!   - `grid_hbm_work_steal`: all blocks in the grid steal from a
//!     single HBM counter.
//!
//! Each test verifies (a) every task is processed exactly once and
//! (b) the per-block `stats` tally sums to the launch-time task count.

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::ptr;

use cudarc::driver::sys as cuda_sys;

use ringkernel_cuda::driver_api::DirectPtxModule;
use ringkernel_cuda::hopper::cluster;

fn is_hopper() -> bool {
    match ringkernel_cuda::CudaDevice::new(0) {
        Ok(d) => d.compute_capability().0 >= 9,
        Err(_) => false,
    }
}

unsafe fn alloc_floats_from_host(values: &[f32]) -> cuda_sys::CUdeviceptr {
    let mut dev: cuda_sys::CUdeviceptr = 0;
    let bytes = values.len() * std::mem::size_of::<f32>();
    assert_eq!(
        cuda_sys::cuMemAlloc_v2(&mut dev, bytes),
        cuda_sys::CUresult::CUDA_SUCCESS
    );
    assert_eq!(
        cuda_sys::cuMemcpyHtoD_v2(dev, values.as_ptr() as *const c_void, bytes),
        cuda_sys::CUresult::CUDA_SUCCESS
    );
    dev
}

unsafe fn alloc_zeroed(bytes: usize) -> cuda_sys::CUdeviceptr {
    let mut dev: cuda_sys::CUdeviceptr = 0;
    assert_eq!(
        cuda_sys::cuMemAlloc_v2(&mut dev, bytes),
        cuda_sys::CUresult::CUDA_SUCCESS
    );
    assert_eq!(
        cuda_sys::cuMemsetD8_v2(dev, 0, bytes),
        cuda_sys::CUresult::CUDA_SUCCESS
    );
    dev
}

#[test]
#[ignore] // Requires H100 (cluster groups)
fn cluster_dsmem_work_steal_conserves_work_and_distributes() {
    if !is_hopper() {
        eprintln!("SKIP: not Hopper or newer");
        return;
    }
    if !cluster::has_cluster_kernel_support() {
        eprintln!("SKIP: cluster kernels not built");
        return;
    }
    let device = ringkernel_cuda::CudaDevice::new(0).expect("device");
    let ptx = cluster::cluster_kernel_ptx();
    let module = DirectPtxModule::load_ptx(&device, ptx).expect("load PTX");
    let func = module
        .get_function("cluster_dsmem_work_steal")
        .expect("cluster_dsmem_work_steal");

    let total_tasks: u32 = 2017; // prime so static partition can't divide evenly
    let cluster_size: u32 = 2;
    let block_size: u32 = 128;

    unsafe {
        let input: Vec<f32> = (0..total_tasks).map(|i| i as f32).collect();
        let input_dev = alloc_floats_from_host(&input);
        let output_dev = alloc_zeroed((total_tasks as usize) * std::mem::size_of::<f32>());
        let stats_dev = alloc_zeroed((cluster_size as usize) * std::mem::size_of::<u32>());

        let mut input_arg = input_dev;
        let mut output_arg = output_dev;
        let mut stats_arg = stats_dev;
        let mut total_tasks_arg = total_tasks;
        let mut params: Vec<*mut c_void> = vec![
            &mut input_arg as *mut u64 as *mut c_void,
            &mut output_arg as *mut u64 as *mut c_void,
            &mut stats_arg as *mut u64 as *mut c_void,
            &mut total_tasks_arg as *mut u32 as *mut c_void,
        ];

        let cfg = cluster::ClusterLaunchConfig {
            grid_dim: (cluster_size, 1, 1),
            block_dim: (block_size, 1, 1),
            cluster_dim: (cluster_size, 1, 1),
            shared_mem_bytes: 0,
            scheduling_policy: cluster::ClusterSchedulingPolicy::Default,
        };
        cluster::launch_kernel_with_cluster(func, &cfg, &mut params, ptr::null_mut())
            .expect("launch");
        assert_eq!(
            cuda_sys::cuCtxSynchronize(),
            cuda_sys::CUresult::CUDA_SUCCESS
        );

        // Every task transformed as expected.
        let mut output_host = vec![0.0f32; total_tasks as usize];
        assert_eq!(
            cuda_sys::cuMemcpyDtoH_v2(
                output_host.as_mut_ptr() as *mut c_void,
                output_dev,
                (total_tasks as usize) * std::mem::size_of::<f32>(),
            ),
            cuda_sys::CUresult::CUDA_SUCCESS
        );
        for (i, &v) in output_host.iter().enumerate() {
            let expected = (input[i]) * 3.0 - 2.0;
            assert_eq!(v, expected, "task {i} mis-processed");
        }

        let mut stats_host = vec![0u32; cluster_size as usize];
        assert_eq!(
            cuda_sys::cuMemcpyDtoH_v2(
                stats_host.as_mut_ptr() as *mut c_void,
                stats_dev,
                (cluster_size as usize) * std::mem::size_of::<u32>(),
            ),
            cuda_sys::CUresult::CUDA_SUCCESS
        );
        let total: u32 = stats_host.iter().sum();
        assert_eq!(total, total_tasks, "stats sum = total tasks");

        // Both blocks should have contributed (no starvation).
        assert!(stats_host.iter().all(|&c| c > 0), "every block contributes");

        let _ = cuda_sys::cuMemFree_v2(input_dev);
        let _ = cuda_sys::cuMemFree_v2(output_dev);
        let _ = cuda_sys::cuMemFree_v2(stats_dev);
    }
}

#[test]
#[ignore] // Requires CUDA (no cluster groups needed for grid-HBM)
fn grid_hbm_work_steal_conserves_work_across_blocks() {
    if ringkernel_cuda::CudaDevice::new(0).is_err() {
        eprintln!("SKIP: no CUDA device");
        return;
    }
    let device = ringkernel_cuda::CudaDevice::new(0).expect("device");
    let ptx = cluster::cluster_kernel_ptx();
    let module = DirectPtxModule::load_ptx(&device, ptx).expect("load PTX");
    let func = module
        .get_function("grid_hbm_work_steal")
        .expect("grid_hbm_work_steal");

    let total_tasks: u32 = 8191; // prime
    let n_blocks: u32 = 8;
    let block_size: u32 = 256;

    unsafe {
        let input: Vec<f32> = (0..total_tasks).map(|i| i as f32 * 0.25).collect();
        let input_dev = alloc_floats_from_host(&input);
        let output_dev = alloc_zeroed((total_tasks as usize) * std::mem::size_of::<f32>());
        let stats_dev = alloc_zeroed((n_blocks as usize) * std::mem::size_of::<u32>());

        // Global counter — initialise to `total_tasks`.
        let mut counter_dev: cuda_sys::CUdeviceptr = 0;
        assert_eq!(
            cuda_sys::cuMemAlloc_v2(&mut counter_dev, std::mem::size_of::<i32>()),
            cuda_sys::CUresult::CUDA_SUCCESS
        );
        let init = total_tasks as i32;
        assert_eq!(
            cuda_sys::cuMemcpyHtoD_v2(counter_dev, &init as *const i32 as *const c_void, 4),
            cuda_sys::CUresult::CUDA_SUCCESS
        );

        let mut input_arg = input_dev;
        let mut output_arg = output_dev;
        let mut counter_arg = counter_dev;
        let mut stats_arg = stats_dev;
        let mut total_tasks_arg = total_tasks;
        let mut params: [*mut c_void; 5] = [
            &mut input_arg as *mut u64 as *mut c_void,
            &mut output_arg as *mut u64 as *mut c_void,
            &mut counter_arg as *mut u64 as *mut c_void,
            &mut stats_arg as *mut u64 as *mut c_void,
            &mut total_tasks_arg as *mut u32 as *mut c_void,
        ];

        assert_eq!(
            cuda_sys::cuLaunchKernel(
                func,
                n_blocks,
                1,
                1,
                block_size,
                1,
                1,
                0,
                ptr::null_mut(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ),
            cuda_sys::CUresult::CUDA_SUCCESS
        );
        assert_eq!(
            cuda_sys::cuCtxSynchronize(),
            cuda_sys::CUresult::CUDA_SUCCESS
        );

        let mut output_host = vec![0.0f32; total_tasks as usize];
        assert_eq!(
            cuda_sys::cuMemcpyDtoH_v2(
                output_host.as_mut_ptr() as *mut c_void,
                output_dev,
                (total_tasks as usize) * std::mem::size_of::<f32>(),
            ),
            cuda_sys::CUresult::CUDA_SUCCESS
        );
        for (i, &v) in output_host.iter().enumerate() {
            let expected = input[i] * 4.0 + 3.0;
            assert_eq!(v, expected, "task {i} mis-processed");
        }

        let mut stats_host = vec![0u32; n_blocks as usize];
        assert_eq!(
            cuda_sys::cuMemcpyDtoH_v2(
                stats_host.as_mut_ptr() as *mut c_void,
                stats_dev,
                (n_blocks as usize) * std::mem::size_of::<u32>(),
            ),
            cuda_sys::CUresult::CUDA_SUCCESS
        );
        let total: u32 = stats_host.iter().sum();
        assert_eq!(total, total_tasks, "stats sum = total tasks");
        assert!(
            stats_host.iter().filter(|&&x| x > 0).count() >= 2,
            "stealing engages at least two blocks"
        );

        let _ = cuda_sys::cuMemFree_v2(input_dev);
        let _ = cuda_sys::cuMemFree_v2(output_dev);
        let _ = cuda_sys::cuMemFree_v2(counter_dev);
        let _ = cuda_sys::cuMemFree_v2(stats_dev);
    }
}
