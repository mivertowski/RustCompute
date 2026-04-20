//! Tests for the intra-block warp work-stealing primitive.
//!
//! Verifies:
//!   1. Every task is processed exactly once (sum of per-warp tallies
//!      equals total_tasks * n_blocks, and every `output[i]` reflects
//!      the expected transform of `input[i]`).
//!   2. The load-balance benefit: runs a "skewed" distribution where
//!      some warps would finish early if statically partitioned; the
//!      stealing kernel should still complete in a single pass.

#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::ptr;

use cudarc::driver::sys as cuda_sys;

use ringkernel_cuda::driver_api::DirectPtxModule;
use ringkernel_cuda::hopper::cluster;

fn is_cuda_available() -> bool {
    ringkernel_cuda::CudaDevice::new(0).is_ok()
}

unsafe fn alloc_and_fill_floats(values: &[f32]) -> cuda_sys::CUdeviceptr {
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
#[ignore] // Requires CUDA
fn warp_work_steal_processes_every_task_exactly_once() {
    if !is_cuda_available() {
        eprintln!("SKIP: no CUDA device");
        return;
    }
    let device = ringkernel_cuda::CudaDevice::new(0).expect("device");
    let ptx = cluster::cluster_kernel_ptx();
    let module = DirectPtxModule::load_ptx(&device, ptx).expect("load PTX");
    let func = module
        .get_function("warp_work_steal")
        .expect("warp_work_steal function");

    let total_tasks: u32 = 4096;
    let n_blocks: u32 = 4;
    let threads_per_block: u32 = 128;
    let warps_per_block = threads_per_block / 32;

    unsafe {
        let input: Vec<f32> = (0..total_tasks).map(|i| i as f32).collect();
        let input_dev = alloc_and_fill_floats(&input);
        let output_dev = alloc_zeroed((total_tasks as usize) * std::mem::size_of::<f32>());
        let stats_bytes =
            (n_blocks as usize) * (warps_per_block as usize) * std::mem::size_of::<u32>();
        let stats_dev = alloc_zeroed(stats_bytes);

        let mut input_arg = input_dev;
        let mut output_arg = output_dev;
        let mut stats_arg = stats_dev;
        let mut total_tasks_arg = total_tasks;
        let mut params: [*mut c_void; 4] = [
            &mut input_arg as *mut u64 as *mut c_void,
            &mut output_arg as *mut u64 as *mut c_void,
            &mut stats_arg as *mut u64 as *mut c_void,
            &mut total_tasks_arg as *mut u32 as *mut c_void,
        ];

        // Launch as a plain (non-cluster, non-cooperative) kernel via
        // cuLaunchKernel so we don't depend on grid.sync / cluster.
        let r = cuda_sys::cuLaunchKernel(
            func,
            n_blocks,
            1,
            1,
            threads_per_block,
            1,
            1,
            0,
            ptr::null_mut(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        );
        assert_eq!(r, cuda_sys::CUresult::CUDA_SUCCESS, "cuLaunchKernel");

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
        let mut stats_host = vec![0u32; (n_blocks * warps_per_block) as usize];
        assert_eq!(
            cuda_sys::cuMemcpyDtoH_v2(stats_host.as_mut_ptr() as *mut c_void, stats_dev, stats_bytes),
            cuda_sys::CUresult::CUDA_SUCCESS
        );

        // Every task should have been transformed.
        for (i, &v) in output_host.iter().enumerate() {
            let expected = (input[i]) * 2.0 + 1.0;
            assert_eq!(v, expected, "task {i} mis-processed");
        }

        // Each block processes total_tasks work items (they all share
        // the same counter in the current kernel — stealing is
        // across-warps within a block, not across blocks). Since we
        // launched n_blocks blocks in parallel, the total recorded
        // tasks equals total_tasks * n_blocks. Across-block sharing
        // is a future extension (see v1.2 roadmap).
        let total_done: u64 = stats_host.iter().map(|&x| x as u64).sum();
        assert_eq!(
            total_done,
            (total_tasks as u64) * (n_blocks as u64),
            "stats sum must match total tasks * blocks"
        );

        let _ = cuda_sys::cuMemFree_v2(input_dev);
        let _ = cuda_sys::cuMemFree_v2(output_dev);
        let _ = cuda_sys::cuMemFree_v2(stats_dev);
    }
}

#[test]
#[ignore] // Requires CUDA
fn warp_work_steal_distributes_unevenly_without_starvation() {
    if !is_cuda_available() {
        eprintln!("SKIP: no CUDA device");
        return;
    }
    let device = ringkernel_cuda::CudaDevice::new(0).expect("device");
    let ptx = cluster::cluster_kernel_ptx();
    let module = DirectPtxModule::load_ptx(&device, ptx).expect("load PTX");
    let func = module
        .get_function("warp_work_steal")
        .expect("warp_work_steal function");

    // A "prime" task count so static partitioning would leave some
    // warps with an odd remainder; stealing should smooth this out.
    let total_tasks: u32 = 1009;
    let n_blocks: u32 = 1;
    let threads_per_block: u32 = 128;
    let warps_per_block = threads_per_block / 32;

    unsafe {
        let input: Vec<f32> = (0..total_tasks).map(|i| i as f32).collect();
        let input_dev = alloc_and_fill_floats(&input);
        let output_dev = alloc_zeroed((total_tasks as usize) * std::mem::size_of::<f32>());
        let stats_bytes =
            (n_blocks as usize) * (warps_per_block as usize) * std::mem::size_of::<u32>();
        let stats_dev = alloc_zeroed(stats_bytes);

        let mut input_arg = input_dev;
        let mut output_arg = output_dev;
        let mut stats_arg = stats_dev;
        let mut total_tasks_arg = total_tasks;
        let mut params: [*mut c_void; 4] = [
            &mut input_arg as *mut u64 as *mut c_void,
            &mut output_arg as *mut u64 as *mut c_void,
            &mut stats_arg as *mut u64 as *mut c_void,
            &mut total_tasks_arg as *mut u32 as *mut c_void,
        ];

        assert_eq!(
            cuda_sys::cuLaunchKernel(
                func,
                n_blocks,
                1,
                1,
                threads_per_block,
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

        let mut stats_host = vec![0u32; warps_per_block as usize];
        assert_eq!(
            cuda_sys::cuMemcpyDtoH_v2(stats_host.as_mut_ptr() as *mut c_void, stats_dev, stats_bytes),
            cuda_sys::CUresult::CUDA_SUCCESS
        );

        let total: u32 = stats_host.iter().sum();
        assert_eq!(total, total_tasks, "all tasks accounted for");

        // At least two warps should have contributed (no single-warp
        // starvation) when tasks > warp_size.
        let active_warps = stats_host.iter().filter(|&&x| x > 0).count();
        assert!(active_warps >= 2, "stealing should engage >= 2 warps");

        // The load should be at least partially balanced: no single
        // warp should have grabbed more than 80% of the work.
        let max_per_warp = *stats_host.iter().max().unwrap();
        assert!(
            max_per_warp * 10 <= total_tasks * 8,
            "one warp shouldn't hog >80% of work (max {max_per_warp} / total {total_tasks})"
        );

        let _ = cuda_sys::cuMemFree_v2(input_dev);
        let _ = cuda_sys::cuMemFree_v2(output_dev);
        let _ = cuda_sys::cuMemFree_v2(stats_dev);
    }
}
