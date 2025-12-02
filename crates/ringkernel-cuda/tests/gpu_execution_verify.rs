//! GPU Execution Verification Tests
//!
//! These tests verify that CUDA kernels actually execute on the GPU hardware
//! by performing real computations and verifying results.

use cudarc::driver::{CudaDevice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

/// CUDA C kernel that increments each element in an array by 1.
const INCREMENT_KERNEL_CUDA: &str = r#"
extern "C" __global__ void increment_kernel(unsigned int* data, unsigned int count) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        data[tid] = data[tid] + 1;
    }
}
"#;

/// CUDA C kernel that multiplies each element by 2
const MULTIPLY_KERNEL_CUDA: &str = r#"
extern "C" __global__ void multiply_kernel(unsigned int* data, unsigned int count) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        data[tid] = data[tid] * 2;
    }
}
"#;

/// CUDA C kernel for vector addition: C = A + B
const VECTOR_ADD_CUDA: &str = r#"
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, unsigned int count) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        c[tid] = a[tid] + b[tid];
    }
}
"#;

/// CUDA C kernel that computes sum reduction
const REDUCTION_KERNEL_CUDA: &str = r#"
extern "C" __global__ void sum_reduction(const float* input, float* output, unsigned int count) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load to shared memory
    sdata[tid] = (i < count) ? input[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
"#;

#[test]
fn test_gpu_increment_kernel_execution() {
    // Skip if no CUDA device
    let device_count = CudaDevice::count().unwrap_or(0);
    if device_count == 0 {
        println!("No CUDA devices found, skipping test");
        return;
    }

    println!("=== GPU Increment Kernel Execution Test ===");
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");
    println!("Device created successfully");

    // Compile CUDA C to PTX
    let ptx = compile_ptx(INCREMENT_KERNEL_CUDA).expect("Failed to compile CUDA kernel");
    println!("CUDA kernel compiled to PTX");

    device
        .load_ptx(ptx, "increment", &["increment_kernel"])
        .expect("Failed to load PTX");
    println!("PTX loaded successfully");

    let kernel = device
        .get_func("increment", "increment_kernel")
        .expect("Failed to get kernel function");
    println!("Kernel function retrieved");

    // Prepare data on host: [0, 1, 2, 3, ..., 255]
    let n: u32 = 256;
    let host_data: Vec<u32> = (0..n).collect();
    println!("Input data: first 10 = {:?}", &host_data[..10]);

    // Allocate and copy to device
    let device_data = device
        .htod_sync_copy(&host_data)
        .expect("Failed to copy to device");

    let device_ptr = *device_data.device_ptr();
    println!("Device pointer: 0x{:x}", device_ptr);

    // Launch kernel
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (n, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("Launching kernel with {} threads...", n);
    unsafe {
        kernel
            .clone()
            .launch(config, (&device_data, n))
            .expect("Kernel launch failed");
    }

    // Synchronize
    device.synchronize().expect("Synchronize failed");
    println!("Kernel execution completed");

    // Copy back to host
    let result = device
        .dtoh_sync_copy(&device_data)
        .expect("Failed to copy from device");

    println!("Output data: first 10 = {:?}", &result[..10]);

    // Verify: each element should be incremented by 1
    let expected: Vec<u32> = (1..=n).collect();
    assert_eq!(result, expected, "GPU increment kernel did not execute correctly!");

    println!("✓ GPU kernel executed correctly - all {} elements verified!", n);
}

#[test]
fn test_gpu_multiply_kernel_execution() {
    let device_count = CudaDevice::count().unwrap_or(0);
    if device_count == 0 {
        println!("No CUDA devices found, skipping test");
        return;
    }

    println!("\n=== GPU Multiply Kernel Execution Test ===");
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");

    // Compile and load
    let ptx = compile_ptx(MULTIPLY_KERNEL_CUDA).expect("Failed to compile CUDA kernel");
    device
        .load_ptx(ptx, "multiply", &["multiply_kernel"])
        .expect("Failed to load PTX");

    let kernel = device
        .get_func("multiply", "multiply_kernel")
        .expect("Failed to get kernel function");

    // Prepare data: [1, 2, 3, 4, ..., 256]
    let n: u32 = 256;
    let host_data: Vec<u32> = (1..=n).collect();
    println!("Input data: first 10 = {:?}", &host_data[..10]);

    let device_data = device
        .htod_sync_copy(&host_data)
        .expect("Failed to copy to device");

    let _device_ptr = *device_data.device_ptr();

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (n, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("Launching multiply kernel...");
    unsafe {
        kernel
            .clone()
            .launch(config, (&device_data, n))
            .expect("Kernel launch failed");
    }
    device.synchronize().expect("Synchronize failed");

    let result = device
        .dtoh_sync_copy(&device_data)
        .expect("Failed to copy from device");

    println!("Output data: first 10 = {:?}", &result[..10]);

    // Verify: each element should be multiplied by 2
    let expected: Vec<u32> = (1..=n).map(|x| x * 2).collect();
    assert_eq!(result, expected, "GPU multiply kernel did not execute correctly!");

    println!("✓ GPU multiply kernel executed correctly!");
}

#[test]
fn test_gpu_vector_addition() {
    let device_count = CudaDevice::count().unwrap_or(0);
    if device_count == 0 {
        println!("No CUDA devices found, skipping test");
        return;
    }

    println!("\n=== GPU Vector Addition Test ===");
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");

    // Compile and load
    let ptx = compile_ptx(VECTOR_ADD_CUDA).expect("Failed to compile CUDA kernel");
    device
        .load_ptx(ptx, "vecadd", &["vector_add"])
        .expect("Failed to load PTX");

    let kernel = device
        .get_func("vecadd", "vector_add")
        .expect("Failed to get kernel function");

    // Prepare vectors
    let n: u32 = 1024;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let c: Vec<f32> = vec![0.0; n as usize];

    println!("A: first 5 = {:?}", &a[..5]);
    println!("B: first 5 = {:?}", &b[..5]);

    // Copy to device
    let device_a = device.htod_sync_copy(&a).expect("Failed to copy A");
    let device_b = device.htod_sync_copy(&b).expect("Failed to copy B");
    let device_c = device.htod_sync_copy(&c).expect("Failed to copy C");

    let config = LaunchConfig {
        grid_dim: ((n + 255) / 256, 1, 1),
        block_dim: (256.min(n), 1, 1),
        shared_mem_bytes: 0,
    };

    println!("Launching vector add kernel with {} elements...", n);
    unsafe {
        kernel
            .clone()
            .launch(config, (&device_a, &device_b, &device_c, n))
            .expect("Kernel launch failed");
    }
    device.synchronize().expect("Synchronize failed");

    let result = device
        .dtoh_sync_copy(&device_c)
        .expect("Failed to copy from device");

    println!("C: first 5 = {:?}", &result[..5]);

    // Verify: C[i] = A[i] + B[i]
    for i in 0..n as usize {
        let expected = a[i] + b[i];
        assert!(
            (result[i] - expected).abs() < 0.001,
            "Mismatch at index {}: expected {}, got {}",
            i,
            expected,
            result[i]
        );
    }

    println!("✓ GPU vector addition executed correctly on {} elements!", n);
}

#[test]
fn test_ringkernel_control_block_modification() {
    use cudarc::nvrtc::Ptx;
    use ringkernel_cuda::{is_cuda_available, RING_KERNEL_PTX_TEMPLATE};

    if !is_cuda_available() {
        println!("CUDA not available, skipping test");
        return;
    }

    println!("\n=== RingKernel Control Block Test ===");

    // Create device using cudarc directly for this test
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");

    // Load the RingKernel PTX
    let ptx = Ptx::from_src(RING_KERNEL_PTX_TEMPLATE);
    device
        .load_ptx(ptx, "ring_kernel", &["ring_kernel_main"])
        .expect("Failed to load RingKernel PTX");

    let kernel = device
        .get_func("ring_kernel", "ring_kernel_main")
        .expect("Failed to get kernel function");

    println!("RingKernel PTX loaded successfully");

    // Create a control block buffer (128 bytes)
    // Layout: is_active (u32), should_terminate (u32), has_terminated (u32), ...
    let control_block = vec![0u32; 32]; // 128 bytes
    let device_cb = device
        .htod_sync_copy(&control_block)
        .expect("Failed to copy control block");

    let cb_ptr = *device_cb.device_ptr();

    // Create dummy queue pointers (not used by current minimal kernel)
    let dummy_queue: Vec<u64> = vec![0; 16];
    let device_input = device.htod_sync_copy(&dummy_queue).expect("copy");
    let device_output = device.htod_sync_copy(&dummy_queue).expect("copy");
    let input_ptr = *device_input.device_ptr();
    let output_ptr = *device_output.device_ptr();
    let shared_ptr = 0u64;

    // Read initial state
    let initial = device
        .dtoh_sync_copy(&device_cb)
        .expect("Failed to read initial state");
    println!("Initial control block: has_terminated={}", initial[2]);
    assert_eq!(initial[2], 0, "has_terminated should be 0 initially");

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("Launching RingKernel...");
    unsafe {
        kernel
            .clone()
            .launch(config, (cb_ptr, input_ptr, output_ptr, shared_ptr))
            .expect("Kernel launch failed");
    }
    device.synchronize().expect("Synchronize failed");

    // Read final state
    let final_state = device
        .dtoh_sync_copy(&device_cb)
        .expect("Failed to read final state");
    println!("Final control block: has_terminated={}", final_state[2]);

    // The RingKernel PTX sets has_terminated to 1
    assert_eq!(
        final_state[2], 1,
        "RingKernel should have set has_terminated to 1"
    );

    println!("✓ RingKernel executed on GPU and modified control block!");
}

#[test]
fn test_large_scale_gpu_computation() {
    let device_count = CudaDevice::count().unwrap_or(0);
    if device_count == 0 {
        println!("No CUDA devices found, skipping test");
        return;
    }

    println!("\n=== Large Scale GPU Computation Test ===");
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");

    let ptx = compile_ptx(INCREMENT_KERNEL_CUDA).expect("Failed to compile CUDA kernel");
    device
        .load_ptx(ptx, "increment_large", &["increment_kernel"])
        .expect("Failed to load PTX");

    let kernel = device
        .get_func("increment_large", "increment_kernel")
        .expect("Failed to get kernel function");

    // Large array: 1 million elements
    let n: u32 = 1_000_000;
    let host_data: Vec<u32> = vec![42; n as usize];

    println!("Processing {} elements on GPU...", n);
    let start = std::time::Instant::now();

    let device_data = device
        .htod_sync_copy(&host_data)
        .expect("Failed to copy to device");
    let htod_time = start.elapsed();

    // Use multiple blocks for large computation
    let threads_per_block = 256u32;
    let blocks = (n + threads_per_block - 1) / threads_per_block;

    let config = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    let kernel_start = std::time::Instant::now();
    unsafe {
        kernel
            .clone()
            .launch(config, (&device_data, n))
            .expect("Kernel launch failed");
    }
    device.synchronize().expect("Synchronize failed");
    let kernel_time = kernel_start.elapsed();

    let dtoh_start = std::time::Instant::now();
    let result = device
        .dtoh_sync_copy(&device_data)
        .expect("Failed to copy from device");
    let dtoh_time = dtoh_start.elapsed();

    let total_time = start.elapsed();

    println!("Timing:");
    println!("  HtoD transfer: {:?}", htod_time);
    println!("  Kernel execution: {:?}", kernel_time);
    println!("  DtoH transfer: {:?}", dtoh_time);
    println!("  Total: {:?}", total_time);

    // Verify a sample of results
    let sample_indices = [0, 100, 1000, 10000, 100000, 500000, 999999];
    for &i in &sample_indices {
        if i < n as usize {
            assert_eq!(
                result[i], 43,
                "Element {} should be 43 (42+1), got {}",
                i, result[i]
            );
        }
    }

    println!("✓ Large scale GPU computation verified on {} elements!", n);
    println!(
        "  Throughput: {:.2} million elements/sec",
        n as f64 / total_time.as_secs_f64() / 1_000_000.0
    );
}

#[test]
fn test_gpu_sum_reduction() {
    let device_count = CudaDevice::count().unwrap_or(0);
    if device_count == 0 {
        println!("No CUDA devices found, skipping test");
        return;
    }

    println!("\n=== GPU Sum Reduction Test ===");
    let device = CudaDevice::new(0).expect("Failed to create CUDA device");

    let ptx = compile_ptx(REDUCTION_KERNEL_CUDA).expect("Failed to compile CUDA kernel");
    device
        .load_ptx(ptx, "reduction", &["sum_reduction"])
        .expect("Failed to load PTX");

    let kernel = device
        .get_func("reduction", "sum_reduction")
        .expect("Failed to get kernel function");

    // Create input: [1.0, 1.0, 1.0, ...] for easy verification
    let n: u32 = 1024;
    let host_data: Vec<f32> = vec![1.0; n as usize];
    let expected_sum = n as f32;

    println!("Computing sum of {} ones...", n);

    let device_input = device.htod_sync_copy(&host_data).expect("Failed to copy input");

    // Output: one partial sum per block
    let threads_per_block = 256u32;
    let blocks = (n + threads_per_block - 1) / threads_per_block;
    let partial_sums: Vec<f32> = vec![0.0; blocks as usize];
    let device_output = device.htod_sync_copy(&partial_sums).expect("Failed to copy output");

    let config = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: threads_per_block as u32 * std::mem::size_of::<f32>() as u32,
    };

    unsafe {
        kernel
            .clone()
            .launch(config, (&device_input, &device_output, n))
            .expect("Kernel launch failed");
    }
    device.synchronize().expect("Synchronize failed");

    let result = device.dtoh_sync_copy(&device_output).expect("Failed to copy result");

    // Sum the partial results on CPU
    let total_sum: f32 = result.iter().sum();
    println!("GPU computed sum: {}", total_sum);
    println!("Expected sum: {}", expected_sum);

    assert!(
        (total_sum - expected_sum).abs() < 0.001,
        "Sum mismatch: expected {}, got {}",
        expected_sum,
        total_sum
    );

    println!("✓ GPU sum reduction computed correctly!");
}
