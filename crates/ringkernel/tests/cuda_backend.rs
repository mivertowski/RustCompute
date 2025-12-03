//! Comprehensive CUDA backend integration tests.
//!
//! These tests require CUDA hardware to run. They are NOT marked with #[ignore]
//! so they will run when the cuda feature is enabled and hardware is available.
//!
//! Run with: cargo test --features cuda -p ringkernel --test cuda_backend
//!
//! For systems without CUDA, these tests will be skipped at runtime via
//! the skip_without_cuda! macro.

#![cfg(feature = "cuda")]

use ringkernel::prelude::*;
use ringkernel_cuda::{
    cuda_device_count, is_cuda_available, CudaBuffer, CudaControlBlock, CudaDevice, CudaRuntime,
};
use std::time::Duration;

/// Helper function to safely check if CUDA is available.
/// This catches panics from cudarc when no CUDA device is present.
fn cuda_is_available_safe() -> bool {
    std::panic::catch_unwind(is_cuda_available).unwrap_or(false)
}

/// Helper macro to skip tests when CUDA is not available.
macro_rules! skip_without_cuda {
    () => {
        if !cuda_is_available_safe() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }
    };
}

// ============================================================================
// Device Detection and Enumeration Tests
// ============================================================================

#[test]
fn test_cuda_availability_detection() {
    // This test always runs - it just checks the detection mechanism
    // Use catch_unwind since cudarc may panic on systems without CUDA drivers
    let available = cuda_is_available_safe();
    let count = std::panic::catch_unwind(cuda_device_count).unwrap_or(0);

    println!("CUDA available: {}", available);
    println!("CUDA device count: {}", count);

    // If available, count should be > 0
    if available {
        assert!(count > 0, "CUDA available but device count is 0");
    }
}

#[test]
fn test_cuda_device_enumeration() {
    skip_without_cuda!();

    let device = CudaDevice::new(0).expect("Failed to create CUDA device 0");

    println!("Device 0: {}", device.name());
    println!(
        "  Compute Capability: {}.{}",
        device.compute_capability().0,
        device.compute_capability().1
    );
    println!("  Total Memory: {} MB", device.total_memory() / 1024 / 1024);
    println!(
        "  Supports Persistent Kernels: {}",
        device.supports_persistent_kernels()
    );

    // Basic sanity checks
    assert!(!device.name().is_empty());
    assert!(device.total_memory() > 0);
    // CC should be at least 3.0 for any modern CUDA device
    assert!(device.compute_capability().0 >= 3);
}

#[test]
fn test_cuda_multi_device_enumeration() {
    skip_without_cuda!();

    let count = cuda_device_count();
    println!("Found {} CUDA device(s)", count);

    for i in 0..count {
        match CudaDevice::new(i) {
            Ok(device) => {
                println!(
                    "  Device {}: {} (CC {}.{})",
                    i,
                    device.name(),
                    device.compute_capability().0,
                    device.compute_capability().1
                );
            }
            Err(e) => {
                println!("  Device {}: Failed to initialize: {}", i, e);
            }
        }
    }
}

// ============================================================================
// Memory Allocation and Transfer Tests
// ============================================================================

#[test]
fn test_cuda_buffer_allocation() {
    skip_without_cuda!();

    let device = CudaDevice::new(0).unwrap();

    // Test various buffer sizes
    let sizes = [64, 1024, 4096, 1024 * 1024]; // 64B to 1MB

    for size in sizes {
        let buffer = CudaBuffer::new(&device, size)
            .unwrap_or_else(|_| panic!("Failed to allocate {} bytes", size));

        assert_eq!(buffer.size(), size);
        assert!(buffer.device_ptr() > 0, "Device pointer should be non-zero");

        println!(
            "Allocated {} bytes at device ptr 0x{:x}",
            size,
            buffer.device_ptr()
        );
    }
}

#[test]
fn test_cuda_buffer_htod_dtoh_transfer() {
    skip_without_cuda!();

    use ringkernel_core::memory::GpuBuffer;

    let device = CudaDevice::new(0).unwrap();

    // Test with different data patterns
    let test_cases: Vec<Vec<u8>> = vec![
        vec![0u8; 1024],                              // All zeros
        vec![0xFFu8; 1024],                           // All ones
        (0u8..=255).cycle().take(1024).collect(),     // Repeating pattern
        (0..1024).map(|i| (i % 256) as u8).collect(), // Sequential
    ];

    for (i, original) in test_cases.iter().enumerate() {
        let buffer = CudaBuffer::new(&device, original.len()).unwrap();

        // Host to device
        buffer.copy_from_host(original).unwrap();

        // Device to host
        let mut result = vec![0u8; original.len()];
        buffer.copy_to_host(&mut result).unwrap();

        assert_eq!(original, &result, "Data mismatch in test case {}", i);
    }

    println!("All HtoD/DtoH transfer patterns verified");
}

#[test]
fn test_cuda_large_transfer() {
    skip_without_cuda!();

    use ringkernel_core::memory::GpuBuffer;

    let device = CudaDevice::new(0).unwrap();

    // Test with 64 MB transfer
    let size = 64 * 1024 * 1024;
    let buffer = CudaBuffer::new(&device, size).unwrap();

    // Create test data
    let original: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    let start = std::time::Instant::now();
    buffer.copy_from_host(&original).unwrap();
    let htod_time = start.elapsed();

    let mut result = vec![0u8; size];
    let start = std::time::Instant::now();
    buffer.copy_to_host(&mut result).unwrap();
    let dtoh_time = start.elapsed();

    assert_eq!(original, result);

    let htod_bandwidth = size as f64 / htod_time.as_secs_f64() / 1e9;
    let dtoh_bandwidth = size as f64 / dtoh_time.as_secs_f64() / 1e9;

    println!(
        "64 MB transfer: HtoD {:.2} GB/s, DtoH {:.2} GB/s",
        htod_bandwidth, dtoh_bandwidth
    );
}

// ============================================================================
// Control Block Tests
// ============================================================================

#[test]
fn test_cuda_control_block_lifecycle() {
    skip_without_cuda!();

    let device = CudaDevice::new(0).unwrap();
    let cb = CudaControlBlock::new(&device).unwrap();

    // Verify initial state
    let state = cb.read().unwrap();
    assert!(!state.is_active());
    assert!(!state.should_terminate());
    assert!(!state.has_terminated());
    assert_eq!(state.messages_processed, 0);

    // Activate
    cb.set_active(true).unwrap();
    let state = cb.read().unwrap();
    assert!(state.is_active());

    // Deactivate
    cb.set_active(false).unwrap();
    let state = cb.read().unwrap();
    assert!(!state.is_active());

    // Request termination
    cb.set_should_terminate(true).unwrap();
    let state = cb.read().unwrap();
    assert!(state.should_terminate());

    println!("Control block lifecycle verified");
}

#[test]
fn test_cuda_control_block_queue_state() {
    skip_without_cuda!();

    use ringkernel_core::control::ControlBlock;

    let device = CudaDevice::new(0).unwrap();
    let cb = CudaControlBlock::new(&device).unwrap();

    // Create a control block with queue state
    let mut host_cb = ControlBlock::with_capacities(1024, 1024);
    host_cb.input_head = 100;
    host_cb.input_tail = 50;
    host_cb.output_head = 200;
    host_cb.output_tail = 150;
    host_cb.messages_processed = 500;

    // Write to device
    cb.write(&host_cb).unwrap();

    // Read back and verify
    let read_cb = cb.read().unwrap();
    assert_eq!(read_cb.input_head, 100);
    assert_eq!(read_cb.input_tail, 50);
    assert_eq!(read_cb.output_head, 200);
    assert_eq!(read_cb.output_tail, 150);
    assert_eq!(read_cb.messages_processed, 500);
    assert_eq!(read_cb.input_queue_size(), 50);
    assert_eq!(read_cb.output_queue_size(), 50);

    println!("Control block queue state verified");
}

// ============================================================================
// Runtime Tests
// ============================================================================

#[tokio::test]
async fn test_cuda_runtime_creation() {
    skip_without_cuda!();

    let runtime = CudaRuntime::new().await.unwrap();
    assert_eq!(runtime.backend(), Backend::Cuda);
    assert!(runtime.is_backend_available(Backend::Cuda));
    assert!(runtime.is_backend_available(Backend::Cpu)); // CPU fallback always available

    let metrics = runtime.metrics();
    assert_eq!(metrics.active_kernels, 0);

    runtime.shutdown().await.unwrap();
    println!("CUDA runtime creation and shutdown verified");
}

#[tokio::test]
async fn test_cuda_kernel_launch() {
    skip_without_cuda!();

    let runtime = CudaRuntime::new().await.unwrap();

    let kernel = runtime
        .launch("test_kernel", LaunchOptions::default())
        .await
        .unwrap();

    assert_eq!(kernel.id().as_str(), "test_kernel");

    let status = kernel.status();
    assert_eq!(status.state, KernelState::Launched);
    assert_eq!(status.mode, ringkernel_core::types::KernelMode::Persistent);

    runtime.shutdown().await.unwrap();
    println!("CUDA kernel launch verified");
}

#[tokio::test]
async fn test_cuda_kernel_lifecycle() {
    skip_without_cuda!();

    let runtime = CudaRuntime::new().await.unwrap();

    let kernel = runtime
        .launch("lifecycle_test", LaunchOptions::default())
        .await
        .unwrap();

    // Initial state is Launched
    assert_eq!(kernel.status().state, KernelState::Launched);

    // Activate
    kernel.activate().await.unwrap();
    assert_eq!(kernel.status().state, KernelState::Active);

    // Deactivate
    kernel.deactivate().await.unwrap();
    assert_eq!(kernel.status().state, KernelState::Deactivated);

    // Reactivate
    kernel.activate().await.unwrap();
    assert_eq!(kernel.status().state, KernelState::Active);

    // Terminate
    kernel.terminate().await.unwrap();

    runtime.shutdown().await.unwrap();
    println!("CUDA kernel lifecycle verified");
}

#[tokio::test]
async fn test_cuda_multiple_kernels() {
    skip_without_cuda!();

    let runtime = CudaRuntime::new().await.unwrap();

    // Launch multiple kernels
    let kernel1 = runtime
        .launch("kernel_1", LaunchOptions::default())
        .await
        .unwrap();
    let kernel2 = runtime
        .launch("kernel_2", LaunchOptions::default())
        .await
        .unwrap();
    let kernel3 = runtime
        .launch("kernel_3", LaunchOptions::default())
        .await
        .unwrap();

    // Verify all are tracked
    let kernels = runtime.list_kernels();
    assert_eq!(kernels.len(), 3);

    let metrics = runtime.metrics();
    assert_eq!(metrics.active_kernels, 3);
    assert_eq!(metrics.total_launched, 3);

    // Activate all
    kernel1.activate().await.unwrap();
    kernel2.activate().await.unwrap();
    kernel3.activate().await.unwrap();

    // Verify all active
    assert_eq!(kernel1.status().state, KernelState::Active);
    assert_eq!(kernel2.status().state, KernelState::Active);
    assert_eq!(kernel3.status().state, KernelState::Active);

    runtime.shutdown().await.unwrap();
    println!("Multiple CUDA kernels verified");
}

#[tokio::test]
async fn test_cuda_kernel_duplicate_detection() {
    skip_without_cuda!();

    let runtime = CudaRuntime::new().await.unwrap();

    // First launch succeeds
    runtime
        .launch("unique_kernel", LaunchOptions::default())
        .await
        .unwrap();

    // Duplicate launch should fail
    let result = runtime
        .launch("unique_kernel", LaunchOptions::default())
        .await;

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(
            matches!(e, RingKernelError::KernelAlreadyActive(_)),
            "Expected KernelAlreadyActive error, got: {:?}",
            e
        );
    }

    runtime.shutdown().await.unwrap();
    println!("Duplicate kernel detection verified");
}

#[tokio::test]
async fn test_cuda_launch_options() {
    skip_without_cuda!();

    let runtime = CudaRuntime::new().await.unwrap();

    let options = LaunchOptions::multi_block(4, 128)
        .with_queue_capacity(256)
        .with_shared_memory(1024);

    let kernel = runtime.launch("configured_kernel", options).await.unwrap();

    // Verify kernel was created with options
    assert_eq!(kernel.id().as_str(), "configured_kernel");

    runtime.shutdown().await.unwrap();
    println!("CUDA launch options verified");
}

// ============================================================================
// RingKernel Facade Tests (using Backend::Cuda)
// ============================================================================

#[tokio::test]
async fn test_ringkernel_cuda_backend() {
    skip_without_cuda!();

    let runtime = RingKernel::builder()
        .backend(Backend::Cuda)
        .build()
        .await
        .unwrap();

    assert_eq!(runtime.backend(), Backend::Cuda);

    let kernel = runtime
        .launch("facade_test", LaunchOptions::default())
        .await
        .unwrap();

    assert_eq!(kernel.id().as_str(), "facade_test");

    runtime.shutdown().await.unwrap();
    println!("RingKernel CUDA facade verified");
}

#[tokio::test]
async fn test_ringkernel_auto_selects_cuda() {
    skip_without_cuda!();

    // Auto should select CUDA when available
    let runtime = RingKernel::builder()
        .backend(Backend::Auto)
        .build()
        .await
        .unwrap();

    // On a system with CUDA, Auto should select CUDA
    assert_eq!(runtime.backend(), Backend::Cuda);

    runtime.shutdown().await.unwrap();
    println!("RingKernel auto-selection verified (selected CUDA)");
}

// ============================================================================
// Performance Tests
// ============================================================================

#[tokio::test]
async fn test_cuda_kernel_launch_latency() {
    skip_without_cuda!();

    let runtime = CudaRuntime::new().await.unwrap();

    let iterations = 10;
    let mut launch_times = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let start = std::time::Instant::now();
        let kernel = runtime
            .launch(&format!("latency_test_{}", i), LaunchOptions::default())
            .await
            .unwrap();
        launch_times.push(start.elapsed());
        kernel.terminate().await.unwrap();
    }

    let avg_launch = launch_times.iter().sum::<Duration>() / iterations as u32;
    let min_launch = launch_times.iter().min().unwrap();
    let max_launch = launch_times.iter().max().unwrap();

    println!("Kernel launch latency over {} iterations:", iterations);
    println!("  Average: {:?}", avg_launch);
    println!("  Min: {:?}", min_launch);
    println!("  Max: {:?}", max_launch);

    // Target: <1ms launch time (based on README performance targets)
    assert!(
        avg_launch < Duration::from_millis(100),
        "Average launch time {:?} exceeds 100ms threshold",
        avg_launch
    );

    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_cuda_activation_latency() {
    skip_without_cuda!();

    let runtime = CudaRuntime::new().await.unwrap();

    let kernel = runtime
        .launch("activation_latency", LaunchOptions::default())
        .await
        .unwrap();

    let iterations = 100;
    let mut activate_times = Vec::with_capacity(iterations);
    let mut deactivate_times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = std::time::Instant::now();
        kernel.activate().await.unwrap();
        activate_times.push(start.elapsed());

        let start = std::time::Instant::now();
        kernel.deactivate().await.unwrap();
        deactivate_times.push(start.elapsed());
    }

    let avg_activate = activate_times.iter().sum::<Duration>() / iterations as u32;
    let avg_deactivate = deactivate_times.iter().sum::<Duration>() / iterations as u32;

    println!("Activation latency over {} iterations:", iterations);
    println!("  Activate avg: {:?}", avg_activate);
    println!("  Deactivate avg: {:?}", avg_deactivate);

    runtime.shutdown().await.unwrap();
}

// ============================================================================
// Stress Tests
// ============================================================================

#[tokio::test]
async fn test_cuda_many_kernels_stress() {
    skip_without_cuda!();

    let runtime = CudaRuntime::new().await.unwrap();

    // Launch many kernels
    let num_kernels = 32;
    let mut kernels = Vec::with_capacity(num_kernels);

    let start = std::time::Instant::now();

    for i in 0..num_kernels {
        let kernel = runtime
            .launch(&format!("stress_kernel_{}", i), LaunchOptions::default())
            .await
            .unwrap();
        kernels.push(kernel);
    }

    let launch_time = start.elapsed();
    println!(
        "Launched {} kernels in {:?} ({:.2} kernels/sec)",
        num_kernels,
        launch_time,
        num_kernels as f64 / launch_time.as_secs_f64()
    );

    // Activate all
    for kernel in &kernels {
        kernel.activate().await.unwrap();
    }

    // Verify all active
    let metrics = runtime.metrics();
    assert_eq!(metrics.active_kernels, num_kernels);

    runtime.shutdown().await.unwrap();
    println!("Stress test with {} kernels passed", num_kernels);
}

#[test]
fn test_cuda_memory_stress() {
    skip_without_cuda!();

    let device = CudaDevice::new(0).unwrap();

    // Allocate many small buffers
    let num_buffers = 1000;
    let buffer_size = 4096; // 4KB each

    let start = std::time::Instant::now();
    let buffers: Vec<_> = (0..num_buffers)
        .map(|_| CudaBuffer::new(&device, buffer_size).unwrap())
        .collect();
    let alloc_time = start.elapsed();

    println!(
        "Allocated {} x {} byte buffers in {:?}",
        num_buffers, buffer_size, alloc_time
    );

    // All buffers should have unique device pointers
    let ptrs: std::collections::HashSet<_> = buffers.iter().map(|b| b.device_ptr()).collect();
    assert_eq!(ptrs.len(), num_buffers);

    drop(buffers);
    println!("Memory stress test passed");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_cuda_invalid_device_index() {
    skip_without_cuda!();

    // Try to create device with invalid index
    let result = CudaDevice::new(999);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_cuda_invalid_state_transitions() {
    skip_without_cuda!();

    let runtime = CudaRuntime::new().await.unwrap();

    let kernel = runtime
        .launch("state_test", LaunchOptions::default())
        .await
        .unwrap();

    // Try to deactivate before activating (should fail)
    let result = kernel.deactivate().await;
    assert!(result.is_err());

    // Activate first
    kernel.activate().await.unwrap();

    // Try to activate again (should fail - already active)
    let result = kernel.activate().await;
    assert!(result.is_err());

    runtime.shutdown().await.unwrap();
    println!("Invalid state transition handling verified");
}
