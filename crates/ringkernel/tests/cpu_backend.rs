//! Integration tests for the CPU backend.

use ringkernel::prelude::*;

/// Test basic runtime creation with CPU backend.
#[tokio::test]
async fn test_cpu_runtime_creation() {
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await
        .expect("Failed to create CPU runtime");

    assert_eq!(runtime.backend(), Backend::Cpu);
}

/// Test kernel launch and lifecycle.
#[tokio::test]
async fn test_kernel_lifecycle() {
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await
        .expect("Failed to create runtime");

    // Launch a kernel
    let kernel = runtime
        .launch("test_kernel", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");

    assert_eq!(kernel.id().as_str(), "test_kernel");

    // List kernels
    let kernels = runtime.list_kernels();
    assert_eq!(kernels.len(), 1);
    assert_eq!(kernels[0].as_str(), "test_kernel");

    // Shutdown
    runtime.shutdown().await.expect("Failed to shutdown");
}

/// Test launching multiple kernels.
#[tokio::test]
async fn test_multiple_kernels() {
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await
        .expect("Failed to create runtime");

    // Launch multiple kernels
    let kernel1 = runtime
        .launch("kernel_1", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel 1");

    let kernel2 = runtime
        .launch("kernel_2", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel 2");

    let kernel3 = runtime
        .launch("kernel_3", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel 3");

    assert_eq!(kernel1.id().as_str(), "kernel_1");
    assert_eq!(kernel2.id().as_str(), "kernel_2");
    assert_eq!(kernel3.id().as_str(), "kernel_3");

    // List kernels
    let kernels = runtime.list_kernels();
    assert_eq!(kernels.len(), 3);

    // Check metrics
    let metrics = runtime.metrics();
    assert_eq!(metrics.total_launched, 3);

    runtime.shutdown().await.expect("Failed to shutdown");
}

/// Test runtime metrics.
#[tokio::test]
async fn test_runtime_metrics() {
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await
        .expect("Failed to create runtime");

    // Initial metrics
    let metrics = runtime.metrics();
    assert_eq!(metrics.total_launched, 0);
    assert_eq!(metrics.active_kernels, 0);

    // Launch a kernel
    let _kernel = runtime
        .launch("metrics_test", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");

    // Check updated metrics
    let metrics = runtime.metrics();
    assert_eq!(metrics.total_launched, 1);

    runtime.shutdown().await.expect("Failed to shutdown");
}

/// Test backend availability checking.
#[tokio::test]
async fn test_backend_availability() {
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await
        .expect("Failed to create runtime");

    // CPU should always be available
    assert!(runtime.is_backend_available(Backend::Cpu));

    // Other backends depend on feature flags and hardware
    // but the method should not panic
    let _ = runtime.is_backend_available(Backend::Cuda);
    let _ = runtime.is_backend_available(Backend::Metal);
    let _ = runtime.is_backend_available(Backend::Wgpu);

    runtime.shutdown().await.expect("Failed to shutdown");
}

/// Test auto backend selection falls back to CPU.
#[tokio::test]
async fn test_auto_backend_fallback() {
    let runtime = RingKernel::builder()
        .backend(Backend::Auto)
        .build()
        .await
        .expect("Failed to create runtime with auto backend");

    // In CI without GPU, should fall back to CPU
    // (unless GPU backends are available)
    let backend = runtime.backend();
    assert!(
        matches!(
            backend,
            Backend::Cpu | Backend::Cuda | Backend::Metal | Backend::Wgpu
        ),
        "Backend should be one of the supported types"
    );

    runtime.shutdown().await.expect("Failed to shutdown");
}

/// Test launch options.
#[tokio::test]
async fn test_launch_options() {
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await
        .expect("Failed to create runtime");

    // Custom launch options
    let options = LaunchOptions::multi_block(2, 128).with_queue_capacity(2048);

    let kernel = runtime
        .launch("options_test", options)
        .await
        .expect("Failed to launch kernel with options");

    assert_eq!(kernel.id().as_str(), "options_test");

    runtime.shutdown().await.expect("Failed to shutdown");
}

/// Test duplicate kernel ID handling.
#[tokio::test]
async fn test_duplicate_kernel_id() {
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await
        .expect("Failed to create runtime");

    // Launch first kernel
    let _kernel1 = runtime
        .launch("duplicate_test", LaunchOptions::default())
        .await
        .expect("Failed to launch first kernel");

    // Try to launch second kernel with same ID
    let result = runtime
        .launch("duplicate_test", LaunchOptions::default())
        .await;

    // Should fail with duplicate ID error
    assert!(result.is_err());

    runtime.shutdown().await.expect("Failed to shutdown");
}

/// Test getting kernel by ID.
#[tokio::test]
async fn test_get_kernel() {
    let runtime = RingKernel::builder()
        .backend(Backend::Cpu)
        .build()
        .await
        .expect("Failed to create runtime");

    // Launch kernel
    let kernel = runtime
        .launch("get_test", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");

    // Get kernel by ID
    let retrieved = runtime.get_kernel(kernel.id());
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id(), kernel.id());

    // Try to get non-existent kernel
    let non_existent = runtime.get_kernel(&KernelId::new("non_existent"));
    assert!(non_existent.is_none());

    runtime.shutdown().await.expect("Failed to shutdown");
}
