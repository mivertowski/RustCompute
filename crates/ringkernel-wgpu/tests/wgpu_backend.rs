//! WebGPU backend integration tests.
//!
//! These tests require GPU hardware and the `wgpu-tests` feature.
//! Run with: cargo test -p ringkernel-wgpu --features wgpu-tests -- --ignored

#![cfg(feature = "wgpu-tests")]

use ringkernel_core::runtime::{KernelState, LaunchOptions, RingKernelRuntime};
use ringkernel_wgpu::WgpuRuntime;

/// Helper to check if WebGPU is available on this system.
fn wgpu_available() -> bool {
    ringkernel_wgpu::is_wgpu_available()
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_runtime_creation() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::new().await.expect("Failed to create runtime");
    assert_eq!(runtime.backend(), ringkernel_core::runtime::Backend::Wgpu);
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_runtime_with_k2k() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::with_config(true)
        .await
        .expect("Failed to create runtime");
    assert!(runtime.is_k2k_enabled());
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_runtime_without_k2k() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::with_config(false)
        .await
        .expect("Failed to create runtime");
    assert!(!runtime.is_k2k_enabled());
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_kernel_launch() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::new().await.expect("Failed to create runtime");
    let kernel = runtime
        .launch("test_kernel", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");

    assert_eq!(kernel.id().as_str(), "test_kernel");
    assert_eq!(kernel.status().state, KernelState::Launched);

    runtime.shutdown().await.expect("Shutdown failed");
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_kernel_activate_deactivate() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::new().await.expect("Failed to create runtime");
    let kernel = runtime
        .launch("activate_test", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");

    // Initial state is Launched
    assert_eq!(kernel.status().state, KernelState::Launched);

    // Activate
    kernel.activate().await.expect("Failed to activate");
    assert_eq!(kernel.status().state, KernelState::Active);

    // Deactivate
    kernel.deactivate().await.expect("Failed to deactivate");
    assert_eq!(kernel.status().state, KernelState::Deactivated);

    runtime.shutdown().await.expect("Shutdown failed");
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_kernel_terminate() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::new().await.expect("Failed to create runtime");
    let kernel = runtime
        .launch("terminate_test", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");

    kernel.activate().await.expect("Failed to activate");
    kernel.terminate().await.expect("Failed to terminate");

    // After termination, kernel is in Terminating state
    assert_eq!(kernel.status().state, KernelState::Terminating);

    runtime.shutdown().await.expect("Shutdown failed");
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_duplicate_kernel_error() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::new().await.expect("Failed to create runtime");

    // Launch first kernel
    let _kernel1 = runtime
        .launch("duplicate_test", LaunchOptions::default())
        .await
        .expect("Failed to launch first kernel");

    // Try to launch second kernel with same ID - should fail
    let result = runtime
        .launch("duplicate_test", LaunchOptions::default())
        .await;

    assert!(result.is_err());

    runtime.shutdown().await.expect("Shutdown failed");
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_list_kernels() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::new().await.expect("Failed to create runtime");

    // Initially empty
    assert!(runtime.list_kernels().is_empty());

    // Launch kernels
    let _k1 = runtime
        .launch("kernel_a", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel_a");
    let _k2 = runtime
        .launch("kernel_b", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel_b");

    let kernels = runtime.list_kernels();
    assert_eq!(kernels.len(), 2);

    runtime.shutdown().await.expect("Shutdown failed");
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_get_kernel() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::new().await.expect("Failed to create runtime");

    let _kernel = runtime
        .launch("get_test", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");

    // Get existing kernel
    let kernel_id = ringkernel_core::runtime::KernelId::new("get_test");
    let retrieved = runtime.get_kernel(&kernel_id);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id().as_str(), "get_test");

    // Get non-existent kernel
    let non_existent = ringkernel_core::runtime::KernelId::new("non_existent");
    let not_found = runtime.get_kernel(&non_existent);
    assert!(not_found.is_none());

    runtime.shutdown().await.expect("Shutdown failed");
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_runtime_metrics() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::new().await.expect("Failed to create runtime");

    // Initial metrics
    let metrics = runtime.metrics();
    assert_eq!(metrics.active_kernels, 0);
    assert_eq!(metrics.total_launched, 0);

    // Launch kernels
    let _k1 = runtime
        .launch("metrics_a", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");
    let _k2 = runtime
        .launch("metrics_b", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");

    let metrics = runtime.metrics();
    assert_eq!(metrics.active_kernels, 2);
    assert_eq!(metrics.total_launched, 2);

    runtime.shutdown().await.expect("Shutdown failed");
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_custom_launch_options() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::new().await.expect("Failed to create runtime");

    let options = LaunchOptions::default()
        .with_block_size(128)
        .with_input_queue_capacity(256)
        .with_output_queue_capacity(256);

    let kernel = runtime
        .launch("custom_options", options)
        .await
        .expect("Failed to launch kernel");

    assert_eq!(kernel.id().as_str(), "custom_options");

    runtime.shutdown().await.expect("Shutdown failed");
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_kernel_status() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::new().await.expect("Failed to create runtime");

    let kernel = runtime
        .launch("status_test", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");

    let status = kernel.status();
    assert_eq!(status.id.as_str(), "status_test");
    assert_eq!(status.state, KernelState::Launched);
    assert_eq!(status.input_queue_depth, 0);
    assert_eq!(status.output_queue_depth, 0);
    assert_eq!(status.messages_processed, 0);

    runtime.shutdown().await.expect("Shutdown failed");
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_kernel_metrics() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::new().await.expect("Failed to create runtime");

    let kernel = runtime
        .launch("metrics_test", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");

    let metrics = kernel.metrics();
    assert_eq!(metrics.telemetry.messages_processed, 0);
    assert_eq!(metrics.telemetry.messages_dropped, 0);

    runtime.shutdown().await.expect("Shutdown failed");
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_buffer_roundtrip() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    use ringkernel_core::memory::GpuBuffer;
    use ringkernel_wgpu::{WgpuAdapter, WgpuBuffer};

    let adapter = WgpuAdapter::new().await.expect("Failed to create adapter");
    let buffer = WgpuBuffer::new(&adapter, 1024, Some("Test Buffer"));

    // Write data
    let data: Vec<u8> = (0..1024u32).map(|i| (i % 256) as u8).collect();
    buffer.copy_from_host(&data).expect("Failed to write data");

    // Read back
    let mut read_data = vec![0u8; 1024];
    buffer
        .copy_to_host(&mut read_data)
        .expect("Failed to read data");

    assert_eq!(data, read_data);
}

#[tokio::test]
#[ignore] // Requires GPU
async fn test_wgpu_shutdown_clears_kernels() {
    if !wgpu_available() {
        eprintln!("Skipping test: WebGPU not available");
        return;
    }

    let runtime = WgpuRuntime::new().await.expect("Failed to create runtime");

    let _k1 = runtime
        .launch("shutdown_a", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");
    let _k2 = runtime
        .launch("shutdown_b", LaunchOptions::default())
        .await
        .expect("Failed to launch kernel");

    assert_eq!(runtime.metrics().active_kernels, 2);

    runtime.shutdown().await.expect("Shutdown failed");

    // After shutdown, kernel list should be empty
    assert!(runtime.list_kernels().is_empty());
    assert_eq!(runtime.metrics().active_kernels, 0);
}
