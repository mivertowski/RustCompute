//! Integration tests for ControlBlock functionality.

use ringkernel::prelude::*;
use std::time::Duration;

/// Test control block creation with capacities.
#[test]
fn test_control_block_creation() {
    let cb = ControlBlock::with_capacities(1024, 1024);

    // Initial state should be inactive
    assert!(!cb.is_active());
    assert!(!cb.should_terminate());
    assert!(!cb.has_terminated());

    // Queues should be empty
    assert!(cb.input_queue_empty());
    assert!(cb.output_queue_empty());
}

/// Test control block lifecycle flags.
#[test]
fn test_control_block_lifecycle_flags() {
    let mut cb = ControlBlock::with_capacities(1024, 1024);

    // Set active flag
    cb.is_active = 1;
    assert!(cb.is_active());

    cb.is_active = 0;
    assert!(!cb.is_active());

    // Set termination flags
    cb.should_terminate = 1;
    assert!(cb.should_terminate());

    cb.has_terminated = 1;
    assert!(cb.has_terminated());
}

/// Test control block message counting.
#[test]
fn test_control_block_message_count() {
    let mut cb = ControlBlock::with_capacities(1024, 1024);

    assert_eq!(cb.messages_processed, 0);

    cb.messages_processed = 1;
    assert_eq!(cb.messages_processed, 1);

    cb.messages_processed = 100;
    assert_eq!(cb.messages_processed, 100);
}

/// Test control block size and alignment.
#[test]
fn test_control_block_size_alignment() {
    // Control block should be 128 bytes and 128-byte aligned
    assert_eq!(std::mem::size_of::<ControlBlock>(), 128);
    assert_eq!(std::mem::align_of::<ControlBlock>(), 128);
}

/// Test control block default values.
#[test]
fn test_control_block_defaults() {
    let cb = ControlBlock::default();

    assert!(!cb.is_active());
    assert!(!cb.should_terminate());
    assert!(!cb.has_terminated());
    assert_eq!(cb.messages_processed, 0);
}

/// Test control block queue size calculation.
#[test]
fn test_control_block_queue_size() {
    let mut cb = ControlBlock::with_capacities(1024, 1024);

    // Empty queues
    assert_eq!(cb.input_queue_size(), 0);
    assert_eq!(cb.output_queue_size(), 0);

    // Add some messages
    cb.input_head = 10;
    cb.input_tail = 5;
    assert_eq!(cb.input_queue_size(), 5);

    cb.output_head = 20;
    cb.output_tail = 15;
    assert_eq!(cb.output_queue_size(), 5);
}

/// Test control block queue full/empty status.
#[test]
fn test_control_block_queue_status() {
    let mut cb = ControlBlock::with_capacities(16, 16);

    // Initially empty
    assert!(cb.input_queue_empty());
    assert!(cb.output_queue_empty());
    assert!(!cb.input_queue_full());
    assert!(!cb.output_queue_full());

    // Fill the queue
    cb.input_head = 16;
    cb.input_tail = 0;
    assert!(!cb.input_queue_empty());
    assert!(cb.input_queue_full());
}

/// Test telemetry buffer creation.
#[test]
fn test_telemetry_buffer_creation() {
    let tb = TelemetryBuffer::new();

    assert_eq!(tb.messages_processed, 0);
    assert_eq!(tb.messages_dropped, 0);
    assert_eq!(tb.last_error, 0);
}

/// Test telemetry buffer message tracking.
#[test]
fn test_telemetry_message_tracking() {
    let mut tb = TelemetryBuffer::new();

    tb.messages_processed = 3;
    assert_eq!(tb.messages_processed, 3);
}

/// Test telemetry buffer dropped messages.
#[test]
fn test_telemetry_dropped_messages() {
    let mut tb = TelemetryBuffer::new();

    tb.messages_dropped = 2;
    assert_eq!(tb.messages_dropped, 2);
}

/// Test telemetry buffer latency tracking.
#[test]
fn test_telemetry_latency_tracking() {
    let mut tb = TelemetryBuffer::new();

    tb.messages_processed = 3;
    tb.total_latency_us = 6000; // 6000 us total

    // Average should be 2000us (2ms)
    let avg_us = tb.avg_latency_us();
    assert!((avg_us - 2000.0).abs() < 0.01);
}

/// Test telemetry buffer error tracking.
#[test]
fn test_telemetry_error_tracking() {
    let mut tb = TelemetryBuffer::new();

    tb.last_error = 42;
    assert_eq!(tb.last_error, 42);
}

/// Test telemetry buffer size.
#[test]
fn test_telemetry_buffer_size() {
    // Telemetry buffer should be 64 bytes
    assert_eq!(std::mem::size_of::<TelemetryBuffer>(), 64);
}

/// Test telemetry buffer throughput calculation.
#[test]
fn test_telemetry_throughput() {
    let mut tb = TelemetryBuffer::new();
    tb.messages_processed = 1000;

    assert_eq!(tb.throughput(1.0), 1000.0);
    assert_eq!(tb.throughput(2.0), 500.0);
    assert_eq!(tb.throughput(0.0), 0.0);
}

/// Test telemetry buffer drop rate.
#[test]
fn test_telemetry_drop_rate() {
    let mut tb = TelemetryBuffer::new();
    tb.messages_processed = 90;
    tb.messages_dropped = 10;

    let drop_rate = tb.drop_rate();
    assert!((drop_rate - 0.1).abs() < 0.001);
}

/// Test telemetry buffer merge.
#[test]
fn test_telemetry_merge() {
    let mut tb1 = TelemetryBuffer::new();
    tb1.messages_processed = 100;
    tb1.min_latency_us = 10;
    tb1.max_latency_us = 100;

    let mut tb2 = TelemetryBuffer::new();
    tb2.messages_processed = 50;
    tb2.min_latency_us = 5;
    tb2.max_latency_us = 200;

    tb1.merge(&tb2);

    assert_eq!(tb1.messages_processed, 150);
    assert_eq!(tb1.min_latency_us, 5);
    assert_eq!(tb1.max_latency_us, 200);
}

/// Test launch options defaults.
#[test]
fn test_launch_options_defaults() {
    let options = LaunchOptions::default();

    assert_eq!(options.grid_size, 1);
    assert_eq!(options.block_size, 256);
    assert_eq!(options.input_queue_capacity, 1024);
    assert_eq!(options.output_queue_capacity, 1024);
}

/// Test launch options single block helper.
#[test]
fn test_launch_options_single_block() {
    let options = LaunchOptions::single_block(128);

    assert_eq!(options.grid_size, 1);
    assert_eq!(options.block_size, 128);
}

/// Test launch options multi block helper.
#[test]
fn test_launch_options_multi_block() {
    let options = LaunchOptions::multi_block(4, 512);

    assert_eq!(options.grid_size, 4);
    assert_eq!(options.block_size, 512);
}

/// Test launch options with custom queue capacity.
#[test]
fn test_launch_options_with_queue_capacity() {
    let options = LaunchOptions::default().with_queue_capacity(4096);

    assert_eq!(options.input_queue_capacity, 4096);
    assert_eq!(options.output_queue_capacity, 4096);
}

/// Test launch options with shared memory.
#[test]
fn test_launch_options_with_shared_memory() {
    let options = LaunchOptions::default().with_shared_memory(8192);

    assert_eq!(options.shared_memory_size, 8192);
}

/// Test kernel ID creation and comparison.
#[test]
fn test_kernel_id() {
    let id1 = KernelId::new("test_kernel");
    let id2 = KernelId::new("test_kernel");
    let id3 = KernelId::new("other_kernel");

    assert_eq!(id1, id2);
    assert_ne!(id1, id3);
    assert_eq!(id1.as_str(), "test_kernel");
}

/// Test kernel ID from string.
#[test]
fn test_kernel_id_from_string() {
    let id: KernelId = "my_kernel".into();
    assert_eq!(id.as_str(), "my_kernel");

    let id: KernelId = String::from("another_kernel").into();
    assert_eq!(id.as_str(), "another_kernel");
}

/// Test kernel state transitions.
#[test]
fn test_kernel_state() {
    // Test can_activate
    assert!(KernelState::Launched.can_activate());
    assert!(KernelState::Deactivated.can_activate());
    assert!(!KernelState::Active.can_activate());
    assert!(!KernelState::Terminated.can_activate());

    // Test can_deactivate
    assert!(KernelState::Active.can_deactivate());
    assert!(!KernelState::Launched.can_deactivate());

    // Test can_terminate
    assert!(KernelState::Active.can_terminate());
    assert!(KernelState::Deactivated.can_terminate());
    assert!(!KernelState::Terminated.can_terminate());

    // Test is_running
    assert!(KernelState::Active.is_running());
    assert!(!KernelState::Launched.is_running());

    // Test is_finished
    assert!(KernelState::Terminated.is_finished());
    assert!(!KernelState::Active.is_finished());
}

/// Test kernel status.
#[test]
fn test_kernel_status() {
    let status = KernelStatus {
        id: KernelId::new("test"),
        state: KernelState::Active,
        mode: KernelMode::Persistent,
        input_queue_depth: 10,
        output_queue_depth: 5,
        messages_processed: 100,
        uptime: Duration::from_secs(60),
    };

    assert!(matches!(status.state, KernelState::Active));
    assert_eq!(status.messages_processed, 100);
    assert_eq!(status.input_queue_depth, 10);
    assert_eq!(status.output_queue_depth, 5);
    assert_eq!(status.uptime.as_secs(), 60);
}

/// Test runtime metrics.
#[test]
fn test_runtime_metrics() {
    let metrics = RuntimeMetrics {
        total_launched: 10,
        active_kernels: 5,
        messages_sent: 1000,
        messages_received: 950,
        gpu_memory_used: 1024 * 1024,
        host_memory_used: 2048 * 1024,
    };

    assert_eq!(metrics.total_launched, 10);
    assert_eq!(metrics.active_kernels, 5);
    assert_eq!(metrics.messages_sent, 1000);
    assert_eq!(metrics.messages_received, 950);
    assert_eq!(metrics.gpu_memory_used, 1024 * 1024);
    assert_eq!(metrics.host_memory_used, 2048 * 1024);
}

/// Test backend names.
#[test]
fn test_backend_names() {
    assert_eq!(Backend::Cpu.name(), "CPU");
    assert_eq!(Backend::Cuda.name(), "CUDA");
    assert_eq!(Backend::Metal.name(), "Metal");
    assert_eq!(Backend::Wgpu.name(), "WebGPU");
    assert_eq!(Backend::Auto.name(), "Auto");
}

/// Test kernel metrics.
#[test]
fn test_kernel_metrics() {
    let mut metrics = KernelMetrics::new("test_kernel");
    metrics.telemetry.messages_processed = 100;
    metrics.uptime = Duration::from_secs(10);

    assert_eq!(metrics.kernel_id, "test_kernel");
    assert_eq!(metrics.telemetry.messages_processed, 100);
}

/// Test kernel metrics transfer bandwidth.
#[test]
fn test_kernel_metrics_bandwidth() {
    let mut metrics = KernelMetrics::new("test");
    metrics.bytes_to_device = 1000;
    metrics.bytes_from_device = 1000;
    metrics.uptime = Duration::from_secs(1);

    let bandwidth = metrics.transfer_bandwidth();
    assert_eq!(bandwidth, 2000.0);
}
