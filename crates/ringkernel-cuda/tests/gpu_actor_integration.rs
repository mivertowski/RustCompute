//! GPU Actor Integration Tests
//!
//! These tests verify the new GPU actor functionality:
//! - Phase 1: Message queue operations (send_envelope, try_receive, receive)
//! - Phase 2: Safe cooperative launch via DirectPtxModule
//! - Phase 3: K2K GPU buffers and connections
//!
//! Run with: cargo test -p ringkernel-cuda --features cuda -- --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;

use ringkernel_core::control::ControlBlock;
use ringkernel_core::hlc::HlcTimestamp;
use ringkernel_core::message::{MessageEnvelope, MessageHeader};
use ringkernel_core::runtime::{LaunchOptions, RingKernelRuntime};

use ringkernel_cuda::driver_api::{
    CooperativeLaunchConfig, DirectPtxModule, KernelParams, RingKernelK2KParams, RingKernelParams,
};
use ringkernel_cuda::k2k_gpu::{
    CudaK2KBuffers, K2KConnectionManager, K2KInboxHeader, K2KRouteEntry, MAX_K2K_ROUTES,
};
use ringkernel_cuda::{CudaBuffer, CudaControlBlock, CudaDevice, CudaMessageQueue, CudaRuntime};

/// Helper to check if CUDA is available
fn cuda_available() -> bool {
    CudaContext::device_count().map(|c| c > 0).unwrap_or(false)
}

/// Helper to skip test if no CUDA
macro_rules! require_cuda {
    () => {
        if !cuda_available() {
            println!("No CUDA devices found, skipping test");
            return;
        }
    };
}

// =============================================================================
// Phase 1: Message Queue Operations Tests
// =============================================================================

#[test]
fn test_cuda_buffer_offset_operations() {
    require_cuda!();
    println!("=== CudaBuffer Offset Operations Test ===");

    let device = CudaDevice::new(0).expect("Failed to create device");
    let buffer = CudaBuffer::new(&device, 1024).expect("Failed to create buffer");

    // Write data at different offsets
    let data1 = vec![0xAA_u8; 256];
    let data2 = vec![0xBB_u8; 256];
    let data3 = vec![0xCC_u8; 256];

    buffer
        .copy_from_host_at(&data1, 0)
        .expect("Failed to write at offset 0");
    buffer
        .copy_from_host_at(&data2, 256)
        .expect("Failed to write at offset 256");
    buffer
        .copy_from_host_at(&data3, 512)
        .expect("Failed to write at offset 512");

    // Read back and verify
    let mut read1 = vec![0u8; 256];
    let mut read2 = vec![0u8; 256];
    let mut read3 = vec![0u8; 256];

    buffer
        .copy_to_host_at(&mut read1, 0)
        .expect("Failed to read at offset 0");
    buffer
        .copy_to_host_at(&mut read2, 256)
        .expect("Failed to read at offset 256");
    buffer
        .copy_to_host_at(&mut read3, 512)
        .expect("Failed to read at offset 512");

    assert_eq!(read1, data1, "Data at offset 0 mismatch");
    assert_eq!(read2, data2, "Data at offset 256 mismatch");
    assert_eq!(read3, data3, "Data at offset 512 mismatch");

    println!("Offset operations verified successfully");
}

#[test]
fn test_cuda_buffer_offset_bounds_check() {
    require_cuda!();
    println!("=== CudaBuffer Offset Bounds Check Test ===");

    let device = CudaDevice::new(0).expect("Failed to create device");
    let buffer = CudaBuffer::new(&device, 256).expect("Failed to create buffer");

    // This should fail - writing beyond buffer size
    let data = vec![0u8; 128];
    let result = buffer.copy_from_host_at(&data, 200); // 200 + 128 > 256

    assert!(result.is_err(), "Should fail when writing beyond buffer");
    println!("Bounds check working correctly");
}

#[test]
fn test_message_queue_envelope_roundtrip() {
    require_cuda!();
    println!("=== Message Queue Envelope Roundtrip Test ===");

    let device = CudaDevice::new(0).expect("Failed to create device");
    let queue = CudaMessageQueue::new(&device, 16, 4096).expect("Failed to create queue");

    // Create a test envelope
    let header = MessageHeader::new(
        42, // message_type
        1,  // source_kernel
        2,  // dest_kernel
        12, // payload_size
        HlcTimestamp::new(1000, 1, 0),
    );

    let payload = b"Hello World!".to_vec();
    let envelope = MessageEnvelope {
        header,
        payload: payload.clone(),
    };

    // Write to slot 0
    queue
        .write_envelope(0, &envelope)
        .expect("Failed to write envelope");
    println!("Envelope written to slot 0");

    // Read back
    let read_envelope = queue.read_envelope(0).expect("Failed to read envelope");

    // Verify
    assert_eq!(
        read_envelope.header.message_type, envelope.header.message_type,
        "Message type mismatch"
    );
    assert_eq!(
        read_envelope.header.source_kernel, envelope.header.source_kernel,
        "Source kernel mismatch"
    );
    assert_eq!(
        read_envelope.header.dest_kernel, envelope.header.dest_kernel,
        "Dest kernel mismatch"
    );
    assert_eq!(read_envelope.payload, payload, "Payload mismatch");

    println!("Envelope roundtrip verified successfully");
}

#[test]
fn test_message_queue_multiple_slots() {
    require_cuda!();
    println!("=== Message Queue Multiple Slots Test ===");

    let device = CudaDevice::new(0).expect("Failed to create device");
    let queue = CudaMessageQueue::new(&device, 8, 256).expect("Failed to create queue");

    // Write envelopes to multiple slots
    for slot in 0..8 {
        let header = MessageHeader::new(
            slot as u64,
            slot as u64,
            slot as u64 + 100,
            4,
            HlcTimestamp::new(slot as u64 * 100, slot as u64, 0),
        );
        let payload = (slot as u32).to_le_bytes().to_vec();
        let envelope = MessageEnvelope { header, payload };

        queue
            .write_envelope(slot, &envelope)
            .expect(&format!("Failed to write slot {}", slot));
    }

    println!("Wrote 8 envelopes to queue");

    // Read back and verify
    for slot in 0..8 {
        let envelope = queue
            .read_envelope(slot)
            .expect(&format!("Failed to read slot {}", slot));

        assert_eq!(
            envelope.header.message_type, slot as u64,
            "Slot {} message_type mismatch",
            slot
        );

        let payload_value = u32::from_le_bytes(envelope.payload.try_into().unwrap());
        assert_eq!(payload_value, slot as u32, "Slot {} payload mismatch", slot);
    }

    println!("All 8 slots verified successfully");
}

#[test]
fn test_control_block_queue_pointers() {
    require_cuda!();
    println!("=== Control Block Queue Pointers Test ===");

    let device = CudaDevice::new(0).expect("Failed to create device");
    let control_block = CudaControlBlock::new(&device).expect("Failed to create control block");

    // Read initial state
    let mut cb = control_block.read().expect("Failed to read control block");
    assert_eq!(cb.input_head, 0, "Initial input_head should be 0");
    assert_eq!(cb.input_tail, 0, "Initial input_tail should be 0");
    assert_eq!(cb.output_head, 0, "Initial output_head should be 0");
    assert_eq!(cb.output_tail, 0, "Initial output_tail should be 0");

    // Simulate enqueue (host increments input_head)
    cb.input_head = 5;
    cb.input_capacity = 64;
    cb.input_mask = 63;
    control_block
        .write(&cb)
        .expect("Failed to write control block");

    // Read back and verify
    let cb2 = control_block.read().expect("Failed to read control block");
    assert_eq!(cb2.input_head, 5, "input_head not updated");
    assert_eq!(cb2.input_capacity, 64, "input_capacity not set");
    assert_eq!(cb2.input_queue_size(), 5, "Queue size calculation wrong");

    println!("Control block queue pointers verified");
}

// =============================================================================
// Phase 2: DirectPtxModule Tests
// =============================================================================

/// Simple PTX kernel for testing module loading.
/// Uses PTX 8.0 / sm_89 for Ada Lovelace GPU compatibility (RTX 40xx series).
const SIMPLE_PTX_KERNEL: &str = r#"
.version 8.0
.target sm_89
.address_size 64

.visible .entry simple_kernel(
    .param .u64 param_ptr
) {
    .reg .u64 %ptr;
    .reg .u32 %one;

    // Load parameter
    ld.param.u64 %ptr, [param_ptr];

    // Write value 1 to output
    mov.u32 %one, 1;
    st.global.u32 [%ptr], %one;

    ret;
}
"#;

#[test]
fn test_direct_ptx_module_load() {
    require_cuda!();
    println!("=== DirectPtxModule Load Test ===");

    let device = CudaDevice::new(0).expect("Failed to create device");

    // Load PTX module
    let module =
        DirectPtxModule::load_ptx(&device, SIMPLE_PTX_KERNEL).expect("Failed to load PTX module");

    // Get function handle
    let func = module
        .get_function("simple_kernel")
        .expect("Failed to get function");
    assert!(!func.is_null(), "Function handle should not be null");

    println!("DirectPtxModule loaded and function retrieved successfully");
}

#[test]
fn test_direct_ptx_module_function_not_found() {
    require_cuda!();
    println!("=== DirectPtxModule Function Not Found Test ===");

    let device = CudaDevice::new(0).expect("Failed to create device");
    let module =
        DirectPtxModule::load_ptx(&device, SIMPLE_PTX_KERNEL).expect("Failed to load PTX module");

    // Try to get non-existent function
    let result = module.get_function("nonexistent_kernel");
    assert!(result.is_err(), "Should fail for non-existent function");

    println!("Function not found error handled correctly");
}

#[test]
fn test_kernel_params_layout() {
    println!("=== Kernel Params Layout Test ===");

    // Test RingKernelParams
    let mut params = RingKernelParams {
        control_ptr: 0x1000,
        input_ptr: 0x2000,
        output_ptr: 0x3000,
        shared_ptr: 0x4000,
    };

    let ptrs = params.as_param_ptrs();
    assert_eq!(ptrs.len(), 4, "Should have 4 parameter pointers");

    // Verify pointers point to correct values
    unsafe {
        assert_eq!(*(ptrs[0] as *const u64), 0x1000);
        assert_eq!(*(ptrs[1] as *const u64), 0x2000);
        assert_eq!(*(ptrs[2] as *const u64), 0x3000);
        assert_eq!(*(ptrs[3] as *const u64), 0x4000);
    }

    // Test RingKernelK2KParams
    let mut k2k_params = RingKernelK2KParams {
        control_ptr: 0x1000,
        input_ptr: 0x2000,
        output_ptr: 0x3000,
        shared_ptr: 0x4000,
        k2k_routes_ptr: 0x5000,
        k2k_inbox_ptr: 0x6000,
        k2k_outbox_ptr: 0x7000,
    };

    let k2k_ptrs = k2k_params.as_param_ptrs();
    assert_eq!(k2k_ptrs.len(), 7, "Should have 7 parameter pointers");

    println!("Kernel params layout verified");
}

#[test]
fn test_cooperative_launch_config() {
    println!("=== Cooperative Launch Config Test ===");

    let config = CooperativeLaunchConfig::default();
    assert_eq!(config.grid_dim, (1, 1, 1));
    assert_eq!(config.block_dim, (256, 1, 1));
    assert_eq!(config.shared_mem_bytes, 0);

    let custom_config = CooperativeLaunchConfig {
        grid_dim: (16, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 4096,
    };

    assert_eq!(custom_config.grid_dim.0, 16);
    assert_eq!(custom_config.block_dim.0, 128);
    assert_eq!(custom_config.shared_mem_bytes, 4096);

    println!("Cooperative launch config verified");
}

// =============================================================================
// Phase 3: K2K GPU Tests
// =============================================================================

#[test]
fn test_k2k_inbox_header_operations() {
    println!("=== K2K Inbox Header Operations Test ===");

    let mut header = K2KInboxHeader::new(64, 256);

    assert!(header.is_empty(), "New header should be empty");
    assert!(!header.is_full(), "New header should not be full");
    assert_eq!(header.size(), 0, "New header size should be 0");
    assert_eq!(header.mask, 63, "Mask should be capacity - 1");
    assert_eq!(header.capacity, 64, "Capacity should be 64");
    assert_eq!(header.msg_size, 256, "Message size should be 256");

    // Simulate adding messages
    header.head = 10;
    assert!(
        !header.is_empty(),
        "Header with messages should not be empty"
    );
    assert_eq!(header.size(), 10, "Size should be 10");

    // Simulate consuming messages
    header.tail = 5;
    assert_eq!(header.size(), 5, "Size should be 5 after consuming");

    // Test full condition
    header.head = 64;
    header.tail = 0;
    assert!(header.is_full(), "Header should be full");

    // Test wraparound
    header.head = u64::MAX;
    header.tail = u64::MAX - 10;
    assert_eq!(header.size(), 10, "Wraparound size calculation should work");

    println!("K2K inbox header operations verified");
}

#[test]
fn test_k2k_route_entry() {
    println!("=== K2K Route Entry Test ===");

    let empty_entry = K2KRouteEntry::default();
    assert!(empty_entry.is_empty(), "Default entry should be empty");

    let entry = K2KRouteEntry {
        target_kernel_id: 42,
        target_inbox: 0x10000,
        target_head: 0x10000,
        target_tail: 0x10008,
        capacity: 64,
        mask: 63,
        msg_size: 256,
        _padding: 0,
    };

    assert!(
        !entry.is_empty(),
        "Entry with kernel_id should not be empty"
    );
    assert_eq!(entry.target_kernel_id, 42);
    assert_eq!(entry.capacity, 64);

    println!("K2K route entry verified");
}

#[test]
fn test_k2k_buffers_creation() {
    require_cuda!();
    println!("=== K2K Buffers Creation Test ===");

    let device = CudaDevice::new(0).expect("Failed to create device");

    let buffers = CudaK2KBuffers::new(&device, 64, 256, MAX_K2K_ROUTES)
        .expect("Failed to create K2K buffers");

    assert_eq!(buffers.inbox_capacity(), 64);
    assert_eq!(buffers.msg_size(), 256);
    assert!(buffers.inbox_ptr() > 0, "Inbox pointer should be valid");
    assert!(
        buffers.routing_table_ptr() > 0,
        "Routing table pointer should be valid"
    );
    assert!(buffers.total_size() > 0, "Total size should be positive");

    // Read inbox header to verify initialization
    let header = buffers
        .read_inbox_header()
        .expect("Failed to read inbox header");
    assert_eq!(header.capacity, 64, "Inbox capacity mismatch");
    assert_eq!(header.msg_size, 256, "Inbox msg_size mismatch");
    assert_eq!(header.head, 0, "Inbox head should be 0");
    assert_eq!(header.tail, 0, "Inbox tail should be 0");

    println!("K2K buffers created and initialized correctly");
}

#[test]
fn test_k2k_buffers_invalid_capacity() {
    require_cuda!();
    println!("=== K2K Buffers Invalid Capacity Test ===");

    let device = CudaDevice::new(0).expect("Failed to create device");

    // Non-power-of-2 capacity should fail
    let result = CudaK2KBuffers::new(&device, 65, 256, MAX_K2K_ROUTES);
    assert!(result.is_err(), "Should fail with non-power-of-2 capacity");

    println!("Invalid capacity handled correctly");
}

#[test]
fn test_k2k_route_setup() {
    require_cuda!();
    println!("=== K2K Route Setup Test ===");

    let device = CudaDevice::new(0).expect("Failed to create device");

    // Create two sets of K2K buffers (simulating two kernels)
    let buffers_a =
        CudaK2KBuffers::new(&device, 64, 256, MAX_K2K_ROUTES).expect("Failed to create buffers A");
    let buffers_b =
        CudaK2KBuffers::new(&device, 64, 256, MAX_K2K_ROUTES).expect("Failed to create buffers B");

    // Add route from A to B
    let target_kernel_id = 42u64;
    buffers_a
        .add_route(0, &buffers_b, target_kernel_id)
        .expect("Failed to add route");

    println!("K2K route A -> B set up successfully");

    // Add second route from A to a third buffer
    let buffers_c =
        CudaK2KBuffers::new(&device, 32, 128, MAX_K2K_ROUTES).expect("Failed to create buffers C");
    buffers_a
        .add_route(1, &buffers_c, 43)
        .expect("Failed to add second route");

    println!("K2K route A -> C set up successfully");
    println!("Multiple K2K routes verified");
}

#[test]
fn test_k2k_connection_manager() {
    require_cuda!();
    println!("=== K2K Connection Manager Test ===");

    let device = CudaDevice::new(0).expect("Failed to create device");
    let mut manager = K2KConnectionManager::new();

    // Create and register buffers for two kernels
    let buffers_1 =
        CudaK2KBuffers::new(&device, 64, 256, MAX_K2K_ROUTES).expect("Failed to create buffers 1");
    let buffers_2 =
        CudaK2KBuffers::new(&device, 64, 256, MAX_K2K_ROUTES).expect("Failed to create buffers 2");

    manager.register(1, buffers_1);
    manager.register(2, buffers_2);

    assert!(manager.get(1).is_some(), "Kernel 1 should be registered");
    assert!(manager.get(2).is_some(), "Kernel 2 should be registered");
    assert!(
        manager.get(3).is_none(),
        "Kernel 3 should not be registered"
    );

    // Connect kernel 1 -> kernel 2
    manager.connect(1, 2).expect("Failed to connect 1 -> 2");

    let connections = manager.connections();
    assert_eq!(connections.len(), 1, "Should have 1 connection");
    assert_eq!(connections[0], (1, 2, 0), "Connection should be (1, 2, 0)");

    println!("K2K connection manager verified");
}

#[test]
fn test_k2k_connection_max_routes() {
    require_cuda!();
    println!("=== K2K Connection Max Routes Test ===");

    let device = CudaDevice::new(0).expect("Failed to create device");
    let mut manager = K2KConnectionManager::new();

    // Register source kernel
    let source_buffers =
        CudaK2KBuffers::new(&device, 64, 256, MAX_K2K_ROUTES).expect("Failed to create source");
    manager.register(0, source_buffers);

    // Register MAX_K2K_ROUTES + 1 target kernels
    for i in 1..=(MAX_K2K_ROUTES + 1) {
        let target_buffers = CudaK2KBuffers::new(&device, 64, 256, MAX_K2K_ROUTES)
            .expect(&format!("Failed to create target {}", i));
        manager.register(i as u64, target_buffers);
    }

    // Connect source to MAX_K2K_ROUTES targets (should succeed)
    for i in 1..=MAX_K2K_ROUTES {
        manager
            .connect(0, i as u64)
            .expect(&format!("Failed to connect 0 -> {}", i));
    }

    // Try to connect one more (should fail)
    let result = manager.connect(0, (MAX_K2K_ROUTES + 1) as u64);
    assert!(result.is_err(), "Should fail when exceeding max routes");

    println!("K2K max routes limit enforced correctly");
}

// =============================================================================
// Runtime Integration Tests
// =============================================================================

#[tokio::test]
async fn test_runtime_k2k_enabled() {
    require_cuda!();
    println!("=== Runtime K2K Enabled Test ===");

    let runtime = CudaRuntime::with_config(0, true)
        .await
        .expect("Failed to create runtime");

    assert!(runtime.is_k2k_enabled(), "K2K should be enabled");
    assert!(runtime.k2k_broker().is_some(), "K2K broker should exist");

    runtime.shutdown().await.expect("Failed to shutdown");
    println!("Runtime with K2K enabled verified");
}

#[tokio::test]
async fn test_runtime_k2k_disabled() {
    require_cuda!();
    println!("=== Runtime K2K Disabled Test ===");

    let runtime = CudaRuntime::with_config(0, false)
        .await
        .expect("Failed to create runtime");

    assert!(!runtime.is_k2k_enabled(), "K2K should be disabled");
    assert!(
        runtime.k2k_broker().is_none(),
        "K2K broker should not exist"
    );

    runtime.shutdown().await.expect("Failed to shutdown");
    println!("Runtime with K2K disabled verified");
}

#[tokio::test]
async fn test_kernel_launch_with_k2k() {
    require_cuda!();
    println!("=== Kernel Launch with K2K Test ===");

    let runtime = CudaRuntime::with_config(0, true)
        .await
        .expect("Failed to create runtime");

    // Launch kernel with K2K enabled
    let options = LaunchOptions::default().with_k2k(true);

    let kernel = runtime
        .launch("k2k_test_kernel", options)
        .await
        .expect("Failed to launch kernel");

    println!("Kernel launched with K2K: {:?}", kernel.id());

    // Verify kernel has K2K buffers
    let cuda_kernel = runtime
        .get_cuda_kernel(kernel.id())
        .expect("Failed to get cuda kernel");
    assert!(
        cuda_kernel.k2k_buffers().is_some(),
        "Kernel should have K2K buffers"
    );

    runtime.shutdown().await.expect("Failed to shutdown");
    println!("Kernel with K2K buffers verified");
}

#[tokio::test]
async fn test_kernel_launch_without_k2k() {
    require_cuda!();
    println!("=== Kernel Launch without K2K Test ===");

    let runtime = CudaRuntime::with_config(0, true)
        .await
        .expect("Failed to create runtime");

    // Launch kernel without K2K
    let options = LaunchOptions::default().with_k2k(false);

    let kernel = runtime
        .launch("no_k2k_kernel", options)
        .await
        .expect("Failed to launch kernel");

    // Verify kernel does NOT have K2K buffers
    let cuda_kernel = runtime
        .get_cuda_kernel(kernel.id())
        .expect("Failed to get cuda kernel");
    assert!(
        cuda_kernel.k2k_buffers().is_none(),
        "Kernel should NOT have K2K buffers"
    );

    runtime.shutdown().await.expect("Failed to shutdown");
    println!("Kernel without K2K buffers verified");
}

#[tokio::test]
async fn test_runtime_connect_k2k() {
    require_cuda!();
    println!("=== Runtime Connect K2K Test ===");

    let runtime = CudaRuntime::with_config(0, true)
        .await
        .expect("Failed to create runtime");

    // Launch two kernels with K2K
    let options = LaunchOptions::default().with_k2k(true);

    let kernel_a = runtime
        .launch("kernel_a", options.clone())
        .await
        .expect("Failed to launch kernel A");

    let kernel_b = runtime
        .launch("kernel_b", options)
        .await
        .expect("Failed to launch kernel B");

    // Connect A -> B
    runtime
        .connect_k2k(kernel_a.id(), kernel_b.id())
        .expect("Failed to connect K2K");

    println!(
        "K2K connection established: {} -> {}",
        kernel_a.id(),
        kernel_b.id()
    );

    runtime.shutdown().await.expect("Failed to shutdown");
    println!("Runtime K2K connection verified");
}

#[tokio::test]
async fn test_runtime_connect_k2k_without_buffers() {
    require_cuda!();
    println!("=== Runtime Connect K2K Without Buffers Test ===");

    let runtime = CudaRuntime::with_config(0, true)
        .await
        .expect("Failed to create runtime");

    // Launch kernels WITHOUT K2K
    let options = LaunchOptions::default().with_k2k(false);

    let kernel_a = runtime
        .launch("no_k2k_a", options.clone())
        .await
        .expect("Failed to launch kernel A");

    let kernel_b = runtime
        .launch("no_k2k_b", options)
        .await
        .expect("Failed to launch kernel B");

    // Try to connect - should fail
    let result = runtime.connect_k2k(kernel_a.id(), kernel_b.id());
    assert!(
        result.is_err(),
        "Should fail to connect kernels without K2K buffers"
    );

    runtime.shutdown().await.expect("Failed to shutdown");
    println!("K2K connection without buffers correctly rejected");
}

// =============================================================================
// Size and Layout Tests
// =============================================================================

#[test]
fn test_struct_sizes() {
    println!("=== Struct Sizes Test ===");

    use std::mem::size_of;

    // K2K structures
    println!("K2KInboxHeader: {} bytes", size_of::<K2KInboxHeader>());
    println!("K2KRouteEntry: {} bytes", size_of::<K2KRouteEntry>());

    // Kernel params
    println!("RingKernelParams: {} bytes", size_of::<RingKernelParams>());
    println!(
        "RingKernelK2KParams: {} bytes",
        size_of::<RingKernelK2KParams>()
    );

    // Control block
    println!("ControlBlock: {} bytes", size_of::<ControlBlock>());

    // Verify critical sizes
    assert_eq!(
        size_of::<ControlBlock>(),
        128,
        "ControlBlock should be 128 bytes"
    );
    assert_eq!(
        size_of::<K2KInboxHeader>(),
        64,
        "K2KInboxHeader should be 64 bytes"
    );
    assert_eq!(
        size_of::<K2KRouteEntry>(),
        48,
        "K2KRouteEntry should be 48 bytes"
    );

    println!("All struct sizes verified");
}
