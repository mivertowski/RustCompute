//! Integration tests for GPU actor code generation.
//!
//! These tests verify that the transpiler generates complete, working GPU actor
//! code that integrates properly with the ringkernel-cuda runtime structures:
//! - MessageEnvelope with 256-byte headers
//! - K2K routing with envelope format
//! - HLC timestamp propagation
//! - Proper response serialization

use ringkernel_cuda_codegen::{ring_kernel::RingKernelConfig, transpile_ring_kernel};
use syn::parse_quote;

/// Test that envelope format generates MessageHeader struct.
#[test]
fn test_envelope_format_generates_message_header() {
    let handler: syn::ItemFn = parse_quote! {
        fn handle(ctx: &RingContext, msg: &Request) -> Response {
            Response { value: msg.value * 2.0 }
        }
    };

    let config = RingKernelConfig::new("envelope_test")
        .with_envelope_format(true)
        .with_hlc(true)
        .with_kernel_id(42)
        .with_hlc_node_id(1);

    let cuda_code = transpile_ring_kernel(&handler, &config).expect("Transpilation should succeed");

    // Debug output
    println!("Generated code:\n{}", cuda_code);

    // Verify MessageHeader struct is included
    assert!(
        cuda_code.contains("MessageHeader"),
        "Should generate MessageHeader struct"
    );
    assert!(
        cuda_code.contains("MESSAGE_MAGIC"),
        "Should include message magic constant"
    );
    assert!(
        cuda_code.contains("MESSAGE_HEADER_SIZE"),
        "Should define header size"
    );
    assert!(
        cuda_code.contains("HlcTimestamp"),
        "Header should include HlcTimestamp"
    );
    assert!(
        cuda_code.contains("correlation_id"),
        "Header should include correlation_id"
    );
}

/// Test that envelope format generates payload access.
#[test]
fn test_envelope_format_generates_payload_access() {
    let handler: syn::ItemFn = parse_quote! {
        fn process(ctx: &RingContext, msg: &MyMessage) -> MyResponse {
            MyResponse { result: msg.data + 1.0 }
        }
    };

    let config = RingKernelConfig::new("payload_test").with_envelope_format(true);

    let cuda_code = transpile_ring_kernel(&handler, &config).expect("Transpilation should succeed");

    // Verify message envelope parsing
    assert!(
        cuda_code.contains("message_get_header(envelope_ptr)"),
        "Should parse message header"
    );
    assert!(
        cuda_code.contains("message_get_payload(envelope_ptr)"),
        "Should get payload pointer"
    );
    assert!(
        cuda_code.contains("message_header_validate"),
        "Should validate message header"
    );
}

/// Test that envelope format generates HLC updates from incoming messages.
#[test]
fn test_envelope_hlc_update_from_message() {
    let handler: syn::ItemFn = parse_quote! {
        fn handle(ctx: &RingContext, msg: &Request) -> Response {
            Response { value: 1.0 }
        }
    };

    let config = RingKernelConfig::new("hlc_test")
        .with_envelope_format(true)
        .with_hlc(true);

    let cuda_code = transpile_ring_kernel(&handler, &config).expect("Transpilation should succeed");

    // Verify HLC update from message timestamp
    assert!(
        cuda_code.contains("Update HLC from message timestamp"),
        "Should update HLC from incoming message"
    );
    assert!(
        cuda_code.contains("msg_header->timestamp.physical"),
        "Should read message timestamp"
    );
    assert!(
        cuda_code.contains("hlc_physical = msg_header->timestamp.physical"),
        "Should update physical clock"
    );
}

/// Test that envelope format generates proper response headers.
#[test]
fn test_envelope_response_serialization() {
    let handler: syn::ItemFn = parse_quote! {
        fn handle(ctx: &RingContext, msg: &Request) -> Response {
            Response { value: msg.value * 2.0 }
        }
    };

    let config = RingKernelConfig::new("response_test")
        .with_envelope_format(true)
        .with_hlc(true)
        .with_kernel_id(123)
        .with_hlc_node_id(5);

    let cuda_code = transpile_ring_kernel(&handler, &config).expect("Transpilation should succeed");

    // Verify response envelope generation
    assert!(
        cuda_code.contains("Response serialization (envelope format)"),
        "Should generate envelope response serialization"
    );
    assert!(
        cuda_code.contains("message_create_response_header"),
        "Should create response header"
    );
    assert!(
        cuda_code.contains("resp_header, msg_header, KERNEL_ID"),
        "Should pass correlation data"
    );
    assert!(
        cuda_code.contains("memcpy(resp_envelope + MESSAGE_HEADER_SIZE"),
        "Should copy payload after header"
    );
    assert!(
        cuda_code.contains("__threadfence()"),
        "Should fence after write"
    );
}

/// Test kernel identity constants are generated.
#[test]
fn test_kernel_identity_constants() {
    let handler: syn::ItemFn = parse_quote! {
        fn handle(ctx: &RingContext, msg: &Msg) -> Resp { Resp {} }
    };

    let config = RingKernelConfig::new("identity_test")
        .with_kernel_id(12345)
        .with_hlc_node_id(99)
        .with_envelope_format(true);

    let cuda_code = transpile_ring_kernel(&handler, &config).expect("Transpilation should succeed");

    // Verify kernel identity constants
    assert!(
        cuda_code.contains("KERNEL_ID = 12345ULL"),
        "Should define kernel ID"
    );
    assert!(
        cuda_code.contains("HLC_NODE_ID = 99ULL"),
        "Should define HLC node ID"
    );
}

/// Test K2K with envelope format generates envelope-aware functions.
#[test]
fn test_k2k_envelope_format() {
    let handler: syn::ItemFn = parse_quote! {
        fn handle(ctx: &RingContext, msg: &Request) -> Response {
            Response { value: 1.0 }
        }
    };

    let config = RingKernelConfig::new("k2k_envelope_test")
        .with_envelope_format(true)
        .with_k2k(true)
        .with_hlc(true);

    let cuda_code = transpile_ring_kernel(&handler, &config).expect("Transpilation should succeed");

    // Verify K2K envelope functions
    assert!(
        cuda_code.contains("k2k_send_envelope"),
        "Should generate envelope-aware K2K send"
    );
    assert!(
        cuda_code.contains("k2k_try_recv_envelope"),
        "Should generate envelope-aware K2K receive"
    );
    assert!(
        cuda_code.contains("k2k_peek_envelope"),
        "Should generate envelope-aware K2K peek"
    );
    assert!(
        cuda_code.contains("K2KRoutingTable"),
        "Should include K2K routing table"
    );
    assert!(
        cuda_code.contains("K2KInboxHeader"),
        "Should include K2K inbox header"
    );
}

/// Test message size constants with envelope format.
#[test]
fn test_envelope_size_constants() {
    let handler: syn::ItemFn = parse_quote! {
        fn handle(ctx: &RingContext, msg: &Msg) -> Resp { Resp {} }
    };

    let config = RingKernelConfig::new("size_test")
        .with_message_sizes(64, 32) // 64-byte payload, 32-byte response
        .with_envelope_format(true);

    let cuda_code = transpile_ring_kernel(&handler, &config).expect("Transpilation should succeed");

    // Verify size constants include header
    assert!(
        cuda_code.contains("PAYLOAD_SIZE = 64"),
        "Should define payload size"
    );
    assert!(
        cuda_code.contains("MSG_SIZE = MESSAGE_HEADER_SIZE + PAYLOAD_SIZE"),
        "MSG_SIZE should include header"
    );
    assert!(
        cuda_code.contains("RESP_PAYLOAD_SIZE = 32"),
        "Should define response payload size"
    );
    assert!(
        cuda_code.contains("RESP_SIZE = MESSAGE_HEADER_SIZE + RESP_PAYLOAD_SIZE"),
        "RESP_SIZE should include header"
    );
}

/// Test raw format (non-envelope) still works.
#[test]
fn test_raw_format_backwards_compatible() {
    let handler: syn::ItemFn = parse_quote! {
        fn handle(ctx: &RingContext, msg: &Msg) -> Resp { Resp {} }
    };

    let config = RingKernelConfig::new("raw_test")
        .with_envelope_format(false)
        .with_message_sizes(64, 32);

    let cuda_code = transpile_ring_kernel(&handler, &config).expect("Transpilation should succeed");

    // Verify raw format doesn't use envelope structures
    assert!(
        !cuda_code.contains("MessageHeader"),
        "Raw format should not include MessageHeader"
    );
    assert!(
        cuda_code.contains("MSG_SIZE = 64"),
        "Raw format should use direct message size"
    );
    assert!(
        cuda_code.contains("msg_ptr = envelope_ptr"),
        "Raw format should use buffer directly"
    );
}

/// Test full kernel structure with all features enabled.
#[test]
fn test_full_kernel_with_all_features() {
    let handler: syn::ItemFn = parse_quote! {
        fn process_transaction(ctx: &RingContext, msg: &TransactionRequest) -> TransactionResponse {
            let tid = ctx.global_thread_id();
            ctx.sync_threads();
            TransactionResponse {
                success: true,
                processed_by: tid as u64
            }
        }
    };

    let config = RingKernelConfig::new("full_featured")
        .with_block_size(256)
        .with_queue_capacity(2048)
        .with_envelope_format(true)
        .with_k2k(true)
        .with_hlc(true)
        .with_kernel_id(1000)
        .with_hlc_node_id(42);

    let cuda_code = transpile_ring_kernel(&handler, &config).expect("Transpilation should succeed");

    // Verify kernel structure
    assert!(
        cuda_code.contains("extern \"C\" __global__ void ring_kernel_full_featured"),
        "Should generate correct kernel name"
    );
    assert!(
        cuda_code.contains("ControlBlock* __restrict__ control"),
        "Should have control block parameter"
    );
    assert!(
        cuda_code.contains("K2KRoutingTable* __restrict__ k2k_routes"),
        "Should have K2K routes parameter"
    );

    // Verify all struct definitions
    assert!(
        cuda_code.contains("struct __align__(128) ControlBlock"),
        "Should include ControlBlock"
    );
    assert!(
        cuda_code.contains("struct HlcState"),
        "Should include HlcState"
    );
    assert!(
        cuda_code.contains("struct __align__(64) MessageHeader"),
        "Should include MessageHeader"
    );
    assert!(
        cuda_code.contains("struct K2KRoute"),
        "Should include K2KRoute"
    );

    // Verify message loop structure
    assert!(
        cuda_code.contains("while (true)"),
        "Should have persistent loop"
    );
    assert!(
        cuda_code.contains("should_terminate"),
        "Should check termination"
    );
    assert!(cuda_code.contains("is_active"), "Should check active state");
    assert!(
        cuda_code.contains("message_header_validate"),
        "Should validate messages"
    );

    // Verify handler code is included
    assert!(
        cuda_code.contains("USER HANDLER CODE"),
        "Should include handler section"
    );
    assert!(
        cuda_code.contains("__syncthreads()"),
        "Should transpile sync_threads"
    );

    println!("Generated full-featured kernel:\n{}", cuda_code);
}

/// Test that generated code has all necessary helper functions.
#[test]
fn test_helper_functions_generated() {
    let handler: syn::ItemFn = parse_quote! {
        fn handle(ctx: &RingContext, msg: &Msg) -> Resp { Resp {} }
    };

    let config = RingKernelConfig::new("helpers_test")
        .with_envelope_format(true)
        .with_k2k(true)
        .with_hlc(true);

    let cuda_code = transpile_ring_kernel(&handler, &config).expect("Transpilation should succeed");

    // Message helpers
    assert!(
        cuda_code.contains("__device__ inline int message_header_validate"),
        "Should have header validation function"
    );
    assert!(
        cuda_code.contains("__device__ inline unsigned char* message_get_payload"),
        "Should have payload getter"
    );
    assert!(
        cuda_code.contains("__device__ inline MessageHeader* message_get_header"),
        "Should have header getter"
    );
    assert!(
        cuda_code.contains("__device__ inline void message_create_response_header"),
        "Should have response header creator"
    );

    // K2K helpers
    assert!(
        cuda_code.contains("__device__ inline int k2k_send"),
        "Should have k2k_send"
    );
    assert!(
        cuda_code.contains("__device__ inline int k2k_has_message"),
        "Should have k2k_has_message"
    );
    assert!(
        cuda_code.contains("__device__ inline unsigned int k2k_pending_count"),
        "Should have k2k_pending_count"
    );
}

/// Test that correlation ID is preserved in responses.
#[test]
fn test_correlation_id_preserved() {
    let handler: syn::ItemFn = parse_quote! {
        fn handle(ctx: &RingContext, msg: &Request) -> Response {
            Response { value: 1.0 }
        }
    };

    let config = RingKernelConfig::new("correlation_test").with_envelope_format(true);

    let cuda_code = transpile_ring_kernel(&handler, &config).expect("Transpilation should succeed");

    // Verify correlation ID handling in response header
    assert!(
        cuda_code.contains("correlation_id = request->correlation_id"),
        "Response should preserve correlation ID"
    );
    assert!(
        cuda_code.contains("dest_kernel = request->source_kernel"),
        "Response should route back to sender"
    );
}

/// Test message type extraction for code generation.
#[test]
fn test_message_type_in_generated_code() {
    let handler: syn::ItemFn = parse_quote! {
        fn handle(ctx: &RingContext, msg: &MyCustomMessage) -> MyCustomResponse {
            MyCustomResponse { result: msg.input * 2.0 }
        }
    };

    let config = RingKernelConfig::new("type_test").with_envelope_format(true);

    let cuda_code = transpile_ring_kernel(&handler, &config).expect("Transpilation should succeed");

    // Verify message type is used correctly
    assert!(
        cuda_code.contains("MyCustomMessage* msg"),
        "Should cast to correct message type"
    );
    assert!(
        cuda_code.contains("(MyCustomMessage*)msg_ptr"),
        "Should cast payload to message type"
    );
}
