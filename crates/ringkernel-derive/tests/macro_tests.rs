//! Comprehensive tests for ringkernel-derive proc macros.
//!
//! Tests for:
//! - `#[derive(RingMessage)]` - Message serialization and field attributes
//! - `#[derive(PersistentMessage)]` - Persistent GPU kernel dispatch
//! - `#[ring_kernel]` - Kernel registration and handler generation
//! - `#[derive(GpuType)]` - GPU-compatible type assertions

use ringkernel_core::message::{CorrelationId, MessageId, Priority, RingMessage};
use ringkernel_derive::{GpuType, RingMessage};

// ============================================================================
// RingMessage derive tests
// ============================================================================

/// Basic struct with no special field attributes
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
struct BasicMessage {
    value: u32,
    data: f32,
}

#[test]
fn test_basic_ring_message_derive() {
    let msg = BasicMessage {
        value: 42,
        data: 3.125, // Avoid using PI approximation
    };

    // message_type() should return a hash of the struct name
    let type_id = BasicMessage::message_type();
    assert_ne!(type_id, 0, "type_id should not be zero");

    // message_id() should return default since no #[message(id)] field
    let id = msg.message_id();
    assert_eq!(id.0, 0, "default message_id should be 0");

    // correlation_id() should return none
    let corr = msg.correlation_id();
    assert_eq!(corr, CorrelationId::none());

    // priority() should return Normal
    let prio = msg.priority();
    assert_eq!(prio, Priority::Normal);
}

/// Struct with explicit type_id
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
#[message(type_id = 12345)]
struct ExplicitTypeIdMessage {
    payload: Vec<u8>,
}

#[test]
fn test_explicit_type_id() {
    let type_id = ExplicitTypeIdMessage::message_type();
    assert_eq!(type_id, 12345, "type_id should match explicit value");
}

#[test]
fn test_auto_type_id_hash_differs() {
    // Different struct names should produce different type IDs
    let basic_id = BasicMessage::message_type();
    let explicit_id = ExplicitTypeIdMessage::message_type();

    // ExplicitTypeIdMessage has explicit type_id = 12345
    assert_eq!(explicit_id, 12345);

    // BasicMessage should have a hashed type_id
    assert_ne!(
        basic_id, explicit_id,
        "different structs should have different type IDs"
    );
}

/// Struct with all field attributes
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
#[message(type_id = 100)]
struct FullyAnnotatedMessage {
    #[message(id)]
    id: MessageId,
    #[message(correlation)]
    correlation: CorrelationId,
    #[message(priority)]
    priority: Priority,
    data: String,
}

#[test]
fn test_field_attributes() {
    let msg = FullyAnnotatedMessage {
        id: MessageId::new(999),
        correlation: CorrelationId::new(888),
        priority: Priority::High,
        data: "test".to_string(),
    };

    assert_eq!(
        msg.message_id().0,
        999,
        "message_id should use annotated field"
    );
    assert_eq!(
        msg.correlation_id().0,
        888,
        "correlation_id should use annotated field"
    );
    assert_eq!(
        msg.priority(),
        Priority::High,
        "priority should use annotated field"
    );
}

#[test]
fn test_serialization_roundtrip() {
    let original = FullyAnnotatedMessage {
        id: MessageId::new(123),
        correlation: CorrelationId::new(456),
        priority: Priority::Critical,
        data: "hello world".to_string(),
    };

    // Serialize
    let bytes = original.serialize();
    assert!(!bytes.is_empty(), "serialized bytes should not be empty");

    // Deserialize
    let restored =
        FullyAnnotatedMessage::deserialize(&bytes).expect("deserialization should succeed");

    assert_eq!(restored.id.0, 123);
    assert_eq!(restored.correlation.0, 456);
    assert_eq!(restored.priority, Priority::Critical);
    assert_eq!(restored.data, "hello world");
}

/// Test with larger payload to verify buffer sizing
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
struct LargePayloadMessage {
    #[message(id)]
    id: MessageId,
    // 1KB of data - larger than the default 256-byte buffer
    data: Vec<u8>,
}

#[test]
fn test_large_payload_serialization() {
    let large_data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();

    let original = LargePayloadMessage {
        id: MessageId::new(42),
        data: large_data.clone(),
    };

    // This should work even with payloads larger than 256 bytes
    let bytes = original.serialize();
    assert!(
        !bytes.is_empty(),
        "serialization of large payload should succeed"
    );

    let restored =
        LargePayloadMessage::deserialize(&bytes).expect("deserialization should succeed");

    assert_eq!(restored.id.0, 42);
    assert_eq!(restored.data.len(), 1024);
    assert_eq!(restored.data, large_data);
}

/// Test size_hint
#[test]
fn test_size_hint() {
    let msg = BasicMessage {
        value: 0,
        data: 0.0,
    };

    let hint = msg.size_hint();
    // size_hint returns size_of::<Self>() which is 8 bytes for BasicMessage
    assert_eq!(hint, std::mem::size_of::<BasicMessage>());
}

// ============================================================================
// GpuType derive tests
// ============================================================================

/// A simple GPU-compatible type
#[derive(Debug, Clone, Copy, GpuType)]
#[repr(C)]
struct GpuVector3 {
    x: f32,
    y: f32,
    z: f32,
}

#[test]
fn test_gpu_type_copy_assertion() {
    // This compiles only if GpuVector3 implements Copy
    fn assert_copy<T: Copy>() {}
    assert_copy::<GpuVector3>();
}

#[test]
fn test_gpu_type_pod_zeroable() {
    use bytemuck::{Pod, Zeroable};

    // These compile only if the traits are implemented
    fn assert_pod<T: Pod>() {}
    fn assert_zeroable<T: Zeroable>() {}

    assert_pod::<GpuVector3>();
    assert_zeroable::<GpuVector3>();

    // Test zeroable behavior
    let zeroed: GpuVector3 = bytemuck::Zeroable::zeroed();
    assert_eq!(zeroed.x, 0.0);
    assert_eq!(zeroed.y, 0.0);
    assert_eq!(zeroed.z, 0.0);
}

#[test]
fn test_gpu_type_byte_cast() {
    use bytemuck::bytes_of;

    let vec = GpuVector3 {
        x: 1.0,
        y: 2.0,
        z: 3.0,
    };

    let bytes = bytes_of(&vec);
    assert_eq!(bytes.len(), std::mem::size_of::<GpuVector3>());
    assert_eq!(bytes.len(), 12); // 3 * 4 bytes
}

/// GPU matrix type
#[derive(Debug, Clone, Copy, GpuType)]
#[repr(C)]
struct GpuMatrix4x4 {
    data: [[f32; 4]; 4],
}

#[test]
fn test_gpu_type_matrix() {
    use bytemuck::{Pod, Zeroable};

    fn assert_pod<T: Pod>() {}
    fn assert_zeroable<T: Zeroable>() {}

    assert_pod::<GpuMatrix4x4>();
    assert_zeroable::<GpuMatrix4x4>();

    assert_eq!(std::mem::size_of::<GpuMatrix4x4>(), 64); // 4 * 4 * 4 bytes
}

// ============================================================================
// ring_kernel macro tests
// ============================================================================

// Note: Full ring_kernel tests require the runtime infrastructure.
// These tests verify kernel registration collection works.

mod ring_kernel_tests {
    use ringkernel_core::__private::registered_kernels;

    #[test]
    fn test_kernel_registration_collection() {
        // After using #[ring_kernel], the registration should be collected
        let kernels: Vec<_> = registered_kernels().collect();
        // Without actual kernel definitions using the macro, this will be empty
        // This test just verifies the collection mechanism works
        println!("Registered kernels: {}", kernels.len());
    }
}

// ============================================================================
// Edge case tests
// ============================================================================

// Note: Empty structs (unit-like) are not supported by RingMessage
// The macro requires named fields

/// Struct with only ID field
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
struct IdOnlyMessage {
    #[message(id)]
    id: MessageId,
}

#[test]
fn test_id_only_message() {
    let msg = IdOnlyMessage {
        id: MessageId::new(777),
    };

    assert_eq!(msg.message_id().0, 777);
    assert_eq!(msg.correlation_id(), CorrelationId::none());
    assert_eq!(msg.priority(), Priority::Normal);
}

/// Test that different structs with same content have different type IDs
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
struct MessageA {
    value: u32,
}

#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
struct MessageB {
    value: u32,
}

#[test]
fn test_different_structs_different_type_ids() {
    let type_a = MessageA::message_type();
    let type_b = MessageB::message_type();

    assert_ne!(
        type_a, type_b,
        "Different struct names should produce different type IDs"
    );
}

// ============================================================================
// gpu_kernel macro tests
// ============================================================================

mod gpu_kernel_tests {
    use ringkernel_core::__private::{
        backend_supports_capability, find_gpu_kernel, registered_gpu_kernels, select_backend,
    };

    #[test]
    fn test_gpu_kernel_registration_collection() {
        // Collect all registered GPU kernels
        let kernels: Vec<_> = registered_gpu_kernels().collect();
        println!("Registered GPU kernels: {}", kernels.len());
    }

    #[test]
    fn test_backend_supports_capability() {
        // CUDA supports everything
        assert!(backend_supports_capability("cuda", "f64"));
        assert!(backend_supports_capability("cuda", "atomic64"));
        assert!(backend_supports_capability("cuda", "cooperative_groups"));

        // Metal limitations
        assert!(!backend_supports_capability("metal", "f64"));
        assert!(!backend_supports_capability("metal", "cooperative_groups"));
        assert!(backend_supports_capability("metal", "subgroups"));

        // WebGPU limitations
        assert!(!backend_supports_capability("wgpu", "f64"));
        assert!(!backend_supports_capability("wgpu", "i64"));
        assert!(!backend_supports_capability("wgpu", "atomic64"));
        assert!(backend_supports_capability("wgpu", "shared_memory"));

        // CPU supports everything (emulation)
        assert!(backend_supports_capability("cpu", "f64"));
        assert!(backend_supports_capability("cpu", "cooperative_groups"));
    }

    #[test]
    fn test_select_backend() {
        let fallback_order = &["cuda", "metal", "wgpu", "cpu"];
        let available = &["cuda", "metal", "wgpu", "cpu"];

        // No required capabilities - should select first available
        let selected = select_backend(fallback_order, &[], available);
        assert_eq!(selected, Some("cuda"));

        // Require f64 - should skip metal/wgpu
        let selected = select_backend(fallback_order, &["f64"], available);
        assert_eq!(selected, Some("cuda"));

        // Only wgpu available, require f64 - should fall through to cpu
        let selected = select_backend(fallback_order, &["f64"], &["wgpu", "cpu"]);
        assert_eq!(selected, Some("cpu"));

        // CUDA supports all capabilities including unknown ones (via catch-all)
        let selected = select_backend(fallback_order, &["quantum_compute"], available);
        assert_eq!(selected, Some("cuda"));
    }

    #[test]
    fn test_select_backend_respects_fallback_order() {
        // If metal is preferred but doesn't support f64
        let fallback_order = &["metal", "wgpu", "cuda", "cpu"];
        let available = &["cuda", "metal", "wgpu", "cpu"];

        // Should skip metal and wgpu due to f64 requirement
        let selected = select_backend(fallback_order, &["f64"], available);
        assert_eq!(selected, Some("cuda"));
    }

    #[test]
    fn test_find_nonexistent_kernel() {
        let kernel = find_gpu_kernel("nonexistent_kernel_xyz");
        assert!(kernel.is_none());
    }
}

// ============================================================================
// ControlBlockState derive tests (FR-4)
// ============================================================================

mod control_block_state_tests {
    use ringkernel_core::control::ControlBlock;
    use ringkernel_core::state::{ControlBlockStateHelper, EmbeddedState, GpuState};
    use ringkernel_derive::ControlBlockState;

    /// Test state that fits in 24 bytes (exactly)
    #[derive(ControlBlockState, Default, Clone, Copy, Debug, PartialEq)]
    #[repr(C, align(8))]
    #[state(version = 1)]
    struct OrderBookState {
        best_bid: u64,    // 8 bytes
        best_ask: u64,    // 8 bytes
        order_count: u32, // 4 bytes
        _pad: u32,        // 4 bytes
    } // Total: 24 bytes

    #[test]
    fn test_control_block_state_size() {
        assert_eq!(std::mem::size_of::<OrderBookState>(), 24);
    }

    #[test]
    fn test_embedded_state_trait() {
        // Verify EmbeddedState is implemented
        fn assert_embedded<T: EmbeddedState>() {}
        assert_embedded::<OrderBookState>();

        // Check version
        assert_eq!(OrderBookState::VERSION, 1);
        assert!(OrderBookState::is_embedded());
    }

    #[test]
    fn test_gpu_state_trait() {
        // EmbeddedState implies GpuState
        fn assert_gpu_state<T: GpuState>() {}
        assert_gpu_state::<OrderBookState>();

        let state = OrderBookState {
            best_bid: 100,
            best_ask: 101,
            order_count: 42,
            _pad: 0,
        };

        // Test serialization
        let bytes = state.to_control_block_bytes();
        assert_eq!(bytes.len(), 24);

        // Test deserialization
        let restored = OrderBookState::from_control_block_bytes(&bytes).unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_write_read_embedded_state() {
        let mut block = ControlBlock::new();
        let state = OrderBookState {
            best_bid: 0x1234567890ABCDEF,
            best_ask: 0xFEDCBA0987654321,
            order_count: 999,
            _pad: 0,
        };

        ControlBlockStateHelper::write_embedded(&mut block, &state).unwrap();
        let restored: OrderBookState = ControlBlockStateHelper::read_embedded(&block).unwrap();

        assert_eq!(state, restored);
    }

    /// Smaller state (8 bytes)
    #[derive(ControlBlockState, Default, Clone, Copy, Debug, PartialEq)]
    #[repr(C)]
    struct SmallState {
        counter: u64,
    }

    #[test]
    fn test_small_state() {
        assert!(std::mem::size_of::<SmallState>() <= 24);

        let mut block = ControlBlock::new();
        let state = SmallState { counter: 42 };

        ControlBlockStateHelper::write_embedded(&mut block, &state).unwrap();
        let restored: SmallState = ControlBlockStateHelper::read_embedded(&block).unwrap();

        assert_eq!(state, restored);
    }

    /// State with custom version
    #[derive(ControlBlockState, Default, Clone, Copy, Debug)]
    #[repr(C)]
    #[state(version = 2)]
    struct VersionedState {
        value: u64,
    }

    #[test]
    fn test_versioned_state() {
        assert_eq!(VersionedState::VERSION, 2);
        assert_eq!(VersionedState::state_version(), 2);
    }

    #[test]
    fn test_pod_zeroable_impl() {
        use bytemuck::{Pod, Zeroable};

        fn assert_pod<T: Pod>() {}
        fn assert_zeroable<T: Zeroable>() {}

        assert_pod::<OrderBookState>();
        assert_zeroable::<OrderBookState>();
        assert_pod::<SmallState>();
        assert_zeroable::<SmallState>();

        // Test zeroable
        let zeroed: OrderBookState = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.best_bid, 0);
        assert_eq!(zeroed.best_ask, 0);
        assert_eq!(zeroed.order_count, 0);
    }
}

// ============================================================================
// PersistentMessage derive tests
// ============================================================================

mod persistent_message_tests {
    use ringkernel_core::message::RingMessage;
    use ringkernel_core::persistent_message::{PersistentMessage, MAX_INLINE_PAYLOAD_SIZE};
    use ringkernel_derive::{PersistentMessage, RingMessage};

    /// Small message that fits in inline payload (16 bytes)
    #[derive(
        RingMessage,
        PersistentMessage,
        Clone,
        Copy,
        Debug,
        PartialEq,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    #[archive(check_bytes)]
    #[repr(C)]
    #[message(type_id = 1001)]
    #[persistent_message(handler_id = 1, requires_response = true)]
    struct FraudCheckRequest {
        transaction_id: u64,
        amount: f32,
        account_id: u32,
    }

    #[test]
    fn test_handler_id() {
        assert_eq!(FraudCheckRequest::handler_id(), 1);
    }

    #[test]
    fn test_requires_response() {
        assert!(FraudCheckRequest::requires_response());
    }

    #[test]
    fn test_payload_size() {
        let expected_size = std::mem::size_of::<FraudCheckRequest>();
        assert_eq!(FraudCheckRequest::payload_size(), expected_size);
        assert_eq!(expected_size, 16); // 8 + 4 + 4
    }

    #[test]
    fn test_can_inline() {
        assert!(FraudCheckRequest::can_inline());
        assert!(FraudCheckRequest::payload_size() <= MAX_INLINE_PAYLOAD_SIZE);
    }

    #[test]
    fn test_inline_payload_roundtrip() {
        let original = FraudCheckRequest {
            transaction_id: 0x123456789ABCDEF0,
            amount: 99.99,
            account_id: 42,
        };

        // Serialize to inline payload
        let payload = original
            .to_inline_payload()
            .expect("should serialize to inline");

        assert_eq!(payload.len(), MAX_INLINE_PAYLOAD_SIZE);

        // Deserialize from inline payload
        let restored =
            FraudCheckRequest::from_inline_payload(&payload).expect("should deserialize");

        assert_eq!(restored.transaction_id, original.transaction_id);
        assert_eq!(restored.amount, original.amount);
        assert_eq!(restored.account_id, original.account_id);
    }

    /// Message without response
    #[derive(
        RingMessage,
        PersistentMessage,
        Clone,
        Copy,
        Debug,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    #[archive(check_bytes)]
    #[repr(C)]
    #[message(type_id = 1002)]
    #[persistent_message(handler_id = 2)]
    struct AuditEvent {
        event_type: u32,
        timestamp: u64,
    }

    #[test]
    fn test_no_response() {
        assert_eq!(AuditEvent::handler_id(), 2);
        assert!(!AuditEvent::requires_response()); // Default is false
    }

    /// Exactly 32 bytes (maximum inline size)
    #[derive(
        RingMessage,
        PersistentMessage,
        Clone,
        Copy,
        Debug,
        PartialEq,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    #[archive(check_bytes)]
    #[repr(C)]
    #[message(type_id = 1003)]
    #[persistent_message(handler_id = 3)]
    struct MaxSizeMessage {
        data: [u64; 4], // 32 bytes exactly
    }

    #[test]
    fn test_max_size_inline() {
        assert_eq!(std::mem::size_of::<MaxSizeMessage>(), 32);
        assert!(MaxSizeMessage::can_inline());

        let original = MaxSizeMessage { data: [1, 2, 3, 4] };

        let payload = original
            .to_inline_payload()
            .expect("should fit in inline payload");

        let restored = MaxSizeMessage::from_inline_payload(&payload).expect("should deserialize");

        assert_eq!(restored, original);
    }

    /// Too large for inline (40 bytes)
    #[derive(
        RingMessage,
        PersistentMessage,
        Clone,
        Copy,
        Debug,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    #[archive(check_bytes)]
    #[repr(C)]
    #[message(type_id = 1004)]
    #[persistent_message(handler_id = 4)]
    struct LargeMessage {
        data: [u64; 5], // 40 bytes - too big for inline
    }

    #[test]
    fn test_too_large_for_inline() {
        assert_eq!(std::mem::size_of::<LargeMessage>(), 40);
        assert!(!LargeMessage::can_inline());

        let msg = LargeMessage {
            data: [1, 2, 3, 4, 5],
        };

        // to_inline_payload should return None for oversized messages
        let payload = msg.to_inline_payload();
        assert!(payload.is_none(), "should not serialize oversized message");
    }

    #[test]
    fn test_from_inline_payload_too_small() {
        let small_payload = [0u8; 8]; // Only 8 bytes, need 16 for FraudCheckRequest

        let result = FraudCheckRequest::from_inline_payload(&small_payload);
        assert!(result.is_err());
    }

    /// Test that RingMessage type_id and PersistentMessage handler_id are independent
    #[test]
    fn test_type_id_and_handler_id_independence() {
        // type_id from RingMessage
        assert_eq!(FraudCheckRequest::message_type(), 1001);

        // handler_id from PersistentMessage
        assert_eq!(FraudCheckRequest::handler_id(), 1);

        // They can be different
        assert_ne!(
            FraudCheckRequest::message_type() as u32,
            FraudCheckRequest::handler_id()
        );
    }

    /// Test handler ID range (0-255)
    #[derive(
        RingMessage,
        PersistentMessage,
        Clone,
        Copy,
        Debug,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    #[archive(check_bytes)]
    #[repr(C)]
    #[message(type_id = 1005)]
    #[persistent_message(handler_id = 255)]
    struct MaxHandlerIdMessage {
        value: u32,
    }

    #[test]
    fn test_max_handler_id() {
        assert_eq!(MaxHandlerIdMessage::handler_id(), 255);
    }

    #[derive(
        RingMessage,
        PersistentMessage,
        Clone,
        Copy,
        Debug,
        rkyv::Archive,
        rkyv::Serialize,
        rkyv::Deserialize,
    )]
    #[archive(check_bytes)]
    #[repr(C)]
    #[message(type_id = 1006)]
    #[persistent_message(handler_id = 0)]
    struct MinHandlerIdMessage {
        value: u32,
    }

    #[test]
    fn test_min_handler_id() {
        assert_eq!(MinHandlerIdMessage::handler_id(), 0);
    }

    /// Test different handler_ids for different message types
    #[test]
    fn test_different_handler_ids() {
        assert_eq!(FraudCheckRequest::handler_id(), 1);
        assert_eq!(AuditEvent::handler_id(), 2);
        assert_eq!(MaxSizeMessage::handler_id(), 3);
        assert_eq!(LargeMessage::handler_id(), 4);
    }
}
