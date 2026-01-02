//! Comprehensive tests for ringkernel-derive proc macros.
//!
//! Tests for:
//! - `#[derive(RingMessage)]` - Message serialization and field attributes
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
