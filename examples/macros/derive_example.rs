//! # Derive Macros Example
//!
//! This example demonstrates the use of RingKernel's derive macros:
//!
//! - `#[derive(RingMessage)]` - For message types with automatic serialization
//! - `#[derive(GpuType)]` - For GPU-compatible data types
//!
//! ## Run this example:
//! ```bash
//! cargo run -p ringkernel --example derive_example
//! ```

use ringkernel::prelude::*;
use ringkernel_derive::{GpuType, RingMessage};

// ============================================================================
// RingMessage Examples
// ============================================================================

/// A simple request message with explicit type ID.
///
/// The `#[message(type_id = 1)]` attribute sets a stable type discriminator
/// for message routing.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
#[message(type_id = 1)]
struct AddRequest {
    /// Message ID field - marked with `#[message(id)]`
    #[message(id)]
    id: MessageId,
    /// First operand
    a: f32,
    /// Second operand
    b: f32,
}

/// Response message with correlation tracking.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
#[message(type_id = 2)]
struct AddResponse {
    #[message(id)]
    id: MessageId,
    /// Correlation ID links response to request
    #[message(correlation)]
    correlation: CorrelationId,
    /// The result of the addition
    result: f32,
}

/// A high-priority message with all field attributes.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
#[message(type_id = 100)]
struct PriorityMessage {
    #[message(id)]
    id: MessageId,
    #[message(correlation)]
    correlation: CorrelationId,
    #[message(priority)]
    priority: Priority,
    /// Message payload
    data: Vec<u8>,
}

/// Message without explicit type_id - uses auto-generated hash of struct name.
#[derive(Debug, Clone, RingMessage, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[archive(check_bytes)]
struct AutoTypedMessage {
    #[message(id)]
    id: MessageId,
    content: String,
}

// ============================================================================
// GpuType Examples
// ============================================================================

/// A 3D vector suitable for GPU transfer.
///
/// `#[derive(GpuType)]` ensures the type is:
/// - Copy (no heap allocations)
/// - Pod (plain old data, safe for raw byte transfer)
/// - Zeroable (can be initialized with zeros)
#[derive(Debug, Clone, Copy, GpuType)]
#[repr(C)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

/// A 4x4 transformation matrix for GPU computations.
#[derive(Debug, Clone, Copy, GpuType)]
#[repr(C)]
struct Matrix4x4 {
    data: [[f32; 4]; 4],
}

/// A particle for physics simulations.
#[derive(Debug, Clone, Copy, GpuType)]
#[repr(C)]
struct Particle {
    position: [f32; 3],
    velocity: [f32; 3],
    mass: f32,
    lifetime: f32,
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== RingKernel Derive Macros Example ===\n");

    // ========== RingMessage Examples ==========
    println!("=== RingMessage Derive ===\n");

    // Create messages
    let request = AddRequest {
        id: MessageId::generate(),
        a: 3.14,
        b: 2.71,
    };

    println!("AddRequest:");
    println!("  Message Type ID: {}", AddRequest::message_type());
    println!("  Message ID: {}", request.message_id());
    println!("  Correlation: {:?}", request.correlation_id());
    println!("  Priority: {:?}", request.priority());

    // Serialize and deserialize
    let bytes = request.serialize();
    println!("  Serialized size: {} bytes", bytes.len());

    let restored = AddRequest::deserialize(&bytes).expect("deserialization should succeed");
    println!("  Restored: a={}, b={}", restored.a, restored.b);

    println!();

    // Priority message with all attributes
    let priority_msg = PriorityMessage {
        id: MessageId::generate(),
        correlation: CorrelationId::new(12345),
        priority: Priority::Critical,
        data: vec![1, 2, 3, 4, 5],
    };

    println!("PriorityMessage:");
    println!("  Message Type ID: {}", PriorityMessage::message_type());
    println!("  Correlation: {:?}", priority_msg.correlation_id());
    println!("  Priority: {:?}", priority_msg.priority());

    let bytes = priority_msg.serialize();
    println!("  Serialized size: {} bytes", bytes.len());

    println!();

    // Auto-typed message
    let auto_msg = AutoTypedMessage {
        id: MessageId::generate(),
        content: "Hello, RingKernel!".to_string(),
    };

    println!("AutoTypedMessage:");
    println!("  Message Type ID: {} (auto-generated hash)", AutoTypedMessage::message_type());
    println!("  Content: {}", auto_msg.content);

    let bytes = auto_msg.serialize();
    let restored = AutoTypedMessage::deserialize(&bytes).expect("deserialization should succeed");
    println!("  Round-trip content: {}", restored.content);

    println!();

    // ========== GpuType Examples ==========
    println!("=== GpuType Derive ===\n");

    let vec = Vec3 {
        x: 1.0,
        y: 2.0,
        z: 3.0,
    };

    println!("Vec3:");
    println!("  Size: {} bytes", std::mem::size_of::<Vec3>());
    println!("  Alignment: {} bytes", std::mem::align_of::<Vec3>());

    // Use bytemuck for safe byte casting
    let bytes = bytemuck::bytes_of(&vec);
    println!("  As bytes: {:?}", bytes);

    // Create from zeros
    let zeroed: Vec3 = bytemuck::Zeroable::zeroed();
    println!("  Zeroed: ({}, {}, {})", zeroed.x, zeroed.y, zeroed.z);

    println!();

    let _matrix = Matrix4x4 {
        data: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    println!("Matrix4x4:");
    println!("  Size: {} bytes", std::mem::size_of::<Matrix4x4>());
    println!("  Is identity: diagonal all 1s");

    println!();

    let particle = Particle {
        position: [0.0, 10.0, 0.0],
        velocity: [1.0, -0.5, 0.0],
        mass: 1.0,
        lifetime: 5.0,
    };

    println!("Particle:");
    println!("  Size: {} bytes", std::mem::size_of::<Particle>());
    println!("  Position: {:?}", particle.position);
    println!("  Velocity: {:?}", particle.velocity);
    println!("  Mass: {}, Lifetime: {}", particle.mass, particle.lifetime);

    // Demonstrate GPU buffer compatibility
    println!();
    println!("=== GPU Buffer Compatibility ===\n");
    println!("All GpuType structs are:");
    println!("  - Copy: Can be passed by value");
    println!("  - Pod: Plain old data, safe for memcpy");
    println!("  - Zeroable: Can be initialized with zeros");
    println!("  - #[repr(C)]: Predictable memory layout");
    println!();
    println!("This makes them suitable for:");
    println!("  - GPU buffer uploads (cuMemcpyHtoD, wgpu::Queue::write_buffer)");
    println!("  - Shared memory on GPU");
    println!("  - Zero-copy interop with GPU kernels");

    println!("\n=== Example completed! ===");
}
