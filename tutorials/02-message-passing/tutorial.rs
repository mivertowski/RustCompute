//! # Tutorial 02: Message Passing
//!
//! Learn how to communicate with GPU kernels using RingKernel's
//! lock-free message passing system.
//!
//! ## What You'll Learn
//!
//! 1. Creating messages with the `RingMessage` trait
//! 2. Sending messages to kernels (H2K)
//! 3. Receiving responses from kernels (K2H)
//! 4. Understanding message serialization
//! 5. Using Hybrid Logical Clocks for ordering
//!
//! ## Prerequisites
//!
//! - Completed Tutorial 01
//! - Understanding of Rust structs and traits

use ringkernel_core::prelude::*;
use ringkernel_cpu::CpuRuntime;

// ============================================================================
// STEP 1: Defining Messages
// ============================================================================

/// Messages in RingKernel use rkyv for zero-copy serialization.
/// The `RingMessage` trait provides:
/// - Type ID for routing
/// - Serialization/deserialization
/// - GPU-safe memory layout

// Example message structures (for demonstration)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ComputeRequest {
    values: Vec<f32>,
    operation: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ComputeResponse {
    result: f32,
    elapsed_us: u64,
}

// ============================================================================
// STEP 2: Understanding Queues
// ============================================================================

/// RingKernel uses lock-free ring buffers for message passing:
///
/// ```text
/// ┌─────────────────────────────────────────────────┐
/// │                  HOST (CPU)                      │
/// │                                                  │
/// │   [Producer] ──────────────────┐                │
/// │                                │                │
/// │                    H2K Queue   ▼                │
/// │                    ┌────────────────┐           │
/// │                    │ ○ ○ ● ● ○ ○ ○ │           │
/// │                    └────────────────┘           │
/// │                                │                │
/// └────────────────────────────────│────────────────┘
///                                  │
///                      ╔═══════════╧═══════════╗
///                      ║      GPU KERNEL       ║
///                      ║                       ║
///                      ║    [Consumer]         ║
///                      ║         │             ║
///                      ║    [Processing]       ║
///                      ║         │             ║
///                      ║    [Producer]         ║
///                      ║                       ║
///                      ╚═══════════╤═══════════╝
///                                  │
/// ┌────────────────────────────────│────────────────┐
/// │                                │                │
/// │                    K2H Queue   ▼                │
/// │                    ┌────────────────┐           │
/// │                    │ ○ ● ● ○ ○ ○ ○ │           │
/// │                    └────────────────┘           │
/// │                                │                │
/// │   [Consumer] ◄─────────────────┘                │
/// │                                                  │
/// │                  HOST (CPU)                      │
/// └─────────────────────────────────────────────────┘
/// ```
///
/// ○ = empty slot, ● = message

// ============================================================================
// STEP 3: Hybrid Logical Clocks
// ============================================================================

/// HLC provides causal ordering across distributed GPU actors.
/// Each timestamp has:
/// - Physical time (wall clock)
/// - Logical counter (for ordering within same physical time)
/// - Node ID (for uniqueness)

fn demonstrate_hlc() {
    println!("   HLC Timestamp Structure:\n");
    println!("   ┌────────────────────────────────────┐");
    println!("   │  Physical Time (64 bits)           │");
    println!("   │  Logical Counter (64 bits)         │");
    println!("   │  Node ID (64 bits)                 │");
    println!("   └────────────────────────────────────┘\n");

    // Create an HLC clock
    let clock = HlcClock::new(1);  // Node ID = 1

    // Generate timestamps
    let ts1 = clock.now();
    let ts2 = clock.now();
    let ts3 = clock.now();

    println!("   Generated timestamps:");
    println!("   ts1: physical={}, logical={}", ts1.physical, ts1.logical);
    println!("   ts2: physical={}, logical={}", ts2.physical, ts2.logical);
    println!("   ts3: physical={}, logical={}", ts3.physical, ts3.logical);
    println!();

    // Demonstrate ordering
    println!("   Causal ordering:");
    println!("   ts1 < ts2: {}", ts1 < ts2);
    println!("   ts2 < ts3: {}", ts2 < ts3);
    println!();

    // Demonstrate update from remote timestamp
    let remote_ts = HlcTimestamp::new(ts3.physical + 1000, 5, 2);  // From node 2
    if let Ok(updated) = clock.update(&remote_ts) {
        println!("   After receiving remote timestamp:");
        println!("   remote: physical={}, logical={}, node=2", remote_ts.physical, remote_ts.logical);
        println!("   updated local: physical={}, logical={}", updated.physical, updated.logical);
    }
    println!();
}

// ============================================================================
// STEP 4: Message Queues in Practice
// ============================================================================

fn demonstrate_queue_concepts() {
    println!("   Queue Characteristics:\n");
    println!("   - Lock-free SPSC (Single Producer Single Consumer)");
    println!("   - Power-of-2 capacity for fast modulo");
    println!("   - Cache-line padding to avoid false sharing");
    println!("   - Zero-copy when possible");
    println!();

    println!("   Available Queue Types:\n");
    println!("   ┌─────────────────────────────────────────┐");
    println!("   │ SpscQueue   - Single Producer/Consumer  │");
    println!("   │               Best for H2K/K2H          │");
    println!("   │                                         │");
    println!("   │ MpscQueue   - Multiple Producers        │");
    println!("   │               Good for fan-in patterns  │");
    println!("   │                                         │");
    println!("   │ BoundedQueue - Thread-safe bounded      │");
    println!("   │                General purpose          │");
    println!("   └─────────────────────────────────────────┘");
    println!();

    println!("   Common Operations (MessageQueue trait):\n");
    println!("   ┌─────────────────────────────────────────┐");
    println!("   │ try_push(msg)  → Option<()>             │");
    println!("   │   Returns None if queue full            │");
    println!("   │                                         │");
    println!("   │ try_pop()      → Option<T>              │");
    println!("   │   Returns None if queue empty           │");
    println!("   │                                         │");
    println!("   │ is_empty()     → bool                   │");
    println!("   │ is_full()      → bool                   │");
    println!("   │ len()          → usize                  │");
    println!("   └─────────────────────────────────────────┘");
    println!();
}

// ============================================================================
// STEP 5: Message Flow Patterns
// ============================================================================

fn demonstrate_patterns() {
    println!("   Common Message Patterns:\n");

    println!("   1. Request/Response (most common):");
    println!("      Host → H2K → Kernel → K2H → Host");
    println!();

    println!("   2. Fire-and-Forget:");
    println!("      Host → H2K → Kernel (no response needed)");
    println!();

    println!("   3. Streaming:");
    println!("      Kernel → K2H → K2H → K2H → Host");
    println!("      (continuous output from kernel)");
    println!();

    println!("   4. Kernel-to-Kernel (K2K):");
    println!("      Kernel A → K2K → Kernel B");
    println!("      (direct inter-kernel messaging)");
    println!();
}

// ============================================================================
// MAIN TUTORIAL CODE
// ============================================================================

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("===========================================");
    println!("   Tutorial 02: Message Passing");
    println!("===========================================\n");

    // Create runtime
    let runtime = CpuRuntime::new().await?;

    println!("Step 1: Message Definition\n");
    println!("   Messages implement the RingMessage trait:");
    println!("   ```rust");
    println!("   #[derive(RingMessage)]");
    println!("   struct ComputeRequest {{");
    println!("       values: Vec<f32>,");
    println!("       operation: String,");
    println!("   }}");
    println!("   ```\n");

    println!("Step 2: Queue Architecture\n");
    demonstrate_queue_concepts();

    println!("Step 3: Hybrid Logical Clocks\n");
    demonstrate_hlc();

    println!("Step 4: Message Patterns\n");
    demonstrate_patterns();

    println!("Step 5: Using Queues with Kernels\n");

    // Launch a kernel to demonstrate
    let kernel = runtime.launch("message_demo", LaunchOptions::default()).await?;
    println!("   Launched kernel: {}", kernel.id());
    println!();

    println!("   Message flow:");
    println!("   1. Create message with timestamp");
    println!("   2. Serialize via rkyv (zero-copy)");
    println!("   3. Push to H2K queue");
    println!("   4. Kernel processes message");
    println!("   5. Kernel pushes response to K2H queue");
    println!("   6. Host pops and deserializes response");
    println!();

    // Clean up
    kernel.terminate().await?;

    // Summary
    println!("===========================================");
    println!("   Tutorial Complete!");
    println!("===========================================");
    println!();
    println!("What you learned:");
    println!("  - RingMessage trait for GPU-safe messages");
    println!("  - Lock-free queues for H2K/K2H communication");
    println!("  - Hybrid Logical Clocks for causal ordering");
    println!("  - Common message passing patterns");
    println!();
    println!("Next: Tutorial 03 - Writing GPU Kernels");
    println!("      Learn the Rust DSL for GPU kernel development");

    Ok(())
}

// ============================================================================
// EXERCISES
// ============================================================================

// Exercise 1: Define your own RingMessage type with multiple fields
//
// Exercise 2: Experiment with different queue capacities and observe
//             when try_push starts returning None
//
// Exercise 3: Create two HlcClock instances and simulate message exchange
//             with update() calls to see how timestamps merge

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hlc_ordering() {
        let clock = HlcClock::new(1);
        let ts1 = clock.now();
        let ts2 = clock.now();
        assert!(ts1 < ts2);
    }

    #[tokio::test]
    async fn test_tutorial_completes() {
        let runtime = CpuRuntime::new().await.unwrap();
        let kernel = runtime.launch("test", LaunchOptions::default()).await.unwrap();
        kernel.terminate().await.unwrap();
    }
}
