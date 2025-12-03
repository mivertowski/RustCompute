//! # CUDA Ring Kernel Transpilation Example
//!
//! Demonstrates transpiling Rust handler functions to persistent CUDA ring kernels.
//! Ring kernels are the core of RingKernel's actor model on GPU.
//!
//! ## Run this example:
//! ```bash
//! cargo run -p ringkernel --example ring_kernel_codegen
//! ```
//!
//! ## What this demonstrates:
//!
//! - ControlBlock struct for kernel lifecycle management
//! - Hybrid Logical Clock (HLC) for causal ordering
//! - Persistent `while(true)` message processing loop
//! - RingContext method inlining (`ctx.thread_id()` -> `threadIdx.x`)
//! - K2K (kernel-to-kernel) messaging support

use ringkernel_cuda_codegen::{transpile_ring_kernel, RingKernelConfig};
use syn::parse_quote;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== CUDA Ring Kernel Transpilation ===\n");

    // ========== Example 1: Simple Handler ==========
    println!("--- Example 1: Simple Value Processing Handler ---\n");

    let simple_handler: syn::ItemFn = parse_quote! {
        fn process(value: f32) -> f32 {
            let result = value * 2.0;
            result
        }
    };

    let config = RingKernelConfig::new("processor")
        .with_block_size(128)
        .with_queue_capacity(1024)
        .with_hlc(true);

    let cuda = transpile_ring_kernel(&simple_handler, &config)?;

    println!("Rust Handler:\n");
    println!("  fn process(value: f32) -> f32 {{");
    println!("      let result = value * 2.0;");
    println!("      result");
    println!("  }}\n");
    println!("Config: block_size=128, queue_capacity=1024, hlc=true\n");
    println!("Generated CUDA ({} lines):\n", cuda.lines().count());

    // Show key sections
    println!("Key Components Generated:\n");

    if cuda.contains("struct __align__(128) ControlBlock") {
        println!("  [x] ControlBlock struct (128-byte aligned)");
    }
    if cuda.contains("is_active") && cuda.contains("should_terminate") {
        println!("  [x] Lifecycle flags (is_active, should_terminate, has_terminated)");
    }
    if cuda.contains("messages_processed") {
        println!("  [x] Message counter");
    }
    if cuda.contains("hlc_physical") && cuda.contains("hlc_logical") {
        println!("  [x] HLC timestamp fields");
    }
    if cuda.contains("ring_kernel_processor") {
        println!("  [x] Kernel function: ring_kernel_processor");
    }
    if cuda.contains("while (true)") {
        println!("  [x] Persistent message loop");
    }
    if cuda.contains("USER HANDLER CODE") {
        println!("  [x] Handler code embedded in loop");
    }
    if cuda.contains("has_terminated") {
        println!("  [x] Graceful termination");
    }

    println!("\n--- First 80 lines of generated code ---\n");
    for (i, line) in cuda.lines().take(80).enumerate() {
        println!("{:3}: {}", i + 1, line);
    }
    if cuda.lines().count() > 80 {
        println!("\n... ({} more lines)\n", cuda.lines().count() - 80);
    }

    // ========== Example 2: Handler with RingContext ==========
    println!("--- Example 2: Handler with RingContext ---\n");

    let context_handler: syn::ItemFn = parse_quote! {
        fn process_with_ctx(ctx: &RingContext, value: f32) -> f32 {
            let tid = ctx.global_thread_id();
            ctx.sync_threads();
            let result = value * (tid as f32);
            result
        }
    };

    let config = RingKernelConfig::new("ctx_processor")
        .with_block_size(256)
        .with_hlc(true);

    let _cuda = transpile_ring_kernel(&context_handler, &config)?;

    println!("Rust Handler:\n");
    println!("  fn process_with_ctx(ctx: &RingContext, value: f32) -> f32 {{");
    println!("      let tid = ctx.global_thread_id();");
    println!("      ctx.sync_threads();");
    println!("      let result = value * (tid as f32);");
    println!("      result");
    println!("  }}\n");
    println!("Context Method Inlining:");
    println!("  ctx.global_thread_id() -> (blockIdx.x * blockDim.x + threadIdx.x)");
    println!("  ctx.sync_threads()     -> __syncthreads()\n");

    // ========== Example 3: K2K-Enabled Kernel ==========
    println!("--- Example 3: K2K-Enabled Ring Kernel ---\n");

    let k2k_handler: syn::ItemFn = parse_quote! {
        fn forward(msg: f32) -> f32 {
            msg
        }
    };

    let config = RingKernelConfig::new("forwarder")
        .with_block_size(64)
        .with_queue_capacity(512)
        .with_hlc(true)
        .with_k2k(true);

    let cuda = transpile_ring_kernel(&k2k_handler, &config)?;

    println!("Config: block_size=64, queue_capacity=512, hlc=true, k2k=true\n");
    println!("Additional K2K Components Generated:\n");

    if cuda.contains("K2KRoutingTable") {
        println!("  [x] K2KRoutingTable struct");
    }
    if cuda.contains("K2KRoute") {
        println!("  [x] K2KRoute struct (target_id, buffer_offset, capacity)");
    }
    if cuda.contains("K2KInboxHeader") {
        println!("  [x] K2KInboxHeader struct");
    }
    if cuda.contains("k2k_routes") {
        println!("  [x] k2k_routes parameter");
    }
    if cuda.contains("k2k_inbox") {
        println!("  [x] k2k_inbox parameter");
    }
    if cuda.contains("k2k_outbox") {
        println!("  [x] k2k_outbox parameter");
    }
    if cuda.contains("k2k_send") {
        println!("  [x] k2k_send() helper function");
    }
    if cuda.contains("k2k_try_recv") {
        println!("  [x] k2k_try_recv() helper function");
    }

    // ========== Ring Kernel Summary ==========
    println!("\n=== Ring Kernel Features Summary ===\n");

    println!("ControlBlock (128-byte GPU-resident):");
    println!("  - is_active: u32         Activation flag");
    println!("  - should_terminate: u32  Termination request");
    println!("  - has_terminated: u32    Termination complete");
    println!("  - messages_processed: u64 Counter");
    println!("  - input_head/tail: u64   Queue pointers");
    println!("  - hlc_physical: u64      HLC wall clock");
    println!("  - hlc_logical: u32       HLC logical counter\n");

    println!("RingContext Methods (inlined to CUDA):");
    println!("  ctx.thread_id()       -> threadIdx.x");
    println!("  ctx.block_id()        -> blockIdx.x");
    println!("  ctx.global_thread_id()-> blockIdx.x * blockDim.x + threadIdx.x");
    println!("  ctx.sync_threads()    -> __syncthreads()");
    println!("  ctx.lane_id()         -> threadIdx.x & 31");
    println!("  ctx.warp_id()         -> threadIdx.x >> 5\n");

    println!("K2K Intrinsics (when enabled):");
    println!("  k2k_send(target_id, msg)   Send to another kernel");
    println!("  k2k_try_recv()             Try to receive message");
    println!("  k2k_peek()                 Peek without consuming");
    println!("  k2k_pending_count()        Check inbox size\n");

    println!("Configuration Builder:");
    println!("  RingKernelConfig::new(\"name\")");
    println!("      .with_block_size(n)       Threads per block");
    println!("      .with_queue_capacity(n)   Message queue size");
    println!("      .with_hlc(bool)           Enable HLC");
    println!("      .with_k2k(bool)           Enable K2K messaging\n");

    println!("=== Ring Kernel Example Complete! ===");

    Ok(())
}
