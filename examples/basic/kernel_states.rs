//! # Kernel States and Lifecycle Example
//!
//! This example demonstrates the full kernel lifecycle and state machine:
//!
//! ```text
//!                 ┌──────────┐
//!                 │ Created  │
//!                 └────┬─────┘
//!                      │ launch()
//!                      ▼
//!                 ┌──────────┐
//!                 │ Launched │
//!                 └────┬─────┘
//!                      │ activate()
//!                      ▼
//!    deactivate() ┌──────────┐
//!         ┌───────│  Active  │
//!         │       └────┬─────┘
//!         │            │
//!         ▼            │ terminate()
//!    ┌────────────┐    │
//!    │Deactivated │────┘
//!    └────────────┘
//!         │
//!         │ terminate()
//!         ▼
//!    ┌──────────┐
//!    │Terminated│ (Final state)
//!    └──────────┘
//! ```
//!
//! ## Run this example:
//! ```bash
//! cargo run -p ringkernel --example kernel_states
//! ```

use ringkernel::prelude::*;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Kernel Lifecycle Example ===\n");

    let runtime = RingKernel::builder().backend(Backend::Cpu).build().await?;

    // Create multiple kernels to demonstrate different states
    println!("Creating kernels...\n");

    let kernel_a = runtime
        .launch("kernel_a", LaunchOptions::default().without_auto_activate())
        .await?;
    let kernel_b = runtime
        .launch("kernel_b", LaunchOptions::default().without_auto_activate())
        .await?;
    let kernel_c = runtime
        .launch("kernel_c", LaunchOptions::default().without_auto_activate())
        .await?;

    // ====== State: Launched ======
    println!("=== State: LAUNCHED ===");
    println!("Kernels start in Launched state after launch (auto_activate=false).");
    print_states(&[&kernel_a, &kernel_b, &kernel_c]);

    // ====== State: Active ======
    println!("\n=== State: ACTIVE ===");
    println!("After activate(), kernels begin processing.");

    kernel_a.activate().await?;
    kernel_b.activate().await?;
    // kernel_c stays Launched

    print_states(&[&kernel_a, &kernel_b, &kernel_c]);

    // ====== State: Deactivated ======
    println!("\n=== State: DEACTIVATED ===");
    println!("Deactivated kernels pause but preserve state.");

    kernel_b.deactivate().await?; // or suspend()

    print_states(&[&kernel_a, &kernel_b, &kernel_c]);

    // Resume deactivated kernel
    println!("\nResuming kernel_b...");
    kernel_b.activate().await?; // or resume()
    print_states(&[&kernel_a, &kernel_b, &kernel_c]);

    // ====== Checking Status Details ======
    println!("\n=== Kernel Status Details ===");
    for (name, kernel) in [("A", &kernel_a), ("B", &kernel_b), ("C", &kernel_c)] {
        let status = kernel.status();
        println!("\nKernel {}:", name);
        println!("  State: {:?}", status.state);
        println!("  Mode: {:?}", status.mode);
        println!("  Input queue depth: {}", status.input_queue_depth);
        println!("  Output queue depth: {}", status.output_queue_depth);
        println!("  Messages processed: {}", status.messages_processed);
        println!("  Uptime: {:?}", status.uptime);
    }

    // ====== Using convenience methods ======
    println!("\n=== Convenience Methods ===");
    println!("kernel_a.is_active(): {}", kernel_a.is_active());
    println!("kernel_b.is_active(): {}", kernel_b.is_active());
    println!("kernel_c.is_active(): {}", kernel_c.is_active());

    // ====== State: Terminated ======
    println!("\n=== State: TERMINATED ===");
    println!("Terminated kernels release all resources.");

    kernel_a.terminate().await?;
    kernel_b.terminate().await?;
    kernel_c.terminate().await?;

    println!("\nKernel states after termination:");
    println!("kernel_a.is_terminated(): {}", kernel_a.is_terminated());
    println!("kernel_b.is_terminated(): {}", kernel_b.is_terminated());
    println!("kernel_c.is_terminated(): {}", kernel_c.is_terminated());

    // After termination, kernels are removed from the runtime
    println!("\nKernels remaining: {:?}", runtime.list_kernels());

    // ====== Demonstrating Multiple Kernels ======
    println!("\n=== Multi-Kernel Orchestration ===");
    demonstrate_orchestration(&runtime).await?;

    runtime.shutdown().await?;
    println!("\n=== Example completed! ===");

    Ok(())
}

fn print_states(kernels: &[&KernelHandle]) {
    for kernel in kernels {
        println!("  {} -> {:?}", kernel.id(), kernel.state());
    }
}

/// Demonstrates orchestrating multiple kernels in a pipeline
async fn demonstrate_orchestration(
    runtime: &RingKernel,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("\nLaunching pipeline: ingress -> processor -> egress");

    // Create a processing pipeline
    let ingress = runtime
        .launch("ingress", LaunchOptions::default().without_auto_activate())
        .await?;
    let processor = runtime
        .launch(
            "processor",
            LaunchOptions::default().without_auto_activate(),
        )
        .await?;
    let egress = runtime
        .launch("egress", LaunchOptions::default().without_auto_activate())
        .await?;

    // Activate in order (downstream first for back-pressure handling)
    println!("Activating pipeline (downstream first)...");
    egress.activate().await?;
    processor.activate().await?;
    ingress.activate().await?;

    println!("Pipeline active. Processing would happen here...");
    println!(
        "  All active: {}",
        ingress.is_active() && processor.is_active() && egress.is_active()
    );

    // Graceful shutdown (upstream first)
    println!("Shutting down pipeline (upstream first)...");
    ingress.suspend().await?;
    processor.suspend().await?;
    egress.suspend().await?;

    // Final termination
    ingress.terminate().await?;
    processor.terminate().await?;
    egress.terminate().await?;

    println!("Pipeline shut down cleanly.");

    Ok(())
}
