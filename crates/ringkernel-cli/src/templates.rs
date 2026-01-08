//! Project templates for scaffolding.

/// Cargo.toml template.
pub const CARGO_TOML_TEMPLATE: &str = r#"[package]
name = "{{name}}"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <your.email@example.com>"]
description = "A RingKernel GPU application"

[dependencies]
# RingKernel framework
ringkernel = { version = "0.1", features = [{{#each backends}}"{{this}}"{{#unless @last}}, {{/unless}}{{/each}}] }

# Async runtime
tokio = { version = "1.48", features = ["full"] }

# Error handling
thiserror = "2.0"
anyhow = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[dev-dependencies]
criterion = "0.5"

[[example]]
name = "basic"
path = "examples/basic.rs"

[[bench]]
name = "kernel_bench"
harness = false
path = "benches/kernel_bench.rs"
"#;

/// Main.rs template.
pub const MAIN_RS_TEMPLATE: &str = r#"//! {{name}} - A RingKernel GPU application.

use anyhow::Result;
use tracing_subscriber::EnvFilter;

mod kernels;

fn setup_logging() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_logging();

    tracing::info!("Starting {{name}}...");

    // Initialize the RingKernel runtime
    let runtime = ringkernel::CpuRuntime::new().await?;

    // Launch your kernel
    let kernel = runtime
        .launch("{{name_pascal}}Kernel", Default::default())
        .await?;

    tracing::info!("Kernel launched: {:?}", kernel.id());

    // Send a test message
    // kernel.send(YourMessage { ... }).await?;

    // Shutdown
    runtime.shutdown().await?;

    tracing::info!("{{name}} completed successfully");
    Ok(())
}
"#;

/// Lib.rs template.
pub const LIB_RS_TEMPLATE: &str = r#"//! {{name}} library.
//!
//! This crate provides GPU-accelerated functionality using the RingKernel framework.

pub mod kernels;

pub use kernels::*;
"#;

/// Kernel module template.
pub const KERNEL_RS_TEMPLATE: &str = r#"//! GPU kernel definitions for {{name}}.

use ringkernel::prelude::*;

/// Request message for the kernel.
#[derive(Debug, Clone, RingMessage)]
#[message(type_id = 1)]
pub struct {{name_pascal}}Request {
    /// Request data.
    pub data: Vec<f32>,
}

/// Response message from the kernel.
#[derive(Debug, Clone, RingMessage)]
#[message(type_id = 2)]
pub struct {{name_pascal}}Response {
    /// Result data.
    pub result: Vec<f32>,
    /// Processing time in microseconds.
    pub elapsed_us: u64,
}

{{#if is_persistent}}
/// Persistent kernel handler for {{name_pascal}}.
///
/// This kernel remains active and processes messages continuously,
/// providing sub-microsecond command latency.
#[ring_kernel(id = "{{name_pascal}}Kernel", mode = "persistent", block_size = 256)]
pub async fn handle_{{name}}_request(
    ctx: &mut RingContext,
    msg: {{name_pascal}}Request,
) -> {{name_pascal}}Response {
    let start = std::time::Instant::now();

    // Process the request
    let result: Vec<f32> = msg.data.iter().map(|x| x * 2.0).collect();

    {{name_pascal}}Response {
        result,
        elapsed_us: start.elapsed().as_micros() as u64,
    }
}
{{else}}
/// Kernel handler for {{name_pascal}}.
#[ring_kernel(id = "{{name_pascal}}Kernel", block_size = 256)]
pub async fn handle_{{name}}_request(
    ctx: &mut RingContext,
    msg: {{name_pascal}}Request,
) -> {{name_pascal}}Response {
    let start = std::time::Instant::now();

    // Process the request
    let result: Vec<f32> = msg.data.iter().map(|x| x * 2.0).collect();

    {{name_pascal}}Response {
        result,
        elapsed_us: start.elapsed().as_micros() as u64,
    }
}
{{/if}}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_creation() {
        let req = {{name_pascal}}Request {
            data: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(req.data.len(), 3);
    }
}
"#;

/// Example template.
pub const EXAMPLE_RS_TEMPLATE: &str = r#"//! Basic example for {{name}}.

use anyhow::Result;
use {{name}}::{{name_pascal}}Request;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("{{name}} - Basic Example");
    println!("========================\n");

    // Create the runtime
    let runtime = ringkernel::CpuRuntime::new().await?;
    println!("Runtime created");

    // Launch the kernel
    let kernel = runtime
        .launch("{{name_pascal}}Kernel", Default::default())
        .await?;
    println!("Kernel launched: {:?}", kernel.id());

    // Create a test request
    let request = {{name_pascal}}Request {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
    };
    println!("Sending request with {} elements", request.data.len());

    // Send and wait for response
    // let response = kernel.send(request).await?;
    // println!("Response received in {} µs", response.elapsed_us);

    // Shutdown
    runtime.shutdown().await?;
    println!("\nExample completed successfully!");

    Ok(())
}
"#;

/// .gitignore template.
pub const GITIGNORE_TEMPLATE: &str = r#"# Generated files
/target/
/src/generated/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# GPU build artifacts
*.ptx
*.cubin
*.fatbin
*.spv

# Profiling
*.nvvp
*.nsys-rep
*.ncu-rep
perf.data
flamegraph.svg

# Environment
.env
.env.local

# Logs
*.log
"#;
