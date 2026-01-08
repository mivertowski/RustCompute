//! `ringkernel init` command - Initialize RingKernel in an existing project.

use std::fs;
use std::path::Path;

use colored::Colorize;

use crate::error::{CliError, CliResult};

use super::parse_backends;

/// Execute the `init` command.
pub async fn execute(backends: &str, force: bool) -> CliResult<()> {
    let current_dir = std::env::current_dir()?;

    // Check if Cargo.toml exists
    let cargo_toml = current_dir.join("Cargo.toml");
    if !cargo_toml.exists() {
        return Err(CliError::Config(
            "No Cargo.toml found. Run this command from a Rust project directory.".to_string(),
        ));
    }

    // Check if ringkernel.toml already exists
    let config_path = current_dir.join("ringkernel.toml");
    if config_path.exists() && !force {
        return Err(CliError::Config(
            "ringkernel.toml already exists. Use --force to overwrite.".to_string(),
        ));
    }

    let backend_list = parse_backends(backends);

    println!(
        "{} Initializing RingKernel in current project",
        "→".bright_cyan()
    );
    println!(
        "  {} Backends: {}",
        "•".dimmed(),
        backend_list.join(", ").bright_yellow()
    );
    println!();

    // Create kernels directory
    let kernels_dir = current_dir.join("src/kernels");
    if !kernels_dir.exists() {
        fs::create_dir_all(&kernels_dir)?;
        println!("  {} Created src/kernels/", "✓".bright_green());
    }

    // Generate ringkernel.toml
    generate_config(&current_dir, &backend_list)?;
    println!("  {} Created ringkernel.toml", "✓".bright_green());

    // Create sample kernel if kernels directory is empty
    let kernel_mod = kernels_dir.join("mod.rs");
    if !kernel_mod.exists() {
        generate_sample_kernel(&kernels_dir)?;
        println!("  {} Created src/kernels/mod.rs", "✓".bright_green());
    }

    println!();
    println!(
        "{} RingKernel initialized successfully!",
        "✓".bright_green().bold()
    );
    println!();
    println!("  Add RingKernel to your Cargo.toml dependencies:");
    println!(
        "    {}",
        format!(
            r#"ringkernel = {{ version = "0.1", features = [{}] }}"#,
            backend_list
                .iter()
                .map(|b| format!("\"{}\"", b))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .bright_white()
    );
    println!();

    Ok(())
}

fn generate_config(project_path: &Path, backends: &[String]) -> CliResult<()> {
    let config = format!(
        r#"# RingKernel Project Configuration

[backends]
{}

[codegen]
# Output directory for generated GPU code
output_dir = "src/generated"
# Generate debug symbols in GPU code
debug = false

[kernel.defaults]
# Default block size for kernels
block_size = 256
# Default queue capacity
queue_capacity = 1024
"#,
        backends
            .iter()
            .map(|b| format!("{} = true", b))
            .collect::<Vec<_>>()
            .join("\n")
    );

    fs::write(project_path.join("ringkernel.toml"), config)?;
    Ok(())
}

fn generate_sample_kernel(kernels_dir: &Path) -> CliResult<()> {
    let sample = r#"//! GPU kernel definitions.

use ringkernel::prelude::*;

/// Sample request message.
#[derive(Debug, Clone, RingMessage)]
#[message(type_id = 1)]
pub struct SampleRequest {
    pub data: Vec<f32>,
}

/// Sample response message.
#[derive(Debug, Clone, RingMessage)]
#[message(type_id = 2)]
pub struct SampleResponse {
    pub result: Vec<f32>,
}

/// Sample kernel handler.
#[ring_kernel(id = "SampleKernel", block_size = 256)]
pub async fn handle_sample(
    ctx: &mut RingContext,
    msg: SampleRequest,
) -> SampleResponse {
    // Process the request
    let result: Vec<f32> = msg.data.iter().map(|x| x * 2.0).collect();
    SampleResponse { result }
}
"#;

    fs::write(kernels_dir.join("mod.rs"), sample)?;
    Ok(())
}
