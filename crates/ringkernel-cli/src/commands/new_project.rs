//! `ringkernel new` command - Create a new RingKernel project.

use std::fs;
use std::path::Path;
use std::process::Command;

use colored::Colorize;
use handlebars::Handlebars;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::json;

use crate::error::{CliError, CliResult};
use crate::templates;

use super::{parse_backends, validate_project_name};

/// Execute the `new` command.
pub async fn execute(
    name: &str,
    template: &str,
    path: Option<&str>,
    backends: &str,
    no_git: bool,
) -> CliResult<()> {
    // Validate project name
    validate_project_name(name).map_err(CliError::InvalidProjectName)?;

    // Determine project path
    let base_path = path.map(Path::new).unwrap_or(Path::new("."));
    let project_path = base_path.join(name);

    // Check if project already exists
    if project_path.exists() {
        return Err(CliError::ProjectExists(project_path.display().to_string()));
    }

    // Parse backends
    let backend_list = parse_backends(backends);

    println!(
        "{} Creating new RingKernel project: {}",
        "→".bright_cyan(),
        name.bright_white().bold()
    );
    println!(
        "  {} Template: {}",
        "•".dimmed(),
        template.bright_yellow()
    );
    println!(
        "  {} Backends: {}",
        "•".dimmed(),
        backend_list.join(", ").bright_yellow()
    );
    println!(
        "  {} Path: {}",
        "•".dimmed(),
        project_path.display().to_string().bright_yellow()
    );
    println!();

    // Create progress bar
    let pb = ProgressBar::new(5);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    // Step 1: Create directory structure
    pb.set_message("Creating directory structure...");
    create_directory_structure(&project_path)?;
    pb.inc(1);

    // Step 2: Generate Cargo.toml
    pb.set_message("Generating Cargo.toml...");
    generate_cargo_toml(&project_path, name, template, &backend_list)?;
    pb.inc(1);

    // Step 3: Generate source files
    pb.set_message("Generating source files...");
    generate_source_files(&project_path, name, template, &backend_list)?;
    pb.inc(1);

    // Step 4: Generate configuration
    pb.set_message("Generating configuration...");
    generate_config_files(&project_path, name, &backend_list)?;
    pb.inc(1);

    // Step 5: Initialize git
    pb.set_message("Initializing git...");
    if !no_git {
        initialize_git(&project_path)?;
    }
    pb.inc(1);

    pb.finish_with_message("Done!");
    println!();

    // Print success message
    println!(
        "{} Project created successfully!",
        "✓".bright_green().bold()
    );
    println!();
    println!("  Next steps:");
    println!(
        "    {} {}",
        "cd".bright_white(),
        name.bright_yellow()
    );
    println!(
        "    {} {}",
        "cargo build".bright_white(),
        "--release".dimmed()
    );
    println!(
        "    {} {}",
        "cargo run".bright_white(),
        "--example basic".dimmed()
    );
    println!();

    Ok(())
}

fn create_directory_structure(project_path: &Path) -> CliResult<()> {
    let dirs = [
        "",
        "src",
        "src/kernels",
        "examples",
        "benches",
        "tests",
    ];

    for dir in dirs {
        fs::create_dir_all(project_path.join(dir))?;
    }

    Ok(())
}

fn generate_cargo_toml(
    project_path: &Path,
    name: &str,
    template: &str,
    backends: &[String],
) -> CliResult<()> {
    let mut handlebars = Handlebars::new();
    handlebars.register_template_string("cargo_toml", templates::CARGO_TOML_TEMPLATE)?;

    let features = backends
        .iter()
        .map(|b| format!("ringkernel/{}", b))
        .collect::<Vec<_>>()
        .join(", ");

    let data = json!({
        "name": name,
        "template": template,
        "backends": backends,
        "features": features,
        "has_cuda": backends.contains(&"cuda".to_string()),
        "has_wgpu": backends.contains(&"wgpu".to_string()),
        "has_metal": backends.contains(&"msl".to_string()),
    });

    let content = handlebars.render("cargo_toml", &data)?;
    fs::write(project_path.join("Cargo.toml"), content)?;

    Ok(())
}

fn generate_source_files(
    project_path: &Path,
    name: &str,
    template: &str,
    backends: &[String],
) -> CliResult<()> {
    let mut handlebars = Handlebars::new();

    // Register templates
    handlebars.register_template_string("main", templates::MAIN_RS_TEMPLATE)?;
    handlebars.register_template_string("lib", templates::LIB_RS_TEMPLATE)?;
    handlebars.register_template_string("kernel", templates::KERNEL_RS_TEMPLATE)?;
    handlebars.register_template_string("example", templates::EXAMPLE_RS_TEMPLATE)?;

    let data = json!({
        "name": name,
        "name_upper": name.to_uppercase().replace('-', "_"),
        "name_pascal": to_pascal_case(name),
        "template": template,
        "backends": backends,
        "has_cuda": backends.contains(&"cuda".to_string()),
        "has_wgpu": backends.contains(&"wgpu".to_string()),
        "is_persistent": template == "persistent-actor" || template == "persistent",
    });

    // Generate main.rs
    let main_content = handlebars.render("main", &data)?;
    fs::write(project_path.join("src/main.rs"), main_content)?;

    // Generate lib.rs
    let lib_content = handlebars.render("lib", &data)?;
    fs::write(project_path.join("src/lib.rs"), lib_content)?;

    // Generate kernel file
    let kernel_content = handlebars.render("kernel", &data)?;
    fs::write(project_path.join("src/kernels/mod.rs"), kernel_content)?;

    // Generate example
    let example_content = handlebars.render("example", &data)?;
    fs::write(project_path.join("examples/basic.rs"), example_content)?;

    Ok(())
}

fn generate_config_files(project_path: &Path, name: &str, backends: &[String]) -> CliResult<()> {
    // Generate .gitignore
    fs::write(
        project_path.join(".gitignore"),
        templates::GITIGNORE_TEMPLATE,
    )?;

    // Generate README.md
    let readme = format!(
        "# {}\n\nA RingKernel GPU application.\n\n## Building\n\n```bash\ncargo build --release\n```\n\n## Running\n\n```bash\ncargo run --example basic\n```\n",
        name
    );
    fs::write(project_path.join("README.md"), readme)?;

    // Generate ringkernel.toml configuration
    let config = format!(
        r#"# RingKernel Project Configuration

[project]
name = "{}"
version = "0.1.0"

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
        name,
        backends
            .iter()
            .map(|b| format!("{} = true", b))
            .collect::<Vec<_>>()
            .join("\n")
    );
    fs::write(project_path.join("ringkernel.toml"), config)?;

    Ok(())
}

fn initialize_git(project_path: &Path) -> CliResult<()> {
    // Check if git is available
    if Command::new("git").arg("--version").output().is_err() {
        return Ok(()); // Git not available, skip
    }

    // Initialize git repository
    Command::new("git")
        .arg("init")
        .current_dir(project_path)
        .output()?;

    // Create initial commit
    Command::new("git")
        .args(["add", "."])
        .current_dir(project_path)
        .output()?;

    Command::new("git")
        .args(["commit", "-m", "Initial commit"])
        .current_dir(project_path)
        .output()?;

    Ok(())
}

fn to_pascal_case(s: &str) -> String {
    s.split(|c| c == '-' || c == '_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        })
        .collect()
}
