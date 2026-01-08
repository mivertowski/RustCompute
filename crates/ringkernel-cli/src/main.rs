//! RingKernel CLI - Project scaffolding, kernel code generation, and profiling tool.
//!
//! # Commands
//!
//! - `ringkernel new <name>` - Create a new RingKernel project
//! - `ringkernel codegen <file>` - Generate GPU kernel code from Rust DSL
//! - `ringkernel check` - Validate kernel compatibility across backends
//! - `ringkernel profile` - Profile kernel performance
//!
//! # Examples
//!
//! ```bash
//! # Create a new project with persistent actor template
//! ringkernel new my-gpu-app --template persistent-actor
//!
//! # Generate CUDA and WGSL code from kernel file
//! ringkernel codegen src/kernels/processor.rs --backend cuda,wgsl
//!
//! # Check all kernels for backend compatibility
//! ringkernel check --backends all
//! ```

use clap::{Parser, Subcommand};
use colored::Colorize;
use std::process::ExitCode;
use tracing_subscriber::EnvFilter;

mod commands;
mod error;
mod templates;

use commands::{check, codegen, init, new_project};

/// RingKernel CLI - GPU-native persistent actor framework tooling
#[derive(Parser)]
#[command(name = "ringkernel")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Suppress all output except errors
    #[arg(short, long, global = true)]
    quiet: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new RingKernel project
    New {
        /// Project name
        name: String,

        /// Project template
        #[arg(short, long, default_value = "basic")]
        template: String,

        /// Target directory (default: current directory)
        #[arg(short, long)]
        path: Option<String>,

        /// GPU backends to enable
        #[arg(short, long, default_value = "cuda")]
        backends: String,

        /// Skip git initialization
        #[arg(long)]
        no_git: bool,
    },

    /// Initialize RingKernel in an existing project
    Init {
        /// GPU backends to enable
        #[arg(short, long, default_value = "cuda")]
        backends: String,

        /// Overwrite existing configuration
        #[arg(long)]
        force: bool,
    },

    /// Generate GPU kernel code from Rust DSL
    Codegen {
        /// Source file containing kernel definitions
        file: String,

        /// Target backends (comma-separated: cuda,wgsl,msl)
        #[arg(short, long, default_value = "cuda")]
        backend: String,

        /// Output directory for generated code
        #[arg(short, long)]
        output: Option<String>,

        /// Kernel name to generate (default: all kernels in file)
        #[arg(short, long)]
        kernel: Option<String>,

        /// Show generated code without writing files
        #[arg(long)]
        dry_run: bool,
    },

    /// Validate kernel compatibility across backends
    Check {
        /// Directory to scan for kernel files
        #[arg(short, long, default_value = "src")]
        path: String,

        /// Backends to check against (comma-separated or 'all')
        #[arg(short, long, default_value = "all")]
        backends: String,

        /// Show detailed compatibility report
        #[arg(long)]
        detailed: bool,
    },

    /// Profile kernel performance (placeholder for future implementation)
    Profile {
        /// Kernel to profile
        kernel: String,

        /// Number of iterations
        #[arg(short, long, default_value = "1000")]
        iterations: u32,

        /// Output format (text, json, flamegraph)
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

fn setup_logging(verbose: bool, quiet: bool) {
    let filter = if quiet {
        EnvFilter::new("error")
    } else if verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .init();
}

fn print_banner() {
    println!(
        "{}",
        r#"
  ____  _             _  __                    _
 |  _ \(_)_ __   __ _| |/ /___ _ __ _ __   ___| |
 | |_) | | '_ \ / _` | ' // _ \ '__| '_ \ / _ \ |
 |  _ <| | | | | (_| | . \  __/ |  | | | |  __/ |
 |_| \_\_|_| |_|\__, |_|\_\___|_|  |_| |_|\___|_|
                |___/
"#
        .bright_cyan()
    );
    println!(
        "  {} {}\n",
        "GPU-Native Persistent Actor Framework".bright_white(),
        format!("v{}", env!("CARGO_PKG_VERSION")).dimmed()
    );
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();

    setup_logging(cli.verbose, cli.quiet);

    if !cli.quiet {
        print_banner();
    }

    let result = match cli.command {
        Commands::New {
            name,
            template,
            path,
            backends,
            no_git,
        } => new_project::execute(&name, &template, path.as_deref(), &backends, no_git).await,

        Commands::Init { backends, force } => init::execute(&backends, force).await,

        Commands::Codegen {
            file,
            backend,
            output,
            kernel,
            dry_run,
        } => codegen::execute(&file, &backend, output.as_deref(), kernel.as_deref(), dry_run).await,

        Commands::Check {
            path,
            backends,
            detailed,
        } => check::execute(&path, &backends, detailed).await,

        Commands::Profile {
            kernel,
            iterations,
            format,
        } => {
            println!(
                "{} Profile command is not yet implemented",
                "Warning:".yellow()
            );
            println!(
                "  Would profile kernel '{}' for {} iterations with {} format",
                kernel.bright_white(),
                iterations.to_string().bright_white(),
                format.bright_white()
            );
            Ok(())
        }

        Commands::Completions { shell } => {
            use clap::CommandFactory;
            clap_complete::generate(
                shell,
                &mut Cli::command(),
                "ringkernel",
                &mut std::io::stdout(),
            );
            Ok(())
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{} {}", "Error:".red().bold(), e);
            ExitCode::FAILURE
        }
    }
}
