//! Build script for ringkernel-cuda cooperative groups support.
//!
//! This script:
//! 1. Detects nvcc at build time
//! 2. Compiles cooperative groups kernels to PTX
//! 3. Embeds PTX for runtime loading
//! 4. Provides graceful fallback if nvcc is not found

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/cooperative_kernels.cu");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Check if cooperative feature is enabled
    let cooperative_enabled = env::var("CARGO_FEATURE_COOPERATIVE").is_ok();

    if !cooperative_enabled {
        // Generate stub when cooperative feature is not enabled
        generate_stub(&out_dir, "Cooperative feature not enabled");
        return;
    }

    // Try to find nvcc
    match find_nvcc() {
        Some(nvcc) => {
            println!("cargo:warning=Found nvcc at: {:?}", nvcc);
            match compile_cooperative_kernels(&nvcc, &out_dir) {
                Ok(()) => {
                    println!("cargo:rustc-cfg=has_nvcc");
                    println!("cargo:warning=Cooperative groups kernels compiled successfully");
                }
                Err(e) => {
                    println!("cargo:warning=Failed to compile cooperative kernels: {}", e);
                    generate_stub(&out_dir, &format!("Compilation failed: {}", e));
                }
            }
        }
        None => {
            println!("cargo:warning=nvcc not found - cooperative groups will use fallback");
            generate_stub(&out_dir, "nvcc not found at build time");
        }
    }
}

/// Find nvcc executable.
fn find_nvcc() -> Option<PathBuf> {
    // Check CUDA_PATH environment variable
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc");
        if nvcc.exists() {
            return Some(nvcc);
        }
    }

    // Check CUDA_HOME environment variable
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let nvcc = PathBuf::from(&cuda_home).join("bin").join("nvcc");
        if nvcc.exists() {
            return Some(nvcc);
        }
    }

    // Check common Linux locations
    for path in &[
        "/usr/local/cuda/bin/nvcc",
        "/opt/cuda/bin/nvcc",
        "/usr/bin/nvcc",
    ] {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    // Try PATH
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    None
}

/// Compile cooperative kernels to PTX.
fn compile_cooperative_kernels(nvcc: &Path, out_dir: &Path) -> Result<(), String> {
    // Get the CUDA source from the crate source directory
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cuda_src_path = manifest_dir
        .join("src")
        .join("cuda")
        .join("cooperative_kernels.cu");

    if !cuda_src_path.exists() {
        return Err(format!("CUDA source not found: {:?}", cuda_src_path));
    }

    let ptx_file = out_dir.join("cooperative_kernels.ptx");

    // Compile with nvcc
    // Target sm_89 (Ada Lovelace) for RTX 40xx series
    // CUDA 13.0+ dropped support for sm_70, minimum is now sm_75
    // PTX is forward-compatible, so sm_89 PTX works on newer GPUs
    let status = Command::new(nvcc)
        .args([
            "-ptx",
            "-O3",
            "--generate-line-info",
            "-arch=sm_89", // Ada Lovelace (RTX 40xx)
            "-std=c++17",
            "-w", // Suppress warnings (unused variables in templates)
            "-o",
        ])
        .arg(ptx_file.to_str().unwrap())
        .arg(cuda_src_path.to_str().unwrap())
        .status()
        .map_err(|e| format!("Failed to execute nvcc: {}", e))?;

    if !status.success() {
        return Err(format!(
            "nvcc compilation failed with exit code: {:?}",
            status.code()
        ));
    }

    // Read PTX and generate Rust code
    let ptx_content =
        fs::read_to_string(&ptx_file).map_err(|e| format!("Failed to read PTX: {}", e))?;

    // Write the Rust code with embedded PTX
    let rust_file = out_dir.join("cooperative_kernels.rs");
    write_rust_code(
        &rust_file,
        &ptx_content,
        true,
        "Cooperative groups compiled successfully with nvcc",
    )
    .map_err(|e| format!("Failed to write Rust bindings: {}", e))?;

    Ok(())
}

/// Generate stub when nvcc is not available.
fn generate_stub(out_dir: &Path, reason: &str) {
    let rust_file = out_dir.join("cooperative_kernels.rs");
    write_rust_code(&rust_file, "", false, reason).expect("Failed to write Rust stub");
}

/// Write the Rust code with PTX constant.
fn write_rust_code(
    path: &Path,
    ptx: &str,
    has_support: bool,
    message: &str,
) -> std::io::Result<()> {
    let mut code = String::new();

    code.push_str("// Auto-generated cooperative kernel PTX.\n");
    code.push_str("// Generated by build.rs at build time.\n\n");

    code.push_str("/// Pre-compiled PTX for cooperative groups kernels.\n");
    code.push_str("/// Contains:\n");
    code.push_str("/// - coop_persistent_fdtd: Block-based FDTD with grid.sync()\n");
    code.push_str("/// - coop_ring_kernel_entry: Generic cooperative ring kernel\n");

    // Use raw string with enough # to not conflict with PTX content
    code.push_str("pub const COOPERATIVE_KERNEL_PTX: &str = r####\"");
    code.push_str(ptx);
    code.push_str("\"####;\n\n");

    code.push_str("/// Check if cooperative groups support was compiled.\n");
    code.push_str(&format!(
        "pub const HAS_COOPERATIVE_SUPPORT: bool = {};\n\n",
        has_support
    ));

    code.push_str("/// Build-time message about cooperative support.\n");
    // Escape quotes in message
    let escaped_message = message.replace('\\', "\\\\").replace('"', "\\\"");
    code.push_str(&format!(
        "pub const COOPERATIVE_BUILD_MESSAGE: &str = \"{}\";\n",
        escaped_message
    ));

    fs::write(path, code)
}
