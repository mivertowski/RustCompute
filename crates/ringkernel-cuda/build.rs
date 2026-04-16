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
    println!("cargo:rerun-if-changed=src/cuda/cluster_kernels.cu");
    println!("cargo:rerun-if-changed=src/cuda/actor_lifecycle_kernel.cu");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=RINGKERNEL_CUDA_ARCH");

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

            // Compile cluster kernels (requires sm_90+)
            match compile_cluster_kernels(&nvcc, &out_dir) {
                Ok(()) => {
                    println!("cargo:rustc-cfg=has_cluster_kernels");
                    println!("cargo:warning=Cluster kernels compiled successfully");
                }
                Err(e) => {
                    println!("cargo:warning=Cluster kernels not available: {}", e);
                    generate_cluster_stub(&out_dir, &format!("Compilation failed: {}", e));
                }
            }

            // Compile actor lifecycle kernel (requires sm_90+ for cooperative groups)
            match compile_lifecycle_kernel(&nvcc, &out_dir) {
                Ok(()) => {
                    println!("cargo:rustc-cfg=has_lifecycle_kernel");
                    println!("cargo:warning=Actor lifecycle kernel compiled successfully");
                }
                Err(e) => {
                    println!("cargo:warning=Actor lifecycle kernel not available: {}", e);
                    generate_lifecycle_stub(&out_dir, &format!("Compilation failed: {}", e));
                }
            }
        }
        None => {
            println!("cargo:warning=nvcc not found - cooperative groups will use fallback");
            generate_stub(&out_dir, "nvcc not found at build time");
            generate_cluster_stub(&out_dir, "nvcc not found at build time");
            generate_lifecycle_stub(&out_dir, "nvcc not found at build time");
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
    // Architecture selection priority:
    // 1. RINGKERNEL_CUDA_ARCH env var (e.g., RINGKERNEL_CUDA_ARCH=sm_90)
    // 2. -arch=native (CUDA 12+, auto-detects installed GPU)
    // 3. Multi-arch fallback covering sm_75 through sm_90
    let arch_args = determine_cuda_arch(nvcc);

    let mut cmd = Command::new(nvcc);
    cmd.args([
        "-ptx",
        "-O3",
        "--generate-line-info",
    ]);
    for arg in &arch_args {
        cmd.arg(arg);
    }
    cmd.args([
        "-std=c++17",
        "-w", // Suppress warnings (unused variables in templates)
        "-o",
    ]);
    cmd.arg(ptx_file.to_str().unwrap());
    cmd.arg(cuda_src_path.to_str().unwrap());

    let status = cmd
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

/// Determine CUDA architecture flags for nvcc.
///
/// Priority:
/// 1. `RINGKERNEL_CUDA_ARCH` env var (e.g., `sm_90`)
/// 2. `-arch=native` (CUDA 12+, auto-detects installed GPU)
/// 3. Multi-arch fallback: sm_75, sm_80, sm_89, sm_90
fn determine_cuda_arch(nvcc: &Path) -> Vec<String> {
    // 1. Check for explicit environment variable
    if let Ok(arch) = env::var("RINGKERNEL_CUDA_ARCH") {
        let arch = arch.trim().to_string();
        println!("cargo:warning=Using RINGKERNEL_CUDA_ARCH={}", arch);
        // Support both "sm_90" and "-arch=sm_90" forms
        if arch.starts_with("-arch=") || arch.starts_with("-gencode") {
            return vec![arch];
        }
        return vec![format!("-arch={}", arch)];
    }

    // 2. Try -arch=native (CUDA 12+ feature, auto-detects installed GPU)
    if try_native_arch(nvcc) {
        println!("cargo:warning=Using -arch=native (auto-detected GPU)");
        return vec!["-arch=native".to_string()];
    }

    // 3. Fall back to multi-arch covering common architectures
    // sm_75: Turing (T4, RTX 2000)
    // sm_80: Ampere (A100, RTX 3000)
    // sm_89: Ada Lovelace (L40, RTX 4000)
    // sm_90: Hopper (H100, H200)
    // sm_100: Blackwell (B200, B300) — requires CUDA 12.8+
    println!("cargo:warning=Using multi-arch fallback (sm_75, sm_80, sm_89, sm_90, sm_100)");
    vec![
        "-gencode".to_string(),
        "arch=compute_75,code=sm_75".to_string(),
        "-gencode".to_string(),
        "arch=compute_80,code=sm_80".to_string(),
        "-gencode".to_string(),
        "arch=compute_89,code=sm_89".to_string(),
        "-gencode".to_string(),
        "arch=compute_90,code=sm_90".to_string(),
        "-gencode".to_string(),
        "arch=compute_100,code=sm_100".to_string(),
    ]
}

/// Test whether nvcc supports -arch=native by running a quick dry-run compilation.
fn try_native_arch(nvcc: &Path) -> bool {
    // Create a minimal CUDA source to test with
    let test_src = env::temp_dir().join("ringkernel_arch_test.cu");
    let _ = fs::write(&test_src, "extern \"C\" __global__ void _test() {}\n");

    let result = Command::new(nvcc)
        .args(["-ptx", "-arch=native", "-o", "/dev/null"])
        .arg(&test_src)
        .output();

    let _ = fs::remove_file(&test_src);

    match result {
        Ok(output) => output.status.success(),
        Err(_) => false,
    }
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

/// Compile cluster kernels (requires sm_90+) to PTX.
fn compile_cluster_kernels(nvcc: &Path, out_dir: &Path) -> Result<(), String> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cuda_src_path = manifest_dir
        .join("src")
        .join("cuda")
        .join("cluster_kernels.cu");

    if !cuda_src_path.exists() {
        return Err(format!("Cluster CUDA source not found: {:?}", cuda_src_path));
    }

    // Cluster kernels require sm_90 minimum (Hopper)
    let arch = env::var("RINGKERNEL_CUDA_ARCH").unwrap_or_default();
    let arch_num: u32 = arch
        .trim_start_matches("sm_")
        .parse()
        .unwrap_or(0);

    if arch_num > 0 && arch_num < 90 {
        return Err(format!(
            "Cluster kernels require sm_90+, but RINGKERNEL_CUDA_ARCH={}",
            arch
        ));
    }

    let ptx_file = out_dir.join("cluster_kernels.ptx");

    // Always use sm_90 for cluster kernels (minimum for cluster support)
    let mut cmd = Command::new(nvcc);
    cmd.args(["-ptx", "-O3", "--generate-line-info", "-arch=sm_90", "-std=c++17", "-w", "-o"]);
    cmd.arg(ptx_file.to_str().unwrap());
    cmd.arg(cuda_src_path.to_str().unwrap());

    let status = cmd
        .status()
        .map_err(|e| format!("Failed to execute nvcc for cluster kernels: {}", e))?;

    if !status.success() {
        return Err(format!(
            "nvcc cluster kernel compilation failed with exit code: {:?}",
            status.code()
        ));
    }

    let ptx_content =
        fs::read_to_string(&ptx_file).map_err(|e| format!("Failed to read cluster PTX: {}", e))?;

    let rust_file = out_dir.join("cluster_kernels.rs");
    write_cluster_rust_code(&rust_file, &ptx_content, true, "Cluster kernels compiled for sm_90")
        .map_err(|e| format!("Failed to write cluster Rust bindings: {}", e))?;

    Ok(())
}

/// Generate cluster stub when nvcc is not available or arch < sm_90.
fn generate_cluster_stub(out_dir: &Path, reason: &str) {
    let rust_file = out_dir.join("cluster_kernels.rs");
    write_cluster_rust_code(&rust_file, "", false, reason)
        .expect("Failed to write cluster Rust stub");
}

/// Write Rust code with cluster kernel PTX constant.
fn write_cluster_rust_code(
    path: &Path,
    ptx: &str,
    has_support: bool,
    message: &str,
) -> std::io::Result<()> {
    let mut code = String::new();

    code.push_str("// Auto-generated cluster kernel PTX.\n");
    code.push_str("// Generated by build.rs at build time.\n");
    code.push_str("// Requires compute capability 9.0+ (Hopper/Blackwell).\n\n");

    code.push_str("/// Pre-compiled PTX for Hopper cluster kernels.\n");
    code.push_str("/// Contains:\n");
    code.push_str("/// - cluster_test_sync: Cluster-level sync test\n");
    code.push_str("/// - cluster_dsmem_k2k: DSMEM-based K2K messaging\n");
    code.push_str("/// - cluster_persistent_actor: Persistent actors with cluster support\n");

    code.push_str("pub const CLUSTER_KERNEL_PTX: &str = r####\"");
    code.push_str(ptx);
    code.push_str("\"####;\n\n");

    code.push_str(&format!(
        "pub const HAS_CLUSTER_KERNEL_SUPPORT: bool = {};\n\n",
        has_support
    ));

    let escaped_message = message.replace('\\', "\\\\").replace('"', "\\\"");
    code.push_str(&format!(
        "pub const CLUSTER_KERNEL_BUILD_MESSAGE: &str = \"{}\";\n",
        escaped_message
    ));

    fs::write(path, code)
}

/// Compile actor lifecycle kernel to PTX.
fn compile_lifecycle_kernel(nvcc: &Path, out_dir: &Path) -> Result<(), String> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cuda_src = manifest_dir.join("src/cuda/actor_lifecycle_kernel.cu");

    if !cuda_src.exists() {
        return Err(format!("Actor lifecycle CUDA source not found: {:?}", cuda_src));
    }

    let ptx_file = out_dir.join("actor_lifecycle_kernel.ptx");

    let mut cmd = Command::new(nvcc);
    cmd.args(["-ptx", "-O3", "--generate-line-info", "-arch=sm_90", "-std=c++17", "-w", "-o"]);
    cmd.arg(ptx_file.to_str().unwrap());
    cmd.arg(cuda_src.to_str().unwrap());

    let status = cmd.status().map_err(|e| format!("nvcc failed: {}", e))?;
    if !status.success() {
        return Err(format!("nvcc lifecycle kernel failed: {:?}", status.code()));
    }

    let ptx_content = fs::read_to_string(&ptx_file)
        .map_err(|e| format!("Failed to read lifecycle PTX: {}", e))?;

    let rust_file = out_dir.join("actor_lifecycle_kernel.rs");
    let mut code = String::new();
    code.push_str("// Auto-generated actor lifecycle kernel PTX.\n\n");
    code.push_str("pub const LIFECYCLE_KERNEL_PTX: &str = r####\"");
    code.push_str(&ptx_content);
    code.push_str("\"####;\n\n");
    code.push_str("pub const HAS_LIFECYCLE_KERNEL: bool = true;\n");

    fs::write(&rust_file, code).map_err(|e| format!("Write failed: {}", e))
}

/// Generate lifecycle kernel stub.
fn generate_lifecycle_stub(out_dir: &Path, reason: &str) {
    let rust_file = out_dir.join("actor_lifecycle_kernel.rs");
    let code = format!(
        "// Actor lifecycle kernel not available: {}\n\n\
         pub const LIFECYCLE_KERNEL_PTX: &str = \"\";\n\
         pub const HAS_LIFECYCLE_KERNEL: bool = false;\n",
        reason
    );
    fs::write(&rust_file, code).expect("Failed to write lifecycle stub");
}
