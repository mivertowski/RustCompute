//! `ringkernel codegen` command - Generate GPU kernel code from Rust DSL.

use std::fs;
use std::path::Path;

use colored::Colorize;

use crate::error::{CliError, CliResult};

use super::parse_backends;

/// Execute the `codegen` command.
pub async fn execute(
    file: &str,
    backend: &str,
    output: Option<&str>,
    kernel: Option<&str>,
    dry_run: bool,
) -> CliResult<()> {
    let source_path = Path::new(file);

    // Check if source file exists
    if !source_path.exists() {
        return Err(CliError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Source file not found: {}", file),
        )));
    }

    // Read source file
    let source_content = fs::read_to_string(source_path)?;

    // Parse the source file
    let syntax_tree = syn::parse_file(&source_content)?;

    // Find kernel functions
    let kernels = find_kernel_functions(&syntax_tree);

    if kernels.is_empty() {
        println!(
            "{} No kernel functions found in {}",
            "Warning:".yellow(),
            file.bright_white()
        );
        println!("  Kernel functions should be annotated with #[ring_kernel(...)]");
        return Ok(());
    }

    let backend_list = parse_backends(backend);

    println!(
        "{} Generating code for {} kernel(s)",
        "→".bright_cyan(),
        kernels.len().to_string().bright_white()
    );
    println!("  {} Source: {}", "•".dimmed(), file.bright_yellow());
    println!(
        "  {} Backends: {}",
        "•".dimmed(),
        backend_list.join(", ").bright_yellow()
    );
    println!();

    // Filter kernels if specific one requested
    let kernels_to_generate: Vec<_> = if let Some(name) = kernel {
        kernels.into_iter().filter(|k| k.name == name).collect()
    } else {
        kernels
    };

    if kernels_to_generate.is_empty() {
        if let Some(name) = kernel {
            return Err(CliError::CodegenError(format!(
                "Kernel '{}' not found in source file",
                name
            )));
        }
    }

    // Generate code for each backend
    for kernel_info in &kernels_to_generate {
        for backend_name in &backend_list {
            println!(
                "  {} Generating {} code for {}...",
                "→".bright_cyan(),
                backend_name.bright_yellow(),
                kernel_info.name.bright_white()
            );

            let generated_code = generate_kernel_code(kernel_info, backend_name)?;

            if dry_run {
                println!();
                println!("{}:", "Generated code".bright_white().underline());
                println!("{}", generated_code.dimmed());
                println!();
            } else {
                // Determine output path
                let output_dir = output
                    .map(Path::new)
                    .unwrap_or_else(|| Path::new("src/generated"));

                fs::create_dir_all(output_dir)?;

                let extension = match backend_name.as_str() {
                    "cuda" => "cu",
                    "wgsl" => "wgsl",
                    "msl" => "metal",
                    _ => "txt",
                };

                let output_file = output_dir.join(format!(
                    "{}_{}.{}",
                    kernel_info.name, backend_name, extension
                ));

                fs::write(&output_file, &generated_code)?;

                println!(
                    "    {} Written to {}",
                    "✓".bright_green(),
                    output_file.display().to_string().bright_white()
                );
            }
        }
    }

    println!();
    println!("{} Code generation completed!", "✓".bright_green().bold());

    Ok(())
}

/// Information about a kernel function.
#[derive(Debug)]
#[allow(dead_code)]
struct KernelInfo {
    name: String,
    mode: String,
    block_size: u32,
    function: syn::ItemFn,
}

/// Find kernel functions in the syntax tree.
fn find_kernel_functions(syntax_tree: &syn::File) -> Vec<KernelInfo> {
    let mut kernels = Vec::new();

    for item in &syntax_tree.items {
        if let syn::Item::Fn(func) = item {
            // Look for #[ring_kernel(...)] attribute
            for attr in &func.attrs {
                if attr.path().is_ident("ring_kernel") {
                    let (name, mode, block_size) = parse_ring_kernel_attr(attr, &func.sig.ident);
                    kernels.push(KernelInfo {
                        name,
                        mode,
                        block_size,
                        function: func.clone(),
                    });
                    break;
                }
            }
        }
    }

    kernels
}

/// Parse the #[ring_kernel(...)] attribute.
fn parse_ring_kernel_attr(attr: &syn::Attribute, fn_name: &syn::Ident) -> (String, String, u32) {
    let mut name = fn_name.to_string();
    let mut mode = "standard".to_string();
    let mut block_size = 256u32;

    if let Ok(nested) = attr
        .parse_args_with(syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated)
    {
        for meta in nested {
            if let syn::Meta::NameValue(nv) = meta {
                let key = nv
                    .path
                    .get_ident()
                    .map(|i| i.to_string())
                    .unwrap_or_default();
                if let syn::Expr::Lit(syn::ExprLit { lit, .. }) = &nv.value {
                    match lit {
                        syn::Lit::Str(s) => {
                            let value = s.value();
                            match key.as_str() {
                                "id" => name = value,
                                "mode" => mode = value,
                                _ => {}
                            }
                        }
                        syn::Lit::Int(i) => {
                            if key == "block_size" {
                                block_size = i.base10_parse().unwrap_or(256);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    (name, mode, block_size)
}

/// Generate kernel code for a specific backend.
fn generate_kernel_code(kernel: &KernelInfo, backend: &str) -> CliResult<String> {
    match backend {
        "cuda" => generate_cuda_kernel(kernel),
        "wgsl" => generate_wgsl_kernel(kernel),
        "msl" => generate_msl_kernel(kernel),
        _ => Err(CliError::InvalidBackend(backend.to_string())),
    }
}

/// Generate CUDA kernel code.
fn generate_cuda_kernel(kernel: &KernelInfo) -> CliResult<String> {
    // Use ringkernel-cuda-codegen if available
    #[cfg(feature = "cuda")]
    {
        use ringkernel_cuda_codegen::{transpile_ring_kernel, RingKernelConfig};

        let config =
            RingKernelConfig::new(&kernel.name).with_block_size(kernel.block_size as usize);

        match transpile_ring_kernel(&kernel.function, &config) {
            Ok(code) => return Ok(code),
            Err(e) => return Err(CliError::CodegenError(e.to_string())),
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Fallback: generate a placeholder
        Ok(format!(
            r#"// Generated CUDA kernel: {}
// Mode: {}
// Block size: {}

// Note: Full CUDA codegen requires the 'cuda' feature.
// Enable with: ringkernel-cli --features cuda

__global__ void {}(/* parameters */) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Kernel implementation
}}
"#,
            kernel.name, kernel.mode, kernel.block_size, kernel.name
        ))
    }
}

/// Generate WGSL kernel code.
fn generate_wgsl_kernel(kernel: &KernelInfo) -> CliResult<String> {
    #[cfg(feature = "wgpu")]
    {
        use ringkernel_wgpu_codegen::transpile_global_kernel;

        match transpile_global_kernel(&kernel.function) {
            Ok(code) => return Ok(code),
            Err(e) => return Err(CliError::CodegenError(e.to_string())),
        }
    }

    #[cfg(not(feature = "wgpu"))]
    {
        // Fallback: generate a placeholder
        Ok(format!(
            r#"// Generated WGSL kernel: {}
// Mode: {}
// Block size: {}

// Note: Full WGSL codegen requires the 'wgpu' feature.
// Enable with: ringkernel-cli --features wgpu

@compute @workgroup_size({}, 1, 1)
fn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let tid = gid.x;
    // Kernel implementation
}}
"#,
            kernel.name, kernel.mode, kernel.block_size, kernel.block_size, kernel.name
        ))
    }
}

/// Generate MSL kernel code.
fn generate_msl_kernel(kernel: &KernelInfo) -> CliResult<String> {
    // MSL codegen via ringkernel-ir
    Ok(format!(
        r#"// Generated Metal Shading Language kernel: {}
// Mode: {}
// Block size: {}

#include <metal_stdlib>
using namespace metal;

kernel void {}(
    device float* data [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {{
    // Kernel implementation
}}
"#,
        kernel.name, kernel.mode, kernel.block_size, kernel.name
    ))
}
