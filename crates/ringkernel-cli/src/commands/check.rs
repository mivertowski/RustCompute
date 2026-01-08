//! `ringkernel check` command - Validate kernel compatibility across backends.

use std::fs;
use std::path::Path;

use colored::Colorize;
use walkdir::WalkDir;

use crate::error::{CliError, CliResult};

use super::parse_backends;

/// Execute the `check` command.
pub async fn execute(path: &str, backends: &str, detailed: bool) -> CliResult<()> {
    let source_path = Path::new(path);

    if !source_path.exists() {
        return Err(CliError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Path not found: {}", path),
        )));
    }

    let backend_list = parse_backends(backends);

    println!("{} Checking kernel compatibility", "→".bright_cyan());
    println!("  {} Path: {}", "•".dimmed(), path.bright_yellow());
    println!(
        "  {} Backends: {}",
        "•".dimmed(),
        backend_list.join(", ").bright_yellow()
    );
    println!();

    // Find all Rust source files
    let mut kernel_files = Vec::new();
    for entry in WalkDir::new(source_path).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().map(|e| e == "rs").unwrap_or(false) {
            kernel_files.push(entry.path().to_path_buf());
        }
    }

    if kernel_files.is_empty() {
        println!(
            "{} No Rust source files found in {}",
            "Warning:".yellow(),
            path.bright_white()
        );
        return Ok(());
    }

    // Analyze each file for kernels
    let mut total_kernels = 0;
    let mut compatible_counts: std::collections::HashMap<String, usize> =
        backend_list.iter().map(|b| (b.clone(), 0)).collect();
    let mut issues: Vec<CompatibilityIssue> = Vec::new();

    for file_path in &kernel_files {
        let content = fs::read_to_string(file_path)?;

        // Parse the file
        if let Ok(syntax_tree) = syn::parse_file(&content) {
            let file_kernels = analyze_file(&syntax_tree, file_path, &backend_list, detailed);

            for kernel_result in file_kernels {
                total_kernels += 1;

                for (backend, compatible) in &kernel_result.backend_compatibility {
                    if *compatible {
                        *compatible_counts.get_mut(backend).unwrap() += 1;
                    }
                }

                issues.extend(kernel_result.issues);
            }
        }
    }

    // Print results
    println!("{}:", "Compatibility Report".bright_white().underline());
    println!();

    println!(
        "  {} kernel(s) found in {} file(s)",
        total_kernels.to_string().bright_white(),
        kernel_files.len().to_string().bright_white()
    );
    println!();

    println!("  Backend Compatibility:");
    for backend in &backend_list {
        let count = compatible_counts.get(backend).unwrap_or(&0);
        let percentage = if total_kernels > 0 {
            (*count as f64 / total_kernels as f64) * 100.0
        } else {
            0.0
        };

        let status = if *count == total_kernels {
            "✓".bright_green()
        } else if *count > 0 {
            "⚠".yellow()
        } else {
            "✗".bright_red()
        };

        println!(
            "    {} {} {}/{} ({:.0}%)",
            status,
            format!("{:>6}:", backend).bright_white(),
            count,
            total_kernels,
            percentage
        );
    }

    // Print issues if any
    if !issues.is_empty() {
        println!();
        println!("{}:", "Issues".bright_red().underline());

        for issue in &issues {
            println!(
                "    {} {} in {} ({}:{})",
                match issue.severity {
                    Severity::Error => "✗".bright_red(),
                    Severity::Warning => "⚠".yellow(),
                    Severity::Info => "ℹ".bright_cyan(),
                },
                issue.message.bright_white(),
                issue.kernel_name.yellow(),
                issue.file.display(),
                issue.line
            );

            if detailed {
                if let Some(suggestion) = &issue.suggestion {
                    println!(
                        "      {} {}",
                        "Suggestion:".dimmed(),
                        suggestion.bright_white()
                    );
                }
            }
        }
    }

    println!();

    // Summary
    let all_compatible = issues.iter().all(|i| i.severity != Severity::Error);
    if all_compatible {
        println!(
            "{} All kernels are compatible with selected backends!",
            "✓".bright_green().bold()
        );
    } else {
        println!(
            "{} Some kernels have compatibility issues",
            "✗".bright_red().bold()
        );
    }

    Ok(())
}

/// Compatibility issue severity.
#[derive(Debug, PartialEq)]
enum Severity {
    Error,
    Warning,
    Info,
}

/// A compatibility issue found during analysis.
#[derive(Debug)]
struct CompatibilityIssue {
    file: std::path::PathBuf,
    line: usize,
    kernel_name: String,
    message: String,
    severity: Severity,
    suggestion: Option<String>,
}

/// Result of analyzing a kernel.
#[derive(Debug)]
#[allow(dead_code)]
struct KernelAnalysisResult {
    name: String,
    backend_compatibility: std::collections::HashMap<String, bool>,
    issues: Vec<CompatibilityIssue>,
}

/// Analyze a file for kernel definitions and their compatibility.
fn analyze_file(
    syntax_tree: &syn::File,
    file_path: &Path,
    backends: &[String],
    detailed: bool,
) -> Vec<KernelAnalysisResult> {
    let mut results = Vec::new();

    for item in &syntax_tree.items {
        if let syn::Item::Fn(func) = item {
            // Check for ring_kernel attribute
            for attr in &func.attrs {
                if attr.path().is_ident("ring_kernel") {
                    let name = func.sig.ident.to_string();
                    let mut compatibility: std::collections::HashMap<String, bool> =
                        backends.iter().map(|b| (b.clone(), true)).collect();
                    let mut issues = Vec::new();

                    // Check for features that might not be compatible
                    let analysis = analyze_kernel_features(func);

                    // Check WGSL compatibility
                    if backends.contains(&"wgsl".to_string()) {
                        if analysis.uses_f64 {
                            compatibility.insert("wgsl".to_string(), false);
                            issues.push(CompatibilityIssue {
                                file: file_path.to_path_buf(),
                                line: 0, // Line info not available without span-locations
                                kernel_name: name.clone(),
                                message: "Uses f64 (not supported in WGSL)".to_string(),
                                severity: Severity::Error,
                                suggestion: Some("Convert f64 to f32 or use emulation".to_string()),
                            });
                        }

                        if analysis.uses_64bit_atomics {
                            issues.push(CompatibilityIssue {
                                file: file_path.to_path_buf(),
                                line: 0,
                                kernel_name: name.clone(),
                                message: "Uses 64-bit atomics (emulated in WGSL)".to_string(),
                                severity: Severity::Warning,
                                suggestion: Some("Performance may be reduced".to_string()),
                            });
                        }

                        if analysis.uses_cooperative_groups {
                            compatibility.insert("wgsl".to_string(), false);
                            issues.push(CompatibilityIssue {
                                file: file_path.to_path_buf(),
                                line: 0,
                                kernel_name: name.clone(),
                                message: "Uses cooperative groups (not available in WGSL)"
                                    .to_string(),
                                severity: Severity::Error,
                                suggestion: Some(
                                    "Remove grid-wide synchronization or use workgroup sync"
                                        .to_string(),
                                ),
                            });
                        }
                    }

                    // Check MSL compatibility
                    if backends.contains(&"msl".to_string()) && analysis.uses_cooperative_groups {
                        issues.push(CompatibilityIssue {
                            file: file_path.to_path_buf(),
                            line: 0,
                            kernel_name: name.clone(),
                            message: "Uses cooperative groups (limited in Metal)".to_string(),
                            severity: Severity::Warning,
                            suggestion: Some("Use threadgroup_barrier instead".to_string()),
                        });
                    }

                    if detailed && issues.is_empty() {
                        // Add info about features used
                        if analysis.is_persistent {
                            issues.push(CompatibilityIssue {
                                file: file_path.to_path_buf(),
                                line: 0,
                                kernel_name: name.clone(),
                                message: "Persistent kernel mode".to_string(),
                                severity: Severity::Info,
                                suggestion: None,
                            });
                        }
                    }

                    results.push(KernelAnalysisResult {
                        name,
                        backend_compatibility: compatibility,
                        issues,
                    });
                    break;
                }
            }
        }
    }

    results
}

/// Features detected in a kernel.
#[derive(Debug, Default)]
struct KernelFeatures {
    uses_f64: bool,
    uses_64bit_atomics: bool,
    uses_cooperative_groups: bool,
    is_persistent: bool,
}

/// Analyze kernel function for feature usage.
fn analyze_kernel_features(func: &syn::ItemFn) -> KernelFeatures {
    let mut features = KernelFeatures::default();

    // Check attributes for persistent mode
    for attr in &func.attrs {
        if attr.path().is_ident("ring_kernel") {
            if let Ok(nested) = attr.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            ) {
                for meta in nested {
                    if let syn::Meta::NameValue(nv) = meta {
                        if nv.path.is_ident("mode") {
                            if let syn::Expr::Lit(syn::ExprLit {
                                lit: syn::Lit::Str(s),
                                ..
                            }) = &nv.value
                            {
                                if s.value() == "persistent" {
                                    features.is_persistent = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Check function body for feature usage
    let code = quote::quote!(#func).to_string();

    // Simple pattern matching for now
    if code.contains("f64") {
        features.uses_f64 = true;
    }

    if code.contains("AtomicU64") || code.contains("atomic_u64") || code.contains("atomic64") {
        features.uses_64bit_atomics = true;
    }

    if code.contains("grid.sync") || code.contains("cg::grid_group") || code.contains("grid_sync") {
        features.uses_cooperative_groups = true;
    }

    features
}
