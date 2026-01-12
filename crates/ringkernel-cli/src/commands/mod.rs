//! CLI command implementations.

pub mod check;
pub mod codegen;
pub mod init;
pub mod new_project;
pub mod profile;

use std::path::Path;

/// Parse a comma-separated backend list.
pub fn parse_backends(backends: &str) -> Vec<String> {
    if backends == "all" {
        vec!["cuda".to_string(), "wgsl".to_string(), "msl".to_string()]
    } else {
        backends
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    }
}

/// Validate that a project name is valid.
pub fn validate_project_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Project name cannot be empty".to_string());
    }

    if !name.chars().next().unwrap().is_alphabetic() && !name.starts_with('_') {
        return Err("Project name must start with a letter or underscore".to_string());
    }

    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        return Err(
            "Project name can only contain letters, numbers, underscores, and hyphens".to_string(),
        );
    }

    // Reserved names
    let reserved = ["test", "self", "super", "crate", "std", "core", "alloc"];
    if reserved.contains(&name) {
        return Err(format!("'{}' is a reserved name", name));
    }

    Ok(())
}

/// Find the workspace root by looking for Cargo.toml with [workspace].
#[allow(dead_code)]
pub fn find_workspace_root(start: &Path) -> Option<std::path::PathBuf> {
    let mut current = start.to_path_buf();

    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                if content.contains("[workspace]") {
                    return Some(current);
                }
            }
        }

        if !current.pop() {
            break;
        }
    }

    None
}
