//! Buffer binding layout generation for WGSL.
//!
//! Generates @group/@binding declarations from kernel parameters.

use crate::types::WgslType;

/// Access mode for storage buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// Read-only access.
    Read,
    /// Write-only access (rare in WGSL).
    Write,
    /// Read-write access.
    ReadWrite,
}

impl AccessMode {
    /// Get the WGSL access mode string.
    pub fn to_wgsl(&self) -> &'static str {
        match self {
            AccessMode::Read => "read",
            AccessMode::Write => "write",
            AccessMode::ReadWrite => "read_write",
        }
    }
}

/// Description of a buffer binding.
#[derive(Debug, Clone)]
pub struct BindingLayout {
    /// Binding group (usually 0).
    pub group: u32,
    /// Binding number within the group.
    pub binding: u32,
    /// Variable name in the shader.
    pub name: String,
    /// Type of the binding.
    pub ty: WgslType,
    /// Access mode for storage buffers.
    pub access: AccessMode,
}

impl BindingLayout {
    /// Create a new binding layout.
    pub fn new(group: u32, binding: u32, name: &str, ty: WgslType, access: AccessMode) -> Self {
        Self {
            group,
            binding,
            name: name.to_string(),
            ty,
            access,
        }
    }

    /// Create a read-only storage buffer binding.
    pub fn storage_read(binding: u32, name: &str, element_type: WgslType) -> Self {
        Self::new(
            0,
            binding,
            name,
            WgslType::Array {
                element: Box::new(element_type),
                size: None,
            },
            AccessMode::Read,
        )
    }

    /// Create a read-write storage buffer binding.
    pub fn storage_read_write(binding: u32, name: &str, element_type: WgslType) -> Self {
        Self::new(
            0,
            binding,
            name,
            WgslType::Array {
                element: Box::new(element_type),
                size: None,
            },
            AccessMode::ReadWrite,
        )
    }

    /// Create a uniform buffer binding.
    pub fn uniform(binding: u32, name: &str, ty: WgslType) -> Self {
        Self::new(0, binding, name, ty, AccessMode::Read)
    }

    /// Generate the WGSL binding declaration.
    pub fn to_wgsl(&self) -> String {
        let type_str = self.ty.to_wgsl();

        match &self.ty {
            WgslType::Array { .. } => {
                // Storage buffer
                format!(
                    "@group({}) @binding({}) var<storage, {}> {}: {};",
                    self.group,
                    self.binding,
                    self.access.to_wgsl(),
                    self.name,
                    type_str
                )
            }
            WgslType::Struct(_) if self.access == AccessMode::Read => {
                // Uniform buffer
                format!(
                    "@group({}) @binding({}) var<uniform> {}: {};",
                    self.group, self.binding, self.name, type_str
                )
            }
            _ => {
                // Generic storage
                format!(
                    "@group({}) @binding({}) var<storage, {}> {}: {};",
                    self.group,
                    self.binding,
                    self.access.to_wgsl(),
                    self.name,
                    type_str
                )
            }
        }
    }
}

/// Generate binding declarations from a list of parameters.
pub fn generate_bindings(bindings: &[BindingLayout]) -> String {
    bindings
        .iter()
        .map(|b| b.to_wgsl())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Generate bindings for kernel parameters.
///
/// Slices become storage buffers, scalars become push constants or uniforms.
pub fn bindings_from_params(params: &[(String, WgslType, bool)]) -> Vec<BindingLayout> {
    let mut bindings = Vec::new();
    let mut binding_idx = 0u32;

    for (name, ty, is_mutable) in params {
        match ty {
            WgslType::Ptr { inner, .. } => {
                let access = if *is_mutable {
                    AccessMode::ReadWrite
                } else {
                    AccessMode::Read
                };
                bindings.push(BindingLayout::new(
                    0,
                    binding_idx,
                    name,
                    WgslType::Array {
                        element: inner.clone(),
                        size: None,
                    },
                    access,
                ));
                binding_idx += 1;
            }
            WgslType::Array { element, .. } => {
                let access = if *is_mutable {
                    AccessMode::ReadWrite
                } else {
                    AccessMode::Read
                };
                bindings.push(BindingLayout::new(
                    0,
                    binding_idx,
                    name,
                    WgslType::Array {
                        element: element.clone(),
                        size: None,
                    },
                    access,
                ));
                binding_idx += 1;
            }
            // Scalars are typically passed via uniforms or push constants
            // For now, we'll add them as uniform buffer fields
            _ => {
                bindings.push(BindingLayout::new(
                    0,
                    binding_idx,
                    name,
                    ty.clone(),
                    AccessMode::Read,
                ));
                binding_idx += 1;
            }
        }
    }

    bindings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_read_binding() {
        let binding = BindingLayout::storage_read(0, "input", WgslType::F32);
        assert_eq!(
            binding.to_wgsl(),
            "@group(0) @binding(0) var<storage, read> input: array<f32>;"
        );
    }

    #[test]
    fn test_storage_read_write_binding() {
        let binding = BindingLayout::storage_read_write(1, "output", WgslType::F32);
        assert_eq!(
            binding.to_wgsl(),
            "@group(0) @binding(1) var<storage, read_write> output: array<f32>;"
        );
    }

    #[test]
    fn test_generate_bindings() {
        let bindings = vec![
            BindingLayout::storage_read(0, "input", WgslType::F32),
            BindingLayout::storage_read_write(1, "output", WgslType::F32),
        ];

        let wgsl = generate_bindings(&bindings);
        assert!(wgsl.contains("@binding(0)"));
        assert!(wgsl.contains("@binding(1)"));
        assert!(wgsl.contains("read>"));
        assert!(wgsl.contains("read_write>"));
    }
}
