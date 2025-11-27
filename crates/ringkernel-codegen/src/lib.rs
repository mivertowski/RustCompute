//! Code Generation for RingKernel
//!
//! This crate generates GPU kernel source code (CUDA PTX, Metal MSL, WGSL)
//! from Rust kernel definitions.
//!
//! # Supported Targets
//!
//! - CUDA PTX (sm_70+)
//! - Metal MSL
//! - WebGPU WGSL
//!
//! # Example
//!
//! ```
//! use ringkernel_codegen::{CodeGenerator, Target};
//!
//! let generator = CodeGenerator::new();
//! let source = generator.generate_kernel_source(
//!     "my_kernel",
//!     "// custom kernel code",
//!     Target::Cuda,
//! );
//! ```

#![warn(missing_docs)]

use std::collections::HashMap;
use thiserror::Error;

/// Code generation errors.
#[derive(Error, Debug)]
pub enum CodegenError {
    /// Template error.
    #[error("template error: {0}")]
    TemplateError(String),

    /// Unsupported target.
    #[error("unsupported target: {0}")]
    UnsupportedTarget(String),

    /// Invalid kernel definition.
    #[error("invalid kernel: {0}")]
    InvalidKernel(String),
}

/// Code generation result type.
pub type Result<T> = std::result::Result<T, CodegenError>;

/// Target GPU platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Target {
    /// NVIDIA CUDA (PTX).
    Cuda,
    /// Apple Metal (MSL).
    Metal,
    /// WebGPU (WGSL).
    Wgsl,
}

impl Target {
    /// Get file extension for the target.
    pub fn extension(&self) -> &'static str {
        match self {
            Target::Cuda => "ptx",
            Target::Metal => "metal",
            Target::Wgsl => "wgsl",
        }
    }

    /// Get target name.
    pub fn name(&self) -> &'static str {
        match self {
            Target::Cuda => "CUDA",
            Target::Metal => "Metal",
            Target::Wgsl => "WebGPU",
        }
    }
}

/// Kernel configuration.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Kernel identifier.
    pub id: String,
    /// Grid size (blocks).
    pub grid_size: u32,
    /// Block size (threads).
    pub block_size: u32,
    /// Shared memory size in bytes.
    pub shared_memory: usize,
    /// Input message types.
    pub input_types: Vec<String>,
    /// Output message types.
    pub output_types: Vec<String>,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            id: "kernel".to_string(),
            grid_size: 1,
            block_size: 256,
            shared_memory: 0,
            input_types: vec![],
            output_types: vec![],
        }
    }
}

/// Code generator for GPU kernels.
pub struct CodeGenerator {
    /// Template variables.
    variables: HashMap<String, String>,
}

impl CodeGenerator {
    /// Create a new code generator.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Set a template variable.
    pub fn set_variable(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.variables.insert(key.into(), value.into());
    }

    /// Generate kernel source code for the specified target.
    pub fn generate_kernel_source(
        &self,
        kernel_id: &str,
        user_code: &str,
        target: Target,
    ) -> Result<String> {
        let template = self.get_template(target);
        let source = self.substitute_template(template, kernel_id, user_code);
        Ok(source)
    }

    /// Generate complete kernel file.
    pub fn generate_kernel_file(
        &self,
        config: &KernelConfig,
        user_code: &str,
        target: Target,
    ) -> Result<GeneratedFile> {
        let source = self.generate_kernel_source(&config.id, user_code, target)?;
        Ok(GeneratedFile {
            filename: format!("{}.{}", config.id, target.extension()),
            content: source,
            target,
        })
    }

    /// Generate for all targets.
    pub fn generate_all_targets(
        &self,
        config: &KernelConfig,
        user_code: &str,
    ) -> Result<Vec<GeneratedFile>> {
        let targets = [Target::Cuda, Target::Metal, Target::Wgsl];
        let mut files = Vec::with_capacity(targets.len());

        for target in targets {
            files.push(self.generate_kernel_file(config, user_code, target)?);
        }

        Ok(files)
    }

    fn get_template(&self, target: Target) -> &'static str {
        match target {
            Target::Cuda => include_str!("templates/cuda.ptx.template"),
            Target::Metal => include_str!("templates/metal.msl.template"),
            Target::Wgsl => include_str!("templates/wgsl.template"),
        }
    }

    fn substitute_template(&self, template: &str, kernel_id: &str, user_code: &str) -> String {
        let mut result = template.to_string();
        result = result.replace("{{KERNEL_ID}}", kernel_id);
        result = result.replace("{{USER_CODE}}", user_code);
        result = result.replace("// USER_KERNEL_CODE", user_code);

        // Apply custom variables
        for (key, value) in &self.variables {
            result = result.replace(&format!("{{{{{}}}}}", key), value);
        }

        result
    }
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Generated kernel file.
#[derive(Debug, Clone)]
pub struct GeneratedFile {
    /// Output filename.
    pub filename: String,
    /// Generated source code.
    pub content: String,
    /// Target platform.
    pub target: Target,
}

/// Intrinsic mapping from Rust to GPU code.
#[derive(Debug, Clone)]
pub struct IntrinsicMap {
    /// Rust function name.
    pub rust_name: String,
    /// CUDA equivalent.
    pub cuda: String,
    /// Metal equivalent.
    pub metal: String,
    /// WGSL equivalent.
    pub wgsl: String,
}

impl IntrinsicMap {
    /// Get intrinsic for the specified target.
    pub fn get(&self, target: Target) -> &str {
        match target {
            Target::Cuda => &self.cuda,
            Target::Metal => &self.metal,
            Target::Wgsl => &self.wgsl,
        }
    }
}

/// Standard intrinsic mappings.
pub fn standard_intrinsics() -> Vec<IntrinsicMap> {
    vec![
        IntrinsicMap {
            rust_name: "sync_threads".to_string(),
            cuda: "__syncthreads()".to_string(),
            metal: "threadgroup_barrier(mem_flags::mem_threadgroup)".to_string(),
            wgsl: "workgroupBarrier()".to_string(),
        },
        IntrinsicMap {
            rust_name: "thread_fence_block".to_string(),
            cuda: "__threadfence_block()".to_string(),
            metal: "threadgroup_barrier(mem_flags::mem_device)".to_string(),
            wgsl: "storageBarrier()".to_string(),
        },
        IntrinsicMap {
            rust_name: "thread_fence".to_string(),
            cuda: "__threadfence()".to_string(),
            metal: "threadgroup_barrier(mem_flags::mem_device)".to_string(),
            wgsl: "storageBarrier()".to_string(),
        },
        IntrinsicMap {
            rust_name: "atomic_add".to_string(),
            cuda: "atomicAdd".to_string(),
            metal: "atomic_fetch_add_explicit".to_string(),
            wgsl: "atomicAdd".to_string(),
        },
        IntrinsicMap {
            rust_name: "atomic_cas".to_string(),
            cuda: "atomicCAS".to_string(),
            metal: "atomic_compare_exchange_weak_explicit".to_string(),
            wgsl: "atomicCompareExchangeWeak".to_string(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_generator() {
        let gen = CodeGenerator::new();
        let source = gen
            .generate_kernel_source("test_kernel", "// test code", Target::Cuda)
            .unwrap();

        assert!(source.contains("test_kernel") || source.contains("ring_kernel"));
    }

    #[test]
    fn test_target_extension() {
        assert_eq!(Target::Cuda.extension(), "ptx");
        assert_eq!(Target::Metal.extension(), "metal");
        assert_eq!(Target::Wgsl.extension(), "wgsl");
    }

    #[test]
    fn test_intrinsic_mapping() {
        let intrinsics = standard_intrinsics();
        let sync = intrinsics.iter().find(|i| i.rust_name == "sync_threads").unwrap();

        assert_eq!(sync.get(Target::Cuda), "__syncthreads()");
        assert!(sync.get(Target::Metal).contains("barrier"));
    }
}
