//! Ring kernel generation for WGSL.
//!
//! Generates WGSL compute shaders for ring kernel message processing.
//! Note: WebGPU does not support true persistent kernels, so ring kernels
//! are emulated using host-driven dispatch loops.

/// Configuration for ring kernel generation.
#[derive(Debug, Clone)]
pub struct RingKernelConfig {
    /// Kernel name.
    pub name: String,
    /// Workgroup size (number of threads).
    pub workgroup_size: u32,
    /// Enable hybrid logical clock support.
    pub enable_hlc: bool,
    /// Enable kernel-to-kernel messaging (NOT SUPPORTED in WGPU).
    pub enable_k2k: bool,
    /// Maximum messages per dispatch.
    pub max_messages_per_dispatch: u32,
}

impl RingKernelConfig {
    /// Create a new ring kernel configuration.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            workgroup_size: 256,
            enable_hlc: false,
            enable_k2k: false,
            max_messages_per_dispatch: 1024,
        }
    }

    /// Set workgroup size.
    pub fn with_workgroup_size(mut self, size: u32) -> Self {
        self.workgroup_size = size;
        self
    }

    /// Enable HLC support.
    pub fn with_hlc(mut self, enable: bool) -> Self {
        self.enable_hlc = enable;
        self
    }

    /// Enable K2K support (will error during transpilation - not supported in WGPU).
    pub fn with_k2k(mut self, enable: bool) -> Self {
        self.enable_k2k = enable;
        self
    }

    /// Set maximum messages per dispatch.
    pub fn with_max_messages(mut self, max: u32) -> Self {
        self.max_messages_per_dispatch = max;
        self
    }

    /// Get the workgroup size annotation.
    pub fn workgroup_size_annotation(&self) -> String {
        format!("@workgroup_size({}, 1, 1)", self.workgroup_size)
    }
}

/// Generate the WGSL ControlBlock struct definition.
pub fn generate_control_block_struct(config: &RingKernelConfig) -> String {
    let mut fields = vec![
        "    is_active: atomic<u32>,".to_string(),
        "    should_terminate: atomic<u32>,".to_string(),
        "    has_terminated: atomic<u32>,".to_string(),
        "    // 64-bit counters as lo/hi pairs".to_string(),
        "    messages_processed_lo: atomic<u32>,".to_string(),
        "    messages_processed_hi: atomic<u32>,".to_string(),
        "    messages_pending_lo: atomic<u32>,".to_string(),
        "    messages_pending_hi: atomic<u32>,".to_string(),
    ];

    if config.enable_hlc {
        fields.push("    // HLC timestamp".to_string());
        fields.push("    hlc_physical_lo: atomic<u32>,".to_string());
        fields.push("    hlc_physical_hi: atomic<u32>,".to_string());
        fields.push("    hlc_logical: atomic<u32>,".to_string());
    }

    format!("struct ControlBlock {{\n{}\n}}", fields.join("\n"))
}

/// Generate the 64-bit helper functions.
pub fn generate_u64_helpers() -> &'static str {
    r#"
// 64-bit operations using lo/hi u32 pairs
fn read_u64(lo: ptr<storage, atomic<u32>, read_write>, hi: ptr<storage, atomic<u32>, read_write>) -> vec2<u32> {
    return vec2<u32>(atomicLoad(lo), atomicLoad(hi));
}

fn atomic_inc_u64(lo: ptr<storage, atomic<u32>, read_write>, hi: ptr<storage, atomic<u32>, read_write>) {
    let old_lo = atomicAdd(lo, 1u);
    if (old_lo == 0xFFFFFFFFu) {
        atomicAdd(hi, 1u);
    }
}

fn atomic_add_u64(lo: ptr<storage, atomic<u32>, read_write>, hi: ptr<storage, atomic<u32>, read_write>, addend: u32) {
    let old_lo = atomicAdd(lo, addend);
    if (old_lo > 0xFFFFFFFFu - addend) {
        atomicAdd(hi, 1u);
    }
}

fn compare_u64(a: vec2<u32>, b: vec2<u32>) -> i32 {
    if (a.y > b.y) { return 1; }
    if (a.y < b.y) { return -1; }
    if (a.x > b.x) { return 1; }
    if (a.x < b.x) { return -1; }
    return 0;
}
"#
}

/// Generate standard ring kernel bindings.
pub fn generate_ring_kernel_bindings() -> &'static str {
    r#"@group(0) @binding(0) var<storage, read_write> control: ControlBlock;
@group(0) @binding(1) var<storage, read_write> input_queue: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_queue: array<u32>;"#
}

/// Generate the ring kernel preamble (activation/termination checks).
pub fn generate_ring_kernel_preamble() -> &'static str {
    r#"    // Check if kernel is active
    if (atomicLoad(&control.is_active) == 0u) {
        return;
    }

    // Check for termination request
    if (atomicLoad(&control.should_terminate) != 0u) {
        if (local_invocation_id.x == 0u) {
            atomicStore(&control.has_terminated, 1u);
        }
        return;
    }"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_kernel_config() {
        let config = RingKernelConfig::new("processor")
            .with_workgroup_size(128)
            .with_hlc(true);

        assert_eq!(config.name, "processor");
        assert_eq!(config.workgroup_size, 128);
        assert!(config.enable_hlc);
        assert!(!config.enable_k2k);
    }

    #[test]
    fn test_control_block_generation() {
        let config = RingKernelConfig::new("test").with_hlc(true);
        let wgsl = generate_control_block_struct(&config);

        assert!(wgsl.contains("is_active: atomic<u32>"));
        assert!(wgsl.contains("should_terminate: atomic<u32>"));
        assert!(wgsl.contains("hlc_physical_lo: atomic<u32>"));
    }

    #[test]
    fn test_workgroup_size_annotation() {
        let config = RingKernelConfig::new("test").with_workgroup_size(64);
        assert_eq!(
            config.workgroup_size_annotation(),
            "@workgroup_size(64, 1, 1)"
        );
    }
}
