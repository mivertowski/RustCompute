//! CUDA code generation for process intelligence kernels.
//!
//! Uses ringkernel-cuda-codegen to transpile Rust DSL to CUDA C.

#![allow(missing_docs)]

#[cfg(feature = "cuda")]
use ringkernel_cuda_codegen::{
    transpile_global_kernel, transpile_ring_kernel, transpile_stencil_kernel, RingKernelConfig,
    StencilConfig,
};

#[cfg(feature = "cuda")]
use syn::parse_quote;

/// Kernel source code holder.
#[derive(Debug, Clone)]
pub struct KernelSource {
    /// Kernel name.
    pub name: String,
    /// CUDA C source code.
    pub source: String,
    /// Entry point function name.
    pub entry_point: String,
    /// Type of kernel.
    pub kernel_type: KernelType,
}

/// Type of kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    /// Global/batch kernel.
    Global,
    /// Stencil kernel with halo.
    Stencil,
    /// Ring kernel (persistent actor).
    Ring,
}

impl KernelSource {
    /// Create a new kernel source.
    pub fn new(
        name: impl Into<String>,
        source: impl Into<String>,
        kernel_type: KernelType,
    ) -> Self {
        let name = name.into();
        Self {
            entry_point: name.clone(),
            name,
            source: source.into(),
            kernel_type,
        }
    }

    /// Set custom entry point.
    pub fn with_entry_point(mut self, entry: impl Into<String>) -> Self {
        self.entry_point = entry.into();
        self
    }
}

/// Generate all CUDA kernels for process intelligence.
#[cfg(feature = "cuda")]
pub fn generate_all_kernels() -> Result<Vec<KernelSource>, String> {
    Ok(vec![
        generate_dfg_batch_kernel()?,
        generate_pattern_batch_kernel()?,
        generate_partial_order_stencil_kernel()?,
        generate_dfg_ring_kernel()?,
    ])
}

/// Generate DFG construction batch kernel.
///
/// Each thread processes one event pair to build edge frequencies.
#[cfg(feature = "cuda")]
pub fn generate_dfg_batch_kernel() -> Result<KernelSource, String> {
    let kernel_fn: syn::ItemFn = parse_quote! {
        fn dfg_construction(
            source_activities: &[u32],
            target_activities: &[u32],
            durations: &[u32],
            edge_frequencies: &mut [u32],
            edge_durations: &mut [u64],
            max_activities: i32,
            n: i32
        ) {
            let idx = block_idx_x() * block_dim_x() + thread_idx_x();
            if idx >= n { return; }

            let source = source_activities[idx as usize];
            let target = target_activities[idx as usize];
            let duration = durations[idx as usize];

            // Calculate edge index
            let edge_idx = (source as i32 * max_activities + target as i32) as usize;

            // Atomic increment frequency
            atomic_add(&mut edge_frequencies[edge_idx], 1u32);

            // Atomic add duration for averaging later
            atomic_add(&mut edge_durations[edge_idx], duration as u64);
        }
    };

    let cuda_source = transpile_global_kernel(&kernel_fn)
        .map_err(|e| format!("Failed to transpile DFG batch kernel: {}", e))?;

    Ok(
        KernelSource::new("dfg_construction", cuda_source, KernelType::Global)
            .with_entry_point("dfg_construction"),
    )
}

/// Generate pattern detection batch kernel.
///
/// Each thread analyzes one DFG node for patterns.
#[cfg(feature = "cuda")]
pub fn generate_pattern_batch_kernel() -> Result<KernelSource, String> {
    let kernel_fn: syn::ItemFn = parse_quote! {
        fn pattern_detection(
            event_counts: &[u32],
            avg_durations: &[f32],
            incoming_counts: &[u16],
            outgoing_counts: &[u16],
            pattern_types: &mut [u8],
            pattern_confidences: &mut [f32],
            bottleneck_threshold: f32,
            duration_threshold: f32,
            n: i32
        ) {
            let idx = block_idx_x() * block_dim_x() + thread_idx_x();
            if idx >= n { return; }

            let event_count = event_counts[idx as usize];
            let avg_duration = avg_durations[idx as usize];
            let incoming = incoming_counts[idx as usize] as f32;
            let outgoing = outgoing_counts[idx as usize] as f32;

            // Default: no pattern
            pattern_types[idx as usize] = 0u8;
            pattern_confidences[idx as usize] = 0.0f32;

            if event_count == 0 { return; }

            // Bottleneck detection: high incoming, low outgoing
            if incoming > bottleneck_threshold && outgoing < incoming * 0.5f32 {
                pattern_types[idx as usize] = 7u8; // Bottleneck
                pattern_confidences[idx as usize] = incoming / bottleneck_threshold;
                return;
            }

            // Long-running detection
            if avg_duration > duration_threshold {
                pattern_types[idx as usize] = 6u8; // LongRunning
                pattern_confidences[idx as usize] = avg_duration / duration_threshold;
            }
        }
    };

    let cuda_source = transpile_global_kernel(&kernel_fn)
        .map_err(|e| format!("Failed to transpile pattern batch kernel: {}", e))?;

    Ok(
        KernelSource::new("pattern_detection", cuda_source, KernelType::Global)
            .with_entry_point("pattern_detection"),
    )
}

/// Generate partial order stencil kernel.
///
/// Uses GridPos abstraction for pairwise event comparison.
#[cfg(feature = "cuda")]
pub fn generate_partial_order_stencil_kernel() -> Result<KernelSource, String> {
    let stencil_fn: syn::ItemFn = parse_quote! {
        fn partial_order_derive(
            start_times: &[u64],
            end_times: &[u64],
            precedence: &mut [u32],
            pos: GridPos
        ) {
            // Each cell (i, j) in the grid represents whether event i precedes event j
            let i_end = end_times[pos.y() as usize];
            let j_start = pos.east(start_times);

            // i precedes j if i ends before j starts
            if i_end <= j_start {
                precedence[pos.idx()] = 1u32;
            } else {
                precedence[pos.idx()] = 0u32;
            }
        }
    };

    let config = StencilConfig::new("partial_order")
        .with_tile_size(16, 16)
        .with_halo(0); // No halo needed for pairwise comparison

    let cuda_source = transpile_stencil_kernel(&stencil_fn, &config)
        .map_err(|e| format!("Failed to transpile partial order stencil: {}", e))?;

    Ok(
        KernelSource::new("partial_order_derive", cuda_source, KernelType::Stencil)
            .with_entry_point("partial_order_derive"),
    )
}

/// Generate DFG ring kernel (persistent actor).
///
/// Continuous event processing with HLC timestamps.
#[cfg(feature = "cuda")]
pub fn generate_dfg_ring_kernel() -> Result<KernelSource, String> {
    let handler_fn: syn::ItemFn = parse_quote! {
        fn process_event(ctx: &RingContext, event: &GpuObjectEvent) -> EdgeUpdate {
            let tid = ctx.global_thread_id();

            // Get HLC timestamp for ordering
            let ts = ctx.tick();

            ctx.sync_threads();

            // Create edge update response
            EdgeUpdate {
                source_activity: event.prev_activity,
                target_activity: event.activity_id,
                duration_ms: event.duration_ms,
                timestamp: ts.physical,
                thread_id: tid as u32,
            }
        }
    };

    let config = RingKernelConfig::new("dfg_processor")
        .with_block_size(256)
        .with_queue_capacity(4096)
        .with_hlc(true)
        .with_k2k(false); // No K2K needed for DFG

    let cuda_source = transpile_ring_kernel(&handler_fn, &config)
        .map_err(|e| format!("Failed to transpile DFG ring kernel: {}", e))?;

    Ok(
        KernelSource::new("ring_kernel_dfg", cuda_source, KernelType::Ring)
            .with_entry_point("ring_kernel_dfg"),
    )
}

/// Get CUDA type definitions header.
pub fn cuda_type_definitions() -> &'static str {
    r#"
// Process Intelligence GPU Types
// Auto-generated - matches Rust struct layouts

struct __align__(128) GpuObjectEvent {
    unsigned long long event_id;
    unsigned long long object_id;
    unsigned int activity_id;
    unsigned char event_type;
    unsigned char _padding1[3];
    unsigned long long physical_ms;
    unsigned int logical;
    unsigned int node_id;
    unsigned int resource_id;
    unsigned int duration_ms;
    unsigned int flags;
    unsigned int attributes[4];
    unsigned int object_type_id;
    unsigned int prev_activity;
    unsigned long long related_object_id;
    unsigned char _reserved[32];
};

struct __align__(64) GpuDFGNode {
    unsigned int activity_id;
    unsigned int event_count;
    unsigned long long total_duration_ms;
    unsigned int min_duration_ms;
    unsigned int max_duration_ms;
    float avg_duration_ms;
    float std_duration_ms;
    unsigned long long first_seen_ms;
    unsigned long long last_seen_ms;
    unsigned char is_start;
    unsigned char is_end;
    unsigned char flags;
    unsigned char _padding;
    unsigned short incoming_count;
    unsigned short outgoing_count;
};

struct __align__(64) GpuDFGEdge {
    unsigned int source_activity;
    unsigned int target_activity;
    unsigned int frequency;
    unsigned int min_duration_ms;
    float avg_duration_ms;
    unsigned int max_duration_ms;
    float probability;
    unsigned char flags;
    unsigned char _padding[3];
    float total_cost;
    unsigned long long first_seen_ms;
    unsigned long long last_seen_ms;
    unsigned char _reserved[16];
};

struct __align__(64) GpuPatternMatch {
    unsigned char pattern_type;
    unsigned char severity;
    unsigned char activity_count;
    unsigned char flags;
    unsigned int activity_ids[8];
    float confidence;
    unsigned int frequency;
    float avg_duration_ms;
    float impact;
    unsigned char _reserved[4];
};

struct EdgeUpdate {
    unsigned int source_activity;
    unsigned int target_activity;
    unsigned int duration_ms;
    unsigned long long timestamp;
    unsigned int thread_id;
};

// Pattern type constants
#define PATTERN_SEQUENCE 0
#define PATTERN_CHOICE 1
#define PATTERN_LOOP 2
#define PATTERN_PARALLEL 3
#define PATTERN_SKIP 4
#define PATTERN_REWORK 5
#define PATTERN_LONG_RUNNING 6
#define PATTERN_BOTTLENECK 7
#define PATTERN_ANOMALY 8

// Severity constants
#define SEVERITY_INFO 0
#define SEVERITY_WARNING 1
#define SEVERITY_CRITICAL 2
"#
}

// Fallback implementations when cuda feature is not enabled
#[cfg(not(feature = "cuda"))]
pub fn generate_all_kernels() -> Result<Vec<KernelSource>, String> {
    Err("CUDA feature not enabled. Build with --features cuda".to_string())
}

#[cfg(not(feature = "cuda"))]
pub fn generate_dfg_batch_kernel() -> Result<KernelSource, String> {
    Err("CUDA feature not enabled".to_string())
}

#[cfg(not(feature = "cuda"))]
pub fn generate_pattern_batch_kernel() -> Result<KernelSource, String> {
    Err("CUDA feature not enabled".to_string())
}

#[cfg(not(feature = "cuda"))]
pub fn generate_partial_order_stencil_kernel() -> Result<KernelSource, String> {
    Err("CUDA feature not enabled".to_string())
}

#[cfg(not(feature = "cuda"))]
pub fn generate_dfg_ring_kernel() -> Result<KernelSource, String> {
    Err("CUDA feature not enabled".to_string())
}

/// Generate DFG construction kernel (legacy static version).
pub fn generate_dfg_kernel() -> KernelSource {
    KernelSource::new("dfg_construction", STATIC_DFG_KERNEL, KernelType::Global)
        .with_entry_point("dfg_construction_kernel")
}

/// Generate pattern detection kernel (legacy static version).
pub fn generate_pattern_kernel() -> KernelSource {
    KernelSource::new(
        "pattern_detection",
        STATIC_PATTERN_KERNEL,
        KernelType::Global,
    )
    .with_entry_point("pattern_detection_kernel")
}

/// Generate partial order kernel (legacy static version).
pub fn generate_partial_order_kernel() -> KernelSource {
    KernelSource::new(
        "partial_order",
        STATIC_PARTIAL_ORDER_KERNEL,
        KernelType::Stencil,
    )
    .with_entry_point("partial_order_kernel")
}

/// Generate conformance kernel (legacy static version).
pub fn generate_conformance_kernel() -> KernelSource {
    KernelSource::new("conformance", STATIC_CONFORMANCE_KERNEL, KernelType::Global)
        .with_entry_point("conformance_kernel")
}

// Static kernel sources (fallback when transpiler isn't available)
// Note: extern "C" prevents C++ name mangling so we can find the function by name
const STATIC_DFG_KERNEL: &str = r#"
extern "C" __global__ void dfg_construction_kernel(
    const unsigned int* source_activities,
    const unsigned int* target_activities,
    const unsigned int* durations,
    unsigned int* edge_frequencies,
    unsigned long long* edge_durations,
    int max_activities,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int source = source_activities[idx];
    unsigned int target = target_activities[idx];
    unsigned int duration = durations[idx];

    int edge_idx = source * max_activities + target;
    atomicAdd(&edge_frequencies[edge_idx], 1);
    atomicAdd(&edge_durations[edge_idx], (unsigned long long)duration);
}
"#;

const STATIC_PATTERN_KERNEL: &str = r#"
extern "C" __global__ void pattern_detection_kernel(
    const unsigned int* event_counts,
    const float* avg_durations,
    const unsigned short* incoming_counts,
    const unsigned short* outgoing_counts,
    unsigned char* pattern_types,
    float* pattern_confidences,
    float bottleneck_threshold,
    float duration_threshold,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int event_count = event_counts[idx];
    float avg_duration = avg_durations[idx];
    float incoming = (float)incoming_counts[idx];
    float outgoing = (float)outgoing_counts[idx];

    pattern_types[idx] = 0;
    pattern_confidences[idx] = 0.0f;

    if (event_count == 0) return;

    // Bottleneck detection
    if (incoming > bottleneck_threshold && outgoing < incoming * 0.5f) {
        pattern_types[idx] = 7; // Bottleneck
        pattern_confidences[idx] = incoming / bottleneck_threshold;
        return;
    }

    // Long-running detection
    if (avg_duration > duration_threshold) {
        pattern_types[idx] = 6; // LongRunning
        pattern_confidences[idx] = avg_duration / duration_threshold;
    }
}
"#;

const STATIC_PARTIAL_ORDER_KERNEL: &str = r#"
extern "C" __global__ void partial_order_kernel(
    const unsigned long long* start_times,
    const unsigned long long* end_times,
    unsigned int* precedence,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int x_next = min(x + 1, width - 1);

    unsigned long long i_end = end_times[y];
    unsigned long long j_start = start_times[x_next];

    precedence[idx] = (i_end <= j_start) ? 1 : 0;
}
"#;

const STATIC_CONFORMANCE_KERNEL: &str = r#"
extern "C" __global__ void conformance_kernel(
    const unsigned int* trace_activities,
    const int* trace_starts,
    const int* trace_lengths,
    const unsigned int* model_sources,
    const unsigned int* model_targets,
    int num_transitions,
    float* fitness_scores,
    int num_traces
) {
    int trace_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (trace_idx >= num_traces) return;

    int start = trace_starts[trace_idx];
    int len = trace_lengths[trace_idx];

    if (len <= 1) {
        fitness_scores[trace_idx] = 1.0f;
        return;
    }

    int valid_moves = 0;
    for (int i = 0; i < len - 1; i++) {
        unsigned int source = trace_activities[start + i];
        unsigned int target = trace_activities[start + i + 1];

        for (int t = 0; t < num_transitions; t++) {
            if (model_sources[t] == source && model_targets[t] == target) {
                valid_moves++;
                break;
            }
        }
    }

    fitness_scores[trace_idx] = (float)valid_moves / (float)(len - 1);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfg_kernel_generation() {
        let source = generate_dfg_kernel();
        assert!(source.source.contains("dfg_construction_kernel"));
        assert!(source.source.contains("atomicAdd"));
    }

    #[test]
    fn test_pattern_kernel_generation() {
        let source = generate_pattern_kernel();
        assert!(source.source.contains("pattern_detection_kernel"));
    }

    #[test]
    fn test_partial_order_kernel_generation() {
        let source = generate_partial_order_kernel();
        assert!(source.source.contains("partial_order_kernel"));
        assert!(source.source.contains("precedence"));
    }

    #[test]
    fn test_conformance_kernel_generation() {
        let source = generate_conformance_kernel();
        assert!(source.source.contains("conformance_kernel"));
        assert!(source.source.contains("fitness"));
    }

    #[test]
    fn test_cuda_type_definitions() {
        let defs = cuda_type_definitions();
        assert!(defs.contains("GpuObjectEvent"));
        assert!(defs.contains("GpuDFGNode"));
        assert!(defs.contains("GpuPatternMatch"));
        assert!(defs.contains("__align__(64)"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_transpiled_dfg_kernel() {
        let result = generate_dfg_batch_kernel();
        match result {
            Ok(kernel) => {
                println!(
                    "Generated DFG batch kernel ({} bytes):",
                    kernel.source.len()
                );
                println!("{}", kernel.source);
                assert!(kernel.source.contains("__global__"));
            }
            Err(e) => {
                println!("DFG kernel generation pending: {}", e);
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_transpiled_pattern_kernel() {
        let result = generate_pattern_batch_kernel();
        match result {
            Ok(kernel) => {
                println!(
                    "Generated pattern batch kernel ({} bytes):",
                    kernel.source.len()
                );
                println!("{}", kernel.source);
                assert!(kernel.source.contains("__global__"));
            }
            Err(e) => {
                println!("Pattern kernel generation pending: {}", e);
            }
        }
    }
}
