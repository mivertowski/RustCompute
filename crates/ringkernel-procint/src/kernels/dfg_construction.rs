//! DFG construction kernel.
//!
//! Builds Directly-Follows Graph from event streams using GPU acceleration.

use crate::cuda::{
    generate_dfg_kernel, CpuFallbackExecutor, ExecutionResult, GpuStats, GpuStatus, KernelExecutor,
};
use crate::models::{DFGGraph, GpuDFGEdge, GpuDFGNode, GpuObjectEvent};

/// DFG construction kernel.
pub struct DfgConstructionKernel {
    /// Maximum number of activities.
    max_activities: usize,
    /// Kernel executor.
    executor: KernelExecutor,
    /// Use GPU if available.
    use_gpu: bool,
    /// Whether kernel is compiled.
    kernel_compiled: bool,
}

impl Default for DfgConstructionKernel {
    fn default() -> Self {
        Self::new(64)
    }
}

impl DfgConstructionKernel {
    /// Create a new DFG construction kernel.
    pub fn new(max_activities: usize) -> Self {
        let mut kernel = Self {
            max_activities,
            executor: KernelExecutor::new(),
            use_gpu: true,
            kernel_compiled: false,
        };

        // Try to compile the CUDA kernel at creation time
        kernel.try_compile_kernel();
        kernel
    }

    /// Try to compile the CUDA kernel.
    fn try_compile_kernel(&mut self) {
        if self.executor.is_cuda_available() && !self.kernel_compiled {
            let source = generate_dfg_kernel();
            match self.executor.compile(&source) {
                Ok(_) => {
                    log::info!("DFG CUDA kernel compiled successfully");
                    self.kernel_compiled = true;
                }
                Err(e) => {
                    log::warn!("DFG CUDA kernel compilation failed: {}", e);
                    self.kernel_compiled = false;
                }
            }
        }
    }

    /// Disable GPU (use CPU fallback).
    pub fn with_cpu_only(mut self) -> Self {
        self.use_gpu = false;
        self
    }

    /// Get GPU status.
    pub fn gpu_status(&self) -> GpuStatus {
        self.executor.gpu_status()
    }

    /// Get GPU stats.
    pub fn gpu_stats(&self) -> &GpuStats {
        &self.executor.stats
    }

    /// Check if GPU is being used.
    pub fn is_using_gpu(&self) -> bool {
        self.use_gpu && self.kernel_compiled && self.executor.is_cuda_available()
    }

    /// Process events and build DFG.
    pub fn process(&mut self, events: &[GpuObjectEvent]) -> DfgResult {
        let start = std::time::Instant::now();

        // Try GPU path first if available and compiled
        #[cfg(feature = "cuda")]
        let (gpu_edges, exec_result) = if self.is_using_gpu() {
            match self.executor.execute_dfg_gpu(events) {
                Ok((edges, result)) => {
                    log::debug!(
                        "DFG GPU execution: {} events -> {} edges in {}µs",
                        events.len(),
                        edges.len(),
                        result.execution_time_us
                    );
                    (Some(edges), result)
                }
                Err(e) => {
                    log::warn!("DFG GPU execution failed, falling back to CPU: {}", e);
                    (None, ExecutionResult::default())
                }
            }
        } else {
            (None, ExecutionResult::default())
        };

        #[cfg(not(feature = "cuda"))]
        let gpu_edges: Option<Vec<GpuDFGEdge>> = None;
        #[cfg(not(feature = "cuda"))]
        let exec_result = ExecutionResult::default();

        // Use GPU results or fall back to CPU
        let (edges, exec_result) = if let Some(gpu_edges) = gpu_edges {
            (gpu_edges, exec_result)
        } else {
            // CPU fallback path
            let edge_count = self.max_activities * self.max_activities;
            let mut edges = vec![GpuDFGEdge::default(); edge_count];
            let result = CpuFallbackExecutor::execute_dfg_construction(
                events,
                &mut edges,
                self.max_activities,
            );
            let active_edges: Vec<GpuDFGEdge> =
                edges.into_iter().filter(|e| e.frequency > 0).collect();
            (active_edges, result)
        };

        // Build node statistics
        let mut nodes = vec![GpuDFGNode::default(); self.max_activities];
        let mut activity_events: std::collections::HashMap<u32, Vec<&GpuObjectEvent>> =
            std::collections::HashMap::new();

        for event in events {
            activity_events
                .entry(event.activity_id)
                .or_default()
                .push(event);
        }

        for (activity_id, evts) in &activity_events {
            let idx = *activity_id as usize;
            if idx < nodes.len() {
                nodes[idx].activity_id = *activity_id;
                nodes[idx].event_count = evts.len() as u32;

                // Calculate duration stats
                let durations: Vec<u32> = evts.iter().map(|e| e.duration_ms).collect();
                if !durations.is_empty() {
                    nodes[idx].min_duration_ms = *durations.iter().min().unwrap();
                    nodes[idx].max_duration_ms = *durations.iter().max().unwrap();
                    nodes[idx].avg_duration_ms =
                        durations.iter().sum::<u32>() as f32 / durations.len() as f32;
                }
            }
        }

        // Calculate incoming/outgoing counts from edge list
        for edge in &edges {
            let src = edge.source_activity as usize;
            let tgt = edge.target_activity as usize;
            if src < nodes.len() {
                nodes[src].outgoing_count = nodes[src]
                    .outgoing_count
                    .saturating_add(edge.frequency.min(u16::MAX as u32) as u16);
            }
            if tgt < nodes.len() {
                nodes[tgt].incoming_count = nodes[tgt]
                    .incoming_count
                    .saturating_add(edge.frequency.min(u16::MAX as u32) as u16);
            }
        }

        // Build DFG
        let dfg = DFGGraph::from_gpu(nodes, edges);

        let total_time = start.elapsed().as_micros() as u64;

        DfgResult {
            dfg,
            execution_result: exec_result,
            total_time_us: total_time,
        }
    }
}

/// Result of DFG construction.
#[derive(Debug)]
pub struct DfgResult {
    /// Constructed DFG.
    pub dfg: DFGGraph,
    /// Kernel execution result.
    pub execution_result: ExecutionResult,
    /// Total processing time in microseconds.
    pub total_time_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::HybridTimestamp;

    fn create_test_events() -> Vec<GpuObjectEvent> {
        let mut events = Vec::new();

        // Case 1: A -> B -> C
        for (i, activity_id) in [1u32, 2, 3].iter().enumerate() {
            events.push(GpuObjectEvent {
                event_id: i as u64,
                object_id: 100, // case ID
                activity_id: *activity_id,
                timestamp: HybridTimestamp::new(i as u64 * 1000, 0),
                duration_ms: 1000,
                ..Default::default()
            });
        }

        // Case 2: A -> B -> C
        for (i, activity_id) in [1u32, 2, 3].iter().enumerate() {
            events.push(GpuObjectEvent {
                event_id: (i + 3) as u64,
                object_id: 101, // case ID
                activity_id: *activity_id,
                timestamp: HybridTimestamp::new(i as u64 * 1000, 0),
                duration_ms: 1000,
                ..Default::default()
            });
        }

        events
    }

    #[test]
    fn test_dfg_construction() {
        let mut kernel = DfgConstructionKernel::new(10).with_cpu_only();
        let events = create_test_events();
        let result = kernel.process(&events);

        // Should have edges A->B and B->C
        assert!(result.dfg.edge_count() > 0);
    }
}
