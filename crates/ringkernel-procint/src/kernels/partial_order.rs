//! Partial order derivation kernel.
//!
//! Builds precedence matrices from interval events using GPU acceleration.

use crate::cuda::{
    generate_partial_order_kernel, CpuFallbackExecutor, ExecutionResult, GpuStats, GpuStatus,
    KernelExecutor,
};
use crate::models::{GpuObjectEvent, GpuPartialOrderTrace};

/// Partial order derivation kernel.
pub struct PartialOrderKernel {
    /// Kernel executor.
    executor: KernelExecutor,
    /// Use GPU if available.
    use_gpu: bool,
    /// Whether kernel is compiled.
    kernel_compiled: bool,
}

impl Default for PartialOrderKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialOrderKernel {
    /// Create a new partial order kernel.
    pub fn new() -> Self {
        let mut kernel = Self {
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
            let source = generate_partial_order_kernel();
            match self.executor.compile(&source) {
                Ok(_) => {
                    log::info!("Partial order CUDA kernel compiled successfully");
                    self.kernel_compiled = true;
                }
                Err(e) => {
                    log::warn!("Partial order CUDA kernel compilation failed: {}", e);
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

    /// Derive partial order traces from events.
    pub fn derive(&mut self, events: &[GpuObjectEvent]) -> PartialOrderResult {
        let start = std::time::Instant::now();

        // Try GPU path first if available and compiled
        #[cfg(feature = "cuda")]
        let (gpu_traces, exec_result) = if self.is_using_gpu() {
            match self.executor.execute_partial_order_gpu(events) {
                Ok((traces, result)) => {
                    log::debug!(
                        "Partial order GPU execution: {} events -> {} traces in {}µs",
                        events.len(),
                        traces.len(),
                        result.execution_time_us
                    );
                    (Some(traces), result)
                }
                Err(e) => {
                    log::warn!(
                        "Partial order GPU execution failed, falling back to CPU: {}",
                        e
                    );
                    (None, ExecutionResult::default())
                }
            }
        } else {
            (None, ExecutionResult::default())
        };

        #[cfg(not(feature = "cuda"))]
        let gpu_traces: Option<Vec<GpuPartialOrderTrace>> = None;
        #[cfg(not(feature = "cuda"))]
        let exec_result = ExecutionResult::default();

        // Use GPU results or fall back to CPU
        let (traces, exec_result) = if let Some(gpu_traces) = gpu_traces {
            (gpu_traces, exec_result)
        } else {
            // CPU fallback path
            let mut traces = Vec::new();
            let result = CpuFallbackExecutor::execute_partial_order(events, &mut traces);
            (traces, result)
        };

        let total_time = start.elapsed().as_micros() as u64;

        PartialOrderResult {
            traces,
            execution_result: exec_result,
            total_time_us: total_time,
        }
    }
}

/// Result of partial order derivation.
#[derive(Debug)]
pub struct PartialOrderResult {
    /// Derived partial order traces.
    pub traces: Vec<GpuPartialOrderTrace>,
    /// Kernel execution result.
    pub execution_result: ExecutionResult,
    /// Total processing time in microseconds.
    pub total_time_us: u64,
}

impl PartialOrderResult {
    /// Get total traces.
    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }

    /// Get average width (concurrency level).
    pub fn avg_width(&self) -> f32 {
        if self.traces.is_empty() {
            return 0.0;
        }
        let total: u32 = self.traces.iter().map(|t| t.max_width).sum();
        total as f32 / self.traces.len() as f32
    }

    /// Get traces with cycles.
    pub fn traces_with_cycles(&self) -> Vec<&GpuPartialOrderTrace> {
        self.traces.iter().filter(|t| t.has_cycles()).collect()
    }

    /// Get traces that are total orders.
    pub fn total_order_traces(&self) -> Vec<&GpuPartialOrderTrace> {
        self.traces.iter().filter(|t| t.is_total_order()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::HybridTimestamp;

    fn create_test_events() -> Vec<GpuObjectEvent> {
        // Case with sequential activities: A(0-100) -> B(100-200) -> C(200-300)
        vec![
            GpuObjectEvent {
                event_id: 1,
                object_id: 100,
                activity_id: 1,
                timestamp: HybridTimestamp::new(0, 0),
                duration_ms: 100,
                ..Default::default()
            },
            GpuObjectEvent {
                event_id: 2,
                object_id: 100,
                activity_id: 2,
                timestamp: HybridTimestamp::new(100, 0),
                duration_ms: 100,
                ..Default::default()
            },
            GpuObjectEvent {
                event_id: 3,
                object_id: 100,
                activity_id: 3,
                timestamp: HybridTimestamp::new(200, 0),
                duration_ms: 100,
                ..Default::default()
            },
        ]
    }

    #[test]
    fn test_partial_order_derivation() {
        let mut kernel = PartialOrderKernel::new().with_cpu_only();
        let events = create_test_events();
        let result = kernel.derive(&events);

        assert_eq!(result.trace_count(), 1);

        let trace = &result.traces[0];
        assert_eq!(trace.activity_count, 3);

        // A precedes B, B precedes C
        assert!(trace.precedes(0, 1));
        assert!(trace.precedes(1, 2));

        // After transitive closure, A precedes C
        assert!(trace.precedes(0, 2));

        // Should be a total order
        assert!(trace.is_total_order());
    }

    #[test]
    fn test_concurrent_activities() {
        // Case with concurrent activities: A(0-100), B(50-150) (overlap = concurrent)
        let events = vec![
            GpuObjectEvent {
                event_id: 1,
                object_id: 100,
                activity_id: 1,
                timestamp: HybridTimestamp::new(0, 0),
                duration_ms: 100,
                ..Default::default()
            },
            GpuObjectEvent {
                event_id: 2,
                object_id: 100,
                activity_id: 2,
                timestamp: HybridTimestamp::new(50, 0),
                duration_ms: 100,
                ..Default::default()
            },
        ];

        let mut kernel = PartialOrderKernel::new().with_cpu_only();
        let result = kernel.derive(&events);

        let trace = &result.traces[0];
        // Neither precedes the other (concurrent)
        assert!(trace.is_concurrent(0, 1));
    }
}
