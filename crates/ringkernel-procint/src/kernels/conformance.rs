//! Conformance checking kernel.
//!
//! Validates traces against reference process models using GPU acceleration.

use crate::cuda::{
    generate_conformance_kernel, CpuFallbackExecutor, ExecutionResult, GpuStats, GpuStatus,
    KernelExecutor,
};
use crate::models::{ComplianceLevel, ConformanceResult, GpuObjectEvent, ProcessModel};

/// Conformance checking kernel.
pub struct ConformanceKernel {
    /// Reference model.
    model: ProcessModel,
    /// Kernel executor.
    executor: KernelExecutor,
    /// Use GPU if available.
    use_gpu: bool,
    /// Whether kernel is compiled.
    kernel_compiled: bool,
}

impl ConformanceKernel {
    /// Create a new conformance kernel with reference model.
    pub fn new(model: ProcessModel) -> Self {
        let mut kernel = Self {
            model,
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
            let source = generate_conformance_kernel();
            match self.executor.compile(&source) {
                Ok(_) => {
                    log::info!("Conformance CUDA kernel compiled successfully");
                    self.kernel_compiled = true;
                }
                Err(e) => {
                    log::warn!("Conformance CUDA kernel compilation failed: {}", e);
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

    /// Get the reference model.
    pub fn model(&self) -> &ProcessModel {
        &self.model
    }

    /// Check conformance for a batch of events.
    pub fn check(&mut self, events: &[GpuObjectEvent]) -> ConformanceCheckResult {
        let start = std::time::Instant::now();

        // Try GPU path first if available and compiled
        #[cfg(feature = "cuda")]
        let (gpu_results, exec_result) = if self.is_using_gpu() {
            match self.executor.execute_conformance_gpu(events, &self.model) {
                Ok((results, result)) => {
                    log::debug!(
                        "Conformance GPU execution: {} events -> {} results in {}µs",
                        events.len(),
                        results.len(),
                        result.execution_time_us
                    );
                    (Some(results), result)
                }
                Err(e) => {
                    log::warn!(
                        "Conformance GPU execution failed, falling back to CPU: {}",
                        e
                    );
                    (None, ExecutionResult::default())
                }
            }
        } else {
            (None, ExecutionResult::default())
        };

        #[cfg(not(feature = "cuda"))]
        let gpu_results: Option<Vec<ConformanceResult>> = None;
        #[cfg(not(feature = "cuda"))]
        let exec_result = ExecutionResult::default();

        // Use GPU results or fall back to CPU
        let (results, exec_result) = if let Some(gpu_results) = gpu_results {
            (gpu_results, exec_result)
        } else {
            // CPU fallback path
            let (results, result) = CpuFallbackExecutor::execute_conformance(events, &self.model);
            (results, result)
        };

        let total_time = start.elapsed().as_micros() as u64;

        ConformanceCheckResult {
            results,
            execution_result: exec_result,
            total_time_us: total_time,
        }
    }
}

/// Result of conformance checking.
#[derive(Debug, Clone)]
pub struct ConformanceCheckResult {
    /// Individual conformance results.
    pub results: Vec<ConformanceResult>,
    /// Kernel execution result.
    pub execution_result: ExecutionResult,
    /// Total processing time in microseconds.
    pub total_time_us: u64,
}

impl ConformanceCheckResult {
    /// Get number of conformant traces.
    pub fn conformant_count(&self) -> usize {
        self.results.iter().filter(|r| r.is_conformant()).count()
    }

    /// Get number of non-conformant traces.
    pub fn non_conformant_count(&self) -> usize {
        self.results.len() - self.conformant_count()
    }

    /// Get average fitness.
    pub fn avg_fitness(&self) -> f32 {
        if self.results.is_empty() {
            return 1.0;
        }
        let total: f32 = self.results.iter().map(|r| r.fitness).sum();
        total / self.results.len() as f32
    }

    /// Get results by compliance level.
    pub fn by_compliance(&self, level: ComplianceLevel) -> Vec<&ConformanceResult> {
        self.results
            .iter()
            .filter(|r| r.get_compliance_level() == level)
            .collect()
    }

    /// Get conformance distribution.
    pub fn distribution(&self) -> ConformanceDistribution {
        let mut dist = ConformanceDistribution::default();
        for result in &self.results {
            match result.get_compliance_level() {
                ComplianceLevel::FullyCompliant => dist.fully_compliant += 1,
                ComplianceLevel::MostlyCompliant => dist.mostly_compliant += 1,
                ComplianceLevel::PartiallyCompliant => dist.partially_compliant += 1,
                ComplianceLevel::NonCompliant => dist.non_compliant += 1,
            }
        }
        dist
    }
}

/// Distribution of conformance levels.
#[derive(Debug, Clone, Default)]
pub struct ConformanceDistribution {
    /// Fully compliant (fitness >= 0.95).
    pub fully_compliant: usize,
    /// Mostly compliant (fitness >= 0.8).
    pub mostly_compliant: usize,
    /// Partially compliant (fitness >= 0.5).
    pub partially_compliant: usize,
    /// Non-compliant (fitness < 0.5).
    pub non_compliant: usize,
}

impl ConformanceDistribution {
    /// Get total traces.
    pub fn total(&self) -> usize {
        self.fully_compliant + self.mostly_compliant + self.partially_compliant + self.non_compliant
    }

    /// Get percentages.
    pub fn percentages(&self) -> [f32; 4] {
        let total = self.total() as f32;
        if total == 0.0 {
            return [0.0; 4];
        }
        [
            self.fully_compliant as f32 / total * 100.0,
            self.mostly_compliant as f32 / total * 100.0,
            self.partially_compliant as f32 / total * 100.0,
            self.non_compliant as f32 / total * 100.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{HybridTimestamp, ProcessModelType};

    fn create_test_model() -> ProcessModel {
        let mut model = ProcessModel::new(1, "Test", ProcessModelType::DFG);
        model.start_activities = vec![1];
        model.end_activities = vec![3];
        model.add_transition(1, 2); // A -> B
        model.add_transition(2, 3); // B -> C
        model
    }

    fn create_conformant_events() -> Vec<GpuObjectEvent> {
        // A -> B -> C (conformant)
        vec![
            GpuObjectEvent {
                event_id: 1,
                object_id: 100,
                activity_id: 1,
                timestamp: HybridTimestamp::new(0, 0),
                ..Default::default()
            },
            GpuObjectEvent {
                event_id: 2,
                object_id: 100,
                activity_id: 2,
                timestamp: HybridTimestamp::new(100, 0),
                ..Default::default()
            },
            GpuObjectEvent {
                event_id: 3,
                object_id: 100,
                activity_id: 3,
                timestamp: HybridTimestamp::new(200, 0),
                ..Default::default()
            },
        ]
    }

    #[test]
    fn test_conformant_trace() {
        let model = create_test_model();
        let mut kernel = ConformanceKernel::new(model).with_cpu_only();
        let events = create_conformant_events();
        let result = kernel.check(&events);

        assert_eq!(result.results.len(), 1);
        assert!(result.results[0].is_conformant());
        assert_eq!(result.results[0].fitness, 1.0);
    }

    #[test]
    fn test_non_conformant_trace() {
        let model = create_test_model();
        let mut kernel = ConformanceKernel::new(model).with_cpu_only();

        // A -> C (missing B)
        let events = vec![
            GpuObjectEvent {
                event_id: 1,
                object_id: 100,
                activity_id: 1,
                timestamp: HybridTimestamp::new(0, 0),
                ..Default::default()
            },
            GpuObjectEvent {
                event_id: 2,
                object_id: 100,
                activity_id: 3,
                timestamp: HybridTimestamp::new(100, 0),
                ..Default::default()
            },
        ];

        let result = kernel.check(&events);
        assert!(!result.results[0].is_conformant());
    }

    #[test]
    fn test_conformance_distribution() {
        let model = create_test_model();
        let mut kernel = ConformanceKernel::new(model).with_cpu_only();
        let events = create_conformant_events();
        let result = kernel.check(&events);

        let dist = result.distribution();
        assert_eq!(dist.fully_compliant, 1);
    }
}
