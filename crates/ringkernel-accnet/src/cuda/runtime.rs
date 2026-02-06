//! GPU runtime with CUDA acceleration and CPU fallback.
//!
//! This module provides a unified runtime interface that uses
//! CUDA GPU acceleration when available, falling back to CPU execution.

use crate::kernels::{AnalysisConfig, AnalysisKernel, AnalysisResult};
use crate::models::AccountingNetwork;

/// Backend selection for kernel execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Automatically select best available backend.
    #[default]
    Auto,
    /// Force CUDA GPU execution.
    Cuda,
    /// Force CPU execution.
    Cpu,
}

/// Runtime status and capabilities.
#[derive(Debug, Clone)]
pub struct RuntimeStatus {
    /// Active backend.
    pub backend: Backend,
    /// Whether CUDA is available.
    pub cuda_available: bool,
    /// CUDA device name (if available).
    pub cuda_device_name: Option<String>,
    /// CUDA compute capability (if available).
    pub cuda_compute_capability: Option<(u32, u32)>,
    /// Whether GPU kernels are compiled and ready.
    pub gpu_kernels_ready: bool,
}

impl Default for RuntimeStatus {
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            cuda_available: false,
            cuda_device_name: None,
            cuda_compute_capability: None,
            gpu_kernels_ready: false,
        }
    }
}

/// GPU-accelerated analysis runtime.
///
/// Uses CUDA for analysis kernels when available, with automatic fallback to CPU.
pub struct AnalysisRuntime {
    /// Selected backend.
    backend: Backend,
    /// CPU kernel (always available).
    cpu_kernel: AnalysisKernel,
    /// Runtime status.
    status: RuntimeStatus,
    /// GPU executor (when cuda feature enabled and CUDA available).
    #[cfg(feature = "cuda")]
    gpu_executor: Option<super::executor::GpuExecutor>,
}

impl AnalysisRuntime {
    /// Create a new runtime with automatic backend selection.
    pub fn new() -> Self {
        Self::with_backend(Backend::Auto)
    }

    /// Create a runtime with a specific backend preference.
    pub fn with_backend(backend: Backend) -> Self {
        let cpu_kernel = AnalysisKernel::new(AnalysisConfig::default());
        let mut status = RuntimeStatus::default();

        // Try to initialize GPU executor
        #[cfg(feature = "cuda")]
        let (gpu_executor, cuda_available) = Self::try_init_gpu(&mut status);

        #[cfg(not(feature = "cuda"))]
        let cuda_available = false;

        // Determine actual backend
        let active_backend = match backend {
            Backend::Auto => {
                if cuda_available {
                    Backend::Cuda
                } else {
                    Backend::Cpu
                }
            }
            Backend::Cuda => {
                if cuda_available {
                    Backend::Cuda
                } else {
                    tracing::warn!("CUDA requested but not available, falling back to CPU");
                    Backend::Cpu
                }
            }
            Backend::Cpu => Backend::Cpu,
        };

        status.backend = active_backend;
        status.cuda_available = cuda_available;

        Self {
            backend: active_backend,
            cpu_kernel,
            status,
            #[cfg(feature = "cuda")]
            gpu_executor,
        }
    }

    /// Try to initialize GPU executor.
    #[cfg(feature = "cuda")]
    fn try_init_gpu(status: &mut RuntimeStatus) -> (Option<super::executor::GpuExecutor>, bool) {
        // Check if CUDA is available
        if !ringkernel_cuda::is_cuda_available() {
            return (None, false);
        }

        // Try to create GPU executor
        match super::executor::GpuExecutor::new() {
            Ok(mut executor) => {
                status.cuda_device_name = Some(executor.device_name().to_string());
                status.cuda_compute_capability = Some(executor.compute_capability());

                // Try to compile kernels
                match executor.compile_kernels() {
                    Ok(()) => {
                        status.gpu_kernels_ready = true;
                        tracing::info!(
                            device = %executor.device_name(),
                            cc_major = executor.compute_capability().0,
                            cc_minor = executor.compute_capability().1,
                            "GPU kernels compiled"
                        );
                        (Some(executor), true)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "failed to compile GPU kernels");
                        (None, false)
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to initialize GPU");
                (None, false)
            }
        }
    }

    /// Get the active backend.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Get runtime status.
    pub fn status(&self) -> &RuntimeStatus {
        &self.status
    }

    /// Check if CUDA is being used.
    pub fn is_cuda_active(&self) -> bool {
        self.backend == Backend::Cuda && self.status.gpu_kernels_ready
    }

    /// Analyze the network using the best available backend.
    pub fn analyze(&self, network: &AccountingNetwork) -> AnalysisResult {
        #[cfg(feature = "cuda")]
        {
            if self.is_cuda_active() {
                if let Some(ref executor) = self.gpu_executor {
                    match executor.analyze(network) {
                        Ok(gpu_result) => {
                            // Convert GPU result to AnalysisResult format
                            return self.convert_gpu_result(network, gpu_result);
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "GPU analysis failed, falling back to CPU");
                        }
                    }
                }
            }
        }

        // Fallback to CPU
        self.cpu_kernel.analyze(network)
    }

    /// Convert GPU result to standard AnalysisResult format.
    #[cfg(feature = "cuda")]
    fn convert_gpu_result(
        &self,
        network: &AccountingNetwork,
        gpu_result: super::executor::GpuAnalysisResult,
    ) -> AnalysisResult {
        use crate::kernels::AnalysisStats;
        use crate::models::{GaapViolation, GaapViolationType, HybridTimestamp, ViolationSeverity};

        // Build suspense results as (index, score) tuples
        let suspense_accounts: Vec<(u16, f32)> = gpu_result
            .suspense_scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score > 0.5)
            .map(|(idx, &score)| (idx as u16, score))
            .collect();

        // Build GAAP violation results
        let gaap_violations: Vec<GaapViolation> = gpu_result
            .gaap_violations
            .iter()
            .enumerate()
            .filter(|(_, &flag)| flag > 0)
            .map(|(idx, &flag)| {
                let flow = network.flows.get(idx);
                GaapViolation {
                    id: uuid::Uuid::new_v4(),
                    violation_type: match flag {
                        1 => GaapViolationType::RevenueToCashDirect,
                        2 => GaapViolationType::RevenueToExpense,
                        _ => GaapViolationType::UnbalancedEntry,
                    },
                    severity: if flag == 1 {
                        ViolationSeverity::High
                    } else {
                        ViolationSeverity::Medium
                    },
                    source_account: flow.map(|f| f.source_account_index).unwrap_or(0),
                    target_account: flow.map(|f| f.target_account_index).unwrap_or(0),
                    amount: flow.map(|f| f.amount).unwrap_or_default(),
                    journal_entry_id: uuid::Uuid::nil(),
                    detected_at: HybridTimestamp::now(),
                    description: match flag {
                        1 => "Revenue to Asset Direct (GPU detected)".to_string(),
                        2 => "Revenue to Expense Direct (GPU detected)".to_string(),
                        _ => "Unknown violation".to_string(),
                    },
                }
            })
            .collect();

        AnalysisResult {
            stats: AnalysisStats {
                accounts_analyzed: network.accounts.len(),
                flows_analyzed: network.flows.len(),
                suspense_count: suspense_accounts.len(),
                gaap_violation_count: gaap_violations.len(),
                fraud_pattern_count: 0,
            },
            suspense_accounts,
            gaap_violations,
            fraud_patterns: Vec::new(), // Not implemented in GPU yet
        }
    }

    /// Run benchmarks comparing CPU vs GPU performance.
    #[cfg(feature = "cuda")]
    pub fn run_benchmarks(
        &self,
        network: &AccountingNetwork,
    ) -> Option<super::executor::BenchmarkResults> {
        if let Some(ref executor) = self.gpu_executor {
            match executor.run_benchmarks(network) {
                Ok(results) => Some(results),
                Err(e) => {
                    tracing::warn!(error = %e, "benchmark failed");
                    None
                }
            }
        } else {
            None
        }
    }

    /// Get the generated CUDA kernel code (for inspection/debugging).
    #[cfg(feature = "cuda")]
    pub fn cuda_kernel_code(&self, kernel_type: super::CudaKernelType) -> Option<String> {
        super::GeneratedKernels::generate()
            .ok()
            .map(|k| match kernel_type {
                super::CudaKernelType::SuspenseDetection => k.suspense_detection,
                super::CudaKernelType::GaapViolation => k.gaap_violation,
                super::CudaKernelType::BenfordAnalysis => k.benford_analysis,
                _ => String::new(),
            })
    }
}

impl Default for AnalysisRuntime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_runtime_creation() {
        let runtime = AnalysisRuntime::new();
        // Should work regardless of CUDA availability
        assert!(runtime.backend() == Backend::Cpu || runtime.backend() == Backend::Cuda);
    }

    #[test]
    fn test_cpu_fallback() {
        let runtime = AnalysisRuntime::with_backend(Backend::Cpu);
        assert_eq!(runtime.backend(), Backend::Cpu);
    }

    #[test]
    fn test_analysis() {
        let runtime = AnalysisRuntime::new();
        let network = AccountingNetwork::new(Uuid::new_v4(), 2024, 1);
        let result = runtime.analyze(&network);
        assert_eq!(result.stats.accounts_analyzed, network.accounts.len());
    }
}
