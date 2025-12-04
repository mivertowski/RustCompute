//! CUDA code generation and GPU execution for accelerated analysis.
//!
//! This module provides:
//! - Transpiled CUDA kernels from Rust DSL
//! - Actual GPU execution via ringkernel-cuda
//! - Benchmarking infrastructure for CPU vs GPU comparison

#[cfg(feature = "cuda")]
pub mod codegen;
#[cfg(feature = "cuda")]
pub mod executor;
pub mod runtime;

#[cfg(feature = "cuda")]
pub use codegen::*;
#[cfg(feature = "cuda")]
pub use executor::{BenchmarkResults, GpuAnalysisResult, GpuExecutor, KernelBenchmark};
pub use runtime::{AnalysisRuntime, Backend, RuntimeStatus};

/// CUDA kernel configuration.
#[derive(Debug, Clone)]
pub struct CudaKernelConfig {
    /// Block size (threads per block).
    pub block_size: u32,
    /// Shared memory size in bytes.
    pub shared_mem_size: u32,
    /// Enable async execution.
    pub async_execution: bool,
}

impl Default for CudaKernelConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            shared_mem_size: 0,
            async_execution: true,
        }
    }
}

/// Available CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaKernelType {
    /// Journal entry transformation.
    JournalTransform,
    /// Suspense account detection.
    SuspenseDetection,
    /// GAAP violation detection.
    GaapViolation,
    /// Fraud pattern detection.
    FraudPattern,
    /// Benford's Law analysis.
    BenfordAnalysis,
    /// Network PageRank.
    PageRank,
    /// Circular flow detection.
    CircularFlowDetection,
    /// Temporal anomaly detection.
    TemporalAnomaly,
}

impl CudaKernelType {
    /// Get recommended block size for this kernel.
    pub fn recommended_block_size(&self) -> u32 {
        match self {
            Self::JournalTransform => 256,
            Self::SuspenseDetection => 128,
            Self::GaapViolation => 256,
            Self::FraudPattern => 128,
            Self::BenfordAnalysis => 256,
            Self::PageRank => 256,
            Self::CircularFlowDetection => 64,
            Self::TemporalAnomaly => 128,
        }
    }

    /// Get kernel name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::JournalTransform => "journal_transform_kernel",
            Self::SuspenseDetection => "suspense_detection_kernel",
            Self::GaapViolation => "gaap_violation_kernel",
            Self::FraudPattern => "fraud_pattern_kernel",
            Self::BenfordAnalysis => "benford_analysis_kernel",
            Self::PageRank => "pagerank_kernel",
            Self::CircularFlowDetection => "circular_flow_kernel",
            Self::TemporalAnomaly => "temporal_anomaly_kernel",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_config() {
        let config = CudaKernelConfig::default();
        assert_eq!(config.block_size, 256);
    }

    #[test]
    fn test_kernel_types() {
        assert_eq!(
            CudaKernelType::JournalTransform.name(),
            "journal_transform_kernel"
        );
        assert_eq!(CudaKernelType::PageRank.recommended_block_size(), 256);
    }
}
