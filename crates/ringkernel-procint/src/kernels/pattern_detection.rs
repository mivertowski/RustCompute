//! Pattern detection kernel.
//!
//! Detects process patterns from DFG nodes using GPU acceleration.

use crate::cuda::{
    generate_pattern_kernel, CpuFallbackExecutor, ExecutionResult, GpuStats, GpuStatus,
    KernelExecutor,
};
use crate::models::{GpuDFGNode, GpuPatternMatch, PatternSeverity, PatternType};

/// Pattern detection configuration.
#[derive(Debug, Clone)]
pub struct PatternConfig {
    /// Bottleneck detection threshold (incoming rate).
    pub bottleneck_threshold: f32,
    /// Long-running duration threshold (ms).
    pub duration_threshold: f32,
    /// Minimum confidence to report pattern.
    pub min_confidence: f32,
    /// Maximum patterns to detect.
    pub max_patterns: usize,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            bottleneck_threshold: 2.0,
            duration_threshold: 60000.0, // 1 minute
            min_confidence: 0.5,
            max_patterns: 100,
        }
    }
}

/// Pattern detection kernel.
pub struct PatternDetectionKernel {
    /// Configuration.
    config: PatternConfig,
    /// Kernel executor.
    executor: KernelExecutor,
    /// Use GPU if available.
    use_gpu: bool,
    /// Whether kernel is compiled.
    kernel_compiled: bool,
}

impl Default for PatternDetectionKernel {
    fn default() -> Self {
        Self::new(PatternConfig::default())
    }
}

impl PatternDetectionKernel {
    /// Create a new pattern detection kernel.
    pub fn new(config: PatternConfig) -> Self {
        let mut kernel = Self {
            config,
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
            let source = generate_pattern_kernel();
            match self.executor.compile(&source) {
                Ok(_) => {
                    log::info!("Pattern detection CUDA kernel compiled successfully");
                    self.kernel_compiled = true;
                }
                Err(e) => {
                    log::warn!("Pattern detection CUDA kernel compilation failed: {}", e);
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

    /// Detect patterns from DFG nodes.
    pub fn detect(&mut self, nodes: &[GpuDFGNode]) -> PatternResult {
        let start = std::time::Instant::now();

        // Try GPU path first if available and compiled
        #[cfg(feature = "cuda")]
        let (gpu_patterns, exec_result) = if self.is_using_gpu() {
            match self.executor.execute_pattern_gpu(
                nodes,
                self.config.bottleneck_threshold,
                self.config.duration_threshold,
            ) {
                Ok((patterns, result)) => {
                    log::debug!(
                        "Pattern GPU execution: {} nodes -> {} patterns in {}µs",
                        nodes.len(),
                        patterns.len(),
                        result.execution_time_us
                    );
                    (Some(patterns), result)
                }
                Err(e) => {
                    log::warn!("Pattern GPU execution failed, falling back to CPU: {}", e);
                    (None, ExecutionResult::default())
                }
            }
        } else {
            (None, ExecutionResult::default())
        };

        #[cfg(not(feature = "cuda"))]
        let gpu_patterns: Option<Vec<GpuPatternMatch>> = None;
        #[cfg(not(feature = "cuda"))]
        let exec_result = ExecutionResult::default();

        // Use GPU results or fall back to CPU
        let (mut patterns, exec_result) = if let Some(gpu_patterns) = gpu_patterns {
            (gpu_patterns, exec_result)
        } else {
            let mut patterns = Vec::with_capacity(self.config.max_patterns);
            let result = CpuFallbackExecutor::execute_pattern_detection(
                nodes,
                &mut patterns,
                self.config.bottleneck_threshold,
                self.config.duration_threshold,
            );
            (patterns, result)
        };

        // Additional pattern detection: loops and rework
        self.detect_loop_patterns(nodes, &mut patterns);

        // Filter by confidence
        patterns.retain(|p| p.confidence >= self.config.min_confidence);

        // Limit results
        patterns.truncate(self.config.max_patterns);

        let total_time = start.elapsed().as_micros() as u64;

        PatternResult {
            patterns,
            execution_result: exec_result,
            total_time_us: total_time,
        }
    }

    /// Detect loop patterns from node self-transitions.
    fn detect_loop_patterns(&self, nodes: &[GpuDFGNode], patterns: &mut Vec<GpuPatternMatch>) {
        for node in nodes {
            // Check for high frequency self-activity (rework indicator)
            if node.incoming_count > 0 && node.outgoing_count > 0 {
                // Use degree ratio as a proxy for loop detection
                let in_count = node.incoming_count as f32;
                let out_count = node.outgoing_count as f32;
                let degree_ratio = in_count.min(out_count) / in_count.max(out_count).max(1.0);
                if degree_ratio > 0.3 && node.event_count > 10 {
                    let mut pattern =
                        GpuPatternMatch::new(PatternType::Loop, PatternSeverity::Warning);
                    pattern.add_activity(node.activity_id);
                    pattern.confidence = degree_ratio;
                    pattern.frequency = node.event_count;
                    pattern.avg_duration_ms = node.avg_duration_ms;
                    patterns.push(pattern);
                }
            }
        }
    }
}

/// Result of pattern detection.
#[derive(Debug)]
pub struct PatternResult {
    /// Detected patterns.
    pub patterns: Vec<GpuPatternMatch>,
    /// Kernel execution result.
    pub execution_result: ExecutionResult,
    /// Total processing time in microseconds.
    pub total_time_us: u64,
}

impl PatternResult {
    /// Get patterns by type.
    pub fn by_type(&self, pattern_type: PatternType) -> Vec<&GpuPatternMatch> {
        self.patterns
            .iter()
            .filter(|p| p.get_pattern_type() == pattern_type)
            .collect()
    }

    /// Get patterns by severity.
    pub fn by_severity(&self, severity: PatternSeverity) -> Vec<&GpuPatternMatch> {
        self.patterns
            .iter()
            .filter(|p| p.get_severity() == severity)
            .collect()
    }

    /// Count patterns by type.
    pub fn count_by_type(&self) -> std::collections::HashMap<PatternType, usize> {
        let mut counts = std::collections::HashMap::new();
        for pattern in &self.patterns {
            *counts.entry(pattern.get_pattern_type()).or_insert(0) += 1;
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_nodes() -> Vec<GpuDFGNode> {
        vec![
            GpuDFGNode {
                activity_id: 1,
                event_count: 100,
                avg_duration_ms: 5000.0,
                incoming_count: 5, // High incoming
                outgoing_count: 1, // Low outgoing (bottleneck)
                ..Default::default()
            },
            GpuDFGNode {
                activity_id: 2,
                event_count: 50,
                avg_duration_ms: 120000.0, // Long running
                incoming_count: 2,
                outgoing_count: 2,
                ..Default::default()
            },
        ]
    }

    #[test]
    fn test_pattern_detection() {
        let mut kernel = PatternDetectionKernel::default().with_cpu_only();
        let nodes = create_test_nodes();
        let result = kernel.detect(&nodes);

        // Should detect bottleneck and long-running patterns
        assert!(!result.patterns.is_empty()); // At least long-running
    }

    #[test]
    fn test_pattern_filtering() {
        let mut kernel = PatternDetectionKernel::default().with_cpu_only();
        let nodes = create_test_nodes();
        let result = kernel.detect(&nodes);

        let long_running = result.by_type(PatternType::LongRunning);
        assert!(!long_running.is_empty());
    }
}
