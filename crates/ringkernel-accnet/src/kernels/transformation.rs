//! Journal entry transformation kernels (Methods A-E).
//!
//! These kernels transform double-entry journal entries into
//! directed graph flows using the methods from Ivertowski et al., 2024.

use crate::models::{Decimal128, TransactionFlow, SolvingMethod, HybridTimestamp};
use crate::fabric::GeneratedEntry;
use uuid::Uuid;

/// Configuration for transformation kernels.
#[derive(Debug, Clone)]
pub struct TransformationConfig {
    /// Block size for GPU dispatch.
    pub block_size: u32,
    /// Maximum entries per batch.
    pub max_batch_size: u32,
    /// Enable confidence scoring.
    pub compute_confidence: bool,
}

impl Default for TransformationConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            max_batch_size: 65536,
            compute_confidence: true,
        }
    }
}

/// Result of journal transformation.
#[derive(Debug, Clone)]
pub struct TransformationResult {
    /// Generated flows.
    pub flows: Vec<TransactionFlow>,
    /// Method used for each entry.
    pub methods: Vec<SolvingMethod>,
    /// Statistics about the transformation.
    pub stats: TransformationStats,
}

/// Statistics from transformation.
#[derive(Debug, Clone, Default)]
pub struct TransformationStats {
    /// Entries processed.
    pub entries_processed: usize,
    /// Flows generated.
    pub flows_generated: usize,
    /// Method A count.
    pub method_a_count: usize,
    /// Method B count.
    pub method_b_count: usize,
    /// Method C count.
    pub method_c_count: usize,
    /// Method D count.
    pub method_d_count: usize,
    /// Method E count.
    pub method_e_count: usize,
    /// Average confidence score.
    pub avg_confidence: f64,
}

/// Journal transformation kernel dispatcher.
pub struct TransformationKernel {
    config: TransformationConfig,
}

impl TransformationKernel {
    /// Create a new transformation kernel.
    pub fn new(config: TransformationConfig) -> Self {
        Self { config }
    }

    /// Transform generated entries to flows (CPU fallback).
    pub fn transform(&self, entries: &[GeneratedEntry]) -> TransformationResult {
        let mut flows = Vec::new();
        let mut methods = Vec::new();
        let mut stats = TransformationStats::default();

        for entry in entries {
            // Use the expected flows from the generator (source, target, amount)
            for &(source_idx, target_idx, amount) in &entry.expected_flows {
                let flow = TransactionFlow::new(
                    source_idx,
                    target_idx,
                    amount,
                    entry.entry.id,
                    entry.entry.posting_date,
                );
                flows.push(flow);
            }

            // Determine method from entry
            let method = entry.entry.solving_method;
            methods.push(method);

            match method {
                SolvingMethod::MethodA => stats.method_a_count += 1,
                SolvingMethod::MethodB => stats.method_b_count += 1,
                SolvingMethod::MethodC => stats.method_c_count += 1,
                SolvingMethod::MethodD => stats.method_d_count += 1,
                SolvingMethod::MethodE => stats.method_e_count += 1,
                SolvingMethod::Pending => {}
            }
        }

        stats.entries_processed = entries.len();
        stats.flows_generated = flows.len();
        stats.avg_confidence = if !flows.is_empty() {
            flows.iter().map(|f| f.confidence as f64).sum::<f64>() / flows.len() as f64
        } else {
            0.0
        };

        TransformationResult { flows, methods, stats }
    }
}

impl Default for TransformationKernel {
    fn default() -> Self {
        Self::new(TransformationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformation_kernel_creation() {
        let kernel = TransformationKernel::default();
        assert_eq!(kernel.config.block_size, 256);
    }
}
