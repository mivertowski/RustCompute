//! GPU kernel code generation for accounting analytics actors.
//!
//! Uses ringkernel-cuda-codegen to generate persistent ring kernels
//! that process messages in a continuous loop on the GPU.

use ringkernel_cuda_codegen::{transpile_ring_kernel, RingKernelConfig};

/// Generated CUDA kernel code for all analytics actors.
pub struct AnalyticsKernels {
    /// PageRank computation kernel.
    pub pagerank: String,
    /// Fraud detection kernel.
    pub fraud_detector: String,
    /// GAAP validation kernel.
    pub gaap_validator: String,
    /// Benford analysis kernel.
    pub benford_analyzer: String,
    /// Suspense detection kernel.
    pub suspense_detector: String,
    /// Results aggregator kernel.
    pub results_aggregator: String,
}

impl AnalyticsKernels {
    /// Generate all analytics kernels.
    pub fn generate() -> Result<Self, String> {
        Ok(Self {
            pagerank: generate_pagerank_kernel()?,
            fraud_detector: generate_fraud_detector_kernel()?,
            gaap_validator: generate_gaap_validator_kernel()?,
            benford_analyzer: generate_benford_analyzer_kernel()?,
            suspense_detector: generate_suspense_detector_kernel()?,
            results_aggregator: generate_results_aggregator_kernel()?,
        })
    }
}

/// Generate the PageRank ring kernel.
///
/// This kernel computes PageRank scores using power iteration.
/// It processes PageRankRequest messages and sends PageRankResponse.
fn generate_pagerank_kernel() -> Result<String, String> {
    let handler: syn::ItemFn = syn::parse_quote! {
        fn pagerank_step(
            ctx: &RingContext,
            // Graph structure (CSR format)
            row_ptr: &[u32],        // row_ptr[i] = start of neighbors for node i
            col_idx: &[u32],        // col_idx[j] = neighbor node index
            out_degree: &[u32],     // out_degree[i] = number of outgoing edges
            // PageRank values
            pr_current: &[f32],     // Current PageRank values
            pr_next: &mut [f32],    // Next iteration values
            // Parameters
            damping: f32,
            n_nodes: u32
        ) -> u32 {
            let tid = ctx.global_thread_id();

            // Each thread handles one node
            if tid >= n_nodes { return 0; }

            let i = tid as usize;

            // Base score for dangling nodes
            let base_score = (1.0 - damping) / (n_nodes as f32);

            // Sum contributions from incoming neighbors
            let start = row_ptr[i] as usize;
            let end = row_ptr[i + 1] as usize;

            let mut sum: f32 = 0.0;

            // Iterate over incoming edges
            let mut j = start;
            while j < end {
                let neighbor = col_idx[j] as usize;
                let neighbor_degree = out_degree[neighbor] as f32;
                if neighbor_degree > 0.0 {
                    sum = sum + pr_current[neighbor] / neighbor_degree;
                }
                j = j + 1;
            }

            // Update PageRank
            pr_next[i] = base_score + damping * sum;

            // Synchronize before next iteration
            ctx.sync_threads();

            1 // Return success
        }
    };

    let config = RingKernelConfig::new("pagerank_actor")
        .with_block_size(256)
        .with_queue_capacity(64)
        .with_hlc(true)
        .with_k2k(true);

    transpile_ring_kernel(&handler, &config)
        .map_err(|e| format!("Failed to generate pagerank kernel: {}", e))
}

/// Generate the fraud detection ring kernel.
///
/// Detects fraud patterns: circular flows, velocity anomalies, round amounts.
fn generate_fraud_detector_kernel() -> Result<String, String> {
    let handler: syn::ItemFn = syn::parse_quote! {
        fn detect_fraud(
            ctx: &RingContext,
            // Flow data
            flow_source: &[u32],      // Source account for each flow
            flow_target: &[u32],      // Target account for each flow
            flow_amount: &[f64],      // Amount for each flow
            flow_timestamp: &[u64],   // Timestamp for each flow
            // Account data
            account_risk: &[f32],     // Current risk score per account
            // Output
            pattern_flags: &mut [u32], // Detected patterns (bitflags)
            risk_delta: &mut [f32],    // Risk score change per flow
            // Parameters
            n_flows: u32,
            velocity_threshold: f32,   // Max flows per time unit
            round_amount_threshold: f64
        ) -> u32 {
            let tid = ctx.global_thread_id();

            if tid >= n_flows { return 0; }

            let i = tid as usize;
            let mut flags: u32 = 0;
            let mut risk: f32 = 0.0;

            let source = flow_source[i];
            let target = flow_target[i];
            let amount = flow_amount[i];
            let _timestamp = flow_timestamp[i];

            // Check for round amount (potential structuring)
            let abs_amount = if amount > 0.0 { amount } else { -amount };
            let thousands = abs_amount / 1000.0;
            let truncated = (thousands as i32) as f64;
            let fractional = thousands - truncated;

            if abs_amount >= 1000.0 && fractional < 0.001 {
                flags = flags | 0x01; // ROUND_AMOUNT flag
                risk = risk + 0.2;
            }

            // Check for self-loop (immediate circular)
            if source == target {
                flags = flags | 0x02; // SELF_LOOP flag
                risk = risk + 0.5;
            }

            // Check for threshold proximity (structuring indicator)
            let dist_10k = (abs_amount - 10000.0) as f32;
            let abs_dist_10k = if dist_10k > 0.0 { dist_10k } else { -dist_10k };
            if abs_dist_10k < 500.0 {
                flags = flags | 0x04; // THRESHOLD_PROXIMITY flag
                risk = risk + 0.3;
            }

            // Add base risk from accounts involved
            risk = risk + account_risk[source as usize] * 0.3;
            risk = risk + account_risk[target as usize] * 0.3;

            // Clamp risk to 1.0
            if risk > 1.0 { risk = 1.0; }

            pattern_flags[i] = flags;
            risk_delta[i] = risk;

            ctx.sync_threads();

            flags
        }
    };

    let config = RingKernelConfig::new("fraud_detector_actor")
        .with_block_size(256)
        .with_queue_capacity(128)
        .with_hlc(true)
        .with_k2k(true);

    transpile_ring_kernel(&handler, &config)
        .map_err(|e| format!("Failed to generate fraud detector kernel: {}", e))
}

/// Generate the GAAP validation ring kernel.
fn generate_gaap_validator_kernel() -> Result<String, String> {
    let handler: syn::ItemFn = syn::parse_quote! {
        fn validate_gaap(
            ctx: &RingContext,
            // Flow data
            flow_source: &[u32],
            flow_target: &[u32],
            // Account types (0=Asset, 1=Liability, 2=Equity, 3=Revenue, 4=Expense, 5=Contra)
            account_types: &[u32],
            // Output
            violation_flags: &mut [u32],
            violation_type: &mut [u32],
            // Parameters
            n_flows: u32
        ) -> u32 {
            let tid = ctx.global_thread_id();

            if tid >= n_flows { return 0; }

            let i = tid as usize;
            let source_idx = flow_source[i] as usize;
            let target_idx = flow_target[i] as usize;
            let source_type = account_types[source_idx];
            let target_type = account_types[target_idx];

            let mut violation: u32 = 0;
            let mut vtype: u32 = 0;

            // Rule 1: Revenue (3) -> Asset (0) direct is improper
            // Should go through Accounts Receivable
            if source_type == 3 && target_type == 0 {
                violation = 1;
                vtype = 1; // IMPROPER_REVENUE_RECOGNITION
            }

            // Rule 2: Revenue (3) -> Expense (4) direct is improper netting
            if source_type == 3 && target_type == 4 {
                violation = 1;
                vtype = 2; // IMPROPER_NETTING
            }

            // Rule 3: Expense (4) -> Revenue (3) is reversal (needs review)
            if source_type == 4 && target_type == 3 {
                violation = 1;
                vtype = 3; // SUSPICIOUS_REVERSAL
            }

            // Rule 4: Equity (2) -> Expense (4) bypasses income statement
            if source_type == 2 && target_type == 4 {
                violation = 1;
                vtype = 4; // EQUITY_EXPENSE_BYPASS
            }

            violation_flags[i] = violation;
            violation_type[i] = vtype;

            ctx.sync_threads();

            violation
        }
    };

    let config = RingKernelConfig::new("gaap_validator_actor")
        .with_block_size(256)
        .with_queue_capacity(128)
        .with_hlc(true)
        .with_k2k(true);

    transpile_ring_kernel(&handler, &config)
        .map_err(|e| format!("Failed to generate GAAP validator kernel: {}", e))
}

/// Generate the Benford analysis ring kernel.
fn generate_benford_analyzer_kernel() -> Result<String, String> {
    let handler: syn::ItemFn = syn::parse_quote! {
        fn analyze_benford(
            ctx: &RingContext,
            // Input amounts
            amounts: &[f64],
            // Output digit counts (shared, use atomics)
            digit_counts: &mut [u32],
            // Parameters
            n_amounts: u32
        ) -> u32 {
            let tid = ctx.global_thread_id();

            if tid >= n_amounts { return 0; }

            let i = tid as usize;
            let amount = amounts[i];
            let abs_amount = if amount > 0.0 { amount } else { -amount };

            // Skip amounts less than 1
            if abs_amount < 1.0 { return 0; }

            // Extract first digit
            let v1 = if abs_amount >= 10000000000000.0 { abs_amount / 10000000000000.0 } else { abs_amount };
            let v2 = if v1 >= 1000000.0 { v1 / 1000000.0 } else { v1 };
            let v3 = if v2 >= 1000.0 { v2 / 1000.0 } else { v2 };
            let v4 = if v3 >= 100.0 { v3 / 100.0 } else { v3 };
            let v5 = if v4 >= 10.0 { v4 / 10.0 } else { v4 };

            let first_digit = v5 as i32;

            // Valid digits are 1-9
            if first_digit < 1 { return 0; }
            if first_digit > 9 { return 0; }

            // Atomically increment the count for this digit
            atomic_add(&mut digit_counts[(first_digit - 1) as usize], 1);

            1
        }
    };

    let config = RingKernelConfig::new("benford_analyzer_actor")
        .with_block_size(256)
        .with_queue_capacity(64)
        .with_hlc(true)
        .with_k2k(true);

    transpile_ring_kernel(&handler, &config)
        .map_err(|e| format!("Failed to generate Benford analyzer kernel: {}", e))
}

/// Generate the suspense detection ring kernel.
fn generate_suspense_detector_kernel() -> Result<String, String> {
    let handler: syn::ItemFn = syn::parse_quote! {
        fn detect_suspense(
            ctx: &RingContext,
            // Account data
            balance_debit: &[f64],
            balance_credit: &[f64],
            risk_scores: &[f32],
            inflow_counts: &[u32],
            outflow_counts: &[u32],
            // Output
            suspense_scores: &mut [f32],
            // Parameters
            n_accounts: u32
        ) -> u32 {
            let tid = ctx.global_thread_id();

            if tid >= n_accounts { return 0; }

            let i = tid as usize;
            let mut score: f32 = 0.0;

            // Calculate balance and check for round numbers
            let balance = balance_debit[i] - balance_credit[i];
            let abs_balance = if balance > 0.0 { balance } else { -balance };

            // Round number check (divisible by 1000)
            let thousands = abs_balance / 1000.0;
            let truncated = (thousands as i32) as f64;
            let fractional = thousands - truncated;

            if abs_balance >= 1000.0 && fractional < 0.001 {
                score = score + 0.3;
            }

            // High existing risk
            if risk_scores[i] > 0.5 {
                score = score + 0.4;
            }

            // Flow imbalance (many inflows, few outflows = holding pattern)
            let inflows = inflow_counts[i] as f32;
            let outflows = outflow_counts[i] as f32;
            let flow_ratio = if outflows > 0.0 { inflows / outflows } else { 10.0 };

            if flow_ratio > 5.0 {
                score = score + 0.3;
            }

            // Clamp
            if score > 1.0 { score = 1.0; }

            suspense_scores[i] = score;

            ctx.sync_threads();

            if score > 0.5 { 1 } else { 0 }
        }
    };

    let config = RingKernelConfig::new("suspense_detector_actor")
        .with_block_size(256)
        .with_queue_capacity(64)
        .with_hlc(true)
        .with_k2k(true);

    transpile_ring_kernel(&handler, &config)
        .map_err(|e| format!("Failed to generate suspense detector kernel: {}", e))
}

/// Generate the results aggregator ring kernel.
///
/// This kernel aggregates results from all analytics kernels and
/// prepares the final AnalyticsResult message for the host.
fn generate_results_aggregator_kernel() -> Result<String, String> {
    let handler: syn::ItemFn = syn::parse_quote! {
        fn aggregate_results(
            ctx: &RingContext,
            // Results from various analyzers
            fraud_flags: &[u32],
            gaap_violations: &[u32],
            suspense_scores: &[f32],
            // Counts (single element arrays for atomic accumulation)
            fraud_count: &mut [u32],
            gaap_count: &mut [u32],
            suspense_count: &mut [u32],
            risk_sum: &mut [f32],
            // Parameters
            n_flows: u32,
            n_accounts: u32
        ) -> u32 {
            let tid = ctx.global_thread_id();

            // Count fraud patterns (each thread processes one flow)
            if tid < n_flows {
                let i = tid as usize;
                if fraud_flags[i] != 0 {
                    atomic_add(&mut fraud_count[0], 1);
                }
                if gaap_violations[i] != 0 {
                    atomic_add(&mut gaap_count[0], 1);
                }
            }

            ctx.sync_threads();

            // Count suspense accounts (use threads for accounts)
            if tid < n_accounts {
                let i = tid as usize;
                if suspense_scores[i] > 0.5 {
                    atomic_add(&mut suspense_count[0], 1);
                }
                // Note: atomic_add for f32 would need special handling
                // For now, we skip risk_sum or use integer approximation
            }

            ctx.sync_threads();

            1
        }
    };

    let config = RingKernelConfig::new("results_aggregator_actor")
        .with_block_size(256)
        .with_queue_capacity(32)
        .with_hlc(true)
        .with_k2k(true);

    transpile_ring_kernel(&handler, &config)
        .map_err(|e| format!("Failed to generate results aggregator kernel: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_all_kernels() {
        let kernels = AnalyticsKernels::generate();
        assert!(
            kernels.is_ok(),
            "Failed to generate kernels: {:?}",
            kernels.err()
        );

        let kernels = kernels.unwrap();
        assert!(kernels.pagerank.contains("pagerank_actor"));
        assert!(kernels.fraud_detector.contains("fraud_detector_actor"));
        assert!(kernels.gaap_validator.contains("gaap_validator_actor"));
        assert!(kernels.benford_analyzer.contains("benford_analyzer_actor"));
        assert!(kernels
            .suspense_detector
            .contains("suspense_detector_actor"));
        assert!(kernels
            .results_aggregator
            .contains("results_aggregator_actor"));
    }

    #[test]
    fn test_pagerank_kernel() {
        let kernel = generate_pagerank_kernel();
        assert!(kernel.is_ok(), "PageRank kernel failed: {:?}", kernel.err());
        let code = kernel.unwrap();
        assert!(code.contains("damping"));
        assert!(code.contains("pr_current"));
        assert!(code.contains("pr_next"));
    }

    #[test]
    fn test_fraud_detector_kernel() {
        let kernel = generate_fraud_detector_kernel();
        assert!(
            kernel.is_ok(),
            "Fraud detector kernel failed: {:?}",
            kernel.err()
        );
        let code = kernel.unwrap();
        assert!(code.contains("pattern_flags"));
        assert!(code.contains("risk_delta"));
    }

    #[test]
    fn test_gaap_validator_kernel() {
        let kernel = generate_gaap_validator_kernel();
        assert!(
            kernel.is_ok(),
            "GAAP validator kernel failed: {:?}",
            kernel.err()
        );
        let code = kernel.unwrap();
        assert!(code.contains("violation_flags"));
        assert!(code.contains("account_types"));
    }

    #[test]
    fn test_benford_analyzer_kernel() {
        let kernel = generate_benford_analyzer_kernel();
        assert!(
            kernel.is_ok(),
            "Benford analyzer kernel failed: {:?}",
            kernel.err()
        );
        let code = kernel.unwrap();
        assert!(code.contains("digit_counts"));
        assert!(code.contains("atomic"));
    }

    #[test]
    fn test_suspense_detector_kernel() {
        let kernel = generate_suspense_detector_kernel();
        assert!(
            kernel.is_ok(),
            "Suspense detector kernel failed: {:?}",
            kernel.err()
        );
        let code = kernel.unwrap();
        assert!(code.contains("suspense_scores"));
        assert!(code.contains("balance_debit"));
    }
}
