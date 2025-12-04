//! CUDA kernel code generation using ringkernel-cuda-codegen.
//!
//! Transpiles accounting network analysis kernels from Rust DSL to CUDA C.

use ringkernel_cuda_codegen::transpile_global_kernel;

/// Generated CUDA code for accounting network kernels.
pub struct GeneratedKernels {
    /// Suspense detection kernel.
    pub suspense_detection: String,
    /// GAAP violation kernel.
    pub gaap_violation: String,
    /// Benford analysis kernel.
    pub benford_analysis: String,
}

impl GeneratedKernels {
    /// Generate all CUDA kernels.
    pub fn generate() -> Result<Self, String> {
        Ok(Self {
            suspense_detection: generate_suspense_detection_kernel()?,
            gaap_violation: generate_gaap_violation_kernel()?,
            benford_analysis: generate_benford_analysis_kernel()?,
        })
    }
}

/// Generate suspense account detection kernel.
/// Each thread processes one account.
fn generate_suspense_detection_kernel() -> Result<String, String> {
    let kernel_fn: syn::ItemFn = syn::parse_quote! {
        fn suspense_detection(
            balance_debit: &[f64],
            balance_credit: &[f64],
            risk_scores: &[f32],
            inflow_counts: &[u32],
            outflow_counts: &[u32],
            suspense_scores: &mut [f32],
            n_accounts: i32
        ) {
            let idx = block_idx_x() * block_dim_x() + thread_idx_x();
            if idx >= n_accounts { return; }

            let i = idx as usize;

            // Suspense indicators
            let mut score: f32 = 0.0;

            // Check if balance is a round number (potential suspense)
            let balance = balance_debit[i] - balance_credit[i];
            let abs_balance = if balance > 0.0 { balance } else { -balance };

            // Check if balance is a round number (divisible by 1000)
            // Use integer comparison: abs_balance >= 1000 and the last 3 digits are ~0
            // We can approximate: if (abs_balance / 1000.0) is close to an integer
            let thousands = abs_balance / 1000.0;
            let truncated = (thousands as i32) as f64;
            let fractional_part = thousands - truncated;
            // If fractional part is very small, it's a round thousand
            if abs_balance >= 1000.0 && fractional_part < 0.001 {
                score = score + 0.3;
            }

            // Check for high risk indicators
            if risk_scores[i] > 0.5 {
                score = score + 0.4;
            }

            // Check flow imbalance
            let inflows = inflow_counts[i] as f32;
            let outflows = outflow_counts[i] as f32;
            let flow_ratio = if outflows > 0.0 {
                inflows / outflows
            } else {
                10.0
            };

            if flow_ratio > 5.0 {
                score = score + 0.3;
            }

            if score > 1.0 {
                score = 1.0;
            }

            suspense_scores[i] = score;
        }
    };

    transpile_global_kernel(&kernel_fn)
        .map_err(|e| format!("Failed to transpile suspense_detection: {}", e))
}

/// Generate GAAP violation detection kernel.
/// Each thread processes one flow and checks for violations.
fn generate_gaap_violation_kernel() -> Result<String, String> {
    let kernel_fn: syn::ItemFn = syn::parse_quote! {
        fn gaap_violation(
            flow_source: &[u16],
            flow_target: &[u16],
            account_types: &[u8],
            violation_flags: &mut [u8],
            n_flows: i32
        ) {
            let idx = block_idx_x() * block_dim_x() + thread_idx_x();
            if idx >= n_flows { return; }

            let i = idx as usize;
            let source_idx = flow_source[i] as usize;
            let target_idx = flow_target[i] as usize;
            let source_type = account_types[source_idx];
            let target_type = account_types[target_idx];

            // AccountType: 0=Asset, 1=Liability, 2=Equity, 3=Revenue, 4=Expense, 5=Contra

            // Check for Revenue -> Asset direct (Revenue=3, Asset=0)
            // Should go through A/R
            if source_type == 3 && target_type == 0 {
                violation_flags[i] = 1;
                return;
            }

            // Check for Revenue -> Expense direct (Revenue=3, Expense=4)
            // Improper netting
            if source_type == 3 && target_type == 4 {
                violation_flags[i] = 2;
                return;
            }

            violation_flags[i] = 0;
        }
    };

    transpile_global_kernel(&kernel_fn)
        .map_err(|e| format!("Failed to transpile gaap_violation: {}", e))
}

/// Generate Benford's Law analysis kernel.
/// Each thread processes one amount and atomically updates digit counts.
fn generate_benford_analysis_kernel() -> Result<String, String> {
    let kernel_fn: syn::ItemFn = syn::parse_quote! {
        fn benford_analysis(
            amounts: &[f64],
            digit_counts: &mut [u32],
            n_amounts: i32
        ) {
            let idx = block_idx_x() * block_dim_x() + thread_idx_x();
            if idx >= n_amounts { return; }

            let i = idx as usize;
            let amount = amounts[i];
            let abs_amount = if amount > 0.0 { amount } else { -amount };

            // Skip values less than 1 (no leading digit)
            if abs_amount < 1.0 { return; }

            // Extract first digit by repeatedly dividing
            let mut value = abs_amount;

            // Reduce to single digit range using cascading division
            // Each division is conditional but not nested (at top level)
            let v1 = if value >= 10000000000000.0 { value / 10000000000000.0 } else { value };
            let v2 = if v1 >= 1000000.0 { v1 / 1000000.0 } else { v1 };
            let v3 = if v2 >= 1000.0 { v2 / 1000.0 } else { v2 };
            let v4 = if v3 >= 100.0 { v3 / 100.0 } else { v3 };
            let v5 = if v4 >= 10.0 { v4 / 10.0 } else { v4 };

            let first_digit = v5 as i32;

            // Valid first digits are 1-9
            if first_digit < 1 { return; }
            if first_digit > 9 { return; }

            atomic_add(&mut digit_counts[(first_digit - 1) as usize], 1);
        }
    };

    transpile_global_kernel(&kernel_fn)
        .map_err(|e| format!("Failed to transpile benford_analysis: {}", e))
}

/// Generate PageRank initialization kernel (sets initial values).
#[allow(dead_code)]
fn generate_pagerank_init_kernel() -> Result<String, String> {
    let kernel_fn: syn::ItemFn = syn::parse_quote! {
        fn pagerank_init(
            pagerank: &mut [f32],
            n_nodes: i32
        ) {
            let idx = block_idx_x() * block_dim_x() + thread_idx_x();
            if idx >= n_nodes { return; }

            pagerank[idx as usize] = 1.0 / n_nodes as f32;
        }
    };

    transpile_global_kernel(&kernel_fn)
        .map_err(|e| format!("Failed to transpile pagerank_init: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_kernels() {
        let result = GeneratedKernels::generate();
        assert!(
            result.is_ok(),
            "Failed to generate kernels: {:?}",
            result.err()
        );

        let kernels = result.unwrap();
        assert!(kernels.suspense_detection.contains("__global__"));
        assert!(kernels.gaap_violation.contains("__global__"));
        assert!(kernels.benford_analysis.contains("__global__"));
    }

    #[test]
    fn test_suspense_detection_kernel() {
        let result = generate_suspense_detection_kernel();
        assert!(result.is_ok(), "Error: {:?}", result.err());
        let cuda_code = result.unwrap();
        assert!(cuda_code.contains("suspense_detection"));
        assert!(cuda_code.contains("blockIdx.x"));
    }

    #[test]
    fn test_gaap_violation_kernel() {
        let result = generate_gaap_violation_kernel();
        assert!(result.is_ok(), "Error: {:?}", result.err());
        let cuda_code = result.unwrap();
        assert!(cuda_code.contains("gaap_violation"));
    }

    #[test]
    fn test_benford_kernel() {
        let result = generate_benford_analysis_kernel();
        assert!(result.is_ok(), "Error: {:?}", result.err());
        let cuda_code = result.unwrap();
        assert!(cuda_code.contains("benford_analysis"));
        assert!(cuda_code.contains("atomicAdd"));
    }

    #[test]
    fn test_pagerank_init_kernel() {
        let result = generate_pagerank_init_kernel();
        assert!(result.is_ok(), "Error: {:?}", result.err());
        let cuda_code = result.unwrap();
        assert!(cuda_code.contains("pagerank_init"));
    }
}
