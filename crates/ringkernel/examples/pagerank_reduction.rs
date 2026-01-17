//! PageRank with Global Reduction for Dangling Nodes
//!
//! This example demonstrates how to use RingKernel's global reduction primitives
//! to correctly compute PageRank on graphs with dangling nodes (nodes with no
//! outgoing edges).
//!
//! # The Problem
//!
//! Standard PageRank formula:
//! ```text
//! PR(v) = (1-d)/N + d × Σ(PR(u)/out_degree(u)) for all u→v
//! ```
//!
//! For nodes with `out_degree = 0` (dangling nodes), their PageRank "leaks" out
//! of the system. The solution redistributes this mass:
//! ```text
//! dangling_sum = Σ PR(u) for all u where out_degree(u) = 0
//! PR(v) = (1-d)/N + d × (dangling_sum/N + Σ(PR(u)/out_degree(u)))
//! ```
//!
//! Computing `dangling_sum` requires a global reduction across all GPU threads.
//!
//! # Architecture
//!
//! ```text
//! Phase 1: Accumulate              Phase 2: Apply
//! ┌─────────────────────┐          ┌─────────────────────┐
//! │ Each thread:        │          │ Each thread:        │
//! │ if out_degree == 0: │  sync    │ new_rank = base +   │
//! │   contrib = rank    │ ──────>  │   d*(sum + dangling │
//! │ block_reduce(contrib)          │      / N)           │
//! │ atomicAdd(dangling) │          │                     │
//! └─────────────────────┘          └─────────────────────┘
//! ```
//!
//! # Running the Example
//!
//! ```bash
//! # Requires NVIDIA GPU with CUDA
//! cargo run -p ringkernel --example pagerank_reduction --features cuda
//! ```

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("PageRank with Global Reduction for Dangling Nodes");
    println!("================================================\n");

    // Example graphs to demonstrate the difference
    demonstrate_triangle_graph()?;
    demonstrate_star_graph()?;
    demonstrate_chain_with_sink()?;

    // Show generated CUDA kernel code
    generate_kernel_code()?;

    println!("\n=== Summary ===");
    println!("Without global reduction, dangling nodes cause PageRank mass to leak,");
    println!("leading to incorrect results (ranks don't sum to 1.0).");
    println!("Global reduction enables proper redistribution of dangling mass.");

    Ok(())
}

/// Triangle graph: A → B → C → A (no dangling nodes)
fn demonstrate_triangle_graph() -> Result<(), Box<dyn Error>> {
    println!("=== Triangle Graph (no dangling nodes) ===");
    println!("A → B → C → A");
    println!();

    // Graph structure
    let num_nodes = 3;
    let out_degrees = [1u32, 1, 1]; // Each node has one outgoing edge
    let ranks = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let damping = 0.85;

    // No dangling nodes, so dangling_sum = 0
    let dangling_sum: f64 = ranks
        .iter()
        .zip(out_degrees.iter())
        .filter(|(_, &d)| d == 0)
        .map(|(&r, _)| r)
        .sum();

    println!("Dangling sum: {:.6}", dangling_sum);
    println!("Expected: 0.0 (no dangling nodes)");

    // PageRank iteration step
    let base = (1.0 - damping) / num_nodes as f64;
    let dangling_contrib = damping * dangling_sum / num_nodes as f64;

    for i in 0..num_nodes {
        // Incoming contribution (in triangle: from previous node)
        let incoming = ranks[(i + 2) % 3] / out_degrees[(i + 2) % 3] as f64;
        let new_rank = base + damping * (incoming + dangling_contrib / damping);
        println!(
            "Node {}: base={:.4} + d*incoming={:.4} + d*dangling={:.4} = {:.6}",
            i,
            base,
            damping * incoming,
            dangling_contrib,
            new_rank
        );
    }

    println!("Sum of ranks: {:.6}", ranks.iter().sum::<f64>());
    println!("✓ Triangle graph works without reduction\n");

    Ok(())
}

/// Star graph: Center node A has edges to B, C, D (which have no outgoing edges)
fn demonstrate_star_graph() -> Result<(), Box<dyn Error>> {
    println!("=== Star Graph (75% dangling nodes) ===");
    println!("A → {{B, C, D}} (B, C, D have no outgoing edges)");
    println!();

    let num_nodes = 4;
    let out_degrees = [3u32, 0, 0, 0]; // A has 3 outgoing, B/C/D have 0
    let ranks = [0.25, 0.25, 0.25, 0.25];
    let damping = 0.85;

    // Dangling nodes: B, C, D
    let dangling_sum: f64 = ranks
        .iter()
        .zip(out_degrees.iter())
        .filter(|(_, &d)| d == 0)
        .map(|(&r, _)| r)
        .sum();

    println!("Dangling sum: {:.6} (from B + C + D)", dangling_sum);
    println!("This is 75% of total rank mass!\n");

    println!("WITHOUT reduction (incorrect):");
    let base = (1.0 - damping) / num_nodes as f64;
    let mut sum_without = 0.0;
    for i in 0..num_nodes {
        let incoming = if i == 0 {
            0.0 // A has no incoming edges
        } else {
            ranks[0] / out_degrees[0] as f64 // B, C, D receive from A
        };
        let new_rank = base + damping * incoming;
        sum_without += new_rank;
        println!("Node {}: {:.6}", i, new_rank);
    }
    println!("Sum: {:.6} (should be 1.0!)", sum_without);
    println!("⚠ Rank mass leaked through dangling nodes!\n");

    println!("WITH reduction (correct):");
    let dangling_contrib = dangling_sum / num_nodes as f64;
    let mut sum_with = 0.0;
    for i in 0..num_nodes {
        let incoming = if i == 0 {
            0.0
        } else {
            ranks[0] / out_degrees[0] as f64
        };
        let new_rank = base + damping * (incoming + dangling_contrib);
        sum_with += new_rank;
        println!("Node {}: {:.6}", i, new_rank);
    }
    println!("Sum: {:.6}", sum_with);
    println!("✓ Rank mass preserved with global reduction\n");

    Ok(())
}

/// Chain with sink: A → B → C → D (D is a sink/dangling node)
fn demonstrate_chain_with_sink() -> Result<(), Box<dyn Error>> {
    println!("=== Chain with Sink (25% dangling) ===");
    println!("A → B → C → D (D has no outgoing edges)");
    println!();

    let num_nodes = 4;
    let out_degrees = [1u32, 1, 1, 0]; // D is dangling
    let ranks = [0.25, 0.25, 0.25, 0.25];
    let _damping = 0.85;

    let dangling_sum: f64 = ranks
        .iter()
        .zip(out_degrees.iter())
        .filter(|(_, &d)| d == 0)
        .map(|(&r, _)| r)
        .sum();

    println!("Dangling sum: {:.6} (from D only)", dangling_sum);

    println!(
        "\nDangling contribution per node: {:.6}",
        dangling_sum / num_nodes as f64
    );
    println!("This represents the redistributed mass that would otherwise be lost.\n");

    Ok(())
}

/// Generate and display the CUDA kernel code that uses reduction
fn generate_kernel_code() -> Result<(), Box<dyn Error>> {
    use ringkernel_cuda_codegen::{
        generate_reduction_helpers, reduction_intrinsics::ReductionOp as CodegenReductionOp,
        ReductionCodegenConfig,
    };

    println!("=== Generated CUDA Reduction Helpers ===\n");

    let config = ReductionCodegenConfig {
        block_size: 256,
        value_type: "double".to_string(),
        op: CodegenReductionOp::Sum,
        use_cooperative: true,
        generate_helpers: true,
    };

    let helpers = generate_reduction_helpers(&config);
    println!("Reduction helper functions (cooperative groups):\n");
    println!("{}", &helpers[..helpers.len().min(2000)]); // Truncate for display

    // Show inline reduction code
    use ringkernel_cuda_codegen::generate_inline_reduce_and_broadcast;

    println!("\n=== Inline Reduce-and-Broadcast Code ===\n");
    let inline_code = generate_inline_reduce_and_broadcast(
        "my_dangling_contrib",
        "shared_reduce",
        "&dangling_sum",
        "global_dangling",
        "double",
        256,
        &CodegenReductionOp::Sum,
        true,
    );
    println!("{}", inline_code);

    // Show PageRank kernel structure
    println!("\n=== PageRank Kernel Structure ===\n");
    println!(
        r#"// PageRank kernel with dangling node handling
extern "C" __global__ void pagerank_iteration(
    const double* __restrict__ ranks,
    double* __restrict__ new_ranks,
    const uint32_t* __restrict__ out_degrees,
    const uint64_t* __restrict__ row_ptrs,
    const uint32_t* __restrict__ col_indices,
    double* __restrict__ dangling_accumulator,  // ReductionBuffer
    uint32_t num_nodes,
    double damping
) {{
    // Shared memory for block reduction
    __shared__ double shared_reduce[256];

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    // Phase 1: Compute dangling contribution
    double my_dangling = (out_degrees[idx] == 0) ? ranks[idx] : 0.0;

    // Reduce-and-broadcast: all threads get global dangling sum
    // reduce_and_broadcast code // <-- Inline reduction code inserted here

    // Phase 2: Compute new rank with dangling redistribution
    double base = (1.0 - damping) / num_nodes;
    double dangling_contrib = damping * global_dangling / num_nodes;

    // Sum incoming rank contributions
    double incoming_sum = 0.0;
    uint64_t start = row_ptrs[idx];
    uint64_t end = row_ptrs[idx + 1];
    for (uint64_t i = start; i < end; i++) {{
        uint32_t src = col_indices[i];
        if (out_degrees[src] > 0) {{
            incoming_sum += ranks[src] / out_degrees[src];
        }}
    }}

    new_ranks[idx] = base + damping * incoming_sum + dangling_contrib;
}}
"#
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_no_dangling() {
        let out_degrees = [1u32, 1, 1];
        let ranks = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

        let dangling_sum: f64 = ranks
            .iter()
            .zip(out_degrees.iter())
            .filter(|(_, &d)| d == 0)
            .map(|(&r, _)| r)
            .sum();

        assert!((dangling_sum - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_star_dangling_sum() {
        let out_degrees = [3u32, 0, 0, 0];
        let ranks = [0.25, 0.25, 0.25, 0.25];

        let dangling_sum: f64 = ranks
            .iter()
            .zip(out_degrees.iter())
            .filter(|(_, &d)| d == 0)
            .map(|(&r, _)| r)
            .sum();

        assert!((dangling_sum - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_mass_conservation_with_reduction() {
        let num_nodes = 4;
        let out_degrees = [3u32, 0, 0, 0];
        let ranks = [0.25, 0.25, 0.25, 0.25];
        let damping = 0.85;

        let dangling_sum: f64 = ranks
            .iter()
            .zip(out_degrees.iter())
            .filter(|(_, &d)| d == 0)
            .map(|(&r, _)| r)
            .sum();

        let base = (1.0 - damping) / num_nodes as f64;
        let dangling_contrib = dangling_sum / num_nodes as f64;

        let mut sum = 0.0;
        for i in 0..num_nodes {
            let incoming = if i == 0 {
                0.0
            } else {
                ranks[0] / out_degrees[0] as f64
            };
            sum += base + damping * (incoming + dangling_contrib);
        }

        // With proper dangling handling, sum should be ~1.0
        assert!((sum - 1.0).abs() < 0.01);
    }
}
