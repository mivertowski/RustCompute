//! # CUDA Global Kernel Transpilation Example
//!
//! Demonstrates transpiling Rust functions to CUDA `__global__` kernels.
//!
//! ## Run this example:
//! ```bash
//! cargo run -p ringkernel --example global_kernel
//! ```
//!
//! ## What this demonstrates:
//!
//! - DSL intrinsics: `block_idx_x()`, `thread_idx_x()`, `block_dim_x()`
//! - Early return patterns for bounds checking
//! - Array/slice indexing with type casting
//! - Generated `extern "C" __global__` kernel signature

use ringkernel_cuda_codegen::transpile_global_kernel;
use syn::parse_quote;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== CUDA Global Kernel Transpilation ===\n");

    // ========== Example 1: SAXPY Kernel ==========
    println!("--- Example 1: SAXPY (y = a*x + y) ---\n");

    let saxpy: syn::ItemFn = parse_quote! {
        fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
            let idx = block_idx_x() * block_dim_x() + thread_idx_x();
            if idx >= n {
                return;
            }
            y[idx as usize] = a * x[idx as usize] + y[idx as usize];
        }
    };

    let cuda = transpile_global_kernel(&saxpy)?;
    println!("Rust DSL:\n");
    println!("  fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {{");
    println!("      let idx = block_idx_x() * block_dim_x() + thread_idx_x();");
    println!("      if idx >= n {{ return; }}");
    println!("      y[idx as usize] = a * x[idx as usize] + y[idx as usize];");
    println!("  }}\n");
    println!("Generated CUDA:\n{}\n", cuda);

    // ========== Example 2: Halo Exchange Kernel ==========
    println!("--- Example 2: Halo Exchange (from WaveSim) ---\n");

    let halo: syn::ItemFn = parse_quote! {
        fn exchange_halos(buffer: &mut [f32], copies: &[u32], num_copies: i32) {
            let idx = block_idx_x() * block_dim_x() + thread_idx_x();
            if idx >= num_copies {
                return;
            }
            let src = copies[(idx * 2) as usize] as usize;
            let dst = copies[(idx * 2 + 1) as usize] as usize;
            buffer[dst] = buffer[src];
        }
    };

    let cuda = transpile_global_kernel(&halo)?;
    println!("Rust DSL:\n");
    println!("  fn exchange_halos(buffer: &mut [f32], copies: &[u32], num_copies: i32) {{");
    println!("      let idx = block_idx_x() * block_dim_x() + thread_idx_x();");
    println!("      if idx >= num_copies {{ return; }}");
    println!("      let src = copies[(idx * 2) as usize] as usize;");
    println!("      let dst = copies[(idx * 2 + 1) as usize] as usize;");
    println!("      buffer[dst] = buffer[src];");
    println!("  }}\n");
    println!("Generated CUDA:\n{}\n", cuda);

    // ========== Example 3: Array Initialization ==========
    println!("--- Example 3: Array Initialization ---\n");

    let init: syn::ItemFn = parse_quote! {
        fn init_array(data: &mut [f32], value: f32, n: i32) {
            let i = block_idx_x() * block_dim_x() + thread_idx_x();
            if i < n {
                data[i as usize] = value;
            }
        }
    };

    let cuda = transpile_global_kernel(&init)?;
    println!("Rust DSL:\n");
    println!("  fn init_array(data: &mut [f32], value: f32, n: i32) {{");
    println!("      let i = block_idx_x() * block_dim_x() + thread_idx_x();");
    println!("      if i < n {{ data[i as usize] = value; }}");
    println!("  }}\n");
    println!("Generated CUDA:\n{}\n", cuda);

    // ========== DSL Summary ==========
    println!("=== DSL Intrinsics Summary ===\n");
    println!("Thread/Block Indices:");
    println!("  thread_idx_x(), thread_idx_y(), thread_idx_z() -> threadIdx.x/y/z");
    println!("  block_idx_x(), block_idx_y(), block_idx_z()   -> blockIdx.x/y/z");
    println!("  block_dim_x(), block_dim_y(), block_dim_z()   -> blockDim.x/y/z");
    println!("  grid_dim_x(), grid_dim_y(), grid_dim_z()      -> gridDim.x/y/z\n");

    println!("Synchronization:");
    println!("  sync_threads() -> __syncthreads()\n");

    println!("Type Mappings:");
    println!("  f32 -> float, f64 -> double");
    println!("  i32 -> int, u32 -> unsigned int");
    println!("  &[T] -> const T* __restrict__");
    println!("  &mut [T] -> T* __restrict__\n");

    println!("=== Global Kernel Example Complete! ===");

    Ok(())
}
