//! # CUDA Stencil Kernel Transpilation Example
//!
//! Demonstrates transpiling Rust stencil functions to CUDA kernels
//! using the `GridPos` abstraction for neighbor access.
//!
//! ## Run this example:
//! ```bash
//! cargo run -p ringkernel --example stencil_kernel
//! ```
//!
//! ## What this demonstrates:
//!
//! - GridPos intrinsics: `pos.north()`, `pos.south()`, `pos.east()`, `pos.west()`
//! - `pos.idx()` for linear array index
//! - `pos.at(buf, dx, dy)` for arbitrary neighbor offsets
//! - Automatic tiling with configurable tile size and halo region

use ringkernel_cuda_codegen::{transpile_stencil_kernel, StencilConfig};
use syn::parse_quote;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== CUDA Stencil Kernel Transpilation ===\n");

    // ========== Example 1: FDTD Wave Equation ==========
    println!("--- Example 1: FDTD Wave Equation (from WaveSim) ---\n");

    let fdtd: syn::ItemFn = parse_quote! {
        fn fdtd_step(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {
            let curr = p[pos.idx()];
            let prev = p_prev[pos.idx()];
            let laplacian = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p)
                           - 4.0 * curr;
            p_prev[pos.idx()] = 2.0 * curr - prev + c2 * laplacian;
        }
    };

    let config = StencilConfig::new("fdtd")
        .with_tile_size(16, 16)
        .with_halo(1);

    let cuda = transpile_stencil_kernel(&fdtd, &config)?;

    println!("Rust DSL:\n");
    println!("  fn fdtd_step(p: &[f32], p_prev: &mut [f32], c2: f32, pos: GridPos) {{");
    println!("      let curr = p[pos.idx()];");
    println!("      let prev = p_prev[pos.idx()];");
    println!("      let laplacian = pos.north(p) + pos.south(p) + pos.east(p) + pos.west(p)");
    println!("                     - 4.0 * curr;");
    println!("      p_prev[pos.idx()] = 2.0 * curr - prev + c2 * laplacian;");
    println!("  }}\n");
    println!("Config: tile_size=16x16, halo=1\n");
    println!("Generated CUDA ({} lines):\n", cuda.lines().count());

    // Print first 60 lines to show structure
    for (i, line) in cuda.lines().take(60).enumerate() {
        println!("{:3}: {}", i + 1, line);
    }
    if cuda.lines().count() > 60 {
        println!("... ({} more lines)", cuda.lines().count() - 60);
    }
    println!();

    // ========== Example 2: Heat Diffusion ==========
    println!("--- Example 2: Heat Diffusion ---\n");

    let heat: syn::ItemFn = parse_quote! {
        fn heat_diffusion(temp: &[f32], temp_new: &mut [f32], alpha: f32, pos: GridPos) {
            let t = temp[pos.idx()];
            let neighbors = pos.north(temp) + pos.south(temp) + pos.east(temp) + pos.west(temp);
            temp_new[pos.idx()] = t + alpha * (neighbors - 4.0 * t);
        }
    };

    let config = StencilConfig::new("heat")
        .with_tile_size(32, 8)
        .with_halo(1);

    let cuda = transpile_stencil_kernel(&heat, &config)?;

    println!("Rust DSL:\n");
    println!("  fn heat_diffusion(temp: &[f32], temp_new: &mut [f32], alpha: f32, pos: GridPos) {{");
    println!("      let t = temp[pos.idx()];");
    println!("      let neighbors = pos.north(temp) + pos.south(temp) + pos.east(temp) + pos.west(temp);");
    println!("      temp_new[pos.idx()] = t + alpha * (neighbors - 4.0 * t);");
    println!("  }}\n");
    println!("Config: tile_size=32x8, halo=1\n");
    println!("Generated CUDA has {} lines\n", cuda.lines().count());

    // ========== Example 3: Simple Smooth (Box Filter) ==========
    println!("--- Example 3: Box Filter (5-point average) ---\n");

    let smooth: syn::ItemFn = parse_quote! {
        fn smooth(input: &[f32], output: &mut [f32], pos: GridPos) {
            let sum = pos.north(input) + pos.south(input) + pos.east(input)
                    + pos.west(input) + input[pos.idx()];
            output[pos.idx()] = sum * 0.2;
        }
    };

    let config = StencilConfig::new("smooth")
        .with_tile_size(16, 16)
        .with_halo(1);

    let cuda = transpile_stencil_kernel(&smooth, &config)?;

    println!("Rust DSL:\n");
    println!("  fn smooth(input: &[f32], output: &mut [f32], pos: GridPos) {{");
    println!("      let sum = pos.north(input) + pos.south(input) + pos.east(input)");
    println!("              + pos.west(input) + input[pos.idx()];");
    println!("      output[pos.idx()] = sum * 0.2;");
    println!("  }}\n");
    println!("Generated CUDA has {} lines\n", cuda.lines().count());

    // ========== GridPos Summary ==========
    println!("=== GridPos Intrinsics Summary ===\n");
    println!("Neighbor Access:");
    println!("  pos.north(buf) -> buf[linear_index - width]  (y-1)");
    println!("  pos.south(buf) -> buf[linear_index + width]  (y+1)");
    println!("  pos.east(buf)  -> buf[linear_index + 1]      (x+1)");
    println!("  pos.west(buf)  -> buf[linear_index - 1]      (x-1)\n");

    println!("Index Access:");
    println!("  pos.idx()      -> linear index (y * width + x)\n");

    println!("Arbitrary Offset:");
    println!("  pos.at(buf, dx, dy) -> buf[(y+dy) * width + (x+dx)]\n");

    println!("Configuration:");
    println!("  StencilConfig::new(\"name\")");
    println!("      .with_tile_size(width, height)  // Thread block tile size");
    println!("      .with_halo(size)                // Halo region for neighbors");
    println!("      .with_grid(Grid::Grid2D)        // 2D or 3D grid\n");

    println!("=== Stencil Kernel Example Complete! ===");

    Ok(())
}
