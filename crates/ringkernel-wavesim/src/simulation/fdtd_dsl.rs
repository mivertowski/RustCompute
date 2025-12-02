//! FDTD kernel defined using the Rust DSL.
//!
//! This module demonstrates how to define stencil kernels in pure Rust
//! that get transpiled to CUDA at compile time.
//!
//! The generated CUDA code is equivalent to the handwritten version in
//! `shaders/fdtd_tile.cu`, but is written in a high-level Rust DSL.

#[cfg(feature = "cuda-codegen")]
use ringkernel_cuda_codegen::{transpile_stencil_kernel, Grid, StencilConfig};

// Re-export or define GridPos
#[cfg(feature = "cuda-codegen")]
pub use ringkernel_cuda_codegen::GridPos;

// Stub GridPos for when cuda-codegen is not enabled
#[cfg(not(feature = "cuda-codegen"))]
#[derive(Debug, Clone, Copy)]
pub struct GridPos {
    _private: (),
}

#[cfg(not(feature = "cuda-codegen"))]
impl GridPos {
    /// Placeholder - actual implementation is generated.
    pub fn idx(&self) -> usize { 0 }
    /// Placeholder - actual implementation is generated.
    pub fn north<T: Copy>(&self, _buf: &[T]) -> T { unsafe { std::mem::zeroed() } }
    /// Placeholder - actual implementation is generated.
    pub fn south<T: Copy>(&self, _buf: &[T]) -> T { unsafe { std::mem::zeroed() } }
    /// Placeholder - actual implementation is generated.
    pub fn east<T: Copy>(&self, _buf: &[T]) -> T { unsafe { std::mem::zeroed() } }
    /// Placeholder - actual implementation is generated.
    pub fn west<T: Copy>(&self, _buf: &[T]) -> T { unsafe { std::mem::zeroed() } }
}

/// FDTD wave equation kernel for 16x16 tiles with 1-cell halo.
///
/// This is a pure Rust implementation of the FDTD step that can be
/// used for CPU fallback and verification.
pub fn fdtd_wave_step_cpu(
    pressure: &[f32],
    pressure_prev: &mut [f32],
    c2: f32,
    damping: f32,
    idx: usize,
    buffer_width: usize,
) {
    let p = pressure[idx];
    let p_prev = pressure_prev[idx];

    // Compute discrete Laplacian using neighbor access
    let laplacian = pressure[idx - buffer_width]  // North
        + pressure[idx + buffer_width]             // South
        + pressure[idx + 1]                        // East
        + pressure[idx - 1]                        // West
        - 4.0 * p;

    // FDTD update: p_new = 2*p - p_prev + c²*laplacian
    let p_new = 2.0 * p - p_prev + c2 * laplacian;

    // Apply damping and store result
    pressure_prev[idx] = p_new * damping;
}

/// Generate CUDA source for the FDTD kernel using the DSL transpiler.
///
/// This function demonstrates using the transpiler directly to generate
/// CUDA code from a Rust function AST.
#[cfg(feature = "cuda-codegen")]
pub fn generate_fdtd_cuda() -> String {
    use syn::parse_quote;

    // Define the kernel as a syn AST
    let kernel_fn: syn::ItemFn = parse_quote! {
        fn fdtd_wave_step(
            pressure: &[f32],
            pressure_prev: &mut [f32],
            c2: f32,
            damping: f32,
            pos: GridPos,
        ) {
            let p = pressure[pos.idx()];
            let p_prev = pressure_prev[pos.idx()];
            let laplacian = pos.north(pressure)
                + pos.south(pressure)
                + pos.east(pressure)
                + pos.west(pressure)
                - 4.0 * p;
            let p_new = 2.0 * p - p_prev + c2 * laplacian;
            pressure_prev[pos.idx()] = p_new * damping;
        }
    };

    // Configure the stencil
    let config = StencilConfig::new("fdtd_wave_step")
        .with_grid(Grid::Grid2D)
        .with_tile_size(16, 16)
        .with_halo(1);

    // Transpile to CUDA
    match transpile_stencil_kernel(&kernel_fn, &config) {
        Ok(cuda) => cuda,
        Err(e) => format!("// Transpilation error: {}", e),
    }
}

#[cfg(not(feature = "cuda-codegen"))]
pub fn generate_fdtd_cuda() -> String {
    "// CUDA codegen not enabled".to_string()
}

/// Get the handwritten CUDA source for comparison.
pub fn handwritten_fdtd_cuda() -> &'static str {
    include_str!("../shaders/fdtd_tile.cu")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fdtd_step() {
        // Create a simple 4x4 buffer (with 1-cell halo = 6x6 total)
        let buffer_width = 6;
        let mut pressure = vec![0.0f32; 36];
        let mut pressure_prev = vec![0.0f32; 36];

        // Set center cell to 1.0
        pressure[14] = 1.0; // (2,2) in 6x6 buffer

        // Run one FDTD step on the center cell
        fdtd_wave_step_cpu(&pressure, &mut pressure_prev, 0.25, 0.99, 14, buffer_width);

        // The Laplacian of a single spike is -4 (since neighbors are 0)
        // p_new = 2*1 - 0 + 0.25*(-4) = 2 - 1 = 1.0
        // With damping: 1.0 * 0.99 = 0.99
        assert!((pressure_prev[14] - 0.99).abs() < 0.01);
    }

    #[test]
    #[cfg(feature = "cuda-codegen")]
    fn test_generated_cuda_source() {
        let source = generate_fdtd_cuda();

        // Check that it looks like valid CUDA
        assert!(source.contains("extern \"C\" __global__"), "Should have CUDA kernel declaration");
        assert!(source.contains("threadIdx.x"), "Should use thread index X");
        assert!(source.contains("threadIdx.y"), "Should use thread index Y");
        assert!(source.contains("buffer_width = 18"), "Should have correct buffer width (16 + 2*1)");

        println!("Generated CUDA:\n{}", source);
    }

    #[test]
    #[cfg(feature = "cuda-codegen")]
    fn test_cuda_source_structure() {
        let source = generate_fdtd_cuda();

        // Verify key structural elements
        assert!(source.contains("if (lx >= 16 || ly >= 16) return;"), "Should have bounds check");
        assert!(source.contains("float p ="), "Should have pressure variable");

        // Verify stencil access pattern (using buffer_width constant)
        assert!(source.contains("- 18") || source.contains("- buffer_width"), "Should access north neighbor");
        assert!(source.contains("+ 18") || source.contains("+ buffer_width"), "Should access south neighbor");
    }
}
