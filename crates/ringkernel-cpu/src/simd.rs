//! SIMD-accelerated operations for CPU backend.
//!
//! This module provides high-performance implementations of common GPU-like
//! operations using SIMD (Single Instruction, Multiple Data) instructions.
//!
//! # Operations
//!
//! - **Vector Operations**: SAXPY, dot product, element-wise operations
//! - **Reductions**: Sum, min, max, mean
//! - **Stencil Operations**: 2D/3D Laplacian for FDTD simulations
//! - **Array Operations**: Fill, copy, compare
//!
//! # Example
//!
//! ```
//! use ringkernel_cpu::simd::SimdOps;
//!
//! // SAXPY: y = a * x + y
//! let x = vec![1.0f32; 1024];
//! let mut y = vec![2.0f32; 1024];
//! SimdOps::saxpy(2.0, &x, &mut y);
//!
//! // Reduction
//! let sum = SimdOps::sum_f32(&y);
//! ```

use rayon::prelude::*;
use wide::{f32x8, f64x4, i32x8};

/// SIMD-accelerated operations.
pub struct SimdOps;

// ============================================================================
// VECTOR OPERATIONS
// ============================================================================

impl SimdOps {
    /// SAXPY: y = a * x + y (f32)
    ///
    /// Single-precision A*X Plus Y operation, fundamental to linear algebra.
    #[inline]
    pub fn saxpy(a: f32, x: &[f32], y: &mut [f32]) {
        let n = x.len().min(y.len());
        let a_vec = f32x8::splat(a);

        // Process 8 elements at a time
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let x_vec = f32x8::new([
                x[offset],
                x[offset + 1],
                x[offset + 2],
                x[offset + 3],
                x[offset + 4],
                x[offset + 5],
                x[offset + 6],
                x[offset + 7],
            ]);
            let y_vec = f32x8::new([
                y[offset],
                y[offset + 1],
                y[offset + 2],
                y[offset + 3],
                y[offset + 4],
                y[offset + 5],
                y[offset + 6],
                y[offset + 7],
            ]);

            let result = a_vec * x_vec + y_vec;
            let arr: [f32; 8] = result.into();
            y[offset..offset + 8].copy_from_slice(&arr);
        }

        // Handle remainder
        let tail_start = chunks * 8;
        for i in 0..remainder {
            y[tail_start + i] += a * x[tail_start + i];
        }
    }

    /// DAXPY: y = a * x + y (f64)
    ///
    /// Double-precision A*X Plus Y operation.
    #[inline]
    pub fn daxpy(a: f64, x: &[f64], y: &mut [f64]) {
        let n = x.len().min(y.len());
        let a_vec = f64x4::splat(a);

        // Process 4 elements at a time
        let chunks = n / 4;
        let remainder = n % 4;

        for i in 0..chunks {
            let offset = i * 4;
            let x_vec = f64x4::new([x[offset], x[offset + 1], x[offset + 2], x[offset + 3]]);
            let y_vec = f64x4::new([y[offset], y[offset + 1], y[offset + 2], y[offset + 3]]);

            let result = a_vec * x_vec + y_vec;
            let arr: [f64; 4] = result.into();
            y[offset..offset + 4].copy_from_slice(&arr);
        }

        // Handle remainder
        let tail_start = chunks * 4;
        for i in 0..remainder {
            y[tail_start + i] += a * x[tail_start + i];
        }
    }

    /// Element-wise addition: z = x + y
    #[inline]
    pub fn add_f32(x: &[f32], y: &[f32], z: &mut [f32]) {
        let n = x.len().min(y.len()).min(z.len());
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let x_vec = f32x8::new([
                x[offset],
                x[offset + 1],
                x[offset + 2],
                x[offset + 3],
                x[offset + 4],
                x[offset + 5],
                x[offset + 6],
                x[offset + 7],
            ]);
            let y_vec = f32x8::new([
                y[offset],
                y[offset + 1],
                y[offset + 2],
                y[offset + 3],
                y[offset + 4],
                y[offset + 5],
                y[offset + 6],
                y[offset + 7],
            ]);

            let result = x_vec + y_vec;
            let arr: [f32; 8] = result.into();
            z[offset..offset + 8].copy_from_slice(&arr);
        }

        let tail_start = chunks * 8;
        for i in 0..remainder {
            z[tail_start + i] = x[tail_start + i] + y[tail_start + i];
        }
    }

    /// Element-wise subtraction: z = x - y
    #[inline]
    pub fn sub_f32(x: &[f32], y: &[f32], z: &mut [f32]) {
        let n = x.len().min(y.len()).min(z.len());
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let x_vec = f32x8::new([
                x[offset],
                x[offset + 1],
                x[offset + 2],
                x[offset + 3],
                x[offset + 4],
                x[offset + 5],
                x[offset + 6],
                x[offset + 7],
            ]);
            let y_vec = f32x8::new([
                y[offset],
                y[offset + 1],
                y[offset + 2],
                y[offset + 3],
                y[offset + 4],
                y[offset + 5],
                y[offset + 6],
                y[offset + 7],
            ]);

            let result = x_vec - y_vec;
            let arr: [f32; 8] = result.into();
            z[offset..offset + 8].copy_from_slice(&arr);
        }

        let tail_start = chunks * 8;
        for i in 0..remainder {
            z[tail_start + i] = x[tail_start + i] - y[tail_start + i];
        }
    }

    /// Element-wise multiplication: z = x * y
    #[inline]
    pub fn mul_f32(x: &[f32], y: &[f32], z: &mut [f32]) {
        let n = x.len().min(y.len()).min(z.len());
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let x_vec = f32x8::new([
                x[offset],
                x[offset + 1],
                x[offset + 2],
                x[offset + 3],
                x[offset + 4],
                x[offset + 5],
                x[offset + 6],
                x[offset + 7],
            ]);
            let y_vec = f32x8::new([
                y[offset],
                y[offset + 1],
                y[offset + 2],
                y[offset + 3],
                y[offset + 4],
                y[offset + 5],
                y[offset + 6],
                y[offset + 7],
            ]);

            let result = x_vec * y_vec;
            let arr: [f32; 8] = result.into();
            z[offset..offset + 8].copy_from_slice(&arr);
        }

        let tail_start = chunks * 8;
        for i in 0..remainder {
            z[tail_start + i] = x[tail_start + i] * y[tail_start + i];
        }
    }

    /// Dot product: sum(x * y)
    #[inline]
    pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
        let n = x.len().min(y.len());
        let chunks = n / 8;
        let remainder = n % 8;

        let mut acc = f32x8::splat(0.0);

        for i in 0..chunks {
            let offset = i * 8;
            let x_vec = f32x8::new([
                x[offset],
                x[offset + 1],
                x[offset + 2],
                x[offset + 3],
                x[offset + 4],
                x[offset + 5],
                x[offset + 6],
                x[offset + 7],
            ]);
            let y_vec = f32x8::new([
                y[offset],
                y[offset + 1],
                y[offset + 2],
                y[offset + 3],
                y[offset + 4],
                y[offset + 5],
                y[offset + 6],
                y[offset + 7],
            ]);

            acc += x_vec * y_vec;
        }

        // Horizontal sum
        let arr: [f32; 8] = acc.into();
        let mut sum: f32 = arr.iter().sum();

        // Handle remainder
        let tail_start = chunks * 8;
        for i in 0..remainder {
            sum += x[tail_start + i] * y[tail_start + i];
        }

        sum
    }

    /// Scale vector: x *= a
    #[inline]
    pub fn scale_f32(a: f32, x: &mut [f32]) {
        let n = x.len();
        let a_vec = f32x8::splat(a);
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let x_vec = f32x8::new([
                x[offset],
                x[offset + 1],
                x[offset + 2],
                x[offset + 3],
                x[offset + 4],
                x[offset + 5],
                x[offset + 6],
                x[offset + 7],
            ]);

            let result = a_vec * x_vec;
            let arr: [f32; 8] = result.into();
            x[offset..offset + 8].copy_from_slice(&arr);
        }

        let tail_start = chunks * 8;
        for i in 0..remainder {
            x[tail_start + i] *= a;
        }
    }
}

// ============================================================================
// REDUCTION OPERATIONS
// ============================================================================

impl SimdOps {
    /// Sum of f32 array using SIMD.
    #[inline]
    pub fn sum_f32(x: &[f32]) -> f32 {
        let n = x.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut acc = f32x8::splat(0.0);

        for i in 0..chunks {
            let offset = i * 8;
            let x_vec = f32x8::new([
                x[offset],
                x[offset + 1],
                x[offset + 2],
                x[offset + 3],
                x[offset + 4],
                x[offset + 5],
                x[offset + 6],
                x[offset + 7],
            ]);
            acc += x_vec;
        }

        let arr: [f32; 8] = acc.into();
        let mut sum: f32 = arr.iter().sum();

        let tail_start = chunks * 8;
        for i in 0..remainder {
            sum += x[tail_start + i];
        }

        sum
    }

    /// Sum of f64 array using SIMD.
    #[inline]
    pub fn sum_f64(x: &[f64]) -> f64 {
        let n = x.len();
        let chunks = n / 4;
        let remainder = n % 4;

        let mut acc = f64x4::splat(0.0);

        for i in 0..chunks {
            let offset = i * 4;
            let x_vec = f64x4::new([x[offset], x[offset + 1], x[offset + 2], x[offset + 3]]);
            acc += x_vec;
        }

        let arr: [f64; 4] = acc.into();
        let mut sum: f64 = arr.iter().sum();

        let tail_start = chunks * 4;
        for i in 0..remainder {
            sum += x[tail_start + i];
        }

        sum
    }

    /// Maximum of f32 array.
    #[inline]
    pub fn max_f32(x: &[f32]) -> f32 {
        if x.is_empty() {
            return f32::NEG_INFINITY;
        }

        let n = x.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut max_vec = f32x8::splat(f32::NEG_INFINITY);

        for i in 0..chunks {
            let offset = i * 8;
            let x_vec = f32x8::new([
                x[offset],
                x[offset + 1],
                x[offset + 2],
                x[offset + 3],
                x[offset + 4],
                x[offset + 5],
                x[offset + 6],
                x[offset + 7],
            ]);
            max_vec = max_vec.max(x_vec);
        }

        let arr: [f32; 8] = max_vec.into();
        let mut max_val = arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let tail_start = chunks * 8;
        for i in 0..remainder {
            max_val = max_val.max(x[tail_start + i]);
        }

        max_val
    }

    /// Minimum of f32 array.
    #[inline]
    pub fn min_f32(x: &[f32]) -> f32 {
        if x.is_empty() {
            return f32::INFINITY;
        }

        let n = x.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut min_vec = f32x8::splat(f32::INFINITY);

        for i in 0..chunks {
            let offset = i * 8;
            let x_vec = f32x8::new([
                x[offset],
                x[offset + 1],
                x[offset + 2],
                x[offset + 3],
                x[offset + 4],
                x[offset + 5],
                x[offset + 6],
                x[offset + 7],
            ]);
            min_vec = min_vec.min(x_vec);
        }

        let arr: [f32; 8] = min_vec.into();
        let mut min_val = arr.iter().cloned().fold(f32::INFINITY, f32::min);

        let tail_start = chunks * 8;
        for i in 0..remainder {
            min_val = min_val.min(x[tail_start + i]);
        }

        min_val
    }

    /// Mean of f32 array.
    #[inline]
    pub fn mean_f32(x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        Self::sum_f32(x) / x.len() as f32
    }
}

// ============================================================================
// STENCIL OPERATIONS
// ============================================================================

impl SimdOps {
    /// 2D Laplacian stencil (5-point).
    ///
    /// Computes: laplacian[i,j] = p[i-1,j] + p[i+1,j] + p[i,j-1] + p[i,j+1] - 4*p[i,j]
    ///
    /// This is the core operation for FDTD wave simulations.
    #[inline]
    pub fn laplacian_2d_f32(p: &[f32], laplacian: &mut [f32], width: usize, height: usize) {
        let four = f32x8::splat(4.0);

        // Skip boundary cells (halo of 1)
        for y in 1..height - 1 {
            let row_start = y * width;
            let row_above = (y - 1) * width;
            let row_below = (y + 1) * width;

            // Process 8 cells at a time
            let inner_width = width - 2;
            let chunks = inner_width / 8;
            let remainder = inner_width % 8;

            for chunk in 0..chunks {
                let x = 1 + chunk * 8;
                let idx = row_start + x;

                // Center
                let center = f32x8::new([
                    p[idx],
                    p[idx + 1],
                    p[idx + 2],
                    p[idx + 3],
                    p[idx + 4],
                    p[idx + 5],
                    p[idx + 6],
                    p[idx + 7],
                ]);

                // North (y - 1)
                let north_idx = row_above + x;
                let north = f32x8::new([
                    p[north_idx],
                    p[north_idx + 1],
                    p[north_idx + 2],
                    p[north_idx + 3],
                    p[north_idx + 4],
                    p[north_idx + 5],
                    p[north_idx + 6],
                    p[north_idx + 7],
                ]);

                // South (y + 1)
                let south_idx = row_below + x;
                let south = f32x8::new([
                    p[south_idx],
                    p[south_idx + 1],
                    p[south_idx + 2],
                    p[south_idx + 3],
                    p[south_idx + 4],
                    p[south_idx + 5],
                    p[south_idx + 6],
                    p[south_idx + 7],
                ]);

                // West (x - 1)
                let west = f32x8::new([
                    p[idx - 1],
                    p[idx],
                    p[idx + 1],
                    p[idx + 2],
                    p[idx + 3],
                    p[idx + 4],
                    p[idx + 5],
                    p[idx + 6],
                ]);

                // East (x + 1)
                let east = f32x8::new([
                    p[idx + 1],
                    p[idx + 2],
                    p[idx + 3],
                    p[idx + 4],
                    p[idx + 5],
                    p[idx + 6],
                    p[idx + 7],
                    p[idx + 8],
                ]);

                // Laplacian = north + south + west + east - 4 * center
                let result = north + south + west + east - four * center;
                let arr: [f32; 8] = result.into();
                laplacian[idx..idx + 8].copy_from_slice(&arr);
            }

            // Handle remainder
            let tail_start = 1 + chunks * 8;
            for i in 0..remainder {
                let x = tail_start + i;
                let idx = row_start + x;
                laplacian[idx] =
                    p[row_above + x] + p[row_below + x] + p[idx - 1] + p[idx + 1] - 4.0 * p[idx];
            }
        }
    }

    /// 2D FDTD wave equation step.
    ///
    /// Computes: p_next[i,j] = 2*p[i,j] - p_prev[i,j] + c2 * laplacian(p)[i,j]
    ///
    /// This is a complete wave simulation timestep.
    #[inline]
    pub fn fdtd_step_2d_f32(p: &[f32], p_prev: &mut [f32], c2: f32, width: usize, height: usize) {
        let two = f32x8::splat(2.0);
        let four = f32x8::splat(4.0);
        let c2_vec = f32x8::splat(c2);

        for y in 1..height - 1 {
            let row_start = y * width;
            let row_above = (y - 1) * width;
            let row_below = (y + 1) * width;

            let inner_width = width - 2;
            let chunks = inner_width / 8;
            let remainder = inner_width % 8;

            for chunk in 0..chunks {
                let x = 1 + chunk * 8;
                let idx = row_start + x;

                let center = f32x8::new([
                    p[idx],
                    p[idx + 1],
                    p[idx + 2],
                    p[idx + 3],
                    p[idx + 4],
                    p[idx + 5],
                    p[idx + 6],
                    p[idx + 7],
                ]);

                let prev = f32x8::new([
                    p_prev[idx],
                    p_prev[idx + 1],
                    p_prev[idx + 2],
                    p_prev[idx + 3],
                    p_prev[idx + 4],
                    p_prev[idx + 5],
                    p_prev[idx + 6],
                    p_prev[idx + 7],
                ]);

                let north_idx = row_above + x;
                let north = f32x8::new([
                    p[north_idx],
                    p[north_idx + 1],
                    p[north_idx + 2],
                    p[north_idx + 3],
                    p[north_idx + 4],
                    p[north_idx + 5],
                    p[north_idx + 6],
                    p[north_idx + 7],
                ]);

                let south_idx = row_below + x;
                let south = f32x8::new([
                    p[south_idx],
                    p[south_idx + 1],
                    p[south_idx + 2],
                    p[south_idx + 3],
                    p[south_idx + 4],
                    p[south_idx + 5],
                    p[south_idx + 6],
                    p[south_idx + 7],
                ]);

                let west = f32x8::new([
                    p[idx - 1],
                    p[idx],
                    p[idx + 1],
                    p[idx + 2],
                    p[idx + 3],
                    p[idx + 4],
                    p[idx + 5],
                    p[idx + 6],
                ]);

                let east = f32x8::new([
                    p[idx + 1],
                    p[idx + 2],
                    p[idx + 3],
                    p[idx + 4],
                    p[idx + 5],
                    p[idx + 6],
                    p[idx + 7],
                    p[idx + 8],
                ]);

                let laplacian = north + south + west + east - four * center;
                let result = two * center - prev + c2_vec * laplacian;

                let arr: [f32; 8] = result.into();
                p_prev[idx..idx + 8].copy_from_slice(&arr);
            }

            let tail_start = 1 + chunks * 8;
            for i in 0..remainder {
                let x = tail_start + i;
                let idx = row_start + x;
                let laplacian =
                    p[row_above + x] + p[row_below + x] + p[idx - 1] + p[idx + 1] - 4.0 * p[idx];
                p_prev[idx] = 2.0 * p[idx] - p_prev[idx] + c2 * laplacian;
            }
        }
    }

    /// 3D Laplacian stencil (7-point).
    ///
    /// Computes the 3D discrete Laplacian for volumetric simulations.
    #[inline]
    pub fn laplacian_3d_f32(
        p: &[f32],
        laplacian: &mut [f32],
        width: usize,
        height: usize,
        depth: usize,
    ) {
        let stride_y = width;
        let stride_z = width * height;
        let six = f32x8::splat(6.0);

        for z in 1..depth - 1 {
            for y in 1..height - 1 {
                let row_start = z * stride_z + y * stride_y;
                let inner_width = width - 2;
                let chunks = inner_width / 8;
                let remainder = inner_width % 8;

                for chunk in 0..chunks {
                    let x = 1 + chunk * 8;
                    let idx = row_start + x;

                    let center = f32x8::new([
                        p[idx],
                        p[idx + 1],
                        p[idx + 2],
                        p[idx + 3],
                        p[idx + 4],
                        p[idx + 5],
                        p[idx + 6],
                        p[idx + 7],
                    ]);

                    // X neighbors
                    let west = f32x8::new([
                        p[idx - 1],
                        p[idx],
                        p[idx + 1],
                        p[idx + 2],
                        p[idx + 3],
                        p[idx + 4],
                        p[idx + 5],
                        p[idx + 6],
                    ]);
                    let east = f32x8::new([
                        p[idx + 1],
                        p[idx + 2],
                        p[idx + 3],
                        p[idx + 4],
                        p[idx + 5],
                        p[idx + 6],
                        p[idx + 7],
                        p[idx + 8],
                    ]);

                    // Y neighbors
                    let north_idx = idx - stride_y;
                    let south_idx = idx + stride_y;
                    let north = f32x8::new([
                        p[north_idx],
                        p[north_idx + 1],
                        p[north_idx + 2],
                        p[north_idx + 3],
                        p[north_idx + 4],
                        p[north_idx + 5],
                        p[north_idx + 6],
                        p[north_idx + 7],
                    ]);
                    let south = f32x8::new([
                        p[south_idx],
                        p[south_idx + 1],
                        p[south_idx + 2],
                        p[south_idx + 3],
                        p[south_idx + 4],
                        p[south_idx + 5],
                        p[south_idx + 6],
                        p[south_idx + 7],
                    ]);

                    // Z neighbors
                    let up_idx = idx - stride_z;
                    let down_idx = idx + stride_z;
                    let up = f32x8::new([
                        p[up_idx],
                        p[up_idx + 1],
                        p[up_idx + 2],
                        p[up_idx + 3],
                        p[up_idx + 4],
                        p[up_idx + 5],
                        p[up_idx + 6],
                        p[up_idx + 7],
                    ]);
                    let down = f32x8::new([
                        p[down_idx],
                        p[down_idx + 1],
                        p[down_idx + 2],
                        p[down_idx + 3],
                        p[down_idx + 4],
                        p[down_idx + 5],
                        p[down_idx + 6],
                        p[down_idx + 7],
                    ]);

                    let result = west + east + north + south + up + down - six * center;
                    let arr: [f32; 8] = result.into();
                    laplacian[idx..idx + 8].copy_from_slice(&arr);
                }

                let tail_start = 1 + chunks * 8;
                for i in 0..remainder {
                    let x = tail_start + i;
                    let idx = row_start + x;
                    laplacian[idx] = p[idx - 1]
                        + p[idx + 1]
                        + p[idx - stride_y]
                        + p[idx + stride_y]
                        + p[idx - stride_z]
                        + p[idx + stride_z]
                        - 6.0 * p[idx];
                }
            }
        }
    }
}

// ============================================================================
// PARALLEL OPERATIONS (SIMD + Rayon)
// ============================================================================

impl SimdOps {
    /// Parallel SAXPY using Rayon + SIMD.
    ///
    /// Best for large arrays (> 100K elements).
    pub fn par_saxpy(a: f32, x: &[f32], y: &mut [f32]) {
        const CHUNK_SIZE: usize = 4096;

        y.par_chunks_mut(CHUNK_SIZE)
            .zip(x.par_chunks(CHUNK_SIZE))
            .for_each(|(y_chunk, x_chunk)| {
                Self::saxpy(a, x_chunk, y_chunk);
            });
    }

    /// Parallel sum using Rayon + SIMD.
    pub fn par_sum_f32(x: &[f32]) -> f32 {
        const CHUNK_SIZE: usize = 4096;

        x.par_chunks(CHUNK_SIZE).map(Self::sum_f32).sum()
    }

    /// Parallel 2D FDTD step using Rayon + SIMD.
    ///
    /// Parallelizes over rows for better cache efficiency.
    pub fn par_fdtd_step_2d_f32(
        p: &[f32],
        p_prev: &mut [f32],
        c2: f32,
        width: usize,
        height: usize,
    ) {
        // Each row can be processed independently
        p_prev
            .par_chunks_mut(width)
            .enumerate()
            .skip(1)
            .take(height - 2)
            .for_each(|(y, row)| {
                let row_above = (y - 1) * width;
                let row_below = (y + 1) * width;
                let row_start = y * width;

                let two = f32x8::splat(2.0);
                let four = f32x8::splat(4.0);
                let c2_vec = f32x8::splat(c2);

                let inner_width = width - 2;
                let chunks = inner_width / 8;
                let remainder = inner_width % 8;

                for chunk in 0..chunks {
                    let x = 1 + chunk * 8;
                    let idx = row_start + x;
                    let local_x = x;

                    let center = f32x8::new([
                        p[idx],
                        p[idx + 1],
                        p[idx + 2],
                        p[idx + 3],
                        p[idx + 4],
                        p[idx + 5],
                        p[idx + 6],
                        p[idx + 7],
                    ]);

                    let prev = f32x8::new([
                        row[local_x],
                        row[local_x + 1],
                        row[local_x + 2],
                        row[local_x + 3],
                        row[local_x + 4],
                        row[local_x + 5],
                        row[local_x + 6],
                        row[local_x + 7],
                    ]);

                    let north_idx = row_above + x;
                    let north = f32x8::new([
                        p[north_idx],
                        p[north_idx + 1],
                        p[north_idx + 2],
                        p[north_idx + 3],
                        p[north_idx + 4],
                        p[north_idx + 5],
                        p[north_idx + 6],
                        p[north_idx + 7],
                    ]);

                    let south_idx = row_below + x;
                    let south = f32x8::new([
                        p[south_idx],
                        p[south_idx + 1],
                        p[south_idx + 2],
                        p[south_idx + 3],
                        p[south_idx + 4],
                        p[south_idx + 5],
                        p[south_idx + 6],
                        p[south_idx + 7],
                    ]);

                    let west = f32x8::new([
                        p[idx - 1],
                        p[idx],
                        p[idx + 1],
                        p[idx + 2],
                        p[idx + 3],
                        p[idx + 4],
                        p[idx + 5],
                        p[idx + 6],
                    ]);

                    let east = f32x8::new([
                        p[idx + 1],
                        p[idx + 2],
                        p[idx + 3],
                        p[idx + 4],
                        p[idx + 5],
                        p[idx + 6],
                        p[idx + 7],
                        p[idx + 8],
                    ]);

                    let laplacian = north + south + west + east - four * center;
                    let result = two * center - prev + c2_vec * laplacian;

                    let arr: [f32; 8] = result.into();
                    row[local_x..local_x + 8].copy_from_slice(&arr);
                }

                let tail_start = 1 + chunks * 8;
                for i in 0..remainder {
                    let x = tail_start + i;
                    let idx = row_start + x;
                    let laplacian = p[row_above + x] + p[row_below + x] + p[idx - 1] + p[idx + 1]
                        - 4.0 * p[idx];
                    row[x] = 2.0 * p[idx] - row[x] + c2 * laplacian;
                }
            });
    }
}

// ============================================================================
// INTEGER OPERATIONS
// ============================================================================

impl SimdOps {
    /// Sum of i32 array using SIMD.
    #[inline]
    pub fn sum_i32(x: &[i32]) -> i64 {
        let n = x.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut acc = i32x8::splat(0);

        for i in 0..chunks {
            let offset = i * 8;
            let x_vec = i32x8::new([
                x[offset],
                x[offset + 1],
                x[offset + 2],
                x[offset + 3],
                x[offset + 4],
                x[offset + 5],
                x[offset + 6],
                x[offset + 7],
            ]);
            acc += x_vec;
        }

        let arr: [i32; 8] = acc.into();
        let mut sum: i64 = arr.iter().map(|&v| v as i64).sum();

        let tail_start = chunks * 8;
        for i in 0..remainder {
            sum += x[tail_start + i] as i64;
        }

        sum
    }

    /// Element-wise i32 addition.
    #[inline]
    pub fn add_i32(x: &[i32], y: &[i32], z: &mut [i32]) {
        let n = x.len().min(y.len()).min(z.len());
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let offset = i * 8;
            let x_vec = i32x8::new([
                x[offset],
                x[offset + 1],
                x[offset + 2],
                x[offset + 3],
                x[offset + 4],
                x[offset + 5],
                x[offset + 6],
                x[offset + 7],
            ]);
            let y_vec = i32x8::new([
                y[offset],
                y[offset + 1],
                y[offset + 2],
                y[offset + 3],
                y[offset + 4],
                y[offset + 5],
                y[offset + 6],
                y[offset + 7],
            ]);

            let result = x_vec + y_vec;
            let arr: [i32; 8] = result.into();
            z[offset..offset + 8].copy_from_slice(&arr);
        }

        let tail_start = chunks * 8;
        for i in 0..remainder {
            z[tail_start + i] = x[tail_start + i] + y[tail_start + i];
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saxpy() {
        let x = vec![1.0f32; 100];
        let mut y = vec![2.0f32; 100];

        SimdOps::saxpy(3.0, &x, &mut y);

        for v in y.iter() {
            assert!((v - 5.0).abs() < 1e-6, "Expected 5.0, got {}", v);
        }
    }

    #[test]
    fn test_saxpy_unaligned() {
        let x = vec![1.0f32; 13]; // Not divisible by 8
        let mut y = vec![2.0f32; 13];

        SimdOps::saxpy(2.0, &x, &mut y);

        for v in y.iter() {
            assert!((v - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_daxpy() {
        let x = vec![1.0f64; 100];
        let mut y = vec![2.0f64; 100];

        SimdOps::daxpy(3.0, &x, &mut y);

        for v in y.iter() {
            assert!((v - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dot_product() {
        let x = vec![1.0f32; 100];
        let y = vec![2.0f32; 100];

        let dot = SimdOps::dot_f32(&x, &y);
        assert!((dot - 200.0).abs() < 1e-4);
    }

    #[test]
    fn test_sum() {
        let x = vec![1.0f32; 1000];
        let sum = SimdOps::sum_f32(&x);
        assert!((sum - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn test_max_min() {
        let x = vec![1.0f32, -5.0, 3.0, 7.0, -2.0, 4.0, 6.0, 8.0, -1.0];

        let max = SimdOps::max_f32(&x);
        let min = SimdOps::min_f32(&x);

        assert!((max - 8.0).abs() < 1e-6);
        assert!((min - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn test_laplacian_2d() {
        // 5x5 grid
        let width = 5;
        let height = 5;
        let mut p = vec![0.0f32; width * height];

        // Set center to 1.0
        p[12] = 1.0; // (2, 2)

        let mut laplacian = vec![0.0f32; width * height];
        SimdOps::laplacian_2d_f32(&p, &mut laplacian, width, height);

        // Center should have laplacian of -4
        assert!((laplacian[12] - (-4.0)).abs() < 1e-6);

        // Neighbors should have laplacian of 1
        assert!((laplacian[11] - 1.0).abs() < 1e-6); // (1, 2)
        assert!((laplacian[13] - 1.0).abs() < 1e-6); // (3, 2)
        assert!((laplacian[7] - 1.0).abs() < 1e-6); // (2, 1)
        assert!((laplacian[17] - 1.0).abs() < 1e-6); // (2, 3)
    }

    #[test]
    fn test_fdtd_step_2d() {
        let width = 10;
        let height = 10;
        let mut p = vec![0.0f32; width * height];
        let mut p_prev = vec![0.0f32; width * height];

        // Initial impulse at center
        p[55] = 1.0; // (5, 5)

        let c2 = 0.1;
        SimdOps::fdtd_step_2d_f32(&p, &mut p_prev, c2, width, height);

        // After one step, energy should spread from center
        // Center should now be: 2*1 - 0 + 0.1*(-4) = 1.6
        assert!((p_prev[55] - 1.6).abs() < 1e-6);
    }

    #[test]
    fn test_par_saxpy() {
        let x = vec![1.0f32; 10000];
        let mut y = vec![2.0f32; 10000];

        SimdOps::par_saxpy(3.0, &x, &mut y);

        for v in y.iter() {
            assert!((v - 5.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_par_sum() {
        let x = vec![1.0f32; 100000];
        let sum = SimdOps::par_sum_f32(&x);
        assert!((sum - 100000.0).abs() < 1.0); // Allow small floating point error
    }

    #[test]
    fn test_sum_i32() {
        let x = vec![1i32; 1000];
        let sum = SimdOps::sum_i32(&x);
        assert_eq!(sum, 1000);
    }

    #[test]
    fn test_add_vectors() {
        let x = vec![1.0f32; 100];
        let y = vec![2.0f32; 100];
        let mut z = vec![0.0f32; 100];

        SimdOps::add_f32(&x, &y, &mut z);

        for v in z.iter() {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_scale() {
        let mut x = vec![2.0f32; 100];
        SimdOps::scale_f32(3.0, &mut x);

        for v in x.iter() {
            assert!((v - 6.0).abs() < 1e-6);
        }
    }
}
