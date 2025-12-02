//! SIMD-accelerated FDTD kernel for wave simulation.
//!
//! Uses `std::simd` portable SIMD (requires nightly Rust).
//! Processes 8 cells at a time using f32x8 vectors for AVX2.
//!
//! The main SIMD implementation is now inlined in grid.rs
//! (fdtd_row_simd_inline) for better integration with rayon parallel processing.

#![cfg(feature = "simd")]

#[cfg(test)]
mod tests {
    use std::simd::f32x8;

    /// Test helper: SIMD implementation of FDTD for a single row.
    fn fdtd_row_simd_test(
        pressure: &[f32],
        pressure_prev: &mut [f32],
        width: usize,
        y: usize,
        c2: f32,
        damping: f32,
    ) {
        let row_start = y * width;
        let c2_vec = f32x8::splat(c2);
        let damping_vec = f32x8::splat(damping);
        let four = f32x8::splat(4.0);
        let two = f32x8::splat(2.0);

        let mut x = 1;
        while x + 8 <= width - 1 {
            let idx = row_start + x;

            let p_curr = f32x8::from_slice(&pressure[idx..idx + 8]);
            let p_prev = f32x8::from_slice(&pressure_prev[idx..idx + 8]);
            let p_north = f32x8::from_slice(&pressure[idx - width..idx - width + 8]);
            let p_south = f32x8::from_slice(&pressure[idx + width..idx + width + 8]);
            let p_west = f32x8::from_slice(&pressure[idx - 1..idx - 1 + 8]);
            let p_east = f32x8::from_slice(&pressure[idx + 1..idx + 1 + 8]);

            let laplacian = p_north + p_south + p_east + p_west - four * p_curr;
            let p_new = two * p_curr - p_prev + c2_vec * laplacian;
            let p_damped = p_new * damping_vec;

            pressure_prev[idx..idx + 8].copy_from_slice(&p_damped.to_array());
            x += 8;
        }

        // Scalar fallback
        while x < width - 1 {
            let idx = row_start + x;

            let p_curr = pressure[idx];
            let p_prev = pressure_prev[idx];
            let p_north = pressure[idx - width];
            let p_south = pressure[idx + width];
            let p_west = pressure[idx - 1];
            let p_east = pressure[idx + 1];

            let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
            let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;

            pressure_prev[idx] = p_new * damping;
            x += 1;
        }
    }

    #[test]
    fn test_simd_row_basic() {
        let width = 16;
        let height = 4;
        let size = width * height;

        let mut pressure = vec![0.0f32; size];
        let mut pressure_prev = vec![0.0f32; size];

        // Set up a pattern with neighbors at non-zero values
        pressure[width + 8] = 0.5; // Center cell
        pressure[width + 7] = 0.3; // West
        pressure[width + 9] = 0.3; // East
        pressure[8] = 0.2;         // North (row 0)
        pressure[2 * width + 8] = 0.2; // South (row 2)

        let c2 = 0.5f32;
        let damping = 0.99f32;

        fdtd_row_simd_test(&pressure, &mut pressure_prev, width, 1, c2, damping);

        assert!(pressure_prev[width + 8].is_finite(), "Result should be finite");

        // Manual calculation:
        // laplacian = 0.2 + 0.2 + 0.3 + 0.3 - 4*0.5 = 1.0 - 2.0 = -1.0
        // p_new = 2*0.5 - 0.0 + 0.5*(-1.0) = 1.0 - 0.5 = 0.5
        // p_damped = 0.5 * 0.99 = 0.495
        let expected = 0.495f32;
        let diff = (pressure_prev[width + 8] - expected).abs();
        assert!(diff < 1e-5, "Expected ~{}, got {}", expected, pressure_prev[width + 8]);
    }

    #[test]
    fn test_simd_matches_scalar() {
        let width = 32;
        let height = 8;
        let size = width * height;

        let pressure: Vec<f32> = (0..size)
            .map(|i| ((i * 17 + 3) % 100) as f32 / 100.0)
            .collect();
        let mut pressure_prev_simd = pressure.clone();
        let mut pressure_prev_scalar = pressure.clone();

        let c2 = 0.444f32;
        let damping = 0.999f32;

        // SIMD version
        fdtd_row_simd_test(&pressure, &mut pressure_prev_simd, width, 4, c2, damping);

        // Scalar version
        for x in 1..width - 1 {
            let idx = 4 * width + x;
            let p_curr = pressure[idx];
            let p_prev = pressure_prev_scalar[idx];
            let p_north = pressure[idx - width];
            let p_south = pressure[idx + width];
            let p_west = pressure[idx - 1];
            let p_east = pressure[idx + 1];

            let laplacian = p_north + p_south + p_east + p_west - 4.0 * p_curr;
            let p_new = 2.0 * p_curr - p_prev + c2 * laplacian;
            pressure_prev_scalar[idx] = p_new * damping;
        }

        // Compare results
        for x in 1..width - 1 {
            let idx = 4 * width + x;
            let diff = (pressure_prev_simd[idx] - pressure_prev_scalar[idx]).abs();
            assert!(
                diff < 1e-5,
                "Mismatch at x={}: simd={}, scalar={}",
                x,
                pressure_prev_simd[idx],
                pressure_prev_scalar[idx]
            );
        }
    }
}
