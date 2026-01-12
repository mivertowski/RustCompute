//! Sparse Matrix-Vector Multiplication (SpMV).
//!
//! Computes y = A * x where A is a sparse matrix in CSR format.
//! This is a fundamental operation for iterative graph algorithms like PageRank.

use crate::models::CsrMatrix;
use crate::{GraphError, Result};

/// SpMV configuration.
#[derive(Debug, Clone)]
pub struct SpmvConfig {
    /// Use parallel implementation.
    pub parallel: bool,
    /// Alpha scaling factor (y = alpha * A * x).
    pub alpha: f64,
}

impl Default for SpmvConfig {
    fn default() -> Self {
        Self {
            parallel: false,
            alpha: 1.0,
        }
    }
}

impl SpmvConfig {
    /// Create new SpMV configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable parallel execution.
    pub fn parallel(mut self) -> Self {
        self.parallel = true;
        self
    }

    /// Set scaling factor.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
}

/// Sequential SpMV: y = A * x.
///
/// # Arguments
///
/// * `matrix` - Sparse matrix in CSR format
/// * `x` - Input vector
///
/// # Returns
///
/// Output vector y = A * x
pub fn spmv(matrix: &CsrMatrix, x: &[f64]) -> Result<Vec<f64>> {
    spmv_with_config(matrix, x, &SpmvConfig::default())
}

/// SpMV with configuration.
pub fn spmv_with_config(matrix: &CsrMatrix, x: &[f64], config: &SpmvConfig) -> Result<Vec<f64>> {
    if x.len() != matrix.num_cols {
        return Err(GraphError::DimensionMismatch {
            expected: matrix.num_cols,
            actual: x.len(),
        });
    }

    if config.parallel {
        spmv_parallel_impl(matrix, x, config.alpha)
    } else {
        spmv_sequential_impl(matrix, x, config.alpha)
    }
}

/// Sequential implementation.
fn spmv_sequential_impl(matrix: &CsrMatrix, x: &[f64], alpha: f64) -> Result<Vec<f64>> {
    let mut y = vec![0.0; matrix.num_rows];

    for (row, y_row) in y.iter_mut().enumerate() {
        let start = matrix.row_ptr[row] as usize;
        let end = matrix.row_ptr[row + 1] as usize;

        let mut sum = 0.0;
        for i in start..end {
            let col = matrix.col_idx[i] as usize;
            let val = matrix.values.as_ref().map(|v| v[i]).unwrap_or(1.0);
            sum += val * x[col];
        }

        *y_row = alpha * sum;
    }

    Ok(y)
}

/// Parallel SpMV using row-based parallelism.
///
/// Each row is processed independently, making this suitable for GPU execution.
pub fn spmv_parallel(matrix: &CsrMatrix, x: &[f64]) -> Result<Vec<f64>> {
    spmv_with_config(matrix, x, &SpmvConfig::default().parallel())
}

/// Parallel implementation (currently sequential, ready for GPU/rayon).
fn spmv_parallel_impl(matrix: &CsrMatrix, x: &[f64], alpha: f64) -> Result<Vec<f64>> {
    // Each row can be computed independently (parallelizable)
    let y: Vec<f64> = (0..matrix.num_rows)
        .map(|row| {
            let start = matrix.row_ptr[row] as usize;
            let end = matrix.row_ptr[row + 1] as usize;

            let mut sum = 0.0;
            for i in start..end {
                let col = matrix.col_idx[i] as usize;
                let val = matrix.values.as_ref().map(|v| v[i]).unwrap_or(1.0);
                sum += val * x[col];
            }

            alpha * sum
        })
        .collect();

    Ok(y)
}

/// SpMV with y = alpha * A * x + beta * y (BLAS-style).
///
/// This allows accumulating results without allocating new vectors.
pub fn spmv_axpby(
    matrix: &CsrMatrix,
    x: &[f64],
    y: &mut [f64],
    alpha: f64,
    beta: f64,
) -> Result<()> {
    if x.len() != matrix.num_cols {
        return Err(GraphError::DimensionMismatch {
            expected: matrix.num_cols,
            actual: x.len(),
        });
    }

    if y.len() != matrix.num_rows {
        return Err(GraphError::DimensionMismatch {
            expected: matrix.num_rows,
            actual: y.len(),
        });
    }

    for (row, y_row) in y.iter_mut().enumerate() {
        let start = matrix.row_ptr[row] as usize;
        let end = matrix.row_ptr[row + 1] as usize;

        let mut sum = 0.0;
        for i in start..end {
            let col = matrix.col_idx[i] as usize;
            let val = matrix.values.as_ref().map(|v| v[i]).unwrap_or(1.0);
            sum += val * x[col];
        }

        *y_row = alpha * sum + beta * *y_row;
    }

    Ok(())
}

/// Compute dot product of two vectors.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

/// Compute L2 norm of a vector.
pub fn norm2(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

/// Normalize vector in-place.
pub fn normalize(x: &mut [f64]) {
    let n = norm2(x);
    if n > 1e-10 {
        for xi in x.iter_mut() {
            *xi /= n;
        }
    }
}

/// Power iteration for dominant eigenvector.
///
/// Computes the eigenvector corresponding to the largest eigenvalue
/// of the matrix using iterative multiplication.
pub fn power_iteration(
    matrix: &CsrMatrix,
    max_iterations: usize,
    tolerance: f64,
) -> Result<(Vec<f64>, f64)> {
    if matrix.num_rows == 0 {
        return Err(GraphError::EmptyGraph);
    }

    // Initialize with uniform vector
    let n = matrix.num_rows;
    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    let mut eigenvalue = 0.0;

    for _ in 0..max_iterations {
        // y = A * x
        let y = spmv(matrix, &x)?;

        // Compute Rayleigh quotient (eigenvalue estimate)
        let new_eigenvalue = dot(&x, &y);

        // Normalize
        let norm = norm2(&y);
        if norm < 1e-10 {
            break;
        }

        x = y.into_iter().map(|yi| yi / norm).collect();

        // Check convergence
        if (new_eigenvalue - eigenvalue).abs() < tolerance {
            return Ok((x, new_eigenvalue));
        }

        eigenvalue = new_eigenvalue;
    }

    Ok((x, eigenvalue))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spmv_identity() {
        // Identity matrix (diagonal with 1s)
        let edges = [(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
        let matrix = CsrMatrix::from_weighted_edges(3, &edges);

        let x = vec![1.0, 2.0, 3.0];
        let y = spmv(&matrix, &x).unwrap();

        // Identity matrix: y = I * x = x
        assert!((y[0] - 1.0).abs() < 1e-10);
        assert!((y[1] - 2.0).abs() < 1e-10);
        assert!((y[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_spmv_unweighted() {
        // Adjacency matrix: 0 -> 1, 0 -> 2, 1 -> 2
        let edges = [(0, 1), (0, 2), (1, 2)];
        let matrix = CsrMatrix::from_edges(3, &edges);

        // x = [1, 1, 1]
        // y[0] = 0 (no outgoing edges counted in row 0? Actually row 0 has edges to 1 and 2)
        // Wait, CSR row i contains edges FROM i, so:
        // y[0] = x[1] + x[2] = 2
        // y[1] = x[2] = 1
        // y[2] = 0

        let x = vec![1.0, 1.0, 1.0];
        let y = spmv(&matrix, &x).unwrap();

        assert!((y[0] - 2.0).abs() < 1e-10);
        assert!((y[1] - 1.0).abs() < 1e-10);
        assert!((y[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_spmv_weighted() {
        let edges = [(0, 1, 2.0), (0, 2, 3.0), (1, 2, 4.0)];
        let matrix = CsrMatrix::from_weighted_edges(3, &edges);

        let x = vec![1.0, 1.0, 1.0];
        let y = spmv(&matrix, &x).unwrap();

        // y[0] = 2.0 * 1 + 3.0 * 1 = 5.0
        // y[1] = 4.0 * 1 = 4.0
        // y[2] = 0

        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 4.0).abs() < 1e-10);
        assert!((y[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_spmv_alpha() {
        let edges = [(0, 1, 1.0)];
        let matrix = CsrMatrix::from_weighted_edges(2, &edges);

        let x = vec![1.0, 2.0];
        let config = SpmvConfig::new().with_alpha(0.5);
        let y = spmv_with_config(&matrix, &x, &config).unwrap();

        // y[0] = 0.5 * (1.0 * 2.0) = 1.0
        assert!((y[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spmv_dimension_mismatch() {
        let matrix = CsrMatrix::from_edges(3, &[(0, 1)]);
        let x = vec![1.0, 2.0]; // Wrong size

        let result = spmv(&matrix, &x);
        assert!(matches!(result, Err(GraphError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_spmv_parallel_same_as_sequential() {
        let edges = [(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0), (2, 0, 4.0)];
        let matrix = CsrMatrix::from_weighted_edges(3, &edges);

        let x = vec![1.0, 2.0, 3.0];

        let seq = spmv(&matrix, &x).unwrap();
        let par = spmv_parallel(&matrix, &x).unwrap();

        for i in 0..3 {
            assert!((seq[i] - par[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_spmv_axpby() {
        let edges = [(0, 1, 1.0)];
        let matrix = CsrMatrix::from_weighted_edges(2, &edges);

        let x = vec![1.0, 2.0];
        let mut y = vec![1.0, 1.0];

        // y = 2 * A * x + 3 * y
        spmv_axpby(&matrix, &x, &mut y, 2.0, 3.0).unwrap();

        // y[0] = 2 * (1 * 2) + 3 * 1 = 7
        // y[1] = 2 * 0 + 3 * 1 = 3
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        assert!((dot(&x, &y) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm2() {
        let x = vec![3.0, 4.0];
        assert!((norm2(&x) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let mut x = vec![3.0, 4.0];
        normalize(&mut x);

        assert!((norm2(&x) - 1.0).abs() < 1e-10);
        assert!((x[0] - 0.6).abs() < 1e-10);
        assert!((x[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_power_iteration() {
        // Simple 2x2 matrix with known eigenvector
        // A = [[2, 1], [1, 2]]
        // Eigenvalues: 3, 1
        // Eigenvector for 3: [1/sqrt(2), 1/sqrt(2)]

        let mut builder = crate::models::CsrMatrixBuilder::new(2);
        builder.add_weighted_edge(0, 0, 2.0);
        builder.add_weighted_edge(0, 1, 1.0);
        builder.add_weighted_edge(1, 0, 1.0);
        builder.add_weighted_edge(1, 1, 2.0);
        let matrix = builder.build();

        let (eigenvector, eigenvalue) = power_iteration(&matrix, 100, 1e-6).unwrap();

        // Eigenvalue should be close to 3
        assert!((eigenvalue - 3.0).abs() < 0.01);

        // Eigenvector should be [1/sqrt(2), 1/sqrt(2)] or [-1/sqrt(2), -1/sqrt(2)]
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((eigenvector[0].abs() - expected).abs() < 0.01);
        assert!((eigenvector[1].abs() - expected).abs() < 0.01);
    }
}
