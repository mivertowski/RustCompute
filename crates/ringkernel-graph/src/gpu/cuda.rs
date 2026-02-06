//! CUDA implementation for graph algorithms.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use super::{BFS_KERNEL_SOURCE, SPMV_KERNEL_SOURCE};
use crate::models::CsrMatrix;

/// Error type for GPU graph operations.
#[derive(Debug, thiserror::Error)]
pub enum GpuGraphError {
    /// CUDA driver error.
    #[error("CUDA error: {0}")]
    CudaError(String),
    /// Compilation error.
    #[error("Compilation error: {0}")]
    CompilationError(String),
    /// Invalid input.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

type Result<T> = std::result::Result<T, GpuGraphError>;

/// GPU-accelerated BFS (Breadth-First Search).
pub struct GpuBfs {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    init_fn: CudaFunction,
    advance_fn: CudaFunction,
    copy_fn: CudaFunction,
}

impl GpuBfs {
    /// Create a new GPU BFS instance.
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(device_ordinal)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        let stream = context.default_stream();

        // Compile BFS kernels
        let ptx = compile_ptx(BFS_KERNEL_SOURCE)
            .map_err(|e| GpuGraphError::CompilationError(e.to_string()))?;

        let module = context
            .load_module(ptx)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        let init_fn = module
            .load_function("bfs_init")
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        let advance_fn = module
            .load_function("bfs_advance")
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        let copy_fn = module
            .load_function("bfs_copy")
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        Ok(Self {
            context,
            stream,
            init_fn,
            advance_fn,
            copy_fn,
        })
    }

    /// Run BFS from given source nodes.
    ///
    /// Returns distance from nearest source for each node (-1 if unreachable).
    pub fn bfs(&self, matrix: &CsrMatrix, sources: &[u32]) -> Result<Vec<i32>> {
        if sources.is_empty() {
            return Err(GpuGraphError::InvalidInput("No source nodes".to_string()));
        }

        let num_nodes = matrix.num_rows as u32;

        // Copy CSR to device
        let row_ptr_dev = self.htod_u64(&matrix.row_ptr)?;
        let col_idx_dev = self.htod_u32(&matrix.col_idx)?;
        let sources_dev = self.htod_u32(sources)?;

        // Allocate distance arrays
        let mut distances = self.alloc_i32(num_nodes as usize)?;
        let mut new_distances = self.alloc_i32(num_nodes as usize)?;

        // Frontier size (single int)
        let mut frontier_size_dev = self.alloc_i32(1)?;

        let block_size = 256u32;
        let grid_size = num_nodes.div_ceil(block_size);

        // Initialize distances
        let num_sources = sources.len() as u32;
        // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
        // are valid and allocated with sufficient size.
        unsafe {
            self.stream
                .launch_builder(&self.init_fn)
                .arg(&mut distances)
                .arg(&sources_dev)
                .arg(&num_nodes)
                .arg(&num_sources)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;
        }

        // Copy distances to new_distances
        // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
        // are valid and allocated with sufficient size.
        unsafe {
            self.stream
                .launch_builder(&self.copy_fn)
                .arg(&distances)
                .arg(&mut new_distances)
                .arg(&num_nodes)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;
        }

        // BFS iterations
        let max_iterations = num_nodes; // At most n levels
        for level in 0..max_iterations as i32 {
            // Reset frontier size
            let zero = vec![0i32];
            self.stream
                .memcpy_htod(&zero, &mut frontier_size_dev)
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

            // Advance frontier
            // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
            // are valid and allocated with sufficient size.
            unsafe {
                self.stream
                    .launch_builder(&self.advance_fn)
                    .arg(&row_ptr_dev)
                    .arg(&col_idx_dev)
                    .arg(&distances)
                    .arg(&mut new_distances)
                    .arg(&mut frontier_size_dev)
                    .arg(&num_nodes)
                    .arg(&level)
                    .launch(cudarc::driver::LaunchConfig {
                        grid_dim: (grid_size, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;
            }

            // Check if frontier is empty
            let mut frontier_size = vec![0i32];
            self.stream
                .memcpy_dtoh(&frontier_size_dev, &mut frontier_size)
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

            if frontier_size[0] == 0 {
                break;
            }

            // Copy new_distances to distances
            // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
            // are valid and allocated with sufficient size.
            unsafe {
                self.stream
                    .launch_builder(&self.copy_fn)
                    .arg(&new_distances)
                    .arg(&mut distances)
                    .arg(&num_nodes)
                    .launch(cudarc::driver::LaunchConfig {
                        grid_dim: (grid_size, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;
            }
        }

        // Copy final distances back
        // Use new_distances which has the latest values
        let mut result = vec![0i32; num_nodes as usize];
        self.stream
            .memcpy_dtoh(&new_distances, &mut result)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        Ok(result)
    }

    /// Synchronize.
    pub fn synchronize(&self) -> Result<()> {
        self.context
            .synchronize()
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))
    }

    fn alloc_i32(&self, n: usize) -> Result<CudaSlice<i32>> {
        // SAFETY: cudarc's alloc returns properly aligned device memory. The size
        // is computed from the input data.
        unsafe {
            self.stream
                .alloc::<i32>(n)
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))
        }
    }

    fn htod_u64(&self, data: &[u64]) -> Result<CudaSlice<u64>> {
        // SAFETY: cudarc's alloc returns properly aligned device memory. The size
        // is computed from the input data.
        let mut slice = unsafe {
            self.stream
                .alloc::<u64>(data.len())
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))?
        };
        self.stream
            .memcpy_htod(data, &mut slice)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;
        Ok(slice)
    }

    fn htod_u32(&self, data: &[u32]) -> Result<CudaSlice<u32>> {
        // SAFETY: cudarc's alloc returns properly aligned device memory. The size
        // is computed from the input data.
        let mut slice = unsafe {
            self.stream
                .alloc::<u32>(data.len())
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))?
        };
        self.stream
            .memcpy_htod(data, &mut slice)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;
        Ok(slice)
    }
}

/// GPU-accelerated SpMV (Sparse Matrix-Vector Multiplication).
pub struct GpuSpmv {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    spmv_fn: CudaFunction,
    spmv_axpby_fn: CudaFunction,
    dot_fn: CudaFunction,
    #[allow(dead_code)]
    scale_fn: CudaFunction,
    #[allow(dead_code)]
    copy_fn: CudaFunction,
}

impl GpuSpmv {
    /// Create a new GPU SpMV instance.
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(device_ordinal)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        let stream = context.default_stream();

        // Compile SpMV kernels
        let ptx = compile_ptx(SPMV_KERNEL_SOURCE)
            .map_err(|e| GpuGraphError::CompilationError(e.to_string()))?;

        let module = context
            .load_module(ptx)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        let spmv_fn = module
            .load_function("spmv_csr")
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        let spmv_axpby_fn = module
            .load_function("spmv_csr_axpby")
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        let dot_fn = module
            .load_function("dot_product")
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        let scale_fn = module
            .load_function("vector_scale")
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        let copy_fn = module
            .load_function("vector_copy")
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        Ok(Self {
            context,
            stream,
            spmv_fn,
            spmv_axpby_fn,
            dot_fn,
            scale_fn,
            copy_fn,
        })
    }

    /// Compute y = alpha * A * x.
    pub fn spmv(&self, matrix: &CsrMatrix, x: &[f64], alpha: f64) -> Result<Vec<f64>> {
        if x.len() != matrix.num_cols {
            return Err(GpuGraphError::InvalidInput(format!(
                "Vector size {} doesn't match matrix columns {}",
                x.len(),
                matrix.num_cols
            )));
        }

        let num_rows = matrix.num_rows as u32;

        // Copy CSR to device
        let row_ptr_dev = self.htod_u64(&matrix.row_ptr)?;
        let col_idx_dev = self.htod_u32(&matrix.col_idx)?;
        let x_dev = self.htod_f64(x)?;

        // Handle values (may be None for unweighted)
        let (values_dev, has_values) = if let Some(ref values) = matrix.values {
            (self.htod_f64(values)?, 1i32)
        } else {
            // Allocate dummy buffer
            // SAFETY: cudarc's alloc returns properly aligned device memory. The size
            // is computed from the input data.
            let dummy = unsafe {
                self.stream
                    .alloc::<f64>(1)
                    .map_err(|e| GpuGraphError::CudaError(e.to_string()))?
            };
            (dummy, 0i32)
        };

        // Allocate output
        let mut y_dev = self.alloc_f64(matrix.num_rows)?;

        let block_size = 256u32;
        let grid_size = num_rows.div_ceil(block_size);

        // Launch SpMV
        // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
        // are valid and allocated with sufficient size.
        unsafe {
            self.stream
                .launch_builder(&self.spmv_fn)
                .arg(&row_ptr_dev)
                .arg(&col_idx_dev)
                .arg(&values_dev)
                .arg(&x_dev)
                .arg(&mut y_dev)
                .arg(&num_rows)
                .arg(&alpha)
                .arg(&has_values)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;
        }

        // Copy result back
        let mut y = vec![0.0f64; matrix.num_rows];
        self.stream
            .memcpy_dtoh(&y_dev, &mut y)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        Ok(y)
    }

    /// Compute y = alpha * A * x + beta * y.
    pub fn spmv_axpby(
        &self,
        matrix: &CsrMatrix,
        x: &[f64],
        y: &mut [f64],
        alpha: f64,
        beta: f64,
    ) -> Result<()> {
        if x.len() != matrix.num_cols {
            return Err(GpuGraphError::InvalidInput(format!(
                "Input vector size {} doesn't match matrix columns {}",
                x.len(),
                matrix.num_cols
            )));
        }
        if y.len() != matrix.num_rows {
            return Err(GpuGraphError::InvalidInput(format!(
                "Output vector size {} doesn't match matrix rows {}",
                y.len(),
                matrix.num_rows
            )));
        }

        let num_rows = matrix.num_rows as u32;

        // Copy data to device
        let row_ptr_dev = self.htod_u64(&matrix.row_ptr)?;
        let col_idx_dev = self.htod_u32(&matrix.col_idx)?;
        let x_dev = self.htod_f64(x)?;
        let mut y_dev = self.htod_f64(y)?;

        let (values_dev, has_values) = if let Some(ref values) = matrix.values {
            (self.htod_f64(values)?, 1i32)
        } else {
            // SAFETY: cudarc's alloc returns properly aligned device memory. The size
            // is computed from the input data.
            let dummy = unsafe {
                self.stream
                    .alloc::<f64>(1)
                    .map_err(|e| GpuGraphError::CudaError(e.to_string()))?
            };
            (dummy, 0i32)
        };

        let block_size = 256u32;
        let grid_size = num_rows.div_ceil(block_size);

        // Launch SpMV with accumulation
        // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
        // are valid and allocated with sufficient size.
        unsafe {
            self.stream
                .launch_builder(&self.spmv_axpby_fn)
                .arg(&row_ptr_dev)
                .arg(&col_idx_dev)
                .arg(&values_dev)
                .arg(&x_dev)
                .arg(&mut y_dev)
                .arg(&num_rows)
                .arg(&alpha)
                .arg(&beta)
                .arg(&has_values)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;
        }

        // Copy result back
        self.stream
            .memcpy_dtoh(&y_dev, y)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        Ok(())
    }

    /// Compute dot product x · y.
    pub fn dot(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(GpuGraphError::InvalidInput(
                "Vector sizes don't match".to_string(),
            ));
        }

        let n = x.len() as u32;
        let block_size = 256u32;
        let num_blocks = n.div_ceil(block_size);

        let x_dev = self.htod_f64(x)?;
        let y_dev = self.htod_f64(y)?;
        let mut partial_sums = self.alloc_f64(num_blocks as usize)?;

        // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
        // are valid and allocated with sufficient size.
        unsafe {
            self.stream
                .launch_builder(&self.dot_fn)
                .arg(&x_dev)
                .arg(&y_dev)
                .arg(&mut partial_sums)
                .arg(&n)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;
        }

        // Sum partial results on host
        let mut sums = vec![0.0f64; num_blocks as usize];
        self.stream
            .memcpy_dtoh(&partial_sums, &mut sums)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;

        Ok(sums.iter().sum())
    }

    /// Compute L2 norm of vector.
    pub fn norm2(&self, x: &[f64]) -> Result<f64> {
        let dot_xx = self.dot(x, x)?;
        Ok(dot_xx.sqrt())
    }

    /// Power iteration for dominant eigenvector.
    ///
    /// Returns (eigenvector, eigenvalue).
    pub fn power_iteration(
        &self,
        matrix: &CsrMatrix,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<(Vec<f64>, f64)> {
        if matrix.num_rows == 0 {
            return Err(GpuGraphError::InvalidInput("Empty matrix".to_string()));
        }

        let n = matrix.num_rows;

        // Initialize with uniform vector
        let mut x: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];
        let mut eigenvalue = 0.0;

        for _ in 0..max_iterations {
            // y = A * x
            let y = self.spmv(matrix, &x, 1.0)?;

            // Compute Rayleigh quotient
            let new_eigenvalue = self.dot(&x, &y)?;

            // Normalize
            let norm = self.norm2(&y)?;
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

    /// Synchronize.
    pub fn synchronize(&self) -> Result<()> {
        self.context
            .synchronize()
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))
    }

    fn alloc_f64(&self, n: usize) -> Result<CudaSlice<f64>> {
        // SAFETY: cudarc's alloc returns properly aligned device memory. The size
        // is computed from the input data.
        unsafe {
            self.stream
                .alloc::<f64>(n)
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))
        }
    }

    fn htod_u64(&self, data: &[u64]) -> Result<CudaSlice<u64>> {
        // SAFETY: cudarc's alloc returns properly aligned device memory. The size
        // is computed from the input data.
        let mut slice = unsafe {
            self.stream
                .alloc::<u64>(data.len())
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))?
        };
        self.stream
            .memcpy_htod(data, &mut slice)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;
        Ok(slice)
    }

    fn htod_u32(&self, data: &[u32]) -> Result<CudaSlice<u32>> {
        // SAFETY: cudarc's alloc returns properly aligned device memory. The size
        // is computed from the input data.
        let mut slice = unsafe {
            self.stream
                .alloc::<u32>(data.len())
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))?
        };
        self.stream
            .memcpy_htod(data, &mut slice)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;
        Ok(slice)
    }

    fn htod_f64(&self, data: &[f64]) -> Result<CudaSlice<f64>> {
        // SAFETY: cudarc's alloc returns properly aligned device memory. The size
        // is computed from the input data.
        let mut slice = unsafe {
            self.stream
                .alloc::<f64>(data.len())
                .map_err(|e| GpuGraphError::CudaError(e.to_string()))?
        };
        self.stream
            .memcpy_htod(data, &mut slice)
            .map_err(|e| GpuGraphError::CudaError(e.to_string()))?;
        Ok(slice)
    }
}

/// Check if CUDA is available for graph operations.
pub fn is_cuda_available() -> bool {
    std::panic::catch_unwind(|| {
        cudarc::driver::CudaContext::device_count()
            .map(|c| c > 0)
            .unwrap_or(false)
    })
    .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_if_no_cuda() -> bool {
        if !is_cuda_available() {
            println!("Skipping test: CUDA not available");
            return true;
        }
        false
    }

    #[test]
    fn test_gpu_bfs_simple() {
        if skip_if_no_cuda() {
            return;
        }

        // Simple path: 0 -> 1 -> 2 -> 3
        let edges = [(0, 1), (1, 2), (2, 3)];
        let matrix = CsrMatrix::from_edges(4, &edges);

        let gpu_bfs = GpuBfs::new(0).unwrap();
        let distances = gpu_bfs.bfs(&matrix, &[0]).unwrap();

        assert_eq!(distances[0], 0);
        assert_eq!(distances[1], 1);
        assert_eq!(distances[2], 2);
        assert_eq!(distances[3], 3);
    }

    #[test]
    fn test_gpu_bfs_unreachable() {
        if skip_if_no_cuda() {
            return;
        }

        // Disconnected: 0 -> 1, 2 isolated
        let edges = [(0, 1)];
        let matrix = CsrMatrix::from_edges(3, &edges);

        let gpu_bfs = GpuBfs::new(0).unwrap();
        let distances = gpu_bfs.bfs(&matrix, &[0]).unwrap();

        assert_eq!(distances[0], 0);
        assert_eq!(distances[1], 1);
        assert_eq!(distances[2], -1); // Unreachable
    }

    #[test]
    fn test_gpu_bfs_multi_source() {
        if skip_if_no_cuda() {
            return;
        }

        // Two paths: 0 -> 1 -> 2, 3 -> 4
        let edges = [(0, 1), (1, 2), (3, 4)];
        let matrix = CsrMatrix::from_edges(5, &edges);

        let gpu_bfs = GpuBfs::new(0).unwrap();
        let distances = gpu_bfs.bfs(&matrix, &[0, 3]).unwrap();

        assert_eq!(distances[0], 0); // Source
        assert_eq!(distances[1], 1);
        assert_eq!(distances[2], 2);
        assert_eq!(distances[3], 0); // Source
        assert_eq!(distances[4], 1);
    }

    #[test]
    fn test_gpu_spmv_identity() {
        if skip_if_no_cuda() {
            return;
        }

        // Identity matrix
        let edges = [(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
        let matrix = CsrMatrix::from_weighted_edges(3, &edges);

        let gpu_spmv = GpuSpmv::new(0).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let y = gpu_spmv.spmv(&matrix, &x, 1.0).unwrap();

        assert!((y[0] - 1.0).abs() < 1e-10);
        assert!((y[1] - 2.0).abs() < 1e-10);
        assert!((y[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_spmv_weighted() {
        if skip_if_no_cuda() {
            return;
        }

        let edges = [(0, 1, 2.0), (0, 2, 3.0), (1, 2, 4.0)];
        let matrix = CsrMatrix::from_weighted_edges(3, &edges);

        let gpu_spmv = GpuSpmv::new(0).unwrap();
        let x = vec![1.0, 1.0, 1.0];
        let y = gpu_spmv.spmv(&matrix, &x, 1.0).unwrap();

        // y[0] = 2.0 * 1 + 3.0 * 1 = 5.0
        // y[1] = 4.0 * 1 = 4.0
        // y[2] = 0
        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 4.0).abs() < 1e-10);
        assert!((y[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_spmv_unweighted() {
        if skip_if_no_cuda() {
            return;
        }

        // Adjacency: 0 -> 1, 0 -> 2, 1 -> 2
        let edges = [(0, 1), (0, 2), (1, 2)];
        let matrix = CsrMatrix::from_edges(3, &edges);

        let gpu_spmv = GpuSpmv::new(0).unwrap();
        let x = vec![1.0, 1.0, 1.0];
        let y = gpu_spmv.spmv(&matrix, &x, 1.0).unwrap();

        // y[0] = x[1] + x[2] = 2.0
        // y[1] = x[2] = 1.0
        // y[2] = 0
        assert!((y[0] - 2.0).abs() < 1e-10);
        assert!((y[1] - 1.0).abs() < 1e-10);
        assert!((y[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_dot_product() {
        if skip_if_no_cuda() {
            return;
        }

        let gpu_spmv = GpuSpmv::new(0).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let dot = gpu_spmv.dot(&x, &y).unwrap();

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((dot - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_norm2() {
        if skip_if_no_cuda() {
            return;
        }

        let gpu_spmv = GpuSpmv::new(0).unwrap();
        let x = vec![3.0, 4.0];
        let norm = gpu_spmv.norm2(&x).unwrap();

        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_power_iteration() {
        if skip_if_no_cuda() {
            return;
        }

        // 2x2 symmetric matrix [[2, 1], [1, 2]]
        // Eigenvalues: 3, 1
        // Eigenvector for 3: [1/sqrt(2), 1/sqrt(2)]
        let mut builder = crate::models::CsrMatrixBuilder::new(2);
        builder.add_weighted_edge(0, 0, 2.0);
        builder.add_weighted_edge(0, 1, 1.0);
        builder.add_weighted_edge(1, 0, 1.0);
        builder.add_weighted_edge(1, 1, 2.0);
        let matrix = builder.build();

        let gpu_spmv = GpuSpmv::new(0).unwrap();
        let (eigenvector, eigenvalue) = gpu_spmv.power_iteration(&matrix, 100, 1e-6).unwrap();

        // Eigenvalue should be close to 3
        assert!(
            (eigenvalue - 3.0).abs() < 0.01,
            "Eigenvalue {} far from 3.0",
            eigenvalue
        );

        // Eigenvector should be [1/sqrt(2), 1/sqrt(2)] or [-1/sqrt(2), -1/sqrt(2)]
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (eigenvector[0].abs() - expected).abs() < 0.01,
            "Eigenvector[0] = {} far from {}",
            eigenvector[0],
            expected
        );
        assert!(
            (eigenvector[1].abs() - expected).abs() < 0.01,
            "Eigenvector[1] = {} far from {}",
            eigenvector[1],
            expected
        );
    }
}
