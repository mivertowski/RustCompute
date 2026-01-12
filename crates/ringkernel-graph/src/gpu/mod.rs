//! GPU-accelerated graph algorithms.
//!
//! This module provides CUDA implementations of graph primitives.

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::*;

/// CUDA kernel source for BFS (Breadth-First Search).
///
/// Implements level-synchronous parallel BFS on CSR graph.
pub const BFS_KERNEL_SOURCE: &str = r#"
// Level-synchronous BFS kernel
// Each iteration processes all nodes at the current frontier level

// Advance frontier: mark neighbors of frontier nodes
extern "C" __global__ void bfs_advance(
    const unsigned long long* row_ptr,    // CSR row pointers
    const unsigned int* col_idx,          // CSR column indices
    const int* distances,                 // Current distances
    int* new_distances,                   // New distances (output)
    int* frontier_size,                   // Output: size of next frontier
    unsigned int num_nodes,
    int current_level
) {
    unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    // Only process nodes at current level
    if (distances[node] != current_level) return;

    // Explore neighbors
    unsigned long long row_start = row_ptr[node];
    unsigned long long row_end = row_ptr[node + 1];

    for (unsigned long long i = row_start; i < row_end; i++) {
        unsigned int neighbor = col_idx[i];

        // Try to claim this neighbor (if unvisited)
        if (distances[neighbor] == -1) {
            // Atomic CAS to prevent races
            int old = atomicCAS(&new_distances[neighbor], -1, current_level + 1);
            if (old == -1) {
                // Successfully claimed - increment frontier
                atomicAdd(frontier_size, 1);
            }
        }
    }
}

// Initialize distances: source nodes get 0, others get -1
extern "C" __global__ void bfs_init(
    int* distances,
    const unsigned int* sources,
    unsigned int num_nodes,
    unsigned int num_sources
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize all to -1
    if (idx < num_nodes) {
        distances[idx] = -1;
    }

    // Set sources to 0
    if (idx < num_sources) {
        distances[sources[idx]] = 0;
    }
}

// Copy distances for next iteration
extern "C" __global__ void bfs_copy(
    const int* src,
    int* dst,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}
"#;

/// CUDA kernel source for SpMV (Sparse Matrix-Vector Multiplication).
///
/// Implements CSR-based SpMV: y = alpha * A * x + beta * y
pub const SPMV_KERNEL_SOURCE: &str = r#"
// CSR SpMV: y = alpha * A * x
// Each thread handles one row
extern "C" __global__ void spmv_csr(
    const unsigned long long* row_ptr,    // CSR row pointers
    const unsigned int* col_idx,          // CSR column indices
    const double* values,                 // CSR values (can be NULL for unweighted)
    const double* x,                      // Input vector
    double* y,                            // Output vector
    unsigned int num_rows,
    double alpha,
    int has_values                        // 1 if values array is valid
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    unsigned long long row_start = row_ptr[row];
    unsigned long long row_end = row_ptr[row + 1];

    double sum = 0.0;

    if (has_values) {
        for (unsigned long long i = row_start; i < row_end; i++) {
            unsigned int col = col_idx[i];
            sum += values[i] * x[col];
        }
    } else {
        // Unweighted: treat all edges as 1.0
        for (unsigned long long i = row_start; i < row_end; i++) {
            unsigned int col = col_idx[i];
            sum += x[col];
        }
    }

    y[row] = alpha * sum;
}

// CSR SpMV with accumulation: y = alpha * A * x + beta * y
extern "C" __global__ void spmv_csr_axpby(
    const unsigned long long* row_ptr,
    const unsigned int* col_idx,
    const double* values,
    const double* x,
    double* y,
    unsigned int num_rows,
    double alpha,
    double beta,
    int has_values
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    unsigned long long row_start = row_ptr[row];
    unsigned long long row_end = row_ptr[row + 1];

    double sum = 0.0;

    if (has_values) {
        for (unsigned long long i = row_start; i < row_end; i++) {
            unsigned int col = col_idx[i];
            sum += values[i] * x[col];
        }
    } else {
        for (unsigned long long i = row_start; i < row_end; i++) {
            unsigned int col = col_idx[i];
            sum += x[col];
        }
    }

    y[row] = alpha * sum + beta * y[row];
}

// Dot product with block reduction
extern "C" __global__ void dot_product(
    const double* x,
    const double* y,
    double* partial_sums,
    unsigned int n
) {
    __shared__ double sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? x[idx] * y[idx] : 0.0;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Vector scaling: x = x * scale
extern "C" __global__ void vector_scale(
    double* x,
    unsigned int n,
    double scale
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= scale;
    }
}

// Vector copy
extern "C" __global__ void vector_copy(
    const double* src,
    double* dst,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}
"#;
