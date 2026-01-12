//! GPU-accelerated Monte Carlo operations.
//!
//! This module provides CUDA implementations of Monte Carlo primitives.

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::*;

/// CUDA kernel source for Philox RNG.
///
/// Implements Philox4x32-10 counter-based RNG on GPU.
pub const PHILOX_KERNEL_SOURCE: &str = r#"
// Philox4x32-10 constants
#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

// Single Philox round
__device__ void philox_round(unsigned int* ctr, unsigned int* key) {
    // Multiply and get hi/lo parts
    unsigned int lo0 = PHILOX_M0 * ctr[0];
    unsigned int hi0 = __umulhi(PHILOX_M0, ctr[0]);
    unsigned int lo1 = PHILOX_M1 * ctr[2];
    unsigned int hi1 = __umulhi(PHILOX_M1, ctr[2]);

    // Feistel-like mixing
    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
}

// Bump key
__device__ void philox_bump_key(unsigned int* key) {
    key[0] += PHILOX_W0;
    key[1] += PHILOX_W1;
}

// Full Philox4x32-10 generation
__device__ void philox4x32_10(unsigned int* ctr, unsigned int* key) {
    philox_round(ctr, key); philox_bump_key(key);
    philox_round(ctr, key); philox_bump_key(key);
    philox_round(ctr, key); philox_bump_key(key);
    philox_round(ctr, key); philox_bump_key(key);
    philox_round(ctr, key); philox_bump_key(key);
    philox_round(ctr, key); philox_bump_key(key);
    philox_round(ctr, key); philox_bump_key(key);
    philox_round(ctr, key); philox_bump_key(key);
    philox_round(ctr, key); philox_bump_key(key);
    philox_round(ctr, key);
}

// Convert u32 to uniform f32 in [0, 1)
__device__ float u32_to_uniform(unsigned int x) {
    return (float)(x >> 8) * (1.0f / 16777216.0f);
}

// Box-Muller transform for normal distribution
__device__ void box_muller(float u1, float u2, float* n1, float* n2) {
    float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
    float theta = 2.0f * 3.14159265358979f * u2;
    *n1 = r * cosf(theta);
    *n2 = r * sinf(theta);
}

// Fill array with uniform random numbers
extern "C" __global__ void philox_fill_uniform(
    float* output,
    unsigned int n,
    unsigned int seed_lo,
    unsigned int seed_hi
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Each thread has unique counter based on index
    unsigned int ctr[4] = {idx, 0, 0, 0};
    unsigned int key[2] = {seed_lo, seed_hi};

    philox4x32_10(ctr, key);

    output[idx] = u32_to_uniform(ctr[0]);
}

// Fill array with normal random numbers
extern "C" __global__ void philox_fill_normal(
    float* output,
    unsigned int n,
    unsigned int seed_lo,
    unsigned int seed_hi
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Each thread generates its own unique random values
    // Philox4x32 generates 4 u32 values, we use positions 0 and 2 for less correlation
    unsigned int ctr[4] = {idx, 0, idx ^ 0xDEADBEEF, 0};
    unsigned int key[2] = {seed_lo, seed_hi};

    philox4x32_10(ctr, key);

    // Use ctr[0] and ctr[2] for the two uniforms (less correlated)
    float u1 = u32_to_uniform(ctr[0]);
    float u2 = u32_to_uniform(ctr[2]);

    // Ensure u1 > 0 to avoid log(0)
    if (u1 < 1e-7f) u1 = 1e-7f;

    // Box-Muller transform
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * 3.14159265358979f * u2;

    output[idx] = r * cosf(theta);
}

// Antithetic variates: generate (x, -x) pairs
extern "C" __global__ void antithetic_transform(
    const float* input,
    float* output_pos,
    float* output_neg,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];
    output_pos[idx] = x;
    output_neg[idx] = -x;
}

// Monte Carlo estimation with antithetic variates
// Computes mean of f(x) and f(-x) for variance reduction
extern "C" __global__ void antithetic_mean(
    const float* f_pos,
    const float* f_neg,
    float* output,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = 0.5f * (f_pos[idx] + f_neg[idx]);
}
"#;

/// CUDA kernel source for importance sampling.
pub const IMPORTANCE_KERNEL_SOURCE: &str = r#"
// Importance sampling weight computation
extern "C" __global__ void importance_weights(
    const float* log_p,      // log target density
    const float* log_q,      // log proposal density
    float* weights,          // output weights
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    weights[idx] = expf(log_p[idx] - log_q[idx]);
}

// Weighted sum reduction (single block version for simplicity)
extern "C" __global__ void weighted_sum_reduce(
    const float* values,
    const float* weights,
    float* partial_sums,     // output: weighted sum
    float* weight_sums,      // output: weight sum
    unsigned int n
) {
    __shared__ float s_weighted[256];
    __shared__ float s_weights[256];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    s_weighted[tid] = (idx < n) ? values[idx] * weights[idx] : 0.0f;
    s_weights[tid] = (idx < n) ? weights[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_weighted[tid] += s_weighted[tid + s];
            s_weights[tid] += s_weights[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        partial_sums[blockIdx.x] = s_weighted[0];
        weight_sums[blockIdx.x] = s_weights[0];
    }
}
"#;
