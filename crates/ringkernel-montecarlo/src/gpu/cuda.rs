//! CUDA implementation for Monte Carlo operations.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use super::{IMPORTANCE_KERNEL_SOURCE, PHILOX_KERNEL_SOURCE};

/// Error type for GPU Monte Carlo operations.
#[derive(Debug, thiserror::Error)]
pub enum GpuMonteCarloError {
    /// CUDA driver error.
    #[error("CUDA error: {0}")]
    CudaError(String),
    /// Compilation error.
    #[error("Compilation error: {0}")]
    CompilationError(String),
    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

type Result<T> = std::result::Result<T, GpuMonteCarloError>;

/// GPU-accelerated Philox RNG.
pub struct GpuPhiloxRng {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    fill_uniform: CudaFunction,
    fill_normal: CudaFunction,
    seed: (u32, u32),
}

impl GpuPhiloxRng {
    /// Create a new GPU Philox RNG.
    pub fn new(device_ordinal: usize) -> Result<Self> {
        Self::with_seed(device_ordinal, 42, 0)
    }

    /// Create a new GPU Philox RNG with specific seed.
    pub fn with_seed(device_ordinal: usize, seed_lo: u32, seed_hi: u32) -> Result<Self> {
        let context = CudaContext::new(device_ordinal)
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        let stream = context.default_stream();

        // Compile Philox kernels
        let ptx = compile_ptx(PHILOX_KERNEL_SOURCE)
            .map_err(|e| GpuMonteCarloError::CompilationError(e.to_string()))?;

        let module = context
            .load_module(ptx)
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        let fill_uniform = module
            .load_function("philox_fill_uniform")
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        let fill_normal = module
            .load_function("philox_fill_normal")
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        Ok(Self {
            context,
            stream,
            fill_uniform,
            fill_normal,
            seed: (seed_lo, seed_hi),
        })
    }

    /// Set seed.
    pub fn set_seed(&mut self, seed_lo: u32, seed_hi: u32) {
        self.seed = (seed_lo, seed_hi);
    }

    /// Fill a device buffer with uniform random numbers in [0, 1).
    pub fn fill_uniform(&self, output: &mut CudaSlice<f32>) -> Result<()> {
        let n = output.len() as u32;
        let block_size = 256u32;
        let grid_size = n.div_ceil(block_size);

        // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
        // are valid and allocated with sufficient size.
        unsafe {
            self.stream
                .launch_builder(&self.fill_uniform)
                .arg(output)
                .arg(&n)
                .arg(&self.seed.0)
                .arg(&self.seed.1)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;
        }

        Ok(())
    }

    /// Fill a device buffer with standard normal random numbers.
    pub fn fill_normal(&self, output: &mut CudaSlice<f32>) -> Result<()> {
        let n = output.len() as u32;
        let block_size = 256u32;
        let grid_size = n.div_ceil(block_size);

        // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
        // are valid and allocated with sufficient size.
        unsafe {
            self.stream
                .launch_builder(&self.fill_normal)
                .arg(output)
                .arg(&n)
                .arg(&self.seed.0)
                .arg(&self.seed.1)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;
        }

        Ok(())
    }

    /// Generate uniform random numbers and copy to host.
    pub fn generate_uniform(&self, n: usize) -> Result<Vec<f32>> {
        // SAFETY: cudarc's alloc returns properly aligned device memory. The size
        // is computed from the input data.
        let mut output = unsafe {
            self.stream
                .alloc::<f32>(n)
                .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?
        };

        self.fill_uniform(&mut output)?;

        let mut host = vec![0.0f32; n];
        self.stream
            .memcpy_dtoh(&output, &mut host)
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        Ok(host)
    }

    /// Generate standard normal random numbers and copy to host.
    pub fn generate_normal(&self, n: usize) -> Result<Vec<f32>> {
        // SAFETY: cudarc's alloc returns properly aligned device memory. The size
        // is computed from the input data.
        let mut output = unsafe {
            self.stream
                .alloc::<f32>(n)
                .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?
        };

        self.fill_normal(&mut output)?;

        let mut host = vec![0.0f32; n];
        self.stream
            .memcpy_dtoh(&output, &mut host)
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        Ok(host)
    }

    /// Allocate device buffer.
    pub fn alloc(&self, n: usize) -> Result<CudaSlice<f32>> {
        // SAFETY: cudarc's alloc returns properly aligned device memory. The size
        // is computed from the input data.
        unsafe {
            self.stream
                .alloc::<f32>(n)
                .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))
        }
    }

    /// Copy data from host to device.
    pub fn htod(&self, data: &[f32]) -> Result<CudaSlice<f32>> {
        let mut slice = self.alloc(data.len())?;
        self.stream
            .memcpy_htod(data, &mut slice)
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;
        Ok(slice)
    }

    /// Copy data from device to host.
    pub fn dtoh(&self, slice: &CudaSlice<f32>) -> Result<Vec<f32>> {
        let mut host = vec![0.0f32; slice.len()];
        self.stream
            .memcpy_dtoh(slice, &mut host)
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;
        Ok(host)
    }

    /// Synchronize the stream.
    pub fn synchronize(&self) -> Result<()> {
        self.context
            .synchronize()
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))
    }

    /// Get the underlying context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get the stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// GPU-accelerated antithetic variates.
pub struct GpuAntitheticVariates {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    transform_fn: CudaFunction,
    mean_fn: CudaFunction,
}

impl GpuAntitheticVariates {
    /// Create from an existing RNG context.
    pub fn new(rng: &GpuPhiloxRng) -> Result<Self> {
        // Compile the antithetic kernels
        let ptx = compile_ptx(PHILOX_KERNEL_SOURCE)
            .map_err(|e| GpuMonteCarloError::CompilationError(e.to_string()))?;

        let module = rng
            .context()
            .load_module(ptx)
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        let transform_fn = module
            .load_function("antithetic_transform")
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        let mean_fn = module
            .load_function("antithetic_mean")
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        Ok(Self {
            context: rng.context().clone(),
            stream: rng.stream().clone(),
            transform_fn,
            mean_fn,
        })
    }

    /// Transform input samples to antithetic pairs.
    ///
    /// Given input x, produces (x, -x) pairs.
    pub fn transform(
        &self,
        input: &CudaSlice<f32>,
        output_pos: &mut CudaSlice<f32>,
        output_neg: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let n = input.len() as u32;
        let block_size = 256u32;
        let grid_size = n.div_ceil(block_size);

        // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
        // are valid and allocated with sufficient size.
        unsafe {
            self.stream
                .launch_builder(&self.transform_fn)
                .arg(input)
                .arg(output_pos)
                .arg(output_neg)
                .arg(&n)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;
        }

        Ok(())
    }

    /// Compute antithetic mean: (f_pos + f_neg) / 2.
    pub fn compute_mean(
        &self,
        f_pos: &CudaSlice<f32>,
        f_neg: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let n = f_pos.len() as u32;
        let block_size = 256u32;
        let grid_size = n.div_ceil(block_size);

        // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
        // are valid and allocated with sufficient size.
        unsafe {
            self.stream
                .launch_builder(&self.mean_fn)
                .arg(f_pos)
                .arg(f_neg)
                .arg(output)
                .arg(&n)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;
        }

        Ok(())
    }

    /// Synchronize.
    pub fn synchronize(&self) -> Result<()> {
        self.context
            .synchronize()
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))
    }
}

/// GPU-accelerated importance sampling.
pub struct GpuImportanceSampling {
    #[allow(dead_code)]
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    weights_fn: CudaFunction,
    reduce_fn: CudaFunction,
}

impl GpuImportanceSampling {
    /// Create from an existing RNG context.
    pub fn new(rng: &GpuPhiloxRng) -> Result<Self> {
        let ptx = compile_ptx(IMPORTANCE_KERNEL_SOURCE)
            .map_err(|e| GpuMonteCarloError::CompilationError(e.to_string()))?;

        let module = rng
            .context()
            .load_module(ptx)
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        let weights_fn = module
            .load_function("importance_weights")
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        let reduce_fn = module
            .load_function("weighted_sum_reduce")
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        Ok(Self {
            context: rng.context().clone(),
            stream: rng.stream().clone(),
            weights_fn,
            reduce_fn,
        })
    }

    /// Compute importance weights: w = exp(log_p - log_q).
    pub fn compute_weights(
        &self,
        log_p: &CudaSlice<f32>,
        log_q: &CudaSlice<f32>,
        weights: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let n = log_p.len() as u32;
        let block_size = 256u32;
        let grid_size = n.div_ceil(block_size);

        // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
        // are valid and allocated with sufficient size.
        unsafe {
            self.stream
                .launch_builder(&self.weights_fn)
                .arg(log_p)
                .arg(log_q)
                .arg(weights)
                .arg(&n)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;
        }

        Ok(())
    }

    /// Compute weighted sum using block reduction.
    ///
    /// Returns (weighted_sum, weight_sum) for self-normalized estimation.
    pub fn weighted_reduce(
        &self,
        values: &CudaSlice<f32>,
        weights: &CudaSlice<f32>,
    ) -> Result<(f32, f32)> {
        let n = values.len() as u32;
        let block_size = 256u32;
        let num_blocks = n.div_ceil(block_size);

        // Allocate partial sums
        // SAFETY: cudarc's alloc returns properly aligned device memory. The size
        // is computed from the input data.
        let mut partial_weighted = unsafe {
            self.stream
                .alloc::<f32>(num_blocks as usize)
                .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?
        };
        // SAFETY: cudarc's alloc returns properly aligned device memory. The size
        // is computed from the input data.
        let mut partial_weights = unsafe {
            self.stream
                .alloc::<f32>(num_blocks as usize)
                .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?
        };

        // First pass: block-level reduction
        // SAFETY: Kernel arguments match the compiled PTX signature. Device pointers
        // are valid and allocated with sufficient size.
        unsafe {
            self.stream
                .launch_builder(&self.reduce_fn)
                .arg(values)
                .arg(weights)
                .arg(&mut partial_weighted)
                .arg(&mut partial_weights)
                .arg(&n)
                .launch(cudarc::driver::LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;
        }

        // Copy to host and sum
        let mut weighted_sums = vec![0.0f32; num_blocks as usize];
        let mut weight_sums = vec![0.0f32; num_blocks as usize];

        self.stream
            .memcpy_dtoh(&partial_weighted, &mut weighted_sums)
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;
        self.stream
            .memcpy_dtoh(&partial_weights, &mut weight_sums)
            .map_err(|e| GpuMonteCarloError::CudaError(e.to_string()))?;

        let total_weighted: f32 = weighted_sums.iter().sum();
        let total_weight: f32 = weight_sums.iter().sum();

        Ok((total_weighted, total_weight))
    }
}

/// Check if CUDA is available for Monte Carlo operations.
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
    fn test_gpu_philox_uniform() {
        if skip_if_no_cuda() {
            return;
        }

        let rng = GpuPhiloxRng::new(0).unwrap();
        let samples = rng.generate_uniform(10000).unwrap();

        // Check all values are in [0, 1)
        for &x in &samples {
            assert!(x >= 0.0 && x < 1.0, "Uniform sample {} out of range", x);
        }

        // Check mean is approximately 0.5
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(
            (mean - 0.5).abs() < 0.05,
            "Uniform mean {} far from 0.5",
            mean
        );
    }

    #[test]
    fn test_gpu_philox_normal() {
        if skip_if_no_cuda() {
            return;
        }

        let rng = GpuPhiloxRng::new(0).unwrap();
        let samples = rng.generate_normal(10000).unwrap();

        // Check mean is approximately 0
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 0.1, "Normal mean {} far from 0", mean);

        // Check std is approximately 1
        let variance: f32 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 0.1, "Normal std {} far from 1", std);
    }

    #[test]
    fn test_gpu_antithetic() {
        if skip_if_no_cuda() {
            return;
        }

        let rng = GpuPhiloxRng::new(0).unwrap();
        let av = GpuAntitheticVariates::new(&rng).unwrap();

        // Generate samples
        let n = 1000;
        let mut input = rng.alloc(n).unwrap();
        rng.fill_normal(&mut input).unwrap();

        let mut pos = rng.alloc(n).unwrap();
        let mut neg = rng.alloc(n).unwrap();

        av.transform(&input, &mut pos, &mut neg).unwrap();
        av.synchronize().unwrap();

        // Verify: pos should equal input, neg should equal -input
        let input_host = rng.dtoh(&input).unwrap();
        let pos_host = rng.dtoh(&pos).unwrap();
        let neg_host = rng.dtoh(&neg).unwrap();

        for i in 0..n {
            assert!(
                (pos_host[i] - input_host[i]).abs() < 1e-6,
                "pos[{}] = {} != input = {}",
                i,
                pos_host[i],
                input_host[i]
            );
            assert!(
                (neg_host[i] + input_host[i]).abs() < 1e-6,
                "neg[{}] = {} != -input = {}",
                i,
                neg_host[i],
                -input_host[i]
            );
        }
    }

    #[test]
    fn test_gpu_importance_weights() {
        if skip_if_no_cuda() {
            return;
        }

        let rng = GpuPhiloxRng::new(0).unwrap();
        let is = GpuImportanceSampling::new(&rng).unwrap();

        // Test with equal distributions (weights should be 1)
        let n = 100;
        let log_p: Vec<f32> = vec![-1.0; n];
        let log_q: Vec<f32> = vec![-1.0; n];

        let log_p_dev = rng.htod(&log_p).unwrap();
        let log_q_dev = rng.htod(&log_q).unwrap();
        let mut weights_dev = rng.alloc(n).unwrap();

        is.compute_weights(&log_p_dev, &log_q_dev, &mut weights_dev)
            .unwrap();

        let weights = rng.dtoh(&weights_dev).unwrap();

        // All weights should be exp(0) = 1
        for (i, &w) in weights.iter().enumerate() {
            assert!((w - 1.0).abs() < 1e-5, "Weight[{}] = {} != 1.0", i, w);
        }
    }

    #[test]
    fn test_gpu_weighted_reduce() {
        if skip_if_no_cuda() {
            return;
        }

        let rng = GpuPhiloxRng::new(0).unwrap();
        let is = GpuImportanceSampling::new(&rng).unwrap();

        // Test: sum([1,2,3,4] * [1,1,1,1]) / sum([1,1,1,1]) = 10/4 = 2.5
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let weights: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

        let values_dev = rng.htod(&values).unwrap();
        let weights_dev = rng.htod(&weights).unwrap();

        let (weighted_sum, weight_sum) = is.weighted_reduce(&values_dev, &weights_dev).unwrap();

        assert!(
            (weighted_sum - 10.0).abs() < 1e-5,
            "Weighted sum {} != 10.0",
            weighted_sum
        );
        assert!(
            (weight_sum - 4.0).abs() < 1e-5,
            "Weight sum {} != 4.0",
            weight_sum
        );
    }
}
