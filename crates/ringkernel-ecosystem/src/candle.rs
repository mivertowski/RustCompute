//! Candle ML framework integration for RingKernel.
//!
//! This module provides a bridge between Candle tensors and GPU Ring Kernels,
//! enabling custom GPU-accelerated operations in ML pipelines.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ecosystem::candle::{TensorOps, gpu_matmul};
//!
//! let result = a.gpu_add(&b, &runtime, "add_kernel").await?;
//! ```

use candle_core::{DType, Device, Shape, Tensor};
use std::sync::Arc;
use std::time::Duration;

use crate::error::{EcosystemError, Result};

/// Configuration for Candle GPU operations.
#[derive(Debug, Clone)]
pub struct CandleConfig {
    /// Default timeout for GPU operations.
    pub timeout: Duration,
    /// Device to use for result tensors.
    pub result_device: Device,
    /// Enable async operations.
    pub enable_async: bool,
}

impl Default for CandleConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            result_device: Device::Cpu,
            enable_async: true,
        }
    }
}

/// Runtime handle trait for Candle integration.
#[async_trait::async_trait]
pub trait RuntimeHandle: Send + Sync + 'static {
    /// Send tensor data for binary operation.
    async fn send_binary_tensor_op(
        &self,
        kernel_id: &str,
        a: Vec<u8>,
        b: Vec<u8>,
        shape: Vec<usize>,
        dtype: &str,
    ) -> Result<()>;

    /// Receive tensor result.
    async fn receive_tensor(&self, kernel_id: &str, timeout: Duration) -> Result<Vec<u8>>;

    /// Send tensor data for unary operation.
    async fn send_unary_tensor_op(
        &self,
        kernel_id: &str,
        data: Vec<u8>,
        shape: Vec<usize>,
        dtype: &str,
    ) -> Result<()>;

    /// Send matrix multiplication data.
    async fn send_matmul(
        &self,
        kernel_id: &str,
        a: Vec<u8>,
        b: Vec<u8>,
        m: usize,
        k: usize,
        n: usize,
        dtype: &str,
    ) -> Result<()>;
}

/// Extension trait for GPU operations on Candle tensors.
#[async_trait::async_trait]
pub trait TensorOps {
    /// Add two tensors on GPU.
    async fn gpu_add<R: RuntimeHandle>(
        &self,
        other: &Tensor,
        runtime: &R,
        kernel_id: &str,
    ) -> Result<Tensor>;

    /// Multiply two tensors element-wise on GPU.
    async fn gpu_mul<R: RuntimeHandle>(
        &self,
        other: &Tensor,
        runtime: &R,
        kernel_id: &str,
    ) -> Result<Tensor>;

    /// Apply a custom kernel to the tensor.
    async fn gpu_apply<R: RuntimeHandle>(
        &self,
        runtime: &R,
        kernel_id: &str,
        config: &CandleConfig,
    ) -> Result<Tensor>;
}

#[async_trait::async_trait]
impl TensorOps for Tensor {
    async fn gpu_add<R: RuntimeHandle>(
        &self,
        other: &Tensor,
        runtime: &R,
        kernel_id: &str,
    ) -> Result<Tensor> {
        gpu_binary_op(self, other, runtime, kernel_id, &CandleConfig::default()).await
    }

    async fn gpu_mul<R: RuntimeHandle>(
        &self,
        other: &Tensor,
        runtime: &R,
        kernel_id: &str,
    ) -> Result<Tensor> {
        gpu_binary_op(self, other, runtime, kernel_id, &CandleConfig::default()).await
    }

    async fn gpu_apply<R: RuntimeHandle>(
        &self,
        runtime: &R,
        kernel_id: &str,
        config: &CandleConfig,
    ) -> Result<Tensor> {
        let (data, dtype_str) = tensor_to_bytes(self)?;
        let shape: Vec<usize> = self.dims().to_vec();

        runtime
            .send_unary_tensor_op(kernel_id, data, shape.clone(), &dtype_str)
            .await?;

        let result_bytes = runtime.receive_tensor(kernel_id, config.timeout).await?;

        bytes_to_tensor(&result_bytes, &shape, self.dtype(), &config.result_device)
    }
}

/// Perform binary operation on GPU.
async fn gpu_binary_op<R: RuntimeHandle>(
    a: &Tensor,
    b: &Tensor,
    runtime: &R,
    kernel_id: &str,
    config: &CandleConfig,
) -> Result<Tensor> {
    if a.dims() != b.dims() {
        return Err(EcosystemError::DataConversion(
            "Tensor shapes must match".into(),
        ));
    }

    if a.dtype() != b.dtype() {
        return Err(EcosystemError::DataConversion(
            "Tensor dtypes must match".into(),
        ));
    }

    let (a_data, dtype_str) = tensor_to_bytes(a)?;
    let (b_data, _) = tensor_to_bytes(b)?;
    let shape: Vec<usize> = a.dims().to_vec();

    runtime
        .send_binary_tensor_op(kernel_id, a_data, b_data, shape.clone(), &dtype_str)
        .await?;

    let result_bytes = runtime.receive_tensor(kernel_id, config.timeout).await?;

    bytes_to_tensor(&result_bytes, &shape, a.dtype(), &config.result_device)
}

/// GPU matrix multiplication.
pub async fn gpu_matmul<R: RuntimeHandle>(
    a: &Tensor,
    b: &Tensor,
    runtime: &R,
    kernel_id: &str,
    config: &CandleConfig,
) -> Result<Tensor> {
    let a_dims = a.dims();
    let b_dims = b.dims();

    if a_dims.len() != 2 || b_dims.len() != 2 {
        return Err(EcosystemError::DataConversion(
            "Matrix multiplication requires 2D tensors".into(),
        ));
    }

    let m = a_dims[0];
    let k = a_dims[1];
    let n = b_dims[1];

    if k != b_dims[0] {
        return Err(EcosystemError::DataConversion(format!(
            "Inner dimensions must match: {} vs {}",
            k, b_dims[0]
        )));
    }

    let (a_data, dtype_str) = tensor_to_bytes(a)?;
    let (b_data, _) = tensor_to_bytes(b)?;

    runtime
        .send_matmul(kernel_id, a_data, b_data, m, k, n, &dtype_str)
        .await?;

    let result_bytes = runtime.receive_tensor(kernel_id, config.timeout).await?;

    bytes_to_tensor(&result_bytes, &[m, n], a.dtype(), &config.result_device)
}

/// Convert tensor to bytes.
fn tensor_to_bytes(tensor: &Tensor) -> Result<(Vec<u8>, String)> {
    let dtype_str = match tensor.dtype() {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::I64 => "i64",
        DType::U32 => "u32",
        DType::U8 => "u8",
        dt => {
            return Err(EcosystemError::DataConversion(format!(
                "Unsupported dtype: {:?}",
                dt
            )))
        }
    };

    // Flatten to CPU and get bytes
    let flattened = tensor
        .flatten_all()
        .map_err(|e| EcosystemError::Candle(e.to_string()))?;

    let bytes = match tensor.dtype() {
        DType::F32 => {
            let vec: Vec<f32> = flattened
                .to_vec1()
                .map_err(|e| EcosystemError::Candle(e.to_string()))?;
            vec.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        DType::F64 => {
            let vec: Vec<f64> = flattened
                .to_vec1()
                .map_err(|e| EcosystemError::Candle(e.to_string()))?;
            vec.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        DType::I64 => {
            let vec: Vec<i64> = flattened
                .to_vec1()
                .map_err(|e| EcosystemError::Candle(e.to_string()))?;
            vec.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        DType::U32 => {
            let vec: Vec<u32> = flattened
                .to_vec1()
                .map_err(|e| EcosystemError::Candle(e.to_string()))?;
            vec.iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        DType::U8 => {
            let vec: Vec<u8> = flattened
                .to_vec1()
                .map_err(|e| EcosystemError::Candle(e.to_string()))?;
            vec
        }
        _ => {
            return Err(EcosystemError::DataConversion(
                "Unsupported dtype for conversion".into(),
            ))
        }
    };

    Ok((bytes, dtype_str.to_string()))
}

/// Convert bytes to tensor.
fn bytes_to_tensor(
    bytes: &[u8],
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let tensor_shape = Shape::from_dims(shape);

    match dtype {
        DType::F32 => {
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Tensor::from_vec(values, tensor_shape, device)
                .map_err(|e| EcosystemError::Candle(e.to_string()))
        }
        DType::F64 => {
            let values: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|chunk| {
                    f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7],
                    ])
                })
                .collect();
            Tensor::from_vec(values, tensor_shape, device)
                .map_err(|e| EcosystemError::Candle(e.to_string()))
        }
        DType::I64 => {
            let values: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7],
                    ])
                })
                .collect();
            Tensor::from_vec(values, tensor_shape, device)
                .map_err(|e| EcosystemError::Candle(e.to_string()))
        }
        DType::U32 => {
            let values: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Tensor::from_vec(values, tensor_shape, device)
                .map_err(|e| EcosystemError::Candle(e.to_string()))
        }
        DType::U8 => {
            Tensor::from_vec(bytes.to_vec(), tensor_shape, device)
                .map_err(|e| EcosystemError::Candle(e.to_string()))
        }
        dt => Err(EcosystemError::DataConversion(format!(
            "Unsupported dtype: {:?}",
            dt
        ))),
    }
}

/// Builder for Candle GPU operations pipeline.
pub struct CandlePipelineBuilder<R> {
    runtime: Arc<R>,
    config: CandleConfig,
    operations: Vec<String>,
}

impl<R: RuntimeHandle> CandlePipelineBuilder<R> {
    /// Create a new pipeline builder.
    pub fn new(runtime: Arc<R>) -> Self {
        Self {
            runtime,
            config: CandleConfig::default(),
            operations: Vec::new(),
        }
    }

    /// Set the configuration.
    pub fn config(mut self, config: CandleConfig) -> Self {
        self.config = config;
        self
    }

    /// Add an operation to the pipeline.
    pub fn operation(mut self, kernel_id: &str) -> Self {
        self.operations.push(kernel_id.to_string());
        self
    }

    /// Execute the pipeline on a tensor.
    pub async fn execute(&self, input: Tensor) -> Result<Tensor> {
        let mut current = input;

        for kernel_id in &self.operations {
            current = current
                .gpu_apply(&*self.runtime, kernel_id, &self.config)
                .await?;
        }

        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRuntime;

    #[async_trait::async_trait]
    impl RuntimeHandle for MockRuntime {
        async fn send_binary_tensor_op(
            &self,
            _kernel_id: &str,
            _a: Vec<u8>,
            _b: Vec<u8>,
            _shape: Vec<usize>,
            _dtype: &str,
        ) -> Result<()> {
            Ok(())
        }

        async fn receive_tensor(&self, _kernel_id: &str, _timeout: Duration) -> Result<Vec<u8>> {
            // Return 2 f32 values
            Ok(vec![0, 0, 128, 64, 0, 0, 0, 65]) // [4.0f32, 8.0f32]
        }

        async fn send_unary_tensor_op(
            &self,
            _kernel_id: &str,
            _data: Vec<u8>,
            _shape: Vec<usize>,
            _dtype: &str,
        ) -> Result<()> {
            Ok(())
        }

        async fn send_matmul(
            &self,
            _kernel_id: &str,
            _a: Vec<u8>,
            _b: Vec<u8>,
            _m: usize,
            _k: usize,
            _n: usize,
            _dtype: &str,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_candle_config_default() {
        let config = CandleConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.enable_async);
    }

    #[test]
    fn test_tensor_to_bytes_f32() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &Device::Cpu).unwrap();
        let (bytes, dtype) = tensor_to_bytes(&tensor).unwrap();
        assert_eq!(bytes.len(), 12); // 3 * 4 bytes
        assert_eq!(dtype, "f32");
    }

    #[test]
    fn test_bytes_to_tensor_f32() {
        // 1.0f32 and 2.0f32 in little-endian
        let bytes = vec![0, 0, 128, 63, 0, 0, 0, 64];
        let tensor = bytes_to_tensor(&bytes, &[2], DType::F32, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[2]);
    }

    #[test]
    fn test_pipeline_builder() {
        let runtime = Arc::new(MockRuntime);
        let builder = CandlePipelineBuilder::new(runtime)
            .operation("kernel1")
            .operation("kernel2");

        assert_eq!(builder.operations.len(), 2);
    }
}
