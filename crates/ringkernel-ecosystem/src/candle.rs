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
fn bytes_to_tensor(bytes: &[u8], shape: &[usize], dtype: DType, device: &Device) -> Result<Tensor> {
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
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
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
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
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
        DType::U8 => Tensor::from_vec(bytes.to_vec(), tensor_shape, device)
            .map_err(|e| EcosystemError::Candle(e.to_string())),
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

// ============================================================================
// Enhanced GPU Operations for Candle
// ============================================================================

/// GPU activation function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuActivation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Leaky ReLU: x if x > 0 else alpha * x
    LeakyReLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Softmax (applied along last dimension)
    Softmax,
    /// Log softmax
    LogSoftmax,
    /// Swish: x * sigmoid(x)
    Swish,
    /// Mish: x * tanh(softplus(x))
    Mish,
    /// SiLU (same as Swish)
    SiLU,
    /// Hard sigmoid
    HardSigmoid,
    /// Hard swish
    HardSwish,
}

/// Convolution configuration.
#[derive(Debug, Clone)]
pub struct GpuConv2dConfig {
    /// Kernel size (height, width)
    pub kernel_size: (usize, usize),
    /// Stride (height, width)
    pub stride: (usize, usize),
    /// Padding (height, width)
    pub padding: (usize, usize),
    /// Dilation (height, width)
    pub dilation: (usize, usize),
    /// Number of groups for grouped convolution
    pub groups: usize,
    /// Include bias
    pub bias: bool,
}

impl Default for GpuConv2dConfig {
    fn default() -> Self {
        Self {
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            bias: true,
        }
    }
}

impl GpuConv2dConfig {
    /// Create a new conv2d config.
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            ..Default::default()
        }
    }

    /// Set stride.
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding.
    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Same padding (output same size as input).
    pub fn same_padding(mut self) -> Self {
        self.padding = (self.kernel_size.0 / 2, self.kernel_size.1 / 2);
        self
    }
}

/// Pooling type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPoolingType {
    /// Max pooling
    Max,
    /// Average pooling
    Avg,
    /// Global max pooling
    GlobalMax,
    /// Global average pooling
    GlobalAvg,
}

/// Pooling configuration.
#[derive(Debug, Clone)]
pub struct GpuPoolingConfig {
    /// Pooling type
    pub pool_type: GpuPoolingType,
    /// Kernel size
    pub kernel_size: (usize, usize),
    /// Stride
    pub stride: (usize, usize),
    /// Padding
    pub padding: (usize, usize),
}

impl Default for GpuPoolingConfig {
    fn default() -> Self {
        Self {
            pool_type: GpuPoolingType::Max,
            kernel_size: (2, 2),
            stride: (2, 2),
            padding: (0, 0),
        }
    }
}

/// Normalization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuNormalization {
    /// Batch normalization
    BatchNorm,
    /// Layer normalization
    LayerNorm,
    /// Instance normalization
    InstanceNorm,
    /// Group normalization
    GroupNorm,
    /// RMS normalization
    RMSNorm,
}

/// Attention configuration.
#[derive(Debug, Clone)]
pub struct GpuAttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Use causal mask
    pub causal: bool,
    /// Use flash attention
    pub flash_attention: bool,
}

impl Default for GpuAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            dropout: 0.0,
            causal: false,
            flash_attention: true,
        }
    }
}

/// Extended runtime handle for enhanced Candle GPU operations.
#[async_trait::async_trait]
pub trait GpuCandleOps: Send + Sync + 'static {
    /// GPU-accelerated activation function.
    async fn gpu_activation(
        &self,
        kernel_id: &str,
        data: Vec<u8>,
        shape: Vec<usize>,
        activation: GpuActivation,
    ) -> Result<Vec<u8>>;

    /// GPU-accelerated convolution.
    async fn gpu_conv2d(
        &self,
        kernel_id: &str,
        input: Vec<u8>,
        weight: Vec<u8>,
        bias: Option<Vec<u8>>,
        config: &GpuConv2dConfig,
    ) -> Result<Vec<u8>>;

    /// GPU-accelerated pooling.
    async fn gpu_pooling(
        &self,
        kernel_id: &str,
        data: Vec<u8>,
        shape: Vec<usize>,
        config: &GpuPoolingConfig,
    ) -> Result<Vec<u8>>;

    /// GPU-accelerated normalization.
    async fn gpu_normalize(
        &self,
        kernel_id: &str,
        data: Vec<u8>,
        shape: Vec<usize>,
        norm_type: GpuNormalization,
    ) -> Result<Vec<u8>>;

    /// GPU-accelerated attention.
    async fn gpu_attention(
        &self,
        kernel_id: &str,
        q: Vec<u8>,
        k: Vec<u8>,
        v: Vec<u8>,
        config: &GpuAttentionConfig,
    ) -> Result<Vec<u8>>;

    /// GPU-accelerated linear layer.
    async fn gpu_linear(
        &self,
        kernel_id: &str,
        input: Vec<u8>,
        weight: Vec<u8>,
        bias: Option<Vec<u8>>,
    ) -> Result<Vec<u8>>;

    /// GPU-accelerated embedding lookup.
    async fn gpu_embedding(
        &self,
        kernel_id: &str,
        indices: Vec<u32>,
        weight: Vec<u8>,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<Vec<u8>>;
}

/// Enhanced Candle GPU executor with ML operations.
pub struct GpuCandleExecutor<R> {
    runtime: Arc<R>,
    config: CandleConfig,
}

impl<R: RuntimeHandle + GpuCandleOps> GpuCandleExecutor<R> {
    /// Create a new GPU Candle executor.
    pub fn new(runtime: Arc<R>) -> Self {
        Self {
            runtime,
            config: CandleConfig::default(),
        }
    }

    /// Set configuration.
    pub fn with_config(mut self, config: CandleConfig) -> Self {
        self.config = config;
        self
    }

    /// Apply activation function on GPU.
    pub async fn activation(&self, tensor: &Tensor, activation: GpuActivation) -> Result<Tensor> {
        let (data, _dtype_str) = tensor_to_bytes(tensor)?;
        let shape = tensor.dims().to_vec();

        let result_bytes = self
            .runtime
            .gpu_activation("activation", data, shape.clone(), activation)
            .await?;

        bytes_to_tensor(
            &result_bytes,
            &shape,
            tensor.dtype(),
            &self.config.result_device,
        )
    }

    /// ReLU activation.
    pub async fn relu(&self, tensor: &Tensor) -> Result<Tensor> {
        self.activation(tensor, GpuActivation::ReLU).await
    }

    /// GELU activation.
    pub async fn gelu(&self, tensor: &Tensor) -> Result<Tensor> {
        self.activation(tensor, GpuActivation::GELU).await
    }

    /// Sigmoid activation.
    pub async fn sigmoid(&self, tensor: &Tensor) -> Result<Tensor> {
        self.activation(tensor, GpuActivation::Sigmoid).await
    }

    /// Softmax along last dimension.
    pub async fn softmax(&self, tensor: &Tensor) -> Result<Tensor> {
        self.activation(tensor, GpuActivation::Softmax).await
    }

    /// Apply 2D convolution.
    pub async fn conv2d(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        config: &GpuConv2dConfig,
    ) -> Result<Tensor> {
        let (input_data, _) = tensor_to_bytes(input)?;
        let (weight_data, _) = tensor_to_bytes(weight)?;
        let bias_data = if let Some(b) = bias {
            Some(tensor_to_bytes(b)?.0)
        } else {
            None
        };

        let result_bytes = self
            .runtime
            .gpu_conv2d("conv2d", input_data, weight_data, bias_data, config)
            .await?;

        // Calculate output shape
        let in_shape = input.dims();
        let out_h =
            (in_shape[2] + 2 * config.padding.0 - config.kernel_size.0) / config.stride.0 + 1;
        let out_w =
            (in_shape[3] + 2 * config.padding.1 - config.kernel_size.1) / config.stride.1 + 1;
        let out_channels = weight.dims()[0];

        bytes_to_tensor(
            &result_bytes,
            &[in_shape[0], out_channels, out_h, out_w],
            input.dtype(),
            &self.config.result_device,
        )
    }

    /// Apply 2D max pooling.
    pub async fn max_pool2d(&self, tensor: &Tensor, kernel_size: (usize, usize)) -> Result<Tensor> {
        let config = GpuPoolingConfig {
            pool_type: GpuPoolingType::Max,
            kernel_size,
            stride: kernel_size,
            padding: (0, 0),
        };

        let (data, _) = tensor_to_bytes(tensor)?;
        let shape = tensor.dims().to_vec();

        let result_bytes = self
            .runtime
            .gpu_pooling("max_pool", data, shape.clone(), &config)
            .await?;

        let out_h = (shape[2] - config.kernel_size.0) / config.stride.0 + 1;
        let out_w = (shape[3] - config.kernel_size.1) / config.stride.1 + 1;

        bytes_to_tensor(
            &result_bytes,
            &[shape[0], shape[1], out_h, out_w],
            tensor.dtype(),
            &self.config.result_device,
        )
    }

    /// Apply batch normalization.
    pub async fn batch_norm(&self, tensor: &Tensor) -> Result<Tensor> {
        let (data, _) = tensor_to_bytes(tensor)?;
        let shape = tensor.dims().to_vec();

        let result_bytes = self
            .runtime
            .gpu_normalize(
                "batch_norm",
                data,
                shape.clone(),
                GpuNormalization::BatchNorm,
            )
            .await?;

        bytes_to_tensor(
            &result_bytes,
            &shape,
            tensor.dtype(),
            &self.config.result_device,
        )
    }

    /// Apply layer normalization.
    pub async fn layer_norm(&self, tensor: &Tensor) -> Result<Tensor> {
        let (data, _) = tensor_to_bytes(tensor)?;
        let shape = tensor.dims().to_vec();

        let result_bytes = self
            .runtime
            .gpu_normalize(
                "layer_norm",
                data,
                shape.clone(),
                GpuNormalization::LayerNorm,
            )
            .await?;

        bytes_to_tensor(
            &result_bytes,
            &shape,
            tensor.dtype(),
            &self.config.result_device,
        )
    }

    /// Apply multi-head attention.
    pub async fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        config: &GpuAttentionConfig,
    ) -> Result<Tensor> {
        let (q_data, _) = tensor_to_bytes(q)?;
        let (k_data, _) = tensor_to_bytes(k)?;
        let (v_data, _) = tensor_to_bytes(v)?;

        let result_bytes = self
            .runtime
            .gpu_attention("attention", q_data, k_data, v_data, config)
            .await?;

        bytes_to_tensor(
            &result_bytes,
            q.dims(),
            q.dtype(),
            &self.config.result_device,
        )
    }

    /// Apply linear transformation.
    pub async fn linear(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (input_data, _) = tensor_to_bytes(input)?;
        let (weight_data, _) = tensor_to_bytes(weight)?;
        let bias_data = if let Some(b) = bias {
            Some(tensor_to_bytes(b)?.0)
        } else {
            None
        };

        let result_bytes = self
            .runtime
            .gpu_linear("linear", input_data, weight_data, bias_data)
            .await?;

        let in_shape = input.dims();
        let out_features = weight.dims()[0];
        let mut out_shape = in_shape.to_vec();
        *out_shape.last_mut().unwrap() = out_features;

        bytes_to_tensor(
            &result_bytes,
            &out_shape,
            input.dtype(),
            &self.config.result_device,
        )
    }

    /// Embedding lookup.
    pub async fn embedding(&self, indices: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let indices_vec: Vec<u32> = indices
            .flatten_all()
            .map_err(|e| EcosystemError::Candle(e.to_string()))?
            .to_vec1()
            .map_err(|e| EcosystemError::Candle(e.to_string()))?;

        let (weight_data, _) = tensor_to_bytes(weight)?;
        let vocab_size = weight.dims()[0];
        let embed_dim = weight.dims()[1];

        let result_bytes = self
            .runtime
            .gpu_embedding("embedding", indices_vec, weight_data, vocab_size, embed_dim)
            .await?;

        let mut out_shape = indices.dims().to_vec();
        out_shape.push(embed_dim);

        bytes_to_tensor(
            &result_bytes,
            &out_shape,
            weight.dtype(),
            &self.config.result_device,
        )
    }
}

/// GPU model layer abstraction.
#[derive(Debug, Clone)]
pub enum GpuLayer {
    /// Linear layer
    Linear {
        in_features: usize,
        out_features: usize,
    },
    /// Conv2d layer
    Conv2d(GpuConv2dConfig),
    /// Pooling layer
    Pooling(GpuPoolingConfig),
    /// Activation layer
    Activation(GpuActivation),
    /// Normalization layer
    Normalization(GpuNormalization),
    /// Attention layer
    Attention(GpuAttentionConfig),
    /// Dropout layer
    Dropout { p: f32 },
    /// Flatten layer
    Flatten,
}

/// GPU model builder for creating neural network architectures.
pub struct GpuModelBuilder {
    layers: Vec<GpuLayer>,
    name: String,
}

impl GpuModelBuilder {
    /// Create a new model builder.
    pub fn new(name: &str) -> Self {
        Self {
            layers: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Add a linear layer.
    pub fn linear(mut self, in_features: usize, out_features: usize) -> Self {
        self.layers.push(GpuLayer::Linear {
            in_features,
            out_features,
        });
        self
    }

    /// Add a conv2d layer.
    pub fn conv2d(mut self, config: GpuConv2dConfig) -> Self {
        self.layers.push(GpuLayer::Conv2d(config));
        self
    }

    /// Add ReLU activation.
    pub fn relu(mut self) -> Self {
        self.layers.push(GpuLayer::Activation(GpuActivation::ReLU));
        self
    }

    /// Add GELU activation.
    pub fn gelu(mut self) -> Self {
        self.layers.push(GpuLayer::Activation(GpuActivation::GELU));
        self
    }

    /// Add max pooling.
    pub fn max_pool2d(mut self, kernel_size: (usize, usize)) -> Self {
        self.layers.push(GpuLayer::Pooling(GpuPoolingConfig {
            pool_type: GpuPoolingType::Max,
            kernel_size,
            stride: kernel_size,
            padding: (0, 0),
        }));
        self
    }

    /// Add batch normalization.
    pub fn batch_norm(mut self) -> Self {
        self.layers
            .push(GpuLayer::Normalization(GpuNormalization::BatchNorm));
        self
    }

    /// Add layer normalization.
    pub fn layer_norm(mut self) -> Self {
        self.layers
            .push(GpuLayer::Normalization(GpuNormalization::LayerNorm));
        self
    }

    /// Add dropout.
    pub fn dropout(mut self, p: f32) -> Self {
        self.layers.push(GpuLayer::Dropout { p });
        self
    }

    /// Add flatten.
    pub fn flatten(mut self) -> Self {
        self.layers.push(GpuLayer::Flatten);
        self
    }

    /// Get the layers.
    pub fn layers(&self) -> &[GpuLayer] {
        &self.layers
    }

    /// Get the model name.
    pub fn name(&self) -> &str {
        &self.name
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
