//! ML Framework Bridge Integration for RingKernel.
//!
//! This module provides bridges to external ML frameworks:
//!
//! - **PyTorch**: Export/import tensors with PyTorch via FFI
//! - **ONNX Runtime**: Load and execute ONNX models on GPU ring kernels
//! - **Hugging Face**: Integration with Transformers models
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ecosystem::ml_bridge::{OnnxExecutor, PyTorchBridge, HuggingFacePipeline};
//!
//! // Load ONNX model
//! let executor = OnnxExecutor::new(&runtime)?;
//! let output = executor.run("model.onnx", &input_tensors).await?;
//!
//! // PyTorch tensor interop
//! let bridge = PyTorchBridge::new();
//! let pt_tensor = bridge.to_pytorch(&candle_tensor)?;
//!
//! // Hugging Face inference
//! let pipeline = HuggingFacePipeline::text_classification(&runtime, "bert-base");
//! let result = pipeline.run("Hello world").await?;
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::error::{EcosystemError, Result};

// ============================================================================
// PyTorch Bridge
// ============================================================================

/// Data type for PyTorch tensor interop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyTorchDType {
    /// 32-bit float
    Float32,
    /// 64-bit float
    Float64,
    /// 16-bit float (half)
    Float16,
    /// BFloat16
    BFloat16,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
    /// 8-bit integer
    Int8,
    /// 8-bit unsigned integer
    UInt8,
    /// Boolean
    Bool,
}

impl PyTorchDType {
    /// Get the size in bytes of this dtype.
    pub fn size(&self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float64 | Self::Int64 => 8,
            Self::Float16 | Self::BFloat16 => 2,
            Self::Int8 | Self::UInt8 | Self::Bool => 1,
        }
    }

    /// Convert to string representation for PyTorch.
    pub fn to_torch_str(&self) -> &'static str {
        match self {
            Self::Float32 => "torch.float32",
            Self::Float64 => "torch.float64",
            Self::Float16 => "torch.float16",
            Self::BFloat16 => "torch.bfloat16",
            Self::Int32 => "torch.int32",
            Self::Int64 => "torch.int64",
            Self::Int8 => "torch.int8",
            Self::UInt8 => "torch.uint8",
            Self::Bool => "torch.bool",
        }
    }
}

/// PyTorch tensor representation for interop.
#[derive(Debug, Clone)]
pub struct PyTorchTensor {
    /// Raw tensor data
    pub data: Vec<u8>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: PyTorchDType,
    /// Whether tensor requires gradient
    pub requires_grad: bool,
    /// Device string (cpu, cuda:0, etc.)
    pub device: String,
}

impl PyTorchTensor {
    /// Create a new PyTorch tensor.
    pub fn new(data: Vec<u8>, shape: Vec<usize>, dtype: PyTorchDType) -> Self {
        Self {
            data,
            shape,
            dtype,
            requires_grad: false,
            device: "cpu".to_string(),
        }
    }

    /// Set requires_grad.
    pub fn with_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Set device.
    pub fn to_device(mut self, device: &str) -> Self {
        self.device = device.to_string();
        self
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size()
    }
}

/// Configuration for PyTorch bridge.
#[derive(Debug, Clone)]
pub struct PyTorchConfig {
    /// Default device for tensors
    pub default_device: String,
    /// Enable CUDA if available
    pub enable_cuda: bool,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Use pinned memory for CPU tensors
    pub use_pinned_memory: bool,
}

impl Default for PyTorchConfig {
    fn default() -> Self {
        Self {
            default_device: "cpu".to_string(),
            enable_cuda: true,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            use_pinned_memory: false,
        }
    }
}

/// Bridge for PyTorch tensor interop.
///
/// Enables bidirectional tensor conversion between RingKernel and PyTorch.
pub struct PyTorchBridge {
    config: PyTorchConfig,
    /// Cached tensor metadata
    tensor_cache: HashMap<String, PyTorchTensor>,
}

impl PyTorchBridge {
    /// Create a new PyTorch bridge.
    pub fn new() -> Self {
        Self {
            config: PyTorchConfig::default(),
            tensor_cache: HashMap::new(),
        }
    }

    /// Create with configuration.
    pub fn with_config(config: PyTorchConfig) -> Self {
        Self {
            config,
            tensor_cache: HashMap::new(),
        }
    }

    /// Convert raw bytes to PyTorch tensor format.
    pub fn to_pytorch(
        &self,
        data: &[u8],
        shape: &[usize],
        dtype: PyTorchDType,
    ) -> Result<PyTorchTensor> {
        let expected_size = shape.iter().product::<usize>() * dtype.size();
        if data.len() != expected_size {
            return Err(EcosystemError::DataConversion(format!(
                "Data size {} doesn't match expected {}",
                data.len(),
                expected_size
            )));
        }

        Ok(PyTorchTensor {
            data: data.to_vec(),
            shape: shape.to_vec(),
            dtype,
            requires_grad: false,
            device: self.config.default_device.clone(),
        })
    }

    /// Convert PyTorch tensor to raw bytes.
    pub fn from_pytorch(&self, tensor: &PyTorchTensor) -> Result<(Vec<u8>, Vec<usize>)> {
        Ok((tensor.data.clone(), tensor.shape.clone()))
    }

    /// Cache a tensor for later use.
    pub fn cache_tensor(&mut self, name: &str, tensor: PyTorchTensor) {
        self.tensor_cache.insert(name.to_string(), tensor);
    }

    /// Get a cached tensor.
    pub fn get_cached(&self, name: &str) -> Option<&PyTorchTensor> {
        self.tensor_cache.get(name)
    }

    /// Clear tensor cache.
    pub fn clear_cache(&mut self) {
        self.tensor_cache.clear();
    }

    /// Get configuration.
    pub fn config(&self) -> &PyTorchConfig {
        &self.config
    }

    /// Export tensor metadata for PyTorch loading.
    pub fn export_metadata(&self, tensor: &PyTorchTensor) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("dtype".to_string(), tensor.dtype.to_torch_str().to_string());
        metadata.insert(
            "shape".to_string(),
            format!("{:?}", tensor.shape),
        );
        metadata.insert("device".to_string(), tensor.device.clone());
        metadata.insert("requires_grad".to_string(), tensor.requires_grad.to_string());
        metadata.insert("numel".to_string(), tensor.numel().to_string());
        metadata
    }
}

impl Default for PyTorchBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ONNX Runtime Integration
// ============================================================================

/// ONNX model input specification.
#[derive(Debug, Clone)]
pub struct OnnxInputSpec {
    /// Input name
    pub name: String,
    /// Expected shape (None for dynamic dimensions)
    pub shape: Vec<Option<usize>>,
    /// Data type
    pub dtype: OnnxDType,
}

/// ONNX model output specification.
#[derive(Debug, Clone)]
pub struct OnnxOutputSpec {
    /// Output name
    pub name: String,
    /// Shape (may be dynamic)
    pub shape: Vec<Option<usize>>,
    /// Data type
    pub dtype: OnnxDType,
}

/// ONNX data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDType {
    /// 32-bit float
    Float,
    /// 64-bit float
    Double,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
    /// 8-bit integer
    Int8,
    /// 8-bit unsigned
    UInt8,
    /// 16-bit float
    Float16,
    /// String
    String,
    /// Boolean
    Bool,
}

/// ONNX model metadata.
#[derive(Debug, Clone)]
pub struct OnnxModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: u64,
    /// Producer name
    pub producer: String,
    /// Input specifications
    pub inputs: Vec<OnnxInputSpec>,
    /// Output specifications
    pub outputs: Vec<OnnxOutputSpec>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

/// ONNX inference result.
#[derive(Debug, Clone)]
pub struct OnnxOutput {
    /// Output name
    pub name: String,
    /// Output data
    pub data: Vec<u8>,
    /// Output shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: OnnxDType,
}

/// Configuration for ONNX executor.
#[derive(Debug, Clone)]
pub struct OnnxConfig {
    /// Execution provider (CPU, CUDA, TensorRT, etc.)
    pub execution_provider: OnnxExecutionProvider,
    /// Graph optimization level
    pub optimization_level: OnnxOptLevel,
    /// Number of intra-op threads
    pub intra_op_threads: usize,
    /// Number of inter-op threads
    pub inter_op_threads: usize,
    /// Enable memory pattern optimization
    pub enable_mem_pattern: bool,
    /// Enable memory arena
    pub enable_mem_arena: bool,
}

/// ONNX execution providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxExecutionProvider {
    /// CPU execution
    Cpu,
    /// CUDA execution
    Cuda,
    /// TensorRT execution
    TensorRT,
    /// ROCm execution (AMD)
    ROCm,
    /// DirectML execution (Windows)
    DirectML,
    /// CoreML execution (Apple)
    CoreML,
    /// OpenVINO execution (Intel)
    OpenVINO,
}

/// ONNX optimization levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxOptLevel {
    /// No optimization
    None,
    /// Basic optimizations
    Basic,
    /// Extended optimizations
    Extended,
    /// All optimizations
    All,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            execution_provider: OnnxExecutionProvider::Cpu,
            optimization_level: OnnxOptLevel::All,
            intra_op_threads: 0, // Use default
            inter_op_threads: 0, // Use default
            enable_mem_pattern: true,
            enable_mem_arena: true,
        }
    }
}

/// Runtime handle trait for ONNX operations.
#[async_trait::async_trait]
pub trait OnnxRuntime: Send + Sync + 'static {
    /// Load an ONNX model.
    async fn load_model(&self, path: &Path) -> Result<String>;

    /// Get model metadata.
    async fn get_metadata(&self, model_id: &str) -> Result<OnnxModelMetadata>;

    /// Run inference.
    async fn run_inference(
        &self,
        model_id: &str,
        inputs: HashMap<String, Vec<u8>>,
    ) -> Result<Vec<OnnxOutput>>;

    /// Unload a model.
    async fn unload_model(&self, model_id: &str) -> Result<()>;
}

/// ONNX model executor for GPU ring kernels.
pub struct OnnxExecutor<R> {
    runtime: Arc<R>,
    config: OnnxConfig,
    /// Loaded models
    loaded_models: HashMap<String, OnnxModelMetadata>,
}

impl<R: OnnxRuntime> OnnxExecutor<R> {
    /// Create a new ONNX executor.
    pub fn new(runtime: Arc<R>) -> Self {
        Self {
            runtime,
            config: OnnxConfig::default(),
            loaded_models: HashMap::new(),
        }
    }

    /// Create with configuration.
    pub fn with_config(runtime: Arc<R>, config: OnnxConfig) -> Self {
        Self {
            runtime,
            config,
            loaded_models: HashMap::new(),
        }
    }

    /// Load a model from file.
    pub async fn load(&mut self, path: impl AsRef<Path>) -> Result<String> {
        let model_id = self.runtime.load_model(path.as_ref()).await?;
        let metadata = self.runtime.get_metadata(&model_id).await?;
        self.loaded_models.insert(model_id.clone(), metadata);
        Ok(model_id)
    }

    /// Run inference on a loaded model.
    pub async fn run(
        &self,
        model_id: &str,
        inputs: HashMap<String, Vec<u8>>,
    ) -> Result<Vec<OnnxOutput>> {
        if !self.loaded_models.contains_key(model_id) {
            return Err(EcosystemError::Configuration(format!(
                "Model {} not loaded",
                model_id
            )));
        }
        self.runtime.run_inference(model_id, inputs).await
    }

    /// Get model metadata.
    pub fn metadata(&self, model_id: &str) -> Option<&OnnxModelMetadata> {
        self.loaded_models.get(model_id)
    }

    /// Unload a model.
    pub async fn unload(&mut self, model_id: &str) -> Result<()> {
        self.runtime.unload_model(model_id).await?;
        self.loaded_models.remove(model_id);
        Ok(())
    }

    /// List loaded models.
    pub fn loaded_models(&self) -> Vec<&str> {
        self.loaded_models.keys().map(|s| s.as_str()).collect()
    }

    /// Get configuration.
    pub fn config(&self) -> &OnnxConfig {
        &self.config
    }
}

// ============================================================================
// Hugging Face Integration
// ============================================================================

/// Hugging Face model task types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuggingFaceTask {
    /// Text classification
    TextClassification,
    /// Token classification (NER)
    TokenClassification,
    /// Question answering
    QuestionAnswering,
    /// Text generation
    TextGeneration,
    /// Summarization
    Summarization,
    /// Translation
    Translation,
    /// Fill mask
    FillMask,
    /// Sentence similarity
    SentenceSimilarity,
    /// Feature extraction
    FeatureExtraction,
    /// Image classification
    ImageClassification,
    /// Object detection
    ObjectDetection,
    /// Image segmentation
    ImageSegmentation,
    /// Zero-shot classification
    ZeroShotClassification,
    /// Conversational
    Conversational,
}

impl HuggingFaceTask {
    /// Get the pipeline task name.
    pub fn task_name(&self) -> &'static str {
        match self {
            Self::TextClassification => "text-classification",
            Self::TokenClassification => "token-classification",
            Self::QuestionAnswering => "question-answering",
            Self::TextGeneration => "text-generation",
            Self::Summarization => "summarization",
            Self::Translation => "translation",
            Self::FillMask => "fill-mask",
            Self::SentenceSimilarity => "sentence-similarity",
            Self::FeatureExtraction => "feature-extraction",
            Self::ImageClassification => "image-classification",
            Self::ObjectDetection => "object-detection",
            Self::ImageSegmentation => "image-segmentation",
            Self::ZeroShotClassification => "zero-shot-classification",
            Self::Conversational => "conversational",
        }
    }
}

/// Hugging Face model specification.
#[derive(Debug, Clone)]
pub struct HuggingFaceModel {
    /// Model ID (e.g., "bert-base-uncased")
    pub model_id: String,
    /// Model revision (commit hash or branch)
    pub revision: Option<String>,
    /// Task type
    pub task: HuggingFaceTask,
    /// Whether to use GPU
    pub use_gpu: bool,
    /// Model configuration overrides
    pub config: HashMap<String, String>,
}

impl HuggingFaceModel {
    /// Create a new model specification.
    pub fn new(model_id: &str, task: HuggingFaceTask) -> Self {
        Self {
            model_id: model_id.to_string(),
            revision: None,
            task,
            use_gpu: true,
            config: HashMap::new(),
        }
    }

    /// Set revision.
    pub fn revision(mut self, rev: &str) -> Self {
        self.revision = Some(rev.to_string());
        self
    }

    /// Disable GPU.
    pub fn cpu_only(mut self) -> Self {
        self.use_gpu = false;
        self
    }

    /// Add configuration.
    pub fn with_config(mut self, key: &str, value: &str) -> Self {
        self.config.insert(key.to_string(), value.to_string());
        self
    }
}

/// Configuration for Hugging Face pipeline.
#[derive(Debug, Clone)]
pub struct HuggingFaceConfig {
    /// Cache directory for models
    pub cache_dir: Option<PathBuf>,
    /// Maximum sequence length
    pub max_length: usize,
    /// Batch size
    pub batch_size: usize,
    /// Inference timeout
    pub timeout: Duration,
    /// Use FP16 precision
    pub use_fp16: bool,
    /// Quantization config
    pub quantization: Option<HuggingFaceQuantization>,
}

/// Quantization options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuggingFaceQuantization {
    /// INT8 quantization
    Int8,
    /// INT4 quantization
    Int4,
    /// FP8 quantization
    Fp8,
    /// GPTQ quantization
    Gptq,
    /// AWQ quantization
    Awq,
}

impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            cache_dir: None,
            max_length: 512,
            batch_size: 1,
            timeout: Duration::from_secs(60),
            use_fp16: true,
            quantization: None,
        }
    }
}

/// Text classification result.
#[derive(Debug, Clone)]
pub struct TextClassificationResult {
    /// Predicted label
    pub label: String,
    /// Confidence score
    pub score: f32,
}

/// Token classification result.
#[derive(Debug, Clone)]
pub struct TokenClassificationResult {
    /// Token text
    pub word: String,
    /// Entity type
    pub entity: String,
    /// Confidence score
    pub score: f32,
    /// Start character index
    pub start: usize,
    /// End character index
    pub end: usize,
}

/// Question answering result.
#[derive(Debug, Clone)]
pub struct QuestionAnsweringResult {
    /// Answer text
    pub answer: String,
    /// Confidence score
    pub score: f32,
    /// Start character index
    pub start: usize,
    /// End character index
    pub end: usize,
}

/// Text generation result.
#[derive(Debug, Clone)]
pub struct TextGenerationResult {
    /// Generated text
    pub generated_text: String,
}

/// Feature extraction result.
#[derive(Debug, Clone)]
pub struct FeatureExtractionResult {
    /// Embedding vector
    pub embeddings: Vec<f32>,
    /// Shape of embeddings
    pub shape: Vec<usize>,
}

/// Runtime handle trait for Hugging Face operations.
#[async_trait::async_trait]
pub trait HuggingFaceRuntime: Send + Sync + 'static {
    /// Load a model.
    async fn load_model(&self, model: &HuggingFaceModel) -> Result<String>;

    /// Run text classification.
    async fn text_classification(
        &self,
        model_id: &str,
        texts: &[&str],
    ) -> Result<Vec<TextClassificationResult>>;

    /// Run token classification.
    async fn token_classification(
        &self,
        model_id: &str,
        text: &str,
    ) -> Result<Vec<TokenClassificationResult>>;

    /// Run question answering.
    async fn question_answering(
        &self,
        model_id: &str,
        question: &str,
        context: &str,
    ) -> Result<QuestionAnsweringResult>;

    /// Run text generation.
    async fn text_generation(
        &self,
        model_id: &str,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<TextGenerationResult>;

    /// Run feature extraction.
    async fn feature_extraction(
        &self,
        model_id: &str,
        texts: &[&str],
    ) -> Result<Vec<FeatureExtractionResult>>;

    /// Unload a model.
    async fn unload_model(&self, model_id: &str) -> Result<()>;
}

/// Hugging Face pipeline for GPU-accelerated inference.
pub struct HuggingFacePipeline<R> {
    runtime: Arc<R>,
    config: HuggingFaceConfig,
    model: HuggingFaceModel,
    model_handle: Option<String>,
}

impl<R: HuggingFaceRuntime> HuggingFacePipeline<R> {
    /// Create a new pipeline.
    pub fn new(runtime: Arc<R>, model: HuggingFaceModel) -> Self {
        Self {
            runtime,
            config: HuggingFaceConfig::default(),
            model,
            model_handle: None,
        }
    }

    /// Create with configuration.
    pub fn with_config(mut self, config: HuggingFaceConfig) -> Self {
        self.config = config;
        self
    }

    /// Create a text classification pipeline.
    pub fn text_classification(runtime: Arc<R>, model_id: &str) -> Self {
        Self::new(
            runtime,
            HuggingFaceModel::new(model_id, HuggingFaceTask::TextClassification),
        )
    }

    /// Create a text generation pipeline.
    pub fn text_generation(runtime: Arc<R>, model_id: &str) -> Self {
        Self::new(
            runtime,
            HuggingFaceModel::new(model_id, HuggingFaceTask::TextGeneration),
        )
    }

    /// Create a feature extraction pipeline.
    pub fn feature_extraction(runtime: Arc<R>, model_id: &str) -> Self {
        Self::new(
            runtime,
            HuggingFaceModel::new(model_id, HuggingFaceTask::FeatureExtraction),
        )
    }

    /// Create a question answering pipeline.
    pub fn question_answering(runtime: Arc<R>, model_id: &str) -> Self {
        Self::new(
            runtime,
            HuggingFaceModel::new(model_id, HuggingFaceTask::QuestionAnswering),
        )
    }

    /// Load the model.
    pub async fn load(&mut self) -> Result<()> {
        let handle = self.runtime.load_model(&self.model).await?;
        self.model_handle = Some(handle);
        Ok(())
    }

    /// Ensure model is loaded.
    async fn ensure_loaded(&mut self) -> Result<&str> {
        if self.model_handle.is_none() {
            self.load().await?;
        }
        Ok(self.model_handle.as_ref().unwrap())
    }

    /// Run text classification.
    pub async fn classify(&mut self, texts: &[&str]) -> Result<Vec<TextClassificationResult>> {
        let handle = self.ensure_loaded().await?.to_string();
        self.runtime.text_classification(&handle, texts).await
    }

    /// Run text generation.
    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<TextGenerationResult> {
        let handle = self.ensure_loaded().await?.to_string();
        self.runtime.text_generation(&handle, prompt, max_tokens).await
    }

    /// Run feature extraction.
    pub async fn extract_features(&mut self, texts: &[&str]) -> Result<Vec<FeatureExtractionResult>> {
        let handle = self.ensure_loaded().await?.to_string();
        self.runtime.feature_extraction(&handle, texts).await
    }

    /// Run question answering.
    pub async fn answer(&mut self, question: &str, context: &str) -> Result<QuestionAnsweringResult> {
        let handle = self.ensure_loaded().await?.to_string();
        self.runtime.question_answering(&handle, question, context).await
    }

    /// Get the model specification.
    pub fn model(&self) -> &HuggingFaceModel {
        &self.model
    }

    /// Get the configuration.
    pub fn config(&self) -> &HuggingFaceConfig {
        &self.config
    }

    /// Unload the model.
    pub async fn unload(&mut self) -> Result<()> {
        if let Some(handle) = self.model_handle.take() {
            self.runtime.unload_model(&handle).await?;
        }
        Ok(())
    }
}

// ============================================================================
// Common Utilities
// ============================================================================

/// Tokenizer configuration for NLP models.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Padding token ID
    pub pad_token_id: u32,
    /// Start of sequence token ID
    pub bos_token_id: u32,
    /// End of sequence token ID
    pub eos_token_id: u32,
    /// Unknown token ID
    pub unk_token_id: u32,
    /// Whether to add special tokens
    pub add_special_tokens: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522, // BERT default
            max_length: 512,
            pad_token_id: 0,
            bos_token_id: 101, // [CLS]
            eos_token_id: 102, // [SEP]
            unk_token_id: 100, // [UNK]
            add_special_tokens: true,
        }
    }
}

/// Model loading statistics.
#[derive(Debug, Clone, Default)]
pub struct ModelLoadStats {
    /// Time to load model (milliseconds)
    pub load_time_ms: u64,
    /// Model size in bytes
    pub model_size_bytes: u64,
    /// Number of parameters
    pub num_parameters: u64,
    /// Memory allocated on device
    pub device_memory_bytes: u64,
}

/// Inference statistics.
#[derive(Debug, Clone, Default)]
pub struct InferenceStats {
    /// Total inference time (milliseconds)
    pub total_time_ms: u64,
    /// Tokens processed
    pub tokens_processed: u64,
    /// Throughput (tokens/sec)
    pub throughput: f64,
    /// Latency per token (milliseconds)
    pub latency_per_token_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_dtype_size() {
        assert_eq!(PyTorchDType::Float32.size(), 4);
        assert_eq!(PyTorchDType::Float64.size(), 8);
        assert_eq!(PyTorchDType::Float16.size(), 2);
        assert_eq!(PyTorchDType::Int8.size(), 1);
    }

    #[test]
    fn test_pytorch_tensor() {
        let tensor = PyTorchTensor::new(
            vec![0; 16],
            vec![2, 2],
            PyTorchDType::Float32,
        );
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.size_bytes(), 16);
    }

    #[test]
    fn test_pytorch_bridge() {
        let bridge = PyTorchBridge::new();
        let data = vec![0u8; 16];
        let tensor = bridge.to_pytorch(&data, &[2, 2], PyTorchDType::Float32).unwrap();
        assert_eq!(tensor.shape, vec![2, 2]);
    }

    #[test]
    fn test_huggingface_model() {
        let model = HuggingFaceModel::new("bert-base-uncased", HuggingFaceTask::TextClassification)
            .revision("main")
            .cpu_only();

        assert_eq!(model.model_id, "bert-base-uncased");
        assert!(!model.use_gpu);
    }

    #[test]
    fn test_huggingface_task() {
        assert_eq!(HuggingFaceTask::TextClassification.task_name(), "text-classification");
        assert_eq!(HuggingFaceTask::TextGeneration.task_name(), "text-generation");
    }

    #[test]
    fn test_onnx_config_default() {
        let config = OnnxConfig::default();
        assert_eq!(config.execution_provider, OnnxExecutionProvider::Cpu);
        assert_eq!(config.optimization_level, OnnxOptLevel::All);
    }

    #[test]
    fn test_tokenizer_config_default() {
        let config = TokenizerConfig::default();
        assert_eq!(config.vocab_size, 30522);
        assert_eq!(config.max_length, 512);
    }
}
