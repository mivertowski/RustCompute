---
layout: default
title: ML Bridges
nav_order: 19
---

# ML Framework Bridges

RingKernel provides bridges to popular ML frameworks for seamless tensor interop and model execution.

## PyTorch Bridge

Bidirectional tensor conversion between RingKernel and PyTorch.

### Configuration

```rust
use ringkernel_ecosystem::ml_bridge::{PyTorchBridge, PyTorchConfig};

let config = PyTorchConfig {
    default_device: "cuda:0".to_string(),
    enable_cuda: true,
    memory_pool_size: 1024 * 1024 * 1024, // 1GB pool
    use_pinned_memory: true,              // Faster CPU↔GPU transfers
};

let bridge = PyTorchBridge::with_config(config);
```

### Tensor Conversion

```rust
use ringkernel_ecosystem::ml_bridge::PyTorchDType;

// Raw bytes → PyTorch tensor
let tensor = bridge.to_pytorch(
    &raw_data,
    &[batch_size, channels, height, width],  // NCHW format
    PyTorchDType::Float32,
)?;

// PyTorch tensor → Raw bytes
let raw_output = bridge.from_pytorch(&output_tensor)?;

// Create tensor with gradient tracking
let trainable = bridge.to_pytorch(&data, &shape, PyTorchDType::Float32)?
    .with_grad();

// Move between devices
let gpu_tensor = tensor.to_device("cuda:0");
let cpu_tensor = tensor.to_device("cpu");
```

### Data Types

| PyTorch | RingKernel Enum | Size | Notes |
|---------|-----------------|------|-------|
| `torch.float32` | `Float32` | 4 bytes | Default for most operations |
| `torch.float64` | `Float64` | 8 bytes | High precision |
| `torch.float16` | `Float16` | 2 bytes | Mixed precision training |
| `torch.bfloat16` | `BFloat16` | 2 bytes | Better range than FP16 |
| `torch.int32` | `Int32` | 4 bytes | Integer operations |
| `torch.int64` | `Int64` | 8 bytes | Large indices |
| `torch.int8` | `Int8` | 1 byte | Quantized models |
| `torch.uint8` | `UInt8` | 1 byte | Images, masks |
| `torch.bool` | `Bool` | 1 byte | Boolean operations |

---

## ONNX Executor

Load and execute ONNX models on GPU ring kernels.

### Loading Models

```rust
use ringkernel_ecosystem::ml_bridge::{OnnxExecutor, OnnxConfig};

// Create executor
let executor = OnnxExecutor::new(&runtime)?;

// Load from file
executor.load_model("models/resnet50.onnx")?;

// Load from bytes (embedded models)
let model_bytes = include_bytes!("models/classifier.onnx");
executor.load_model_from_bytes(model_bytes)?;

// Load with configuration
let config = OnnxConfig {
    execution_provider: ExecutionProvider::CUDA,
    optimization_level: OptimizationLevel::All,
    intra_op_parallelism: 4,
    inter_op_parallelism: 2,
};
executor.load_model_with_config("model.onnx", config)?;
```

### Inference

```rust
// Prepare inputs
let image = preprocess_image(&raw_image)?;
let inputs = vec![
    ("input", image),
];

// Run inference
let outputs = executor.run(&inputs).await?;

// Get results
let logits = outputs.get("logits").unwrap();
let predictions = softmax(&logits);

// Batch inference
let batch_inputs = vec![
    vec![("input", image1)],
    vec![("input", image2)],
    vec![("input", image3)],
];
let batch_outputs = executor.run_batch(&batch_inputs).await?;
```

### Model Information

```rust
let info = executor.model_info()?;

println!("Inputs:");
for input in &info.inputs {
    println!("  {}: {:?} ({:?})", input.name, input.shape, input.dtype);
}

println!("Outputs:");
for output in &info.outputs {
    println!("  {}: {:?} ({:?})", output.name, output.shape, output.dtype);
}

println!("ONNX version: {}", info.onnx_version);
println!("Producer: {}", info.producer);
```

---

## Hugging Face Pipeline

Integration with Hugging Face Transformers for NLP and more.

### Text Classification

```rust
use ringkernel_ecosystem::ml_bridge::HuggingFacePipeline;

// Load sentiment analysis model
let classifier = HuggingFacePipeline::text_classification(
    &runtime,
    "distilbert-base-uncased-finetuned-sst-2-english",
)?;

// Single prediction
let result = classifier.run("I love this product!").await?;
println!("Label: {}, Score: {:.4}", result.label, result.score);
// Output: Label: POSITIVE, Score: 0.9987

// Batch prediction
let texts = vec![
    "Great service!",
    "Terrible experience.",
    "It was okay.",
];
let results = classifier.run_batch(&texts).await?;
```

### Text Generation

```rust
// Load GPT-2
let generator = HuggingFacePipeline::text_generation(
    &runtime,
    "gpt2-medium",
)?;

// Generate text
let generated = generator.generate(
    "Once upon a time",
    GenerationConfig {
        max_length: 100,
        temperature: 0.8,
        top_p: 0.9,
        do_sample: true,
    },
).await?;

// Streaming generation
let mut stream = generator.generate_stream("In the beginning", config);
while let Some(token) = stream.next().await {
    print!("{}", token);
}
```

### Embeddings

```rust
// Load sentence transformer
let embedder = HuggingFacePipeline::embeddings(
    &runtime,
    "sentence-transformers/all-MiniLM-L6-v2",
)?;

// Get embedding
let embedding = embedder.embed("Hello, world!").await?;
println!("Embedding dimension: {}", embedding.len());  // 384

// Batch embeddings
let sentences = vec![
    "The cat sat on the mat.",
    "Dogs are loyal companions.",
    "Machine learning is fascinating.",
];
let embeddings = embedder.embed_batch(&sentences).await?;

// Compute similarity
let similarity = cosine_similarity(&embeddings[0], &embeddings[1]);
```

### Question Answering

```rust
let qa = HuggingFacePipeline::question_answering(
    &runtime,
    "deepset/roberta-base-squad2",
)?;

let context = "RingKernel is a GPU-native persistent actor model framework for Rust. \
               It enables GPU-accelerated actor systems with persistent kernels.";

let answer = qa.answer(
    "What is RingKernel?",
    context,
).await?;

println!("Answer: {}", answer.text);
println!("Confidence: {:.4}", answer.score);
```

### Named Entity Recognition

```rust
let ner = HuggingFacePipeline::token_classification(
    &runtime,
    "dslim/bert-base-NER",
)?;

let entities = ner.extract("Apple Inc. was founded by Steve Jobs in Cupertino.").await?;

for entity in entities {
    println!("{}: {} ({:.2})", entity.word, entity.entity, entity.score);
}
// Output:
// Apple Inc.: B-ORG (0.99)
// Steve Jobs: B-PER (0.98)
// Cupertino: B-LOC (0.97)
```

---

## Performance Optimization

### Caching

```rust
// Enable model caching
let pipeline = HuggingFacePipeline::text_classification(&runtime, "bert-base")?
    .with_cache(CacheConfig {
        cache_dir: PathBuf::from("/models/cache"),
        max_size_gb: 10,
    });
```

### Batching

```rust
// Automatic batching for throughput
let config = BatchConfig {
    max_batch_size: 32,
    max_wait_time: Duration::from_millis(10),
};

let pipeline = HuggingFacePipeline::embeddings(&runtime, "model")?
    .with_batching(config);
```

### Quantization

```rust
// Load quantized model (INT8)
let pipeline = HuggingFacePipeline::text_classification(&runtime, "model")?
    .with_quantization(QuantizationConfig::Int8);

// Dynamic quantization
let pipeline = HuggingFacePipeline::text_classification(&runtime, "model")?
    .with_quantization(QuantizationConfig::Dynamic);
```

---

## Integration with Ring Kernels

### Custom GPU Processing

```rust
// Combine HuggingFace embeddings with custom GPU kernel
let embedder = HuggingFacePipeline::embeddings(&runtime, "model")?;

// Get embeddings
let embeddings = embedder.embed_batch(&texts).await?;

// Process on custom ring kernel
let kernel = runtime.launch("similarity_matrix", config).await?;
kernel.activate().await?;

let request = SimilarityRequest {
    embeddings: embeddings.clone(),
};
kernel.send(request).await?;

let result: SimilarityMatrix = kernel.receive(timeout).await?;
```

### Streaming Pipeline

```rust
// Stream text through multiple models
let classifier = HuggingFacePipeline::text_classification(&runtime, "sentiment")?;
let embedder = HuggingFacePipeline::embeddings(&runtime, "embeddings")?;
let kernel = runtime.launch("processor", config).await?;

// Process stream
let mut stream = text_stream();
while let Some(text) = stream.next().await {
    let sentiment = classifier.run(&text).await?;
    let embedding = embedder.embed(&text).await?;

    kernel.send(ProcessRequest {
        text,
        sentiment,
        embedding,
    }).await?;
}
```
