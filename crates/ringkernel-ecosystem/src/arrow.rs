//! Apache Arrow integration for RingKernel.
//!
//! This module provides utilities for processing Arrow arrays and RecordBatches
//! on GPU Ring Kernels, enabling high-performance data processing pipelines.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ecosystem::arrow::{process_record_batch, ArrowKernelOps};
//!
//! let result = process_record_batch(&runtime, "kernel_id", batch).await?;
//! ```

use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use std::time::Duration;

use crate::error::{EcosystemError, Result};

/// Configuration for Arrow processing.
#[derive(Debug, Clone)]
pub struct ArrowConfig {
    /// Default timeout for GPU operations.
    pub timeout: Duration,
    /// Maximum batch size for processing.
    pub max_batch_size: usize,
    /// Enable automatic chunking for large batches.
    pub auto_chunk: bool,
    /// Chunk size when auto-chunking.
    pub chunk_size: usize,
}

impl Default for ArrowConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_batch_size: 1_000_000,
            auto_chunk: true,
            chunk_size: 100_000,
        }
    }
}

/// Runtime handle trait for Arrow integration.
#[async_trait::async_trait]
pub trait RuntimeHandle: Send + Sync + 'static {
    /// Send array data to a kernel.
    async fn send_array(&self, kernel_id: &str, data: Vec<u8>) -> Result<()>;

    /// Receive array data from a kernel with timeout.
    async fn receive_array(&self, kernel_id: &str, timeout: Duration) -> Result<Vec<u8>>;
}

/// Extension trait for GPU operations on Arrow arrays.
#[async_trait::async_trait]
pub trait ArrowKernelOps {
    /// Process array through a GPU kernel.
    async fn gpu_process<R: RuntimeHandle>(
        &self,
        runtime: &R,
        kernel_id: &str,
        config: &ArrowConfig,
    ) -> Result<ArrayRef>;
}

#[async_trait::async_trait]
impl ArrowKernelOps for Float32Array {
    async fn gpu_process<R: RuntimeHandle>(
        &self,
        runtime: &R,
        kernel_id: &str,
        config: &ArrowConfig,
    ) -> Result<ArrayRef> {
        // Extract values as bytes
        let values = self.values();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Send to kernel
        runtime.send_array(kernel_id, bytes).await?;

        // Receive result
        let result_bytes = runtime.receive_array(kernel_id, config.timeout).await?;

        // Reconstruct array
        let result_values: Vec<f32> = result_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(Arc::new(Float32Array::from(result_values)))
    }
}

#[async_trait::async_trait]
impl ArrowKernelOps for Float64Array {
    async fn gpu_process<R: RuntimeHandle>(
        &self,
        runtime: &R,
        kernel_id: &str,
        config: &ArrowConfig,
    ) -> Result<ArrayRef> {
        let values = self.values();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        runtime.send_array(kernel_id, bytes).await?;
        let result_bytes = runtime.receive_array(kernel_id, config.timeout).await?;

        let result_values: Vec<f64> = result_bytes
            .chunks_exact(8)
            .map(|chunk| {
                f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();

        Ok(Arc::new(Float64Array::from(result_values)))
    }
}

/// Process a RecordBatch through a GPU kernel.
///
/// This function sends each column to the kernel for processing
/// and reconstructs the result as a new RecordBatch.
pub async fn process_record_batch<R: RuntimeHandle>(
    runtime: &R,
    kernel_id: &str,
    batch: &RecordBatch,
    config: &ArrowConfig,
) -> Result<RecordBatch> {
    if batch.num_rows() > config.max_batch_size {
        return Err(EcosystemError::DataConversion(format!(
            "Batch size {} exceeds maximum {}",
            batch.num_rows(),
            config.max_batch_size
        )));
    }

    let mut result_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());

    for col in batch.columns() {
        let result_col = process_array(runtime, kernel_id, col, config).await?;
        result_columns.push(result_col);
    }

    RecordBatch::try_new(batch.schema(), result_columns)
        .map_err(|e| EcosystemError::Arrow(e.to_string()))
}

/// Process a single Arrow array through a GPU kernel.
pub async fn process_array<R: RuntimeHandle>(
    runtime: &R,
    kernel_id: &str,
    array: &ArrayRef,
    config: &ArrowConfig,
) -> Result<ArrayRef> {
    match array.data_type() {
        DataType::Float32 => {
            let typed = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| EcosystemError::DataConversion("Invalid Float32 array".into()))?;
            typed.gpu_process(runtime, kernel_id, config).await
        }
        DataType::Float64 => {
            let typed = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| EcosystemError::DataConversion("Invalid Float64 array".into()))?;
            typed.gpu_process(runtime, kernel_id, config).await
        }
        DataType::Int32 => {
            let typed = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| EcosystemError::DataConversion("Invalid Int32 array".into()))?;
            process_int32_array(runtime, kernel_id, typed, config).await
        }
        DataType::Int64 => {
            let typed = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| EcosystemError::DataConversion("Invalid Int64 array".into()))?;
            process_int64_array(runtime, kernel_id, typed, config).await
        }
        dt => Err(EcosystemError::DataConversion(format!(
            "Unsupported data type: {:?}",
            dt
        ))),
    }
}

async fn process_int32_array<R: RuntimeHandle>(
    runtime: &R,
    kernel_id: &str,
    array: &Int32Array,
    config: &ArrowConfig,
) -> Result<ArrayRef> {
    let values = array.values();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    runtime.send_array(kernel_id, bytes).await?;
    let result_bytes = runtime.receive_array(kernel_id, config.timeout).await?;

    let result_values: Vec<i32> = result_bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(Arc::new(Int32Array::from(result_values)))
}

async fn process_int64_array<R: RuntimeHandle>(
    runtime: &R,
    kernel_id: &str,
    array: &Int64Array,
    config: &ArrowConfig,
) -> Result<ArrayRef> {
    let values = array.values();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    runtime.send_array(kernel_id, bytes).await?;
    let result_bytes = runtime.receive_array(kernel_id, config.timeout).await?;

    let result_values: Vec<i64> = result_bytes
        .chunks_exact(8)
        .map(|chunk| {
            i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect();

    Ok(Arc::new(Int64Array::from(result_values)))
}

/// Chunk a RecordBatch into smaller batches.
pub fn chunk_record_batch(batch: &RecordBatch, chunk_size: usize) -> Vec<RecordBatch> {
    let num_rows = batch.num_rows();
    if num_rows <= chunk_size {
        return vec![batch.clone()];
    }

    let mut chunks = Vec::new();
    let mut offset = 0;

    while offset < num_rows {
        let length = std::cmp::min(chunk_size, num_rows - offset);
        let chunk = batch.slice(offset, length);
        chunks.push(chunk);
        offset += length;
    }

    chunks
}

/// Concatenate multiple RecordBatches into one.
pub fn concat_record_batches(batches: &[RecordBatch]) -> Result<RecordBatch> {
    if batches.is_empty() {
        return Err(EcosystemError::DataConversion(
            "No batches to concatenate".into(),
        ));
    }

    let schema = batches[0].schema();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

    for i in 0..schema.fields().len() {
        let arrays: Vec<&dyn Array> = batches.iter().map(|b| b.column(i).as_ref()).collect();

        let concatenated =
            arrow::compute::concat(&arrays).map_err(|e| EcosystemError::Arrow(e.to_string()))?;
        columns.push(concatenated);
    }

    RecordBatch::try_new(schema, columns).map_err(|e| EcosystemError::Arrow(e.to_string()))
}

/// Builder for GPU-accelerated Arrow processing pipelines.
pub struct ArrowPipelineBuilder<R> {
    runtime: Arc<R>,
    config: ArrowConfig,
    operations: Vec<String>,
}

impl<R: RuntimeHandle> ArrowPipelineBuilder<R> {
    /// Create a new pipeline builder.
    pub fn new(runtime: Arc<R>) -> Self {
        Self {
            runtime,
            config: ArrowConfig::default(),
            operations: Vec::new(),
        }
    }

    /// Set the configuration.
    pub fn config(mut self, config: ArrowConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a kernel operation to the pipeline.
    pub fn operation(mut self, kernel_id: &str) -> Self {
        self.operations.push(kernel_id.to_string());
        self
    }

    /// Execute the pipeline on a RecordBatch.
    pub async fn execute(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let mut current = batch;

        for kernel_id in &self.operations {
            current =
                process_record_batch(&*self.runtime, kernel_id, &current, &self.config).await?;
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
        async fn send_array(&self, _kernel_id: &str, _data: Vec<u8>) -> Result<()> {
            Ok(())
        }

        async fn receive_array(&self, _kernel_id: &str, _timeout: Duration) -> Result<Vec<u8>> {
            // Return doubled values for testing
            Ok(vec![0, 0, 128, 64, 0, 0, 0, 65]) // [4.0f32, 8.0f32]
        }
    }

    #[test]
    fn test_arrow_config_default() {
        let config = ArrowConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.auto_chunk);
        assert_eq!(config.chunk_size, 100_000);
    }

    #[test]
    fn test_chunk_record_batch() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, false)]));
        let array = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();

        let chunks = chunk_record_batch(&batch, 2);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].num_rows(), 2);
        assert_eq!(chunks[1].num_rows(), 2);
        assert_eq!(chunks[2].num_rows(), 1);
    }

    #[test]
    fn test_pipeline_builder() {
        let runtime = Arc::new(MockRuntime);
        let builder = ArrowPipelineBuilder::new(runtime)
            .operation("kernel1")
            .operation("kernel2");

        assert_eq!(builder.operations.len(), 2);
    }
}
