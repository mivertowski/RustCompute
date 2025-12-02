//! # Batch Data Processing Pipeline
//!
//! This example demonstrates building a GPU-accelerated data processing
//! pipeline for batch workloads using RingKernel with Arrow/Polars.
//!
//! ## Use Case
//!
//! Processing large datasets where:
//! - Data arrives in batches (files, database queries, API responses)
//! - CPU preprocessing followed by GPU computation
//! - Results aggregated and exported
//!
//! ## Pipeline Architecture
//!
//! ```text
//! ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
//! │  Load   │───▶│Preprocess│───▶│  GPU    │───▶│ Export  │
//! │  Data   │    │  (CPU)   │    │ Compute │    │ Results │
//! └─────────┘    └─────────┘    └─────────┘    └─────────┘
//!                    │               │
//!                    ▼               ▼
//!              Chunking &     Persistent kernel
//!              Validation     maintains state
//! ```
//!
//! ## Run this example:
//! ```bash
//! cargo run --example batch_processor
//! ```

use ringkernel::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============ Data Types ============

/// A batch of records to process
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DataBatch {
    /// Batch identifier
    id: u64,
    /// Feature matrix (rows x cols)
    features: Vec<Vec<f32>>,
    /// Labels (if available)
    labels: Option<Vec<f32>>,
    /// Metadata
    metadata: HashMap<String, String>,
}

#[allow(dead_code)]
impl DataBatch {
    fn new(id: u64, features: Vec<Vec<f32>>) -> Self {
        Self {
            id,
            features,
            labels: None,
            metadata: HashMap::new(),
        }
    }

    fn with_labels(mut self, labels: Vec<f32>) -> Self {
        self.labels = Some(labels);
        self
    }

    fn row_count(&self) -> usize {
        self.features.len()
    }

    fn col_count(&self) -> usize {
        self.features.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Flatten features for GPU processing
    fn flatten(&self) -> Vec<f32> {
        self.features.iter().flatten().copied().collect()
    }
}

/// Processed result
#[allow(dead_code)]
#[derive(Debug)]
struct ProcessedResult {
    batch_id: u64,
    predictions: Vec<f32>,
    confidence: Vec<f32>,
    processing_time: Duration,
}

// ============ Pipeline Components ============

/// Configuration for the processing pipeline
#[derive(Debug, Clone)]
struct PipelineConfig {
    /// Maximum rows per GPU batch
    max_batch_size: usize,
    /// Number of parallel GPU streams
    num_streams: usize,
    /// Enable result caching
    enable_caching: bool,
    /// Timeout for GPU operations
    gpu_timeout: Duration,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 10000,
            num_streams: 4,
            enable_caching: true,
            gpu_timeout: Duration::from_secs(30),
        }
    }
}

/// Statistics for the pipeline
#[derive(Debug, Default)]
struct PipelineStats {
    batches_processed: u64,
    total_rows: u64,
    total_time_ms: f64,
    gpu_time_ms: f64,
    preprocessing_time_ms: f64,
}

impl PipelineStats {
    fn throughput_rows_per_sec(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            (self.total_rows as f64 / self.total_time_ms) * 1000.0
        } else {
            0.0
        }
    }

    fn gpu_utilization(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            (self.gpu_time_ms / self.total_time_ms) * 100.0
        } else {
            0.0
        }
    }
}

/// The main processing pipeline
#[allow(dead_code)]
struct DataPipeline {
    runtime: Arc<RingKernel>,
    compute_kernel: KernelHandle,
    config: PipelineConfig,
    stats: PipelineStats,
}

impl DataPipeline {
    async fn new(config: PipelineConfig) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let runtime = Arc::new(
            RingKernel::builder()
                .backend(Backend::Cpu)
                .build()
                .await?,
        );

        // Queue capacity must be a power of 2
        let queue_capacity = (config.max_batch_size * 2).next_power_of_two();
        let options = LaunchOptions::default()
            .with_queue_capacity(queue_capacity)
            .without_auto_activate();

        let compute_kernel = runtime.launch("batch_compute", options).await?;
        compute_kernel.activate().await?;

        Ok(Self {
            runtime,
            compute_kernel,
            config,
            stats: PipelineStats::default(),
        })
    }

    /// Process a single batch
    async fn process_batch(&mut self, batch: DataBatch) -> std::result::Result<ProcessedResult, Box<dyn std::error::Error>> {
        let start = Instant::now();

        // Step 1: Preprocess on CPU
        let preprocess_start = Instant::now();
        let preprocessed = self.preprocess(&batch)?;
        self.stats.preprocessing_time_ms += preprocess_start.elapsed().as_secs_f64() * 1000.0;

        // Step 2: Send to GPU for computation
        let gpu_start = Instant::now();

        // In real implementation:
        // self.compute_kernel.send(preprocessed).await?;
        // let gpu_result = self.compute_kernel.receive(self.config.gpu_timeout).await?;

        // Simulated GPU computation
        let (predictions, confidence) = self.simulate_gpu_compute(&preprocessed);
        self.stats.gpu_time_ms += gpu_start.elapsed().as_secs_f64() * 1000.0;

        let processing_time = start.elapsed();
        self.stats.batches_processed += 1;
        self.stats.total_rows += batch.row_count() as u64;
        self.stats.total_time_ms += processing_time.as_secs_f64() * 1000.0;

        Ok(ProcessedResult {
            batch_id: batch.id,
            predictions,
            confidence,
            processing_time,
        })
    }

    /// Process multiple batches with parallel streams
    async fn process_batches(&mut self, batches: Vec<DataBatch>) -> Vec<ProcessedResult> {
        let mut results = Vec::with_capacity(batches.len());

        // Process in chunks matching stream count
        for batch in batches {
            match self.process_batch(batch).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!("Error processing batch: {}", e);
                }
            }
        }

        results
    }

    /// Preprocess data on CPU
    fn preprocess(&self, batch: &DataBatch) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error>> {
        let flat = batch.flatten();

        // Normalize features (z-score normalization)
        let n = flat.len() as f32;
        let mean: f32 = flat.iter().sum::<f32>() / n;
        let variance: f32 = flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt().max(1e-7);

        let normalized: Vec<f32> = flat.iter().map(|x| (x - mean) / std_dev).collect();

        Ok(normalized)
    }

    /// Simulated GPU computation
    fn simulate_gpu_compute(&self, data: &[f32]) -> (Vec<f32>, Vec<f32>) {
        // Simulate predictions (in reality, this runs on GPU)
        let predictions: Vec<f32> = data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();

        // Simulate confidence scores
        let confidence: Vec<f32> = predictions
            .iter()
            .map(|p| (p - 0.5).abs() * 2.0)
            .collect();

        (predictions, confidence)
    }

    fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    async fn shutdown(self) -> std::result::Result<(), Box<dyn std::error::Error>> {
        self.compute_kernel.terminate().await?;
        // Note: runtime will be dropped when Arc count goes to 0
        Ok(())
    }
}

// ============ Main ============

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("=== Batch Data Processing Pipeline Example ===\n");

    // Create pipeline with custom configuration
    let config = PipelineConfig {
        max_batch_size: 5000,
        num_streams: 4,
        enable_caching: true,
        gpu_timeout: Duration::from_secs(30),
    };

    println!("Pipeline configuration:");
    println!("  Max batch size: {}", config.max_batch_size);
    println!("  Parallel streams: {}", config.num_streams);
    println!("  Caching enabled: {}", config.enable_caching);
    println!("  GPU timeout: {:?}\n", config.gpu_timeout);

    let mut pipeline = DataPipeline::new(config).await?;

    // Generate sample data batches
    println!("Generating sample data...");
    let batches = generate_sample_batches(10, 1000, 16);
    println!("  Generated {} batches", batches.len());
    println!("  Total rows: {}", batches.iter().map(|b| b.row_count()).sum::<usize>());
    println!();

    // Process batches
    println!("Processing batches...\n");
    let start = Instant::now();

    let results = pipeline.process_batches(batches).await;

    let total_time = start.elapsed();

    // Print results summary
    println!("=== Results Summary ===\n");

    for result in &results {
        println!(
            "Batch {}: {} predictions in {:?}",
            result.batch_id,
            result.predictions.len(),
            result.processing_time
        );
    }

    // Print statistics
    println!("\n=== Pipeline Statistics ===\n");
    let stats = pipeline.stats();
    println!("  Batches processed: {}", stats.batches_processed);
    println!("  Total rows: {}", stats.total_rows);
    println!("  Total time: {:.2} ms", stats.total_time_ms);
    println!("  GPU time: {:.2} ms", stats.gpu_time_ms);
    println!("  Preprocessing time: {:.2} ms", stats.preprocessing_time_ms);
    println!("  Throughput: {:.0} rows/sec", stats.throughput_rows_per_sec());
    println!("  GPU utilization: {:.1}%", stats.gpu_utilization());

    println!("\nOverall processing time: {:?}", total_time);

    // Cleanup
    pipeline.shutdown().await?;

    println!("\n=== Example completed! ===");
    Ok(())
}

/// Generate sample data batches for testing
fn generate_sample_batches(num_batches: usize, rows_per_batch: usize, features: usize) -> Vec<DataBatch> {
    (0..num_batches)
        .map(|id| {
            let features: Vec<Vec<f32>> = (0..rows_per_batch)
                .map(|_| {
                    (0..features)
                        .map(|_| rand_f32() * 10.0 - 5.0)
                        .collect()
                })
                .collect();

            DataBatch::new(id as u64, features)
        })
        .collect()
}

/// Simple random number generator (for demo purposes)
fn rand_f32() -> f32 {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    ((seed as f32 * 0.0000001) % 1.0).abs()
}
