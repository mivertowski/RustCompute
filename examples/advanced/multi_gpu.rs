//! # Multi-GPU Coordination
//!
//! This example demonstrates coordinating multiple GPUs for parallel
//! processing using RingKernel's multi-GPU features.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                   Multi-GPU Coordination                            │
//! │                                                                     │
//! │      ┌─────────────────────────────────────────────────────┐       │
//! │      │              GPU Coordinator                         │       │
//! │      │   • Load balancing    • Fault tolerance             │       │
//! │      │   • Work distribution • Result aggregation          │       │
//! │      └─────────────────────────────────────────────────────┘       │
//! │                              │                                      │
//! │          ┌───────────────────┼───────────────────┐                 │
//! │          ▼                   ▼                   ▼                 │
//! │    ┌──────────┐       ┌──────────┐       ┌──────────┐             │
//! │    │  GPU 0   │       │  GPU 1   │       │  GPU 2   │             │
//! │    │ RTX 4090 │       │ RTX 4090 │       │ RTX 4090 │             │
//! │    │  24 GB   │       │  24 GB   │       │  24 GB   │             │
//! │    └──────────┘       └──────────┘       └──────────┘             │
//! │         │                   │                   │                  │
//! │    ┌──────────┐       ┌──────────┐       ┌──────────┐             │
//! │    │ Kernel A │       │ Kernel B │       │ Kernel C │             │
//! │    │ Batch 1-3│       │ Batch 4-6│       │ Batch 7-9│             │
//! │    └──────────┘       └──────────┘       └──────────┘             │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Use Cases
//!
//! - **Large Batch Processing**: Distribute work across GPUs
//! - **Model Parallelism**: Split large models across GPUs
//! - **Data Parallelism**: Same model, different data on each GPU
//! - **Fault Tolerance**: Redistribute work if GPU fails
//!
//! ## Run this example:
//! ```bash
//! cargo run --example multi_gpu
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RingKernel Multi-GPU Coordination ===\n");

    // ====== GPU Discovery ======
    println!("=== GPU Discovery ===\n");

    let gpu_manager = GpuManager::new();
    let devices = gpu_manager.discover_devices();

    println!("Discovered {} GPU devices:", devices.len());
    for device in &devices {
        println!("  GPU {}: {}", device.id, device.name);
        println!("    Memory: {} GB", device.memory_gb);
        println!("    Compute capability: {}", device.compute_capability);
        println!("    Available: {}", device.available);
    }

    // ====== Load Balancing Strategies ======
    println!("\n=== Load Balancing Strategies ===\n");

    demonstrate_load_balancing(&devices);

    // ====== Work Distribution ======
    println!("\n=== Work Distribution ===\n");

    let coordinator = GpuCoordinator::new(devices.clone());

    // Create a batch of work items
    let work_items: Vec<WorkItem> = (0..100)
        .map(|i| WorkItem {
            id: format!("work-{:03}", i),
            priority: if i % 10 == 0 { Priority::High } else { Priority::Normal },
            data_size_mb: 10 + (i % 50) as u64,
            estimated_time_ms: 100 + (i % 200) as u64,
        })
        .collect();

    println!("Distributing {} work items across {} GPUs", work_items.len(), devices.len());

    let distribution = coordinator.distribute_work(&work_items);

    for (gpu_id, items) in &distribution {
        let total_size: u64 = items.iter().map(|w| w.data_size_mb).sum();
        let total_time: u64 = items.iter().map(|w| w.estimated_time_ms).sum();
        println!(
            "  GPU {}: {} items, {} MB total, ~{} ms estimated",
            gpu_id,
            items.len(),
            total_size,
            total_time
        );
    }

    // ====== Parallel Execution ======
    println!("\n=== Parallel Execution ===\n");

    let start = Instant::now();
    let results = coordinator.execute_parallel(&distribution).await?;
    let total_time = start.elapsed();

    println!("Parallel execution completed in {:?}", total_time);
    println!("Results summary:");
    println!("  Total items processed: {}", results.items_processed);
    println!("  Success rate: {:.1}%", results.success_rate * 100.0);
    println!("  Avg latency: {:.2} ms", results.avg_latency_ms);

    // ====== Data Parallelism ======
    println!("\n=== Data Parallelism Demo ===\n");

    demonstrate_data_parallelism(&coordinator).await;

    // ====== Model Parallelism ======
    println!("\n=== Model Parallelism Demo ===\n");

    demonstrate_model_parallelism(&coordinator).await;

    // ====== Fault Tolerance ======
    println!("\n=== Fault Tolerance ===\n");

    demonstrate_fault_tolerance(&coordinator).await;

    // ====== GPU Memory Management ======
    println!("\n=== GPU Memory Management ===\n");

    demonstrate_memory_management(&coordinator);

    // ====== Performance Scaling ======
    println!("\n=== Performance Scaling ===\n");

    demonstrate_scaling(&coordinator).await;

    println!("\n=== Example completed! ===");
    Ok(())
}

// ============ Types ============

#[derive(Clone)]
struct GpuDevice {
    id: usize,
    name: String,
    memory_gb: u64,
    compute_capability: String,
    available: bool,
}

struct GpuManager;

impl GpuManager {
    fn new() -> Self {
        Self
    }

    fn discover_devices(&self) -> Vec<GpuDevice> {
        // Simulate discovering 3 GPUs
        vec![
            GpuDevice {
                id: 0,
                name: "NVIDIA RTX 4090".to_string(),
                memory_gb: 24,
                compute_capability: "8.9".to_string(),
                available: true,
            },
            GpuDevice {
                id: 1,
                name: "NVIDIA RTX 4090".to_string(),
                memory_gb: 24,
                compute_capability: "8.9".to_string(),
                available: true,
            },
            GpuDevice {
                id: 2,
                name: "NVIDIA RTX 4080".to_string(),
                memory_gb: 16,
                compute_capability: "8.9".to_string(),
                available: true,
            },
        ]
    }
}

#[derive(Clone)]
struct WorkItem {
    id: String,
    priority: Priority,
    data_size_mb: u64,
    estimated_time_ms: u64,
}

#[derive(Clone, PartialEq)]
enum Priority {
    High,
    Normal,
    Low,
}

struct GpuCoordinator {
    devices: Vec<GpuDevice>,
    stats: Arc<CoordinatorStats>,
}

struct CoordinatorStats {
    items_processed: AtomicU64,
    items_failed: AtomicU64,
    total_latency_us: AtomicU64,
}

struct ExecutionResults {
    items_processed: u64,
    success_rate: f64,
    avg_latency_ms: f64,
}

impl GpuCoordinator {
    fn new(devices: Vec<GpuDevice>) -> Self {
        Self {
            devices,
            stats: Arc::new(CoordinatorStats {
                items_processed: AtomicU64::new(0),
                items_failed: AtomicU64::new(0),
                total_latency_us: AtomicU64::new(0),
            }),
        }
    }

    fn distribute_work(&self, work_items: &[WorkItem]) -> HashMap<usize, Vec<WorkItem>> {
        let mut distribution: HashMap<usize, Vec<WorkItem>> = HashMap::new();
        let mut gpu_loads: Vec<u64> = vec![0; self.devices.len()];

        // Sort by priority (high first) then by estimated time (largest first)
        let mut sorted_items = work_items.to_vec();
        sorted_items.sort_by(|a, b| {
            if a.priority == Priority::High && b.priority != Priority::High {
                std::cmp::Ordering::Less
            } else if b.priority == Priority::High && a.priority != Priority::High {
                std::cmp::Ordering::Greater
            } else {
                b.estimated_time_ms.cmp(&a.estimated_time_ms)
            }
        });

        // Distribute using least-loaded strategy
        for item in sorted_items {
            // Find GPU with least load
            let min_load_gpu = gpu_loads
                .iter()
                .enumerate()
                .min_by_key(|(_, load)| *load)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            gpu_loads[min_load_gpu] += item.estimated_time_ms;

            distribution
                .entry(min_load_gpu)
                .or_default()
                .push(item);
        }

        distribution
    }

    async fn execute_parallel(
        &self,
        distribution: &HashMap<usize, Vec<WorkItem>>,
    ) -> Result<ExecutionResults, Box<dyn std::error::Error>> {
        let mut handles = vec![];

        for (gpu_id, items) in distribution {
            let gpu_id = *gpu_id;
            let items = items.clone();
            let stats = self.stats.clone();

            let handle = tokio::spawn(async move {
                for item in items {
                    // Simulate processing
                    tokio::time::sleep(Duration::from_micros(item.estimated_time_ms * 10)).await;
                    stats.items_processed.fetch_add(1, Ordering::Relaxed);
                    stats.total_latency_us.fetch_add(
                        item.estimated_time_ms * 1000,
                        Ordering::Relaxed,
                    );
                }
                gpu_id
            });
            handles.push(handle);
        }

        // Wait for all GPUs to complete
        for handle in handles {
            let _ = handle.await;
        }

        let processed = self.stats.items_processed.load(Ordering::Relaxed);
        let failed = self.stats.items_failed.load(Ordering::Relaxed);
        let total_latency = self.stats.total_latency_us.load(Ordering::Relaxed);

        Ok(ExecutionResults {
            items_processed: processed,
            success_rate: if processed + failed > 0 {
                processed as f64 / (processed + failed) as f64
            } else {
                1.0
            },
            avg_latency_ms: if processed > 0 {
                (total_latency as f64 / processed as f64) / 1000.0
            } else {
                0.0
            },
        })
    }
}

fn demonstrate_load_balancing(devices: &[GpuDevice]) {
    println!("Available load balancing strategies:\n");

    println!("1. Round Robin:");
    println!("   - Simple, even distribution");
    println!("   - Best for homogeneous GPUs with uniform work");
    println!();

    println!("2. Least Loaded:");
    println!("   - Track GPU utilization");
    println!("   - Assign to GPU with lowest current load");
    println!("   - Best for variable workloads");
    println!();

    println!("3. Weighted:");
    println!("   - Account for GPU performance differences");
    println!("   - Faster GPUs get more work");
    println!();

    // Calculate weights based on memory
    let total_memory: u64 = devices.iter().map(|d| d.memory_gb).sum();
    println!("   Weighted distribution for current GPUs:");
    for device in devices {
        let weight = device.memory_gb as f64 / total_memory as f64 * 100.0;
        println!("     GPU {}: {:.1}% of work", device.id, weight);
    }

    println!();
    println!("4. Affinity-based:");
    println!("   - Assign work to specific GPUs based on data locality");
    println!("   - Minimize data transfers");
}

async fn demonstrate_data_parallelism(coordinator: &GpuCoordinator) {
    println!("Data parallelism: Same model on all GPUs, different data\n");

    let batch_size = 128;
    let num_gpus = coordinator.devices.len();
    let per_gpu_batch = batch_size / num_gpus;

    println!("Configuration:");
    println!("  Total batch size: {}", batch_size);
    println!("  GPUs: {}", num_gpus);
    println!("  Per-GPU batch: {}", per_gpu_batch);
    println!();

    println!("Execution flow:");
    println!("  1. Split batch into {} chunks", num_gpus);
    println!("  2. Each GPU processes its chunk in parallel");
    println!("  3. Gather gradients from all GPUs");
    println!("  4. Average gradients (all-reduce)");
    println!("  5. Update model weights");
    println!();

    // Simulate data parallel training step
    let start = Instant::now();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let elapsed = start.elapsed();

    println!("Training step completed in {:?}", elapsed);
    println!("Effective throughput: {} samples/sec", batch_size * 1000 / elapsed.as_millis() as usize);
}

async fn demonstrate_model_parallelism(coordinator: &GpuCoordinator) {
    println!("Model parallelism: Split large model across GPUs\n");

    let layers_per_gpu = vec![
        ("GPU 0", vec!["Embedding", "Transformer 1-4"]),
        ("GPU 1", vec!["Transformer 5-8"]),
        ("GPU 2", vec!["Transformer 9-12", "Output"]),
    ];

    println!("Layer distribution:");
    for (gpu, layers) in &layers_per_gpu {
        println!("  {}: {:?}", gpu, layers);
    }
    println!();

    println!("Pipeline parallelism:");
    println!("  1. GPU 0 processes micro-batch 1");
    println!("  2. GPU 0 → GPU 1 (activation transfer)");
    println!("  3. GPU 1 processes while GPU 0 starts micro-batch 2");
    println!("  4. Continue pipeline...");
    println!();

    // Simulate pipeline
    let micro_batches = 4;
    let start = Instant::now();

    for mb in 0..micro_batches {
        tokio::time::sleep(Duration::from_millis(10)).await;
        println!("  Micro-batch {} completed", mb + 1);
    }

    let elapsed = start.elapsed();
    println!("\nPipeline throughput: {} micro-batches in {:?}", micro_batches, elapsed);
}

async fn demonstrate_fault_tolerance(coordinator: &GpuCoordinator) {
    println!("Fault tolerance strategies:\n");

    println!("1. Health Monitoring:");
    println!("   - Periodic GPU health checks");
    println!("   - Memory usage monitoring");
    println!("   - Temperature alerts");
    println!();

    println!("2. Work Redistribution:");
    println!("   - Detect GPU failure");
    println!("   - Redistribute pending work to healthy GPUs");
    println!("   - Example: GPU 1 fails, work moves to GPU 0 and 2");
    println!();

    // Simulate failure and recovery
    println!("Simulating GPU 1 failure...");
    tokio::time::sleep(Duration::from_millis(100)).await;
    println!("  [ALERT] GPU 1 health check failed");
    println!("  [ACTION] Redistributing 15 pending work items");
    println!("  [RESULT] Work migrated to GPU 0 (8 items) and GPU 2 (7 items)");
    println!();

    println!("3. Checkpointing:");
    println!("   - Save progress periodically");
    println!("   - Resume from last checkpoint on failure");
}

fn demonstrate_memory_management(coordinator: &GpuCoordinator) {
    println!("GPU Memory Management:\n");

    for device in &coordinator.devices {
        let used = (device.memory_gb as f64 * 0.6) as u64; // Simulate 60% usage
        let free = device.memory_gb - used;

        println!("GPU {}:", device.id);
        println!("  Total: {} GB", device.memory_gb);
        println!("  Used:  {} GB ({:.0}%)", used, used as f64 / device.memory_gb as f64 * 100.0);
        println!("  Free:  {} GB", free);
        println!();
    }

    println!("Memory optimization strategies:");
    println!("  1. Gradient checkpointing (trade compute for memory)");
    println!("  2. Mixed precision (FP16 reduces memory by 50%)");
    println!("  3. Memory pooling (pre-allocate, reuse buffers)");
    println!("  4. Streaming (load data in chunks)");
}

async fn demonstrate_scaling(coordinator: &GpuCoordinator) {
    println!("Scaling efficiency analysis:\n");

    let single_gpu_throughput = 1000.0; // items/sec
    let num_gpus = coordinator.devices.len();

    // Simulate different scaling efficiencies
    let scaling_scenarios = vec![
        ("Linear (ideal)", 1.0),
        ("95% efficient", 0.95),
        ("85% efficient", 0.85),
        ("70% efficient", 0.70),
    ];

    println!("Theoretical throughput with {} GPUs:", num_gpus);
    println!("  Single GPU baseline: {:.0} items/sec\n", single_gpu_throughput);

    for (scenario, efficiency) in scaling_scenarios {
        let multi_gpu_throughput = single_gpu_throughput * num_gpus as f64 * efficiency;
        let speedup = multi_gpu_throughput / single_gpu_throughput;

        println!(
            "  {}: {:.0} items/sec ({:.2}x speedup)",
            scenario, multi_gpu_throughput, speedup
        );
    }

    println!();
    println!("Factors affecting scaling efficiency:");
    println!("  - Communication overhead (PCIe, NVLink)");
    println!("  - Synchronization barriers");
    println!("  - Load imbalance");
    println!("  - Memory bandwidth saturation");
}
