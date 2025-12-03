//! # gRPC Server for Distributed GPU Computing
//!
//! This example demonstrates exposing RingKernel functionality over gRPC,
//! enabling distributed GPU computing across multiple machines.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                     Distributed GPU Cluster                         │
//! │                                                                     │
//! │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
//! │  │   Client    │────▶│ gRPC Server │────▶│ RingKernel  │          │
//! │  │ (Python/Go) │◀────│  (Tonic)    │◀────│   (GPU)     │          │
//! │  └─────────────┘     └─────────────┘     └─────────────┘          │
//! │                                                                     │
//! │  Features:                                                          │
//! │  • Protocol buffer serialization                                    │
//! │  • Bidirectional streaming                                          │
//! │  • Health checks and metrics                                        │
//! │  • Load balancing ready                                             │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Use Cases
//!
//! - **Microservices**: GPU acceleration as a service
//! - **Edge Computing**: Distribute work to GPU-equipped nodes
//! - **Multi-Language**: Python/Go clients calling Rust GPU backend
//! - **Horizontal Scaling**: Multiple GPU nodes behind load balancer
//!
//! ## Run this example:
//! ```bash
//! cargo run --example grpc_server --features "ringkernel-ecosystem/grpc"
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== RingKernel gRPC Server Example ===\n");

    // ====== Server Configuration ======
    println!("=== gRPC Server Configuration ===\n");

    let config = GrpcServerConfig {
        address: "[::1]:50051".to_string(),
        max_message_size: 16 * 1024 * 1024, // 16MB
        timeout: Duration::from_secs(30),
        enable_reflection: true,
        enable_health_check: true,
        max_concurrent_streams: 200,
    };

    println!("Server Configuration:");
    println!("  Address: {}", config.address);
    println!(
        "  Max message size: {} MB",
        config.max_message_size / (1024 * 1024)
    );
    println!("  Timeout: {:?}", config.timeout);
    println!("  Reflection enabled: {}", config.enable_reflection);
    println!("  Health checks: {}", config.enable_health_check);
    println!(
        "  Max concurrent streams: {}",
        config.max_concurrent_streams
    );

    // ====== Kernel Registry ======
    println!("\n=== Kernel Registry ===\n");

    let registry = KernelRegistry::new();

    // Register available kernels
    registry.register(
        "vector_add",
        KernelInfo {
            name: "Vector Addition".to_string(),
            description: "Add two vectors element-wise on GPU".to_string(),
            input_format: "float32[]".to_string(),
            output_format: "float32[]".to_string(),
        },
    );

    registry.register(
        "matrix_multiply",
        KernelInfo {
            name: "Matrix Multiplication".to_string(),
            description: "Multiply two matrices on GPU".to_string(),
            input_format: "float32[], dims".to_string(),
            output_format: "float32[]".to_string(),
        },
    );

    registry.register(
        "neural_inference",
        KernelInfo {
            name: "Neural Network Inference".to_string(),
            description: "Run neural network forward pass on GPU".to_string(),
            input_format: "float32[], model_id".to_string(),
            output_format: "float32[]".to_string(),
        },
    );

    println!("Registered kernels:");
    for (id, info) in registry.list_kernels() {
        println!("  {} - {}", id, info.name);
        println!("    {}", info.description);
    }

    // ====== Request Processing ======
    println!("\n=== Request Processing Demo ===\n");

    let server = GpuGrpcServer::new(config.clone(), registry.clone());

    // Simulate incoming requests
    let requests = vec![
        GrpcRequest {
            id: "req-001".to_string(),
            kernel_id: "vector_add".to_string(),
            payload: encode_vector(&[1.0, 2.0, 3.0, 4.0]),
            metadata: HashMap::from([
                ("client".to_string(), "python-sdk".to_string()),
                ("priority".to_string(), "high".to_string()),
            ]),
        },
        GrpcRequest {
            id: "req-002".to_string(),
            kernel_id: "matrix_multiply".to_string(),
            payload: encode_matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2),
            metadata: HashMap::from([("client".to_string(), "go-sdk".to_string())]),
        },
        GrpcRequest {
            id: "req-003".to_string(),
            kernel_id: "neural_inference".to_string(),
            payload: encode_vector(&[0.5; 784]),
            metadata: HashMap::from([("model_id".to_string(), "mnist-v1".to_string())]),
        },
    ];

    for request in requests {
        println!("Processing request: {}", request.id);
        println!("  Kernel: {}", request.kernel_id);
        println!("  Payload size: {} bytes", request.payload.len());

        let response = server.process(request).await?;

        println!("  Response:");
        println!("    Status: {:?}", response.status);
        println!("    Latency: {:?}", response.latency);
        println!("    Output size: {} bytes", response.payload.len());
        println!();
    }

    // ====== Health Checks ======
    println!("=== Health Check System ===\n");

    let health_status = server.health_check().await;

    println!("Server Health:");
    println!("  Status: {:?}", health_status.status);
    println!("  Active connections: {}", health_status.active_connections);
    println!("  Available kernels: {}", health_status.available_kernels);
    println!("  GPU memory available: {} MB", health_status.gpu_memory_mb);

    // ====== Metrics ======
    println!("\n=== Server Metrics ===\n");

    let metrics = server.get_metrics();

    println!("Request Statistics:");
    println!("  Total requests: {}", metrics.total_requests);
    println!("  Successful: {}", metrics.successful_requests);
    println!("  Failed: {}", metrics.failed_requests);
    println!("  Avg latency: {:.2} ms", metrics.avg_latency_ms);
    println!("  Bytes received: {} KB", metrics.bytes_received / 1024);
    println!("  Bytes sent: {} KB", metrics.bytes_sent / 1024);

    // ====== Streaming Demo ======
    println!("\n=== Streaming RPC Demo ===\n");

    println!("Bidirectional streaming for large datasets:");
    println!("  1. Client streams batches of data");
    println!("  2. Server processes each batch on GPU");
    println!("  3. Server streams results back");
    println!();

    demonstrate_streaming(&server).await;

    // ====== Load Balancing Ready ======
    println!("\n=== Load Balancing Configuration ===\n");

    println!("For production deployment with multiple GPU nodes:\n");
    println!("```yaml");
    println!("# Kubernetes gRPC Load Balancer");
    println!("apiVersion: v1");
    println!("kind: Service");
    println!("metadata:");
    println!("  name: ringkernel-grpc");
    println!("spec:");
    println!("  type: LoadBalancer");
    println!("  ports:");
    println!("  - port: 50051");
    println!("    targetPort: 50051");
    println!("  selector:");
    println!("    app: ringkernel-gpu");
    println!("```");

    println!("\n=== Example completed! ===");
    Ok(())
}

// ============ gRPC Server Types ============

#[derive(Clone)]
struct GrpcServerConfig {
    address: String,
    max_message_size: usize,
    timeout: Duration,
    enable_reflection: bool,
    enable_health_check: bool,
    max_concurrent_streams: u32,
}

#[allow(dead_code)]
#[derive(Clone)]
struct KernelInfo {
    name: String,
    description: String,
    input_format: String,
    output_format: String,
}

#[derive(Clone)]
struct KernelRegistry {
    kernels: Arc<RwLock<HashMap<String, KernelInfo>>>,
}

impl KernelRegistry {
    fn new() -> Self {
        Self {
            kernels: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn register(&self, id: &str, info: KernelInfo) {
        self.kernels.write().unwrap().insert(id.to_string(), info);
    }

    fn list_kernels(&self) -> Vec<(String, KernelInfo)> {
        self.kernels
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

#[allow(dead_code)]
struct GrpcRequest {
    id: String,
    kernel_id: String,
    payload: Vec<u8>,
    metadata: HashMap<String, String>,
}

#[allow(dead_code)]
#[derive(Debug)]
enum ResponseStatus {
    Success,
    Error(String),
}

#[allow(dead_code)]
struct GrpcResponse {
    id: String,
    status: ResponseStatus,
    payload: Vec<u8>,
    latency: Duration,
}

#[allow(dead_code)]
struct HealthStatus {
    status: ServerStatus,
    active_connections: u32,
    available_kernels: usize,
    gpu_memory_mb: u64,
}

#[allow(dead_code)]
#[derive(Debug)]
enum ServerStatus {
    Serving,
    NotServing,
    Unknown,
}

struct ServerMetrics {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    avg_latency_ms: f64,
    bytes_received: u64,
    bytes_sent: u64,
}

#[allow(dead_code)]
struct GpuGrpcServer {
    config: GrpcServerConfig,
    registry: KernelRegistry,
    stats: Arc<RequestStats>,
}

struct RequestStats {
    total: AtomicU64,
    success: AtomicU64,
    failed: AtomicU64,
    total_latency_us: AtomicU64,
    bytes_in: AtomicU64,
    bytes_out: AtomicU64,
}

impl GpuGrpcServer {
    fn new(config: GrpcServerConfig, registry: KernelRegistry) -> Self {
        Self {
            config,
            registry,
            stats: Arc::new(RequestStats {
                total: AtomicU64::new(0),
                success: AtomicU64::new(0),
                failed: AtomicU64::new(0),
                total_latency_us: AtomicU64::new(0),
                bytes_in: AtomicU64::new(0),
                bytes_out: AtomicU64::new(0),
            }),
        }
    }

    async fn process(
        &self,
        request: GrpcRequest,
    ) -> std::result::Result<GrpcResponse, Box<dyn std::error::Error>> {
        let start = Instant::now();

        self.stats.total.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_in
            .fetch_add(request.payload.len() as u64, Ordering::Relaxed);

        // Simulate GPU processing
        tokio::time::sleep(Duration::from_micros(100)).await;

        // Generate result based on kernel type
        let result = match request.kernel_id.as_str() {
            "vector_add" => {
                let input = decode_vector(&request.payload);
                let output: Vec<f32> = input.iter().map(|x| x * 2.0).collect();
                encode_vector(&output)
            }
            "matrix_multiply" => {
                // Simulate matrix multiplication result
                encode_vector(&[1.0, 4.0, 9.0, 16.0])
            }
            "neural_inference" => {
                // Simulate neural network output (10 class probabilities)
                encode_vector(&[0.1, 0.05, 0.8, 0.02, 0.01, 0.01, 0.005, 0.003, 0.001, 0.001])
            }
            _ => vec![],
        };

        let latency = start.elapsed();
        self.stats.success.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_latency_us
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);
        self.stats
            .bytes_out
            .fetch_add(result.len() as u64, Ordering::Relaxed);

        Ok(GrpcResponse {
            id: request.id,
            status: ResponseStatus::Success,
            payload: result,
            latency,
        })
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus {
            status: ServerStatus::Serving,
            active_connections: 5,
            available_kernels: self.registry.list_kernels().len(),
            gpu_memory_mb: 8192,
        }
    }

    fn get_metrics(&self) -> ServerMetrics {
        let total = self.stats.total.load(Ordering::Relaxed);
        let success = self.stats.success.load(Ordering::Relaxed);
        let total_latency = self.stats.total_latency_us.load(Ordering::Relaxed);

        ServerMetrics {
            total_requests: total,
            successful_requests: success,
            failed_requests: self.stats.failed.load(Ordering::Relaxed),
            avg_latency_ms: if success > 0 {
                (total_latency as f64 / success as f64) / 1000.0
            } else {
                0.0
            },
            bytes_received: self.stats.bytes_in.load(Ordering::Relaxed),
            bytes_sent: self.stats.bytes_out.load(Ordering::Relaxed),
        }
    }
}

async fn demonstrate_streaming(server: &GpuGrpcServer) {
    println!("Simulating streaming RPC for large dataset processing:");

    let batch_count = 5;
    let batch_size = 1000;

    for i in 0..batch_count {
        let batch: Vec<f32> = (0..batch_size)
            .map(|x| x as f32 + i as f32 * batch_size as f32)
            .collect();

        let request = GrpcRequest {
            id: format!("stream-batch-{}", i),
            kernel_id: "vector_add".to_string(),
            payload: encode_vector(&batch),
            metadata: HashMap::new(),
        };

        let response = server.process(request).await.unwrap();

        println!(
            "  Batch {} processed: {} elements, latency: {:?}",
            i + 1,
            batch_size,
            response.latency
        );
    }

    println!(
        "\nTotal: {} elements processed across {} batches",
        batch_count * batch_size,
        batch_count
    );
}

// ============ Encoding Helpers ============

fn encode_vector(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn decode_vector(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn encode_matrix(data: &[f32], _rows: usize, _cols: usize) -> Vec<u8> {
    encode_vector(data)
}
