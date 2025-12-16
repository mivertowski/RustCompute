//! Example: Axum REST API with Persistent GPU Kernel
//!
//! This example demonstrates how to use ringkernel-ecosystem to create
//! a REST API that controls a persistent GPU kernel with sub-microsecond
//! command latency.
//!
//! # Features Demonstrated
//!
//! - Setting up Axum routes for persistent kernel control
//! - Using `PersistentGpuState` for shared kernel access
//! - REST endpoints for step execution, impulse injection, and stats
//! - Graceful shutdown handling
//!
//! # Running
//!
//! ```bash
//! cargo run -p ringkernel-ecosystem --example axum_persistent_api --features "axum,persistent"
//! ```
//!
//! # API Endpoints
//!
//! - `GET /health` - Health check
//! - `GET /stats` - Get kernel statistics
//! - `POST /steps` - Run N simulation steps
//! - `POST /impulse` - Inject an impulse at a position
//! - `POST /pause` - Pause the kernel
//! - `POST /resume` - Resume the kernel

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use ringkernel_ecosystem::error::{EcosystemError, Result};
use ringkernel_ecosystem::persistent::{
    CommandId, PersistentCommand, PersistentConfig, PersistentHandle, PersistentResponse,
    PersistentStats,
};

/// Mock persistent handle for demonstration.
///
/// In a real application, you would use `CudaPersistentHandle` from
/// `ringkernel_ecosystem::cuda_bridge` with an actual CUDA kernel.
struct MockGpuKernel {
    kernel_id: String,
    running: AtomicBool,
    cmd_counter: AtomicU64,
    step_counter: AtomicU64,
    config: PersistentConfig,
}

impl MockGpuKernel {
    fn new(kernel_id: impl Into<String>) -> Self {
        Self {
            kernel_id: kernel_id.into(),
            running: AtomicBool::new(true),
            cmd_counter: AtomicU64::new(1),
            step_counter: AtomicU64::new(0),
            config: PersistentConfig::default(),
        }
    }
}

#[async_trait::async_trait]
impl PersistentHandle for MockGpuKernel {
    fn kernel_id(&self) -> &str {
        &self.kernel_id
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    fn send_command(&self, cmd: PersistentCommand) -> Result<CommandId> {
        if !self.is_running() {
            return Err(EcosystemError::KernelNotRunning(self.kernel_id.clone()));
        }

        let cmd_id = CommandId(self.cmd_counter.fetch_add(1, Ordering::Relaxed));

        // Simulate command processing
        match cmd {
            PersistentCommand::RunSteps { count } => {
                self.step_counter.fetch_add(count, Ordering::Relaxed);
                println!(
                    "  Executed {} steps (total: {})",
                    count,
                    self.step_counter.load(Ordering::Relaxed)
                );
            }
            PersistentCommand::Inject { position, value } => {
                println!(
                    "  Injected impulse at {:?} with amplitude {}",
                    position, value
                );
            }
            PersistentCommand::Pause => {
                println!("  Kernel paused");
            }
            PersistentCommand::Resume => {
                println!("  Kernel resumed");
            }
            PersistentCommand::Terminate => {
                self.running.store(false, Ordering::Release);
                println!("  Kernel terminated");
            }
            _ => {}
        }

        Ok(cmd_id)
    }

    fn poll_responses(&self) -> Vec<PersistentResponse> {
        Vec::new() // Mock doesn't produce async responses
    }

    fn stats(&self) -> PersistentStats {
        PersistentStats {
            current_step: self.step_counter.load(Ordering::Relaxed),
            is_running: self.is_running(),
            has_terminated: !self.is_running(),
            ..Default::default()
        }
    }

    async fn wait_for_command(&self, _cmd_id: CommandId, _timeout: Duration) -> Result<()> {
        Ok(()) // Mock commands complete immediately
    }

    async fn shutdown(&self) -> Result<()> {
        self.running.store(false, Ordering::Release);
        Ok(())
    }

    fn config(&self) -> &PersistentConfig {
        &self.config
    }
}

/// Application state shared across handlers.
#[derive(Clone)]
struct AppState {
    kernel: Arc<MockGpuKernel>,
}

/// Health check response.
#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    kernel_id: String,
    is_running: bool,
}

/// Statistics response.
#[derive(Serialize)]
struct StatsResponse {
    current_step: u64,
    steps_remaining: u64,
    is_running: bool,
    has_terminated: bool,
    total_energy: f32,
}

/// Request body for running steps.
#[derive(Deserialize)]
struct StepsRequest {
    count: u64,
}

/// Request body for injecting an impulse.
#[derive(Deserialize)]
struct ImpulseRequest {
    x: u32,
    y: u32,
    z: u32,
    amplitude: f32,
}

/// Generic command response.
#[derive(Serialize)]
struct CommandResponse {
    success: bool,
    command_id: u64,
    message: String,
}

// ============================================================================
// HANDLERS
// ============================================================================

async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: if state.kernel.is_running() {
            "healthy"
        } else {
            "stopped"
        },
        kernel_id: state.kernel.kernel_id().to_string(),
        is_running: state.kernel.is_running(),
    })
}

async fn stats_handler(State(state): State<AppState>) -> Json<StatsResponse> {
    let stats = state.kernel.stats();
    Json(StatsResponse {
        current_step: stats.current_step,
        steps_remaining: stats.steps_remaining,
        is_running: stats.is_running,
        has_terminated: stats.has_terminated,
        total_energy: stats.total_energy,
    })
}

async fn steps_handler(
    State(state): State<AppState>,
    Json(req): Json<StepsRequest>,
) -> std::result::Result<Json<CommandResponse>, StatusCode> {
    match state
        .kernel
        .send_command(PersistentCommand::RunSteps { count: req.count })
    {
        Ok(cmd_id) => Ok(Json(CommandResponse {
            success: true,
            command_id: cmd_id.0,
            message: format!("Queued {} steps for execution", req.count),
        })),
        Err(e) => {
            eprintln!("Error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn impulse_handler(
    State(state): State<AppState>,
    Json(req): Json<ImpulseRequest>,
) -> std::result::Result<Json<CommandResponse>, StatusCode> {
    match state.kernel.send_command(PersistentCommand::Inject {
        position: (req.x, req.y, req.z),
        value: req.amplitude,
    }) {
        Ok(cmd_id) => Ok(Json(CommandResponse {
            success: true,
            command_id: cmd_id.0,
            message: format!(
                "Injected impulse at ({}, {}, {}) with amplitude {}",
                req.x, req.y, req.z, req.amplitude
            ),
        })),
        Err(e) => {
            eprintln!("Error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn pause_handler(
    State(state): State<AppState>,
) -> std::result::Result<Json<CommandResponse>, StatusCode> {
    match state.kernel.send_command(PersistentCommand::Pause) {
        Ok(cmd_id) => Ok(Json(CommandResponse {
            success: true,
            command_id: cmd_id.0,
            message: "Kernel paused".to_string(),
        })),
        Err(e) => {
            eprintln!("Error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn resume_handler(
    State(state): State<AppState>,
) -> std::result::Result<Json<CommandResponse>, StatusCode> {
    match state.kernel.send_command(PersistentCommand::Resume) {
        Ok(cmd_id) => Ok(Json(CommandResponse {
            success: true,
            command_id: cmd_id.0,
            message: "Kernel resumed".to_string(),
        })),
        Err(e) => {
            eprintln!("Error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

#[tokio::main]
async fn main() {
    println!("RingKernel Ecosystem - Axum Persistent API Example");
    println!("{}", "=".repeat(50));
    println!();

    // Create mock GPU kernel
    let kernel = Arc::new(MockGpuKernel::new("fdtd_3d_simulation"));
    let state = AppState {
        kernel: kernel.clone(),
    };

    // Build router
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/stats", get(stats_handler))
        .route("/steps", post(steps_handler))
        .route("/impulse", post(impulse_handler))
        .route("/pause", post(pause_handler))
        .route("/resume", post(resume_handler))
        .with_state(state);

    // Start server
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Starting server on http://{}", addr);
    println!();
    println!("Available endpoints:");
    println!("  GET  /health   - Health check");
    println!("  GET  /stats    - Kernel statistics");
    println!("  POST /steps    - Run N steps (body: {{\"count\": N}})");
    println!("  POST /impulse  - Inject impulse (body: {{\"x\": N, \"y\": N, \"z\": N, \"amplitude\": F}})");
    println!("  POST /pause    - Pause kernel");
    println!("  POST /resume   - Resume kernel");
    println!();
    println!("Example commands:");
    println!("  curl http://localhost:3000/health");
    println!("  curl http://localhost:3000/stats");
    println!("  curl -X POST http://localhost:3000/steps -H 'Content-Type: application/json' -d '{{\"count\": 100}}'");
    println!("  curl -X POST http://localhost:3000/impulse -H 'Content-Type: application/json' -d '{{\"x\": 32, \"y\": 32, \"z\": 32, \"amplitude\": 1.5}}'");
    println!();

    // Setup graceful shutdown
    let kernel_shutdown = kernel.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        println!("\nShutting down...");
        kernel_shutdown.shutdown().await.ok();
    });

    // Run server
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
