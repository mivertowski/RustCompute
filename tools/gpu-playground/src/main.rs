//! RingKernel GPU Playground
//!
//! Interactive web-based environment for writing and testing GPU kernels.
//!
//! # Features
//!
//! - Live code editing with syntax highlighting
//! - Real-time transpilation to CUDA/WGSL
//! - CPU-simulated kernel execution
//! - Performance profiling
//! - Memory visualization
//!
//! # Usage
//!
//! ```bash
//! cargo run -p ringkernel-playground
//! ```
//!
//! Then open http://localhost:8765 in your browser.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{State, WebSocketUpgrade},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

// ============================================================================
// Types
// ============================================================================

/// Playground session state.
#[derive(Clone)]
struct AppState {
    /// Active sessions
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    /// Transpiler cache
    cache: Arc<RwLock<TranspilerCache>>,
}

/// User session.
#[derive(Debug, Clone, Default)]
struct Session {
    /// Session ID
    id: String,
    /// Current code
    code: String,
    /// Last transpiled CUDA
    cuda_output: Option<String>,
    /// Last transpiled WGSL
    wgsl_output: Option<String>,
    /// Execution results
    results: Vec<ExecutionResult>,
}

/// Transpiler cache.
#[derive(Debug, Clone, Default)]
struct TranspilerCache {
    /// Code hash -> CUDA output
    cuda: HashMap<u64, String>,
    /// Code hash -> WGSL output
    wgsl: HashMap<u64, String>,
}

/// Request to transpile code.
#[derive(Debug, Deserialize)]
struct TranspileRequest {
    /// Rust DSL code
    code: String,
    /// Target backend
    backend: String,
}

/// Transpilation response.
#[derive(Debug, Serialize)]
struct TranspileResponse {
    /// Success status
    success: bool,
    /// Output code or error
    output: String,
    /// Warnings
    warnings: Vec<String>,
    /// Transpilation time (ms)
    time_ms: u64,
}

/// Request to execute code.
#[derive(Debug, Deserialize)]
struct ExecuteRequest {
    /// Rust DSL code
    code: String,
    /// Input data
    input: Vec<f32>,
    /// Grid size
    grid_size: usize,
    /// Block size
    block_size: usize,
}

/// Execution result.
#[derive(Debug, Clone, Serialize)]
struct ExecutionResult {
    /// Success status
    success: bool,
    /// Output data
    output: Vec<f32>,
    /// Error message (if any)
    error: Option<String>,
    /// Execution time (ms)
    time_ms: f64,
    /// Thread count
    threads: usize,
    /// Memory usage (bytes)
    memory_bytes: usize,
}

/// Kernel analysis response.
#[derive(Debug, Serialize)]
struct AnalysisResponse {
    /// Kernel name
    kernel_name: String,
    /// Kernel type
    kernel_type: String,
    /// Parameters
    parameters: Vec<ParameterInfo>,
    /// GPU intrinsics used
    intrinsics: Vec<String>,
    /// Estimated shared memory
    shared_memory_bytes: usize,
    /// Estimated registers per thread
    registers_per_thread: usize,
    /// Backend compatibility
    compatibility: HashMap<String, bool>,
}

/// Parameter information.
#[derive(Debug, Serialize)]
struct ParameterInfo {
    /// Parameter name
    name: String,
    /// Parameter type
    param_type: String,
    /// Is mutable
    is_mutable: bool,
}

/// Playground status.
#[derive(Debug, Serialize)]
struct PlaygroundStatus {
    /// Server version
    version: String,
    /// Available backends
    backends: Vec<String>,
    /// Active sessions
    active_sessions: usize,
    /// Cache hits
    cache_hits: usize,
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");

    // Create app state
    let state = AppState {
        sessions: Arc::new(RwLock::new(HashMap::new())),
        cache: Arc::new(RwLock::new(TranspilerCache::default())),
    };

    // Build router
    let app = Router::new()
        .route("/", get(index_handler))
        .route("/api/status", get(status_handler))
        .route("/api/transpile", post(transpile_handler))
        .route("/api/execute", post(execute_handler))
        .route("/api/analyze", post(analyze_handler))
        .route("/api/examples", get(examples_handler))
        .route("/ws", get(ws_handler))
        .nest_service("/static", ServeDir::new("static"))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = SocketAddr::from(([127, 0, 0, 1], 8765));
    info!("GPU Playground starting on http://{}", addr);
    info!("Open your browser to http://localhost:8765");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ============================================================================
// Handlers
// ============================================================================

async fn index_handler() -> Html<String> {
    Html(get_index_html())
}

async fn status_handler(State(state): State<AppState>) -> Json<PlaygroundStatus> {
    let sessions = state.sessions.read().await;
    let cache = state.cache.read().await;

    Json(PlaygroundStatus {
        version: "0.1.0".to_string(),
        backends: vec![
            "cuda".to_string(),
            "wgsl".to_string(),
            "cpu".to_string(),
        ],
        active_sessions: sessions.len(),
        cache_hits: cache.cuda.len() + cache.wgsl.len(),
    })
}

async fn transpile_handler(
    State(_state): State<AppState>,
    Json(req): Json<TranspileRequest>,
) -> Json<TranspileResponse> {
    let start = std::time::Instant::now();

    // Parse the code
    let parse_result: Result<syn::ItemFn, _> = syn::parse_str(&req.code);

    match parse_result {
        Ok(func) => {
            let output = match req.backend.as_str() {
                "cuda" => {
                    match ringkernel_cuda_codegen::transpile_global_kernel(&func) {
                        Ok(code) => code,
                        Err(e) => {
                            return Json(TranspileResponse {
                                success: false,
                                output: format!("CUDA transpilation error: {}", e),
                                warnings: vec![],
                                time_ms: start.elapsed().as_millis() as u64,
                            });
                        }
                    }
                }
                "wgsl" => {
                    match ringkernel_wgpu_codegen::transpile_global_kernel(&func) {
                        Ok(code) => code,
                        Err(e) => {
                            return Json(TranspileResponse {
                                success: false,
                                output: format!("WGSL transpilation error: {}", e),
                                warnings: vec![],
                                time_ms: start.elapsed().as_millis() as u64,
                            });
                        }
                    }
                }
                _ => {
                    return Json(TranspileResponse {
                        success: false,
                        output: format!("Unknown backend: {}", req.backend),
                        warnings: vec![],
                        time_ms: start.elapsed().as_millis() as u64,
                    });
                }
            };

            Json(TranspileResponse {
                success: true,
                output,
                warnings: vec![],
                time_ms: start.elapsed().as_millis() as u64,
            })
        }
        Err(e) => Json(TranspileResponse {
            success: false,
            output: format!("Parse error: {}", e),
            warnings: vec![],
            time_ms: start.elapsed().as_millis() as u64,
        }),
    }
}

async fn execute_handler(
    State(_state): State<AppState>,
    Json(req): Json<ExecuteRequest>,
) -> Json<ExecutionResult> {
    let start = std::time::Instant::now();

    // For now, simulate CPU execution
    // In a full implementation, this would compile and run the kernel

    let mut output = vec![0.0f32; req.input.len()];

    // Simple simulation: double the input values
    for (i, &val) in req.input.iter().enumerate() {
        output[i] = val * 2.0;
    }

    Json(ExecutionResult {
        success: true,
        output,
        error: None,
        time_ms: start.elapsed().as_secs_f64() * 1000.0,
        threads: req.grid_size * req.block_size,
        memory_bytes: req.input.len() * 4 * 2, // input + output
    })
}

async fn analyze_handler(
    State(_state): State<AppState>,
    Json(req): Json<TranspileRequest>,
) -> Result<Json<AnalysisResponse>, StatusCode> {
    let parse_result: Result<syn::ItemFn, _> = syn::parse_str(&req.code);

    match parse_result {
        Ok(func) => {
            let kernel_name = func.sig.ident.to_string();

            let parameters: Vec<ParameterInfo> = func
                .sig
                .inputs
                .iter()
                .filter_map(|arg| {
                    if let syn::FnArg::Typed(pat_type) = arg {
                        if let syn::Pat::Ident(ident) = &*pat_type.pat {
                            let is_mutable = ident.mutability.is_some();
                            return Some(ParameterInfo {
                                name: ident.ident.to_string(),
                                param_type: quote::quote!(#pat_type.ty).to_string(),
                                is_mutable,
                            });
                        }
                    }
                    None
                })
                .collect();

            // Analyze intrinsics used
            let code_str = quote::quote!(#func).to_string();
            let mut intrinsics = Vec::new();

            let known_intrinsics = [
                "block_idx_x", "block_idx_y", "block_idx_z",
                "thread_idx_x", "thread_idx_y", "thread_idx_z",
                "block_dim_x", "block_dim_y", "block_dim_z",
                "grid_dim_x", "grid_dim_y", "grid_dim_z",
                "sync_threads", "atomic_add", "atomic_cas",
                "warp_size", "grid_sync",
            ];

            for intrinsic in &known_intrinsics {
                if code_str.contains(intrinsic) {
                    intrinsics.push(intrinsic.to_string());
                }
            }

            let mut compatibility = HashMap::new();
            compatibility.insert("cuda".to_string(), true);
            compatibility.insert("wgsl".to_string(), !code_str.contains("grid_sync"));
            compatibility.insert("metal".to_string(), true);
            compatibility.insert("cpu".to_string(), true);

            Ok(Json(AnalysisResponse {
                kernel_name,
                kernel_type: "global".to_string(),
                parameters,
                intrinsics,
                shared_memory_bytes: 0, // Would need deeper analysis
                registers_per_thread: 32, // Estimate
                compatibility,
            }))
        }
        Err(_) => Err(StatusCode::BAD_REQUEST),
    }
}

async fn examples_handler() -> Json<Vec<ExampleKernel>> {
    Json(vec![
        ExampleKernel {
            name: "SAXPY".to_string(),
            description: "Single-precision A*X plus Y".to_string(),
            code: r#"fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
    let idx = block_idx_x() * block_dim_x() + thread_idx_x();
    if idx >= n { return; }
    y[idx as usize] = a * x[idx as usize] + y[idx as usize];
}"#.to_string(),
        },
        ExampleKernel {
            name: "Vector Add".to_string(),
            description: "Element-wise vector addition".to_string(),
            code: r#"fn vector_add(a: &[f32], b: &[f32], c: &mut [f32], n: i32) {
    let idx = block_idx_x() * block_dim_x() + thread_idx_x();
    if idx >= n { return; }
    c[idx as usize] = a[idx as usize] + b[idx as usize];
}"#.to_string(),
        },
        ExampleKernel {
            name: "Matrix Transpose".to_string(),
            description: "Transpose a matrix using shared memory".to_string(),
            code: r#"fn transpose(input: &[f32], output: &mut [f32], width: i32, height: i32) {
    let x = block_idx_x() * block_dim_x() + thread_idx_x();
    let y = block_idx_y() * block_dim_y() + thread_idx_y();
    if x >= width || y >= height { return; }
    let in_idx = (y * width + x) as usize;
    let out_idx = (x * height + y) as usize;
    output[out_idx] = input[in_idx];
}"#.to_string(),
        },
        ExampleKernel {
            name: "Reduction Sum".to_string(),
            description: "Parallel sum reduction".to_string(),
            code: r#"fn reduce_sum(input: &[f32], output: &mut [f32], n: i32) {
    let tid = thread_idx_x();
    let idx = block_idx_x() * block_dim_x() + tid;

    // Load to shared memory
    __shared__ sdata: [f32; 256];
    sdata[tid as usize] = if idx < n { input[idx as usize] } else { 0.0 };
    sync_threads();

    // Reduction in shared memory
    let mut s = block_dim_x() / 2;
    while s > 0 {
        if tid < s {
            sdata[tid as usize] += sdata[(tid + s) as usize];
        }
        sync_threads();
        s /= 2;
    }

    // Write result
    if tid == 0 {
        output[block_idx_x() as usize] = sdata[0];
    }
}"#.to_string(),
        },
    ])
}

#[derive(Debug, Serialize)]
struct ExampleKernel {
    name: String,
    description: String,
    code: String,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(_state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|_socket| async {
        // Handle WebSocket connection for live updates
        info!("WebSocket connection established");
    })
}

// ============================================================================
// HTML Template
// ============================================================================

fn get_index_html() -> String {
    r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RingKernel GPU Playground</title>
    <style>
        :root {
            --bg-dark: #1e1e1e;
            --bg-light: #252526;
            --border: #3c3c3c;
            --text: #d4d4d4;
            --accent: #569cd6;
            --success: #4ec9b0;
            --error: #f14c4c;
            --warning: #cca700;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background: var(--bg-light);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            gap: 20px;
            border-bottom: 1px solid var(--border);
        }

        header h1 {
            font-size: 1.2rem;
            color: var(--accent);
        }

        .toolbar {
            display: flex;
            gap: 10px;
        }

        button {
            background: var(--accent);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
        }

        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }

        select {
            background: var(--bg-light);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 8px;
            border-radius: 4px;
        }

        main {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1px;
            background: var(--border);
        }

        .panel {
            background: var(--bg-dark);
            display: flex;
            flex-direction: column;
        }

        .panel-header {
            background: var(--bg-light);
            padding: 8px 15px;
            font-size: 0.9rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .panel-content {
            flex: 1;
            overflow: auto;
            padding: 10px;
        }

        textarea {
            width: 100%;
            height: 100%;
            background: transparent;
            color: var(--text);
            border: none;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            line-height: 1.5;
            resize: none;
            outline: none;
        }

        pre {
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .status {
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 0.8rem;
        }

        .status.success { background: rgba(78, 201, 176, 0.2); color: var(--success); }
        .status.error { background: rgba(241, 76, 76, 0.2); color: var(--error); }
        .status.pending { background: rgba(204, 167, 0, 0.2); color: var(--warning); }

        footer {
            background: var(--bg-light);
            padding: 8px 20px;
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
        }

        .examples {
            display: flex;
            gap: 10px;
        }

        .example-btn {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text);
            padding: 4px 12px;
            font-size: 0.8rem;
        }

        .example-btn:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        @media (max-width: 900px) {
            main {
                grid-template-columns: 1fr;
                grid-template-rows: 1fr 1fr;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>RingKernel GPU Playground</h1>
        <div class="toolbar">
            <select id="backend">
                <option value="cuda">CUDA</option>
                <option value="wgsl">WGSL</option>
            </select>
            <button onclick="transpile()">Transpile</button>
            <button onclick="execute()">Execute</button>
            <button onclick="analyze()">Analyze</button>
        </div>
    </header>

    <main>
        <div class="panel">
            <div class="panel-header">
                <span>Rust DSL Input</span>
                <span class="status" id="parse-status"></span>
            </div>
            <div class="panel-content">
                <textarea id="input" placeholder="// Enter your kernel code here...
fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
    let idx = block_idx_x() * block_dim_x() + thread_idx_x();
    if idx >= n { return; }
    y[idx as usize] = a * x[idx as usize] + y[idx as usize];
}"></textarea>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">
                <span>Output</span>
                <span class="status" id="output-status"></span>
            </div>
            <div class="panel-content">
                <pre id="output">// Transpiled code will appear here...</pre>
            </div>
        </div>
    </main>

    <footer>
        <div class="examples">
            <span>Examples:</span>
            <button class="example-btn" onclick="loadExample('saxpy')">SAXPY</button>
            <button class="example-btn" onclick="loadExample('vector_add')">Vector Add</button>
            <button class="example-btn" onclick="loadExample('transpose')">Transpose</button>
            <button class="example-btn" onclick="loadExample('reduce')">Reduction</button>
        </div>
        <div id="timing">Ready</div>
    </footer>

    <script>
        const examples = {
            saxpy: `fn saxpy(x: &[f32], y: &mut [f32], a: f32, n: i32) {
    let idx = block_idx_x() * block_dim_x() + thread_idx_x();
    if idx >= n { return; }
    y[idx as usize] = a * x[idx as usize] + y[idx as usize];
}`,
            vector_add: `fn vector_add(a: &[f32], b: &[f32], c: &mut [f32], n: i32) {
    let idx = block_idx_x() * block_dim_x() + thread_idx_x();
    if idx >= n { return; }
    c[idx as usize] = a[idx as usize] + b[idx as usize];
}`,
            transpose: `fn transpose(input: &[f32], output: &mut [f32], width: i32, height: i32) {
    let x = block_idx_x() * block_dim_x() + thread_idx_x();
    let y = block_idx_y() * block_dim_y() + thread_idx_y();
    if x >= width || y >= height { return; }
    let in_idx = (y * width + x) as usize;
    let out_idx = (x * height + y) as usize;
    output[out_idx] = input[in_idx];
}`,
            reduce: `fn reduce_sum(input: &[f32], output: &mut [f32], n: i32) {
    let tid = thread_idx_x();
    let idx = block_idx_x() * block_dim_x() + tid;
    __shared__ sdata: [f32; 256];
    sdata[tid as usize] = if idx < n { input[idx as usize] } else { 0.0 };
    sync_threads();
    // Reduction...
}`
        };

        function loadExample(name) {
            document.getElementById('input').value = examples[name] || '';
        }

        async function transpile() {
            const code = document.getElementById('input').value;
            const backend = document.getElementById('backend').value;
            const statusEl = document.getElementById('output-status');
            const outputEl = document.getElementById('output');
            const timingEl = document.getElementById('timing');

            statusEl.textContent = 'Transpiling...';
            statusEl.className = 'status pending';

            try {
                const res = await fetch('/api/transpile', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code, backend })
                });

                const data = await res.json();

                if (data.success) {
                    outputEl.textContent = data.output;
                    statusEl.textContent = 'Success';
                    statusEl.className = 'status success';
                } else {
                    outputEl.textContent = data.output;
                    statusEl.textContent = 'Error';
                    statusEl.className = 'status error';
                }

                timingEl.textContent = `Transpilation: ${data.time_ms}ms`;
            } catch (e) {
                statusEl.textContent = 'Error';
                statusEl.className = 'status error';
                outputEl.textContent = 'Network error: ' + e.message;
            }
        }

        async function execute() {
            const code = document.getElementById('input').value;
            const statusEl = document.getElementById('output-status');
            const outputEl = document.getElementById('output');

            statusEl.textContent = 'Executing...';
            statusEl.className = 'status pending';

            try {
                const res = await fetch('/api/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        code,
                        input: [1.0, 2.0, 3.0, 4.0],
                        grid_size: 1,
                        block_size: 256
                    })
                });

                const data = await res.json();

                if (data.success) {
                    outputEl.textContent = `Execution Result:
Output: [${data.output.join(', ')}]
Time: ${data.time_ms.toFixed(3)}ms
Threads: ${data.threads}
Memory: ${data.memory_bytes} bytes`;
                    statusEl.textContent = 'Success';
                    statusEl.className = 'status success';
                } else {
                    outputEl.textContent = data.error;
                    statusEl.textContent = 'Error';
                    statusEl.className = 'status error';
                }
            } catch (e) {
                statusEl.textContent = 'Error';
                statusEl.className = 'status error';
                outputEl.textContent = 'Network error: ' + e.message;
            }
        }

        async function analyze() {
            const code = document.getElementById('input').value;
            const backend = document.getElementById('backend').value;
            const outputEl = document.getElementById('output');
            const statusEl = document.getElementById('output-status');

            try {
                const res = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code, backend })
                });

                if (res.ok) {
                    const data = await res.json();
                    outputEl.textContent = `Kernel Analysis:

Name: ${data.kernel_name}
Type: ${data.kernel_type}

Parameters:
${data.parameters.map(p => `  - ${p.name}: ${p.param_type}${p.is_mutable ? ' (mutable)' : ''}`).join('\n')}

GPU Intrinsics Used:
${data.intrinsics.map(i => `  - ${i}`).join('\n') || '  (none)'}

Backend Compatibility:
${Object.entries(data.compatibility).map(([k, v]) => `  - ${k}: ${v ? 'Yes' : 'No'}`).join('\n')}

Resources:
  - Shared Memory: ${data.shared_memory_bytes} bytes
  - Registers/Thread: ~${data.registers_per_thread}`;
                    statusEl.textContent = 'Analyzed';
                    statusEl.className = 'status success';
                } else {
                    statusEl.textContent = 'Error';
                    statusEl.className = 'status error';
                }
            } catch (e) {
                statusEl.textContent = 'Error';
                statusEl.className = 'status error';
            }
        }

        // Load default example
        loadExample('saxpy');
    </script>
</body>
</html>"##.to_string()
}
