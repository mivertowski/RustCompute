# Developer Experience Roadmap

> Making GPU Actor Programming Accessible and Productive

## Vision

Transform GPU programming from a specialized skill requiring CUDA/Metal expertise into an accessible extension of everyday Rust development. Developers should be able to write, test, and deploy GPU actors as naturally as they write async Rust code today.

---

## 1. CLI Tooling

### 1.1 `ringkernel` CLI

A comprehensive command-line tool for RingKernel development.

**Installation**:
```bash
cargo install ringkernel-cli
# Or from source
cargo install --path crates/ringkernel-cli
```

**Core Commands**:

```bash
# Project scaffolding
ringkernel new my-gpu-app
ringkernel new my-gpu-app --template persistent-actor
ringkernel new my-gpu-app --template web-api

# Code generation
ringkernel codegen src/kernels/processor.rs --backend cuda
ringkernel codegen src/kernels/processor.rs --backend cuda,metal,wgpu --output-dir generated/

# Validation
ringkernel check                              # Validate all kernels
ringkernel check --kernel processor           # Validate specific kernel
ringkernel check --backends cuda,metal        # Check backend compatibility

# Performance
ringkernel profile --kernel processor --iterations 1000
ringkernel benchmark --suite standard
ringkernel flame --kernel processor --duration 10s

# Development
ringkernel watch                              # Auto-rebuild on changes
ringkernel dev                                # Development server with hot reload
ringkernel test --gpu                         # Run GPU tests

# Deployment
ringkernel build --release --target x86_64-unknown-linux-gnu
ringkernel package --format docker
ringkernel deploy --environment production
```

**Project Templates**:

| Template | Description |
|----------|-------------|
| `basic` | Minimal GPU kernel example |
| `persistent-actor` | Persistent kernel with H2K/K2H messaging |
| `web-api` | Axum REST API with GPU backend |
| `realtime` | Real-time processing with WebSocket |
| `batch` | Batch processing pipeline |
| `simulation` | Physics simulation with visualization |

**Template Structure** (`persistent-actor`):
```
my-gpu-app/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── kernels/
│   │   ├── mod.rs
│   │   └── processor.rs        # GPU kernel definition
│   ├── messages/
│   │   ├── mod.rs
│   │   └── commands.rs         # RingMessage types
│   └── handlers/
│       └── mod.rs              # H2K/K2H handlers
├── generated/
│   ├── cuda/
│   │   └── processor.ptx
│   └── wgsl/
│       └── processor.wgsl
├── tests/
│   ├── integration.rs
│   └── gpu_tests.rs
├── benches/
│   └── throughput.rs
└── ringkernel.toml             # Project configuration
```

**Configuration File** (`ringkernel.toml`):
```toml
[project]
name = "my-gpu-app"
version = "0.1.0"

[kernels]
default_backend = "cuda"
fallback_backend = "cpu"

[kernel.processor]
source = "src/kernels/processor.rs"
backends = ["cuda", "metal"]
block_size = 128
queue_capacity = 1024

[codegen]
output_dir = "generated"
optimize_level = 3
debug_info = false

[development]
hot_reload = true
profile_by_default = false

[testing]
gpu_tests_ignored_by_default = true
mock_backend = "cpu"
```

### 1.2 Watch Mode

Automatic recompilation on file changes.

```bash
# Watch and rebuild
ringkernel watch

# Watch with auto-test
ringkernel watch --test

# Watch with profiling
ringkernel watch --profile

# Output
[2026-01-02 10:30:45] Watching src/kernels/
[2026-01-02 10:30:46] Changed: src/kernels/processor.rs
[2026-01-02 10:30:46] Compiling processor...
[2026-01-02 10:30:47] Generated: generated/cuda/processor.ptx
[2026-01-02 10:30:47] Generated: generated/wgsl/processor.wgsl
[2026-01-02 10:30:47] Tests: 12 passed, 0 failed
```

---

## 2. IDE Integration

### 2.1 VSCode Extension

**Features**:

| Feature | Description |
|---------|-------------|
| **Syntax Highlighting** | GPU DSL keywords and intrinsics |
| **IntelliSense** | Autocomplete for GPU functions |
| **Hover Documentation** | Inline docs for intrinsics |
| **Diagnostics** | Real-time error checking |
| **Code Lens** | Backend compatibility indicators |
| **Snippets** | Common kernel patterns |
| **Debugging** | GPU kernel debugging support |
| **Profiling** | Integrated profiler visualization |

**Syntax Highlighting Example**:
```rust
#[ring_kernel(
    id = "processor",
    mode = "persistent",
    block_size = 128,
)]
async fn handle(ctx: &mut RingContext, msg: Request) -> Response {
    //  ↑ Highlighted: RingContext is a GPU type

    let tid = ctx.global_thread_id();
    //         ↑ Highlighted: GPU intrinsic

    ctx.sync_threads();
    //  ↑ Highlighted: Synchronization primitive

    let neighbor_value = ctx.k2k_recv::<f32>(neighbor_id).await;
    //                      ↑ Highlighted: K2K messaging

    Response { value: msg.value * 2.0 }
}
```

**Code Lens Display**:
```rust
// ✅ CUDA ✅ Metal ⚠️ WebGPU (no K2K)
#[ring_kernel(id = "processor", mode = "persistent")]
async fn handle(ctx: &mut RingContext, msg: Request) -> Response {
    // ...
}
```

**Diagnostic Examples**:
```
error[RK001]: `f64` not supported on WebGPU backend
  --> src/kernels/processor.rs:15:12
   |
15 |     let x: f64 = msg.value;
   |            ^^^ consider using `f32` instead
   |
   = note: WebGPU does not support 64-bit floats
   = help: add `#[gpu_kernel(requires = [f64])]` to exclude WebGPU

warning[RK002]: `k2k_recv` blocks in persistent kernel
  --> src/kernels/processor.rs:20:9
   |
20 |     ctx.k2k_recv(neighbor_id);
   |         ^^^^^^^^ this may cause deadlock if neighbor is not ready
   |
   = help: use `k2k_try_recv` for non-blocking receive
```

**Snippets**:
```json
{
  "Ring Kernel": {
    "prefix": "ringkernel",
    "body": [
      "#[ring_kernel(",
      "    id = \"${1:kernel_name}\",",
      "    mode = \"${2|persistent,transient|}\",",
      "    block_size = ${3:128},",
      ")]",
      "async fn handle(ctx: &mut RingContext, msg: ${4:Request}) -> ${5:Response} {",
      "    $0",
      "}"
    ]
  },
  "K2K Send": {
    "prefix": "k2ksend",
    "body": "ctx.k2k_send(${1:dest_id}, ${2:message}).await?;"
  }
}
```

### 2.2 JetBrains Plugin (IntelliJ IDEA / CLion)

Similar feature set to VSCode:
- Rust plugin integration
- GPU-specific inspections
- Run configurations for GPU tests
- Profiler integration

### 2.3 Neovim/Vim Support

```lua
-- init.lua configuration
require('lspconfig').ringkernel_lsp.setup {
  cmd = { 'ringkernel', 'lsp' },
  filetypes = { 'rust' },
  root_dir = function(fname)
    return vim.fn.findfile('ringkernel.toml', fname .. ';')
  end,
}
```

---

## 3. Testing Infrastructure

### 3.1 GPU Mock Testing

Test GPU code without hardware.

```rust
use ringkernel::testing::{MockGpu, MockRuntime};

#[tokio::test]
async fn test_processor_kernel() {
    // Create mock GPU environment
    let mock = MockGpu::new()
        .with_device_memory(1024 * 1024 * 1024)  // 1GB
        .with_compute_units(80)
        .with_latency_ns(100);

    let runtime = MockRuntime::new(mock);

    // Launch kernel on mock GPU
    let kernel = runtime.launch("processor", Default::default()).await?;

    // Send test message
    let response = kernel.send(Request { value: 42.0 }).await?;

    // Assert response
    assert_eq!(response.value, 84.0);

    // Inspect mock state
    assert_eq!(mock.messages_processed(), 1);
    assert!(mock.peak_memory_usage() < 1024 * 1024);
}
```

**Mock GPU Capabilities**:
```rust
pub struct MockGpu {
    /// Simulate device memory
    pub memory: MockMemory,
    /// Simulate compute units
    pub compute_units: u32,
    /// Simulate kernel execution
    pub execution_mode: ExecutionMode,
    /// Record all operations
    pub recording: bool,
}

pub enum ExecutionMode {
    /// Execute actual kernel logic on CPU
    CpuEmulation,
    /// Record operations without execution
    RecordOnly,
    /// Inject specific responses
    Scripted(Vec<ScriptedResponse>),
    /// Random responses for fuzzing
    Fuzz(FuzzConfig),
}
```

### 3.2 Property-Based Testing

QuickCheck-style testing for kernel invariants.

```rust
use ringkernel::testing::proptest::*;

proptest! {
    #[test]
    fn kernel_preserves_message_order(messages: Vec<Request>) {
        let runtime = MockRuntime::new(MockGpu::default());
        let kernel = block_on(runtime.launch("processor", Default::default()))?;

        // Send all messages
        let responses: Vec<Response> = block_on(async {
            let mut responses = Vec::new();
            for msg in &messages {
                responses.push(kernel.send(msg.clone()).await?);
            }
            Ok::<_, Error>(responses)
        })?;

        // Verify order preserved via correlation IDs
        for (req, resp) in messages.iter().zip(responses.iter()) {
            prop_assert_eq!(req.correlation_id, resp.correlation_id);
        }
    }

    #[test]
    fn hlc_timestamps_monotonic(ops: Vec<HlcOp>) {
        let clock = HlcClock::new(1);
        let mut prev_ts = HlcTimestamp::default();

        for op in ops {
            let ts = match op {
                HlcOp::Tick => clock.tick(),
                HlcOp::Update(received) => clock.update(received),
            };
            prop_assert!(ts > prev_ts);
            prev_ts = ts;
        }
    }
}
```

### 3.3 Fuzzing

AFL/libFuzzer integration for message parsing.

```rust
// fuzz/fuzz_targets/message_parsing.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use ringkernel::message::MessageHeader;

fuzz_target!(|data: &[u8]| {
    // Fuzz message header parsing
    if data.len() >= std::mem::size_of::<MessageHeader>() {
        let header: &MessageHeader = unsafe {
            &*(data.as_ptr() as *const MessageHeader)
        };

        // Should not panic
        let _ = header.validate();
        let _ = header.type_id();
        let _ = header.correlation_id();
    }
});
```

**Running Fuzz Tests**:
```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# Run fuzzer
cargo +nightly fuzz run message_parsing

# Run with corpus
cargo +nightly fuzz run message_parsing corpus/
```

### 3.4 CI GPU Testing

GitHub Actions configuration for GPU tests.

```yaml
# .github/workflows/gpu-tests.yml
name: GPU Tests

on: [push, pull_request]

jobs:
  gpu-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4

      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Run GPU tests
        run: |
          cargo test --workspace --features cuda
        env:
          CUDA_VISIBLE_DEVICES: 0

      - name: Run benchmarks
        run: |
          cargo bench --package ringkernel -- --save-baseline gpu-bench

      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: target/criterion/
```

---

## 4. Documentation

### 4.1 Interactive Tutorials

Step-by-step tutorials with runnable code.

**Tutorial Structure**:
```
docs/tutorials/
├── 01-hello-gpu/
│   ├── README.md
│   ├── Cargo.toml
│   └── src/main.rs
├── 02-persistent-actors/
│   ├── README.md
│   ├── Cargo.toml
│   └── src/main.rs
├── 03-k2k-messaging/
│   ├── README.md
│   ├── Cargo.toml
│   └── src/main.rs
├── 04-web-integration/
│   ├── README.md
│   ├── Cargo.toml
│   └── src/main.rs
└── 05-production-deployment/
    ├── README.md
    └── docker-compose.yml
```

**Tutorial 01: Hello GPU**:
```rust
//! # Tutorial 01: Hello GPU
//!
//! This tutorial introduces the basics of GPU kernel programming
//! with RingKernel.
//!
//! ## What You'll Learn
//! - How to define a GPU kernel
//! - How to launch a kernel
//! - How to send and receive messages
//!
//! ## Prerequisites
//! - Rust 1.75+
//! - CUDA toolkit (optional, will use CPU fallback)

use ringkernel::prelude::*;

// Step 1: Define your message types
#[derive(RingMessage)]
#[message(type_id = 1)]
struct GreetRequest {
    #[message(id)]
    id: MessageId,
    name: String,
}

#[derive(RingMessage)]
#[message(type_id = 2)]
struct GreetResponse {
    #[message(id)]
    id: MessageId,
    #[message(correlation)]
    correlation: CorrelationId,
    greeting: String,
}

// Step 2: Define your kernel handler
#[ring_kernel(id = "greeter", mode = "transient")]
async fn greet(ctx: &mut RingContext, req: GreetRequest) -> GreetResponse {
    GreetResponse {
        id: MessageId::new(),
        correlation: req.id.into(),
        greeting: format!("Hello, {}! From thread {}", req.name, ctx.thread_id()),
    }
}

// Step 3: Launch and use the kernel
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create runtime (auto-detects GPU)
    let runtime = Runtime::new().await?;

    println!("Using backend: {:?}", runtime.backend());

    // Launch kernel
    let kernel = runtime.launch("greeter", Default::default()).await?;

    // Send a message
    let response: GreetResponse = kernel.send(GreetRequest {
        id: MessageId::new(),
        name: "World".to_string(),
    }).await?;

    println!("{}", response.greeting);

    Ok(())
}
```

### 4.2 Architecture Guide

Deep-dive documentation on system internals.

**Chapters**:
1. **Introduction to GPU Actors**
   - Traditional GPU programming vs actor model
   - Benefits of persistent kernels
   - When to use RingKernel

2. **Core Concepts**
   - Message passing fundamentals
   - Hybrid logical clocks
   - Control block architecture

3. **Backend Deep Dives**
   - CUDA implementation details
   - Metal implementation details
   - WebGPU patterns and limitations

4. **Code Generation**
   - Rust DSL syntax
   - Transpilation process
   - Optimization techniques

5. **Ecosystem Integration**
   - Web frameworks
   - Data processing
   - ML frameworks

6. **Production Operations**
   - Monitoring and observability
   - Fault tolerance
   - Performance tuning

### 4.3 API Reference

Complete rustdoc with examples.

```rust
/// A persistent GPU kernel handle for managing long-running GPU computations.
///
/// # Overview
///
/// `PersistentHandle` represents a GPU kernel that runs for the lifetime of your
/// application, processing commands with sub-microsecond latency. Unlike traditional
/// GPU programming where each operation requires a kernel launch (~300µs), persistent
/// kernels achieve command injection in ~0.03µs (11,327x faster).
///
/// # Example
///
/// ```rust
/// use ringkernel::prelude::*;
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     let runtime = Runtime::new().await?;
///     let kernel = runtime.launch_persistent("simulation", PersistentConfig {
///         queue_capacity: 1024,
///         progress_interval: 100,
///     }).await?;
///
///     // Start running steps (non-blocking)
///     kernel.run_steps(1000).await?;
///
///     // Inject impulse at location (sub-microsecond latency!)
///     kernel.inject(32, 32, 32, 1.0).await?;
///
///     // Check progress
///     let stats = kernel.stats().await?;
///     println!("Step: {}/{}", stats.current_step, stats.target_step);
///
///     // Graceful shutdown
///     kernel.shutdown().await?;
///     Ok(())
/// }
/// ```
///
/// # Performance Characteristics
///
/// | Operation | Latency |
/// |-----------|---------|
/// | Command injection | ~0.03µs |
/// | Response polling | ~0.01µs |
/// | Step execution | ~3µs per step |
///
/// # Thread Safety
///
/// `PersistentHandle` is `Send + Sync` and can be shared across threads using `Arc`.
///
/// # See Also
///
/// - [`PersistentConfig`] - Configuration options
/// - [`PersistentCommand`] - Available commands
/// - [`PersistentResponse`] - Response types
pub trait PersistentHandle: Send + Sync {
    // ...
}
```

### 4.4 Example Gallery

Real-world applications with full source code.

| Example | Description | Complexity |
|---------|-------------|------------|
| **basic_hello** | Minimal kernel example | Beginner |
| **persistent_counter** | Persistent state management | Beginner |
| **chat_server** | Real-time chat with WebSocket | Intermediate |
| **image_processor** | GPU image filtering pipeline | Intermediate |
| **trading_engine** | Low-latency order matching | Advanced |
| **physics_sim** | 3D physics with visualization | Advanced |
| **ml_inference** | Neural network inference | Advanced |

---

## 5. Error Messages

### 5.1 Actionable Error Messages

Every error should tell the user what to do.

**Before** (bad):
```
error: incompatible type
```

**After** (good):
```
error[RK0042]: `Vec<f64>` cannot be transferred to GPU

  --> src/kernels/processor.rs:15:12
   |
15 |     let data: Vec<f64> = input.clone();
   |               ^^^^^^^^
   |
   = note: GPU kernels require types that implement `GpuType`
   = note: `Vec<T>` is a heap-allocated type that cannot be directly
           transferred to GPU memory

help: use a GPU-compatible buffer type instead:

   |     let data: GpuBuffer<f64> = GpuBuffer::from_slice(&input);
   |               ^^^^^^^^^^^^^^

help: or use a fixed-size array if the size is known at compile time:

   |     let data: [f64; 1024] = input.try_into()?;
   |               ^^^^^^^^^^^

For more information about GPU-compatible types, see:
  https://ringkernel.dev/docs/gpu-types
```

### 5.2 Error Catalog

Comprehensive error documentation.

```rust
/// Error codes and their meanings
pub enum ErrorCode {
    // ════════════════════════════════════════════════════════════════
    // Type Errors (RK00xx)
    // ════════════════════════════════════════════════════════════════

    /// RK0001: Type does not implement GpuType
    #[error(
        "type `{type_name}` does not implement `GpuType`",
        help = "add `#[derive(GpuType)]` to your type definition"
    )]
    TypeNotGpuCompatible { type_name: String },

    /// RK0002: Type has incorrect alignment
    #[error(
        "type `{type_name}` has alignment {actual}, but GPU requires {required}",
        help = "add `#[repr(C, align({required}))]` to your type"
    )]
    IncorrectAlignment { type_name: String, actual: usize, required: usize },

    // ════════════════════════════════════════════════════════════════
    // Backend Errors (RK01xx)
    // ════════════════════════════════════════════════════════════════

    /// RK0100: No GPU backend available
    #[error(
        "no GPU backend available",
        help = "install CUDA toolkit or enable 'wgpu' feature for cross-platform support"
    )]
    NoBackendAvailable,

    /// RK0101: Backend initialization failed
    #[error(
        "failed to initialize {backend} backend: {reason}",
        help = "check that GPU drivers are installed and up to date"
    )]
    BackendInitFailed { backend: String, reason: String },

    // ════════════════════════════════════════════════════════════════
    // Kernel Errors (RK02xx)
    // ════════════════════════════════════════════════════════════════

    /// RK0200: Kernel not found
    #[error(
        "kernel `{kernel_id}` not found",
        help = "available kernels: {available:?}"
    )]
    KernelNotFound { kernel_id: String, available: Vec<String> },

    // ... more errors
}
```

---

## 6. Performance Tools

### 6.1 Built-in Profiler

Integrated profiling without external tools.

```bash
# Profile a specific kernel
ringkernel profile --kernel processor --duration 10s

# Output:
╔══════════════════════════════════════════════════════════════════╗
║                    Kernel Profile: processor                      ║
╠══════════════════════════════════════════════════════════════════╣
║ Duration: 10.00s                                                  ║
║ Total Steps: 1,000,000                                            ║
║ Throughput: 100,000 steps/sec                                     ║
╠══════════════════════════════════════════════════════════════════╣
║                          Timing Breakdown                         ║
╠══════════════════════════════════════════════════════════════════╣
║ Component              │ Time (ms) │ % Total │ Calls    │ Avg     ║
╟────────────────────────┼───────────┼─────────┼──────────┼─────────╢
║ Simulation Step        │   7,234   │  72.3%  │ 1,000,000│  7.2µs  ║
║ K2K Halo Exchange      │   1,823   │  18.2%  │ 1,000,000│  1.8µs  ║
║ Grid Sync              │     521   │   5.2%  │ 1,000,000│  0.5µs  ║
║ H2K Processing         │     234   │   2.3%  │    10,000│ 23.4µs  ║
║ K2H Response           │     188   │   1.9%  │    10,000│ 18.8µs  ║
╠══════════════════════════════════════════════════════════════════╣
║                          Memory Usage                             ║
╠══════════════════════════════════════════════════════════════════╣
║ GPU Memory Used: 256 MB / 8 GB (3.1%)                             ║
║ Peak Allocation: 312 MB at t=4.2s                                 ║
║ Memory Bandwidth: 412 GB/s (51% of theoretical)                   ║
╠══════════════════════════════════════════════════════════════════╣
║                         Recommendations                           ║
╠══════════════════════════════════════════════════════════════════╣
║ ⚠ K2K Halo Exchange taking 18% of time                           ║
║   → Consider increasing tile size to reduce halo overhead         ║
║                                                                   ║
║ ℹ Memory bandwidth at 51% utilization                            ║
║   → Computation is likely compute-bound, not memory-bound         ║
╚══════════════════════════════════════════════════════════════════╝
```

### 6.2 Flame Graphs

Visual profiling with flame graphs.

```bash
# Generate flame graph
ringkernel flame --kernel processor --duration 5s --output profile.svg

# Interactive HTML report
ringkernel flame --kernel processor --duration 5s --format html --output profile.html
```

### 6.3 Benchmark Suite

Standardized benchmarks for comparison.

```bash
# Run standard benchmark suite
ringkernel benchmark --suite standard

# Compare with baseline
ringkernel benchmark --suite standard --compare baseline.json

# Output:
╔════════════════════════════════════════════════════════════════╗
║                   Benchmark Results                             ║
╠════════════════════════════════════════════════════════════════╣
║ Benchmark               │ Current    │ Baseline   │ Change     ║
╟─────────────────────────┼────────────┼────────────┼────────────╢
║ h2k_latency            │   0.03µs   │   0.03µs   │    +0%     ║
║ k2h_latency            │   0.01µs   │   0.01µs   │    +0%     ║
║ step_throughput        │ 100k/s     │  95k/s     │   +5.3%    ║
║ k2k_bandwidth          │ 1.2M msg/s │ 1.1M msg/s │   +9.1%    ║
║ memory_bandwidth       │ 412 GB/s   │ 398 GB/s   │   +3.5%    ║
║ checkpoint_time_1gb    │   0.82s    │   0.95s    │  -13.7%    ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 7. Community & Ecosystem

### 7.1 Package Registry

Centralized registry for RingKernel extensions.

```bash
# Search for packages
ringkernel search "image processing"

# Install package
ringkernel add ringkernel-image

# Publish package
ringkernel publish
```

### 7.2 Example Repository

Community-contributed examples.

```
examples.ringkernel.dev/
├── official/           # Maintained by RingKernel team
├── community/          # Community contributions
└── showcase/           # Production case studies
```

### 7.3 Discord/Forum

Community support channels with:
- #help - General questions
- #showcase - Share your projects
- #performance - Optimization discussions
- #contributing - Development discussions

---

## Implementation Priority

### Phase 1: Core DX (Q1 2026)
- [ ] ringkernel-cli with new/codegen/check commands
- [ ] VSCode extension (syntax highlighting + diagnostics)
- [ ] Mock GPU testing framework
- [ ] Tutorial 01-03

### Phase 2: Testing & Docs (Q2 2026)
- [ ] Property-based testing integration
- [ ] Fuzzing targets
- [ ] CI GPU testing templates
- [ ] Complete API reference
- [ ] Tutorial 04-05

### Phase 3: Performance Tools (Q3 2026)
- [ ] Built-in profiler
- [ ] Flame graph generation
- [ ] Benchmark suite
- [ ] VSCode profiler integration

### Phase 4: Ecosystem (Q4 2026)
- [ ] Package registry
- [ ] JetBrains plugin
- [ ] Example gallery
- [ ] Community forum

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Time to first GPU kernel | < 5 minutes |
| Test coverage | > 90% |
| Documentation coverage | > 95% |
| VSCode extension installs | 10,000+ |
| Community packages | 50+ |
| Discord members | 1,000+ |
