---
layout: default
title: Crate Structure
nav_order: 3
---

# Crate Structure

## Workspace Organization

```
RustCompute/
├── Cargo.toml                    # Workspace manifest
├── README.md
├── CLAUDE.md                     # AI assistant guidance
├── LICENSE-MIT / LICENSE-APACHE
│
├── crates/
│   ├── ringkernel/               # Main facade crate (re-exports)
│   │   ├── Cargo.toml            # Dependencies + 11 examples
│   │   └── src/lib.rs
│   │
│   ├── ringkernel-core/          # Core traits and types
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── message.rs        # RingMessage trait, priority constants
│   │       ├── queue.rs          # MessageQueue trait, QueueTier, QueueFactory, QueueMonitor (v0.3.0)
│   │       ├── runtime.rs        # RingKernel, KernelHandle, LaunchOptions
│   │       ├── context.rs        # RingContext struct
│   │       ├── control.rs        # ControlBlock struct
│   │       ├── telemetry.rs      # TelemetryBuffer, MetricsCollector
│   │       ├── pubsub.rs         # PubSubBroker, Topic wildcards
│   │       ├── hlc.rs            # HlcTimestamp, HlcClock
│   │       ├── error.rs          # Error types
│   │       ├── memory.rs         # (v0.3.0) SizeBucket, StratifiedMemoryPool, PressureHandler
│   │       ├── reduction.rs      # (v0.3.0) ReductionOp, ReductionScalar, GlobalReduction
│   │       ├── analytics_context.rs  # (v0.3.0) AnalyticsContext, AllocationHandle
│   │       ├── dispatcher.rs     # (v0.3.0) KernelDispatcher, DispatcherBuilder
│   │       ├── persistent_message.rs # (v0.3.0) PersistentMessage trait, DispatchTable
│   │       └── domain.rs         # (v0.3.0) Domain enum (20 business domains)
│   │
│   ├── ringkernel-derive/        # Proc macros (in development)
│   │   └── src/lib.rs
│   │
│   ├── ringkernel-cpu/           # CPU backend (working)
│   │   └── src/
│   │       ├── lib.rs
│   │       └── runtime.rs
│   │
│   ├── ringkernel-cuda/          # CUDA backend (working)
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs            # compile_ptx() function (v0.3.0)
│   │   │   ├── runtime.rs        # CudaRuntime implementation
│   │   │   ├── ptx.rs            # PTX template for persistent kernels
│   │   │   ├── reduction.rs      # (v0.3.0) ReductionBuffer, ReductionBufferCache
│   │   │   ├── phases.rs         # (v0.3.0) SyncMode, MultiPhaseConfig, MultiPhaseExecutor
│   │   │   └── persistent.rs     # (v0.3.0) PersistentSimulation, PersistentControlBlock
│   │   └── tests/
│   │       └── gpu_execution_verify.rs  # GPU execution verification
│   │
│   ├── ringkernel-metal/         # Metal backend (scaffolded)
│   │   └── src/lib.rs
│   │
│   ├── ringkernel-wgpu/          # WebGPU backend (scaffolded)
│   │   └── src/lib.rs
│   │
│   ├── ringkernel-codegen/       # Kernel code generation
│   │   └── src/lib.rs
│   │
│   ├── ringkernel-ir/            # Unified IR for multi-backend codegen
│   │   └── src/
│   │       ├── lib.rs            # IrModule, IrBuilder, ValueId, BlockId
│   │       ├── builder.rs        # Fluent IR construction
│   │       ├── nodes.rs          # IR node types (BinaryOp, Load, Store, etc.)
│   │       ├── types.rs          # IrType, ScalarType, VectorType
│   │       ├── optimize.rs       # Optimization passes (DCE, constant folding)
│   │       ├── lower_cuda.rs     # CUDA backend lowering
│   │       ├── lower_wgsl.rs     # WGSL backend lowering
│   │       ├── lower_msl.rs      # MSL backend lowering
│   │       └── validation.rs     # IR validation
│   │
│   ├── ringkernel-cli/           # CLI tool for scaffolding and codegen
│   │   └── src/
│   │       ├── main.rs           # Command-line interface
│   │       ├── commands/         # new, init, codegen, check
│   │       └── templates/        # Project templates
│   │
│   ├── ringkernel-cuda-codegen/  # Rust-to-CUDA transpiler
│   │   └── src/
│   │       ├── lib.rs            # Public API (3 kernel types)
│   │       ├── transpiler.rs     # Core transpilation engine
│   │       ├── intrinsics.rs     # GPU intrinsic mappings (40+)
│   │       ├── stencil.rs        # Stencil kernel support
│   │       ├── types.rs          # Type mapping (Rust → CUDA)
│   │       ├── dsl.rs            # DSL functions (block_idx_x, etc.)
│   │       ├── validation.rs     # Code validation with modes
│   │       ├── loops.rs          # Loop transpilation (for/while/loop)
│   │       ├── shared.rs         # Shared memory (__shared__)
│   │       ├── ring_kernel.rs    # Ring kernel generation
│   │       └── handler.rs        # Handler function integration
│   │
│   ├── ringkernel-ecosystem/     # Integration utilities
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── actix.rs          # GpuPersistentActor for Actix
│   │       ├── tower.rs          # PersistentKernelService middleware
│   │       ├── axum.rs           # PersistentGpuState, REST/SSE endpoints
│   │       ├── grpc.rs           # gRPC streaming server
│   │       └── cuda_bridge.rs    # CudaPersistentHandle
│   │
│   ├── ringkernel-montecarlo/    # (v0.3.0) Monte Carlo primitives
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── philox.rs         # Philox PRNG
│   │       ├── antithetic.rs     # Antithetic variates
│   │       ├── control_variate.rs # Control variates
│   │       └── importance.rs     # Importance sampling
│   │
│   ├── ringkernel-graph/         # (v0.3.0) Graph algorithms
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── csr.rs            # CSR sparse matrix
│   │       ├── bfs.rs            # Breadth-first search
│   │       ├── scc.rs            # SCC (Tarjan, Kosaraju)
│   │       ├── union_find.rs     # Parallel Union-Find (Shiloach-Vishkin)
│   │       └── spmv.rs           # Sparse matrix-vector multiply
│   │
│   ├── ringkernel-audio-fft/     # Example: GPU audio processing
│   │   └── src/lib.rs
│   │
│   ├── ringkernel-wavesim/       # Example: 2D wave simulation
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── simulation/       # Grid, kernels, backends
│   │       └── gui/              # Interactive visualization
│   │
│   ├── ringkernel-wavesim3d/     # Showcase: 3D wave simulation
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── simulation/       # 3D FDTD, actor backend, physics
│   │       ├── audio/            # Binaural audio, sources, virtual head
│   │       ├── visualization/    # Volume renderer, slices, camera
│   │       └── gui/              # Controls panel
│   │
│   ├── ringkernel-txmon/         # Showcase: Transaction monitoring
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── types/            # Transaction, CustomerProfile, Alert
│   │       ├── factory/          # Transaction generator
│   │       ├── monitoring/       # Rule engine
│   │       ├── gui/              # Real-time fraud detection UI
│   │       ├── cuda/             # GPU backends (batch, ring, stencil)
│   │       └── bin/
│   │           ├── txmon.rs      # Main GUI binary
│   │           └── benchmark.rs  # GPU benchmark
│   │
│   ├── ringkernel-accnet/        # Showcase: Accounting network analytics
│   │   └── src/
│   │       ├── lib.rs
│   │       └── ...               # Network analysis, fraud detection
│   │
│   └── ringkernel-procint/       # Showcase: Process intelligence
│       └── src/
│           ├── lib.rs
│           └── ...               # DFG mining, pattern detection
│
├── examples/                     # 20+ working examples
│   ├── basic/
│   │   ├── hello_kernel.rs       # Runtime, lifecycle, suspend/resume
│   │   ├── kernel_states.rs      # State machine, multi-kernel
│   │   └── wgpu_hello.rs         # WebGPU backend
│   ├── messaging/
│   │   ├── request_response.rs   # Correlation IDs, priorities
│   │   ├── pub_sub.rs            # Topic wildcards, QoS
│   │   └── kernel_to_kernel.rs   # K2K direct messaging
│   ├── cuda-codegen/             # CUDA code generation examples
│   │   ├── global_kernel.rs      # SAXPY, halo exchange, array init
│   │   ├── stencil_kernel.rs     # FDTD wave, heat diffusion, GridPos
│   │   └── ring_kernel.rs        # Persistent kernels, HLC, K2K
│   ├── web-api/
│   │   └── axum_api.rs           # REST API integration
│   ├── data-processing/
│   │   └── batch_processor.rs    # Data pipelines
│   ├── monitoring/
│   │   └── telemetry.rs          # Metrics, alerts
│   ├── ecosystem/
│   │   ├── grpc_server.rs        # gRPC patterns
│   │   ├── config_management.rs  # TOML, env vars
│   │   └── ml_pipeline.rs        # ML inference
│   ├── macros/
│   │   └── derive_example.rs     # RingMessage derive macro
│   └── advanced/
│       ├── multi_gpu.rs          # Load balancing
│       └── educational_modes.rs  # WaveSim parallel computing modes
│
├── docs/                         # Architecture documentation
│   ├── 01-architecture-overview.md
│   ├── 02-crate-structure.md
│   └── ...
│
└── benches/
    └── serialization.rs
```

---

## Crate Dependencies

```
                    ┌──────────────────┐
                    │   ringkernel     │  (facade - re-exports all)
                    └────────┬─────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ ringkernel- │  │ ringkernel- │  │ ringkernel- │
    │    cpu      │  │    cuda     │  │  ecosystem  │
    │  (working)  │  │  (working)  │  │             │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                 │
           └────────────────┼─────────────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │ ringkernel-core  │
                    │ (traits, types)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌───────────┐  ┌───────────┐  ┌───────────┐
       │ ringkernel│  │ ringkernel│  │   rkyv    │
       │  -derive  │  │ -codegen  │  │ (zero-copy│
       │ (working) │  │           │  │  serde)   │
       └───────────┘  └───────────┘  └───────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ ringkernel-   │
                    │ cuda-codegen  │
                    │ (transpiler)  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ ringkernel-   │
                    │    txmon      │
                    │ (showcase)    │
                    └───────────────┘
```

### Crate Descriptions

| Crate | Status | Description |
|-------|--------|-------------|
| `ringkernel` | Working | Main facade, re-exports everything |
| `ringkernel-core` | Working | Core traits: RingKernel, KernelHandle, HLC, PubSub, K2K |
| `ringkernel-cpu` | Working | CPU backend for development/testing |
| `ringkernel-cuda` | Working | NVIDIA CUDA backend with PTX kernels |
| `ringkernel-metal` | Scaffolded | Apple Metal backend (API defined) |
| `ringkernel-wgpu` | Working | WebGPU cross-platform backend |
| `ringkernel-derive` | Working | Proc macros for message/kernel definitions |
| `ringkernel-codegen` | In Development | GPU kernel code generation |
| `ringkernel-cuda-codegen` | Working | Rust-to-CUDA transpiler for GPU kernels |
| `ringkernel-wgpu-codegen` | Working | Rust-to-WGSL transpiler for GPU kernels |
| `ringkernel-ecosystem` | Working | Integration utilities |
| `ringkernel-audio-fft` | Working | Example: GPU audio FFT processing |
| `ringkernel-wavesim` | Working | Example: 2D wave simulation with FDTD |
| `ringkernel-wavesim3d` | Working | Showcase: 3D wave simulation with binaural audio |
| `ringkernel-txmon` | Working | Showcase: GPU-accelerated transaction monitoring |
| `ringkernel-accnet` | Working | Showcase: Accounting network analytics |
| `ringkernel-procint` | Working | Showcase: Process intelligence with DFG mining |

---

## Cargo.toml (Workspace Root)

```toml
[workspace]
resolver = "2"
members = [
    "crates/ringkernel",
    "crates/ringkernel-core",
    "crates/ringkernel-derive",
    "crates/ringkernel-cuda",
    "crates/ringkernel-metal",
    "crates/ringkernel-wgpu",
    "crates/ringkernel-cpu",
    "crates/ringkernel-codegen",
]

[workspace.package]
version = "0.3.0"
edition = "2021"
rust-version = "1.75"
license = "MIT OR Apache-2.0"
repository = "https://github.com/example/ringkernel"
keywords = ["gpu", "cuda", "actor", "hpc", "compute"]
categories = ["concurrency", "asynchronous", "science"]

[workspace.dependencies]
# Async runtime
tokio = { version = "1.48", features = ["rt-multi-thread", "sync", "macros"] }
async-trait = "0.1"
futures = "0.3"
rayon = "1.11"

# Serialization (zero-copy)
rkyv = { version = "0.7", features = ["validation", "strict"] }
zerocopy = { version = "0.7", features = ["derive"] }
bytemuck = { version = "1.14", features = ["derive"] }

# Error handling
thiserror = "2.0"
anyhow = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# GPU backends
cudarc = { version = "0.18.2", optional = true }       # CUDA (updated API)
metal = { version = "0.31", optional = true }          # Metal
wgpu = { version = "27.0", optional = true }           # WebGPU (Arc-based)

# Web frameworks
axum = { version = "0.8", optional = true }
tower = { version = "0.5", optional = true }
tonic = { version = "0.14", optional = true }          # gRPC
prost = { version = "0.14", optional = true }          # Protobuf

# GUI
iced = { version = "0.13", optional = true }
egui = { version = "0.31", optional = true }
winit = { version = "0.30", optional = true }

# Data
arrow = { version = "54", optional = true }
polars = { version = "0.46", optional = true }
glam = "0.29"

# Proc macros
syn = { version = "2.0", features = ["full", "parsing"] }
quote = "1.0"
proc-macro2 = "1.0"

# Testing
criterion = "0.5"
proptest = "1.4"
```

---

## Feature Flags

```toml
# crates/ringkernel/Cargo.toml
[features]
default = ["cpu"]

# Backends (enable as needed)
cpu = ["ringkernel-cpu"]       # Always available
cuda = ["ringkernel-cuda"]     # Requires NVIDIA GPU + CUDA toolkit

# All backends
full = ["cpu", "cuda"]

# Built-in features (always available in ringkernel-core)
# - telemetry: TelemetryPipeline, MetricsCollector
# - pubsub: PubSubBroker, Topic wildcards
# - hlc: HlcTimestamp, HlcClock
```

### Enabling CUDA

```toml
[dependencies]
ringkernel = { version = "0.1", features = ["cuda"] }
```

Requires:
- NVIDIA GPU with compute capability 3.5+
- CUDA toolkit installed
- `CUDA_PATH` environment variable (or `/usr/local/cuda`)

---

## Platform-Specific Compilation

```toml
# crates/ringkernel-cuda/Cargo.toml
[target.'cfg(any(target_os = "linux", target_os = "windows"))'.dependencies]
cudarc = { workspace = true }

# crates/ringkernel-metal/Cargo.toml
[target.'cfg(any(target_os = "macos", target_os = "ios"))'.dependencies]
metal = { workspace = true }
```

---

## Build Scripts

### CUDA Backend (build.rs)

```rust
// crates/ringkernel-cuda/build.rs
fn main() {
    println!("cargo:rerun-if-changed=src/kernels/");

    // Find CUDA installation
    let cuda_path = std::env::var("CUDA_PATH")
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=nvrtc");

    // Compile PTX at build time (optional)
    #[cfg(feature = "precompile-ptx")]
    compile_ptx_kernels();
}
```

---

## Next: [Core Abstractions](./03-core-abstractions.md)
