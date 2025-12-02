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
│   │       ├── queue.rs          # MessageQueue trait
│   │       ├── runtime.rs        # RingKernel, KernelHandle, LaunchOptions
│   │       ├── context.rs        # RingContext struct
│   │       ├── control.rs        # ControlBlock struct
│   │       ├── telemetry.rs      # TelemetryBuffer, MetricsCollector
│   │       ├── pubsub.rs         # PubSubBroker, Topic wildcards
│   │       ├── hlc.rs            # HlcTimestamp, HlcClock
│   │       └── error.rs          # Error types
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
│   │   │   ├── lib.rs
│   │   │   ├── runtime.rs        # CudaRuntime implementation
│   │   │   └── ptx.rs            # PTX template for persistent kernels
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
│   ├── ringkernel-ecosystem/     # Integration utilities
│   │   └── src/lib.rs
│   │
│   └── ringkernel-audio-fft/     # Example: GPU audio processing
│       └── src/lib.rs
│
├── examples/                     # 11 working examples
│   ├── basic/
│   │   ├── hello_kernel.rs       # Runtime, lifecycle, suspend/resume
│   │   └── kernel_states.rs      # State machine, multi-kernel
│   ├── messaging/
│   │   ├── request_response.rs   # Correlation IDs, priorities
│   │   └── pub_sub.rs            # Topic wildcards, QoS
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
│   └── advanced/
│       └── multi_gpu.rs          # Load balancing
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
       │(in dev)   │  │           │  │  serde)   │
       └───────────┘  └───────────┘  └───────────┘
```

### Crate Descriptions

| Crate | Status | Description |
|-------|--------|-------------|
| `ringkernel` | Working | Main facade, re-exports everything |
| `ringkernel-core` | Working | Core traits: RingKernel, KernelHandle, HLC, PubSub |
| `ringkernel-cpu` | Working | CPU backend for development/testing |
| `ringkernel-cuda` | Working | NVIDIA CUDA backend with PTX kernels |
| `ringkernel-metal` | Scaffolded | Apple Metal backend (API defined) |
| `ringkernel-wgpu` | Scaffolded | WebGPU cross-platform backend (API defined) |
| `ringkernel-derive` | In Development | Proc macros for message/kernel definitions |
| `ringkernel-codegen` | In Development | GPU kernel code generation |
| `ringkernel-ecosystem` | Working | Integration utilities |
| `ringkernel-audio-fft` | Working | Example: GPU audio FFT processing |

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
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
license = "MIT OR Apache-2.0"
repository = "https://github.com/example/ringkernel"
keywords = ["gpu", "cuda", "actor", "hpc", "compute"]
categories = ["concurrency", "asynchronous", "science"]

[workspace.dependencies]
# Async runtime
tokio = { version = "1.35", features = ["rt-multi-thread", "sync", "macros"] }
async-trait = "0.1"
futures = "0.3"

# Serialization (zero-copy)
rkyv = { version = "0.7", features = ["validation", "strict"] }
zerocopy = { version = "0.7", features = ["derive"] }
bytemuck = { version = "1.14", features = ["derive"] }

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# GPU backends
cudarc = { version = "0.10", optional = true }         # CUDA
metal = { version = "0.27", optional = true }          # Metal
wgpu = { version = "0.19", optional = true }           # WebGPU
ash = { version = "0.37", optional = true }            # Vulkan

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
