# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-12-03

### Added

#### Core Framework
- GPU-native persistent actor model with `RingKernelRuntime` trait
- Lock-free `MessageQueue` (SPSC ring buffer) for host-GPU message passing
- `ControlBlock` - 128-byte GPU-resident structure for kernel lifecycle management
- `RingContext` - GPU intrinsics facade for kernel handlers
- Hybrid Logical Clocks (`HlcTimestamp`, `HlcClock`) for causal ordering across distributed kernels
- `KernelHandle` for managing kernel lifecycle (launch, activate, terminate)

#### Messaging
- `RingMessage` trait with zero-copy serialization via rkyv
- Kernel-to-Kernel (K2K) direct messaging with `K2KBroker` and `K2KEndpoint`
- Topic-based Publish/Subscribe with wildcard support via `PubSubBroker`
- Message correlation tracking and priority support

#### Procedural Macros (`ringkernel-derive`)
- `#[derive(RingMessage)]` - Automatic message serialization with field annotations
- `#[ring_kernel]` - Kernel handler definition with configuration
- `#[derive(GpuType)]` - GPU-compatible type generation

#### Backend Support
- **CPU Backend** (`ringkernel-cpu`) - Always available for testing and fallback
- **CUDA Backend** (`ringkernel-cuda`) - NVIDIA GPU support via cudarc
- **WebGPU Backend** (`ringkernel-wgpu`) - Cross-platform GPU support (Vulkan, Metal, DX12)
- **Metal Backend** (`ringkernel-metal`) - Apple GPU support (scaffolded)
- Auto-detection with `Backend::Auto` (tries CUDA → Metal → WebGPU → CPU)

#### Code Generation
- **CUDA Codegen** (`ringkernel-cuda-codegen`) - Rust DSL to CUDA C transpiler
  - Global kernels with block/grid indices
  - Stencil kernels with `GridPos` abstraction and tiled shared memory
  - Ring kernels for persistent actor model with HLC and K2K support
  - 45+ GPU intrinsics (atomics, warp ops, sync, math)
- **WGSL Codegen** (`ringkernel-wgpu-codegen`) - Rust DSL to WGSL transpiler
  - Full parity with CUDA codegen for portable shaders
  - 64-bit emulation via lo/hi u32 pairs
  - Subgroup operations support

#### Ecosystem Integrations (`ringkernel-ecosystem`)
- Actor framework integrations (Actix, Tower)
- Web framework integrations (Axum)
- Data processing (Arrow, Polars)
- gRPC support (Tonic)
- Machine learning (Candle)
- Configuration management
- Metrics and observability (Prometheus, tracing)

#### Example Applications
- **WaveSim** (`ringkernel-wavesim`) - Interactive 2D acoustic wave simulation
  - FDTD solver with GPU acceleration
  - Educational modes demonstrating parallel computing evolution
  - Multiple backends (CPU, CUDA, WebGPU)
- **TxMon** (`ringkernel-txmon`) - Real-time transaction monitoring
  - GPU-accelerated fraud detection patterns
  - Structuring detection, velocity checks, PEP monitoring
  - Interactive GUI with real-time visualization
- **Audio FFT** (`ringkernel-audio-fft`) - GPU-accelerated audio processing
  - Direct/ambience source separation
  - Real-time FFT processing with actor model

### Performance
- CUDA codegen achieves ~93B elem/sec on RTX Ada (12,378x vs CPU)
- Lock-free message queue with sub-microsecond latency
- Zero-copy serialization for GPU transfer

### Documentation
- Comprehensive README files for all crates
- CLAUDE.md with build commands and architecture overview
- Code examples for all major features

[Unreleased]: https://github.com/mivertowski/RustCompute/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mivertowski/RustCompute/releases/tag/v0.1.0
