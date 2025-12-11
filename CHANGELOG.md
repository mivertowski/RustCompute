# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2025-12-11

### Added

#### New Crate
- **WaveSim3D** (`ringkernel-wavesim3d`) - 3D acoustic wave simulation with realistic physics
  - Full 3D FDTD (Finite-Difference Time-Domain) wave propagation solver
  - Binaural audio rendering with HRTF (Head-Related Transfer Function) support
  - Volumetric ray marching visualization for real-time 3D pressure field rendering
  - GPU-native actor system for distributed 3D wave simulation
  - Support for multiple sound sources with frequency-dependent propagation
  - Material absorption modeling with frequency-dependent coefficients
  - Interactive 3D camera controls and visualization modes

### Changed
- Added `required-features` to CUDA-only wavesim binaries to fix build without CUDA
- Updated GitHub Actions release workflow with proper feature flags and Ubuntu version

### Fixed
- Fixed release workflow feature flags for showcase applications
- Fixed Ubuntu version compatibility in CI/CD pipeline

## [0.1.1] - 2025-12-04

### Added

#### New Showcase Applications
- **AccNet** (`ringkernel-accnet`) - GPU-accelerated accounting network analytics
  - Network visualization with force-directed graph layout
  - Fraud detection: circular flows, threshold clustering, Benford's Law violations
  - GAAP compliance checking for accounting rule violations
  - Temporal analysis for seasonality, trends, and behavioral anomalies
  - GPU kernels: Suspense detection, GAAP violation, Benford analysis, PageRank
- **ProcInt** (`ringkernel-procint`) - GPU-accelerated process intelligence
  - DFG (Directly-Follows Graph) mining from event streams
  - Pattern detection: bottlenecks, loops, rework, long-running activities
  - Conformance checking with fitness and precision metrics
  - Timeline view with partial order traces and concurrent activity visualization
  - Multi-sector templates: Healthcare, Manufacturing, Finance, IT
  - GPU kernels: DFG construction, pattern detection, partial order derivation, conformance checking

### Changed
- Updated showcase documentation with AccNet and ProcInt sections
- Updated CI workflow to exclude CUDA tests on runners without GPU hardware

### Fixed
- Fixed 14 clippy warnings in ringkernel-accnet (needless_range_loop, manual_range_contains, clamp patterns, etc.)
- Fixed benchmark API compatibility in ringkernel-accnet
- Fixed code formatting issues across showcase applications

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

[Unreleased]: https://github.com/mivertowski/RustCompute/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/mivertowski/RustCompute/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/mivertowski/RustCompute/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mivertowski/RustCompute/releases/tag/v0.1.0
