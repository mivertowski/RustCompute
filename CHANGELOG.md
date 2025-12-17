# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2025-12-14

### Added

#### Cooperative Groups Support
- **Grid-wide GPU synchronization** via CUDA cooperative groups (`grid.sync()`)
- **`cuLaunchCooperativeKernel` driver API interop** - Direct FFI calls to CUDA driver for true cooperative launch
- **Build-time PTX compilation** - `build.rs` with nvcc detection and automatic kernel compilation
- **`cooperative` feature flag** for `ringkernel-cuda` and `ringkernel-wavesim3d`
- **`cooperative` field in `LaunchOptions`** for cooperative launch mode

#### Block Actor Backend (WaveSim3D)
- **8×8×8 block-based actor model** - Hybrid approach combining stencil and actor patterns
  - Intra-block: Fast stencil computation with shared memory
  - Inter-block: Double-buffered message passing (no atomics)
- **`BlockActorGpuBackend`** with `step_fused()` for single-kernel-launch execution
- **Performance**: 8,165 Mcells/s (59.6× faster than per-cell actors)
- **Grid size validation** with `max_cooperative_blocks` (144 on RTX 4090)

#### New Computation Method
- **`ComputationMethod::BlockActor`** - Third GPU computation method for wavesim3d
  - Combines actor model benefits with stencil performance
  - 10-50× faster than per-cell Actor method

### Changed
- Added `CooperativeKernel` wrapper in `ringkernel-cuda::cooperative` module
- Added cooperative kernel infrastructure to wavesim3d benchmark

#### Dependency Updates
- **tokio**: 1.35 → 1.48 (improved task scheduling, better cancellation handling)
- **thiserror**: 1.0 → 2.0 (updated derive macros)
- **wgpu**: 0.19 → 27.0 (Arc-based resource tracking, 40%+ performance improvement)
  - Migrated to new Instance/Adapter/Device creation API
  - Updated pipeline descriptors with `entry_point: Option<&str>`, `compilation_options`, `cache`
  - Renamed `ImageCopyTexture` → `TexelCopyTextureInfo`, `ImageDataLayout` → `TexelCopyBufferLayout`
  - Updated `device.poll()` to use `PollType::wait_indefinitely()`
- **winit**: 0.29 → 0.30 (new window creation API)
- **egui/egui-wgpu/egui-winit**: 0.27 → 0.31 (updated for wgpu 27 compatibility)
- **glam**: 0.27 → 0.29 (linear algebra updates)
- **metal**: 0.27 → 0.31 (Apple GPU backend updates)
- **axum**: 0.7 → 0.8 (improved routing, better error handling)
- **tower**: 0.4 → 0.5 (service abstraction updates)
- **tonic**: 0.11 → 0.14 (better gRPC streaming, improved health checking)
- **prost**: 0.12 → 0.14 (protobuf updates to match tonic)
- **actix-rt**: 2.9 → 2.10
- **rayon**: 1.10 → 1.11 (requires MSRV 1.80)
- **arrow**: 52 → 54 (columnar data updates)
- **polars**: 0.39 → 0.46 (DataFrame updates)

#### Deferred Updates
- **iced**: Kept at 0.13 (0.14 requires major application API rewrite)
- **rkyv**: Kept at 0.7 (0.8 has incompatible data format, requires significant migration)

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

#### CUDA Codegen Intrinsics Expansion
- Expanded GPU intrinsics from ~45 to **120+ operations** across 13 categories
- **Atomic Operations** (11 ops): `atomic_add`, `atomic_sub`, `atomic_min`, `atomic_max`, `atomic_exchange`, `atomic_cas`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_inc`, `atomic_dec`
- **Synchronization** (7 ops): `sync_threads`, `sync_threads_count`, `sync_threads_and`, `sync_threads_or`, `thread_fence`, `thread_fence_block`, `thread_fence_system`
- **Trigonometric** (11 ops): `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sincos`, `sinpi`, `cospi`
- **Hyperbolic** (6 ops): `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- **Exponential/Logarithmic** (18 ops): `exp`, `exp2`, `exp10`, `expm1`, `log`, `ln`, `log2`, `log10`, `log1p`, `pow`, `ldexp`, `scalbn`, `ilogb`, `erf`, `erfc`, `erfinv`, `erfcinv`, `lgamma`, `tgamma`
- **Classification** (8 ops): `is_nan`, `is_infinite`, `is_finite`, `is_normal`, `signbit`, `nextafter`, `fdim`
- **Warp Operations** (16 ops): `warp_active_mask`, `warp_shfl`, `warp_shfl_up`, `warp_shfl_down`, `warp_shfl_xor`, `warp_ballot`, `warp_all`, `warp_any`, `warp_match_any`, `warp_match_all`, `warp_reduce_add/min/max/and/or/xor`
- **Bit Manipulation** (8 ops): `popc`, `clz`, `ctz`, `ffs`, `brev`, `byte_perm`, `funnel_shift_left`, `funnel_shift_right`
- **Memory Operations** (3 ops): `ldg`, `prefetch_l1`, `prefetch_l2`
- **Special Functions** (13 ops): `rcp`, `fast_div`, `saturate`, `j0`, `j1`, `jn`, `y0`, `y1`, `yn`, `normcdf`, `normcdfinv`, `cyl_bessel_i0`, `cyl_bessel_i1`
- **Timing** (3 ops): `clock`, `clock64`, `nanosleep`
- **3D Stencil Intrinsics**: `pos.up(buf)`, `pos.down(buf)`, `pos.at(buf, dx, dy, dz)` for volumetric kernels

### Changed
- Added `required-features` to CUDA-only wavesim binaries to fix build without CUDA
- Updated GitHub Actions release workflow with proper feature flags and Ubuntu version
- Updated ringkernel-cuda-codegen tests from 143 to 171 tests

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

[Unreleased]: https://github.com/mivertowski/RustCompute/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/mivertowski/RustCompute/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/mivertowski/RustCompute/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/mivertowski/RustCompute/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mivertowski/RustCompute/releases/tag/v0.1.0
