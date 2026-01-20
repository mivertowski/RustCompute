//! GPU profiling infrastructure for CUDA backend.
//!
//! This module provides comprehensive profiling capabilities for CUDA kernels:
//!
//! - **CUDA Events**: GPU-side timing without CPU overhead
//! - **NVTX Integration**: Timeline visualization in Nsight Systems/Compute
//! - **Kernel Metrics**: Execution metadata (grid/block dims, occupancy)
//! - **Memory Tracking**: Allocation tracking with leak detection
//! - **Chrome Trace Export**: Rich GPU timeline visualization
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cuda::profiling::{CudaNvtxProfiler, GpuTimer, ProfilingSession};
//!
//! // NVTX profiler for Nsight integration
//! let profiler = CudaNvtxProfiler::new();
//! {
//!     let _range = profiler.push_range("kernel_execution", ProfilerColor::CYAN);
//!     // ... kernel execution ...
//! }
//!
//! // GPU timer for precise timing
//! let mut timer = GpuTimer::new()?;
//! timer.start(stream)?;
//! // ... kernel execution ...
//! timer.stop(stream)?;
//! println!("Kernel time: {:.3} ms", timer.elapsed_ms()?);
//! ```

pub mod chrome_trace;
pub mod events;
pub mod memory_tracker;
pub mod metrics;
pub mod nvtx;

pub use chrome_trace::{GpuChromeTraceBuilder, GpuEventArgs, GpuTraceEvent};
pub use events::{CudaEvent, CudaEventFlags, GpuTimer, GpuTimerPool};
// Note: GpuTimerPool now uses index-based API instead of handles for interior mutability
pub use memory_tracker::{CudaMemoryKind, CudaMemoryTracker, TrackedAllocation};
pub use metrics::{KernelMetrics, ProfilingSession, TransferDirection, TransferMetrics};
pub use nvtx::CudaNvtxProfiler;
