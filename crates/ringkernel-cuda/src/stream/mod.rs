//! Multi-Stream Concurrent Execution Manager.
//!
//! This module enables multi-stream concurrent execution for overlapping
//! compute and transfer operations on CUDA GPUs.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                       StreamManager                                  │
//! │  ┌─────────────────────────────────────────────────────────────┐   │
//! │  │                    CUDA Streams                              │   │
//! │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
//! │  │  │ Compute[0]  │  │ Compute[1]  │  │ Compute[2]  │   ...   │   │
//! │  │  │ Kernel A    │  │ Kernel B    │  │ Kernel C    │         │   │
//! │  │  └─────────────┘  └─────────────┘  └─────────────┘         │   │
//! │  │                                                              │   │
//! │  │  ┌─────────────────────────────────────────────────────┐   │   │
//! │  │  │              Transfer Stream                         │   │   │
//! │  │  │  (Async host-GPU copies overlapped with compute)     │   │   │
//! │  │  └─────────────────────────────────────────────────────┘   │   │
//! │  └─────────────────────────────────────────────────────────────┘   │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Execution Model
//!
//! ```text
//! Time →
//! Stream 0: [Kernel A Scatter][Kernel A Apply]
//! Stream 1: [Kernel B Propagate][Kernel B Apply]  (concurrent with Stream 0)
//! Stream 2: [Kernel C Expand][Kernel C Apply]     (concurrent with Stream 0,1)
//! Transfer: [HtoD Transfer]                        (overlapped with compute)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use ringkernel_cuda::stream::{StreamManager, StreamConfig, StreamId};
//!
//! let manager = StreamManager::new(&device, StreamConfig::default())?;
//!
//! // Launch kernels on different streams
//! let stream_0 = manager.stream(StreamId::Compute(0))?;
//! // ... launch kernel on stream_0 ...
//!
//! // Synchronize specific stream
//! manager.sync_stream(StreamId::Compute(0))?;
//!
//! // Or sync all streams
//! manager.sync_all()?;
//! ```

mod config;
mod manager;
mod metrics;
mod pool;

pub use config::{StreamConfig, StreamConfigBuilder};
pub use manager::{StreamError, StreamId, StreamManager, StreamResult};
pub use metrics::OverlapMetrics;
pub use pool::{StreamPool, StreamPoolStats};
