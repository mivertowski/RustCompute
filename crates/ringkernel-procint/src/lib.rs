//! # RingKernel Process Intelligence
//!
//! GPU-accelerated process mining and intelligence with real-time visualization.
//!
//! This crate provides tools for analyzing business processes through:
//!
//! - **DFG Construction**: Build Directly-Follows Graphs from event streams
//! - **Partial Order Mining**: Discover concurrent activity patterns
//! - **Pattern Detection**: Identify bottlenecks, loops, rework, and anomalies
//! - **Conformance Checking**: Validate traces against reference models
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
//! │   Data Fabric   │────▶│   GPU Kernels    │────▶│  Visualization │
//! │ (Event Stream)  │     │ (CUDA/WGSL)      │     │  (egui Canvas) │
//! └─────────────────┘     └──────────────────┘     └────────────────┘
//!         │                        │                       │
//!         ▼                        ▼                       ▼
//! ┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
//! │ Sector Templates│     │ DFG Construction │     │ Force Layout   │
//! │ Event Generator │     │ Pattern Detection│     │ Token Animation│
//! │ Anomaly Inject  │     │ Conformance Check│     │ Timeline View  │
//! └─────────────────┘     └──────────────────┘     └────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use ringkernel_procint::prelude::*;
//!
//! // Create event generator for healthcare sector
//! let config = GeneratorConfig::default()
//!     .with_sector(SectorTemplate::Healthcare)
//!     .with_events_per_second(10_000);
//! let mut generator = ProcessEventGenerator::new(config);
//!
//! // Generate events and build DFG
//! let events = generator.generate_batch(1000);
//! let dfg = DFGBuilder::new().build_from_events(&events);
//!
//! // Detect patterns
//! let patterns = PatternDetector::new().detect(&dfg);
//! ```
//!
//! ## GPU Kernel Types
//!
//! 1. **DFG Construction** - Batch kernel for edge frequency counting
//! 2. **Partial Order Derivation** - Stencil kernel for precedence matrix
//! 3. **Pattern Detection** - Batch kernel for bottleneck/loop/rework detection
//! 4. **Conformance Checking** - Batch kernel for fitness scoring

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod actors;
pub mod analytics;
pub mod cuda;
pub mod fabric;
pub mod gui;
pub mod kernels;
pub mod models;

/// Prelude for convenient imports.
pub mod prelude {
    // Core types
    pub use crate::models::{
        Activity, ActivityId, AlignmentMove, AlignmentType, ComplianceLevel, ConformanceResult,
        ConformanceStatus, DFGGraph, EventType, GpuDFGEdge, GpuDFGGraph, GpuDFGNode,
        GpuObjectEvent, GpuPartialOrderTrace, GpuPatternMatch, HybridTimestamp, PatternSeverity,
        PatternType, ProcessModel, ProcessTrace, TraceId,
    };

    // Data fabric
    pub use crate::fabric::{
        AnomalyConfig, GeneratorConfig, GeneratorStats, PipelineConfig, ProcessEventGenerator,
        ProcessingPipeline, SectorTemplate,
    };

    // Analytics
    pub use crate::analytics::{
        AnalyticsEngine, DFGMetrics, DFGMetricsCalculator, KPITracker, PatternAggregator,
        ProcessKPIs,
    };

    // Actors
    pub use crate::actors::{GpuActorRuntime, PipelineCoordinator, PipelineStats, RuntimeConfig};

    // Kernels
    pub use crate::kernels::{
        ConformanceKernel, DfgConstructionKernel, PartialOrderKernel, PatternDetectionKernel,
    };
}

/// Version information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Crate name.
pub const NAME: &str = env!("CARGO_PKG_NAME");
