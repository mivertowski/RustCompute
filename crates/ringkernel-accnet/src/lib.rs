//! # RingKernel Accounting Network Analytics
//!
//! GPU-accelerated accounting network analysis with real-time visualization.
//!
//! This crate transforms traditional double-entry bookkeeping into a graph
//! representation, enabling advanced analytics like:
//!
//! - **Fraud Pattern Detection**: Circular flows, threshold clustering, Benford violations
//! - **GAAP Compliance**: Automated violation detection for accounting rules
//! - **Behavioral Anomalies**: Time-series based anomaly detection
//! - **Network Metrics**: Centrality, PageRank, community detection
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
//! │   Data Fabric   │────▶│  GPU Kernels     │────▶│  Visualization │
//! │ (Synthetic Gen) │     │ (CUDA/WGSL)      │     │  (egui Canvas) │
//! └─────────────────┘     └──────────────────┘     └────────────────┘
//!         │                        │                       │
//!         ▼                        ▼                       ▼
//! ┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
//! │ Journal Entries │     │ Network Analysis │     │ Graph Layout   │
//! │ Transaction Gen │     │ Fraud Detection  │     │ Flow Animation │
//! │ Anomaly Inject  │     │ Temporal Analysis│     │ Analytics UI   │
//! └─────────────────┘     └──────────────────┘     └────────────────┘
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use ringkernel_accnet::prelude::*;
//!
//! // Create a network
//! let mut network = AccountingNetwork::new(entity_id, 2024, 1);
//!
//! // Add accounts
//! let cash = network.add_account(
//!     AccountNode::new(Uuid::new_v4(), AccountType::Asset, 0),
//!     AccountMetadata::new("1100", "Cash")
//! );
//!
//! // Add flows
//! network.add_flow(TransactionFlow::new(
//!     source, target, amount, journal_id, timestamp
//! ));
//!
//! // Run analysis
//! network.calculate_pagerank(0.85, 20);
//! ```
//!
//! ## GPU Kernel Types
//!
//! 1. **Journal Transformation** - Methods A-E for converting entries to flows
//! 2. **Network Analysis** - Suspense detection, GAAP violations, fraud patterns
//! 3. **Temporal Analysis** - Seasonality, trends, behavioral anomalies

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
    pub use crate::models::{
        AccountFlags,
        AccountMetadata,
        // Core types
        AccountNode,
        AccountSemantics,
        AccountType,
        // Network
        AccountingNetwork,
        AggregatedFlow,
        BalanceSide,
        // Temporal
        BehavioralBaseline,
        BookingPatternType,
        Decimal128,
        FlowDirection,
        FlowFlags,
        // Patterns
        FraudPattern,
        FraudPatternType,
        GaapViolation,
        GaapViolationRule,
        GaapViolationType,
        GpuNetworkHeader,
        GraphEdge,
        HybridTimestamp,
        // Journal entries
        JournalEntry,
        JournalEntryFlags,
        JournalLineItem,
        LineItemFlags,
        LineType,
        NetworkSnapshot,
        NetworkStatistics,
        SeasonalPattern,
        SeasonalityType,
        SolvingMethod,
        TemporalAlert,
        TemporalAlertType,
        TimeGranularity,
        TimeSeriesMetrics,
        // Flows
        TransactionFlow,
        ViolationSeverity,
    };

    pub use crate::fabric::{
        AnomalyInjectionConfig, ChartOfAccountsTemplate, CompanyArchetype, DataFabricPipeline,
        GeneratorConfig, PipelineConfig, PipelineEvent, TransactionGenerator,
    };

    pub use crate::analytics::{AnalyticsEngine, AnalyticsSnapshot, RiskScore};

    pub use crate::actors::{CoordinatorConfig, GpuActorRuntime, GpuAnalyticsResult};
}

/// Version information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Crate name.
pub const NAME: &str = env!("CARGO_PKG_NAME");
