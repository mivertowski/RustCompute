//! GPU-native actor system for accounting network analytics.
//!
//! This module implements the RingKernel actor model for accounting analytics,
//! with persistent GPU kernels processing messages in a pipeline:
//!
//! ```text
//! ┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
//! │   Flow      │──▶│   PageRank   │──▶│   Fraud     │──▶│   Results    │
//! │  Generator  │   │   Computer   │   │  Detector   │   │  Aggregator  │
//! └─────────────┘   └──────────────┘   └─────────────┘   └──────────────┘
//!       ▲                  │                  │                  │
//!       │                  ▼                  ▼                  ▼
//!       │           ┌──────────────┐   ┌─────────────┐          │
//!       │           │   GAAP       │   │   Benford   │          │
//!       │           │   Validator  │   │   Analyzer  │          │
//!       │           └──────────────┘   └─────────────┘          │
//!       │                                                        │
//!       └────────────────── K2K Messaging ──────────────────────┘
//! ```
//!
//! ## Kernel Actors
//!
//! - **FlowGeneratorKernel**: Generates transaction flows from journal entries
//! - **PageRankKernel**: Computes PageRank scores for account influence
//! - **FraudDetectorKernel**: Detects fraud patterns (circular flows, velocity, etc.)
//! - **GaapValidatorKernel**: Validates GAAP compliance rules
//! - **BenfordAnalyzerKernel**: Performs Benford's Law analysis
//! - **ResultsAggregatorKernel**: Aggregates results and sends to host

pub mod coordinator;
#[cfg(feature = "cuda")]
pub mod kernels;
pub mod messages;
pub mod runtime;

pub use coordinator::*;
#[cfg(feature = "cuda")]
pub use kernels::*;
pub use messages::*;
pub use runtime::*;
