//! RingKernel Transaction Monitoring (TxMon)
//!
//! GPU-accelerated real-time transaction monitoring showcase demonstrating
//! RingKernel capabilities for banking/AML compliance.
//!
//! # Features
//!
//! - **Transaction Factory**: Configurable synthetic transaction generation
//! - **Compliance Rules**: Velocity breach, amount threshold, structuring, geographic anomaly
//! - **Real-time GUI**: Live transaction feed, alerts panel, statistics dashboard
//! - **Multi-backend**: CPU and optional CUDA GPU acceleration
//!
//! # GPU Backend Approaches
//!
//! The `cuda` module provides three different GPU-accelerated approaches:
//!
//! ## 1. Batch Kernel (High Throughput)
//! Each thread processes one transaction independently. Best for maximum throughput
//! on large batches (10M+ TPS on modern GPUs).
//!
//! ## 2. Ring Kernel (Actor-based Streaming)
//! Persistent GPU kernels with HLC timestamps and K2K messaging. Best for low-latency
//! streaming with complex multi-stage pipelines.
//!
//! ## 3. Stencil Kernel (Pattern Detection)
//! Uses GridPos for spatial pattern detection in transaction networks. Detects circular
//! trading, velocity anomalies, and coordinated activity patterns.

pub mod factory;
pub mod gui;
pub mod monitoring;
pub mod types;

#[cfg(feature = "cuda-codegen")]
pub mod cuda;

pub use factory::{FactoryState, GeneratorConfig, TransactionGenerator};
pub use monitoring::{MonitoringConfig, MonitoringEngine};
pub use types::{
    AlertSeverity, AlertStatus, AlertType, CustomerRiskLevel, CustomerRiskProfile, MonitoringAlert,
    Transaction, TransactionType,
};
