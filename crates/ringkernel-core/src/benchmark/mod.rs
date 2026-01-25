//! Benchmark Framework for GPU Performance Tracking.
//!
//! This module provides comprehensive benchmarking, regression detection,
//! and performance validation for GPU workloads.
//!
//! # Key Types
//!
//! - [`Benchmarkable`] - Trait for workloads that can be benchmarked
//! - [`BenchmarkSuite`] - Runs and collects benchmark results
//! - [`BenchmarkResult`] - Single benchmark result with statistics
//! - [`RegressionReport`] - Compares results against baseline
//!
//! # Usage
//!
//! ```ignore
//! use ringkernel_core::benchmark::{BenchmarkSuite, BenchmarkConfig, Benchmarkable};
//!
//! struct MyWorkload { /* ... */ }
//!
//! impl Benchmarkable for MyWorkload {
//!     fn name(&self) -> &str { "my_workload" }
//!     fn code(&self) -> &str { "MW" }
//!     fn execute(&self, config: &WorkloadConfig) -> BenchmarkResult {
//!         // Run workload and return results
//!         // ...
//!     }
//! }
//!
//! let mut suite = BenchmarkSuite::new(BenchmarkConfig::default());
//! suite.run(&MyWorkload { /* ... */ }, &suite.config().clone());
//! println!("{}", suite.generate_markdown_report());
//! ```

mod config;
mod regression;
mod result;
mod statistics;
mod suite;
mod traits;

pub use config::BenchmarkConfig;
pub use regression::{RegressionEntry, RegressionReport, RegressionStatus};
pub use result::{BenchmarkBaseline, BenchmarkResult, WorkloadSize};
pub use statistics::{ConfidenceInterval, DetailedStatistics, ScalingMetrics};
pub use suite::BenchmarkSuite;
pub use traits::{Benchmarkable, WorkloadConfig};
