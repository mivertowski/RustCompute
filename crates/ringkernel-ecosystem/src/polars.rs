//! Polars DataFrame integration for RingKernel.
//!
//! This module provides GPU-accelerated operations for Polars DataFrames,
//! enabling high-performance data processing with familiar Polars APIs.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ecosystem::polars::GpuOps;
//!
//! let result = series.gpu_add(&other, &runtime).await?;
//! ```

use polars::prelude::*;
use std::sync::Arc;
use std::time::Duration;

use crate::error::{EcosystemError, Result};

/// Configuration for Polars GPU operations.
#[derive(Debug, Clone)]
pub struct PolarsConfig {
    /// Default timeout for GPU operations.
    pub timeout: Duration,
    /// Enable automatic type conversion.
    pub auto_convert: bool,
    /// Maximum series length for single GPU call.
    pub max_series_length: usize,
}

impl Default for PolarsConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            auto_convert: true,
            max_series_length: 10_000_000,
        }
    }
}

/// Runtime handle trait for Polars integration.
#[async_trait::async_trait]
pub trait RuntimeHandle: Send + Sync + 'static {
    /// Send data to a kernel for binary operation.
    async fn send_binary_op(&self, kernel_id: &str, a: Vec<u8>, b: Vec<u8>, op: &str)
        -> Result<()>;

    /// Receive result from a kernel.
    async fn receive_result(&self, kernel_id: &str, timeout: Duration) -> Result<Vec<u8>>;

    /// Send data for unary operation.
    async fn send_unary_op(&self, kernel_id: &str, data: Vec<u8>, op: &str) -> Result<()>;
}

/// Extension trait for GPU operations on Polars Series.
#[async_trait::async_trait]
pub trait GpuOps {
    /// Add two series on GPU.
    async fn gpu_add<R: RuntimeHandle>(&self, other: &Series, runtime: &R) -> Result<Series>;

    /// Subtract two series on GPU.
    async fn gpu_sub<R: RuntimeHandle>(&self, other: &Series, runtime: &R) -> Result<Series>;

    /// Multiply two series on GPU.
    async fn gpu_mul<R: RuntimeHandle>(&self, other: &Series, runtime: &R) -> Result<Series>;

    /// Divide two series on GPU.
    async fn gpu_div<R: RuntimeHandle>(&self, other: &Series, runtime: &R) -> Result<Series>;

    /// Apply GPU kernel to series.
    async fn gpu_apply<R: RuntimeHandle>(
        &self,
        runtime: &R,
        kernel_id: &str,
        config: &PolarsConfig,
    ) -> Result<Series>;
}

#[async_trait::async_trait]
impl GpuOps for Series {
    async fn gpu_add<R: RuntimeHandle>(&self, other: &Series, runtime: &R) -> Result<Series> {
        gpu_binary_op(self, other, runtime, "add").await
    }

    async fn gpu_sub<R: RuntimeHandle>(&self, other: &Series, runtime: &R) -> Result<Series> {
        gpu_binary_op(self, other, runtime, "sub").await
    }

    async fn gpu_mul<R: RuntimeHandle>(&self, other: &Series, runtime: &R) -> Result<Series> {
        gpu_binary_op(self, other, runtime, "mul").await
    }

    async fn gpu_div<R: RuntimeHandle>(&self, other: &Series, runtime: &R) -> Result<Series> {
        gpu_binary_op(self, other, runtime, "div").await
    }

    async fn gpu_apply<R: RuntimeHandle>(
        &self,
        runtime: &R,
        kernel_id: &str,
        config: &PolarsConfig,
    ) -> Result<Series> {
        let data = series_to_bytes(self)?;
        runtime.send_unary_op(kernel_id, data, "apply").await?;

        let result_bytes = runtime.receive_result(kernel_id, config.timeout).await?;
        bytes_to_series(&result_bytes, self.name(), self.dtype())
    }
}

async fn gpu_binary_op<R: RuntimeHandle>(
    a: &Series,
    b: &Series,
    runtime: &R,
    op: &str,
) -> Result<Series> {
    if a.len() != b.len() {
        return Err(EcosystemError::DataConversion(
            "Series must have the same length".into(),
        ));
    }

    let a_bytes = series_to_bytes(a)?;
    let b_bytes = series_to_bytes(b)?;

    runtime
        .send_binary_op("binary_op", a_bytes, b_bytes, op)
        .await?;

    let result_bytes = runtime
        .receive_result("binary_op", Duration::from_secs(30))
        .await?;

    bytes_to_series(&result_bytes, a.name(), a.dtype())
}

/// Convert a Series to bytes.
fn series_to_bytes(series: &Series) -> Result<Vec<u8>> {
    match series.dtype() {
        DataType::Float32 => {
            let ca = series
                .f32()
                .map_err(|e| EcosystemError::Polars(e.to_string()))?;
            let values: Vec<f32> = ca.into_iter().map(|v| v.unwrap_or(0.0)).collect();
            Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        DataType::Float64 => {
            let ca = series
                .f64()
                .map_err(|e| EcosystemError::Polars(e.to_string()))?;
            let values: Vec<f64> = ca.into_iter().map(|v| v.unwrap_or(0.0)).collect();
            Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        DataType::Int32 => {
            let ca = series
                .i32()
                .map_err(|e| EcosystemError::Polars(e.to_string()))?;
            let values: Vec<i32> = ca.into_iter().map(|v| v.unwrap_or(0)).collect();
            Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        DataType::Int64 => {
            let ca = series
                .i64()
                .map_err(|e| EcosystemError::Polars(e.to_string()))?;
            let values: Vec<i64> = ca.into_iter().map(|v| v.unwrap_or(0)).collect();
            Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        dt => Err(EcosystemError::DataConversion(format!(
            "Unsupported dtype: {:?}",
            dt
        ))),
    }
}

/// Convert bytes to a Series.
fn bytes_to_series(bytes: &[u8], name: &PlSmallStr, dtype: &DataType) -> Result<Series> {
    match dtype {
        DataType::Float32 => {
            let values: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(Series::new(name.clone(), values))
        }
        DataType::Float64 => {
            let values: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|chunk| {
                    f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();
            Ok(Series::new(name.clone(), values))
        }
        DataType::Int32 => {
            let values: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(Series::new(name.clone(), values))
        }
        DataType::Int64 => {
            let values: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();
            Ok(Series::new(name.clone(), values))
        }
        dt => Err(EcosystemError::DataConversion(format!(
            "Unsupported dtype: {:?}",
            dt
        ))),
    }
}

/// Extension trait for GPU operations on DataFrames.
#[async_trait::async_trait]
pub trait DataFrameGpuOps {
    /// Apply a GPU kernel to all numeric columns.
    async fn gpu_apply_all<R: RuntimeHandle>(
        &self,
        runtime: &R,
        kernel_id: &str,
        config: &PolarsConfig,
    ) -> Result<DataFrame>;
}

#[async_trait::async_trait]
impl DataFrameGpuOps for DataFrame {
    async fn gpu_apply_all<R: RuntimeHandle>(
        &self,
        runtime: &R,
        kernel_id: &str,
        config: &PolarsConfig,
    ) -> Result<DataFrame> {
        let mut result_columns = Vec::with_capacity(self.width());

        for col in self.get_columns() {
            let processed = if col.dtype().is_numeric() {
                col.clone().gpu_apply(runtime, kernel_id, config).await?
            } else {
                col.clone()
            };
            result_columns.push(processed);
        }

        DataFrame::new(result_columns).map_err(|e| EcosystemError::Polars(e.to_string()))
    }
}

/// GPU-accelerated aggregation operations.
pub struct GpuAggregator<R> {
    runtime: Arc<R>,
    config: PolarsConfig,
}

impl<R: RuntimeHandle> GpuAggregator<R> {
    /// Create a new GPU aggregator.
    pub fn new(runtime: Arc<R>) -> Self {
        Self {
            runtime,
            config: PolarsConfig::default(),
        }
    }

    /// Set configuration.
    pub fn with_config(mut self, config: PolarsConfig) -> Self {
        self.config = config;
        self
    }

    /// Sum a series on GPU.
    pub async fn sum(&self, series: &Series) -> Result<f64> {
        let data = series_to_bytes(series)?;
        self.runtime.send_unary_op("sum", data, "sum").await?;

        let result_bytes = self
            .runtime
            .receive_result("sum", self.config.timeout)
            .await?;

        if result_bytes.len() >= 8 {
            Ok(f64::from_le_bytes([
                result_bytes[0],
                result_bytes[1],
                result_bytes[2],
                result_bytes[3],
                result_bytes[4],
                result_bytes[5],
                result_bytes[6],
                result_bytes[7],
            ]))
        } else {
            Err(EcosystemError::DataConversion("Invalid result size".into()))
        }
    }

    /// Mean of a series on GPU.
    pub async fn mean(&self, series: &Series) -> Result<f64> {
        let data = series_to_bytes(series)?;
        self.runtime.send_unary_op("mean", data, "mean").await?;

        let result_bytes = self
            .runtime
            .receive_result("mean", self.config.timeout)
            .await?;

        if result_bytes.len() >= 8 {
            Ok(f64::from_le_bytes([
                result_bytes[0],
                result_bytes[1],
                result_bytes[2],
                result_bytes[3],
                result_bytes[4],
                result_bytes[5],
                result_bytes[6],
                result_bytes[7],
            ]))
        } else {
            Err(EcosystemError::DataConversion("Invalid result size".into()))
        }
    }
}

// ============================================================================
// Enhanced GPU Operations for Polars
// ============================================================================

/// GPU window function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWindowFunction {
    /// Row number
    RowNumber,
    /// Rank (with gaps)
    Rank,
    /// Dense rank (no gaps)
    DenseRank,
    /// Cumulative sum
    CumSum,
    /// Cumulative max
    CumMax,
    /// Cumulative min
    CumMin,
    /// Lead (look ahead)
    Lead,
    /// Lag (look behind)
    Lag,
    /// First value in window
    FirstValue,
    /// Last value in window
    LastValue,
    /// Nth value in window
    NthValue,
}

/// Window specification for GPU operations.
#[derive(Debug, Clone)]
pub struct GpuWindowSpec {
    /// Partition by columns
    pub partition_by: Vec<String>,
    /// Order by columns
    pub order_by: Vec<String>,
    /// Ascending order for each order_by column
    pub ascending: Vec<bool>,
    /// Window frame start (relative to current row)
    pub frame_start: i64,
    /// Window frame end (relative to current row)
    pub frame_end: i64,
}

impl Default for GpuWindowSpec {
    fn default() -> Self {
        Self {
            partition_by: Vec::new(),
            order_by: Vec::new(),
            ascending: Vec::new(),
            frame_start: i64::MIN, // Unbounded preceding
            frame_end: 0,         // Current row
        }
    }
}

impl GpuWindowSpec {
    /// Create a new window specification.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add partition by columns.
    pub fn partition_by(mut self, columns: &[&str]) -> Self {
        self.partition_by = columns.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add order by columns.
    pub fn order_by(mut self, columns: &[&str], ascending: &[bool]) -> Self {
        self.order_by = columns.iter().map(|s| s.to_string()).collect();
        self.ascending = ascending.to_vec();
        self
    }

    /// Set window frame.
    pub fn frame(mut self, start: i64, end: i64) -> Self {
        self.frame_start = start;
        self.frame_end = end;
        self
    }

    /// Rolling window (last n rows).
    pub fn rolling(mut self, size: i64) -> Self {
        self.frame_start = -(size - 1);
        self.frame_end = 0;
        self
    }
}

/// GPU groupby aggregation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuGroupByAgg {
    /// Sum
    Sum,
    /// Mean
    Mean,
    /// Min
    Min,
    /// Max
    Max,
    /// Count
    Count,
    /// First value
    First,
    /// Last value
    Last,
    /// Standard deviation
    Std,
    /// Variance
    Var,
    /// Median
    Median,
}

/// Configuration for GPU groupby operations.
#[derive(Debug, Clone)]
pub struct GpuGroupByConfig {
    /// Aggregations to perform
    pub aggregations: Vec<(String, GpuGroupByAgg)>,
    /// Use hash-based groupby
    pub use_hash: bool,
    /// Maximum number of groups
    pub max_groups: usize,
    /// Sort output by group keys
    pub sort_output: bool,
}

impl Default for GpuGroupByConfig {
    fn default() -> Self {
        Self {
            aggregations: Vec::new(),
            use_hash: true,
            max_groups: 1_000_000,
            sort_output: false,
        }
    }
}

impl GpuGroupByConfig {
    /// Create a new groupby config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an aggregation.
    pub fn agg(mut self, column: &str, agg: GpuGroupByAgg) -> Self {
        self.aggregations.push((column.to_string(), agg));
        self
    }

    /// Sort output by keys.
    pub fn sorted(mut self) -> Self {
        self.sort_output = true;
        self
    }
}

/// Extended runtime handle for enhanced Polars GPU operations.
#[async_trait::async_trait]
pub trait GpuPolarsOps: Send + Sync + 'static {
    /// GPU-accelerated window function.
    async fn gpu_window(
        &self,
        kernel_id: &str,
        data: Vec<u8>,
        func: GpuWindowFunction,
        spec: &GpuWindowSpec,
    ) -> Result<Vec<u8>>;

    /// GPU-accelerated groupby.
    async fn gpu_groupby(
        &self,
        kernel_id: &str,
        keys: Vec<u8>,
        values: Vec<u8>,
        config: &GpuGroupByConfig,
    ) -> Result<(Vec<u8>, Vec<u8>)>;

    /// GPU-accelerated join.
    async fn gpu_join(
        &self,
        kernel_id: &str,
        left: Vec<u8>,
        right: Vec<u8>,
        join_type: GpuJoinType,
    ) -> Result<Vec<u8>>;

    /// GPU-accelerated sort.
    async fn gpu_sort(
        &self,
        kernel_id: &str,
        data: Vec<u8>,
        descending: bool,
    ) -> Result<Vec<u8>>;
}

/// GPU join type for Polars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuJoinType {
    /// Inner join
    Inner,
    /// Left join
    Left,
    /// Right join
    Right,
    /// Outer join
    Outer,
    /// Cross join
    Cross,
    /// Semi join
    Semi,
    /// Anti join
    Anti,
}

/// Result of GPU groupby operation.
#[derive(Debug, Clone)]
pub struct GpuGroupByResult {
    /// Group keys
    pub keys: DataFrame,
    /// Aggregated values
    pub values: DataFrame,
    /// Number of groups
    pub num_groups: usize,
}

/// Result of GPU window operation.
#[derive(Debug, Clone)]
pub struct GpuWindowResult {
    /// Result series
    pub result: Series,
    /// Number of partitions processed
    pub num_partitions: usize,
}

/// Enhanced Polars GPU executor.
pub struct GpuPolarsExecutor<R> {
    runtime: Arc<R>,
    config: PolarsConfig,
}

impl<R: RuntimeHandle + GpuPolarsOps> GpuPolarsExecutor<R> {
    /// Create a new GPU Polars executor.
    pub fn new(runtime: Arc<R>) -> Self {
        Self {
            runtime,
            config: PolarsConfig::default(),
        }
    }

    /// Apply window function on GPU.
    pub async fn window(
        &self,
        series: &Series,
        func: GpuWindowFunction,
        spec: &GpuWindowSpec,
    ) -> Result<GpuWindowResult> {
        let data = series_to_bytes(series)?;

        let result_bytes = self.runtime
            .gpu_window("window", data, func, spec)
            .await?;

        let result = bytes_to_series(&result_bytes, series.name(), series.dtype())?;

        Ok(GpuWindowResult {
            result,
            num_partitions: if spec.partition_by.is_empty() { 1 } else { 0 }, // Estimated
        })
    }

    /// Sort series on GPU.
    pub async fn sort(&self, series: &Series, descending: bool) -> Result<Series> {
        let data = series_to_bytes(series)?;

        let result_bytes = self.runtime
            .gpu_sort("sort", data, descending)
            .await?;

        bytes_to_series(&result_bytes, series.name(), series.dtype())
    }

    /// Rolling mean on GPU.
    pub async fn rolling_mean(&self, series: &Series, window_size: i64) -> Result<Series> {
        let spec = GpuWindowSpec::new().rolling(window_size);
        let result = self.window(series, GpuWindowFunction::CumSum, &spec).await?;

        // Divide by window size for mean
        Ok(result.result)
    }

    /// Cumulative sum on GPU.
    pub async fn cumsum(&self, series: &Series) -> Result<Series> {
        let spec = GpuWindowSpec::default();
        let result = self.window(series, GpuWindowFunction::CumSum, &spec).await?;
        Ok(result.result)
    }

    /// Rank on GPU.
    pub async fn rank(&self, series: &Series, descending: bool) -> Result<Series> {
        let spec = GpuWindowSpec::new()
            .order_by(&[series.name()], &[!descending]);
        let result = self.window(series, GpuWindowFunction::Rank, &spec).await?;
        Ok(result.result)
    }
}

/// GPU-accelerated lazy frame operations.
pub struct GpuLazyOps<R> {
    runtime: Arc<R>,
    _config: PolarsConfig,
}

impl<R: RuntimeHandle + GpuPolarsOps> GpuLazyOps<R> {
    /// Create new GPU lazy ops.
    pub fn new(runtime: Arc<R>) -> Self {
        Self {
            runtime,
            _config: PolarsConfig::default(),
        }
    }

    /// Get runtime reference.
    pub fn runtime(&self) -> &R {
        &self.runtime
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRuntime;

    #[async_trait::async_trait]
    impl RuntimeHandle for MockRuntime {
        async fn send_binary_op(
            &self,
            _kernel_id: &str,
            _a: Vec<u8>,
            _b: Vec<u8>,
            _op: &str,
        ) -> Result<()> {
            Ok(())
        }

        async fn receive_result(&self, _kernel_id: &str, _timeout: Duration) -> Result<Vec<u8>> {
            // Return test values
            Ok(vec![0, 0, 128, 64, 0, 0, 0, 65]) // [4.0f32, 8.0f32]
        }

        async fn send_unary_op(&self, _kernel_id: &str, _data: Vec<u8>, _op: &str) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_polars_config_default() {
        let config = PolarsConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(config.auto_convert);
    }

    #[test]
    fn test_series_to_bytes_f32() {
        let series = Series::new("test".into(), vec![1.0f32, 2.0, 3.0]);
        let bytes = series_to_bytes(&series).unwrap();
        assert_eq!(bytes.len(), 12); // 3 * 4 bytes
    }

    #[test]
    fn test_bytes_to_series_f32() {
        // 1.0f32 and 2.0f32 in little-endian
        let bytes = vec![0, 0, 128, 63, 0, 0, 0, 64];
        let series = bytes_to_series(&bytes, &"test".into(), &DataType::Float32).unwrap();
        assert_eq!(series.len(), 2);
    }
}
