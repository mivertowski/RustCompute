//! Kernel and transfer metrics for performance analysis.
//!
//! This module provides structures for capturing and analyzing GPU execution
//! metadata including kernel launch parameters, execution times, and memory
//! transfer statistics.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cuda::profiling::{KernelMetrics, ProfilingSession};
//!
//! let mut session = ProfilingSession::new();
//!
//! // Record a kernel execution
//! session.record_kernel(KernelMetrics {
//!     kernel_name: "vector_add".to_string(),
//!     gpu_time_ms: 0.5,
//!     grid_dim: (128, 1, 1),
//!     block_dim: (256, 1, 1),
//!     ..Default::default()
//! });
//!
//! // Get statistics
//! let stats = session.statistics();
//! println!("Total kernel time: {:.3} ms", stats.total_kernel_time_ms);
//! ```

use std::collections::HashMap;
use std::time::Instant;

use cudarc::driver::sys as cuda_sys;

use ringkernel_core::error::{Result, RingKernelError};

/// Metrics for a single kernel execution.
#[derive(Debug, Clone, Default)]
pub struct KernelMetrics {
    /// Name of the kernel function.
    pub kernel_name: String,
    /// GPU execution time in milliseconds (from CUDA events).
    pub gpu_time_ms: f32,
    /// CPU-side wall clock time in microseconds (fallback).
    pub cpu_time_us: u64,
    /// Grid dimensions (blocks in x, y, z).
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (threads per block in x, y, z).
    pub block_dim: (u32, u32, u32),
    /// Dynamic shared memory per block in bytes.
    pub shared_mem_bytes: u32,
    /// Registers used per thread (from cuFuncGetAttribute).
    pub registers_per_thread: Option<u32>,
    /// Theoretical occupancy (0.0 - 1.0).
    pub occupancy: Option<f32>,
    /// Static shared memory per block in bytes.
    pub static_shared_mem: Option<u32>,
    /// Local memory per thread in bytes.
    pub local_mem_per_thread: Option<u32>,
    /// GPU device index.
    pub device_index: u32,
    /// Stream ID (for multi-stream profiling).
    pub stream_id: Option<u64>,
    /// Timestamp when the kernel was launched (microseconds since session start).
    pub launch_timestamp_us: u64,
}

impl KernelMetrics {
    /// Create new kernel metrics with basic information.
    pub fn new(kernel_name: impl Into<String>) -> Self {
        Self {
            kernel_name: kernel_name.into(),
            ..Default::default()
        }
    }

    /// Set grid dimensions.
    pub fn with_grid_dim(mut self, x: u32, y: u32, z: u32) -> Self {
        self.grid_dim = (x, y, z);
        self
    }

    /// Set block dimensions.
    pub fn with_block_dim(mut self, x: u32, y: u32, z: u32) -> Self {
        self.block_dim = (x, y, z);
        self
    }

    /// Set shared memory size.
    pub fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Set GPU time.
    pub fn with_gpu_time(mut self, ms: f32) -> Self {
        self.gpu_time_ms = ms;
        self
    }

    /// Calculate total threads launched.
    pub fn total_threads(&self) -> u64 {
        let grid_size = self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64;
        let block_size =
            self.block_dim.0 as u64 * self.block_dim.1 as u64 * self.block_dim.2 as u64;
        grid_size * block_size
    }

    /// Calculate total blocks launched.
    pub fn total_blocks(&self) -> u64 {
        self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64
    }

    /// Calculate threads per block.
    pub fn threads_per_block(&self) -> u32 {
        self.block_dim.0 * self.block_dim.1 * self.block_dim.2
    }

    /// Calculate throughput in millions of threads per second.
    pub fn throughput_mthreads_per_sec(&self) -> Option<f64> {
        if self.gpu_time_ms > 0.0 {
            Some(self.total_threads() as f64 / (self.gpu_time_ms as f64 * 1000.0))
        } else {
            None
        }
    }
}

/// Direction of a memory transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TransferDirection {
    /// Host to device (upload).
    #[default]
    HostToDevice,
    /// Device to host (download).
    DeviceToHost,
    /// Device to device (peer copy).
    DeviceToDevice,
}

impl TransferDirection {
    /// Get a short label for the direction.
    pub fn label(&self) -> &'static str {
        match self {
            TransferDirection::HostToDevice => "H2D",
            TransferDirection::DeviceToHost => "D2H",
            TransferDirection::DeviceToDevice => "D2D",
        }
    }
}

/// Metrics for a memory transfer operation.
#[derive(Debug, Clone, Default)]
pub struct TransferMetrics {
    /// Transfer direction.
    pub direction: TransferDirection,
    /// Size of the transfer in bytes.
    pub size_bytes: usize,
    /// GPU execution time in milliseconds.
    pub gpu_time_ms: f32,
    /// CPU-side wall clock time in microseconds.
    pub cpu_time_us: u64,
    /// Source device index (-1 for host).
    pub src_device: i32,
    /// Destination device index (-1 for host).
    pub dst_device: i32,
    /// Whether this was an async transfer.
    pub is_async: bool,
    /// Stream ID (for async transfers).
    pub stream_id: Option<u64>,
    /// Timestamp when the transfer started (microseconds since session start).
    pub timestamp_us: u64,
}

impl TransferMetrics {
    /// Create new transfer metrics.
    pub fn new(direction: TransferDirection, size_bytes: usize) -> Self {
        Self {
            direction,
            size_bytes,
            ..Default::default()
        }
    }

    /// Set GPU time.
    pub fn with_gpu_time(mut self, ms: f32) -> Self {
        self.gpu_time_ms = ms;
        self
    }

    /// Calculate effective bandwidth in GB/s.
    pub fn bandwidth_gbps(&self) -> f64 {
        if self.gpu_time_ms > 0.0 {
            let bytes_per_sec = self.size_bytes as f64 / (self.gpu_time_ms as f64 / 1000.0);
            bytes_per_sec / 1_000_000_000.0
        } else {
            0.0
        }
    }

    /// Calculate effective bandwidth in MB/s.
    pub fn bandwidth_mbps(&self) -> f64 {
        self.bandwidth_gbps() * 1000.0
    }
}

/// Snapshot of memory state at a point in time.
#[derive(Debug, Clone, Default)]
pub struct MemorySnapshot {
    /// Timestamp (microseconds since session start).
    pub timestamp_us: u64,
    /// Total allocated bytes.
    pub allocated_bytes: u64,
    /// Peak allocated bytes so far.
    pub peak_bytes: u64,
    /// Number of active allocations.
    pub allocation_count: usize,
    /// Free memory on the device.
    pub free_bytes: Option<u64>,
    /// Total memory on the device.
    pub total_bytes: Option<u64>,
}

/// Statistics computed from a profiling session.
#[derive(Debug, Clone, Default)]
pub struct ProfilingStatistics {
    /// Total kernel execution time (GPU) in milliseconds.
    pub total_kernel_time_ms: f64,
    /// Total transfer time in milliseconds.
    pub total_transfer_time_ms: f64,
    /// Total bytes transferred (all directions).
    pub total_bytes_transferred: u64,
    /// Number of kernel executions.
    pub kernel_count: usize,
    /// Number of transfer operations.
    pub transfer_count: usize,
    /// Average kernel time in milliseconds.
    pub avg_kernel_time_ms: f64,
    /// Average transfer bandwidth in GB/s.
    pub avg_transfer_bandwidth_gbps: f64,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: u64,
    /// Per-kernel statistics.
    pub kernel_stats: HashMap<String, KernelStatistics>,
    /// Per-direction transfer statistics.
    pub transfer_stats: HashMap<TransferDirection, TransferStatistics>,
}

/// Statistics for a specific kernel.
#[derive(Debug, Clone, Default)]
pub struct KernelStatistics {
    /// Number of invocations.
    pub invocations: usize,
    /// Total GPU time in milliseconds.
    pub total_time_ms: f64,
    /// Minimum GPU time in milliseconds.
    pub min_time_ms: f32,
    /// Maximum GPU time in milliseconds.
    pub max_time_ms: f32,
    /// Average GPU time in milliseconds.
    pub avg_time_ms: f64,
}

/// Statistics for a transfer direction.
#[derive(Debug, Clone, Default)]
pub struct TransferStatistics {
    /// Number of transfers.
    pub count: usize,
    /// Total bytes transferred.
    pub total_bytes: u64,
    /// Total time in milliseconds.
    pub total_time_ms: f64,
    /// Average bandwidth in GB/s.
    pub avg_bandwidth_gbps: f64,
}

/// A profiling session that collects kernel and transfer metrics.
#[derive(Debug)]
pub struct ProfilingSession {
    /// Kernel execution events (timestamp_us, metrics).
    kernel_events: Vec<(u64, KernelMetrics)>,
    /// Transfer events (timestamp_us, metrics).
    transfer_events: Vec<(u64, TransferMetrics)>,
    /// Memory snapshots.
    memory_snapshots: Vec<MemorySnapshot>,
    /// Session start time.
    start_time: Instant,
    /// Session name/label.
    name: String,
    /// Additional metadata.
    metadata: HashMap<String, String>,
}

impl ProfilingSession {
    /// Create a new profiling session.
    pub fn new() -> Self {
        Self {
            kernel_events: Vec::new(),
            transfer_events: Vec::new(),
            memory_snapshots: Vec::new(),
            start_time: Instant::now(),
            name: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a new named profiling session.
    pub fn with_name(name: impl Into<String>) -> Self {
        let mut session = Self::new();
        session.name = name.into();
        session
    }

    /// Get the session name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add metadata to the session.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get metadata by key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Get elapsed time since session start in microseconds.
    pub fn elapsed_us(&self) -> u64 {
        self.start_time.elapsed().as_micros() as u64
    }

    /// Record a kernel execution.
    pub fn record_kernel(&mut self, mut metrics: KernelMetrics) {
        let timestamp = self.elapsed_us();
        metrics.launch_timestamp_us = timestamp;
        self.kernel_events.push((timestamp, metrics));
    }

    /// Record a memory transfer.
    pub fn record_transfer(&mut self, mut metrics: TransferMetrics) {
        let timestamp = self.elapsed_us();
        metrics.timestamp_us = timestamp;
        self.transfer_events.push((timestamp, metrics));
    }

    /// Record a memory snapshot.
    pub fn record_memory_snapshot(&mut self, mut snapshot: MemorySnapshot) {
        snapshot.timestamp_us = self.elapsed_us();
        self.memory_snapshots.push(snapshot);
    }

    /// Get the number of kernel events.
    pub fn kernel_count(&self) -> usize {
        self.kernel_events.len()
    }

    /// Get the number of transfer events.
    pub fn transfer_count(&self) -> usize {
        self.transfer_events.len()
    }

    /// Get all kernel events.
    pub fn kernel_events(&self) -> &[(u64, KernelMetrics)] {
        &self.kernel_events
    }

    /// Get all transfer events.
    pub fn transfer_events(&self) -> &[(u64, TransferMetrics)] {
        &self.transfer_events
    }

    /// Get all memory snapshots.
    pub fn memory_snapshots(&self) -> &[MemorySnapshot] {
        &self.memory_snapshots
    }

    /// Calculate session statistics.
    pub fn statistics(&self) -> ProfilingStatistics {
        let mut stats = ProfilingStatistics::default();

        // Kernel statistics
        for (_, metrics) in &self.kernel_events {
            stats.total_kernel_time_ms += metrics.gpu_time_ms as f64;
            stats.kernel_count += 1;

            // Per-kernel stats
            let entry = stats
                .kernel_stats
                .entry(metrics.kernel_name.clone())
                .or_insert_with(|| KernelStatistics {
                    min_time_ms: f32::MAX,
                    max_time_ms: f32::MIN,
                    ..Default::default()
                });

            entry.invocations += 1;
            entry.total_time_ms += metrics.gpu_time_ms as f64;
            entry.min_time_ms = entry.min_time_ms.min(metrics.gpu_time_ms);
            entry.max_time_ms = entry.max_time_ms.max(metrics.gpu_time_ms);
        }

        // Calculate kernel averages
        for (_, entry) in stats.kernel_stats.iter_mut() {
            if entry.invocations > 0 {
                entry.avg_time_ms = entry.total_time_ms / entry.invocations as f64;
            }
        }

        if stats.kernel_count > 0 {
            stats.avg_kernel_time_ms = stats.total_kernel_time_ms / stats.kernel_count as f64;
        }

        // Transfer statistics
        for (_, metrics) in &self.transfer_events {
            stats.total_transfer_time_ms += metrics.gpu_time_ms as f64;
            stats.total_bytes_transferred += metrics.size_bytes as u64;
            stats.transfer_count += 1;

            // Per-direction stats
            let entry = stats.transfer_stats.entry(metrics.direction).or_default();

            entry.count += 1;
            entry.total_bytes += metrics.size_bytes as u64;
            entry.total_time_ms += metrics.gpu_time_ms as f64;
        }

        // Calculate transfer averages
        for (_, entry) in stats.transfer_stats.iter_mut() {
            if entry.total_time_ms > 0.0 {
                let bytes_per_sec = entry.total_bytes as f64 / (entry.total_time_ms / 1000.0);
                entry.avg_bandwidth_gbps = bytes_per_sec / 1_000_000_000.0;
            }
        }

        if stats.total_transfer_time_ms > 0.0 {
            let bytes_per_sec =
                stats.total_bytes_transferred as f64 / (stats.total_transfer_time_ms / 1000.0);
            stats.avg_transfer_bandwidth_gbps = bytes_per_sec / 1_000_000_000.0;
        }

        // Memory statistics
        for snapshot in &self.memory_snapshots {
            stats.peak_memory_bytes = stats.peak_memory_bytes.max(snapshot.peak_bytes);
        }

        stats
    }

    /// Clear all recorded events.
    pub fn clear(&mut self) {
        self.kernel_events.clear();
        self.transfer_events.clear();
        self.memory_snapshots.clear();
        self.start_time = Instant::now();
    }
}

impl Default for ProfilingSession {
    fn default() -> Self {
        Self::new()
    }
}

/// Query kernel attributes using cuFuncGetAttribute.
///
/// # Safety
///
/// The function handle must be valid.
pub unsafe fn query_kernel_attributes(func: cuda_sys::CUfunction) -> Result<KernelAttributes> {
    use cuda_sys::CUfunction_attribute_enum;
    use cudarc::driver::result::function::get_function_attribute;

    let num_regs =
        get_function_attribute(func, CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_NUM_REGS)
            .map_err(|e| {
                RingKernelError::BackendError(format!("Failed to get NUM_REGS: {:?}", e))
            })?;

    let shared_size = get_function_attribute(
        func,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
    )
    .map_err(|e| {
        RingKernelError::BackendError(format!("Failed to get SHARED_SIZE_BYTES: {:?}", e))
    })?;

    let local_size = get_function_attribute(
        func,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
    )
    .map_err(|e| {
        RingKernelError::BackendError(format!("Failed to get LOCAL_SIZE_BYTES: {:?}", e))
    })?;

    let max_threads = get_function_attribute(
        func,
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    )
    .map_err(|e| {
        RingKernelError::BackendError(format!("Failed to get MAX_THREADS_PER_BLOCK: {:?}", e))
    })?;

    Ok(KernelAttributes {
        num_registers: num_regs as u32,
        static_shared_mem_bytes: shared_size as u32,
        local_mem_per_thread_bytes: local_size as u32,
        max_threads_per_block: max_threads as u32,
    })
}

/// Kernel attributes queried from CUDA driver.
#[derive(Debug, Clone, Default)]
pub struct KernelAttributes {
    /// Number of registers used per thread.
    pub num_registers: u32,
    /// Static shared memory per block in bytes.
    pub static_shared_mem_bytes: u32,
    /// Local memory per thread in bytes.
    pub local_mem_per_thread_bytes: u32,
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
}

impl KernelAttributes {
    /// Calculate theoretical occupancy.
    ///
    /// This is a simplified calculation. For accurate occupancy, use
    /// `cuOccupancyMaxActiveBlocksPerMultiprocessor`.
    pub fn calculate_occupancy(
        &self,
        threads_per_block: u32,
        dynamic_shared_mem: u32,
        max_threads_per_sm: u32,
        max_blocks_per_sm: u32,
        registers_per_sm: u32,
        shared_mem_per_sm: u32,
    ) -> f32 {
        // Register limit
        let regs_per_block = self.num_registers * threads_per_block;
        let blocks_by_regs = if regs_per_block > 0 {
            registers_per_sm / regs_per_block
        } else {
            max_blocks_per_sm
        };

        // Shared memory limit
        let shared_per_block = self.static_shared_mem_bytes + dynamic_shared_mem;
        let blocks_by_shared = if shared_per_block > 0 {
            shared_mem_per_sm / shared_per_block
        } else {
            max_blocks_per_sm
        };

        // Thread limit
        let blocks_by_threads = max_threads_per_sm / threads_per_block;

        // Take minimum
        let active_blocks = blocks_by_regs
            .min(blocks_by_shared)
            .min(blocks_by_threads)
            .min(max_blocks_per_sm);

        // Occupancy = active warps / max warps
        let active_threads = active_blocks * threads_per_block;
        active_threads as f32 / max_threads_per_sm as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_metrics_basic() {
        let metrics = KernelMetrics::new("test_kernel")
            .with_grid_dim(128, 1, 1)
            .with_block_dim(256, 1, 1)
            .with_gpu_time(0.5);

        assert_eq!(metrics.kernel_name, "test_kernel");
        assert_eq!(metrics.grid_dim, (128, 1, 1));
        assert_eq!(metrics.block_dim, (256, 1, 1));
        assert_eq!(metrics.gpu_time_ms, 0.5);
    }

    #[test]
    fn test_kernel_metrics_threads() {
        let metrics = KernelMetrics::new("test")
            .with_grid_dim(10, 20, 5)
            .with_block_dim(32, 8, 4);

        assert_eq!(metrics.total_blocks(), 10 * 20 * 5);
        assert_eq!(metrics.threads_per_block(), 32 * 8 * 4);
        assert_eq!(metrics.total_threads(), 10 * 20 * 5 * 32 * 8 * 4);
    }

    #[test]
    fn test_kernel_metrics_throughput() {
        let metrics = KernelMetrics::new("test")
            .with_grid_dim(1000, 1, 1)
            .with_block_dim(1024, 1, 1)
            .with_gpu_time(1.0); // 1ms

        let throughput = metrics.throughput_mthreads_per_sec().unwrap();
        // 1,024,000 threads in 1ms = 1,024,000,000 threads/sec = 1024 M threads/sec
        assert!((throughput - 1024.0).abs() < 0.1);
    }

    #[test]
    fn test_transfer_metrics_bandwidth() {
        let metrics = TransferMetrics::new(TransferDirection::HostToDevice, 1_000_000_000) // 1GB
            .with_gpu_time(100.0); // 100ms

        // 1GB in 100ms = 10 GB/s
        let bandwidth = metrics.bandwidth_gbps();
        assert!((bandwidth - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_transfer_direction_labels() {
        assert_eq!(TransferDirection::HostToDevice.label(), "H2D");
        assert_eq!(TransferDirection::DeviceToHost.label(), "D2H");
        assert_eq!(TransferDirection::DeviceToDevice.label(), "D2D");
    }

    #[test]
    fn test_profiling_session_basic() {
        let mut session = ProfilingSession::with_name("test_session");
        assert_eq!(session.name(), "test_session");

        session.add_metadata("device", "RTX 4090");
        assert_eq!(session.get_metadata("device"), Some("RTX 4090"));

        assert_eq!(session.kernel_count(), 0);
        assert_eq!(session.transfer_count(), 0);
    }

    #[test]
    fn test_profiling_session_record_kernel() {
        let mut session = ProfilingSession::new();

        session.record_kernel(
            KernelMetrics::new("kernel1")
                .with_grid_dim(100, 1, 1)
                .with_block_dim(256, 1, 1)
                .with_gpu_time(0.5),
        );

        session.record_kernel(
            KernelMetrics::new("kernel1")
                .with_grid_dim(100, 1, 1)
                .with_block_dim(256, 1, 1)
                .with_gpu_time(0.6),
        );

        session.record_kernel(
            KernelMetrics::new("kernel2")
                .with_grid_dim(200, 1, 1)
                .with_block_dim(128, 1, 1)
                .with_gpu_time(1.0),
        );

        assert_eq!(session.kernel_count(), 3);

        let stats = session.statistics();
        assert_eq!(stats.kernel_count, 3);
        assert!((stats.total_kernel_time_ms - 2.1).abs() < 0.01);

        let kernel1_stats = stats.kernel_stats.get("kernel1").unwrap();
        assert_eq!(kernel1_stats.invocations, 2);
        assert!((kernel1_stats.total_time_ms - 1.1).abs() < 0.01);
        assert!((kernel1_stats.avg_time_ms - 0.55).abs() < 0.01);
        assert!((kernel1_stats.min_time_ms - 0.5).abs() < 0.01);
        assert!((kernel1_stats.max_time_ms - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_profiling_session_record_transfer() {
        let mut session = ProfilingSession::new();

        session.record_transfer(
            TransferMetrics::new(TransferDirection::HostToDevice, 1_000_000).with_gpu_time(1.0),
        );

        session.record_transfer(
            TransferMetrics::new(TransferDirection::DeviceToHost, 500_000).with_gpu_time(0.5),
        );

        assert_eq!(session.transfer_count(), 2);

        let stats = session.statistics();
        assert_eq!(stats.transfer_count, 2);
        assert_eq!(stats.total_bytes_transferred, 1_500_000);

        let h2d_stats = stats
            .transfer_stats
            .get(&TransferDirection::HostToDevice)
            .unwrap();
        assert_eq!(h2d_stats.count, 1);
        assert_eq!(h2d_stats.total_bytes, 1_000_000);
    }

    #[test]
    fn test_profiling_session_memory_snapshot() {
        let mut session = ProfilingSession::new();

        session.record_memory_snapshot(MemorySnapshot {
            allocated_bytes: 1_000_000,
            peak_bytes: 1_000_000,
            allocation_count: 10,
            ..Default::default()
        });

        session.record_memory_snapshot(MemorySnapshot {
            allocated_bytes: 500_000,
            peak_bytes: 1_500_000,
            allocation_count: 5,
            ..Default::default()
        });

        assert_eq!(session.memory_snapshots().len(), 2);

        let stats = session.statistics();
        assert_eq!(stats.peak_memory_bytes, 1_500_000);
    }

    #[test]
    fn test_profiling_session_clear() {
        let mut session = ProfilingSession::new();

        session.record_kernel(KernelMetrics::new("test").with_gpu_time(1.0));
        session.record_transfer(TransferMetrics::new(TransferDirection::HostToDevice, 1000));

        assert_eq!(session.kernel_count(), 1);
        assert_eq!(session.transfer_count(), 1);

        session.clear();

        assert_eq!(session.kernel_count(), 0);
        assert_eq!(session.transfer_count(), 0);
    }

    #[test]
    fn test_kernel_attributes_occupancy() {
        let attrs = KernelAttributes {
            num_registers: 32,
            static_shared_mem_bytes: 4096,
            local_mem_per_thread_bytes: 0,
            max_threads_per_block: 1024,
        };

        // Simplified occupancy calculation
        let occupancy = attrs.calculate_occupancy(
            256,   // threads_per_block
            0,     // dynamic_shared_mem
            2048,  // max_threads_per_sm
            32,    // max_blocks_per_sm
            65536, // registers_per_sm
            49152, // shared_mem_per_sm
        );

        // Should be limited by threads: 2048/256 = 8 blocks, 8*256/2048 = 1.0
        assert!(occupancy > 0.0 && occupancy <= 1.0);
    }
}
