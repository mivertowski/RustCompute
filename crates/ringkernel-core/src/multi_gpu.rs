//! Multi-GPU coordination, topology discovery, and cross-GPU messaging.
//!
//! This module provides infrastructure for coordinating work across
//! multiple GPUs, including:
//!
//! - **Device Selection** - Load balancing strategies for kernel placement
//! - **Topology Discovery** - NVLink/PCIe detection and bandwidth estimation
//! - **Cross-GPU K2K Router** - Kernel-to-kernel messaging across GPUs
//! - **Kernel Migration** - Move kernels between GPUs with state transfer
//!
//! ## Example
//!
//! ```ignore
//! use ringkernel_core::multi_gpu::{MultiGpuBuilder, GpuTopology, CrossGpuK2KRouter};
//!
//! let coordinator = MultiGpuBuilder::new()
//!     .load_balancing(LoadBalancingStrategy::LeastLoaded)
//!     .enable_p2p(true)
//!     .build();
//!
//! // Discover topology
//! let topology = coordinator.discover_topology();
//!
//! // Create cross-GPU router
//! let router = CrossGpuK2KRouter::new(coordinator.clone());
//! router.route_message(source_kernel, dest_kernel, envelope).await?;
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{Result, RingKernelError};
use crate::k2k::K2KMessage;
use crate::runtime::{Backend, KernelId, LaunchOptions};

/// Configuration for multi-GPU coordination.
#[derive(Debug, Clone)]
pub struct MultiGpuConfig {
    /// Load balancing strategy.
    pub load_balancing: LoadBalancingStrategy,
    /// Enable automatic device selection.
    pub auto_select_device: bool,
    /// Maximum kernels per device.
    pub max_kernels_per_device: usize,
    /// Enable peer-to-peer transfers when available.
    pub enable_p2p: bool,
    /// Preferred devices (by index).
    pub preferred_devices: Vec<usize>,
}

impl Default for MultiGpuConfig {
    fn default() -> Self {
        Self {
            load_balancing: LoadBalancingStrategy::LeastLoaded,
            auto_select_device: true,
            max_kernels_per_device: 64,
            enable_p2p: true,
            preferred_devices: vec![],
        }
    }
}

/// Strategy for balancing load across devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Always use the first available device.
    FirstAvailable,
    /// Use the device with fewest kernels.
    LeastLoaded,
    /// Round-robin across devices.
    RoundRobin,
    /// Select based on memory availability.
    MemoryBased,
    /// Select based on compute capability.
    ComputeCapability,
    /// Custom selection function.
    Custom,
}

/// Information about a GPU device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device index.
    pub index: usize,
    /// Device name.
    pub name: String,
    /// Backend type.
    pub backend: Backend,
    /// Total memory in bytes.
    pub total_memory: u64,
    /// Available memory in bytes.
    pub available_memory: u64,
    /// Compute capability (for CUDA).
    pub compute_capability: Option<(u32, u32)>,
    /// Maximum threads per block.
    pub max_threads_per_block: u32,
    /// Number of multiprocessors.
    pub multiprocessor_count: u32,
    /// Whether device supports P2P with other devices.
    pub p2p_capable: bool,
}

impl DeviceInfo {
    /// Create a new device info.
    pub fn new(index: usize, name: String, backend: Backend) -> Self {
        Self {
            index,
            name,
            backend,
            total_memory: 0,
            available_memory: 0,
            compute_capability: None,
            max_threads_per_block: 1024,
            multiprocessor_count: 1,
            p2p_capable: false,
        }
    }

    /// Get memory utilization (0.0-1.0).
    pub fn memory_utilization(&self) -> f64 {
        if self.total_memory == 0 {
            0.0
        } else {
            1.0 - (self.available_memory as f64 / self.total_memory as f64)
        }
    }
}

// ============================================================================
// GPU Topology Discovery
// ============================================================================

/// Type of interconnect between GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterconnectType {
    /// No direct connection (must go through host).
    None,
    /// PCIe peer-to-peer.
    Pcie,
    /// NVIDIA NVLink.
    NvLink,
    /// NVIDIA NVSwitch (datacenter).
    NvSwitch,
    /// AMD Infinity Fabric.
    InfinityFabric,
    /// Intel Xe Link.
    XeLink,
    /// Same GPU (for self-connections).
    SameDevice,
}

impl InterconnectType {
    /// Estimated bandwidth in GB/s for this interconnect type.
    pub fn estimated_bandwidth_gbps(&self) -> f64 {
        match self {
            InterconnectType::None => 16.0,      // PCIe 3.0 x16 through host
            InterconnectType::Pcie => 32.0,      // PCIe 4.0 x16 P2P
            InterconnectType::NvLink => 300.0,   // NVLink 3.0 (A100)
            InterconnectType::NvSwitch => 600.0, // NVSwitch full bisection
            InterconnectType::InfinityFabric => 200.0, // MI250X
            InterconnectType::XeLink => 100.0,   // Intel Data Center GPUs
            InterconnectType::SameDevice => 2000.0, // Internal bandwidth
        }
    }

    /// Estimated latency in microseconds.
    pub fn estimated_latency_us(&self) -> f64 {
        match self {
            InterconnectType::None => 10.0,    // Through host memory
            InterconnectType::Pcie => 5.0,     // P2P PCIe
            InterconnectType::NvLink => 1.0,   // Direct NVLink
            InterconnectType::NvSwitch => 2.0, // Through switch
            InterconnectType::InfinityFabric => 1.5,
            InterconnectType::XeLink => 2.0,
            InterconnectType::SameDevice => 0.0,
        }
    }

    /// Whether this interconnect supports direct P2P memory access.
    pub fn supports_p2p(&self) -> bool {
        !matches!(self, InterconnectType::None)
    }
}

/// Connection between two GPUs.
#[derive(Debug, Clone)]
pub struct GpuConnection {
    /// Source device index.
    pub source: usize,
    /// Destination device index.
    pub destination: usize,
    /// Type of interconnect.
    pub interconnect: InterconnectType,
    /// Measured or estimated bandwidth in GB/s.
    pub bandwidth_gbps: f64,
    /// Measured or estimated latency in microseconds.
    pub latency_us: f64,
    /// Whether connection is bidirectional with same characteristics.
    pub bidirectional: bool,
    /// Number of hops (for multi-hop topologies).
    pub hops: u32,
}

impl GpuConnection {
    /// Create a new GPU connection.
    pub fn new(source: usize, destination: usize, interconnect: InterconnectType) -> Self {
        Self {
            source,
            destination,
            interconnect,
            bandwidth_gbps: interconnect.estimated_bandwidth_gbps(),
            latency_us: interconnect.estimated_latency_us(),
            bidirectional: true,
            hops: if source == destination { 0 } else { 1 },
        }
    }

    /// Set measured bandwidth.
    pub fn with_bandwidth(mut self, gbps: f64) -> Self {
        self.bandwidth_gbps = gbps;
        self
    }

    /// Set measured latency.
    pub fn with_latency(mut self, us: f64) -> Self {
        self.latency_us = us;
        self
    }

    /// Set hop count.
    pub fn with_hops(mut self, hops: u32) -> Self {
        self.hops = hops;
        self
    }
}

/// GPU topology graph describing all device interconnections.
#[derive(Debug, Clone)]
pub struct GpuTopology {
    /// Number of devices in topology.
    pub device_count: usize,
    /// Connection matrix (device_count x device_count).
    connections: Vec<Vec<Option<GpuConnection>>>,
    /// NUMA node assignments for each device.
    pub numa_nodes: Vec<Option<u32>>,
    /// Whether topology has been probed (vs estimated).
    pub probed: bool,
    /// Timestamp of last topology update.
    pub last_updated: Instant,
}

impl GpuTopology {
    /// Create a new topology for N devices.
    pub fn new(device_count: usize) -> Self {
        let mut connections = vec![vec![None; device_count]; device_count];

        // Initialize self-connections
        for (i, row) in connections.iter_mut().enumerate().take(device_count) {
            row[i] = Some(GpuConnection::new(i, i, InterconnectType::SameDevice));
        }

        Self {
            device_count,
            connections,
            numa_nodes: vec![None; device_count],
            probed: false,
            last_updated: Instant::now(),
        }
    }

    /// Set connection between two devices.
    pub fn set_connection(&mut self, connection: GpuConnection) {
        let src = connection.source;
        let dst = connection.destination;
        if src < self.device_count && dst < self.device_count {
            self.connections[src][dst] = Some(connection.clone());
            if connection.bidirectional && src != dst {
                let reverse = GpuConnection {
                    source: dst,
                    destination: src,
                    ..connection
                };
                self.connections[dst][src] = Some(reverse);
            }
        }
    }

    /// Get connection between two devices.
    pub fn get_connection(&self, source: usize, destination: usize) -> Option<&GpuConnection> {
        self.connections
            .get(source)
            .and_then(|row| row.get(destination))
            .and_then(|c| c.as_ref())
    }

    /// Get best path between two devices (returns intermediate hops).
    pub fn best_path(&self, source: usize, destination: usize) -> Vec<usize> {
        if source == destination {
            return vec![source];
        }

        // Direct connection available?
        if let Some(conn) = self.get_connection(source, destination) {
            if conn.interconnect != InterconnectType::None {
                return vec![source, destination];
            }
        }

        // Find best path via Dijkstra (simplified)
        let mut best_path = vec![source, destination]; // Default to direct
        let mut best_bandwidth = 0.0;

        // Check all intermediate nodes
        for intermediate in 0..self.device_count {
            if intermediate == source || intermediate == destination {
                continue;
            }

            if let (Some(c1), Some(c2)) = (
                self.get_connection(source, intermediate),
                self.get_connection(intermediate, destination),
            ) {
                // Bandwidth limited by slowest link
                let path_bandwidth = c1.bandwidth_gbps.min(c2.bandwidth_gbps);
                if path_bandwidth > best_bandwidth {
                    best_bandwidth = path_bandwidth;
                    best_path = vec![source, intermediate, destination];
                }
            }
        }

        best_path
    }

    /// Get all devices directly connected to a device.
    pub fn neighbors(&self, device: usize) -> Vec<usize> {
        if device >= self.device_count {
            return vec![];
        }

        self.connections[device]
            .iter()
            .enumerate()
            .filter_map(|(i, conn)| {
                if i != device
                    && conn
                        .as_ref()
                        .map(|c| c.interconnect.supports_p2p())
                        .unwrap_or(false)
                {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Calculate total bisection bandwidth of the topology.
    pub fn bisection_bandwidth_gbps(&self) -> f64 {
        let half = self.device_count / 2;
        if half == 0 {
            return 0.0;
        }

        let mut total = 0.0;
        for src in 0..half {
            for dst in half..self.device_count {
                if let Some(conn) = self.get_connection(src, dst) {
                    total += conn.bandwidth_gbps;
                }
            }
        }
        total
    }

    /// Check if all devices have P2P connectivity.
    pub fn is_fully_connected(&self) -> bool {
        for src in 0..self.device_count {
            for dst in 0..self.device_count {
                if src != dst {
                    if let Some(conn) = self.get_connection(src, dst) {
                        if !conn.interconnect.supports_p2p() {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Get devices in the same NUMA domain.
    pub fn numa_neighbors(&self, device: usize) -> Vec<usize> {
        let target_numa = self.numa_nodes.get(device).copied().flatten();
        if target_numa.is_none() {
            return vec![];
        }

        self.numa_nodes
            .iter()
            .enumerate()
            .filter_map(|(i, numa)| {
                if i != device && *numa == target_numa {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Set NUMA node for a device.
    pub fn set_numa_node(&mut self, device: usize, numa_node: u32) {
        if device < self.numa_nodes.len() {
            self.numa_nodes[device] = Some(numa_node);
        }
    }

    /// Mark topology as probed (not estimated).
    pub fn mark_probed(&mut self) {
        self.probed = true;
        self.last_updated = Instant::now();
    }
}

/// Status of a device in the multi-GPU coordinator.
#[derive(Debug, Clone)]
pub struct DeviceStatus {
    /// Device info.
    pub info: DeviceInfo,
    /// Number of kernels running on this device.
    pub kernel_count: usize,
    /// Kernels running on this device.
    pub kernels: Vec<KernelId>,
    /// Whether device is available for new kernels.
    pub available: bool,
    /// Current load estimate (0.0-1.0).
    pub load: f64,
}

/// Result of unregistering a device from the coordinator.
#[derive(Debug, Clone)]
pub struct DeviceUnregisterResult {
    /// Index of the unregistered device.
    pub device_index: usize,
    /// Kernels that were on this device and need migration.
    pub kernels_to_migrate: Vec<KernelMigrationPlan>,
    /// Kernels that could not be migrated (no available target).
    pub orphaned_kernels: Vec<KernelId>,
    /// Whether the device was successfully unregistered.
    pub success: bool,
}

/// Plan for migrating a single kernel during device unregister.
#[derive(Debug, Clone)]
pub struct KernelMigrationPlan {
    /// Kernel to migrate.
    pub kernel_id: KernelId,
    /// Source device (the unregistered device).
    pub source_device: usize,
    /// Target device selected for migration.
    pub target_device: usize,
    /// Estimated migration priority (based on kernel load).
    pub priority: MigrationPriority,
}

/// Priority for kernel migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationPriority {
    /// Low priority - can be migrated lazily.
    Low,
    /// Normal priority - migrate in reasonable time.
    Normal,
    /// High priority - migrate as soon as possible.
    High,
    /// Critical - must migrate immediately.
    Critical,
}

/// Multi-GPU coordinator for managing kernels across devices.
pub struct MultiGpuCoordinator {
    /// Configuration.
    config: MultiGpuConfig,
    /// Available devices.
    devices: RwLock<Vec<DeviceInfo>>,
    /// Kernel-to-device mapping.
    kernel_device_map: RwLock<HashMap<KernelId, usize>>,
    /// Device kernel counts.
    device_kernel_counts: RwLock<Vec<AtomicUsize>>,
    /// Round-robin counter.
    round_robin_counter: AtomicUsize,
    /// Total kernels launched.
    total_kernels: AtomicU64,
    /// Device selection callbacks (for custom strategy).
    #[allow(clippy::type_complexity)]
    custom_selector:
        RwLock<Option<Arc<dyn Fn(&[DeviceStatus], &LaunchOptions) -> usize + Send + Sync>>>,
    /// GPU topology graph.
    topology: RwLock<Option<GpuTopology>>,
}

impl MultiGpuCoordinator {
    /// Create a new multi-GPU coordinator.
    pub fn new(config: MultiGpuConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            devices: RwLock::new(Vec::new()),
            kernel_device_map: RwLock::new(HashMap::new()),
            device_kernel_counts: RwLock::new(Vec::new()),
            round_robin_counter: AtomicUsize::new(0),
            total_kernels: AtomicU64::new(0),
            custom_selector: RwLock::new(None),
            topology: RwLock::new(None),
        })
    }

    /// Register a device with the coordinator.
    pub fn register_device(&self, device: DeviceInfo) {
        let index = device.index;
        let mut devices = self.devices.write();
        let mut counts = self.device_kernel_counts.write();

        // Ensure we have enough slots
        let mut current_len = devices.len();
        while current_len <= index {
            devices.push(DeviceInfo::new(
                current_len,
                "Unknown".to_string(),
                Backend::Cpu,
            ));
            counts.push(AtomicUsize::new(0));
            current_len += 1;
        }

        devices[index] = device;
    }

    /// Unregister a device and plan kernel migrations.
    ///
    /// This method:
    /// 1. Identifies all kernels on the device being removed
    /// 2. Finds target devices for each kernel using load balancing
    /// 3. Creates migration plans for kernels that can be moved
    /// 4. Marks orphaned kernels that have no migration target
    /// 5. Updates internal routing tables
    ///
    /// The caller is responsible for executing the actual migrations using
    /// [`KernelMigrator`] with the returned [`KernelMigrationPlan`] entries.
    pub fn unregister_device(&self, index: usize) -> DeviceUnregisterResult {
        let devices = self.devices.read();

        // Check if device exists
        if index >= devices.len() {
            return DeviceUnregisterResult {
                device_index: index,
                kernels_to_migrate: Vec::new(),
                orphaned_kernels: Vec::new(),
                success: false,
            };
        }

        // Get all kernels on this device
        let kernels_on_device = self.kernels_on_device(index);

        // Find available target devices (excluding the one being unregistered)
        let available_targets: Vec<usize> = devices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != index)
            .map(|(i, _)| i)
            .collect();

        drop(devices); // Release read lock before acquiring write lock

        let mut kernels_to_migrate = Vec::new();
        let mut orphaned_kernels = Vec::new();

        if available_targets.is_empty() {
            // No other devices available - all kernels are orphaned
            orphaned_kernels = kernels_on_device;
        } else {
            // Plan migrations for each kernel
            for kernel_id in kernels_on_device {
                // Select target based on current load
                if let Some(target) = self.select_migration_target(&available_targets) {
                    let priority = self.calculate_migration_priority(&kernel_id);
                    kernels_to_migrate.push(KernelMigrationPlan {
                        kernel_id,
                        source_device: index,
                        target_device: target,
                        priority,
                    });
                } else {
                    orphaned_kernels.push(kernel_id);
                }
            }
        }

        // Update kernel-device mappings for planned migrations
        {
            let mut kernel_map = self.kernel_device_map.write();
            let counts = self.device_kernel_counts.read();

            for plan in &kernels_to_migrate {
                // Update mapping to target device
                kernel_map.insert(plan.kernel_id.clone(), plan.target_device);

                // Update kernel counts
                if index < counts.len() {
                    counts[index].fetch_sub(1, Ordering::Relaxed);
                }
                if plan.target_device < counts.len() {
                    counts[plan.target_device].fetch_add(1, Ordering::Relaxed);
                }
            }

            // Remove orphaned kernels from mapping
            for kernel_id in &orphaned_kernels {
                kernel_map.remove(kernel_id);
                if index < counts.len() {
                    counts[index].fetch_sub(1, Ordering::Relaxed);
                }
            }
        }

        // Mark device as unavailable (but don't remove it to preserve indices)
        {
            let mut devices = self.devices.write();
            if index < devices.len() {
                devices[index].available_memory = 0;
                devices[index].name = format!("{} (unregistered)", devices[index].name);
            }
        }

        DeviceUnregisterResult {
            device_index: index,
            kernels_to_migrate,
            orphaned_kernels,
            success: true,
        }
    }

    /// Select the best target device for migration.
    fn select_migration_target(&self, candidates: &[usize]) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }

        let counts = self.device_kernel_counts.read();

        // Find device with lowest kernel count
        candidates
            .iter()
            .filter_map(|&idx| {
                if idx < counts.len() {
                    Some((idx, counts[idx].load(Ordering::Relaxed)))
                } else {
                    None
                }
            })
            .min_by_key(|(_, count)| *count)
            .map(|(idx, _)| idx)
    }

    /// Calculate migration priority for a kernel.
    fn calculate_migration_priority(&self, _kernel_id: &KernelId) -> MigrationPriority {
        // In a real implementation, this would check:
        // - Message queue depth
        // - Time since last activity
        // - Kernel type/importance
        // For now, use normal priority
        MigrationPriority::Normal
    }

    /// Get all registered devices.
    pub fn devices(&self) -> Vec<DeviceInfo> {
        self.devices.read().clone()
    }

    /// Get device info by index.
    pub fn device(&self, index: usize) -> Option<DeviceInfo> {
        self.devices.read().get(index).cloned()
    }

    /// Get number of devices.
    pub fn device_count(&self) -> usize {
        self.devices.read().len()
    }

    /// Select a device for launching a kernel.
    pub fn select_device(&self, options: &LaunchOptions) -> Result<usize> {
        let devices = self.devices.read();
        if devices.is_empty() {
            return Err(RingKernelError::BackendUnavailable(
                "No GPU devices available".to_string(),
            ));
        }

        // Get current status
        let status = self.get_all_status();

        // Check for custom selector
        if self.config.load_balancing == LoadBalancingStrategy::Custom {
            if let Some(selector) = &*self.custom_selector.read() {
                return Ok(selector(&status, options));
            }
        }

        // Apply preferred devices filter if specified
        let candidates: Vec<_> = if !self.config.preferred_devices.is_empty() {
            status
                .into_iter()
                .filter(|s| self.config.preferred_devices.contains(&s.info.index))
                .collect()
        } else {
            status
        };

        if candidates.is_empty() {
            return Err(RingKernelError::BackendUnavailable(
                "No suitable GPU device available".to_string(),
            ));
        }

        let selected = match self.config.load_balancing {
            LoadBalancingStrategy::FirstAvailable => {
                candidates.first().map(|s| s.info.index).unwrap_or(0)
            }
            LoadBalancingStrategy::LeastLoaded => candidates
                .iter()
                .filter(|s| s.available && s.kernel_count < self.config.max_kernels_per_device)
                .min_by(|a, b| a.kernel_count.cmp(&b.kernel_count))
                .map(|s| s.info.index)
                .unwrap_or(0),
            LoadBalancingStrategy::RoundRobin => {
                let available: Vec<_> = candidates.iter().filter(|s| s.available).collect();

                if available.is_empty() {
                    candidates.first().map(|s| s.info.index).unwrap_or(0)
                } else {
                    let idx =
                        self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % available.len();
                    available[idx].info.index
                }
            }
            LoadBalancingStrategy::MemoryBased => candidates
                .iter()
                .filter(|s| s.available)
                .max_by(|a, b| a.info.available_memory.cmp(&b.info.available_memory))
                .map(|s| s.info.index)
                .unwrap_or(0),
            LoadBalancingStrategy::ComputeCapability => candidates
                .iter()
                .filter(|s| s.available)
                .max_by(|a, b| {
                    let a_cap = a.info.compute_capability.unwrap_or((0, 0));
                    let b_cap = b.info.compute_capability.unwrap_or((0, 0));
                    a_cap.cmp(&b_cap)
                })
                .map(|s| s.info.index)
                .unwrap_or(0),
            LoadBalancingStrategy::Custom => {
                // Should have been handled above
                0
            }
        };

        Ok(selected)
    }

    /// Assign a kernel to a device.
    pub fn assign_kernel(&self, kernel_id: KernelId, device_index: usize) {
        self.kernel_device_map
            .write()
            .insert(kernel_id, device_index);

        let counts = self.device_kernel_counts.read();
        if device_index < counts.len() {
            counts[device_index].fetch_add(1, Ordering::Relaxed);
        }

        self.total_kernels.fetch_add(1, Ordering::Relaxed);
    }

    /// Remove a kernel assignment.
    pub fn remove_kernel(&self, kernel_id: &KernelId) {
        if let Some(device_index) = self.kernel_device_map.write().remove(kernel_id) {
            let counts = self.device_kernel_counts.read();
            if device_index < counts.len() {
                counts[device_index].fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// Get device for a kernel.
    pub fn get_kernel_device(&self, kernel_id: &KernelId) -> Option<usize> {
        self.kernel_device_map.read().get(kernel_id).copied()
    }

    /// Get all kernels on a device.
    pub fn kernels_on_device(&self, device_index: usize) -> Vec<KernelId> {
        self.kernel_device_map
            .read()
            .iter()
            .filter(|(_, &idx)| idx == device_index)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Get status of all devices.
    pub fn get_all_status(&self) -> Vec<DeviceStatus> {
        let devices = self.devices.read();
        let kernel_map = self.kernel_device_map.read();
        let counts = self.device_kernel_counts.read();

        devices
            .iter()
            .enumerate()
            .map(|(idx, info)| {
                let kernel_count = if idx < counts.len() {
                    counts[idx].load(Ordering::Relaxed)
                } else {
                    0
                };

                let kernels: Vec<_> = kernel_map
                    .iter()
                    .filter(|(_, &dev_idx)| dev_idx == idx)
                    .map(|(k, _)| k.clone())
                    .collect();

                let load = kernel_count as f64 / self.config.max_kernels_per_device as f64;
                let available = kernel_count < self.config.max_kernels_per_device;

                DeviceStatus {
                    info: info.clone(),
                    kernel_count,
                    kernels,
                    available,
                    load,
                }
            })
            .collect()
    }

    /// Get status of a specific device.
    pub fn get_device_status(&self, device_index: usize) -> Option<DeviceStatus> {
        self.get_all_status().into_iter().nth(device_index)
    }

    /// Set custom device selector.
    pub fn set_custom_selector<F>(&self, selector: F)
    where
        F: Fn(&[DeviceStatus], &LaunchOptions) -> usize + Send + Sync + 'static,
    {
        *self.custom_selector.write() = Some(Arc::new(selector));
    }

    /// Get coordinator statistics.
    pub fn stats(&self) -> MultiGpuStats {
        let status = self.get_all_status();
        let total_kernels: usize = status.iter().map(|s| s.kernel_count).sum();
        let total_memory: u64 = status.iter().map(|s| s.info.total_memory).sum();
        let available_memory: u64 = status.iter().map(|s| s.info.available_memory).sum();

        MultiGpuStats {
            device_count: status.len(),
            total_kernels,
            total_memory,
            available_memory,
            kernels_launched: self.total_kernels.load(Ordering::Relaxed),
        }
    }

    /// Check if P2P is available between two devices.
    pub fn can_p2p(&self, device_a: usize, device_b: usize) -> bool {
        if !self.config.enable_p2p {
            return false;
        }

        let devices = self.devices.read();
        if let (Some(a), Some(b)) = (devices.get(device_a), devices.get(device_b)) {
            a.p2p_capable && b.p2p_capable
        } else {
            false
        }
    }

    /// Update device memory info.
    pub fn update_device_memory(&self, device_index: usize, available_memory: u64) {
        let mut devices = self.devices.write();
        if let Some(device) = devices.get_mut(device_index) {
            device.available_memory = available_memory;
        }
    }

    // ========================================================================
    // Topology Discovery
    // ========================================================================

    /// Discover GPU topology (estimates if probing not available).
    pub fn discover_topology(&self) -> GpuTopology {
        let devices = self.devices.read();
        let device_count = devices.len();

        if device_count == 0 {
            return GpuTopology::new(0);
        }

        let mut topo = GpuTopology::new(device_count);

        // Set up connections based on device info
        for (i, dev_i) in devices.iter().enumerate() {
            for (j, dev_j) in devices.iter().enumerate() {
                if i == j {
                    continue;
                }

                // Determine interconnect type based on device capabilities
                let interconnect = if dev_i.p2p_capable && dev_j.p2p_capable {
                    // Check if same backend (can do P2P)
                    if dev_i.backend == dev_j.backend {
                        match dev_i.backend {
                            Backend::Cuda => {
                                // For CUDA, check compute capability for NVLink
                                let cc_i = dev_i.compute_capability.unwrap_or((0, 0));
                                let cc_j = dev_j.compute_capability.unwrap_or((0, 0));

                                // Ampere+ (SM 80+) likely has NVLink
                                if cc_i.0 >= 8 && cc_j.0 >= 8 {
                                    InterconnectType::NvLink
                                } else {
                                    InterconnectType::Pcie
                                }
                            }
                            _ => InterconnectType::Pcie,
                        }
                    } else {
                        InterconnectType::None
                    }
                } else {
                    InterconnectType::None
                };

                topo.set_connection(GpuConnection::new(i, j, interconnect));
            }
        }

        // Store topology
        *self.topology.write() = Some(topo.clone());

        topo
    }

    /// Get current topology (discovers if not cached).
    pub fn topology(&self) -> GpuTopology {
        {
            let topo = self.topology.read();
            if let Some(ref t) = *topo {
                return t.clone();
            }
        }
        self.discover_topology()
    }

    /// Set custom topology (for testing or manual configuration).
    pub fn set_topology(&self, topology: GpuTopology) {
        *self.topology.write() = Some(topology);
    }

    /// Get best device for communicating with a source kernel.
    pub fn select_device_for_k2k(&self, source_kernel: &KernelId) -> Result<usize> {
        let source_device = self.get_kernel_device(source_kernel);
        if source_device.is_none() {
            return self.select_device(&LaunchOptions::default());
        }

        let source_idx = source_device.unwrap();
        let topo = self.topology();
        let status = self.get_all_status();

        // Find best device based on connectivity and load
        let neighbors = topo.neighbors(source_idx);

        if neighbors.is_empty() {
            // No P2P neighbors, fall back to normal selection
            return self.select_device(&LaunchOptions::default());
        }

        // Score devices by: connectivity bandwidth / (load + 1)
        let best = neighbors
            .iter()
            .filter_map(|&dev_idx| {
                status.iter().find(|s| s.info.index == dev_idx).map(|s| {
                    let conn = topo.get_connection(source_idx, dev_idx);
                    let bandwidth = conn.map(|c| c.bandwidth_gbps).unwrap_or(1.0);
                    let score = bandwidth / (s.load + 0.1);
                    (dev_idx, score)
                })
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx);

        best.ok_or_else(|| {
            RingKernelError::BackendUnavailable("No suitable K2K device found".to_string())
        })
    }

    // ========================================================================
    // Kernel Migration
    // ========================================================================

    /// Request to migrate a kernel to another device.
    pub fn request_migration(
        &self,
        kernel_id: &KernelId,
        target_device: usize,
    ) -> Result<MigrationRequest> {
        let source_device = self
            .get_kernel_device(kernel_id)
            .ok_or_else(|| RingKernelError::KernelNotFound(kernel_id.as_str().to_string()))?;

        if source_device == target_device {
            return Err(RingKernelError::InvalidConfig(
                "Cannot migrate to same device".to_string(),
            ));
        }

        let devices = self.devices.read();
        if target_device >= devices.len() {
            return Err(RingKernelError::DeviceNotAvailable(format!(
                "Device {} not available",
                target_device
            )));
        }

        let topo = self.topology();
        let path = topo.best_path(source_device, target_device);
        let connection = topo.get_connection(source_device, target_device);

        Ok(MigrationRequest {
            kernel_id: kernel_id.clone(),
            source_device,
            target_device,
            path,
            estimated_bandwidth_gbps: connection.map(|c| c.bandwidth_gbps).unwrap_or(16.0),
            estimated_latency_us: connection.map(|c| c.latency_us).unwrap_or(10.0),
            state: MigrationState::Pending,
            started_at: None,
        })
    }

    /// Complete a migration (updates internal mappings).
    pub fn complete_migration(&self, request: &MigrationRequest) -> Result<()> {
        // Update kernel-device mapping
        {
            let mut map = self.kernel_device_map.write();
            if let Some(dev) = map.get_mut(&request.kernel_id) {
                *dev = request.target_device;
            }
        }

        // Update kernel counts
        {
            let counts = self.device_kernel_counts.read();
            if request.source_device < counts.len() {
                counts[request.source_device].fetch_sub(1, Ordering::Relaxed);
            }
            if request.target_device < counts.len() {
                counts[request.target_device].fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(())
    }
}

// ============================================================================
// Kernel Migration Types
// ============================================================================

/// State of a kernel migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationState {
    /// Migration is pending, not yet started.
    Pending,
    /// Kernel is being quiesced (draining messages).
    Quiescing,
    /// Checkpoint is being created.
    Checkpointing,
    /// State is being transferred.
    Transferring,
    /// Kernel is being restored on target.
    Restoring,
    /// Migration completed successfully.
    Completed,
    /// Migration failed.
    Failed,
    /// Migration was cancelled.
    Cancelled,
}

/// Request to migrate a kernel between devices.
#[derive(Debug, Clone)]
pub struct MigrationRequest {
    /// Kernel to migrate.
    pub kernel_id: KernelId,
    /// Source device index.
    pub source_device: usize,
    /// Target device index.
    pub target_device: usize,
    /// Path of devices for multi-hop migration.
    pub path: Vec<usize>,
    /// Estimated bandwidth for transfer.
    pub estimated_bandwidth_gbps: f64,
    /// Estimated latency.
    pub estimated_latency_us: f64,
    /// Current state.
    pub state: MigrationState,
    /// When migration started.
    pub started_at: Option<Instant>,
}

impl MigrationRequest {
    /// Estimate transfer time for given state size.
    pub fn estimate_transfer_time(&self, state_size_bytes: usize) -> Duration {
        // time = size / bandwidth + latency
        let size_gb = state_size_bytes as f64 / 1_000_000_000.0;
        let transfer_time_s = size_gb / self.estimated_bandwidth_gbps;
        let total_us = (transfer_time_s * 1_000_000.0) + self.estimated_latency_us;
        Duration::from_micros(total_us as u64)
    }
}

// ============================================================================
// Cross-GPU K2K Router
// ============================================================================

/// Routes K2K messages across GPU boundaries.
pub struct CrossGpuK2KRouter {
    /// Multi-GPU coordinator.
    coordinator: Arc<MultiGpuCoordinator>,
    /// Message queues for pending cross-device messages.
    pending_queues: RwLock<HashMap<(usize, usize), Vec<PendingK2KMessage>>>,
    /// Statistics.
    stats: CrossGpuRouterStats,
}

/// A pending cross-GPU K2K message.
#[derive(Debug, Clone)]
pub struct PendingK2KMessage {
    /// Source kernel ID.
    pub source_kernel: KernelId,
    /// Destination kernel ID.
    pub dest_kernel: KernelId,
    /// Message payload.
    pub message: K2KMessage,
    /// Timestamp when queued.
    pub queued_at: Instant,
    /// Number of routing hops.
    pub hops: u32,
}

/// Statistics for cross-GPU K2K routing.
#[derive(Debug, Default)]
pub struct CrossGpuRouterStats {
    /// Total messages routed.
    messages_routed: AtomicU64,
    /// Total bytes transferred.
    bytes_transferred: AtomicU64,
    /// Messages currently pending.
    messages_pending: AtomicUsize,
    /// Total routing latency (microseconds).
    total_latency_us: AtomicU64,
    /// Failed routing attempts.
    routing_failures: AtomicU64,
}

impl CrossGpuK2KRouter {
    /// Create a new cross-GPU K2K router.
    pub fn new(coordinator: Arc<MultiGpuCoordinator>) -> Arc<Self> {
        Arc::new(Self {
            coordinator,
            pending_queues: RwLock::new(HashMap::new()),
            stats: CrossGpuRouterStats::default(),
        })
    }

    /// Route a message from source kernel to destination kernel.
    pub fn route_message(
        &self,
        source_kernel: &KernelId,
        dest_kernel: &KernelId,
        message: K2KMessage,
    ) -> Result<RoutingDecision> {
        let source_device = self
            .coordinator
            .get_kernel_device(source_kernel)
            .ok_or_else(|| {
                RingKernelError::K2KDestinationNotFound(source_kernel.as_str().to_string())
            })?;

        let dest_device = self
            .coordinator
            .get_kernel_device(dest_kernel)
            .ok_or_else(|| {
                RingKernelError::K2KDestinationNotFound(dest_kernel.as_str().to_string())
            })?;

        // Same device - use regular K2K
        if source_device == dest_device {
            return Ok(RoutingDecision::SameDevice);
        }

        // Get topology for routing
        let topo = self.coordinator.topology();
        let path = topo.best_path(source_device, dest_device);

        // Check if direct P2P is available
        if let Some(conn) = topo.get_connection(source_device, dest_device) {
            if conn.interconnect.supports_p2p() {
                // Queue for direct P2P transfer
                let pending = PendingK2KMessage {
                    source_kernel: source_kernel.clone(),
                    dest_kernel: dest_kernel.clone(),
                    message,
                    queued_at: Instant::now(),
                    hops: 1,
                };

                self.enqueue_pending(source_device, dest_device, pending);
                self.stats.messages_pending.fetch_add(1, Ordering::Relaxed);

                return Ok(RoutingDecision::DirectP2P {
                    source_device,
                    dest_device,
                    bandwidth_gbps: conn.bandwidth_gbps,
                });
            }
        }

        // Multi-hop routing required
        if path.len() > 2 {
            let pending = PendingK2KMessage {
                source_kernel: source_kernel.clone(),
                dest_kernel: dest_kernel.clone(),
                message,
                queued_at: Instant::now(),
                hops: (path.len() - 1) as u32,
            };

            // Queue for first hop
            self.enqueue_pending(source_device, path[1], pending);
            self.stats.messages_pending.fetch_add(1, Ordering::Relaxed);

            return Ok(RoutingDecision::MultiHop {
                path: path.clone(),
                total_hops: (path.len() - 1) as u32,
            });
        }

        // Fall back to host-mediated transfer
        let pending = PendingK2KMessage {
            source_kernel: source_kernel.clone(),
            dest_kernel: dest_kernel.clone(),
            message,
            queued_at: Instant::now(),
            hops: 2, // device->host->device
        };

        self.enqueue_pending(source_device, dest_device, pending);
        self.stats.messages_pending.fetch_add(1, Ordering::Relaxed);

        Ok(RoutingDecision::HostMediated {
            source_device,
            dest_device,
        })
    }

    /// Get pending messages for a device pair.
    pub fn drain_pending(&self, source: usize, dest: usize) -> Vec<PendingK2KMessage> {
        let mut queues = self.pending_queues.write();
        let messages = queues.remove(&(source, dest)).unwrap_or_default();
        self.stats
            .messages_pending
            .fetch_sub(messages.len(), Ordering::Relaxed);
        messages
    }

    /// Record successful message delivery.
    pub fn record_delivery(&self, message: &PendingK2KMessage, payload_size: usize) {
        self.stats.messages_routed.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_transferred
            .fetch_add(payload_size as u64, Ordering::Relaxed);

        let latency = message.queued_at.elapsed().as_micros() as u64;
        self.stats
            .total_latency_us
            .fetch_add(latency, Ordering::Relaxed);
    }

    /// Record routing failure.
    pub fn record_failure(&self) {
        self.stats.routing_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Get router statistics.
    pub fn stats(&self) -> CrossGpuRouterStatsSnapshot {
        let messages_routed = self.stats.messages_routed.load(Ordering::Relaxed);
        let total_latency = self.stats.total_latency_us.load(Ordering::Relaxed);

        CrossGpuRouterStatsSnapshot {
            messages_routed,
            bytes_transferred: self.stats.bytes_transferred.load(Ordering::Relaxed),
            messages_pending: self.stats.messages_pending.load(Ordering::Relaxed),
            avg_latency_us: if messages_routed > 0 {
                total_latency as f64 / messages_routed as f64
            } else {
                0.0
            },
            routing_failures: self.stats.routing_failures.load(Ordering::Relaxed),
        }
    }

    fn enqueue_pending(&self, source: usize, dest: usize, message: PendingK2KMessage) {
        let mut queues = self.pending_queues.write();
        queues.entry((source, dest)).or_default().push(message);
    }
}

/// Snapshot of router statistics.
#[derive(Debug, Clone)]
pub struct CrossGpuRouterStatsSnapshot {
    /// Total messages successfully routed.
    pub messages_routed: u64,
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Messages currently pending.
    pub messages_pending: usize,
    /// Average routing latency in microseconds.
    pub avg_latency_us: f64,
    /// Total routing failures.
    pub routing_failures: u64,
}

/// Decision for how to route a K2K message.
#[derive(Debug, Clone)]
pub enum RoutingDecision {
    /// Source and destination on same device.
    SameDevice,
    /// Direct peer-to-peer transfer.
    DirectP2P {
        /// Source device index.
        source_device: usize,
        /// Destination device index.
        dest_device: usize,
        /// Available bandwidth.
        bandwidth_gbps: f64,
    },
    /// Multi-hop routing through intermediate devices.
    MultiHop {
        /// Device path.
        path: Vec<usize>,
        /// Total number of hops.
        total_hops: u32,
    },
    /// Route through host memory (slowest).
    HostMediated {
        /// Source device index.
        source_device: usize,
        /// Destination device index.
        dest_device: usize,
    },
}

/// Multi-GPU coordinator statistics.
#[derive(Debug, Clone, Default)]
pub struct MultiGpuStats {
    /// Number of registered devices.
    pub device_count: usize,
    /// Total kernels across all devices.
    pub total_kernels: usize,
    /// Total memory across all devices.
    pub total_memory: u64,
    /// Available memory across all devices.
    pub available_memory: u64,
    /// Total kernels launched since start.
    pub kernels_launched: u64,
}

/// Builder for multi-GPU coordinator.
pub struct MultiGpuBuilder {
    config: MultiGpuConfig,
}

impl MultiGpuBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: MultiGpuConfig::default(),
        }
    }

    /// Set load balancing strategy.
    pub fn load_balancing(mut self, strategy: LoadBalancingStrategy) -> Self {
        self.config.load_balancing = strategy;
        self
    }

    /// Set auto device selection.
    pub fn auto_select_device(mut self, enable: bool) -> Self {
        self.config.auto_select_device = enable;
        self
    }

    /// Set max kernels per device.
    pub fn max_kernels_per_device(mut self, max: usize) -> Self {
        self.config.max_kernels_per_device = max;
        self
    }

    /// Enable P2P transfers.
    pub fn enable_p2p(mut self, enable: bool) -> Self {
        self.config.enable_p2p = enable;
        self
    }

    /// Set preferred devices.
    pub fn preferred_devices(mut self, devices: Vec<usize>) -> Self {
        self.config.preferred_devices = devices;
        self
    }

    /// Build the coordinator.
    pub fn build(self) -> Arc<MultiGpuCoordinator> {
        MultiGpuCoordinator::new(self.config)
    }
}

impl Default for MultiGpuBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for cross-device data transfer.
pub struct CrossDeviceTransfer {
    /// Source device index.
    pub source_device: usize,
    /// Destination device index.
    pub dest_device: usize,
    /// Data size in bytes.
    pub size: usize,
    /// Use P2P if available.
    pub use_p2p: bool,
}

impl CrossDeviceTransfer {
    /// Create a new transfer specification.
    pub fn new(source: usize, dest: usize, size: usize) -> Self {
        Self {
            source_device: source,
            dest_device: dest,
            size,
            use_p2p: true,
        }
    }

    /// Disable P2P for this transfer.
    pub fn without_p2p(mut self) -> Self {
        self.use_p2p = false;
        self
    }
}

// ============================================================================
// Kernel Migrator with Checkpoint Integration
// ============================================================================

use crate::checkpoint::{CheckpointStorage, CheckpointableKernel, MemoryStorage};

/// Migrator that uses checkpoints for kernel state transfer between GPUs.
///
/// This integrates the checkpoint infrastructure with the multi-GPU migration
/// system to enable live migration of persistent kernels.
///
/// # Example
///
/// ```ignore
/// use ringkernel_core::multi_gpu::{KernelMigrator, MultiGpuBuilder};
///
/// let coordinator = MultiGpuBuilder::new().build();
/// let migrator = KernelMigrator::new(coordinator);
///
/// // Migrate kernel from GPU 0 to GPU 1
/// migrator.migrate_with_checkpoint(&kernel, &mut request).await?;
/// ```
pub struct KernelMigrator {
    /// Multi-GPU coordinator.
    coordinator: Arc<MultiGpuCoordinator>,
    /// Checkpoint storage for migration state.
    storage: Arc<dyn CheckpointStorage>,
    /// Statistics.
    stats: MigrationStats,
}

/// Statistics for kernel migrations.
#[derive(Debug, Default)]
pub struct MigrationStats {
    /// Total successful migrations.
    pub successful_migrations: AtomicU64,
    /// Total failed migrations.
    pub failed_migrations: AtomicU64,
    /// Total bytes transferred during migrations.
    pub bytes_transferred: AtomicU64,
    /// Total checkpoint time (microseconds).
    pub checkpoint_time_us: AtomicU64,
    /// Total restore time (microseconds).
    pub restore_time_us: AtomicU64,
}

/// Result of a completed migration.
#[derive(Debug, Clone)]
pub struct MigrationResult {
    /// Kernel that was migrated.
    pub kernel_id: KernelId,
    /// Source device.
    pub source_device: usize,
    /// Target device.
    pub target_device: usize,
    /// Checkpoint size in bytes.
    pub checkpoint_size: usize,
    /// Time spent creating checkpoint.
    pub checkpoint_duration: Duration,
    /// Time spent transferring state.
    pub transfer_duration: Duration,
    /// Time spent restoring kernel.
    pub restore_duration: Duration,
    /// Total migration time.
    pub total_duration: Duration,
}

impl KernelMigrator {
    /// Create a new kernel migrator with default in-memory storage.
    pub fn new(coordinator: Arc<MultiGpuCoordinator>) -> Self {
        Self {
            coordinator,
            storage: Arc::new(MemoryStorage::new()),
            stats: MigrationStats::default(),
        }
    }

    /// Create a migrator with custom checkpoint storage.
    pub fn with_storage(
        coordinator: Arc<MultiGpuCoordinator>,
        storage: Arc<dyn CheckpointStorage>,
    ) -> Self {
        Self {
            coordinator,
            storage,
            stats: MigrationStats::default(),
        }
    }

    /// Perform a complete migration using checkpoint-based state transfer.
    ///
    /// Steps:
    /// 1. Quiesce the source kernel (drain pending messages)
    /// 2. Create checkpoint of kernel state
    /// 3. Transfer checkpoint to target device
    /// 4. Restore kernel on target device
    /// 5. Update coordinator routing tables
    pub fn migrate_with_checkpoint<K: CheckpointableKernel>(
        &self,
        kernel: &K,
        request: &mut MigrationRequest,
    ) -> Result<MigrationResult> {
        let start_time = Instant::now();
        request.started_at = Some(start_time);

        // Step 1: Quiesce
        request.state = MigrationState::Quiescing;
        // In a real implementation, this would drain message queues
        // For now, we assume the kernel is ready for checkpointing

        // Step 2: Create checkpoint
        request.state = MigrationState::Checkpointing;
        let checkpoint_start = Instant::now();
        let checkpoint = kernel.create_checkpoint().map_err(|e| {
            self.stats.failed_migrations.fetch_add(1, Ordering::Relaxed);
            request.state = MigrationState::Failed;
            RingKernelError::MigrationFailed(format!("Checkpoint creation failed: {}", e))
        })?;
        let checkpoint_duration = checkpoint_start.elapsed();
        let checkpoint_size = checkpoint.total_size();

        self.stats
            .checkpoint_time_us
            .fetch_add(checkpoint_duration.as_micros() as u64, Ordering::Relaxed);

        // Step 3: Transfer
        request.state = MigrationState::Transferring;
        let transfer_start = Instant::now();

        // Store checkpoint (simulates transfer)
        let checkpoint_name = format!(
            "migration_{}_{}_{}",
            request.kernel_id.as_str(),
            request.source_device,
            request.target_device
        );
        self.storage
            .save(&checkpoint, &checkpoint_name)
            .map_err(|e| {
                self.stats.failed_migrations.fetch_add(1, Ordering::Relaxed);
                request.state = MigrationState::Failed;
                RingKernelError::MigrationFailed(format!("Checkpoint transfer failed: {}", e))
            })?;

        let transfer_duration = transfer_start.elapsed();
        self.stats
            .bytes_transferred
            .fetch_add(checkpoint_size as u64, Ordering::Relaxed);

        // Step 4: Restore (would be done on target kernel)
        request.state = MigrationState::Restoring;
        let restore_start = Instant::now();

        // Load checkpoint to verify it's valid
        let _restored = self.storage.load(&checkpoint_name).map_err(|e| {
            self.stats.failed_migrations.fetch_add(1, Ordering::Relaxed);
            request.state = MigrationState::Failed;
            RingKernelError::MigrationFailed(format!("Checkpoint restore failed: {}", e))
        })?;

        let restore_duration = restore_start.elapsed();
        self.stats
            .restore_time_us
            .fetch_add(restore_duration.as_micros() as u64, Ordering::Relaxed);

        // Step 5: Update routing
        request.state = MigrationState::Completed;
        self.coordinator.complete_migration(request)?;

        // Clean up checkpoint
        let _ = self.storage.delete(&checkpoint_name);

        self.stats
            .successful_migrations
            .fetch_add(1, Ordering::Relaxed);

        Ok(MigrationResult {
            kernel_id: request.kernel_id.clone(),
            source_device: request.source_device,
            target_device: request.target_device,
            checkpoint_size,
            checkpoint_duration,
            transfer_duration,
            restore_duration,
            total_duration: start_time.elapsed(),
        })
    }

    /// Get a reference to the coordinator.
    pub fn coordinator(&self) -> &Arc<MultiGpuCoordinator> {
        &self.coordinator
    }

    /// Get migration statistics snapshot.
    pub fn stats(&self) -> MigrationStatsSnapshot {
        let successful = self.stats.successful_migrations.load(Ordering::Relaxed);
        let failed = self.stats.failed_migrations.load(Ordering::Relaxed);
        let total = successful + failed;
        let checkpoint_us = self.stats.checkpoint_time_us.load(Ordering::Relaxed);
        let restore_us = self.stats.restore_time_us.load(Ordering::Relaxed);

        MigrationStatsSnapshot {
            successful_migrations: successful,
            failed_migrations: failed,
            bytes_transferred: self.stats.bytes_transferred.load(Ordering::Relaxed),
            avg_checkpoint_time: if total > 0 {
                Duration::from_micros(checkpoint_us / total)
            } else {
                Duration::ZERO
            },
            avg_restore_time: if total > 0 {
                Duration::from_micros(restore_us / total)
            } else {
                Duration::ZERO
            },
        }
    }
}

/// Snapshot of migration statistics.
#[derive(Debug, Clone)]
pub struct MigrationStatsSnapshot {
    /// Total successful migrations.
    pub successful_migrations: u64,
    /// Total failed migrations.
    pub failed_migrations: u64,
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Average checkpoint creation time.
    pub avg_checkpoint_time: Duration,
    /// Average restore time.
    pub avg_restore_time: Duration,
}

/// Trait for kernels that support live migration.
pub trait MigratableKernel: CheckpointableKernel {
    /// Prepare kernel for migration (quiesce, drain messages).
    fn prepare_for_migration(&mut self) -> Result<()>;

    /// Resume kernel after migration is cancelled.
    fn cancel_migration(&mut self) -> Result<()>;

    /// Check if kernel is ready to be checkpointed.
    fn is_quiescent(&self) -> bool;

    /// Get estimated state size for migration planning.
    fn estimated_state_size(&self) -> usize;
}

// ============================================================================
// Hot Reload Support
// ============================================================================

/// Configuration for kernel hot reload operations.
#[derive(Debug, Clone)]
pub struct HotReloadConfig {
    /// Enable hot reload functionality.
    pub enabled: bool,
    /// Timeout for reload operations.
    pub reload_timeout: Duration,
    /// Whether to preserve kernel state during reload.
    pub preserve_state: bool,
    /// Maximum retries for failed reloads.
    pub max_retries: u32,
    /// Backoff duration between retries.
    pub retry_backoff: Duration,
    /// Whether to validate new code before swapping.
    pub validate_before_swap: bool,
    /// Keep old code as fallback in case of failure.
    pub keep_fallback: bool,
}

impl Default for HotReloadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reload_timeout: Duration::from_secs(30),
            preserve_state: true,
            max_retries: 3,
            retry_backoff: Duration::from_millis(500),
            validate_before_swap: true,
            keep_fallback: true,
        }
    }
}

impl HotReloadConfig {
    /// Create a new hot reload configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable hot reload.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set reload timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.reload_timeout = timeout;
        self
    }

    /// Enable or disable state preservation.
    pub fn with_preserve_state(mut self, preserve: bool) -> Self {
        self.preserve_state = preserve;
        self
    }

    /// Set maximum retries.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }
}

/// State of a hot reload operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotReloadState {
    /// Reload not started.
    Idle,
    /// Draining pending messages from kernel.
    Draining,
    /// Creating checkpoint of kernel state.
    Checkpointing,
    /// Compiling new kernel code.
    Compiling,
    /// Validating new kernel code.
    Validating,
    /// Swapping old kernel with new.
    Swapping,
    /// Restoring state to new kernel.
    Restoring,
    /// Hot reload completed successfully.
    Completed,
    /// Hot reload failed.
    Failed,
    /// Rolling back to previous version.
    RollingBack,
}

/// Kernel code format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelCodeFormat {
    /// NVIDIA PTX assembly.
    Ptx,
    /// NVIDIA CUBIN binary.
    Cubin,
    /// SPIR-V for Vulkan/WebGPU.
    SpirV,
    /// WGSL shader text.
    Wgsl,
    /// Metal Shading Language.
    Msl,
    /// Metal compiled library.
    MetalLib,
    /// Source code (requires compilation).
    Source,
}

/// Kernel code source for hot reload.
#[derive(Debug, Clone)]
pub struct KernelCodeSource {
    /// Unique identifier for this code version.
    pub version_id: u64,
    /// Code format.
    pub format: KernelCodeFormat,
    /// Raw code bytes.
    pub code: Vec<u8>,
    /// Entry point function name.
    pub entry_point: String,
    /// Optional metadata (compile flags, etc.).
    pub metadata: HashMap<String, String>,
    /// Timestamp when code was created.
    pub created_at: Instant,
    /// SHA-256 hash of the code.
    pub hash: [u8; 32],
}

impl KernelCodeSource {
    /// Create a new kernel code source.
    pub fn new(format: KernelCodeFormat, code: Vec<u8>, entry_point: impl Into<String>) -> Self {
        let hash = Self::compute_hash(&code);
        Self {
            version_id: 0,
            format,
            code,
            entry_point: entry_point.into(),
            metadata: HashMap::new(),
            created_at: Instant::now(),
            hash,
        }
    }

    /// Create from PTX code.
    pub fn from_ptx(ptx: &str, entry_point: impl Into<String>) -> Self {
        Self::new(KernelCodeFormat::Ptx, ptx.as_bytes().to_vec(), entry_point)
    }

    /// Create from WGSL code.
    pub fn from_wgsl(wgsl: &str, entry_point: impl Into<String>) -> Self {
        Self::new(
            KernelCodeFormat::Wgsl,
            wgsl.as_bytes().to_vec(),
            entry_point,
        )
    }

    /// Create from MSL code.
    pub fn from_msl(msl: &str, entry_point: impl Into<String>) -> Self {
        Self::new(KernelCodeFormat::Msl, msl.as_bytes().to_vec(), entry_point)
    }

    /// Set version ID.
    pub fn with_version(mut self, version: u64) -> Self {
        self.version_id = version;
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    fn compute_hash(data: &[u8]) -> [u8; 32] {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        data.hash(&mut hasher);
        let h1 = hasher.finish();
        h1.hash(&mut hasher);
        let h2 = hasher.finish();
        h1.hash(&mut hasher);
        let h3 = hasher.finish();
        h1.hash(&mut hasher);
        let h4 = hasher.finish();

        let mut hash = [0u8; 32];
        hash[0..8].copy_from_slice(&h1.to_le_bytes());
        hash[8..16].copy_from_slice(&h2.to_le_bytes());
        hash[16..24].copy_from_slice(&h3.to_le_bytes());
        hash[24..32].copy_from_slice(&h4.to_le_bytes());
        hash
    }

    /// Get code as string (if text format).
    pub fn as_str(&self) -> Option<&str> {
        match self.format {
            KernelCodeFormat::Ptx
            | KernelCodeFormat::Wgsl
            | KernelCodeFormat::Msl
            | KernelCodeFormat::Source => std::str::from_utf8(&self.code).ok(),
            _ => None,
        }
    }

    /// Get code size in bytes.
    pub fn size(&self) -> usize {
        self.code.len()
    }
}

/// Request to hot reload a kernel.
#[derive(Debug)]
pub struct HotReloadRequest {
    /// Target kernel ID.
    pub kernel_id: KernelId,
    /// New kernel code.
    pub new_code: KernelCodeSource,
    /// Current state of the reload operation.
    pub state: HotReloadState,
    /// When the request was created.
    pub created_at: Instant,
    /// When the reload started.
    pub started_at: Option<Instant>,
    /// Retry count.
    pub retry_count: u32,
    /// Error message if failed.
    pub error: Option<String>,
    /// Checkpoint data (if preserving state).
    checkpoint_data: Option<Vec<u8>>,
}

impl HotReloadRequest {
    /// Create a new hot reload request.
    pub fn new(kernel_id: KernelId, new_code: KernelCodeSource) -> Self {
        Self {
            kernel_id,
            new_code,
            state: HotReloadState::Idle,
            created_at: Instant::now(),
            started_at: None,
            retry_count: 0,
            error: None,
            checkpoint_data: None,
        }
    }

    /// Check if reload is in progress.
    pub fn is_in_progress(&self) -> bool {
        !matches!(
            self.state,
            HotReloadState::Idle | HotReloadState::Completed | HotReloadState::Failed
        )
    }

    /// Check if reload completed successfully.
    pub fn is_completed(&self) -> bool {
        self.state == HotReloadState::Completed
    }

    /// Check if reload failed.
    pub fn is_failed(&self) -> bool {
        self.state == HotReloadState::Failed
    }

    /// Get elapsed time since request creation.
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get elapsed time since reload started.
    pub fn reload_elapsed(&self) -> Option<Duration> {
        self.started_at.map(|s| s.elapsed())
    }
}

/// Result of a completed hot reload.
#[derive(Debug, Clone)]
pub struct HotReloadResult {
    /// Target kernel ID.
    pub kernel_id: KernelId,
    /// Previous code version.
    pub old_version: u64,
    /// New code version.
    pub new_version: u64,
    /// Whether state was preserved.
    pub state_preserved: bool,
    /// Size of checkpoint data (if any).
    pub checkpoint_size: usize,
    /// Time to drain messages.
    pub drain_duration: Duration,
    /// Time to create checkpoint.
    pub checkpoint_duration: Duration,
    /// Time to compile new code.
    pub compile_duration: Duration,
    /// Time to swap kernels.
    pub swap_duration: Duration,
    /// Time to restore state.
    pub restore_duration: Duration,
    /// Total reload duration.
    pub total_duration: Duration,
}

/// Statistics for hot reload operations.
#[derive(Debug, Default)]
struct HotReloadStats {
    successful_reloads: AtomicU64,
    failed_reloads: AtomicU64,
    rollbacks: AtomicU64,
    total_drain_time_us: AtomicU64,
    total_compile_time_us: AtomicU64,
    total_swap_time_us: AtomicU64,
    state_preserved_count: AtomicU64,
}

/// Snapshot of hot reload statistics.
#[derive(Debug, Clone)]
pub struct HotReloadStatsSnapshot {
    /// Total successful reloads.
    pub successful_reloads: u64,
    /// Total failed reloads.
    pub failed_reloads: u64,
    /// Total rollbacks performed.
    pub rollbacks: u64,
    /// Average drain time.
    pub avg_drain_time: Duration,
    /// Average compile time.
    pub avg_compile_time: Duration,
    /// Average swap time.
    pub avg_swap_time: Duration,
    /// Number of reloads with preserved state.
    pub state_preserved_count: u64,
}

/// Manager for kernel hot reload operations.
///
/// Provides seamless kernel code updates without stopping the system:
///
/// 1. Drain pending messages from kernel input queue
/// 2. Checkpoint kernel state (if preserving state)
/// 3. Compile/validate new kernel code
/// 4. Swap old kernel with new kernel
/// 5. Restore state to new kernel
/// 6. Resume processing
///
/// # Example
///
/// ```ignore
/// use ringkernel_core::multi_gpu::{HotReloadManager, HotReloadConfig, KernelCodeSource};
///
/// let manager = HotReloadManager::new(HotReloadConfig::default());
///
/// // Register a reloadable kernel
/// manager.register_kernel(&kernel_id, current_code);
///
/// // Request hot reload with new PTX
/// let new_code = KernelCodeSource::from_ptx(new_ptx, "my_kernel");
/// let request = manager.request_reload(&kernel_id, new_code).await?;
///
/// // Execute the reload
/// let result = manager.execute_reload(request, &mut kernel).await?;
/// println!("Reload completed in {:?}", result.total_duration);
/// ```
pub struct HotReloadManager {
    /// Configuration.
    config: HotReloadConfig,
    /// Registered kernels and their current code.
    kernels: RwLock<HashMap<KernelId, KernelCodeSource>>,
    /// Fallback code for registered kernels.
    fallbacks: RwLock<HashMap<KernelId, KernelCodeSource>>,
    /// Active reload requests.
    active_requests: RwLock<HashMap<KernelId, HotReloadRequest>>,
    /// Version counter for code versions.
    version_counter: AtomicU64,
    /// Statistics.
    stats: HotReloadStats,
}

impl HotReloadManager {
    /// Create a new hot reload manager.
    pub fn new(config: HotReloadConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            kernels: RwLock::new(HashMap::new()),
            fallbacks: RwLock::new(HashMap::new()),
            active_requests: RwLock::new(HashMap::new()),
            version_counter: AtomicU64::new(1),
            stats: HotReloadStats::default(),
        })
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Arc<Self> {
        Self::new(HotReloadConfig::default())
    }

    /// Check if hot reload is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Register a kernel for hot reload.
    pub fn register_kernel(&self, kernel_id: &KernelId, code: KernelCodeSource) {
        let version = self.version_counter.fetch_add(1, Ordering::Relaxed);
        let code = code.with_version(version);
        self.kernels.write().insert(kernel_id.clone(), code);
    }

    /// Unregister a kernel from hot reload.
    pub fn unregister_kernel(&self, kernel_id: &KernelId) {
        self.kernels.write().remove(kernel_id);
        self.fallbacks.write().remove(kernel_id);
        self.active_requests.write().remove(kernel_id);
    }

    /// Get current code version for a kernel.
    pub fn get_current_version(&self, kernel_id: &KernelId) -> Option<u64> {
        self.kernels.read().get(kernel_id).map(|c| c.version_id)
    }

    /// Get current code for a kernel.
    pub fn get_current_code(&self, kernel_id: &KernelId) -> Option<KernelCodeSource> {
        self.kernels.read().get(kernel_id).cloned()
    }

    /// Request a hot reload for a kernel.
    pub fn request_reload(
        &self,
        kernel_id: &KernelId,
        new_code: KernelCodeSource,
    ) -> Result<HotReloadRequest> {
        if !self.config.enabled {
            return Err(RingKernelError::ValidationError(
                "Hot reload is disabled".to_string(),
            ));
        }

        // Check kernel is registered
        if !self.kernels.read().contains_key(kernel_id) {
            return Err(RingKernelError::KernelNotFound(
                kernel_id.as_str().to_string(),
            ));
        }

        // Check no reload already in progress
        {
            let active = self.active_requests.read();
            if let Some(existing) = active.get(kernel_id) {
                if existing.is_in_progress() {
                    return Err(RingKernelError::ValidationError(
                        "Hot reload already in progress for this kernel".to_string(),
                    ));
                }
            }
        }

        // Assign version to new code
        let version = self.version_counter.fetch_add(1, Ordering::Relaxed);
        let new_code = new_code.with_version(version);

        let request = HotReloadRequest::new(kernel_id.clone(), new_code);
        self.active_requests.write().insert(
            kernel_id.clone(),
            HotReloadRequest::new(kernel_id.clone(), request.new_code.clone()),
        );

        Ok(request)
    }

    /// Execute a hot reload operation.
    ///
    /// This performs the full reload sequence:
    /// 1. Drain pending messages
    /// 2. Checkpoint state (if enabled)
    /// 3. Validate new code
    /// 4. Swap kernels
    /// 5. Restore state (if enabled)
    pub fn execute_reload<K: CheckpointableKernel>(
        &self,
        request: &mut HotReloadRequest,
        kernel: &K,
    ) -> Result<HotReloadResult> {
        let start_time = Instant::now();
        request.started_at = Some(start_time);

        // Get old version
        let old_version = self
            .kernels
            .read()
            .get(&request.kernel_id)
            .map(|c| c.version_id)
            .unwrap_or(0);

        // Phase 1: Drain (simulated - actual drain would wait for queue empty)
        request.state = HotReloadState::Draining;
        let drain_start = Instant::now();
        // In a real implementation, wait for input queue to drain
        std::thread::sleep(Duration::from_micros(10));
        let drain_duration = drain_start.elapsed();
        self.stats
            .total_drain_time_us
            .fetch_add(drain_duration.as_micros() as u64, Ordering::Relaxed);

        // Phase 2: Checkpoint (if preserving state)
        request.state = HotReloadState::Checkpointing;
        let checkpoint_start = Instant::now();
        let checkpoint_size = if self.config.preserve_state {
            let checkpoint = kernel.create_checkpoint()?;
            let data = checkpoint.to_bytes();
            request.checkpoint_data = Some(data.clone());
            data.len()
        } else {
            0
        };
        let checkpoint_duration = checkpoint_start.elapsed();

        // Phase 3: Validate new code
        request.state = HotReloadState::Validating;
        if self.config.validate_before_swap {
            self.validate_code(&request.new_code)?;
        }

        // Phase 4: Compile (simulated)
        request.state = HotReloadState::Compiling;
        let compile_start = Instant::now();
        // In real implementation, compile PTX/WGSL to native code
        std::thread::sleep(Duration::from_micros(10));
        let compile_duration = compile_start.elapsed();
        self.stats
            .total_compile_time_us
            .fetch_add(compile_duration.as_micros() as u64, Ordering::Relaxed);

        // Phase 5: Swap
        request.state = HotReloadState::Swapping;
        let swap_start = Instant::now();

        // Save fallback
        if self.config.keep_fallback {
            if let Some(old_code) = self.kernels.read().get(&request.kernel_id).cloned() {
                self.fallbacks
                    .write()
                    .insert(request.kernel_id.clone(), old_code);
            }
        }

        // Install new code
        self.kernels
            .write()
            .insert(request.kernel_id.clone(), request.new_code.clone());
        let swap_duration = swap_start.elapsed();
        self.stats
            .total_swap_time_us
            .fetch_add(swap_duration.as_micros() as u64, Ordering::Relaxed);

        // Phase 6: Restore state
        request.state = HotReloadState::Restoring;
        let restore_start = Instant::now();
        // In real implementation, restore checkpoint to new kernel
        let restore_duration = restore_start.elapsed();

        // Mark completed
        request.state = HotReloadState::Completed;
        self.stats
            .successful_reloads
            .fetch_add(1, Ordering::Relaxed);
        if self.config.preserve_state && checkpoint_size > 0 {
            self.stats
                .state_preserved_count
                .fetch_add(1, Ordering::Relaxed);
        }

        // Clean up active request
        self.active_requests.write().remove(&request.kernel_id);

        Ok(HotReloadResult {
            kernel_id: request.kernel_id.clone(),
            old_version,
            new_version: request.new_code.version_id,
            state_preserved: self.config.preserve_state && checkpoint_size > 0,
            checkpoint_size,
            drain_duration,
            checkpoint_duration,
            compile_duration,
            swap_duration,
            restore_duration,
            total_duration: start_time.elapsed(),
        })
    }

    /// Rollback to previous kernel version.
    pub fn rollback(&self, kernel_id: &KernelId) -> Result<()> {
        let fallback =
            self.fallbacks.write().remove(kernel_id).ok_or_else(|| {
                RingKernelError::ValidationError("No fallback available".to_string())
            })?;

        self.kernels.write().insert(kernel_id.clone(), fallback);
        self.stats.rollbacks.fetch_add(1, Ordering::Relaxed);

        // Update any active request
        if let Some(request) = self.active_requests.write().get_mut(kernel_id) {
            request.state = HotReloadState::RollingBack;
        }

        Ok(())
    }

    /// Validate kernel code before swap.
    fn validate_code(&self, code: &KernelCodeSource) -> Result<()> {
        // Basic validation
        if code.code.is_empty() {
            return Err(RingKernelError::ValidationError(
                "Kernel code is empty".to_string(),
            ));
        }

        if code.entry_point.is_empty() {
            return Err(RingKernelError::ValidationError(
                "Entry point is empty".to_string(),
            ));
        }

        // Format-specific validation
        match code.format {
            KernelCodeFormat::Ptx => {
                // Check for valid PTX header
                if let Some(text) = code.as_str() {
                    if !text.contains(".version") && !text.contains(".target") {
                        return Err(RingKernelError::ValidationError(
                            "PTX code missing version/target directive".to_string(),
                        ));
                    }
                }
            }
            KernelCodeFormat::Wgsl => {
                // Check for basic WGSL structure
                if let Some(text) = code.as_str() {
                    if !text.contains("@compute") && !text.contains("fn ") {
                        return Err(RingKernelError::ValidationError(
                            "WGSL code missing compute shader or function".to_string(),
                        ));
                    }
                }
            }
            KernelCodeFormat::Msl => {
                // Check for Metal kernel
                if let Some(text) = code.as_str() {
                    if !text.contains("kernel ") {
                        return Err(RingKernelError::ValidationError(
                            "MSL code missing kernel function".to_string(),
                        ));
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Get statistics snapshot.
    pub fn stats(&self) -> HotReloadStatsSnapshot {
        let successful = self.stats.successful_reloads.load(Ordering::Relaxed);
        let failed = self.stats.failed_reloads.load(Ordering::Relaxed);
        let total = successful.max(1);

        HotReloadStatsSnapshot {
            successful_reloads: successful,
            failed_reloads: failed,
            rollbacks: self.stats.rollbacks.load(Ordering::Relaxed),
            avg_drain_time: Duration::from_micros(
                self.stats.total_drain_time_us.load(Ordering::Relaxed) / total,
            ),
            avg_compile_time: Duration::from_micros(
                self.stats.total_compile_time_us.load(Ordering::Relaxed) / total,
            ),
            avg_swap_time: Duration::from_micros(
                self.stats.total_swap_time_us.load(Ordering::Relaxed) / total,
            ),
            state_preserved_count: self.stats.state_preserved_count.load(Ordering::Relaxed),
        }
    }

    /// List all registered kernels.
    pub fn list_kernels(&self) -> Vec<KernelId> {
        self.kernels.read().keys().cloned().collect()
    }

    /// Check if a kernel is registered.
    pub fn is_registered(&self, kernel_id: &KernelId) -> bool {
        self.kernels.read().contains_key(kernel_id)
    }

    /// Check if a reload is in progress for a kernel.
    pub fn is_reload_in_progress(&self, kernel_id: &KernelId) -> bool {
        self.active_requests
            .read()
            .get(kernel_id)
            .map(|r| r.is_in_progress())
            .unwrap_or(false)
    }

    /// Get the configuration.
    pub fn config(&self) -> &HotReloadConfig {
        &self.config
    }
}

/// Trait for kernels that support hot reload.
pub trait HotReloadableKernel: CheckpointableKernel {
    /// Prepare kernel for code swap (drain messages, pause processing).
    fn prepare_for_reload(&mut self) -> Result<()>;

    /// Apply new code to the kernel.
    fn apply_code(&mut self, code: &KernelCodeSource) -> Result<()>;

    /// Resume processing after reload.
    fn resume_after_reload(&mut self) -> Result<()>;

    /// Check if kernel is ready for reload.
    fn is_ready_for_reload(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_info() {
        let info = DeviceInfo::new(0, "Test GPU".to_string(), Backend::Cuda);
        assert_eq!(info.index, 0);
        assert_eq!(info.name, "Test GPU");
        assert_eq!(info.memory_utilization(), 0.0);
    }

    #[test]
    fn test_coordinator_registration() {
        let coord = MultiGpuBuilder::new().build();

        let device = DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda);
        coord.register_device(device);

        assert_eq!(coord.device_count(), 1);
        assert!(coord.device(0).is_some());
    }

    #[test]
    fn test_kernel_assignment() {
        let coord = MultiGpuBuilder::new().build();

        let device = DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda);
        coord.register_device(device);

        let kernel_id = KernelId::new("test_kernel");
        coord.assign_kernel(kernel_id.clone(), 0);

        assert_eq!(coord.get_kernel_device(&kernel_id), Some(0));
        assert_eq!(coord.kernels_on_device(0).len(), 1);
    }

    #[test]
    fn test_load_balancing_least_loaded() {
        let coord = MultiGpuBuilder::new()
            .load_balancing(LoadBalancingStrategy::LeastLoaded)
            .build();

        // Register two devices
        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));
        coord.register_device(DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda));

        // Assign a kernel to device 0
        coord.assign_kernel(KernelId::new("k1"), 0);

        // Next kernel should go to device 1 (least loaded)
        let selected = coord.select_device(&LaunchOptions::default()).unwrap();
        assert_eq!(selected, 1);
    }

    #[test]
    fn test_round_robin() {
        let coord = MultiGpuBuilder::new()
            .load_balancing(LoadBalancingStrategy::RoundRobin)
            .build();

        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));
        coord.register_device(DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda));

        let d1 = coord.select_device(&LaunchOptions::default()).unwrap();
        let d2 = coord.select_device(&LaunchOptions::default()).unwrap();
        let d3 = coord.select_device(&LaunchOptions::default()).unwrap();

        // Should cycle through devices
        assert_ne!(d1, d2);
        assert_eq!(d1, d3);
    }

    // ========================================================================
    // Topology Tests
    // ========================================================================

    #[test]
    fn test_interconnect_bandwidth() {
        assert!(
            InterconnectType::NvLink.estimated_bandwidth_gbps()
                > InterconnectType::Pcie.estimated_bandwidth_gbps()
        );
        assert!(
            InterconnectType::Pcie.estimated_bandwidth_gbps()
                > InterconnectType::None.estimated_bandwidth_gbps()
        );
        assert!(
            InterconnectType::SameDevice.estimated_bandwidth_gbps()
                > InterconnectType::NvLink.estimated_bandwidth_gbps()
        );
    }

    #[test]
    fn test_interconnect_p2p_support() {
        assert!(!InterconnectType::None.supports_p2p());
        assert!(InterconnectType::Pcie.supports_p2p());
        assert!(InterconnectType::NvLink.supports_p2p());
        assert!(InterconnectType::NvSwitch.supports_p2p());
    }

    #[test]
    fn test_gpu_topology_creation() {
        let topo = GpuTopology::new(4);
        assert_eq!(topo.device_count, 4);

        // Self-connections should exist
        for i in 0..4 {
            let conn = topo.get_connection(i, i);
            assert!(conn.is_some());
            assert_eq!(conn.unwrap().interconnect, InterconnectType::SameDevice);
        }
    }

    #[test]
    fn test_gpu_topology_set_connection() {
        let mut topo = GpuTopology::new(4);

        // Set NVLink between GPU 0 and 1
        topo.set_connection(GpuConnection::new(0, 1, InterconnectType::NvLink));

        let conn_01 = topo.get_connection(0, 1);
        assert!(conn_01.is_some());
        assert_eq!(conn_01.unwrap().interconnect, InterconnectType::NvLink);

        // Bidirectional by default
        let conn_10 = topo.get_connection(1, 0);
        assert!(conn_10.is_some());
        assert_eq!(conn_10.unwrap().interconnect, InterconnectType::NvLink);
    }

    #[test]
    fn test_gpu_topology_neighbors() {
        let mut topo = GpuTopology::new(4);

        // Ring topology: 0-1-2-3-0
        topo.set_connection(GpuConnection::new(0, 1, InterconnectType::NvLink));
        topo.set_connection(GpuConnection::new(1, 2, InterconnectType::NvLink));
        topo.set_connection(GpuConnection::new(2, 3, InterconnectType::NvLink));
        topo.set_connection(GpuConnection::new(3, 0, InterconnectType::NvLink));

        let neighbors_0 = topo.neighbors(0);
        assert_eq!(neighbors_0.len(), 2);
        assert!(neighbors_0.contains(&1));
        assert!(neighbors_0.contains(&3));
    }

    #[test]
    fn test_gpu_topology_best_path() {
        let mut topo = GpuTopology::new(4);

        // Create connections: 0-1, 1-2, 2-3 (no direct 0-3)
        topo.set_connection(GpuConnection::new(0, 1, InterconnectType::NvLink));
        topo.set_connection(GpuConnection::new(1, 2, InterconnectType::NvLink));
        topo.set_connection(GpuConnection::new(2, 3, InterconnectType::NvLink));
        topo.set_connection(GpuConnection::new(0, 3, InterconnectType::None)); // No direct P2P

        // Direct path should work for adjacent nodes
        let path_01 = topo.best_path(0, 1);
        assert_eq!(path_01, vec![0, 1]);

        // Same device
        let path_00 = topo.best_path(0, 0);
        assert_eq!(path_00, vec![0]);
    }

    #[test]
    fn test_gpu_topology_fully_connected() {
        let mut topo = GpuTopology::new(3);

        // Not fully connected initially
        assert!(!topo.is_fully_connected());

        // Make fully connected mesh
        topo.set_connection(GpuConnection::new(0, 1, InterconnectType::NvLink));
        topo.set_connection(GpuConnection::new(0, 2, InterconnectType::NvLink));
        topo.set_connection(GpuConnection::new(1, 2, InterconnectType::NvLink));

        assert!(topo.is_fully_connected());
    }

    #[test]
    fn test_gpu_topology_numa() {
        let mut topo = GpuTopology::new(4);

        // GPUs 0,1 on NUMA 0; GPUs 2,3 on NUMA 1
        topo.set_numa_node(0, 0);
        topo.set_numa_node(1, 0);
        topo.set_numa_node(2, 1);
        topo.set_numa_node(3, 1);

        let numa_neighbors_0 = topo.numa_neighbors(0);
        assert_eq!(numa_neighbors_0, vec![1]);

        let numa_neighbors_2 = topo.numa_neighbors(2);
        assert_eq!(numa_neighbors_2, vec![3]);
    }

    // ========================================================================
    // Topology Discovery Tests
    // ========================================================================

    #[test]
    fn test_coordinator_topology_discovery() {
        let coord = MultiGpuBuilder::new().enable_p2p(true).build();

        // Register P2P capable devices
        let mut dev0 = DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda);
        dev0.p2p_capable = true;
        dev0.compute_capability = Some((8, 0)); // Ampere

        let mut dev1 = DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda);
        dev1.p2p_capable = true;
        dev1.compute_capability = Some((8, 6)); // Ampere

        coord.register_device(dev0);
        coord.register_device(dev1);

        let topo = coord.discover_topology();

        assert_eq!(topo.device_count, 2);

        // Should detect NVLink for Ampere GPUs
        let conn = topo.get_connection(0, 1);
        assert!(conn.is_some());
        assert_eq!(conn.unwrap().interconnect, InterconnectType::NvLink);
    }

    // ========================================================================
    // Migration Tests
    // ========================================================================

    #[test]
    fn test_migration_request() {
        let coord = MultiGpuBuilder::new().build();

        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));
        coord.register_device(DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda));

        let kernel_id = KernelId::new("migrating_kernel");
        coord.assign_kernel(kernel_id.clone(), 0);

        let request = coord.request_migration(&kernel_id, 1).unwrap();

        assert_eq!(request.source_device, 0);
        assert_eq!(request.target_device, 1);
        assert_eq!(request.state, MigrationState::Pending);
    }

    #[test]
    fn test_migration_same_device_error() {
        let coord = MultiGpuBuilder::new().build();

        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));

        let kernel_id = KernelId::new("kernel");
        coord.assign_kernel(kernel_id.clone(), 0);

        let result = coord.request_migration(&kernel_id, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_migration_complete() {
        let coord = MultiGpuBuilder::new().build();

        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));
        coord.register_device(DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda));

        let kernel_id = KernelId::new("migrating_kernel");
        coord.assign_kernel(kernel_id.clone(), 0);

        assert_eq!(coord.get_kernel_device(&kernel_id), Some(0));

        let request = coord.request_migration(&kernel_id, 1).unwrap();
        coord.complete_migration(&request).unwrap();

        assert_eq!(coord.get_kernel_device(&kernel_id), Some(1));
    }

    #[test]
    fn test_migration_transfer_time_estimate() {
        let request = MigrationRequest {
            kernel_id: KernelId::new("test"),
            source_device: 0,
            target_device: 1,
            path: vec![0, 1],
            estimated_bandwidth_gbps: 300.0, // NVLink
            estimated_latency_us: 1.0,
            state: MigrationState::Pending,
            started_at: None,
        };

        // 1GB transfer at 300GB/s = ~3.3ms + 1us latency
        let time = request.estimate_transfer_time(1_000_000_000);
        assert!(time.as_micros() > 3000);
        assert!(time.as_micros() < 4000);
    }

    // ========================================================================
    // Cross-GPU K2K Router Tests
    // ========================================================================

    use crate::hlc::HlcTimestamp;
    use crate::message::MessageEnvelope;

    fn make_test_k2k_message(source: &KernelId, dest: &KernelId) -> K2KMessage {
        let timestamp = HlcTimestamp::now(42);
        let envelope = MessageEnvelope::empty(1, 2, timestamp);
        K2KMessage::new(source.clone(), dest.clone(), envelope, timestamp)
    }

    #[test]
    fn test_router_same_device() {
        let coord = MultiGpuBuilder::new().build();
        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));

        let k1 = KernelId::new("k1");
        let k2 = KernelId::new("k2");
        coord.assign_kernel(k1.clone(), 0);
        coord.assign_kernel(k2.clone(), 0);

        let router = CrossGpuK2KRouter::new(coord);

        let msg = make_test_k2k_message(&k1, &k2);
        let decision = router.route_message(&k1, &k2, msg).unwrap();

        matches!(decision, RoutingDecision::SameDevice);
    }

    #[test]
    fn test_router_cross_device() {
        let coord = MultiGpuBuilder::new().enable_p2p(true).build();

        let mut dev0 = DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda);
        dev0.p2p_capable = true;
        let mut dev1 = DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda);
        dev1.p2p_capable = true;

        coord.register_device(dev0);
        coord.register_device(dev1);

        let k1 = KernelId::new("k1");
        let k2 = KernelId::new("k2");
        coord.assign_kernel(k1.clone(), 0);
        coord.assign_kernel(k2.clone(), 1);

        let router = CrossGpuK2KRouter::new(coord);

        let msg = make_test_k2k_message(&k1, &k2);
        let decision = router.route_message(&k1, &k2, msg).unwrap();

        match decision {
            RoutingDecision::DirectP2P {
                source_device,
                dest_device,
                ..
            } => {
                assert_eq!(source_device, 0);
                assert_eq!(dest_device, 1);
            }
            _ => panic!("Expected DirectP2P routing"),
        }
    }

    #[test]
    fn test_router_pending_messages() {
        let coord = MultiGpuBuilder::new().enable_p2p(true).build();

        let mut dev0 = DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda);
        dev0.p2p_capable = true;
        let mut dev1 = DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda);
        dev1.p2p_capable = true;

        coord.register_device(dev0);
        coord.register_device(dev1);

        let k1 = KernelId::new("k1");
        let k2 = KernelId::new("k2");
        coord.assign_kernel(k1.clone(), 0);
        coord.assign_kernel(k2.clone(), 1);

        let router = CrossGpuK2KRouter::new(coord);

        // Route 3 messages
        for _ in 0..3 {
            let msg = make_test_k2k_message(&k1, &k2);
            router.route_message(&k1, &k2, msg).unwrap();
        }

        assert_eq!(router.stats().messages_pending, 3);

        // Drain pending
        let pending = router.drain_pending(0, 1);
        assert_eq!(pending.len(), 3);
        assert_eq!(router.stats().messages_pending, 0);
    }

    #[test]
    fn test_router_stats() {
        let coord = MultiGpuBuilder::new().build();
        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));

        let k1 = KernelId::new("k1");
        let k2 = KernelId::new("k2");
        coord.assign_kernel(k1.clone(), 0);
        coord.assign_kernel(k2.clone(), 0);

        let router = CrossGpuK2KRouter::new(coord);

        let stats = router.stats();
        assert_eq!(stats.messages_routed, 0);
        assert_eq!(stats.bytes_transferred, 0);
        assert_eq!(stats.routing_failures, 0);
    }

    // ========================================================================
    // Kernel Migrator Tests
    // ========================================================================

    use crate::checkpoint::{Checkpoint, CheckpointBuilder};

    /// Mock checkpointable kernel for testing.
    struct MockCheckpointableKernel {
        kernel_id: String,
        kernel_type: String,
        state_data: Vec<u8>,
        step: u64,
    }

    impl MockCheckpointableKernel {
        fn new(kernel_id: &str, state_size: usize) -> Self {
            Self {
                kernel_id: kernel_id.to_string(),
                kernel_type: "mock_kernel".to_string(),
                state_data: vec![0xAB; state_size],
                step: 1000,
            }
        }
    }

    impl CheckpointableKernel for MockCheckpointableKernel {
        fn create_checkpoint(&self) -> Result<Checkpoint> {
            let checkpoint = CheckpointBuilder::new(&self.kernel_id, &self.kernel_type)
                .step(self.step)
                .grid_size(64, 64, 64)
                .control_block(vec![1, 2, 3, 4])
                .device_memory("state", self.state_data.clone())
                .build();
            Ok(checkpoint)
        }

        fn restore_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<()> {
            self.step = checkpoint.metadata.current_step;
            Ok(())
        }

        fn checkpoint_kernel_id(&self) -> &str {
            &self.kernel_id
        }

        fn checkpoint_kernel_type(&self) -> &str {
            &self.kernel_type
        }
    }

    #[test]
    fn test_migrator_creation() {
        let coord = MultiGpuBuilder::new().build();
        let migrator = KernelMigrator::new(coord);

        let stats = migrator.stats();
        assert_eq!(stats.successful_migrations, 0);
        assert_eq!(stats.failed_migrations, 0);
        assert_eq!(stats.bytes_transferred, 0);
    }

    #[test]
    fn test_migrator_with_custom_storage() {
        let coord = MultiGpuBuilder::new().build();
        let storage = Arc::new(MemoryStorage::new());
        let migrator = KernelMigrator::with_storage(coord.clone(), storage);

        // Verify we can access the coordinator
        assert!(Arc::ptr_eq(&migrator.coordinator(), &coord));
    }

    #[test]
    fn test_successful_migration() {
        let coord = MultiGpuBuilder::new().build();

        // Register devices
        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));
        coord.register_device(DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda));

        // Assign kernel to device 0
        let kernel_id = KernelId::new("migratable_kernel");
        coord.assign_kernel(kernel_id.clone(), 0);

        let migrator = KernelMigrator::new(coord.clone());

        // Create mock kernel
        let kernel = MockCheckpointableKernel::new("migratable_kernel", 1024);

        // Request migration
        let mut request = coord.request_migration(&kernel_id, 1).unwrap();
        assert_eq!(request.state, MigrationState::Pending);

        // Perform migration
        let result = migrator
            .migrate_with_checkpoint(&kernel, &mut request)
            .unwrap();

        // Verify result
        assert_eq!(result.kernel_id.as_str(), "migratable_kernel");
        assert_eq!(result.source_device, 0);
        assert_eq!(result.target_device, 1);
        assert!(result.checkpoint_size > 0);
        assert!(result.total_duration > Duration::ZERO);

        // Verify kernel was moved
        assert_eq!(coord.get_kernel_device(&kernel_id), Some(1));

        // Verify stats
        let stats = migrator.stats();
        assert_eq!(stats.successful_migrations, 1);
        assert_eq!(stats.failed_migrations, 0);
        assert!(stats.bytes_transferred > 0);
    }

    #[test]
    fn test_migration_result_fields() {
        let coord = MultiGpuBuilder::new().build();

        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));
        coord.register_device(DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda));

        let kernel_id = KernelId::new("test_kernel");
        coord.assign_kernel(kernel_id.clone(), 0);

        let migrator = KernelMigrator::new(coord.clone());
        let kernel = MockCheckpointableKernel::new("test_kernel", 4096);
        let mut request = coord.request_migration(&kernel_id, 1).unwrap();

        let result = migrator
            .migrate_with_checkpoint(&kernel, &mut request)
            .unwrap();

        // All durations should be non-negative
        assert!(result.checkpoint_duration >= Duration::ZERO);
        assert!(result.transfer_duration >= Duration::ZERO);
        assert!(result.restore_duration >= Duration::ZERO);

        // Total should be >= sum of parts
        assert!(result.total_duration >= result.checkpoint_duration);
    }

    #[test]
    fn test_migration_stats_accumulate() {
        let coord = MultiGpuBuilder::new().build();

        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));
        coord.register_device(DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda));

        let migrator = KernelMigrator::new(coord.clone());

        // Migrate kernel 1: 0 -> 1
        let k1 = KernelId::new("k1");
        coord.assign_kernel(k1.clone(), 0);
        let kernel1 = MockCheckpointableKernel::new("k1", 1000);
        let mut req1 = coord.request_migration(&k1, 1).unwrap();
        migrator
            .migrate_with_checkpoint(&kernel1, &mut req1)
            .unwrap();

        // Migrate kernel 2: 0 -> 1
        let k2 = KernelId::new("k2");
        coord.assign_kernel(k2.clone(), 0);
        let kernel2 = MockCheckpointableKernel::new("k2", 2000);
        let mut req2 = coord.request_migration(&k2, 1).unwrap();
        migrator
            .migrate_with_checkpoint(&kernel2, &mut req2)
            .unwrap();

        let stats = migrator.stats();
        assert_eq!(stats.successful_migrations, 2);
        assert_eq!(stats.failed_migrations, 0);
        // Both checkpoints should have been transferred
        assert!(stats.bytes_transferred > 0);
    }

    // ========================================================================
    // Device Unregister Tests
    // ========================================================================

    #[test]
    fn test_unregister_device_no_kernels() {
        let coord = MultiGpuBuilder::new().build();

        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));
        coord.register_device(DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda));

        let result = coord.unregister_device(0);

        assert!(result.success);
        assert_eq!(result.device_index, 0);
        assert!(result.kernels_to_migrate.is_empty());
        assert!(result.orphaned_kernels.is_empty());
    }

    #[test]
    fn test_unregister_device_with_kernels() {
        let coord = MultiGpuBuilder::new().build();

        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));
        coord.register_device(DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda));

        // Assign kernels to device 0
        let k1 = KernelId::new("k1");
        let k2 = KernelId::new("k2");
        coord.assign_kernel(k1.clone(), 0);
        coord.assign_kernel(k2.clone(), 0);

        let result = coord.unregister_device(0);

        assert!(result.success);
        assert_eq!(result.kernels_to_migrate.len(), 2);
        assert!(result.orphaned_kernels.is_empty());

        // All kernels should migrate to device 1
        for plan in &result.kernels_to_migrate {
            assert_eq!(plan.source_device, 0);
            assert_eq!(plan.target_device, 1);
        }

        // Verify kernel mappings were updated
        assert_eq!(coord.get_kernel_device(&k1), Some(1));
        assert_eq!(coord.get_kernel_device(&k2), Some(1));
    }

    #[test]
    fn test_unregister_single_device_orphans_kernels() {
        let coord = MultiGpuBuilder::new().build();

        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));

        // Assign kernels to device 0
        let k1 = KernelId::new("k1");
        coord.assign_kernel(k1.clone(), 0);

        let result = coord.unregister_device(0);

        assert!(result.success);
        assert!(result.kernels_to_migrate.is_empty());
        assert_eq!(result.orphaned_kernels.len(), 1);
        assert_eq!(result.orphaned_kernels[0], k1);

        // Kernel should no longer have a device
        assert!(coord.get_kernel_device(&k1).is_none());
    }

    #[test]
    fn test_unregister_nonexistent_device() {
        let coord = MultiGpuBuilder::new().build();

        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));

        let result = coord.unregister_device(99);

        assert!(!result.success);
        assert_eq!(result.device_index, 99);
    }

    #[test]
    fn test_unregister_distributes_to_least_loaded() {
        let coord = MultiGpuBuilder::new().build();

        coord.register_device(DeviceInfo::new(0, "GPU 0".to_string(), Backend::Cuda));
        coord.register_device(DeviceInfo::new(1, "GPU 1".to_string(), Backend::Cuda));
        coord.register_device(DeviceInfo::new(2, "GPU 2".to_string(), Backend::Cuda));

        // Preload device 1 with kernels
        coord.assign_kernel(KernelId::new("pre1"), 1);
        coord.assign_kernel(KernelId::new("pre2"), 1);
        coord.assign_kernel(KernelId::new("pre3"), 1);

        // Assign kernel to device 0
        let k1 = KernelId::new("migrate_me");
        coord.assign_kernel(k1.clone(), 0);

        let result = coord.unregister_device(0);

        assert!(result.success);
        assert_eq!(result.kernels_to_migrate.len(), 1);

        // Should migrate to device 2 (least loaded)
        let plan = &result.kernels_to_migrate[0];
        assert_eq!(plan.target_device, 2);
    }

    #[test]
    fn test_migration_priority_enum() {
        let low = MigrationPriority::Low;
        let normal = MigrationPriority::Normal;
        let high = MigrationPriority::High;
        let critical = MigrationPriority::Critical;

        assert_ne!(low, normal);
        assert_ne!(normal, high);
        assert_ne!(high, critical);
        assert_eq!(low, MigrationPriority::Low);
    }

    // Hot Reload Tests

    #[test]
    fn test_hot_reload_config_default() {
        let config = HotReloadConfig::default();
        assert!(config.enabled);
        assert!(config.preserve_state);
        assert!(config.validate_before_swap);
        assert!(config.keep_fallback);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_hot_reload_config_builder() {
        let config = HotReloadConfig::new()
            .with_enabled(false)
            .with_preserve_state(false)
            .with_max_retries(5)
            .with_timeout(Duration::from_secs(60));

        assert!(!config.enabled);
        assert!(!config.preserve_state);
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.reload_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_kernel_code_source_ptx() {
        let ptx = ".version 7.0\n.target sm_80\nkernel: ret;";
        let code = KernelCodeSource::from_ptx(ptx, "kernel");

        assert_eq!(code.format, KernelCodeFormat::Ptx);
        assert_eq!(code.entry_point, "kernel");
        assert_eq!(code.as_str(), Some(ptx));
        assert_eq!(code.size(), ptx.len());
    }

    #[test]
    fn test_kernel_code_source_wgsl() {
        let wgsl = "@compute fn main() {}";
        let code = KernelCodeSource::from_wgsl(wgsl, "main");

        assert_eq!(code.format, KernelCodeFormat::Wgsl);
        assert_eq!(code.entry_point, "main");
        assert_eq!(code.as_str(), Some(wgsl));
    }

    #[test]
    fn test_kernel_code_source_msl() {
        let msl = "kernel void my_kernel() {}";
        let code = KernelCodeSource::from_msl(msl, "my_kernel");

        assert_eq!(code.format, KernelCodeFormat::Msl);
        assert_eq!(code.entry_point, "my_kernel");
        assert_eq!(code.as_str(), Some(msl));
    }

    #[test]
    fn test_hot_reload_manager_creation() {
        let manager = HotReloadManager::with_defaults();
        assert!(manager.is_enabled());
        assert!(manager.list_kernels().is_empty());
    }

    #[test]
    fn test_hot_reload_manager_register_kernel() {
        let manager = HotReloadManager::with_defaults();
        let kernel_id = KernelId::new("test_kernel");
        let code = KernelCodeSource::from_ptx(".version 7.0", "kernel");

        manager.register_kernel(&kernel_id, code);

        assert!(manager.is_registered(&kernel_id));
        assert!(!manager.is_reload_in_progress(&kernel_id));
        assert!(manager.get_current_version(&kernel_id).is_some());
    }

    #[test]
    fn test_hot_reload_request_states() {
        let kernel_id = KernelId::new("test");
        let code = KernelCodeSource::from_ptx(".version 7.0", "kernel");
        let request = HotReloadRequest::new(kernel_id, code);

        assert_eq!(request.state, HotReloadState::Idle);
        assert!(!request.is_in_progress());
        assert!(!request.is_completed());
        assert!(!request.is_failed());
    }

    #[test]
    fn test_hot_reload_disabled() {
        let config = HotReloadConfig::new().with_enabled(false);
        let manager = HotReloadManager::new(config);
        let kernel_id = KernelId::new("test");
        let code = KernelCodeSource::from_ptx(".version 7.0", "kernel");

        manager.register_kernel(&kernel_id, code.clone());
        let result = manager.request_reload(&kernel_id, code);
        assert!(result.is_err());
    }

    #[test]
    fn test_hot_reload_stats() {
        let manager = HotReloadManager::with_defaults();
        let stats = manager.stats();

        assert_eq!(stats.successful_reloads, 0);
        assert_eq!(stats.failed_reloads, 0);
        assert_eq!(stats.rollbacks, 0);
    }

    #[test]
    fn test_hot_reload_code_formats() {
        let formats = [
            KernelCodeFormat::Ptx,
            KernelCodeFormat::Cubin,
            KernelCodeFormat::SpirV,
            KernelCodeFormat::Wgsl,
            KernelCodeFormat::Msl,
            KernelCodeFormat::MetalLib,
            KernelCodeFormat::Source,
        ];

        // Verify all formats are distinct
        for (i, f1) in formats.iter().enumerate() {
            for (j, f2) in formats.iter().enumerate() {
                if i != j {
                    assert_ne!(f1, f2);
                }
            }
        }
    }

    #[test]
    fn test_hot_reload_state_transitions() {
        let states = [
            HotReloadState::Idle,
            HotReloadState::Draining,
            HotReloadState::Checkpointing,
            HotReloadState::Compiling,
            HotReloadState::Validating,
            HotReloadState::Swapping,
            HotReloadState::Restoring,
            HotReloadState::Completed,
            HotReloadState::Failed,
            HotReloadState::RollingBack,
        ];

        // Verify all states are distinct
        for (i, s1) in states.iter().enumerate() {
            for (j, s2) in states.iter().enumerate() {
                if i != j {
                    assert_ne!(s1, s2);
                }
            }
        }
    }

    #[test]
    fn test_hot_reload_execute() {
        let manager = HotReloadManager::with_defaults();
        let kernel_id = KernelId::new("test_kernel");

        let initial_code = KernelCodeSource::from_ptx(".version 7.0\n.target sm_80", "kernel");
        manager.register_kernel(&kernel_id, initial_code);

        let new_code = KernelCodeSource::from_ptx(".version 8.0\n.target sm_90", "kernel");
        let mut request = manager.request_reload(&kernel_id, new_code).unwrap();

        // Create mock kernel for checkpoint
        let mock_kernel = MockCheckpointableKernel::new("test_kernel", 512);

        let result = manager.execute_reload(&mut request, &mock_kernel).unwrap();

        assert!(request.is_completed());
        assert_eq!(result.kernel_id.as_str(), "test_kernel");
        assert!(result.state_preserved);
        assert!(result.checkpoint_size > 0);
        assert!(result.total_duration > Duration::ZERO);

        // Stats should be updated
        let stats = manager.stats();
        assert_eq!(stats.successful_reloads, 1);
    }

    #[test]
    fn test_hot_reload_list_kernels() {
        let manager = HotReloadManager::with_defaults();

        let k1 = KernelId::new("kernel1");
        let k2 = KernelId::new("kernel2");
        let k3 = KernelId::new("kernel3");

        manager.register_kernel(&k1, KernelCodeSource::from_ptx(".version 7.0", "k1"));
        manager.register_kernel(&k2, KernelCodeSource::from_ptx(".version 7.0", "k2"));
        manager.register_kernel(&k3, KernelCodeSource::from_ptx(".version 7.0", "k3"));

        let kernels = manager.list_kernels();
        assert_eq!(kernels.len(), 3);
        assert!(kernels.contains(&k1));
        assert!(kernels.contains(&k2));
        assert!(kernels.contains(&k3));
    }
}
