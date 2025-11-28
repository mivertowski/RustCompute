//! Multi-GPU coordination and load balancing.
//!
//! This module provides infrastructure for coordinating work across
//! multiple GPUs, including device selection, load balancing, and
//! cross-device communication.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::error::{Result, RingKernelError};
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

    /// Unregister a device.
    pub fn unregister_device(&self, index: usize) {
        let devices = self.devices.read();
        if index < devices.len() {
            // Move kernels to another device (TODO: implement migration)
        }
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
}
