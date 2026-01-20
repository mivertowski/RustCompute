//! GPU memory tracking with allocation profiling and leak detection.
//!
//! This module provides comprehensive memory tracking for CUDA allocations,
//! including timing information, leak detection, and integration with NVTX
//! for visualization in Nsight tools.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cuda::profiling::{CudaMemoryTracker, CudaMemoryKind};
//!
//! let tracker = CudaMemoryTracker::new();
//!
//! // Track an allocation
//! let id = tracker.track_alloc(0x1000, 4096, CudaMemoryKind::Device, "vertex_buffer");
//!
//! // Get statistics
//! println!("Current usage: {} bytes", tracker.current_usage());
//! println!("Peak usage: {} bytes", tracker.peak_usage());
//!
//! // Track deallocation
//! tracker.track_free(id);
//!
//! // Check for leaks
//! if let Some(leaks) = tracker.detect_leaks() {
//!     for leak in leaks {
//!         println!("Leak: {} at 0x{:x}", leak.label, leak.ptr);
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;

use ringkernel_core::observability::{GpuMemoryDashboard, GpuMemoryType, ProfilerColor};

use super::nvtx::{CudaNvtxProfiler, NvtxCategory};

/// Kind of CUDA memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CudaMemoryKind {
    /// Device memory (cuMemAlloc).
    Device,
    /// Pinned host memory (cuMemHostAlloc).
    Pinned,
    /// Mapped memory (CU_MEMHOSTALLOC_DEVICEMAP).
    Mapped,
    /// Managed memory (cuMemAllocManaged).
    Managed,
    /// Array memory (cuArrayCreate).
    Array,
    /// Texture memory.
    Texture,
}

impl CudaMemoryKind {
    /// Get a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            CudaMemoryKind::Device => "Device",
            CudaMemoryKind::Pinned => "Pinned",
            CudaMemoryKind::Mapped => "Mapped",
            CudaMemoryKind::Managed => "Managed",
            CudaMemoryKind::Array => "Array",
            CudaMemoryKind::Texture => "Texture",
        }
    }

    /// Get color for NVTX markers.
    pub fn color(&self) -> ProfilerColor {
        match self {
            CudaMemoryKind::Device => ProfilerColor::CYAN,
            CudaMemoryKind::Pinned => ProfilerColor::GREEN,
            CudaMemoryKind::Mapped => ProfilerColor::YELLOW,
            CudaMemoryKind::Managed => ProfilerColor::ORANGE,
            CudaMemoryKind::Array => ProfilerColor::MAGENTA,
            CudaMemoryKind::Texture => ProfilerColor::BLUE,
        }
    }

    /// Convert to GpuMemoryType for dashboard integration.
    pub fn to_gpu_memory_type(&self) -> GpuMemoryType {
        match self {
            CudaMemoryKind::Device => GpuMemoryType::DeviceLocal,
            CudaMemoryKind::Pinned => GpuMemoryType::HostCoherent,
            CudaMemoryKind::Mapped => GpuMemoryType::Mapped,
            CudaMemoryKind::Managed => GpuMemoryType::HostVisible,
            CudaMemoryKind::Array => GpuMemoryType::DeviceLocal,
            CudaMemoryKind::Texture => GpuMemoryType::DeviceLocal,
        }
    }
}

/// A tracked memory allocation.
#[derive(Debug, Clone)]
pub struct TrackedAllocation {
    /// Unique allocation ID.
    pub id: u64,
    /// Device pointer address.
    pub ptr: u64,
    /// Size in bytes.
    pub size: usize,
    /// Type of memory.
    pub kind: CudaMemoryKind,
    /// User-provided label.
    pub label: String,
    /// Time when allocation was made.
    pub allocated_at: Instant,
    /// GPU-measured allocation time (if available).
    pub allocation_time_us: Option<u64>,
    /// Stack trace hint (file:line if available).
    pub source_location: Option<String>,
    /// Device index.
    pub device_index: u32,
}

impl TrackedAllocation {
    /// Get the age of this allocation in seconds.
    pub fn age_secs(&self) -> f64 {
        self.allocated_at.elapsed().as_secs_f64()
    }
}

/// Statistics for memory allocations by kind.
#[derive(Debug, Clone, Default)]
pub struct MemoryKindStats {
    /// Number of allocations.
    pub count: usize,
    /// Total bytes allocated.
    pub total_bytes: u64,
    /// Average allocation size.
    pub avg_size: u64,
    /// Largest allocation.
    pub max_size: u64,
    /// Smallest allocation.
    pub min_size: u64,
}

/// Overall memory tracking statistics.
#[derive(Debug, Clone, Default)]
pub struct MemoryTrackerStats {
    /// Current total allocated bytes.
    pub current_bytes: u64,
    /// Peak allocated bytes.
    pub peak_bytes: u64,
    /// Total number of allocations (lifetime).
    pub total_allocations: u64,
    /// Total number of deallocations (lifetime).
    pub total_deallocations: u64,
    /// Current number of active allocations.
    pub active_allocations: usize,
    /// Statistics per memory kind.
    pub by_kind: HashMap<CudaMemoryKind, MemoryKindStats>,
    /// Total allocation time (if tracked).
    pub total_alloc_time_us: u64,
    /// Total deallocation time (if tracked).
    pub total_free_time_us: u64,
}

/// CUDA memory tracker with allocation profiling.
pub struct CudaMemoryTracker {
    /// Integration with ringkernel-core memory dashboard.
    dashboard: Arc<GpuMemoryDashboard>,
    /// Optional NVTX profiler for markers.
    nvtx: Option<Arc<CudaNvtxProfiler>>,
    /// Active allocations indexed by ID.
    allocations: RwLock<HashMap<u64, TrackedAllocation>>,
    /// Pointer to ID mapping for fast lookup.
    ptr_to_id: RwLock<HashMap<u64, u64>>,
    /// Current total allocated bytes.
    current_bytes: AtomicU64,
    /// Peak allocated bytes.
    peak_bytes: AtomicU64,
    /// Next allocation ID.
    next_id: AtomicU64,
    /// Total allocations made.
    total_allocations: AtomicU64,
    /// Total deallocations made.
    total_deallocations: AtomicU64,
    /// Total allocation time in microseconds.
    total_alloc_time_us: AtomicU64,
    /// Total deallocation time in microseconds.
    total_free_time_us: AtomicU64,
    /// Whether tracking is enabled.
    enabled: std::sync::atomic::AtomicBool,
}

impl CudaMemoryTracker {
    /// Create a new memory tracker.
    pub fn new() -> Self {
        Self {
            dashboard: GpuMemoryDashboard::new(),
            nvtx: None,
            allocations: RwLock::new(HashMap::new()),
            ptr_to_id: RwLock::new(HashMap::new()),
            current_bytes: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
            next_id: AtomicU64::new(1),
            total_allocations: AtomicU64::new(0),
            total_deallocations: AtomicU64::new(0),
            total_alloc_time_us: AtomicU64::new(0),
            total_free_time_us: AtomicU64::new(0),
            enabled: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Create a new memory tracker with NVTX integration.
    pub fn with_nvtx(nvtx: Arc<CudaNvtxProfiler>) -> Self {
        let mut tracker = Self::new();
        tracker.nvtx = Some(nvtx);
        tracker
    }

    /// Create a new memory tracker with a shared dashboard.
    pub fn with_dashboard(dashboard: Arc<GpuMemoryDashboard>) -> Self {
        let mut tracker = Self::new();
        tracker.dashboard = dashboard;
        tracker
    }

    /// Enable or disable tracking.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if tracking is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Track a new allocation.
    ///
    /// Returns the allocation ID for later deallocation tracking.
    pub fn track_alloc(&self, ptr: u64, size: usize, kind: CudaMemoryKind, label: &str) -> u64 {
        self.track_alloc_with_time(ptr, size, kind, label, None, None)
    }

    /// Track a new allocation with timing information.
    pub fn track_alloc_with_time(
        &self,
        ptr: u64,
        size: usize,
        kind: CudaMemoryKind,
        label: &str,
        allocation_time_us: Option<u64>,
        device_index: Option<u32>,
    ) -> u64 {
        if !self.is_enabled() {
            return 0;
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let allocation = TrackedAllocation {
            id,
            ptr,
            size,
            kind,
            label: label.to_string(),
            allocated_at: Instant::now(),
            allocation_time_us,
            source_location: None,
            device_index: device_index.unwrap_or(0),
        };

        // Update statistics
        let new_bytes = self.current_bytes.fetch_add(size as u64, Ordering::Relaxed) + size as u64;
        self.peak_bytes.fetch_max(new_bytes, Ordering::Relaxed);
        self.total_allocations.fetch_add(1, Ordering::Relaxed);

        if let Some(time_us) = allocation_time_us {
            self.total_alloc_time_us
                .fetch_add(time_us, Ordering::Relaxed);
        }

        // Store allocation
        {
            let mut allocations = self.allocations.write();
            allocations.insert(id, allocation);
        }
        {
            let mut ptr_to_id = self.ptr_to_id.write();
            ptr_to_id.insert(ptr, id);
        }

        // Update dashboard
        self.dashboard.track_allocation(
            id,
            label,
            size,
            kind.to_gpu_memory_type(),
            device_index.unwrap_or(0),
            None,
        );

        // NVTX marker
        if let Some(nvtx) = &self.nvtx {
            nvtx.mark_with_category(
                &format!("Alloc {} ({} bytes)", label, size),
                kind.color(),
                NvtxCategory::Memory,
            );
        }

        id
    }

    /// Track deallocation by allocation ID.
    pub fn track_free(&self, id: u64) -> Option<TrackedAllocation> {
        self.track_free_with_time(id, None)
    }

    /// Track deallocation by pointer address.
    pub fn track_free_by_ptr(&self, ptr: u64) -> Option<TrackedAllocation> {
        let id = {
            let ptr_to_id = self.ptr_to_id.read();
            ptr_to_id.get(&ptr).copied()
        };

        id.and_then(|id| self.track_free(id))
    }

    /// Track deallocation with timing information.
    pub fn track_free_with_time(
        &self,
        id: u64,
        free_time_us: Option<u64>,
    ) -> Option<TrackedAllocation> {
        if !self.is_enabled() || id == 0 {
            return None;
        }

        let allocation = {
            let mut allocations = self.allocations.write();
            allocations.remove(&id)
        };

        if let Some(ref alloc) = allocation {
            // Remove from ptr mapping
            {
                let mut ptr_to_id = self.ptr_to_id.write();
                ptr_to_id.remove(&alloc.ptr);
            }

            // Update statistics
            self.current_bytes
                .fetch_sub(alloc.size as u64, Ordering::Relaxed);
            self.total_deallocations.fetch_add(1, Ordering::Relaxed);

            if let Some(time_us) = free_time_us {
                self.total_free_time_us
                    .fetch_add(time_us, Ordering::Relaxed);
            }

            // Update dashboard
            self.dashboard.track_deallocation(alloc.id);

            // NVTX marker
            if let Some(nvtx) = &self.nvtx {
                nvtx.mark_with_category(
                    &format!("Free {} ({} bytes)", alloc.label, alloc.size),
                    ProfilerColor::RED,
                    NvtxCategory::Memory,
                );
            }
        }

        allocation
    }

    /// Get current allocated bytes.
    pub fn current_usage(&self) -> u64 {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Get peak allocated bytes.
    pub fn peak_usage(&self) -> u64 {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Get number of active allocations.
    pub fn allocation_count(&self) -> usize {
        self.allocations.read().len()
    }

    /// Get total allocations made.
    pub fn total_allocations(&self) -> u64 {
        self.total_allocations.load(Ordering::Relaxed)
    }

    /// Get total deallocations made.
    pub fn total_deallocations(&self) -> u64 {
        self.total_deallocations.load(Ordering::Relaxed)
    }

    /// Get comprehensive statistics.
    pub fn stats(&self) -> MemoryTrackerStats {
        let allocations = self.allocations.read();

        let mut by_kind: HashMap<CudaMemoryKind, MemoryKindStats> = HashMap::new();

        for alloc in allocations.values() {
            let entry = by_kind
                .entry(alloc.kind)
                .or_insert_with(|| MemoryKindStats {
                    min_size: u64::MAX,
                    ..Default::default()
                });

            entry.count += 1;
            entry.total_bytes += alloc.size as u64;
            entry.max_size = entry.max_size.max(alloc.size as u64);
            entry.min_size = entry.min_size.min(alloc.size as u64);
        }

        // Calculate averages
        for entry in by_kind.values_mut() {
            if entry.count > 0 {
                entry.avg_size = entry.total_bytes / entry.count as u64;
            }
            if entry.min_size == u64::MAX {
                entry.min_size = 0;
            }
        }

        MemoryTrackerStats {
            current_bytes: self.current_bytes.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_deallocations: self.total_deallocations.load(Ordering::Relaxed),
            active_allocations: allocations.len(),
            by_kind,
            total_alloc_time_us: self.total_alloc_time_us.load(Ordering::Relaxed),
            total_free_time_us: self.total_free_time_us.load(Ordering::Relaxed),
        }
    }

    /// Detect potential memory leaks (allocations older than threshold).
    ///
    /// Returns allocations that have been alive longer than `min_age_secs`.
    pub fn detect_leaks(&self, min_age_secs: f64) -> Option<Vec<TrackedAllocation>> {
        let allocations = self.allocations.read();

        let leaks: Vec<_> = allocations
            .values()
            .filter(|a| a.age_secs() > min_age_secs)
            .cloned()
            .collect();

        if leaks.is_empty() {
            None
        } else {
            Some(leaks)
        }
    }

    /// Get all active allocations.
    pub fn active_allocations(&self) -> Vec<TrackedAllocation> {
        self.allocations.read().values().cloned().collect()
    }

    /// Get allocations by kind.
    pub fn allocations_by_kind(&self, kind: CudaMemoryKind) -> Vec<TrackedAllocation> {
        self.allocations
            .read()
            .values()
            .filter(|a| a.kind == kind)
            .cloned()
            .collect()
    }

    /// Get the largest allocations (top N by size).
    pub fn largest_allocations(&self, n: usize) -> Vec<TrackedAllocation> {
        let mut allocs: Vec<_> = self.allocations.read().values().cloned().collect();
        allocs.sort_by(|a, b| b.size.cmp(&a.size));
        allocs.truncate(n);
        allocs
    }

    /// Get the oldest allocations (top N by age).
    pub fn oldest_allocations(&self, n: usize) -> Vec<TrackedAllocation> {
        let mut allocs: Vec<_> = self.allocations.read().values().cloned().collect();
        allocs.sort_by(|a, b| b.allocated_at.cmp(&a.allocated_at));
        allocs.truncate(n);
        allocs
    }

    /// Clear all tracking data.
    pub fn clear(&self) {
        self.allocations.write().clear();
        self.ptr_to_id.write().clear();
        self.current_bytes.store(0, Ordering::Relaxed);
        self.peak_bytes.store(0, Ordering::Relaxed);
        self.total_allocations.store(0, Ordering::Relaxed);
        self.total_deallocations.store(0, Ordering::Relaxed);
        self.total_alloc_time_us.store(0, Ordering::Relaxed);
        self.total_free_time_us.store(0, Ordering::Relaxed);
    }

    /// Reset peak usage to current usage.
    pub fn reset_peak(&self) {
        let current = self.current_bytes.load(Ordering::Relaxed);
        self.peak_bytes.store(current, Ordering::Relaxed);
    }

    /// Get the underlying memory dashboard.
    pub fn dashboard(&self) -> &GpuMemoryDashboard {
        &self.dashboard
    }
}

impl Default for CudaMemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

// CudaMemoryTracker is Send+Sync due to internal synchronization
unsafe impl Send for CudaMemoryTracker {}
unsafe impl Sync for CudaMemoryTracker {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_kind_labels() {
        assert_eq!(CudaMemoryKind::Device.label(), "Device");
        assert_eq!(CudaMemoryKind::Pinned.label(), "Pinned");
        assert_eq!(CudaMemoryKind::Mapped.label(), "Mapped");
        assert_eq!(CudaMemoryKind::Managed.label(), "Managed");
        assert_eq!(CudaMemoryKind::Array.label(), "Array");
        assert_eq!(CudaMemoryKind::Texture.label(), "Texture");
    }

    #[test]
    fn test_tracker_basic() {
        let tracker = CudaMemoryTracker::new();
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.peak_usage(), 0);
        assert_eq!(tracker.allocation_count(), 0);
    }

    #[test]
    fn test_track_alloc() {
        let tracker = CudaMemoryTracker::new();

        let id = tracker.track_alloc(0x1000, 4096, CudaMemoryKind::Device, "buffer1");
        assert!(id > 0);
        assert_eq!(tracker.current_usage(), 4096);
        assert_eq!(tracker.peak_usage(), 4096);
        assert_eq!(tracker.allocation_count(), 1);

        let id2 = tracker.track_alloc(0x2000, 8192, CudaMemoryKind::Pinned, "buffer2");
        assert!(id2 > id);
        assert_eq!(tracker.current_usage(), 4096 + 8192);
        assert_eq!(tracker.peak_usage(), 4096 + 8192);
        assert_eq!(tracker.allocation_count(), 2);
    }

    #[test]
    fn test_track_free() {
        let tracker = CudaMemoryTracker::new();

        let id1 = tracker.track_alloc(0x1000, 4096, CudaMemoryKind::Device, "buffer1");
        let id2 = tracker.track_alloc(0x2000, 8192, CudaMemoryKind::Device, "buffer2");

        assert_eq!(tracker.current_usage(), 4096 + 8192);

        let freed = tracker.track_free(id1);
        assert!(freed.is_some());
        assert_eq!(freed.unwrap().label, "buffer1");
        assert_eq!(tracker.current_usage(), 8192);
        assert_eq!(tracker.peak_usage(), 4096 + 8192); // Peak unchanged

        let freed2 = tracker.track_free(id2);
        assert!(freed2.is_some());
        assert_eq!(tracker.current_usage(), 0);
    }

    #[test]
    fn test_track_free_by_ptr() {
        let tracker = CudaMemoryTracker::new();

        tracker.track_alloc(0x1000, 4096, CudaMemoryKind::Device, "buffer1");
        assert_eq!(tracker.current_usage(), 4096);

        let freed = tracker.track_free_by_ptr(0x1000);
        assert!(freed.is_some());
        assert_eq!(tracker.current_usage(), 0);

        // Free again should return None
        let freed_again = tracker.track_free_by_ptr(0x1000);
        assert!(freed_again.is_none());
    }

    #[test]
    fn test_stats() {
        let tracker = CudaMemoryTracker::new();

        tracker.track_alloc(0x1000, 1024, CudaMemoryKind::Device, "d1");
        tracker.track_alloc(0x2000, 2048, CudaMemoryKind::Device, "d2");
        tracker.track_alloc(0x3000, 4096, CudaMemoryKind::Pinned, "p1");

        let stats = tracker.stats();
        assert_eq!(stats.current_bytes, 1024 + 2048 + 4096);
        assert_eq!(stats.active_allocations, 3);
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.total_deallocations, 0);

        let device_stats = stats.by_kind.get(&CudaMemoryKind::Device).unwrap();
        assert_eq!(device_stats.count, 2);
        assert_eq!(device_stats.total_bytes, 1024 + 2048);

        let pinned_stats = stats.by_kind.get(&CudaMemoryKind::Pinned).unwrap();
        assert_eq!(pinned_stats.count, 1);
        assert_eq!(pinned_stats.total_bytes, 4096);
    }

    #[test]
    fn test_largest_allocations() {
        let tracker = CudaMemoryTracker::new();

        tracker.track_alloc(0x1000, 1000, CudaMemoryKind::Device, "small");
        tracker.track_alloc(0x2000, 5000, CudaMemoryKind::Device, "medium");
        tracker.track_alloc(0x3000, 10000, CudaMemoryKind::Device, "large");

        let largest = tracker.largest_allocations(2);
        assert_eq!(largest.len(), 2);
        assert_eq!(largest[0].label, "large");
        assert_eq!(largest[1].label, "medium");
    }

    #[test]
    fn test_allocations_by_kind() {
        let tracker = CudaMemoryTracker::new();

        tracker.track_alloc(0x1000, 1024, CudaMemoryKind::Device, "d1");
        tracker.track_alloc(0x2000, 2048, CudaMemoryKind::Pinned, "p1");
        tracker.track_alloc(0x3000, 4096, CudaMemoryKind::Device, "d2");

        let device_allocs = tracker.allocations_by_kind(CudaMemoryKind::Device);
        assert_eq!(device_allocs.len(), 2);

        let pinned_allocs = tracker.allocations_by_kind(CudaMemoryKind::Pinned);
        assert_eq!(pinned_allocs.len(), 1);

        let managed_allocs = tracker.allocations_by_kind(CudaMemoryKind::Managed);
        assert_eq!(managed_allocs.len(), 0);
    }

    #[test]
    fn test_enable_disable() {
        let tracker = CudaMemoryTracker::new();
        assert!(tracker.is_enabled());

        tracker.set_enabled(false);
        assert!(!tracker.is_enabled());

        // Allocations when disabled should return 0 and not track
        let id = tracker.track_alloc(0x1000, 4096, CudaMemoryKind::Device, "test");
        assert_eq!(id, 0);
        assert_eq!(tracker.allocation_count(), 0);

        tracker.set_enabled(true);
        let id = tracker.track_alloc(0x1000, 4096, CudaMemoryKind::Device, "test");
        assert!(id > 0);
        assert_eq!(tracker.allocation_count(), 1);
    }

    #[test]
    fn test_clear() {
        let tracker = CudaMemoryTracker::new();

        tracker.track_alloc(0x1000, 4096, CudaMemoryKind::Device, "test");
        tracker.track_alloc(0x2000, 8192, CudaMemoryKind::Device, "test2");

        assert_eq!(tracker.allocation_count(), 2);
        assert!(tracker.current_usage() > 0);

        tracker.clear();

        assert_eq!(tracker.allocation_count(), 0);
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.peak_usage(), 0);
        assert_eq!(tracker.total_allocations(), 0);
    }

    #[test]
    fn test_reset_peak() {
        let tracker = CudaMemoryTracker::new();

        let id1 = tracker.track_alloc(0x1000, 10000, CudaMemoryKind::Device, "big");
        tracker.track_free(id1);

        let _id2 = tracker.track_alloc(0x2000, 1000, CudaMemoryKind::Device, "small");

        assert_eq!(tracker.peak_usage(), 10000);
        assert_eq!(tracker.current_usage(), 1000);

        tracker.reset_peak();

        assert_eq!(tracker.peak_usage(), 1000);
        assert_eq!(tracker.current_usage(), 1000);
    }

    #[test]
    fn test_alloc_with_time() {
        let tracker = CudaMemoryTracker::new();

        let id = tracker.track_alloc_with_time(
            0x1000,
            4096,
            CudaMemoryKind::Device,
            "timed_buffer",
            Some(150), // 150 microseconds
            Some(0),   // Device 0
        );

        let stats = tracker.stats();
        assert_eq!(stats.total_alloc_time_us, 150);

        tracker.track_free_with_time(id, Some(50));
        let stats = tracker.stats();
        assert_eq!(stats.total_free_time_us, 50);
    }

    #[test]
    fn test_tracked_allocation_age() {
        let tracker = CudaMemoryTracker::new();

        tracker.track_alloc(0x1000, 1024, CudaMemoryKind::Device, "test");

        let allocs = tracker.active_allocations();
        assert_eq!(allocs.len(), 1);

        // Age should be very small (just created)
        assert!(allocs[0].age_secs() < 1.0);
    }
}
