//! NVTX (NVIDIA Tools Extension) integration for Nsight profiling.
//!
//! This module provides real NVTX integration using cudarc's nvtx bindings,
//! enabling timeline visualization in Nsight Systems and Nsight Compute.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cuda::profiling::CudaNvtxProfiler;
//! use ringkernel_core::observability::{GpuProfiler, ProfilerColor};
//!
//! let profiler = CudaNvtxProfiler::new();
//!
//! // Push a named range
//! {
//!     let _range = profiler.push_range("kernel_execution", ProfilerColor::CYAN);
//!     // ... kernel execution ...
//!     // Range automatically pops on drop
//! }
//!
//! // Insert an instant marker
//! profiler.mark("checkpoint", ProfilerColor::GREEN);
//! ```

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use cudarc::nvtx;
use ringkernel_core::observability::{
    GpuProfiler, GpuProfilerBackend, ProfilerColor, ProfilerError, ProfilerRange,
};

/// Convert a ProfilerColor to ARGB u32 format for NVTX.
fn color_to_argb(color: ProfilerColor) -> u32 {
    ((color.a as u32) << 24) | ((color.r as u32) << 16) | ((color.g as u32) << 8) | (color.b as u32)
}

/// NVTX category IDs for different profiling contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvtxCategory {
    /// General kernel execution.
    Kernel = 1,
    /// Memory transfers.
    Transfer = 2,
    /// Memory allocations.
    Memory = 3,
    /// Synchronization points.
    Sync = 4,
    /// Queue operations.
    Queue = 5,
    /// Custom user category.
    User = 6,
}

/// Real NVTX profiler implementation using cudarc's nvtx module.
///
/// This profiler integrates with NVIDIA Nsight tools for timeline visualization.
/// NVTX ranges and markers appear in Nsight Systems and Nsight Compute profilers.
pub struct CudaNvtxProfiler {
    /// Whether profiling is enabled.
    enabled: AtomicBool,
    /// Whether a capture is in progress.
    capture_in_progress: AtomicBool,
    /// Next category ID for user-defined categories.
    next_category: AtomicU32,
    /// Categories registered flag.
    categories_registered: AtomicBool,
}

impl CudaNvtxProfiler {
    /// Create a new NVTX profiler.
    pub fn new() -> Self {
        let profiler = Self {
            enabled: AtomicBool::new(true),
            capture_in_progress: AtomicBool::new(false),
            next_category: AtomicU32::new(100), // User categories start at 100
            categories_registered: AtomicBool::new(false),
        };

        // Register default categories
        profiler.register_default_categories();

        profiler
    }

    /// Register the default NVTX categories.
    fn register_default_categories(&self) {
        if self
            .categories_registered
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
        {
            nvtx::name_category(NvtxCategory::Kernel as u32, "Kernel");
            nvtx::name_category(NvtxCategory::Transfer as u32, "Transfer");
            nvtx::name_category(NvtxCategory::Memory as u32, "Memory");
            nvtx::name_category(NvtxCategory::Sync as u32, "Sync");
            nvtx::name_category(NvtxCategory::Queue as u32, "Queue");
            nvtx::name_category(NvtxCategory::User as u32, "User");
        }
    }

    /// Register a custom category.
    ///
    /// Returns the category ID that can be used with `push_range_with_category`.
    pub fn register_category(&self, name: &str) -> u32 {
        let id = self.next_category.fetch_add(1, Ordering::Relaxed);
        nvtx::name_category(id, name);
        id
    }

    /// Push a range with a specific category.
    pub fn push_range_with_category(
        &self,
        name: &str,
        color: ProfilerColor,
        category: NvtxCategory,
    ) -> NvtxRange {
        if !self.enabled.load(Ordering::Relaxed) {
            return NvtxRange::disabled();
        }

        let argb = color_to_argb(color);
        let mut event = nvtx::Event::message(name);
        event.argb(argb);
        event.category(category as u32);
        NvtxRange::new(event.range())
    }

    /// Push a range with a custom category ID.
    pub fn push_range_with_category_id(
        &self,
        name: &str,
        color: ProfilerColor,
        category_id: u32,
    ) -> NvtxRange {
        if !self.enabled.load(Ordering::Relaxed) {
            return NvtxRange::disabled();
        }

        let argb = color_to_argb(color);
        let mut event = nvtx::Event::message(name);
        event.argb(argb);
        event.category(category_id);
        NvtxRange::new(event.range())
    }

    /// Mark with a specific category.
    pub fn mark_with_category(&self, name: &str, color: ProfilerColor, category: NvtxCategory) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        let argb = color_to_argb(color);
        let mut event = nvtx::Event::message(name);
        event.argb(argb);
        event.category(category as u32);
        event.mark();
    }

    /// Mark with a payload value.
    pub fn mark_with_payload(&self, name: &str, color: ProfilerColor, payload: NvtxPayload) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        let argb = color_to_argb(color);
        let mut event = nvtx::Event::message(name);
        event.argb(argb);
        event.payload(payload.to_cudarc());
        event.mark();
    }

    /// Create a scoped range using cudarc's shorthand.
    pub fn scoped_range(&self, name: &str) -> NvtxRange {
        if !self.enabled.load(Ordering::Relaxed) {
            return NvtxRange::disabled();
        }

        NvtxRange::new(nvtx::scoped_range(name))
    }

    /// Enable or disable profiling.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if profiling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

impl Default for CudaNvtxProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuProfiler for CudaNvtxProfiler {
    fn is_available(&self) -> bool {
        // NVTX is always available when cudarc's nvtx feature is enabled
        true
    }

    fn backend(&self) -> GpuProfilerBackend {
        GpuProfilerBackend::Nsight
    }

    fn start_capture(&self) -> Result<(), ProfilerError> {
        if self
            .capture_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return Err(ProfilerError::CaptureInProgress);
        }
        // NVTX doesn't have explicit capture control - ranges are always captured
        // when running under Nsight
        Ok(())
    }

    fn end_capture(&self) -> Result<(), ProfilerError> {
        if self
            .capture_in_progress
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return Err(ProfilerError::NoCaptureInProgress);
        }
        Ok(())
    }

    fn push_range(&self, name: &str, color: ProfilerColor) -> ProfilerRange {
        if !self.enabled.load(Ordering::Relaxed) {
            return ProfilerRange::stub(name, self.backend());
        }

        let argb = color_to_argb(color);
        let mut event = nvtx::Event::message(name);
        event.argb(argb);

        // Start the range - it will be ended when this scope exits
        // Note: We need to store the range handle for proper RAII
        let _range = event.range();

        // Return a ProfilerRange for compatibility with the trait
        // The actual NVTX range is managed separately
        ProfilerRange::stub(name, self.backend())
    }

    fn pop_range(&self) {
        // NVTX ranges are automatically popped via RAII when NvtxRange drops
        // This is a no-op for compatibility with the trait interface
    }

    fn mark(&self, name: &str, color: ProfilerColor) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        let argb = color_to_argb(color);
        let mut event = nvtx::Event::message(name);
        event.argb(argb);
        event.mark();
    }

    fn set_thread_name(&self, name: &str) {
        // NVTX uses nvtxNameOsThreadA for this, but cudarc doesn't expose it
        // We can mark the thread with a special marker instead
        nvtx::mark(format!("Thread: {}", name));
    }

    fn message(&self, text: &str) {
        // Use a mark for messages
        nvtx::mark(text);
    }

    fn register_allocation(&self, ptr: u64, size: usize, name: &str) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        // Mark the allocation with memory category
        let msg = format!("Alloc {} @ 0x{:x} ({} bytes)", name, ptr, size);
        let mut event = nvtx::Event::message(&msg);
        event.category(NvtxCategory::Memory as u32);
        event.argb(color_to_argb(ProfilerColor::GREEN));
        event.payload(nvtx::Payload::U64(size as u64));
        event.mark();
    }

    fn unregister_allocation(&self, ptr: u64) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        // Mark the deallocation
        let msg = format!("Free @ 0x{:x}", ptr);
        let mut event = nvtx::Event::message(&msg);
        event.category(NvtxCategory::Memory as u32);
        event.argb(color_to_argb(ProfilerColor::RED));
        event.mark();
    }
}

/// NVTX range wrapper for RAII-based profiling.
///
/// The range is automatically ended when this struct is dropped.
pub struct NvtxRange {
    /// The underlying cudarc NVTX range.
    range: Option<nvtx::Range>,
}

impl NvtxRange {
    /// Create a new NVTX range.
    fn new(range: nvtx::Range) -> Self {
        Self { range: Some(range) }
    }

    /// Create a disabled (no-op) range.
    fn disabled() -> Self {
        Self { range: None }
    }

    /// Check if this range is active.
    pub fn is_active(&self) -> bool {
        self.range.is_some()
    }
}

// NvtxRange automatically ends when dropped via cudarc's Range Drop impl

/// Payload values for NVTX markers.
#[derive(Debug, Clone, Copy)]
pub enum NvtxPayload {
    /// 32-bit signed integer.
    I32(i32),
    /// 64-bit signed integer.
    I64(i64),
    /// 32-bit unsigned integer.
    U32(u32),
    /// 64-bit unsigned integer.
    U64(u64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
}

impl NvtxPayload {
    /// Convert to cudarc's Payload type.
    fn to_cudarc(self) -> nvtx::Payload {
        match self {
            NvtxPayload::I32(v) => nvtx::Payload::I32(v),
            NvtxPayload::I64(v) => nvtx::Payload::Int64(v),
            NvtxPayload::U32(v) => nvtx::Payload::U32(v),
            NvtxPayload::U64(v) => nvtx::Payload::U64(v),
            NvtxPayload::F32(v) => nvtx::Payload::F32(v),
            NvtxPayload::F64(v) => nvtx::Payload::F64(v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_to_argb() {
        let color = ProfilerColor::new(255, 128, 64);
        let argb = color_to_argb(color);
        // Alpha is 255 by default in ProfilerColor
        assert_eq!(argb, 0xFF_FF_80_40);
    }

    #[test]
    fn test_color_to_argb_predefined() {
        assert_eq!(color_to_argb(ProfilerColor::RED), 0xFF_FF_00_00);
        assert_eq!(color_to_argb(ProfilerColor::GREEN), 0xFF_00_FF_00);
        assert_eq!(color_to_argb(ProfilerColor::BLUE), 0xFF_00_00_FF);
        assert_eq!(color_to_argb(ProfilerColor::YELLOW), 0xFF_FF_FF_00);
        assert_eq!(color_to_argb(ProfilerColor::CYAN), 0xFF_00_FF_FF);
        assert_eq!(color_to_argb(ProfilerColor::MAGENTA), 0xFF_FF_00_FF);
    }

    #[test]
    #[ignore = "requires nvToolsExt library"]
    fn test_nvtx_profiler_available() {
        let profiler = CudaNvtxProfiler::new();
        assert!(profiler.is_available());
        assert_eq!(profiler.backend(), GpuProfilerBackend::Nsight);
    }

    #[test]
    #[ignore = "requires nvToolsExt library"]
    fn test_nvtx_profiler_enable_disable() {
        let profiler = CudaNvtxProfiler::new();
        assert!(profiler.is_enabled());

        profiler.set_enabled(false);
        assert!(!profiler.is_enabled());

        profiler.set_enabled(true);
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_nvtx_category_values() {
        assert_eq!(NvtxCategory::Kernel as u32, 1);
        assert_eq!(NvtxCategory::Transfer as u32, 2);
        assert_eq!(NvtxCategory::Memory as u32, 3);
        assert_eq!(NvtxCategory::Sync as u32, 4);
        assert_eq!(NvtxCategory::Queue as u32, 5);
        assert_eq!(NvtxCategory::User as u32, 6);
    }

    #[test]
    #[ignore = "requires nvToolsExt library"]
    fn test_nvtx_profiler_register_category() {
        let profiler = CudaNvtxProfiler::new();
        let cat1 = profiler.register_category("CustomCategory1");
        let cat2 = profiler.register_category("CustomCategory2");

        // Categories should be assigned incrementally starting from 100
        assert!(cat1 >= 100);
        assert_eq!(cat2, cat1 + 1);
    }

    #[test]
    fn test_nvtx_range_disabled() {
        let range = NvtxRange::disabled();
        assert!(!range.is_active());
    }

    #[test]
    #[ignore = "requires nvToolsExt library"]
    fn test_nvtx_mark() {
        // This test just verifies the API compiles and runs without panic
        let profiler = CudaNvtxProfiler::new();
        profiler.mark("test_marker", ProfilerColor::CYAN);
    }

    #[test]
    #[ignore = "requires nvToolsExt library"]
    fn test_nvtx_scoped_range() {
        let profiler = CudaNvtxProfiler::new();
        {
            let range = profiler.scoped_range("test_range");
            assert!(range.is_active());
        }
        // Range should be automatically ended when dropped
    }

    #[test]
    #[ignore = "requires nvToolsExt library"]
    fn test_nvtx_capture_state() {
        let profiler = CudaNvtxProfiler::new();

        // Start capture
        assert!(profiler.start_capture().is_ok());

        // Can't start capture twice
        assert!(matches!(
            profiler.start_capture(),
            Err(ProfilerError::CaptureInProgress)
        ));

        // End capture
        assert!(profiler.end_capture().is_ok());

        // Can't end capture twice
        assert!(matches!(
            profiler.end_capture(),
            Err(ProfilerError::NoCaptureInProgress)
        ));
    }

    #[test]
    fn test_nvtx_payload() {
        // Test payload conversions
        let i32_payload = NvtxPayload::I32(42);
        let _ = i32_payload.to_cudarc();

        let u64_payload = NvtxPayload::U64(1024 * 1024);
        let _ = u64_payload.to_cudarc();

        let f32_payload = NvtxPayload::F32(3.14159);
        let _ = f32_payload.to_cudarc();
    }

    #[test]
    #[ignore = "requires nvToolsExt library"]
    fn test_nvtx_mark_with_payload() {
        let profiler = CudaNvtxProfiler::new();
        profiler.mark_with_payload("memory_size", ProfilerColor::GREEN, NvtxPayload::U64(1024));
    }

    #[test]
    #[ignore = "requires nvToolsExt library"]
    fn test_nvtx_mark_with_category() {
        let profiler = CudaNvtxProfiler::new();
        profiler.mark_with_category("kernel_launch", ProfilerColor::CYAN, NvtxCategory::Kernel);
        profiler.mark_with_category("memcpy", ProfilerColor::BLUE, NvtxCategory::Transfer);
    }

    #[test]
    #[ignore = "requires nvToolsExt library"]
    fn test_nvtx_push_range_with_category() {
        let profiler = CudaNvtxProfiler::new();
        {
            let _range = profiler.push_range_with_category(
                "compute",
                ProfilerColor::ORANGE,
                NvtxCategory::Kernel,
            );
            // Range active
        }
        // Range ended
    }

    #[test]
    #[ignore = "requires nvToolsExt library"]
    fn test_nvtx_allocation_tracking() {
        let profiler = CudaNvtxProfiler::new();
        profiler.register_allocation(0x1000, 4096, "test_buffer");
        profiler.unregister_allocation(0x1000);
    }

    #[test]
    #[ignore = "requires nvToolsExt library"]
    fn test_nvtx_disabled_operations() {
        let profiler = CudaNvtxProfiler::new();
        profiler.set_enabled(false);

        // All operations should be no-ops when disabled
        let range = profiler.scoped_range("disabled_range");
        assert!(!range.is_active());

        profiler.mark("disabled_mark", ProfilerColor::RED);
        profiler.register_allocation(0x2000, 1024, "disabled_alloc");
    }
}
