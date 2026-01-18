//! Analytics context for grouped buffer lifecycle management.
//!
//! This module provides the `AnalyticsContext` type for managing buffer
//! allocations during analytics operations like DFG mining, BFS traversal,
//! and pattern detection.
//!
//! # Purpose
//!
//! Analytics operations often need multiple temporary buffers that are
//! used together and released together. The `AnalyticsContext` provides:
//!
//! - **Grouped lifecycle**: All buffers are released when the context drops
//! - **Statistics tracking**: Peak memory usage, allocation count
//! - **Named contexts**: For debugging and profiling
//!
//! # Example
//!
//! ```
//! use ringkernel_core::analytics_context::AnalyticsContext;
//!
//! // Create context for a BFS operation
//! let mut ctx = AnalyticsContext::new("bfs_traversal");
//!
//! // Allocate buffers for frontier, visited, and distances
//! let frontier_idx = ctx.allocate(1024);
//! let visited_idx = ctx.allocate(1024);
//! let distances_idx = ctx.allocate_typed::<u32>(256);
//!
//! // Use the buffers
//! ctx.get_mut(frontier_idx)[0] = 1;
//!
//! // Check stats
//! let stats = ctx.stats();
//! println!("Peak memory: {} bytes", stats.peak_bytes);
//!
//! // All buffers released when ctx drops
//! ```

use std::any::TypeId;
use std::collections::HashMap;

/// Statistics for an analytics context.
#[derive(Debug, Clone, Default)]
pub struct ContextStats {
    /// Total number of allocations made.
    pub allocations: usize,
    /// Peak memory usage in bytes.
    pub peak_bytes: usize,
    /// Current memory usage in bytes.
    pub current_bytes: usize,
    /// Number of typed allocations (via allocate_typed).
    pub typed_allocations: usize,
    /// Allocation counts by type (for typed allocations).
    pub allocations_by_type: HashMap<TypeId, usize>,
}

impl ContextStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get memory efficiency (1.0 = all allocated memory still in use).
    pub fn memory_efficiency(&self) -> f64 {
        if self.peak_bytes > 0 {
            self.current_bytes as f64 / self.peak_bytes as f64
        } else {
            1.0
        }
    }
}

/// Handle to an allocation within an AnalyticsContext.
///
/// This is an opaque index type for type safety.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocationHandle(usize);

impl AllocationHandle {
    /// Get the raw index (for advanced use).
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Context for analytics operations with grouped buffer lifecycle.
///
/// All buffers allocated through this context are released together
/// when the context is dropped. This is useful for analytics operations
/// that need multiple temporary buffers.
///
/// # Thread Safety
///
/// `AnalyticsContext` is `Send` but not `Sync`. Each context should be
/// used from a single thread. For parallel analytics, create a separate
/// context per thread.
pub struct AnalyticsContext {
    /// Context name for debugging.
    name: String,
    /// Allocated buffers.
    allocations: Vec<Box<[u8]>>,
    /// Allocation sizes (for potential future deallocation).
    sizes: Vec<usize>,
    /// Statistics.
    stats: ContextStats,
}

impl AnalyticsContext {
    /// Create a new analytics context.
    ///
    /// # Arguments
    ///
    /// * `name` - Descriptive name for debugging and profiling
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            allocations: Vec::new(),
            sizes: Vec::new(),
            stats: ContextStats::new(),
        }
    }

    /// Create a context with pre-allocated capacity.
    ///
    /// Reserves space for the expected number of allocations to avoid
    /// reallocations of the internal vectors.
    pub fn with_capacity(name: impl Into<String>, expected_allocations: usize) -> Self {
        Self {
            name: name.into(),
            allocations: Vec::with_capacity(expected_allocations),
            sizes: Vec::with_capacity(expected_allocations),
            stats: ContextStats::new(),
        }
    }

    /// Get the context name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Allocate a buffer of the specified size.
    ///
    /// Returns a handle that can be used with `get()` and `get_mut()`.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes
    ///
    /// # Returns
    ///
    /// Handle to the allocated buffer
    pub fn allocate(&mut self, size: usize) -> AllocationHandle {
        let buf = vec![0u8; size].into_boxed_slice();
        let handle = AllocationHandle(self.allocations.len());

        self.allocations.push(buf);
        self.sizes.push(size);

        self.stats.allocations += 1;
        self.stats.current_bytes += size;
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.current_bytes);

        handle
    }

    /// Allocate a typed buffer.
    ///
    /// Allocates space for `count` elements of type `T`, zero-initialized.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of elements
    ///
    /// # Returns
    ///
    /// Handle to the allocated buffer
    ///
    /// # Type Parameters
    ///
    /// * `T` - Element type (must be Copy and have a meaningful zero value)
    pub fn allocate_typed<T: Copy + Default + 'static>(
        &mut self,
        count: usize,
    ) -> AllocationHandle {
        let size = count * std::mem::size_of::<T>();
        let handle = self.allocate(size);

        // Track typed allocation
        self.stats.typed_allocations += 1;
        *self
            .stats
            .allocations_by_type
            .entry(TypeId::of::<T>())
            .or_insert(0) += 1;

        handle
    }

    /// Get a reference to an allocated buffer.
    ///
    /// # Arguments
    ///
    /// * `handle` - Handle returned from `allocate()` or `allocate_typed()`
    ///
    /// # Panics
    ///
    /// Panics if the handle is invalid.
    pub fn get(&self, handle: AllocationHandle) -> &[u8] {
        &self.allocations[handle.0]
    }

    /// Get a mutable reference to an allocated buffer.
    ///
    /// # Arguments
    ///
    /// * `handle` - Handle returned from `allocate()` or `allocate_typed()`
    ///
    /// # Panics
    ///
    /// Panics if the handle is invalid.
    pub fn get_mut(&mut self, handle: AllocationHandle) -> &mut [u8] {
        &mut self.allocations[handle.0]
    }

    /// Get a typed reference to an allocated buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure the buffer was allocated with the correct type
    /// and size using `allocate_typed::<T>()`.
    ///
    /// # Panics
    ///
    /// Panics if the handle is invalid.
    pub fn get_typed<T: Copy>(&self, handle: AllocationHandle) -> &[T] {
        let bytes = &self.allocations[handle.0];
        let len = bytes.len() / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, len) }
    }

    /// Get a mutable typed reference to an allocated buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure the buffer was allocated with the correct type
    /// and size using `allocate_typed::<T>()`.
    ///
    /// # Panics
    ///
    /// Panics if the handle is invalid.
    pub fn get_typed_mut<T: Copy>(&mut self, handle: AllocationHandle) -> &mut [T] {
        let bytes = &mut self.allocations[handle.0];
        let len = bytes.len() / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, len) }
    }

    /// Try to get a reference to an allocated buffer.
    ///
    /// Returns `None` if the handle is invalid.
    pub fn try_get(&self, handle: AllocationHandle) -> Option<&[u8]> {
        self.allocations.get(handle.0).map(|b| b.as_ref())
    }

    /// Try to get a mutable reference to an allocated buffer.
    ///
    /// Returns `None` if the handle is invalid.
    pub fn try_get_mut(&mut self, handle: AllocationHandle) -> Option<&mut [u8]> {
        self.allocations.get_mut(handle.0).map(|b| b.as_mut())
    }

    /// Get the size of an allocation.
    ///
    /// # Panics
    ///
    /// Panics if the handle is invalid.
    pub fn allocation_size(&self, handle: AllocationHandle) -> usize {
        self.sizes[handle.0]
    }

    /// Get the number of allocations in this context.
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Get statistics for this context.
    pub fn stats(&self) -> &ContextStats {
        &self.stats
    }

    /// Release all allocations and reset the context.
    ///
    /// After calling this, all handles become invalid.
    pub fn release_all(&mut self) {
        self.allocations.clear();
        self.sizes.clear();
        self.stats.current_bytes = 0;
    }

    /// Create a sub-context for a nested operation.
    ///
    /// The sub-context shares no allocations with the parent.
    pub fn sub_context(&self, name: impl Into<String>) -> Self {
        Self::new(format!("{}::{}", self.name, name.into()))
    }
}

impl Drop for AnalyticsContext {
    fn drop(&mut self) {
        // All allocations are automatically freed when Vec drops
        // We could add logging here if needed:
        // log::trace!("Dropping AnalyticsContext '{}': {} bytes released", self.name, self.stats.current_bytes);
    }
}

/// Builder for AnalyticsContext with pre-configuration.
pub struct AnalyticsContextBuilder {
    name: String,
    expected_allocations: Option<usize>,
    preallocations: Vec<usize>,
}

impl AnalyticsContextBuilder {
    /// Create a new builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            expected_allocations: None,
            preallocations: Vec::new(),
        }
    }

    /// Set expected number of allocations (reserves vector capacity).
    pub fn with_expected_allocations(mut self, count: usize) -> Self {
        self.expected_allocations = Some(count);
        self
    }

    /// Pre-allocate a buffer of the given size when building.
    pub fn with_preallocation(mut self, size: usize) -> Self {
        self.preallocations.push(size);
        self
    }

    /// Build the context.
    pub fn build(self) -> AnalyticsContext {
        let mut ctx = match self.expected_allocations {
            Some(cap) => {
                AnalyticsContext::with_capacity(self.name, cap.max(self.preallocations.len()))
            }
            None => AnalyticsContext::new(self.name),
        };

        for size in self.preallocations {
            ctx.allocate(size);
        }

        ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = AnalyticsContext::new("test");
        assert_eq!(ctx.name(), "test");
        assert_eq!(ctx.allocation_count(), 0);
    }

    #[test]
    fn test_context_with_capacity() {
        let ctx = AnalyticsContext::with_capacity("test", 10);
        assert_eq!(ctx.name(), "test");
        assert_eq!(ctx.allocation_count(), 0);
    }

    #[test]
    fn test_allocate() {
        let mut ctx = AnalyticsContext::new("test");

        let h1 = ctx.allocate(100);
        let h2 = ctx.allocate(200);

        assert_eq!(ctx.allocation_count(), 2);
        assert_eq!(ctx.allocation_size(h1), 100);
        assert_eq!(ctx.allocation_size(h2), 200);
        assert_eq!(ctx.get(h1).len(), 100);
        assert_eq!(ctx.get(h2).len(), 200);
    }

    #[test]
    fn test_allocate_typed() {
        let mut ctx = AnalyticsContext::new("test");

        let h = ctx.allocate_typed::<u32>(10);

        assert_eq!(ctx.allocation_size(h), 40); // 10 * 4 bytes
        assert_eq!(ctx.get_typed::<u32>(h).len(), 10);

        let stats = ctx.stats();
        assert_eq!(stats.typed_allocations, 1);
    }

    #[test]
    fn test_get_mut() {
        let mut ctx = AnalyticsContext::new("test");

        let h = ctx.allocate(10);
        ctx.get_mut(h)[0] = 42;

        assert_eq!(ctx.get(h)[0], 42);
    }

    #[test]
    fn test_get_typed_mut() {
        let mut ctx = AnalyticsContext::new("test");

        let h = ctx.allocate_typed::<u32>(5);
        ctx.get_typed_mut::<u32>(h)[2] = 12345;

        assert_eq!(ctx.get_typed::<u32>(h)[2], 12345);
    }

    #[test]
    fn test_try_get() {
        let mut ctx = AnalyticsContext::new("test");

        let h = ctx.allocate(10);
        let invalid = AllocationHandle(999);

        assert!(ctx.try_get(h).is_some());
        assert!(ctx.try_get(invalid).is_none());
        assert!(ctx.try_get_mut(h).is_some());
        assert!(ctx.try_get_mut(invalid).is_none());
    }

    #[test]
    fn test_stats_tracking() {
        let mut ctx = AnalyticsContext::new("test");

        ctx.allocate(100);
        ctx.allocate(200);
        ctx.allocate(50);

        let stats = ctx.stats();
        assert_eq!(stats.allocations, 3);
        assert_eq!(stats.current_bytes, 350);
        assert_eq!(stats.peak_bytes, 350);
    }

    #[test]
    fn test_stats_peak_bytes() {
        let mut ctx = AnalyticsContext::new("test");

        ctx.allocate(100);
        ctx.allocate(200);
        let peak = ctx.stats().peak_bytes;

        ctx.release_all();
        assert_eq!(ctx.stats().current_bytes, 0);
        assert_eq!(ctx.stats().peak_bytes, peak); // Peak preserved after release
    }

    #[test]
    fn test_release_all() {
        let mut ctx = AnalyticsContext::new("test");

        let h1 = ctx.allocate(100);
        let h2 = ctx.allocate(200);

        assert_eq!(ctx.allocation_count(), 2);

        ctx.release_all();

        assert_eq!(ctx.allocation_count(), 0);
        assert_eq!(ctx.stats().current_bytes, 0);
        // Handles are now invalid - don't use them!
        assert!(ctx.try_get(h1).is_none());
        assert!(ctx.try_get(h2).is_none());
    }

    #[test]
    fn test_sub_context() {
        let parent = AnalyticsContext::new("parent");
        let child = parent.sub_context("child");

        assert_eq!(child.name(), "parent::child");
    }

    #[test]
    fn test_builder() {
        let ctx = AnalyticsContextBuilder::new("builder_test")
            .with_expected_allocations(10)
            .with_preallocation(100)
            .with_preallocation(200)
            .build();

        assert_eq!(ctx.name(), "builder_test");
        assert_eq!(ctx.allocation_count(), 2);
        assert_eq!(ctx.stats().current_bytes, 300);
    }

    #[test]
    fn test_context_stats_default() {
        let stats = ContextStats::default();
        assert_eq!(stats.allocations, 0);
        assert_eq!(stats.peak_bytes, 0);
        assert_eq!(stats.current_bytes, 0);
        assert_eq!(stats.memory_efficiency(), 1.0);
    }

    #[test]
    fn test_memory_efficiency() {
        let mut ctx = AnalyticsContext::new("test");

        ctx.allocate(100);
        ctx.allocate(100);
        // peak = 200, current = 200
        assert_eq!(ctx.stats().memory_efficiency(), 1.0);

        ctx.release_all();
        // peak = 200, current = 0
        assert_eq!(ctx.stats().memory_efficiency(), 0.0);
    }

    #[test]
    fn test_handle_index() {
        let mut ctx = AnalyticsContext::new("test");

        let h0 = ctx.allocate(10);
        let h1 = ctx.allocate(20);
        let h2 = ctx.allocate(30);

        assert_eq!(h0.index(), 0);
        assert_eq!(h1.index(), 1);
        assert_eq!(h2.index(), 2);
    }

    #[test]
    fn test_zero_allocation() {
        let mut ctx = AnalyticsContext::new("test");

        let h = ctx.allocate(0);
        assert_eq!(ctx.get(h).len(), 0);
        assert_eq!(ctx.allocation_size(h), 0);
    }

    #[test]
    fn test_large_allocation() {
        let mut ctx = AnalyticsContext::new("test");

        // 1 MB allocation
        let h = ctx.allocate(1024 * 1024);
        assert_eq!(ctx.get(h).len(), 1024 * 1024);
        assert_eq!(ctx.stats().current_bytes, 1024 * 1024);
    }
}
