//! Resource guard for preventing system overload.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

use super::error::{ResourceError, ResourceResult};
use super::estimate::MemoryEstimate;
use super::system::get_available_memory;
use super::{DEFAULT_MAX_MEMORY_BYTES, SYSTEM_MEMORY_MARGIN};

/// Resource guard for preventing system overload.
///
/// Tracks memory allocation and provides checks before large allocations.
#[derive(Debug)]
pub struct ResourceGuard {
    /// Maximum allowed memory usage.
    max_memory_bytes: AtomicU64,
    /// Current tracked memory usage.
    current_memory_bytes: AtomicU64,
    /// Current reserved memory.
    reserved_bytes: AtomicU64,
    /// Whether to enforce limits (can be disabled for testing).
    enforce_limits: AtomicBool,
    /// Safety margin ratio (0.0-1.0).
    safety_margin: f32,
}

impl Default for ResourceGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceGuard {
    /// Creates a new resource guard with default limits.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_memory_bytes: AtomicU64::new(DEFAULT_MAX_MEMORY_BYTES),
            current_memory_bytes: AtomicU64::new(0),
            reserved_bytes: AtomicU64::new(0),
            enforce_limits: AtomicBool::new(true),
            safety_margin: 0.3, // 30% headroom
        }
    }

    /// Creates a resource guard with a specific memory limit.
    #[must_use]
    pub fn with_max_memory(max_bytes: u64) -> Self {
        Self {
            max_memory_bytes: AtomicU64::new(max_bytes),
            current_memory_bytes: AtomicU64::new(0),
            reserved_bytes: AtomicU64::new(0),
            enforce_limits: AtomicBool::new(true),
            safety_margin: 0.3,
        }
    }

    /// Creates a resource guard with custom safety margin.
    #[must_use]
    pub fn with_safety_margin(mut self, margin: f32) -> Self {
        self.safety_margin = margin.clamp(0.0, 0.9);
        self
    }

    /// Creates a resource guard that doesn't enforce limits (for testing).
    #[must_use]
    pub fn unguarded() -> Self {
        let guard = Self::new();
        guard.enforce_limits.store(false, Ordering::SeqCst);
        guard
    }

    /// Sets the maximum memory limit.
    pub fn set_max_memory(&self, max_bytes: u64) {
        self.max_memory_bytes.store(max_bytes, Ordering::SeqCst);
    }

    /// Gets the maximum memory limit.
    #[must_use]
    pub fn max_memory(&self) -> u64 {
        self.max_memory_bytes.load(Ordering::SeqCst)
    }

    /// Gets the current tracked memory usage.
    #[must_use]
    pub fn current_memory(&self) -> u64 {
        self.current_memory_bytes.load(Ordering::SeqCst)
    }

    /// Gets the currently reserved memory.
    #[must_use]
    pub fn reserved_memory(&self) -> u64 {
        self.reserved_bytes.load(Ordering::SeqCst)
    }

    /// Gets the effective available memory (max - current - reserved).
    #[must_use]
    pub fn available_memory(&self) -> u64 {
        let max = self.max_memory();
        let current = self.current_memory();
        let reserved = self.reserved_memory();
        max.saturating_sub(current).saturating_sub(reserved)
    }

    /// Checks if a given allocation can proceed.
    #[must_use]
    pub fn can_allocate(&self, bytes: u64) -> bool {
        if !self.enforce_limits.load(Ordering::SeqCst) {
            return true;
        }

        bytes <= self.available_memory()
    }

    /// Checks if an allocation can proceed, also checking system memory.
    pub fn can_allocate_safe(&self, bytes: u64) -> ResourceResult<()> {
        if !self.enforce_limits.load(Ordering::SeqCst) {
            return Ok(());
        }

        // Check against our limit
        let current = self.current_memory();
        let reserved = self.reserved_memory();
        let max = self.max_memory();
        let used = current.saturating_add(reserved);

        if used.saturating_add(bytes) > max {
            return Err(ResourceError::MemoryLimitExceeded {
                requested: bytes,
                current: used,
                max,
            });
        }

        // Check system memory
        if let Some(available) = get_available_memory() {
            if bytes > available.saturating_sub(SYSTEM_MEMORY_MARGIN) {
                return Err(ResourceError::InsufficientSystemMemory {
                    requested: bytes,
                    available,
                    margin: SYSTEM_MEMORY_MARGIN,
                });
            }
        }

        Ok(())
    }

    /// Records a memory allocation.
    pub fn record_allocation(&self, bytes: u64) {
        self.current_memory_bytes.fetch_add(bytes, Ordering::SeqCst);
    }

    /// Records a memory deallocation.
    pub fn record_deallocation(&self, bytes: u64) {
        self.current_memory_bytes.fetch_sub(bytes, Ordering::SeqCst);
    }

    /// Reserves memory for a future allocation.
    ///
    /// Returns a guard that releases the reservation on drop.
    pub fn reserve(&self, bytes: u64) -> ResourceResult<ReservationGuard<'_>> {
        self.can_allocate_safe(bytes)?;
        self.reserved_bytes.fetch_add(bytes, Ordering::SeqCst);
        Ok(ReservationGuard {
            guard: self,
            bytes,
            committed: false,
        })
    }

    /// Validates a memory estimate before allocation.
    pub fn validate(&self, estimate: &MemoryEstimate) -> ResourceResult<()> {
        self.can_allocate_safe(estimate.peak_bytes)
    }

    /// Returns the maximum safe element count for a given per-element byte cost.
    #[must_use]
    pub fn max_safe_elements(&self, bytes_per_element: usize) -> usize {
        if bytes_per_element == 0 {
            return usize::MAX;
        }

        let available = self.available_memory();
        let safe_bytes = (available as f64 * (1.0 - self.safety_margin as f64)) as u64;

        (safe_bytes / bytes_per_element as u64) as usize
    }

    /// Enables or disables limit enforcement.
    pub fn set_enforce_limits(&self, enforce: bool) {
        self.enforce_limits.store(enforce, Ordering::SeqCst);
    }

    /// Returns whether limits are being enforced.
    #[must_use]
    pub fn is_enforcing(&self) -> bool {
        self.enforce_limits.load(Ordering::SeqCst)
    }
}

/// RAII guard for memory reservations.
///
/// Releases the reservation when dropped, unless committed.
#[derive(Debug)]
pub struct ReservationGuard<'a> {
    guard: &'a ResourceGuard,
    bytes: u64,
    committed: bool,
}

impl<'a> ReservationGuard<'a> {
    /// Returns the reserved bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Commits the reservation by recording it as an allocation.
    ///
    /// The reservation is released and replaced with an actual allocation record.
    pub fn commit(mut self) {
        self.guard
            .reserved_bytes
            .fetch_sub(self.bytes, Ordering::SeqCst);
        self.guard.record_allocation(self.bytes);
        self.committed = true;
    }

    /// Releases the reservation without allocating.
    pub fn release(mut self) {
        self.guard
            .reserved_bytes
            .fetch_sub(self.bytes, Ordering::SeqCst);
        self.committed = true; // Prevent double-release in drop
    }
}

impl<'a> Drop for ReservationGuard<'a> {
    fn drop(&mut self) {
        if !self.committed {
            self.guard
                .reserved_bytes
                .fetch_sub(self.bytes, Ordering::SeqCst);
        }
    }
}

// ============================================================================
// Global Guard
// ============================================================================

static GLOBAL_GUARD: OnceLock<ResourceGuard> = OnceLock::new();

/// Gets the global resource guard.
pub fn global_guard() -> &'static ResourceGuard {
    GLOBAL_GUARD.get_or_init(ResourceGuard::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_guard_new() {
        let guard = ResourceGuard::new();
        assert_eq!(guard.current_memory(), 0);
        assert_eq!(guard.max_memory(), DEFAULT_MAX_MEMORY_BYTES);
    }

    #[test]
    fn test_resource_guard_allocation() {
        let guard = ResourceGuard::with_max_memory(1_000_000);

        // Should allow small allocation
        assert!(guard.can_allocate(100_000));

        // Should deny large allocation
        assert!(!guard.can_allocate(2_000_000));

        // After recording, should update current
        guard.record_allocation(500_000);
        assert_eq!(guard.current_memory(), 500_000);

        // Now should deny medium allocation
        assert!(!guard.can_allocate(600_000));
        assert!(guard.can_allocate(400_000));
    }

    #[test]
    fn test_resource_guard_deallocation() {
        let guard = ResourceGuard::with_max_memory(1_000_000);

        guard.record_allocation(500_000);
        assert_eq!(guard.current_memory(), 500_000);

        guard.record_deallocation(200_000);
        assert_eq!(guard.current_memory(), 300_000);
    }

    #[test]
    fn test_reservation_guard() {
        let guard = ResourceGuard::with_max_memory(1_000_000);

        {
            let reservation = guard.reserve(500_000).unwrap();
            assert_eq!(guard.reserved_memory(), 500_000);
            assert_eq!(reservation.bytes(), 500_000);

            // Can't allocate more than remaining
            assert!(!guard.can_allocate(600_000));
        } // Reservation released on drop

        assert_eq!(guard.reserved_memory(), 0);
        assert!(guard.can_allocate(600_000));
    }

    #[test]
    fn test_reservation_commit() {
        let guard = ResourceGuard::with_max_memory(1_000_000);

        let reservation = guard.reserve(500_000).unwrap();
        reservation.commit();

        assert_eq!(guard.reserved_memory(), 0);
        assert_eq!(guard.current_memory(), 500_000);
    }

    #[test]
    fn test_reservation_release() {
        let guard = ResourceGuard::with_max_memory(1_000_000);

        let reservation = guard.reserve(500_000).unwrap();
        reservation.release();

        assert_eq!(guard.reserved_memory(), 0);
        assert_eq!(guard.current_memory(), 0);
    }

    #[test]
    fn test_max_safe_elements() {
        let guard = ResourceGuard::with_max_memory(1_000_000);

        // With 30% safety margin, can use ~70% = ~700_000 bytes
        // At 100 bytes per element, that's ~7000 elements
        // Allow +/- 1 for floating point rounding
        let max_elements = guard.max_safe_elements(100);
        assert!(
            (6999..=7001).contains(&max_elements),
            "max_elements {} not in range [6999, 7001]",
            max_elements
        );
    }

    #[test]
    fn test_unguarded() {
        let guard = ResourceGuard::unguarded();

        // Should always allow allocation
        assert!(guard.can_allocate(u64::MAX / 2));
    }

    #[test]
    fn test_global_guard() {
        let guard = global_guard();
        assert_eq!(guard.current_memory(), 0);
    }
}
