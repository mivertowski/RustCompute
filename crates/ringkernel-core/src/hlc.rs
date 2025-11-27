//! Hybrid Logical Clock (HLC) implementation for causal ordering.
//!
//! HLC combines physical time with logical counters to provide total ordering
//! of events across distributed GPU kernels while maintaining close relationship
//! with real time.
//!
//! ## Properties
//!
//! - **Total Ordering**: All timestamps can be compared
//! - **Causality**: If event A causes event B, then HLC(A) < HLC(B)
//! - **Bounded Drift**: Physical component stays within bounded drift of real time
//!
//! ## Usage
//!
//! ```
//! use ringkernel_core::hlc::{HlcTimestamp, HlcClock};
//!
//! let mut clock = HlcClock::new(1); // Node ID = 1
//! let ts1 = clock.now();
//! let ts2 = clock.tick();
//! assert!(ts1 < ts2);
//! ```

use bytemuck::{Pod, Zeroable};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

use crate::error::{RingKernelError, Result};

/// Maximum allowed clock skew in milliseconds.
pub const MAX_CLOCK_SKEW_MS: u64 = 60_000; // 1 minute

/// Hybrid Logical Clock timestamp.
///
/// Composed of:
/// - Physical time (wall clock in microseconds since epoch)
/// - Logical counter (for events at same physical time)
/// - Node ID (for tie-breaking across nodes)
///
/// This struct is 24 bytes and cache-line friendly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[derive(AsBytes, FromBytes, FromZeroes, Pod, Zeroable)]
#[repr(C, align(8))]
pub struct HlcTimestamp {
    /// Physical time component (microseconds since UNIX epoch).
    pub physical: u64,
    /// Logical counter for events at the same physical time.
    pub logical: u64,
    /// Node identifier for tie-breaking.
    pub node_id: u64,
}

impl HlcTimestamp {
    /// Create a new HLC timestamp.
    pub const fn new(physical: u64, logical: u64, node_id: u64) -> Self {
        Self {
            physical,
            logical,
            node_id,
        }
    }

    /// Create a zero timestamp (minimum value).
    pub const fn zero() -> Self {
        Self {
            physical: 0,
            logical: 0,
            node_id: 0,
        }
    }

    /// Create a timestamp from the current wall clock.
    pub fn now(node_id: u64) -> Self {
        let physical = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_micros() as u64;

        Self {
            physical,
            logical: 0,
            node_id,
        }
    }

    /// Check if this timestamp is zero/uninitialized.
    pub const fn is_zero(&self) -> bool {
        self.physical == 0 && self.logical == 0
    }

    /// Get physical time as microseconds since epoch.
    pub const fn as_micros(&self) -> u64 {
        self.physical
    }

    /// Get physical time as milliseconds since epoch.
    pub const fn as_millis(&self) -> u64 {
        self.physical / 1000
    }

    /// Pack timestamp into a single u128 for atomic comparison.
    /// Format: [physical:64][logical:48][node_id:16]
    pub const fn pack(&self) -> u128 {
        ((self.physical as u128) << 64)
            | ((self.logical as u128) << 16)
            | (self.node_id as u128 & 0xFFFF)
    }

    /// Unpack timestamp from u128.
    pub const fn unpack(packed: u128) -> Self {
        Self {
            physical: (packed >> 64) as u64,
            logical: ((packed >> 16) & 0xFFFF_FFFF_FFFF) as u64,
            node_id: (packed & 0xFFFF) as u64,
        }
    }
}

impl Default for HlcTimestamp {
    fn default() -> Self {
        Self::zero()
    }
}

impl Ord for HlcTimestamp {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare physical time first
        match self.physical.cmp(&other.physical) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        // Then logical counter
        match self.logical.cmp(&other.logical) {
            std::cmp::Ordering::Equal => {}
            ord => return ord,
        }
        // Finally node_id for total ordering
        self.node_id.cmp(&other.node_id)
    }
}

impl PartialOrd for HlcTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::fmt::Display for HlcTimestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HLC({}.{}.{})", self.physical, self.logical, self.node_id)
    }
}

/// Hybrid Logical Clock for generating causally-ordered timestamps.
///
/// Thread-safe implementation using atomics for the state.
pub struct HlcClock {
    /// Current physical time (atomically updated).
    physical: AtomicU64,
    /// Current logical counter (atomically updated).
    logical: AtomicU64,
    /// Node identifier.
    node_id: u64,
    /// Maximum allowed clock drift in microseconds.
    max_drift_us: u64,
}

impl HlcClock {
    /// Create a new HLC clock with the given node ID.
    pub fn new(node_id: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_micros() as u64;

        Self {
            physical: AtomicU64::new(now),
            logical: AtomicU64::new(0),
            node_id,
            max_drift_us: MAX_CLOCK_SKEW_MS * 1000,
        }
    }

    /// Create a new HLC clock with custom max drift.
    pub fn with_max_drift(node_id: u64, max_drift_ms: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_micros() as u64;

        Self {
            physical: AtomicU64::new(now),
            logical: AtomicU64::new(0),
            node_id,
            max_drift_us: max_drift_ms * 1000,
        }
    }

    /// Get the node ID.
    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    /// Get current timestamp without advancing the clock.
    pub fn now(&self) -> HlcTimestamp {
        let wall = Self::wall_time();
        let physical = self.physical.load(Ordering::Acquire);
        let logical = self.logical.load(Ordering::Acquire);

        // Use max of wall clock and stored physical
        let new_physical = physical.max(wall);

        HlcTimestamp {
            physical: new_physical,
            logical,
            node_id: self.node_id,
        }
    }

    /// Generate a new timestamp, advancing the clock.
    pub fn tick(&self) -> HlcTimestamp {
        let wall = Self::wall_time();

        loop {
            let old_physical = self.physical.load(Ordering::Acquire);
            let old_logical = self.logical.load(Ordering::Acquire);

            let (new_physical, new_logical) = if wall > old_physical {
                // Wall clock advanced: use wall time, reset logical
                (wall, 0)
            } else {
                // Same or past: increment logical counter
                (old_physical, old_logical + 1)
            };

            // Try to update atomically
            if self
                .physical
                .compare_exchange(old_physical, new_physical, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                self.logical.store(new_logical, Ordering::Release);
                return HlcTimestamp {
                    physical: new_physical,
                    logical: new_logical,
                    node_id: self.node_id,
                };
            }
            // CAS failed, retry
        }
    }

    /// Update clock on receiving a message with the given timestamp.
    ///
    /// Returns the new local timestamp that causally follows the received timestamp.
    pub fn update(&self, received: &HlcTimestamp) -> Result<HlcTimestamp> {
        let wall = Self::wall_time();

        // Check for clock skew
        if received.physical > wall + self.max_drift_us {
            return Err(RingKernelError::ClockSkew {
                skew_ms: (received.physical - wall) / 1000,
                max_ms: self.max_drift_us / 1000,
            });
        }

        loop {
            let old_physical = self.physical.load(Ordering::Acquire);
            let old_logical = self.logical.load(Ordering::Acquire);

            // Take max of wall, local, and received physical
            let max_physical = wall.max(old_physical).max(received.physical);

            let new_logical = if max_physical == old_physical && max_physical == received.physical {
                // All three equal: take max logical + 1
                old_logical.max(received.logical) + 1
            } else if max_physical == old_physical {
                // Local physical wins: increment local logical
                old_logical + 1
            } else if max_physical == received.physical {
                // Received physical wins: use received logical + 1
                received.logical + 1
            } else {
                // Wall clock wins: reset logical
                0
            };

            // Try to update atomically
            if self
                .physical
                .compare_exchange(old_physical, max_physical, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                self.logical.store(new_logical, Ordering::Release);
                return Ok(HlcTimestamp {
                    physical: max_physical,
                    logical: new_logical,
                    node_id: self.node_id,
                });
            }
            // CAS failed, retry
        }
    }

    /// Get current wall clock time in microseconds.
    fn wall_time() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_micros() as u64
    }
}

impl std::fmt::Debug for HlcClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HlcClock")
            .field("physical", &self.physical.load(Ordering::Relaxed))
            .field("logical", &self.logical.load(Ordering::Relaxed))
            .field("node_id", &self.node_id)
            .finish()
    }
}

/// Compact HLC state for GPU-side storage (16 bytes).
#[derive(Debug, Clone, Copy, Default)]
#[derive(AsBytes, FromBytes, FromZeroes, Pod, Zeroable)]
#[repr(C, align(16))]
pub struct HlcState {
    /// Physical time in microseconds.
    pub physical: u64,
    /// Logical counter.
    pub logical: u64,
}

impl HlcState {
    /// Create new HLC state.
    pub const fn new(physical: u64, logical: u64) -> Self {
        Self { physical, logical }
    }

    /// Convert to full timestamp with node ID.
    pub const fn to_timestamp(&self, node_id: u64) -> HlcTimestamp {
        HlcTimestamp {
            physical: self.physical,
            logical: self.logical,
            node_id,
        }
    }

    /// Create from full timestamp (drops node_id).
    pub const fn from_timestamp(ts: &HlcTimestamp) -> Self {
        Self {
            physical: ts.physical,
            logical: ts.logical,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_ordering() {
        let ts1 = HlcTimestamp::new(100, 0, 1);
        let ts2 = HlcTimestamp::new(100, 1, 1);
        let ts3 = HlcTimestamp::new(101, 0, 1);

        assert!(ts1 < ts2);
        assert!(ts2 < ts3);
        assert!(ts1 < ts3);
    }

    #[test]
    fn test_timestamp_node_id_tiebreak() {
        let ts1 = HlcTimestamp::new(100, 5, 1);
        let ts2 = HlcTimestamp::new(100, 5, 2);

        assert!(ts1 < ts2);
    }

    #[test]
    fn test_clock_tick() {
        let clock = HlcClock::new(1);

        let ts1 = clock.tick();
        let ts2 = clock.tick();
        let ts3 = clock.tick();

        assert!(ts1 < ts2);
        assert!(ts2 < ts3);
    }

    #[test]
    fn test_clock_update() {
        let clock1 = HlcClock::new(1);
        let clock2 = HlcClock::new(2);

        let ts1 = clock1.tick();
        let ts2 = clock2.update(&ts1).unwrap();

        // ts2 should causally follow ts1
        assert!(ts1 < ts2);
    }

    #[test]
    fn test_pack_unpack() {
        let original = HlcTimestamp::new(12345678901234, 42, 7);
        let packed = original.pack();
        let unpacked = HlcTimestamp::unpack(packed);

        assert_eq!(original.physical, unpacked.physical);
        // Note: node_id is truncated to 16 bits in pack format
        assert_eq!(original.logical, unpacked.logical);
    }

    #[test]
    fn test_clock_skew_detection() {
        let clock = HlcClock::with_max_drift(1, 100); // 100ms max drift

        // Create a timestamp far in the future
        let future = HlcTimestamp::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64
                + 200_000_000, // 200 seconds in future
            0,
            2,
        );

        let result = clock.update(&future);
        assert!(matches!(result, Err(RingKernelError::ClockSkew { .. })));
    }

    #[test]
    fn test_timestamp_display() {
        let ts = HlcTimestamp::new(1234567890, 42, 7);
        let s = format!("{}", ts);
        assert!(s.contains("1234567890"));
        assert!(s.contains("42"));
        assert!(s.contains("7"));
    }
}
