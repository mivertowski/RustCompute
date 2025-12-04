//! Process event types for GPU processing.
//!
//! GPU-aligned structures for efficient kernel execution.

use rkyv::{Archive, Deserialize, Serialize};

/// Hybrid timestamp combining physical and logical clocks.
///
/// Provides causal ordering across distributed systems.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Archive, Serialize, Deserialize,
)]
#[repr(C)]
pub struct HybridTimestamp {
    /// Physical time in milliseconds since epoch.
    pub physical_ms: u64,
    /// Logical counter for ordering events at same physical time.
    pub logical: u64,
}

impl HybridTimestamp {
    /// Create a new timestamp.
    pub fn new(physical_ms: u64, logical: u64) -> Self {
        Self {
            physical_ms,
            logical,
        }
    }

    /// Create a timestamp from current time.
    pub fn now() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let physical_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        Self {
            physical_ms,
            logical: 0,
        }
    }

    /// Create a zero timestamp.
    pub fn zero() -> Self {
        Self::default()
    }

    /// Tick the logical counter.
    pub fn tick(&mut self) {
        self.logical += 1;
    }

    /// Update from a remote timestamp (for HLC sync).
    pub fn update(&mut self, remote: &HybridTimestamp) {
        if remote.physical_ms > self.physical_ms {
            self.physical_ms = remote.physical_ms;
            self.logical = remote.logical + 1;
        } else if remote.physical_ms == self.physical_ms {
            self.logical = self.logical.max(remote.logical) + 1;
        } else {
            self.logical += 1;
        }
    }
}

/// Event type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum EventType {
    /// Activity started.
    Start = 0,
    /// Activity completed.
    #[default]
    Complete = 1,
    /// Activity scheduled.
    Schedule = 2,
    /// Activity assigned to resource.
    Assign = 3,
    /// Activity suspended.
    Suspend = 4,
    /// Activity resumed.
    Resume = 5,
    /// Activity aborted.
    Abort = 6,
}

/// GPU-compatible process event (128 bytes, cache-line aligned).
///
/// Represents a single event in an event log, optimized for GPU processing.
#[derive(Debug, Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C, align(128))]
pub struct GpuObjectEvent {
    /// Unique event identifier.
    pub event_id: u64,
    /// Case/trace identifier (object ID in OCPM terms).
    pub object_id: u64,
    /// Activity identifier.
    pub activity_id: u32,
    /// Event type (start, complete, etc.).
    pub event_type: u8,
    /// Padding for alignment.
    pub _padding1: [u8; 3],
    /// Event timestamp with HLC.
    pub timestamp: HybridTimestamp,
    /// Resource that performed the activity.
    pub resource_id: u32,
    /// Cost associated with the event.
    pub cost: f32,
    /// Duration in milliseconds (for interval events).
    pub duration_ms: u32,
    /// Event flags (custom attributes).
    pub flags: u32,
    /// Custom attribute values.
    pub attributes: [u32; 4],
    /// Object type identifier (for OCPM).
    pub object_type_id: u32,
    /// Related object ID (for object relationships).
    pub related_object_id: u64,
    /// Reserved for future use.
    pub _reserved: [u8; 36],
}

impl Default for GpuObjectEvent {
    fn default() -> Self {
        Self {
            event_id: 0,
            object_id: 0,
            activity_id: 0,
            event_type: EventType::Complete as u8,
            _padding1: [0; 3],
            timestamp: HybridTimestamp::default(),
            resource_id: 0,
            cost: 0.0,
            duration_ms: 0,
            flags: 0,
            attributes: [0; 4],
            object_type_id: 0,
            related_object_id: 0,
            _reserved: [0; 36],
        }
    }
}

impl GpuObjectEvent {
    /// Create a new event.
    pub fn new(
        event_id: u64,
        object_id: u64,
        activity_id: u32,
        event_type: EventType,
        timestamp: HybridTimestamp,
    ) -> Self {
        Self {
            event_id,
            object_id,
            activity_id,
            event_type: event_type as u8,
            timestamp,
            ..Default::default()
        }
    }

    /// Get the event type.
    pub fn get_event_type(&self) -> EventType {
        match self.event_type {
            0 => EventType::Start,
            1 => EventType::Complete,
            2 => EventType::Schedule,
            3 => EventType::Assign,
            4 => EventType::Suspend,
            5 => EventType::Resume,
            6 => EventType::Abort,
            _ => EventType::Complete,
        }
    }

    /// Check if this is a start event.
    pub fn is_start(&self) -> bool {
        self.event_type == EventType::Start as u8
    }

    /// Check if this is a complete event.
    pub fn is_complete(&self) -> bool {
        self.event_type == EventType::Complete as u8
    }
}

// Verify size at compile time
const _: () = assert!(std::mem::size_of::<GpuObjectEvent>() == 128);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_size() {
        assert_eq!(std::mem::size_of::<GpuObjectEvent>(), 128);
    }

    #[test]
    fn test_timestamp_ordering() {
        let t1 = HybridTimestamp::new(1000, 0);
        let t2 = HybridTimestamp::new(1000, 1);
        let t3 = HybridTimestamp::new(1001, 0);

        assert!(t1 < t2);
        assert!(t2 < t3);
    }

    #[test]
    fn test_hlc_update() {
        let mut local = HybridTimestamp::new(1000, 5);
        let remote = HybridTimestamp::new(1000, 10);
        local.update(&remote);
        assert_eq!(local.logical, 11);
    }
}
