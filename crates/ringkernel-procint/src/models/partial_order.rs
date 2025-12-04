//! Partial order structures for concurrent activity analysis.
//!
//! Represents precedence relationships between activities using bit-packed matrices.

use super::{ActivityId, HybridTimestamp};
use rkyv::{Archive, Deserialize, Serialize};

/// GPU-compatible partial order trace (256 bytes).
///
/// Contains a 16x16 bit precedence matrix representing ordering relationships.
#[derive(Debug, Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C, align(256))]
pub struct GpuPartialOrderTrace {
    /// Trace identifier.
    pub trace_id: u64,
    /// Case identifier.
    pub case_id: u64,
    /// Number of events in the trace.
    pub event_count: u32,
    /// Number of unique activities.
    pub activity_count: u32,
    /// Start timestamp.
    pub start_time: HybridTimestamp,
    /// End timestamp.
    pub end_time: HybridTimestamp,
    /// Maximum concurrency level (width).
    pub max_width: u32,
    /// Flags: has_cycles, is_total_order, has_transitive_closure.
    pub flags: u32,
    /// Precedence matrix (16x16 bits = 32 bytes).
    /// Row i bit j set means activity i precedes activity j.
    pub precedence_matrix: [u16; 16],
    /// Activity ID mapping (index -> activity ID), up to 16 activities.
    pub activity_ids: [u32; 16],
    /// Activity start times (relative to trace start, in SECONDS), up to 16 activities.
    /// Using seconds instead of ms to avoid u16 overflow for long-running activities.
    pub activity_start_secs: [u16; 16],
    /// Activity durations (in SECONDS), up to 16 activities.
    /// Using seconds instead of ms to avoid u16 overflow for long-running activities.
    pub activity_duration_secs: [u16; 16],
    /// Reserved padding to reach 256 bytes.
    pub _reserved: [u8; 32],
}

impl Default for GpuPartialOrderTrace {
    fn default() -> Self {
        Self {
            trace_id: 0,
            case_id: 0,
            event_count: 0,
            activity_count: 0,
            start_time: HybridTimestamp::default(),
            end_time: HybridTimestamp::default(),
            max_width: 0,
            flags: 0,
            precedence_matrix: [0; 16],
            activity_ids: [u32::MAX; 16],
            activity_start_secs: [0; 16],
            activity_duration_secs: [0; 16],
            _reserved: [0; 32],
        }
    }
}

// Verify size: 8+8+4+4+16+16+4+4+32+64+32+32+32 = 256
const _: () = assert!(std::mem::size_of::<GpuPartialOrderTrace>() == 256);

impl GpuPartialOrderTrace {
    /// Create a new partial order trace.
    pub fn new(trace_id: u64, case_id: u64) -> Self {
        Self {
            trace_id,
            case_id,
            ..Default::default()
        }
    }

    /// Set precedence: i precedes j.
    pub fn set_precedence(&mut self, i: usize, j: usize) {
        if i < 16 && j < 16 {
            self.precedence_matrix[i] |= 1 << j;
        }
    }

    /// Check if i precedes j.
    pub fn precedes(&self, i: usize, j: usize) -> bool {
        if i < 16 && j < 16 {
            (self.precedence_matrix[i] & (1 << j)) != 0
        } else {
            false
        }
    }

    /// Check if i and j are concurrent (neither precedes the other).
    pub fn is_concurrent(&self, i: usize, j: usize) -> bool {
        !self.precedes(i, j) && !self.precedes(j, i)
    }

    /// Add an activity and return its index.
    pub fn add_activity(&mut self, activity_id: ActivityId) -> Option<usize> {
        // Check if already exists
        for i in 0..self.activity_count as usize {
            if self.activity_ids[i] == activity_id {
                return Some(i);
            }
        }

        // Add new (max 16 activities)
        if (self.activity_count as usize) < 16 {
            let idx = self.activity_count as usize;
            self.activity_ids[idx] = activity_id;
            self.activity_count += 1;
            Some(idx)
        } else {
            None
        }
    }

    /// Get activity index by ID.
    pub fn get_activity_index(&self, activity_id: ActivityId) -> Option<usize> {
        (0..self.activity_count as usize).find(|&i| self.activity_ids[i] == activity_id)
    }

    /// Compute transitive closure using Floyd-Warshall.
    pub fn compute_transitive_closure(&mut self) {
        let n = self.activity_count as usize;
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if self.precedes(i, k) && self.precedes(k, j) {
                        self.set_precedence(i, j);
                    }
                }
            }
        }
        self.flags |= 0x01; // Mark as having transitive closure
    }

    /// Check if has cycles (after transitive closure, if i precedes i).
    pub fn has_cycles(&self) -> bool {
        for i in 0..self.activity_count as usize {
            if self.precedes(i, i) {
                return true;
            }
        }
        false
    }

    /// Check if this is a total order (all pairs are ordered).
    pub fn is_total_order(&self) -> bool {
        let n = self.activity_count as usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if !self.precedes(i, j) && !self.precedes(j, i) {
                    return false;
                }
            }
        }
        true
    }

    /// Compute width (maximum antichain size = maximum concurrency).
    pub fn compute_width(&mut self) -> u32 {
        // Simple greedy approach: find largest set of mutually concurrent activities
        let n = self.activity_count as usize;
        let mut max_width = 1u32;

        // For each activity, count how many are concurrent with it
        for i in 0..n {
            let mut concurrent_count = 1;
            for j in 0..n {
                if i != j && self.is_concurrent(i, j) {
                    concurrent_count += 1;
                }
            }
            max_width = max_width.max(concurrent_count);
        }

        self.max_width = max_width;
        max_width
    }

    /// Get concurrent activities with given activity.
    pub fn concurrent_with(&self, activity_idx: usize) -> Vec<usize> {
        let n = self.activity_count as usize;
        (0..n)
            .filter(|&j| j != activity_idx && self.is_concurrent(activity_idx, j))
            .collect()
    }
}

/// Conflict matrix for tracking conflicts across multiple traces.
#[derive(Debug, Clone)]
pub struct ConflictMatrix {
    /// Alphabet size (number of unique activities).
    pub alphabet_size: usize,
    /// Number of traces analyzed.
    pub trace_count: u32,
    /// Conflict threshold for detecting conflict relationship.
    pub conflict_threshold: f32,
    /// Conflict counts: conflicts[i][j] = times i and j were concurrent.
    pub conflict_counts: Vec<Vec<u32>>,
    /// Co-occurrence counts: cooccurrence[i][j] = times i and j appeared together.
    pub cooccurrence_counts: Vec<Vec<u32>>,
}

impl ConflictMatrix {
    /// Create a new conflict matrix.
    pub fn new(alphabet_size: usize) -> Self {
        Self {
            alphabet_size,
            trace_count: 0,
            conflict_threshold: 0.1,
            conflict_counts: vec![vec![0; alphabet_size]; alphabet_size],
            cooccurrence_counts: vec![vec![0; alphabet_size]; alphabet_size],
        }
    }

    /// Update from a partial order trace.
    pub fn update(&mut self, trace: &GpuPartialOrderTrace) {
        let n = trace.activity_count as usize;

        for i in 0..n {
            for j in (i + 1)..n {
                let ai = trace.activity_ids[i] as usize;
                let aj = trace.activity_ids[j] as usize;

                if ai < self.alphabet_size && aj < self.alphabet_size {
                    // Both appear together
                    self.cooccurrence_counts[ai][aj] += 1;
                    self.cooccurrence_counts[aj][ai] += 1;

                    // If concurrent, they're in conflict
                    if trace.is_concurrent(i, j) {
                        self.conflict_counts[ai][aj] += 1;
                        self.conflict_counts[aj][ai] += 1;
                    }
                }
            }
        }

        self.trace_count += 1;
    }

    /// Get conflict ratio between two activities.
    pub fn conflict_ratio(&self, i: usize, j: usize) -> f32 {
        let cooc = self.cooccurrence_counts[i][j];
        if cooc > 0 {
            self.conflict_counts[i][j] as f32 / cooc as f32
        } else {
            0.0
        }
    }

    /// Check if two activities are in conflict relation.
    pub fn are_in_conflict(&self, i: usize, j: usize) -> bool {
        self.conflict_ratio(i, j) >= self.conflict_threshold
    }
}

/// Interval event for partial order derivation.
#[derive(Debug, Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct IntervalEvent {
    /// Event identifier.
    pub event_id: u64,
    /// Activity identifier.
    pub activity_id: u32,
    /// Padding.
    pub _padding: u32,
    /// Start timestamp.
    pub start_time: HybridTimestamp,
    /// End timestamp.
    pub end_time: HybridTimestamp,
}

impl IntervalEvent {
    /// Check if this event precedes another (ends before other starts).
    pub fn precedes(&self, other: &IntervalEvent) -> bool {
        self.end_time.physical_ms <= other.start_time.physical_ms
    }

    /// Check if this event overlaps with another.
    pub fn overlaps(&self, other: &IntervalEvent) -> bool {
        self.start_time.physical_ms < other.end_time.physical_ms
            && other.start_time.physical_ms < self.end_time.physical_ms
    }

    /// Check if concurrent (overlaps but neither strictly precedes).
    pub fn is_concurrent(&self, other: &IntervalEvent) -> bool {
        !self.precedes(other) && !other.precedes(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_order_size() {
        assert_eq!(std::mem::size_of::<GpuPartialOrderTrace>(), 256);
    }

    #[test]
    fn test_precedence_operations() {
        let mut po = GpuPartialOrderTrace::new(1, 100);

        po.add_activity(10);
        po.add_activity(20);
        po.add_activity(30);

        po.set_precedence(0, 1); // A precedes B
        po.set_precedence(1, 2); // B precedes C

        assert!(po.precedes(0, 1));
        assert!(po.precedes(1, 2));
        assert!(!po.precedes(0, 2)); // Not yet transitive

        po.compute_transitive_closure();
        assert!(po.precedes(0, 2)); // Now transitive

        assert!(!po.has_cycles());
        assert!(po.is_total_order());
    }

    #[test]
    fn test_concurrent_detection() {
        let mut po = GpuPartialOrderTrace::new(1, 100);

        po.add_activity(10);
        po.add_activity(20);
        po.add_activity(30);

        po.set_precedence(0, 2); // A precedes C
        po.set_precedence(1, 2); // B precedes C
                                 // A and B are concurrent (neither precedes the other)

        assert!(po.is_concurrent(0, 1));
        assert!(!po.is_concurrent(0, 2));
    }

    #[test]
    fn test_interval_events() {
        let e1 = IntervalEvent {
            event_id: 1,
            activity_id: 1,
            _padding: 0,
            start_time: HybridTimestamp::new(0, 0),
            end_time: HybridTimestamp::new(100, 0),
        };

        let e2 = IntervalEvent {
            event_id: 2,
            activity_id: 2,
            _padding: 0,
            start_time: HybridTimestamp::new(100, 0),
            end_time: HybridTimestamp::new(200, 0),
        };

        let e3 = IntervalEvent {
            event_id: 3,
            activity_id: 3,
            _padding: 0,
            start_time: HybridTimestamp::new(50, 0),
            end_time: HybridTimestamp::new(150, 0),
        };

        assert!(e1.precedes(&e2));
        assert!(!e1.precedes(&e3)); // They overlap
        assert!(e1.is_concurrent(&e3));
    }
}
