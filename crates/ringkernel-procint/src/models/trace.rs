//! Process trace definitions.

use super::{ActivityId, GpuObjectEvent, HybridTimestamp};
use rkyv::{Archive, Deserialize, Serialize};

/// Trace identifier type.
pub type TraceId = u64;

/// A process trace (case) containing a sequence of events.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct ProcessTrace {
    /// Unique trace identifier.
    pub id: TraceId,
    /// Case identifier (business key).
    pub case_id: String,
    /// Sequence of activity IDs in execution order.
    pub activities: Vec<ActivityId>,
    /// Timestamps for each activity.
    pub timestamps: Vec<HybridTimestamp>,
    /// Durations for each activity (ms).
    pub durations: Vec<u32>,
    /// Total trace duration (ms).
    pub total_duration_ms: u64,
    /// First event timestamp.
    pub start_time: HybridTimestamp,
    /// Last event timestamp.
    pub end_time: HybridTimestamp,
    /// Whether trace is complete.
    pub is_complete: bool,
    /// Variant identifier (for trace clustering).
    pub variant_id: u32,
}

impl ProcessTrace {
    /// Create a new empty trace.
    pub fn new(id: TraceId, case_id: impl Into<String>) -> Self {
        Self {
            id,
            case_id: case_id.into(),
            activities: Vec::new(),
            timestamps: Vec::new(),
            durations: Vec::new(),
            total_duration_ms: 0,
            start_time: HybridTimestamp::default(),
            end_time: HybridTimestamp::default(),
            is_complete: false,
            variant_id: 0,
        }
    }

    /// Add an activity to the trace.
    pub fn add_activity(
        &mut self,
        activity_id: ActivityId,
        timestamp: HybridTimestamp,
        duration_ms: u32,
    ) {
        if self.activities.is_empty() {
            self.start_time = timestamp;
        }
        self.activities.push(activity_id);
        self.timestamps.push(timestamp);
        self.durations.push(duration_ms);
        self.end_time = timestamp;
        self.total_duration_ms = self
            .end_time
            .physical_ms
            .saturating_sub(self.start_time.physical_ms);
    }

    /// Get trace length.
    pub fn len(&self) -> usize {
        self.activities.len()
    }

    /// Check if trace is empty.
    pub fn is_empty(&self) -> bool {
        self.activities.is_empty()
    }

    /// Get activity at index.
    pub fn get(&self, index: usize) -> Option<ActivityId> {
        self.activities.get(index).copied()
    }

    /// Get edges (directly-follows pairs).
    pub fn edges(&self) -> impl Iterator<Item = (ActivityId, ActivityId)> + '_ {
        self.activities.windows(2).map(|w| (w[0], w[1]))
    }

    /// Mark trace as complete.
    pub fn complete(&mut self) {
        self.is_complete = true;
    }
}

/// Builder for constructing traces from events.
#[derive(Debug, Default)]
pub struct TraceBuilder {
    traces: std::collections::HashMap<u64, ProcessTrace>,
    next_trace_id: TraceId,
}

impl TraceBuilder {
    /// Create a new trace builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Process an event and update the corresponding trace.
    pub fn process_event(&mut self, event: &GpuObjectEvent) {
        let trace = self.traces.entry(event.object_id).or_insert_with(|| {
            let id = self.next_trace_id;
            self.next_trace_id += 1;
            ProcessTrace::new(id, event.object_id.to_string())
        });

        trace.add_activity(event.activity_id, event.timestamp, event.duration_ms);
    }

    /// Get all traces.
    pub fn traces(&self) -> impl Iterator<Item = &ProcessTrace> {
        self.traces.values()
    }

    /// Get trace by object ID.
    pub fn get_trace(&self, object_id: u64) -> Option<&ProcessTrace> {
        self.traces.get(&object_id)
    }

    /// Take ownership of all traces.
    pub fn into_traces(self) -> Vec<ProcessTrace> {
        self.traces.into_values().collect()
    }

    /// Number of traces.
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }
}

/// GPU-compatible trace representation for batch processing.
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct GpuProcessTrace {
    /// Trace identifier.
    pub trace_id: u64,
    /// Case identifier hash.
    pub case_id_hash: u64,
    /// Number of activities.
    pub activity_count: u32,
    /// Variant identifier.
    pub variant_id: u32,
    /// Start timestamp.
    pub start_time_ms: u64,
    /// End timestamp.
    pub end_time_ms: u64,
    /// Total duration.
    pub total_duration_ms: u64,
    /// Flags.
    pub flags: u32,
    /// Padding.
    pub _padding: [u8; 4],
}

// Verify size: 8+8+4+4+8+8+8+4+4 = 56, aligned to 64
const _: () = assert!(std::mem::size_of::<GpuProcessTrace>() == 64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_builder() {
        let mut builder = TraceBuilder::new();

        let event1 = GpuObjectEvent::new(
            1,
            100,
            1,
            super::super::EventType::Complete,
            HybridTimestamp::new(1000, 0),
        );
        let event2 = GpuObjectEvent::new(
            2,
            100,
            2,
            super::super::EventType::Complete,
            HybridTimestamp::new(2000, 0),
        );

        builder.process_event(&event1);
        builder.process_event(&event2);

        let trace = builder.get_trace(100).unwrap();
        assert_eq!(trace.len(), 2);
        assert_eq!(trace.activities, vec![1, 2]);
    }
}
