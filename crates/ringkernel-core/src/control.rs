//! Control block for kernel state management.
//!
//! The control block is a fixed-size structure in GPU memory that manages
//! kernel lifecycle, message queue pointers, and synchronization state.

use crate::hlc::HlcState;

/// Kernel control block (128 bytes, cache-line aligned).
///
/// This structure resides in GPU global memory and is accessed atomically
/// by both host and device code for kernel lifecycle management.
///
/// ## Memory Layout
///
/// The structure is carefully designed to minimize false sharing:
/// - Frequently written fields are grouped together
/// - Read-only fields are separated
/// - Padding ensures proper alignment
#[derive(Debug, Clone, Copy)]
#[repr(C, align(128))]
pub struct ControlBlock {
    // === Lifecycle State (frequently accessed) ===
    /// Kernel is actively processing messages (atomic bool).
    pub is_active: u32,
    /// Signal to terminate the kernel (atomic bool).
    pub should_terminate: u32,
    /// Kernel has completed termination (atomic bool).
    pub has_terminated: u32,
    /// Reserved for alignment.
    pub _pad1: u32,

    // === Counters (frequently updated) ===
    /// Total messages processed.
    pub messages_processed: u64,
    /// Messages currently being processed.
    pub messages_in_flight: u64,

    // === Queue Pointers ===
    /// Input queue head pointer (producer writes).
    pub input_head: u64,
    /// Input queue tail pointer (consumer reads).
    pub input_tail: u64,
    /// Output queue head pointer (producer writes).
    pub output_head: u64,
    /// Output queue tail pointer (consumer reads).
    pub output_tail: u64,

    // === Queue Metadata (read-mostly) ===
    /// Input queue capacity (power of 2).
    pub input_capacity: u32,
    /// Output queue capacity (power of 2).
    pub output_capacity: u32,
    /// Input queue mask (capacity - 1).
    pub input_mask: u32,
    /// Output queue mask (capacity - 1).
    pub output_mask: u32,

    // === Timing ===
    /// HLC state for this kernel.
    pub hlc_state: HlcState,

    // === Error State ===
    /// Last error code (0 = no error).
    pub last_error: u32,
    /// Error count.
    pub error_count: u32,

    // === Reserved (pad to 128 bytes) ===
    /// Reserved for future use.
    pub _reserved: [u8; 24],
}

// Verify size at compile time
const _: () = assert!(std::mem::size_of::<ControlBlock>() == 128);

impl ControlBlock {
    /// Create a new control block with default values.
    pub const fn new() -> Self {
        Self {
            is_active: 0,
            should_terminate: 0,
            has_terminated: 0,
            _pad1: 0,
            messages_processed: 0,
            messages_in_flight: 0,
            input_head: 0,
            input_tail: 0,
            output_head: 0,
            output_tail: 0,
            input_capacity: 0,
            output_capacity: 0,
            input_mask: 0,
            output_mask: 0,
            hlc_state: HlcState::new(0, 0),
            last_error: 0,
            error_count: 0,
            _reserved: [0; 24],
        }
    }

    /// Create with specified queue capacities.
    ///
    /// Capacities must be powers of 2.
    pub fn with_capacities(input_capacity: u32, output_capacity: u32) -> Self {
        debug_assert!(input_capacity.is_power_of_two());
        debug_assert!(output_capacity.is_power_of_two());

        Self {
            is_active: 0,
            should_terminate: 0,
            has_terminated: 0,
            _pad1: 0,
            messages_processed: 0,
            messages_in_flight: 0,
            input_head: 0,
            input_tail: 0,
            output_head: 0,
            output_tail: 0,
            input_capacity,
            output_capacity,
            input_mask: input_capacity.saturating_sub(1),
            output_mask: output_capacity.saturating_sub(1),
            hlc_state: HlcState::new(0, 0),
            last_error: 0,
            error_count: 0,
            _reserved: [0; 24],
        }
    }

    /// Check if the kernel is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.is_active != 0
    }

    /// Check if termination was requested.
    #[inline]
    pub fn should_terminate(&self) -> bool {
        self.should_terminate != 0
    }

    /// Check if the kernel has terminated.
    #[inline]
    pub fn has_terminated(&self) -> bool {
        self.has_terminated != 0
    }

    /// Get input queue size.
    #[inline]
    pub fn input_queue_size(&self) -> u64 {
        self.input_head.wrapping_sub(self.input_tail)
    }

    /// Get output queue size.
    #[inline]
    pub fn output_queue_size(&self) -> u64 {
        self.output_head.wrapping_sub(self.output_tail)
    }

    /// Check if input queue is empty.
    #[inline]
    pub fn input_queue_empty(&self) -> bool {
        self.input_head == self.input_tail
    }

    /// Check if output queue is empty.
    #[inline]
    pub fn output_queue_empty(&self) -> bool {
        self.output_head == self.output_tail
    }

    /// Check if input queue is full.
    #[inline]
    pub fn input_queue_full(&self) -> bool {
        self.input_queue_size() >= self.input_capacity as u64
    }

    /// Check if output queue is full.
    #[inline]
    pub fn output_queue_full(&self) -> bool {
        self.output_queue_size() >= self.output_capacity as u64
    }
}

impl Default for ControlBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// Error codes for control block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ControlError {
    /// No error.
    None = 0,
    /// Input queue overflow.
    InputOverflow = 1,
    /// Output queue overflow.
    OutputOverflow = 2,
    /// Invalid message.
    InvalidMessage = 3,
    /// Memory allocation failed.
    AllocationFailed = 4,
    /// Serialization error.
    SerializationError = 5,
    /// Timeout waiting for message.
    Timeout = 6,
    /// Internal kernel error.
    InternalError = 7,
}

impl ControlError {
    /// Convert from u32.
    pub const fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::InputOverflow,
            2 => Self::OutputOverflow,
            3 => Self::InvalidMessage,
            4 => Self::AllocationFailed,
            5 => Self::SerializationError,
            6 => Self::Timeout,
            _ => Self::InternalError,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_block_size() {
        assert_eq!(std::mem::size_of::<ControlBlock>(), 128);
    }

    #[test]
    fn test_control_block_alignment() {
        assert_eq!(std::mem::align_of::<ControlBlock>(), 128);
    }

    #[test]
    fn test_queue_size_calculation() {
        let mut cb = ControlBlock::with_capacities(1024, 1024);

        cb.input_head = 10;
        cb.input_tail = 5;
        assert_eq!(cb.input_queue_size(), 5);

        // Test wraparound
        cb.input_head = 2;
        cb.input_tail = u64::MAX - 3;
        assert_eq!(cb.input_queue_size(), 6);
    }

    #[test]
    fn test_queue_full_empty() {
        let mut cb = ControlBlock::with_capacities(16, 16);

        assert!(cb.input_queue_empty());
        assert!(!cb.input_queue_full());

        cb.input_head = 16;
        cb.input_tail = 0;
        assert!(!cb.input_queue_empty());
        assert!(cb.input_queue_full());
    }

    #[test]
    fn test_lifecycle_flags() {
        let mut cb = ControlBlock::new();

        assert!(!cb.is_active());
        assert!(!cb.should_terminate());
        assert!(!cb.has_terminated());

        cb.is_active = 1;
        assert!(cb.is_active());

        cb.should_terminate = 1;
        assert!(cb.should_terminate());

        cb.has_terminated = 1;
        assert!(cb.has_terminated());
    }
}
