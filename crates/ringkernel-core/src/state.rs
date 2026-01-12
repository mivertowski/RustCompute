//! Control block state helpers for GPU-compatible kernel state.
//!
//! This module provides utilities for storing kernel state either:
//! - **Embedded** in ControlBlock's 24-byte `_reserved` field (zero-copy)
//! - **External** in separate GPU memory with a pointer in the descriptor
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::prelude::*;
//!
//! #[derive(Default, Clone, Copy)]
//! #[repr(C, align(8))]
//! struct OrderBookState {
//!     best_bid: u64,
//!     best_ask: u64,
//!     order_count: u32,
//!     _pad: u32,
//! }  // 24 bytes - fits in ControlBlock._reserved
//!
//! impl EmbeddedState for OrderBookState {}
//!
//! // Write state to control block
//! let mut block = ControlBlock::new();
//! let state = OrderBookState { best_bid: 100, best_ask: 101, order_count: 42, _pad: 0 };
//! ControlBlockStateHelper::write_embedded(&mut block, &state)?;
//!
//! // Read state from control block
//! let restored: OrderBookState = ControlBlockStateHelper::read_embedded(&block)?;
//! ```

use crate::control::ControlBlock;
use crate::error::{Result, RingKernelError};
use bytemuck::{Pod, Zeroable};

/// Size of the reserved field in ControlBlock available for state storage.
pub const CONTROL_BLOCK_STATE_SIZE: usize = 24;

/// Magic number for state descriptor ("STAT" in little-endian).
pub const STATE_DESCRIPTOR_MAGIC: u32 = 0x54415453; // "STAT"

// ============================================================================
// Embedded State Trait
// ============================================================================

/// Trait for state types that can be embedded directly in ControlBlock.
///
/// Types implementing this trait must:
/// - Be `Pod` + `Zeroable` (plain old data, safe to reinterpret)
/// - Be `Default` for initialization
/// - Be `Copy` for efficient transfer
/// - Fit within 24 bytes (checked at compile time via `EmbeddedStateSize`)
///
/// # Example
///
/// ```ignore
/// #[derive(Default, Clone, Copy, Pod, Zeroable)]
/// #[repr(C, align(8))]
/// struct MyState {
///     value_a: u64,
///     value_b: u64,
///     counter: u32,
///     _pad: u32,
/// }
///
/// impl EmbeddedState for MyState {}
/// ```
pub trait EmbeddedState: Pod + Zeroable + Default + Copy + Send + Sync + 'static {
    /// State version for forward compatibility.
    /// Override to support migrations.
    const VERSION: u32 = 1;

    /// Whether this state is embedded (true) or external (false).
    fn is_embedded() -> bool {
        true
    }
}

/// Marker trait to verify state fits in 24 bytes at compile time.
///
/// Use `assert_embedded_size!()` macro or manual assertion.
pub trait EmbeddedStateSize: EmbeddedState {
    /// Compile-time assertion that size fits.
    const SIZE_CHECK: () = assert!(
        std::mem::size_of::<Self>() <= CONTROL_BLOCK_STATE_SIZE,
        "EmbeddedState must fit in 24 bytes"
    );
}

// Automatically implement EmbeddedStateSize for all EmbeddedState types
impl<T: EmbeddedState> EmbeddedStateSize for T {}

// ============================================================================
// State Descriptor
// ============================================================================

/// Descriptor stored in `_reserved` when using external state.
///
/// When state is too large to embed, this 24-byte descriptor points
/// to the external GPU memory location.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(8))]
pub struct StateDescriptor {
    /// Magic number for validation (STATE_DESCRIPTOR_MAGIC).
    pub magic: u32,
    /// State version number.
    pub version: u32,
    /// Total size of external state in bytes.
    pub total_size: u64,
    /// Pointer to external state buffer (GPU address).
    pub external_ptr: u64,
}

// SAFETY: StateDescriptor is #[repr(C)] with only primitive types
unsafe impl Zeroable for StateDescriptor {}
unsafe impl Pod for StateDescriptor {}

impl EmbeddedState for StateDescriptor {}

const _: () = assert!(std::mem::size_of::<StateDescriptor>() == 24);

impl StateDescriptor {
    /// Create a new state descriptor.
    pub const fn new(version: u32, total_size: u64, external_ptr: u64) -> Self {
        Self {
            magic: STATE_DESCRIPTOR_MAGIC,
            version,
            total_size,
            external_ptr,
        }
    }

    /// Check if this descriptor is valid.
    pub fn is_valid(&self) -> bool {
        self.magic == STATE_DESCRIPTOR_MAGIC
    }

    /// Check if state is external (has external pointer).
    pub fn is_external(&self) -> bool {
        self.is_valid() && self.external_ptr != 0
    }

    /// Check if state is embedded (no external pointer).
    pub fn is_embedded(&self) -> bool {
        !self.is_valid() || self.external_ptr == 0
    }
}

// ============================================================================
// GPU State Trait (for external state)
// ============================================================================

/// Trait for GPU-compatible state types that may be stored externally.
///
/// Unlike `EmbeddedState`, types implementing `GpuState` can be larger
/// than 24 bytes and are stored in separate GPU memory.
pub trait GpuState: Send + Sync + 'static {
    /// Serialize state to bytes for GPU transfer.
    fn to_control_block_bytes(&self) -> Vec<u8>;

    /// Deserialize state from bytes read from GPU.
    fn from_control_block_bytes(bytes: &[u8]) -> Result<Self>
    where
        Self: Sized;

    /// State version for compatibility checking.
    fn state_version() -> u32 {
        1
    }

    /// Whether this state should be embedded (if small enough).
    fn prefer_embedded() -> bool
    where
        Self: Sized,
    {
        std::mem::size_of::<Self>() <= CONTROL_BLOCK_STATE_SIZE
    }
}

// Blanket implementation for EmbeddedState types
impl<T: EmbeddedState> GpuState for T {
    fn to_control_block_bytes(&self) -> Vec<u8> {
        bytemuck::bytes_of(self).to_vec()
    }

    fn from_control_block_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return Err(RingKernelError::InvalidState {
                expected: format!("{} bytes", std::mem::size_of::<Self>()),
                actual: format!("{} bytes", bytes.len()),
            });
        }
        Ok(*bytemuck::from_bytes(&bytes[..std::mem::size_of::<Self>()]))
    }

    fn state_version() -> u32 {
        Self::VERSION
    }

    fn prefer_embedded() -> bool {
        true
    }
}

// ============================================================================
// ControlBlock State Helper
// ============================================================================

/// Helper for reading/writing state to/from ControlBlock.
pub struct ControlBlockStateHelper;

impl ControlBlockStateHelper {
    /// Write embedded state to ControlBlock's reserved field.
    ///
    /// # Errors
    ///
    /// Returns error if state doesn't fit in 24 bytes.
    pub fn write_embedded<S: EmbeddedState>(block: &mut ControlBlock, state: &S) -> Result<()> {
        let bytes = bytemuck::bytes_of(state);
        if bytes.len() > CONTROL_BLOCK_STATE_SIZE {
            return Err(RingKernelError::InvalidState {
                expected: format!("<= {} bytes", CONTROL_BLOCK_STATE_SIZE),
                actual: format!("{} bytes", bytes.len()),
            });
        }

        // Clear reserved field first
        block._reserved = [0u8; 24];

        // Copy state bytes
        block._reserved[..bytes.len()].copy_from_slice(bytes);

        Ok(())
    }

    /// Read embedded state from ControlBlock's reserved field.
    ///
    /// # Errors
    ///
    /// Returns error if state type size exceeds 24 bytes.
    pub fn read_embedded<S: EmbeddedState>(block: &ControlBlock) -> Result<S> {
        let size = std::mem::size_of::<S>();
        if size > CONTROL_BLOCK_STATE_SIZE {
            return Err(RingKernelError::InvalidState {
                expected: format!("<= {} bytes", CONTROL_BLOCK_STATE_SIZE),
                actual: format!("{} bytes", size),
            });
        }

        Ok(*bytemuck::from_bytes(&block._reserved[..size]))
    }

    /// Write state descriptor for external state.
    pub fn write_descriptor(block: &mut ControlBlock, descriptor: &StateDescriptor) -> Result<()> {
        Self::write_embedded(block, descriptor)
    }

    /// Read state descriptor from ControlBlock.
    ///
    /// Returns `None` if no valid descriptor is present.
    pub fn read_descriptor(block: &ControlBlock) -> Option<StateDescriptor> {
        let desc: StateDescriptor =
            *bytemuck::from_bytes::<StateDescriptor>(&block._reserved[..24]);
        if desc.is_valid() {
            Some(desc)
        } else {
            None
        }
    }

    /// Check if ControlBlock has embedded state (no external pointer).
    pub fn has_embedded_state(block: &ControlBlock) -> bool {
        match Self::read_descriptor(block) {
            Some(desc) => desc.is_embedded(),
            None => true, // No descriptor means raw embedded bytes
        }
    }

    /// Check if ControlBlock references external state.
    pub fn has_external_state(block: &ControlBlock) -> bool {
        match Self::read_descriptor(block) {
            Some(desc) => desc.is_external(),
            None => false,
        }
    }

    /// Clear all state from ControlBlock.
    pub fn clear_state(block: &mut ControlBlock) {
        block._reserved = [0u8; 24];
    }

    /// Get raw bytes from reserved field.
    pub fn raw_bytes(block: &ControlBlock) -> &[u8; 24] {
        &block._reserved
    }

    /// Get mutable raw bytes from reserved field.
    pub fn raw_bytes_mut(block: &mut ControlBlock) -> &mut [u8; 24] {
        &mut block._reserved
    }
}

// ============================================================================
// State Snapshot
// ============================================================================

/// Snapshot of kernel state for checkpointing.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// State data bytes.
    pub data: Vec<u8>,
    /// State version.
    pub version: u32,
    /// Whether state was embedded or external.
    pub was_embedded: bool,
    /// Kernel ID this state belongs to.
    pub kernel_id: u64,
    /// Timestamp when snapshot was taken (HLC counter).
    pub timestamp: u64,
}

impl StateSnapshot {
    /// Create a new state snapshot.
    pub fn new(data: Vec<u8>, version: u32, was_embedded: bool, kernel_id: u64) -> Self {
        Self {
            data,
            version,
            was_embedded,
            kernel_id,
            timestamp: 0,
        }
    }

    /// Create snapshot with timestamp.
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Deserialize state from snapshot.
    pub fn restore<S: GpuState>(&self) -> Result<S> {
        S::from_control_block_bytes(&self.data)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test embedded state type (exactly 24 bytes)
    #[derive(Default, Clone, Copy, Debug, PartialEq)]
    #[repr(C, align(8))]
    struct TestState {
        value_a: u64,
        value_b: u64,
        counter: u32,
        flags: u32,
    }

    // SAFETY: TestState is #[repr(C)] with only primitive types
    unsafe impl Zeroable for TestState {}
    unsafe impl Pod for TestState {}

    impl EmbeddedState for TestState {}

    // Small state (8 bytes)
    #[derive(Default, Clone, Copy, Debug, PartialEq)]
    #[repr(C)]
    struct SmallState {
        value: u64,
    }

    unsafe impl Zeroable for SmallState {}
    unsafe impl Pod for SmallState {}

    impl EmbeddedState for SmallState {}

    #[test]
    fn test_state_size_constant() {
        assert_eq!(CONTROL_BLOCK_STATE_SIZE, 24);
    }

    #[test]
    fn test_state_descriptor_size() {
        assert_eq!(std::mem::size_of::<StateDescriptor>(), 24);
    }

    #[test]
    fn test_state_descriptor_validation() {
        let desc = StateDescriptor::new(1, 256, 0x1000);
        assert!(desc.is_valid());
        assert!(desc.is_external());
        assert!(!desc.is_embedded());

        let embedded_desc = StateDescriptor::new(1, 24, 0);
        assert!(embedded_desc.is_valid());
        assert!(!embedded_desc.is_external());
        assert!(embedded_desc.is_embedded());

        let invalid_desc = StateDescriptor::default();
        assert!(!invalid_desc.is_valid());
    }

    #[test]
    fn test_write_read_embedded_state() {
        let mut block = ControlBlock::new();
        let state = TestState {
            value_a: 0x1234567890ABCDEF,
            value_b: 0xFEDCBA0987654321,
            counter: 42,
            flags: 0xFF,
        };

        ControlBlockStateHelper::write_embedded(&mut block, &state).unwrap();
        let restored: TestState = ControlBlockStateHelper::read_embedded(&block).unwrap();

        assert_eq!(state, restored);
    }

    #[test]
    fn test_write_read_small_state() {
        let mut block = ControlBlock::new();
        let state = SmallState { value: 42 };

        ControlBlockStateHelper::write_embedded(&mut block, &state).unwrap();
        let restored: SmallState = ControlBlockStateHelper::read_embedded(&block).unwrap();

        assert_eq!(state, restored);
    }

    #[test]
    fn test_write_read_descriptor() {
        let mut block = ControlBlock::new();
        let desc = StateDescriptor::new(2, 1024, 0xDEADBEEF);

        ControlBlockStateHelper::write_descriptor(&mut block, &desc).unwrap();

        let restored = ControlBlockStateHelper::read_descriptor(&block).unwrap();
        assert_eq!(restored.magic, STATE_DESCRIPTOR_MAGIC);
        assert_eq!(restored.version, 2);
        assert_eq!(restored.total_size, 1024);
        assert_eq!(restored.external_ptr, 0xDEADBEEF);
    }

    #[test]
    fn test_has_embedded_external_state() {
        let mut block = ControlBlock::new();

        // Fresh block has embedded state (no descriptor)
        assert!(ControlBlockStateHelper::has_embedded_state(&block));
        assert!(!ControlBlockStateHelper::has_external_state(&block));

        // Write external descriptor
        let desc = StateDescriptor::new(1, 256, 0x1000);
        ControlBlockStateHelper::write_descriptor(&mut block, &desc).unwrap();

        assert!(!ControlBlockStateHelper::has_embedded_state(&block));
        assert!(ControlBlockStateHelper::has_external_state(&block));

        // Write embedded descriptor (external_ptr = 0)
        let desc = StateDescriptor::new(1, 24, 0);
        ControlBlockStateHelper::write_descriptor(&mut block, &desc).unwrap();

        assert!(ControlBlockStateHelper::has_embedded_state(&block));
        assert!(!ControlBlockStateHelper::has_external_state(&block));
    }

    #[test]
    fn test_clear_state() {
        let mut block = ControlBlock::new();
        let state = TestState {
            value_a: 123,
            value_b: 456,
            counter: 789,
            flags: 0xABC,
        };

        ControlBlockStateHelper::write_embedded(&mut block, &state).unwrap();
        assert!(block._reserved.iter().any(|&b| b != 0));

        ControlBlockStateHelper::clear_state(&mut block);
        assert!(block._reserved.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_raw_bytes_access() {
        let mut block = ControlBlock::new();
        block._reserved[0] = 0x42;
        block._reserved[23] = 0xFF;

        let bytes = ControlBlockStateHelper::raw_bytes(&block);
        assert_eq!(bytes[0], 0x42);
        assert_eq!(bytes[23], 0xFF);

        let bytes_mut = ControlBlockStateHelper::raw_bytes_mut(&mut block);
        bytes_mut[1] = 0x99;
        assert_eq!(block._reserved[1], 0x99);
    }

    #[test]
    fn test_gpu_state_trait() {
        let state = TestState {
            value_a: 100,
            value_b: 200,
            counter: 300,
            flags: 400,
        };

        let bytes = state.to_control_block_bytes();
        assert_eq!(bytes.len(), 24);

        let restored = TestState::from_control_block_bytes(&bytes).unwrap();
        assert_eq!(state, restored);

        assert!(TestState::prefer_embedded());
        assert_eq!(TestState::state_version(), 1);
    }

    #[test]
    fn test_state_snapshot() {
        let state = TestState {
            value_a: 1,
            value_b: 2,
            counter: 3,
            flags: 4,
        };

        let snapshot =
            StateSnapshot::new(state.to_control_block_bytes(), 1, true, 42).with_timestamp(1000);

        assert_eq!(snapshot.version, 1);
        assert!(snapshot.was_embedded);
        assert_eq!(snapshot.kernel_id, 42);
        assert_eq!(snapshot.timestamp, 1000);

        let restored: TestState = snapshot.restore().unwrap();
        assert_eq!(state, restored);
    }

    #[test]
    fn test_embedded_state_size_check() {
        // This should compile - TestState is exactly 24 bytes
        assert_eq!(std::mem::size_of::<TestState>(), 24);
        let _ = <TestState as EmbeddedStateSize>::SIZE_CHECK;

        // SmallState is smaller - also OK
        assert!(std::mem::size_of::<SmallState>() <= CONTROL_BLOCK_STATE_SIZE);
        let _ = <SmallState as EmbeddedStateSize>::SIZE_CHECK;
    }
}
