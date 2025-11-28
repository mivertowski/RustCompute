//! Message types and traits for kernel-to-kernel communication.
//!
//! This module defines the core message abstraction used for communication
//! between GPU kernels and between host and device.

use bytemuck::{Pod, Zeroable};
use rkyv::{Archive, Deserialize, Serialize};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

use crate::hlc::HlcTimestamp;

/// Unique message identifier.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
    AsBytes,
    FromBytes,
    FromZeroes,
    Pod,
    Zeroable,
    Archive,
    Serialize,
    Deserialize,
)]
#[repr(C)]
pub struct MessageId(pub u64);

impl MessageId {
    /// Create a new message ID.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Generate a new unique message ID.
    pub fn generate() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the inner value.
    pub const fn inner(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for MessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "msg:{:016x}", self.0)
    }
}

/// Correlation ID for request-response patterns.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
    AsBytes,
    FromBytes,
    FromZeroes,
    Pod,
    Zeroable,
    Archive,
    Serialize,
    Deserialize,
)]
#[repr(C)]
pub struct CorrelationId(pub u64);

impl CorrelationId {
    /// Create a new correlation ID.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Generate a new unique correlation ID.
    pub fn generate() -> Self {
        Self(MessageId::generate().0)
    }

    /// No correlation (for fire-and-forget messages).
    pub const fn none() -> Self {
        Self(0)
    }

    /// Check if this is a valid correlation ID.
    pub const fn is_some(&self) -> bool {
        self.0 != 0
    }
}

/// Message priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Priority {
    /// Low priority (background tasks).
    Low = 0,
    /// Normal priority (default).
    #[default]
    Normal = 1,
    /// High priority (important tasks).
    High = 2,
    /// Critical priority (system messages).
    Critical = 3,
}

impl Priority {
    /// Convert from u8.
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Low,
            1 => Self::Normal,
            2 => Self::High,
            _ => Self::Critical,
        }
    }
}

/// Fixed-size message header (256 bytes, cache-line aligned).
///
/// This header precedes the variable-length payload and provides
/// all metadata needed for routing and processing.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct MessageHeader {
    /// Magic number for validation (0xRINGKERN).
    pub magic: u64,
    /// Header version for compatibility.
    pub version: u32,
    /// Message flags.
    pub flags: u32,
    /// Unique message identifier.
    pub message_id: MessageId,
    /// Correlation ID for request-response.
    pub correlation_id: CorrelationId,
    /// Source kernel ID (0 for host).
    pub source_kernel: u64,
    /// Destination kernel ID (0 for host).
    pub dest_kernel: u64,
    /// Message type discriminator.
    pub message_type: u64,
    /// Priority level.
    pub priority: u8,
    /// Reserved for alignment.
    pub _reserved1: [u8; 7],
    /// Payload size in bytes.
    pub payload_size: u64,
    /// Checksum of payload (CRC32).
    pub checksum: u32,
    /// Reserved for alignment.
    pub _reserved2: u32,
    /// HLC timestamp when message was created.
    pub timestamp: HlcTimestamp,
    /// Deadline timestamp (0 = no deadline).
    pub deadline: HlcTimestamp,
    /// Reserved for future use (split for derive compatibility).
    pub _reserved3a: [u8; 32],
    /// Reserved for future use.
    pub _reserved3b: [u8; 32],
    /// Reserved for future use.
    pub _reserved3c: [u8; 32],
    /// Reserved for future use.
    pub _reserved3d: [u8; 8],
}

impl MessageHeader {
    /// Magic number for validation.
    pub const MAGIC: u64 = 0x52494E474B45524E; // "RINGKERN"

    /// Current header version.
    pub const VERSION: u32 = 1;

    /// Maximum payload size (64KB).
    pub const MAX_PAYLOAD_SIZE: usize = 64 * 1024;

    /// Convert header to bytes.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    /// Read header from bytes.
    pub fn read_from(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return None;
        }
        unsafe { Some(std::ptr::read_unaligned(bytes.as_ptr() as *const Self)) }
    }

    /// Create a new message header.
    pub fn new(
        message_type: u64,
        source_kernel: u64,
        dest_kernel: u64,
        payload_size: usize,
        timestamp: HlcTimestamp,
    ) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            flags: 0,
            message_id: MessageId::generate(),
            correlation_id: CorrelationId::none(),
            source_kernel,
            dest_kernel,
            message_type,
            priority: Priority::Normal as u8,
            _reserved1: [0; 7],
            payload_size: payload_size as u64,
            checksum: 0,
            _reserved2: 0,
            timestamp,
            deadline: HlcTimestamp::zero(),
            _reserved3a: [0; 32],
            _reserved3b: [0; 32],
            _reserved3c: [0; 32],
            _reserved3d: [0; 8],
        }
    }

    /// Validate the header.
    pub fn validate(&self) -> bool {
        self.magic == Self::MAGIC
            && self.version <= Self::VERSION
            && self.payload_size <= Self::MAX_PAYLOAD_SIZE as u64
    }

    /// Set correlation ID.
    pub fn with_correlation(mut self, correlation_id: CorrelationId) -> Self {
        self.correlation_id = correlation_id;
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority as u8;
        self
    }

    /// Set deadline.
    pub fn with_deadline(mut self, deadline: HlcTimestamp) -> Self {
        self.deadline = deadline;
        self
    }
}

impl Default for MessageHeader {
    fn default() -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            flags: 0,
            message_id: MessageId::default(),
            correlation_id: CorrelationId::none(),
            source_kernel: 0,
            dest_kernel: 0,
            message_type: 0,
            priority: Priority::Normal as u8,
            _reserved1: [0; 7],
            payload_size: 0,
            checksum: 0,
            _reserved2: 0,
            timestamp: HlcTimestamp::zero(),
            deadline: HlcTimestamp::zero(),
            _reserved3a: [0; 32],
            _reserved3b: [0; 32],
            _reserved3c: [0; 32],
            _reserved3d: [0; 8],
        }
    }
}

// Verify size at compile time
const _: () = assert!(std::mem::size_of::<MessageHeader>() == 256);

/// Trait for types that can be sent as kernel messages.
///
/// This trait is typically implemented via the `#[derive(RingMessage)]` macro.
///
/// # Example
///
/// ```ignore
/// #[derive(RingMessage)]
/// struct MyRequest {
///     #[message(id)]
///     id: MessageId,
///     data: Vec<f32>,
/// }
/// ```
pub trait RingMessage: Send + Sync + 'static {
    /// Get the message type discriminator.
    fn message_type() -> u64;

    /// Get the message ID.
    fn message_id(&self) -> MessageId;

    /// Get the correlation ID (if any).
    fn correlation_id(&self) -> CorrelationId {
        CorrelationId::none()
    }

    /// Get the priority.
    fn priority(&self) -> Priority {
        Priority::Normal
    }

    /// Serialize the message to bytes.
    fn serialize(&self) -> Vec<u8>;

    /// Deserialize a message from bytes.
    fn deserialize(bytes: &[u8]) -> crate::error::Result<Self>
    where
        Self: Sized;

    /// Get the serialized size hint.
    fn size_hint(&self) -> usize
    where
        Self: Sized,
    {
        std::mem::size_of::<Self>()
    }
}

/// Envelope containing header and serialized payload.
#[derive(Debug, Clone)]
pub struct MessageEnvelope {
    /// Message header.
    pub header: MessageHeader,
    /// Serialized payload.
    pub payload: Vec<u8>,
}

impl MessageEnvelope {
    /// Create a new envelope from a message.
    pub fn new<M: RingMessage>(
        message: &M,
        source_kernel: u64,
        dest_kernel: u64,
        timestamp: HlcTimestamp,
    ) -> Self {
        let payload = message.serialize();
        let header = MessageHeader::new(
            M::message_type(),
            source_kernel,
            dest_kernel,
            payload.len(),
            timestamp,
        )
        .with_correlation(message.correlation_id())
        .with_priority(message.priority());

        Self { header, payload }
    }

    /// Get total size (header + payload).
    pub fn total_size(&self) -> usize {
        std::mem::size_of::<MessageHeader>() + self.payload.len()
    }

    /// Serialize to contiguous bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.total_size());
        bytes.extend_from_slice(self.header.as_bytes());
        bytes.extend_from_slice(&self.payload);
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> crate::error::Result<Self> {
        if bytes.len() < std::mem::size_of::<MessageHeader>() {
            return Err(crate::error::RingKernelError::DeserializationError(
                "buffer too small for header".to_string(),
            ));
        }

        let header_bytes = &bytes[..std::mem::size_of::<MessageHeader>()];
        let header = MessageHeader::read_from(header_bytes).ok_or_else(|| {
            crate::error::RingKernelError::DeserializationError("invalid header".to_string())
        })?;

        if !header.validate() {
            return Err(crate::error::RingKernelError::ValidationError(
                "header validation failed".to_string(),
            ));
        }

        let payload_start = std::mem::size_of::<MessageHeader>();
        let payload_end = payload_start + header.payload_size as usize;

        if bytes.len() < payload_end {
            return Err(crate::error::RingKernelError::DeserializationError(
                "buffer too small for payload".to_string(),
            ));
        }

        let payload = bytes[payload_start..payload_end].to_vec();

        Ok(Self { header, payload })
    }

    /// Create an empty envelope (for testing).
    pub fn empty(source_kernel: u64, dest_kernel: u64, timestamp: HlcTimestamp) -> Self {
        let header = MessageHeader::new(0, source_kernel, dest_kernel, 0, timestamp);
        Self {
            header,
            payload: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_id_generation() {
        let id1 = MessageId::generate();
        let id2 = MessageId::generate();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_header_validation() {
        let header = MessageHeader::new(1, 0, 1, 100, HlcTimestamp::zero());
        assert!(header.validate());

        let mut invalid = header;
        invalid.magic = 0;
        assert!(!invalid.validate());
    }

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<MessageHeader>(), 256);
    }

    #[test]
    fn test_priority_conversion() {
        assert_eq!(Priority::from_u8(0), Priority::Low);
        assert_eq!(Priority::from_u8(1), Priority::Normal);
        assert_eq!(Priority::from_u8(2), Priority::High);
        assert_eq!(Priority::from_u8(3), Priority::Critical);
        assert_eq!(Priority::from_u8(255), Priority::Critical);
    }

    #[test]
    fn test_envelope_roundtrip() {
        let header = MessageHeader::new(42, 0, 1, 8, HlcTimestamp::now(1));
        let envelope = MessageEnvelope {
            header,
            payload: vec![1, 2, 3, 4, 5, 6, 7, 8],
        };

        let bytes = envelope.to_bytes();
        let restored = MessageEnvelope::from_bytes(&bytes).unwrap();

        assert_eq!(envelope.header.message_type, restored.header.message_type);
        assert_eq!(envelope.payload, restored.payload);
    }
}
