//! Persistent Message Traits for Type-Based Kernel Dispatch
//!
//! This module provides traits and types for user-defined message dispatch within
//! persistent GPU kernels. It enables multiple analytics types (fraud detection,
//! aggregations, pattern detection) to run within a single persistent kernel with
//! type-based routing to specialized handlers.
//!
//! # Architecture
//!
//! ```text
//! Host                    GPU (Persistent Kernel)
//! ┌──────────────┐       ┌─────────────────────────────────────┐
//! │ send_message │──────▶│ H2K Queue                           │
//! │ <FraudCheck> │       │   ↓                                 │
//! │              │       │ Type Dispatcher (switch on type_id) │
//! │              │       │   ├─▶ handle_fraud_check()          │
//! │              │       │   ├─▶ handle_aggregate()            │
//! │              │       │   └─▶ handle_pattern_detect()       │
//! │              │       │         ↓                           │
//! │ poll_typed   │◀──────│ K2H Queue                           │
//! │ <FraudResult>│       └─────────────────────────────────────┘
//! └──────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::persistent_message::{PersistentMessage, DispatchTable};
//! use ringkernel_derive::{RingMessage, PersistentMessage};
//!
//! #[derive(RingMessage, PersistentMessage)]
//! #[message(type_id = 1001)]
//! #[persistent_message(handler_id = 1, requires_response = true)]
//! pub struct FraudCheckRequest {
//!     pub transaction_id: u64,
//!     pub amount: f32,
//!     pub account_id: u32,
//! }
//!
//! // Runtime usage
//! sim.send_message(FraudCheckRequest { ... })?;  // ~0.03µs
//! let results: Vec<FraudCheckResult> = sim.poll_typed();
//! ```

use crate::message::RingMessage;

/// Maximum size for inline payload in extended messages.
/// Messages larger than this must use external buffer references.
pub const MAX_INLINE_PAYLOAD_SIZE: usize = 32;

/// Flags for extended H2K messages.
pub mod message_flags {
    /// Flag indicating this is an extended message format.
    pub const FLAG_EXTENDED: u32 = 0x01;
    /// Flag indicating this message has high priority.
    pub const FLAG_HIGH_PRIORITY: u32 = 0x02;
    /// Flag indicating message uses external buffer.
    pub const FLAG_EXTERNAL_BUFFER: u32 = 0x04;
    /// Flag indicating this message requires a response.
    pub const FLAG_REQUIRES_RESPONSE: u32 = 0x08;
}

/// Trait for messages that can be dispatched within a persistent GPU kernel.
///
/// This trait extends `RingMessage` with additional metadata needed for
/// type-based dispatch within a unified kernel. Each message type is
/// associated with a handler ID that maps to a CUDA device function.
///
/// # Implementation
///
/// Use the `#[derive(PersistentMessage)]` macro for automatic implementation:
///
/// ```ignore
/// #[derive(RingMessage, PersistentMessage)]
/// #[message(type_id = 1001)]
/// #[persistent_message(handler_id = 1, requires_response = true)]
/// pub struct FraudCheckRequest {
///     pub transaction_id: u64,
///     pub amount: f32,
///     pub account_id: u32,
/// }
/// ```
pub trait PersistentMessage: RingMessage + Sized {
    /// Handler ID for CUDA dispatch (0-255).
    ///
    /// This maps to a case in the generated switch statement:
    /// ```cuda
    /// switch (msg->handler_id) {
    ///     case 1: handle_fraud_check(msg, state, response); break;
    ///     // ...
    /// }
    /// ```
    fn handler_id() -> u32;

    /// Whether this message type expects a response.
    ///
    /// When true, the kernel will generate a response message after
    /// processing. The caller should use `poll_typed::<ResponseType>()`
    /// to retrieve responses.
    fn requires_response() -> bool {
        false
    }

    /// Convert message to inline payload bytes.
    ///
    /// Returns `Some([u8; 32])` if the message fits in 32 bytes,
    /// `None` if the message requires external buffer allocation.
    fn to_inline_payload(&self) -> Option<[u8; MAX_INLINE_PAYLOAD_SIZE]>;

    /// Reconstruct message from inline payload bytes.
    ///
    /// # Errors
    ///
    /// Returns error if the payload is invalid or incomplete.
    fn from_inline_payload(payload: &[u8]) -> crate::error::Result<Self>;

    /// Get the serialized payload size in bytes.
    fn payload_size() -> usize;

    /// Check if this message type can be inlined (fits in 32 bytes).
    fn can_inline() -> bool {
        Self::payload_size() <= MAX_INLINE_PAYLOAD_SIZE
    }
}

/// Handler registration entry for the dispatch table.
#[derive(Debug, Clone)]
pub struct HandlerRegistration {
    /// Handler ID (0-255).
    pub handler_id: u32,
    /// Name of the handler function.
    pub name: String,
    /// Message type ID (from RingMessage::message_type()).
    pub message_type_id: u64,
    /// Whether this handler produces responses.
    pub produces_response: bool,
    /// Response type ID (if produces_response is true).
    pub response_type_id: Option<u64>,
    /// CUDA function body (for code generation).
    pub cuda_body: Option<String>,
}

impl HandlerRegistration {
    /// Create a new handler registration.
    pub fn new(handler_id: u32, name: impl Into<String>, message_type_id: u64) -> Self {
        Self {
            handler_id,
            name: name.into(),
            message_type_id,
            produces_response: false,
            response_type_id: None,
            cuda_body: None,
        }
    }

    /// Set whether this handler produces responses.
    pub fn with_response(mut self, response_type_id: u64) -> Self {
        self.produces_response = true;
        self.response_type_id = Some(response_type_id);
        self
    }

    /// Set the CUDA function body for code generation.
    pub fn with_cuda_body(mut self, body: impl Into<String>) -> Self {
        self.cuda_body = Some(body.into());
        self
    }
}

/// Dispatch table mapping handler IDs to functions.
///
/// Used during code generation to build the CUDA switch statement.
#[derive(Debug, Clone, Default)]
pub struct DispatchTable {
    /// Registered handlers.
    handlers: Vec<HandlerRegistration>,
    /// Maximum handler ID seen.
    max_handler_id: u32,
}

impl DispatchTable {
    /// Create a new empty dispatch table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a handler.
    ///
    /// # Panics
    ///
    /// Panics if a handler with the same ID is already registered.
    pub fn register(&mut self, registration: HandlerRegistration) {
        // Check for duplicate handler ID
        if self
            .handlers
            .iter()
            .any(|h| h.handler_id == registration.handler_id)
        {
            panic!(
                "Duplicate handler ID: {} ({})",
                registration.handler_id, registration.name
            );
        }

        self.max_handler_id = self.max_handler_id.max(registration.handler_id);
        self.handlers.push(registration);
    }

    /// Register a handler from a PersistentMessage type.
    pub fn register_message<M: PersistentMessage>(&mut self, name: impl Into<String>) {
        let registration = HandlerRegistration::new(M::handler_id(), name, M::message_type());

        let registration = if M::requires_response() {
            // Note: Response type ID would need to be provided separately
            registration
        } else {
            registration
        };

        self.register(registration);
    }

    /// Get all registered handlers.
    pub fn handlers(&self) -> &[HandlerRegistration] {
        &self.handlers
    }

    /// Get a handler by ID.
    pub fn get(&self, handler_id: u32) -> Option<&HandlerRegistration> {
        self.handlers.iter().find(|h| h.handler_id == handler_id)
    }

    /// Get the maximum handler ID.
    pub fn max_handler_id(&self) -> u32 {
        self.max_handler_id
    }

    /// Get the number of registered handlers.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Check if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_table_registration() {
        let mut table = DispatchTable::new();

        table.register(HandlerRegistration::new(1, "fraud_check", 1001));
        table.register(HandlerRegistration::new(2, "aggregate", 1002));
        table.register(HandlerRegistration::new(3, "pattern_detect", 1003).with_response(2003));

        assert_eq!(table.len(), 3);
        assert_eq!(table.max_handler_id(), 3);

        let handler = table.get(2).unwrap();
        assert_eq!(handler.name, "aggregate");
        assert_eq!(handler.message_type_id, 1002);
        assert!(!handler.produces_response);

        let handler = table.get(3).unwrap();
        assert!(handler.produces_response);
        assert_eq!(handler.response_type_id, Some(2003));
    }

    #[test]
    #[should_panic(expected = "Duplicate handler ID")]
    fn test_duplicate_handler_panics() {
        let mut table = DispatchTable::new();
        table.register(HandlerRegistration::new(1, "first", 1001));
        table.register(HandlerRegistration::new(1, "second", 1002)); // Should panic
    }

    #[test]
    fn test_message_flags() {
        assert_eq!(message_flags::FLAG_EXTENDED, 0x01);
        assert_eq!(message_flags::FLAG_HIGH_PRIORITY, 0x02);
        assert_eq!(message_flags::FLAG_EXTERNAL_BUFFER, 0x04);
        assert_eq!(message_flags::FLAG_REQUIRES_RESPONSE, 0x08);
    }

    #[test]
    fn test_max_inline_payload_size() {
        assert_eq!(MAX_INLINE_PAYLOAD_SIZE, 32);
    }
}
