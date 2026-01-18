//! Multi-Kernel Message Dispatcher
//!
//! This module provides a `KernelDispatcher` that routes messages by type_id
//! to appropriate handler kernels. It builds on the K2K broker infrastructure
//! to enable type-based message routing across multiple GPU kernels.
//!
//! # Architecture
//!
//! ```text
//! Host Application
//!       │
//!       ▼
//! ┌─────────────────────────────────────────────────┐
//! │            KernelDispatcher                      │
//! │  ┌─────────────────────────────────────────┐   │
//! │  │ Route Table (type_id → kernel_id)       │   │
//! │  │  1001 → fraud_processor                 │   │
//! │  │  1002 → aggregator                      │   │
//! │  │  1003 → pattern_detector                │   │
//! │  └─────────────────────────────────────────┘   │
//! │                     │                           │
//! │                     ▼                           │
//! │  ┌─────────────────────────────────────────┐   │
//! │  │            K2K Broker                   │   │
//! │  └─────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────┘
//!                       │
//!       ┌───────────────┼───────────────┐
//!       ▼               ▼               ▼
//! ┌──────────┐   ┌──────────┐   ┌──────────┐
//! │ Kernel A │   │ Kernel B │   │ Kernel C │
//! └──────────┘   └──────────┘   └──────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::dispatcher::{KernelDispatcher, DispatcherBuilder};
//! use ringkernel_core::k2k::K2KBroker;
//!
//! // Create dispatcher with routes
//! let broker = K2KBroker::new(K2KConfig::default());
//! let dispatcher = DispatcherBuilder::new()
//!     .route::<FraudCheckRequest>(KernelId::new("fraud_processor"))
//!     .route::<AggregateRequest>(KernelId::new("aggregator"))
//!     .build(broker);
//!
//! // Dispatch a message (routing determined by type_id)
//! let envelope = MessageEnvelope::from_message(&fraud_check, clock.now());
//! let receipt = dispatcher.dispatch(envelope).await?;
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Result, RingKernelError};
use crate::hlc::HlcTimestamp;
use crate::k2k::{DeliveryReceipt, DeliveryStatus, K2KBroker, K2KConfig};
use crate::message::MessageEnvelope;
use crate::persistent_message::{DispatchTable, PersistentMessage};
use crate::runtime::KernelId;

/// Configuration for the kernel dispatcher.
#[derive(Debug, Clone)]
pub struct DispatcherConfig {
    /// Enable logging of dispatch operations.
    pub enable_logging: bool,
    /// Enable metrics collection.
    pub enable_metrics: bool,
    /// Default priority for dispatched messages.
    pub default_priority: u8,
}

impl Default for DispatcherConfig {
    fn default() -> Self {
        Self {
            enable_logging: false,
            enable_metrics: true,
            default_priority: 0,
        }
    }
}

/// Metrics for dispatcher operations.
#[derive(Debug, Default)]
pub struct DispatcherMetrics {
    /// Total messages dispatched.
    pub messages_dispatched: u64,
    /// Messages successfully delivered.
    pub messages_delivered: u64,
    /// Messages that failed to route (unknown type).
    pub unknown_type_errors: u64,
    /// Messages that failed to deliver (queue full, etc.).
    pub delivery_errors: u64,
}

/// Routes messages by type_id to registered handler kernels.
///
/// The dispatcher maintains a routing table mapping message type IDs to kernel IDs.
/// When a message envelope is dispatched, the dispatcher looks up the type_id
/// in the routing table and forwards the message to the appropriate kernel
/// via the K2K broker.
pub struct KernelDispatcher {
    /// Routing table: type_id -> kernel_id
    routes: RwLock<HashMap<u64, KernelId>>,
    /// Handler dispatch tables per kernel (for CUDA codegen)
    handler_tables: RwLock<HashMap<KernelId, DispatchTable>>,
    /// K2K broker for message delivery
    broker: Arc<K2KBroker>,
    /// Configuration
    config: DispatcherConfig,
    /// Metrics
    metrics: RwLock<DispatcherMetrics>,
}

impl KernelDispatcher {
    /// Create a new dispatcher builder.
    pub fn builder() -> DispatcherBuilder {
        DispatcherBuilder::new()
    }

    /// Create a new dispatcher with the given broker.
    pub fn new(broker: Arc<K2KBroker>) -> Self {
        Self::with_config(broker, DispatcherConfig::default())
    }

    /// Create a new dispatcher with custom configuration.
    pub fn with_config(broker: Arc<K2KBroker>, config: DispatcherConfig) -> Self {
        Self {
            routes: RwLock::new(HashMap::new()),
            handler_tables: RwLock::new(HashMap::new()),
            broker,
            config,
            metrics: RwLock::new(DispatcherMetrics::default()),
        }
    }

    /// Register a message type to route to a specific kernel.
    ///
    /// # Type Parameters
    ///
    /// - `M`: A message type implementing `PersistentMessage`
    ///
    /// # Arguments
    ///
    /// - `kernel_id`: The kernel that will handle messages of this type
    pub fn register<M: PersistentMessage>(&self, kernel_id: KernelId) {
        self.register_with_name::<M>(kernel_id, std::any::type_name::<M>());
    }

    /// Register a message type with a custom handler name.
    pub fn register_with_name<M: PersistentMessage>(
        &self,
        kernel_id: KernelId,
        handler_name: &str,
    ) {
        let type_id = M::message_type();

        // Add to routing table
        self.routes.write().insert(type_id, kernel_id.clone());

        // Add to handler table for the kernel
        let mut handler_tables = self.handler_tables.write();
        let table = handler_tables.entry(kernel_id).or_default();
        table.register_message::<M>(handler_name);
    }

    /// Register a route with explicit type_id (for dynamic registration).
    pub fn register_route(&self, type_id: u64, kernel_id: KernelId) {
        self.routes.write().insert(type_id, kernel_id);
    }

    /// Unregister a message type.
    pub fn unregister(&self, type_id: u64) {
        self.routes.write().remove(&type_id);
    }

    /// Get the kernel ID for a message type.
    pub fn get_route(&self, type_id: u64) -> Option<KernelId> {
        self.routes.read().get(&type_id).cloned()
    }

    /// Check if a route exists for a type.
    pub fn has_route(&self, type_id: u64) -> bool {
        self.routes.read().contains_key(&type_id)
    }

    /// Get all registered routes.
    pub fn routes(&self) -> Vec<(u64, KernelId)> {
        self.routes
            .read()
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    }

    /// Get the dispatch table for a kernel (for CUDA codegen).
    pub fn get_dispatch_table(&self, kernel_id: &KernelId) -> Option<DispatchTable> {
        self.handler_tables.read().get(kernel_id).cloned()
    }

    /// Dispatch a message envelope to the appropriate kernel.
    ///
    /// The type_id from the envelope header is used to look up the destination
    /// kernel. If no route exists for the type_id, returns an error.
    ///
    /// # Returns
    ///
    /// - `Ok(DeliveryReceipt)` with delivery status
    /// - `Err(RingKernelError::UnknownMessageType)` if no route exists
    pub async fn dispatch(&self, envelope: MessageEnvelope) -> Result<DeliveryReceipt> {
        // Use "host" as the default source for dispatched messages
        self.dispatch_from(KernelId::new("host"), envelope).await
    }

    /// Dispatch a message from a specific source kernel.
    pub async fn dispatch_from(
        &self,
        source: KernelId,
        envelope: MessageEnvelope,
    ) -> Result<DeliveryReceipt> {
        let type_id = envelope.header.message_type;

        // Look up the destination kernel
        let kernel_id = {
            let routes = self.routes.read();
            routes.get(&type_id).cloned()
        };

        let kernel_id = match kernel_id {
            Some(id) => id,
            None => {
                // Update metrics
                {
                    let mut metrics = self.metrics.write();
                    metrics.messages_dispatched += 1;
                    metrics.unknown_type_errors += 1;
                }
                return Err(RingKernelError::K2KError(format!(
                    "No route for message type_id: {}",
                    type_id
                )));
            }
        };

        // Dispatch via K2K broker
        let receipt = self
            .broker
            .send_priority(
                source,
                kernel_id,
                envelope,
                self.config.default_priority,
            )
            .await?;

        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.messages_dispatched += 1;
            match receipt.status {
                DeliveryStatus::Delivered => metrics.messages_delivered += 1,
                DeliveryStatus::Pending => {} // Still in flight
                _ => metrics.delivery_errors += 1,
            }
        }

        Ok(receipt)
    }

    /// Dispatch a typed message.
    ///
    /// Creates an envelope from the message and dispatches it.
    pub async fn dispatch_message<M: PersistentMessage>(
        &self,
        message: &M,
        timestamp: HlcTimestamp,
    ) -> Result<DeliveryReceipt> {
        // Use 0 for source/dest kernel IDs - the dispatcher will route based on type_id
        let envelope = MessageEnvelope::new(message, 0, 0, timestamp);
        self.dispatch(envelope).await
    }

    /// Get current metrics.
    pub fn metrics(&self) -> DispatcherMetrics {
        let metrics = self.metrics.read();
        DispatcherMetrics {
            messages_dispatched: metrics.messages_dispatched,
            messages_delivered: metrics.messages_delivered,
            unknown_type_errors: metrics.unknown_type_errors,
            delivery_errors: metrics.delivery_errors,
        }
    }

    /// Reset metrics.
    pub fn reset_metrics(&self) {
        *self.metrics.write() = DispatcherMetrics::default();
    }

    /// Get a reference to the underlying K2K broker.
    pub fn broker(&self) -> &Arc<K2KBroker> {
        &self.broker
    }
}

/// Builder for creating a KernelDispatcher.
pub struct DispatcherBuilder {
    /// Pending routes to register
    routes: Vec<Route>,
    /// Configuration
    config: DispatcherConfig,
    /// K2K configuration
    k2k_config: K2KConfig,
}

/// A route registration.
struct Route {
    /// Message type ID
    type_id: u64,
    /// Target kernel ID
    kernel_id: KernelId,
    /// Handler name
    handler_name: String,
    /// Handler ID (for PersistentMessage types)
    handler_id: Option<u32>,
    /// Whether response is required
    requires_response: bool,
}

impl DispatcherBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            config: DispatcherConfig::default(),
            k2k_config: K2KConfig::default(),
        }
    }

    /// Add a route for a PersistentMessage type.
    pub fn route<M: PersistentMessage>(mut self, kernel_id: KernelId) -> Self {
        self.routes.push(Route {
            type_id: M::message_type(),
            kernel_id,
            handler_name: std::any::type_name::<M>().to_string(),
            handler_id: Some(M::handler_id()),
            requires_response: M::requires_response(),
        });
        self
    }

    /// Add a route with custom handler name.
    pub fn route_named<M: PersistentMessage>(
        mut self,
        kernel_id: KernelId,
        handler_name: &str,
    ) -> Self {
        self.routes.push(Route {
            type_id: M::message_type(),
            kernel_id,
            handler_name: handler_name.to_string(),
            handler_id: Some(M::handler_id()),
            requires_response: M::requires_response(),
        });
        self
    }

    /// Add a raw route (for dynamic type_ids).
    pub fn route_raw(mut self, type_id: u64, kernel_id: KernelId) -> Self {
        self.routes.push(Route {
            type_id,
            kernel_id,
            handler_name: format!("handler_{}", type_id),
            handler_id: None,
            requires_response: false,
        });
        self
    }

    /// Set dispatcher configuration.
    pub fn with_config(mut self, config: DispatcherConfig) -> Self {
        self.config = config;
        self
    }

    /// Set K2K configuration.
    pub fn with_k2k_config(mut self, config: K2KConfig) -> Self {
        self.k2k_config = config;
        self
    }

    /// Enable logging.
    pub fn with_logging(mut self) -> Self {
        self.config.enable_logging = true;
        self
    }

    /// Set default message priority.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.config.default_priority = priority;
        self
    }

    /// Build the dispatcher with a new K2K broker.
    pub fn build(self) -> KernelDispatcher {
        let broker = K2KBroker::new(self.k2k_config.clone());
        self.build_with_broker(broker)
    }

    /// Build the dispatcher with an existing K2K broker.
    pub fn build_with_broker(self, broker: Arc<K2KBroker>) -> KernelDispatcher {
        let dispatcher = KernelDispatcher::with_config(broker, self.config);

        // Register all routes
        for route in self.routes {
            dispatcher
                .routes
                .write()
                .insert(route.type_id, route.kernel_id.clone());

            // Also register in handler tables if we have handler_id
            if let Some(handler_id) = route.handler_id {
                use crate::persistent_message::HandlerRegistration;

                let mut handler_tables = dispatcher.handler_tables.write();
                let table = handler_tables.entry(route.kernel_id).or_default();

                let mut registration =
                    HandlerRegistration::new(handler_id, &route.handler_name, route.type_id);

                if route.requires_response {
                    // Note: We don't have the response type_id here, so we use 0
                    // In practice, the response type would be registered separately
                    registration = registration.with_response(0);
                }

                table.register(registration);
            }
        }

        dispatcher
    }
}

impl Default for DispatcherBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hlc::HlcClock;
    use crate::message::{MessageHeader, RingMessage};

    // Test message type
    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    struct TestRequest {
        value: u64,
    }

    impl RingMessage for TestRequest {
        fn message_type() -> u64 {
            5001
        }

        fn message_id(&self) -> crate::message::MessageId {
            crate::message::MessageId::new(0)
        }

        fn correlation_id(&self) -> crate::message::CorrelationId {
            crate::message::CorrelationId::none()
        }

        fn priority(&self) -> crate::message::Priority {
            crate::message::Priority::Normal
        }

        fn serialize(&self) -> Vec<u8> {
            self.value.to_le_bytes().to_vec()
        }

        fn deserialize(bytes: &[u8]) -> Result<Self> {
            if bytes.len() < 8 {
                return Err(RingKernelError::DeserializationError(
                    "Too small".to_string(),
                ));
            }
            let value = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            Ok(Self { value })
        }

        fn size_hint(&self) -> usize {
            8
        }
    }

    impl PersistentMessage for TestRequest {
        fn handler_id() -> u32 {
            1
        }

        fn requires_response() -> bool {
            true
        }

        fn payload_size() -> usize {
            8
        }

        fn to_inline_payload(
            &self,
        ) -> Option<[u8; crate::persistent_message::MAX_INLINE_PAYLOAD_SIZE]> {
            let mut payload = [0u8; 32];
            payload[..8].copy_from_slice(&self.value.to_le_bytes());
            Some(payload)
        }

        fn from_inline_payload(payload: &[u8]) -> Result<Self> {
            if payload.len() < 8 {
                return Err(RingKernelError::DeserializationError(
                    "Too small".to_string(),
                ));
            }
            let value = u64::from_le_bytes(payload[..8].try_into().unwrap());
            Ok(Self { value })
        }
    }

    #[test]
    fn test_dispatcher_builder() {
        let kernel_id = KernelId::new("test_kernel");

        let dispatcher = DispatcherBuilder::new()
            .route::<TestRequest>(kernel_id.clone())
            .build();

        assert!(dispatcher.has_route(5001));
        assert_eq!(dispatcher.get_route(5001), Some(kernel_id));
    }

    #[test]
    fn test_dispatcher_registration() {
        let dispatcher = DispatcherBuilder::new().build();

        let kernel_id = KernelId::new("processor");
        dispatcher.register::<TestRequest>(kernel_id.clone());

        assert!(dispatcher.has_route(5001));
        assert_eq!(dispatcher.get_route(5001), Some(kernel_id));
    }

    #[test]
    fn test_dispatcher_unregister() {
        let dispatcher = DispatcherBuilder::new()
            .route::<TestRequest>(KernelId::new("processor"))
            .build();

        assert!(dispatcher.has_route(5001));
        dispatcher.unregister(5001);
        assert!(!dispatcher.has_route(5001));
    }

    #[test]
    fn test_dispatcher_routes() {
        let kernel_a = KernelId::new("kernel_a");
        let kernel_b = KernelId::new("kernel_b");

        let dispatcher = DispatcherBuilder::new()
            .route::<TestRequest>(kernel_a.clone())
            .route_raw(9999, kernel_b.clone())
            .build();

        let routes = dispatcher.routes();
        assert_eq!(routes.len(), 2);
        assert!(routes.contains(&(5001, kernel_a)));
        assert!(routes.contains(&(9999, kernel_b)));
    }

    #[test]
    fn test_dispatch_table_generation() {
        let kernel_id = KernelId::new("test_kernel");

        let dispatcher = DispatcherBuilder::new()
            .route::<TestRequest>(kernel_id.clone())
            .build();

        let table = dispatcher.get_dispatch_table(&kernel_id);
        assert!(table.is_some());

        let table = table.unwrap();
        assert_eq!(table.len(), 1);

        let handler = table.get(1).unwrap();
        assert_eq!(handler.handler_id, 1);
        assert_eq!(handler.message_type_id, 5001);
    }

    #[tokio::test]
    async fn test_dispatch_unknown_type() {
        let dispatcher = DispatcherBuilder::new().build();

        let clock = HlcClock::new(1);
        let header = MessageHeader::new(9999, 0, 0, 0, clock.now());
        let envelope = MessageEnvelope {
            header,
            payload: vec![],
        };

        let result = dispatcher.dispatch(envelope).await;
        assert!(result.is_err());

        let metrics = dispatcher.metrics();
        assert_eq!(metrics.messages_dispatched, 1);
        assert_eq!(metrics.unknown_type_errors, 1);
    }

    #[tokio::test]
    async fn test_dispatch_to_registered_kernel() {
        let kernel_id = KernelId::new("test_kernel");

        let broker = K2KBroker::new(K2KConfig::default());
        let _endpoint = broker.register(kernel_id.clone());

        let dispatcher = DispatcherBuilder::new()
            .route::<TestRequest>(kernel_id)
            .build_with_broker(broker);

        let clock = HlcClock::new(1);
        let msg = TestRequest { value: 42 };
        let envelope = MessageEnvelope::new(&msg, 0, 0, clock.now());

        let receipt = dispatcher.dispatch(envelope).await.unwrap();
        assert_eq!(receipt.status, DeliveryStatus::Delivered);

        let metrics = dispatcher.metrics();
        assert_eq!(metrics.messages_dispatched, 1);
        assert_eq!(metrics.messages_delivered, 1);
    }

    #[test]
    fn test_metrics_reset() {
        let dispatcher = DispatcherBuilder::new().build();

        {
            let mut metrics = dispatcher.metrics.write();
            metrics.messages_dispatched = 100;
            metrics.messages_delivered = 50;
        }

        let metrics = dispatcher.metrics();
        assert_eq!(metrics.messages_dispatched, 100);

        dispatcher.reset_metrics();

        let metrics = dispatcher.metrics();
        assert_eq!(metrics.messages_dispatched, 0);
        assert_eq!(metrics.messages_delivered, 0);
    }
}
