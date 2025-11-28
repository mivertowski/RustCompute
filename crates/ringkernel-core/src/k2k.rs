//! Kernel-to-Kernel (K2K) direct messaging.
//!
//! This module provides infrastructure for direct communication between
//! GPU kernels without host-side mediation.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::error::{Result, RingKernelError};
use crate::hlc::HlcTimestamp;
use crate::message::{MessageEnvelope, MessageId};
use crate::runtime::KernelId;

/// Configuration for K2K messaging.
#[derive(Debug, Clone)]
pub struct K2KConfig {
    /// Maximum pending messages per kernel pair.
    pub max_pending_messages: usize,
    /// Timeout for delivery in milliseconds.
    pub delivery_timeout_ms: u64,
    /// Enable message tracing.
    pub enable_tracing: bool,
    /// Maximum hop count for routed messages.
    pub max_hops: u8,
}

impl Default for K2KConfig {
    fn default() -> Self {
        Self {
            max_pending_messages: 1024,
            delivery_timeout_ms: 5000,
            enable_tracing: false,
            max_hops: 8,
        }
    }
}

/// A K2K message with routing information.
#[derive(Debug, Clone)]
pub struct K2KMessage {
    /// Unique message ID.
    pub id: MessageId,
    /// Source kernel.
    pub source: KernelId,
    /// Destination kernel.
    pub destination: KernelId,
    /// The message envelope.
    pub envelope: MessageEnvelope,
    /// Hop count (for detecting routing loops).
    pub hops: u8,
    /// Timestamp when message was sent.
    pub sent_at: HlcTimestamp,
    /// Priority (higher = more urgent).
    pub priority: u8,
}

impl K2KMessage {
    /// Create a new K2K message.
    pub fn new(
        source: KernelId,
        destination: KernelId,
        envelope: MessageEnvelope,
        timestamp: HlcTimestamp,
    ) -> Self {
        Self {
            id: MessageId::generate(),
            source,
            destination,
            envelope,
            hops: 0,
            sent_at: timestamp,
            priority: 0,
        }
    }

    /// Create with priority.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Increment hop count.
    pub fn increment_hops(&mut self) -> Result<()> {
        self.hops += 1;
        if self.hops > 16 {
            return Err(RingKernelError::K2KError(
                "Maximum hop count exceeded".to_string(),
            ));
        }
        Ok(())
    }
}

/// Receipt for a K2K message delivery.
#[derive(Debug, Clone)]
pub struct DeliveryReceipt {
    /// Message ID.
    pub message_id: MessageId,
    /// Source kernel.
    pub source: KernelId,
    /// Destination kernel.
    pub destination: KernelId,
    /// Delivery status.
    pub status: DeliveryStatus,
    /// Timestamp of delivery/failure.
    pub timestamp: HlcTimestamp,
}

/// Status of message delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryStatus {
    /// Message delivered successfully.
    Delivered,
    /// Message pending delivery.
    Pending,
    /// Destination kernel not found.
    NotFound,
    /// Destination queue full.
    QueueFull,
    /// Delivery timed out.
    Timeout,
    /// Maximum hops exceeded.
    MaxHopsExceeded,
}

/// K2K endpoint for a single kernel.
pub struct K2KEndpoint {
    /// Kernel ID.
    kernel_id: KernelId,
    /// Incoming message channel.
    receiver: mpsc::Receiver<K2KMessage>,
    /// Reference to the broker.
    broker: Arc<K2KBroker>,
}

impl K2KEndpoint {
    /// Receive a K2K message (blocking).
    pub async fn receive(&mut self) -> Option<K2KMessage> {
        self.receiver.recv().await
    }

    /// Try to receive a K2K message (non-blocking).
    pub fn try_receive(&mut self) -> Option<K2KMessage> {
        self.receiver.try_recv().ok()
    }

    /// Send a message to another kernel.
    pub async fn send(
        &self,
        destination: KernelId,
        envelope: MessageEnvelope,
    ) -> Result<DeliveryReceipt> {
        self.broker
            .send(self.kernel_id.clone(), destination, envelope)
            .await
    }

    /// Send a high-priority message.
    pub async fn send_priority(
        &self,
        destination: KernelId,
        envelope: MessageEnvelope,
        priority: u8,
    ) -> Result<DeliveryReceipt> {
        self.broker
            .send_priority(self.kernel_id.clone(), destination, envelope, priority)
            .await
    }

    /// Get pending message count.
    pub fn pending_count(&self) -> usize {
        // Note: This is an estimate since the channel may be modified concurrently
        0 // mpsc doesn't provide len() directly
    }
}

/// K2K message broker for routing messages between kernels.
pub struct K2KBroker {
    /// Configuration.
    config: K2KConfig,
    /// Registered endpoints (kernel_id -> sender).
    endpoints: RwLock<HashMap<KernelId, mpsc::Sender<K2KMessage>>>,
    /// Message counter.
    message_counter: AtomicU64,
    /// Delivery receipts (for acknowledgment).
    receipts: RwLock<HashMap<MessageId, DeliveryReceipt>>,
    /// Routing table for indirect delivery.
    routing_table: RwLock<HashMap<KernelId, KernelId>>,
}

impl K2KBroker {
    /// Create a new K2K broker.
    pub fn new(config: K2KConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            endpoints: RwLock::new(HashMap::new()),
            message_counter: AtomicU64::new(0),
            receipts: RwLock::new(HashMap::new()),
            routing_table: RwLock::new(HashMap::new()),
        })
    }

    /// Register a kernel endpoint.
    pub fn register(self: &Arc<Self>, kernel_id: KernelId) -> K2KEndpoint {
        let (sender, receiver) = mpsc::channel(self.config.max_pending_messages);

        self.endpoints.write().insert(kernel_id.clone(), sender);

        K2KEndpoint {
            kernel_id,
            receiver,
            broker: Arc::clone(self),
        }
    }

    /// Unregister a kernel endpoint.
    pub fn unregister(&self, kernel_id: &KernelId) {
        self.endpoints.write().remove(kernel_id);
        self.routing_table.write().remove(kernel_id);
    }

    /// Check if a kernel is registered.
    pub fn is_registered(&self, kernel_id: &KernelId) -> bool {
        self.endpoints.read().contains_key(kernel_id)
    }

    /// Get all registered kernels.
    pub fn registered_kernels(&self) -> Vec<KernelId> {
        self.endpoints.read().keys().cloned().collect()
    }

    /// Send a message from one kernel to another.
    pub async fn send(
        &self,
        source: KernelId,
        destination: KernelId,
        envelope: MessageEnvelope,
    ) -> Result<DeliveryReceipt> {
        self.send_priority(source, destination, envelope, 0).await
    }

    /// Send a priority message.
    pub async fn send_priority(
        &self,
        source: KernelId,
        destination: KernelId,
        envelope: MessageEnvelope,
        priority: u8,
    ) -> Result<DeliveryReceipt> {
        let timestamp = envelope.header.timestamp;
        let mut message = K2KMessage::new(source.clone(), destination.clone(), envelope, timestamp);
        message.priority = priority;

        self.deliver(message).await
    }

    /// Deliver a message to its destination.
    async fn deliver(&self, message: K2KMessage) -> Result<DeliveryReceipt> {
        let message_id = message.id;
        let source = message.source.clone();
        let destination = message.destination.clone();
        let timestamp = message.sent_at;

        // Try direct delivery first
        let endpoints = self.endpoints.read();
        if let Some(sender) = endpoints.get(&destination) {
            match sender.try_send(message) {
                Ok(()) => {
                    self.message_counter.fetch_add(1, Ordering::Relaxed);
                    let receipt = DeliveryReceipt {
                        message_id,
                        source,
                        destination,
                        status: DeliveryStatus::Delivered,
                        timestamp,
                    };
                    self.receipts.write().insert(message_id, receipt.clone());
                    return Ok(receipt);
                }
                Err(mpsc::error::TrySendError::Full(_)) => {
                    return Ok(DeliveryReceipt {
                        message_id,
                        source,
                        destination,
                        status: DeliveryStatus::QueueFull,
                        timestamp,
                    });
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    return Ok(DeliveryReceipt {
                        message_id,
                        source,
                        destination,
                        status: DeliveryStatus::NotFound,
                        timestamp,
                    });
                }
            }
        }
        drop(endpoints);

        // Try routing table
        let next_hop = {
            let routing = self.routing_table.read();
            routing.get(&destination).cloned()
        };

        if let Some(next_hop) = next_hop {
            let routed_message = K2KMessage {
                id: message_id,
                source,
                destination: destination.clone(),
                envelope: message.envelope,
                hops: message.hops + 1,
                sent_at: message.sent_at,
                priority: message.priority,
            };

            if routed_message.hops > self.config.max_hops {
                return Ok(DeliveryReceipt {
                    message_id,
                    source: routed_message.source,
                    destination,
                    status: DeliveryStatus::MaxHopsExceeded,
                    timestamp,
                });
            }

            // Try to deliver to next hop
            let endpoints = self.endpoints.read();
            if let Some(sender) = endpoints.get(&next_hop) {
                if sender.try_send(routed_message).is_ok() {
                    self.message_counter.fetch_add(1, Ordering::Relaxed);
                    return Ok(DeliveryReceipt {
                        message_id,
                        source: message.source,
                        destination,
                        status: DeliveryStatus::Pending,
                        timestamp,
                    });
                }
            }
        }

        // Destination not found
        Ok(DeliveryReceipt {
            message_id,
            source: message.source,
            destination,
            status: DeliveryStatus::NotFound,
            timestamp,
        })
    }

    /// Add a route to the routing table.
    pub fn add_route(&self, destination: KernelId, next_hop: KernelId) {
        self.routing_table.write().insert(destination, next_hop);
    }

    /// Remove a route from the routing table.
    pub fn remove_route(&self, destination: &KernelId) {
        self.routing_table.write().remove(destination);
    }

    /// Get statistics.
    pub fn stats(&self) -> K2KStats {
        K2KStats {
            registered_endpoints: self.endpoints.read().len(),
            messages_delivered: self.message_counter.load(Ordering::Relaxed),
            routes_configured: self.routing_table.read().len(),
        }
    }

    /// Get delivery receipt for a message.
    pub fn get_receipt(&self, message_id: &MessageId) -> Option<DeliveryReceipt> {
        self.receipts.read().get(message_id).cloned()
    }
}

/// K2K messaging statistics.
#[derive(Debug, Clone, Default)]
pub struct K2KStats {
    /// Number of registered endpoints.
    pub registered_endpoints: usize,
    /// Total messages delivered.
    pub messages_delivered: u64,
    /// Number of routes configured.
    pub routes_configured: usize,
}

/// Builder for creating K2K infrastructure.
pub struct K2KBuilder {
    config: K2KConfig,
}

impl K2KBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: K2KConfig::default(),
        }
    }

    /// Set maximum pending messages.
    pub fn max_pending_messages(mut self, count: usize) -> Self {
        self.config.max_pending_messages = count;
        self
    }

    /// Set delivery timeout.
    pub fn delivery_timeout_ms(mut self, timeout: u64) -> Self {
        self.config.delivery_timeout_ms = timeout;
        self
    }

    /// Enable message tracing.
    pub fn enable_tracing(mut self, enable: bool) -> Self {
        self.config.enable_tracing = enable;
        self
    }

    /// Set maximum hop count.
    pub fn max_hops(mut self, hops: u8) -> Self {
        self.config.max_hops = hops;
        self
    }

    /// Build the K2K broker.
    pub fn build(self) -> Arc<K2KBroker> {
        K2KBroker::new(self.config)
    }
}

impl Default for K2KBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_k2k_broker_registration() {
        let broker = K2KBuilder::new().build();

        let kernel1 = KernelId::new("kernel1");
        let kernel2 = KernelId::new("kernel2");

        let _endpoint1 = broker.register(kernel1.clone());
        let _endpoint2 = broker.register(kernel2.clone());

        assert!(broker.is_registered(&kernel1));
        assert!(broker.is_registered(&kernel2));
        assert_eq!(broker.registered_kernels().len(), 2);
    }

    #[tokio::test]
    async fn test_k2k_message_delivery() {
        let broker = K2KBuilder::new().build();

        let kernel1 = KernelId::new("kernel1");
        let kernel2 = KernelId::new("kernel2");

        let endpoint1 = broker.register(kernel1.clone());
        let mut endpoint2 = broker.register(kernel2.clone());

        // Create a test envelope
        let envelope = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));

        // Send from kernel1 to kernel2
        let receipt = endpoint1.send(kernel2.clone(), envelope).await.unwrap();
        assert_eq!(receipt.status, DeliveryStatus::Delivered);

        // Receive on kernel2
        let message = endpoint2.try_receive();
        assert!(message.is_some());
        assert_eq!(message.unwrap().source, kernel1);
    }

    #[test]
    fn test_k2k_config_default() {
        let config = K2KConfig::default();
        assert_eq!(config.max_pending_messages, 1024);
        assert_eq!(config.delivery_timeout_ms, 5000);
    }
}
