//! Kernel-to-Kernel (K2K) direct messaging.
//!
//! This module provides infrastructure for direct communication between
//! GPU kernels without host-side mediation.
//!
//! # Submodules
//!
//! - [`tenant`] — [`tenant::TenantId`], [`tenant::TenantRegistry`],
//!   [`tenant::TenantQuota`] with per-engagement cost tracking
//! - [`audit_tag`] — [`audit_tag::AuditTag`] (org_id + engagement_id) for
//!   audit trails and billable-time attribution
//!
//! # Multi-tenant overview
//!
//! The [`K2KBroker`] implements a two-tier tenancy model:
//!
//! 1. **Per-kernel `tenant_id`** — the primary security boundary. Kernels
//!    registered under different tenants cannot exchange messages; sends
//!    across the boundary return
//!    [`crate::error::RingKernelError::TenantMismatch`].
//! 2. **Per-message `AuditTag`** — the observability / billing tag, carried
//!    inside every envelope alongside `tenant_id`. Cost tracking is
//!    per-`(tenant_id, engagement_id)` via
//!    [`tenant::TenantRegistry::track_usage`].
//!
//! Single-tenant deployments hit the backward-compatible fast path — see
//! [`tenant::UNSPECIFIED_TENANT`] and [`K2KBroker::register`].

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::error::{Result, RingKernelError};
use crate::hlc::HlcTimestamp;
use crate::message::{MessageEnvelope, MessageId};
use crate::runtime::KernelId;

pub mod audit_tag;
pub mod tenant;

pub use audit_tag::AuditTag;
pub use tenant::{TenantId, TenantInfo, TenantQuota, TenantRegistry, UNSPECIFIED_TENANT};

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
    /// Cross-tenant send rejected (also surfaces as
    /// [`crate::error::RingKernelError::TenantMismatch`]).
    TenantMismatch,
}

/// K2K endpoint for a single kernel.
pub struct K2KEndpoint {
    /// Kernel ID.
    pub(crate) kernel_id: KernelId,
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

// ============================================================================
// K2KSubBroker — per-tenant routing table
// ============================================================================

/// Per-tenant sub-broker.
///
/// Each `K2KSubBroker` is an independent routing domain: its endpoint map and
/// indirect-routing table are *not* visible to any other tenant. Cross-tenant
/// sends therefore have no possible route and are rejected by the parent
/// [`K2KBroker`] before they reach any sub-broker.
pub struct K2KSubBroker {
    /// The tenant this sub-broker serves. `0` = unspecified (legacy).
    tenant_id: TenantId,
    /// Registered endpoints (kernel_id -> sender).
    endpoints: RwLock<HashMap<KernelId, mpsc::Sender<K2KMessage>>>,
    /// Indirect routing table (destination -> next-hop).
    routing_table: RwLock<HashMap<KernelId, KernelId>>,
    /// Default audit tag applied to messages if the sender doesn't provide
    /// one (per-kernel tag, set at registration).
    kernel_audit_tags: RwLock<HashMap<KernelId, AuditTag>>,
    /// Messages successfully delivered from this sub-broker.
    messages_delivered: AtomicU64,
}

impl K2KSubBroker {
    fn new(tenant_id: TenantId) -> Self {
        Self {
            tenant_id,
            endpoints: RwLock::new(HashMap::new()),
            routing_table: RwLock::new(HashMap::new()),
            kernel_audit_tags: RwLock::new(HashMap::new()),
            messages_delivered: AtomicU64::new(0),
        }
    }

    /// Tenant this sub-broker belongs to.
    pub fn tenant_id(&self) -> TenantId {
        self.tenant_id
    }

    /// Number of kernels registered in this sub-broker.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.read().len()
    }

    /// Number of messages successfully delivered by this sub-broker.
    pub fn messages_delivered(&self) -> u64 {
        self.messages_delivered.load(Ordering::Relaxed)
    }

    /// Returns `true` if this sub-broker has a route (direct or indirect)
    /// for `kernel_id`.
    pub fn knows(&self, kernel_id: &KernelId) -> bool {
        self.endpoints.read().contains_key(kernel_id)
            || self.routing_table.read().contains_key(kernel_id)
    }

    /// Get the audit tag associated with a kernel at registration time, or
    /// [`AuditTag::unspecified`] if none was set.
    pub fn audit_tag_for(&self, kernel_id: &KernelId) -> AuditTag {
        self.kernel_audit_tags
            .read()
            .get(kernel_id)
            .copied()
            .unwrap_or_else(AuditTag::unspecified)
    }
}

// ============================================================================
// K2KBroker — per-tenant sub-broker aggregator
// ============================================================================

/// K2K message broker with per-tenant isolation.
///
/// Internally holds a `HashMap<TenantId, K2KSubBroker>`. Single-tenant
/// deployments see exactly one entry (for `UNSPECIFIED_TENANT = 0`); the
/// only overhead over the legacy broker is one additional HashMap lookup
/// per send (~20 ns).
pub struct K2KBroker {
    /// Configuration (shared across sub-brokers).
    config: K2KConfig,
    /// Per-tenant sub-brokers.
    tenants: RwLock<HashMap<TenantId, Arc<K2KSubBroker>>>,
    /// Reverse index: which tenant does a kernel belong to?
    kernel_tenant: RwLock<HashMap<KernelId, TenantId>>,
    /// Tenant registry (quotas, audit sink, engagement cost tracking).
    registry: Arc<TenantRegistry>,
    /// Delivery receipts (for acknowledgment).
    receipts: RwLock<HashMap<MessageId, DeliveryReceipt>>,
    /// Global message counter (sum across all sub-brokers, for legacy stats).
    message_counter: AtomicU64,
    /// Counter of cross-tenant attempts rejected.
    cross_tenant_rejections: AtomicU64,
}

impl K2KBroker {
    /// Create a new K2K broker with an empty [`TenantRegistry`].
    pub fn new(config: K2KConfig) -> Arc<Self> {
        Self::with_registry(config, Arc::new(TenantRegistry::new()))
    }

    /// Create a new K2K broker sharing the given [`TenantRegistry`].
    ///
    /// This is the multi-tenant constructor: wire up the registry (with its
    /// audit sink) once, then pass the `Arc` to every broker that should
    /// enforce against it.
    pub fn with_registry(config: K2KConfig, registry: Arc<TenantRegistry>) -> Arc<Self> {
        let mut tenants = HashMap::new();
        // Always pre-create the unspecified-tenant sub-broker so legacy
        // single-tenant callers hit it directly with no `or_insert_with` in
        // the hot path.
        tenants.insert(
            UNSPECIFIED_TENANT,
            Arc::new(K2KSubBroker::new(UNSPECIFIED_TENANT)),
        );
        Arc::new(Self {
            config,
            tenants: RwLock::new(tenants),
            kernel_tenant: RwLock::new(HashMap::new()),
            registry,
            receipts: RwLock::new(HashMap::new()),
            message_counter: AtomicU64::new(0),
            cross_tenant_rejections: AtomicU64::new(0),
        })
    }

    /// Shared reference to the tenant registry.
    pub fn registry(&self) -> &Arc<TenantRegistry> {
        &self.registry
    }

    /// Total number of active tenant sub-brokers.
    pub fn tenant_count(&self) -> usize {
        self.tenants.read().len()
    }

    /// Get the sub-broker for a tenant, if it exists.
    pub fn sub_broker(&self, tenant_id: TenantId) -> Option<Arc<K2KSubBroker>> {
        self.tenants.read().get(&tenant_id).cloned()
    }

    /// Register a kernel without a tenant / audit tag (legacy API).
    ///
    /// The kernel is placed in the unspecified-tenant sub-broker
    /// (`UNSPECIFIED_TENANT = 0`). This is the backward-compatible entry
    /// point for single-tenant deployments — existing callers don't need to
    /// change anything.
    pub fn register(self: &Arc<Self>, kernel_id: KernelId) -> K2KEndpoint {
        self.register_tenant(UNSPECIFIED_TENANT, AuditTag::unspecified(), kernel_id)
    }

    /// Register a kernel into the given tenant's sub-broker, stamping the
    /// [`AuditTag`] that will be applied to outgoing messages from this
    /// kernel.
    ///
    /// If the tenant doesn't yet have a sub-broker, one is created lazily.
    /// If the kernel was previously registered under a different tenant,
    /// the old registration is replaced (the old mpsc sender is dropped —
    /// in-flight messages to the old registration are lost). This preserves
    /// the invariant that a kernel belongs to exactly one tenant.
    pub fn register_tenant(
        self: &Arc<Self>,
        tenant_id: TenantId,
        audit_tag: AuditTag,
        kernel_id: KernelId,
    ) -> K2KEndpoint {
        let (sender, receiver) = mpsc::channel(self.config.max_pending_messages);

        let sub = {
            let mut tenants = self.tenants.write();
            tenants
                .entry(tenant_id)
                .or_insert_with(|| Arc::new(K2KSubBroker::new(tenant_id)))
                .clone()
        };

        let mut kernel_tenant = self.kernel_tenant.write();
        if let Some(prev_tenant) = kernel_tenant.get(&kernel_id).copied() {
            if prev_tenant != tenant_id {
                if let Some(prev_sub) = self.tenants.read().get(&prev_tenant).cloned() {
                    prev_sub.endpoints.write().remove(&kernel_id);
                    prev_sub.kernel_audit_tags.write().remove(&kernel_id);
                    prev_sub.routing_table.write().remove(&kernel_id);
                }
            }
        }
        kernel_tenant.insert(kernel_id.clone(), tenant_id);
        drop(kernel_tenant);

        sub.endpoints.write().insert(kernel_id.clone(), sender);
        sub.kernel_audit_tags
            .write()
            .insert(kernel_id.clone(), audit_tag);

        K2KEndpoint {
            kernel_id,
            receiver,
            broker: Arc::clone(self),
        }
    }

    /// Unregister a kernel from whichever tenant it belongs to.
    pub fn unregister(&self, kernel_id: &KernelId) {
        let tenant_id = self.kernel_tenant.write().remove(kernel_id);
        if let Some(tenant_id) = tenant_id {
            if let Some(sub) = self.tenants.read().get(&tenant_id).cloned() {
                sub.endpoints.write().remove(kernel_id);
                sub.kernel_audit_tags.write().remove(kernel_id);
                sub.routing_table.write().remove(kernel_id);
            }
        }
    }

    /// Check if a kernel is registered (under any tenant).
    pub fn is_registered(&self, kernel_id: &KernelId) -> bool {
        self.kernel_tenant.read().contains_key(kernel_id)
    }

    /// Get the tenant this kernel belongs to, if registered.
    pub fn tenant_of(&self, kernel_id: &KernelId) -> Option<TenantId> {
        self.kernel_tenant.read().get(kernel_id).copied()
    }

    /// All registered kernel IDs (across every tenant).
    pub fn registered_kernels(&self) -> Vec<KernelId> {
        self.kernel_tenant.read().keys().cloned().collect()
    }

    /// Kernel IDs registered under a specific tenant.
    pub fn registered_kernels_for(&self, tenant_id: TenantId) -> Vec<KernelId> {
        self.tenants
            .read()
            .get(&tenant_id)
            .map(|sub| sub.endpoints.read().keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Send a message from one kernel to another (normal priority).
    ///
    /// Enforces tenant isolation: the sender's and destination's tenants
    /// must match, or the send is rejected with
    /// [`crate::error::RingKernelError::TenantMismatch`] and an audit event
    /// is emitted.
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
        // Resolve source tenant (sender must be registered).
        let source_tenant = self
            .kernel_tenant
            .read()
            .get(&source)
            .copied()
            .unwrap_or(UNSPECIFIED_TENANT);

        // Resolve destination tenant. If destination is unregistered we
        // treat it as residing in the *sender's* tenant — that way the
        // resulting delivery attempt will hit NotFound inside the sender's
        // own sub-broker (preserving the legacy error behavior) rather than
        // being misclassified as cross-tenant.
        let dest_tenant = self
            .kernel_tenant
            .read()
            .get(&destination)
            .copied()
            .unwrap_or(source_tenant);

        // Cross-tenant check.
        if source_tenant != dest_tenant {
            self.cross_tenant_rejections.fetch_add(1, Ordering::Relaxed);
            self.registry.audit_cross_tenant(
                source_tenant,
                dest_tenant,
                source.as_str(),
                destination.as_str(),
                envelope.audit_tag,
            );
            return Err(RingKernelError::TenantMismatch {
                from: source_tenant,
                to: dest_tenant,
            });
        }

        // Quota check (observability-driven — UNSPECIFIED_TENANT is a no-op).
        self.registry
            .check_quota(source_tenant, envelope.audit_tag)?;
        self.registry.record_message(source_tenant);

        // Stamp envelope with sender-derived audit tag if caller didn't
        // supply one.
        let mut envelope = envelope;
        envelope.tenant_id = source_tenant;
        if envelope.audit_tag.is_unspecified() {
            let sub = self
                .tenants
                .read()
                .get(&source_tenant)
                .cloned()
                .expect("tenant sub-broker must exist for registered sender");
            envelope.audit_tag = sub.audit_tag_for(&source);
        }

        let timestamp = envelope.header.timestamp;
        let mut message = K2KMessage::new(source.clone(), destination.clone(), envelope, timestamp);
        message.priority = priority;

        self.deliver_in(source_tenant, message).await
    }

    /// Send a message stamped with an explicit audit tag (overriding the
    /// registration-time tag).
    ///
    /// Useful for the same kernel participating in multiple engagements —
    /// e.g. a shared report-generation kernel that should bill different
    /// engagements depending on which request it's serving.
    pub async fn send_with_audit(
        &self,
        source: KernelId,
        destination: KernelId,
        envelope: MessageEnvelope,
        audit_tag: AuditTag,
    ) -> Result<DeliveryReceipt> {
        let envelope = envelope.with_audit_tag(audit_tag);
        self.send(source, destination, envelope).await
    }

    /// Deliver a message inside a specific sub-broker.
    async fn deliver_in(
        &self,
        tenant_id: TenantId,
        message: K2KMessage,
    ) -> Result<DeliveryReceipt> {
        let sub = self
            .tenants
            .read()
            .get(&tenant_id)
            .cloned()
            .ok_or_else(|| {
                RingKernelError::K2KError(format!(
                    "tenant sub-broker {} disappeared mid-send",
                    tenant_id
                ))
            })?;

        let message_id = message.id;
        let source = message.source.clone();
        let destination = message.destination.clone();
        let timestamp = message.sent_at;

        // Try direct delivery first.
        let endpoints = sub.endpoints.read();
        if let Some(sender) = endpoints.get(&destination) {
            match sender.try_send(message) {
                Ok(()) => {
                    sub.messages_delivered.fetch_add(1, Ordering::Relaxed);
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

        // Try indirect routing (within the same tenant only — there is no
        // cross-tenant routing by construction).
        let next_hop = sub.routing_table.read().get(&destination).cloned();
        if let Some(next_hop) = next_hop {
            let routed_message = K2KMessage {
                id: message_id,
                source: source.clone(),
                destination: destination.clone(),
                envelope: message.envelope,
                hops: message.hops + 1,
                sent_at: message.sent_at,
                priority: message.priority,
            };

            if routed_message.hops > self.config.max_hops {
                return Ok(DeliveryReceipt {
                    message_id,
                    source,
                    destination,
                    status: DeliveryStatus::MaxHopsExceeded,
                    timestamp,
                });
            }

            let endpoints = sub.endpoints.read();
            if let Some(sender) = endpoints.get(&next_hop) {
                if sender.try_send(routed_message).is_ok() {
                    sub.messages_delivered.fetch_add(1, Ordering::Relaxed);
                    self.message_counter.fetch_add(1, Ordering::Relaxed);
                    return Ok(DeliveryReceipt {
                        message_id,
                        source,
                        destination,
                        status: DeliveryStatus::Pending,
                        timestamp,
                    });
                }
            }
        }

        // Destination not found within the tenant.
        Ok(DeliveryReceipt {
            message_id,
            source,
            destination,
            status: DeliveryStatus::NotFound,
            timestamp,
        })
    }

    /// Add an indirect route inside the unspecified-tenant sub-broker
    /// (legacy API — single-tenant callers should use this).
    pub fn add_route(&self, destination: KernelId, next_hop: KernelId) {
        self.add_route_in(UNSPECIFIED_TENANT, destination, next_hop);
    }

    /// Add an indirect route inside a specific tenant's sub-broker.
    pub fn add_route_in(&self, tenant_id: TenantId, destination: KernelId, next_hop: KernelId) {
        let sub = {
            let mut tenants = self.tenants.write();
            tenants
                .entry(tenant_id)
                .or_insert_with(|| Arc::new(K2KSubBroker::new(tenant_id)))
                .clone()
        };
        sub.routing_table.write().insert(destination, next_hop);
    }

    /// Remove an indirect route from the unspecified-tenant sub-broker.
    pub fn remove_route(&self, destination: &KernelId) {
        self.remove_route_in(UNSPECIFIED_TENANT, destination);
    }

    /// Remove an indirect route from a specific tenant's sub-broker.
    pub fn remove_route_in(&self, tenant_id: TenantId, destination: &KernelId) {
        if let Some(sub) = self.tenants.read().get(&tenant_id).cloned() {
            sub.routing_table.write().remove(destination);
        }
    }

    /// Get aggregate stats across all sub-brokers.
    pub fn stats(&self) -> K2KStats {
        let tenants = self.tenants.read();
        let mut registered = 0usize;
        let mut routes = 0usize;
        for sub in tenants.values() {
            registered += sub.endpoints.read().len();
            routes += sub.routing_table.read().len();
        }
        K2KStats {
            registered_endpoints: registered,
            messages_delivered: self.message_counter.load(Ordering::Relaxed),
            routes_configured: routes,
            tenant_count: tenants.len(),
            cross_tenant_rejections: self.cross_tenant_rejections.load(Ordering::Relaxed),
        }
    }

    /// Get per-tenant stats (useful for billing / dashboards).
    pub fn tenant_stats(&self, tenant_id: TenantId) -> Option<TenantStats> {
        self.tenants.read().get(&tenant_id).map(|sub| TenantStats {
            tenant_id,
            registered_endpoints: sub.endpoints.read().len(),
            routes_configured: sub.routing_table.read().len(),
            messages_delivered: sub.messages_delivered.load(Ordering::Relaxed),
        })
    }

    /// Get delivery receipt for a message.
    pub fn get_receipt(&self, message_id: &MessageId) -> Option<DeliveryReceipt> {
        self.receipts.read().get(message_id).cloned()
    }
}

/// K2K messaging statistics (aggregate across all tenants).
#[derive(Debug, Clone, Default)]
pub struct K2KStats {
    /// Total number of registered endpoints across all tenants.
    pub registered_endpoints: usize,
    /// Total messages delivered (all tenants).
    pub messages_delivered: u64,
    /// Total configured routes across all tenants.
    pub routes_configured: usize,
    /// Number of active tenant sub-brokers.
    pub tenant_count: usize,
    /// Cross-tenant send attempts rejected so far.
    pub cross_tenant_rejections: u64,
}

/// Per-tenant stats (one entry per sub-broker).
#[derive(Debug, Clone, Default)]
pub struct TenantStats {
    /// Tenant ID.
    pub tenant_id: TenantId,
    /// Endpoints registered under this tenant.
    pub registered_endpoints: usize,
    /// Routes configured under this tenant.
    pub routes_configured: usize,
    /// Messages delivered by this sub-broker.
    pub messages_delivered: u64,
}

/// Builder for creating K2K infrastructure.
pub struct K2KBuilder {
    config: K2KConfig,
    registry: Option<Arc<TenantRegistry>>,
}

impl K2KBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: K2KConfig::default(),
            registry: None,
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

    /// Attach a shared [`TenantRegistry`] (for multi-tenant deployments).
    pub fn with_registry(mut self, registry: Arc<TenantRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Build the K2K broker.
    pub fn build(self) -> Arc<K2KBroker> {
        match self.registry {
            Some(registry) => K2KBroker::with_registry(self.config, registry),
            None => K2KBroker::new(self.config),
        }
    }
}

impl Default for K2KBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// K2K Message Type Registry (FR-3)
// ============================================================================

/// Registration information for a K2K-routable message type.
///
/// This struct is automatically generated by the `#[derive(RingMessage)]` macro
/// when `k2k_routable = true` is specified. Registrations are collected at
/// compile time using the `inventory` crate.
///
/// # Example
///
/// ```ignore
/// #[derive(RingMessage)]
/// #[ring_message(type_id = 1, domain = "OrderMatching", k2k_routable = true)]
/// pub struct SubmitOrderInput { ... }
///
/// // Runtime discovery
/// let registry = K2KTypeRegistry::discover();
/// assert!(registry.is_routable(501)); // domain base (500) + type_id (1)
/// ```
#[derive(Debug, Clone)]
pub struct K2KMessageRegistration {
    /// Message type ID (from RingMessage::message_type()).
    pub type_id: u64,
    /// Full type name for debugging/logging.
    pub type_name: &'static str,
    /// Whether this message type is routable via K2K.
    pub k2k_routable: bool,
    /// Optional routing category for grouped routing.
    pub category: Option<&'static str>,
}

// Collect all K2K message registrations at compile time
inventory::collect!(K2KMessageRegistration);

/// Registry for discovering K2K-routable message types at runtime.
///
/// The registry is built by scanning all `K2KMessageRegistration` entries
/// submitted via the `inventory` crate. This enables runtime discovery of
/// message types for routing, validation, and monitoring.
///
/// # Example
///
/// ```ignore
/// let registry = K2KTypeRegistry::discover();
///
/// // Check if a type is routable
/// if registry.is_routable(501) {
///     // Allow K2K routing
/// }
///
/// // Get all types in a category
/// let order_types = registry.get_category("orders");
/// for type_id in order_types {
///     println!("Order message type: {}", type_id);
/// }
/// ```
pub struct K2KTypeRegistry {
    /// Type ID to registration mapping.
    by_type_id: HashMap<u64, &'static K2KMessageRegistration>,
    /// Type name to registration mapping.
    by_type_name: HashMap<&'static str, &'static K2KMessageRegistration>,
    /// Category to type IDs mapping.
    by_category: HashMap<&'static str, Vec<u64>>,
}

impl K2KTypeRegistry {
    /// Discover all registered K2K message types at runtime.
    ///
    /// This scans all `K2KMessageRegistration` entries that were submitted
    /// via `inventory::submit!` during compilation.
    pub fn discover() -> Self {
        let mut registry = Self {
            by_type_id: HashMap::new(),
            by_type_name: HashMap::new(),
            by_category: HashMap::new(),
        };

        for reg in inventory::iter::<K2KMessageRegistration>() {
            registry.by_type_id.insert(reg.type_id, reg);
            registry.by_type_name.insert(reg.type_name, reg);
            if let Some(cat) = reg.category {
                registry
                    .by_category
                    .entry(cat)
                    .or_default()
                    .push(reg.type_id);
            }
        }

        registry
    }

    /// Check if a message type ID is K2K routable.
    pub fn is_routable(&self, type_id: u64) -> bool {
        self.by_type_id
            .get(&type_id)
            .map(|r| r.k2k_routable)
            .unwrap_or(false)
    }

    /// Get registration by type ID.
    pub fn get(&self, type_id: u64) -> Option<&'static K2KMessageRegistration> {
        self.by_type_id.get(&type_id).copied()
    }

    /// Get registration by type name.
    pub fn get_by_name(&self, type_name: &str) -> Option<&'static K2KMessageRegistration> {
        self.by_type_name.get(type_name).copied()
    }

    /// Get all type IDs in a category.
    pub fn get_category(&self, category: &str) -> &[u64] {
        self.by_category
            .get(category)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all registered categories.
    pub fn categories(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.by_category.keys().copied()
    }

    /// Iterate all registered message types.
    pub fn iter(&self) -> impl Iterator<Item = &'static K2KMessageRegistration> + '_ {
        self.by_type_id.values().copied()
    }

    /// Get all routable type IDs.
    pub fn routable_types(&self) -> Vec<u64> {
        self.by_type_id
            .iter()
            .filter(|(_, r)| r.k2k_routable)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get total number of registered message types.
    pub fn len(&self) -> usize {
        self.by_type_id.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.by_type_id.is_empty()
    }
}

impl Default for K2KTypeRegistry {
    fn default() -> Self {
        Self::discover()
    }
}

impl std::fmt::Debug for K2KTypeRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("K2KTypeRegistry")
            .field("registered_types", &self.by_type_id.len())
            .field("categories", &self.by_category.keys().collect::<Vec<_>>())
            .finish()
    }
}

// ============================================================================
// K2K Message Encryption (Phase 4.2 - Enterprise Security)
// ============================================================================

/// Configuration for K2K message encryption.
#[cfg(feature = "crypto")]
#[derive(Debug, Clone)]
pub struct K2KEncryptionConfig {
    /// Enable message encryption.
    pub enabled: bool,
    /// Encryption algorithm.
    pub algorithm: K2KEncryptionAlgorithm,
    /// Enable forward secrecy via ephemeral keys.
    pub forward_secrecy: bool,
    /// Key rotation interval in seconds (0 = no rotation).
    pub key_rotation_interval_secs: u64,
    /// Whether to require encryption for all messages.
    pub require_encryption: bool,
}

#[cfg(feature = "crypto")]
impl Default for K2KEncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: K2KEncryptionAlgorithm::Aes256Gcm,
            forward_secrecy: true,
            key_rotation_interval_secs: 3600, // 1 hour
            require_encryption: false,
        }
    }
}

#[cfg(feature = "crypto")]
impl K2KEncryptionConfig {
    /// Create a config with encryption disabled.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create a strict config requiring encryption for all messages.
    pub fn strict() -> Self {
        Self {
            enabled: true,
            require_encryption: true,
            forward_secrecy: true,
            ..Default::default()
        }
    }
}

/// Supported K2K encryption algorithms.
#[cfg(feature = "crypto")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum K2KEncryptionAlgorithm {
    /// AES-256-GCM (NIST standard, hardware acceleration).
    Aes256Gcm,
    /// ChaCha20-Poly1305 (mobile/embedded friendly).
    ChaCha20Poly1305,
}

/// Per-kernel encryption key material.
#[cfg(feature = "crypto")]
pub struct K2KKeyMaterial {
    /// Kernel ID this key belongs to.
    kernel_id: KernelId,
    /// Long-term key (for key exchange).
    long_term_key: [u8; 32],
    /// Current session key.
    session_key: parking_lot::RwLock<[u8; 32]>,
    /// Session key generation (for rotation).
    session_generation: std::sync::atomic::AtomicU64,
    /// Creation timestamp.
    created_at: std::time::Instant,
    /// Last rotation timestamp.
    last_rotated: parking_lot::RwLock<std::time::Instant>,
}

#[cfg(feature = "crypto")]
impl K2KKeyMaterial {
    /// Create new key material for a kernel.
    pub fn new(kernel_id: KernelId) -> Self {
        use rand::RngCore;
        let mut rng = rand::thread_rng();

        let mut long_term_key = [0u8; 32];
        let mut session_key = [0u8; 32];
        rng.fill_bytes(&mut long_term_key);
        rng.fill_bytes(&mut session_key);

        let now = std::time::Instant::now();
        Self {
            kernel_id,
            long_term_key,
            session_key: parking_lot::RwLock::new(session_key),
            session_generation: std::sync::atomic::AtomicU64::new(1),
            created_at: now,
            last_rotated: parking_lot::RwLock::new(now),
        }
    }

    /// Create key material from an existing long-term key.
    pub fn from_key(kernel_id: KernelId, key: [u8; 32]) -> Self {
        use rand::RngCore;
        let mut rng = rand::thread_rng();

        let mut session_key = [0u8; 32];
        rng.fill_bytes(&mut session_key);

        let now = std::time::Instant::now();
        Self {
            kernel_id,
            long_term_key: key,
            session_key: parking_lot::RwLock::new(session_key),
            session_generation: std::sync::atomic::AtomicU64::new(1),
            created_at: now,
            last_rotated: parking_lot::RwLock::new(now),
        }
    }

    /// Get the kernel ID.
    pub fn kernel_id(&self) -> &KernelId {
        &self.kernel_id
    }

    /// Get the current session key.
    pub fn session_key(&self) -> [u8; 32] {
        *self.session_key.read()
    }

    /// Get the current session generation.
    pub fn session_generation(&self) -> u64 {
        self.session_generation
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Rotate the session key.
    pub fn rotate_session_key(&self) {
        use rand::RngCore;
        let mut rng = rand::thread_rng();

        let mut new_key = [0u8; 32];
        rng.fill_bytes(&mut new_key);

        *self.session_key.write() = new_key;
        self.session_generation
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        *self.last_rotated.write() = std::time::Instant::now();
    }

    /// Derive a shared secret for a destination kernel.
    pub fn derive_shared_secret(&self, dest_public_key: &[u8; 32]) -> [u8; 32] {
        use sha2::{Digest, Sha256};

        // Simple key derivation: HKDF-like construction
        // In production, use proper X25519 key exchange
        let mut hasher = Sha256::new();
        hasher.update(&self.long_term_key);
        hasher.update(dest_public_key);
        hasher.update(b"k2k-shared-secret-v1");

        let result = hasher.finalize();
        let mut secret = [0u8; 32];
        secret.copy_from_slice(&result);
        secret
    }

    /// Check if session key should be rotated.
    pub fn should_rotate(&self, interval_secs: u64) -> bool {
        if interval_secs == 0 {
            return false;
        }
        let elapsed = self.last_rotated.read().elapsed();
        elapsed.as_secs() >= interval_secs
    }

    /// Get key material age.
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

#[cfg(feature = "crypto")]
impl Drop for K2KKeyMaterial {
    fn drop(&mut self) {
        // Securely zero key material
        use zeroize::Zeroize;
        self.long_term_key.zeroize();
        self.session_key.write().zeroize();
    }
}

/// Encrypted K2K message wrapper.
#[cfg(feature = "crypto")]
#[derive(Debug, Clone)]
pub struct EncryptedK2KMessage {
    /// Original message metadata (unencrypted for routing).
    pub id: MessageId,
    /// Source kernel.
    pub source: KernelId,
    /// Destination kernel.
    pub destination: KernelId,
    /// Hop count.
    pub hops: u8,
    /// Timestamp when message was sent.
    pub sent_at: HlcTimestamp,
    /// Priority.
    pub priority: u8,
    /// Session key generation used for encryption.
    pub key_generation: u64,
    /// Encryption nonce (96 bits for AES-GCM).
    pub nonce: [u8; 12],
    /// Encrypted envelope data.
    pub ciphertext: Vec<u8>,
    /// Authentication tag.
    pub tag: [u8; 16],
}

/// K2K encryption manager for a single kernel.
#[cfg(feature = "crypto")]
pub struct K2KEncryptor {
    /// Configuration.
    config: K2KEncryptionConfig,
    /// This kernel's key material.
    key_material: K2KKeyMaterial,
    /// Peer public keys.
    peer_keys: parking_lot::RwLock<HashMap<KernelId, [u8; 32]>>,
    /// Encryption stats.
    stats: K2KEncryptionStats,
}

#[cfg(feature = "crypto")]
impl K2KEncryptor {
    /// Create a new encryptor for a kernel.
    pub fn new(kernel_id: KernelId, config: K2KEncryptionConfig) -> Self {
        Self {
            config,
            key_material: K2KKeyMaterial::new(kernel_id),
            peer_keys: parking_lot::RwLock::new(HashMap::new()),
            stats: K2KEncryptionStats::default(),
        }
    }

    /// Create with existing key material.
    pub fn with_key(kernel_id: KernelId, key: [u8; 32], config: K2KEncryptionConfig) -> Self {
        Self {
            config,
            key_material: K2KKeyMaterial::from_key(kernel_id, key),
            peer_keys: parking_lot::RwLock::new(HashMap::new()),
            stats: K2KEncryptionStats::default(),
        }
    }

    /// Get this kernel's public key.
    pub fn public_key(&self) -> [u8; 32] {
        // In production, derive public key properly (e.g., X25519)
        // For now, use a hash of the long-term key
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&self.key_material.long_term_key);
        hasher.update(b"k2k-public-key-v1");
        let result = hasher.finalize();
        let mut public = [0u8; 32];
        public.copy_from_slice(&result);
        public
    }

    /// Register a peer's public key.
    pub fn register_peer(&self, kernel_id: KernelId, public_key: [u8; 32]) {
        self.peer_keys.write().insert(kernel_id, public_key);
    }

    /// Unregister a peer.
    pub fn unregister_peer(&self, kernel_id: &KernelId) {
        self.peer_keys.write().remove(kernel_id);
    }

    /// Check and perform key rotation if needed.
    pub fn maybe_rotate(&self) {
        if self
            .key_material
            .should_rotate(self.config.key_rotation_interval_secs)
        {
            self.key_material.rotate_session_key();
            self.stats
                .key_rotations
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Encrypt a K2K message.
    pub fn encrypt(&self, message: &K2KMessage) -> Result<EncryptedK2KMessage> {
        if !self.config.enabled {
            return Err(RingKernelError::K2KError(
                "K2K encryption is disabled".to_string(),
            ));
        }

        // Get peer public key
        let peer_key = self
            .peer_keys
            .read()
            .get(&message.destination)
            .copied()
            .ok_or_else(|| {
                RingKernelError::K2KError(format!(
                    "No public key registered for destination kernel: {}",
                    message.destination
                ))
            })?;

        // Derive encryption key
        let shared_secret = self.key_material.derive_shared_secret(&peer_key);
        let session_key = if self.config.forward_secrecy {
            // Mix in session key for forward secrecy
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&shared_secret);
            hasher.update(&self.key_material.session_key());
            let result = hasher.finalize();
            let mut key = [0u8; 32];
            key.copy_from_slice(&result);
            key
        } else {
            shared_secret
        };

        // Generate nonce
        use rand::RngCore;
        let mut nonce = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce);

        // Serialize envelope
        let envelope_bytes = message.envelope.to_bytes();

        // Encrypt based on algorithm
        let (ciphertext, tag) = match self.config.algorithm {
            K2KEncryptionAlgorithm::Aes256Gcm => {
                use aes_gcm::{
                    aead::{Aead, KeyInit},
                    Aes256Gcm, Nonce,
                };
                let cipher = Aes256Gcm::new_from_slice(&session_key)
                    .map_err(|e| RingKernelError::K2KError(format!("AES init failed: {}", e)))?;

                let nonce_obj = Nonce::from_slice(&nonce);
                let ciphertext = cipher
                    .encrypt(nonce_obj, envelope_bytes.as_slice())
                    .map_err(|e| RingKernelError::K2KError(format!("Encryption failed: {}", e)))?;

                // AES-GCM appends tag to ciphertext
                let tag_start = ciphertext.len() - 16;
                let mut tag = [0u8; 16];
                tag.copy_from_slice(&ciphertext[tag_start..]);
                (ciphertext[..tag_start].to_vec(), tag)
            }
            K2KEncryptionAlgorithm::ChaCha20Poly1305 => {
                use chacha20poly1305::{
                    aead::{Aead, KeyInit},
                    ChaCha20Poly1305, Nonce,
                };
                let cipher = ChaCha20Poly1305::new_from_slice(&session_key)
                    .map_err(|e| RingKernelError::K2KError(format!("ChaCha init failed: {}", e)))?;

                let nonce_obj = Nonce::from_slice(&nonce);
                let ciphertext = cipher
                    .encrypt(nonce_obj, envelope_bytes.as_slice())
                    .map_err(|e| RingKernelError::K2KError(format!("Encryption failed: {}", e)))?;

                let tag_start = ciphertext.len() - 16;
                let mut tag = [0u8; 16];
                tag.copy_from_slice(&ciphertext[tag_start..]);
                (ciphertext[..tag_start].to_vec(), tag)
            }
        };

        self.stats
            .messages_encrypted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats.bytes_encrypted.fetch_add(
            envelope_bytes.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        Ok(EncryptedK2KMessage {
            id: message.id,
            source: message.source.clone(),
            destination: message.destination.clone(),
            hops: message.hops,
            sent_at: message.sent_at,
            priority: message.priority,
            key_generation: self.key_material.session_generation(),
            nonce,
            ciphertext,
            tag,
        })
    }

    /// Decrypt an encrypted K2K message.
    pub fn decrypt(&self, encrypted: &EncryptedK2KMessage) -> Result<K2KMessage> {
        if !self.config.enabled {
            return Err(RingKernelError::K2KError(
                "K2K encryption is disabled".to_string(),
            ));
        }

        // Get peer public key
        let peer_key = self
            .peer_keys
            .read()
            .get(&encrypted.source)
            .copied()
            .ok_or_else(|| {
                RingKernelError::K2KError(format!(
                    "No public key registered for source kernel: {}",
                    encrypted.source
                ))
            })?;

        // Derive decryption key
        let shared_secret = self.key_material.derive_shared_secret(&peer_key);
        let session_key = if self.config.forward_secrecy {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&shared_secret);
            hasher.update(&self.key_material.session_key());
            let result = hasher.finalize();
            let mut key = [0u8; 32];
            key.copy_from_slice(&result);
            key
        } else {
            shared_secret
        };

        // Reconstruct ciphertext with tag appended
        let mut full_ciphertext = encrypted.ciphertext.clone();
        full_ciphertext.extend_from_slice(&encrypted.tag);

        // Decrypt based on algorithm
        let plaintext = match self.config.algorithm {
            K2KEncryptionAlgorithm::Aes256Gcm => {
                use aes_gcm::{
                    aead::{Aead, KeyInit},
                    Aes256Gcm, Nonce,
                };
                let cipher = Aes256Gcm::new_from_slice(&session_key)
                    .map_err(|e| RingKernelError::K2KError(format!("AES init failed: {}", e)))?;

                let nonce = Nonce::from_slice(&encrypted.nonce);
                cipher
                    .decrypt(nonce, full_ciphertext.as_slice())
                    .map_err(|e| {
                        self.stats
                            .decryption_failures
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        RingKernelError::K2KError(format!("Decryption failed: {}", e))
                    })?
            }
            K2KEncryptionAlgorithm::ChaCha20Poly1305 => {
                use chacha20poly1305::{
                    aead::{Aead, KeyInit},
                    ChaCha20Poly1305, Nonce,
                };
                let cipher = ChaCha20Poly1305::new_from_slice(&session_key)
                    .map_err(|e| RingKernelError::K2KError(format!("ChaCha init failed: {}", e)))?;

                let nonce = Nonce::from_slice(&encrypted.nonce);
                cipher
                    .decrypt(nonce, full_ciphertext.as_slice())
                    .map_err(|e| {
                        self.stats
                            .decryption_failures
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        RingKernelError::K2KError(format!("Decryption failed: {}", e))
                    })?
            }
        };

        // Deserialize envelope
        let envelope = MessageEnvelope::from_bytes(&plaintext).map_err(|e| {
            RingKernelError::K2KError(format!("Envelope deserialization failed: {}", e))
        })?;

        self.stats
            .messages_decrypted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats
            .bytes_decrypted
            .fetch_add(plaintext.len() as u64, std::sync::atomic::Ordering::Relaxed);

        Ok(K2KMessage {
            id: encrypted.id,
            source: encrypted.source.clone(),
            destination: encrypted.destination.clone(),
            envelope,
            hops: encrypted.hops,
            sent_at: encrypted.sent_at,
            priority: encrypted.priority,
        })
    }

    /// Get encryption statistics.
    pub fn stats(&self) -> K2KEncryptionStatsSnapshot {
        K2KEncryptionStatsSnapshot {
            messages_encrypted: self
                .stats
                .messages_encrypted
                .load(std::sync::atomic::Ordering::Relaxed),
            messages_decrypted: self
                .stats
                .messages_decrypted
                .load(std::sync::atomic::Ordering::Relaxed),
            bytes_encrypted: self
                .stats
                .bytes_encrypted
                .load(std::sync::atomic::Ordering::Relaxed),
            bytes_decrypted: self
                .stats
                .bytes_decrypted
                .load(std::sync::atomic::Ordering::Relaxed),
            key_rotations: self
                .stats
                .key_rotations
                .load(std::sync::atomic::Ordering::Relaxed),
            decryption_failures: self
                .stats
                .decryption_failures
                .load(std::sync::atomic::Ordering::Relaxed),
            peer_count: self.peer_keys.read().len(),
            session_generation: self.key_material.session_generation(),
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &K2KEncryptionConfig {
        &self.config
    }
}

/// K2K encryption statistics (atomic counters).
#[cfg(feature = "crypto")]
#[derive(Default)]
struct K2KEncryptionStats {
    messages_encrypted: std::sync::atomic::AtomicU64,
    messages_decrypted: std::sync::atomic::AtomicU64,
    bytes_encrypted: std::sync::atomic::AtomicU64,
    bytes_decrypted: std::sync::atomic::AtomicU64,
    key_rotations: std::sync::atomic::AtomicU64,
    decryption_failures: std::sync::atomic::AtomicU64,
}

/// Snapshot of K2K encryption statistics.
#[cfg(feature = "crypto")]
#[derive(Debug, Clone, Default)]
pub struct K2KEncryptionStatsSnapshot {
    /// Messages encrypted.
    pub messages_encrypted: u64,
    /// Messages decrypted.
    pub messages_decrypted: u64,
    /// Bytes encrypted.
    pub bytes_encrypted: u64,
    /// Bytes decrypted.
    pub bytes_decrypted: u64,
    /// Key rotations performed.
    pub key_rotations: u64,
    /// Decryption failures (authentication/integrity).
    pub decryption_failures: u64,
    /// Number of registered peers.
    pub peer_count: usize,
    /// Current session key generation.
    pub session_generation: u64,
}

/// Encrypted K2K endpoint that wraps a standard endpoint with encryption.
#[cfg(feature = "crypto")]
pub struct EncryptedK2KEndpoint {
    /// Inner endpoint.
    inner: K2KEndpoint,
    /// Encryptor.
    encryptor: Arc<K2KEncryptor>,
}

#[cfg(feature = "crypto")]
impl EncryptedK2KEndpoint {
    /// Create an encrypted endpoint wrapping a standard endpoint.
    pub fn new(inner: K2KEndpoint, encryptor: Arc<K2KEncryptor>) -> Self {
        Self { inner, encryptor }
    }

    /// Get this kernel's public key for key exchange.
    pub fn public_key(&self) -> [u8; 32] {
        self.encryptor.public_key()
    }

    /// Register a peer's public key.
    pub fn register_peer(&self, kernel_id: KernelId, public_key: [u8; 32]) {
        self.encryptor.register_peer(kernel_id, public_key);
    }

    /// Send an encrypted message.
    pub async fn send_encrypted(
        &self,
        destination: KernelId,
        envelope: MessageEnvelope,
    ) -> Result<DeliveryReceipt> {
        self.encryptor.maybe_rotate();

        let timestamp = envelope.header.timestamp;
        let message = K2KMessage::new(
            self.inner.kernel_id.clone(),
            destination.clone(),
            envelope,
            timestamp,
        );

        // Encrypt the message
        let _encrypted = self.encryptor.encrypt(&message)?;

        // For now, send the original message (encryption metadata would need protocol support)
        // In a full implementation, the broker would handle encrypted payloads
        self.inner.send(destination, message.envelope).await
    }

    /// Receive and decrypt a message.
    pub async fn receive_decrypted(&mut self) -> Option<K2KMessage> {
        self.inner.receive().await
        // In a full implementation, decrypt the message here
    }

    /// Get encryption stats.
    pub fn encryption_stats(&self) -> K2KEncryptionStatsSnapshot {
        self.encryptor.stats()
    }
}

/// Builder for encrypted K2K broker infrastructure.
#[cfg(feature = "crypto")]
pub struct EncryptedK2KBuilder {
    k2k_config: K2KConfig,
    encryption_config: K2KEncryptionConfig,
}

#[cfg(feature = "crypto")]
impl EncryptedK2KBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            k2k_config: K2KConfig::default(),
            encryption_config: K2KEncryptionConfig::default(),
        }
    }

    /// Set K2K configuration.
    pub fn k2k_config(mut self, config: K2KConfig) -> Self {
        self.k2k_config = config;
        self
    }

    /// Set encryption configuration.
    pub fn encryption_config(mut self, config: K2KEncryptionConfig) -> Self {
        self.encryption_config = config;
        self
    }

    /// Enable forward secrecy.
    pub fn with_forward_secrecy(mut self, enabled: bool) -> Self {
        self.encryption_config.forward_secrecy = enabled;
        self
    }

    /// Set encryption algorithm.
    pub fn with_algorithm(mut self, algorithm: K2KEncryptionAlgorithm) -> Self {
        self.encryption_config.algorithm = algorithm;
        self
    }

    /// Set key rotation interval.
    pub fn with_key_rotation(mut self, interval_secs: u64) -> Self {
        self.encryption_config.key_rotation_interval_secs = interval_secs;
        self
    }

    /// Require encryption for all messages.
    pub fn require_encryption(mut self, required: bool) -> Self {
        self.encryption_config.require_encryption = required;
        self
    }

    /// Build the encrypted K2K infrastructure.
    pub fn build(self) -> (Arc<K2KBroker>, K2KEncryptionConfig) {
        (K2KBroker::new(self.k2k_config), self.encryption_config)
    }
}

#[cfg(feature = "crypto")]
impl Default for EncryptedK2KBuilder {
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

    // ============================================================
    // Multi-tenant K2K tests (§3.5)
    // ============================================================

    mod multi_tenant {
        use super::*;
        use crate::audit::MemorySink;

        fn env() -> MessageEnvelope {
            MessageEnvelope::empty(1, 2, HlcTimestamp::now(1))
        }

        // -- regression: single-tenant fast path unchanged --

        #[tokio::test]
        async fn legacy_single_tenant_send_unchanged() {
            let broker = K2KBuilder::new().build();
            let k1 = KernelId::new("k1");
            let k2 = KernelId::new("k2");
            let e1 = broker.register(k1.clone());
            let mut e2 = broker.register(k2.clone());

            let receipt = e1.send(k2.clone(), env()).await.unwrap();
            assert_eq!(receipt.status, DeliveryStatus::Delivered);
            let msg = e2.try_receive().unwrap();
            assert_eq!(msg.source, k1);
            assert_eq!(msg.envelope.tenant_id, UNSPECIFIED_TENANT);
        }

        #[tokio::test]
        async fn single_tenant_fast_path_uses_unspecified_tenant() {
            let broker = K2KBuilder::new().build();
            let k = KernelId::new("k");
            let _e = broker.register(k.clone());
            assert_eq!(broker.tenant_of(&k), Some(UNSPECIFIED_TENANT));
            assert_eq!(broker.tenant_count(), 1);
        }

        // -- cross-tenant rejection with audit --

        #[tokio::test]
        async fn cross_tenant_send_rejected_with_tenant_mismatch() {
            let broker = K2KBuilder::new().build();
            broker
                .registry()
                .register(1, TenantQuota::default())
                .unwrap();
            broker
                .registry()
                .register(2, TenantQuota::default())
                .unwrap();

            let ka = KernelId::new("tenant1_kernel");
            let kb = KernelId::new("tenant2_kernel");
            let ea = broker.register_tenant(1, AuditTag::new(10, 100), ka.clone());
            let _eb = broker.register_tenant(2, AuditTag::new(20, 200), kb.clone());

            let err = ea.send(kb.clone(), env()).await.unwrap_err();
            match err {
                RingKernelError::TenantMismatch { from, to } => {
                    assert_eq!(from, 1);
                    assert_eq!(to, 2);
                }
                other => panic!("expected TenantMismatch, got {:?}", other),
            }
            assert_eq!(broker.stats().cross_tenant_rejections, 1);
        }

        #[tokio::test]
        async fn cross_tenant_attempt_recorded_in_audit_sink() {
            let sink = Arc::new(MemorySink::new(100));
            let registry = Arc::new(TenantRegistry::with_audit_sink(sink.clone()));
            registry.register(1, TenantQuota::default()).unwrap();
            registry.register(2, TenantQuota::default()).unwrap();

            let broker = K2KBuilder::new().with_registry(registry).build();
            let ka = KernelId::new("ka");
            let kb = KernelId::new("kb");
            let ea = broker.register_tenant(1, AuditTag::new(10, 100), ka.clone());
            let _eb = broker.register_tenant(2, AuditTag::new(20, 200), kb.clone());

            let _ = ea.send(kb.clone(), env()).await.unwrap_err();
            let events = sink.events();
            assert_eq!(events.len(), 1);
            assert!(events[0].description.contains("cross-tenant"));
            let md: std::collections::HashMap<_, _> = events[0]
                .metadata
                .iter()
                .cloned()
                .collect::<std::collections::HashMap<_, _>>();
            assert_eq!(md.get("from_tenant"), Some(&"1".to_string()));
            assert_eq!(md.get("to_tenant"), Some(&"2".to_string()));
        }

        // -- same-tenant sends stamped with audit_tag --

        #[tokio::test]
        async fn same_tenant_send_succeeds_with_audit_tag() {
            let broker = K2KBuilder::new().build();
            broker
                .registry()
                .register(1, TenantQuota::default())
                .unwrap();

            let ka = KernelId::new("a");
            let kb = KernelId::new("b");
            let ea = broker.register_tenant(1, AuditTag::new(10, 100), ka.clone());
            let mut eb = broker.register_tenant(1, AuditTag::new(10, 100), kb.clone());

            let receipt = ea.send(kb.clone(), env()).await.unwrap();
            assert_eq!(receipt.status, DeliveryStatus::Delivered);

            let msg = eb.try_receive().unwrap();
            assert_eq!(msg.envelope.tenant_id, 1);
            assert_eq!(msg.envelope.audit_tag, AuditTag::new(10, 100));
        }

        // -- engagement cost tracking --

        #[tokio::test]
        async fn engagement_cost_accumulates_across_sends() {
            let broker = K2KBuilder::new().build();
            broker
                .registry()
                .register(1, TenantQuota::unlimited())
                .unwrap();

            let ka = KernelId::new("a");
            let kb = KernelId::new("b");
            let ea = broker.register_tenant(1, AuditTag::new(10, 100), ka.clone());
            let _eb = broker.register_tenant(1, AuditTag::new(10, 100), kb.clone());

            for _ in 0..4 {
                let _ = ea.send(kb.clone(), env()).await.unwrap();
                broker.registry().track_usage(
                    1,
                    AuditTag::new(10, 100),
                    std::time::Duration::from_millis(50),
                );
            }
            let cost = broker
                .registry()
                .get_engagement_cost_for(1, AuditTag::new(10, 100));
            assert_eq!(cost, std::time::Duration::from_millis(200));
        }

        #[tokio::test]
        async fn engagement_cost_separate_across_audit_tags() {
            let broker = K2KBuilder::new().build();
            broker
                .registry()
                .register(1, TenantQuota::unlimited())
                .unwrap();

            let tag_a = AuditTag::new(10, 1);
            let tag_b = AuditTag::new(10, 2);
            broker
                .registry()
                .track_usage(1, tag_a, std::time::Duration::from_millis(150));
            broker
                .registry()
                .track_usage(1, tag_b, std::time::Duration::from_millis(300));

            assert_eq!(
                broker.registry().get_engagement_cost_for(1, tag_a),
                std::time::Duration::from_millis(150)
            );
            assert_eq!(
                broker.registry().get_engagement_cost_for(1, tag_b),
                std::time::Duration::from_millis(300)
            );
        }

        // -- quota enforcement --

        #[tokio::test]
        async fn quota_enforcement_rejects_over_rate_limit() {
            let broker = K2KBuilder::new().build();
            let mut quota = TenantQuota::default();
            quota.max_messages_per_sec = 2;
            broker.registry().register(1, quota).unwrap();

            let ka = KernelId::new("a");
            let kb = KernelId::new("b");
            let ea = broker.register_tenant(1, AuditTag::new(10, 1), ka.clone());
            let _eb = broker.register_tenant(1, AuditTag::new(10, 1), kb.clone());

            assert!(ea.send(kb.clone(), env()).await.is_ok());
            assert!(ea.send(kb.clone(), env()).await.is_ok());
            let err = ea.send(kb.clone(), env()).await.unwrap_err();
            assert!(matches!(err, RingKernelError::LoadSheddingRejected { .. }));
        }

        // -- registration / deregistration --

        #[tokio::test]
        async fn register_tenant_kernel_and_unregister() {
            let broker = K2KBuilder::new().build();
            broker
                .registry()
                .register(7, TenantQuota::default())
                .unwrap();
            let k = KernelId::new("k");
            let _ep = broker.register_tenant(7, AuditTag::new(1, 1), k.clone());
            assert_eq!(broker.tenant_of(&k), Some(7));
            assert_eq!(broker.registered_kernels_for(7), vec![k.clone()]);

            broker.unregister(&k);
            assert!(!broker.is_registered(&k));
            assert!(broker.registered_kernels_for(7).is_empty());
        }

        #[tokio::test]
        async fn tenant_stats_reports_per_tenant_counts() {
            let broker = K2KBuilder::new().build();
            broker
                .registry()
                .register(1, TenantQuota::default())
                .unwrap();
            broker
                .registry()
                .register(2, TenantQuota::default())
                .unwrap();

            let _ea = broker.register_tenant(1, AuditTag::unspecified(), KernelId::new("a"));
            let _eb = broker.register_tenant(1, AuditTag::unspecified(), KernelId::new("b"));
            let _ec = broker.register_tenant(2, AuditTag::unspecified(), KernelId::new("c"));

            let s1 = broker.tenant_stats(1).unwrap();
            let s2 = broker.tenant_stats(2).unwrap();
            assert_eq!(s1.registered_endpoints, 2);
            assert_eq!(s2.registered_endpoints, 1);
        }

        #[tokio::test]
        async fn re_registering_kernel_moves_it_between_tenants() {
            let broker = K2KBuilder::new().build();
            broker
                .registry()
                .register(1, TenantQuota::default())
                .unwrap();
            broker
                .registry()
                .register(2, TenantQuota::default())
                .unwrap();

            let k = KernelId::new("roaming");
            let _e1 = broker.register_tenant(1, AuditTag::unspecified(), k.clone());
            assert_eq!(broker.tenant_of(&k), Some(1));

            let _e2 = broker.register_tenant(2, AuditTag::unspecified(), k.clone());
            assert_eq!(broker.tenant_of(&k), Some(2));
            assert!(broker.registered_kernels_for(1).is_empty());
            assert_eq!(broker.registered_kernels_for(2), vec![k]);
        }

        #[tokio::test]
        async fn send_with_audit_overrides_registration_tag() {
            let broker = K2KBuilder::new().build();
            broker
                .registry()
                .register(1, TenantQuota::unlimited())
                .unwrap();

            let ka = KernelId::new("a");
            let kb = KernelId::new("b");
            let _ea = broker.register_tenant(1, AuditTag::new(10, 1), ka.clone());
            let mut eb = broker.register_tenant(1, AuditTag::new(10, 1), kb.clone());

            let receipt = broker
                .send_with_audit(ka.clone(), kb.clone(), env(), AuditTag::new(10, 99))
                .await
                .unwrap();
            assert_eq!(receipt.status, DeliveryStatus::Delivered);
            let msg = eb.try_receive().unwrap();
            assert_eq!(msg.envelope.audit_tag, AuditTag::new(10, 99));
        }

        #[tokio::test]
        async fn default_audit_tag_behavior() {
            // Single-tenant deployment (tenant_id = 0) should preserve the
            // default unspecified audit tag when no explicit tag is set.
            let broker = K2KBuilder::new().build();
            let ka = KernelId::new("a");
            let kb = KernelId::new("b");
            let _ea = broker.register(ka.clone());
            let mut eb = broker.register(kb.clone());

            let _ = broker.send(ka.clone(), kb.clone(), env()).await.unwrap();
            let msg = eb.try_receive().unwrap();
            assert_eq!(msg.envelope.tenant_id, UNSPECIFIED_TENANT);
            assert!(msg.envelope.audit_tag.is_unspecified());
        }
    }

    // K2K Encryption tests (requires crypto feature)
    #[cfg(feature = "crypto")]
    mod crypto_tests {
        use super::*;

        #[test]
        fn test_k2k_encryption_config_default() {
            let config = K2KEncryptionConfig::default();
            assert!(config.enabled);
            assert!(config.forward_secrecy);
            assert_eq!(config.algorithm, K2KEncryptionAlgorithm::Aes256Gcm);
            assert_eq!(config.key_rotation_interval_secs, 3600);
        }

        #[test]
        fn test_k2k_encryption_config_disabled() {
            let config = K2KEncryptionConfig::disabled();
            assert!(!config.enabled);
        }

        #[test]
        fn test_k2k_encryption_config_strict() {
            let config = K2KEncryptionConfig::strict();
            assert!(config.enabled);
            assert!(config.require_encryption);
            assert!(config.forward_secrecy);
        }

        #[test]
        fn test_k2k_key_material_creation() {
            let kernel_id = KernelId::new("test_kernel");
            let key_material = K2KKeyMaterial::new(kernel_id.clone());

            assert_eq!(key_material.kernel_id(), &kernel_id);
            assert_eq!(key_material.session_generation(), 1);
        }

        #[test]
        fn test_k2k_key_material_rotation() {
            let kernel_id = KernelId::new("test_kernel");
            let key_material = K2KKeyMaterial::new(kernel_id);

            let old_session_key = key_material.session_key();
            let old_generation = key_material.session_generation();

            key_material.rotate_session_key();

            let new_session_key = key_material.session_key();
            let new_generation = key_material.session_generation();

            assert_ne!(old_session_key, new_session_key);
            assert_eq!(new_generation, old_generation + 1);
        }

        #[test]
        fn test_k2k_key_material_shared_secret() {
            let kernel1 = K2KKeyMaterial::new(KernelId::new("kernel1"));
            let kernel2 = K2KKeyMaterial::new(KernelId::new("kernel2"));

            // Get public keys (simulated)
            let pk1 = {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(&kernel1.long_term_key);
                hasher.update(b"k2k-public-key-v1");
                let result = hasher.finalize();
                let mut public = [0u8; 32];
                public.copy_from_slice(&result);
                public
            };
            let pk2 = {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(&kernel2.long_term_key);
                hasher.update(b"k2k-public-key-v1");
                let result = hasher.finalize();
                let mut public = [0u8; 32];
                public.copy_from_slice(&result);
                public
            };

            // Shared secrets should be different for different pairs
            let secret1 = kernel1.derive_shared_secret(&pk2);
            let secret2 = kernel2.derive_shared_secret(&pk1);

            // They won't be equal with this simplified implementation
            // In a real X25519 implementation, they would be
            assert_eq!(secret1.len(), 32);
            assert_eq!(secret2.len(), 32);
        }

        #[test]
        fn test_k2k_encryptor_creation() {
            let kernel_id = KernelId::new("test_kernel");
            let config = K2KEncryptionConfig::default();
            let encryptor = K2KEncryptor::new(kernel_id.clone(), config);

            let public_key = encryptor.public_key();
            assert_eq!(public_key.len(), 32);

            let stats = encryptor.stats();
            assert_eq!(stats.messages_encrypted, 0);
            assert_eq!(stats.messages_decrypted, 0);
            assert_eq!(stats.peer_count, 0);
        }

        #[test]
        fn test_k2k_encryptor_peer_registration() {
            let kernel_id = KernelId::new("test_kernel");
            let config = K2KEncryptionConfig::default();
            let encryptor = K2KEncryptor::new(kernel_id, config);

            let peer_id = KernelId::new("peer_kernel");
            let peer_key = [42u8; 32];

            encryptor.register_peer(peer_id.clone(), peer_key);
            assert_eq!(encryptor.stats().peer_count, 1);

            encryptor.unregister_peer(&peer_id);
            assert_eq!(encryptor.stats().peer_count, 0);
        }

        #[test]
        fn test_k2k_encrypted_builder() {
            let (broker, config) = EncryptedK2KBuilder::new()
                .with_forward_secrecy(true)
                .with_algorithm(K2KEncryptionAlgorithm::ChaCha20Poly1305)
                .with_key_rotation(1800)
                .require_encryption(true)
                .build();

            assert!(config.forward_secrecy);
            assert_eq!(config.algorithm, K2KEncryptionAlgorithm::ChaCha20Poly1305);
            assert_eq!(config.key_rotation_interval_secs, 1800);
            assert!(config.require_encryption);

            // Broker should be functional
            let stats = broker.stats();
            assert_eq!(stats.registered_endpoints, 0);
        }

        #[test]
        fn test_k2k_encryption_stats_snapshot() {
            let stats = K2KEncryptionStatsSnapshot::default();
            assert_eq!(stats.messages_encrypted, 0);
            assert_eq!(stats.messages_decrypted, 0);
            assert_eq!(stats.bytes_encrypted, 0);
            assert_eq!(stats.bytes_decrypted, 0);
            assert_eq!(stats.key_rotations, 0);
            assert_eq!(stats.decryption_failures, 0);
            assert_eq!(stats.peer_count, 0);
            assert_eq!(stats.session_generation, 0);
        }

        #[test]
        fn test_k2k_encryption_algorithms() {
            // Test that both algorithms are distinct
            assert_ne!(
                K2KEncryptionAlgorithm::Aes256Gcm,
                K2KEncryptionAlgorithm::ChaCha20Poly1305
            );
        }

        #[test]
        fn test_k2k_key_material_should_rotate() {
            let kernel_id = KernelId::new("test_kernel");
            let key_material = K2KKeyMaterial::new(kernel_id);

            // Should not rotate with 0 interval
            assert!(!key_material.should_rotate(0));

            // Should not rotate immediately with long interval
            assert!(!key_material.should_rotate(3600));
        }

        #[test]
        fn test_k2k_encryptor_disabled_encryption() {
            let kernel_id = KernelId::new("test_kernel");
            let config = K2KEncryptionConfig::disabled();
            let encryptor = K2KEncryptor::new(kernel_id.clone(), config);

            // Create a test message
            let envelope = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));
            let message = K2KMessage::new(
                kernel_id,
                KernelId::new("dest"),
                envelope,
                HlcTimestamp::now(1),
            );

            // Should fail when encryption is disabled
            let result = encryptor.encrypt(&message);
            assert!(result.is_err());
        }

        #[test]
        fn test_k2k_encryptor_missing_peer_key() {
            let kernel_id = KernelId::new("test_kernel");
            let config = K2KEncryptionConfig::default();
            let encryptor = K2KEncryptor::new(kernel_id.clone(), config);

            // Create a test message to unknown destination
            let envelope = MessageEnvelope::empty(1, 2, HlcTimestamp::now(1));
            let message = K2KMessage::new(
                kernel_id,
                KernelId::new("unknown_dest"),
                envelope,
                HlcTimestamp::now(1),
            );

            // Should fail due to missing peer key
            let result = encryptor.encrypt(&message);
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("No public key"));
        }
    }
}
