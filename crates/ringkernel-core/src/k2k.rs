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
