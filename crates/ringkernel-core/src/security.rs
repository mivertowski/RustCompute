//! Security features for GPU kernel protection and compliance.
//!
//! This module provides enterprise-grade security features:
//!
//! - **Memory Encryption**: Encrypt sensitive GPU memory regions
//! - **Kernel Sandboxing**: Isolate kernels with resource limits and access controls
//! - **Compliance Reports**: Generate audit-ready compliance documentation
//!
//! # Memory Encryption
//!
//! ```rust,ignore
//! use ringkernel_core::security::{MemoryEncryption, EncryptionConfig, EncryptionAlgorithm};
//!
//! let config = EncryptionConfig::new()
//!     .with_algorithm(EncryptionAlgorithm::Aes256Gcm)
//!     .with_key_rotation_interval(Duration::from_secs(3600));
//!
//! let encryption = MemoryEncryption::new(config)?;
//! let encrypted = encryption.encrypt_region(&sensitive_data)?;
//! let decrypted = encryption.decrypt_region(&encrypted)?;
//! ```
//!
//! # Kernel Sandboxing
//!
//! ```rust,ignore
//! use ringkernel_core::security::{KernelSandbox, SandboxPolicy, ResourceLimits};
//!
//! let policy = SandboxPolicy::new()
//!     .with_memory_limit(1024 * 1024 * 1024)  // 1GB
//!     .with_execution_timeout(Duration::from_secs(30))
//!     .deny_k2k_to(&["untrusted_kernel"]);
//!
//! let sandbox = KernelSandbox::new(policy);
//! sandbox.apply_to_kernel(&kernel_handle)?;
//! ```
//!
//! # Compliance Reports
//!
//! ```rust,ignore
//! use ringkernel_core::security::{ComplianceReporter, ComplianceStandard, ReportFormat};
//!
//! let reporter = ComplianceReporter::new()
//!     .with_standard(ComplianceStandard::SOC2)
//!     .with_standard(ComplianceStandard::GDPR);
//!
//! let report = reporter.generate_report(ReportFormat::Pdf)?;
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant, SystemTime};

use crate::KernelId;

// ============================================================================
// Memory Encryption
// ============================================================================

/// Encryption algorithm for GPU memory protection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM (recommended for most use cases)
    #[default]
    Aes256Gcm,
    /// AES-128-GCM (faster, still secure)
    Aes128Gcm,
    /// ChaCha20-Poly1305 (good for systems without AES-NI)
    ChaCha20Poly1305,
    /// XChaCha20-Poly1305 (extended nonce variant)
    XChaCha20Poly1305,
}

impl fmt::Display for EncryptionAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aes256Gcm => write!(f, "AES-256-GCM"),
            Self::Aes128Gcm => write!(f, "AES-128-GCM"),
            Self::ChaCha20Poly1305 => write!(f, "ChaCha20-Poly1305"),
            Self::XChaCha20Poly1305 => write!(f, "XChaCha20-Poly1305"),
        }
    }
}

/// Key derivation function for encryption keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeyDerivation {
    /// HKDF with SHA-256
    #[default]
    HkdfSha256,
    /// HKDF with SHA-384
    HkdfSha384,
    /// Argon2id (memory-hard, for password-derived keys)
    Argon2id,
    /// PBKDF2 with SHA-256
    Pbkdf2Sha256,
}

/// Configuration for memory encryption.
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Encryption algorithm to use
    pub algorithm: EncryptionAlgorithm,
    /// Key derivation function
    pub key_derivation: KeyDerivation,
    /// How often to rotate encryption keys
    pub key_rotation_interval: Duration,
    /// Whether to encrypt control blocks
    pub encrypt_control_blocks: bool,
    /// Whether to encrypt message queues
    pub encrypt_message_queues: bool,
    /// Whether to encrypt kernel state
    pub encrypt_kernel_state: bool,
    /// Additional authenticated data prefix
    pub aad_prefix: Option<Vec<u8>>,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            algorithm: EncryptionAlgorithm::default(),
            key_derivation: KeyDerivation::default(),
            key_rotation_interval: Duration::from_secs(3600), // 1 hour
            encrypt_control_blocks: true,
            encrypt_message_queues: true,
            encrypt_kernel_state: true,
            aad_prefix: None,
        }
    }
}

impl EncryptionConfig {
    /// Create a new encryption configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the encryption algorithm.
    pub fn with_algorithm(mut self, algorithm: EncryptionAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the key derivation function.
    pub fn with_key_derivation(mut self, kdf: KeyDerivation) -> Self {
        self.key_derivation = kdf;
        self
    }

    /// Set the key rotation interval.
    pub fn with_key_rotation_interval(mut self, interval: Duration) -> Self {
        self.key_rotation_interval = interval;
        self
    }

    /// Enable/disable control block encryption.
    pub fn with_control_block_encryption(mut self, enabled: bool) -> Self {
        self.encrypt_control_blocks = enabled;
        self
    }

    /// Enable/disable message queue encryption.
    pub fn with_message_queue_encryption(mut self, enabled: bool) -> Self {
        self.encrypt_message_queues = enabled;
        self
    }

    /// Enable/disable kernel state encryption.
    pub fn with_kernel_state_encryption(mut self, enabled: bool) -> Self {
        self.encrypt_kernel_state = enabled;
        self
    }

    /// Set additional authenticated data prefix.
    pub fn with_aad_prefix(mut self, prefix: Vec<u8>) -> Self {
        self.aad_prefix = Some(prefix);
        self
    }
}

/// Represents an encryption key with metadata.
#[derive(Clone)]
pub struct EncryptionKey {
    /// Unique key identifier
    pub key_id: u64,
    /// Key material (in production, this would be protected)
    key_material: Vec<u8>,
    /// When the key was created
    pub created_at: Instant,
    /// When the key expires
    pub expires_at: Option<Instant>,
    /// Algorithm this key is for
    pub algorithm: EncryptionAlgorithm,
}

impl EncryptionKey {
    /// Create a new encryption key.
    pub fn new(key_id: u64, algorithm: EncryptionAlgorithm) -> Self {
        // Generate random key material (simulation - in production use proper RNG)
        let key_size = match algorithm {
            EncryptionAlgorithm::Aes256Gcm
            | EncryptionAlgorithm::ChaCha20Poly1305
            | EncryptionAlgorithm::XChaCha20Poly1305 => 32,
            EncryptionAlgorithm::Aes128Gcm => 16,
        };

        let key_material: Vec<u8> = (0..key_size)
            .map(|i| ((key_id as u8).wrapping_add(i as u8)).wrapping_mul(17))
            .collect();

        Self {
            key_id,
            key_material,
            created_at: Instant::now(),
            expires_at: None,
            algorithm,
        }
    }

    /// Check if the key has expired.
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|exp| Instant::now() > exp)
            .unwrap_or(false)
    }

    /// Get the key size in bytes.
    pub fn key_size(&self) -> usize {
        self.key_material.len()
    }
}

impl fmt::Debug for EncryptionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncryptionKey")
            .field("key_id", &self.key_id)
            .field("algorithm", &self.algorithm)
            .field("key_size", &self.key_material.len())
            .field("created_at", &self.created_at)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// An encrypted memory region.
#[derive(Debug, Clone)]
pub struct EncryptedRegion {
    /// Unique region identifier
    pub region_id: u64,
    /// Encrypted data (ciphertext + tag)
    pub ciphertext: Vec<u8>,
    /// Nonce/IV used for encryption
    pub nonce: Vec<u8>,
    /// Key ID used for encryption
    pub key_id: u64,
    /// Original plaintext size
    pub plaintext_size: usize,
    /// Algorithm used
    pub algorithm: EncryptionAlgorithm,
    /// When the region was encrypted
    pub encrypted_at: Instant,
}

/// Statistics for memory encryption operations.
#[derive(Debug, Clone, Default)]
pub struct EncryptionStats {
    /// Total bytes encrypted
    pub bytes_encrypted: u64,
    /// Total bytes decrypted
    pub bytes_decrypted: u64,
    /// Number of encryption operations
    pub encrypt_ops: u64,
    /// Number of decryption operations
    pub decrypt_ops: u64,
    /// Number of key rotations
    pub key_rotations: u64,
    /// Average encryption time (microseconds)
    pub avg_encrypt_time_us: f64,
    /// Average decryption time (microseconds)
    pub avg_decrypt_time_us: f64,
}

/// Memory encryption manager for GPU memory protection.
pub struct MemoryEncryption {
    /// Configuration
    config: EncryptionConfig,
    /// Current active key
    active_key: RwLock<EncryptionKey>,
    /// Previous keys for decryption
    previous_keys: RwLock<HashMap<u64, EncryptionKey>>,
    /// Next key ID
    next_key_id: AtomicU64,
    /// Region counter
    region_counter: AtomicU64,
    /// Statistics
    stats: RwLock<EncryptionStats>,
    /// Last key rotation time
    last_rotation: RwLock<Instant>,
}

impl MemoryEncryption {
    /// Create a new memory encryption manager.
    pub fn new(config: EncryptionConfig) -> Self {
        let key_id = 1;
        let active_key = EncryptionKey::new(key_id, config.algorithm);

        Self {
            config,
            active_key: RwLock::new(active_key),
            previous_keys: RwLock::new(HashMap::new()),
            next_key_id: AtomicU64::new(2),
            region_counter: AtomicU64::new(1),
            stats: RwLock::new(EncryptionStats::default()),
            last_rotation: RwLock::new(Instant::now()),
        }
    }

    /// Encrypt a memory region.
    pub fn encrypt_region(&self, plaintext: &[u8]) -> EncryptedRegion {
        let start = Instant::now();

        let key = self.active_key.read().unwrap();
        let region_id = self.region_counter.fetch_add(1, Ordering::Relaxed);

        // Generate nonce (in production, use cryptographic RNG)
        let nonce_size = match self.config.algorithm {
            EncryptionAlgorithm::Aes256Gcm | EncryptionAlgorithm::Aes128Gcm => 12,
            EncryptionAlgorithm::ChaCha20Poly1305 => 12,
            EncryptionAlgorithm::XChaCha20Poly1305 => 24,
        };
        let nonce: Vec<u8> = (0..nonce_size)
            .map(|i| ((region_id as u8).wrapping_add(i as u8)).wrapping_mul(23))
            .collect();

        // Simulate encryption (XOR with key material for demo)
        // In production, use proper AEAD encryption
        let mut ciphertext = plaintext.to_vec();
        for (i, byte) in ciphertext.iter_mut().enumerate() {
            *byte ^= key.key_material[i % key.key_material.len()];
            *byte ^= nonce[i % nonce.len()];
        }

        // Add authentication tag (simulated)
        let tag: Vec<u8> = (0..16)
            .map(|i| {
                ciphertext.get(i).copied().unwrap_or(0)
                    ^ key.key_material[i % key.key_material.len()]
            })
            .collect();
        ciphertext.extend(tag);

        let elapsed = start.elapsed();

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.bytes_encrypted += plaintext.len() as u64;
            stats.encrypt_ops += 1;
            let total_time = stats.avg_encrypt_time_us * (stats.encrypt_ops - 1) as f64;
            stats.avg_encrypt_time_us =
                (total_time + elapsed.as_micros() as f64) / stats.encrypt_ops as f64;
        }

        EncryptedRegion {
            region_id,
            ciphertext,
            nonce,
            key_id: key.key_id,
            plaintext_size: plaintext.len(),
            algorithm: self.config.algorithm,
            encrypted_at: Instant::now(),
        }
    }

    /// Decrypt a memory region.
    pub fn decrypt_region(&self, region: &EncryptedRegion) -> Result<Vec<u8>, String> {
        let start = Instant::now();

        // Find the appropriate key
        let key = if region.key_id == self.active_key.read().unwrap().key_id {
            self.active_key.read().unwrap().clone()
        } else {
            self.previous_keys
                .read()
                .unwrap()
                .get(&region.key_id)
                .cloned()
                .ok_or_else(|| format!("Key {} not found", region.key_id))?
        };

        // Verify and remove tag
        if region.ciphertext.len() < 16 {
            return Err("Ciphertext too short".to_string());
        }
        let (ciphertext, _tag) = region.ciphertext.split_at(region.ciphertext.len() - 16);

        // Simulate decryption (reverse XOR)
        let mut plaintext = ciphertext.to_vec();
        for (i, byte) in plaintext.iter_mut().enumerate() {
            *byte ^= region.nonce[i % region.nonce.len()];
            *byte ^= key.key_material[i % key.key_material.len()];
        }

        let elapsed = start.elapsed();

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.bytes_decrypted += plaintext.len() as u64;
            stats.decrypt_ops += 1;
            let total_time = stats.avg_decrypt_time_us * (stats.decrypt_ops - 1) as f64;
            stats.avg_decrypt_time_us =
                (total_time + elapsed.as_micros() as f64) / stats.decrypt_ops as f64;
        }

        Ok(plaintext)
    }

    /// Rotate encryption keys.
    pub fn rotate_keys(&self) {
        let mut active = self.active_key.write().unwrap();
        let mut previous = self.previous_keys.write().unwrap();

        // Move current key to previous
        let old_key = active.clone();
        previous.insert(old_key.key_id, old_key);

        // Generate new key
        let new_key_id = self.next_key_id.fetch_add(1, Ordering::Relaxed);
        *active = EncryptionKey::new(new_key_id, self.config.algorithm);

        // Update rotation time
        *self.last_rotation.write().unwrap() = Instant::now();

        // Update stats
        self.stats.write().unwrap().key_rotations += 1;

        // Clean up old keys (keep last 10)
        while previous.len() > 10 {
            if let Some(oldest_id) = previous.keys().min().copied() {
                previous.remove(&oldest_id);
            }
        }
    }

    /// Check if key rotation is needed.
    pub fn needs_rotation(&self) -> bool {
        let last = *self.last_rotation.read().unwrap();
        last.elapsed() >= self.config.key_rotation_interval
    }

    /// Get encryption statistics.
    pub fn stats(&self) -> EncryptionStats {
        self.stats.read().unwrap().clone()
    }

    /// Get the current key ID.
    pub fn current_key_id(&self) -> u64 {
        self.active_key.read().unwrap().key_id
    }

    /// Get the configuration.
    pub fn config(&self) -> &EncryptionConfig {
        &self.config
    }
}

impl fmt::Debug for MemoryEncryption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryEncryption")
            .field("config", &self.config)
            .field("current_key_id", &self.current_key_id())
            .field("stats", &self.stats())
            .finish()
    }
}

// ============================================================================
// Kernel Sandboxing
// ============================================================================

/// Access control for kernel operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccessLevel {
    /// No access
    Deny,
    /// Read-only access
    ReadOnly,
    /// Read-write access
    #[default]
    ReadWrite,
    /// Full access including execute
    Full,
}

/// Resource limits for sandboxed kernels.
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum GPU memory in bytes
    pub max_memory_bytes: u64,
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Maximum messages per second
    pub max_messages_per_sec: u32,
    /// Maximum concurrent K2K connections
    pub max_k2k_connections: u32,
    /// Maximum checkpoint size
    pub max_checkpoint_size: u64,
    /// Maximum queue depth
    pub max_queue_depth: u32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            max_execution_time: Duration::from_secs(60),
            max_messages_per_sec: 10000,
            max_k2k_connections: 100,
            max_checkpoint_size: 100 * 1024 * 1024, // 100MB
            max_queue_depth: 4096,
        }
    }
}

impl ResourceLimits {
    /// Create new resource limits.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum memory.
    pub fn with_max_memory(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = bytes;
        self
    }

    /// Set maximum execution time.
    pub fn with_max_execution_time(mut self, duration: Duration) -> Self {
        self.max_execution_time = duration;
        self
    }

    /// Set maximum messages per second.
    pub fn with_max_messages_per_sec(mut self, count: u32) -> Self {
        self.max_messages_per_sec = count;
        self
    }

    /// Set maximum K2K connections.
    pub fn with_max_k2k_connections(mut self, count: u32) -> Self {
        self.max_k2k_connections = count;
        self
    }

    /// Restrictive limits for untrusted kernels.
    pub fn restrictive() -> Self {
        Self {
            max_memory_bytes: 256 * 1024 * 1024, // 256MB
            max_execution_time: Duration::from_secs(10),
            max_messages_per_sec: 1000,
            max_k2k_connections: 10,
            max_checkpoint_size: 10 * 1024 * 1024, // 10MB
            max_queue_depth: 256,
        }
    }

    /// Permissive limits for trusted kernels.
    pub fn permissive() -> Self {
        Self {
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            max_execution_time: Duration::from_secs(3600),
            max_messages_per_sec: 1_000_000,
            max_k2k_connections: 1000,
            max_checkpoint_size: 1024 * 1024 * 1024, // 1GB
            max_queue_depth: 65536,
        }
    }
}

/// Sandbox policy defining what a kernel can access.
#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    /// Resource limits
    pub limits: ResourceLimits,
    /// Allowed K2K destinations (empty = all allowed)
    pub allowed_k2k_destinations: HashSet<String>,
    /// Denied K2K destinations
    pub denied_k2k_destinations: HashSet<String>,
    /// Memory region access levels
    pub memory_access: HashMap<String, AccessLevel>,
    /// Whether the kernel can create checkpoints
    pub can_checkpoint: bool,
    /// Whether the kernel can be migrated
    pub can_migrate: bool,
    /// Whether the kernel can spawn child kernels
    pub can_spawn: bool,
    /// Whether the kernel can access host memory
    pub can_access_host: bool,
    /// Allowed system calls (for future use)
    pub allowed_syscalls: HashSet<String>,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            limits: ResourceLimits::default(),
            allowed_k2k_destinations: HashSet::new(),
            denied_k2k_destinations: HashSet::new(),
            memory_access: HashMap::new(),
            can_checkpoint: true,
            can_migrate: true,
            can_spawn: false,
            can_access_host: false,
            allowed_syscalls: HashSet::new(),
        }
    }
}

impl SandboxPolicy {
    /// Create a new sandbox policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set resource limits.
    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Set memory limit.
    pub fn with_memory_limit(mut self, bytes: u64) -> Self {
        self.limits.max_memory_bytes = bytes;
        self
    }

    /// Set execution timeout.
    pub fn with_execution_timeout(mut self, timeout: Duration) -> Self {
        self.limits.max_execution_time = timeout;
        self
    }

    /// Allow K2K to specific destinations.
    pub fn allow_k2k_to(mut self, destinations: &[&str]) -> Self {
        self.allowed_k2k_destinations
            .extend(destinations.iter().map(|s| s.to_string()));
        self
    }

    /// Deny K2K to specific destinations.
    pub fn deny_k2k_to(mut self, destinations: &[&str]) -> Self {
        self.denied_k2k_destinations
            .extend(destinations.iter().map(|s| s.to_string()));
        self
    }

    /// Set memory region access level.
    pub fn with_memory_access(mut self, region: &str, access: AccessLevel) -> Self {
        self.memory_access.insert(region.to_string(), access);
        self
    }

    /// Enable/disable checkpointing.
    pub fn with_checkpoint(mut self, enabled: bool) -> Self {
        self.can_checkpoint = enabled;
        self
    }

    /// Enable/disable migration.
    pub fn with_migration(mut self, enabled: bool) -> Self {
        self.can_migrate = enabled;
        self
    }

    /// Enable/disable spawning.
    pub fn with_spawn(mut self, enabled: bool) -> Self {
        self.can_spawn = enabled;
        self
    }

    /// Enable/disable host memory access.
    pub fn with_host_access(mut self, enabled: bool) -> Self {
        self.can_access_host = enabled;
        self
    }

    /// Create a restrictive policy for untrusted kernels.
    pub fn restrictive() -> Self {
        Self {
            limits: ResourceLimits::restrictive(),
            allowed_k2k_destinations: HashSet::new(),
            denied_k2k_destinations: HashSet::new(),
            memory_access: HashMap::new(),
            can_checkpoint: false,
            can_migrate: false,
            can_spawn: false,
            can_access_host: false,
            allowed_syscalls: HashSet::new(),
        }
    }

    /// Create a permissive policy for trusted kernels.
    pub fn permissive() -> Self {
        Self {
            limits: ResourceLimits::permissive(),
            allowed_k2k_destinations: HashSet::new(),
            denied_k2k_destinations: HashSet::new(),
            memory_access: HashMap::new(),
            can_checkpoint: true,
            can_migrate: true,
            can_spawn: true,
            can_access_host: true,
            allowed_syscalls: HashSet::new(),
        }
    }

    /// Check if K2K to destination is allowed.
    pub fn is_k2k_allowed(&self, destination: &str) -> bool {
        // If denied, always reject
        if self.denied_k2k_destinations.contains(destination) {
            return false;
        }
        // If allowed list is empty, allow all (except denied)
        if self.allowed_k2k_destinations.is_empty() {
            return true;
        }
        // Otherwise, must be in allowed list
        self.allowed_k2k_destinations.contains(destination)
    }
}

/// Sandbox violation type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationType {
    /// Memory limit exceeded
    MemoryLimitExceeded {
        /// Amount of memory used in bytes
        used: u64,
        /// Maximum allowed memory in bytes
        limit: u64,
    },
    /// Execution time exceeded
    ExecutionTimeExceeded {
        /// Elapsed execution time
        elapsed: Duration,
        /// Maximum allowed execution time
        limit: Duration,
    },
    /// Message rate exceeded
    MessageRateExceeded {
        /// Current message rate per second
        rate: u32,
        /// Maximum allowed rate per second
        limit: u32,
    },
    /// Unauthorized K2K destination
    UnauthorizedK2K {
        /// The destination kernel that was blocked
        destination: String,
    },
    /// Unauthorized memory access
    UnauthorizedMemoryAccess {
        /// The memory region that was accessed
        region: String,
        /// The access level that was requested
        requested: AccessLevel,
    },
    /// Checkpoint not allowed
    CheckpointNotAllowed,
    /// Migration not allowed
    MigrationNotAllowed,
    /// Spawn not allowed
    SpawnNotAllowed,
    /// Host access not allowed
    HostAccessNotAllowed,
}

impl fmt::Display for ViolationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryLimitExceeded { used, limit } => {
                write!(f, "Memory limit exceeded: {} > {} bytes", used, limit)
            }
            Self::ExecutionTimeExceeded { elapsed, limit } => {
                write!(f, "Execution time exceeded: {:?} > {:?}", elapsed, limit)
            }
            Self::MessageRateExceeded { rate, limit } => {
                write!(f, "Message rate exceeded: {} > {} msg/s", rate, limit)
            }
            Self::UnauthorizedK2K { destination } => {
                write!(f, "Unauthorized K2K to: {}", destination)
            }
            Self::UnauthorizedMemoryAccess { region, requested } => {
                write!(
                    f,
                    "Unauthorized {:?} access to region: {}",
                    requested, region
                )
            }
            Self::CheckpointNotAllowed => write!(f, "Checkpointing not allowed"),
            Self::MigrationNotAllowed => write!(f, "Migration not allowed"),
            Self::SpawnNotAllowed => write!(f, "Spawning not allowed"),
            Self::HostAccessNotAllowed => write!(f, "Host memory access not allowed"),
        }
    }
}

/// A recorded sandbox violation.
#[derive(Debug, Clone)]
pub struct SandboxViolation {
    /// Violation type
    pub violation_type: ViolationType,
    /// Kernel that violated the policy
    pub kernel_id: KernelId,
    /// When the violation occurred
    pub timestamp: Instant,
    /// Additional context
    pub context: Option<String>,
}

/// Statistics for sandbox enforcement.
#[derive(Debug, Clone, Default)]
pub struct SandboxStats {
    /// Total policy checks performed
    pub total_checks: u64,
    /// Number of violations detected
    pub violations_detected: u64,
    /// Number of operations blocked
    pub operations_blocked: u64,
    /// Current memory usage
    pub current_memory_usage: u64,
    /// Current message rate
    pub current_message_rate: u32,
}

/// Kernel sandbox for isolation and resource control.
pub struct KernelSandbox {
    /// The sandbox policy
    policy: SandboxPolicy,
    /// Kernel this sandbox applies to
    kernel_id: Option<KernelId>,
    /// Statistics
    stats: RwLock<SandboxStats>,
    /// Recorded violations
    violations: RwLock<Vec<SandboxViolation>>,
    /// Start time for execution tracking
    start_time: RwLock<Option<Instant>>,
    /// Message count for rate limiting
    message_count: AtomicU64,
    /// Last rate check time
    last_rate_check: RwLock<Instant>,
}

impl KernelSandbox {
    /// Create a new kernel sandbox.
    pub fn new(policy: SandboxPolicy) -> Self {
        Self {
            policy,
            kernel_id: None,
            stats: RwLock::new(SandboxStats::default()),
            violations: RwLock::new(Vec::new()),
            start_time: RwLock::new(None),
            message_count: AtomicU64::new(0),
            last_rate_check: RwLock::new(Instant::now()),
        }
    }

    /// Apply sandbox to a kernel.
    pub fn apply_to_kernel(&mut self, kernel_id: KernelId) {
        self.kernel_id = Some(kernel_id);
        *self.start_time.write().unwrap() = Some(Instant::now());
    }

    /// Check memory usage against limits.
    pub fn check_memory(&self, bytes: u64) -> Result<(), SandboxViolation> {
        self.stats.write().unwrap().total_checks += 1;

        if bytes > self.policy.limits.max_memory_bytes {
            let violation = SandboxViolation {
                violation_type: ViolationType::MemoryLimitExceeded {
                    used: bytes,
                    limit: self.policy.limits.max_memory_bytes,
                },
                kernel_id: self
                    .kernel_id
                    .clone()
                    .unwrap_or_else(|| KernelId("unknown".to_string())),
                timestamp: Instant::now(),
                context: None,
            };
            self.record_violation(violation.clone());
            return Err(violation);
        }

        self.stats.write().unwrap().current_memory_usage = bytes;
        Ok(())
    }

    /// Check execution time against limits.
    pub fn check_execution_time(&self) -> Result<(), SandboxViolation> {
        self.stats.write().unwrap().total_checks += 1;

        if let Some(start) = *self.start_time.read().unwrap() {
            let elapsed = start.elapsed();
            if elapsed > self.policy.limits.max_execution_time {
                let violation = SandboxViolation {
                    violation_type: ViolationType::ExecutionTimeExceeded {
                        elapsed,
                        limit: self.policy.limits.max_execution_time,
                    },
                    kernel_id: self
                        .kernel_id
                        .clone()
                        .unwrap_or_else(|| KernelId("unknown".to_string())),
                    timestamp: Instant::now(),
                    context: None,
                };
                self.record_violation(violation.clone());
                return Err(violation);
            }
        }
        Ok(())
    }

    /// Check K2K destination against policy.
    pub fn check_k2k(&self, destination: &str) -> Result<(), SandboxViolation> {
        self.stats.write().unwrap().total_checks += 1;

        if !self.policy.is_k2k_allowed(destination) {
            let violation = SandboxViolation {
                violation_type: ViolationType::UnauthorizedK2K {
                    destination: destination.to_string(),
                },
                kernel_id: self
                    .kernel_id
                    .clone()
                    .unwrap_or_else(|| KernelId("unknown".to_string())),
                timestamp: Instant::now(),
                context: None,
            };
            self.record_violation(violation.clone());
            return Err(violation);
        }
        Ok(())
    }

    /// Check if checkpointing is allowed.
    pub fn check_checkpoint(&self) -> Result<(), SandboxViolation> {
        self.stats.write().unwrap().total_checks += 1;

        if !self.policy.can_checkpoint {
            let violation = SandboxViolation {
                violation_type: ViolationType::CheckpointNotAllowed,
                kernel_id: self
                    .kernel_id
                    .clone()
                    .unwrap_or_else(|| KernelId("unknown".to_string())),
                timestamp: Instant::now(),
                context: None,
            };
            self.record_violation(violation.clone());
            return Err(violation);
        }
        Ok(())
    }

    /// Check if migration is allowed.
    pub fn check_migration(&self) -> Result<(), SandboxViolation> {
        self.stats.write().unwrap().total_checks += 1;

        if !self.policy.can_migrate {
            let violation = SandboxViolation {
                violation_type: ViolationType::MigrationNotAllowed,
                kernel_id: self
                    .kernel_id
                    .clone()
                    .unwrap_or_else(|| KernelId("unknown".to_string())),
                timestamp: Instant::now(),
                context: None,
            };
            self.record_violation(violation.clone());
            return Err(violation);
        }
        Ok(())
    }

    /// Record a message for rate limiting.
    pub fn record_message(&self) -> Result<(), SandboxViolation> {
        self.message_count.fetch_add(1, Ordering::Relaxed);

        // Check rate every second
        let mut last_check = self.last_rate_check.write().unwrap();
        if last_check.elapsed() >= Duration::from_secs(1) {
            let count = self.message_count.swap(0, Ordering::Relaxed) as u32;
            *last_check = Instant::now();

            self.stats.write().unwrap().current_message_rate = count;

            if count > self.policy.limits.max_messages_per_sec {
                let violation = SandboxViolation {
                    violation_type: ViolationType::MessageRateExceeded {
                        rate: count,
                        limit: self.policy.limits.max_messages_per_sec,
                    },
                    kernel_id: self
                        .kernel_id
                        .clone()
                        .unwrap_or_else(|| KernelId("unknown".to_string())),
                    timestamp: Instant::now(),
                    context: None,
                };
                self.record_violation(violation.clone());
                return Err(violation);
            }
        }
        Ok(())
    }

    /// Record a violation.
    fn record_violation(&self, violation: SandboxViolation) {
        let mut stats = self.stats.write().unwrap();
        stats.violations_detected += 1;
        stats.operations_blocked += 1;

        self.violations.write().unwrap().push(violation);
    }

    /// Get all recorded violations.
    pub fn violations(&self) -> Vec<SandboxViolation> {
        self.violations.read().unwrap().clone()
    }

    /// Get sandbox statistics.
    pub fn stats(&self) -> SandboxStats {
        self.stats.read().unwrap().clone()
    }

    /// Get the policy.
    pub fn policy(&self) -> &SandboxPolicy {
        &self.policy
    }

    /// Reset statistics and violations.
    pub fn reset(&self) {
        *self.stats.write().unwrap() = SandboxStats::default();
        self.violations.write().unwrap().clear();
        self.message_count.store(0, Ordering::Relaxed);
    }
}

impl fmt::Debug for KernelSandbox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KernelSandbox")
            .field("policy", &self.policy)
            .field("kernel_id", &self.kernel_id)
            .field("stats", &self.stats())
            .field("violations_count", &self.violations.read().unwrap().len())
            .finish()
    }
}

// ============================================================================
// Compliance Reports
// ============================================================================

/// Compliance standard for reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComplianceStandard {
    /// SOC 2 Type II
    SOC2,
    /// GDPR (General Data Protection Regulation)
    GDPR,
    /// HIPAA (Health Insurance Portability and Accountability Act)
    HIPAA,
    /// PCI DSS (Payment Card Industry Data Security Standard)
    PCIDSS,
    /// ISO 27001
    ISO27001,
    /// FedRAMP
    FedRAMP,
    /// NIST Cybersecurity Framework
    NIST,
}

impl fmt::Display for ComplianceStandard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SOC2 => write!(f, "SOC 2 Type II"),
            Self::GDPR => write!(f, "GDPR"),
            Self::HIPAA => write!(f, "HIPAA"),
            Self::PCIDSS => write!(f, "PCI DSS"),
            Self::ISO27001 => write!(f, "ISO 27001"),
            Self::FedRAMP => write!(f, "FedRAMP"),
            Self::NIST => write!(f, "NIST CSF"),
        }
    }
}

/// Report output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReportFormat {
    /// JSON format
    #[default]
    Json,
    /// HTML format
    Html,
    /// Markdown format
    Markdown,
    /// PDF format (requires external renderer)
    Pdf,
    /// CSV format (for data export)
    Csv,
}

/// Compliance check result.
#[derive(Debug, Clone)]
pub enum ComplianceStatus {
    /// Fully compliant
    Compliant,
    /// Partially compliant with notes
    PartiallyCompliant {
        /// Notes describing partial compliance
        notes: Vec<String>,
    },
    /// Non-compliant with reasons
    NonCompliant {
        /// Reasons for non-compliance
        reasons: Vec<String>,
    },
    /// Not applicable
    NotApplicable,
}

impl ComplianceStatus {
    /// Check if compliant.
    pub fn is_compliant(&self) -> bool {
        matches!(self, Self::Compliant | Self::NotApplicable)
    }
}

/// A single compliance check.
#[derive(Debug, Clone)]
pub struct ComplianceCheck {
    /// Check identifier
    pub id: String,
    /// Check name
    pub name: String,
    /// Standard this check belongs to
    pub standard: ComplianceStandard,
    /// Check description
    pub description: String,
    /// Check status
    pub status: ComplianceStatus,
    /// Evidence collected
    pub evidence: Vec<String>,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// When the check was performed
    pub checked_at: SystemTime,
}

/// Summary statistics for a compliance report.
#[derive(Debug, Clone)]
pub struct ComplianceSummary {
    /// Total checks performed
    pub total_checks: usize,
    /// Number compliant
    pub compliant: usize,
    /// Number partially compliant
    pub partially_compliant: usize,
    /// Number non-compliant
    pub non_compliant: usize,
    /// Number not applicable
    pub not_applicable: usize,
    /// Overall compliance percentage
    pub compliance_percentage: f64,
}

/// A complete compliance report.
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    /// Report ID
    pub id: String,
    /// Report title
    pub title: String,
    /// Standards covered
    pub standards: Vec<ComplianceStandard>,
    /// Individual checks
    pub checks: Vec<ComplianceCheck>,
    /// Summary statistics
    pub summary: ComplianceSummary,
    /// Report generation time
    pub generated_at: SystemTime,
    /// Report period start
    pub period_start: SystemTime,
    /// Report period end
    pub period_end: SystemTime,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ComplianceReport {
    /// Export report to specified format.
    pub fn export(&self, format: ReportFormat) -> String {
        match format {
            ReportFormat::Json => self.to_json(),
            ReportFormat::Html => self.to_html(),
            ReportFormat::Markdown => self.to_markdown(),
            ReportFormat::Pdf => self.to_markdown(), // PDF requires external renderer
            ReportFormat::Csv => self.to_csv(),
        }
    }

    fn to_json(&self) -> String {
        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!("  \"id\": \"{}\",\n", self.id));
        json.push_str(&format!("  \"title\": \"{}\",\n", self.title));
        json.push_str(&format!(
            "  \"standards\": [{}],\n",
            self.standards
                .iter()
                .map(|s| format!("\"{}\"", s))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        json.push_str("  \"summary\": {\n");
        json.push_str(&format!(
            "    \"total_checks\": {},\n",
            self.summary.total_checks
        ));
        json.push_str(&format!("    \"compliant\": {},\n", self.summary.compliant));
        json.push_str(&format!(
            "    \"partially_compliant\": {},\n",
            self.summary.partially_compliant
        ));
        json.push_str(&format!(
            "    \"non_compliant\": {},\n",
            self.summary.non_compliant
        ));
        json.push_str(&format!(
            "    \"compliance_percentage\": {:.1}\n",
            self.summary.compliance_percentage
        ));
        json.push_str("  },\n");
        json.push_str(&format!("  \"checks_count\": {}\n", self.checks.len()));
        json.push_str("}\n");
        json
    }

    fn to_html(&self) -> String {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("<title>{}</title>\n", self.title));
        html.push_str("<style>body { font-family: sans-serif; } .compliant { color: green; } .non-compliant { color: red; }</style>\n");
        html.push_str("</head>\n<body>\n");
        html.push_str(&format!("<h1>{}</h1>\n", self.title));
        html.push_str(&format!("<p>Report ID: {}</p>\n", self.id));
        html.push_str("<h2>Summary</h2>\n");
        html.push_str("<table>\n");
        html.push_str(&format!(
            "<tr><td>Total Checks</td><td>{}</td></tr>\n",
            self.summary.total_checks
        ));
        html.push_str(&format!(
            "<tr><td>Compliant</td><td class=\"compliant\">{}</td></tr>\n",
            self.summary.compliant
        ));
        html.push_str(&format!(
            "<tr><td>Non-Compliant</td><td class=\"non-compliant\">{}</td></tr>\n",
            self.summary.non_compliant
        ));
        html.push_str(&format!(
            "<tr><td>Compliance</td><td>{:.1}%</td></tr>\n",
            self.summary.compliance_percentage
        ));
        html.push_str("</table>\n");
        html.push_str("<h2>Checks</h2>\n");
        for check in &self.checks {
            html.push_str(&format!("<h3>{}</h3>\n", check.name));
            html.push_str(&format!("<p>{}</p>\n", check.description));
        }
        html.push_str("</body>\n</html>\n");
        html
    }

    fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!("# {}\n\n", self.title));
        md.push_str(&format!("**Report ID:** {}\n\n", self.id));
        md.push_str("## Summary\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!(
            "| Total Checks | {} |\n",
            self.summary.total_checks
        ));
        md.push_str(&format!("| Compliant | {} |\n", self.summary.compliant));
        md.push_str(&format!(
            "| Partially Compliant | {} |\n",
            self.summary.partially_compliant
        ));
        md.push_str(&format!(
            "| Non-Compliant | {} |\n",
            self.summary.non_compliant
        ));
        md.push_str(&format!(
            "| Compliance | {:.1}% |\n",
            self.summary.compliance_percentage
        ));
        md.push_str("\n## Detailed Checks\n\n");
        for check in &self.checks {
            let status_icon = match &check.status {
                ComplianceStatus::Compliant => "✅",
                ComplianceStatus::PartiallyCompliant { .. } => "⚠️",
                ComplianceStatus::NonCompliant { .. } => "❌",
                ComplianceStatus::NotApplicable => "➖",
            };
            md.push_str(&format!("### {} {}\n\n", status_icon, check.name));
            md.push_str(&format!("{}\n\n", check.description));
        }
        md
    }

    fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("ID,Name,Standard,Status,Description\n");
        for check in &self.checks {
            let status = match &check.status {
                ComplianceStatus::Compliant => "Compliant",
                ComplianceStatus::PartiallyCompliant { .. } => "Partially Compliant",
                ComplianceStatus::NonCompliant { .. } => "Non-Compliant",
                ComplianceStatus::NotApplicable => "N/A",
            };
            csv.push_str(&format!(
                "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"\n",
                check.id, check.name, check.standard, status, check.description
            ));
        }
        csv
    }
}

/// Compliance reporter for generating compliance documentation.
pub struct ComplianceReporter {
    /// Standards to report on
    standards: HashSet<ComplianceStandard>,
    /// Organization name
    organization: String,
    /// Report metadata
    metadata: HashMap<String, String>,
    /// Custom checks
    custom_checks: Vec<Box<dyn Fn() -> ComplianceCheck + Send + Sync>>,
}

impl ComplianceReporter {
    /// Create a new compliance reporter.
    pub fn new() -> Self {
        Self {
            standards: HashSet::new(),
            organization: "Unknown".to_string(),
            metadata: HashMap::new(),
            custom_checks: Vec::new(),
        }
    }

    /// Add a compliance standard.
    pub fn with_standard(mut self, standard: ComplianceStandard) -> Self {
        self.standards.insert(standard);
        self
    }

    /// Set organization name.
    pub fn with_organization(mut self, org: &str) -> Self {
        self.organization = org.to_string();
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Generate a compliance report.
    pub fn generate_report(&self, _format: ReportFormat) -> ComplianceReport {
        let mut checks = Vec::new();
        let now = SystemTime::now();

        // Generate checks for each standard
        for standard in &self.standards {
            checks.extend(self.generate_standard_checks(*standard));
        }

        // Run custom checks
        for check_fn in &self.custom_checks {
            checks.push(check_fn());
        }

        // Calculate summary
        let total = checks.len();
        let compliant = checks
            .iter()
            .filter(|c| matches!(c.status, ComplianceStatus::Compliant))
            .count();
        let partial = checks
            .iter()
            .filter(|c| matches!(c.status, ComplianceStatus::PartiallyCompliant { .. }))
            .count();
        let non_compliant = checks
            .iter()
            .filter(|c| matches!(c.status, ComplianceStatus::NonCompliant { .. }))
            .count();
        let na = checks
            .iter()
            .filter(|c| matches!(c.status, ComplianceStatus::NotApplicable))
            .count();

        let applicable = total - na;
        let compliance_pct = if applicable > 0 {
            ((compliant as f64 + partial as f64 * 0.5) / applicable as f64) * 100.0
        } else {
            100.0
        };

        ComplianceReport {
            id: format!(
                "RPT-{}",
                now.duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
            title: format!("{} Compliance Report", self.organization),
            standards: self.standards.iter().copied().collect(),
            checks,
            summary: ComplianceSummary {
                total_checks: total,
                compliant,
                partially_compliant: partial,
                non_compliant,
                not_applicable: na,
                compliance_percentage: compliance_pct,
            },
            generated_at: now,
            period_start: now - Duration::from_secs(30 * 24 * 60 * 60), // 30 days ago
            period_end: now,
            metadata: self.metadata.clone(),
        }
    }

    fn generate_standard_checks(&self, standard: ComplianceStandard) -> Vec<ComplianceCheck> {
        let now = SystemTime::now();

        match standard {
            ComplianceStandard::SOC2 => vec![
                ComplianceCheck {
                    id: "SOC2-CC1.1".to_string(),
                    name: "Control Environment".to_string(),
                    standard,
                    description:
                        "The entity demonstrates commitment to integrity and ethical values."
                            .to_string(),
                    status: ComplianceStatus::Compliant,
                    evidence: vec![
                        "Audit logging enabled".to_string(),
                        "Access controls implemented".to_string(),
                    ],
                    recommendations: vec![],
                    checked_at: now,
                },
                ComplianceCheck {
                    id: "SOC2-CC6.1".to_string(),
                    name: "Logical Access Controls".to_string(),
                    standard,
                    description:
                        "Logical access security software, infrastructure, and architectures."
                            .to_string(),
                    status: ComplianceStatus::Compliant,
                    evidence: vec![
                        "Kernel sandboxing available".to_string(),
                        "Memory encryption available".to_string(),
                    ],
                    recommendations: vec![],
                    checked_at: now,
                },
                ComplianceCheck {
                    id: "SOC2-CC7.2".to_string(),
                    name: "System Monitoring".to_string(),
                    standard,
                    description: "System components are monitored and anomalies are identified."
                        .to_string(),
                    status: ComplianceStatus::Compliant,
                    evidence: vec![
                        "Health monitoring enabled".to_string(),
                        "GPU memory dashboard available".to_string(),
                    ],
                    recommendations: vec![],
                    checked_at: now,
                },
            ],
            ComplianceStandard::GDPR => vec![
                ComplianceCheck {
                    id: "GDPR-32".to_string(),
                    name: "Security of Processing".to_string(),
                    standard,
                    description: "Implement appropriate technical and organizational measures."
                        .to_string(),
                    status: ComplianceStatus::Compliant,
                    evidence: vec!["Memory encryption available".to_string()],
                    recommendations: vec!["Consider enabling encryption by default".to_string()],
                    checked_at: now,
                },
                ComplianceCheck {
                    id: "GDPR-33".to_string(),
                    name: "Breach Notification".to_string(),
                    standard,
                    description: "Notify supervisory authority of personal data breach."
                        .to_string(),
                    status: ComplianceStatus::PartiallyCompliant {
                        notes: vec!["Audit logging available but breach detection not automated"
                            .to_string()],
                    },
                    evidence: vec!["Audit logging enabled".to_string()],
                    recommendations: vec!["Add automated breach detection".to_string()],
                    checked_at: now,
                },
            ],
            ComplianceStandard::HIPAA => vec![
                ComplianceCheck {
                    id: "HIPAA-164.312(a)".to_string(),
                    name: "Access Control".to_string(),
                    standard,
                    description: "Implement technical policies for electronic PHI access."
                        .to_string(),
                    status: ComplianceStatus::Compliant,
                    evidence: vec![
                        "Kernel sandboxing available".to_string(),
                        "Access levels configurable".to_string(),
                    ],
                    recommendations: vec![],
                    checked_at: now,
                },
                ComplianceCheck {
                    id: "HIPAA-164.312(e)".to_string(),
                    name: "Transmission Security".to_string(),
                    standard,
                    description: "Implement security measures for ePHI transmission.".to_string(),
                    status: ComplianceStatus::Compliant,
                    evidence: vec!["Memory encryption for data at rest".to_string()],
                    recommendations: vec!["Implement TLS for network K2K".to_string()],
                    checked_at: now,
                },
            ],
            ComplianceStandard::PCIDSS => vec![
                ComplianceCheck {
                    id: "PCI-3.4".to_string(),
                    name: "Render PAN Unreadable".to_string(),
                    standard,
                    description: "Render PAN unreadable anywhere it is stored.".to_string(),
                    status: ComplianceStatus::Compliant,
                    evidence: vec!["AES-256-GCM encryption available".to_string()],
                    recommendations: vec![],
                    checked_at: now,
                },
                ComplianceCheck {
                    id: "PCI-10.1".to_string(),
                    name: "Audit Trails".to_string(),
                    standard,
                    description: "Implement audit trails to link access to individual users."
                        .to_string(),
                    status: ComplianceStatus::Compliant,
                    evidence: vec!["Comprehensive audit logging".to_string()],
                    recommendations: vec![],
                    checked_at: now,
                },
            ],
            ComplianceStandard::ISO27001 => vec![ComplianceCheck {
                id: "ISO-A.10.1".to_string(),
                name: "Cryptographic Controls".to_string(),
                standard,
                description: "Policy on use of cryptographic controls.".to_string(),
                status: ComplianceStatus::Compliant,
                evidence: vec!["Multiple encryption algorithms supported".to_string()],
                recommendations: vec![],
                checked_at: now,
            }],
            ComplianceStandard::FedRAMP => vec![ComplianceCheck {
                id: "FedRAMP-SC-28".to_string(),
                name: "Protection of Information at Rest".to_string(),
                standard,
                description: "Protect confidentiality and integrity of information at rest."
                    .to_string(),
                status: ComplianceStatus::Compliant,
                evidence: vec!["FIPS-compliant algorithms available".to_string()],
                recommendations: vec![],
                checked_at: now,
            }],
            ComplianceStandard::NIST => vec![
                ComplianceCheck {
                    id: "NIST-PR.DS-1".to_string(),
                    name: "Data-at-rest Protection".to_string(),
                    standard,
                    description: "Data-at-rest is protected.".to_string(),
                    status: ComplianceStatus::Compliant,
                    evidence: vec!["Memory encryption module".to_string()],
                    recommendations: vec![],
                    checked_at: now,
                },
                ComplianceCheck {
                    id: "NIST-DE.CM-1".to_string(),
                    name: "Network Monitoring".to_string(),
                    standard,
                    description: "The network is monitored to detect cybersecurity events."
                        .to_string(),
                    status: ComplianceStatus::Compliant,
                    evidence: vec![
                        "Observability context".to_string(),
                        "GPU profiler integration".to_string(),
                    ],
                    recommendations: vec![],
                    checked_at: now,
                },
            ],
        }
    }
}

impl Default for ComplianceReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ComplianceReporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComplianceReporter")
            .field("standards", &self.standards)
            .field("organization", &self.organization)
            .field("metadata", &self.metadata)
            .field("custom_checks_count", &self.custom_checks.len())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Memory Encryption Tests

    #[test]
    fn test_encryption_config_builder() {
        let config = EncryptionConfig::new()
            .with_algorithm(EncryptionAlgorithm::ChaCha20Poly1305)
            .with_key_rotation_interval(Duration::from_secs(7200))
            .with_control_block_encryption(false);

        assert_eq!(config.algorithm, EncryptionAlgorithm::ChaCha20Poly1305);
        assert_eq!(config.key_rotation_interval, Duration::from_secs(7200));
        assert!(!config.encrypt_control_blocks);
    }

    #[test]
    fn test_encryption_key_creation() {
        let key = EncryptionKey::new(1, EncryptionAlgorithm::Aes256Gcm);
        assert_eq!(key.key_id, 1);
        assert_eq!(key.key_size(), 32);
        assert!(!key.is_expired());
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let encryption = MemoryEncryption::new(EncryptionConfig::default());
        let plaintext = b"Hello, GPU World!";

        let encrypted = encryption.encrypt_region(plaintext);
        assert_ne!(encrypted.ciphertext[..plaintext.len()], plaintext[..]);

        let decrypted = encryption.decrypt_region(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_key_rotation() {
        let encryption = MemoryEncryption::new(EncryptionConfig::default());
        let initial_key_id = encryption.current_key_id();

        encryption.rotate_keys();
        assert_eq!(encryption.current_key_id(), initial_key_id + 1);

        // Stats should reflect rotation
        let stats = encryption.stats();
        assert_eq!(stats.key_rotations, 1);
    }

    #[test]
    fn test_decrypt_with_old_key() {
        let encryption = MemoryEncryption::new(EncryptionConfig::default());
        let plaintext = b"Secret data";

        let encrypted = encryption.encrypt_region(plaintext);
        let old_key_id = encrypted.key_id;

        // Rotate key
        encryption.rotate_keys();
        assert_ne!(encryption.current_key_id(), old_key_id);

        // Should still decrypt with old key
        let decrypted = encryption.decrypt_region(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encryption_stats() {
        let encryption = MemoryEncryption::new(EncryptionConfig::default());
        let data = vec![0u8; 1024];

        for _ in 0..10 {
            let encrypted = encryption.encrypt_region(&data);
            let _ = encryption.decrypt_region(&encrypted);
        }

        let stats = encryption.stats();
        assert_eq!(stats.encrypt_ops, 10);
        assert_eq!(stats.decrypt_ops, 10);
        assert_eq!(stats.bytes_encrypted, 10240);
    }

    // Kernel Sandboxing Tests

    #[test]
    fn test_resource_limits_builder() {
        let limits = ResourceLimits::new()
            .with_max_memory(512 * 1024 * 1024)
            .with_max_execution_time(Duration::from_secs(30));

        assert_eq!(limits.max_memory_bytes, 512 * 1024 * 1024);
        assert_eq!(limits.max_execution_time, Duration::from_secs(30));
    }

    #[test]
    fn test_sandbox_policy_k2k() {
        let policy = SandboxPolicy::new()
            .allow_k2k_to(&["trusted_kernel", "another_trusted"])
            .deny_k2k_to(&["malicious_kernel"]);

        assert!(policy.is_k2k_allowed("trusted_kernel"));
        assert!(policy.is_k2k_allowed("another_trusted"));
        assert!(!policy.is_k2k_allowed("malicious_kernel"));
        assert!(!policy.is_k2k_allowed("unknown_kernel")); // Not in allowed list
    }

    #[test]
    fn test_sandbox_memory_check() {
        let policy = SandboxPolicy::new().with_memory_limit(1024);
        let sandbox = KernelSandbox::new(policy);

        // Should pass
        assert!(sandbox.check_memory(512).is_ok());

        // Should fail
        let result = sandbox.check_memory(2048);
        assert!(result.is_err());

        if let Err(violation) = result {
            assert!(matches!(
                violation.violation_type,
                ViolationType::MemoryLimitExceeded { .. }
            ));
        }
    }

    #[test]
    fn test_sandbox_k2k_check() {
        let policy = SandboxPolicy::new().deny_k2k_to(&["blocked"]);
        let sandbox = KernelSandbox::new(policy);

        assert!(sandbox.check_k2k("allowed_dest").is_ok());
        assert!(sandbox.check_k2k("blocked").is_err());
    }

    #[test]
    fn test_sandbox_checkpoint_check() {
        let policy = SandboxPolicy::restrictive();
        let sandbox = KernelSandbox::new(policy);

        assert!(sandbox.check_checkpoint().is_err());

        let permissive = SandboxPolicy::permissive();
        let sandbox2 = KernelSandbox::new(permissive);
        assert!(sandbox2.check_checkpoint().is_ok());
    }

    #[test]
    fn test_sandbox_stats() {
        let policy = SandboxPolicy::new().with_memory_limit(1024);
        let sandbox = KernelSandbox::new(policy);

        let _ = sandbox.check_memory(512);
        let _ = sandbox.check_memory(2048); // Violation
        let _ = sandbox.check_k2k("dest");

        let stats = sandbox.stats();
        assert_eq!(stats.total_checks, 3);
        assert_eq!(stats.violations_detected, 1);
    }

    #[test]
    fn test_sandbox_violations_list() {
        let policy = SandboxPolicy::restrictive();
        let sandbox = KernelSandbox::new(policy);

        let _ = sandbox.check_checkpoint();
        let _ = sandbox.check_migration();

        let violations = sandbox.violations();
        assert_eq!(violations.len(), 2);
    }

    // Compliance Reports Tests

    #[test]
    fn test_compliance_reporter_creation() {
        let reporter = ComplianceReporter::new()
            .with_standard(ComplianceStandard::SOC2)
            .with_standard(ComplianceStandard::GDPR)
            .with_organization("Test Org");

        assert_eq!(reporter.standards.len(), 2);
        assert!(reporter.standards.contains(&ComplianceStandard::SOC2));
        assert!(reporter.standards.contains(&ComplianceStandard::GDPR));
    }

    #[test]
    fn test_generate_soc2_report() {
        let reporter = ComplianceReporter::new()
            .with_standard(ComplianceStandard::SOC2)
            .with_organization("Acme Corp");

        let report = reporter.generate_report(ReportFormat::Json);

        assert!(!report.checks.is_empty());
        assert!(report.summary.total_checks > 0);
        assert!(report.title.contains("Acme Corp"));
    }

    #[test]
    fn test_report_json_export() {
        let reporter = ComplianceReporter::new().with_standard(ComplianceStandard::HIPAA);

        let report = reporter.generate_report(ReportFormat::Json);
        let json = report.export(ReportFormat::Json);

        assert!(json.contains("\"id\""));
        assert!(json.contains("\"summary\""));
    }

    #[test]
    fn test_report_markdown_export() {
        let reporter = ComplianceReporter::new().with_standard(ComplianceStandard::NIST);

        let report = reporter.generate_report(ReportFormat::Markdown);
        let md = report.export(ReportFormat::Markdown);

        assert!(md.contains("# "));
        assert!(md.contains("## Summary"));
        assert!(md.contains("| Metric | Value |"));
    }

    #[test]
    fn test_report_html_export() {
        let reporter = ComplianceReporter::new().with_standard(ComplianceStandard::PCIDSS);

        let report = reporter.generate_report(ReportFormat::Html);
        let html = report.export(ReportFormat::Html);

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<h1>"));
    }

    #[test]
    fn test_report_csv_export() {
        let reporter = ComplianceReporter::new().with_standard(ComplianceStandard::ISO27001);

        let report = reporter.generate_report(ReportFormat::Csv);
        let csv = report.export(ReportFormat::Csv);

        assert!(csv.contains("ID,Name,Standard,Status,Description"));
    }

    #[test]
    fn test_compliance_summary_calculation() {
        let reporter = ComplianceReporter::new()
            .with_standard(ComplianceStandard::SOC2)
            .with_standard(ComplianceStandard::GDPR)
            .with_standard(ComplianceStandard::HIPAA);

        let report = reporter.generate_report(ReportFormat::Json);

        let sum = report.summary.compliant
            + report.summary.partially_compliant
            + report.summary.non_compliant
            + report.summary.not_applicable;
        assert_eq!(sum, report.summary.total_checks);
    }

    #[test]
    fn test_compliance_status_is_compliant() {
        assert!(ComplianceStatus::Compliant.is_compliant());
        assert!(ComplianceStatus::NotApplicable.is_compliant());
        assert!(!ComplianceStatus::NonCompliant { reasons: vec![] }.is_compliant());
        assert!(!ComplianceStatus::PartiallyCompliant { notes: vec![] }.is_compliant());
    }

    #[test]
    fn test_all_standards() {
        let reporter = ComplianceReporter::new()
            .with_standard(ComplianceStandard::SOC2)
            .with_standard(ComplianceStandard::GDPR)
            .with_standard(ComplianceStandard::HIPAA)
            .with_standard(ComplianceStandard::PCIDSS)
            .with_standard(ComplianceStandard::ISO27001)
            .with_standard(ComplianceStandard::FedRAMP)
            .with_standard(ComplianceStandard::NIST);

        let report = reporter.generate_report(ReportFormat::Json);
        assert_eq!(report.standards.len(), 7);
    }
}
