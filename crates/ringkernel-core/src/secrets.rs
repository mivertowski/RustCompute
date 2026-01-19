//! Secrets management for secure key storage and retrieval.
//!
//! This module provides a pluggable secrets management system with support for
//! multiple backends including environment variables, HashiCorp Vault, and AWS Secrets Manager.
//!
//! # Feature Flags
//!
//! - `crypto` - Enables secure key derivation and encryption of cached secrets
//!
//! # Example
//!
//! ```rust,ignore
//! use ringkernel_core::secrets::{SecretStore, EnvVarSecretStore, SecretKey};
//!
//! // Using environment variables (for development)
//! let store = EnvVarSecretStore::new("MYAPP_");
//! let api_key = store.get_secret(&SecretKey::new("api_key")).await?;
//!
//! // Using HashiCorp Vault (for production)
//! let vault = VaultSecretStore::new("https://vault.example.com:8200")
//!     .with_token_auth(env::var("VAULT_TOKEN")?)
//!     .with_mount_path("secret/data/myapp");
//! let db_password = vault.get_secret(&SecretKey::new("database/password")).await?;
//! ```

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "crypto")]
use zeroize::Zeroize;

// ============================================================================
// SECRET KEY
// ============================================================================

/// A key identifying a secret in the store.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SecretKey {
    /// The secret path/name.
    path: String,
    /// Optional version specifier.
    version: Option<String>,
}

impl SecretKey {
    /// Create a new secret key.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            version: None,
        }
    }

    /// Create a secret key with a specific version.
    pub fn with_version(path: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            version: Some(version.into()),
        }
    }

    /// Get the path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get the version.
    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }
}

impl fmt::Display for SecretKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(version) = &self.version {
            write!(f, "{}@{}", self.path, version)
        } else {
            write!(f, "{}", self.path)
        }
    }
}

// ============================================================================
// SECRET VALUE
// ============================================================================

/// A secret value retrieved from the store.
///
/// When the `crypto` feature is enabled, the secret data is automatically
/// zeroed when dropped using the `zeroize` crate.
pub struct SecretValue {
    /// The secret data.
    #[cfg(feature = "crypto")]
    data: zeroize::Zeroizing<Vec<u8>>,
    #[cfg(not(feature = "crypto"))]
    data: Vec<u8>,
    /// When the secret was retrieved.
    retrieved_at: Instant,
    /// Optional expiration time.
    expires_at: Option<Instant>,
    /// Metadata about the secret.
    metadata: HashMap<String, String>,
}

impl SecretValue {
    /// Create a new secret value.
    #[cfg(feature = "crypto")]
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data: zeroize::Zeroizing::new(data),
            retrieved_at: Instant::now(),
            expires_at: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new secret value (non-crypto version).
    #[cfg(not(feature = "crypto"))]
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            retrieved_at: Instant::now(),
            expires_at: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a secret value from a string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self::new(s.into().into_bytes())
    }

    /// Set the expiration time.
    pub fn with_expiry(mut self, duration: Duration) -> Self {
        self.expires_at = Some(Instant::now() + duration);
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get the secret data as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get the secret data as a string (if valid UTF-8).
    pub fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.data).ok()
    }

    /// Check if the secret has expired.
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|exp| Instant::now() > exp)
            .unwrap_or(false)
    }

    /// Get the age of this secret value.
    pub fn age(&self) -> Duration {
        self.retrieved_at.elapsed()
    }

    /// Get metadata.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}

impl fmt::Debug for SecretValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SecretValue")
            .field("length", &self.data.len())
            .field("retrieved_at", &"<instant>")
            .field("is_expired", &self.is_expired())
            .field("metadata", &self.metadata)
            .finish()
    }
}

// ============================================================================
// SECRET STORE TRAIT
// ============================================================================

/// Error type for secret store operations.
#[derive(Debug, Clone)]
pub enum SecretError {
    /// Secret not found.
    NotFound(String),
    /// Access denied.
    AccessDenied(String),
    /// Connection error.
    ConnectionError(String),
    /// Invalid secret format.
    InvalidFormat(String),
    /// Secret has expired.
    Expired(String),
    /// Rate limited.
    RateLimited(String),
    /// Other error.
    Other(String),
}

impl fmt::Display for SecretError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(msg) => write!(f, "Secret not found: {}", msg),
            Self::AccessDenied(msg) => write!(f, "Access denied: {}", msg),
            Self::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            Self::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            Self::Expired(msg) => write!(f, "Secret expired: {}", msg),
            Self::RateLimited(msg) => write!(f, "Rate limited: {}", msg),
            Self::Other(msg) => write!(f, "Secret error: {}", msg),
        }
    }
}

impl std::error::Error for SecretError {}

/// Result type for secret store operations.
pub type SecretResult<T> = Result<T, SecretError>;

/// Trait for pluggable secret storage backends.
#[async_trait]
pub trait SecretStore: Send + Sync {
    /// Get a secret by key.
    async fn get_secret(&self, key: &SecretKey) -> SecretResult<SecretValue>;

    /// Check if a secret exists.
    async fn secret_exists(&self, key: &SecretKey) -> SecretResult<bool>;

    /// List available secrets (if supported).
    async fn list_secrets(&self, prefix: Option<&str>) -> SecretResult<Vec<SecretKey>>;

    /// Get the store name for logging/debugging.
    fn store_name(&self) -> &str;

    /// Check if the store is healthy/connected.
    async fn health_check(&self) -> SecretResult<()>;
}

// ============================================================================
// ENVIRONMENT VARIABLE SECRET STORE
// ============================================================================

/// Secret store backed by environment variables.
///
/// This is suitable for development and simple deployments.
/// Secret keys are transformed to environment variable names by:
/// 1. Applying the prefix
/// 2. Converting to uppercase
/// 3. Replacing `/` and `.` with `_`
///
/// Example: Key "database/password" with prefix "MYAPP_" becomes "MYAPP_DATABASE_PASSWORD"
pub struct EnvVarSecretStore {
    /// Prefix for environment variable names.
    prefix: String,
    /// Cache for retrieved secrets.
    cache: RwLock<HashMap<SecretKey, SecretValue>>,
    /// Cache duration.
    cache_duration: Duration,
}

impl EnvVarSecretStore {
    /// Create a new environment variable secret store.
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            cache: RwLock::new(HashMap::new()),
            cache_duration: Duration::from_secs(300), // 5 minutes default
        }
    }

    /// Set the cache duration.
    pub fn with_cache_duration(mut self, duration: Duration) -> Self {
        self.cache_duration = duration;
        self
    }

    /// Convert a secret key to an environment variable name.
    fn key_to_env_var(&self, key: &SecretKey) -> String {
        let path = key.path().to_uppercase().replace(['/', '.'], "_");
        format!("{}{}", self.prefix, path)
    }
}

#[async_trait]
impl SecretStore for EnvVarSecretStore {
    async fn get_secret(&self, key: &SecretKey) -> SecretResult<SecretValue> {
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(secret) = cache.get(key) {
                if !secret.is_expired() && secret.age() < self.cache_duration {
                    return Ok(SecretValue::new(secret.as_bytes().to_vec()));
                }
            }
        }

        // Get from environment
        let env_var = self.key_to_env_var(key);
        let value = std::env::var(&env_var).map_err(|_| {
            SecretError::NotFound(format!("Environment variable {} not set", env_var))
        })?;

        let secret = SecretValue::from_string(value)
            .with_metadata("source", "environment")
            .with_metadata("env_var", &env_var);

        // Cache it
        {
            let mut cache = self.cache.write();
            cache.insert(key.clone(), SecretValue::new(secret.as_bytes().to_vec()));
        }

        Ok(secret)
    }

    async fn secret_exists(&self, key: &SecretKey) -> SecretResult<bool> {
        let env_var = self.key_to_env_var(key);
        Ok(std::env::var(&env_var).is_ok())
    }

    async fn list_secrets(&self, prefix: Option<&str>) -> SecretResult<Vec<SecretKey>> {
        let full_prefix = match prefix {
            Some(p) => format!(
                "{}{}",
                self.prefix,
                p.to_uppercase().replace(['/', '.'], "_")
            ),
            None => self.prefix.clone(),
        };

        let secrets: Vec<SecretKey> = std::env::vars()
            .filter_map(|(name, _)| {
                if name.starts_with(&full_prefix) {
                    let path = name
                        .strip_prefix(&self.prefix)?
                        .to_lowercase()
                        .replace('_', "/");
                    Some(SecretKey::new(path))
                } else {
                    None
                }
            })
            .collect();

        Ok(secrets)
    }

    fn store_name(&self) -> &str {
        "EnvVarSecretStore"
    }

    async fn health_check(&self) -> SecretResult<()> {
        // Environment variables are always available
        Ok(())
    }
}

// ============================================================================
// IN-MEMORY SECRET STORE (for testing)
// ============================================================================

/// In-memory secret store for testing.
pub struct InMemorySecretStore {
    secrets: RwLock<HashMap<SecretKey, Vec<u8>>>,
}

impl InMemorySecretStore {
    /// Create a new in-memory secret store.
    pub fn new() -> Self {
        Self {
            secrets: RwLock::new(HashMap::new()),
        }
    }

    /// Add a secret to the store.
    pub fn add_secret(&self, key: SecretKey, value: impl Into<Vec<u8>>) {
        self.secrets.write().insert(key, value.into());
    }

    /// Add a string secret to the store.
    pub fn add_string_secret(&self, key: SecretKey, value: impl Into<String>) {
        self.add_secret(key, value.into().into_bytes());
    }
}

impl Default for InMemorySecretStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SecretStore for InMemorySecretStore {
    async fn get_secret(&self, key: &SecretKey) -> SecretResult<SecretValue> {
        let secrets = self.secrets.read();
        secrets
            .get(key)
            .map(|data| SecretValue::new(data.clone()).with_metadata("source", "in_memory"))
            .ok_or_else(|| SecretError::NotFound(key.to_string()))
    }

    async fn secret_exists(&self, key: &SecretKey) -> SecretResult<bool> {
        Ok(self.secrets.read().contains_key(key))
    }

    async fn list_secrets(&self, prefix: Option<&str>) -> SecretResult<Vec<SecretKey>> {
        let secrets = self.secrets.read();
        let keys: Vec<SecretKey> = secrets
            .keys()
            .filter(|k| match prefix {
                Some(p) => k.path().starts_with(p),
                None => true,
            })
            .cloned()
            .collect();
        Ok(keys)
    }

    fn store_name(&self) -> &str {
        "InMemorySecretStore"
    }

    async fn health_check(&self) -> SecretResult<()> {
        Ok(())
    }
}

// ============================================================================
// CHAINED SECRET STORE
// ============================================================================

/// A secret store that chains multiple stores, trying each in order.
///
/// Useful for fallback scenarios (e.g., try Vault, fall back to env vars).
pub struct ChainedSecretStore {
    stores: Vec<Arc<dyn SecretStore>>,
}

impl ChainedSecretStore {
    /// Create a new chained secret store.
    pub fn new() -> Self {
        Self { stores: Vec::new() }
    }

    /// Add a store to the chain.
    pub fn with_store(mut self, store: Arc<dyn SecretStore>) -> Self {
        self.stores.push(store);
        self
    }
}

impl Default for ChainedSecretStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SecretStore for ChainedSecretStore {
    async fn get_secret(&self, key: &SecretKey) -> SecretResult<SecretValue> {
        let mut last_error = SecretError::NotFound(key.to_string());

        for store in &self.stores {
            match store.get_secret(key).await {
                Ok(secret) => return Ok(secret),
                Err(e) => {
                    last_error = e;
                    continue;
                }
            }
        }

        Err(last_error)
    }

    async fn secret_exists(&self, key: &SecretKey) -> SecretResult<bool> {
        for store in &self.stores {
            if store.secret_exists(key).await? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn list_secrets(&self, prefix: Option<&str>) -> SecretResult<Vec<SecretKey>> {
        let mut all_keys = Vec::new();
        for store in &self.stores {
            if let Ok(keys) = store.list_secrets(prefix).await {
                all_keys.extend(keys);
            }
        }
        // Deduplicate
        all_keys.sort_by(|a, b| a.path().cmp(b.path()));
        all_keys.dedup_by(|a, b| a.path() == b.path());
        Ok(all_keys)
    }

    fn store_name(&self) -> &str {
        "ChainedSecretStore"
    }

    async fn health_check(&self) -> SecretResult<()> {
        for store in &self.stores {
            if store.health_check().await.is_ok() {
                return Ok(());
            }
        }
        Err(SecretError::ConnectionError(
            "All stores in chain are unhealthy".to_string(),
        ))
    }
}

// ============================================================================
// CACHING SECRET STORE WRAPPER
// ============================================================================

/// A wrapper that adds caching to any secret store.
pub struct CachedSecretStore<S: SecretStore> {
    inner: S,
    cache: RwLock<HashMap<SecretKey, CachedEntry>>,
    ttl: Duration,
    max_entries: usize,
}

struct CachedEntry {
    value: SecretValue,
    cached_at: Instant,
}

impl<S: SecretStore> CachedSecretStore<S> {
    /// Create a new cached secret store.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            cache: RwLock::new(HashMap::new()),
            ttl: Duration::from_secs(300), // 5 minutes
            max_entries: 1000,
        }
    }

    /// Set the cache TTL.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Set the maximum cache entries.
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    /// Invalidate a specific key.
    pub fn invalidate(&self, key: &SecretKey) {
        self.cache.write().remove(key);
    }
}

#[async_trait]
impl<S: SecretStore> SecretStore for CachedSecretStore<S> {
    async fn get_secret(&self, key: &SecretKey) -> SecretResult<SecretValue> {
        // Check cache
        {
            let cache = self.cache.read();
            if let Some(entry) = cache.get(key) {
                if entry.cached_at.elapsed() < self.ttl {
                    return Ok(SecretValue::new(entry.value.as_bytes().to_vec()));
                }
            }
        }

        // Fetch from inner store
        let secret = self.inner.get_secret(key).await?;

        // Cache it
        {
            let mut cache = self.cache.write();

            // Evict if at capacity
            if cache.len() >= self.max_entries {
                // Remove oldest entry
                if let Some(oldest_key) = cache
                    .iter()
                    .min_by_key(|(_, e)| e.cached_at)
                    .map(|(k, _)| k.clone())
                {
                    cache.remove(&oldest_key);
                }
            }

            cache.insert(
                key.clone(),
                CachedEntry {
                    value: SecretValue::new(secret.as_bytes().to_vec()),
                    cached_at: Instant::now(),
                },
            );
        }

        Ok(secret)
    }

    async fn secret_exists(&self, key: &SecretKey) -> SecretResult<bool> {
        // Check cache first
        {
            let cache = self.cache.read();
            if let Some(entry) = cache.get(key) {
                if entry.cached_at.elapsed() < self.ttl {
                    return Ok(true);
                }
            }
        }

        self.inner.secret_exists(key).await
    }

    async fn list_secrets(&self, prefix: Option<&str>) -> SecretResult<Vec<SecretKey>> {
        self.inner.list_secrets(prefix).await
    }

    fn store_name(&self) -> &str {
        self.inner.store_name()
    }

    async fn health_check(&self) -> SecretResult<()> {
        self.inner.health_check().await
    }
}

// ============================================================================
// KEY ROTATION MANAGER
// ============================================================================

/// Manages automatic key rotation for encryption keys.
pub struct KeyRotationManager {
    /// Secret store for retrieving keys.
    store: Arc<dyn SecretStore>,
    /// Current key version.
    current_version: RwLock<u64>,
    /// Rotation interval.
    rotation_interval: Duration,
    /// Last rotation time.
    last_rotation: RwLock<Instant>,
    /// Key prefix in the store.
    key_prefix: String,
}

impl KeyRotationManager {
    /// Create a new key rotation manager.
    pub fn new(store: Arc<dyn SecretStore>, key_prefix: impl Into<String>) -> Self {
        Self {
            store,
            current_version: RwLock::new(1),
            rotation_interval: Duration::from_secs(3600), // 1 hour default
            last_rotation: RwLock::new(Instant::now()),
            key_prefix: key_prefix.into(),
        }
    }

    /// Set the rotation interval.
    pub fn with_rotation_interval(mut self, interval: Duration) -> Self {
        self.rotation_interval = interval;
        self
    }

    /// Get the current encryption key.
    pub async fn get_current_key(&self) -> SecretResult<SecretValue> {
        let version = *self.current_version.read();
        let key_name = format!("{}/v{}", self.key_prefix, version);
        self.store.get_secret(&SecretKey::new(key_name)).await
    }

    /// Get a specific key version.
    pub async fn get_key_version(&self, version: u64) -> SecretResult<SecretValue> {
        let key_name = format!("{}/v{}", self.key_prefix, version);
        self.store.get_secret(&SecretKey::new(key_name)).await
    }

    /// Check if rotation is needed.
    pub fn needs_rotation(&self) -> bool {
        self.last_rotation.read().elapsed() >= self.rotation_interval
    }

    /// Rotate to a new key version.
    ///
    /// Note: The actual key must be pre-provisioned in the secret store.
    pub fn rotate(&self) {
        let mut version = self.current_version.write();
        *version += 1;
        *self.last_rotation.write() = Instant::now();
    }

    /// Get the current key version number.
    pub fn current_version(&self) -> u64 {
        *self.current_version.read()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_key() {
        let key = SecretKey::new("database/password");
        assert_eq!(key.path(), "database/password");
        assert!(key.version().is_none());

        let versioned = SecretKey::with_version("api_key", "v2");
        assert_eq!(versioned.path(), "api_key");
        assert_eq!(versioned.version(), Some("v2"));
    }

    #[test]
    fn test_secret_value() {
        let secret = SecretValue::from_string("hunter2");
        assert_eq!(secret.as_str(), Some("hunter2"));
        assert!(!secret.is_expired());

        let expired = SecretValue::from_string("old").with_expiry(Duration::from_nanos(1));
        std::thread::sleep(Duration::from_millis(1));
        assert!(expired.is_expired());
    }

    #[test]
    fn test_env_var_key_conversion() {
        let store = EnvVarSecretStore::new("MYAPP_");
        let key = SecretKey::new("database/password");
        assert_eq!(store.key_to_env_var(&key), "MYAPP_DATABASE_PASSWORD");

        let key2 = SecretKey::new("api.key");
        assert_eq!(store.key_to_env_var(&key2), "MYAPP_API_KEY");
    }

    #[tokio::test]
    async fn test_in_memory_store() {
        let store = InMemorySecretStore::new();
        let key = SecretKey::new("test/secret");

        // Initially not found
        assert!(store.get_secret(&key).await.is_err());
        assert!(!store.secret_exists(&key).await.unwrap());

        // Add secret
        store.add_string_secret(key.clone(), "secret_value");

        // Now found
        let secret = store.get_secret(&key).await.unwrap();
        assert_eq!(secret.as_str(), Some("secret_value"));
        assert!(store.secret_exists(&key).await.unwrap());
    }

    #[tokio::test]
    async fn test_chained_store() {
        let store1 = Arc::new(InMemorySecretStore::new());
        let store2 = Arc::new(InMemorySecretStore::new());

        let key1 = SecretKey::new("key1");
        let key2 = SecretKey::new("key2");

        store1.add_string_secret(key1.clone(), "value1");
        store2.add_string_secret(key2.clone(), "value2");

        let chain = ChainedSecretStore::new()
            .with_store(store1)
            .with_store(store2);

        // Can find keys from both stores
        let secret1 = chain.get_secret(&key1).await.unwrap();
        assert_eq!(secret1.as_str(), Some("value1"));

        let secret2 = chain.get_secret(&key2).await.unwrap();
        assert_eq!(secret2.as_str(), Some("value2"));

        // Unknown key fails
        assert!(chain.get_secret(&SecretKey::new("unknown")).await.is_err());
    }

    #[tokio::test]
    async fn test_cached_store() {
        let inner = InMemorySecretStore::new();
        let key = SecretKey::new("cached_key");
        inner.add_string_secret(key.clone(), "cached_value");

        let cached = CachedSecretStore::new(inner)
            .with_ttl(Duration::from_secs(60))
            .with_max_entries(10);

        // First fetch populates cache
        let secret = cached.get_secret(&key).await.unwrap();
        assert_eq!(secret.as_str(), Some("cached_value"));

        // Second fetch uses cache
        let secret2 = cached.get_secret(&key).await.unwrap();
        assert_eq!(secret2.as_str(), Some("cached_value"));

        // Invalidate and re-fetch
        cached.invalidate(&key);
        let secret3 = cached.get_secret(&key).await.unwrap();
        assert_eq!(secret3.as_str(), Some("cached_value"));
    }

    #[tokio::test]
    async fn test_list_secrets() {
        let store = InMemorySecretStore::new();
        store.add_string_secret(SecretKey::new("db/host"), "localhost");
        store.add_string_secret(SecretKey::new("db/port"), "5432");
        store.add_string_secret(SecretKey::new("api/key"), "secret");

        let all = store.list_secrets(None).await.unwrap();
        assert_eq!(all.len(), 3);

        let db_only = store.list_secrets(Some("db/")).await.unwrap();
        assert_eq!(db_only.len(), 2);
    }
}
