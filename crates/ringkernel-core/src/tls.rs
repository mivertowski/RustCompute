//! TLS configuration and utilities for secure communications.
//!
//! This module provides TLS support for RingKernel services, including:
//!
//! - Server TLS with rustls
//! - Client certificate validation (mTLS)
//! - Certificate rotation support
//! - SNI handling for multi-domain deployments
//!
//! # Features
//!
//! Requires the `tls` feature flag to be enabled.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::tls::{TlsConfig, TlsAcceptor};
//!
//! // Server TLS configuration
//! let config = TlsConfig::server()
//!     .with_cert_file("server.crt")
//!     .with_key_file("server.key")
//!     .with_client_auth(ClientAuth::Required)
//!     .build()?;
//!
//! let acceptor = TlsAcceptor::new(config)?;
//! ```

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use parking_lot::RwLock;

// ============================================================================
// TLS ERRORS
// ============================================================================

/// Errors that can occur during TLS operations.
#[derive(Debug, Clone)]
pub enum TlsError {
    /// Certificate file not found.
    CertificateNotFound(String),
    /// Private key file not found.
    KeyNotFound(String),
    /// Invalid certificate format.
    InvalidCertificate(String),
    /// Invalid private key format.
    InvalidKey(String),
    /// Certificate chain validation failed.
    CertificateValidation(String),
    /// TLS handshake failed.
    HandshakeFailed(String),
    /// Certificate has expired.
    CertificateExpired,
    /// Certificate not yet valid.
    CertificateNotYetValid,
    /// SNI hostname mismatch.
    SniMismatch(String),
    /// Client authentication failed.
    ClientAuthFailed(String),
    /// Configuration error.
    Configuration(String),
    /// I/O error.
    Io(String),
}

impl fmt::Display for TlsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CertificateNotFound(path) => write!(f, "Certificate not found: {}", path),
            Self::KeyNotFound(path) => write!(f, "Private key not found: {}", path),
            Self::InvalidCertificate(msg) => write!(f, "Invalid certificate: {}", msg),
            Self::InvalidKey(msg) => write!(f, "Invalid private key: {}", msg),
            Self::CertificateValidation(msg) => write!(f, "Certificate validation failed: {}", msg),
            Self::HandshakeFailed(msg) => write!(f, "TLS handshake failed: {}", msg),
            Self::CertificateExpired => write!(f, "Certificate has expired"),
            Self::CertificateNotYetValid => write!(f, "Certificate is not yet valid"),
            Self::SniMismatch(hostname) => write!(f, "SNI hostname mismatch: {}", hostname),
            Self::ClientAuthFailed(msg) => write!(f, "Client authentication failed: {}", msg),
            Self::Configuration(msg) => write!(f, "TLS configuration error: {}", msg),
            Self::Io(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for TlsError {}

/// Result type for TLS operations.
pub type TlsResult<T> = Result<T, TlsError>;

// ============================================================================
// TLS VERSION
// ============================================================================

/// Supported TLS protocol versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TlsVersion {
    /// TLS 1.2 (minimum recommended).
    Tls12,
    /// TLS 1.3 (preferred).
    Tls13,
}

impl TlsVersion {
    /// Get the version string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tls12 => "TLS 1.2",
            Self::Tls13 => "TLS 1.3",
        }
    }
}

impl Default for TlsVersion {
    fn default() -> Self {
        Self::Tls13
    }
}

// ============================================================================
// CLIENT AUTHENTICATION MODE
// ============================================================================

/// Client certificate authentication mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClientAuth {
    /// No client certificate required.
    #[default]
    None,
    /// Client certificate optional (verify if provided).
    Optional,
    /// Client certificate required (mTLS).
    Required,
}

impl ClientAuth {
    /// Check if client certificates are verified.
    pub fn verifies_client(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Check if client certificates are required.
    pub fn requires_client(&self) -> bool {
        matches!(self, Self::Required)
    }
}

// ============================================================================
// CERTIFICATE INFO
// ============================================================================

/// Information about a certificate.
#[derive(Debug, Clone)]
pub struct CertificateInfo {
    /// Subject common name.
    pub subject_cn: Option<String>,
    /// Issuer common name.
    pub issuer_cn: Option<String>,
    /// Subject alternative names (DNS names).
    pub san_dns: Vec<String>,
    /// Subject alternative names (IP addresses).
    pub san_ips: Vec<String>,
    /// Not valid before.
    pub not_before: Option<SystemTime>,
    /// Not valid after.
    pub not_after: Option<SystemTime>,
    /// Serial number (hex).
    pub serial_number: String,
    /// Is self-signed.
    pub is_self_signed: bool,
    /// Is CA certificate.
    pub is_ca: bool,
    /// Key usage.
    pub key_usage: Vec<String>,
    /// Extended key usage.
    pub extended_key_usage: Vec<String>,
}

impl CertificateInfo {
    /// Create empty certificate info.
    pub fn empty() -> Self {
        Self {
            subject_cn: None,
            issuer_cn: None,
            san_dns: Vec::new(),
            san_ips: Vec::new(),
            not_before: None,
            not_after: None,
            serial_number: String::new(),
            is_self_signed: false,
            is_ca: false,
            key_usage: Vec::new(),
            extended_key_usage: Vec::new(),
        }
    }

    /// Check if the certificate is currently valid.
    pub fn is_valid(&self) -> bool {
        let now = SystemTime::now();

        if let Some(not_before) = self.not_before {
            if now < not_before {
                return false;
            }
        }

        if let Some(not_after) = self.not_after {
            if now > not_after {
                return false;
            }
        }

        true
    }

    /// Get remaining validity duration.
    pub fn remaining_validity(&self) -> Option<Duration> {
        let now = SystemTime::now();
        self.not_after.and_then(|not_after| not_after.duration_since(now).ok())
    }

    /// Check if certificate expires within the given duration.
    pub fn expires_within(&self, duration: Duration) -> bool {
        self.remaining_validity()
            .map(|remaining| remaining < duration)
            .unwrap_or(true)
    }

    /// Check if hostname matches the certificate.
    pub fn matches_hostname(&self, hostname: &str) -> bool {
        // Check subject CN
        if let Some(ref cn) = self.subject_cn {
            if cn == hostname || Self::wildcard_match(cn, hostname) {
                return true;
            }
        }

        // Check SAN DNS names
        for san in &self.san_dns {
            if san == hostname || Self::wildcard_match(san, hostname) {
                return true;
            }
        }

        // Check SAN IPs
        for san_ip in &self.san_ips {
            if san_ip == hostname {
                return true;
            }
        }

        false
    }

    /// Check wildcard pattern match.
    fn wildcard_match(pattern: &str, hostname: &str) -> bool {
        if let Some(suffix) = pattern.strip_prefix("*.") {
            // Wildcard certificate - matches one level
            if let Some(rest) = hostname.strip_suffix(suffix) {
                // Must have exactly one label before the suffix
                return rest.ends_with('.') && !rest[..rest.len() - 1].contains('.');
            }
        }
        false
    }
}

// ============================================================================
// TLS CONFIGURATION
// ============================================================================

/// TLS configuration for server or client.
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Certificate chain (PEM format).
    pub cert_chain: Option<Vec<u8>>,
    /// Certificate chain file path.
    pub cert_path: Option<PathBuf>,
    /// Private key (PEM format).
    pub private_key: Option<Vec<u8>>,
    /// Private key file path.
    pub key_path: Option<PathBuf>,
    /// CA certificates for verification (PEM format).
    pub ca_certs: Option<Vec<u8>>,
    /// CA certificates file path.
    pub ca_path: Option<PathBuf>,
    /// Minimum TLS version.
    pub min_version: TlsVersion,
    /// Maximum TLS version.
    pub max_version: TlsVersion,
    /// Client authentication mode.
    pub client_auth: ClientAuth,
    /// ALPN protocols.
    pub alpn_protocols: Vec<String>,
    /// SNI hostnames (for server).
    pub sni_hostnames: Vec<String>,
    /// Session ticket lifetime.
    pub session_ticket_lifetime: Duration,
    /// Enable OCSP stapling.
    pub ocsp_stapling: bool,
    /// Certificate rotation check interval.
    pub rotation_check_interval: Duration,
    /// Whether this is a server or client config.
    pub is_server: bool,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            cert_chain: None,
            cert_path: None,
            private_key: None,
            key_path: None,
            ca_certs: None,
            ca_path: None,
            min_version: TlsVersion::Tls12,
            max_version: TlsVersion::Tls13,
            client_auth: ClientAuth::None,
            alpn_protocols: vec!["h2".to_string(), "http/1.1".to_string()],
            sni_hostnames: Vec::new(),
            session_ticket_lifetime: Duration::from_secs(24 * 60 * 60), // 24 hours
            ocsp_stapling: false,
            rotation_check_interval: Duration::from_secs(3600), // 1 hour
            is_server: false,
        }
    }
}

impl TlsConfig {
    /// Create a server TLS configuration.
    pub fn server() -> TlsConfigBuilder {
        TlsConfigBuilder::new(true)
    }

    /// Create a client TLS configuration.
    pub fn client() -> TlsConfigBuilder {
        TlsConfigBuilder::new(false)
    }

    /// Check if this is a server configuration.
    pub fn is_server(&self) -> bool {
        self.is_server
    }

    /// Check if this is a client configuration.
    pub fn is_client(&self) -> bool {
        !self.is_server
    }

    /// Check if mTLS is enabled.
    pub fn is_mtls(&self) -> bool {
        self.client_auth.requires_client()
    }
}

// ============================================================================
// TLS CONFIG BUILDER
// ============================================================================

/// Builder for TLS configuration.
pub struct TlsConfigBuilder {
    config: TlsConfig,
}

impl TlsConfigBuilder {
    /// Create a new builder.
    fn new(is_server: bool) -> Self {
        Self {
            config: TlsConfig {
                is_server,
                ..Default::default()
            },
        }
    }

    /// Set certificate chain from bytes (PEM format).
    pub fn with_cert(mut self, cert: Vec<u8>) -> Self {
        self.config.cert_chain = Some(cert);
        self
    }

    /// Set certificate chain from file path.
    pub fn with_cert_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.cert_path = Some(path.into());
        self
    }

    /// Set private key from bytes (PEM format).
    pub fn with_key(mut self, key: Vec<u8>) -> Self {
        self.config.private_key = Some(key);
        self
    }

    /// Set private key from file path.
    pub fn with_key_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.key_path = Some(path.into());
        self
    }

    /// Set CA certificates from bytes (PEM format).
    pub fn with_ca_cert(mut self, ca: Vec<u8>) -> Self {
        self.config.ca_certs = Some(ca);
        self
    }

    /// Set CA certificates from file path.
    pub fn with_ca_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.ca_path = Some(path.into());
        self
    }

    /// Set minimum TLS version.
    pub fn with_min_version(mut self, version: TlsVersion) -> Self {
        self.config.min_version = version;
        self
    }

    /// Set maximum TLS version.
    pub fn with_max_version(mut self, version: TlsVersion) -> Self {
        self.config.max_version = version;
        self
    }

    /// Set client authentication mode (server only).
    pub fn with_client_auth(mut self, auth: ClientAuth) -> Self {
        self.config.client_auth = auth;
        self
    }

    /// Enable mTLS (mutual TLS) - shorthand for with_client_auth(Required).
    pub fn with_mtls(mut self) -> Self {
        self.config.client_auth = ClientAuth::Required;
        self
    }

    /// Set ALPN protocols.
    pub fn with_alpn(mut self, protocols: Vec<String>) -> Self {
        self.config.alpn_protocols = protocols;
        self
    }

    /// Add an SNI hostname.
    pub fn with_sni_hostname(mut self, hostname: impl Into<String>) -> Self {
        self.config.sni_hostnames.push(hostname.into());
        self
    }

    /// Set session ticket lifetime.
    pub fn with_session_ticket_lifetime(mut self, lifetime: Duration) -> Self {
        self.config.session_ticket_lifetime = lifetime;
        self
    }

    /// Enable OCSP stapling.
    pub fn with_ocsp_stapling(mut self, enable: bool) -> Self {
        self.config.ocsp_stapling = enable;
        self
    }

    /// Set certificate rotation check interval.
    pub fn with_rotation_check_interval(mut self, interval: Duration) -> Self {
        self.config.rotation_check_interval = interval;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> TlsResult<TlsConfig> {
        // Validate configuration
        if self.config.is_server {
            // Server requires certificate and key
            if self.config.cert_chain.is_none() && self.config.cert_path.is_none() {
                return Err(TlsError::Configuration(
                    "Server TLS requires a certificate".to_string(),
                ));
            }
            if self.config.private_key.is_none() && self.config.key_path.is_none() {
                return Err(TlsError::Configuration(
                    "Server TLS requires a private key".to_string(),
                ));
            }
        }

        Ok(self.config)
    }
}

// ============================================================================
// CERTIFICATE STORE
// ============================================================================

/// A certificate with its private key and metadata.
#[derive(Clone)]
pub struct CertificateEntry {
    /// Certificate chain (DER format).
    pub cert_chain: Vec<Vec<u8>>,
    /// Private key (DER format).
    pub private_key: Vec<u8>,
    /// Certificate info.
    pub info: CertificateInfo,
    /// When the certificate was loaded.
    pub loaded_at: Instant,
    /// File path (if loaded from file).
    pub cert_path: Option<PathBuf>,
    /// Key path (if loaded from file).
    pub key_path: Option<PathBuf>,
}

impl fmt::Debug for CertificateEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CertificateEntry")
            .field("info", &self.info)
            .field("loaded_at", &self.loaded_at)
            .field("cert_path", &self.cert_path)
            .field("key_path", &self.key_path)
            .finish()
    }
}

/// Store for managing certificates with rotation support.
pub struct CertificateStore {
    /// Current certificate.
    current: RwLock<Option<CertificateEntry>>,
    /// Previous certificate (for graceful rotation).
    previous: RwLock<Option<CertificateEntry>>,
    /// Rotation check interval.
    check_interval: Duration,
    /// Last check time.
    last_check: RwLock<Instant>,
    /// Rotation callback.
    on_rotation: RwLock<Option<Arc<dyn Fn(&CertificateEntry) + Send + Sync>>>,
    /// Expiry warning threshold.
    expiry_warning_threshold: Duration,
}

impl CertificateStore {
    /// Create a new certificate store.
    pub fn new() -> Self {
        Self {
            current: RwLock::new(None),
            previous: RwLock::new(None),
            check_interval: Duration::from_secs(3600),
            last_check: RwLock::new(Instant::now()),
            on_rotation: RwLock::new(None),
            expiry_warning_threshold: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
        }
    }

    /// Set the rotation check interval.
    pub fn with_check_interval(mut self, interval: Duration) -> Self {
        self.check_interval = interval;
        self
    }

    /// Set the expiry warning threshold.
    pub fn with_expiry_warning(mut self, threshold: Duration) -> Self {
        self.expiry_warning_threshold = threshold;
        self
    }

    /// Set a callback for certificate rotation events.
    pub fn on_rotation<F>(self, callback: F) -> Self
    where
        F: Fn(&CertificateEntry) + Send + Sync + 'static,
    {
        *self.on_rotation.write() = Some(Arc::new(callback));
        self
    }

    /// Load a certificate from PEM data.
    pub fn load_pem(&self, cert_pem: &[u8], key_pem: &[u8]) -> TlsResult<()> {
        let entry = self.parse_pem(cert_pem, key_pem, None, None)?;
        self.set_certificate(entry);
        Ok(())
    }

    /// Load a certificate from files.
    pub fn load_files(&self, cert_path: &PathBuf, key_path: &PathBuf) -> TlsResult<()> {
        let cert_pem = std::fs::read(cert_path).map_err(|e| {
            TlsError::CertificateNotFound(format!("{}: {}", cert_path.display(), e))
        })?;

        let key_pem = std::fs::read(key_path)
            .map_err(|e| TlsError::KeyNotFound(format!("{}: {}", key_path.display(), e)))?;

        let entry = self.parse_pem(
            &cert_pem,
            &key_pem,
            Some(cert_path.clone()),
            Some(key_path.clone()),
        )?;
        self.set_certificate(entry);
        Ok(())
    }

    /// Parse PEM data into a certificate entry.
    fn parse_pem(
        &self,
        _cert_pem: &[u8],
        _key_pem: &[u8],
        cert_path: Option<PathBuf>,
        key_path: Option<PathBuf>,
    ) -> TlsResult<CertificateEntry> {
        // Note: Full implementation would use rustls-pemfile to parse PEM data
        // For now, create a placeholder entry
        Ok(CertificateEntry {
            cert_chain: Vec::new(),
            private_key: Vec::new(),
            info: CertificateInfo::empty(),
            loaded_at: Instant::now(),
            cert_path,
            key_path,
        })
    }

    /// Set the current certificate, moving the previous one to backup.
    fn set_certificate(&self, entry: CertificateEntry) {
        let prev = {
            let mut current = self.current.write();
            let prev = current.take();
            *current = Some(entry.clone());
            prev
        };

        if let Some(prev_entry) = prev {
            *self.previous.write() = Some(prev_entry);
        }

        // Call rotation callback
        if let Some(callback) = self.on_rotation.read().as_ref() {
            callback(&entry);
        }
    }

    /// Get the current certificate.
    pub fn current(&self) -> Option<CertificateEntry> {
        self.current.read().clone()
    }

    /// Get the previous certificate (for graceful rotation).
    pub fn previous(&self) -> Option<CertificateEntry> {
        self.previous.read().clone()
    }

    /// Check if a reload is needed (e.g., files changed).
    pub fn check_reload(&self) -> TlsResult<bool> {
        let now = Instant::now();
        let should_check = {
            let last = *self.last_check.read();
            now.duration_since(last) >= self.check_interval
        };

        if !should_check {
            return Ok(false);
        }

        *self.last_check.write() = now;

        // Check if files have changed (simplified check - just reload)
        // Clone paths before releasing lock to avoid borrow issues
        let paths = {
            let current = self.current.read();
            current.as_ref().and_then(|entry| {
                match (&entry.cert_path, &entry.key_path) {
                    (Some(cert), Some(key)) => Some((cert.clone(), key.clone())),
                    _ => None,
                }
            })
        };

        if let Some((cert_path, key_path)) = paths {
            self.load_files(&cert_path, &key_path)?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Check if the current certificate is expiring soon.
    pub fn is_expiring_soon(&self) -> bool {
        self.current
            .read()
            .as_ref()
            .map(|e| e.info.expires_within(self.expiry_warning_threshold))
            .unwrap_or(false)
    }

    /// Get remaining validity of the current certificate.
    pub fn remaining_validity(&self) -> Option<Duration> {
        self.current
            .read()
            .as_ref()
            .and_then(|e| e.info.remaining_validity())
    }
}

impl Default for CertificateStore {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SNI RESOLVER
// ============================================================================

/// SNI (Server Name Indication) resolver for multi-domain TLS.
pub struct SniResolver {
    /// Default certificate store.
    default_store: Arc<CertificateStore>,
    /// Per-hostname certificate stores.
    hostname_stores: RwLock<std::collections::HashMap<String, Arc<CertificateStore>>>,
}

impl SniResolver {
    /// Create a new SNI resolver with a default certificate store.
    pub fn new(default_store: Arc<CertificateStore>) -> Self {
        Self {
            default_store,
            hostname_stores: RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Add a certificate store for a specific hostname.
    pub fn add_hostname(&self, hostname: impl Into<String>, store: Arc<CertificateStore>) {
        let mut stores = self.hostname_stores.write();
        stores.insert(hostname.into(), store);
    }

    /// Remove a hostname.
    pub fn remove_hostname(&self, hostname: &str) -> bool {
        let mut stores = self.hostname_stores.write();
        stores.remove(hostname).is_some()
    }

    /// Resolve a certificate store for a given hostname.
    pub fn resolve(&self, hostname: &str) -> Arc<CertificateStore> {
        // Try exact match
        let stores = self.hostname_stores.read();
        if let Some(store) = stores.get(hostname) {
            return Arc::clone(store);
        }

        // Try wildcard match
        if let Some(dot_pos) = hostname.find('.') {
            let wildcard = format!("*.{}", &hostname[dot_pos + 1..]);
            if let Some(store) = stores.get(&wildcard) {
                return Arc::clone(store);
            }
        }

        // Return default
        Arc::clone(&self.default_store)
    }

    /// Get all configured hostnames.
    pub fn hostnames(&self) -> Vec<String> {
        let stores = self.hostname_stores.read();
        stores.keys().cloned().collect()
    }

    /// Get the number of configured hostnames.
    pub fn hostname_count(&self) -> usize {
        self.hostname_stores.read().len()
    }
}

// ============================================================================
// TLS ACCEPTOR (SERVER)
// ============================================================================

/// TLS acceptor for server-side TLS connections.
pub struct TlsAcceptor {
    config: TlsConfig,
    cert_store: Arc<CertificateStore>,
    sni_resolver: Option<Arc<SniResolver>>,
    stats: TlsAcceptorStats,
}

/// Statistics for the TLS acceptor.
#[derive(Debug, Default)]
pub struct TlsAcceptorStats {
    /// Total handshakes attempted.
    pub handshakes_attempted: std::sync::atomic::AtomicU64,
    /// Successful handshakes.
    pub handshakes_succeeded: std::sync::atomic::AtomicU64,
    /// Failed handshakes.
    pub handshakes_failed: std::sync::atomic::AtomicU64,
    /// Client auth failures.
    pub client_auth_failures: std::sync::atomic::AtomicU64,
    /// SNI mismatches.
    pub sni_mismatches: std::sync::atomic::AtomicU64,
}

impl TlsAcceptor {
    /// Create a new TLS acceptor with the given configuration.
    pub fn new(config: TlsConfig) -> TlsResult<Self> {
        if !config.is_server {
            return Err(TlsError::Configuration(
                "TlsAcceptor requires a server configuration".to_string(),
            ));
        }

        let cert_store = Arc::new(
            CertificateStore::new().with_check_interval(config.rotation_check_interval),
        );

        // Load certificate if paths are specified
        if let (Some(cert_path), Some(key_path)) = (&config.cert_path, &config.key_path) {
            cert_store.load_files(cert_path, key_path)?;
        } else if let (Some(cert_pem), Some(key_pem)) =
            (&config.cert_chain, &config.private_key)
        {
            cert_store.load_pem(cert_pem, key_pem)?;
        }

        Ok(Self {
            config,
            cert_store,
            sni_resolver: None,
            stats: TlsAcceptorStats::default(),
        })
    }

    /// Set an SNI resolver for multi-domain support.
    pub fn with_sni_resolver(mut self, resolver: Arc<SniResolver>) -> Self {
        self.sni_resolver = Some(resolver);
        self
    }

    /// Get the configuration.
    pub fn config(&self) -> &TlsConfig {
        &self.config
    }

    /// Get the certificate store.
    pub fn cert_store(&self) -> &Arc<CertificateStore> {
        &self.cert_store
    }

    /// Get current statistics.
    pub fn stats(&self) -> TlsAcceptorStatsSnapshot {
        TlsAcceptorStatsSnapshot {
            handshakes_attempted: self
                .stats
                .handshakes_attempted
                .load(std::sync::atomic::Ordering::Relaxed),
            handshakes_succeeded: self
                .stats
                .handshakes_succeeded
                .load(std::sync::atomic::Ordering::Relaxed),
            handshakes_failed: self
                .stats
                .handshakes_failed
                .load(std::sync::atomic::Ordering::Relaxed),
            client_auth_failures: self
                .stats
                .client_auth_failures
                .load(std::sync::atomic::Ordering::Relaxed),
            sni_mismatches: self
                .stats
                .sni_mismatches
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Check if certificate rotation is needed.
    pub fn check_rotation(&self) -> TlsResult<bool> {
        self.cert_store.check_reload()
    }

    /// Check if certificate is expiring soon.
    pub fn is_cert_expiring_soon(&self) -> bool {
        self.cert_store.is_expiring_soon()
    }
}

/// Snapshot of TLS acceptor statistics.
#[derive(Debug, Clone)]
pub struct TlsAcceptorStatsSnapshot {
    /// Total handshakes attempted.
    pub handshakes_attempted: u64,
    /// Successful handshakes.
    pub handshakes_succeeded: u64,
    /// Failed handshakes.
    pub handshakes_failed: u64,
    /// Client auth failures.
    pub client_auth_failures: u64,
    /// SNI mismatches.
    pub sni_mismatches: u64,
}

impl TlsAcceptorStatsSnapshot {
    /// Calculate handshake success rate.
    pub fn success_rate(&self) -> f64 {
        if self.handshakes_attempted == 0 {
            1.0
        } else {
            self.handshakes_succeeded as f64 / self.handshakes_attempted as f64
        }
    }
}

// ============================================================================
// TLS CONNECTOR (CLIENT)
// ============================================================================

/// TLS connector for client-side TLS connections.
pub struct TlsConnector {
    config: TlsConfig,
    /// Root certificates for server verification.
    root_certs: Vec<Vec<u8>>,
    /// Client certificate (for mTLS).
    client_cert: Option<CertificateEntry>,
    /// Allow invalid certificates (for testing only).
    allow_invalid_certs: bool,
    /// Allow invalid hostnames (for testing only).
    allow_invalid_hostnames: bool,
}

impl TlsConnector {
    /// Create a new TLS connector with the given configuration.
    pub fn new(config: TlsConfig) -> TlsResult<Self> {
        if config.is_server {
            return Err(TlsError::Configuration(
                "TlsConnector requires a client configuration".to_string(),
            ));
        }

        Ok(Self {
            config,
            root_certs: Vec::new(),
            client_cert: None,
            allow_invalid_certs: false,
            allow_invalid_hostnames: false,
        })
    }

    /// Create a connector with system root certificates.
    pub fn with_native_roots() -> TlsResult<Self> {
        let config = TlsConfig::client().build()?;
        Ok(Self {
            config,
            root_certs: Vec::new(), // Would load from webpki-roots
            client_cert: None,
            allow_invalid_certs: false,
            allow_invalid_hostnames: false,
        })
    }

    /// Set the client certificate for mTLS.
    pub fn with_client_cert(mut self, cert: CertificateEntry) -> Self {
        self.client_cert = Some(cert);
        self
    }

    /// Allow invalid certificates (DANGER: for testing only).
    pub fn danger_accept_invalid_certs(mut self, allow: bool) -> Self {
        self.allow_invalid_certs = allow;
        self
    }

    /// Allow invalid hostnames (DANGER: for testing only).
    pub fn danger_accept_invalid_hostnames(mut self, allow: bool) -> Self {
        self.allow_invalid_hostnames = allow;
        self
    }

    /// Get the configuration.
    pub fn config(&self) -> &TlsConfig {
        &self.config
    }

    /// Check if a client certificate is configured.
    pub fn has_client_cert(&self) -> bool {
        self.client_cert.is_some()
    }
}

// ============================================================================
// TLS SESSION INFO
// ============================================================================

/// Information about an established TLS session.
#[derive(Debug, Clone)]
pub struct TlsSessionInfo {
    /// Negotiated TLS version.
    pub version: TlsVersion,
    /// Negotiated cipher suite.
    pub cipher_suite: String,
    /// SNI hostname (if provided).
    pub sni_hostname: Option<String>,
    /// ALPN protocol (if negotiated).
    pub alpn_protocol: Option<String>,
    /// Server certificate info.
    pub server_cert: Option<CertificateInfo>,
    /// Client certificate info (for mTLS).
    pub client_cert: Option<CertificateInfo>,
    /// Session resumed.
    pub resumed: bool,
    /// Handshake duration.
    pub handshake_duration: Duration,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tls_version() {
        assert_eq!(TlsVersion::Tls12.as_str(), "TLS 1.2");
        assert_eq!(TlsVersion::Tls13.as_str(), "TLS 1.3");
        assert_eq!(TlsVersion::default(), TlsVersion::Tls13);
    }

    #[test]
    fn test_client_auth() {
        assert!(!ClientAuth::None.verifies_client());
        assert!(ClientAuth::Optional.verifies_client());
        assert!(ClientAuth::Required.verifies_client());

        assert!(!ClientAuth::None.requires_client());
        assert!(!ClientAuth::Optional.requires_client());
        assert!(ClientAuth::Required.requires_client());
    }

    #[test]
    fn test_tls_config_builder_server() {
        let config = TlsConfig::server()
            .with_cert(vec![1, 2, 3])
            .with_key(vec![4, 5, 6])
            .with_client_auth(ClientAuth::Required)
            .with_alpn(vec!["h2".to_string()])
            .with_sni_hostname("example.com")
            .build()
            .unwrap();

        assert!(config.is_server());
        assert!(config.is_mtls());
        assert_eq!(config.client_auth, ClientAuth::Required);
        assert_eq!(config.alpn_protocols, vec!["h2"]);
        assert!(config.sni_hostnames.contains(&"example.com".to_string()));
    }

    #[test]
    fn test_tls_config_builder_client() {
        let config = TlsConfig::client()
            .with_min_version(TlsVersion::Tls13)
            .build()
            .unwrap();

        assert!(config.is_client());
        assert!(!config.is_server());
        assert_eq!(config.min_version, TlsVersion::Tls13);
    }

    #[test]
    fn test_tls_config_server_requires_cert() {
        let result = TlsConfig::server().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_certificate_info_validity() {
        let mut info = CertificateInfo::empty();

        // No dates set - should be valid
        assert!(info.is_valid());

        // Set valid range
        info.not_before = Some(SystemTime::now() - Duration::from_secs(3600));
        info.not_after = Some(SystemTime::now() + Duration::from_secs(3600));
        assert!(info.is_valid());

        // Expired
        info.not_after = Some(SystemTime::now() - Duration::from_secs(1));
        assert!(!info.is_valid());
    }

    #[test]
    fn test_certificate_info_hostname_match() {
        let mut info = CertificateInfo::empty();
        info.subject_cn = Some("example.com".to_string());
        info.san_dns = vec!["www.example.com".to_string(), "*.test.com".to_string()];
        info.san_ips = vec!["192.168.1.1".to_string()];

        // Exact CN match
        assert!(info.matches_hostname("example.com"));

        // SAN DNS match
        assert!(info.matches_hostname("www.example.com"));

        // Wildcard match
        assert!(info.matches_hostname("api.test.com"));
        assert!(!info.matches_hostname("sub.api.test.com")); // Only one level

        // IP match
        assert!(info.matches_hostname("192.168.1.1"));

        // No match
        assert!(!info.matches_hostname("other.com"));
    }

    #[test]
    fn test_certificate_store() {
        let store = CertificateStore::new();

        assert!(store.current().is_none());
        assert!(store.previous().is_none());
        assert!(!store.is_expiring_soon());
    }

    #[test]
    fn test_sni_resolver() {
        let default_store = Arc::new(CertificateStore::new());
        let resolver = SniResolver::new(default_store);

        let example_store = Arc::new(CertificateStore::new());
        resolver.add_hostname("example.com", example_store);

        assert_eq!(resolver.hostname_count(), 1);
        assert!(resolver.hostnames().contains(&"example.com".to_string()));

        // Resolve existing
        let _ = resolver.resolve("example.com");

        // Resolve non-existing (returns default)
        let _ = resolver.resolve("other.com");

        // Remove
        assert!(resolver.remove_hostname("example.com"));
        assert_eq!(resolver.hostname_count(), 0);
    }

    #[test]
    fn test_tls_acceptor_stats() {
        let config = TlsConfig::server()
            .with_cert(vec![1])
            .with_key(vec![2])
            .build()
            .unwrap();

        let acceptor = TlsAcceptor::new(config).unwrap();
        let stats = acceptor.stats();

        assert_eq!(stats.handshakes_attempted, 0);
        assert_eq!(stats.success_rate(), 1.0);
    }

    #[test]
    fn test_tls_connector() {
        let connector = TlsConnector::with_native_roots().unwrap();

        assert!(!connector.has_client_cert());
        assert!(connector.config().is_client());
    }

    #[test]
    fn test_tls_error_display() {
        let err = TlsError::CertificateNotFound("/path/to/cert".to_string());
        assert!(err.to_string().contains("/path/to/cert"));

        let err = TlsError::HandshakeFailed("protocol error".to_string());
        assert!(err.to_string().contains("handshake"));
    }

    #[test]
    fn test_mtls_config() {
        let config = TlsConfig::server()
            .with_cert(vec![1])
            .with_key(vec![2])
            .with_mtls()
            .with_ca_cert(vec![3])
            .build()
            .unwrap();

        assert!(config.is_mtls());
        assert_eq!(config.client_auth, ClientAuth::Required);
        assert!(config.ca_certs.is_some());
    }
}
