//! Authentication framework for RingKernel.
//!
//! This module provides a pluggable authentication system with support for
//! multiple methods including API keys, JWT tokens, and OAuth2.
//!
//! # Feature Flags
//!
//! - `auth` - Enables JWT token validation (requires `jsonwebtoken` crate)
//!
//! # Example
//!
//! ```rust,ignore
//! use ringkernel_core::auth::{AuthProvider, ApiKeyAuth, AuthContext};
//!
//! // Simple API key authentication
//! let auth = ApiKeyAuth::new()
//!     .add_key("admin", "secret-key-123", &["admin", "read", "write"])
//!     .add_key("readonly", "readonly-key-456", &["read"]);
//!
//! let ctx = auth.authenticate(&Credentials::ApiKey("secret-key-123".to_string())).await?;
//! assert!(ctx.has_permission("write"));
//!
//! // JWT authentication
//! let jwt_auth = JwtAuth::new(JwtConfig {
//!     secret: "your-256-bit-secret".to_string(),
//!     issuer: Some("ringkernel".to_string()),
//!     audience: Some("api".to_string()),
//!     ..Default::default()
//! });
//! let ctx = jwt_auth.authenticate(&Credentials::Bearer(token)).await?;
//! ```

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
#[cfg(feature = "auth")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "auth")]
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};

// ============================================================================
// CREDENTIALS
// ============================================================================

/// Credentials provided by a client for authentication.
#[derive(Debug, Clone)]
pub enum Credentials {
    /// API key authentication.
    ApiKey(String),
    /// Bearer token (JWT) authentication.
    Bearer(String),
    /// Basic authentication (username:password).
    Basic {
        /// Username for basic auth.
        username: String,
        /// Password for basic auth.
        password: String,
    },
    /// Custom credential type.
    Custom {
        /// Authentication scheme name.
        scheme: String,
        /// Credential value.
        value: String,
    },
}

impl Credentials {
    /// Create API key credentials.
    pub fn api_key(key: impl Into<String>) -> Self {
        Self::ApiKey(key.into())
    }

    /// Create bearer token credentials.
    pub fn bearer(token: impl Into<String>) -> Self {
        Self::Bearer(token.into())
    }

    /// Create basic auth credentials.
    pub fn basic(username: impl Into<String>, password: impl Into<String>) -> Self {
        Self::Basic {
            username: username.into(),
            password: password.into(),
        }
    }

    /// Parse from Authorization header value.
    pub fn from_header(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.splitn(2, ' ').collect();
        if parts.len() != 2 {
            return None;
        }

        match parts[0].to_lowercase().as_str() {
            "bearer" => Some(Self::Bearer(parts[1].to_string())),
            "basic" => {
                #[cfg(feature = "auth")]
                {
                    use base64::Engine;
                    let decoded = base64::engine::general_purpose::STANDARD
                        .decode(parts[1])
                        .ok()?;
                    let decoded_str = String::from_utf8(decoded).ok()?;
                    let creds: Vec<&str> = decoded_str.splitn(2, ':').collect();
                    if creds.len() == 2 {
                        Some(Self::Basic {
                            username: creds[0].to_string(),
                            password: creds[1].to_string(),
                        })
                    } else {
                        None
                    }
                }
                #[cfg(not(feature = "auth"))]
                {
                    None
                }
            }
            "apikey" | "api-key" | "x-api-key" => Some(Self::ApiKey(parts[1].to_string())),
            scheme => Some(Self::Custom {
                scheme: scheme.to_string(),
                value: parts[1].to_string(),
            }),
        }
    }
}

// ============================================================================
// AUTH CONTEXT
// ============================================================================

/// Identity of an authenticated principal.
#[derive(Debug, Clone)]
pub struct Identity {
    /// Unique identifier (user ID, service account, etc.).
    pub id: String,
    /// Display name.
    pub name: Option<String>,
    /// Email address.
    pub email: Option<String>,
    /// Tenant/organization ID (for multi-tenancy).
    pub tenant_id: Option<String>,
    /// Additional claims/attributes.
    pub claims: HashMap<String, String>,
}

impl Identity {
    /// Create a new identity.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            email: None,
            tenant_id: None,
            claims: HashMap::new(),
        }
    }

    /// Set the name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the email.
    pub fn with_email(mut self, email: impl Into<String>) -> Self {
        self.email = Some(email.into());
        self
    }

    /// Set the tenant ID.
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Add a claim.
    pub fn with_claim(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.claims.insert(key.into(), value.into());
        self
    }
}

/// Authentication context for an authenticated request.
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// The authenticated identity.
    pub identity: Identity,
    /// Roles assigned to this identity.
    pub roles: HashSet<String>,
    /// Permissions granted.
    pub permissions: HashSet<String>,
    /// When authentication occurred.
    pub authenticated_at: Instant,
    /// When the authentication expires.
    pub expires_at: Option<Instant>,
    /// The authentication method used.
    pub auth_method: String,
}

impl AuthContext {
    /// Create a new auth context.
    pub fn new(identity: Identity, auth_method: impl Into<String>) -> Self {
        Self {
            identity,
            roles: HashSet::new(),
            permissions: HashSet::new(),
            authenticated_at: Instant::now(),
            expires_at: None,
            auth_method: auth_method.into(),
        }
    }

    /// Add a role.
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.roles.insert(role.into());
        self
    }

    /// Add roles.
    pub fn with_roles<I, S>(mut self, roles: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.roles.extend(roles.into_iter().map(Into::into));
        self
    }

    /// Add a permission.
    pub fn with_permission(mut self, permission: impl Into<String>) -> Self {
        self.permissions.insert(permission.into());
        self
    }

    /// Add permissions.
    pub fn with_permissions<I, S>(mut self, permissions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.permissions
            .extend(permissions.into_iter().map(Into::into));
        self
    }

    /// Set expiration.
    pub fn with_expiry(mut self, duration: Duration) -> Self {
        self.expires_at = Some(Instant::now() + duration);
        self
    }

    /// Check if the context has expired.
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|exp| Instant::now() > exp)
            .unwrap_or(false)
    }

    /// Check if the identity has a role.
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.contains(role)
    }

    /// Check if the identity has any of the specified roles.
    pub fn has_any_role(&self, roles: &[&str]) -> bool {
        roles.iter().any(|r| self.roles.contains(*r))
    }

    /// Check if the identity has all of the specified roles.
    pub fn has_all_roles(&self, roles: &[&str]) -> bool {
        roles.iter().all(|r| self.roles.contains(*r))
    }

    /// Check if the identity has a permission.
    pub fn has_permission(&self, permission: &str) -> bool {
        self.permissions.contains(permission)
    }

    /// Check if the identity has any of the specified permissions.
    pub fn has_any_permission(&self, permissions: &[&str]) -> bool {
        permissions.iter().any(|p| self.permissions.contains(*p))
    }

    /// Get the tenant ID (for multi-tenant operations).
    pub fn tenant_id(&self) -> Option<&str> {
        self.identity.tenant_id.as_deref()
    }
}

// ============================================================================
// AUTH ERROR
// ============================================================================

/// Error type for authentication operations.
#[derive(Debug, Clone)]
pub enum AuthError {
    /// Invalid credentials provided.
    InvalidCredentials(String),
    /// Credentials have expired.
    Expired(String),
    /// Missing required credentials.
    MissingCredentials(String),
    /// Access denied (authenticated but not authorized).
    AccessDenied(String),
    /// Token validation failed.
    TokenInvalid(String),
    /// Authentication service unavailable.
    ServiceUnavailable(String),
    /// Rate limited.
    RateLimited(String),
    /// Other error.
    Other(String),
}

impl fmt::Display for AuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCredentials(msg) => write!(f, "Invalid credentials: {}", msg),
            Self::Expired(msg) => write!(f, "Credentials expired: {}", msg),
            Self::MissingCredentials(msg) => write!(f, "Missing credentials: {}", msg),
            Self::AccessDenied(msg) => write!(f, "Access denied: {}", msg),
            Self::TokenInvalid(msg) => write!(f, "Token invalid: {}", msg),
            Self::ServiceUnavailable(msg) => write!(f, "Auth service unavailable: {}", msg),
            Self::RateLimited(msg) => write!(f, "Rate limited: {}", msg),
            Self::Other(msg) => write!(f, "Auth error: {}", msg),
        }
    }
}

impl std::error::Error for AuthError {}

/// Result type for authentication operations.
pub type AuthResult<T> = Result<T, AuthError>;

// ============================================================================
// AUTH PROVIDER TRAIT
// ============================================================================

/// Trait for pluggable authentication providers.
#[async_trait]
pub trait AuthProvider: Send + Sync {
    /// Authenticate credentials and return an auth context.
    async fn authenticate(&self, credentials: &Credentials) -> AuthResult<AuthContext>;

    /// Validate an existing auth context (e.g., check if still valid).
    async fn validate(&self, context: &AuthContext) -> AuthResult<()>;

    /// Revoke authentication (e.g., invalidate a token).
    async fn revoke(&self, context: &AuthContext) -> AuthResult<()>;

    /// Get the provider name.
    fn provider_name(&self) -> &str;
}

// ============================================================================
// API KEY AUTHENTICATION
// ============================================================================

/// API key entry in the store.
#[derive(Debug, Clone)]
struct ApiKeyEntry {
    /// The API key hash (we store hash, not plaintext).
    _key_hash: u64,
    /// Identity associated with this key.
    identity: Identity,
    /// Permissions granted.
    permissions: HashSet<String>,
    /// Roles assigned.
    roles: HashSet<String>,
    /// When the key was created.
    _created_at: Instant,
    /// Optional expiration.
    expires_at: Option<Instant>,
    /// Whether the key is active.
    active: bool,
}

/// Simple hash function for API keys (in production, use a proper hash).
fn hash_api_key(key: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

/// API key authentication provider.
pub struct ApiKeyAuth {
    /// Registered API keys (hash -> entry).
    keys: RwLock<HashMap<u64, ApiKeyEntry>>,
    /// Default expiration for new keys.
    default_expiry: Option<Duration>,
}

impl ApiKeyAuth {
    /// Create a new API key auth provider.
    pub fn new() -> Self {
        Self {
            keys: RwLock::new(HashMap::new()),
            default_expiry: None,
        }
    }

    /// Set default expiration for new keys.
    pub fn with_default_expiry(mut self, expiry: Duration) -> Self {
        self.default_expiry = Some(expiry);
        self
    }

    /// Add an API key.
    pub fn add_key(
        self,
        identity_id: impl Into<String>,
        api_key: &str,
        permissions: &[&str],
    ) -> Self {
        let identity_id = identity_id.into();
        let key_hash = hash_api_key(api_key);

        let entry = ApiKeyEntry {
            _key_hash: key_hash,
            identity: Identity::new(&identity_id),
            permissions: permissions.iter().map(|s| s.to_string()).collect(),
            roles: HashSet::new(),
            _created_at: Instant::now(),
            expires_at: self.default_expiry.map(|d| Instant::now() + d),
            active: true,
        };

        self.keys.write().insert(key_hash, entry);
        self
    }

    /// Add an API key with roles.
    pub fn add_key_with_roles(
        self,
        identity_id: impl Into<String>,
        api_key: &str,
        permissions: &[&str],
        roles: &[&str],
    ) -> Self {
        let identity_id = identity_id.into();
        let key_hash = hash_api_key(api_key);

        let entry = ApiKeyEntry {
            _key_hash: key_hash,
            identity: Identity::new(&identity_id),
            permissions: permissions.iter().map(|s| s.to_string()).collect(),
            roles: roles.iter().map(|s| s.to_string()).collect(),
            _created_at: Instant::now(),
            expires_at: self.default_expiry.map(|d| Instant::now() + d),
            active: true,
        };

        self.keys.write().insert(key_hash, entry);
        self
    }

    /// Revoke an API key.
    pub fn revoke_key(&self, api_key: &str) -> bool {
        let key_hash = hash_api_key(api_key);
        let mut keys = self.keys.write();
        if let Some(entry) = keys.get_mut(&key_hash) {
            entry.active = false;
            true
        } else {
            false
        }
    }

    /// Get the number of registered keys.
    pub fn key_count(&self) -> usize {
        self.keys.read().len()
    }
}

impl Default for ApiKeyAuth {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AuthProvider for ApiKeyAuth {
    async fn authenticate(&self, credentials: &Credentials) -> AuthResult<AuthContext> {
        let api_key = match credentials {
            Credentials::ApiKey(key) => key,
            _ => {
                return Err(AuthError::InvalidCredentials(
                    "Expected API key".to_string(),
                ))
            }
        };

        let key_hash = hash_api_key(api_key);
        let keys = self.keys.read();

        let entry = keys
            .get(&key_hash)
            .ok_or_else(|| AuthError::InvalidCredentials("Unknown API key".to_string()))?;

        if !entry.active {
            return Err(AuthError::InvalidCredentials(
                "API key has been revoked".to_string(),
            ));
        }

        if let Some(expires) = entry.expires_at {
            if Instant::now() > expires {
                return Err(AuthError::Expired("API key has expired".to_string()));
            }
        }

        let mut ctx = AuthContext::new(entry.identity.clone(), "api_key")
            .with_permissions(entry.permissions.iter().cloned())
            .with_roles(entry.roles.iter().cloned());

        if let Some(expires) = entry.expires_at {
            let remaining = expires.saturating_duration_since(Instant::now());
            ctx = ctx.with_expiry(remaining);
        }

        Ok(ctx)
    }

    async fn validate(&self, context: &AuthContext) -> AuthResult<()> {
        if context.is_expired() {
            return Err(AuthError::Expired("Auth context has expired".to_string()));
        }
        Ok(())
    }

    async fn revoke(&self, _context: &AuthContext) -> AuthResult<()> {
        // API keys are revoked by key, not by context
        Ok(())
    }

    fn provider_name(&self) -> &str {
        "ApiKeyAuth"
    }
}

// ============================================================================
// JWT AUTHENTICATION (requires auth feature)
// ============================================================================

/// JWT claims structure.
#[cfg(feature = "auth")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JwtClaims {
    /// Subject (user ID).
    pub sub: String,
    /// Issued at timestamp.
    pub iat: u64,
    /// Expiration timestamp.
    pub exp: u64,
    /// Issuer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iss: Option<String>,
    /// Audience.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
    /// Roles.
    #[serde(default)]
    pub roles: Vec<String>,
    /// Permissions.
    #[serde(default)]
    pub permissions: Vec<String>,
    /// Tenant ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    /// Additional custom claims.
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// JWT authentication configuration.
#[cfg(feature = "auth")]
#[derive(Debug, Clone)]
pub struct JwtConfig {
    /// Secret key for HS256/HS384/HS512 algorithms.
    pub secret: Option<String>,
    /// Public key for RS256/ES256 algorithms (PEM format).
    pub public_key: Option<String>,
    /// Expected issuer.
    pub issuer: Option<String>,
    /// Expected audience.
    pub audience: Option<String>,
    /// Algorithm to use.
    pub algorithm: Algorithm,
    /// Leeway for time validation (seconds).
    pub leeway_seconds: u64,
}

#[cfg(feature = "auth")]
impl Default for JwtConfig {
    fn default() -> Self {
        Self {
            secret: None,
            public_key: None,
            issuer: None,
            audience: None,
            algorithm: Algorithm::HS256,
            leeway_seconds: 60,
        }
    }
}

/// JWT authentication provider.
#[cfg(feature = "auth")]
pub struct JwtAuth {
    config: JwtConfig,
    /// Revoked token IDs (jti claims).
    revoked_tokens: RwLock<HashSet<String>>,
}

#[cfg(feature = "auth")]
impl JwtAuth {
    /// Create a new JWT auth provider.
    pub fn new(config: JwtConfig) -> Self {
        Self {
            config,
            revoked_tokens: RwLock::new(HashSet::new()),
        }
    }

    /// Create with a simple secret (HS256).
    pub fn with_secret(secret: impl Into<String>) -> Self {
        Self::new(JwtConfig {
            secret: Some(secret.into()),
            algorithm: Algorithm::HS256,
            ..Default::default()
        })
    }

    /// Generate a JWT token for the given claims.
    pub fn generate_token(&self, claims: &JwtClaims) -> AuthResult<String> {
        let secret = self.config.secret.as_ref().ok_or_else(|| {
            AuthError::Other("No secret configured for token generation".to_string())
        })?;

        let token = encode(
            &Header::new(self.config.algorithm),
            claims,
            &EncodingKey::from_secret(secret.as_bytes()),
        )
        .map_err(|e| AuthError::Other(format!("Token generation failed: {}", e)))?;

        Ok(token)
    }

    /// Decode and validate a JWT token.
    fn decode_token(&self, token: &str) -> AuthResult<JwtClaims> {
        let mut validation = Validation::new(self.config.algorithm);
        validation.leeway = self.config.leeway_seconds;

        if let Some(ref issuer) = self.config.issuer {
            validation.set_issuer(&[issuer]);
        }

        if let Some(ref audience) = self.config.audience {
            validation.set_audience(&[audience]);
        }

        let decoding_key = if let Some(ref secret) = self.config.secret {
            DecodingKey::from_secret(secret.as_bytes())
        } else if let Some(ref _public_key) = self.config.public_key {
            // For RSA/EC keys, you'd use from_rsa_pem or from_ec_pem
            return Err(AuthError::Other(
                "Public key decoding not implemented".to_string(),
            ));
        } else {
            return Err(AuthError::Other(
                "No secret or public key configured".to_string(),
            ));
        };

        let token_data = decode::<JwtClaims>(token, &decoding_key, &validation)
            .map_err(|e| AuthError::TokenInvalid(format!("Token validation failed: {}", e)))?;

        Ok(token_data.claims)
    }

    /// Revoke a token by its jti claim.
    pub fn revoke_token(&self, jti: impl Into<String>) {
        self.revoked_tokens.write().insert(jti.into());
    }

    /// Check if a token is revoked.
    pub fn is_revoked(&self, jti: &str) -> bool {
        self.revoked_tokens.read().contains(jti)
    }
}

#[cfg(feature = "auth")]
#[async_trait]
impl AuthProvider for JwtAuth {
    async fn authenticate(&self, credentials: &Credentials) -> AuthResult<AuthContext> {
        let token = match credentials {
            Credentials::Bearer(t) => t,
            _ => {
                return Err(AuthError::InvalidCredentials(
                    "Expected Bearer token".to_string(),
                ))
            }
        };

        let claims = self.decode_token(token)?;

        // Check if token is revoked (if jti is present in custom claims)
        if let Some(serde_json::Value::String(jti)) = claims.custom.get("jti") {
            if self.is_revoked(jti) {
                return Err(AuthError::TokenInvalid(
                    "Token has been revoked".to_string(),
                ));
            }
        }

        let mut identity = Identity::new(&claims.sub);
        if let Some(tenant) = &claims.tenant_id {
            identity = identity.with_tenant(tenant);
        }

        // Add custom claims to identity
        for (key, value) in &claims.custom {
            if let serde_json::Value::String(s) = value {
                identity = identity.with_claim(key, s);
            }
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let remaining = claims.exp.saturating_sub(now);

        let ctx = AuthContext::new(identity, "jwt")
            .with_roles(claims.roles)
            .with_permissions(claims.permissions)
            .with_expiry(Duration::from_secs(remaining));

        Ok(ctx)
    }

    async fn validate(&self, context: &AuthContext) -> AuthResult<()> {
        if context.is_expired() {
            return Err(AuthError::Expired("Token has expired".to_string()));
        }
        Ok(())
    }

    async fn revoke(&self, context: &AuthContext) -> AuthResult<()> {
        // If there's a jti claim, add it to revoked set
        if let Some(jti) = context.identity.claims.get("jti") {
            self.revoke_token(jti);
        }
        Ok(())
    }

    fn provider_name(&self) -> &str {
        "JwtAuth"
    }
}

// ============================================================================
// CHAINED AUTH PROVIDER
// ============================================================================

/// Authentication provider that tries multiple providers in order.
pub struct ChainedAuthProvider {
    providers: Vec<Arc<dyn AuthProvider>>,
}

impl ChainedAuthProvider {
    /// Create a new chained auth provider.
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    /// Add a provider to the chain.
    pub fn with_provider(mut self, provider: Arc<dyn AuthProvider>) -> Self {
        self.providers.push(provider);
        self
    }
}

impl Default for ChainedAuthProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AuthProvider for ChainedAuthProvider {
    async fn authenticate(&self, credentials: &Credentials) -> AuthResult<AuthContext> {
        let mut last_error = AuthError::MissingCredentials("No providers configured".to_string());

        for provider in &self.providers {
            match provider.authenticate(credentials).await {
                Ok(ctx) => return Ok(ctx),
                Err(e) => {
                    last_error = e;
                    continue;
                }
            }
        }

        Err(last_error)
    }

    async fn validate(&self, context: &AuthContext) -> AuthResult<()> {
        // Validate using the provider that authenticated
        for provider in &self.providers {
            if provider.provider_name() == context.auth_method {
                return provider.validate(context).await;
            }
        }
        Ok(())
    }

    async fn revoke(&self, context: &AuthContext) -> AuthResult<()> {
        for provider in &self.providers {
            if provider.provider_name() == context.auth_method {
                return provider.revoke(context).await;
            }
        }
        Ok(())
    }

    fn provider_name(&self) -> &str {
        "ChainedAuthProvider"
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credentials_from_header() {
        let bearer = Credentials::from_header("Bearer token123");
        assert!(matches!(bearer, Some(Credentials::Bearer(_))));

        let api_key = Credentials::from_header("ApiKey secret123");
        assert!(matches!(api_key, Some(Credentials::ApiKey(_))));

        let invalid = Credentials::from_header("invalid");
        assert!(invalid.is_none());
    }

    #[test]
    fn test_identity() {
        let identity = Identity::new("user123")
            .with_name("John Doe")
            .with_email("john@example.com")
            .with_tenant("tenant1")
            .with_claim("department", "engineering");

        assert_eq!(identity.id, "user123");
        assert_eq!(identity.name, Some("John Doe".to_string()));
        assert_eq!(identity.tenant_id, Some("tenant1".to_string()));
    }

    #[test]
    fn test_auth_context() {
        let identity = Identity::new("user1");
        let ctx = AuthContext::new(identity, "test")
            .with_role("admin")
            .with_role("user")
            .with_permission("read")
            .with_permission("write");

        assert!(ctx.has_role("admin"));
        assert!(ctx.has_role("user"));
        assert!(!ctx.has_role("superadmin"));

        assert!(ctx.has_permission("read"));
        assert!(ctx.has_permission("write"));
        assert!(!ctx.has_permission("delete"));

        assert!(ctx.has_any_role(&["admin", "guest"]));
        assert!(ctx.has_all_roles(&["admin", "user"]));
        assert!(!ctx.has_all_roles(&["admin", "superadmin"]));
    }

    #[test]
    fn test_auth_context_expiry() {
        let identity = Identity::new("user1");
        let ctx = AuthContext::new(identity, "test").with_expiry(Duration::from_nanos(1));

        std::thread::sleep(Duration::from_millis(1));
        assert!(ctx.is_expired());
    }

    #[tokio::test]
    async fn test_api_key_auth() {
        let auth = ApiKeyAuth::new()
            .add_key("admin", "secret-key-123", &["admin", "read", "write"])
            .add_key("readonly", "readonly-key-456", &["read"]);

        // Valid admin key
        let ctx = auth
            .authenticate(&Credentials::ApiKey("secret-key-123".to_string()))
            .await
            .unwrap();
        assert_eq!(ctx.identity.id, "admin");
        assert!(ctx.has_permission("write"));

        // Valid readonly key
        let ctx2 = auth
            .authenticate(&Credentials::ApiKey("readonly-key-456".to_string()))
            .await
            .unwrap();
        assert_eq!(ctx2.identity.id, "readonly");
        assert!(ctx2.has_permission("read"));
        assert!(!ctx2.has_permission("write"));

        // Invalid key
        let result = auth
            .authenticate(&Credentials::ApiKey("invalid-key".to_string()))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_api_key_revocation() {
        let auth = ApiKeyAuth::new().add_key("user1", "key-to-revoke", &["read"]);

        // Works initially
        let result = auth
            .authenticate(&Credentials::ApiKey("key-to-revoke".to_string()))
            .await;
        assert!(result.is_ok());

        // Revoke
        auth.revoke_key("key-to-revoke");

        // Now fails
        let result = auth
            .authenticate(&Credentials::ApiKey("key-to-revoke".to_string()))
            .await;
        assert!(result.is_err());
    }

    #[cfg(feature = "auth")]
    #[tokio::test]
    async fn test_jwt_auth() {
        let auth = JwtAuth::with_secret("test-secret-key-256-bits-long!");

        // Generate a token
        let claims = JwtClaims {
            sub: "user123".to_string(),
            iat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            exp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 3600,
            iss: None,
            aud: None,
            roles: vec!["admin".to_string()],
            permissions: vec!["read".to_string(), "write".to_string()],
            tenant_id: Some("tenant1".to_string()),
            custom: HashMap::new(),
        };

        let token = auth.generate_token(&claims).unwrap();

        // Authenticate with the token
        let ctx = auth
            .authenticate(&Credentials::Bearer(token))
            .await
            .unwrap();

        assert_eq!(ctx.identity.id, "user123");
        assert!(ctx.has_role("admin"));
        assert!(ctx.has_permission("read"));
        assert_eq!(ctx.tenant_id(), Some("tenant1"));
    }

    #[tokio::test]
    async fn test_chained_auth() {
        let api_auth = Arc::new(ApiKeyAuth::new().add_key("api_user", "api-key-123", &["api"]));

        let chain = ChainedAuthProvider::new().with_provider(api_auth);

        // API key works
        let ctx = chain
            .authenticate(&Credentials::ApiKey("api-key-123".to_string()))
            .await
            .unwrap();
        assert_eq!(ctx.identity.id, "api_user");

        // Unknown key fails
        let result = chain
            .authenticate(&Credentials::ApiKey("unknown".to_string()))
            .await;
        assert!(result.is_err());
    }
}
