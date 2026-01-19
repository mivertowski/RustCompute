---
layout: default
title: Security Module
nav_order: 18
---

# Security Module

RingKernel v0.3.1 provides comprehensive enterprise-grade security features for GPU memory encryption, kernel sandboxing, authentication, authorization, TLS, and compliance reporting.

## Feature Flags

Enable enterprise security with feature flags in `Cargo.toml`:

```toml
[dependencies]
ringkernel-core = { version = "0.3.1", features = ["enterprise"] }

# Or select specific features:
ringkernel-core = { version = "0.3.1", features = ["crypto", "auth", "tls"] }
```

| Feature | Description |
|---------|-------------|
| `crypto` | AES-256-GCM, ChaCha20-Poly1305, Argon2 key derivation |
| `auth` | JWT authentication, API keys |
| `tls` | TLS/mTLS with rustls, certificate rotation |
| `rate-limiting` | Token bucket, sliding window, leaky bucket |
| `alerting` | Webhook alerts for Slack/Teams/PagerDuty |
| `enterprise` | All of the above combined |

## Memory Encryption

Encrypt sensitive data in GPU memory using industry-standard algorithms.

### Encryption Algorithms

| Algorithm | Key Size | Use Case |
|-----------|----------|----------|
| AES-256-GCM | 256-bit | General purpose, hardware acceleration |
| AES-128-GCM | 128-bit | Performance-critical, lower security |
| ChaCha20-Poly1305 | 256-bit | Software-optimized, constant-time |
| XChaCha20-Poly1305 | 256-bit | Extended nonce, safer for random nonces |

### Usage

```rust
use ringkernel_core::security::{MemoryEncryption, EncryptionConfig, Algorithm};

// Create encryption with AES-256-GCM
let config = EncryptionConfig::new()
    .algorithm(Algorithm::Aes256Gcm)
    .key_derivation(KeyDerivation::Argon2id)
    .auto_rotate(Duration::from_hours(24));

let encryption = MemoryEncryption::new(config)?;

// Encrypt control block
let encrypted = encryption.encrypt(&control_block)?;

// Decrypt
let decrypted: ControlBlock = encryption.decrypt(&encrypted)?;

// Key rotation
encryption.rotate_key()?;
```

### Key Derivation Functions

| Function | Description |
|----------|-------------|
| HKDF-SHA256 | Fast, suitable for high-entropy keys |
| HKDF-SHA384 | Stronger variant |
| Argon2id | Memory-hard, password-based |
| PBKDF2-SHA256 | Legacy compatibility |

```rust
// Derive key from password
let encryption = MemoryEncryption::from_password(
    "secure-passphrase",
    KeyDerivation::Argon2id {
        memory_cost: 65536,
        time_cost: 3,
        parallelism: 4,
    },
)?;
```

---

## Kernel Sandbox

Isolate kernels with resource limits and access controls.

### Resource Limits

```rust
use ringkernel_core::security::{KernelSandbox, ResourceLimits, SandboxPolicy};

let limits = ResourceLimits::new()
    .max_memory_bytes(1024 * 1024 * 1024)  // 1GB
    .max_execution_time(Duration::from_secs(60))
    .max_messages_per_second(10000)
    .max_k2k_connections(10);

let sandbox = KernelSandbox::new(limits);
```

### Sandbox Policies

```rust
// Restrictive policy for untrusted kernels
let policy = SandboxPolicy::restrictive()
    .deny_k2k_to(&["admin_kernel", "audit_kernel"])
    .memory_access(MemoryAccess::ReadOnly)
    .allow_topics(&["public.*"]);

// Permissive policy for trusted kernels
let policy = SandboxPolicy::permissive()
    .allow_all_k2k()
    .memory_access(MemoryAccess::ReadWrite);

let sandbox = KernelSandbox::with_policy(limits, policy);
```

### K2K Access Control Lists

```rust
// Allow list
let policy = SandboxPolicy::new()
    .k2k_allow_list(&[
        "processor_1",
        "processor_2",
        "aggregator",
    ]);

// Deny list
let policy = SandboxPolicy::new()
    .k2k_deny_list(&[
        "untrusted_kernel",
        "external_*",  // Wildcards supported
    ]);
```

### Violation Handling

```rust
// Check for violations
let violations = sandbox.check_violations(&kernel)?;

for violation in violations {
    match violation.violation_type {
        ViolationType::MemoryExceeded { used, limit } => {
            log::warn!("Memory limit exceeded: {} > {}", used, limit);
        }
        ViolationType::ExecutionTimeout { elapsed } => {
            log::warn!("Execution timeout: {:?}", elapsed);
        }
        ViolationType::UnauthorizedK2K { target } => {
            log::error!("Unauthorized K2K to: {}", target);
        }
        ViolationType::RateLimitExceeded { rate } => {
            log::warn!("Rate limit exceeded: {} msg/s", rate);
        }
    }
}
```

---

## Compliance Reporter

Generate audit-ready compliance documentation.

### Supported Standards

| Standard | Description |
|----------|-------------|
| SOC2 | Service Organization Control 2 |
| GDPR | General Data Protection Regulation |
| HIPAA | Health Insurance Portability and Accountability |
| PCI-DSS | Payment Card Industry Data Security Standard |
| ISO 27001 | Information Security Management |
| FedRAMP | Federal Risk and Authorization Management |
| NIST CSF | NIST Cybersecurity Framework |

### Generating Reports

```rust
use ringkernel_core::security::{ComplianceReporter, ComplianceStandard, ExportFormat};

let reporter = ComplianceReporter::new(&runtime);

// Generate SOC2 report
let report = reporter.generate(ComplianceStandard::Soc2)?;

// Check compliance status
for check in &report.checks {
    println!("{}: {:?}", check.name, check.status);
    if let Some(evidence) = &check.evidence {
        println!("  Evidence: {}", evidence);
    }
    if let Some(recommendation) = &check.recommendation {
        println!("  Recommendation: {}", recommendation);
    }
}

// Export to various formats
reporter.export(&report, ExportFormat::Json, "soc2_report.json")?;
reporter.export(&report, ExportFormat::Html, "soc2_report.html")?;
reporter.export(&report, ExportFormat::Markdown, "soc2_report.md")?;
reporter.export(&report, ExportFormat::Pdf, "soc2_report.pdf")?;
```

### Custom Compliance Checks

```rust
use ringkernel_core::security::{ComplianceCheck, ComplianceStatus};

let custom_check = ComplianceCheck::new("GPU_MEM_ENCRYPT")
    .name("GPU Memory Encryption")
    .description("All GPU memory must be encrypted at rest")
    .standard(ComplianceStandard::Custom("INTERNAL_SEC_001"))
    .check(|ctx| {
        if ctx.memory_encryption.is_enabled() {
            ComplianceStatus::Compliant {
                evidence: "AES-256-GCM encryption enabled".to_string(),
            }
        } else {
            ComplianceStatus::NonCompliant {
                finding: "GPU memory encryption not enabled".to_string(),
                recommendation: "Enable memory encryption with MemoryEncryption::new()".to_string(),
            }
        }
    });

reporter.add_check(custom_check);
```

---

## Best Practices

### Defense in Depth

```rust
// Layer 1: Memory encryption
let encryption = MemoryEncryption::new(EncryptionConfig::secure())?;

// Layer 2: Kernel sandboxing
let sandbox = KernelSandbox::with_policy(
    ResourceLimits::conservative(),
    SandboxPolicy::restrictive(),
);

// Layer 3: Compliance monitoring
let reporter = ComplianceReporter::new(&runtime)
    .enable_continuous_monitoring(Duration::from_hours(1));

// Apply all layers
let runtime = RuntimeBuilder::new()
    .production()
    .with_encryption(encryption)
    .with_sandbox(sandbox)
    .with_compliance(reporter)
    .build()?;
```

### Secure Configuration

```rust
// Never in production
let config = EncryptionConfig::new()
    .algorithm(Algorithm::None);  // INSECURE

// Always use secure defaults
let config = EncryptionConfig::secure();  // AES-256-GCM + Argon2id
```

### Audit Logging

```rust
// Enable security audit logging
let runtime = RuntimeBuilder::new()
    .production()
    .with_audit_logging(AuditConfig {
        log_k2k_messages: true,
        log_memory_access: true,
        log_violations: true,
        retention_days: 90,
    })
    .build()?;
```

---

## Authentication

RingKernel provides pluggable authentication with multiple providers.

### API Key Authentication

```rust
use ringkernel_core::prelude::*;

let auth = ApiKeyAuth::new()
    .add_key("sk-prod-abc123", Identity::new("service-a"))
    .add_key("sk-prod-xyz789", Identity::new("service-b"));

let result = auth.authenticate(&Credentials::ApiKey("sk-prod-abc123".into())).await;
match result {
    Ok(identity) => println!("Authenticated: {}", identity.name()),
    Err(AuthError::InvalidCredentials) => println!("Invalid API key"),
    Err(AuthError::Expired) => println!("API key expired"),
}
```

### JWT Authentication (requires `auth` feature)

```rust
use ringkernel_core::prelude::*;

let jwt_config = JwtConfig {
    secret: "your-secret-key".into(),
    issuer: Some("ringkernel".into()),
    audience: Some("api".into()),
    expiry_tolerance: Duration::from_secs(60),
};

let jwt_auth = JwtAuth::new(jwt_config);
let result = jwt_auth.authenticate(&Credentials::Bearer(token)).await?;
```

### Chained Authentication

```rust
// Try API key first, fall back to JWT
let auth = ChainedAuthProvider::new()
    .add(api_key_auth)
    .add(jwt_auth);

let identity = auth.authenticate(&credentials).await?;
```

---

## Authorization (RBAC)

Role-based access control with deny-by-default policy evaluation.

### Roles and Permissions

```rust
use ringkernel_core::prelude::*;

let policy = RbacPolicy::new()
    .grant(Subject::User("alice".into()), Role::Admin)
    .grant(Subject::User("bob".into()), Role::Developer)
    .grant(Subject::Service("worker".into()), Role::Operator);

let evaluator = PolicyEvaluator::new(policy);

// Check permissions
assert!(evaluator.check(&Subject::User("alice".into()), Permission::Admin));
assert!(evaluator.check(&Subject::User("bob".into()), Permission::Write));
assert!(!evaluator.check(&Subject::User("bob".into()), Permission::Admin));
```

### Resource Rules

```rust
let rule = ResourceRule::new("kernels/*")
    .allow(Permission::Read)
    .allow(Permission::Write)
    .deny_for(Subject::Service("untrusted".into()));

policy.add_rule(rule);
```

---

## TLS Support (requires `tls` feature)

TLS/mTLS with certificate rotation and SNI support.

### Server Configuration

```rust
use ringkernel_core::prelude::*;

let config = TlsConfigBuilder::new()
    .with_cert_file("server.crt")
    .with_key_file("server.key")
    .with_client_auth(ClientAuth::Required)  // mTLS
    .with_min_version(TlsVersion::Tls13)
    .build()?;

let acceptor = TlsAcceptor::new(config)?;
```

### Client Configuration

```rust
let connector = TlsConnector::new()
    .with_root_cert_file("ca.crt")
    .with_client_cert("client.crt", "client.key")
    .build()?;
```

### Certificate Rotation

```rust
let store = CertificateStore::new()
    .with_rotation_check_interval(Duration::from_secs(3600))
    .on_rotation(|new_cert| {
        println!("Certificate rotated: {:?}", new_cert.not_after());
    });

store.load_files("server.crt", "server.key")?;
```

---

## K2K Message Encryption (requires `crypto` feature)

Encrypt kernel-to-kernel messages with forward secrecy.

```rust
use ringkernel_core::prelude::*;

let encryptor = K2KEncryptor::new(K2KEncryptionConfig {
    algorithm: K2KEncryptionAlgorithm::Aes256Gcm,
    key: K2KKeyMaterial::generate()?,
});

// Create encrypted endpoint
let endpoint = EncryptedK2KBuilder::new(broker, kernel_id)
    .with_encryptor(encryptor)
    .build();

// Messages are automatically encrypted/decrypted
endpoint.send(dest_id, message).await?;
```

---

## Rate Limiting (requires `rate-limiting` feature)

Protect services from overload with configurable rate limiting.

### Token Bucket

```rust
use ringkernel_core::prelude::*;

let limiter = RateLimiterBuilder::new()
    .algorithm(RateLimitAlgorithm::TokenBucket)
    .rate(1000)  // 1000 requests per second
    .burst(100)  // Allow burst of 100
    .build();

match limiter.acquire() {
    Ok(_guard) => { /* proceed */ },
    Err(RateLimitError::Exceeded { retry_after }) => {
        println!("Rate limited, retry after {:?}", retry_after);
    }
}
```

### Sliding Window

```rust
let limiter = RateLimiterBuilder::new()
    .algorithm(RateLimitAlgorithm::SlidingWindow)
    .rate(100)
    .window(Duration::from_secs(60))  // 100 per minute
    .build();
```

---

## Secrets Management

Secure storage and rotation of sensitive keys.

### Secret Stores

```rust
use ringkernel_core::prelude::*;

// In-memory (development)
let store = InMemorySecretStore::new();
store.set("api_key", SecretValue::new("sk-secret"))?;

// Environment variables
let env_store = EnvVarSecretStore::with_prefix("RINGKERNEL_");
let key = env_store.get(&SecretKey::new("DB_PASSWORD"))?;

// Cached with TTL
let cached = CachedSecretStore::new(underlying_store)
    .with_ttl(Duration::from_secs(300));

// Chained (try multiple stores)
let chained = ChainedSecretStore::new()
    .add(env_store)
    .add(cached_vault_store);
```

### Key Rotation

```rust
let rotation_manager = KeyRotationManager::new(secret_store)
    .with_rotation_interval(Duration::from_days(30))
    .on_rotation(|key_name, _old, _new| {
        println!("Rotated key: {}", key_name);
    });

rotation_manager.start();
```

---

## Multi-tenancy

Isolate tenants with resource quotas and usage tracking.

```rust
use ringkernel_core::prelude::*;

let registry = TenantRegistry::new();
registry.register("tenant-a", ResourceQuota {
    max_memory_bytes: 1024 * 1024 * 1024,  // 1 GB
    max_kernels: 10,
    max_message_rate: 10_000,
})?;

let ctx = TenantContext::new("tenant-a");
ctx.track_memory(1024 * 1024)?;

let utilization = ctx.quota_utilization();
println!("Memory: {:.1}%", utilization.memory_percent());
```

---

## Next: [ML Bridges](./18-ml-bridges.md)
