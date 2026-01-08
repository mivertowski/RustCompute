---
layout: default
title: Security Module
nav_order: 18
---

# Security Module

RingKernel provides enterprise-grade security features for GPU memory encryption, kernel sandboxing, and compliance reporting.

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

## Next: [ML Bridges](./18-ml-bridges.md)
