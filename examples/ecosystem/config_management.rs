//! # Configuration Management for RingKernel
//!
//! This example demonstrates flexible configuration management using
//! TOML files, environment variables, and programmatic configuration.
//!
//! ## Configuration Sources
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    Configuration Hierarchy                          │
//! │                                                                     │
//! │  Priority (highest to lowest):                                      │
//! │                                                                     │
//! │  1. ┌─────────────────────┐  RINGKERNEL_BACKEND=cuda               │
//! │     │ Environment Vars    │  RINGKERNEL_MEMORY__POOL_SIZE=512MB    │
//! │     └─────────────────────┘                                        │
//! │                ▼                                                    │
//! │  2. ┌─────────────────────┐  config/local.toml                     │
//! │     │ Local Config File   │  (git-ignored, developer settings)     │
//! │     └─────────────────────┘                                        │
//! │                ▼                                                    │
//! │  3. ┌─────────────────────┐  config/production.toml                │
//! │     │ Environment Config  │  config/staging.toml                   │
//! │     └─────────────────────┘                                        │
//! │                ▼                                                    │
//! │  4. ┌─────────────────────┐  Sensible defaults for all options     │
//! │     │ Default Values      │  Works out-of-the-box                  │
//! │     └─────────────────────┘                                        │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Use Cases
//!
//! - **Local Development**: Override settings without changing files
//! - **CI/CD Pipelines**: Configure via environment variables
//! - **Multi-Environment**: Different configs for dev/staging/prod
//! - **Containerization**: Kubernetes ConfigMaps and Secrets
//!
//! ## Run this example:
//! ```bash
//! cargo run --example config_management --features "ringkernel-ecosystem/config"
//! ```

use std::collections::HashMap;
use std::time::Duration;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== RingKernel Configuration Management ===\n");

    // ====== Default Configuration ======
    println!("=== Default Configuration ===\n");

    let default_config = RingKernelConfig::default();
    print_config("Default Configuration", &default_config);

    // ====== TOML Configuration ======
    println!("\n=== Loading from TOML ===\n");

    let toml_content = r#"
# RingKernel Configuration File
# This is an example production configuration

backend = "cuda"
device_id = 0
queue_capacity = 2048
poll_interval_us = 50
telemetry_enabled = true

[kernel]
default_block_size = 512
max_kernels = 128
timeout_ms = 60000
auto_restart = true
max_restarts = 5

[memory]
pool_size = 536870912  # 512MB
enable_pooling = true
preallocate = true
max_allocation = 134217728  # 128MB

[network]
grpc_address = "[::1]:50051"
http_address = "0.0.0.0:8080"
enable_grpc = true
enable_http = true

[network.tls]
cert_path = "/etc/ringkernel/certs/server.crt"
key_path = "/etc/ringkernel/certs/server.key"

[logging]
level = "info"
format = "json"
enable_file = true
file_path = "/var/log/ringkernel/app.log"
structured = true
"#;

    let toml_config = RingKernelConfig::from_toml(toml_content)?;
    print_config("TOML Configuration", &toml_config);

    // ====== Environment Variables ======
    println!("\n=== Environment Variable Overrides ===\n");

    println!("Environment variables follow the pattern:");
    println!("  RINGKERNEL_<SECTION>__<KEY>");
    println!();
    println!("Examples:");
    println!("  RINGKERNEL_BACKEND=cuda");
    println!("  RINGKERNEL_DEVICE_ID=1");
    println!("  RINGKERNEL_MEMORY__POOL_SIZE=1073741824");
    println!("  RINGKERNEL_KERNEL__TIMEOUT_MS=120000");
    println!("  RINGKERNEL_LOGGING__LEVEL=debug");

    // Simulate environment overrides
    let env_overrides = HashMap::from([
        ("RINGKERNEL_BACKEND", "metal"),
        ("RINGKERNEL_DEVICE_ID", "2"),
        ("RINGKERNEL_MEMORY__POOL_SIZE", "1073741824"),
    ]);

    println!("\nApplying overrides:");
    for (key, value) in &env_overrides {
        println!("  {} = {}", key, value);
    }

    let mut env_config = toml_config.clone();
    env_config.apply_env_overrides(&env_overrides);
    print_config("\nConfiguration with Env Overrides", &env_config);

    // ====== Programmatic Configuration ======
    println!("\n=== Programmatic Configuration (Builder Pattern) ===\n");

    let builder_config = RingKernelConfigBuilder::new()
        .backend("cuda")
        .device_id(0)
        .queue_capacity(4096)
        .poll_interval(Duration::from_micros(25))
        .telemetry(true)
        .kernel_timeout(Duration::from_secs(120))
        .memory_pool_size(1024 * 1024 * 1024) // 1GB
        .enable_pooling(true)
        .preallocate(true)
        .grpc_address("[::1]:50051")
        .http_address("0.0.0.0:3000")
        .log_level("info")
        .build()?;

    print_config("Builder Configuration", &builder_config);

    // ====== Configuration Validation ======
    println!("\n=== Configuration Validation ===\n");

    let validation_tests = vec![
        ("Valid config", RingKernelConfig::default()),
        ("Zero queue capacity", {
            RingKernelConfig {
                queue_capacity: 0,
                ..Default::default()
            }
        }),
        ("Invalid backend", {
            RingKernelConfig {
                backend: "invalid".to_string(),
                ..Default::default()
            }
        }),
        ("Pool < max allocation", {
            let mut c = RingKernelConfig::default();
            c.memory.pool_size = 1024;
            c.memory.max_allocation = 2048;
            c
        }),
    ];

    for (name, config) in validation_tests {
        match config.validate() {
            Ok(()) => println!("  [OK] {}", name),
            Err(e) => println!("  [FAIL] {}: {}", name, e),
        }
    }

    // ====== Multi-Environment Configuration ======
    println!("\n=== Multi-Environment Setup ===\n");

    let environments = vec![
        ("development", DevelopmentConfig::generate()),
        ("staging", StagingConfig::generate()),
        ("production", ProductionConfig::generate()),
    ];

    for (env_name, config) in environments {
        println!("{}:", env_name.to_uppercase());
        println!("  Backend: {}", config.backend);
        println!("  Queue capacity: {}", config.queue_capacity);
        println!(
            "  Memory pool: {} MB",
            config.memory.pool_size / (1024 * 1024)
        );
        println!("  Telemetry: {}", config.telemetry_enabled);
        println!("  Log level: {}", config.logging.level);
        println!();
    }

    // ====== Configuration Hot Reload ======
    println!("=== Configuration Hot Reload ===\n");

    println!("For production systems, implement hot reload:");
    println!();
    println!("```rust");
    println!("let config_watcher = ConfigWatcher::new(\"config/app.toml\");");
    println!("config_watcher.on_change(|new_config| {{");
    println!("    // Validate new configuration");
    println!("    if new_config.validate().is_ok() {{");
    println!("        runtime.update_config(new_config);");
    println!("        tracing::info!(\"Configuration reloaded\");");
    println!("    }}");
    println!("}});");
    println!("```");

    // ====== Kubernetes Integration ======
    println!("\n=== Kubernetes ConfigMap Example ===\n");

    println!("```yaml");
    println!("apiVersion: v1");
    println!("kind: ConfigMap");
    println!("metadata:");
    println!("  name: ringkernel-config");
    println!("data:");
    println!("  config.toml: |");
    println!("    backend = \"cuda\"");
    println!("    queue_capacity = 4096");
    println!("    [memory]");
    println!("    pool_size = 1073741824");
    println!("---");
    println!("apiVersion: v1");
    println!("kind: Secret");
    println!("metadata:");
    println!("  name: ringkernel-secrets");
    println!("type: Opaque");
    println!("data:");
    println!("  tls-cert: <base64-encoded-cert>");
    println!("  tls-key: <base64-encoded-key>");
    println!("```");

    println!("\n=== Example completed! ===");
    Ok(())
}

// ============ Configuration Types ============

#[derive(Debug, Clone)]
struct RingKernelConfig {
    backend: String,
    device_id: usize,
    queue_capacity: usize,
    poll_interval_us: u64,
    telemetry_enabled: bool,
    kernel: KernelConfig,
    memory: MemoryConfig,
    network: NetworkConfig,
    logging: LoggingConfig,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct KernelConfig {
    default_block_size: usize,
    max_kernels: usize,
    timeout_ms: u64,
    auto_restart: bool,
    max_restarts: u32,
}

#[derive(Debug, Clone)]
struct MemoryConfig {
    pool_size: usize,
    enable_pooling: bool,
    preallocate: bool,
    max_allocation: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct NetworkConfig {
    grpc_address: String,
    http_address: String,
    enable_grpc: bool,
    enable_http: bool,
    tls: Option<TlsConfig>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TlsConfig {
    cert_path: String,
    key_path: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct LoggingConfig {
    level: String,
    format: String,
    enable_file: bool,
    file_path: String,
    structured: bool,
}

impl Default for RingKernelConfig {
    fn default() -> Self {
        Self {
            backend: "cpu".to_string(),
            device_id: 0,
            queue_capacity: 1024,
            poll_interval_us: 100,
            telemetry_enabled: true,
            kernel: KernelConfig::default(),
            memory: MemoryConfig::default(),
            network: NetworkConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            default_block_size: 256,
            max_kernels: 64,
            timeout_ms: 30_000,
            auto_restart: false,
            max_restarts: 3,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            pool_size: 256 * 1024 * 1024, // 256MB
            enable_pooling: true,
            preallocate: false,
            max_allocation: 64 * 1024 * 1024, // 64MB
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            grpc_address: "[::1]:50051".to_string(),
            http_address: "0.0.0.0:3000".to_string(),
            enable_grpc: false,
            enable_http: false,
            tls: None,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "pretty".to_string(),
            enable_file: false,
            file_path: "logs/ringkernel.log".to_string(),
            structured: false,
        }
    }
}

impl RingKernelConfig {
    fn from_toml(content: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // In real implementation, use the `toml` crate
        // This is a simplified parser for demonstration
        let mut config = Self::default();

        for line in content.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() {
                continue;
            }

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"');

                match key {
                    "backend" => config.backend = value.to_string(),
                    "device_id" => config.device_id = value.parse().unwrap_or(0),
                    "queue_capacity" => config.queue_capacity = value.parse().unwrap_or(1024),
                    "poll_interval_us" => config.poll_interval_us = value.parse().unwrap_or(100),
                    "telemetry_enabled" => config.telemetry_enabled = value == "true",
                    _ => {}
                }
            }
        }

        Ok(config)
    }

    fn apply_env_overrides(&mut self, overrides: &HashMap<&str, &str>) {
        if let Some(value) = overrides.get("RINGKERNEL_BACKEND") {
            self.backend = value.to_string();
        }
        if let Some(value) = overrides.get("RINGKERNEL_DEVICE_ID") {
            self.device_id = value.parse().unwrap_or(self.device_id);
        }
        if let Some(value) = overrides.get("RINGKERNEL_MEMORY__POOL_SIZE") {
            self.memory.pool_size = value.parse().unwrap_or(self.memory.pool_size);
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.queue_capacity == 0 {
            return Err("Queue capacity must be greater than 0".to_string());
        }

        let valid_backends = ["cpu", "cuda", "metal", "wgpu"];
        if !valid_backends.contains(&self.backend.as_str()) {
            return Err(format!("Invalid backend: {}", self.backend));
        }

        if self.memory.pool_size < self.memory.max_allocation {
            return Err("Pool size must be >= max allocation".to_string());
        }

        Ok(())
    }
}

// ============ Configuration Builder ============

struct RingKernelConfigBuilder {
    config: RingKernelConfig,
}

impl RingKernelConfigBuilder {
    fn new() -> Self {
        Self {
            config: RingKernelConfig::default(),
        }
    }

    fn backend(mut self, backend: &str) -> Self {
        self.config.backend = backend.to_string();
        self
    }

    fn device_id(mut self, id: usize) -> Self {
        self.config.device_id = id;
        self
    }

    fn queue_capacity(mut self, capacity: usize) -> Self {
        self.config.queue_capacity = capacity;
        self
    }

    fn poll_interval(mut self, interval: Duration) -> Self {
        self.config.poll_interval_us = interval.as_micros() as u64;
        self
    }

    fn telemetry(mut self, enabled: bool) -> Self {
        self.config.telemetry_enabled = enabled;
        self
    }

    fn kernel_timeout(mut self, timeout: Duration) -> Self {
        self.config.kernel.timeout_ms = timeout.as_millis() as u64;
        self
    }

    fn memory_pool_size(mut self, size: usize) -> Self {
        self.config.memory.pool_size = size;
        self
    }

    fn enable_pooling(mut self, enabled: bool) -> Self {
        self.config.memory.enable_pooling = enabled;
        self
    }

    fn preallocate(mut self, enabled: bool) -> Self {
        self.config.memory.preallocate = enabled;
        self
    }

    fn grpc_address(mut self, address: &str) -> Self {
        self.config.network.grpc_address = address.to_string();
        self.config.network.enable_grpc = true;
        self
    }

    fn http_address(mut self, address: &str) -> Self {
        self.config.network.http_address = address.to_string();
        self.config.network.enable_http = true;
        self
    }

    fn log_level(mut self, level: &str) -> Self {
        self.config.logging.level = level.to_string();
        self
    }

    fn build(self) -> Result<RingKernelConfig, String> {
        self.config.validate()?;
        Ok(self.config)
    }
}

// ============ Environment-Specific Configs ============

struct DevelopmentConfig;
struct StagingConfig;
struct ProductionConfig;

impl DevelopmentConfig {
    fn generate() -> RingKernelConfig {
        let mut config = RingKernelConfig {
            backend: "cpu".to_string(),
            queue_capacity: 256,
            telemetry_enabled: true,
            ..Default::default()
        };
        config.logging.level = "debug".to_string();
        config.logging.format = "pretty".to_string();
        config
    }
}

impl StagingConfig {
    fn generate() -> RingKernelConfig {
        let mut config = RingKernelConfig {
            backend: "cuda".to_string(),
            queue_capacity: 1024,
            telemetry_enabled: true,
            ..Default::default()
        };
        config.memory.pool_size = 512 * 1024 * 1024;
        config.logging.level = "info".to_string();
        config.logging.format = "json".to_string();
        config
    }
}

impl ProductionConfig {
    fn generate() -> RingKernelConfig {
        let mut config = RingKernelConfig {
            backend: "cuda".to_string(),
            queue_capacity: 4096,
            telemetry_enabled: true,
            ..Default::default()
        };
        config.memory.pool_size = 1024 * 1024 * 1024;
        config.memory.preallocate = true;
        config.kernel.auto_restart = true;
        config.logging.level = "warn".to_string();
        config.logging.format = "json".to_string();
        config.logging.structured = true;
        config
    }
}

// ============ Helper Functions ============

fn print_config(title: &str, config: &RingKernelConfig) {
    println!("{}:", title);
    println!("  Backend: {}", config.backend);
    println!("  Device ID: {}", config.device_id);
    println!("  Queue capacity: {}", config.queue_capacity);
    println!("  Poll interval: {} µs", config.poll_interval_us);
    println!("  Telemetry: {}", config.telemetry_enabled);
    println!("  Kernel:");
    println!("    Block size: {}", config.kernel.default_block_size);
    println!("    Max kernels: {}", config.kernel.max_kernels);
    println!("    Timeout: {} ms", config.kernel.timeout_ms);
    println!("  Memory:");
    println!(
        "    Pool size: {} MB",
        config.memory.pool_size / (1024 * 1024)
    );
    println!("    Pooling: {}", config.memory.enable_pooling);
    println!("  Logging:");
    println!("    Level: {}", config.logging.level);
    println!("    Format: {}", config.logging.format);
}
