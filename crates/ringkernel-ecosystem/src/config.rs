//! Configuration management for RingKernel.
//!
//! This module provides configuration file loading and management
//! using the `config` crate, supporting TOML files and environment variables.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_ecosystem::config::{RingKernelConfig, load_config};
//!
//! let config = load_config("config/ringkernel.toml")?;
//! let runtime_config = config.to_runtime_config();
//! ```

use config::{Config, ConfigError, Environment, File, FileFormat};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Main configuration structure for RingKernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingKernelConfig {
    /// Backend to use (cpu, cuda, metal, wgpu).
    #[serde(default = "default_backend")]
    pub backend: String,

    /// Device ID to use.
    #[serde(default)]
    pub device_id: usize,

    /// Default queue capacity.
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,

    /// Bridge poll interval in microseconds.
    #[serde(default = "default_poll_interval")]
    pub poll_interval_us: u64,

    /// Enable telemetry collection.
    #[serde(default = "default_telemetry")]
    pub telemetry_enabled: bool,

    /// Kernel configuration.
    #[serde(default)]
    pub kernel: KernelConfig,

    /// Memory configuration.
    #[serde(default)]
    pub memory: MemoryConfig,

    /// Network configuration.
    #[serde(default)]
    pub network: NetworkConfig,

    /// Logging configuration.
    #[serde(default)]
    pub logging: LoggingConfig,
}

fn default_backend() -> String {
    "cpu".to_string()
}

fn default_queue_capacity() -> usize {
    1024
}

fn default_poll_interval() -> u64 {
    100
}

fn default_telemetry() -> bool {
    true
}

impl Default for RingKernelConfig {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            device_id: 0,
            queue_capacity: default_queue_capacity(),
            poll_interval_us: default_poll_interval(),
            telemetry_enabled: default_telemetry(),
            kernel: KernelConfig::default(),
            memory: MemoryConfig::default(),
            network: NetworkConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

/// Kernel-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Default block size.
    #[serde(default = "default_block_size")]
    pub default_block_size: usize,

    /// Maximum kernels per runtime.
    #[serde(default = "default_max_kernels")]
    pub max_kernels: usize,

    /// Default timeout for operations in milliseconds.
    #[serde(default = "default_kernel_timeout")]
    pub timeout_ms: u64,

    /// Enable auto-restart on failure.
    #[serde(default)]
    pub auto_restart: bool,

    /// Maximum restart attempts.
    #[serde(default = "default_max_restarts")]
    pub max_restarts: u32,
}

fn default_block_size() -> usize {
    256
}

fn default_max_kernels() -> usize {
    64
}

fn default_kernel_timeout() -> u64 {
    30_000
}

fn default_max_restarts() -> u32 {
    3
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            default_block_size: default_block_size(),
            max_kernels: default_max_kernels(),
            timeout_ms: default_kernel_timeout(),
            auto_restart: false,
            max_restarts: default_max_restarts(),
        }
    }
}

/// Memory configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Memory pool size in bytes.
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,

    /// Enable memory pooling.
    #[serde(default = "default_enable_pooling")]
    pub enable_pooling: bool,

    /// Preallocate memory on startup.
    #[serde(default)]
    pub preallocate: bool,

    /// Maximum allocation size in bytes.
    #[serde(default = "default_max_allocation")]
    pub max_allocation: usize,
}

fn default_pool_size() -> usize {
    256 * 1024 * 1024 // 256MB
}

fn default_enable_pooling() -> bool {
    true
}

fn default_max_allocation() -> usize {
    64 * 1024 * 1024 // 64MB
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            pool_size: default_pool_size(),
            enable_pooling: default_enable_pooling(),
            preallocate: false,
            max_allocation: default_max_allocation(),
        }
    }
}

/// Network configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// gRPC server address.
    #[serde(default = "default_grpc_address")]
    pub grpc_address: String,

    /// HTTP server address.
    #[serde(default = "default_http_address")]
    pub http_address: String,

    /// Enable gRPC server.
    #[serde(default)]
    pub enable_grpc: bool,

    /// Enable HTTP server.
    #[serde(default)]
    pub enable_http: bool,

    /// TLS configuration.
    #[serde(default)]
    pub tls: Option<TlsConfig>,
}

fn default_grpc_address() -> String {
    "[::1]:50051".to_string()
}

fn default_http_address() -> String {
    "0.0.0.0:3000".to_string()
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            grpc_address: default_grpc_address(),
            http_address: default_http_address(),
            enable_grpc: false,
            enable_http: false,
            tls: None,
        }
    }
}

/// TLS configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Path to certificate file.
    pub cert_path: String,
    /// Path to key file.
    pub key_path: String,
    /// Path to CA certificate for client verification.
    pub ca_path: Option<String>,
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error).
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Log format (pretty, json, compact).
    #[serde(default = "default_log_format")]
    pub format: String,

    /// Enable file logging.
    #[serde(default)]
    pub enable_file: bool,

    /// Log file path.
    #[serde(default = "default_log_path")]
    pub file_path: String,

    /// Enable structured logging.
    #[serde(default = "default_structured")]
    pub structured: bool,
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_log_format() -> String {
    "pretty".to_string()
}

fn default_log_path() -> String {
    "logs/ringkernel.log".to_string()
}

fn default_structured() -> bool {
    false
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
            enable_file: false,
            file_path: default_log_path(),
            structured: default_structured(),
        }
    }
}

impl RingKernelConfig {
    /// Load configuration from a file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let builder = Config::builder()
            .add_source(File::from(path.as_ref()))
            .add_source(Environment::with_prefix("RINGKERNEL").separator("__"));

        builder.build()?.try_deserialize()
    }

    /// Load configuration with fallback to defaults.
    pub fn load_or_default<P: AsRef<Path>>(path: P) -> Self {
        Self::load(path).unwrap_or_default()
    }

    /// Create from environment variables only.
    pub fn from_env() -> Result<Self, ConfigError> {
        let builder =
            Config::builder().add_source(Environment::with_prefix("RINGKERNEL").separator("__"));

        builder.build()?.try_deserialize()
    }

    /// Get poll interval as Duration.
    pub fn poll_interval(&self) -> Duration {
        Duration::from_micros(self.poll_interval_us)
    }

    /// Get kernel timeout as Duration.
    pub fn kernel_timeout(&self) -> Duration {
        Duration::from_millis(self.kernel.timeout_ms)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.queue_capacity == 0 {
            return Err("Queue capacity must be greater than 0".to_string());
        }

        if self.kernel.default_block_size == 0 {
            return Err("Block size must be greater than 0".to_string());
        }

        if self.memory.pool_size < self.memory.max_allocation {
            return Err("Pool size must be >= max allocation size".to_string());
        }

        let valid_backends = ["cpu", "cuda", "metal", "wgpu"];
        if !valid_backends.contains(&self.backend.as_str()) {
            return Err(format!(
                "Invalid backend '{}'. Valid options: {:?}",
                self.backend, valid_backends
            ));
        }

        Ok(())
    }
}

/// Load configuration from a TOML file.
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<RingKernelConfig, ConfigError> {
    RingKernelConfig::load(path)
}

/// Load configuration from TOML string.
pub fn load_config_from_str(content: &str) -> Result<RingKernelConfig, ConfigError> {
    let builder = Config::builder()
        .add_source(File::from_str(content, FileFormat::Toml))
        .add_source(Environment::with_prefix("RINGKERNEL").separator("__"));

    builder.build()?.try_deserialize()
}

/// Configuration builder for programmatic configuration.
pub struct ConfigBuilder {
    config: RingKernelConfig,
}

impl ConfigBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self {
            config: RingKernelConfig::default(),
        }
    }

    /// Set the backend.
    pub fn backend(mut self, backend: impl Into<String>) -> Self {
        self.config.backend = backend.into();
        self
    }

    /// Set the device ID.
    pub fn device_id(mut self, id: usize) -> Self {
        self.config.device_id = id;
        self
    }

    /// Set queue capacity.
    pub fn queue_capacity(mut self, capacity: usize) -> Self {
        self.config.queue_capacity = capacity;
        self
    }

    /// Set poll interval.
    pub fn poll_interval(mut self, interval: Duration) -> Self {
        self.config.poll_interval_us = interval.as_micros() as u64;
        self
    }

    /// Enable telemetry.
    pub fn telemetry(mut self, enabled: bool) -> Self {
        self.config.telemetry_enabled = enabled;
        self
    }

    /// Set kernel timeout.
    pub fn kernel_timeout(mut self, timeout: Duration) -> Self {
        self.config.kernel.timeout_ms = timeout.as_millis() as u64;
        self
    }

    /// Set memory pool size.
    pub fn memory_pool_size(mut self, size: usize) -> Self {
        self.config.memory.pool_size = size;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> Result<RingKernelConfig, String> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RingKernelConfig::default();
        assert_eq!(config.backend, "cpu");
        assert_eq!(config.device_id, 0);
        assert_eq!(config.queue_capacity, 1024);
        assert!(config.telemetry_enabled);
    }

    #[test]
    fn test_config_validation() {
        let config = RingKernelConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid = RingKernelConfig::default();
        invalid.queue_capacity = 0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .backend("cuda")
            .device_id(1)
            .queue_capacity(2048)
            .build()
            .unwrap();

        assert_eq!(config.backend, "cuda");
        assert_eq!(config.device_id, 1);
        assert_eq!(config.queue_capacity, 2048);
    }

    #[test]
    fn test_poll_interval_conversion() {
        let config = RingKernelConfig::default();
        assert_eq!(config.poll_interval(), Duration::from_micros(100));
    }

    #[test]
    fn test_load_from_str() {
        let toml = r#"
            backend = "cuda"
            device_id = 2
            queue_capacity = 4096
        "#;

        let config = load_config_from_str(toml).unwrap();
        assert_eq!(config.backend, "cuda");
        assert_eq!(config.device_id, 2);
        assert_eq!(config.queue_capacity, 4096);
    }
}
