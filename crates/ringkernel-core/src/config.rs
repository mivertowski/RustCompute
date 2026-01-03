//! Unified configuration for RingKernel enterprise features.
//!
//! This module provides a comprehensive configuration system that ties together
//! observability, health monitoring, multi-GPU coordination, and migration features.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::config::{RingKernelConfig, ConfigBuilder};
//!
//! let config = ConfigBuilder::new()
//!     .with_observability(|obs| obs
//!         .enable_tracing(true)
//!         .enable_metrics(true)
//!         .metrics_port(9090))
//!     .with_health(|health| health
//!         .heartbeat_interval(Duration::from_secs(5))
//!         .circuit_breaker_threshold(5))
//!     .with_multi_gpu(|gpu| gpu
//!         .load_balancing(LoadBalancingStrategy::LeastLoaded)
//!         .enable_p2p(true))
//!     .build()?;
//!
//! let runtime = RingKernelRuntime::with_config(config)?;
//! ```
//!
//! # Configuration File Support
//!
//! With the `config-file` feature enabled, you can load configurations from TOML or YAML files:
//!
//! ```ignore
//! use ringkernel_core::config::RingKernelConfig;
//!
//! // Load from TOML file
//! let config = RingKernelConfig::from_toml_file("config.toml")?;
//!
//! // Load from YAML file
//! let config = RingKernelConfig::from_yaml_file("config.yaml")?;
//!
//! // Load from string
//! let config = RingKernelConfig::from_toml_str(toml_content)?;
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use crate::error::{Result, RingKernelError};
use crate::health::{BackoffStrategy, CircuitBreakerConfig, LoadSheddingPolicy};
use crate::multi_gpu::LoadBalancingStrategy;
use crate::runtime::Backend;

#[cfg(feature = "config-file")]
use std::path::Path;

// ============================================================================
// Main Configuration
// ============================================================================

/// Unified configuration for RingKernel.
#[derive(Debug, Clone)]
pub struct RingKernelConfig {
    /// General settings.
    pub general: GeneralConfig,
    /// Observability settings.
    pub observability: ObservabilityConfig,
    /// Health monitoring settings.
    pub health: HealthConfig,
    /// Multi-GPU settings.
    pub multi_gpu: MultiGpuConfig,
    /// Migration settings.
    pub migration: MigrationConfig,
    /// Custom settings.
    pub custom: HashMap<String, String>,
}

impl Default for RingKernelConfig {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            observability: ObservabilityConfig::default(),
            health: HealthConfig::default(),
            multi_gpu: MultiGpuConfig::default(),
            migration: MigrationConfig::default(),
            custom: HashMap::new(),
        }
    }
}

impl RingKernelConfig {
    /// Create a new configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for fluent configuration.
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::new()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        self.general.validate()?;
        self.observability.validate()?;
        self.health.validate()?;
        self.multi_gpu.validate()?;
        self.migration.validate()?;
        Ok(())
    }

    /// Get a custom setting by key.
    pub fn get_custom(&self, key: &str) -> Option<&str> {
        self.custom.get(key).map(|s| s.as_str())
    }

    /// Set a custom setting.
    pub fn set_custom(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.custom.insert(key.into(), value.into());
    }
}

// ============================================================================
// General Configuration
// ============================================================================

/// General runtime settings.
#[derive(Debug, Clone)]
pub struct GeneralConfig {
    /// Preferred backend.
    pub backend: Backend,
    /// Application name (for metrics/tracing).
    pub app_name: String,
    /// Application version.
    pub app_version: String,
    /// Environment (dev, staging, prod).
    pub environment: Environment,
    /// Log level.
    pub log_level: LogLevel,
    /// Data directory for checkpoints, logs, etc.
    pub data_dir: Option<PathBuf>,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            app_name: "ringkernel".to_string(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            environment: Environment::Development,
            log_level: LogLevel::Info,
            data_dir: None,
        }
    }
}

impl GeneralConfig {
    /// Validate general configuration.
    pub fn validate(&self) -> Result<()> {
        if self.app_name.is_empty() {
            return Err(RingKernelError::InvalidConfig(
                "app_name cannot be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// Runtime environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Environment {
    /// Development environment.
    #[default]
    Development,
    /// Staging/testing environment.
    Staging,
    /// Production environment.
    Production,
}

impl Environment {
    /// Returns true if this is a production environment.
    pub fn is_production(&self) -> bool {
        matches!(self, Environment::Production)
    }

    /// Get the environment as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Environment::Development => "development",
            Environment::Staging => "staging",
            Environment::Production => "production",
        }
    }
}

/// Log level configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogLevel {
    /// Trace level (most verbose).
    Trace,
    /// Debug level.
    Debug,
    /// Info level (default).
    #[default]
    Info,
    /// Warning level.
    Warn,
    /// Error level (least verbose).
    Error,
}

impl LogLevel {
    /// Get the log level as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

// ============================================================================
// Observability Configuration
// ============================================================================

/// Observability settings.
#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    /// Enable tracing.
    pub tracing_enabled: bool,
    /// Enable metrics.
    pub metrics_enabled: bool,
    /// Metrics port for Prometheus scraping.
    pub metrics_port: u16,
    /// Metrics path (default: /metrics).
    pub metrics_path: String,
    /// Trace sampling rate (0.0 to 1.0).
    pub trace_sample_rate: f64,
    /// Enable Grafana dashboard generation.
    pub grafana_enabled: bool,
    /// OTLP endpoint for trace export.
    pub otlp_endpoint: Option<String>,
    /// Custom metric labels.
    pub metric_labels: HashMap<String, String>,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            tracing_enabled: true,
            metrics_enabled: true,
            metrics_port: 9090,
            metrics_path: "/metrics".to_string(),
            trace_sample_rate: 1.0,
            grafana_enabled: false,
            otlp_endpoint: None,
            metric_labels: HashMap::new(),
        }
    }
}

impl ObservabilityConfig {
    /// Validate observability configuration.
    pub fn validate(&self) -> Result<()> {
        if self.trace_sample_rate < 0.0 || self.trace_sample_rate > 1.0 {
            return Err(RingKernelError::InvalidConfig(format!(
                "trace_sample_rate must be between 0.0 and 1.0, got {}",
                self.trace_sample_rate
            )));
        }
        if self.metrics_port == 0 {
            return Err(RingKernelError::InvalidConfig(
                "metrics_port cannot be 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Health Configuration
// ============================================================================

/// Health monitoring settings.
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Enable health checks.
    pub health_checks_enabled: bool,
    /// Health check interval.
    pub check_interval: Duration,
    /// Heartbeat timeout.
    pub heartbeat_timeout: Duration,
    /// Circuit breaker configuration.
    pub circuit_breaker: CircuitBreakerConfig,
    /// Retry policy for transient failures.
    pub retry: RetryConfig,
    /// Load shedding policy.
    pub load_shedding: LoadSheddingPolicy,
    /// Kernel watchdog enabled.
    pub watchdog_enabled: bool,
    /// Watchdog failure threshold.
    pub watchdog_failure_threshold: u32,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            health_checks_enabled: true,
            check_interval: Duration::from_secs(10),
            heartbeat_timeout: Duration::from_secs(30),
            circuit_breaker: CircuitBreakerConfig::default(),
            retry: RetryConfig::default(),
            load_shedding: LoadSheddingPolicy::default(),
            watchdog_enabled: true,
            watchdog_failure_threshold: 3,
        }
    }
}

impl HealthConfig {
    /// Validate health configuration.
    pub fn validate(&self) -> Result<()> {
        if self.check_interval.is_zero() {
            return Err(RingKernelError::InvalidConfig(
                "check_interval cannot be zero".to_string(),
            ));
        }
        if self.heartbeat_timeout.is_zero() {
            return Err(RingKernelError::InvalidConfig(
                "heartbeat_timeout cannot be zero".to_string(),
            ));
        }
        if self.heartbeat_timeout < self.check_interval {
            return Err(RingKernelError::InvalidConfig(
                "heartbeat_timeout should be >= check_interval".to_string(),
            ));
        }
        Ok(())
    }
}

/// Retry configuration.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum retry attempts.
    pub max_attempts: u32,
    /// Backoff strategy.
    pub backoff: BackoffStrategy,
    /// Enable jitter.
    pub jitter: bool,
    /// Maximum backoff duration.
    pub max_backoff: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff: BackoffStrategy::Exponential {
                initial: Duration::from_millis(100),
                max: Duration::from_secs(30),
                multiplier: 2.0,
            },
            jitter: true,
            max_backoff: Duration::from_secs(30),
        }
    }
}

// ============================================================================
// Multi-GPU Configuration
// ============================================================================

/// Multi-GPU coordination settings.
#[derive(Debug, Clone)]
pub struct MultiGpuConfig {
    /// Enable multi-GPU support.
    pub enabled: bool,
    /// Load balancing strategy.
    pub load_balancing: LoadBalancingStrategy,
    /// Enable peer-to-peer transfers.
    pub p2p_enabled: bool,
    /// Auto-select devices.
    pub auto_select_device: bool,
    /// Maximum kernels per device.
    pub max_kernels_per_device: usize,
    /// Preferred device indices.
    pub preferred_devices: Vec<usize>,
    /// Enable topology discovery.
    pub topology_discovery: bool,
    /// Enable cross-GPU K2K routing.
    pub cross_gpu_k2k: bool,
}

impl Default for MultiGpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            load_balancing: LoadBalancingStrategy::LeastLoaded,
            p2p_enabled: true,
            auto_select_device: true,
            max_kernels_per_device: 32,
            preferred_devices: Vec::new(),
            topology_discovery: true,
            cross_gpu_k2k: true,
        }
    }
}

impl MultiGpuConfig {
    /// Validate multi-GPU configuration.
    pub fn validate(&self) -> Result<()> {
        if self.max_kernels_per_device == 0 {
            return Err(RingKernelError::InvalidConfig(
                "max_kernels_per_device cannot be 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Migration Configuration
// ============================================================================

/// Kernel migration settings.
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    /// Enable live migration.
    pub enabled: bool,
    /// Checkpoint storage type.
    pub storage: CheckpointStorageType,
    /// Checkpoint directory (for file storage).
    pub checkpoint_dir: PathBuf,
    /// Maximum checkpoint size.
    pub max_checkpoint_size: usize,
    /// Enable compression.
    pub compression_enabled: bool,
    /// Compression level (1-9).
    pub compression_level: u32,
    /// Migration timeout.
    pub migration_timeout: Duration,
    /// Enable incremental checkpoints.
    pub incremental_enabled: bool,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            storage: CheckpointStorageType::Memory,
            checkpoint_dir: PathBuf::from("/tmp/ringkernel/checkpoints"),
            max_checkpoint_size: 1024 * 1024 * 1024, // 1 GB
            compression_enabled: false,
            compression_level: 3,
            migration_timeout: Duration::from_secs(60),
            incremental_enabled: false,
        }
    }
}

impl MigrationConfig {
    /// Validate migration configuration.
    pub fn validate(&self) -> Result<()> {
        if self.compression_level == 0 || self.compression_level > 9 {
            return Err(RingKernelError::InvalidConfig(format!(
                "compression_level must be between 1 and 9, got {}",
                self.compression_level
            )));
        }
        if self.max_checkpoint_size == 0 {
            return Err(RingKernelError::InvalidConfig(
                "max_checkpoint_size cannot be 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Checkpoint storage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CheckpointStorageType {
    /// In-memory storage (fast, non-persistent).
    #[default]
    Memory,
    /// File-based storage (persistent).
    File,
    /// Cloud storage (S3, GCS).
    Cloud,
}

impl CheckpointStorageType {
    /// Get the storage type as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            CheckpointStorageType::Memory => "memory",
            CheckpointStorageType::File => "file",
            CheckpointStorageType::Cloud => "cloud",
        }
    }
}

// ============================================================================
// Configuration Builder
// ============================================================================

/// Fluent builder for RingKernelConfig.
#[derive(Debug, Clone, Default)]
pub struct ConfigBuilder {
    config: RingKernelConfig,
}

impl ConfigBuilder {
    /// Create a new configuration builder.
    pub fn new() -> Self {
        Self {
            config: RingKernelConfig::default(),
        }
    }

    /// Configure general settings.
    pub fn with_general<F>(mut self, f: F) -> Self
    where
        F: FnOnce(GeneralConfigBuilder) -> GeneralConfigBuilder,
    {
        let builder = f(GeneralConfigBuilder::new());
        self.config.general = builder.build();
        self
    }

    /// Configure observability settings.
    pub fn with_observability<F>(mut self, f: F) -> Self
    where
        F: FnOnce(ObservabilityConfigBuilder) -> ObservabilityConfigBuilder,
    {
        let builder = f(ObservabilityConfigBuilder::new());
        self.config.observability = builder.build();
        self
    }

    /// Configure health settings.
    pub fn with_health<F>(mut self, f: F) -> Self
    where
        F: FnOnce(HealthConfigBuilder) -> HealthConfigBuilder,
    {
        let builder = f(HealthConfigBuilder::new());
        self.config.health = builder.build();
        self
    }

    /// Configure multi-GPU settings.
    pub fn with_multi_gpu<F>(mut self, f: F) -> Self
    where
        F: FnOnce(MultiGpuConfigBuilder) -> MultiGpuConfigBuilder,
    {
        let builder = f(MultiGpuConfigBuilder::new());
        self.config.multi_gpu = builder.build();
        self
    }

    /// Configure migration settings.
    pub fn with_migration<F>(mut self, f: F) -> Self
    where
        F: FnOnce(MigrationConfigBuilder) -> MigrationConfigBuilder,
    {
        let builder = f(MigrationConfigBuilder::new());
        self.config.migration = builder.build();
        self
    }

    /// Add a custom setting.
    pub fn custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.custom.insert(key.into(), value.into());
        self
    }

    /// Build and validate the configuration.
    pub fn build(self) -> Result<RingKernelConfig> {
        self.config.validate()?;
        Ok(self.config)
    }

    /// Build without validation.
    pub fn build_unchecked(self) -> RingKernelConfig {
        self.config
    }
}

// ============================================================================
// Sub-Builders
// ============================================================================

/// Builder for GeneralConfig.
#[derive(Debug, Clone)]
pub struct GeneralConfigBuilder {
    config: GeneralConfig,
}

impl GeneralConfigBuilder {
    /// Create a new general config builder.
    pub fn new() -> Self {
        Self {
            config: GeneralConfig::default(),
        }
    }

    /// Set the backend.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.config.backend = backend;
        self
    }

    /// Set the application name.
    pub fn app_name(mut self, name: impl Into<String>) -> Self {
        self.config.app_name = name.into();
        self
    }

    /// Set the application version.
    pub fn app_version(mut self, version: impl Into<String>) -> Self {
        self.config.app_version = version.into();
        self
    }

    /// Set the environment.
    pub fn environment(mut self, env: Environment) -> Self {
        self.config.environment = env;
        self
    }

    /// Set the log level.
    pub fn log_level(mut self, level: LogLevel) -> Self {
        self.config.log_level = level;
        self
    }

    /// Set the data directory.
    pub fn data_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.data_dir = Some(path.into());
        self
    }

    /// Build the configuration.
    pub fn build(self) -> GeneralConfig {
        self.config
    }
}

impl Default for GeneralConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for ObservabilityConfig.
#[derive(Debug, Clone)]
pub struct ObservabilityConfigBuilder {
    config: ObservabilityConfig,
}

impl ObservabilityConfigBuilder {
    /// Create a new observability config builder.
    pub fn new() -> Self {
        Self {
            config: ObservabilityConfig::default(),
        }
    }

    /// Enable or disable tracing.
    pub fn enable_tracing(mut self, enabled: bool) -> Self {
        self.config.tracing_enabled = enabled;
        self
    }

    /// Enable or disable metrics.
    pub fn enable_metrics(mut self, enabled: bool) -> Self {
        self.config.metrics_enabled = enabled;
        self
    }

    /// Set the metrics port.
    pub fn metrics_port(mut self, port: u16) -> Self {
        self.config.metrics_port = port;
        self
    }

    /// Set the metrics path.
    pub fn metrics_path(mut self, path: impl Into<String>) -> Self {
        self.config.metrics_path = path.into();
        self
    }

    /// Set the trace sample rate.
    pub fn trace_sample_rate(mut self, rate: f64) -> Self {
        self.config.trace_sample_rate = rate;
        self
    }

    /// Enable Grafana dashboard generation.
    pub fn enable_grafana(mut self, enabled: bool) -> Self {
        self.config.grafana_enabled = enabled;
        self
    }

    /// Set the OTLP endpoint.
    pub fn otlp_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.config.otlp_endpoint = Some(endpoint.into());
        self
    }

    /// Add a metric label.
    pub fn metric_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.metric_labels.insert(key.into(), value.into());
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ObservabilityConfig {
        self.config
    }
}

impl Default for ObservabilityConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for HealthConfig.
#[derive(Debug, Clone)]
pub struct HealthConfigBuilder {
    config: HealthConfig,
}

impl HealthConfigBuilder {
    /// Create a new health config builder.
    pub fn new() -> Self {
        Self {
            config: HealthConfig::default(),
        }
    }

    /// Enable or disable health checks.
    pub fn enable_health_checks(mut self, enabled: bool) -> Self {
        self.config.health_checks_enabled = enabled;
        self
    }

    /// Set the check interval.
    pub fn check_interval(mut self, interval: Duration) -> Self {
        self.config.check_interval = interval;
        self
    }

    /// Set the heartbeat timeout.
    pub fn heartbeat_timeout(mut self, timeout: Duration) -> Self {
        self.config.heartbeat_timeout = timeout;
        self
    }

    /// Set circuit breaker failure threshold.
    pub fn circuit_breaker_threshold(mut self, threshold: u32) -> Self {
        self.config.circuit_breaker.failure_threshold = threshold;
        self
    }

    /// Set circuit breaker recovery timeout.
    pub fn circuit_breaker_recovery_timeout(mut self, timeout: Duration) -> Self {
        self.config.circuit_breaker.recovery_timeout = timeout;
        self
    }

    /// Set circuit breaker half-open max requests.
    pub fn circuit_breaker_half_open_max_requests(mut self, requests: u32) -> Self {
        self.config.circuit_breaker.half_open_max_requests = requests;
        self
    }

    /// Configure retry policy.
    pub fn retry_max_attempts(mut self, attempts: u32) -> Self {
        self.config.retry.max_attempts = attempts;
        self
    }

    /// Enable or disable retry jitter.
    pub fn retry_jitter(mut self, enabled: bool) -> Self {
        self.config.retry.jitter = enabled;
        self
    }

    /// Set load shedding policy.
    pub fn load_shedding(mut self, policy: LoadSheddingPolicy) -> Self {
        self.config.load_shedding = policy;
        self
    }

    /// Enable or disable kernel watchdog.
    pub fn enable_watchdog(mut self, enabled: bool) -> Self {
        self.config.watchdog_enabled = enabled;
        self
    }

    /// Set watchdog failure threshold.
    pub fn watchdog_failure_threshold(mut self, threshold: u32) -> Self {
        self.config.watchdog_failure_threshold = threshold;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> HealthConfig {
        self.config
    }
}

impl Default for HealthConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for MultiGpuConfig.
#[derive(Debug, Clone)]
pub struct MultiGpuConfigBuilder {
    config: MultiGpuConfig,
}

impl MultiGpuConfigBuilder {
    /// Create a new multi-GPU config builder.
    pub fn new() -> Self {
        Self {
            config: MultiGpuConfig::default(),
        }
    }

    /// Enable or disable multi-GPU support.
    pub fn enable(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set the load balancing strategy.
    pub fn load_balancing(mut self, strategy: LoadBalancingStrategy) -> Self {
        self.config.load_balancing = strategy;
        self
    }

    /// Enable or disable P2P transfers.
    pub fn enable_p2p(mut self, enabled: bool) -> Self {
        self.config.p2p_enabled = enabled;
        self
    }

    /// Enable or disable auto device selection.
    pub fn auto_select_device(mut self, enabled: bool) -> Self {
        self.config.auto_select_device = enabled;
        self
    }

    /// Set maximum kernels per device.
    pub fn max_kernels_per_device(mut self, max: usize) -> Self {
        self.config.max_kernels_per_device = max;
        self
    }

    /// Set preferred devices.
    pub fn preferred_devices(mut self, devices: Vec<usize>) -> Self {
        self.config.preferred_devices = devices;
        self
    }

    /// Enable or disable topology discovery.
    pub fn topology_discovery(mut self, enabled: bool) -> Self {
        self.config.topology_discovery = enabled;
        self
    }

    /// Enable or disable cross-GPU K2K routing.
    pub fn cross_gpu_k2k(mut self, enabled: bool) -> Self {
        self.config.cross_gpu_k2k = enabled;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> MultiGpuConfig {
        self.config
    }
}

impl Default for MultiGpuConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for MigrationConfig.
#[derive(Debug, Clone)]
pub struct MigrationConfigBuilder {
    config: MigrationConfig,
}

impl MigrationConfigBuilder {
    /// Create a new migration config builder.
    pub fn new() -> Self {
        Self {
            config: MigrationConfig::default(),
        }
    }

    /// Enable or disable migration.
    pub fn enable(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set the storage type.
    pub fn storage(mut self, storage: CheckpointStorageType) -> Self {
        self.config.storage = storage;
        self
    }

    /// Set the checkpoint directory.
    pub fn checkpoint_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.checkpoint_dir = path.into();
        self
    }

    /// Set maximum checkpoint size.
    pub fn max_checkpoint_size(mut self, size: usize) -> Self {
        self.config.max_checkpoint_size = size;
        self
    }

    /// Enable or disable compression.
    pub fn enable_compression(mut self, enabled: bool) -> Self {
        self.config.compression_enabled = enabled;
        self
    }

    /// Set compression level.
    pub fn compression_level(mut self, level: u32) -> Self {
        self.config.compression_level = level;
        self
    }

    /// Set migration timeout.
    pub fn migration_timeout(mut self, timeout: Duration) -> Self {
        self.config.migration_timeout = timeout;
        self
    }

    /// Enable or disable incremental checkpoints.
    pub fn enable_incremental(mut self, enabled: bool) -> Self {
        self.config.incremental_enabled = enabled;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> MigrationConfig {
        self.config
    }
}

impl Default for MigrationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Configuration Presets
// ============================================================================

impl RingKernelConfig {
    /// Create a minimal configuration for development.
    pub fn development() -> Self {
        ConfigBuilder::new()
            .with_general(|g| g.environment(Environment::Development).log_level(LogLevel::Debug))
            .with_observability(|o| o.trace_sample_rate(1.0))
            .with_health(|h| h.enable_health_checks(true))
            .build_unchecked()
    }

    /// Create a production-ready configuration.
    pub fn production() -> Self {
        ConfigBuilder::new()
            .with_general(|g| g.environment(Environment::Production).log_level(LogLevel::Info))
            .with_observability(|o| {
                o.enable_tracing(true)
                    .enable_metrics(true)
                    .trace_sample_rate(0.1) // 10% sampling in production
                    .enable_grafana(true)
            })
            .with_health(|h| {
                h.enable_health_checks(true)
                    .check_interval(Duration::from_secs(5))
                    .heartbeat_timeout(Duration::from_secs(15))
                    .circuit_breaker_threshold(5)
                    .enable_watchdog(true)
            })
            .with_multi_gpu(|g| {
                g.enable(true)
                    .load_balancing(LoadBalancingStrategy::LeastLoaded)
                    .enable_p2p(true)
                    .topology_discovery(true)
            })
            .with_migration(|m| {
                m.enable(true)
                    .storage(CheckpointStorageType::File)
                    .enable_compression(true)
                    .compression_level(3)
            })
            .build_unchecked()
    }

    /// Create a high-performance configuration.
    pub fn high_performance() -> Self {
        ConfigBuilder::new()
            .with_general(|g| g.environment(Environment::Production).log_level(LogLevel::Warn))
            .with_observability(|o| {
                o.enable_tracing(false) // Disable tracing for max performance
                    .enable_metrics(true)
                    .trace_sample_rate(0.0)
            })
            .with_health(|h| {
                h.enable_health_checks(true)
                    .check_interval(Duration::from_secs(30)) // Less frequent checks
                    .watchdog_failure_threshold(5)
            })
            .with_multi_gpu(|g| {
                g.enable(true)
                    .load_balancing(LoadBalancingStrategy::LeastLoaded)
                    .enable_p2p(true)
                    .max_kernels_per_device(64)
                    .cross_gpu_k2k(true)
            })
            .with_migration(|m| {
                m.enable(true)
                    .storage(CheckpointStorageType::Memory)
                    .enable_compression(false) // Skip compression for speed
            })
            .build_unchecked()
    }
}

// ============================================================================
// Configuration File Support
// ============================================================================

/// File format for configuration loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    /// TOML format.
    Toml,
    /// YAML format.
    Yaml,
}

impl ConfigFormat {
    /// Detect format from file extension.
    pub fn from_extension(path: &std::path::Path) -> Option<Self> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
            .and_then(|ext| match ext.as_str() {
                "toml" => Some(ConfigFormat::Toml),
                "yaml" | "yml" => Some(ConfigFormat::Yaml),
                _ => None,
            })
    }
}

#[cfg(feature = "config-file")]
mod file_config {
    use super::*;
    use serde::{Deserialize, Serialize};

    /// File-format configuration (serialization-friendly).
    ///
    /// This struct uses primitive types that are easy to serialize/deserialize.
    /// It can be converted to/from `RingKernelConfig`.
    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    #[serde(default)]
    pub struct FileConfig {
        /// General settings.
        #[serde(default)]
        pub general: FileGeneralConfig,
        /// Observability settings.
        #[serde(default)]
        pub observability: FileObservabilityConfig,
        /// Health monitoring settings.
        #[serde(default)]
        pub health: FileHealthConfig,
        /// Multi-GPU settings.
        #[serde(default)]
        pub multi_gpu: FileMultiGpuConfig,
        /// Migration settings.
        #[serde(default)]
        pub migration: FileMigrationConfig,
        /// Custom settings.
        #[serde(default)]
        pub custom: HashMap<String, String>,
    }

    /// File-format general configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(default)]
    pub struct FileGeneralConfig {
        /// Backend: "auto", "cpu", "cuda", "wgpu", "metal".
        pub backend: String,
        /// Application name.
        pub app_name: String,
        /// Application version.
        pub app_version: String,
        /// Environment: "development", "staging", "production".
        pub environment: String,
        /// Log level: "trace", "debug", "info", "warn", "error".
        pub log_level: String,
        /// Data directory path.
        pub data_dir: Option<String>,
    }

    impl Default for FileGeneralConfig {
        fn default() -> Self {
            Self {
                backend: "auto".to_string(),
                app_name: "ringkernel".to_string(),
                app_version: env!("CARGO_PKG_VERSION").to_string(),
                environment: "development".to_string(),
                log_level: "info".to_string(),
                data_dir: None,
            }
        }
    }

    /// File-format observability configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(default)]
    pub struct FileObservabilityConfig {
        /// Enable tracing.
        pub tracing_enabled: bool,
        /// Enable metrics.
        pub metrics_enabled: bool,
        /// Metrics port.
        pub metrics_port: u16,
        /// Metrics path.
        pub metrics_path: String,
        /// Trace sample rate (0.0 to 1.0).
        pub trace_sample_rate: f64,
        /// Enable Grafana dashboard generation.
        pub grafana_enabled: bool,
        /// OTLP endpoint.
        pub otlp_endpoint: Option<String>,
        /// Custom metric labels.
        #[serde(default)]
        pub metric_labels: HashMap<String, String>,
    }

    impl Default for FileObservabilityConfig {
        fn default() -> Self {
            Self {
                tracing_enabled: true,
                metrics_enabled: true,
                metrics_port: 9090,
                metrics_path: "/metrics".to_string(),
                trace_sample_rate: 1.0,
                grafana_enabled: false,
                otlp_endpoint: None,
                metric_labels: HashMap::new(),
            }
        }
    }

    /// File-format health configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(default)]
    pub struct FileHealthConfig {
        /// Enable health checks.
        pub health_checks_enabled: bool,
        /// Health check interval in milliseconds.
        pub check_interval_ms: u64,
        /// Heartbeat timeout in milliseconds.
        pub heartbeat_timeout_ms: u64,
        /// Circuit breaker failure threshold.
        pub circuit_breaker_failure_threshold: u32,
        /// Circuit breaker recovery timeout in milliseconds.
        pub circuit_breaker_recovery_timeout_ms: u64,
        /// Circuit breaker half-open max requests.
        pub circuit_breaker_half_open_max_requests: u32,
        /// Retry max attempts.
        pub retry_max_attempts: u32,
        /// Enable retry jitter.
        pub retry_jitter: bool,
        /// Max backoff in milliseconds.
        pub retry_max_backoff_ms: u64,
        /// Enable kernel watchdog.
        pub watchdog_enabled: bool,
        /// Watchdog failure threshold.
        pub watchdog_failure_threshold: u32,
    }

    impl Default for FileHealthConfig {
        fn default() -> Self {
            Self {
                health_checks_enabled: true,
                check_interval_ms: 10_000,
                heartbeat_timeout_ms: 30_000,
                circuit_breaker_failure_threshold: 5,
                circuit_breaker_recovery_timeout_ms: 30_000,
                circuit_breaker_half_open_max_requests: 3,
                retry_max_attempts: 3,
                retry_jitter: true,
                retry_max_backoff_ms: 30_000,
                watchdog_enabled: true,
                watchdog_failure_threshold: 3,
            }
        }
    }

    /// File-format multi-GPU configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(default)]
    pub struct FileMultiGpuConfig {
        /// Enable multi-GPU support.
        pub enabled: bool,
        /// Load balancing: "round_robin", "least_loaded", "random", "preferred".
        pub load_balancing: String,
        /// Enable P2P transfers.
        pub p2p_enabled: bool,
        /// Auto-select device.
        pub auto_select_device: bool,
        /// Maximum kernels per device.
        pub max_kernels_per_device: usize,
        /// Preferred device indices.
        #[serde(default)]
        pub preferred_devices: Vec<usize>,
        /// Enable topology discovery.
        pub topology_discovery: bool,
        /// Enable cross-GPU K2K routing.
        pub cross_gpu_k2k: bool,
    }

    impl Default for FileMultiGpuConfig {
        fn default() -> Self {
            Self {
                enabled: true,
                load_balancing: "least_loaded".to_string(),
                p2p_enabled: true,
                auto_select_device: true,
                max_kernels_per_device: 32,
                preferred_devices: Vec::new(),
                topology_discovery: true,
                cross_gpu_k2k: true,
            }
        }
    }

    /// File-format migration configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(default)]
    pub struct FileMigrationConfig {
        /// Enable migration.
        pub enabled: bool,
        /// Storage type: "memory", "file", "cloud".
        pub storage: String,
        /// Checkpoint directory.
        pub checkpoint_dir: String,
        /// Maximum checkpoint size in bytes.
        pub max_checkpoint_size: usize,
        /// Enable compression.
        pub compression_enabled: bool,
        /// Compression level (1-9).
        pub compression_level: u32,
        /// Migration timeout in milliseconds.
        pub migration_timeout_ms: u64,
        /// Enable incremental checkpoints.
        pub incremental_enabled: bool,
    }

    impl Default for FileMigrationConfig {
        fn default() -> Self {
            Self {
                enabled: true,
                storage: "memory".to_string(),
                checkpoint_dir: "/tmp/ringkernel/checkpoints".to_string(),
                max_checkpoint_size: 1024 * 1024 * 1024,
                compression_enabled: false,
                compression_level: 3,
                migration_timeout_ms: 60_000,
                incremental_enabled: false,
            }
        }
    }

    // ========================================================================
    // Conversion Implementations
    // ========================================================================

    impl From<FileConfig> for RingKernelConfig {
        fn from(file: FileConfig) -> Self {
            RingKernelConfig {
                general: file.general.into(),
                observability: file.observability.into(),
                health: file.health.into(),
                multi_gpu: file.multi_gpu.into(),
                migration: file.migration.into(),
                custom: file.custom,
            }
        }
    }

    impl From<&RingKernelConfig> for FileConfig {
        fn from(config: &RingKernelConfig) -> Self {
            FileConfig {
                general: (&config.general).into(),
                observability: (&config.observability).into(),
                health: (&config.health).into(),
                multi_gpu: (&config.multi_gpu).into(),
                migration: (&config.migration).into(),
                custom: config.custom.clone(),
            }
        }
    }

    impl From<FileGeneralConfig> for GeneralConfig {
        fn from(file: FileGeneralConfig) -> Self {
            GeneralConfig {
                backend: match file.backend.to_lowercase().as_str() {
                    "cpu" => Backend::Cpu,
                    "cuda" => Backend::Cuda,
                    "wgpu" => Backend::Wgpu,
                    "metal" => Backend::Metal,
                    _ => Backend::Auto,
                },
                app_name: file.app_name,
                app_version: file.app_version,
                environment: match file.environment.to_lowercase().as_str() {
                    "staging" => Environment::Staging,
                    "production" | "prod" => Environment::Production,
                    _ => Environment::Development,
                },
                log_level: match file.log_level.to_lowercase().as_str() {
                    "trace" => LogLevel::Trace,
                    "debug" => LogLevel::Debug,
                    "warn" | "warning" => LogLevel::Warn,
                    "error" => LogLevel::Error,
                    _ => LogLevel::Info,
                },
                data_dir: file.data_dir.map(PathBuf::from),
            }
        }
    }

    impl From<&GeneralConfig> for FileGeneralConfig {
        fn from(config: &GeneralConfig) -> Self {
            FileGeneralConfig {
                backend: match config.backend {
                    Backend::Auto => "auto".to_string(),
                    Backend::Cpu => "cpu".to_string(),
                    Backend::Cuda => "cuda".to_string(),
                    Backend::Wgpu => "wgpu".to_string(),
                    Backend::Metal => "metal".to_string(),
                },
                app_name: config.app_name.clone(),
                app_version: config.app_version.clone(),
                environment: config.environment.as_str().to_string(),
                log_level: config.log_level.as_str().to_string(),
                data_dir: config.data_dir.as_ref().map(|p| p.display().to_string()),
            }
        }
    }

    impl From<FileObservabilityConfig> for ObservabilityConfig {
        fn from(file: FileObservabilityConfig) -> Self {
            ObservabilityConfig {
                tracing_enabled: file.tracing_enabled,
                metrics_enabled: file.metrics_enabled,
                metrics_port: file.metrics_port,
                metrics_path: file.metrics_path,
                trace_sample_rate: file.trace_sample_rate,
                grafana_enabled: file.grafana_enabled,
                otlp_endpoint: file.otlp_endpoint,
                metric_labels: file.metric_labels,
            }
        }
    }

    impl From<&ObservabilityConfig> for FileObservabilityConfig {
        fn from(config: &ObservabilityConfig) -> Self {
            FileObservabilityConfig {
                tracing_enabled: config.tracing_enabled,
                metrics_enabled: config.metrics_enabled,
                metrics_port: config.metrics_port,
                metrics_path: config.metrics_path.clone(),
                trace_sample_rate: config.trace_sample_rate,
                grafana_enabled: config.grafana_enabled,
                otlp_endpoint: config.otlp_endpoint.clone(),
                metric_labels: config.metric_labels.clone(),
            }
        }
    }

    impl From<FileHealthConfig> for HealthConfig {
        fn from(file: FileHealthConfig) -> Self {
            HealthConfig {
                health_checks_enabled: file.health_checks_enabled,
                check_interval: Duration::from_millis(file.check_interval_ms),
                heartbeat_timeout: Duration::from_millis(file.heartbeat_timeout_ms),
                circuit_breaker: CircuitBreakerConfig {
                    failure_threshold: file.circuit_breaker_failure_threshold,
                    success_threshold: 1, // Default: 1 success to close
                    recovery_timeout: Duration::from_millis(file.circuit_breaker_recovery_timeout_ms),
                    window_duration: Duration::from_secs(60), // Default: 60 second window
                    half_open_max_requests: file.circuit_breaker_half_open_max_requests,
                },
                retry: RetryConfig {
                    max_attempts: file.retry_max_attempts,
                    backoff: BackoffStrategy::Exponential {
                        initial: Duration::from_millis(100),
                        max: Duration::from_millis(file.retry_max_backoff_ms),
                        multiplier: 2.0,
                    },
                    jitter: file.retry_jitter,
                    max_backoff: Duration::from_millis(file.retry_max_backoff_ms),
                },
                load_shedding: LoadSheddingPolicy::default(),
                watchdog_enabled: file.watchdog_enabled,
                watchdog_failure_threshold: file.watchdog_failure_threshold,
            }
        }
    }

    impl From<&HealthConfig> for FileHealthConfig {
        fn from(config: &HealthConfig) -> Self {
            FileHealthConfig {
                health_checks_enabled: config.health_checks_enabled,
                check_interval_ms: config.check_interval.as_millis() as u64,
                heartbeat_timeout_ms: config.heartbeat_timeout.as_millis() as u64,
                circuit_breaker_failure_threshold: config.circuit_breaker.failure_threshold,
                circuit_breaker_recovery_timeout_ms: config.circuit_breaker.recovery_timeout.as_millis() as u64,
                circuit_breaker_half_open_max_requests: config.circuit_breaker.half_open_max_requests,
                retry_max_attempts: config.retry.max_attempts,
                retry_jitter: config.retry.jitter,
                retry_max_backoff_ms: config.retry.max_backoff.as_millis() as u64,
                watchdog_enabled: config.watchdog_enabled,
                watchdog_failure_threshold: config.watchdog_failure_threshold,
            }
        }
    }

    impl From<FileMultiGpuConfig> for MultiGpuConfig {
        fn from(file: FileMultiGpuConfig) -> Self {
            MultiGpuConfig {
                enabled: file.enabled,
                load_balancing: match file.load_balancing.to_lowercase().as_str() {
                    "round_robin" | "roundrobin" => LoadBalancingStrategy::RoundRobin,
                    "first_available" | "firstavailable" => LoadBalancingStrategy::FirstAvailable,
                    "memory_based" | "memorybased" => LoadBalancingStrategy::MemoryBased,
                    "compute_capability" | "computecapability" => LoadBalancingStrategy::ComputeCapability,
                    "custom" => LoadBalancingStrategy::Custom,
                    _ => LoadBalancingStrategy::LeastLoaded,
                },
                p2p_enabled: file.p2p_enabled,
                auto_select_device: file.auto_select_device,
                max_kernels_per_device: file.max_kernels_per_device,
                preferred_devices: file.preferred_devices,
                topology_discovery: file.topology_discovery,
                cross_gpu_k2k: file.cross_gpu_k2k,
            }
        }
    }

    impl From<&MultiGpuConfig> for FileMultiGpuConfig {
        fn from(config: &MultiGpuConfig) -> Self {
            FileMultiGpuConfig {
                enabled: config.enabled,
                load_balancing: match config.load_balancing {
                    LoadBalancingStrategy::FirstAvailable => "first_available".to_string(),
                    LoadBalancingStrategy::LeastLoaded => "least_loaded".to_string(),
                    LoadBalancingStrategy::RoundRobin => "round_robin".to_string(),
                    LoadBalancingStrategy::MemoryBased => "memory_based".to_string(),
                    LoadBalancingStrategy::ComputeCapability => "compute_capability".to_string(),
                    LoadBalancingStrategy::Custom => "custom".to_string(),
                },
                p2p_enabled: config.p2p_enabled,
                auto_select_device: config.auto_select_device,
                max_kernels_per_device: config.max_kernels_per_device,
                preferred_devices: config.preferred_devices.clone(),
                topology_discovery: config.topology_discovery,
                cross_gpu_k2k: config.cross_gpu_k2k,
            }
        }
    }

    impl From<FileMigrationConfig> for MigrationConfig {
        fn from(file: FileMigrationConfig) -> Self {
            MigrationConfig {
                enabled: file.enabled,
                storage: match file.storage.to_lowercase().as_str() {
                    "file" => CheckpointStorageType::File,
                    "cloud" => CheckpointStorageType::Cloud,
                    _ => CheckpointStorageType::Memory,
                },
                checkpoint_dir: PathBuf::from(file.checkpoint_dir),
                max_checkpoint_size: file.max_checkpoint_size,
                compression_enabled: file.compression_enabled,
                compression_level: file.compression_level,
                migration_timeout: Duration::from_millis(file.migration_timeout_ms),
                incremental_enabled: file.incremental_enabled,
            }
        }
    }

    impl From<&MigrationConfig> for FileMigrationConfig {
        fn from(config: &MigrationConfig) -> Self {
            FileMigrationConfig {
                enabled: config.enabled,
                storage: config.storage.as_str().to_string(),
                checkpoint_dir: config.checkpoint_dir.display().to_string(),
                max_checkpoint_size: config.max_checkpoint_size,
                compression_enabled: config.compression_enabled,
                compression_level: config.compression_level,
                migration_timeout_ms: config.migration_timeout.as_millis() as u64,
                incremental_enabled: config.incremental_enabled,
            }
        }
    }
}

#[cfg(feature = "config-file")]
pub use file_config::*;

#[cfg(feature = "config-file")]
impl RingKernelConfig {
    /// Load configuration from a TOML file.
    pub fn from_toml_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            RingKernelError::InvalidConfig(format!("Failed to read config file: {}", e))
        })?;
        Self::from_toml_str(&content)
    }

    /// Load configuration from a TOML string.
    pub fn from_toml_str(content: &str) -> Result<Self> {
        let file_config: FileConfig = toml::from_str(content).map_err(|e| {
            RingKernelError::InvalidConfig(format!("Failed to parse TOML config: {}", e))
        })?;
        let config: RingKernelConfig = file_config.into();
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from a YAML file.
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            RingKernelError::InvalidConfig(format!("Failed to read config file: {}", e))
        })?;
        Self::from_yaml_str(&content)
    }

    /// Load configuration from a YAML string.
    pub fn from_yaml_str(content: &str) -> Result<Self> {
        let file_config: FileConfig = serde_yaml::from_str(content).map_err(|e| {
            RingKernelError::InvalidConfig(format!("Failed to parse YAML config: {}", e))
        })?;
        let config: RingKernelConfig = file_config.into();
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from a file, auto-detecting format from extension.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let format = ConfigFormat::from_extension(path).ok_or_else(|| {
            RingKernelError::InvalidConfig(format!(
                "Unknown config file extension: {}",
                path.display()
            ))
        })?;

        match format {
            ConfigFormat::Toml => Self::from_toml_file(path),
            ConfigFormat::Yaml => Self::from_yaml_file(path),
        }
    }

    /// Write configuration to a TOML string.
    pub fn to_toml_str(&self) -> Result<String> {
        let file_config: FileConfig = self.into();
        toml::to_string_pretty(&file_config).map_err(|e| {
            RingKernelError::InvalidConfig(format!("Failed to serialize to TOML: {}", e))
        })
    }

    /// Write configuration to a YAML string.
    pub fn to_yaml_str(&self) -> Result<String> {
        let file_config: FileConfig = self.into();
        serde_yaml::to_string(&file_config).map_err(|e| {
            RingKernelError::InvalidConfig(format!("Failed to serialize to YAML: {}", e))
        })
    }

    /// Write configuration to a file.
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let format = ConfigFormat::from_extension(path).ok_or_else(|| {
            RingKernelError::InvalidConfig(format!(
                "Unknown config file extension: {}",
                path.display()
            ))
        })?;

        let content = match format {
            ConfigFormat::Toml => self.to_toml_str()?,
            ConfigFormat::Yaml => self.to_yaml_str()?,
        };

        std::fs::write(path, content).map_err(|e| {
            RingKernelError::InvalidConfig(format!("Failed to write config file: {}", e))
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RingKernelConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder_basic() {
        let config = ConfigBuilder::new().build().unwrap();

        assert_eq!(config.general.environment, Environment::Development);
        assert!(config.observability.tracing_enabled);
        assert!(config.health.health_checks_enabled);
        assert!(config.multi_gpu.enabled);
    }

    #[test]
    fn test_builder_with_general() {
        let config = ConfigBuilder::new()
            .with_general(|g| {
                g.app_name("test_app")
                    .environment(Environment::Production)
                    .log_level(LogLevel::Warn)
            })
            .build()
            .unwrap();

        assert_eq!(config.general.app_name, "test_app");
        assert_eq!(config.general.environment, Environment::Production);
        assert_eq!(config.general.log_level, LogLevel::Warn);
    }

    #[test]
    fn test_builder_with_observability() {
        let config = ConfigBuilder::new()
            .with_observability(|o| {
                o.enable_tracing(false)
                    .metrics_port(8080)
                    .trace_sample_rate(0.5)
            })
            .build()
            .unwrap();

        assert!(!config.observability.tracing_enabled);
        assert_eq!(config.observability.metrics_port, 8080);
        assert_eq!(config.observability.trace_sample_rate, 0.5);
    }

    #[test]
    fn test_builder_with_health() {
        let config = ConfigBuilder::new()
            .with_health(|h| {
                h.check_interval(Duration::from_secs(5))
                    .heartbeat_timeout(Duration::from_secs(15))
                    .circuit_breaker_threshold(10)
            })
            .build()
            .unwrap();

        assert_eq!(config.health.check_interval, Duration::from_secs(5));
        assert_eq!(config.health.heartbeat_timeout, Duration::from_secs(15));
        assert_eq!(config.health.circuit_breaker.failure_threshold, 10);
    }

    #[test]
    fn test_builder_with_multi_gpu() {
        let config = ConfigBuilder::new()
            .with_multi_gpu(|g| {
                g.load_balancing(LoadBalancingStrategy::RoundRobin)
                    .enable_p2p(false)
                    .max_kernels_per_device(64)
            })
            .build()
            .unwrap();

        assert_eq!(
            config.multi_gpu.load_balancing,
            LoadBalancingStrategy::RoundRobin
        );
        assert!(!config.multi_gpu.p2p_enabled);
        assert_eq!(config.multi_gpu.max_kernels_per_device, 64);
    }

    #[test]
    fn test_builder_with_migration() {
        let config = ConfigBuilder::new()
            .with_migration(|m| {
                m.storage(CheckpointStorageType::File)
                    .enable_compression(true)
                    .compression_level(5)
            })
            .build()
            .unwrap();

        assert_eq!(config.migration.storage, CheckpointStorageType::File);
        assert!(config.migration.compression_enabled);
        assert_eq!(config.migration.compression_level, 5);
    }

    #[test]
    fn test_validation_invalid_sample_rate() {
        let result = ConfigBuilder::new()
            .with_observability(|o| o.trace_sample_rate(1.5))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_validation_invalid_compression_level() {
        let result = ConfigBuilder::new()
            .with_migration(|m| m.compression_level(10))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_validation_invalid_check_interval() {
        let result = ConfigBuilder::new()
            .with_health(|h| h.check_interval(Duration::ZERO))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_custom_settings() {
        let config = ConfigBuilder::new()
            .custom("feature_flag", "enabled")
            .custom("custom_param", "42")
            .build()
            .unwrap();

        assert_eq!(config.get_custom("feature_flag"), Some("enabled"));
        assert_eq!(config.get_custom("custom_param"), Some("42"));
        assert_eq!(config.get_custom("nonexistent"), None);
    }

    #[test]
    fn test_environment() {
        assert!(!Environment::Development.is_production());
        assert!(!Environment::Staging.is_production());
        assert!(Environment::Production.is_production());

        assert_eq!(Environment::Development.as_str(), "development");
        assert_eq!(Environment::Staging.as_str(), "staging");
        assert_eq!(Environment::Production.as_str(), "production");
    }

    #[test]
    fn test_log_level() {
        assert_eq!(LogLevel::Trace.as_str(), "trace");
        assert_eq!(LogLevel::Debug.as_str(), "debug");
        assert_eq!(LogLevel::Info.as_str(), "info");
        assert_eq!(LogLevel::Warn.as_str(), "warn");
        assert_eq!(LogLevel::Error.as_str(), "error");
    }

    #[test]
    fn test_storage_type() {
        assert_eq!(CheckpointStorageType::Memory.as_str(), "memory");
        assert_eq!(CheckpointStorageType::File.as_str(), "file");
        assert_eq!(CheckpointStorageType::Cloud.as_str(), "cloud");
    }

    #[test]
    fn test_preset_development() {
        let config = RingKernelConfig::development();
        assert_eq!(config.general.environment, Environment::Development);
        assert_eq!(config.general.log_level, LogLevel::Debug);
    }

    #[test]
    fn test_preset_production() {
        let config = RingKernelConfig::production();
        assert_eq!(config.general.environment, Environment::Production);
        assert!(config.observability.grafana_enabled);
        assert!(config.migration.compression_enabled);
    }

    #[test]
    fn test_preset_high_performance() {
        let config = RingKernelConfig::high_performance();
        assert!(!config.observability.tracing_enabled);
        assert_eq!(config.observability.trace_sample_rate, 0.0);
        assert!(!config.migration.compression_enabled);
    }

    #[test]
    fn test_config_format_from_extension() {
        use std::path::Path;

        assert_eq!(
            ConfigFormat::from_extension(Path::new("config.toml")),
            Some(ConfigFormat::Toml)
        );
        assert_eq!(
            ConfigFormat::from_extension(Path::new("config.yaml")),
            Some(ConfigFormat::Yaml)
        );
        assert_eq!(
            ConfigFormat::from_extension(Path::new("config.yml")),
            Some(ConfigFormat::Yaml)
        );
        assert_eq!(
            ConfigFormat::from_extension(Path::new("config.TOML")),
            Some(ConfigFormat::Toml)
        );
        assert_eq!(
            ConfigFormat::from_extension(Path::new("config.json")),
            None
        );
        assert_eq!(ConfigFormat::from_extension(Path::new("config")), None);
    }
}

// ============================================================================
// Configuration File Tests (feature-gated)
// ============================================================================

#[cfg(all(test, feature = "config-file"))]
mod file_config_tests {
    use super::*;
    use std::time::Duration;

    const SAMPLE_TOML: &str = r#"
[general]
app_name = "test-app"
app_version = "2.0.0"
environment = "production"
log_level = "debug"
backend = "cuda"

[observability]
tracing_enabled = true
metrics_enabled = true
metrics_port = 8080
trace_sample_rate = 0.5

[health]
health_checks_enabled = true
check_interval_ms = 5000
heartbeat_timeout_ms = 15000
circuit_breaker_failure_threshold = 10
watchdog_enabled = true

[multi_gpu]
enabled = true
load_balancing = "round_robin"
p2p_enabled = false
max_kernels_per_device = 64

[migration]
enabled = true
storage = "file"
checkpoint_dir = "/data/checkpoints"
compression_enabled = true
compression_level = 5

[custom]
feature_x = "enabled"
max_retries = "5"
"#;

    const SAMPLE_YAML: &str = r#"
general:
  app_name: test-app
  app_version: "2.0.0"
  environment: production
  log_level: debug
  backend: cuda

observability:
  tracing_enabled: true
  metrics_enabled: true
  metrics_port: 8080
  trace_sample_rate: 0.5

health:
  health_checks_enabled: true
  check_interval_ms: 5000
  heartbeat_timeout_ms: 15000
  circuit_breaker_failure_threshold: 10
  watchdog_enabled: true

multi_gpu:
  enabled: true
  load_balancing: round_robin
  p2p_enabled: false
  max_kernels_per_device: 64

migration:
  enabled: true
  storage: file
  checkpoint_dir: /data/checkpoints
  compression_enabled: true
  compression_level: 5

custom:
  feature_x: enabled
  max_retries: "5"
"#;

    #[test]
    fn test_from_toml_str() {
        let config = RingKernelConfig::from_toml_str(SAMPLE_TOML).unwrap();

        assert_eq!(config.general.app_name, "test-app");
        assert_eq!(config.general.app_version, "2.0.0");
        assert_eq!(config.general.environment, Environment::Production);
        assert_eq!(config.general.log_level, LogLevel::Debug);
        assert_eq!(config.general.backend, Backend::Cuda);

        assert!(config.observability.tracing_enabled);
        assert_eq!(config.observability.metrics_port, 8080);
        assert_eq!(config.observability.trace_sample_rate, 0.5);

        assert_eq!(config.health.check_interval, Duration::from_millis(5000));
        assert_eq!(config.health.heartbeat_timeout, Duration::from_millis(15000));
        assert_eq!(config.health.circuit_breaker.failure_threshold, 10);

        assert_eq!(config.multi_gpu.load_balancing, LoadBalancingStrategy::RoundRobin);
        assert!(!config.multi_gpu.p2p_enabled);
        assert_eq!(config.multi_gpu.max_kernels_per_device, 64);

        assert_eq!(config.migration.storage, CheckpointStorageType::File);
        assert!(config.migration.compression_enabled);
        assert_eq!(config.migration.compression_level, 5);

        assert_eq!(config.get_custom("feature_x"), Some("enabled"));
        assert_eq!(config.get_custom("max_retries"), Some("5"));
    }

    #[test]
    fn test_from_yaml_str() {
        let config = RingKernelConfig::from_yaml_str(SAMPLE_YAML).unwrap();

        assert_eq!(config.general.app_name, "test-app");
        assert_eq!(config.general.app_version, "2.0.0");
        assert_eq!(config.general.environment, Environment::Production);
        assert_eq!(config.general.log_level, LogLevel::Debug);
        assert_eq!(config.general.backend, Backend::Cuda);

        assert!(config.observability.tracing_enabled);
        assert_eq!(config.observability.metrics_port, 8080);
        assert_eq!(config.observability.trace_sample_rate, 0.5);

        assert_eq!(config.health.check_interval, Duration::from_millis(5000));
        assert_eq!(config.health.heartbeat_timeout, Duration::from_millis(15000));
        assert_eq!(config.health.circuit_breaker.failure_threshold, 10);

        assert_eq!(config.multi_gpu.load_balancing, LoadBalancingStrategy::RoundRobin);
        assert!(!config.multi_gpu.p2p_enabled);
        assert_eq!(config.multi_gpu.max_kernels_per_device, 64);

        assert_eq!(config.migration.storage, CheckpointStorageType::File);
        assert!(config.migration.compression_enabled);
        assert_eq!(config.migration.compression_level, 5);

        assert_eq!(config.get_custom("feature_x"), Some("enabled"));
        assert_eq!(config.get_custom("max_retries"), Some("5"));
    }

    #[test]
    fn test_to_toml_str() {
        let config = RingKernelConfig::production();
        let toml_str = config.to_toml_str().unwrap();

        // Parse back and verify
        let parsed = RingKernelConfig::from_toml_str(&toml_str).unwrap();
        assert_eq!(parsed.general.environment, Environment::Production);
        assert!(parsed.observability.grafana_enabled);
    }

    #[test]
    fn test_to_yaml_str() {
        let config = RingKernelConfig::production();
        let yaml_str = config.to_yaml_str().unwrap();

        // Parse back and verify
        let parsed = RingKernelConfig::from_yaml_str(&yaml_str).unwrap();
        assert_eq!(parsed.general.environment, Environment::Production);
        assert!(parsed.observability.grafana_enabled);
    }

    #[test]
    fn test_roundtrip_toml() {
        let original = ConfigBuilder::new()
            .with_general(|g| {
                g.app_name("roundtrip-test")
                    .environment(Environment::Staging)
                    .log_level(LogLevel::Warn)
            })
            .with_observability(|o| o.metrics_port(9999).trace_sample_rate(0.25))
            .with_multi_gpu(|m| m.max_kernels_per_device(128))
            .custom("test_key", "test_value")
            .build()
            .unwrap();

        let toml_str = original.to_toml_str().unwrap();
        let parsed = RingKernelConfig::from_toml_str(&toml_str).unwrap();

        assert_eq!(parsed.general.app_name, "roundtrip-test");
        assert_eq!(parsed.general.environment, Environment::Staging);
        assert_eq!(parsed.general.log_level, LogLevel::Warn);
        assert_eq!(parsed.observability.metrics_port, 9999);
        assert_eq!(parsed.observability.trace_sample_rate, 0.25);
        assert_eq!(parsed.multi_gpu.max_kernels_per_device, 128);
        assert_eq!(parsed.get_custom("test_key"), Some("test_value"));
    }

    #[test]
    fn test_roundtrip_yaml() {
        let original = ConfigBuilder::new()
            .with_general(|g| {
                g.app_name("roundtrip-test")
                    .environment(Environment::Staging)
                    .log_level(LogLevel::Warn)
            })
            .with_observability(|o| o.metrics_port(9999).trace_sample_rate(0.25))
            .with_multi_gpu(|m| m.max_kernels_per_device(128))
            .custom("test_key", "test_value")
            .build()
            .unwrap();

        let yaml_str = original.to_yaml_str().unwrap();
        let parsed = RingKernelConfig::from_yaml_str(&yaml_str).unwrap();

        assert_eq!(parsed.general.app_name, "roundtrip-test");
        assert_eq!(parsed.general.environment, Environment::Staging);
        assert_eq!(parsed.general.log_level, LogLevel::Warn);
        assert_eq!(parsed.observability.metrics_port, 9999);
        assert_eq!(parsed.observability.trace_sample_rate, 0.25);
        assert_eq!(parsed.multi_gpu.max_kernels_per_device, 128);
        assert_eq!(parsed.get_custom("test_key"), Some("test_value"));
    }

    #[test]
    fn test_partial_config() {
        // Test that missing sections use defaults
        let minimal_toml = r#"
[general]
app_name = "minimal"
"#;
        let config = RingKernelConfig::from_toml_str(minimal_toml).unwrap();
        assert_eq!(config.general.app_name, "minimal");
        assert_eq!(config.general.environment, Environment::Development); // default
        assert!(config.observability.tracing_enabled); // default
        assert!(config.health.health_checks_enabled); // default
    }

    #[test]
    fn test_invalid_toml() {
        let invalid = "this is not valid toml { }";
        let result = RingKernelConfig::from_toml_str(invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_yaml() {
        let invalid = "{{invalid yaml}}";
        let result = RingKernelConfig::from_yaml_str(invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_on_load() {
        // Invalid: trace_sample_rate > 1.0
        let invalid_toml = r#"
[observability]
trace_sample_rate = 1.5
"#;
        let result = RingKernelConfig::from_toml_str(invalid_toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_file_config_defaults() {
        let file_config = FileConfig::default();
        let config: RingKernelConfig = file_config.into();

        assert_eq!(config.general.app_name, "ringkernel");
        assert_eq!(config.general.environment, Environment::Development);
        assert!(config.observability.tracing_enabled);
        assert!(config.health.health_checks_enabled);
        assert!(config.multi_gpu.enabled);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_environment_aliases() {
        // Test "prod" alias for production
        let toml = r#"
[general]
environment = "prod"
"#;
        let config = RingKernelConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.general.environment, Environment::Production);
    }

    #[test]
    fn test_load_balancing_aliases() {
        // Test "roundrobin" alias
        let toml = r#"
[multi_gpu]
load_balancing = "roundrobin"
"#;
        let config = RingKernelConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.multi_gpu.load_balancing, LoadBalancingStrategy::RoundRobin);
    }
}
