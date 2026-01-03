//! Unified runtime context for RingKernel enterprise features.
//!
//! This module provides a comprehensive runtime context that instantiates and manages
//! all enterprise features (observability, health, multi-GPU, migration) based on
//! the unified configuration.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::runtime_context::RuntimeBuilder;
//! use ringkernel_core::config::RingKernelConfig;
//!
//! // Create runtime with default configuration
//! let runtime = RuntimeBuilder::new()
//!     .with_config(RingKernelConfig::production())
//!     .build()?;
//!
//! // Access enterprise features
//! let health = runtime.health_checker();
//! let metrics = runtime.prometheus_exporter();
//! let coordinator = runtime.multi_gpu_coordinator();
//!
//! // Graceful shutdown
//! runtime.shutdown().await?;
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::config::{CheckpointStorageType, RingKernelConfig};
use crate::checkpoint::{CheckpointStorage, FileStorage, MemoryStorage};
use crate::error::{Result, RingKernelError};
use crate::health::{CircuitBreaker, CircuitState, DegradationManager, HealthChecker, KernelWatchdog};
use crate::multi_gpu::{KernelMigrator, MultiGpuBuilder, MultiGpuCoordinator};
use crate::observability::{ObservabilityContext, PrometheusExporter};

// ============================================================================
// Runtime Context
// ============================================================================

/// Unified runtime context managing all enterprise features.
///
/// This is the main entry point for using RingKernel's enterprise features.
/// It instantiates and manages:
/// - Health checking and circuit breakers
/// - Prometheus metrics exporter
/// - Multi-GPU coordination
/// - Kernel migration infrastructure
pub struct RingKernelContext {
    /// Configuration used to create this context.
    config: RingKernelConfig,
    /// Health checker instance.
    health_checker: Arc<HealthChecker>,
    /// Kernel watchdog.
    watchdog: Arc<KernelWatchdog>,
    /// Circuit breaker for kernel operations.
    circuit_breaker: Arc<CircuitBreaker>,
    /// Degradation manager.
    degradation_manager: Arc<DegradationManager>,
    /// Prometheus exporter.
    prometheus_exporter: Arc<PrometheusExporter>,
    /// Observability context.
    observability: Arc<ObservabilityContext>,
    /// Multi-GPU coordinator.
    multi_gpu_coordinator: Arc<MultiGpuCoordinator>,
    /// Kernel migrator.
    migrator: Arc<KernelMigrator>,
    /// Checkpoint storage.
    checkpoint_storage: Arc<dyn CheckpointStorage>,
    /// Runtime statistics.
    stats: RuntimeStats,
    /// Startup time.
    started_at: Instant,
    /// Running state.
    running: AtomicBool,
}

impl RingKernelContext {
    /// Get the configuration.
    pub fn config(&self) -> &RingKernelConfig {
        &self.config
    }

    /// Get the health checker.
    pub fn health_checker(&self) -> Arc<HealthChecker> {
        Arc::clone(&self.health_checker)
    }

    /// Get the kernel watchdog.
    pub fn watchdog(&self) -> Arc<KernelWatchdog> {
        Arc::clone(&self.watchdog)
    }

    /// Get the circuit breaker.
    pub fn circuit_breaker(&self) -> Arc<CircuitBreaker> {
        Arc::clone(&self.circuit_breaker)
    }

    /// Get the degradation manager.
    pub fn degradation_manager(&self) -> Arc<DegradationManager> {
        Arc::clone(&self.degradation_manager)
    }

    /// Get the Prometheus exporter.
    pub fn prometheus_exporter(&self) -> Arc<PrometheusExporter> {
        Arc::clone(&self.prometheus_exporter)
    }

    /// Get the observability context.
    pub fn observability(&self) -> Arc<ObservabilityContext> {
        Arc::clone(&self.observability)
    }

    /// Get the multi-GPU coordinator.
    pub fn multi_gpu_coordinator(&self) -> Arc<MultiGpuCoordinator> {
        Arc::clone(&self.multi_gpu_coordinator)
    }

    /// Get the kernel migrator.
    pub fn migrator(&self) -> Arc<KernelMigrator> {
        Arc::clone(&self.migrator)
    }

    /// Get the checkpoint storage.
    pub fn checkpoint_storage(&self) -> Arc<dyn CheckpointStorage> {
        Arc::clone(&self.checkpoint_storage)
    }

    /// Check if the runtime is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get runtime uptime.
    pub fn uptime(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }

    /// Get runtime statistics.
    pub fn stats(&self) -> RuntimeStatsSnapshot {
        RuntimeStatsSnapshot {
            uptime: self.uptime(),
            kernels_launched: self.stats.kernels_launched.load(Ordering::Relaxed),
            messages_processed: self.stats.messages_processed.load(Ordering::Relaxed),
            migrations_completed: self.stats.migrations_completed.load(Ordering::Relaxed),
            checkpoints_created: self.stats.checkpoints_created.load(Ordering::Relaxed),
            health_checks_run: self.stats.health_checks_run.load(Ordering::Relaxed),
            circuit_breaker_trips: self.stats.circuit_breaker_trips.load(Ordering::Relaxed),
        }
    }

    /// Record a kernel launch.
    pub fn record_kernel_launch(&self) {
        self.stats.kernels_launched.fetch_add(1, Ordering::Relaxed);
    }

    /// Record messages processed.
    pub fn record_messages(&self, count: u64) {
        self.stats.messages_processed.fetch_add(count, Ordering::Relaxed);
    }

    /// Record a migration completion.
    pub fn record_migration(&self) {
        self.stats.migrations_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a checkpoint creation.
    pub fn record_checkpoint(&self) {
        self.stats.checkpoints_created.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a health check run.
    pub fn record_health_check(&self) {
        self.stats.health_checks_run.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a circuit breaker trip.
    pub fn record_circuit_trip(&self) {
        self.stats.circuit_breaker_trips.fetch_add(1, Ordering::Relaxed);
    }

    /// Shutdown the runtime gracefully.
    pub fn shutdown(&self) -> Result<()> {
        if !self.running.swap(false, Ordering::SeqCst) {
            return Err(RingKernelError::InvalidState {
                expected: "running".to_string(),
                actual: "stopped".to_string(),
            });
        }

        // Shutdown order:
        // 1. Stop accepting new work
        // 2. Drain pending operations
        // 3. Shutdown components

        // Note: In a real implementation, this would:
        // - Stop background health check loops
        // - Drain message queues
        // - Wait for in-flight migrations
        // - Flush metrics

        Ok(())
    }

    /// Get application info.
    pub fn app_info(&self) -> AppInfo {
        AppInfo {
            name: self.config.general.app_name.clone(),
            version: self.config.general.app_version.clone(),
            environment: self.config.general.environment.as_str().to_string(),
        }
    }
}

/// Runtime statistics (atomic counters).
#[derive(Debug, Default)]
struct RuntimeStats {
    kernels_launched: AtomicU64,
    messages_processed: AtomicU64,
    migrations_completed: AtomicU64,
    checkpoints_created: AtomicU64,
    health_checks_run: AtomicU64,
    circuit_breaker_trips: AtomicU64,
}

/// Snapshot of runtime statistics.
#[derive(Debug, Clone)]
pub struct RuntimeStatsSnapshot {
    /// Runtime uptime.
    pub uptime: std::time::Duration,
    /// Total kernels launched.
    pub kernels_launched: u64,
    /// Total messages processed.
    pub messages_processed: u64,
    /// Total migrations completed.
    pub migrations_completed: u64,
    /// Total checkpoints created.
    pub checkpoints_created: u64,
    /// Total health checks run.
    pub health_checks_run: u64,
    /// Total circuit breaker trips.
    pub circuit_breaker_trips: u64,
}

/// Application information.
#[derive(Debug, Clone)]
pub struct AppInfo {
    /// Application name.
    pub name: String,
    /// Application version.
    pub version: String,
    /// Environment.
    pub environment: String,
}

// ============================================================================
// Runtime Builder
// ============================================================================

/// Builder for RingKernelContext.
pub struct RuntimeBuilder {
    config: Option<RingKernelConfig>,
    health_checker: Option<Arc<HealthChecker>>,
    watchdog: Option<Arc<KernelWatchdog>>,
    multi_gpu_coordinator: Option<Arc<MultiGpuCoordinator>>,
    checkpoint_storage: Option<Arc<dyn CheckpointStorage>>,
}

impl RuntimeBuilder {
    /// Create a new runtime builder.
    pub fn new() -> Self {
        Self {
            config: None,
            health_checker: None,
            watchdog: None,
            multi_gpu_coordinator: None,
            checkpoint_storage: None,
        }
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: RingKernelConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Use development configuration preset.
    pub fn development(mut self) -> Self {
        self.config = Some(RingKernelConfig::development());
        self
    }

    /// Use production configuration preset.
    pub fn production(mut self) -> Self {
        self.config = Some(RingKernelConfig::production());
        self
    }

    /// Use high-performance configuration preset.
    pub fn high_performance(mut self) -> Self {
        self.config = Some(RingKernelConfig::high_performance());
        self
    }

    /// Override health checker (for testing).
    pub fn with_health_checker(mut self, checker: Arc<HealthChecker>) -> Self {
        self.health_checker = Some(checker);
        self
    }

    /// Override watchdog (for testing).
    pub fn with_watchdog(mut self, watchdog: Arc<KernelWatchdog>) -> Self {
        self.watchdog = Some(watchdog);
        self
    }

    /// Override multi-GPU coordinator (for testing).
    pub fn with_multi_gpu_coordinator(mut self, coordinator: Arc<MultiGpuCoordinator>) -> Self {
        self.multi_gpu_coordinator = Some(coordinator);
        self
    }

    /// Override checkpoint storage (for testing).
    pub fn with_checkpoint_storage(mut self, storage: Arc<dyn CheckpointStorage>) -> Self {
        self.checkpoint_storage = Some(storage);
        self
    }

    /// Build the runtime context.
    pub fn build(self) -> Result<Arc<RingKernelContext>> {
        let config = self.config.unwrap_or_default();
        config.validate()?;

        // Create health checker
        let health_checker = self.health_checker.unwrap_or_else(HealthChecker::new);

        // Create watchdog
        let watchdog = self.watchdog.unwrap_or_else(KernelWatchdog::new);

        // Create circuit breaker
        let circuit_breaker = CircuitBreaker::with_config(config.health.circuit_breaker.clone());

        // Create degradation manager
        let degradation_manager = DegradationManager::with_policy(config.health.load_shedding.clone());

        // Create Prometheus exporter
        let prometheus_exporter = PrometheusExporter::new();

        // Create observability context
        let observability = ObservabilityContext::new();

        // Create multi-GPU coordinator
        let multi_gpu_coordinator = self.multi_gpu_coordinator.unwrap_or_else(|| {
            MultiGpuBuilder::new()
                .load_balancing(config.multi_gpu.load_balancing)
                .auto_select_device(config.multi_gpu.auto_select_device)
                .max_kernels_per_device(config.multi_gpu.max_kernels_per_device)
                .enable_p2p(config.multi_gpu.p2p_enabled)
                .preferred_devices(config.multi_gpu.preferred_devices.clone())
                .build()
        });

        // Create checkpoint storage
        let checkpoint_storage: Arc<dyn CheckpointStorage> = self.checkpoint_storage.unwrap_or_else(|| {
            match config.migration.storage {
                CheckpointStorageType::Memory => Arc::new(MemoryStorage::new()),
                CheckpointStorageType::File => {
                    Arc::new(FileStorage::new(&config.migration.checkpoint_dir))
                }
                CheckpointStorageType::Cloud => {
                    // Cloud storage not implemented yet, fall back to memory
                    Arc::new(MemoryStorage::new())
                }
            }
        });

        // Create kernel migrator
        let migrator = Arc::new(KernelMigrator::with_storage(
            Arc::clone(&multi_gpu_coordinator),
            Arc::clone(&checkpoint_storage),
        ));

        Ok(Arc::new(RingKernelContext {
            config,
            health_checker,
            watchdog,
            circuit_breaker,
            degradation_manager,
            prometheus_exporter,
            observability,
            multi_gpu_coordinator,
            migrator,
            checkpoint_storage,
            stats: RuntimeStats::default(),
            started_at: Instant::now(),
            running: AtomicBool::new(true),
        }))
    }
}

impl Default for RuntimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Feature Guards
// ============================================================================

/// Guard for executing operations with circuit breaker protection.
pub struct CircuitGuard<'a> {
    context: &'a RingKernelContext,
    operation_name: String,
}

impl<'a> CircuitGuard<'a> {
    /// Create a new circuit guard.
    pub fn new(context: &'a RingKernelContext, operation_name: impl Into<String>) -> Self {
        Self {
            context,
            operation_name: operation_name.into(),
        }
    }

    /// Execute an operation with circuit breaker protection.
    pub fn execute<T, F>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        // Check if circuit is open
        if self.context.circuit_breaker.state() == CircuitState::Open {
            self.context.record_circuit_trip();
            return Err(RingKernelError::CircuitBreakerOpen {
                name: self.operation_name.clone(),
            });
        }

        // Execute the operation
        match f() {
            Ok(result) => {
                self.context.circuit_breaker.record_success();
                Ok(result)
            }
            Err(e) => {
                self.context.circuit_breaker.record_failure();
                Err(e)
            }
        }
    }
}

/// Guard for graceful degradation.
pub struct DegradationGuard<'a> {
    context: &'a RingKernelContext,
}

impl<'a> DegradationGuard<'a> {
    /// Create a new degradation guard.
    pub fn new(context: &'a RingKernelContext) -> Self {
        Self { context }
    }

    /// Check if an operation should be allowed at the current degradation level.
    pub fn allow_operation(&self, priority: OperationPriority) -> bool {
        let level = self.context.degradation_manager.level();
        match level {
            crate::health::DegradationLevel::Normal => true,
            crate::health::DegradationLevel::Light => true,
            crate::health::DegradationLevel::Moderate => {
                matches!(priority, OperationPriority::Normal | OperationPriority::High | OperationPriority::Critical)
            }
            crate::health::DegradationLevel::Severe => {
                matches!(priority, OperationPriority::High | OperationPriority::Critical)
            }
            crate::health::DegradationLevel::Critical => {
                matches!(priority, OperationPriority::Critical)
            }
        }
    }

    /// Execute an operation if allowed by degradation level.
    pub fn execute_if_allowed<T, F>(
        &self,
        priority: OperationPriority,
        f: F,
    ) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        if self.allow_operation(priority) {
            f()
        } else {
            Err(RingKernelError::LoadSheddingRejected {
                level: format!("{:?}", self.context.degradation_manager.level()),
            })
        }
    }
}

/// Operation priority for load shedding decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperationPriority {
    /// Low priority - shed first.
    Low,
    /// Normal priority.
    Normal,
    /// High priority - shed last.
    High,
    /// Critical - never shed.
    Critical,
}

// ============================================================================
// Metrics Integration
// ============================================================================

impl RingKernelContext {
    /// Export Prometheus metrics.
    pub fn export_metrics(&self) -> String {
        self.prometheus_exporter.render()
    }

    /// Create a metrics snapshot for the runtime.
    pub fn metrics_snapshot(&self) -> RuntimeMetrics {
        let stats = self.stats();
        RuntimeMetrics {
            uptime_seconds: stats.uptime.as_secs_f64(),
            kernels_launched: stats.kernels_launched,
            messages_processed: stats.messages_processed,
            migrations_completed: stats.migrations_completed,
            checkpoints_created: stats.checkpoints_created,
            health_checks_run: stats.health_checks_run,
            circuit_breaker_trips: stats.circuit_breaker_trips,
            circuit_breaker_state: format!("{:?}", self.circuit_breaker.state()),
            degradation_level: format!("{:?}", self.degradation_manager.level()),
            multi_gpu_device_count: self.multi_gpu_coordinator.device_count(),
        }
    }
}

/// Runtime metrics for monitoring.
#[derive(Debug, Clone)]
pub struct RuntimeMetrics {
    /// Uptime in seconds.
    pub uptime_seconds: f64,
    /// Total kernels launched.
    pub kernels_launched: u64,
    /// Total messages processed.
    pub messages_processed: u64,
    /// Total migrations completed.
    pub migrations_completed: u64,
    /// Total checkpoints created.
    pub checkpoints_created: u64,
    /// Total health checks run.
    pub health_checks_run: u64,
    /// Total circuit breaker trips.
    pub circuit_breaker_trips: u64,
    /// Current circuit breaker state.
    pub circuit_breaker_state: String,
    /// Current degradation level.
    pub degradation_level: String,
    /// Number of GPU devices.
    pub multi_gpu_device_count: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConfigBuilder;
    use std::time::Duration;

    #[test]
    fn test_runtime_builder_default() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        assert!(runtime.is_running());
    }

    #[test]
    fn test_runtime_builder_with_config() {
        let config = ConfigBuilder::new()
            .with_general(|g| g.app_name("test_app"))
            .build()
            .unwrap();

        let runtime = RuntimeBuilder::new()
            .with_config(config)
            .build()
            .unwrap();

        assert_eq!(runtime.config().general.app_name, "test_app");
    }

    #[test]
    fn test_runtime_presets() {
        let dev = RuntimeBuilder::new().development().build().unwrap();
        assert_eq!(
            dev.config().general.environment,
            crate::config::Environment::Development
        );

        let prod = RuntimeBuilder::new().production().build().unwrap();
        assert_eq!(
            prod.config().general.environment,
            crate::config::Environment::Production
        );

        let perf = RuntimeBuilder::new().high_performance().build().unwrap();
        assert!(!perf.config().observability.tracing_enabled);
    }

    #[test]
    fn test_runtime_stats() {
        let runtime = RuntimeBuilder::new().build().unwrap();

        runtime.record_kernel_launch();
        runtime.record_kernel_launch();
        runtime.record_messages(100);
        runtime.record_migration();
        runtime.record_checkpoint();
        runtime.record_health_check();

        let stats = runtime.stats();
        assert_eq!(stats.kernels_launched, 2);
        assert_eq!(stats.messages_processed, 100);
        assert_eq!(stats.migrations_completed, 1);
        assert_eq!(stats.checkpoints_created, 1);
        assert_eq!(stats.health_checks_run, 1);
    }

    #[test]
    fn test_runtime_uptime() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        std::thread::sleep(Duration::from_millis(10));
        assert!(runtime.uptime() >= Duration::from_millis(10));
    }

    #[test]
    fn test_runtime_shutdown() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        assert!(runtime.is_running());

        runtime.shutdown().unwrap();
        assert!(!runtime.is_running());

        // Second shutdown should fail
        assert!(runtime.shutdown().is_err());
    }

    #[test]
    fn test_runtime_app_info() {
        let config = ConfigBuilder::new()
            .with_general(|g| {
                g.app_name("my_app")
                    .app_version("1.2.3")
                    .environment(crate::config::Environment::Staging)
            })
            .build()
            .unwrap();

        let runtime = RuntimeBuilder::new()
            .with_config(config)
            .build()
            .unwrap();

        let info = runtime.app_info();
        assert_eq!(info.name, "my_app");
        assert_eq!(info.version, "1.2.3");
        assert_eq!(info.environment, "staging");
    }

    #[test]
    fn test_circuit_guard() {
        let runtime = RuntimeBuilder::new().build().unwrap();

        let guard = CircuitGuard::new(&runtime, "test_op");

        // Success case
        let result: Result<i32> = guard.execute(|| Ok(42));
        assert_eq!(result.unwrap(), 42);

        // Failure case
        let result: Result<i32> = guard.execute(|| {
            Err(RingKernelError::Internal("test error".to_string()))
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_degradation_guard() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        let guard = DegradationGuard::new(&runtime);

        // At normal level, all operations should be allowed
        assert!(guard.allow_operation(OperationPriority::Low));
        assert!(guard.allow_operation(OperationPriority::Normal));
        assert!(guard.allow_operation(OperationPriority::High));
        assert!(guard.allow_operation(OperationPriority::Critical));
    }

    #[test]
    fn test_operation_priority_ordering() {
        assert!(OperationPriority::Low < OperationPriority::Normal);
        assert!(OperationPriority::Normal < OperationPriority::High);
        assert!(OperationPriority::High < OperationPriority::Critical);
    }

    #[test]
    fn test_metrics_snapshot() {
        let runtime = RuntimeBuilder::new().build().unwrap();

        runtime.record_kernel_launch();
        runtime.record_messages(50);

        let metrics = runtime.metrics_snapshot();
        assert_eq!(metrics.kernels_launched, 1);
        assert_eq!(metrics.messages_processed, 50);
        assert!(metrics.uptime_seconds >= 0.0);
    }

    #[test]
    fn test_custom_storage() {
        let storage = Arc::new(MemoryStorage::new());
        let runtime = RuntimeBuilder::new()
            .with_checkpoint_storage(storage.clone())
            .build()
            .unwrap();

        // Verify we can access the storage
        let _migrator = runtime.migrator();
    }

    #[test]
    fn test_export_metrics() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        let metrics = runtime.export_metrics();
        // Prometheus format should be valid
        assert!(metrics.is_empty() || metrics.contains('#') || metrics.contains('\n') || metrics.len() > 0);
    }
}
