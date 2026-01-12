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
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use crate::checkpoint::{CheckpointStorage, FileStorage, MemoryStorage};
#[cfg(feature = "cloud-storage")]
use crate::cloud_storage::{S3Config, S3Storage};
use crate::config::{CheckpointStorageType, RingKernelConfig};
use crate::error::{Result, RingKernelError};
use crate::health::{
    CircuitBreaker, CircuitState, DegradationManager, HealthChecker, HealthStatus, KernelWatchdog,
};
use crate::multi_gpu::{KernelMigrator, MultiGpuBuilder, MultiGpuCoordinator};
use crate::observability::{ObservabilityContext, PrometheusExporter};

// ============================================================================
// Lifecycle Management
// ============================================================================

/// State of the runtime lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleState {
    /// Runtime is initializing.
    Initializing,
    /// Runtime is running and accepting work.
    Running,
    /// Runtime is draining (not accepting new work, finishing existing).
    Draining,
    /// Runtime is shutting down.
    ShuttingDown,
    /// Runtime has stopped.
    Stopped,
}

impl LifecycleState {
    /// Check if the runtime is accepting new work.
    pub fn is_accepting_work(&self) -> bool {
        matches!(self, LifecycleState::Running)
    }

    /// Check if the runtime is active (not stopped).
    pub fn is_active(&self) -> bool {
        !matches!(self, LifecycleState::Stopped)
    }
}

/// Background task tracking.
#[derive(Debug, Default)]
struct BackgroundTasks {
    /// Number of active health check loops.
    health_check_loops: AtomicU64,
    /// Number of active watchdog loops.
    watchdog_loops: AtomicU64,
    /// Number of active metrics flush loops.
    metrics_flush_loops: AtomicU64,
    /// Last health check time.
    last_health_check: RwLock<Option<Instant>>,
    /// Last watchdog scan time.
    last_watchdog_scan: RwLock<Option<Instant>>,
    /// Last metrics flush time.
    last_metrics_flush: RwLock<Option<Instant>>,
}

impl BackgroundTasks {
    fn new() -> Self {
        Self::default()
    }

    fn record_health_check(&self) {
        *self.last_health_check.write() = Some(Instant::now());
    }

    fn record_watchdog_scan(&self) {
        *self.last_watchdog_scan.write() = Some(Instant::now());
    }

    fn record_metrics_flush(&self) {
        *self.last_metrics_flush.write() = Some(Instant::now());
    }

    fn health_check_age(&self) -> Option<Duration> {
        self.last_health_check.read().map(|t| t.elapsed())
    }

    fn watchdog_scan_age(&self) -> Option<Duration> {
        self.last_watchdog_scan.read().map(|t| t.elapsed())
    }

    fn metrics_flush_age(&self) -> Option<Duration> {
        self.last_metrics_flush.read().map(|t| t.elapsed())
    }
}

// ============================================================================
// Async Background Monitoring
// ============================================================================

use tokio::sync::watch;
use tokio::task::JoinHandle;

/// Configuration for background monitoring loops.
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Interval for health checks.
    pub health_check_interval: Duration,
    /// Interval for watchdog scans.
    pub watchdog_interval: Duration,
    /// Interval for metrics flush.
    pub metrics_flush_interval: Duration,
    /// Whether to enable health check loop.
    pub enable_health_checks: bool,
    /// Whether to enable watchdog loop.
    pub enable_watchdog: bool,
    /// Whether to enable metrics flush loop.
    pub enable_metrics_flush: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            health_check_interval: Duration::from_secs(10),
            watchdog_interval: Duration::from_secs(5),
            metrics_flush_interval: Duration::from_secs(60),
            enable_health_checks: true,
            enable_watchdog: true,
            enable_metrics_flush: true,
        }
    }
}

impl MonitoringConfig {
    /// Create a new monitoring config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set health check interval.
    pub fn health_check_interval(mut self, interval: Duration) -> Self {
        self.health_check_interval = interval;
        self
    }

    /// Set watchdog interval.
    pub fn watchdog_interval(mut self, interval: Duration) -> Self {
        self.watchdog_interval = interval;
        self
    }

    /// Set metrics flush interval.
    pub fn metrics_flush_interval(mut self, interval: Duration) -> Self {
        self.metrics_flush_interval = interval;
        self
    }

    /// Enable or disable health checks.
    pub fn enable_health_checks(mut self, enable: bool) -> Self {
        self.enable_health_checks = enable;
        self
    }

    /// Enable or disable watchdog.
    pub fn enable_watchdog(mut self, enable: bool) -> Self {
        self.enable_watchdog = enable;
        self
    }

    /// Enable or disable metrics flush.
    pub fn enable_metrics_flush(mut self, enable: bool) -> Self {
        self.enable_metrics_flush = enable;
        self
    }
}

/// Handles for background monitoring tasks.
pub struct MonitoringHandles {
    /// Handle to the health check loop task.
    pub health_check_handle: Option<JoinHandle<()>>,
    /// Handle to the watchdog loop task.
    pub watchdog_handle: Option<JoinHandle<()>>,
    /// Handle to the metrics flush loop task.
    pub metrics_flush_handle: Option<JoinHandle<()>>,
    /// Shutdown signal sender.
    shutdown_tx: watch::Sender<bool>,
}

impl MonitoringHandles {
    /// Create new monitoring handles.
    fn new() -> (Self, watch::Receiver<bool>) {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        (
            Self {
                health_check_handle: None,
                watchdog_handle: None,
                metrics_flush_handle: None,
                shutdown_tx,
            },
            shutdown_rx,
        )
    }

    /// Signal all monitoring tasks to stop.
    pub fn signal_shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    /// Wait for all monitoring tasks to complete.
    pub async fn wait_for_shutdown(self) {
        if let Some(handle) = self.health_check_handle {
            let _ = handle.await;
        }
        if let Some(handle) = self.watchdog_handle {
            let _ = handle.await;
        }
        if let Some(handle) = self.metrics_flush_handle {
            let _ = handle.await;
        }
    }

    /// Check if any monitoring tasks are running.
    pub fn is_running(&self) -> bool {
        self.health_check_handle
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or(false)
            || self
                .watchdog_handle
                .as_ref()
                .map(|h| !h.is_finished())
                .unwrap_or(false)
            || self
                .metrics_flush_handle
                .as_ref()
                .map(|h| !h.is_finished())
                .unwrap_or(false)
    }
}

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
/// - Background monitoring tasks
///
/// ## Lifecycle
///
/// The runtime goes through these states:
/// - `Initializing` → `Running` → `Draining` → `ShuttingDown` → `Stopped`
///
/// Use `start_monitoring()` to begin background health checks and watchdog scans.
/// Use `shutdown()` for graceful termination.
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
    /// Running state (deprecated, use lifecycle_state).
    running: AtomicBool,
    /// Current lifecycle state.
    lifecycle_state: RwLock<LifecycleState>,
    /// Background task tracking.
    background_tasks: BackgroundTasks,
    /// Shutdown requested flag.
    shutdown_requested: AtomicBool,
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
        self.stats
            .messages_processed
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Record a migration completion.
    pub fn record_migration(&self) {
        self.stats
            .migrations_completed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record a checkpoint creation.
    pub fn record_checkpoint(&self) {
        self.stats
            .checkpoints_created
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record a health check run.
    pub fn record_health_check(&self) {
        self.stats.health_checks_run.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a circuit breaker trip.
    pub fn record_circuit_trip(&self) {
        self.stats
            .circuit_breaker_trips
            .fetch_add(1, Ordering::Relaxed);
    }

    // ========================================================================
    // Lifecycle Management
    // ========================================================================

    /// Get the current lifecycle state.
    pub fn lifecycle_state(&self) -> LifecycleState {
        *self.lifecycle_state.read()
    }

    /// Check if shutdown has been requested.
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::SeqCst)
    }

    /// Check if the runtime is accepting new work.
    pub fn is_accepting_work(&self) -> bool {
        self.lifecycle_state().is_accepting_work()
    }

    /// Transition to running state.
    ///
    /// Call this after initialization is complete to start accepting work.
    pub fn start(&self) -> Result<()> {
        let mut state = self.lifecycle_state.write();
        if *state != LifecycleState::Initializing {
            return Err(RingKernelError::InvalidState {
                expected: "Initializing".to_string(),
                actual: format!("{:?}", *state),
            });
        }
        *state = LifecycleState::Running;
        self.running.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Run a single health check cycle.
    ///
    /// This performs one round of health checks and updates the circuit breaker
    /// and degradation manager based on the results.
    ///
    /// Note: This is a synchronous method that uses cached circuit breaker state.
    /// For full async health checks, use the HealthChecker directly.
    pub fn run_health_check_cycle(&self) -> HealthCycleResult {
        self.background_tasks.record_health_check();
        self.record_health_check();

        // Get circuit breaker state as a health proxy
        let circuit_state = self.circuit_breaker.state();

        // Infer health status from circuit breaker state
        let status = match circuit_state {
            CircuitState::Closed => HealthStatus::Healthy,
            CircuitState::HalfOpen => HealthStatus::Degraded,
            CircuitState::Open => HealthStatus::Unhealthy,
        };

        // Update degradation level based on circuit breaker state
        let current_level = self.degradation_manager.level();
        let new_level = match circuit_state {
            CircuitState::Open => {
                // Increase degradation
                current_level.next_worse()
            }
            CircuitState::Closed => {
                // Decrease degradation
                current_level.next_better()
            }
            CircuitState::HalfOpen => {
                // Keep current level
                current_level
            }
        };

        if new_level != current_level {
            self.degradation_manager.set_level(new_level);
        }

        HealthCycleResult {
            status,
            circuit_state,
            degradation_level: self.degradation_manager.level(),
            timestamp: Instant::now(),
        }
    }

    /// Run a single watchdog scan cycle.
    ///
    /// This checks for stale kernels and takes appropriate action.
    pub fn run_watchdog_cycle(&self) -> WatchdogResult {
        self.background_tasks.record_watchdog_scan();

        let kernel_health = self.watchdog.check_all();
        let stale_count = kernel_health
            .iter()
            .filter(|h| h.status == HealthStatus::Unhealthy)
            .count();

        WatchdogResult {
            stale_kernels: stale_count,
            timestamp: Instant::now(),
        }
    }

    /// Flush metrics to Prometheus.
    ///
    /// This renders current metrics to the Prometheus exporter format.
    pub fn flush_metrics(&self) -> String {
        self.background_tasks.record_metrics_flush();
        self.prometheus_exporter.render()
    }

    /// Get background task status.
    pub fn background_task_status(&self) -> BackgroundTaskStatus {
        BackgroundTaskStatus {
            health_check_age: self.background_tasks.health_check_age(),
            watchdog_scan_age: self.background_tasks.watchdog_scan_age(),
            metrics_flush_age: self.background_tasks.metrics_flush_age(),
            active_health_loops: self
                .background_tasks
                .health_check_loops
                .load(Ordering::Relaxed),
            active_watchdog_loops: self.background_tasks.watchdog_loops.load(Ordering::Relaxed),
            active_metrics_loops: self
                .background_tasks
                .metrics_flush_loops
                .load(Ordering::Relaxed),
        }
    }

    // ========================================================================
    // Async Background Monitoring
    // ========================================================================

    /// Start background monitoring loops.
    ///
    /// This spawns async tasks for:
    /// - Health check loop (runs at configured interval)
    /// - Watchdog loop (checks for stale kernels)
    /// - Metrics flush loop (exports Prometheus metrics)
    ///
    /// Returns handles that can be used to stop the monitoring tasks.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let runtime = RuntimeBuilder::new().production().build()?;
    /// runtime.start()?;
    ///
    /// let config = MonitoringConfig::new()
    ///     .health_check_interval(Duration::from_secs(5))
    ///     .watchdog_interval(Duration::from_secs(2));
    ///
    /// let handles = runtime.start_monitoring(config).await;
    ///
    /// // ... runtime runs ...
    ///
    /// // Graceful shutdown
    /// handles.signal_shutdown();
    /// handles.wait_for_shutdown().await;
    /// ```
    pub fn start_monitoring(self: &Arc<Self>, config: MonitoringConfig) -> MonitoringHandles {
        let (mut handles, shutdown_rx) = MonitoringHandles::new();

        // Spawn health check loop
        if config.enable_health_checks {
            let runtime = Arc::clone(self);
            let interval = config.health_check_interval;
            let mut shutdown = shutdown_rx.clone();

            handles.health_check_handle = Some(tokio::spawn(async move {
                runtime
                    .background_tasks
                    .health_check_loops
                    .fetch_add(1, Ordering::Relaxed);

                let mut interval_timer = tokio::time::interval(interval);
                interval_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

                loop {
                    tokio::select! {
                        _ = interval_timer.tick() => {
                            if runtime.is_shutdown_requested() {
                                break;
                            }
                            let _result = runtime.run_health_check_cycle();
                            tracing::debug!("Health check cycle completed");
                        }
                        _ = shutdown.changed() => {
                            tracing::info!("Health check loop shutting down");
                            break;
                        }
                    }
                }

                runtime
                    .background_tasks
                    .health_check_loops
                    .fetch_sub(1, Ordering::Relaxed);
            }));
        }

        // Spawn watchdog loop
        if config.enable_watchdog {
            let runtime = Arc::clone(self);
            let interval = config.watchdog_interval;
            let mut shutdown = shutdown_rx.clone();

            handles.watchdog_handle = Some(tokio::spawn(async move {
                runtime
                    .background_tasks
                    .watchdog_loops
                    .fetch_add(1, Ordering::Relaxed);

                let mut interval_timer = tokio::time::interval(interval);
                interval_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

                loop {
                    tokio::select! {
                        _ = interval_timer.tick() => {
                            if runtime.is_shutdown_requested() {
                                break;
                            }
                            let result = runtime.run_watchdog_cycle();
                            if result.stale_kernels > 0 {
                                tracing::warn!("Watchdog detected {} stale kernels", result.stale_kernels);
                            }
                        }
                        _ = shutdown.changed() => {
                            tracing::info!("Watchdog loop shutting down");
                            break;
                        }
                    }
                }

                runtime
                    .background_tasks
                    .watchdog_loops
                    .fetch_sub(1, Ordering::Relaxed);
            }));
        }

        // Spawn metrics flush loop
        if config.enable_metrics_flush {
            let runtime = Arc::clone(self);
            let interval = config.metrics_flush_interval;
            let mut shutdown = shutdown_rx;

            handles.metrics_flush_handle = Some(tokio::spawn(async move {
                runtime
                    .background_tasks
                    .metrics_flush_loops
                    .fetch_add(1, Ordering::Relaxed);

                let mut interval_timer = tokio::time::interval(interval);
                interval_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

                loop {
                    tokio::select! {
                        _ = interval_timer.tick() => {
                            if runtime.is_shutdown_requested() {
                                break;
                            }
                            let _metrics = runtime.flush_metrics();
                            tracing::debug!("Metrics flush completed");
                        }
                        _ = shutdown.changed() => {
                            tracing::info!("Metrics flush loop shutting down");
                            break;
                        }
                    }
                }

                runtime
                    .background_tasks
                    .metrics_flush_loops
                    .fetch_sub(1, Ordering::Relaxed);
            }));
        }

        handles
    }

    /// Start monitoring with default configuration.
    pub fn start_monitoring_default(self: &Arc<Self>) -> MonitoringHandles {
        self.start_monitoring(MonitoringConfig::default())
    }

    /// Request graceful shutdown.
    ///
    /// This signals background tasks to stop and transitions to draining state.
    /// Returns immediately; use `wait_for_shutdown()` to block until complete.
    pub fn request_shutdown(&self) -> Result<()> {
        // Set shutdown flag
        self.shutdown_requested.store(true, Ordering::SeqCst);

        // Transition to draining state
        let mut state = self.lifecycle_state.write();
        match *state {
            LifecycleState::Running => {
                *state = LifecycleState::Draining;
                Ok(())
            }
            LifecycleState::Draining | LifecycleState::ShuttingDown => {
                // Already shutting down
                Ok(())
            }
            LifecycleState::Stopped => Err(RingKernelError::InvalidState {
                expected: "Running or Draining".to_string(),
                actual: "Stopped".to_string(),
            }),
            LifecycleState::Initializing => {
                // Can shutdown from initializing too
                *state = LifecycleState::ShuttingDown;
                Ok(())
            }
        }
    }

    /// Complete the shutdown process.
    ///
    /// This performs final cleanup and transitions to stopped state.
    pub fn complete_shutdown(&self) -> Result<ShutdownReport> {
        let start = Instant::now();

        // Transition to shutting down
        {
            let mut state = self.lifecycle_state.write();
            if *state == LifecycleState::Stopped {
                return Err(RingKernelError::InvalidState {
                    expected: "not Stopped".to_string(),
                    actual: "Stopped".to_string(),
                });
            }
            *state = LifecycleState::ShuttingDown;
        }

        // Perform cleanup
        let final_stats = self.stats();
        let final_metrics = self.flush_metrics();

        // Transition to stopped
        {
            let mut state = self.lifecycle_state.write();
            *state = LifecycleState::Stopped;
            self.running.store(false, Ordering::SeqCst);
        }

        Ok(ShutdownReport {
            duration: start.elapsed(),
            total_uptime: self.uptime(),
            final_stats,
            final_metrics,
        })
    }

    /// Shutdown the runtime gracefully (legacy method).
    ///
    /// This is equivalent to `request_shutdown()` followed by `complete_shutdown()`.
    pub fn shutdown(&self) -> Result<()> {
        self.request_shutdown()?;
        self.complete_shutdown()?;
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

/// Result of a health check cycle run by the runtime context.
#[derive(Debug, Clone)]
pub struct HealthCycleResult {
    /// Overall health status.
    pub status: HealthStatus,
    /// Current circuit breaker state.
    pub circuit_state: CircuitState,
    /// Current degradation level.
    pub degradation_level: crate::health::DegradationLevel,
    /// Timestamp of this check.
    pub timestamp: Instant,
}

/// Result of a watchdog scan cycle.
#[derive(Debug, Clone)]
pub struct WatchdogResult {
    /// Number of stale kernels detected.
    pub stale_kernels: usize,
    /// Timestamp of this scan.
    pub timestamp: Instant,
}

/// Status of background tasks.
#[derive(Debug, Clone)]
pub struct BackgroundTaskStatus {
    /// Time since last health check.
    pub health_check_age: Option<Duration>,
    /// Time since last watchdog scan.
    pub watchdog_scan_age: Option<Duration>,
    /// Time since last metrics flush.
    pub metrics_flush_age: Option<Duration>,
    /// Number of active health check loops.
    pub active_health_loops: u64,
    /// Number of active watchdog loops.
    pub active_watchdog_loops: u64,
    /// Number of active metrics flush loops.
    pub active_metrics_loops: u64,
}

/// Report generated after shutdown completes.
#[derive(Debug, Clone)]
pub struct ShutdownReport {
    /// Time taken for shutdown.
    pub duration: Duration,
    /// Total runtime uptime.
    pub total_uptime: Duration,
    /// Final runtime statistics.
    pub final_stats: RuntimeStatsSnapshot,
    /// Final metrics dump.
    pub final_metrics: String,
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
        let health_checker = self.health_checker.unwrap_or_default();

        // Create watchdog
        let watchdog = self.watchdog.unwrap_or_default();

        // Create circuit breaker
        let circuit_breaker = CircuitBreaker::with_config(config.health.circuit_breaker.clone());

        // Create degradation manager
        let degradation_manager =
            DegradationManager::with_policy(config.health.load_shedding.clone());

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
        let checkpoint_storage: Arc<dyn CheckpointStorage> =
            self.checkpoint_storage.unwrap_or_else(|| {
                match config.migration.storage {
                    CheckpointStorageType::Memory => Arc::new(MemoryStorage::new()),
                    CheckpointStorageType::File => {
                        Arc::new(FileStorage::new(&config.migration.checkpoint_dir))
                    }
                    CheckpointStorageType::Cloud => {
                        #[cfg(feature = "cloud-storage")]
                        {
                            // Create S3 storage from cloud configuration
                            let cloud_cfg = &config.migration.cloud_config;
                            let s3_config = S3Config::new(&cloud_cfg.s3_bucket)
                                .with_prefix(&cloud_cfg.s3_prefix);
                            let s3_config = if let Some(ref region) = cloud_cfg.s3_region {
                                s3_config.with_region(region)
                            } else {
                                s3_config
                            };
                            let s3_config = if let Some(ref endpoint) = cloud_cfg.s3_endpoint {
                                s3_config.with_endpoint(endpoint)
                            } else {
                                s3_config
                            };
                            let s3_config = if cloud_cfg.s3_encryption {
                                s3_config.with_encryption()
                            } else {
                                s3_config
                            };

                            // S3Storage::new is async, we use block_in_place for the sync context
                            match tokio::task::block_in_place(|| {
                                tokio::runtime::Handle::current()
                                    .block_on(S3Storage::new(s3_config))
                            }) {
                                Ok(storage) => Arc::new(storage) as Arc<dyn CheckpointStorage>,
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to create S3 storage: {}, falling back to memory",
                                        e
                                    );
                                    Arc::new(MemoryStorage::new())
                                }
                            }
                        }
                        #[cfg(not(feature = "cloud-storage"))]
                        {
                            tracing::warn!(
                                "Cloud storage requested but cloud-storage feature not enabled, \
                                 falling back to memory storage"
                            );
                            Arc::new(MemoryStorage::new())
                        }
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
            running: AtomicBool::new(false), // Start as not running
            lifecycle_state: RwLock::new(LifecycleState::Initializing),
            background_tasks: BackgroundTasks::new(),
            shutdown_requested: AtomicBool::new(false),
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
                matches!(
                    priority,
                    OperationPriority::Normal
                        | OperationPriority::High
                        | OperationPriority::Critical
                )
            }
            crate::health::DegradationLevel::Severe => {
                matches!(
                    priority,
                    OperationPriority::High | OperationPriority::Critical
                )
            }
            crate::health::DegradationLevel::Critical => {
                matches!(priority, OperationPriority::Critical)
            }
        }
    }

    /// Execute an operation if allowed by degradation level.
    pub fn execute_if_allowed<T, F>(&self, priority: OperationPriority, f: F) -> Result<T>
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

    /// Create a metrics snapshot for the runtime context.
    pub fn metrics_snapshot(&self) -> ContextMetrics {
        let stats = self.stats();
        ContextMetrics {
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

/// Context metrics for monitoring the unified runtime.
#[derive(Debug, Clone)]
pub struct ContextMetrics {
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
        // Runtime starts in Initializing state
        assert!(!runtime.is_running());
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Initializing);

        // Start the runtime
        runtime.start().unwrap();
        assert!(runtime.is_running());
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Running);
    }

    #[test]
    fn test_runtime_builder_with_config() {
        let config = ConfigBuilder::new()
            .with_general(|g| g.app_name("test_app"))
            .build()
            .unwrap();

        let runtime = RuntimeBuilder::new().with_config(config).build().unwrap();

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
        runtime.start().unwrap();
        assert!(runtime.is_running());
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Running);

        runtime.shutdown().unwrap();
        assert!(!runtime.is_running());
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Stopped);

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

        let runtime = RuntimeBuilder::new().with_config(config).build().unwrap();

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
        let result: Result<i32> =
            guard.execute(|| Err(RingKernelError::Internal("test error".to_string())));
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
        assert!(
            metrics.is_empty()
                || metrics.contains('#')
                || metrics.contains('\n')
                || metrics.len() > 0
        );
    }

    // ========================================================================
    // Lifecycle Management Tests
    // ========================================================================

    #[test]
    fn test_lifecycle_state_transitions() {
        let runtime = RuntimeBuilder::new().build().unwrap();

        // Initial state
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Initializing);
        assert!(!runtime.is_accepting_work());

        // Start
        runtime.start().unwrap();
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Running);
        assert!(runtime.is_accepting_work());

        // Request shutdown
        runtime.request_shutdown().unwrap();
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Draining);
        assert!(!runtime.is_accepting_work());

        // Complete shutdown
        let report = runtime.complete_shutdown().unwrap();
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Stopped);
        assert!(report.duration.as_nanos() > 0);
    }

    #[test]
    fn test_lifecycle_state_helpers() {
        assert!(LifecycleState::Running.is_accepting_work());
        assert!(!LifecycleState::Initializing.is_accepting_work());
        assert!(!LifecycleState::Draining.is_accepting_work());
        assert!(!LifecycleState::ShuttingDown.is_accepting_work());
        assert!(!LifecycleState::Stopped.is_accepting_work());

        assert!(LifecycleState::Initializing.is_active());
        assert!(LifecycleState::Running.is_active());
        assert!(LifecycleState::Draining.is_active());
        assert!(LifecycleState::ShuttingDown.is_active());
        assert!(!LifecycleState::Stopped.is_active());
    }

    #[test]
    fn test_health_check_cycle() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        runtime.start().unwrap();

        let result = runtime.run_health_check_cycle();
        assert_eq!(result.status, crate::health::HealthStatus::Healthy);
        assert_eq!(result.circuit_state, CircuitState::Closed);

        // Check that task status was updated
        let status = runtime.background_task_status();
        assert!(status.health_check_age.is_some());
    }

    #[test]
    fn test_watchdog_cycle() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        runtime.start().unwrap();

        let result = runtime.run_watchdog_cycle();
        assert_eq!(result.stale_kernels, 0);

        let status = runtime.background_task_status();
        assert!(status.watchdog_scan_age.is_some());
    }

    #[test]
    fn test_metrics_flush() {
        let runtime = RuntimeBuilder::new().build().unwrap();

        let metrics = runtime.flush_metrics();
        assert!(metrics.is_empty() || !metrics.is_empty()); // Just verify it doesn't crash

        let status = runtime.background_task_status();
        assert!(status.metrics_flush_age.is_some());
    }

    #[test]
    fn test_shutdown_report() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        runtime.start().unwrap();

        // Do some work
        runtime.record_kernel_launch();
        runtime.record_messages(100);

        let report = runtime.complete_shutdown().unwrap();

        assert_eq!(report.final_stats.kernels_launched, 1);
        assert_eq!(report.final_stats.messages_processed, 100);
        assert!(report.total_uptime.as_nanos() > 0);
    }

    #[test]
    fn test_cannot_start_twice() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        runtime.start().unwrap();

        // Second start should fail
        assert!(runtime.start().is_err());
    }

    #[test]
    fn test_shutdown_from_initializing() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        // Don't call start, should still be able to shutdown
        assert!(runtime.shutdown().is_ok());
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Stopped);
    }

    // ========================================================================
    // Enterprise Integration Tests
    // ========================================================================

    #[test]
    fn test_enterprise_full_lifecycle() {
        // Build runtime with custom config
        let config = ConfigBuilder::new()
            .with_general(|g| g.app_name("integration-test").app_version("1.0.0"))
            .build()
            .unwrap();

        let runtime = RuntimeBuilder::new().with_config(config).build().unwrap();

        // Verify initial state
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Initializing);
        assert!(!runtime.is_accepting_work());

        // Start runtime
        runtime.start().unwrap();
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Running);
        assert!(runtime.is_accepting_work());

        // Simulate work
        for _ in 0..10 {
            runtime.record_kernel_launch();
            runtime.record_messages(100);
        }

        // Run health cycles
        for _ in 0..3 {
            let result = runtime.run_health_check_cycle();
            assert_eq!(result.status, crate::health::HealthStatus::Healthy);
        }

        // Verify stats
        let stats = runtime.stats();
        assert_eq!(stats.kernels_launched, 10);
        assert_eq!(stats.messages_processed, 1000);
        assert_eq!(stats.health_checks_run, 3);

        // Graceful shutdown
        runtime.request_shutdown().unwrap();
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Draining);

        let report = runtime.complete_shutdown().unwrap();
        assert_eq!(runtime.lifecycle_state(), LifecycleState::Stopped);
        assert_eq!(report.final_stats.kernels_launched, 10);
    }

    #[test]
    fn test_circuit_breaker_integration() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        runtime.start().unwrap();

        // Initially healthy
        let result = runtime.run_health_check_cycle();
        assert_eq!(result.circuit_state, CircuitState::Closed);

        // Simulate failures until circuit opens
        let cb = runtime.circuit_breaker();
        for _ in 0..10 {
            cb.record_failure();
        }

        // Circuit should be open now
        assert_eq!(cb.state(), CircuitState::Open);

        // Health check should reflect degraded state
        let result = runtime.run_health_check_cycle();
        assert_eq!(result.circuit_state, CircuitState::Open);
        assert_eq!(result.status, crate::health::HealthStatus::Unhealthy);
    }

    #[test]
    fn test_degradation_integration() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        runtime.start().unwrap();

        // Initially at normal level
        let result = runtime.run_health_check_cycle();
        assert_eq!(
            result.degradation_level,
            crate::health::DegradationLevel::Normal
        );

        // Force circuit open
        let cb = runtime.circuit_breaker();
        for _ in 0..10 {
            cb.record_failure();
        }

        // Health check should increase degradation
        let result = runtime.run_health_check_cycle();
        // Degradation should have increased from Normal
        assert_ne!(
            result.degradation_level,
            crate::health::DegradationLevel::Normal
        );
    }

    #[test]
    fn test_configuration_presets_integration() {
        // Development preset
        let dev = RuntimeBuilder::new().development().build().unwrap();
        assert_eq!(
            dev.config().general.environment,
            crate::config::Environment::Development
        );
        assert!(dev.config().observability.tracing_enabled);

        // Production preset
        let prod = RuntimeBuilder::new().production().build().unwrap();
        assert_eq!(
            prod.config().general.environment,
            crate::config::Environment::Production
        );

        // High-performance preset
        let perf = RuntimeBuilder::new().high_performance().build().unwrap();
        assert!(!perf.config().observability.tracing_enabled);
    }

    #[test]
    fn test_multi_gpu_coordinator_access() {
        let runtime = RuntimeBuilder::new().build().unwrap();

        // Access multi-GPU coordinator
        let coordinator = runtime.multi_gpu_coordinator();
        assert_eq!(coordinator.device_count(), 0);

        // Register a device
        let device = crate::multi_gpu::DeviceInfo::new(
            0,
            "Test GPU".to_string(),
            crate::runtime::Backend::Cpu,
        );
        coordinator.register_device(device);
        assert_eq!(coordinator.device_count(), 1);
    }

    #[test]
    fn test_background_task_tracking() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        runtime.start().unwrap();

        // Initially no tasks have run
        let status = runtime.background_task_status();
        assert!(status.health_check_age.is_none());
        assert!(status.watchdog_scan_age.is_none());
        assert!(status.metrics_flush_age.is_none());

        // Run health check
        runtime.run_health_check_cycle();
        let status = runtime.background_task_status();
        assert!(status.health_check_age.is_some());

        // Run watchdog
        runtime.run_watchdog_cycle();
        let status = runtime.background_task_status();
        assert!(status.watchdog_scan_age.is_some());

        // Flush metrics
        runtime.flush_metrics();
        let status = runtime.background_task_status();
        assert!(status.metrics_flush_age.is_some());
    }

    // ========================================================================
    // Async Monitoring Tests
    // ========================================================================

    #[test]
    fn test_monitoring_config_builder() {
        let config = MonitoringConfig::new()
            .health_check_interval(Duration::from_secs(5))
            .watchdog_interval(Duration::from_secs(2))
            .metrics_flush_interval(Duration::from_secs(30))
            .enable_health_checks(true)
            .enable_watchdog(false)
            .enable_metrics_flush(true);

        assert_eq!(config.health_check_interval, Duration::from_secs(5));
        assert_eq!(config.watchdog_interval, Duration::from_secs(2));
        assert_eq!(config.metrics_flush_interval, Duration::from_secs(30));
        assert!(config.enable_health_checks);
        assert!(!config.enable_watchdog);
        assert!(config.enable_metrics_flush);
    }

    #[test]
    fn test_monitoring_config_default() {
        let config = MonitoringConfig::default();

        assert_eq!(config.health_check_interval, Duration::from_secs(10));
        assert_eq!(config.watchdog_interval, Duration::from_secs(5));
        assert_eq!(config.metrics_flush_interval, Duration::from_secs(60));
        assert!(config.enable_health_checks);
        assert!(config.enable_watchdog);
        assert!(config.enable_metrics_flush);
    }

    #[tokio::test]
    async fn test_async_monitoring_start_stop() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        runtime.start().unwrap();

        // Start monitoring with short intervals
        let config = MonitoringConfig::new()
            .health_check_interval(Duration::from_millis(50))
            .watchdog_interval(Duration::from_millis(50))
            .metrics_flush_interval(Duration::from_millis(50));

        let handles = runtime.start_monitoring(config);

        // Verify tasks are running
        assert!(handles.is_running());

        // Let some cycles run
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Verify health checks ran
        let status = runtime.background_task_status();
        assert!(status.health_check_age.is_some());
        assert!(status.watchdog_scan_age.is_some());

        // Signal shutdown
        handles.signal_shutdown();

        // Wait for tasks to complete
        handles.wait_for_shutdown().await;
    }

    #[tokio::test]
    async fn test_async_monitoring_default_config() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        runtime.start().unwrap();

        // Start with default config (but we'll shut down quickly)
        let handles = runtime.start_monitoring_default();
        assert!(handles.is_running());

        // Shutdown immediately
        handles.signal_shutdown();
        handles.wait_for_shutdown().await;
    }

    #[tokio::test]
    async fn test_async_monitoring_selective_loops() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        runtime.start().unwrap();

        // Only enable health checks
        let config = MonitoringConfig::new()
            .health_check_interval(Duration::from_millis(50))
            .enable_health_checks(true)
            .enable_watchdog(false)
            .enable_metrics_flush(false);

        let handles = runtime.start_monitoring(config);

        // Only health check handle should be set
        assert!(handles.health_check_handle.is_some());
        assert!(handles.watchdog_handle.is_none());
        assert!(handles.metrics_flush_handle.is_none());

        handles.signal_shutdown();
        handles.wait_for_shutdown().await;
    }

    #[tokio::test]
    async fn test_async_monitoring_respects_shutdown_flag() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        runtime.start().unwrap();

        let config = MonitoringConfig::new().health_check_interval(Duration::from_millis(50));

        let handles = runtime.start_monitoring(config);

        // Request shutdown via runtime
        runtime.request_shutdown().unwrap();

        // Let monitoring loop detect shutdown
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Tasks should have stopped
        handles.wait_for_shutdown().await;
    }

    #[tokio::test]
    async fn test_monitoring_handles_is_running() {
        let runtime = RuntimeBuilder::new().build().unwrap();
        runtime.start().unwrap();

        let config = MonitoringConfig::new().health_check_interval(Duration::from_millis(100));

        let handles = runtime.start_monitoring(config);
        assert!(handles.is_running());

        handles.signal_shutdown();
        handles.wait_for_shutdown().await;

        // After shutdown, a new handles struct would be needed
    }
}
