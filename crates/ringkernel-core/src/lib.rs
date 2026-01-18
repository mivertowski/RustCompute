//! # RingKernel Core
//!
//! Core traits and types for the RingKernel GPU-native persistent actor system.
//!
//! This crate provides the foundational abstractions for building GPU-accelerated
//! actor systems with persistent kernels, lock-free message passing, and hybrid
//! logical clocks for temporal ordering.
//!
//! ## Core Abstractions
//!
//! - [`RingMessage`] - Trait for messages between kernels
//! - [`MessageQueue`] - Lock-free ring buffer for message passing
//! - [`RingKernelRuntime`] - Backend-agnostic runtime management
//! - [`RingContext`] - GPU intrinsics facade for kernel handlers
//! - [`HlcTimestamp`] - Hybrid Logical Clock for causal ordering
//!
//! ## Example
//!
//! ```ignore
//! use ringkernel_core::prelude::*;
//!
//! #[derive(RingMessage)]
//! struct MyMessage {
//!     #[message(id)]
//!     id: MessageId,
//!     payload: Vec<f32>,
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod analytics_context;
pub mod audit;
pub mod checkpoint;

/// Cloud storage backends for checkpoints (requires `cloud-storage` feature).
#[cfg(feature = "cloud-storage")]
pub mod cloud_storage;

pub mod config;
pub mod context;
pub mod control;
pub mod dispatcher;
pub mod domain;
pub mod error;
pub mod health;
pub mod hlc;
pub mod k2k;
pub mod memory;
pub mod message;
pub mod multi_gpu;
pub mod observability;
pub mod persistent_message;
pub mod pubsub;
pub mod queue;
pub mod reduction;
pub mod runtime;
pub mod runtime_context;
pub mod security;
pub mod state;
pub mod telemetry;
pub mod telemetry_pipeline;
pub mod types;

/// Private module for proc macro integration.
/// Not part of the public API - exposed for macro-generated code only.
#[doc(hidden)]
pub mod __private;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::analytics_context::{
        AllocationHandle, AnalyticsContext, AnalyticsContextBuilder, ContextStats,
    };
    pub use crate::audit::{
        AuditConfig, AuditEvent, AuditEventType, AuditLevel, AuditLogger, AuditLoggerBuilder,
        AuditSink, FileSink, MemorySink,
    };
    pub use crate::config::{
        CheckpointStorageType, CloudStorageConfig, ConfigBuilder, Environment, GeneralConfig,
        GeneralConfigBuilder, HealthConfig, HealthConfigBuilder, LogLevel, MigrationConfig,
        MigrationConfigBuilder, MultiGpuConfig, MultiGpuConfigBuilder, ObservabilityConfig,
        ObservabilityConfigBuilder, RetryConfig, RingKernelConfig,
    };
    pub use crate::context::*;
    pub use crate::control::*;
    pub use crate::dispatcher::{
        DispatcherBuilder, DispatcherConfig, DispatcherMetrics, KernelDispatcher,
    };
    pub use crate::domain::{Domain, DomainMessage, DomainParseError};
    pub use crate::error::*;
    pub use crate::health::{
        BackoffStrategy, CircuitBreaker, CircuitBreakerConfig, CircuitBreakerStats, CircuitState,
        DegradationLevel, DegradationManager, DegradationStats, HealthCheck, HealthCheckResult,
        HealthChecker, HealthStatus, KernelHealth, KernelWatchdog, LoadSheddingPolicy, RetryPolicy,
    };
    pub use crate::hlc::*;
    pub use crate::k2k::{
        DeliveryStatus, K2KBroker, K2KBuilder, K2KConfig, K2KEndpoint, K2KMessage,
        K2KMessageRegistration, K2KTypeRegistry,
    };
    pub use crate::memory::*;
    pub use crate::message::{
        priority, CorrelationId, MessageEnvelope, MessageHeader, MessageId, Priority, RingMessage,
    };
    pub use crate::multi_gpu::{
        CrossGpuK2KRouter, CrossGpuRouterStatsSnapshot, DeviceInfo, DeviceStatus,
        DeviceUnregisterResult, GpuConnection, GpuTopology, HotReloadConfig, HotReloadManager,
        HotReloadRequest, HotReloadResult, HotReloadState, HotReloadStatsSnapshot,
        HotReloadableKernel, InterconnectType, KernelCodeFormat, KernelCodeSource,
        KernelMigrationPlan, KernelMigrator, LoadBalancingStrategy, MigratableKernel,
        MigrationPriority, MigrationRequest, MigrationResult, MigrationState,
        MigrationStatsSnapshot, MultiGpuBuilder, MultiGpuCoordinator, PendingK2KMessage,
        RoutingDecision,
    };
    pub use crate::observability::{
        GpuDeviceMemoryStats, GpuMemoryAllocation, GpuMemoryDashboard, GpuMemoryPoolStats,
        GpuMemoryThresholds, GpuMemoryType, GrafanaDashboard, GrafanaPanel, MemoryPressureLevel,
        ObservabilityContext, PanelType, PrometheusCollector, PrometheusExporter,
        RingKernelCollector, Span, SpanBuilder, SpanEvent, SpanId, SpanKind, SpanStatus, TraceId,
    };
    pub use crate::persistent_message::{
        message_flags, DispatchTable, HandlerRegistration, PersistentMessage,
        MAX_INLINE_PAYLOAD_SIZE,
    };
    pub use crate::pubsub::{PubSubBroker, PubSubBuilder, Publication, QoS, Subscription, Topic};
    pub use crate::queue::*;
    pub use crate::reduction::{
        GlobalReduction, ReductionConfig, ReductionHandle, ReductionOp, ReductionScalar,
    };
    pub use crate::runtime::*;
    pub use crate::runtime_context::{
        AppInfo, BackgroundTaskStatus, CircuitGuard, ContextMetrics, DegradationGuard,
        HealthCycleResult, LifecycleState, MonitoringConfig, MonitoringHandles, OperationPriority,
        RingKernelContext, RuntimeBuilder, RuntimeStatsSnapshot, ShutdownReport, WatchdogResult,
    };
    pub use crate::security::{
        AccessLevel, ComplianceCheck, ComplianceReport, ComplianceReporter, ComplianceStandard,
        ComplianceStatus, ComplianceSummary, EncryptedRegion, EncryptionAlgorithm,
        EncryptionConfig, EncryptionKey, EncryptionStats, KernelSandbox, KeyDerivation,
        MemoryEncryption, ReportFormat, ResourceLimits, SandboxPolicy, SandboxStats,
        SandboxViolation, ViolationType,
    };
    pub use crate::state::{
        ControlBlockStateHelper, EmbeddedState, EmbeddedStateSize, GpuState, StateDescriptor,
        StateSnapshot, CONTROL_BLOCK_STATE_SIZE, STATE_DESCRIPTOR_MAGIC,
    };
    pub use crate::telemetry::*;
    pub use crate::telemetry_pipeline::{
        MetricsCollector, MetricsSnapshot, TelemetryAlert, TelemetryConfig, TelemetryEvent,
        TelemetryPipeline,
    };
    pub use crate::types::*;

    // Cloud storage types (feature-gated)
    #[cfg(feature = "cloud-storage")]
    pub use crate::cloud_storage::{AsyncCheckpointStorage, CloudProvider, S3Config, S3Storage};
}

// Re-exports for convenience
pub use context::RingContext;
pub use control::ControlBlock;
pub use domain::{Domain, DomainMessage};
pub use error::{Result, RingKernelError};
pub use hlc::HlcTimestamp;
pub use memory::{DeviceMemory, GpuBuffer, MemoryPool, PinnedMemory};
pub use message::{priority, MessageHeader, MessageId, Priority, RingMessage};
pub use queue::{MessageQueue, QueueStats};
pub use runtime::{
    Backend, KernelHandle, KernelId, KernelState, KernelStatus, LaunchOptions, RingKernelRuntime,
};
pub use telemetry::TelemetryBuffer;
pub use types::{BlockId, GlobalThreadId, ThreadId, WarpId};
