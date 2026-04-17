//! K2K tenant registry, quotas, and per-engagement cost tracking.
//!
//! This module implements the lower-level, per-message tenancy boundary used
//! inside the K2K broker. It is distinct from the governance-level
//! [`crate::tenancy`] module (which uses string tenant IDs and coarse
//! kernel-count / memory quotas). Here, [`TenantId`] is a `u64` that matches
//! the format stamped into per-message envelopes and into GPU-side routing
//! tables.
//!
//! # Responsibilities
//!
//! - Register tenants with [`TenantQuota`] limits
//! - Check quotas on kernel registration and message sends
//! - Track billable GPU-seconds per `(tenant_id, audit_tag)` pair
//! - Emit audit events for cross-tenant attempts via [`AuditSink`]

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use crate::audit::{AuditEvent, AuditEventType, AuditLevel, AuditSink};
use crate::error::{Result, RingKernelError};
use crate::k2k::audit_tag::AuditTag;

/// Stable tenant identifier as used in K2K routing tables and message envelopes.
///
/// `0` is reserved for the "unspecified tenant" — the default for legacy
/// single-tenant deployments. Tenant IDs are `u64` so they can be stamped
/// directly into GPU-shared routing entries without string marshalling.
pub type TenantId = u64;

/// Reserved tenant ID meaning "no tenant specified" (backward-compat default).
pub const UNSPECIFIED_TENANT: TenantId = 0;

// ============================================================================
// TenantQuota
// ============================================================================

/// Per-tenant quota and cost tracking limits.
///
/// Unlike the governance-level [`crate::tenancy::ResourceQuota`], this is the
/// hot-path quota checked on every K2K send.
#[derive(Debug, Clone)]
pub struct TenantQuota {
    /// Maximum concurrent kernels this tenant may register.
    pub max_concurrent_kernels: u32,
    /// Maximum GPU memory (bytes) the tenant may allocate.
    pub max_gpu_memory_bytes: u64,
    /// Maximum messages per second across all of the tenant's kernels.
    pub max_messages_per_sec: u64,
    /// Per-engagement billable-time budgets, keyed by
    /// [`AuditTag::engagement_id`]. Absent entries = no budget configured.
    pub per_engagement_budget: HashMap<u64, Duration>,
}

impl TenantQuota {
    /// Create a quota with no hard limits (useful for tests / trusted tenants).
    pub fn unlimited() -> Self {
        Self {
            max_concurrent_kernels: u32::MAX,
            max_gpu_memory_bytes: u64::MAX,
            max_messages_per_sec: u64::MAX,
            per_engagement_budget: HashMap::new(),
        }
    }

    /// Create a modest default quota: 16 kernels, 2 GiB, 100k msgs/sec.
    pub fn standard() -> Self {
        Self {
            max_concurrent_kernels: 16,
            max_gpu_memory_bytes: 2 * 1024 * 1024 * 1024,
            max_messages_per_sec: 100_000,
            per_engagement_budget: HashMap::new(),
        }
    }

    /// Set a budget for a specific engagement.
    pub fn with_engagement_budget(mut self, engagement_id: u64, budget: Duration) -> Self {
        self.per_engagement_budget.insert(engagement_id, budget);
        self
    }
}

impl Default for TenantQuota {
    fn default() -> Self {
        Self::standard()
    }
}

// ============================================================================
// TenantInfo (internal registry entry)
// ============================================================================

/// Per-tenant bookkeeping stored inside [`TenantRegistry`].
#[derive(Debug)]
pub struct TenantInfo {
    /// Tenant identity.
    pub tenant_id: TenantId,
    /// Quota (immutable after registration; re-register to update).
    pub quota: TenantQuota,
    /// Number of kernels currently registered for this tenant.
    pub current_kernels: AtomicU64,
    /// Number of messages sent in the current rate-limit window.
    pub messages_this_window: AtomicU64,
    /// Start of the current rate-limit window (seconds-granularity).
    pub window_start_secs: AtomicU64,
    /// Per-engagement billable nanoseconds (keyed by `engagement_id`).
    pub engagement_cost_ns: RwLock<HashMap<u64, u64>>,
    /// When the tenant was registered.
    pub registered_at: Instant,
}

impl TenantInfo {
    fn new(tenant_id: TenantId, quota: TenantQuota) -> Self {
        let now_secs = secs_since_epoch();
        Self {
            tenant_id,
            quota,
            current_kernels: AtomicU64::new(0),
            messages_this_window: AtomicU64::new(0),
            window_start_secs: AtomicU64::new(now_secs),
            engagement_cost_ns: RwLock::new(HashMap::new()),
            registered_at: Instant::now(),
        }
    }
}

fn secs_since_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ============================================================================
// TenantRegistry
// ============================================================================

/// Registry of active tenants with per-tenant quotas and per-engagement cost
/// accounting.
pub struct TenantRegistry {
    tenants: RwLock<HashMap<TenantId, Arc<TenantInfo>>>,
    audit_sink: Option<Arc<dyn AuditSink>>,
}

impl TenantRegistry {
    /// Create an empty registry with no audit sink.
    pub fn new() -> Self {
        Self {
            tenants: RwLock::new(HashMap::new()),
            audit_sink: None,
        }
    }

    /// Create a registry that forwards audit events (cross-tenant attempts,
    /// quota violations) to the given sink.
    pub fn with_audit_sink(sink: Arc<dyn AuditSink>) -> Self {
        Self {
            tenants: RwLock::new(HashMap::new()),
            audit_sink: Some(sink),
        }
    }

    /// Replace the audit sink on an existing registry.
    pub fn set_audit_sink(&mut self, sink: Arc<dyn AuditSink>) {
        self.audit_sink = Some(sink);
    }

    /// Register a new tenant with the given quota.
    pub fn register(&self, tenant_id: TenantId, quota: TenantQuota) -> Result<()> {
        let mut tenants = self.tenants.write();
        if tenants.contains_key(&tenant_id) {
            return Err(RingKernelError::InvalidConfig(format!(
                "tenant {} already registered",
                tenant_id
            )));
        }
        tenants.insert(tenant_id, Arc::new(TenantInfo::new(tenant_id, quota)));
        Ok(())
    }

    /// Deregister a tenant, dropping all quota and cost state.
    pub fn deregister(&self, tenant_id: TenantId) -> bool {
        self.tenants.write().remove(&tenant_id).is_some()
    }

    /// Returns `true` if the tenant is registered.
    pub fn is_registered(&self, tenant_id: TenantId) -> bool {
        self.tenants.read().contains_key(&tenant_id)
    }

    /// Get a snapshot of tenant info (fast `Arc` clone, no lock held).
    pub fn get(&self, tenant_id: TenantId) -> Option<Arc<TenantInfo>> {
        self.tenants.read().get(&tenant_id).cloned()
    }

    /// Number of registered tenants.
    pub fn tenant_count(&self) -> usize {
        self.tenants.read().len()
    }

    /// All registered tenant IDs.
    pub fn tenant_ids(&self) -> Vec<TenantId> {
        self.tenants.read().keys().copied().collect()
    }

    /// Check whether a send from `tenant_id` with `audit_tag` is within quota.
    pub fn check_quota(&self, tenant_id: TenantId, audit_tag: AuditTag) -> Result<()> {
        if tenant_id == UNSPECIFIED_TENANT {
            return Ok(());
        }

        let info = self
            .tenants
            .read()
            .get(&tenant_id)
            .cloned()
            .ok_or_else(|| {
                RingKernelError::InvalidConfig(format!("tenant {} not registered", tenant_id))
            })?;

        // Rate limit: reset the window every second.
        let now = secs_since_epoch();
        let window_start = info.window_start_secs.load(Ordering::Relaxed);
        if now != window_start {
            let _ = info.window_start_secs.compare_exchange(
                window_start,
                now,
                Ordering::AcqRel,
                Ordering::Relaxed,
            );
            info.messages_this_window.store(0, Ordering::Relaxed);
        }
        let sent = info.messages_this_window.load(Ordering::Relaxed);
        if sent >= info.quota.max_messages_per_sec {
            let err = RingKernelError::LoadSheddingRejected {
                level: format!(
                    "tenant {} message-rate: {}/{}",
                    tenant_id, sent, info.quota.max_messages_per_sec
                ),
            };
            self.audit_quota_exceeded(tenant_id, audit_tag, &err);
            return Err(err);
        }

        if let Some(budget) = info
            .quota
            .per_engagement_budget
            .get(&audit_tag.engagement_id)
        {
            let used_ns = info
                .engagement_cost_ns
                .read()
                .get(&audit_tag.engagement_id)
                .copied()
                .unwrap_or(0);
            if Duration::from_nanos(used_ns) >= *budget {
                let err = RingKernelError::LoadSheddingRejected {
                    level: format!(
                        "tenant {} engagement {} budget exceeded: {}ns >= {}ns",
                        tenant_id,
                        audit_tag.engagement_id,
                        used_ns,
                        budget.as_nanos()
                    ),
                };
                self.audit_quota_exceeded(tenant_id, audit_tag, &err);
                return Err(err);
            }
        }

        Ok(())
    }

    /// Record a message send for rate-limit accounting.
    pub fn record_message(&self, tenant_id: TenantId) {
        if tenant_id == UNSPECIFIED_TENANT {
            return;
        }
        if let Some(info) = self.tenants.read().get(&tenant_id) {
            info.messages_this_window.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record billable GPU-seconds against a `(tenant_id, audit_tag)` pair.
    pub fn track_usage(&self, tenant_id: TenantId, audit_tag: AuditTag, gpu_seconds: Duration) {
        if tenant_id == UNSPECIFIED_TENANT {
            return;
        }
        if let Some(info) = self.tenants.read().get(&tenant_id) {
            let mut map = info.engagement_cost_ns.write();
            let entry = map.entry(audit_tag.engagement_id).or_insert(0);
            *entry = entry.saturating_add(gpu_seconds.as_nanos() as u64);
        }
    }

    /// Get total billable duration for a specific engagement across all
    /// registered tenants that have reported cost for it.
    pub fn get_engagement_cost(&self, audit_tag: AuditTag) -> Duration {
        let mut total_ns: u128 = 0;
        for info in self.tenants.read().values() {
            if let Some(ns) = info
                .engagement_cost_ns
                .read()
                .get(&audit_tag.engagement_id)
                .copied()
            {
                total_ns = total_ns.saturating_add(ns as u128);
            }
        }
        Duration::from_nanos(total_ns.min(u64::MAX as u128) as u64)
    }

    /// Get billable duration for a specific `(tenant_id, engagement_id)` pair.
    pub fn get_engagement_cost_for(&self, tenant_id: TenantId, audit_tag: AuditTag) -> Duration {
        self.tenants
            .read()
            .get(&tenant_id)
            .and_then(|info| {
                info.engagement_cost_ns
                    .read()
                    .get(&audit_tag.engagement_id)
                    .copied()
            })
            .map(Duration::from_nanos)
            .unwrap_or(Duration::ZERO)
    }

    /// Returns `true` if a send would be cross-tenant.
    pub fn is_cross_tenant(&self, from: TenantId, to: TenantId) -> bool {
        from != to
    }

    /// Emit an audit event for a cross-tenant K2K send attempt (rejected).
    pub fn audit_cross_tenant(
        &self,
        from_tenant: TenantId,
        to_tenant: TenantId,
        source_kernel: &str,
        destination_kernel: &str,
        audit_tag: AuditTag,
    ) {
        let Some(sink) = self.audit_sink.as_ref() else {
            return;
        };
        let event = AuditEvent::new(
            AuditLevel::Security,
            AuditEventType::SecurityViolation,
            "k2k_broker",
            format!(
                "cross-tenant K2K send rejected: from tenant {} to tenant {}",
                from_tenant, to_tenant
            ),
        )
        .with_target(destination_kernel.to_string())
        .with_metadata("from_tenant", from_tenant.to_string())
        .with_metadata("to_tenant", to_tenant.to_string())
        .with_metadata("source_kernel", source_kernel.to_string())
        .with_metadata("destination_kernel", destination_kernel.to_string())
        .with_metadata("org_id", audit_tag.org_id.to_string())
        .with_metadata("engagement_id", audit_tag.engagement_id.to_string());
        let _ = sink.write(&event);
    }

    fn audit_quota_exceeded(
        &self,
        tenant_id: TenantId,
        audit_tag: AuditTag,
        err: &RingKernelError,
    ) {
        let Some(sink) = self.audit_sink.as_ref() else {
            return;
        };
        let event = AuditEvent::new(
            AuditLevel::Warning,
            AuditEventType::ResourceLimitExceeded,
            "k2k_tenant_registry",
            format!("tenant {} quota exceeded: {}", tenant_id, err),
        )
        .with_metadata("tenant", tenant_id.to_string())
        .with_metadata("org_id", audit_tag.org_id.to_string())
        .with_metadata("engagement_id", audit_tag.engagement_id.to_string());
        let _ = sink.write(&event);
    }

    /// Increment the registered-kernel counter for this tenant, enforcing
    /// [`TenantQuota::max_concurrent_kernels`].
    pub fn acquire_kernel_slot(&self, tenant_id: TenantId) -> Result<()> {
        if tenant_id == UNSPECIFIED_TENANT {
            return Ok(());
        }
        let info = self
            .tenants
            .read()
            .get(&tenant_id)
            .cloned()
            .ok_or_else(|| {
                RingKernelError::InvalidConfig(format!("tenant {} not registered", tenant_id))
            })?;
        let max = info.quota.max_concurrent_kernels as u64;
        loop {
            let current = info.current_kernels.load(Ordering::Acquire);
            if current >= max {
                return Err(RingKernelError::LoadSheddingRejected {
                    level: format!("tenant {} kernel-slots: {}/{}", tenant_id, current, max),
                });
            }
            if info
                .current_kernels
                .compare_exchange(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    /// Release a kernel slot (call on kernel termination / deregister).
    pub fn release_kernel_slot(&self, tenant_id: TenantId) {
        if tenant_id == UNSPECIFIED_TENANT {
            return;
        }
        if let Some(info) = self.tenants.read().get(&tenant_id) {
            loop {
                let current = info.current_kernels.load(Ordering::Acquire);
                if current == 0 {
                    return;
                }
                if info
                    .current_kernels
                    .compare_exchange(current, current - 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return;
                }
            }
        }
    }
}

impl Default for TenantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TenantRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TenantRegistry")
            .field("tenants", &self.tenants.read().len())
            .field("has_audit_sink", &self.audit_sink.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::MemorySink;

    #[test]
    fn test_register_and_query() {
        let reg = TenantRegistry::new();
        assert_eq!(reg.tenant_count(), 0);
        reg.register(1, TenantQuota::default()).unwrap();
        assert!(reg.is_registered(1));
        assert_eq!(reg.tenant_count(), 1);
    }

    #[test]
    fn test_duplicate_registration_fails() {
        let reg = TenantRegistry::new();
        reg.register(1, TenantQuota::default()).unwrap();
        let err = reg.register(1, TenantQuota::default()).unwrap_err();
        assert!(matches!(err, RingKernelError::InvalidConfig(_)));
    }

    #[test]
    fn test_deregister() {
        let reg = TenantRegistry::new();
        reg.register(1, TenantQuota::default()).unwrap();
        assert!(reg.deregister(1));
        assert!(!reg.is_registered(1));
        assert!(!reg.deregister(1));
    }

    #[test]
    fn test_unspecified_tenant_fast_path() {
        let reg = TenantRegistry::new();
        assert!(reg
            .check_quota(UNSPECIFIED_TENANT, AuditTag::default())
            .is_ok());
        reg.record_message(UNSPECIFIED_TENANT);
        reg.track_usage(
            UNSPECIFIED_TENANT,
            AuditTag::default(),
            Duration::from_millis(100),
        );
    }

    #[test]
    fn test_track_usage_and_cost() {
        let reg = TenantRegistry::new();
        reg.register(42, TenantQuota::unlimited()).unwrap();

        let tag_a = AuditTag::new(42, 1);
        let tag_b = AuditTag::new(42, 2);

        reg.track_usage(42, tag_a, Duration::from_millis(250));
        reg.track_usage(42, tag_a, Duration::from_millis(750));
        reg.track_usage(42, tag_b, Duration::from_millis(100));

        assert_eq!(
            reg.get_engagement_cost_for(42, tag_a),
            Duration::from_millis(1000)
        );
        assert_eq!(
            reg.get_engagement_cost_for(42, tag_b),
            Duration::from_millis(100)
        );
        assert_eq!(reg.get_engagement_cost(tag_a), Duration::from_millis(1000));
    }

    #[test]
    fn test_cross_tenant_engagement_aggregation() {
        let reg = TenantRegistry::new();
        reg.register(1, TenantQuota::unlimited()).unwrap();
        reg.register(2, TenantQuota::unlimited()).unwrap();

        let tag = AuditTag::new(99, 7);
        reg.track_usage(1, tag, Duration::from_millis(500));
        reg.track_usage(2, tag, Duration::from_millis(500));

        assert_eq!(reg.get_engagement_cost(tag), Duration::from_millis(1000));
    }

    #[test]
    fn test_check_quota_message_rate() {
        let mut quota = TenantQuota::default();
        quota.max_messages_per_sec = 5;
        let reg = TenantRegistry::new();
        reg.register(1, quota).unwrap();

        for _ in 0..5 {
            assert!(reg.check_quota(1, AuditTag::default()).is_ok());
            reg.record_message(1);
        }
        let err = reg.check_quota(1, AuditTag::default()).unwrap_err();
        assert!(matches!(err, RingKernelError::LoadSheddingRejected { .. }));
    }

    #[test]
    fn test_check_quota_engagement_budget() {
        let quota = TenantQuota::default().with_engagement_budget(7, Duration::from_millis(100));
        let reg = TenantRegistry::new();
        reg.register(1, quota).unwrap();

        let tag = AuditTag::new(0, 7);
        assert!(reg.check_quota(1, tag).is_ok());
        reg.track_usage(1, tag, Duration::from_millis(150));
        let err = reg.check_quota(1, tag).unwrap_err();
        assert!(matches!(err, RingKernelError::LoadSheddingRejected { .. }));
    }

    #[test]
    fn test_audit_cross_tenant_logs_event() {
        let sink = Arc::new(MemorySink::new(100));
        let reg = TenantRegistry::with_audit_sink(sink.clone());
        reg.register(1, TenantQuota::default()).unwrap();
        reg.register(2, TenantQuota::default()).unwrap();

        reg.audit_cross_tenant(1, 2, "src", "dst", AuditTag::new(100, 200));
        let events = sink.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, AuditEventType::SecurityViolation);
        assert!(events[0].description.contains("cross-tenant"));
    }

    #[test]
    fn test_kernel_slot_acquire_release() {
        let mut quota = TenantQuota::default();
        quota.max_concurrent_kernels = 2;
        let reg = TenantRegistry::new();
        reg.register(1, quota).unwrap();

        assert!(reg.acquire_kernel_slot(1).is_ok());
        assert!(reg.acquire_kernel_slot(1).is_ok());
        let err = reg.acquire_kernel_slot(1).unwrap_err();
        assert!(matches!(err, RingKernelError::LoadSheddingRejected { .. }));

        reg.release_kernel_slot(1);
        assert!(reg.acquire_kernel_slot(1).is_ok());
    }

    #[test]
    fn test_kernel_slot_unregistered_tenant_fails() {
        let reg = TenantRegistry::new();
        let err = reg.acquire_kernel_slot(99).unwrap_err();
        assert!(matches!(err, RingKernelError::InvalidConfig(_)));
    }

    #[test]
    fn test_is_cross_tenant() {
        let reg = TenantRegistry::new();
        assert!(!reg.is_cross_tenant(1, 1));
        assert!(reg.is_cross_tenant(1, 2));
        assert!(reg.is_cross_tenant(UNSPECIFIED_TENANT, 1));
    }
}
