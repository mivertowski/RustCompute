//! Multi-tenancy support for RingKernel.
//!
//! This module provides tenant isolation, resource quotas, and tenant-aware
//! operations for multi-tenant deployments.
//!
//! # Example
//!
//! ```rust,ignore
//! use ringkernel_core::tenancy::{TenantContext, TenantRegistry, ResourceQuota};
//!
//! let registry = TenantRegistry::new()
//!     .with_tenant("tenant_a", ResourceQuota::new()
//!         .with_max_kernels(100)
//!         .with_max_gpu_memory_mb(8192)
//!         .with_max_messages_per_sec(10000))
//!     .with_tenant("tenant_b", ResourceQuota::new()
//!         .with_max_kernels(50)
//!         .with_max_gpu_memory_mb(4096));
//!
//! let ctx = TenantContext::new("tenant_a");
//! if let Some(quota) = registry.get_quota(&ctx.tenant_id) {
//!     if quota.check_kernel_limit(current_kernels) {
//!         // Launch kernel
//!     }
//! }
//! ```

use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// RESOURCE QUOTA
// ============================================================================

/// Resource quotas for a tenant.
#[derive(Debug, Clone)]
pub struct ResourceQuota {
    /// Maximum number of concurrent kernels.
    pub max_kernels: Option<u64>,
    /// Maximum GPU memory in megabytes.
    pub max_gpu_memory_mb: Option<u64>,
    /// Maximum messages per second.
    pub max_messages_per_sec: Option<u64>,
    /// Maximum K2K endpoints.
    pub max_k2k_endpoints: Option<u64>,
    /// Maximum PubSub subscriptions.
    pub max_pubsub_subscriptions: Option<u64>,
    /// Maximum checkpoint storage in megabytes.
    pub max_checkpoint_storage_mb: Option<u64>,
    /// Maximum CPU time per hour (seconds).
    pub max_cpu_time_per_hour: Option<u64>,
    /// Maximum API requests per minute.
    pub max_api_requests_per_min: Option<u64>,
}

impl ResourceQuota {
    /// Create a new resource quota with no limits.
    pub fn new() -> Self {
        Self {
            max_kernels: None,
            max_gpu_memory_mb: None,
            max_messages_per_sec: None,
            max_k2k_endpoints: None,
            max_pubsub_subscriptions: None,
            max_checkpoint_storage_mb: None,
            max_cpu_time_per_hour: None,
            max_api_requests_per_min: None,
        }
    }

    /// Create an unlimited quota.
    pub fn unlimited() -> Self {
        Self::new()
    }

    /// Set maximum kernels.
    pub fn with_max_kernels(mut self, max: u64) -> Self {
        self.max_kernels = Some(max);
        self
    }

    /// Set maximum GPU memory (MB).
    pub fn with_max_gpu_memory_mb(mut self, max: u64) -> Self {
        self.max_gpu_memory_mb = Some(max);
        self
    }

    /// Set maximum messages per second.
    pub fn with_max_messages_per_sec(mut self, max: u64) -> Self {
        self.max_messages_per_sec = Some(max);
        self
    }

    /// Set maximum K2K endpoints.
    pub fn with_max_k2k_endpoints(mut self, max: u64) -> Self {
        self.max_k2k_endpoints = Some(max);
        self
    }

    /// Set maximum PubSub subscriptions.
    pub fn with_max_pubsub_subscriptions(mut self, max: u64) -> Self {
        self.max_pubsub_subscriptions = Some(max);
        self
    }

    /// Set maximum checkpoint storage (MB).
    pub fn with_max_checkpoint_storage_mb(mut self, max: u64) -> Self {
        self.max_checkpoint_storage_mb = Some(max);
        self
    }

    /// Set maximum CPU time per hour (seconds).
    pub fn with_max_cpu_time_per_hour(mut self, max: u64) -> Self {
        self.max_cpu_time_per_hour = Some(max);
        self
    }

    /// Set maximum API requests per minute.
    pub fn with_max_api_requests_per_min(mut self, max: u64) -> Self {
        self.max_api_requests_per_min = Some(max);
        self
    }

    /// Check if kernel limit allows another kernel.
    pub fn check_kernel_limit(&self, current: u64) -> bool {
        self.max_kernels.map(|max| current < max).unwrap_or(true)
    }

    /// Check if GPU memory limit allows allocation.
    pub fn check_gpu_memory_limit(&self, current_mb: u64, requested_mb: u64) -> bool {
        self.max_gpu_memory_mb
            .map(|max| current_mb + requested_mb <= max)
            .unwrap_or(true)
    }

    /// Check if message rate limit allows another message.
    pub fn check_message_rate(&self, current_rate: u64) -> bool {
        self.max_messages_per_sec
            .map(|max| current_rate < max)
            .unwrap_or(true)
    }

    /// Create a standard small tier quota.
    pub fn tier_small() -> Self {
        Self::new()
            .with_max_kernels(10)
            .with_max_gpu_memory_mb(2048)
            .with_max_messages_per_sec(1000)
            .with_max_k2k_endpoints(20)
            .with_max_pubsub_subscriptions(50)
            .with_max_checkpoint_storage_mb(1024)
            .with_max_api_requests_per_min(100)
    }

    /// Create a standard medium tier quota.
    pub fn tier_medium() -> Self {
        Self::new()
            .with_max_kernels(50)
            .with_max_gpu_memory_mb(8192)
            .with_max_messages_per_sec(10000)
            .with_max_k2k_endpoints(100)
            .with_max_pubsub_subscriptions(200)
            .with_max_checkpoint_storage_mb(10240)
            .with_max_api_requests_per_min(1000)
    }

    /// Create a standard large tier quota.
    pub fn tier_large() -> Self {
        Self::new()
            .with_max_kernels(200)
            .with_max_gpu_memory_mb(32768)
            .with_max_messages_per_sec(100000)
            .with_max_k2k_endpoints(500)
            .with_max_pubsub_subscriptions(1000)
            .with_max_checkpoint_storage_mb(102400)
            .with_max_api_requests_per_min(10000)
    }
}

impl Default for ResourceQuota {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RESOURCE USAGE
// ============================================================================

/// Current resource usage for a tenant.
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Current kernel count.
    pub kernels: u64,
    /// Current GPU memory usage (MB).
    pub gpu_memory_mb: u64,
    /// Messages sent in current window.
    pub messages_this_window: u64,
    /// Current K2K endpoint count.
    pub k2k_endpoints: u64,
    /// Current PubSub subscription count.
    pub pubsub_subscriptions: u64,
    /// Current checkpoint storage (MB).
    pub checkpoint_storage_mb: u64,
    /// API requests in current window.
    pub api_requests_this_window: u64,
    /// Window start time.
    pub window_start: Instant,
}

impl ResourceUsage {
    /// Create new resource usage tracking.
    pub fn new() -> Self {
        Self {
            kernels: 0,
            gpu_memory_mb: 0,
            messages_this_window: 0,
            k2k_endpoints: 0,
            pubsub_subscriptions: 0,
            checkpoint_storage_mb: 0,
            api_requests_this_window: 0,
            window_start: Instant::now(),
        }
    }

    /// Reset windowed counters (messages, API requests).
    pub fn reset_window(&mut self) {
        self.messages_this_window = 0;
        self.api_requests_this_window = 0;
        self.window_start = Instant::now();
    }

    /// Calculate utilization against quota.
    pub fn utilization(&self, quota: &ResourceQuota) -> QuotaUtilization {
        QuotaUtilization {
            kernel_pct: quota
                .max_kernels
                .map(|max| self.kernels as f64 / max as f64 * 100.0),
            gpu_memory_pct: quota
                .max_gpu_memory_mb
                .map(|max| self.gpu_memory_mb as f64 / max as f64 * 100.0),
            message_rate_pct: quota
                .max_messages_per_sec
                .map(|max| self.messages_this_window as f64 / max as f64 * 100.0),
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self::new()
    }
}

/// Quota utilization percentages.
#[derive(Debug, Clone)]
pub struct QuotaUtilization {
    /// Kernel utilization percentage.
    pub kernel_pct: Option<f64>,
    /// GPU memory utilization percentage.
    pub gpu_memory_pct: Option<f64>,
    /// Message rate utilization percentage.
    pub message_rate_pct: Option<f64>,
}

// ============================================================================
// TENANT CONTEXT
// ============================================================================

/// Context for tenant-scoped operations.
#[derive(Debug, Clone)]
pub struct TenantContext {
    /// Tenant ID.
    pub tenant_id: String,
    /// Tenant display name.
    pub display_name: Option<String>,
    /// Tenant metadata.
    pub metadata: HashMap<String, String>,
    /// When the context was created.
    pub created_at: Instant,
}

impl TenantContext {
    /// Create a new tenant context.
    pub fn new(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            display_name: None,
            metadata: HashMap::new(),
            created_at: Instant::now(),
        }
    }

    /// Set display name.
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get the tenant-prefixed resource name.
    pub fn resource_name(&self, resource: &str) -> String {
        format!("{}:{}", self.tenant_id, resource)
    }
}

// ============================================================================
// TENANT REGISTRY
// ============================================================================

/// Error type for tenant operations.
#[derive(Debug, Clone)]
pub enum TenantError {
    /// Tenant not found.
    NotFound(String),
    /// Quota exceeded.
    QuotaExceeded(String),
    /// Tenant already exists.
    AlreadyExists(String),
    /// Tenant is suspended.
    Suspended(String),
    /// Other error.
    Other(String),
}

impl fmt::Display for TenantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(msg) => write!(f, "Tenant not found: {}", msg),
            Self::QuotaExceeded(msg) => write!(f, "Quota exceeded: {}", msg),
            Self::AlreadyExists(msg) => write!(f, "Tenant already exists: {}", msg),
            Self::Suspended(msg) => write!(f, "Tenant suspended: {}", msg),
            Self::Other(msg) => write!(f, "Tenant error: {}", msg),
        }
    }
}

impl std::error::Error for TenantError {}

/// Result type for tenant operations.
pub type TenantResult<T> = Result<T, TenantError>;

/// Internal tenant entry.
struct TenantEntry {
    /// Tenant context.
    _context: TenantContext,
    /// Resource quota.
    quota: ResourceQuota,
    /// Resource usage (with interior mutability).
    usage: RwLock<ResourceUsage>,
    /// Whether tenant is active (interior mutability for suspension).
    active: std::sync::atomic::AtomicBool,
    /// When tenant was registered.
    _registered_at: Instant,
}

/// Registry for managing tenants.
pub struct TenantRegistry {
    /// Registered tenants.
    tenants: RwLock<HashMap<String, Arc<TenantEntry>>>,
    /// Default quota for new tenants.
    default_quota: ResourceQuota,
    /// Window duration for rate limiting.
    window_duration: Duration,
}

impl TenantRegistry {
    /// Create a new tenant registry.
    pub fn new() -> Self {
        Self {
            tenants: RwLock::new(HashMap::new()),
            default_quota: ResourceQuota::tier_small(),
            window_duration: Duration::from_secs(60), // 1 minute windows
        }
    }

    /// Set default quota for new tenants.
    pub fn with_default_quota(mut self, quota: ResourceQuota) -> Self {
        self.default_quota = quota;
        self
    }

    /// Set rate limit window duration.
    pub fn with_window_duration(mut self, duration: Duration) -> Self {
        self.window_duration = duration;
        self
    }

    /// Register a tenant with quota.
    pub fn with_tenant(self, tenant_id: impl Into<String>, quota: ResourceQuota) -> Self {
        let tenant_id = tenant_id.into();
        let entry = TenantEntry {
            _context: TenantContext::new(&tenant_id),
            quota,
            usage: RwLock::new(ResourceUsage::new()),
            active: std::sync::atomic::AtomicBool::new(true),
            _registered_at: Instant::now(),
        };

        self.tenants.write().insert(tenant_id, Arc::new(entry));
        self
    }

    /// Register a new tenant.
    pub fn register_tenant(
        &self,
        tenant_id: impl Into<String>,
        quota: ResourceQuota,
    ) -> TenantResult<()> {
        let tenant_id = tenant_id.into();
        let mut tenants = self.tenants.write();

        if tenants.contains_key(&tenant_id) {
            return Err(TenantError::AlreadyExists(tenant_id));
        }

        let entry = TenantEntry {
            _context: TenantContext::new(&tenant_id),
            quota,
            usage: RwLock::new(ResourceUsage::new()),
            active: std::sync::atomic::AtomicBool::new(true),
            _registered_at: Instant::now(),
        };

        tenants.insert(tenant_id, Arc::new(entry));
        Ok(())
    }

    /// Get a tenant's quota.
    pub fn get_quota(&self, tenant_id: &str) -> Option<ResourceQuota> {
        self.tenants.read().get(tenant_id).map(|e| e.quota.clone())
    }

    /// Get a tenant's current usage.
    pub fn get_usage(&self, tenant_id: &str) -> Option<ResourceUsage> {
        self.tenants
            .read()
            .get(tenant_id)
            .map(|e| e.usage.read().clone())
    }

    /// Check if tenant exists.
    pub fn tenant_exists(&self, tenant_id: &str) -> bool {
        self.tenants.read().contains_key(tenant_id)
    }

    /// Check if tenant is active.
    pub fn is_tenant_active(&self, tenant_id: &str) -> bool {
        self.tenants
            .read()
            .get(tenant_id)
            .map(|e| e.active.load(std::sync::atomic::Ordering::Acquire))
            .unwrap_or(false)
    }

    /// Suspend a tenant. Suspended tenants cannot allocate resources or send messages.
    pub fn suspend_tenant(&self, tenant_id: &str) -> TenantResult<()> {
        let tenants = self.tenants.read();
        let entry = tenants
            .get(tenant_id)
            .ok_or_else(|| TenantError::NotFound(tenant_id.to_string()))?;
        entry
            .active
            .store(false, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Resume a previously suspended tenant.
    pub fn resume_tenant(&self, tenant_id: &str) -> TenantResult<()> {
        let tenants = self.tenants.read();
        let entry = tenants
            .get(tenant_id)
            .ok_or_else(|| TenantError::NotFound(tenant_id.to_string()))?;
        entry
            .active
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Check and increment kernel count.
    pub fn try_allocate_kernel(&self, tenant_id: &str) -> TenantResult<()> {
        let tenants = self.tenants.read();
        let entry = tenants
            .get(tenant_id)
            .ok_or_else(|| TenantError::NotFound(tenant_id.to_string()))?;

        if !entry.active.load(std::sync::atomic::Ordering::Acquire) {
            return Err(TenantError::Suspended(tenant_id.to_string()));
        }

        let mut usage = entry.usage.write();
        if !entry.quota.check_kernel_limit(usage.kernels) {
            return Err(TenantError::QuotaExceeded(format!(
                "Kernel limit reached: {}/{}",
                usage.kernels,
                entry.quota.max_kernels.unwrap_or(0)
            )));
        }

        usage.kernels += 1;
        Ok(())
    }

    /// Release a kernel allocation.
    pub fn release_kernel(&self, tenant_id: &str) {
        if let Some(entry) = self.tenants.read().get(tenant_id) {
            let mut usage = entry.usage.write();
            usage.kernels = usage.kernels.saturating_sub(1);
        }
    }

    /// Check and increment GPU memory.
    pub fn try_allocate_gpu_memory(&self, tenant_id: &str, mb: u64) -> TenantResult<()> {
        let tenants = self.tenants.read();
        let entry = tenants
            .get(tenant_id)
            .ok_or_else(|| TenantError::NotFound(tenant_id.to_string()))?;

        if !entry.active.load(std::sync::atomic::Ordering::Acquire) {
            return Err(TenantError::Suspended(tenant_id.to_string()));
        }

        let mut usage = entry.usage.write();
        if !entry.quota.check_gpu_memory_limit(usage.gpu_memory_mb, mb) {
            return Err(TenantError::QuotaExceeded(format!(
                "GPU memory limit reached: {}MB + {}MB > {}MB",
                usage.gpu_memory_mb,
                mb,
                entry.quota.max_gpu_memory_mb.unwrap_or(0)
            )));
        }

        usage.gpu_memory_mb += mb;
        Ok(())
    }

    /// Release GPU memory allocation.
    pub fn release_gpu_memory(&self, tenant_id: &str, mb: u64) {
        if let Some(entry) = self.tenants.read().get(tenant_id) {
            let mut usage = entry.usage.write();
            usage.gpu_memory_mb = usage.gpu_memory_mb.saturating_sub(mb);
        }
    }

    /// Record a message sent.
    pub fn record_message(&self, tenant_id: &str) -> TenantResult<()> {
        let tenants = self.tenants.read();
        let entry = tenants
            .get(tenant_id)
            .ok_or_else(|| TenantError::NotFound(tenant_id.to_string()))?;

        if !entry.active.load(std::sync::atomic::Ordering::Acquire) {
            return Err(TenantError::Suspended(tenant_id.to_string()));
        }

        let mut usage = entry.usage.write();

        // Reset window if needed
        if usage.window_start.elapsed() >= self.window_duration {
            usage.reset_window();
        }

        if !entry.quota.check_message_rate(usage.messages_this_window) {
            return Err(TenantError::QuotaExceeded(format!(
                "Message rate limit reached: {}/{} per {:?}",
                usage.messages_this_window,
                entry.quota.max_messages_per_sec.unwrap_or(0),
                self.window_duration
            )));
        }

        usage.messages_this_window += 1;
        Ok(())
    }

    /// Get utilization for a tenant.
    pub fn get_utilization(&self, tenant_id: &str) -> Option<QuotaUtilization> {
        self.tenants
            .read()
            .get(tenant_id)
            .map(|entry| entry.usage.read().utilization(&entry.quota))
    }

    /// Get all tenant IDs.
    pub fn tenant_ids(&self) -> Vec<String> {
        self.tenants.read().keys().cloned().collect()
    }

    /// Get tenant count.
    pub fn tenant_count(&self) -> usize {
        self.tenants.read().len()
    }
}

impl Default for TenantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_quota() {
        let quota = ResourceQuota::new()
            .with_max_kernels(10)
            .with_max_gpu_memory_mb(8192);

        assert!(quota.check_kernel_limit(5));
        assert!(quota.check_kernel_limit(9));
        assert!(!quota.check_kernel_limit(10));

        assert!(quota.check_gpu_memory_limit(4096, 2048));
        assert!(!quota.check_gpu_memory_limit(8192, 1));
    }

    #[test]
    fn test_tier_quotas() {
        let small = ResourceQuota::tier_small();
        assert_eq!(small.max_kernels, Some(10));
        assert_eq!(small.max_gpu_memory_mb, Some(2048));

        let large = ResourceQuota::tier_large();
        assert_eq!(large.max_kernels, Some(200));
        assert_eq!(large.max_gpu_memory_mb, Some(32768));
    }

    #[test]
    fn test_tenant_context() {
        let ctx = TenantContext::new("tenant_a")
            .with_display_name("Tenant A")
            .with_metadata("tier", "enterprise");

        assert_eq!(ctx.tenant_id, "tenant_a");
        assert_eq!(ctx.display_name, Some("Tenant A".to_string()));
        assert_eq!(ctx.resource_name("kernel_1"), "tenant_a:kernel_1");
    }

    #[test]
    fn test_tenant_registry() {
        let registry = TenantRegistry::new()
            .with_tenant("tenant_a", ResourceQuota::tier_small())
            .with_tenant("tenant_b", ResourceQuota::tier_medium());

        assert!(registry.tenant_exists("tenant_a"));
        assert!(registry.tenant_exists("tenant_b"));
        assert!(!registry.tenant_exists("tenant_c"));

        let quota_a = registry.get_quota("tenant_a").unwrap();
        assert_eq!(quota_a.max_kernels, Some(10));

        let quota_b = registry.get_quota("tenant_b").unwrap();
        assert_eq!(quota_b.max_kernels, Some(50));
    }

    #[test]
    fn test_kernel_allocation() {
        let registry =
            TenantRegistry::new().with_tenant("tenant_a", ResourceQuota::new().with_max_kernels(2));

        // First two allocations succeed
        assert!(registry.try_allocate_kernel("tenant_a").is_ok());
        assert!(registry.try_allocate_kernel("tenant_a").is_ok());

        // Third fails
        assert!(registry.try_allocate_kernel("tenant_a").is_err());

        // Release one
        registry.release_kernel("tenant_a");

        // Now can allocate again
        assert!(registry.try_allocate_kernel("tenant_a").is_ok());
    }

    #[test]
    fn test_gpu_memory_allocation() {
        let registry = TenantRegistry::new().with_tenant(
            "tenant_a",
            ResourceQuota::new().with_max_gpu_memory_mb(1024),
        );

        assert!(registry.try_allocate_gpu_memory("tenant_a", 512).is_ok());
        assert!(registry.try_allocate_gpu_memory("tenant_a", 256).is_ok());
        // Would exceed limit
        assert!(registry.try_allocate_gpu_memory("tenant_a", 512).is_err());

        // Release and retry
        registry.release_gpu_memory("tenant_a", 256);
        assert!(registry.try_allocate_gpu_memory("tenant_a", 512).is_ok());
    }

    #[test]
    fn test_utilization() {
        let quota = ResourceQuota::new()
            .with_max_kernels(100)
            .with_max_gpu_memory_mb(8192);

        let mut usage = ResourceUsage::new();
        usage.kernels = 50;
        usage.gpu_memory_mb = 4096;

        let utilization = usage.utilization(&quota);
        assert_eq!(utilization.kernel_pct, Some(50.0));
        assert_eq!(utilization.gpu_memory_pct, Some(50.0));
    }

    #[test]
    fn test_unknown_tenant() {
        let registry = TenantRegistry::new();

        assert!(registry.try_allocate_kernel("unknown").is_err());
        assert!(registry.get_quota("unknown").is_none());
    }

    #[test]
    fn test_suspend_and_resume_tenant() {
        let registry = TenantRegistry::new()
            .with_tenant("tenant_a", ResourceQuota::new().with_max_kernels(10));

        assert!(registry.is_tenant_active("tenant_a"));
        assert!(registry.try_allocate_kernel("tenant_a").is_ok());

        // Suspend
        registry.suspend_tenant("tenant_a").unwrap();
        assert!(!registry.is_tenant_active("tenant_a"));

        // Operations should fail while suspended
        assert!(matches!(
            registry.try_allocate_kernel("tenant_a"),
            Err(TenantError::Suspended(_))
        ));
        assert!(matches!(
            registry.try_allocate_gpu_memory("tenant_a", 100),
            Err(TenantError::Suspended(_))
        ));
        assert!(matches!(
            registry.record_message("tenant_a"),
            Err(TenantError::Suspended(_))
        ));

        // Resume
        registry.resume_tenant("tenant_a").unwrap();
        assert!(registry.is_tenant_active("tenant_a"));
        assert!(registry.try_allocate_kernel("tenant_a").is_ok());
    }

    #[test]
    fn test_suspend_unknown_tenant() {
        let registry = TenantRegistry::new();
        assert!(matches!(
            registry.suspend_tenant("unknown"),
            Err(TenantError::NotFound(_))
        ));
        assert!(matches!(
            registry.resume_tenant("unknown"),
            Err(TenantError::NotFound(_))
        ));
    }
}
