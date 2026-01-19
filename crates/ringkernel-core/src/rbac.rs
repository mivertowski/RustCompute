//! Role-Based Access Control (RBAC) for RingKernel.
//!
//! This module provides fine-grained access control for kernel operations
//! with predefined roles and customizable permission policies.
//!
//! # Example
//!
//! ```rust,ignore
//! use ringkernel_core::rbac::{RbacPolicy, Role, Permission, PolicyEvaluator};
//! use ringkernel_core::auth::AuthContext;
//!
//! let policy = RbacPolicy::new()
//!     .with_role(Role::admin())
//!     .with_role(Role::operator())
//!     .with_role(Role::developer())
//!     .with_role(Role::readonly());
//!
//! let evaluator = PolicyEvaluator::new(policy);
//!
//! // Check if user can launch kernels
//! if evaluator.is_allowed(&auth_context, Permission::KernelLaunch) {
//!     // Launch kernel
//! }
//! ```

use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::KernelId;

// ============================================================================
// PERMISSIONS
// ============================================================================

/// Fine-grained permissions for kernel operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Permission {
    // Kernel lifecycle
    /// Launch new kernels.
    KernelLaunch,
    /// Terminate kernels.
    KernelTerminate,
    /// Activate kernels.
    KernelActivate,
    /// View kernel status.
    KernelView,
    /// Migrate kernels between GPUs.
    KernelMigrate,
    /// Create kernel checkpoints.
    KernelCheckpoint,
    /// Restore kernels from checkpoints.
    KernelRestore,

    // Messaging
    /// Send messages to kernels.
    MessageSend,
    /// Receive messages from kernels.
    MessageReceive,
    /// Send K2K messages.
    K2KSend,
    /// Register K2K endpoints.
    K2KRegister,
    /// Publish to topics.
    PubSubPublish,
    /// Subscribe to topics.
    PubSubSubscribe,

    // Resource management
    /// Allocate GPU memory.
    MemoryAllocate,
    /// View memory usage.
    MemoryView,
    /// Configure resource limits.
    ResourceConfigure,

    // Configuration
    /// View configuration.
    ConfigView,
    /// Modify configuration.
    ConfigModify,
    /// View security settings.
    SecurityView,
    /// Modify security settings.
    SecurityModify,

    // Observability
    /// View metrics.
    MetricsView,
    /// Export metrics.
    MetricsExport,
    /// View audit logs.
    AuditView,
    /// Export audit logs.
    AuditExport,
    /// View traces.
    TracesView,

    // Admin
    /// Full administrative access.
    Admin,
    /// Manage users and roles.
    UserManage,
    /// Manage tenants.
    TenantManage,
}

impl Permission {
    /// Get the permission name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::KernelLaunch => "kernel:launch",
            Self::KernelTerminate => "kernel:terminate",
            Self::KernelActivate => "kernel:activate",
            Self::KernelView => "kernel:view",
            Self::KernelMigrate => "kernel:migrate",
            Self::KernelCheckpoint => "kernel:checkpoint",
            Self::KernelRestore => "kernel:restore",
            Self::MessageSend => "message:send",
            Self::MessageReceive => "message:receive",
            Self::K2KSend => "k2k:send",
            Self::K2KRegister => "k2k:register",
            Self::PubSubPublish => "pubsub:publish",
            Self::PubSubSubscribe => "pubsub:subscribe",
            Self::MemoryAllocate => "memory:allocate",
            Self::MemoryView => "memory:view",
            Self::ResourceConfigure => "resource:configure",
            Self::ConfigView => "config:view",
            Self::ConfigModify => "config:modify",
            Self::SecurityView => "security:view",
            Self::SecurityModify => "security:modify",
            Self::MetricsView => "metrics:view",
            Self::MetricsExport => "metrics:export",
            Self::AuditView => "audit:view",
            Self::AuditExport => "audit:export",
            Self::TracesView => "traces:view",
            Self::Admin => "admin",
            Self::UserManage => "user:manage",
            Self::TenantManage => "tenant:manage",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "kernel:launch" => Some(Self::KernelLaunch),
            "kernel:terminate" => Some(Self::KernelTerminate),
            "kernel:activate" => Some(Self::KernelActivate),
            "kernel:view" => Some(Self::KernelView),
            "kernel:migrate" => Some(Self::KernelMigrate),
            "kernel:checkpoint" => Some(Self::KernelCheckpoint),
            "kernel:restore" => Some(Self::KernelRestore),
            "message:send" => Some(Self::MessageSend),
            "message:receive" => Some(Self::MessageReceive),
            "k2k:send" => Some(Self::K2KSend),
            "k2k:register" => Some(Self::K2KRegister),
            "pubsub:publish" => Some(Self::PubSubPublish),
            "pubsub:subscribe" => Some(Self::PubSubSubscribe),
            "memory:allocate" => Some(Self::MemoryAllocate),
            "memory:view" => Some(Self::MemoryView),
            "resource:configure" => Some(Self::ResourceConfigure),
            "config:view" => Some(Self::ConfigView),
            "config:modify" => Some(Self::ConfigModify),
            "security:view" => Some(Self::SecurityView),
            "security:modify" => Some(Self::SecurityModify),
            "metrics:view" => Some(Self::MetricsView),
            "metrics:export" => Some(Self::MetricsExport),
            "audit:view" => Some(Self::AuditView),
            "audit:export" => Some(Self::AuditExport),
            "traces:view" => Some(Self::TracesView),
            "admin" => Some(Self::Admin),
            "user:manage" => Some(Self::UserManage),
            "tenant:manage" => Some(Self::TenantManage),
            _ => None,
        }
    }

    /// Get all permissions in a category.
    pub fn category_permissions(category: &str) -> Vec<Permission> {
        match category {
            "kernel" => vec![
                Self::KernelLaunch,
                Self::KernelTerminate,
                Self::KernelActivate,
                Self::KernelView,
                Self::KernelMigrate,
                Self::KernelCheckpoint,
                Self::KernelRestore,
            ],
            "message" => vec![Self::MessageSend, Self::MessageReceive],
            "k2k" => vec![Self::K2KSend, Self::K2KRegister],
            "pubsub" => vec![Self::PubSubPublish, Self::PubSubSubscribe],
            "memory" => vec![Self::MemoryAllocate, Self::MemoryView],
            "config" => vec![Self::ConfigView, Self::ConfigModify],
            "security" => vec![Self::SecurityView, Self::SecurityModify],
            "metrics" => vec![Self::MetricsView, Self::MetricsExport],
            "audit" => vec![Self::AuditView, Self::AuditExport],
            _ => vec![],
        }
    }
}

impl fmt::Display for Permission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// ROLE
// ============================================================================

/// A role with a set of permissions.
#[derive(Debug, Clone)]
pub struct Role {
    /// Role name.
    pub name: String,
    /// Role description.
    pub description: String,
    /// Granted permissions.
    pub permissions: HashSet<Permission>,
    /// Whether this role has all permissions.
    pub is_superuser: bool,
}

impl Role {
    /// Create a new role.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            permissions: HashSet::new(),
            is_superuser: false,
        }
    }

    /// Add a permission.
    pub fn with_permission(mut self, permission: Permission) -> Self {
        self.permissions.insert(permission);
        self
    }

    /// Add multiple permissions.
    pub fn with_permissions<I>(mut self, permissions: I) -> Self
    where
        I: IntoIterator<Item = Permission>,
    {
        self.permissions.extend(permissions);
        self
    }

    /// Mark as superuser (has all permissions).
    pub fn with_superuser(mut self, is_superuser: bool) -> Self {
        self.is_superuser = is_superuser;
        self
    }

    /// Check if role has a permission.
    pub fn has_permission(&self, permission: Permission) -> bool {
        self.is_superuser || self.permissions.contains(&permission)
    }

    /// Create the Admin role (full access).
    pub fn admin() -> Self {
        Self::new("admin", "Full administrative access to all operations")
            .with_superuser(true)
    }

    /// Create the Operator role (run/manage kernels, no config changes).
    pub fn operator() -> Self {
        Self::new("operator", "Can launch, manage, and monitor kernels")
            .with_permissions([
                Permission::KernelLaunch,
                Permission::KernelTerminate,
                Permission::KernelActivate,
                Permission::KernelView,
                Permission::KernelMigrate,
                Permission::KernelCheckpoint,
                Permission::KernelRestore,
                Permission::MessageSend,
                Permission::MessageReceive,
                Permission::K2KSend,
                Permission::K2KRegister,
                Permission::PubSubPublish,
                Permission::PubSubSubscribe,
                Permission::MemoryAllocate,
                Permission::MemoryView,
                Permission::MetricsView,
                Permission::TracesView,
            ])
    }

    /// Create the Developer role (launch and debug kernels).
    pub fn developer() -> Self {
        Self::new("developer", "Can launch kernels and view debug information")
            .with_permissions([
                Permission::KernelLaunch,
                Permission::KernelActivate,
                Permission::KernelView,
                Permission::MessageSend,
                Permission::MessageReceive,
                Permission::K2KSend,
                Permission::MemoryView,
                Permission::MetricsView,
                Permission::TracesView,
                Permission::ConfigView,
            ])
    }

    /// Create the ReadOnly role (view only).
    pub fn readonly() -> Self {
        Self::new("readonly", "Can only view status and metrics")
            .with_permissions([
                Permission::KernelView,
                Permission::MemoryView,
                Permission::MetricsView,
                Permission::ConfigView,
            ])
    }

    /// Create the Service role (for automated systems).
    pub fn service() -> Self {
        Self::new("service", "For automated services and integrations")
            .with_permissions([
                Permission::KernelLaunch,
                Permission::KernelTerminate,
                Permission::KernelActivate,
                Permission::KernelView,
                Permission::MessageSend,
                Permission::MessageReceive,
                Permission::K2KSend,
                Permission::K2KRegister,
                Permission::PubSubPublish,
                Permission::PubSubSubscribe,
                Permission::MemoryAllocate,
                Permission::MemoryView,
                Permission::MetricsView,
                Permission::MetricsExport,
            ])
    }
}

// ============================================================================
// RBAC POLICY
// ============================================================================

/// RBAC policy definition.
#[derive(Debug, Clone)]
pub struct RbacPolicy {
    /// Defined roles.
    roles: HashMap<String, Role>,
    /// Resource-specific rules.
    resource_rules: Vec<ResourceRule>,
    /// Default deny (if true, explicit allow required).
    default_deny: bool,
}

/// A rule for specific resources.
#[derive(Debug, Clone)]
pub struct ResourceRule {
    /// Resource pattern (supports wildcards).
    pub resource_pattern: String,
    /// Required permissions for this resource.
    pub required_permissions: HashSet<Permission>,
    /// Additional required roles.
    pub required_roles: HashSet<String>,
    /// Explicit deny (overrides allows).
    pub deny: bool,
}

impl ResourceRule {
    /// Create a new resource rule.
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            resource_pattern: pattern.into(),
            required_permissions: HashSet::new(),
            required_roles: HashSet::new(),
            deny: false,
        }
    }

    /// Require a permission.
    pub fn require_permission(mut self, permission: Permission) -> Self {
        self.required_permissions.insert(permission);
        self
    }

    /// Require a role.
    pub fn require_role(mut self, role: impl Into<String>) -> Self {
        self.required_roles.insert(role.into());
        self
    }

    /// Make this a deny rule.
    pub fn deny(mut self) -> Self {
        self.deny = true;
        self
    }

    /// Check if resource matches pattern.
    pub fn matches(&self, resource: &str) -> bool {
        if self.resource_pattern == "*" {
            return true;
        }

        if self.resource_pattern.ends_with('*') {
            let prefix = &self.resource_pattern[..self.resource_pattern.len() - 1];
            resource.starts_with(prefix)
        } else {
            resource == self.resource_pattern
        }
    }
}

impl RbacPolicy {
    /// Create a new RBAC policy.
    pub fn new() -> Self {
        Self {
            roles: HashMap::new(),
            resource_rules: Vec::new(),
            default_deny: true,
        }
    }

    /// Set default deny mode.
    pub fn with_default_deny(mut self, deny: bool) -> Self {
        self.default_deny = deny;
        self
    }

    /// Add a role.
    pub fn with_role(mut self, role: Role) -> Self {
        self.roles.insert(role.name.clone(), role);
        self
    }

    /// Add a resource rule.
    pub fn with_resource_rule(mut self, rule: ResourceRule) -> Self {
        self.resource_rules.push(rule);
        self
    }

    /// Get a role by name.
    pub fn get_role(&self, name: &str) -> Option<&Role> {
        self.roles.get(name)
    }

    /// Create a standard policy with default roles.
    pub fn standard() -> Self {
        Self::new()
            .with_role(Role::admin())
            .with_role(Role::operator())
            .with_role(Role::developer())
            .with_role(Role::readonly())
            .with_role(Role::service())
    }
}

impl Default for RbacPolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// POLICY EVALUATOR
// ============================================================================

/// Error type for RBAC evaluation.
#[derive(Debug, Clone)]
pub enum RbacError {
    /// Permission denied.
    PermissionDenied(String),
    /// Role not found.
    RoleNotFound(String),
    /// Invalid resource.
    InvalidResource(String),
}

impl fmt::Display for RbacError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            Self::RoleNotFound(msg) => write!(f, "Role not found: {}", msg),
            Self::InvalidResource(msg) => write!(f, "Invalid resource: {}", msg),
        }
    }
}

impl std::error::Error for RbacError {}

/// Result type for RBAC evaluation.
pub type RbacResult<T> = Result<T, RbacError>;

/// Subject for RBAC evaluation (who is making the request).
#[derive(Debug, Clone)]
pub struct Subject {
    /// Subject ID (user ID, service account, etc.).
    pub id: String,
    /// Assigned roles.
    pub roles: HashSet<String>,
    /// Direct permissions (in addition to role permissions).
    pub permissions: HashSet<String>,
    /// Tenant ID (for multi-tenant isolation).
    pub tenant_id: Option<String>,
}

impl Subject {
    /// Create a new subject.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            roles: HashSet::new(),
            permissions: HashSet::new(),
            tenant_id: None,
        }
    }

    /// Add a role.
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.roles.insert(role.into());
        self
    }

    /// Add roles.
    pub fn with_roles<I, S>(mut self, roles: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.roles.extend(roles.into_iter().map(Into::into));
        self
    }

    /// Add a direct permission.
    pub fn with_permission(mut self, permission: impl Into<String>) -> Self {
        self.permissions.insert(permission.into());
        self
    }

    /// Set tenant ID.
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Create from auth context.
    pub fn from_auth_context(ctx: &crate::auth::AuthContext) -> Self {
        let mut subject = Self::new(&ctx.identity.id)
            .with_roles(ctx.roles.iter().cloned())
            .with_permission(ctx.permissions.iter().cloned().collect::<Vec<_>>().join(","));

        if let Some(tenant) = &ctx.identity.tenant_id {
            subject = subject.with_tenant(tenant);
        }

        subject
    }
}

/// RBAC policy evaluator.
pub struct PolicyEvaluator {
    /// The policy to evaluate.
    policy: RbacPolicy,
    /// Evaluation cache.
    cache: RwLock<HashMap<(String, String), bool>>,
    /// Cache TTL.
    cache_ttl: std::time::Duration,
}

impl PolicyEvaluator {
    /// Create a new policy evaluator.
    pub fn new(policy: RbacPolicy) -> Self {
        Self {
            policy,
            cache: RwLock::new(HashMap::new()),
            cache_ttl: std::time::Duration::from_secs(60),
        }
    }

    /// Set cache TTL.
    pub fn with_cache_ttl(mut self, ttl: std::time::Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Clear the evaluation cache.
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    /// Check if subject has a permission.
    pub fn is_allowed(&self, subject: &Subject, permission: Permission) -> bool {
        // Check direct permissions
        if subject.permissions.contains(permission.as_str()) {
            return true;
        }

        // Check role permissions
        for role_name in &subject.roles {
            if let Some(role) = self.policy.get_role(role_name) {
                if role.has_permission(permission) {
                    return true;
                }
            }
        }

        !self.policy.default_deny
    }

    /// Check if subject can access a resource.
    pub fn can_access(&self, subject: &Subject, resource: &str, permission: Permission) -> RbacResult<()> {
        // Check basic permission first
        if !self.is_allowed(subject, permission) {
            return Err(RbacError::PermissionDenied(format!(
                "Subject {} lacks permission {} for resource {}",
                subject.id,
                permission,
                resource
            )));
        }

        // Check resource-specific rules
        for rule in &self.policy.resource_rules {
            if rule.matches(resource) {
                // Check deny rules first
                if rule.deny {
                    return Err(RbacError::PermissionDenied(format!(
                        "Resource {} is denied by policy",
                        resource
                    )));
                }

                // Check required permissions
                for required_perm in &rule.required_permissions {
                    if !self.is_allowed(subject, *required_perm) {
                        return Err(RbacError::PermissionDenied(format!(
                            "Resource {} requires permission {}",
                            resource, required_perm
                        )));
                    }
                }

                // Check required roles
                for required_role in &rule.required_roles {
                    if !subject.roles.contains(required_role) {
                        return Err(RbacError::PermissionDenied(format!(
                            "Resource {} requires role {}",
                            resource, required_role
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Check kernel access.
    pub fn can_access_kernel(&self, subject: &Subject, kernel_id: &KernelId, permission: Permission) -> RbacResult<()> {
        let resource = format!("kernel:{}", kernel_id.as_str());
        self.can_access(subject, &resource, permission)
    }

    /// Get all permissions for a subject.
    pub fn get_permissions(&self, subject: &Subject) -> HashSet<Permission> {
        let mut permissions = HashSet::new();

        // Collect from roles
        for role_name in &subject.roles {
            if let Some(role) = self.policy.get_role(role_name) {
                if role.is_superuser {
                    // Return all permissions for superuser
                    return [
                        Permission::KernelLaunch,
                        Permission::KernelTerminate,
                        Permission::KernelActivate,
                        Permission::KernelView,
                        Permission::KernelMigrate,
                        Permission::KernelCheckpoint,
                        Permission::KernelRestore,
                        Permission::MessageSend,
                        Permission::MessageReceive,
                        Permission::K2KSend,
                        Permission::K2KRegister,
                        Permission::PubSubPublish,
                        Permission::PubSubSubscribe,
                        Permission::MemoryAllocate,
                        Permission::MemoryView,
                        Permission::ResourceConfigure,
                        Permission::ConfigView,
                        Permission::ConfigModify,
                        Permission::SecurityView,
                        Permission::SecurityModify,
                        Permission::MetricsView,
                        Permission::MetricsExport,
                        Permission::AuditView,
                        Permission::AuditExport,
                        Permission::TracesView,
                        Permission::Admin,
                        Permission::UserManage,
                        Permission::TenantManage,
                    ]
                    .into_iter()
                    .collect();
                }
                permissions.extend(role.permissions.iter().cloned());
            }
        }

        // Add direct permissions
        for perm_str in &subject.permissions {
            if let Some(perm) = Permission::from_str(perm_str) {
                permissions.insert(perm);
            }
        }

        permissions
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_string() {
        assert_eq!(Permission::KernelLaunch.as_str(), "kernel:launch");
        assert_eq!(Permission::from_str("kernel:launch"), Some(Permission::KernelLaunch));
        assert_eq!(Permission::from_str("invalid"), None);
    }

    #[test]
    fn test_role_permissions() {
        let role = Role::new("test", "Test role")
            .with_permission(Permission::KernelLaunch)
            .with_permission(Permission::KernelView);

        assert!(role.has_permission(Permission::KernelLaunch));
        assert!(role.has_permission(Permission::KernelView));
        assert!(!role.has_permission(Permission::KernelTerminate));
    }

    #[test]
    fn test_superuser_role() {
        let admin = Role::admin();
        assert!(admin.is_superuser);
        assert!(admin.has_permission(Permission::KernelLaunch));
        assert!(admin.has_permission(Permission::Admin));
        assert!(admin.has_permission(Permission::SecurityModify));
    }

    #[test]
    fn test_predefined_roles() {
        let operator = Role::operator();
        assert!(operator.has_permission(Permission::KernelLaunch));
        assert!(operator.has_permission(Permission::KernelTerminate));
        assert!(!operator.has_permission(Permission::ConfigModify));

        let developer = Role::developer();
        assert!(developer.has_permission(Permission::KernelLaunch));
        assert!(!developer.has_permission(Permission::KernelTerminate));

        let readonly = Role::readonly();
        assert!(readonly.has_permission(Permission::KernelView));
        assert!(!readonly.has_permission(Permission::KernelLaunch));
    }

    #[test]
    fn test_resource_rule_matching() {
        let rule = ResourceRule::new("kernel:*");
        assert!(rule.matches("kernel:test"));
        assert!(rule.matches("kernel:production"));
        assert!(!rule.matches("message:test"));

        let exact_rule = ResourceRule::new("kernel:specific");
        assert!(exact_rule.matches("kernel:specific"));
        assert!(!exact_rule.matches("kernel:other"));
    }

    #[test]
    fn test_policy_evaluation() {
        let policy = RbacPolicy::standard();
        let evaluator = PolicyEvaluator::new(policy);

        let admin_subject = Subject::new("admin_user").with_role("admin");
        assert!(evaluator.is_allowed(&admin_subject, Permission::KernelLaunch));
        assert!(evaluator.is_allowed(&admin_subject, Permission::SecurityModify));

        let readonly_subject = Subject::new("readonly_user").with_role("readonly");
        assert!(evaluator.is_allowed(&readonly_subject, Permission::KernelView));
        assert!(!evaluator.is_allowed(&readonly_subject, Permission::KernelLaunch));
    }

    #[test]
    fn test_resource_access() {
        let policy = RbacPolicy::standard()
            .with_resource_rule(
                ResourceRule::new("kernel:production*")
                    .require_role("operator")
            );

        let evaluator = PolicyEvaluator::new(policy);

        let developer = Subject::new("dev").with_role("developer");
        let operator = Subject::new("ops").with_role("operator");

        // Developers can't access production kernels
        assert!(evaluator
            .can_access(&developer, "kernel:production-1", Permission::KernelView)
            .is_err());

        // Operators can
        assert!(evaluator
            .can_access(&operator, "kernel:production-1", Permission::KernelView)
            .is_ok());
    }

    #[test]
    fn test_get_permissions() {
        let policy = RbacPolicy::standard();
        let evaluator = PolicyEvaluator::new(policy);

        let operator = Subject::new("ops").with_role("operator");
        let permissions = evaluator.get_permissions(&operator);

        assert!(permissions.contains(&Permission::KernelLaunch));
        assert!(permissions.contains(&Permission::KernelTerminate));
        assert!(!permissions.contains(&Permission::ConfigModify));
    }
}
