//! Rule registry: version history, validation, and swap orchestration.
//!
//! The registry is the heart of hot rule reload. It owns the per-rule
//! version history, enforces monotonic versioning and dependency integrity,
//! and delegates the physical GPU swap to a pluggable [`RuleSwapBackend`].
//!
//! For v1.1 the production CUDA backend is not yet wired here — use
//! [`NoopSwapBackend`] in tests and inject the real backend from
//! `ringkernel-cuda` when available.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use super::{CompiledRule, ReloadReport, RuleError, RuleHandle, RuleStatus};

/// Pluggable signature verifier.
///
/// Implementations decide the signature format (Ed25519, RSA-PSS, etc.).
/// Returning `Ok(())` means the rule's `signature` field matched the PTX
/// bytes under the verifier's policy.
pub trait SignatureVerifier: Send + Sync {
    /// Verify the rule's signature.
    ///
    /// `rule.signature` is guaranteed to be `Some(_)` when this is called.
    fn verify(&self, rule: &CompiledRule) -> Result<(), RuleError>;
}

/// Pluggable GPU-side swap backend.
///
/// The registry is hardware-agnostic; concrete backends drive actual
/// CUDA module loading, actor quiescing, and atomic pointer swaps.
///
/// Default implementation is [`NoopSwapBackend`] — it returns success
/// instantly for every method, which is useful for unit tests and for
/// the v1.1 pre-hardware development phase.
pub trait RuleSwapBackend: Send + Sync {
    /// Pre-stage a rule artifact on the device (e.g. load the PTX into a
    /// CUDA module but do not yet make it the active actor). Called
    /// before `quiesce`.
    fn pre_stage(&self, rule: &CompiledRule) -> Result<(), RuleError>;

    /// Quiesce the old actor. Returns the count of messages drained.
    /// Called only for `reload` / `rollback`, not for `register`.
    fn quiesce(&self, rule_id: &str, version: u64) -> Result<u64, RuleError>;

    /// Perform the atomic pointer swap that activates `new_version`.
    fn swap(&self, rule_id: &str, new_version: u64) -> Result<(), RuleError>;

    /// Terminate the old actor after the grace period.
    fn terminate_old(&self, rule_id: &str, old_version: u64) -> Result<(), RuleError>;
}

/// No-op backend that succeeds instantly. Useful for tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopSwapBackend;

impl RuleSwapBackend for NoopSwapBackend {
    fn pre_stage(&self, _rule: &CompiledRule) -> Result<(), RuleError> {
        Ok(())
    }

    fn quiesce(&self, _rule_id: &str, _version: u64) -> Result<u64, RuleError> {
        Ok(0)
    }

    fn swap(&self, _rule_id: &str, _new_version: u64) -> Result<(), RuleError> {
        Ok(())
    }

    fn terminate_old(&self, _rule_id: &str, _old_version: u64) -> Result<(), RuleError> {
        Ok(())
    }
}

/// Per-rule version history and status tracking.
struct RuleVersionHistory {
    /// Versions ordered by the registry's insertion order (FIFO). The
    /// oldest entry is evicted first when we exceed `max_history`.
    versions: Vec<CompiledRule>,
    /// Currently active version (if any).
    active_version: Option<u64>,
    /// Lifecycle status per version.
    status_by_version: HashMap<u64, RuleStatus>,
    /// When each version was registered.
    registered_at: HashMap<u64, SystemTime>,
}

impl RuleVersionHistory {
    fn new() -> Self {
        Self {
            versions: Vec::new(),
            active_version: None,
            status_by_version: HashMap::new(),
            registered_at: HashMap::new(),
        }
    }

    /// Insert a new version. Does not touch status — the caller sets it.
    /// Evicts the oldest version if we exceed `max_history`.
    fn insert_version(&mut self, rule: CompiledRule, max_history: usize) {
        let version = rule.version;
        self.registered_at.insert(version, SystemTime::now());
        self.versions.push(rule);

        while self.versions.len() > max_history {
            let evicted = self.versions.remove(0);
            self.status_by_version.remove(&evicted.version);
            self.registered_at.remove(&evicted.version);
        }
    }

    fn get(&self, version: u64) -> Option<&CompiledRule> {
        self.versions.iter().find(|r| r.version == version)
    }

    fn active(&self) -> Option<&CompiledRule> {
        self.active_version.and_then(|v| self.get(v))
    }
}

/// Hot-swappable rule registry.
///
/// Thread-safe: all public operations take `&self` and synchronize
/// internally via `RwLock`.
pub struct RuleRegistry {
    rules: RwLock<HashMap<String, RuleVersionHistory>>,
    signature_verifier: Option<Arc<dyn SignatureVerifier>>,
    swap_backend: Arc<dyn RuleSwapBackend>,
    max_history_per_rule: usize,
}

impl RuleRegistry {
    /// Create a new registry with the given backend.
    ///
    /// `max_history_per_rule` determines how many prior versions we keep
    /// available for rollback. When the limit is exceeded, the oldest
    /// version is evicted (FIFO).
    pub fn new(max_history_per_rule: usize, swap_backend: Arc<dyn RuleSwapBackend>) -> Self {
        let max_history_per_rule = max_history_per_rule.max(1);
        Self {
            rules: RwLock::new(HashMap::new()),
            signature_verifier: None,
            swap_backend,
            max_history_per_rule,
        }
    }

    /// Attach a signature verifier. Rules without signatures are rejected
    /// once a verifier is set.
    pub fn with_verifier(mut self, verifier: Arc<dyn SignatureVerifier>) -> Self {
        self.signature_verifier = Some(verifier);
        self
    }

    /// Number of rules currently registered.
    pub fn rule_count(&self) -> usize {
        self.rules.read().len()
    }

    /// Configured history depth per rule.
    pub fn max_history(&self) -> usize {
        self.max_history_per_rule
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Register a rule for the first time (or register a new version of
    /// an existing rule without making it active).
    ///
    /// On success, the new version has status [`RuleStatus::Registered`]
    /// if the rule already had an active version; otherwise it is
    /// immediately activated and returned with [`RuleStatus::Active`].
    pub async fn register_rule(
        &self,
        rule: CompiledRule,
        device_compute_cap: &str,
    ) -> Result<RuleHandle, RuleError> {
        self.validate(&rule, device_compute_cap, /*is_reload=*/ false)?;

        // Pre-stage on the device before we mutate any state.
        self.swap_backend.pre_stage(&rule)?;

        let version = rule.version;
        let rule_id = rule.rule_id.clone();

        let mut rules = self.rules.write();
        let history = rules
            .entry(rule_id.clone())
            .or_insert_with(RuleVersionHistory::new);

        if history.get(version).is_some() {
            return Err(RuleError::DuplicateVersion { rule_id, version });
        }

        let status = if history.active_version.is_some() {
            RuleStatus::Registered
        } else {
            RuleStatus::Active
        };

        history.insert_version(rule, self.max_history_per_rule);
        history.status_by_version.insert(version, status);
        if matches!(status, RuleStatus::Active) {
            history.active_version = Some(version);
        }

        let registered_at = history
            .registered_at
            .get(&version)
            .copied()
            .unwrap_or_else(SystemTime::now);

        Ok(RuleHandle {
            rule_id,
            version,
            status,
            registered_at,
        })
    }

    /// Atomically hot-swap a new version of an existing rule.
    ///
    /// Preconditions:
    /// - rule is already registered
    /// - proposed version strictly greater than current active version
    /// - validation passes (signature, compute cap, deps)
    ///
    /// Postconditions:
    /// - new version has status [`RuleStatus::Active`]
    /// - old active version has status [`RuleStatus::Superseded(new)`]
    /// - [`ReloadReport`] returned with timing information
    pub async fn reload_rule(
        &self,
        rule: CompiledRule,
        device_compute_cap: &str,
    ) -> Result<ReloadReport, RuleError> {
        self.validate(&rule, device_compute_cap, /*is_reload=*/ true)?;

        let rule_id = rule.rule_id.clone();
        let new_version = rule.version;

        // Pre-stage on the device before touching any state.
        self.swap_backend.pre_stage(&rule)?;

        // Snapshot current active version so we can tell the backend who
        // to quiesce. We also detect "reload without active" as an
        // implicit register-and-activate (spec says reload of a fresh
        // rule is allowed).
        let old_version = {
            let rules = self.rules.read();
            rules.get(&rule_id).and_then(|h| h.active_version)
        };

        // Quiesce the old actor (if any) and measure the duration.
        let quiesce_start = Instant::now();
        let messages_in_flight = if let Some(old_v) = old_version {
            self.swap_backend.quiesce(&rule_id, old_v)?
        } else {
            0
        };
        let quiesce_duration = quiesce_start.elapsed();

        // Perform the atomic swap and measure it.
        let swap_start = Instant::now();
        self.swap_backend.swap(&rule_id, new_version)?;
        let swap_duration = swap_start.elapsed();

        // Apply the state changes under the lock.
        let mut rules = self.rules.write();
        let history = rules
            .entry(rule_id.clone())
            .or_insert_with(RuleVersionHistory::new);

        if history.get(new_version).is_some() {
            return Err(RuleError::DuplicateVersion {
                rule_id,
                version: new_version,
            });
        }

        history.insert_version(rule, self.max_history_per_rule);

        if let Some(old_v) = old_version {
            history
                .status_by_version
                .insert(old_v, RuleStatus::Superseded(new_version));
        }
        history
            .status_by_version
            .insert(new_version, RuleStatus::Active);
        history.active_version = Some(new_version);

        let rollback_available = old_version
            .and_then(|v| history.versions.iter().find(|r| r.version == v))
            .is_some();

        drop(rules);

        // Terminate the old actor after metadata flipped. A failure here
        // does not undo the swap — the new actor is already live.
        if let Some(old_v) = old_version {
            self.swap_backend.terminate_old(&rule_id, old_v)?;
        }

        Ok(ReloadReport {
            rule_id,
            from_version: old_version.unwrap_or(0),
            to_version: new_version,
            quiesce_duration,
            swap_duration,
            messages_in_flight_during_swap: messages_in_flight,
            rollback_available,
        })
    }

    /// Roll back to a specific earlier version kept in history.
    ///
    /// Unlike `reload_rule`, rollback marks the previously active version
    /// as [`RuleStatus::Rolledback`] (not `Superseded`) so auditors can
    /// tell the transition apart.
    pub async fn rollback_rule(
        &self,
        rule_id: &str,
        to_version: u64,
    ) -> Result<ReloadReport, RuleError> {
        // Phase 0: pre-flight under the read lock.
        let (current_active, target_rule) = {
            let rules = self.rules.read();
            let history = rules
                .get(rule_id)
                .ok_or_else(|| RuleError::NotFound(rule_id.to_string()))?;

            let active = history
                .active_version
                .ok_or(RuleError::NoActiveVersion)?;
            if active == to_version {
                // No-op rollback. Still emit a report.
                return Ok(ReloadReport {
                    rule_id: rule_id.to_string(),
                    from_version: active,
                    to_version,
                    quiesce_duration: Duration::from_nanos(0),
                    swap_duration: Duration::from_nanos(0),
                    messages_in_flight_during_swap: 0,
                    rollback_available: true,
                });
            }

            let target = history
                .get(to_version)
                .cloned()
                .ok_or(RuleError::RollbackTargetMissing(to_version))?;

            (active, target)
        };

        // Phase 1: pre-stage. The rule is still in history, but the device
        // may need to re-install it if it was torn down.
        self.swap_backend.pre_stage(&target_rule)?;

        let quiesce_start = Instant::now();
        let drained = self.swap_backend.quiesce(rule_id, current_active)?;
        let quiesce_duration = quiesce_start.elapsed();

        let swap_start = Instant::now();
        self.swap_backend.swap(rule_id, to_version)?;
        let swap_duration = swap_start.elapsed();

        // Apply state changes under the write lock.
        let mut rules = self.rules.write();
        let history = rules
            .get_mut(rule_id)
            .ok_or_else(|| RuleError::NotFound(rule_id.to_string()))?;

        history
            .status_by_version
            .insert(current_active, RuleStatus::Rolledback);
        history
            .status_by_version
            .insert(to_version, RuleStatus::Active);
        history.active_version = Some(to_version);

        drop(rules);

        self.swap_backend
            .terminate_old(rule_id, current_active)?;

        Ok(ReloadReport {
            rule_id: rule_id.to_string(),
            from_version: current_active,
            to_version,
            quiesce_duration,
            swap_duration,
            messages_in_flight_during_swap: drained,
            rollback_available: false,
        })
    }

    /// List the active handle for every registered rule.
    pub fn list_rules(&self) -> Vec<RuleHandle> {
        let rules = self.rules.read();
        let mut out = Vec::new();
        for (rule_id, history) in rules.iter() {
            if let Some(active) = history.active_version {
                if let Some(status) = history.status_by_version.get(&active).copied() {
                    let registered_at = history
                        .registered_at
                        .get(&active)
                        .copied()
                        .unwrap_or_else(SystemTime::now);
                    out.push(RuleHandle {
                        rule_id: rule_id.clone(),
                        version: active,
                        status,
                        registered_at,
                    });
                }
            }
        }
        out
    }

    /// Return a specific `(rule_id, version)` artifact if still in history.
    pub fn get_rule(&self, rule_id: &str, version: u64) -> Option<CompiledRule> {
        let rules = self.rules.read();
        rules.get(rule_id).and_then(|h| h.get(version).cloned())
    }

    /// Return the currently active rule artifact, if any.
    pub fn get_active(&self, rule_id: &str) -> Option<CompiledRule> {
        let rules = self.rules.read();
        rules
            .get(rule_id)
            .and_then(|h| h.active().cloned())
    }

    /// Full history for a rule (oldest first).
    pub fn history(&self, rule_id: &str) -> Vec<RuleHandle> {
        let rules = self.rules.read();
        let Some(history) = rules.get(rule_id) else {
            return Vec::new();
        };
        history
            .versions
            .iter()
            .map(|rule| RuleHandle {
                rule_id: rule.rule_id.clone(),
                version: rule.version,
                status: history
                    .status_by_version
                    .get(&rule.version)
                    .copied()
                    .unwrap_or(RuleStatus::Registered),
                registered_at: history
                    .registered_at
                    .get(&rule.version)
                    .copied()
                    .unwrap_or_else(SystemTime::now),
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    /// Shared validation for register/reload paths.
    fn validate(
        &self,
        rule: &CompiledRule,
        device_compute_cap: &str,
        is_reload: bool,
    ) -> Result<(), RuleError> {
        // Signature, if a verifier is configured.
        if let Some(verifier) = self.signature_verifier.as_ref() {
            if rule.signature.is_none() {
                return Err(RuleError::InvalidSignature);
            }
            verifier.verify(rule)?;
        }

        // Compute capability compatibility.
        if !compute_cap_compatible(&rule.compute_cap, device_compute_cap) {
            return Err(RuleError::ComputeCapMismatch {
                required: rule.compute_cap.clone(),
                available: device_compute_cap.to_string(),
            });
        }

        // Dependencies must already be registered.
        {
            let rules = self.rules.read();
            for dep in &rule.depends_on {
                if !rules
                    .get(dep)
                    .map(|h| h.active_version.is_some())
                    .unwrap_or(false)
                {
                    return Err(RuleError::MissingDependency(dep.clone()));
                }
            }

            // Version checks.
            if let Some(history) = rules.get(&rule.rule_id) {
                // Reject duplicate (rule_id, version) regardless of path —
                // check this BEFORE monotonicity so the caller gets the
                // more specific error when they retry the same version.
                if history.get(rule.version).is_some() {
                    return Err(RuleError::DuplicateVersion {
                        rule_id: rule.rule_id.clone(),
                        version: rule.version,
                    });
                }

                // Monotonic version: a new version must be strictly newer
                // than the currently active version.
                if let Some(active) = history.active_version {
                    if rule.version <= active {
                        return Err(RuleError::VersionDowngrade {
                            current: active,
                            proposed: rule.version,
                        });
                    }
                } else if is_reload {
                    // Reload of a rule that has no active version is allowed;
                    // this behaves like register + immediate activate.
                }
            }
        }

        Ok(())
    }
}

/// Whether `rule_cap` can run on a device reporting `device_cap`.
///
/// PTX compiled for `sm_X` runs on any device with compute capability
/// `>= sm_X`. We accept strings of the form `"sm_90"` / `"sm_86"` and
/// do a numeric compare; anything else is treated as an exact-match
/// requirement.
fn compute_cap_compatible(rule_cap: &str, device_cap: &str) -> bool {
    match (parse_sm(rule_cap), parse_sm(device_cap)) {
        (Some(req), Some(dev)) => dev >= req,
        _ => rule_cap == device_cap,
    }
}

fn parse_sm(s: &str) -> Option<u32> {
    let digits = s.strip_prefix("sm_").or_else(|| s.strip_prefix("SM_"))?;
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{ActorConfig, RuleMetadata};

    fn base_rule(rule_id: &str, version: u64) -> CompiledRule {
        CompiledRule {
            rule_id: rule_id.to_string(),
            version,
            ptx: vec![0xCA, 0xFE, 0xBA, 0xBE],
            compute_cap: "sm_90".to_string(),
            depends_on: Vec::new(),
            signature: None,
            actor_config: ActorConfig::default(),
            metadata: RuleMetadata::default(),
        }
    }

    fn registry() -> RuleRegistry {
        RuleRegistry::new(5, Arc::new(NoopSwapBackend))
    }

    #[tokio::test]
    async fn register_first_version_activates_immediately() {
        let reg = registry();
        let handle = reg
            .register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("register");
        assert_eq!(handle.version, 1);
        assert_eq!(handle.status, RuleStatus::Active);
        assert_eq!(reg.get_active("r1").map(|r| r.version), Some(1));
    }

    #[tokio::test]
    async fn register_duplicate_version_rejected() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("initial");
        let err = reg
            .register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect_err("duplicate should fail");
        assert!(matches!(err, RuleError::DuplicateVersion { .. }));
    }

    #[tokio::test]
    async fn register_additional_version_stays_registered_not_active() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("v1");
        let h2 = reg
            .register_rule(base_rule("r1", 2), "sm_90")
            .await
            .expect("v2");
        assert_eq!(h2.status, RuleStatus::Registered);
        assert_eq!(reg.get_active("r1").map(|r| r.version), Some(1));
    }

    #[tokio::test]
    async fn reload_higher_version_succeeds() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("v1");
        let report = reg
            .reload_rule(base_rule("r1", 2), "sm_90")
            .await
            .expect("reload");
        assert_eq!(report.from_version, 1);
        assert_eq!(report.to_version, 2);
        assert!(report.rollback_available);
        assert_eq!(reg.get_active("r1").map(|r| r.version), Some(2));
    }

    #[tokio::test]
    async fn reload_lower_version_rejected() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 5), "sm_90")
            .await
            .expect("v5");
        let err = reg
            .reload_rule(base_rule("r1", 4), "sm_90")
            .await
            .expect_err("downgrade should fail");
        assert!(matches!(
            err,
            RuleError::VersionDowngrade {
                current: 5,
                proposed: 4
            }
        ));
        assert_eq!(reg.get_active("r1").map(|r| r.version), Some(5));
    }

    #[tokio::test]
    async fn reload_equal_version_rejected() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 5), "sm_90")
            .await
            .expect("v5");
        let err = reg
            .reload_rule(base_rule("r1", 5), "sm_90")
            .await
            .expect_err("equal version rejected");
        // Equal version is both a duplicate and a downgrade; the registry
        // reports the more specific DuplicateVersion error first.
        assert!(matches!(
            err,
            RuleError::DuplicateVersion { .. } | RuleError::VersionDowngrade { .. }
        ));
    }

    #[tokio::test]
    async fn compute_cap_mismatch_rejected() {
        let reg = registry();
        let mut rule = base_rule("r1", 1);
        rule.compute_cap = "sm_100".to_string();
        let err = reg
            .register_rule(rule, "sm_90")
            .await
            .expect_err("cap mismatch");
        assert!(matches!(err, RuleError::ComputeCapMismatch { .. }));
    }

    #[tokio::test]
    async fn compute_cap_higher_device_is_compatible() {
        let reg = registry();
        // Rule needs sm_80, device is sm_90 — must succeed.
        let mut rule = base_rule("r1", 1);
        rule.compute_cap = "sm_80".to_string();
        let handle = reg.register_rule(rule, "sm_90").await.expect("compatible");
        assert_eq!(handle.status, RuleStatus::Active);
    }

    #[tokio::test]
    async fn missing_dependency_rejected() {
        let reg = registry();
        let mut rule = base_rule("r1", 1);
        rule.depends_on = vec!["not_present".to_string()];
        let err = reg
            .register_rule(rule, "sm_90")
            .await
            .expect_err("missing dep");
        assert!(matches!(err, RuleError::MissingDependency(_)));
    }

    #[tokio::test]
    async fn present_dependency_accepted() {
        let reg = registry();
        reg.register_rule(base_rule("dep", 1), "sm_90")
            .await
            .expect("dep");
        let mut rule = base_rule("main", 1);
        rule.depends_on = vec!["dep".to_string()];
        reg.register_rule(rule, "sm_90").await.expect("main");
    }

    struct RejectAllVerifier;
    impl SignatureVerifier for RejectAllVerifier {
        fn verify(&self, _rule: &CompiledRule) -> Result<(), RuleError> {
            Err(RuleError::InvalidSignature)
        }
    }

    struct AcceptAllVerifier;
    impl SignatureVerifier for AcceptAllVerifier {
        fn verify(&self, _rule: &CompiledRule) -> Result<(), RuleError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn signature_rejection() {
        let reg = RuleRegistry::new(5, Arc::new(NoopSwapBackend))
            .with_verifier(Arc::new(RejectAllVerifier));
        let mut rule = base_rule("r1", 1);
        rule.signature = Some(vec![1, 2, 3]);
        let err = reg
            .register_rule(rule, "sm_90")
            .await
            .expect_err("bad signature");
        assert!(matches!(err, RuleError::InvalidSignature));
    }

    #[tokio::test]
    async fn signature_required_when_verifier_set() {
        let reg = RuleRegistry::new(5, Arc::new(NoopSwapBackend))
            .with_verifier(Arc::new(AcceptAllVerifier));
        // Signature field is None.
        let err = reg
            .register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect_err("missing signature");
        assert!(matches!(err, RuleError::InvalidSignature));
    }

    #[tokio::test]
    async fn signature_acceptance() {
        let reg = RuleRegistry::new(5, Arc::new(NoopSwapBackend))
            .with_verifier(Arc::new(AcceptAllVerifier));
        let mut rule = base_rule("r1", 1);
        rule.signature = Some(vec![1]);
        let handle = reg
            .register_rule(rule, "sm_90")
            .await
            .expect("valid signature");
        assert_eq!(handle.status, RuleStatus::Active);
    }

    #[tokio::test]
    async fn rollback_to_prior_version() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("v1");
        reg.reload_rule(base_rule("r1", 2), "sm_90")
            .await
            .expect("v2");
        let report = reg.rollback_rule("r1", 1).await.expect("rollback");
        assert_eq!(report.from_version, 2);
        assert_eq!(report.to_version, 1);
        assert_eq!(reg.get_active("r1").map(|r| r.version), Some(1));

        // Prior "active" v2 is now Rolledback.
        let history = reg.history("r1");
        let v2 = history
            .iter()
            .find(|h| h.version == 2)
            .expect("v2 in history");
        assert_eq!(v2.status, RuleStatus::Rolledback);
        let v1 = history
            .iter()
            .find(|h| h.version == 1)
            .expect("v1 in history");
        assert_eq!(v1.status, RuleStatus::Active);
    }

    #[tokio::test]
    async fn rollback_to_nonexistent_version_rejected() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("v1");
        let err = reg
            .rollback_rule("r1", 99)
            .await
            .expect_err("no such version");
        assert!(matches!(err, RuleError::RollbackTargetMissing(99)));
    }

    #[tokio::test]
    async fn rollback_unknown_rule_rejected() {
        let reg = registry();
        let err = reg
            .rollback_rule("nope", 1)
            .await
            .expect_err("unknown rule");
        assert!(matches!(err, RuleError::NotFound(_)));
    }

    #[tokio::test]
    async fn rollback_to_active_is_noop() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("v1");
        let report = reg.rollback_rule("r1", 1).await.expect("noop rollback");
        assert_eq!(report.from_version, 1);
        assert_eq!(report.to_version, 1);
        assert_eq!(reg.get_active("r1").map(|r| r.version), Some(1));
    }

    #[tokio::test]
    async fn history_retention_evicts_oldest() {
        let reg = RuleRegistry::new(3, Arc::new(NoopSwapBackend));
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("v1");
        for v in 2..=5 {
            reg.reload_rule(base_rule("r1", v), "sm_90")
                .await
                .unwrap_or_else(|e| panic!("reload v{} failed: {:?}", v, e));
        }
        let history = reg.history("r1");
        assert_eq!(history.len(), 3, "retains most recent 3 versions");
        let versions: Vec<u64> = history.iter().map(|h| h.version).collect();
        // Oldest evicted first, so we keep the last 3.
        assert_eq!(versions, vec![3, 4, 5]);
    }

    #[tokio::test]
    async fn multiple_concurrent_rules() {
        let reg = registry();
        reg.register_rule(base_rule("a", 1), "sm_90")
            .await
            .expect("a");
        reg.register_rule(base_rule("b", 7), "sm_90")
            .await
            .expect("b");
        reg.register_rule(base_rule("c", 3), "sm_90")
            .await
            .expect("c");
        assert_eq!(reg.rule_count(), 3);
        assert_eq!(reg.get_active("a").map(|r| r.version), Some(1));
        assert_eq!(reg.get_active("b").map(|r| r.version), Some(7));
        assert_eq!(reg.get_active("c").map(|r| r.version), Some(3));
    }

    #[tokio::test]
    async fn list_rules_returns_active_only() {
        let reg = registry();
        reg.register_rule(base_rule("a", 1), "sm_90")
            .await
            .expect("a");
        reg.register_rule(base_rule("b", 2), "sm_90")
            .await
            .expect("b");
        let listed = reg.list_rules();
        assert_eq!(listed.len(), 2);
        for h in &listed {
            assert!(matches!(h.status, RuleStatus::Active));
        }
    }

    #[tokio::test]
    async fn get_rule_returns_specific_version() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("v1");
        reg.reload_rule(base_rule("r1", 2), "sm_90")
            .await
            .expect("v2");
        assert!(reg.get_rule("r1", 1).is_some());
        assert!(reg.get_rule("r1", 2).is_some());
        assert!(reg.get_rule("r1", 3).is_none());
    }

    #[tokio::test]
    async fn reload_report_fields_populated() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("v1");
        let report = reg
            .reload_rule(base_rule("r1", 2), "sm_90")
            .await
            .expect("v2");
        assert_eq!(report.rule_id, "r1");
        assert_eq!(report.from_version, 1);
        assert_eq!(report.to_version, 2);
        assert!(report.rollback_available);
        // Durations come from Instant::now() deltas; they are non-negative by
        // construction. We don't assert > 0 because Noop backend is instant.
        let _: Duration = report.quiesce_duration;
        let _: Duration = report.swap_duration;
    }

    #[tokio::test]
    async fn rollback_report_marks_no_further_rollback_available() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("v1");
        reg.reload_rule(base_rule("r1", 2), "sm_90")
            .await
            .expect("v2");
        let report = reg.rollback_rule("r1", 1).await.expect("rollback");
        assert!(!report.rollback_available);
    }

    struct CountingBackend {
        pre_stage: std::sync::atomic::AtomicU64,
        quiesce: std::sync::atomic::AtomicU64,
        swap: std::sync::atomic::AtomicU64,
        terminate: std::sync::atomic::AtomicU64,
        drained_per_call: u64,
    }

    impl CountingBackend {
        fn new(drained: u64) -> Self {
            Self {
                pre_stage: std::sync::atomic::AtomicU64::new(0),
                quiesce: std::sync::atomic::AtomicU64::new(0),
                swap: std::sync::atomic::AtomicU64::new(0),
                terminate: std::sync::atomic::AtomicU64::new(0),
                drained_per_call: drained,
            }
        }
    }

    impl RuleSwapBackend for CountingBackend {
        fn pre_stage(&self, _rule: &CompiledRule) -> Result<(), RuleError> {
            self.pre_stage
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
        fn quiesce(&self, _rule_id: &str, _version: u64) -> Result<u64, RuleError> {
            self.quiesce
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(self.drained_per_call)
        }
        fn swap(&self, _rule_id: &str, _new_version: u64) -> Result<(), RuleError> {
            self.swap
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
        fn terminate_old(&self, _rule_id: &str, _old_version: u64) -> Result<(), RuleError> {
            self.terminate
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
    }

    #[tokio::test]
    async fn backend_called_in_correct_order_for_reload() {
        let backend = Arc::new(CountingBackend::new(42));
        let reg = RuleRegistry::new(5, backend.clone());
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("v1");
        // After register: pre_stage 1, quiesce 0, swap 0, terminate 0.
        assert_eq!(
            backend
                .pre_stage
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            backend.quiesce.load(std::sync::atomic::Ordering::Relaxed),
            0
        );

        let report = reg
            .reload_rule(base_rule("r1", 2), "sm_90")
            .await
            .expect("v2");
        assert_eq!(report.messages_in_flight_during_swap, 42);
        assert_eq!(
            backend
                .pre_stage
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
        assert_eq!(
            backend.quiesce.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(backend.swap.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(
            backend
                .terminate
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    struct FailingSwapBackend;
    impl RuleSwapBackend for FailingSwapBackend {
        fn pre_stage(&self, _rule: &CompiledRule) -> Result<(), RuleError> {
            Err(RuleError::BackendError("pre_stage failed".into()))
        }
        fn quiesce(&self, _rule_id: &str, _version: u64) -> Result<u64, RuleError> {
            Ok(0)
        }
        fn swap(&self, _rule_id: &str, _new_version: u64) -> Result<(), RuleError> {
            Ok(())
        }
        fn terminate_old(&self, _rule_id: &str, _old_version: u64) -> Result<(), RuleError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn backend_pre_stage_failure_propagates_without_state_change() {
        let reg = RuleRegistry::new(5, Arc::new(FailingSwapBackend));
        let err = reg
            .register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect_err("pre_stage fails");
        assert!(matches!(err, RuleError::BackendError(_)));
        assert_eq!(reg.rule_count(), 0);
    }

    #[tokio::test]
    async fn history_lists_all_retained_versions_with_statuses() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("v1");
        reg.reload_rule(base_rule("r1", 2), "sm_90")
            .await
            .expect("v2");
        reg.reload_rule(base_rule("r1", 3), "sm_90")
            .await
            .expect("v3");
        let history = reg.history("r1");
        assert_eq!(history.len(), 3);
        // v1: Superseded(2), v2: Superseded(3), v3: Active
        let v1 = history
            .iter()
            .find(|h| h.version == 1)
            .expect("v1 in history");
        let v2 = history
            .iter()
            .find(|h| h.version == 2)
            .expect("v2 in history");
        let v3 = history
            .iter()
            .find(|h| h.version == 3)
            .expect("v3 in history");
        assert_eq!(v1.status, RuleStatus::Superseded(2));
        assert_eq!(v2.status, RuleStatus::Superseded(3));
        assert_eq!(v3.status, RuleStatus::Active);
    }

    #[tokio::test]
    async fn reload_rule_with_no_existing_rule_activates_it() {
        let reg = registry();
        // reload_rule on a fresh rule_id is allowed and activates it.
        let report = reg
            .reload_rule(base_rule("fresh", 1), "sm_90")
            .await
            .expect("initial reload");
        assert_eq!(report.from_version, 0);
        assert_eq!(report.to_version, 1);
        assert_eq!(reg.get_active("fresh").map(|r| r.version), Some(1));
    }

    #[tokio::test]
    async fn get_active_none_when_no_rule() {
        let reg = registry();
        assert!(reg.get_active("missing").is_none());
    }

    #[tokio::test]
    async fn history_empty_for_unknown_rule() {
        let reg = registry();
        assert!(reg.history("unknown").is_empty());
    }

    #[test]
    fn compute_cap_compatibility_matrix() {
        assert!(compute_cap_compatible("sm_80", "sm_90"));
        assert!(compute_cap_compatible("sm_90", "sm_90"));
        assert!(!compute_cap_compatible("sm_90", "sm_80"));
        assert!(!compute_cap_compatible("sm_90", "sm_86"));
        // Non-standard strings fall back to exact-match.
        assert!(compute_cap_compatible("custom", "custom"));
        assert!(!compute_cap_compatible("custom", "other"));
    }

    #[test]
    fn default_history_is_at_least_one() {
        let reg = RuleRegistry::new(0, Arc::new(NoopSwapBackend));
        assert!(reg.max_history() >= 1);
    }

    #[tokio::test]
    async fn duplicate_version_rejected_on_reload() {
        let reg = registry();
        reg.register_rule(base_rule("r1", 1), "sm_90")
            .await
            .expect("v1");
        // Pre-register v2 so it is in history but not active.
        reg.register_rule(base_rule("r1", 2), "sm_90")
            .await
            .expect("v2");
        let err = reg
            .reload_rule(base_rule("r1", 2), "sm_90")
            .await
            .expect_err("duplicate");
        assert!(matches!(err, RuleError::DuplicateVersion { .. }));
    }
}
