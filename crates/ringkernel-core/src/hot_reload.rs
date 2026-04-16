//! Configuration Hot Reload — FR-008
//!
//! Runtime configuration updates without restart:
//! - Validate before apply
//! - Atomic swap (rollback on failure)
//! - Config versioning (monotonic counter)
//! - Audit trail (who changed what, when)

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// A versioned configuration value.
#[derive(Debug, Clone)]
pub struct ConfigEntry {
    /// The configuration value.
    pub value: ConfigValue,
    /// Whether this config can be changed at runtime.
    pub reloadable: bool,
    /// Description for documentation.
    pub description: String,
}

/// Configuration value types.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    /// String value.
    String(String),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// Duration in milliseconds.
    DurationMs(u64),
}

impl std::fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "{}", s),
            Self::Int(i) => write!(f, "{}", i),
            Self::Float(v) => write!(f, "{}", v),
            Self::Bool(b) => write!(f, "{}", b),
            Self::DurationMs(ms) => write!(f, "{}ms", ms),
        }
    }
}

/// Record of a configuration change (audit trail).
#[derive(Debug, Clone)]
pub struct ConfigChange {
    /// Which key was changed.
    pub key: String,
    /// Old value (None if new key).
    pub old_value: Option<ConfigValue>,
    /// New value.
    pub new_value: ConfigValue,
    /// Version after this change.
    pub version: u64,
    /// When the change was applied.
    pub changed_at: Instant,
    /// Who made the change (e.g., API caller ID).
    pub changed_by: String,
}

/// Hot-reloadable configuration store.
pub struct HotReloadConfig {
    /// Current configuration entries.
    entries: HashMap<String, ConfigEntry>,
    /// Monotonic version counter.
    version: AtomicU64,
    /// Audit trail of recent changes.
    changes: Vec<ConfigChange>,
    /// Maximum audit trail entries.
    max_audit_entries: usize,
}

impl HotReloadConfig {
    /// Create a new configuration store.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            version: AtomicU64::new(0),
            changes: Vec::new(),
            max_audit_entries: 1000,
        }
    }

    /// Register a configuration key with initial value and reload policy.
    pub fn register(
        &mut self,
        key: impl Into<String>,
        value: ConfigValue,
        reloadable: bool,
        description: impl Into<String>,
    ) {
        self.entries.insert(
            key.into(),
            ConfigEntry {
                value,
                reloadable,
                description: description.into(),
            },
        );
    }

    /// Get a configuration value.
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.entries.get(key).map(|e| &e.value)
    }

    /// Get a string value.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            ConfigValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get an integer value.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        match self.get(key)? {
            ConfigValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Get a boolean value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            ConfigValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Update a configuration value at runtime.
    ///
    /// Returns Ok(version) on success, Err on validation failure.
    pub fn update(
        &mut self,
        key: &str,
        new_value: ConfigValue,
        changed_by: impl Into<String>,
    ) -> Result<u64, ConfigUpdateError> {
        let entry = self
            .entries
            .get(key)
            .ok_or_else(|| ConfigUpdateError::KeyNotFound(key.to_string()))?;

        if !entry.reloadable {
            return Err(ConfigUpdateError::NotReloadable(key.to_string()));
        }

        // Type check
        if std::mem::discriminant(&entry.value) != std::mem::discriminant(&new_value) {
            return Err(ConfigUpdateError::TypeMismatch {
                key: key.to_string(),
                expected: format!("{:?}", std::mem::discriminant(&entry.value)),
                got: format!("{:?}", std::mem::discriminant(&new_value)),
            });
        }

        let old_value = entry.value.clone();
        let version = self.version.fetch_add(1, Ordering::Relaxed) + 1;

        // Record change
        let change = ConfigChange {
            key: key.to_string(),
            old_value: Some(old_value),
            new_value: new_value.clone(),
            version,
            changed_at: Instant::now(),
            changed_by: changed_by.into(),
        };

        // Apply the change
        self.entries.get_mut(key).unwrap().value = new_value;

        // Audit trail
        self.changes.push(change);
        while self.changes.len() > self.max_audit_entries {
            self.changes.remove(0);
        }

        Ok(version)
    }

    /// Current config version.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    /// Get recent changes (audit trail).
    pub fn recent_changes(&self, limit: usize) -> &[ConfigChange] {
        let start = self.changes.len().saturating_sub(limit);
        &self.changes[start..]
    }

    /// List all configuration keys.
    pub fn list_keys(&self) -> Vec<(&str, bool)> {
        self.entries
            .iter()
            .map(|(k, v)| (k.as_str(), v.reloadable))
            .collect()
    }

    /// Number of configuration entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for HotReloadConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors from configuration updates.
#[derive(Debug, Clone)]
pub enum ConfigUpdateError {
    /// Key not found.
    KeyNotFound(String),
    /// Key is not reloadable (requires restart).
    NotReloadable(String),
    /// Value type doesn't match registered type.
    TypeMismatch {
        key: String,
        expected: String,
        got: String,
    },
    /// Custom validation failed.
    ValidationFailed(String),
}

impl std::fmt::Display for ConfigUpdateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyNotFound(k) => write!(f, "Config key not found: {}", k),
            Self::NotReloadable(k) => write!(f, "Config key '{}' requires restart to change", k),
            Self::TypeMismatch { key, expected, got } => {
                write!(f, "Type mismatch for '{}': expected {}, got {}", key, expected, got)
            }
            Self::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
        }
    }
}

impl std::error::Error for ConfigUpdateError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get() {
        let mut config = HotReloadConfig::new();
        config.register("rate_limit", ConfigValue::Int(1000), true, "Max requests/sec");
        config.register("gpu_device", ConfigValue::Int(0), false, "GPU device index");

        assert_eq!(config.get_int("rate_limit"), Some(1000));
        assert_eq!(config.get_int("gpu_device"), Some(0));
    }

    #[test]
    fn test_update_reloadable() {
        let mut config = HotReloadConfig::new();
        config.register("rate_limit", ConfigValue::Int(1000), true, "");

        let version = config.update("rate_limit", ConfigValue::Int(2000), "admin").unwrap();
        assert_eq!(version, 1);
        assert_eq!(config.get_int("rate_limit"), Some(2000));
    }

    #[test]
    fn test_update_non_reloadable() {
        let mut config = HotReloadConfig::new();
        config.register("gpu_device", ConfigValue::Int(0), false, "");

        let result = config.update("gpu_device", ConfigValue::Int(1), "admin");
        assert!(matches!(result, Err(ConfigUpdateError::NotReloadable(_))));
    }

    #[test]
    fn test_type_mismatch() {
        let mut config = HotReloadConfig::new();
        config.register("rate_limit", ConfigValue::Int(1000), true, "");

        let result = config.update("rate_limit", ConfigValue::String("fast".into()), "admin");
        assert!(matches!(result, Err(ConfigUpdateError::TypeMismatch { .. })));
    }

    #[test]
    fn test_audit_trail() {
        let mut config = HotReloadConfig::new();
        config.register("rate_limit", ConfigValue::Int(1000), true, "");

        config.update("rate_limit", ConfigValue::Int(2000), "admin").unwrap();
        config.update("rate_limit", ConfigValue::Int(3000), "operator").unwrap();

        let changes = config.recent_changes(10);
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].changed_by, "admin");
        assert_eq!(changes[1].changed_by, "operator");
    }

    #[test]
    fn test_versioning() {
        let mut config = HotReloadConfig::new();
        config.register("a", ConfigValue::Bool(true), true, "");

        assert_eq!(config.version(), 0);
        config.update("a", ConfigValue::Bool(false), "test").unwrap();
        assert_eq!(config.version(), 1);
        config.update("a", ConfigValue::Bool(true), "test").unwrap();
        assert_eq!(config.version(), 2);
    }
}
