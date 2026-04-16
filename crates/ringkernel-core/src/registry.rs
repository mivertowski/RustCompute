//! Named Actor Registry — FR-002
//!
//! Global actor registry with symbolic names for service discovery.
//! Actors register by name; callers look up actors by name or wildcard pattern.
//!
//! # Usage
//!
//! ```ignore
//! use ringkernel_core::registry::ActorRegistry;
//! use ringkernel_core::runtime::KernelId;
//!
//! let mut registry = ActorRegistry::new();
//!
//! // Register actors by name
//! registry.register("isa_ontology", KernelId::new("kernel_1"));
//! registry.register("pcaob_rules", KernelId::new("kernel_2"));
//! registry.register("standards/isa/500", KernelId::new("kernel_3"));
//!
//! // Lookup by exact name
//! let actor = registry.lookup("isa_ontology"); // Some(KernelId("kernel_1"))
//!
//! // Lookup by wildcard
//! let actors = registry.lookup_pattern("standards/*"); // ["standards/isa/500"]
//! ```

use std::collections::HashMap;
use std::time::Instant;

use crate::runtime::KernelId;

/// A named actor registration entry.
#[derive(Debug, Clone)]
pub struct RegistryEntry {
    /// Symbolic name.
    pub name: String,
    /// The kernel ID this name resolves to.
    pub kernel_id: KernelId,
    /// When the registration was created.
    pub registered_at: Instant,
    /// Metadata tags.
    pub tags: HashMap<String, String>,
}

/// Event emitted when the registry changes.
#[derive(Debug, Clone)]
pub enum RegistryEvent {
    /// A new actor was registered.
    Registered {
        name: String,
        kernel_id: KernelId,
    },
    /// An actor was deregistered.
    Deregistered {
        name: String,
        kernel_id: KernelId,
    },
    /// An actor's registration was updated.
    Updated {
        name: String,
        old_kernel_id: KernelId,
        new_kernel_id: KernelId,
    },
}

/// Global actor registry with symbolic names.
///
/// Provides service discovery for the actor system: LLM tool calls,
/// external APIs, and inter-actor routing can find actors by name
/// instead of opaque kernel IDs.
pub struct ActorRegistry {
    /// Name → Entry mapping.
    entries: HashMap<String, RegistryEntry>,
    /// Reverse mapping: KernelId → names (one kernel can have multiple names).
    reverse: HashMap<KernelId, Vec<String>>,
    /// Watchers: pattern → callback channels.
    watchers: Vec<(String, Vec<RegistryEvent>)>,
}

impl ActorRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            reverse: HashMap::new(),
            watchers: Vec::new(),
        }
    }

    /// Register an actor by name.
    ///
    /// If the name is already registered, updates the mapping and
    /// emits an `Updated` event.
    pub fn register(&mut self, name: impl Into<String>, kernel_id: KernelId) -> RegistryEvent {
        let name = name.into();

        let event = if let Some(existing) = self.entries.get(&name) {
            let old_id = existing.kernel_id.clone();
            // Remove from old reverse mapping
            if let Some(names) = self.reverse.get_mut(&old_id) {
                names.retain(|n| n != &name);
            }
            RegistryEvent::Updated {
                name: name.clone(),
                old_kernel_id: old_id,
                new_kernel_id: kernel_id.clone(),
            }
        } else {
            RegistryEvent::Registered {
                name: name.clone(),
                kernel_id: kernel_id.clone(),
            }
        };

        // Add to reverse mapping
        self.reverse
            .entry(kernel_id.clone())
            .or_default()
            .push(name.clone());

        self.entries.insert(
            name.clone(),
            RegistryEntry {
                name,
                kernel_id,
                registered_at: Instant::now(),
                tags: HashMap::new(),
            },
        );

        self.notify_watchers(&event);
        event
    }

    /// Register with metadata tags.
    pub fn register_with_tags(
        &mut self,
        name: impl Into<String>,
        kernel_id: KernelId,
        tags: HashMap<String, String>,
    ) -> RegistryEvent {
        let event = self.register(name, kernel_id);
        if let RegistryEvent::Registered { ref name, .. }
        | RegistryEvent::Updated { ref name, .. } = event
        {
            if let Some(entry) = self.entries.get_mut(name) {
                entry.tags = tags;
            }
        }
        event
    }

    /// Deregister an actor by name.
    pub fn deregister(&mut self, name: &str) -> Option<RegistryEvent> {
        if let Some(entry) = self.entries.remove(name) {
            // Remove from reverse mapping
            if let Some(names) = self.reverse.get_mut(&entry.kernel_id) {
                names.retain(|n| n != name);
                if names.is_empty() {
                    self.reverse.remove(&entry.kernel_id);
                }
            }

            let event = RegistryEvent::Deregistered {
                name: name.to_string(),
                kernel_id: entry.kernel_id,
            };
            self.notify_watchers(&event);
            Some(event)
        } else {
            None
        }
    }

    /// Deregister all names associated with a kernel ID.
    pub fn deregister_kernel(&mut self, kernel_id: &KernelId) -> Vec<RegistryEvent> {
        let mut events = Vec::new();
        if let Some(names) = self.reverse.remove(kernel_id) {
            for name in names {
                if let Some(entry) = self.entries.remove(&name) {
                    events.push(RegistryEvent::Deregistered {
                        name,
                        kernel_id: entry.kernel_id,
                    });
                }
            }
        }
        for event in &events {
            self.notify_watchers(event);
        }
        events
    }

    /// Lookup an actor by exact name.
    pub fn lookup(&self, name: &str) -> Option<&KernelId> {
        self.entries.get(name).map(|e| &e.kernel_id)
    }

    /// Lookup the full registry entry by name.
    pub fn lookup_entry(&self, name: &str) -> Option<&RegistryEntry> {
        self.entries.get(name)
    }

    /// Lookup all names registered to a kernel ID.
    pub fn names_for(&self, kernel_id: &KernelId) -> Vec<&str> {
        self.reverse
            .get(kernel_id)
            .map(|names| names.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Lookup actors by wildcard pattern.
    ///
    /// Supports:
    /// - `*` matches any sequence of characters (excluding `/`)
    /// - `**` matches any sequence including `/`
    /// - `?` matches a single character
    ///
    /// Examples:
    /// - `"standards/*"` matches `"standards/isa"` but not `"standards/isa/500"`
    /// - `"standards/**"` matches `"standards/isa/500"`
    /// - `"isa_*"` matches `"isa_ontology"`, `"isa_rules"`
    pub fn lookup_pattern(&self, pattern: &str) -> Vec<(&str, &KernelId)> {
        self.entries
            .iter()
            .filter(|(name, _)| wildcard_match(pattern, name))
            .map(|(name, entry)| (name.as_str(), &entry.kernel_id))
            .collect()
    }

    /// List all registered names.
    pub fn list_names(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    /// Number of registered entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add a watcher for registry changes.
    ///
    /// Returns a watcher ID that can be used to retrieve events.
    pub fn watch(&mut self, pattern: impl Into<String>) -> usize {
        let id = self.watchers.len();
        self.watchers.push((pattern.into(), Vec::new()));
        id
    }

    /// Drain events for a watcher.
    pub fn drain_events(&mut self, watcher_id: usize) -> Vec<RegistryEvent> {
        if let Some((_, events)) = self.watchers.get_mut(watcher_id) {
            std::mem::take(events)
        } else {
            Vec::new()
        }
    }

    fn notify_watchers(&mut self, event: &RegistryEvent) {
        let name = match event {
            RegistryEvent::Registered { name, .. } => name,
            RegistryEvent::Deregistered { name, .. } => name,
            RegistryEvent::Updated { name, .. } => name,
        };

        for (pattern, events) in &mut self.watchers {
            if wildcard_match(pattern, name) {
                events.push(event.clone());
            }
        }
    }
}

impl Default for ActorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple wildcard pattern matching.
///
/// `*` matches any characters except `/`.
/// `**` matches any characters including `/`.
/// `?` matches a single character.
fn wildcard_match(pattern: &str, text: &str) -> bool {
    let mut p = pattern.chars().peekable();
    let mut t = text.chars().peekable();

    wildcard_match_recursive(&mut p.collect::<Vec<_>>(), &mut t.collect::<Vec<_>>(), 0, 0)
}

fn wildcard_match_recursive(pattern: &[char], text: &[char], pi: usize, ti: usize) -> bool {
    if pi == pattern.len() && ti == text.len() {
        return true;
    }
    if pi == pattern.len() {
        return false;
    }

    // Check for ** (matches everything including /)
    if pi + 1 < pattern.len() && pattern[pi] == '*' && pattern[pi + 1] == '*' {
        // Try matching ** against 0 or more characters
        for i in ti..=text.len() {
            if wildcard_match_recursive(pattern, text, pi + 2, i) {
                return true;
            }
        }
        return false;
    }

    // Check for * (matches everything except /)
    if pattern[pi] == '*' {
        for i in ti..=text.len() {
            if i > ti && i <= text.len() && text[i - 1] == '/' {
                // * doesn't cross /
                break;
            }
            if wildcard_match_recursive(pattern, text, pi + 1, i) {
                return true;
            }
        }
        return false;
    }

    // Check for ? (matches any single char)
    if pattern[pi] == '?' && ti < text.len() {
        return wildcard_match_recursive(pattern, text, pi + 1, ti + 1);
    }

    // Exact character match
    if ti < text.len() && pattern[pi] == text[ti] {
        return wildcard_match_recursive(pattern, text, pi + 1, ti + 1);
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_lookup() {
        let mut reg = ActorRegistry::new();

        reg.register("isa_ontology", KernelId::new("k1"));
        reg.register("pcaob_rules", KernelId::new("k2"));

        assert_eq!(reg.lookup("isa_ontology"), Some(&KernelId::new("k1")));
        assert_eq!(reg.lookup("pcaob_rules"), Some(&KernelId::new("k2")));
        assert_eq!(reg.lookup("nonexistent"), None);
    }

    #[test]
    fn test_deregister() {
        let mut reg = ActorRegistry::new();
        reg.register("actor_a", KernelId::new("k1"));
        assert_eq!(reg.len(), 1);

        reg.deregister("actor_a");
        assert_eq!(reg.len(), 0);
        assert_eq!(reg.lookup("actor_a"), None);
    }

    #[test]
    fn test_update_registration() {
        let mut reg = ActorRegistry::new();
        reg.register("my_actor", KernelId::new("k1"));
        let event = reg.register("my_actor", KernelId::new("k2"));

        assert!(matches!(event, RegistryEvent::Updated { .. }));
        assert_eq!(reg.lookup("my_actor"), Some(&KernelId::new("k2")));
    }

    #[test]
    fn test_reverse_lookup() {
        let mut reg = ActorRegistry::new();
        reg.register("name_a", KernelId::new("k1"));
        reg.register("name_b", KernelId::new("k1"));

        let names = reg.names_for(&KernelId::new("k1"));
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"name_a"));
        assert!(names.contains(&"name_b"));
    }

    #[test]
    fn test_wildcard_exact() {
        assert!(wildcard_match("hello", "hello"));
        assert!(!wildcard_match("hello", "world"));
    }

    #[test]
    fn test_wildcard_star() {
        assert!(wildcard_match("isa_*", "isa_ontology"));
        assert!(wildcard_match("isa_*", "isa_rules"));
        assert!(!wildcard_match("isa_*", "pcaob_rules"));
    }

    #[test]
    fn test_wildcard_star_no_slash() {
        assert!(wildcard_match("standards/*", "standards/isa"));
        assert!(!wildcard_match("standards/*", "standards/isa/500"));
    }

    #[test]
    fn test_wildcard_double_star() {
        assert!(wildcard_match("standards/**", "standards/isa/500"));
        assert!(wildcard_match("standards/**", "standards/isa"));
        assert!(!wildcard_match("standards/**", "other/isa"));
    }

    #[test]
    fn test_wildcard_question() {
        assert!(wildcard_match("actor_?", "actor_a"));
        assert!(wildcard_match("actor_?", "actor_b"));
        assert!(!wildcard_match("actor_?", "actor_ab"));
    }

    #[test]
    fn test_pattern_lookup() {
        let mut reg = ActorRegistry::new();
        reg.register("standards/isa/500", KernelId::new("k1"));
        reg.register("standards/isa/700", KernelId::new("k2"));
        reg.register("standards/pcaob/101", KernelId::new("k3"));
        reg.register("other/thing", KernelId::new("k4"));

        let isa = reg.lookup_pattern("standards/isa/*");
        assert_eq!(isa.len(), 2);

        let all_standards = reg.lookup_pattern("standards/**");
        assert_eq!(all_standards.len(), 3);
    }

    #[test]
    fn test_watcher() {
        let mut reg = ActorRegistry::new();
        let watcher = reg.watch("isa_*");

        reg.register("isa_ontology", KernelId::new("k1"));
        reg.register("pcaob_rules", KernelId::new("k2")); // Shouldn't trigger
        reg.register("isa_rules", KernelId::new("k3"));

        let events = reg.drain_events(watcher);
        assert_eq!(events.len(), 2); // Only isa_* matches
    }

    #[test]
    fn test_deregister_kernel() {
        let mut reg = ActorRegistry::new();
        reg.register("name_a", KernelId::new("k1"));
        reg.register("name_b", KernelId::new("k1"));
        reg.register("name_c", KernelId::new("k2"));

        let events = reg.deregister_kernel(&KernelId::new("k1"));
        assert_eq!(events.len(), 2);
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.lookup("name_c"), Some(&KernelId::new("k2")));
    }

    #[test]
    fn test_register_with_tags() {
        let mut reg = ActorRegistry::new();
        let mut tags = HashMap::new();
        tags.insert("domain".to_string(), "audit".to_string());
        tags.insert("version".to_string(), "2.0".to_string());

        reg.register_with_tags("isa_ontology", KernelId::new("k1"), tags);

        let entry = reg.lookup_entry("isa_ontology").unwrap();
        assert_eq!(entry.tags.get("domain").unwrap(), "audit");
    }
}
