//! Activity definitions for process mining.

use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

/// Activity identifier type.
pub type ActivityId = u32;

/// Activity definition with metadata.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct Activity {
    /// Unique activity identifier.
    pub id: ActivityId,
    /// Activity name.
    pub name: String,
    /// Activity category/type.
    pub category: ActivityCategory,
    /// Expected duration in milliseconds.
    pub expected_duration_ms: u32,
    /// Cost per execution.
    pub cost: f32,
    /// Required resources.
    pub required_resources: Vec<String>,
}

impl Activity {
    /// Create a new activity.
    pub fn new(id: ActivityId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            category: ActivityCategory::Task,
            expected_duration_ms: 0,
            cost: 0.0,
            required_resources: Vec::new(),
        }
    }

    /// Set the expected duration.
    pub fn with_duration(mut self, duration_ms: u32) -> Self {
        self.expected_duration_ms = duration_ms;
        self
    }

    /// Set the category.
    pub fn with_category(mut self, category: ActivityCategory) -> Self {
        self.category = category;
        self
    }
}

/// Activity category classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum ActivityCategory {
    /// Start event.
    Start = 0,
    /// End event.
    End = 1,
    /// Regular task/activity.
    #[default]
    Task = 2,
    /// Decision/gateway.
    Gateway = 3,
    /// Sub-process.
    SubProcess = 4,
    /// System/automated task.
    System = 5,
    /// Manual/human task.
    Manual = 6,
}

/// Activity registry for managing activity definitions.
#[derive(Debug, Clone, Default)]
pub struct ActivityRegistry {
    /// Activities by ID.
    activities: HashMap<ActivityId, Activity>,
    /// Name to ID mapping.
    name_to_id: HashMap<String, ActivityId>,
    /// Next available ID.
    next_id: ActivityId,
}

impl ActivityRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an activity.
    pub fn register(&mut self, activity: Activity) -> ActivityId {
        let id = activity.id;
        self.name_to_id.insert(activity.name.clone(), id);
        self.activities.insert(id, activity);
        self.next_id = self.next_id.max(id + 1);
        id
    }

    /// Create and register a new activity by name.
    pub fn get_or_create(&mut self, name: &str) -> ActivityId {
        if let Some(&id) = self.name_to_id.get(name) {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            let activity = Activity::new(id, name);
            self.register(activity)
        }
    }

    /// Get activity by ID.
    pub fn get(&self, id: ActivityId) -> Option<&Activity> {
        self.activities.get(&id)
    }

    /// Get activity by name.
    pub fn get_by_name(&self, name: &str) -> Option<&Activity> {
        self.name_to_id
            .get(name)
            .and_then(|id| self.activities.get(id))
    }

    /// Get activity name by ID.
    pub fn get_name(&self, id: ActivityId) -> Option<&str> {
        self.activities.get(&id).map(|a| a.name.as_str())
    }

    /// Get all activities.
    pub fn all(&self) -> impl Iterator<Item = &Activity> {
        self.activities.values()
    }

    /// Number of registered activities.
    pub fn len(&self) -> usize {
        self.activities.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.activities.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activity_registry() {
        let mut registry = ActivityRegistry::new();

        let id1 = registry.get_or_create("Register");
        let id2 = registry.get_or_create("Process");
        let id3 = registry.get_or_create("Register"); // Should return same ID

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(registry.len(), 2);
    }
}
