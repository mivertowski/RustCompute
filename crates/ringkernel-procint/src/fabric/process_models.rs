//! Reference process models for conformance checking.
//!
//! Provides pre-defined BPMN and DFG models for each sector.

use crate::models::{ActivityId, ProcessModel, ProcessModelType};

/// Reference model repository for sectors.
#[derive(Debug, Clone)]
pub struct ReferenceModelRepository {
    models: Vec<ProcessModel>,
}

impl Default for ReferenceModelRepository {
    fn default() -> Self {
        Self::new()
    }
}

impl ReferenceModelRepository {
    /// Create a new repository with default models.
    pub fn new() -> Self {
        let mut repo = Self { models: Vec::new() };
        repo.add_healthcare_models();
        repo.add_manufacturing_models();
        repo.add_finance_models();
        repo.add_it_models();
        repo
    }

    /// Get model by ID.
    pub fn get(&self, id: u32) -> Option<&ProcessModel> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Get models by sector name.
    pub fn get_by_sector(&self, sector: &str) -> Vec<&ProcessModel> {
        self.models
            .iter()
            .filter(|m| m.name.starts_with(sector))
            .collect()
    }

    /// Get all models.
    pub fn all(&self) -> &[ProcessModel] {
        &self.models
    }

    fn add_healthcare_models(&mut self) {
        // Healthcare DFG Model
        let mut model = ProcessModel::new(1, "Healthcare-DFG", ProcessModelType::DFG);
        model.start_activities = vec![1]; // Registration
        model.end_activities = vec![6]; // Discharge

        // Valid transitions (based on SectorTemplate::Healthcare)
        model.add_transition(1, 2); // Registration -> Triage
        model.add_transition(2, 3); // Triage -> Examination
        model.add_transition(3, 4); // Examination -> Diagnosis
        model.add_transition(4, 5); // Diagnosis -> Treatment
        model.add_transition(5, 6); // Treatment -> Discharge
                                    // Alternative paths
        model.add_transition(2, 4); // Triage -> Diagnosis (urgent)
        model.add_transition(5, 3); // Treatment -> Examination (follow-up)

        self.models.push(model);

        // Healthcare Petri Net Model
        let mut petri = ProcessModel::new(2, "Healthcare-PetriNet", ProcessModelType::PetriNet);
        petri.start_activities = vec![1];
        petri.end_activities = vec![6];
        petri.transitions = vec![(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)];
        self.models.push(petri);
    }

    fn add_manufacturing_models(&mut self) {
        let mut model = ProcessModel::new(10, "Manufacturing-DFG", ProcessModelType::DFG);
        model.start_activities = vec![11]; // Order
        model.end_activities = vec![16]; // Ship

        model.add_transition(11, 12); // Order -> Plan
        model.add_transition(12, 13); // Plan -> Produce
        model.add_transition(13, 14); // Produce -> QualityCheck
        model.add_transition(14, 15); // QualityCheck -> Package
        model.add_transition(15, 16); // Package -> Ship
                                      // Rework loop
        model.add_transition(14, 13); // QualityCheck -> Produce (rework)

        self.models.push(model);
    }

    fn add_finance_models(&mut self) {
        let mut model = ProcessModel::new(20, "Finance-DFG", ProcessModelType::DFG);
        model.start_activities = vec![21]; // Application
        model.end_activities = vec![25]; // Disbursement

        model.add_transition(21, 22); // Application -> Verification
        model.add_transition(22, 23); // Verification -> CreditCheck
        model.add_transition(23, 24); // CreditCheck -> Approval
        model.add_transition(24, 25); // Approval -> Disbursement
                                      // Review loop
        model.add_transition(23, 22); // CreditCheck -> Verification (review)

        self.models.push(model);
    }

    fn add_it_models(&mut self) {
        let mut model = ProcessModel::new(30, "IT-IncidentManagement-DFG", ProcessModelType::DFG);
        model.start_activities = vec![31]; // Report
        model.end_activities = vec![36]; // Close

        model.add_transition(31, 32); // Report -> Classify
        model.add_transition(32, 33); // Classify -> Assign
        model.add_transition(33, 34); // Assign -> Investigate
        model.add_transition(34, 35); // Investigate -> Resolve
        model.add_transition(35, 36); // Resolve -> Close
                                      // Escalation
        model.add_transition(34, 33); // Investigate -> Assign (reassign)
        model.add_transition(35, 34); // Resolve -> Investigate (reopen)

        self.models.push(model);
    }
}

/// Model builder for creating custom process models.
#[derive(Debug)]
pub struct ProcessModelBuilder {
    id: u32,
    name: String,
    model_type: ProcessModelType,
    start_activities: Vec<ActivityId>,
    end_activities: Vec<ActivityId>,
    transitions: Vec<(ActivityId, ActivityId)>,
}

impl ProcessModelBuilder {
    /// Create a new builder.
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            model_type: ProcessModelType::DFG,
            start_activities: Vec::new(),
            end_activities: Vec::new(),
            transitions: Vec::new(),
        }
    }

    /// Set model type.
    pub fn with_type(mut self, model_type: ProcessModelType) -> Self {
        self.model_type = model_type;
        self
    }

    /// Add start activity.
    pub fn with_start(mut self, activity: ActivityId) -> Self {
        self.start_activities.push(activity);
        self
    }

    /// Add end activity.
    pub fn with_end(mut self, activity: ActivityId) -> Self {
        self.end_activities.push(activity);
        self
    }

    /// Add transition.
    pub fn with_transition(mut self, source: ActivityId, target: ActivityId) -> Self {
        self.transitions.push((source, target));
        self
    }

    /// Build the process model.
    pub fn build(self) -> ProcessModel {
        ProcessModel {
            id: self.id,
            name: self.name,
            model_type: self.model_type,
            start_activities: self.start_activities,
            end_activities: self.end_activities,
            transitions: self.transitions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repository_creation() {
        let repo = ReferenceModelRepository::new();
        assert!(!repo.all().is_empty());
    }

    #[test]
    fn test_get_model() {
        let repo = ReferenceModelRepository::new();
        let model = repo.get(1);
        assert!(model.is_some());
        assert_eq!(model.unwrap().name, "Healthcare-DFG");
    }

    #[test]
    fn test_builder() {
        let model = ProcessModelBuilder::new(100, "Test")
            .with_type(ProcessModelType::DFG)
            .with_start(1)
            .with_end(5)
            .with_transition(1, 2)
            .with_transition(2, 5)
            .build();

        assert_eq!(model.id, 100);
        assert_eq!(model.start_activities, vec![1]);
        assert_eq!(model.end_activities, vec![5]);
        assert!(model.has_transition(1, 2));
    }
}
