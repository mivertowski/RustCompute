//! Industry sector templates for process generation.
//!
//! Defines process structures for different domains.

use crate::models::{Activity, ActivityCategory, ActivityId, ActivityRegistry};

/// Industry sector template for process generation.
#[derive(Debug, Clone)]
pub enum SectorTemplate {
    /// Healthcare: Patient journey through hospital.
    Healthcare(HealthcareConfig),
    /// Manufacturing: Production workflow.
    Manufacturing(ManufacturingConfig),
    /// Finance: Loan approval process.
    Finance(FinanceConfig),
    /// IT: Incident management (ITIL).
    IncidentManagement(IncidentConfig),
}

impl PartialEq for SectorTemplate {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl Eq for SectorTemplate {}

impl Default for SectorTemplate {
    fn default() -> Self {
        Self::Healthcare(HealthcareConfig::default())
    }
}

impl SectorTemplate {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            SectorTemplate::Healthcare(_) => "Healthcare",
            SectorTemplate::Manufacturing(_) => "Manufacturing",
            SectorTemplate::Finance(_) => "Finance",
            SectorTemplate::IncidentManagement(_) => "IT Incident Management",
        }
    }

    /// Get short code.
    pub fn code(&self) -> &'static str {
        match self {
            SectorTemplate::Healthcare(_) => "HC",
            SectorTemplate::Manufacturing(_) => "MF",
            SectorTemplate::Finance(_) => "FN",
            SectorTemplate::IncidentManagement(_) => "IT",
        }
    }

    /// Get activity definitions for this sector.
    pub fn activities(&self) -> Vec<ActivityDef> {
        match self {
            SectorTemplate::Healthcare(cfg) => cfg.activities(),
            SectorTemplate::Manufacturing(cfg) => cfg.activities(),
            SectorTemplate::Finance(cfg) => cfg.activities(),
            SectorTemplate::IncidentManagement(cfg) => cfg.activities(),
        }
    }

    /// Get valid transitions for this sector (source_name, target_name, probability).
    pub fn transitions(&self) -> Vec<TransitionDef> {
        match self {
            SectorTemplate::Healthcare(cfg) => cfg.transitions(),
            SectorTemplate::Manufacturing(cfg) => cfg.transitions(),
            SectorTemplate::Finance(cfg) => cfg.transitions(),
            SectorTemplate::IncidentManagement(cfg) => cfg.transitions(),
        }
    }

    /// Get parallel activity definitions (fork/join patterns).
    pub fn parallel_activities(&self) -> Vec<ParallelActivityDef> {
        match self {
            SectorTemplate::Healthcare(cfg) => cfg.parallel_activities(),
            SectorTemplate::Manufacturing(_) => vec![], // Sequential process
            SectorTemplate::Finance(cfg) => cfg.parallel_activities(),
            SectorTemplate::IncidentManagement(_) => vec![], // Sequential process
        }
    }

    /// Get start activity names.
    pub fn start_activities(&self) -> Vec<&'static str> {
        match self {
            SectorTemplate::Healthcare(_) => vec!["Registration"],
            SectorTemplate::Manufacturing(_) => vec!["Order Received"],
            SectorTemplate::Finance(_) => vec!["Application Submitted"],
            SectorTemplate::IncidentManagement(_) => vec!["Incident Reported"],
        }
    }

    /// Get end activity names.
    pub fn end_activities(&self) -> Vec<&'static str> {
        match self {
            SectorTemplate::Healthcare(_) => vec!["Discharge", "Transfer"],
            SectorTemplate::Manufacturing(_) => vec!["Shipped"],
            SectorTemplate::Finance(_) => vec!["Loan Disbursed", "Application Rejected"],
            SectorTemplate::IncidentManagement(_) => vec!["Incident Closed"],
        }
    }

    /// Build activity registry for this sector.
    pub fn build_registry(&self) -> ActivityRegistry {
        let mut registry = ActivityRegistry::new();
        for def in self.activities() {
            let mut activity = Activity::new(def.id, def.name);
            activity.category = def.category;
            activity.expected_duration_ms = def.avg_duration_ms;
            registry.register(activity);
        }
        registry
    }
}

/// Activity definition.
#[derive(Debug, Clone)]
pub struct ActivityDef {
    /// Activity ID.
    pub id: ActivityId,
    /// Activity name.
    pub name: &'static str,
    /// Category.
    pub category: ActivityCategory,
    /// Average duration in ms.
    pub avg_duration_ms: u32,
    /// Duration variance (0.0 - 1.0).
    pub duration_variance: f32,
}

/// Transition definition.
#[derive(Debug, Clone)]
pub struct TransitionDef {
    /// Source activity name.
    pub source: &'static str,
    /// Target activity name.
    pub target: &'static str,
    /// Transition probability (0.0 - 1.0).
    pub probability: f32,
    /// Average transition time in ms.
    pub avg_transition_ms: u32,
}

/// Parallel activities that can run concurrently (fork/join pattern).
#[derive(Debug, Clone)]
pub struct ParallelActivityDef {
    /// Source activity that triggers the fork.
    pub fork_from: &'static str,
    /// Activities that run in parallel.
    pub parallel_activities: Vec<&'static str>,
    /// Activity to join to after all parallel activities complete.
    pub join_to: &'static str,
    /// Probability of parallel execution (vs sequential).
    pub probability: f32,
}

// ============================================================================
// Healthcare Configuration
// ============================================================================

/// Healthcare sector configuration.
#[derive(Debug, Clone)]
pub struct HealthcareConfig {
    /// Departments.
    pub departments: Vec<String>,
    /// Average patient stay in hours.
    pub avg_stay_hours: f32,
    /// Emergency bypass ratio.
    pub emergency_ratio: f32,
    /// Readmission rate.
    pub readmission_rate: f32,
}

impl Default for HealthcareConfig {
    fn default() -> Self {
        Self {
            departments: vec![
                "Emergency".into(),
                "Cardiology".into(),
                "Orthopedics".into(),
                "General".into(),
            ],
            avg_stay_hours: 48.0,
            emergency_ratio: 0.15,
            readmission_rate: 0.08,
        }
    }
}

impl HealthcareConfig {
    fn activities(&self) -> Vec<ActivityDef> {
        vec![
            ActivityDef {
                id: 1,
                name: "Registration",
                category: ActivityCategory::Start,
                avg_duration_ms: 300_000,
                duration_variance: 0.3,
            },
            ActivityDef {
                id: 2,
                name: "Triage",
                category: ActivityCategory::Task,
                avg_duration_ms: 600_000,
                duration_variance: 0.4,
            },
            ActivityDef {
                id: 3,
                name: "Examination",
                category: ActivityCategory::Task,
                avg_duration_ms: 1_800_000,
                duration_variance: 0.5,
            },
            ActivityDef {
                id: 4,
                name: "Lab Tests",
                category: ActivityCategory::Task,
                avg_duration_ms: 3_600_000,
                duration_variance: 0.6,
            },
            ActivityDef {
                id: 5,
                name: "Imaging",
                category: ActivityCategory::Task,
                avg_duration_ms: 2_400_000,
                duration_variance: 0.4,
            },
            ActivityDef {
                id: 6,
                name: "Diagnosis",
                category: ActivityCategory::Gateway,
                avg_duration_ms: 1_200_000,
                duration_variance: 0.3,
            },
            ActivityDef {
                id: 7,
                name: "Treatment",
                category: ActivityCategory::Task,
                avg_duration_ms: 7_200_000,
                duration_variance: 0.7,
            },
            ActivityDef {
                id: 8,
                name: "Surgery",
                category: ActivityCategory::Task,
                avg_duration_ms: 14_400_000,
                duration_variance: 0.5,
            },
            ActivityDef {
                id: 9,
                name: "Recovery",
                category: ActivityCategory::Task,
                avg_duration_ms: 86_400_000,
                duration_variance: 0.8,
            },
            ActivityDef {
                id: 10,
                name: "Discharge",
                category: ActivityCategory::End,
                avg_duration_ms: 600_000,
                duration_variance: 0.3,
            },
            ActivityDef {
                id: 11,
                name: "Transfer",
                category: ActivityCategory::End,
                avg_duration_ms: 1_200_000,
                duration_variance: 0.4,
            },
        ]
    }

    /// Get parallel activity definitions for healthcare.
    /// Lab Tests and Imaging can run concurrently after Examination.
    fn parallel_activities(&self) -> Vec<ParallelActivityDef> {
        vec![ParallelActivityDef {
            fork_from: "Examination",
            parallel_activities: vec!["Lab Tests", "Imaging"],
            join_to: "Diagnosis",
            probability: 0.4, // 40% of cases do both tests in parallel
        }]
    }

    fn transitions(&self) -> Vec<TransitionDef> {
        vec![
            TransitionDef {
                source: "Registration",
                target: "Triage",
                probability: 0.85,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Registration",
                target: "Examination",
                probability: 0.15,
                avg_transition_ms: 30_000,
            }, // Emergency bypass
            TransitionDef {
                source: "Triage",
                target: "Examination",
                probability: 1.0,
                avg_transition_ms: 120_000,
            },
            TransitionDef {
                source: "Examination",
                target: "Lab Tests",
                probability: 0.6,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Examination",
                target: "Imaging",
                probability: 0.3,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Examination",
                target: "Diagnosis",
                probability: 0.1,
                avg_transition_ms: 30_000,
            },
            TransitionDef {
                source: "Lab Tests",
                target: "Diagnosis",
                probability: 0.7,
                avg_transition_ms: 120_000,
            },
            TransitionDef {
                source: "Lab Tests",
                target: "Imaging",
                probability: 0.3,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Imaging",
                target: "Diagnosis",
                probability: 1.0,
                avg_transition_ms: 120_000,
            },
            TransitionDef {
                source: "Diagnosis",
                target: "Treatment",
                probability: 0.7,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Diagnosis",
                target: "Surgery",
                probability: 0.2,
                avg_transition_ms: 120_000,
            },
            TransitionDef {
                source: "Diagnosis",
                target: "Discharge",
                probability: 0.1,
                avg_transition_ms: 300_000,
            },
            TransitionDef {
                source: "Treatment",
                target: "Recovery",
                probability: 0.3,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Treatment",
                target: "Discharge",
                probability: 0.6,
                avg_transition_ms: 300_000,
            },
            TransitionDef {
                source: "Treatment",
                target: "Treatment",
                probability: 0.1,
                avg_transition_ms: 3_600_000,
            }, // Rework
            TransitionDef {
                source: "Surgery",
                target: "Recovery",
                probability: 1.0,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Recovery",
                target: "Discharge",
                probability: 0.85,
                avg_transition_ms: 600_000,
            },
            TransitionDef {
                source: "Recovery",
                target: "Transfer",
                probability: 0.1,
                avg_transition_ms: 600_000,
            },
            TransitionDef {
                source: "Recovery",
                target: "Surgery",
                probability: 0.05,
                avg_transition_ms: 1_800_000,
            }, // Complication
        ]
    }
}

// ============================================================================
// Manufacturing Configuration
// ============================================================================

/// Manufacturing sector configuration.
#[derive(Debug, Clone)]
pub struct ManufacturingConfig {
    /// Number of production lines.
    pub production_lines: u32,
    /// Average batch size.
    pub batch_size: u32,
    /// Defect rate (0.0 - 1.0).
    pub defect_rate: f32,
    /// Rework rate.
    pub rework_rate: f32,
}

impl Default for ManufacturingConfig {
    fn default() -> Self {
        Self {
            production_lines: 4,
            batch_size: 100,
            defect_rate: 0.03,
            rework_rate: 0.08,
        }
    }
}

impl ManufacturingConfig {
    fn activities(&self) -> Vec<ActivityDef> {
        vec![
            ActivityDef {
                id: 1,
                name: "Order Received",
                category: ActivityCategory::Start,
                avg_duration_ms: 60_000,
                duration_variance: 0.2,
            },
            ActivityDef {
                id: 2,
                name: "Planning",
                category: ActivityCategory::Task,
                avg_duration_ms: 1_800_000,
                duration_variance: 0.4,
            },
            ActivityDef {
                id: 3,
                name: "Material Prep",
                category: ActivityCategory::Task,
                avg_duration_ms: 3_600_000,
                duration_variance: 0.3,
            },
            ActivityDef {
                id: 4,
                name: "Production",
                category: ActivityCategory::Task,
                avg_duration_ms: 14_400_000,
                duration_variance: 0.5,
            },
            ActivityDef {
                id: 5,
                name: "Quality Check",
                category: ActivityCategory::Gateway,
                avg_duration_ms: 1_800_000,
                duration_variance: 0.4,
            },
            ActivityDef {
                id: 6,
                name: "Rework",
                category: ActivityCategory::Task,
                avg_duration_ms: 7_200_000,
                duration_variance: 0.6,
            },
            ActivityDef {
                id: 7,
                name: "Packaging",
                category: ActivityCategory::Task,
                avg_duration_ms: 1_200_000,
                duration_variance: 0.3,
            },
            ActivityDef {
                id: 8,
                name: "Shipped",
                category: ActivityCategory::End,
                avg_duration_ms: 600_000,
                duration_variance: 0.2,
            },
        ]
    }

    fn transitions(&self) -> Vec<TransitionDef> {
        vec![
            TransitionDef {
                source: "Order Received",
                target: "Planning",
                probability: 1.0,
                avg_transition_ms: 30_000,
            },
            TransitionDef {
                source: "Planning",
                target: "Material Prep",
                probability: 1.0,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Material Prep",
                target: "Production",
                probability: 1.0,
                avg_transition_ms: 120_000,
            },
            TransitionDef {
                source: "Production",
                target: "Quality Check",
                probability: 1.0,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Quality Check",
                target: "Packaging",
                probability: 0.92,
                avg_transition_ms: 30_000,
            },
            TransitionDef {
                source: "Quality Check",
                target: "Rework",
                probability: 0.08,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Rework",
                target: "Quality Check",
                probability: 1.0,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Packaging",
                target: "Shipped",
                probability: 1.0,
                avg_transition_ms: 120_000,
            },
        ]
    }
}

// ============================================================================
// Finance Configuration
// ============================================================================

/// Finance sector configuration.
#[derive(Debug, Clone)]
pub struct FinanceConfig {
    /// Loan types.
    pub loan_types: Vec<String>,
    /// Average approval days.
    pub avg_approval_days: f32,
    /// Rejection rate.
    pub rejection_rate: f32,
    /// Additional review rate.
    pub additional_review_rate: f32,
}

impl Default for FinanceConfig {
    fn default() -> Self {
        Self {
            loan_types: vec![
                "Personal".into(),
                "Mortgage".into(),
                "Auto".into(),
                "Business".into(),
            ],
            avg_approval_days: 5.0,
            rejection_rate: 0.15,
            additional_review_rate: 0.20,
        }
    }
}

impl FinanceConfig {
    fn activities(&self) -> Vec<ActivityDef> {
        vec![
            ActivityDef {
                id: 1,
                name: "Application Submitted",
                category: ActivityCategory::Start,
                avg_duration_ms: 300_000,
                duration_variance: 0.2,
            },
            ActivityDef {
                id: 2,
                name: "Document Verification",
                category: ActivityCategory::Task,
                avg_duration_ms: 7_200_000,
                duration_variance: 0.5,
            },
            ActivityDef {
                id: 3,
                name: "Credit Check",
                category: ActivityCategory::Task,
                avg_duration_ms: 3_600_000,
                duration_variance: 0.3,
            },
            ActivityDef {
                id: 4,
                name: "Income Verification",
                category: ActivityCategory::Task,
                avg_duration_ms: 14_400_000,
                duration_variance: 0.6,
            },
            ActivityDef {
                id: 5,
                name: "Risk Assessment",
                category: ActivityCategory::Gateway,
                avg_duration_ms: 7_200_000,
                duration_variance: 0.4,
            },
            ActivityDef {
                id: 6,
                name: "Additional Review",
                category: ActivityCategory::Task,
                avg_duration_ms: 86_400_000,
                duration_variance: 0.7,
            },
            ActivityDef {
                id: 7,
                name: "Final Approval",
                category: ActivityCategory::Gateway,
                avg_duration_ms: 3_600_000,
                duration_variance: 0.3,
            },
            ActivityDef {
                id: 8,
                name: "Loan Disbursed",
                category: ActivityCategory::End,
                avg_duration_ms: 1_800_000,
                duration_variance: 0.2,
            },
            ActivityDef {
                id: 9,
                name: "Application Rejected",
                category: ActivityCategory::End,
                avg_duration_ms: 600_000,
                duration_variance: 0.2,
            },
        ]
    }

    /// Get parallel activity definitions for finance.
    /// Credit Check and Income Verification can run concurrently.
    fn parallel_activities(&self) -> Vec<ParallelActivityDef> {
        vec![ParallelActivityDef {
            fork_from: "Document Verification",
            parallel_activities: vec!["Credit Check", "Income Verification"],
            join_to: "Risk Assessment",
            probability: 0.5, // 50% of cases run checks in parallel
        }]
    }

    fn transitions(&self) -> Vec<TransitionDef> {
        vec![
            TransitionDef {
                source: "Application Submitted",
                target: "Document Verification",
                probability: 1.0,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Document Verification",
                target: "Credit Check",
                probability: 0.9,
                avg_transition_ms: 30_000,
            },
            TransitionDef {
                source: "Document Verification",
                target: "Document Verification",
                probability: 0.1,
                avg_transition_ms: 86_400_000,
            }, // Missing docs
            TransitionDef {
                source: "Credit Check",
                target: "Income Verification",
                probability: 0.85,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Credit Check",
                target: "Application Rejected",
                probability: 0.15,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Income Verification",
                target: "Risk Assessment",
                probability: 1.0,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Risk Assessment",
                target: "Final Approval",
                probability: 0.75,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Risk Assessment",
                target: "Additional Review",
                probability: 0.20,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Risk Assessment",
                target: "Application Rejected",
                probability: 0.05,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Additional Review",
                target: "Final Approval",
                probability: 0.7,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Additional Review",
                target: "Application Rejected",
                probability: 0.3,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Final Approval",
                target: "Loan Disbursed",
                probability: 1.0,
                avg_transition_ms: 120_000,
            },
        ]
    }
}

// ============================================================================
// IT Incident Management Configuration
// ============================================================================

/// IT Incident Management configuration.
#[derive(Debug, Clone)]
pub struct IncidentConfig {
    /// Priority levels.
    pub priority_levels: u32,
    /// Average resolution hours.
    pub avg_resolution_hours: f32,
    /// Escalation rate.
    pub escalation_rate: f32,
    /// Reopen rate.
    pub reopen_rate: f32,
}

impl Default for IncidentConfig {
    fn default() -> Self {
        Self {
            priority_levels: 4,
            avg_resolution_hours: 8.0,
            escalation_rate: 0.12,
            reopen_rate: 0.05,
        }
    }
}

impl IncidentConfig {
    fn activities(&self) -> Vec<ActivityDef> {
        vec![
            ActivityDef {
                id: 1,
                name: "Incident Reported",
                category: ActivityCategory::Start,
                avg_duration_ms: 300_000,
                duration_variance: 0.3,
            },
            ActivityDef {
                id: 2,
                name: "Classification",
                category: ActivityCategory::Task,
                avg_duration_ms: 600_000,
                duration_variance: 0.4,
            },
            ActivityDef {
                id: 3,
                name: "Assignment",
                category: ActivityCategory::Task,
                avg_duration_ms: 300_000,
                duration_variance: 0.3,
            },
            ActivityDef {
                id: 4,
                name: "Investigation",
                category: ActivityCategory::Task,
                avg_duration_ms: 7_200_000,
                duration_variance: 0.6,
            },
            ActivityDef {
                id: 5,
                name: "Escalation",
                category: ActivityCategory::Task,
                avg_duration_ms: 1_800_000,
                duration_variance: 0.4,
            },
            ActivityDef {
                id: 6,
                name: "Resolution",
                category: ActivityCategory::Task,
                avg_duration_ms: 3_600_000,
                duration_variance: 0.5,
            },
            ActivityDef {
                id: 7,
                name: "Verification",
                category: ActivityCategory::Gateway,
                avg_duration_ms: 1_200_000,
                duration_variance: 0.3,
            },
            ActivityDef {
                id: 8,
                name: "Incident Closed",
                category: ActivityCategory::End,
                avg_duration_ms: 300_000,
                duration_variance: 0.2,
            },
        ]
    }

    fn transitions(&self) -> Vec<TransitionDef> {
        vec![
            TransitionDef {
                source: "Incident Reported",
                target: "Classification",
                probability: 1.0,
                avg_transition_ms: 30_000,
            },
            TransitionDef {
                source: "Classification",
                target: "Assignment",
                probability: 1.0,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Assignment",
                target: "Investigation",
                probability: 1.0,
                avg_transition_ms: 120_000,
            },
            TransitionDef {
                source: "Investigation",
                target: "Resolution",
                probability: 0.85,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Investigation",
                target: "Escalation",
                probability: 0.12,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Investigation",
                target: "Investigation",
                probability: 0.03,
                avg_transition_ms: 3_600_000,
            }, // Additional investigation
            TransitionDef {
                source: "Escalation",
                target: "Investigation",
                probability: 0.7,
                avg_transition_ms: 1_800_000,
            },
            TransitionDef {
                source: "Escalation",
                target: "Resolution",
                probability: 0.3,
                avg_transition_ms: 600_000,
            },
            TransitionDef {
                source: "Resolution",
                target: "Verification",
                probability: 1.0,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Verification",
                target: "Incident Closed",
                probability: 0.95,
                avg_transition_ms: 60_000,
            },
            TransitionDef {
                source: "Verification",
                target: "Investigation",
                probability: 0.05,
                avg_transition_ms: 300_000,
            }, // Reopen
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healthcare_template() {
        let template = SectorTemplate::Healthcare(HealthcareConfig::default());
        assert_eq!(template.name(), "Healthcare");
        assert!(!template.activities().is_empty());
        assert!(!template.transitions().is_empty());
    }

    #[test]
    fn test_all_templates() {
        let templates = vec![
            SectorTemplate::Healthcare(HealthcareConfig::default()),
            SectorTemplate::Manufacturing(ManufacturingConfig::default()),
            SectorTemplate::Finance(FinanceConfig::default()),
            SectorTemplate::IncidentManagement(IncidentConfig::default()),
        ];

        for template in templates {
            let registry = template.build_registry();
            assert!(
                !registry.is_empty(),
                "Registry for {} should not be empty",
                template.name()
            );
        }
    }
}
