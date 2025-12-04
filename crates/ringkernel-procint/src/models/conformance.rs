//! Conformance checking types for process intelligence.
//!
//! Defines metrics and results for validating traces against models.

use super::{ActivityId, TraceId};
use rkyv::{Archive, Deserialize, Serialize};

/// Conformance checking status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum ConformanceStatus {
    /// Trace fully conforms to model.
    #[default]
    Conformant = 0,
    /// Wrong activity sequence.
    WrongSequence = 1,
    /// Missing expected activity.
    MissingActivity = 2,
    /// Extra unexpected activity.
    ExtraActivity = 3,
    /// General deviation.
    Deviation = 4,
    /// Timing violation.
    TimingViolation = 5,
}

impl ConformanceStatus {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            ConformanceStatus::Conformant => "Conformant",
            ConformanceStatus::WrongSequence => "Wrong Sequence",
            ConformanceStatus::MissingActivity => "Missing Activity",
            ConformanceStatus::ExtraActivity => "Extra Activity",
            ConformanceStatus::Deviation => "Deviation",
            ConformanceStatus::TimingViolation => "Timing Violation",
        }
    }
}

/// Compliance level classification.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Default,
    Archive,
    Serialize,
    Deserialize,
)]
#[repr(u8)]
pub enum ComplianceLevel {
    /// 95%+ fitness.
    #[default]
    FullyCompliant = 0,
    /// 80-95% fitness.
    MostlyCompliant = 1,
    /// 50-80% fitness.
    PartiallyCompliant = 2,
    /// <50% fitness.
    NonCompliant = 3,
}

impl ComplianceLevel {
    /// Get color for UI (RGB).
    pub fn color(&self) -> [u8; 3] {
        match self {
            ComplianceLevel::FullyCompliant => [40, 167, 69], // Green
            ComplianceLevel::MostlyCompliant => [255, 193, 7], // Yellow
            ComplianceLevel::PartiallyCompliant => [255, 152, 0], // Orange
            ComplianceLevel::NonCompliant => [220, 53, 69],   // Red
        }
    }

    /// Get from fitness score.
    pub fn from_fitness(fitness: f32) -> Self {
        if fitness >= 0.95 {
            ComplianceLevel::FullyCompliant
        } else if fitness >= 0.80 {
            ComplianceLevel::MostlyCompliant
        } else if fitness >= 0.50 {
            ComplianceLevel::PartiallyCompliant
        } else {
            ComplianceLevel::NonCompliant
        }
    }

    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            ComplianceLevel::FullyCompliant => "Fully Compliant",
            ComplianceLevel::MostlyCompliant => "Mostly Compliant",
            ComplianceLevel::PartiallyCompliant => "Partially Compliant",
            ComplianceLevel::NonCompliant => "Non-Compliant",
        }
    }
}

/// Alignment move type for trace-model alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum AlignmentType {
    /// Both log and model advance (matching).
    #[default]
    Synchronous = 0,
    /// Extra activity in log (not in model).
    LogMove = 1,
    /// Missing activity in log (expected by model).
    ModelMove = 2,
}

impl AlignmentType {
    /// Get the cost of this alignment move.
    pub fn cost(&self) -> u32 {
        match self {
            AlignmentType::Synchronous => 0,
            AlignmentType::LogMove => 1,
            AlignmentType::ModelMove => 1,
        }
    }
}

/// Single alignment move.
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct AlignmentMove {
    /// Type of move.
    pub move_type: u8,
    /// Padding.
    pub _padding: [u8; 3],
    /// Activity in log (0xFFFFFFFF if model-only).
    pub log_activity: u32,
    /// Activity in model (0xFFFFFFFF if log-only).
    pub model_activity: u32,
    /// Cost of this move.
    pub cost: u32,
}

impl AlignmentMove {
    /// Create a synchronous move.
    pub fn synchronous(activity: ActivityId) -> Self {
        Self {
            move_type: AlignmentType::Synchronous as u8,
            log_activity: activity,
            model_activity: activity,
            cost: 0,
            _padding: [0; 3],
        }
    }

    /// Create a log move (extra activity in log).
    pub fn log_move(activity: ActivityId) -> Self {
        Self {
            move_type: AlignmentType::LogMove as u8,
            log_activity: activity,
            model_activity: u32::MAX,
            cost: 1,
            _padding: [0; 3],
        }
    }

    /// Create a model move (missing activity in log).
    pub fn model_move(activity: ActivityId) -> Self {
        Self {
            move_type: AlignmentType::ModelMove as u8,
            log_activity: u32::MAX,
            model_activity: activity,
            cost: 1,
            _padding: [0; 3],
        }
    }

    /// Get the alignment type.
    pub fn get_type(&self) -> AlignmentType {
        match self.move_type {
            0 => AlignmentType::Synchronous,
            1 => AlignmentType::LogMove,
            2 => AlignmentType::ModelMove,
            _ => AlignmentType::Synchronous,
        }
    }
}

/// GPU-compatible conformance result (64 bytes).
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct ConformanceResult {
    /// Trace identifier.
    pub trace_id: u64,
    /// Model identifier.
    pub model_id: u32,
    /// Conformance status.
    pub status: u8,
    /// Compliance level.
    pub compliance_level: u8,
    /// Padding.
    pub _padding1: [u8; 2],
    /// Fitness score (0.0 - 1.0).
    pub fitness: f32,
    /// Precision score (0.0 - 1.0).
    pub precision: f32,
    /// Generalization score (0.0 - 1.0).
    pub generalization: f32,
    /// Simplicity score (0.0 - 1.0).
    pub simplicity: f32,
    /// Number of missing activities.
    pub missing_count: u16,
    /// Number of extra activities.
    pub extra_count: u16,
    /// Total alignment cost.
    pub alignment_cost: u32,
    /// Number of alignment moves.
    pub alignment_length: u32,
    /// Reserved.
    pub _reserved: [u8; 16],
}

// Verify size
const _: () = assert!(std::mem::size_of::<ConformanceResult>() == 64);

impl ConformanceResult {
    /// Create a new conformant result.
    pub fn conformant(trace_id: TraceId, model_id: u32) -> Self {
        Self {
            trace_id,
            model_id,
            status: ConformanceStatus::Conformant as u8,
            compliance_level: ComplianceLevel::FullyCompliant as u8,
            fitness: 1.0,
            precision: 1.0,
            generalization: 0.8,
            simplicity: 1.0,
            ..Default::default()
        }
    }

    /// Create a non-conformant result with deviations.
    pub fn with_deviations(
        trace_id: TraceId,
        model_id: u32,
        missing: u16,
        extra: u16,
        alignment_cost: u32,
    ) -> Self {
        let total = missing + extra;
        let fitness = if total > 0 {
            1.0 - (alignment_cost as f32 / (alignment_cost + 10) as f32)
        } else {
            1.0
        };

        let status = if missing > 0 {
            ConformanceStatus::MissingActivity
        } else if extra > 0 {
            ConformanceStatus::ExtraActivity
        } else {
            ConformanceStatus::Deviation
        };

        Self {
            trace_id,
            model_id,
            status: status as u8,
            compliance_level: ComplianceLevel::from_fitness(fitness) as u8,
            fitness,
            precision: 1.0 - (extra as f32 / 10.0).min(1.0),
            generalization: 0.8,
            simplicity: 1.0,
            missing_count: missing,
            extra_count: extra,
            alignment_cost,
            ..Default::default()
        }
    }

    /// Get conformance status.
    pub fn get_status(&self) -> ConformanceStatus {
        match self.status {
            0 => ConformanceStatus::Conformant,
            1 => ConformanceStatus::WrongSequence,
            2 => ConformanceStatus::MissingActivity,
            3 => ConformanceStatus::ExtraActivity,
            4 => ConformanceStatus::Deviation,
            5 => ConformanceStatus::TimingViolation,
            _ => ConformanceStatus::Conformant,
        }
    }

    /// Get compliance level.
    pub fn get_compliance_level(&self) -> ComplianceLevel {
        match self.compliance_level {
            0 => ComplianceLevel::FullyCompliant,
            1 => ComplianceLevel::MostlyCompliant,
            2 => ComplianceLevel::PartiallyCompliant,
            3 => ComplianceLevel::NonCompliant,
            _ => ComplianceLevel::FullyCompliant,
        }
    }

    /// Check if conformant.
    pub fn is_conformant(&self) -> bool {
        self.status == ConformanceStatus::Conformant as u8
    }

    /// Calculate F-score from fitness and precision.
    pub fn f_score(&self) -> f32 {
        if self.fitness + self.precision > 0.0 {
            2.0 * self.fitness * self.precision / (self.fitness + self.precision)
        } else {
            0.0
        }
    }
}

/// Process model for conformance checking.
#[derive(Debug, Clone)]
pub struct ProcessModel {
    /// Model identifier.
    pub id: u32,
    /// Model name.
    pub name: String,
    /// Model type.
    pub model_type: ProcessModelType,
    /// Start activities.
    pub start_activities: Vec<ActivityId>,
    /// End activities.
    pub end_activities: Vec<ActivityId>,
    /// Valid transitions (source, target).
    pub transitions: Vec<(ActivityId, ActivityId)>,
}

impl ProcessModel {
    /// Create a new process model.
    pub fn new(id: u32, name: impl Into<String>, model_type: ProcessModelType) -> Self {
        Self {
            id,
            name: name.into(),
            model_type,
            start_activities: Vec::new(),
            end_activities: Vec::new(),
            transitions: Vec::new(),
        }
    }

    /// Add a transition.
    pub fn add_transition(&mut self, source: ActivityId, target: ActivityId) {
        self.transitions.push((source, target));
    }

    /// Check if a transition is valid.
    pub fn has_transition(&self, source: ActivityId, target: ActivityId) -> bool {
        self.transitions.contains(&(source, target))
    }
}

/// Process model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum ProcessModelType {
    /// Directly-Follows Graph.
    #[default]
    DFG = 0,
    /// Petri Net.
    PetriNet = 1,
    /// BPMN model.
    BPMN = 2,
    /// Process Tree.
    ProcessTree = 3,
    /// DECLARE constraints.
    Declare = 4,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformance_result_size() {
        assert_eq!(std::mem::size_of::<ConformanceResult>(), 64);
    }

    #[test]
    fn test_compliance_level_from_fitness() {
        assert_eq!(
            ComplianceLevel::from_fitness(0.98),
            ComplianceLevel::FullyCompliant
        );
        assert_eq!(
            ComplianceLevel::from_fitness(0.85),
            ComplianceLevel::MostlyCompliant
        );
        assert_eq!(
            ComplianceLevel::from_fitness(0.60),
            ComplianceLevel::PartiallyCompliant
        );
        assert_eq!(
            ComplianceLevel::from_fitness(0.30),
            ComplianceLevel::NonCompliant
        );
    }

    #[test]
    fn test_alignment_moves() {
        let sync = AlignmentMove::synchronous(1);
        assert_eq!(sync.cost, 0);
        assert_eq!(sync.get_type(), AlignmentType::Synchronous);

        let log = AlignmentMove::log_move(2);
        assert_eq!(log.cost, 1);
        assert_eq!(log.get_type(), AlignmentType::LogMove);

        let model = AlignmentMove::model_move(3);
        assert_eq!(model.cost, 1);
        assert_eq!(model.get_type(), AlignmentType::ModelMove);
    }
}
