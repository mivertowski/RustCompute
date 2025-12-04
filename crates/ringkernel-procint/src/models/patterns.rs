//! Pattern definitions for process intelligence.
//!
//! Defines pattern types detected by GPU kernels.

use super::ActivityId;
use rkyv::{Archive, Deserialize, Serialize};

/// Pattern type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
#[repr(u8)]
pub enum PatternType {
    /// Activities in strict sequence.
    Sequential = 0,
    /// Activities with overlapping execution.
    Parallel = 1,
    /// Repeated activity occurrence.
    Loop = 2,
    /// Decision point with multiple paths.
    Conditional = 3,
    /// Exclusive choice (XOR split).
    XorSplit = 4,
    /// Parallel execution (AND split).
    AndSplit = 5,
    /// Activity exceeding duration threshold.
    LongRunning = 6,
    /// Activity causing waiting time.
    Bottleneck = 7,
    /// Activity repeated due to quality issues.
    Rework = 8,
    /// Circular flow pattern (A → B → A).
    Circular = 9,
}

impl PatternType {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            PatternType::Sequential => "Sequential Flow",
            PatternType::Parallel => "Parallel Flow",
            PatternType::Loop => "Loop",
            PatternType::Conditional => "Conditional Branch",
            PatternType::XorSplit => "XOR Split",
            PatternType::AndSplit => "AND Split",
            PatternType::LongRunning => "Long-Running",
            PatternType::Bottleneck => "Bottleneck",
            PatternType::Rework => "Rework",
            PatternType::Circular => "Circular Flow",
        }
    }

    /// Get icon for UI display.
    pub fn icon(&self) -> &'static str {
        match self {
            PatternType::Sequential => "→",
            PatternType::Parallel => "⇉",
            PatternType::Loop => "↺",
            PatternType::Conditional => "◇",
            PatternType::XorSplit => "⊕",
            PatternType::AndSplit => "⊗",
            PatternType::LongRunning => "⏱",
            PatternType::Bottleneck => "⚠",
            PatternType::Rework => "↩",
            PatternType::Circular => "⟳",
        }
    }

    /// Get description.
    pub fn description(&self) -> &'static str {
        match self {
            PatternType::Sequential => "Activities executed in strict order",
            PatternType::Parallel => "Activities executed concurrently",
            PatternType::Loop => "Activity repeated multiple times",
            PatternType::Conditional => "Decision point with multiple paths",
            PatternType::XorSplit => "Exclusive choice between paths",
            PatternType::AndSplit => "Parallel execution of paths",
            PatternType::LongRunning => "Activity duration exceeds threshold",
            PatternType::Bottleneck => "Activity causing delays in the process",
            PatternType::Rework => "Activity repeated due to issues",
            PatternType::Circular => "Circular dependency between activities",
        }
    }
}

/// Pattern severity level.
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
pub enum PatternSeverity {
    /// Informational pattern.
    #[default]
    Info = 0,
    /// Warning - may indicate issue.
    Warning = 1,
    /// Critical - definite issue.
    Critical = 2,
}

impl PatternSeverity {
    /// Get color for UI (RGB).
    pub fn color(&self) -> [u8; 3] {
        match self {
            PatternSeverity::Info => [100, 149, 237], // Cornflower blue
            PatternSeverity::Warning => [255, 165, 0], // Orange
            PatternSeverity::Critical => [220, 53, 69], // Red
        }
    }

    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            PatternSeverity::Info => "Info",
            PatternSeverity::Warning => "Warning",
            PatternSeverity::Critical => "Critical",
        }
    }
}

/// GPU-compatible pattern match (64 bytes, cache-line aligned).
#[derive(Debug, Clone, Copy, Default, Archive, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct GpuPatternMatch {
    /// Pattern type.
    pub pattern_type: u8,
    /// Severity level.
    pub severity: u8,
    /// Number of activities in the pattern.
    pub activity_count: u8,
    /// Status flags.
    pub flags: u8,
    /// Activity IDs involved (up to 8).
    pub activity_ids: [u32; 8],
    /// Detection confidence (0.0 - 1.0).
    pub confidence: f32,
    /// Pattern frequency (how many times detected).
    pub frequency: u32,
    /// Average duration of the pattern (ms).
    pub avg_duration_ms: f32,
    /// Impact score (0.0 - 1.0).
    pub impact: f32,
    /// Reserved.
    pub _reserved: [u8; 4],
}

// Verify size
const _: () = assert!(std::mem::size_of::<GpuPatternMatch>() == 64);

impl GpuPatternMatch {
    /// Create a new pattern match.
    pub fn new(pattern_type: PatternType, severity: PatternSeverity) -> Self {
        Self {
            pattern_type: pattern_type as u8,
            severity: severity as u8,
            ..Default::default()
        }
    }

    /// Get the pattern type.
    pub fn get_pattern_type(&self) -> PatternType {
        match self.pattern_type {
            0 => PatternType::Sequential,
            1 => PatternType::Parallel,
            2 => PatternType::Loop,
            3 => PatternType::Conditional,
            4 => PatternType::XorSplit,
            5 => PatternType::AndSplit,
            6 => PatternType::LongRunning,
            7 => PatternType::Bottleneck,
            8 => PatternType::Rework,
            9 => PatternType::Circular,
            _ => PatternType::Sequential,
        }
    }

    /// Get the severity level.
    pub fn get_severity(&self) -> PatternSeverity {
        match self.severity {
            0 => PatternSeverity::Info,
            1 => PatternSeverity::Warning,
            2 => PatternSeverity::Critical,
            _ => PatternSeverity::Info,
        }
    }

    /// Add an activity to the pattern.
    pub fn add_activity(&mut self, activity_id: ActivityId) {
        if (self.activity_count as usize) < 8 {
            self.activity_ids[self.activity_count as usize] = activity_id;
            self.activity_count += 1;
        }
    }

    /// Get activities involved in the pattern.
    pub fn activities(&self) -> &[u32] {
        &self.activity_ids[..self.activity_count as usize]
    }

    /// Set confidence and impact based on pattern type.
    pub fn with_metrics(mut self, confidence: f32, frequency: u32, avg_duration: f32) -> Self {
        self.confidence = confidence;
        self.frequency = frequency;
        self.avg_duration_ms = avg_duration;
        self.impact = self.calculate_impact();
        self
    }

    /// Calculate impact score based on pattern type and metrics.
    fn calculate_impact(&self) -> f32 {
        let base = match self.get_pattern_type() {
            PatternType::Bottleneck => 0.8,
            PatternType::Rework => 0.7,
            PatternType::LongRunning => 0.6,
            PatternType::Loop => 0.5,
            PatternType::Circular => 0.9,
            _ => 0.3,
        };

        let freq_factor = (self.frequency as f32 / 100.0).min(0.2);
        (base + freq_factor).min(1.0) * self.confidence
    }
}

/// High-level pattern definition for detection configuration.
#[derive(Debug, Clone)]
pub struct PatternDefinition {
    /// Pattern type.
    pub pattern_type: PatternType,
    /// Default severity.
    pub severity: PatternSeverity,
    /// Minimum confidence threshold.
    pub min_confidence: f32,
    /// Minimum frequency to report.
    pub min_frequency: u32,
    /// Duration threshold for long-running detection (ms).
    pub duration_threshold_ms: u32,
    /// Is detection enabled?
    pub enabled: bool,
}

impl PatternDefinition {
    /// Create a new pattern definition.
    pub fn new(pattern_type: PatternType) -> Self {
        Self {
            pattern_type,
            severity: PatternSeverity::Info,
            min_confidence: 0.5,
            min_frequency: 1,
            duration_threshold_ms: 60000, // 1 minute
            enabled: true,
        }
    }

    /// Set severity.
    pub fn with_severity(mut self, severity: PatternSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set minimum confidence.
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Set duration threshold.
    pub fn with_duration_threshold(mut self, threshold_ms: u32) -> Self {
        self.duration_threshold_ms = threshold_ms;
        self
    }
}

/// Default pattern definitions for common patterns.
pub fn default_pattern_definitions() -> Vec<PatternDefinition> {
    vec![
        PatternDefinition::new(PatternType::Bottleneck)
            .with_severity(PatternSeverity::Critical)
            .with_min_confidence(0.7),
        PatternDefinition::new(PatternType::Loop)
            .with_severity(PatternSeverity::Warning)
            .with_min_confidence(0.6),
        PatternDefinition::new(PatternType::Rework)
            .with_severity(PatternSeverity::Warning)
            .with_min_confidence(0.6),
        PatternDefinition::new(PatternType::LongRunning)
            .with_severity(PatternSeverity::Warning)
            .with_duration_threshold(300000), // 5 minutes
        PatternDefinition::new(PatternType::Circular)
            .with_severity(PatternSeverity::Critical)
            .with_min_confidence(0.8),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_match_size() {
        assert_eq!(std::mem::size_of::<GpuPatternMatch>(), 64);
    }

    #[test]
    fn test_pattern_match_activities() {
        let mut pattern = GpuPatternMatch::new(PatternType::Bottleneck, PatternSeverity::Critical);
        pattern.add_activity(1);
        pattern.add_activity(2);
        pattern.add_activity(3);

        assert_eq!(pattern.activity_count, 3);
        assert_eq!(pattern.activities(), &[1, 2, 3]);
    }

    #[test]
    fn test_pattern_impact() {
        let pattern = GpuPatternMatch::new(PatternType::Bottleneck, PatternSeverity::Critical)
            .with_metrics(0.9, 50, 10000.0);

        assert!(pattern.impact > 0.7);
    }
}
