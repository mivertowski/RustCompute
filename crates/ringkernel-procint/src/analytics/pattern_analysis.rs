//! Pattern analysis and aggregation.
//!
//! Aggregates detected patterns and provides trend analysis.

use crate::models::{GpuPatternMatch, PatternSeverity, PatternType};
use std::collections::HashMap;

/// Pattern aggregator for tracking pattern statistics.
#[derive(Debug, Default)]
pub struct PatternAggregator {
    /// Pattern counts by type.
    pub type_counts: HashMap<PatternType, u64>,
    /// Pattern counts by severity.
    pub severity_counts: HashMap<PatternSeverity, u64>,
    /// Recent patterns (for display).
    pub recent_patterns: Vec<PatternSummary>,
    /// Maximum recent patterns to keep.
    max_recent: usize,
    /// Total patterns detected.
    pub total_detected: u64,
}

impl PatternAggregator {
    /// Create a new pattern aggregator.
    pub fn new() -> Self {
        Self {
            max_recent: 100,
            ..Default::default()
        }
    }

    /// Set maximum recent patterns.
    pub fn with_max_recent(mut self, max: usize) -> Self {
        self.max_recent = max;
        self
    }

    /// Add patterns from detection result.
    pub fn add_patterns(&mut self, patterns: &[GpuPatternMatch]) {
        for pattern in patterns {
            let pattern_type = pattern.get_pattern_type();
            let severity = pattern.get_severity();

            *self.type_counts.entry(pattern_type).or_insert(0) += 1;
            *self.severity_counts.entry(severity).or_insert(0) += 1;
            self.total_detected += 1;

            // Add to recent
            let summary = PatternSummary {
                pattern_type,
                severity,
                confidence: pattern.confidence,
                frequency: pattern.frequency,
                activities: pattern.activities().to_vec(),
                impact: pattern.impact,
            };
            self.recent_patterns.push(summary);

            // Trim if needed
            if self.recent_patterns.len() > self.max_recent {
                self.recent_patterns.remove(0);
            }
        }
    }

    /// Get count by pattern type.
    pub fn count_by_type(&self, pattern_type: PatternType) -> u64 {
        *self.type_counts.get(&pattern_type).unwrap_or(&0)
    }

    /// Get count by severity.
    pub fn count_by_severity(&self, severity: PatternSeverity) -> u64 {
        *self.severity_counts.get(&severity).unwrap_or(&0)
    }

    /// Get top patterns by frequency.
    pub fn top_patterns(&self, n: usize) -> Vec<&PatternSummary> {
        let mut sorted: Vec<_> = self.recent_patterns.iter().collect();
        sorted.sort_by_key(|a| std::cmp::Reverse(a.frequency));
        sorted.truncate(n);
        sorted
    }

    /// Get critical patterns.
    pub fn critical_patterns(&self) -> Vec<&PatternSummary> {
        self.recent_patterns
            .iter()
            .filter(|p| p.severity == PatternSeverity::Critical)
            .collect()
    }

    /// Get pattern distribution.
    pub fn distribution(&self) -> PatternDistribution {
        PatternDistribution {
            bottleneck: self.count_by_type(PatternType::Bottleneck),
            loop_count: self.count_by_type(PatternType::Loop),
            rework: self.count_by_type(PatternType::Rework),
            long_running: self.count_by_type(PatternType::LongRunning),
            circular: self.count_by_type(PatternType::Circular),
            other: self.total_detected
                - self.count_by_type(PatternType::Bottleneck)
                - self.count_by_type(PatternType::Loop)
                - self.count_by_type(PatternType::Rework)
                - self.count_by_type(PatternType::LongRunning)
                - self.count_by_type(PatternType::Circular),
        }
    }

    /// Reset aggregator.
    pub fn reset(&mut self) {
        self.type_counts.clear();
        self.severity_counts.clear();
        self.recent_patterns.clear();
        self.total_detected = 0;
    }
}

/// Summary of a detected pattern.
#[derive(Debug, Clone)]
pub struct PatternSummary {
    /// Pattern type.
    pub pattern_type: PatternType,
    /// Severity level.
    pub severity: PatternSeverity,
    /// Detection confidence.
    pub confidence: f32,
    /// Pattern frequency.
    pub frequency: u32,
    /// Involved activities.
    pub activities: Vec<u32>,
    /// Impact score.
    pub impact: f32,
}

/// Distribution of pattern types.
#[derive(Debug, Clone, Default)]
pub struct PatternDistribution {
    /// Bottleneck pattern count.
    pub bottleneck: u64,
    /// Loop pattern count.
    pub loop_count: u64,
    /// Rework pattern count.
    pub rework: u64,
    /// Long-running pattern count.
    pub long_running: u64,
    /// Circular dependency count.
    pub circular: u64,
    /// Other patterns count.
    pub other: u64,
}

impl PatternDistribution {
    /// Get total patterns.
    pub fn total(&self) -> u64 {
        self.bottleneck
            + self.loop_count
            + self.rework
            + self.long_running
            + self.circular
            + self.other
    }

    /// Get percentages.
    pub fn percentages(&self) -> [f32; 6] {
        let total = self.total() as f32;
        if total == 0.0 {
            return [0.0; 6];
        }
        [
            self.bottleneck as f32 / total * 100.0,
            self.loop_count as f32 / total * 100.0,
            self.rework as f32 / total * 100.0,
            self.long_running as f32 / total * 100.0,
            self.circular as f32 / total * 100.0,
            self.other as f32 / total * 100.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_patterns() -> Vec<GpuPatternMatch> {
        vec![
            GpuPatternMatch::new(PatternType::Bottleneck, PatternSeverity::Critical)
                .with_metrics(0.9, 50, 10000.0),
            GpuPatternMatch::new(PatternType::Loop, PatternSeverity::Warning)
                .with_metrics(0.7, 30, 5000.0),
        ]
    }

    #[test]
    fn test_pattern_aggregation() {
        let mut agg = PatternAggregator::new();
        let patterns = create_test_patterns();
        agg.add_patterns(&patterns);

        assert_eq!(agg.total_detected, 2);
        assert_eq!(agg.count_by_type(PatternType::Bottleneck), 1);
        assert_eq!(agg.count_by_severity(PatternSeverity::Critical), 1);
    }

    #[test]
    fn test_distribution() {
        let mut agg = PatternAggregator::new();
        let patterns = create_test_patterns();
        agg.add_patterns(&patterns);

        let dist = agg.distribution();
        assert_eq!(dist.bottleneck, 1);
        assert_eq!(dist.loop_count, 1);
    }
}
