//! KPI tracking for process intelligence.
//!
//! Tracks key performance indicators across the process mining pipeline.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// KPI tracker for real-time metrics.
#[derive(Debug)]
pub struct KPITracker {
    /// Events per second (recent window).
    throughput_samples: VecDeque<(Instant, u64)>,
    /// Average case duration samples.
    duration_samples: VecDeque<f32>,
    /// Fitness scores.
    fitness_samples: VecDeque<f32>,
    /// Window size for rolling averages.
    window_size: usize,
    /// Last update time.
    last_update: Instant,
    /// Current KPIs.
    pub current: ProcessKPIs,
}

impl Default for KPITracker {
    fn default() -> Self {
        Self::new()
    }
}

impl KPITracker {
    /// Create a new KPI tracker.
    pub fn new() -> Self {
        Self {
            throughput_samples: VecDeque::with_capacity(100),
            duration_samples: VecDeque::with_capacity(100),
            fitness_samples: VecDeque::with_capacity(100),
            window_size: 100,
            last_update: Instant::now(),
            current: ProcessKPIs::default(),
        }
    }

    /// Set window size for rolling averages.
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Record events processed.
    pub fn record_events(&mut self, count: u64) {
        let now = Instant::now();
        self.throughput_samples.push_back((now, count));

        // Remove old samples (older than 10 seconds)
        while let Some(&(time, _)) = self.throughput_samples.front() {
            if now.duration_since(time) > Duration::from_secs(10) {
                self.throughput_samples.pop_front();
            } else {
                break;
            }
        }

        self.update_throughput();
        self.last_update = now;
    }

    /// Record case duration.
    pub fn record_duration(&mut self, duration_ms: f32) {
        self.duration_samples.push_back(duration_ms);
        if self.duration_samples.len() > self.window_size {
            self.duration_samples.pop_front();
        }
        self.update_duration();
    }

    /// Record fitness score.
    pub fn record_fitness(&mut self, fitness: f32) {
        self.fitness_samples.push_back(fitness);
        if self.fitness_samples.len() > self.window_size {
            self.fitness_samples.pop_front();
        }
        self.update_fitness();
    }

    /// Update all KPIs.
    pub fn update(&mut self, events: u64, duration_ms: f32, fitness: f32) {
        self.record_events(events);
        self.record_duration(duration_ms);
        self.record_fitness(fitness);
    }

    fn update_throughput(&mut self) {
        if self.throughput_samples.len() < 2 {
            return;
        }

        let (first_time, _) = self.throughput_samples.front().unwrap();
        let (last_time, _) = self.throughput_samples.back().unwrap();
        let elapsed = last_time.duration_since(*first_time).as_secs_f64();

        if elapsed > 0.0 {
            let total_events: u64 = self.throughput_samples.iter().map(|(_, c)| c).sum();
            self.current.events_per_second = total_events as f64 / elapsed;
        }
    }

    fn update_duration(&mut self) {
        if self.duration_samples.is_empty() {
            return;
        }

        let sum: f32 = self.duration_samples.iter().sum();
        self.current.avg_case_duration_ms = sum / self.duration_samples.len() as f32;

        // Calculate min/max
        self.current.min_duration_ms = self
            .duration_samples
            .iter()
            .copied()
            .fold(f32::MAX, f32::min);
        self.current.max_duration_ms = self
            .duration_samples
            .iter()
            .copied()
            .fold(f32::MIN, f32::max);
    }

    fn update_fitness(&mut self) {
        if self.fitness_samples.is_empty() {
            return;
        }

        let sum: f32 = self.fitness_samples.iter().sum();
        self.current.avg_fitness = sum / self.fitness_samples.len() as f32;
    }

    /// Set pattern count directly.
    pub fn set_pattern_count(&mut self, count: u64) {
        self.current.pattern_count = count;
    }

    /// Set cases completed.
    pub fn set_cases_completed(&mut self, count: u64) {
        self.current.cases_completed = count;
    }

    /// Set conformance rate.
    pub fn set_conformance_rate(&mut self, rate: f32) {
        self.current.conformance_rate = rate;
    }

    /// Get current KPIs.
    pub fn kpis(&self) -> &ProcessKPIs {
        &self.current
    }

    /// Reset tracker.
    pub fn reset(&mut self) {
        self.throughput_samples.clear();
        self.duration_samples.clear();
        self.fitness_samples.clear();
        self.current = ProcessKPIs::default();
    }
}

/// Process KPIs snapshot.
#[derive(Debug, Clone, Default)]
pub struct ProcessKPIs {
    /// Events processed per second.
    pub events_per_second: f64,
    /// Average case duration in milliseconds.
    pub avg_case_duration_ms: f32,
    /// Minimum case duration.
    pub min_duration_ms: f32,
    /// Maximum case duration.
    pub max_duration_ms: f32,
    /// Average fitness score.
    pub avg_fitness: f32,
    /// Number of patterns detected.
    pub pattern_count: u64,
    /// Cases completed.
    pub cases_completed: u64,
    /// Conformance rate (% conformant).
    pub conformance_rate: f32,
}

impl ProcessKPIs {
    /// Format events per second for display.
    pub fn format_throughput(&self) -> String {
        if self.events_per_second >= 1_000_000.0 {
            format!("{:.2}M/s", self.events_per_second / 1_000_000.0)
        } else if self.events_per_second >= 1_000.0 {
            format!("{:.2}K/s", self.events_per_second / 1_000.0)
        } else {
            format!("{:.0}/s", self.events_per_second)
        }
    }

    /// Format duration for display.
    pub fn format_duration(&self) -> String {
        if self.avg_case_duration_ms >= 60000.0 {
            format!("{:.1}m", self.avg_case_duration_ms / 60000.0)
        } else if self.avg_case_duration_ms >= 1000.0 {
            format!("{:.1}s", self.avg_case_duration_ms / 1000.0)
        } else {
            format!("{:.0}ms", self.avg_case_duration_ms)
        }
    }

    /// Format fitness as percentage.
    pub fn format_fitness(&self) -> String {
        format!("{:.1}%", self.avg_fitness * 100.0)
    }

    /// Get health status based on KPIs.
    pub fn health_status(&self) -> HealthStatus {
        if self.avg_fitness >= 0.95 && self.conformance_rate >= 0.90 {
            HealthStatus::Excellent
        } else if self.avg_fitness >= 0.80 && self.conformance_rate >= 0.70 {
            HealthStatus::Good
        } else if self.avg_fitness >= 0.50 {
            HealthStatus::Warning
        } else {
            HealthStatus::Critical
        }
    }
}

/// Process health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Excellent health (fitness > 90%).
    Excellent,
    /// Good health (fitness > 75%).
    Good,
    /// Warning status (fitness > 50%).
    Warning,
    /// Critical status (fitness <= 50%).
    Critical,
}

impl HealthStatus {
    /// Get color for UI (RGB).
    pub fn color(&self) -> [u8; 3] {
        match self {
            HealthStatus::Excellent => [40, 167, 69], // Green
            HealthStatus::Good => [23, 162, 184],     // Cyan
            HealthStatus::Warning => [255, 193, 7],   // Yellow
            HealthStatus::Critical => [220, 53, 69],  // Red
        }
    }

    /// Get name.
    pub fn name(&self) -> &'static str {
        match self {
            HealthStatus::Excellent => "Excellent",
            HealthStatus::Good => "Good",
            HealthStatus::Warning => "Warning",
            HealthStatus::Critical => "Critical",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kpi_tracking() {
        let mut tracker = KPITracker::new();

        tracker.record_events(1000);
        tracker.record_duration(5000.0);
        tracker.record_fitness(0.95);

        let kpis = tracker.kpis();
        assert_eq!(kpis.avg_fitness, 0.95);
        assert_eq!(kpis.avg_case_duration_ms, 5000.0);
    }

    #[test]
    fn test_throughput_formatting() {
        let mut kpis = ProcessKPIs {
            events_per_second: 1_500_000.0,
            ..Default::default()
        };
        assert!(kpis.format_throughput().contains("M/s"));

        kpis.events_per_second = 42_000.0;
        assert!(kpis.format_throughput().contains("K/s"));
    }

    #[test]
    fn test_health_status() {
        let mut kpis = ProcessKPIs {
            avg_fitness: 0.98,
            conformance_rate: 0.95,
            ..Default::default()
        };
        assert_eq!(kpis.health_status(), HealthStatus::Excellent);

        kpis.avg_fitness = 0.40;
        assert_eq!(kpis.health_status(), HealthStatus::Critical);
    }
}
