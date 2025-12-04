//! Anomaly injection for synthetic process data.
//!
//! Provides configurable anomaly patterns for realistic test data.

/// Anomaly configuration for event generation.
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Rate of bottleneck anomalies (0.0 - 1.0).
    pub bottleneck_rate: f32,
    /// Rate of rework loops (0.0 - 1.0).
    pub rework_rate: f32,
    /// Rate of long-running activities (0.0 - 1.0).
    pub long_running_rate: f32,
    /// Rate of skipped activities (0.0 - 1.0).
    pub skip_rate: f32,
    /// Bottleneck duration multiplier.
    pub bottleneck_multiplier: f32,
    /// Long-running duration multiplier.
    pub long_running_multiplier: f32,
    /// Maximum rework iterations.
    pub max_rework_iterations: u32,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            bottleneck_rate: 0.05,
            rework_rate: 0.03,
            long_running_rate: 0.02,
            skip_rate: 0.01,
            bottleneck_multiplier: 5.0,
            long_running_multiplier: 10.0,
            max_rework_iterations: 3,
        }
    }
}

impl AnomalyConfig {
    /// Create config with no anomalies (perfect conformance).
    pub fn none() -> Self {
        Self {
            bottleneck_rate: 0.0,
            rework_rate: 0.0,
            long_running_rate: 0.0,
            skip_rate: 0.0,
            ..Default::default()
        }
    }

    /// Create config with high anomaly rates for testing.
    pub fn high() -> Self {
        Self {
            bottleneck_rate: 0.15,
            rework_rate: 0.10,
            long_running_rate: 0.08,
            skip_rate: 0.05,
            ..Default::default()
        }
    }

    /// Total anomaly rate.
    pub fn total_rate(&self) -> f32 {
        self.bottleneck_rate + self.rework_rate + self.long_running_rate + self.skip_rate
    }

    /// Set bottleneck rate.
    pub fn with_bottleneck_rate(mut self, rate: f32) -> Self {
        self.bottleneck_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set rework rate.
    pub fn with_rework_rate(mut self, rate: f32) -> Self {
        self.rework_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set long-running rate.
    pub fn with_long_running_rate(mut self, rate: f32) -> Self {
        self.long_running_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set skip rate.
    pub fn with_skip_rate(mut self, rate: f32) -> Self {
        self.skip_rate = rate.clamp(0.0, 1.0);
        self
    }
}

/// Anomaly injector for modifying generated events.
#[derive(Debug, Clone)]
pub struct AnomalyInjector {
    config: AnomalyConfig,
}

impl AnomalyInjector {
    /// Create a new anomaly injector.
    pub fn new(config: AnomalyConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &AnomalyConfig {
        &self.config
    }

    /// Check if bottleneck should be injected.
    pub fn should_inject_bottleneck(&self, random: f32) -> bool {
        random < self.config.bottleneck_rate
    }

    /// Check if rework should be injected.
    pub fn should_inject_rework(&self, random: f32) -> bool {
        random < self.config.bottleneck_rate + self.config.rework_rate
            && random >= self.config.bottleneck_rate
    }

    /// Check if long-running should be injected.
    pub fn should_inject_long_running(&self, random: f32) -> bool {
        let threshold = self.config.bottleneck_rate + self.config.rework_rate;
        random < threshold + self.config.long_running_rate && random >= threshold
    }

    /// Check if skip should be injected.
    pub fn should_inject_skip(&self, random: f32) -> bool {
        let threshold =
            self.config.bottleneck_rate + self.config.rework_rate + self.config.long_running_rate;
        random < threshold + self.config.skip_rate && random >= threshold
    }

    /// Apply bottleneck duration multiplier.
    pub fn apply_bottleneck(&self, base_duration: u32) -> u32 {
        (base_duration as f32 * self.config.bottleneck_multiplier) as u32
    }

    /// Apply long-running duration multiplier.
    pub fn apply_long_running(&self, base_duration: u32) -> u32 {
        (base_duration as f32 * self.config.long_running_multiplier) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AnomalyConfig::default();
        assert!(config.total_rate() > 0.0);
        assert!(config.total_rate() < 0.2);
    }

    #[test]
    fn test_no_anomalies() {
        let config = AnomalyConfig::none();
        assert_eq!(config.total_rate(), 0.0);
    }

    #[test]
    fn test_injector() {
        let injector = AnomalyInjector::new(AnomalyConfig::default());

        // Test bottleneck check
        assert!(injector.should_inject_bottleneck(0.01));
        assert!(!injector.should_inject_bottleneck(0.99));

        // Test duration multiplier
        let base = 1000;
        let bottleneck = injector.apply_bottleneck(base);
        assert!(bottleneck > base);
    }
}
