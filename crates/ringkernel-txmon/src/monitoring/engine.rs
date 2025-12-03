//! Monitoring engine abstraction.

use super::rules::{monitor_transaction, MonitoringConfig};
use crate::types::{CustomerRiskProfile, MonitoringAlert, Transaction};
use std::sync::atomic::{AtomicU64, Ordering};

/// Monitoring engine that processes transactions and generates alerts.
pub struct MonitoringEngine {
    config: MonitoringConfig,
    alert_counter: AtomicU64,
}

impl MonitoringEngine {
    /// Create a new CPU-based monitoring engine.
    pub fn new_cpu(config: MonitoringConfig) -> Self {
        Self {
            config,
            alert_counter: AtomicU64::new(1),
        }
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &MonitoringConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: MonitoringConfig) {
        self.config = config;
    }

    /// Process a batch of transactions and return generated alerts.
    ///
    /// Each transaction is paired with its corresponding customer profile.
    pub fn process_batch(
        &self,
        transactions: &[Transaction],
        profiles: &[CustomerRiskProfile],
    ) -> Vec<MonitoringAlert> {
        let mut all_alerts = Vec::new();

        for (tx, profile) in transactions.iter().zip(profiles.iter()) {
            let alert_id = self.alert_counter.fetch_add(10, Ordering::Relaxed);
            let alerts = monitor_transaction(tx, profile, &self.config, alert_id);
            all_alerts.extend(alerts);
        }

        all_alerts
    }

    /// Process a single transaction.
    pub fn process_single(
        &self,
        tx: &Transaction,
        profile: &CustomerRiskProfile,
    ) -> Vec<MonitoringAlert> {
        let alert_id = self.alert_counter.fetch_add(10, Ordering::Relaxed);
        monitor_transaction(tx, profile, &self.config, alert_id)
    }

    /// Get the current alert counter value.
    pub fn alert_count(&self) -> u64 {
        self.alert_counter.load(Ordering::Relaxed)
    }

    /// Reset the alert counter.
    pub fn reset_counter(&self) {
        self.alert_counter.store(1, Ordering::Relaxed);
    }
}

impl Default for MonitoringEngine {
    fn default() -> Self {
        Self::new_cpu(MonitoringConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_batch() {
        let engine = MonitoringEngine::default();

        let transactions = vec![
            Transaction::new(1, 1, 15_000_00, 1, 0), // Over threshold
            Transaction::new(2, 2, 500_00, 1, 0),    // Normal
            Transaction::new(3, 3, 9_500_00, 1, 0),  // Just under threshold
        ];

        let mut profiles: Vec<CustomerRiskProfile> = (1..=3)
            .map(|id| CustomerRiskProfile::new(id, 1))
            .collect();

        // Set up profile 3 for structuring detection
        profiles[2].velocity_count = 5;

        let alerts = engine.process_batch(&transactions, &profiles);

        // Should have alerts for tx 1 (amount) and tx 3 (structuring)
        assert!(alerts.len() >= 2);
    }

    #[test]
    fn test_alert_counter_increment() {
        let engine = MonitoringEngine::default();

        let initial = engine.alert_count();

        let tx = Transaction::new(1, 1, 15_000_00, 1, 0);
        let profile = CustomerRiskProfile::new(1, 1);
        let _ = engine.process_single(&tx, &profile);

        assert!(engine.alert_count() > initial);
    }
}
