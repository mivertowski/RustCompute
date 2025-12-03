//! Transaction generator for synthetic workloads.

use super::{AccountGenerator, TransactionPattern};
use crate::types::{CustomerRiskLevel, CustomerRiskProfile, Transaction};
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::collections::HashMap;

/// Transaction factory state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FactoryState {
    /// Factory is stopped.
    #[default]
    Stopped,
    /// Factory is running and generating transactions.
    Running,
    /// Factory is paused.
    Paused,
}

impl std::fmt::Display for FactoryState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactoryState::Stopped => write!(f, "Stopped"),
            FactoryState::Running => write!(f, "Running"),
            FactoryState::Paused => write!(f, "Paused"),
        }
    }
}

/// Configuration for the transaction generator.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Target transactions per second.
    pub transactions_per_second: u32,
    /// Number of simulated customer accounts.
    pub customer_count: u32,
    /// Percentage of suspicious transactions (0-100).
    pub suspicious_rate: u8,
    /// Batch size for transaction generation.
    pub batch_size: u32,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            transactions_per_second: 100,
            customer_count: 1000,
            suspicious_rate: 5,
            batch_size: 64,
        }
    }
}

/// Transaction generator that produces synthetic transaction workloads.
pub struct TransactionGenerator {
    config: GeneratorConfig,
    rng: SmallRng,
    customers: Vec<CustomerRiskProfile>,
    customer_map: HashMap<u64, usize>,
    next_tx_id: u64,
}

impl TransactionGenerator {
    /// Create a new transaction generator with the given configuration.
    pub fn new(config: GeneratorConfig) -> Self {
        let mut acct_gen = AccountGenerator::new(42);
        let customers = acct_gen.generate_customers(config.customer_count);

        let customer_map: HashMap<u64, usize> = customers
            .iter()
            .enumerate()
            .map(|(i, c)| (c.customer_id, i))
            .collect();

        Self {
            config,
            rng: SmallRng::seed_from_u64(12345),
            customers,
            customer_map,
            next_tx_id: 1,
        }
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &GeneratorConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: GeneratorConfig) {
        self.config = config;
    }

    /// Get the number of customers.
    pub fn customer_count(&self) -> usize {
        self.customers.len()
    }

    /// Get a reference to all customers.
    pub fn customers(&self) -> &[CustomerRiskProfile] {
        &self.customers
    }

    /// Get a mutable reference to a customer by ID.
    pub fn get_customer_mut(&mut self, customer_id: u64) -> Option<&mut CustomerRiskProfile> {
        self.customer_map
            .get(&customer_id)
            .copied()
            .and_then(|idx| self.customers.get_mut(idx))
    }

    /// Generate a batch of transactions and return with their customer profiles.
    ///
    /// Returns (transactions, corresponding profiles).
    pub fn generate_batch(&mut self) -> (Vec<Transaction>, Vec<CustomerRiskProfile>) {
        let batch_size = self.config.batch_size as usize;
        let mut transactions = Vec::with_capacity(batch_size);
        let mut profiles = Vec::with_capacity(batch_size);

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Track which transactions are suspicious for velocity updates
        let mut suspicious_tx_indices = Vec::new();

        for i in 0..batch_size {
            // Select a random customer
            let customer_idx = self.rng.gen_range(0..self.customers.len());
            let customer = &self.customers[customer_idx];

            // Determine if this should be a suspicious transaction
            let is_suspicious = self.config.suspicious_rate > 0
                && self.rng.gen_range(0..100) < self.config.suspicious_rate;

            // Choose pattern
            let pattern = if is_suspicious {
                TransactionPattern::random_suspicious(&mut self.rng)
            } else {
                TransactionPattern::random_normal(&mut self.rng)
            };

            // Generate transaction
            let tx_id = self.next_tx_id;
            self.next_tx_id += 1;

            // Vary timestamp slightly within batch
            let tx_timestamp = timestamp + i as u64;

            let tx = pattern.generate(tx_id, customer, tx_timestamp, &mut self.rng);

            transactions.push(tx);
            profiles.push(*customer);

            if is_suspicious {
                suspicious_tx_indices.push(i);
            }
        }

        // Only update velocity counts for suspicious transactions
        // This prevents normal activity from triggering velocity alerts
        for &idx in &suspicious_tx_indices {
            let tx = &transactions[idx];
            if let Some(customer) = self.get_customer_mut(tx.customer_id) {
                customer.increment_velocity();
            }
        }

        // Update transaction counts for all transactions (this doesn't affect alerts)
        for tx in &transactions {
            if let Some(customer) = self.get_customer_mut(tx.customer_id) {
                customer.increment_transactions();
                customer.last_transaction_ts = tx.timestamp;
            }
        }

        (transactions, profiles)
    }

    /// Reset velocity counts for all customers (simulating new time window).
    pub fn reset_velocity_window(&mut self) {
        for customer in &mut self.customers {
            customer.reset_velocity();
        }
    }

    /// Get statistics about the customer base.
    pub fn customer_stats(&self) -> CustomerStats {
        let mut stats = CustomerStats::default();
        stats.total = self.customers.len();

        for c in &self.customers {
            match c.risk_level() {
                CustomerRiskLevel::Low => stats.low_risk += 1,
                CustomerRiskLevel::Medium => stats.medium_risk += 1,
                CustomerRiskLevel::High => stats.high_risk += 1,
                CustomerRiskLevel::Prohibited => stats.prohibited += 1,
            }
            if c.is_pep() {
                stats.pep += 1;
            }
            if c.requires_edd() {
                stats.edd_required += 1;
            }
        }

        stats
    }
}

/// Statistics about the customer base.
#[derive(Debug, Clone, Default)]
pub struct CustomerStats {
    /// Total number of customers.
    pub total: usize,
    /// Low risk customers.
    pub low_risk: usize,
    /// Medium risk customers.
    pub medium_risk: usize,
    /// High risk customers.
    pub high_risk: usize,
    /// Prohibited customers.
    pub prohibited: usize,
    /// Politically Exposed Persons.
    pub pep: usize,
    /// Customers requiring Enhanced Due Diligence.
    pub edd_required: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_batch() {
        let config = GeneratorConfig {
            batch_size: 100,
            ..Default::default()
        };
        let mut gen = TransactionGenerator::new(config);

        let (transactions, profiles) = gen.generate_batch();

        assert_eq!(transactions.len(), 100);
        assert_eq!(profiles.len(), 100);

        // Check all transactions have valid customer IDs
        for (tx, profile) in transactions.iter().zip(profiles.iter()) {
            assert_eq!(tx.customer_id, profile.customer_id);
        }
    }

    #[test]
    fn test_suspicious_rate() {
        let config = GeneratorConfig {
            batch_size: 1000,
            suspicious_rate: 10, // 10% suspicious
            ..Default::default()
        };
        let mut gen = TransactionGenerator::new(config);

        let (transactions, _) = gen.generate_batch();

        // Count transactions with amounts near threshold (smurfing indicator)
        let near_threshold = transactions
            .iter()
            .filter(|tx| tx.amount_cents >= 9_000_00 && tx.amount_cents < 10_000_00)
            .count();

        // Should have some suspicious transactions
        assert!(near_threshold > 0);
    }

    #[test]
    fn test_velocity_increment() {
        let config = GeneratorConfig {
            batch_size: 10,
            customer_count: 1,
            suspicious_rate: 100, // All transactions are suspicious
            ..Default::default()
        };
        let mut gen = TransactionGenerator::new(config);

        // Initial velocity should be 0
        assert_eq!(gen.customers[0].velocity_count, 0);

        // Generate batch (all suspicious, so all increment velocity)
        let _ = gen.generate_batch();

        // All transactions go to the same customer, all are suspicious
        assert_eq!(gen.customers[0].velocity_count, 10);
    }

    #[test]
    fn test_velocity_no_increment_normal() {
        let config = GeneratorConfig {
            batch_size: 10,
            customer_count: 1,
            suspicious_rate: 0, // No suspicious transactions
            ..Default::default()
        };
        let mut gen = TransactionGenerator::new(config);

        // Initial velocity should be 0
        assert_eq!(gen.customers[0].velocity_count, 0);

        // Generate batch (none suspicious, so no velocity increment)
        let _ = gen.generate_batch();

        // Velocity should still be 0 since no suspicious transactions
        assert_eq!(gen.customers[0].velocity_count, 0);
    }
}
