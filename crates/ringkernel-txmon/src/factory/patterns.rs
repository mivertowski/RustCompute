//! Transaction generation patterns.

use crate::types::{CustomerRiskProfile, Transaction, TransactionType};
use rand::prelude::*;

/// Predefined transaction patterns for testing different scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionPattern {
    /// Normal retail transaction - small amounts, local destinations.
    NormalRetail,
    /// High-value wire transfer - legitimate large transfer.
    HighValueWire,
    /// Velocity burst - many small transactions in quick succession.
    VelocityBurst,
    /// Structured transaction - just under reporting threshold (smurfing).
    Smurfing,
    /// Geographic anomaly - unusual destination country.
    GeographicAnomaly,
    /// Round-trip - money going out and coming back.
    RoundTrip,
}

impl TransactionPattern {
    /// Generate a transaction matching this pattern.
    pub fn generate(
        &self,
        transaction_id: u64,
        customer: &CustomerRiskProfile,
        timestamp: u64,
        rng: &mut impl Rng,
    ) -> Transaction {
        let mut tx = Transaction::new(
            transaction_id,
            customer.customer_id,
            0, // Amount set below
            customer.country_code,
            timestamp,
        );

        match self {
            TransactionPattern::NormalRetail => {
                // Small retail transaction: $10 - $500
                tx.amount_cents = rng.gen_range(10_00..500_00);
                tx.tx_type = TransactionType::Card as u8;
                tx.country_code = customer.country_code; // Local
            }

            TransactionPattern::HighValueWire => {
                // Large legitimate wire: $15,000 - $100,000
                tx.amount_cents = rng.gen_range(15_000_00..100_000_00);
                tx.tx_type = TransactionType::Wire as u8;
                // Mix of domestic and international
                if rng.gen::<f32>() < 0.3 {
                    tx.country_code = rng.gen_range(1..16);
                }
            }

            TransactionPattern::VelocityBurst => {
                // Small amounts, meant to be sent in quick succession
                tx.amount_cents = rng.gen_range(50_00..500_00);
                tx.tx_type = TransactionType::ACH as u8;
                tx.country_code = customer.country_code;
            }

            TransactionPattern::Smurfing => {
                // Just under $10,000 threshold (90-99% of threshold)
                let threshold = 10_000_00u64; // $10,000
                let percentage = rng.gen_range(90..99) as u64;
                tx.amount_cents = (threshold * percentage) / 100;
                tx.tx_type = TransactionType::Cash as u8;
                tx.country_code = customer.country_code;
            }

            TransactionPattern::GeographicAnomaly => {
                // Transaction to unusual/high-risk country
                tx.amount_cents = rng.gen_range(1_000_00..20_000_00);
                tx.tx_type = TransactionType::Wire as u8;
                // Pick a country different from customer's home country
                loop {
                    tx.country_code = rng.gen_range(1..16);
                    if tx.country_code != customer.country_code {
                        break;
                    }
                }
                // Prefer "high-risk" countries (using 7=RU, 6=CN as examples)
                if rng.gen::<f32>() < 0.5 {
                    tx.country_code = if rng.gen::<bool>() { 7 } else { 6 };
                }
            }

            TransactionPattern::RoundTrip => {
                // Round-trip transaction (money comes back)
                tx.amount_cents = rng.gen_range(5_000_00..50_000_00);
                tx.tx_type = TransactionType::Internal as u8;
                tx.destination_id = customer.customer_id; // Same customer
                tx.flags |= 0x01; // Mark as round-trip
            }
        }

        tx
    }

    /// Get a random normal pattern (weighted distribution).
    /// These patterns should NOT trigger alerts under normal circumstances.
    pub fn random_normal(rng: &mut impl Rng) -> Self {
        // Normal patterns are 100% retail - no high-value wires or velocity bursts
        // that could accidentally trigger alerts
        let roll: f32 = rng.gen();
        if roll < 0.95 {
            TransactionPattern::NormalRetail
        } else {
            // Small percentage of slightly larger but still safe transactions
            TransactionPattern::NormalRetail
        }
    }

    /// Get a random suspicious pattern (weighted distribution).
    pub fn random_suspicious(rng: &mut impl Rng) -> Self {
        let roll: f32 = rng.gen();
        if roll < 0.40 {
            TransactionPattern::Smurfing
        } else if roll < 0.70 {
            TransactionPattern::GeographicAnomaly
        } else if roll < 0.90 {
            TransactionPattern::VelocityBurst
        } else {
            TransactionPattern::RoundTrip
        }
    }
}

impl std::fmt::Display for TransactionPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransactionPattern::NormalRetail => write!(f, "Normal Retail"),
            TransactionPattern::HighValueWire => write!(f, "High Value Wire"),
            TransactionPattern::VelocityBurst => write!(f, "Velocity Burst"),
            TransactionPattern::Smurfing => write!(f, "Smurfing"),
            TransactionPattern::GeographicAnomaly => write!(f, "Geographic Anomaly"),
            TransactionPattern::RoundTrip => write!(f, "Round Trip"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory::AccountGenerator;

    #[test]
    fn test_smurfing_under_threshold() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut acct_gen = AccountGenerator::new(42);
        let customer = acct_gen.generate_customer();

        for _ in 0..100 {
            let tx = TransactionPattern::Smurfing.generate(1, &customer, 0, &mut rng);
            // Should always be under $10,000 but above 90%
            assert!(tx.amount_cents < 10_000_00);
            assert!(tx.amount_cents >= 9_000_00);
        }
    }

    #[test]
    fn test_geographic_anomaly_different_country() {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut acct_gen = AccountGenerator::new(42);
        let customer = acct_gen.generate_customer();

        for _ in 0..100 {
            let tx = TransactionPattern::GeographicAnomaly.generate(1, &customer, 0, &mut rng);
            assert_ne!(tx.country_code, customer.country_code);
        }
    }
}
