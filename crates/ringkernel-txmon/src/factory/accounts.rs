//! Customer account generation.

use crate::types::{CustomerRiskLevel, CustomerRiskProfile};
use rand::prelude::*;
use rand::rngs::SmallRng;

/// Generator for synthetic customer accounts.
pub struct AccountGenerator {
    rng: SmallRng,
    next_id: u64,
}

impl AccountGenerator {
    /// Create a new account generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
            next_id: 1,
        }
    }

    /// Generate a batch of customer profiles.
    pub fn generate_customers(&mut self, count: u32) -> Vec<CustomerRiskProfile> {
        (0..count).map(|_| self.generate_customer()).collect()
    }

    /// Generate a single customer profile.
    pub fn generate_customer(&mut self) -> CustomerRiskProfile {
        let customer_id = self.next_id;
        self.next_id += 1;

        // Country distribution (weighted towards US)
        let country_code = self.random_country();

        let mut profile = CustomerRiskProfile::new(customer_id, country_code);

        // Risk level distribution: 70% Low, 20% Medium, 8% High, 2% Prohibited
        let risk_roll: f32 = self.rng.gen();
        profile.risk_level = if risk_roll < 0.70 {
            CustomerRiskLevel::Low as u8
        } else if risk_roll < 0.90 {
            CustomerRiskLevel::Medium as u8
        } else if risk_roll < 0.98 {
            CustomerRiskLevel::High as u8
        } else {
            CustomerRiskLevel::Prohibited as u8
        };

        // Risk score based on level
        profile.risk_score = match CustomerRiskLevel::from_u8(profile.risk_level).unwrap() {
            CustomerRiskLevel::Low => self.rng.gen_range(5..30),
            CustomerRiskLevel::Medium => self.rng.gen_range(30..60),
            CustomerRiskLevel::High => self.rng.gen_range(60..85),
            CustomerRiskLevel::Prohibited => self.rng.gen_range(85..100),
        };

        // PEP status (1% chance) - but disabled by default to avoid unexpected alerts
        // PEP alerts are triggered by suspicious_rate > 0 in the monitoring engine
        profile.is_pep = 0; // PEP status now controlled by suspicious transactions only

        // EDD requirement (5% chance, higher if high risk)
        let edd_chance = if profile.risk_level >= CustomerRiskLevel::High as u8 {
            0.30
        } else {
            0.05
        };
        profile.requires_edd = if self.rng.gen::<f32>() < edd_chance {
            1
        } else {
            0
        };

        // Adverse media (2% chance)
        profile.has_adverse_media = if self.rng.gen::<f32>() < 0.02 { 1 } else { 0 };

        // Component risk scores
        profile.geographic_risk = self.rng.gen_range(5..50);
        profile.business_risk = self.rng.gen_range(5..50);
        profile.behavioral_risk = self.rng.gen_range(5..50);

        // Custom thresholds (most use defaults, some have custom)
        if self.rng.gen::<f32>() < 0.1 {
            // 10% have custom amount threshold
            profile.amount_threshold = self.rng.gen_range(500_000..5_000_000);
        }
        if self.rng.gen::<f32>() < 0.1 {
            // 10% have custom velocity threshold
            profile.velocity_threshold = self.rng.gen_range(5..50);
        }

        // Allowed destinations - all countries allowed by default
        // Destination restrictions would cause unexpected geographic anomaly alerts
        // Geographic anomaly alerts are instead triggered by suspicious transaction patterns
        profile.allowed_destinations = !0; // All countries allowed

        // Average monthly volume
        profile.avg_monthly_volume = self.rng.gen_range(100_000..50_000_000);

        // Created timestamp (random time in past year)
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let one_year_ms = 365 * 24 * 60 * 60 * 1000;
        profile.created_ts = now_ms - self.rng.gen_range(0..one_year_ms);

        profile
    }

    /// Generate a random country code with weighted distribution.
    fn random_country(&mut self) -> u16 {
        // Weighted distribution: 60% US, 10% UK, 30% others
        let roll: f32 = self.rng.gen();
        if roll < 0.60 {
            1 // US
        } else if roll < 0.70 {
            2 // UK
        } else {
            // Random from other countries
            self.rng.gen_range(3..16)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_customers() {
        let mut gen = AccountGenerator::new(42);
        let customers = gen.generate_customers(100);

        assert_eq!(customers.len(), 100);

        // Check IDs are sequential
        for (i, c) in customers.iter().enumerate() {
            assert_eq!(c.customer_id, (i + 1) as u64);
        }

        // Check risk distribution (should have some variety)
        let low_count = customers
            .iter()
            .filter(|c| c.risk_level == CustomerRiskLevel::Low as u8)
            .count();
        assert!(low_count > 50); // Should be majority low risk
    }

    #[test]
    fn test_pep_disabled_by_default() {
        let mut gen = AccountGenerator::new(42);
        let customers = gen.generate_customers(1000);

        let pep_count = customers.iter().filter(|c| c.is_pep != 0).count();
        // PEP is now disabled by default to prevent unexpected alerts
        assert_eq!(pep_count, 0);
    }
}
