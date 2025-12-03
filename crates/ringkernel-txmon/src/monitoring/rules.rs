//! Compliance monitoring rules.
//!
//! This module implements the core transaction monitoring logic, ported from
//! the C# TransactionMonitoringKernel.

use crate::types::{
    AlertSeverity, AlertType, CustomerRiskLevel, CustomerRiskProfile, MonitoringAlert, Transaction,
};

/// Configuration for monitoring rules.
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Default amount threshold in cents ($10,000 = 1_000_000 cents).
    pub amount_threshold: u64,
    /// Default velocity threshold (transactions per window).
    pub velocity_threshold: u32,
    /// Structuring detection threshold percentage (e.g., 90 = 90% of amount threshold).
    pub structuring_threshold_pct: u8,
    /// Minimum velocity count for structuring detection.
    pub structuring_min_velocity: u32,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            amount_threshold: 10_000_00, // $10,000 in cents
            velocity_threshold: 10,       // 10 transactions per window
            structuring_threshold_pct: 90,
            structuring_min_velocity: 3,
        }
    }
}

/// Monitor a single transaction and generate any applicable alerts.
///
/// This is a direct port of the C# TransactionMonitoringKernel.MonitorTransaction logic:
/// 1. Check velocity breach (too many transactions in time window)
/// 2. Check amount threshold (single large transaction)
/// 3. Check structured transactions (smurfing - amounts just under threshold)
/// 4. Check geographic anomaly (unusual destination country)
pub fn monitor_transaction(
    tx: &Transaction,
    profile: &CustomerRiskProfile,
    config: &MonitoringConfig,
    alert_id_start: u64,
) -> Vec<MonitoringAlert> {
    let mut alerts = Vec::new();
    let mut alert_id = alert_id_start;

    // Determine thresholds (customer-specific or default)
    let amount_threshold = if profile.amount_threshold > 0 {
        profile.amount_threshold
    } else {
        config.amount_threshold
    };

    let velocity_threshold = if profile.velocity_threshold > 0 {
        profile.velocity_threshold
    } else {
        config.velocity_threshold
    };

    // 1. Velocity breach check
    if profile.velocity_count > velocity_threshold {
        let severity = calculate_velocity_severity(profile.velocity_count, velocity_threshold);
        let mut alert = MonitoringAlert::new(
            alert_id,
            tx.transaction_id,
            tx.customer_id,
            AlertType::VelocityBreach,
            severity,
            tx.amount_cents,
            tx.timestamp,
        );
        alert.velocity_count = profile.velocity_count;
        alert.risk_score = calculate_risk_score(profile, &AlertType::VelocityBreach);
        alerts.push(alert);
        alert_id += 1;
    }

    // 2. Amount threshold check
    if tx.amount_cents > amount_threshold {
        let severity = calculate_amount_severity(tx.amount_cents, amount_threshold);
        let mut alert = MonitoringAlert::new(
            alert_id,
            tx.transaction_id,
            tx.customer_id,
            AlertType::AmountThreshold,
            severity,
            tx.amount_cents,
            tx.timestamp,
        );
        alert.risk_score = calculate_risk_score(profile, &AlertType::AmountThreshold);
        alerts.push(alert);
        alert_id += 1;
    }

    // 3. Structured transaction (smurfing) check
    // Amount > 90% of threshold AND velocity >= 3
    let structuring_threshold =
        (amount_threshold * config.structuring_threshold_pct as u64) / 100;

    if tx.amount_cents > structuring_threshold
        && tx.amount_cents <= amount_threshold
        && profile.velocity_count >= config.structuring_min_velocity
    {
        let mut alert = MonitoringAlert::new(
            alert_id,
            tx.transaction_id,
            tx.customer_id,
            AlertType::StructuredTransaction,
            AlertSeverity::High,
            tx.amount_cents,
            tx.timestamp,
        );
        alert.velocity_count = profile.velocity_count;
        alert.risk_score = calculate_risk_score(profile, &AlertType::StructuredTransaction);
        alerts.push(alert);
        alert_id += 1;
    }

    // 4. Geographic anomaly check
    if tx.country_code != profile.country_code {
        let is_allowed = profile.is_destination_allowed(tx.country_code);
        if !is_allowed {
            let severity = if is_high_risk_country(tx.country_code) {
                AlertSeverity::High
            } else {
                AlertSeverity::Medium
            };

            let mut alert = MonitoringAlert::new(
                alert_id,
                tx.transaction_id,
                tx.customer_id,
                AlertType::GeographicAnomaly,
                severity,
                tx.amount_cents,
                tx.timestamp,
            );
            alert.country_code = tx.country_code;
            alert.risk_score = calculate_risk_score(profile, &AlertType::GeographicAnomaly);
            alerts.push(alert);
            alert_id += 1;
        }
    }

    // 5. PEP-related check (any transaction from PEP gets flagged)
    if profile.is_pep() && tx.amount_cents > 5_000_00 {
        // Flag PEP transactions over $5,000
        let mut alert = MonitoringAlert::new(
            alert_id,
            tx.transaction_id,
            tx.customer_id,
            AlertType::PEPRelated,
            AlertSeverity::High,
            tx.amount_cents,
            tx.timestamp,
        );
        alert.risk_score = calculate_risk_score(profile, &AlertType::PEPRelated);
        alerts.push(alert);
        // alert_id += 1; // Unused but kept for consistency
    }

    alerts
}

/// Calculate severity based on how much velocity exceeds threshold.
fn calculate_velocity_severity(velocity: u32, threshold: u32) -> AlertSeverity {
    let ratio = velocity as f32 / threshold as f32;
    if ratio >= 3.0 {
        AlertSeverity::Critical
    } else if ratio >= 2.0 {
        AlertSeverity::High
    } else {
        AlertSeverity::Medium
    }
}

/// Calculate severity based on how much amount exceeds threshold.
fn calculate_amount_severity(amount: u64, threshold: u64) -> AlertSeverity {
    let ratio = amount as f64 / threshold as f64;
    if ratio >= 5.0 {
        AlertSeverity::Critical
    } else if ratio >= 2.0 {
        AlertSeverity::High
    } else {
        AlertSeverity::Medium
    }
}

/// Calculate risk score based on customer profile and alert type.
fn calculate_risk_score(profile: &CustomerRiskProfile, alert_type: &AlertType) -> u32 {
    let base_score = profile.risk_score as u32;

    // Add modifiers based on alert type
    let type_modifier = match alert_type {
        AlertType::SanctionsHit => 50,
        AlertType::PEPRelated => 30,
        AlertType::StructuredTransaction => 25,
        AlertType::VelocityBreach => 20,
        AlertType::AmountThreshold => 15,
        AlertType::GeographicAnomaly => 15,
        _ => 10,
    };

    // Add modifier for customer risk level
    let risk_modifier = match profile.risk_level() {
        CustomerRiskLevel::Prohibited => 30,
        CustomerRiskLevel::High => 20,
        CustomerRiskLevel::Medium => 10,
        CustomerRiskLevel::Low => 0,
    };

    // PEP and EDD modifiers
    let pep_modifier = if profile.is_pep() { 15 } else { 0 };
    let edd_modifier = if profile.requires_edd() { 10 } else { 0 };

    // Cap at 100
    (base_score + type_modifier + risk_modifier + pep_modifier + edd_modifier).min(100)
}

/// Check if a country code is considered high-risk.
fn is_high_risk_country(country_code: u16) -> bool {
    // Simplified high-risk country list (using our arbitrary codes)
    // 7 = RU, 6 = CN in our simplified mapping
    matches!(country_code, 6 | 7)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_customer(velocity: u32, country: u16) -> CustomerRiskProfile {
        let mut p = CustomerRiskProfile::new(1, country);
        p.velocity_count = velocity;
        p
    }

    fn make_transaction(amount_cents: u64, country: u16) -> Transaction {
        Transaction::new(1, 1, amount_cents, country, 0)
    }

    #[test]
    fn test_velocity_breach() {
        let config = MonitoringConfig::default();
        let customer = make_customer(15, 1); // 15 transactions, threshold is 10
        let tx = make_transaction(100_00, 1);

        let alerts = monitor_transaction(&tx, &customer, &config, 1);

        assert!(alerts.iter().any(|a| a.alert_type() == AlertType::VelocityBreach));
    }

    #[test]
    fn test_no_velocity_breach() {
        let config = MonitoringConfig::default();
        let customer = make_customer(5, 1); // 5 transactions, under threshold
        let tx = make_transaction(100_00, 1);

        let alerts = monitor_transaction(&tx, &customer, &config, 1);

        assert!(!alerts.iter().any(|a| a.alert_type() == AlertType::VelocityBreach));
    }

    #[test]
    fn test_amount_threshold() {
        let config = MonitoringConfig::default();
        let customer = make_customer(0, 1);
        let tx = make_transaction(15_000_00, 1); // $15,000, over threshold

        let alerts = monitor_transaction(&tx, &customer, &config, 1);

        assert!(alerts.iter().any(|a| a.alert_type() == AlertType::AmountThreshold));
    }

    #[test]
    fn test_structured_transaction() {
        let config = MonitoringConfig::default();
        let customer = make_customer(5, 1); // 5 transactions (>= min_velocity of 3)
        let tx = make_transaction(9_500_00, 1); // $9,500 (95% of threshold)

        let alerts = monitor_transaction(&tx, &customer, &config, 1);

        assert!(alerts.iter().any(|a| a.alert_type() == AlertType::StructuredTransaction));
    }

    #[test]
    fn test_no_structured_without_velocity() {
        let config = MonitoringConfig::default();
        let customer = make_customer(1, 1); // Only 1 transaction
        let tx = make_transaction(9_500_00, 1); // $9,500

        let alerts = monitor_transaction(&tx, &customer, &config, 1);

        // Should not trigger structuring because velocity is too low
        assert!(!alerts.iter().any(|a| a.alert_type() == AlertType::StructuredTransaction));
    }

    #[test]
    fn test_geographic_anomaly() {
        let config = MonitoringConfig::default();
        let mut customer = make_customer(0, 1); // US customer
        customer.allowed_destinations = 0b11; // Only countries 0 and 1 allowed
        let tx = make_transaction(5_000_00, 7); // Transaction to country 7 (not allowed)

        let alerts = monitor_transaction(&tx, &customer, &config, 1);

        assert!(alerts.iter().any(|a| a.alert_type() == AlertType::GeographicAnomaly));
    }

    #[test]
    fn test_severity_calculation() {
        assert_eq!(
            calculate_velocity_severity(30, 10),
            AlertSeverity::Critical
        ); // 3x
        assert_eq!(calculate_velocity_severity(20, 10), AlertSeverity::High); // 2x
        assert_eq!(calculate_velocity_severity(15, 10), AlertSeverity::Medium); // 1.5x
    }
}
