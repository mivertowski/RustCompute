//! Customer risk profile types.

use bytemuck::Zeroable;

/// Customer risk level classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u8)]
pub enum CustomerRiskLevel {
    /// Low risk - standard monitoring sufficient.
    #[default]
    Low = 0,
    /// Medium risk - enhanced monitoring recommended.
    Medium = 1,
    /// High risk - enhanced due diligence required.
    High = 2,
    /// Prohibited - customer cannot be onboarded.
    Prohibited = 3,
}

impl CustomerRiskLevel {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            CustomerRiskLevel::Low => "Low",
            CustomerRiskLevel::Medium => "Medium",
            CustomerRiskLevel::High => "High",
            CustomerRiskLevel::Prohibited => "Prohibited",
        }
    }

    /// Convert from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(CustomerRiskLevel::Low),
            1 => Some(CustomerRiskLevel::Medium),
            2 => Some(CustomerRiskLevel::High),
            3 => Some(CustomerRiskLevel::Prohibited),
            _ => None,
        }
    }
}

impl std::fmt::Display for CustomerRiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// 128-byte GPU-aligned customer risk profile.
///
/// Contains KYC/AML risk information for a customer.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(128))]
pub struct CustomerRiskProfile {
    /// Unique customer identifier.
    pub customer_id: u64, // 8 bytes, offset 0
    /// Overall risk level (cast to CustomerRiskLevel).
    pub risk_level: u8, // 1 byte, offset 8
    /// Numeric risk score (0-100).
    pub risk_score: u8, // 1 byte, offset 9
    /// Country code (ISO 3166-1 numeric).
    pub country_code: u16, // 2 bytes, offset 10
    /// Is Politically Exposed Person.
    pub is_pep: u8, // 1 byte, offset 12
    /// Requires Enhanced Due Diligence.
    pub requires_edd: u8, // 1 byte, offset 13
    /// Has adverse media mentions.
    pub has_adverse_media: u8, // 1 byte, offset 14
    /// Geographic risk component (0-100).
    pub geographic_risk: u8, // 1 byte, offset 15
    /// Business/industry risk component (0-100).
    pub business_risk: u8, // 1 byte, offset 16
    /// Behavioral risk component (0-100).
    pub behavioral_risk: u8, // 1 byte, offset 17
    /// Padding for alignment.
    _padding1: [u8; 2], // 2 bytes, offset 18-19
    /// Total transaction count (lifetime).
    pub transaction_count: u32, // 4 bytes, offset 20
    /// Total alert count (lifetime).
    pub alert_count: u32, // 4 bytes, offset 24
    /// Velocity window transaction count (current window).
    pub velocity_count: u32, // 4 bytes, offset 28
    /// Custom amount threshold for this customer (cents, 0 = use default).
    pub amount_threshold: u64, // 8 bytes, offset 32
    /// Custom velocity threshold (0 = use default).
    pub velocity_threshold: u32, // 4 bytes, offset 40
    /// Padding for alignment.
    _padding2: u32, // 4 bytes, offset 44
    /// Allowed destination countries (bitmask, bit N = country code N allowed).
    pub allowed_destinations: u64, // 8 bytes, offset 48
    /// Average monthly transaction volume (cents).
    pub avg_monthly_volume: u64, // 8 bytes, offset 56
    /// Last transaction timestamp (Unix epoch ms).
    pub last_transaction_ts: u64, // 8 bytes, offset 64
    /// Account creation timestamp (Unix epoch ms).
    pub created_ts: u64, // 8 bytes, offset 72
    /// Reserved for future use.
    _reserved: [u8; 48], // 48 bytes, offset 80-127
}

// Manual implementations since derive(Pod) is strict about padding
unsafe impl bytemuck::Zeroable for CustomerRiskProfile {}
unsafe impl bytemuck::Pod for CustomerRiskProfile {}

const _: () = assert!(std::mem::size_of::<CustomerRiskProfile>() == 128);

impl CustomerRiskProfile {
    /// Create a new customer profile with default settings.
    pub fn new(customer_id: u64, country_code: u16) -> Self {
        Self {
            customer_id,
            risk_level: CustomerRiskLevel::Low as u8,
            risk_score: 10,
            country_code,
            is_pep: 0,
            requires_edd: 0,
            has_adverse_media: 0,
            geographic_risk: 10,
            business_risk: 10,
            behavioral_risk: 10,
            _padding1: [0; 2],
            transaction_count: 0,
            alert_count: 0,
            velocity_count: 0,
            amount_threshold: 0,
            velocity_threshold: 0,
            _padding2: 0,
            allowed_destinations: !0, // All countries allowed by default
            avg_monthly_volume: 0,
            last_transaction_ts: 0,
            created_ts: 0,
            _reserved: [0; 48],
        }
    }

    /// Get the risk level enum.
    pub fn risk_level(&self) -> CustomerRiskLevel {
        CustomerRiskLevel::from_u8(self.risk_level).unwrap_or_default()
    }

    /// Check if customer is PEP.
    pub fn is_pep(&self) -> bool {
        self.is_pep != 0
    }

    /// Check if customer requires EDD.
    pub fn requires_edd(&self) -> bool {
        self.requires_edd != 0
    }

    /// Check if customer has adverse media.
    pub fn has_adverse_media(&self) -> bool {
        self.has_adverse_media != 0
    }

    /// Check if customer is high risk.
    pub fn is_high_risk(&self) -> bool {
        self.risk_level() >= CustomerRiskLevel::High || self.risk_score >= 70
    }

    /// Check if a destination country is allowed.
    pub fn is_destination_allowed(&self, country_code: u16) -> bool {
        if country_code >= 64 {
            // Countries beyond bitmask range are allowed by default
            true
        } else {
            (self.allowed_destinations >> country_code) & 1 == 1
        }
    }

    /// Increment velocity count.
    pub fn increment_velocity(&mut self) {
        self.velocity_count = self.velocity_count.saturating_add(1);
    }

    /// Reset velocity count (called at start of new window).
    pub fn reset_velocity(&mut self) {
        self.velocity_count = 0;
    }

    /// Increment transaction count.
    pub fn increment_transactions(&mut self) {
        self.transaction_count = self.transaction_count.saturating_add(1);
    }

    /// Increment alert count.
    pub fn increment_alerts(&mut self) {
        self.alert_count = self.alert_count.saturating_add(1);
    }
}

impl Default for CustomerRiskProfile {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_size() {
        assert_eq!(std::mem::size_of::<CustomerRiskProfile>(), 128);
    }

    #[test]
    fn test_destination_allowed() {
        let mut profile = CustomerRiskProfile::new(1, 1);
        profile.allowed_destinations = 0b1111; // Only countries 0-3 allowed

        assert!(profile.is_destination_allowed(0));
        assert!(profile.is_destination_allowed(1));
        assert!(profile.is_destination_allowed(2));
        assert!(profile.is_destination_allowed(3));
        assert!(!profile.is_destination_allowed(4));
        assert!(!profile.is_destination_allowed(10));
        // Countries beyond 63 are always allowed
        assert!(profile.is_destination_allowed(100));
    }
}
