//! Alert types and monitoring alert structure.

use bytemuck::Zeroable;
use iced::Color;

/// Types of compliance alerts that can be triggered during transaction monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum AlertType {
    /// Transaction velocity exceeds threshold (too many in time window).
    #[default]
    VelocityBreach = 0,
    /// Single transaction amount exceeds configured threshold.
    AmountThreshold = 1,
    /// Geographic anomaly (unusual destination country).
    GeographicAnomaly = 2,
    /// Structured transactions to avoid reporting (smurfing).
    StructuredTransaction = 3,
    /// Circular trading pattern detected.
    CircularTrading = 4,
    /// Entity matched against sanctions list.
    SanctionsHit = 5,
    /// Politically Exposed Person related transaction.
    PEPRelated = 6,
    /// Adverse media mention found.
    AdverseMedia = 7,
    /// Unusual pattern detected (generic).
    UnusualPattern = 8,
}

impl AlertType {
    /// Get display name for the alert type.
    pub fn name(&self) -> &'static str {
        match self {
            AlertType::VelocityBreach => "Velocity Breach",
            AlertType::AmountThreshold => "Amount Threshold",
            AlertType::GeographicAnomaly => "Geographic Anomaly",
            AlertType::StructuredTransaction => "Structured Transaction",
            AlertType::CircularTrading => "Circular Trading",
            AlertType::SanctionsHit => "Sanctions Hit",
            AlertType::PEPRelated => "PEP Related",
            AlertType::AdverseMedia => "Adverse Media",
            AlertType::UnusualPattern => "Unusual Pattern",
        }
    }

    /// Get short code for display.
    pub fn code(&self) -> &'static str {
        match self {
            AlertType::VelocityBreach => "VEL",
            AlertType::AmountThreshold => "AMT",
            AlertType::GeographicAnomaly => "GEO",
            AlertType::StructuredTransaction => "STR",
            AlertType::CircularTrading => "CIR",
            AlertType::SanctionsHit => "SAN",
            AlertType::PEPRelated => "PEP",
            AlertType::AdverseMedia => "ADV",
            AlertType::UnusualPattern => "UNS",
        }
    }

    /// Convert from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(AlertType::VelocityBreach),
            1 => Some(AlertType::AmountThreshold),
            2 => Some(AlertType::GeographicAnomaly),
            3 => Some(AlertType::StructuredTransaction),
            4 => Some(AlertType::CircularTrading),
            5 => Some(AlertType::SanctionsHit),
            6 => Some(AlertType::PEPRelated),
            7 => Some(AlertType::AdverseMedia),
            8 => Some(AlertType::UnusualPattern),
            _ => None,
        }
    }
}

impl std::fmt::Display for AlertType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Severity level of a compliance alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u8)]
pub enum AlertSeverity {
    /// Low severity - informational, minimal risk.
    #[default]
    Low = 0,
    /// Medium severity - requires review but not urgent.
    Medium = 1,
    /// High severity - requires prompt attention.
    High = 2,
    /// Critical severity - immediate action required.
    Critical = 3,
}

impl AlertSeverity {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            AlertSeverity::Low => "LOW",
            AlertSeverity::Medium => "MEDIUM",
            AlertSeverity::High => "HIGH",
            AlertSeverity::Critical => "CRITICAL",
        }
    }

    /// Get color for GUI display.
    pub fn color(&self) -> Color {
        match self {
            AlertSeverity::Low => Color::from_rgb(0.4, 0.7, 0.4),      // Green
            AlertSeverity::Medium => Color::from_rgb(0.9, 0.8, 0.2),   // Yellow
            AlertSeverity::High => Color::from_rgb(0.9, 0.5, 0.1),     // Orange
            AlertSeverity::Critical => Color::from_rgb(0.9, 0.2, 0.2), // Red
        }
    }

    /// Get indicator character for compact display.
    pub fn indicator(&self) -> char {
        match self {
            AlertSeverity::Low => '\u{25CB}',      // ○
            AlertSeverity::Medium => '\u{25D0}',   // ◐
            AlertSeverity::High => '\u{25CF}',     // ●
            AlertSeverity::Critical => '\u{25C9}', // ◉
        }
    }

    /// Convert from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(AlertSeverity::Low),
            1 => Some(AlertSeverity::Medium),
            2 => Some(AlertSeverity::High),
            3 => Some(AlertSeverity::Critical),
            _ => None,
        }
    }
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Alert workflow status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum AlertStatus {
    /// Newly created alert.
    #[default]
    New = 0,
    /// Alert has been acknowledged.
    Acknowledged = 1,
    /// Alert is under review by analyst.
    UnderReview = 2,
    /// Alert has been escalated.
    Escalated = 3,
    /// Alert has been closed.
    Closed = 4,
}

impl AlertStatus {
    /// Convert from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(AlertStatus::New),
            1 => Some(AlertStatus::Acknowledged),
            2 => Some(AlertStatus::UnderReview),
            3 => Some(AlertStatus::Escalated),
            4 => Some(AlertStatus::Closed),
            _ => None,
        }
    }
}

/// 128-byte GPU-aligned monitoring alert structure.
///
/// Contains all information about a triggered compliance alert.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(128))]
pub struct MonitoringAlert {
    /// Unique alert identifier.
    pub alert_id: u64,            // 8 bytes, offset 0
    /// Related transaction identifier.
    pub transaction_id: u64,      // 8 bytes, offset 8
    /// Customer identifier who triggered the alert.
    pub customer_id: u64,         // 8 bytes, offset 16
    /// Alert type (cast to AlertType).
    pub alert_type: u8,           // 1 byte, offset 24
    /// Alert severity (cast to AlertSeverity).
    pub severity: u8,             // 1 byte, offset 25
    /// Alert status (cast to AlertStatus).
    pub status: u8,               // 1 byte, offset 26
    /// Padding bytes.
    _padding1: [u8; 5],           // 5 bytes, offset 27-31
    /// Transaction amount that triggered alert (cents).
    pub amount_cents: u64,        // 8 bytes, offset 32
    /// Computed risk score for this alert (0-100).
    pub risk_score: u32,          // 4 bytes, offset 40
    /// Velocity count at time of alert.
    pub velocity_count: u32,      // 4 bytes, offset 44
    /// Timestamp when alert was raised (Unix epoch ms).
    pub timestamp: u64,           // 8 bytes, offset 48
    /// Destination country code (if relevant).
    pub country_code: u16,        // 2 bytes, offset 56
    /// Additional flags (bitmask).
    pub flags: u16,               // 2 bytes, offset 58
    /// Padding for alignment.
    _padding2: [u8; 4],           // 4 bytes, offset 60-63
    /// Reserved for future use.
    _reserved: [u8; 64],          // 64 bytes, offset 64-127
}

// Manual implementations since we can't derive Pod due to padding sensitivity
unsafe impl bytemuck::Zeroable for MonitoringAlert {}
unsafe impl bytemuck::Pod for MonitoringAlert {}

const _: () = assert!(std::mem::size_of::<MonitoringAlert>() == 128);

impl MonitoringAlert {
    /// Create a new monitoring alert.
    pub fn new(
        alert_id: u64,
        transaction_id: u64,
        customer_id: u64,
        alert_type: AlertType,
        severity: AlertSeverity,
        amount_cents: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            alert_id,
            transaction_id,
            customer_id,
            alert_type: alert_type as u8,
            severity: severity as u8,
            status: AlertStatus::New as u8,
            _padding1: [0; 5],
            amount_cents,
            risk_score: 0,
            velocity_count: 0,
            timestamp,
            country_code: 0,
            flags: 0,
            _padding2: [0; 4],
            _reserved: [0; 64],
        }
    }

    /// Get the alert type enum.
    pub fn alert_type(&self) -> AlertType {
        AlertType::from_u8(self.alert_type).unwrap_or_default()
    }

    /// Get the severity enum.
    pub fn severity(&self) -> AlertSeverity {
        AlertSeverity::from_u8(self.severity).unwrap_or_default()
    }

    /// Get the status enum.
    pub fn status(&self) -> AlertStatus {
        AlertStatus::from_u8(self.status).unwrap_or_default()
    }

    /// Format amount as currency string.
    pub fn format_amount(&self) -> String {
        let dollars = self.amount_cents / 100;
        let cents = self.amount_cents % 100;
        format!("${}.{:02}", dollars, cents)
    }

    /// Check if this alert requires immediate attention.
    pub fn requires_immediate_action(&self) -> bool {
        self.severity() == AlertSeverity::Critical
            || self.alert_type() == AlertType::SanctionsHit
            || self.alert_type() == AlertType::PEPRelated
    }
}

impl Default for MonitoringAlert {
    fn default() -> Self {
        Self::zeroed()
    }
}
