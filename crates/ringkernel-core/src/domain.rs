//! Business domain classification for kernel messages.
//!
//! This module provides domain enumeration and traits for categorizing messages
//! by their business function. Domains enable:
//!
//! - Type ID range allocation per domain
//! - Domain-specific routing and observability
//! - Cross-domain access control
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::domain::{Domain, DomainMessage};
//!
//! // Messages with domain derive get auto-assigned type IDs
//! #[derive(RingMessage)]
//! #[ring_message(type_id = 1, domain = "OrderMatching")]
//! pub struct SubmitOrder {
//!     #[message(id)]
//!     id: MessageId,
//!     symbol: String,
//! }
//! // Final type ID = 500 (OrderMatching base) + 1 = 501
//! ```

use std::fmt;

/// Business domain classification for kernel messages.
///
/// Each domain has an assigned type ID range to prevent collisions:
/// - General: 0-99
/// - GraphAnalytics: 100-199
/// - StatisticalML: 200-299
/// - Compliance: 300-399
/// - RiskManagement: 400-499
/// - OrderMatching: 500-599
/// - MarketData: 600-699
/// - Settlement: 700-799
/// - Accounting: 800-899
/// - NetworkAnalysis: 900-999
/// - FraudDetection: 1000-1099
/// - TimeSeries: 1100-1199
/// - Simulation: 1200-1299
/// - Banking: 1300-1399
/// - BehavioralAnalytics: 1400-1499
/// - ProcessIntelligence: 1500-1599
/// - Clearing: 1600-1699
/// - TreasuryManagement: 1700-1799
/// - PaymentProcessing: 1800-1899
/// - FinancialAudit: 1900-1999
/// - Custom: 10000+
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u16)]
#[non_exhaustive]
pub enum Domain {
    /// General-purpose messages (type IDs: 0-99).
    #[default]
    General = 0,

    /// Graph analytics messages (type IDs: 100-199).
    /// Includes: PageRank, community detection, centrality measures.
    GraphAnalytics = 1,

    /// Statistical/ML messages (type IDs: 200-299).
    /// Includes: regression, clustering, classification.
    StatisticalML = 2,

    /// Compliance/regulatory messages (type IDs: 300-399).
    /// Includes: AML checks, KYC validation, regulatory reporting.
    Compliance = 3,

    /// Risk management messages (type IDs: 400-499).
    /// Includes: VaR calculation, stress testing, exposure analysis.
    RiskManagement = 4,

    /// Order matching messages (type IDs: 500-599).
    /// Includes: order submission, matching, cancellation.
    OrderMatching = 5,

    /// Market data messages (type IDs: 600-699).
    /// Includes: quotes, trades, order book updates.
    MarketData = 6,

    /// Settlement messages (type IDs: 700-799).
    /// Includes: trade settlement, netting, reconciliation.
    Settlement = 7,

    /// Accounting messages (type IDs: 800-899).
    /// Includes: journal entries, ledger updates, trial balance.
    Accounting = 8,

    /// Network analysis messages (type IDs: 900-999).
    /// Includes: transaction flow, counterparty analysis.
    NetworkAnalysis = 9,

    /// Fraud detection messages (type IDs: 1000-1099).
    /// Includes: anomaly detection, pattern matching.
    FraudDetection = 10,

    /// Time series messages (type IDs: 1100-1199).
    /// Includes: forecasting, trend analysis, seasonality.
    TimeSeries = 11,

    /// Simulation messages (type IDs: 1200-1299).
    /// Includes: Monte Carlo, scenario analysis, stress testing.
    Simulation = 12,

    /// Banking messages (type IDs: 1300-1399).
    /// Includes: account management, transfers, statements.
    Banking = 13,

    /// Behavioral analytics messages (type IDs: 1400-1499).
    /// Includes: user behavior, clickstream, session analysis.
    BehavioralAnalytics = 14,

    /// Process intelligence messages (type IDs: 1500-1599).
    /// Includes: process mining, DFG, conformance checking.
    ProcessIntelligence = 15,

    /// Clearing messages (type IDs: 1600-1699).
    /// Includes: CCP clearing, margin calculation, position netting.
    Clearing = 16,

    /// Treasury management messages (type IDs: 1700-1799).
    /// Includes: cash management, liquidity, FX hedging.
    TreasuryManagement = 17,

    /// Payment processing messages (type IDs: 1800-1899).
    /// Includes: payment initiation, routing, confirmation.
    PaymentProcessing = 18,

    /// Financial audit messages (type IDs: 1900-1999).
    /// Includes: audit trails, evidence gathering, compliance verification.
    FinancialAudit = 19,

    /// Custom domain (type IDs: 10000+).
    /// For application-specific domains not covered by predefined ones.
    Custom = 100,
}

impl Domain {
    /// Number of type IDs reserved per domain (except Custom).
    pub const RANGE_SIZE: u64 = 100;

    /// Base type ID for custom domains.
    pub const CUSTOM_BASE: u64 = 10000;

    /// Get the base type ID for this domain.
    ///
    /// Type IDs for messages in this domain should be: `base_type_id() + offset`
    /// where offset is 0-99.
    ///
    /// # Example
    ///
    /// ```
    /// use ringkernel_core::domain::Domain;
    ///
    /// assert_eq!(Domain::General.base_type_id(), 0);
    /// assert_eq!(Domain::OrderMatching.base_type_id(), 500);
    /// assert_eq!(Domain::Custom.base_type_id(), 10000);
    /// ```
    #[inline]
    pub const fn base_type_id(&self) -> u64 {
        match self {
            Self::General => 0,
            Self::GraphAnalytics => 100,
            Self::StatisticalML => 200,
            Self::Compliance => 300,
            Self::RiskManagement => 400,
            Self::OrderMatching => 500,
            Self::MarketData => 600,
            Self::Settlement => 700,
            Self::Accounting => 800,
            Self::NetworkAnalysis => 900,
            Self::FraudDetection => 1000,
            Self::TimeSeries => 1100,
            Self::Simulation => 1200,
            Self::Banking => 1300,
            Self::BehavioralAnalytics => 1400,
            Self::ProcessIntelligence => 1500,
            Self::Clearing => 1600,
            Self::TreasuryManagement => 1700,
            Self::PaymentProcessing => 1800,
            Self::FinancialAudit => 1900,
            Self::Custom => Self::CUSTOM_BASE,
        }
    }

    /// Get the maximum type ID for this domain.
    ///
    /// # Example
    ///
    /// ```
    /// use ringkernel_core::domain::Domain;
    ///
    /// assert_eq!(Domain::General.max_type_id(), 99);
    /// assert_eq!(Domain::OrderMatching.max_type_id(), 599);
    /// ```
    #[inline]
    pub const fn max_type_id(&self) -> u64 {
        match self {
            Self::Custom => u64::MAX,
            _ => self.base_type_id() + Self::RANGE_SIZE - 1,
        }
    }

    /// Check if a type ID is within this domain's range.
    ///
    /// # Example
    ///
    /// ```
    /// use ringkernel_core::domain::Domain;
    ///
    /// assert!(Domain::OrderMatching.contains_type_id(500));
    /// assert!(Domain::OrderMatching.contains_type_id(599));
    /// assert!(!Domain::OrderMatching.contains_type_id(600));
    /// ```
    #[inline]
    pub const fn contains_type_id(&self, type_id: u64) -> bool {
        type_id >= self.base_type_id() && type_id <= self.max_type_id()
    }

    /// Determine which domain a type ID belongs to.
    ///
    /// Returns `None` if the type ID doesn't match any standard domain.
    ///
    /// # Example
    ///
    /// ```
    /// use ringkernel_core::domain::Domain;
    ///
    /// assert_eq!(Domain::from_type_id(501), Some(Domain::OrderMatching));
    /// assert_eq!(Domain::from_type_id(10500), Some(Domain::Custom));
    /// ```
    pub const fn from_type_id(type_id: u64) -> Option<Self> {
        match type_id {
            0..=99 => Some(Self::General),
            100..=199 => Some(Self::GraphAnalytics),
            200..=299 => Some(Self::StatisticalML),
            300..=399 => Some(Self::Compliance),
            400..=499 => Some(Self::RiskManagement),
            500..=599 => Some(Self::OrderMatching),
            600..=699 => Some(Self::MarketData),
            700..=799 => Some(Self::Settlement),
            800..=899 => Some(Self::Accounting),
            900..=999 => Some(Self::NetworkAnalysis),
            1000..=1099 => Some(Self::FraudDetection),
            1100..=1199 => Some(Self::TimeSeries),
            1200..=1299 => Some(Self::Simulation),
            1300..=1399 => Some(Self::Banking),
            1400..=1499 => Some(Self::BehavioralAnalytics),
            1500..=1599 => Some(Self::ProcessIntelligence),
            1600..=1699 => Some(Self::Clearing),
            1700..=1799 => Some(Self::TreasuryManagement),
            1800..=1899 => Some(Self::PaymentProcessing),
            1900..=1999 => Some(Self::FinancialAudit),
            10000.. => Some(Self::Custom),
            _ => None,
        }
    }

    /// Parse domain from string (case-insensitive).
    ///
    /// Supports various naming conventions:
    /// - PascalCase: "OrderMatching"
    /// - snake_case: "order_matching"
    /// - lowercase: "ordermatching"
    /// - Short forms: "risk", "ml", "sim"
    ///
    /// # Example
    ///
    /// ```
    /// use ringkernel_core::domain::Domain;
    ///
    /// assert_eq!(Domain::from_str("OrderMatching"), Some(Domain::OrderMatching));
    /// assert_eq!(Domain::from_str("order_matching"), Some(Domain::OrderMatching));
    /// assert_eq!(Domain::from_str("risk"), Some(Domain::RiskManagement));
    /// assert_eq!(Domain::from_str("unknown"), None);
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        let normalized: String = s
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();

        match normalized.as_str() {
            "general" | "gen" => Some(Self::General),
            "graphanalytics" | "graph" => Some(Self::GraphAnalytics),
            "statisticalml" | "ml" | "machinelearning" => Some(Self::StatisticalML),
            "compliance" | "comp" | "regulatory" => Some(Self::Compliance),
            "riskmanagement" | "risk" => Some(Self::RiskManagement),
            "ordermatching" | "orders" | "order" | "matching" => Some(Self::OrderMatching),
            "marketdata" | "market" | "mktdata" => Some(Self::MarketData),
            "settlement" | "settle" => Some(Self::Settlement),
            "accounting" | "acct" | "ledger" => Some(Self::Accounting),
            "networkanalysis" | "network" | "netanalysis" => Some(Self::NetworkAnalysis),
            "frauddetection" | "fraud" | "aml" => Some(Self::FraudDetection),
            "timeseries" | "ts" | "temporal" => Some(Self::TimeSeries),
            "simulation" | "sim" | "montecarlo" => Some(Self::Simulation),
            "banking" | "bank" => Some(Self::Banking),
            "behavioralanalytics" | "behavioral" | "behavior" => Some(Self::BehavioralAnalytics),
            "processintelligence" | "process" | "processmining" => Some(Self::ProcessIntelligence),
            "clearing" | "ccp" => Some(Self::Clearing),
            "treasurymanagement" | "treasury" => Some(Self::TreasuryManagement),
            "paymentprocessing" | "payment" | "payments" => Some(Self::PaymentProcessing),
            "financialaudit" | "audit" => Some(Self::FinancialAudit),
            "custom" => Some(Self::Custom),
            _ => None,
        }
    }

    /// Get the domain name as a static string.
    ///
    /// Returns the PascalCase canonical name.
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::General => "General",
            Self::GraphAnalytics => "GraphAnalytics",
            Self::StatisticalML => "StatisticalML",
            Self::Compliance => "Compliance",
            Self::RiskManagement => "RiskManagement",
            Self::OrderMatching => "OrderMatching",
            Self::MarketData => "MarketData",
            Self::Settlement => "Settlement",
            Self::Accounting => "Accounting",
            Self::NetworkAnalysis => "NetworkAnalysis",
            Self::FraudDetection => "FraudDetection",
            Self::TimeSeries => "TimeSeries",
            Self::Simulation => "Simulation",
            Self::Banking => "Banking",
            Self::BehavioralAnalytics => "BehavioralAnalytics",
            Self::ProcessIntelligence => "ProcessIntelligence",
            Self::Clearing => "Clearing",
            Self::TreasuryManagement => "TreasuryManagement",
            Self::PaymentProcessing => "PaymentProcessing",
            Self::FinancialAudit => "FinancialAudit",
            Self::Custom => "Custom",
        }
    }

    /// Get all standard domains (excluding Custom).
    pub const fn all_standard() -> &'static [Domain] {
        &[
            Self::General,
            Self::GraphAnalytics,
            Self::StatisticalML,
            Self::Compliance,
            Self::RiskManagement,
            Self::OrderMatching,
            Self::MarketData,
            Self::Settlement,
            Self::Accounting,
            Self::NetworkAnalysis,
            Self::FraudDetection,
            Self::TimeSeries,
            Self::Simulation,
            Self::Banking,
            Self::BehavioralAnalytics,
            Self::ProcessIntelligence,
            Self::Clearing,
            Self::TreasuryManagement,
            Self::PaymentProcessing,
            Self::FinancialAudit,
        ]
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for Domain {
    type Err = DomainParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Domain::from_str(s).ok_or_else(|| DomainParseError(s.to_string()))
    }
}

/// Error returned when parsing an invalid domain string.
#[derive(Debug, Clone)]
pub struct DomainParseError(pub String);

impl fmt::Display for DomainParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown domain: '{}'", self.0)
    }
}

impl std::error::Error for DomainParseError {}

/// Trait for messages that belong to a specific business domain.
///
/// This trait is automatically implemented by the `#[derive(RingMessage)]` macro
/// when a `domain` attribute is specified.
///
/// # Example
///
/// ```ignore
/// #[derive(RingMessage)]
/// #[ring_message(type_id = 1, domain = "OrderMatching")]
/// pub struct SubmitOrder {
///     #[message(id)]
///     id: MessageId,
///     symbol: String,
/// }
///
/// // Auto-generated:
/// impl DomainMessage for SubmitOrder {
///     fn domain() -> Domain { Domain::OrderMatching }
/// }
/// ```
pub trait DomainMessage: crate::message::RingMessage {
    /// Get the domain this message belongs to.
    fn domain() -> Domain;

    /// Get the type ID offset within the domain (0-99).
    ///
    /// This is calculated as: `message_type() - domain().base_type_id()`
    fn domain_type_id() -> u64 {
        Self::message_type().saturating_sub(Self::domain().base_type_id())
    }

    /// Check if this message type is within its domain's valid range.
    fn is_valid_domain_type_id() -> bool {
        Self::domain().contains_type_id(Self::message_type())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_base_type_ids() {
        assert_eq!(Domain::General.base_type_id(), 0);
        assert_eq!(Domain::GraphAnalytics.base_type_id(), 100);
        assert_eq!(Domain::StatisticalML.base_type_id(), 200);
        assert_eq!(Domain::OrderMatching.base_type_id(), 500);
        assert_eq!(Domain::Custom.base_type_id(), 10000);
    }

    #[test]
    fn test_domain_max_type_ids() {
        assert_eq!(Domain::General.max_type_id(), 99);
        assert_eq!(Domain::OrderMatching.max_type_id(), 599);
        assert_eq!(Domain::Custom.max_type_id(), u64::MAX);
    }

    #[test]
    fn test_domain_contains_type_id() {
        assert!(Domain::General.contains_type_id(0));
        assert!(Domain::General.contains_type_id(99));
        assert!(!Domain::General.contains_type_id(100));

        assert!(Domain::OrderMatching.contains_type_id(500));
        assert!(Domain::OrderMatching.contains_type_id(599));
        assert!(!Domain::OrderMatching.contains_type_id(600));

        assert!(Domain::Custom.contains_type_id(10000));
        assert!(Domain::Custom.contains_type_id(u64::MAX));
    }

    #[test]
    fn test_domain_from_type_id() {
        assert_eq!(Domain::from_type_id(0), Some(Domain::General));
        assert_eq!(Domain::from_type_id(50), Some(Domain::General));
        assert_eq!(Domain::from_type_id(99), Some(Domain::General));
        assert_eq!(Domain::from_type_id(100), Some(Domain::GraphAnalytics));
        assert_eq!(Domain::from_type_id(501), Some(Domain::OrderMatching));
        assert_eq!(Domain::from_type_id(10500), Some(Domain::Custom));
        assert_eq!(Domain::from_type_id(2500), None); // Gap in range
    }

    #[test]
    fn test_domain_from_str() {
        // PascalCase
        assert_eq!(
            Domain::from_str("OrderMatching"),
            Some(Domain::OrderMatching)
        );
        assert_eq!(
            Domain::from_str("RiskManagement"),
            Some(Domain::RiskManagement)
        );

        // snake_case
        assert_eq!(
            Domain::from_str("order_matching"),
            Some(Domain::OrderMatching)
        );
        assert_eq!(
            Domain::from_str("risk_management"),
            Some(Domain::RiskManagement)
        );

        // lowercase
        assert_eq!(
            Domain::from_str("ordermatching"),
            Some(Domain::OrderMatching)
        );

        // Short forms
        assert_eq!(Domain::from_str("risk"), Some(Domain::RiskManagement));
        assert_eq!(Domain::from_str("ml"), Some(Domain::StatisticalML));
        assert_eq!(Domain::from_str("sim"), Some(Domain::Simulation));

        // Unknown
        assert_eq!(Domain::from_str("unknown"), None);
        assert_eq!(Domain::from_str(""), None);
    }

    #[test]
    fn test_domain_as_str() {
        assert_eq!(Domain::General.as_str(), "General");
        assert_eq!(Domain::OrderMatching.as_str(), "OrderMatching");
        assert_eq!(Domain::RiskManagement.as_str(), "RiskManagement");
    }

    #[test]
    fn test_domain_display() {
        assert_eq!(format!("{}", Domain::OrderMatching), "OrderMatching");
    }

    #[test]
    fn test_domain_default() {
        assert_eq!(Domain::default(), Domain::General);
    }

    #[test]
    fn test_domain_all_standard() {
        let all = Domain::all_standard();
        assert_eq!(all.len(), 20);
        assert!(all.contains(&Domain::General));
        assert!(all.contains(&Domain::OrderMatching));
        assert!(!all.contains(&Domain::Custom));
    }

    #[test]
    fn test_domain_ranges_no_overlap() {
        let domains = Domain::all_standard();
        for (i, d1) in domains.iter().enumerate() {
            for d2 in domains.iter().skip(i + 1) {
                // Check that ranges don't overlap
                assert!(
                    d1.max_type_id() < d2.base_type_id() || d2.max_type_id() < d1.base_type_id(),
                    "Domains {:?} and {:?} have overlapping ranges",
                    d1,
                    d2
                );
            }
        }
    }

    #[test]
    fn test_std_from_str() {
        let domain: Domain = "OrderMatching".parse().unwrap();
        assert_eq!(domain, Domain::OrderMatching);

        let err = "invalid".parse::<Domain>().unwrap_err();
        assert!(err.to_string().contains("unknown domain"));
    }
}
