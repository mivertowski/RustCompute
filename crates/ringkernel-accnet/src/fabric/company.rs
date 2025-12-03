//! Company archetypes for realistic synthetic data generation.
//!
//! Each archetype represents a distinct business model with unique
//! financial characteristics, transaction patterns, and seasonality.

use std::ops::Range;

/// Industry classification for company archetypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Industry {
    /// Retail: High volume, low margins, seasonal peaks
    Retail,
    /// SaaS: Subscription revenue, deferred recognition
    SaaS,
    /// Manufacturing: Complex supply chain, WIP inventory
    Manufacturing,
    /// Professional Services: Time-based billing
    ProfessionalServices,
    /// Financial Services: Complex instruments
    FinancialServices,
    /// Healthcare: Insurance billing, regulatory compliance
    Healthcare,
    /// Construction: Project-based, progress billing
    Construction,
    /// Education: Tuition cycles, grant accounting
    Education,
}

/// Company size classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompanySize {
    /// Small: <$10M revenue, <50 employees
    Small,
    /// Medium: $10M-$100M revenue, 50-500 employees
    Medium,
    /// Large: $100M-$1B revenue, 500-5000 employees
    Large,
    /// Enterprise: >$1B revenue, >5000 employees
    Enterprise,
}

impl CompanySize {
    /// Typical account count for this size.
    pub fn typical_account_count(&self) -> Range<usize> {
        match self {
            CompanySize::Small => 50..100,
            CompanySize::Medium => 100..200,
            CompanySize::Large => 200..400,
            CompanySize::Enterprise => 400..800,
        }
    }

    /// Typical daily transaction volume.
    pub fn typical_daily_transactions(&self) -> Range<u32> {
        match self {
            CompanySize::Small => 10..100,
            CompanySize::Medium => 100..500,
            CompanySize::Large => 500..2000,
            CompanySize::Enterprise => 2000..10000,
        }
    }
}

/// A company archetype defining financial behavior patterns.
#[derive(Debug, Clone)]
pub enum CompanyArchetype {
    /// Retail business with physical/online stores.
    Retail(RetailConfig),

    /// Software-as-a-Service subscription business.
    SaaS(SaaSConfig),

    /// Manufacturing with supply chain and inventory.
    Manufacturing(ManufacturingConfig),

    /// Professional services (consulting, legal, accounting).
    ProfessionalServices(ProfServicesConfig),

    /// Financial services (banking, investment, insurance).
    FinancialServices(FinServicesConfig),
}

/// Configuration for retail company archetype.
#[derive(Debug, Clone)]
pub struct RetailConfig {
    /// Number of store locations
    pub store_count: u32,
    /// Average transaction value
    pub avg_transaction: f64,
    /// Months with seasonal peaks (1-12)
    pub seasonal_peaks: Vec<u8>,
    /// Online vs physical split (0.0 - 1.0 = % online)
    pub online_ratio: f64,
    /// Inventory turnover rate (times per year)
    pub inventory_turnover: f64,
    /// Gross margin percentage
    pub gross_margin: f64,
    /// Company size
    pub size: CompanySize,
}

impl Default for RetailConfig {
    fn default() -> Self {
        Self {
            store_count: 5,
            avg_transaction: 75.0,
            seasonal_peaks: vec![11, 12], // Nov-Dec holiday season
            online_ratio: 0.3,
            inventory_turnover: 6.0,
            gross_margin: 0.35,
            size: CompanySize::Medium,
        }
    }
}

/// Configuration for SaaS company archetype.
#[derive(Debug, Clone)]
pub struct SaaSConfig {
    /// Monthly recurring revenue
    pub mrr: f64,
    /// Annual churn rate
    pub churn_rate: f64,
    /// Average contract length in months
    pub contract_length_months: u32,
    /// Mix of monthly vs annual contracts
    pub annual_contract_ratio: f64,
    /// Revenue recognition: point-in-time vs over-time
    pub deferred_revenue_ratio: f64,
    /// Company size
    pub size: CompanySize,
}

impl Default for SaaSConfig {
    fn default() -> Self {
        Self {
            mrr: 500_000.0,
            churn_rate: 0.05,
            contract_length_months: 12,
            annual_contract_ratio: 0.6,
            deferred_revenue_ratio: 0.8,
            size: CompanySize::Medium,
        }
    }
}

/// Configuration for manufacturing company archetype.
#[derive(Debug, Clone)]
pub struct ManufacturingConfig {
    /// Number of product lines
    pub product_lines: u32,
    /// Number of suppliers
    pub supplier_count: u32,
    /// Production cycles per month
    pub production_cycles_per_month: u32,
    /// Work-in-progress as % of inventory
    pub wip_ratio: f64,
    /// Raw materials lead time (days)
    pub lead_time_days: u32,
    /// Company size
    pub size: CompanySize,
}

impl Default for ManufacturingConfig {
    fn default() -> Self {
        Self {
            product_lines: 3,
            supplier_count: 25,
            production_cycles_per_month: 4,
            wip_ratio: 0.3,
            lead_time_days: 30,
            size: CompanySize::Large,
        }
    }
}

/// Configuration for professional services archetype.
#[derive(Debug, Clone)]
pub struct ProfServicesConfig {
    /// Number of billable employees
    pub billable_headcount: u32,
    /// Average hourly billing rate
    pub avg_hourly_rate: f64,
    /// Target utilization rate
    pub utilization_target: f64,
    /// Average project duration (days)
    pub avg_project_duration: u32,
    /// % of work billed on completion vs progress
    pub completion_billing_ratio: f64,
    /// Company size
    pub size: CompanySize,
}

impl Default for ProfServicesConfig {
    fn default() -> Self {
        Self {
            billable_headcount: 50,
            avg_hourly_rate: 200.0,
            utilization_target: 0.75,
            avg_project_duration: 45,
            completion_billing_ratio: 0.4,
            size: CompanySize::Medium,
        }
    }
}

/// Configuration for financial services archetype.
#[derive(Debug, Clone)]
pub struct FinServicesConfig {
    /// Assets under management
    pub aum: f64,
    /// Number of product types
    pub product_type_count: u32,
    /// Intercompany transaction ratio
    pub intercompany_ratio: f64,
    /// Regulatory capital ratio
    pub capital_ratio: f64,
    /// Company size
    pub size: CompanySize,
}

impl Default for FinServicesConfig {
    fn default() -> Self {
        Self {
            aum: 1_000_000_000.0,
            product_type_count: 5,
            intercompany_ratio: 0.15,
            capital_ratio: 0.12,
            size: CompanySize::Large,
        }
    }
}

impl CompanyArchetype {
    /// Create a standard retail company.
    pub fn retail_standard() -> Self {
        Self::Retail(RetailConfig::default())
    }

    /// Create a small retail company.
    pub fn retail_small() -> Self {
        Self::Retail(RetailConfig {
            store_count: 1,
            avg_transaction: 50.0,
            size: CompanySize::Small,
            ..Default::default()
        })
    }

    /// Create a standard SaaS company.
    pub fn saas_standard() -> Self {
        Self::SaaS(SaaSConfig::default())
    }

    /// Create a startup SaaS company.
    pub fn saas_startup() -> Self {
        Self::SaaS(SaaSConfig {
            mrr: 50_000.0,
            churn_rate: 0.08,
            size: CompanySize::Small,
            ..Default::default()
        })
    }

    /// Create a standard manufacturing company.
    pub fn manufacturing_standard() -> Self {
        Self::Manufacturing(ManufacturingConfig::default())
    }

    /// Create a standard professional services company.
    pub fn professional_services_standard() -> Self {
        Self::ProfessionalServices(ProfServicesConfig::default())
    }

    /// Create a standard financial services company.
    pub fn financial_services_standard() -> Self {
        Self::FinancialServices(FinServicesConfig::default())
    }

    /// Get the industry for this archetype.
    pub fn industry(&self) -> Industry {
        match self {
            CompanyArchetype::Retail(_) => Industry::Retail,
            CompanyArchetype::SaaS(_) => Industry::SaaS,
            CompanyArchetype::Manufacturing(_) => Industry::Manufacturing,
            CompanyArchetype::ProfessionalServices(_) => Industry::ProfessionalServices,
            CompanyArchetype::FinancialServices(_) => Industry::FinancialServices,
        }
    }

    /// Get the company size.
    pub fn size(&self) -> CompanySize {
        match self {
            CompanyArchetype::Retail(c) => c.size,
            CompanyArchetype::SaaS(c) => c.size,
            CompanyArchetype::Manufacturing(c) => c.size,
            CompanyArchetype::ProfessionalServices(c) => c.size,
            CompanyArchetype::FinancialServices(c) => c.size,
        }
    }

    /// Get display name for this archetype.
    pub fn display_name(&self) -> &'static str {
        match self {
            CompanyArchetype::Retail(_) => "Retail",
            CompanyArchetype::SaaS(_) => "SaaS",
            CompanyArchetype::Manufacturing(_) => "Manufacturing",
            CompanyArchetype::ProfessionalServices(_) => "Professional Services",
            CompanyArchetype::FinancialServices(_) => "Financial Services",
        }
    }

    /// Get a brief description.
    pub fn description(&self) -> &'static str {
        match self {
            CompanyArchetype::Retail(_) => {
                "High-volume retail with inventory management and seasonal patterns"
            }
            CompanyArchetype::SaaS(_) => {
                "Subscription software with deferred revenue and monthly billing cycles"
            }
            CompanyArchetype::Manufacturing(_) => {
                "Manufacturing with supply chain, WIP inventory, and production cycles"
            }
            CompanyArchetype::ProfessionalServices(_) => {
                "Time-based billing with utilization tracking and project accounting"
            }
            CompanyArchetype::FinancialServices(_) => {
                "Complex financial instruments with intercompany transactions"
            }
        }
    }

    /// Get typical transaction patterns for this archetype.
    pub fn transaction_patterns(&self) -> Vec<TransactionPattern> {
        match self {
            CompanyArchetype::Retail(_) => vec![
                TransactionPattern::new("Sales", 0.40, 50.0..500.0),
                TransactionPattern::new("Inventory Purchase", 0.25, 500.0..10000.0),
                TransactionPattern::new("Payroll", 0.10, 5000.0..50000.0),
                TransactionPattern::new("Rent", 0.05, 2000.0..20000.0),
                TransactionPattern::new("Utilities", 0.05, 200.0..2000.0),
                TransactionPattern::new("Marketing", 0.05, 100.0..5000.0),
                TransactionPattern::new("Returns", 0.05, 20.0..200.0),
                TransactionPattern::new("Other", 0.05, 50.0..1000.0),
            ],
            CompanyArchetype::SaaS(_) => vec![
                TransactionPattern::new("Subscription Revenue", 0.30, 50.0..5000.0),
                TransactionPattern::new("Deferred Revenue Recog", 0.25, 100.0..10000.0),
                TransactionPattern::new("Payroll", 0.20, 10000.0..100000.0),
                TransactionPattern::new("Cloud Infrastructure", 0.10, 1000.0..50000.0),
                TransactionPattern::new("Marketing", 0.08, 500.0..20000.0),
                TransactionPattern::new("Refunds", 0.02, 50.0..500.0),
                TransactionPattern::new("Other", 0.05, 100.0..2000.0),
            ],
            CompanyArchetype::Manufacturing(_) => vec![
                TransactionPattern::new("Raw Materials", 0.30, 1000.0..100000.0),
                TransactionPattern::new("WIP Transfer", 0.20, 500.0..50000.0),
                TransactionPattern::new("Finished Goods Sale", 0.15, 1000.0..50000.0),
                TransactionPattern::new("Payroll", 0.15, 20000.0..200000.0),
                TransactionPattern::new("Equipment Depreciation", 0.05, 1000.0..20000.0),
                TransactionPattern::new("Utilities", 0.05, 5000.0..30000.0),
                TransactionPattern::new("Maintenance", 0.05, 500.0..10000.0),
                TransactionPattern::new("Other", 0.05, 200.0..5000.0),
            ],
            CompanyArchetype::ProfessionalServices(_) => vec![
                TransactionPattern::new("Time Billing", 0.35, 500.0..50000.0),
                TransactionPattern::new("Expense Reimbursement", 0.15, 100.0..5000.0),
                TransactionPattern::new("Payroll", 0.25, 15000.0..150000.0),
                TransactionPattern::new("Rent", 0.08, 5000.0..50000.0),
                TransactionPattern::new("Professional Development", 0.05, 200.0..5000.0),
                TransactionPattern::new("Insurance", 0.05, 1000.0..10000.0),
                TransactionPattern::new("Other", 0.07, 100.0..2000.0),
            ],
            CompanyArchetype::FinancialServices(_) => vec![
                TransactionPattern::new("Investment Transaction", 0.25, 10000.0..1000000.0),
                TransactionPattern::new("Fee Revenue", 0.15, 100.0..50000.0),
                TransactionPattern::new("Interest Income", 0.15, 1000.0..100000.0),
                TransactionPattern::new("Intercompany Transfer", 0.15, 50000.0..5000000.0),
                TransactionPattern::new("Payroll", 0.10, 50000.0..500000.0),
                TransactionPattern::new("Regulatory Reserve", 0.10, 10000.0..500000.0),
                TransactionPattern::new("Other", 0.10, 500.0..10000.0),
            ],
        }
    }

    /// Get seasonal multipliers by month (1-12).
    pub fn seasonal_multipliers(&self) -> [f64; 12] {
        match self {
            CompanyArchetype::Retail(config) => {
                let mut mults = [1.0; 12];
                // Apply peaks
                for &peak_month in &config.seasonal_peaks {
                    if peak_month >= 1 && peak_month <= 12 {
                        mults[(peak_month - 1) as usize] = 2.5;
                        // Adjacent months get smaller boost
                        if peak_month > 1 {
                            mults[(peak_month - 2) as usize] *= 1.5;
                        }
                    }
                }
                // January typically slow after holidays
                mults[0] = 0.7;
                mults
            }
            CompanyArchetype::SaaS(_) => {
                // Relatively flat with Q4 push and year-end renewals
                [1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.0, 1.1, 1.3]
            }
            CompanyArchetype::Manufacturing(_) => {
                // Production cycles, quieter in summer
                [1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8, 0.8, 1.0, 1.1, 1.2, 1.0]
            }
            CompanyArchetype::ProfessionalServices(_) => {
                // Busy in Q1 (audits), slower in summer
                [1.3, 1.2, 1.1, 1.0, 1.0, 0.9, 0.8, 0.8, 1.0, 1.0, 1.0, 0.9]
            }
            CompanyArchetype::FinancialServices(_) => {
                // Quarter-end spikes
                [1.0, 1.0, 1.4, 1.0, 1.0, 1.4, 1.0, 1.0, 1.4, 1.0, 1.0, 1.5]
            }
        }
    }
}

/// A transaction pattern with frequency and amount range.
#[derive(Debug, Clone)]
pub struct TransactionPattern {
    /// Pattern name
    pub name: String,
    /// Relative frequency (0.0 - 1.0, should sum to 1.0)
    pub frequency: f64,
    /// Amount range
    pub amount_range: Range<f64>,
}

impl TransactionPattern {
    /// Create a new transaction pattern.
    pub fn new(name: impl Into<String>, frequency: f64, amount_range: Range<f64>) -> Self {
        Self {
            name: name.into(),
            frequency,
            amount_range,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retail_archetype() {
        let retail = CompanyArchetype::retail_standard();
        assert_eq!(retail.industry(), Industry::Retail);
        assert_eq!(retail.size(), CompanySize::Medium);

        let patterns = retail.transaction_patterns();
        let total_freq: f64 = patterns.iter().map(|p| p.frequency).sum();
        assert!((total_freq - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_seasonal_multipliers() {
        let retail = CompanyArchetype::retail_standard();
        let mults = retail.seasonal_multipliers();

        // November and December should be peaks
        assert!(mults[10] > 1.5); // November
        assert!(mults[11] > 2.0); // December
        // January should be low
        assert!(mults[0] < 1.0);
    }

    #[test]
    fn test_company_sizes() {
        assert!(CompanySize::Small.typical_account_count().start < 100);
        assert!(CompanySize::Enterprise.typical_daily_transactions().end >= 10000);
    }
}
