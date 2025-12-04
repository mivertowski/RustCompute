//! Chart of Accounts templates for different industries.
//!
//! Each template provides a realistic account structure with
//! typical account relationships and expected flow patterns.

use super::{CompanyArchetype, Industry};
use crate::models::{AccountMetadata, AccountNode, AccountSemantics, AccountType};
use uuid::Uuid;

/// A Chart of Accounts template.
#[derive(Debug, Clone)]
pub struct ChartOfAccountsTemplate {
    /// Industry this template is designed for
    pub industry: Industry,
    /// Template name
    pub name: String,
    /// Account definitions
    pub accounts: Vec<AccountDefinition>,
    /// Expected flow patterns between accounts
    pub expected_flows: Vec<ExpectedFlow>,
}

/// Definition of an account in the template.
#[derive(Debug, Clone)]
pub struct AccountDefinition {
    /// Account code (e.g., "1100")
    pub code: String,
    /// Account name
    pub name: String,
    /// Account type
    pub account_type: AccountType,
    /// Account class (for grouping)
    pub class_id: u8,
    /// Account subclass
    pub subclass_id: u8,
    /// Parent account code (for hierarchy)
    pub parent_code: Option<String>,
    /// Semantic flags
    pub semantics: u32,
    /// Typical activity level (transactions per month)
    pub typical_activity: f32,
    /// Description
    pub description: String,
}

impl AccountDefinition {
    /// Create a new account definition.
    pub fn new(
        code: impl Into<String>,
        name: impl Into<String>,
        account_type: AccountType,
    ) -> Self {
        Self {
            code: code.into(),
            name: name.into(),
            account_type,
            class_id: 0,
            subclass_id: 0,
            parent_code: None,
            semantics: 0,
            typical_activity: 10.0,
            description: String::new(),
        }
    }

    /// Set the class and subclass.
    pub fn with_class(mut self, class_id: u8, subclass_id: u8) -> Self {
        self.class_id = class_id;
        self.subclass_id = subclass_id;
        self
    }

    /// Set the parent account.
    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent_code = Some(parent.into());
        self
    }

    /// Set semantic flags.
    pub fn with_semantics(mut self, semantics: u32) -> Self {
        self.semantics = semantics;
        self
    }

    /// Set typical activity level.
    pub fn with_activity(mut self, activity: f32) -> Self {
        self.typical_activity = activity;
        self
    }

    /// Set description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Convert to AccountNode and AccountMetadata.
    pub fn to_account(&self, index: u16) -> (AccountNode, AccountMetadata) {
        let mut node = AccountNode::new(Uuid::new_v4(), self.account_type, index);
        node.class_id = self.class_id;
        node.subclass_id = self.subclass_id;

        let mut metadata = AccountMetadata::new(&self.code, &self.name);
        metadata.description = self.description.clone();
        metadata.semantics = AccountSemantics {
            flags: self.semantics,
            typical_frequency: self.typical_activity,
            avg_amount_scale: 50.0, // Default mid-range
        };

        (node, metadata)
    }
}

/// An expected flow pattern between accounts.
#[derive(Debug, Clone)]
pub struct ExpectedFlow {
    /// Source account code
    pub from_code: String,
    /// Target account code
    pub to_code: String,
    /// Relative frequency (0.0 - 1.0)
    pub frequency: f64,
    /// Typical amount range
    pub amount_range: (f64, f64),
    /// Description of this flow
    pub description: String,
}

impl ExpectedFlow {
    /// Create a new expected flow.
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        frequency: f64,
        amount_range: (f64, f64),
    ) -> Self {
        Self {
            from_code: from.into(),
            to_code: to.into(),
            frequency,
            amount_range,
            description: String::new(),
        }
    }

    /// Add description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

impl ChartOfAccountsTemplate {
    /// Create a minimal GAAP-compliant template (for education).
    pub fn gaap_minimal() -> Self {
        Self {
            industry: Industry::Retail,
            name: "GAAP Minimal".to_string(),
            accounts: vec![
                // Assets (1xxx)
                AccountDefinition::new("1000", "Assets", AccountType::Asset)
                    .with_class(1, 0)
                    .with_description("Total assets"),
                AccountDefinition::new("1100", "Cash", AccountType::Asset)
                    .with_class(1, 1)
                    .with_parent("1000")
                    .with_semantics(AccountSemantics::IS_CASH)
                    .with_activity(100.0)
                    .with_description("Cash and cash equivalents"),
                AccountDefinition::new("1200", "Accounts Receivable", AccountType::Asset)
                    .with_class(1, 2)
                    .with_parent("1000")
                    .with_semantics(AccountSemantics::IS_RECEIVABLE)
                    .with_activity(50.0)
                    .with_description("Amounts owed by customers"),
                AccountDefinition::new("1300", "Inventory", AccountType::Asset)
                    .with_class(1, 3)
                    .with_parent("1000")
                    .with_semantics(AccountSemantics::IS_INVENTORY)
                    .with_activity(30.0)
                    .with_description("Goods held for sale"),
                AccountDefinition::new("1400", "Prepaid Expenses", AccountType::Asset)
                    .with_class(1, 4)
                    .with_parent("1000")
                    .with_activity(5.0)
                    .with_description("Expenses paid in advance"),
                AccountDefinition::new("1500", "Fixed Assets", AccountType::Asset)
                    .with_class(1, 5)
                    .with_parent("1000")
                    .with_activity(2.0)
                    .with_description("Property, plant, and equipment"),
                AccountDefinition::new("1510", "Accumulated Depreciation", AccountType::Contra)
                    .with_class(1, 5)
                    .with_parent("1500")
                    .with_semantics(AccountSemantics::IS_DEPRECIATION)
                    .with_activity(12.0)
                    .with_description("Accumulated depreciation on fixed assets"),
                // Liabilities (2xxx)
                AccountDefinition::new("2000", "Liabilities", AccountType::Liability)
                    .with_class(2, 0)
                    .with_description("Total liabilities"),
                AccountDefinition::new("2100", "Accounts Payable", AccountType::Liability)
                    .with_class(2, 1)
                    .with_parent("2000")
                    .with_semantics(AccountSemantics::IS_PAYABLE)
                    .with_activity(40.0)
                    .with_description("Amounts owed to suppliers"),
                AccountDefinition::new("2200", "Accrued Expenses", AccountType::Liability)
                    .with_class(2, 2)
                    .with_parent("2000")
                    .with_activity(12.0)
                    .with_description("Expenses incurred but not yet paid"),
                AccountDefinition::new("2300", "Unearned Revenue", AccountType::Liability)
                    .with_class(2, 3)
                    .with_parent("2000")
                    .with_activity(10.0)
                    .with_description("Revenue received but not yet earned"),
                AccountDefinition::new("2400", "Notes Payable", AccountType::Liability)
                    .with_class(2, 4)
                    .with_parent("2000")
                    .with_activity(2.0)
                    .with_description("Short-term loans"),
                // Equity (3xxx)
                AccountDefinition::new("3000", "Equity", AccountType::Equity)
                    .with_class(3, 0)
                    .with_description("Owner's equity"),
                AccountDefinition::new("3100", "Common Stock", AccountType::Equity)
                    .with_class(3, 1)
                    .with_parent("3000")
                    .with_activity(1.0)
                    .with_description("Issued common shares"),
                AccountDefinition::new("3200", "Retained Earnings", AccountType::Equity)
                    .with_class(3, 2)
                    .with_parent("3000")
                    .with_activity(1.0)
                    .with_description("Accumulated profits"),
                // Revenue (4xxx)
                AccountDefinition::new("4000", "Revenue", AccountType::Revenue)
                    .with_class(4, 0)
                    .with_semantics(AccountSemantics::IS_REVENUE)
                    .with_description("Total revenue"),
                AccountDefinition::new("4100", "Sales Revenue", AccountType::Revenue)
                    .with_class(4, 1)
                    .with_parent("4000")
                    .with_semantics(AccountSemantics::IS_REVENUE)
                    .with_activity(100.0)
                    .with_description("Revenue from product sales"),
                AccountDefinition::new("4200", "Service Revenue", AccountType::Revenue)
                    .with_class(4, 2)
                    .with_parent("4000")
                    .with_semantics(AccountSemantics::IS_REVENUE)
                    .with_activity(20.0)
                    .with_description("Revenue from services"),
                // Expenses (5xxx-6xxx)
                AccountDefinition::new("5000", "Cost of Goods Sold", AccountType::Expense)
                    .with_class(5, 0)
                    .with_semantics(AccountSemantics::IS_COGS)
                    .with_activity(80.0)
                    .with_description("Direct cost of products sold"),
                AccountDefinition::new("6000", "Operating Expenses", AccountType::Expense)
                    .with_class(6, 0)
                    .with_semantics(AccountSemantics::IS_EXPENSE)
                    .with_description("General operating expenses"),
                AccountDefinition::new("6100", "Salaries Expense", AccountType::Expense)
                    .with_class(6, 1)
                    .with_parent("6000")
                    .with_semantics(AccountSemantics::IS_PAYROLL)
                    .with_activity(24.0)
                    .with_description("Employee salaries"),
                AccountDefinition::new("6200", "Rent Expense", AccountType::Expense)
                    .with_class(6, 2)
                    .with_parent("6000")
                    .with_activity(12.0)
                    .with_description("Rent for facilities"),
                AccountDefinition::new("6300", "Utilities Expense", AccountType::Expense)
                    .with_class(6, 3)
                    .with_parent("6000")
                    .with_activity(12.0)
                    .with_description("Electricity, water, gas"),
                AccountDefinition::new("6400", "Depreciation Expense", AccountType::Expense)
                    .with_class(6, 4)
                    .with_parent("6000")
                    .with_activity(12.0)
                    .with_description("Depreciation of fixed assets"),
                AccountDefinition::new("6500", "Marketing Expense", AccountType::Expense)
                    .with_class(6, 5)
                    .with_parent("6000")
                    .with_activity(20.0)
                    .with_description("Advertising and marketing"),
                AccountDefinition::new("6900", "Other Expenses", AccountType::Expense)
                    .with_class(6, 9)
                    .with_parent("6000")
                    .with_activity(15.0)
                    .with_description("Miscellaneous expenses"),
                // Suspense/Clearing (9xxx)
                AccountDefinition::new("9100", "Clearing Account", AccountType::Asset)
                    .with_class(9, 1)
                    .with_semantics(AccountSemantics::IS_SUSPENSE)
                    .with_activity(30.0)
                    .with_description("Temporary clearing account"),
            ],
            expected_flows: vec![
                // Sales cycle
                ExpectedFlow::new("4100", "1200", 0.30, (50.0, 5000.0))
                    .with_description("Credit sale: Revenue → A/R"),
                ExpectedFlow::new("1200", "1100", 0.25, (50.0, 5000.0))
                    .with_description("Collection: A/R → Cash"),
                ExpectedFlow::new("4100", "1100", 0.10, (20.0, 500.0))
                    .with_description("Cash sale: Revenue → Cash"),
                // Purchasing cycle
                ExpectedFlow::new("1300", "2100", 0.15, (100.0, 10000.0))
                    .with_description("Purchase: Inventory → A/P"),
                ExpectedFlow::new("2100", "1100", 0.15, (100.0, 10000.0))
                    .with_description("Payment: A/P → Cash"),
                ExpectedFlow::new("5000", "1300", 0.15, (50.0, 5000.0))
                    .with_description("Cost of sale: COGS → Inventory"),
                // Operating expenses
                ExpectedFlow::new("6100", "1100", 0.08, (5000.0, 50000.0))
                    .with_description("Payroll: Salaries → Cash"),
                ExpectedFlow::new("6200", "1100", 0.04, (1000.0, 10000.0))
                    .with_description("Rent payment"),
                ExpectedFlow::new("6300", "1100", 0.04, (200.0, 2000.0))
                    .with_description("Utilities payment"),
                ExpectedFlow::new("6400", "1510", 0.04, (500.0, 5000.0))
                    .with_description("Depreciation: Expense → Accum Depr"),
            ],
        }
    }

    /// Create a retail-specific template with ~150 accounts.
    pub fn retail_standard() -> Self {
        let mut template = Self::gaap_minimal();
        template.industry = Industry::Retail;
        template.name = "Retail Standard".to_string();

        // Add retail-specific accounts
        template.accounts.extend(vec![
            // More detailed inventory
            AccountDefinition::new("1310", "Merchandise Inventory", AccountType::Asset)
                .with_class(1, 3)
                .with_parent("1300")
                .with_semantics(AccountSemantics::IS_INVENTORY)
                .with_activity(50.0),
            AccountDefinition::new("1320", "Inventory in Transit", AccountType::Asset)
                .with_class(1, 3)
                .with_parent("1300")
                .with_activity(10.0),
            // Point of Sale
            AccountDefinition::new("1110", "Cash Registers", AccountType::Asset)
                .with_class(1, 1)
                .with_parent("1100")
                .with_semantics(AccountSemantics::IS_CASH)
                .with_activity(200.0),
            AccountDefinition::new("1120", "Undeposited Funds", AccountType::Asset)
                .with_class(1, 1)
                .with_parent("1100")
                .with_semantics(AccountSemantics::IS_SUSPENSE | AccountSemantics::IS_CASH)
                .with_activity(100.0),
            // Credit card processing
            AccountDefinition::new("1130", "Credit Card Receivable", AccountType::Asset)
                .with_class(1, 1)
                .with_parent("1100")
                .with_activity(80.0),
            // Sales returns
            AccountDefinition::new("4110", "Sales Returns", AccountType::Contra)
                .with_class(4, 1)
                .with_parent("4100")
                .with_activity(20.0),
            AccountDefinition::new("4120", "Sales Discounts", AccountType::Contra)
                .with_class(4, 1)
                .with_parent("4100")
                .with_activity(15.0),
            // Retail expenses
            AccountDefinition::new("6510", "Store Supplies", AccountType::Expense)
                .with_class(6, 5)
                .with_parent("6500")
                .with_activity(10.0),
            AccountDefinition::new("6520", "Credit Card Fees", AccountType::Expense)
                .with_class(6, 5)
                .with_parent("6500")
                .with_activity(30.0),
            AccountDefinition::new("6530", "Shrinkage", AccountType::Expense)
                .with_class(6, 5)
                .with_parent("6500")
                .with_activity(12.0),
        ]);

        // Add retail-specific flows
        template.expected_flows.extend(vec![
            ExpectedFlow::new("1110", "1120", 0.20, (100.0, 5000.0))
                .with_description("Register → Undeposited"),
            ExpectedFlow::new("1120", "1100", 0.15, (1000.0, 20000.0))
                .with_description("Bank deposit"),
            ExpectedFlow::new("4110", "1200", 0.05, (20.0, 500.0)).with_description("Sales return"),
            ExpectedFlow::new("6520", "1130", 0.08, (50.0, 500.0))
                .with_description("CC processing fees"),
        ]);

        template
    }

    /// Create a SaaS-specific template.
    pub fn saas_standard() -> Self {
        let mut template = Self::gaap_minimal();
        template.industry = Industry::SaaS;
        template.name = "SaaS Standard".to_string();

        // Replace/add SaaS-specific accounts
        template.accounts.extend(vec![
            // Deferred revenue is big in SaaS
            AccountDefinition::new("2310", "Deferred Revenue - Current", AccountType::Liability)
                .with_class(2, 3)
                .with_parent("2300")
                .with_activity(100.0),
            AccountDefinition::new(
                "2320",
                "Deferred Revenue - Long-term",
                AccountType::Liability,
            )
            .with_class(2, 3)
            .with_parent("2300")
            .with_activity(20.0),
            // Subscription metrics
            AccountDefinition::new("4110", "Subscription Revenue", AccountType::Revenue)
                .with_class(4, 1)
                .with_parent("4100")
                .with_semantics(AccountSemantics::IS_REVENUE)
                .with_activity(200.0),
            AccountDefinition::new("4120", "Professional Services", AccountType::Revenue)
                .with_class(4, 2)
                .with_parent("4200")
                .with_activity(20.0),
            // SaaS costs
            AccountDefinition::new("5100", "Hosting Costs", AccountType::Expense)
                .with_class(5, 1)
                .with_parent("5000")
                .with_activity(12.0),
            AccountDefinition::new("5200", "Third-party Software", AccountType::Expense)
                .with_class(5, 2)
                .with_parent("5000")
                .with_activity(24.0),
            // Customer acquisition
            AccountDefinition::new("6510", "Customer Acquisition Cost", AccountType::Expense)
                .with_class(6, 5)
                .with_parent("6500")
                .with_activity(50.0),
            AccountDefinition::new("1410", "Deferred Commission", AccountType::Asset)
                .with_class(1, 4)
                .with_parent("1400")
                .with_activity(30.0),
        ]);

        // SaaS-specific flows
        template.expected_flows = vec![
            ExpectedFlow::new("1100", "2310", 0.25, (500.0, 50000.0))
                .with_description("Annual subscription cash → Deferred revenue"),
            ExpectedFlow::new("2310", "4110", 0.30, (50.0, 5000.0))
                .with_description("Revenue recognition"),
            ExpectedFlow::new("4110", "1200", 0.10, (100.0, 10000.0))
                .with_description("Invoiced subscription"),
            ExpectedFlow::new("1200", "1100", 0.15, (100.0, 10000.0))
                .with_description("Collection"),
            ExpectedFlow::new("5100", "2100", 0.08, (5000.0, 50000.0))
                .with_description("Hosting invoice"),
            ExpectedFlow::new("2100", "1100", 0.10, (1000.0, 50000.0))
                .with_description("Vendor payment"),
            ExpectedFlow::new("6510", "1100", 0.05, (1000.0, 20000.0))
                .with_description("Marketing spend"),
        ];

        template
    }

    /// Get template for a company archetype.
    pub fn for_archetype(archetype: &CompanyArchetype) -> Self {
        match archetype.industry() {
            Industry::Retail => Self::retail_standard(),
            Industry::SaaS => Self::saas_standard(),
            Industry::Manufacturing => Self::gaap_minimal(), // TODO: manufacturing template
            Industry::ProfessionalServices => Self::gaap_minimal(), // TODO
            Industry::FinancialServices => Self::gaap_minimal(), // TODO
            _ => Self::gaap_minimal(),
        }
    }

    /// Get account count.
    pub fn account_count(&self) -> usize {
        self.accounts.len()
    }

    /// Find account definition by code.
    pub fn find_by_code(&self, code: &str) -> Option<&AccountDefinition> {
        self.accounts.iter().find(|a| a.code == code)
    }

    /// Get accounts by type.
    pub fn accounts_by_type(&self, account_type: AccountType) -> Vec<&AccountDefinition> {
        self.accounts
            .iter()
            .filter(|a| a.account_type == account_type)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaap_minimal() {
        let template = ChartOfAccountsTemplate::gaap_minimal();
        assert!(template.account_count() > 20);

        // Should have basic account types
        assert!(!template.accounts_by_type(AccountType::Asset).is_empty());
        assert!(!template.accounts_by_type(AccountType::Liability).is_empty());
        assert!(!template.accounts_by_type(AccountType::Revenue).is_empty());
        assert!(!template.accounts_by_type(AccountType::Expense).is_empty());
    }

    #[test]
    fn test_retail_template() {
        let template = ChartOfAccountsTemplate::retail_standard();
        assert!(template.account_count() > 30);
        assert_eq!(template.industry, Industry::Retail);

        // Should have retail-specific accounts
        assert!(template.find_by_code("1110").is_some()); // Cash Registers
        assert!(template.find_by_code("1120").is_some()); // Undeposited Funds
    }

    #[test]
    fn test_saas_template() {
        let template = ChartOfAccountsTemplate::saas_standard();
        assert_eq!(template.industry, Industry::SaaS);

        // Should have SaaS-specific accounts
        assert!(template.find_by_code("2310").is_some()); // Deferred Revenue
        assert!(template.find_by_code("4110").is_some()); // Subscription Revenue
    }
}
