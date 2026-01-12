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

    /// Create a manufacturing-specific template with inventory tracking.
    ///
    /// Includes accounts for:
    /// - Raw materials, work-in-progress, and finished goods inventory
    /// - Manufacturing overhead allocation
    /// - Direct and indirect labor
    /// - Factory equipment and depreciation
    pub fn manufacturing_standard() -> Self {
        let mut template = Self::gaap_minimal();
        template.industry = Industry::Manufacturing;
        template.name = "Manufacturing Standard".to_string();

        // Manufacturing-specific accounts
        template.accounts.extend(vec![
            // Detailed inventory accounts
            AccountDefinition::new("1310", "Raw Materials Inventory", AccountType::Asset)
                .with_class(1, 3)
                .with_parent("1300")
                .with_semantics(AccountSemantics::IS_INVENTORY)
                .with_activity(40.0)
                .with_description("Materials awaiting production"),
            AccountDefinition::new("1320", "Work in Progress", AccountType::Asset)
                .with_class(1, 3)
                .with_parent("1300")
                .with_semantics(AccountSemantics::IS_INVENTORY)
                .with_activity(60.0)
                .with_description("Partially completed goods"),
            AccountDefinition::new("1330", "Finished Goods Inventory", AccountType::Asset)
                .with_class(1, 3)
                .with_parent("1300")
                .with_semantics(AccountSemantics::IS_INVENTORY)
                .with_activity(50.0)
                .with_description("Completed goods ready for sale"),
            AccountDefinition::new("1340", "Factory Supplies", AccountType::Asset)
                .with_class(1, 3)
                .with_parent("1300")
                .with_activity(15.0)
                .with_description("Consumable manufacturing supplies"),
            // Manufacturing equipment
            AccountDefinition::new("1520", "Manufacturing Equipment", AccountType::Asset)
                .with_class(1, 5)
                .with_parent("1500")
                .with_activity(5.0)
                .with_description("Production machinery"),
            AccountDefinition::new(
                "1525",
                "Accum Depreciation - Mfg Equipment",
                AccountType::Contra,
            )
            .with_class(1, 5)
            .with_parent("1520")
            .with_semantics(AccountSemantics::IS_DEPRECIATION)
            .with_activity(12.0)
            .with_description("Accumulated depreciation on manufacturing equipment"),
            // Manufacturing costs
            AccountDefinition::new("5100", "Direct Materials", AccountType::Expense)
                .with_class(5, 1)
                .with_parent("5000")
                .with_semantics(AccountSemantics::IS_COGS)
                .with_activity(100.0)
                .with_description("Raw materials used in production"),
            AccountDefinition::new("5200", "Direct Labor", AccountType::Expense)
                .with_class(5, 2)
                .with_parent("5000")
                .with_semantics(AccountSemantics::IS_COGS | AccountSemantics::IS_PAYROLL)
                .with_activity(60.0)
                .with_description("Labor directly applied to production"),
            AccountDefinition::new("5300", "Manufacturing Overhead", AccountType::Expense)
                .with_class(5, 3)
                .with_parent("5000")
                .with_activity(30.0)
                .with_description("Indirect manufacturing costs"),
            AccountDefinition::new("5310", "Factory Utilities", AccountType::Expense)
                .with_class(5, 3)
                .with_parent("5300")
                .with_activity(12.0)
                .with_description("Factory electricity, water, gas"),
            AccountDefinition::new("5320", "Indirect Labor", AccountType::Expense)
                .with_class(5, 3)
                .with_parent("5300")
                .with_semantics(AccountSemantics::IS_PAYROLL)
                .with_activity(24.0)
                .with_description("Supervisory and maintenance labor"),
            AccountDefinition::new("5330", "Equipment Maintenance", AccountType::Expense)
                .with_class(5, 3)
                .with_parent("5300")
                .with_activity(12.0)
                .with_description("Machine repairs and maintenance"),
            AccountDefinition::new("5340", "Factory Depreciation", AccountType::Expense)
                .with_class(5, 3)
                .with_parent("5300")
                .with_activity(12.0)
                .with_description("Depreciation on manufacturing assets"),
            // Work order clearing
            AccountDefinition::new("9110", "Work Order Clearing", AccountType::Asset)
                .with_class(9, 1)
                .with_parent("9100")
                .with_semantics(AccountSemantics::IS_SUSPENSE)
                .with_activity(80.0)
                .with_description("Production job cost accumulation"),
        ]);

        // Manufacturing-specific flows
        template.expected_flows = vec![
            // Material purchasing
            ExpectedFlow::new("1310", "2100", 0.20, (1000.0, 50000.0))
                .with_description("Purchase raw materials on credit"),
            ExpectedFlow::new("2100", "1100", 0.20, (1000.0, 50000.0))
                .with_description("Pay supplier for materials"),
            // Production flow
            ExpectedFlow::new("5100", "1310", 0.25, (500.0, 20000.0))
                .with_description("Issue materials to production"),
            ExpectedFlow::new("1320", "5100", 0.15, (500.0, 20000.0))
                .with_description("Accumulate direct materials in WIP"),
            ExpectedFlow::new("1320", "5200", 0.15, (2000.0, 30000.0))
                .with_description("Accumulate direct labor in WIP"),
            ExpectedFlow::new("1320", "5300", 0.10, (1000.0, 15000.0))
                .with_description("Allocate overhead to WIP"),
            ExpectedFlow::new("1330", "1320", 0.20, (5000.0, 100000.0))
                .with_description("Transfer completed goods from WIP"),
            // Sales
            ExpectedFlow::new("5000", "1330", 0.25, (3000.0, 80000.0))
                .with_description("Cost of goods sold from finished goods"),
            ExpectedFlow::new("4100", "1200", 0.30, (5000.0, 150000.0))
                .with_description("Revenue recognized on sale"),
            ExpectedFlow::new("1200", "1100", 0.25, (5000.0, 150000.0))
                .with_description("Customer payment received"),
            // Payroll
            ExpectedFlow::new("5200", "1100", 0.08, (10000.0, 100000.0))
                .with_description("Direct labor payroll"),
            ExpectedFlow::new("5320", "1100", 0.05, (5000.0, 40000.0))
                .with_description("Indirect labor payroll"),
        ];

        template
    }

    /// Create a professional services template.
    ///
    /// Includes accounts for:
    /// - Billable and unbilled work in progress
    /// - Client retainers and advances
    /// - Professional fee revenue categories
    /// - Reimbursable expenses
    pub fn professional_services_standard() -> Self {
        let mut template = Self::gaap_minimal();
        template.industry = Industry::ProfessionalServices;
        template.name = "Professional Services".to_string();

        // Professional services-specific accounts
        template.accounts.extend(vec![
            // Unbilled work
            AccountDefinition::new("1210", "Unbilled Receivables", AccountType::Asset)
                .with_class(1, 2)
                .with_parent("1200")
                .with_semantics(AccountSemantics::IS_RECEIVABLE)
                .with_activity(100.0)
                .with_description("Work completed but not yet invoiced"),
            AccountDefinition::new("1220", "Work in Progress - Billable", AccountType::Asset)
                .with_class(1, 2)
                .with_parent("1200")
                .with_activity(150.0)
                .with_description("Time and expenses in progress"),
            AccountDefinition::new("1230", "Reimbursable Expenses", AccountType::Asset)
                .with_class(1, 2)
                .with_parent("1200")
                .with_activity(40.0)
                .with_description("Client-reimbursable expenses pending"),
            // Client advances
            AccountDefinition::new("2310", "Client Retainers", AccountType::Liability)
                .with_class(2, 3)
                .with_parent("2300")
                .with_activity(30.0)
                .with_description("Advance payments from clients"),
            AccountDefinition::new("2320", "Client Deposits", AccountType::Liability)
                .with_class(2, 3)
                .with_parent("2300")
                .with_activity(20.0)
                .with_description("Project deposits held"),
            // Revenue categories
            AccountDefinition::new("4110", "Professional Fees - Hourly", AccountType::Revenue)
                .with_class(4, 1)
                .with_parent("4100")
                .with_semantics(AccountSemantics::IS_REVENUE)
                .with_activity(200.0)
                .with_description("Revenue from billable hours"),
            AccountDefinition::new("4120", "Professional Fees - Fixed", AccountType::Revenue)
                .with_class(4, 1)
                .with_parent("4100")
                .with_semantics(AccountSemantics::IS_REVENUE)
                .with_activity(50.0)
                .with_description("Revenue from fixed-fee projects"),
            AccountDefinition::new("4130", "Retainer Fees", AccountType::Revenue)
                .with_class(4, 1)
                .with_parent("4100")
                .with_semantics(AccountSemantics::IS_REVENUE)
                .with_activity(40.0)
                .with_description("Monthly retainer revenue"),
            AccountDefinition::new("4210", "Reimbursed Expenses", AccountType::Revenue)
                .with_class(4, 2)
                .with_parent("4200")
                .with_activity(30.0)
                .with_description("Client-reimbursed expenses"),
            // Direct costs
            AccountDefinition::new("5100", "Professional Labor Cost", AccountType::Expense)
                .with_class(5, 1)
                .with_parent("5000")
                .with_semantics(AccountSemantics::IS_PAYROLL)
                .with_activity(24.0)
                .with_description("Cost of billable staff time"),
            AccountDefinition::new("5200", "Subcontractor Fees", AccountType::Expense)
                .with_class(5, 2)
                .with_parent("5000")
                .with_activity(20.0)
                .with_description("Outside contractor costs"),
            // Indirect expenses
            AccountDefinition::new("6110", "Professional Development", AccountType::Expense)
                .with_class(6, 1)
                .with_parent("6100")
                .with_activity(12.0)
                .with_description("Training and certifications"),
            AccountDefinition::new("6510", "Business Development", AccountType::Expense)
                .with_class(6, 5)
                .with_parent("6500")
                .with_activity(25.0)
                .with_description("Client acquisition and networking"),
            AccountDefinition::new(
                "6520",
                "Professional Liability Insurance",
                AccountType::Expense,
            )
            .with_class(6, 5)
            .with_parent("6500")
            .with_activity(12.0)
            .with_description("E&O insurance premiums"),
        ]);

        // Professional services flows
        template.expected_flows = vec![
            // Time & expense billing
            ExpectedFlow::new("1220", "5100", 0.30, (500.0, 20000.0))
                .with_description("Record billable time as WIP"),
            ExpectedFlow::new("1210", "1220", 0.25, (1000.0, 50000.0))
                .with_description("Transfer completed WIP to unbilled"),
            ExpectedFlow::new("1200", "1210", 0.25, (1000.0, 50000.0))
                .with_description("Invoice unbilled work"),
            ExpectedFlow::new("4110", "1200", 0.30, (1000.0, 50000.0))
                .with_description("Recognize hourly fee revenue"),
            ExpectedFlow::new("1200", "1100", 0.25, (1000.0, 50000.0))
                .with_description("Client payment received"),
            // Retainer flow
            ExpectedFlow::new("1100", "2310", 0.10, (5000.0, 50000.0))
                .with_description("Receive client retainer"),
            ExpectedFlow::new("2310", "4130", 0.15, (1000.0, 10000.0))
                .with_description("Apply retainer to revenue"),
            // Reimbursables
            ExpectedFlow::new("1230", "1100", 0.08, (100.0, 2000.0))
                .with_description("Pay reimbursable expense"),
            ExpectedFlow::new("4210", "1230", 0.08, (100.0, 2000.0))
                .with_description("Bill reimbursable to client"),
            // Payroll
            ExpectedFlow::new("6100", "1100", 0.08, (10000.0, 100000.0))
                .with_description("Staff payroll"),
        ];

        template
    }

    /// Create a financial services template.
    ///
    /// Includes accounts for:
    /// - Trading and investment securities
    /// - Customer custody and fiduciary accounts
    /// - Regulatory capital requirements
    /// - Interest income and expense
    pub fn financial_services_standard() -> Self {
        let mut template = Self::gaap_minimal();
        template.industry = Industry::FinancialServices;
        template.name = "Financial Services".to_string();

        // Financial services-specific accounts
        template.accounts.extend(vec![
            // Trading and investment accounts
            AccountDefinition::new("1110", "Trading Securities", AccountType::Asset)
                .with_class(1, 1)
                .with_parent("1100")
                .with_activity(500.0)
                .with_description("Securities held for trading"),
            AccountDefinition::new("1120", "Available-for-Sale Securities", AccountType::Asset)
                .with_class(1, 1)
                .with_parent("1100")
                .with_activity(100.0)
                .with_description("Securities available for sale"),
            AccountDefinition::new("1130", "Held-to-Maturity Securities", AccountType::Asset)
                .with_class(1, 1)
                .with_parent("1100")
                .with_activity(20.0)
                .with_description("Securities held to maturity"),
            // Loan portfolio
            AccountDefinition::new("1250", "Loans Receivable", AccountType::Asset)
                .with_class(1, 2)
                .with_parent("1200")
                .with_semantics(AccountSemantics::IS_RECEIVABLE)
                .with_activity(100.0)
                .with_description("Outstanding loan portfolio"),
            AccountDefinition::new("1255", "Allowance for Loan Losses", AccountType::Contra)
                .with_class(1, 2)
                .with_parent("1250")
                .with_activity(24.0)
                .with_description("Reserve for expected credit losses"),
            AccountDefinition::new("1260", "Accrued Interest Receivable", AccountType::Asset)
                .with_class(1, 2)
                .with_parent("1200")
                .with_activity(50.0)
                .with_description("Interest earned but not yet received"),
            // Fiduciary accounts (off-balance sheet tracking)
            AccountDefinition::new("1610", "Customer Custody Assets", AccountType::Asset)
                .with_class(1, 6)
                .with_parent("1600")
                .with_activity(200.0)
                .with_description("Assets held in custody for clients"),
            AccountDefinition::new(
                "2610",
                "Customer Custody Liabilities",
                AccountType::Liability,
            )
            .with_class(2, 6)
            .with_parent("2600")
            .with_activity(200.0)
            .with_description("Obligation to return custody assets"),
            // Deposits
            AccountDefinition::new("2110", "Customer Deposits - Demand", AccountType::Liability)
                .with_class(2, 1)
                .with_parent("2100")
                .with_activity(300.0)
                .with_description("Checking/demand deposit accounts"),
            AccountDefinition::new(
                "2120",
                "Customer Deposits - Savings",
                AccountType::Liability,
            )
            .with_class(2, 1)
            .with_parent("2100")
            .with_activity(100.0)
            .with_description("Savings deposit accounts"),
            AccountDefinition::new("2130", "Customer Deposits - Time", AccountType::Liability)
                .with_class(2, 1)
                .with_parent("2100")
                .with_activity(50.0)
                .with_description("CDs and time deposits"),
            // Regulatory accounts
            AccountDefinition::new("3110", "Regulatory Capital Reserve", AccountType::Equity)
                .with_class(3, 1)
                .with_parent("3100")
                .with_activity(12.0)
                .with_description("Capital held for regulatory requirements"),
            AccountDefinition::new("2510", "Accrued Interest Payable", AccountType::Liability)
                .with_class(2, 5)
                .with_parent("2500")
                .with_activity(50.0)
                .with_description("Interest owed but not yet paid"),
            // Revenue accounts
            AccountDefinition::new("4110", "Interest Income - Loans", AccountType::Revenue)
                .with_class(4, 1)
                .with_parent("4100")
                .with_semantics(AccountSemantics::IS_REVENUE)
                .with_activity(100.0)
                .with_description("Interest earned on loan portfolio"),
            AccountDefinition::new("4120", "Interest Income - Securities", AccountType::Revenue)
                .with_class(4, 1)
                .with_parent("4100")
                .with_semantics(AccountSemantics::IS_REVENUE)
                .with_activity(80.0)
                .with_description("Interest and dividends from investments"),
            AccountDefinition::new("4130", "Trading Gains (Losses)", AccountType::Revenue)
                .with_class(4, 1)
                .with_parent("4100")
                .with_activity(200.0)
                .with_description("Net gains/losses from trading"),
            AccountDefinition::new("4210", "Fee Income", AccountType::Revenue)
                .with_class(4, 2)
                .with_parent("4200")
                .with_activity(150.0)
                .with_description("Transaction and service fees"),
            AccountDefinition::new("4220", "Custody Fees", AccountType::Revenue)
                .with_class(4, 2)
                .with_parent("4200")
                .with_activity(50.0)
                .with_description("Asset custody service fees"),
            // Expense accounts
            AccountDefinition::new("5100", "Interest Expense - Deposits", AccountType::Expense)
                .with_class(5, 1)
                .with_parent("5000")
                .with_activity(100.0)
                .with_description("Interest paid on customer deposits"),
            AccountDefinition::new(
                "5200",
                "Interest Expense - Borrowings",
                AccountType::Expense,
            )
            .with_class(5, 2)
            .with_parent("5000")
            .with_activity(50.0)
            .with_description("Interest on wholesale borrowings"),
            AccountDefinition::new("5300", "Provision for Loan Losses", AccountType::Expense)
                .with_class(5, 3)
                .with_parent("5000")
                .with_activity(24.0)
                .with_description("Additions to loan loss reserve"),
            AccountDefinition::new("6110", "Compliance Costs", AccountType::Expense)
                .with_class(6, 1)
                .with_parent("6100")
                .with_activity(24.0)
                .with_description("Regulatory compliance expenses"),
            AccountDefinition::new("6120", "Technology Infrastructure", AccountType::Expense)
                .with_class(6, 1)
                .with_parent("6100")
                .with_activity(36.0)
                .with_description("Trading systems and IT infrastructure"),
        ]);

        // Financial services flows
        template.expected_flows = vec![
            // Deposit flows
            ExpectedFlow::new("1100", "2110", 0.20, (1000.0, 500000.0))
                .with_description("Customer deposit received"),
            ExpectedFlow::new("2110", "1100", 0.15, (500.0, 100000.0))
                .with_description("Customer withdrawal"),
            // Lending flows
            ExpectedFlow::new("1250", "1100", 0.10, (10000.0, 1000000.0))
                .with_description("Loan disbursement"),
            ExpectedFlow::new("1100", "1250", 0.12, (1000.0, 100000.0))
                .with_description("Loan repayment principal"),
            ExpectedFlow::new("1100", "4110", 0.15, (100.0, 50000.0))
                .with_description("Loan interest payment"),
            // Trading flows
            ExpectedFlow::new("1110", "1100", 0.25, (10000.0, 10000000.0))
                .with_description("Security purchase"),
            ExpectedFlow::new("1100", "1110", 0.25, (10000.0, 10000000.0))
                .with_description("Security sale"),
            ExpectedFlow::new("4130", "1110", 0.20, (100.0, 500000.0))
                .with_description("Trading gain recognized"),
            // Interest expense
            ExpectedFlow::new("5100", "2510", 0.10, (100.0, 100000.0))
                .with_description("Accrue interest on deposits"),
            ExpectedFlow::new("2510", "1100", 0.08, (100.0, 100000.0))
                .with_description("Pay accrued interest"),
            // Fee income
            ExpectedFlow::new("4210", "1100", 0.15, (10.0, 10000.0))
                .with_description("Transaction fees collected"),
            // Loan loss provision
            ExpectedFlow::new("5300", "1255", 0.05, (1000.0, 100000.0))
                .with_description("Provision for expected losses"),
        ];

        template
    }

    /// Get template for a company archetype.
    pub fn for_archetype(archetype: &CompanyArchetype) -> Self {
        match archetype.industry() {
            Industry::Retail => Self::retail_standard(),
            Industry::SaaS => Self::saas_standard(),
            Industry::Manufacturing => Self::manufacturing_standard(),
            Industry::ProfessionalServices => Self::professional_services_standard(),
            Industry::FinancialServices => Self::financial_services_standard(),
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

    #[test]
    fn test_manufacturing_template() {
        let template = ChartOfAccountsTemplate::manufacturing_standard();
        assert_eq!(template.industry, Industry::Manufacturing);
        assert!(template.account_count() > 35);

        // Should have manufacturing-specific inventory accounts
        assert!(template.find_by_code("1310").is_some()); // Raw Materials
        assert!(template.find_by_code("1320").is_some()); // Work in Progress
        assert!(template.find_by_code("1330").is_some()); // Finished Goods

        // Should have manufacturing cost accounts
        assert!(template.find_by_code("5100").is_some()); // Direct Materials
        assert!(template.find_by_code("5200").is_some()); // Direct Labor
        assert!(template.find_by_code("5300").is_some()); // Manufacturing Overhead

        // Should have manufacturing equipment
        assert!(template.find_by_code("1520").is_some()); // Manufacturing Equipment

        // Should have expected flows
        assert!(!template.expected_flows.is_empty());
    }

    #[test]
    fn test_professional_services_template() {
        let template = ChartOfAccountsTemplate::professional_services_standard();
        assert_eq!(template.industry, Industry::ProfessionalServices);
        assert!(template.account_count() > 35);

        // Should have professional services-specific accounts
        assert!(template.find_by_code("1210").is_some()); // Unbilled Receivables
        assert!(template.find_by_code("1220").is_some()); // WIP - Billable
        assert!(template.find_by_code("2310").is_some()); // Client Retainers

        // Should have fee revenue accounts
        assert!(template.find_by_code("4110").is_some()); // Professional Fees - Hourly
        assert!(template.find_by_code("4130").is_some()); // Retainer Fees

        // Should have expected flows for billing cycle
        assert!(!template.expected_flows.is_empty());
    }

    #[test]
    fn test_financial_services_template() {
        let template = ChartOfAccountsTemplate::financial_services_standard();
        assert_eq!(template.industry, Industry::FinancialServices);
        assert!(template.account_count() > 40);

        // Should have securities accounts
        assert!(template.find_by_code("1110").is_some()); // Trading Securities
        assert!(template.find_by_code("1120").is_some()); // Available-for-Sale
        assert!(template.find_by_code("1130").is_some()); // Held-to-Maturity

        // Should have loan portfolio accounts
        assert!(template.find_by_code("1250").is_some()); // Loans Receivable
        assert!(template.find_by_code("1255").is_some()); // Allowance for Loan Losses

        // Should have deposit accounts
        assert!(template.find_by_code("2110").is_some()); // Customer Deposits - Demand
        assert!(template.find_by_code("2120").is_some()); // Customer Deposits - Savings

        // Should have interest income/expense
        assert!(template.find_by_code("4110").is_some()); // Interest Income - Loans
        assert!(template.find_by_code("5100").is_some()); // Interest Expense - Deposits

        // Should have custody accounts
        assert!(template.find_by_code("1610").is_some()); // Customer Custody Assets
        assert!(template.find_by_code("4220").is_some()); // Custody Fees

        // Should have expected flows
        assert!(!template.expected_flows.is_empty());
    }
}
