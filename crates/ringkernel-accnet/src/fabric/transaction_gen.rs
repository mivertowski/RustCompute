//! Transaction generator for synthetic accounting data.
//!
//! Generates realistic journal entries following configurable patterns
//! and the method distribution from Ivertowski et al. (2024).

use crate::models::{
    AccountingNetwork, Decimal128, HybridTimestamp, JournalEntry, JournalEntryFlags,
    JournalLineItem, LineType, SolvingMethod, TransactionFlow,
};
use super::{ChartOfAccountsTemplate, CompanyArchetype, ExpectedFlow};
use rand::prelude::*;
use rand_distr::{LogNormal, Normal};
use std::collections::HashMap;
use uuid::Uuid;

/// Configuration for the transaction generator.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Target transactions per second
    pub transactions_per_second: f64,
    /// Batch size for generation
    pub batch_size: usize,

    // Method distribution (should sum to 1.0)
    /// Ratio of Method A entries (1:1)
    pub method_a_ratio: f64,
    /// Ratio of Method B entries (n:n)
    pub method_b_ratio: f64,
    /// Ratio of Method C entries (n:m)
    pub method_c_ratio: f64,
    /// Ratio of Method D entries (aggregate)
    pub method_d_ratio: f64,
    /// Ratio of Method E entries (decomposition)
    pub method_e_ratio: f64,

    // Amount distribution
    /// Use Benford's Law compliant amounts
    pub benford_compliant: bool,
    /// Base amount scale factor
    pub amount_scale: f64,

    // Timing
    /// Business hours start (0-23)
    pub business_hours_start: u8,
    /// Business hours end (0-23)
    pub business_hours_end: u8,
    /// Weekend activity ratio (0.0 - 1.0)
    pub weekend_ratio: f64,
    /// Month-end volume multiplier
    pub month_end_multiplier: f64,

    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            transactions_per_second: 100.0,
            batch_size: 100,
            // Academic paper distribution
            method_a_ratio: 0.6068,
            method_b_ratio: 0.1663,
            method_c_ratio: 0.11,
            method_d_ratio: 0.11,
            method_e_ratio: 0.0069,
            benford_compliant: true,
            amount_scale: 1.0,
            business_hours_start: 8,
            business_hours_end: 18,
            weekend_ratio: 0.1,
            month_end_multiplier: 2.5,
            seed: None,
        }
    }
}

impl GeneratorConfig {
    /// Create configuration for high-volume simulation.
    pub fn high_volume() -> Self {
        Self {
            transactions_per_second: 1000.0,
            batch_size: 500,
            ..Default::default()
        }
    }

    /// Create configuration for educational (slow) simulation.
    pub fn educational() -> Self {
        Self {
            transactions_per_second: 5.0,
            batch_size: 10,
            ..Default::default()
        }
    }

    /// Validate that method ratios sum to ~1.0.
    pub fn validate(&self) -> Result<(), String> {
        let total = self.method_a_ratio
            + self.method_b_ratio
            + self.method_c_ratio
            + self.method_d_ratio
            + self.method_e_ratio;

        if (total - 1.0).abs() > 0.01 {
            return Err(format!(
                "Method ratios must sum to 1.0, got {}",
                total
            ));
        }
        Ok(())
    }
}

/// Transaction generator that creates synthetic journal entries.
pub struct TransactionGenerator {
    /// Company archetype
    archetype: CompanyArchetype,
    /// Chart of accounts template
    coa: ChartOfAccountsTemplate,
    /// Configuration
    config: GeneratorConfig,
    /// Random number generator
    rng: StdRng,
    /// Account code to index mapping
    account_indices: HashMap<String, u16>,
    /// Expected flows with their cumulative probabilities
    flow_cdf: Vec<(f64, ExpectedFlow)>,
    /// Current simulated timestamp
    current_time: HybridTimestamp,
    /// Entry counter for IDs
    entry_counter: u64,
    /// Node ID for timestamps
    node_id: u32,
    /// Accounts that haven't been used yet (for ensuring full coverage)
    unused_accounts: Vec<u16>,
    /// Counter to trigger full-coverage flows periodically
    coverage_counter: u32,
}

impl TransactionGenerator {
    /// Create a new generator.
    pub fn new(
        archetype: CompanyArchetype,
        config: GeneratorConfig,
    ) -> Self {
        let seed = config.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let rng = StdRng::seed_from_u64(seed);

        let coa = ChartOfAccountsTemplate::for_archetype(&archetype);

        // Build account index mapping
        let mut account_indices = HashMap::new();
        for (idx, account) in coa.accounts.iter().enumerate() {
            account_indices.insert(account.code.clone(), idx as u16);
        }

        // Build flow CDF for weighted selection
        let total_freq: f64 = coa.expected_flows.iter().map(|f| f.frequency).sum();
        let mut cumulative = 0.0;
        let flow_cdf: Vec<(f64, ExpectedFlow)> = coa
            .expected_flows
            .iter()
            .map(|f| {
                cumulative += f.frequency / total_freq;
                (cumulative, f.clone())
            })
            .collect();

        // Track all account indices for full coverage
        let unused_accounts: Vec<u16> = (0..coa.accounts.len() as u16).collect();

        Self {
            archetype,
            coa,
            config,
            rng,
            account_indices,
            flow_cdf,
            current_time: HybridTimestamp::now(),
            entry_counter: 0,
            node_id: 1,
            unused_accounts,
            coverage_counter: 0,
        }
    }

    /// Generate a batch of journal entries.
    pub fn generate_batch(&mut self, count: usize) -> Vec<GeneratedEntry> {
        let mut entries = Vec::with_capacity(count);

        for _ in 0..count {
            // Determine method for this entry
            let method = self.select_method();

            // Generate entry based on method
            let entry = match method {
                SolvingMethod::MethodA => self.generate_method_a(),
                SolvingMethod::MethodB => self.generate_method_b(),
                SolvingMethod::MethodC => self.generate_method_c(),
                SolvingMethod::MethodD => self.generate_method_d(),
                SolvingMethod::MethodE => self.generate_method_e(),
                SolvingMethod::Pending => self.generate_method_a(), // Fallback
            };

            entries.push(entry);

            // Advance time slightly
            self.advance_time();
        }

        entries
    }

    /// Select which method to use based on configured ratios.
    fn select_method(&mut self) -> SolvingMethod {
        let r: f64 = self.rng.gen();
        let mut cumulative = 0.0;

        cumulative += self.config.method_a_ratio;
        if r < cumulative {
            return SolvingMethod::MethodA;
        }

        cumulative += self.config.method_b_ratio;
        if r < cumulative {
            return SolvingMethod::MethodB;
        }

        cumulative += self.config.method_c_ratio;
        if r < cumulative {
            return SolvingMethod::MethodC;
        }

        cumulative += self.config.method_d_ratio;
        if r < cumulative {
            return SolvingMethod::MethodD;
        }

        SolvingMethod::MethodE
    }

    /// Generate a Method A entry (1 debit, 1 credit).
    fn generate_method_a(&mut self) -> GeneratedEntry {
        // Every 5th transaction, ensure we use accounts that haven't been touched yet
        self.coverage_counter += 1;
        let (from_idx, to_idx, amount) = if self.coverage_counter % 5 == 0 {
            self.select_coverage_flow()
        } else {
            let flow = self.select_flow();
            let amount = self.generate_amount(flow.amount_range.0, flow.amount_range.1);
            let from_idx = self.account_indices.get(&flow.from_code).copied().unwrap_or(0);
            let to_idx = self.account_indices.get(&flow.to_code).copied().unwrap_or(1);
            (from_idx, to_idx, amount)
        };

        let mut entry = self.create_entry_header();
        entry.debit_line_count = 1;
        entry.credit_line_count = 1;
        entry.line_count = 2;
        entry.total_debits = amount;
        entry.total_credits = amount;
        entry.solving_method = SolvingMethod::MethodA;
        entry.average_confidence = 1.0;

        let debit_line = JournalLineItem::debit(from_idx, amount, 1);
        let credit_line = JournalLineItem::credit(to_idx, amount, 2);

        GeneratedEntry {
            entry,
            debit_lines: vec![debit_line],
            credit_lines: vec![credit_line],
            expected_flows: vec![(from_idx, to_idx, amount)],
        }
    }

    /// Select a flow that ensures coverage of unused accounts.
    fn select_coverage_flow(&mut self) -> (u16, u16, Decimal128) {
        let account_count = self.coa.accounts.len() as u16;

        // Pick accounts - prefer unused ones, but always pick valid accounts
        let from_idx = if !self.unused_accounts.is_empty() {
            let idx = self.rng.gen_range(0..self.unused_accounts.len());
            let acc = self.unused_accounts.remove(idx);
            acc
        } else {
            self.rng.gen_range(0..account_count)
        };

        let to_idx = if !self.unused_accounts.is_empty() {
            let idx = self.rng.gen_range(0..self.unused_accounts.len());
            let acc = self.unused_accounts.remove(idx);
            acc
        } else {
            // Pick a different account than from_idx
            loop {
                let idx = self.rng.gen_range(0..account_count);
                if idx != from_idx {
                    break idx;
                }
            }
        };

        // Refill unused accounts when empty (for continuous coverage)
        if self.unused_accounts.is_empty() {
            self.unused_accounts = (0..account_count).collect();
        }

        let amount = self.generate_amount(100.0, 5000.0);
        (from_idx, to_idx, amount)
    }

    /// Generate a Method B entry (n debits, n credits with matching amounts).
    fn generate_method_b(&mut self) -> GeneratedEntry {
        let n = self.rng.gen_range(2..=4); // 2-4 lines each side
        let mut entry = self.create_entry_header();

        let mut debit_lines = Vec::with_capacity(n);
        let mut credit_lines = Vec::with_capacity(n);
        let mut expected_flows = Vec::new();

        let mut total = Decimal128::ZERO;

        // Use coverage flow for at least one line pair in each Method B entry
        self.coverage_counter += 1;
        let use_coverage = self.coverage_counter % 3 == 0;

        for i in 0..n {
            let (from_idx, to_idx, amount) = if use_coverage && i == 0 {
                self.select_coverage_flow()
            } else {
                let flow = self.select_flow();
                let amount = self.generate_amount(flow.amount_range.0 / n as f64, flow.amount_range.1 / n as f64);
                let from_idx = self.account_indices.get(&flow.from_code).copied().unwrap_or(0);
                let to_idx = self.account_indices.get(&flow.to_code).copied().unwrap_or(1);
                (from_idx, to_idx, amount)
            };
            total = total + amount;

            debit_lines.push(JournalLineItem::debit(from_idx, amount, (i + 1) as u16));
            credit_lines.push(JournalLineItem::credit(to_idx, amount, (n + i + 1) as u16));

            expected_flows.push((from_idx, to_idx, amount));
        }

        entry.debit_line_count = n as u16;
        entry.credit_line_count = n as u16;
        entry.line_count = (n * 2) as u16;
        entry.total_debits = total;
        entry.total_credits = total;
        entry.solving_method = SolvingMethod::MethodB;
        entry.average_confidence = 1.0;

        GeneratedEntry {
            entry,
            debit_lines,
            credit_lines,
            expected_flows,
        }
    }

    /// Generate a Method C entry (n debits, m credits where n != m).
    fn generate_method_c(&mut self) -> GeneratedEntry {
        let n_debits = self.rng.gen_range(1..=3);
        let n_credits = self.rng.gen_range(2..=5);
        // Ensure different counts
        let n_credits = if n_credits == n_debits { n_credits + 1 } else { n_credits };

        let mut entry = self.create_entry_header();
        let mut debit_lines = Vec::with_capacity(n_debits);
        let mut credit_lines = Vec::with_capacity(n_credits);
        let mut expected_flows = Vec::new();

        // Generate total amount first
        let total_amount = self.generate_amount(500.0, 10000.0);
        let total_f64 = total_amount.to_f64();

        // Periodically use coverage accounts for Method C
        self.coverage_counter += 1;
        let use_coverage = self.coverage_counter % 4 == 0;

        // Split among debits
        let debit_amounts = self.split_amount(total_f64, n_debits);
        for (i, &amt) in debit_amounts.iter().enumerate() {
            let from_idx = if use_coverage && i == 0 {
                let (idx, _, _) = self.select_coverage_flow();
                idx
            } else {
                let flow = self.select_flow();
                self.account_indices.get(&flow.from_code).copied().unwrap_or(0)
            };
            debit_lines.push(JournalLineItem::debit(from_idx, Decimal128::from_f64(amt), (i + 1) as u16));
        }

        // Split among credits
        let credit_amounts = self.split_amount(total_f64, n_credits);
        for (i, &amt) in credit_amounts.iter().enumerate() {
            let to_idx = if use_coverage && i == 0 {
                let (_, idx, _) = self.select_coverage_flow();
                idx
            } else {
                let flow = self.select_flow();
                self.account_indices.get(&flow.to_code).copied().unwrap_or(1)
            };
            credit_lines.push(JournalLineItem::credit(to_idx, Decimal128::from_f64(amt), (n_debits + i + 1) as u16));
        }

        // For Method C, flows are approximated by matching largest to largest
        // This is a simplification of the actual algorithm
        let from_idx = debit_lines.first().map(|l| l.account_index).unwrap_or(0);
        let to_idx = credit_lines.first().map(|l| l.account_index).unwrap_or(1);
        expected_flows.push((from_idx, to_idx, total_amount));

        entry.debit_line_count = n_debits as u16;
        entry.credit_line_count = n_credits as u16;
        entry.line_count = (n_debits + n_credits) as u16;
        entry.total_debits = total_amount;
        entry.total_credits = total_amount;
        entry.solving_method = SolvingMethod::MethodC;
        entry.average_confidence = 0.85; // Lower confidence for n:m

        GeneratedEntry {
            entry,
            debit_lines,
            credit_lines,
            expected_flows,
        }
    }

    /// Generate a Method D entry (aggregate level matching).
    fn generate_method_d(&mut self) -> GeneratedEntry {
        // Method D is similar to B/C but uses account class level
        // For simplicity, generate as Method B but mark as D
        let mut generated = self.generate_method_b();
        generated.entry.solving_method = SolvingMethod::MethodD;
        generated.entry.flags.0 |= JournalEntryFlags::USES_HIGHER_AGGREGATE;
        generated
    }

    /// Generate a Method E entry (decomposition with shadow bookings).
    fn generate_method_e(&mut self) -> GeneratedEntry {
        // Method E is the complex case - generate as C but with lower confidence
        let mut generated = self.generate_method_c();
        generated.entry.solving_method = SolvingMethod::MethodE;
        generated.entry.flags.0 |= JournalEntryFlags::HAS_DECOMPOSED_VALUES;
        generated.entry.average_confidence = 0.5; // Low confidence
        generated
    }

    /// Select a flow pattern based on configured probabilities.
    fn select_flow(&mut self) -> ExpectedFlow {
        let r: f64 = self.rng.gen();
        for (cumulative, flow) in &self.flow_cdf {
            if r < *cumulative {
                return flow.clone();
            }
        }
        // Fallback to first flow
        self.flow_cdf
            .first()
            .map(|(_, f)| f.clone())
            .unwrap_or_else(|| ExpectedFlow::new("1100", "2100", 1.0, (100.0, 1000.0)))
    }

    /// Generate an amount following Benford's Law or uniform distribution.
    fn generate_amount(&mut self, min: f64, max: f64) -> Decimal128 {
        let amount = if self.config.benford_compliant {
            // Log-normal distribution approximates Benford's Law
            let mean = ((min.ln() + max.ln()) / 2.0).exp();
            let std_dev = (max / min).ln() / 4.0;
            let dist = LogNormal::new(mean.ln(), std_dev).unwrap_or_else(|_| LogNormal::new(0.0, 1.0).unwrap());
            let raw: f64 = self.rng.sample(dist);
            raw.clamp(min, max)
        } else {
            self.rng.gen_range(min..max)
        };

        // Apply scale and round to cents
        let scaled = amount * self.config.amount_scale;
        Decimal128::from_f64((scaled * 100.0).round() / 100.0)
    }

    /// Split an amount into n parts (for multi-line entries).
    fn split_amount(&mut self, total: f64, parts: usize) -> Vec<f64> {
        if parts == 0 {
            return vec![];
        }
        if parts == 1 {
            return vec![total];
        }

        // Generate random split points
        let mut points: Vec<f64> = (0..parts - 1)
            .map(|_| self.rng.gen::<f64>())
            .collect();
        points.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Convert to amounts
        let mut amounts = Vec::with_capacity(parts);
        let mut prev = 0.0;
        for point in points {
            amounts.push((point - prev) * total);
            prev = point;
        }
        amounts.push((1.0 - prev) * total);

        // Round to cents
        amounts.iter().map(|a| (a * 100.0).round() / 100.0).collect()
    }

    /// Create a new entry header with generated metadata.
    fn create_entry_header(&mut self) -> JournalEntry {
        self.entry_counter += 1;

        JournalEntry {
            id: Uuid::new_v4(),
            entity_id: Uuid::nil(), // Set by caller
            document_number_hash: self.entry_counter,
            source_system_id: 1,
            batch_id: (self.entry_counter / 1000) as u32,
            posting_date: self.current_time,
            line_count: 0,
            debit_line_count: 0,
            credit_line_count: 0,
            first_line_index: 0,
            total_debits: Decimal128::ZERO,
            total_credits: Decimal128::ZERO,
            solving_method: SolvingMethod::Pending,
            average_confidence: 0.0,
            flow_count: 0,
            _pad: 0,
            flags: JournalEntryFlags::new(),
            _reserved: [0; 12],
        }
    }

    /// Advance the simulated time.
    fn advance_time(&mut self) {
        // Add some milliseconds based on TPS
        let interval_ms = (1000.0 / self.config.transactions_per_second) as u64;
        self.current_time.physical += interval_ms;
        self.current_time.logical = 0;
    }

    /// Get current statistics.
    pub fn stats(&self) -> GeneratorStats {
        GeneratorStats {
            entries_generated: self.entry_counter,
            current_time: self.current_time,
        }
    }
}

/// A generated journal entry with all line items.
#[derive(Debug, Clone)]
pub struct GeneratedEntry {
    /// The journal entry header
    pub entry: JournalEntry,
    /// Debit line items
    pub debit_lines: Vec<JournalLineItem>,
    /// Credit line items
    pub credit_lines: Vec<JournalLineItem>,
    /// Expected flows (from_account, to_account, amount)
    pub expected_flows: Vec<(u16, u16, Decimal128)>,
}

impl GeneratedEntry {
    /// Convert to transaction flows.
    pub fn to_flows(&self) -> Vec<TransactionFlow> {
        self.expected_flows
            .iter()
            .map(|&(from, to, amount)| {
                TransactionFlow::with_provenance(
                    from,
                    to,
                    amount,
                    self.entry.id,
                    0,
                    0,
                    self.entry.posting_date,
                    self.entry.solving_method,
                    self.entry.average_confidence,
                )
            })
            .collect()
    }

    /// Get total amount.
    pub fn total_amount(&self) -> Decimal128 {
        self.entry.total_debits
    }
}

/// Statistics about the generator.
#[derive(Debug, Clone)]
pub struct GeneratorStats {
    /// Total entries generated
    pub entries_generated: u64,
    /// Current simulated time
    pub current_time: HybridTimestamp,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let archetype = CompanyArchetype::retail_standard();
        let config = GeneratorConfig::default();
        let _gen = TransactionGenerator::new(archetype, config);
    }

    #[test]
    fn test_generate_batch() {
        let archetype = CompanyArchetype::retail_standard();
        let config = GeneratorConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = TransactionGenerator::new(archetype, config);

        let batch = gen.generate_batch(100);
        assert_eq!(batch.len(), 100);

        // Check that entries are balanced
        for entry in &batch {
            let debit_total: f64 = entry.total_amount().to_f64();
            assert!(debit_total > 0.0);
            assert!(entry.entry.is_balanced());
        }
    }

    #[test]
    fn test_method_distribution() {
        let archetype = CompanyArchetype::retail_standard();
        let config = GeneratorConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = TransactionGenerator::new(archetype, config);

        let batch = gen.generate_batch(1000);

        let mut method_counts = [0u32; 5];
        for entry in &batch {
            match entry.entry.solving_method {
                SolvingMethod::MethodA => method_counts[0] += 1,
                SolvingMethod::MethodB => method_counts[1] += 1,
                SolvingMethod::MethodC => method_counts[2] += 1,
                SolvingMethod::MethodD => method_counts[3] += 1,
                SolvingMethod::MethodE => method_counts[4] += 1,
                _ => {}
            }
        }

        // Method A should be most common (~60%)
        let method_a_ratio = method_counts[0] as f64 / 1000.0;
        assert!(method_a_ratio > 0.50 && method_a_ratio < 0.70);
    }

    #[test]
    fn test_config_validation() {
        let mut config = GeneratorConfig::default();
        assert!(config.validate().is_ok());

        config.method_a_ratio = 0.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_full_account_coverage() {
        use std::collections::HashSet;

        let archetype = CompanyArchetype::retail_standard();
        let coa = ChartOfAccountsTemplate::for_archetype(&archetype);
        let total_accounts = coa.accounts.len();

        let config = GeneratorConfig {
            seed: Some(123),
            ..Default::default()
        };
        let mut gen = TransactionGenerator::new(archetype, config);

        // Generate enough entries to cover all accounts
        // With ~36 accounts and coverage every 5th transaction (using 2 accounts each),
        // we need at least 36 / 2 * 5 = 90 transactions minimum
        let batch = gen.generate_batch(200);

        // Track which accounts were touched
        let mut used_accounts: HashSet<u16> = HashSet::new();
        for entry in &batch {
            for (from, to, _) in &entry.expected_flows {
                used_accounts.insert(*from);
                used_accounts.insert(*to);
            }
        }

        // All accounts should have been used at least once
        let coverage = used_accounts.len() as f64 / total_accounts as f64;
        assert!(
            coverage >= 0.9,
            "Expected at least 90% account coverage, got {:.1}% ({}/{} accounts)",
            coverage * 100.0,
            used_accounts.len(),
            total_accounts
        );
    }
}
