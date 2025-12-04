//! Accounting network graph representation.
//!
//! The network is a directed graph where:
//! - Nodes = Accounts
//! - Edges = Transaction flows (monetary movements)
//!
//! This provides the foundation for graph analytics like centrality,
//! cycle detection, and anomaly identification.

use super::{
    AccountFlags, AccountMetadata, AccountNode, AccountType, AggregatedFlow, FlowDirection,
    GraphEdge, HybridTimestamp, TransactionFlow,
};
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Statistics about the accounting network.
#[derive(Debug, Clone, Default)]
pub struct NetworkStatistics {
    /// Number of account nodes
    pub node_count: usize,
    /// Number of flow edges
    pub edge_count: usize,
    /// Number of unique edges (ignoring direction)
    pub unique_edge_count: usize,
    /// Graph density (edges / possible edges)
    pub density: f64,
    /// Average node degree (in + out)
    pub avg_degree: f64,
    /// Maximum node degree
    pub max_degree: u32,
    /// Number of connected components
    pub component_count: usize,
    /// Total monetary flow
    pub total_flow_amount: f64,
    /// Average flow confidence
    pub avg_confidence: f64,
    /// Number of suspense accounts detected
    pub suspense_account_count: usize,
    /// Number of GAAP violations
    pub gaap_violation_count: usize,
    /// Number of fraud patterns detected
    pub fraud_pattern_count: usize,
    /// Timestamp of last update
    pub last_updated: HybridTimestamp,
}

/// The complete accounting network graph.
#[derive(Debug, Clone)]
pub struct AccountingNetwork {
    /// Network identifier
    pub id: Uuid,
    /// Entity (company) this network belongs to
    pub entity_id: Uuid,
    /// Fiscal year
    pub fiscal_year: u16,
    /// Fiscal period (1-12 for monthly, 1-4 for quarterly)
    pub fiscal_period: u8,

    /// Account nodes (indexed by account index 0-255)
    pub accounts: Vec<AccountNode>,
    /// Account metadata (names, codes, etc.)
    pub account_metadata: HashMap<u16, AccountMetadata>,

    /// Transaction flows (edges)
    pub flows: Vec<TransactionFlow>,
    /// Aggregated flows by (source, target) pair
    pub aggregated_flows: HashMap<(u16, u16), AggregatedFlow>,

    /// Adjacency list: for each account, list of (target, flow_index)
    pub adjacency_out: Vec<Vec<(u16, usize)>>,
    /// Reverse adjacency: for each account, list of (source, flow_index)
    pub adjacency_in: Vec<Vec<(u16, usize)>>,

    /// Network statistics
    pub statistics: NetworkStatistics,

    /// Period start timestamp.
    pub period_start: HybridTimestamp,
    /// Period end timestamp.
    pub period_end: HybridTimestamp,
}

impl AccountingNetwork {
    /// Create a new empty network.
    pub fn new(entity_id: Uuid, fiscal_year: u16, fiscal_period: u8) -> Self {
        Self {
            id: Uuid::new_v4(),
            entity_id,
            fiscal_year,
            fiscal_period,
            accounts: Vec::new(),
            account_metadata: HashMap::new(),
            flows: Vec::new(),
            aggregated_flows: HashMap::new(),
            adjacency_out: Vec::new(),
            adjacency_in: Vec::new(),
            statistics: NetworkStatistics::default(),
            period_start: HybridTimestamp::zero(),
            period_end: HybridTimestamp::zero(),
        }
    }

    /// Add an account to the network.
    pub fn add_account(&mut self, mut account: AccountNode, metadata: AccountMetadata) -> u16 {
        let index = self.accounts.len() as u16;
        account.index = index;
        self.accounts.push(account);
        self.account_metadata.insert(index, metadata);
        self.adjacency_out.push(Vec::new());
        self.adjacency_in.push(Vec::new());
        self.statistics.node_count = self.accounts.len();
        index
    }

    /// Add a flow to the network.
    pub fn add_flow(&mut self, flow: TransactionFlow) {
        let flow_index = self.flows.len();
        let source = flow.source_account_index;
        let target = flow.target_account_index;

        // Update adjacency lists
        if (source as usize) < self.adjacency_out.len() {
            self.adjacency_out[source as usize].push((target, flow_index));
        }
        if (target as usize) < self.adjacency_in.len() {
            self.adjacency_in[target as usize].push((source, flow_index));
        }

        // Update aggregated flows
        let key = (source, target);
        self.aggregated_flows
            .entry(key)
            .or_insert_with(|| AggregatedFlow::new(source, target))
            .add(&flow);

        // Update account degrees and balances (using saturating add to prevent overflow)
        let flow_amount = flow.amount;
        if (source as usize) < self.accounts.len() {
            let acc = &mut self.accounts[source as usize];
            acc.out_degree = acc.out_degree.saturating_add(1);
            acc.transaction_count = acc.transaction_count.saturating_add(1);
            // Source account: debit (outflow)
            acc.total_debits = acc.total_debits + flow_amount;
            acc.closing_balance = acc.closing_balance - flow_amount;
        }
        if (target as usize) < self.accounts.len() {
            let acc = &mut self.accounts[target as usize];
            acc.in_degree = acc.in_degree.saturating_add(1);
            acc.transaction_count = acc.transaction_count.saturating_add(1);
            // Target account: credit (inflow)
            acc.total_credits = acc.total_credits + flow_amount;
            acc.closing_balance = acc.closing_balance + flow_amount;
        }

        // Update timestamps
        if self.period_start.physical == 0 || flow.timestamp < self.period_start {
            self.period_start = flow.timestamp;
        }
        if flow.timestamp > self.period_end {
            self.period_end = flow.timestamp;
        }

        self.flows.push(flow);
        self.statistics.edge_count = self.flows.len();
        self.statistics.unique_edge_count = self.aggregated_flows.len();
    }

    /// Incorporate multiple flows efficiently.
    pub fn incorporate_flows(&mut self, flows: &[TransactionFlow]) {
        for flow in flows {
            self.add_flow(flow.clone());
        }
        self.update_statistics();
    }

    /// Update network statistics.
    pub fn update_statistics(&mut self) {
        let n = self.accounts.len();
        let e = self.aggregated_flows.len();

        // Density: e / (n * (n-1)) for directed graph
        self.statistics.density = if n > 1 {
            e as f64 / (n * (n - 1)) as f64
        } else {
            0.0
        };

        // Average degree
        let total_degree: u32 = self
            .accounts
            .iter()
            .map(|a| a.in_degree as u32 + a.out_degree as u32)
            .sum();
        self.statistics.avg_degree = if n > 0 {
            total_degree as f64 / n as f64
        } else {
            0.0
        };

        // Max degree
        self.statistics.max_degree = self
            .accounts
            .iter()
            .map(|a| a.in_degree as u32 + a.out_degree as u32)
            .max()
            .unwrap_or(0);

        // Total flow and average confidence
        let mut total_amount = 0.0;
        let mut total_confidence = 0.0;
        for flow in &self.flows {
            total_amount += flow.amount.to_f64().abs();
            total_confidence += flow.confidence as f64;
        }
        self.statistics.total_flow_amount = total_amount;
        self.statistics.avg_confidence = if !self.flows.is_empty() {
            total_confidence / self.flows.len() as f64
        } else {
            0.0
        };

        // Count flagged accounts
        self.statistics.suspense_account_count = self
            .accounts
            .iter()
            .filter(|a| a.flags.has(AccountFlags::IS_SUSPENSE_ACCOUNT))
            .count();
        self.statistics.gaap_violation_count = self
            .accounts
            .iter()
            .filter(|a| a.flags.has(AccountFlags::HAS_GAAP_VIOLATION))
            .count();
        self.statistics.fraud_pattern_count = self
            .accounts
            .iter()
            .filter(|a| a.flags.has(AccountFlags::HAS_FRAUD_PATTERN))
            .count();

        self.statistics.last_updated = HybridTimestamp::now();
    }

    /// Get neighbors of an account.
    pub fn neighbors(&self, account_index: u16, direction: FlowDirection) -> Vec<u16> {
        let mut result = Vec::new();
        let idx = account_index as usize;

        match direction {
            FlowDirection::Outflow => {
                if idx < self.adjacency_out.len() {
                    for &(target, _) in &self.adjacency_out[idx] {
                        if !result.contains(&target) {
                            result.push(target);
                        }
                    }
                }
            }
            FlowDirection::Inflow => {
                if idx < self.adjacency_in.len() {
                    for &(source, _) in &self.adjacency_in[idx] {
                        if !result.contains(&source) {
                            result.push(source);
                        }
                    }
                }
            }
            FlowDirection::Both => {
                result.extend(self.neighbors(account_index, FlowDirection::Outflow));
                for neighbor in self.neighbors(account_index, FlowDirection::Inflow) {
                    if !result.contains(&neighbor) {
                        result.push(neighbor);
                    }
                }
            }
        }

        result
    }

    /// Get all edges as GraphEdge structs (for algorithms).
    pub fn edges(&self) -> Vec<GraphEdge> {
        self.aggregated_flows
            .iter()
            .map(|(&(from, to), agg)| GraphEdge {
                from,
                to,
                weight: agg.total_amount,
            })
            .collect()
    }

    /// Get account by index.
    pub fn get_account(&self, index: u16) -> Option<&AccountNode> {
        self.accounts.get(index as usize)
    }

    /// Get account metadata by index.
    pub fn get_metadata(&self, index: u16) -> Option<&AccountMetadata> {
        self.account_metadata.get(&index)
    }

    /// Find account by code.
    pub fn find_by_code(&self, code: &str) -> Option<u16> {
        for (idx, meta) in &self.account_metadata {
            if meta.code == code {
                return Some(*idx);
            }
        }
        None
    }

    /// Find account by name (partial match).
    pub fn find_by_name(&self, name: &str) -> Vec<u16> {
        let lower_name = name.to_lowercase();
        self.account_metadata
            .iter()
            .filter(|(_, meta)| meta.name.to_lowercase().contains(&lower_name))
            .map(|(&idx, _)| idx)
            .collect()
    }

    /// Get accounts by type.
    pub fn accounts_by_type(&self, account_type: AccountType) -> Vec<u16> {
        self.accounts
            .iter()
            .filter(|a| a.account_type == account_type)
            .map(|a| a.index)
            .collect()
    }

    /// Calculate PageRank for all nodes.
    /// Uses power iteration method.
    pub fn calculate_pagerank(&mut self, damping: f64, iterations: usize) {
        let n = self.accounts.len();
        if n == 0 {
            return;
        }

        let mut pagerank = vec![1.0 / n as f64; n];
        let mut new_pagerank = vec![0.0; n];

        for _ in 0..iterations {
            new_pagerank.fill((1.0 - damping) / n as f64);

            for (i, account) in self.accounts.iter().enumerate() {
                let out_degree = account.out_degree as usize;
                if out_degree > 0 {
                    let contribution = damping * pagerank[i] / out_degree as f64;
                    for &(target, _) in &self.adjacency_out[i] {
                        new_pagerank[target as usize] += contribution;
                    }
                } else {
                    // Dangling node: distribute evenly
                    let contribution = damping * pagerank[i] / n as f64;
                    for pr in new_pagerank.iter_mut() {
                        *pr += contribution;
                    }
                }
            }

            std::mem::swap(&mut pagerank, &mut new_pagerank);
        }

        // Store results
        for (i, account) in self.accounts.iter_mut().enumerate() {
            account.pagerank = pagerank[i] as f32;
        }
    }

    /// Create a snapshot of current state for visualization.
    pub fn snapshot(&self) -> NetworkSnapshot {
        NetworkSnapshot {
            timestamp: HybridTimestamp::now(),
            node_count: self.accounts.len(),
            edge_count: self.flows.len(),
            unique_edges: self.aggregated_flows.len(),
            total_flow: self.statistics.total_flow_amount,
            avg_confidence: self.statistics.avg_confidence,
            suspense_count: self.statistics.suspense_account_count,
            violation_count: self.statistics.gaap_violation_count,
            fraud_count: self.statistics.fraud_pattern_count,
        }
    }

    /// Compute PageRank scores without modifying the network.
    /// Returns a vector of PageRank scores (one per account).
    pub fn compute_pagerank(&self, iterations: usize, damping: f64) -> Vec<f64> {
        let n = self.accounts.len();
        if n == 0 {
            return Vec::new();
        }

        let mut pagerank = vec![1.0 / n as f64; n];
        let mut new_pagerank = vec![0.0; n];

        for _ in 0..iterations {
            new_pagerank.fill((1.0 - damping) / n as f64);

            for (i, account) in self.accounts.iter().enumerate() {
                let out_degree = account.out_degree as usize;
                if out_degree > 0 {
                    let contribution = damping * pagerank[i] / out_degree as f64;
                    for &(target, _) in &self.adjacency_out[i] {
                        new_pagerank[target as usize] += contribution;
                    }
                } else {
                    // Dangling node: distribute evenly
                    let contribution = damping * pagerank[i] / n as f64;
                    for pr in new_pagerank.iter_mut() {
                        *pr += contribution;
                    }
                }
            }

            std::mem::swap(&mut pagerank, &mut new_pagerank);
        }

        pagerank
    }
}

/// Lightweight snapshot of network state for UI updates.
#[derive(Debug, Clone)]
pub struct NetworkSnapshot {
    /// Timestamp of the snapshot.
    pub timestamp: HybridTimestamp,
    /// Number of account nodes.
    pub node_count: usize,
    /// Number of transaction edges.
    pub edge_count: usize,
    /// Number of unique account pairs.
    pub unique_edges: usize,
    /// Total monetary flow in the network.
    pub total_flow: f64,
    /// Average confidence across all flows.
    pub avg_confidence: f64,
    /// Number of suspense accounts detected.
    pub suspense_count: usize,
    /// Number of GAAP violations.
    pub violation_count: usize,
    /// Number of fraud patterns.
    pub fraud_count: usize,
}

/// GPU-compatible network header structure.
/// Used for kernel dispatch.
#[derive(Debug, Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C, align(128))]
pub struct GpuNetworkHeader {
    /// Network ID (first 8 bytes of UUID)
    pub network_id: u64,
    /// Entity ID (first 8 bytes of UUID)
    pub entity_id: u64,
    /// Fiscal year
    pub fiscal_year: u16,
    /// Fiscal period
    pub fiscal_period: u8,
    /// Number of accounts (max 256 for GPU)
    pub account_count: u8,
    /// Number of flows
    pub flow_count: u32,
    /// Padding
    pub _pad1: [u8; 4],
    /// Period start timestamp
    pub period_start: HybridTimestamp,
    /// Period end timestamp
    pub period_end: HybridTimestamp,
    /// Network density
    pub density: f32,
    /// Average degree
    pub avg_degree: f32,
    /// Reserved
    pub _reserved: [u8; 72],
}

impl From<&AccountingNetwork> for GpuNetworkHeader {
    fn from(network: &AccountingNetwork) -> Self {
        Self {
            network_id: network.id.as_u128() as u64,
            entity_id: network.entity_id.as_u128() as u64,
            fiscal_year: network.fiscal_year,
            fiscal_period: network.fiscal_period,
            account_count: network.accounts.len().min(256) as u8,
            flow_count: network.flows.len() as u32,
            _pad1: [0; 4],
            period_start: network.period_start,
            period_end: network.period_end,
            density: network.statistics.density as f32,
            avg_degree: network.statistics.avg_degree as f32,
            _reserved: [0; 72],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Decimal128;

    fn create_test_network() -> AccountingNetwork {
        let mut network = AccountingNetwork::new(Uuid::new_v4(), 2024, 1);

        // Add accounts
        let cash = AccountNode::new(Uuid::new_v4(), AccountType::Asset, 0);
        let ar = AccountNode::new(Uuid::new_v4(), AccountType::Asset, 1);
        let revenue = AccountNode::new(Uuid::new_v4(), AccountType::Revenue, 2);

        network.add_account(cash, AccountMetadata::new("1100", "Cash"));
        network.add_account(ar, AccountMetadata::new("1200", "Accounts Receivable"));
        network.add_account(revenue, AccountMetadata::new("4000", "Sales Revenue"));

        // Add flows: Revenue -> A/R -> Cash
        let flow1 = TransactionFlow::new(
            2, // Revenue (credit side is source in double-entry)
            1, // A/R
            Decimal128::from_f64(1000.0),
            Uuid::new_v4(),
            HybridTimestamp::now(),
        );
        let flow2 = TransactionFlow::new(
            1, // A/R
            0, // Cash
            Decimal128::from_f64(1000.0),
            Uuid::new_v4(),
            HybridTimestamp::now(),
        );

        network.add_flow(flow1);
        network.add_flow(flow2);

        network
    }

    #[test]
    fn test_network_creation() {
        let network = create_test_network();
        assert_eq!(network.accounts.len(), 3);
        assert_eq!(network.flows.len(), 2);
        assert_eq!(network.aggregated_flows.len(), 2);
    }

    #[test]
    fn test_neighbors() {
        let network = create_test_network();

        // A/R (index 1) should have:
        // - Inflow from Revenue (2)
        // - Outflow to Cash (0)
        let in_neighbors = network.neighbors(1, FlowDirection::Inflow);
        let out_neighbors = network.neighbors(1, FlowDirection::Outflow);

        assert!(in_neighbors.contains(&2));
        assert!(out_neighbors.contains(&0));
    }

    #[test]
    fn test_pagerank() {
        let mut network = create_test_network();
        network.calculate_pagerank(0.85, 20);

        // Cash (end of chain) should have highest PageRank
        let cash_pr = network.accounts[0].pagerank;
        let revenue_pr = network.accounts[2].pagerank;
        assert!(cash_pr > revenue_pr);
    }

    #[test]
    fn test_gpu_header_size() {
        let size = std::mem::size_of::<GpuNetworkHeader>();
        assert!(
            size >= 128,
            "GpuNetworkHeader should be at least 128 bytes, got {}",
            size
        );
        assert!(
            size.is_multiple_of(128),
            "GpuNetworkHeader should be 128-byte aligned, got {}",
            size
        );
    }
}
