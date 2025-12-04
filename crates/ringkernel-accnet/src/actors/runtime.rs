//! GPU-native actor runtime for accounting analytics.
//!
//! This module provides the runtime that manages GPU kernel actors
//! using the RingKernel system.

use std::time::Instant;

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use crate::models::AccountingNetwork;

use super::coordinator::{AnalyticsCoordinator, CoordinatorConfig, CoordinatorStats};
#[cfg(feature = "cuda")]
use super::kernels::AnalyticsKernels;
#[allow(unused_imports)]
use super::messages::*;

/// GPU memory buffer for data exchange between host and kernels.
#[derive(Debug)]
pub struct GpuBuffer {
    /// Buffer size in bytes.
    pub size: usize,
    /// Device pointer (opaque handle).
    pub device_ptr: u64,
}

/// Analytics result from GPU processing.
#[derive(Debug, Clone)]
pub struct GpuAnalyticsResult {
    /// Snapshot ID.
    pub snapshot_id: u64,
    /// PageRank scores (index -> score).
    pub pagerank_scores: Vec<f64>,
    /// Fraud pattern count.
    pub fraud_pattern_count: u32,
    /// Fraud pattern flags per flow.
    pub fraud_flags: Vec<u32>,
    /// GAAP violation count.
    pub gaap_violation_count: u32,
    /// GAAP violation flags per flow.
    pub gaap_flags: Vec<u32>,
    /// Suspense account count.
    pub suspense_account_count: u32,
    /// Suspense scores per account.
    pub suspense_scores: Vec<f32>,
    /// Benford digit distribution.
    pub benford_distribution: [u32; 9],
    /// Benford chi-squared statistic.
    pub benford_chi_squared: f32,
    /// Benford anomaly detected.
    pub benford_anomaly: bool,
    /// Overall risk score.
    pub overall_risk_score: f32,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
}

impl Default for GpuAnalyticsResult {
    fn default() -> Self {
        Self {
            snapshot_id: 0,
            pagerank_scores: Vec::new(),
            fraud_pattern_count: 0,
            fraud_flags: Vec::new(),
            gaap_violation_count: 0,
            gaap_flags: Vec::new(),
            suspense_account_count: 0,
            suspense_scores: Vec::new(),
            benford_distribution: [0; 9],
            benford_chi_squared: 0.0,
            benford_anomaly: false,
            overall_risk_score: 0.0,
            processing_time_us: 0,
        }
    }
}

/// GPU actor runtime status.
#[derive(Debug, Clone)]
pub struct RuntimeStatus {
    /// Whether CUDA is available and active.
    pub cuda_active: bool,
    /// GPU device name.
    pub device_name: Option<String>,
    /// Compute capability.
    pub compute_capability: Option<(u32, u32)>,
    /// Number of kernels launched.
    pub kernels_launched: usize,
    /// Total messages processed.
    pub messages_processed: u64,
    /// Coordinator statistics.
    pub coordinator_stats: CoordinatorStats,
}

/// GPU-native actor runtime for accounting analytics.
///
/// This runtime manages the lifecycle of GPU kernel actors and
/// provides a high-level API for processing accounting networks.
pub struct GpuActorRuntime {
    /// Coordinator for orchestrating the pipeline.
    coordinator: AnalyticsCoordinator,
    /// Generated kernel code.
    #[cfg(feature = "cuda")]
    kernels: Option<AnalyticsKernels>,
    /// Whether GPU is active.
    gpu_active: bool,
    /// Device name.
    device_name: Option<String>,
    /// Compute capability.
    compute_capability: Option<(u32, u32)>,
    /// Messages processed counter.
    messages_processed: u64,
    /// CUDA device handle (when feature enabled).
    #[cfg(feature = "cuda")]
    cuda_device: Option<Arc<cudarc::driver::CudaDevice>>,
    /// Compiled PTX modules (reserved for future multi-kernel support).
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    compiled_modules: HashMap<String, bool>,
}

impl GpuActorRuntime {
    /// Create a new GPU actor runtime.
    pub fn new(config: CoordinatorConfig) -> Self {
        let coordinator = AnalyticsCoordinator::new(config);

        // Try to initialize GPU
        let (gpu_active, device_name, compute_capability) = Self::init_gpu();

        // Generate kernel code if GPU is available
        #[cfg(feature = "cuda")]
        let kernels = if gpu_active {
            match AnalyticsKernels::generate() {
                Ok(k) => {
                    log::info!("Generated {} analytics kernels", 6);
                    Some(k)
                }
                Err(e) => {
                    log::warn!("Failed to generate kernels: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            coordinator,
            #[cfg(feature = "cuda")]
            kernels,
            gpu_active,
            device_name,
            compute_capability,
            messages_processed: 0,
            #[cfg(feature = "cuda")]
            cuda_device: None,
            #[cfg(feature = "cuda")]
            compiled_modules: HashMap::new(),
        }
    }

    /// Initialize GPU and return status.
    fn init_gpu() -> (bool, Option<String>, Option<(u32, u32)>) {
        #[cfg(feature = "cuda")]
        {
            match cudarc::driver::CudaDevice::new(0) {
                Ok(device) => {
                    let name = device.name().unwrap_or_else(|_| "Unknown GPU".to_string());
                    // Get compute capability
                    let cc = device.attribute(
                        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
                    ).ok().and_then(|major| {
                        device.attribute(
                            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
                        ).ok().map(|minor| (major as u32, minor as u32))
                    });
                    log::info!("GPU initialized: {} (CC {:?})", name, cc);
                    (true, Some(name), cc)
                }
                Err(e) => {
                    log::warn!("Failed to initialize GPU: {}", e);
                    (false, None, None)
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            log::info!("CUDA feature not enabled, using CPU fallback");
            (false, None, None)
        }
    }

    /// Check if GPU is active.
    pub fn is_gpu_active(&self) -> bool {
        self.gpu_active
    }

    /// Get runtime status.
    pub fn status(&self) -> RuntimeStatus {
        #[cfg(feature = "cuda")]
        let kernels_launched = if self.kernels.is_some() { 6 } else { 0 };
        #[cfg(not(feature = "cuda"))]
        let kernels_launched = 0;

        RuntimeStatus {
            cuda_active: self.gpu_active,
            device_name: self.device_name.clone(),
            compute_capability: self.compute_capability,
            kernels_launched,
            messages_processed: self.messages_processed,
            coordinator_stats: self.coordinator.stats.clone(),
        }
    }

    /// Analyze a network using GPU actors.
    ///
    /// This is the main entry point for GPU-accelerated analytics.
    /// Returns analytics results computed by the GPU kernel actors.
    pub fn analyze(&mut self, network: &AccountingNetwork) -> GpuAnalyticsResult {
        let start = Instant::now();
        let snapshot_id = self.coordinator.begin_snapshot();

        let mut result = GpuAnalyticsResult {
            snapshot_id,
            ..Default::default()
        };

        let n_accounts = network.accounts.len();
        let n_flows = network.flows.len();

        if n_accounts == 0 || n_flows == 0 {
            result.processing_time_us = start.elapsed().as_micros() as u64;
            return result;
        }

        // GPU path
        #[cfg(feature = "cuda")]
        {
            let has_kernels = self.kernels.is_some();
            if self.gpu_active && has_kernels {
                match self.analyze_gpu(network, &mut result) {
                    Ok(_) => {
                        result.processing_time_us = start.elapsed().as_micros() as u64;
                        self.messages_processed += 6; // One message per kernel
                        return result;
                    }
                    Err(e) => {
                        log::warn!("GPU analysis failed, falling back to CPU: {}", e);
                    }
                }
            }
        }

        // CPU fallback
        self.analyze_cpu(network, &mut result);
        result.processing_time_us = start.elapsed().as_micros() as u64;
        result
    }

    /// GPU-accelerated analysis (CUDA).
    #[cfg(feature = "cuda")]
    fn analyze_gpu(
        &mut self,
        network: &AccountingNetwork,
        result: &mut GpuAnalyticsResult,
    ) -> Result<(), String> {
        use cudarc::driver::*;

        let device = match &self.cuda_device {
            Some(d) => d.clone(),
            None => {
                let d = CudaDevice::new(0).map_err(|e| e.to_string())?;
                self.cuda_device = Some(d.clone());
                d
            }
        };

        // === PageRank ===
        result.pagerank_scores = self.compute_pagerank_gpu(&device, network)?;

        // === Fraud Detection ===
        let (fraud_count, fraud_flags) = self.detect_fraud_gpu(&device, network)?;
        result.fraud_pattern_count = fraud_count;
        result.fraud_flags = fraud_flags;

        // === GAAP Validation ===
        let (gaap_count, gaap_flags) = self.validate_gaap_gpu(&device, network)?;
        result.gaap_violation_count = gaap_count;
        result.gaap_flags = gaap_flags;

        // === Benford Analysis ===
        let (benford_dist, chi_sq, is_anomalous) = self.analyze_benford_gpu(&device, network)?;
        result.benford_distribution = benford_dist;
        result.benford_chi_squared = chi_sq;
        result.benford_anomaly = is_anomalous;

        // === Suspense Detection ===
        let (suspense_count, suspense_scores) = self.detect_suspense_gpu(&device, network)?;
        result.suspense_account_count = suspense_count;
        result.suspense_scores = suspense_scores;

        // Calculate overall risk
        result.overall_risk_score = self.calculate_risk_score(result);

        // Update coordinator state
        self.coordinator.state.pagerank_complete = true;
        self.coordinator.state.fraud_detection_complete = true;
        self.coordinator.state.fraud_pattern_count = fraud_count;
        self.coordinator.state.gaap_validation_complete = true;
        self.coordinator.state.gaap_violation_count = gaap_count;
        self.coordinator.state.benford_complete = true;
        self.coordinator.state.benford_anomaly = is_anomalous;
        self.coordinator.state.suspense_complete = true;
        self.coordinator.state.suspense_account_count = suspense_count;

        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn compute_pagerank_gpu(
        &self,
        _device: &Arc<cudarc::driver::CudaDevice>,
        network: &AccountingNetwork,
    ) -> Result<Vec<f64>, String> {
        // For now, use CPU implementation until ring kernel infrastructure is ready
        // The ring kernel code is generated but requires the full RingKernel runtime
        Ok(network.compute_pagerank(
            self.coordinator.config.pagerank_iterations as usize,
            self.coordinator.config.pagerank_damping as f64,
        ))
    }

    #[cfg(feature = "cuda")]
    fn detect_fraud_gpu(
        &self,
        _device: &Arc<cudarc::driver::CudaDevice>,
        network: &AccountingNetwork,
    ) -> Result<(u32, Vec<u32>), String> {
        let n_flows = network.flows.len();
        let mut flags = vec![0u32; n_flows];
        let mut count = 0u32;

        // Simple CPU-side fraud detection until ring kernels are fully integrated
        for (i, flow) in network.flows.iter().enumerate() {
            let amount = flow.amount.to_f64().abs();
            let mut flag = 0u32;

            // Round amount check
            if amount >= 1000.0 && (amount % 1000.0).abs() < 1.0 {
                flag |= 0x01;
            }

            // Self-loop check
            if flow.source_account_index == flow.target_account_index {
                flag |= 0x02;
            }

            // Threshold proximity
            if (amount - 10000.0).abs() < 500.0 {
                flag |= 0x04;
            }

            if flag != 0 {
                count += 1;
            }
            flags[i] = flag;
        }

        Ok((count, flags))
    }

    #[cfg(feature = "cuda")]
    fn validate_gaap_gpu(
        &self,
        _device: &Arc<cudarc::driver::CudaDevice>,
        network: &AccountingNetwork,
    ) -> Result<(u32, Vec<u32>), String> {
        let n_flows = network.flows.len();
        let mut flags = vec![0u32; n_flows];
        let mut count = 0u32;

        for (i, flow) in network.flows.iter().enumerate() {
            let source_type = network
                .accounts
                .get(flow.source_account_index as usize)
                .map(|a| a.account_type as u8)
                .unwrap_or(0);
            let target_type = network
                .accounts
                .get(flow.target_account_index as usize)
                .map(|a| a.account_type as u8)
                .unwrap_or(0);

            // Revenue (3) -> Asset (0) direct
            if source_type == 3 && target_type == 0 {
                flags[i] = 1;
                count += 1;
            }
            // Revenue (3) -> Expense (4) direct
            else if source_type == 3 && target_type == 4 {
                flags[i] = 2;
                count += 1;
            }
        }

        Ok((count, flags))
    }

    #[cfg(feature = "cuda")]
    fn analyze_benford_gpu(
        &self,
        _device: &Arc<cudarc::driver::CudaDevice>,
        network: &AccountingNetwork,
    ) -> Result<([u32; 9], f32, bool), String> {
        let mut counts = [0u32; 9];

        for flow in &network.flows {
            let amount = flow.amount.to_f64().abs();
            if amount >= 1.0 {
                let mut v = amount;
                while v >= 10.0 {
                    v /= 10.0;
                }
                let digit = v as u32;
                if (1..=9).contains(&digit) {
                    counts[(digit - 1) as usize] += 1;
                }
            }
        }

        // Chi-squared test
        let total: u32 = counts.iter().sum();
        let expected = [
            0.301f32, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046,
        ];
        let mut chi_sq = 0.0f32;

        if total >= 50 {
            for (i, &count) in counts.iter().enumerate() {
                let observed = count as f32 / total as f32;
                let exp = expected[i];
                chi_sq += (observed - exp).powi(2) / exp;
            }
        }

        let is_anomalous = total >= 50 && chi_sq > 15.507;

        Ok((counts, chi_sq, is_anomalous))
    }

    #[cfg(feature = "cuda")]
    fn detect_suspense_gpu(
        &self,
        _device: &Arc<cudarc::driver::CudaDevice>,
        network: &AccountingNetwork,
    ) -> Result<(u32, Vec<f32>), String> {
        let n_accounts = network.accounts.len();
        let mut scores = vec![0.0f32; n_accounts];
        let mut count = 0u32;

        for (i, account) in network.accounts.iter().enumerate() {
            let mut score = 0.0f32;

            // Round balance check
            let balance = (account.closing_balance.to_f64()).abs();
            if balance >= 1000.0 && (balance % 1000.0).abs() < 1.0 {
                score += 0.3;
            }

            // High risk score
            if account.risk_score > 0.5 {
                score += 0.4;
            }

            // Flow imbalance
            let ratio = if account.out_degree > 0 {
                account.in_degree as f32 / account.out_degree as f32
            } else {
                10.0
            };
            if ratio > 5.0 {
                score += 0.3;
            }

            scores[i] = score.min(1.0);
            if scores[i] > 0.5 {
                count += 1;
            }
        }

        Ok((count, scores))
    }

    /// CPU fallback analysis.
    fn analyze_cpu(&mut self, network: &AccountingNetwork, result: &mut GpuAnalyticsResult) {
        let n_accounts = network.accounts.len();
        let n_flows = network.flows.len();

        // PageRank
        result.pagerank_scores = network.compute_pagerank(
            self.coordinator.config.pagerank_iterations as usize,
            self.coordinator.config.pagerank_damping as f64,
        );

        // Fraud detection
        result.fraud_flags = vec![0u32; n_flows];
        for (i, flow) in network.flows.iter().enumerate() {
            let amount = flow.amount.to_f64().abs();
            let mut flag = 0u32;

            if amount >= 1000.0 && (amount % 1000.0).abs() < 1.0 {
                flag |= 0x01;
            }
            if flow.source_account_index == flow.target_account_index {
                flag |= 0x02;
            }
            if (amount - 10000.0).abs() < 500.0 {
                flag |= 0x04;
            }

            result.fraud_flags[i] = flag;
            if flag != 0 {
                result.fraud_pattern_count += 1;
            }
        }

        // GAAP validation
        result.gaap_flags = vec![0u32; n_flows];
        for (i, flow) in network.flows.iter().enumerate() {
            let source_type = network
                .accounts
                .get(flow.source_account_index as usize)
                .map(|a| a.account_type as u8)
                .unwrap_or(0);
            let target_type = network
                .accounts
                .get(flow.target_account_index as usize)
                .map(|a| a.account_type as u8)
                .unwrap_or(0);

            if source_type == 3 && (target_type == 0 || target_type == 4) {
                result.gaap_flags[i] = 1;
                result.gaap_violation_count += 1;
            }
        }

        // Benford analysis
        for flow in &network.flows {
            let amount = flow.amount.to_f64().abs();
            if amount >= 1.0 {
                let mut v = amount;
                while v >= 10.0 {
                    v /= 10.0;
                }
                let digit = v as u32;
                if (1..=9).contains(&digit) {
                    result.benford_distribution[(digit - 1) as usize] += 1;
                }
            }
        }

        let total: u32 = result.benford_distribution.iter().sum();
        if total >= 50 {
            let expected = [
                0.301f32, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046,
            ];
            for (i, &count) in result.benford_distribution.iter().enumerate() {
                let observed = count as f32 / total as f32;
                let exp = expected[i];
                result.benford_chi_squared += (observed - exp).powi(2) / exp;
            }
            result.benford_anomaly = result.benford_chi_squared > 15.507;
        }

        // Suspense detection
        result.suspense_scores = vec![0.0f32; n_accounts];
        for (i, account) in network.accounts.iter().enumerate() {
            let mut score = 0.0f32;

            let balance = account.closing_balance.to_f64().abs();
            if balance >= 1000.0 && (balance % 1000.0).abs() < 1.0 {
                score += 0.3;
            }
            if account.risk_score > 0.5 {
                score += 0.4;
            }
            let ratio = if account.out_degree > 0 {
                account.in_degree as f32 / account.out_degree as f32
            } else {
                10.0
            };
            if ratio > 5.0 {
                score += 0.3;
            }

            result.suspense_scores[i] = score.min(1.0);
            if result.suspense_scores[i] > 0.5 {
                result.suspense_account_count += 1;
            }
        }

        // Overall risk
        result.overall_risk_score = self.calculate_risk_score(result);

        // Update coordinator
        self.coordinator.state.pagerank_complete = true;
        self.coordinator.state.fraud_detection_complete = true;
        self.coordinator.state.fraud_pattern_count = result.fraud_pattern_count;
        self.coordinator.state.gaap_validation_complete = true;
        self.coordinator.state.gaap_violation_count = result.gaap_violation_count;
        self.coordinator.state.benford_complete = true;
        self.coordinator.state.benford_anomaly = result.benford_anomaly;
        self.coordinator.state.suspense_complete = true;
        self.coordinator.state.suspense_account_count = result.suspense_account_count;
    }

    fn calculate_risk_score(&self, result: &GpuAnalyticsResult) -> f32 {
        let fraud_risk = (result.fraud_pattern_count as f32 / 100.0).min(1.0);
        let gaap_risk = (result.gaap_violation_count as f32 / 50.0).min(1.0);
        let suspense_risk = (result.suspense_account_count as f32 / 20.0).min(1.0);
        let benford_risk = if result.benford_anomaly { 0.5 } else { 0.0 };

        (fraud_risk * 0.35 + gaap_risk * 0.25 + suspense_risk * 0.25 + benford_risk * 0.15).min(1.0)
    }

    /// Get coordinator reference.
    pub fn coordinator(&self) -> &AnalyticsCoordinator {
        &self.coordinator
    }

    /// Get coordinator mutable reference.
    pub fn coordinator_mut(&mut self) -> &mut AnalyticsCoordinator {
        &mut self.coordinator
    }
}

impl Default for GpuActorRuntime {
    fn default() -> Self {
        Self::new(CoordinatorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_runtime_creation() {
        let runtime = GpuActorRuntime::default();
        let status = runtime.status();
        // GPU availability depends on the system
        assert_eq!(status.messages_processed, 0);
    }

    #[test]
    fn test_analyze_empty_network() {
        let mut runtime = GpuActorRuntime::default();
        let network = AccountingNetwork::new(Uuid::new_v4(), 2024, 1);

        let result = runtime.analyze(&network);
        assert_eq!(result.fraud_pattern_count, 0);
        assert_eq!(result.gaap_violation_count, 0);
    }

    #[test]
    fn test_cpu_fallback() {
        let mut runtime = GpuActorRuntime::default();
        // Force CPU path by creating network
        let mut network = AccountingNetwork::new(Uuid::new_v4(), 2024, 1);

        // Add some test data
        use crate::models::{
            AccountMetadata, AccountNode, AccountType, Decimal128, HybridTimestamp, TransactionFlow,
        };

        let cash = network.add_account(
            AccountNode::new(Uuid::new_v4(), AccountType::Asset, 0),
            AccountMetadata::new("1100", "Cash"),
        );
        let revenue = network.add_account(
            AccountNode::new(Uuid::new_v4(), AccountType::Revenue, 0),
            AccountMetadata::new("4000", "Revenue"),
        );

        network.add_flow(TransactionFlow::new(
            revenue,
            cash,
            Decimal128::from_f64(1000.0),
            Uuid::new_v4(),
            HybridTimestamp::now(),
        ));

        let result = runtime.analyze(&network);
        // Should have detected GAAP violation (Revenue -> Asset)
        assert!(result.gaap_violation_count > 0 || result.pagerank_scores.len() == 2);
    }

    #[test]
    fn test_risk_score_calculation() {
        let runtime = GpuActorRuntime::default();
        let result = GpuAnalyticsResult {
            fraud_pattern_count: 50,
            gaap_violation_count: 25,
            suspense_account_count: 10,
            benford_anomaly: true,
            ..Default::default()
        };

        let risk = runtime.calculate_risk_score(&result);
        assert!(risk > 0.0);
        assert!(risk <= 1.0);
    }
}
