//! Parallel batch kernel backend for high-throughput transaction monitoring.
//!
//! This backend processes transactions in large batches where each CUDA thread
//! handles one transaction independently. Best for maximum throughput.
//!
//! ## Architecture
//!
//! ```text
//! Host                          GPU
//! ┌──────────────┐             ┌──────────────────────────────────┐
//! │ Transaction  │  ──copy──>  │ Global Memory                    │
//! │ Batch        │             │ ┌────────────────────────────┐   │
//! │              │             │ │ transactions[N]            │   │
//! │ Customer     │  ──copy──>  │ │ profiles[N]                │   │
//! │ Profiles     │             │ │ alerts[N*4] (max 4/tx)     │   │
//! └──────────────┘             │ └────────────────────────────┘   │
//!                              │                                  │
//!                              │ ┌────────────────────────────┐   │
//!                              │ │ Thread Block (256 threads) │   │
//!                              │ │ ┌──────┬──────┬──────┬───┐ │   │
//!                              │ │ │ T0   │ T1   │ T2   │...│ │   │
//!                              │ │ │tx[0] │tx[1] │tx[2] │   │ │   │
//!                              │ │ └──────┴──────┴──────┴───┘ │   │
//!                              │ └────────────────────────────┘   │
//!                              └──────────────────────────────────┘
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Throughput**: ~10M+ transactions/second on modern GPUs
//! - **Latency**: Batch latency (transfer + compute + transfer back)
//! - **Memory**: O(N) for N transactions, plus 4*N alert slots
//! - **Occupancy**: High (simple kernel, few registers)

use super::types::*;

/// Configuration for the batch kernel backend.
#[derive(Debug, Clone)]
pub struct BatchKernelConfig {
    /// Number of threads per block.
    pub block_size: u32,
    /// Maximum alerts per transaction (default: 4).
    pub max_alerts_per_tx: u32,
    /// Whether to use pinned (page-locked) memory for transfers.
    pub use_pinned_memory: bool,
}

impl Default for BatchKernelConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            max_alerts_per_tx: 4,
            use_pinned_memory: true,
        }
    }
}

/// Statistics for the batch kernel backend.
#[derive(Debug, Clone, Default)]
pub struct BatchKernelStats {
    pub buffer_capacity: usize,
    pub block_size: u32,
    pub max_alerts_per_tx: u32,
}

/// CPU fallback implementation (for testing without GPU).
pub struct BatchKernelCpuFallback {
    config: GpuMonitoringConfig,
}

impl BatchKernelCpuFallback {
    pub fn new(config: GpuMonitoringConfig) -> Self {
        Self { config }
    }

    /// Process batch on CPU (matches GPU logic).
    pub fn process_batch(
        &self,
        transactions: &[GpuTransaction],
        profiles: &[GpuCustomerProfile],
    ) -> Vec<GpuAlert> {
        let mut alerts = Vec::new();

        for (tx, profile) in transactions.iter().zip(profiles.iter()) {
            let amount_threshold = if profile.amount_threshold > 0 {
                profile.amount_threshold
            } else {
                self.config.amount_threshold
            };

            let velocity_threshold = if profile.velocity_threshold > 0 {
                profile.velocity_threshold
            } else {
                self.config.velocity_threshold
            };

            // Velocity breach
            if profile.velocity_count > velocity_threshold {
                let severity = if profile.velocity_count >= velocity_threshold * 3 {
                    alert_severity::CRITICAL
                } else if profile.velocity_count >= velocity_threshold * 2 {
                    alert_severity::HIGH
                } else {
                    alert_severity::MEDIUM
                };

                alerts.push(GpuAlert {
                    alert_type: alert_types::VELOCITY_BREACH,
                    severity,
                    transaction_id: tx.transaction_id,
                    customer_id: tx.customer_id,
                    amount_cents: tx.amount_cents,
                    velocity_count: profile.velocity_count,
                    timestamp: tx.timestamp,
                    ..Default::default()
                });
            }

            // Amount threshold
            if tx.amount_cents > amount_threshold {
                let ratio = tx.amount_cents / amount_threshold;
                let severity = if ratio >= 5 {
                    alert_severity::CRITICAL
                } else if ratio >= 2 {
                    alert_severity::HIGH
                } else {
                    alert_severity::MEDIUM
                };

                alerts.push(GpuAlert {
                    alert_type: alert_types::AMOUNT_THRESHOLD,
                    severity,
                    transaction_id: tx.transaction_id,
                    customer_id: tx.customer_id,
                    amount_cents: tx.amount_cents,
                    timestamp: tx.timestamp,
                    ..Default::default()
                });
            }

            // Structured transaction
            let structuring_threshold =
                (amount_threshold * self.config.structuring_threshold_pct as u64) / 100;

            if tx.amount_cents > structuring_threshold
                && tx.amount_cents <= amount_threshold
                && profile.velocity_count >= self.config.structuring_min_velocity as u32
            {
                alerts.push(GpuAlert {
                    alert_type: alert_types::STRUCTURED_TRANSACTION,
                    severity: alert_severity::HIGH,
                    transaction_id: tx.transaction_id,
                    customer_id: tx.customer_id,
                    amount_cents: tx.amount_cents,
                    velocity_count: profile.velocity_count,
                    timestamp: tx.timestamp,
                    ..Default::default()
                });
            }

            // Geographic anomaly
            if tx.country_code != profile.country_code && tx.country_code < 64 {
                let country_bit = 1u64 << (tx.country_code as u64);
                let is_allowed = (profile.allowed_destinations & country_bit) != 0;

                if !is_allowed {
                    let is_high_risk = tx.country_code == 6 || tx.country_code == 7;
                    let severity = if is_high_risk {
                        alert_severity::HIGH
                    } else {
                        alert_severity::MEDIUM
                    };

                    alerts.push(GpuAlert {
                        alert_type: alert_types::GEOGRAPHIC_ANOMALY,
                        severity,
                        transaction_id: tx.transaction_id,
                        customer_id: tx.customer_id,
                        amount_cents: tx.amount_cents,
                        country_code: tx.country_code,
                        timestamp: tx.timestamp,
                        ..Default::default()
                    });
                }
            }
        }

        alerts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback_velocity_breach() {
        let backend = BatchKernelCpuFallback::new(GpuMonitoringConfig::default());

        let tx = GpuTransaction {
            transaction_id: 1,
            customer_id: 100,
            amount_cents: 500_00,
            country_code: 1,
            ..Default::default()
        };

        let profile = GpuCustomerProfile {
            customer_id: 100,
            country_code: 1,
            velocity_count: 15, // Over threshold of 10
            allowed_destinations: !0,
            ..Default::default()
        };

        let alerts = backend.process_batch(&[tx], &[profile]);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].alert_type, alert_types::VELOCITY_BREACH);
    }

    #[test]
    fn test_cpu_fallback_amount_threshold() {
        let backend = BatchKernelCpuFallback::new(GpuMonitoringConfig::default());

        let tx = GpuTransaction {
            transaction_id: 1,
            customer_id: 100,
            amount_cents: 15_000_00, // Over $10,000 threshold
            country_code: 1,
            ..Default::default()
        };

        let profile = GpuCustomerProfile {
            customer_id: 100,
            country_code: 1,
            allowed_destinations: !0,
            ..Default::default()
        };

        let alerts = backend.process_batch(&[tx], &[profile]);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].alert_type, alert_types::AMOUNT_THRESHOLD);
    }

    #[test]
    fn test_cpu_fallback_no_alerts() {
        let backend = BatchKernelCpuFallback::new(GpuMonitoringConfig::default());

        let tx = GpuTransaction {
            transaction_id: 1,
            customer_id: 100,
            amount_cents: 100_00, // Small amount
            country_code: 1,
            ..Default::default()
        };

        let profile = GpuCustomerProfile {
            customer_id: 100,
            country_code: 1,
            velocity_count: 2, // Under threshold
            allowed_destinations: !0,
            ..Default::default()
        };

        let alerts = backend.process_batch(&[tx], &[profile]);
        assert!(alerts.is_empty());
    }
}
