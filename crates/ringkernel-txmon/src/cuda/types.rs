//! CUDA-compatible type definitions for GPU kernels.
//!
//! These types are designed to be identical in memory layout between
//! Rust and CUDA C, enabling zero-copy data transfer.

use bytemuck::Zeroable;

/// CUDA-compatible transaction structure (128 bytes, cache-line aligned).
///
/// This mirrors the Rust `Transaction` struct for GPU processing.
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy)]
pub struct GpuTransaction {
    pub transaction_id: u64,
    pub customer_id: u64,
    pub amount_cents: u64,
    pub timestamp: u64,
    pub country_code: u16,
    pub tx_type: u8,
    pub flags: u8,
    pub destination_id: u64,
    pub _reserved: [u8; 80],
}

unsafe impl Zeroable for GpuTransaction {}

impl Default for GpuTransaction {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// CUDA-compatible customer risk profile (128 bytes).
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy)]
pub struct GpuCustomerProfile {
    pub customer_id: u64,
    pub risk_level: u8,
    pub risk_score: u8,
    pub country_code: u16,
    pub is_pep: u8,
    pub requires_edd: u8,
    pub has_adverse_media: u8,
    pub geographic_risk: u8,
    pub business_risk: u8,
    pub behavioral_risk: u8,
    pub _padding1: [u8; 2],
    pub transaction_count: u32,
    pub alert_count: u32,
    pub velocity_count: u32,
    pub amount_threshold: u64,
    pub velocity_threshold: u32,
    pub _padding2: u32,
    pub allowed_destinations: u64,
    pub avg_monthly_volume: u64,
    pub last_transaction_ts: u64,
    pub created_ts: u64,
    pub _reserved: [u8; 48],
}

unsafe impl Zeroable for GpuCustomerProfile {}

impl Default for GpuCustomerProfile {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// CUDA-compatible monitoring alert (128 bytes).
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy)]
pub struct GpuAlert {
    pub alert_id: u64,
    pub transaction_id: u64,
    pub customer_id: u64,
    pub alert_type: u8,
    pub severity: u8,
    pub status: u8,
    pub _padding1: [u8; 5],
    pub amount_cents: u64,
    pub risk_score: u32,
    pub velocity_count: u32,
    pub timestamp: u64,
    pub country_code: u16,
    pub flags: u16,
    pub _padding2: [u8; 4],
    pub _reserved: [u8; 64],
}

unsafe impl Zeroable for GpuAlert {}

impl Default for GpuAlert {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// Monitoring configuration for GPU kernels.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuMonitoringConfig {
    pub amount_threshold: u64,
    pub velocity_threshold: u32,
    pub structuring_threshold_pct: u8,
    pub structuring_min_velocity: u8,
    pub _padding: [u8; 2],
}

impl Default for GpuMonitoringConfig {
    fn default() -> Self {
        Self {
            amount_threshold: 10_000_00,
            velocity_threshold: 10,
            structuring_threshold_pct: 90,
            structuring_min_velocity: 3,
            _padding: [0; 2],
        }
    }
}

/// Alert type codes matching the Rust enum.
pub mod alert_types {
    pub const VELOCITY_BREACH: u8 = 0;
    pub const AMOUNT_THRESHOLD: u8 = 1;
    pub const GEOGRAPHIC_ANOMALY: u8 = 2;
    pub const STRUCTURED_TRANSACTION: u8 = 3;
    pub const CIRCULAR_TRADING: u8 = 4;
    pub const SANCTIONS_HIT: u8 = 5;
    pub const PEP_RELATED: u8 = 6;
    pub const ADVERSE_MEDIA: u8 = 7;
    pub const UNUSUAL_PATTERN: u8 = 8;
}

/// Alert severity codes matching the Rust enum.
pub mod alert_severity {
    pub const LOW: u8 = 0;
    pub const MEDIUM: u8 = 1;
    pub const HIGH: u8 = 2;
    pub const CRITICAL: u8 = 3;
}

/// Transaction adjacency entry for network analysis (stencil kernel).
/// Used to represent transaction graph connections.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct TxAdjacency {
    pub from_customer: u64,
    pub to_customer: u64,
    pub amount_cents: u64,
    pub timestamp: u64,
    pub tx_count: u32,
    pub flags: u32,
}

/// Network cell for stencil-based pattern detection.
/// Represents aggregated transaction data in a 2D customer-time grid.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct NetworkCell {
    pub total_amount: u64,
    pub tx_count: u32,
    pub unique_counterparties: u32,
    pub inbound_count: u32,
    pub outbound_count: u32,
    pub flags: u32,
    pub _padding: u32,
}

// Compile-time size assertions
const _: () = {
    assert!(std::mem::size_of::<GpuTransaction>() == 128);
    assert!(std::mem::size_of::<GpuCustomerProfile>() == 128);
    assert!(std::mem::size_of::<GpuAlert>() == 128);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_type_sizes() {
        assert_eq!(std::mem::size_of::<GpuTransaction>(), 128);
        assert_eq!(std::mem::size_of::<GpuCustomerProfile>(), 128);
        assert_eq!(std::mem::size_of::<GpuAlert>(), 128);
    }

    #[test]
    fn test_gpu_type_alignment() {
        assert_eq!(std::mem::align_of::<GpuTransaction>(), 128);
        assert_eq!(std::mem::align_of::<GpuCustomerProfile>(), 128);
        assert_eq!(std::mem::align_of::<GpuAlert>(), 128);
    }
}
