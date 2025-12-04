//! Core data models for accounting network analytics.
//!
//! These models represent the fundamental structures of double-entry bookkeeping
//! transformed into a graph representation for GPU-accelerated analysis.

mod account;
mod flow;
mod journal;
mod network;
mod patterns;
mod temporal;

pub use account::*;
pub use flow::*;
pub use journal::*;
pub use network::*;
pub use patterns::*;
pub use temporal::*;

use rkyv::{Archive, Deserialize, Serialize};

/// Decimal128 representation for precise monetary amounts.
/// Uses fixed-point arithmetic: value = mantissa * 10^(-scale)
#[derive(Debug, Clone, Copy, PartialEq, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[repr(C)]
pub struct Decimal128 {
    /// The mantissa (integer part when scale=0)
    pub mantissa: i128,
    /// Decimal places (typically 2 for currency)
    pub scale: u8,
}

impl Decimal128 {
    /// Zero value constant with scale 2.
    pub const ZERO: Self = Self {
        mantissa: 0,
        scale: 2,
    };

    /// Create a new Decimal128 with given mantissa and scale.
    pub fn new(mantissa: i128, scale: u8) -> Self {
        Self { mantissa, scale }
    }

    /// Create a zero value.
    pub fn zero() -> Self {
        Self::ZERO
    }

    /// Create from cents (scale 2).
    pub fn from_cents(cents: i64) -> Self {
        Self {
            mantissa: cents as i128,
            scale: 2,
        }
    }

    /// Create from f64 (scale 2).
    pub fn from_f64(value: f64) -> Self {
        Self {
            mantissa: (value * 100.0).round() as i128,
            scale: 2,
        }
    }

    /// Convert to f64.
    pub fn to_f64(&self) -> f64 {
        self.mantissa as f64 / 10f64.powi(self.scale as i32)
    }

    /// Get absolute value.
    pub fn abs(&self) -> Self {
        Self {
            mantissa: self.mantissa.abs(),
            scale: self.scale,
        }
    }

    /// Check if value is zero.
    pub fn is_zero(&self) -> bool {
        self.mantissa == 0
    }

    /// Check if value is positive.
    pub fn is_positive(&self) -> bool {
        self.mantissa > 0
    }

    /// Check if value is negative.
    pub fn is_negative(&self) -> bool {
        self.mantissa < 0
    }
}

impl std::ops::Add for Decimal128 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        // Assume same scale for simplicity
        Self {
            mantissa: self.mantissa + rhs.mantissa,
            scale: self.scale,
        }
    }
}

impl std::ops::Sub for Decimal128 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            mantissa: self.mantissa - rhs.mantissa,
            scale: self.scale,
        }
    }
}

impl std::ops::Neg for Decimal128 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            mantissa: -self.mantissa,
            scale: self.scale,
        }
    }
}

impl Default for Decimal128 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for Decimal128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.to_f64();
        write!(f, "${:.2}", value)
    }
}

/// Hybrid Logical Timestamp for causal ordering.
/// Combines physical time with logical counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Archive, Serialize, Deserialize)]
#[archive(compare(PartialEq))]
#[repr(C)]
pub struct HybridTimestamp {
    /// Physical time component (milliseconds since epoch)
    pub physical: u64,
    /// Logical counter for same-millisecond ordering
    pub logical: u32,
    /// Node ID for tie-breaking
    pub node_id: u32,
}

impl HybridTimestamp {
    /// Create a timestamp with current physical time.
    pub fn now() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let physical = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        Self {
            physical,
            logical: 0,
            node_id: 0,
        }
    }

    /// Create a timestamp with current time and specified node ID.
    pub fn with_node_id(node_id: u32) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let physical = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        Self {
            physical,
            logical: 0,
            node_id,
        }
    }

    /// Create a timestamp with specific physical and logical components.
    pub fn new(physical: u64, logical: u32) -> Self {
        Self {
            physical,
            logical,
            node_id: 0,
        }
    }

    /// Create a zero timestamp.
    pub fn zero() -> Self {
        Self {
            physical: 0,
            logical: 0,
            node_id: 0,
        }
    }
}

impl Default for HybridTimestamp {
    fn default() -> Self {
        Self::zero()
    }
}
