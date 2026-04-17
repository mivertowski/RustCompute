//! Audit tagging for K2K messages (multi-tenant cost/audit accounting).
//!
//! This module defines [`AuditTag`] — a two-level hierarchy used to attribute
//! GPU work to a specific organization (audit firm) and engagement (audit
//! project). Every K2K message envelope carries an `AuditTag` alongside its
//! `TenantId`, enabling per-engagement cost tracking and audit trails without
//! affecting the primary security boundary (which is [`TenantId`]).
//!
//! # Two-tier model
//!
//! - `TenantId` = security boundary (routing, isolation, quotas)
//! - `AuditTag { org_id, engagement_id }` = observability / billing
//!
//! A single tenant may run many concurrent engagements (e.g. one audit firm
//! conducting audits of several unrelated clients). Cross-engagement sends
//! inside the same tenant are allowed (they are not a security boundary), but
//! the envelope's `AuditTag` is preserved so cost is accounted correctly.
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_core::k2k::audit_tag::AuditTag;
//!
//! // Audit firm 42 running engagement 7 for client Acme Corp
//! let tag = AuditTag::new(42, 7);
//! println!("{}", tag); // "org=42 engagement=7"
//! ```
//!
//! [`TenantId`]: crate::k2k::tenant::TenantId

use bytemuck::{Pod, Zeroable};
use rkyv::{Archive, Deserialize, Serialize};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

/// Two-level hierarchy for audit firm workload accounting.
///
/// Every K2K [`MessageEnvelope`] carries an `AuditTag` so that GPU work can be
/// attributed back to a specific engagement — critical for per-engagement
/// billing and tamper-evident audit trails.
///
/// `AuditTag::default()` returns `{ org_id: 0, engagement_id: 0 }`, denoting
/// an unspecified / system-level engagement. This is the backward-compatible
/// default for single-tenant deployments.
///
/// # Layout
///
/// Pod, Zeroable, AsBytes, FromBytes — safe for direct blit into GPU-shared
/// memory. Rkyv-compatible for zero-copy serialization in message queues.
///
/// [`MessageEnvelope`]: crate::message::MessageEnvelope
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
    AsBytes,
    FromBytes,
    FromZeroes,
    Pod,
    Zeroable,
    Archive,
    Serialize,
    Deserialize,
)]
#[repr(C)]
pub struct AuditTag {
    /// Audit firm / organization ID.
    ///
    /// Stable across all engagements belonging to the same organization.
    /// `0` = unspecified (default).
    pub org_id: u64,
    /// Specific engagement / audit project ID.
    ///
    /// Distinct per billable unit of work. `0` = unspecified (default).
    pub engagement_id: u64,
}

impl AuditTag {
    /// Create a new `AuditTag`.
    #[inline]
    pub const fn new(org_id: u64, engagement_id: u64) -> Self {
        Self {
            org_id,
            engagement_id,
        }
    }

    /// The unspecified / default audit tag (`{0, 0}`).
    ///
    /// Used in single-tenant deployments and as the default for legacy APIs
    /// that don't propagate audit information.
    #[inline]
    pub const fn unspecified() -> Self {
        Self {
            org_id: 0,
            engagement_id: 0,
        }
    }

    /// Returns `true` if this tag is the default / unspecified tag (`{0, 0}`).
    #[inline]
    pub const fn is_unspecified(&self) -> bool {
        self.org_id == 0 && self.engagement_id == 0
    }

    /// Returns the raw bytes of this tag (16 bytes, little-endian).
    ///
    /// Used when stamping audit tags into wire-format K2K message envelopes.
    #[inline]
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.org_id.to_le_bytes());
        out[8..].copy_from_slice(&self.engagement_id.to_le_bytes());
        out
    }

    /// Reconstruct an `AuditTag` from its raw byte representation.
    #[inline]
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        let org_id = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let engagement_id = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        Self {
            org_id,
            engagement_id,
        }
    }
}

impl std::fmt::Display for AuditTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "org={} engagement={}", self.org_id, self.engagement_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_tag_default() {
        let tag = AuditTag::default();
        assert_eq!(tag.org_id, 0);
        assert_eq!(tag.engagement_id, 0);
        assert!(tag.is_unspecified());
    }

    #[test]
    fn test_audit_tag_new() {
        let tag = AuditTag::new(42, 7);
        assert_eq!(tag.org_id, 42);
        assert_eq!(tag.engagement_id, 7);
        assert!(!tag.is_unspecified());
    }

    #[test]
    fn test_audit_tag_display() {
        let tag = AuditTag::new(42, 7);
        assert_eq!(format!("{}", tag), "org=42 engagement=7");
    }

    #[test]
    fn test_audit_tag_roundtrip_bytes() {
        let tag = AuditTag::new(0xDEADBEEF, 0xCAFEF00D);
        let bytes = tag.to_bytes();
        let roundtrip = AuditTag::from_bytes(bytes);
        assert_eq!(tag, roundtrip);
    }

    #[test]
    fn test_audit_tag_hash_eq() {
        use std::collections::HashMap;
        let mut map: HashMap<AuditTag, u32> = HashMap::new();
        map.insert(AuditTag::new(1, 1), 10);
        map.insert(AuditTag::new(1, 2), 20);
        assert_eq!(map.get(&AuditTag::new(1, 1)), Some(&10));
        assert_eq!(map.get(&AuditTag::new(1, 2)), Some(&20));
        assert_eq!(map.get(&AuditTag::new(1, 3)), None);
    }

    #[test]
    fn test_audit_tag_size() {
        // 2 x u64 = 16 bytes, no padding
        assert_eq!(std::mem::size_of::<AuditTag>(), 16);
    }
}
