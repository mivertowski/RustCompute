//! 64-bit integer workarounds for WGSL.
//!
//! WGSL 1.0 does not support 64-bit integers or atomics. This module provides
//! helper functions that emulate 64-bit operations using lo/hi u32 pairs.

/// Helper functions for 64-bit emulation in WGSL.
pub struct U64Helpers;

impl U64Helpers {
    /// Generate the complete set of 64-bit helper functions.
    pub fn generate_all() -> String {
        [
            Self::generate_read_u64(),
            Self::generate_write_u64(),
            Self::generate_atomic_inc_u64(),
            Self::generate_atomic_add_u64(),
            Self::generate_compare_u64(),
            Self::generate_add_u64(),
            Self::generate_sub_u64(),
        ]
        .join("\n\n")
    }

    /// Generate read_u64 function.
    pub fn generate_read_u64() -> &'static str {
        r#"// Read a 64-bit value from lo/hi atomic pair
fn read_u64(lo: ptr<storage, atomic<u32>, read_write>, hi: ptr<storage, atomic<u32>, read_write>) -> vec2<u32> {
    return vec2<u32>(atomicLoad(lo), atomicLoad(hi));
}"#
    }

    /// Generate write_u64 function.
    pub fn generate_write_u64() -> &'static str {
        r#"// Write a 64-bit value to lo/hi atomic pair
fn write_u64(lo: ptr<storage, atomic<u32>, read_write>, hi: ptr<storage, atomic<u32>, read_write>, value: vec2<u32>) {
    atomicStore(lo, value.x);
    atomicStore(hi, value.y);
}"#
    }

    /// Generate atomic_inc_u64 function.
    pub fn generate_atomic_inc_u64() -> &'static str {
        r#"// Atomically increment a 64-bit value
fn atomic_inc_u64(lo: ptr<storage, atomic<u32>, read_write>, hi: ptr<storage, atomic<u32>, read_write>) {
    let old_lo = atomicAdd(lo, 1u);
    if (old_lo == 0xFFFFFFFFu) {
        atomicAdd(hi, 1u);
    }
}"#
    }

    /// Generate atomic_add_u64 function.
    pub fn generate_atomic_add_u64() -> &'static str {
        r#"// Atomically add to a 64-bit value
fn atomic_add_u64(lo: ptr<storage, atomic<u32>, read_write>, hi: ptr<storage, atomic<u32>, read_write>, addend: u32) {
    let old_lo = atomicAdd(lo, addend);
    if (old_lo > 0xFFFFFFFFu - addend) {
        atomicAdd(hi, 1u);
    }
}"#
    }

    /// Generate compare_u64 function.
    pub fn generate_compare_u64() -> &'static str {
        r#"// Compare two 64-bit values: returns -1 if a < b, 0 if a == b, 1 if a > b
fn compare_u64(a: vec2<u32>, b: vec2<u32>) -> i32 {
    if (a.y > b.y) { return 1; }
    if (a.y < b.y) { return -1; }
    if (a.x > b.x) { return 1; }
    if (a.x < b.x) { return -1; }
    return 0;
}"#
    }

    /// Generate add_u64 function (non-atomic).
    pub fn generate_add_u64() -> &'static str {
        r#"// Add two 64-bit values (non-atomic)
fn add_u64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    let hi = a.y + b.y + carry;
    return vec2<u32>(lo, hi);
}"#
    }

    /// Generate sub_u64 function (non-atomic).
    pub fn generate_sub_u64() -> &'static str {
        r#"// Subtract two 64-bit values (non-atomic): a - b
fn sub_u64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let borrow = select(0u, 1u, a.x < b.x);
    let lo = a.x - b.x;
    let hi = a.y - b.y - borrow;
    return vec2<u32>(lo, hi);
}"#
    }

    /// Generate mul_u64_u32 function (multiply 64-bit by 32-bit).
    pub fn generate_mul_u64_u32() -> &'static str {
        r#"// Multiply 64-bit value by 32-bit value
fn mul_u64_u32(a: vec2<u32>, b: u32) -> vec2<u32> {
    // Split into 16-bit parts to avoid overflow
    let a_lo_lo = a.x & 0xFFFFu;
    let a_lo_hi = a.x >> 16u;
    let a_hi_lo = a.y & 0xFFFFu;

    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    // Partial products
    let p0 = a_lo_lo * b_lo;
    let p1 = a_lo_lo * b_hi + a_lo_hi * b_lo;
    let p2 = a_lo_hi * b_hi + a_hi_lo * b_lo;

    // Combine
    let lo = p0 + (p1 << 16u);
    let carry1 = select(0u, 1u, lo < p0);
    let hi = (p1 >> 16u) + p2 + carry1;

    return vec2<u32>(lo, hi);
}"#
    }
}

/// Convert a Rust u64 value to WGSL vec2<u32> literal.
pub fn u64_to_vec2_literal(value: u64) -> String {
    let lo = (value & 0xFFFFFFFF) as u32;
    let hi = (value >> 32) as u32;
    format!("vec2<u32>({}u, {}u)", lo, hi)
}

/// Convert a Rust i64 value to WGSL vec2<i32> literal.
pub fn i64_to_vec2_literal(value: i64) -> String {
    let lo = (value as u64 & 0xFFFFFFFF) as i32;
    let hi = ((value as u64) >> 32) as i32;
    format!("vec2<i32>({}, {})", lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u64_to_vec2_literal() {
        assert_eq!(u64_to_vec2_literal(0), "vec2<u32>(0u, 0u)");
        assert_eq!(u64_to_vec2_literal(1), "vec2<u32>(1u, 0u)");
        assert_eq!(
            u64_to_vec2_literal(0x1_0000_0000),
            "vec2<u32>(0u, 1u)"
        );
        assert_eq!(
            u64_to_vec2_literal(0xFFFF_FFFF_FFFF_FFFF),
            "vec2<u32>(4294967295u, 4294967295u)"
        );
    }

    #[test]
    fn test_generate_all_helpers() {
        let helpers = U64Helpers::generate_all();
        assert!(helpers.contains("fn read_u64"));
        assert!(helpers.contains("fn write_u64"));
        assert!(helpers.contains("fn atomic_inc_u64"));
        assert!(helpers.contains("fn atomic_add_u64"));
        assert!(helpers.contains("fn compare_u64"));
        assert!(helpers.contains("fn add_u64"));
        assert!(helpers.contains("fn sub_u64"));
    }
}
