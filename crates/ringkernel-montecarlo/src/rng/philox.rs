//! Philox counter-based PRNG.
//!
//! Philox is a counter-based PRNG designed for parallel random number generation
//! on GPUs. It was introduced in the paper "Parallel Random Numbers: As Easy as 1, 2, 3"
//! by Salmon et al. (2011).
//!
//! Key properties:
//! - Counter-based: state is just a 128-bit counter and 128-bit key
//! - Statistically excellent: passes all BigCrush tests
//! - Fast on GPU: uses only integer operations, no branches
//! - Reproducible: same counter + key always gives same output

use super::traits::GpuRng;
use bytemuck::{Pod, Zeroable};

/// Philox4x32-10 state (32 bytes, GPU-friendly).
///
/// This implements the Philox4x32 variant with 10 rounds.
/// It generates 4 x 32-bit outputs per round.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(16))]
pub struct PhiloxState {
    /// 128-bit counter (4 x 32-bit)
    pub counter: [u32; 4],
    /// 128-bit key (4 x 32-bit, but we only use 2)
    pub key: [u32; 4],
}

// SAFETY: PhiloxState is #[repr(C)] with only primitive types
unsafe impl Zeroable for PhiloxState {}
unsafe impl Pod for PhiloxState {}

/// Philox PRNG instance.
///
/// Wraps PhiloxState with convenient methods.
pub struct PhiloxRng {
    state: PhiloxState,
    /// Buffer for generated values (we generate 4 at a time)
    buffer: [u32; 4],
    /// Index into buffer
    buffer_idx: usize,
}

// Philox round constants
const PHILOX_M4X32_0: u32 = 0xD2511F53;
const PHILOX_M4X32_1: u32 = 0xCD9E8D57;
const PHILOX_W32_0: u32 = 0x9E3779B9;
const PHILOX_W32_1: u32 = 0xBB67AE85;

impl PhiloxRng {
    /// Create new Philox RNG with seed and stream ID.
    pub fn new(seed: u64, stream: u64) -> Self {
        Self {
            state: <Self as GpuRng>::seed(seed, stream),
            buffer: [0; 4],
            buffer_idx: 4, // Force generation on first call
        }
    }

    /// Create from existing state.
    pub fn from_state(state: PhiloxState) -> Self {
        Self {
            state,
            buffer: [0; 4],
            buffer_idx: 4,
        }
    }

    /// Get current state (for checkpointing).
    pub fn state(&self) -> PhiloxState {
        self.state
    }

    /// Generate next u32 value.
    pub fn next_u32_raw(&mut self) -> u32 {
        if self.buffer_idx >= 4 {
            self.buffer = philox4x32_10(&mut self.state);
            self.buffer_idx = 0;
        }
        let val = self.buffer[self.buffer_idx];
        self.buffer_idx += 1;
        val
    }

    /// Generate next uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        let u = self.next_u32_raw();
        // Convert to float in [0, 1)
        // Use the upper 24 bits for f32 mantissa precision
        (u >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Generate next uniform f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        let u1 = self.next_u32_raw() as u64;
        let u2 = self.next_u32_raw() as u64;
        let combined = (u1 << 32) | u2;
        // Use upper 53 bits for f64 mantissa precision
        (combined >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Single round of Philox mixing.
#[inline]
fn philox_round(ctr: &mut [u32; 4], key: &[u32; 2]) {
    let hi0 = ((ctr[0] as u64 * PHILOX_M4X32_0 as u64) >> 32) as u32;
    let lo0 = ctr[0].wrapping_mul(PHILOX_M4X32_0);
    let hi1 = ((ctr[2] as u64 * PHILOX_M4X32_1 as u64) >> 32) as u32;
    let lo1 = ctr[2].wrapping_mul(PHILOX_M4X32_1);

    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
}

/// Full Philox4x32-10 function (10 rounds).
fn philox4x32_10(state: &mut PhiloxState) -> [u32; 4] {
    let mut ctr = state.counter;
    // Use all 4 key elements: mix seed (0,1) with stream (2,3)
    let mut key = [state.key[0] ^ state.key[2], state.key[1] ^ state.key[3]];

    // 10 rounds
    for _ in 0..10 {
        philox_round(&mut ctr, &key);
        key[0] = key[0].wrapping_add(PHILOX_W32_0);
        key[1] = key[1].wrapping_add(PHILOX_W32_1);
    }

    // Increment counter
    state.counter[0] = state.counter[0].wrapping_add(1);
    if state.counter[0] == 0 {
        state.counter[1] = state.counter[1].wrapping_add(1);
        if state.counter[1] == 0 {
            state.counter[2] = state.counter[2].wrapping_add(1);
            if state.counter[2] == 0 {
                state.counter[3] = state.counter[3].wrapping_add(1);
            }
        }
    }

    ctr
}

impl GpuRng for PhiloxRng {
    type State = PhiloxState;

    fn next_uniform(state: &mut Self::State) -> f32 {
        let output = philox4x32_10(state);
        // Use first output, convert to float
        (output[0] >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    fn skip(state: &mut Self::State, n: u64) {
        // Increment counter by n
        let n_lo = (n & 0xFFFFFFFF) as u32;
        let n_hi = ((n >> 32) & 0xFFFFFFFF) as u32;

        let (new_c0, carry0) = state.counter[0].overflowing_add(n_lo);
        state.counter[0] = new_c0;

        let (new_c1, carry1) = state.counter[1].overflowing_add(n_hi);
        let (new_c1, carry1b) = new_c1.overflowing_add(carry0 as u32);
        state.counter[1] = new_c1;

        if carry1 || carry1b {
            state.counter[2] = state.counter[2].wrapping_add(1);
            if state.counter[2] == 0 {
                state.counter[3] = state.counter[3].wrapping_add(1);
            }
        }
    }

    fn seed(seed: u64, stream: u64) -> Self::State {
        PhiloxState {
            counter: [0, 0, 0, 0],
            key: [
                (seed & 0xFFFFFFFF) as u32,
                ((seed >> 32) & 0xFFFFFFFF) as u32,
                (stream & 0xFFFFFFFF) as u32,
                ((stream >> 32) & 0xFFFFFFFF) as u32,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_philox_state_size() {
        assert_eq!(std::mem::size_of::<PhiloxState>(), 32);
    }

    #[test]
    fn test_philox_reproducible() {
        let mut rng1 = PhiloxRng::new(42, 0);
        let mut rng2 = PhiloxRng::new(42, 0);

        for _ in 0..100 {
            assert_eq!(rng1.next_u32_raw(), rng2.next_u32_raw());
        }
    }

    #[test]
    fn test_philox_different_seeds() {
        let mut rng1 = PhiloxRng::new(42, 0);
        let mut rng2 = PhiloxRng::new(43, 0);

        let mut same = true;
        for _ in 0..10 {
            if rng1.next_u32_raw() != rng2.next_u32_raw() {
                same = false;
                break;
            }
        }
        assert!(!same, "Different seeds should produce different sequences");
    }

    #[test]
    fn test_philox_different_streams() {
        let mut rng1 = PhiloxRng::new(42, 0);
        let mut rng2 = PhiloxRng::new(42, 1);

        let mut same = true;
        for _ in 0..10 {
            if rng1.next_u32_raw() != rng2.next_u32_raw() {
                same = false;
                break;
            }
        }
        assert!(
            !same,
            "Different streams should produce different sequences"
        );
    }

    #[test]
    fn test_philox_uniform_range() {
        let mut rng = PhiloxRng::new(12345, 0);

        for _ in 0..1000 {
            let u = rng.next_f32();
            assert!(
                (0.0..1.0).contains(&u),
                "Uniform should be in [0, 1), got {}",
                u
            );
        }
    }

    #[test]
    fn test_philox_uniform_f64_range() {
        let mut rng = PhiloxRng::new(12345, 0);

        for _ in 0..1000 {
            let u = rng.next_f64();
            assert!(
                (0.0..1.0).contains(&u),
                "Uniform f64 should be in [0, 1), got {}",
                u
            );
        }
    }

    #[test]
    fn test_gpu_rng_trait() {
        let mut state = PhiloxRng::seed(42, 0);

        // Test uniform generation
        let u = PhiloxRng::next_uniform(&mut state);
        assert!((0.0..1.0).contains(&u));

        // Test normal generation
        let z = PhiloxRng::next_normal(&mut state);
        // Normal should be roughly in [-4, 4] almost always
        assert!(z.abs() < 10.0);

        // Test skip
        let state_before = state;
        PhiloxRng::skip(&mut state, 100);
        assert_ne!(state.counter, state_before.counter);
    }

    #[test]
    fn test_philox_distribution_mean() {
        let mut rng = PhiloxRng::new(99, 0);
        let n = 10000;

        let sum: f64 = (0..n).map(|_| rng.next_f64()).sum();
        let mean = sum / n as f64;

        // Mean should be close to 0.5 for uniform [0, 1)
        assert!(
            (mean - 0.5).abs() < 0.02,
            "Mean {} should be close to 0.5",
            mean
        );
    }
}
