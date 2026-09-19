// SPDX-License-Identifier: MIT

//! Deterministic pseudo-randomness for the signal generator.
//!
//! In-house on purpose: the siggen's determinism contract ("a seed produces
//! bit-identical output") must not hinge on an external crate's algorithm
//! staying put across versions. SplitMix64 (Steele/Lea/Flood's SplittableRandom
//! finalizer, published public-domain by Vigna) is tiny, statistically fine for
//! noise synthesis, and fully under our control. Not cryptographic — nothing
//! here needs to be.

/// The SplitMix64 output finalizer as a pure function: hashes `z` to a
/// well-mixed `u64`. Used both by the sequential generator below and directly
/// as a stateless per-slot hash (the hopper's pseudorandom hop order).
pub fn mix(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Sequential SplitMix64 generator.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

/// The golden-ratio increment γ of SplitMix64.
const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

impl SplitMix64 {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GAMMA);
        mix(self.state)
    }

    /// Next uniform sample in the half-open interval `(0, 1]` — the open-zero
    /// end matters because Box-Muller takes `ln` of it.
    fn next_unit(&mut self) -> f64 {
        // Top 53 bits → an integer in [1, 2^53], scaled by 2^-53.
        ((self.next_u64() >> 11) + 1) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Next pair of independent standard-normal samples (Box-Muller). One call
    /// per complex noise sample: the pair is its I and Q.
    pub fn next_gaussian_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn known_first_output_for_seed_zero() {
        // First output for seed 0 is mix(γ) — pinned so an accidental
        // algorithm change cannot pass silently.
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), mix(GAMMA));
        assert_eq!(mix(GAMMA), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn gaussian_pairs_have_roughly_unit_variance_and_zero_mean() {
        let mut r = SplitMix64::new(7);
        let n = 100_000;
        let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let (a, b) = r.next_gaussian_pair();
            sum += a + b;
            sum_sq += a * a + b * b;
        }
        let count = (2 * n) as f64;
        assert!((sum / count).abs() < 0.01);
        assert!((sum_sq / count - 1.0).abs() < 0.02);
    }
}
