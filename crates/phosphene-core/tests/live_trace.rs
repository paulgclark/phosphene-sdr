// SPDX-License-Identifier: MIT

//! Live-trace seal tests (spec §7.4, lane M1-A R2): the closed-form batch
//! path is the production path, and its correctness argument is agreement
//! with the naive per-spectrum recurrence — identical in exact arithmetic,
//! asserted here in `f32` within a stated tolerance over realistic batches.

use phosphene_core::LiveTrace;

const BINS: usize = 1024;
/// Per-spectrum interval at the NFR-P1/D-012 operating point:
/// N = 1024 at 20 MS/s.
const SPECTRUM_INTERVAL: f32 = 1024.0 / 20.0e6;
const TAU_LIVE: f32 = 0.1;
/// §7.0 sizes batches at 2–8 ms of work; at this operating point that is
/// ~39–156 spectra. K = 128 sits in that range.
const BATCH_SPECTRA: usize = 128;
/// Stated equivalence tolerance: the two paths differ only by `f32`
/// rounding over the K-step folds; 0.05 dB is invisible against the
/// ~100 dB displayed range while still far tighter than any real effect.
const TOLERANCE_DB: f32 = 0.05;

/// Deterministic SplitMix64 dBFS spectra in [−90, −10].
struct SpectrumGen {
    state: u64,
}

impl SpectrumGen {
    fn fill(&mut self, out: &mut [f32]) {
        for v in out {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let unit = (z >> 11) as f64 / (1u64 << 53) as f64;
            *v = (-90.0 + 80.0 * unit) as f32;
        }
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn closed_form_batch_matches_the_per_spectrum_oracle() {
    let mut production = LiveTrace::new(BINS, SPECTRUM_INTERVAL, TAU_LIVE).unwrap();
    let mut oracle = LiveTrace::new(BINS, SPECTRUM_INTERVAL, TAU_LIVE).unwrap();
    let mut gen = SpectrumGen { state: 0xF00D_F00D };

    // Several consecutive batches, cold start included, so the equivalence
    // covers seeding, the first blend, and steady state.
    let mut batch = vec![0.0f32; BATCH_SPECTRA * BINS];
    for _ in 0..4 {
        gen.fill(&mut batch);
        production.accumulate_batch(&batch);
        for spectrum in batch.as_chunks::<BINS>().0 {
            oracle.accumulate(spectrum);
        }
        let diff = max_abs_diff(production.trace(), oracle.trace());
        assert!(
            diff <= TOLERANCE_DB,
            "closed form diverged from the per-spectrum oracle by {diff} dB \
             (tolerance {TOLERANCE_DB} dB) over K = {BATCH_SPECTRA}"
        );
    }
}

#[test]
fn single_spectrum_batch_is_bit_identical_to_the_oracle() {
    // For K = 1 the closed form algebraically reduces to the naive update;
    // in f32 the two must agree to the bit.
    let mut production = LiveTrace::new(8, SPECTRUM_INTERVAL, TAU_LIVE).unwrap();
    let mut oracle = LiveTrace::new(8, SPECTRUM_INTERVAL, TAU_LIVE).unwrap();
    let mut gen = SpectrumGen { state: 42 };

    let mut spectrum = [0.0f32; 8];
    for _ in 0..50 {
        gen.fill(&mut spectrum);
        production.accumulate_batch(&spectrum);
        oracle.accumulate(&spectrum);
        let (p, o): (Vec<u32>, Vec<u32>) = (
            production.trace().iter().map(|v| v.to_bits()).collect(),
            oracle.trace().iter().map(|v| v.to_bits()).collect(),
        );
        assert_eq!(p, o);
    }
}

#[test]
fn empty_batch_is_a_no_op() {
    let mut lt = LiveTrace::new(8, SPECTRUM_INTERVAL, TAU_LIVE).unwrap();
    let mut spectrum = [0.0f32; 8];
    SpectrumGen { state: 1 }.fill(&mut spectrum);
    lt.accumulate(&spectrum);
    let before: Vec<f32> = lt.trace().to_vec();
    lt.accumulate_batch(&[]);
    assert_eq!(lt.trace(), &before[..]);
}

#[test]
#[should_panic(expected = "whole number of spectra")]
fn ragged_batch_is_rejected() {
    let mut lt = LiveTrace::new(8, SPECTRUM_INTERVAL, TAU_LIVE).unwrap();
    lt.accumulate_batch(&[0.0; 12]);
}
