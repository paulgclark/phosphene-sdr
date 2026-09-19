// SPDX-License-Identifier: MIT

//! Deterministic-mode seal test (lane M1-A): same seed and same Δt →
//! bit-identical accumulator state across runs. Every later golden test
//! (headless PNGs, perceptual diffs) leans on this.

use phosphene_core::{
    Complex, LiveTrace, MaxHold, PersistenceHistogram, SpectrumAnalyzer, WindowKind,
};

const N: usize = 512;
const FRAMES: usize = 60;
const SPECTRA_PER_TICK: usize = 6;
const DT: f32 = 1.0 / 60.0;
/// Per-spectrum interval for the live-trace EMA (N / a nominal 1 MS/s).
const SPECTRUM_INTERVAL: f32 = N as f32 / 1.0e6;

/// SplitMix64 — the fixed seed of "deterministic mode".
struct Rng {
    state: u64,
}

impl Rng {
    fn next_unit(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

/// One full pipeline run: seeded noise + tone frames → spectra → all three
/// accumulators, ticked at fixed Δt. Returns the final state as raw bits.
fn run(seed: u64) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut rng = Rng { state: seed };
    let mut analyzer = SpectrumAnalyzer::new(N, WindowKind::Hann).unwrap();
    let mut histogram = PersistenceHistogram::new(N, 128, -100.0, 0.0).unwrap();
    let mut live = LiveTrace::new(N, SPECTRUM_INTERVAL, 0.1).unwrap();
    let mut max_hold = MaxHold::new(N).unwrap();

    let mut frame = vec![Complex::new(0.0f32, 0.0f32); N];
    // Spectra collect into a contiguous batch so the live trace runs its
    // production closed-form path.
    let mut batch = vec![0.0f32; SPECTRA_PER_TICK * N];
    for f in 0..FRAMES {
        // §7.5 frame order: decay the RETAINED max-hold trace first (toward
        // the live trace as of the previous frame), so the maxima folded in
        // below are never decayed on the frame they appear. On the first
        // frame both traces are unseeded and this is a no-op.
        max_hold.tick(DT, live.trace());
        for s in 0..SPECTRA_PER_TICK {
            for (i, x) in frame.iter_mut().enumerate() {
                let phase = 2.0 * std::f32::consts::PI * 40.0 * i as f32 / N as f32;
                let noise_re = 0.01 * (rng.next_unit() - 0.5);
                let noise_im = 0.01 * (rng.next_unit() - 0.5);
                // Tone bursts on and off so every accumulator branch runs.
                let amp = if (f / 10) % 2 == 0 { 0.5 } else { 0.0 };
                *x = Complex::new(amp * phase.cos() + noise_re, amp * phase.sin() + noise_im);
            }
            let dbfs = &mut batch[s * N..(s + 1) * N];
            analyzer.process(&frame, dbfs);
            histogram.accumulate(dbfs);
            max_hold.accumulate(dbfs);
        }
        live.accumulate_batch(&batch);
        histogram.tick(DT);
    }

    (
        histogram.intensity().iter().map(|v| v.to_bits()).collect(),
        live.trace().iter().map(|v| v.to_bits()).collect(),
        max_hold.trace().iter().map(|v| v.to_bits()).collect(),
    )
}

#[test]
fn same_seed_and_dt_give_bit_identical_state() {
    let a = run(0xDEAD_BEEF);
    let b = run(0xDEAD_BEEF);
    assert_eq!(a.0, b.0, "intensity grids differ between identical runs");
    assert_eq!(a.1, b.1, "live traces differ between identical runs");
    assert_eq!(a.2, b.2, "max-hold traces differ between identical runs");
}

#[test]
fn different_seed_actually_changes_the_state() {
    // Guard against the vacuous version of the test above.
    let a = run(0xDEAD_BEEF);
    let b = run(0x1234_5678);
    assert_ne!(a.1, b.1, "live trace insensitive to the seed");
}
