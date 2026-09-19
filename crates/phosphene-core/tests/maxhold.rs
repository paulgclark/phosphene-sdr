// SPDX-License-Identifier: MIT

//! Max-hold seal tests (spec §7.5, D-007, lane M1-A): pure hold never
//! decreases until reset; decay-toward-live converges toward the live trace
//! at its own time constant.

mod common;

use common::{assert_tau_close, fit_exponential_tau};
use phosphene_core::{MaxHold, MaxHoldMode, DBFS_FLOOR};

const DT: f32 = 1.0 / 60.0;
const BINS: usize = 4;

/// Deterministic SplitMix64 stream of synthetic dBFS spectra in [−90, −10].
struct SpectrumGen {
    state: u64,
}

impl SpectrumGen {
    fn new(seed: u64) -> Self {
        SpectrumGen { state: seed }
    }

    fn next_spectrum(&mut self) -> [f32; BINS] {
        let mut s = [0.0f32; BINS];
        for v in &mut s {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let unit = (z >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
            *v = (-90.0 + 80.0 * unit) as f32;
        }
        s
    }
}

#[test]
fn pure_hold_is_the_running_max_and_never_decreases_until_reset() {
    let mut mh = MaxHold::new(BINS).unwrap();
    mh.set_mode(MaxHoldMode::PureHold);
    let mut gen = SpectrumGen::new(7);
    let live = [-90.0f32; BINS];

    let mut running_max = [f32::NEG_INFINITY; BINS];
    for _ in 0..500 {
        // §7.5 frame order: tick first (a no-op here — d = 1), then fold the
        // frame's spectrum.
        let before: Vec<f32> = mh.trace().to_vec();
        mh.tick(DT, &live);
        assert_eq!(mh.trace(), &before[..], "pure-hold tick moved the trace");

        let s = gen.next_spectrum();
        for (m, &p) in running_max.iter_mut().zip(&s) {
            *m = m.max(p);
        }
        mh.accumulate(&s);
        assert_eq!(mh.trace(), &running_max[..]);
    }

    // Reset, then the trace re-seeds from the next spectrum alone.
    mh.reset();
    let s = gen.next_spectrum();
    mh.accumulate(&s);
    assert_eq!(mh.trace(), &s[..]);
}

/// D-007 decay-toward-live at one τ setting: after a burst, the excess over a
/// steady live trace relaxes as `exp(−t/τ)`.
fn decay_toward_live_case(tau: f32) {
    let mut mh = MaxHold::new(BINS).unwrap();
    assert_eq!(mh.mode(), MaxHoldMode::DecayTowardLive);
    mh.set_tau(tau).unwrap();

    let live = [-60.0f32; BINS];
    let steady = [-60.0f32; BINS];

    // Burst seeds the trace well above the live level...
    mh.accumulate(&[-10.0; BINS]);
    // ...then the input sits at the live level while the hold decays.
    // §7.5 frame order: decay the retained trace first, then fold the
    // frame's batch.
    let mut samples = Vec::new();
    let mut t = 0.0f32;
    while samples.len() < 4000 {
        mh.tick(DT, &live);
        mh.accumulate(&steady);
        t += DT;
        let excess = mh.trace()[0] - live[0];
        if excess < 0.05 {
            break;
        }
        samples.push((t, excess));
    }
    let fitted = fit_exponential_tau(&samples);
    assert_tau_close(fitted, tau, 0.05, "max-hold decay-toward-live");

    // Converged onto the live trace, never through it.
    for &m in mh.trace() {
        assert!(m >= live[0] - 1e-3, "hold fell below live: {m}");
        assert!(m - live[0] < 0.1, "hold failed to converge: {m}");
    }
}

#[test]
fn decay_toward_live_converges_at_fast_tau() {
    decay_toward_live_case(0.5);
}

#[test]
fn decay_toward_live_converges_at_slow_tau() {
    decay_toward_live_case(2.0);
}

#[test]
fn decay_mode_still_captures_new_maxima_instantly() {
    let mut mh = MaxHold::new(BINS).unwrap();
    mh.set_tau(1.0).unwrap();
    let live = [-60.0f32; BINS];

    mh.accumulate(&[-30.0; BINS]);
    for _ in 0..30 {
        mh.tick(DT, &live);
        mh.accumulate(&[-60.0; BINS]);
    }
    let decayed = mh.trace()[0];
    assert!(
        decayed < -30.0,
        "hold should have decayed below the old peak"
    );

    // A fresh, louder burst lands via max, not via the decay path.
    mh.accumulate(&[-5.0; BINS]);
    assert_eq!(mh.trace()[0], -5.0);
}

#[test]
fn fresh_peak_reads_its_exact_undecayed_value_on_the_frame_it_appears() {
    // §7.5: M_k = max(M_k · d, max over batch of P_k) — d multiplies ONLY
    // the retained M_k. Under the tick-then-accumulate frame order, a peak
    // first appearing in the current batch must be reported bit-exactly,
    // with no decay applied on its first frame.
    let mut mh = MaxHold::new(BINS).unwrap();
    mh.set_tau(1.0).unwrap();
    let live = [-70.0f32; BINS];

    // Establish a retained trace below the coming peak.
    mh.accumulate(&[-50.0; BINS]);
    let peak = -12.5f32; // exactly representable; nothing to blame on rounding

    // The frame in which the peak first appears: decay the retained trace,
    // then fold the batch (the peak spectrum plus a quieter one).
    mh.tick(DT, &live);
    mh.accumulate(&[peak; BINS]);
    mh.accumulate(&[-60.0; BINS]);
    assert_eq!(
        mh.trace()[0].to_bits(),
        peak.to_bits(),
        "first-frame peak was decayed: read {} instead of {peak}",
        mh.trace()[0]
    );
}

#[test]
fn retained_peak_decays_on_subsequent_frames_at_the_rate_alpha_implies() {
    // The same peak, on the frames AFTER it appeared: the retained value
    // relaxes toward live by exactly α = 1 − exp(−Δt/τ) per frame.
    let tau = 1.0f32;
    let mut mh = MaxHold::new(BINS).unwrap();
    mh.set_tau(tau).unwrap();
    let live = [-70.0f32; BINS];
    let peak = -12.5f32;

    mh.accumulate(&[-50.0; BINS]);
    mh.tick(DT, &live);
    mh.accumulate(&[peak; BINS]);
    assert_eq!(mh.trace()[0], peak);

    // Mirror the tick recurrence exactly; quieter batches must not disturb
    // the hold. 100 frames keeps the trajectory above the −60 dB batches.
    let alpha = 1.0 - (-DT / tau).exp();
    let mut expected = peak;
    for frame in 0..100 {
        mh.tick(DT, &live);
        mh.accumulate(&[-60.0; BINS]);
        expected += alpha * (live[0] - expected);
        let m = mh.trace()[0];
        assert!(
            (m - expected).abs() <= 1e-4,
            "frame {frame}: hold at {m}, α-recurrence expects {expected}"
        );
    }
    // Sanity: it actually decayed a measurable amount in that time.
    assert!(mh.trace()[0] < peak - 20.0);
}

#[test]
fn tick_before_seeding_and_after_reset_is_a_no_op() {
    // The tick-first frame order leads with a tick on the very first frame;
    // §7.5 has nothing retained to decay yet, so the trace must stay at the
    // floor — not get pulled toward live — and the first spectrum must still
    // seed exactly.
    let mut mh = MaxHold::new(BINS).unwrap();
    let live = [-40.0f32; BINS];
    for _ in 0..10 {
        mh.tick(DT, &live);
    }
    assert_eq!(mh.trace(), &[DBFS_FLOOR; BINS]);
    mh.accumulate(&[-33.0; BINS]);
    assert_eq!(mh.trace(), &[-33.0f32; BINS]);

    // Same contract immediately after a reset.
    mh.reset();
    mh.tick(DT, &live);
    assert_eq!(mh.trace(), &[DBFS_FLOOR; BINS]);
    mh.accumulate(&[-27.0; BINS]);
    assert_eq!(mh.trace(), &[-27.0f32; BINS]);
}

#[test]
fn mode_is_selectable_at_runtime() {
    // D-007: the accumulator carries a mode; pure hold is user-selectable.
    let mut mh = MaxHold::new(BINS).unwrap();
    mh.set_mode(MaxHoldMode::PureHold);
    let live = [-90.0f32; BINS];

    mh.accumulate(&[-20.0; BINS]);
    for _ in 0..60 {
        mh.tick(DT, &live);
    }
    assert_eq!(mh.trace()[0], -20.0, "pure hold must not decay");

    // Switching to decay-toward-live takes effect on subsequent ticks.
    mh.set_mode(MaxHoldMode::DecayTowardLive);
    mh.tick(DT, &live);
    assert!(
        mh.trace()[0] < -20.0,
        "decay mode did not engage after switch"
    );
}
