// SPDX-License-Identifier: MIT

//! Seal tests for the CL-1 classify-core lane
//! (`specs/002-nextgen-analyze/lanes/cl-1-classify-core.md`), all against M0
//! signal-generator ground truth through the offline harness. Each D-102
//! item 9 criterion, and what proves it:
//!
//! * OOK symbol rate within 1% at ≥12 dB SNR —
//!   `ook_symbol_rate_within_one_percent_at_grid_of_rates` (~250 symbols;
//!   see below for the bare-100-symbol point D-102 states);
//! * "with at least 100 symbols observed" —
//!   `ook_symbol_rate_never_wrong_at_the_100_symbol_floor` proves the
//!   invariant that survives at exactly 100 (never confidently wrong); a
//!   measured relaxation to ~250 symbols for *reliable* assertion is
//!   documented on `SYMBOL_COUNT_HALF` in `mod.rs`, not silently applied;
//! * FSK M exact for M = 2 and 4, deviation within 5% at ≥10 dB —
//!   `fsk_levels_exact_and_deviation_within_five_percent`;
//! * FSK symbol rate — `fsk_symbol_rate_is_never_asserted`: deliberately not
//!   delivered, with the measurement that shows why on the FSK branch of
//!   `classify` in `mod.rs` (future work, not a threshold to retune) —
//!   `symbol_rate_bd` is `None` there always, so there is no rate for a
//!   gating sweep to test either;
//! * gating switches within ~2 dB of the calibrated floor (this *is* the
//!   NFR-A3 honesty test, spec §10) — the coarse form,
//!   `ook_symbol_rate_gating_switches_near_its_calibrated_floor` and
//!   `fsk_deviation_gating_switches_near_its_calibrated_floor` (assert vs.
//!   withhold at floor ± 6 dB), and the precise form,
//!   `confidence_follows_the_snr_gate_curve_near_the_floor` (the *reported
//!   confidence* at floor − 2, floor, and floor + 2 dB, checked against
//!   `snr_gate`'s own calibrated curve — this is `snr_gate`'s specific
//!   negative control; the coarse sweeps and the false-positive budget stay
//!   green even with `snr_gate` disabled, since other qualitative gates
//!   already withhold at floor − 6 dB, so this is the test that actually
//!   exercises it);
//! * the false-positive budget (NFR-A3, a merge gate): ≥1,000 trials of
//!   noise-only and out-of-scope inputs (tones, chirps, hoppers), at most 1%
//!   confident-wrong — `false_positive_budget_on_noise_and_out_of_scope_signals`;
//! * determinism (NFR-A4): identical input and config give an identical
//!   `Classification` — `classification_is_deterministic_for_a_fixed_seed`.
//!
//! Negative controls (for the reviewer, per the lane's seal) — two distinct
//! ones, since two distinct kinds of gate are load-bearing here:
//! * `snr_gate` (the quantitative floor): see
//!   `confidence_follows_the_snr_gate_curve_near_the_floor`, above — the
//!   *only* test that fails when `snr_gate` is replaced with an
//!   unconditional `1.0`. Removing the `>= ASSERT_THRESHOLD` checks in
//!   `classify`'s OOK/FSK branches entirely (always take the "confident"
//!   arm) does not fail anything below either — confirmed by hand, not
//!   asserted here — because the qualitative gate, next, is what is
//!   actually filtering noise out;
//! * `OOK_LINE_PROMINENCE_MIN` (the qualitative gate that keeps a noise
//!   realization's own spurious "line" from reading as keying — see its
//!   doc comment in `mod.rs`): setting it to `0.0` makes
//!   `false_positive_budget_on_noise_and_out_of_scope_signals` fail
//!   deterministically (confirmed by hand: ~22% confident-wrong, far past
//!   the 1% budget), on any machine, unloaded.

use phosphene_core::Complex;
use phosphene_sources::siggen::{FskConfig, OokConfig};
use phosphene_sources::{
    ChirpConfig, HopOrder, HopperConfig, NoiseConfig, SigGen, SigGenConfig, ToneConfig,
};

use super::classify;
use crate::harness::{OfflineTap, OfflineTapConfig};
use crate::tap::{AnalysisTap, BandMeta, IqBuf, TimeSpan};
use crate::track::{Classification, ModulationClass, Track, TrackId, TrackState};

const RATE: f64 = 200_000.0;
const NOISE_TOTAL_DBFS: f32 = -30.0;

/// A minimal confirmed track over the given carrier/bandwidth, carrying the
/// SNR classify's gates key off (as if the measure stage had already
/// computed it).
fn track(offset_hz: f64, bandwidth_hz: f64, snr_db: f32) -> Track {
    Track {
        id: TrackId(0),
        state: TrackState::Confirmed,
        center_offset_hz: offset_hz,
        bandwidth_hz,
        power_dbfs: 0.0,
        snr_db,
        born_frame: 0,
        last_seen_frame: 0,
        measurements: None,
        class: None,
        signatures: Vec::new(),
    }
}

/// Feed `scene` through the offline harness and pull a `duration_s` IQ
/// snapshot, along with the matching band metadata.
fn snapshot_of(scene: &SigGenConfig, duration_s: f64) -> (IqBuf, BandMeta) {
    let mut cfg = OfflineTapConfig::new(scene.sample_rate_hz);
    cfg.iq_history_s = duration_s * 1.5;
    let mut tap = OfflineTap::new(cfg).unwrap();
    let mut gen = SigGen::new(scene.clone()).unwrap();

    let mut chunk = vec![Complex::new(0.0f32, 0.0); 4096];
    let needed = (duration_s * scene.sample_rate_hz).ceil() as usize + chunk.len();
    let mut fed = 0usize;
    while fed < needed {
        gen.fill(&mut chunk);
        tap.feed(&chunk);
        fed += chunk.len();
    }

    let mut buf = IqBuf::default();
    tap.snapshot(TimeSpan::last(duration_s), &mut buf).unwrap();
    let band = BandMeta {
        center_freq_hz: scene.center_freq_hz,
        span_hz: scene.sample_rate_hz,
        bins: 1024,
        bin_hz: scene.sample_rate_hz / 1024.0,
        frame_dt_s: 1024.0 / scene.sample_rate_hz,
        cal_offset_db: 0.0,
    };
    (buf, band)
}

/// The per-bin noise power a generator `NoiseConfig` puts inside a
/// `bandwidth_hz`-wide slice of the full `sample_rate_hz` span (dBFS): the
/// generator's total noise power is spread uniformly over the whole span.
fn noise_in_band_dbfs(noise_total_dbfs: f32, bandwidth_hz: f64, sample_rate_hz: f64) -> f64 {
    f64::from(noise_total_dbfs) + 10.0 * (bandwidth_hz / sample_rate_hz).log10()
}

/// The carrier level (dBFS) that gives `snr_db` against the generator's
/// noise floor, in-band.
fn level_for_snr(
    snr_db: f64,
    noise_total_dbfs: f32,
    bandwidth_hz: f64,
    sample_rate_hz: f64,
) -> f32 {
    (snr_db + noise_in_band_dbfs(noise_total_dbfs, bandwidth_hz, sample_rate_hz)) as f32
}

fn ook_scene(
    seed: u64,
    offset_hz: f64,
    symbol_rate_bd: f64,
    snr_db: f64,
    bandwidth_hz: f64,
) -> SigGenConfig {
    let mut scene = SigGenConfig::new(RATE);
    scene.seed = seed;
    scene.noise = Some(NoiseConfig {
        level_dbfs: NOISE_TOTAL_DBFS,
    });
    scene.ook.push(OokConfig {
        offset_hz,
        symbol_rate_bd,
        level_dbfs: level_for_snr(snr_db, NOISE_TOTAL_DBFS, bandwidth_hz, RATE),
    });
    scene
}

fn fsk_scene(
    seed: u64,
    offset_hz: f64,
    levels: u32,
    deviation_hz: f64,
    symbol_rate_bd: f64,
    snr_db: f64,
    bandwidth_hz: f64,
) -> SigGenConfig {
    let mut scene = SigGenConfig::new(RATE);
    scene.seed = seed;
    scene.noise = Some(NoiseConfig {
        level_dbfs: NOISE_TOTAL_DBFS,
    });
    scene.fsk.push(FskConfig {
        offset_hz,
        levels,
        deviation_hz,
        symbol_rate_bd,
        level_dbfs: level_for_snr(snr_db, NOISE_TOTAL_DBFS, bandwidth_hz, RATE),
    });
    scene
}

// ---------------------------------------------------------------------
// Ground-truth grid (D-102 item 9)
// ---------------------------------------------------------------------

#[test]
fn ook_symbol_rate_within_one_percent_at_grid_of_rates() {
    // ~250 symbols: measured reliable (see SYMBOL_COUNT_HALF's doc comment
    // and the test below for the bare-100-symbol point D-102 item 9 states,
    // and why it is tested separately rather than folded in here).
    for &rate_bd in &[2_000.0, 5_000.0, 10_000.0] {
        let bandwidth_hz = 4.0 * rate_bd;
        let offset_hz = 40_000.0;
        let duration_s = 250.0 / rate_bd;
        let scene = ook_scene(rate_bd as u64 + 1, offset_hz, rate_bd, 12.0, bandwidth_hz);
        let (snapshot, band) = snapshot_of(&scene, duration_s);
        let t = track(offset_hz, bandwidth_hz, 12.0);
        let result = classify(&snapshot, &t, &band).expect("classify must run");

        assert_eq!(result.class, ModulationClass::OokAsk, "{result:#?}");
        let measured = result
            .symbol_rate_bd
            .unwrap_or_else(|| panic!("rate must be asserted at 12 dB / 250 symbols: {result:#?}"));
        let err = (measured.value - rate_bd).abs() / rate_bd;
        assert!(
            err < 0.01,
            "rate {rate_bd} Bd: got {}, err {err}",
            measured.value
        );
    }
}

/// D-102 item 9 states the symbol-rate floor "with at least 100 symbols
/// observed". Measured against the generator at exactly 100 symbols / 12 dB
/// (30 seeds): the rate is asserted 0/30 times — [`OOK_LINE_PROMINENCE_MIN`]
/// (calibrated for the false-positive budget) is the binding constraint
/// there, not the observed-symbol-count factor, and reliable assertion
/// (~29/30) does not begin until ~250 symbols. This is a measured relaxation
/// of "100" to "~250" (reported here, not silently) — but the invariant
/// D-102/NFR-A3 actually care about, that a bare 100 symbols never produces
/// a *confidently wrong* rate, is real and is what this test pins.
///
/// [`OOK_LINE_PROMINENCE_MIN`]: super::OOK_LINE_PROMINENCE_MIN
#[test]
fn ook_symbol_rate_never_wrong_at_the_100_symbol_floor() {
    const RATE_BD: f64 = 2_000.0;
    const BANDWIDTH_HZ: f64 = 4.0 * RATE_BD;
    const OFFSET_HZ: f64 = 40_000.0;
    let duration_s = 100.0 / RATE_BD;
    for seed in 0..10u64 {
        let scene = ook_scene(seed, OFFSET_HZ, RATE_BD, 12.0, BANDWIDTH_HZ);
        let (snapshot, band) = snapshot_of(&scene, duration_s);
        let t = track(OFFSET_HZ, BANDWIDTH_HZ, 12.0);
        let result = classify(&snapshot, &t, &band).unwrap();
        if let Some(m) = result.symbol_rate_bd {
            let err = (m.value - RATE_BD).abs() / RATE_BD;
            assert!(
                err < 0.01,
                "seed {seed}: asserted {} Bd, err {err} — confidently wrong: {result:#?}",
                m.value
            );
        }
    }
}

#[test]
fn fsk_levels_exact_and_deviation_within_five_percent() {
    for &levels in &[2u32, 4] {
        // A denser tone ladder (more levels over the same span) needs more
        // separation per level to stay resolvable against the same 10 dB
        // noise floor — the generator scene's own parameter, not a change
        // to the estimator.
        let deviation_hz = if levels == 2 { 8_000.0 } else { 16_000.0 };
        let symbol_rate_bd = 4_000.0;
        // A denser tone ladder needs more margin beyond the outermost tone
        // for its filter to stay flat there (the classifier's own DDC
        // low-pass, per the track's *reported* bandwidth) — again the
        // scene's own parameter, matching what a real measurement of a
        // wider guard band would report.
        let margin = if levels == 2 { 1.0 } else { 1.3 };
        let bandwidth_hz = ((levels as f64 - 1.0) * deviation_hz + 2.0 * symbol_rate_bd) * margin;
        let offset_hz = -30_000.0;
        let duration_s = 250.0 / symbol_rate_bd;
        let scene = fsk_scene(
            u64::from(levels) + 100,
            offset_hz,
            levels,
            deviation_hz,
            symbol_rate_bd,
            10.0,
            bandwidth_hz,
        );
        let (snapshot, band) = snapshot_of(&scene, duration_s);
        let t = track(offset_hz, bandwidth_hz, 10.0);
        let result = classify(&snapshot, &t, &band).expect("classify must run");

        match &result.class {
            ModulationClass::Fsk {
                levels: got_levels,
                deviation_hz: got_dev,
            } => {
                assert_eq!(*got_levels, levels, "{result:#?}");
                let err = (got_dev.value - deviation_hz).abs() / deviation_hz;
                assert!(err < 0.05, "deviation: got {}, err {err}", got_dev.value);
            }
            other => panic!("expected FSK, got {other:?}: {result:#?}"),
        }
    }
}

/// D-102 item 9 also asks for FSK's own symbol rate, and the brief names it
/// as an explicit consumer of [`OOK_LINE_PROMINENCE_MIN`]-style gating.
/// `classify`'s FSK branch does not attempt it: the quantized-transition
/// estimate, measured against the generator across M = 2 and 4, ≥12 dB, and
/// 100–300 symbols, is not merely noisy near the true rate but frequently
/// and substantially wrong (tens of percent), with no symbol count in that
/// range showing convergence — asserting it under any confidence gate would
/// mean gating passes wrong answers through, which is worse than a gate that
/// never opens. `symbol_rate_bd` for FSK is `None` unconditionally; this
/// pins that, so a future change to the FSK branch has to touch this test
/// rather than silently starting to assert an unverified number again. A
/// working FSK rate estimator is scoped as future work.
///
/// [`OOK_LINE_PROMINENCE_MIN`]: super::OOK_LINE_PROMINENCE_MIN
#[test]
fn fsk_symbol_rate_is_never_asserted() {
    const LEVELS: u32 = 4;
    const DEVIATION_HZ: f64 = 16_000.0;
    const RATE_BD: f64 = 4_000.0;
    const MARGIN: f64 = 1.3;
    let bandwidth_hz = ((LEVELS as f64 - 1.0) * DEVIATION_HZ + 2.0 * RATE_BD) * MARGIN;
    const OFFSET_HZ: f64 = -30_000.0;
    let duration_s = 300.0 / RATE_BD;
    let scene = fsk_scene(
        2,
        OFFSET_HZ,
        LEVELS,
        DEVIATION_HZ,
        RATE_BD,
        20.0,
        bandwidth_hz,
    );
    let (snapshot, band) = snapshot_of(&scene, duration_s);
    let t = track(OFFSET_HZ, bandwidth_hz, 20.0);
    let result = classify(&snapshot, &t, &band).unwrap();
    assert!(
        matches!(result.class, ModulationClass::Fsk { .. }),
        "{result:#?}"
    );
    assert!(result.symbol_rate_bd.is_none(), "{result:#?}");
}

// ---------------------------------------------------------------------
// SNR sweep (the NFR-A3 honesty test): gating switches near the floor.
// ---------------------------------------------------------------------

#[test]
fn ook_symbol_rate_gating_switches_near_its_calibrated_floor() {
    const RATE_BD: f64 = 5_000.0;
    const BANDWIDTH_HZ: f64 = 20_000.0;
    const OFFSET_HZ: f64 = 25_000.0;
    let duration_s = 300.0 / RATE_BD;

    let asserts_at = |snr_db: f64| -> bool {
        let scene = ook_scene(42, OFFSET_HZ, RATE_BD, snr_db, BANDWIDTH_HZ);
        let (snapshot, band) = snapshot_of(&scene, duration_s);
        let t = track(OFFSET_HZ, BANDWIDTH_HZ, snr_db as f32);
        let result = classify(&snapshot, &t, &band).unwrap();
        result.symbol_rate_bd.is_some()
    };

    // Well above the 12 dB floor: asserted. Well below: not.
    assert!(asserts_at(12.0 + 6.0), "must assert well above the floor");
    assert!(
        !asserts_at(12.0 - 6.0),
        "must not assert well below the floor"
    );
}

#[test]
fn fsk_deviation_gating_switches_near_its_calibrated_floor() {
    const LEVELS: u32 = 4;
    const DEVIATION_HZ: f64 = 8_000.0;
    const RATE_BD: f64 = 4_000.0;
    const BANDWIDTH_HZ: f64 = 40_000.0;
    const OFFSET_HZ: f64 = -20_000.0;
    let duration_s = 300.0 / RATE_BD;

    let asserts_fsk_at = |snr_db: f64| -> bool {
        let scene = fsk_scene(
            7,
            OFFSET_HZ,
            LEVELS,
            DEVIATION_HZ,
            RATE_BD,
            snr_db,
            BANDWIDTH_HZ,
        );
        let (snapshot, band) = snapshot_of(&scene, duration_s);
        let t = track(OFFSET_HZ, BANDWIDTH_HZ, snr_db as f32);
        let result = classify(&snapshot, &t, &band).unwrap();
        matches!(result.class, ModulationClass::Fsk { .. })
    };

    assert!(
        asserts_fsk_at(10.0 + 6.0),
        "must assert FSK well above the floor"
    );
    assert!(
        !asserts_fsk_at(10.0 - 6.0),
        "must not assert FSK well below the floor"
    );
}

/// The precise version of the SNR-sweep tests above: not just "asserts /
/// withholds" at floor ± 6 dB, but the *reported confidence* at floor − 2,
/// floor, and floor + 2 dB, checked against [`snr_gate`]'s own calibrated
/// logistic curve (`snr_gate`'s value at those three offsets is exactly
/// `1/(1+e²)` ≈ 0.119, `0.5`, and `1/(1+e⁻²)` ≈ 0.881 for
/// [`SNR_GATE_STEEPNESS_DB`] `= 1.0`) — this is what D-102 item 9's
/// "switches within 2 dB of its calibrated floor" actually asserts, rather
/// than a single pass/fail point on either side of it. Confidence is
/// checked even where the class itself is `Unknown` (floor − 2 dB, below
/// [`ASSERT_THRESHOLD`]): [`unknown_with_confidence`] carries the gate's raw
/// value there rather than a flat zero, specifically so this is observable.
///
/// Also pins the *decision* at floor ± 2 dB (withhold below, assert above),
/// not only the confidence number: the number alone does not depend on
/// [`ASSERT_THRESHOLD`] at all (it is `snr_gate`'s raw output regardless of
/// where the threshold sits), so moving `ASSERT_THRESHOLD` far enough to
/// pull the assert/withhold transition outside floor ± 2 dB — e.g. from
/// `0.4` to `0.05`, which drops the transition to about floor − 2.94 dB —
/// left every confidence-number assertion here passing. ~800 symbols
/// observed keeps the observed-symbol-count factor from being what decides
/// the boundary instead of `snr_gate`.
///
/// This test **is** the negative control the seal's "reverting the
/// confidence gates... makes the SNR-sweep... tests fail deterministically"
/// line promises, for two distinct mutations: replacing `snr_gate`'s body
/// with a constant `1.0` makes every confidence assertion fail immediately
/// (confidence would read `1.0` at all three points instead of the curve);
/// moving `ASSERT_THRESHOLD` outside the ~[0.12, 0.88] band the floor ± 2 dB
/// window produces makes the decision assertions fail instead, even though
/// the confidence numbers stay correct. Neither mutation is caught by the
/// coarser assert/withhold sweeps above — at floor − 6 dB the qualitative
/// gates (line prominence for OOK, mode count for FSK) already withhold on
/// their own, so those sweeps never isolate either gate.
#[test]
fn confidence_follows_the_snr_gate_curve_near_the_floor() {
    /// `snr_gate`'s exact values at floor − 2, floor, and floor + 2 dB for
    /// `SNR_GATE_STEEPNESS_DB = 1.0`, checked to a tolerance loose enough
    /// for the small amount of noise in an otherwise-deterministic estimate
    /// (a real capture, not a hand-fed number) but far tighter than the gap
    /// between them — nowhere close to what a disabled gate (flat `1.0`)
    /// would produce.
    const EXPECTED: [f32; 3] = [0.119_203, 0.5, 0.880_797];
    const TOLERANCE: f32 = 0.03;

    const OOK_RATE_BD: f64 = 5_000.0;
    const OOK_BANDWIDTH_HZ: f64 = 20_000.0;
    const OOK_OFFSET_HZ: f64 = 25_000.0;
    const OOK_FLOOR_DB: f64 = 12.0; // SYMBOL_RATE_SNR_FLOOR_DB
    let ook_duration_s = 800.0 / OOK_RATE_BD; // well past the reliable-assertion point
    for (i, &snr) in [OOK_FLOOR_DB - 2.0, OOK_FLOOR_DB, OOK_FLOOR_DB + 2.0]
        .iter()
        .enumerate()
    {
        let scene = ook_scene(42, OOK_OFFSET_HZ, OOK_RATE_BD, snr, OOK_BANDWIDTH_HZ);
        let (snapshot, band) = snapshot_of(&scene, ook_duration_s);
        let t = track(OOK_OFFSET_HZ, OOK_BANDWIDTH_HZ, snr as f32);
        let result = classify(&snapshot, &t, &band).unwrap();
        assert!(
            (result.confidence - EXPECTED[i]).abs() < TOLERANCE,
            "OOK at {snr} dB: confidence {} not within {TOLERANCE} of {}: {result:#?}",
            result.confidence,
            EXPECTED[i]
        );
        // The decision itself, not just the confidence number: pins the
        // assert/withhold transition to inside floor ± 2 dB (D-102 item 9),
        // independent of ASSERT_THRESHOLD's exact value — a threshold moved
        // far enough to put the transition outside this window (e.g. 0.4 to
        // 0.05, which pulls it to ~floor − 2.94 dB) fails this even though
        // the confidence *number* above is unaffected by ASSERT_THRESHOLD
        // at all. ~800 symbols observed here keeps the count factor away
        // from being what decides it.
        if i == 0 {
            assert!(
                result.symbol_rate_bd.is_none(),
                "OOK at floor - 2 dB ({snr} dB) must withhold the rate: {result:#?}"
            );
        } else if i == 2 {
            assert!(
                result.symbol_rate_bd.is_some(),
                "OOK at floor + 2 dB ({snr} dB) must assert the rate: {result:#?}"
            );
        }
    }

    const LEVELS: u32 = 2;
    const DEVIATION_HZ: f64 = 20_000.0;
    const RATE_BD: f64 = 4_000.0;
    const BANDWIDTH_HZ: f64 = 40_000.0;
    const OFFSET_HZ: f64 = -20_000.0;
    const FSK_FLOOR_DB: f64 = 10.0; // FSK_DEVIATION_SNR_FLOOR_DB
    let duration_s = 800.0 / RATE_BD;
    for (i, &snr) in [FSK_FLOOR_DB - 2.0, FSK_FLOOR_DB, FSK_FLOOR_DB + 2.0]
        .iter()
        .enumerate()
    {
        let scene = fsk_scene(
            20,
            OFFSET_HZ,
            LEVELS,
            DEVIATION_HZ,
            RATE_BD,
            snr,
            BANDWIDTH_HZ,
        );
        let (snapshot, band) = snapshot_of(&scene, duration_s);
        let t = track(OFFSET_HZ, BANDWIDTH_HZ, snr as f32);
        let result = classify(&snapshot, &t, &band).unwrap();
        assert!(
            (result.confidence - EXPECTED[i]).abs() < TOLERANCE,
            "FSK at {snr} dB: confidence {} not within {TOLERANCE} of {}: {result:#?}",
            result.confidence,
            EXPECTED[i]
        );
        // As above: the decision, not just the number.
        if i == 0 {
            assert!(
                !matches!(result.class, ModulationClass::Fsk { .. }),
                "FSK at floor - 2 dB ({snr} dB) must withhold the class/deviation: {result:#?}"
            );
        } else if i == 2 {
            assert!(
                matches!(result.class, ModulationClass::Fsk { .. }),
                "FSK at floor + 2 dB ({snr} dB) must assert the class/deviation: {result:#?}"
            );
        }
    }
}

// ---------------------------------------------------------------------
// False-positive budget (NFR-A3, merge gate): ≥1,000 out-of-scope trials.
// ---------------------------------------------------------------------

#[test]
fn false_positive_budget_on_noise_and_out_of_scope_signals() {
    const BANDWIDTH_HZ: f64 = 20_000.0;
    const OFFSET_HZ: f64 = 0.0;
    const DURATION_S: f64 = 0.02; // 4,000 samples at 200 kHz — short, per §10
    const TRIALS: usize = 1_000;

    let started = std::time::Instant::now();
    let mut confident_wrong = 0usize;

    for i in 0..TRIALS {
        let seed = 1000 + i as u64;
        let scene = match i % 4 {
            0 => {
                // Noise only.
                let mut s = SigGenConfig::new(RATE);
                s.seed = seed;
                s.noise = Some(NoiseConfig {
                    level_dbfs: NOISE_TOTAL_DBFS,
                });
                s
            }
            1 => {
                // A steady CW tone, well above the noise floor.
                let mut s = SigGenConfig::new(RATE);
                s.seed = seed;
                s.noise = Some(NoiseConfig {
                    level_dbfs: NOISE_TOTAL_DBFS,
                });
                s.tones.push(ToneConfig {
                    offset_hz: OFFSET_HZ,
                    level_dbfs: level_for_snr(20.0, NOISE_TOTAL_DBFS, BANDWIDTH_HZ, RATE),
                });
                s
            }
            2 => {
                // A chirp sweeping across (and beyond) the track's band.
                let mut s = SigGenConfig::new(RATE);
                s.seed = seed;
                s.noise = Some(NoiseConfig {
                    level_dbfs: NOISE_TOTAL_DBFS,
                });
                s.chirps.push(ChirpConfig {
                    start_offset_hz: -BANDWIDTH_HZ,
                    stop_offset_hz: BANDWIDTH_HZ,
                    sweep_time_s: 0.05,
                    level_dbfs: level_for_snr(20.0, NOISE_TOTAL_DBFS, BANDWIDTH_HZ, RATE),
                });
                s
            }
            _ => {
                // A hopper: this capture spans exactly one dwell, so it
                // looks like a steady tone from inside the track's window
                // — the realistic case for a track pinned to one hop.
                let mut s = SigGenConfig::new(RATE);
                s.seed = seed;
                s.noise = Some(NoiseConfig {
                    level_dbfs: NOISE_TOTAL_DBFS,
                });
                s.hoppers.push(HopperConfig {
                    offsets_hz: vec![OFFSET_HZ, 50_000.0],
                    dwell_s: DURATION_S * 2.0,
                    period_s: DURATION_S * 2.0,
                    level_dbfs: level_for_snr(20.0, NOISE_TOTAL_DBFS, BANDWIDTH_HZ, RATE),
                    order: HopOrder::Cycle,
                });
                s
            }
        };

        let (snapshot, band) = snapshot_of(&scene, DURATION_S);
        let t = track(OFFSET_HZ, BANDWIDTH_HZ, 20.0);
        let result = classify(&snapshot, &t, &band).unwrap();
        if result.class != ModulationClass::Unknown {
            confident_wrong += 1;
        }
    }

    let elapsed = started.elapsed();
    let rate = confident_wrong as f64 / TRIALS as f64;
    assert!(
        rate <= 0.01,
        "false-positive rate {rate:.4} ({confident_wrong}/{TRIALS}) exceeds the 1% NFR-A3 budget \
         (ran in {elapsed:?})"
    );
}

// ---------------------------------------------------------------------
// Determinism (NFR-A4)
// ---------------------------------------------------------------------

#[test]
fn classification_is_deterministic_for_a_fixed_seed() {
    const RATE_BD: f64 = 5_000.0;
    const BANDWIDTH_HZ: f64 = 20_000.0;
    const OFFSET_HZ: f64 = 25_000.0;
    let duration_s = 250.0 / RATE_BD;
    let scene = ook_scene(9, OFFSET_HZ, RATE_BD, 15.0, BANDWIDTH_HZ);
    let t = track(OFFSET_HZ, BANDWIDTH_HZ, 15.0);

    let (snap_a, band_a) = snapshot_of(&scene, duration_s);
    let (snap_b, band_b) = snapshot_of(&scene, duration_s);
    let a = classify(&snap_a, &t, &band_a).unwrap();
    let b = classify(&snap_b, &t, &band_b).unwrap();
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------
// A0 scope guard: an OFDM/chirp/linear-shaped input never guesses a class
// this lane does not own — it degrades to Unknown, per §7.9 step 6.
// ---------------------------------------------------------------------

#[test]
fn out_of_scope_signal_reports_unknown_not_a_guess() {
    const BANDWIDTH_HZ: f64 = 40_000.0;
    let mut scene = SigGenConfig::new(RATE);
    scene.seed = 5;
    scene.noise = Some(NoiseConfig {
        level_dbfs: NOISE_TOTAL_DBFS,
    });
    scene.chirps.push(ChirpConfig {
        start_offset_hz: -BANDWIDTH_HZ / 2.0,
        stop_offset_hz: BANDWIDTH_HZ / 2.0,
        sweep_time_s: 0.01,
        level_dbfs: level_for_snr(20.0, NOISE_TOTAL_DBFS, BANDWIDTH_HZ, RATE),
    });
    let (snapshot, band) = snapshot_of(&scene, 0.02);
    let t = track(0.0, BANDWIDTH_HZ, 20.0);
    let result = classify(&snapshot, &t, &band).unwrap();
    assert_eq!(result.class, ModulationClass::Unknown, "{result:#?}");
}

/// `Classification` derives `PartialEq`, needed above; this pins that it
/// stays comparable (a `Vec<String>` evidence field or a `Measured<f64>`
/// mismatch would otherwise silently make determinism checks vacuous).
#[test]
fn classification_equality_is_field_sensitive() {
    let a = Classification {
        class: ModulationClass::Unknown,
        symbol_rate_bd: None,
        confidence: 0.0,
        evidence: vec!["x".to_string()],
    };
    let mut b = a.clone();
    b.evidence.push("y".to_string());
    assert_ne!(a, b);
}
