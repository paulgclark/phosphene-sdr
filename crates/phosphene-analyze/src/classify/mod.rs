// SPDX-License-Identifier: MIT

//! Offline modulation classification, phase A1 scope (nextgen spec §5.3,
//! §7.6–§7.9; FR-AC1, FR-AC3 restricted to OOK/ASK and FSK, FR-AC6). FR-AC2
//! (symbol rate) is delivered for OOK/ASK; FSK's own rate is not — see item
//! 5 below.
//!
//! [`classify`] is the crate's one public entry point for this stage: given
//! one track's raw-IQ snapshot (an [`AnalysisTap::snapshot`] capture), the
//! track's own kinematic state (carrier offset, bandwidth — already
//! measured), and the band metadata the snapshot's sample rate came from, it
//! runs the whole §7.6–§7.9 pipeline and returns a confidence-gated
//! [`Classification`]:
//!
//! 1. **[`ddc`]** — coarse-then-fine digital down-conversion to the track's
//!    carrier and bandwidth (§7.6): mix to DC, low-pass, decimate; a coarse
//!    envelope rate guess sizes a second, tighter decimation.
//! 2. **[`features`]** — the A1 feature subset from the fine baseband
//!    (§7.7): envelope mean/variance/normalized-variance, and the
//!    instantaneous-frequency histogram's modes and spacing.
//! 3. **[`symbol_rate`]** — the envelope-transition spectral-line estimator
//!    (§7.8), OOK/ASK's only consumer here — see item 5 below for why FSK's
//!    "baud line in the frequency-deviation signal" bonus is not wired up.
//! 4. **Confidence gating** (FR-AC6/NFR-A3, here in this module): every
//!    number is asserted only once a logistic gate on the track's own
//!    measured SNR — centered on a calibrated floor, switching within about
//!    2 dB of it — and (for the rate) an observed-symbol-count factor both
//!    clear a fixed threshold. Below it, the output degrades to the most
//!    specific defensible statement: a class without a rate, or
//!    [`ModulationClass::Unknown`] — carrying the gate's own sub-threshold
//!    confidence rather than a flat zero, so the gate's calibrated curve is
//!    visible on both sides of the floor, not just as an assert/withhold
//!    switch (see `unknown_with_confidence`).
//! 5. **The decision** (§7.9, A1 scope only): OOK/ASK, FSK, or Unknown.
//!    OFDM, chirp and "linear" are later-phase scope (A2) and are never
//!    guessed at here — an out-of-scope signal reports Unknown, honestly,
//!    rather than being forced into one of the two classes this lane knows.
//!    FSK's own class (`levels`, `deviation_hz`) is asserted the same way as
//!    OOK's rate; its `symbol_rate_bd` is deliberately always `None` — see
//!    the FSK branch of [`classify`] for the measurement that shows why a
//!    gate on it would not be honest.
//!
//! No live wiring: this module's only input is a snapshot the caller already
//! pulled (CL-3 connects it to the running app).

mod ddc;
mod features;
mod symbol_rate;

use crate::measure::Measured;
use crate::tap::{BandMeta, IqBuf};
use crate::track::{Classification, ModulationClass, Track};

/// Below this many raw input samples there is not enough data to attempt
/// even a coarse down-conversion.
const MIN_INPUT_SAMPLES: usize = 64;
/// Below this many coarse-baseband samples the coarse pass is too short to
/// say anything (*tune*).
const MIN_COARSE_SAMPLES: usize = 64;
/// A track bandwidth is clamped to at least this many Hz before sizing the
/// DDC filters — defends against a degenerate zero/negative measurement
/// rather than dividing by it.
const MIN_BANDWIDTH_HZ: f64 = 1.0;
/// Lower search bound for any spectral-line estimate: DC/drift dominates
/// below this and is never a symbol rate (*tune*).
const MIN_SYMBOL_RATE_HZ: f64 = 1.0;

/// Coarse-pass oversampling relative to the track's bandwidth: `fs1 ≈
/// oversample · bandwidth` (*tune* — generous margin over the Nyquist
/// minimum of `1×` for filter roll-off).
const COARSE_OVERSAMPLE: f64 = 4.0;
/// Target samples-per-symbol for the fine pass, once a coarse rate is known
/// (*tune*).
const FINE_SAMPLES_PER_SYMBOL: f64 = 8.0;
/// A coarse rate guess below this raw prominence is not a real detection —
/// never used to size the fine pass (*tune*: FSK's constant envelope has no
/// on/off transitions, so this is what keeps its coarse pass a no-op rather
/// than chasing noise into an arbitrary further decimation).
const COARSE_RATE_PROMINENCE_MIN: f64 = 0.3;

/// Width, in dB, of the logistic SNR gate's transition — chosen so
/// confidence crosses from ~12% to ~88% within about 2 dB of the calibrated
/// floor (D-102 item 9: "gating switches within 2 dB of its calibrated
/// floor").
const SNR_GATE_STEEPNESS_DB: f64 = 1.0;
/// Calibrated SNR floor for OOK's symbol-rate estimate (D-102 item 9:
/// "symbol rate within 1% at 12 dB SNR or more"). FSK does not assert a
/// rate — see the FSK branch of [`classify`] — so this applies to OOK only,
/// despite the D-102 wording naming both.
const SYMBOL_RATE_SNR_FLOOR_DB: f64 = 12.0;
/// Calibrated SNR floor for the FSK deviation estimate (D-102 item 9: "FSK
/// ... deviation within 5% at 10 dB or more").
const FSK_DEVIATION_SNR_FLOOR_DB: f64 = 10.0;
/// `count_factor` reaches 0.5 at this many observed symbols (*tune*). D-102
/// item 9 states the floor "with at least 100 symbols observed"; measured
/// against the generator (12 dB, 30 seeds per point), the rate is asserted
/// 0/30 times at exactly 100 symbols — never wrongly, only never at all,
/// since [`OOK_LINE_PROMINENCE_MIN`]'s qualitative gate (calibrated for the
/// false-positive budget, NFR-A3) is the actual bottleneck there, not this
/// factor. Reliable assertion (~29/30) begins around 250 symbols. This is a
/// measured relaxation of "100" to "~250", reported here rather than
/// silently changed; `ook_symbol_rate_never_wrong_at_the_100_symbol_floor`
/// (seal_tests) pins the invariant that actually matters: 100 symbols never
/// produces a confidently wrong number, even though it usually produces none.
const SYMBOL_COUNT_HALF: f64 = 20.0;
/// A number (or class) is asserted only once its combined confidence
/// reaches this (*tune*).
const ASSERT_THRESHOLD: f64 = 0.4;

/// Normalized envelope variance below this counts as "constant envelope"
/// (§7.9 step 4's FSK precondition) (*tune*: additive noise alone gives a
/// constant-envelope signal a normalized envelope variance of order
/// `1/SNR_linear` — about 0.1 at the 10 dB FSK floor — so this must clear
/// that noise floor, not just distinguish from zero).
const ENVELOPE_CONST_VAR_THRESHOLD: f64 = 0.14;
/// Normalized envelope variance above this counts as "clear on-off" (§7.9
/// step 3's OOK/ASK precondition) (*tune*).
const OOK_VAR_THRESHOLD: f64 = 0.25;
/// An envelope-transition spectral line below this raw prominence is not
/// confirmed enough to call the signal keyed at all — the qualitative gate
/// that keeps pure noise from tripping the OOK branch. Noise alone clears a
/// much lower bar than this: filtered noise's own threshold-crossings have
/// a "characteristic rate" tied to the filter bandwidth and commonly show
/// 0.35–0.5 raw prominence (measured against the generator's noise-only
/// scene), which real 12 dB+ OOK keying clears with room to spare — this
/// sits above that noise band (*tune*, false-positive budget NFR-A3).
const OOK_LINE_PROMINENCE_MIN: f64 = 0.55;

/// Classify one track from a raw-IQ snapshot (nextgen spec §5.3, A1 scope).
///
/// `snapshot` is a capture from [`crate::tap::AnalysisTap::snapshot`] of the
/// band `band` describes; `track` supplies the carrier offset and bandwidth
/// to down-convert to, and the SNR every confidence gate below is measured
/// against (already computed by the measure stage). Returns `None` only when
/// there is not enough input to attempt classification at all (too few
/// samples, or a degenerate band/track); otherwise always returns
/// `Some(Classification)`, including [`ModulationClass::Unknown`] when no
/// class clears its evidence threshold — measurements elsewhere still stand,
/// per §7.9 step 6.
pub fn classify(snapshot: &IqBuf, track: &Track, band: &BandMeta) -> Option<Classification> {
    let fs_hz = band.span_hz;
    if !fs_hz.is_finite() || fs_hz <= 0.0 || snapshot.samples.len() < MIN_INPUT_SAMPLES {
        return None;
    }
    let offset_hz = track.center_offset_hz;
    let bandwidth_hz = track.bandwidth_hz.max(MIN_BANDWIDTH_HZ);
    let snr_db = f64::from(track.snr_db);
    // A windowed-sinc low-pass is not flat all the way to its nominal
    // cutoff — its transition band starts rolling off partway through the
    // passband — so a signal component right at the edge of the stated
    // bandwidth (e.g. a dense M-ary FSK ladder's outermost tones) is
    // measurably attenuated relative to one near DC. The track's reported
    // bandwidth is the contract this stage filters to; a ladder wide enough
    // to reach the edge of it needs the caller to report a bandwidth with
    // some headroom past the outermost tone; this stage does not invent one.
    let cutoff_hz = bandwidth_hz / 2.0;

    // §7.6 — coarse-then-fine DDC.
    let d1 = ddc::pick_decimation(fs_hz, bandwidth_hz, COARSE_OVERSAMPLE);
    let coarse = ddc::mix_and_decimate(&snapshot.samples, fs_hz, offset_hz, cutoff_hz, d1);
    if coarse.samples.len() < MIN_COARSE_SAMPLES {
        return None;
    }
    let coarse_envelope: Vec<f64> = coarse.samples.iter().map(|c| c.norm()).collect();
    let coarse_edges =
        symbol_rate::transition_indicator(&symbol_rate::threshold_envelope(&coarse_envelope));
    let coarse_rate = symbol_rate::estimate_line(
        &coarse_edges,
        coarse.fs_hz,
        MIN_SYMBOL_RATE_HZ,
        coarse.fs_hz / 2.0,
    );
    // A coarse guess only sizes the fine pass when it is a real detection
    // (FSK's constant envelope has no on/off transitions to guess from at
    // all, and a near-zero-prominence "line" there is noise, not a rate —
    // trusting it would pick an arbitrary, sometimes wildly excessive,
    // further decimation).
    let fine = match coarse_rate {
        Some(r) if r.freq_hz > 0.0 && r.prominence >= COARSE_RATE_PROMINENCE_MIN => {
            let d2 = ddc::pick_decimation(coarse.fs_hz, r.freq_hz, FINE_SAMPLES_PER_SYMBOL);
            ddc::decimate_further(&coarse, cutoff_hz, d2)
        }
        _ => coarse,
    };
    let Some(features) = features::compute(&fine.samples, fine.fs_hz) else {
        return Some(unknown(vec![
            "too little baseband data after down-conversion".to_string(),
        ]));
    };

    let mut evidence = vec![format!(
        "envelope mean {:.4}, variance {:.6}, normalized variance {:.3}",
        features.envelope_mean, features.envelope_var, features.envelope_norm_var
    )];

    // §7.9 step 3 — OOK/ASK: a clear on-off envelope, confirmed by a real
    // spectral line (never variance alone — see OOK_LINE_PROMINENCE_MIN).
    if features.envelope_norm_var > OOK_VAR_THRESHOLD {
        // §7.8: a spectral line in the envelope's *transitions*, not its raw
        // value — see the symbol_rate module docs for why the raw on/off
        // sequence itself carries no such line, transition timing does.
        let envelope: Vec<f64> = fine.samples.iter().map(|c| c.norm()).collect();
        let edges = symbol_rate::transition_indicator(&symbol_rate::threshold_envelope(&envelope));
        let line =
            symbol_rate::estimate_line(&edges, fine.fs_hz, MIN_SYMBOL_RATE_HZ, fine.fs_hz / 2.0);
        if let Some(line) = line.filter(|l| l.prominence >= OOK_LINE_PROMINENCE_MIN) {
            evidence.push(format!(
                "envelope-transition spectral line at {:.1} Hz, prominence {:.2} (on-off keying confirmed)",
                line.freq_hz, line.prominence
            ));
            let class_confidence = snr_gate(snr_db, SYMBOL_RATE_SNR_FLOOR_DB);
            if class_confidence >= ASSERT_THRESHOLD {
                let observed_symbols = fine.samples.len() as f64 * line.freq_hz / fine.fs_hz;
                let rate_confidence =
                    class_confidence * count_factor(observed_symbols, SYMBOL_COUNT_HALF);
                let symbol_rate_bd = (rate_confidence >= ASSERT_THRESHOLD)
                    .then(|| Measured::new(line.freq_hz, rate_confidence as f32));
                if symbol_rate_bd.is_none() {
                    evidence.push(format!(
                        "symbol rate not asserted: confidence {rate_confidence:.2} \
                         (SNR {snr_db:.1} dB, ~{observed_symbols:.0} symbols observed)"
                    ));
                }
                return Some(Classification {
                    class: ModulationClass::OokAsk,
                    symbol_rate_bd,
                    confidence: class_confidence as f32,
                    evidence,
                });
            }
            evidence.push(format!(
                "OOK/ASK not asserted: class confidence {class_confidence:.2} (SNR {snr_db:.1} dB)"
            ));
            return Some(unknown_with_confidence(class_confidence as f32, evidence));
        } else {
            evidence.push(
                "on-off envelope present but no confirmed spectral line — not asserted as OOK/ASK"
                    .to_string(),
            );
        }
    } else if features.envelope_norm_var < ENVELOPE_CONST_VAR_THRESHOLD && features.modes.len() >= 2
    {
        // §7.9 step 4 — FSK: constant envelope, a multi-modal IF histogram
        // with consistent (evenly spaced) mode spacing.
        if let Some(spacing_hz) = features.mode_spacing_hz {
            let levels = features.modes.len() as u32;
            evidence.push(format!(
                "instantaneous-frequency histogram: {levels} modes, spacing {spacing_hz:.1} Hz"
            ));
            let dev_confidence = snr_gate(snr_db, FSK_DEVIATION_SNR_FLOOR_DB);
            if dev_confidence >= ASSERT_THRESHOLD {
                // FSK's own symbol rate (§7.8 "FSK bonus": the baud line in
                // the frequency-deviation signal, quantized to the found
                // modes and read via the same transition-indicator technique
                // as OOK) is deliberately NOT asserted here. Measured against
                // the generator at the seal's own operating points (12 dB,
                // M = 2 and 4, 100–300 symbols), the quantized-transition
                // estimate is unreliable: it is frequently and substantially
                // wrong (tens of percent, not a rounding error) rather than
                // simply noisy near the true value, so no confidence gate on
                // it would be honest — a gate that passes wrong answers
                // through is worse than one that never opens. Per NFR-A3,
                // "class + deviation, no rate" is the specific defensible
                // statement here; a working FSK rate estimator is future
                // work, not a threshold to retune.
                return Some(Classification {
                    class: ModulationClass::Fsk {
                        levels,
                        deviation_hz: Measured::new(spacing_hz, dev_confidence as f32),
                    },
                    symbol_rate_bd: None,
                    confidence: dev_confidence as f32,
                    evidence,
                });
            }
            evidence.push(format!(
                "frequency deviation not asserted: confidence {dev_confidence:.2} (SNR {snr_db:.1} dB)"
            ));
            return Some(unknown_with_confidence(dev_confidence as f32, evidence));
        }
    }

    Some(unknown(evidence))
}

/// `Unknown` with no meaningful confidence to report — nothing SNR-gated
/// was even computed (e.g. no evidence at all cleared the qualitative gates
/// upstream of any [`snr_gate`] call).
fn unknown(evidence: Vec<String>) -> Classification {
    unknown_with_confidence(0.0, evidence)
}

/// `Unknown`, but carrying the sub-threshold confidence a [`snr_gate`] call
/// actually computed before falling short of [`ASSERT_THRESHOLD`] — this is
/// what lets a caller (and the seal tests) see that the gate is doing real,
/// continuous work near its calibrated floor, not just flipping a switch.
fn unknown_with_confidence(confidence: f32, mut evidence: Vec<String>) -> Classification {
    evidence.push("no class met its evidence threshold".to_string());
    Classification {
        class: ModulationClass::Unknown,
        symbol_rate_bd: None,
        confidence,
        evidence,
    }
}

/// Logistic confidence gate on `snr_db` centered at `floor_db`: `0.5` at the
/// floor, rising toward `1.0` above it and falling toward `0.0` below,
/// crossing roughly `0.12..0.88` within [`SNR_GATE_STEEPNESS_DB`] · 2 of the
/// floor (D-102 item 9's "switches within 2 dB").
fn snr_gate(snr_db: f64, floor_db: f64) -> f64 {
    1.0 / (1.0 + (-(snr_db - floor_db) / SNR_GATE_STEEPNESS_DB).exp())
}

/// Confidence factor rising with an observation count: `n / (n + half)`,
/// reaching 0.5 at `half` observations (same shape as
/// `measure::psd::count_factor`, kept local rather than shared to avoid
/// coupling this module to measure's internals for one line of arithmetic).
fn count_factor(n: f64, half: f64) -> f64 {
    (n / (n + half)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod seal_tests;
