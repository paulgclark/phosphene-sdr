// SPDX-License-Identifier: MIT

//! The envelope symbol-rate estimator (nextgen spec §7.8), OOK/ASK only —
//! see the crate-level classify docs for why FSK's "baud line in the
//! frequency-deviation signal" bonus is not wired up.
//!
//! Two primitives:
//!
//! 1. [`transition_indicator`] turns a keyed signal into a sparse 0/1 pulse
//!    train marking where its *label* changed — [`threshold_envelope`]
//!    labels OOK/ASK's on/off state. A label can only change at a symbol
//!    boundary, so every pulse lands on a multiple of the (unknown) symbol
//!    period — this is what turns "find the symbol rate" into "find a real
//!    spectral line", which the raw keyed values themselves do not have: a
//!    linearly filtered sequence of i.i.d. symbols has a *continuous*
//!    spectrum with no discrete tone at its own rate (a textbook
//!    random-telegraph result; public DSP theory, not anything LC-1
//!    restricts). Transition timing does not share that limitation, in real
//!    bandlimited signals as much as idealized ones — it is squarely a §7.8
//!    "spectral line in the envelope" reading, just of the envelope's
//!    *transitions* rather than its raw value, and this is what actually
//!    meets the seal's ≥12 dB / ≥100-symbol tolerance (verified empirically
//!    against the M0 generator, per spec §7.0's calibration methodology — a
//!    plain envelope/envelope² line needed closer to 10,000 symbols to
//!    separate from a random bit sequence's own low-frequency-heavy
//!    realization noise).
//! 2. [`estimate_line`] finds the strongest harmonically-validated spectral
//!    line in any such real-valued signal, refined by parabolic
//!    interpolation across the peak's three bins.
//!
//! The FFT itself reuses [`phosphene_core::SpectrumAnalyzer`] — the crate's
//! one FFT pipeline (§4.5: no new dependencies) — applied to a real signal
//! (imaginary part zero) rather than the complex baseband it was built for;
//! only the magnitude spectrum is used, so the pipeline's own dBFS
//! calibration is irrelevant here, just its correctness.

use phosphene_core::{SpectrumAnalyzer, WindowKind, MAX_FFT_SIZE, MIN_FFT_SIZE};

/// One validated spectral-line estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineEstimate {
    /// The fundamental frequency, Hz.
    pub freq_hz: f64,
    /// Raw evidence strength in `0.0..=1.0`: how far the peak bin's power
    /// stands above the spectrum's median (a proxy for the noise floor),
    /// saturating — *not* yet combined with SNR or observation-count
    /// factors (that is [`crate::classify`]'s job, against the track's own
    /// measured SNR).
    pub prominence: f64,
}

/// Divisors checked when validating a peak against its possible harmonics
/// (*tune*: covers the low-order harmonics real transition timing produces
/// — e.g. a duty cycle away from 50% strengthens the second harmonic).
const HARMONIC_DIVISORS: [u32; 4] = [2, 3, 4, 5];
/// A sub-harmonic candidate must retain at least this fraction of the
/// original peak's linear power to be preferred as the true fundamental
/// (*tune*).
const HARMONIC_POWER_FRACTION: f64 = 0.35;
/// Bins nearest DC excluded from the search — mean-removal is never exact,
/// and drift/filter transients otherwise dominate the low end (*tune*).
const DC_GUARD_BINS: usize = 3;

/// A 0/1 "transition indicator": `1.0` wherever consecutive labels differ,
/// `0.0` where they repeat. See the module docs for why this — not the raw
/// labels — is what carries a genuine spectral line at the symbol rate.
pub fn transition_indicator(labels: &[f64]) -> Vec<f64> {
    labels.windows(2).map(|w| f64::from(w[0] != w[1])).collect()
}

/// Threshold a real-valued envelope at half its peak, giving a clean 0/1
/// on-off label sequence (§7.8, OOK/ASK).
pub fn threshold_envelope(envelope: &[f64]) -> Vec<f64> {
    let peak = envelope.iter().cloned().fold(0.0, f64::max);
    let threshold = peak * 0.5;
    envelope.iter().map(|&e| f64::from(e > threshold)).collect()
}

/// Find the strongest harmonically-validated spectral line of `signal`
/// (sampled at `fs_hz`) in `(min_hz, max_hz)`. `None` if there are too few
/// samples or no candidate line clears the DC guard band.
pub fn estimate_line(signal: &[f64], fs_hz: f64, min_hz: f64, max_hz: f64) -> Option<LineEstimate> {
    if signal.len() < 8
        || !min_hz.is_finite()
        || !max_hz.is_finite()
        || max_hz <= min_hz
        || min_hz < 0.0
    {
        return None;
    }

    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let centered: Vec<f64> = signal.iter().map(|&v| v - mean).collect();

    let n_fft = fft_size_for(centered.len());
    let dbfs = windowed_power_spectrum(&centered, n_fft);
    let bin_hz = fs_hz / n_fft as f64;

    // Positive-frequency half only (DC at n_fft/2 in the fftshifted output).
    let dc = n_fft / 2;
    let lo_bin = (dc + DC_GUARD_BINS).max(dc + (min_hz / bin_hz).round() as usize);
    let hi_bin = (dc + (max_hz / bin_hz).round() as usize).min(n_fft - 1);
    if lo_bin >= hi_bin {
        return None;
    }

    let (peak_bin, _) = argmax(&dbfs[lo_bin..=hi_bin]);
    let peak_bin = lo_bin + peak_bin;
    let peak_freq = (peak_bin - dc) as f64 * bin_hz;
    let peak_power = db_to_lin(dbfs[peak_bin]);

    // Harmonic validation: prefer the lowest-order sub-harmonic that is
    // itself a comparably strong local line.
    let mut fundamental_bin = peak_bin;
    let mut fundamental_freq = peak_freq;
    for &d in HARMONIC_DIVISORS.iter().rev() {
        let candidate_freq = peak_freq / d as f64;
        if candidate_freq < min_hz {
            continue;
        }
        let candidate_bin = dc + (candidate_freq / bin_hz).round() as usize;
        if candidate_bin <= dc + DC_GUARD_BINS || candidate_bin >= n_fft - 1 {
            continue;
        }
        let candidate_power = db_to_lin(dbfs[candidate_bin]);
        let is_local_peak = dbfs[candidate_bin] >= dbfs[candidate_bin - 1]
            && dbfs[candidate_bin] >= dbfs[candidate_bin + 1];
        if is_local_peak && candidate_power >= peak_power * HARMONIC_POWER_FRACTION {
            fundamental_bin = candidate_bin;
            fundamental_freq = candidate_freq;
            break;
        }
    }

    // Parabolic interpolation (log-power domain) across the chosen peak's
    // three bins for sub-bin precision.
    let refined_freq = if fundamental_bin > 0 && fundamental_bin + 1 < n_fft {
        let (a, b, c) = (
            f64::from(dbfs[fundamental_bin - 1]),
            f64::from(dbfs[fundamental_bin]),
            f64::from(dbfs[fundamental_bin + 1]),
        );
        let denom = a - 2.0 * b + c;
        let delta = if denom.abs() > 1e-9 {
            0.5 * (a - c) / denom
        } else {
            0.0
        };
        fundamental_freq + delta.clamp(-1.0, 1.0) * bin_hz
    } else {
        fundamental_freq
    };

    // Prominence: peak power over the spectrum's median (over the searched
    // band), saturating via the same shape used elsewhere for SNR-style
    // factors.
    let mut sorted: Vec<f32> = dbfs[lo_bin..=hi_bin].to_vec();
    sorted.sort_by(f32::total_cmp);
    let median_db = f64::from(sorted[sorted.len() / 2]);
    let prom_db = f64::from(dbfs[fundamental_bin]) - median_db;
    let prominence = (prom_db / (prom_db + 12.0)).clamp(0.0, 1.0);

    Some(LineEstimate {
        freq_hz: refined_freq.max(0.0),
        prominence,
    })
}

/// The largest power-of-two FFT size that fits within `len` samples
/// (truncating any remainder), clamped to `[MIN_FFT_SIZE, MAX_FFT_SIZE]` —
/// a short input is zero-padded up to the minimum instead of truncated.
fn fft_size_for(len: usize) -> usize {
    if len <= MIN_FFT_SIZE {
        MIN_FFT_SIZE
    } else {
        largest_pow2_at_most(len).min(MAX_FFT_SIZE)
    }
}

/// Largest power of two ≤ `n` (or `1` if `n == 0`).
fn largest_pow2_at_most(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        1usize << (usize::BITS - 1 - n.leading_zeros())
    }
}

/// Hann-window the first `data.len().min(n_fft)` samples (windowing the
/// *real* segment, never the zero-padding — an asymmetric partial window
/// would bias the spectrum), zero-pad to `n_fft`, and return the DC-centered
/// linear power spectrum.
fn windowed_power_spectrum(data: &[f64], n_fft: usize) -> Vec<f32> {
    let take = data.len().min(n_fft);
    // The most recent `take` samples: symbol-rate content is stationary
    // through the capture, so any contiguous window works equally.
    let start = data.len() - take;
    let window = phosphene_core::Window::new(WindowKind::Hann, take);
    let mut frame = vec![phosphene_core::Complex::new(0.0f32, 0.0); n_fft];
    for (i, (&d, &w)) in data[start..].iter().zip(window.coefficients()).enumerate() {
        frame[i] = phosphene_core::Complex::new((d * f64::from(w)) as f32, 0.0);
    }
    let mut analyzer = SpectrumAnalyzer::new(n_fft, WindowKind::Rectangular)
        .expect("fft_size_for always returns a valid power-of-two size");
    let mut out = vec![0.0f32; n_fft];
    analyzer.process(&frame, &mut out);
    out
}

fn argmax(xs: &[f32]) -> (usize, f32) {
    let mut best_i = 0;
    let mut best_v = xs[0];
    for (i, &v) in xs.iter().enumerate().skip(1) {
        if v > best_v {
            best_i = i;
            best_v = v;
        }
    }
    (best_i, best_v)
}

fn db_to_lin(db: f32) -> f64 {
    10f64.powf(f64::from(db) / 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq_hz: f64, fs_hz: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (std::f64::consts::TAU * freq_hz * i as f64 / fs_hz).sin())
            .collect()
    }

    #[test]
    fn finds_a_clean_tone_within_a_bin() {
        const FS: f64 = 100_000.0;
        const F: f64 = 5_000.0;
        let signal = sine(F, FS, 4096);
        let est = estimate_line(&signal, FS, 100.0, FS / 2.0).unwrap();
        assert!(
            (est.freq_hz - F).abs() < FS / 4096.0,
            "got {} Hz",
            est.freq_hz
        );
        assert!(est.prominence > 0.5);
    }

    #[test]
    fn prefers_the_fundamental_over_a_stronger_second_harmonic() {
        const FS: f64 = 100_000.0;
        const F0: f64 = 1_000.0;
        let n = 8192;
        let fundamental = sine(F0, FS, n);
        let harmonic = sine(2.0 * F0, FS, n);
        // Harmonic component is a bit stronger than the fundamental, as it
        // realistically can be, but the fundamental is still present and a
        // real local peak.
        let signal: Vec<f64> = fundamental
            .iter()
            .zip(&harmonic)
            .map(|(&a, &b)| a + 1.3 * b)
            .collect();
        let est = estimate_line(&signal, FS, 50.0, FS / 2.0).unwrap();
        assert!(
            (est.freq_hz - F0).abs() < FS / n as f64 * 2.0,
            "got {} Hz, wanted {F0} Hz",
            est.freq_hz
        );
    }

    #[test]
    fn no_signal_returns_none() {
        assert!(estimate_line(&[0.0; 4], 1000.0, 1.0, 100.0).is_none());
    }

    #[test]
    fn pure_noise_has_low_prominence() {
        // A deterministic LCG stands in for true randomness here (no new
        // dependency, and reproducible): the point is a signal with no
        // dominant line at all, unlike a sum of fixed tones (which always
        // has a strongest one).
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        let signal: Vec<f64> = (0..4096).map(|_| next()).collect();
        let est = estimate_line(&signal, 100_000.0, 100.0, 50_000.0).unwrap();
        assert!(est.prominence < 0.5, "prominence {}", est.prominence);
    }

    #[test]
    fn transition_indicator_marks_only_changes() {
        let labels = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        assert_eq!(transition_indicator(&labels), vec![0.0, 1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn threshold_envelope_splits_at_half_peak() {
        let envelope = [0.0, 0.4, 0.6, 1.0, 0.1];
        assert_eq!(threshold_envelope(&envelope), vec![0.0, 0.0, 1.0, 1.0, 0.0]);
    }
}
