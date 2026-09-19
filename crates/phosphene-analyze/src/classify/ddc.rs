// SPDX-License-Identifier: MIT

//! Digital down-conversion for the classification path (nextgen spec §7.6).
//!
//! Hand-rolled per §4.5 (no new dependencies): mix the track's carrier to DC
//! with an NCO, low-pass to (half of) the track's occupied bandwidth with a
//! windowed-sinc FIR, and decimate. [`mix_and_decimate`] is the first
//! (coarse) stage; [`decimate_further`] runs a second low-pass + decimate on
//! an already-basebanded signal once a rough symbol rate narrows the target
//! samples-per-symbol (§7.6: "coarse, then fine").
//!
//! All internal math runs in `f64` regardless of the `cf32` input/output
//! convention used elsewhere — this is an offline analysis path, not the
//! streaming hot path, and the extra precision costs nothing here.

use phosphene_core::Complex;

/// Complex baseband produced by a down-conversion stage, at its own rate.
#[derive(Debug, Clone)]
pub struct Baseband {
    /// Baseband samples, oldest first.
    pub samples: Vec<Complex<f64>>,
    /// Sample rate of `samples`, Hz.
    pub fs_hz: f64,
}

/// Smallest FIR low-pass, in taps (odd). Below this a windowed-sinc design
/// has essentially no stopband rejection.
const MIN_TAPS: usize = 31;
/// Largest FIR low-pass, in taps — bounds worst-case decimation cost for a
/// very narrow cutoff (§10: the false-positive suite must stay fast).
const MAX_TAPS: usize = 255;

/// Number of taps for a windowed-sinc low-pass at normalized cutoff
/// `cutoff_norm` (cutoff Hz / sample rate Hz): scales inversely with cutoff
/// (a narrower passband needs a longer filter for the same relative
/// transition width), clamped to a fixed, cheap range (*tune*).
fn taps_for_cutoff(cutoff_norm: f64) -> usize {
    let raw = (3.0 / cutoff_norm.max(1e-6)).round() as usize;
    let clamped = raw.clamp(MIN_TAPS, MAX_TAPS);
    clamped | 1 // force odd (Type I linear phase)
}

/// A windowed-sinc FIR low-pass with unity DC gain, Blackman-windowed for a
/// deep, well-behaved stopband (~74 dB) without the ripple of a plain
/// rectangular truncation.
fn design_lowpass(cutoff_norm: f64, ntaps: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    let m = (ntaps - 1) as f64;
    let mut taps = vec![0.0f64; ntaps];
    let mut sum = 0.0;
    for (n, tap) in taps.iter_mut().enumerate() {
        let k = n as f64 - m / 2.0;
        let sinc = if k == 0.0 {
            2.0 * cutoff_norm
        } else {
            (2.0 * PI * cutoff_norm * k).sin() / (PI * k)
        };
        let window =
            0.42 - 0.5 * (2.0 * PI * n as f64 / m).cos() + 0.08 * (4.0 * PI * n as f64 / m).cos();
        let v = sinc * window;
        *tap = v;
        sum += v;
    }
    for tap in &mut taps {
        *tap /= sum;
    }
    taps
}

/// `x[n]·e^{-j2π f_o n/f_s}` — the NCO mixer, `f64` phase accumulator
/// wrapped each sample (as the generator does) to avoid the precision loss
/// of computing `phase = n·inc` directly for large `n`.
fn nco_mix(iq: &[Complex<f32>], fs_hz: f64, offset_hz: f64) -> Vec<Complex<f64>> {
    use std::f64::consts::TAU;
    let inc = -TAU * offset_hz / fs_hz;
    let mut phase = 0.0f64;
    let mut out = Vec::with_capacity(iq.len());
    for &s in iq {
        let rot = Complex::new(phase.cos(), phase.sin());
        out.push(Complex::new(f64::from(s.re), f64::from(s.im)) * rot);
        phase += inc;
        if phase >= TAU {
            phase -= TAU;
        } else if phase < 0.0 {
            phase += TAU;
        }
    }
    out
}

/// Direct-form FIR filter, decimated: only the kept output samples are
/// computed (`O(len(x)/decim · ntaps)`, not a full convolution followed by
/// discarding). `taps[0]` is applied to the newest sample in each window.
fn fir_decimate(x: &[Complex<f64>], taps: &[f64], decim: usize) -> Vec<Complex<f64>> {
    let ntaps = taps.len();
    if x.len() < ntaps {
        return Vec::new();
    }
    let mut out = Vec::with_capacity((x.len() - ntaps) / decim + 1);
    let mut n = ntaps - 1;
    while n < x.len() {
        let mut acc = Complex::new(0.0, 0.0);
        for (j, &tap) in taps.iter().enumerate() {
            acc += x[n - j] * tap;
        }
        out.push(acc);
        n += decim;
    }
    out
}

/// Stage 1 (coarse): mix `iq` to the track's carrier, low-pass to
/// `cutoff_hz`, and decimate by `decim`. `cutoff_hz` is normally half the
/// track's occupied bandwidth (the baseband signal of interest occupies
/// `±cutoff_hz` around the new DC).
pub fn mix_and_decimate(
    iq: &[Complex<f32>],
    fs_hz: f64,
    offset_hz: f64,
    cutoff_hz: f64,
    decim: usize,
) -> Baseband {
    let mixed = nco_mix(iq, fs_hz, offset_hz);
    let decim = decim.max(1);
    let cutoff_norm = (cutoff_hz / fs_hz).clamp(1e-4, 0.49);
    let taps = design_lowpass(cutoff_norm, taps_for_cutoff(cutoff_norm));
    let samples = fir_decimate(&mixed, &taps, decim);
    Baseband {
        samples,
        fs_hz: fs_hz / decim as f64,
    }
}

/// Stage 2 (fine): a further low-pass + decimate on an already-basebanded
/// signal (no re-mixing — it is already at DC). `cutoff_hz` is normally the
/// same track bandwidth used for the coarse pass; `decim` narrows the rate
/// toward a target samples-per-symbol once a rough rate is known.
pub fn decimate_further(input: &Baseband, cutoff_hz: f64, decim: usize) -> Baseband {
    if decim <= 1 {
        return input.clone();
    }
    let cutoff_norm = (cutoff_hz / input.fs_hz).clamp(1e-4, 0.49);
    let taps = design_lowpass(cutoff_norm, taps_for_cutoff(cutoff_norm));
    let samples = fir_decimate(&input.samples, &taps, decim);
    Baseband {
        samples,
        fs_hz: input.fs_hz / decim as f64,
    }
}

/// Decimation factor bringing `fs_hz` down to approximately
/// `oversample · target_hz`, at least `1`.
pub fn pick_decimation(fs_hz: f64, target_hz: f64, oversample: f64) -> usize {
    if !target_hz.is_finite() || target_hz <= 0.0 {
        return 1;
    }
    (fs_hz / (oversample * target_hz)).floor().max(1.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowpass_has_unity_dc_gain() {
        let taps = design_lowpass(0.1, 63);
        let sum: f64 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "DC gain {sum}");
    }

    #[test]
    fn taps_scale_with_cutoff_and_stay_bounded() {
        assert_eq!(taps_for_cutoff(0.001).max(MIN_TAPS), MAX_TAPS);
        assert!(taps_for_cutoff(0.4) >= MIN_TAPS);
        for c in [0.001, 0.01, 0.1, 0.3, 0.49] {
            assert_eq!(taps_for_cutoff(c) % 2, 1, "must be odd for cutoff {c}");
        }
    }

    #[test]
    fn mixing_a_tone_to_dc_leaves_a_near_constant_phase() {
        // A pure tone at `offset_hz`, mixed to DC, should have (after the
        // filter settles) a phase that advances far more slowly than the
        // original tone — ideally not at all, up to residual filter effects.
        const FS: f64 = 1_000_000.0;
        const OFFSET: f64 = 100_000.0;
        let n = 4000;
        let iq: Vec<Complex<f32>> = (0..n)
            .map(|i| {
                let theta = std::f64::consts::TAU * OFFSET * i as f64 / FS;
                Complex::new(theta.cos() as f32, theta.sin() as f32)
            })
            .collect();
        let baseband = mix_and_decimate(&iq, FS, OFFSET, 50_000.0, 4);
        assert!(baseband.samples.len() > 100);
        // Instantaneous frequency near the tail (filter transient settled)
        // should be within a few hundred Hz of 0.
        let tail = &baseband.samples[baseband.samples.len() - 50..];
        for w in tail.windows(2) {
            let d = (w[1] * w[0].conj()).arg();
            let f_hz = d * baseband.fs_hz / std::f64::consts::TAU;
            assert!(f_hz.abs() < 500.0, "residual frequency {f_hz} Hz too high");
        }
    }

    #[test]
    fn decimate_further_halves_the_rate_for_decim_two() {
        const FS: f64 = 1_000_000.0;
        let iq: Vec<Complex<f32>> = (0..2000).map(|_| Complex::new(1.0, 0.0)).collect();
        let coarse = mix_and_decimate(&iq, FS, 0.0, 100_000.0, 2);
        let fine = decimate_further(&coarse, 50_000.0, 2);
        assert!((fine.fs_hz - coarse.fs_hz / 2.0).abs() < 1e-6);
    }

    #[test]
    fn out_of_band_tone_is_heavily_attenuated() {
        const FS: f64 = 200_000.0;
        const CUTOFF: f64 = 4000.0;
        let n = 20000;
        // In-band tone at 1 kHz, out-of-band tone at 30 kHz, equal amplitude
        // (each contributing 0.5 average power before filtering).
        let iq: Vec<Complex<f32>> = (0..n)
            .map(|i| {
                let t = i as f64 / FS;
                let a = (std::f64::consts::TAU * 1000.0 * t).cos();
                let b = (std::f64::consts::TAU * 30_000.0 * t).cos();
                Complex::new((a + b) as f32, 0.0)
            })
            .collect();
        let out = mix_and_decimate(&iq, FS, 0.0, CUTOFF, 6);
        let power: f64 =
            out.samples.iter().map(|s| s.norm_sqr()).sum::<f64>() / out.samples.len() as f64;
        // The out-of-band tone should be almost entirely rejected, leaving
        // close to the in-band tone's own 0.5 average power.
        assert!(
            (power - 0.5).abs() < 0.05,
            "average power {power}, expected ~0.5"
        );
    }

    #[test]
    fn pick_decimation_never_returns_zero() {
        assert_eq!(pick_decimation(1_000_000.0, 0.0, 4.0), 1);
        // A tiny target relative to the sample rate legitimately calls for
        // heavy decimation — this only checks it never rounds to zero.
        assert!(pick_decimation(1_000_000.0, 10.0, 4.0) >= 1);
        assert!(pick_decimation(2_000_000.0, 1_000.0, 4.0) > 1);
    }
}
