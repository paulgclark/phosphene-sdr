// SPDX-License-Identifier: MIT

//! The A1 feature subset (nextgen spec §7.7): envelope mean/variance and
//! normalized envelope variance, and the instantaneous-frequency histogram
//! with its modality and mode spacing. Computed from a down-converted
//! baseband ([`crate::classify::ddc::Baseband`]).
//!
//! Instantaneous frequency is only meaningful while the carrier is actually
//! present: an OOK "off" symbol is near-zero envelope, and the phase of a
//! near-zero complex sample is dominated by whatever noise put it there.
//! [`compute`] excludes such samples from the IF histogram (but not from the
//! envelope statistics, where the on/off contrast is exactly the OOK
//! evidence).

use phosphene_core::Complex;

/// The features one down-converted snippet yields.
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Mean envelope `|z|` over every sample.
    pub envelope_mean: f64,
    /// Variance of `|z|` over every sample.
    pub envelope_var: f64,
    /// `envelope_var / envelope_mean²` — scale-invariant, so the same
    /// on/off signal reads the same regardless of its level.
    pub envelope_norm_var: f64,
    /// Distinct instantaneous-frequency modes, ascending (from the
    /// instantaneous-frequency histogram at each sample transition where
    /// both samples cleared the presence gate).
    pub modes: Vec<f64>,
    /// Mean spacing between adjacent modes, if `modes.len() >= 2` and the
    /// spacings are consistent (evenly spaced — the FSK ladder signature).
    pub mode_spacing_hz: Option<f64>,
}

/// Presence gate as a fraction of the peak envelope: samples below this are
/// excluded from the IF histogram (*tune*).
const PRESENCE_FRACTION: f64 = 0.5;
/// Histogram bins spanning the observed (gated) IF range (*tune*: enough
/// resolution to separate FSK levels without being noise-sensitive).
const HIST_BINS: usize = 128;
/// A histogram bin counts as a candidate mode only above this fraction of
/// the tallest bin (*tune*).
const PEAK_FRACTION: f64 = 0.2;
/// Smoothing window (bins) applied before peak-picking, as a fraction of
/// [`HIST_BINS`] (*tune*): a real FSK tone's IF estimate is noisy/jittery
/// (filter transients, SNR) and spreads over many adjacent bins, which
/// without smoothing shows up as several small local maxima instead of one
/// broad hump.
const SMOOTH_FRACTION: f64 = 1.0 / 24.0;
/// Two candidate peaks closer than this fraction of the histogram's total
/// span are treated as one mode (*tune*): a fraction of the *span*, not a
/// fixed bin count, so it scales with however wide the observed jitter
/// turns out to be.
const MIN_PEAK_SEPARATION_FRACTION: f64 = 1.0 / 10.0;
/// Two accepted peaks only count as distinct modes if the smoothed
/// histogram dips, somewhere between them, to at most this fraction of the
/// shorter peak (*tune*): rejects a smoothly-varying (chirp-like) IF trace
/// whose "peaks" are just noise on an otherwise near-flat histogram.
const VALLEY_TO_PEAK_MAX: f64 = 0.7;
/// Mode spacings are "consistent" (an evenly spaced ladder) when every
/// spacing is within this fraction of the mean spacing (*tune*).
const SPACING_TOLERANCE: f64 = 0.25;
/// Fraction of the (seed-estimated) half-gap to a neighboring mode,
/// centered on the boundary between them, excluded from either mode's
/// refinement average (*tune*): trims the filter-smeared transition samples
/// a hard nearest/equal-count split would otherwise redistribute, biasing
/// the estimated separation down.
const BOUNDARY_TRIM_FRACTION: f64 = 0.9;
/// Below this Hz spread, the instantaneous-frequency samples are treated as
/// numerically constant rather than histogrammed (guards against
/// floating-point rounding scattering an exact single value across bins).
const MIN_RANGE_HZ: f64 = 1.0;

/// Compute the feature vector for `baseband`, sampled at `fs_hz`. `None` if
/// there are too few samples to say anything (fewer than 8).
pub fn compute(baseband: &[Complex<f64>], fs_hz: f64) -> Option<FeatureVector> {
    if baseband.len() < 8 {
        return None;
    }

    let envelopes: Vec<f64> = baseband.iter().map(|c| c.norm()).collect();
    let n = envelopes.len() as f64;
    let envelope_mean = envelopes.iter().sum::<f64>() / n;
    let envelope_var = envelopes
        .iter()
        .map(|&e| (e - envelope_mean).powi(2))
        .sum::<f64>()
        / n;
    let envelope_norm_var = if envelope_mean > 0.0 {
        envelope_var / (envelope_mean * envelope_mean)
    } else {
        0.0
    };

    let peak = envelopes.iter().cloned().fold(0.0f64, f64::max);
    let gate = peak * PRESENCE_FRACTION;

    let mut if_hz = Vec::with_capacity(baseband.len());
    for w in baseband.windows(2) {
        if w[0].norm() >= gate && w[1].norm() >= gate {
            let d = (w[1] * w[0].conj()).arg();
            if_hz.push(d * fs_hz / std::f64::consts::TAU);
        }
    }

    let (modes, mode_spacing_hz) = find_modes(&if_hz);

    Some(FeatureVector {
        envelope_mean,
        envelope_var,
        envelope_norm_var,
        modes,
        mode_spacing_hz,
    })
}

/// Greedy peak extraction over a histogram of `if_hz`.
fn find_modes(if_hz: &[f64]) -> (Vec<f64>, Option<f64>) {
    if if_hz.len() < 8 {
        return (Vec::new(), None);
    }
    let lo = if_hz.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = if_hz.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // A near-constant value (floating-point noise aside) is exactly one
    // mode — below this the histogram below would scatter numerically
    // identical values across spurious bins purely from float rounding.
    if hi - lo <= MIN_RANGE_HZ {
        let mean = if_hz.iter().sum::<f64>() / if_hz.len() as f64;
        return (vec![mean], None);
    }
    let bin_width = (hi - lo) / HIST_BINS as f64;
    let mut hist = vec![0usize; HIST_BINS];
    for &v in if_hz {
        let idx = (((v - lo) / bin_width) as usize).min(HIST_BINS - 1);
        hist[idx] += 1;
    }

    // Smooth with a box filter before peak-picking: a real tone's jitter
    // spreads over many bins, and without smoothing that shows up as
    // several small local maxima rather than one broad hump.
    let radius = ((HIST_BINS as f64 * SMOOTH_FRACTION) as usize / 2).max(1);
    let smoothed: Vec<f64> = (0..HIST_BINS)
        .map(|i| {
            let lo_j = i.saturating_sub(radius);
            let hi_j = (i + radius).min(HIST_BINS - 1);
            hist[lo_j..=hi_j].iter().sum::<usize>() as f64 / (hi_j - lo_j + 1) as f64
        })
        .collect();

    let peak_count = smoothed.iter().cloned().fold(0.0, f64::max);
    if peak_count <= 0.0 {
        return (Vec::new(), None);
    }
    let threshold = peak_count * PEAK_FRACTION;
    let min_separation_bins = ((HIST_BINS as f64 * MIN_PEAK_SEPARATION_FRACTION) as usize).max(1);

    // Local maxima of the smoothed histogram above threshold, merging
    // anything within `min_separation_bins` of a taller neighbor already
    // accepted.
    let mut candidates: Vec<usize> = (0..HIST_BINS)
        .filter(|&i| {
            let c = smoothed[i];
            if c < threshold {
                return false;
            }
            let left = if i == 0 { c } else { smoothed[i - 1] };
            let right = if i + 1 == HIST_BINS {
                c
            } else {
                smoothed[i + 1]
            };
            c >= left && c >= right
        })
        .collect();
    candidates.sort_by(|&a, &b| smoothed[b].total_cmp(&smoothed[a]));

    let mut accepted: Vec<usize> = Vec::new();
    for c in candidates {
        if accepted
            .iter()
            .all(|&a| a.abs_diff(c) >= min_separation_bins)
        {
            accepted.push(c);
        }
    }
    accepted.sort_unstable();

    // Reject peaks with no real valley between them: a continuously
    // varying instantaneous frequency (a chirp, or "linear" modulation's
    // undetermined-order symbols) spreads roughly uniformly across its
    // range and can clear the bin-count threshold at several bins without
    // ever dipping between them — unlike genuinely discrete FSK tones,
    // which are separated by bins with very little mass at all. Out of
    // scope for A1 (§7.10), so this must not read as multi-modal.
    if accepted.len() >= 2 {
        let real_gaps = accepted.windows(2).all(|w| {
            let valley = smoothed[w[0]..=w[1]]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            valley <= VALLEY_TO_PEAK_MAX * smoothed[w[0]].min(smoothed[w[1]])
        });
        if !real_gaps {
            return (Vec::new(), None);
        }
    }

    // The histogram peaks are a first (biased-toward-center) guess at each
    // mode's location. Refine by averaging only the samples clearly closer
    // to one peak than to the boundary with its neighbor, discarding a
    // trimmed band around each boundary — exactly the samples a filter's
    // transition smearing between two adjacent symbols contaminates. This
    // is what a naive nearest-peak split (or an equal-count order-statistic
    // split) does not do: both still redistribute every contaminated
    // sample to one side or the other, which biases the estimated
    // separation down by however much the true modes overlap.
    let seeds: Vec<f64> = accepted
        .iter()
        .map(|&i| lo + (i as f64 + 0.5) * bin_width)
        .collect();
    let boundaries: Vec<f64> = seeds.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
    let modes = refine_modes_trimmed(if_hz, &seeds, &boundaries);

    let spacing = spacing_if_consistent(&modes);
    (modes, spacing)
}

/// Refine `seeds` (one per mode, ascending) against the raw samples,
/// averaging only samples that fall clearly nearer their assigned seed than
/// the boundary shared with a neighbor — see [`BOUNDARY_TRIM_FRACTION`].
/// `boundaries[i]` is the midpoint between `seeds[i]` and `seeds[i + 1]`.
fn refine_modes_trimmed(if_hz: &[f64], seeds: &[f64], boundaries: &[f64]) -> Vec<f64> {
    let mut sums = vec![0.0; seeds.len()];
    let mut counts = vec![0usize; seeds.len()];
    for &v in if_hz {
        let (k, _) = seeds
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (**a - v).abs().total_cmp(&(**b - v).abs()))
            .expect("seeds is non-empty");
        if boundaries.is_empty() {
            sums[k] += v;
            counts[k] += 1;
            continue;
        }
        let half_gap = match k {
            0 => (seeds[1] - seeds[0]) / 2.0,
            k if k + 1 == seeds.len() => (seeds[k] - seeds[k - 1]) / 2.0,
            k => ((seeds[k] - seeds[k - 1]).min(seeds[k + 1] - seeds[k])) / 2.0,
        };
        let nearest_boundary_dist = boundaries
            .iter()
            .map(|&b| (b - v).abs())
            .fold(f64::INFINITY, f64::min);
        if nearest_boundary_dist < half_gap * BOUNDARY_TRIM_FRACTION {
            continue; // trimmed: too close to the boundary with a neighbor
        }
        sums[k] += v;
        counts[k] += 1;
    }
    seeds
        .iter()
        .enumerate()
        .map(|(k, &seed)| {
            if counts[k] > 0 {
                sums[k] / counts[k] as f64
            } else {
                seed
            }
        })
        .collect()
}

/// The mean spacing between adjacent (sorted) modes, if every gap is within
/// [`SPACING_TOLERANCE`] of the mean — the signature of an evenly spaced
/// FSK tone ladder rather than an arbitrary set of peaks.
fn spacing_if_consistent(modes: &[f64]) -> Option<f64> {
    if modes.len() < 2 {
        return None;
    }
    let gaps: Vec<f64> = modes.windows(2).map(|w| w[1] - w[0]).collect();
    let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
    if mean <= 0.0 {
        return None;
    }
    let consistent = gaps
        .iter()
        .all(|&g| ((g - mean) / mean).abs() <= SPACING_TOLERANCE);
    consistent.then_some(mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cx(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }

    #[test]
    fn constant_tone_has_zero_envelope_variance_and_one_mode() {
        let samples: Vec<Complex<f64>> = (0..200)
            .map(|i| {
                let theta = 0.1 * i as f64;
                cx(theta.cos(), theta.sin())
            })
            .collect();
        let f = compute(&samples, 1.0).unwrap();
        assert!(f.envelope_norm_var < 1e-9);
        assert_eq!(f.modes.len(), 1);
        assert!(f.mode_spacing_hz.is_none());
    }

    #[test]
    fn on_off_keying_has_high_normalized_envelope_variance() {
        let mut samples = Vec::new();
        for sym in 0..40 {
            let on = sym % 2 == 0;
            for _ in 0..20 {
                samples.push(if on { cx(1.0, 0.0) } else { cx(0.0, 0.0) });
            }
        }
        let f = compute(&samples, 1.0).unwrap();
        assert!(
            f.envelope_norm_var > 0.5,
            "norm var {}",
            f.envelope_norm_var
        );
    }

    #[test]
    fn two_tone_signal_yields_two_well_spaced_modes() {
        // Alternate 40-sample blocks at two distinct instantaneous
        // frequencies, constant envelope throughout.
        const FS: f64 = 1_000_000.0;
        const F_LO: f64 = 5_000.0;
        const F_HI: f64 = 35_000.0;
        let mut samples = Vec::new();
        let mut phase = 0.0f64;
        for sym in 0..30 {
            let inc = std::f64::consts::TAU * (if sym % 2 == 0 { F_LO } else { F_HI }) / FS;
            for _ in 0..40 {
                samples.push(cx(phase.cos(), phase.sin()));
                phase += inc;
            }
        }
        let f = compute(&samples, FS).unwrap();
        assert_eq!(f.modes.len(), 2, "modes: {:?}", f.modes);
        let spacing = f
            .mode_spacing_hz
            .expect("two clean tones must be consistent");
        assert!(
            (spacing - (F_HI - F_LO)).abs() < 2_000.0,
            "spacing {spacing}"
        );
    }

    #[test]
    fn too_few_samples_yields_no_features() {
        assert!(compute(&[cx(1.0, 0.0); 4], 1.0).is_none());
    }
}
