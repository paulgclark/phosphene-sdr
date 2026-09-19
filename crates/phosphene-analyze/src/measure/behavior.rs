// SPDX-License-Identifier: MIT

//! Behavior flags (FR-AM5): hopping, bursty vs continuous, drifting center.
//!
//! All three derive from the per-frame power centroids of the measured
//! region:
//!
//! * **bursty** — the temporal analysis found repeated on/off transitions
//!   rather than continuous presence;
//! * **hopping** — the centroid dwells at ≥2 discrete frequencies, jumping
//!   between them (dwell segmentation + gap clustering; hop set and rate
//!   reported when resolvable, per FR-AM5);
//! * **drifting** — the centroid moves coherently (least-squares fit) by
//!   more than the configured span over the window, without hopping.

use super::psd::count_factor;
use super::temporal::TemporalAnalysis;
use super::{BehaviorFlags, HopDetail, Measured};
use crate::tap::BandMeta;

/// Inputs the engine hands to the behavior analysis.
pub(super) struct BehaviorInputs<'a> {
    /// Per-frame centroid in absolute bin coordinates (`None` when off).
    pub centroids: &'a [Option<f64>],
    /// Temporal analysis (bursty/continuous and the run structure).
    pub temporal: &'a TemporalAnalysis,
    /// Minimum hop jump, bins (config). Deliberately *not* scaled by the
    /// measured OBW: a hopper's aggregate PSD spans its whole hop set, so
    /// scaling by it would raise the threshold above the very jumps it is
    /// meant to detect. Wideband signals whose centroid wobbles beyond the
    /// default need a larger configured value.
    pub hop_jump_bins: f64,
    /// Minimum fitted drift, bins (config).
    pub drift_min_bins: f64,
    /// Seconds per frame.
    pub frame_dt_s: f64,
}

/// A maximal stretch of consecutive on-frames whose centroid stays put —
/// one dwell at one frequency.
struct Segment {
    start_frame: usize,
    n_frames: usize,
    centroid_sum: f64,
}

impl Segment {
    fn centroid(&self) -> f64 {
        self.centroid_sum / self.n_frames as f64
    }
}

/// Compute the FR-AM5 flags. `meta` maps clustered hop centroids to Hz in
/// the same frame of reference as the reported center (absolute when the
/// band center is known, offsets otherwise).
pub(super) fn analyze(inputs: &BehaviorInputs<'_>, meta: &BandMeta) -> BehaviorFlags {
    let jump_thr = inputs.hop_jump_bins;
    // Confidence in a NEGATIVE flag comes from how much was observed: an
    // absence asserted over sixty on-frames is better defended than one
    // over five (NFR-A3 cuts both ways).
    let n_on = inputs.centroids.iter().filter(|c| c.is_some()).count();
    let absence_conf = count_factor(n_on, 16.0) as f32;

    let segments = segment_dwells(inputs.centroids, jump_thr);
    let (hopping, hop) = match detect_hopping(&segments, jump_thr, inputs.frame_dt_s, meta) {
        // A positive hop call is as defended as the dwell evidence behind
        // it — the same count that bounds the rate confidence.
        Some((detail, conf)) => (Measured::new(true, conf), Some(detail)),
        None => (Measured::new(false, absence_conf), None),
    };

    let bursty = if !inputs.temporal.continuous && inputs.temporal.runs.len() >= 2 {
        // Defended by the number of on/off repetitions actually seen.
        Measured::new(true, count_factor(inputs.temporal.runs.len(), 2.0) as f32)
    } else {
        Measured::new(false, absence_conf)
    };

    // A hopper's centroid jumps would masquerade as drift; only a
    // non-hopping signal can drift, and the "not drifting" claim is then
    // exactly as defended as the hop call that explains the motion.
    let drifting = if hopping.value {
        Measured::new(false, hopping.confidence)
    } else {
        detect_drift(inputs.centroids, inputs.drift_min_bins)
    };

    BehaviorFlags {
        hopping,
        bursty,
        drifting,
        hop,
    }
}

/// Split the on-frames into dwell segments, breaking at off-gaps and at
/// centroid jumps larger than `jump_thr` bins.
fn segment_dwells(centroids: &[Option<f64>], jump_thr: f64) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();
    let mut current: Option<Segment> = None;
    let mut prev_centroid: Option<f64> = None;

    for (i, c) in centroids.iter().enumerate() {
        match c {
            Some(c) => {
                let continues = matches!(prev_centroid, Some(p) if (c - p).abs() <= jump_thr);
                match &mut current {
                    Some(seg) if continues => {
                        seg.n_frames += 1;
                        seg.centroid_sum += c;
                    }
                    _ => {
                        if let Some(seg) = current.take() {
                            segments.push(seg);
                        }
                        current = Some(Segment {
                            start_frame: i,
                            n_frames: 1,
                            centroid_sum: *c,
                        });
                    }
                }
                prev_centroid = Some(*c);
            }
            None => {
                if let Some(seg) = current.take() {
                    segments.push(seg);
                }
                prev_centroid = None;
            }
        }
    }
    if let Some(seg) = current {
        segments.push(seg);
    }
    segments
}

/// Cluster dwell centroids by gap and decide hopping; on success returns
/// the detail plus the confidence of the hop call itself. Single-frame
/// dwells are discarded when longer ones exist — a frame straddling a hop
/// boundary carries a split-energy centroid that belongs to no real dwell.
fn detect_hopping(
    segments: &[Segment],
    jump_thr: f64,
    frame_dt_s: f64,
    meta: &BandMeta,
) -> Option<(HopDetail, f32)> {
    let longest = segments.iter().map(|s| s.n_frames).max().unwrap_or(0);
    let kept: Vec<&Segment> = segments
        .iter()
        .filter(|s| s.n_frames >= 2 || longest < 2)
        .collect();
    if kept.len() < 3 {
        return None;
    }

    // Gap clustering over sorted dwell centroids.
    let mut sorted: Vec<f64> = kept.iter().map(|s| s.centroid()).collect();
    sorted.sort_by(f64::total_cmp);
    let mut clusters: Vec<Vec<f64>> = vec![vec![sorted[0]]];
    for &c in &sorted[1..] {
        let last = clusters.last_mut().expect("clusters is never empty");
        let last_val = *last.last().expect("cluster is never empty");
        if c - last_val > jump_thr {
            clusters.push(vec![c]);
        } else {
            last.push(c);
        }
    }
    if clusters.len() < 2 {
        return None;
    }

    let to_hz = |bin: f64| match meta.center_freq_hz {
        Some(_) => meta
            .bin_absolute_hz(bin)
            .expect("center known implies absolute mapping"),
        None => meta.bin_offset_hz(bin),
    };
    // Per-hop confidence from the evidence this stage genuinely holds: how
    // many dwells were observed at that frequency. A hop visited a dozen
    // times and one visited twice are not equally certain (NFR-A3).
    let set_hz: Vec<Measured<f64>> = clusters
        .iter()
        .map(|c| {
            Measured::new(
                to_hz(c.iter().sum::<f64>() / c.len() as f64),
                count_factor(c.len(), 2.0) as f32,
            )
        })
        .collect();

    // Hop rate: dwell starts are one hop apart; (n − 1) intervals span the
    // first to last start. The rate — and the hop call itself — are as
    // defended as the number of hops observed.
    let evidence_conf = count_factor(kept.len() - 1, 3.0) as f32;
    let first = kept.first().expect("kept has ≥3 segments").start_frame;
    let last = kept.last().expect("kept has ≥3 segments").start_frame;
    let rate_hz = (last > first).then(|| {
        let hops = (kept.len() - 1) as f64;
        let span_s = (last - first) as f64 * frame_dt_s;
        Measured::new(hops / span_s, evidence_conf)
    });

    Some((HopDetail { set_hz, rate_hz }, evidence_conf))
}

/// Least-squares fit of centroid against frame index; drifting when the
/// fitted motion across the window exceeds `min_bins` and the fit explains
/// most of the variance. Either way the flag carries the confidence its
/// evidence supports: a positive call by how well the fit explains the
/// motion (R²) over how many points, a negative call by how much was
/// observed without coherent motion.
fn detect_drift(centroids: &[Option<f64>], min_bins: f64) -> Measured<bool> {
    let pts: Vec<(f64, f64)> = centroids
        .iter()
        .enumerate()
        .filter_map(|(i, c)| c.map(|c| (i as f64, c)))
        .collect();
    let points_conf = count_factor(pts.len(), 8.0) as f32;
    if pts.len() < 4 {
        return Measured::new(false, points_conf);
    }
    let n = pts.len() as f64;
    let mean_x = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = pts.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
    let sxy: f64 = pts.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
    let syy: f64 = pts.iter().map(|p| (p.1 - mean_y).powi(2)).sum();
    if sxx <= 0.0 || syy <= 0.0 {
        // A perfectly stationary centroid: no motion at all is strong
        // evidence of no drift.
        return Measured::new(false, points_conf);
    }
    let slope = sxy / sxx;
    let fitted_span = slope.abs() * (pts.last().expect("≥4 points").0 - pts[0].0);
    let r2 = (sxy * sxy) / (sxx * syy);
    if fitted_span >= min_bins && r2 >= 0.6 {
        Measured::new(true, (r2.clamp(0.0, 1.0) as f32) * points_conf)
    } else {
        Measured::new(false, points_conf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::temporal;

    fn meta(center: Option<f64>) -> BandMeta {
        BandMeta {
            center_freq_hz: center,
            span_hz: 1_024_000.0,
            bins: 1024,
            bin_hz: 1000.0,
            frame_dt_s: 0.001,
            cal_offset_db: 0.0,
        }
    }

    fn inputs<'a>(
        centroids: &'a [Option<f64>],
        temporal: &'a temporal::TemporalAnalysis,
    ) -> BehaviorInputs<'a> {
        BehaviorInputs {
            centroids,
            temporal,
            hop_jump_bins: 3.0,
            drift_min_bins: 2.0,
            frame_dt_s: 0.001,
        }
    }

    fn analyzed(on: &[bool]) -> temporal::TemporalAnalysis {
        temporal::analyze(on, 0.001, 0.98, 0.5)
    }

    #[test]
    fn steady_tone_raises_no_flags_yet_defends_the_absences() {
        let centroids: Vec<Option<f64>> = (0..32).map(|_| Some(600.0)).collect();
        let on = vec![true; 32];
        let t = analyzed(&on);
        let flags = analyze(&inputs(&centroids, &t), &meta(None));
        assert!(!flags.hopping.value && !flags.bursty.value && !flags.drifting.value);
        for (what, c) in [
            ("hopping", flags.hopping.confidence),
            ("bursty", flags.bursty.confidence),
            ("drifting", flags.drifting.confidence),
        ] {
            assert!(c > 0.0 && c <= 1.0, "negative {what} flag undefended: {c}");
        }
        assert!(flags.hop.is_none());
    }

    #[test]
    fn cycling_centroid_is_hopping_with_set_and_rate() {
        // 5-frame dwells cycling three frequencies, back to back.
        let hops = [312.0, 512.0, 712.0];
        let centroids: Vec<Option<f64>> = (0..60).map(|i| Some(hops[(i / 5) % 3])).collect();
        let on = vec![true; 60];
        let t = analyzed(&on);
        let flags = analyze(&inputs(&centroids, &t), &meta(None));
        assert!(flags.hopping.value);
        assert!(flags.hopping.confidence > 0.0);
        assert!(!flags.drifting.value, "hop jumps must not read as drift");
        let hop = flags.hop.unwrap();
        assert_eq!(hop.set_hz.len(), 3);
        assert!((hop.set_hz[0].value - -200_000.0).abs() < 1.0);
        assert!((hop.set_hz[2].value - 200_000.0).abs() < 1.0);
        for f in &hop.set_hz {
            assert!(f.confidence > 0.0, "hop frequency undefended");
        }
        let rate = hop.rate_hz.unwrap();
        // One hop per 5 frames of 1 ms.
        assert!((rate.value - 200.0).abs() < 20.0, "rate {}", rate.value);
        assert!(rate.confidence > 0.0);
    }

    #[test]
    fn hop_frequencies_seen_more_often_are_more_confident() {
        // Dwell pattern A B A B A B A B A A-merged…: frequency A is visited
        // as a distinct dwell more often than B, so A's set entry must carry
        // the higher confidence — they are not equally certain (NFR-A3).
        let pattern = [312.0, 712.0, 312.0, 512.0, 312.0, 512.0, 312.0, 712.0];
        let centroids: Vec<Option<f64>> = pattern
            .iter()
            .flat_map(|&f| std::iter::repeat_n(Some(f), 5))
            .collect();
        let on = vec![true; centroids.len()];
        let t = analyzed(&on);
        let flags = analyze(&inputs(&centroids, &t), &meta(None));
        let hop = flags.hop.unwrap();
        assert_eq!(hop.set_hz.len(), 3);
        let a = &hop.set_hz[0]; // 312: 4 dwells
        let b = &hop.set_hz[2]; // 712: 2 dwells
        assert!(
            a.confidence > b.confidence,
            "4 dwells ({}) must outrank 2 dwells ({})",
            a.confidence,
            b.confidence
        );
    }

    #[test]
    fn hop_set_is_absolute_when_the_band_center_is_known() {
        let hops = [312.0, 712.0];
        let centroids: Vec<Option<f64>> = (0..40).map(|i| Some(hops[(i / 5) % 2])).collect();
        let on = vec![true; 40];
        let t = analyzed(&on);
        let flags = analyze(&inputs(&centroids, &t), &meta(Some(100_000_000.0)));
        let hop = flags.hop.unwrap();
        assert!((hop.set_hz[0].value - 99_800_000.0).abs() < 1.0);
        assert!((hop.set_hz[1].value - 100_200_000.0).abs() < 1.0);
    }

    #[test]
    fn straddle_frames_do_not_invent_hop_frequencies() {
        // Two real dwell frequencies, with a single-frame split-energy
        // centroid between each hop (the straddle artifact).
        let mut centroids: Vec<Option<f64>> = Vec::new();
        for hop in 0..8 {
            let f = if hop % 2 == 0 { 312.0 } else { 712.0 };
            centroids.extend(std::iter::repeat_n(Some(f), 4));
            centroids.push(Some(512.0)); // straddle frame
        }
        let on = vec![true; centroids.len()];
        let t = analyzed(&on);
        let flags = analyze(&inputs(&centroids, &t), &meta(None));
        let hop = flags.hop.unwrap();
        assert_eq!(hop.set_hz.len(), 2, "straddle centroid must be discarded");
    }

    #[test]
    fn coherent_motion_is_drift_not_hopping() {
        // 1 bin/frame: smooth motion, far exceeding drift_min_bins.
        let centroids: Vec<Option<f64>> = (0..32).map(|i| Some(400.0 + i as f64)).collect();
        let on = vec![true; 32];
        let t = analyzed(&on);
        let flags = analyze(&inputs(&centroids, &t), &meta(None));
        assert!(flags.drifting.value);
        assert!(flags.drifting.confidence > 0.0);
        assert!(!flags.hopping.value);
    }

    #[test]
    fn burst_train_at_one_frequency_is_bursty_only() {
        let mut centroids = Vec::new();
        let mut on = Vec::new();
        for _ in 0..4 {
            centroids.extend(std::iter::repeat_n(Some(600.0), 5));
            on.extend(std::iter::repeat_n(true, 5));
            centroids.extend(std::iter::repeat_n(None, 10));
            on.extend(std::iter::repeat_n(false, 10));
        }
        let t = analyzed(&on);
        let flags = analyze(&inputs(&centroids, &t), &meta(None));
        assert!(flags.bursty.value);
        assert!(flags.bursty.confidence > 0.0);
        assert!(!flags.hopping.value && !flags.drifting.value);
    }
}
