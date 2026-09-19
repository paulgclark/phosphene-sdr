// SPDX-License-Identifier: MIT

//! Internal spectrum bookkeeping for the measure stage: presence per frame,
//! the on-frame-averaged in-band PSD (raw and noise-stripped), and per-frame
//! power centroids. Everything downstream (§7.5 OBW, power/SNR, temporal,
//! behavior) reads these once-computed statistics.

use phosphene_core::DBFS_FLOOR;

use crate::tap::FrameView;

/// dB → linear power.
pub(super) fn db_to_lin(db: f64) -> f64 {
    10f64.powf(db / 10.0)
}

/// Linear power → dB, floored at the core convention's −200 dB.
pub(super) fn lin_to_db(lin: f64) -> f64 {
    if lin > 0.0 {
        (10.0 * lin.log10()).max(f64::from(DBFS_FLOOR))
    } else {
        f64::from(DBFS_FLOOR)
    }
}

/// Confidence factor rising with SNR: 0 at ≤0 dB, saturating toward 1.
pub(super) fn snr_factor(snr_db: f64) -> f64 {
    if snr_db <= 0.0 {
        0.0
    } else {
        (snr_db / (snr_db + 12.0)).clamp(0.0, 1.0)
    }
}

/// Confidence factor rising with an observation count: `n / (n + half)`,
/// reaching 0.5 at `half` observations.
pub(super) fn count_factor(n: usize, half: f64) -> f64 {
    let n = n as f64;
    (n / (n + half)).clamp(0.0, 1.0)
}

/// Per-region statistics over one [`FrameView`] window.
pub(super) struct RegionStats {
    /// Total frames analyzed.
    pub n_frames: usize,
    /// Presence per frame: in-band power exceeded the in-band floor by the
    /// configured margin.
    pub on: Vec<bool>,
    /// Number of `on` frames.
    pub n_on: usize,
    /// Per-bin linear power averaged over ON frames with the per-bin noise
    /// floor subtracted, clamped at zero, span-relative (index 0 is
    /// `span.lo`) — the in-blob signal PSD the §7.5 measurements integrate.
    /// Calibration offset applied.
    pub avg_stripped: Vec<f64>,
    /// Per-frame power-weighted centroid (absolute bin coordinate, from the
    /// noise-stripped frame); `None` when the frame is off or empty.
    pub centroids: Vec<Option<f64>>,
    /// Σ linear noise-floor power over the span's bins (per-bin sum, no
    /// ENBW correction).
    pub noise_sum: f64,
    /// Per-bin linear noise floor at the bin holding the peak of `avg_raw`.
    pub floor_at_peak: f64,
    /// Peak of `avg_raw`, linear.
    pub peak_raw: f64,
}

impl RegionStats {
    /// Compute the statistics for `span.lo..=span.hi` over every frame in
    /// `frames`. `floor_lin` is the full-band per-bin noise floor in linear
    /// power, already on the calibrated scale.
    pub fn compute(
        frames: &FrameView,
        span: (usize, usize),
        floor_lin: &[f64],
        cal_offset_db: f32,
        presence_margin_db: f32,
    ) -> Self {
        let (lo, hi) = span;
        let width = hi - lo + 1;
        let n_frames = frames.len();
        let cal = f64::from(cal_offset_db);

        let noise_sum: f64 = floor_lin[lo..=hi].iter().sum();
        let margin_lin = db_to_lin(f64::from(presence_margin_db));
        // A frame is "on" when in-band power exceeds the floor by the margin.
        // With no real floor supplied (all −200 dB) this degrades gracefully:
        // silence sits at the floor and never crosses the margin.
        let on_threshold = noise_sum * margin_lin;

        let mut on = Vec::with_capacity(n_frames);
        let mut centroids = Vec::with_capacity(n_frames);
        let mut sum_on = vec![0.0f64; width];
        let mut n_on = 0usize;

        let mut frame_lin = vec![0.0f64; width];
        for i in 0..n_frames {
            let frame = frames.frame(i);
            let mut total = 0.0f64;
            for (j, v) in frame[lo..=hi].iter().enumerate() {
                let p = db_to_lin(f64::from(*v) + cal);
                frame_lin[j] = p;
                total += p;
            }
            let present = total > on_threshold;
            on.push(present);
            if present {
                n_on += 1;
                let mut stripped_sum = 0.0f64;
                let mut weighted = 0.0f64;
                for (j, p) in frame_lin.iter().enumerate() {
                    let s = (p - floor_lin[lo + j]).max(0.0);
                    sum_on[j] += *p;
                    stripped_sum += s;
                    weighted += s * (lo + j) as f64;
                }
                centroids.push((stripped_sum > 0.0).then(|| weighted / stripped_sum));
            } else {
                centroids.push(None);
            }
        }
        // `sum_on` accumulated raw (unstripped) power; derive both views.
        let avg_raw: Vec<f64> = if n_on > 0 {
            sum_on.iter().map(|s| s / n_on as f64).collect()
        } else {
            vec![0.0; width]
        };
        let avg_stripped: Vec<f64> = avg_raw
            .iter()
            .enumerate()
            .map(|(j, p)| (p - floor_lin[lo + j]).max(0.0))
            .collect();

        let (peak_idx, peak_raw) =
            avg_raw
                .iter()
                .copied()
                .enumerate()
                .fold(
                    (0usize, 0.0f64),
                    |acc, (j, p)| {
                        if p > acc.1 {
                            (j, p)
                        } else {
                            acc
                        }
                    },
                );

        RegionStats {
            n_frames,
            on,
            n_on,
            avg_stripped,
            centroids,
            noise_sum,
            floor_at_peak: floor_lin[lo + peak_idx],
            peak_raw,
        }
    }

    /// Peak-bin SNR of the averaged PSD, dB — bounds how deep below the peak
    /// any dB-down convention can honestly look.
    pub fn peak_snr_db(&self) -> f64 {
        if self.floor_at_peak > 0.0 && self.peak_raw > 0.0 {
            lin_to_db(self.peak_raw / self.floor_at_peak)
        } else {
            0.0
        }
    }
}
