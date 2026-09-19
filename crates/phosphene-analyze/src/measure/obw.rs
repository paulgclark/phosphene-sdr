// SPDX-License-Identifier: MIT

//! Occupied-bandwidth computation (FR-AM2, spec §7.5): integrate the in-blob
//! PSD; report the width by power fraction (99%-power OBW trims equal power
//! tails) or by x-dB-down crossings, always naming the convention.
//!
//! Both conventions run on the noise-stripped averaged PSD, so the noise
//! floor inside the blob neither inflates a power-fraction width nor hides a
//! dB-down crossing under residual floor energy. Edge positions are
//! interpolated to sub-bin precision — linearly on cumulative power for the
//! fraction convention, linearly in dB along the skirt for dB-down.

use super::psd::lin_to_db;
use super::ObwConvention;

/// One computed width, in span-relative fractional bin coordinates
/// (`lo`/`hi` are positions on the bin axis where bin `j` spans
/// `j − 0.5 .. j + 0.5`).
pub(super) struct ObwResult {
    /// Lower edge, fractional bins from the span start.
    pub lo_bins: f64,
    /// Upper edge, fractional bins from the span start.
    pub hi_bins: f64,
    /// True when an edge ran into the measured span's boundary — the real
    /// width may extend beyond what was measured.
    pub edge_limited: bool,
    /// The dynamic range (dB) the convention needs below the peak to place
    /// its edges honestly; confidence discounts results whose peak SNR does
    /// not clear this depth.
    pub needed_db: f64,
}

impl ObwResult {
    /// Width in bins (never negative).
    pub fn width_bins(&self) -> f64 {
        (self.hi_bins - self.lo_bins).max(0.0)
    }
}

/// Compute the width of `psd` (linear, noise-stripped, span-relative) under
/// `convention`. Returns `None` when the PSD holds no power at all.
pub(super) fn compute(psd: &[f64], convention: ObwConvention) -> Option<ObwResult> {
    match convention {
        ObwConvention::PowerFraction { fraction } => power_fraction(psd, f64::from(fraction)),
        ObwConvention::DbDown { db } => db_down(psd, f64::from(db)),
    }
}

/// Power-fraction OBW: trim `(1 − fraction) / 2` of the total power from
/// each tail; the width is what remains (the ITU 99% convention trims 0.5%
/// per side). Edges are interpolated inside the bin where the cumulative
/// crosses the tail target, treating each bin's power as uniform over its
/// width.
fn power_fraction(psd: &[f64], fraction: f64) -> Option<ObwResult> {
    let total: f64 = psd.iter().sum();
    if total <= 0.0 || !(0.0..1.0).contains(&fraction) {
        return None;
    }
    let tail = (1.0 - fraction) / 2.0 * total;

    let lo_bins = trim_from(psd.iter().copied(), tail, psd.len());
    let hi_rev = trim_from(psd.iter().rev().copied(), tail, psd.len());
    let hi_bins = (psd.len() as f64 - 1.0) - hi_rev;

    Some(ObwResult {
        lo_bins,
        hi_bins,
        edge_limited: false,
        // Placing a (1−f)/2 tail needs the PSD trustworthy down to roughly
        // that fraction of total power: 10·log10(1/(1−f)) dB below the bulk
        // (20 dB for the 99% convention).
        needed_db: -10.0 * (1.0 - fraction).log10(),
    })
}

/// Walk bins accumulating power until `tail` is reached; return the
/// fractional bin position (from the walk's origin) where the cumulative
/// crosses it.
fn trim_from(psd: impl Iterator<Item = f64>, tail: f64, len: usize) -> f64 {
    let mut cum = 0.0f64;
    for (j, p) in psd.enumerate() {
        if p > 0.0 && cum + p > tail {
            return (j as f64 - 0.5) + (tail - cum) / p;
        }
        cum += p;
    }
    // Unreachable for tail < total, but degrade to the far edge.
    len as f64 - 0.5
}

/// x-dB-down width: the outermost points where the PSD crosses `db` below
/// its peak, interpolated in dB between bin centers.
fn db_down(psd: &[f64], db: f64) -> Option<ObwResult> {
    let (peak_idx, peak) = psd
        .iter()
        .copied()
        .enumerate()
        .fold(
            (0usize, 0.0f64),
            |acc, (j, p)| if p > acc.1 { (j, p) } else { acc },
        );
    if peak <= 0.0 || db <= 0.0 {
        return None;
    }
    let threshold = peak * 10f64.powf(-db / 10.0);
    let thr_db = lin_to_db(threshold);

    // First bin at-or-above threshold from each side; the crossing sits
    // between it and its below-threshold neighbor.
    let first = psd.iter().position(|&p| p >= threshold)?;
    let last = psd.iter().rposition(|&p| p >= threshold)?;
    // The peak bin always meets the threshold, so first ≤ peak_idx ≤ last.
    debug_assert!(first <= peak_idx && peak_idx <= last);

    let mut edge_limited = false;
    let lo_bins = if first == 0 {
        edge_limited = true;
        -0.5
    } else {
        interp_crossing(
            first as f64,
            psd[first],
            (first - 1) as f64,
            psd[first - 1],
            thr_db,
        )
    };
    let hi_bins = if last == psd.len() - 1 {
        edge_limited = true;
        psd.len() as f64 - 0.5
    } else {
        interp_crossing(
            last as f64,
            psd[last],
            (last + 1) as f64,
            psd[last + 1],
            thr_db,
        )
    };

    Some(ObwResult {
        lo_bins,
        hi_bins,
        edge_limited,
        needed_db: db,
    })
}

/// Position between an above-threshold bin (`x_in`, `p_in`) and its
/// below-threshold neighbor (`x_out`, `p_out`) where the skirt crosses
/// `thr_db`, interpolating linearly in dB.
fn interp_crossing(x_in: f64, p_in: f64, x_out: f64, p_out: f64, thr_db: f64) -> f64 {
    let d_in = lin_to_db(p_in);
    let d_out = lin_to_db(p_out);
    if d_in <= d_out {
        // Degenerate (flat or rising) skirt: place the edge halfway.
        return (x_in + x_out) / 2.0;
    }
    let t = ((d_in - thr_db) / (d_in - d_out)).clamp(0.0, 1.0);
    x_in + (x_out - x_in) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_fraction_of_a_single_bin_is_the_fraction_of_its_width() {
        // All power in one bin, treated as uniform density over the bin:
        // the 99% width is 0.99 bins centered on it.
        let psd = [0.0, 0.0, 8.0, 0.0, 0.0];
        let r = compute(&psd, ObwConvention::POWER_99).unwrap();
        // Tolerance: the fraction is carried as f32 in the convention.
        assert!((r.width_bins() - 0.99).abs() < 1e-6);
        assert!(((r.lo_bins + r.hi_bins) / 2.0 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn power_fraction_trims_equal_tails_of_a_flat_psd() {
        // 10 equal bins spanning positions −0.5..9.5: trimming 0.5% per
        // side leaves 9.9 bins.
        let psd = [1.0f64; 10];
        let r = compute(&psd, ObwConvention::POWER_99).unwrap();
        assert!((r.width_bins() - 9.9).abs() < 1e-5);
    }

    #[test]
    fn db_down_interpolates_the_crossing_in_db() {
        // Peak 1.0 with −6.0206 dB (quarter-power) neighbors: the −3 dB
        // points sit 3/6.0206 of the way to the neighbors in dB.
        let quarter = 0.25f64;
        let six_db = 10.0 * 4f64.log10();
        let psd = [0.0, quarter, 1.0, quarter, 0.0];
        let r = compute(&psd, ObwConvention::DB_DOWN_3).unwrap();
        assert!(!r.edge_limited);
        assert!((r.width_bins() - 2.0 * (3.0 / six_db)).abs() < 1e-9);
    }

    #[test]
    fn db_down_flags_an_edge_limited_width() {
        let psd = [1.0, 1.0, 1.0];
        let r = compute(&psd, ObwConvention::DB_DOWN_26).unwrap();
        assert!(r.edge_limited);
        assert!((r.width_bins() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn deeper_db_down_conventions_are_never_narrower() {
        let psd: Vec<f64> = (-8..=8)
            .map(|d: i32| 10f64.powf(-0.5 * f64::from(d * d) / 10.0))
            .collect();
        let w3 = compute(&psd, ObwConvention::DB_DOWN_3)
            .unwrap()
            .width_bins();
        let w20 = compute(&psd, ObwConvention::DB_DOWN_20)
            .unwrap()
            .width_bins();
        let w26 = compute(&psd, ObwConvention::DB_DOWN_26)
            .unwrap()
            .width_bins();
        assert!(w3 <= w20 && w20 <= w26, "{w3} {w20} {w26}");
    }

    #[test]
    fn empty_psd_yields_no_width() {
        let psd = [0.0f64; 8];
        assert!(compute(&psd, ObwConvention::POWER_99).is_none());
        assert!(compute(&psd, ObwConvention::DB_DOWN_26).is_none());
    }
}
