// SPDX-License-Identifier: MIT

//! CFAR detection along the frequency axis (nextgen spec §7.2, FR-AD2), with
//! the reference level derived from the FR-AD1 robust floor (D-023).
//!
//! For each bin (the cell under test, CUT), the local noise level `N_k` is
//! the **median of the per-bin robust noise floors** (§7.1) over
//! `ref_cells` reference cells on each side, skipping `guard_cells`
//! immediately adjacent bins so the tested signal's own mainlobe does not
//! sit in its reference set. The threshold keeps the §7.2 `α(Pfa)` form:
//!
//! ```text
//! T_k = α·N_k        α = −ln(Pfa)
//! ```
//!
//! the exact false-alarm scale for exponentially distributed power (each
//! FFT bin of the §7.1 noise model) about a known noise level.
//!
//! ## Why the reference is the robust floor, not a cell average (D-023)
//!
//! A plain cell average over instantaneous reference powers is pulled up by
//! a strong occupant sitting anywhere in the reference window — the classic
//! CA-CFAR *masking* failure: the weak signal next to a strong one
//! disappears, and finding that signal is the product. FR-AD1's rank-order
//! floor is resistant to occupied bins, so referencing it makes the
//! threshold immune on both axes: the *temporal* rank rejects bursty
//! occupants in each reference bin's own history, and the *spatial median*
//! across the reference cells rejects the handful of floors a continuous
//! occupant does contaminate (a continuous signal is indistinguishable from
//! noise by any per-bin temporal statistic in its own bins — only there).
//! The local reference window and guard cells of §7.2 are retained: the
//! threshold still adapts to the local level, not a band-wide constant.
//!
//! ## Reference-cell stride (*tune*)
//!
//! Neighboring FFT bins of a windowed transform are **correlated** (the
//! window's squared spectrum has support of a few bins: exactly ±2 bins for
//! Hann), so neighboring bins' floor estimates are correlated too, which
//! would inflate the variance of the reference median and lift the realized
//! Pfa. Reference cells are therefore taken every
//! [`CfarConfig::ref_stride`] bins; the default of 3 makes them exactly
//! uncorrelated under the harness's Hann window, and the measured
//! false-alarm rate matches the configured Pfa within statistical tolerance
//! (the seal test states ±25%; measured ≈3%). Windows whose squared
//! spectrum is wider (e.g. Blackman-Harris: ±6 bins) need a larger stride.
//!
//! At band edges the missing side is simply truncated: the median there
//! rests on fewer reference cells (a somewhat noisier `N_k`, same expected
//! level), so the false-alarm rate stays essentially constant across the
//! band. Because `N_k` comes from the sliding-window floor, thresholds
//! during the first noise window are warmup-coarse (biased conservative);
//! steady state begins once the window fills.

use crate::detect::DetectError;

/// Configuration for [`Cfar`].
#[derive(Debug, Clone, PartialEq)]
pub struct CfarConfig {
    /// Target false-alarm probability per cell, in (0, 1).
    pub pfa: f64,
    /// Reference cells per side.
    pub ref_cells: usize,
    /// Guard cells per side, excluded between the CUT and the first
    /// reference cell. May be zero.
    pub guard_cells: usize,
    /// Spacing between successive reference cells, in bins (module docs).
    pub ref_stride: usize,
    /// dB added to the reference median before thresholding (*tune*): a
    /// calibration slot for any systematic offset between the
    /// median-of-floor-estimates and the true mean noise power. `0.0` with
    /// the default §7.1 estimator — the seal's Pfa test is the check.
    pub floor_debias_db: f32,
}

impl CfarConfig {
    /// Defaults: `Pfa = 10⁻⁴`, 16 reference cells per side, 2 guard cells,
    /// stride 3, no extra debias.
    ///
    /// `Pfa` moved one order of magnitude tighter than the original
    /// `10⁻³` (AN-1/AN-3, D-126): on a real radio the owner saw "many, many
    /// false positives", measured at `10⁻³` to about one false-alarm cell
    /// per 1000 bins per frame — thousands per second at FFT 1024. Paired
    /// with [`crate::detect::DetectorConfig::min_blob_cells`]'s own default
    /// raise (the two are tuned together — a tighter Pfa alone was measured
    /// insufficient without the blob-size gate, and the reverse) and the
    /// tracker's `birth_m` default dropping to `1` (a blob now publishes on
    /// its first detection instead of surviving an M-of-N window), so both
    /// changes together are what the false-alarm rate actually reflects; see
    /// [`crate::track::TrackerConfig::default`]'s doc for the measured
    /// before/after table on the real 913 MHz capture.
    pub fn new() -> Self {
        CfarConfig {
            pfa: 1e-4,
            ref_cells: 16,
            guard_cells: 2,
            ref_stride: 3,
            floor_debias_db: 0.0,
        }
    }
}

impl Default for CfarConfig {
    fn default() -> Self {
        CfarConfig::new()
    }
}

/// Per-frame CFAR detector producing the FR-AD2 occupancy mask from a dB
/// power frame and the FR-AD1 per-bin robust floor. See the [module
/// docs](self).
///
/// D-125 Amendment 4 item 1: this type carried a cached threshold
/// (D-125 Amendment 3 item 2) between commits 320af1b and this one, removed
/// here on the owner's ruling — "it was reused on 0.6% and 0% of bins, so
/// it is code for no gain" (measured in `crates/phosphene-app/src/analyze.
/// rs`'s `optimised_full_coverage_analysis_time_at_10_and_20_msps`; real
/// noise floors change nearly every bin nearly every frame, so almost
/// nothing was ever reusable). `detect::identity_tests` stays green across
/// the removal because it is exact: dropping the cache cannot change what
/// gets computed, only how often.
#[derive(Debug, Clone)]
pub struct Cfar {
    bins: usize,
    ref_cells: usize,
    guard_cells: usize,
    ref_stride: usize,
    /// `−ln(Pfa)`, applied to the linear reference level.
    alpha: f64,
    debias_db: f32,
    /// Scratch: the frame in linear power.
    linear: Vec<f64>,
}

impl Cfar {
    /// Build a detector for frames of `bins` bins; validates the
    /// configuration up front.
    pub fn new(bins: usize, config: CfarConfig) -> Result<Self, DetectError> {
        if bins == 0 {
            return Err(DetectError::InvalidConfig(
                "CFAR needs at least one bin".to_string(),
            ));
        }
        if !config.pfa.is_finite() || config.pfa <= 0.0 || config.pfa >= 1.0 {
            return Err(DetectError::InvalidConfig(format!(
                "CFAR Pfa must lie strictly inside (0, 1), got {}",
                config.pfa
            )));
        }
        if config.ref_cells == 0 {
            return Err(DetectError::InvalidConfig(
                "CFAR needs at least one reference cell per side".to_string(),
            ));
        }
        if config.ref_stride == 0 {
            return Err(DetectError::InvalidConfig(
                "CFAR reference stride must be at least one bin".to_string(),
            ));
        }
        if !config.floor_debias_db.is_finite() {
            return Err(DetectError::InvalidConfig(format!(
                "CFAR floor debias must be finite, got {} dB",
                config.floor_debias_db
            )));
        }
        // Every bin must reach at least one reference cell: the nearest one
        // sits guard + stride bins away on either side.
        let nearest = config.guard_cells + config.ref_stride;
        if nearest >= bins {
            return Err(DetectError::InvalidConfig(format!(
                "CFAR guard ({}) + stride ({}) spans the whole {bins}-bin frame — \
                 no bin would have any reference cell",
                config.guard_cells, config.ref_stride
            )));
        }
        Ok(Cfar {
            bins,
            ref_cells: config.ref_cells,
            guard_cells: config.guard_cells,
            ref_stride: config.ref_stride,
            alpha: -config.pfa.ln(),
            debias_db: config.floor_debias_db,
            linear: vec![0.0; bins],
        })
    }

    /// Threshold one dB power frame against the FR-AD1 robust floor into
    /// the per-frame occupancy mask: `mask[k]` is true where bin `k`
    /// exceeds `α · N_k`, with `N_k` the median reference floor (module
    /// docs). `floor_db` is the current
    /// [`crate::detect::NoiseFloor::floor_dbfs`] slice.
    ///
    /// # Panics
    ///
    /// Panics if `frame.len()`, `floor_db.len()` or `mask.len()` differs
    /// from the constructed bin count.
    pub fn detect(&mut self, frame: &[f32], floor_db: &[f32], mask: &mut [bool]) {
        assert_eq!(
            frame.len(),
            self.bins,
            "frame length must equal the detector's bin count"
        );
        assert_eq!(
            floor_db.len(),
            self.bins,
            "floor length must equal the detector's bin count"
        );
        assert_eq!(
            mask.len(),
            self.bins,
            "mask length must equal the detector's bin count"
        );
        for (lin, &db) in self.linear.iter_mut().zip(frame) {
            *lin = 10f64.powf(db as f64 / 10.0);
        }

        // D-125 Amendment 4 item 2 (the owner: "yes" to parallelising): a
        // CUT's threshold depends only on `floor_db` (read-only here, so
        // freely shareable across threads) at its own reference-cell
        // indices, never on another bin's mask output, so `mask` alone
        // needs splitting — into the same chunk size `NoiseFloor::
        // push_frame` uses. Dispatched through rayon's persistent pool, not
        // raw `std::thread::scope` spawns — see `push_frame`'s own doc
        // comment for the measurement that ruled the naive approach out.
        let bins = self.bins;
        let ref_cells = self.ref_cells;
        let guard_cells = self.guard_cells;
        let ref_stride = self.ref_stride;
        let alpha = self.alpha;
        let debias_db = self.debias_db;
        let threads = rayon::current_num_threads();
        let chunk_bins = bins.div_ceil(threads.max(1)).max(1);
        let linear: &[f64] = &self.linear;

        use rayon::prelude::*;
        mask.par_chunks_mut(chunk_bins)
            .enumerate()
            .for_each(|(chunk_idx, mask_c)| {
                let chunk_start = chunk_idx * chunk_bins;
                // Each chunk's own scratch — `Cfar::refs` (a single,
                // struct-persisted buffer) could not be shared across
                // concurrent chunks, so this scratch is local here instead,
                // allocated once per chunk (a handful per frame) rather
                // than once per `Cfar` (the previous, single-threaded
                // shape) or once per bin (which `for_each` alone would do).
                let mut refs = vec![0.0f32; 2 * ref_cells];
                for (local_k, out) in mask_c.iter_mut().enumerate() {
                    let k = chunk_start + local_k;
                    let mut n = 0usize;
                    for j in 1..=ref_cells {
                        let d = guard_cells + ref_stride * j;
                        if let Some(i) = k.checked_sub(d) {
                            refs[n] = floor_db[i];
                            n += 1;
                        }
                        if k + d < bins {
                            refs[n] = floor_db[k + d];
                            n += 1;
                        }
                    }
                    // n ≥ 1 by the construction-time span check. Upper
                    // median by rank: robust to up to half the reference
                    // floors being contaminated by continuous occupants.
                    let refs_slice = &mut refs[..n];
                    let (_, &mut median_db, _) =
                        refs_slice.select_nth_unstable_by(n / 2, f32::total_cmp);
                    let n_k = 10f64.powf((median_db + debias_db) as f64 / 10.0);
                    *out = linear[k] > alpha * n_k;
                }
            });
    }

    /// Bins per frame this detector was built for.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// D-125 Amendment 4 item 2's power proof: identical to [`Self::
    /// detect`] except every chunk computes its bins' global index as if
    /// the chunk started at bin 0, instead of its own real chunk start —
    /// a plausible parallelisation *coordination* bug (the kind a
    /// hand-rolled offset-tracking loop can introduce), not a math bug
    /// (Amendment 3's own broken variant already covers those). Every
    /// index used stays in-bounds — this produces silently wrong output,
    /// not a panic, which is exactly what the identity check must catch.
    /// `chunks` (forced rather than derived from `rayon::
    /// current_num_threads()`) must be `> 1` for the bug to have anything
    /// to misalign — a single chunk covering every bin starts at 0
    /// correctly by coincidence. `#[cfg(test)]`-only; never called from
    /// [`Self::detect`].
    #[cfg(test)]
    pub(crate) fn detect_with_wrong_chunk_offsets(
        &mut self,
        frame: &[f32],
        floor_db: &[f32],
        mask: &mut [bool],
        chunks: usize,
    ) {
        for (lin, &db) in self.linear.iter_mut().zip(frame) {
            *lin = 10f64.powf(db as f64 / 10.0);
        }
        let bins = self.bins;
        let ref_cells = self.ref_cells;
        let guard_cells = self.guard_cells;
        let ref_stride = self.ref_stride;
        let alpha = self.alpha;
        let debias_db = self.debias_db;
        let chunk_bins = bins.div_ceil(chunks.max(1)).max(1);
        let linear: &[f64] = &self.linear;

        use rayon::prelude::*;
        mask.par_chunks_mut(chunk_bins).for_each(|mask_c| {
            let mut refs = vec![0.0f32; 2 * ref_cells];
            for (local_k, out) in mask_c.iter_mut().enumerate() {
                // BUG (deliberate): should be `chunk_start + local_k`.
                let k = local_k;
                let mut n = 0usize;
                for j in 1..=ref_cells {
                    let d = guard_cells + ref_stride * j;
                    if let Some(i) = k.checked_sub(d) {
                        refs[n] = floor_db[i];
                        n += 1;
                    }
                    if k + d < bins {
                        refs[n] = floor_db[k + d];
                        n += 1;
                    }
                }
                let refs_slice = &mut refs[..n];
                let (_, &mut median_db, _) =
                    refs_slice.select_nth_unstable_by(n / 2, f32::total_cmp);
                let n_k = 10f64.powf((median_db + debias_db) as f64 / 10.0);
                *out = linear[k] > alpha * n_k;
            }
        });
    }
}

/// D-125 Amendment 3 item 3: a frozen copy of the pre-caching
/// [`Cfar::detect`] — every bin's threshold recomputed from `floor_db` every
/// frame, no cache — kept **only** as the identity proof's oracle in
/// `detect::identity_tests`. Deliberately never "improved" alongside
/// [`Cfar`] itself.
#[cfg(test)]
pub(crate) mod reference {
    use super::CfarConfig;

    #[derive(Debug, Clone)]
    pub(crate) struct CfarRef {
        bins: usize,
        ref_cells: usize,
        guard_cells: usize,
        ref_stride: usize,
        alpha: f64,
        debias_db: f32,
        linear: Vec<f64>,
        refs: Vec<f32>,
    }

    impl CfarRef {
        pub(crate) fn new(bins: usize, config: CfarConfig) -> Self {
            CfarRef {
                bins,
                ref_cells: config.ref_cells,
                guard_cells: config.guard_cells,
                ref_stride: config.ref_stride,
                alpha: -config.pfa.ln(),
                debias_db: config.floor_debias_db,
                linear: vec![0.0; bins],
                refs: vec![0.0; 2 * config.ref_cells],
            }
        }

        pub(crate) fn detect(&mut self, frame: &[f32], floor_db: &[f32], mask: &mut [bool]) {
            for (lin, &db) in self.linear.iter_mut().zip(frame) {
                *lin = 10f64.powf(db as f64 / 10.0);
            }
            for (k, out) in mask.iter_mut().enumerate() {
                let mut n = 0usize;
                for j in 1..=self.ref_cells {
                    let d = self.guard_cells + self.ref_stride * j;
                    if let Some(i) = k.checked_sub(d) {
                        self.refs[n] = floor_db[i];
                        n += 1;
                    }
                    if k + d < self.bins {
                        self.refs[n] = floor_db[k + d];
                        n += 1;
                    }
                }
                let refs = &mut self.refs[..n];
                let (_, &mut median_db, _) = refs.select_nth_unstable_by(n / 2, f32::total_cmp);
                let n_k = 10f64.powf((median_db + self.debias_db) as f64 / 10.0);
                *out = self.linear[k] > self.alpha * n_k;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic i.i.d. exponential power samples (linear mean 1.0),
    /// reported in dB — the noise model's per-bin power distribution.
    struct ExpDb {
        state: u64,
    }

    impl ExpDb {
        fn next(&mut self) -> f32 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = ((z >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
            (10.0 * (-u.ln()).log10()) as f32
        }
    }

    #[test]
    fn rejects_bad_configs_with_named_fields() {
        let ok = CfarConfig::new();
        assert!(matches!(
            Cfar::new(0, ok.clone()),
            Err(DetectError::InvalidConfig(_))
        ));
        for pfa in [0.0, 1.0, -0.1, f64::NAN] {
            let mut c = ok.clone();
            c.pfa = pfa;
            let DetectError::InvalidConfig(msg) = Cfar::new(64, c).unwrap_err();
            assert!(msg.contains("Pfa"), "unhelpful: {msg}");
        }
        let mut c = ok.clone();
        c.ref_cells = 0;
        assert!(matches!(
            Cfar::new(64, c),
            Err(DetectError::InvalidConfig(_))
        ));
        let mut c = ok.clone();
        c.ref_stride = 0;
        assert!(matches!(
            Cfar::new(64, c),
            Err(DetectError::InvalidConfig(_))
        ));
        let mut c = ok.clone();
        c.floor_debias_db = f32::NAN;
        assert!(matches!(
            Cfar::new(64, c),
            Err(DetectError::InvalidConfig(_))
        ));
        // Guard + stride swallowing the frame leaves some bin referenceless.
        let mut c = ok;
        c.guard_cells = 60;
        c.ref_stride = 4;
        let DetectError::InvalidConfig(msg) = Cfar::new(64, c).unwrap_err();
        assert!(msg.contains("reference"), "unhelpful: {msg}");
    }

    #[test]
    fn false_alarm_rate_matches_pfa_given_an_exact_floor() {
        // With the reference floor exactly on the true level, α = −ln(Pfa)
        // must reproduce Pfa to binomial precision. 512 bins × 2000 frames
        // of i.i.d. exponential power at Pfa = 10⁻² ⇒ ~10240 expected.
        let pfa = 1e-2;
        let mut cfar = Cfar::new(
            512,
            CfarConfig {
                pfa,
                ..CfarConfig::new()
            },
        )
        .unwrap();
        let floor = vec![0.0f32; 512]; // exact: mean power 1.0 ⇔ 0 dB
        let mut gen = ExpDb { state: 99 };
        let mut frame = vec![0.0f32; 512];
        let mut mask = vec![false; 512];
        let (mut hits, mut cells) = (0u64, 0u64);
        for _ in 0..2000 {
            for v in frame.iter_mut() {
                *v = gen.next();
            }
            cfar.detect(&frame, &floor, &mut mask);
            hits += mask.iter().filter(|&&m| m).count() as u64;
            cells += mask.len() as u64;
        }
        let measured = hits as f64 / cells as f64;
        assert!(
            (measured / pfa - 1.0).abs() < 0.10,
            "measured Pfa {measured:.3e} vs configured {pfa:.0e}"
        );
    }

    #[test]
    fn a_strong_cell_is_detected_and_a_quiet_band_is_not() {
        let mut cfar = Cfar::new(256, CfarConfig::new()).unwrap();
        let frame = vec![-90.0f32; 256];
        let floor = vec![-90.0f32; 256];
        let mut mask = vec![false; 256];
        // A frame sitting exactly on the floor is below the α > 1 threshold
        // everywhere, edges included.
        cfar.detect(&frame, &floor, &mut mask);
        assert!(mask.iter().all(|&m| !m), "on-floor frame raised alarms");

        for &bin in &[0usize, 17, 128, 255] {
            let mut f = frame.clone();
            f[bin] = -50.0;
            cfar.detect(&f, &floor, &mut mask);
            assert!(mask[bin], "strong cell at bin {bin} missed");
            assert_eq!(
                mask.iter().filter(|&&m| m).count(),
                1,
                "strong cell at bin {bin} spilled into other cells"
            );
        }
    }

    #[test]
    fn strong_power_in_the_reference_window_does_not_mask_a_weak_neighbor() {
        // The D-023 property at the unit level: the threshold references
        // the robust floor, not instantaneous frame power, so a +50 dB
        // occupant 14 bins away (inside the reference span) cannot raise a
        // weak neighbor's threshold. A few contaminated *floors* nearby are
        // rejected by the reference median just the same.
        let mut cfar = Cfar::new(256, CfarConfig::new()).unwrap();
        let mut frame = vec![-90.0f32; 256];
        let mut floor = vec![-90.0f32; 256];
        frame[100] = -40.0; // strong occupant, instantaneous
        floor[99..=101].fill(-45.0); // its floor contamination (continuous case)
        frame[114] = -78.0; // weak neighbor, 12 dB over the floor
        let mut mask = vec![false; 256];
        cfar.detect(&frame, &floor, &mut mask);
        assert!(mask[100], "strong occupant missed");
        assert!(
            mask[114],
            "weak neighbor masked by strong power in its reference window"
        );
    }
}
