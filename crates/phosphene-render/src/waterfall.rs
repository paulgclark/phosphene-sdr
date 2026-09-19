// SPDX-License-Identifier: MIT

//! Waterfall row arithmetic (FR-D4, spec §7.6): speed modes, row cadence,
//! and per-row spectrum aggregation.
//!
//! §7.6 defines two speed modes (D-046, owner-accepted). They differ in what
//! the user sets:
//!
//! * **traditional** — the user sets the panel **duration** (time span); the
//!   row interval derives as `span / panel_rows`. Good for slow trends.
//! * **fast** — the user sets the **row interval** (time resolution); the
//!   panel duration derives as `rows × interval`. Good for signal structure;
//!   the 1 ms default resolves an LTE subframe exactly.
//!
//! Both modes aggregate **every** spectrum of a row's interval by max
//! (default — transients survive) or mean (D-051: a feed that samples one
//! spectrum per display frame structurally cannot deliver that, which is why
//! the aggregator runs against the pipeline's spectra stream, not the display
//! loop).
//!
//! **Time bases (D-054).** A rated source measures rows in **seconds** — one
//! spectrum advances time by `fft_size / sample_rate`. A rate-less source
//! (clarification C5) has no wall-clock mapping at all, so its cadence is
//! measured in **spectra**: fast mode is pinned to one row per spectrum (the
//! finest resolution the only available clock — the FFT itself — can state),
//! traditional mode spans a row *count*, and no seconds figure is ever
//! derived or displayed. The same rule the frequency axis follows: a display
//! never invents a unit it does not have.
//!
//! The mean aggregate is taken **in the dB domain** (an arithmetic mean of
//! `P_k` values per bin), consistent with §7.4's live-trace EMA, which the
//! spec likewise defines directly over the dBFS spectra.
//!
//! An interval that contained no spectrum at all (a counted drop gap, or a
//! single `dt` spanning several rows) emits a floor row rather than
//! repeating stale data — an empty stretch of time must *look* empty.

/// dBFS value used for rows (or bins) with no data — matches the §7.2
/// "floor at −200 dB" convention.
pub const ROW_FLOOR_DBFS: f32 = -200.0;

/// The fast mode's default row interval, seconds: 1 ms (D-046,
/// owner-accepted; tune-class). Resolves the LTE subframe exactly.
pub const FAST_INTERVAL_DEFAULT_S: f32 = 1e-3;

/// How a row folds the spectra of its interval into one value per bin (§7.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RowAggregation {
    /// Per-bin maximum — the default; transients survive.
    #[default]
    Max,
    /// Per-bin arithmetic mean of the dBFS values.
    Mean,
}

/// The §7.6 speed mode (D-046): which quantity the user sets.
///
/// **Fast is the default** (D-044/D-046, owner-ruled): at 1 ms per row the
/// display resolves the structure fast signals actually have — the owner's
/// LTE case sees congestion change subframe by subframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaterfallMode {
    /// User sets the panel duration; the row interval derives from it.
    Traditional,
    /// User sets the row interval; the panel duration derives from it.
    #[default]
    Fast,
}

/// The unit a cadence measures row intervals in (D-054): seconds for a rated
/// source, spectra for a rate-less one. Never mixed, never converted — there
/// is no conversion without a rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowTimeBase {
    /// Row intervals are seconds of signal time (`fft_size / sample_rate`
    /// per spectrum).
    Seconds,
    /// Row intervals are counted in spectra (one FFT = one unit); the only
    /// clock a rate-less source has.
    Spectra,
}

/// The §7.6 cadence: a row interval in one honest unit.
///
/// Inputs are sanitized (`panel_rows ≥ 1`, spans and intervals floored) so a
/// collapsed panel — the split-ratio endpoint of FR-D4 — can never divide by
/// zero, and a held-down key can never drive the interval degenerate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowCadence {
    interval: f32,
    base: RowTimeBase,
}

impl RowCadence {
    /// Traditional mode over a rated source: `time_span_s` seconds mapped
    /// onto `panel_rows` texture rows, so each row aggregates
    /// `time_span_s / panel_rows` seconds. Non-finite or tiny spans clamp to
    /// 1 ms; zero rows clamp to 1.
    pub fn new(time_span_s: f32, panel_rows: u32) -> Self {
        let time_span_s = if time_span_s.is_finite() {
            time_span_s.max(1e-3)
        } else {
            1.0
        };
        Self {
            interval: time_span_s / panel_rows.max(1) as f32,
            base: RowTimeBase::Seconds,
        }
    }

    /// Fast mode over a rated source: the user's `interval_s`, clamped no
    /// finer than one spectrum interval `fft_size / sample_rate` (D-046) —
    /// a row cannot resolve finer than the FFTs feeding it. A non-finite
    /// user interval falls back to the 1 ms default.
    pub fn fast(interval_s: f32, spectrum_interval_s: f32) -> Self {
        let user = if interval_s.is_finite() && interval_s > 0.0 {
            interval_s
        } else {
            FAST_INTERVAL_DEFAULT_S
        };
        let floor = if spectrum_interval_s.is_finite() && spectrum_interval_s > 0.0 {
            spectrum_interval_s
        } else {
            1e-6
        };
        Self {
            interval: user.max(floor),
            base: RowTimeBase::Seconds,
        }
    }

    /// A bare seconds-per-row cadence — for producers that state an
    /// already-derived interval (tests, the synthetic scene). Sanitized like
    /// [`RowCadence::fast`].
    pub fn seconds_per_row(interval_s: f32) -> Self {
        let interval = if interval_s.is_finite() && interval_s > 0.0 {
            interval_s
        } else {
            FAST_INTERVAL_DEFAULT_S
        };
        Self {
            interval,
            base: RowTimeBase::Seconds,
        }
    }

    /// Fast mode over a rate-less source (D-054): one row per spectrum —
    /// the finest resolution the FFT clock can state.
    pub fn fast_per_spectrum() -> Self {
        Self {
            interval: 1.0,
            base: RowTimeBase::Spectra,
        }
    }

    /// Traditional mode over a rate-less source (D-054): the span is a row
    /// **count** (`span_rows` spectra across the panel), deriving nothing
    /// from an absent rate. Each row aggregates
    /// `span_rows / panel_rows` spectra, never finer than one.
    pub fn spectra_span(span_rows: f32, panel_rows: u32) -> Self {
        let span_rows = if span_rows.is_finite() {
            span_rows.max(1.0)
        } else {
            1024.0
        };
        Self {
            interval: (span_rows / panel_rows.max(1) as f32).max(1.0),
            base: RowTimeBase::Spectra,
        }
    }

    /// The row interval in this cadence's own unit.
    pub fn interval(&self) -> f32 {
        self.interval
    }

    /// The unit the interval is measured in.
    pub fn base(&self) -> RowTimeBase {
        self.base
    }

    /// Seconds of signal time per row — `Some` only when the cadence has a
    /// seconds base. A spectra-based cadence answers `None`: a time base,
    /// like a rate, is never invented (D-054).
    pub fn row_interval_s(&self) -> Option<f32> {
        match self.base {
            RowTimeBase::Seconds => Some(self.interval),
            RowTimeBase::Spectra => None,
        }
    }

    /// Spectra per row — `Some` only for the spectra base (D-054), the
    /// figure the rate-less FR-D5 axis is labelled from. The exact dual of
    /// [`Self::row_interval_s`]: one of the two is always `Some`, never
    /// both.
    pub fn spectra_per_row(&self) -> Option<f32> {
        match self.base {
            RowTimeBase::Spectra => Some(self.interval),
            RowTimeBase::Seconds => None,
        }
    }
}

/// The one §7.6 cadence derivation (D-046 + D-054), from the active mode,
/// the source's spectrum interval (`fft_size / sample_rate`, `None` for a
/// rate-less source), the view's mode-owned settings, and the panel height.
/// Display math, kept here so a wrong cadence — which would silently rescale
/// time on the display — is testable as plain numbers.
pub fn derive_cadence(
    mode: WaterfallMode,
    spectrum_interval_s: Option<f32>,
    span_s: f32,
    fast_interval_s: f32,
    span_rows: f32,
    panel_rows: u32,
) -> RowCadence {
    match (mode, spectrum_interval_s) {
        (WaterfallMode::Traditional, Some(_)) => RowCadence::new(span_s, panel_rows),
        (WaterfallMode::Fast, Some(si)) => RowCadence::fast(fast_interval_s, si),
        (WaterfallMode::Fast, None) => RowCadence::fast_per_spectrum(),
        (WaterfallMode::Traditional, None) => RowCadence::spectra_span(span_rows, panel_rows),
    }
}

/// Folds a stream of spectra into waterfall rows at a [`RowCadence`].
///
/// Feed each spectrum with the time step it advanced **in the cadence's own
/// unit** (seconds of signal time for a rated source, exactly `1.0` per
/// spectrum for a rate-less one); completed rows are handed to the `emit`
/// callback in order. A spectrum is accumulated into the interval that is
/// open when it arrives, then the clock advances — so a spectrum arriving
/// exactly on a boundary closes its interval rather than opening the next.
#[derive(Debug)]
pub struct RowAggregator {
    bins: usize,
    mode: RowAggregation,
    cadence: RowCadence,
    /// Running per-bin max, or per-bin dB sum (mode-dependent).
    acc: Vec<f32>,
    count: u32,
    elapsed: f32,
    row: Vec<f32>,
}

impl RowAggregator {
    /// New aggregator over `bins`-wide spectra. `bins` must be non-zero.
    pub fn new(bins: usize, mode: RowAggregation, cadence: RowCadence) -> Self {
        assert!(bins > 0, "aggregator needs at least one bin");
        Self {
            bins,
            mode,
            cadence,
            acc: vec![f32::NEG_INFINITY; bins],
            count: 0,
            elapsed: 0.0,
            row: vec![ROW_FLOOR_DBFS; bins],
        }
    }

    /// The active aggregation mode.
    pub fn mode(&self) -> RowAggregation {
        self.mode
    }

    /// Spectrum width this aggregator folds.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// The cadence this aggregator is actually running.
    pub fn cadence(&self) -> RowCadence {
        self.cadence
    }

    /// Seconds of signal time each emitted row represents — the value a
    /// producer hands to the composer at the row-push seam so the FR-D5 time
    /// labels state the cadence the rows really carried (D-031). `None` for
    /// a spectra-based cadence: no seconds figure exists to state (D-054).
    pub fn row_interval_s(&self) -> Option<f32> {
        self.cadence.row_interval_s()
    }

    /// Switch max/mean. Resets the open interval — mixing the two aggregates
    /// mid-row would produce a value that is neither.
    pub fn set_mode(&mut self, mode: RowAggregation) {
        if mode != self.mode {
            self.mode = mode;
            self.reset_interval();
        }
    }

    /// Adopt a new cadence (mode, span, interval or panel height changed).
    /// A no-op when the cadence is unchanged — safe to call every frame —
    /// and resets the open interval on a real change (spectra accumulated
    /// against one interval cannot honestly finish another).
    pub fn set_cadence(&mut self, cadence: RowCadence) {
        if cadence != self.cadence {
            self.cadence = cadence;
            self.reset_interval();
        }
    }

    fn reset_interval(&mut self) {
        self.acc.fill(f32::NEG_INFINITY);
        self.count = 0;
        self.elapsed = 0.0;
    }

    /// Accumulate one spectrum, advance time by `dt` (cadence units), and
    /// emit every row whose interval completed. Panics if
    /// `spectrum.len() != bins` — a programmer error, same contract as the
    /// core accumulators.
    pub fn push_spectrum(&mut self, spectrum: &[f32], dt: f32, emit: impl FnMut(&[f32])) {
        assert_eq!(
            spectrum.len(),
            self.bins,
            "spectrum length {} != aggregator bins {}",
            spectrum.len(),
            self.bins
        );
        match self.mode {
            RowAggregation::Max => {
                for (a, &p) in self.acc.iter_mut().zip(spectrum) {
                    *a = a.max(p);
                }
            }
            RowAggregation::Mean => {
                if self.count == 0 {
                    self.acc.copy_from_slice(spectrum);
                } else {
                    for (a, &p) in self.acc.iter_mut().zip(spectrum) {
                        *a += p;
                    }
                }
            }
        }
        self.count += 1;
        self.advance(dt, emit);
    }

    /// Advance the clock by `dt` (cadence units) **without** a spectrum and
    /// emit every row whose interval completed — the open interval's spectra
    /// (if any) fold into the first row; intervals that close with no data
    /// emit the floor. This is how a counted gap — batches the pipeline shed
    /// under pressure — renders honestly as empty time instead of silently
    /// splicing the axis (§7.6, D-051's floor-on-empty contract).
    pub fn advance(&mut self, dt: f32, mut emit: impl FnMut(&[f32])) {
        if dt.is_finite() && dt > 0.0 {
            self.elapsed += dt;
        }
        while self.elapsed >= self.cadence.interval {
            self.elapsed -= self.cadence.interval;
            self.finish_row();
            emit(&self.row);
        }
    }

    /// Fold the open interval into `self.row` and reset for the next one.
    fn finish_row(&mut self) {
        if self.count == 0 {
            self.row.fill(ROW_FLOOR_DBFS);
        } else {
            match self.mode {
                RowAggregation::Max => self.row.copy_from_slice(&self.acc),
                RowAggregation::Mean => {
                    let n = self.count as f32;
                    for (r, &a) in self.row.iter_mut().zip(&self.acc) {
                        *r = a / n;
                    }
                }
            }
        }
        self.acc.fill(f32::NEG_INFINITY);
        self.count = 0;
    }
}

/// A bounded, preallocated ring of pending waterfall rows — the hand-off
/// between the compute thread (which folds every spectrum, D-051) and the
/// display frame (which drains and uploads the batch, D-047: one batched
/// upload per frame, never a GPU write per row).
///
/// When full, pushing overwrites the **oldest** pending row: the ring
/// texture downstream holds only its newest rows anyway, so a row that would
/// be overwritten there before ever being seen carries no information. No
/// allocation after construction (§7.7).
#[derive(Debug)]
pub struct RowRing {
    bins: usize,
    capacity_rows: usize,
    buf: Vec<f32>,
    /// Index of the oldest pending row.
    start: usize,
    /// Pending row count.
    len: usize,
}

impl RowRing {
    /// A ring for `capacity_rows` rows of `bins` values each. Both must be
    /// non-zero.
    pub fn new(bins: usize, capacity_rows: usize) -> Self {
        assert!(bins > 0, "row ring needs at least one bin");
        assert!(capacity_rows > 0, "row ring needs at least one row");
        Self {
            bins,
            capacity_rows,
            buf: vec![0.0; bins * capacity_rows],
            start: 0,
            len: 0,
        }
    }

    /// Spectrum width the ring stores.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Pending rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// No rows pending.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Append one row, overwriting the oldest pending row when full. Panics
    /// if `row.len() != bins` — a programmer error.
    pub fn push_row(&mut self, row: &[f32]) {
        assert_eq!(
            row.len(),
            self.bins,
            "row length {} != ring bins {}",
            row.len(),
            self.bins
        );
        let slot = (self.start + self.len) % self.capacity_rows;
        self.buf[slot * self.bins..(slot + 1) * self.bins].copy_from_slice(row);
        if self.len == self.capacity_rows {
            // Overwrote the oldest; the ring stays full.
            self.start = (self.start + 1) % self.capacity_rows;
        } else {
            self.len += 1;
        }
    }

    /// Move every pending row into `out` (appended oldest → newest, flat
    /// `rows × bins`) and clear the ring. Returns the number of rows moved.
    /// `out` should be pre-reserved to `bins × capacity_rows` once by the
    /// caller so the per-frame drain never allocates (§7.7).
    pub fn drain_into(&mut self, out: &mut Vec<f32>) -> usize {
        let rows = self.len;
        let first = (self.capacity_rows - self.start).min(self.len);
        out.extend_from_slice(&self.buf[self.start * self.bins..(self.start + first) * self.bins]);
        let rest = self.len - first;
        out.extend_from_slice(&self.buf[..rest * self.bins]);
        self.start = 0;
        self.len = 0;
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traditional_cadence_is_the_span_over_height_arithmetic() {
        // 10 s across a 500-row panel: 20 ms aggregated per row.
        let c = RowCadence::new(10.0, 500);
        assert_eq!(c.base(), RowTimeBase::Seconds);
        assert!((c.interval() - 0.02).abs() < 1e-9);
        assert!((c.row_interval_s().unwrap() - 0.02).abs() < 1e-9);

        // 2 s across 1024 rows: 512 rows/s.
        let c = RowCadence::new(2.0, 1024);
        assert!((1.0 / c.interval() - 512.0).abs() < 1e-4);

        // Interval × derived rate is exactly 1 — or time is silently
        // rescaled on the display.
        for (span, rows) in [(0.5, 720), (30.0, 1024), (3.7, 333)] {
            let c = RowCadence::new(span, rows);
            assert!(((1.0 / c.interval()) * c.interval() - 1.0).abs() < 1e-5);
            assert!((c.interval() * rows as f32 - span).abs() < span * 1e-5);
        }
    }

    #[test]
    fn cadence_survives_degenerate_inputs() {
        // The FR-D4 split endpoint can hand this a zero-height panel; it
        // must not divide by zero (lane requirement) or go non-finite.
        for c in [
            RowCadence::new(10.0, 0),
            RowCadence::new(0.0, 500),
            RowCadence::new(f32::NAN, 500),
            RowCadence::new(-3.0, 0),
            RowCadence::fast(f32::NAN, f32::NAN),
            RowCadence::fast(-1.0, 0.0),
            RowCadence::spectra_span(f32::NAN, 0),
            RowCadence::spectra_span(-5.0, 100),
        ] {
            assert!(c.interval().is_finite());
            assert!(c.interval() > 0.0);
        }
    }

    /// D-046: the fast cadence is the user's interval, clamped no finer than
    /// one spectrum interval.
    #[test]
    fn fast_cadence_clamps_to_the_spectrum_interval() {
        // 1 ms rows over a 2.048 MS/s / N=1024 source (0.5 ms spectra):
        // the user's setting stands.
        let c = RowCadence::fast(1e-3, 0.5e-3);
        assert_eq!(c.row_interval_s(), Some(1e-3));

        // The user asks finer than the FFT cadence: clamped to it exactly.
        let c = RowCadence::fast(1e-5, 0.5e-3);
        assert_eq!(c.row_interval_s(), Some(0.5e-3));
    }

    /// D-054: a rate-less source measures cadence in spectra and states no
    /// seconds figure at all.
    #[test]
    fn rateless_cadences_are_spectra_based_and_state_no_seconds() {
        let fast = RowCadence::fast_per_spectrum();
        assert_eq!(fast.base(), RowTimeBase::Spectra);
        assert_eq!(fast.interval(), 1.0, "fast: one row per spectrum");
        assert_eq!(fast.row_interval_s(), None);

        // Traditional: a 1200-spectra span over 300 panel rows = 4 spectra
        // per row; never finer than one spectrum per row.
        let trad = RowCadence::spectra_span(1200.0, 300);
        assert_eq!(trad.interval(), 4.0);
        assert_eq!(trad.row_interval_s(), None);
        assert_eq!(RowCadence::spectra_span(10.0, 300).interval(), 1.0);
    }

    /// The one derivation table (D-046 + D-054), all four cells.
    #[test]
    fn derive_cadence_covers_all_mode_and_rate_combinations() {
        let si = Some(0.5e-3); // 2.048 MS/s, N=1024
        let c = derive_cadence(WaterfallMode::Traditional, si, 30.0, 1e-3, 1024.0, 600);
        assert_eq!(c, RowCadence::new(30.0, 600));

        let c = derive_cadence(WaterfallMode::Fast, si, 30.0, 1e-3, 1024.0, 600);
        assert_eq!(c.row_interval_s(), Some(1e-3));

        let c = derive_cadence(WaterfallMode::Fast, None, 30.0, 1e-3, 1024.0, 600);
        assert_eq!(c, RowCadence::fast_per_spectrum());

        let c = derive_cadence(WaterfallMode::Traditional, None, 30.0, 1e-3, 1200.0, 300);
        assert_eq!(c, RowCadence::spectra_span(1200.0, 300));
    }

    /// Feed `n` spectra of `dt` each; collect emitted rows.
    fn run(
        agg: &mut RowAggregator,
        spectra: impl IntoIterator<Item = Vec<f32>>,
        dt: f32,
    ) -> Vec<Vec<f32>> {
        let mut rows = Vec::new();
        for s in spectra {
            agg.push_spectrum(&s, dt, |r| rows.push(r.to_vec()));
        }
        rows
    }

    #[test]
    fn max_aggregation_keeps_the_transient() {
        // 4 spectra per row interval; one carries a burst in bin 2. The
        // emitted row must hold the burst (§7.6: transients survive).
        let mut agg = RowAggregator::new(4, RowAggregation::Max, RowCadence::new(1.0, 1));
        let quiet = vec![-90.0; 4];
        let mut burst = quiet.clone();
        burst[2] = -10.0;
        let rows = run(&mut agg, [quiet.clone(), burst, quiet.clone(), quiet], 0.25);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], vec![-90.0, -90.0, -10.0, -90.0]);
    }

    #[test]
    fn mean_aggregation_averages_in_db() {
        let mut agg = RowAggregator::new(2, RowAggregation::Mean, RowCadence::new(1.0, 1));
        let rows = run(&mut agg, [vec![-80.0, -40.0], vec![-60.0, -20.0]], 0.5);
        assert_eq!(rows.len(), 1);
        assert!((rows[0][0] - -70.0).abs() < 1e-4);
        assert!((rows[0][1] - -30.0).abs() < 1e-4);
    }

    #[test]
    fn each_row_covers_exactly_its_interval() {
        // 8 s across 512 rows = 1/64 s rows; spectra every 1/256 s → 4
        // spectra per row, 12 spectra → exactly 3 rows, each the max of its
        // own quartet. (Binary-exact values, so the boundary arithmetic is
        // deterministic.)
        let mut agg = RowAggregator::new(1, RowAggregation::Max, RowCadence::new(8.0, 512));
        let spectra: Vec<Vec<f32>> = (0..12).map(|i| vec![i as f32]).collect();
        let rows = run(&mut agg, spectra, 1.0 / 256.0);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec![3.0]);
        assert_eq!(rows[1], vec![7.0]);
        assert_eq!(rows[2], vec![11.0]);
    }

    /// The spectra base runs the same arithmetic with dt = 1 per spectrum:
    /// a 4-spectra row closes after every 4th FFT (D-054).
    #[test]
    fn spectra_based_rows_close_on_spectrum_counts() {
        let mut agg = RowAggregator::new(1, RowAggregation::Max, RowCadence::spectra_span(8.0, 2));
        let spectra: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32]).collect();
        let rows = run(&mut agg, spectra, 1.0);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], vec![3.0]);
        assert_eq!(rows[1], vec![7.0]);
    }

    #[test]
    fn a_gap_emits_floor_rows_not_stale_data() {
        // One spectrum, then a dt spanning 3.5 intervals: the spectrum's own
        // row plus two floor rows — an empty stretch of time looks empty.
        let mut agg = RowAggregator::new(2, RowAggregation::Max, RowCadence::new(1.0, 1));
        let mut rows = Vec::new();
        agg.push_spectrum(&[-30.0, -50.0], 3.5, |r| rows.push(r.to_vec()));
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec![-30.0, -50.0]);
        assert_eq!(rows[1], vec![ROW_FLOOR_DBFS; 2]);
        assert_eq!(rows[2], vec![ROW_FLOOR_DBFS; 2]);
    }

    /// `advance` is the counted-drop gap path (D-051's floor-on-empty
    /// contract at the production seam): time passes with no spectra, the
    /// open interval's data folds into its own row, and the gap renders as
    /// floor rows.
    #[test]
    fn advance_without_spectra_emits_floor_rows_for_the_gap() {
        let mut agg = RowAggregator::new(1, RowAggregation::Max, RowCadence::new(1.0, 1));
        let mut rows = Vec::new();
        agg.push_spectrum(&[-40.0], 0.5, |r| rows.push(r.to_vec()));
        assert!(rows.is_empty(), "the interval is still open");
        // 2.5 intervals of counted gap: the open interval closes with its
        // real data, then one interval of pure floor.
        agg.advance(2.5, |r| rows.push(r.to_vec()));
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec![-40.0]);
        assert_eq!(rows[1], vec![ROW_FLOOR_DBFS]);
        assert_eq!(rows[2], vec![ROW_FLOOR_DBFS]);
    }

    #[test]
    fn mode_switch_resets_the_open_interval() {
        let mut agg = RowAggregator::new(1, RowAggregation::Max, RowCadence::new(1.0, 1));
        agg.push_spectrum(&[0.0], 0.25, |_| unreachable!("interval still open"));
        agg.set_mode(RowAggregation::Mean);
        // After the reset the earlier 0 dBFS spectrum must not contaminate
        // the mean.
        let rows = run(&mut agg, [vec![-40.0], vec![-60.0]], 0.5);
        assert_eq!(rows.len(), 1);
        assert!((rows[0][0] - -50.0).abs() < 1e-4);
    }

    /// `set_cadence` is called every frame by the production feed: equal
    /// cadences must not reset the open interval, changed ones must.
    #[test]
    fn set_cadence_resets_only_on_a_real_change() {
        let mut agg = RowAggregator::new(1, RowAggregation::Max, RowCadence::new(1.0, 1));
        agg.push_spectrum(&[-20.0], 0.5, |_| unreachable!("interval still open"));
        // Same cadence: the accumulated spectrum and elapsed time survive…
        agg.set_cadence(RowCadence::new(1.0, 1));
        let mut rows = Vec::new();
        agg.push_spectrum(&[-90.0], 0.5, |r| rows.push(r.to_vec()));
        assert_eq!(
            rows,
            vec![vec![-20.0]],
            "an equal cadence reset the interval"
        );
        // …a changed one resets.
        agg.push_spectrum(&[-30.0], 0.25, |_| unreachable!());
        agg.set_cadence(RowCadence::new(2.0, 1));
        rows.clear();
        agg.push_spectrum(&[-90.0], 2.0, |r| rows.push(r.to_vec()));
        assert_eq!(rows, vec![vec![-90.0]], "a changed cadence must reset");
    }

    #[test]
    fn row_ring_preserves_order_and_overwrites_oldest_when_full() {
        let mut ring = RowRing::new(2, 3);
        assert!(ring.is_empty());
        ring.push_row(&[1.0, 1.5]);
        ring.push_row(&[2.0, 2.5]);
        let mut out = Vec::with_capacity(6);
        assert_eq!(ring.drain_into(&mut out), 2);
        assert_eq!(out, vec![1.0, 1.5, 2.0, 2.5]);
        assert!(ring.is_empty());

        // Overfill: 5 rows into 3 slots keeps the newest 3, in order.
        out.clear();
        for i in 0..5 {
            ring.push_row(&[i as f32, -(i as f32)]);
        }
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.drain_into(&mut out), 3);
        assert_eq!(out, vec![2.0, -2.0, 3.0, -3.0, 4.0, -4.0]);

        // Wrap-around drain after a partial refill.
        out.clear();
        ring.push_row(&[7.0, 7.5]);
        assert_eq!(ring.drain_into(&mut out), 1);
        assert_eq!(out, vec![7.0, 7.5]);
    }
}
