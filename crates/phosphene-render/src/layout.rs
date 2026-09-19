// SPDX-License-Identifier: MIT

//! Display parameters, view state and the data-surface layout.
//!
//! Clarification C6 (spec §13 item 9) requires the data surface and its axes
//! to be custom-drawn so that axis text aligns *exactly* with the wgpu-drawn
//! grid. Both are therefore computed from one [`Layout`]: the grid geometry
//! (GPU side) and the axis label positions (egui side) read the same rects and
//! division positions, in egui points.

use egui::{Pos2, Rect};

use crate::overlay::AnnotationLevel;
use crate::waterfall::{RowAggregation, WaterfallMode};
use crate::zoom::ZoomSpan;

/// Default declutter cap N (annotation design §2.4): reused from NFR-A1's
/// "≤8 concurrent DDC channelizers" figure for a shared mental model, not
/// because the two are functionally linked. Tunable via `G`/`H`, `1..=32`.
pub const DEFAULT_DECLUTTER_CAP: u32 = 8;

/// Ceiling on the width-scaled declutter cap (D-102 item 3), matching the
/// `G`/`H` keys' own upper bound in `keymap.rs`.
pub const DECLUTTER_CAP_MAX: u32 = 32;

/// The app's default window width, points (`phosphene-app`'s windowed
/// default and its `--width` default both agree on 1280 — see
/// `phosphene-app::window` and `phosphene-app::main`). The declutter cap
/// scaling rule (D-102 item 3) is anchored to the plot grid width this
/// produces once [`Layout::compute`]'s own margins are subtracted, so "the
/// value at the default window size" is computed from the same margins the
/// grid itself uses, not a second, independently-chosen number.
pub const DEFAULT_WINDOW_WIDTH_PT: f32 = 1280.0;

/// Plot grid width, points, at [`DEFAULT_WINDOW_WIDTH_PT`] — the anchor
/// point of [`scaled_declutter_cap`]'s scaling rule.
pub const DEFAULT_GRID_WIDTH_PT: f32 = DEFAULT_WINDOW_WIDTH_PT - MARGIN_LEFT - MARGIN_RIGHT;

/// Scale the user's keyed declutter cap (D-102 item 3) to the effective cap
/// for a plot grid `grid_width_pt` points wide: `user_cap` stays the value at
/// [`DEFAULT_GRID_WIDTH_PT`] (the app's default window size) and grows in
/// direct proportion to the grid's width above that, clamped to
/// [`DECLUTTER_CAP_MAX`] and never dropping below `user_cap` itself at
/// narrower widths — the keyed value is always a floor, never something a
/// small window can shrink further.
///
/// Monotone non-decreasing in `grid_width_pt` by construction: the scaled
/// term grows with the (non-negative) ratio, the floor is a constant, and
/// clamping to a fixed ceiling preserves monotonicity of whatever it's
/// applied to.
pub fn scaled_declutter_cap(user_cap: u32, grid_width_pt: f32) -> u32 {
    if !grid_width_pt.is_finite() || grid_width_pt <= 0.0 {
        return user_cap.min(DECLUTTER_CAP_MAX);
    }
    let ratio = f64::from(grid_width_pt) / f64::from(DEFAULT_GRID_WIDTH_PT);
    let scaled = (f64::from(user_cap) * ratio).round();
    let floored = scaled.max(f64::from(user_cap));
    (floored as u32).min(DECLUTTER_CAP_MAX)
}

/// What the display shows: the dB range and the frequency span.
#[derive(Debug, Clone, Copy)]
pub struct DisplayParams {
    /// Top of the power axis, dBFS (the reference level, FR-D5).
    pub db_top: f32,
    /// Bottom of the power axis, dBFS. With `db_top` and `db_per_div` this
    /// fixes [`Self::db_divs`]; the default pair spans **100 dB at 10 dB/div
    /// — exactly 10 power divisions** (D-065 §5, reversing D-042's L3).
    pub db_bottom: f32,
    /// dB per vertical division (FR-D5).
    pub db_per_div: f32,
    /// Number of horizontal (frequency) divisions — FR-D5 asks for ~10.
    pub freq_divs: usize,
    /// Sample rate in Hz **as the source declared it** — the displayed span
    /// (complex baseband: span = rate). `None` means the source declared no
    /// rate: the frequency axis is then labelled in **normalized frequency**
    /// (cycles/sample, −0.5 … +0.5) with no Hz unit anywhere, and no center
    /// offset — a rate is never invented (clarification C5, D-028 §2). This
    /// is the explicit optional rate D-028 requires; the interim
    /// `HudStats::normalised_axis` flag it replaces is gone.
    pub sample_rate: Option<f64>,
    /// Center frequency if known. `None` labels a rated axis in relative Hz
    /// (FR-C3). Ignored entirely when `sample_rate` is `None` — a Hz center
    /// cannot be placed on a normalized axis.
    pub center: Option<f64>,
    /// The visible window of the span (FR-D6 display-side zoom, D-011).
    /// Identity ([`ZoomSpan::FULL`]) when unzoomed. Axis labels, the cursor
    /// readout and the data surfaces all read this one field, which is what
    /// keeps them in agreement at every zoom level.
    pub zoom: ZoomSpan,
    /// Seconds of signal time each waterfall row represents. **The
    /// production frame loop mirrors this each frame from
    /// [`crate::gpu::FrameComposer::waterfall_row_interval`]**, which
    /// learned it at the row-push seam — so the FR-D5 time labels state the
    /// cadence the displayed rows actually carried (D-031), never a value
    /// set by hand. `None` (no rows yet, or a producer with no honest time
    /// base) draws no time labels — a time base, like a sample rate, is
    /// never invented.
    pub wf_row_interval_s: Option<f32>,
    /// Spectra folded into each waterfall row — the **rate-less** mirror of
    /// `wf_row_interval_s` (D-054): the frame loop copies it each frame from
    /// [`crate::gpu::FrameComposer::waterfall_row_spectra`], and the FR-D5
    /// axis labels the waterfall in rows of spectra when this — and never a
    /// seconds figure — is the honest unit. At most one of the two is
    /// `Some`: a cadence has exactly one time base.
    pub wf_row_spectra: Option<f32>,
}

impl Default for DisplayParams {
    fn default() -> Self {
        Self {
            db_top: 0.0,
            // D-065 §5 (owner, from live use): the displayed range is
            // **100 dB**, 0 → −100, which is exactly **10** power divisions
            // at 10 dB/div where −110 gave 11. FR-D5 asks for "~10 grid
            // divisions" and §7.3 already writes the histogram's span as
            // `[P_ref − 10·div·dB/div, P_ref]` — the code was the odd one
            // out. Reverses D-042's L3, which was confirmed from a still
            // image; this came from operating a live radio.
            db_bottom: -100.0,
            db_per_div: 10.0,
            freq_divs: 10,
            sample_rate: None,
            center: None,
            zoom: ZoomSpan::FULL,
            wf_row_interval_s: None,
            wf_row_spectra: None,
        }
    }
}

impl DisplayParams {
    /// Number of vertical (power) divisions.
    pub fn db_divs(&self) -> usize {
        ((self.db_top - self.db_bottom) / self.db_per_div)
            .round()
            .max(1.0) as usize
    }

    /// Map a power in dBFS to a normalized vertical position: 1.0 at
    /// `db_top`, 0.0 at `db_bottom`, clamped.
    pub fn db_to_unit(&self, db: f32) -> f32 {
        ((db - self.db_bottom) / (self.db_top - self.db_bottom)).clamp(0.0, 1.0)
    }
}

/// Everything the user can steer about the view, in one place: the axis
/// parameters (including zoom), the FR-D4 split, pause and the help overlay.
///
/// Owned by the caller (the app holds it across frames), mutated only by
/// [`crate::chrome::draw`] — the keyboard map and mouse gestures of FR-D6/
/// FR-D7 apply here, and the composer reads the same struct, so what the
/// labels claim and what the surfaces draw cannot diverge.
#[derive(Debug, Clone, Copy)]
pub struct ViewState {
    /// Axis parameters — dB range, span, zoom.
    pub params: DisplayParams,
    /// Histogram/waterfall split ratio (FR-D4): 0 = waterfall only,
    /// 1 = histogram only. Sanitized by [`Layout::split`]; any value is safe.
    /// Defaults to [`DEFAULT_SPLIT`] — **both** surfaces visible on first
    /// run (D-040, owner ruling): a feature the default view hides is not
    /// delivered.
    pub split: f32,
    /// FR-D3 max-hold trace visibility. Defaults to shown: spec §2 asks for
    /// simultaneous live, max-hold and waterfall from the first frame. When
    /// hidden, the frame loop hands `RtsaInput` an empty trace ("empty means
    /// not shown") while the accumulator keeps running — like the waterfall
    /// while its panel is closed, the history is there when it reappears.
    pub max_hold: bool,
    /// The D-007 max-hold mode: `true` is decay-toward-live (the default),
    /// `false` is pure hold + reset. The frame loop maps this onto the
    /// accumulator's own mode every frame.
    pub max_hold_decay: bool,
    /// The §7.6 speed mode (D-046): traditional (the user sets the span) or
    /// fast (the user sets the row interval). **Fast is the default**
    /// (owner-ruled, D-044/D-046). Toggled by the F key; the frame loop
    /// hands it to the waterfall feed, which derives the cadence.
    pub wf_mode: WaterfallMode,
    /// FR-D4 waterfall time span, seconds — **traditional** mode's setting
    /// over a rated source (§7.6: the row cadence derives from this and the
    /// panel height). Stepped by the `-`/`=` keys when traditional owns
    /// them (D-046).
    pub wf_time_span_s: f32,
    /// **Fast** mode's setting over a rated source: seconds of signal time
    /// per row (D-046; default 1 ms — resolves an LTE subframe). Stepped by
    /// the `-`/`=` keys when fast owns them; the feed clamps it no finer
    /// than one spectrum interval.
    pub wf_fast_interval_s: f32,
    /// **Traditional** mode's setting over a rate-less source: the span as
    /// a row count — `span_rows` spectra across the panel (D-054; nothing
    /// is derived from an absent rate). Stepped by the `-`/`=` keys when it
    /// is the owned quantity.
    pub wf_span_rows: f32,
    /// FR-D4/§7.6 row aggregation: max (the default — transients survive) or
    /// mean, toggled by the A key (M1-J). The frame loop maps this onto the
    /// aggregator's own mode every frame, like `max_hold_decay` above.
    pub wf_aggregation: RowAggregation,
    /// FR-D4/§7.6 waterfall independent auto-range: on (**the default**
    /// since D-065 §2), the waterfall's colour scale spans the retained
    /// rows; off, it shares the histogram's displayed dB range. Toggled by
    /// the W key (M1-J); the frame loop maps it onto the composer every
    /// frame.
    ///
    /// The honesty cost of the new default is already paid: an auto-ranged
    /// waterfall no longer agrees with the dB axis above it, and
    /// [`crate::chrome`]'s `wf` readout appends `auto` exactly when this is
    /// on, so the display states which mapping it is using.
    pub wf_auto_range: bool,
    /// §7.6 waterfall **intensity** exponent `γ_wf`: the power law applied
    /// to the row's normalised power before the colormap lookup
    /// (`colour = cmap(t^γ_wf)`). Below 1 it lifts low-power detail out of
    /// the floor; above 1 it keeps only the strongest returns; 1.0 — the
    /// default — is the linear mapping the waterfall had before the control
    /// existed.
    ///
    /// **A separate quantity from §7.3's `gamma_adjust`, deliberately**
    /// (D-065 §3 as corrected by the owner): that exponent is read only by
    /// the histogram's fragment shader and has never reached the waterfall,
    /// which is why `PgUp`/`PgDn` appeared to do nothing to it. This one is
    /// read only by the waterfall's.
    ///
    /// Stepped by `Shift+PgUp`/`Shift+PgDn` and by the chrome's intensity
    /// buttons; the frame loop mirrors it onto the composer every frame,
    /// exactly as it does `wf_auto_range`.
    pub wf_gamma: f32,
    /// FR-D12 display freeze: the composer stops taking new data while the
    /// chrome (and this state) stays live. The source keeps draining.
    pub paused: bool,
    /// Whether the FR-D7 `?` help overlay is open.
    pub help_open: bool,
    /// In-progress drag-select zoom: the x position (egui points) where the
    /// drag started, while the button is down. Interaction state, not policy;
    /// lives here so the whole view remains one plain `Copy` value.
    pub drag_from: Option<f32>,
    /// Annotation design §2.7 `N` key: full (labels + shade) or shade-only.
    pub annotation_level: AnnotationLevel,
    /// Annotation design §2.4/§2.7 `G`/`H` keys: the declutter cap N,
    /// `1..=32`, default [`DEFAULT_DECLUTTER_CAP`].
    pub declutter_cap: u32,
    /// D-102 item 1 / D-105: which of the two label styles the overlay
    /// draws. **Not a keymap-bound, user-facing setting** — the owner chose
    /// [`crate::overlay::LabelStyle::Stacked`] as the default (D-105); the
    /// only way to reach the single-line alternative is a caller setting
    /// this field directly, which is exactly what the look-review command
    /// does to keep producing a single-line set for comparison.
    pub label_style: crate::overlay::LabelStyle,
}

/// Default histogram/waterfall split (D-040): the persistence histogram —
/// FR-D1's centerpiece — keeps just under two thirds of the surface, and
/// the waterfall gets a clearly readable band below it from the first
/// frame. Owner-ruled that both surfaces show by default; the exact ratio
/// is a tune-class starting value for the M1 visual pass.
pub const DEFAULT_SPLIT: f32 = 0.65;

impl Default for ViewState {
    fn default() -> Self {
        Self {
            params: DisplayParams::default(),
            split: DEFAULT_SPLIT,
            max_hold: true,
            max_hold_decay: true,
            wf_mode: WaterfallMode::default(),
            wf_time_span_s: 30.0,
            wf_fast_interval_s: crate::waterfall::FAST_INTERVAL_DEFAULT_S,
            // D-054: the rate-less traditional span defaults to one ring's
            // worth of spectra — the §7.6 H, not a number derived from any
            // rate.
            wf_span_rows: crate::surface::RING_ROWS as f32,
            wf_aggregation: RowAggregation::Max,
            // D-065 §2 (owner, from live use): auto-range ON by default —
            // it reverses D-042's F6, and it is safe because the `wf`
            // readout says `auto` whenever it is on.
            wf_auto_range: true,
            wf_gamma: crate::surface::DEFAULT_WF_GAMMA,
            paused: false,
            help_open: false,
            drag_from: None,
            annotation_level: AnnotationLevel::Full,
            declutter_cap: DEFAULT_DECLUTTER_CAP,
            label_style: crate::overlay::LabelStyle::default(),
        }
    }
}

/// The screen regions of one frame, in egui points.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    /// The grid rect — the wgpu data surface. Axis labels hang off its edges.
    pub grid: Rect,
}

/// Margins reserved inside the central area for axis labels, in points.
const MARGIN_LEFT: f32 = 56.0;
const MARGIN_RIGHT: f32 = 16.0;
const MARGIN_TOP: f32 = 10.0;
const MARGIN_BOTTOM: f32 = 24.0;

impl Layout {
    /// Compute the layout from the rect egui leaves between the chrome
    /// panels.
    pub fn compute(central: Rect) -> Self {
        let grid = Rect::from_min_max(
            Pos2::new(central.min.x + MARGIN_LEFT, central.min.y + MARGIN_TOP),
            Pos2::new(central.max.x - MARGIN_RIGHT, central.max.y - MARGIN_BOTTOM),
        );
        Self { grid }
    }

    /// X position (points) of vertical grid line `i` in `0..=freq_divs`.
    pub fn freq_div_x(&self, i: usize, freq_divs: usize) -> f32 {
        self.grid.min.x + self.grid.width() * (i as f32 / freq_divs as f32)
    }

    /// Y position (points) of horizontal grid line `i` in `0..=db_divs`,
    /// counted from the top (i = 0 is the reference level).
    pub fn db_div_y(&self, i: usize, db_divs: usize) -> f32 {
        self.grid.min.y + self.grid.height() * (i as f32 / db_divs as f32)
    }

    /// Split the data surface into histogram (top) and waterfall (bottom)
    /// regions by `ratio` (FR-D4): 0 is waterfall-only, 1 is histogram-only,
    /// and every value between is reachable by continuous drag. Both regions
    /// are sub-rects of `self.grid` — the waterfall is a region of this same
    /// layout, never a second layout computed alongside it (M0-C's C6
    /// alignment guarantee extends to it).
    ///
    /// The endpoints are exact: at 0 the waterfall gets the *full* grid rect
    /// and the histogram collapses to a zero-height rect at the top edge (and
    /// symmetrically at 1). Collapsed rects are valid geometry — callers skip
    /// drawing a region with no area rather than dividing by it. `ratio` is
    /// sanitized (clamped to `[0, 1]`; non-finite reads as 1), so no input
    /// can panic or produce a degenerate viewport.
    pub fn split(&self, ratio: f32) -> SplitRegions {
        let r = if ratio.is_finite() {
            ratio.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let g = self.grid;
        if r <= 0.0 {
            return SplitRegions {
                histogram: Rect::from_min_max(g.min, Pos2::new(g.max.x, g.min.y)),
                waterfall: g,
            };
        }
        if r >= 1.0 {
            return SplitRegions {
                histogram: g,
                waterfall: Rect::from_min_max(Pos2::new(g.min.x, g.max.y), g.max),
            };
        }
        let gap = SPLIT_GAP.min(g.height() * 0.25).max(0.0);
        let avail = (g.height() - gap).max(0.0);
        let hist_h = avail * r;
        SplitRegions {
            histogram: Rect::from_min_max(g.min, Pos2::new(g.max.x, g.min.y + hist_h)),
            waterfall: Rect::from_min_max(Pos2::new(g.min.x, g.max.y - (avail - hist_h)), g.max),
        }
    }

    /// The inverse of [`split`](Self::split) for the FR-D4 divider drag
    /// (M1-J): the split ratio whose divider midpoint sits at height `y`
    /// (egui points). Exact round trip for interior ratios; y at or beyond
    /// the grid edges clamps to the 0/1 endpoints, so the full closed range
    /// is reachable by continuous drag. Non-finite `y` reads as 1 (the same
    /// sanitization stance as `split`).
    pub fn split_ratio_at_y(&self, y: f32) -> f32 {
        if !y.is_finite() {
            return 1.0;
        }
        let g = self.grid;
        let gap = SPLIT_GAP.min(g.height() * 0.25).max(0.0);
        let avail = (g.height() - gap).max(1e-3);
        ((y - g.min.y - gap * 0.5) / avail).clamp(0.0, 1.0)
    }
}

/// Vertical gap between the histogram and waterfall panels when both are
/// visible, in points.
const SPLIT_GAP: f32 = 6.0;

/// The two FR-D4 data-surface regions produced by [`Layout::split`], in egui
/// points.
#[derive(Debug, Clone, Copy)]
pub struct SplitRegions {
    /// Persistence histogram region (top). Zero-height at ratio 0.
    pub histogram: Rect,
    /// Waterfall region (bottom). Zero-height at ratio 1.
    pub waterfall: Rect,
}

/// Compact frequency label: `-960K`, `0`, `+1.2M`, `+2.45G`.
///
/// Two significant decimals — the span/status readout format. Axis tick
/// labels use [`crate::axis::format_freq_tick`] instead, whose precision
/// follows the tick spacing so zoomed ticks stay distinct.
pub fn format_hz(hz: f64) -> String {
    if hz == 0.0 {
        return "0".to_owned();
    }
    let sign = if hz > 0.0 { "+" } else { "-" };
    let a = hz.abs();
    let (value, unit) = if a >= 1e9 {
        (a / 1e9, "G")
    } else if a >= 1e6 {
        (a / 1e6, "M")
    } else if a >= 1e3 {
        (a / 1e3, "K")
    } else {
        (a, "")
    };
    // Two significant decimals, trailing zeros trimmed.
    let s = format!("{value:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    format!("{sign}{s}{unit}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_are_the_fr_d5_shape() {
        let p = DisplayParams::default();
        // D-065 §5: 0 → −100 at 10 dB/div is exactly ten power divisions —
        // FR-D5's "~10 grid divisions", and §7.3's own `[P_ref − 10·div·
        // dB/div, P_ref]`.
        assert_eq!(p.db_divs(), 10);
        assert_eq!(p.freq_divs, 10);
        assert_eq!(p.db_to_unit(0.0), 1.0);
        assert_eq!(p.db_to_unit(-100.0), 0.0);
        assert!((p.db_to_unit(-50.0) - 0.5).abs() < 1e-6);
        // Out-of-range powers clamp instead of leaving the surface.
        assert_eq!(p.db_to_unit(20.0), 1.0);
        assert_eq!(p.db_to_unit(-300.0), 0.0);
        // D-028: the honest default is "no rate declared" — a rate only ever
        // comes from the source or the CLI, never from a default.
        assert_eq!(p.sample_rate, None);
        assert_eq!(p.center, None);
        assert!(p.zoom.is_full());
        assert_eq!(p.wf_row_interval_s, None);
    }

    #[test]
    fn layout_nests_inside_central() {
        let central = Rect::from_min_max(Pos2::new(0.0, 20.0), Pos2::new(1280.0, 700.0));
        let l = Layout::compute(central);
        assert!(central.contains_rect(l.grid));
        // Division endpoints hit the grid edges exactly — the C6 alignment
        // contract between GPU grid and egui labels.
        assert_eq!(l.freq_div_x(0, 10), l.grid.min.x);
        assert_eq!(l.freq_div_x(10, 10), l.grid.max.x);
        assert_eq!(l.db_div_y(0, 11), l.grid.min.y);
        assert_eq!(l.db_div_y(11, 11), l.grid.max.y);
    }

    #[test]
    fn split_endpoints_are_exact_and_safe() {
        let l = Layout::compute(Rect::from_min_max(
            Pos2::new(0.0, 20.0),
            Pos2::new(1280.0, 700.0),
        ));
        // Ratio 0: waterfall-only — the waterfall takes the whole grid and
        // the histogram has no area (FR-D4's closed range includes both
        // endpoints).
        let s = l.split(0.0);
        assert_eq!(s.waterfall, l.grid);
        assert_eq!(s.histogram.height(), 0.0);
        assert_eq!(s.histogram.width(), l.grid.width());
        // Ratio 1: histogram-only.
        let s = l.split(1.0);
        assert_eq!(s.histogram, l.grid);
        assert_eq!(s.waterfall.height(), 0.0);
        // Out-of-range and non-finite inputs are sanitized, never panic.
        for r in [-3.0, 2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let s = l.split(r);
            assert!(s.histogram.height().is_finite());
            assert!(s.waterfall.height().is_finite());
        }
    }

    #[test]
    fn split_interior_partitions_the_grid() {
        let l = Layout::compute(Rect::from_min_max(
            Pos2::new(0.0, 20.0),
            Pos2::new(1280.0, 700.0),
        ));
        for r in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let s = l.split(r);
            assert!(l.grid.contains_rect(s.histogram));
            assert!(l.grid.contains_rect(s.waterfall));
            // Top edge of the histogram and bottom edge of the waterfall pin
            // to the grid; the two panels never overlap.
            assert_eq!(s.histogram.min.y, l.grid.min.y);
            assert_eq!(s.waterfall.max.y, l.grid.max.y);
            assert!(s.histogram.max.y <= s.waterfall.min.y);
            // Heights track the ratio over the available (grid − gap) space.
            let avail = s.histogram.height() + s.waterfall.height();
            assert!((s.histogram.height() - avail * r).abs() < 1e-3);
        }
    }

    /// D-040 (owner): the default view gives BOTH surfaces real area — the
    /// waterfall is visible on first run, and the histogram keeps the larger
    /// share (FR-D1 is the centerpiece). The rendered half of this seal —
    /// lit waterfall pixels at the default view — is in the app's headless
    /// tests, which drive the production loop.
    #[test]
    fn default_split_shows_both_surfaces() {
        let split = ViewState::default().split;
        assert!(
            (0.5..1.0).contains(&split),
            "default split {split} must show both surfaces, histogram-major"
        );
        let l = Layout::compute(Rect::from_min_max(
            Pos2::new(0.0, 20.0),
            Pos2::new(1280.0, 700.0),
        ));
        let s = l.split(split);
        assert!(s.histogram.height() > 100.0, "histogram area is real");
        assert!(s.waterfall.height() > 100.0, "waterfall area is real");
        assert!(s.histogram.height() > s.waterfall.height());
    }

    /// D-105 (owner ruling, superseding D-102 item 1's "single line stays
    /// the default"): stacked labels are the production default. The
    /// single-line style stays reachable, but only a caller (the
    /// look-review command) setting `label_style` directly, never
    /// `ViewState::default()`.
    #[test]
    fn default_label_style_is_stacked() {
        assert_eq!(
            ViewState::default().label_style,
            crate::overlay::LabelStyle::Stacked
        );
    }

    /// The drag inverse (M1-J): the ratio recovered from a divider's own
    /// midpoint is the ratio that placed it there, and the grid edges clamp
    /// to the exact FR-D4 endpoints.
    #[test]
    fn split_ratio_at_y_inverts_split() {
        let l = Layout::compute(Rect::from_min_max(
            Pos2::new(0.0, 20.0),
            Pos2::new(1280.0, 700.0),
        ));
        for r in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] {
            let s = l.split(r);
            let mid = (s.histogram.max.y + s.waterfall.min.y) * 0.5;
            assert!(
                (l.split_ratio_at_y(mid) - r).abs() < 1e-5,
                "round trip failed at ratio {r}"
            );
        }
        // Edges clamp to the closed endpoints; garbage is sanitized.
        assert_eq!(l.split_ratio_at_y(l.grid.min.y), 0.0);
        assert_eq!(l.split_ratio_at_y(l.grid.min.y - 100.0), 0.0);
        assert_eq!(l.split_ratio_at_y(l.grid.max.y), 1.0);
        assert_eq!(l.split_ratio_at_y(l.grid.max.y + 100.0), 1.0);
        assert_eq!(l.split_ratio_at_y(f32::NAN), 1.0);
    }

    #[test]
    fn hz_formatting() {
        assert_eq!(format_hz(0.0), "0");
        assert_eq!(format_hz(480_000.0), "+480K");
        assert_eq!(format_hz(-960_000.0), "-960K");
        assert_eq!(format_hz(1_200_000.0), "+1.2M");
        assert_eq!(format_hz(-2_450_000_000.0), "-2.45G");
        assert_eq!(format_hz(250.0), "+250");
    }

    // --- D-102 item 3: the width-scaled declutter cap ----------------------

    #[test]
    fn scaled_cap_is_the_user_value_exactly_at_the_default_width() {
        assert_eq!(
            scaled_declutter_cap(DEFAULT_DECLUTTER_CAP, DEFAULT_GRID_WIDTH_PT),
            DEFAULT_DECLUTTER_CAP
        );
        // Holds for any keyed value, not just the constant's own default.
        for user_cap in [1, 5, 12, 32] {
            assert_eq!(
                scaled_declutter_cap(user_cap, DEFAULT_GRID_WIDTH_PT),
                user_cap
            );
        }
    }

    #[test]
    fn scaled_cap_never_drops_below_the_keyed_value_at_narrower_widths() {
        for width in [
            1.0,
            100.0,
            DEFAULT_GRID_WIDTH_PT * 0.5,
            DEFAULT_GRID_WIDTH_PT - 1.0,
        ] {
            assert_eq!(scaled_declutter_cap(8, width), 8);
            assert_eq!(scaled_declutter_cap(20, width), 20);
        }
    }

    #[test]
    fn scaled_cap_grows_past_the_default_width_and_clamps_at_the_ceiling() {
        assert!(scaled_declutter_cap(8, DEFAULT_GRID_WIDTH_PT * 2.0) > 8);
        assert_eq!(
            scaled_declutter_cap(8, DEFAULT_GRID_WIDTH_PT * 100.0),
            DECLUTTER_CAP_MAX
        );
        assert_eq!(
            scaled_declutter_cap(DECLUTTER_CAP_MAX, DEFAULT_GRID_WIDTH_PT * 2.0),
            DECLUTTER_CAP_MAX
        );
    }

    #[test]
    fn scaled_cap_is_monotone_non_decreasing_as_width_grows() {
        let widths: Vec<f32> = (0..40)
            .map(|i| 50.0 + i as f32 * (DEFAULT_GRID_WIDTH_PT * 3.0 / 40.0))
            .collect();
        for user_cap in [1, 8, 32] {
            let mut prev = 0;
            for &w in &widths {
                let cap = scaled_declutter_cap(user_cap, w);
                assert!(cap >= prev, "cap dropped from {prev} to {cap} at width {w}");
                prev = cap;
            }
        }
    }

    #[test]
    fn scaled_cap_handles_degenerate_widths_without_panicking() {
        for w in [0.0, -10.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(scaled_declutter_cap(8, w), 8);
            assert_eq!(scaled_declutter_cap(40, w), DECLUTTER_CAP_MAX);
        }
    }
}
