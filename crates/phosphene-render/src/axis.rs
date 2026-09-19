// SPDX-License-Identifier: MIT

//! Axis, tick and cursor-readout arithmetic (FR-D5, FR-D6) — pure math,
//! deliberately free of GPU and egui-context types so every number the
//! instrument displays can be tested headlessly.
//!
//! One mapping rules everything here: a horizontal fraction `t` across the
//! grid passes through the zoom window ([`crate::zoom::ZoomSpan`]) to a
//! fraction of the full span, and from there to a frequency. The axis labels
//! ([`freq_ticks`]) and the cursor readout ([`cursor_readout`]) both call
//! [`freq_at`], so they cannot disagree — the divergence of those two paths
//! is the bug this module's structure exists to prevent.
//!
//! Units are honest by construction (D-028 §2, clarification C5): with no
//! declared sample rate every frequency here is a [`FreqValue::Normalized`]
//! cycles/sample fraction, and no formatting path attaches a Hz-flavoured
//! unit to it.

use egui::{Pos2, Rect};

use crate::layout::{DisplayParams, SplitRegions};

/// A frequency the display can state: real Hz (rate declared) or normalized
/// cycles/sample (no rate declared — D-028 §2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FreqValue {
    /// Absolute or relative frequency in Hz (relative when no center is
    /// declared, FR-C3).
    Hz(f64),
    /// Normalized frequency in cycles/sample, −0.5 … +0.5.
    Normalized(f64),
}

/// The frequency at horizontal fraction `t` (0 = left grid edge, 1 = right)
/// of the *visible* (zoomed) axis.
pub fn freq_at(params: &DisplayParams, t: f64) -> FreqValue {
    let u = params.zoom.view_to_full(t);
    match params.sample_rate {
        Some(rate) => FreqValue::Hz((u - 0.5) * rate + params.center.unwrap_or(0.0)),
        // No center on a normalized axis: a Hz offset cannot be placed on a
        // cycles/sample scale.
        None => FreqValue::Normalized(u - 0.5),
    }
}

/// The power in dBFS at vertical fraction `v` of the histogram region
/// (0 = top edge = reference level, 1 = bottom edge).
pub fn db_at(params: &DisplayParams, v: f32) -> f32 {
    params.db_top - v * (params.db_top - params.db_bottom)
}

/// The span of the visible (zoomed) window, in the axis's own unit — Hz when
/// a rate is declared, cycles/sample otherwise.
pub fn visible_span(params: &DisplayParams) -> f64 {
    let w = params.zoom.width();
    match params.sample_rate {
        Some(rate) => rate * w,
        None => w,
    }
}

/// One labelled frequency tick.
#[derive(Debug, Clone, PartialEq)]
pub struct FreqTick {
    /// Grid division index (`0..=freq_divs`).
    pub div: usize,
    /// Horizontal fraction of the grid, `div / freq_divs`.
    pub x_fraction: f64,
    /// The frequency this tick sits at.
    pub value: FreqValue,
    /// Its label text. Never contains a Hz unit on a normalized axis.
    pub label: String,
}

/// The labelled frequency ticks for the current view: every other division
/// (odd indices), so labels have room and, unzoomed, DC gets one. Label
/// precision follows the spacing between labelled ticks, so ticks stay
/// distinct at every zoom level.
pub fn freq_ticks(params: &DisplayParams) -> Vec<FreqTick> {
    if params.freq_divs == 0 {
        return Vec::new();
    }
    // Spacing between adjacent *labelled* ticks (every 2nd division), in the
    // axis unit.
    let step = visible_span(params) * 2.0 / params.freq_divs as f64;
    (1..params.freq_divs)
        .step_by(2)
        .map(|i| {
            let x_fraction = i as f64 / params.freq_divs as f64;
            let value = freq_at(params, x_fraction);
            let label = match value {
                FreqValue::Hz(hz) => format_freq_tick(hz, step),
                FreqValue::Normalized(v) => format_norm_tick(v, step),
            };
            FreqTick {
                div: i,
                x_fraction,
                value,
                label,
            }
        })
        .collect()
}

/// Label for one power-axis division line. Integer dB renders with no
/// decimals; fractional reference levels keep one.
pub fn db_label(db: f32) -> String {
    if db.fract() == 0.0 {
        format!("{db:.0}")
    } else {
        format!("{db:.1}")
    }
}

/// Number of decimals needed so that values `resolution` apart stay distinct
/// when shown in a unit of scale `unit`.
fn decimals_for(resolution: f64, unit: f64) -> usize {
    if resolution <= 0.0 || unit <= 0.0 || !resolution.is_finite() || !unit.is_finite() {
        return 2;
    }
    let d = -(resolution / unit).log10();
    d.ceil().clamp(0.0, 9.0) as usize
}

/// Pick the K/M/G display unit for a magnitude.
fn hz_unit(a: f64) -> (f64, &'static str) {
    if a >= 1e9 {
        (1e9, "G")
    } else if a >= 1e6 {
        (1e6, "M")
    } else if a >= 1e3 {
        (1e3, "K")
    } else {
        (1.0, "")
    }
}

/// Signed, trimmed decimal string; `""` sign for zero. Values that round to
/// zero at this precision display as `0`. Only fractional trailing zeros are
/// trimmed — integer zeros are significant digits.
fn signed_trimmed(value: f64, decimals: usize) -> String {
    let mut s = format!("{:.*}", decimals, value.abs());
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_owned();
    }
    if s == "0" {
        return "0".to_owned();
    }
    let sign = if value > 0.0 { "+" } else { "-" };
    format!("{sign}{s}")
}

/// Hz-axis tick label whose precision follows the tick spacing `step_hz`:
/// `-960K`, `0`, `+2.4501G`. Adjacent ticks a `step_hz` apart always render
/// distinct — a tick that collides with its neighbour misstates frequency.
pub fn format_freq_tick(hz: f64, step_hz: f64) -> String {
    let (unit, suffix) = hz_unit(hz.abs().max(step_hz.abs()));
    // One guard digit beyond the distinctness minimum, so a tick label
    // states its value to a tenth of the tick spacing (trailing zeros trim
    // away on round values).
    let d = (decimals_for(step_hz.abs(), unit) + 1).min(9);
    let s = signed_trimmed(hz / unit, d);
    if s == "0" {
        s
    } else {
        format!("{s}{suffix}")
    }
}

/// Normalized-axis tick label (cycles/sample): a plain signed fraction with
/// **no unit** — `-0.4`, `0`, `+0.2134` (D-028 §2: no Hz anywhere, and no
/// invented unit in its place).
pub fn format_norm_tick(v: f64, step: f64) -> String {
    let d = (decimals_for(step.abs(), 1.0) + 1).clamp(1, 9);
    signed_trimmed(v, d)
}

/// Cursor-readout frequency string. Precision resolves ~1/1000 of the
/// visible span (roughly per-pixel), fixed-width so the readout doesn't
/// jitter as the pointer moves. Normalized values stay bare fractions.
pub fn format_freq_readout(value: FreqValue, params: &DisplayParams) -> String {
    let res = (visible_span(params) / 1000.0).abs();
    match value {
        FreqValue::Hz(hz) => {
            let (unit, suffix) = hz_unit(hz.abs().max(res));
            let d = decimals_for(res, unit);
            let sign = if hz < 0.0 { "-" } else { "+" };
            format!("{sign}{:.*}{suffix}", d, (hz / unit).abs())
        }
        FreqValue::Normalized(v) => {
            let d = decimals_for(res, 1.0).max(1);
            let sign = if v < 0.0 { "-" } else { "+" };
            format!("{sign}{:.*}", d, v.abs())
        }
    }
}

/// What the cursor readout states for a pointer position, as **values** —
/// the chrome formats them. `None` when the pointer is outside the data
/// regions: a readout that is confidently wrong (a clamped edge value) is
/// worse than none (FR-D6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Readout {
    /// Frequency under the pointer, by the same mapping the axis labels use.
    pub freq: FreqValue,
    /// Power in dBFS at the pointer height — only inside the histogram
    /// region, where the vertical axis *is* power.
    pub power_dbfs: Option<f32>,
    /// Time before "now" at the pointer height — only inside the waterfall
    /// region, and only when a row interval is declared (never invented).
    pub time_s: Option<f32>,
    /// Rows of spectra before "now" at the pointer height — the waterfall
    /// region of a **rate-less** source, whose only clock is the FFT itself
    /// (D-054): stated in the axis's own row unit, never converted to
    /// seconds.
    pub time_rows: Option<f32>,
}

/// Compute the cursor readout for `pos` (egui points). `regions` must come
/// from the same [`crate::layout::Layout::split`] the frame was drawn with —
/// one geometry, read once. `pixels_per_point` converts waterfall rows
/// (physical pixels) to points for the time readout.
pub fn cursor_readout(
    params: &DisplayParams,
    grid: Rect,
    regions: &SplitRegions,
    pos: Pos2,
    pixels_per_point: f32,
) -> Option<Readout> {
    if grid.width() <= 0.0 || !pos.x.is_finite() || !pos.y.is_finite() {
        return None;
    }
    let in_hist = regions.histogram.height() > 0.0 && regions.histogram.contains(pos);
    let in_wf = regions.waterfall.height() > 0.0 && regions.waterfall.contains(pos);
    if !in_hist && !in_wf {
        // Outside the data regions (including the split gap): no readout,
        // never a clamped one.
        return None;
    }
    let t = f64::from((pos.x - grid.min.x) / grid.width());
    let freq = freq_at(params, t);
    if in_hist {
        let v = (pos.y - regions.histogram.min.y) / regions.histogram.height();
        Some(Readout {
            freq,
            power_dbfs: Some(db_at(params, v)),
            time_s: None,
            time_rows: None,
        })
    } else {
        let rows = (pos.y - regions.waterfall.min.y) * pixels_per_point;
        // Seconds when the rows carried a seconds cadence; rows of spectra
        // when they carried a spectra cadence (D-054); nothing when no rows
        // have flowed — a time base is never invented.
        let time_s = params.wf_row_interval_s.map(|interval| rows * interval);
        let time_rows = if time_s.is_none() {
            params.wf_row_spectra.map(|spectra| rows * spectra)
        } else {
            None
        };
        Some(Readout {
            freq,
            power_dbfs: None,
            time_s,
            time_rows,
        })
    }
}

/// One labelled waterfall time tick (FR-D5).
#[derive(Debug, Clone, PartialEq)]
pub struct TimeTick {
    /// Row offset from the top of the waterfall region (physical pixels —
    /// one ring row per pixel row).
    pub row: f32,
    /// Label text, e.g. `-2s`, `-0.5s`.
    pub label: String,
}

/// Time ticks for a waterfall region `rows_visible` rows tall at
/// `interval_s` seconds per row: 1-2-5 stepped, newest row (top) is time
/// zero. Empty when the inputs cannot support an honest time base.
pub fn time_ticks(interval_s: f32, rows_visible: f32) -> Vec<TimeTick> {
    stepped_ticks(interval_s, rows_visible, "s")
}

/// Row-count ticks for a **rate-less** waterfall (D-054): `spectra_per_row`
/// spectra folded into each panel row, labelled in the axis's own unit —
/// rows of spectra (`-200r`), the same unit the `wf` status readout's
/// row-count span uses — because no seconds figure exists and one is never
/// invented. Same 1-2-5 stepping as the seconds axis.
pub fn row_ticks(spectra_per_row: f32, rows_visible: f32) -> Vec<TimeTick> {
    stepped_ticks(spectra_per_row, rows_visible, "r")
}

/// The shared FR-D5 tick generator: 1-2-5 stepped values over
/// `interval × rows_visible` of the axis's unit, newest row (top) at zero.
fn stepped_ticks(interval: f32, rows_visible: f32, unit: &str) -> Vec<TimeTick> {
    if interval <= 0.0 || rows_visible < 1.0 || !interval.is_finite() || !rows_visible.is_finite() {
        return Vec::new();
    }
    let span = f64::from(interval) * f64::from(rows_visible);
    // Smallest 1-2-5 step giving at most ~5 labels.
    let raw = span / 5.0;
    let e = raw.log10().floor();
    let base = 10f64.powf(e);
    let step = [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|m| m * base)
        .find(|s| *s >= raw)
        .unwrap_or(base * 10.0);
    let decimals = decimals_for(step, 1.0);
    let mut ticks = Vec::new();
    let mut t = step;
    // Stop shy of the bottom edge so the last label isn't clipped.
    while t < span * 0.98 {
        let s = format!("{t:.decimals$}");
        // Trim only fractional zeros: "0.50" → "0.5". An integer like "10"
        // must keep its zeros — trimming them relabelled −10 s as −1 s, a
        // silently rescaled time axis (found by the D-054 row-tick seal).
        let s = if decimals > 0 {
            s.trim_end_matches('0').trim_end_matches('.')
        } else {
            &s
        };
        ticks.push(TimeTick {
            row: (t / f64::from(interval)) as f32,
            label: format!("-{s}{unit}"),
        });
        t += step;
    }
    ticks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Layout;
    use crate::zoom::ZoomSpan;
    use egui::pos2;

    fn hz_params(rate: f64, center: Option<f64>, zoom: ZoomSpan) -> DisplayParams {
        DisplayParams {
            sample_rate: Some(rate),
            center,
            zoom,
            ..Default::default()
        }
    }

    fn norm_params(zoom: ZoomSpan) -> DisplayParams {
        DisplayParams {
            sample_rate: None,
            center: None,
            zoom,
            ..Default::default()
        }
    }

    /// Parse a tick label back to its numeric value (Hz or cycles/sample).
    fn parse_label(s: &str) -> f64 {
        let (body, mult) = match s.chars().last().unwrap() {
            'G' => (&s[..s.len() - 1], 1e9),
            'M' => (&s[..s.len() - 1], 1e6),
            'K' => (&s[..s.len() - 1], 1e3),
            _ => (s, 1.0),
        };
        body.parse::<f64>().unwrap() * mult
    }

    #[test]
    fn unzoomed_relative_ticks_match_the_established_axis() {
        // The seal: given a span, a center, a zoom range and a division
        // count, the tick positions and label strings follow.
        let p = hz_params(2_400_000.0, None, ZoomSpan::FULL);
        let ticks = freq_ticks(&p);
        let labels: Vec<&str> = ticks.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(labels, ["-960K", "-480K", "0", "+480K", "+960K"]);
        let fractions: Vec<f64> = ticks.iter().map(|t| t.x_fraction).collect();
        assert_eq!(fractions, [0.1, 0.3, 0.5, 0.7, 0.9]);
    }

    #[test]
    fn absolute_ticks_carry_the_center() {
        let p = hz_params(2_400_000.0, Some(100_000_000.0), ZoomSpan::FULL);
        let ticks = freq_ticks(&p);
        // 100 MHz − 960 kHz = 99.04 MHz, etc.
        let expect = [99.04e6, 99.52e6, 100.0e6, 100.48e6, 100.96e6];
        for (tick, e) in ticks.iter().zip(expect) {
            let FreqValue::Hz(hz) = tick.value else {
                panic!("expected Hz");
            };
            assert!((hz - e).abs() < 1e-3, "tick {tick:?} != {e}");
            assert!(
                (parse_label(&tick.label) - e).abs() < 1.0,
                "label {} does not parse back to {e}",
                tick.label
            );
        }
    }

    #[test]
    fn zoomed_ticks_reflect_the_zoomed_span() {
        // Middle half of a 2.4 MHz span: visible span 1.2 MHz.
        let p = hz_params(2_400_000.0, None, ZoomSpan::new(0.25, 0.75));
        let labels: Vec<String> = freq_ticks(&p).into_iter().map(|t| t.label).collect();
        assert_eq!(labels, ["-480K", "-240K", "0", "+240K", "+480K"]);
    }

    #[test]
    fn deep_zoom_ticks_stay_distinct_and_truthful() {
        // A 2.4 kHz window on a 2.45 GHz center: labels need enough digits
        // that adjacent ticks differ — colliding labels misstate frequency.
        let zoom = ZoomSpan::new(0.4995, 0.5005);
        let p = hz_params(2_400_000.0, Some(2_450_000_000.0), zoom);
        let ticks = freq_ticks(&p);
        for pair in ticks.windows(2) {
            assert_ne!(
                pair[0].label, pair[1].label,
                "adjacent tick labels collide at deep zoom"
            );
        }
        // And each label parses back to its true value within half a label
        // step.
        let step = visible_span(&p) * 2.0 / p.freq_divs as f64;
        for t in &ticks {
            let FreqValue::Hz(hz) = t.value else {
                panic!("expected Hz");
            };
            assert!(
                (parse_label(&t.label) - hz).abs() <= step / 2.0,
                "label {} misstates {hz}",
                t.label
            );
        }
    }

    #[test]
    fn rateless_ticks_are_unitless_at_every_zoom_level() {
        // The D-028 seal, at the arithmetic layer: no output string contains
        // "Hz" (any case) — and none smuggles a K/M/G unit either.
        for zoom in [
            ZoomSpan::FULL,
            ZoomSpan::new(0.2, 0.7),
            ZoomSpan::new(0.45, 0.55),
            ZoomSpan::new(0.499, 0.501),
        ] {
            let p = norm_params(zoom);
            let ticks = freq_ticks(&p);
            assert!(!ticks.is_empty());
            for t in &ticks {
                let lower = t.label.to_lowercase();
                assert!(
                    !lower.contains("hz"),
                    "normalized label says Hz: {}",
                    t.label
                );
                assert!(
                    t.label
                        .chars()
                        .all(|c| c.is_ascii_digit() || c == '.' || c == '+' || c == '-'),
                    "normalized label is not a bare fraction: {}",
                    t.label
                );
                // Truthful too, not merely unitless.
                let FreqValue::Normalized(v) = t.value else {
                    panic!("expected normalized");
                };
                let step = visible_span(&p) * 2.0 / p.freq_divs as f64;
                assert!((t.label.parse::<f64>().unwrap() - v).abs() <= step / 2.0);
            }
            // The readout formatter is bound by the same rule.
            let r = format_freq_readout(freq_at(&p, 0.37), &p);
            assert!(!r.to_lowercase().contains("hz"));
        }
    }

    #[test]
    fn unzoomed_normalized_ticks_match_d028() {
        let p = norm_params(ZoomSpan::FULL);
        let labels: Vec<String> = freq_ticks(&p).into_iter().map(|t| t.label).collect();
        assert_eq!(labels, ["-0.4", "-0.2", "0", "+0.2", "+0.4"]);
    }

    /// The seal's "readout agrees with the axis": for pixel positions on the
    /// labelled divisions, the readout's frequency equals what the label
    /// states, unzoomed and zoomed. Exercised through the *string* the user
    /// reads, so the formatter is under test too.
    #[test]
    fn readout_agrees_with_axis_labels() {
        let grid = Rect::from_min_max(pos2(56.0, 10.0), pos2(1256.0, 610.0));
        let layout = Layout { grid };
        let regions = layout.split(1.0);
        for (rate, center) in [
            (2_400_000.0, None),
            (2_400_000.0, Some(2_450_000_000.0)),
            (61_440_000.0, Some(100_000_000.0)),
        ] {
            for zoom in [
                ZoomSpan::FULL,
                ZoomSpan::new(0.2, 0.7),
                ZoomSpan::new(0.48, 0.52),
            ] {
                let p = hz_params(rate, center, zoom);
                let step = visible_span(&p) * 2.0 / p.freq_divs as f64;
                for tick in freq_ticks(&p) {
                    let x = grid.min.x + grid.width() * tick.x_fraction as f32;
                    let pos = pos2(x, grid.center().y);
                    let r = cursor_readout(&p, grid, &regions, pos, 1.0)
                        .expect("pointer on the grid must have a readout");
                    let FreqValue::Hz(readout_hz) = r.freq else {
                        panic!("expected Hz readout");
                    };
                    // f32 pixel → fraction round-trip costs a sub-pixel of
                    // frequency; stay well under half a label step.
                    let tol = (step / 2.0).max(visible_span(&p) * 1e-5);
                    assert!(
                        (parse_label(&tick.label) - readout_hz).abs() <= tol,
                        "readout {readout_hz} disagrees with label {} \
                         (rate {rate}, center {center:?}, zoom {zoom:?})",
                        tick.label
                    );
                }
            }
        }
    }

    #[test]
    fn readout_outside_the_grid_is_none_not_clamped() {
        let grid = Rect::from_min_max(pos2(56.0, 10.0), pos2(1256.0, 610.0));
        let layout = Layout { grid };
        let regions = layout.split(1.0);
        let p = hz_params(2_400_000.0, None, ZoomSpan::FULL);
        for pos in [
            pos2(10.0, 300.0),   // left of the grid
            pos2(1300.0, 300.0), // right of it
            pos2(600.0, 5.0),    // above
            pos2(600.0, 700.0),  // below
            pos2(f32::NAN, 300.0),
        ] {
            assert_eq!(cursor_readout(&p, grid, &regions, pos, 1.0), None);
        }
        // The split gap between the panels yields nothing either.
        let regions = layout.split(0.5);
        let gap_y = (regions.histogram.max.y + regions.waterfall.min.y) / 2.0;
        assert_eq!(
            cursor_readout(&p, grid, &regions, pos2(600.0, gap_y), 1.0),
            None
        );
    }

    #[test]
    fn readout_regions_carry_the_right_measures() {
        let grid = Rect::from_min_max(pos2(0.0, 0.0), pos2(1000.0, 500.0));
        let layout = Layout { grid };
        let regions = layout.split(0.5);
        let mut p = hz_params(1_000_000.0, None, ZoomSpan::FULL);
        // Histogram: frequency + power, no time.
        let r = cursor_readout(&p, grid, &regions, regions.histogram.center(), 1.0).unwrap();
        assert!(r.power_dbfs.is_some() && r.time_s.is_none());
        // Power at the top edge is the reference level, at the bottom edge
        // the scale bottom.
        let top = cursor_readout(
            &p,
            grid,
            &regions,
            pos2(500.0, regions.histogram.min.y),
            1.0,
        )
        .unwrap();
        assert!((top.power_dbfs.unwrap() - p.db_top).abs() < 1e-3);
        // Waterfall without a declared row interval: frequency only — a time
        // base is never invented.
        let r = cursor_readout(&p, grid, &regions, regions.waterfall.center(), 1.0).unwrap();
        assert!(r.power_dbfs.is_none() && r.time_s.is_none());
        // With an interval: time appears, scaled by rows (physical pixels).
        p.wf_row_interval_s = Some(0.1);
        let pos = pos2(500.0, regions.waterfall.min.y + 20.0);
        let r = cursor_readout(&p, grid, &regions, pos, 2.0).unwrap();
        assert!((r.time_s.unwrap() - 20.0 * 2.0 * 0.1).abs() < 1e-4);
    }

    #[test]
    fn db_labels_and_readout_power() {
        assert_eq!(db_label(0.0), "0");
        assert_eq!(db_label(-110.0), "-110");
        assert_eq!(db_label(-100.0), "-100");
        assert_eq!(db_label(-7.5), "-7.5");
        let p = DisplayParams::default();
        assert!((db_at(&p, 0.0) - 0.0).abs() < 1e-6);
        assert!((db_at(&p, 1.0) - -100.0).abs() < 1e-6);
        assert!((db_at(&p, 0.5) - -50.0).abs() < 1e-6);
    }

    #[test]
    fn time_ticks_are_1_2_5_stepped_and_bounded() {
        // 400 rows at 10 ms/row = 4 s span → 1 s steps.
        let ticks = time_ticks(0.01, 400.0);
        let labels: Vec<&str> = ticks.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(labels, ["-1s", "-2s", "-3s"]);
        assert!((ticks[0].row - 100.0).abs() < 1e-3);
        // Sub-second spans get decimal labels.
        let ticks = time_ticks(0.001, 300.0); // 0.3 s span
        assert!(ticks.iter().all(|t| t.label.starts_with("-0.")));
        // Integer steps of 10 keep their zeros: a 30 s span over 600 rows
        // labels −10 s and −20 s — not "−1s"/"−2s", the silently rescaled
        // axis this seal caught.
        let ticks = time_ticks(0.05, 600.0); // 30 s span
        let labels: Vec<&str> = ticks.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(labels, ["-10s", "-20s"]);
        // Degenerate inputs yield nothing rather than nonsense.
        assert!(time_ticks(0.0, 100.0).is_empty());
        assert!(time_ticks(f32::NAN, 100.0).is_empty());
        assert!(time_ticks(0.1, 0.0).is_empty());
    }

    /// D-054: the rate-less waterfall axis is labelled in rows of spectra —
    /// same 1-2-5 machinery, the `r` unit, and never an `s` anywhere.
    #[test]
    fn row_ticks_label_in_rows_of_spectra_never_seconds() {
        // Fast + rate-less: one spectrum per row over a 300-row panel →
        // a 300-row span, 1-2-5 stepped at 100.
        let ticks = row_ticks(1.0, 300.0);
        let labels: Vec<&str> = ticks.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(labels, ["-100r", "-200r"]);
        assert!((ticks[0].row - 100.0).abs() < 1e-3);

        // Traditional + rate-less: 4 spectra per row over 250 panel rows =
        // a 1000-row span; a tick's panel position is its row count divided
        // back by the fold.
        let ticks = row_ticks(4.0, 250.0);
        let labels: Vec<&str> = ticks.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(labels, ["-200r", "-400r", "-600r", "-800r"]);
        assert!((ticks[0].row - 50.0).abs() < 1e-3);
        assert!(ticks.iter().all(|t| !t.label.contains('s')));

        // Degenerate inputs yield nothing rather than nonsense.
        assert!(row_ticks(0.0, 100.0).is_empty());
        assert!(row_ticks(f32::NAN, 100.0).is_empty());
        assert!(row_ticks(1.0, 0.0).is_empty());
    }

    #[test]
    fn readout_format_is_fixed_width_and_truthful() {
        let p = hz_params(2_400_000.0, Some(2_450_000_000.0), ZoomSpan::FULL);
        let s = format_freq_readout(FreqValue::Hz(2_450_123_456.0), &p);
        // Resolves span/1000 = 2.4 kHz on a GHz value → 6 decimals in G.
        assert_eq!(s, "+2.450123G");
        let p = norm_params(ZoomSpan::FULL);
        assert_eq!(
            format_freq_readout(FreqValue::Normalized(-0.2134), &p),
            "-0.213"
        );
    }
}
