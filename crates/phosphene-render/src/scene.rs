// SPDX-License-Identifier: MIT

//! CPU-side geometry for the data surface: grid and live trace.
//!
//! Everything here is pure math over slices — vertex buffers are built on the
//! CPU each frame (cheap at trace sizes) and handed to the single colored
//! pipeline in [`crate::gpu`]. Keeping geometry generation free of GPU types
//! makes it unit-testable without a device.

use bytemuck::{Pod, Zeroable};
use egui::{Color32, Rect};

use crate::layout::DisplayParams;
use crate::theme::{srgb_to_linear, Theme};

/// One colored vertex in normalized device coordinates.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    /// Position in NDC (x right, y up, both −1..1).
    pub pos: [f32; 2],
    /// Straight-alpha color in linear space.
    pub color: [f32; 4],
}

/// Converts physical-pixel coordinates to NDC for a given target size.
#[derive(Debug, Clone, Copy)]
struct PxToNdc {
    w: f32,
    h: f32,
}

impl PxToNdc {
    fn x(&self, px: f32) -> f32 {
        px / self.w * 2.0 - 1.0
    }
    fn y(&self, px: f32) -> f32 {
        1.0 - px / self.h * 2.0
    }
}

/// Grid rect in physical pixels.
#[derive(Debug, Clone, Copy)]
pub struct GridPx {
    /// Left edge, physical pixels.
    pub min_x: f32,
    /// Top edge, physical pixels.
    pub min_y: f32,
    /// Right edge, physical pixels.
    pub max_x: f32,
    /// Bottom edge, physical pixels.
    pub max_y: f32,
}

impl GridPx {
    /// Scale an egui-points rect into physical pixels.
    pub fn from_points(rect: Rect, pixels_per_point: f32) -> Self {
        Self {
            min_x: rect.min.x * pixels_per_point,
            min_y: rect.min.y * pixels_per_point,
            max_x: rect.max.x * pixels_per_point,
            max_y: rect.max.y * pixels_per_point,
        }
    }

    fn width(&self) -> f32 {
        self.max_x - self.min_x
    }
    fn height(&self) -> f32 {
        self.max_y - self.min_y
    }
}

/// Allocating convenience wrapper over [`resample_view_into`]. Tests and
/// one-off callers; the per-frame trace path uses the `_into` form with a
/// reused buffer (§7.7, D-035).
pub fn resample_view(bins: &[f32], lo: f64, hi: f64, columns: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(columns);
    resample_view_into(bins, lo, hi, columns, &mut out);
    out
}

/// Resample the visible window `lo..hi` (fractions of the full span, as
/// [`crate::zoom::ZoomSpan`] holds them — FR-D6 display-side zoom, D-011) of
/// `bins` to `columns` values, one per output column, into `out` (cleared
/// first; once its capacity covers `columns` this allocates nothing — §7.7).
///
/// When a column covers at least one whole bin, it takes the **max** over
/// its bin range so narrow spikes survive — the same transient-honesty
/// argument as §7.6's max row aggregation. When zoomed past bin resolution,
/// values interpolate linearly.
pub fn resample_view_into(bins: &[f32], lo: f64, hi: f64, columns: usize, out: &mut Vec<f32>) {
    assert!(!bins.is_empty() && columns > 0);
    assert!(lo.is_finite() && hi.is_finite() && lo < hi);
    out.clear();
    let n = bins.len();
    if n == 1 {
        out.extend(std::iter::repeat_n(bins[0], columns));
        return;
    }
    let last = (n - 1) as f64;
    // Fractional bin coordinate of view fraction `t` — the same linear
    // full-span mapping the axis labels use.
    let pos = |t: f64| (lo + t * (hi - lo)) * last;
    let sample = |b: f64| -> f32 {
        let b = b.clamp(0.0, last);
        let i = (b.floor() as usize).min(n - 2);
        let t = (b - i as f64) as f32;
        bins[i] * (1.0 - t) + bins[i + 1] * t
    };
    if columns == 1 {
        let b0 = pos(0.0).clamp(0.0, last);
        let b1 = pos(1.0).clamp(0.0, last);
        let v = if b1 - b0 >= 1.0 {
            bins[b0.ceil() as usize..=b1.floor() as usize]
                .iter()
                .copied()
                .fold(f32::MIN, f32::max)
        } else {
            sample((b0 + b1) * 0.5)
        };
        out.push(v);
        return;
    }
    // Bins each column is responsible for; the per-column windows tile the
    // visible range exactly, so no bin in view escapes the max.
    let colspan = (hi - lo) * last / (columns - 1) as f64;
    out.extend((0..columns).map(|c| {
        let b = pos(c as f64 / (columns - 1) as f64);
        if colspan >= 1.0 {
            let b0 = (b - colspan / 2.0).clamp(0.0, last);
            let b1 = (b + colspan / 2.0).clamp(0.0, last);
            // A closed interval at least one bin wide always contains an
            // integer bin index.
            bins[b0.ceil() as usize..=b1.floor() as usize]
                .iter()
                .copied()
                .fold(f32::MIN, f32::max)
        } else {
            sample(b)
        }
    }));
}

/// Append a solid axis-aligned quad.
fn push_quad(out: &mut Vec<Vertex>, to: PxToNdc, x0: f32, y0: f32, x1: f32, y1: f32, c: [f32; 4]) {
    let (x0, x1) = (to.x(x0), to.x(x1));
    let (y0, y1) = (to.y(y0), to.y(y1));
    let v = |x, y| Vertex {
        pos: [x, y],
        color: c,
    };
    out.extend_from_slice(&[
        v(x0, y0),
        v(x1, y0),
        v(x1, y1),
        v(x0, y0),
        v(x1, y1),
        v(x0, y1),
    ]);
}

/// Build the grid: minor division lines plus the surface frame, 1 physical
/// pixel wide, snapped to pixel centers so they stay crisp at any DPI
/// (FR-D5).
pub fn build_grid(
    out: &mut Vec<Vertex>,
    grid: GridPx,
    surface: [u32; 2],
    params: &DisplayParams,
    theme: &Theme,
) {
    let to = PxToNdc {
        w: surface[0] as f32,
        h: surface[1] as f32,
    };
    let minor = srgb_to_linear(theme.grid_minor);
    let frame = srgb_to_linear(theme.grid_frame);
    let snap = |v: f32| v.round();

    // Vertical lines (frequency divisions).
    for i in 0..=params.freq_divs {
        let x = snap(grid.min_x + grid.width() * i as f32 / params.freq_divs as f32);
        let c = if i == 0 || i == params.freq_divs {
            frame
        } else {
            minor
        };
        push_quad(out, to, x - 0.5, grid.min_y, x + 0.5, grid.max_y, c);
    }
    // Horizontal lines (power divisions).
    let db_divs = params.db_divs();
    for i in 0..=db_divs {
        let y = snap(grid.min_y + grid.height() * i as f32 / db_divs as f32);
        let c = if i == 0 || i == db_divs { frame } else { minor };
        push_quad(out, to, grid.min_x, y - 0.5, grid.max_x, y + 0.5, c);
    }
}

/// Build a plain 1-px frame around `rect` (no division lines) — the border
/// of the waterfall panel, matching the histogram frame's colour and
/// snapping so the two read as one instrument (FR-D4).
pub fn build_frame(out: &mut Vec<Vertex>, rect: GridPx, surface: [u32; 2], theme: &Theme) {
    let to = PxToNdc {
        w: surface[0] as f32,
        h: surface[1] as f32,
    };
    let frame = srgb_to_linear(theme.grid_frame);
    let (x0, x1) = (rect.min_x.round(), rect.max_x.round());
    let (y0, y1) = (rect.min_y.round(), rect.max_y.round());
    push_quad(out, to, x0 - 0.5, y0, x0 + 0.5, y1, frame);
    push_quad(out, to, x1 - 0.5, y0, x1 + 0.5, y1, frame);
    push_quad(out, to, x0, y0 - 0.5, x1, y0 + 0.5, frame);
    push_quad(out, to, x0, y1 - 0.5, x1, y1 + 0.5, frame);
}

/// Append a stroke along the polyline `(xs[i], ys[i])` as one vertical bar
/// per column, each spanning the y-interval the trace crosses in that column
/// plus `half` pixels of thickness. Continuous by construction — steep
/// spikes stay solid instead of breaking into dashes — which is the classic
/// way instrument traces are rasterized. `half` is in physical pixels.
fn push_stroke(out: &mut Vec<Vertex>, to: PxToNdc, xs: &[f32], ys: &[f32], half: f32, c: [f32; 4]) {
    let n = xs.len();
    if n < 2 {
        return;
    }
    for i in 0..n - 1 {
        let (y_lo, y_hi) = if ys[i] <= ys[i + 1] {
            (ys[i], ys[i + 1])
        } else {
            (ys[i + 1], ys[i])
        };
        push_quad(
            out,
            to,
            xs[i] - half,
            y_lo - half,
            xs[i + 1] + half,
            y_hi + half,
            c,
        );
    }
}

/// Half-thickness of the glow halo stroke, physical pixels.
const GLOW_HALF_PX: f32 = 2.5;

/// Style of one trace overlay. The live trace (FR-D2) and the max-hold
/// trace (FR-D3) share [`build_trace_styled`]'s geometry and differ only
/// here — one trace pipeline, two looks, never a fork.
#[derive(Debug, Clone, Copy)]
pub struct TraceStyle {
    /// Top of the gradient fill under the trace (fades to transparent at
    /// the surface floor); `None` draws no fill.
    pub fill: Option<Color32>,
    /// Wide soft halo under the core line; `None` draws no glow.
    pub glow: Option<Color32>,
    /// The crisp core line.
    pub core: Color32,
    /// Half-thickness of the core line, physical pixels.
    pub core_half_px: f32,
}

impl TraceStyle {
    /// The FR-D2 live-trace look: gradient fill, glow halo, crisp core —
    /// the luminous trace D-010 asks for.
    pub fn live(theme: &Theme) -> Self {
        Self {
            fill: Some(theme.trace_fill),
            glow: Some(theme.trace_glow),
            core: theme.trace_core,
            core_half_px: 0.75,
        }
    }

    /// The FR-D3 max-hold look: a bare thin line, visually distinct from
    /// the live trace and subordinate to it (D-010: it must not out-shout
    /// the live spectrum or the persistence surface) — no fill, no glow.
    pub fn max_hold(theme: &Theme) -> Self {
        Self {
            fill: None,
            glow: None,
            core: theme.max_hold_line,
            core_half_px: 0.6,
        }
    }
}

/// Reusable scratch for trace geometry (§7.7, D-035): the resampled columns
/// and the pixel-space polyline. Owned by the caller and handed to every
/// [`build_trace`] / [`build_trace_styled`] call; each build clears and
/// refills the same buffers, so once warm at a given surface width the trace
/// path allocates nothing per frame. One scratch serves any number of traces
/// built sequentially (the composer builds max hold, then live, through the
/// same one).
#[derive(Debug, Default)]
pub struct TraceScratch {
    columns: Vec<f32>,
    xs: Vec<f32>,
    ys: Vec<f32>,
}

/// Build the live trace (FR-D2): a gradient fill from the trace down to the
/// surface floor, a wide soft glow, and a crisp core line on top — the
/// luminous-trace look D-010 asks for, built from geometry rather than any
/// CRT pastiche. Equivalent to [`build_trace_styled`] with
/// [`TraceStyle::live`].
pub fn build_trace(
    out: &mut Vec<Vertex>,
    bins: &[f32],
    grid: GridPx,
    surface: [u32; 2],
    params: &DisplayParams,
    theme: &Theme,
    scratch: &mut TraceScratch,
) {
    build_trace_styled(
        out,
        bins,
        grid,
        surface,
        params,
        &TraceStyle::live(theme),
        scratch,
    );
}

/// Build one trace overlay in the given [`TraceStyle`] — the single trace
/// geometry pipeline behind both the live (FR-D2) and max-hold (FR-D3)
/// overlays. `scratch` is cleared and refilled; the build allocates nothing
/// once the scratch and `out` are warm at the current surface width (§7.7 —
/// the D-035 seal proves it with a counting allocator).
pub fn build_trace_styled(
    out: &mut Vec<Vertex>,
    bins: &[f32],
    grid: GridPx,
    surface: [u32; 2],
    params: &DisplayParams,
    style: &TraceStyle,
    scratch: &mut TraceScratch,
) {
    let width_px = grid.width().round().max(2.0) as usize;
    let TraceScratch { columns, xs, ys } = scratch;
    // The visible window of the spectrum (FR-D6 zoom, D-011): the trace
    // reinterprets the existing FFT through the same zoom the labels use.
    resample_view_into(bins, params.zoom.lo(), params.zoom.hi(), width_px, columns);
    let to = PxToNdc {
        w: surface[0] as f32,
        h: surface[1] as f32,
    };

    xs.clear();
    xs.extend((0..width_px).map(|i| grid.min_x + i as f32 * grid.width() / (width_px - 1) as f32));
    ys.clear();
    ys.extend(
        columns
            .iter()
            .map(|&db| grid.max_y - params.db_to_unit(db) * grid.height()),
    );

    // 1. Gradient fill: luminous at the trace, transparent at the floor.
    if let Some(fill) = style.fill {
        let fill_top = srgb_to_linear(fill);
        let fill_bottom = [fill_top[0], fill_top[1], fill_top[2], 0.0];
        for i in 0..width_px - 1 {
            let v = |x: f32, y: f32, c: [f32; 4]| Vertex {
                pos: [to.x(x), to.y(y)],
                color: c,
            };
            let (t0, t1) = (v(xs[i], ys[i], fill_top), v(xs[i + 1], ys[i + 1], fill_top));
            let (b0, b1) = (
                v(xs[i], grid.max_y, fill_bottom),
                v(xs[i + 1], grid.max_y, fill_bottom),
            );
            out.extend_from_slice(&[t0, t1, b1, t0, b1, b0]);
        }
    }

    // 2. Soft glow halo, then 3. crisp core.
    if let Some(glow) = style.glow {
        push_stroke(out, to, xs, ys, GLOW_HALF_PX, srgb_to_linear(glow));
    }
    push_stroke(
        out,
        to,
        xs,
        ys,
        style.core_half_px,
        srgb_to_linear(style.core),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_view_places_a_zoomed_spike_where_the_axis_says() {
        // Spike at bin 517 of 1024: full-span fraction u = 517/1023. Zoomed
        // to (0.4, 0.6), the axis puts it at view fraction (u-0.4)/0.2; the
        // resampled trace must light the matching column — trace and labels
        // reading one geometry.
        let mut bins = vec![-100.0f32; 1024];
        bins[517] = -3.0;
        let columns = 200;
        let r = resample_view(&bins, 0.4, 0.6, columns);
        let u = 517.0 / 1023.0;
        let expect = ((u - 0.4) / 0.2 * (columns - 1) as f64).round() as usize;
        let hot = r
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert!(
            hot.abs_diff(expect) <= 1,
            "spike at column {hot}, axis says {expect}"
        );
        assert_eq!(r.iter().copied().fold(f32::MIN, f32::max), -3.0);
        // Outside the window the spike is gone entirely.
        let r = resample_view(&bins, 0.0, 0.3, columns);
        assert!(r.iter().all(|&v| v == -100.0));
        // Deep zoom past bin resolution interpolates instead of inventing.
        let r = resample_view(&bins, 0.505, 0.506, 64);
        assert!(r.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn scene_vertices_stay_in_ndc() {
        let grid = GridPx {
            min_x: 56.0,
            min_y: 40.0,
            max_x: 1264.0,
            max_y: 676.0,
        };
        let params = DisplayParams::default();
        let theme = Theme::default();
        let bins: Vec<f32> = (0..1024).map(|i| -100.0 + (i % 97) as f32).collect();
        let mut verts = Vec::new();
        build_grid(&mut verts, grid, [1280, 720], &params, &theme);
        let mut scratch = TraceScratch::default();
        build_trace(
            &mut verts,
            &bins,
            grid,
            [1280, 720],
            &params,
            &theme,
            &mut scratch,
        );
        assert!(!verts.is_empty());
        for v in &verts {
            assert!(
                v.pos[0] >= -1.0 && v.pos[0] <= 1.0,
                "x out of NDC: {}",
                v.pos[0]
            );
            assert!(
                v.pos[1] >= -1.0 && v.pos[1] <= 1.0,
                "y out of NDC: {}",
                v.pos[1]
            );
            assert!(v.color.iter().all(|c| (0.0..=1.0).contains(c)));
        }
        // Triangle list: vertex count divisible by 3.
        assert_eq!(verts.len() % 3, 0);
    }

    #[test]
    fn grid_line_count_matches_divisions() {
        let grid = GridPx {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 100.0,
            max_y: 100.0,
        };
        let params = DisplayParams::default();
        // D-065 §5: the default is exactly **ten** power divisions, so the
        // grid the GPU draws carries eleven horizontal lines. Pinned as a
        // literal as well as against `db_divs()` — the point of the change
        // is the number itself, and an assertion that only compared the grid
        // to the accessor would agree with any number at all.
        assert_eq!(params.db_divs(), 10);
        let mut verts = Vec::new();
        build_grid(&mut verts, grid, [100, 100], &params, &Theme::default());
        // (freq_divs + 1) + (db_divs + 1) lines, 6 vertices each.
        let lines = (params.freq_divs + 1) + (params.db_divs() + 1);
        assert_eq!(lines, 22, "10 frequency + 10 power divisions");
        assert_eq!(verts.len(), lines * 6);
    }
}
