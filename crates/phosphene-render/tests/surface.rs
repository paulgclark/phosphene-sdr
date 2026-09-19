// SPDX-License-Identifier: MIT

//! Deterministic offscreen seals for the M1-B data surfaces (D-014): the
//! goldens render a fixed synthetic intensity grid at a fixed gamma through
//! `FrameComposer` into an `OffscreenTarget` — the same code path a user
//! sees — and every test asserts what the picture must *be*, not merely that
//! a picture appeared.
//!
//! Golden PNGs live in `tests/goldens/` and are compared with a perceptual
//! tolerance, never byte equality: GPU rasterisation differs across the four
//! CI legs (lavapipe/Vulkan vs Metal), and a byte-exact golden would flap.
//! Regenerate deliberately with `PHOSPHENE_UPDATE_GOLDENS=1 cargo test`.

use std::path::PathBuf;

use phosphene_render::{
    Colormap, EguiFrame, FrameComposer, OffscreenTarget, RtsaInput, Theme, ViewState, ZoomSpan,
    INTENSITY_FLOOR_EPSILON, OFFSCREEN_FORMAT,
};

const W: u32 = 400;
const H: u32 = 300;
const BINS: usize = 256;
const LEVELS: usize = 64;
const GOLDEN_GAMMA: f32 = 0.8;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Gpu {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("no GPU adapter — the offscreen seals cannot run");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("phosphene-render-tests"),
        ..Default::default()
    }))
    .expect("GPU device request failed");
    Gpu { device, queue }
}

/// An egui frame with no chrome: the goldens seal the data surfaces alone,
/// keeping text rasterisation out of the cross-platform diff budget.
fn empty_chrome() -> EguiFrame {
    EguiFrame {
        primitives: Vec::new(),
        textures_delta: egui::TexturesDelta::default(),
        pixels_per_point: 1.0,
    }
}

/// The fixed synthetic intensity grid (D-014: a golden must come from a
/// known input). Blocks at known bin/level ranges plus a smooth ramp:
///
/// * bins 0..16              — vertical ramp, I = 0.9·level/(levels−1)
/// * bins 32..64, lv 48..56  — full-scale block, I = 1.0
/// * bins 96..128, lv 24..40 — mid block, I = 0.45
/// * bins 160..192, lv 8..16 — low block, I = 0.08
/// * bins 208..240, lv 30..34 — sub-ε strip, I = ε/2: must render as
///   background, not as the colormap's zero end
fn synthetic_grid() -> Vec<f32> {
    let mut g = vec![0.0f32; BINS * LEVELS];
    for k in 0..16 {
        for l in 0..LEVELS {
            g[k * LEVELS + l] = 0.9 * l as f32 / (LEVELS - 1) as f32;
        }
    }
    let mut set = |bins: std::ops::Range<usize>, levels: std::ops::Range<usize>, v: f32| {
        for k in bins {
            for l in levels.clone() {
                g[k * LEVELS + l] = v;
            }
        }
    };
    set(32..64, 48..56, 1.0);
    set(96..128, 24..40, 0.45);
    set(160..192, 8..16, 0.08);
    let sub_epsilon = INTENSITY_FLOOR_EPSILON / 2.0;
    set(208..240, 30..34, sub_epsilon);
    g
}

/// Render one RTSA frame with the given view state, live and max-hold
/// traces, and return the RGBA bytes.
fn render_traces(
    g: &Gpu,
    composer: &mut FrameComposer,
    target: &OffscreenTarget,
    intensity: &[f32],
    view: &ViewState,
    live_bins: &[f32],
    max_hold_bins: &[f32],
) -> Vec<u8> {
    let theme = Theme::default();
    composer.render_rtsa_frame(
        &g.device,
        &g.queue,
        &target.view,
        target.size,
        RtsaInput {
            live_bins,
            max_hold_bins,
            intensity,
            intensity_bins: if intensity.is_empty() { 0 } else { BINS },
            intensity_levels: if intensity.is_empty() { 0 } else { LEVELS },
            surface_points: egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(target.size[0] as f32, target.size[1] as f32),
            ),
            view,
            theme: &theme,
        },
        empty_chrome(),
    );
    target.read_rgba(&g.device, &g.queue)
}

/// Render one RTSA frame with the given view state and return the RGBA
/// bytes.
fn render_view(
    g: &Gpu,
    composer: &mut FrameComposer,
    target: &OffscreenTarget,
    intensity: &[f32],
    view: &ViewState,
) -> Vec<u8> {
    render_traces(g, composer, target, intensity, view, &[], &[])
}

/// Render one RTSA frame at the given split (the M1-B harness shape).
fn render(
    g: &Gpu,
    composer: &mut FrameComposer,
    target: &OffscreenTarget,
    intensity: &[f32],
    split: f32,
) -> Vec<u8> {
    let view = ViewState {
        split,
        ..Default::default()
    };
    render_view(g, composer, target, intensity, &view)
}

fn px(img: &[u8], x: u32, y: u32, width: u32) -> [u8; 4] {
    let o = ((y * width + x) * 4) as usize;
    [img[o], img[o + 1], img[o + 2], img[o + 3]]
}

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(format!("{name}.png"))
}

/// Compare `img` against the stored golden with a perceptual tolerance
/// (D-014), or rewrite the golden when `PHOSPHENE_UPDATE_GOLDENS=1`.
fn assert_matches_golden(img: &[u8], width: u32, height: u32, name: &str) {
    let path = golden_path(name);
    if std::env::var_os("PHOSPHENE_UPDATE_GOLDENS").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let file = std::fs::File::create(&path).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width, height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(img).unwrap();
        return;
    }
    let file = std::fs::File::open(&path).unwrap_or_else(|e| {
        panic!("missing golden {path:?} ({e}); generate with PHOSPHENE_UPDATE_GOLDENS=1")
    });
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .unwrap();
    let mut golden = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut golden).unwrap();
    assert_eq!(
        (info.width, info.height),
        (width, height),
        "golden {name} size"
    );
    golden.truncate((width * height * 4) as usize);

    // Perceptual gate: mean per-channel error and an outlier budget, both
    // sized to absorb rasteriser differences while failing loudly on a wrong
    // colormap, a missing surface, or a broken floor.
    let mut sum = 0u64;
    let mut outliers = 0u64;
    let mut n = 0u64;
    for (a, b) in img.as_chunks::<4>().0.iter().zip(golden.as_chunks::<4>().0) {
        let d = a[..3]
            .iter()
            .zip(&b[..3])
            .map(|(x, y)| x.abs_diff(*y) as u64)
            .collect::<Vec<_>>();
        sum += d.iter().sum::<u64>();
        if d.iter().copied().max().unwrap() > 32 {
            outliers += 1;
        }
        n += 1;
    }
    let mean = sum as f64 / (n as f64 * 3.0);
    let outlier_frac = outliers as f64 / n as f64;
    assert!(
        mean <= 3.0 && outlier_frac <= 0.02,
        "{name} deviates from golden: mean abs diff {mean:.2} (limit 3.0), \
         outlier fraction {outlier_frac:.4} (limit 0.02)"
    );
}

fn assert_close(actual: [u8; 4], expected: [u8; 3], tol: u8, what: &str) {
    for c in 0..3 {
        assert!(
            actual[c].abs_diff(expected[c]) <= tol,
            "{what}: got {actual:?}, expected ~{expected:?} (±{tol})"
        );
    }
}

// Probe pixels, derived from the block geometry above at 400×300 with the
// surface covering the whole target and split = 1.0 (histogram only):
//   full-scale block centre → (75, 54); mid block centre → (175, 147);
//   sub-ε strip → (351, 144), inside its rows 139..=160 and clear of the
//   grid (see `assert_off_grid`).
const FULL_SCALE_PX: (u32, u32) = (75, 54);
const MID_PX: (u32, u32) = (175, 147);
const SUB_EPSILON_PX: (u32, u32) = (351, 144);

/// The pixel rows and columns the grid lines occupy at `W`×`H` for the
/// current division counts.
///
/// `build_grid` snaps each division to `round(v)` and strokes it ±0.5 px,
/// so the stroke's edges fall exactly on sample centres and *which* of the
/// two adjacent rows lights up is the rasteriser's fill rule to decide —
/// lavapipe and a hardware GPU disagree by one count there. Both are
/// treated as covered.
fn grid_lines(divs: usize, extent: u32) -> Vec<u32> {
    (0..=divs)
        .flat_map(|i| {
            let v = (extent as f32 * i as f32 / divs as f32).round() as i32;
            [v - 1, v]
        })
        .filter(|v| (0..extent as i32).contains(v))
        .map(|v| v as u32)
        .collect()
}

/// A probe that lands on a grid line measures the *grid*, not the surface
/// underneath it — and it does so quietly, because a minor grid line is
/// dim enough to sit inside a background tolerance.
///
/// D-065 §5 moved the power axis from eleven divisions to ten, which walked
/// a horizontal line onto row 149 — the sub-ε strip's old centre, and the
/// pixel the §7.3 floor assertion read. It measured the grid line from that
/// moment on, passing on a hardware GPU by a single count and failing on
/// lavapipe. Every probe is checked here, so the next change to a division
/// count fails loudly instead of silently reading chrome.
fn assert_off_grid(p: (u32, u32), what: &str) {
    let params = ViewState::default().params;
    let rows = grid_lines(params.db_divs(), H);
    let cols = grid_lines(params.freq_divs, W);
    assert!(
        !rows.contains(&p.1),
        "{what}: probe row {} is a horizontal grid line ({rows:?}) — it would \
         measure the grid, not the surface",
        p.1
    );
    assert!(
        !cols.contains(&p.0),
        "{what}: probe column {} is a vertical grid line ({cols:?})",
        p.0
    );
}

#[test]
fn histogram_goldens_one_per_colormap() {
    let g = gpu();
    let mut composer = FrameComposer::new(&g.device, OFFSCREEN_FORMAT);
    composer.set_gamma(GOLDEN_GAMMA);
    let target = OffscreenTarget::new(&g.device, W, H);
    let grid = synthetic_grid();
    for map in Colormap::ALL {
        composer.set_colormap(map);
        let img = render(&g, &mut composer, &target, &grid, 1.0);
        assert_matches_golden(&img, W, H, &format!("histogram-{}", map.name()));
    }
}

#[test]
fn floor_is_background_and_full_scale_is_colormap_top() {
    let g = gpu();
    let mut composer = FrameComposer::new(&g.device, OFFSCREEN_FORMAT);
    composer.set_gamma(GOLDEN_GAMMA);
    // Viridis: its zero end (#440154) is far from the background, so "floor
    // renders as background" and "floor renders as colormap-zero" are
    // unmistakably different pixels.
    composer.set_colormap(Colormap::Viridis);
    let target = OffscreenTarget::new(&g.device, W, H);
    let img = render(&g, &mut composer, &target, &synthetic_grid(), 1.0);

    let theme = Theme::default();
    let bg = [theme.bg.r(), theme.bg.g(), theme.bg.b()];
    let lut = Colormap::Viridis.lut();

    const EMPTY_PX: (u32, u32) = (390, 10);
    for (p, what) in [
        (SUB_EPSILON_PX, "sub-ε probe"),
        (FULL_SCALE_PX, "full-scale probe"),
        (MID_PX, "mid probe"),
        (EMPTY_PX, "empty-region probe"),
    ] {
        assert_off_grid(p, what);
    }

    // Positive control for the probe itself: "this pixel renders as
    // background" only says something about §7.3's floor if the pixel is a
    // sub-ε *cell*. Light the same strip to full scale and the same probe
    // must go bright — so it is inside the strip, not beside it. Without
    // this, moving the strip or the probe would leave the floor assertion
    // passing on empty background forever.
    let mut lit = synthetic_grid();
    for k in 208..240 {
        for l in 30..34 {
            lit[k * LEVELS + l] = 1.0;
        }
    }
    let lit_img = render(&g, &mut composer, &target, &lit, 1.0);
    assert_close(
        px(&lit_img, SUB_EPSILON_PX.0, SUB_EPSILON_PX.1, W),
        lut[255],
        8,
        "the sub-ε probe does not sit on the strip at all",
    );

    let floor = px(&img, SUB_EPSILON_PX.0, SUB_EPSILON_PX.1, W);
    assert_close(floor, bg, 6, "sub-ε cell must render as background (§7.3)");
    let zero_end = lut[0];
    assert!(
        (0..3).any(|c| floor[c].abs_diff(zero_end[c]) > 24),
        "sub-ε cell rendered as the colormap zero end — the floor is glowing"
    );

    // gamma_adjust(1.0) = 1.0 for any gamma: a full-scale cell must be the
    // colormap's top end.
    let full = px(&img, FULL_SCALE_PX.0, FULL_SCALE_PX.1, W);
    assert_close(
        full,
        lut[255],
        8,
        "full-scale cell must be the colormap top",
    );

    // And an untouched region (intensity 0) is background too.
    let empty = px(&img, EMPTY_PX.0, EMPTY_PX.1, W);
    assert_close(empty, bg, 6, "empty region must stay background");
}

#[test]
fn colormap_switch_changes_pixels_and_rebuilds_no_pipeline() {
    let g = gpu();
    let mut composer = FrameComposer::new(&g.device, OFFSCREEN_FORMAT);
    composer.set_gamma(GOLDEN_GAMMA);
    let target = OffscreenTarget::new(&g.device, W, H);
    let grid = synthetic_grid();

    assert_eq!(composer.colormap(), Colormap::P7, "D-010 default is p7");
    let builds_before = composer.surface_pipeline_builds();
    let before = render(&g, &mut composer, &target, &grid, 1.0);

    let next = composer.cycle_colormap();
    assert_eq!(next, Colormap::Inferno);
    let after = render(&g, &mut composer, &target, &grid, 1.0);

    // The switch must actually change the output...
    let mid_before = px(&before, MID_PX.0, MID_PX.1, W);
    let mid_after = px(&after, MID_PX.0, MID_PX.1, W);
    assert!(
        (0..3).any(|c| mid_before[c].abs_diff(mid_after[c]) > 20),
        "colormap switch left the mid-intensity block unchanged: \
         {mid_before:?} vs {mid_after:?}"
    );
    // ...and must not have built any pipeline to do it (D-009).
    assert_eq!(
        composer.surface_pipeline_builds(),
        builds_before,
        "colormap switch rebuilt a pipeline — D-009 violation"
    );

    // Cycling through all four maps returns to the start, still no rebuild.
    for _ in 0..3 {
        composer.cycle_colormap();
    }
    assert_eq!(composer.colormap(), Colormap::P7);
    let _ = render(&g, &mut composer, &target, &grid, 1.0);
    assert_eq!(composer.surface_pipeline_builds(), builds_before);
}

#[test]
fn split_endpoints_render_without_panic() {
    let g = gpu();
    let mut composer = FrameComposer::new(&g.device, OFFSCREEN_FORMAT);
    // Odd target size: exercises non-round rects at both endpoints.
    let (w, h) = (173, 131);
    let target = OffscreenTarget::new(&g.device, w, h);
    let grid = synthetic_grid();
    // Give the waterfall content so both endpoint frames have data flowing.
    let mut row = vec![-100.0f32; BINS];
    row[40] = 0.0;
    composer.push_waterfall_row(&g.device, &g.queue, &row, None);

    // The full closed range, both endpoints included (FR-D4), plus the
    // sanitized non-finite case: none may panic or wedge the device.
    for split in [0.0, 0.25, 0.5, 0.75, 1.0, f32::NAN] {
        let img = render(&g, &mut composer, &target, &grid, split);
        assert_eq!(img.len(), (w * h * 4) as usize);
    }

    // At split 0 the waterfall fills the surface: its hot bin must be lit.
    let img = render(&g, &mut composer, &target, &grid, 0.0);
    let hot_x = ((40.5 / (BINS - 1) as f32) * w as f32) as u32;
    let hot = px(&img, hot_x, 0, w);
    let quiet = px(&img, hot_x + 40, 0, w);
    assert!(
        hot[..3].iter().map(|&c| c as u32).sum::<u32>()
            > quiet[..3].iter().map(|&c| c as u32).sum::<u32>() + 60,
        "waterfall-only endpoint shows no data: hot {hot:?} vs quiet {quiet:?}"
    );

    // At split 1 the histogram fills the surface: the full-scale block must
    // be lit somewhere in its region.
    let img = render(&g, &mut composer, &target, &grid, 1.0);
    let lut_top = composer.colormap().lut()[255];
    let bx = ((48.0 / (BINS - 1) as f32) * w as f32) as u32;
    let by = ((1.0 - 51.5 / (LEVELS - 1) as f32) * h as f32) as u32;
    let cell = px(&img, bx, by, w);
    assert_close(cell, lut_top, 24, "histogram-only endpoint shows the block");
}

/// Push `n` diagonal-pattern rows: row `i` is −100 dBFS with a 0 dBFS hot
/// bin at `(i·3) mod BINS`.
fn push_diagonal_rows(g: &Gpu, composer: &mut FrameComposer, n: usize) {
    let mut row = vec![-100.0f32; BINS];
    for i in 0..n {
        row.fill(-100.0);
        row[(i * 3) % BINS] = 0.0;
        composer.push_waterfall_row(&g.device, &g.queue, &row, None);
    }
}

/// Screen x of a waterfall bin's hot pixel at width `W`.
fn wf_bin_x(bin: usize) -> u32 {
    ((bin as f32 / (BINS - 1) as f32) * W as f32) as u32
}

#[test]
fn waterfall_golden_and_newest_row_on_top() {
    let g = gpu();
    let mut composer = FrameComposer::new(&g.device, OFFSCREEN_FORMAT);
    let target = OffscreenTarget::new(&g.device, W, H);
    push_diagonal_rows(&g, &mut composer, 300);
    let img = render(&g, &mut composer, &target, &[], 0.0);
    assert_matches_golden(&img, W, H, "waterfall");

    let bright = |p: [u8; 4]| p[..3].iter().map(|&c| c as u32).sum::<u32>();
    // Newest row (i=299, hot bin 129) at the top of the panel...
    let hot = px(&img, wf_bin_x(299 * 3 % BINS), 0, W);
    let cold = px(&img, wf_bin_x(299 * 3 % BINS) + 60, 0, W);
    assert!(
        bright(hot) > bright(cold) + 60,
        "newest row not on top: {hot:?} vs {cold:?}"
    );
    // ...and ten rows down the screen shows the row from ten rows earlier —
    // the scroll is the ring offset, in order.
    let hot10 = px(&img, wf_bin_x(289 * 3 % BINS), 10, W);
    let cold10 = px(&img, wf_bin_x(289 * 3 % BINS) + 60, 10, W);
    assert!(
        bright(hot10) > bright(cold10) + 60,
        "row order broken 10 rows down: {hot10:?} vs {cold10:?}"
    );
}

#[test]
fn waterfall_ring_wraps_and_still_shows_the_latest_rows() {
    let g = gpu();
    let mut composer = FrameComposer::new(&g.device, OFFSCREEN_FORMAT);
    let target = OffscreenTarget::new(&g.device, W, H);
    // More rows than the ring holds: the head wraps and the newest rows must
    // still be the ones on screen.
    let n = phosphene_render::RING_ROWS as usize + 50;
    push_diagonal_rows(&g, &mut composer, n);
    let img = render(&g, &mut composer, &target, &[], 0.0);
    let bright = |p: [u8; 4]| p[..3].iter().map(|&c| c as u32).sum::<u32>();
    let newest_bin = (n - 1) * 3 % BINS;
    let hot = px(&img, wf_bin_x(newest_bin), 0, W);
    let cold = px(&img, wf_bin_x(newest_bin) + 60, 0, W);
    assert!(
        bright(hot) > bright(cold) + 60,
        "after wrap the newest row is wrong: {hot:?} vs {cold:?}"
    );
}

#[test]
fn waterfall_auto_range_toggle_changes_the_mapping() {
    let g = gpu();
    let mut composer = FrameComposer::new(&g.device, OFFSCREEN_FORMAT);
    let target = OffscreenTarget::new(&g.device, W, H);
    // All content well below the shared top: auto-range should stretch it.
    let mut row = vec![-90.0f32; BINS];
    row[100] = -40.0;
    for _ in 0..50 {
        composer.push_waterfall_row(&g.device, &g.queue, &row, None);
    }
    let shared = render(&g, &mut composer, &target, &[], 0.0);
    composer.set_waterfall_auto_range(true);
    assert!(composer.waterfall_auto_range());
    let auto = render(&g, &mut composer, &target, &[], 0.0);
    let p_shared = px(&shared, wf_bin_x(100), 20, W);
    let p_auto = px(&auto, wf_bin_x(100), 20, W);
    assert!(
        (0..3).any(|c| p_shared[c].abs_diff(p_auto[c]) > 16),
        "auto-range toggle changed nothing: {p_shared:?} vs {p_auto:?}"
    );
}

#[test]
fn zoom_reinterprets_the_surfaces_and_reset_restores_them() {
    let g = gpu();
    let mut composer = FrameComposer::new(&g.device, OFFSCREEN_FORMAT);
    composer.set_gamma(GOLDEN_GAMMA);
    let target = OffscreenTarget::new(&g.device, W, H);
    let grid = synthetic_grid();

    // Unzoomed histogram reference frame.
    let full = render(&g, &mut composer, &target, &grid, 1.0);

    // Zoom into the left half of the span: the full-scale block (bins
    // 32..64) must appear at the x the zoomed axis assigns it — the surface
    // reinterprets the existing grid (D-011), it never resamples the data.
    let mut view = ViewState {
        split: 1.0,
        ..Default::default()
    };
    view.params.zoom = ZoomSpan::new(0.0, 0.5);
    let zoomed = render_view(&g, &mut composer, &target, &grid, &view);
    let lut_top = composer.colormap().lut()[255];
    // Block centre bin 48 → full-span fraction 48/255 → view fraction
    // (48/255)/0.5 of the width.
    let x = (((48.0 / (BINS - 1) as f32) / 0.5) * W as f32) as u32;
    let y = ((1.0 - 51.5 / (LEVELS - 1) as f32) * H as f32) as u32;
    assert_close(
        px(&zoomed, x, y, W),
        lut_top,
        24,
        "zoomed block must sit at the axis-claimed x",
    );

    // Resetting the zoom restores the unzoomed rendering exactly (FR-D6:
    // zoom is reversible; same process, same device, same state).
    let reset_view = ViewState {
        split: 1.0,
        ..Default::default()
    };
    let reset = render_view(&g, &mut composer, &target, &grid, &reset_view);
    assert_eq!(
        full, reset,
        "zoom reset must restore the unzoomed rendering"
    );

    // The waterfall obeys the same window: hot bin 40 moves to
    // (40/255)/0.5 of the width when the left half is the view.
    let mut row = vec![-100.0f32; BINS];
    row[40] = 0.0;
    for _ in 0..30 {
        composer.push_waterfall_row(&g.device, &g.queue, &row, None);
    }
    let mut view = ViewState {
        split: 0.0,
        ..Default::default()
    };
    view.params.zoom = ZoomSpan::new(0.0, 0.5);
    let img = render_view(&g, &mut composer, &target, &[], &view);
    let bright = |p: [u8; 4]| p[..3].iter().map(|&c| c as u32).sum::<u32>();
    let hot_x = (((40.0 / (BINS - 1) as f32) / 0.5) * W as f32) as u32;
    let hot = px(&img, hot_x, 5, W);
    let quiet = px(&img, hot_x + 60, 5, W);
    assert!(
        bright(hot) > bright(quiet) + 60,
        "zoomed waterfall hot bin misplaced: hot {hot:?} vs quiet {quiet:?}"
    );
}

#[test]
fn pause_freezes_the_data_and_resume_releases_it() {
    let g = gpu();
    let mut composer = FrameComposer::new(&g.device, OFFSCREEN_FORMAT);
    composer.set_gamma(GOLDEN_GAMMA);
    let target = OffscreenTarget::new(&g.device, W, H);
    let a = synthetic_grid();
    // A visibly different scene: a full-scale block where `a` is empty.
    let mut b = vec![0.0f32; BINS * LEVELS];
    for k in 208..240 {
        for l in 40..60 {
            b[k * LEVELS + l] = 1.0;
        }
    }
    let live = ViewState {
        split: 1.0,
        ..Default::default()
    };
    let img_a = render_view(&g, &mut composer, &target, &a, &live);

    // Paused: new data must not reach the screen (FR-D12 — the display
    // freezes, the source keeps draining upstream).
    let paused = ViewState {
        split: 1.0,
        paused: true,
        ..Default::default()
    };
    let img_frozen = render_view(&g, &mut composer, &target, &b, &paused);
    assert_eq!(img_a, img_frozen, "paused frame took new data");

    // Resume: the new data appears.
    let img_b = render_view(&g, &mut composer, &target, &b, &live);
    assert_ne!(img_a, img_b, "resume did not release the freeze");
}

/// D-031 produce-and-consume for the FR-D5 waterfall time labels, production
/// types end to end: a real §7.6 [`RowCadence`] drives a real
/// [`RowAggregator`]; its rows enter the display through the same
/// `push_waterfall_row` seam the app uses, stating the aggregator's own
/// interval; the view mirrors `waterfall_row_interval()` exactly as the
/// windowed and headless frame loops do; and the chrome then renders time
/// labels whose values follow that cadence. No hand-typed interval anywhere.
#[test]
fn waterfall_time_labels_come_from_the_real_cadence() {
    use phosphene_render::{
        axis, chrome, AnnotationFeed, AnnotationOverlayState, HudStats, Layout, RowAggregation,
        RowAggregator, RowCadence,
    };

    let g = gpu();
    let mut composer = FrameComposer::new(&g.device, OFFSCREEN_FORMAT);

    // Produce: 30 s over 300 panel rows → 0.1 s per row.
    let cadence = RowCadence::new(30.0, 300);
    let mut agg = RowAggregator::new(BINS, RowAggregation::Max, cadence);
    let interval = agg
        .row_interval_s()
        .expect("a seconds-based cadence states its interval");
    let spectrum = vec![-80.0f32; BINS];
    // 2 s of signal at 50 ms per spectrum → 20 emitted rows.
    for _ in 0..40 {
        agg.push_spectrum(&spectrum, 0.05, |row| {
            composer.push_waterfall_row(&g.device, &g.queue, row, Some(interval));
        });
    }
    assert_eq!(composer.waterfall_row_interval(), Some(interval));

    // Consume: the production frame loops' mirror line, verbatim.
    let mut view = ViewState {
        split: 0.0,
        ..Default::default()
    };
    view.params.wf_row_interval_s = composer.waterfall_row_interval();

    // The chrome labels the waterfall time axis from that mirrored value.
    let ctx = egui::Context::default();
    let theme = Theme::default();
    theme.install(&ctx);
    let stats = HudStats {
        fps: 60.0,
        frame_ms: 2.5,
        fft_size: 1024,
        window_name: "HANN",
        source_label: "TEST".into(),
        backend: "cpu",
        colormap: "p7",
        input_rate_sps: 2_048_000.0,
        processed_pct: 100.0,
        ffts_per_s: 2000.0,
        dropped_batches: 0,
        // This test only asserts the waterfall time labels; the τ/gamma
        // readouts are exercised in `chrome.rs`'s own tests, so arbitrary
        // fixed values suffice here (phosphene-render has no dependency on
        // phosphene-core's defaults).
        persistence_tau_s: 1.0,
        gamma: 0.7,
        live_tau_s: 0.1,
        max_hold_tau_s: 2.0,
        overflow_events: 0,
        ..Default::default()
    };
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 720.0),
        )),
        ..Default::default()
    };
    let mut grid = egui::Rect::NOTHING;
    let feed = AnnotationFeed::new();
    let mut overlay_state = AnnotationOverlayState::new();
    let out = ctx.run_ui(input, |ui| {
        grid = chrome::draw(ui, &theme, &mut view, &stats, &feed, &mut overlay_state).grid;
    });
    let mut texts = Vec::new();
    for clipped in &out.shapes {
        if let egui::Shape::Text(t) = &clipped.shape {
            texts.push(t.galley.text().to_owned());
        }
    }
    out.drop_without_applying_deltas();

    // Expected labels derive from the aggregator's cadence and the real
    // waterfall region height — the same arithmetic the chrome uses.
    let regions = Layout { grid }.split(view.split);
    let expected = axis::time_ticks(interval, regions.waterfall.height());
    assert!(
        !expected.is_empty(),
        "test premise: the panel must be tall enough for time labels"
    );
    for tick in &expected {
        assert!(
            texts.contains(&tick.label),
            "time label {:?} from the real cadence missing; texts: {texts:?}",
            tick.label
        );
    }
}

/// FR-D3 render seal: the max-hold overlay draws from `RtsaInput`'s
/// `max_hold_bins` — visually distinct from the live trace (amber against
/// the yellow-green core), subordinate to it (D-010: it must not out-shout
/// the live spectrum), gone when the input is empty ("empty means not
/// shown"), and held — but still hideable — under the FR-D12 freeze.
/// Golden per the M1-B harness (D-014 perceptual tolerance).
#[test]
fn max_hold_trace_draws_distinct_subordinate_and_hideable() {
    let g = gpu();
    let mut composer = FrameComposer::new(&g.device, OFFSCREEN_FORMAT);
    composer.set_gamma(GOLDEN_GAMMA);
    let target = OffscreenTarget::new(&g.device, W, H);

    // A live spike at bin 64 (−20 dBFS) over a −85 dBFS floor; the max hold
    // shares the floor but keeps its own peak at bin 192 (−30 dBFS), where
    // the live trace has fallen away — the retained-transient shape FR-D3
    // exists to show. The peaks span three bins so the 256-bin → 400-column
    // upsample keeps their tops (a one-bin spike interpolates away).
    let mut live = vec![-85.0f32; BINS];
    live[63..=65].fill(-20.0);
    let mut maxh = vec![-85.0f32; BINS];
    maxh[191..=193].fill(-30.0);
    let view = ViewState {
        split: 1.0,
        ..Default::default()
    };

    let base = render_traces(&g, &mut composer, &target, &[], &view, &live, &maxh);
    assert_matches_golden(&base, W, H, "max-hold-trace");

    let bright = |p: [u8; 4]| p[..3].iter().map(|&c| c as u32).sum::<u32>();
    let probe = |img: &[u8], x0: i32, y0: i32| {
        let mut best = [0u8; 4];
        for dy in -3i32..=3 {
            for dx in -2i32..=2 {
                let x = (x0 + dx).clamp(0, W as i32 - 1) as u32;
                let y = (y0 + dy).clamp(0, H as i32 - 1) as u32;
                let p = px(img, x, y, W);
                if bright(p) > bright(best) {
                    best = p;
                }
            }
        }
        best
    };
    // The max-hold-only peak: x = 192/255·(W−1) ≈ 300. Its row is taken
    // from the view's OWN dB window rather than a literal — D-065 §5 moved
    // the default range from 110 dB to 100, and a probe carrying the old
    // arithmetic would sit on the wrong pixels while still passing. The
    // live spike column sits at x ≈ 100; x = 200 has only the −85 floor.
    let row_of = |db: f32| ((1.0 - view.params.db_to_unit(db)) * H as f32) as i32;
    let amber = probe(&base, 300, row_of(-30.0));
    let quiet = probe(&base, 200, row_of(-30.0));
    assert!(
        bright(amber) > bright(quiet) + 60,
        "the max-hold line is not drawn: {amber:?} vs {quiet:?}"
    );
    // Distinct: the amber line is red-dominant, the live core
    // green-dominant.
    assert!(
        amber[0] > amber[1],
        "max-hold line is not the amber style: {amber:?}"
    );
    let live_core = probe(&base, 100, row_of(-20.0));
    assert!(
        live_core[1] > live_core[0],
        "live core hue changed: {live_core:?}"
    );
    // Subordinate (D-010): the max-hold line never outshines the live core.
    assert!(
        bright(amber) < bright(live_core),
        "the max-hold line out-shouts the live trace: {amber:?} vs {live_core:?}"
    );

    // Empty means "not shown".
    let hidden = render_traces(&g, &mut composer, &target, &[], &view, &live, &[]);
    assert!(
        bright(probe(&hidden, 300, row_of(-30.0))) + 60 < bright(amber),
        "an empty max_hold_bins still drew a line"
    );

    // FR-D12 freeze: while paused, new max-hold data must not reach the
    // screen…
    let shown = render_traces(&g, &mut composer, &target, &[], &view, &live, &maxh);
    let paused_view = ViewState {
        split: 1.0,
        paused: true,
        ..Default::default()
    };
    let mut moved = vec![-85.0f32; BINS];
    moved[99..=101].fill(-30.0);
    let frozen = render_traces(&g, &mut composer, &target, &[], &paused_view, &live, &moved);
    assert_eq!(shown, frozen, "the paused frame took new max-hold data");
    // …but hiding is a view command, honoured even while frozen.
    let frozen_hidden = render_traces(&g, &mut composer, &target, &[], &paused_view, &live, &[]);
    assert!(
        bright(probe(&frozen_hidden, 300, row_of(-30.0))) + 60 < bright(amber),
        "the M-key hide was ignored while paused"
    );
}

/// The FR-D12 screenshot path renders in the window's own surface format —
/// on most platforms BGRA. The readback must swizzle so the PNG matches
/// what an RGBA render shows.
#[test]
fn bgra_offscreen_readback_matches_rgba() {
    let g = gpu();
    let grid = synthetic_grid();
    let theme = Theme::default();
    let view = ViewState {
        split: 1.0,
        ..Default::default()
    };
    let mut images = Vec::new();
    for format in [
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8UnormSrgb,
    ] {
        let mut composer = FrameComposer::new(&g.device, format);
        composer.set_gamma(GOLDEN_GAMMA);
        let target = OffscreenTarget::with_format(&g.device, W, H, format);
        composer.render_rtsa_frame(
            &g.device,
            &g.queue,
            &target.view,
            [W, H],
            RtsaInput {
                live_bins: &[],
                max_hold_bins: &[],
                intensity: &grid,
                intensity_bins: BINS,
                intensity_levels: LEVELS,
                surface_points: egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(W as f32, H as f32),
                ),
                view: &view,
                theme: &theme,
            },
            empty_chrome(),
        );
        images.push(target.read_rgba(&g.device, &g.queue));
    }
    let mean = images[0]
        .iter()
        .zip(&images[1])
        .map(|(a, b)| a.abs_diff(*b) as u64)
        .sum::<u64>() as f64
        / images[0].len() as f64;
    assert!(
        mean < 1.5,
        "BGRA readback diverges from RGBA (mean abs diff {mean:.2}) — swizzle broken"
    );
}
