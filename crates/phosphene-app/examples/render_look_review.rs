// SPDX-License-Identifier: MIT

//! D-100/AC-15: produces the four-colormap look-review renders in **one
//! command**, deterministically — no retry, no luck. `main` exits non-zero
//! (after printing which map failed and why) if any render has no
//! annotations, per the fix-pass-3 instruction: "never rerun until it
//! works."
//!
//! ## Why this is a separate example, not `--headless --analyze --source file:...`
//!
//! The CLI's file-source path runs the live analysis worker on its own
//! thread, wall-clock-cadenced (NFR-A1) against a source that is itself
//! either real-time-paced or racing the compute thread — two independent
//! timers with no rendezvous. A duty-cycled signal's track can legitimately
//! be between bursts at the exact instant a fixed `--frames` count ends
//! (`analyze_goldens.rs` documents and retries past exactly this), and nemo
//! amount of retrying turns that into a **guarantee**, only a probability.
//!
//! This example never starts that worker. It drives the same public
//! detect → track → measure → annotate pipeline
//! (`phosphene_analyze::{Detector, Tracker, Measurer, build_annotation}` —
//! the exact sequence `phosphene_app::analyze::AnalysisState::cycle` runs,
//! copied here since that type is private to the `phosphene` binary and an
//! example cannot reach it) directly, synchronously, frame-count only — the
//! same NFR-A4 determinism `--headless`'s own synthetic path relies on.
//!
//! ## Why steady carriers, not a burst
//!
//! A **continuously**-occupied bin is the A0 per-bin noise-floor
//! estimator's own documented blind spot (`phosphene_analyze::detect::noise`
//! module docs, confirmed by
//! `phosphene_analyze::detect::seal_tests::
//! robust_floor_is_not_pulled_up_by_strong_occupants`): once a signal has
//! occupied a bin for the estimator's whole rolling window (64 frames by
//! default, 16 ms at this rate), the window contains nothing but the
//! signal, and the "floor" the CFAR threshold is set against becomes the
//! signal's own level. A signal that is *always on* is not detectable
//! forever — only for the ~16 ms after it turns on, while the window still
//! holds some of the noise-only history from before.
//!
//! So "steady, present at the capture moment by construction" is
//! implemented as: warm the floor on noise alone first, then turn the
//! carriers on and analyze only a short, fixed window of frames
//! immediately afterward (`SIGNAL_FRAMES`, chosen with wide margin under
//! the 64-frame saturation point) — comfortably enough for the tracker's
//! M-of-N birth (3 of 5) to confirm, nowhere near enough to pull the floor
//! up. The resulting annotation set is then frozen and republished every
//! display frame for the rest of the render: the carriers stay on-screen in
//! the spectrum trace, persistence and waterfall throughout (they are
//! genuinely present the whole time), but the *live worker's* window is
//! deliberately kept inside its safe zone rather than left running for the
//! whole capture, which would eventually rediscover the same blind spot.
//!
//! ## The scene
//!
//! Four signals, chosen against the design's §8 look questions (does the
//! band/label/declutter/detail read well at a glance): two narrow single
//! tones close enough to force a fixed-lane label collision (AC-4's
//! degrade ladder), and two multi-tone clusters of different widths and
//! strengths, so a blob-merged wider occupied bandwidth reads differently
//! from a point signal.

use std::io::BufWriter;
use std::path::PathBuf;

use phosphene_analyze::detect::{Detector, DetectorConfig};
use phosphene_analyze::measure::{MeasureConfig, Measurer};
use phosphene_analyze::track::{Tracker, TrackerConfig};
use phosphene_analyze::{build_annotation, BandMeta, FrameView};
use phosphene_core::{
    Complex, MaxHold, PersistenceHistogram, SpectrumAnalyzer, WindowKind, DBFS_FLOOR,
    DEFAULT_LEVELS,
};
use phosphene_render::{
    chrome, derive_cadence, Colormap, DisplayParams, FrameComposer, HudStats, LabelStyle, Layout,
    OffscreenTarget, RowAggregation, RowAggregator, RtsaInput, Theme, ViewState,
    DEFAULT_DECLUTTER_CAP,
};
use phosphene_sources::{SigGen, SigGenConfig, ToneConfig};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FFT: usize = 512;
const RATE_HZ: f64 = 2_048_000.0;
const DT: f32 = 1.0 / 60.0;

/// Raw frames of pure noise fed to the detector before any carrier turns
/// on — long enough (75 ms) to fill the 64-frame noise-floor window several
/// times over with a clean, signal-free reference.
const WARMUP_FRAMES: u32 = 300;
/// Raw frames of carrier-on analysis after warmup — comfortably under the
/// 64-frame window-saturation point (see the module doc), comfortably over
/// the 3-of-5 birth window.
const SIGNAL_FRAMES: u32 = 24;
/// Display frames rendered after the frozen annotation set is established,
/// to give the persistence histogram and waterfall something to show.
const RENDER_FRAMES: u32 = 180;

fn scene_config(seed: u64) -> SigGenConfig {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = seed;
    cfg.noise = Some(phosphene_sources::NoiseConfig { level_dbfs: -70.0 });
    // Two narrow tones close enough to collide in the fixed label lane
    // (AC-4): the design's own degrade ladder is only visible with a real
    // collision to resolve.
    cfg.tones.push(ToneConfig {
        offset_hz: -400_000.0,
        level_dbfs: -12.0,
    });
    cfg.tones.push(ToneConfig {
        offset_hz: -370_000.0,
        level_dbfs: -18.0,
    });
    // A medium cluster: 5 tones 4 kHz apart (adjacent bins at this FFT
    // size), spanning 16 kHz — merges into one wider blob than a point
    // tone, at a middling strength.
    for i in 0..5i64 {
        cfg.tones.push(ToneConfig {
            offset_hz: -100_000.0 + (i - 2) as f64 * 4_000.0,
            level_dbfs: -22.0,
        });
    }
    // A wide, weaker cluster: 15 tones 4 kHz apart, spanning 56 kHz.
    for i in 0..15i64 {
        cfg.tones.push(ToneConfig {
            offset_hz: 150_000.0 + (i - 7) as f64 * 4_000.0,
            level_dbfs: -30.0,
        });
    }
    cfg
}

fn band_meta() -> BandMeta {
    BandMeta {
        center_freq_hz: None,
        span_hz: RATE_HZ,
        bins: FFT,
        bin_hz: RATE_HZ / FFT as f64,
        frame_dt_s: FFT as f64 / RATE_HZ,
        cal_offset_db: 0.0,
    }
}

/// The exact detect → track → measure → annotate sequence
/// `phosphene_app::analyze::AnalysisState::cycle` runs (that type is
/// private to the `phosphene` binary crate, unreachable from an example) —
/// copied here against the same public `phosphene_analyze` API, so this
/// stays the same engine, not a reimplementation of it.
struct Engine {
    detector: Detector,
    tracker: Tracker,
    measurer: Measurer,
}

impl Engine {
    fn new(meta: &BandMeta) -> Self {
        Engine {
            detector: Detector::new(meta.bins, DetectorConfig::new())
                .expect("default detector config is valid"),
            tracker: Tracker::new(TrackerConfig::default())
                .expect("default tracker config is valid"),
            measurer: Measurer::new(MeasureConfig::default())
                .expect("default measure config is valid"),
        }
    }

    fn cycle(&mut self, view: &FrameView, meta: &BandMeta) -> Vec<phosphene_render::Annotation> {
        let last_index = view.len() - 1;
        for i in 0..view.len() {
            let seq = view.seq(i);
            let mut detections = self.detector.push_frame(view.frame(i), seq);
            if i == last_index {
                detections.extend(self.detector.finish());
            }
            self.tracker.observe(seq, &detections, meta);
        }
        let noise_floor = self.detector.noise_floor().to_vec();
        let mut annotations = Vec::new();
        for track in self.tracker.tracks() {
            let mut track = track.clone();
            if let Ok(m) = self
                .measurer
                .measure_track(&track, view, meta, &noise_floor)
            {
                track.measurements = Some(m);
            }
            if let Some(ann) = build_annotation(&track, meta, true) {
                annotations.push(convert(ann));
            }
        }
        annotations
    }
}

fn convert(a: phosphene_analyze::Annotation) -> phosphene_render::Annotation {
    phosphene_render::Annotation {
        track: phosphene_render::TrackId(a.track.0),
        span: phosphene_render::FreqSpanHz {
            low_hz: a.span.low_hz,
            high_hz: a.span.high_hz,
        },
        anchor_hz: a.anchor_hz,
        label: a.label,
        detail: a.detail,
        confidence: a.confidence,
        strength_db: a.strength_db,
    }
}

/// The waterfall panel's height in physical pixel rows for this frame —
/// copied from `phosphene_app::window::waterfall_panel_rows`, which is
/// `pub(crate)` and unreachable from an example; the logic itself is
/// trivial (one `Layout::split` call), so this is not a maintenance risk.
fn waterfall_panel_rows(grid: egui::Rect, split: f32, pixels_per_point: f32) -> u32 {
    let regions = Layout { grid }.split(split);
    let wf = (regions.waterfall.height() * pixels_per_point).round() as u32;
    if wf >= 1 {
        wf
    } else {
        ((grid.height() * pixels_per_point).round() as u32).max(1)
    }
}

fn write_png(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) {
    let file = std::fs::File::create(path)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", path.display()));
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("PNG header");
    writer.write_image_data(rgba).expect("PNG data");
}

/// Renders one colormap's PNG, at `width`x`height`, over `scene`, in
/// `label_style` (D-102 items 1-3). Returns the annotation labels present in
/// the frozen set (never empty on success — panics with a clear message
/// before ever reaching the render if the deterministic analysis window
/// found nothing, which would mean the scene or the engine changed under
/// this example, not bad luck) and the **effective** declutter cap the final
/// rendered frame actually used (D-102 item 3) — `view_state.declutter_cap`
/// scaled to that frame's real plot-grid width, read back from the same
/// `ChromeResponse` the production frame loop gets, not recomputed by hand.
#[allow(clippy::too_many_arguments)] // every argument is a genuinely distinct scene/render parameter
fn render_one(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    map: Colormap,
    width: u32,
    height: u32,
    label_style: LabelStyle,
    scene: SigGenConfig,
    out: &std::path::Path,
) -> (Vec<String>, u32) {
    let meta = band_meta();
    let mut engine = Engine::new(&meta);

    // Phase 1: warm the noise floor on noise alone, then analyze a short
    // fixed window right after the carriers turn on (module doc explains
    // why both halves of this are necessary). One `SigGen`, one continuous
    // deterministic stream — the carriers are configured from the start,
    // but only fed to the engine from `WARMUP_FRAMES` on, which is exactly
    // equivalent to "silent, then on" for the engine's own purposes since
    // it never sees the earlier samples.
    let mut noise_only = SigGenConfig::new(RATE_HZ);
    noise_only.seed = 1;
    noise_only.noise = Some(phosphene_sources::NoiseConfig { level_dbfs: -70.0 });
    let mut noise_gen = SigGen::new(noise_only).expect("valid noise-only scene");
    let mut analyzer = SpectrumAnalyzer::new(FFT, WindowKind::Hann).expect("valid FFT size");
    let mut chunk = vec![Complex::new(0.0f32, 0.0); FFT];
    let mut spectrum = vec![0.0f32; FFT];

    let total = WARMUP_FRAMES + SIGNAL_FRAMES;
    let mut view = FrameView::new(FFT, total as usize);
    let mut seq = 0u64;
    for _ in 0..WARMUP_FRAMES {
        noise_gen.fill(&mut chunk);
        analyzer.process(&chunk, &mut spectrum);
        view.push(&spectrum, seq);
        seq += 1;
    }
    let mut scene_gen = SigGen::new(scene).expect("valid look-review scene");
    for _ in 0..SIGNAL_FRAMES {
        scene_gen.fill(&mut chunk);
        analyzer.process(&chunk, &mut spectrum);
        view.push(&spectrum, seq);
        seq += 1;
    }
    let annotations = engine.cycle(&view, &meta);
    assert!(
        !annotations.is_empty(),
        "{}: the deterministic analysis window produced no annotations — \
         the scene or the engine's tuning changed under this example; \
         this is a real bug, not bad luck (see the module doc)",
        map.name()
    );

    // Phase 2: render. The frozen `annotations` are republished unchanged
    // every display frame (never re-derived — the point is a stable,
    // non-flickering picture of a known-good analysis result), while the
    // scene generator keeps producing fresh samples for the visible
    // spectrum trace, persistence histogram and waterfall, exactly as a
    // real live carrier would look moments after this capture.
    let theme = Theme::default();
    let mut view_state = ViewState {
        params: DisplayParams {
            sample_rate: Some(RATE_HZ),
            ..Default::default()
        },
        label_style,
        ..Default::default()
    };
    let ctx = egui::Context::default();
    theme.install(&ctx);

    let target = OffscreenTarget::new(device, width, height);
    let mut composer = FrameComposer::new(device, phosphene_render::OFFSCREEN_FORMAT);
    composer.set_colormap(map);

    let wf_spectrum_dt = (FFT as f64 / RATE_HZ) as f32;
    let wf_cadence = |panel_rows: u32, view: &ViewState| {
        derive_cadence(
            view.wf_mode,
            Some(wf_spectrum_dt),
            view.wf_time_span_s,
            view.wf_fast_interval_s,
            view.wf_span_rows,
            panel_rows,
        )
    };
    let mut wf_agg = RowAggregator::new(FFT, RowAggregation::default(), wf_cadence(1, &view_state));
    let mut wf_rows: Vec<f32> = Vec::new();
    let mut wf_panel_rows = 1u32;
    let mut histo = PersistenceHistogram::new(
        FFT,
        DEFAULT_LEVELS,
        view_state.params.db_bottom,
        view_state.params.db_top,
    )
    .expect("valid persistence histogram");
    let mut max_hold = MaxHold::new(FFT).expect("valid max-hold trace");
    let mut bins = vec![DBFS_FLOOR; FFT];

    let annotation_feed = phosphene_render::AnnotationFeed::new();
    let mut overlay_state = phosphene_render::AnnotationOverlayState::new();

    let mut hud = HudStats {
        fps: 1.0 / DT,
        frame_ms: 0.0,
        fft_size: FFT,
        window_name: "HANN",
        source_label: "LOOK-REVIEW".into(),
        backend: "cpu",
        colormap: composer.colormap().name(),
        input_rate_sps: RATE_HZ,
        processed_pct: 100.0,
        ffts_per_s: (RATE_HZ / FFT as f64),
        dropped_batches: 0,
        persistence_tau_s: histo.tau_decay(),
        gamma: composer.gamma(),
        live_tau_s: phosphene_core::DEFAULT_TAU_LIVE,
        max_hold_tau_s: max_hold.tau(),
        overflow_events: 0,
        inspector_active: true,
        ..Default::default()
    };

    let mut effective_cap = view_state.declutter_cap;
    for _ in 0..RENDER_FRAMES {
        view_state.params.wf_row_interval_s = composer.waterfall_row_interval();
        view_state.params.wf_row_spectra = composer.waterfall_row_spectra();
        max_hold.tick(DT, &bins);
        wf_agg.set_mode(view_state.wf_aggregation);
        wf_agg.set_cadence(wf_cadence(wf_panel_rows, &view_state));
        wf_rows.clear();
        // One display frame's worth of spectra — this only feeds the
        // display accumulators (trace, persistence, waterfall), never the
        // engine again: the analysis window stays frozen at Phase 1's
        // result (module doc).
        let spectra_per_frame = ((RATE_HZ * f64::from(DT)) / FFT as f64).round() as usize;
        for i in 0..spectra_per_frame.max(1) {
            scene_gen.fill(&mut chunk);
            analyzer.process(&chunk, &mut spectrum);
            if i == spectra_per_frame.max(1) - 1 {
                bins.copy_from_slice(&spectrum);
            }
            histo.accumulate(&spectrum);
            max_hold.accumulate(&spectrum);
            wf_agg.push_spectrum(&spectrum, wf_spectrum_dt, |row| {
                wf_rows.extend_from_slice(row);
            });
        }
        histo.tick(DT);
        annotation_feed.push_annotations(&annotations);
        hud.track_count = annotation_feed.snapshot().len();

        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width as f32, height as f32),
            )),
            ..Default::default()
        };
        let mut grid = egui::Rect::NOTHING;
        let output = ctx.run_ui(input, |ui| {
            grid = chrome::draw(
                ui,
                &theme,
                &mut view_state,
                &hud,
                &annotation_feed,
                &mut overlay_state,
            )
            .grid;
        });
        let primitives = ctx.tessellate(output.shapes, output.pixels_per_point);
        wf_panel_rows = waterfall_panel_rows(grid, view_state.split, output.pixels_per_point);
        composer.push_waterfall_rows(device, queue, &wf_rows, FFT, Some(wf_agg.cadence()));
        composer.set_waterfall_auto_range(view_state.wf_auto_range);
        composer.set_waterfall_gamma(view_state.wf_gamma);
        composer.render_rtsa_frame(
            device,
            queue,
            &target.view,
            [width, height],
            RtsaInput {
                live_bins: &bins,
                max_hold_bins: if view_state.max_hold {
                    max_hold.trace()
                } else {
                    &[]
                },
                intensity: histo.intensity(),
                intensity_bins: histo.bins(),
                intensity_levels: histo.levels(),
                surface_points: grid,
                view: &view_state,
                theme: &theme,
            },
            phosphene_render::EguiFrame {
                primitives,
                textures_delta: output.textures_delta,
                pixels_per_point: output.pixels_per_point,
            },
        );
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        // D-102 item 3: the effective declutter cap at *this* frame's real
        // plot-grid width — read after the last frame below, from the exact
        // `grid` the production frame loop just used, never a hand-guessed
        // margin subtraction.
        if grid.width() > 0.0 {
            effective_cap = phosphene_render::layout::scaled_declutter_cap(
                view_state.declutter_cap,
                grid.width(),
            );
        }
    }

    let rgba = target.read_rgba(device, queue);
    write_png(out, width, height, &rgba);

    let final_snapshot = annotation_feed.snapshot();
    assert!(
        !final_snapshot.is_empty(),
        "{}: the annotation feed went empty by the final frame even though \
         it was pushed unchanged every frame — this is a real bug",
        map.name()
    );
    (
        final_snapshot.iter().map(|a| a.label.clone()).collect(),
        effective_cap,
    )
}

/// D-100 names exactly these four maps. Sourced from `Colormap::ALL` (the
/// same list `parse_colormap` answers to) rather than a second, independent
/// hard-coded array — a fix-pass-4 review deleted `Turbo` from a local copy
/// of this array and the command still exited 0, wrote three files, and
/// printed "all four" (the print was a literal, not derived from what was
/// actually written). Asserting `Colormap::ALL` still equals this exact
/// four-tuple means the same class of mutation, even applied to the shared
/// enum's own list, fails loudly here before anything renders, instead of
/// this script silently adapting to fewer maps.
fn expected_maps() -> [Colormap; 4] {
    let maps = Colormap::ALL;
    assert_eq!(
        maps,
        [
            Colormap::P7,
            Colormap::Inferno,
            Colormap::Viridis,
            Colormap::Turbo
        ],
        "D-100 names exactly four maps (p7, inferno, viridis, turbo); \
         Colormap::ALL no longer matches — this script's contract with the \
         design has changed upstream and needs a human, not a silent \
         fewer-file run"
    );
    maps
}

fn expected_path(out_dir: &std::path::Path, map: Colormap) -> PathBuf {
    out_dir.join(format!("ac15-colormap-{}.png", map.name()))
}

/// D-102 item 1: the stacked style, rendered in P7 and Turbo beside the
/// matching single-line files above — enough for the owner to compare both
/// styles without rendering all four maps twice.
const STACKED_REVIEW_MAPS: [Colormap; 2] = [Colormap::P7, Colormap::Turbo];

fn stacked_path(out_dir: &std::path::Path, map: Colormap) -> PathBuf {
    out_dir.join(format!("d102-stacked-{}.png", map.name()))
}

/// D-102 item 2: a crowded scene at a large window, showing the declutter
/// cap (D-102 item 3) grown past its default-width value of 8.
fn crowded_path(out_dir: &std::path::Path) -> PathBuf {
    out_dir.join("d102-crowded-large-window.png")
}

const CROWDED_WIDTH: u32 = 1920;
const CROWDED_HEIGHT: u32 = 1080;

/// A crowded scene (D-102 item 2): more isolated tones than the default
/// declutter cap of 8, spaced far enough apart (tens of bin widths at this
/// FFT size) to track as distinct signals rather than merging into one
/// blob, at a range of levels so the §2.3/§2.4 ranking has real work to do.
fn crowded_scene_config(seed: u64) -> SigGenConfig {
    let mut cfg = SigGenConfig::new(RATE_HZ);
    cfg.seed = seed;
    cfg.noise = Some(phosphene_sources::NoiseConfig { level_dbfs: -70.0 });
    const N: i64 = 12;
    for i in 0..N {
        cfg.tones.push(ToneConfig {
            offset_hz: -900_000.0 + (i as f64) * (1_700_000.0 / (N - 1) as f64),
            level_dbfs: -12.0 - (i % 4) as f32 * 6.0,
        });
    }
    cfg
}

/// Removes each of `maps`' expected output files if present, so a stale
/// file surviving from an earlier run into this (reused) output directory
/// can never be mistaken for evidence that *this* run wrote it. Missing
/// files are not an error — the common case, a fresh directory.
fn clean_expected_files(out_dir: &std::path::Path, maps: &[Colormap]) -> std::io::Result<()> {
    for &map in maps {
        match std::fs::remove_file(expected_path(out_dir, map)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// A fast, non-cryptographic content hash (FNV-1a, 64-bit) — sufficient to
/// catch an accidentally-identical file (a renderer bug that silently
/// reused one map's output for another, or a stale write that landed
/// unmodified); this is not a security boundary, just an equality check
/// that need not hold the whole file in memory twice to compare.
fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes
        .iter()
        .fold(OFFSET, |h, &b| (h ^ u64::from(b)).wrapping_mul(PRIME))
}

/// The negative control this command's success claim is built from (fix
/// pass 4): reads every expected file back **from the filesystem**, after
/// rendering has finished, independent of whatever the in-process render
/// loop believed happened. Fails if any expected file is missing, empty, or
/// content-identical to another (by [`fnv1a`]) — a renderer that silently
/// skips a map, or reuses another map's output, is caught here even if the
/// render loop itself reported no error. Returns `(filename, byte size,
/// content hash)` per map, in `maps` order, which is the only source the
/// final success line is built from — never a literal count.
fn verify_renders(
    out_dir: &std::path::Path,
    maps: &[Colormap],
) -> Result<Vec<(String, u64, u64)>, String> {
    let mut found = Vec::with_capacity(maps.len());
    for &map in maps {
        let path = expected_path(out_dir, map);
        let bytes = std::fs::read(&path)
            .map_err(|e| format!("{}: expected but missing ({e})", path.display()))?;
        if bytes.is_empty() {
            return Err(format!("{}: exists but is empty", path.display()));
        }
        let name = path
            .file_name()
            .expect("expected_path always has a file name")
            .to_string_lossy()
            .into_owned();
        found.push((name, bytes.len() as u64, fnv1a(&bytes)));
    }
    for i in 0..found.len() {
        for j in (i + 1)..found.len() {
            if found[i].2 == found[j].2 {
                return Err(format!(
                    "{} and {} have identical content (hash {:016x}) — \
                     one of them was not actually (re)written this run",
                    found[i].0, found[j].0, found[i].2
                ));
            }
        }
    }
    Ok(found)
}

fn main() {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/tmp/ac15-colormap-review"));
    std::fs::create_dir_all(&out_dir)
        .unwrap_or_else(|e| panic!("cannot create {}: {e}", out_dir.display()));

    let maps = expected_maps();
    clean_expected_files(&out_dir, &maps)
        .unwrap_or_else(|e| panic!("cannot clean stale renders from {}: {e}", out_dir.display()));
    let mut extra_paths: Vec<PathBuf> = STACKED_REVIEW_MAPS
        .iter()
        .map(|&m| stacked_path(&out_dir, m))
        .collect();
    extra_paths.push(crowded_path(&out_dir));
    for path in &extra_paths {
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => panic!("cannot clean stale render {}: {e}", path.display()),
        }
    }

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .unwrap_or_else(|e| panic!("no usable GPU adapter ({e})"));
    let adapter_info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("phosphene-look-review"),
        ..Default::default()
    }))
    .expect("GPU device request failed");
    println!(
        "adapter: {} ({:?})",
        adapter_info.name, adapter_info.backend
    );

    let mut failures: Vec<String> = Vec::new();
    for map in maps {
        let out = expected_path(&out_dir, map);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            render_one(
                &device,
                &queue,
                map,
                WIDTH,
                HEIGHT,
                LabelStyle::SingleLine,
                scene_config(2),
                &out,
            )
        }));
        match result {
            Ok((labels, cap)) => {
                println!(
                    "rendered {} ({} annotation(s), cap {}: {})",
                    out.display(),
                    labels.len(),
                    cap,
                    labels.join(" | ")
                );
            }
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown panic".to_string());
                eprintln!("FAILED {}: {msg}", map.name());
                failures.push(map.name().to_string());
            }
        }
    }

    if !failures.is_empty() {
        eprintln!(
            "look-review render failed for: {} — not writing a partial set as success",
            failures.join(", ")
        );
        std::process::exit(1);
    }

    // The actual success claim: not "the loop above didn't panic", but what
    // is verifiably sitting on disk right now, read back independently of
    // the render loop's own belief that it succeeded.
    match verify_renders(&out_dir, &maps) {
        Ok(found) => {
            println!(
                "wrote {} distinct, non-empty look-review render(s) to {}: {}",
                found.len(),
                out_dir.display(),
                found
                    .iter()
                    .map(|(name, size, hash)| format!("{name} ({size} bytes, hash {hash:016x})"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Err(e) => {
            eprintln!("look-review verification failed: {e}");
            std::process::exit(1);
        }
    }

    // D-102 item 1: the stacked style, in P7 and Turbo, beside the matching
    // single-line files above.
    let mut extra_failures: Vec<String> = Vec::new();
    for map in STACKED_REVIEW_MAPS {
        let out = stacked_path(&out_dir, map);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            render_one(
                &device,
                &queue,
                map,
                WIDTH,
                HEIGHT,
                LabelStyle::Stacked,
                scene_config(2),
                &out,
            )
        }));
        match result {
            Ok((labels, cap)) => {
                println!(
                    "rendered {} (stacked style, {} annotation(s), cap {}): {}",
                    out.display(),
                    labels.len(),
                    cap,
                    labels.join(" | ")
                );
            }
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown panic".to_string());
                eprintln!("FAILED stacked-{}: {msg}", map.name());
                extra_failures.push(format!("stacked-{}", map.name()));
            }
        }
    }

    // D-102 items 2/3: one crowded scene at a large window, so the cap-
    // scaling rule (`phosphene_render::layout::scaled_declutter_cap`) has
    // actually grown the effective cap past its default-width value of 8 by
    // the time the production frame loop reads it back.
    let crowded_out = crowded_path(&out_dir);
    let crowded_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (labels, cap) = render_one(
            &device,
            &queue,
            Colormap::P7,
            CROWDED_WIDTH,
            CROWDED_HEIGHT,
            LabelStyle::SingleLine,
            crowded_scene_config(3),
            &crowded_out,
        );
        assert!(
            labels.len() > DEFAULT_DECLUTTER_CAP as usize,
            "the crowded scene must produce more tracks than the default cap \
             ({DEFAULT_DECLUTTER_CAP}) or growing the cap past it proves nothing \
             — got {} tracks",
            labels.len()
        );
        assert!(
            cap > DEFAULT_DECLUTTER_CAP,
            "D-102 item 3: a {CROWDED_WIDTH}x{CROWDED_HEIGHT} window must scale the \
             cap past its default-width value of {DEFAULT_DECLUTTER_CAP} — got {cap}"
        );
        (labels, cap)
    }));
    match crowded_result {
        Ok((labels, cap)) => {
            println!(
                "rendered {} ({CROWDED_WIDTH}x{CROWDED_HEIGHT}, {} annotation(s), \
                 effective declutter cap {cap}): {}",
                crowded_out.display(),
                labels.len(),
                labels.join(" | ")
            );
        }
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".to_string());
            eprintln!("FAILED crowded-large-window: {msg}");
            extra_failures.push("crowded-large-window".to_string());
        }
    }

    if !extra_failures.is_empty() {
        eprintln!(
            "look-review render failed for: {} — not writing a partial set as success",
            extra_failures.join(", ")
        );
        std::process::exit(1);
    }

    // Same filesystem negative control as `verify_renders`, for the extra
    // D-102 files: read back from disk, independent of the render loop's
    // own belief that it succeeded.
    for path in &extra_paths {
        match std::fs::read(path) {
            Ok(bytes) if bytes.is_empty() => {
                eprintln!(
                    "look-review verification failed: {}: exists but is empty",
                    path.display()
                );
                std::process::exit(1);
            }
            Ok(bytes) => {
                println!(
                    "wrote {} ({} bytes, hash {:016x})",
                    path.display(),
                    bytes.len(),
                    fnv1a(&bytes)
                );
            }
            Err(e) => {
                eprintln!(
                    "look-review verification failed: {}: expected but missing ({e})",
                    path.display()
                );
                std::process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &std::path::Path, map: Colormap, content: &[u8]) {
        std::fs::write(expected_path(dir, map), content).unwrap();
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "phosphene-look-review-verify-test-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn expected_maps_is_exactly_the_four_d100_maps() {
        assert_eq!(
            expected_maps(),
            [
                Colormap::P7,
                Colormap::Inferno,
                Colormap::Viridis,
                Colormap::Turbo
            ]
        );
    }

    #[test]
    fn verify_renders_passes_on_four_distinct_nonempty_files() {
        let dir = temp_dir("happy");
        let maps = expected_maps();
        for (i, &map) in maps.iter().enumerate() {
            write(&dir, map, format!("scene {i}").as_bytes());
        }
        let found = verify_renders(&dir, &maps).expect("four distinct files must verify");
        assert_eq!(found.len(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Fix pass 4's own negative control: a renderer that silently drops a
    /// map (the exact review-4 mutation, one level down — here the *output*
    /// is short by one file rather than the map list) must fail
    /// verification, not report success for what was actually written.
    #[test]
    fn verify_renders_fails_on_a_missing_map() {
        let dir = temp_dir("missing");
        let maps = expected_maps();
        for &map in &maps[..3] {
            write(&dir, map, b"present");
        }
        let err = verify_renders(&dir, &maps).expect_err("the 4th map's file was never written");
        assert!(err.contains(&expected_path(&dir, maps[3]).display().to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A stale file left over from a previous run, that this run's
    /// (simulated, buggy) renderer failed to regenerate, must not be
    /// mistaken for this run's own output: `clean_expected_files` removes
    /// it first, so a renderer that then fails to rewrite it leaves it
    /// correctly absent for `verify_renders` to catch — never present with
    /// old content passing as new.
    #[test]
    fn a_stale_file_does_not_survive_the_clean_step_to_fool_verification() {
        let dir = temp_dir("stale");
        let maps = expected_maps();
        for &map in &maps {
            write(&dir, map, b"stale content from a previous run");
        }
        clean_expected_files(&dir, &maps).unwrap();
        // Simulate a renderer that regenerates every map except one.
        for &map in &maps[..3] {
            write(&dir, map, format!("{:?}", map).as_bytes());
        }
        let err = verify_renders(&dir, &maps)
            .expect_err("the un-regenerated map must read as missing, not as its stale content");
        assert!(err.contains("missing"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_renders_fails_on_two_identical_files() {
        let dir = temp_dir("duplicate");
        let maps = expected_maps();
        write(&dir, maps[0], b"same bytes");
        write(&dir, maps[1], b"same bytes");
        write(&dir, maps[2], b"different");
        write(&dir, maps[3], b"also different");
        let err = verify_renders(&dir, &maps).expect_err("two identical files must be rejected");
        assert!(err.contains("identical content"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_renders_fails_on_an_empty_file() {
        let dir = temp_dir("empty");
        let maps = expected_maps();
        for &map in &maps[..3] {
            write(&dir, map, b"present");
        }
        write(&dir, maps[3], b"");
        let err = verify_renders(&dir, &maps).expect_err("an empty file must be rejected");
        assert!(err.contains("empty"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
