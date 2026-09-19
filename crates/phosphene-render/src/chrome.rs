// SPDX-License-Identifier: MIT

//! The egui chrome: wordmark bar, status readouts, axis labels, cursor
//! readout, zoom gestures, keyboard map and help overlay.
//!
//! egui draws the chrome; the data surface and its grid are wgpu-drawn
//! (clarification C6, §13 item 9). Axis label positions are computed from the
//! same [`Layout`] the grid geometry uses — and both are snapped to the same
//! physical-pixel grid — which is what keeps text and grid exactly aligned at
//! any scale factor, fractional HiDPI included (FR-D5).
//!
//! All display numbers come from [`crate::axis`]: the labels and the cursor
//! readout share one mapping, so they agree at every zoom level by
//! construction (FR-D6). Per D-006 the power scale is labelled **dBFS/bin** —
//! never dBm, never an unqualified dB. Per D-028 §2 a source with no declared
//! rate renders a normalized axis with no Hz unit anywhere.

use egui::{Align2, FontId, Frame, Margin, Pos2, Rect, Sense, Ui};

use crate::annotation::AnnotationFeed;
use crate::axis;
use crate::keymap::{self, ChromeRequests, BINDINGS};
use crate::layout::{format_hz, DisplayParams, Layout, SplitRegions, ViewState};
use crate::overlay::{self, AnnotationOverlayState, VisibleTrack};
use crate::theme::{chrome_text, Theme};
use crate::zoom::ZoomSpan;

/// Live readouts for the status bar. The fps and frame time are *measured*
/// by the caller (the M0-C seal measures, it does not assert), and the four
/// FR-D11 honesty figures come straight from the pipeline's own D-013
/// accounting — never from a second estimate that could disagree with it.
#[derive(Debug, Clone, Default)]
pub struct HudStats {
    /// Frames per second over a recent window.
    pub fps: f32,
    /// Mean frame time in milliseconds over the same window.
    pub frame_ms: f32,
    /// Current FFT size.
    pub fft_size: usize,
    /// Analysis window name (e.g. "HANN").
    pub window_name: &'static str,
    /// Label of the active source (e.g. "SIGGEN").
    pub source_label: String,
    /// Active compute backend, as actually selected at runtime (FR-D11:
    /// `cpu` / `cpu-simd` / `gpu`).
    pub backend: &'static str,
    /// Name of the active colormap, read from the composer's own selection
    /// each frame ([`crate::gpu::FrameComposer::colormap`]) — the readout
    /// AUDIT-1 found missing: the C key cycles four maps, and the display
    /// must say which one is showing (M1-J, D-010's honest chrome).
    pub colormap: &'static str,
    /// FR-D11: measured input rate in samples/s, over the accounting's
    /// rolling window.
    pub input_rate_sps: f64,
    /// FR-D11 / D-013: `samples_in_completed_FFTs / samples_received` over a
    /// rolling window, in percent — exactly 100 until overloaded.
    pub processed_pct: f32,
    /// FR-D11: completed FFTs per second, from the same accounting.
    pub ffts_per_s: f64,
    /// FR-D11 / D-013: cumulative dropped **batches** (never samples).
    pub dropped_batches: u64,
    /// The §7.3 persistence decay τ now in effect, seconds — FR-D1's
    /// "persistence time" control (`,`/`.`), read from the pipeline's own
    /// accumulator (D-043: the readout must state the value the accumulator
    /// actually holds, never a mirrored copy).
    pub persistence_tau_s: f32,
    /// The §7.3 `gamma_adjust` exponent now in effect — the persistence
    /// intensity/gamma control (`PgUp`/`PgDn`), read from the composer's own
    /// selection (D-043).
    pub gamma: f32,
    /// The §7.4 live-trace averaging τ now in effect, seconds — FR-D2's
    /// "averaging time configurable" (`K`/`L`), read from the pipeline
    /// (D-043).
    pub live_tau_s: f32,
    /// The D-007 max-hold decay τ now in effect, seconds (`;`/`'`), read
    /// from the pipeline's max-hold accumulator (D-043). Shown regardless of
    /// the active mode: a value the keys silently changed must be visible
    /// even while max hold is off or in pure-hold mode, where it currently
    /// has no on-screen effect.
    pub max_hold_tau_s: f32,
    /// D-050: cumulative hardware overflow **events** reported by the
    /// source (`SoapySourceStats.overflows`) — "the device lost an unknown
    /// amount of data", never converted into a sample or batch count.
    /// Distinct from `dropped_batches`, which counts batches the pipeline
    /// itself shed.
    pub overflow_events: u64,
    /// Whether the FR-AU1 analysis pipeline is active this frame (the `I`
    /// key / `--analyze` flag). Drives the "INSPECTOR" status chip and the
    /// annotation overlay draw pass — `false` (the default, and always
    /// `false` when the binary was not compiled with the `analyze` feature)
    /// draws exactly v1's display (design AC-11).
    pub inspector_active: bool,
    /// Live track count this cycle, shown in the chip when
    /// `inspector_active` (design situations 1/3: "INSPECTOR · N TRACKS").
    pub track_count: usize,
    /// NFR-A5 / D-098 §7.1: the analysis pipeline is shedding tracks under
    /// overload this cycle — shown as a status-bar chip in the same
    /// honesty-HUD idiom as the drop/overflow readouts, silent otherwise.
    pub shedding: bool,
    /// D-125 Amendment 4 item 3, the owner's ruling: "show an honest
    /// coverage count: frames analysed out of frames received, never a
    /// silent drop." Cumulative since analysis started or last retuned —
    /// copied once per cycle from [`crate::AnalysisLoad`]'s field of the
    /// same name.
    pub frames_analysed: u64,
    /// D-125 Amendment 4 item 3: cumulative frames the source has produced
    /// over the same window. Always `>= frames_analysed`.
    pub frames_received: u64,
}

/// What one chrome pass hands back to the caller.
#[derive(Debug, Clone, Copy)]
pub struct ChromeResponse {
    /// The full data-surface rect (egui points) the wgpu scene must render
    /// into.
    pub grid: Rect,
    /// Requests for state the [`crate::gpu::FrameComposer`] owns — pass to
    /// [`crate::gpu::FrameComposer::apply_requests`].
    pub requests: ChromeRequests,
    /// The FR-D7 screenshot key was pressed. Acting on it (offscreen render
    /// + PNG write) is FR-D12 app wiring; this is the render-side seam.
    pub screenshot: bool,
    /// A **retune** was requested this frame, from the D-058 frequency box or
    /// the window-stepping slider — the frequency the user asked for, in Hz.
    ///
    /// Deliberately a *request*, not an action. The render crate has no
    /// device in it (§8.1, and structurally: it does not depend on
    /// `phosphene-sources`), so both widgets can only ask; the app applies it
    /// through the one `set(Control::CenterFreqHz)` path, and what the widgets
    /// then display is the device's readback. `None` on every frame the user
    /// did not ask.
    pub retune_hz: Option<f64>,
}

/// Minimum horizontal drag, in points, for a drag-select to count as a zoom
/// rather than a click.
const MIN_DRAG_PT: f32 = 4.0;

/// Scroll-wheel zoom rate: one ~40-point notch multiplies the zoom by 1.25.
const SCROLL_ZOOM_BASE: f64 = 1.25;
const SCROLL_NOTCH_PT: f64 = 40.0;

/// Histogram region shorter than this (points) draws no power labels — at
/// extreme splits there is no room for legible text (D-010: never at the
/// cost of legibility).
const MIN_LABELLED_HEIGHT: f32 = 40.0;

/// Draw all chrome for one frame into the root `Ui` (from
/// `Context::run_ui`), applying keyboard and mouse input to `view`, and
/// returning the grid rect plus the input's composer/app requests.
///
/// This is the form for a display with no device behind it — a file replay, a
/// pipe, the signal generator, and the headless renderer. A source that can
/// actually be tuned is drawn with [`draw_tunable`], which adds the D-058
/// retune controls to the same control bar and nothing else.
pub fn draw(
    ui: &mut Ui,
    theme: &Theme,
    view: &mut ViewState,
    stats: &HudStats,
    annotations: &AnnotationFeed,
    overlay_state: &mut AnnotationOverlayState,
) -> ChromeResponse {
    draw_tunable(
        ui,
        theme,
        view,
        stats,
        &TuneUi::default(),
        annotations,
        overlay_state,
    )
}

/// [`draw`], plus the **D-058 retune controls** for a source that advertises
/// tuning: a frequency box and a slider that steps the centre by one sample
/// rate — one window width — per increment.
///
/// Split from `draw` rather than folded into it so that `TuneUi`'s "no
/// device" default is the one `draw` can express: a source with nothing to
/// tune is drawn with no retune widgets at all, since an inert frequency box
/// in front of a file replay would be a control that cannot work.
///
/// The **control bar itself is always drawn**, tunable or not, because
/// D-065 §3's waterfall intensity control lives on it and every source has a
/// waterfall.
pub fn draw_tunable(
    ui: &mut Ui,
    theme: &Theme,
    view: &mut ViewState,
    stats: &HudStats,
    tune: &TuneUi,
    annotations: &AnnotationFeed,
    overlay_state: &mut AnnotationOverlayState,
) -> ChromeResponse {
    let mut requests = ChromeRequests::default();
    let now = ui.ctx().input(|i| i.time);
    let mut screenshot = false;
    // Keyboard first, so this frame's labels already reflect this frame's
    // keys. One table binds the keys (FR-D7); the help overlay below renders
    // from the same table.
    //
    // Not while a D-058 retune widget has the keyboard, though: every
    // character typed into the frequency box is also a key, so `751M` would
    // otherwise cycle the colormap and toggle the max-hold mode on its way to
    // the radio — and a focused slider answers the arrow keys itself, which
    // are the FR-D7 reference-level bindings. A focused widget owns the
    // keyboard; everything else still reaches the map.
    let typing = tune_widget_has_focus(ui);
    if !typing {
        ui.ctx().input(|i| {
            for event in &i.events {
                // Read from the events rather than `key_pressed`, which
                // ignores modifiers entirely — D-062 put `Shift+\u{2190}` on the
                // same key as FR-D7's `\u{2190}`, so which of the two fired is a
                // question only the event's own modifiers can answer.
                let egui::Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } = event
                else {
                    continue;
                };
                if let Some(b) = keymap::binding_for(*key, *modifiers) {
                    if keymap::apply(b.action, view, &mut requests) {
                        screenshot = true;
                    }
                }
            }
        });
    }

    top_bar(ui, theme, view, stats);
    // The D-062 tune keys hand the bar a signed count of windows, which is
    // all the keymap can honestly produce: only the bar holds the device's
    // readback and reported range, and only those turn a count into a
    // frequency. Taken here so the count never reaches the composer's
    // request bag — the keys converge on the same `retune_hz` the box, the
    // slider and the buttons produce, before this response exists.
    let key_steps = std::mem::take(&mut requests.tune_steps);
    let fine_key_steps = std::mem::take(&mut requests.fine_tune_steps);
    let retune_hz = control_bar(
        ui,
        theme,
        view,
        tune,
        key_steps,
        fine_key_steps,
        &mut requests,
    );
    status_bar(ui, theme, view, stats);

    let mut grid = Rect::NOTHING;
    egui::CentralPanel::no_frame().show(ui, |ui| {
        let layout = Layout::compute(ui.max_rect());
        grid = layout.grid;
        // D-102 item 3: the declutter cap scales with the plot grid's own
        // width, only known once the layout above is computed — so the
        // membership update (which needs the *effective* cap) happens here,
        // not before the chrome bars have claimed their space.
        let effective_cap =
            crate::layout::scaled_declutter_cap(view.declutter_cap, layout.grid.width());
        overlay_state.update(annotations, effective_cap as usize, now);
        let visible_annotations = overlay_state.visible(now);
        let regions = layout.split(view.split);
        let params_for_hit_test = view.params;
        let annotation_ctx = stats
            .inspector_active
            .then_some((&params_for_hit_test, visible_annotations.as_slice()));
        surface_interactions(ui, &layout, view, annotation_ctx, overlay_state, now);
        // Registered after the data surface, so within its band the divider
        // is on top and the FR-D6 zoom drag cannot start there — and vice
        // versa, a zoom drag on the data surface never moves the split.
        divider_interactions(ui, &layout, &regions, view);
        let ppp = ui.pixels_per_point();
        axis_labels(ui.painter(), &layout, &regions, &view.params, theme, ppp);
        // The annotation overlay draws beside the axis labels, same Layout,
        // same frequency-to-pixel mapping (design §1) — and only when
        // analysis is active: with it off the display is exactly v1's
        // (design AC-11).
        if stats.inspector_active {
            overlay::draw(
                ui.painter(),
                &layout,
                &view.params,
                theme,
                ppp,
                view.annotation_level,
                view.label_style,
                &visible_annotations,
            );
            annotation_detail_panel(ui, theme, &visible_annotations, overlay_state, grid);
        }
        cursor_overlay(ui, &layout, &regions, &view.params, theme, ppp);
        selection_overlay(ui, &layout, view, theme);
        if view.help_open {
            help_overlay(ui, theme);
        }
    });
    ChromeResponse {
        grid,
        requests,
        screenshot,
        retune_hz,
    }
}

/// Everything the chrome needs to offer the **D-058 retune controls**, and
/// nothing it could use to invent one.
///
/// Every field is the device's own answer, passed straight through from the
/// pipeline: the centre it read back (C5), the range it reported, the rate it
/// is running at. `Default` is "no tunable source" — which is exactly what a
/// file, a pipe or the signal generator is, and what [`draw`] passes, so a
/// replay session draws precisely the chrome it always did.
#[derive(Debug, Clone, Default)]
pub struct TuneUi {
    /// Whether the active source advertises `ControlCaps::tune`. False draws
    /// no retune controls at all: an inert box in front of a file replay
    /// would be a control that cannot work.
    pub tunable: bool,
    /// The centre frequency the **device** reports, Hz — never the value a
    /// user asked for. `None` when the source declares none.
    pub center_hz: Option<f64>,
    /// The centre-frequency range the **device** reports, Hz. `None` means
    /// the device reported none: the slider is then not offered at all, since
    /// D-058 forbids bounding it by a guess. The box still works — the device
    /// remains the arbiter of what it will accept.
    pub range_hz: Option<(f64, f64)>,
    /// One sample rate, Hz — the slider's step **and** the width of the
    /// displayed window. Stepping the centre by exactly this moves the window
    /// by exactly its own width, so adjacent positions tile the spectrum with
    /// no gap and no overlap (D-058: 1× span is the point, do not "improve"
    /// it). `None` leaves the slider unstepped.
    pub step_hz: Option<f64>,
    /// The outcome of the last retune the app attempted, surfaced here
    /// because D-058 requires a rejected retune to be *said*, not merely
    /// declined.
    pub status: Option<TuneStatus>,
}

/// The outcome of the last retune, for the chrome to show.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuneStatus {
    /// The message to show.
    pub message: String,
    /// `false` renders it in the accent colour — D-010's rule that a failure
    /// is flagged, never smoothed away.
    pub ok: bool,
}

/// Highest frequency the box accepts, Hz. Not a device limit and not
/// pretending to be one — the device's own range bounds the slider, and the
/// device itself rejects what it cannot reach. This is only the point past
/// which a typed number is certainly a typo (1 THz is ~40× the top of any
/// SDR SoapySDR fronts).
const MAX_TYPED_HZ: f64 = 1e12;

/// Parse a frequency the way the product writes one: an optional `k`/`M`/`G`
/// multiplier and an optional `Hz` tail, with `p` accepted as the decimal
/// point.
///
/// This is the **same family D-055 fixed for capture filenames**
/// (`cap_1p985GHz_30p72Msps…`), so `1p985GHz` and `1.985G` and `1985000000`
/// are three spellings of one frequency rather than two conventions in one
/// product. Case is not significant: there are no millihertz in a spectrum
/// analyser, so `751m` can only mean megahertz.
///
/// Errors name the offending text (§8.4). A frequency the box rejects is
/// never sent to the device, so the display cannot move for one.
pub fn parse_frequency_hz(text: &str) -> Result<f64, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("enter a center frequency, e.g. 751M or 1.985G".to_owned());
    }
    let body = trimmed
        .strip_suffix("Hz")
        .or_else(|| trimmed.strip_suffix("HZ"))
        .or_else(|| trimmed.strip_suffix("hz"))
        .or_else(|| trimmed.strip_suffix("hZ"))
        .unwrap_or(trimmed)
        .trim_end();
    let (digits, scale) = match body.chars().last() {
        Some('k') | Some('K') => (&body[..body.len() - 1], 1e3),
        Some('M') | Some('m') => (&body[..body.len() - 1], 1e6),
        Some('G') | Some('g') => (&body[..body.len() - 1], 1e9),
        _ => (body, 1.0),
    };
    // D-055's filename spelling uses `p` where a number uses `.`; one
    // grammar, both spellings.
    let digits = digits.trim().replace(['p', 'P'], ".");
    if digits.is_empty() {
        return Err(format!(
            "\"{trimmed}\" has a unit but no number (try 751M or 1.985G)"
        ));
    }
    let value: f64 = digits.parse().map_err(|_| {
        format!("\"{trimmed}\" is not a frequency (try 751M, 1.985G, or 751000000)")
    })?;
    let hz = value * scale;
    if !hz.is_finite() || hz <= 0.0 {
        return Err(format!(
            "a center frequency must be positive, got \"{trimmed}\""
        ));
    }
    if hz > MAX_TYPED_HZ {
        return Err(format!(
            "\"{trimmed}\" is {} — that is far above any radio's range; \
             did you mean a unit suffix (751M, 1.985G)?",
            format_tune_hz(hz)
        ));
    }
    Ok(hz)
}

/// How the product writes a centre frequency back to the user: the same
/// `k`/`M`/`G` family [`parse_frequency_hz`] reads, so what the box shows can
/// be edited and re-entered as-is.
pub fn format_tune_hz(hz: f64) -> String {
    let a = hz.abs();
    let (value, unit) = if a >= 1e9 {
        (hz / 1e9, "G")
    } else if a >= 1e6 {
        (hz / 1e6, "M")
    } else if a >= 1e3 {
        (hz / 1e3, "k")
    } else {
        (hz, "")
    };
    // Six decimals in the scaled unit is 1 Hz at GHz — finer than any tuner
    // resolves, and trimmed so a round frequency reads round.
    let mut s = format!("{value:.6}");
    if s.contains('.') {
        s = s.trim_end_matches('0').trim_end_matches('.').to_owned();
    }
    format!("{s}{unit}")
}

/// egui id of the D-058 frequency box. Stable and named, so the box keeps its
/// focus and its half-typed text across frames — and so a seal can drive the
/// real widget rather than a copy of it (D-031).
pub fn tune_box_id() -> egui::Id {
    egui::Id::new("phosphene-tune-box")
}

/// egui id of the D-058 slider, for the same reasons.
pub fn tune_slider_id() -> egui::Id {
    egui::Id::new("phosphene-tune-slider")
}

/// Where the last drawn slider was, in egui points. Stashed by [`tune_bar`]
/// so a seal can send real pointer input to the real widget (D-031); `None`
/// before the first frame that drew one.
pub fn tune_slider_rect(ctx: &egui::Context) -> Option<Rect> {
    ctx.data(|d| d.get_temp::<Rect>(tune_slider_rect_id()))
}

fn tune_slider_rect_id() -> egui::Id {
    egui::Id::new("phosphene-tune-slider-rect")
}

/// Where the last drawn D-062 step button was, in egui points — `up` selects
/// the raise-the-centre button, otherwise the lower-the-centre one. Stashed
/// by [`tune_bar`] for the same reason the slider's rect is: a seal must be
/// able to click the **real** button with real pointer input rather than call
/// the step arithmetic behind it (D-031 — a test that calls the step function
/// proves arithmetic, not that a user can reach it). `None` before the first
/// frame that drew one.
pub fn tune_step_rect(ctx: &egui::Context, up: bool) -> Option<Rect> {
    ctx.data(|d| d.get_temp::<Rect>(tune_step_rect_id(up)))
}

fn tune_step_rect_id(up: bool) -> egui::Id {
    if up {
        egui::Id::new("phosphene-tune-step-up-rect")
    } else {
        egui::Id::new("phosphene-tune-step-down-rect")
    }
}

/// Where the last drawn **waterfall intensity** button was, in egui points —
/// `up` selects the more-intensity button. Stashed for the same reason the
/// tuner's rects are (D-031): a seal must click the real button with real
/// pointer input, not call the step arithmetic behind it. `None` before the
/// first frame that drew one.
pub fn wf_intensity_step_rect(ctx: &egui::Context, up: bool) -> Option<Rect> {
    ctx.data(|d| d.get_temp::<Rect>(wf_intensity_step_rect_id(up)))
}

fn wf_intensity_step_rect_id(up: bool) -> egui::Id {
    if up {
        egui::Id::new("phosphene-wf-intensity-up-rect")
    } else {
        egui::Id::new("phosphene-wf-intensity-down-rect")
    }
}

/// Turn a raw slider position into the frequency the radio is actually asked
/// for: **a whole number of sample rates away from where it is now**.
///
/// This is D-058's increment, and the ruling is explicit that it must not be
/// "improved": the displayed span *is* the sample rate at complex baseband,
/// so one step moves the window by exactly its own width and adjacent
/// positions **tile the spectrum — no gap, no overlap**. Any other step
/// either skips spectrum or re-shows it, and both are worse for walking a
/// band.
///
/// The tiling is anchored to the **current centre**, not to the bottom of the
/// device's range, which is why the snapping happens here rather than through
/// egui's own `step_by`: that rounds to absolute multiples of the step, so
/// unless the radio happened to sit on that grid the first move would be some
/// fraction of a window and every position after it would be off by the same
/// fraction.
///
/// The step *count* is clamped so the target stays inside the device's
/// reported range — the count rather than the frequency, so even an edge
/// position is a whole number of windows from the last one. Returns `None`
/// when the answer is "stay where you are", so no round trip to the radio is
/// made for a nudge that changes nothing.
pub fn window_step_target(
    center_hz: f64,
    slider_hz: f64,
    step_hz: Option<f64>,
    range_hz: (f64, f64),
) -> Option<f64> {
    let (min_hz, max_hz) = range_hz;
    let Some(step) = step_hz.filter(|s| s.is_finite() && *s > 0.0) else {
        // No declared rate means no window width to step by (C5: it is not
        // invented). The slider then asks for exactly where it was put.
        return (slider_hz != center_hz).then_some(slider_hz);
    };
    let steps = ((slider_hz - center_hz) / step).round();
    let up = ((max_hz - center_hz) / step).floor().max(0.0);
    let down = ((center_hz - min_hz) / step).floor().max(0.0);
    let steps = steps.clamp(-down, up);
    if steps == 0.0 {
        return None;
    }
    Some(center_hz + steps * step)
}

/// The frequency a whole number of **window steps** from where the radio is
/// — the arithmetic the D-062 step buttons and global keys share with the
/// D-058 slider, so all four entry points mean the same thing by
/// construction rather than by agreement.
///
/// One step is one sample rate, which is one window width: adjacent positions
/// tile the spectrum with no gap and no overlap. That is D-058's property and
/// D-062 restates it for the new controls — it is not a free choice.
///
/// `None` — no request, no round trip to the radio — when:
///
/// * `steps` is zero;
/// * the source declares no rate, so there is no window width to step by.
///   C5 forbids inventing one, and a button that guessed an increment would
///   move the radio somewhere nobody chose. The buttons are drawn disabled in
///   that state rather than silently doing nothing;
/// * the device reported a range and the step would leave it. The clamp is on
///   the step **count**, via [`window_step_target`], so even an edge position
///   is a whole number of windows from the last one.
///
/// A device that reported **no** range is not bounded here: D-058 forbids
/// inventing limits, and the device remains the arbiter of what it accepts —
/// exactly as it is for a frequency typed into the box.
pub fn step_target(
    center_hz: f64,
    steps: i32,
    step_hz: Option<f64>,
    range_hz: Option<(f64, f64)>,
) -> Option<f64> {
    let step = step_hz.filter(|s| s.is_finite() && *s > 0.0)?;
    if steps == 0 {
        return None;
    }
    let raw = center_hz + f64::from(steps) * step;
    match range_hz {
        Some(range) => window_step_target(center_hz, raw, Some(step), range),
        None => Some(raw),
    }
}

/// What one **fine** tune step divides the window by (D-066 §2): the shifted
/// arrows move the centre by `sample_rate / 10`.
///
/// **This step deliberately does not tile, and that is the point.** D-058's
/// 1×span increment exists so adjacent *coarse* positions abut with no gap
/// and no overlap — the property that makes walking a band exhaustive. A
/// tenth of a window is for centring a signal *inside* the window that is
/// already on screen, which is a different job. D-066 §2 says in as many
/// words: do not "fix" the fine step to preserve tiling.
pub const FINE_TUNE_DIVISOR: f64 = 10.0;

/// The frequency the tune keys of D-062 and **D-066 §2** ask for together:
/// `steps` whole windows plus `fine_steps` tenths of one, from where the
/// radio actually is.
///
/// One function rather than two entry points, because both counts can arrive
/// in the same frame (two keys, one 16 ms frame) and D-058 §3 requires them
/// to converge on **one** request before the response exists — two requests
/// would be two round trips to the radio for one gesture, and the second
/// would be computed from a readback the first had already invalidated.
///
/// The two are clamped differently at the edge of a reported range, and that
/// asymmetry is deliberate:
///
/// * the coarse part goes through [`step_target`], which clamps the step
///   **count**, so even an edge position is a whole number of windows from
///   the last one — D-058's tiling survives the edge;
/// * the fine part never tiled, so it is simply clamped into the range.
///
/// `None` — no request, no round trip — when nothing was asked for, when the
/// source declares no rate (C5: a window width is never invented), or when
/// the answer is "stay where you are".
pub fn tune_target(
    center_hz: f64,
    steps: i32,
    fine_steps: i32,
    step_hz: Option<f64>,
    range_hz: Option<(f64, f64)>,
) -> Option<f64> {
    let coarse = step_target(center_hz, steps, step_hz, range_hz);
    if fine_steps == 0 {
        return coarse;
    }
    let step = step_hz.filter(|s| s.is_finite() && *s > 0.0)?;
    let raw = coarse.unwrap_or(center_hz) + f64::from(fine_steps) * step / FINE_TUNE_DIVISOR;
    let target = match range_hz {
        Some((min_hz, max_hz)) if max_hz > min_hz => raw.clamp(min_hz, max_hz),
        _ => raw,
    };
    (target != center_hz).then_some(target)
}

/// The tuner's type sizes. D-062 asked for it to be **larger** — it is the
/// primary hardware control and read as an afterthought — and D-010 governs
/// how: legibility over ornament, and the data still does the talking. So the
/// growth is spent on the two things a hand actually uses, the frequency
/// itself and the step targets, and not on padding.
///
/// `tune_bar_cost_to_the_spectrum` pins what this costs the display.
const TUNE_LABEL_PT: f32 = 11.0;
const TUNE_FREQ_PT: f32 = 17.0;
const TUNE_BOX_WIDTH_PT: f32 = 150.0;
const TUNE_BUTTON_PT: f32 = 15.0;
/// The control bar's non-tuner value type size — the waterfall intensity
/// exponent. Smaller than the frequency deliberately: D-062 spent the
/// tuner's growth on the radio's own number, and D-010 says the data does the
/// talking, so a second value on the same bar must not compete with it.
const CONTROL_VALUE_PT: f32 = 13.0;
/// Step-button hit target, points. Comfortably past the ~24 pt floor a
/// pointer wants, without becoming the loudest thing on screen.
const TUNE_BUTTON_SIZE: egui::Vec2 = egui::vec2(32.0, 28.0);
/// The slider is the band-walking control; at the old default width one
/// window was a sub-pixel nudge on a multi-GHz range.
const TUNE_SLIDER_WIDTH_PT: f32 = 280.0;

/// Dress the control bar's interactive widgets in the D-010 palette, scoped
/// to this bar rather than installed globally — `Theme::install` styles the chrome as
/// a whole, and the wgpu data surface reads the same `Theme`, so a global
/// change here would reach further than the control it is meant to serve.
///
/// egui's stock dark visuals put mid-grey fills on buttons and the slider
/// rail. Against this near-black ground they read as a stock form, which is
/// exactly the failure D-010 names: a screenshot that looks like a default
/// egui form has failed it. So the fills come down to the panel's own darks,
/// the borders become the single hairline the rest of the chrome uses, and
/// the accent is spent only where a hand is — hover and press.
fn control_visuals(ui: &mut Ui, theme: &Theme) {
    let visuals = ui.visuals_mut();
    visuals.selection.bg_fill = theme.panel_edge;
    visuals.selection.stroke = egui::Stroke::new(1.0, theme.accent);
    let hairline = egui::Stroke::new(1.0, theme.panel_edge);
    let w = &mut visuals.widgets;

    // The slider rail and any inert frame.
    w.noninteractive.bg_fill = theme.bg;
    w.noninteractive.weak_bg_fill = theme.bg;
    w.noninteractive.bg_stroke = hairline;

    // At rest: the control is present, not loud.
    w.inactive.bg_fill = theme.panel_edge;
    w.inactive.weak_bg_fill = theme.bg;
    w.inactive.bg_stroke = hairline;
    w.inactive.fg_stroke = egui::Stroke::new(1.0, theme.text);

    // Under the pointer, and pressed: the accent, and only here.
    for state in [&mut w.hovered, &mut w.active] {
        state.bg_fill = theme.panel_edge;
        state.weak_bg_fill = theme.panel_edge;
        state.bg_stroke = egui::Stroke::new(1.0, theme.accent);
        state.fg_stroke = egui::Stroke::new(1.0, theme.accent);
    }

    // A control the source cannot honour is dimmed, never hidden.
    w.open.bg_fill = theme.bg;
    w.open.weak_bg_fill = theme.bg;
    w.open.bg_stroke = hairline;
}

/// One step button: a glyph, a real hit target, and a tooltip that names the
/// keyboard equivalent **from the keymap table** — so a button and its key
/// cannot come to disagree, and neither can drift from the `?` overlay.
///
/// Drawn disabled rather than omitted when the source declares no rate: the
/// control exists, and the honest thing is to show it inert (with the reason
/// on hover) rather than to make the chrome silently different.
fn step_button(
    ui: &mut Ui,
    theme: &Theme,
    glyph: &str,
    action: keymap::Action,
    what: &str,
    enabled: bool,
) -> egui::Response {
    let button = egui::Button::new(
        egui::RichText::new(glyph)
            .font(FontId::monospace(TUNE_BUTTON_PT))
            .color(theme.text),
    )
    .min_size(TUNE_BUTTON_SIZE);
    let hint = match keymap::binding_for_action(action) {
        Some(b) => format!("{what} ({})", keymap::binding_label(b)),
        None => what.to_owned(),
    };
    let hint = if enabled {
        hint
    } else {
        format!("{hint} - unavailable: the source declares no sample rate, so there is no window width to step by")
    };
    ui.add_enabled(enabled, button).on_hover_text(hint)
}

/// The **control bar**: the D-065 §3 waterfall intensity control, plus the
/// **D-058 retune controls** — a frequency box and a window-stepping slider
/// — for a source that can actually tune.
///
/// The bar is drawn for every source. The tuner half is not: an inert
/// frequency box in front of a file replay would be a control that cannot
/// work, while the intensity control applies to every waterfall there is.
fn control_bar(
    ui: &mut Ui,
    theme: &Theme,
    view: &mut ViewState,
    tune: &TuneUi,
    key_steps: i32,
    fine_key_steps: i32,
    requests: &mut ChromeRequests,
) -> Option<f64> {
    let mut requested = None;
    egui::Panel::top("phosphene-controls")
        .frame(
            Frame::new()
                .fill(theme.panel_bg)
                .inner_margin(Margin::symmetric(12, 6)),
        )
        .show_separator_line(false)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().slider_width = TUNE_SLIDER_WIDTH_PT;
                control_visuals(ui, theme);
                if tune.tunable {
                    requested = tune_group(ui, theme, tune, key_steps, fine_key_steps);
                    ui.add_space(18.0);
                }
                // D-065 §3: the §7.6 waterfall intensity control. Built from
                // the same helpers the tuner is, so the bar gained a control
                // and not a second dialect.
                wf_intensity_group(ui, theme, view, requests);
            });
        });
    requested
}

/// The **D-065 §3 waterfall intensity control**: two step buttons flanking
/// the exponent they move, in the tuner's own visual language.
///
/// Four properties, each of them a ruling rather than a preference:
///
/// * **It is the same idiom as the tuner**, down to the helper — same button
///   size, same hover accent, same tooltip built from the FR-D7 table — so
///   the chrome gained a control and not a second dialect.
/// * **The buttons and the keys are one implementation.** A click calls the
///   same [`keymap::apply`] the `Shift+PgUp`/`Shift+PgDn` events call, so
///   there is no second arithmetic to drift from the first — D-062's
///   convergence property, restated for a two-entry-point control.
/// * **The value is beside the control**, and it is the same
///   [`ViewState::wf_gamma`] the status bar's `wf` field states and the frame
///   loop mirrors onto the composer. One field, three readers; a control
///   whose effect you cannot read is indistinguishable from one that does not
///   work (D-043).
/// * **It stays meaningful in both §7.6 range modes**, including D-065 §2's
///   new auto-range default: `γ_wf` shapes the *normalised* power, so the
///   range mode chooses which dB window the colours span and this chooses how
///   they are distributed inside it. Neither takes the other's job.
fn wf_intensity_group(
    ui: &mut Ui,
    theme: &Theme,
    view: &mut ViewState,
    requests: &mut ChromeRequests,
) {
    ui.label(chrome_text("wf intensity", TUNE_LABEL_PT, theme.text_dim));

    // The buttons are placed first and acted on afterwards, so the value
    // between them is this frame's — a click applies on the next frame, the
    // same way a keypress does.
    fn step(ui: &mut Ui, theme: &Theme, up: bool) -> bool {
        let (glyph, action, what) = if up {
            (
                "\u{25b2}",
                keymap::Action::WaterfallIntensityUp,
                "waterfall intensity up",
            )
        } else {
            (
                "\u{25bc}",
                keymap::Action::WaterfallIntensityDown,
                "waterfall intensity down",
            )
        };
        let button = step_button(ui, theme, glyph, action, what, true);
        ui.ctx()
            .data_mut(|d| d.insert_temp(wf_intensity_step_rect_id(up), button.rect));
        button.clicked()
    }

    let down = step(ui, theme, false);
    ui.label(chrome_text(
        &format!("g{}", num_text(view.wf_gamma)),
        CONTROL_VALUE_PT,
        theme.text,
    ));
    let up = step(ui, theme, true);
    // Buttons and keys, one implementation: a click runs the same
    // `keymap::apply` the shifted PgUp/PgDn events run.
    if down {
        keymap::apply(keymap::Action::WaterfallIntensityDown, view, requests);
    }
    if up {
        keymap::apply(keymap::Action::WaterfallIntensityUp, view, requests);
    }
}

/// The **D-058 retune group**: a frequency box and a window-stepping slider,
/// drawn only for a source that can actually tune.
///
/// Three things this function will not do, each of them a ruling:
///
/// * **It does not retune.** It returns the frequency the user asked for; the
///   app applies it through the one `set(Control::CenterFreqHz)` path both
///   widgets share (D-058: two entry points, one mechanism). §8.1's crate
///   boundary makes that structural rather than a promise — the render crate
///   does not depend on `phosphene-sources` and cannot name `Control`, which
///   is also why D-011's ban on a zoom-driven retune survives this lane
///   untouched.
/// * **It does not show where the user dragged to.** Both widgets are fed
///   `tune.center_hz`, the device's readback, on every frame. A retune the
///   device refuses or clamps therefore leaves them sitting at the radio's
///   real frequency the very next frame, with the error beside them — a
///   slider parked where the user let go while the radio is elsewhere is the
///   same lie as an invented sample rate, with a handle on it.
/// * **It does not invent bounds.** No device range, no slider. The box stays,
///   because the device is still free to accept or refuse what is typed.
fn tune_group(
    ui: &mut Ui,
    theme: &Theme,
    tune: &TuneUi,
    key_steps: i32,
    fine_key_steps: i32,
) -> Option<f64> {
    let mut requested = None;
    ui.label(chrome_text("tune", TUNE_LABEL_PT, theme.text_dim));

    // A step needs a window width to step by, and only the
    // device's declared rate provides one (C5: never invented).
    let steppable =
        tune.center_hz.is_some() && tune.step_hz.is_some_and(|s| s.is_finite() && s > 0.0);
    // Whole windows asked for this frame: the global keys, plus
    // the buttons flanking the box below. One count, one target,
    // one request — the same one the slider and the box produce.
    let mut steps = key_steps;

    // Down: toward the low end of the band, the direction the
    // slider's left is and the direction the bare `\u{2190}` means.
    let down = step_button(
        ui,
        theme,
        "\u{25c0}",
        keymap::Action::TuneDown,
        "tune down one window",
        steppable,
    );
    ui.ctx()
        .data_mut(|d| d.insert_temp(tune_step_rect_id(false), down.rect));
    if down.clicked() {
        steps -= 1;
    }

    // The box. Unfocused, it mirrors the readback; focused, it is
    // the user's own text until they commit with Enter.
    let id = tune_box_id();
    let focused = ui.memory(|m| m.has_focus(id));
    let mut text = ui.ctx().data_mut(|d| {
        d.get_temp::<String>(id)
            .unwrap_or_else(|| readback_text(tune))
    });
    if !focused {
        text = readback_text(tune);
    }
    let box_response = ui.add(
        egui::TextEdit::singleline(&mut text)
            .id(id)
            .font(FontId::monospace(TUNE_FREQ_PT))
            .margin(Margin::symmetric(8, 5))
            // Centred so the two arrows flank the frequency
            // itself (D-062) rather than the far edges of a box
            // the number sits at one end of.
            .horizontal_align(egui::Align::Center)
            .desired_width(TUNE_BOX_WIDTH_PT)
            .hint_text("751M"),
    );
    let committed =
        box_response.lost_focus() && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter));
    if committed {
        match parse_frequency_hz(&text) {
            Ok(hz) => requested = Some(hz),
            // A parse failure never reaches the device: the box
            // says what is wrong and the radio does not move.
            Err(message) => {
                ui.ctx().data_mut(|d| {
                    d.insert_temp(tune_status_id(), TuneStatus { message, ok: false });
                });
            }
        }
    }
    ui.ctx().data_mut(|d| d.insert_temp(id, text));

    let up = step_button(
        ui,
        theme,
        "\u{25b6}",
        keymap::Action::TuneUp,
        "tune up one window",
        steppable,
    );
    ui.ctx()
        .data_mut(|d| d.insert_temp(tune_step_rect_id(true), up.rect));
    if up.clicked() {
        steps += 1;
    }

    // Buttons and keys — coarse windows and D-066 §2's fine tenths
    // of one — resolved together against the device's own readback
    // and range, so two keys in one frame are one request. A typed
    // frequency committed in the same frame wins: it is the more
    // specific request of the two.
    if requested.is_none() {
        if let Some(center) = tune.center_hz {
            requested = tune_target(center, steps, fine_key_steps, tune.step_hz, tune.range_hz);
        }
    }

    // The slider: one increment is one sample rate, which is one
    // window width, so adjacent positions tile the spectrum
    // (D-058 — 1× span is the point).
    match (tune.center_hz, tune.range_hz) {
        (Some(center), Some((min_hz, max_hz))) if max_hz > min_hz => {
            let mut value = center.clamp(min_hz, max_hz);
            // Deliberately NOT `step_by`: egui rounds to absolute
            // multiples of the step, which only tiles when the
            // radio already sits on that grid. `window_step_target`
            // anchors the tiling to where the radio actually is.
            let slider = egui::Slider::new(&mut value, min_hz..=max_hz)
                .show_value(false)
                .custom_formatter(|v, _| format_tune_hz(v));
            let response = ui.push_id(tune_slider_id(), |ui| ui.add(slider)).inner;
            // The slider's own rect, remembered under a named id.
            // egui derives a widget's id from its parent and its
            // order, so this is the only stable way for a seal to
            // drive the REAL slider — D-031's production path —
            // with real pointer input rather than a copy of it.
            ui.ctx().data_mut(|d| {
                d.insert_temp(tune_slider_rect_id(), response.rect);
                d.insert_temp(tune_slider_widget_id(), response.id);
            });
            // Committed on release (and on a click or a keyboard
            // nudge, which are not drags): a retune is a blocking
            // round trip to the radio, so one is issued per
            // gesture rather than per frame of a drag.
            let settled = response.drag_stopped() || (response.changed() && !response.dragged());
            // Only ever *adds* a request: `window_step_target`
            // returns `None` for "stay where you are", which must
            // not cancel a step a button or key already asked for
            // in the same frame.
            if settled {
                if let Some(target) =
                    window_step_target(center, value, tune.step_hz, (min_hz, max_hz))
                {
                    requested = Some(target);
                }
            }
        }
        _ => {
            // D-058's stop-and-surface, answered honestly rather
            // than papered over: no reported range, no slider.
            ui.label(chrome_text("no device range", 10.0, theme.text_dim));
        }
    }

    if let Some((min_hz, max_hz)) = tune.range_hz {
        ui.label(chrome_text(
            &format!("{}-{}", format_tune_hz(min_hz), format_tune_hz(max_hz)),
            10.0,
            theme.text_dim,
        ));
    }

    // A fresh request supersedes any parse error still on show:
    // whatever the device answers is the current truth.
    if requested.is_some() {
        ui.ctx()
            .data_mut(|d| d.remove::<TuneStatus>(tune_status_id()));
    }

    // The outcome of the last retune, from either widget. A
    // refusal is stated, never a silent no-op (D-056/D-058).
    let status = ui
        .ctx()
        .data_mut(|d| d.get_temp::<TuneStatus>(tune_status_id()))
        .or_else(|| tune.status.clone());
    if let Some(status) = status {
        let color = if status.ok { theme.text } else { theme.accent };
        ui.add_space(8.0);
        ui.label(chrome_text(&status.message, 10.0, color));
    }
    requested
}

/// egui id of the chrome-owned half of the retune status line: a parse error
/// never leaves this crate (it has no device in it), so it is remembered here
/// rather than round-tripped through the app.
/// Whether a D-058 retune widget currently owns the keyboard.
///
/// Scoped to those two widgets on purpose rather than "anything focused": the
/// FR-D7 map is the primary interface and must keep working, so only the
/// widgets that genuinely consume keys take them.
fn tune_widget_has_focus(ui: &Ui) -> bool {
    let slider = ui
        .ctx()
        .data(|d| d.get_temp::<egui::Id>(tune_slider_widget_id()));
    ui.memory(|m| m.has_focus(tune_box_id()) || slider.is_some_and(|id| m.has_focus(id)))
}

/// egui's own id for the slider widget, as it assigned it — stashed by
/// [`tune_bar`] so the focus guard above can name a widget whose id egui
/// derives rather than us.
fn tune_slider_widget_id() -> egui::Id {
    egui::Id::new("phosphene-tune-slider-widget")
}

fn tune_status_id() -> egui::Id {
    egui::Id::new("phosphene-tune-status")
}

/// What the box shows when it is not being typed into: the device's own
/// centre, or an explicit dash when the source declares none. Never a
/// remembered request.
fn readback_text(tune: &TuneUi) -> String {
    match tune.center_hz {
        Some(hz) => format_tune_hz(hz),
        None => String::new(),
    }
}

fn top_bar(ui: &mut Ui, theme: &Theme, view: &ViewState, stats: &HudStats) {
    egui::Panel::top("phosphene-top")
        .frame(
            Frame::new()
                .fill(theme.panel_bg)
                .inner_margin(Margin::symmetric(12, 7)),
        )
        .show_separator_line(false)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(chrome_text("phosphene", 15.0, theme.accent));
                ui.add_space(10.0);
                ui.label(chrome_text("rtsa", 11.0, theme.text_dim));
                if view.paused {
                    ui.add_space(10.0);
                    // A frozen display must say so (FR-D12; D-010 binds the
                    // look to honesty).
                    ui.label(chrome_text("paused", 11.0, theme.accent));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(chrome_text(&stats.source_label, 11.0, theme.text));
                    ui.label(chrome_text("src", 11.0, theme.text_dim));
                });
            });
        });
}

fn status_bar(ui: &mut Ui, theme: &Theme, view: &ViewState, stats: &HudStats) {
    let params = &view.params;
    egui::Panel::bottom("phosphene-status")
        .frame(
            Frame::new()
                .fill(theme.panel_bg)
                .inner_margin(Margin::symmetric(12, 5)),
        )
        .show_separator_line(false)
        .show(ui, |ui| {
            // Wrapped, not a plain horizontal: D-010 already called this bar
            // dense, and M2-B adds five more fields (D-043). A plain
            // horizontal lets widgets run past the panel's right edge, where
            // egui culls them from painting — an honesty-HUD field silently
            // invisible at a narrower window is worse than a two-line bar,
            // so overflow wraps onto a second row instead of disappearing.
            ui.horizontal_wrapped(|ui| {
                fn item(ui: &mut Ui, theme: &Theme, label: &str, value: &str) {
                    item_colored(ui, theme, label, value, theme.text);
                }
                fn item_colored(
                    ui: &mut Ui,
                    theme: &Theme,
                    label: &str,
                    value: &str,
                    color: egui::Color32,
                ) {
                    ui.label(chrome_text(label, 10.0, theme.text_dim));
                    ui.label(chrome_text(value, 10.0, color));
                    ui.add_space(10.0);
                }
                item(ui, theme, "fps", &format!("{:5.1}", stats.fps));
                item(ui, theme, "frame", &format!("{:5.2}ms", stats.frame_ms));
                item(ui, theme, "fft", &format!("{}", stats.fft_size));
                item(ui, theme, "win", stats.window_name);
                // FR-D8/D-009: which colormap the C key has cycled to — the
                // state readout for a control whose effect is otherwise the
                // only evidence (M1-J).
                item(ui, theme, "map", stats.colormap);
                // D-043: FR-D1 names the persistence histogram's two user
                // controls together — "persistence time (decay τ),
                // intensity/gamma" (`,`/`.` and PgUp/PgDn) — and neither had
                // a readout before this lane (D-043's report). One compact
                // field, since they are one spec sentence and one row on the
                // status bar was already full before this lane (D-010).
                item(
                    ui,
                    theme,
                    "pers",
                    &format!(
                        "{} g{}",
                        tau_text(stats.persistence_tau_s),
                        num_text(stats.gamma)
                    ),
                );
                // FR-D3: the max-hold state as the keys set it — off, or the
                // active D-007 mode — plus its own decay τ (`;`/`'`), which
                // D-043 found had no readout at all: only the mode showed.
                item(ui, theme, "max", &max_hold_text(view, stats.max_hold_tau_s));
                // FR-D2 (D-043): the live-trace averaging τ the `K`/`L` keys
                // move.
                item(ui, theme, "ltau", &tau_text(stats.live_tau_s));
                // FR-D4/§7.6: the waterfall controls as the keys set them —
                // time span, row aggregation, and the auto-range marker —
                // read from the same ViewState the keys mutate.
                item(ui, theme, "wf", &waterfall_text(view));
                // The span is the VISIBLE (zoomed) span — the status bar and
                // the axis must tell the same story. C5/D-028: with no
                // declared rate the span is normalized — no Hz unit is ever
                // shown or implied.
                item(ui, theme, "span", &span_text(params));
                if !params.zoom.is_full() {
                    let factor = 1.0 / params.zoom.width();
                    let f = format!("{factor:.1}");
                    let f = f.trim_end_matches('0').trim_end_matches('.');
                    item(ui, theme, "zoom", &format!("x{f}"));
                }
                // FR-D11, the honesty readouts. D-010 binds them: overload is
                // flagged in the accent color, never smoothed away.
                let overloaded = stats.dropped_batches > 0 || stats.processed_pct < 100.0;
                let hot = if overloaded { theme.accent } else { theme.text };
                item(
                    ui,
                    theme,
                    "in",
                    &format!("{}/s", format_count(stats.input_rate_sps)),
                );
                item_colored(
                    ui,
                    theme,
                    "proc",
                    &format!("{:5.1}%", stats.processed_pct),
                    hot,
                );
                item(ui, theme, "fft/s", &format_count(stats.ffts_per_s));
                item_colored(
                    ui,
                    theme,
                    "drops",
                    &format!("{}", stats.dropped_batches),
                    hot,
                );
                // D-050: the device's own overflow-event count — home given
                // by this lane. Reported as events, explicitly labelled so
                // it cannot be mistaken for a sample or batch count; it
                // never feeds `drops`, which counts only batches the
                // pipeline itself shed.
                let overflow_hot = if stats.overflow_events > 0 {
                    theme.accent
                } else {
                    theme.text
                };
                item_colored(
                    ui,
                    theme,
                    "ovfl",
                    &format!("{} evt", stats.overflow_events),
                    overflow_hot,
                );
                // Plain left-flowing items, like every other field: a
                // right-to-left trailing group stays pinned to the panel's
                // right edge regardless of how far the wrapped row above it
                // grows, which the M2-B fields made wide enough to overlap
                // (D-043 added five fields to an already-dense bar, D-010).
                item_colored(ui, theme, "backend", stats.backend, theme.accent);
                // D-006: the scale is dBFS per bin and the UI must say so.
                item(ui, theme, "scale", "dbfs/bin");
                // Annotation design situations 1/3/10: the inspector-active
                // chip with its live track count, and the NFR-A5 shedding
                // chip — visible exactly when its condition holds, silent
                // otherwise (D-010: never hide overload), matching the
                // existing drop/overflow readouts' honesty-HUD idiom.
                //
                // D-125 Amendment 4 item 3: the chip's own visibility is the
                // *cumulative* coverage count (frames_analysed < frames_
                // received), not just this cycle's `shedding` flag — a
                // cycle can read `shedding: false` while a real backlog
                // from an earlier overload is still outstanding, and D-010
                // says overload is never hidden. The count itself is the
                // "frames analysed out of frames received" the ruling asks
                // for, never just a bare "shedding" word.
                if stats.inspector_active {
                    item(
                        ui,
                        theme,
                        "inspector",
                        &format!("{} tracks", stats.track_count),
                    );
                    if stats.frames_analysed < stats.frames_received {
                        item_colored(
                            ui,
                            theme,
                            "analysis",
                            &format!(
                                "shedding {}/{}",
                                stats.frames_analysed, stats.frames_received
                            ),
                            theme.accent,
                        );
                    }
                }
            });
        });
}

/// The FR-D3 status readout: the max-hold trace's state — hidden, or its
/// active D-007 mode — read from the same [`ViewState`] the M/D keys mutate,
/// plus the D-007 decay τ (D-043) the `;`/`'` keys move. The τ is shown in
/// every mode, including `off`/`hold` where it currently has no visible
/// effect: the readout's job is to confirm the keypress changed the value
/// the accumulator holds, not to editorialize about whether that value
/// matters right now.
fn max_hold_text(view: &ViewState, tau_s: f32) -> String {
    let mode = if !view.max_hold {
        "off"
    } else if view.max_hold_decay {
        "decay"
    } else {
        "hold"
    };
    format!("{mode} {}", tau_text(tau_s))
}

/// Compact seconds: `1.00` renders `1s`, `0.71` renders `0.71s` — the same
/// trim-trailing-zeros idiom [`span_text`] and [`waterfall_text`] already
/// use, kept short so the four new D-043 readouts do not blow out an
/// already-dense bar (D-010).
fn tau_text(seconds: f32) -> String {
    format!("{}s", num_text(seconds))
}

/// Compact 2-decimal number with trailing zeros trimmed (`0.70` → `0.7`,
/// `1.00` → `1`).
fn num_text(value: f32) -> String {
    let s = format!("{value:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_owned()
}

/// The FR-D4 waterfall status readout (M1-J, mode-aware since M2-C): the
/// quantity the active §7.6 speed mode owns — **named**, so a bare number
/// can never silently mean a span in one mode and a row interval in the
/// other — plus the row aggregation and the auto-range marker, all read
/// from the same [`ViewState`] the F, `-`/`=`, A and W keys mutate, so the
/// readout cannot disagree with them.
///
/// Traditional shows `span 30s` (or `span 1024r`, a row count, for a
/// rate-less source — D-054: no seconds figure exists). Fast shows the row
/// interval as `1ms/row` (`spec/row` — one spectrum per row, pinned — for a
/// rate-less source). Like the max-hold τ, this is the **setting** the keys
/// move; the FR-D5 axis labels state the interval the rows actually carried,
/// clamp included (D-031).
///
/// The trailing `g` is D-065 §3's intensity exponent — the §7.6 mapping's
/// own gamma, distinct from the `pers` field's §7.3 one, and stated for the
/// same D-043 reason: an unreadable control is an untrustworthy one.
fn waterfall_text(view: &ViewState) -> String {
    use crate::waterfall::WaterfallMode;

    let owned = match (view.wf_mode, view.params.sample_rate.is_some()) {
        (WaterfallMode::Traditional, true) => format!("span {}", tau_text(view.wf_time_span_s)),
        (WaterfallMode::Traditional, false) => format!("span {}r", num_text(view.wf_span_rows)),
        (WaterfallMode::Fast, true) => format!("{}/row", interval_text(view.wf_fast_interval_s)),
        (WaterfallMode::Fast, false) => "spec/row".to_owned(),
    };
    let mode = match view.wf_aggregation {
        crate::waterfall::RowAggregation::Max => "max",
        crate::waterfall::RowAggregation::Mean => "mean",
    };
    let auto = if view.wf_auto_range { " auto" } else { "" };
    // D-065 §3: the §7.6 intensity exponent, stated the way M2-B stated
    // §7.3's on this bar — the same `g` field, read from the same
    // `ViewState` the keys and the control-bar buttons move, so the number
    // here and the number beside the buttons are one value.
    format!("{owned} {mode}{auto} g{}", num_text(view.wf_gamma))
}

/// Compact interval: sub-second values render in milliseconds (`0.001` →
/// `1ms`, `0.0008` → `0.8ms`), one second and up in seconds — the fast
/// mode's 1 ms default must not collapse to `0s` under [`num_text`]'s two
/// decimals.
fn interval_text(seconds: f32) -> String {
    if seconds < 1.0 {
        format!("{}ms", num_text(seconds * 1e3))
    } else {
        tau_text(seconds)
    }
}

/// The status-bar span text: the visible (zoomed) span in the axis's honest
/// unit.
fn span_text(params: &DisplayParams) -> String {
    match params.sample_rate {
        Some(rate) => format_hz(rate * params.zoom.width())
            .trim_start_matches('+')
            .to_owned(),
        None if params.zoom.is_full() => "±0.5 norm".to_owned(),
        None => {
            let w = format!("{:.4}", params.zoom.width());
            let w = w.trim_end_matches('0').trim_end_matches('.');
            format!("{w} norm")
        }
    }
}

/// Compact non-negative count/rate: `0`, `953`, `19.5K`, `2.05M`, `1.2G`.
/// Unlike [`format_hz`] there is no sign — these are magnitudes.
fn format_count(value: f64) -> String {
    let v = value.max(0.0);
    let (scaled, unit) = if v >= 1e9 {
        (v / 1e9, "G")
    } else if v >= 1e6 {
        (v / 1e6, "M")
    } else if v >= 1e3 {
        (v / 1e3, "K")
    } else {
        return format!("{v:.0}");
    };
    let s = format!("{scaled:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    format!("{s}{unit}")
}

/// Mouse gestures on the data surface (FR-D6): drag-select zoom, scroll
/// zoom about the cursor, double-click reset. All display-side (D-011) —
/// they only ever touch `view.params.zoom`.
/// `annotations`, when `Some`, is `(params, currently visible tracks)` — used
/// to hit-test the same click/hover this function already discriminates from
/// a drag (design §2.6: "reuses `chrome.rs`'s own click/drag discrimination
/// and cannot fight the drag-zoom ... gestures already on the same surface").
fn surface_interactions(
    ui: &mut Ui,
    layout: &Layout,
    view: &mut ViewState,
    annotations: Option<(&DisplayParams, &[VisibleTrack])>,
    overlay_state: &mut AnnotationOverlayState,
    now: f64,
) {
    let grid = layout.grid;
    if grid.width() <= 0.0 {
        return;
    }
    let resp = ui.interact(grid, ui.id().with("data-surface"), Sense::click_and_drag());
    let x_to_view = |x: f32| f64::from((x - grid.min.x) / grid.width());

    if resp.double_clicked() {
        view.params.zoom = ZoomSpan::FULL;
        view.drag_from = None;
        return;
    }
    if resp.drag_started() {
        view.drag_from = resp.interact_pointer_pos().map(|p| p.x);
    }
    if resp.drag_stopped() {
        if let (Some(x0), Some(p)) = (view.drag_from, resp.interact_pointer_pos()) {
            if (p.x - x0).abs() >= MIN_DRAG_PT {
                view.params.zoom = view.params.zoom.zoom_to(x_to_view(x0), x_to_view(p.x));
            } else if let Some((params, visible)) = annotations {
                // A non-dragged click (§2.6): open/pin the detail panel.
                if let Some(id) = overlay::hit_test(visible, params, grid, p.x) {
                    overlay_state.detail_mut().click(id);
                }
            }
        }
        view.drag_from = None;
    }
    if resp.hovered() {
        let scroll = ui.ctx().input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            if let Some(p) = resp.hover_pos() {
                let factor = SCROLL_ZOOM_BASE.powf(f64::from(scroll) / SCROLL_NOTCH_PT);
                view.params.zoom = view.params.zoom.zoom_about(x_to_view(p.x), factor);
            }
        }
    }
    match annotations {
        Some((params, visible)) => {
            let hovered = resp
                .hover_pos()
                .and_then(|p| overlay::hit_test(visible, params, grid, p.x));
            overlay_state.detail_mut().hover(hovered, now);
        }
        None => overlay_state.detail_mut().hover(None, now),
    }
}

/// Half-height, in points, of the grab band centered on the
/// histogram/waterfall divider (FR-D4: the split really is draggable —
/// M1-J closes AUDIT-1's mislabelled row).
const DIVIDER_GRAB_PT: f32 = 5.0;

/// The FR-D4 divider drag (M1-J): a grab band centered between the two data
/// regions; dragging it moves `view.split` continuously through the full
/// closed 0..=1 range via [`Layout::split_ratio_at_y`] — the exact inverse
/// of the [`Layout::split`] that placed the divider. The `[`/`]` keys keep
/// working alongside it.
fn divider_interactions(
    ui: &mut Ui,
    layout: &Layout,
    regions: &SplitRegions,
    view: &mut ViewState,
) {
    let grid = layout.grid;
    if grid.width() <= 0.0 || grid.height() <= 0.0 {
        return;
    }
    let mid = (regions.histogram.max.y + regions.waterfall.min.y) * 0.5;
    let band = Rect::from_min_max(
        Pos2::new(grid.min.x, mid - DIVIDER_GRAB_PT),
        Pos2::new(grid.max.x, mid + DIVIDER_GRAB_PT),
    );
    let resp = ui.interact(band, ui.id().with("split-divider"), Sense::drag());
    if resp.hovered() || resp.dragged() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
    }
    if resp.dragged() {
        if let Some(pos) = resp.interact_pointer_pos() {
            view.split = layout.split_ratio_at_y(pos.y);
        }
    }
}

/// Round an egui-points coordinate onto the physical-pixel grid — the same
/// rounding the wgpu grid lines use, so labels and grid stay aligned at
/// fractional HiDPI scale factors, and text anchors land on whole pixels
/// (FR-D5 legibility).
pub(crate) fn snap(v: f32, ppp: f32) -> f32 {
    (v * ppp).round() / ppp
}

fn axis_labels(
    painter: &egui::Painter,
    layout: &Layout,
    regions: &SplitRegions,
    params: &DisplayParams,
    theme: &Theme,
    ppp: f32,
) {
    let font = FontId::monospace(10.0);
    // Power axis: one label per division, right-aligned against the grid,
    // spanning the histogram region — the region whose vertical axis is
    // power.
    let hist = regions.histogram;
    if hist.height() >= MIN_LABELLED_HEIGHT {
        // One invariant, one place (C6, M1-J scope grant): the label y comes
        // from Layout::db_div_y over the histogram region — the same rect
        // and the same fraction the wgpu grid lines use, not an inline copy
        // of that arithmetic.
        let hist_layout = Layout { grid: hist };
        let db_divs = params.db_divs();
        for i in 0..=db_divs {
            let db = params.db_top - i as f32 * params.db_per_div;
            let y = hist_layout.db_div_y(i, db_divs);
            painter.text(
                egui::pos2(snap(layout.grid.min.x - 7.0, ppp), snap(y, ppp)),
                Align2::RIGHT_CENTER,
                axis::db_label(db),
                font.clone(),
                theme.text_dim,
            );
        }
    }
    // Frequency axis: ticks and labels from the one axis mapping
    // (crate::axis) — zoom-aware, and normalized with no Hz unit when no
    // rate is declared (D-028 §2).
    for tick in axis::freq_ticks(params) {
        let x = layout.freq_div_x(tick.div, params.freq_divs);
        painter.text(
            egui::pos2(snap(x, ppp), snap(layout.grid.max.y + 5.0, ppp)),
            Align2::CENTER_TOP,
            tick.label,
            font.clone(),
            theme.text_dim,
        );
    }
    // Waterfall time labels (FR-D5), only when a row cadence is declared —
    // a time base is never invented. Seconds for a rated source; rows of
    // spectra for a rate-less one (D-054): its only clock is the FFT
    // itself, and the axis states that unit rather than going blank or —
    // worse — inventing seconds.
    let wf = regions.waterfall;
    if wf.height() >= MIN_LABELLED_HEIGHT {
        let ticks = if let Some(interval) = params.wf_row_interval_s {
            axis::time_ticks(interval, wf.height() * ppp)
        } else if let Some(spectra) = params.wf_row_spectra {
            axis::row_ticks(spectra, wf.height() * ppp)
        } else {
            Vec::new()
        };
        for tick in ticks {
            let y = wf.min.y + tick.row / ppp;
            painter.text(
                egui::pos2(snap(layout.grid.min.x - 7.0, ppp), snap(y, ppp)),
                Align2::RIGHT_CENTER,
                tick.label,
                font.clone(),
                theme.text_dim,
            );
        }
    }
}

/// The FR-D6 cursor readout: crosshair plus frequency/power text, from the
/// same axis mapping the labels use. Outside the data regions it shows
/// nothing — never a clamped edge value.
fn cursor_overlay(
    ui: &Ui,
    layout: &Layout,
    regions: &SplitRegions,
    params: &DisplayParams,
    theme: &Theme,
    ppp: f32,
) {
    let Some(pos) = ui.ctx().pointer_latest_pos() else {
        return;
    };
    let Some(readout) = axis::cursor_readout(params, layout.grid, regions, pos, ppp) else {
        return;
    };
    let painter = ui.painter();
    // Crosshair: a frequency hairline through both panels (they share the
    // axis), a power hairline only inside the histogram.
    let x = snap(pos.x, ppp);
    painter.line_segment(
        [
            Pos2::new(x, layout.grid.min.y),
            Pos2::new(x, layout.grid.max.y),
        ],
        (1.0 / ppp, theme.grid_frame),
    );
    if readout.power_dbfs.is_some() {
        let y = snap(pos.y, ppp);
        painter.line_segment(
            [
                Pos2::new(regions.histogram.min.x, y),
                Pos2::new(regions.histogram.max.x, y),
            ],
            (1.0 / ppp, theme.grid_frame),
        );
    }
    let mut parts = vec![axis::format_freq_readout(readout.freq, params)];
    if let Some(db) = readout.power_dbfs {
        parts.push(format!("{db:.1} dBFS"));
    }
    if let Some(t) = readout.time_s {
        parts.push(format!("-{t:.2}s"));
    }
    if let Some(r) = readout.time_rows {
        // The rate-less waterfall's unit (D-054): rows of spectra, matching
        // the axis ticks and the `wf` readout's row-count span.
        parts.push(format!("-{r:.0}r"));
    }
    painter.text(
        egui::pos2(
            snap(layout.grid.max.x - 8.0, ppp),
            snap(layout.grid.min.y + 6.0, ppp),
        ),
        Align2::RIGHT_TOP,
        parts.join("  "),
        FontId::monospace(11.0),
        theme.accent,
    );
}

/// The §2.6 detail panel: track id, center/bandwidth and every
/// `Annotation.detail` line verbatim, plus copy-as-text (FR-AU4). An
/// `egui::Area`, the same idiom `help_overlay` already uses. A hover preview
/// has no close button (it closes itself on pointer-leave); a pinned panel
/// does.
fn annotation_detail_panel(
    ui: &mut Ui,
    theme: &Theme,
    visible: &[VisibleTrack],
    overlay_state: &mut AnnotationOverlayState,
    grid: Rect,
) {
    let Some(id) = overlay_state.detail().open_track() else {
        return;
    };
    let Some(track) = visible.iter().find(|v| v.track == id) else {
        overlay_state.detail_mut().close();
        return;
    };
    let pinned = overlay_state.detail().is_pinned();
    let mut still_open = true;
    let anchor = Pos2::new(grid.center().x, grid.min.y + 20.0);
    egui::Area::new(egui::Id::new(("annotation-detail", id.0)))
        .default_pos(anchor)
        .constrain_to(grid)
        .order(egui::Order::Foreground)
        .show(ui.ctx(), |ui| {
            Frame::popup(ui.style()).show(ui, |ui| {
                ui.set_max_width(340.0);
                ui.horizontal(|ui| {
                    ui.label(chrome_text(&format!("track {}", id.0), 11.0, theme.accent));
                    if pinned {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("×").clicked() {
                                still_open = false;
                            }
                        });
                    }
                });
                ui.separator();
                ui.label(
                    egui::RichText::new(&track.label)
                        .font(FontId::monospace(11.0))
                        .color(theme.text),
                );
                for line in &track.detail {
                    ui.label(
                        egui::RichText::new(line)
                            .font(FontId::monospace(11.0))
                            .color(theme.text_dim),
                    );
                }
                if pinned && ui.small_button("copy").clicked() {
                    let mut text = track.label.clone();
                    for line in &track.detail {
                        text.push('\n');
                        text.push_str(line);
                    }
                    ui.ctx().copy_text(text);
                }
            });
        });
    if !still_open {
        overlay_state.detail_mut().close();
    }
}

/// The translucent drag-select rectangle while a zoom drag is in progress.
fn selection_overlay(ui: &Ui, layout: &Layout, view: &ViewState, theme: &Theme) {
    let Some(x0) = view.drag_from else {
        return;
    };
    let Some(pos) = ui.ctx().pointer_latest_pos() else {
        return;
    };
    let grid = layout.grid;
    let (a, b) = if x0 <= pos.x {
        (x0, pos.x)
    } else {
        (pos.x, x0)
    };
    let a = a.clamp(grid.min.x, grid.max.x);
    let b = b.clamp(grid.min.x, grid.max.x);
    let rect = Rect::from_min_max(Pos2::new(a, grid.min.y), Pos2::new(b, grid.max.y));
    let mut fill = theme.accent;
    fill = fill.gamma_multiply(0.12);
    ui.painter().rect_filled(rect, 0.0, fill);
    ui.painter()
        .rect_stroke(rect, 0.0, (1.0, theme.accent), egui::StrokeKind::Inside);
}

/// The FR-D7 help overlay, generated from [`BINDINGS`] — the same table that
/// binds the keys, so no binding can exist that this overlay does not list.
/// The mouse gestures (not key bindings) are listed after the table.
/// Screen-centered by anchor rather than by a hand-tuned offset, so the
/// overlay stays fully visible as the table grows (M1-J added six rows and
/// the fixed offset ran it off the bottom edge).
fn help_overlay(ui: &mut Ui, theme: &Theme) {
    // Two columns, not one. The single column fitted while the table held 29
    // bindings; D-062 and D-065 §3 took it to 31 and the tail — including
    // the FR-D6 gestures — fell off the bottom of a 720 pt window, which is
    // the one failure mode an overlay generated from the table must not
    // have. Splitting the rows is height-proportional to the table, so the
    // next binding does not reintroduce it.
    let mut rows: Vec<(String, &str, bool)> = BINDINGS
        .iter()
        .map(|b| (keymap::binding_label(b), b.what, true))
        .collect();
    rows.extend(
        [
            ("drag", "zoom to selection"),
            ("drag divider", "histogram/waterfall split"),
            ("scroll", "zoom at cursor"),
            ("dbl-click", "reset zoom"),
        ]
        .into_iter()
        .map(|(gesture, what)| (gesture.to_owned(), what, false)),
    );
    let per_column = rows.len().div_ceil(2);

    egui::Area::new(ui.id().with("help-overlay"))
        .order(egui::Order::Foreground)
        .anchor(Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ui.ctx(), |ui| {
            Frame::new()
                .fill(theme.panel_bg)
                .stroke((1.0, theme.panel_edge))
                .inner_margin(Margin::symmetric(16, 12))
                .show(ui, |ui| {
                    ui.label(chrome_text("keyboard", 12.0, theme.accent));
                    ui.add_space(6.0);
                    ui.horizontal_top(|ui| {
                        for column in rows.chunks(per_column) {
                            ui.vertical(|ui| {
                                for (label, what, is_key) in column {
                                    let color = if *is_key {
                                        theme.accent
                                    } else {
                                        theme.text_dim
                                    };
                                    ui.horizontal(|ui| {
                                        ui.label(chrome_text(&format!("{label:>12}"), 10.0, color));
                                        ui.add_space(8.0);
                                        ui.label(chrome_text(what, 10.0, theme.text));
                                    });
                                }
                            });
                            ui.add_space(24.0);
                        }
                    });
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keymap::WF_SPAN_STEP;
    use crate::theme::Theme;
    use crate::zoom::ZoomSpan;

    fn stats() -> HudStats {
        HudStats {
            fps: 60.0,
            frame_ms: 2.5,
            fft_size: 1024,
            window_name: "HANN",
            source_label: "SYNTH".into(),
            backend: "cpu",
            colormap: "p7",
            input_rate_sps: 2_048_000.0,
            processed_pct: 100.0,
            ffts_per_s: 2000.0,
            dropped_batches: 0,
            persistence_tau_s: 1.0,
            gamma: 0.7,
            live_tau_s: 0.1,
            max_hold_tau_s: 2.0,
            overflow_events: 0,
            ..Default::default()
        }
    }

    fn raw_input(events: Vec<egui::Event>) -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 720.0),
            )),
            events,
            ..Default::default()
        }
    }

    fn run_chrome_on(
        ctx: &egui::Context,
        view: &mut ViewState,
        stats: &HudStats,
        events: Vec<egui::Event>,
    ) -> (Vec<String>, ChromeResponse, Rect) {
        let theme = Theme::default();
        let mut resp = None;
        let mut grid = Rect::NOTHING;
        let feed = AnnotationFeed::new();
        let mut overlay_state = AnnotationOverlayState::new();
        let out = ctx.run_ui(raw_input(events), |ui| {
            let r = draw(ui, &theme, view, stats, &feed, &mut overlay_state);
            grid = r.grid;
            resp = Some(r);
        });
        let mut texts = Vec::new();
        collect_text(&out.shapes, &mut texts);
        out.drop_without_applying_deltas();
        (texts, resp.unwrap(), grid)
    }

    fn run_chrome(
        view: &mut ViewState,
        stats: &HudStats,
        events: Vec<egui::Event>,
    ) -> (Vec<String>, ChromeResponse, Rect) {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        run_chrome_on(&ctx, view, stats, events)
    }

    fn collect_text(shapes: &[egui::epaint::ClippedShape], out: &mut Vec<String>) {
        for clipped in shapes {
            collect_shape(&clipped.shape, out);
        }
    }

    fn collect_shape(shape: &egui::Shape, out: &mut Vec<String>) {
        match shape {
            egui::Shape::Text(t) => out.push(t.galley.text().to_owned()),
            egui::Shape::Vec(nested) => {
                for s in nested {
                    collect_shape(s, out);
                }
            }
            _ => {}
        }
    }

    fn key_event(key: egui::Key) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        }
    }

    /// The chrome must run headlessly (no GPU, no window) — this is the same
    /// path the offscreen render uses.
    #[test]
    fn chrome_runs_headless_and_returns_grid() {
        let mut view = ViewState::default();
        view.params.sample_rate = Some(2_400_000.0);
        let (texts, resp, grid) = run_chrome(&mut view, &stats(), Vec::new());
        assert!(grid.width() > 100.0 && grid.height() > 100.0);
        assert!(!texts.is_empty());
        assert!(!resp.screenshot);
        assert_eq!(resp.requests, ChromeRequests::default());
    }

    /// C5/D-028: a rate-less source renders a normalized axis — signed
    /// fractions, a normalized span readout, and **no string containing
    /// "Hz" in any case, at several zoom levels** (the D-028 requirement
    /// stated as a test). The cursor readout is included by parking the
    /// pointer on the grid.
    #[test]
    fn rateless_source_shows_no_hz_anywhere_at_any_zoom() {
        for zoom in [
            ZoomSpan::FULL,
            ZoomSpan::new(0.2, 0.7),
            ZoomSpan::new(0.45, 0.55),
            ZoomSpan::new(0.499, 0.501),
        ] {
            let mut view = ViewState::default();
            view.params.sample_rate = None;
            view.params.center = None;
            view.params.zoom = zoom;
            let (texts, _, grid) = run_chrome(
                &mut view,
                &stats(),
                vec![egui::Event::PointerMoved(egui::pos2(640.0, 300.0))],
            );
            assert!(grid.width() > 0.0);
            for t in &texts {
                assert!(
                    !t.to_lowercase().contains("hz"),
                    "rate-less display shows a Hz unit at zoom {zoom:?}: {t:?}"
                );
            }
            if zoom.is_full() {
                for expected in ["-0.4", "-0.2", "0", "+0.2", "+0.4"] {
                    assert!(
                        texts.iter().any(|t| t == expected),
                        "axis label {expected:?} missing; texts: {texts:?}"
                    );
                }
                assert!(
                    texts.iter().any(|t| t == "±0.5 NORM"),
                    "span must be labelled normalized; texts: {texts:?}"
                );
            }
        }
    }

    /// With a declared rate the axis is honest Hz, and zooming changes the
    /// labels to the zoomed span (FR-D6: zoomed axes stay truthful).
    #[test]
    fn rated_axis_labels_follow_the_zoom() {
        let mut view = ViewState::default();
        view.params.sample_rate = Some(2_400_000.0);
        let (texts, _, _) = run_chrome(&mut view, &stats(), Vec::new());
        for expected in ["-960K", "-480K", "0", "+480K", "+960K"] {
            assert!(texts.iter().any(|t| t == expected), "missing {expected}");
        }
        view.params.zoom = ZoomSpan::new(0.25, 0.75);
        let (texts, _, _) = run_chrome(&mut view, &stats(), Vec::new());
        for expected in ["-480K", "-240K", "0", "+240K", "+480K"] {
            assert!(
                texts.iter().any(|t| t == expected),
                "zoomed label {expected} missing; texts: {texts:?}"
            );
        }
        // The span readout follows too.
        assert!(texts.iter().any(|t| t == "1.2M"));
    }

    /// The rendered readout string agrees with the rendered axis label when
    /// the pointer sits exactly on a labelled division — the two code paths
    /// reading one geometry, compared end to end through the actual chrome.
    #[test]
    fn rendered_readout_matches_rendered_axis_label() {
        for zoom in [ZoomSpan::FULL, ZoomSpan::new(0.2, 0.7)] {
            let mut view = ViewState::default();
            view.params.sample_rate = Some(2_400_000.0);
            view.params.zoom = zoom;
            // First pass discovers the grid rect.
            let (_, _, grid) = run_chrome(&mut view, &stats(), Vec::new());
            // Pointer on the center division (x fraction 0.5).
            let pos = egui::pos2(grid.min.x + grid.width() * 0.5, grid.center().y);
            let (texts, _, _) =
                run_chrome(&mut view, &stats(), vec![egui::Event::PointerMoved(pos)]);
            // The readout line contains the frequency and a dBFS power.
            let readout = texts
                .iter()
                .find(|t| t.contains("dBFS"))
                .expect("no cursor readout rendered");
            // x fraction 0.5 of the view is the view center: 0 Hz unzoomed,
            // -108 kHz at zoom (0.2..0.7) — the same value the axis math
            // yields, formatted at readout precision.
            let expect = axis::format_freq_readout(axis::freq_at(&view.params, 0.5), &view.params);
            assert!(
                readout.starts_with(&expect),
                "readout {readout:?} does not state {expect:?} at zoom {zoom:?}"
            );
        }
    }

    /// Outside the grid there is no readout at all (FR-D6: nothing rather
    /// than a clamped edge value).
    #[test]
    fn no_readout_outside_the_grid() {
        let mut view = ViewState::default();
        view.params.sample_rate = Some(2_400_000.0);
        let (texts, _, _) = run_chrome(
            &mut view,
            &stats(),
            vec![egui::Event::PointerMoved(egui::pos2(5.0, 300.0))],
        );
        assert!(
            !texts.iter().any(|t| t.contains("dBFS")),
            "readout rendered for a pointer outside the grid"
        );
    }

    /// FR-D7 keys reach the view through the chrome: reference level, pause
    /// (with its honesty indicator), help toggle, composer requests.
    #[test]
    fn keys_drive_the_view_and_requests() {
        let mut view = ViewState::default();
        view.params.sample_rate = Some(2_400_000.0);
        let (_, resp, _) = run_chrome(&mut view, &stats(), vec![key_event(egui::Key::ArrowUp)]);
        assert_eq!(view.params.db_top, 5.0);
        assert!(!resp.screenshot);

        let (texts, resp, _) = run_chrome(&mut view, &stats(), vec![key_event(egui::Key::Space)]);
        assert!(view.paused);
        assert!(
            texts.iter().any(|t| t == "PAUSED"),
            "paused display must say so"
        );
        assert!(!resp.requests.cycle_colormap);

        let (_, resp, _) = run_chrome(&mut view, &stats(), vec![key_event(egui::Key::C)]);
        assert!(resp.requests.cycle_colormap);

        let (_, resp, _) = run_chrome(&mut view, &stats(), vec![key_event(egui::Key::S)]);
        assert!(resp.screenshot);

        view.params.zoom = ZoomSpan::new(0.3, 0.6);
        run_chrome(&mut view, &stats(), vec![key_event(egui::Key::Z)]);
        assert!(view.params.zoom.is_full());
    }

    /// The FR-D3 keys reach the view and the app request through the real
    /// chrome, and the status bar states the max-hold mode the keys set —
    /// read from the same `ViewState`, so it cannot lie — alongside the
    /// D-007 decay τ (D-043), which stays visible in every mode.
    #[test]
    fn max_hold_keys_drive_the_view_and_the_status_readout() {
        let mut view = ViewState::default();
        view.params.sample_rate = Some(2_400_000.0);
        let s = stats();

        // Default: shown, decay mode, τ from the HUD — and the status bar
        // says so.
        let (texts, resp, _) = run_chrome(&mut view, &s, Vec::new());
        assert!(view.max_hold && view.max_hold_decay);
        assert!(!resp.requests.reset_max_hold);
        assert!(
            texts.iter().any(|t| t == "DECAY 2S"),
            "status bar must state the D-007 mode and τ; texts: {texts:?}"
        );

        // D switches to pure hold; the τ readout carries over unchanged.
        let (texts, _, _) = run_chrome(&mut view, &s, vec![key_event(egui::Key::D)]);
        assert!(!view.max_hold_decay);
        assert!(
            texts.iter().any(|t| t == "HOLD 2S"),
            "status bar must follow the mode key and keep stating τ; texts: {texts:?}"
        );

        // M hides the trace; τ still shows — the value did not stop
        // existing just because the trace is hidden.
        let (texts, _, _) = run_chrome(&mut view, &s, vec![key_event(egui::Key::M)]);
        assert!(!view.max_hold);
        assert!(
            texts.iter().any(|t| t == "OFF 2S"),
            "status bar must say the trace is hidden and still state τ; texts: {texts:?}"
        );

        // R raises the app-owned reset request.
        let (_, resp, _) = run_chrome(&mut view, &stats(), vec![key_event(egui::Key::R)]);
        assert!(resp.requests.reset_max_hold);
    }

    /// The D-035 τ keys reach the app requests through the real chrome:
    /// both time constants — the §7.3 persistence decay τ and the D-007
    /// max-hold decay τ — are exposed together in the one FR-D7 table.
    #[test]
    fn tau_keys_raise_the_app_requests() {
        let mut view = ViewState::default();
        view.params.sample_rate = Some(2_400_000.0);
        for (key, persistence, max_hold, live) in [
            (egui::Key::Comma, -1, 0, 0),
            (egui::Key::Period, 1, 0, 0),
            (egui::Key::Semicolon, 0, -1, 0),
            (egui::Key::Quote, 0, 1, 0),
            (egui::Key::K, 0, 0, -1),
            (egui::Key::L, 0, 0, 1),
        ] {
            let (_, resp, _) = run_chrome(&mut view, &stats(), vec![key_event(key)]);
            assert_eq!(
                resp.requests.persistence_tau_steps, persistence,
                "persistence τ steps for {key:?}"
            );
            assert_eq!(
                resp.requests.max_hold_tau_steps, max_hold,
                "max-hold τ steps for {key:?}"
            );
            assert_eq!(
                resp.requests.live_tau_steps, live,
                "live τ steps for {key:?}"
            );
        }
    }

    /// M1-J/M2-C status readouts: the active colormap name is on the status
    /// bar (AUDIT-1's missing readout), and the FR-D4 waterfall item follows
    /// the F, `-`, A and W keys — the mode-owned quantity (named, so a bare
    /// number can never mean a span in one mode and a row interval in the
    /// other), the aggregation and the auto marker — because it reads the
    /// same ViewState the keys mutate.
    #[test]
    fn status_bar_states_the_colormap_and_the_waterfall_controls() {
        let mut view = ViewState::default();
        view.params.sample_rate = Some(2_400_000.0);
        let (texts, _, _) = run_chrome(&mut view, &stats(), Vec::new());
        assert!(
            texts.iter().any(|t| t == "P7"),
            "the active colormap name is not on the status bar; texts: {texts:?}"
        );
        // Fast is the default (D-044/D-046): the readout states the row
        // interval, named as one.
        assert!(
            texts.iter().any(|t| t == "1MS/ROW MAX AUTO G1"),
            "the fast-default waterfall readout is not stated; texts: {texts:?}"
        );

        // A different colormap name renders as handed in (the value is the
        // composer's own; the app seal drives the C key end to end).
        let mut s = stats();
        s.colormap = "inferno";
        let (texts, _, _) = run_chrome(&mut view, &s, Vec::new());
        assert!(texts.iter().any(|t| t == "INFERNO"), "texts: {texts:?}");

        // Minus adjusts the quantity fast owns — the row interval — and the
        // readout follows.
        let (texts, _, _) = run_chrome(&mut view, &stats(), vec![key_event(egui::Key::Minus)]);
        assert!((view.wf_fast_interval_s - 1e-3 / WF_SPAN_STEP).abs() < 1e-7);
        assert_eq!(
            view.wf_time_span_s, 30.0,
            "fast keys must not move the span"
        );
        assert!(
            texts.iter().any(|t| t == "0.8MS/ROW MAX AUTO G1"),
            "the interval readout did not follow the key; texts: {texts:?}"
        );

        // F switches to traditional: the readout now names the span — and
        // Minus steps that span, leaving the fast interval alone.
        let (texts, _, _) = run_chrome(&mut view, &stats(), vec![key_event(egui::Key::F)]);
        assert!(
            texts.iter().any(|t| t == "SPAN 30S MAX AUTO G1"),
            "the traditional readout must name the span; texts: {texts:?}"
        );
        let interval_before = view.wf_fast_interval_s;
        let (texts, _, _) = run_chrome(&mut view, &stats(), vec![key_event(egui::Key::Minus)]);
        assert!((view.wf_time_span_s - 24.0).abs() < 1e-3);
        assert_eq!(view.wf_fast_interval_s, interval_before);
        assert!(
            texts.iter().any(|t| t == "SPAN 24S MAX AUTO G1"),
            "the span readout did not follow the key; texts: {texts:?}"
        );

        // A switches to mean; W turns the auto-range marker OFF — D-065 §2
        // made `auto` the default, so what the key does to this bar now is
        // take the word away.
        let (texts, _, _) = run_chrome(&mut view, &stats(), vec![key_event(egui::Key::A)]);
        assert!(
            texts.iter().any(|t| t == "SPAN 24S MEAN AUTO G1"),
            "the aggregation readout did not follow; texts: {texts:?}"
        );
        let (texts, _, _) = run_chrome(&mut view, &stats(), vec![key_event(egui::Key::W)]);
        assert!(!view.wf_auto_range);
        assert!(
            texts.iter().any(|t| t == "SPAN 24S MEAN G1"),
            "the auto-range readout did not follow; texts: {texts:?}"
        );

        // And back to fast: the interval readout returns, exactly as left.
        let (texts, _, _) = run_chrome(&mut view, &stats(), vec![key_event(egui::Key::F)]);
        assert!(
            texts.iter().any(|t| t == "0.8MS/ROW MEAN G1"),
            "toggling back must restore the interval readout; texts: {texts:?}"
        );
    }

    /// D-054: a rate-less source's waterfall readout never states seconds —
    /// fast mode is pinned at one row per spectrum, traditional names its
    /// span as a row count.
    #[test]
    fn rateless_waterfall_readout_states_spectra_not_seconds() {
        let mut view = ViewState::default();
        assert_eq!(view.params.sample_rate, None, "test premise: rate-less");
        let (texts, _, _) = run_chrome(&mut view, &stats(), Vec::new());
        assert!(
            texts.iter().any(|t| t == "SPEC/ROW MAX AUTO G1"),
            "fast + rate-less must state one row per spectrum; texts: {texts:?}"
        );
        let (texts, _, _) = run_chrome(&mut view, &stats(), vec![key_event(egui::Key::F)]);
        assert!(
            texts.iter().any(|t| t == "SPAN 1024R MAX AUTO G1"),
            "traditional + rate-less must state a row-count span; texts: {texts:?}"
        );
    }

    /// FR-D4's split really is draggable (M1-J, the gesture seal): press on
    /// the divider through the real chrome, move, release — the split
    /// follows the pointer continuously — and an FR-D6 zoom drag on the
    /// data surface still works unchanged, with neither gesture stealing
    /// from the other.
    #[test]
    fn divider_drag_moves_the_split_and_zoom_drag_still_zooms() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        view.params.sample_rate = Some(2_400_000.0);

        // Frame 1 registers the widgets (egui hit-tests against the previous
        // frame's rects) and discovers the geometry.
        let (_, _, grid) = run_chrome_on(&ctx, &mut view, &stats(), Vec::new());
        assert!(grid.height() > 100.0);

        // Press on the divider, where the DEFAULT view places it (D-040:
        // both surfaces show, so it sits mid-grid, not on an edge).
        let layout = Layout { grid };
        let default_regions = layout.split(view.split);
        let start = egui::pos2(
            grid.center().x,
            (default_regions.histogram.max.y + default_regions.waterfall.min.y) * 0.5,
        );
        run_chrome_on(
            &ctx,
            &mut view,
            &stats(),
            vec![egui::Event::PointerButton {
                pos: start,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::default(),
            }],
        );
        // Drag upward in steps: the split follows each position (continuous,
        // not stepped — every value between is reachable).
        let mut last = view.split;
        for frac in [0.5, 0.35, 0.25] {
            let y = grid.min.y + grid.height() * frac;
            run_chrome_on(
                &ctx,
                &mut view,
                &stats(),
                vec![egui::Event::PointerMoved(egui::pos2(grid.center().x, y))],
            );
            let expect = layout.split_ratio_at_y(y);
            assert!(
                (view.split - expect).abs() < 0.02,
                "split {} did not follow the drag to y fraction {frac} (expect {expect})",
                view.split
            );
            assert!(view.split < last, "the split must move with the pointer");
            last = view.split;
        }
        // Release; the split stays where the drag left it, and no zoom was
        // set by the divider drag.
        run_chrome_on(
            &ctx,
            &mut view,
            &stats(),
            vec![egui::Event::PointerButton {
                pos: egui::pos2(grid.center().x, grid.min.y + grid.height() * 0.25),
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        let dragged_split = view.split;
        assert!((0.15..0.35).contains(&dragged_split));
        assert!(
            view.params.zoom.is_full(),
            "a divider drag must not zoom (FR-D6 must not be stolen)"
        );

        // An FR-D6 zoom drag on the data surface (inside the histogram
        // region, away from the divider band) still zooms — and does not
        // move the split.
        let regions = layout.split(view.split);
        let y = regions.histogram.center().y;
        let (x0, x1) = (
            grid.min.x + grid.width() * 0.25,
            grid.min.x + grid.width() * 0.75,
        );
        run_chrome_on(
            &ctx,
            &mut view,
            &stats(),
            vec![egui::Event::PointerButton {
                pos: egui::pos2(x0, y),
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::default(),
            }],
        );
        // Incremental motion, as a real mouse delivers it: egui latches the
        // drag origin at the first post-threshold position.
        run_chrome_on(
            &ctx,
            &mut view,
            &stats(),
            vec![egui::Event::PointerMoved(egui::pos2(x0 + 8.0, y))],
        );
        run_chrome_on(
            &ctx,
            &mut view,
            &stats(),
            vec![egui::Event::PointerMoved(egui::pos2(x1, y))],
        );
        run_chrome_on(
            &ctx,
            &mut view,
            &stats(),
            vec![egui::Event::PointerButton {
                pos: egui::pos2(x1, y),
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        assert!(
            !view.params.zoom.is_full(),
            "the FR-D6 zoom drag stopped working"
        );
        assert!(
            (view.params.zoom.lo() - 0.25).abs() < 0.05
                && (view.params.zoom.hi() - 0.75).abs() < 0.05,
            "zoom {:?} does not match the dragged selection",
            view.params.zoom
        );
        assert_eq!(
            view.split, dragged_split,
            "a zoom drag must not move the split"
        );
    }

    /// The help overlay lists every binding — generated from the same table,
    /// verified against the rendered text (uppercased by the chrome idiom).
    /// An egui Area is sized on its first frame and painted from the next,
    /// so the overlay is asserted on a second frame with help still open.
    #[test]
    fn help_overlay_lists_every_binding() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        run_chrome_on(
            &ctx,
            &mut view,
            &stats(),
            vec![key_event(egui::Key::Questionmark)],
        );
        assert!(view.help_open);
        let (texts, _, _) = run_chrome_on(&ctx, &mut view, &stats(), Vec::new());
        assert!(view.help_open);
        let all = texts.join("\n");
        for b in BINDINGS {
            let key = keymap::binding_label(b).to_uppercase();
            let what = b.what.to_uppercase();
            assert!(
                all.contains(key.trim()),
                "overlay missing key {key:?}; texts: {texts:?}"
            );
            assert!(all.contains(&what), "overlay missing description {what:?}");
        }
        // Second press closes it — same context, so the sized Area would
        // otherwise still paint.
        run_chrome_on(
            &ctx,
            &mut view,
            &stats(),
            vec![key_event(egui::Key::Questionmark)],
        );
        assert!(!view.help_open);
        let (texts, _, _) = run_chrome_on(&ctx, &mut view, &stats(), Vec::new());
        assert!(!texts.join("\n").contains("RESET ZOOM"));
    }

    /// FR-D5 HiDPI: at fractional scale factors the frequency-label anchors
    /// land on whole physical pixels — the same snap the wgpu grid lines
    /// use, so text stays crisp and cannot drift from the grid. The label's
    /// top edge *is* the snapped anchor (CENTER_TOP alignment), so it is
    /// what the shape exposes.
    #[test]
    fn label_anchors_snap_to_physical_pixels_at_fractional_scale() {
        // "0" is deliberately absent: the status bar renders its own "0"
        // (drops) which is egui-laid-out, not painter-snapped.
        let tick_labels = ["-960K", "-480K", "+480K", "+960K"];
        for ppp in [1.25f32, 1.5, 1.75] {
            let ctx = egui::Context::default();
            let theme = Theme::default();
            theme.install(&ctx);
            ctx.set_pixels_per_point(ppp);
            let mut view = ViewState::default();
            view.params.sample_rate = Some(2_400_000.0);
            let s = stats();
            let feed = AnnotationFeed::new();
            let mut overlay_state = AnnotationOverlayState::new();
            // set_pixels_per_point applies from the next frame; run one
            // warm-up pass, then inspect the scaled frame.
            for _ in 0..2 {
                let out = ctx.run_ui(raw_input(Vec::new()), |ui| {
                    draw(ui, &theme, &mut view, &s, &feed, &mut overlay_state);
                });
                out.drop_without_applying_deltas();
            }
            assert_eq!(ctx.pixels_per_point(), ppp);
            let out = ctx.run_ui(raw_input(Vec::new()), |ui| {
                draw(ui, &theme, &mut view, &s, &feed, &mut overlay_state);
            });
            let mut checked = 0;
            for clipped in &out.shapes {
                if let egui::Shape::Text(t) = &clipped.shape {
                    if tick_labels.contains(&t.galley.text()) {
                        let py = t.pos.y * ppp;
                        assert!(
                            (py - py.round()).abs() < 1e-2,
                            "label {:?} anchor y {} off the pixel grid at ppp {ppp}",
                            t.galley.text(),
                            t.pos.y
                        );
                        checked += 1;
                    }
                }
            }
            assert!(
                checked >= tick_labels.len(),
                "only {checked} axis labels found to check at ppp {ppp}"
            );
            out.drop_without_applying_deltas();
        }
    }

    /// A tunable source, as the app describes one to the chrome: every value
    /// the device's own (D-058).
    fn tunable() -> TuneUi {
        TuneUi {
            tunable: true,
            center_hz: Some(100e6),
            range_hz: Some((42e6, 6.008e9)),
            step_hz: Some(2.048e6),
            status: None,
        }
    }

    fn run_tunable(
        ctx: &egui::Context,
        view: &mut ViewState,
        tune: &TuneUi,
        events: Vec<egui::Event>,
    ) -> (Vec<String>, ChromeResponse) {
        let theme = Theme::default();
        let mut resp = None;
        let feed = AnnotationFeed::new();
        let mut overlay_state = AnnotationOverlayState::new();
        let out = ctx.run_ui(raw_input(events), |ui| {
            resp = Some(draw_tunable(
                ui,
                &theme,
                view,
                &stats(),
                tune,
                &feed,
                &mut overlay_state,
            ));
        });
        let mut texts = Vec::new();
        collect_text(&out.shapes, &mut texts);
        out.drop_without_applying_deltas();
        (texts, resp.unwrap())
    }

    /// D-058 / D-055: one way of writing a frequency, not two. The filename
    /// convention's `1p985GHz`, the box's `1.985G`, and a bare count of hertz
    /// are three spellings of the same number.
    #[test]
    fn the_frequency_box_reads_the_products_own_unit_family() {
        for (text, hz) in [
            ("751M", 751e6),
            ("751m", 751e6),
            ("751MHz", 751e6),
            (" 751 M ", 751e6),
            ("1.985G", 1.985e9),
            ("1p985GHz", 1.985e9),
            ("94.9M", 94.9e6),
            ("751000000", 751e6),
            ("2048k", 2.048e6),
            ("632MHz", 632e6),
        ] {
            let got = parse_frequency_hz(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
            assert!(
                (got - hz).abs() < 1.0,
                "{text:?} parsed as {got} Hz, expected {hz} Hz"
            );
        }
    }

    /// A frequency the box cannot read never reaches the radio, and the
    /// message names the text that was wrong (§8.4).
    #[test]
    fn the_frequency_box_refuses_what_it_cannot_read() {
        for bad in ["", "   ", "M", "seven fifty one", "751X", "-751M", "0"] {
            let err = parse_frequency_hz(bad).unwrap_err();
            assert!(!err.is_empty(), "{bad:?} refused without saying why");
        }
        let err = parse_frequency_hz("751000000000000").unwrap_err();
        assert!(err.contains("751"), "unhelpful: {err}");
        // What the box shows is what the box can read back.
        for hz in [751e6, 1.985e9, 94.9e6, 42e6, 2.048e6] {
            let round = parse_frequency_hz(&format_tune_hz(hz)).unwrap();
            assert!((round - hz).abs() < 1.0, "{hz} did not round-trip");
        }
    }

    /// **D-058's increment, pinned.** One step is one sample rate, which is
    /// one window width, so adjacent positions tile the spectrum: the
    /// distance between any two reachable centres is a whole number of
    /// windows — never a fraction, which would either skip spectrum or
    /// re-show it.
    #[test]
    fn the_slider_steps_by_exactly_one_window_width() {
        let center = 100e6;
        let rate = Some(2.048e6);
        let range = (42e6, 6.008e9);

        // A nudge to the right of the handle lands exactly one rate away —
        // and NOT on an absolute multiple of the rate, which is what egui's
        // own `step_by` would have produced from a centre off that grid.
        let up = window_step_target(center, center + 1.5e6, rate, range).unwrap();
        assert_eq!(up, center + 2.048e6);
        let down = window_step_target(center, center - 1.5e6, rate, range).unwrap();
        assert_eq!(down, center - 2.048e6);
        // Several windows at once are still whole windows.
        let far = window_step_target(center, center + 7.1e6, rate, range).unwrap();
        assert_eq!(far, center + 3.0 * 2.048e6);
        // Adjacent positions tile: no gap, no overlap.
        assert!((up - center - (center - down)).abs() < 1e-6);
        // Staying put asks the radio for nothing.
        assert_eq!(window_step_target(center, center, rate, range), None);
        assert_eq!(
            window_step_target(center, center + 0.4e6, rate, range),
            None
        );

        // The device's own bounds clamp the step COUNT, so an edge position
        // is still a whole number of windows from the last one.
        let top = window_step_target(6.007e9, 6.008e9, rate, range);
        assert_eq!(top, None, "no room for a whole window: stay put");
        let near_top = window_step_target(6.0e9, 6.008e9, rate, range).unwrap();
        assert!(near_top <= range.1);
        assert!(((near_top - 6.0e9) / 2.048e6).fract().abs() < 1e-6);
    }

    /// The retune controls are drawn for a source that can tune and are
    /// **absent** for one that cannot: an inert box in front of a file replay
    /// is a control that cannot work.
    #[test]
    fn retune_controls_appear_only_for_a_tunable_source() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();

        let (texts, response) = run_tunable(&ctx, &mut view, &TuneUi::default(), Vec::new());
        assert!(
            !texts.iter().any(|t| t == "TUNE"),
            "a source with nothing to tune must draw no retune bar: {texts:?}"
        );
        assert_eq!(response.retune_hz, None);
        assert!(tune_slider_rect(&ctx).is_none());

        let (texts, _) = run_tunable(&ctx, &mut view, &tunable(), Vec::new());
        assert!(
            texts.iter().any(|t| t == "TUNE"),
            "a tunable source must offer the retune bar: {texts:?}"
        );
        // The device's own range is shown beside the slider, so the bounds
        // are visibly the radio's and not the app's invention.
        assert!(
            texts
                .iter()
                .any(|t| t.contains("42M") && t.contains("6.008G")),
            "the device's reported range must be shown: {texts:?}"
        );
        assert!(tune_slider_rect(&ctx).is_some());
    }

    /// D-058's stop-and-surface, answered: a device that reports no range
    /// gets no slider — never one bounded by a guess. The box remains,
    /// because the device is still the arbiter of what it will accept.
    #[test]
    fn no_reported_range_means_no_slider_rather_than_invented_bounds() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        let tune = TuneUi {
            range_hz: None,
            ..tunable()
        };
        let (texts, _) = run_tunable(&ctx, &mut view, &tune, Vec::new());
        assert!(
            texts.iter().any(|t| t == "NO DEVICE RANGE"),
            "the absence of a range must be stated: {texts:?}"
        );
        assert!(
            tune_slider_rect(&ctx).is_none(),
            "a slider was drawn with no device range behind it"
        );
    }

    /// The box shows **where the radio is**, not where it was asked to go:
    /// its unfocused text is re-read from the readback every frame, so a
    /// retune the device refused or clamped leaves the box at the real
    /// frequency (D-058).
    #[test]
    fn the_box_mirrors_the_device_readback_when_not_being_typed_into() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        run_tunable(&ctx, &mut view, &tunable(), Vec::new());
        let shown = ctx.data(|d| d.get_temp::<String>(tune_box_id())).unwrap();
        assert_eq!(shown, "100M");

        // The radio moved (or was clamped somewhere else): the box follows it
        // without being told to.
        let moved = TuneUi {
            center_hz: Some(751e6),
            ..tunable()
        };
        run_tunable(&ctx, &mut view, &moved, Vec::new());
        let shown = ctx.data(|d| d.get_temp::<String>(tune_box_id())).unwrap();
        assert_eq!(shown, "751M");
    }

    /// D-058 put a text box on screen and every character typed into it is
    /// also an FR-D7 key. Typing `751M` must not cycle the colormap on its
    /// way to the radio.
    #[test]
    fn typing_a_frequency_does_not_fire_the_keyboard_map() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        run_tunable(&ctx, &mut view, &tunable(), Vec::new());
        ctx.memory_mut(|m| m.request_focus(tune_box_id()));

        let before = view;
        let keys = [egui::Key::C, egui::Key::M, egui::Key::D, egui::Key::Space];
        let events = keys
            .iter()
            .map(|&key| egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::default(),
            })
            .collect();
        let (_, response) = run_tunable(&ctx, &mut view, &tunable(), events);
        assert_eq!(response.requests, ChromeRequests::default(), "keys leaked");
        assert_eq!(view.max_hold, before.max_hold);
        assert_eq!(view.max_hold_decay, before.max_hold_decay);
        assert_eq!(view.paused, before.paused);

        // With nothing focused, the same keys work exactly as they always did.
        ctx.memory_mut(|m| m.surrender_focus(tune_box_id()));
        let events = keys
            .iter()
            .map(|&key| egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::default(),
            })
            .collect();
        let (_, response) = run_tunable(&ctx, &mut view, &tunable(), events);
        assert!(response.requests.cycle_colormap, "the map is still bound");
        assert!(view.paused, "space still pauses");
    }

    /// Click the REAL step button at `up`, with real pointer input.
    fn click_step_button(
        ctx: &egui::Context,
        view: &mut ViewState,
        tune: &TuneUi,
        up: bool,
    ) -> ChromeResponse {
        let rect = tune_step_rect(ctx, up).expect("a tunable source must be given step buttons");
        let at = rect.center();
        run_tunable(
            ctx,
            view,
            tune,
            vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
        run_tunable(
            ctx,
            view,
            tune,
            vec![egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        )
        .1
    }

    /// **D-062's step buttons, clicked for real.** A pointer press and
    /// release on the button the chrome actually drew produces a retune
    /// request exactly one sample rate — one window width — from where the
    /// radio is, in the direction the arrow points.
    #[test]
    fn the_step_buttons_ask_for_exactly_one_window_in_each_direction() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        let tune = tunable();
        let center = tune.center_hz.unwrap();
        let rate = tune.step_hz.unwrap();

        // Frame 1 places the buttons; drawing must not retune.
        let (texts, response) = run_tunable(&ctx, &mut view, &tune, Vec::new());
        assert_eq!(response.retune_hz, None, "drawing must not retune");
        assert!(
            texts.iter().any(|t| t == "\u{25c0}") && texts.iter().any(|t| t == "\u{25b6}"),
            "the frequency must be flanked by a down and an up arrow: {texts:?}"
        );

        let response = click_step_button(&ctx, &mut view, &tune, true);
        assert_eq!(
            response.retune_hz,
            Some(center + rate),
            "the up button must ask for exactly one window up"
        );
        let response = click_step_button(&ctx, &mut view, &tune, false);
        assert_eq!(
            response.retune_hz,
            Some(center - rate),
            "the down button must ask for exactly one window down"
        );
        // Adjacent positions tile: the two are exactly one window apart on
        // either side of where the radio is.
        assert!(!view.paused, "a step must not touch the display state");
    }

    /// **D-066 §2's assignment, through the real chrome on a real (fake)
    /// radio.** The bare arrows tune by one window, the shifted arrows tune
    /// by a tenth of one, and **no arrow chord touches dB/div any more** —
    /// that moved to `O`/`P`, which in turn asks the radio for nothing.
    ///
    /// Asserted in both directions on every chord, because the coarse and
    /// fine steps differ only in size: a test that checked "the shifted arrow
    /// retunes" would pass just as happily if both chords produced the same
    /// step, which is the exact way this remap goes wrong.
    #[test]
    fn the_arrow_keys_are_coarse_and_fine_tune_and_db_per_div_moved_off_them() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        let tune = tunable();
        let center = tune.center_hz.unwrap();
        let rate = tune.step_hz.unwrap();
        run_tunable(&ctx, &mut view, &tune, Vec::new());

        let key = |key, modifiers| egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        };
        let shift = egui::Modifiers::SHIFT;
        let none = egui::Modifiers::default();

        // The bare arrows tune one whole window, and move nothing on the
        // display.
        let db_per_div = view.params.db_per_div;
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            vec![key(egui::Key::ArrowRight, none)],
        );
        assert_eq!(
            response.retune_hz,
            Some(center + rate),
            "the bare right arrow must ask for one whole window up"
        );
        assert_eq!(
            view.params.db_per_div, db_per_div,
            "the bare right arrow moved dB/div - it is off the arrows now"
        );

        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            vec![key(egui::Key::ArrowLeft, none)],
        );
        assert_eq!(response.retune_hz, Some(center - rate));
        assert_eq!(view.params.db_per_div, db_per_div);

        // The shifted arrows tune a TENTH of a window — a different size, not
        // a different quantity — and the bare vertical arrows are still the
        // reference level.
        let db_top = view.params.db_top;
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            vec![
                key(egui::Key::ArrowRight, shift),
                key(egui::Key::ArrowUp, none),
            ],
        );
        assert_eq!(
            response.retune_hz,
            Some(center + rate / FINE_TUNE_DIVISOR),
            "shift+right must ask for one tenth of a window up"
        );
        assert_ne!(
            response.retune_hz,
            Some(center + rate),
            "shift+right asked for a whole window - the fine step is not fine"
        );
        assert_eq!(
            view.params.db_per_div, db_per_div,
            "shift+right moved dB/div - it is off the arrows now"
        );
        assert!(
            view.params.db_top > db_top,
            "the bare up arrow must still raise the reference level"
        );

        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            vec![key(egui::Key::ArrowLeft, shift)],
        );
        assert_eq!(response.retune_hz, Some(center - rate / FINE_TUNE_DIVISOR));

        // dB/div is reachable on its own pair, and asks the radio for
        // nothing.
        let (_, response) = run_tunable(&ctx, &mut view, &tune, vec![key(egui::Key::P, none)]);
        assert_eq!(response.retune_hz, None, "the dB/div key reached the radio");
        assert!(view.params.db_per_div > db_per_div, "P must coarsen dB/div");
        let coarser = view.params.db_per_div;
        let (_, response) = run_tunable(&ctx, &mut view, &tune, vec![key(egui::Key::O, none)]);
        assert_eq!(response.retune_hz, None);
        assert!(view.params.db_per_div < coarser, "O must go finer");

        // Both counts are consumed by the bar, so neither reaches the
        // composer's request bag.
        assert_eq!(response.requests.tune_steps, 0);
        assert_eq!(response.requests.fine_tune_steps, 0);
    }

    /// **Coarse and fine in one frame are one request (D-058 §3).** Two keys
    /// can land in the same 16 ms frame; two round trips to the radio for one
    /// gesture would compute the second from a readback the first had already
    /// invalidated.
    #[test]
    fn a_coarse_and_a_fine_step_in_one_frame_converge_on_one_request() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        let tune = tunable();
        let center = tune.center_hz.unwrap();
        let rate = tune.step_hz.unwrap();
        run_tunable(&ctx, &mut view, &tune, Vec::new());

        let key = |key, modifiers| egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        };
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            vec![
                key(egui::Key::ArrowRight, egui::Modifiers::default()),
                key(egui::Key::ArrowRight, egui::Modifiers::SHIFT),
                key(egui::Key::ArrowRight, egui::Modifiers::SHIFT),
            ],
        );
        assert_eq!(
            response.retune_hz,
            Some(center + rate + 2.0 * rate / FINE_TUNE_DIVISOR),
            "one window plus two tenths, as one request"
        );
    }

    /// The fine step needs a declared rate exactly as the coarse one does —
    /// C5 forbids inventing a window width, and a tenth of an invented width
    /// is still invented.
    #[test]
    fn a_fine_step_needs_a_declared_rate() {
        assert_eq!(tune_target(750e6, 0, 1, None, None), None);
        assert_eq!(tune_target(750e6, 0, 1, Some(0.0), None), None);
        assert_eq!(tune_target(750e6, 0, 0, Some(2.4e6), None), None);
        // And it lands where it says it does, both ways.
        assert_eq!(tune_target(750e6, 0, 1, Some(2.4e6), None), Some(750.24e6));
        assert_eq!(tune_target(750e6, 0, -2, Some(2.4e6), None), Some(749.52e6));
        // The coarse count still tiles; the fine one rides on top of it.
        assert_eq!(
            tune_target(750e6, 1, 1, Some(2.4e6), None),
            Some(750e6 + 2.4e6 + 0.24e6)
        );
        // A reported range clamps the fine step into the band — it never
        // tiled, so there is no step count to clamp instead.
        assert_eq!(
            tune_target(750e6, 0, 5, Some(2.4e6), Some((700e6, 750.5e6))),
            Some(750.5e6)
        );
        // …and a fine step that would not move the radio asks for nothing.
        assert_eq!(
            tune_target(750e6, 0, 1, Some(2.4e6), Some((700e6, 750e6))),
            None
        );
    }

    /// The steps are **counted**, so two presses in one frame move two
    /// windows rather than one — and a press with nothing to tune asks for
    /// nothing at all.
    #[test]
    fn tune_steps_accumulate_within_a_frame_and_need_a_device() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        let tune = tunable();
        let center = tune.center_hz.unwrap();
        let rate = tune.step_hz.unwrap();
        run_tunable(&ctx, &mut view, &tune, Vec::new());

        // D-065 §1: the bare arrow is the tune key.
        let key = |k| egui::Event::Key {
            key: k,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        };
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            vec![key(egui::Key::ArrowRight), key(egui::Key::ArrowRight)],
        );
        assert_eq!(response.retune_hz, Some(center + 2.0 * rate));

        // Up then down in the same frame is a request for nothing: no round
        // trip to the radio for a gesture that changes nothing.
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            vec![key(egui::Key::ArrowRight), key(egui::Key::ArrowLeft)],
        );
        assert_eq!(response.retune_hz, None);

        // A source with nothing to tune draws no buttons and answers no keys.
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let (texts, response) = run_tunable(
            &ctx,
            &mut view,
            &TuneUi::default(),
            vec![key(egui::Key::ArrowRight)],
        );
        assert_eq!(response.retune_hz, None);
        assert!(
            !texts.iter().any(|t| t == "\u{25c0}" || t == "\u{25b6}"),
            "a file replay must be offered no step buttons: {texts:?}"
        );
        assert!(tune_step_rect(&ctx, true).is_none());
    }

    /// A device that declares **no sample rate** has no window width to step
    /// by, and C5 forbids inventing one. The buttons are drawn — the control
    /// exists — but disabled, and neither they nor the keys move the radio.
    #[test]
    fn no_declared_rate_means_the_steps_are_offered_but_inert() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        let tune = TuneUi {
            step_hz: None,
            ..tunable()
        };
        let (texts, _) = run_tunable(&ctx, &mut view, &tune, Vec::new());
        assert!(
            texts.iter().any(|t| t == "\u{25c0}"),
            "the control must still be shown, inert: {texts:?}"
        );
        let response = click_step_button(&ctx, &mut view, &tune, true);
        assert_eq!(
            response.retune_hz, None,
            "a step with no window width to step by moved the radio"
        );
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            vec![egui::Event::Key {
                key: egui::Key::ArrowRight,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::SHIFT,
            }],
        );
        assert_eq!(response.retune_hz, None);
    }

    /// **The focus rule, over D-066 §2's assignment (D-062 §4, M3-B's
    /// gate).** While the frequency box has the keyboard, **no** chord
    /// leaves it: the bare `\u{2192}` is cursor movement inside the box rather
    /// than a coarse tune, `shift+\u{2192}` is text selection rather than a fine
    /// one, and `P` is a character rather than a dB/div change. The gate is
    /// on the widget, not on a list of keys, which is why moving an action
    /// from one chord to another does not reopen it — and this test asserts
    /// every chord the remap touched so that stays true.
    #[test]
    fn a_focused_frequency_box_swallows_the_tune_keys_too() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        let tune = tunable();
        run_tunable(&ctx, &mut view, &tune, Vec::new());
        ctx.memory_mut(|m| m.request_focus(tune_box_id()));
        // One settling frame so the real `TextEdit` runs *while focused* and
        // installs its own key filter, exactly as it does on the frame a
        // user clicks into it. Without it the box holds focus but not the
        // filter, and egui's focus navigation walks the bare arrow straight
        // out of the widget — an artefact of granting focus from outside,
        // not something a user can produce.
        run_tunable(&ctx, &mut view, &tune, Vec::new());

        let press = |key, modifiers| {
            vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers,
            }]
        };
        let arrow = |modifiers| press(egui::Key::ArrowRight, modifiers);
        let db_per_div = view.params.db_per_div;
        // The bare arrow — the coarse tune key — never reaches the radio
        // while the box has the keyboard.
        let (_, response) = run_tunable(&ctx, &mut view, &tune, arrow(egui::Modifiers::default()));
        assert_eq!(
            response.retune_hz, None,
            "a keystroke aimed at the frequency box reached the radio"
        );
        assert_eq!(response.requests, ChromeRequests::default(), "keys leaked");
        // …nor does the shifted arrow, which is the fine tune since D-066 §2.
        let (_, response) = run_tunable(&ctx, &mut view, &tune, arrow(egui::Modifiers::SHIFT));
        assert_eq!(
            response.retune_hz, None,
            "the fine tune key reached the radio while the box had the keyboard"
        );
        assert_eq!(response.requests, ChromeRequests::default(), "keys leaked");
        // …and dB/div, wherever it lives, does not move while typing.
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            press(egui::Key::P, egui::Modifiers::default()),
        );
        assert_eq!(response.retune_hz, None);
        assert_eq!(
            view.params.db_per_div, db_per_div,
            "dB/div moved while typing"
        );

        // Focus surrendered, all three work again — the gate is a gate, not a
        // removal.
        ctx.memory_mut(|m| m.surrender_focus(tune_box_id()));
        let center = tune.center_hz.unwrap();
        let rate = tune.step_hz.unwrap();
        let (_, response) = run_tunable(&ctx, &mut view, &tune, arrow(egui::Modifiers::default()));
        assert_eq!(
            response.retune_hz,
            Some(center + rate),
            "the coarse tune key stayed gated after the box gave up the keyboard"
        );
        let (_, response) = run_tunable(&ctx, &mut view, &tune, arrow(egui::Modifiers::SHIFT));
        assert_eq!(
            response.retune_hz,
            Some(center + rate / FINE_TUNE_DIVISOR),
            "the fine tune key stayed gated"
        );
        run_tunable(
            &ctx,
            &mut view,
            &tune,
            press(egui::Key::P, egui::Modifiers::default()),
        );
        assert_ne!(view.params.db_per_div, db_per_div, "the map stayed gated");
    }

    /// Click the real waterfall-intensity button the chrome drew, with real
    /// pointer input — press and release on its own rect, exactly as
    /// `click_step_button` does for the tuner (D-031: a test that calls the
    /// step function proves arithmetic, not that a user can reach it).
    fn click_intensity_button(
        ctx: &egui::Context,
        view: &mut ViewState,
        tune: &TuneUi,
        up: bool,
    ) -> ChromeResponse {
        let rect = wf_intensity_step_rect(ctx, up).expect("the intensity buttons must be drawn");
        let at = rect.center();
        run_tunable(
            ctx,
            view,
            tune,
            vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
        run_tunable(
            ctx,
            view,
            tune,
            vec![egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        )
        .1
    }

    /// **D-065 §5's ten divisions, where a user can see them.**
    ///
    /// The seal the lane document asks for is not "the accessor returns 10"
    /// — that is asserted in `layout` — but that the **grid and the labels
    /// agree with it**. So this drives the production chrome and reads the
    /// power labels off the rendered frame: eleven of them, `0` down to
    /// `-100` in 10 dB steps, one per division boundary, at the same
    /// `Layout::db_div_y` fractions the wgpu grid lines use (`scene`'s
    /// `grid_line_count_matches_divisions` pins the GPU half against the
    /// same count).
    #[test]
    fn the_power_axis_is_ten_divisions_and_the_labels_agree() {
        let mut view = ViewState::default();
        view.params.sample_rate = Some(2_400_000.0);
        assert_eq!(view.params.db_divs(), 10, "D-065 §5: ten power divisions");

        let (texts, _, _) = run_chrome(&mut view, &stats(), Vec::new());
        let want: Vec<String> = (0..=view.params.db_divs())
            .map(|i| axis::db_label(view.params.db_top - i as f32 * view.params.db_per_div))
            .collect();
        assert_eq!(
            want,
            ["0", "-10", "-20", "-30", "-40", "-50", "-60", "-70", "-80", "-90", "-100"],
            "the division boundaries are not the D-065 §5 decade ladder"
        );
        for label in &want {
            assert!(
                texts.iter().any(|t| t == label),
                "the power axis is missing the {label} dB label; texts: {texts:?}"
            );
        }
        // Eleven boundaries and no twelfth: the −110 the display used to
        // carry must be gone, not merely unlabelled.
        assert!(
            !texts.iter().any(|t| t == "-110"),
            "the old 11th division is still labelled; texts: {texts:?}"
        );
        // One label per boundary, once each — `0` excepted, which the DC
        // frequency tick also renders on the other axis.
        for label in want.iter().filter(|l| *l != "0") {
            assert_eq!(
                texts.iter().filter(|t| *t == label).count(),
                1,
                "the {label} dB label is drawn more than once; texts: {texts:?}"
            );
        }
    }

    /// **D-065 §3's control, clicked and typed for real.**
    ///
    /// The §7.6 waterfall intensity has two entry points and they must be one
    /// mechanism: a click on the button the chrome actually drew and a press
    /// of `Shift+PgUp`/`Shift+PgDn` move the same `ViewState::wf_gamma` by the
    /// same step, in the same direction — and the value is on the bar beside
    /// the buttons the whole time (D-043).
    ///
    /// It is also drawn for a source with **nothing to tune**: the waterfall
    /// exists whether or not a radio does, so unlike the tuner this control
    /// is never absent.
    #[test]
    fn the_waterfall_intensity_buttons_and_keys_are_one_control() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        // No device: the tuner is not drawn, the intensity control is.
        let plain = TuneUi::default();

        let (texts, _) = run_tunable(&ctx, &mut view, &plain, Vec::new());
        assert!(
            texts.iter().any(|t| t == "WF INTENSITY"),
            "the intensity control is missing for a source with no device; texts: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| t == "\u{25bc}") && texts.iter().any(|t| t == "\u{25b2}"),
            "the exponent must be flanked by a down and an up button; texts: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| t == "G1"),
            "the exponent's value is not beside its control; texts: {texts:?}"
        );

        // Up: more intensity, so the exponent goes DOWN one step.
        let before = view.wf_gamma;
        let response = click_intensity_button(&ctx, &mut view, &plain, true);
        assert!(
            (view.wf_gamma - before / keymap::GAMMA_STEP).abs() < 1e-6,
            "the up button did not step the exponent (now {})",
            view.wf_gamma
        );
        assert_eq!(
            response.retune_hz, None,
            "an intensity click asked a radio for something"
        );
        let clicked = view.wf_gamma;
        let (texts, _) = run_tunable(&ctx, &mut view, &plain, Vec::new());
        let shown = format!("G{}", num_text(clicked).to_uppercase());
        assert!(
            texts.iter().any(|t| t == &shown),
            "the readout did not follow the click (want {shown}); texts: {texts:?}"
        );

        // Down: back where it started, through the other button.
        click_intensity_button(&ctx, &mut view, &plain, false);
        assert!(
            (view.wf_gamma - before).abs() < 1e-6,
            "the step is reversible"
        );

        // And the keys move the same field by the same step, so the two
        // entry points cannot come to disagree.
        let key = |k, m| {
            vec![egui::Event::Key {
                key: k,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: m,
            }]
        };
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &plain,
            key(egui::Key::PageUp, egui::Modifiers::SHIFT),
        );
        assert!(
            (view.wf_gamma - clicked).abs() < 1e-6,
            "shift+pgup and the up button disagree ({} vs {clicked})",
            view.wf_gamma
        );
        assert_eq!(
            response.requests.gamma_steps, 0,
            "the waterfall key moved §7.3's histogram gamma"
        );
        // The bare key is still §7.3's, and leaves the waterfall alone.
        let wf_before = view.wf_gamma;
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &plain,
            key(egui::Key::PageUp, egui::Modifiers::default()),
        );
        assert_eq!(response.requests.gamma_steps, -1);
        assert_eq!(view.wf_gamma, wf_before, "the bare key moved the waterfall");
    }

    /// **What the control bar costs the spectrum, pinned (D-010).**
    ///
    /// D-062 asked for a larger tuner and D-065 §3 added a second control to
    /// the same bar; D-010 makes the display's legibility a product
    /// requirement and the lane document makes "would this cost the spectrum
    /// meaningful area?" an owner call. So the cost is measured rather than
    /// asserted to be small, in two parts:
    ///
    /// * the **tuner's incremental** cost — a tunable source's data surface
    ///   against a file replay's, in the same window;
    /// * the **whole chrome's** cost — every bar, including the control bar
    ///   a source with no device now also gets.
    ///
    /// Measured at 1280x720: a file replay's data surface is **571.0 pt**
    /// (chrome 20.7% of the window) and a tunable source's **566.6 pt**
    /// (21.3%), so the tuner's own widgets cost **4.4 pt**. Before this lane
    /// the two were 584.0 and 566.6 — so D-065 §3's control cost the replay
    /// case **13 pt** and the tunable case **nothing at all**: it fits inside
    /// the bar the tuner already stood the height of. That is the trade,
    /// stated rather than asserted to be small.
    ///
    /// Both bounds have headroom over the measured figures so font-metric
    /// drift does not flap them; growth past either fails here instead of
    /// quietly eating the spectrum.
    #[test]
    fn control_bar_cost_to_the_spectrum() {
        const WINDOW_H: f32 = 720.0;
        const TUNER_BUDGET: f32 = 0.08;
        const CHROME_BUDGET: f32 = 0.26;

        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        // Two frames each: egui sizes a panel on the first and reports the
        // settled layout on the second.
        run_tunable(&ctx, &mut view, &TuneUi::default(), Vec::new());
        let (_, plain) = run_tunable(&ctx, &mut view, &TuneUi::default(), Vec::new());
        run_tunable(&ctx, &mut view, &tunable(), Vec::new());
        let (_, tuned) = run_tunable(&ctx, &mut view, &tunable(), Vec::new());

        let tuner_cost = plain.grid.height() - tuned.grid.height();
        assert!(
            tuner_cost >= 0.0 && tuner_cost / WINDOW_H <= TUNER_BUDGET,
            "the tuner's widgets take {tuner_cost:.1} pt of a {WINDOW_H} pt window \
             ({:.1}% of its height) - past the {:.0}% D-010 budget the owner \
             signed off on. Enlarging the tuner further is an owner call, not \
             a lane one.",
            tuner_cost / WINDOW_H * 100.0,
            TUNER_BUDGET * 100.0
        );

        for (what, response) in [("file replay", plain), ("tunable source", tuned)] {
            let chrome = WINDOW_H - response.grid.height();
            let fraction = chrome / WINDOW_H;
            assert!(
                fraction <= CHROME_BUDGET,
                "all chrome takes {chrome:.1} pt of a {WINDOW_H} pt window for a \
                 {what} ({:.1}% of its height) - past the {:.0}% budget. The data \
                 does the talking (D-010); another bar is an owner call.",
                fraction * 100.0,
                CHROME_BUDGET * 100.0
            );
            eprintln!(
                "{what}: chrome {chrome:.1} pt of {WINDOW_H} pt ({:.2}%); data \
                 surface {:.1} pt",
                fraction * 100.0,
                response.grid.height()
            );
        }
        eprintln!("the tuner's own share: {tuner_cost:.1} pt");
    }

    /// **D-011 stands after this lane.** The zoom gestures reinterpret the
    /// existing FFT and ask nothing of any device: no drag, scroll or reset
    /// raises a retune request. (The structural half of the assertion — that
    /// the render crate cannot reach `set()` at all — lives in the app crate,
    /// where the dependency graph can be read.)
    #[test]
    fn no_zoom_gesture_asks_for_a_retune() {
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        view.params.sample_rate = Some(2.048e6);
        let tune = tunable();
        run_tunable(&ctx, &mut view, &tune, Vec::new());

        // A scroll-wheel zoom at the cursor.
        let pos = egui::pos2(640.0, 400.0);
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, 40.0),
                    phase: egui::TouchPhase::Move,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
        assert_eq!(response.retune_hz, None, "a zoom asked the device to move");

        // A drag-select zoom across the surface.
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerMoved(egui::pos2(900.0, 400.0)),
            ],
        );
        assert_eq!(response.retune_hz, None);
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            vec![egui::Event::PointerButton {
                pos: egui::pos2(900.0, 400.0),
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        assert_eq!(response.retune_hz, None);
        assert!(!view.params.zoom.is_full(), "test premise: a zoom happened");

        // And the Z reset, on a context of its own so the earlier wheel
        // event's smooth-scroll tail cannot re-zoom after the reset.
        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        view.params.sample_rate = Some(2.048e6);
        view.params.zoom = ZoomSpan::new(0.3, 0.4);
        let (_, response) = run_tunable(
            &ctx,
            &mut view,
            &tune,
            vec![egui::Event::Key {
                key: egui::Key::Z,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        assert_eq!(response.retune_hz, None);
        assert!(view.params.zoom.is_full(), "Z still resets the zoom");
    }
}
