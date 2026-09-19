// SPDX-License-Identifier: MIT

//! Windowed mode: winit event loop + wgpu surface + egui chrome, fed by the
//! batch pipeline ([`crate::pipeline`]).
//!
//! winit gives native Wayland *and* X11 on Linux (NFR-I2: Wayland is
//! first-class, no XWayland anywhere) and native macOS; wgpu picks
//! Vulkan/Metal accordingly. Rendering runs continuously — every
//! `RedrawRequested` draws a frame and immediately requests the next; vsync
//! (`PresentMode::AutoVsync`) paces the loop at the display rate (FR-D9
//! targets 60 fps). The render thread only *reads*: it copies the pipeline's
//! latest published trace and health snapshot at its own cadence (§7.0 —
//! compute and display rates are fully decoupled), so the HUD's honesty
//! figures come from the pipeline's own D-013 accounting and nowhere else.
//! Frame time is measured over a rolling window and shown in the HUD, per
//! the M0-C seal: measured, not asserted.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use phosphene_core::{MaxHoldMode, DBFS_FLOOR};
use phosphene_render::chrome::{TuneStatus, TuneUi};
use phosphene_render::{
    ChromeRequests, ChromeResponse, DisplayParams, FrameComposer, HudStats, Layout,
    OffscreenTarget, RtsaInput, Theme, ViewState, TAU_STEP,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use phosphene_sources::SourceMeta;

use crate::pipeline::{open_source, Pipeline, PipelineConfig, WfDrain, WfFrameConfig};
use crate::SourceSetup;

/// Map source metadata onto the display parameters (clarification C5 /
/// D-028): the declared rate passes through as the `Option` it is — a
/// rate-less source stays rate-less all the way to the render crate, which
/// labels its axis in normalized cycles/sample with no Hz unit anywhere. A
/// Hz center cannot be placed on a normalized axis, so it is not handed over
/// either. A rate is never invented.
///
/// This is the one mapping between what a source declares and what the
/// screen claims; the end-to-end test below drives it from a real rate-less
/// pipeline all the way to the rendered label text.
pub(crate) fn apply_meta(params: &mut DisplayParams, meta: &SourceMeta) {
    params.sample_rate = meta.sample_rate_hz;
    params.center = match meta.sample_rate_hz {
        Some(_) => meta.center_freq_hz,
        None => None,
    };
}

/// The per-frame waterfall display step (§7.6 — D-051's architecture):
/// the aggregation itself runs on the pipeline's compute thread, folding
/// **every** completed spectrum; this step hands the view's mode-owned
/// settings down ([`Pipeline::waterfall_frame`] derives the cadence in the
/// unit the source's rate makes honest, D-046/D-054), drains the rows the
/// compute thread finished since the last frame, and pushes them into the
/// composer as **one batched upload** (D-047) stating the interval the rows
/// actually carried — the value `waterfall_row_interval()` then hands back
/// for the FR-D5 time labels. Shared verbatim by the windowed and headless
/// frame loops; `rows_scratch` is the caller's preallocated drain buffer
/// (§7.7 — it grows once to the pending ring's bound and never again).
pub(crate) fn waterfall_frame_step(
    pipeline: &Pipeline,
    composer: &mut FrameComposer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view: &ViewState,
    panel_rows: u32,
    rows_scratch: &mut Vec<f32>,
) -> WfDrain {
    let drain = pipeline.waterfall_frame(&wf_config_for(view, panel_rows), rows_scratch);
    composer.push_waterfall_rows(device, queue, rows_scratch, drain.bins, Some(drain.cadence));
    drain
}

/// The one mapping from the view's waterfall state to a frame's
/// [`WfFrameConfig`] — every mode-owned quantity travels; the feed picks the
/// one the mode and the source's rate make meaningful (D-046/D-054).
pub(crate) fn wf_config_for(view: &ViewState, panel_rows: u32) -> WfFrameConfig {
    WfFrameConfig {
        mode: view.wf_mode,
        aggregation: view.wf_aggregation,
        span_s: view.wf_time_span_s,
        fast_interval_s: view.wf_fast_interval_s,
        span_rows: view.wf_span_rows,
        panel_rows,
    }
}

/// The D-035 τ-control step, shared verbatim by the windowed redraw and the
/// seal tests (D-031): apply this frame's persistence-decay, max-hold-decay
/// and live-trace averaging τ keys to the pipeline's accumulators,
/// [`TAU_STEP`] per keypress, multiplicatively. All three time constants
/// travel through the one FR-D7 table together (FR-D1 "persistence time" +
/// D-007 max-hold decay + FR-D2 averaging time, M1-J — one idiom).
/// Keypress-rate events; a frame with no τ key is a no-op.
pub(crate) fn apply_tau_requests(pipeline: &Pipeline, requests: &ChromeRequests) {
    if requests.persistence_tau_steps != 0 {
        pipeline.adjust_persistence_tau(TAU_STEP.powi(requests.persistence_tau_steps));
    }
    if requests.max_hold_tau_steps != 0 {
        pipeline.adjust_max_hold_tau(TAU_STEP.powi(requests.max_hold_tau_steps));
    }
    if requests.live_tau_steps != 0 {
        pipeline.adjust_live_tau(TAU_STEP.powi(requests.live_tau_steps));
    }
}

/// The FR-D3 per-frame step (§7.5, D-034), shared verbatim by the windowed
/// redraw and the seal tests (D-031): forward any R-key reset to the
/// pipeline's accumulator, apply the view's D-007 mode, tick the decay
/// against the frame's live trace, and hand back the slice `RtsaInput`
/// should carry — empty when the M key hides the trace ("empty means not
/// shown"). The accumulator keeps running while hidden, like the waterfall
/// behind a closed panel, so the history is there when the trace reappears.
pub(crate) fn max_hold_frame_step<'a>(
    pipeline: &Pipeline,
    view: &ViewState,
    requests: &ChromeRequests,
    dt: f32,
    live: &[f32],
    out: &'a mut Vec<f32>,
) -> &'a [f32] {
    if requests.reset_max_hold {
        pipeline.reset_max_hold();
    }
    let mode = if view.max_hold_decay {
        MaxHoldMode::DecayTowardLive
    } else {
        MaxHoldMode::PureHold
    };
    pipeline.max_hold_frame(dt, mode, live, out);
    if view.max_hold {
        out.as_slice()
    } else {
        &[]
    }
}

/// The waterfall cadence's panel height for this frame, in physical pixel
/// rows: the real waterfall region when the split shows it, else the full
/// grid height as the nominal panel — so history accumulates at a sensible
/// cadence even while the panel is closed, and is there when it opens.
pub(crate) fn waterfall_panel_rows(grid: egui::Rect, split: f32, pixels_per_point: f32) -> u32 {
    let regions = Layout { grid }.split(split);
    let wf = (regions.waterfall.height() * pixels_per_point).round() as u32;
    if wf >= 1 {
        wf
    } else {
        ((grid.height() * pixels_per_point).round() as u32).max(1)
    }
}

/// Build the chrome's view of the retune controls (D-058) from the pipeline's
/// own answers — the single production mapping, so a seal that drives this
/// cannot diverge from what a user's window offers.
///
/// Every value here is the **device's**: the centre it read back (C5), the
/// range it reported, the rate it is running at. Nothing is derived from what
/// anyone asked for, which is what makes the slider show where the radio is
/// rather than where it was dragged.
pub(crate) fn tune_ui(
    pipeline: &Pipeline,
    meta: &SourceMeta,
    status: Option<TuneStatus>,
) -> TuneUi {
    TuneUi {
        tunable: pipeline.caps().tune,
        center_hz: meta.center_freq_hz,
        range_hz: pipeline.tune_range().map(|r| (r.min_hz, r.max_hz)),
        // D-058: one increment is one sample rate, because the displayed span
        // IS the sample rate at complex baseband — so a step moves the window
        // by exactly its own width and adjacent positions tile the spectrum.
        step_hz: meta.sample_rate_hz,
        status,
    }
}

/// Act on one frame's retune request (D-056/D-058) — the consume half of what
/// the chrome's box and slider produce, exactly as
/// [`consume_screenshot_request`] is for the screenshot key.
///
/// Returns `None` when no retune was asked for. Otherwise it applies the one
/// `set(Control::CenterFreqHz)` path — both widgets, one mechanism — and
/// hands back what the **device** said, which is also what lands in `status`
/// for the chrome to show. A refusal is a message the user reads, never a
/// silent no-op: the display is left exactly where the radio is.
pub(crate) fn consume_retune_request(
    pipeline: &Pipeline,
    response: &ChromeResponse,
    status: &mut Option<TuneStatus>,
) -> Option<Result<f64, String>> {
    let target_hz = response.retune_hz?;
    let outcome = pipeline.retune(target_hz);
    *status = Some(match &outcome {
        // The device's readback, not the request — a tuner that snapped to
        // its own grid says so here (C5).
        Ok(hz) => TuneStatus {
            message: format!("tuned {}", phosphene_render::chrome::format_tune_hz(*hz)),
            ok: true,
        },
        Err(e) => TuneStatus {
            message: e.clone(),
            ok: false,
        },
    });
    Some(outcome)
}

/// Clear the render-side waterfall history when the pipeline crosses a retune
/// boundary — the consume half of D-056's reset.
///
/// The pipeline resets everything it owns (persistence, max hold, live trace,
/// the pending-row ring) at the exact batch the new centre becomes true of.
/// The rows the display has **already** taken live in the composer's ring
/// texture; without this they would keep scrolling under the new axis —
/// history measured at one frequency, labelled with another, which is
/// precisely the plausible lie D-056 exists to forbid.
///
/// The ring is overwritten with a full screen of `ROW_FLOOR_DBFS`, which is
/// **the render crate's own encoding of "no data"** — the identical value it
/// writes itself when it blanks a freshly allocated ring, the value §7.6
/// already renders for gap time, and the value the waterfall's auto-range
/// explicitly excludes because it "is 'no data' and must not drag the
/// auto-range floor down". So this asserts nothing about the spectrum at the
/// new centre; it states that nothing has been measured there yet, in the one
/// vocabulary the surface already has for saying so.
///
/// Rebuilding the [`FrameComposer`] instead would be the obvious move and is
/// **wrong**: the composer owns the retained `egui_wgpu` texture set, and
/// egui only sends *deltas*, so a fresh renderer never receives the font
/// atlas again. egui-wgpu does not fail loudly on that — it logs a missing
/// texture and skips the mesh — so the entire chrome would silently stop
/// drawing from the first retune onward.
///
/// Deferred while the display is paused (FR-D12: the frozen picture stays
/// frozen, and the composer discards pushed rows anyway) — the generation is
/// not consumed, so the clear happens on the first unpaused frame instead of
/// being lost.
///
/// Returns `true` when it cleared.
pub(crate) fn consume_waterfall_reset(
    pipeline: &Pipeline,
    composer: &mut FrameComposer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    paused: bool,
    seen: &mut u64,
) -> bool {
    let generation = pipeline.retune_generation();
    if generation == *seen || paused {
        return false;
    }
    *seen = generation;
    let bins = pipeline.fft_size();
    // One allocation per retune — a keypress-rate event, the same class as
    // the histogram rebuild — rather than keeping a screen's worth of floor
    // permanently resident for something that happens once in a while.
    let blank = vec![phosphene_render::ROW_FLOOR_DBFS; phosphene_render::RING_ROWS as usize * bins];
    // No cadence: there is no history left to put a time label on, and
    // inventing one for a screen of "no data" is the FR-D5 dishonesty D-031
    // ruled on. The next real row push restores it.
    composer.push_waterfall_rows(device, queue, &blank, bins, None);
    true
}

/// Build this frame's [`HudStats`] from the pipeline's own accounting and
/// the composer's own selection — the single production function both the
/// windowed redraw loop and the real-source headless loop call. D-031's
/// lesson (six controls shipped dead behind green tests that only exercised
/// hand-built inputs) is why this is a function at all rather than inlined
/// at each call site: a seal that drives this exact path can never diverge
/// from what a user's window actually shows.
pub(crate) fn build_hud_stats(
    pipeline: &Pipeline,
    composer: &FrameComposer,
    frame_dts: &VecDeque<f32>,
    meta: &SourceMeta,
) -> HudStats {
    let n = frame_dts.len().max(1) as f32;
    let sum: f32 = frame_dts.iter().sum();
    let mean_dt = if sum > 0.0 { sum / n } else { 1.0 / 60.0 };
    let health = pipeline.health();
    HudStats {
        fps: 1.0 / mean_dt,
        frame_ms: mean_dt * 1e3,
        fft_size: pipeline.fft_size(),
        window_name: "HANN",
        source_label: meta.label.clone(),
        backend: pipeline.backend(),
        // The name of the map the composer is actually drawing with —
        // AUDIT-1's missing readout (M1-J). Read from the composer's own
        // selection, so it can never disagree with the pixels.
        colormap: composer.colormap().name(),
        input_rate_sps: health.input_rate_sps,
        processed_pct: health.processed_pct,
        ffts_per_s: health.ffts_per_s,
        dropped_batches: health.dropped_batches,
        // D-043: the four tunables' current values, read from the same
        // accumulators the keys mutate — never a mirrored copy.
        persistence_tau_s: pipeline.persistence_tau(),
        gamma: composer.gamma(),
        live_tau_s: pipeline.live_tau(),
        max_hold_tau_s: pipeline.max_hold_tau(),
        // D-050: the device's own overflow-event count.
        overflow_events: pipeline.device_overflow_events(),
        inspector_active: pipeline.inspector_active(),
        track_count: pipeline.annotation_track_count(),
        shedding: pipeline.analysis_shedding(),
        // D-125 Amendment 4 item 3: the honest coverage count.
        frames_analysed: pipeline.analysis_coverage().0,
        frames_received: pipeline.analysis_coverage().1,
    }
}

/// Run the windowed app over the resolved source. Blocks until the window
/// closes or the source fails.
pub fn run(
    fft_size: usize,
    setup: SourceSetup,
    analyze: bool,
    colormap: phosphene_render::Colormap,
) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| {
        format!(
            "cannot create a window ({e}); phosphene needs a Wayland or X11 \
             session on Linux — for scripted use, try `phosphene --headless`"
        )
    })?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        fft_size,
        setup: Some(setup),
        analyze,
        colormap,
        state: None,
        error: None,
    };
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("event loop error: {e}"))?;
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

struct App {
    fft_size: usize,
    /// Consumed by the first `resumed`.
    setup: Option<SourceSetup>,
    /// FR-AU1: the analysis worker's initial on/off state (the `--analyze`
    /// flag); the `I` key toggles it thereafter.
    analyze: bool,
    /// The starting colormap (`--colormap`); the `C` key cycles it
    /// thereafter (D-009).
    colormap: phosphene_render::Colormap,
    state: Option<WindowState>,
    error: Option<String>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let Some(setup) = self.setup.take() else {
                return;
            };
            match WindowState::new(
                event_loop,
                self.fft_size,
                setup,
                self.analyze,
                self.colormap,
            ) {
                Ok(state) => self.state = Some(state),
                Err(e) => {
                    self.error = Some(e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let _ = state.egui_state.on_window_event(&state.window, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            // Escape quits — unless a chrome widget has the keyboard, where
            // it means "abandon what I am typing". D-058 put a frequency box
            // on screen, and a box you cannot back out of without killing the
            // app is a trap.
            WindowEvent::KeyboardInput { event, .. }
                if event.state.is_pressed()
                    && event.logical_key == Key::Named(NamedKey::Escape)
                    && state.egui_ctx.memory(|m| m.focused().is_none()) =>
            {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                if let Err(e) = state.redraw() {
                    // A failed source is an exit with its own §8.4 message,
                    // not a frozen window.
                    self.error = Some(e);
                    event_loop.exit();
                    return;
                }
                // Continuous rendering: the live trace never idles.
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

struct WindowState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    composer: FrameComposer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    pipeline: Pipeline,
    theme: Theme,
    view: ViewState,
    /// Drain buffer for the §7.6 waterfall rows (D-051): filled by
    /// [`waterfall_frame_step`] each frame, uploaded as one batch (§7.7 —
    /// grows once to the pending ring's bound, never per frame).
    wf_rows: Vec<f32>,
    /// Render-thread copy of the latest published trace (preallocated).
    bins: Vec<f32>,
    /// Render-thread copy of the §7.5 max-hold trace (FR-D3), refreshed by
    /// [`max_hold_frame_step`] each frame (preallocated).
    max_hold: Vec<f32>,
    /// Render-thread copy of the §7.3 intensity grid (FR-D1), refreshed by
    /// [`Pipeline::intensity_frame`] each frame. Sized `bins × levels` on
    /// the first frame and stable after (§7.7).
    intensity: Vec<f32>,
    last_frame: Instant,
    /// Rolling window of recent frame-to-frame intervals, seconds.
    frame_dts: VecDeque<f32>,
    /// The outcome of the last retune (D-058), shown in the chrome until the
    /// next one replaces it.
    tune_status: Option<TuneStatus>,
    /// Retune generation this window has already cleared its waterfall for
    /// (D-056) — see [`consume_waterfall_reset`].
    seen_retune_gen: u64,
    /// AI-1's render-owned overlay state (interpolation, declutter, fade,
    /// the detail panel) — not `Copy`, so it lives beside `view` rather than
    /// inside it.
    overlay_state: phosphene_render::AnnotationOverlayState,
}

impl WindowState {
    fn new(
        event_loop: &ActiveEventLoop,
        fft_size: usize,
        setup: SourceSetup,
        analyze: bool,
        colormap: phosphene_render::Colormap,
    ) -> Result<Self, String> {
        // Source and pipeline first: a usage error (bad path, bad FFT size)
        // must never require a GPU to report itself (§8.4).
        let source = open_source(&setup.desc)?;
        // The view exists before the pipeline so the §7.3 histogram starts
        // accumulating against the display's own dB window, not a copy of it
        // (redraw keeps them in step when the keyboard moves it).
        let view = ViewState::default();
        let pipeline = Pipeline::start(
            source,
            PipelineConfig {
                fft_size,
                window: phosphene_core::WindowKind::Hann,
                paced: setup.paced,
                // D-125 Amendment 4 item 3: a file source lags behind real
                // time rather than shedding — every other source (radios,
                // the generator) keeps today's honest-shed behaviour.
                backpressure: matches!(setup.desc, phosphene_sources::SourceDesc::File(_)),
                db_bottom: view.params.db_bottom,
                db_top: view.params.db_top,
            },
        )?;
        pipeline.set_analyze_active(analyze);

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("phosphene")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)),
                )
                .map_err(|e| format!("window creation failed: {e}"))?,
        );

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("cannot create a render surface: {e}"))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))
        .map_err(|e| {
            format!(
                "no usable GPU adapter ({e}); phosphene needs Vulkan on Linux \
                 or Metal on macOS (mesa's lavapipe works as a software fallback)"
            )
        })?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("phosphene"),
            ..Default::default()
        }))
        .map_err(|e| format!("GPU device request failed: {e}"))?;

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or("surface is incompatible with the chosen adapter")?;
        // Prefer an sRGB view of the swapchain so scene colors land correctly.
        let caps = surface.get_capabilities(&adapter);
        if let Some(srgb) = caps.formats.iter().copied().find(|f| f.is_srgb()) {
            config.format = srgb;
        }
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        let mut composer = FrameComposer::new(&device, config.format);
        composer.set_colormap(colormap);
        let egui_ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&egui_ctx);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        window.request_redraw();
        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            composer,
            egui_ctx,
            egui_state,
            pipeline,
            theme,
            view,
            wf_rows: Vec::new(),
            bins: vec![DBFS_FLOOR; fft_size],
            max_hold: vec![DBFS_FLOOR; fft_size],
            intensity: Vec::new(),
            last_frame: Instant::now(),
            frame_dts: VecDeque::with_capacity(240),
            tune_status: None,
            seen_retune_gen: 0,
            overlay_state: phosphene_render::AnnotationOverlayState::new(),
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    fn hud_stats(&self, meta: &SourceMeta) -> HudStats {
        build_hud_stats(&self.pipeline, &self.composer, &self.frame_dts, meta)
    }

    fn redraw(&mut self) -> Result<(), String> {
        if let Some(e) = self.pipeline.source_error() {
            return Err(format!("source failed: {e}"));
        }
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        if self.frame_dts.len() >= 240 {
            self.frame_dts.pop_front();
        }
        self.frame_dts.push_back(dt);

        // Latest pipeline state, at display cadence; `apply_meta` is the one
        // C5/D-028 mapping from what the source declares to what the screen
        // claims.
        let meta = self.pipeline.meta();
        apply_meta(&mut self.view.params, &meta);
        // FR-D5 / D-031: the waterfall time base the labels state is the one
        // the composer's rows actually carried — both units, mirrored here
        // every frame, never set by hand. None until a waterfall producer
        // pushes rows; at most one is Some (D-054: one honest time base).
        self.view.params.wf_row_interval_s = self.composer.waterfall_row_interval();
        self.view.params.wf_row_spectra = self.composer.waterfall_row_spectra();
        self.pipeline.copy_trace(&mut self.bins);
        let hud = self.hud_stats(&meta);

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                eprintln!("skipping frame: surface validation error");
                return Ok(());
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let input = self.egui_state.take_egui_input(&self.window);
        let theme = self.theme;
        let mut view_state = self.view;
        let mut response = None;
        // D-058: the retune controls are drawn from the device's own answers,
        // and are absent entirely for a source with nothing to tune.
        let tune = tune_ui(&self.pipeline, &meta, self.tune_status.clone());
        let feed = self.pipeline.annotation_feed();
        let mut overlay_state = std::mem::take(&mut self.overlay_state);
        let output = self.egui_ctx.run_ui(input, |ui| {
            response = Some(phosphene_render::chrome::draw_tunable(
                ui,
                &theme,
                &mut view_state,
                &hud,
                &tune,
                &feed,
                &mut overlay_state,
            ));
        });
        self.view = view_state;
        self.overlay_state = overlay_state;
        let response = response.expect("chrome::draw runs inside run_ui");
        // D-056/D-058: apply this frame's retune through the one
        // `set(Control::CenterFreqHz)` path, and say what the device answered.
        // The axis does not move here — it moves when the samples at the new
        // centre arrive, which is the same instant the accumulators reset.
        if let Some(Err(e)) =
            consume_retune_request(&self.pipeline, &response, &mut self.tune_status)
        {
            eprintln!("phosphene: retune failed: {e}");
        }
        // D-056's reset, render half: clear the waterfall rows the composer
        // already holds when the pipeline crosses a retune boundary. Checked
        // BEFORE this frame's drain, so no row measured at the old centre is
        // ever composited under the new axis.
        let waterfall_cleared = consume_waterfall_reset(
            &self.pipeline,
            &mut self.composer,
            &self.device,
            &self.queue,
            self.view.paused,
            &mut self.seen_retune_gen,
        );
        // AC-9/D-056: the annotation overlay clears the instant the same
        // reset the waterfall/persistence/max-hold accumulators just took
        // is observed — never a stale, pre-retune track surviving under a
        // label that no longer describes the samples.
        if waterfall_cleared {
            self.overlay_state.clear();
        }
        // FR-AU1 (design §2.7): the `I` key surfaces as a one-shot request
        // since render has no analysis pipeline of its own to toggle.
        if response.requests.toggle_inspector {
            self.pipeline.toggle_analyze_active();
        }
        self.composer.apply_requests(&response.requests);
        self.egui_state
            .handle_platform_output(&self.window, output.platform_output);
        let primitives = self
            .egui_ctx
            .tessellate(output.shapes, output.pixels_per_point);
        // The screenshot re-render needs the same primitives; clone them
        // only when the key actually fired.
        let shot_primitives = if response.screenshot {
            primitives.clone()
        } else {
            Vec::new()
        };
        // §7.6 / D-051: the waterfall display step runs every frame —
        // configure the compute-side aggregator from the view, drain the
        // rows it finished (every spectrum reaches it, so `Max` means what
        // the spec says), and upload them as one batch. Pushed before the
        // render below so this frame's rows are in the ring it composites.
        let panel_rows =
            waterfall_panel_rows(response.grid, self.view.split, output.pixels_per_point);
        waterfall_frame_step(
            &self.pipeline,
            &mut self.composer,
            &self.device,
            &self.queue,
            &self.view,
            panel_rows,
            &mut self.wf_rows,
        );
        // §7.6 / M1-J: the W key's auto-range toggle is view state; the
        // composer adopts it every frame, the same way the max-hold mode is
        // mapped below.
        self.composer
            .set_waterfall_auto_range(self.view.wf_auto_range);
        // §7.6 / D-065 §3: the waterfall intensity exponent is view state on
        // the same footing, mirrored here so the exponent the shader receives
        // is the one the status bar and the control-bar buttons state.
        self.composer.set_waterfall_gamma(self.view.wf_gamma);
        // D-035: both decay time constants are live keyboard controls — the
        // §7.3 persistence τ and the D-007 max-hold τ, applied at keypress
        // rate before the ticks below use them.
        apply_tau_requests(&self.pipeline, &response.requests);
        // §7.3 / D-033: fold the counts accumulated since the last frame
        // into the intensity grid and read it — against the dB window the
        // chrome just finished applying this frame's keys to, so the grid
        // is always accumulated against the range it is drawn against
        // (a reference-level or dB/div change rebuilds it; see
        // `Pipeline::intensity_frame`).
        let (intensity_bins, intensity_levels) = self.pipeline.intensity_frame(
            dt,
            self.view.params.db_bottom,
            self.view.params.db_top,
            &mut self.intensity,
        );
        // §7.5 / FR-D3 (D-034: the app-side wiring this lane owns): forward
        // this frame's reset key, apply the D-007 mode the keys set, decay
        // toward the live trace copied above, and hand the trace — or
        // nothing, when hidden — to the render input.
        let max_hold_bins = max_hold_frame_step(
            &self.pipeline,
            &self.view,
            &response.requests,
            dt,
            &self.bins,
            &mut self.max_hold,
        );
        // The full RTSA composition (M1-B's surfaces consumed as merged):
        // the fed persistence histogram (FR-D1) with grid + trace in the
        // histogram region, the fed waterfall in its FR-D4 region — the
        // `[`/`]` split keys reveal it.
        self.composer.render_rtsa_frame(
            &self.device,
            &self.queue,
            &view,
            [self.config.width, self.config.height],
            RtsaInput {
                live_bins: &self.bins,
                max_hold_bins,
                intensity: &self.intensity,
                intensity_bins,
                intensity_levels,
                surface_points: response.grid,
                view: &self.view,
                theme: &theme,
            },
            phosphene_render::EguiFrame {
                primitives,
                textures_delta: output.textures_delta,
                pixels_per_point: output.pixels_per_point,
            },
        );
        // FR-D7/FR-D12: act on the screenshot request. Runs after the main
        // render so this frame's egui texture deltas are already applied;
        // the offscreen pass reuses the retained textures.
        if let Some(result) = consume_screenshot_request(
            &response,
            &self.device,
            &self.queue,
            &mut self.composer,
            self.config.format,
            [self.config.width, self.config.height],
            RtsaInput {
                live_bins: &self.bins,
                max_hold_bins,
                intensity: &self.intensity,
                intensity_bins,
                intensity_levels,
                surface_points: response.grid,
                view: &self.view,
                theme: &theme,
            },
            phosphene_render::EguiFrame {
                primitives: shot_primitives,
                textures_delta: egui::TexturesDelta::default(),
                pixels_per_point: output.pixels_per_point,
            },
            Path::new("."),
        ) {
            // A failed screenshot is reported, never a dead window.
            match result {
                Ok(path) => println!("screenshot: {}", path.display()),
                Err(e) => eprintln!("screenshot failed: {e}"),
            }
        }
        self.queue.present(frame);
        Ok(())
    }
}

/// Act on one frame's [`ChromeResponse`] screenshot request (FR-D7 key,
/// FR-D12 behaviour): re-render exactly what the user is seeing into an
/// offscreen target of the window's own surface format and write it to a
/// timestamped PNG in `dir`. Returns `None` when the response carries no
/// request — this function is the consume half of the request the chrome
/// produced, and the test below drives it with a response produced by the
/// real chrome (D-031).
#[allow(clippy::too_many_arguments)]
pub(crate) fn consume_screenshot_request(
    response: &ChromeResponse,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    composer: &mut FrameComposer,
    format: wgpu::TextureFormat,
    size: [u32; 2],
    scene: RtsaInput<'_>,
    egui_frame: phosphene_render::EguiFrame,
    dir: &Path,
) -> Option<Result<PathBuf, String>> {
    if !response.screenshot {
        return None;
    }
    let write = || -> Result<PathBuf, String> {
        let target = OffscreenTarget::with_format(device, size[0], size[1], format);
        composer.render_rtsa_frame(device, queue, &target.view, size, scene, egui_frame);
        let rgba = target.read_rgba(device, queue);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("system clock error: {e}"))?
            .as_millis();
        let path = dir.join(format!("phosphene-{stamp}.png"));
        crate::headless::write_png(&path, size[0], size[1], &rgba)?;
        Ok(path)
    };
    Some(write())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use phosphene_core::Complex;
    use phosphene_render::{RowAggregation, Theme};
    use std::sync::atomic::{AtomicBool, Ordering};

    use phosphene_sources::source::{apply_and_verify, settle_control, TuneRange};
    use phosphene_sources::{
        Control, ControlCaps, FileSource, FileSourceConfig, Input, IqFormat, SampleSink,
        SampleSource, SinkFlow, SourceDesc, SourceError,
    };

    use crate::pipeline::Pipeline;

    /// A wgpu device for the seal tests that need a real composer — the
    /// same request every existing GPU seal makes inline.
    fn gpu(label: &str) -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .expect("no GPU adapter — the M1-J seals cannot run");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            ..Default::default()
        }))
        .expect("GPU device request failed")
    }

    /// Drive one chrome frame exactly as redraw does, with an optional
    /// keypress, over a caller-owned view — the produce half of the M1-J
    /// produce-and-consume seals (D-031).
    fn chrome_frame(
        ctx: &egui::Context,
        theme: &Theme,
        view: &mut ViewState,
        hud: &HudStats,
        key: Option<egui::Key>,
    ) -> (Vec<String>, ChromeResponse) {
        let events = key
            .map(|key| {
                vec![egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::default(),
                }]
            })
            .unwrap_or_default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 720.0),
            )),
            events,
            ..Default::default()
        };
        let mut response = None;
        let feed = phosphene_render::AnnotationFeed::new();
        let mut overlay_state = phosphene_render::AnnotationOverlayState::new();
        let out = ctx.run_ui(input, |ui| {
            response = Some(phosphene_render::chrome::draw(
                ui,
                theme,
                view,
                hud,
                &feed,
                &mut overlay_state,
            ));
        });
        let mut texts = Vec::new();
        collect_text(&out.shapes, &mut texts);
        out.drop_without_applying_deltas();
        (texts, response.unwrap())
    }

    /// A hand-built HUD for chrome-driving seals whose assertions do not
    /// concern the honesty figures.
    fn test_hud() -> HudStats {
        HudStats {
            fps: 60.0,
            frame_ms: 2.5,
            fft_size: 512,
            window_name: "HANN",
            source_label: "TEST".into(),
            backend: "cpu",
            colormap: "p7",
            input_rate_sps: 0.0,
            processed_pct: 100.0,
            ffts_per_s: 0.0,
            dropped_batches: 0,
            persistence_tau_s: phosphene_core::DEFAULT_TAU_DECAY,
            gamma: phosphene_render::DEFAULT_GAMMA,
            live_tau_s: phosphene_core::DEFAULT_TAU_LIVE,
            max_hold_tau_s: phosphene_core::DEFAULT_TAU_MAX_HOLD,
            overflow_events: 0,
            ..Default::default()
        }
    }

    /// A hand-built HUD for the rate-less chrome seals — no rate anywhere.
    fn test_hud_rateless() -> HudStats {
        HudStats {
            input_rate_sps: 0.0,
            ffts_per_s: 0.0,
            ..test_hud()
        }
    }

    fn collect_text(shapes: &[egui::epaint::ClippedShape], out: &mut Vec<String>) {
        for clipped in shapes {
            match &clipped.shape {
                egui::Shape::Text(t) => out.push(t.galley.text().to_owned()),
                egui::Shape::Vec(nested) => {
                    for shape in nested {
                        if let egui::Shape::Text(t) = shape {
                            out.push(t.galley.text().to_owned());
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// C5/D-028 across the app→render boundary: a source that honestly
    /// declares NO rate, running through the real pipeline, must reach the
    /// screen as a normalized axis — signed cycles/sample fractions and a
    /// "±0.5 norm" span, with the geometry-only 1.0 span never leaking into
    /// a label. This starts from the real `SourceMeta`, not from a
    /// hand-set flag.
    #[test]
    fn rateless_pipeline_reaches_the_screen_as_a_normalized_axis() {
        let n = 512;
        // Two whole batches of cf32 zeros; rate deliberately undeclared.
        let bytes = vec![0u8; 2 * n * 8];
        let config = FileSourceConfig::new(
            Input::Path(std::path::PathBuf::from("in-memory.cf32")),
            IqFormat::Cf32,
        );
        let source =
            FileSource::from_reader(Box::new(std::io::Cursor::new(bytes)), config).unwrap();
        let mut pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: phosphene_core::WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();

        // The REAL metadata, as the redraw path reads it.
        let meta = pipeline.meta();
        assert_eq!(meta.sample_rate_hz, None, "test premise: rate-less");
        let mut view = ViewState::default();
        apply_meta(&mut view.params, &meta);
        // D-028: the absence of a rate survives as an absence — no
        // fabricated span exists anywhere between the source and the screen.
        assert_eq!(view.params.sample_rate, None);
        assert_eq!(view.params.center, None);

        // Render the chrome exactly as redraw() would and read the labels.
        let health = pipeline.health();
        let hud = HudStats {
            fps: 60.0,
            frame_ms: 2.5,
            fft_size: n,
            window_name: "HANN",
            source_label: meta.label.clone(),
            backend: pipeline.backend(),
            colormap: "p7",
            input_rate_sps: health.input_rate_sps,
            processed_pct: health.processed_pct,
            ffts_per_s: health.ffts_per_s,
            dropped_batches: health.dropped_batches,
            persistence_tau_s: phosphene_core::DEFAULT_TAU_DECAY,
            gamma: phosphene_render::DEFAULT_GAMMA,
            live_tau_s: phosphene_core::DEFAULT_TAU_LIVE,
            max_hold_tau_s: phosphene_core::DEFAULT_TAU_MAX_HOLD,
            overflow_events: 0,
            ..Default::default()
        };
        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 720.0),
            )),
            ..Default::default()
        };
        let feed = phosphene_render::AnnotationFeed::new();
        let mut overlay_state = phosphene_render::AnnotationOverlayState::new();
        let out = ctx.run_ui(input, |ui| {
            phosphene_render::chrome::draw(ui, &theme, &mut view, &hud, &feed, &mut overlay_state);
        });
        let mut texts = Vec::new();
        collect_text(&out.shapes, &mut texts);
        out.drop_without_applying_deltas();

        for expected in ["-0.4", "-0.2", "0", "+0.2", "+0.4"] {
            assert!(
                texts.iter().any(|t| t == expected),
                "normalized axis label {expected:?} missing; texts: {texts:?}"
            );
        }
        assert!(
            texts.iter().any(|t| t == "±0.5 NORM"),
            "span must say normalized; texts: {texts:?}"
        );
        assert!(
            !texts.iter().any(|t| t == "1"),
            "the geometry-only 1.0 span leaked into a label; texts: {texts:?}"
        );
        pipeline.shutdown();
    }

    /// Rate for the §7.6 feed seals: 2^21 S/s, so one N=512 spectrum spans
    /// exactly 2^-12 s in f32 and every row-boundary computation below is
    /// binary-exact, not approximate.
    const WF_TEST_RATE: f64 = 2_097_152.0;

    /// One spectrum's span at [`WF_TEST_RATE`], seconds (exact in f32).
    fn wf_test_si(n: usize) -> f32 {
        (n as f64 / WF_TEST_RATE) as f32
    }

    /// A **rated** in-memory cf32 replay, unthrottled (the §7.6 seals need
    /// signal-time determinism, not wall-clock pacing), behind the test
    /// gate so the aggregator can be configured before spectra flow.
    fn rated_gated_pipeline(
        n: usize,
        bytes: Vec<u8>,
        gate: std::sync::Arc<std::sync::atomic::AtomicU64>,
    ) -> Pipeline {
        use phosphene_sources::ReplayPace;
        let config = FileSourceConfig {
            input: Input::Path(std::path::PathBuf::from("in-memory.cf32")),
            format: IqFormat::Cf32,
            sample_rate_hz: Some(WF_TEST_RATE),
            center_freq_hz: None,
            pace: ReplayPace::Unthrottled,
            loop_replay: false,
        };
        let source =
            FileSource::from_reader(Box::new(std::io::Cursor::new(bytes)), config).unwrap();
        Pipeline::start_gated(
            Box::new(source),
            crate::pipeline::PipelineConfig {
                fft_size: n,
                window: phosphene_core::WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
            gate,
        )
        .unwrap()
    }

    /// A **rate-less** gated replay (clarification C5 — no rate declared,
    /// none invented): the D-054 spectra-cadence seals run on this.
    fn rateless_gated_pipeline(
        n: usize,
        bytes: Vec<u8>,
        gate: std::sync::Arc<std::sync::atomic::AtomicU64>,
    ) -> Pipeline {
        let config = FileSourceConfig::new(
            Input::Path(std::path::PathBuf::from("in-memory.cf32")),
            IqFormat::Cf32,
        );
        let source =
            FileSource::from_reader(Box::new(std::io::Cursor::new(bytes)), config).unwrap();
        Pipeline::start_gated(
            Box::new(source),
            crate::pipeline::PipelineConfig {
                fft_size: n,
                window: phosphene_core::WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
            gate,
        )
        .unwrap()
    }

    fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while !done() {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for {what}"
            );
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// Whether a drained row carries any real data (vs the −200 dB floor).
    fn row_has_data(row: &[f32]) -> bool {
        row.iter().any(|&v| v > -150.0)
    }

    /// Drain the production seam repeatedly until `expect` rows have
    /// accumulated (the compute thread records a batch as completed just
    /// before folding it into the waterfall, so a single drain right after
    /// the completion count settles can land mid-fold — the display, which
    /// drains every frame, never cares). Returns the rows and the last
    /// stated interval; asserts the expected count actually arrived.
    fn collect_rows(
        pipeline: &Pipeline,
        cfg: &crate::pipeline::WfFrameConfig,
        n: usize,
        expect: usize,
    ) -> (Vec<f32>, Option<f32>) {
        let mut all = Vec::new();
        let mut scratch = Vec::new();
        let mut interval = None;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while all.len() / n < expect && std::time::Instant::now() < deadline {
            let drain = pipeline.waterfall_frame(cfg, &mut scratch);
            if drain.rows > 0 {
                interval = drain.row_interval_s;
                all.extend_from_slice(&scratch);
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(
            all.len() / n >= expect,
            "expected at least {expect} waterfall rows from the production \
             seam, got {}",
            all.len() / n
        );
        (all, interval)
    }

    /// After a capture is fully drained, one settled drain must hand over
    /// nothing — the exactness half of the row-count seals.
    fn assert_no_more_rows(pipeline: &Pipeline, cfg: &crate::pipeline::WfFrameConfig, what: &str) {
        std::thread::sleep(std::time::Duration::from_millis(50));
        let mut scratch = Vec::new();
        let drain = pipeline.waterfall_frame(cfg, &mut scratch);
        assert_eq!(drain.rows, 0, "extra rows after {what}");
    }

    /// D-032/D-051 seal, windowed loop, GPU half: the exact per-frame step
    /// `redraw` runs (`waterfall_panel_rows` + `waterfall_frame_step`),
    /// driven against the REAL two-thread pipeline and a real composer,
    /// must land rows in the ring, make `waterfall_row_interval()` state
    /// the real span-derived cadence, and put the capture's tone on the
    /// composited waterfall pixels.
    #[test]
    fn windowed_feed_step_lands_rows_on_the_surface() {
        let (device, queue) = gpu("phosphene-wf-feed-test");
        let mut composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);
        assert_eq!(composer.waterfall_rows_written(), 0);
        assert_eq!(composer.waterfall_row_interval(), None);

        let n = 512usize;
        let batches = 256usize;
        let gate = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut pipeline = rated_gated_pipeline(
            n,
            crate::pipeline::tone_then_silence_bytes(n, batches, 0),
            gate.clone(),
        );

        // The production view mapping, at the UNTOUCHED defaults: fast mode
        // at 1 ms/row (D-044/D-046, owner-ruled) — this seal is the "fast
        // default through the production path" the lane demands, and the
        // 62.5 ms capture yields ~62 rows at that cadence.
        let mut view = ViewState {
            split: 0.0,
            ..Default::default()
        };
        apply_meta(&mut view.params, &pipeline.meta());
        assert_eq!(
            view.wf_mode,
            phosphene_render::WaterfallMode::Fast,
            "D-046: fast must be the default"
        );
        let grid = egui::Rect::from_min_max(egui::pos2(56.0, 39.0), egui::pos2(1264.0, 676.0));
        let panel_rows = waterfall_panel_rows(grid, view.split, 1.0);
        let mut rows = Vec::new();

        // Configure the aggregator (frame 1), let the capture flow, then
        // drain — exactly the steps redraw runs each frame.
        waterfall_frame_step(
            &pipeline,
            &mut composer,
            &device,
            &queue,
            &view,
            panel_rows,
            &mut rows,
        );
        gate.fetch_add(batches as u64, std::sync::atomic::Ordering::Relaxed);
        wait_until("the capture to drain", || {
            pipeline.health().batches_completed == batches as u64
        });
        let drain = waterfall_frame_step(
            &pipeline,
            &mut composer,
            &device,
            &queue,
            &view,
            panel_rows,
            &mut rows,
        );

        assert!(
            drain.rows > 10 && composer.waterfall_rows_written() > 10,
            "the production feed step landed {} rows (D-032/D-051)",
            drain.rows
        );
        let interval = composer
            .waterfall_row_interval()
            .expect("rows flowed, so the display interval must be Some");
        assert_eq!(
            interval, 1e-3,
            "the fast default must run — and state — 1 ms rows (D-046)"
        );

        // And the fed rows are on the composited waterfall: render the same
        // RTSA frame the window renders (split 0) and probe the tone column.
        let (w, h) = (400u32, 300u32);
        let target =
            OffscreenTarget::with_format(&device, w, h, phosphene_render::OFFSCREEN_FORMAT);
        let theme = Theme::default();
        composer.render_rtsa_frame(
            &device,
            &queue,
            &target.view,
            [w, h],
            RtsaInput {
                live_bins: &[],
                max_hold_bins: &[],
                intensity: &[],
                intensity_bins: 0,
                intensity_levels: 0,
                surface_points: egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(w as f32, h as f32),
                ),
                view: &view,
                theme: &theme,
            },
            phosphene_render::EguiFrame {
                primitives: Vec::new(),
                textures_delta: egui::TexturesDelta::default(),
                pixels_per_point: 1.0,
            },
        );
        let rgba = target.read_rgba(&device, &queue);
        let px = |x: u32, y: u32| {
            let o = ((y * w + x) * 4) as usize;
            u32::from(rgba[o]) + u32::from(rgba[o + 1]) + u32::from(rgba[o + 2])
        };
        let bin = phosphene_core::fftshift_index(n / 4, n);
        let hot_x = ((bin as f32 / (n - 1) as f32) * w as f32) as u32;
        assert!(
            px(hot_x, 5) > px(hot_x / 2, 5) + 60,
            "fed rows are not visible on the composited waterfall"
        );
        pipeline.shutdown();
    }

    /// FR-D4 "adjustable time span" end to end: the `-` key is produced by
    /// the REAL chrome, lands in the view, and the SAME production step
    /// redraw runs then states a DIFFERENT row cadence to the display — the
    /// changed cadence is the observable effect (the FR-D5 time labels read
    /// exactly this interval), not the assignment.
    #[test]
    fn span_key_changes_the_row_cadence_through_the_production_feed() {
        let (device, queue) = gpu("phosphene-wf-span-test");
        let mut composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);
        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        let hud = test_hud();

        let n = 512usize;
        let half = 4096usize; // 1 s of signal per half at 2^21 S/s
        let gate = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut pipeline = rated_gated_pipeline(
            n,
            crate::pipeline::tone_then_silence_bytes(n, 2 * half, 0),
            gate.clone(),
        );
        let mut view = ViewState {
            split: 0.0,
            // The subject is traditional's span key; fast is the default.
            wf_mode: phosphene_render::WaterfallMode::Traditional,
            ..Default::default()
        };
        apply_meta(&mut view.params, &pipeline.meta());
        let grid = egui::Rect::from_min_max(egui::pos2(56.0, 39.0), egui::pos2(1264.0, 676.0));
        let panel_rows = waterfall_panel_rows(grid, view.split, 1.0);
        let mut rows = Vec::new();
        assert_eq!(view.wf_time_span_s, 30.0);

        // Default span: configure, feed the first half, drain.
        waterfall_frame_step(
            &pipeline,
            &mut composer,
            &device,
            &queue,
            &view,
            panel_rows,
            &mut rows,
        );
        gate.fetch_add(half as u64, std::sync::atomic::Ordering::Relaxed);
        wait_until("the first half to drain", || {
            pipeline.health().batches_completed == half as u64
        });
        let drain = waterfall_frame_step(
            &pipeline,
            &mut composer,
            &device,
            &queue,
            &view,
            panel_rows,
            &mut rows,
        );
        assert!(drain.rows > 0, "no rows at the default span");
        let before = composer
            .waterfall_row_interval()
            .expect("rows flowed at the default span");
        assert!((before - 30.0 / panel_rows as f32).abs() < 1e-6);

        // Minus through the real chrome shortens the span one step…
        let (_, _) = chrome_frame(&ctx, &theme, &mut view, &hud, Some(egui::Key::Minus));
        let expect_span = 30.0 / phosphene_render::WF_SPAN_STEP;
        assert!(
            (view.wf_time_span_s - expect_span).abs() < 1e-3,
            "the - key did not step the span (got {})",
            view.wf_time_span_s
        );

        // …and the production step now runs — and states — the new cadence.
        waterfall_frame_step(
            &pipeline,
            &mut composer,
            &device,
            &queue,
            &view,
            panel_rows,
            &mut rows,
        );
        gate.fetch_add(half as u64, std::sync::atomic::Ordering::Relaxed);
        wait_until("the second half to drain", || {
            pipeline.health().batches_completed == 2 * half as u64
        });
        let drain = waterfall_frame_step(
            &pipeline,
            &mut composer,
            &device,
            &queue,
            &view,
            panel_rows,
            &mut rows,
        );
        assert!(drain.rows > 0, "no rows at the shortened span");
        let after = composer
            .waterfall_row_interval()
            .expect("rows flowed at the shortened span");
        assert!(
            (after - expect_span / panel_rows as f32).abs() < 1e-6,
            "the stated cadence {after} does not follow the new span"
        );
        assert!(
            after < before,
            "a shorter span must shorten the row interval ({after} vs {before})"
        );
        pipeline.shutdown();
    }

    /// THE D-051 seal: a transient that lives in a spectrum a per-frame
    /// sampler would have **skipped** survives into its row under `Max` —
    /// through the REAL two-thread pipeline, which is the only feed there
    /// is. The capture puts the burst at index 2 of every 4-spectra row
    /// interval: never the spectrum open at a row boundary, never the last
    /// one published before a display frame — the exact shape the old
    /// one-spectrum-per-frame feed could not see. A test that pushes one
    /// spectrum per frame cannot tell the old behaviour from the new, so
    /// this one deliberately is not that shape.
    ///
    /// The A key (REAL chrome) then flips the same capture to `Mean` on a
    /// second pipeline, and the same rows come out diluted — the toggle
    /// reaches the rows through the production seam.
    #[test]
    fn transient_in_a_skipped_spectrum_survives_max_and_dilutes_under_mean() {
        let n = 512usize;
        let (groups, per_group) = (8usize, 4usize);
        let batches = (groups * per_group) as u64;
        let si = wf_test_si(n); // 2^-12, exact
        let bin = phosphene_core::fftshift_index(n / 4, n);

        // Row interval = exactly 4 spectra: panel of 100 rows spanning
        // 100 × 4 × si (all dyadic, so the boundary arithmetic is exact).
        let panel_rows = 100u32;
        let span_s = panel_rows as f32 * per_group as f32 * si;

        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        let hud = test_hud();

        let run_capture = |view: &ViewState| -> (Vec<f32>, Option<f32>) {
            let gate = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            let mut pipeline = rated_gated_pipeline(
                n,
                crate::pipeline::burst_capture_bytes(n, groups, per_group, 2),
                gate.clone(),
            );
            let mut rows = Vec::new();
            // Configure the aggregator BEFORE the spectra flow (the window
            // does this on its first frame), then drain everything.
            pipeline.waterfall_frame(&wf_config_for(view, panel_rows), &mut rows);
            gate.fetch_add(batches, std::sync::atomic::Ordering::Relaxed);
            let out = collect_rows(&pipeline, &wf_config_for(view, panel_rows), n, groups);
            assert_no_more_rows(
                &pipeline,
                &wf_config_for(view, panel_rows),
                "the burst capture",
            );
            pipeline.shutdown();
            out
        };

        // Max (the §7.6 default): every row holds the burst its interval
        // contained, at full strength.
        let mut view_max = ViewState {
            // The exact 4-spectra-per-row arithmetic runs on traditional's
            // span derivation; the burst-survival property itself is
            // mode-independent (one aggregator serves both).
            wf_mode: phosphene_render::WaterfallMode::Traditional,
            ..Default::default()
        };
        view_max.params.sample_rate = Some(WF_TEST_RATE);
        view_max.wf_time_span_s = span_s;
        assert_eq!(view_max.wf_aggregation, RowAggregation::Max);
        let (rows, interval) = run_capture(&view_max);
        assert_eq!(rows.len() / n, groups, "one row per 4-spectra group");
        assert_eq!(interval, Some(per_group as f32 * si));
        for (i, row) in rows.chunks_exact(n).enumerate() {
            assert!(
                (row[bin] - -20.0).abs() < 2.0,
                "row {i}: the burst a frame-sampler would have skipped did \
                 not survive max aggregation (bin {bin} = {} dBFS, §7.6/D-051)",
                row[bin]
            );
            assert!(
                row[(bin + 40) % n] < -150.0,
                "row {i}: quiet bins must stay at the floor"
            );
        }

        // Mean, switched by the REAL chrome's A key: the same burst is
        // averaged over its 4-spectra interval — visible as a value far
        // below the max rows, exactly as §7.6 defines the toggle.
        let mut view_mean = view_max;
        let (_, _) = chrome_frame(&ctx, &theme, &mut view_mean, &hud, Some(egui::Key::A));
        assert_eq!(view_mean.wf_aggregation, RowAggregation::Mean);
        let (rows, _) = run_capture(&view_mean);
        for (i, row) in rows.chunks_exact(n).enumerate() {
            assert!(
                row[bin] < -100.0 && row[bin] > -180.0,
                "row {i}: mean aggregation should dilute the 1-in-4 burst \
                 (bin {bin} = {} dBFS)",
                row[bin]
            );
        }
    }

    /// The stripes seal (D-051, the owner's report): at a row interval far
    /// below the display frame interval, a continuously-producing source
    /// yields **no floor-filled row** — under the old one-spectrum-per-frame
    /// feed every interval past the first per frame was an empty floor row,
    /// which is exactly the black-space defect. Fast mode at its 1 ms
    /// default (D-046) against 16.7 ms display frames is the reported case;
    /// the F key that selects it is produced by the REAL chrome, so the
    /// keymap-table binding is the wiring under test too.
    #[test]
    fn fast_mode_1ms_rows_emit_no_floor_rows_while_the_source_produces() {
        let n = 512usize;
        let batches = 4096u64; // 1 s of signal
        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        let hud = test_hud();

        let gate = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut pipeline = rated_gated_pipeline(
            n,
            crate::pipeline::tone_then_silence_bytes(n, batches as usize, 0),
            gate.clone(),
        );
        let mut view = ViewState::default();
        apply_meta(&mut view.params, &pipeline.meta());
        // Fast is the default (D-044/D-046); F through the real chrome —
        // the mode toggle is in the one keymap table (its `?` overlay is
        // generated from it), no second key path — round-trips it.
        assert_eq!(view.wf_mode, phosphene_render::WaterfallMode::Fast);
        let (_, _) = chrome_frame(&ctx, &theme, &mut view, &hud, Some(egui::Key::F));
        assert_eq!(
            view.wf_mode,
            phosphene_render::WaterfallMode::Traditional,
            "the F key must switch the speed mode"
        );
        let (_, _) = chrome_frame(&ctx, &theme, &mut view, &hud, Some(egui::Key::F));
        assert_eq!(view.wf_mode, phosphene_render::WaterfallMode::Fast);
        // And `-` now adjusts the quantity fast mode owns — the row
        // interval — leaving the traditional span untouched (D-046).
        let (_, _) = chrome_frame(&ctx, &theme, &mut view, &hud, Some(egui::Key::Minus));
        assert!(
            (view.wf_fast_interval_s - 1e-3 / phosphene_render::WF_SPAN_STEP).abs() < 1e-7,
            "the - key did not step the fast interval (got {})",
            view.wf_fast_interval_s
        );
        assert_eq!(
            view.wf_time_span_s, 30.0,
            "fast keys must not move the span"
        );
        view.wf_fast_interval_s = 1e-3; // back to the D-046 default

        let panel_rows = 600u32;
        let mut rows = Vec::new();
        pipeline.waterfall_frame(&wf_config_for(&view, panel_rows), &mut rows);
        gate.fetch_add(batches, std::sync::atomic::Ordering::Relaxed);
        wait_until("the capture to drain", || {
            pipeline.health().batches_completed == batches
        });
        // 1 s of signal at 1 ms rows: the elapsed accumulator carries a
        // sub-interval remainder, so expect within one row of 999.
        let (all_rows, interval) =
            collect_rows(&pipeline, &wf_config_for(&view, panel_rows), n, 999);
        let total = all_rows.len() / n;
        assert_eq!(interval, Some(1e-3), "fast mode must state its interval");
        for (i, row) in all_rows.chunks_exact(n).enumerate() {
            assert!(
                row_has_data(row),
                "row {i} of {total} is a floor row while the source was \
                 producing — the black-stripe defect (D-051)"
            );
        }
        pipeline.shutdown();

        // The D-046 clamp, same production seam: asking for rows finer than
        // one spectrum interval pins the cadence to the spectrum interval —
        // one row per FFT, stated exactly.
        let batches = 64u64;
        let gate = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut pipeline = rated_gated_pipeline(
            n,
            crate::pipeline::tone_then_silence_bytes(n, batches as usize, 0),
            gate.clone(),
        );
        view.wf_fast_interval_s = 1e-5;
        let mut rows = Vec::new();
        pipeline.waterfall_frame(&wf_config_for(&view, panel_rows), &mut rows);
        gate.fetch_add(batches, std::sync::atomic::Ordering::Relaxed);
        wait_until("the clamp capture to drain", || {
            pipeline.health().batches_completed == batches
        });
        // At the clamp, every spectrum is its own row, and the stated
        // interval pins to one spectrum exactly.
        let (clamp_rows, interval) = collect_rows(
            &pipeline,
            &wf_config_for(&view, panel_rows),
            n,
            batches as usize,
        );
        assert_eq!(
            clamp_rows.len() / n,
            batches as usize,
            "at the clamp, every spectrum is its own row"
        );
        assert_no_more_rows(
            &pipeline,
            &wf_config_for(&view, panel_rows),
            "the clamp capture",
        );
        assert_eq!(
            interval,
            Some(wf_test_si(n)),
            "the clamp must pin the stated interval to one spectrum"
        );
        pipeline.shutdown();
    }

    /// D-054: a rate-less source measures the waterfall in spectra. Fast
    /// mode runs one row per spectrum and the composer's stated seconds
    /// interval stays `None` — the FR-D5 time labels never invent a unit —
    /// and traditional mode spans a row count.
    #[test]
    fn rateless_source_measures_the_waterfall_in_spectra() {
        let (device, queue) = gpu("phosphene-wf-rateless-test");
        let mut composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);
        let n = 512usize;
        let batches = 32u64;

        // Fast: one row per spectrum, through the full production step.
        let gate = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut pipeline = rateless_gated_pipeline(
            n,
            crate::pipeline::tone_then_silence_bytes(n, batches as usize, 0),
            gate.clone(),
        );
        let mut view = ViewState {
            wf_mode: phosphene_render::WaterfallMode::Fast,
            ..Default::default()
        };
        apply_meta(&mut view.params, &pipeline.meta());
        assert_eq!(view.params.sample_rate, None, "test premise: rate-less");
        let mut rows = Vec::new();
        waterfall_frame_step(
            &pipeline,
            &mut composer,
            &device,
            &queue,
            &view,
            300,
            &mut rows,
        );
        gate.fetch_add(batches, std::sync::atomic::Ordering::Relaxed);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut last_interval = Some(0.0);
        while composer.waterfall_rows_written() < batches as u32
            && std::time::Instant::now() < deadline
        {
            let drain = waterfall_frame_step(
                &pipeline,
                &mut composer,
                &device,
                &queue,
                &view,
                300,
                &mut rows,
            );
            last_interval = drain.row_interval_s;
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(
            composer.waterfall_rows_written(),
            batches as u32,
            "fast + rate-less is one row per spectrum (D-054)"
        );
        assert_eq!(last_interval, None);
        assert_eq!(
            composer.waterfall_row_interval(),
            None,
            "a rate-less waterfall must never state a seconds interval (D-054)"
        );
        assert_eq!(
            composer.waterfall_row_spectra(),
            Some(1.0),
            "fast + rate-less rows carry a one-spectrum cadence (D-054)"
        );

        // And the FR-D5 axis renders in the row unit, not blank and never
        // seconds: mirror both time-base fields from the composer exactly
        // as the frame loops do, run the real chrome, and read the labels.
        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        view.split = 0.0; // waterfall fully open, labels tall enough
        view.params.wf_row_interval_s = composer.waterfall_row_interval();
        view.params.wf_row_spectra = composer.waterfall_row_spectra();
        let (texts, _) = chrome_frame(&ctx, &theme, &mut view, &test_hud_rateless(), None);
        let row_label = |t: &String| {
            t.starts_with('-') && t.ends_with('r') && t[1..t.len() - 1].parse::<u32>().is_ok()
        };
        assert!(
            texts.iter().any(row_label),
            "the rate-less waterfall axis must be labelled in rows of \
             spectra (D-054); texts: {texts:?}"
        );
        let seconds_label = |t: &String| {
            t.starts_with('-')
                && (t.ends_with('s') || t.ends_with("ms"))
                && t[1..].chars().next().is_some_and(|c| c.is_ascii_digit())
        };
        assert!(
            !texts.iter().any(seconds_label),
            "a rate-less waterfall rendered a seconds label (D-054); \
             texts: {texts:?}"
        );
        assert_eq!(
            composer.waterfall_row_interval(),
            None,
            "a rate-less waterfall must never state a seconds interval — \
             the FR-D5 labels would otherwise lie in a unit the source does \
             not have (D-054)"
        );
        pipeline.shutdown();

        // Traditional: the span is a row count — 40 rows of spectra over a
        // 10-row panel is exactly 4 spectra per row.
        let gate = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut pipeline = rateless_gated_pipeline(
            n,
            crate::pipeline::tone_then_silence_bytes(n, batches as usize, 0),
            gate.clone(),
        );
        let mut view = ViewState {
            wf_mode: phosphene_render::WaterfallMode::Traditional,
            ..Default::default()
        };
        apply_meta(&mut view.params, &pipeline.meta());
        view.wf_span_rows = 40.0;
        let mut rows = Vec::new();
        pipeline.waterfall_frame(&wf_config_for(&view, 10), &mut rows);
        gate.fetch_add(batches, std::sync::atomic::Ordering::Relaxed);
        let (trad_rows, interval) = collect_rows(
            &pipeline,
            &wf_config_for(&view, 10),
            n,
            batches as usize / 4,
        );
        assert_eq!(
            trad_rows.len() / n,
            batches as usize / 4,
            "a 40-row span over a 10-row panel aggregates 4 spectra per row"
        );
        assert_no_more_rows(
            &pipeline,
            &wf_config_for(&view, 10),
            "the rate-less capture",
        );
        assert_eq!(interval, None);
        pipeline.shutdown();
    }

    /// FR-D4/§7.6 independent auto-range end to end: the W key is produced
    /// by the REAL chrome, the same per-frame mapping step redraw runs
    /// hands it to the composer, and the SAME retained rows then render
    /// through a DIFFERENT power→color mapping — asserted on the pixels,
    /// not on the flag.
    #[test]
    fn auto_range_key_changes_the_waterfall_mapping_through_the_production_path() {
        let (device, queue) = gpu("phosphene-wf-auto-range-test");
        let mut composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);
        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        let hud = test_hud();
        let mut view = ViewState {
            split: 0.0,
            ..Default::default()
        };

        // Rows spanning a narrow 15 dB band well below the displayed range's
        // top: under the shared −100..0 mapping the −80 dBFS peak sits low
        // in the ramp; under auto-range it IS the top of the ramp. Retained
        // rows enter through the same batched seam the frame step uses.
        let n = 512usize;
        let mut spectrum = vec![-95.0f32; n];
        spectrum[100] = -80.0;
        let mut batch = Vec::new();
        for _ in 0..120 {
            batch.extend_from_slice(&spectrum);
        }
        composer.push_waterfall_rows(
            &device,
            &queue,
            &batch,
            n,
            Some(phosphene_render::RowCadence::seconds_per_row(0.05)),
        );
        assert!(composer.waterfall_rows_written() > 0);

        let (w, h) = (400u32, 300u32);
        let target =
            OffscreenTarget::with_format(&device, w, h, phosphene_render::OFFSCREEN_FORMAT);
        let render = |composer: &mut FrameComposer, view: &ViewState| -> Vec<u8> {
            // The production mapping step redraw runs every frame, verbatim.
            composer.set_waterfall_auto_range(view.wf_auto_range);
            composer.render_rtsa_frame(
                &device,
                &queue,
                &target.view,
                [w, h],
                RtsaInput {
                    live_bins: &[],
                    max_hold_bins: &[],
                    intensity: &[],
                    intensity_bins: 0,
                    intensity_levels: 0,
                    surface_points: egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(w as f32, h as f32),
                    ),
                    view,
                    theme: &theme,
                },
                phosphene_render::EguiFrame {
                    primitives: Vec::new(),
                    textures_delta: egui::TexturesDelta::default(),
                    pixels_per_point: 1.0,
                },
            );
            target.read_rgba(&device, &queue)
        };
        let px = |rgba: &[u8], x: u32, y: u32| {
            let o = ((y * w + x) * 4) as usize;
            u32::from(rgba[o]) + u32::from(rgba[o + 1]) + u32::from(rgba[o + 2])
        };
        let hot_x = ((100.0 / (n - 1) as f32) * w as f32) as u32;

        // **The D-065 §2 default, on the pixels and on the chrome.**
        // Auto-range is what a first-run user gets, and the `wf` readout
        // says `auto` while it is — the honesty condition that makes the
        // new default acceptable, since an auto-ranged waterfall no longer
        // agrees with the dB axis above it.
        assert!(
            view.wf_auto_range,
            "D-065 §2: auto-range is the shipped default"
        );
        let (texts, _) = chrome_frame(&ctx, &theme, &mut view, &hud, None);
        assert!(
            texts.iter().any(|t| t.contains("AUTO")),
            "the default auto-range must be announced on the wf readout; texts: {texts:?}"
        );
        let on = px(&render(&mut composer, &view), hot_x, 5);

        // W through the real chrome turns it OFF: the same retained rows
        // re-render through the shared −100..0 mapping and the peak column
        // goes visibly dimmer at the same pixel.
        let (texts, _) = chrome_frame(&ctx, &theme, &mut view, &hud, Some(egui::Key::W));
        assert!(!view.wf_auto_range, "the W key must toggle the auto-range");
        assert!(
            !texts.iter().any(|t| t.contains("AUTO")),
            "the readout still says auto with auto-range off; texts: {texts:?}"
        );
        let off = px(&render(&mut composer, &view), hot_x, 5);
        assert!(
            on > off + 60,
            "auto-range did not change the mapping (on {on} vs off {off})"
        );

        // And back on again restores the auto mapping.
        let (_, _) = chrome_frame(&ctx, &theme, &mut view, &hud, Some(egui::Key::W));
        assert!(view.wf_auto_range);
        let on_again = px(&render(&mut composer, &view), hot_x, 5);
        assert!(on_again > off, "toggling back did not restore the mapping");
    }
    /// **D-065 §3, end to end on the production path.**
    ///
    /// The §7.6 waterfall intensity control is new, so the thing that must be
    /// proven is not that the arithmetic works but that pressing the control
    /// changes what the waterfall looks like — the D-031 shape, and the
    /// reason the owner's report existed at all: `PgUp`/`PgDn` moved a real
    /// number that no waterfall pixel ever read.
    ///
    /// Driven three ways on one composer, in this order:
    ///
    /// 1. **the bare `PgUp`** — §7.3's histogram gamma — must leave the
    ///    waterfall's pixels **exactly** where they were. That is the owner's
    ///    finding, asserted rather than assumed;
    /// 2. **`Shift+PgUp`** through the real chrome, then the production
    ///    mirror `redraw` runs, must make the same retained rows render
    ///    **brighter** (γ_wf below 1 lifts low-power detail);
    /// 3. **a click on the real button** the chrome drew must reach the same
    ///    exponent by the same step.
    ///
    /// And it runs with **auto-range on** — the D-065 §2 default — because
    /// that is the mode the control has to work in: if `γ_wf` were consumed
    /// by the range mode rather than composed with it, this test would show
    /// no change at all.
    #[test]
    fn waterfall_intensity_reaches_the_pixels_through_the_production_path() {
        let (device, queue) = gpu("phosphene-wf-gamma-test");
        let mut composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);
        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        let hud = test_hud();
        let mut view = ViewState {
            split: 0.0,
            ..Default::default()
        };
        assert!(
            view.wf_auto_range,
            "the control must work in the default mode"
        );
        assert_eq!(view.wf_gamma, phosphene_render::DEFAULT_WF_GAMMA);

        // Rows with a spread of powers, so a power law has something to
        // reshape: a floor, a mid-level shoulder and a peak.
        let n = 512usize;
        let mut spectrum = vec![-95.0f32; n];
        for (i, v) in spectrum.iter_mut().enumerate() {
            if (200..300).contains(&i) {
                *v = -60.0;
            }
        }
        spectrum[100] = -20.0;
        let mut batch = Vec::new();
        for _ in 0..120 {
            batch.extend_from_slice(&spectrum);
        }
        composer.push_waterfall_rows(
            &device,
            &queue,
            &batch,
            n,
            Some(phosphene_render::RowCadence::seconds_per_row(0.05)),
        );
        assert!(composer.waterfall_rows_written() > 0);

        let (w, h) = (400u32, 300u32);
        let target =
            OffscreenTarget::with_format(&device, w, h, phosphene_render::OFFSCREEN_FORMAT);
        let render = |composer: &mut FrameComposer, view: &ViewState| -> Vec<u8> {
            // Verbatim the two mapping lines `redraw` runs every frame.
            composer.set_waterfall_auto_range(view.wf_auto_range);
            composer.set_waterfall_gamma(view.wf_gamma);
            composer.render_rtsa_frame(
                &device,
                &queue,
                &target.view,
                [w, h],
                RtsaInput {
                    live_bins: &[],
                    max_hold_bins: &[],
                    intensity: &[],
                    intensity_bins: 0,
                    intensity_levels: 0,
                    surface_points: egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(w as f32, h as f32),
                    ),
                    view,
                    theme: &theme,
                },
                phosphene_render::EguiFrame {
                    primitives: Vec::new(),
                    textures_delta: egui::TexturesDelta::default(),
                    pixels_per_point: 1.0,
                },
            );
            target.read_rgba(&device, &queue)
        };
        // The shoulder column: mid-scale under the auto-ranged mapping, so a
        // power law moves it. (The peak and the floor are the endpoints of
        // `t`, which `t^γ` fixes at 0 and 1 — pinned below.)
        let px = |rgba: &[u8], x: u32, y: u32| {
            let o = ((y * w + x) * 4) as usize;
            u32::from(rgba[o]) + u32::from(rgba[o + 1]) + u32::from(rgba[o + 2])
        };
        let shoulder_x = ((250.0 / (n - 1) as f32) * w as f32) as u32;
        let shoulder_feed = phosphene_render::AnnotationFeed::new();
        let mut shoulder_overlay_state = phosphene_render::AnnotationOverlayState::new();

        let base = render(&mut composer, &view);
        let before = px(&base, shoulder_x, 5);

        // 1. The bare PgUp is §7.3's control: it must not touch these pixels.
        let (_, response) = chrome_frame(&ctx, &theme, &mut view, &hud, Some(egui::Key::PageUp));
        assert_eq!(
            response.requests.gamma_steps, -1,
            "the bare PgUp must still be the histogram's control"
        );
        composer.apply_requests(&response.requests);
        assert_eq!(view.wf_gamma, phosphene_render::DEFAULT_WF_GAMMA);
        let histogram_only = render(&mut composer, &view);
        assert_eq!(
            px(&histogram_only, shoulder_x, 5),
            before,
            "§7.3's gamma moved a waterfall pixel - the two controls are crossed"
        );

        // 2. Shift+PgUp, through the real chrome, several steps so the
        //    change is far outside rasteriser noise.
        for _ in 0..6 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 720.0),
                )),
                events: vec![egui::Event::Key {
                    key: egui::Key::PageUp,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::SHIFT,
                }],
                ..Default::default()
            };
            let out = ctx.run_ui(input, |ui| {
                phosphene_render::chrome::draw(
                    ui,
                    &theme,
                    &mut view,
                    &hud,
                    &shoulder_feed,
                    &mut shoulder_overlay_state,
                );
            });
            out.drop_without_applying_deltas();
        }
        assert!(
            view.wf_gamma < phosphene_render::DEFAULT_WF_GAMMA,
            "shift+PgUp did not lower the exponent (now {})",
            view.wf_gamma
        );
        let lifted = render(&mut composer, &view);
        let after = px(&lifted, shoulder_x, 5);
        assert!(
            after > before + 30,
            "the intensity control did not reach the waterfall's pixels \
             (before {before}, after {after}) - which is precisely the defect \
             D-065 §3 reported against §7.3's gamma"
        );
        // The composer holds what the view asked for: the mirror ran.
        assert!((composer.waterfall_gamma() - view.wf_gamma).abs() < 1e-6);

        // 3. The real button, clicked with real pointer input, moves the same
        //    exponent by the same step as the key.
        let before_click = view.wf_gamma;
        let rect = phosphene_render::chrome::wf_intensity_step_rect(&ctx, false)
            .expect("the intensity buttons must be drawn");
        let at = rect.center();
        for pressed in [true, false] {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 720.0),
                )),
                events: vec![
                    egui::Event::PointerMoved(at),
                    egui::Event::PointerButton {
                        pos: at,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::default(),
                    },
                ],
                ..Default::default()
            };
            let out = ctx.run_ui(input, |ui| {
                phosphene_render::chrome::draw(
                    ui,
                    &theme,
                    &mut view,
                    &hud,
                    &shoulder_feed,
                    &mut shoulder_overlay_state,
                );
            });
            out.drop_without_applying_deltas();
        }
        assert!(
            (view.wf_gamma - before_click * phosphene_render::keymap::GAMMA_STEP).abs() < 1e-6,
            "the down button did not step the exponent (was {before_click}, now {})",
            view.wf_gamma
        );
        let dimmed = render(&mut composer, &view);
        assert!(
            px(&dimmed, shoulder_x, 5) < after,
            "the button's step did not reach the pixels"
        );
    }

    /// **D-031's residue on the new control, closed directly.**
    ///
    /// `waterfall_intensity_reaches_the_pixels_through_the_production_path`
    /// runs the mapping step verbatim rather than calling `redraw` (which
    /// needs a window and a swapchain), so it proves the step works and not
    /// that the frame loops call it — the exact gap D-031 was written about.
    /// D-031's remedy for that case is to assert the wiring directly, so:
    /// **both** production frame loops must mirror the exponent onto the
    /// composer, in their production halves, not inside a `#[cfg(test)]`
    /// module.
    #[test]
    fn both_frame_loops_mirror_the_waterfall_intensity_onto_the_composer() {
        for file in ["src/window.rs", "src/headless.rs"] {
            let text = std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(file),
            )
            .unwrap_or_else(|e| panic!("cannot read {file}: {e}"));
            let production = text
                .split("#[cfg(test)]")
                .next()
                .unwrap_or_else(|| panic!("{file} has a production half"));
            assert!(
                production.contains("set_waterfall_gamma("),
                "{file}'s frame loop does not mirror ViewState::wf_gamma onto \
                 the composer: the control would move a number no pixel reads, \
                 which is the defect D-065 §3 reported (D-031)"
            );
            // The auto-range mirror is its neighbour and its precedent; if
            // one is dropped the other should not silently carry the test.
            assert!(
                production.contains("set_waterfall_auto_range("),
                "{file}'s frame loop no longer mirrors the auto-range either"
            );
        }
    }

    /// The colormap readout (M1-J closes AUDIT-1's minor row): the C key is
    /// produced by the REAL chrome, consumed by the same `apply_requests`
    /// redraw calls on a real composer, and the composer's OWN selection —
    /// the value redraw hands to `HudStats` — then renders on the status
    /// bar. The only line not under test is redraw's own field assignment,
    /// the same residue as the screenshot seal.
    #[test]
    fn colormap_cycle_reaches_the_status_readout() {
        let (device, _queue) = gpu("phosphene-colormap-readout-test");
        let mut composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);
        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        let mut view = ViewState::default();
        assert_eq!(composer.colormap().name(), "p7", "D-010: p7 is default");

        // Produce: press C through the real chrome. Consume: the same
        // composer call redraw makes.
        let (_, response) = chrome_frame(&ctx, &theme, &mut view, &test_hud(), Some(egui::Key::C));
        assert!(response.requests.cycle_colormap);
        composer.apply_requests(&response.requests);
        assert_eq!(composer.colormap().name(), "inferno");

        // The readout: the next chrome pass renders the composer's own name.
        let mut hud = test_hud();
        hud.colormap = composer.colormap().name();
        let (texts, _) = chrome_frame(&ctx, &theme, &mut view, &hud, None);
        assert!(
            texts.iter().any(|t| t == "INFERNO"),
            "the active colormap name is not on the status bar; texts: {texts:?}"
        );
    }

    /// Fix pass 2, blocker 4: `--colormap` used to be parsed by
    /// `parse_colormap` and then never reach `window::run` at all — the
    /// windowed path always opened on P7 regardless of the flag. This
    /// proves both ends of the fix: `parse_colormap` accepts every name
    /// [`phosphene_render::Colormap::ALL`] answers to (and names all of them
    /// on a bad value, per §8.4), and a composer built and seeded the exact
    /// way `WindowState::new` now does — `set_colormap` right after
    /// `FrameComposer::new`, before anything else touches it — actually
    /// starts on the requested map instead of the `P7` default. The OS
    /// window/surface half of `WindowState::new` itself is untested here
    /// (no test in this file creates a real `EventLoop`/window, the same
    /// residue `colormap_cycle_reaches_the_status_readout`'s own doc
    /// comment notes): what's left unverified is one straight-line field
    /// pass from `App.colormap` into that call, not any logic.
    #[test]
    fn colormap_flag_is_parsed_and_actually_seeds_the_composer() {
        for map in phosphene_render::Colormap::ALL {
            assert_eq!(
                crate::parse_colormap(map.name()).unwrap(),
                map,
                "parse_colormap must round-trip every name Colormap::ALL answers to"
            );
        }
        let err = crate::parse_colormap("nonexistent").unwrap_err();
        for map in phosphene_render::Colormap::ALL {
            assert!(
                err.contains(map.name()),
                "a bad --colormap value must name every valid choice; missing {:?} in {err:?}",
                map.name()
            );
        }

        let (device, _queue) = gpu("phosphene-colormap-flag-test");
        for map in phosphene_render::Colormap::ALL {
            let mut composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);
            // The exact order `WindowState::new` now uses: seed immediately
            // after construction, before the composer renders anything.
            composer.set_colormap(map);
            assert_eq!(
                composer.colormap(),
                map,
                "the composer must start on the --colormap the CLI asked for"
            );
        }
    }

    /// D-033 seal, producer side, windowed loop: the exact per-frame feed
    /// steps `redraw` runs — `copy_trace`, `intensity_frame`, and the
    /// `RtsaInput` handoff — driven against the
    /// REAL two-thread pipeline over the demo scene and a real composer,
    /// must hand a non-empty `bins × levels` grid to `RtsaInput` and light
    /// the histogram surface where the −20 dBFS tone sits. A test that
    /// builds its own `PersistenceHistogram` is the shape that passed while
    /// FR-D1 was dead (D-033); this drives the production seam instead.
    ///
    /// `live_bins` is deliberately empty in the render call: a steady tone's
    /// live trace draws on exactly the pixels its persistence cell occupies,
    /// so including it would keep this probe green with a dead intensity
    /// feed. The data path under test is unchanged.
    #[test]
    fn windowed_feed_lights_the_histogram_surface() {
        use std::time::Duration;

        use phosphene_sources::{SigGen, SigGenConfig};

        let n = 1024usize;
        let source = SigGen::new(SigGenConfig::demo()).unwrap();
        let mut pipeline = Pipeline::start(
            Box::new(source),
            crate::pipeline::PipelineConfig {
                fft_size: n,
                window: phosphene_core::WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .expect("no GPU adapter — the D-033 seal cannot run");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("phosphene-persistence-feed-test"),
            ..Default::default()
        }))
        .expect("GPU device request failed");
        let mut composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);
        let (w, h) = (640u32, 360u32);
        let target =
            OffscreenTarget::with_format(&device, w, h, phosphene_render::OFFSCREEN_FORMAT);
        let surface_rect =
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32));
        // Probe geometry: histogram-only, parameterised explicitly like the
        // render-crate goldens — the D-040 default-view seal lives in the
        // headless tests; this test's subject is the intensity feed.
        let view = ViewState {
            split: 1.0,
            ..Default::default()
        };
        let theme = Theme::default();

        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while pipeline.health().batches_completed < 32 {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for the first batches"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        // A second of display frames, exactly the steps redraw runs, with
        // spectra flowing between ticks.
        let mut bins = Vec::new();
        let mut intensity = Vec::new();
        let mut dims = (0usize, 0usize);
        for _ in 0..60 {
            std::thread::sleep(Duration::from_millis(8));
            pipeline.copy_trace(&mut bins);
            dims = pipeline.intensity_frame(
                1.0 / 60.0,
                view.params.db_bottom,
                view.params.db_top,
                &mut intensity,
            );
            composer.render_rtsa_frame(
                &device,
                &queue,
                &target.view,
                [w, h],
                RtsaInput {
                    live_bins: &[],
                    max_hold_bins: &[],
                    intensity: &intensity,
                    intensity_bins: dims.0,
                    intensity_levels: dims.1,
                    surface_points: surface_rect,
                    view: &view,
                    theme: &theme,
                },
                phosphene_render::EguiFrame {
                    primitives: Vec::new(),
                    textures_delta: egui::TexturesDelta::default(),
                    pixels_per_point: 1.0,
                },
            );
        }
        assert_eq!(dims, (n, 128), "the grid must be bins × levels");
        assert!(
            intensity.iter().any(|&v| v > 0.0),
            "an all-zero grid reached RtsaInput (D-033)"
        );

        // The +300 kHz tone at 2.048 MS/s: raw bin 150, fftshifted to the
        // upper half. Probe its cell (−20 dBFS) against the same height in
        // a column nothing occupies — the −20 dB horizontal gridline runs
        // through both, so it cancels out of the comparison.
        let rgba = target.read_rgba(&device, &queue);
        let px = |x: u32, y: u32| {
            let o = ((y * w + x) * 4) as usize;
            u32::from(rgba[o]) + u32::from(rgba[o + 1]) + u32::from(rgba[o + 2])
        };
        let probe = |fx: f32, db: f32| {
            let x0 = fx * (w - 1) as f32;
            let y0 = (1.0 - (db + 110.0) / 110.0) * (h - 1) as f32;
            let mut max = 0u32;
            for dy in -5i32..=5 {
                for dx in -3i32..=3 {
                    let x = (x0 as i32 + dx).clamp(0, w as i32 - 1) as u32;
                    let y = (y0 as i32 + dy).clamp(0, h as i32 - 1) as u32;
                    max = max.max(px(x, y));
                }
            }
            max
        };
        let tone_fx = phosphene_core::fftshift_index(150, n) as f32 / (n - 1) as f32;
        let lit = probe(tone_fx, -20.0);
        let quiet = probe(0.85, -20.0);
        assert!(
            lit > quiet + 60,
            "the fed grid is not visible on the histogram surface \
             (tone {lit} vs quiet {quiet})"
        );
        pipeline.shutdown();
    }

    /// FR-D3 end to end through the windowed frame steps (D-031, D-034):
    /// the three FR-D7 keys are produced by the REAL chrome from real
    /// keypresses, consumed by the same `max_hold_frame_step` redraw runs,
    /// against the REAL two-thread pipeline — and the resulting trace
    /// reaches `RtsaInput` and is DRAWN. Over a tone-then-silence capture
    /// the max-hold line is visible where the live trace has already fallen
    /// away; D switches to pure hold (no decay through the production
    /// step); M hides the line; R clears the trace. The pixel probes are
    /// perceptual in the D-014 sense (brightness against a matched quiet
    /// region, wide tolerance), never byte-exact.
    #[test]
    fn max_hold_keys_drive_the_production_frame_step_and_the_line_is_drawn() {
        use std::time::{Duration, Instant};

        use phosphene_render::ChromeRequests;
        use phosphene_sources::{FileSource, FileSourceConfig, Input, IqFormat};

        let n = 512usize;
        let (tone_batches, silent_batches) = (100usize, 400usize);
        let bytes = crate::pipeline::tone_then_silence_bytes(n, tone_batches, silent_batches);
        let config = FileSourceConfig::new(
            Input::Path(std::path::PathBuf::from("in-memory.cf32")),
            IqFormat::Cf32,
        );
        let source =
            FileSource::from_reader(Box::new(std::io::Cursor::new(bytes)), config).unwrap();
        let mut pipeline = Pipeline::start(
            Box::new(source),
            crate::pipeline::PipelineConfig {
                fft_size: n,
                window: phosphene_core::WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        while pipeline.health().batches_completed < (tone_batches + silent_batches) as u64 {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for the capture to drain"
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        // The frame's live trace, exactly as redraw copies it: after the
        // silence it has fallen to the floor at the tone bin.
        let mut live = Vec::new();
        pipeline.copy_trace(&mut live);
        let bin = phosphene_core::fftshift_index(n / 4, n);
        assert!(
            live[bin] < -150.0,
            "test premise: the live trace has fallen away, got {}",
            live[bin]
        );

        // The real chrome, driven exactly as redraw drives it. Probe
        // geometry: histogram-only, parameterised explicitly (this test's
        // subject is FR-D3; the D-040 default-view seal is headless).
        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        let mut view = ViewState {
            split: 1.0,
            ..Default::default()
        };
        view.params.sample_rate = Some(2_048_000.0);
        let hud = HudStats {
            fps: 60.0,
            frame_ms: 2.5,
            fft_size: n,
            window_name: "HANN",
            source_label: "TEST".into(),
            backend: pipeline.backend(),
            colormap: "p7",
            input_rate_sps: 0.0,
            processed_pct: 100.0,
            ffts_per_s: 0.0,
            dropped_batches: 0,
            persistence_tau_s: phosphene_core::DEFAULT_TAU_DECAY,
            gamma: phosphene_render::DEFAULT_GAMMA,
            live_tau_s: phosphene_core::DEFAULT_TAU_LIVE,
            max_hold_tau_s: phosphene_core::DEFAULT_TAU_MAX_HOLD,
            overflow_events: 0,
            ..Default::default()
        };
        let chrome_feed = phosphene_render::AnnotationFeed::new();
        let mut chrome_overlay_state = phosphene_render::AnnotationOverlayState::new();
        let mut chrome = |view: &mut ViewState, key: Option<egui::Key>| {
            let events = key
                .map(|key| {
                    vec![egui::Event::Key {
                        key,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::default(),
                    }]
                })
                .unwrap_or_default();
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 720.0),
                )),
                events,
                ..Default::default()
            };
            let mut response = None;
            let out = ctx.run_ui(input, |ui| {
                response = Some(phosphene_render::chrome::draw(
                    ui,
                    &theme,
                    view,
                    &hud,
                    &chrome_feed,
                    &mut chrome_overlay_state,
                ));
            });
            out.drop_without_applying_deltas();
            response.unwrap()
        };

        // Default view (shown, decay-toward-live): a second of production
        // frame steps decays the retained −20 dBFS peak toward the floor —
        // exp(−1/2) of the gap remains, ≈ −90 dBFS.
        let response = chrome(&mut view, None);
        assert!(view.max_hold && view.max_hold_decay);
        let mut buf = Vec::new();
        for _ in 0..4 {
            let s =
                max_hold_frame_step(&pipeline, &view, &response.requests, 0.25, &live, &mut buf);
            assert!(!s.is_empty(), "the default view must show the trace");
        }
        assert!(
            (-95.0..=-84.0).contains(&buf[bin]),
            "1 s of decay-toward-live should leave ≈ −90 dBFS, got {}",
            buf[bin]
        );

        // D through the real chrome: pure hold — no decay through the
        // production step, tick after tick.
        let response = chrome(&mut view, Some(egui::Key::D));
        assert!(!view.max_hold_decay, "D must switch the D-007 mode");
        max_hold_frame_step(&pipeline, &view, &response.requests, 0.25, &live, &mut buf);
        let v_hold = buf[bin];
        let shown = max_hold_frame_step(
            &pipeline,
            &view,
            &ChromeRequests::default(),
            0.25,
            &live,
            &mut buf,
        )
        .to_vec();
        assert_eq!(
            shown[bin], v_hold,
            "pure hold decayed through the frame step"
        );

        // Drawn: render the same RtsaInput handoff redraw makes and probe
        // the line where the live trace is at the floor.
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .expect("no GPU adapter — the FR-D3 seal cannot run");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("phosphene-max-hold-test"),
            ..Default::default()
        }))
        .expect("GPU device request failed");
        let mut composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);
        let (w, h) = (400u32, 300u32);
        let target =
            OffscreenTarget::with_format(&device, w, h, phosphene_render::OFFSCREEN_FORMAT);
        let surface_rect =
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32));
        let render = |composer: &mut FrameComposer, view: &ViewState, max_hold_bins: &[f32]| {
            composer.render_rtsa_frame(
                &device,
                &queue,
                &target.view,
                [w, h],
                RtsaInput {
                    live_bins: &live,
                    max_hold_bins,
                    intensity: &[],
                    intensity_bins: 0,
                    intensity_levels: 0,
                    surface_points: surface_rect,
                    view,
                    theme: &theme,
                },
                phosphene_render::EguiFrame {
                    primitives: Vec::new(),
                    textures_delta: egui::TexturesDelta::default(),
                    pixels_per_point: 1.0,
                },
            );
            target.read_rgba(&device, &queue)
        };
        let bright_at = |rgba: &[u8], x0: i32, y0: i32| {
            let mut max = 0u32;
            for dy in -6i32..=6 {
                for dx in -3i32..=3 {
                    let x = (x0 + dx).clamp(0, w as i32 - 1) as u32;
                    let y = (y0 + dy).clamp(0, h as i32 - 1) as u32;
                    let o = ((y * w + x) * 4) as usize;
                    max = max
                        .max(u32::from(rgba[o]) + u32::from(rgba[o + 1]) + u32::from(rgba[o + 2]));
                }
            }
            max
        };
        let x = ((bin as f32 / (n - 1) as f32) * (w - 1) as f32) as i32;
        // From the view's OWN dB window rather than a literal copy of it:
        // D-065 §5 moved the default range, and a probe that restated the
        // old numbers would have gone quietly green on the wrong pixels.
        let y = ((1.0 - view.params.db_to_unit(v_hold)) * (h - 1) as f32) as i32;
        let img = render(&mut composer, &view, &shown);
        let lit = bright_at(&img, x, y);
        let quiet = bright_at(&img, (0.33 * (w - 1) as f32) as i32, y);
        assert!(
            lit > quiet + 60,
            "the max-hold line is not drawn where live fell away \
             (lit {lit} vs quiet {quiet} at ({x}, {y}))"
        );

        // M through the real chrome hides it: the step hands RtsaInput an
        // empty slice, and the line leaves the screen.
        let response = chrome(&mut view, Some(egui::Key::M));
        assert!(!view.max_hold, "M must hide the trace");
        let hidden =
            max_hold_frame_step(&pipeline, &view, &response.requests, 0.25, &live, &mut buf)
                .to_vec();
        assert!(
            hidden.is_empty(),
            "a hidden trace must reach RtsaInput as empty"
        );
        let img = render(&mut composer, &view, &hidden);
        assert!(bright_at(&img, x, y) + 60 < lit, "the M key hid nothing");

        // M again to show, then R through the real chrome: the production
        // step forwards the reset and the trace clears to the floor (the
        // stream has ended, so nothing re-seeds it).
        chrome(&mut view, Some(egui::Key::M));
        assert!(view.max_hold);
        let response = chrome(&mut view, Some(egui::Key::R));
        assert!(response.requests.reset_max_hold, "R must raise the reset");
        let cleared =
            max_hold_frame_step(&pipeline, &view, &response.requests, 0.25, &live, &mut buf);
        assert!(
            !cleared.is_empty() && cleared.iter().all(|&v| v == DBFS_FLOOR),
            "the R key did not clear the trace through the production step"
        );
        pipeline.shutdown();
    }

    /// D-035 end to end through the windowed step: the τ keys are produced
    /// by the REAL chrome and consumed by the same `apply_tau_requests`
    /// redraw runs — and the max-hold decay the production frame read
    /// applies actually changes. Over a drained tone-then-silence capture
    /// the arithmetic is exact: each `max_hold_frame` tick relaxes the
    /// retained peak's gap to the live trace by exp(−dt/τ), measured before
    /// and after the semicolon keypress. The persistence τ's own decay
    /// behaviour is sealed in `pipeline::tests` (it needs spectra flowing
    /// between ticks); here the comma key's chain down to the τ in effect
    /// is asserted through the same production step.
    #[test]
    fn tau_keys_drive_the_production_tau_step() {
        use std::time::{Duration, Instant};

        use phosphene_core::MaxHoldMode;
        use phosphene_sources::{FileSource, FileSourceConfig, Input, IqFormat};

        let n = 512usize;
        let (tone_batches, silent_batches) = (100usize, 400usize);
        let bytes = crate::pipeline::tone_then_silence_bytes(n, tone_batches, silent_batches);
        let config = FileSourceConfig::new(
            Input::Path(std::path::PathBuf::from("in-memory.cf32")),
            IqFormat::Cf32,
        );
        let source =
            FileSource::from_reader(Box::new(std::io::Cursor::new(bytes)), config).unwrap();
        let mut pipeline = Pipeline::start(
            Box::new(source),
            crate::pipeline::PipelineConfig {
                fft_size: n,
                window: phosphene_core::WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        while pipeline.health().batches_completed < (tone_batches + silent_batches) as u64 {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for the capture to drain"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let mut live = Vec::new();
        pipeline.copy_trace(&mut live);
        let bin = phosphene_core::fftshift_index(n / 4, n);

        // The real chrome, as redraw drives it.
        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        let mut view = ViewState::default();
        let hud = HudStats {
            fps: 60.0,
            frame_ms: 2.5,
            fft_size: n,
            window_name: "HANN",
            source_label: "TEST".into(),
            backend: pipeline.backend(),
            colormap: "p7",
            input_rate_sps: 0.0,
            processed_pct: 100.0,
            ffts_per_s: 0.0,
            dropped_batches: 0,
            persistence_tau_s: phosphene_core::DEFAULT_TAU_DECAY,
            gamma: phosphene_render::DEFAULT_GAMMA,
            live_tau_s: phosphene_core::DEFAULT_TAU_LIVE,
            max_hold_tau_s: phosphene_core::DEFAULT_TAU_MAX_HOLD,
            overflow_events: 0,
            ..Default::default()
        };
        let chrome_feed = phosphene_render::AnnotationFeed::new();
        let mut chrome_overlay_state = phosphene_render::AnnotationOverlayState::new();
        let mut chrome = |view: &mut ViewState, key: egui::Key| {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 720.0),
                )),
                events: vec![egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::default(),
                }],
                ..Default::default()
            };
            let mut response = None;
            let out = ctx.run_ui(input, |ui| {
                response = Some(phosphene_render::chrome::draw(
                    ui,
                    &theme,
                    view,
                    &hud,
                    &chrome_feed,
                    &mut chrome_overlay_state,
                ));
            });
            out.drop_without_applying_deltas();
            response.unwrap()
        };

        // Baseline: at the default τ = 2 s, each 0.5 s tick relaxes the
        // retained gap by exactly exp(−0.25).
        let mut buf = Vec::new();
        let gap = |buf: &Vec<f32>| buf[bin] - live[bin];
        pipeline.max_hold_frame(0.5, MaxHoldMode::DecayTowardLive, &live, &mut buf);
        let g1 = gap(&buf);
        pipeline.max_hold_frame(0.5, MaxHoldMode::DecayTowardLive, &live, &mut buf);
        let r_default = gap(&buf) / g1;
        let expect = (-0.25f32).exp();
        assert!(
            (r_default - expect).abs() < 0.005,
            "default max-hold decay ratio {r_default} != {expect}"
        );

        // Semicolon through the real chrome, consumed by the production
        // step: the next tick must follow the SHORTENED τ.
        let g2 = gap(&buf);
        let response = chrome(&mut view, egui::Key::Semicolon);
        assert_eq!(response.requests.max_hold_tau_steps, -1);
        apply_tau_requests(&pipeline, &response.requests);
        let tau_short = phosphene_core::DEFAULT_TAU_MAX_HOLD / TAU_STEP;
        pipeline.max_hold_frame(0.5, MaxHoldMode::DecayTowardLive, &live, &mut buf);
        let r_short = gap(&buf) / g2;
        let expect_short = (-0.5f32 / tau_short).exp();
        assert!(
            (r_short - expect_short).abs() < 0.005,
            "shortened max-hold decay ratio {r_short} != {expect_short}"
        );
        assert!(
            r_short < r_default,
            "a shorter max-hold τ must decay faster"
        );

        // Comma through the real chrome, same production step: the
        // persistence τ now in effect is one step shorter (an identity
        // multiply reads it back; its decay behaviour is sealed in
        // pipeline::tests).
        let response = chrome(&mut view, egui::Key::Comma);
        assert_eq!(response.requests.persistence_tau_steps, -1);
        apply_tau_requests(&pipeline, &response.requests);
        let tau_now = pipeline.adjust_persistence_tau(1.0);
        assert!(
            (tau_now - phosphene_core::DEFAULT_TAU_DECAY / TAU_STEP).abs() < 1e-6,
            "the comma key did not reach the persistence τ (now {tau_now})"
        );

        // K through the real chrome, same production step: the §7.4 live τ
        // now in effect is one step shorter (M1-J — FR-D2's averaging time,
        // exposed beside the other two, one idiom; an identity multiply
        // reads it back). Its measured settling behaviour is sealed in
        // pipeline::tests (it needs spectra flowing through the EMA).
        let response = chrome(&mut view, egui::Key::K);
        assert_eq!(response.requests.live_tau_steps, -1);
        apply_tau_requests(&pipeline, &response.requests);
        let live_now = pipeline.adjust_live_tau(1.0);
        assert!(
            (live_now - phosphene_core::DEFAULT_TAU_LIVE / TAU_STEP).abs() < 1e-6,
            "the K key did not reach the live-trace τ (now {live_now})"
        );
        pipeline.shutdown();
    }

    /// D-043/D-031: the four tunables the owner reported adjusting blind now
    /// show their value, and the seal proves it through the production
    /// path — press the real FR-D7 binding, apply it the same way `redraw`
    /// does, then build the HUD through the SAME `build_hud_stats` function
    /// `WindowState::hud_stats` calls in production (not a hand-reconstructed
    /// copy that could drift from it — that shape is exactly what let six
    /// controls ship dead behind green tests before, D-031).
    #[test]
    fn tunable_readouts_state_the_real_value_after_a_real_keypress() {
        let n = 512usize;
        let bytes = crate::pipeline::tone_then_silence_bytes(n, 4, 4);
        let config = FileSourceConfig::new(
            Input::Path(std::path::PathBuf::from("in-memory.cf32")),
            IqFormat::Cf32,
        );
        let source =
            FileSource::from_reader(Box::new(std::io::Cursor::new(bytes)), config).unwrap();
        let mut pipeline = Pipeline::start(
            Box::new(source),
            crate::pipeline::PipelineConfig {
                fft_size: n,
                window: phosphene_core::WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();
        let (device, _queue) = gpu("tunable-readouts-seal");
        let mut composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);

        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        let mut view = ViewState::default();

        // The real production function — the same one `WindowState::hud_stats`
        // calls — not a hand-reconstructed copy (D-031). An empty `frame_dts`
        // matches the first frame's real fallback; fps/frame_ms are not what
        // this seal is about.
        let frame_dts: VecDeque<f32> = VecDeque::new();
        let build_hud = |pipeline: &Pipeline, composer: &FrameComposer| {
            build_hud_stats(pipeline, composer, &frame_dts, &pipeline.meta())
        };

        // The chrome trims trailing zeros on these readouts to keep the bar
        // compact (`1.00` renders `1`, `0.71` renders `0.71`) — reproduced
        // here only to build the *expected* string from the real accessor's
        // live value, never to invent one.
        fn trimmed(value: f32) -> String {
            let s = format!("{value:.2}");
            let s = s.trim_end_matches('0').trim_end_matches('.');
            s.to_owned()
        }

        // Baseline: the defaults are already on the bar before any key is
        // pressed (D-043's whole complaint was that they never were). The
        // persistence τ and gamma share one compact "pers" field (FR-D1
        // names them as one control pair).
        let hud = build_hud(&pipeline, &composer);
        let (before, _) = chrome_frame(&ctx, &theme, &mut view, &hud, None);
        assert!(
            before.iter().any(|t| t
                == &format!(
                    "{}S G{}",
                    trimmed(phosphene_core::DEFAULT_TAU_DECAY),
                    trimmed(phosphene_render::DEFAULT_GAMMA)
                )),
            "persistence τ/gamma defaults not on the bar; texts: {before:?}"
        );
        assert!(
            before
                .iter()
                .any(|t| t == &format!("{}S", trimmed(phosphene_core::DEFAULT_TAU_LIVE))),
            "live τ default not on the bar; texts: {before:?}"
        );
        assert!(
            before
                .iter()
                .any(|t| t == &format!("DECAY {}S", trimmed(phosphene_core::DEFAULT_TAU_MAX_HOLD))),
            "max-hold τ default not on the bar; texts: {before:?}"
        );

        // Comma: persistence τ shortens by TAU_STEP. Produce through the
        // real chrome, consume through the real `apply_tau_requests` step.
        // Gamma has not moved yet, so the combined field states the new τ
        // beside the still-default gamma.
        let (_, response) = chrome_frame(&ctx, &theme, &mut view, &hud, Some(egui::Key::Comma));
        apply_tau_requests(&pipeline, &response.requests);
        let hud = build_hud(&pipeline, &composer);
        let expect_tau = pipeline.persistence_tau();
        assert!((expect_tau - phosphene_core::DEFAULT_TAU_DECAY / TAU_STEP).abs() < 1e-6);
        let (texts, _) = chrome_frame(&ctx, &theme, &mut view, &hud, None);
        assert!(
            texts.iter().any(|t| t
                == &format!(
                    "{}S G{}",
                    trimmed(expect_tau),
                    trimmed(phosphene_render::DEFAULT_GAMMA)
                )),
            "persistence τ readout did not follow the comma key; texts: {texts:?}"
        );

        // PageUp: gamma steps up, alongside the τ the comma key just set.
        // Consumed through the same `composer.apply_requests` the redraw
        // loop calls.
        let (_, response) = chrome_frame(&ctx, &theme, &mut view, &hud, Some(egui::Key::PageUp));
        composer.apply_requests(&response.requests);
        let hud = build_hud(&pipeline, &composer);
        let expect_gamma = composer.gamma();
        assert!((expect_gamma - phosphene_render::DEFAULT_GAMMA).abs() > 1e-6);
        let (texts, _) = chrome_frame(&ctx, &theme, &mut view, &hud, None);
        assert!(
            texts
                .iter()
                .any(|t| t == &format!("{}S G{}", trimmed(expect_tau), trimmed(expect_gamma))),
            "gamma readout did not follow the PgUp key; texts: {texts:?}"
        );

        // K: live-trace τ shortens.
        let (_, response) = chrome_frame(&ctx, &theme, &mut view, &hud, Some(egui::Key::K));
        apply_tau_requests(&pipeline, &response.requests);
        let hud = build_hud(&pipeline, &composer);
        let expect = pipeline.live_tau();
        assert!((expect - phosphene_core::DEFAULT_TAU_LIVE / TAU_STEP).abs() < 1e-6);
        let (texts, _) = chrome_frame(&ctx, &theme, &mut view, &hud, None);
        assert!(
            texts.iter().any(|t| t == &format!("{}S", trimmed(expect))),
            "live τ readout did not follow the K key; texts: {texts:?}"
        );

        // Semicolon: max-hold τ shortens, and the mode word stays alongside
        // it (D-043: the field showed the mode but not the τ before this
        // lane).
        let (_, response) = chrome_frame(&ctx, &theme, &mut view, &hud, Some(egui::Key::Semicolon));
        apply_tau_requests(&pipeline, &response.requests);
        let hud = build_hud(&pipeline, &composer);
        let expect = pipeline.max_hold_tau();
        assert!((expect - phosphene_core::DEFAULT_TAU_MAX_HOLD / TAU_STEP).abs() < 1e-6);
        let (texts, _) = chrome_frame(&ctx, &theme, &mut view, &hud, None);
        assert!(
            texts
                .iter()
                .any(|t| t == &format!("DECAY {}S", trimmed(expect))),
            "max-hold τ readout did not follow the semicolon key; texts: {texts:?}"
        );

        pipeline.shutdown();
    }

    /// D-050/D-031: a nonzero overflow count, driven through a real source
    /// that reports it and a real pipeline, renders on the HUD as **events**
    /// — never a bare number indistinguishable from a sample or batch count
    /// — and `DROPS` is untouched by it (an event carries no quantity; the
    /// pipeline's own D-050 seal in `pipeline::tests` covers the accounting
    /// separation, this covers the render side the lane doc names
    /// explicitly).
    #[test]
    fn overflow_events_render_as_events_and_drops_stays_untouched() {
        use phosphene_core::Complex;
        use phosphene_sources::{
            Control, ControlCaps, SampleSink, SampleSource, SinkFlow, SourceDesc, SourceError,
        };

        /// A source that streams a few clean batches, then reports overflow
        /// EVENTS (D-050) — the minimal fault it takes to drive
        /// `Pipeline::device_overflow_events` above zero through the real
        /// production path, rather than asserting the HUD field in isolation.
        struct OverflowSource {
            meta: SourceMeta,
            batches: usize,
            batch_len: usize,
            overflow_events: usize,
        }

        impl SampleSource for OverflowSource {
            fn open(_: &SourceDesc) -> Result<Self, SourceError> {
                Err(SourceError::WrongBackend {
                    backend: "test-overflow",
                })
            }

            fn stream(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError> {
                sink.meta_changed(&self.meta);
                let block = vec![Complex::new(0.05f32, 0.0); self.batch_len];
                for _ in 0..self.batches {
                    if sink.push(&block) == SinkFlow::Stop {
                        return Ok(());
                    }
                }
                for _ in 0..self.overflow_events {
                    sink.device_overflow();
                }
                Ok(())
            }

            fn caps(&self) -> ControlCaps {
                ControlCaps::NONE
            }

            fn set(&mut self, ctl: Control) -> Result<(), SourceError> {
                Err(SourceError::UnsupportedControl {
                    backend: "test-overflow",
                    control: ctl.name(),
                })
            }

            fn meta(&self) -> SourceMeta {
                self.meta.clone()
            }
        }

        let n = 512usize;
        let source = OverflowSource {
            meta: SourceMeta {
                sample_rate_hz: Some(8_000.0),
                center_freq_hz: None,
                label: "overflow-test".to_owned(),
                provenance: "test".to_owned(),
            },
            batches: 4,
            batch_len: n,
            overflow_events: 3,
        };
        let mut pipeline = Pipeline::start(
            Box::new(source),
            crate::pipeline::PipelineConfig {
                fft_size: n,
                window: phosphene_core::WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();

        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while pipeline.device_overflow_events() < 3 {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for the 3 overflow events"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        // The pipeline's own accounting: an event never becomes a dropped
        // batch (D-050) — checked before the HUD is even built, so a HUD-side
        // bug could not hide an accounting bug.
        assert_eq!(
            pipeline.health().dropped_batches,
            0,
            "an overflow event must never be counted as a dropped batch (D-050)"
        );

        let (device, _queue) = gpu("overflow-readout-seal");
        let composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);
        let frame_dts: VecDeque<f32> = VecDeque::new();
        // The same production `build_hud_stats` every other seal in this
        // file drives (D-031) — not a HudStats built by hand for this test.
        let hud = build_hud_stats(&pipeline, &composer, &frame_dts, &pipeline.meta());
        assert_eq!(hud.overflow_events, 3);
        assert_eq!(hud.dropped_batches, 0);

        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        let mut view = ViewState::default();
        let (texts, _) = chrome_frame(&ctx, &theme, &mut view, &hud, None);
        assert!(
            texts.iter().any(|t| t == "3 EVT"),
            "overflow count did not render as events; texts: {texts:?}"
        );
        // Never a bare number standing in for it — that would be
        // indistinguishable from a sample or batch count, exactly the
        // failure D-050 removed on the accounting side.
        assert!(
            !texts.iter().any(|t| t == "3"),
            "overflow count rendered as a bare number, not labelled as events; texts: {texts:?}"
        );

        pipeline.shutdown();
    }

    /// D-031 produce-and-consume for the FR-D7 screenshot key: the response
    /// is produced by the REAL chrome from a real `S` keypress, then
    /// consumed by the same `consume_screenshot_request` the redraw path
    /// calls, on a real GPU — ending in a decodable PNG on disk. The only
    /// line not under test is redraw's own call into that function.
    #[test]
    fn screenshot_key_travels_from_chrome_to_a_png_on_disk() {
        // Produce: press S through the real chrome.
        let ctx = egui::Context::default();
        let theme = Theme::default();
        theme.install(&ctx);
        let mut view = ViewState::default();
        view.params.sample_rate = Some(2_400_000.0);
        let hud = HudStats {
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
            persistence_tau_s: phosphene_core::DEFAULT_TAU_DECAY,
            gamma: phosphene_render::DEFAULT_GAMMA,
            live_tau_s: phosphene_core::DEFAULT_TAU_LIVE,
            max_hold_tau_s: phosphene_core::DEFAULT_TAU_MAX_HOLD,
            overflow_events: 0,
            ..Default::default()
        };
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 720.0),
            )),
            events: vec![egui::Event::Key {
                key: egui::Key::S,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::default(),
            }],
            ..Default::default()
        };
        let mut response = None;
        let feed = phosphene_render::AnnotationFeed::new();
        let mut overlay_state = phosphene_render::AnnotationOverlayState::new();
        let out = ctx.run_ui(input, |ui| {
            response = Some(phosphene_render::chrome::draw(
                ui,
                &theme,
                &mut view,
                &hud,
                &feed,
                &mut overlay_state,
            ));
        });
        out.drop_without_applying_deltas();
        let response = response.unwrap();
        assert!(response.screenshot, "the S key must request a screenshot");

        // Consume: the same function redraw calls.
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .expect("no GPU adapter — the screenshot seal cannot run");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("phosphene-screenshot-test"),
            ..Default::default()
        }))
        .expect("GPU device request failed");
        let mut composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);
        let bins = vec![-80.0f32; 1024];
        let dir =
            std::env::temp_dir().join(format!("phosphene-screenshot-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let scene = || RtsaInput {
            live_bins: &bins,
            max_hold_bins: &[],
            intensity: &[],
            intensity_bins: 0,
            intensity_levels: 0,
            surface_points: response.grid,
            view: &view,
            theme: &theme,
        };
        let egui_frame = || phosphene_render::EguiFrame {
            primitives: Vec::new(),
            textures_delta: egui::TexturesDelta::default(),
            pixels_per_point: 1.0,
        };
        let path = consume_screenshot_request(
            &response,
            &device,
            &queue,
            &mut composer,
            phosphene_render::OFFSCREEN_FORMAT,
            [640, 360],
            scene(),
            egui_frame(),
            &dir,
        )
        .expect("a set screenshot flag must be consumed")
        .expect("screenshot write failed");

        // The PNG exists and decodes at the render size.
        let file = std::fs::File::open(&path).unwrap();
        let mut reader = png::Decoder::new(std::io::BufReader::new(file))
            .read_info()
            .unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height), (640, 360));
        std::fs::remove_dir_all(&dir).ok();

        // A response without the flag is not consumed.
        let quiet = phosphene_render::ChromeResponse {
            screenshot: false,
            ..response
        };
        assert!(consume_screenshot_request(
            &quiet,
            &device,
            &queue,
            &mut composer,
            phosphene_render::OFFSCREEN_FORMAT,
            [64, 64],
            scene(),
            egui_frame(),
            &dir,
        )
        .is_none());
    }

    // ---------------------------------------------------------------------
    // D-056 / D-058: live retune, end to end through the production path.
    // ---------------------------------------------------------------------

    /// Centre the fake radio opens on, Hz.
    const RADIO_START_HZ: f64 = 100e6;
    /// Its sample rate — one window width, and therefore one D-058 slider
    /// step (2^21 S/s keeps every derived interval binary-exact in f32).
    const RADIO_RATE: f64 = WF_TEST_RATE;
    /// The tuning grid the fake radio snaps to, Hz. Real tuners land on their
    /// own grid, and the whole point of C5 is that the display follows where
    /// the radio *went*, not where it was asked to go — so this fake refuses
    /// to be exactly obedient, and the seals below can tell the difference.
    const RADIO_SNAP_HZ: f64 = 25e3;
    /// The range the fake radio reports (a B200mini's, near enough).
    const RADIO_MIN_HZ: f64 = 42e6;
    const RADIO_MAX_HZ: f64 = 6.008e9;
    /// Level of the tone the radio emits at its opening centre, dBFS.
    const RADIO_TONE_DBFS: f32 = -20.0;

    /// A tunable radio with no hardware in it: it streams a recognisable tone
    /// at its opening centre and **silence at every other centre**, so a seal
    /// can tell at a glance whether any accumulator still holds pre-retune
    /// history (D-056).
    ///
    /// It is a real [`SampleSource`]: it hands out a real
    /// [`ControlHandle`](phosphene_sources::source::ControlHandle), drains it
    /// inside its own stream loop exactly as the SoapySDR backend does,
    /// snaps like a real tuner, and announces the readback in band. What the
    /// seals drive is therefore the production mechanism, with only the
    /// device swapped out.
    struct FakeRadio {
        /// Where the tuner actually is — what the radio *receives*. Set by
        /// the mutation half of a control.
        hw_center_hz: f64,
        /// The last centre successfully **read back**, and therefore the only
        /// one [`FakeRadio::meta`] reports (C5). The two are separate fields
        /// for the same reason they are separate operations on a real radio:
        /// D-060's whole case is the window between them.
        center_hz: f64,
        n: usize,
        control: Option<phosphene_sources::source::ControlInbox>,
        /// Every centre frequency the source was actually asked for — the
        /// proof that the key/box/slider reached `set(Control::…)` at all.
        asked: Arc<Mutex<Vec<f64>>>,
        /// Set to refuse every retune, for the "a rejected retune is a real
        /// error" seal.
        refuse: bool,
        /// Set to make the **readback** fail while the tune itself succeeds
        /// — D-060's state, the one no success path can produce. Shared, so a
        /// seal can break the readback of a radio that is already streaming.
        readback_fails: Arc<AtomicBool>,
    }

    impl FakeRadio {
        fn new(n: usize, asked: Arc<Mutex<Vec<f64>>>) -> FakeRadio {
            FakeRadio {
                hw_center_hz: RADIO_START_HZ,
                center_hz: RADIO_START_HZ,
                n,
                control: None,
                asked,
                refuse: false,
                readback_fails: Arc::new(AtomicBool::new(false)),
            }
        }

        /// One batch of what this radio is receiving *right now*: an on-bin
        /// tone at the opening centre, silence anywhere else. Keyed off where
        /// the tuner **is**, not off what was read back — a radio that moved
        /// receives the new frequency whether or not anyone could confirm it.
        fn fill(&self, buf: &mut [Complex<f32>]) {
            if self.hw_center_hz == RADIO_START_HZ {
                // Raw bin N/4 at −20 dBFS; phase repeats every 4 samples, so
                // it is exact in f32.
                for (i, s) in buf.iter_mut().enumerate() {
                    let phase = std::f32::consts::TAU * 0.25 * (i % 4) as f32;
                    let (im, re) = phase.sin_cos();
                    *s = Complex::new(re * 0.1, im * 0.1);
                }
            } else {
                buf.fill(Complex::new(0.0, 0.0));
            }
        }

        /// Service pending live controls, exactly as the Soapy backend does
        /// — through the **production** [`apply_and_verify`] /
        /// [`settle_control`] pair, not a copy of them. That is what makes
        /// the seals below evidence about the shipped ordering rather than
        /// about this fixture: only the device is swapped out.
        ///
        /// Returns the stream's verdict: `Err` exactly when D-060 says the
        /// stream must end.
        fn drain(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError> {
            while let Some(request) = self
                .control
                .as_ref()
                .and_then(phosphene_sources::source::ControlInbox::try_recv)
            {
                let control = request.control().clone();
                let before = self.meta();
                let outcome = {
                    // `Cell` so the mutation and the readback can each touch
                    // the radio without borrowing the other's field.
                    let hw = std::cell::Cell::new(self.hw_center_hz);
                    let readback = std::cell::Cell::new(self.center_hz);
                    let refuse = self.refuse;
                    let blind = self.readback_fails.load(Ordering::Relaxed);
                    let asked = &self.asked;
                    let outcome = apply_and_verify(
                        "fake-radio",
                        &control,
                        || match control {
                            Control::CenterFreqHz(hz) => {
                                asked.lock().unwrap().push(hz);
                                if refuse {
                                    return Err(SourceError::InvalidConfig(format!(
                                        "device refused center frequency {hz} Hz"
                                    )));
                                }
                                // A tuner lands on its own grid — the readback
                                // is not the request.
                                hw.set((hz / RADIO_SNAP_HZ).round() * RADIO_SNAP_HZ);
                                Ok(())
                            }
                            ref other => Err(SourceError::UnsupportedControl {
                                backend: "fake-radio",
                                control: other.name(),
                            }),
                        },
                        || {
                            if blind {
                                Err(SourceError::Io(
                                    "cannot read the center frequency back from fake-radio".into(),
                                ))
                            } else {
                                readback.set(hw.get());
                                Ok(())
                            }
                        },
                    );
                    self.hw_center_hz = hw.get();
                    self.center_hz = readback.get();
                    outcome
                };
                let after = self.meta();
                settle_control(outcome, request, sink, &before, &after)?;
            }
            Ok(())
        }
    }

    impl SampleSource for FakeRadio {
        fn open(_: &SourceDesc) -> Result<Self, SourceError> {
            Err(SourceError::WrongBackend {
                backend: "fake-radio",
            })
        }

        fn stream(&mut self, sink: &mut dyn SampleSink) -> Result<(), SourceError> {
            sink.meta_changed(&self.meta());
            let mut buf = vec![Complex::new(0.0, 0.0); self.n];
            loop {
                // D-060: a tune the radio took but would not confirm ends the
                // stream, exactly as it does in the SoapySDR backend.
                self.drain(sink)?;
                self.fill(&mut buf);
                if sink.push(&buf) == SinkFlow::Stop {
                    return Ok(());
                }
                // A live radio is time-paced; this one is paced coarsely so
                // the seals run in wall-clock time without spinning a core.
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        }

        fn caps(&self) -> ControlCaps {
            ControlCaps {
                tune: true,
                rate: false,
                gain: false,
                antenna: false,
                tune_range: TuneRange::hull([(RADIO_MIN_HZ, RADIO_MAX_HZ)]),
            }
        }

        fn set(&mut self, ctl: Control) -> Result<(), SourceError> {
            match ctl {
                Control::CenterFreqHz(hz) => {
                    self.hw_center_hz = (hz / RADIO_SNAP_HZ).round() * RADIO_SNAP_HZ;
                    self.center_hz = self.hw_center_hz;
                    Ok(())
                }
                other => Err(SourceError::UnsupportedControl {
                    backend: "fake-radio",
                    control: other.name(),
                }),
            }
        }

        fn control_handle(&mut self) -> Option<phosphene_sources::source::ControlHandle> {
            let (handle, inbox) = phosphene_sources::source::control_channel();
            self.control = Some(inbox);
            Some(handle)
        }

        fn meta(&self) -> SourceMeta {
            SourceMeta {
                sample_rate_hz: Some(RADIO_RATE),
                center_freq_hz: Some(self.center_hz),
                label: "fake-radio".into(),
                provenance: "fake-radio (cf32)".into(),
            }
        }
    }

    /// A pipeline over a [`FakeRadio`], paced like a live device.
    fn radio_pipeline(n: usize, radio: FakeRadio) -> Pipeline {
        Pipeline::start(
            Box::new(radio),
            PipelineConfig {
                fft_size: n,
                window: phosphene_core::WindowKind::Hann,
                paced: true,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap()
    }

    /// One chrome frame with the D-058 retune controls, driven by real egui
    /// input — the production `draw_tunable`, not a copy of it.
    fn tune_frame(
        ctx: &egui::Context,
        view: &mut ViewState,
        hud: &HudStats,
        tune: &TuneUi,
        events: Vec<egui::Event>,
    ) -> ChromeResponse {
        let theme = Theme::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 720.0),
            )),
            events,
            ..Default::default()
        };
        let mut response = None;
        let feed = phosphene_render::AnnotationFeed::new();
        let mut overlay_state = phosphene_render::AnnotationOverlayState::new();
        let out = ctx.run_ui(input, |ui| {
            response = Some(phosphene_render::chrome::draw_tunable(
                ui,
                &theme,
                view,
                hud,
                tune,
                &feed,
                &mut overlay_state,
            ));
        });
        out.drop_without_applying_deltas();
        response.unwrap()
    }

    fn key_event(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        }
    }

    /// **The lane's headline seal (D-031): the retune box, end to end.**
    ///
    /// A user types a frequency into the REAL chrome widget and presses
    /// Enter; the request travels the REAL `consume_retune_request` the
    /// redraw path calls; the source receives `Control::CenterFreqHz`; and
    /// the **device's readback** — deliberately not the typed value, because
    /// this radio snaps to a 25 kHz grid like a real one — reaches the axis
    /// through the same `apply_meta` a window uses.
    ///
    /// A test that called `pipeline.retune()` directly would prove the seam
    /// and nothing else; that shape is why six controls once shipped dead
    /// (D-031/D-032/D-033).
    #[test]
    fn the_retune_box_drives_the_source_and_the_readback_reaches_the_axis() {
        let n = 512;
        let asked = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = radio_pipeline(n, FakeRadio::new(n, asked.clone()));
        wait_until("the radio to stream", || {
            pipeline.meta().center_freq_hz == Some(RADIO_START_HZ)
        });

        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        let mut status = None;
        let hud = test_hud();

        // Frame 1: the bar as a user first sees it, showing the readback.
        let meta = pipeline.meta();
        let tune = tune_ui(&pipeline, &meta, status.clone());
        assert!(tune.tunable, "the fake radio advertises tuning");
        assert_eq!(tune.range_hz, Some((RADIO_MIN_HZ, RADIO_MAX_HZ)));
        assert_eq!(tune.step_hz, Some(RADIO_RATE), "one step is one window");
        let response = tune_frame(&ctx, &mut view, &hud, &tune, Vec::new());
        assert_eq!(response.retune_hz, None, "drawing must not retune");

        // Frame 2: click into the box, select all, type an OFF-GRID
        // frequency, press Enter. 751.01 MHz is deliberately not on the
        // radio's 25 kHz grid.
        ctx.memory_mut(|m| m.request_focus(phosphene_render::chrome::tune_box_id()));
        let events = vec![
            key_event(egui::Key::A, egui::Modifiers::COMMAND),
            egui::Event::Text("751.01M".into()),
            key_event(egui::Key::Enter, egui::Modifiers::default()),
        ];
        let response = tune_frame(&ctx, &mut view, &hud, &tune, events);
        assert_eq!(
            response.retune_hz,
            Some(751_010_000.0),
            "the box must ask for exactly what was typed"
        );

        // The production consumer — the same call the redraw path makes.
        let outcome = consume_retune_request(&pipeline, &response, &mut status)
            .expect("a retune request must be consumed");
        let readback = outcome.expect("the radio accepted the retune");
        assert_eq!(
            readback, 751_000_000.0,
            "the answer must be where the radio WENT, not what was asked"
        );

        // The source really was asked — through `Control::CenterFreqHz`.
        assert_eq!(&*asked.lock().unwrap(), &[751_010_000.0]);

        // …and the readback reaches the axis, in band, at the samples it
        // describes.
        wait_until("the retune to reach the published metadata", || {
            pipeline.meta().center_freq_hz == Some(751_000_000.0)
        });
        let meta = pipeline.meta();
        apply_meta(&mut view.params, &meta);
        assert_eq!(view.params.center, Some(751_000_000.0));
        assert_eq!(view.params.sample_rate, Some(RADIO_RATE), "rate is fixed");

        // And the chrome now shows the radio's own frequency, not the typed
        // one (D-058: where the radio IS).
        let tune = tune_ui(&pipeline, &meta, status.clone());
        assert_eq!(tune.center_hz, Some(751_000_000.0));
        tune_frame(&ctx, &mut view, &hud, &tune, Vec::new());
        let shown = ctx
            .data(|d| d.get_temp::<String>(phosphene_render::chrome::tune_box_id()))
            .unwrap();
        assert_eq!(shown, "751M", "the box shows the readback, not the request");
        let status = status.expect("the outcome is surfaced");
        assert!(status.ok && status.message.contains("751M"), "{status:?}");

        pipeline.shutdown();
    }

    /// **The second D-058 entry point, through the same one mechanism.**
    ///
    /// A real pointer drag on the REAL slider produces a retune request that
    /// is a whole number of sample rates from where the radio is — one window
    /// width per increment, so adjacent positions tile the spectrum — and it
    /// travels the same `consume_retune_request` the box does, reaching the
    /// same `Control::CenterFreqHz`.
    #[test]
    fn the_retune_slider_steps_whole_windows_through_the_same_path() {
        let n = 512;
        let asked = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = radio_pipeline(n, FakeRadio::new(n, asked.clone()));
        wait_until("the radio to stream", || {
            pipeline.meta().center_freq_hz == Some(RADIO_START_HZ)
        });

        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        let mut status = None;
        let hud = test_hud();
        let tune = tune_ui(&pipeline, &pipeline.meta(), None);

        // Draw once to place the slider, then drive the real widget.
        tune_frame(&ctx, &mut view, &hud, &tune, Vec::new());
        let rect = phosphene_render::chrome::tune_slider_rect(&ctx)
            .expect("the tunable source must have been given a slider");
        let grab = rect.center();
        // A small drag to the right: a few windows along the band.
        let release = egui::pos2(grab.x + 12.0, grab.y);
        tune_frame(
            &ctx,
            &mut view,
            &hud,
            &tune,
            vec![
                egui::Event::PointerMoved(grab),
                egui::Event::PointerButton {
                    pos: grab,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerMoved(release),
            ],
        );
        let response = tune_frame(
            &ctx,
            &mut view,
            &hud,
            &tune,
            vec![egui::Event::PointerButton {
                pos: release,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        let target = response
            .retune_hz
            .expect("dragging the slider must ask for a retune");

        // D-058's increment, measured off the request the widget produced:
        // a whole number of sample rates from where the radio is.
        let windows = (target - RADIO_START_HZ) / RADIO_RATE;
        assert!(
            windows > 0.0 && (windows - windows.round()).abs() < 1e-9,
            "the slider moved {windows} windows — the step must be a whole \
             number of sample rates so adjacent positions tile the spectrum"
        );

        // Same consumer, same `set(Control::CenterFreqHz)`: one mechanism.
        let readback = consume_retune_request(&pipeline, &response, &mut status)
            .expect("consumed")
            .expect("accepted");
        assert_eq!(&*asked.lock().unwrap(), &[target]);
        wait_until("the slider retune to reach the axis", || {
            pipeline.meta().center_freq_hz == Some(readback)
        });
        apply_meta(&mut view.params, &pipeline.meta());
        assert_eq!(view.params.center, Some(readback));
        pipeline.shutdown();
    }

    /// Click the REAL D-062 step button — the one the chrome drew, found by
    /// its own stashed rect — with real pointer input, over two frames
    /// (press, then release, which is when egui reports a click).
    fn click_tune_step(
        ctx: &egui::Context,
        view: &mut ViewState,
        hud: &HudStats,
        tune: &TuneUi,
        up: bool,
    ) -> ChromeResponse {
        let rect = phosphene_render::chrome::tune_step_rect(ctx, up)
            .expect("a tunable source must be given step buttons");
        let at = rect.center();
        tune_frame(
            ctx,
            view,
            hud,
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
        tune_frame(
            ctx,
            view,
            hud,
            tune,
            vec![egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        )
    }

    /// **The lane's headline seal (D-062, D-066 §2, D-031): the step button
    /// and the global keys, each end to end.**
    ///
    /// A user clicks the arrow the chrome actually drew, and presses the bare
    /// and shifted arrows on the real keymap; each request travels the REAL
    /// `consume_retune_request` the redraw path calls; the source receives
    /// `Control::CenterFreqHz` with a centre **one sample rate away** for the
    /// coarse chords and **one tenth of one** for the fine chord; and the
    /// device's readback reaches the axis every time.
    ///
    /// A test that called `step_target` would prove the arithmetic and
    /// nothing else — that shape is why six controls once shipped dead
    /// (D-031/D-032/D-033), and it is exactly what this lane was told not to
    /// repeat.
    #[test]
    fn the_step_button_and_the_global_key_each_reach_the_radio_one_window_away() {
        let n = 512;
        let asked = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = radio_pipeline(n, FakeRadio::new(n, asked.clone()));
        wait_until("the radio to stream", || {
            pipeline.meta().center_freq_hz == Some(RADIO_START_HZ)
        });

        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        let mut status = None;
        let hud = test_hud();

        // --- The button. ---
        let tune = tune_ui(&pipeline, &pipeline.meta(), status.clone());
        assert_eq!(tune.step_hz, Some(RADIO_RATE), "one step is one window");
        // Frame 1 places the buttons; drawing must not retune.
        let response = tune_frame(&ctx, &mut view, &hud, &tune, Vec::new());
        assert_eq!(response.retune_hz, None);

        let response = click_tune_step(&ctx, &mut view, &hud, &tune, true);
        assert_eq!(
            response.retune_hz,
            Some(RADIO_START_HZ + RADIO_RATE),
            "the up button must ask for exactly one sample rate up"
        );
        let readback = consume_retune_request(&pipeline, &response, &mut status)
            .expect("a button retune must be consumed")
            .expect("the radio accepted it");
        assert_eq!(
            &*asked.lock().unwrap(),
            &[RADIO_START_HZ + RADIO_RATE],
            "the button must reach Control::CenterFreqHz itself"
        );
        wait_until("the button retune to reach the published metadata", || {
            pipeline.meta().center_freq_hz == Some(readback)
        });
        apply_meta(&mut view.params, &pipeline.meta());
        assert_eq!(view.params.center, Some(readback));

        // --- The global keys, from wherever the button left the radio.
        //
        // **D-066 §2's assignment, on a live radio.** The whole horizontal
        // axis is the radio's: the bare `←` steps one window, `Shift+←` a
        // tenth of one, and dB/div — off the arrows now, on `O`/`P` — asks
        // the radio for nothing at all. Every one of the three is asserted in
        // both directions, because a test that only checked "the arrow
        // retunes" would pass just as happily if the two steps were the same
        // size, which is exactly how this remap goes wrong. ---
        let center = pipeline.meta().center_freq_hz.unwrap();
        let tune = tune_ui(&pipeline, &pipeline.meta(), status.clone());

        // `O`: dB/div moves, and the radio is not asked for anything.
        let db_per_div = view.params.db_per_div;
        let db_key = tune_frame(
            &ctx,
            &mut view,
            &hud,
            &tune,
            vec![key_event(egui::Key::O, egui::Modifiers::default())],
        );
        assert_eq!(
            db_key.retune_hz, None,
            "the dB/div key reached the radio - it is a display control"
        );
        assert!(
            view.params.db_per_div < db_per_div,
            "O must step dB/div finer (now {})",
            view.params.db_per_div
        );
        assert_eq!(
            asked.lock().unwrap().len(),
            1,
            "the dB/div key issued a device control call"
        );

        // Bare ←: the radio moves exactly one window down, and the display's
        // dB/div does not move with it.
        let db_per_div = view.params.db_per_div;
        let response = tune_frame(
            &ctx,
            &mut view,
            &hud,
            &tune,
            vec![key_event(egui::Key::ArrowLeft, egui::Modifiers::default())],
        );
        assert_eq!(
            response.retune_hz,
            Some(center - RADIO_RATE),
            "the bare left arrow must ask for exactly one sample rate down"
        );
        assert_eq!(
            view.params.db_per_div, db_per_div,
            "the bare left arrow also moved dB/div"
        );
        let readback = consume_retune_request(&pipeline, &response, &mut status)
            .expect("a key retune must be consumed")
            .expect("the radio accepted it");
        assert_eq!(
            asked.lock().unwrap().len(),
            2,
            "the key must reach Control::CenterFreqHz too"
        );
        assert_eq!(asked.lock().unwrap()[1], center - RADIO_RATE);
        wait_until("the key retune to reach the published metadata", || {
            pipeline.meta().center_freq_hz == Some(readback)
        });
        apply_meta(&mut view.params, &pipeline.meta());
        assert_eq!(view.params.center, Some(readback));

        // --- The fine key (D-066 §2), from wherever the coarse key left the
        // radio: **a tenth of a window, and demonstrably not a whole one.**
        let fine_center = pipeline.meta().center_freq_hz.unwrap();
        let tune = tune_ui(&pipeline, &pipeline.meta(), status.clone());
        let db_per_div = view.params.db_per_div;
        let response = tune_frame(
            &ctx,
            &mut view,
            &hud,
            &tune,
            vec![key_event(egui::Key::ArrowRight, egui::Modifiers::SHIFT)],
        );
        assert_eq!(
            response.retune_hz,
            Some(fine_center + RADIO_RATE / phosphene_render::chrome::FINE_TUNE_DIVISOR),
            "shift+right must ask for exactly one tenth of a window up"
        );
        assert_ne!(
            response.retune_hz,
            Some(fine_center + RADIO_RATE),
            "shift+right asked for a whole window - the fine step is not fine"
        );
        assert_eq!(
            view.params.db_per_div, db_per_div,
            "shift+right also moved dB/div - it is off the arrows now"
        );
        let readback = consume_retune_request(&pipeline, &response, &mut status)
            .expect("a fine key retune must be consumed")
            .expect("the radio accepted it");
        assert_eq!(
            asked.lock().unwrap().len(),
            3,
            "the fine key must reach Control::CenterFreqHz too"
        );
        wait_until("the fine retune to reach the published metadata", || {
            pipeline.meta().center_freq_hz == Some(readback)
        });
        apply_meta(&mut view.params, &pipeline.meta());
        assert_eq!(view.params.center, Some(readback));

        // D-058's tiling, and the anchor that makes it hold: each COARSE step
        // is one window from **where the radio is**, not from an absolute
        // grid. The fake radio snaps to its own 25 kHz grid, so the second
        // step is one window from the snapped readback rather than from the
        // first request — which is the whole reason the increment is anchored
        // to the readback.
        let asked = asked.lock().unwrap().clone();
        assert_eq!(asked[0] - RADIO_START_HZ, RADIO_RATE, "up: one window");
        assert_eq!(center - asked[1], RADIO_RATE, "down: one window");
        assert_ne!(center, asked[0], "test premise: the radio snapped");
        // And the fine step deliberately does NOT tile — D-066 §2 says so in
        // as many words: it is for centring a signal inside the window, not
        // for walking a band, and a later pass must not "fix" it.
        // Within a hertz: the request is one f64 sum away from the readback,
        // and the assertion is about the *size of the step*, not about the
        // last bit of a 200 kHz figure.
        assert!(
            (asked[2] - fine_center - RADIO_RATE / 10.0).abs() < 1.0,
            "fine: a tenth of a window, got {}",
            asked[2] - fine_center
        );
        assert!(
            (asked[2] - fine_center - RADIO_RATE).abs() > 1.0,
            "the fine step tiled"
        );

        // And the display state never moved for a *tune*: a tune step is
        // device state. (dB/div did move — `O` above moved it, from its new
        // home off the arrows, and D-066 §1 has that derive a new ceiling
        // while the **floor stays pinned**, which is the half of the display
        // an arrow key must never have touched.)
        assert!(!view.paused);
        assert_eq!(
            view.params.db_bottom,
            ViewState::default().params.db_bottom,
            "the noise floor moved under the user"
        );
        assert_eq!(
            view.params.db_top,
            view.params.db_bottom + 10.0 * view.params.db_per_div,
            "the ceiling is derived from the pinned floor"
        );
        pipeline.shutdown();
    }

    /// **Four ways in, one mechanism (D-058 §3, D-062).**
    ///
    /// The box, the slider, the step buttons and the global keys are driven
    /// in turn on one live radio, each through the real widget, and every one
    /// of them:
    ///
    /// * surfaces as the single [`ChromeResponse::retune_hz`] seam — the
    ///   render crate has no other way to ask, and no device to ask with;
    /// * is applied by the one `consume_retune_request` → `Pipeline::retune`
    ///   call the frame loop makes (`the_display_crate_cannot_reach_device_control`
    ///   pins that there is exactly one, in the production half of this file);
    /// * reaches `Control::CenterFreqHz` on the source itself;
    /// * advances the same D-056 retune generation, so the accumulator reset
    ///   fires for all four.
    ///
    /// Four entry points is four chances to drift, so the convergence is
    /// asserted rather than assumed.
    #[test]
    fn all_four_entry_points_converge_on_one_retune_path() {
        let n = 512;
        let asked = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = radio_pipeline(n, FakeRadio::new(n, asked.clone()));
        wait_until("the radio to stream", || {
            pipeline.meta().center_freq_hz == Some(RADIO_START_HZ)
        });

        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        let mut status = None;
        let hud = test_hud();
        // Frame 1 places the widgets (the slider and the buttons stash their
        // rects as they draw).
        tune_frame(
            &ctx,
            &mut view,
            &hud,
            &tune_ui(&pipeline, &pipeline.meta(), None),
            Vec::new(),
        );

        let mut generations = Vec::new();
        for entry in ["box", "slider", "button", "key"] {
            let tune = tune_ui(&pipeline, &pipeline.meta(), status.clone());
            let center = tune.center_hz.expect("the radio reports a centre");
            let before_generation = pipeline.retune_generation();
            let before_asked = asked.lock().unwrap().len();

            let response = match entry {
                "box" => {
                    ctx.memory_mut(|m| m.request_focus(phosphene_render::chrome::tune_box_id()));
                    let r = tune_frame(
                        &ctx,
                        &mut view,
                        &hud,
                        &tune,
                        vec![
                            key_event(egui::Key::A, egui::Modifiers::COMMAND),
                            egui::Event::Text("300M".into()),
                            key_event(egui::Key::Enter, egui::Modifiers::default()),
                        ],
                    );
                    ctx.memory_mut(|m| m.surrender_focus(phosphene_render::chrome::tune_box_id()));
                    r
                }
                "slider" => {
                    let rect = phosphene_render::chrome::tune_slider_rect(&ctx)
                        .expect("the tunable source must have been given a slider");
                    let grab = rect.center();
                    let release = egui::pos2(grab.x + 12.0, grab.y);
                    tune_frame(
                        &ctx,
                        &mut view,
                        &hud,
                        &tune,
                        vec![
                            egui::Event::PointerMoved(grab),
                            egui::Event::PointerButton {
                                pos: grab,
                                button: egui::PointerButton::Primary,
                                pressed: true,
                                modifiers: egui::Modifiers::default(),
                            },
                            egui::Event::PointerMoved(release),
                        ],
                    );
                    tune_frame(
                        &ctx,
                        &mut view,
                        &hud,
                        &tune,
                        vec![egui::Event::PointerButton {
                            pos: release,
                            button: egui::PointerButton::Primary,
                            pressed: false,
                            modifiers: egui::Modifiers::default(),
                        }],
                    )
                }
                "button" => click_tune_step(&ctx, &mut view, &hud, &tune, true),
                // D-065 §1: the BARE arrow is the global tune key now.
                _ => tune_frame(
                    &ctx,
                    &mut view,
                    &hud,
                    &tune,
                    vec![key_event(egui::Key::ArrowLeft, egui::Modifiers::default())],
                ),
            };

            // 1. One seam: whatever the user touched, it asked here.
            let target = response
                .retune_hz
                .unwrap_or_else(|| panic!("the {entry} raised no retune request"));
            assert_ne!(target, center, "the {entry} asked for a no-op");

            // 2. One consumer, and it is the frame loop's.
            let readback = consume_retune_request(&pipeline, &response, &mut status)
                .unwrap_or_else(|| panic!("the {entry} request was not consumed"))
                .unwrap_or_else(|e| panic!("the radio refused the {entry} retune: {e}"));

            // 3. It reached the source, as `Control::CenterFreqHz`.
            let after_asked = asked.lock().unwrap();
            assert_eq!(
                after_asked.len(),
                before_asked + 1,
                "the {entry} did not reach the device"
            );
            assert_eq!(*after_asked.last().unwrap(), target);
            drop(after_asked);

            // 4. It crossed a D-056 retune boundary, so the accumulators are
            //    reset for this entry point exactly as for the others.
            wait_until("the retune boundary to be crossed", || {
                pipeline.retune_generation() != before_generation
            });
            generations.push(pipeline.retune_generation());
            wait_until("the readback to reach the metadata", || {
                pipeline.meta().center_freq_hz == Some(readback)
            });
            apply_meta(&mut view.params, &pipeline.meta());
            assert_eq!(
                view.params.center,
                Some(readback),
                "the {entry} left the axis somewhere the radio is not"
            );
        }

        // Every entry point advanced the same counter, in order: one
        // mechanism, four doors.
        assert_eq!(generations.len(), 4);
        for pair in generations.windows(2) {
            assert!(pair[1] > pair[0], "generations went {generations:?}");
        }
        assert_eq!(asked.lock().unwrap().len(), 4);
        pipeline.shutdown();
    }

    /// **D-060 still holds through a new entry point.**
    ///
    /// The rule is about what a retune *means*, not about how it was asked
    /// for — so a step button that mutates the tuner and then cannot read it
    /// back must reset the accumulators anyway and end the stream, exactly as
    /// the box does. This drives the REAL button on a radio whose readback
    /// has been broken mid-stream.
    #[test]
    fn a_step_button_tune_that_cannot_be_read_back_still_resets_and_ends_the_stream() {
        let n = 512;
        let asked = Arc::new(Mutex::new(Vec::new()));
        let blind = Arc::new(AtomicBool::new(false));
        let mut radio = FakeRadio::new(n, asked.clone());
        radio.readback_fails = blind.clone();
        let mut pipeline = radio_pipeline(n, radio);
        let live = vec![DBFS_FLOOR; n];
        let mut intensity = Vec::new();
        let mut max_hold = Vec::new();
        wait_until("the persistence grid to light up at the tone", || {
            let (bins, levels) = pipeline.intensity_frame(1.0 / 60.0, -110.0, 0.0, &mut intensity);
            bins == n && hot_cells(&intensity, levels) > 0
        });

        let ctx = egui::Context::default();
        Theme::default().install(&ctx);
        let mut view = ViewState::default();
        let mut status = None;
        let hud = test_hud();
        let tune = tune_ui(&pipeline, &pipeline.meta(), None);
        tune_frame(&ctx, &mut view, &hud, &tune, Vec::new());

        // Break the readback, then click the real button.
        let seen = pipeline.retune_generation();
        blind.store(true, Ordering::Relaxed);
        let response = click_tune_step(&ctx, &mut view, &hud, &tune, true);
        assert_eq!(response.retune_hz, Some(RADIO_START_HZ + RADIO_RATE));
        let err = consume_retune_request(&pipeline, &response, &mut status)
            .expect("consumed")
            .expect_err("a tune that cannot be read back is not a success");
        assert!(
            err.contains("applied") && err.contains("read"),
            "the error must name BOTH facts - applied, and unread: {err}"
        );
        assert_eq!(
            asked.lock().unwrap().as_slice(),
            &[RADIO_START_HZ + RADIO_RATE],
            "the tuner was mutated: that is what makes the old centre a lie"
        );
        assert!(!status.expect("the chrome is told").ok);

        // 1. The accumulators are reset anyway.
        wait_until("the forced reset to be applied", || {
            pipeline.retune_generation() != seen
        });
        // Sustained, not one-shot — see the same check in
        // `a_tune_that_cannot_be_read_back_resets_everything_and_ends_the_stream`:
        // a straggler batch queued deeper in the ring than one drain pass can
        // carry must never be folded back in once the generation says the
        // reset is done, and a single read right after the flip could win
        // that race by luck.
        for _ in 0..100 {
            let (_, levels) = pipeline.intensity_frame(1.0 / 60.0, -110.0, 0.0, &mut intensity);
            let survivors = hot_cells(&intensity, levels);
            assert_eq!(
                survivors, 0,
                "{survivors} persistence cells carry history from a button \
                 tune the radio took but would not confirm — a pre-retune \
                 frame landed after the generation said the reset was done"
            );
            pipeline.max_hold_frame(0.0, MaxHoldMode::PureHold, &live, &mut max_hold);
            let peak = max_hold.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                peak <= DBFS_FLOOR + 1.0,
                "the max hold holds {peak} dBFS from before the button tune \
                 — a pre-retune frame landed after the generation said the \
                 reset was done"
            );
            // Paces the observation window (D-094): the condition asserted
            // above is what this loop waits on, not this sleep.
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        // 2. The stream ends with a real error.
        wait_until("the stream to end", || pipeline.source_error().is_some());
        let fatal = pipeline.source_error().expect("the stream must fail");
        assert!(
            fatal.contains("applied") && fatal.contains("read") && fatal.contains("fake-radio"),
            "the stream error must name the device and both facts: {fatal}"
        );

        // 3. The display stops claiming a centre it cannot substantiate, so
        //    the step buttons have nothing left to step from either.
        let meta = pipeline.meta();
        assert_eq!(meta.center_freq_hz, None);
        let tune = tune_ui(&pipeline, &meta, None);
        assert_eq!(tune.center_hz, None);
        let response = click_tune_step(&ctx, &mut view, &hud, &tune, true);
        assert_eq!(
            response.retune_hz, None,
            "a step from a centre nothing can substantiate is a guess"
        );
        pipeline.shutdown();
    }

    /// **A rejected retune is a real error, and the display does not move.**
    ///
    /// Two refusals, both through the production path: a source that cannot
    /// tune at all (a file replay), and a device that refuses the request.
    /// In both, the error names the problem and the axis stays exactly where
    /// the samples say it is.
    #[test]
    fn a_refused_retune_is_an_error_and_the_axis_does_not_move() {
        // 1. A source with `tune: false` — refused before any channel exists.
        let n = 512;
        let bytes = crate::pipeline::tone_then_silence_bytes(n, 2, 0);
        let mut config = FileSourceConfig::new(
            Input::Path(std::path::PathBuf::from("in-memory.cf32")),
            IqFormat::Cf32,
        );
        config.sample_rate_hz = Some(RADIO_RATE);
        config.center_freq_hz = Some(RADIO_START_HZ);
        let source =
            FileSource::from_reader(Box::new(std::io::Cursor::new(bytes)), config).unwrap();
        let mut pipeline = Pipeline::start(
            Box::new(source),
            PipelineConfig {
                fft_size: n,
                window: phosphene_core::WindowKind::Hann,
                paced: false,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();
        assert!(!pipeline.caps().tune, "a file has no device to retune");

        let mut view = ViewState::default();
        apply_meta(&mut view.params, &pipeline.meta());
        let before = view.params.center;

        // The chrome offers nothing to click, which is the first half of the
        // refusal…
        let tune = tune_ui(&pipeline, &pipeline.meta(), None);
        assert!(!tune.tunable);

        // …and the path refuses in as many words if it is called anyway.
        let response = ChromeResponse {
            retune_hz: Some(751e6),
            ..quiet_response()
        };
        let mut status = None;
        let err = consume_retune_request(&pipeline, &response, &mut status)
            .expect("consumed")
            .unwrap_err();
        assert!(
            err.contains("cannot be tuned") && err.contains("file"),
            "the refusal must name the source and why: {err}"
        );
        apply_meta(&mut view.params, &pipeline.meta());
        assert_eq!(view.params.center, before, "the axis must not have moved");
        let status = status.expect("a refusal is surfaced, never silent");
        assert!(!status.ok && status.message == err);
        pipeline.shutdown();

        // 2. A real device that refuses the frequency.
        let asked = Arc::new(Mutex::new(Vec::new()));
        let mut radio = FakeRadio::new(n, asked.clone());
        radio.refuse = true;
        let mut pipeline = radio_pipeline(n, radio);
        wait_until("the radio to stream", || {
            pipeline.meta().center_freq_hz == Some(RADIO_START_HZ)
        });
        let mut view = ViewState::default();
        apply_meta(&mut view.params, &pipeline.meta());
        let generation = pipeline.retune_generation();

        let response = ChromeResponse {
            retune_hz: Some(751e6),
            ..quiet_response()
        };
        let mut status = None;
        let err = consume_retune_request(&pipeline, &response, &mut status)
            .expect("consumed")
            .unwrap_err();
        assert!(err.contains("refused"), "the device's own words: {err}");
        assert_eq!(&*asked.lock().unwrap(), &[751e6], "it really was asked");

        // Nothing moved and nothing was reset: the radio is where it was.
        apply_meta(&mut view.params, &pipeline.meta());
        assert_eq!(view.params.center, Some(RADIO_START_HZ));
        assert_eq!(pipeline.retune_generation(), generation);
        let tune = tune_ui(&pipeline, &pipeline.meta(), status.clone());
        assert_eq!(tune.center_hz, Some(RADIO_START_HZ));
        assert!(!status.unwrap().ok);
        pipeline.shutdown();
    }

    /// A [`ChromeResponse`] asking for nothing, for the seals that build one
    /// by hand to exercise the consumer's refusal paths.
    fn quiet_response() -> ChromeResponse {
        ChromeResponse {
            grid: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(64.0, 64.0)),
            requests: ChromeRequests::default(),
            screenshot: false,
            retune_hz: None,
        }
    }

    /// **D-056's honesty requirement, and the thing a plausible
    /// implementation would silently skip.**
    ///
    /// The persistence grid, the max-hold trace and the waterfall are filled
    /// with a recognisable tone; the radio is retuned; and **no cell and no
    /// row survives**. The fake radio emits that tone only at its opening
    /// centre and silence at every other, so anything left above the floor
    /// afterwards is history measured at one frequency being drawn against
    /// another — exactly the plausible lie D-056 forbids.
    ///
    /// The render-side half is asserted too: the rows already uploaded to the
    /// composer's ring texture are cleared by the production
    /// `consume_waterfall_reset`, because the pipeline cannot reach them.
    #[test]
    fn a_retune_leaves_no_cell_and_no_row_from_before_it() {
        let n = 512;
        let (device, queue) = gpu("retune-reset");
        let mut composer = FrameComposer::new(&device, phosphene_render::OFFSCREEN_FORMAT);
        let asked = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = radio_pipeline(n, FakeRadio::new(n, asked.clone()));
        let view = ViewState::default();
        let mut rows = Vec::new();
        let mut intensity = Vec::new();
        let mut max_hold = Vec::new();
        let live = vec![DBFS_FLOOR; n];

        // --- Fill everything with the tone. ---
        wait_until("the persistence grid to light up at the tone", || {
            let (bins, levels) = pipeline.intensity_frame(1.0 / 60.0, -110.0, 0.0, &mut intensity);
            bins == n && hot_cells(&intensity, levels) > 0
        });
        pipeline.max_hold_frame(0.0, MaxHoldMode::PureHold, &live, &mut max_hold);
        let (tone_bin, tone_dbfs) =
            max_hold
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
                    if v > best.1 {
                        (i, v)
                    } else {
                        best
                    }
                });
        assert!(
            (tone_dbfs - RADIO_TONE_DBFS).abs() < 3.0,
            "test premise: the max hold must be holding the −20 dBFS tone, \
             got {tone_dbfs} dBFS at bin {tone_bin}"
        );
        let (_, levels) = pipeline.intensity_frame(1.0 / 60.0, -110.0, 0.0, &mut intensity);
        assert!(
            intensity[tone_bin * levels + level_of(tone_dbfs, levels)] > 0.0,
            "test premise: the tone must be in the persistence grid at bin {tone_bin}"
        );
        // Enough rows to fill the panel the probe below reads, so the tone
        // column is unmistakably on screen before the retune.
        wait_until("the waterfall to fill with the tone", || {
            waterfall_frame_step(
                &pipeline,
                &mut composer,
                &device,
                &queue,
                &view,
                256,
                &mut rows,
            );
            composer.waterfall_rows_written() >= 256
        });
        assert!(
            composer.waterfall_rows_written() > 0,
            "test premise: the waterfall has history"
        );
        let before = waterfall_column_brightness(&device, &queue, &mut composer, &view, n);
        assert!(
            before.hot > before.cold + 30,
            "test premise: the tone column must be visible on the composited \
             waterfall (hot {} vs cold {})",
            before.hot,
            before.cold
        );

        // --- Retune, through the production consumer. ---
        let mut seen = pipeline.retune_generation();
        let response = ChromeResponse {
            retune_hz: Some(751e6),
            ..quiet_response()
        };
        let mut status = None;
        consume_retune_request(&pipeline, &response, &mut status)
            .expect("consumed")
            .expect("the radio accepted the retune");
        wait_until("the reset to be applied at the retune boundary", || {
            pipeline.retune_generation() != seen
        });

        // --- Nothing from before survives. ---
        // Persistence: the radio now emits silence, which lands in the bottom
        // level of every bin. Any cell above the bottom band is pre-retune
        // history that was not discarded.
        let (_, levels) = pipeline.intensity_frame(1.0 / 60.0, -110.0, 0.0, &mut intensity);
        let survivors = hot_cells(&intensity, levels);
        assert_eq!(
            survivors, 0,
            "{survivors} persistence cells survived the retune — history at \
             the old centre is being drawn against the new axis"
        );

        // Max hold: reset to the floor and re-seeded from silence. Read in
        // pure-hold mode so a surviving peak could not have decayed away.
        pipeline.max_hold_frame(0.0, MaxHoldMode::PureHold, &live, &mut max_hold);
        let peak = max_hold.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            peak <= DBFS_FLOOR + 1.0,
            "the max hold still holds {peak} dBFS from before the retune"
        );

        // Waterfall, render side: the rows the composer already held are gone.
        assert!(
            consume_waterfall_reset(&pipeline, &mut composer, &device, &queue, false, &mut seen,),
            "the frame loop must clear the composer's waterfall on a retune"
        );
        // The composited panel is the proof: before the retune the tone
        // column was visibly hotter than the rest of the waterfall; after it,
        // no column stands out, because no row from before it survives.
        let after = waterfall_column_brightness(&device, &queue, &mut composer, &view, n);
        assert!(
            after.hot <= after.cold + 8,
            "the tone column is still {} brighter than its neighbours on the \
             composited waterfall — rows measured at the old centre survived \
             the retune",
            after.hot as i64 - after.cold as i64
        );

        // Waterfall, pipeline side: every row produced from here on is
        // silence — no tone row was left pending across the boundary.
        for _ in 0..200 {
            waterfall_frame_step(
                &pipeline,
                &mut composer,
                &device,
                &queue,
                &view,
                256,
                &mut rows,
            );
            let hottest = rows.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                rows.is_empty() || hottest < -100.0,
                "a waterfall row carrying {hottest} dBFS crossed the retune"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        // And D-013 is untouched: a retune is not a drop.
        let health = pipeline.health();
        assert_eq!(
            health.dropped_batches, 0,
            "a retune must not be counted as a drop (D-013)"
        );
        pipeline.shutdown();
    }

    /// **D-060: a tune that succeeds but cannot be read back.**
    ///
    /// The state no success path can produce, and the one a green seal
    /// missed: the mutation lands, the readback fails, and the old guard —
    /// *"did the metadata change?"* — is silently told "no" by the very
    /// failure it was meant to catch. The radio is at the new frequency; the
    /// axis, the persistence grid, the max hold and the waterfall are all
    /// still the old one.
    ///
    /// Three things are asserted, and a test that checked only the first
    /// would not have caught the bug:
    ///
    /// 1. **The accumulators are reset anyway** — no cell, no held peak, no
    ///    row from before survives. Their contents went stale the moment the
    ///    hardware moved, whatever the readback did.
    /// 2. **The stream ends with a real error naming both facts** — the tune
    ///    was applied, the readback failed — rather than continuing under a
    ///    centre nothing can substantiate.
    /// 3. **The display stops claiming a centre at all.** The rate, label and
    ///    provenance are still true and survive; `center_freq_hz` does not,
    ///    so the axis falls back to relative labelling instead of naming a
    ///    frequency the radio has left.
    #[test]
    fn a_tune_that_cannot_be_read_back_resets_everything_and_ends_the_stream() {
        let n = 512;
        let asked = Arc::new(Mutex::new(Vec::new()));
        let blind = Arc::new(AtomicBool::new(false));
        let mut radio = FakeRadio::new(n, asked.clone());
        radio.readback_fails = blind.clone();
        let mut pipeline = radio_pipeline(n, radio);
        let cfg = wf_config_for(&ViewState::default(), 256);
        let live = vec![DBFS_FLOOR; n];
        let mut intensity = Vec::new();
        let mut max_hold = Vec::new();

        // --- Fill everything with the tone at the opening centre. ---
        //
        // Waited on directly, not via the persistence grid as a proxy for it:
        // nothing orders "the grid has lit up" before "max hold holds the
        // tone" — they are two different accumulators fed by the same
        // spectra, on different schedules — so a wait on one is not a wait
        // on the other. Waiting on the exact condition this then asserts is
        // what makes the assertion meaningful the first time it runs.
        wait_until(
            "persistence and max hold to both reflect the opening tone",
            || {
                let (bins, levels) =
                    pipeline.intensity_frame(1.0 / 60.0, -110.0, 0.0, &mut intensity);
                pipeline.max_hold_frame(0.0, MaxHoldMode::PureHold, &live, &mut max_hold);
                let peak = max_hold.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                bins == n
                    && hot_cells(&intensity, levels) > 0
                    && (peak - RADIO_TONE_DBFS).abs() < 3.0
            },
        );
        let peak = max_hold.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (peak - RADIO_TONE_DBFS).abs() < 3.0,
            "test premise: the max hold must be holding the −20 dBFS tone, got {peak} dBFS"
        );
        let (rows, _) = collect_rows(&pipeline, &cfg, n, 4);
        let hottest_before = rows.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            hottest_before > -60.0,
            "test premise: the waterfall must carry the tone before the \
             retune, hottest row sample was {hottest_before} dBFS"
        );
        // Creates the stimulus (D-094): builds the backlog the defect needs
        // — leaves more rows in flight, undrained, so the reset has pending
        // history to discard and not merely an empty ring to rebuild.
        std::thread::sleep(std::time::Duration::from_millis(50));

        // --- Break the readback, then retune through the production path. ---
        let seen = pipeline.retune_generation();
        blind.store(true, Ordering::Relaxed);
        let response = ChromeResponse {
            retune_hz: Some(751e6),
            ..quiet_response()
        };
        let mut status = None;
        let err = consume_retune_request(&pipeline, &response, &mut status)
            .expect("consumed")
            .expect_err("a tune that cannot be read back is not a success");
        assert!(
            err.contains("applied") && err.contains("read"),
            "the error must name BOTH facts — applied, and unread: {err}"
        );
        assert_eq!(
            asked.lock().unwrap().as_slice(),
            &[751e6],
            "the tuner was mutated: that is what makes the old centre a lie"
        );
        assert!(!status.expect("the chrome is told").ok);

        // --- 1. The accumulators are reset anyway (D-060 rule 1). ---
        wait_until("the forced reset to be applied", || {
            pipeline.retune_generation() != seen
        });
        // Sustained, not one-shot: a straggler batch queued deeper in the
        // ring than one drain pass can carry is still tagged with the
        // pre-retune epoch, and must never be folded back into the display
        // accumulators once the generation says the reset is done — no
        // reader may observe the new generation together with pre-retune
        // state (D-056/D-060). A single read taken right after the
        // generation flips could win the race against that straggler by
        // luck and miss the exact window this is guarding; polling for a
        // while after the generation change is a wait on the invariant
        // holding, not a fixed sleep standing in for it.
        let mut scratch = Vec::new();
        let mut trace = Vec::new();
        for _ in 0..100 {
            let (_, levels) = pipeline.intensity_frame(1.0 / 60.0, -110.0, 0.0, &mut intensity);
            let survivors = hot_cells(&intensity, levels);
            assert_eq!(
                survivors, 0,
                "{survivors} persistence cells carry history at a centre the \
                 hardware has left — a pre-retune frame landed after the \
                 generation said the reset was done"
            );
            pipeline.max_hold_frame(0.0, MaxHoldMode::PureHold, &live, &mut max_hold);
            let peak = max_hold.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                peak <= DBFS_FLOOR + 1.0,
                "the max hold holds {peak} dBFS at a centre the hardware has \
                 left — a pre-retune frame landed after the generation said \
                 the reset was done"
            );
            // The §7.4 live trace (D-056's fourth named accumulator) is fed
            // by the same `fold_run` as persistence, max hold and the
            // waterfall, so it is exposed to exactly the same straggler-fold
            // hazard — checked here because nothing else in this lane reads
            // it after a D-060 reset.
            pipeline.copy_trace(&mut trace);
            let live_peak = trace.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                live_peak <= DBFS_FLOOR + 1.0,
                "the live trace holds {live_peak} dBFS at a centre the \
                 hardware has left — a pre-retune frame landed after the \
                 generation said the reset was done"
            );
            let drain = pipeline.waterfall_frame(&cfg, &mut scratch);
            let hottest_after = scratch.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                drain.rows == 0 || hottest_after < -100.0,
                "a waterfall row carrying {hottest_after} dBFS survived the \
                 tune — a pre-retune frame landed after the generation said \
                 the reset was done"
            );
            // Paces the observation window (D-094): the conditions asserted
            // above are what this loop waits on, not this sleep.
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        // --- 2. The stream ends, with an error naming the situation. ---
        wait_until("the stream to end", || pipeline.source_error().is_some());
        let fatal = pipeline.source_error().expect("the stream must fail");
        assert!(
            fatal.contains("applied") && fatal.contains("read") && fatal.contains("fake-radio"),
            "the stream error must name the device and both facts: {fatal}"
        );
        assert!(
            !pipeline.source_finished(),
            "this is a failure, not a clean end of stream"
        );

        // --- 3. The display stops claiming a centre it cannot substantiate. ---
        let meta = pipeline.meta();
        assert_eq!(
            meta.center_freq_hz, None,
            "the axis must not keep asserting a centre the radio has left"
        );
        assert_eq!(
            meta.sample_rate_hz,
            Some(RADIO_RATE),
            "the rate is still true and must survive"
        );
        let mut params = DisplayParams::default();
        apply_meta(&mut params, &meta);
        assert_eq!(params.center, None, "and it reaches the display that way");
        let tune = tune_ui(&pipeline, &meta, None);
        assert_eq!(
            tune.center_hz, None,
            "the D-058 box and slider follow the readback, and there is none"
        );

        // And D-013 is untouched: an unverified retune is still not a drop.
        assert_eq!(
            pipeline.health().dropped_batches,
            0,
            "a retune, verified or not, must not be counted as a drop (D-013)"
        );
        pipeline.shutdown();
    }

    /// The level index the §7.3 grid puts `dbfs` in, for the display window
    /// these seals use (−110…0 dBFS).
    fn level_of(dbfs: f32, levels: usize) -> usize {
        let raw = ((dbfs + 110.0) / 110.0 * (levels - 1) as f32).round();
        (raw as usize).min(levels - 1)
    }

    /// How many persistence cells sit **above the bottom band** — i.e. carry
    /// something other than "silence". Silence lands in level 0; the tone and
    /// its window sidelobes land far above it.
    fn hot_cells(intensity: &[f32], levels: usize) -> usize {
        intensity
            .chunks_exact(levels)
            .flat_map(|bin| bin.iter().enumerate().skip(levels / 2))
            .filter(|(_, &v)| v > 0.0)
            .count()
    }

    /// **D-011 still holds after this lane, structurally.**
    ///
    /// D-011 ruled v1 zoom display-side only, and gave the reason: a device
    /// retune driven from the display "would couple the display path to
    /// device control — the one coupling §8.1's crate boundaries exist to
    /// prevent". This lane adds a retune, so that reasoning has to be shown
    /// still true rather than asserted.
    ///
    /// It is true by construction: **`phosphene-render` does not depend on
    /// `phosphene-sources`**, so no code in the display crate — zoom
    /// included — can so much as *name* `Control` or `SampleSource::set`. The
    /// widgets D-058 adds return a number the app applies; they cannot apply
    /// anything themselves. This test reads the dependency graph rather than
    /// trusting the claim, so a future edit that reached for the coupling
    /// would fail here before it could compile a retune into the zoom path.
    #[test]
    fn the_display_crate_cannot_reach_device_control() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("phosphene-render/Cargo.toml");
        let text = std::fs::read_to_string(&manifest)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", manifest.display()));
        assert!(
            !text.contains("phosphene-sources"),
            "phosphene-render now depends on phosphene-sources: the display \
             path can reach device control, which is exactly the coupling \
             D-011 and §8.1 exist to prevent"
        );

        // The app-side half: the ONE place a retune is applied is
        // `Pipeline::retune`, and the only thing that calls it is the
        // consumer of the chrome's retune request. Zoom state never reaches
        // it — a zoomed display asks the device for nothing.
        let window = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/window.rs"),
        )
        .unwrap();
        let production = window
            .split("#[cfg(test)]")
            .next()
            .expect("window.rs has a production half");
        assert_eq!(
            production.matches(".retune(").count(),
            1,
            "there must be exactly one call to Pipeline::retune in the frame \
             loop — a second entry point is a second source of truth (D-058)"
        );
        assert!(
            !production.contains("zoom") || !production.contains("retune("),
            "the frame loop must not derive a retune from zoom state (D-011)"
        );
    }

    /// **Hardware validation (D-031's "prove it in the product", with a real
    /// radio).** CI has no SDR, so this is `#[ignore]`d and run by hand:
    ///
    /// ```text
    /// cargo test -p phosphene-app --features phosphene-sources/soapy \
    ///     -- --ignored --nocapture live_retune_moves_a_real_radio
    /// ```
    ///
    /// It retunes a real device between two known emitters while it streams
    /// and shows the **spectrum actually changes** — because a retune that
    /// returns `Ok(())` and leaves the radio where it was passes every unit
    /// test in this file. It asserts three things a no-op could not fake:
    /// the device's readback moved, D-056's reset fired at the boundary, and
    /// the received spectrum at the second centre is materially different
    /// from the first.
    #[test]
    #[ignore = "needs a real SDR and the `soapy` feature; run with --ignored"]
    fn live_retune_moves_a_real_radio() {
        use phosphene_sources::soapy::SoapySourceConfig;

        const FM_HZ: f64 = 94.9e6;
        const LTE_HZ: f64 = 751e6;
        let n = 1024;
        let mut config = SoapySourceConfig::new("driver=uhd");
        config.sample_rate_hz = Some(2e6);
        config.center_freq_hz = Some(FM_HZ);
        config.gain_db = Some(40.0);
        let source = match crate::pipeline::open_source(&SourceDesc::Soapy(config)) {
            Ok(source) => source,
            Err(e) => {
                eprintln!("SKIPPED: no SoapySDR device available ({e})");
                return;
            }
        };
        let mut pipeline = Pipeline::start(
            source,
            PipelineConfig {
                fft_size: n,
                window: phosphene_core::WindowKind::Hann,
                paced: true,
                backpressure: false,
                db_bottom: -110.0,
                db_top: 0.0,
            },
        )
        .unwrap();

        let caps = pipeline.caps();
        assert!(caps.tune, "a SoapySDR device must advertise tuning");
        let range = caps.tune_range.expect("the device must report its range");
        eprintln!(
            "device range: {:.3} MHz .. {:.3} MHz",
            range.min_hz / 1e6,
            range.max_hz / 1e6
        );
        assert!(range.contains(FM_HZ) && range.contains(LTE_HZ));

        let settle = |pipeline: &Pipeline| {
            std::thread::sleep(std::time::Duration::from_millis(1500));
            let mut trace = Vec::new();
            assert!(pipeline.copy_trace(&mut trace), "no spectrum published");
            trace
        };

        let fm = settle(&pipeline);
        let fm_center = pipeline.meta().center_freq_hz.unwrap();
        eprintln!(
            "at {:.3} MHz: peak {:.1} dBFS, mean {:.1} dBFS",
            fm_center / 1e6,
            peak(&fm),
            mean(&fm)
        );

        let generation = pipeline.retune_generation();
        let readback = pipeline.retune(LTE_HZ).expect("the retune must succeed");
        eprintln!("retune asked {LTE_HZ:.0} Hz, device answered {readback:.0} Hz");
        wait_until("the retune reset to be applied", || {
            pipeline.retune_generation() != generation
        });
        wait_until("the new centre to reach the metadata", || {
            pipeline.meta().center_freq_hz == Some(readback)
        });

        let lte = settle(&pipeline);
        eprintln!(
            "at {:.3} MHz: peak {:.1} dBFS, mean {:.1} dBFS",
            pipeline.meta().center_freq_hz.unwrap() / 1e6,
            peak(&lte),
            mean(&lte)
        );

        // 1. The radio moved, and the display followed the READBACK.
        assert!((readback - LTE_HZ).abs() < 1e3, "readback {readback} Hz");
        assert_ne!(fm_center, readback);
        // 2. D-056's reset fired at the boundary.
        assert_ne!(pipeline.retune_generation(), generation);
        // 3. The spectrum really changed — this is the assertion a retune
        //    that returned Ok() and left the radio in place would fail.
        let moved = fm
            .iter()
            .zip(&lte)
            .filter(|(a, b)| (*a - *b).abs() > 3.0)
            .count();
        eprintln!("{moved}/{n} bins moved by more than 3 dB");
        assert!(
            moved > n / 10,
            "only {moved}/{n} bins changed between {FM_HZ} Hz and {LTE_HZ} Hz — \
             the radio may not have actually retuned"
        );
        pipeline.shutdown();
    }

    fn peak(trace: &[f32]) -> f32 {
        trace.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    fn mean(trace: &[f32]) -> f32 {
        trace.iter().sum::<f32>() / trace.len() as f32
    }

    /// Brightness of the tone column versus a quiet column on the
    /// **composited** waterfall — the same RTSA frame the window renders,
    /// at split 0 so the panel fills the target.
    struct ColumnBrightness {
        hot: u32,
        cold: u32,
    }

    fn waterfall_column_brightness(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        composer: &mut FrameComposer,
        view: &ViewState,
        n: usize,
    ) -> ColumnBrightness {
        let (w, h) = (400u32, 300u32);
        let target = OffscreenTarget::with_format(device, w, h, phosphene_render::OFFSCREEN_FORMAT);
        let theme = Theme::default();
        let mut wf_view = *view;
        wf_view.split = 0.0;
        composer.render_rtsa_frame(
            device,
            queue,
            &target.view,
            [w, h],
            RtsaInput {
                live_bins: &[],
                max_hold_bins: &[],
                intensity: &[],
                intensity_bins: 0,
                intensity_levels: 0,
                surface_points: egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(w as f32, h as f32),
                ),
                view: &wf_view,
                theme: &theme,
            },
            phosphene_render::EguiFrame {
                primitives: Vec::new(),
                textures_delta: egui::TexturesDelta::default(),
                pixels_per_point: 1.0,
            },
        );
        let rgba = target.read_rgba(device, queue);
        let px = |x: u32, y: u32| {
            let o = ((y * w + x) * 4) as usize;
            u32::from(rgba[o]) + u32::from(rgba[o + 1]) + u32::from(rgba[o + 2])
        };
        let bin = phosphene_core::fftshift_index(n / 4, n);
        let hot_x = ((bin as f32 / (n - 1) as f32) * w as f32) as u32;
        ColumnBrightness {
            hot: px(hot_x, 5),
            cold: px(hot_x / 2, 5),
        }
    }
}
