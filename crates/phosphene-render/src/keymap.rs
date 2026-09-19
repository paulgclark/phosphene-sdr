// SPDX-License-Identifier: MIT

//! The FR-D7 keyboard map: one table binds the keys, and the `?` help
//! overlay is generated from that same table — structurally, a binding
//! cannot exist that the overlay does not list.
//!
//! Actions either mutate the caller-owned [`ViewState`] directly (reference
//! level, dB/div, split, zoom reset, pause, help) or accumulate into
//! [`ChromeRequests`] for state the [`crate::gpu::FrameComposer`] owns
//! (colormap cycle — routed to M1-B's `cycle_colormap`, which guarantees the
//! D-009 no-rebuild swap — and the §7.3 persistence gamma). The screenshot
//! binding surfaces as a request on the chrome response for the app to act
//! on (FR-D12 wiring, outside this crate).

use egui::Key;

use crate::layout::ViewState;
use crate::zoom::ZoomSpan;

/// Reference-level step per keypress, dB (FR-D7 Up/Down).
pub const REF_LEVEL_STEP_DB: f32 = 5.0;

/// The dB/div values Left/Right cycle through (FR-D7).
pub const DB_PER_DIV_STEPS: [f32; 5] = [1.0, 2.0, 5.0, 10.0, 20.0];

/// Split-ratio step per keypress (FR-D4 ratio, FR-D7 binding).
pub const SPLIT_STEP: f32 = 0.05;

/// Multiplicative step per **intensity** keypress — §7.3's persistence
/// `gamma_adjust` (`PgUp`/`PgDn`) and §7.6's waterfall `γ_wf`
/// (`Shift+PgUp`/`Shift+PgDn`).
///
/// One step size for both, deliberately: they are the same power law on two
/// surfaces, and D-035 already settled that a second idiom for the same kind
/// of control is a liability, not a feature.
pub const GAMMA_STEP: f32 = 1.1;

/// Multiplicative step per τ keypress, applied by the app to the §7.3
/// persistence decay τ (FR-D1's "persistence time" control), the D-007
/// max-hold decay τ (D-035: both time constants exposed together), and the
/// §7.4 live-trace averaging τ (FR-D2, M1-J — same stepping, not a third
/// idiom).
///
/// M2-B (D-043) reconsidered this once the value became visible on the
/// status bar: at the previous 1.25× it took ~3 presses to halve or double,
/// which read as sluggish even with the number moving in front of you. 1.4×
/// halves/doubles in ~2 presses while still covering the pipeline's full
/// τ range (0.05–30 s) in under 20 presses.
pub const TAU_STEP: f32 = 1.4;

/// Multiplicative step per waterfall `-`/`=` keypress. The keys adjust
/// whichever quantity the active §7.6 speed mode owns (D-046): the panel
/// time span in traditional mode, the row interval in fast mode — and, for
/// a rate-less source, their spectra-based counterparts (D-054).
pub const WF_SPAN_STEP: f32 = 1.25;

/// Waterfall time-span bounds, seconds (traditional mode, rated source):
/// generous around the 30 s default, but held-down keys can never drive the
/// cadence degenerate.
const WF_SPAN_MIN_S: f32 = 1.0;
const WF_SPAN_MAX_S: f32 = 600.0;

/// Fast-mode row-interval bounds, seconds (rated source). The lower bound
/// sits below any realistic spectrum interval — the D-046 clamp to one
/// spectrum interval is applied where the cadence is derived, since only
/// the feed knows the rate; this is just the keyboard's own runaway guard.
const WF_FAST_MIN_S: f32 = 1e-5;
const WF_FAST_MAX_S: f32 = 1.0;

/// Traditional-mode span bounds for a rate-less source, in rows of spectra
/// (D-054: the span is a row count; nothing is derived from an absent rate).
const WF_SPAN_ROWS_MIN: f32 = 64.0;
const WF_SPAN_ROWS_MAX: f32 = 65536.0;

/// Everything a frame's input asks of the [`crate::gpu::FrameComposer`],
/// applied by [`crate::gpu::FrameComposer::apply_requests`] — plus the
/// FR-D3 max-hold reset, which the composer ignores: like the FR-D12
/// screenshot, it is app wiring (the accumulator lives app-side), consumed
/// by the frame loop from the same [`crate::chrome::ChromeResponse`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ChromeRequests {
    /// Cycle the colormap (D-009) — forwarded to `cycle_colormap`.
    pub cycle_colormap: bool,
    /// Signed count of persistence-intensity steps: negative lowers the
    /// gamma exponent (stronger persistence emphasis), positive raises it.
    pub gamma_steps: i32,
    /// Reset the FR-D3 max-hold trace. App-owned: the frame loop forwards
    /// it to the pipeline's accumulator, which re-seeds from the next
    /// spectrum.
    pub reset_max_hold: bool,
    /// Signed count of §7.3 persistence decay-τ steps (FR-D1's
    /// "persistence time" control, D-035): negative shortens the phosphor
    /// decay, positive lengthens it, [`TAU_STEP`] per step. App-owned — the
    /// accumulator lives in the pipeline.
    pub persistence_tau_steps: i32,
    /// Signed count of D-007 max-hold decay-τ steps (D-035): negative
    /// shortens the decay toward the live trace, positive lengthens it,
    /// [`TAU_STEP`] per step. App-owned.
    pub max_hold_tau_steps: i32,
    /// Signed count of §7.4 live-trace averaging-τ steps (FR-D2 "averaging
    /// time configurable", M1-J): negative shortens the averaging, positive
    /// lengthens it, [`TAU_STEP`] per step. App-owned — the EMA lives on the
    /// pipeline's compute thread.
    pub live_tau_steps: i32,
    /// Signed count of **window steps** the D-062 global tune keys asked for:
    /// negative steps the centre down, positive up, one sample rate — one
    /// window width — per step (D-058's increment, unchanged).
    ///
    /// Unlike every other field here this one never leaves the chrome: it is
    /// a count, and only the chrome holds the device's readback and reported
    /// range that turn a count into a frequency. [`crate::chrome::draw_tunable`]
    /// consumes it into the single [`crate::chrome::ChromeResponse::retune_hz`]
    /// the box, the slider and the step buttons all produce, and zeroes it —
    /// so all four entry points converge before the response is even built,
    /// and there is no second route to the radio to drift from the first.
    pub tune_steps: i32,
    /// Signed count of **fine** tune steps the D-066 §2 `Shift+←`/`Shift+→`
    /// keys asked for: one sample rate ÷ 10 per step, resolved by the chrome
    /// exactly as [`Self::tune_steps`] is and converging on the same single
    /// [`crate::chrome::ChromeResponse::retune_hz`].
    ///
    /// A separate count rather than a fractional one, because the two steps
    /// are clamped differently at the edge of the device's range: a coarse
    /// step is clamped in **whole windows** so tiling survives the edge, and
    /// a fine step — which never tiled — is simply clamped to the range.
    pub fine_tune_steps: i32,
    /// The `I` key asked to toggle the FR-AU1 analysis pipeline on/off
    /// (annotation design §2.7). App-owned: render has no analysis pipeline
    /// of its own to toggle (spec §8.1), so this is a one-shot request the
    /// app resolves against whatever it currently holds true/false, exactly
    /// like `reset_max_hold`.
    pub toggle_inspector: bool,
}

/// One keyboard action (FR-D7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Reference level up ([`REF_LEVEL_STEP_DB`]).
    RefLevelUp,
    /// Reference level down.
    RefLevelDown,
    /// Finer dB/div (previous entry of [`DB_PER_DIV_STEPS`]). On `O`
    /// since **D-066 §2** took the arrows away from it entirely: D-065 §1
    /// had already moved it from the bare arrow to `Shift+←`, and use then
    /// gave the shifted pair to fine tuning. Both halves of the horizontal
    /// axis belong to the radio now.
    DbPerDivFiner,
    /// Coarser dB/div (next entry), on `P`.
    DbPerDivCoarser,
    /// Stronger persistence emphasis (gamma exponent down).
    PersistenceUp,
    /// Weaker persistence emphasis (gamma exponent up).
    PersistenceDown,
    /// **More** §7.6 waterfall intensity: `γ_wf` down by [`GAMMA_STEP`], so
    /// low-power detail lifts out of the floor. View-owned (D-065 §3).
    WaterfallIntensityUp,
    /// **Less** waterfall intensity: `γ_wf` up, so only the strongest
    /// returns keep colour.
    WaterfallIntensityDown,
    /// Shorter §7.3 persistence decay τ (FR-D1 "persistence time", D-035).
    PersistenceTauShorter,
    /// Longer persistence decay τ.
    PersistenceTauLonger,
    /// Shorter D-007 max-hold decay τ (D-035).
    MaxHoldTauShorter,
    /// Longer max-hold decay τ.
    MaxHoldTauLonger,
    /// Shorter §7.4 live-trace averaging τ (FR-D2, M1-J).
    LiveTauShorter,
    /// Longer live-trace averaging τ.
    LiveTauLonger,
    /// Shift the FR-D4 split toward the waterfall.
    SplitMoreWaterfall,
    /// Shift the FR-D4 split toward the histogram.
    SplitMoreHistogram,
    /// Finer waterfall time ([`WF_SPAN_STEP`] per press): shortens whichever
    /// quantity the active §7.6 speed mode owns — the time span in
    /// traditional mode, the row interval in fast mode (D-046) — or their
    /// spectra-based counterparts for a rate-less source (D-054).
    WaterfallSpanShorter,
    /// Coarser waterfall time: lengthens the mode-owned quantity.
    WaterfallSpanLonger,
    /// Switch the §7.6 speed mode (D-046): traditional (set the span) or
    /// fast (set the row interval).
    ToggleWaterfallMode,
    /// Switch the §7.6 row aggregation: max (default — transients survive)
    /// or mean.
    ToggleWaterfallAggregation,
    /// Toggle the §7.6 waterfall independent auto-range: on, the colour
    /// scale spans the retained rows; off, it shares the histogram's dB
    /// range.
    ToggleWaterfallAutoRange,
    /// Cycle the colormap (D-009).
    CycleColormap,
    /// Show/hide the FR-D3 max-hold trace.
    ToggleMaxHold,
    /// Switch the D-007 max-hold mode: decay-toward-live (the default) or
    /// pure hold.
    ToggleMaxHoldMode,
    /// Reset the FR-D3 max-hold trace (surfaced to the app, which owns the
    /// accumulator).
    ResetMaxHold,
    /// Toggle the FR-D12 display freeze (the source keeps draining).
    TogglePause,
    /// Request a screenshot (FR-D12; surfaced to the app).
    Screenshot,
    /// Reset the FR-D6 zoom to the full span.
    ResetZoom,
    /// Toggle the help overlay.
    ToggleHelp,
    /// Step the centre frequency **down** by one sample rate — one window
    /// width (D-058's increment, D-062's global key, on the **bare** `←`
    /// since D-065 §1). Accumulates into [`ChromeRequests::tune_steps`]; the
    /// chrome turns the count into the one retune request the box, the
    /// slider and the buttons also produce.
    TuneDown,
    /// Step the centre frequency **up** by one window width.
    TuneUp,
    /// Tune **down** by one *fine* step — one sample rate ÷ 10 (D-066 §2),
    /// on `Shift+←`. Accumulates into [`ChromeRequests::fine_tune_steps`],
    /// which the chrome resolves against the same readback the coarse count
    /// uses.
    ///
    /// **The fine step deliberately does not tile** (D-066 §2): D-058's
    /// 1×span increment exists so adjacent *coarse* positions abut with no
    /// gap and no overlap, and this step is for centring a signal inside the
    /// window instead. Do not "fix" it to preserve tiling.
    TuneDownFine,
    /// Tune **up** by one fine step (sample rate ÷ 10), on `Shift+→`.
    TuneUpFine,
    /// Toggle the FR-AU1 analysis pipeline on/off (annotation design §2.7).
    /// Render has no analysis pipeline of its own (spec §8.1 crate
    /// boundaries) — like `retune_hz`, this surfaces as a request on
    /// [`ChromeRequests`] for the app to act on, never a state render owns.
    ToggleInspector,
    /// Toggle the annotation level (§2.7): full (labels + shade) ↔
    /// shade-only (bands + anchor ticks, labels hidden, detail-on-click
    /// unaffected). A two-state toggle, matching FR-AU2's "dial down"
    /// wording exactly.
    ToggleAnnotationLevel,
    /// Step the declutter cap N **down** (§2.4/§2.7), bounds `1..=32`.
    DeclutterCapDown,
    /// Step the declutter cap N **up**.
    DeclutterCapUp,
}

/// The modifier a binding requires (D-062 §2).
///
/// `None` is the default and covers every binding FR-D7 shipped with; `Shift`
/// exists because two actions want the same horizontal axis — tuning and
/// dB/div — and only one of them can have the bare key.
///
/// **D-066 §2 decided which, and it is the radio both times.** M3-D gave the
/// bare `←`/`→` to dB/div; D-065 §1 swapped them, on the owner's
/// frequency-of-use argument — tuning is constant on a live radio, dB/div is
/// set once, and the common action gets the unmodified key. Use then went one
/// step further: the shifted pair is **fine tuning** (±rate ÷ 10) and dB/div
/// is off the arrows altogether, on `O`/`P`. So the whole horizontal axis is
/// the radio's now, coarse bare and fine shifted, and the modifier means
/// *smaller step* rather than *different quantity*.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Modifier {
    /// No modifier. See [`binding_for`] for what "no modifier" means on a
    /// layout where the glyph itself needs one.
    #[default]
    None,
    /// Shift held.
    Shift,
}

/// One key binding: the key, the modifier it needs, its action, and the
/// help-overlay description.
#[derive(Debug, Clone, Copy)]
pub struct Binding {
    /// The bound key. Its display name in the overlay comes from
    /// [`Key::name`], so the overlay cannot drift from the binding.
    pub key: Key,
    /// The modifier the key needs. Built through [`Binding::new`] /
    /// [`Binding::shift`] rather than written per entry, so a binding cannot
    /// be added without deciding this — and the default is [`Modifier::None`].
    pub modifier: Modifier,
    /// What the key does.
    pub action: Action,
    /// Help-overlay description.
    pub what: &'static str,
}

impl Binding {
    /// A binding on the bare key.
    const fn new(key: Key, action: Action, what: &'static str) -> Binding {
        Binding {
            key,
            modifier: Modifier::None,
            action,
            what,
        }
    }

    /// A binding that needs Shift.
    const fn shift(key: Key, action: Action, what: &'static str) -> Binding {
        Binding {
            key,
            modifier: Modifier::Shift,
            action,
            what,
        }
    }
}

/// The keyboard map (FR-D7) — the single table both the input handler and
/// the help overlay read.
pub const BINDINGS: &[Binding] = &[
    Binding::new(
        Key::ArrowUp,
        Action::RefLevelUp,
        "reference level up (+5 dB)",
    ),
    Binding::new(
        Key::ArrowDown,
        Action::RefLevelDown,
        "reference level down (-5 dB)",
    ),
    Binding::new(Key::ArrowLeft, Action::TuneDown, "tune down one window"),
    Binding::new(Key::ArrowRight, Action::TuneUp, "tune up one window"),
    Binding::shift(
        Key::ArrowLeft,
        Action::TuneDownFine,
        "fine tune down (1/10 window)",
    ),
    Binding::shift(
        Key::ArrowRight,
        Action::TuneUpFine,
        "fine tune up (1/10 window)",
    ),
    Binding::new(
        Key::PageUp,
        Action::PersistenceUp,
        "persistence emphasis up",
    ),
    Binding::new(
        Key::PageDown,
        Action::PersistenceDown,
        "persistence emphasis down",
    ),
    Binding::shift(
        Key::PageUp,
        Action::WaterfallIntensityUp,
        "waterfall intensity up",
    ),
    Binding::shift(
        Key::PageDown,
        Action::WaterfallIntensityDown,
        "waterfall intensity down",
    ),
    Binding::new(
        Key::Comma,
        Action::PersistenceTauShorter,
        "persistence decay shorter",
    ),
    Binding::new(
        Key::Period,
        Action::PersistenceTauLonger,
        "persistence decay longer",
    ),
    Binding::new(
        Key::K,
        Action::LiveTauShorter,
        "live trace averaging shorter",
    ),
    Binding::new(Key::L, Action::LiveTauLonger, "live trace averaging longer"),
    Binding::new(
        Key::OpenBracket,
        Action::SplitMoreWaterfall,
        "split: more waterfall",
    ),
    Binding::new(
        Key::CloseBracket,
        Action::SplitMoreHistogram,
        "split: more histogram",
    ),
    Binding::new(
        Key::Minus,
        Action::WaterfallSpanShorter,
        "waterfall finer (span / row interval)",
    ),
    Binding::new(
        Key::Equals,
        Action::WaterfallSpanLonger,
        "waterfall coarser (span / row interval)",
    ),
    Binding::new(
        Key::F,
        Action::ToggleWaterfallMode,
        "waterfall mode: traditional / fast",
    ),
    Binding::new(
        Key::A,
        Action::ToggleWaterfallAggregation,
        "waterfall rows: max / mean",
    ),
    Binding::new(
        Key::W,
        Action::ToggleWaterfallAutoRange,
        "waterfall auto-range on/off",
    ),
    Binding::new(Key::C, Action::CycleColormap, "cycle colormap"),
    Binding::new(Key::M, Action::ToggleMaxHold, "max hold on/off"),
    Binding::new(
        Key::D,
        Action::ToggleMaxHoldMode,
        "max hold: decay / pure hold",
    ),
    Binding::new(Key::R, Action::ResetMaxHold, "reset max hold"),
    Binding::new(
        Key::Semicolon,
        Action::MaxHoldTauShorter,
        "max hold decay shorter",
    ),
    Binding::new(
        Key::Quote,
        Action::MaxHoldTauLonger,
        "max hold decay longer",
    ),
    Binding::new(Key::Space, Action::TogglePause, "pause / resume display"),
    Binding::new(Key::S, Action::Screenshot, "screenshot"),
    Binding::new(Key::Z, Action::ResetZoom, "reset zoom"),
    Binding::new(Key::Questionmark, Action::ToggleHelp, "toggle this help"),
    // **dB/div, off the arrows (D-066 §2).** The master's recommendation was
    // `Shift+PgUp`/`Shift+PgDn`, and that pair is **not free**: D-065 §3 gave
    // it to the waterfall intensity earlier in this same lane, and D-066
    // leaves the choice to the lane "with justification". `O`/`P` is an
    // adjacent free pair, which is the idiom every other stepped control in
    // this table already uses — `,`/`.`, `[`/`]`, `-`/`=`, `;`/`'`, `K`/`L`,
    // each with the finer/shorter action on the left key. dB/div now reads
    // the same way.
    Binding::new(Key::O, Action::DbPerDivFiner, "finer dB/div"),
    Binding::new(Key::P, Action::DbPerDivCoarser, "coarser dB/div"),
    // Annotation design §2.7: four free keys, chosen against the exhaustive
    // audit of the table above — I, N, G, H were unbound.
    Binding::new(
        Key::I,
        Action::ToggleInspector,
        "toggle analysis (inspector)",
    ),
    Binding::new(
        Key::N,
        Action::ToggleAnnotationLevel,
        "annotation level: full / shade-only",
    ),
    Binding::new(Key::G, Action::DeclutterCapDown, "declutter cap down"),
    Binding::new(Key::H, Action::DeclutterCapUp, "declutter cap up"),
];

/// Display name for a bound key in the help overlay — [`Key::name`] with the
/// punctuation and arrow keys shortened to the glyph on the keycap. Derived
/// from the binding's own key, never written by hand, so it cannot drift from
/// what is actually bound.
///
/// Every glyph here is in JetBrains Mono, the one face the chrome installs
/// (there is no fallback font, so a missing glyph would render as tofu).
pub fn key_label(key: Key) -> &'static str {
    match key {
        Key::Questionmark => "?",
        Key::OpenBracket => "[",
        Key::CloseBracket => "]",
        Key::Comma => ",",
        Key::Period => ".",
        Key::Semicolon => ";",
        Key::Quote => "'",
        Key::Minus => "-",
        Key::Equals => "=",
        Key::ArrowUp => "\u{2191}",
        Key::ArrowDown => "\u{2193}",
        Key::ArrowLeft => "\u{2190}",
        Key::ArrowRight => "\u{2192}",
        _ => key.name(),
    }
}

/// How the help overlay names one binding: the modifier, then the key.
///
/// Rendered from the binding itself rather than written beside it, which is
/// what keeps D-062's requirement structural — the overlay is generated from
/// the table, so a modifier the table holds is a modifier the overlay shows,
/// and there is no second place for the two to disagree.
pub fn binding_label(binding: &Binding) -> String {
    match binding.modifier {
        Modifier::None => key_label(binding.key).to_owned(),
        Modifier::Shift => format!("shift+{}", key_label(binding.key)),
    }
}

/// The binding one key event fires, or `None` when the event is not bound.
///
/// Two rules, both of which fall out of the table rather than out of a list
/// of special cases:
///
/// * **No binding uses ctrl, alt or command**, so an event carrying one is
///   not ours — `Cmd+A` in the frequency box must not also reset max hold.
/// * **Shift is significant exactly where the table makes it so.** A shifted
///   event prefers a [`Modifier::Shift`] binding on that key, and falls back
///   to the bare one only when the table has nothing shifted for it. That
///   fallback is not laxity: on many layouts shift is how the glyph is
///   produced at all (`?` is Shift+`/` on a US keyboard, and egui reports the
///   shift with it), so a bare binding that demanded shift be absent would be
///   unreachable there. Because the fallback is taken only in the absence of
///   a shifted binding, `Shift+←` can never also fire the bare `←` — which is
///   the collision D-062 flagged, and which D-065 §1's swap of the two does
///   not reopen: the rule is about the *chord*, not about which action sits
///   on which side of it.
pub fn binding_for(key: Key, modifiers: egui::Modifiers) -> Option<&'static Binding> {
    if modifiers.alt || modifiers.ctrl || modifiers.command || modifiers.mac_cmd {
        return None;
    }
    let wanted = if modifiers.shift {
        Modifier::Shift
    } else {
        Modifier::None
    };
    let exact = BINDINGS
        .iter()
        .find(|b| b.key == key && b.modifier == wanted);
    if exact.is_some() || wanted == Modifier::None {
        return exact;
    }
    BINDINGS
        .iter()
        .find(|b| b.key == key && b.modifier == Modifier::None)
}

/// The binding that fires `action`, for chrome that wants to name the key a
/// button duplicates. Reads the same table, so a tooltip cannot cite a
/// binding that does not exist.
pub fn binding_for_action(action: Action) -> Option<&'static Binding> {
    BINDINGS.iter().find(|b| b.action == action)
}

/// Apply one action. View-owned state mutates in place; composer-owned and
/// app-owned effects accumulate into `requests` / the returned screenshot
/// flag. Returns `true` when the action requested a screenshot.
pub fn apply(action: Action, view: &mut ViewState, requests: &mut ChromeRequests) -> bool {
    let p = &mut view.params;
    match action {
        Action::RefLevelUp => shift_ref_level(p, REF_LEVEL_STEP_DB),
        Action::RefLevelDown => shift_ref_level(p, -REF_LEVEL_STEP_DB),
        Action::DbPerDivFiner => step_db_per_div(p, -1),
        Action::DbPerDivCoarser => step_db_per_div(p, 1),
        Action::PersistenceUp => requests.gamma_steps -= 1,
        Action::PersistenceDown => requests.gamma_steps += 1,
        // View-owned, like every other §7.6 waterfall control: the frame
        // loop mirrors it onto the composer, and the status readout reads
        // the same field, so the number on the bar cannot disagree with the
        // mapping on screen.
        Action::WaterfallIntensityUp => step_wf_gamma(view, 1.0 / GAMMA_STEP),
        Action::WaterfallIntensityDown => step_wf_gamma(view, GAMMA_STEP),
        Action::PersistenceTauShorter => requests.persistence_tau_steps -= 1,
        Action::PersistenceTauLonger => requests.persistence_tau_steps += 1,
        Action::MaxHoldTauShorter => requests.max_hold_tau_steps -= 1,
        Action::MaxHoldTauLonger => requests.max_hold_tau_steps += 1,
        Action::LiveTauShorter => requests.live_tau_steps -= 1,
        Action::LiveTauLonger => requests.live_tau_steps += 1,
        Action::SplitMoreWaterfall => view.split = (view.split - SPLIT_STEP).clamp(0.0, 1.0),
        Action::SplitMoreHistogram => view.split = (view.split + SPLIT_STEP).clamp(0.0, 1.0),
        Action::WaterfallSpanShorter => step_wf_time(view, 1.0 / WF_SPAN_STEP),
        Action::WaterfallSpanLonger => step_wf_time(view, WF_SPAN_STEP),
        Action::ToggleWaterfallMode => {
            view.wf_mode = match view.wf_mode {
                crate::waterfall::WaterfallMode::Traditional => {
                    crate::waterfall::WaterfallMode::Fast
                }
                crate::waterfall::WaterfallMode::Fast => {
                    crate::waterfall::WaterfallMode::Traditional
                }
            }
        }
        Action::ToggleWaterfallAggregation => {
            view.wf_aggregation = match view.wf_aggregation {
                crate::waterfall::RowAggregation::Max => crate::waterfall::RowAggregation::Mean,
                crate::waterfall::RowAggregation::Mean => crate::waterfall::RowAggregation::Max,
            }
        }
        Action::ToggleWaterfallAutoRange => view.wf_auto_range = !view.wf_auto_range,
        Action::CycleColormap => requests.cycle_colormap = true,
        Action::ToggleMaxHold => view.max_hold = !view.max_hold,
        Action::ToggleMaxHoldMode => view.max_hold_decay = !view.max_hold_decay,
        Action::ResetMaxHold => requests.reset_max_hold = true,
        Action::TogglePause => view.paused = !view.paused,
        Action::Screenshot => return true,
        Action::ResetZoom => p.zoom = ZoomSpan::FULL,
        Action::ToggleHelp => view.help_open = !view.help_open,
        Action::TuneDown => requests.tune_steps -= 1,
        Action::TuneUp => requests.tune_steps += 1,
        Action::TuneDownFine => requests.fine_tune_steps -= 1,
        Action::TuneUpFine => requests.fine_tune_steps += 1,
        Action::ToggleInspector => requests.toggle_inspector = true,
        Action::ToggleAnnotationLevel => {
            view.annotation_level = view.annotation_level.toggled();
        }
        Action::DeclutterCapDown => {
            view.declutter_cap = view.declutter_cap.saturating_sub(1).max(DECLUTTER_CAP_MIN)
        }
        Action::DeclutterCapUp => {
            view.declutter_cap = (view.declutter_cap + 1).min(DECLUTTER_CAP_MAX)
        }
    }
    false
}

/// Declutter-cap bounds (§2.7): `1..=32`.
const DECLUTTER_CAP_MIN: u32 = 1;
const DECLUTTER_CAP_MAX: u32 = 32;

/// Reference-level bounds, dBFS: generous, but the scale can never run away.
const REF_LEVEL_MAX: f32 = 50.0;
const SCALE_BOTTOM_MIN: f32 = -250.0;

fn shift_ref_level(p: &mut crate::layout::DisplayParams, delta: f32) {
    let top = p.db_top + delta;
    let bottom = p.db_bottom + delta;
    if top <= REF_LEVEL_MAX && bottom >= SCALE_BOTTOM_MIN {
        p.db_top = top;
        p.db_bottom = bottom;
    }
}

/// The `-`/`=` step (D-046): adjust whichever quantity the active §7.6 mode
/// owns, in the unit the source's rate makes honest (D-054). Fast mode over
/// a rate-less source is pinned to one row per spectrum — the finest
/// resolution its only clock can state — so the keys have nothing to adjust
/// there.
fn step_wf_time(view: &mut ViewState, factor: f32) {
    use crate::waterfall::WaterfallMode;

    let rated = view.params.sample_rate.is_some();
    match (view.wf_mode, rated) {
        (WaterfallMode::Traditional, true) => {
            view.wf_time_span_s =
                (view.wf_time_span_s * factor).clamp(WF_SPAN_MIN_S, WF_SPAN_MAX_S);
        }
        (WaterfallMode::Fast, true) => {
            view.wf_fast_interval_s =
                (view.wf_fast_interval_s * factor).clamp(WF_FAST_MIN_S, WF_FAST_MAX_S);
        }
        (WaterfallMode::Traditional, false) => {
            view.wf_span_rows =
                (view.wf_span_rows * factor).clamp(WF_SPAN_ROWS_MIN, WF_SPAN_ROWS_MAX);
        }
        (WaterfallMode::Fast, false) => {}
    }
}

/// Step the §7.6 waterfall intensity exponent, clamped to the same range
/// the composer clamps either gamma to — so the readout on the status bar
/// and the exponent the shader receives are the same number at the ends of
/// the range as well as in the middle.
fn step_wf_gamma(view: &mut ViewState, factor: f32) {
    view.wf_gamma =
        (view.wf_gamma * factor).clamp(crate::surface::GAMMA_MIN, crate::surface::GAMMA_MAX);
}

/// The division count the derived scale aims for — FR-D5's "~10 grid
/// divisions", and D-065 §5's exactly-10 at the default range.
///
/// Aiming at the nominal count rather than at whatever the display currently
/// shows is what makes the control **reversible across the ceiling clamp**:
/// coarsening to 20 dB/div at a −100 floor loses divisions to the clamp
/// below, and re-deriving from the surviving five would strand the user
/// there when they pressed finer again. The count shown is the count that
/// fits; the count the next press starts from is the one FR-D5 names.
const DB_DIVS_NOMINAL: f32 = 10.0;

/// The highest the derived ceiling may be, dBFS. **Full scale, and this is
/// unconditional** — nothing in a dBFS display can exceed it, so a ceiling
/// above it is never a scale, only empty screen above the highest value any
/// signal can take (D-066 §1).
const DB_TOP_CEILING: f32 = 0.0;

/// **D-066 §1: dB/div pins the floor and moves the top.**
///
/// The owner found the control useless because it did the opposite — the
/// code pinned `db_top` and slid `db_bottom` out from under the noise floor,
/// which is where the eye is. So the floor stays put and the ceiling is
/// derived, which makes finer dB/div **zoom toward the floor**: the useful
/// direction. This supersedes the previous behaviour, which was a deliberate
/// choice the other way and is recorded as such in D-066.
///
/// The ceiling needs a rule of its own, because the naive derivation
/// produces nonsense: ten divisions of 20 dB above a −100 floor is
/// `db_top = +100 dBFS`. So the derived ceiling is **clamped at 0 dBFS,
/// unconditionally**, and the division count falls out of what fits — five,
/// there. Fewer divisions at a coarse setting is the honest outcome; empty
/// screen above full scale is not.
///
/// **The clamp is not conditional on where the ceiling already was.** An
/// earlier revision of this function preserved a `db_top` that was already
/// positive, on the reasoning that FR-D7's `↑` is allowed to park the
/// reference level above full scale. That is wrong for *this* path: a
/// positive ceiling is not a scale whoever put it there. `shift_ref_level`
/// is a different control — it slides both edges together and is free to
/// move the window — and this rule governs the dB/div path only.
fn step_db_per_div(p: &mut crate::layout::DisplayParams, dir: i32) {
    let idx = DB_PER_DIV_STEPS
        .iter()
        .position(|&s| (s - p.db_per_div).abs() < 0.5)
        .unwrap_or(3); // nearest to the 10 dB default if off-list
    let new_idx = (idx as i32 + dir).clamp(0, DB_PER_DIV_STEPS.len() as i32 - 1) as usize;
    let new = DB_PER_DIV_STEPS[new_idx];
    if (new - p.db_per_div).abs() < f32::EPSILON {
        return;
    }
    p.db_per_div = new;
    p.db_top = (p.db_bottom + DB_DIVS_NOMINAL * new).min(DB_TOP_CEILING);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::ViewState;

    #[test]
    fn every_binding_has_a_unique_chord_and_a_description() {
        for (i, a) in BINDINGS.iter().enumerate() {
            assert!(!a.what.is_empty());
            for b in &BINDINGS[i + 1..] {
                assert!(
                    a.key != b.key || a.modifier != b.modifier,
                    "chord {:?}+{:?} bound twice",
                    a.modifier,
                    a.key
                );
            }
        }
    }

    /// **D-066 §2's assignment, asserted in the direction that can fail.**
    /// The whole horizontal axis is the radio's: the bare `←`/`→` tune by one
    /// window (coarse), `Shift+←`/`Shift+→` tune by one tenth of one (fine),
    /// `↑`/`↓` are still the reference level, and dB/div is **off the arrows
    /// entirely**, on `O`/`P`.
    ///
    /// The assertions are deliberately two-sided. A test that only checked
    /// "the bare arrow is bound to a tune" would pass just as happily with
    /// **both** chords bound to the *same* tune, which is the exact way this
    /// remap goes wrong — the two steps differ only in size, so a collision
    /// here is invisible in a one-sided check. So each chord is asserted to
    /// produce its own step and **not** the other's, and dB/div is asserted
    /// to be reachable *and* to be absent from every arrow chord.
    ///
    /// Also the layout trap this rule exists to avoid: `?` is Shift+`/` on a
    /// US keyboard, so egui reports the help key *with* shift held. A rule
    /// that simply demanded "no shift" for an unmodified binding would make
    /// the help overlay unreachable on those layouts — which is why the
    /// fallback is keyed off the table having nothing shifted for that key,
    /// not off a list of exceptions.
    #[test]
    fn shift_selects_the_fine_tune_and_db_per_div_is_off_the_arrows() {
        let shift = egui::Modifiers::SHIFT;
        let none = egui::Modifiers::default();

        for (key, bare, shifted) in [
            (Key::ArrowLeft, Action::TuneDown, Action::TuneDownFine),
            (Key::ArrowRight, Action::TuneUp, Action::TuneUpFine),
        ] {
            assert_eq!(
                binding_for(key, none).unwrap().action,
                bare,
                "D-066 §2: the bare arrow tunes coarse"
            );
            assert_eq!(
                binding_for(key, shift).unwrap().action,
                shifted,
                "D-066 §2: the shifted arrow tunes fine"
            );
            assert_ne!(
                binding_for(key, none).unwrap().action,
                binding_for(key, shift).unwrap().action,
                "both chords on {key:?} do the same thing - the remap is half-done"
            );
        }

        // The same statement made through `apply`, where the effect lives:
        // each chord raises its OWN step count and neither raises the
        // other's, and no arrow chord moves dB/div any more.
        for (key, delta) in [(Key::ArrowRight, 1), (Key::ArrowLeft, -1)] {
            let mut v = ViewState::default();
            let mut r = ChromeRequests::default();
            let before = v.params.db_per_div;

            apply(binding_for(key, none).unwrap().action, &mut v, &mut r);
            assert_eq!(r.tune_steps, delta, "bare {key:?} must tune coarse");
            assert_eq!(r.fine_tune_steps, 0, "bare {key:?} also tuned fine");
            assert_eq!(
                v.params.db_per_div, before,
                "bare {key:?} moved dB/div - it is off the arrows now"
            );

            let mut v = ViewState::default();
            let mut r = ChromeRequests::default();
            apply(binding_for(key, shift).unwrap().action, &mut v, &mut r);
            assert_eq!(r.fine_tune_steps, delta, "shift+{key:?} must tune fine");
            assert_eq!(r.tune_steps, 0, "shift+{key:?} also tuned coarse");
            assert_eq!(
                v.params.db_per_div, before,
                "shift+{key:?} moved dB/div - it is off the arrows now"
            );
        }

        // dB/div is still reachable, on its own free adjacent pair, and asks
        // the radio for nothing.
        for (key, dir) in [(Key::P, 1.0f32), (Key::O, -1.0)] {
            let mut v = ViewState::default();
            let mut r = ChromeRequests::default();
            let before = v.params.db_per_div;
            apply(binding_for(key, none).unwrap().action, &mut v, &mut r);
            let moved = v.params.db_per_div - before;
            assert!(
                moved.signum() == dir && moved != 0.0,
                "{key:?} must step dB/div {} (now {})",
                if dir > 0.0 { "coarser" } else { "finer" },
                v.params.db_per_div
            );
            assert_eq!(
                r,
                ChromeRequests::default(),
                "{key:?} asked the radio for a step"
            );
        }

        // A key with nothing shifted in the table keeps working with shift
        // held — the `?` case, and every letter typed on a layout that needs
        // shift for its glyph.
        assert_eq!(
            binding_for(Key::Questionmark, shift).unwrap().action,
            Action::ToggleHelp
        );
        assert_eq!(
            binding_for(Key::ArrowUp, shift).unwrap().action,
            Action::RefLevelUp
        );

        // No binding uses ctrl, alt or command, so an event carrying one is
        // not ours: Cmd+A in the frequency box must not toggle the waterfall
        // aggregation on its way through.
        for m in [
            egui::Modifiers::COMMAND,
            egui::Modifiers::CTRL,
            egui::Modifiers::ALT,
            egui::Modifiers::MAC_CMD,
        ] {
            assert!(binding_for(Key::A, m).is_none(), "{m:?} leaked");
            assert!(binding_for(Key::ArrowLeft, m).is_none(), "{m:?} leaked");
        }

        // An unbound key is unbound.
        assert!(binding_for(Key::B, none).is_none());
    }

    /// The tune keys accumulate a signed **count of windows**, and nothing
    /// else: the render crate holds no device, so a count is all it can
    /// honestly produce here (the chrome turns it into a frequency using the
    /// readback it was handed).
    #[test]
    fn the_tune_keys_accumulate_window_steps() {
        let mut v = ViewState::default();
        let mut r = ChromeRequests::default();
        assert_eq!(r.tune_steps, 0);
        assert!(!apply(Action::TuneUp, &mut v, &mut r));
        assert!(!apply(Action::TuneUp, &mut v, &mut r));
        assert!(!apply(Action::TuneDown, &mut v, &mut r));
        assert_eq!(r.tune_steps, 1);
        // A tune step is device state, never view state: nothing on the
        // display moved.
        let fresh = ViewState::default();
        assert_eq!(v.params.db_top, fresh.params.db_top);
        assert_eq!(v.params.db_per_div, fresh.params.db_per_div);
        assert_eq!(v.split, fresh.split);
        assert_eq!(v.paused, fresh.paused);
        assert_eq!(
            ChromeRequests { tune_steps: 0, ..r },
            ChromeRequests::default(),
            "the tune keys raise no other request"
        );
        apply(Action::TuneDown, &mut v, &mut r);
        apply(Action::TuneDown, &mut v, &mut r);
        assert_eq!(r.tune_steps, -1);

        // D-066 §2's fine steps count separately, in the same way and on the
        // same view — the chrome resolves the two against one readback.
        let mut v = ViewState::default();
        let mut r = ChromeRequests::default();
        apply(Action::TuneUpFine, &mut v, &mut r);
        apply(Action::TuneUpFine, &mut v, &mut r);
        apply(Action::TuneDownFine, &mut v, &mut r);
        assert_eq!(r.fine_tune_steps, 1);
        assert_eq!(r.tune_steps, 0, "a fine step counted as a whole window");
        assert_eq!(
            ChromeRequests {
                fine_tune_steps: 0,
                ..r
            },
            ChromeRequests::default(),
            "the fine tune keys raise no other request"
        );
        let fresh = ViewState::default();
        assert_eq!(v.params.db_top, fresh.params.db_top);
        assert_eq!(v.params.db_per_div, fresh.params.db_per_div);
    }

    /// The overlay's name for a binding carries the modifier, and comes from
    /// the binding itself — the anti-drift property D-062 asked for.
    #[test]
    fn the_binding_label_shows_the_modifier() {
        // D-066 §2: the modifier is on the fine tune now, and the overlay
        // says so.
        let fine_down = binding_for_action(Action::TuneDownFine).expect("bound");
        let fine_up = binding_for_action(Action::TuneUpFine).expect("bound");
        assert_eq!(binding_label(fine_down), "shift+\u{2190}");
        assert_eq!(binding_label(fine_up), "shift+\u{2192}");
        // An unmodified binding is still just its key — the coarse tune
        // steps, and dB/div on its own pair now that it is off the arrows.
        let down = binding_for_action(Action::TuneDown).expect("bound");
        let up = binding_for_action(Action::TuneUp).expect("bound");
        assert_eq!(binding_label(down), "\u{2190}");
        assert_eq!(binding_label(up), "\u{2192}");
        let finer = binding_for_action(Action::DbPerDivFiner).expect("bound");
        let coarser = binding_for_action(Action::DbPerDivCoarser).expect("bound");
        assert_eq!(binding_label(finer), "O");
        assert_eq!(binding_label(coarser), "P");
        // Every binding renders to something non-empty, and every shifted one
        // says so.
        for b in BINDINGS {
            let label = binding_label(b);
            assert!(!label.is_empty(), "{b:?} has no overlay label");
            assert_eq!(
                label.starts_with("shift+"),
                b.modifier == Modifier::Shift,
                "{b:?} label {label:?} disagrees with its modifier"
            );
        }
    }

    #[test]
    fn ref_level_moves_both_edges_and_stops_at_bounds() {
        let mut v = ViewState::default();
        let mut r = ChromeRequests::default();
        apply(Action::RefLevelUp, &mut v, &mut r);
        assert_eq!(v.params.db_top, 5.0);
        assert_eq!(v.params.db_bottom, -95.0);
        // The range never changes with the reference level — and it is the
        // D-065 §5 range, 100 dB.
        for _ in 0..100 {
            apply(Action::RefLevelUp, &mut v, &mut r);
        }
        assert!(v.params.db_top <= REF_LEVEL_MAX);
        assert_eq!(v.params.db_top - v.params.db_bottom, 100.0);
        for _ in 0..200 {
            apply(Action::RefLevelDown, &mut v, &mut r);
        }
        assert!(v.params.db_bottom >= SCALE_BOTTOM_MIN);
        assert_eq!(v.params.db_top - v.params.db_bottom, 100.0);
    }

    /// **D-066 §1: the floor is pinned and the ceiling is derived.**
    ///
    /// The owner reported the control as useless because it did the reverse —
    /// it held `db_top` and slid `db_bottom`, moving the noise floor out from
    /// under the eye while the resolution changed. So: `db_bottom` never
    /// moves, `db_top` follows, and finer dB/div zooms **toward the floor**.
    #[test]
    fn db_per_div_pins_the_floor_and_derives_the_ceiling() {
        let mut v = ViewState::default();
        let mut r = ChromeRequests::default();
        let floor = v.params.db_bottom;
        assert_eq!(v.params.db_per_div, 10.0);
        assert_eq!(v.params.db_top, 0.0);
        assert_eq!(v.params.db_divs(), 10, "D-065 §5: exactly 10 divisions");

        // Finer: the floor holds, the ceiling comes down, and the division
        // count is preserved — the scale zoomed toward the noise.
        apply(Action::DbPerDivFiner, &mut v, &mut r);
        assert_eq!(v.params.db_per_div, 5.0);
        assert_eq!(v.params.db_bottom, floor, "the floor moved");
        assert_eq!(v.params.db_top, floor + 50.0);
        assert_eq!(v.params.db_divs(), 10);

        // …and all the way down, still pinned.
        for _ in 0..10 {
            apply(Action::DbPerDivFiner, &mut v, &mut r);
        }
        assert_eq!(v.params.db_per_div, 1.0);
        assert_eq!(v.params.db_bottom, floor);
        assert_eq!(v.params.db_top, floor + 10.0);
        assert_eq!(v.params.db_divs(), 10);

        // Back up to the default, exactly reversing.
        for _ in 0..3 {
            apply(Action::DbPerDivCoarser, &mut v, &mut r);
        }
        assert_eq!(v.params.db_per_div, 10.0);
        assert_eq!(v.params.db_bottom, floor);
        assert_eq!(v.params.db_top, 0.0);
        assert_eq!(v.params.db_divs(), 10);

        // **The ceiling clamp.** Ten divisions of 20 dB above a −100 floor
        // would be +100 dBFS — 100 dB above full scale, which no signal can
        // occupy. The ceiling stops at 0 dBFS and the division count falls
        // out of what fits: five.
        apply(Action::DbPerDivCoarser, &mut v, &mut r);
        assert_eq!(v.params.db_per_div, 20.0);
        assert_eq!(v.params.db_bottom, floor, "the floor moved");
        assert_eq!(v.params.db_top, 0.0, "the ceiling ran past full scale");
        assert_eq!(v.params.db_divs(), 5, "the division count must fall out");
        // Saturates at the coarse end of the list.
        apply(Action::DbPerDivCoarser, &mut v, &mut r);
        assert_eq!(v.params.db_per_div, 20.0);
        assert_eq!(v.params.db_top, 0.0);

        // And the clamp is not a trap: pressing finer again returns exactly
        // the scale that was there before, ten divisions and all.
        apply(Action::DbPerDivFiner, &mut v, &mut r);
        assert_eq!(v.params.db_per_div, 10.0);
        assert_eq!(v.params.db_bottom, floor);
        assert_eq!(v.params.db_top, 0.0);
        assert_eq!(v.params.db_divs(), 10);

        assert_eq!(
            r,
            ChromeRequests::default(),
            "dB/div asked the radio for something"
        );
    }

    /// **The clamp is unconditional (D-066 §1, as ruled).** A `db_top` that
    /// is already positive — FR-D7's `↑` may put it there — is *not*
    /// preserved by a dB/div change: nothing in a dBFS display can exceed
    /// full scale, so a positive ceiling is empty screen above the highest
    /// value any signal can take, whoever put it there. The floor stays
    /// pinned, the ceiling is derived, and the derivation stops at 0 dBFS.
    ///
    /// `shift_ref_level` is a different control and keeps its freedom to
    /// move the window: it slides both edges together, which is asserted in
    /// `ref_level_moves_both_edges_and_stops_at_bounds`. This rule is about
    /// the dB/div path only.
    #[test]
    fn a_positive_ceiling_is_clamped_to_full_scale_by_a_db_per_div_step() {
        let mut v = ViewState::default();
        let mut r = ChromeRequests::default();
        apply(Action::RefLevelUp, &mut v, &mut r);
        assert_eq!(v.params.db_top, 5.0, "test premise: a positive ceiling");
        let floor = v.params.db_bottom;
        assert_eq!(floor, -95.0);

        // Coarser would derive +105. It is clamped to full scale — not to
        // the +5 the reference level had parked it at — and the division
        // count falls out of the 95 dB that is left.
        apply(Action::DbPerDivCoarser, &mut v, &mut r);
        assert_eq!(v.params.db_per_div, 20.0);
        assert_eq!(v.params.db_bottom, floor, "the floor moved");
        assert_eq!(v.params.db_top, 0.0, "a positive ceiling was preserved");
        assert_eq!(v.params.db_divs(), 5, "the division count must fall out");

        // The same at 10 dB/div, where the naive derivation lands exactly on
        // the old positive ceiling: +5 is still not a ceiling.
        apply(Action::DbPerDivFiner, &mut v, &mut r);
        assert_eq!(v.params.db_per_div, 10.0);
        assert_eq!(v.params.db_bottom, floor);
        assert_eq!(v.params.db_top, 0.0, "a positive ceiling was preserved");

        // Finer derives a ceiling of its own, below full scale, so the clamp
        // has nothing to say.
        apply(Action::DbPerDivFiner, &mut v, &mut r);
        assert_eq!(v.params.db_per_div, 5.0);
        assert_eq!(v.params.db_bottom, floor);
        assert_eq!(v.params.db_top, floor + 50.0);
        assert_eq!(v.params.db_divs(), 10);

        // And no dB/div press, at any setting reachable from here, ever
        // leaves the ceiling above full scale.
        for _ in 0..8 {
            apply(Action::DbPerDivCoarser, &mut v, &mut r);
            assert!(v.params.db_top <= 0.0, "ceiling {}", v.params.db_top);
            assert_eq!(v.params.db_bottom, floor, "the floor moved");
        }
        for _ in 0..8 {
            apply(Action::DbPerDivFiner, &mut v, &mut r);
            assert!(v.params.db_top <= 0.0, "ceiling {}", v.params.db_top);
            assert_eq!(v.params.db_bottom, floor, "the floor moved");
        }

        assert_eq!(
            r,
            ChromeRequests::default(),
            "dB/div asked the radio for something"
        );
    }

    #[test]
    fn composer_and_app_actions_accumulate_as_requests() {
        let mut v = ViewState::default();
        let mut r = ChromeRequests::default();
        assert!(!apply(Action::CycleColormap, &mut v, &mut r));
        assert!(r.cycle_colormap);
        apply(Action::PersistenceUp, &mut v, &mut r);
        apply(Action::PersistenceUp, &mut v, &mut r);
        apply(Action::PersistenceDown, &mut v, &mut r);
        assert_eq!(r.gamma_steps, -1);
        assert!(!r.reset_max_hold);
        assert!(!apply(Action::ResetMaxHold, &mut v, &mut r));
        assert!(r.reset_max_hold);
        // Both τ controls accumulate signed steps (D-035).
        apply(Action::PersistenceTauShorter, &mut v, &mut r);
        apply(Action::PersistenceTauShorter, &mut v, &mut r);
        apply(Action::PersistenceTauLonger, &mut v, &mut r);
        assert_eq!(r.persistence_tau_steps, -1);
        apply(Action::MaxHoldTauLonger, &mut v, &mut r);
        assert_eq!(r.max_hold_tau_steps, 1);
        // The §7.4 live-trace τ steps the same way (FR-D2, M1-J).
        apply(Action::LiveTauShorter, &mut v, &mut r);
        apply(Action::LiveTauShorter, &mut v, &mut r);
        apply(Action::LiveTauLonger, &mut v, &mut r);
        assert_eq!(r.live_tau_steps, -1);
        assert!(apply(Action::Screenshot, &mut v, &mut r));
    }

    /// M1-J's FR-D4 waterfall controls are view-owned: the time span steps
    /// multiplicatively and clamps, the aggregation toggles max/mean with
    /// max the default (§7.6), and the auto-range toggles off/on with
    /// shared-range the default.
    #[test]
    fn waterfall_span_and_toggles_drive_the_view() {
        use crate::waterfall::RowAggregation;

        let mut v = ViewState::default();
        // A rated source, switched to traditional: that mode owns the
        // seconds span (D-046/D-054; fast is the default).
        v.params.sample_rate = Some(2_048_000.0);
        v.wf_mode = crate::waterfall::WaterfallMode::Traditional;
        let mut r = ChromeRequests::default();
        assert_eq!(v.wf_time_span_s, 30.0);
        apply(Action::WaterfallSpanShorter, &mut v, &mut r);
        assert!((v.wf_time_span_s - 30.0 / WF_SPAN_STEP).abs() < 1e-4);
        apply(Action::WaterfallSpanLonger, &mut v, &mut r);
        assert!((v.wf_time_span_s - 30.0).abs() < 1e-4);
        // Held-down keys clamp at both ends.
        for _ in 0..100 {
            apply(Action::WaterfallSpanShorter, &mut v, &mut r);
        }
        assert_eq!(v.wf_time_span_s, WF_SPAN_MIN_S);
        for _ in 0..100 {
            apply(Action::WaterfallSpanLonger, &mut v, &mut r);
        }
        assert_eq!(v.wf_time_span_s, WF_SPAN_MAX_S);

        assert_eq!(v.wf_aggregation, RowAggregation::Max, "§7.6: max default");
        apply(Action::ToggleWaterfallAggregation, &mut v, &mut r);
        assert_eq!(v.wf_aggregation, RowAggregation::Mean);
        apply(Action::ToggleWaterfallAggregation, &mut v, &mut r);
        assert_eq!(v.wf_aggregation, RowAggregation::Max);

        assert!(v.wf_auto_range, "D-065 §2: auto-range is the default");
        apply(Action::ToggleWaterfallAutoRange, &mut v, &mut r);
        assert!(!v.wf_auto_range);
        apply(Action::ToggleWaterfallAutoRange, &mut v, &mut r);
        assert!(v.wf_auto_range);

        assert_eq!(
            r,
            ChromeRequests::default(),
            "view-owned waterfall controls raise no requests"
        );
    }

    /// D-046: the `-`/`=` keys adjust whichever quantity the active speed
    /// mode owns, and the F toggle switches the mode. D-054: for a rate-less
    /// source the quantities are the spectra-based counterparts, and fast
    /// mode — pinned at one row per spectrum — has nothing to adjust.
    #[test]
    fn span_keys_adjust_the_mode_owned_quantity() {
        use crate::waterfall::{WaterfallMode, FAST_INTERVAL_DEFAULT_S};

        let mut v = ViewState::default();
        v.params.sample_rate = Some(2_048_000.0);
        let mut r = ChromeRequests::default();

        // Fast is the default (D-044/D-046, owner-ruled); the F toggle
        // switches modes both ways.
        assert_eq!(v.wf_mode, WaterfallMode::Fast);
        apply(Action::ToggleWaterfallMode, &mut v, &mut r);
        assert_eq!(v.wf_mode, WaterfallMode::Traditional);
        apply(Action::ToggleWaterfallMode, &mut v, &mut r);
        assert_eq!(v.wf_mode, WaterfallMode::Fast);

        // Fast + rated: the keys own the row interval; the span holds.
        assert_eq!(v.wf_fast_interval_s, FAST_INTERVAL_DEFAULT_S);
        apply(Action::WaterfallSpanShorter, &mut v, &mut r);
        assert!((v.wf_fast_interval_s - FAST_INTERVAL_DEFAULT_S / WF_SPAN_STEP).abs() < 1e-7);
        assert_eq!(v.wf_time_span_s, 30.0, "fast keys must not move the span");
        for _ in 0..100 {
            apply(Action::WaterfallSpanShorter, &mut v, &mut r);
        }
        assert_eq!(v.wf_fast_interval_s, WF_FAST_MIN_S);
        for _ in 0..100 {
            apply(Action::WaterfallSpanLonger, &mut v, &mut r);
        }
        assert_eq!(v.wf_fast_interval_s, WF_FAST_MAX_S);

        // Rate-less + fast: pinned at one row per spectrum — nothing moves.
        v.params.sample_rate = None;
        let before = v;
        apply(Action::WaterfallSpanShorter, &mut v, &mut r);
        apply(Action::WaterfallSpanLonger, &mut v, &mut r);
        assert_eq!(v.wf_fast_interval_s, before.wf_fast_interval_s);
        assert_eq!(v.wf_span_rows, before.wf_span_rows);
        assert_eq!(v.wf_time_span_s, before.wf_time_span_s);

        // Rate-less + traditional: the keys own the span in rows (D-054).
        apply(Action::ToggleWaterfallMode, &mut v, &mut r);
        assert_eq!(v.wf_mode, WaterfallMode::Traditional);
        let rows0 = v.wf_span_rows;
        apply(Action::WaterfallSpanLonger, &mut v, &mut r);
        assert!((v.wf_span_rows - rows0 * WF_SPAN_STEP).abs() < 1e-2);
        assert_eq!(
            v.wf_time_span_s, before.wf_time_span_s,
            "rate-less keys must not touch the seconds span"
        );
        for _ in 0..100 {
            apply(Action::WaterfallSpanShorter, &mut v, &mut r);
        }
        assert_eq!(v.wf_span_rows, WF_SPAN_ROWS_MIN);
        assert_eq!(
            r,
            ChromeRequests::default(),
            "mode and time keys raise no requests"
        );
    }

    /// FR-D3 view toggles: the defaults are shown + decay-toward-live
    /// (D-007), M hides the trace, D switches to pure hold and back.
    #[test]
    fn max_hold_toggles_flip_visibility_and_mode() {
        let mut v = ViewState::default();
        let mut r = ChromeRequests::default();
        assert!(v.max_hold, "FR-D3 trace defaults to shown (spec §2)");
        assert!(v.max_hold_decay, "D-007: decay-toward-live is the default");
        apply(Action::ToggleMaxHold, &mut v, &mut r);
        assert!(!v.max_hold);
        apply(Action::ToggleMaxHold, &mut v, &mut r);
        assert!(v.max_hold);
        apply(Action::ToggleMaxHoldMode, &mut v, &mut r);
        assert!(!v.max_hold_decay, "D switches to pure hold");
        apply(Action::ToggleMaxHoldMode, &mut v, &mut r);
        assert!(v.max_hold_decay);
        assert_eq!(
            r,
            ChromeRequests::default(),
            "view toggles raise no requests"
        );
    }

    /// **D-065 §3's control, at the table.** `Shift+PgUp`/`Shift+PgDn` step
    /// the §7.6 waterfall exponent — view-owned, multiplicative, clamped —
    /// and the **bare** `PgUp`/`PgDn` still drive §7.3's histogram gamma and
    /// only that. The two are distinct quantities on distinct surfaces, so
    /// neither chord may move the other's number.
    #[test]
    fn the_waterfall_intensity_keys_are_their_own_control() {
        let shift = egui::Modifiers::SHIFT;
        let none = egui::Modifiers::default();
        assert_eq!(
            binding_for(Key::PageUp, none).unwrap().action,
            Action::PersistenceUp
        );
        assert_eq!(
            binding_for(Key::PageUp, shift).unwrap().action,
            Action::WaterfallIntensityUp
        );
        assert_eq!(
            binding_for(Key::PageDown, shift).unwrap().action,
            Action::WaterfallIntensityDown
        );

        let mut v = ViewState::default();
        let mut r = ChromeRequests::default();
        assert_eq!(v.wf_gamma, crate::surface::DEFAULT_WF_GAMMA);
        assert_eq!(v.wf_gamma, 1.0, "the default is the linear mapping");

        // Up lifts low-power detail: the exponent goes DOWN one step.
        apply(Action::WaterfallIntensityUp, &mut v, &mut r);
        assert!((v.wf_gamma - 1.0 / GAMMA_STEP).abs() < 1e-6);
        assert_eq!(
            r,
            ChromeRequests::default(),
            "the waterfall exponent is view state, not a composer request"
        );
        apply(Action::WaterfallIntensityDown, &mut v, &mut r);
        assert!((v.wf_gamma - 1.0).abs() < 1e-6, "the step is reversible");

        // Held-down keys clamp at both ends rather than running away.
        for _ in 0..200 {
            apply(Action::WaterfallIntensityUp, &mut v, &mut r);
        }
        assert_eq!(v.wf_gamma, crate::surface::GAMMA_MIN);
        for _ in 0..400 {
            apply(Action::WaterfallIntensityDown, &mut v, &mut r);
        }
        assert_eq!(v.wf_gamma, crate::surface::GAMMA_MAX);

        // The bare persistence keys never touch it, and the shifted keys
        // never raise a persistence-gamma request — two controls, two
        // surfaces, no crosstalk.
        let mut v = ViewState::default();
        let mut r = ChromeRequests::default();
        apply(Action::PersistenceUp, &mut v, &mut r);
        assert_eq!(r.gamma_steps, -1);
        assert_eq!(v.wf_gamma, crate::surface::DEFAULT_WF_GAMMA);
        apply(Action::WaterfallIntensityUp, &mut v, &mut r);
        assert_eq!(r.gamma_steps, -1, "the shifted key moved §7.3's gamma");
    }

    #[test]
    fn toggles_and_zoom_reset() {
        let mut v = ViewState::default();
        let mut r = ChromeRequests::default();
        v.params.zoom = crate::zoom::ZoomSpan::new(0.3, 0.4);
        apply(Action::ResetZoom, &mut v, &mut r);
        assert!(v.params.zoom.is_full());
        apply(Action::TogglePause, &mut v, &mut r);
        assert!(v.paused);
        apply(Action::TogglePause, &mut v, &mut r);
        assert!(!v.paused);
        apply(Action::ToggleHelp, &mut v, &mut r);
        assert!(v.help_open);
        // Split steps stay clamped.
        for _ in 0..40 {
            apply(Action::SplitMoreHistogram, &mut v, &mut r);
        }
        assert_eq!(v.split, 1.0);
        for _ in 0..40 {
            apply(Action::SplitMoreWaterfall, &mut v, &mut r);
        }
        assert_eq!(v.split, 0.0);
    }

    /// **AC-18 (annotation design §2.7/§2.18).** `I`, `N`, `G`, `H` each fire
    /// their own action and no other, asserted two-sidedly like the D-066
    /// test above: not just "bound", but bound to *its own* effect and never
    /// another's.
    #[test]
    fn inspector_annotation_and_declutter_keys_fire_their_own_action_only() {
        let none = egui::Modifiers::default();
        for key in [Key::I, Key::N, Key::G, Key::H] {
            assert!(binding_for(key, none).is_some(), "{key:?} must be bound");
        }
        assert_eq!(
            binding_for(Key::I, none).unwrap().action,
            Action::ToggleInspector
        );
        assert_eq!(
            binding_for(Key::N, none).unwrap().action,
            Action::ToggleAnnotationLevel
        );
        assert_eq!(
            binding_for(Key::G, none).unwrap().action,
            Action::DeclutterCapDown
        );
        assert_eq!(
            binding_for(Key::H, none).unwrap().action,
            Action::DeclutterCapUp
        );

        let mut v = ViewState::default();
        let mut r = ChromeRequests::default();
        apply(Action::ToggleInspector, &mut v, &mut r);
        assert!(r.toggle_inspector);
        assert_eq!(v.declutter_cap, crate::layout::DEFAULT_DECLUTTER_CAP);
        assert_eq!(v.annotation_level, crate::overlay::AnnotationLevel::Full);

        apply(Action::ToggleAnnotationLevel, &mut v, &mut r);
        assert_eq!(
            v.annotation_level,
            crate::overlay::AnnotationLevel::ShadeOnly
        );
        apply(Action::ToggleAnnotationLevel, &mut v, &mut r);
        assert_eq!(v.annotation_level, crate::overlay::AnnotationLevel::Full);
        // Toggling the level never touched the inspector flag or the cap.
        assert_eq!(v.declutter_cap, crate::layout::DEFAULT_DECLUTTER_CAP);

        apply(Action::DeclutterCapDown, &mut v, &mut r);
        assert_eq!(v.declutter_cap, crate::layout::DEFAULT_DECLUTTER_CAP - 1);
        apply(Action::DeclutterCapUp, &mut v, &mut r);
        apply(Action::DeclutterCapUp, &mut v, &mut r);
        assert_eq!(v.declutter_cap, crate::layout::DEFAULT_DECLUTTER_CAP + 1);
        // Bounds hold at both ends.
        for _ in 0..40 {
            apply(Action::DeclutterCapDown, &mut v, &mut r);
        }
        assert_eq!(v.declutter_cap, DECLUTTER_CAP_MIN);
        for _ in 0..80 {
            apply(Action::DeclutterCapUp, &mut v, &mut r);
        }
        assert_eq!(v.declutter_cap, DECLUTTER_CAP_MAX);
    }
}
